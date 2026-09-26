#!/usr/bin/env python3
# scripts/gemv_m1_bench.py
#
# WHEN IT MAY RUN (read this before running):
#   * Solo GPU only: no ninfer-serve running, no other compute procs, no
#     compile (nvcc/cmake) running. Enforced via scripts/s_gate_preflight.sh:
#     `pre` must print GO (idle util <=5%, 0 compute procs) before any timing,
#     a `stamp` is taken around the run, and `post --ref` must print SEALED.
#     An S-line without a matching START+END stamp pair is void by default.
#   * NEVER run during timing arms A/B/C (or any live measurement window):
#     any extra GPU launch corrupts the arm reps. Running this bench during
#     arm A/B/C is FORBIDDEN. Wait for a solo-GPU idle slot.
#
# WHAT IT DOES:
#   Times each DISTINCT M=1 EXL3 linear shape in the Qwen3.5-27B EXL3 model
#   (decode regime, batch m=1) with N reps and reports per-shape params,
#   weight bytes streamed, median ms, achieved GB/s vs the 1.77 TB/s HBM
#   floor, and efficiency %.
#
# HOW IT TIMES (repo harness convention: python driver + native probe):
#   Existing benches are python drivers around native binaries
#   (tools/bench/run_ninfer_bench_matrix.py -> build/bench/ninfer_bench).
#   Python has no direct EXL3 dispatch binding, so this driver nvcc-builds a
#   small embedded CUDA probe (no CMake changes, no engine rebuild) that
#   streams each shape's EXL3 weight volume with cudaEvent timing, then the
#   driver aggregates reps and prints the roofline table. The probe measures
#   the weight-streaming roofline (achievable read BW for that byte volume),
#   NOT the production exl3_dispatch kernel itself: compare probe GB/s to the
#   1.77 TB/s floor to see headroom, and compare future real-kernel timings
#   (same shapes, same harness) against the probe number.
#
# SHAPES (from src/runtime/engine/exl3_program.cpp; add_single(name, n, k)):
#   full layer : qkv fused n=14336 k=5120 | o n=5120 k=6144
#                gate_up fused n=34816 k=5120 | down n=5120 k=17408
#   GDN layer  : qkv n=10240 k=5120 | z n=6144 k=5120 | o n=5120 k=6144 (=full o)
#   MTP layer  : same as full layer + draft_head n=5120 k=10240
#   lm_head    : n=248320 k=5120 (EXL3)
# Route notes (src/ops/linear/exl3/exl3_dispatch.h, exl3_op.h): groups>1 fused
#   multis (qkv, gate_up) ride the v3 fused row at any m; groups==1 m==1 K3/K4
#   rides legacy gemv; K5/K6 (o_proj, lm_head) ride the v3 fused row (gemv caps
#   at K4). Per-tensor K comes from the checkpoint; the table marks the route
#   per the header comments.
#
# Usage:
#   python3 scripts/gemv_m1_bench.py --shapes-only   # no GPU, safe anytime
#   sh scripts/s_gate_preflight.sh pre --out /tmp/gemv_ref
#   python3 scripts/gemv_m1_bench.py --reps 50 --warmup 10
#   sh scripts/s_gate_preflight.sh post --ref /tmp/gemv_ref
#
import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = REPO_ROOT / "scripts" / "s_gate_preflight.sh"

FLOOR_TB_S = 1.77          # HBM floor all shapes are judged against
EXL3_BPW = 3.5             # effective EXL3 bits/param at ~3.5bpw
WARMUP_DEFAULT = 10

# (name, n, k, route, note). M=1 for all rows (decode GEMV regime).
SHAPES = [
    ("full_qkv",       14336,   5120, "v3-fused-multi", "groups=3 fused q/k/v(+gate)"),
    ("attn_o",          5120,   6144, "v3-single",      "o_proj K5/K6 per dispatch.h; == gdn_o, deduped"),
    ("gate_up",        34816,   5120, "v3-fused-multi", "groups=2 fused gate/up"),
    ("down",            5120,  17408, "legacy-gemv*",   "groups=1; *K3/K4 only, else v3-single"),
    ("gdn_qkv",        10240,   5120, "v3-fused-multi", "GDN q/k/v fused; z rides separately"),
    ("gdn_z",           6144,   5120, "legacy-gemv*",   "GDN z branch; *K3/K4 only, else v3-single"),
    ("mtp_draft_head",  5120,  10240, "legacy-gemv*",   "MTP-only; MTP layer linears == full shapes, deduped"),
    ("lm_head",       248320,   5120, "v3-single",      "EXL3 vocab head, K5/K6 per dispatch.h"),
]

PROBE_CU = r"""
// gemv_m1_probe: stream a shape's EXL3 weight bytes M=1 style, cudaEvent-timed.
// One block per SM sweep reads weight[] (n*k*BPW/8 bytes) + x[k] and writes
// out[n]; the timed region is pure global-memory traffic so ms/volume gives
// the achievable streaming BW for that shape's byte count (roofline probe,
// not the production exl3_dispatch kernel).
#include <cuda_runtime.h>
#include <cstdio>
__global__ void stream_kernel(const unsigned char* __restrict__ w,
                              const __half* __restrict__ x,
                              __half* __restrict__ out,
                              size_t wbytes, int k, int n) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    float acc = 0.0f;
    for (size_t i = tid; i < wbytes; i += stride) acc += (float)w[i];
    for (int i = (int)tid; i < k; i += (int)stride) acc += (float)x[i];
    if (tid < (size_t)n) out[tid] = (__half)(acc * 1e-6f);
    if (tid == 0) out[0] = (__half)(acc * 1e-6f);
}
int main(int argc, char** argv) {
    if (argc != 5) { fprintf(stderr, "usage: probe n k wbytes reps\n"); return 2; }
    int n = atoi(argv[1]), k = atoi(argv[2]);
    size_t wbytes = (size_t)atoll(argv[3]);
    int reps = atoi(argv[4]);
    unsigned char *w; __half *x, *out;
    cudaMalloc(&w, wbytes); cudaMalloc(&x, (size_t)k * 2); cudaMalloc(&out, (size_t)n * 2);
    cudaMemset(w, 0x5a, wbytes); cudaMemset(x, 0, (size_t)k * 2);
    int sms = 0; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    dim3 grid((unsigned)(sms * 4)), block(256);
    for (int i = 0; i < 10; i++) { stream_kernel<<<grid, block>>>(w, x, out, wbytes, k, n); }
    cudaDeviceSynchronize();
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    for (int r = 0; r < reps; r++) {
        cudaEventRecord(s);
        stream_kernel<<<grid, block>>>(w, x, out, wbytes, k, n);
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float ms = 0; cudaEventElapsedTime(&ms, s, e);
        printf("%.6f\n", ms);
    }
    cudaFree(w); cudaFree(x); cudaFree(out);
    return 0;
}
"""


def exl3_bytes(n, k):
    # Packed weight bits + ~2% scale/zero-point overhead (svh/suh side-cars).
    return math.ceil(n * k * EXL3_BPW / 8 * 1.02)


def preflight(args):
    if args.no_preflight:
        print("PREFLIGHT skipped (--no-preflight); results are UNSTAMPED/void.", flush=True)
        return None
    if not PREFLIGHT.is_file():
        sys.exit("REFUSE: %s missing" % PREFLIGHT)
    ref = args.ref or os.path.join(tempfile.gettempdir(), "gemv_m1_ref")
    if not args.skip_pre:
        r = subprocess.run(["sh", str(PREFLIGHT), "pre", "--out", ref],
                           capture_output=True, text=True)
        print(r.stdout.strip(), flush=True)
        if r.returncode != 0 or "GO" not in r.stdout:
            sys.exit("REFUSE: preflight did not print GO; GPU not solo. Aborting.")
    else:
        print("PREFLIGHT pre skipped (--skip-pre); need a START stamp for a sealed run.",
              flush=True)
    return ref


def build_probe(keep=False):
    d = Path(tempfile.mkdtemp(prefix="gemv_m1_"))
    src = d / "gemv_m1_probe.cu"
    exe = d / "gemv_m1_probe"
    src.write_text(PROBE_CU)
    r = subprocess.run(["nvcc", "-O2", "-arch=native", str(src), "-o", str(exe)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("probe build failed:\n%s\n%s" % (r.stdout, r.stderr))
    return exe


def time_shape(exe, n, k, reps):
    wb = exl3_bytes(n, k)
    r = subprocess.run([str(exe), str(n), str(k), str(wb), str(reps)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("probe failed for n=%d k=%d:\n%s" % (n, k, r.stderr))
    ms = [float(x) for x in r.stdout.split()]
    if len(ms) != reps:
        sys.exit("probe returned %d/%d samples (n=%d k=%d)" % (len(ms), reps, n, k))
    return wb, ms


def main():
    ap = argparse.ArgumentParser(description="M=1 EXL3 GEMV roofline bench (solo-GPU only).")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=WARMUP_DEFAULT,
                    help="extra untimed reps folded into probe warmup (informational)")
    ap.add_argument("--floor-tb-s", type=float, default=FLOOR_TB_S)
    ap.add_argument("--ref", default=None, help="preflight ref file (default: $TMPDIR/gemv_m1_ref)")
    ap.add_argument("--no-preflight", action="store_true", help="unstamped run (void)")
    ap.add_argument("--skip-pre", action="store_true", help="skip pre; post-seal separately")
    ap.add_argument("--no-post", action="store_true", help="skip end stamp (caller seals)")
    ap.add_argument("--shapes-only", action="store_true", help="print shape table, no GPU")
    ap.add_argument("--json", default=None, help="write results JSON to path")
    args = ap.parse_args()

    if args.shapes_only:
        print("%-14s %8s %6s %10s %10s %s" % ("shape", "n", "k", "params", "bytes", "route"))
        for name, n, k, route, note in SHAPES:
            print("%-14s %8d %6d %10d %10d %s  # %s"
                  % (name, n, k, n * k, exl3_bytes(n, k), route, note), flush=True)
        return

    ref = preflight(args)
    exe = build_probe()
    floor_gb_s = args.floor_tb_s * 1000.0
    rows = []
    print("%-14s %10s %8s %10s %10s %8s %7s %s"
          % ("shape", "params", "bytes", "ms", "GB/s", "floor", "eff%", "route"), flush=True)
    for name, n, k, route, _ in SHAPES:
        wb, ms = time_shape(exe, n, k, args.reps)
        med = statistics.median(ms)
        gbps = (wb / 1e9) / (med / 1e3)
        eff = 100.0 * gbps / floor_gb_s
        rows.append({"shape": name, "n": n, "k": k, "m": 1, "params": n * k,
                     "bytes": wb, "reps": args.reps, "ms_median": med,
                     "ms_all": ms, "gb_s": gbps,
                     "floor_gb_s": floor_gb_s, "eff_pct": eff, "route": route})
        print("%-14s %10d %8d %10.4f %10.1f %8.0f %6.1f%% %s"
              % (name, n * k, wb, med, gbps, floor_gb_s, eff, route), flush=True)
        subprocess.run(["sh", str(PREFLIGHT), "stamp"], capture_output=True)
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"floor_tb_s": args.floor_tb_s, "reps": args.reps, "rows": rows}, indent=1))
        print("wrote %s" % args.json, flush=True)
    if ref and not args.no_post:
        r = subprocess.run(["sh", str(PREFLIGHT), "post", "--ref", ref],
                           capture_output=True, text=True)
        print(r.stdout.strip(), flush=True)
        if r.returncode != 0:
            sys.exit("run VOID (post seal failed).")


if __name__ == "__main__":
    main()
