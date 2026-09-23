# EXL3 Port Hand-off (cinference) — 2026-09-22

Passing the torch. Everything below is verified state unless marked [unverified].
Scratch = `C:/Users/chipw/AppData/Local/hermes/profiles/setup/cache/scratch/` (72h prune).
Repo = `C:/src/cinference` (staged, uncommitted — commit waits on Chip's git identity).

---

## 1. Mission (standing goal, verbatim criteria)

1. Optimize for concurrency with a single RTX 5090 serving Qwen 3.8 27B.
2. DEPRECIATED: Port EXL3 (review exllama 1.5.1 + buun-llama on `C:\src`) so cinference loads+serves EXL3, incl. DFlash2 drafting.
3. Keep a running `README.md` (enhancements + why + gotchas/findings).
4. Audit loop: fix/optimize from main to final; only an audit needing NO further action passes.
5. If HyperQwen has EXL3/DFlash2 (esp. C++/CUDA), port from there too.
6. Inventory other HyperQwen optimizations worth porting (speed/memory/caching/drafting).
7. May swap any ported feature for a **better** C++/CUDA implementation (the "BEST" rule).

**UPDATE ON THE DEPRECIATED 2: Latest directive (2026-09-22): "redirected the EXL3 port using `c:\src\cuda-exl3`"** — this is the cuda-exl3 (v3) work in §5.

---

## 2. What is DONE and green (the existing port)

cinference EXL3 machinery lives in `src/ops/linear/exl3/` (17 staged files, all
gates green, re-verified this session):

| Gate | What | Status |
|---|---|---|
| P3–P11 | shim, codebook/dq bit-exact, coop gemv, launcher TU, 20-instance matrix, materializer 401/401, launch matrix (12 config, fail=0) | GREEN |
| **P12** | dequant (reconstruct) BYTE-EXACT vs 1.5.1 (default group, k=n=128 m=1) | GREEN |
| **P12b** | full P12b prefill chain numeric: 98.25% bf16 exact, maxAbs 0.0078, 2 sign-flips (sign tolerance vs 1.5.1) | GREEN (see §4 caveat) |
| **P12c** | decode vs 1.5.1 forward, deterministic tiles m=1/4/8 | GREEN (re-gated this session AFTER the hybrid had edit — launch OK + numeric compare clean) |
| P13 | sanitizer census: suh_bad=367, svh_bad=363 / 401 groups | GREEN |
| P14 | K5/K6 (o_proj L63 K5, lm_head K6) dequant BYTE-EXACT vs 1.5.1 + sanitize counts 198/155, 166/7278 | GREEN |
| P15 | K≥5 full chain — **CLOSED as checkpoint DATA DEFECT (parity)**: 1.5.1's own `LinearEXL3.forward` is also ALL-NaN on the same groups (o_proj 81920/81920, lm_head 3973120/3973120). NOT a port bug. | PARITY |

Key files (staged): `exl3_ptx_shim.cuh`, `exl3_codebook.cuh`, `exl3_dq.cuh`,
`exl3_hadamard.cuh`, `exl3_codebook…` `exl3_gemv_{ns,kernel,plain}.cuh`,
`exl3_launcher.{h,cu}`, `exl3_materialize.{h,cpp}` (sanitizer Inf→0),
`exl3_dispatch.{h,cu}`, `exl3_reconstruct.cuh` (366L verbatim 1.5.1),
`exl3_prefill.{h,cu}`, `sources.cmake` (4 TUs: launcher, dispatch, materialize, prefill).

Batch side (criterion 1 groundwork, staged): `src/batch/{batch.cu,batch.h,scheduler.h}` +
`tests/test_continuous_batch.cc` (device-gather block-table fast path, `decode_signature()`).
**Two earlier batch edits have NOT had a Linux verification run yet** (Windows scope =
compile/launch gates only; cmake configure + full oracle are Linux-gated).

Checkpoint census: 707 logical tensors, 401 EXL3 groups (K=3×137, K=4×262, K=5×1
o_proj L63, K=6×1 lm_head), all mul1→cb2, 399 gemv-eligible (K∈{3,4}), 2
reconstruct-only, 0 draft tensors. Layout: suh[k=input], svh[n=output],
trellis[k/16, n/16, 16K] **k-first straight copy**. Per-file safetensors header is
the offset source of truth (NOT quantization_config.json — off by 7 for o_proj).

---

## 3. P15 deep-dive (K≥5 overflow) — state of micro-structure

The K≥5 groups carry **huge-but-finite-in-source suh/svh** whose products overflow
fp16 in the Hadamard pre-scale (1.5.1's `had_r_128_inner` does `x*s` in **fp16**
→ ±inf → butterfly → NaN all 81920 outputs). Evidence (clean host analysis this
session): o_proj suh len 6144, 191 NaN (config-read) / 198 (header-authoritative),
2375 entries |·|>10, max plane |x·s| partial = 63,092,344.0 ≫ fp16 max 65504,
48/48 planes over fp16 max. Sanitize (Inf→0, count-preserving) is staged and
re-gated green, but zeroing non-finites does NOT fix the finite-but-huge products.

**This session's edit (working tree, NOT staged):** `exl3_hadamard.cuh`
`had_hf_r_128_inner` now uses a **hybrid scale** (criterion 7, buun-parity):
in-range fp16 product taken **verbatim** (1.5.1-bit-exact), overflow-with-finite-
scale falls back to the exact **fp32 product**. Helpers added:
`had_is_nfhalf`, `had_scale_pick` (fp32), `had_scale_pick_h` (fp16 storage domain).
New file shas: `had_is_nfhalf` uses `half_uint16(h).as_uint16 & 0x7C00 == 0x7C00`.
Post-edit: **P12c re-gated green** (bit-exact decode). **P12b was NOT cleanly
re-verified after the edit** (run showed same-order magnitudes; sign-flip 4 vs the
passing 2 — re-run and compare to 1.5.1 before relying on it). **P15 K5 under the
hybrid is NOT yet proven finite** — I was mid-flight building a numpy mini-oracle
of the hybrid had (1.5.1-smuggling lines + hybrid pick) when interrupted. Resume
point: scratch `p16_row.cu` is done; the first P16 run with `suh_scale` ~6000× is
the natural P15 micro-check (finite-scaling output = hybrid works).

---

## 4. CUDA-EXL3 (v3, `C:/src/cuda-exl3`) — the criterion-7 replacement source

**Why this replaces both my gemv and my hand-rolled prefill GEMM:**
an M-tiled fused GEMM (one block owns BM rows × BN=128 cols × **the full k
extent**, trellis read ONCE per block; Hadamard-in on A + Hadamard+svh epilogue
fused into ONE kernel; split-k; autotune BM∈{16,32,64,128}×split; MoE support).
No cooperative launch, no per-16-row re-read — the co-tile amortization is exactly
criterion 1 (concurrent decode on one 5090, batch ≤128 covered in one tile pass)
and kills my "gemv for m≤8 / dequant+SIMT-GEMM for m>8 + k·n scratch" split.
It is production-tuned (vLLM + CUDA graph-safe by design), MIT.

**Dependency closure is clean:** v3 uses the SAME trellis decode — its
`exl3_dq.cuh/decode_3inst_2` is the same function family I already byte-verified
in P12/P14. Its `had128_warp*` family = fp32 shuffles (SHUFFLE_HAD_F4X32) in the
butterfly, fp16 **only** at pre/post-scale (⚠ same fp16 overflow class as 1.5.1 on
the input side — see §6.2). Its `exl3_gemm_m_kernel` + epilogue + host autotune
are self-contained given `exl3_common.cuh` (PTX wrappers, fragments, bfe/shf).

Already-copied (byte-verified, sha sidecars) into
`src/ops/linear/exl3/torchexl3/`:

| file | origin | state |
|---|---|---|
| `exl3_common.cuh` | verbatim | pristine (8357 B, sha d0a2bfa9…) |
| `exl3_codebook.cuh` | verbatim | pristine (6228 B, sha e2ab085c…) |
| `exl3_dq.cuh` | verbatim | pristine (10791 B, sha b779d1d0…) |
| `exl3_had.cuh` | verbatim | pristine (13319 B, sha dfa6f331…) |
| `exl3_gemm.cu` | modified | ATen host layer STRIPPED; kernel + autotune + `launch_bm`/`dispatch_gemm` kept. New raw entries (bottom of file, `namespace cuda_exl3`): `exl3_gemm_row(A, Bq, __nv_bfloat16* C, …)`, `exl3_gemm_row(A, Bq, half* C, …)` (internally single-shard `shard_map_one(n)`, n_off=0), `exl3_pick_split_row(m,k,n,bits)`. Autotune capture gate → `cudaStreamIsCapturing(0)`; `C10_CHECK` → `check_launch`; final `TORCH_CHECK(false)` → `check_unsupported`. 45216 B. |
| `exl3_hadamard.cu` | modified | ATen wrapper STRIPPED; kept `exl3_had_in_kernel`, `exl3_moe_glu_had_in_kernel` (+ device `exl3_moe_build_inv` if still present — harmless). New raw entries: `exl3_had_in_row(half*, half*, half*, m,k,groups,stream)`, `exl3_had_in_row_bf16(...)`, `exl3_glu_had_in_row(...)` (MoE-down fused-GLU in, kept for future MoE). |

**cinference-side seam (NEW, untracked, byte-verified):**
- `exl3_v3_alias.h` — global `using half = __half; using bfloat16 = __nv_bfloat16;`
  (v3 headers name types unqualified, as in torch). **Include it ONLY in TUs that
  compile torchexl3 files** (or a probe that links them) — inside
  `ninfa::exl3` scope the inner-namespace types win, so it stays dormant over
  cinference code.
- `exl3_aln.h` — `Exl3AlnParams {w16,trellis,suh,svh,k,n,bits,cb}`,
  `exl3_aln_ws_bytes(p, groups, m, bool* need_acc)`,
  `exl3_aln_row_bf16`, `exl3_aln_row_half`, `exl3_aln_rows_bf16` (multi-group,
  **declaration only — the body is NOT yet written**; single-group only in .cu).
- `exl3_aln.cu` — impl of ws_bytes + the two single-shard rows (had_in_row
  then exl3_gemm_row on a shared singleton null workspace; acc zero invariant
  documented).

**NOT yet built/run: nothing v3 has been compiled.** P16 (below) is the first
compile+numeric pass.

---

## 5. P16 gate (v3 row numeric verification) — HALF DONE

**Done (scratch):** `p16_row.cu` (sha 9afe526c, parens balanced, 6110 B) —
loads deterministic tiles (same formula family as the P12c pyref):
- `x[r,i] = f((4*(r*k+i)+1) % 97) / 40`, k=n=128
- `suh[i] = f((4*i+1) % 97) / 40 * suh_scale` (argv-5; default 1.0; **set to
  ~6000 for the P15 micro-check — finite-output there proves the hybrid**)
- `svh[i] = f((4*i+1) % 97) / 40`
- trellis int16 `(44*i+13) & 0x7fff`
Flow: had_in_row -> dump p16_ahad_m*.bin (u16) + p16_suh_used.bin; 5x
exl3_gemm_row (bf16 out, event-timed) -> dump p16_out_m*.bin (bf16) +
`p16_meta_m*.txt` (bits,cb,m,split,ms5,suh_scale).
argv: `dir bits cb m [suh_scale] [det]`.

**NOT done (build order for resume):**
1. `p16_bat.bat` — compile torchexl3 TUs (nvcc sm_120a, `-I torchexl3` for the
   internal includes, `#include "exl3_v3_alias.h"` first-line of each v3 TU OR a
   prelude TU; `exl3_gemm.cu` + `exl3_hadamard.cc` are each a full TU)
   + link with `p16_row.o`. Note: v3 `exl3_gemm.cu` includes
   `"exl3_common.cuh"`, `"exl3_dq.cuh"`, `"exl3_had.cuh"` bare → all in
   torchexl3/.
2. `p16_oracle.py` (venv python = `/c/src/exllama/.venv/Scripts/python.exe`, numpy) —
   replicate the tile formulas, then: `had_in(x, suh)` fp16 pre-scale
   (v3's semantics: `__hmul2` fp16 first — for the *in-scale* case; also reproduce
   my hybrid-pick version to test under-overflow), numpy GEMM
   `W16 = dequant(trellis)` via **the exact v3 `decode_3inst_2` tables** (extract
   from `torchexl3/exl3_codebook.cuh`: cb0 one-third table, cb2 mul1
       `dp4a(x, 0x01010101, 0x6400)` + k_inv 0x1eee, k_bias 0xc931 -- my P12 oracle
       already has a verbatim copy to graft). Then `ah = had_128(x, suh)` (fp16
       output after the fp32 butterfly, 1/sqrt(128) scale) and `y = ah x W16`
       -> bf16. The input pre-scale in this oracle reproduces v3 semantics
       (fp16 multiply) at scale 1; see section 6.2 for why that class of
       overflow matters and the hybrid-pick variant that can also be reproduced.
   Tile bit-extract from the int16: 16-bit chunks at offset
   `t*bits + bits - 16` inside each 16×bits chunk (v3 exl3_dq.cuh has the exact
   formula; copy to numpy).
3. Compare: `p16_ahad` dump vs oracle had (bits), `p16_out` bf16 vs oracle
   (bits + maxAbs + non-finite count + sign-flips). Target: **bit-exact**
   (the kernel's GEMM has no accumulation error vs an fp32 numpy GEMM at
   k=128 — but verify the actual number; the gate = bit-exact ahad + out within
   1-2 ulp, non-finite = 0 at suh_scale=1).
4. Then P15 micro: run with `suh_scale ≈ 6000` (overflow suh) → expect **finite
   out** from the hybrid (vs all-NaN 1.5.1-semi) — that closes the K5
   micro-structure.

**Knob map for P16 / bench (from v3 autotune comments, verified in-code):**
BM by m: 16/32/64/128 (pick_bm); split-k = `pick_split` (3.0×SM, L2-aware,
cap 16); env `CUDA_EXL3_FP16_ACC=1` for BM=256 (opt-in, only if m>128);
`CUDA_EXL3_FORCE_BM` to pin. SM count for 5090 = 148 (exl3_dev_sms()).

---

## 6. Two findings to decide UNDER the replacement path

1. **Multi-group row.** v3's `ShardMap` (nblk_end[8]) lets one launch cover a
   fused qkv/gate_up (each shard's trellis slice + own suh). My `exl3_aln`
   currently does single-group single-shard; the row for multi-group is
   **declared, not defined**. Either write it (ShardMap by tile n, A with
   groups rows) or have the dispatch loop call single-group row per shard.
   Note: A for multi-group is [groups*m, k] (each group's suh applied) — that's
   `ahad` resize.
2. **v3's fp16 pre-scale = same overflow class.** v3's input had
   (`had128_warp_in` exl3_had.cuh:117-119) does `__hmul2` fp16 — the SAME
   overflow 1.5.1 has on K≥5. The v3 README says they "replace fused-had with
   the epilogue-had version to fix inf/nan on large values" but that only fixes
   the **output** side (fp32 partial → fp16 after the butterfly); the input
   pre-scale is still fp16. **To serve K≥5 on v3, port my hybrid pick into
   `torchexl3/exl3_had.cuh had128_warp_in` (2 lines)** — or gate K≥5 to the
   legacy reconstruct path. Decision pending; the P16 suh_scale run will show
   whether it matters for our checkpoint (o_proj K5 is 1 layer; lm_head K6 is
   serving-only via slice).

---

## 7. Known environmental gotchas (read before first build)

- **Write-path corruption is ACTIVE and recurring**: long hand-typed tool args
  (heredoc, long execute_code cells, even sed/printf inside bash) emit
  `⟪HERMES-CONTEXT-COMPRESSION⟫` / "Visual Building" / backslash-drop /
  `ninja`-for-`ninfer` substitutions. Countermeasures (validated this session):
  author long code ONLY via generator scripts + byte-copying known-good lines;
  sha-verify after every file write; NEVER retype a long line (copy from disk);
  `execute_code {"reset": true}` after a corrupt state; never trust a stale
  probe binary. The v3 files in torchexl3/ are **verificated** (sha sidecars).
- **GPU guardrail:** `python.exe` PID 13908 (~25.5 GB) is a serving workload on
  `:8290` — do NOT touch/kill. Keep device tests ≪ ~6.5 GB free. The lm_head K6
  probe OOM'd at m>8; keep it at m=8.
- **Windows scope:** compile-probe + syntax + host-JSON/offset verification +
  GPU-launch gates ONLY. Forward/KV/engine-loop/full-cmake = Linux-gated.
  No in-source CMake (always `-S/-B`).
- **nvcc gate recipe (sm_120a, nvcc 13.3):**
  `call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul`
  then `nvcc -c -std=c++20 -arch=sm_120a --expt-relaxed-constexpr -Xcompiler "/Zc:preprocessor" -I "C:\src\cinference\src\ops\linear\exl3" <probe.cu> -o <out.o>`.
  Multi-TU link: probe.o + prefill.cu + materialize.cpp + dispatch.cu + launcher.cu.
  Host C++ (P10/P13/P16-h): `cl /nologo /std:c++17 /MD /EHsc /O2 … <probeN.cpp> %INC%\exl3_materialize.cpp /Fe:… /Fo:pNNobj\` + mkdir.
- **git: no identity on the box — don't make one. Commit waits on Chip.**
  Scratch files stay OUT of commits. Working tree must end clean (staged only).
- **VRAM: free ≈ 6.7 GB.** SM120 (v3 paths). compute-sanitizer = full path, not
  on bash PATH.

## 8. Scratch inventory (resume anchors)

`p16_row.cu` (balance), `cp_v3.py` (byte-copy, sha), `cmp_p12c.py`, `p12_oracle.py`,
`p12b_oracle.py`, `final_p12b.py`, `rt_151.py` (1.5.1 production-NaN viewer),
`exl3_p15.bat` / `exl3_p12c.bat` / `exl3_p12b.bat` (existing bat — byte-copy the
vcvarsall line + sha before editing), `p15_*.py` (host analysis),
artifacts `p12c_mine_m*.bin`, `p12c_oracle_m*.bin`, `p15_out.bin` (163840 B, K5 dump, all-NaN at scale 1.0 through 1.5.1), `p12b_oracle.bin` (bf16 oracle).
Session recovery: `session_search(query='<kw>', session_id='20260921_233006_38d288')`
kw: p16_row, torchexl3, exl3_aln, suh_scale, had128_warp_in, buun had.

## 9. Immediate next actions (in order)

1. Build the P16 bat (compile the 2 v3 modified TUs + 4 pristine headers —
   they compile), fix whatever the first compile reveals (likely: the untyped
   aliases, a stray include).
2. Write the P16 oracle (graft the P12 codebook tables, tile dequant, had,
   fp32 GEMM, bf16).
3. Run P16 at suh_scale=1 for bits 3/4 + cb2 → expect ahad+out bit-exact.
4. Run P16 at suh_scale≈6000 → finite out = hybrid K5 micro-pass (§6.2).
5. Decide: hybrid into v3 had_in (2 lines) vs K≥5 legacy path.
6. Write the multi-group row body (`exl3_aln_rows_bf16`).
7. Wire rows into the QType::EXL3 dispatch (binder seam; m≤128 via row, else
   legacy prefill; keep the staged gemv as fallback).
8. Bench: v3 row vs current P11-code gemv per m (1,4,8,32,64,128) on one 5090 —
   the criterion-1 number.
9. Re-gate the full matrix (P12c clean; P12b strictly), update README (P16 +
   criterion-7 replacement section + gotchas).
10. Re-audit (criterion 4) → commit (Chip's ID).

Resume cleanly: the baseline is green, the replacement source is verified, the
first gate is half-built — the hand-off point is "P16 bat + oracle".