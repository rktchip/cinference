// exl3_launcher.h
// EXL3 small-m (m<=8) decode launcher for cinference — raw-pointer, no libtorch,
// CUDA-graph safe (all plain kernel launches; the reference cooperative launch is
// not used — see exl3_gemv_plain.cuh for the provenance of the carve).
//
// Decision heuristic exl3_gemv_cfg is VERBATIM from exllamav3 1.5.1
// quant/exl3_gemv.cu (MIT, turboderp-org/exllamav3). K/cb semantics mirror the
// reference: K in {3,4} only (K=4 cb by mcg/mul1; K=3 cb by mcg/mul1); the gemv
// static_assert in 1.5.1 caps at bits 3/4 — K>=5 routed to reconstruct.
#pragma once

#include "exl3_ptx_shim.cuh"
#include "exl3_codebook.cuh"
#include "exl3_dq.cuh"
#include "exl3_hadamard.cuh"
#include "exl3_gemv_plain.cuh"

#include <cuda_runtime.h>
#include <cstddef>

namespace ninfer {
namespace exl3 {

// Device compute capability class (sm_120 on RTX 5090). The reference uses
// its own DevCtx singleton for cc + num_sms; cinference bakes sm_120 and the
// SM count in at construct time (see exl3_sm_120 helpers below).
#define EXL3_CC_BLACKWELL 120

// Reference heuristic, verbatim (mode==2 for our single-serving-GPU shape;
// see exl3_gemv.cu -- "mode==2 returns n<=8192 ? 0 : 1" under the force path).
static int exl3_gemv_cfg_cc120(int size_m, int size_k, int size_n, int K, int cb, int narrow_coresident)
{
    // forced-mode branch (mode 2): nothing to pre-check; shape heuristic
    // below runs directly, mirroring the reference ordering.
    (void) size_m; (void) cb;
    if (K == 3) return size_n <= 8192 ? 0 : 1;
    if (K == 4) {
        if (size_n / 32 <= narrow_coresident) return 0;
        if (size_k <= 2048 && size_n <= 8192) return 0;
        if (size_n >= 8192 && size_k <= 4096) return 1;
        return 0;
    }
    return -1;
}

struct Exl3GemvPlan {
    int cfg = -1;            // 0 narrow (512t, 2 tiles/warp), 1 wide (256t, 4)
    int cb = 0;              // 0/1/2 as 1.5.1
    int bits = 4;            // K passed to the template (checkpoint K is integer)
    bool c_fp32 = false;     // C dtype: fp32 final accumulation vs fp16
    int mmode = 0;           // 0: m==1 fast path, 1: 2<=m<=8
    bool smem = false;       // smem staging (default: shuffle path)
    int block_dim = 512;
    int cols = 32;
    int grid = 0;
    int m = 0, k = 0, n = 0;
    int num_sms = 148;       // RTX 5090 = 170 SM, but keep 148-style heuristic; caller overrides
    bool ok = false;
};

// Shape eligibility (mirror of the reference host pre-env checks).
__host__ inline bool exl3_gemv_shape_ok(int size_m, int size_k, int size_n)
{
    if (size_m < 1 || size_m > EXL3_GEMV_MAX_M) return false;
    if (size_k % 128 != 0 || size_n % 128 != 0) return false;
    return true;
}

// Codebook per tensor: the checkpoint only needs cb (0/1/2); mcg==true -> 1,
// mul1==true -> 2, else 0. (Reference: same; cb2 requires mul1.)
__host__ inline int exl3_gemv_cb(bool mcg, bool mul1, int K)
{
    (void) K;
    if (mul1) return 2;
    if (mcg) return 1;
    return 0;
}

// host declarations (defined in exl3_launcher.cu — the dispatch TU links
// against it; see exl3_dispatch.h):
extern bool exl3_gemv_run(const Exl3GemvPlan& p, const half* A, const uint16_t* B,
                          void* C, cudaStream_t stream);
extern bool exl3_linear_decode(const half* A, const uint16_t* B, half* A_h, half* C16,
                               uint16_t* outbf, const Exl3GemvPlan& p, const half* suh,
                               const half* svh, cudaStream_t stream);

// Try-fill the plan. Returns false (caller falls to reconstruct) when the
// call is not hard-eligible. Caller supplies device cc + num_sms (cinference
// keeps them as constructor flags; no DevCtx singleton).
__host__ inline bool exl3_gemv_try_plan(Exl3GemvPlan& p,
                                        int size_m, int size_k, int size_n,
                                        int K, bool mcg, bool mul1, bool c_fp32,
                                        int cc, int num_sms)
{
    p.m = size_m; p.k = size_k; p.n = size_n;
    p.ok = false;
    p.cfg = -1;
    p.bits = K;
    p.c_fp32 = c_fp32;
    p.cb = exl3_gemv_cb(mcg, mul1, K);
    p.mmode = size_m == 1 ? 0 : 1;
    p.smem = false;
    // Eligibility: 1.5.1 gemv is hard-capped at bits {2,3,4}; K<2 and K>4
    // route to reconstruct. cb0 is only valid for K==4 (the existing
    // dispatch map has K=2/3 instances only for cb1/cb2). This checkpoint:
    // every EXL3 group has mul1 -> cb2, so the cb0 guard never fires for
    // it; kept for other EXL3 families.
    if (K < 2 || K > 4) return false;
    if (K != 4 && p.cb == 0) return false;
    if (!exl3_gemv_shape_ok(size_m, size_k, size_n)) return false;

    p.num_sms = num_sms;
    int narrow_coresident = num_sms * 2;   // 2 resident blocks of 512t on sm_120
    int cfg = exl3_gemv_cfg_cc120(size_m, size_k, size_n, K, p.cb, narrow_coresident);
    if (cfg < 0) return false;
    p.cfg = cfg;
    p.block_dim = cfg == 0 ? 512 : 256;
    p.cols = cfg == 0 ? 32 : 64;
    // occupancy is measured at launch time by the dispatcher (needs the
    // instantiated function pointer); grid here is an upper-bound fallback:
    int occ_blocks = 2;   // conservative; actual occupancy set by the plan dispatcher
    int max_blocks = occ_blocks * num_sms;
    int grid = size_n / p.cols;
    if (grid > max_blocks) grid = max_blocks;
    if (grid < 1) return false;
    p.grid = grid;
    p.ok = true;
    return true;
}

} // namespace exl3
} // namespace ninfer