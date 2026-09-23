// exl3_launcher.cu
// EXL3 decode (m<=8) plain-launch path for cinference. Order per call:
//   had_in (A -> A_h, Hadamard128 + suh pre-scale)
//   gemv   (reference 1.5.1 main loop, plain launch; no cooperative sync)
//   had_out (C16 -> C16, Hadamard128 + svh post-scale)
//   cast   (fp16 -> bf16 into the engine out plane)
// Instance set sized to the checkpoint K census: bits{3,4} x cb{0,1,2} x
// mmode{0,1} x cfg{0,1}, c_fp32=false. Plan/cfg/grid mirror exl3_gemv.cu
// (1.5.1, MIT). Had planes: 32 threads per 128-element plane; GRID =
// (rows, planes) exactly per 1.5.1 had_r_128 (hadamard.cu:143-147); the
// vendored inner reads its scale window at ((half4*)scale)[blockIdx.y*32+t]
// so PLANE = dim 2 and the scale pointer is the FULL suh/svh vector.
// (P12b audit 2026-09-22: the earlier (planes, rows) grid made row r read
// window (pl+r) - correct only for m=1; fixed to the 1.5.1 geometry.)
#include "exl3_launcher.h"

namespace ninfer {
namespace exl3 {

static constexpr float EXL3_R_SCALE = 0.088388347648f;  // 1/sqrt(128), verbatim ref

__global__ void exl3_had_in_plane(const half* __restrict__ A, half* __restrict__ A_h,
                                  const half* __restrict__ suh, int size_k)
{
    const int row = blockIdx.x, pl = blockIdx.y;
    had_hf_r_128_inner<true, false>(A + (size_t) row * size_k + pl * 128,
                                    A_h + (size_t) row * size_k + pl * 128,
                                    suh, EXL3_R_SCALE);
}

__global__ void exl3_had_out_plane(half* __restrict__ C, const half* __restrict__ svh, int size_n)
{
    const int row = blockIdx.x, pl = blockIdx.y;
    had_hf_r_128_inner<false, true>(C + (size_t) row * size_n + pl * 128,
                                    C + (size_t) row * size_n + pl * 128,
                                    svh, EXL3_R_SCALE);
}

__global__ void exl3_cast16_to_bf16(const half* __restrict__ c16,
                                    uint16_t* __restrict__ out16, size_t n)
{
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out16[i] = __bfloat16_as_ushort(__nv_bfloat16(__half2float(c16[i])));
}

__host__ inline bool exl3_gemv_run(const Exl3GemvPlan& p, const half* A,
                                   const uint16_t* B, void* C, cudaStream_t stream)
{
    dim3 g(p.grid), th(p.block_dim);
#define EXL3_RUN(bits_, cb_, mm_, cfg_)                                                       \
    do { exl3_gemv_plain<bits_, false, cb_, mm_, cfg_, false>                                 \
      <<<g, th, 0, stream>>>(A, B, C, p.m, p.k, p.n,                                          \
                             (int*) nullptr, (const half*) nullptr, (half*) nullptr,          \
                             (const half*) nullptr); } while (false)
    if (!p.ok || p.m < 1 || p.m > 8) return false;
    // The Qwen3.8-27B-EXL3-3.5bpw checkpoint is 100% mul1 => cb2 (1.5.1 rule:
    // mcg->1, mul1->2). cb0/cb1 kept for other EXL3 families (K=4 baseline / mcg).
    // Comma-operand hazard in the old (bits==a, cb==b) spelling is fixed: && now.
    if (p.bits == 3 && p.cb == 2) { if (p.mmode == 0) { if (p.cfg == 0) EXL3_RUN(3, 2, 0, 0); else EXL3_RUN(3, 2, 0, 1); } else { if (p.cfg == 0) EXL3_RUN(3, 2, 1, 0); else EXL3_RUN(3, 2, 1, 1); } return true; }
    if (p.bits == 4 && p.cb == 2) { if (p.mmode == 0) { if (p.cfg == 0) EXL3_RUN(4, 2, 0, 0); else EXL3_RUN(4, 2, 0, 1); } else { if (p.cfg == 0) EXL3_RUN(4, 2, 1, 0); else EXL3_RUN(4, 2, 1, 1); } return true; }
    if (p.bits == 4 && p.cb == 1) { if (p.mmode == 0) { if (p.cfg == 0) EXL3_RUN(4, 1, 0, 0); else EXL3_RUN(4, 1, 0, 1); } else { if (p.cfg == 0) EXL3_RUN(4, 1, 1, 0); else EXL3_RUN(4, 1, 1, 1); } return true; }
    if (p.bits == 3 && p.cb == 1) { if (p.mmode == 0) { if (p.cfg == 0) EXL3_RUN(3, 1, 0, 0); else EXL3_RUN(3, 1, 0, 1); } else { if (p.cfg == 0) EXL3_RUN(3, 1, 1, 0); else EXL3_RUN(3, 1, 1, 1); } return true; }
    if (p.bits == 4 && p.cb == 0) { if (p.mmode == 0) { if (p.cfg == 0) EXL3_RUN(4, 0, 0, 0); else EXL3_RUN(4, 0, 0, 1); } else { if (p.cfg == 0) EXL3_RUN(4, 0, 1, 0); else EXL3_RUN(4, 0, 1, 1); } return true; }
    return false;
}
#undef EXL3_RUN


// ---- top-level one-call EXL3 decode (m<=8): had_in, gemv, had_out, cast ----
// A  : [m, k] fp16 activations (row-major, the engine K x T plane transposed)
// B  : trellis int16 [k/16, n/16, 8K] (K int bits; cb already applied by plan)
// C16: [m, n] fp16 scratch owned by the caller, then cast to bf16 out
// out: [m, n] bf16 destination
// (had plane grids are (m, k/128) / (m, n/128): rows first, per 1.5.1.)
__host__ bool exl3_linear_decode(const half* A, const uint16_t* B,
                                        half* A_h, half* C16, uint16_t* outbf,
                                        const Exl3GemvPlan& p,
                                        const half* suh, const half* svh,
                                        cudaStream_t stream)
{
    if (!p.ok) return false;
    dim3 gi(p.m, p.k / 128), go(p.m, p.n / 128);
    const int kb = p.k, nb = p.n;
    exl3_had_in_plane<<<gi, dim3(32), 0, stream>>>(A, A_h, suh, kb);
    if (!exl3_gemv_run(p, A_h, B, C16, stream)) return false;
    exl3_had_out_plane<<<go, dim3(32), 0, stream>>>(C16, svh, nb);
    const size_t nelem = (size_t) p.m * p.n;
    const int blocks = (int)((nelem + 255) / 256);
    exl3_cast16_to_bf16<<<blocks, 256, 0, stream>>>(C16, outbf, nelem);
    return true;
}
} // namespace exl3
} // namespace ninfer
