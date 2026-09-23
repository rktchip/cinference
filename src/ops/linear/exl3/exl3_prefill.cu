// exl3_prefill.cu
// EXL3 prefill (m > 8) + reconstruct (K >= 5) path for cinference.
// Pipeline (1.5.1 non-fused reconstruct_hgemm, rows < 1024):
//   1. dequant -> naked W16 [k, n] fp16 (exl3_reconstruct_rot; caller)
//   2. had_in  : A -> A_h (H128 * suh pre-scale), grid (m, k/128)
//   3. GEMM    : C16[m, n] = A_h * W16 (SIMT fp16 load, fp32 accumulate;
//                correctness-first; an mma variant may replace it in place)
//   4. had_out : C16 in place (H128 * svh post-scale), grid (m, n/128)
//   5. cast    : fp16 -> bf16 into the engine out plane
// Had planes: 32-thread 128-element blocks; PLANE is the second grid dim
// (scale window = blockIdx.y*32+t in the vendored inner; 1.5.1 had_r_128).
#include "exl3_prefill.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace ninfer {
namespace exl3 {

static constexpr float EXL3_PRE_R_SCALE = 0.088388347648f;  // 1/sqrt(128), ref

// 32-thread 128-plane; grid = (rows, planes); scale window = blockIdx.y (1.5.1).
__global__ void exl3_prefill_had_in(const half* __restrict__ A, half* __restrict__ A_h,
                                    const half* __restrict__ suh, int k)
{
    const int row = blockIdx.x, pl = blockIdx.y;
    had_hf_r_128_inner<true, false>(A + (size_t) row * k + pl * 128,
                                    A_h + (size_t) row * k + pl * 128,
                                    suh, EXL3_PRE_R_SCALE);
}

__global__ void exl3_prefill_had_out(half* __restrict__ C, const half* __restrict__ svh, int n)
{
    const int row = blockIdx.x, pl = blockIdx.y;
    had_hf_r_128_inner<false, true>(C + (size_t) row * n + pl * 128,
                                    C + (size_t) row * n + pl * 128,
                                    svh, EXL3_PRE_R_SCALE);
}

// SIMT fp16 GEMM: C16[m, n] = A_h[m, k] * W16[k, n] (both row-major;
// W16's second dim is n-fastest, per reconstruct_rot emit order).
// One lane per output column (within the warp); W reads coalesce.
__global__ void exl3_prefill_gemm(const half* __restrict__ A, const half* __restrict__ W,
                                  half* __restrict__ C, int k, int n)
{
    const int col = blockIdx.x * 32 + (threadIdx.x & 31);
    const int row = blockIdx.y;
    if (col >= n) return;
    float acc = 0.0f;
    const half* a1 = A + (size_t) row * k;
    const half* w1 = W + col;
    #pragma unroll 4
    for (int j = 0; j < k; ++j) {
        acc += __half2float(a1[j]) * __half2float(w1[(size_t) j * n]);
    }
    C[(size_t) row * n + col] = __float2half_rn(acc);
}

__global__ void exl3_prefill_cast_fp16_to_bf16(const half* __restrict__ in, uint16_t* __restrict__ out,
                                               size_t n)
{
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __bfloat16_as_ushort(__nv_bfloat16(__half2float(in[i])));
}

// ---- raw-pointer dequant dispatch over the vendored 1.5.1 instance arrays ----
bool exl3_reconstruct_rot(Exl3ReconstructArgs& a, cudaStream_t stream)
{
    if (!a.packed || !a.unpacked || a.K < 1 || a.K > 8 || a.k % 16 != 0 || a.n % 128 != 0)
        return false;
    if (a.packed_n_offset % 128 != 0 || a.packed_n_offset + a.n > a.n) return false;
    const int cbi = exl3_reconstruct_cbi(a.K, a.mcg, a.mul1);
    if (cbi < 0 || cbi >= (int) reconstruct_kernel_instances.size()) return false;
    a.packed_cols = a.n / 16;                              // 16-wide tiles per k/16 row
    dim3 g(a.n / 128, a.k / 16);
    reconstruct_kernel_instances[cbi]<<<g, 256, 0, stream>>>(a.unpacked, a.packed, a.packed_cols,
                                                             a.packed_n_offset);
    return cudaGetLastError() == cudaSuccess;
}

bool exl3_reconstruct_had(Exl3ReconstructArgs& a, cudaStream_t stream)
{
    if (!a.packed || !a.unpacked || !a.suh || !a.svh || a.K < 1 || a.K > 8 || a.k % 128 != 0 ||
        a.n % 128 != 0)
        return false;
    if (a.packed_n_offset % 128 != 0 || a.packed_n_offset + a.n > a.n) return false;
    const int cbi = exl3_reconstruct_cbi(a.K, a.mcg, a.mul1);
    if (cbi < 0 || cbi >= (int) reconstruct_had_kernel_instances.size()) return false;
    a.packed_cols = a.n / 16;
    dim3 g(a.n / 128, a.k / 128);
    reconstruct_had_kernel_instances[cbi]<<<g, 256, 0, stream>>>(a.unpacked, a.packed, a.suh,
                                                                 a.svh, a.packed_cols,
                                                                 a.packed_n_offset / 16);
    return cudaGetLastError() == cudaSuccess;
}

// One-shot prefill, m >= 9 (see header for the convention).
bool exl3_prefill_op(const half* A, const half* W16, uint16_t* outbf, half* scratch,
                     const Exl3Weight& w, std::int32_t m, cudaStream_t stream)
{
    if (!A || !W16 || !outbf || !scratch || !w.suh || !w.svh) return false;
    if (m < 9) return false;                                // gemv path owns m <= 8
    if (w.k % 128 || w.n % 128) return false;
    const int k = w.k, n = w.n;
    if (exl3_prefill_workspace_bytes(m, k, n) == 0) return false;
    half* A_h = scratch;                                    // [m, k]
    half* C16 = scratch + (size_t) m * k;                   // [m, n]
    dim3 gi(m, k / 128), go(m, n / 128);                    // (rows, planes)
    exl3_prefill_had_in<<<gi, dim3(32), 0, stream>>>(A, A_h, w.suh, k);
    dim3 gg(n / 32, m);
    exl3_prefill_gemm<<<gg, dim3(32), 0, stream>>>(A_h, W16, C16, k, n);
    exl3_prefill_had_out<<<go, dim3(32), 0, stream>>>(C16, w.svh, n);
    const size_t nelem = (size_t) m * n;
    const int blocks = (int)((nelem + 255) / 256);
    exl3_prefill_cast_fp16_to_bf16<<<blocks, 256, 0, stream>>>(C16, outbf, nelem);
    return cudaGetLastError() == cudaSuccess;
}

std::size_t exl3_prefill_workspace_bytes(std::int32_t m, std::int32_t k, std::int32_t n)
{
    if (m < 1 || k < 1 || n < 1) return 0;
    return (std::size_t) m * k * 2 + (size_t) m * n * 2;
}

} // namespace exl3
} // namespace ninfer
