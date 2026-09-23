// EXL3 input transform: A_had[g] = had128(x * suh[g]).
//
// ExLlamaV3 folds this into the front of its gemm and pays for it with a
// cooperative launch (the grid-wide sync between transform and matmul).
// Splitting it out costs one extra pass over the activations -- a few
// microseconds, since x is small and L2-resident -- and in exchange the gemm
// becomes an ordinary kernel: no cooperative launch, no grid size tied to the SM
// count, and nothing special to do under CUDA graph capture.
//
// A fused layer (qkv_proj, gate_up_proj) needs one transform per shard, because
// each shard was quantized with its own input scales. All of them are produced
// in a single launch here, loading x once and writing G outputs, rather than one
// launch and one re-read of x per shard.

// Cinference raw-pointer: the ATen host wrappers are replaced by _row seams at
// the bottom. Kernels (had_in, glu) are untouched from cuda-exl3 (MIT).
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

#include "exl3_common.cuh"
#include "exl3_had.cuh"

namespace cuda_exl3 {

template <typename IN_T>
__global__ void exl3_had_in_kernel(const IN_T* __restrict__ x, half* __restrict__ a_had,
                                   const half* __restrict__ suh, int m, int k, int groups)
{
    int blocks_per_row = k / 128;
    long long total = (long long) m * blocks_per_row;
    int warps_per_block = blockDim.x / 32;
    long long w = (long long) blockIdx.x * warps_per_block + (threadIdx.x >> 5);
    if (w >= total) return;

    int row = (int) (w / blocks_per_row);
    int blk = (int) (w % blocks_per_row);
    long long off = (long long) row * k + blk * 128;
    int lane = threadIdx.x & 31;

    for (int g = 0; g < groups; ++g)
        had128_warp_in<IN_T>(x + off, a_had + (long long) g * m * k + off,
                             suh + (long long) g * k + blk * 128, lane);
}

void exl3_had_in_row(const half* x, half* a_had, const half* suh,
                     int m, int k, int groups, cudaStream_t stream)
{
    long long total_warps = (long long) m * (k / 128);
    const int threads = 256;
    long long blocks = (total_warps + threads / 32 - 1) / (threads / 32);
    exl3_had_in_kernel<half><<<(unsigned) blocks, threads, 0, stream>>>(
        (const half*) x, a_had, suh, m, k, groups);
}

void exl3_had_in_row_bf16(const __nv_bfloat16* x, half* a_had, const half* suh,
                          int m, int k, int groups, cudaStream_t stream)
{
    long long total_warps = (long long) m * (k / 128);
    const int threads = 256;
    long long blocks = (total_warps + threads / 32 - 1) / (threads / 32);
    exl3_had_in_kernel<__nv_bfloat16><<<(unsigned) blocks, threads, 0, stream>>>(
        (const __nv_bfloat16*) x, a_had, suh, m, k, groups);
}

// ---------------------------------------------------------------------------
// MoE variant: gather the routed rows and apply that row's expert's scales.
//
// Rows arrive already sorted by expert and padded so each expert's run is a
// whole multiple of block_m (vLLM's moe_align_block_size), which is what lets
// the GEMM treat the expert as uniform over a row block. Padding rows are
// zeroed here so they contribute nothing downstream.
// ---------------------------------------------------------------------------

template <typename IN_T>
__global__ void exl3_moe_had_in_kernel(const IN_T* __restrict__ x,
                                       half* __restrict__ a_had,
                                       const half* __restrict__ suh,
                                       const int* __restrict__ sorted_ids,
                                       const int* __restrict__ expert_ids,
                                       const int* __restrict__ n_rows,
                                       int rows, int k, int groups, int block_m,
                                       int top_k, int m_valid, bool skip_padding)
{
    // Row and 128-block come straight from the grid. They used to be recovered
    // from a flat warp index with a 64-bit divide and modulo -- and there is no
    // hardware 64-bit integer divide, so that was a software sequence per warp,
    // once for every (row, 128-block) pair: 65k of them at M=256 and 1.1M at a
    // 2048-token prefill chunk. The kernel was reading and writing at 57% of
    // what the device delivers with nothing else to blame.
    const int blocks_per_row = k / 128;
    const int warps_per_block = blockDim.x / 32;
    const int row = (int) blockIdx.x;
    const int blk = (int) blockIdx.y * warps_per_block + (int) (threadIdx.x >> 5);
    if (row >= rows || blk >= blocks_per_row) return;
    // Surplus rows from the worst-case padding are never read by the gemm
    // (it retires those blocks on the same count), so skip them entirely.
    if (n_rows && row >= *n_rows) return;
    int lane = threadIdx.x & 31;
    long long dst = (long long) row * k + blk * 128;

    // An empty sorted_ids means the rows are already in place (the second
    // projection consumes the first one's output), so the gather is the
    // identity and no index array needs building.
    // Retire a block that belongs to no expert before doing anything else. This
    // has to come above the padding branch, not below it: the gemm skips the
    // whole block, so nobody reads these rows and even writing zeros to them is
    // wasted bandwidth. Under expert parallel most blocks are another rank's,
    // and at small batch most rows are padding, so that was the bulk of this
    // kernel's traffic. Reported by @NNNtrance in #1.
    int e = expert_ids[row / block_m];
    if (e < 0) return;

    int idx = sorted_ids ? sorted_ids[row] : row;
    if (idx >= m_valid)
    {
        // Padding row. When the gemm skips fetching these -- cp.async zero-fills
        // a row it does not read -- writing zeros here is traffic nobody
        // consumes. Both sides take the same decision from the caller and use
        // the same predicate, sorted_ids[row] against m_valid, so they cannot
        // disagree; if the gemm is going to read the row, it needs the zeros.
        if (skip_padding) return;
        for (int g = 0; g < groups; ++g)
            ((half4*) (a_had + (long long) g * rows * k + dst))[lane] =
                half4{__float2half2_rn(0.f), __float2half2_rn(0.f)};
        return;
    }

    int token = idx / top_k;
    const IN_T* src = x + (long long) token * k + blk * 128;

    for (int g = 0; g < groups; ++g)
        had128_warp_in<IN_T>(src, a_had + (long long) g * rows * k + dst,
                             suh + ((long long) e * groups + g) * k + blk * 128, lane);
}

// ---------------------------------------------------------------------------
// Fused SwiGLU + input transform for the MoE down-projection.
//
// The rows here are already in routed order (they are the first GEMM's output),
// so the gather is the identity and there is no sorted_ids, no top_k and only
// one group. Padding rows carry zeros through from the first transform, so
// silu(0)*0 = 0 needs no special case.
// ---------------------------------------------------------------------------
template <typename IN_T>
__global__ void exl3_moe_glu_had_in_kernel(const IN_T* __restrict__ x,
                                           half* __restrict__ a_had,
                                           const half* __restrict__ suh,
                                           const int* __restrict__ expert_ids,
                                           const int* __restrict__ n_rows,
                                           int rows, int k, int block_m)
{
    int blocks_per_row = k / 128;
    long long total = (long long) rows * blocks_per_row;
    int warps_per_block = blockDim.x / 32;
    long long w = (long long) blockIdx.x * warps_per_block + (threadIdx.x >> 5);
    if (w >= total) return;

    int row = (int) (w / blocks_per_row);
    if (n_rows && row >= *n_rows) return;
    int blk = (int) (w % blocks_per_row);
    int lane = threadIdx.x & 31;

    int e = expert_ids[row / block_m];
    if (e < 0) return;                  // block belongs to no expert

    // x is (rows, 2k): gate in the first half of the row, up in the second.
    const IN_T* gate = x + (long long) row * 2 * k + blk * 128;
    had128_warp_glu_in<IN_T>(gate, gate + k,
                             a_had + (long long) row * k + blk * 128,
                             suh + (long long) e * k + blk * 128, lane);
}

void exl3_glu_had_in_row(const __nv_bfloat16* x, half* a_had, const half* suh,
                         const int* expert_ids, int rows, int k, int block_m,
                         cudaStream_t stream)
{
    long long total_warps = (long long) rows * (k / 128);
    const int threads = 256;
    long long blocks = (total_warps + threads / 32 - 1) / (threads / 32);
    exl3_moe_glu_had_in_kernel<__nv_bfloat16><<<(unsigned) blocks, threads, 0,
                                                stream>>>(
        (const __nv_bfloat16*) x, a_had, suh, expert_ids, nullptr, rows, k,
        block_m);
}

}  // namespace cuda_exl3
