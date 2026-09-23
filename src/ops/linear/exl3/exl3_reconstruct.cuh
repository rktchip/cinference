// EXL3 reconstruct (dequant) kernels + instance tables.
// Vendored from exllamav3 v1.5.1 (MIT, turboderp-org/exllamav3),
// exllamav3_ext/quant/reconstruct.cu. Kernel bodies + instance tables
// are VERBATIM; the at::Tensor host contract is replaced below by a
// raw-pointer host (Exl3ReconstructArgs). Includes map onto the vendored
// exl3_dq.cuh (dq_dispatch, FragB) + exl3_hadamard.cuh (shuffle_had_h2x32).
#pragma once

#include <cuda_fp16.h>
#include <cstddef>
#include <cuda_runtime.h>
#include <array>
// order: shim BEFORE dq (dq uses FragB from the shim, cf. exl3_launcher.h order)
#include "exl3_ptx_shim.cuh"
#include "exl3_dq.cuh"
#include "exl3_hadamard.cuh"

namespace ninfer {
namespace exl3 {
template <int K, int cb, bool HALF = false>
__device__ __forceinline__
void reconstruct_tile
(
    half* __restrict__ g_unpacked,
    const uint16_t* __restrict__ g_packed,
    int packed_blocks_n,
    int packed_n_offset
)
{
    constexpr int packed_size = 16 * K + (HALF ? 8 : 0);  // in uint16s

    int t = threadIdx.x;
    int lane_id = t % 32;
    int warp_id = t / 32;
    int k = blockIdx.y;
    int n = blockIdx.x * 8;
    int tiles_n = gridDim.x;
    int out_blocks_n = tiles_n * 8;

    // Load packed 16*128 tile
    __shared__ uint32_t s_packed[8][packed_size / 2];
    g_packed += (k * packed_blocks_n + packed_n_offset + n) * packed_size;
    if (t < packed_size)
        ((int4*) s_packed)[t] = ((int4*) g_packed)[t];
    __syncthreads();

    // Dequant
    register FragB frag[2];
    dq_dispatch<K, cb, HALF>(s_packed[warp_id], lane_id * 8, frag[0], frag[1]);

    // Shuffle from tensor core layout to row major tile
//    __shared__ half tile[16 * 8 * 16];
    __shared__ half2 tile[16][8][8];

    half2 n0 = __shfl_down_sync(0xFFFFFFFF, frag[0][0], 4, 32);
    half2 n1 = __shfl_down_sync(0xFFFFFFFF, frag[0][1], 4, 32);
    half2 n2 = __shfl_down_sync(0xFFFFFFFF, frag[1][0], 4, 32);
    half2 n3 = __shfl_down_sync(0xFFFFFFFF, frag[1][1], 4, 32);

    if (!(lane_id & 4))
    {
        half2 m0 = __halves2half2(__low2half(frag[0][0]), __low2half(n0));
        half2 m1 = __halves2half2(__high2half(frag[0][0]), __high2half(n0));
        half2 m2 = __halves2half2(__low2half(frag[0][1]), __low2half(n1));
        half2 m3 = __halves2half2(__high2half(frag[0][1]), __high2half(n1));
        half2 m4 = __halves2half2(__low2half(frag[1][0]), __low2half(n2));
        half2 m5 = __halves2half2(__high2half(frag[1][0]), __high2half(n2));
        half2 m6 = __halves2half2(__low2half(frag[1][1]), __low2half(n3));
        half2 m7 = __halves2half2(__high2half(frag[1][1]), __high2half(n3));
        int r0 = (lane_id % 4) * 2;
        int r1 = r0 + 1;
        int r2 = r0 + 8;
        int r3 = r0 + 9;
        int c0 = lane_id / 8;
        int c1 = c0 + 4;
        tile[r0][warp_id][c0] = m0;
        tile[r1][warp_id][c0] = m1;
        tile[r2][warp_id][c0] = m2;
        tile[r3][warp_id][c0] = m3;
        tile[r0][warp_id][c1] = m4;
        tile[r1][warp_id][c1] = m5;
        tile[r2][warp_id][c1] = m6;
        tile[r3][warp_id][c1] = m7;
    }
    __syncthreads();

    // Store unpacked tile
    int r = t / 16;
    int c = t % 16;
    int4* tile_int4 = (reinterpret_cast<int4*> (tile));
    int4* out_int4 = ((int4*) g_unpacked) + (k * 16 + r) * 2 * out_blocks_n + n * 2 + c;
    *out_int4 = tile_int4[t];
}

template <int K, int cb, bool HALF = false>
__global__ __launch_bounds__(256)
void reconstruct_kernel
(
    half* __restrict__ g_unpacked,
    const uint16_t* __restrict__ g_packed,
    int packed_blocks_n,
    int packed_n_offset
)
{
    reconstruct_tile<K, cb, HALF>(g_unpacked, g_packed, packed_blocks_n, packed_n_offset);
}

// Batched variant: blockIdx.z selects the matrix from a pointer table, outputs are consecutive
// [k, n] slabs out_stride halfs apart. Whole matrices only.
template <int K, int cb, bool HALF = false>
__global__ __launch_bounds__(256)
void reconstruct_batch_kernel
(
    half* __restrict__ g_unpacked,
    const uint16_t* const* __restrict__ packed_ptrs,
    int packed_blocks_n,
    size_t out_stride
)
{
    int b = blockIdx.z;
    reconstruct_tile<K, cb, HALF>(g_unpacked + (size_t) b * out_stride, packed_ptrs[b], packed_blocks_n, 0);
}

// Index cb * 8 + K - 1 for integer K; 24 + K - 1 for the half-integer rates 1.5 / 2.5 / 3.5 (mul1 only)
#define __(i, cb) reconstruct_batch_kernel<i, cb>
constexpr auto reconstruct_batch_kernel_instances = std::array
{
    __(1, 0), __(2, 0), __(3, 0), __(4, 0), __(5, 0), __(6, 0), __(7, 0), __(8, 0),
    __(1, 1), __(2, 1), __(3, 1), __(4, 1), __(5, 1), __(6, 1), __(7, 1), __(8, 1),
    __(1, 2), __(2, 2), __(3, 2), __(4, 2), __(5, 2), __(6, 2), __(7, 2), __(8, 2),
    reconstruct_batch_kernel<1, 2, true>, reconstruct_batch_kernel<2, 2, true>, reconstruct_batch_kernel<3, 2, true>
};
#undef __

// Index cb * 8 + K - 1 for integer K; 24 + K - 1 for the half-integer rates 1.5 / 2.5 / 3.5 (mul1 only)
#define __(i, cb) reconstruct_kernel<i, cb>
constexpr auto reconstruct_kernel_instances = std::array
{
    __(1, 0), __(2, 0), __(3, 0), __(4, 0), __(5, 0), __(6, 0), __(7, 0), __(8, 0),
    __(1, 1), __(2, 1), __(3, 1), __(4, 1), __(5, 1), __(6, 1), __(7, 1), __(8, 1),
    __(1, 2), __(2, 2), __(3, 2), __(4, 2), __(5, 2), __(6, 2), __(7, 2), __(8, 2),
    reconstruct_kernel<1, 2, true>, reconstruct_kernel<2, 2, true>, reconstruct_kernel<3, 2, true>
};
#undef __

#define RH_THREADS 256

template <int K, int cb, bool HALF = false>
__device__ __forceinline__
void reconstruct_had_tile
(
    half* __restrict__ g_unpacked,
    const uint16_t* __restrict__ g_packed,
    const half* __restrict__ suh,
    const half* __restrict__ svh,
    int packed_blocks_n,
    int packed_n_offset
)
{
    constexpr int packed_size = 16 * K + (HALF ? 8 : 0);
    constexpr float r_scale = 0.08838834764831845f;

    int t = threadIdx.x;
    int lane_id = t % 32;
    int warp_id = t / 32;
    int kb = blockIdx.y;
    int nb = blockIdx.x;
    int n = nb * 8;
    int row_len = gridDim.x * 128;

    __shared__ uint32_t s_packed[8][8][packed_size / 2];
    __shared__ half2 stile[128 * 64];

    auto tix = [&] (int R, int q, int p)
    {
        return R * 64 + (q ^ ((R >> 2) & 31)) * 2 + p;
    };

    constexpr int j_int4 = packed_size / 8;
    for (int u = t; u < 8 * 8 * j_int4; u += RH_THREADS)
    {
        int j = u / (8 * j_int4);
        int r = u % (8 * j_int4);
        const uint16_t* gp = g_packed +
            ((size_t) ((kb * 8 + j) * packed_blocks_n + packed_n_offset + n)) * packed_size;
        ((int4*) s_packed[j])[r] = ((const int4*) gp)[r];
    }
    __syncthreads();

    for (int jj = 0; jj < 8 * 8 / (RH_THREADS / 32); ++jj)
    {
        int j = (warp_id / 8) * (8 / (RH_THREADS / 256)) + jj;
        int wn = warp_id % 8;
        register FragB frag[2];
        dq_dispatch<K, cb, HALF>(s_packed[j][wn], lane_id * 8, frag[0], frag[1]);

        half2 n0 = __shfl_down_sync(0xFFFFFFFF, frag[0][0], 4, 32);
        half2 n1 = __shfl_down_sync(0xFFFFFFFF, frag[0][1], 4, 32);
        half2 n2 = __shfl_down_sync(0xFFFFFFFF, frag[1][0], 4, 32);
        half2 n3 = __shfl_down_sync(0xFFFFFFFF, frag[1][1], 4, 32);

        if (!(lane_id & 4))
        {
            half2 m0 = __halves2half2(__low2half(frag[0][0]), __low2half(n0));
            half2 m1 = __halves2half2(__high2half(frag[0][0]), __high2half(n0));
            half2 m2 = __halves2half2(__low2half(frag[0][1]), __low2half(n1));
            half2 m3 = __halves2half2(__high2half(frag[0][1]), __high2half(n1));
            half2 m4 = __halves2half2(__low2half(frag[1][0]), __low2half(n2));
            half2 m5 = __halves2half2(__high2half(frag[1][0]), __high2half(n2));
            half2 m6 = __halves2half2(__low2half(frag[1][1]), __low2half(n3));
            half2 m7 = __halves2half2(__high2half(frag[1][1]), __high2half(n3));
            int r0 = j * 16 + (lane_id % 4) * 2;
            int r1 = r0 + 1;
            int r2 = r0 + 8;
            int r3 = r0 + 9;
            int c0 = lane_id / 8;
            int q0 = (wn * 8 + c0) >> 1, p0 = c0 & 1;
            int q1 = (wn * 8 + c0 + 4) >> 1, p1 = c0 & 1;
            stile[tix(r0, q0, p0)] = m0;
            stile[tix(r1, q0, p0)] = m1;
            stile[tix(r2, q0, p0)] = m2;
            stile[tix(r3, q0, p0)] = m3;
            stile[tix(r0, q1, p1)] = m4;
            stile[tix(r1, q1, p1)] = m5;
            stile[tix(r2, q1, p1)] = m6;
            stile[tix(r3, q1, p1)] = m7;
        }
    }
    __syncthreads();

    const half2 rs2 = __float2half2_rn(r_scale);
    constexpr int CHUNKS_PW = 32 / (RH_THREADS / 32);
    #pragma unroll
    for (int qq = 0; qq < CHUNKS_PW; ++qq)
    {
        int q = warp_id * CHUNKS_PW + qq;
        int qs = q ^ lane_id;
        // a[i]/b[i]: 4 columns x row (4 * lane + i), all four rows share swizzle == lane
        half2 a[4], b[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            half4 v = *((const half4*) (stile + (lane_id * 4 + i) * 64 + qs * 2));
            a[i] = v.x;
            b[i] = v.y;
        }
        // butterfly over the 4 rows, two columns at a time, fp16 with pre-applied scale
        #pragma unroll
        for (int x = 0; x < 2; ++x)
        {
            half2* v = x == 0 ? a : b;
            half2 s0 = __hadd2(v[0], v[1]), d0 = __hsub2(v[0], v[1]);
            half2 s1 = __hadd2(v[2], v[3]), d1 = __hsub2(v[2], v[3]);
            v[0] = __hmul2(__hadd2(s0, s1), rs2);
            v[1] = __hmul2(__hadd2(d0, d1), rs2);
            v[2] = __hmul2(__hsub2(s0, s1), rs2);
            v[3] = __hmul2(__hsub2(d0, d1), rs2);
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                v[i] = shuffle_had_h2x32(v[i], lane_id);
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            half4 v;
            v.x = a[i];
            v.y = b[i];
            *((half4*) (stile + (lane_id * 4 + i) * 64 + qs * 2)) = v;
        }
    }
    __syncthreads();

    // Row transform fused with the store: after the lane butterfly, lane l holds the
    // FINAL columns 4l..4l+3 of row R. Apply the sign scales in registers and write the
    // coalesced 256-byte row directly.
    constexpr int ROWS_PW = 128 / (RH_THREADS / 32);
    const half4 sv4 = ((const half4*) svh)[nb * 32 + lane_id];
    #pragma unroll
    for (int rr = 0; rr < ROWS_PW; ++rr)
    {
        int R = warp_id * ROWS_PW + rr;
        int base = R * 64 + (lane_id ^ ((R >> 2) & 31)) * 2;
        half2 v01 = stile[base];
        half2 v23 = stile[base + 1];
        float v0 = __low2float(v01), v1 = __high2float(v01);
        float v2 = __low2float(v23), v3 = __high2float(v23);
        float s0 = v0 + v1, d0 = v0 - v1;
        float s1 = v2 + v3, d1 = v2 - v3;
        half2 h01 = __hmul2(__floats2half2_rn(s0 + s1, d0 + d1), rs2);
        half2 h23 = __hmul2(__floats2half2_rn(s0 - s1, d0 - d1), rs2);
        h01 = shuffle_had_h2x32(h01, lane_id);
        h23 = shuffle_had_h2x32(h23, lane_id);
        half2 su2 = __half2half2(suh[kb * 128 + R]);
        half4 v;
        v.x = __hmul2(__hmul2(h01, su2), sv4.x);
        v.y = __hmul2(__hmul2(h23, su2), sv4.y);
        *((half4*) (g_unpacked + (size_t) (kb * 128 + R) * row_len + nb * 128 + lane_id * 4)) = v;
    }
}


template <int K, int cb, bool HALF = false>
__global__ __launch_bounds__(RH_THREADS)
void reconstruct_had_kernel
(
    half* __restrict__ g_unpacked,
    const uint16_t* __restrict__ g_packed,
    const half* __restrict__ suh,
    const half* __restrict__ svh,
    int packed_blocks_n,
    int packed_n_offset
)
{
    reconstruct_had_tile<K, cb, HALF>(g_unpacked, g_packed, suh, svh, packed_blocks_n, packed_n_offset);
}

// Batched variant: blockIdx.z selects the matrix from per-matrix pointer tables, the outputs
// are consecutive [k, n] slabs out_stride halfs apart. Whole matrices only (no n slicing).
template <int K, int cb, bool HALF = false>
__global__ __launch_bounds__(RH_THREADS)
void reconstruct_had_batch_kernel
(
    half* __restrict__ g_unpacked,
    const uint16_t* const* __restrict__ packed_ptrs,
    const half* const* __restrict__ suh_ptrs,
    const half* const* __restrict__ svh_ptrs,
    int packed_blocks_n,
    size_t out_stride
)
{
    int b = blockIdx.z;
    reconstruct_had_tile<K, cb, HALF>
    (
        g_unpacked + (size_t) b * out_stride,
        packed_ptrs[b],
        suh_ptrs[b],
        svh_ptrs[b],
        packed_blocks_n,
        0
    );
}

// Index cb * 8 + K - 1 for integer K; 24 + K - 1 for the half-integer rates 1.5 / 2.5 / 3.5 (mul1 only)
#define __(i, cb) reconstruct_had_batch_kernel<i, cb>
constexpr auto reconstruct_had_batch_kernel_instances = std::array
{
    __(1, 0), __(2, 0), __(3, 0), __(4, 0), __(5, 0), __(6, 0), __(7, 0), __(8, 0),
    __(1, 1), __(2, 1), __(3, 1), __(4, 1), __(5, 1), __(6, 1), __(7, 1), __(8, 1),
    __(1, 2), __(2, 2), __(3, 2), __(4, 2), __(5, 2), __(6, 2), __(7, 2), __(8, 2),
    reconstruct_had_batch_kernel<1, 2, true>, reconstruct_had_batch_kernel<2, 2, true>, reconstruct_had_batch_kernel<3, 2, true>
};
#undef __

// Index cb * 8 + K - 1 for integer K; 24 + K - 1 for the half-integer rates 1.5 / 2.5 / 3.5 (mul1 only)
#define __(i, cb) reconstruct_had_kernel<i, cb>
constexpr auto reconstruct_had_kernel_instances = std::array
{
    __(1, 0), __(2, 0), __(3, 0), __(4, 0), __(5, 0), __(6, 0), __(7, 0), __(8, 0),
    __(1, 1), __(2, 1), __(3, 1), __(4, 1), __(5, 1), __(6, 1), __(7, 1), __(8, 1),
    __(1, 2), __(2, 2), __(3, 2), __(4, 2), __(5, 2), __(6, 2), __(7, 2), __(8, 2),
    reconstruct_had_kernel<1, 2, true>, reconstruct_had_kernel<2, 2, true>, reconstruct_had_kernel<3, 2, true>
};
#undef __
}
}

namespace ninfer {
namespace exl3 {

// Raw-pointer reconstruct host contract. cb rule mirrors 1.5.1:
// mcg -> 1, mul1 -> 2, else 0. K in 1..8 (checkpoint: 3/4; K>=5 route here).
struct Exl3ReconstructArgs {
    const uint16_t* packed = nullptr;   // trellis [k/16, n/16, 16K] int16 (as stored)
    half* unpacked         = nullptr;   // out [k, n] fp16
    const half* suh        = nullptr;   // [k] fp16 (had path)
    const half* svh        = nullptr;   // [n] fp16 (had path)
    std::int32_t k               = 0;
    std::int32_t n               = 0;
    std::int32_t packed_cols     = 0;   // = n/16: 16-wide tiles per k/16 row
    std::int32_t packed_n_offset = 0;   // 16-wide output columns; 0 for whole
    std::int32_t K               = 0;   // 1..8
    bool     mcg                 = false;
    bool     mul1                = false;
};

static inline std::int32_t exl3_reconstruct_cbi(std::int32_t K, bool mcg, bool mul1)
{
    int cbi = K - 1;
    if (mcg) cbi += 8;
    else if (mul1) cbi += 16;
    if (cbi < 0 || cbi >= 24) return -1;
    return cbi;
}

// Rotated-basis reconstruct: unpacked = W_hat (dequant only). Grid (n/128, k/16),
// 256 threads. (The gemv decode path consumes W in this rotated basis.)
bool exl3_reconstruct_rot(Exl3ReconstructArgs& a, cudaStream_t stream);

// Original-basis reconstruct, fused dequant + both-side H128 + suh/svh scales:
// per 128x128 tile, diag(svh).H128.(diag(suh).W_hat.H128) with 1/sqrt(128) per
// side — emits the OB weights the prefill GEMM runs against with raw A.
// Grid (n/128, k/128), 256 threads.
bool exl3_reconstruct_had(Exl3ReconstructArgs& a, cudaStream_t stream);

} // namespace exl3
} // namespace ninfer
