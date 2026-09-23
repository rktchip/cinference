// exl3_gemv_plain.cuh
// Carved VERBATIM from exllamav3 1.5.1 quant/exl3_gemv_kernel.cuh (MIT, turboderp-org/exllamav3).
// Hadamard in/out stages commented out: caller runs had_in on A before and had_out on C after
// (plain launches, CUDA-graph safe). Main loop, reduction and store are the reference, unchanged.

// Vendored VERBATIM (mechanical copy, include remaps + namespace wrap only)
// from exllamav3 v1.5.1 (MIT, turboderp-org/exllamav3),
// exllamav3_ext/quant/exl3_gemv_kernel.cuh.
// LogΔΔ edits made for cinference (plain-launch decomposition) are
// marked separately below this header; see exl3_gemv_local.cu.

#pragma once

// Small-m GEMV path for the EXL3 GEMM, QTIP-style structure (see Cornell-RelaxML/qtip,
// qtip-kernels/src/inference.cu) on the unmodified EXL3 format:
//
// - warps split k and never synchronize during the main loop: no block-wide pipeline barriers;
//   B streams straight to registers with ld.global.cs (evict-first; B is single-use) behind a
//   register prefetch ring
// - the two-word bit windows of the trellis stream are resolved in-warp: with SMEM_STAGE = false
//   via lane shuffles (the extraction helpers in exl3_dq.cuh read exactly two words per lane, at
//   lane-computable indices), with SMEM_STAGE = true by staging the tile words through
//   warp-private shared memory and calling the standard dq_dispatch
// - one m16n8k16 MMA pair per 16x16 weight tile with fp16 accumulation, folded to fp32 on a fixed
//   cadence; per-block cross-warp reduction over the k splits through shared memory
//
// Same launch signature as exl3_gemm_kernel so kernel args and graph parameter patching are
// interchangeable. Cooperative launch: one grid.sync after the input Hadamard stage and one before
// the output stage, no other cross-block coordination. 2, 3 and 4 bpw.
//
// CFG 0 ("narrow", 512 threads, 2 n-tiles/warp, 16 k-splits) wins at attention-projection sizes;
// CFG 1 ("wide", 256 threads, 4 n-tiles/warp, 8 k-splits) wins at large-n FFN sizes. MMODE 0 is
// the m == 1 fast path, MMODE 1 covers 2 <= m <= 8 with row-guarded fragment loads.

// [cinference|plain] cooperative_groups not needed for plain launch
#include "exl3_ptx_shim.cuh"
#include "exl3_dq.cuh"
#include "exl3_gemv_ns.cuh"  // decode8 / dq8_regs_* / mma_ab_h (verbatim vendor)
// -- selects are simplified in exl3_gemv_local.cu)
#include "exl3_hadamard.cuh"

#define EXL3_GEMV_MAX_M 8
// (EXL3_GEMM_ARGS / EXL3_GEMM_T_ARGS verbatim from exllamav3 v1.5.1
//  quant/exl3_kernel_map.cuh; CEIL_DIVIDE per the same file's usage)
#define EXL3_GEMM_T_ARGS \
    const int bits, \
    const bool half_k, \
    const bool c_fp32, \
    const int cb, \
    const int TILESIZE_M, \
    const int TILESIZE_K, \
    const int TILESIZE_N, \
    const int SH_STAGES, \
    const int FRAG_STAGES

#define EXL3_GEMM_ARGS \
    const half* __restrict__  A, \
    const uint16_t* __restrict__ B, \
    void* __restrict__ C, \
    const int size_m, \
    const int size_k, \
    const int size_n, \
    int* __restrict__ locks, \
    const half* __restrict__ suh, \
    half* __restrict__ A_had, \
    const half* __restrict__ svh

#define CEIL_DIVIDE(x, y) (((x) + (y) - 1) / (y))



namespace ninfer {
namespace exl3 {

template <int bits, bool c_fp32, int cb, int MMODE, int CFG, bool SMEM_STAGE, bool HALF = false>
__global__ __launch_bounds__(CFG == 0 ? 512 : 256, HALF && CFG == 0 ? 2 : 1)
void exl3_gemv_plain(EXL3_GEMM_ARGS)
{
    static_assert(HALF ? (bits >= 1 && bits <= 3 && cb == 2) : (bits == 2 || bits == 3 || bits == 4),
                  "exl3_gemv_kernel supports 2, 3 and 4 bpw, and 1.5, 2.5 and 3.5 bpw with mul1");
    constexpr int WK   = CFG == 0 ? 16 : 8;     // k-split (warps per block)
    constexpr int WNT  = CFG == 0 ? 2 : 4;      // adjacent n-tiles per warp
    constexpr int PF   = CFG == 0 ? 4 : 2;      // prefetch ring depth
    constexpr int FOLD = CFG == 0 ? 4 : 2;      // fp16->fp32 fold cadence (divides PF)
    constexpr int THREADS = WK * 32;
    constexpr int ROWS = MMODE == 0 ? 1 : EXL3_GEMV_MAX_M;
    constexpr int COLS = WNT * 16;

    constexpr int TWORDS = HALF ? 4 * (2 * bits + 1) : 8 * bits;              // uint32 per 16x16 tile
    constexpr bool TWO_PER_LOAD = HALF ? bits == 1 : bits == 2;               // two tiles per warp load
    constexpr int LOADS = TWO_PER_LOAD ? WNT / 2 : WNT;                       // warp loads per k-slice
    constexpr int LSTRIDE = TWO_PER_LOAD ? 2 * TWORDS : (TWORDS < 32 ? TWORDS : 32);   // uint32 per load (lanes < LSTRIDE load)
    static_assert(!TWO_PER_LOAD || WNT % 2 == 0, "two tiles per warp load needs an even tile count per warp");

// [cinference|plain] (cooperative grid object removed)

    // Input scales and Hadamard transform, same as exl3_gemm_kernel
    {
// [cinference|plain]         int total_warps = size_m * size_k / 128;
// [cinference|plain]         int warps_grid = gridDim.x * blockDim.x / 32;
// [cinference|plain]         int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;

// [cinference|plain]         for(; this_warp < total_warps; this_warp += warps_grid)
// [cinference|plain]             had_hf_r_128_inner<true, false>
// [cinference|plain]             (
// [cinference|plain]                 A + this_warp * 128,
// [cinference|plain]                 A_had + this_warp * 128,
// [cinference|plain]                 suh + (this_warp * 128) % size_k,
// [cinference|plain]                 0.088388347648f  // 1/sqrt(128)
// [cinference|plain]             );

// [cinference|plain]         grid.sync();
// [cinference|plain]         A = A_had;
    }

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    const int ntiles = size_n / 16;
    const int kslices = size_k / 16;
    const int num_groups = size_n / COLS;

    const int chunk = CEIL_DIVIDE(kslices, WK);
    const int ks0 = warp * chunk;
    const int myn = max(0, min(chunk, kslices - ks0));

    const uint32_t* B32 = (const uint32_t*) B;
    const size_t slice_stride = (size_t) ntiles * TWORDS;   // uint32 per k-slice row
    const half2* A2 = (const half2*) A;
    const half2 hzero = __half2half2(__ushort_as_half(0));

    // A fragment row indices for this lane
    const int r0 = lane >> 2;
    const size_t a_row0 = (size_t) r0 * (size_k / 2);
    const bool r0_ok = MMODE == 0 ? lane < 4 : r0 < size_m;

    // Per-lane extraction constants (see dq8_aligned_2bits / dq8<3, cb, 4> / dq8_half in exl3_dq.cuh)
    [[maybe_unused]] int x_src_a = 0, x_src_b = 0, x_s2 = 0;
    // Half-integer rates: word indices (< TWORDS <= 28) and funnel shifts (< 32) of both window groups, six 5-bit
    // fields in one register, extracted per use (bfe)
    [[maybe_unused]] uint32_t x_pack = 0;
    if constexpr (HALF)
    {
        constexpr int bits2 = 2 * bits + 1;
        constexpr int gspan = 18 + 3 * bits;
        const int t_offset = lane << 3;
        const int e7 = ((t_offset >> 1) + 4) * bits2 + 128 * bits2;
        const int e3 = e7 - 2 * bits2;
        const uint32_t hi7 = ((e7 - 1) / 32) % TWORDS, lo7 = ((e7 - gspan) / 32) % TWORDS, s7 = ((e7 - 1) / 32 + 1) * 32 - e7;
        const uint32_t hi3 = ((e3 - 1) / 32) % TWORDS, lo3 = ((e3 - gspan) / 32) % TWORDS, s3 = ((e3 - 1) / 32 + 1) * 32 - e3;
        x_pack = lo7 | (hi7 << 5) | (s7 << 10) | (lo3 << 15) | (hi3 << 20) | (s3 << 25);
    }
    #define XP(i) ((x_pack >> (5 * (i))) & 31u)   // 0: lo7, 1: hi7, 2: s7, 3: lo3, 4: hi3, 5: s3
    else if constexpr (bits == 2)
    {
        int i1 = lane >> 1;
        x_src_b = i1;
        x_src_a = (i1 + 15) & 15;
    }
    else if constexpr (bits == 3)
    {
        int t_offset = lane << 3;
        int b1 = (t_offset + 257) * 3;
        int b2 = b1 + 21;
        int i0 = (b1 - 16) / 32;
        int i2 = (b2 - 1) / 32;
        x_s2 = (i2 + 1) * 32 - b2;
        x_src_a = i0 % 24;
        x_src_b = i2 % 24;
    }

    __shared__ float sh_red[WK][ROWS][COLS];
    [[maybe_unused]] __shared__ uint32_t sh_stage[SMEM_STAGE ? WK : 1][SMEM_STAGE ? LOADS * LSTRIDE : 1];

    for (int group = blockIdx.x; group < num_groups; group += gridDim.x)
    {
        const uint32_t* bp = B32 + (size_t) ks0 * slice_stride + group * WNT * TWORDS + lane;

        // Prefetch ring (indices must be compile-time or pf lands in local memory)
        auto ld_b = [&] (int i, int l) -> uint32_t
        {
            if constexpr (LSTRIDE < 32)
                return lane < LSTRIDE ? __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE) : 0;
            else
                return __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE);
        };

        uint32_t pf[PF][LOADS];
        #pragma unroll
        for (int d = 0; d < PF; ++d)
            if (d < myn)
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    pf[d][l] = ld_b(d, l);

        FragC_h ch[WNT][2] = {};
        float2 acc0[WNT][2] = {};

        for (int ib = 0; ib < myn; ib += PF)
        {
        #pragma unroll
        for (int d = 0; d < PF; ++d)
        {
            const int i = ib + d;
            if (i >= myn) break;

            uint32_t bw[LOADS];
            #pragma unroll
            for (int l = 0; l < LOADS; ++l)
                bw[l] = pf[d][l];

            if (i + PF < myn)
            {
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    pf[d][l] = ld_b(i + PF, l);
            }

            if constexpr (SMEM_STAGE)
            {
                __syncwarp();
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    if (LSTRIDE == 32 || lane < LSTRIDE)
                        sh_stage[warp][l * LSTRIDE + lane] = bw[l];
                __syncwarp();
            }

            // A fragment: lane covers row lane/4, k pairs (2(lane%4), +1) and (+8, +9)
            const size_t a_col = (size_t) (ks0 + i) * 8 + (lane & 3);
            FragB a01, a23;
            a01[0] = r0_ok ? A2[a_row0 + a_col] : hzero;
            a23[0] = r0_ok ? A2[a_row0 + a_col + 4] : hzero;
            a01[1] = hzero;
            a23[1] = hzero;

            #pragma unroll
            for (int t = 0; t < WNT; ++t)
            {
                FragB f0, f1;
                if constexpr (SMEM_STAGE)
                {
                    const uint32_t* tp = &sh_stage[warp][t * TWORDS];
                    if constexpr (HALF)
                        exl3_gemv_ns::dq8_regs_half<bits, cb>(tp[XP(0)], tp[XP(1)], XP(2), tp[XP(3)], tp[XP(4)], XP(5), f0, f1);
                    else if constexpr (bits == 4)
                        exl3_gemv_ns::dq8_regs_4bits<cb>(tp[(lane + 31) & 31], tp[lane], f0, f1);
                    else if constexpr (bits == 2)
                        exl3_gemv_ns::dq8_regs_2bits<cb>(tp[x_src_a], tp[x_src_b], lane << 3, f0, f1);
                    else
                        exl3_gemv_ns::dq8_regs_3bits<cb>(tp[x_src_a], tp[x_src_b], x_s2, f0, f1);
                }
                else if constexpr (HALF)
                {
                    // 1.5 bpw: tile t lives in lanes (t & 1) * 12 .. +11 of load t / 2; 2.5 / 3.5 bpw: one tile per load
                    const uint32_t w = TWO_PER_LOAD ? bw[t >> 1] : bw[t];
                    const int base = TWO_PER_LOAD ? (t & 1) * TWORDS : 0;
                    uint32_t a7 = __shfl_sync(0xffffffffu, w, base + XP(0));
                    uint32_t b7 = __shfl_sync(0xffffffffu, w, base + XP(1));
                    uint32_t a3 = __shfl_sync(0xffffffffu, w, base + XP(3));
                    uint32_t b3 = __shfl_sync(0xffffffffu, w, base + XP(4));
                    exl3_gemv_ns::dq8_regs_half<bits, cb>(a7, b7, XP(2), a3, b3, XP(5), f0, f1);
                }
                else if constexpr (bits == 4)
                {
                    uint32_t aw = __shfl_sync(0xffffffffu, bw[t], (lane + 31) & 31);
                    exl3_gemv_ns::dq8_regs_4bits<cb>(aw, bw[t], f0, f1);
                }
                else if constexpr (bits == 2)
                {
                    // Two tiles per loaded word group: tile t lives in lanes (t&1)*16 .. +15
                    const uint32_t w = bw[t >> 1];
                    const int base = (t & 1) << 4;
                    uint32_t bwv = __shfl_sync(0xffffffffu, w, base + x_src_b);
                    uint32_t awv = __shfl_sync(0xffffffffu, w, base + x_src_a);
                    exl3_gemv_ns::dq8_regs_2bits<cb>(awv, bwv, lane << 3, f0, f1);
                }
                else  // bits == 3
                {
                    uint32_t awv = __shfl_sync(0xffffffffu, bw[t], x_src_a);
                    uint32_t bwv = __shfl_sync(0xffffffffu, bw[t], x_src_b);
                    exl3_gemv_ns::dq8_regs_3bits<cb>(awv, bwv, x_s2, f0, f1);
                }

                exl3_gemv_ns::mma_ab_h(a01, a23, f0, ch[t][0]);
                exl3_gemv_ns::mma_ab_h(a01, a23, f1, ch[t][1]);
            }

            if ((d + 1) % FOLD == 0 || i + 1 == myn)
            {
                #pragma unroll
                for (int t = 0; t < WNT; ++t)
                    #pragma unroll
                    for (int f = 0; f < 2; ++f)
                    {
                        acc0[t][f].x += __low2float(ch[t][f][0]);
                        acc0[t][f].y += __high2float(ch[t][f][0]);
                        ch[t][f][0] = hzero;
                    }
            }
        }
        }

        // Cross-warp reduction over the k splits. Lane l holds row l/4, cols
        // tile*16 + frag*8 + 2*(l%4) (+1)
        {
            const int c0 = 2 * (lane & 3);
            const bool store0 = MMODE == 0 ? lane < 4 : r0 < ROWS;
            const int sr0 = MMODE == 0 ? 0 : r0;
            if (store0)
            {
                #pragma unroll
                for (int t = 0; t < WNT; ++t)
                    #pragma unroll
                    for (int f = 0; f < 2; ++f)
                    {
                        const int col = t * 16 + f * 8 + c0;
                        sh_red[warp][sr0][col + 0] = acc0[t][f].x;
                        sh_red[warp][sr0][col + 1] = acc0[t][f].y;
                    }
            }
        }
        __syncthreads();

        const int rows_out = MMODE == 0 ? 1 : min(size_m, ROWS);
        for (int idx = threadIdx.x; idx < COLS * rows_out; idx += THREADS)
        {
            const int r = idx / COLS;
            const int c = idx % COLS;
            float sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < WK; ++j)
                sum += sh_red[j][r][c];
            const int col = group * COLS + c;
            if constexpr (c_fp32) ((float*) C)[(size_t) r * size_n + col] = sum;
            else                  ((half*)  C)[(size_t) r * size_n + col] = __float2half_rn(sum);
        }
        __syncthreads();
    }

    #undef XP

    // Output scales and Hadamard transform, same semantics as the inner GEMM epilogue
    {
// [cinference|plain]         grid.sync();

// [cinference|plain]         int total_warps = size_m * size_n / 128;
// [cinference|plain]         int warps_grid = gridDim.x * blockDim.x / 32;
// [cinference|plain]         int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;

// [cinference|plain]         for(; this_warp < total_warps; this_warp += warps_grid)
// [cinference|plain]         {
// [cinference|plain]             if constexpr (c_fp32)
// [cinference|plain]                 had_ff_r_128_inner<false, true>
// [cinference|plain]                 (
// [cinference|plain]                     ((const float*) C) + this_warp * 128,
// [cinference|plain]                     ((float*) C) + this_warp * 128,
// [cinference|plain]                     svh + (this_warp * 128) % size_n,
// [cinference|plain]                     0.088388347648f  // 1/sqrt(128)
// [cinference|plain]                 );
// [cinference|plain]             else
// [cinference|plain]                 had_hf_r_128_inner<false, true>
// [cinference|plain]                 (
// [cinference|plain]                     ((const half*) C) + this_warp * 128,
// [cinference|plain]                     ((half*) C) + this_warp * 128,
// [cinference|plain]                     svh + (this_warp * 128) % size_n,
// [cinference|plain]                     0.088388347648f  // 1/sqrt(128)
// [cinference|plain]                 );
// [cinference|plain]         }
    }
}

} // namespace exl3
} // namespace ninfer
