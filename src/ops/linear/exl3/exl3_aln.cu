// exl3_aln.cu
// Cinference raw-pointer rows over the cuda-exl3 M-tiled fused GEMM
// (criterion-7 swap). Torch API wrappers stripped; kernels unmodified.
#include "exl3_aln.h"
#include "exl3_v3_alias.h"   // global half / bfloat16 aliases for the v3 TUs

#include <cuda_bf16.h>
#include <cuda_fp16.h>


namespace cuda_exl3 {

// v3 row entries (defined in torchexl3/exl3_gemm.cu + exl3_hadamard.cu).
void exl3_gemm_row(const half* a, const uint16_t* bq, __nv_bfloat16* c,
                   const half* svh, int m, int k, int n, int ldc, float* acc,
                   int bits, int cb, cudaStream_t stream);
void exl3_gemm_row(const half* a, const uint16_t* bq, half* c,
                   const half* svh, int m, int k, int n, int ldc, float* acc,
                   int bits, int cb, cudaStream_t stream);
int exl3_pick_split_row(int m, int k, int n, int bits);

void exl3_had_in_row(const half* x, half* ahad, const half* suh, int m, int k,
                     int groups, cudaStream_t stream);
void exl3_had_in_row_bf16(const __nv_bfloat16* x, half* ahad, const half* suh,
                          int m, int k, int groups, cudaStream_t stream);
void exl3_gemm_rows(const half* a, const uint16_t* bq, __nv_bfloat16* c,
                    const half* svh, int m, int k, int n, int ldc, float* acc,
                    int bits, int cb, const int* group_n, int groups,
                    cudaStream_t stream);
int exl3_det_planes_row(int m, int k, int n, int bits);
bool exl3_det_enabled();

} // namespace cuda_exl3

namespace ninfer {
namespace exl3 {

std::size_t exl3_aln_ws_bytes(const Exl3AlnParams& p, std::int32_t groups,
                              std::int32_t m, bool* need_acc)
{
    std::size_t ahad = (std::size_t) groups * m * p.k * 2;
    // Plane count: 0 = no split (no acc), 1 = historical single shared plane,
    // S > 1 = deterministic per-split planes (default on; =0 opts out).
    // Opted out this is exactly the old `split ? m*n*4 : 0`.
    int planes = cuda_exl3::exl3_det_planes_row(m, p.k, p.n, p.bits);
    bool split = planes > 0;
    if (need_acc)
        *need_acc = split;
    return ahad + (split ? (std::size_t) planes * m * p.n * 4ULL : 0);
}

// Exact acc footprint for a dense row call (what the caller must memset).
// Opted out (=0): m*n*4 when the shape splits, 0 otherwise -- identical to the
// historical inline expression. Default on: S*m*n*4 planes.
std::size_t exl3_aln_acc_bytes(const Exl3AlnParams& p, std::int32_t m)
{
    int planes = cuda_exl3::exl3_det_planes_row(m, p.k, p.n, p.bits);
    if (planes <= 0) return 0;
    return (std::size_t) planes * m * p.n * 4ULL;
}

// Plane count for reserve-time max-cover (see exl3_bind.cu).
int exl3_aln_det_planes(const Exl3AlnParams& p, std::int32_t m)
{
    return cuda_exl3::exl3_det_planes_row(m, p.k, p.n, p.bits);
}

// Flag query for reserve-time sizing.
bool exl3_aln_det_enabled()
{
    return cuda_exl3::exl3_det_enabled();
}

// Geometry gate: k/n in 128-multiples (kernel tiles), bits 1..8, groups 1..8
// (ShardMap capacity). Per-shard widths are checked by the multi-group row.
namespace {
bool aln_geometry(const Exl3AlnParams& p, int groups)
{
    if (p.k % 128 != 0 || p.n % 128 != 0)
        return false;
    if (p.bits < 1 || p.bits > 8)
        return false;
    if (groups < 1 || groups > 8)
        return false;
    return true;
}
} // namespace

bool exl3_aln_row_bf16(const bfloat16* x, bfloat16* C, const Exl3AlnParams& p,
                        half* ahad, float* acc, std::int32_t m,
                        cudaStream_t stream)
{
    if (!aln_geometry(p, 1))
        return false;
    cuda_exl3::exl3_had_in_row_bf16((const __nv_bfloat16*) x, ahad, p.suh, m, p.k, 1, stream);
    cuda_exl3::exl3_gemm_row(ahad, p.w16, (__nv_bfloat16*) C, p.svh, m, p.k, p.n, p.n, acc,
                  p.bits, p.cb, stream);
    return true;
}

bool exl3_aln_row_half(const half* x, half* C, const Exl3AlnParams& p,
                        half* ahad, float* acc, std::int32_t m, cudaStream_t stream)
{
    if (!aln_geometry(p, 1))
        return false;
    cuda_exl3::exl3_had_in_row(x, ahad, p.suh, m, p.k, 1, stream);
    cuda_exl3::exl3_gemm_row(ahad, p.w16, C, p.svh, m, p.k, p.n, p.n, acc, p.bits, p.cb, stream);
    return true;
}

// Multi-group row (fused qkv pattern): one had_in launch (shared x, stacked
// suh) then ONE fused GEMM launch over all shards (stacked ahad, absolute
// trellis/svh/C columns). group_n[g] widths sum to p.n, each BN-multiple.
bool exl3_aln_rows_bf16(const bfloat16* x, bfloat16* C, const Exl3AlnParams& p,
                        const int* group_n, std::int32_t groups, half* ahad,
                        float* acc, std::int32_t m, cudaStream_t stream)
{
    if (!aln_geometry(p, groups))
        return false;
    int sum = 0;
    for (int g = 0; g < groups; ++g)
    {
        if (group_n[g] % 128 != 0)
            return false;
        sum += group_n[g];
    }
    if (sum != p.n)
        return false;
    cuda_exl3::exl3_had_in_row_bf16((const __nv_bfloat16*) x, ahad, p.suh, m, p.k, groups, stream);
    cuda_exl3::exl3_gemm_rows(ahad, p.w16, (__nv_bfloat16*) C, p.svh, m, p.k, p.n, p.n, acc,
                   p.bits, p.cb, group_n, groups, stream);
    return true;
}

} // namespace exl3
} // namespace ninfer
