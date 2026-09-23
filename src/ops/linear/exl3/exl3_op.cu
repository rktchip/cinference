// exl3_op.cu
// QType::EXL3 engine dispatch (see exl3_op.h for the contract). Host routing
// + one exact bf16->fp16 cast kernel for the legacy m=1 gemv path.
#include "ops/linear/exl3/exl3_op.h"
#include "ops/linear/exl3/exl3_aln.h"
#include "ops/linear/exl3/exl3_dispatch.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// Exact while |v| <= 65504 (fp16 max): fp16's mantissa is a superset of
// bf16's, so every in-range bf16 value round-trips bit-exact. Activations
// live orders of magnitude below the bound; the gate covers ramp tiles.
__global__ void bf16_to_fp16_kernel(const __nv_bfloat16* __restrict__ in,
                                    __half* __restrict__ out, long long n)
{
    long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long) blockDim.x * gridDim.x;
    for (; i < n; i += stride) out[i] = __float2half_rn(__bfloat162float(in[i]));
}

void cast_bf16_to_fp16(const __nv_bfloat16* in, __half* out, long long n,
                       cudaStream_t stream)
{
    const int threads = 256;
    long long blocks = (n + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    bf16_to_fp16_kernel<<<(unsigned) blocks, threads, 0, stream>>>(in, out, n);
}

const exl3::Exl3Weight& exl3_sidecar(const Weight& w)
{
    if (w.qtype != QType::EXL3)
        throw std::invalid_argument("exl3 linear: qtype is not EXL3");
    const auto* ew = static_cast<const exl3::Exl3Weight*>(w.payload);
    if (ew == nullptr || !ew->ok)
        throw std::invalid_argument("exl3 linear: missing/unbuilt side-car");
    if (ew->bits < 3 || ew->bits > 6)
        throw std::invalid_argument("exl3 linear: K outside {3,4,5,6}");
    if (ew->k % 128 != 0 || ew->n % 128 != 0)
        throw std::invalid_argument("exl3 linear: k/n must be 128-multiples");
    if (ew->groups < 1 || ew->groups > 8)
        throw std::invalid_argument("exl3 linear: groups must be 1..8");
    return *ew;
}

} // namespace

std::size_t exl3_linear_workspace_capacity_bytes(std::int32_t output_rows,
                                                 std::int32_t input_rows, LinearPolicy policy,
                                                 std::int32_t min_tokens,
                                                 std::int32_t max_tokens)
{
    (void) policy;
    if (min_tokens <= 0 || max_tokens < min_tokens)
        throw std::invalid_argument("exl3 linear workspace: invalid token interval");
    if (output_rows <= 0 || input_rows <= 0)
        throw std::invalid_argument("exl3 linear workspace: bad geometry");
    // groups/bits are binder-side, so size the v3 row for the worst servable
    // case (8 groups, K=4). The legacy m=1 path adds its exact footprint.
    exl3::Exl3AlnParams p{};
    p.k = input_rows;
    p.n = output_rows;
    p.bits = 6; // worst servable K (K5/K6 tiles are wider than K4)
    p.cb = 2;
    bool need_acc = false;
    const std::size_t v3 = exl3::exl3_aln_ws_bytes(p, /*groups=*/8, max_tokens, &need_acc);
    const std::size_t legacy = 3ULL * (std::size_t) input_rows * 2      // x_fp16 + A_h
                             + (std::size_t) output_rows * 2;           // C16 (m=1 only)
    return v3 + legacy;
}

void exl3_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                   WorkspaceArena* workspace, cudaStream_t stream)
{
    (void) policy;
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16)
        throw std::invalid_argument("exl3 linear: x/out must be BF16");
    const std::int32_t m = x.ne[1];
    if (m <= 0 || x.ne[0] != w.k || out.ne[0] != w.n || out.ne[1] != m)
        throw std::invalid_argument("exl3 linear: shape mismatch");
    const exl3::Exl3Weight& ew = exl3_sidecar(w);
    if (ew.k != w.k || ew.n != w.n)
        throw std::invalid_argument("exl3 linear: side-car/core geometry mismatch");
    if (workspace == nullptr)
        throw std::invalid_argument("exl3 linear requires caller workspace");
    auto scope = workspace->scope();

    exl3::Exl3AlnParams p{};
    p.w16 = ew.trellis;
    p.suh = ew.suh;
    p.svh = ew.svh;
    p.k = ew.k;
    p.n = ew.n;
    p.bits = ew.bits;
    p.cb = (ew.has_mul1 ? 2 : (ew.has_mcg ? 1 : 0));

    if (ew.groups > 1)
    {
        // Fused layer: one launch over the whole width (any m).
        bool need_acc = false;
        const std::size_t ws = exl3::exl3_aln_ws_bytes(p, ew.groups, m, &need_acc);
        const DeviceSpan span = workspace->alloc_bytes(ws, 256);
        auto* base = static_cast<char*>(span.data);
        auto* ahad = reinterpret_cast<__half*>(base);
        float* acc = need_acc ? reinterpret_cast<float*>(base + (std::size_t) ew.groups * m * p.k * 2) : nullptr;
        if (acc) cudaMemsetAsync(acc, 0, (std::size_t) m * p.n * 4, stream);
        if (!exl3::exl3_aln_rows_bf16(static_cast<const __nv_bfloat16*>(x.data),
                                      static_cast<__nv_bfloat16*>(out.data), p,
                                      ew.group_n, ew.groups, ahad, acc, m, stream))
            throw std::invalid_argument("exl3 linear: multi-group row failed");
        return;
    }

    if (m == 1 && ew.bits <= 4)
    {
        // Single-token decode, K3/K4: 1.5.1-verbatim gemv (P12c tolerance
        // envelope, maxAbs 0.5 vs production forward; gemv-vs-hgemm order).
        exl3::Exl3Weight planned = ew;
        if (!exl3::exl3_plan_for(planned, 1))
            throw std::invalid_argument("exl3 linear: gemv plan failed");
        const std::size_t nxf = (std::size_t) m * p.k * 2;
        const std::size_t ws = exl3::exl3_linear_workspace_bytes(m, p.k, p.n);
        const DeviceSpan span = workspace->alloc_bytes(nxf + ws, 256);
        auto* base = static_cast<char*>(span.data);
        auto* xf = reinterpret_cast<__half*>(base);
        auto* scratch = reinterpret_cast<__half*>(base + nxf);
        cast_bf16_to_fp16(static_cast<const __nv_bfloat16*>(x.data), xf,
                          (long long) m * p.k, stream);
        if (!exl3::exl3_linear_op(xf, static_cast<std::uint16_t*>(out.data),
                                  scratch, planned, stream))
            throw std::invalid_argument("exl3 linear: gemv op failed");
        return;
    }

    // Prefill / batched decode (and K5/K6 single-token: the gemv template
    // caps at K4, so K5/K6 m=1 rides v3 -- measured quirk <=0.04, finite).
    {
        bool need_acc = false;
        const std::size_t ws = exl3::exl3_aln_ws_bytes(p, 1, m, &need_acc);
        const DeviceSpan span = workspace->alloc_bytes(ws, 256);
        auto* base = static_cast<char*>(span.data);
        auto* ahad = reinterpret_cast<__half*>(base);
        float* acc = need_acc ? reinterpret_cast<float*>(base + (std::size_t) m * p.k * 2) : nullptr;
        if (acc) cudaMemsetAsync(acc, 0, (std::size_t) m * p.n * 4, stream);
        if (!exl3::exl3_aln_row_bf16(static_cast<const __nv_bfloat16*>(x.data),
                                     static_cast<__nv_bfloat16*>(out.data), p,
                                     ahad, acc, m, stream))
            throw std::invalid_argument("exl3 linear: v3 single row failed");
        return;
    }
}

} // namespace ninfer::ops::detail
