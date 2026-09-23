// exl3_dispatch.cu
// Host-side EXL3 linear op. See exl3_dispatch.h for the contract.
// Thin layer over the launcher (exl3_launcher.cu): builds the plan from
// the device geometry (cc/num_sms, cached) and runs the one-shot decode.
#include "exl3_dispatch.h"

namespace ninfer {
namespace exl3 {
namespace {
struct DevGeom { int cc = -1; int num_sms = 0; bool valid = false; };
DevGeom g_cached{};
DevGeom dev_geom()
{
    if (g_cached.valid) return g_cached;
    int devid = 0;
    if (cudaGetDevice(&devid) != cudaSuccess) return g_cached;
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, devid) != cudaSuccess) return g_cached;
    g_cached.cc      = prop.major * 10 + prop.minor;  // RTX 5090 -> 120
    g_cached.num_sms = (int) prop.multiProcessorCount;
    g_cached.valid   = true;
    return g_cached;
}
} // namespace

bool exl3_build_plan(Exl3Weight& w, int size_m, int cc, int num_sms)
{
    // 1.5.1 host contract: references forbid mcg and mul1 both.
    if (w.has_mcg && w.has_mul1) return false;
    // The cb2/b1/cb0 instantiation matrix (exl3_launch.cu) covers bits {3,4}
    // only. Bits {2, 5, 6} are rejected here (the launcher defender
    // also returns false): any bits>4 is K>=5 (o_proj L63, lm_head) and
    // MUST go to the reconstruct/dequant path, per static_assert in
    // 1.5.1 gemv.
    if (w.bits < 3 || w.bits > 4) return false;
    Exl3GemvPlan p{};
    if (!exl3_gemv_try_plan(p, size_m, w.k, w.n, w.bits, w.has_mcg, w.has_mul1,
                            /*c_fp32=*/false, cc, num_sms))
        return false;
    w.plan = p;
    w.ok   = true;
    return true;
}

bool exl3_gemv_eligible(const Exl3Weight& w)
{
    // Shape-agnostic (m and 128-multiple not required) embed-level check:
    // is the K of this group gemv-eligible in the first place?
    return w.bits >= 3 && w.bits <= 4;
}

std::size_t exl3_linear_workspace_bytes(int size_m, int size_k, int size_n)
{
    // A_h (m*k fp16) + C16 (m*n fp16)
    return (std::size_t) size_m * size_k * 2 + (std::size_t) size_m * size_n * 2;
}

bool exl3_plan_for(Exl3Weight& w, int size_m)
{
    DevGeom g = dev_geom();
    if (!g.valid) return false;
    return exl3_build_plan(w, size_m, g.cc, g.num_sms);
}

bool exl3_linear_op(const half* A, uint16_t* outbf, half* scratch, const Exl3Weight& w,
                    cudaStream_t stream)
{
    if (!w.ok || !scratch) return false;
    const Exl3GemvPlan& p = w.plan;
    if (p.m < 1 || p.m > 8) return false;
    half* A_h = scratch;                        // [m*k]
    half* C16 = scratch + (std::size_t) p.m * p.k;  // [m*n]
    return exl3_linear_decode(A, w.trellis, A_h, C16, outbf, p, w.suh, w.svh, stream);
}

} // namespace exl3
} // namespace ninfer