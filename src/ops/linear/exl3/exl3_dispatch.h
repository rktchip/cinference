// exl3_dispatch.h
// Host-side EXL3 linear op for cinference: the `dispatch_linear`
// QType::EXL3 case (src/ops/linear/linear.cpp) routes into
// detail::exl3_dispatch (src/ops/linear/exl3/exl3_op.cu), which calls
// exl3_linear_op below for the m==1 K3/K4 decode path. It mirrors the
// 1.5.1 host contract quant/exl3_gemv.cu:186 but is raw-pointer (no
// libtorch) and graph-safe (all plain launches).
//
// The EXL3 pack for one linear is carried in a side-car `Exl3Weight` so the
// core `Weight` struct (1-payload + 1-scale pointer model) is left untouched
// (surgical-diffs rule). The artifact binder populates this side-car at
// weight materialization time (exl3_build_sidecar, exl3_bind.cu) and the
// call site threads it into exl3_linear_op. On the device side it owns the
// A_h (m*k fp16) and C16 (m*n fp16) scratch -- sized via
// exl3_linear_workspace_bytes.
#pragma once

#include "exl3_launcher.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <string>

namespace ninfer {
namespace exl3 {

// Side-car for one EXL3 linear. Pointers are DEVICE addresses (device D2D
// residency is a Linux-side concern; on this box only the host plan +
// plan-tuple is verified). k/n/bit/CB mirror the group layout.
struct Exl3Weight {
    const uint16_t* trellis = nullptr;  // B: int16 [k/16, n/16, 16K]
    const half* suh         = nullptr;  // k-side Hadamard pre-scale [k]
    const half* svh         = nullptr;  // n-side Hadamard post-scale [n]
    std::int32_t k          = 0;
    std::int32_t n          = 0;
    std::int32_t bits       = 0;   // K (3/4/5/6); gemv only for K in {3,4}
    bool has_mcg            = false;
    bool has_mul1           = false;
    std::int32_t mul1_multiplier = 0;
    bool ok                 = false; // plan built + gemv-eligible
    Exl3GemvPlan plan       = {};
    // Fused layers (qkv/gate_up pattern): groups shards along n, group_n[g]
    // widths sum to n. Binder-populated; 1 = dense single. Only the v3 row
    // serves groups > 1 (one launch over the whole fused width).
    std::int32_t groups     = 1;
    std::int32_t group_n[8] = {};
};

// Build the plan + cb for one EXL3 linear. cc/num_sms are read from the
// device (no DevCtx singleton — cached by exl3_linear_op). Returns false
// when the tensor is not gemv-eligible (K outside {3,4} or m>8 or k/n not
// 128-multiple); the engine dispatch then routes the v3 row (K3..K6).
bool exl3_build_plan(Exl3Weight& w, int size_m, int cc, int num_sms);

// Scratch bytes for the EXL3 small-m decode path: A_h (m*k fp16) + C16
// (m*n fp16). The workspace sizing of the EXL3 linear op comes from
// `size_m` here.
std::size_t exl3_linear_workspace_bytes(int size_m, int size_k, int size_n);

// Build the plan for a given batch and return the scratch needed. Pure host;
// usable at pre-run plan time to reserve workspace without touching the GPU.
bool exl3_plan_for(Exl3Weight& w, int size_m);

// True when the EXL3 group is gemv-eligible (K in {3,4} AND m<=8 AND k/n are
// 128-multiples). K5/K6 (o_proj L63, lm_head) are false here and serve via
// the v3 fused row (Gate K5K6); the legacy op below returns false for them.
bool exl3_gemv_eligible(const Exl3Weight& w);

// The linear op. A: device [m, k] fp16 activations (contiguous); out: device
// [m, n] bf16 result. scratch: A_h + C16 fp16 (>= exl3_linear_workspace_bytes),
// or nullptr only if the caller will not provide one (this path requires it).
// Returns false (and leaves out untouched) when w is not gemv-eligible — the
// caller then uses the reconstruct path.
bool exl3_linear_op(const half* A, uint16_t* outbf, half* scratch, const Exl3Weight& w,
                    cudaStream_t stream);

} // namespace exl3
} // namespace ninfer
