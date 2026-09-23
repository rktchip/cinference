// exl3_op.h
// QType::EXL3 engine dispatch: routes one EXL3 linear into the verified path
// for its (groups, m) shape. Lives in ninfer::ops::detail beside the other
// qtype dispatches; called from dispatch_linear (linear.cpp).
//
// Binder contract (populated at weight materialization, Linux-side):
//   w.qtype   = QType::EXL3
//   w.payload -> const ninfer::exl3::Exl3Weight side-car, with DEVICE trellis
//                [k/16, n/16, 16K] int16, suh ([k] dense, [groups,k] fused),
//                svh [n], bits (K 3/4), has_mcg/has_mul1, ok, groups/group_n.
//   w.n / w.k mirror the side-car geometry. x/out are BF16 (checked by
//   linear()); T = x.ne[1] is the batch (m).
//
// Routing (all shapes verified on-device; see README gates P12c/P16/MG):
//   groups > 1      -> v3 fused multi row (one launch, any m; m=1 inherits
//                      the known m=1 quirk, <=0.25 abs, documented).
//   groups == 1:
//     m == 1, K3/K4  -> legacy 1.5.1-verbatim gemv (P12c envelope, maxAbs 0.5;
//                      x cast BF16->FP16 into scratch first).
//     m == 1, K5/K6  -> v3 fused single row (gemv template caps at K4;
//                      measured quirk <=0.04 on real K5/K6, finite).
//     m >= 2        -> v3 fused single row (bit-exact at m>=2 on K3/K4;
//                      <=0.04 on real K5/K6).
// K outside {3,4,5,6}, non-128 k/n, or !ok throws rather than serving NaN.
//
// Workspace (caller arena, sized by exl3_linear_workspace_capacity_bytes):
//   v3 single/multi: ahad [groups*m, k] fp16 + acc [m, n] fp32 (acc memset
//     every call the kernel splits; ~us, keeps the zero invariant trivially).
//   legacy m=1:      x_fp16 [m, k] + A_h [m, k] + C16 [m, n] (all fp16).
#pragma once

#include "core/arena.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

void exl3_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                   WorkspaceArena* workspace, cudaStream_t stream);

std::size_t exl3_linear_workspace_capacity_bytes(std::int32_t output_rows,
                                                 std::int32_t input_rows, LinearPolicy policy,
                                                 std::int32_t min_tokens,
                                                 std::int32_t max_tokens);

} // namespace ninfer::ops::detail
