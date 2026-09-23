#pragma once

// S7: decode-only MTP speculation gate.
//
// Policy: a step runs the ninfer MTP-3 draft+verify transaction
// (target_verify_accept over the K+1-wide verify window followed by the MTP
// draft chain, see program/speculative/mtp.cpp) if and only if the step is
// decode-only. Any step carrying prefill work - a mixed prefill+decode step -
// SKIPS speculation and runs the ordinary target-only decode path instead.
//
// Predicate (documented here, implemented below):
//   RUN  <=>  has_decode && !has_prefill
// where has_decode  = the S1 StepPlan carries at least one decode sequence
//                     (batch::StepPlan::decode_seq_ids is non-empty), and
//       has_prefill = the S1 StepPlan carries at least one prefill slice
//                     (batch::StepPlan::prefill is non-empty, has_prefill()).
// Truth table:
//   decode-only (decode ids, no prefill slices) -> RUN  (MTP-3 draft+verify)
//   mixed (decode ids plus prefill slices)      -> SKIP (ordinary decode)
//   prefill-only (slices, no decode ids)        -> SKIP (ordinary prefill path)
//   empty (neither)                             -> SKIP (nothing to do)
//
// Rationale: prefill slices are ragged and input-dependent, while the MTP
// verify width (K+1 columns) and the autoregressive draft chain assume a
// uniform decode batch. Mixing prefill rows into the speculative transaction
// would break the width/budget accounting in mtp_prepare_next_round, so mixed
// steps take the ordinary path. This mirrors IsGraphSafe in
// batch/scheduler.h, which likewise admits only pure-decode steps to the
// captured graph.
//
// Window: S7 uses MTP-3 (kMtpSpecDecodeDrafts == 3, verify width 4). The value
// sits strictly inside the decode-frame domain
// (kMtpDecodeMaximumDrafts == 10 in program/round_buffers.h), but this gate
// never selects the 10-wide window. No DFlash/DFlash2 state participates:
// this header has no DFlash include and no masked-draft backend switch.
//
// GPU execution of the RUN arm (real draft+verify numerics) is GATED: it is
// covered by the CUDA tests in tests/ops (test_mtp_round,
// test_speculative_round) and is not exercised by the host decision test.

#include <cstdint>

namespace ninfer::batch {
// Forward declaration only: the concrete StepPlan overloads below are defined
// in program/speculative/mtp.cpp, which includes batch/scheduler.h. The host
// decision test uses the duck-typed template with the real StepPlan and needs
// no link against the batch library.
struct StepPlan;
} // namespace ninfer::batch

namespace ninfer::models::qwen3_5::execution {

// MTP-3: three draft tokens per decode row, verify width K+1 == 4.
// planning/graph_profiles.cpp carries a tuned profile branch for
// draft_window == 3.
inline constexpr std::uint32_t kMtpSpecDecodeDrafts = 3;
inline constexpr std::uint32_t kMtpSpecDecodeWidth  = kMtpSpecDecodeDrafts + 1;

// Decode-frame domain maximum (program/round_buffers.h). Named here so the S7
// window choice is visibly pinned below the 10-wide maximum.
inline constexpr std::uint32_t kMtpSpecFrameDomainDrafts = 10;
static_assert(kMtpSpecDecodeDrafts < kMtpSpecFrameDomainDrafts,
              "S7 uses MTP-3, never the 10-wide window");
static_assert(kMtpSpecDecodeWidth == 4, "MTP-3 verify width is K+1");

// Phase mix of one scheduled step, projected from the S1 StepPlan:
//   has_decode  <=> !plan.decode_seq_ids.empty()
//   has_prefill <=> plan.has_prefill()  (i.e. !plan.prefill.empty())
struct MtpStepPhaseMix {
    bool has_decode  = false;
    bool has_prefill = false;
};

enum class MtpSpeculationRoute : std::uint8_t {
    RunDraftVerify = 0, // decode-only: MTP-3 draft+verify
    SkipOrdinary   = 1, // anything else: ordinary target-only decode
};

// THE S7 PREDICATE: RUN <=> has_decode && !has_prefill.
[[nodiscard]] constexpr MtpSpeculationRoute
SelectMtpSpeculationRoute(MtpStepPhaseMix mix) noexcept {
    return (mix.has_decode && !mix.has_prefill) ? MtpSpeculationRoute::RunDraftVerify
                                                : MtpSpeculationRoute::SkipOrdinary;
}

[[nodiscard]] constexpr bool ShouldRunMtpSpeculation(MtpStepPhaseMix mix) noexcept {
    return SelectMtpSpeculationRoute(mix) == MtpSpeculationRoute::RunDraftVerify;
}

// Draft window implied by the route: MTP-3 on RUN, 0 (no speculative call) on
// SKIP. SKIP never issues mtp_decode_batch: k == 0 is outside that entry's
// domain (it throws on k == 0), so a zero window routes to ordinary decode.
[[nodiscard]] constexpr std::uint32_t MtpSpecDraftWindow(MtpSpeculationRoute route) noexcept {
    return route == MtpSpeculationRoute::RunDraftVerify ? kMtpSpecDecodeDrafts : 0;
}

// Branch directly on the S1 StepPlan phase mix: PrefillSlice presence via
// has_prefill(), decode membership via decode_seq_ids. Duck-typed so the
// device TU (with batch/scheduler.h) and host tests (with the real StepPlan,
// no batch link required) share this exact branch.
template <typename StepPlanT>
[[nodiscard]] inline MtpSpeculationRoute
SelectMtpSpeculationForStep(const StepPlanT& plan) noexcept {
    const MtpStepPhaseMix mix{!plan.decode_seq_ids.empty(), plan.has_prefill()};
    return SelectMtpSpeculationRoute(mix);
}

// Concrete S1 StepPlan overloads, defined in program/speculative/mtp.cpp next
// to the batched verify entries. Dispatchers consult MtpDraftWindowForPlan: a
// nonzero window selects the mtp_decode_batch (MTP-3 draft+verify) path, zero
// selects ordinary target-only decode.
[[nodiscard]] MtpSpeculationRoute
SelectMtpSpeculationForPlan(const batch::StepPlan& plan) noexcept;
[[nodiscard]] std::uint32_t MtpDraftWindowForPlan(const batch::StepPlan& plan) noexcept;

} // namespace ninfer::models::qwen3_5::execution
