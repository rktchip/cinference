#pragma once
// S1b runtime execution lane: one mixed StepPlan -> one forward, ragged M.
//
// Construction (why M is ragged and dispatch is exactly-once per layer):
//   1. EngineHooks::dispatch_step pins dispatch.m = step_m(plan) =
//      plan.prefill_tokens + plan.decode_seq_ids.size() and throws
//      std::logic_error unless dispatch.batch.tokens.size() == dispatch.m
//      (src/batch/cinference_hooks.cc). M is the assembled token count,
//      never a padded maximum.
//   2. step_activation_rows() reads the row count from exactly one place,
//      dispatch.batch.tokens.size(), and throws unless it equals
//      dispatch.m. Callers set the activation Tensor with
//      x.ne[1] = step_activation_rows(dispatch), so activation rows equal
//      ragged.total_tokens by construction.
//   3. run_step_layers() owns the per-layer loop. Its body holds exactly
//      one linear call site per iteration, so each layer issues exactly
//      one exl3_dispatch per step. dispatch_linear routes QType::EXL3 to
//      detail::exl3_dispatch (src/ops/linear/linear.cpp); no second linear
//      entry exists in this lane.
//
// Slot-B forward driver (ANY plan mix, ONE forward):
//   run_step_forward() consumes one StepDispatch and drives the pinned path:
//   M check -> embed (x.ne[1] == M) -> per layer one attn_ragged + one
//   exl3_dispatch at m = tokens.size() -> slot-C sample_decode_rows for the
//   decoded pairs. Prefill-only (n_decode == 0), mixed, and decode-only
//   (prefill empty) all run the same body: M = prefill_tokens + n_decode
//   covers every mix, so there is no per-shape branch and no [W,B] reshape
//   (ragged rows keep their variable spans via seq_offsets). An empty plan
//   (M == 0) throws and never runs. set_prefill_lane is never called on
//   this path; the legacy single lane stays behind the legacy_single_lane_
//   flag in engine_core.h.
//
// Device binding (Linux CUDA path; GPU execution GATED on this Windows
// host -- no device call below executes here, and this header takes no
// CUDA types so cl and nvcc -c both pass):
//   upload: device_buffers.upload(dispatch.batch, pool, stream), once per
//           step before the forward (need_host_tables=false: the on-device
//           gather path; the host compact matrix stays empty).
//   embed:  Tensor x = embed_lookup(dispatch.batch.tokens) with
//           x.ne[1] = step_activation_rows(dispatch) == tokens.size().
//   layer:  causal_softmax_attention_ragged(..., device_buffers.view(), ...)
//           then detail::exl3_dispatch(x, w, out, policy, workspace, stream)
//           once per layer with m = tokens.size() (T = x.ne[1]).
//   route:  dispatch.route (pinned by dispatch_step via the signature-gated
//           graph_route: replay only when IsGraphSafe(plan) and the plan's
//           decode_signature() matches the capture) selects graph replay vs
//           eager launch. Either way the single forward above is unchanged;
//           the lane may stay eager.
//   sample: slot-C TextContext::sample_decode_rows(hidden, plan,
//           dispatch.batch, stream) -- declared in
//           models/qwen3_5/execution/text.h, defined in text.cpp, owned by
//           slot C. hidden is the post-final-norm flat-T activation [H,T].
//           B declares no definition and implements no sampling.
//           Prefill-only steps return {} (n_dec == 0); the pairs flow out
//           unchanged as this driver's return value, straight into
//           on_step_done(plan, decoded).
//
// Host-compilable: no CUDA kernel syntax, no device calls. The embed,
// attention, linear, and sample entries are caller-provided callables, so
// host tests count launches without touching the GPU.
#include "batch/cinference_hooks.h"

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::runtime {

// Activation row count for one dispatched step. Throws std::logic_error
// unless the batch payload agrees with the pinned M.
[[nodiscard]] inline std::uint32_t
step_activation_rows(const batch::StepDispatch& dispatch) {
    if (dispatch.m != dispatch.batch.tokens.size()) {
        throw std::logic_error(
            "step_forward: ragged M disagrees with batch tokens");
    }
    return dispatch.m;
}

// Mixed-step pin: M must equal prefill_tokens + one row per decode seq.
// Throws std::logic_error otherwise (a stale plan must never run).
inline void validate_mixed_step_m(const batch::StepPlan& plan,
                                  const batch::StepDispatch& dispatch) {
    const std::uint32_t want =
        plan.prefill_tokens +
        static_cast<std::uint32_t>(plan.decode_seq_ids.size());
    if (dispatch.m != want) {
        throw std::logic_error("step_forward: M is not prefill + n_decode");
    }
    if (dispatch.batch.tokens.size() != dispatch.m) {
        throw std::logic_error(
            "step_forward: ragged M disagrees with batch tokens");
    }
}

// One step, N layers, exactly one linear call per layer. M is fixed once
// before the loop; the loop body has a single call site. Returns the call
// count, which equals num_layers by construction.
template <class LinearFn>
[[nodiscard]] inline std::uint32_t
run_step_layers(const batch::StepDispatch& dispatch, std::uint32_t num_layers,
                LinearFn&& linear) {
    const std::uint32_t m = step_activation_rows(dispatch);
    for (std::uint32_t layer = 0; layer < num_layers; ++layer) {
        linear(layer, m);
    }
    return num_layers;
}

// Decoded pairs for one step: (seq_id, token), one entry per
// plan.decode_seq_ids element in plan order; {} when the plan carries no
// decode rows (prefill-only). This is exactly the vector on_step_done
// consumes and slot-C sample_decode_rows produces, so returning it
// unchanged is how slots A (plan/batch), B (forward), and C (sampling)
// meet the contract.
using StepDecodedPairs = std::vector<std::pair<std::uint64_t, TokenId>>;

// One step, N layers, one forward for ANY plan mix. Hook contracts:
//   embed  -- nullary, returns the installed activation row count (host
//             mirror of x.ne[1]); must equal the ragged M.
//   attn   -- (layer, m), one attention launch per call (production:
//             causal_softmax_attention_ragged over device_buffers.view()).
//   linear -- (layer, m), one linear launch per call (production:
//             detail::exl3_dispatch at QType::EXL3, m = tokens.size()).
//   sample -- nullary, returns StepDecodedPairs (production binds slot-C
//             TextContext::sample_decode_rows(hidden, plan, batch, stream)
//             at the call site; the CUDA types stay there, not here).
// Returns sample() unchanged. Throws std::logic_error on a stale or empty
// plan or a short embed, std::invalid_argument on num_layers == 0. The
// per-layer exactly-once structure is inherited from run_step_layers
// (single call site per iteration).
template <class EmbedFn, class AttnFn, class LinearFn, class SampleFn>
[[nodiscard]] inline StepDecodedPairs
run_step_forward(const batch::StepPlan& plan, const batch::StepDispatch& dispatch,
                 std::uint32_t num_layers, EmbedFn&& embed, AttnFn&& attn,
                 LinearFn&& linear, SampleFn&& sample) {
    // ANY mix, one forward: M == prefill_tokens + n_decode pins prefill-only
    // (n_decode == 0), mixed, and decode-only (prefill empty) alike; a stale
    // plan throws here and never runs.
    validate_mixed_step_m(plan, dispatch);
    const std::uint32_t m = step_activation_rows(dispatch);
    if (m == 0) {
        throw std::logic_error("step_forward: empty plan never runs a forward");
    }
    if (num_layers == 0) {
        throw std::invalid_argument("step_forward: num_layers must be positive");
    }
    // Embed stage pin: x.ne[1] == tokens.size() by construction.
    const std::uint32_t embed_rows = static_cast<std::uint32_t>(embed());
    if (embed_rows != m) {
        throw std::logic_error("step_forward: activation rows disagree with ragged M");
    }
    // Per-layer body: exactly one attention launch then exactly one linear
    // launch per layer, both at m = tokens.size(). (void) keeps the
    // [[nodiscard]] count without restating the proof.
    (void)run_step_layers(dispatch, num_layers,
                          [&](std::uint32_t layer, std::uint32_t lm) {
                              attn(layer, lm);
                              linear(layer, lm);
                          });
    // Decoded pairs flow out unchanged: slot-C sample_decode_rows owns their
    // production; the caller feeds this return value straight into
    // on_step_done(plan, decoded).
    return sample();
}

} // namespace ninfer::runtime
