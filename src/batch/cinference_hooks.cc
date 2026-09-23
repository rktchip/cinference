#include "batch/cinference_hooks.h"

#include <stdexcept>

namespace ninfer::batch {

EngineHooks::EngineHooks(RequestScheduler* scheduler) : scheduler_(scheduler) {
    if (scheduler_ == nullptr) { throw std::invalid_argument("EngineHooks needs a scheduler"); }
}

std::uint64_t EngineHooks::on_new_request(Request request) {
    return scheduler_->submit(std::move(request));
}

StepPlan EngineHooks::schedule_step() {
    return scheduler_->schedule_step();
}

std::uint32_t EngineHooks::step_m(const StepPlan& plan) noexcept {
    return plan.prefill_tokens + static_cast<std::uint32_t>(plan.decode_seq_ids.size());
}

void EngineHooks::validate_step_width(const StepPlan& plan) {
    const std::size_t width = plan.prefill.size() + plan.decode_seq_ids.size();
    if (width > kMaximumConcurrency) {
        throw std::length_error("EngineHooks::dispatch_step: plan exceeds lane capacity; "
                                "keep max_running_seqs <= kMaximumConcurrency");
    }
}

StepDispatch EngineHooks::dispatch_step(const StepPlan& plan, bool need_host_tables) const {
    validate_step_width(plan);
    StepDispatch dispatch;
    dispatch.batch            = AssembleBatch(*scheduler_, plan, need_host_tables);
    dispatch.m                = step_m(plan);
    dispatch.decode_signature = plan.decode_signature();
    // Pin M to the real ragged count: the single forward consumes exactly the
    // assembled tokens, never a padded maximum.
    if (dispatch.batch.tokens.size() != dispatch.m) {
        throw std::logic_error("EngineHooks::dispatch_step: ragged M disagrees with plan");
    }
    if (dispatch.batch.num_seqs() != plan.prefill.size() + plan.decode_seq_ids.size()) {
        throw std::logic_error("EngineHooks::dispatch_step: ragged rows disagree with plan");
    }
    dispatch.route = graph_route(plan, captured_decode_signature(), has_graph_capture());
    return dispatch;
}

std::vector<std::uint64_t> EngineHooks::on_step_done(
    const StepPlan& plan, const std::vector<std::pair<std::uint64_t, TokenId>>& decoded) {
    return scheduler_->on_step_done(plan, decoded);
}

std::vector<LaneAssignment> EngineHooks::pack_lanes(const StepPlan& plan) const {
    validate_step_width(plan);
    std::vector<LaneAssignment> lanes;
    lanes.reserve(plan.decode_seq_ids.size() + plan.prefill.size());
    std::uint32_t lane = 0;
    const auto claim   = [&](std::uint64_t seq_id, bool is_prefill, std::uint32_t tokens) {
        if (lane >= kMaximumConcurrency) {
            throw std::length_error("EngineHooks::pack_lanes: plan exceeds lane capacity; "
                                    "keep max_running_seqs <= kMaximumConcurrency");
        }
        // S1: every entry stages in the single forward. No slice is deferred
        // to a later round; the upstream single staged-prefill lane is gone.
        lanes.push_back(LaneAssignment{.lane            = lane++,
                                       .seq_id         = seq_id,
                                       .is_prefill     = is_prefill,
                                       .tokens         = tokens,
                                       .staged_upstream = true});
    };
    for (const auto& slice : plan.prefill) { claim(slice.seq_id, true, slice.count); }
    for (std::uint64_t seq_id : plan.decode_seq_ids) { claim(seq_id, false, 1); }
    return lanes;
}

GraphRoute EngineHooks::graph_route(const StepPlan& plan) const noexcept {
    return IsGraphSafe(plan) ? GraphRoute::GraphReplay : GraphRoute::Eager;
}

GraphRoute EngineHooks::graph_route(const StepPlan& plan, std::uint64_t captured_signature,
                                    bool has_capture) const noexcept {
    if (!has_capture) { return GraphRoute::Eager; }
    if (!IsGraphSafe(plan)) { return GraphRoute::Eager; }
    return plan.decode_signature() == captured_signature ? GraphRoute::GraphReplay
                                                        : GraphRoute::Eager;
}

void EngineHooks::note_graph_capture(std::uint64_t decode_signature) noexcept {
    graph_capture_ = decode_signature;
}

void EngineHooks::clear_graph_capture() noexcept {
    graph_capture_.reset();
}

} // namespace ninfer::batch
