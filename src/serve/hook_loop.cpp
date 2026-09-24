// S2 serve-side binding of the S1 hook contract. See hook_loop.h for the
// loop order and the (a)-(e) responsibility map. Host-compilable: no CUDA
// kernel syntax, device calls, or port evidence in this TU.

#include "serve/hook_loop.h"

#include "batch/batch.h"
#include "batch/cinference_hooks.h"
#include "batch/paged_kv.h"
#include "batch/request.h"
#include "batch/scheduler.h"
#include "models/qwen3_5/execution/mtp_spec_gate.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::serve {
namespace {

// Logical-block divisor. Must stay identical to the scheduler pool page.
inline constexpr std::uint32_t kLogicalBlockTokens = 16;

static_assert(kLogicalBlockTokens == static_cast<std::uint32_t>(ninfer::batch::kBatchPageTokens),
              "hook loop block math must match PagedKvPool pages");

[[nodiscard]] std::uint32_t ceil_blocks(std::uint32_t tokens) noexcept {
    return (tokens + kLogicalBlockTokens - 1) / kLogicalBlockTokens;
}

} // namespace

HookLoopConfig make_hook_loop_config(std::uint32_t max_concurrency,
                                     std::uint32_t max_context_tokens,
                                     std::uint32_t kv_capacity_tokens,
                                     std::uint32_t prefill_chunk_tokens) {
    HookLoopConfig config;
    // One forward carries at most kMaximumConcurrency rows; clamp the running
    // cap so dispatch_step never throws length_error on width.
    config.max_running_seqs =
        std::clamp(max_concurrency, 1U, ninfer::kMaximumConcurrency);
    config.max_blocks_per_seq =
        std::max<std::uint32_t>(1, ceil_blocks(std::max<std::uint32_t>(1, max_context_tokens)));
    // Minimum viable pool: every running sequence able to hold max context.
    const std::uint32_t floor_pages = config.max_running_seqs * config.max_blocks_per_seq;
    const std::uint32_t kv_pages =
        kv_capacity_tokens == 0 ? 0 : ceil_blocks(kv_capacity_tokens);
    config.total_pages  = std::max<std::uint32_t>(1, std::max(floor_pages, kv_pages));
    config.chunk_tokens = prefill_chunk_tokens == 0 ? 1024 : prefill_chunk_tokens;
    return config;
}

ServeHookLoop::ServeHookLoop(HookLoopConfig config) : config_(config) {
    if (config_.max_running_seqs == 0 || config_.max_running_seqs > ninfer::kMaximumConcurrency) {
        throw std::invalid_argument("ServeHookLoop: max_running_seqs exceeds one-forward width");
    }
    if (config_.total_pages == 0 || config_.max_blocks_per_seq == 0 || config_.chunk_tokens == 0) {
        throw std::invalid_argument("ServeHookLoop: config must be nonzero; use make_hook_loop_config");
    }
    scheduler_ =
        std::make_unique<batch::RequestScheduler>(config_.max_running_seqs, config_.total_pages,
                                                 config_.max_blocks_per_seq, config_.chunk_tokens);
    hooks_ = std::make_unique<batch::EngineHooks>(scheduler_.get());
    // One step carries at most a full prefill slice plus one token per decode
    // row, so chunk_tokens + max_running_seqs bounds every upload.
    device_buffers_ = std::make_unique<batch::BatchDeviceBuffers>(
        config_.max_running_seqs, config_.max_blocks_per_seq,
        config_.chunk_tokens + config_.max_running_seqs);
}

ServeHookLoop::~ServeHookLoop() = default;

// A2: the dead enqueue entry is deleted. Admission enters through the Engine
// enqueue path; run() pumps schedule_step/dispatch_step/on_step_done itself.

batch::StepPlan ServeHookLoop::schedule_step() {
    return hooks_->schedule_step();
}

batch::StepDispatch ServeHookLoop::dispatch_step(const batch::StepPlan& plan,
                                                bool need_host_tables) const {
    // S1: one StepPlan -> one RaggedBatch -> one forward. M is pinned to the
    // real assembled token count inside dispatch_step; width beyond
    // kMaximumConcurrency throws length_error (size the loop with
    // make_hook_loop_config so this never fires on a live server).
    return hooks_->dispatch_step(plan, need_host_tables);
}

std::vector<std::uint64_t> ServeHookLoop::on_step_done(
    const batch::StepPlan& plan, const std::vector<std::pair<std::uint64_t, TokenId>>& decoded) {
    return hooks_->on_step_done(plan, decoded);
}

// F: S7 predicate glued into the hook loop. Header-only template over the S1
// StepPlan phase mix, so this TU takes no link dependency on the CUDA
// speculative TU and imports no draft backend: decode-only steps resolve to
// the MTP-3 window (3), every other mix (mixed prefill+decode, prefill-only,
// empty) resolves to 0, which routes to ordinary target-only decode.
std::uint32_t
ServeHookLoop::mtp_draft_window(const batch::StepPlan& plan) const noexcept {
    using ninfer::models::qwen3_5::execution::MtpSpecDraftWindow;
    using ninfer::models::qwen3_5::execution::SelectMtpSpeculationForStep;
    return MtpSpecDraftWindow(SelectMtpSpeculationForStep(plan));
}

bool ServeHookLoop::should_run_mtp_speculation(const batch::StepPlan& plan) const noexcept {
    using ninfer::models::qwen3_5::execution::MtpSpeculationRoute;
    using ninfer::models::qwen3_5::execution::SelectMtpSpeculationForStep;
    return SelectMtpSpeculationForStep(plan) == MtpSpeculationRoute::RunDraftVerify;
}

bool ServeHookLoop::pages_exhausted() const noexcept {
    return scheduler_->pool().free_pages() == 0;
}

void ServeHookLoop::throw_if_pages_exhausted() const {
    // Page exhaustion is admission pressure, not a fatal error: report it as
    // Overloaded (HTTP 429 via request_error_to_api_error) so the caller can
    // retry. This function never ends the process.
    if (pages_exhausted()) {
        throw ninfer::RequestError(ninfer::RequestErrorKind::Overloaded,
                                   "inference request queue is full: no KV pages free");
    }
}

void ServeHookLoop::publish_text_deltas(
    const std::vector<std::pair<std::uint64_t, std::string>>& deltas, ninfer::OutputSink* sink,
    ninfer::OutputChannel channel) {
    if (sink == nullptr) { return; }
    // Per-seq fan-out with no accumulation: each decoded pair publishes
    // exactly one delta and is then discarded. (seq_id selects the owning
    // stream when one step carries several sequences; the single-request
    // serve sink owns exactly one.)
    for (const auto& [seq_id, text] : deltas) {
        (void)seq_id;
        if (text.empty()) { continue; }
        ninfer::OutputDelta delta;
        delta.channel = channel;
        delta.text    = text;
        sink->publish(std::move(delta));
    }
}

void ServeHookLoop::ensure_warmed_once(const std::function<void()>& warm) {
    std::lock_guard lock(warmup_mutex_);
    if (warmed_) { return; }
    if (warm) { warm(); }
    warmed_ = true;
}

std::uint64_t ServeHookLoop::submit_inbox(batch::Request request,
                                          std::shared_ptr<ServeRequestState> state) {
    if (state == nullptr) { throw std::invalid_argument("ServeHookLoop inbox needs a state"); }
    if (request.seqs.size() != 1) {
        throw std::invalid_argument("ServeHookLoop inbox takes single-sequence requests");
    }
    const std::uint64_t req_id = next_id_.fetch_add(1, std::memory_order_relaxed);
    const std::uint64_t seq_id = next_id_.fetch_add(1, std::memory_order_relaxed);
    request.req_id       = req_id;
    request.seqs[0].seq_id = seq_id;
    {
        std::lock_guard lock(state->mutex);
        state->req_id = req_id;
        state->seq_id = seq_id;
        state->max_new_tokens = request.max_new_tokens;
        state->stop_token_ids = request.stop_token_ids;
    }
    {
        std::lock_guard lock(inbox_mutex_);
        inbox_.push_back(InboxItem{.request = std::move(request), .state = std::move(state)});
    }
    return req_id;
}

void ServeHookLoop::drain_inbox_to_scheduler() {
    std::vector<InboxItem> pending;
    {
        std::lock_guard lock(inbox_mutex_);
        pending.swap(inbox_);
    }
    for (auto& item : pending) {
        // Register stream state BEFORE admission so no decoded pair from the
        // very first step can miss its state.
        seq_states_[item.state->seq_id] = item.state;
        (void)hooks_->on_new_request(std::move(item.request));
    }
}

std::shared_ptr<ServeRequestState> ServeHookLoop::find_state_by_seq(std::uint64_t seq_id) {
    const auto it = seq_states_.find(seq_id);
    return it != seq_states_.end() ? it->second : nullptr;
}

void ServeHookLoop::erase_states_for_reqs(const std::vector<std::uint64_t>& finished_reqs) {
    for (std::uint64_t req_id : finished_reqs) {
        for (auto it = seq_states_.begin(); it != seq_states_.end();) {
            if (it->second->req_id == req_id) {
                it = seq_states_.erase(it);
            } else {
                ++it;
            }
        }
    }
}

} // namespace ninfer::serve
