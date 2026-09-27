#pragma once

// S2 serve-side binding of the S1 hook contract (src/batch/cinference_hooks.h).
//
// HTTP uses the new loop. This header owns the process-lifetime scheduling
// objects and the exact entry points the iteration loop drives:
//
//   on_new_request -> schedule_step -> dispatch_step (one RaggedBatch,
//   ragged M, one forward) -> on_step_done
//
// Forward consumes dispatch.batch.tokens as M rows, never padded.
// graph_route is signature-gated: pure-decode steps replay only when the
// plan's decode_signature() matches the captured signature; any
// prefill-carrying step or decode membership/order change goes eager.
//
// Serve responsibilities in this contract:
//   (a) GenerationService constructs ONE ServeHookLoop (one RequestScheduler
//       plus its EngineHooks) for the process lifetime, alongside the
//       EXL3-loaded Engine whose weights load once at process start.
//   (b) prepare() tokenizes and handles media only; run() pumps the loop
//       itself (schedule_step -> dispatch_step -> on_step_done) and fans
//       decoded tokens out per seq_id. engine enqueue stays enqueue-only.
//   (c) Admit: page exhaustion surfaces as RequestErrorKind::Overloaded
//       (HTTP 429; queue timeout is 503). Serve never aborts the process:
//       mid-sequence pool exhaustion aborts only the owning request and the
//       next schedule_step reclaims its blocks. This TU contains no
//       abort/exit/terminate path.
//   (d) Streaming: on_step_done decoded tokens fan out to OutputSink per
//       seq_id via publish_text_deltas with no full-answer buffer.
//   (e) Warmup loads EXL3 weights once at process start (Engine construction
//       plus one ensure_warmed_once gate), never per request.
//
// Two prompts, one process: prepare() mutates no scheduler state and run()
// consumes only its own single-use PreparedRequest, so two different prompts
// admitted through one GenerationService/Engine stream through the shared
// loop. The curl two-client check is tests/serve/test_two_sessions.sh
// (S5, Linux-only); this TU proves the path by construction plus host
// compile, and never points at a live port as evidence.

#include <ninfer/types.h>

#include "batch/request.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ninfer::batch {
struct Request;
struct StepPlan;
struct StepDispatch;
class RequestScheduler;
class EngineHooks;
class BatchDeviceBuffers;
} // namespace ninfer::batch

namespace ninfer::serve {

// Per-request stream state for the single-scheduler pump (Committee A+B).
// One per admitted serve request (single sequence). The pump publishes
// decoded (seq_id, token) pairs here by seq/req id; run() sets the live
// sink and drains buffered text so no token misses its stream even if a
// step runs before the owning run() registers its sink.
struct ServeRequestState {
    std::mutex mutex;
    std::uint64_t req_id            = 0;
    std::uint64_t seq_id            = 0;
    std::uint32_t max_new_tokens    = 0;
    // Step-1 EOS stop: copy of the admitted stop-token set; the publish loop
    // drops the stop token and any same-step tokens after it.
    std::vector<TokenId> stop_token_ids;
    // Set when the pump observes a stop token (scheduler finish is separate).
    bool stopped_on_token = false;
    std::vector<TokenId> generated_ids;
    std::string text;
    // Deltas produced before the live sink registered (flushed in order on
    // registration so no token misses its stream).
    std::vector<std::string> pending;
    bool finished               = false;
    // Non-owning live sink (ServiceOutputSink owned by the run() frame).
    // Null until the owning run() registers it; decoded text before that
    // is buffered in text/generated_ids and flushed on registration.
    ninfer::OutputSink* sink = nullptr;
};

// Startup-fixed sizing for one ServeHookLoop. max_running_seqs is clamped to
// ninfer::kMaximumConcurrency so every emitted plan fits one forward.
struct HookLoopConfig {
    std::uint32_t max_running_seqs   = 1;
    std::uint32_t total_pages        = 0; // startup-fixed logical blocks
    std::uint32_t max_blocks_per_seq = 0; // per-seq logical max-context
    std::uint32_t chunk_tokens       = 1024;
};

// Derive loop sizing from serve options. kv_capacity_tokens 0 (automatic
// policy) falls back to the minimum viable pool: every running sequence able
// to hold max_context_tokens. Never returns zeros.
[[nodiscard]] HookLoopConfig make_hook_loop_config(std::uint32_t max_concurrency,
                                                  std::uint32_t max_context_tokens,
                                                  std::uint32_t kv_capacity_tokens,
                                                  std::uint32_t prefill_chunk_tokens);

class ServeHookLoop {
public:
    explicit ServeHookLoop(HookLoopConfig config);
    ~ServeHookLoop();

    ServeHookLoop(const ServeHookLoop&)            = delete;
    ServeHookLoop& operator=(const ServeHookLoop&) = delete;
    ServeHookLoop(ServeHookLoop&&)                 = delete;
    ServeHookLoop& operator=(ServeHookLoop&&)      = delete;

    [[nodiscard]] const HookLoopConfig& config() const noexcept { return config_; }

    // A2: the dead enqueue entry is deleted. Admission enters through the
    // Engine enqueue path and run() pumps schedule_step below; no second
    // entry exists on this loop.

    // S1 entries driven by the Engine worker loop, in order, once per step.
    batch::StepPlan schedule_step();
    [[nodiscard]] batch::StepDispatch dispatch_step(const batch::StepPlan& plan,
                                                   bool need_host_tables = false) const;
    std::vector<std::uint64_t>
    on_step_done(const batch::StepPlan& plan,
                 const std::vector<std::pair<std::uint64_t, TokenId>>& decoded);

    // Admit probe: true when no logical page is free for a new sequence.
    // throw_if_pages_exhausted raises RequestError(Overloaded), which serve
    // renders as HTTP 429 (queue timeout stays 503). Never ends the process.
    [[nodiscard]] bool pages_exhausted() const noexcept;
    void throw_if_pages_exhausted() const;

    // Streaming fan-out: each decoded (seq_id, text) pair publishes one
    // OutputDelta to the sink immediately. No full-answer buffer is kept.
    // Empty text is skipped; a null sink is a no-op.
    static void publish_text_deltas(
        const std::vector<std::pair<std::uint64_t, std::string>>& deltas,
        ninfer::OutputSink* sink,
        ninfer::OutputChannel channel = ninfer::OutputChannel::Content);

    // Warmup gate: runs warm() at most once per process lifetime. EXL3
    // weights load at Engine construction; this covers the one warmup pass.
    void ensure_warmed_once(const std::function<void()>& warm);

    [[nodiscard]] batch::EngineHooks* hooks() noexcept { return hooks_.get(); }
    [[nodiscard]] const batch::EngineHooks* hooks() const noexcept { return hooks_.get(); }
    [[nodiscard]] batch::RequestScheduler* scheduler() noexcept { return scheduler_.get(); }
    [[nodiscard]] const batch::RequestScheduler* scheduler() const noexcept {
        return scheduler_.get();
    }
    // Startup-fixed device buffers for the one-forward runner. Serve startup
    // requires them to be device-backed (see RequireDeviceBuffersForServe).
    [[nodiscard]] batch::BatchDeviceBuffers* device_buffers() noexcept {
        return device_buffers_.get();
    }
    [[nodiscard]] const batch::BatchDeviceBuffers* device_buffers() const noexcept {
        return device_buffers_.get();
    }

    // F (MTP-3 speculation gate): S7 predicate bound to one StepPlan.
    // Returns the MTP draft window for the step: kMtpSpecDecodeDrafts (3) on
    // decode-only steps, 0 otherwise. RUN <=> has_decode && !has_prefill, the
    // same decode-only shape as batch::IsGraphSafe, so spec runs ONLY on
    // graph-safe decode-only steps; any mixed prefill+decode step resolves to
    // 0 (spec off, ordinary target-only decode). A nonzero window selects the
    // batched mtp_decode_batch path (one target_verify_accept over the K+1
    // verify width, then the MTP-3 draft chain over the in-checkpoint MTP
    // head); zero never enters it (k == 0 is outside that entry's domain).
    // Header-only S7 gate: no DFlash/DFlash2 backend, never the 10-wide
    // frame-domain window.
    [[nodiscard]] std::uint32_t mtp_draft_window(const batch::StepPlan& plan) const noexcept;
    [[nodiscard]] bool should_run_mtp_speculation(const batch::StepPlan& plan) const noexcept;

    // Committee A+B admission inbox (ONE scheduler): prepare_impl registers
    // the per-request sink state here and queues the batch::Request WITHOUT
    // touching the scheduler; the pump (run(), holding pump_mutex_) drains
    // the inbox into hooks_->on_new_request before schedule_step. IDs are
    // minted here so the state carries req/seq ids before admission.
    // Thread-safe: inbox has its own mutex (prepare path is const).
    std::uint64_t submit_inbox(batch::Request request,
                               std::shared_ptr<ServeRequestState> state);
    // Drain inbox into the scheduler via on_new_request. MUST be called with
    // the GenerationService pump_mutex_ held (it touches scheduler + the
    // seq->state map, which are pump-serialized). Registers each state in
    // the seq map before admission so no decoded pair misses its state.
    void drain_inbox_to_scheduler();
    // Look up stream state by seq id. MUST be called with pump_mutex_ held
    // (map is pump-serialized; states themselves are per-state locked).
    [[nodiscard]] std::shared_ptr<ServeRequestState>
    find_state_by_seq(std::uint64_t seq_id);
    // Remove stream states for finished req ids. MUST be called with
    // pump_mutex_ held.
    void erase_states_for_reqs(const std::vector<std::uint64_t>& finished_reqs);
    // Abort a waiting/running request and detach its stream state. MUST be
    // called with pump_mutex_ held (takes inbox_mutex_ + per-state locks in
    // the established pump -> inbox/state order). Covers all three
    // homes of a request: the admission inbox (swept, so a request aborted
    // before its first drain can never be admitted later), the seq-state
    // registry (states marked finished + sink detached + erased, so no
    // pump can publish to a dead sink again), and the scheduler (abort()
    // marks Aborted; storage is reclaimed on the next schedule_step and
    // find_request reads null, so owning pumps observe own_done and exit).
    // Idempotent: unknown req ids are a no-op.
    void abort_request(std::uint64_t req_id);

private:
    HookLoopConfig config_;
    std::unique_ptr<batch::RequestScheduler> scheduler_;
    std::unique_ptr<batch::EngineHooks> hooks_;
    std::unique_ptr<batch::BatchDeviceBuffers> device_buffers_;
    std::mutex warmup_mutex_;
    bool warmed_ = false;
    // Committee A+B inbox + stream registry. inbox_mutex_ guards inbox_
    // (prepare path); seq_states_ is pump-serialized (pump_mutex_ held).
    // No TextContext/KV/GDN bytes here: page ids + sequencing live in the
    // scheduler pool; this loop holds only ids and host-side stream states.
    struct InboxItem {
        batch::Request request;
        std::shared_ptr<ServeRequestState> state;
    };
    mutable std::mutex inbox_mutex_;
    std::vector<InboxItem> inbox_;
    std::atomic<std::uint64_t> next_id_{1};
    std::unordered_map<std::uint64_t, std::shared_ptr<ServeRequestState>> seq_states_;
};

} // namespace ninfer::serve
