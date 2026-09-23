#pragma once

#include <ninfer/types.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "batch/batch.h"
#include "batch/scheduler.h"

// Integration points between the iteration-level batch module and the existing
// Engine request/transaction/graph loop. This header wraps or replaces ONLY the
// current single-lane scheduling policy; it defines no second engine. It
// includes no upstream runtime headers: the bridge below is duck-typed against
// the exact method set read from src/runtime/engine/scheduler.h, so this
// header compiles on any toolchain while instantiating 1:1 against the real
// ninfer::runtime::Scheduler<Req> on the project's Linux build.
//
// S1 contract: RequestScheduler is the ONLY step oracle in the execute loop.
// The loop is on_new_request -> schedule_step -> dispatch_step (execute one
// forward with the returned ragged M) -> on_step_done. Engine::submit stays an
// enqueue-only wrapper (it must make no step, lane, or graph decisions); the
// worker loop must not call Scheduler::set_prefill_lane / choose_execution
// anymore. Those upstream single-lane symbols are superseded, not redefined,
// here: the replacement path is StepPlan -> one RaggedBatch -> one forward.
//
// Verified upstream touchpoints (do not rename without re-surveying):
//  1. Admission policy -- ninfer::runtime::Scheduler<Req> (src/runtime/engine/
//     scheduler.h): observe_fifo_head / grant_head / qualify_backfill /
//     commit_admission, with AdmissionGrant::{request_id, backfill_class,
//     protection_epoch, service_work_quanta}. UpstreamAdmissionBridge drives
//     exactly this sequence for heads admitted by batch::RequestScheduler.
//  2. Single staged-prefill lane -- SUPERSEDED by S1. Scheduler::set_prefill_lane
//     (throws on a second staged prefill) and choose_execution (alternating
//     Prefill/Decode) are replaced by EngineHooks::dispatch_step: the full
//     StepPlan assembles to one RaggedBatch and runs in one forward with
//     ragged M = prefill_tokens + decode count. pack_lanes() no longer maps
//     extra slices to later rounds: staged_upstream is pinned true for every
//     entry because the whole plan stages in the single forward.
//  3. CUDA-graph replay -- ninfer::DecodeGraphExecutable::launch(cudaStream_t)
//     (src/core/decode_graph.h) replays the captured decode topology while the
//     topology signature is unchanged. graph_route(plan, captured, has_capture)
//     sends a pure-decode step to replay ONLY when IsGraphSafe(plan) and the
//     plan's decode_signature() matches the captured signature; any
//     prefill-carrying step, and any decode membership/order change, goes
//     eager until the graph is recaptured/updated.
//  4. KV residency -- DeviceKVPagePool / KVExecutionTablePool::publish
//     (src/core/paged_kv_cache.h) own all KV bytes at 64-token physical pages;
//     four 16-token logical blocks from PagedKvPool address one physical page.
//  5. Ragged width -- size RequestScheduler with max_running_seqs <=
//     ninfer::kMaximumConcurrency so every emitted plan fits one forward.
//     dispatch_step() enforces this with validate_step_width() (throws
//     std::length_error otherwise) and pins M to the real assembled token
//     count, never a padded maximum.
namespace ninfer::batch {

enum class GraphRoute : std::uint8_t {
    GraphReplay = 0, // pure decode with matching captured signature: replay
    Eager       = 1, // prefill present or signature mismatch: run eager
};

struct LaneAssignment {
    std::uint32_t lane   = 0; // < ninfer::kMaximumConcurrency
    std::uint64_t seq_id = 0;
    bool is_prefill      = false;
    std::uint32_t tokens = 0; // slice length, or 1 for decode
    // S1: pinned true for every entry. The single forward carries the whole
    // plan, so no slice is deferred to a later round. Kept (instead of
    // removed) so downstream lane consumers keep compiling.
    bool staged_upstream = true;
};

// One StepPlan -> one RaggedBatch -> one forward. m is the ragged row count
// the linear dispatch consumes: m == batch.tokens.size() ==
// plan.prefill_tokens + plan.decode_seq_ids.size().
struct StepDispatch {
    RaggedBatch batch;
    std::uint32_t m                 = 0;
    GraphRoute route                = GraphRoute::Eager;
    std::uint64_t decode_signature  = 0;
};

class EngineHooks {
public:
    explicit EngineHooks(RequestScheduler* scheduler);

    // Enqueue-only wrapper. Engine::submit must stay the same shape: enqueue
    // the request, make no step/lane/graph decision.
    std::uint64_t on_new_request(Request request);
    // The only step oracle: evict -> admit -> mix (delegates to the scheduler).
    StepPlan schedule_step();
    // Full plan to one batch to one forward descriptor. Throws
    // std::length_error when the plan width exceeds kMaximumConcurrency
    // (size max_running_seqs at or below it) and std::logic_error when the
    // assembled token count disagrees with the plan (a stale plan must never
    // run). need_host_tables selects the host-only block-table matrix upload;
    // the default device-gather path uploads seq_rows instead.
    [[nodiscard]] StepDispatch dispatch_step(const StepPlan& plan,
                                            bool need_host_tables = false) const;
    // Apply a plan: advance prefill slices, append decoded tokens, finish
    // sequences whose decode budget is exhausted. Returns finished req ids.
    std::vector<std::uint64_t>
    on_step_done(const StepPlan& plan,
                 const std::vector<std::pair<std::uint64_t, TokenId>>& decoded);
    // Lane view of one StepPlan. S1: every entry is staged in the single
    // forward; nothing is mapped to a later round.
    [[nodiscard]] std::vector<LaneAssignment> pack_lanes(const StepPlan& plan) const;
    // Decode-only steps replay; mixed steps go eager. Plan-only form: reports
    // the route a plan shape alone implies (no capture state consulted).
    [[nodiscard]] GraphRoute graph_route(const StepPlan& plan) const noexcept;
    // Signature-gated form the execute loop must use: replay only when
    // IsGraphSafe(plan) AND has_capture AND the plan's decode_signature()
    // matches the captured signature. Mixed prefill+decode stays eager, as
    // does any decode membership/order change.
    [[nodiscard]] GraphRoute graph_route(const StepPlan& plan,
                                        std::uint64_t captured_signature,
                                        bool has_capture) const noexcept;
    // Capture bookkeeping for the gated form above. Call note_graph_capture
    // with the decoded plan's signature right after a successful capture (or
    // update), and clear_graph_capture whenever the topology is invalidated.
    void note_graph_capture(std::uint64_t decode_signature) noexcept;
    void clear_graph_capture() noexcept;
    [[nodiscard]] bool has_graph_capture() const noexcept { return graph_capture_.has_value(); }
    [[nodiscard]] std::uint64_t captured_decode_signature() const noexcept {
        return graph_capture_.value_or(0);
    }
    // Ragged M for one plan: prefill_tokens + one row per decode sequence.
    [[nodiscard]] static std::uint32_t step_m(const StepPlan& plan) noexcept;
    // Width cap: plan rows (prefill slices + decode seqs) must fit one
    // forward (<= kMaximumConcurrency). Throws std::length_error otherwise.
    static void validate_step_width(const StepPlan& plan);
    [[nodiscard]] RequestScheduler* scheduler() const noexcept { return scheduler_; }

private:
    RequestScheduler* scheduler_ = nullptr; // non-owning
    std::optional<std::uint64_t> graph_capture_; // last captured decode signature
};

// Header-only driver for the upstream FIFO/admission protocol. Duck-typed: any
// scheduler exposing observe_fifo_head / grant_head / commit_admission (head)
// and protect_blocked_head / qualify_backfill / commit_admission (backfill)
// with the signatures read from runtime/engine/scheduler.h works, including
// the real ninfer::runtime::Scheduler<Req>. This library never instantiates
// it, so it adds no link dependency.
struct UpstreamAdmissionBridge {
    // Head admission: observe_fifo_head -> grant_head -> commit_admission.
    // grant_head throws unless req_id is the observed head.
    template <class Scheduler>
    static bool admit_head(Scheduler& scheduler, std::uint64_t req_id,
                           std::uint64_t service_work_quanta) {
        scheduler.observe_fifo_head(req_id);
        auto grant = scheduler.grant_head(req_id, service_work_quanta);
        scheduler.commit_admission(std::move(grant));
        return true;
    }

    // Backfill admission past a blocked head: protect the head, qualify the
    // candidate, commit the persistent grant. Returns false when there is no
    // donor or the candidate is not authorized (no throw); throws when the
    // head binding itself is violated.
    template <class Scheduler, class Snapshot, class Revision>
    static bool admit_backfill(Scheduler& scheduler, std::uint64_t head_id,
                               std::uint64_t candidate_id, std::uint64_t service_work_quanta,
                               std::span<const Snapshot> active, Revision revision) {
        scheduler.observe_fifo_head(head_id);
        if (!scheduler.protect_blocked_head(head_id, active, revision)) { return false; }
        auto grant = scheduler.qualify_backfill(candidate_id, service_work_quanta, active,
                                                revision);
        if (!grant.has_value()) { return false; }
        scheduler.commit_admission(std::move(*grant));
        return true;
    }
};

} // namespace ninfer::batch
