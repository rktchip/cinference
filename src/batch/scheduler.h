#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <utility>
#include <vector>

#include "batch/paged_kv.h"
#include "batch/request.h"

// Iteration-level continuous-batching scheduler (HyperQwen-style policy,
// cinference-native execution).
//
// Each schedule_step() call: (1) evicts finished/aborted running requests and
// returns their blocks + table rows, (2) admits waiting requests FIFO while
// running slots, table rows, and prompt blocks allow, (3) emits one StepPlan
// mixing a chunked-prefill slice (capped at chunk_tokens_) with one token for
// every runnable decode sequence. This is iteration-level scheduling: membership
// changes every step, unlike the upstream single staged-prefill lane
// (runtime::Scheduler::set_prefill_lane, which throws on a second staged
// prefill). Admission here is strict FIFO; backfill past a blocked head stays
// with the upstream admission bridge (batch/cinference_hooks.h).
//
// NINFER_FAIR_PREFILL=1 (default unset = legacy behavior exactly) enables the
// fair policy: (a) the per-step prefill budget is shared fair-share across
// waiting prefill sequences instead of head-eats-all, (b) any decode row
// present halves the prefill cap so decodes never starve, (c) a blocked
// admission head is skipped past up to 4 admittable followers per step.
namespace ninfer::batch {

struct PrefillSlice {
    std::uint64_t seq_id  = 0;
    std::uint32_t offset  = 0; // first token index of the slice
    std::uint32_t count   = 0; // slice length, <= chunk budget
};

struct StepPlan {
    std::vector<std::uint64_t> decode_seq_ids;
    std::vector<PrefillSlice> prefill;
    std::uint32_t prefill_tokens = 0;

    [[nodiscard]] bool empty() const noexcept {
        return decode_seq_ids.empty() && prefill.empty();
    }
    [[nodiscard]] bool has_prefill() const noexcept { return !prefill.empty(); }
    // FNV-1a over decode_seq_ids in plan order. Identifies the exact decode
    // membership a captured graph was built for: different membership, order,
    // or set means a different signature and the step must not replay (goes
    // eager until the graph is recaptured/updated).
    [[nodiscard]] std::uint64_t decode_signature() const noexcept {
        std::uint64_t h = 1469598103934665603ULL;
        for (std::uint64_t seq_id : decode_seq_ids) {
            h ^= seq_id;
            h *= 1099511628211ULL;
        }
        return h;
    }
};

// CUDA-graph-safe decode predicate: true only for a pure decode step. Prefill
// (ragged, input-dependent shapes) always takes the eager path; decode replays
// the captured graph at the padded shape the caller established at capture.
[[nodiscard]] inline bool IsGraphSafe(const StepPlan& plan) noexcept {
    return !plan.decode_seq_ids.empty() && !plan.has_prefill();
}

class RequestScheduler {
public:
    // max_running_seqs: running cap. total_pages: startup-fixed logical blocks.
    // max_blocks_per_seq: per-seq logical max-context in blocks. chunk_tokens:
    // prefill slice cap per step.
    RequestScheduler(std::uint32_t max_running_seqs, std::uint32_t total_pages,
                     std::uint32_t max_blocks_per_seq, std::uint32_t chunk_tokens);

    RequestScheduler(const RequestScheduler&)            = delete;
    RequestScheduler& operator=(const RequestScheduler&) = delete;

    // Enqueue; assigns a fresh req_id/seq_id when the caller left them zero.
    std::uint64_t submit(Request request);
    // Abort a waiting or running request; storage is reclaimed on the next step.
    bool abort(std::uint64_t req_id);

    // Build the next step plan (evict -> admit -> mix).
    StepPlan schedule_step();
    // Apply a plan: advance prefill slices, append decoded tokens, finish
    // sequences whose decode budget is exhausted. A request whose block table
    // cannot grow (prefill or decode) is aborted so the next step reclaims
    // its blocks. Returns finished req ids.
    std::vector<std::uint64_t>
    on_step_done(const StepPlan& plan,
                 const std::vector<std::pair<std::uint64_t, TokenId>>& decoded);

    // Test/hook accessors.
    [[nodiscard]] std::size_t waiting() const noexcept { return waiting_.size(); }
    [[nodiscard]] std::size_t running() const noexcept { return running_.size(); }
    [[nodiscard]] std::uint32_t chunk_tokens() const noexcept { return chunk_tokens_; }
    [[nodiscard]] const PagedKvPool& pool() const noexcept { return pool_; }
    [[nodiscard]] PagedKvPool& pool() noexcept { return pool_; }
    [[nodiscard]] const Request* find_request(std::uint64_t req_id) const noexcept;
    [[nodiscard]] const Sequence* find_sequence(std::uint64_t seq_id) const noexcept;
    [[nodiscard]] bool finish_sequence(std::uint64_t seq_id);

private:
    void evict_done();
    void admit_waiting();
    [[nodiscard]] bool admit_one(Request& request);
    [[nodiscard]] Request* find_request_mut(std::uint64_t req_id) noexcept;
    [[nodiscard]] Sequence* find_sequence_mut(std::uint64_t seq_id) noexcept;

    std::uint32_t max_running_ = 0;
    std::uint32_t chunk_tokens_ = 0;
    std::uint64_t next_id_      = 1;
    std::deque<Request> waiting_;
    std::deque<Request> running_;
    PagedKvPool pool_;
};

} // namespace ninfer::batch
