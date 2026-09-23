#include "batch/batch.h"
#include "batch/cinference_hooks.h"
#include "batch/paged_kv.h"
#include "batch/request.h"
#include "batch/scheduler.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

// Continuous-batching port test: 4 concurrent short prompts + 1 long prefill.
// Asserts full completion, chunked-prefill bounds, no overlapping pages across
// live sequences at every step, and no double-free / no leak at teardown.
namespace {

int failures = 0;

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            ++failures;                                                                            \
            std::cout << "FAIL " << __LINE__ << ": " #cond "\n";                                   \
        }                                                                                          \
    } while (0)

std::vector<ninfer::TokenId> make_prompt(int base, std::size_t count) {
    std::vector<ninfer::TokenId> tokens(count);
    for (std::size_t i = 0; i < count; ++i) { tokens[i] = static_cast<ninfer::TokenId>(base + i); }
    return tokens;
}

int test_request_lifecycle() {
    using namespace ninfer::batch;
    const int before = failures;
    Request req(7, make_prompt(1, 8), 4);
    CHECK(req.req_id == 7);
    CHECK(!req.done());
    CHECK(req.seqs.size() == 1);
    Sequence& seq = req.seqs.front();
    CHECK(seq.phase() == Phase::Prefill);
    CHECK(seq.remaining_prefill() == 8);
    seq.advance_computed(8);
    CHECK(seq.phase() == Phase::Decode);
    seq.append_token(42);
    CHECK(seq.phase() == Phase::Decode);
    CHECK(seq.tokens.size() == 9);
    CHECK(req.all_decode());
    CHECK(Sequence::blocks_for(0) == 0);
    CHECK(Sequence::blocks_for(1) == 1);
    CHECK(Sequence::blocks_for(16) == 1);
    CHECK(Sequence::blocks_for(17) == 2);
    req.finish();
    CHECK(req.done());
    Request doomed(9, make_prompt(1, 2), 1);
    doomed.abort();
    CHECK(doomed.done());
    return failures - before;
}

int test_pool_basics() {
    using namespace ninfer::batch;
    const int before = failures;
    PagedKvPool pool(64, 8, 4);
    CHECK(pool.free_pages() == 64);
    CHECK(pool.used_pages() == 0);

    std::vector<std::int32_t> a, b;
    CHECK(pool.grow(a, 3));
    CHECK(pool.grow(b, 3));
    CHECK(pool.free_pages() == 58);
    // No overlapping pages between the two tables.
    std::vector<std::int32_t> both = a;
    both.insert(both.end(), b.begin(), b.end());
    std::sort(both.begin(), both.end());
    CHECK(std::adjacent_find(both.begin(), both.end()) == both.end());
    // Shrink returns the tail.
    pool.shrink(a, 1);
    CHECK(a.size() == 1);
    CHECK(pool.free_pages() == 60);
    // Free is complete and idempotent: double-free changes nothing.
    pool.free(b);
    CHECK(b.empty());
    const std::uint32_t after_first_free = pool.free_pages();
    pool.free(b);
    CHECK(pool.free_pages() == after_first_free);
    pool.free(a);
    CHECK(pool.free_pages() == 64);
    CHECK(pool.used_pages() == 0);
    // Over-capacity grow fails without mutation.
    std::vector<std::int32_t> c;
    CHECK(!pool.grow(c, 9)); // exceeds max_blocks_per_seq
    CHECK(c.empty());
    CHECK(pool.free_pages() == 64);
    // Row leases: exhaust, then -1; duplicate release is a no-op.
    std::vector<int> rows;
    for (int i = 0; i < 4; ++i) { rows.push_back(pool.acquire_row()); }
    CHECK(pool.acquire_row() == -1);
    pool.release_row(rows.back());
    pool.release_row(rows.back()); // duplicate release: no-op
    CHECK(pool.acquire_row() >= 0);
    CHECK(pool.acquire_row() == -1);
    pool.release_row(-1); // invalid rows are ignored
    pool.release_row(99);
    // Watermark path: fresh ids come from the bump until the free-list refills.
    PagedKvPool fresh(8, 8, 1);
    std::vector<std::int32_t> first, second;
    CHECK(fresh.grow(first, 8));
    CHECK(fresh.free_pages() == 0);
    CHECK(!fresh.grow(second, 1)); // exhausted: fail without mutation
    CHECK(second.empty());
    fresh.free(first);
    CHECK(fresh.free_pages() == 8);
    CHECK(fresh.grow(second, 8)); // recycled from the free-list
    CHECK(second.size() == 8);
    return failures - before;
}

int test_scheduler_four_plus_one() {
    using namespace ninfer::batch;
    const int before = failures;
    constexpr std::uint32_t kPages = 256;
    constexpr std::uint32_t kChunk = 512;
    RequestScheduler sched(5, kPages, 192, kChunk);

    std::vector<std::uint64_t> short_ids;
    for (int i = 0; i < 4; ++i) {
        short_ids.push_back(sched.submit(Request(0, make_prompt(100 * (i + 1), 8), 4)));
    }
    const std::uint64_t long_id = sched.submit(Request(0, make_prompt(7, 3000), 2));
    // Abort path: a cancelled waiter is dropped, never admitted.
    const std::uint64_t doomed = sched.submit(Request(0, make_prompt(3, 5), 1));
    CHECK(sched.abort(doomed));
    CHECK(!sched.abort(424242)); // unknown id

    std::uint32_t long_sliced   = 0;
    std::uint32_t long_slices   = 0;
    std::uint32_t steps         = 0;
    bool saw_mixed              = false;
    bool overlap_found          = false;
    bool chunk_violated         = false;
    for (steps = 0; steps < 100; ++steps) {
        StepPlan plan = sched.schedule_step();
        if (plan.empty()) { break; }
        if (plan.prefill_tokens > kChunk) { chunk_violated = true; }
        if (plan.has_prefill() && !plan.decode_seq_ids.empty()) { saw_mixed = true; }
        // No overlapping pages across every live sequence, every step.
        {
            std::vector<std::int32_t> ids;
            for (std::uint64_t req_id : short_ids) {
                const Request* req = sched.find_request(req_id);
                if (req == nullptr) { continue; }
                for (const auto& seq : req->seqs) {
                    ids.insert(ids.end(), seq.block_table.begin(), seq.block_table.end());
                }
            }
            if (const Request* req = sched.find_request(long_id)) {
                for (const auto& seq : req->seqs) {
                    ids.insert(ids.end(), seq.block_table.begin(), seq.block_table.end());
                }
            }
            std::sort(ids.begin(), ids.end());
            if (std::adjacent_find(ids.begin(), ids.end()) != ids.end()) {
                overlap_found = true;
            }
            CHECK(sched.pool().used_pages() <= kPages);
        }
        for (const auto& slice : plan.prefill) {
            if (const Sequence* seq = sched.find_sequence(slice.seq_id)) {
                if (seq->prompt_len == 3000) {
                    long_slices += 1;
                    long_sliced += slice.count;
                }
            }
        }
        // Mixed assembly must succeed on a real step.
        if (plan.has_prefill() && !plan.decode_seq_ids.empty()) {
            const RaggedBatch batch = AssembleBatch(sched, plan);
            CHECK(batch.num_seqs() == plan.prefill.size() + plan.decode_seq_ids.size());
            CHECK(batch.seq_offsets.front() == 0);
            CHECK(batch.seq_offsets.back() == batch.tokens.size());
            CHECK(batch.block_tables.empty()); // device-gather path: no host matrix
            for (std::size_t i = 0; i < batch.num_seqs(); ++i) {
                CHECK(batch.seq_offsets[i + 1] - batch.seq_offsets[i] ==
                      batch.seq_lengths[i]);
            }
            CHECK(!IsGraphSafe(plan)); // mixed steps never replay
        }
        std::vector<std::pair<std::uint64_t, ninfer::TokenId>> decoded;
        for (std::uint64_t seq_id : plan.decode_seq_ids) {
            decoded.emplace_back(seq_id, static_cast<ninfer::TokenId>(9000 + steps));
        }
        sched.on_step_done(plan, decoded);
    }
    CHECK(!chunk_violated);
    CHECK(!overlap_found);
    CHECK(saw_mixed); // chunked prefill overlapped live decodes at least once
    CHECK(long_sliced == 3000);
    CHECK(long_slices >= 6); // ceil(3000 / 512)
    CHECK(sched.running() == 0);
    CHECK(sched.waiting() == 0);
    CHECK(sched.find_request(doomed) == nullptr);
    for (std::uint64_t req_id : short_ids) { CHECK(sched.find_request(req_id) == nullptr); }
    CHECK(sched.find_request(long_id) == nullptr);
    // No leak, no double-free: every block is back.
    CHECK(sched.pool().free_pages() == kPages);
    CHECK(sched.pool().used_pages() == 0);
    std::vector<std::int32_t> empty;
    sched.pool().free(empty);
    CHECK(sched.pool().free_pages() == kPages);
    std::cout << "steps=" << steps << " long_slices=" << long_slices << "\n";
    return failures - before;
}

int test_batch_predicates() {
    using namespace ninfer::batch;
    const int before = failures;
    StepPlan mixed;
    mixed.decode_seq_ids.push_back(1);
    mixed.prefill.push_back(PrefillSlice{.seq_id = 2, .offset = 0, .count = 16});
    mixed.prefill_tokens = 16;
    CHECK(!IsGraphSafe(mixed));
    StepPlan decode_only;
    decode_only.decode_seq_ids = {1, 2, 3};
    CHECK(IsGraphSafe(decode_only));
    StepPlan empty;
    CHECK(!IsGraphSafe(empty));
    // Stale plans fail fast instead of running.
    RequestScheduler sched(2, 64, 32, 128);
    sched.submit(Request(0, make_prompt(1, 4), 1));
    StepPlan stale;
    stale.decode_seq_ids.push_back(987654321);
    bool threw = false;
    try {
        (void)AssembleBatch(sched, stale);
    } catch (const std::out_of_range&) { threw = true; }
    CHECK(threw);
    // Graph-replay signature: same membership in plan order -> equal;
    // different order or set -> different (replay forbidden, eager instead).
    StepPlan s1; s1.decode_seq_ids = {1, 2};
    StepPlan s2c; s2c.decode_seq_ids = {1, 2};
    StepPlan s3; s3.decode_seq_ids = {2, 1};
    StepPlan s4; s4.decode_seq_ids = {1, 3};
    CHECK(s1.decode_signature() == s2c.decode_signature());
    CHECK(s3.decode_signature() != s1.decode_signature());
    CHECK(s4.decode_signature() != s1.decode_signature());
    // Host-tables assembly is an explicit opt-in: dropped by default (the
    // device-gather path uploads seq_rows instead), built when requested.
    StepPlan p2 = sched.schedule_step();
    const Request* req2 = sched.find_request(1);
    CHECK(req2 != nullptr && req2->seqs.front().block_table.size() == 1);
    const RaggedBatch bt_default = AssembleBatch(sched, p2);
    CHECK(bt_default.block_tables.empty());
    const RaggedBatch bt_host = AssembleBatch(sched, p2, true);
    CHECK(bt_host.block_tables.size() ==
          bt_host.num_seqs() * sched.pool().max_blocks_per_seq());
    CHECK(bt_host.block_tables[0] == req2->seqs.front().block_table[0]);
    CHECK(bt_host.block_tables[1] == kUnmappedBlock);
    return failures - before;
}

// Duck-typed double for ninfer::runtime::Scheduler<Req>. It mirrors the exact
// method set and grant/commit semantics read from
// src/runtime/engine/scheduler.h (observe_fifo_head / grant_head /
// qualify_backfill / commit_admission, move-only AdmissionGrant, FIFO-head
// binding), so UpstreamAdmissionBridge instantiates against it exactly as it
// will against the real scheduler on the project's Linux toolchain. Upstream
// headers cannot compile under MSVC (unconditional `unsigned __int128` in
// runtime/contract/resources.h; the supported build is the Linux Docker image),
// hence the double: it verifies the bridge's call order, move semantics,
// exception propagation, and branch handling, not upstream's policy.
struct MockUpstreamScheduler {
    enum class BackfillClass : std::uint8_t { None, Persistent };

    struct AdmissionSnapshot {
        std::uint64_t request_id = 0;
    };

    class AdmissionGrant {
    public:
        AdmissionGrant(AdmissionGrant&&) noexcept            = default;
        AdmissionGrant& operator=(AdmissionGrant&&) noexcept = default;
        AdmissionGrant(const AdmissionGrant&)                = delete;
        AdmissionGrant& operator=(const AdmissionGrant&)     = delete;

        [[nodiscard]] std::uint64_t request_id() const noexcept { return request_id_; }
        [[nodiscard]] BackfillClass backfill_class() const noexcept { return backfill_class_; }
        [[nodiscard]] std::uint64_t protection_epoch() const noexcept {
            return protection_epoch_;
        }
        [[nodiscard]] std::uint64_t service_work_quanta() const noexcept {
            return service_work_quanta_;
        }

    private:
        AdmissionGrant(std::uint64_t request_id, BackfillClass backfill_class,
                       std::uint64_t protection_epoch, std::uint64_t service_work_quanta) noexcept
            : request_id_(request_id)
            , backfill_class_(backfill_class)
            , protection_epoch_(protection_epoch)
            , service_work_quanta_(service_work_quanta) {}

        std::uint64_t request_id_         = 0;
        BackfillClass backfill_class_     = BackfillClass::None;
        std::uint64_t protection_epoch_   = 0;
        std::uint64_t service_work_quanta_ = 0;

        friend struct MockUpstreamScheduler;
    };

    void observe_fifo_head(std::optional<std::uint64_t> request_id) noexcept {
        if (fifo_head_id_ == request_id) { return; }
        fifo_head_id_ = request_id;
        protection_.reset();
    }

    [[nodiscard]] AdmissionGrant grant_head(std::uint64_t request_id,
                                            std::uint64_t service_work_quanta) const {
        if (!fifo_head_id_ || *fifo_head_id_ != request_id || request_id == 0 ||
            service_work_quanta == 0) {
            throw std::logic_error("head admission is not bound to the observed FIFO head");
        }
        return AdmissionGrant(request_id, BackfillClass::None, 0, service_work_quanta);
    }

    [[nodiscard]] bool validate_grant(const AdmissionGrant& grant) const noexcept {
        if (grant.request_id_ == 0 || grant.service_work_quanta_ == 0) { return false; }
        if (grant.backfill_class_ == BackfillClass::None) {
            return grant.protection_epoch_ == 0 && fifo_head_id_ &&
                   *fifo_head_id_ == grant.request_id_;
        }
        if (!fifo_head_id_ || !protection_ || protection_->head_request_id != *fifo_head_id_ ||
            protection_->epoch_id != grant.protection_epoch_ ||
            grant.request_id_ == *fifo_head_id_) {
            return false;
        }
        return grant.backfill_class_ == BackfillClass::Persistent;
    }

    void commit_admission(AdmissionGrant&& grant) {
        if (!validate_grant(grant)) {
            throw std::logic_error("admission grant is stale or inconsistent");
        }
        if (grant.backfill_class_ == BackfillClass::None) {
            fifo_head_id_.reset();
            protection_.reset();
        }
        grant.request_id_          = 0;
        grant.service_work_quanta_ = 0;
    }

    [[nodiscard]] bool protect_blocked_head(std::uint64_t request_id,
                                            std::span<const AdmissionSnapshot> /*active*/,
                                            std::uint64_t /*revision*/) {
        if (!fifo_head_id_ || *fifo_head_id_ != request_id) {
            throw std::logic_error("blocked admission does not match the observed FIFO head");
        }
        if (!protection_ || protection_->head_request_id != request_id) {
            protection_.emplace(Protection{next_epoch_++, request_id});
        }
        return donor_present_;
    }

    [[nodiscard]] std::optional<AdmissionGrant>
    qualify_backfill(std::uint64_t request_id, std::uint64_t service_work_quanta,
                     std::span<const AdmissionSnapshot> /*active*/,
                     std::uint64_t /*revision*/) const {
        if (!fifo_head_id_ || !protection_ || protection_->head_request_id != *fifo_head_id_) {
            throw std::logic_error("backfill qualification has no open protected head");
        }
        if (request_id == 0 || request_id == *fifo_head_id_ || service_work_quanta == 0) {
            throw std::logic_error("backfill candidate has invalid scheduling identity");
        }
        if (authorize_backfill_) {
            return AdmissionGrant(request_id, BackfillClass::Persistent, protection_->epoch_id,
                                  service_work_quanta);
        }
        return std::nullopt;
    }

    // Test controls.
    bool donor_present_      = false;
    bool authorize_backfill_ = false;

private:
    struct Protection {
        std::uint64_t epoch_id        = 0;
        std::uint64_t head_request_id = 0;
    };

    std::optional<std::uint64_t> fifo_head_id_;
    std::optional<Protection> protection_;
    std::uint64_t next_epoch_ = 1;
};

int test_hooks_and_bridge() {
    using namespace ninfer::batch;
    const int before = failures;
    RequestScheduler sched(4, 128, 64, 256);
    EngineHooks hooks(&sched);
    const std::uint64_t id = hooks.on_new_request(Request(0, make_prompt(5, 10), 2));
    CHECK(id != 0);

    StepPlan plan = sched.schedule_step();
    CHECK(!plan.empty());
    const std::vector<LaneAssignment> lanes = hooks.pack_lanes(plan);
    CHECK(lanes.size() == plan.prefill.size() + plan.decode_seq_ids.size());
    // S1: every lane stages in the single forward. No slice is deferred to a
    // later round; staged_upstream is pinned true for every entry (see
    // src/batch/cinference_hooks.h), so the whole plan stages at once.
    CHECK(!lanes.empty());
    for (const auto& lane : lanes) {
        CHECK(lane.lane < ninfer::kMaximumConcurrency);
        CHECK(lane.staged_upstream);
    }
    CHECK(hooks.graph_route(plan) == GraphRoute::Eager);
    StepPlan decode_only;
    decode_only.decode_seq_ids = {plan.decode_seq_ids.empty() ? id : plan.decode_seq_ids[0]};
    CHECK(hooks.graph_route(decode_only) == GraphRoute::GraphReplay);
    // Lane overflow fails fast.
    StepPlan huge;
    for (std::uint64_t i = 0; i < ninfer::kMaximumConcurrency + 1; ++i) {
        huge.decode_seq_ids.push_back(1000 + i);
    }
    bool threw = false;
    try {
        (void)hooks.pack_lanes(huge);
    } catch (const std::length_error&) { threw = true; }
    CHECK(threw);

    // Upstream admission protocol interop against the faithful double.
    MockUpstreamScheduler upstream;
    CHECK(ninfer::batch::UpstreamAdmissionBridge::admit_head(upstream, 7, 3));
    bool grant_threw = false;
    try {
        upstream.observe_fifo_head(std::optional<std::uint64_t>(7));
        (void)upstream.grant_head(9, 3); // not the observed head
    } catch (const std::logic_error&) { grant_threw = true; }
    CHECK(grant_threw);
    // Backfill with no donor: clean false, committed nothing.
    {
        const std::vector<MockUpstreamScheduler::AdmissionSnapshot> active;
        CHECK(!ninfer::batch::UpstreamAdmissionBridge::admit_backfill(
            upstream, 7, 8, 1, std::span<const MockUpstreamScheduler::AdmissionSnapshot>(active),
            0));
    }
    // Backfill authorized: commits the persistent grant.
    upstream.donor_present_      = true;
    upstream.authorize_backfill_ = true;
    {
        const std::vector<MockUpstreamScheduler::AdmissionSnapshot> active;
        CHECK(ninfer::batch::UpstreamAdmissionBridge::admit_backfill(
            upstream, 7, 8, 1, std::span<const MockUpstreamScheduler::AdmissionSnapshot>(active),
            0));
    }
    // Backfill refused by policy: clean false.
    upstream.authorize_backfill_ = false;
    {
        const std::vector<MockUpstreamScheduler::AdmissionSnapshot> active;
        CHECK(!ninfer::batch::UpstreamAdmissionBridge::admit_backfill(
            upstream, 7, 9, 1, std::span<const MockUpstreamScheduler::AdmissionSnapshot>(active),
            0));
    }
    return failures - before;
}

int test_device_table_roundtrip() {
    using namespace ninfer::batch;
    const int before = failures;
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::cout << "no CUDA device: skipping device section\n";
        return 0;
    }
    PagedKvPool pool(32, 8, 2);
    CHECK(pool.device_ok());
    std::vector<std::int32_t> table;
    CHECK(pool.grow(table, 5));
    const int row = pool.acquire_row();
    CHECK(row >= 0);
    pool.publish_row(row, table);
    pool.sync_table_to_device();
    pool.remap_on_device();
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    // Read the device row back and compare with the host shadow.
    std::vector<std::int32_t> back(8, -2);
    CHECK(cudaMemcpy(back.data(), pool.device_table() + static_cast<std::size_t>(row) * 8,
                     8 * sizeof(std::int32_t), cudaMemcpyDeviceToHost) == cudaSuccess);
    for (std::size_t i = 0; i < 5; ++i) { CHECK(back[i] == table[i]); }
    for (std::size_t i = 5; i < 8; ++i) { CHECK(back[i] == -1); }
    BatchDeviceBuffers buffers(2, 8, 64);
    CHECK(buffers.device_ok());
    return failures - before;
}

} // namespace

int main() {
    int total = 0;
    total += test_request_lifecycle();
    total += test_pool_basics();
    total += test_scheduler_four_plus_one();
    total += test_batch_predicates();
    total += test_hooks_and_bridge();
    total += test_device_table_roundtrip();
    if (total == 0) { std::cout << "continuous_batch: PASS\n"; }
    else {
        std::cout << "continuous_batch: " << total << " FAILURES\n";
    }
    return total == 0 ? 0 : 1;
}
