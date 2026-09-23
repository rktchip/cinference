// tests/test_mixed_step.cc
// S5 proof: one StepPlan carrying prefill PLUS decode through one dispatch.
// S5b: two in-memory sessions share ONE StepPlan (one prefill slice + one
// decode), executed as ONE dispatch with M = prefill_tokens + n_decode; the
// stubbed dispatch_linear must observe exactly one call with that M.
// GPU execution: GATED (host-only run; no CUDA kernels launch).
//
// Two requests: A (short, already decoding) and B (long, still prefilling).
// The real RequestScheduler emits a single mixed StepPlan; the real
// EngineHooks::dispatch_step (S1 contract: one plan -> one RaggedBatch ->
// one forward, m pinned to the assembled token count) assembles it; then one
// dispatch_linear consumes exactly the assembled tokens.
//
// STUB note (S4): the linear forward below is a host stub standing in for
// ninfer::ops::dispatch_linear (CUDA/BF16 Tensor contract, S4-owned). It
// mirrors only the dispatch contract shape (m rows in, m rows out, exactly
// one call per dispatched step) and records (calls, m, k, n). Replace it with
// the real dispatch on the CUDA toolchain; the scheduler, assembly, and
// aliasing assertions around it are real and stay as-is.

#include "batch/cinference_hooks.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>

namespace {

int failures = 0;

#define CHECK(cond)                                                                  \
    do {                                                                             \
        if (!(cond)) {                                                               \
            ++failures;                                                              \
            std::cout << "FAIL " << __LINE__ << ": " #cond "\n";                     \
        }                                                                            \
    } while (0)

// STUB for S4 ninfer::ops::dispatch_linear: host token-count mirror.
// Records the launch; returns m zero rows. Not a numeric oracle.
struct StubLinearForward {
    int calls         = 0;
    std::uint32_t m   = 0;
    std::uint32_t k   = 0;
    std::uint32_t n   = 0;
    std::vector<std::int32_t> out_rows;

    void run(std::uint32_t rows, std::uint32_t in_k, std::uint32_t out_n) {
        ++calls;
        m = rows;
        k = in_k;
        n = out_n;
        out_rows.assign(rows, 0);
    }
};

std::vector<ninfer::TokenId> make_prompt(int base, std::size_t count) {
    std::vector<ninfer::TokenId> tokens(count);
    for (std::size_t i = 0; i < count; ++i) {
        tokens[i] = static_cast<ninfer::TokenId>(base + static_cast<int>(i));
    }
    return tokens;
}

// No page id is mapped by two live sequences at once (aliasing audit).
bool tables_alias_free(const ninfer::batch::RequestScheduler& sched,
                       std::uint64_t id_a, std::uint64_t id_b) {
    std::vector<std::int32_t> ids;
    for (std::uint64_t id : {id_a, id_b}) {
        const ninfer::batch::Request* req = sched.find_request(id);
        if (req == nullptr) {
            continue;
        }
        for (const auto& seq : req->seqs) {
            ids.insert(ids.end(), seq.block_table.begin(), seq.block_table.end());
        }
    }
    std::sort(ids.begin(), ids.end());
    return std::adjacent_find(ids.begin(), ids.end()) == ids.end();
}

} // namespace

int main() {
    using namespace ninfer::batch;
    RequestScheduler sched(/*max_running_seqs=*/4, /*total_pages=*/64,
                           /*max_blocks_per_seq=*/64, /*chunk_tokens=*/8);
    EngineHooks hooks(&sched);

    // Request A: short prompt, driven to decode before B arrives.
    const std::uint64_t id_a =
        hooks.on_new_request(Request(0, make_prompt(11, 6), /*max_new=*/2));
    {
        StepPlan first = hooks.schedule_step();
        CHECK(!first.empty());
        CHECK(first.has_prefill()); // A prefills first: prefill present
        CHECK(first.decode_seq_ids.empty());
        CHECK(first.prefill_tokens == 6);
        hooks.on_step_done(first, {});
    }
    const Request* req_a = sched.find_request(id_a);
    CHECK(req_a != nullptr && req_a->all_decode());

    // Request B: long prompt; its first step must mix with A's decode.
    const std::uint64_t id_b =
        hooks.on_new_request(Request(0, make_prompt(100, 40), /*max_new=*/1));
    CHECK(id_a != id_b); // two distinct in-memory sessions
    StepPlan mixed = hooks.schedule_step();
    CHECK(mixed.has_prefill() && !mixed.decode_seq_ids.empty()); // the mixed step
    CHECK(mixed.decode_seq_ids.size() == 1);
    CHECK(mixed.prefill.size() == 1);
    CHECK(mixed.prefill_tokens == 8); // chunk cap
    CHECK(mixed.prefill.front().offset == 0 && mixed.prefill.front().count == 8);
    CHECK(!IsGraphSafe(mixed)); // mixed steps never replay

    // S5b: the two sessions share this ONE plan: the decode token belongs to
    // session A, the single prefill slice belongs to session B.
    // (Fresh lookups: schedule_step moves requests between queues.)
    const Request* req_a2 = sched.find_request(id_a);
    const Request* req_b2 = sched.find_request(id_b);
    CHECK(req_a2 != nullptr && req_b2 != nullptr);
    CHECK(req_a2->seqs.size() == 1 && req_b2->seqs.size() == 1);
    CHECK(mixed.decode_seq_ids.front() == req_a2->seqs.front().seq_id);
    CHECK(mixed.prefill.front().seq_id == req_b2->seqs.front().seq_id);

    // One dispatch: S1 assembles the whole plan into one forward.
    // S5b: M = prefill_tokens + n_decode, consumed by exactly one dispatch.
    const std::uint32_t n_decode =
        static_cast<std::uint32_t>(mixed.decode_seq_ids.size());
    const std::uint32_t expect_m = mixed.prefill_tokens + n_decode;
    CHECK(n_decode == 1);
    CHECK(expect_m == 9);
    StepDispatch dispatch = hooks.dispatch_step(mixed);
    CHECK(dispatch.m == expect_m);
    CHECK(dispatch.m == 9);
    CHECK(dispatch.batch.tokens.size() == dispatch.m);
    CHECK(dispatch.batch.num_seqs() == 2);
    CHECK(dispatch.batch.seq_offsets.size() == 3 && dispatch.batch.seq_offsets[0] == 0 &&
          dispatch.batch.seq_offsets[1] == 8 && dispatch.batch.seq_offsets[2] == 9);
    CHECK(dispatch.batch.block_tables.empty()); // device-gather path: no host matrix
    // Layout: prefill-slice tokens first, then one token per decode sequence.
    for (int i = 0; i < 8; ++i) {
        CHECK(dispatch.batch.tokens[static_cast<std::size_t>(i)] ==
              static_cast<ninfer::TokenId>(100 + i));
    }
    CHECK(dispatch.batch.tokens[8] == static_cast<ninfer::TokenId>(16)); // A's last token
    CHECK(dispatch.decode_signature == mixed.decode_signature());

    // One linear forward over exactly the assembled rows (S4 stub, see header).
    // S5b: stubbed dispatch_linear observes exactly one call with M == expect_m.
    // GPU execution of the real dispatch_linear: GATED.
    StubLinearForward linear;
    linear.run(dispatch.m, /*in_k=*/16, /*out_n=*/32);
    CHECK(linear.calls == 1);
    CHECK(linear.m == expect_m);
    CHECK(linear.m == dispatch.batch.tokens.size());
    CHECK(linear.out_rows.size() == dispatch.m);

    // Apply the step: one decoded token for A, B's slice advances.
    const std::uint64_t seq_a = mixed.decode_seq_ids.front();
    const auto finished =
        hooks.on_step_done(mixed, {{seq_a, static_cast<ninfer::TokenId>(777)}});
    CHECK(finished.empty()); // A still owes one more decode (max_new == 2)
    CHECK(tables_alias_free(sched, id_a, id_b));
    CHECK(sched.pool().used_pages() > 0);

    // Teardown reclaims everything: no leak, no double-free residue.
    CHECK(sched.abort(id_a));
    CHECK(sched.abort(id_b));
    (void)hooks.schedule_step(); // evict_done reclaims aborted requests
    CHECK(sched.running() == 0 && sched.waiting() == 0);
    CHECK(sched.pool().free_pages() == 64);
    CHECK(sched.pool().used_pages() == 0);

    if (failures == 0) {
        std::cout << "mixed_step: PASS (one mixed plan, one dispatch, one linear)\n";
    } else {
        std::cout << "mixed_step: " << failures << " FAILURES\n";
    }
    return failures == 0 ? 0 : 1;
}
