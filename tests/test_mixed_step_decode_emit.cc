// tests/test_mixed_step_decode_emit.cc
// Slot D test (b): a mixed StepPlan emits its decode token through the
// mocked sample step.
//
// Setup mirrors tests/test_mixed_step.cc: request A (short, driven to decode
// before B arrives) plus request B (long, still prefilling) under
// chunk_tokens=8, so the real RequestScheduler emits ONE mixed StepPlan
// carrying both phases (one prefill slice + one decode id, M = 8 + 1 = 9).
//
// The sample step is MOCKED (no device, no GPU): synthetic (seq_id, token)
// pairs are built for exactly the plan's decode_seq_ids and fed into
// EngineHooks::on_step_done. The test asserts the scheduler advances:
// prefill accounting moves, the decoded token appends, the finished ids
// return, and the fed decoded count == decode_seq_ids.size().
//
// Host-only: no kernel launch, no device handle. GPU: GATED.
#include "batch/cinference_hooks.h"

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

std::vector<ninfer::TokenId> make_prompt(int base, std::size_t count) {
    std::vector<ninfer::TokenId> tokens(count);
    for (std::size_t i = 0; i < count; ++i) {
        tokens[i] = static_cast<ninfer::TokenId>(base + static_cast<int>(i));
    }
    return tokens;
}

bool finished_has(const std::vector<std::uint64_t>& finished, std::uint64_t id) {
    for (std::uint64_t got : finished) {
        if (got == id) {
            return true;
        }
    }
    return false;
}

} // namespace

int main() {
    using namespace ninfer::batch;
    RequestScheduler sched(/*max_running_seqs=*/4, /*total_pages=*/64,
                           /*max_blocks_per_seq=*/64, /*chunk_tokens=*/8);
    EngineHooks hooks(&sched);

    // Request A: short prompt, driven to decode before B arrives. max_new=1
    // so a single mocked decode token finishes it.
    const std::uint64_t id_a =
        hooks.on_new_request(Request(0, make_prompt(11, 6), /*max_new=*/1));
    {
        StepPlan first = hooks.schedule_step();
        CHECK(!first.empty());
        CHECK(first.has_prefill());
        CHECK(first.decode_seq_ids.empty());
        CHECK(first.prefill_tokens == 6);
        hooks.on_step_done(first, {});
    }
    const Request* req_a = sched.find_request(id_a);
    CHECK(req_a != nullptr && req_a->all_decode());

    // Request B: long prompt; its first step must mix with A's decode.
    // Two distinct in-memory sessions share the ONE plan below.
    const std::uint64_t id_b =
        hooks.on_new_request(Request(0, make_prompt(100, 40), /*max_new=*/1));
    CHECK(id_a != id_b);
    StepPlan mixed = hooks.schedule_step();
    CHECK(mixed.has_prefill() && !mixed.decode_seq_ids.empty()); // both phases
    CHECK(mixed.decode_seq_ids.size() == 1);
    CHECK(mixed.prefill.size() == 1);
    CHECK(mixed.prefill_tokens == 8); // chunk cap

    // One dispatch carries both phases: M = prefill_tokens + n_decode.
    StepDispatch dispatch = hooks.dispatch_step(mixed);
    CHECK(dispatch.m == mixed.prefill_tokens +
                          static_cast<std::uint32_t>(mixed.decode_seq_ids.size()));
    CHECK(dispatch.m == 9);
    CHECK(dispatch.batch.tokens.size() == dispatch.m);

    // MOCKED sample step: one synthetic token per decode id in the plan.
    const std::uint64_t seq_a = mixed.decode_seq_ids.front();
    const std::uint64_t seq_b = mixed.prefill.front().seq_id;
    const Sequence* before_a = sched.find_sequence(seq_a);
    const Sequence* before_b = sched.find_sequence(seq_b);
    CHECK(before_a != nullptr && before_b != nullptr);
    const std::size_t tokens_a_before =
        (before_a != nullptr) ? before_a->tokens.size() : 0;
    const std::size_t computed_b_before =
        (before_b != nullptr) ? before_b->computed_len : 0;

    std::vector<std::pair<std::uint64_t, ninfer::TokenId>> decoded;
    for (std::size_t i = 0; i < mixed.decode_seq_ids.size(); ++i) {
        decoded.emplace_back(mixed.decode_seq_ids[i],
                             static_cast<ninfer::TokenId>(700 + i));
    }
    // The mock feeds exactly one pair per decode row in the plan.
    CHECK(decoded.size() == mixed.decode_seq_ids.size());
    CHECK(decoded.size() == 1);

    const std::vector<std::uint64_t> finished = hooks.on_step_done(mixed, decoded);

    // A spent its single decode budget: its request id is finished.
    CHECK(finished.size() == 1);
    CHECK(finished_has(finished, id_a));

    // Accounting advanced on both phases: A's decoded token appended, B's
    // prefill slice consumed.
    const Sequence* after_a = sched.find_sequence(seq_a);
    const Sequence* after_b = sched.find_sequence(seq_b);
    CHECK(after_a != nullptr && after_a->tokens.size() == tokens_a_before + 1);
    CHECK(after_a != nullptr && after_a->tokens.back() == decoded.front().second);
    CHECK(after_b != nullptr &&
          after_b->computed_len == computed_b_before + mixed.prefill.front().count);
    CHECK(after_b != nullptr && after_b->computed_len == 8);

    // The scheduler advances past the finished decode: the next step still
    // carries B's remaining prefill (A is evicted as done).
    StepPlan next = hooks.schedule_step();
    CHECK(!next.empty());
    CHECK(next.has_prefill());
    CHECK(next.decode_seq_ids.empty());

    // Teardown reclaims everything.
    CHECK(sched.abort(id_b));
    (void)hooks.schedule_step(); // evict_done reclaims finished/aborted requests
    CHECK(sched.running() == 0 && sched.waiting() == 0);
    CHECK(sched.pool().free_pages() == 64);
    CHECK(sched.pool().used_pages() == 0);

    // GPU execution section: GATED. No kernel launches from this session
    // (shared-RTX-5090 rule); the sample step above is a host-side mock.
    std::cout << "mixed_step_decode_emit GPU section: GATED (mock sample step, no kernel launch)\n";

    if (failures == 0) {
        std::cout << "mixed_step_decode_emit: PASS (mixed plan emitted 1 decode token, finished A)\n";
    } else {
        std::cout << "mixed_step_decode_emit: " << failures << " FAILURES\n";
    }
    return failures == 0 ? 0 : 1;
}
