// tests/test_step_forward_m.cc
// S1b proof: the runtime execution lane sets activation rows to the ragged
// total and issues exactly ONE linear dispatch per layer per step.
//
// Setup mirrors tests/test_mixed_step.cc: request A (short, driven to
// decode) plus request B (long, still prefilling) under chunk_tokens=8, so
// the real RequestScheduler emits one mixed StepPlan with M = 8 + 1 = 9.
// The counting fake stands in for ninfer::ops::dispatch_linear (CUDA/BF16
// Tensor contract): it records (calls, m per call). The S1b lane header
// (src/runtime/engine/step_forward.h) fixes M once before the layer loop
// and holds the single call site, so calls == num_layers and every m == 9
// by construction; this test asserts it on the real dispatch.
//
// Host-only: no kernel launch, no device handle. Run with GPUs hidden;
// the batch pool takes its host-only path.
#include "batch/cinference_hooks.h"
#include "runtime/engine/step_forward.h"

#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>

namespace {

int failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            ++failures;                                                          \
            std::cout << "FAIL " << __LINE__ << ": " #cond "\n";                 \
        }                                                                        \
    } while (0)

// Stand-in for the per-layer linear entry: records one launch per call.
struct CountingLinear {
    int calls              = 0;
    std::uint32_t first_m  = 0;
    bool all_equal         = true;

    void operator()(std::uint32_t layer, std::uint32_t m) {
        (void)layer;
        if (calls == 0) {
            first_m = m;
        } else if (m != first_m) {
            all_equal = false;
        }
        ++calls;
    }
};

std::vector<ninfer::TokenId> make_prompt(int base, std::size_t count) {
    std::vector<ninfer::TokenId> tokens(count);
    for (std::size_t i = 0; i < count; ++i) {
        tokens[i] = static_cast<ninfer::TokenId>(base + static_cast<int>(i));
    }
    return tokens;
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
        CHECK(first.prefill_tokens == 6);
        hooks.on_step_done(first, {});
    }
    (void)id_a;

    // Request B: long prompt; its first step must mix with A's decode.
    const std::uint64_t id_b =
        hooks.on_new_request(Request(0, make_prompt(100, 40), /*max_new=*/1));
    (void)id_b;
    StepPlan mixed = hooks.schedule_step();
    CHECK(mixed.has_prefill() && !mixed.decode_seq_ids.empty());
    CHECK(mixed.decode_seq_ids.size() == 1);
    CHECK(mixed.prefill_tokens == 8);

    // One dispatch: M pinned to prefill_tokens + n_decode.
    StepDispatch dispatch = hooks.dispatch_step(mixed);
    const std::uint32_t want_m =
        mixed.prefill_tokens +
        static_cast<std::uint32_t>(mixed.decode_seq_ids.size());
    CHECK(dispatch.m == want_m);
    CHECK(dispatch.m == 9);
    CHECK(dispatch.batch.tokens.size() == dispatch.m);

    // S1b lane: activation rows come from the one ragged source.
    CHECK(ninfer::runtime::step_activation_rows(dispatch) == 9);
    ninfer::runtime::validate_mixed_step_m(mixed, dispatch);

    // S1b lane: one linear dispatch per layer, every m == 9.
    CountingLinear counter;
    constexpr std::uint32_t kLayers = 4;
    const std::uint32_t launched =
        ninfer::runtime::run_step_layers(dispatch, kLayers, counter);
    CHECK(launched == kLayers);
    CHECK(counter.calls == static_cast<int>(kLayers));
    CHECK(counter.first_m == 9);
    CHECK(counter.all_equal);
    CHECK(counter.first_m == dispatch.batch.tokens.size());

    if (failures == 0) {
        std::cout << "step_forward_m: PASS (M=9, one dispatch x 4 layers)\n";
    } else {
        std::cout << "step_forward_m: " << failures << " FAILURES\n";
    }
    return failures == 0 ? 0 : 1;
}
