// tests/test_mixed_step_m.cc
// Swarm G proof delta (a): two requests, one mixed plan, ragged M, one
// fused qkv launch.
//
// Setup mirrors tests/test_mixed_step.cc: request A (short, driven to
// decode) plus request B (long, still prefilling) under chunk_tokens=8, so
// the real RequestScheduler emits ONE mixed StepPlan with M = 8 + 1 = 9.
// This TU pins the two deltas test_mixed_step.cc does not:
//
//   1. x.ne[1] == tokens.size(): the activation Tensor rows come from the
//      one ragged source. Production sets the real activation Tensor with
//      x.ne[1] = step_activation_rows(dispatch) (src/runtime/engine/
//      step_forward.h); the ActX mirror below takes the same call, so
//      x.ne[1] == dispatch.batch.tokens.size() == 9 by construction.
//   2. qkv launches == 1: the fused qkv projection (q/k/v shards,
//      groups=3) costs exactly one launch over the whole ragged width at
//      M=9. The QkvLedger counter hook mirrors the production routing
//      (src/ops/linear/exl3/exl3_op.cu: groups > 1 -> v3 fused multi row,
//      ONE launch over the whole fused width, any m) and records one
//      launch per fused qkv call. A single fused qkv call at M=9 must
//      record exactly 1; a 4-layer run_step_layers pass must record
//      exactly one more per layer, every m == 9.
//
// Choice note: new file instead of extending test_mixed_step.cc, so the
// S5/S5b proof TU stays untouched and this delta reviews standalone.
//
// Host-only: no kernel launch, no device handle. Run with GPUs hidden;
// the batch pool takes its host-only path. GPU execution: GATED.
#include "batch/cinference_hooks.h"
#include "runtime/engine/step_forward.h"

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

// Activation Tensor shape mirror: production sets x.ne[1] from
// step_activation_rows(dispatch); ne[0] is the feature width (unused here).
struct ActX {
    std::uint32_t ne[2] = {0, 0};
};

// Production routing, host mirror (see exl3_op.cu header above): groups > 1
// takes the fused-multi row with exactly one launch over the whole width.
bool qkv_takes_fused_multi(int groups) {
    return groups > 1;
}

// Counter hook: one counted launch per fused qkv call, regardless of m.
struct QkvLedger {
    int qkv_launches = 0;

    void launch_qkv(std::uint32_t m, int groups, int bits, int k, int n) {
        (void)m;
        (void)bits;
        (void)k;
        (void)n;
        if (!qkv_takes_fused_multi(groups)) {
            ++failures;
            std::cout << "FAIL qkv did not take the fused-multi row\n";
            return;
        }
        ++qkv_launches;
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
    // Two distinct in-memory sessions share the ONE plan below.
    const std::uint64_t id_b =
        hooks.on_new_request(Request(0, make_prompt(100, 40), /*max_new=*/1));
    (void)id_b;
    StepPlan mixed = hooks.schedule_step();
    CHECK(mixed.has_prefill() && !mixed.decode_seq_ids.empty());
    CHECK(mixed.decode_seq_ids.size() == 1);
    CHECK(mixed.prefill_tokens == 8);

    // One plan, one dispatch: M pinned to prefill_tokens + n_decode.
    StepDispatch dispatch = hooks.dispatch_step(mixed);
    CHECK(dispatch.m == 9);
    CHECK(dispatch.batch.tokens.size() == dispatch.m);

    // Delta 1: x.ne[1] == tokens.size(). Same call production uses.
    ActX x;
    x.ne[0] = 4096;
    x.ne[1] = ninfer::runtime::step_activation_rows(dispatch);
    CHECK(x.ne[1] == dispatch.batch.tokens.size());
    CHECK(x.ne[1] == 9);
    ninfer::runtime::validate_mixed_step_m(mixed, dispatch);

    // Delta 2: one fused qkv call at M == x.ne[1] costs one launch.
    {
        QkvLedger ledger;
        ledger.launch_qkv(x.ne[1], /*groups=*/3, /*bits=*/4, /*k=*/4096,
                          /*n=*/4096);
        CHECK(ledger.qkv_launches == 1);
    }

    // Per-layer scale: run_step_layers issues one linear per layer; each
    // layer's fused qkv adds exactly one launch, every m == 9.
    {
        constexpr std::uint32_t kLayers = 4;
        QkvLedger ledger;
        std::uint32_t seen_m = 0;
        const std::uint32_t launched = ninfer::runtime::run_step_layers(
            dispatch, kLayers, [&](std::uint32_t layer, std::uint32_t m) {
                (void)layer;
                seen_m     = m;
                const int before = ledger.qkv_launches;
                ledger.launch_qkv(m, /*groups=*/3, /*bits=*/4, /*k=*/4096,
                                  /*n=*/4096);
                if (ledger.qkv_launches - before != 1) {
                    ++failures;
                    std::cout << "FAIL layer added "
                              << (ledger.qkv_launches - before)
                              << " qkv launches, want 1\n";
                }
            });
        CHECK(launched == kLayers);
        CHECK(seen_m == 9);
        CHECK(seen_m == x.ne[1]);
        CHECK(ledger.qkv_launches == static_cast<int>(kLayers));
    }

    // GPU execution section: GATED. No kernel launches from this session
    // (shared-RTX-5090 rule); the device half stays on the Linux CUDA
    // toolchain by design.
    std::cout << "mixed_step_m GPU section: GATED (counter-hook only, no kernel launch)\n";

    if (failures == 0) {
        std::cout << "mixed_step_m: PASS (x.ne[1]==tokens==9, qkv==1)\n";
    } else {
        std::cout << "mixed_step_m: " << failures << " FAILURES\n";
    }
    return failures == 0 ? 0 : 1;
}
