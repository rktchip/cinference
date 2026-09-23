// tests/test_mtp_spec_gate.cpp
// S7 proof: decode-only steps RUN ninfer MTP-3 draft+verify, mixed
// prefill+decode steps SKIP speculation.
//
// Host-only decision test. It scripts StepPlan phase mixes (decode-only,
// mixed, prefill-only, empty) as plain S1 StepPlan structs, runs the S7
// predicate from models/qwen3_5/execution/mtp_spec_gate.h on each, and logs
// the skip-vs-run decision plus the resolved draft window. Every case asserts
// the expected route, the MTP-3-or-zero window, agreement between the
// StepPlan-template and bool overloads, and agreement with IsGraphSafe (which
// encodes the same decode-only rule for the captured-graph path).
//
// GPU execution of the RUN arm (real draft+verify numerics) is GATED and
// stays with tests/ops (test_mtp_round, test_speculative_round).

#include "batch/scheduler.h"
#include "models/qwen3_5/execution/mtp_spec_gate.h"

#include <cstdint>
#include <iostream>
#include <string>
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

using ninfer::batch::PrefillSlice;
using ninfer::batch::StepPlan;
using ninfer::models::qwen3_5::execution::MtpSpecDraftWindow;
using ninfer::models::qwen3_5::execution::MtpSpeculationRoute;
using ninfer::models::qwen3_5::execution::MtpStepPhaseMix;
using ninfer::models::qwen3_5::execution::SelectMtpSpeculationForStep;
using ninfer::models::qwen3_5::execution::SelectMtpSpeculationRoute;
using ninfer::models::qwen3_5::execution::ShouldRunMtpSpeculation;
using ninfer::models::qwen3_5::execution::kMtpSpecDecodeDrafts;
using ninfer::models::qwen3_5::execution::kMtpSpecDecodeWidth;
using ninfer::models::qwen3_5::execution::kMtpSpecFrameDomainDrafts;

StepPlan decode_only(std::uint64_t first_id, std::size_t n) {
    StepPlan plan;
    for (std::size_t i = 0; i < n; ++i) {
        plan.decode_seq_ids.push_back(first_id + i);
    }
    return plan;
}

PrefillSlice make_slice(std::uint64_t seq_id, std::uint32_t offset, std::uint32_t count) {
    PrefillSlice s;
    s.seq_id = seq_id;
    s.offset = offset;
    s.count  = count;
    return s;
}

void run_case(const std::string& name, const StepPlan& plan, bool expect_run) {
    const MtpSpeculationRoute route  = SelectMtpSpeculationForStep(plan);
    const bool run                   = (route == MtpSpeculationRoute::RunDraftVerify);
    const std::uint32_t window       = MtpSpecDraftWindow(route);
    std::cout << "case " << name << " decode=" << plan.decode_seq_ids.size()
              << " prefill=" << plan.prefill.size() << " -> " << (run ? "RUN" : "SKIP")
              << " window=" << window << "\n";
    CHECK(run == expect_run);
    CHECK(window == (expect_run ? kMtpSpecDecodeDrafts : 0u));
    // Bool overload agrees with the StepPlan branch on every mix.
    const MtpStepPhaseMix mix{!plan.decode_seq_ids.empty(), plan.has_prefill()};
    CHECK(ShouldRunMtpSpeculation(mix) == run);
    CHECK(SelectMtpSpeculationRoute(mix) == route);
    // Same decode-only rule as the captured-graph path (no policy change).
    CHECK(ninfer::batch::IsGraphSafe(plan) == run);
}

} // namespace

int main() {
    // S7 window pins: MTP-3, width 4, strictly below the 10-wide frame max.
    CHECK(kMtpSpecDecodeDrafts == 3u);
    CHECK(kMtpSpecDecodeWidth == 4u);
    CHECK(kMtpSpecFrameDomainDrafts == 10u);
    CHECK(kMtpSpecDecodeDrafts < kMtpSpecFrameDomainDrafts);

    run_case("decode-only/B1", decode_only(7, 1), true);
    run_case("decode-only/B8", decode_only(101, 8), true);

    {
        StepPlan mixed = decode_only(7, 1);
        mixed.prefill.push_back(make_slice(9, 0, 8));
        mixed.prefill_tokens = 8;
        run_case("mixed/1+1slice", mixed, false);
    }
    {
        StepPlan mixed = decode_only(21, 7);
        mixed.prefill.push_back(make_slice(31, 0, 8));
        mixed.prefill.push_back(make_slice(33, 0, 8));
        mixed.prefill_tokens = 16;
        run_case("mixed/7+2slices", mixed, false);
    }
    {
        StepPlan prefill_only;
        prefill_only.prefill.push_back(make_slice(41, 0, 8));
        prefill_only.prefill_tokens = 8;
        run_case("prefill-only", prefill_only, false);
    }
    {
        StepPlan empty;
        run_case("empty", empty, false);
    }

    if (failures == 0) {
        std::cout << "mtp_spec_gate: PASS (decode-only runs MTP-3, mixed skips)\n";
    } else {
        std::cout << "mtp_spec_gate: " << failures << " FAILURES\n";
    }
    return failures == 0 ? 0 : 1;
}
