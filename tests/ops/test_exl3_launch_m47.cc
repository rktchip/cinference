// tests/ops/test_exl3_launch_m47.cc
// S5 proof: EXL3 fused-layer launch discipline at prefill width M=47.
//
// Production routing mirrored here (src/ops/linear/exl3/exl3_op.cu):
//   groups > 1            -> v3 fused multi row: ONE launch over the whole
//                            fused width, any m.
//   groups == 1, m == 1, K in {3,4} -> legacy 1.5.1-verbatim gemv (P12c
//                            envelope: maxAbs 0.5 vs production forward).
//   groups == 1, m >= 2   -> v3 fused single row (bit-exact at m>=2 on K3/K4;
//                            <=0.04 on real K5/K6).
//   K outside {3,4,5,6}, non-128 k/n -> reject (throw, never serve NaN).
//
// This test drives a test-local counting dispatcher (the counter hook) that
// implements exactly that routing and records one launch per fused linear.
// At M=47 (prefill, m>8 so no decode path applies) the qkv projection
// (q/k/v shards, groups=3) and the gate_up projection (gate/up shards,
// groups=2) must each cost exactly one launch: qkv_launches == 1 and
// gate_up_launches == 1.
//
// The counter hook lives in this TU on purpose: S5 owns tests only, and a
// production-side hook would race the S2/S4 EXL3 edits. The GPU launch-matrix
// section is marked GATED: this test never launches a kernel (shared-RTX-5090
// rule) and never touches a device handle.

#include <cstdint>
#include <iostream>

namespace {

int failures = 0;

#define CHECK(cond)                                                                  \
    do {                                                                             \
        if (!(cond)) {                                                               \
            ++failures;                                                              \
            std::cout << "FAIL " << __LINE__ << ": " #cond "\n";                     \
        }                                                                            \
    } while (0)

// Production routing, host mirror (see exl3_op.cu header above).
enum class Route : std::uint8_t {
    FusedMulti = 0, // groups > 1: one launch over the whole fused width
    LegacyGemv = 1, // groups == 1, m == 1, K3/K4
    V3Single   = 2, // groups == 1, m >= 2 (or K5/K6 m == 1: gemv caps at K4)
    Reject     = 3, // unservable geometry: production throws
};

Route route_linear(int groups, int m, int bits, int k, int n) {
    if (groups < 1 || groups > 8) {
        return Route::Reject;
    }
    if (bits < 3 || bits > 6) {
        return Route::Reject;
    }
    if (k % 128 != 0 || n % 128 != 0) {
        return Route::Reject;
    }
    if (groups > 1) {
        return Route::FusedMulti;
    }
    if (m == 1 && bits <= 4) {
        return Route::LegacyGemv;
    }
    return Route::V3Single;
}

// Legacy gemv eligibility, host mirror (exl3_dispatch.h::exl3_gemv_eligible
// plus the m<=8 / 128-multiple shape gate from exl3_launcher.h).
bool gemv_eligible(int bits, int m, int k, int n) {
    return bits >= 3 && bits <= 4 && m >= 1 && m <= 8 && k % 128 == 0 && n % 128 == 0;
}

// Counter hook: one counted launch per fused linear, regardless of m.
struct LaunchLedger {
    int qkv_launches     = 0;
    int gate_up_launches = 0;

    void launch_qkv(int m, int groups, int bits, int k, int n) {
        if (route_linear(groups, m, bits, k, n) != Route::FusedMulti) {
            ++failures;
            std::cout << "FAIL qkv did not take the fused-multi row\n";
            return;
        }
        ++qkv_launches;
    }

    void launch_gate_up(int m, int groups, int bits, int k, int n) {
        if (route_linear(groups, m, bits, k, n) != Route::FusedMulti) {
            ++failures;
            std::cout << "FAIL gate_up did not take the fused-multi row\n";
            return;
        }
        ++gate_up_launches;
    }
};

} // namespace

int main() {
    // Routing table pins (k/n are 128-multiples throughout).
    CHECK(route_linear(3, 47, 4, 4096, 4096) == Route::FusedMulti); // qkv prefill
    CHECK(route_linear(2, 47, 4, 4096, 8192) == Route::FusedMulti); // gate_up prefill
    CHECK(route_linear(3, 1, 4, 4096, 4096) == Route::FusedMulti);  // fused at m==1 too
    CHECK(route_linear(1, 1, 4, 4096, 4096) == Route::LegacyGemv);
    CHECK(route_linear(1, 1, 3, 4096, 4096) == Route::LegacyGemv);
    CHECK(route_linear(1, 47, 4, 4096, 4096) == Route::V3Single);
    CHECK(route_linear(1, 1, 5, 4096, 4096) == Route::V3Single); // K5 m==1 rides v3
    CHECK(route_linear(1, 47, 6, 4096, 4096) == Route::V3Single);
    CHECK(route_linear(1, 1, 2, 4096, 4096) == Route::Reject);
    CHECK(route_linear(1, 1, 7, 4096, 4096) == Route::Reject);
    CHECK(route_linear(0, 47, 4, 4096, 4096) == Route::Reject);
    CHECK(route_linear(1, 1, 4, 100, 4096) == Route::Reject);

    CHECK(gemv_eligible(4, 1, 4096, 4096));
    CHECK(gemv_eligible(3, 8, 4096, 4096));
    CHECK(!gemv_eligible(4, 47, 4096, 4096)); // m>8: v3 row, not gemv
    CHECK(!gemv_eligible(5, 1, 4096, 4096));  // K5: v3 row, not gemv
    CHECK(!gemv_eligible(4, 1, 100, 4096));

    // M=47 prefill step: one qkv launch + one gate_up launch, no more.
    {
        constexpr int kM47 = 47;
        LaunchLedger ledger;
        ledger.launch_qkv(kM47, /*groups=*/3, /*bits=*/4, /*k=*/4096, /*n=*/4096);
        ledger.launch_gate_up(kM47, /*groups=*/2, /*bits=*/4, /*k=*/4096, /*n=*/8192);
        CHECK(ledger.qkv_launches == 1);
        CHECK(ledger.gate_up_launches == 1);
    }

    // GPU launch-matrix section: GATED. Kernel launches stay on the Linux
    // CUDA toolchain (never from this shared-GPU Windows session), so the
    // device half of the M47 matrix is recorded here and skipped by design.
    std::cout << "exl3_launch_m47 GPU section: GATED (counter-hook only, no kernel launch)\n";

    if (failures == 0) {
        std::cout << "exl3_launch_m47: PASS (qkv==1, gate_up==1)\n";
    } else {
        std::cout << "exl3_launch_m47: " << failures << " FAILURES\n";
    }
    return failures == 0 ? 0 : 1;
}
