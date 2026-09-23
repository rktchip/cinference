// tests/ops/test_exl3_dequant.cc
// S5 proof: EXL3 trellis dequant front-end, bit-exact vs recorded 1.5.1 vectors.
//
// What this pins: the integer front-end shared by every EXL3 serve path
// (legacy m==1 gemv and the v3 fused row both dequant through it):
//   1. trellis word extraction  (exl3_dq.cuh::dq  index math + funnel shift)
//   2. cb2/mul1 integer core    (exl3_codebook.cuh::decode_3inst<cb=2> head:
//                                x = w * 0x83DCD12D, dp4a byte-sum + 0x6400)
//   3. mul1 fp16 tail           (decode_mul1_product_2: h * k_inv + k_bias,
//                                k_inv = 0x1eee, k_bias = 0xc931, one rounding)
// The recorded vectors below were produced by an independent Python
// implementation of those same formulas (exact rational arithmetic for the
// fp16 tail, single rounding) and are embedded as constants, so this test
// compares the host C++ mirror against FIXED 1.5.1-derived values, not
// against itself. Small groups only: one trellis column per case
// (K=4: 32 u32 words, K=3: 24 u32 words), t_offset corners + mid.
//
// Byte-sum signedness note: the port claims dp4a(x, 0x01010101, 0x6400) is
// bit-identical to the previous vabsdiff4(x, 0, acc), which sums UNSIGNED
// bytes; the checkpoint calibration confirms it (k_bias = (-1024 - 510) *
// k_inv, i.e. the unsigned byte mean 4 * 127.5 = 510). The mirror therefore
// sums unsigned bytes. A signed interpretation would shift every recorded
// sum (e.g. case K4/t0 0x6637 -> 0x6437) and fail loudly here.
//
// Host-only: no CUDA headers, no device, no __int128. Runs on any toolchain.

#include <cstdint>
#include <iostream>
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

// ---- mirror of exl3_dq.cuh::fshift (verbatim integer semantics) ----
std::uint32_t host_fshift(std::uint32_t b, std::uint32_t a, int shift) {
    const std::uint64_t merged = (static_cast<std::uint64_t>(a) << 32) | b;
    return static_cast<std::uint32_t>(merged >> shift);
}

// CUDA __funnelshift_r(lo, hi, s) as used by dq<>: low word of ((hi:lo) >> s).
std::uint32_t host_funnel_r(std::uint32_t lo, std::uint32_t hi, int shift) {
    return host_fshift(lo, hi, shift);
}

// ---- mirror of dq<bits, cb> word extraction (index math + funnel + mask) ----
struct DqProbe {
    std::uint16_t w = 0;
    int i0          = 0;
    int i1          = 0;
    int s0          = 0;
};

DqProbe host_dq_word(const std::uint32_t* ptr, int words_per_col, int t_offset, int bits) {
    const int b0 = t_offset * bits + bits - 16 + 256 * bits; // bit index, start of word0
    const int b1 = b0 + 16;                                  // bit index, end of word0
    const int i0 = b0 / 32;
    const int i1 = (b1 - 1) / 32;
    const int s0 = (i1 + 1) * 32 - b1;
    const std::uint32_t a = ptr[i0 % words_per_col];
    const std::uint32_t b = ptr[i1 % words_per_col];
    DqProbe probe;
    probe.i0 = i0;
    probe.i1 = i1;
    probe.s0 = s0;
    probe.w  = static_cast<std::uint16_t>(host_funnel_r(b, a, s0) & 0xFFFFu);
    return probe;
}

// ---- mirror of the decode_3inst<cb=2> integer core ----
// __dp4a(x, 0x01010101, acc) with the unsigned-byte semantics documented above.
std::uint32_t host_dp4a_byte_sum(std::uint32_t x, std::uint32_t acc) {
    return acc + (x & 0xFFu) + ((x >> 8) & 0xFFu) + ((x >> 16) & 0xFFu) + (x >> 24);
}

struct Mul1Probe {
    std::uint32_t x   = 0; // w * 0x83DCD12D, wrapped
    std::uint32_t sum = 0; // dp4a product + 0x6400
};

Mul1Probe host_mul1_probe(std::uint16_t w) {
    Mul1Probe probe;
    probe.x   = static_cast<std::uint32_t>(w) * 0x83DCD12Du;
    probe.sum = host_dp4a_byte_sum(probe.x, 0x6400u);
    return probe;
}

// ---- exact fp16 fused multiply-add, round-to-nearest-even, single rounding ----
// All operands are finite fp16; intermediate products of 11-bit significands
// fit int64 with wide margin at these magnitudes, so the sum is exact and the
// single final rounding matches __hfma bit-for-bit. Subnormal results round
// to subnormal/zero; infinities/NaNs never occur in these vectors.
std::uint16_t host_fp16_fma(std::uint16_t a_bits, std::uint16_t b_bits, std::uint16_t c_bits) {
    struct Dec {
        int neg;         // 0/1
        int exp;         // unbiased binary exponent of sig
        std::uint64_t sig; // significand (11 bits incl. hidden 1; raw mantissa if subnormal)
    };
    const auto decode = [](std::uint16_t h) {
        const int e = (h >> 10) & 0x1F;
        const int m = h & 0x3FF;
        if (e == 0) {
            return Dec{(h >> 15) & 1, -24, static_cast<std::uint64_t>(m)};
        }
        return Dec{(h >> 15) & 1, e - 15 - 10, static_cast<std::uint64_t>(1024 + m)};
    };
    const Dec a = decode(a_bits);
    const Dec b = decode(b_bits);
    const Dec c = decode(c_bits);
    const std::int64_t psign = (a.neg == b.neg) ? 1 : -1;
    const std::uint64_t psig = a.sig * b.sig; // <= 2047^2, exact
    const int pexp           = a.exp + b.exp;
    const int emin           = pexp < c.exp ? pexp : c.exp;
    const int sh_p           = pexp - emin;
    const int sh_c           = c.exp - emin;
    if (sh_p > 60 || sh_c > 60) {
        return 0xFFFFu; // out of exact range: never happens for these vectors; fail loudly below
    }
    const std::int64_t acc = psign * static_cast<std::int64_t>(psig << sh_p) +
                             (c.neg ? -1 : 1) * static_cast<std::int64_t>(c.sig << sh_c);
    if (acc == 0) {
        return 0x0000u;
    }
    const int neg      = acc < 0 ? 1 : 0;
    std::uint64_t mag  = acc < 0 ? static_cast<std::uint64_t>(-(acc + 1)) + 1u : static_cast<std::uint64_t>(acc);
    int msb            = 63;
    while (msb > 0 && ((mag >> msb) & 1u) == 0) { --msb; }
    const int true_exp = emin + msb; // exponent of a normalized [1024, 2048) significand
    if (true_exp >= -14) {
        const int biased = true_exp + 15;
        if (biased >= 31) {
            return static_cast<std::uint16_t>((neg << 15) | 0x7BFFu);
        }
        const int shift = msb - 10;
        std::uint64_t sig11;
        std::uint64_t dropped = 0u;
        std::uint64_t half    = 0u;
        if (shift >= 0) {
            sig11   = mag >> shift;
            dropped = mag & ((shift == 64) ? ~0ULL : ((1ULL << shift) - 1u));
            half    = (shift == 0) ? 0u : (1ULL << (shift - 1));
        } else {
            sig11 = mag << (-shift); // tiny normal: exact, no dropped bits
        }
        if (shift > 0 && (dropped > half || (dropped == half && (sig11 & 1u) != 0u))) {
            ++sig11;
        }
        int out_exp = biased;
        if (sig11 == 2048u) {
            sig11   = 1024u;
            out_exp = biased + 1;
            if (out_exp >= 31) {
                return static_cast<std::uint16_t>((neg << 15) | 0x7BFFu);
            }
        }
        return static_cast<std::uint16_t>((neg << 15) | (out_exp << 10) |
                                          static_cast<int>(sig11 - 1024u));
    }
    // Subnormal: m10 = RN(mag * 2^(emin+24)), a 10-bit mantissa at 2^-24.
    const int k = emin + 24;
    std::uint64_t m10;
    if (k >= 0) {
        // Left shift: overflows 10 bits only by rounding into normal 2^-14.
        if (msb + k >= 10) {
            return static_cast<std::uint16_t>((neg << 15) | (15 << 10));
        }
        m10 = mag << k;
    } else if (-k > 64) {
        return static_cast<std::uint16_t>(neg << 15); // underflows to zero
    } else {
        const int rshift = -k;
        m10 = (rshift >= 64) ? 0u : (mag >> rshift);
        const std::uint64_t dropped =
            (rshift >= 64) ? mag : (mag & ((1ULL << rshift) - 1u));
        const std::uint64_t half = (rshift == 0) ? 0u : (1ULL << (rshift - 1));
        if (rshift > 0 && (dropped > half || (dropped == half && (m10 & 1u) != 0u))) {
            ++m10;
        }
    }
    if (m10 >= 1024u) {
        return static_cast<std::uint16_t>((neg << 15) | (15 << 10)); // rounded into normal 2^-14
    }
    return static_cast<std::uint16_t>((neg << 15) | static_cast<int>(m10));
}

// Full single-value mirror: trellis column -> decoded fp16 bits (cb2/mul1).
std::uint16_t host_decode_mul1(const std::uint32_t* ptr, int words_per_col, int t_offset,
                               int bits) {
    const DqProbe probe   = host_dq_word(ptr, words_per_col, t_offset, bits);
    const Mul1Probe mul1  = host_mul1_probe(probe.w);
    const std::uint16_t h = static_cast<std::uint16_t>(mul1.sum & 0xFFFFu);
    return host_fp16_fma(h, 0x1EEEu, 0xC931u);
}

// One trellis column shared by the K4 cases; the K3 cases use its 24-word prefix.
// ptr[i] = (i * 0x9E3779B1 + 0x12345678) mod 2^32 (golden-ratio hash: fixed, spread bits).
const std::uint32_t kTrellisColumn[32] = {
    0x12345678u, 0xB06BD029u, 0x4EA349DAu, 0xECDAC38Bu, 0x8B123D3Cu, 0x2949B6EDu,
    0xC781309Eu, 0x65B8AA4Fu, 0x03F02400u, 0xA2279DB1u, 0x405F1762u, 0xDE969113u,
    0x7CCE0AC4u, 0x1B058475u, 0xB93CFE26u, 0x577477D7u, 0xF5ABF188u, 0x93E36B39u,
    0x321AE4EAu, 0xD0525E9Bu, 0x6E89D84Cu, 0x0CC151FDu, 0xAAF8CBAEu, 0x4930455Fu,
    0xE767BF10u, 0x859F38C1u, 0x23D6B272u, 0xC20E2C23u, 0x6045A5D4u, 0xFE7D1F85u,
    0x9CB49936u, 0x3AEC12E7u,
};

struct RecordedCase {
    int bits;
    int t_offset;
    int i0;
    int i1;
    int s0;
    std::uint16_t w;
    std::uint32_t x;
    std::uint32_t sum;
    std::uint16_t out;
};

// Recorded 1.5.1 vectors: (trellis column above) -> each stage, bit-exact.
const RecordedCase kCases[] = {
    // K=4, cb=2 (mul1): W = bits*256/32 = 32 words per column.
    {4, 0, 31, 32, 28, 0x2E71u, 0xE20E6ADDu, 0x00006637u, 0x3625u},
    {4, 1, 31, 32, 24, 0xE712u, 0x8646502Au, 0x00006546u, 0xBCFDu},
    {4, 127, 47, 47, 0, 0x77D7u, 0x63AF97CBu, 0x00006674u, 0x3A60u},
    {4, 255, 63, 63, 0, 0x12E7u, 0x82F5E99Bu, 0x000066FBu, 0x3ED7u},
    // K=3, cb=2 (mul1): W = 24 words per column (column prefix above).
    {3, 0, 23, 24, 29, 0x2AF8u, 0xF83C0598u, 0x000065D1u, 0xB4E7u},
    {3, 5, 24, 24, 14, 0x48D1u, 0xC21B6DBDu, 0x00006607u, 0x2B90u},
    {3, 255, 47, 47, 0, 0x455Fu, 0x7352C0B3u, 0x00006638u, 0x3640u},
};

} // namespace

int main() {
    for (const RecordedCase& rc : kCases) {
        const int words = rc.bits * 256 / 32;
        const DqProbe probe =
            host_dq_word(kTrellisColumn, words, rc.t_offset, rc.bits);
        CHECK(probe.i0 == rc.i0);
        CHECK(probe.i1 == rc.i1);
        CHECK(probe.s0 == rc.s0);
        CHECK(probe.i1 - probe.i0 <= 1); // 16-bit window spans at most two u32 words
        CHECK(probe.s0 >= 0 && probe.s0 < 32);
        CHECK(probe.w == rc.w);
        const Mul1Probe mul1 = host_mul1_probe(probe.w);
        CHECK(mul1.x == rc.x);
        CHECK(mul1.sum == rc.sum);
        const std::uint16_t h   = static_cast<std::uint16_t>(mul1.sum & 0xFFFFu);
        const std::uint16_t out = host_fp16_fma(h, 0x1EEEu, 0xC931u);
        CHECK(out == rc.out);
        CHECK(host_decode_mul1(kTrellisColumn, words, rc.t_offset, rc.bits) == rc.out);
    }
    // Structural pin: every recorded dp4a sum sits in the mul1 codebook band
    // [0x6400, 0x6400 + 4*255] = [0x6400, 0x67FC].
    for (const RecordedCase& rc : kCases) {
        CHECK(rc.sum >= 0x6400u && rc.sum <= 0x67FCu);
    }
    if (failures == 0) {
        std::cout << "exl3_dequant: PASS (7 recorded 1.5.1 vectors, bit-exact)\n";
    } else {
        std::cout << "exl3_dequant: " << failures << " FAILURES\n";
    }
    return failures == 0 ? 0 : 1;
}
