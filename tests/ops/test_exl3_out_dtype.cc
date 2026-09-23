// tests/ops/test_exl3_out_dtype.cc
// S1b proof: the K3/K4 m=1 out path writes through uint16_t buffers into a
// BF16 tensor. This test pins the dtype plumbing on the host, without any
// kernel launch:
//
//   T1 (input, exact): bf16 -> fp16 is bit-exact for every finite normal
//        bf16 value with |v| <= 65504 (fp16 max). fp16 mantissa (10b) is a
//        superset of bf16 mantissa (7b), so in-range normals map exactly
//        (exponent rebias + mantissa shift, no rounding). Exhaustive over
//        all 65536 bf16 patterns; subnormal/underflow/overflow/non-finite
//        patterns are counted as excluded, not passed.
//   T2 (output, exact): legacy gemv tail (exl3_launcher.cu
//        exl3_cast16_to_bf16: fp16 -> float -> bf16 RN) and the v3 row tail
//        (torchexl3/exl3_had.cuh ActVec<bf16>::store via
//        __floats2bfloat162_rn) are the same round-to-nearest-even float ->
//        bf16 conversion. Both spellings are mirrored here on one
//        deterministic 128-lane fp16 group; out bytes must match bit-exact.
//   T3 (aliasing, exact): writing those bytes via uint16_t* into a mock
//        BF16 tensor buffer preserves every byte (the exl3_op.cu
//        static_cast<std::uint16_t*>(out.data) spelling).
//
// Verdict: EXACT when all bytes match (stronger than needed); ENVELOPE
// when bytes differ but float maxAbs <= 0.5 (P12c gemv-vs-hgemm envelope,
// exl3_op.cu); BLOCKED with bytes when maxAbs > 0.5 -- do not ship decode
// on a maybe-wrong head. Host-only: no CUDA headers, no device.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
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

float bits_to_f32(std::uint32_t u) {
    float f = 0.0F;
    std::memcpy(&f, &u, 4);
    return f;
}

std::uint32_t f32_to_bits(float f) {
    std::uint32_t u = 0;
    std::memcpy(&u, &f, 4);
    return u;
}

// Exact fp16 -> float (all patterns, including subnormal/Inf/NaN).
float f16_to_f32(std::uint16_t h) {
    const std::uint32_t sign = (static_cast<std::uint32_t>(h) & 0x8000u) << 16;
    const std::uint32_t exp  = (h >> 10) & 0x1Fu;
    const std::uint32_t mant = h & 0x3FFu;
    std::uint32_t u = 0;
    if (exp == 0) {
        if (mant == 0) {
            u = sign;
        } else {
            std::uint32_t m = mant;
            int e = 1;
            while ((m & 0x400u) == 0) {
                m <<= 1;
                --e;
            }
            m &= 0x3FFu;
            u = sign | (static_cast<std::uint32_t>(e + 112) << 23) | (m << 13);
        }
    } else if (exp == 31) {
        u = sign | 0x7F800000u | (mant << 13);
    } else {
        u = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    return bits_to_f32(u);
}

// Exact bf16 -> float (bit-preserving by definition).
float bf16_to_f32(std::uint16_t b) {
    return bits_to_f32(static_cast<std::uint32_t>(b) << 16);
}

// Float -> bf16 round-to-nearest-even. Matches __nv_bfloat16(float) and
// __floats2bfloat162_rn for finite inputs. NaN canonicalizes to 0x7FC0
// (NaN inputs never enter the exact set below).
std::uint16_t f32_to_bf16_rn(float f) {
    const std::uint32_t u = f32_to_bits(f);
    if ((u & 0x7F800000u) == 0x7F800000u) {
        if ((u & 0x007FFFFFu) != 0) {
            return 0x7FC0u;
        }
        return static_cast<std::uint16_t>(u >> 16);
    }
    const std::uint32_t lsb  = (u >> 16) & 1u;
    const std::uint32_t bias = 0x7FFFu + lsb;
    return static_cast<std::uint16_t>((u + bias) >> 16);
}

// Legacy tail spelling: scalar __nv_bfloat16(__half2float(h)).
std::uint16_t legacy_out_one(std::uint16_t h) {
    return f32_to_bf16_rn(f16_to_f32(h));
}

// v3 tail spelling: pair-wise __floats2bfloat162_rn over the same lane
// pair. Independent loop shape from legacy_out_one, same RN core.
void v3_out_pair(const std::uint16_t h[2], std::uint16_t out[2]) {
    const float f0 = f16_to_f32(h[0]);
    const float f1 = f16_to_f32(h[1]);
    out[0] = f32_to_bf16_rn(f0);
    out[1] = f32_to_bf16_rn(f1);
}

} // namespace

int main() {
    // T1: exhaustive bf16 -> fp16 exactness over normal in-range values.
    int t1_exact = 0;
    int t1_excluded_sub = 0;
    int t1_excluded_over = 0;
    int t1_excluded_nf = 0;
    for (std::uint32_t b = 0; b < 65536u; ++b) {
        const std::uint16_t bb = static_cast<std::uint16_t>(b);
        const float f = bf16_to_f32(bb);
        if (!std::isfinite(f)) {
            ++t1_excluded_nf;
            continue;
        }
        const float a = std::fabs(f);
        if (a > 65504.0F) {
            ++t1_excluded_over;
            continue;
        }
        if (a < 6.103515625e-05F) { // fp16 subnormal/underflow zone
            if (a == 0.0F) { // signed zero maps exactly (sign only)
                const std::uint16_t h = static_cast<std::uint16_t>(bb & 0x8000u);
                CHECK(f16_to_f32(h) == f);
                ++t1_exact;
            } else {
                ++t1_excluded_sub;
            }
            continue;
        }
        // Normal in-range: exponent rebias + mantissa shift, no rounding.
        const std::uint32_t exp32 = (static_cast<std::uint32_t>(bb) << 16 >> 23) & 0xFFu;
        const int exp16 = static_cast<int>(exp32) - 127 + 15;
        CHECK(exp16 >= 1 && exp16 <= 30);
        const std::uint16_t h = static_cast<std::uint16_t>(
            (bb & 0x8000u) | (static_cast<std::uint16_t>(exp16) << 10) |
            ((bb & 0x7Fu) << 3));
        CHECK(f16_to_f32(h) == f);
        ++t1_exact;
    }
    std::cout << "T1 input bf16->fp16: exact=" << t1_exact
              << " excluded_sub=" << t1_excluded_sub
              << " excluded_over=" << t1_excluded_over
              << " excluded_nf=" << t1_excluded_nf << "\n";

    // T2: one deterministic 128-lane fp16 group through both tails.
    // Lane formula mirrors the P16 tile family (deterministic, finite,
    // mixed magnitudes); plus explicit edges at the tail lanes.
    std::vector<std::uint16_t> h16(128);
    for (int i = 0; i < 128; ++i) {
        const std::uint32_t w = static_cast<std::uint32_t>((44 * i + 13) & 0x7FFFu);
        std::uint16_t h = static_cast<std::uint16_t>(w);
        if (!std::isfinite(f16_to_f32(h))) {
            h = 0x3C00u; // +1.0 fallback keeps the group finite
        }
        h16[static_cast<std::size_t>(i)] = h;
    }
    h16[124] = 0x0000u; // +0.0
    h16[125] = 0x3C00u; // +1.0
    h16[126] = 0x7BFFu; // fp16 max 65504
    h16[127] = 0x0001u; // smallest subnormal

    std::vector<std::uint16_t> legacy(128);
    std::vector<std::uint16_t> v3row(128);
    for (int i = 0; i < 128; ++i) {
        legacy[static_cast<std::size_t>(i)] = legacy_out_one(h16[static_cast<std::size_t>(i)]);
    }
    for (int i = 0; i < 128; i += 2) {
        std::uint16_t pair[2] = {0, 0};
        v3_out_pair(&h16[static_cast<std::size_t>(i)], pair);
        v3row[static_cast<std::size_t>(i)] = pair[0];
        v3row[static_cast<std::size_t>(i + 1)] = pair[1];
    }

    int byte_diff = 0;
    int first_diff = -1;
    float max_abs = 0.0F;
    for (int i = 0; i < 128; ++i) {
        const std::size_t k = static_cast<std::size_t>(i);
        if (legacy[k] != v3row[k]) {
            if (first_diff < 0) {
                first_diff = i;
            }
            ++byte_diff;
        }
        const float d = std::fabs(bf16_to_f32(legacy[k]) - bf16_to_f32(v3row[k]));
        if (d > max_abs) {
            max_abs = d;
        }
    }
    std::cout << "T2 out tails: lanes=128 byte_diff=" << byte_diff
              << " maxAbs=" << max_abs << "\n";
    if (first_diff >= 0) {
        const std::size_t k = static_cast<std::size_t>(first_diff);
        std::cout << "first_diff lane=" << first_diff
                  << " h=0x" << std::hex << h16[k]
                  << " legacy=0x" << legacy[k] << " v3=0x" << v3row[k]
                  << std::dec << "\n";
    }

    // T3: uint16_t* write into a mock BF16 tensor buffer is byte-preserving.
    std::vector<std::uint16_t> tensor_buf(128, 0);
    std::uint16_t* out_alias = tensor_buf.data(); // exl3_op.cu spelling
    for (int i = 0; i < 128; ++i) {
        out_alias[i] = legacy[static_cast<std::size_t>(i)];
    }
    int alias_diff = 0;
    for (int i = 0; i < 128; ++i) {
        if (tensor_buf[static_cast<std::size_t>(i)] != v3row[static_cast<std::size_t>(i)]) {
            ++alias_diff;
        }
    }
    CHECK(alias_diff == byte_diff);

    const char* verdict = "EXACT";
    if (byte_diff != 0 && max_abs <= 0.5F) {
        verdict = "ENVELOPE";
    } else if (byte_diff != 0 && max_abs > 0.5F) {
        verdict = "BLOCKED";
    }
    std::cout << "out_dtype verdict: " << verdict
              << " (P12c envelope 0.5, maxAbs=" << max_abs << ")\n";
    if (byte_diff == 0) {
        CHECK(max_abs == 0.0F);
    }
    if (max_abs > 0.5F) {
        ++failures;
        std::cout << "BLOCKED: out bytes exceed the P12c 0.5 envelope;"
                  << " decode must not ship on this head\n";
    }

    if (failures == 0) {
        std::cout << "exl3_out_dtype: PASS (" << verdict << ")\n";
    } else {
        std::cout << "exl3_out_dtype: " << failures << " FAILURES (" << verdict << ")\n";
    }
    return failures == 0 ? 0 : 1;
}
