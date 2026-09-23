#pragma once
// ladder_trace.h
// ACTIVATION LADDER diagnostics (diagnose-first, env-gated trace only).
//
// When CINFERENCE_LADDER_TRACE is set (any value), ladder_dump() snapshots a
// BF16 activation tensor to host, reduces it to absmax/mean/zero/nan/inf
// stats (global plus per-column for the first 8 columns), and prints one
// [ladder] line per tensor plus [ladder-col] lines to stderr (and appends to
// $CINFERENCE_LADDER_FILE when set). When the variable is unset every call
// is a no-op: no device reads, no stream sync, no numeric effect on the
// forward pass. Diagnostics never throw: failures print a [ladder-warn]
// line and return, so tracing cannot break production forwards.

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace ninfer::models::qwen3_5::execution::ladder {

inline bool ladder_armed() noexcept { return std::getenv("CINFERENCE_LADDER_TRACE") != nullptr; }

inline float ladder_bf16_to_float(std::uint16_t w) noexcept {
    std::uint32_t bits = static_cast<std::uint32_t>(w) << 16;
    float out          = 0.0F;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

// Trace-only snapshot of a BF16 [H, T] activation. Reads device memory,
// synchronizes the stream, prints stats. No writes to device state.
inline void ladder_dump(const ::ninfer::Tensor& t, const char* tag, int layer,
                        cudaStream_t stream) noexcept {
    if (!ladder_armed()) { return; }
    try {
        if (tag == nullptr) { tag = "?"; }
        if (t.data == nullptr) {
            std::fprintf(stderr, "[ladder] tag=%s layer=%d NULL tensor\n", tag, layer);
            std::fflush(stderr);
            return;
        }
        if (t.dtype != ::ninfer::DType::BF16) {
            std::fprintf(stderr, "[ladder] tag=%s layer=%d SKIP non-BF16 dtype=%d\n", tag,
                         layer, static_cast<int>(t.dtype));
            std::fflush(stderr);
            return;
        }
        const std::int64_t rows = t.ne[0] < 0 ? 0 : t.ne[0];
        const std::int64_t cols = t.ne[1] < 0 ? 0 : t.ne[1];
        const std::int64_t n    = rows * cols;
        if (n <= 0 || !t.is_contiguous()) {
            std::fprintf(stderr, "[ladder] tag=%s layer=%d SKIP shape=[%d,%d] contiguous=%d\n",
                         tag, layer, (int)t.ne[0], (int)t.ne[1], (int)t.is_contiguous());
            std::fflush(stderr);
            return;
        }
        std::vector<std::uint16_t> host(static_cast<std::size_t>(n));
        if (cudaMemcpyAsync(host.data(), t.data, static_cast<std::size_t>(n) * 2,
                            cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
            std::fprintf(stderr, "[ladder] tag=%s layer=%d WARN copy-launch failed\n", tag,
                         layer);
            std::fflush(stderr);
            return;
        }
        if (cudaStreamSynchronize(stream) != cudaSuccess) {
            std::fprintf(stderr, "[ladder] tag=%s layer=%d WARN stream-sync failed\n", tag,
                         layer);
            std::fflush(stderr);
            return;
        }
        double absmax = 0.0;
        double sum    = 0.0;
        long finite   = 0;
        long zeros    = 0;
        long nans     = 0;
        long infs     = 0;
        for (std::int64_t i = 0; i < n; ++i) {
            const std::uint16_t w = host[static_cast<std::size_t>(i)];
            if (w == 0x0000u || w == 0x8000u) { ++zeros; }
            const float x = ladder_bf16_to_float(w);
            if (x != x) {
                ++nans;
                continue;
            }
            const double ax = std::fabs(static_cast<double>(x));
            if (ax > 1.0e30) {
                ++infs;
                if (ax > absmax) { absmax = ax; }
                continue;
            }
            ++finite;
            sum += static_cast<double>(x);
            if (ax > absmax) { absmax = ax; }
        }
        const double mean = finite > 0 ? sum / static_cast<double>(finite) : 0.0;
        std::fprintf(stderr,
                     "[ladder] tag=%s layer=%d H=%d T=%d N=%lld absmax=%.6g mean=%.6g "
                     "zeros=%ld nan=%ld inf=%ld\n",
                     tag, layer, (int)rows, (int)cols, (long long)n, absmax, mean, zeros,
                     nans, infs);
        // Per-token columns (first 8): lets the probe match exact token ids
        // against raw checkpoint rows for the embed-bind check. Mirrored to
        // $CINFERENCE_LADDER_FILE alongside the global line.
        std::FILE* fp = nullptr;
        if (const char* path = std::getenv("CINFERENCE_LADDER_FILE");
            path != nullptr && path[0] != '\0') {
            fp = std::fopen(path, "a"); // best-effort; null is silent
        }
        const std::int64_t ccap = cols < 8 ? cols : 8;
        for (std::int64_t c = 0; c < ccap; ++c) {
            double cmax = 0.0;
            double csum = 0.0;
            long cfin = 0, cz = 0, cn = 0, ci = 0;
            for (std::int64_t r = 0; r < rows; ++r) {
                const std::uint16_t w =
                    host[static_cast<std::size_t>(c * rows + r)];
                if (w == 0x0000u || w == 0x8000u) { ++cz; }
                const float x = ladder_bf16_to_float(w);
                if (x != x) {
                    ++cn;
                    continue;
                }
                const double ax = std::fabs(static_cast<double>(x));
                if (ax > 1.0e30) {
                    ++ci;
                    if (ax > cmax) { cmax = ax; }
                    continue;
                }
                ++cfin;
                csum += static_cast<double>(x);
                if (ax > cmax) { cmax = ax; }
            }
            const double cmean = cfin > 0 ? csum / static_cast<double>(cfin) : 0.0;
            std::fprintf(stderr,
                         "[ladder-col] tag=%s col=%d absmax=%.6g mean=%.6g zeros=%ld "
                         "nan=%ld inf=%ld\n",
                         tag, (int)c, cmax, cmean, cz, cn, ci);
            if (fp != nullptr) {
                std::fprintf(fp,
                             "[ladder-col] tag=%s col=%d absmax=%.6g mean=%.6g "
                             "zeros=%ld nan=%ld inf=%ld\n",
                             tag, (int)c, cmax, cmean, cz, cn, ci);
            }
        }
        std::fflush(stderr);
        if (fp != nullptr) {
            std::fprintf(fp,
                         "[ladder] tag=%s layer=%d H=%d T=%d N=%lld absmax=%.6g "
                         "mean=%.6g zeros=%ld nan=%ld inf=%ld\n",
                         tag, layer, (int)rows, (int)cols, (long long)n, absmax,
                         mean, zeros, nans, infs);
            std::fclose(fp);
        }
    } catch (...) {
        std::fprintf(stderr, "[ladder-warn] tag=%s layer=%d exception swallowed\n",
                     tag != nullptr ? tag : "?", layer);
        std::fflush(stderr);
    }
}

// Trace-only raw snapshot of a BF16 [H, T] activation into
// $CINFERENCE_LADDER_DUMP_DIR/<tag>_l<layer>_H<H>_T<T>_n<k>.bin (little-endian
// BF16, column-major [H,T] as stored). Unset dir => no-op. Emits one
// [ladder-raw] stderr line per dump. Never throws, never writes device state.
inline void ladder_dump_raw(const ::ninfer::Tensor& t, const char* tag, int layer,
                            cudaStream_t stream) noexcept {
    try {
        const char* dir = std::getenv("CINFERENCE_LADDER_DUMP_DIR");
        if (dir == nullptr || dir[0] == '\0') { return; }
        if (tag == nullptr) { tag = "?"; }
        if (t.data == nullptr) {
            std::fprintf(stderr, "[ladder-raw] tag=%s layer=%d NULL tensor\n", tag, layer);
            std::fflush(stderr);
            return;
        }
        if (t.dtype != ::ninfer::DType::BF16) {
            std::fprintf(stderr, "[ladder-raw] tag=%s layer=%d SKIP non-BF16 dtype=%d\n", tag,
                         layer, static_cast<int>(t.dtype));
            std::fflush(stderr);
            return;
        }
        const std::int64_t rows = t.ne[0] < 0 ? 0 : t.ne[0];
        const std::int64_t cols = t.ne[1] < 0 ? 0 : t.ne[1];
        const std::int64_t n    = rows * cols;
        if (n <= 0 || !t.is_contiguous()) {
            std::fprintf(stderr, "[ladder-raw] tag=%s layer=%d SKIP shape not dumpable\n", tag, layer);
            std::fflush(stderr);
            return;
        }
        static unsigned long long seq = 0;
        char path[1024];
        std::snprintf(path, sizeof(path), "%s/%s_l%d_H%lld_T%lld_n%llu.bin", dir, tag, layer,
                      (long long)rows, (long long)cols, (unsigned long long)seq);
        std::vector<std::uint16_t> host(static_cast<std::size_t>(n));
        if (cudaMemcpyAsync(host.data(), t.data, static_cast<std::size_t>(n) * 2,
                            cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
            std::fprintf(stderr, "[ladder-raw] tag=%s layer=%d WARN copy-launch failed\n", tag,
                         layer);
            std::fflush(stderr);
            return;
        }
        if (cudaStreamSynchronize(stream) != cudaSuccess) {
            std::fprintf(stderr, "[ladder-raw] tag=%s layer=%d WARN stream-sync failed\n", tag,
                         layer);
            std::fflush(stderr);
            return;
        }
        std::FILE* fp = std::fopen(path, "wb");
        if (fp == nullptr) {
            std::fprintf(stderr, "[ladder-raw] tag=%s layer=%d WARN cannot open path\n", tag,
                         layer);
            std::fflush(stderr);
            return;
        }
        const std::size_t wrote = std::fwrite(host.data(), 2, static_cast<std::size_t>(n), fp);
        std::fclose(fp);
        std::fprintf(stderr,
                     "[ladder-raw] tag=%s layer=%d H=%lld T=%lld N=%lld wrote=%llu\n",
                     tag, layer, (long long)rows, (long long)cols, (long long)n,
                     (unsigned long long)wrote);
        std::fflush(stderr);
        ++seq;
    } catch (...) {
        std::fprintf(stderr, "[ladder-raw-warn] exception swallowed\n");
        std::fflush(stderr);
    }
}
} // namespace ninfer::models::qwen3_5::execution::ladder
