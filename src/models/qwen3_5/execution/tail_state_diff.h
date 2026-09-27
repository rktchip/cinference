#pragma once
// tail_state_diff.h
// SPEC-OFF TICKET step (b): fenced per-layer tail-row state-diff probe.
//
// WHAT: snapshots ONE column (the prompt tail token) of the per-layer
// residual x ([H,T] BF16, post-mlp_tail) to host and prints one
// [tail-diff] stderr line per layer: absmax/mean/sum over the tail column
// plus an FNV-1a checksum of its raw BF16 words (exact-equality check).
//
// WHY: ticket-(a) forced-token scoring (10 prompts x k=0-3) shows all
// divergences at k<=2, pointing at the prompt-to-decode handoff. The two
// candidate readings need per-layer tail state from BOTH formulations:
//   path=serve   — forward_serve_step mixed-step loop (spec-off tail row;
//                  GDN prefill rows ran Phase::Prefill chunked per row)
//   path=prefill — chunked prefill_impl run_layers (folded-prefill reference;
//                  same chunked GDN formulation over the whole prompt)
//   path=verify  — run_layers Phase::Verify (ordinary_decode_batch spec-off
//                  decode / MTP-warmed target-verify equivalent)
// Run the same prompt through serve and prefill with the flag set, then
// diff the per-layer lines offline (match on layer=; compare fnv=/absmax).
//
// FENCING: gated on NINFER_TAIL_STATE_DIFF (any value; per-call getenv, no
// reboot needed). Unset => one getenv + return: no device reads, no stream
// sync, no numeric effect on the forward pass. Set => D2H + stream sync per
// layer (print-only; never writes device state), so DO NOT arm under CUDA
// graph capture or for timed runs. Diagnostics never throw.
//
// PREDICTION + READING (pre-registered):
//   Family-level (every layer differs, smoothly growing or flat offset):
//     chunked-vs-recurrent GDN formulation difference on the tail token.
//     Document, no fix.
//   O(1) jump at some layer L (layers <L match exactly, >=L diverge):
//     spec-off defect at/above L (handoff slot, positions, or envelope).
//     Fix, then re-measure ticket-(a).

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace ninfer::models::qwen3_5::execution::taildiff {

inline float tail_bf16_to_float(std::uint16_t w) noexcept {
    std::uint32_t bits = static_cast<std::uint32_t>(w) << 16;
    float out          = 0.0F;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

// Trace-only snapshot of column tail_col of a BF16 [H,T] activation.
// Reads device memory, synchronizes the stream, prints one stats line.
// No writes to device state. Never throws.
inline void dump_tail_column(const ::ninfer::Tensor& t, const char* path, int layer,
                             int tail_col, cudaStream_t stream) noexcept {
    if (std::getenv("NINFER_TAIL_STATE_DIFF") == nullptr) { return; }
    try {
        if (path == nullptr) { path = "?"; }
        if (t.data == nullptr) {
            std::fprintf(stderr, "[tail-diff] path=%s layer=%d NULL tensor\n", path, layer);
            std::fflush(stderr);
            return;
        }
        if (t.dtype != ::ninfer::DType::BF16) {
            std::fprintf(stderr, "[tail-diff] path=%s layer=%d SKIP non-BF16 dtype=%d\n", path,
                         layer, static_cast<int>(t.dtype));
            std::fflush(stderr);
            return;
        }
        const std::int64_t rows = t.ne[0] < 0 ? 0 : t.ne[0];
        const std::int64_t cols = t.ne[1] < 0 ? 0 : t.ne[1];
        if (rows <= 0 || cols <= 0 || tail_col < 0 || tail_col >= cols || !t.is_contiguous()) {
            std::fprintf(stderr,
                         "[tail-diff] path=%s layer=%d SKIP shape=[%d,%d] tail=%d "
                         "contiguous=%d\n",
                         path, layer, (int)t.ne[0], (int)t.ne[1], tail_col,
                         (int)t.is_contiguous());
            std::fflush(stderr);
            return;
        }
        const std::int64_t n = rows * cols;
        std::vector<std::uint16_t> host(static_cast<std::size_t>(n));
        if (cudaMemcpyAsync(host.data(), t.data, static_cast<std::size_t>(n) * 2,
                            cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
            std::fprintf(stderr, "[tail-diff] path=%s layer=%d WARN copy-launch failed\n", path,
                         layer);
            std::fflush(stderr);
            return;
        }
        if (cudaStreamSynchronize(stream) != cudaSuccess) {
            std::fprintf(stderr, "[tail-diff] path=%s layer=%d WARN stream-sync failed\n", path,
                         layer);
            std::fflush(stderr);
            return;
        }
        // Column-major [H,T]: column c starts at c*rows.
        const std::size_t base = static_cast<std::size_t>(tail_col) * static_cast<std::size_t>(rows);
        double absmax = 0.0;
        double sum    = 0.0;
        long finite   = 0;
        long nans     = 0;
        long infs     = 0;
        std::uint64_t fnv = 1469598103934665603ULL;
        for (std::int64_t r = 0; r < rows; ++r) {
            const std::uint16_t w = host[base + static_cast<std::size_t>(r)];
            fnv ^= static_cast<std::uint64_t>(w);
            fnv *= 1099511628211ULL;
            const float x = tail_bf16_to_float(w);
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
                     "[tail-diff] path=%s layer=%d H=%d T=%d tail=%d absmax=%.6g mean=%.6g "
                     "sum=%.6g fnv=%llx finite=%ld nan=%ld inf=%ld\n",
                     path, layer, (int)rows, (int)cols, tail_col, absmax, mean, sum,
                     (unsigned long long)fnv, finite, nans, infs);
        std::fflush(stderr);
    } catch (...) {
        std::fprintf(stderr, "[tail-diff-warn] path=%s layer=%d exception swallowed\n",
                     path != nullptr ? path : "?", layer);
        std::fflush(stderr);
    }
}

} // namespace ninfer::models::qwen3_5::execution::taildiff
