#pragma once
// Single owner for the checkpoint q_proj per-head interleave geometry:
// [q-head h; gate-head h] per 2*head_dim group (cf. oracle exllamav3
// deinterleave_qg), NOT [q; gate] halves. Scatters the fused GEMM temp into
// contiguous q/gate flats; k/v halves follow unchanged. All pointers are
// BF16 device memory. Call-site shape/range guards stay at the call sites;
// this helper owns only the copy geometry (C1 audit consolidation).
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include <cuda_runtime.h>

#include "core/device.h" // CUDA_CHECK (fail-fast everywhere; see C1 notes)

namespace ninfer::ops::detail {

inline void scatter_qg_heads(void* q, void* gate, void* k, void* v,
                             const void* full, std::size_t full_rows,
                             std::int32_t q_rows, std::int32_t kv_rows,
                             std::int32_t n_heads, std::int32_t head_dim,
                             std::size_t cols_T, cudaStream_t stream) {
    if (q_rows != n_heads * head_dim) {
        throw std::logic_error("scatter_qg_heads: q/g interleave geometry mismatch");
    }
    constexpr std::size_t kElem = sizeof(std::uint16_t);
    const char* src             = static_cast<const char*>(full);
    const std::size_t src_pitch = full_rows * kElem;
    auto block = [&](void* d, std::size_t d_pitch, std::size_t d_row,
                     std::size_t s_row, std::size_t rows) {
        CUDA_CHECK(cudaMemcpy2DAsync(static_cast<char*>(d) + d_row * kElem, d_pitch,
                                     src + s_row * kElem, src_pitch,
                                     rows * kElem, cols_T,
                                     cudaMemcpyDeviceToDevice, stream));
    };
    const std::size_t q_pitch  = static_cast<std::size_t>(q_rows) * kElem;
    const std::size_t kv_pitch = static_cast<std::size_t>(kv_rows) * kElem;
    for (std::int32_t h = 0; h < n_heads; ++h) {
        const std::size_t grp = static_cast<std::size_t>(h) * static_cast<std::size_t>(head_dim);
        block(q, q_pitch, grp, grp * 2U, static_cast<std::size_t>(head_dim));
        block(gate, q_pitch, grp, grp * 2U + static_cast<std::size_t>(head_dim),
              static_cast<std::size_t>(head_dim));
    }
    block(k, kv_pitch, 0, static_cast<std::size_t>(2 * q_rows),
          static_cast<std::size_t>(kv_rows));
    block(v, kv_pitch, 0, static_cast<std::size_t>(2 * q_rows + kv_rows),
          static_cast<std::size_t>(kv_rows));
}

}  // namespace ninfer::ops::detail
