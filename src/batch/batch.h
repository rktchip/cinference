#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "batch/paged_kv.h"
#include "batch/scheduler.h"

// Ragged batch assembly for one scheduler step.
//
// Layout of the flat token buffer: all prefill-slice tokens first (in plan
// order), then exactly one token per decode sequence (its last stored token).
// Row order (seq_ids, seq_rows, and block-table rows) follows the same plan
// order, so attention and EXL3 consume identical token order. M
// (tokens.size()) is the exact live-token count, never padded (in particular
// never rounded up to 16 for EXL3).
// seq_offsets is the exclusive prefix sum of seq_lengths, so row i spans
// tokens[offsets[i], offsets[i+1]). block_tables is the compact
// [num_seqs x max_blocks] int32 matrix (-1 padded) gathered from each
// sequence's pool table; the matching device copy is uploaded by
// BatchDeviceBuffers for the attention kernels to consume.
namespace ninfer::batch {

struct RaggedBatch {
    std::vector<TokenId> tokens;
    std::vector<std::uint32_t> seq_offsets; // size num_seqs + 1, offsets[0] == 0
    std::vector<std::uint64_t> seq_ids;
    std::vector<std::uint32_t> seq_lengths;
    std::vector<std::int32_t> block_tables;  // flat num_seqs * max_blocks; set only when
                                             // need_host_tables (host-only pool path)
    std::vector<std::int32_t> seq_rows;     // pool-matrix row per seq, -1 if none
    std::uint32_t max_blocks = 0;

    [[nodiscard]] std::size_t num_seqs() const noexcept { return seq_ids.size(); }
};

// Assemble one step. Throws std::out_of_range when the plan references an
// unknown sequence (fail fast: a stale plan must never run).
// need_host_tables: when true (host-only pools, no device matrix to gather),
// materialize the host-side [num_seqs x max_blocks] block-table matrix; when
// false, RaggedBatch::block_tables stays empty and BatchDeviceBuffers::upload
// gathers the per-step table on-device (the fast path at width: no host-side
// O(nums x max_blocks) gather per step).
[[nodiscard]] RaggedBatch AssembleBatch(const RequestScheduler& scheduler, const StepPlan& plan,
                                       bool need_host_tables = false);

// Device-side mirrors of RaggedBatch, uploaded once per step.
struct DeviceRaggedBatch {
    TokenId* tokens             = nullptr;
    std::uint32_t* seq_offsets  = nullptr;
    std::int32_t* block_tables  = nullptr;
    std::uint32_t num_seqs      = 0;
    std::uint32_t total_tokens  = 0;
    std::uint32_t max_blocks    = 0;
};

// Gathers compact per-step block-table rows on the device. rows[i] selects the
// pool-matrix row for sequence i; -1-padded tail is preserved verbatim.
// Declared only under nvcc; MSVC never sees CUDA kernel syntax from this header.
#ifdef __CUDACC__
__global__ void GatherBlockTablesKernel(const std::int32_t* pool_table,
                                        const std::int32_t* rows, std::int32_t* out,
                                        std::int32_t max_blocks, std::int32_t num_seqs);
#endif

// Owns device buffers sized to the startup-fixed maxima; upload() copies one
// assembled step (host-only pools upload the token/offset payload and skip the
// table gather, which needs the device matrix).
class BatchDeviceBuffers {
public:
    BatchDeviceBuffers(std::uint32_t max_seqs, std::uint32_t max_blocks_per_seq,
                       std::uint32_t max_tokens_per_step);
    ~BatchDeviceBuffers();

    BatchDeviceBuffers(const BatchDeviceBuffers&)            = delete;
    BatchDeviceBuffers& operator=(const BatchDeviceBuffers&) = delete;

    void upload(const RaggedBatch& batch, const PagedKvPool& pool,
                cudaStream_t stream = nullptr);
    [[nodiscard]] const DeviceRaggedBatch& device() const noexcept { return view_; }
    [[nodiscard]] bool device_ok() const noexcept { return view_.tokens != nullptr; }

private:
    DeviceRaggedBatch view_;
    std::int32_t* staging_rows_ = nullptr; // device scratch: one row id per seq
    std::uint32_t max_seqs_     = 0;
};

// E (attention binding): ragged token -> (seq, page) resolution over the
// gathered tables. seq_offsets is the exclusive prefix sum of seq_lengths
// (offsets[0] == 0, size num_seqs + 1), so flat token t belongs to the unique
// row s with offsets[s] <= t < offsets[s + 1]; its logical block is
// (t - offsets[s]) / kBatchPageTokens and the physical page is row s of the
// compact [num_seqs x max_blocks] matrix. Variable spans are the point: a
// prefill row covers many tokens while a decode row covers exactly one, so
// this is NOT one-token-per-row. Host proof helper: the host matrix exists
// only when need_host_tables is set (the device path leaves it empty; the
// device reader below is the consumer there).
[[nodiscard]] inline bool ResolveRaggedToken(const RaggedBatch& batch, std::uint32_t token,
                                            std::uint32_t& seq_out,
                                            std::uint32_t& offset_in_seq_out) noexcept {
    if (token >= static_cast<std::uint32_t>(batch.tokens.size())) { return false; }
    if (batch.seq_offsets.size() != batch.num_seqs() + 1) { return false; }
    for (std::uint32_t s = 0; s < static_cast<std::uint32_t>(batch.num_seqs()); ++s) {
        if (token >= batch.seq_offsets[s] && token < batch.seq_offsets[s + 1]) {
            seq_out           = s;
            offset_in_seq_out = token - batch.seq_offsets[s];
            return true;
        }
    }
    return false;
}

[[nodiscard]] inline bool RaggedBlockForToken(const RaggedBatch& batch, std::uint32_t token,
                                             std::uint32_t& seq_out,
                                             std::uint32_t& logical_block_out,
                                             std::int32_t& page_out) noexcept {
    std::uint32_t offset_in_seq = 0;
    if (!ResolveRaggedToken(batch, token, seq_out, offset_in_seq)) { return false; }
    if (batch.max_blocks == 0) { return false; }
    logical_block_out = offset_in_seq / static_cast<std::uint32_t>(kBatchPageTokens);
    if (logical_block_out >= batch.max_blocks) { return false; }
    const std::size_t idx = static_cast<std::size_t>(seq_out) * batch.max_blocks + logical_block_out;
    if (idx >= batch.block_tables.size()) { return false; }
    page_out = batch.block_tables[idx];
    return true;
}

// Device-side reader for the attention kernels: resolves a flat token index
// to its gathered page using ONLY view.seq_offsets + view.block_tables (the
// on-device gather output). Attention call sites include this header to bind
// the gathered tables. Declared only under nvcc; MSVC never sees it.
#ifdef __CUDACC__
__device__ __forceinline__ bool DeviceResolveRaggedToken(const DeviceRaggedBatch& view,
                                                         std::uint32_t token,
                                                         std::uint32_t& seq_out,
                                                         std::int32_t& page_out) {
    if (token >= view.total_tokens) { return false; }
    std::uint32_t seq = 0;
    while (seq < view.num_seqs && view.seq_offsets[seq + 1] <= token) { ++seq; }
    if (seq >= view.num_seqs) { return false; }
    const std::uint32_t blk =
        (token - view.seq_offsets[seq]) / static_cast<std::uint32_t>(kBatchPageTokens);
    if (blk >= view.max_blocks) { return false; }
    seq_out  = seq;
    page_out = view.block_tables[static_cast<std::size_t>(seq) * view.max_blocks + blk];
    return true;
}
#endif

// Serve-startup gate (E): BatchDeviceBuffers and PagedKvPool silently run
// host-only when no CUDA device is present. Serve must never run degraded, so
// serve startup must call this once and treat a throw as fatal (process exit,
// never a per-request fallback). The check lives here (consumer side) so
// every startup path shares one spelling.
inline void RequireDeviceBuffersForServe(const BatchDeviceBuffers& buffers, const PagedKvPool& pool) {
    if (!buffers.device_ok() || !pool.device_ok()) {
        throw std::runtime_error("serve startup requires CUDA device buffers and pool table: "
                                 "host-only fallback is not allowed in serve");
    }
}

} // namespace ninfer::batch
