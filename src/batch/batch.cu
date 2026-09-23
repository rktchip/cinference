#include "batch/batch.h"

#include <stdexcept>
#include <string>

namespace ninfer::batch {

RaggedBatch AssembleBatch(const RequestScheduler& scheduler, const StepPlan& plan,
                          bool need_host_tables) {
    RaggedBatch batch;
    batch.max_blocks = scheduler.pool().max_blocks_per_seq();
    const std::size_t n = plan.prefill.size() + plan.decode_seq_ids.size();
    if (need_host_tables) {
        batch.block_tables.reserve(n * batch.max_blocks);
    }
    batch.seq_ids.reserve(n);
    batch.seq_lengths.reserve(n);
    batch.seq_offsets.reserve(n + 1);
    batch.seq_offsets.push_back(0);
    std::uint32_t total = 0;

    auto emit_row = [&](std::uint64_t seq_id, const TokenId* first, std::uint32_t count,
                        const std::vector<std::int32_t>& block_table) {
        const Sequence* seq = scheduler.find_sequence(seq_id);
        if (seq == nullptr) {
            throw std::out_of_range("AssembleBatch: unknown seq " + std::to_string(seq_id));
        }
        batch.seq_ids.push_back(seq_id);
        batch.seq_lengths.push_back(count);
        batch.seq_rows.push_back(seq->table_row);
        batch.tokens.insert(batch.tokens.end(), first, first + count);
        total += count;
        batch.seq_offsets.push_back(total);
        if (need_host_tables) {
            // Host-only path: the device-gather kernel does not exist, so the
            // compact [num_seqs x max_blocks] matrix is uploaded verbatim.
            const std::size_t base = batch.block_tables.size();
            batch.block_tables.resize(base + batch.max_blocks, kUnmappedBlock);
            const std::size_t copy =
                block_table.size() < batch.max_blocks ? block_table.size() : batch.max_blocks;
            for (std::size_t i = 0; i < copy; ++i) {
                batch.block_tables[base + i] = block_table[i];
            }
        }
        // Device path: seq_rows alone drives the on-device gather kernel; the
        // host-side matrix would be dead weight at width.
    };

    for (const auto& slice : plan.prefill) {
        const Sequence* seq = scheduler.find_sequence(slice.seq_id);
        if (seq == nullptr) {
            throw std::out_of_range("AssembleBatch: unknown seq " +
                                    std::to_string(slice.seq_id));
        }
        if (slice.offset + slice.count > seq->tokens.size()) {
            throw std::out_of_range("AssembleBatch: prefill slice out of range");
        }
        emit_row(slice.seq_id, seq->tokens.data() + slice.offset, slice.count,
                 seq->block_table);
    }
    for (std::uint64_t seq_id : plan.decode_seq_ids) {
        const Sequence* seq = scheduler.find_sequence(seq_id);
        if (seq == nullptr || seq->tokens.empty()) {
            throw std::out_of_range("AssembleBatch: unknown/empty decode seq " +
                                    std::to_string(seq_id));
        }
        emit_row(seq_id, seq->tokens.data() + seq->tokens.size() - 1, 1, seq->block_table);
    }
    return batch;
}

__global__ void GatherBlockTablesKernel(const std::int32_t* pool_table,
                                        const std::int32_t* rows, std::int32_t* out,
                                        std::int32_t max_blocks, std::int32_t num_seqs) {
    const std::int32_t flat = static_cast<std::int32_t>(blockIdx.x) *
                                  static_cast<std::int32_t>(blockDim.x) +
                              static_cast<std::int32_t>(threadIdx.x);
    const std::int32_t total = num_seqs * max_blocks;
    if (flat >= total) { return; }
    const std::int32_t seq   = flat / max_blocks;
    const std::int32_t block = flat % max_blocks;
    const std::int32_t row   = rows[seq];
    out[flat] = (row < 0) ? -1 : pool_table[row * max_blocks + block];
}

BatchDeviceBuffers::BatchDeviceBuffers(std::uint32_t max_seqs, std::uint32_t max_blocks_per_seq,
                                       std::uint32_t max_tokens_per_step) {
    if (max_seqs == 0 || max_blocks_per_seq == 0 || max_tokens_per_step == 0) {
        throw std::invalid_argument("BatchDeviceBuffers requires nonzero maxima");
    }
    TokenId* tokens           = nullptr;
    std::uint32_t* offsets    = nullptr;
    std::int32_t* tables      = nullptr;
    const cudaError_t tok_err = cudaMalloc(&tokens, static_cast<std::size_t>(max_tokens_per_step) *
                                                        sizeof(TokenId));
    const cudaError_t off_err =
        cudaMalloc(&offsets, static_cast<std::size_t>(max_seqs + 1) * sizeof(std::uint32_t));
    const cudaError_t tab_err =
        cudaMalloc(&tables, static_cast<std::size_t>(max_seqs) * max_blocks_per_seq *
                                sizeof(std::int32_t));
    std::int32_t* staging = nullptr;
    const cudaError_t row_err =
        cudaMalloc(&staging, static_cast<std::size_t>(max_seqs) * sizeof(std::int32_t));
    if (tok_err != cudaSuccess || off_err != cudaSuccess || tab_err != cudaSuccess ||
        row_err != cudaSuccess) {
        if (tokens != nullptr) { cudaFree(tokens); }
        if (offsets != nullptr) { cudaFree(offsets); }
        if (tables != nullptr) { cudaFree(tables); }
        if (staging != nullptr) { cudaFree(staging); }
        return; // host-only mode: view_ stays null, upload() checks device_ok()
    }
    view_.tokens       = tokens;
    view_.seq_offsets  = offsets;
    view_.block_tables = tables;
    staging_rows_      = staging;
    max_seqs_          = max_seqs;
}

BatchDeviceBuffers::~BatchDeviceBuffers() {
    if (view_.tokens != nullptr) { cudaFree(view_.tokens); }
    if (view_.seq_offsets != nullptr) { cudaFree(view_.seq_offsets); }
    if (view_.block_tables != nullptr) { cudaFree(view_.block_tables); }
    if (staging_rows_ != nullptr) { cudaFree(staging_rows_); }
}

void BatchDeviceBuffers::upload(const RaggedBatch& batch, const PagedKvPool& pool,
                                cudaStream_t stream) {
    if (!device_ok()) { return; }
    if (batch.num_seqs() == 0) {
        view_.num_seqs     = 0;
        view_.total_tokens = 0;
        view_.max_blocks   = batch.max_blocks;
        return;
    }
    check_cuda(cudaMemcpyAsync(view_.tokens, batch.tokens.data(),
                               batch.tokens.size() * sizeof(TokenId), cudaMemcpyHostToDevice,
                               stream),
               "BatchDeviceBuffers::upload tokens");
    check_cuda(cudaMemcpyAsync(view_.seq_offsets, batch.seq_offsets.data(),
                               batch.seq_offsets.size() * sizeof(std::uint32_t),
                               cudaMemcpyHostToDevice, stream),
               "BatchDeviceBuffers::upload offsets");
    if (pool.device_ok()) {
        if (batch.num_seqs() > max_seqs_ || batch.seq_rows.size() != batch.num_seqs()) {
            throw std::out_of_range("BatchDeviceBuffers::upload: step exceeds maxima");
        }
        // Hot path: one narrow rows upload plus an on-device gather of the
        // pool matrix in batch (plan) order. No host walk of the page table.
        check_cuda(cudaMemcpyAsync(staging_rows_, batch.seq_rows.data(),
                                   batch.seq_rows.size() * sizeof(std::int32_t),
                                   cudaMemcpyHostToDevice, stream),
                   "BatchDeviceBuffers::upload rows");
        const std::int32_t total =
            static_cast<std::int32_t>(batch.num_seqs() * batch.max_blocks);
        constexpr int kThreads = 256;
        GatherBlockTablesKernel<<<(total + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
            pool.device_table(), staging_rows_, view_.block_tables,
            static_cast<std::int32_t>(batch.max_blocks),
            static_cast<std::int32_t>(batch.num_seqs()));
        check_cuda(cudaGetLastError(), "GatherBlockTablesKernel launch");
    } else {
        check_cuda(cudaMemcpyAsync(view_.block_tables, batch.block_tables.data(),
                                   batch.block_tables.size() * sizeof(std::int32_t),
                                   cudaMemcpyHostToDevice, stream),
                   "BatchDeviceBuffers::upload tables fallback");
    }
    view_.num_seqs     = static_cast<std::uint32_t>(batch.num_seqs());
    view_.total_tokens = static_cast<std::uint32_t>(batch.tokens.size());
    view_.max_blocks   = batch.max_blocks;
}

} // namespace ninfer::batch
