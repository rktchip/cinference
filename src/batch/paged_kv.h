#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

// HyperQwen-style continuous-batching block allocator for cinference.
//
// This pool manages LOGICAL kBatchPageTokens-token blocks used by the
// iteration-level scheduler (src/batch/scheduler.h) for admission and
// token-budget accounting. Page size is the named constant kBatchPageTokens
// below (16 tokens); code never uses a magic 16. Capacity (num_pages) is
// fixed at startup: the pool never grows after construction, and grow()
// fails atomically (no table or free-count mutation) when capacity is
// exhausted.
// It deliberately does NOT own KV bytes: real KV storage stays in the upstream
// startup-fixed pool (core/paged_kv_cache.h, DeviceKVPagePool, 64-token
// physical pages). Four logical blocks address one physical page; the
// integration layer (src/batch/cinference_hooks.h) owns that mapping.
//
// Device residency: the pool owns one small int32 matrix
// [max_seqs x max_blocks_per_seq] on the device plus a pinned-shape host
// shadow. The host publishes only occupancy (free_pages/used_pages) and the
// free-list; kernels read the device copy. If no CUDA device is present the
// pool transparently runs host-only (device_ok() == false) and every device
// entry point becomes a checked no-op.
namespace ninfer::batch {

inline constexpr std::int32_t kBatchPageTokens = 16;
inline constexpr std::int32_t kUnmappedBlock   = -1;

inline void check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " +
                                 cudaGetErrorString(status));
    }
}

// CUDA page-table maintenance kernel: clamps every entry of the flat
// [num_seqs x max_blocks] device table into [-1, num_pages). Declared only
// under nvcc; MSVC never sees CUDA kernel syntax from this header.
#ifdef __CUDACC__
__global__ void RemapPageTableKernel(std::int32_t* table, std::int32_t num_seqs,
                                     std::int32_t max_blocks, std::int32_t num_pages);
#endif

class PagedKvPool {
public:
    // num_pages: startup-fixed logical-block capacity. max_blocks_per_seq: per-seq
    // logical max-context in blocks (ceil(max_tokens / kBatchPageTokens)).
    // max_seqs: table rows.
    PagedKvPool(std::uint32_t num_pages, std::uint32_t max_blocks_per_seq,
                std::uint32_t max_seqs);
    ~PagedKvPool();

    PagedKvPool(const PagedKvPool&)            = delete;
    PagedKvPool& operator=(const PagedKvPool&) = delete;
    PagedKvPool(PagedKvPool&&)                 = delete;
    PagedKvPool& operator=(PagedKvPool&&)      = delete;

    // Grow block_table to need_blocks entries. Allocation order: free-list first,
    // then the watermark (bump), else failure. Returns false without mutating the
    // table when capacity is exhausted.
    [[nodiscard]] bool grow(std::vector<std::int32_t>& block_table,
                            std::uint32_t need_blocks);
    // Return trailing entries beyond keep_blocks to the free-list.
    void shrink(std::vector<std::int32_t>& block_table, std::uint32_t keep_blocks);
    // Return every mapped entry to the free-list and reset the table to {-1}.
    // Freeing an empty table is a no-op: double-free can never duplicate ids.
    void free(std::vector<std::int32_t>& block_table);

    // Row leases for the device table matrix. Rows are independent of block ids.
    [[nodiscard]] int acquire_row();
    void release_row(int row);

    // Publish one row of the host shadow to the device matrix.
    void publish_row(int row, const std::vector<std::int32_t>& block_table,
                     cudaStream_t stream = nullptr);
    // Full host-shadow -> device copy.
    void sync_table_to_device(cudaStream_t stream = nullptr);
    // Launch RemapPageTableKernel over the device matrix (no-op when host-only).
    void remap_on_device(cudaStream_t stream = nullptr);

    [[nodiscard]] std::uint32_t num_pages() const noexcept { return num_pages_; }
    [[nodiscard]] std::uint32_t max_blocks_per_seq() const noexcept { return max_blocks_; }
    [[nodiscard]] std::uint32_t max_seqs() const noexcept { return max_seqs_; }
    [[nodiscard]] std::uint32_t free_pages() const noexcept { return free_pages_; }
    [[nodiscard]] std::uint32_t used_pages() const noexcept { return num_pages_ - free_pages_; }
    [[nodiscard]] bool device_ok() const noexcept { return d_table_ != nullptr; }
    [[nodiscard]] const std::int32_t* device_table() const noexcept { return d_table_; }
    [[nodiscard]] const std::vector<std::int32_t>& host_shadow() const noexcept {
        return h_table_;
    }

private:
    [[nodiscard]] bool alloc_one(std::int32_t& out) noexcept;

    std::uint32_t num_pages_ = 0;
    std::uint32_t max_blocks_ = 0;
    std::uint32_t max_seqs_   = 0;
    std::uint32_t watermark_  = 0; // bump allocator: ids [0, watermark) minted
    std::uint32_t free_pages_ = 0;
    std::vector<std::int32_t> free_list_; // recycled ids, LIFO
    std::vector<std::int32_t> h_table_;   // host shadow, row-major, -1 = unmapped
    std::vector<int> free_rows_;          // recyclable table rows
    std::int32_t* d_table_ = nullptr;     // device matrix, nullptr when host-only
};

} // namespace ninfer::batch
