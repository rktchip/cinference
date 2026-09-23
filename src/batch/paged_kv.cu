#include "batch/paged_kv.h"

#include <algorithm>

namespace ninfer::batch {

__global__ void RemapPageTableKernel(std::int32_t* table, std::int32_t num_seqs,
                                     std::int32_t max_blocks, std::int32_t num_pages) {
    const std::int32_t flat = static_cast<std::int32_t>(blockIdx.x) *
                                  static_cast<std::int32_t>(blockDim.x) +
                              static_cast<std::int32_t>(threadIdx.x);
    const std::int32_t total = num_seqs * max_blocks;
    if (flat >= total) { return; }
    const std::int32_t entry = table[flat];
    if (entry < -1 || entry >= num_pages) { table[flat] = -1; }
}

PagedKvPool::PagedKvPool(std::uint32_t num_pages, std::uint32_t max_blocks_per_seq,
                         std::uint32_t max_seqs)
    : num_pages_(num_pages)
    , max_blocks_(max_blocks_per_seq)
    , max_seqs_(max_seqs)
    , free_pages_(num_pages) {
    if (num_pages_ == 0 || max_blocks_ == 0 || max_seqs_ == 0) {
        throw std::invalid_argument("PagedKvPool requires nonzero capacity");
    }
    h_table_.assign(static_cast<std::size_t>(max_seqs_) * max_blocks_, kUnmappedBlock);
    free_rows_.reserve(max_seqs_);
    for (std::uint32_t row = 0; row < max_seqs_; ++row) {
        free_rows_.push_back(static_cast<int>(max_seqs_ - 1 - row));
    }
    const std::size_t bytes = h_table_.size() * sizeof(std::int32_t);
    std::int32_t* device   = nullptr;
    if (cudaMalloc(&device, bytes) == cudaSuccess) {
        d_table_ = device;
        check_cuda(cudaMemset(d_table_, 0xFF, bytes), "PagedKvPool table init");
    }
    // cudaMalloc failure (e.g. no device) leaves d_table_ null: host-only mode.
}

PagedKvPool::~PagedKvPool() {
    if (d_table_ != nullptr) { cudaFree(d_table_); }
}

bool PagedKvPool::alloc_one(std::int32_t& out) noexcept {
    if (!free_list_.empty()) {
        out = free_list_.back();
        free_list_.pop_back();
        --free_pages_;
        return true;
    }
    if (watermark_ < num_pages_) {
        out = static_cast<std::int32_t>(watermark_++);
        --free_pages_;
        return true;
    }
    return false;
}

bool PagedKvPool::grow(std::vector<std::int32_t>& block_table, std::uint32_t need_blocks) {
    if (need_blocks > max_blocks_) { return false; }
    if (block_table.size() >= need_blocks) { return true; }
    const std::uint32_t need_new = need_blocks - static_cast<std::uint32_t>(block_table.size());
    // Atomicity: each alloc_one consumes exactly one free page, so fail before
    // minting anything. A failed grow leaves the table and free_pages_
    // untouched: no partial ids for the caller to reclaim, no row involved.
    if (need_new > free_pages_) { return false; }
    while (block_table.size() < need_blocks) {
        std::int32_t id = kUnmappedBlock;
        if (!alloc_one(id)) { return false; } // unreachable after the pre-check
        block_table.push_back(id);
    }
    return true;
}

void PagedKvPool::shrink(std::vector<std::int32_t>& block_table, std::uint32_t keep_blocks) {
    while (block_table.size() > keep_blocks) {
        const std::int32_t id = block_table.back();
        block_table.pop_back();
        if (id != kUnmappedBlock) {
            free_list_.push_back(id);
            ++free_pages_;
        }
    }
}

void PagedKvPool::free(std::vector<std::int32_t>& block_table) {
    shrink(block_table, 0);
}

int PagedKvPool::acquire_row() {
    if (free_rows_.empty()) { return -1; }
    const int row = free_rows_.back();
    free_rows_.pop_back();
    return row;
}

void PagedKvPool::release_row(int row) {
    if (row < 0 || row >= static_cast<int>(max_seqs_)) { return; }
    if (std::find(free_rows_.begin(), free_rows_.end(), row) != free_rows_.end()) {
        return; // already released: never duplicate a row lease
    }
    std::fill_n(h_table_.data() + static_cast<std::size_t>(row) * max_blocks_, max_blocks_,
                kUnmappedBlock);
    free_rows_.push_back(row);
    if (d_table_ != nullptr) {
        check_cuda(cudaMemsetAsync(d_table_ + static_cast<std::size_t>(row) * max_blocks_, 0xFF,
                                   static_cast<std::size_t>(max_blocks_) * sizeof(std::int32_t)),
                   "PagedKvPool row clear");
    }
}

void PagedKvPool::publish_row(int row, const std::vector<std::int32_t>& block_table,
                              cudaStream_t stream) {
    if (row < 0 || row >= static_cast<int>(max_seqs_)) {
        throw std::out_of_range("PagedKvPool::publish_row: bad row");
    }
    if (block_table.size() > max_blocks_) {
        throw std::out_of_range("PagedKvPool::publish_row: table overflow");
    }
    std::int32_t* shadow = h_table_.data() + static_cast<std::size_t>(row) * max_blocks_;
    std::fill_n(shadow, max_blocks_, kUnmappedBlock);
    std::copy(block_table.begin(), block_table.end(), shadow);
    if (d_table_ == nullptr) { return; }
    check_cuda(cudaMemcpyAsync(d_table_ + static_cast<std::size_t>(row) * max_blocks_, shadow,
                               static_cast<std::size_t>(max_blocks_) * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, stream),
               "PagedKvPool::publish_row copy");
}

void PagedKvPool::sync_table_to_device(cudaStream_t stream) {
    if (d_table_ == nullptr) { return; }
    check_cuda(cudaMemcpyAsync(d_table_, h_table_.data(),
                               h_table_.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                               stream),
               "PagedKvPool::sync_table_to_device");
}

void PagedKvPool::remap_on_device(cudaStream_t stream) {
    if (d_table_ == nullptr) { return; }
    const std::int32_t total = static_cast<std::int32_t>(h_table_.size());
    constexpr int kThreads = 256;
    const int blocks       = (total + kThreads - 1) / kThreads;
    RemapPageTableKernel<<<blocks, kThreads, 0, stream>>>(
        d_table_, static_cast<std::int32_t>(max_seqs_), static_cast<std::int32_t>(max_blocks_),
        static_cast<std::int32_t>(num_pages_));
    check_cuda(cudaGetLastError(), "RemapPageTableKernel launch");
}

} // namespace ninfer::batch
