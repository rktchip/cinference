#pragma once

#include <ninfer/types.h>

#include <cstddef>
#include <cstdint>
#include <vector>

// Continuous-batching request model for cinference.
//
// One Request carries one or more Sequences (the common case is exactly one).
// Each Sequence owns its prompt/computed token prefix plus a logical block_table
// of 16-token block ids minted by batch::PagedKvPool. The scheduler advances
// computed_len through chunked prefill, then appends one decoded token per step
// until finish() (EOS / max tokens) or abort() (client cancel / eviction).
namespace ninfer::batch {

enum class Phase : std::uint8_t {
    Prefill = 0,
    Decode  = 1,
};

enum class Status : std::uint8_t {
    Waiting  = 0,
    Running  = 1,
    Finished = 2,
    Aborted  = 3,
};

struct Sequence {
    std::uint64_t seq_id = 0;
    // Prompt tokens followed by decoded tokens. computed_len tracks how many
    // leading tokens have been prefilled; tokens at [computed_len, size) are
    // decoded output when phase() == Decode.
    std::vector<TokenId> tokens;
    std::size_t computed_len = 0;
    // Prompt length frozen at admission; tokens beyond it are decoded output.
    std::size_t prompt_len = 0;
    // Logical-block -> pool-block-id map, entry count == ceil(tokens stored / 16).
    // Entries are pool ids (>= 0); the table never holds the -1 sentinel while
    // the sequence is admitted (unmapped tail is truncation, not a value).
    std::vector<std::int32_t> block_table;
    // Row lease in the pool device matrix, -1 while not admitted.
    int table_row = -1;

    [[nodiscard]] Phase phase() const noexcept {
        return computed_len < tokens.size() ? Phase::Prefill : Phase::Decode;
    }
    [[nodiscard]] std::size_t remaining_prefill() const noexcept {
        return tokens.size() > computed_len ? tokens.size() - computed_len : 0;
    }
    // Blocks needed to store token_count tokens at 16 tokens per block.
    [[nodiscard]] static std::uint32_t blocks_for(std::size_t token_count) noexcept;

    void advance_computed(std::size_t count) noexcept;
    void append_token(TokenId token);
};

struct Request {
    std::uint64_t req_id = 0;
    std::vector<Sequence> seqs;
    Status status             = Status::Waiting;
    std::uint32_t max_new_tokens = 0; // decode budget per sequence

    Request() = default;
    Request(std::uint64_t id, std::vector<TokenId> prompt, std::uint32_t max_new);

    [[nodiscard]] bool done() const noexcept {
        return status == Status::Finished || status == Status::Aborted;
    }
    void finish() noexcept { status = Status::Finished; }
    void abort() noexcept { status = Status::Aborted; }
    [[nodiscard]] bool all_decode() const noexcept;
};

} // namespace ninfer::batch
