// tests/test_ragged_attention_map.cc
// Swarm G2 seam test (b): flat token -> (seq, page) resolution over a mixed
// ragged step.
//
// Production code under test (no mirrors): ResolveRaggedToken and
// RaggedBlockForToken from src/batch/batch.h, with the production page
// constant kBatchPageTokens (src/batch/paged_kv.h). Both resolvers are
// header-inline, so this TU links no library; it needs the src/include/CUDA
// include paths at compile time but no device handle at run time.
//
// Fixture: a hand-built mixed RaggedBatch with the row shape a mixed StepPlan
// assembles to (prefill rows carry many tokens, decode rows carry exactly
// one):
//   row 0: prefill, 20 tokens (spans 16-token blocks 0 and 1)
//   row 1: decode,   1 token  (block 0)
//   row 2: prefill, 33 tokens (spans blocks 0, 1 and 2)
// M = 54, offsets = {0, 20, 21, 54}, block rows (max_blocks = 4):
//   row 0: {11, 12, -1, -1}  row 1: {21, -1, -1, -1}  row 2: {31, 32, 33, -1}
//
// Covered: multi-token rows, 16-token block boundaries (tokens 15/16 and
// 36/37), an every-token sweep against the offset formula, OOB tokens, an
// empty batch, bad offsets, max_blocks == 0, and logical-block-past-max.
//
// Host-only: no kernel launch, no device handle. GPU execution: GATED.
#include "batch/batch.h"

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

using ninfer::batch::RaggedBatch;
using ninfer::batch::RaggedBlockForToken;
using ninfer::batch::ResolveRaggedToken;

static_assert(ninfer::batch::kBatchPageTokens == 16,
              "ragged seam assumes 16-token logical blocks");

// Mixed fixture described in the header comment above.
RaggedBatch make_mixed_batch() {
    RaggedBatch batch;
    batch.max_blocks = 4;
    batch.seq_ids = {7, 8, 9};
    batch.seq_lengths = {20, 1, 33};
    batch.seq_offsets = {0, 20, 21, 54};
    batch.seq_rows = {0, 1, 2};
    batch.tokens.resize(54);
    for (std::uint32_t i = 0; i < 54; ++i) {
        batch.tokens[i] = static_cast<ninfer::TokenId>(1000 + i);
    }
    batch.block_tables = {
        11, 12, -1, -1, // row 0
        21, -1, -1, -1, // row 1
        31, 32, 33, -1, // row 2
    };
    return batch;
}

void check_block(const RaggedBatch& batch, std::uint32_t token, std::uint32_t want_seq,
                 std::uint32_t want_block, std::int32_t want_page) {
    std::uint32_t seq = 0xFFFFFFFF;
    std::uint32_t block = 0xFFFFFFFF;
    std::int32_t page = -12345;
    const bool ok = RaggedBlockForToken(batch, token, seq, block, page);
    if (!ok) {
        ++failures;
        std::cout << "FAIL token " << token << ": resolver returned false\n";
        return;
    }
    CHECK(seq == want_seq);
    CHECK(block == want_block);
    CHECK(page == want_page);
}

} // namespace

int main() {
    const RaggedBatch batch = make_mixed_batch();

    // Row 0 (prefill, 20 tokens): block 0 covers tokens 0..15, block 1 covers
    // tokens 16..19 -- the 16-token boundary lands mid-row.
    check_block(batch, 0, 0, 0, 11);
    check_block(batch, 15, 0, 0, 11);
    check_block(batch, 16, 0, 1, 12);
    check_block(batch, 19, 0, 1, 12);

    // Row 1 (decode, exactly one token).
    check_block(batch, 20, 1, 0, 21);

    // Row 2 (prefill, 33 tokens): block 0 covers offsets 0..15 (tokens
    // 21..36), block 1 covers offsets 16..31 (tokens 37..52), block 2 covers
    // offset 32 (token 53).
    check_block(batch, 21, 2, 0, 31);
    check_block(batch, 36, 2, 0, 31);
    check_block(batch, 37, 2, 1, 32);
    check_block(batch, 52, 2, 1, 32);
    check_block(batch, 53, 2, 2, 33);

    // Direct offset resolution (no page lookup).
    {
        std::uint32_t seq = 0;
        std::uint32_t off = 0;
        CHECK(ResolveRaggedToken(batch, 20, seq, off) && seq == 1 && off == 0);
        CHECK(ResolveRaggedToken(batch, 53, seq, off) && seq == 2 && off == 32);
        CHECK(ResolveRaggedToken(batch, 0, seq, off) && seq == 0 && off == 0);
    }

    // Every-token sweep: each flat token resolves to the row its offsets
    // bracket, with logical block (t - offsets[s]) / kBatchPageTokens and the
    // matching physical page from that row of the compact matrix.
    for (std::uint32_t t = 0; t < 54; ++t) {
        std::uint32_t want_seq = 0;
        if (t >= 21) {
            want_seq = 2;
        } else if (t >= 20) {
            want_seq = 1;
        }
        const std::uint32_t off = t - batch.seq_offsets[want_seq];
        const std::uint32_t want_block =
            off / static_cast<std::uint32_t>(ninfer::batch::kBatchPageTokens);
        const std::int32_t want_page =
            batch.block_tables[want_seq * batch.max_blocks + want_block];
        check_block(batch, t, want_seq, want_block, want_page);
    }

    // OOB tokens resolve false (end of buffer and far past it).
    {
        std::uint32_t seq = 0;
        std::uint32_t off = 0;
        std::uint32_t block = 0;
        std::int32_t page = 0;
        CHECK(!ResolveRaggedToken(batch, 54, seq, off));
        CHECK(!RaggedBlockForToken(batch, 54, seq, block, page));
        CHECK(!ResolveRaggedToken(batch, 1000000, seq, off));
        CHECK(!RaggedBlockForToken(batch, 1000000, seq, block, page));
    }

    // Empty batch resolves false.
    {
        const RaggedBatch empty;
        std::uint32_t seq = 0;
        std::uint32_t off = 0;
        std::uint32_t block = 0;
        std::int32_t page = 0;
        CHECK(!ResolveRaggedToken(empty, 0, seq, off));
        CHECK(!RaggedBlockForToken(empty, 0, seq, block, page));
    }

    // Bad offsets (size != num_seqs + 1) resolve false.
    {
        RaggedBatch bad = batch;
        bad.seq_offsets = {0, 20};
        std::uint32_t seq = 0;
        std::uint32_t off = 0;
        std::uint32_t block = 0;
        std::int32_t page = 0;
        CHECK(!ResolveRaggedToken(bad, 0, seq, off));
        CHECK(!RaggedBlockForToken(bad, 0, seq, block, page));
    }

    // max_blocks == 0 resolves false (no matrix to read).
    {
        RaggedBatch narrow = batch;
        narrow.max_blocks = 0;
        std::uint32_t seq = 0;
        std::uint32_t block = 0;
        std::int32_t page = 0;
        CHECK(!RaggedBlockForToken(narrow, 0, seq, block, page));
    }

    // Logical block past max_blocks resolves false: one 20-token row with a
    // single-block matrix; token 16 needs block 1.
    {
        RaggedBatch one_block;
        one_block.max_blocks = 1;
        one_block.seq_ids = {42};
        one_block.seq_lengths = {20};
        one_block.seq_offsets = {0, 20};
        one_block.seq_rows = {0};
        one_block.tokens.resize(20, 1);
        one_block.block_tables = {77};
        check_block(one_block, 0, 0, 0, 77);
        check_block(one_block, 15, 0, 0, 77);
        std::uint32_t seq = 0;
        std::uint32_t block = 0;
        std::int32_t page = 0;
        CHECK(!RaggedBlockForToken(one_block, 16, seq, block, page));
        CHECK(!RaggedBlockForToken(one_block, 19, seq, block, page));
    }

    std::cout << "ragged_map GPU section: GATED (host resolvers only, no execution)\n";
    if (failures == 0) {
        std::cout << "ragged_map: PASS (54-token mixed map, boundaries, OOB/empty false)\n";
    } else {
        std::cout << "ragged_map: " << failures << " FAILURES\n";
    }
    return failures == 0 ? 0 : 1;
}
