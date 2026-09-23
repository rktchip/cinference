#include "batch/request.h"

#include <algorithm>
#include <cstddef>

#include "batch/paged_kv.h"

namespace ninfer::batch {

std::uint32_t Sequence::blocks_for(std::size_t token_count) noexcept {
    return static_cast<std::uint32_t>((token_count + kBatchPageTokens - 1) / kBatchPageTokens);
}

void Sequence::advance_computed(std::size_t count) noexcept {
    computed_len = std::min(tokens.size(), computed_len + count);
}

void Sequence::append_token(TokenId token) {
    // Capture the phase before the push: appending to a Decode-phase sequence
    // keeps it Decode (decoded tokens are computed by construction).
    const bool was_decode = (phase() == Phase::Decode);
    tokens.push_back(token);
    if (was_decode) { computed_len = tokens.size(); }
}

Request::Request(std::uint64_t id, std::vector<TokenId> prompt, std::uint32_t max_new)
    : req_id(id)
    , max_new_tokens(max_new) {
    Sequence seq;
    seq.seq_id = id;
    seq.tokens = std::move(prompt);
    seqs.push_back(std::move(seq));
}

bool Request::all_decode() const noexcept {
    if (seqs.empty()) { return false; }
    return std::all_of(seqs.begin(), seqs.end(), [](const Sequence& seq) {
        return seq.phase() == Phase::Decode;
    });
}

} // namespace ninfer::batch
