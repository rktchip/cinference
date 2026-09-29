#include "batch/scheduler.h"

#include <algorithm>
#include <cstdlib>
#include <utility>

namespace ninfer::batch {

namespace {

// NINFER_FAIR_PREFILL=1 enables the fair prefill/admission policy; unset (or
// any other value) keeps the legacy strict-FIFO / head-eats-budget behavior
// exactly. Read per call (cheap getenv) so tests can toggle via setenv.
bool fair_prefill_enabled() {
    const char* v = std::getenv("NINFER_FAIR_PREFILL");
    return v != nullptr && v[0] == '1' && v[1] == '\0';
}

// Bound on followers admitted past a blocked head per admit_waiting() call.
constexpr std::uint32_t kFairSkipAdmitCap = 4;

} // namespace

RequestScheduler::RequestScheduler(std::uint32_t max_running_seqs, std::uint32_t total_pages,
                                   std::uint32_t max_blocks_per_seq, std::uint32_t chunk_tokens)
    : max_running_(max_running_seqs)
    , chunk_tokens_(chunk_tokens)
    , pool_(total_pages, max_blocks_per_seq, max_running_seqs) {
    if (max_running_ == 0 || chunk_tokens_ == 0) {
        throw std::invalid_argument("RequestScheduler requires nonzero running cap and chunk");
    }
}

std::uint64_t RequestScheduler::submit(Request request) {
    if (request.req_id == 0) { request.req_id = next_id_++; }
    for (auto& seq : request.seqs) {
        if (seq.seq_id == 0) { seq.seq_id = next_id_++; }
    }
    next_id_ = std::max(next_id_, request.req_id + 1);
    request.status = Status::Waiting;
    const std::uint64_t id = request.req_id;
    waiting_.push_back(std::move(request));
    return id;
}

bool RequestScheduler::abort(std::uint64_t req_id) {
    if (Request* req = find_request_mut(req_id)) {
        req->abort();
        return true;
    }
    return false;
}

Request* RequestScheduler::find_request_mut(std::uint64_t req_id) noexcept {
    for (auto& req : waiting_) {
        if (req.req_id == req_id) { return &req; }
    }
    for (auto& req : running_) {
        if (req.req_id == req_id) { return &req; }
    }
    return nullptr;
}

const Request* RequestScheduler::find_request(std::uint64_t req_id) const noexcept {
    for (const auto& req : waiting_) {
        if (req.req_id == req_id) { return &req; }
    }
    for (const auto& req : running_) {
        if (req.req_id == req_id) { return &req; }
    }
    return nullptr;
}

Sequence* RequestScheduler::find_sequence_mut(std::uint64_t seq_id) noexcept {
    for (auto& req : running_) {
        for (auto& seq : req.seqs) {
            if (seq.seq_id == seq_id) { return &seq; }
        }
    }
    for (auto& req : waiting_) {
        for (auto& seq : req.seqs) {
            if (seq.seq_id == seq_id) { return &seq; }
        }
    }
    return nullptr;
}

const Sequence* RequestScheduler::find_sequence(std::uint64_t seq_id) const noexcept {
    for (const auto& req : running_) {
        for (const auto& seq : req.seqs) {
            if (seq.seq_id == seq_id) { return &seq; }
        }
    }
    for (const auto& req : waiting_) {
        for (const auto& seq : req.seqs) {
            if (seq.seq_id == seq_id) { return &seq; }
        }
    }
    return nullptr;
}

bool RequestScheduler::finish_sequence(std::uint64_t seq_id) {
    Sequence* seq = find_sequence_mut(seq_id);
    if (seq == nullptr) { return false; }
    for (auto& req : running_) {
        for (auto& s : req.seqs) {
            if (s.seq_id == seq_id) {
                req.finish();
                return true;
            }
        }
    }
    return false;
}

void RequestScheduler::evict_done() {
    for (auto it = running_.begin(); it != running_.end();) {
        if (!it->done()) {
            ++it;
            continue;
        }
        for (auto& seq : it->seqs) {
            pool_.free(seq.block_table);
            pool_.release_row(seq.table_row);
            seq.table_row = -1;
        }
        it = running_.erase(it);
    }
    // Aborted waiters never consumed storage; drop them so they cannot be admitted.
    for (auto it = waiting_.begin(); it != waiting_.end();) {
        it = (it->done()) ? waiting_.erase(it) : std::next(it);
    }
}

bool RequestScheduler::admit_one(Request& request) {
    if (running_.size() >= max_running_) { return false; }
    // Reserve prompt blocks plus one decode slot per sequence up front so an
    // admitted sequence can always make progress without mid-step preemption.
    std::size_t admitted = 0;
    for (auto& seq : request.seqs) {
        const std::uint32_t need = Sequence::blocks_for(seq.tokens.size() + 1);
        bool ok                  = need <= pool_.max_blocks_per_seq();
        int row                  = -1;
        if (ok) {
            row = pool_.acquire_row();
            ok  = row >= 0;
        }
        if (ok) {
            seq.table_row = row;
            seq.prompt_len = seq.tokens.size();
            ok = pool_.grow(seq.block_table, need);
        }
        if (!ok) {
            // Roll back the failed sequence (grow may have partially filled its
            // table) plus sequences admitted above: the request never runs, so
            // evict_done() would not reclaim them.
            pool_.free(seq.block_table);
            if (row >= 0) { pool_.release_row(row); }
            for (std::size_t i = 0; i < admitted; ++i) {
                pool_.free(request.seqs[i].block_table);
                pool_.release_row(request.seqs[i].table_row);
                request.seqs[i].table_row = -1;
            }
            seq.table_row = -1;
            return false;
        }
        pool_.publish_row(seq.table_row, seq.block_table);
        ++admitted;
    }
    request.status = Status::Running;
    return true;
}

void RequestScheduler::admit_waiting() {
    if (fair_prefill_enabled()) {
        // Fair admission: the head is retried first every call, but a blocked
        // head no longer holds the queue — up to kFairSkipAdmitCap admittable
        // followers are admitted past it per call, in order. Safe subset:
        // admit_one() is atomic (rolls back rows/blocks on failure), so
        // probing followers cannot leak pool state, and the head keeps
        // front-of-queue priority on the next call, bounding its delay.
        std::uint32_t skipped_admits = 0;
        bool head_blocked             = false;
        for (auto it = waiting_.begin(); it != waiting_.end();) {
            if (it->done()) {
                it = waiting_.erase(it);
                continue;
            }
            if (running_.size() >= max_running_) { break; }
            if (!admit_one(*it)) {
                head_blocked = true;
                ++it;
                continue;
            }
            running_.push_back(std::move(*it));
            it = waiting_.erase(it);
            if (head_blocked && ++skipped_admits >= kFairSkipAdmitCap) { break; }
        }
        return;
    }
    for (auto it = waiting_.begin(); it != waiting_.end();) {
        if (it->done()) {
            it = waiting_.erase(it);
            continue;
        }
        if (!admit_one(*it)) { break; } // strict FIFO: blocked head blocks the queue
        running_.push_back(std::move(*it));
        it = waiting_.erase(it);
    }
}

StepPlan RequestScheduler::schedule_step() {
    evict_done();
    admit_waiting();
    StepPlan plan;
    if (fair_prefill_enabled()) {
        // Fair prefill: (b) any decode row present halves the per-step
        // prefill cap so a full prefill budget can never starve decodes
        // (every decode row still emits its one token); (a) the capped
        // budget is shared fair-share ceil(cap/n_prefill) per prefill
        // sequence in queue order — the head can no longer eat the whole
        // budget — with leftover dealt a second round-robin pass in order.
        bool has_decode               = false;
        std::uint32_t prefill_seqs    = 0;
        for (auto& req : running_) {
            if (req.done()) { continue; }
            for (auto& seq : req.seqs) {
                if (seq.phase() == Phase::Decode) {
                    has_decode = true;
                    plan.decode_seq_ids.push_back(seq.seq_id);
                } else {
                    ++prefill_seqs;
                }
            }
        }
        if (prefill_seqs == 0) { return plan; }
        std::uint32_t prefill_cap = chunk_tokens_;
        if (has_decode) { prefill_cap = std::max<std::uint32_t>(1, chunk_tokens_ / 2); }
        const std::uint32_t share = (prefill_cap + prefill_seqs - 1) / prefill_seqs;
        struct Alloc {
            Sequence* seq;
            std::uint32_t take;
        };
        std::vector<Alloc> allocs;
        for (auto& req : running_) {
            if (req.done()) { continue; }
            for (auto& seq : req.seqs) {
                if (seq.phase() != Phase::Prefill) { continue; }
                allocs.push_back(
                    Alloc{&seq, static_cast<std::uint32_t>(
                                    std::min<std::size_t>(seq.remaining_prefill(), share))});
            }
        }
        std::uint32_t spent = 0;
        for (const auto& a : allocs) { spent += a.take; }
        for (auto& a : allocs) {
            if (spent >= prefill_cap) { break; }
            const std::uint32_t rest =
                static_cast<std::uint32_t>(a.seq->remaining_prefill() - a.take);
            const std::uint32_t extra = std::min(rest, prefill_cap - spent);
            a.take += extra;
            spent += extra;
        }
        for (const auto& a : allocs) {
            if (a.take == 0) { continue; }
            plan.prefill.push_back(
                PrefillSlice{.seq_id = a.seq->seq_id,
                             .offset  = static_cast<std::uint32_t>(a.seq->computed_len),
                             .count   = a.take});
            plan.prefill_tokens += a.take;
        }
        return plan;
    }
    std::uint32_t budget = chunk_tokens_;
    for (auto& req : running_) {
        if (req.done()) { continue; }
        for (auto& seq : req.seqs) {
            if (seq.phase() == Phase::Decode) {
                plan.decode_seq_ids.push_back(seq.seq_id);
            } else if (budget > 0) {
                const std::uint32_t take = static_cast<std::uint32_t>(
                    std::min<std::size_t>(seq.remaining_prefill(), budget));
                plan.prefill.push_back(
                    PrefillSlice{.seq_id = seq.seq_id,
                                 .offset  = static_cast<std::uint32_t>(seq.computed_len),
                                 .count   = take});
                plan.prefill_tokens += take;
                budget -= take;
            }
        }
    }
    return plan;
}

std::vector<std::uint64_t> RequestScheduler::on_step_done(
    const StepPlan& plan, const std::vector<std::pair<std::uint64_t, TokenId>>& decoded) {
    for (const auto& slice : plan.prefill) {
        Sequence* seq = find_sequence_mut(slice.seq_id);
        if (seq == nullptr) { continue; }
        seq->advance_computed(slice.count);
        // A grown prompt (decoded tokens extend storage) may need one more block.
        const std::uint32_t need = Sequence::blocks_for(seq->tokens.size() + 1);
        if (need > seq->block_table.size()) {
            if (!pool_.grow(seq->block_table, need)) {
                // Pool exhausted mid-sequence: abort the owning request so the
                // next evict_done() reclaims its blocks instead of deadlocking.
                for (auto& req : running_) {
                    for (auto& s : req.seqs) {
                        if (s.seq_id == slice.seq_id) { req.abort(); }
                    }
                }
                continue;
            }
            pool_.publish_row(seq->table_row, seq->block_table);
        }
    }
    for (const auto& [seq_id, token] : decoded) {
        Sequence* seq = find_sequence_mut(seq_id);
        if (seq == nullptr) { continue; }
        seq->append_token(token);
        const std::uint32_t need = Sequence::blocks_for(seq->tokens.size() + 1);
        if (need > seq->block_table.size()) {
            if (pool_.grow(seq->block_table, need)) {
                pool_.publish_row(seq->table_row, seq->block_table);
            } else {
                // Pool exhausted mid-sequence: same policy as the prefill
                // path above -- abort the owning request so the next
                // evict_done() reclaims its blocks. Running on with a short
                // table is not an option: the unmapped tail tokens would
                // attend unmapped (-1) blocks. grow() is atomic on failure,
                // so there is nothing to roll back here.
                for (auto& req : running_) {
                    for (auto& s : req.seqs) {
                        if (s.seq_id == seq_id) { req.abort(); }
                    }
                }
            }
        }
    }
    std::vector<std::uint64_t> finished;
    for (auto& req : running_) {
        if (req.done()) { continue; }
        bool req_done = true;
        for (const auto& seq : req.seqs) {
            const std::size_t decoded_count =
                seq.tokens.size() > seq.prompt_len ? seq.tokens.size() - seq.prompt_len : 0;
            if (seq.phase() != Phase::Decode || decoded_count < req.max_new_tokens) {
                req_done = false;
                break;
            }
        }
        // Step-1 EOS stop: any decoded token this step that hits the
        // admitted stop set finishes the request. The accept/commit that
        // produced the tokens is untouched (spec-on accepts through EOS);
        // the pump drops the stop token and anything after it from the emit.
        if (!req_done && !req.stop_token_ids.empty()) {
            for (const auto& [seq_id, token] : decoded) {
                bool mine = false;
                for (const auto& seq : req.seqs) {
                    if (seq.seq_id == seq_id) {
                        mine = true;
                        break;
                    }
                }
                if (!mine) { continue; }
                for (const TokenId stop : req.stop_token_ids) {
                    if (token == stop) {
                        req_done = true;
                        break;
                    }
                }
                if (req_done) { break; }
            }
        }
        if (req_done) {
            req.finish();
            finished.push_back(req.req_id);
        }
    }
    // Audit B3: requests aborted mid-step (pool exhaustion) or while waiting
    // never take the req_done path above, but their hook states were
    // registered at drain and would leak in seq_states_ forever. List every
    // done request; erase_states_for_reqs is idempotent, and the pump treats
    // an aborted own request as done (breaks instead of spinning).
    for (auto& req : running_) {
        if (req.done() &&
            std::find(finished.begin(), finished.end(), req.req_id) == finished.end()) {
            finished.push_back(req.req_id);
        }
    }
    for (auto& req : waiting_) {
        if (req.done() &&
            std::find(finished.begin(), finished.end(), req.req_id) == finished.end()) {
            finished.push_back(req.req_id);
        }
    }
    return finished;
}

} // namespace ninfer::batch
