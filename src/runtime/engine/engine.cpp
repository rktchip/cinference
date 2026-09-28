#include "ninfer/engine.h"

#include "batch/cinference_hooks.h"
#include "batch/scheduler.h"
#include "core/device.h"
#include "core/nvtx.h"
#include "core/startup.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "runtime/contract/sampling.h"
#include "runtime/contract/request.h"
#include "runtime/engine/causal_score_core.h"
#include "runtime/engine/engine_core.h"
#include "runtime/engine/model_instance.h"
#include "runtime/engine/step_forward.h"
#include "batch/batch.h"
#include "core/arena.h"
#include "core/layout.h"
#include "core/linear_attention_state.h"
#include "core/paged_kv_cache.h"
#include "models/qwen3_5/execution/text.h"
#include "models/qwen3_5/program/program.h"
#include "models/qwen3_5/program/round_buffers.h"
#include "models/qwen3_5/state/decoder_state.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/argmax.h"
#include "models/qwen3_5/execution/mtp_spec_gate.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

// Row 20b pulse counters (defined in text.cpp): per-column executions of
// the coltab branches. Declared here to read + reset from the oracle.
namespace ninfer {
namespace models {
namespace qwen3_5 {
namespace execution {
extern std::atomic<std::uint64_t> g_coltab_conv_cols;
extern std::atomic<std::uint64_t> g_coltab_rec_cols;
}  // namespace execution
}  // namespace qwen3_5
}  // namespace models
}  // namespace ninfer

namespace ninfer {
namespace {

DeviceContext initialize_device(const EngineOptions& options) {
    StartupPhaseScope phase(options.startup_observer, StartupPhase::CudaInitialize);
    DeviceContext device(options.device);
    phase.complete();
    return device;
}

runtime::ResolvedRequestOptions resolve_request_options(const ModelSamplingDefaults& defaults,
                                                        SamplingMode mode, RequestOptions options) {
    if (options.execution.thinking.budget && *options.execution.thinking.budget == 0) {
        throw std::invalid_argument("thinking budget must be positive");
    }
    runtime::ResolvedRequestOptions resolved;
    resolved.execution.sampling =
        runtime::resolve_sampling(defaults, mode, options.execution.sampling);
    resolved.execution.requested_output_tokens = options.execution.requested_output_tokens;
    resolved.execution.allow_prefix_reuse      = options.execution.allow_prefix_reuse;
    resolved.execution.thinking                = options.execution.thinking;
    resolved.stop                              = std::move(options.stop);
    resolved.output                            = options.output;
    return resolved;
}

std::string context_capacity_error(std::size_t prompt_tokens, std::uint32_t max_context) {
    return "prepared prompt has " + std::to_string(prompt_tokens) +
           " tokens, exceeding Engine max_context " + std::to_string(max_context);
}

} // namespace

class PreparedPrompt::Impl {
public:
    Impl(PromptSummary prompt_summary, PromptPreparationStats preparation, SamplingMode mode,
         models::qwen3_5::PreparedPrompt prepared)
        : summary(std::move(prompt_summary)), prepare(std::move(preparation)), sampling_mode(mode),
          value(std::move(prepared)) {}

    PromptSummary summary;
    PromptPreparationStats prepare;
    SamplingMode sampling_mode = SamplingMode::Thinking;
    models::qwen3_5::PreparedPrompt value;
};

PreparedPrompt::PreparedPrompt() noexcept                            = default;
PreparedPrompt::~PreparedPrompt()                                    = default;
PreparedPrompt::PreparedPrompt(PreparedPrompt&&) noexcept            = default;
PreparedPrompt& PreparedPrompt::operator=(PreparedPrompt&&) noexcept = default;

PreparedPrompt::PreparedPrompt(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

const PromptSummary& PreparedPrompt::summary() const noexcept {
    static const PromptSummary empty;
    return impl_ != nullptr ? impl_->summary : empty;
}

const PromptPreparationStats& PreparedPrompt::preparation_stats() const noexcept {
    static const PromptPreparationStats empty;
    return impl_ != nullptr ? impl_->prepare : empty;
}

PreparedPrompt::operator bool() const noexcept { return impl_ != nullptr; }

class GenerationHandle::Impl {
public:
    class Concept {
    public:
        virtual ~Concept() = default;
        virtual GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) = 0;
    };

    template <class Submission>
    class Model final : public Concept {
    public:
        Model(std::shared_ptr<void> keep_alive, Submission submission)
            : keep_alive_(std::move(keep_alive)), submission_(std::move(submission)) {}

        GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) override {
            return submission_.wait(sink, cancellation);
        }

    private:
        std::shared_ptr<void> keep_alive_;
        Submission submission_;
    };

    template <class Submission>
    Impl(std::shared_ptr<void> keep_alive, Submission submission,
         ResolvedSamplingParameters sampling)
        : state_(std::make_unique<Model<Submission>>(std::move(keep_alive), std::move(submission))),
          sampling_(sampling) {}

    GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) {
        return state_->wait(sink, cancellation);
    }

    [[nodiscard]] const ResolvedSamplingParameters& resolved_sampling() const noexcept {
        return sampling_;
    }

private:
    std::unique_ptr<Concept> state_;
    ResolvedSamplingParameters sampling_;
};

GenerationHandle::GenerationHandle() noexcept                              = default;
GenerationHandle::~GenerationHandle()                                      = default;
GenerationHandle::GenerationHandle(GenerationHandle&&) noexcept            = default;
GenerationHandle& GenerationHandle::operator=(GenerationHandle&&) noexcept = default;

GenerationHandle::GenerationHandle(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

GenerationHandle::operator bool() const noexcept { return impl_ != nullptr; }

const ResolvedSamplingParameters& GenerationHandle::resolved_sampling() const noexcept {
    static const ResolvedSamplingParameters empty;
    return impl_ != nullptr ? impl_->resolved_sampling() : empty;
}

GenerationResult GenerationHandle::wait(OutputSink* sink, const CancellationView& cancellation) {
    if (impl_ == nullptr) { throw std::logic_error("GenerationHandle is empty"); }
    std::unique_ptr<Impl> impl = std::move(impl_);
    return impl->wait(sink, cancellation);
}

// Serve forward residency (single slot, Engine-owned): the persistent
// TextContext plus device KV/GDN plus workspace plus RoundState behind
// Engine::run_batch_step. The scheduler owns page ids and admission; this
// context owns one private physical page range and one GDN slot per serve
// lane and reads the dispatch rows only. No fused attention, no MTP, no
// second generate loop: one TextContext forward per step, EXL3 linears
// through the existing wrappers, sampling via sample_decode_rows.
class ServeForwardContext {
public:
    ServeForwardContext(DeviceContext& device,
                        const models::qwen3_5::execution::Parameters& parameters,
                        const EngineOptions& options)
        : device_(device), parameters_(parameters) {
        using models::qwen3_5::execution::dimension;
        static_assert(sizeof(TokenId) == sizeof(std::int32_t), "TokenId must be I32");
        const auto& text_config = parameters_.model.config().text;
        if (!text_config.attention || !text_config.rope_parameters) {
            throw std::logic_error("serve forward requires text attention and rope config");
        }
        if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
            throw std::invalid_argument("serve forward max_concurrency is invalid");
        }
        if (options.max_context == 0 || options.prefill_chunk == 0) {
            throw std::invalid_argument("serve forward needs max_context and prefill_chunk");
        }
        max_seqs_    = options.max_concurrency;
        max_context_ = options.max_context;
        warm_chunk_  = options.prefill_chunk;
        // S7 MTP-3 serve residency: --spec mtp --draft-tokens 3 admits the
        // in-checkpoint MTP head (bound at startup in exl3_program.cpp). The
        // window is pinned to kMtpSpecDecodeDrafts (3, full proposal head);
        // anything else stays a startup-fatal there. Spec-off keeps the
        // historical zeros below (no MTP KV, no MTP round state).
        mtp_enabled_ = options.speculative.backend == SpeculativeBackend::Mtp;
        if (mtp_enabled_) {
            if (options.speculative.draft_tokens !=
                models::qwen3_5::execution::kMtpSpecDecodeDrafts) {
                throw std::invalid_argument("serve forward MTP requires draft window 3");
            }
            if (!parameters_.mtp) {
                throw std::logic_error("serve forward MTP needs bound MTP parameters");
            }
        }
        // Row-23 conc resume (task 8, default off): mid-request spec toggle.
        // NINFER_MTP_TOGGLE_AFTER=K runs K pure-decode steps on the lane,
        // then NINFER_MTP_TOGGLE_OFF=N forces the next N steps onto the
        // ordinary (spec-off) path, then spec resumes with a batched
        // catch-up fill over the skipped positions. Unset (both zero) means
        // no toggle: dispatch is exactly as before. Parsed once at startup;
        // per-lane counters reset on lane admission. OFF is clamped so the
        // catch-up staging bound below stays sane.
        toggle_after_ = 0;
        toggle_off_   = 0;
        if (mtp_enabled_) {
            if (const char* a = std::getenv("NINFER_MTP_TOGGLE_AFTER")) {
                long v = std::strtol(a, nullptr, 10);
                if (v > 1000000) { v = 1000000; }
                if (v > 0) { toggle_after_ = static_cast<std::uint32_t>(v); }
            }
            if (const char* o = std::getenv("NINFER_MTP_TOGGLE_OFF")) {
                long v = std::strtol(o, nullptr, 10);
                if (v > 4096) { v = 4096; }
                if (v > 0) { toggle_off_ = static_cast<std::uint32_t>(v); }
            }
        }
        text_vocab_ = text_config.vocab_size;
        if (mtp_enabled_ && text_vocab_ == 0) {
            throw std::logic_error("serve forward MTP needs a vocabulary");
        }
        // Memory diet: the MTP decode step is single-row (only n_dec==1
        // dispatches step_mtp_decode; every other mix stays spec-off), so at
        // most one lane's verify columns are live at a time. With <=2 spec
        // lanes the 4 verify column slots are one shared pool instead of 4
        // per lane; wider configs keep per-lane columns.
        mtp_column_groups_ =
            mtp_enabled_ ? (max_seqs_ <= 2 ? 1U : max_seqs_) : 0U;
        // Oracle presnap slots exist only when the oracle runs (debug only,
        // NINFER_SLOT_ORACLE set at startup). Otherwise zero bytes.
        mtp_oracle_on_ =
            mtp_enabled_ && std::getenv("NINFER_SLOT_ORACLE") != nullptr;
        // Step 12 (slots-only): the per-lane shadow slots are gone. Extras
        // are the verify spare + 4 column slots per group + 3 oracle
        // presnaps (only when the oracle runs, allocation retained).
        mtp_extra_pool_slots_ =
            mtp_enabled_ ? (1U + 4U * mtp_column_groups_ + (mtp_oracle_on_ ? 3U : 0U)) : 0U;
        public_tokens_ =
            static_cast<std::int32_t>(dimension(parameters_.model.resources().public_token_count));
        spare_slot_ = mtp_enabled_ ? static_cast<std::int32_t>(max_seqs_) : -1;
        // Row 20b layout A: 4 consecutive column slots per group (k<=4
        // parametric; k=3 uses the first 3... width-4 snapshot publishes
        // col c -> block+c, commit copies slot[a] -> lane every accept.
        // Static: block base never changes, tables init-filled
        // once, never rebuilt. One group is shared by all lanes when
        // mtp_column_groups_==1 (<=2 spec lanes); otherwise one group
        // per lane (see mtp_column_slot).
        mtp_column_base_ =
            mtp_enabled_ ? static_cast<std::int32_t>(max_seqs_) + 1 : -1;
        // Row 20b oracle: three presnap slots, allocated only when
        // NINFER_SLOT_ORACLE was set at startup (-1 otherwise). The debug
        // probe itself was removed with the legacy rows (Step 12); the
        // reservation stays so pool/budget accounting keeps the 354a2eb
        // semantics (oracle on => +3 slots, else +0).
        mtp_oracle_base_ =
            mtp_oracle_on_ ? mtp_column_base_ + 4 * static_cast<std::int32_t>(mtp_column_groups_) : -1;
        hidden_      = static_cast<std::uint32_t>(dimension(text_config.hidden_size));
        max_tokens_  = options.prefill_chunk + max_seqs_;
        max_blocks_  = (max_context_ + 15U) / 16U;
        pages_per_slot_ =
            1U + (max_context_ - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
        if (static_cast<std::uint64_t>(max_blocks_) >
            static_cast<std::uint64_t>(pages_per_slot_) *
                static_cast<std::uint64_t>(kPagedKVPageSize / 16)) {
            throw std::logic_error("serve forward block stride exceeds the lane page range");
        }
        const std::uint32_t physical_pages = max_seqs_ * pages_per_slot_;

        LayoutBuilder kv_builder;
        const models::qwen3_5::DecoderStateLayout kv_layout =
            models::qwen3_5::plan_decoder_state(
                kv_builder,
                models::qwen3_5::DecoderStateSpec{
                    .full_attention_layers     = text_config.full_attention_layers,
                    // Single MTP layer (mtp.layers.0) when S7 is on; zero preserves spec-off.
                    .mtp_layers                = mtp_enabled_ ? 1u : 0u,
                    .capacity                  = max_context_,
                    .kv_heads                  = dimension(text_config.attention->num_key_value_heads),
                    .attention_head_dim        = dimension(text_config.attention->head_dim),
                    .kv_storage                = options.kv_cache,
                    .enable_mtp                = mtp_enabled_,
                    .kv_table_rows             = static_cast<std::int32_t>(max_seqs_),
                    .text_physical_page_groups = physical_pages,
                    .mtp_physical_page_groups  = mtp_enabled_ ? physical_pages : 0,
                });
        kv_store_ = DeviceBuffer(kv_builder.finish(256));
        kv_       = std::make_unique<models::qwen3_5::PagedKVCache>(
            DeviceSpan{kv_store_.p, kv_store_.bytes}, kv_layout.text_kv);
        // Memory diet (b): pool+KV sizing is derived from the memory budget.
        // NINFER_MTP_BUDGET_MIB caps the MTP device residency (MTP KV pages +
        // MTP-extra GDN pool slots + column tables); unset/0 = uncapped and
        // the sizes stay as derived above. A configuration that would exceed
        // the budget is refused here with the byte breakdown -- over-budget
        // requests are never spilled/demoted to host to make room.
        if (mtp_enabled_) {
            const std::size_t mtp_kv_bytes =
                kv_layout.mtp_kv ? kv_layout.mtp_kv->payload_bytes() : 0;
            std::size_t gdn_slot_bytes = 0;
            if (text_config.gdn && text_config.linear_attention_layers != 0) {
                const Tensor conv1(
                    nullptr, DType::BF16,
                    {static_cast<std::int32_t>(dimension(text_config.gdn->conv_channels())),
                     static_cast<std::int32_t>(
                         dimension(text_config.gdn->linear_conv_kernel_dim)) -
                         1,
                     1});
                const Tensor rec1(
                    nullptr, DType::FP32,
                    {static_cast<std::int32_t>(
                         dimension(text_config.gdn->linear_key_head_dim)),
                     static_cast<std::int32_t>(
                         dimension(text_config.gdn->linear_value_head_dim)),
                     static_cast<std::int32_t>(
                         dimension(text_config.gdn->linear_num_value_heads)),
                     1});
                gdn_slot_bytes = (conv1.bytes() + rec1.bytes()) *
                                 text_config.linear_attention_layers;
            }
            const std::size_t mtp_pool_bytes =
                static_cast<std::size_t>(mtp_extra_pool_slots_) * gdn_slot_bytes;
            const std::size_t mtp_coltab_bytes =
                static_cast<std::size_t>(max_seqs_) * 4U * 4U;
            const std::size_t mtp_residency_bytes =
                mtp_kv_bytes + mtp_pool_bytes + mtp_coltab_bytes;
            std::uint64_t budget_mib = 0;
            if (const char* budget_env = std::getenv("NINFER_MTP_BUDGET_MIB")) {
                budget_mib = std::strtoull(budget_env, nullptr, 10);
            }
            if (budget_mib != 0 &&
                mtp_residency_bytes > budget_mib * (1ULL << 20)) {
                throw std::invalid_argument(
                    "serve forward MTP residency " + std::to_string(mtp_residency_bytes) +
                    "B (kv=" + std::to_string(mtp_kv_bytes) + " pool=" +
                    std::to_string(mtp_pool_bytes) + " coltab=" +
                    std::to_string(mtp_coltab_bytes) + ") exceeds NINFER_MTP_BUDGET_MIB=" +
                    std::to_string(budget_mib) + " (refused, never spilled)");
            }
            std::fprintf(stderr,
                         "[mtp-budget] lanes=%u colgroups=%u oracle=%d extraslots=%u "
                         "pool=%zuB kv=%zuB coltab=%zuB total=%zuB budget=%lluMiB\n",
                         max_seqs_, mtp_column_groups_, mtp_oracle_on_ ? 1 : 0,
                         mtp_extra_pool_slots_, mtp_pool_bytes, mtp_kv_bytes,
                         mtp_coltab_bytes, mtp_residency_bytes,
                         static_cast<unsigned long long>(budget_mib));
        }
        if (mtp_enabled_) {
            // S7 MTP KV: one layer, one private page range per lane, lane L
            // owning the same page indices as its text range. Execution row
            // r serves lane r, so the round-state backend row is the lane.
            if (!kv_layout.mtp_kv) {
                throw std::logic_error("serve forward MTP layout is missing");
            }
            mtp_kv_ = std::make_unique<models::qwen3_5::PagedKVCache>(
                DeviceSpan{kv_store_.p, kv_store_.bytes}, *kv_layout.mtp_kv);
            std::optional<DeviceKVPageReservation> mtp_reservation =
                mtp_kv_->page_pool().reserve(physical_pages);
            if (!mtp_reservation) {
                throw std::runtime_error("serve forward MTP page reservation failed");
            }
            mtp_reservation_ = std::move(*mtp_reservation);
            mtp_page_leases_.reserve(physical_pages);
            mtp_kv_->page_pool().materialize(mtp_reservation_, physical_pages, mtp_page_leases_);
            std::vector<DeviceKVPageHandle> mtp_pages;
            mtp_pages.reserve(physical_pages);
            for (const auto& lease : mtp_page_leases_) { mtp_pages.push_back(lease.handle()); }
            mtp_kv_->page_pool().zero_pages(mtp_pages, device_.stream);
            for (std::uint32_t r = 0; r < max_seqs_; ++r) {
                mtp_row_leases_.push_back(
                    mtp_kv_->execution_tables().acquire(static_cast<std::int32_t>(r)));
                std::vector<DeviceKVPageHandle> slot_pages;
                slot_pages.reserve(pages_per_slot_);
                for (std::uint32_t l = 0; l < pages_per_slot_; ++l) {
                    slot_pages.push_back(mtp_page_leases_[r * pages_per_slot_ + l].handle());
                }
                mtp_kv_->execution_tables().publish(mtp_row_leases_.back().handle(), 0,
                                                std::span(slot_pages.data(), slot_pages.size()),
                                                device_.stream);
            }
        }

        const std::uint32_t gdn_layers = text_config.linear_attention_layers;
        LayoutBuilder pool_builder;
        const LinearAttentionStatePoolLayout pool_layout =
            plan_linear_attention_state_pool(
                pool_builder,
                LinearAttentionStatePoolSpec{
                    .layers         = gdn_layers,
                    .conv_channels  = text_config.gdn ? static_cast<std::int32_t>(
                        dimension(text_config.gdn->conv_channels())) : 0,
                    .conv_width     = text_config.gdn ? static_cast<std::int32_t>(
                        dimension(text_config.gdn->linear_conv_kernel_dim)) - 1 : 0,
                    .value_heads    = text_config.gdn ? static_cast<std::int32_t>(
                        dimension(text_config.gdn->linear_num_value_heads)) : 0,
                    .value_head_dim = text_config.gdn ? static_cast<std::int32_t>(
                        dimension(text_config.gdn->linear_value_head_dim)) : 0,
                    .key_head_dim   = text_config.gdn ? static_cast<std::int32_t>(
                        dimension(text_config.gdn->linear_key_head_dim)) : 0,
                    .slot_count     = static_cast<std::int32_t>(max_seqs_) +
                                    static_cast<std::int32_t>(mtp_extra_pool_slots_),
                    .conv_dtype     = DType::BF16,
                });
        pool_store_ = DeviceBuffer(pool_builder.finish(256));
        pool_       = std::make_unique<LinearAttentionStatePool>(
            DeviceSpan{pool_store_.p, pool_store_.bytes}, pool_layout);

        LayoutBuilder round_builder;
        models::qwen3_5::RoundStateLayout round_layout =
            models::qwen3_5::begin_round_state_layout(
                round_builder,
                models::qwen3_5::RoundStateSpec{
                    .hidden          = static_cast<std::int32_t>(hidden_),
                    .output_rows     = 1,
                    .batch_capacity  = max_seqs_,
                    .draft_window    = mtp_enabled_ ? models::qwen3_5::execution::kMtpSpecDecodeDrafts : 0,
                    .backend         = mtp_enabled_ ? SpeculativeBackend::Mtp : SpeculativeBackend::None,
                    .causal_scoring  = false,
                });
        models::qwen3_5::complete_round_state_layout(round_builder, round_layout);
        round_store_ = DeviceBuffer(round_builder.finish(256));
        io_          = std::make_unique<models::qwen3_5::RoundState>(
            DeviceSpan{round_store_.p, round_store_.bytes}, round_layout);

        models::qwen3_5::SequencePlanner planner =
            models::qwen3_5::make_sequence_planner(parameters_, device_, options);
        models::qwen3_5::SequencePlan seq_plan =
            std::move(planner).finalize(physical_pages);
        const std::size_t ws_bytes =
            seq_plan.workspace_capacity_bytes() + (static_cast<std::size_t>(96) << 20);
        if (ws_bytes == 0) { throw std::logic_error("serve forward workspace plan is empty"); }
        work_ = std::make_unique<DeviceArena>(ws_bytes);

        auto align_up = [](std::size_t bytes) { return (bytes + 255U) & ~static_cast<std::size_t>(255U); };
        std::size_t off = 0;
        ids_ = off; off += align_up(static_cast<std::size_t>(max_tokens_) * 4U);
        cpos_ = off; off += align_up(static_cast<std::size_t>(max_tokens_) * 4U);
        rpos_ = off; off += align_up(static_cast<std::size_t>(max_tokens_) * 4U);
        rows_ = off; off += align_up(static_cast<std::size_t>(max_seqs_) * 4U);
        gsrc_ = off; off += align_up(static_cast<std::size_t>(max_seqs_) * 4U);
        gdst_ = off; off += align_up(static_cast<std::size_t>(max_seqs_) * 4U);
        hidden_off_ = off;
        off += align_up(static_cast<std::size_t>(hidden_) * max_tokens_ * 2U);
        prefill_hidden_ = off;
        off += align_up(static_cast<std::size_t>(hidden_) * 2U);
        offsets_ = off;
        off += align_up(static_cast<std::size_t>(max_seqs_ + 1U) * 4U);
        tables_ = off;
        off += align_up(static_cast<std::size_t>(max_seqs_) * max_blocks_ * 4U);
        // S7 single-row staging (I32) plus MTP hidden/logits scratch. The MTP
        // decode path never runs forward_serve_step, so these regions are
        // disjoint from the plan tensors above by construction.
        mtp_ids_ = off; off += align_up(4);
        mtp_pos_ = off; off += align_up(4);
        mtp_row_ = off; off += align_up(4);
        mtp_ssrc_ = off; off += align_up(4);
        mtp_sdst_ = off; off += align_up(4);
        mtp_tok_ = off; off += align_up(4);
        // Verify argmax outbox: one I32 slot per verify row so all four
        // argmaxes can enqueue before the single batch sync (row 13).
        mtp_vtok_ = off; off += align_up(4U * 4U);
        // Lane B: single width-4 target verify staging (ids/positions over
        // the window, one KV row / src / valid slot, [H,4] + [V,4]
        // outputs). Token outbox reuses mtp_vtok_. Disjoint from the fill
        // regions by construction (decode never runs the prefill fill).
        mtp_vids_ = off; off += align_up(4U * 4U);
        mtp_vpos_ = off; off += align_up(4U * 4U);
        mtp_vrow_ = off; off += align_up(4);
        mtp_vsrc_ = off; off += align_up(4);
        mtp_vval_ = off; off += align_up(4);
        mtp_vhid_ = off; off += align_up(static_cast<std::size_t>(hidden_) * 4U * 2U);
        mtp_vlog_ = off;
        off += align_up(static_cast<std::size_t>(text_vocab_) * 4U * 2U);
        mtp_fill_ids_ = off; off += align_up(8U * 4U);
        mtp_fill_pos_ = off; off += align_up(8U * 4U);
        mtp_fill_rows_ = off; off += align_up(8U * 4U);
        mtp_fill_ssrc_ = off; off += align_up(8U * 4U);
        mtp_fill_sdst_ = off; off += align_up(8U * 4U);
        mtp_hid1_ = off; off += align_up(static_cast<std::size_t>(hidden_) * 2U);
        mtp_log1_ = off;
        off += align_up(static_cast<std::size_t>(text_vocab_) * 2U);
        mtp_mha_ = off; off += align_up(static_cast<std::size_t>(hidden_) * 2U);
        mtp_mhb_ = off; off += align_up(static_cast<std::size_t>(hidden_) * 2U);
        mtp_fill_hid_ = off; off += align_up(static_cast<std::size_t>(hidden_) * 8U * 2U);
        mtp_fill_log_ = off;
        off += align_up(static_cast<std::size_t>(text_vocab_) * 8U * 2U);
        mtp_fill_mh_ = off; off += align_up(static_cast<std::size_t>(hidden_) * 8U * 2U);
        // Row-23 rung-2: batched warming works one prefill_chunk-wide slice
        // at a time over prefill hidden columns (no assembly copy: a slice
        // span is already a contiguous {H,T} block). Discard output + H2D
        // id/pos staging live here.
        mtp_batch_ids_ = off; off += align_up(static_cast<std::size_t>(warm_chunk_) * 4U);
        mtp_batch_pos_ = off; off += align_up(static_cast<std::size_t>(warm_chunk_) * 4U);
        mtp_batch_out_ = off;
        off += align_up(static_cast<std::size_t>(hidden_) * warm_chunk_ * 2U);
        scratch_ = DeviceBuffer(off);
        // Per-lane committed target hidden (MTP draft chain anchor) plus the
        // anchor-as-input target logits (verify row 0 without re-running the
        // anchor through GDN). anchor_valid means both are live.
        anchor_store_ =
            DeviceBuffer(static_cast<std::size_t>(hidden_) * max_seqs_ * 2U);
        anchor_logits_ =
            DeviceBuffer(static_cast<std::size_t>(text_vocab_) * max_seqs_ * 2U);
        // Row-23 conc resume (task 8): per-lane staging for the forced-off
        // window's target-hidden columns (OFF steps x H per lane). Zero bytes
        // unless the toggle is armed; decode steps never touch the rung-2
        // batch regions while staging here, and the fill never touches this.
        if (mtp_enabled_ && toggle_off_ > 0) {
            catchup_store_ = DeviceBuffer(static_cast<std::size_t>(hidden_) *
                                          toggle_off_ * max_seqs_ * 2U);
        }
        // Row 20b layout A: static column-slot tables, init-filled
        // once: t[c] = column block, 4 consecutive slots per group. The
        // width-4 snapshot kernel publishes col c -> base+c, then commit
        // copies slot[a] -> lane (every accept incl a=0). With a shared
        // pool (<=2 spec lanes) every lane's row points at the same 4
        // slots, safe because the MTP step is single-row: one lane's
        // snapshot+commit completes before the next lane starts.
        if (mtp_enabled_) {
            mtp_coltab_store_ = DeviceBuffer(static_cast<std::size_t>(max_seqs_) * 4U * 4U);
            std::vector<std::int32_t> coltab(static_cast<std::size_t>(max_seqs_) * 4U);
            for (std::uint32_t lane = 0; lane < max_seqs_; ++lane) {
                for (std::int32_t c = 0; c < 4; ++c)
                    coltab[static_cast<std::size_t>(lane) * 4U + static_cast<std::size_t>(c)] =
                        mtp_column_slot(static_cast<std::int32_t>(lane), c);
            }
            CUDA_CHECK(cudaMemcpy(mtp_coltab_store_.p, coltab.data(), coltab.size() * 4U,
                                  cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaMemsetAsync(static_cast<char*>(scratch_.p) + tables_, 0,
                                   static_cast<std::size_t>(max_seqs_) * max_blocks_ * 4U,
                                   device_.stream));

        std::optional<DeviceKVPageReservation> reservation =
            kv_->page_pool().reserve(physical_pages);
        if (!reservation) { throw std::runtime_error("serve forward KV page reservation failed"); }
        reservation_ = std::move(*reservation);
        page_leases_.reserve(physical_pages);
        kv_->page_pool().materialize(reservation_, physical_pages, page_leases_);
        std::vector<DeviceKVPageHandle> all_pages;
        all_pages.reserve(physical_pages);
        for (const auto& lease : page_leases_) { all_pages.push_back(lease.handle()); }
        kv_->page_pool().zero_pages(all_pages, device_.stream);
        {
            // Construction invariant for the per-step table gather below.
            const std::size_t npages = all_pages.size();
            if (kv_->page_pool().contiguous_run_count(
                    std::span<const DeviceKVPageHandle>(all_pages.data(), npages)) != 1) {
                throw std::logic_error("serve forward KV pages are not dense in lane order");
            }
        }
        for (std::uint32_t r = 0; r < max_seqs_; ++r) {
            row_leases_.push_back(kv_->execution_tables().acquire(static_cast<std::int32_t>(r)));
            std::vector<DeviceKVPageHandle> slot_pages;
            slot_pages.reserve(pages_per_slot_);
            for (std::uint32_t l = 0; l < pages_per_slot_; ++l) {
                slot_pages.push_back(page_leases_[r * pages_per_slot_ + l].handle());
            }
            kv_->execution_tables().publish(row_leases_.back().handle(), 0,
                                            std::span(slot_pages.data(), slot_pages.size()),
                                            device_.stream);
        }

        Tensor prefill_hidden_tensor(static_cast<char*>(scratch_.p) + prefill_hidden_,
                                     DType::BF16,
                                     {static_cast<std::int32_t>(hidden_), 1});
        card_ = std::make_unique<models::qwen3_5::execution::TextContext>(
            device_, parameters_, *work_, models::qwen3_5::PagedKVCacheView{}, *pool_, *io_,
            prefill_hidden_tensor, options.prefill_chunk, 0,
            models::qwen3_5::PagedKVCacheView{}, kv_.get(),
            mtp_enabled_ ? mtp_kv_.get() : nullptr);
        envelope_ = ops::CausalAttentionExecutionEnvelope{1, max_context_};
        {
            // Tier 0 (2026-09-25): default ON. NINFER_SERVE_GRAPH=0 opts out;
            // =verbose/=dry keep their diagnose modes. Spec-off dec/M=1 seam
            // is gated !mtp (graph_eligible); MTP keeps draft/bonus eager by
            // construction and the ver/M=4 seam stays on under this flag.
            const char* graph_env = std::getenv("NINFER_SERVE_GRAPH");
            const std::string graph_mode(graph_env != nullptr ? graph_env : "1");
            graphs_enabled_ = (graph_mode != "0");
            graph_verbose_  = (graph_mode == "verbose");
            graph_dry_      = (graph_mode == "dry");
            if (graph_dry_) {
                graph_verbose_ = true;
            }
        }
        {
            // Explicit T=0 sampling config: temperature 0 resolves to greedy.
            // Exactness note (audit 2026-09-28): the greedy sampler never
            // consults top_k — single-block does a full scan over
            // token_domain (sampling.cuh sample_row_kernel early-return),
            // multi-block keeps per-tile bests and max-reduces
            // (sampling_candidate_cap is reached only on the T>0 path).
            // top_k = 0 (no truncation) belt-and-braces: full-vocab argmax
            // even if a future kernel routes greedy through the cap.
            ops::SamplingConfig explicit_argmax;
            explicit_argmax.temperature = 0.0F;
            explicit_argmax.top_k = 0;
            explicit_argmax.top_p = 1.0F;
            explicit_argmax.min_p = 0.0F;
            explicit_argmax.presence_penalty = 0.0F;
            explicit_argmax.frequency_penalty = 0.0F;
            explicit_argmax.seed = 0;
            explicit_argmax.token_counts = nullptr;
            sampling_store_ = DeviceBuffer(sizeof(ops::SamplingConfig));
            sampling_store_.copy_from_host(&explicit_argmax, sizeof(explicit_argmax));
            serve_sampling_ = static_cast<const ops::SamplingConfig*>(sampling_store_.p);
            card_->set_sampling(serve_sampling_);
        }
        if (mtp_enabled_) {
            // The MTP AR draft path derives RoPE from positions plus the
            // round-state rope delta: pin it to zero so drafts share the
            // ordinary path's absolute positions.
            CUDA_CHECK(cudaMemsetAsync(io_->rope_delta.data, 0, sizeof(std::int32_t),
                                       device_.stream));
        }
        slots_.resize(max_seqs_);
        device_.synchronize();
    }

    // Row-23 conc resume (task 8): helpers defined after mtp_prefill_fill.
    // (No forward declarations: in-class call sites resolve at the
    // complete-class context, and separate declarations trip the
    // overload checker on this toolchain.)
    [[nodiscard]] bool toggle_armed() const noexcept {
        return mtp_enabled_ && toggle_off_ > 0;
    }
    [[nodiscard]] bool in_toggle_window(std::uint32_t toggle_seen) const noexcept {
        return toggle_seen > toggle_after_ && toggle_seen <= toggle_after_ + toggle_off_;
    }

    runtime::StepDecodedPairs step(const batch::StepPlan& plan,
                                   const batch::StepDispatch& dispatch) {
        // Same pins as run_step_forward: M == prefill + n_decode, rows == M,
        // and an empty plan never runs.
        runtime::validate_mixed_step_m(plan, dispatch);
        const std::uint32_t m = runtime::step_activation_rows(dispatch);
        if (m == 0) { throw std::logic_error("step_forward: empty plan never runs a forward"); }
        if (parameters_.text.layers.empty()) {
            throw std::invalid_argument("step_forward: num_layers must be positive");
        }
        const batch::RaggedBatch& batch = dispatch.batch;
        const std::size_t num_seqs      = batch.num_seqs();
        const std::size_t n_dec         = plan.decode_seq_ids.size();
        const std::size_t n_pref        = plan.prefill.size();
        if (m > max_tokens_) { throw std::logic_error("serve forward step exceeds token budget"); }
        if (num_seqs > max_seqs_) { throw std::logic_error("serve forward step exceeds lane budget"); }
        if (std::getenv("NINFER_SERVE_STEP_TRACE") != nullptr) {
            // DIAGNOSTIC ONLY: clear shared workspace per step to test for
            // read-before-write contamination across requests.
            CUDA_CHECK(cudaMemsetAsync(work_->base(), 0, work_->capacity(), device_.stream));
        }

        // Lane assignment: one persistent (KV range, GDN slot) per seq id.
        // Fresh lanes zero their pages and slot on the step stream, ordered
        // before the forward below.
        std::vector<std::int32_t> row_slot(num_seqs, -1);
        for (std::size_t s = 0; s < num_seqs; ++s) {
            const std::uint64_t seq_id = batch.seq_ids[s];
            std::int32_t found         = -1;
            for (std::uint32_t q = 0; q < max_seqs_; ++q) {
                if (slots_[q].in_use && slots_[q].seq_id == seq_id) {
                    found = static_cast<std::int32_t>(q);
                    break;
                }
            }
            if (found < 0) {
                for (std::uint32_t q = 0; q < max_seqs_; ++q) {
                    if (!slots_[q].in_use) {
                        found = static_cast<std::int32_t>(q);
                        break;
                    }
                }
                if (found < 0) {
                    // Admit-time reclaim of a dead lane: absent from this
                    // batch and decode-ended. Live decode seqs ride every
                    // step, so absence means finished.
                    for (std::uint32_t q = 0; q < max_seqs_; ++q) {
                        if (slots_[q].touched_as_prefill) { continue; }
                        bool alive = false;
                        for (std::size_t s = 0; s < num_seqs; ++s) {
                            if (batch.seq_ids[s] == slots_[q].seq_id) {
                                alive = true;
                                break;
                            }
                        }
                        if (!alive) {
                            found = static_cast<std::int32_t>(q);
                            break;
                        }
                    }
                }
                if (found < 0) { throw std::logic_error("serve forward has no free lane"); }
                ServeSlot& slot         = slots_[static_cast<std::size_t>(found)];
                slot.in_use             = true;
                slot.seq_id             = seq_id;
                slot.next_pos           = 0;
                slot.touched_as_prefill = false;
                slot.mtp_valid_pos      = 0;
                slot.anchor_valid       = false;
                slot.mtp_toggle_seen    = 0;
                slot.stage_start        = 0;
                slot.staged_n           = 0;
                slot.staged_overflow    = false;
                slot.staged_ids.clear();
                slot.staged_pos.clear();
                std::vector<DeviceKVPageHandle> pages;
                pages.reserve(pages_per_slot_);
                for (std::uint32_t l = 0; l < pages_per_slot_; ++l) {
                    pages.push_back(
                        page_leases_[static_cast<std::size_t>(found) * pages_per_slot_ + l]
                            .handle());
                }
                kv_->page_pool().zero_pages(pages, device_.stream);
                pool_->zero_slot(found, device_.stream);
                graph_forget_lane(found);
            }
            row_slot[s] = found;
        }

        // Absolute positions per flat token; the per-lane next_pos cursor is
        // the only cross-step state (scheduler slices arrive in order).
        std::vector<std::int32_t> pos(m, 0);
        for (std::size_t s = 0; s < n_pref; ++s) {
            const batch::PrefillSlice& slice = plan.prefill[s];
            if (slice.seq_id != batch.seq_ids[s]) {
                throw std::logic_error("serve forward prefill row mismatch");
            }
            ServeSlot& slot = slots_[static_cast<std::size_t>(row_slot[s])];
            if (slice.offset != slot.next_pos) {
                throw std::logic_error("serve forward lost position track");
            }
            if (slice.count == 0 ||
                static_cast<std::uint64_t>(slice.offset) + slice.count > max_context_) {
                throw std::logic_error("serve forward prefill slice is out of range");
            }
            const std::uint32_t begin = batch.seq_offsets[s];
            for (std::uint32_t k = 0; k < slice.count; ++k) {
                pos[begin + k] = static_cast<std::int32_t>(slot.next_pos + k);
            }
            slot.next_pos += slice.count;
            slot.touched_as_prefill = true;
        }
        for (std::size_t i = 0; i < n_dec; ++i) {
            const std::size_t s = n_pref + i;
            if (plan.decode_seq_ids[i] != batch.seq_ids[s]) {
                throw std::logic_error("serve forward decode row mismatch");
            }
            ServeSlot& slot = slots_[static_cast<std::size_t>(row_slot[s])];
            if (slot.next_pos >= max_context_) {
                throw std::logic_error("serve forward decode position is out of range");
            }
            pos[batch.seq_offsets[s]] = static_cast<std::int32_t>(slot.next_pos);
            slot.next_pos += 1;
            slot.touched_as_prefill = false;
        }
        std::vector<std::int32_t> rows(num_seqs, 0);
        for (std::size_t s = 0; s < num_seqs; ++s) { rows[s] = row_slot[s]; }
        std::vector<std::int32_t> gslots(n_dec, 0);
        for (std::size_t i = 0; i < n_dec; ++i) { gslots[i] = row_slot[n_pref + i]; }
        for (std::size_t s = 0; s < num_seqs; ++s) {
            if (row_slot[s] < 0) {
                throw std::logic_error("serve forward left a ragged row unbound");
            }
            for (std::size_t t = s + 1; t < num_seqs; ++t) {
                if (row_slot[s] == row_slot[t]) {
                    throw std::logic_error("serve forward aliased two ragged rows to one lane");
                }
            }
        }

        // S7 MTP-3 (single slot): decode-only single-row steps with a warm
        // MTP lane run draft-3 + target-verify + longest-prefix accept below.
        // Every other mix (mixed prefill+decode, multi-row decode, cold lane,
        // frontier at the context edge) runs the ordinary target-only path:
        // mixed plans stay spec-off.
        // NINFER_MTP_FORCE_OFF=1 forces the ordinary path (adaptive gate's
        // fallback + kill switch; default unset = spec allowed). Read once
        // per step (cheap getenv; correctness over caching).
        const bool mtp_force_off = std::getenv("NINFER_MTP_FORCE_OFF") != nullptr;
        // Row-23 conc resume (task 8): mid-request toggle + gap resume. When
        // disarmed (default) the block below reduces to the gate above.
        const bool armed = toggle_armed();
        if (mtp_enabled_ && !mtp_force_off && n_pref == 0 && n_dec == 1) {
            const std::int32_t mtp_lane = row_slot[0];
            ServeSlot& mslot = slots_[static_cast<std::size_t>(mtp_lane)];
            const std::uint32_t frontier = mslot.next_pos - 1;
            bool window = false;
            if (armed) {
                toggle_tick_lane(mtp_lane);
                window = in_toggle_window(mslot.mtp_toggle_seen);
            }
            if (!window && mslot.next_pos >= 1 && mslot.mtp_valid_pos == frontier &&
                frontier + models::qwen3_5::execution::kMtpSpecDecodeDrafts + 1 <=
                    max_context_) {
                return step_mtp_decode(batch, row_slot);
            }
            if (!window && armed && mslot.next_pos >= 1 && mslot.mtp_valid_pos < frontier &&
                frontier + models::qwen3_5::execution::kMtpSpecDecodeDrafts + 1 <=
                    max_context_) {
                // Resume after the forced-off window (or any staged gap): the
                // staged history exactly covers [mtp_valid_pos, frontier), so
                // rebuild the MTP KV prefix + anchor stash with one batched
                // fill and run the MTP step. Coverage mismatch fails closed
                // to the ordinary path below (today's behavior).
                if (mtp_resume_catchup(
                        mtp_lane, batch.tokens[batch.seq_offsets[0]], frontier)) {
                    return step_mtp_decode(batch, row_slot);
                }
            }
            // Window (forced spec-off) and gate misses fall through to the
            // ordinary path; window steps stage hidden below for the resume.
        } else if (armed && !mtp_force_off && n_pref == 0 && n_dec > 1) {
            // Conc-N decode steps are shape-ineligible for MTP but still
            // advance MTP holes: tick every lane so a window spanning conc-2
            // stages the same per-lane history.
            for (std::size_t i = 0; i < n_dec; ++i) {
                toggle_tick_lane(row_slot[n_pref + i]);
            }
        }
        cudaStream_t stream = device_.stream;
        char* base          = static_cast<char*>(scratch_.p);
        CUDA_CHECK(cudaMemcpyAsync(base + ids_, batch.tokens.data(),
                                   static_cast<std::size_t>(m) * sizeof(TokenId),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(base + cpos_, pos.data(),
                                   static_cast<std::size_t>(m) * sizeof(std::int32_t),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(base + rpos_, pos.data(),
                                   static_cast<std::size_t>(m) * sizeof(std::int32_t),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(base + rows_, rows.data(),
                                   num_seqs * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                   stream));
        if (n_dec > 0) {
            CUDA_CHECK(cudaMemcpyAsync(base + gsrc_, gslots.data(),
                                       n_dec * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                       stream));
            CUDA_CHECK(cudaMemcpyAsync(base + gdst_, gslots.data(),
                                       n_dec * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                       stream));
        }
        CUDA_CHECK(cudaMemcpyAsync(base + offsets_, batch.seq_offsets.data(),
                                   (num_seqs + 1U) * sizeof(std::uint32_t),
                                   cudaMemcpyHostToDevice, stream));
        {
            // Per-step block-table gather: device tables must match this
            // steps plan every step, never the startup zeros. Lane L owns
            // pages [L * pages_per_slot_, (L + 1) * pages_per_slot_) and a
            // 64-token page holds four 16-token logical blocks.
            host_tables_.assign(num_seqs * max_blocks_, -1);
            for (std::size_t s = 0; s < num_seqs; ++s) {
                const std::uint32_t lane = static_cast<std::uint32_t>(row_slot[s]);
                for (std::uint32_t b = 0; b < max_blocks_; ++b) {
                    host_tables_[s * max_blocks_ + b] = static_cast<std::int32_t>(
                        lane * pages_per_slot_ + b / (kPagedKVPageSize / 16));
                }
                if (batch.seq_offsets[s + 1] == batch.seq_offsets[s]) {
                    throw std::logic_error("serve forward step has an empty ragged row");
                }
                if (lane > 0 && host_tables_[s * max_blocks_] == 0) {
                    throw std::logic_error(
                        "serve forward block tables were not refreshed for this step");
                }
            }
            CUDA_CHECK(cudaMemcpyAsync(base + tables_, host_tables_.data(),
                                       host_tables_.size() * sizeof(std::int32_t),
                                       cudaMemcpyHostToDevice, stream));
        }

        batch::DeviceRaggedBatch view;
        view.tokens        = reinterpret_cast<TokenId*>(base + ids_);
        view.seq_offsets   = reinterpret_cast<std::uint32_t*>(base + offsets_);
        view.block_tables  = reinterpret_cast<std::int32_t*>(base + tables_);
        view.num_seqs      = static_cast<std::uint32_t>(num_seqs);
        view.total_tokens  = m;
        view.max_blocks    = max_blocks_;

        const std::int32_t T    = static_cast<std::int32_t>(m);
        const std::int32_t nseq = static_cast<std::int32_t>(num_seqs);
        const std::int32_t ndec = static_cast<std::int32_t>(n_dec);
        models::qwen3_5::execution::TextContext::ServeStepTensors tensors{
            .ids                      = Tensor(base + ids_, DType::I32, {T}),
            .cache_positions          = Tensor(base + cpos_, DType::I32, {T}),
            .rope_positions           = Tensor(base + rpos_, DType::I32, {T}),
            .kv_table_rows            = Tensor(base + rows_, DType::I32, {nseq}),
            .decode_source_slots      = ndec > 0 ? Tensor(base + gsrc_, DType::I32, {ndec})
                                                 : Tensor{},
            .decode_destination_slots = ndec > 0 ? Tensor(base + gdst_, DType::I32, {ndec})
                                                 : Tensor{},
            .hidden                   = Tensor(base + hidden_off_, DType::BF16,
                                               {static_cast<std::int32_t>(hidden_), T}),
        };
        card_->set_sampling(serve_sampling_);
        if (std::getenv("NINFER_SERVE_STEP_TRACE") != nullptr) {
            std::fprintf(stderr, "[serve-step] seqs=%zu m=%u prefill=%zu decode=%zu\n",
                         num_seqs, m, n_pref, n_dec);
            for (std::size_t s = 0; s < num_seqs; ++s) {
                const std::uint32_t b0 = batch.seq_offsets[s];
                const std::uint32_t b1 = batch.seq_offsets[s + 1];
                std::fprintf(stderr,
                             "[serve-step] row=%zu seq=%llu lane=%d span=%u tok0=%d tokN=%d pos0=%d posN=%d\n",
                             s, (unsigned long long)batch.seq_ids[s], row_slot[s], b1 - b0,
                             batch.tokens[b0], batch.tokens[b1 - 1], pos[b0], pos[b1 - 1]);
            }
            std::fflush(stderr);
        }
        runtime::StepDecodedPairs decoded;
        if (mtp_enabled_ && n_pref > 0) {
            // S7 fill seeding: snapshot each prefill lane's pre-forward GDN
            // state into fill scratch. The ordinary forward below advances
            // lanes in place (destroying pre-slice states), and the fill
            // chains its rows from these snapshots so every row applies
            // exactly once. Scratch borrows the column region (slice s ->
            // column base + s, at most max_seqs_ <= 4*groups slots): the
            // width-4 verify never runs in a prefill step, so no live
            // verify state is aliased. One scratch slot per prefill lane:
            // concurrent prefill lanes never share.
            for (std::size_t s = 0; s < n_pref; ++s) {
                pool_->copy_slot(row_slot[s],
                                 mtp_column_base_ + static_cast<std::int32_t>(s), stream);
            }
        }
        decoded = step_decode_layers(plan, batch, view, tensors, row_slot, n_pref, n_dec);
        if (armed && n_pref == 0 && n_dec > 0) {
            // Forced-off window steps leave MTP KV behind but keep the true
            // target hidden live in tensors.hidden: stage each in-window
            // lane's column for the batched resume fill.
            mtp_stage_decode_hidden(batch, row_slot, n_pref, n_dec, pos, tensors.hidden);
        }
        if (mtp_enabled_ && n_pref > 0) {
            // S7 MTP-KV fill: the ordinary forward above advanced text KV
            // and GDN only. Mirror each prefill slice through the MTP layer
            // (chunked to the single-row helper width) so the MTP KV prefix
            // stays warm for later decode-only MTP steps. tensors.hidden
            // still holds every prefill row's target hidden (rung-1 input).
            mtp_prefill_fill(plan, batch, row_slot, pos, tensors.hidden);
            if (armed) {
                // The fill rewarmed every prefill lane (mtp_valid_pos held at
                // next_pos, fresh anchor): staged window history for those
                // lanes is stale, drop it. Decode rows riding a mixed step
                // advanced without staging: their coverage is unrecoverable,
                // fail those lanes closed (resume falls back to ordinary).
                for (std::size_t r = 0; r < plan.prefill.size(); ++r) {
                    ServeSlot& pslot = slots_[static_cast<std::size_t>(row_slot[r])];
                    pslot.staged_n        = 0;
                    pslot.staged_overflow = false;
                    pslot.staged_ids.clear();
                    pslot.staged_pos.clear();
                    pslot.stage_start = pslot.mtp_valid_pos;
                }
                for (std::size_t i = 0; i < n_dec; ++i) {
                    slots_[static_cast<std::size_t>(row_slot[n_pref + i])].staged_overflow = true;
                }
            }
        }
        if (std::getenv("NINFER_SERVE_STEP_TRACE") != nullptr) {
            for (std::size_t d = 0; d < decoded.size(); ++d) {
                std::fprintf(stderr, "[serve-step] decoded seq=%llu tok=%d\n",
                             (unsigned long long)decoded[d].first, (int)decoded[d].second);
            }
            std::fflush(stderr);
        }
        // Lanes whose seq left the batch after decoding are done; lanes that
        // vanish mid-prefill are starved (chunk budget), not done: keep them.
        for (ServeSlot& slot : slots_) {
            if (!slot.in_use) { continue; }
            bool seen = false;
            for (std::size_t s = 0; s < num_seqs; ++s) {
                if (batch.seq_ids[s] == slot.seq_id) {
                    seen = true;
                    break;
                }
            }
            if (!seen && !slot.touched_as_prefill) { slot.in_use = false; }
        }
        return decoded;
    }

    // Single target row through the ordinary ladder: KV overwrite-identical at
    // position, GDN from gdn_src into gdn_dst, hidden/logits out. The rewind
    // discipline is the caller's: speculative rows run with dst == spare and
    // only the accepted prefix is replayed lane-into-lane.
    void run_single_row(std::int32_t tok, std::uint32_t position, std::int32_t kv_row,
                        std::int32_t gdn_src, std::int32_t gdn_dst, Tensor& hidden_out,
                        Tensor& logits_out, bool allow_graph = false) {
        cudaStream_t stream = device_.stream;
        char* mbase         = static_cast<char*>(scratch_.p);
        const std::int32_t p = static_cast<std::int32_t>(position);
        // Tight envelope {p+1,p+1} like the reference bridge/target envelopes:
        // the serve envelope_ spans the context window and admits a different
        // (potentially nondeterministic) attention route over unwritten cache.
        const ops::CausalAttentionExecutionEnvelope row_env{static_cast<std::uint32_t>(p + 1),
                                                           static_cast<std::uint32_t>(p + 1)};
        CUDA_CHECK(
            cudaMemcpyAsync(mbase + mtp_ids_, &tok, sizeof(tok), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(
            cudaMemcpyAsync(mbase + mtp_pos_, &p, sizeof(p), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(mbase + mtp_row_, &kv_row, sizeof(kv_row),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(mbase + mtp_ssrc_, &gdn_src, sizeof(gdn_src),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(mbase + mtp_sdst_, &gdn_dst, sizeof(gdn_dst),
                                   cudaMemcpyHostToDevice, stream));
        const Tensor ids_t(mbase + mtp_ids_, DType::I32, {1});
        const Tensor pos_t(mbase + mtp_pos_, DType::I32, {1});
        const Tensor row_t(mbase + mtp_row_, DType::I32, {1});
        const Tensor src_t(mbase + mtp_ssrc_, DType::I32, {1});
        const Tensor dst_t(mbase + mtp_sdst_, DType::I32, {1});
        // Row-18: frozen dec/M=1 commit exec (GLOBAL, same address argument
        // as ver/M=4: pool bases + mbase scratch + reset arena are stable;
        // per-row tok/pos/row/slots are contents at fixed addresses,
        // rewritten above). Commit rows all land in hid1/log1; the anchor
        // replay uses anchor_hid and stays eager (allow_graph=false, once
        // per admission). First commit row eager (seen rule), capture 2nd,
        // immediate replay (WSL rule). Fail-closed to eager.
        const bool cgraph_on =
            allow_graph && graphs_enabled_ && mtp_enabled_ && !graph_dry_ && !commit_dead_;
        if (cgraph_on && !commit_seen_) commit_seen_ = true;
        auto cglog = [&](const char* act) {
            if (graph_verbose_) {
                std::fprintf(stderr, "[cgraph] lane=%d pos=%d action=%s\n", kv_row, p, act);
            }
        };
        if (cgraph_on && commit_seen_ && commit_exec_ != nullptr) {
            CUDA_CHECK(cudaGraphLaunch(commit_exec_, stream));
            cglog("replay");
        } else if (cgraph_on && commit_seen_ && commit_exec_ == nullptr) {
            cudaGraph_t cgraph = nullptr;
            bool cok           = false;
            if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal) == cudaSuccess) {
                try {
                    card_->ordinary_decode_batch(ids_t, pos_t, pos_t, row_t, src_t, dst_t,
                                                 row_env, hidden_out, logits_out);
                } catch (...) {
                    cudaStreamEndCapture(stream, &cgraph);
                    if (cgraph != nullptr) cudaGraphDestroy(cgraph);
                    commit_dead_ = true;
                    throw;
                }
                if (cudaStreamEndCapture(stream, &cgraph) == cudaSuccess &&
                    cgraph != nullptr) {
                    cudaGraphExec_t cexec = nullptr;
                    if (cudaGraphInstantiate(&cexec, cgraph, nullptr, nullptr, 0) ==
                            cudaSuccess &&
                        cexec != nullptr) {
                        cudaGraphDestroy(cgraph);
                        commit_exec_ = cexec;
                        CUDA_CHECK(cudaGraphLaunch(commit_exec_, stream));
                        cglog("capture");
                        cok = true;
                    } else {
                        if (cexec != nullptr) cudaGraphExecDestroy(cexec);
                        cudaGraphDestroy(cgraph);
                    }
                } else if (cgraph != nullptr) {
                    cudaGraphDestroy(cgraph);
                }
            }
            if (!cok) {
                commit_dead_ = true;
                card_->ordinary_decode_batch(ids_t, pos_t, pos_t, row_t, src_t, dst_t,
                                             row_env, hidden_out, logits_out);
                cglog("capture-fail");
            }
        } else {
            card_->ordinary_decode_batch(ids_t, pos_t, pos_t, row_t, src_t, dst_t, row_env,
                                         hidden_out, logits_out);
            cglog(!cgraph_on ? "eager-disabled" : "eager-first");
        }
    }

    // S7 MTP-3 single-slot decode: draft 3 from the in-checkpoint head, one
    // width-4 target verify over bonus + drafts (spare-sourced, one
    // sync), longest-prefix accept, selective replay commit. The
    // lane (text KV cursor, GDN state) is untouched until the commit: a
    // rejection rewinds by construction (cursor discipline, spare discarded,
    // only the accepted prefix replayed). Returns 1 + accepted pairs.
    runtime::StepDecodedPairs step_mtp_decode(const batch::RaggedBatch& batch,
                                              const std::vector<std::int32_t>& row_slot) {
        constexpr std::uint32_t kDrafts = models::qwen3_5::execution::kMtpSpecDecodeDrafts;
        static_assert(kDrafts == 3, "S7 serve uses MTP-3, never the 10-wide window");
        cudaStream_t stream = device_.stream;
        char* mbase         = static_cast<char*>(scratch_.p);
        const std::int32_t H     = static_cast<std::int32_t>(hidden_);
        const std::int32_t V     = static_cast<std::int32_t>(text_vocab_);
        const std::int32_t lane  = row_slot[0];
        ServeSlot& slot          = slots_[static_cast<std::size_t>(lane)];
        const std::uint64_t seq_id = batch.seq_ids[0];
        const std::uint32_t F      = slot.next_pos - 1;
        const TokenId anchor       = batch.tokens[batch.seq_offsets[0]];
        const std::int32_t spare   = spare_slot_;
        const Tensor tok_in(mbase + mtp_ids_, DType::I32, {1});
        const Tensor pos_t(mbase + mtp_pos_, DType::I32, {1});
        Tensor hid1(mbase + mtp_hid1_, DType::BF16, {H, 1});
        Tensor log1(mbase + mtp_log1_, DType::BF16, {V, 1});
        Tensor tok_out(mbase + mtp_tok_, DType::I32, {1});
        Tensor mh[2] = {Tensor(mbase + mtp_mha_, DType::BF16, {H, 1}),
                        Tensor(mbase + mtp_mhb_, DType::BF16, {H, 1})};
        Tensor anchor_hid(static_cast<char*>(anchor_store_.p) +
                                  static_cast<std::size_t>(lane) * hidden_ * 2U,
                          DType::BF16, {H, 1});
        Tensor anchor_log(static_cast<char*>(anchor_logits_.p) +
                                  static_cast<std::size_t>(lane) * text_vocab_ * 2U,
                          DType::BF16, {V, 1});
        // Stash discipline: anchor_valid means anchor_store_ and
        // anchor_logits_ hold this anchor's exact as-input hidden and logits
        // (stashed by the fill or the commit, both chained exactly). The
        // anchor row must never be re-run: the lane already contains it, so a
        // re-run would apply the GDN update twice. Fall back to a single-row
        // anchor replay only when no stash is live (synthetic warmup lanes).
        const bool stash_live = slot.anchor_valid && slot.anchor_token == anchor;
        if (!stash_live) {
            // First MTP step on this lane: materialize the anchor target
            // hidden (replays the anchor row; KV overwrite-identical, GDN
            // into spare so the snapshot below stays exact).
            run_single_row(static_cast<std::int32_t>(anchor), F - 1, lane, lane, spare,
                           anchor_hid, log1);
            slot.anchor_token = anchor;
            slot.anchor_valid = false;
        }
        // The MTP tail consumes the single-row path via the round-state
        // backend row (no batch bindings): point it at this lane's MTP row.
        CUDA_CHECK(cudaMemcpyAsync(io_->backend_kv_table_row.data, &lane, sizeof(lane),
                                   cudaMemcpyHostToDevice, stream));
        std::int32_t host_targets[4] = {0, 0, 0, 0};
        // Stashed anchor logits resolve the bonus token before drafting, so
        // the first draft consumes the reference bridge input (bonus token
        // over the anchor hidden) instead of re-consuming the anchor. Cold
        // lanes (no stash) replay the anchor row once above; its logits are
        // in log1, so the bonus comes from there in both cases.
        if (stash_live) {
            ops::argmax(anchor_log, tok_out, public_tokens_, stream);
        } else {
            ops::argmax(log1, tok_out, public_tokens_, stream);
        }
        {
            // Explicit sync: the compute stream is non-blocking, so a host
            // read must wait for the queued argmax (else stale mtp_tok_).
            device_.synchronize();
            CUDA_CHECK(cudaMemcpy(&host_targets[0], mbase + mtp_tok_, sizeof(std::int32_t),
                                  cudaMemcpyDeviceToHost));
        }
        // Row-23 tie-class probe (diagnosis only, NINFER_LOG_TAIL_GAP=1):
        // top-2 of the stashed anchor logits (the bonus distribution).
        // Synchronous 0.5 MB D2H; never on the hot path.
        if (stash_live && std::getenv("NINFER_LOG_TAIL_GAP") != nullptr) {
            const std::size_t n_vocab = static_cast<std::size_t>(text_vocab_);
            std::vector<std::uint16_t> tail_host(n_vocab);
            CUDA_CHECK(cudaMemcpy(tail_host.data(), anchor_log.data, n_vocab * 2U,
                                  cudaMemcpyDeviceToHost));
            auto bf16_to_float = [](std::uint16_t b) {
                const std::uint32_t f = static_cast<std::uint32_t>(b) << 16;
                float out;
                std::memcpy(&out, &f, sizeof(out));
                return out;
            };
            std::int32_t top1_id = -1, top2_id = -1;
            float top1_v = -1e30f, top2_v = -1e30f;
            for (std::size_t i = 0; i < n_vocab; ++i) {
                const float v = bf16_to_float(tail_host[i]);
                if (v > top1_v) {
                    top2_v = top1_v; top2_id = top1_id;
                    top1_v = v; top1_id = static_cast<std::int32_t>(i);
                } else if (v > top2_v) {
                    top2_v = v; top2_id = static_cast<std::int32_t>(i);
                }
            }
            const float gap = top1_v - top2_v;
            // bf16 ULP of |top1|: 2^(exp-7).
            int exp1 = 0;
            std::frexp(std::fabs(top1_v) > 0.0f ? top1_v : 1.0f, &exp1);
            const float ulp1 = std::ldexp(1.0f, exp1 - 8);
            std::fprintf(stderr,
                         "[tail-gap] seq=%llu lane=%d F=%u top1=%d top2=%d gap=%.6f ulp=%.6f ulps=%.1f %s\n",
                         (unsigned long long)slot.seq_id, lane, F, top1_id, top2_id, gap,
                         ulp1, ulp1 > 0.0f ? gap / ulp1 : -1.0f,
                         (ulp1 > 0.0f && gap <= 2.0f * ulp1) ? "TIE-CLASS" : "NOT-TIE");
            std::fflush(stderr);
        }
        // Draft-3 from the in-checkpoint MTP head. One host round-trip per
        // draft: the next AR input is the previous draft id. Draft j runs at
        // F+j (draft 0 consumes the bonus token at F over the anchor hidden,
        // exactly like the reference bridge mtp_forward_batch at position)
        // and predicts F+1+j. Envelope is per-step {F+j+1,F+j+1} like the
        // reference (bridge_envelope / ar envelope): the serve envelope_
        // spans the whole context window and would attend unwritten MTP KV
        // columns, producing degenerate drafts.
        std::int32_t host_drafts[3] = {0, 0, 0};
        const bool mtp_dbg = std::getenv("NINFER_MTP_DEBUG") != nullptr;
        if (mtp_dbg) {
            std::fprintf(stderr,
                         "[mtp-dbg] layout H=%u V=%d log1=%zu mha=%zu mhb=%zu hid1=%zu tok=%zu "
                         "mh0=%p mh1=%p log1p=%p tokp=%p\n",
                         hidden_, (int)text_vocab_, mtp_log1_, mtp_mha_, mtp_mhb_, mtp_hid1_,
                         mtp_tok_, mh[0].data, mh[1].data, log1.data, tok_out.data);
        }
        {
            std::int32_t tok       = host_targets[0];
            const Tensor* prev_hid = &anchor_hid;
            for (std::uint32_t j = 0; j < kDrafts; ++j) {
                const std::int32_t p = static_cast<std::int32_t>(F + j);
                const ops::CausalAttentionExecutionEnvelope ar_env{
                    static_cast<std::uint32_t>(p + 1), static_cast<std::uint32_t>(p + 1)};
                CUDA_CHECK(cudaMemcpyAsync(mbase + mtp_ids_, &tok, sizeof(tok),
                                           cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync(mbase + mtp_pos_, &p, sizeof(p),
                                           cudaMemcpyHostToDevice, stream));
                card_->mtp_forward_ar_step(tok_in, *prev_hid, pos_t, ar_env, mh[j % 2], log1,
                                           tok_out);
                device_.synchronize();
                CUDA_CHECK(cudaMemcpy(&host_drafts[j], mbase + mtp_tok_, sizeof(std::int32_t),
                                      cudaMemcpyDeviceToHost));
                if (mtp_dbg) {
                    std::uint16_t prev0 = 0, out0 = 0;
                    CUDA_CHECK(cudaMemcpy(&prev0, prev_hid->data, sizeof(prev0),
                                          cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(&out0, mh[j % 2].data, sizeof(out0),
                                          cudaMemcpyDeviceToHost));
                    std::fprintf(stderr,
                                 "[mtp-dbg] seq=%llu F=%u j=%u anchor=%d tok_in=%d pos=%d "
                                 "prev_hid0=0x%04x mtp_hid0=0x%04x draft=%d\n",
                                 (unsigned long long)seq_id, F, j, (int)anchor, tok, p,
                                 prev0, out0, host_drafts[j]);
                }
                tok      = host_drafts[j];
                prev_hid = &mh[j % 2];
            }
        }
        // GDN snapshot, then one width-4 target verify over the drafts. The
        // verify is spare-sourced into the column block, so the lane is
        // untouched until the commit below. Verify inputs are the BONUS draft chain, not the
        // anchor: [b@F, d0@F+1, d1@F+2, d2@F+3]. Each column's argmax is the
        // target pick for the slot AFTER its input, so out[j] is the target
        // pick for the same slot drafts[j] predicts (d_j vs out[j]). The
        // anchor row is NOT re-run: the lane already contains it (a re-run
        // would double-apply GDN) and the stash holds its logits. The window
        // chains across width inside the snapshot kernels, so every column
        // applies exactly once. Text KV writes are positional
        // overwrite-identical to the four serial rows this replaces.
        pool_->copy_slot(lane, spare, stream);
        // The bonus (slot-F token) is consumed as verify column 0's input;
        // save it before the verify overwrites host_targets[0] with out[0].
        const std::int32_t bonus = host_targets[0];
        {
            const std::int32_t in4[4] = {bonus, host_drafts[0], host_drafts[1],
                                         host_drafts[2]};
            const std::int32_t f4 = static_cast<std::int32_t>(F);
            const std::int32_t pos4[4] = {f4, f4 + 1, f4 + 2, f4 + 3};
            const std::int32_t four    = 4;
            CUDA_CHECK(cudaMemcpyAsync(mbase + mtp_vids_, in4, sizeof(in4),
                                       cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(mbase + mtp_vpos_, pos4, sizeof(pos4),
                                       cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(mbase + mtp_vrow_, &lane, sizeof(lane),
                                       cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(mbase + mtp_vsrc_, &spare, sizeof(spare),
                                       cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(mbase + mtp_vval_, &four, sizeof(four),
                                       cudaMemcpyHostToDevice, stream));
            const Tensor vids(mbase + mtp_vids_, DType::I32, {4, 1});
            const Tensor vpos(mbase + mtp_vpos_, DType::I32, {4, 1});
            const Tensor vrow(mbase + mtp_vrow_, DType::I32, {1});
            const Tensor vsrc(mbase + mtp_vsrc_, DType::I32, {1});
            // Step 12: the destination slot is inert under the table
            // overload (conv publishes col c -> block+c; the rec chain
            // walks spare + t[c]), so the spare doubles as dst below.
            // The lane is never written during verify.
            const Tensor vval(mbase + mtp_vval_, DType::I32, {1});
            Tensor vhid(mbase + mtp_vhid_, DType::BF16, {H, 4, 1});
            Tensor vlog(mbase + mtp_vlog_, DType::BF16, {V, 4, 1});
            Tensor vtok(mbase + mtp_vtok_, DType::I32, {4, 1});
            // Step 12 (slots-only): the column table overload is the only
            // verify; commit-by-index lands with it below. The legacy
            // null-table (ping/pong) path is deleted. Table rows stay
            // per-lane device 16B under the diet; only their entries alias
            // the shared pool (safe: single-row step).
            // Envelope mirrors the reference MTP target verify ({1, F+4}):
            // per-column masking is positional, the bound only caps the
            // kernel launch. Argmax runs inside the verify (one sync). 
            const ops::CausalAttentionExecutionEnvelope venv{
                1U, static_cast<std::uint32_t>(f4 + 4)};
            card_->set_gdn_state_action(
                models::qwen3_5::execution::GdnStateAction::UpdateInPlace, nullptr);
            // Phase C: frozen ver/M=4. Pre-work above (copy_slot, H2D uploads
            // to fixed mbase offsets) always runs eager outside capture; only
            // the layers call below is captured. Post-work below (sync + D2H
            // of vtok) stays outside. First verify runs eager (seen rule),
            // capture on the 2nd+, replay after. Default graph-off: active
            // only under NINFER_SERVE_GRAPH with MTP on.
            const bool vgraph_on = graphs_enabled_ && mtp_enabled_ && !graph_dry_;
            auto vglog = [&](const char* act) {
                if (graph_verbose_) {
                    std::fprintf(stderr, "[vgraph] lane=%d F=%u action=%s\n", lane, F, act);
                }
            };
            // Slots path: captures the table overload below
            // (verify_slots_exec_, lane-keyed); the static per-lane column
            // table is bound here.
            const Tensor vcoltab(static_cast<char*>(mtp_coltab_store_.p) +
                                     static_cast<std::size_t>(lane) * 16U,
                                 DType::I32, {4, 1});
            cudaStreamCaptureStatus cap_status = cudaStreamCaptureStatusNone;
            const bool capturing =
                (cudaStreamIsCapturing(stream, &cap_status) == cudaSuccess) &&
                (cap_status != cudaStreamCaptureStatusNone);
            // Step 12 (slots-only): the legacy verify replay/capture arms
            // are deleted; the column-table overload below is the only
            // verify.
            if (vgraph_on && !capturing && !verify_slots_dead_ &&
                verify_slots_exec_ != nullptr && lane == verify_slots_lane_) {
                // Slots-verify replay: the frozen ver/M=4 graph.
                // Per-step device-data (vids/vpos/vrow/vsrc/vval
                // contents at fixed mbase offsets, pool slot contents via
                // the lane->spare copy above) was uploaded eagerly before
                // this point; only this launch replays.
                CUDA_CHECK(cudaGraphLaunch(verify_slots_exec_, stream));
                vglog("replay-slots");
            } else if (vgraph_on && !capturing && !verify_slots_dead_) {
                // Slots-verify capture: the table overload only
                // (UpdateInPlace + dst + vcoltab). Pre-work (copy_slot, H2D
                // uploads) stays eager outside capture; post-work (sync +
                // D2H of vtok) stays outside below. Seen rule: first slots
                // verify runs eager, capture on the 2nd+, immediate replay
                // (WSL record-only rule). Lane-keyed: the vcoltab slice
                // address (lane*16) bakes in, so a lane change destroys and
                // recaptures. Draft shapes (bonus argmax, 3x
                // mtp_forward_ar_step with per-step ar_env, all host
                // syncs/D2H) never enter capture: MTP decode keeps
                // --no-cuda-graph semantics (program_impl.cpp:49,80-84).
                if (!verify_slots_seen_) {
                    card_->target_verify_batch(vids, vpos, vpos, vval, vrow, vsrc, vsrc,
                                               venv, vhid, vlog, vtok, &vcoltab);
                    verify_slots_seen_ = true;
                    vglog("eager-slots-first");
                } else {
                    if (verify_slots_exec_ != nullptr && lane != verify_slots_lane_) {
                        cudaGraphExecDestroy(verify_slots_exec_);
                        verify_slots_exec_ = nullptr;
                    }
                    cudaGraph_t sgraph = nullptr;
                    bool sok           = false;
                    if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal) ==
                        cudaSuccess) {
                        try {
                            card_->target_verify_batch(vids, vpos, vpos, vval, vrow, vsrc,
                                                       vsrc, venv, vhid, vlog, vtok, &vcoltab);
                        } catch (...) {
                            cudaStreamEndCapture(stream, &sgraph);
                            if (sgraph != nullptr) cudaGraphDestroy(sgraph);
                            verify_slots_dead_ = true;
                            throw;
                        }
                        if (cudaStreamEndCapture(stream, &sgraph) == cudaSuccess &&
                            sgraph != nullptr) {
                            cudaGraphExec_t sexec = nullptr;
                            if (cudaGraphInstantiate(&sexec, sgraph, nullptr, nullptr, 0) ==
                                    cudaSuccess &&
                                sexec != nullptr) {
                                cudaGraphDestroy(sgraph);
                                verify_slots_exec_ = sexec;
                                verify_slots_lane_ = lane;
                                // WSL record-only: the capture pass never
                                // executed, so this step's outputs come from
                                // an immediate replay (same rule as ver/M=4).
                                CUDA_CHECK(cudaGraphLaunch(verify_slots_exec_, stream));
                                vglog("capture-slots");
                                sok = true;
                            } else {
                                if (sexec != nullptr) cudaGraphExecDestroy(sexec);
                                cudaGraphDestroy(sgraph);
                            }
                        } else if (sgraph != nullptr) {
                            cudaGraphDestroy(sgraph);
                        }
                    }
                    if (!sok) {
                        // Capture failed without throwing: park dead and run
                        // this step eager (capture didn't execute).
                        verify_slots_dead_ = true;
                        card_->target_verify_batch(vids, vpos, vpos, vval, vrow, vsrc, vsrc,
                                                   venv, vhid, vlog, vtok, &vcoltab);
                        vglog("capture-slots-fail");
                    }
                }
            } else {
                card_->target_verify_batch(vids, vpos, vpos, vval, vrow, vsrc, vsrc, venv,
                                           vhid, vlog, vtok, &vcoltab);
                vglog(!vgraph_on ? "eager-disabled" : "eager-slots");
            }
            // One sync for the whole window: hidden/logits/argmax for all
            // four columns enqueue before the host reads anything.
            device_.synchronize();
            CUDA_CHECK(cudaMemcpy(host_targets, mbase + mtp_vtok_, sizeof(host_targets),
                                  cudaMemcpyDeviceToHost));
        }
        // Longest matching prefix: drafts[j] predicts slot F+1+j and
        // host_targets[j] (fresh verify argmax) is the target pick for that
        // same slot, so drafts[j] == host_targets[j] extends the run. On a
        // full run (a==3) host_targets[3] is the extra bonus sampled from
        // the d2 row (slot F+4); otherwise host_targets[a] is the corrected
        // token for slot F+1+a, already computed by the verify row.
        std::uint32_t accepted = 0;
        while (accepted < kDrafts && host_drafts[accepted] == host_targets[accepted]) {
            ++accepted;
        }
        // Fuzz by capping only (NINFER_MTP_MAX_ACCEPT, default 3): forced a =
        // min(natural, target). Discarding accepted drafts is still correct
        // output, so token-identity vs spec-off stays valid. Never raises.
        if (const char* cap = std::getenv("NINFER_MTP_MAX_ACCEPT")) {
            const long c = std::strtol(cap, nullptr, 10);
            if (c >= 0 && accepted > static_cast<std::uint32_t>(c)) {
                accepted = static_cast<std::uint32_t>(c);
            }
        }
        // Commit list: bonus + accepted drafts + (extra | correction).
        // Full run (a==3): out[3] is the extra bonus for slot F+4, commit
        // is [b, d0, d1, d2, out3] (5 tokens). Partial (a<3): out[a] is the
        // corrected token for slot F+1+a, commit is [b, d0..d_{a-1}, out_a]
        // (a+2 tokens).
        std::int32_t commit[5];
        commit[0] = bonus;
        for (std::uint32_t j = 0; j < accepted; ++j) {
            commit[1 + j] = host_drafts[j];
        }
        commit[1 + accepted]       = host_targets[accepted];
        const std::uint32_t commit_len = (accepted == kDrafts) ? 5 : accepted + 2;
        // Near-tie test (committee): top-5 logits at every verify column
        // (vcol 0..3 = positions 0-3 along the window; the deciding column
        // is vcol a, where host_targets[a] is the correction/extra and
        // host_drafts[c] vs host_targets[c] names a divergence at c).
        // Small top1-top2 gap at a divergence = allowed class; large gap =
        // state bug. NINFER_MTP_LOGGAP=1, debug only (print-only: reads
        // vlog, never writes device state; ~600KB D2H per column).
        if (std::getenv("NINFER_MTP_LOGGAP") != nullptr) {
            const std::size_t Vb =
                static_cast<std::size_t>(text_vocab_);
            std::vector<std::uint16_t> col(Vb);
            for (std::uint32_t c = 0; c < 4; ++c) {
                CUDA_CHECK(cudaMemcpy(col.data(), mbase + mtp_vlog_ +
                                                          static_cast<std::size_t>(c) * Vb * 2U,
                                      Vb * 2U, cudaMemcpyDeviceToHost));
                float tops[5]     = {-1e30f, -1e30f, -1e30f, -1e30f, -1e30f};
                std::int32_t toks[5] = {-1, -1, -1, -1, -1};
                for (std::size_t i = 0; i < Vb; ++i) {
                    std::uint32_t f = static_cast<std::uint32_t>(col[i]) << 16;
                    float v         = 0.0f;
                    std::memcpy(&v, &f, 4);
                    if (v != v) { continue; }
                    for (int t = 0; t < 5; ++t) {
                        if (v > tops[t]) {
                            for (int s = 4; s > t; --s) {
                                tops[s] = tops[s - 1];
                                toks[s] = toks[s - 1];
                            }
                            tops[t] = v;
                            toks[t] = static_cast<std::int32_t>(i);
                            break;
                        }
                    }
                }
                // bf16 ULP of |top1|: 2^(exp-7). DIVERGED names a spec-off
                // divergence (draft[c] != target[c]); c==a is the deciding col.
                const float gap5 = tops[0] - tops[1];
                int exp5         = 0;
                std::frexp(std::fabs(tops[0]) > 0.0f ? tops[0] : 1.0f, &exp5);
                const float ulp5 = std::ldexp(1.0f, exp5 - 8);
                const bool diverged5 = (host_drafts[c] != host_targets[c]);
                std::fprintf(stderr,
                             "[mtp-gap] F=%u c=%u%s t1=%d:%.6f t2=%d:%.6f t3=%d:%.6f "
                             "t4=%d:%.6f t5=%d:%.6f gap=%.6f ulp=%.6f ulps=%.1f %s%s "
                             "draft=%d target=%d\n",
                             F, c, (c == accepted ? "[a]" : ""), toks[0], (double)tops[0],
                             toks[1], (double)tops[1], toks[2], (double)tops[2],
                             toks[3], (double)tops[3], toks[4], (double)tops[4],
                             (double)gap5, (double)ulp5,
                             ulp5 > 0.0f ? (double)(gap5 / ulp5) : -1.0,
                             (ulp5 > 0.0f && gap5 <= 2.0f * ulp5) ? "TIE-CLASS" : "NOT-TIE",
                             diverged5 ? " DIVERGED" : " agree", (int)host_drafts[c],
                             (int)host_targets[c]);
            }
            std::fflush(stderr);
        }
        // Commit: replay the accepted rows lane-into-lane (GDN exact, KV
        // overwrite-identical) and refill the MTP KV rows they own. The MTP
        // refill follows bridge semantics (previous hidden + current token):
        // row j consumes the hidden of F+j-1 (anchor stash for j=0, the
        // previous row's hidden after), staged through mh[1] because hid1
        // holds the current row. Token/position/envelope are per-row: the
        // draft loop leaves tok_in/pos_t holding the LAST draft, and
        // re-running that would poison the prefix the next step drafts from.
        //
        // Step 12 (slots-only, layout A): the target rows are NOT re-run.
        // Verify wrote every accepted position into the consecutive block
        // (causal: hidden/logits/state for col j are final), so commit =
        // (a) copy-back slot[a]->lane at EVERY accept incl a==0 (the lane
        // holds the pre-step window and is never written during verify) +
        // conv runs every step, (b) stage last-position hidden/logits from
        // verify columns, (c) MTP refills only. The legacy commit_len
        // replay rows are deleted.
        CUDA_CHECK(cudaMemcpyAsync(mh[1].data, anchor_hid.data,
                                   static_cast<std::size_t>(hidden_) * 2U,
                                   cudaMemcpyDeviceToDevice, stream));
        {
            // (a) states: accept-a = t[a] in the consecutive column block.
            // The last commit token (out_a @ F+1+a) was only argmaxed, never
            // executed, so the lane state is one position behind next_pos:
            // (c) runs it as a single trailing M=1 row. One row instead of
            // commit_len.
            std::int32_t coltab_host[4];
            for (std::int32_t c = 0; c < 4; ++c) {
                coltab_host[c] = mtp_column_slot(lane, c);
            }
            card_->commit_verify_slots(lane, coltab_host, accepted, 4);
            // (b) MTP refills for all but the last commit row: hidden of
            // commit[j] is verify column min(j,a).
            const char* vhid_base = mbase + mtp_vhid_;
            for (std::uint32_t j = 0; j + 1 < commit_len; ++j) {
                std::int32_t ctok = commit[j];
                std::int32_t cpos = static_cast<std::int32_t>(F + j);
                const ops::CausalAttentionExecutionEnvelope cenv{
                    static_cast<std::uint32_t>(cpos + 1), static_cast<std::uint32_t>(cpos + 1)};
                CUDA_CHECK(cudaMemcpyAsync(mbase + mtp_ids_, &ctok, sizeof(ctok),
                                           cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync(mbase + mtp_pos_, &cpos, sizeof(cpos),
                                           cudaMemcpyHostToDevice, stream));
                card_->mtp_forward_batch(tok_in, mh[1], pos_t, cenv, mh[0], -1, nullptr, nullptr);
                const std::uint32_t hj = (j < accepted) ? j : accepted;
                CUDA_CHECK(cudaMemcpyAsync(mh[1].data,
                                           vhid_base + static_cast<std::size_t>(hj) * hidden_ * 2U,
                                           static_cast<std::size_t>(hidden_) * 2U,
                                           cudaMemcpyDeviceToDevice, stream));
            }
            // (c) trailing row: executes out_a, advancing lane state to
            // F+commit_len-1. hid1/log1 come from the row (M=1 authoritative)
            // for the stash + final refill below.
            run_single_row(commit[commit_len - 1], F + commit_len - 1, lane, lane, lane, hid1,
                           log1, true);
            {
                std::int32_t ctok = commit[commit_len - 1];
                std::int32_t cpos = static_cast<std::int32_t>(F + commit_len - 1);
                const ops::CausalAttentionExecutionEnvelope cenv{
                    static_cast<std::uint32_t>(cpos + 1), static_cast<std::uint32_t>(cpos + 1)};
                CUDA_CHECK(cudaMemcpyAsync(mbase + mtp_ids_, &ctok, sizeof(ctok),
                                           cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync(mbase + mtp_pos_, &cpos, sizeof(cpos),
                                           cudaMemcpyHostToDevice, stream));
                card_->mtp_forward_batch(tok_in, mh[1], pos_t, cenv, mh[0], -1, nullptr, nullptr);
                CUDA_CHECK(cudaMemcpyAsync(mh[1].data, hid1.data,
                                           static_cast<std::size_t>(hidden_) * 2U,
                                           cudaMemcpyDeviceToDevice, stream));
            }
        }
        CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(anchor_store_.p) +
                                           static_cast<std::size_t>(lane) * hidden_ * 2U,
                                   hid1.data, static_cast<std::size_t>(hidden_) * 2U,
                                   cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(anchor_logits_.p) +
                                           static_cast<std::size_t>(lane) * text_vocab_ * 2U,
                                   log1.data, static_cast<std::size_t>(text_vocab_) * 2U,
                                   cudaMemcpyDeviceToDevice, stream));
        slot.anchor_token      = static_cast<TokenId>(commit[commit_len - 1]);
        slot.anchor_valid      = true;
        slot.next_pos          = F + commit_len;
        slot.mtp_valid_pos     = slot.next_pos;
        // S1 workload check: per-step accepts + tokens alongside ms/tok.
        // Gated on NINFER_MTP_DEBUG (stderr only, no GPU effect).
        if (mtp_dbg) {
            std::fprintf(stderr, "[mtp-step] F=%u a=%u n=%u\n", F, accepted, commit_len);
        }
        // Rejected positions must never become readable: both frontiers
        // advance to the same accepted frontier, so nothing reads stale
        // draft rows beyond it (fail-closed, both paths).
        if (slot.mtp_valid_pos != slot.next_pos) {
            throw std::logic_error("MTP commit left readable length past accept");
        }
        slot.touched_as_prefill = false;
        runtime::StepDecodedPairs decoded;
        for (std::uint32_t j = 0; j < commit_len; ++j) {
            decoded.emplace_back(seq_id, static_cast<TokenId>(commit[j]));
        }
        if (mtp_dbg) {
            std::fprintf(stderr, "[serve-mtp] seq=%llu lane=%d F=%u stash=%d anchor=%d drafts=[%d %d %d] verify=[%d %d %d %d] accepted=%u commit=%u\n",
                         (unsigned long long)seq_id, lane, F, stash_live ? 1 : 0, (int)anchor,
                         host_drafts[0], host_drafts[1],
                         host_drafts[2], host_targets[0], host_targets[1], host_targets[2],
                         host_targets[3], accepted, commit_len);
        }
        // Same lane release as the ordinary path: lanes whose seq left the
        // batch after decoding are done; mid-prefill vanishers are starved.
        const std::size_t num_seqs = batch.num_seqs();
        for (ServeSlot& rel : slots_) {
            if (!rel.in_use) { continue; }
            bool seen = false;
            for (std::size_t q = 0; q < num_seqs; ++q) {
                if (batch.seq_ids[q] == rel.seq_id) {
                    seen = true;
                    break;
                }
            }
            if (!seen && !rel.touched_as_prefill) { rel.in_use = false; }
        }
        return decoded;
    }

    // Mirror prefill slices through the MTP layer one row at a time so the
    // MTP KV prefix stays warm. Rows chain through the slice's fill scratch
    // slot (snapshotted pre-forward in step()): the ordinary forward advanced
    // the lane in place, so re-running from the lane would apply every row
    // twice (once by the forward, once here) and poison the MTP KV prefix.
    // From the scratch every row applies exactly once, reproducing the true
    // sequential GDN trajectory bit-for-bit. Width-1 additionally keeps the
    // GDN Verify snapshot inside its single-column domain on every
    // concurrency (a width-n batch needs n state columns and trips
    // gdn_input_proj_conv_snapshot on small servers). Text KV writes are
    // positional overwrite-identical; the lane GDN is never touched. The
    // slice's last row is the lane's draft anchor: its hidden and logits are
    // stashed so the first MTP step needs no replay.
    void mtp_prefill_fill(const batch::StepPlan& plan, const batch::RaggedBatch& batch,
                          const std::vector<std::int32_t>& row_slot,
                          const std::vector<std::int32_t>& pos, const Tensor& prefill_hidden) {
        cudaStream_t stream = device_.stream;
        char* mbase         = static_cast<char*>(scratch_.p);
        const std::int32_t H     = static_cast<std::int32_t>(hidden_);
        const std::int32_t V     = static_cast<std::int32_t>(text_vocab_);
        // Row-23 rung-1 (DEFAULT ON since consolidation;
        // NINFER_MTP_WARM_HIDDENS=0 opts out): the ordinary prefill forward
        // above already computed every row's target hidden into prefill_hidden
        // ({H,T}, still live: nothing overwrote it since). Reuse the column
        // per row and skip the mirrored ordinary row (and its per-row
        // logits) entirely. GDN scratch chaining is untouched; text KV keeps
        // the prefill writes (overwrite-identical). Only the slice tail pays
        // one lm_head projection for its anchor logits (bonus token).
        // Spec-off-matching split: the tail column reused here is the
        // ordinary width-1 Verify row computed by forward_serve_step
        // (NINFER_MTP_SPLIT_TAIL=0 restores the scan-tail column).
        const char* warm_hiddens_env = std::getenv("NINFER_MTP_WARM_HIDDENS");
        const bool warm_hiddens =
            warm_hiddens_env == nullptr || std::string(warm_hiddens_env) != "0";
        // Row-23 rung-2 (DEFAULT ON since consolidation;
        // NINFER_MTP_WARM_BATCHED=0 opts out): one batched MTP pass per
        // prefill_chunk-wide span instead of the sequential per-row forwards.
        // Rung-1 (above) is the fallback and the tail-anchor source in both
        // modes.
        const char* warm_batched_env = std::getenv("NINFER_MTP_WARM_BATCHED");
        const bool warm_batched =
            warm_hiddens &&
            (warm_batched_env == nullptr || std::string(warm_batched_env) != "0");
        const char* prefill_base =
            warm_hiddens ? static_cast<const char*>(prefill_hidden.data) : nullptr;
        const std::uint32_t chunk =
            warm_chunk_ != 0 ? warm_chunk_ : 1U;
        for (std::size_t r = 0; r < plan.prefill.size(); ++r) {
            const std::int32_t lane = row_slot[r];
            // Fill scratch: column region slot r, matching the step()
            // seeding snapshot (slice r -> column base + r).
            const std::int32_t fill_scratch = mtp_column_base_ + static_cast<std::int32_t>(r);
            CUDA_CHECK(cudaMemcpyAsync(io_->backend_kv_table_row.data, &lane, sizeof(lane),
                                       cudaMemcpyHostToDevice, stream));
            const std::uint32_t off   = batch.seq_offsets[r];
            const std::uint32_t count = batch.seq_offsets[r + 1] - off;
            if (warm_batched && count > 0) {
                // Batched warming: chunk the slice span. Hidden columns are
                // already a contiguous {H,T} block; ids/positions stage
                // through the batch regions. Outputs are discard (KV fill is
                // the product). Envelope is the tight per-chunk span, same
                // convention as the per-row loop.
                for (std::uint32_t base = 0; base < count;) {
                    std::uint32_t span = count - base;
                    if (span > chunk) { span = chunk; }
                    const std::int32_t T = static_cast<std::int32_t>(span);
                    CUDA_CHECK(cudaMemcpyAsync(
                        mbase + mtp_batch_ids_, batch.tokens.data() + off + base,
                        static_cast<std::size_t>(span) * sizeof(TokenId),
                        cudaMemcpyHostToDevice, stream));
                    CUDA_CHECK(cudaMemcpyAsync(
                        mbase + mtp_batch_pos_, pos.data() + off + base,
                        static_cast<std::size_t>(span) * 4U, cudaMemcpyHostToDevice,
                        stream));
                    const Tensor ids_b(mbase + mtp_batch_ids_, DType::I32, {T});
                    const Tensor pos_b(mbase + mtp_batch_pos_, DType::I32, {T});
                    const Tensor hid_b(
                        const_cast<char*>(prefill_base) +
                            (static_cast<std::size_t>(off) + base) * hidden_ * 2U,
                        DType::BF16, {H, T});
                    Tensor out_b(mbase + mtp_batch_out_, DType::BF16, {H, T});
                    const std::uint32_t p_first = pos[off + base];
                    const std::uint32_t p_last  = pos[off + base + span - 1];
                    const ops::CausalAttentionExecutionEnvelope batch_env{
                        p_first + 1, p_last + 1};
                    card_->mtp_forward_batch(ids_b, hid_b, pos_b, batch_env, out_b, -1,
                                             nullptr, nullptr);
                    base += span;
                }
            }
            for (std::uint32_t base = 0; base < count; ++base) {
                // Rung-2: the batched block above already filled MTP KV for
                // every row; only the tail anchor still runs per slice.
                if (warm_batched && base + 1 != count) { continue; }
                const std::int32_t one = 1;
                std::int32_t h_ids     = static_cast<std::int32_t>(batch.tokens[off + base]);
                std::int32_t h_pos     = pos[off + base];
                std::int32_t h_row     = lane;
                std::int32_t h_ss      = fill_scratch;
                std::int32_t h_sd      = fill_scratch;
                CUDA_CHECK(cudaMemcpyAsync(mbase + mtp_fill_ids_, &h_ids, sizeof(h_ids),
                                           cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync(mbase + mtp_fill_pos_, &h_pos, sizeof(h_pos),
                                           cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync(mbase + mtp_fill_rows_, &h_row, sizeof(h_row),
                                           cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync(mbase + mtp_fill_ssrc_, &h_ss, sizeof(h_ss),
                                           cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync(mbase + mtp_fill_sdst_, &h_sd, sizeof(h_sd),
                                           cudaMemcpyHostToDevice, stream));
                const Tensor ids_t(mbase + mtp_fill_ids_, DType::I32, {one});
                const Tensor pos_t(mbase + mtp_fill_pos_, DType::I32, {one});
                const Tensor row_t(mbase + mtp_fill_rows_, DType::I32, {one});
                const Tensor ssrc_t(mbase + mtp_fill_ssrc_, DType::I32, {one});
                const Tensor sdst_t(mbase + mtp_fill_sdst_, DType::I32, {one});
                Tensor hid_t(mbase + mtp_fill_hid_, DType::BF16, {H, one});
                Tensor log_t(mbase + mtp_fill_log_, DType::BF16, {V, one});
                Tensor mh_t(mbase + mtp_fill_mh_, DType::BF16, {H, one});
                const ops::CausalAttentionExecutionEnvelope fill_env{
                    static_cast<std::uint32_t>(h_pos + 1), static_cast<std::uint32_t>(h_pos + 1)};
                if (warm_hiddens) {
                    // Rung-1: copy this row's prefill hidden column in (one
                    // HxBF16 D2D); no ordinary row, no per-row logits.
                    CUDA_CHECK(cudaMemcpyAsync(
                        hid_t.data,
                        prefill_base +
                            (static_cast<std::size_t>(off) + base) * hidden_ * 2U,
                        static_cast<std::size_t>(hidden_) * 2U, cudaMemcpyDeviceToDevice,
                        stream));
                } else {
                    card_->ordinary_decode_batch(ids_t, pos_t, pos_t, row_t, ssrc_t, sdst_t,
                                                 fill_env, hid_t, log_t);
                }
                if (!warm_batched) {
                    card_->mtp_forward_batch(ids_t, hid_t, pos_t, fill_env, mh_t, -1, nullptr,
                                             nullptr);
                }
                if (base + 1 == count) {
                    // Slice tail is the anchor: stash its exact hidden and
                    // as-input logits for the first MTP step (no replay).
                    if (warm_hiddens) {
                        // Rung-1: the one projection per request. Warming
                        // fills MTP KV only; per-row logits are skipped.
                        card_->project_target_tail(hid_t, log_t);
                    }
                    CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(anchor_store_.p) +
                                                       static_cast<std::size_t>(lane) *
                                                           hidden_ * 2U,
                                               hid_t.data,
                                               static_cast<std::size_t>(hidden_) * 2U,
                                               cudaMemcpyDeviceToDevice, stream));
                    CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(anchor_logits_.p) +
                                                       static_cast<std::size_t>(lane) *
                                                           text_vocab_ * 2U,
                                               log_t.data,
                                               static_cast<std::size_t>(text_vocab_) * 2U,
                                               cudaMemcpyDeviceToDevice, stream));
                }
            }
            ServeSlot& slot     = slots_[static_cast<std::size_t>(lane)];
            slot.mtp_valid_pos  = slot.next_pos;
            slot.anchor_token   = batch.tokens[batch.seq_offsets[r + 1] - 1];
            slot.anchor_valid   = count > 0;
        }
    }

    // Row-23 conc resume (task 8): count one pure-decode step on the lane.
    // On window entry the gap starts at the current MTP frontier; a lane
    // that enters with a pre-existing (unstaged) hole can never match
    // coverage and fails closed at resume via staged_overflow below.
    void toggle_tick_lane(std::int32_t lane) {
        ServeSlot& slot = slots_[static_cast<std::size_t>(lane)];
        ++slot.mtp_toggle_seen;
        if (slot.staged_n == 0 && !slot.staged_overflow &&
            in_toggle_window(slot.mtp_toggle_seen)) {
            slot.stage_start = slot.mtp_valid_pos;
        }
    }

    // Stage one in-window lane column per decode row: the ordinary forward's
    // target hidden (device D2D into the lane's catchup_store_ block) plus
    // the row's (token, pos) on host. Positions must extend the staged span
    // contiguously; anything else (overflow, non-unit rows, out-of-window)
    // marks the lane so resume fails closed to the ordinary path.
    void mtp_stage_decode_hidden(const batch::RaggedBatch& batch,
                                 const std::vector<std::int32_t>& row_slot, std::size_t n_pref,
                                 std::size_t n_dec, const std::vector<std::int32_t>& pos,
                                 const Tensor& hidden) {
        cudaStream_t stream = device_.stream;
        char* cbase         = static_cast<char*>(catchup_store_.p);
        for (std::size_t i = 0; i < n_dec; ++i) {
            const std::size_t s    = n_pref + i;
            const std::int32_t lane = row_slot[s];
            ServeSlot& slot         = slots_[static_cast<std::size_t>(lane)];
            if (!in_toggle_window(slot.mtp_toggle_seen) || slot.staged_overflow) { continue; }
            const std::uint32_t flat  = batch.seq_offsets[s];
            const std::uint32_t width = batch.seq_offsets[s + 1] - flat;
            const std::int32_t p      = pos[flat];
            if (width != 1 || slot.staged_n >= toggle_off_ ||
                p != static_cast<std::int32_t>(slot.stage_start + slot.staged_n)) {
                slot.staged_overflow = true;
                continue;
            }
            CUDA_CHECK(cudaMemcpyAsync(
                cbase +
                    (static_cast<std::size_t>(lane) * toggle_off_ + slot.staged_n) * hidden_ *
                        2U,
                static_cast<const char*>(hidden.data) +
                    static_cast<std::size_t>(flat) * hidden_ * 2U,
                static_cast<std::size_t>(hidden_) * 2U, cudaMemcpyDeviceToDevice, stream));
            slot.staged_ids.push_back(static_cast<std::int32_t>(batch.tokens[flat]));
            slot.staged_pos.push_back(p);
            ++slot.staged_n;
        }
    }

    // Batched catch-up over the skipped span [mtp_valid_pos, frontier),
    // reusing the rung-2 chunked-mtp_forward_batch shape (tight per-chunk
    // envelope, discard output; KV fill is the product), then the resume
    // anchor row (ordinary M=1 lane-into-lane, eager) + its MTP refill +
    // anchor stash, mirroring the commit tail. On success the lane meets
    // the MTP dispatch preconditions exactly (mtp_valid_pos == frontier,
    // live stash on the resume input). Returns false without touching
    // device state when coverage is inexact; the caller then runs the
    // ordinary path (today's behavior).
    bool mtp_resume_catchup(std::int32_t lane, TokenId anchor_tok, std::uint32_t frontier) {
        ServeSlot& slot = slots_[static_cast<std::size_t>(lane)];
        if (slot.staged_overflow || slot.staged_n == 0 ||
            slot.stage_start != slot.mtp_valid_pos ||
            slot.stage_start + slot.staged_n != frontier) {
            return false;
        }
        cudaStream_t stream = device_.stream;
        char* mbase         = static_cast<char*>(scratch_.p);
        const char* cbase   = static_cast<const char*>(catchup_store_.p);
        const std::int32_t H     = static_cast<std::int32_t>(hidden_);
        const std::int32_t V     = static_cast<std::int32_t>(text_vocab_);
        const std::uint32_t chunk = warm_chunk_ != 0 ? warm_chunk_ : 1U;
        CUDA_CHECK(cudaMemcpyAsync(io_->backend_kv_table_row.data, &lane, sizeof(lane),
                                   cudaMemcpyHostToDevice, stream));
        for (std::uint32_t base = 0; base < slot.staged_n;) {
            std::uint32_t span = slot.staged_n - base;
            if (span > chunk) { span = chunk; }
            const std::int32_t T = static_cast<std::int32_t>(span);
            CUDA_CHECK(cudaMemcpyAsync(
                mbase + mtp_batch_ids_, slot.staged_ids.data() + base,
                static_cast<std::size_t>(span) * sizeof(std::int32_t),
                cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(
                mbase + mtp_batch_pos_, slot.staged_pos.data() + base,
                static_cast<std::size_t>(span) * sizeof(std::int32_t),
                cudaMemcpyHostToDevice, stream));
            const Tensor ids_b(mbase + mtp_batch_ids_, DType::I32, {T});
            const Tensor pos_b(mbase + mtp_batch_pos_, DType::I32, {T});
            const Tensor hid_b(
                const_cast<char*>(cbase) +
                    (static_cast<std::size_t>(lane) * toggle_off_ + base) * hidden_ * 2U,
                DType::BF16, {H, T});
            Tensor out_b(mbase + mtp_batch_out_, DType::BF16, {H, T});
            const std::uint32_t p_first =
                static_cast<std::uint32_t>(slot.staged_pos[base]);
            const std::uint32_t p_last =
                static_cast<std::uint32_t>(slot.staged_pos[base + span - 1]);
            const ops::CausalAttentionExecutionEnvelope batch_env{p_first + 1, p_last + 1};
            card_->mtp_forward_batch(ids_b, hid_b, pos_b, batch_env, out_b, -1, nullptr,
                                     nullptr);
            base += span;
        }
        // Anchor row: the resume input was never executed (the ordinary path
        // below never ran for this step), so execute it lane-into-lane here.
        // The lane holds the prefix through frontier - 1, exactly the commit
        // trailing-row precondition.
        Tensor hid1(mbase + mtp_hid1_, DType::BF16, {H, 1});
        Tensor log1(mbase + mtp_log1_, DType::BF16, {V, 1});
        run_single_row(static_cast<std::int32_t>(anchor_tok), frontier, lane, lane, lane,
                       hid1, log1);
        // Anchor MTP refill: consumes the last skipped hidden (bridge
        // semantics: previous hidden + current token), same as the commit's
        // final refill.
        {
            const Tensor tok_in(mbase + mtp_ids_, DType::I32, {1});
            const Tensor pos_t(mbase + mtp_pos_, DType::I32, {1});
            Tensor mh0(mbase + mtp_mha_, DType::BF16, {H, 1});
            Tensor mh1(mbase + mtp_mhb_, DType::BF16, {H, 1});
            CUDA_CHECK(cudaMemcpyAsync(
                mh1.data,
                cbase +
                    (static_cast<std::size_t>(lane) * toggle_off_ + slot.staged_n - 1) *
                        hidden_ * 2U,
                static_cast<std::size_t>(hidden_) * 2U, cudaMemcpyDeviceToDevice, stream));
            const std::int32_t atok = static_cast<std::int32_t>(anchor_tok);
            const std::int32_t apos = static_cast<std::int32_t>(frontier);
            CUDA_CHECK(cudaMemcpyAsync(mbase + mtp_ids_, &atok, sizeof(atok),
                                       cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(mbase + mtp_pos_, &apos, sizeof(apos),
                                       cudaMemcpyHostToDevice, stream));
            const ops::CausalAttentionExecutionEnvelope cenv{frontier + 1, frontier + 1};
            card_->mtp_forward_batch(tok_in, mh1, pos_t, cenv, mh0, -1, nullptr, nullptr);
        }
        CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(anchor_store_.p) +
                                           static_cast<std::size_t>(lane) * hidden_ * 2U,
                                   hid1.data, static_cast<std::size_t>(hidden_) * 2U,
                                   cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(anchor_logits_.p) +
                                           static_cast<std::size_t>(lane) * text_vocab_ * 2U,
                                   log1.data,
                                   static_cast<std::size_t>(text_vocab_) * 2U,
                                   cudaMemcpyDeviceToDevice, stream));
        slot.anchor_token  = anchor_tok;
        slot.anchor_valid  = true;
        slot.mtp_valid_pos = frontier;
        if (std::getenv("NINFER_MTP_DEBUG") != nullptr) {
            std::fprintf(stderr, "[mtp-resume] lane=%d F=%u gap=%u\n", lane, frontier,
                         slot.staged_n);
            std::fflush(stderr);
        }
        slot.staged_n = 0;
        slot.staged_ids.clear();
        slot.staged_pos.clear();
        slot.stage_start = frontier;
        return true;
    }

private:
    struct ServeSlot {
        bool in_use             = false;
        std::uint64_t seq_id    = 0;
        std::uint32_t next_pos  = 0;
        bool touched_as_prefill = false;
        // S7 MTP ledger: positions [0, mtp_valid_pos) are warm in MTP KV;
        // anchor caches the committed target hidden for the draft chain.
        std::uint32_t mtp_valid_pos = 0;
        TokenId anchor_token        = 0;
        bool anchor_valid           = false;
        // Row-23 conc resume (task 8, default off): mid-request spec toggle
        // bookkeeping. mtp_toggle_seen counts pure-decode steps on this lane
        // while armed; stage_* records the forced-off window's (token, pos)
        // plus staged target-hidden columns in catchup_store_ for the batched
        // resume fill. All zero/unused unless NINFER_MTP_TOGGLE_OFF is set.
        std::uint32_t mtp_toggle_seen = 0;
        std::uint32_t stage_start     = 0;
        std::uint32_t staged_n        = 0;
        bool staged_overflow          = false;
        std::vector<std::int32_t> staged_ids;
        std::vector<std::int32_t> staged_pos;
    };

    DeviceContext& device_;
    const models::qwen3_5::execution::Parameters& parameters_;
    std::uint32_t max_seqs_       = 0;
    std::uint32_t max_context_    = 0;
    std::uint32_t hidden_         = 0;
    std::uint32_t max_tokens_     = 0;
    std::uint32_t max_blocks_     = 0;
    std::uint32_t pages_per_slot_ = 0;
    DeviceBuffer kv_store_;
    bool mtp_enabled_            = false;
    std::int32_t spare_slot_     = -1;
    std::uint32_t text_vocab_    = 0;
    std::int32_t public_tokens_  = 0;
    std::unique_ptr<models::qwen3_5::PagedKVCache> mtp_kv_;
    DeviceKVPageReservation mtp_reservation_;
    std::vector<DeviceKVPageLease> mtp_page_leases_;
    std::vector<KVExecutionRowLease> mtp_row_leases_;
    DeviceBuffer anchor_store_;
    DeviceBuffer anchor_logits_;
    // Step 12: the per-lane shadow slots are gone. The prefill fill chains
    // through fill scratch borrowed from the column region (slice r ->
    // column base + r; the width-4 verify never runs in a prefill step).
    // Layout after the max_seqs_ lane slots: [max_seqs_] is the verify
    // spare, then the column groups.
    // Row 20b layout B: base of the column slots (static tables,
    // init-filled once). -1 when MTP is off. With a shared pool
    // (mtp_column_groups_==1) all lanes map into the same 4 slots via
    // mtp_column_slot; otherwise group g serves lane g.
    std::int32_t mtp_column_base_ = -1;
    // Number of 4-slot column groups: 1 shared pool when max_seqs_<=2,
    // else one group per lane. 0 when MTP is off.
    std::uint32_t mtp_column_groups_ = 0;
    // MTP-extra GDN pool slots: 1 verify spare + 4 column slots per group
    // + 3 oracle presnaps (only when the oracle runs). 0 when MTP is off.
    // The pool holds max_seqs_ lane slots plus these extras.
    std::uint32_t mtp_extra_pool_slots_ = 0;
    // True when NINFER_SLOT_ORACLE was set at startup (oracle presnaps
    // allocated). The debug probe itself was removed with the legacy
    // rows (Step 12); the reservation stays for diet accounting.
    bool mtp_oracle_on_ = false;
    // Column slot for (lane, c): the shared pool base when grouped,
    // the lane's own group otherwise. Call only when MTP is on.
    [[nodiscard]] std::int32_t mtp_column_slot(std::int32_t lane, std::int32_t c) const noexcept {
        const std::int32_t group = (mtp_column_groups_ <= 1) ? 0 : lane;
        return mtp_column_base_ + group * 4 + c;
    }
    // Row 20b: per-lane {4} I32 column-slot tables (device, init-filled).
    DeviceBuffer mtp_coltab_store_;
    // Row 20b oracle presnap slots, all three (reserved, debug only). -1
    // when MTP is off or NINFER_SLOT_ORACLE was unset at startup.
    std::int32_t mtp_oracle_base_ = -1;
    std::size_t mtp_ids_       = 0;
    std::size_t mtp_pos_       = 0;
    std::size_t mtp_row_       = 0;
    std::size_t mtp_ssrc_      = 0;
    std::size_t mtp_sdst_      = 0;
    std::size_t mtp_tok_       = 0;
    std::size_t mtp_vtok_      = 0;
    std::size_t mtp_vids_      = 0;
    std::size_t mtp_vpos_      = 0;
    std::size_t mtp_vrow_      = 0;
    std::size_t mtp_vsrc_      = 0;
    std::size_t mtp_vval_      = 0;
    std::size_t mtp_vhid_      = 0;
    std::size_t mtp_vlog_      = 0;
    std::size_t mtp_fill_ids_  = 0;
    std::size_t mtp_fill_pos_  = 0;
    std::size_t mtp_fill_rows_ = 0;
    std::size_t mtp_fill_ssrc_ = 0;
    std::size_t mtp_fill_sdst_ = 0;
    std::size_t mtp_hid1_      = 0;
    std::size_t mtp_log1_      = 0;
    std::size_t mtp_mha_       = 0;
    std::size_t mtp_mhb_       = 0;
    std::size_t mtp_fill_hid_  = 0;
    std::size_t mtp_fill_log_  = 0;
    std::size_t mtp_fill_mh_   = 0;
    std::size_t mtp_batch_ids_ = 0;
    std::size_t mtp_batch_pos_ = 0;
    std::size_t mtp_batch_out_ = 0;
    std::uint32_t warm_chunk_  = 0;
    // Row-23 conc resume (task 8): toggle window + catch-up staging store.
    std::uint32_t toggle_after_ = 0;
    std::uint32_t toggle_off_   = 0;
    DeviceBuffer catchup_store_;
    DeviceBuffer pool_store_;
    DeviceBuffer round_store_;
    std::unique_ptr<DeviceArena> work_;
    DeviceBuffer scratch_;
    DeviceBuffer sampling_store_;
    const ops::SamplingConfig* serve_sampling_ = nullptr;
    std::vector<std::int32_t> host_tables_;
    std::size_t ids_            = 0;
    std::size_t cpos_           = 0;
    std::size_t rpos_           = 0;
    std::size_t rows_           = 0;
    std::size_t gsrc_           = 0;
    std::size_t gdst_           = 0;
    std::size_t hidden_off_     = 0;
    std::size_t prefill_hidden_ = 0;
    std::size_t offsets_        = 0;
    std::size_t tables_         = 0;
    std::unique_ptr<models::qwen3_5::PagedKVCache> kv_;
    std::unique_ptr<LinearAttentionStatePool> pool_;
    std::unique_ptr<models::qwen3_5::RoundState> io_;
    std::unique_ptr<models::qwen3_5::execution::TextContext> card_;
    DeviceKVPageReservation reservation_;
    std::vector<DeviceKVPageLease> page_leases_;
    std::vector<KVExecutionRowLease> row_leases_;
    ops::CausalAttentionExecutionEnvelope envelope_{0, 0};
    std::vector<ServeSlot> slots_;
    // Graph ticket 2: eager decode-only replay (spec-off, single seq).
    // Uploads, embed and sample stay eager; only the 64-layer loop + final
    // norm replay. Keyed by GDN lane, cap 2 live execs; a capture miss falls
    // back to eager and parks the lane dead. NINFER_SERVE_GRAPH=0 disables.
    struct ServeDecodeGraph {
        bool live = false;
        int lane  = -1;
        cudaGraphExec_t exec = nullptr;
    };
    ServeDecodeGraph decode_graphs_[2];
    std::vector<int> graph_dead_lanes_;
    std::vector<int> graph_seen_lanes_;
    bool graphs_enabled_ = true;
    bool graph_verbose_  = false;
    bool graph_dry_      = false;
    // Step 12 (slots-only): one frozen ver/M=4 exec for the column-table
    // overload. All baked addresses are lane-independent except the
    // vcoltab slice address (lane*16, lane-keyed); lane-varying data
    // (slot indices, ids, positions, valid) lives in fixed-address
    // buffers rewritten per step outside capture. First verify runs eager
    // (settles lazy state), capture on the 2nd+, immediate replay (WSL
    // record-only rule). Fail-closed: capture failure parks it dead and
    // verify stays eager.
    cudaGraphExec_t verify_slots_exec_ = nullptr;
    bool verify_slots_seen_            = false;
    bool verify_slots_dead_            = false;
    int verify_slots_lane_             = -1;
    // Row-18: one frozen dec/M=1 commit exec (GLOBAL). Commit rows share
    // hid1/log1; the anchor replay's anchor_hid never enters (allow_graph).
    cudaGraphExec_t commit_exec_ = nullptr;
    bool commit_seen_            = false;
    bool commit_dead_            = false;

public:
    ~ServeForwardContext() {
        for (auto& slot : decode_graphs_) {
            if (slot.exec != nullptr) {
                cudaGraphExecDestroy(slot.exec);
                slot.exec = nullptr;
                slot.live = false;
            }
        }
        if (verify_slots_exec_ != nullptr) {
            cudaGraphExecDestroy(verify_slots_exec_);
            verify_slots_exec_ = nullptr;
        }
        if (commit_exec_ != nullptr) {
            cudaGraphExecDestroy(commit_exec_);
            commit_exec_ = nullptr;
        }
    }

    bool graph_eligible(std::size_t n_pref, std::size_t n_dec) const {
        // Off-branch graphing (row 20b split 1): ordinary single-token
        // decodes may graph even with MTP enabled. MTP steps never reach
        // here (step() dispatches them to step_mtp_decode first), and the
        // ordinary path never touches MTP rows/slots — so the !mtp gate
        // only served to keep the MTP server's off-branch eager (the 58%).
        return graphs_enabled_ && n_pref == 0 && n_dec == 1;
    }

    ServeDecodeGraph* graph_for_lane(int lane) {
        if (std::find(graph_dead_lanes_.begin(), graph_dead_lanes_.end(), lane) !=
            graph_dead_lanes_.end()) {
            return nullptr;
        }
        for (auto& slot : decode_graphs_) {
            if (slot.live && slot.lane == lane) return &slot;
        }
        return nullptr;
    }

    // Fresh admission invalidates any graph keyed by this lane: the next
    // decode on it runs eager (seen-rule), recapture happens on decode 2+.
    void graph_forget_lane(int lane) {
        for (auto& slot : decode_graphs_) {
            if (slot.live && slot.lane == lane) {
                cudaGraphExecDestroy(slot.exec);
                slot.exec = nullptr;
                slot.live = false;
            }
        }
        graph_seen_lanes_.erase(
            std::remove(graph_seen_lanes_.begin(), graph_seen_lanes_.end(), lane),
            graph_seen_lanes_.end());
        // Fresh admission re-arms a capture-parked lane: dead is a
        // per-admission verdict (capture failed for the old sequence), not a
        // permanent lane property. Without this, a lane parked dead stays
        // eager for process lifetime even after the lane is recycled.
        graph_dead_lanes_.erase(
            std::remove(graph_dead_lanes_.begin(), graph_dead_lanes_.end(), lane),
            graph_dead_lanes_.end());
    }

    void graph_mark_dead(int lane) {
        for (auto& slot : decode_graphs_) {
            if (slot.live && slot.lane == lane) {
                cudaGraphExecDestroy(slot.exec);
                slot.exec = nullptr;
                slot.live = false;
            }
        }
        if (std::find(graph_dead_lanes_.begin(), graph_dead_lanes_.end(), lane) ==
            graph_dead_lanes_.end()) {
            graph_dead_lanes_.push_back(lane);
        }
    }

    void graph_store(int lane, cudaGraphExec_t exec) {
        ServeDecodeGraph* slot = &decode_graphs_[0];
        if (decode_graphs_[0].live && decode_graphs_[1].live) {
            // Both live: evict slot 0 (oldest-first ring would need age bits;
            // single-seq steady state never evicts, so any choice is fine).
            cudaGraphExecDestroy(decode_graphs_[0].exec);
            decode_graphs_[0].exec = nullptr;
            decode_graphs_[0].live = false;
        } else if (decode_graphs_[0].live) {
            slot = &decode_graphs_[1];
        }
        slot->live = true;
        slot->lane = lane;
        slot->exec = exec;
    }

    runtime::StepDecodedPairs step_decode_layers(const batch::StepPlan& plan,
                                                 const batch::RaggedBatch& batch,
                                                 const batch::DeviceRaggedBatch& view,
                                                 models::qwen3_5::execution::TextContext::ServeStepTensors& tensors,
                                                 const std::vector<std::int32_t>& row_slot,
                                                 std::size_t n_pref, std::size_t n_dec) {
        cudaStream_t stream = device_.stream;
        // Capture gate: pure single-token decode only. The first HTTP
        // request's early steps may carry a different M/phase than steady
        // decodes, so a lane's first decode-shaped step always runs eager
        // (warmup) and capture starts on decode 2+ of the same admission.
        const std::size_t T = batch.tokens.size();
        const bool shape_ok = (n_pref == 0 && n_dec == 1 && T == 1);
        int lane = -1;
        bool seen = false;
        if (shape_ok && graphs_enabled_) {
            lane = row_slot[n_pref];
            seen = std::find(graph_seen_lanes_.begin(), graph_seen_lanes_.end(), lane) !=
                   graph_seen_lanes_.end();
            if (!seen) graph_seen_lanes_.push_back(lane);
        }
        const char* action = "eager-shape";
        if (!graphs_enabled_ || mtp_enabled_) {
            action = "eager-disabled";
        } else if (shape_ok && !seen) {
            action = "eager-first";
        }
        // Verbose-only step tracer: input id + first 8 hidden bytes pin down
        // whether a divergent step saw stale inputs or wrote stale hidden.
        // Synchronous D2H: diagnosis only, never on the hot path.
        auto vlog_step = [&](const char* act, const runtime::StepDecodedPairs& out) {
            if (!graph_verbose_ || out.empty()) return;
            std::int32_t in_id = -1;
            std::uint64_t h01  = 0;
            if (cudaMemcpy(&in_id, tensors.ids.data, sizeof(in_id),
                           cudaMemcpyDeviceToHost) != cudaSuccess) {
                return;
            }
            if (cudaMemcpy(&h01, tensors.hidden.data, sizeof(h01),
                           cudaMemcpyDeviceToHost) != cudaSuccess) {
                return;
            }
            std::fprintf(stderr, "[gtok] lane=%d action=%s in=%d out=%u h=%016llx\n", lane,
                         act, in_id, static_cast<unsigned>(out[0].second),
                         static_cast<unsigned long long>(h01));
        };
        // Pointer audit: device addresses are host-visible. x/hidden must be
        // identical across capture and replay runs (static scratch + reset
        // arena); any drift here is the alias bug.
        auto plog_step = [&](const char* act, const Tensor& x) {
            if (!graph_verbose_) return;
            std::fprintf(stderr, "[gptr] lane=%d action=%s x=%p hidden=%p ids=%p\n", lane, act,
                         x.data, tensors.hidden.data, tensors.ids.data);
        };
        if (!shape_ok || !seen) {
            if (graph_verbose_) {
                std::fprintf(stderr,
                             "[graph] lane=%d n_pref=%zu n_dec=%zu T=%zu action=%s\n",
                             lane, n_pref, n_dec, T, action);
            }
            auto out = card_->forward_serve_step(plan, batch, view, batch.seq_offsets.data(),
                                                 row_slot.data(), tensors, envelope_);
            vlog_step(action, out);
            return out;
        }
        ServeDecodeGraph* hit = graph_for_lane(lane);
        if (graph_verbose_) {
            std::fprintf(stderr, "[graph] lane=%d n_pref=%zu n_dec=%zu T=%zu action=%s\n",
                         lane, n_pref, n_dec, T, hit != nullptr ? "replay" : "capture");
        }
        if (hit == nullptr) {
            // No live exec and lane not dead: capture the layers-only call.
            // Embed runs eager first (outside the capture); uploads already
            // ran above. Capture failure parks the lane dead and recomputes
            // the step eager (same as the dry-run path below): work launched
            // under capture records but never executes, so no state was
            // applied and the layers call is safe to re-invoke eagerly.
            Tensor x = card_->embed_serve_input(tensors);
            if (graph_dry_) {
                // Diagnosis only: run the layers eagerly with no capture.
                // Fresh hidden here proves the method; stale hidden here
                // proves the seam (not capture mode) is broken.
                card_->forward_serve_decode_layers(plan, batch, view,
                                                   batch.seq_offsets.data(), row_slot.data(),
                                                   tensors, x, envelope_);
                auto out = card_->sample_decode_rows(tensors.hidden, plan, batch, stream);
                vlog_step("capture-dry", out);
                return out;
            }
            cudaGraph_t graph = nullptr;
            const cudaError_t begin = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
            if (begin == cudaSuccess) {
                try {
                    card_->forward_serve_decode_layers(plan, batch, view,
                                                       batch.seq_offsets.data(), row_slot.data(),
                                                       tensors, x, envelope_);
                } catch (...) {
                    cudaStreamEndCapture(stream, &graph);
                    if (graph != nullptr) cudaGraphDestroy(graph);
                    graph_mark_dead(lane);
                    throw;
                }
                if (cudaStreamEndCapture(stream, &graph) == cudaSuccess && graph != nullptr) {
                    cudaGraphExec_t exec = nullptr;
                    if (cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) == cudaSuccess &&
                        exec != nullptr) {
                        cudaGraphDestroy(graph);
                        graph_store(lane, exec);
                        // PLATFORM (2026-09-24, capmini-proven on this WSL box
                        // in Global AND ThreadLocal modes): work launched
                        // under capture records but never executes eagerly;
                        // replay is the only working executor. So the capture
                        // step's outputs must come from an immediate replay —
                        // sampling pre-launch reads the previous step's hidden
                        // (dup token). Costs one extra launch, once per lane
                        // admission; steady-state replays are untouched.
                        CUDA_CHECK(cudaGraphLaunch(exec, stream));
                        auto out = card_->sample_decode_rows(tensors.hidden, plan, batch, stream);
                        vlog_step("capture", out);
                        plog_step("capture", x);
                        return out;
                    }
                    if (exec != nullptr) cudaGraphExecDestroy(exec);
                    cudaGraphDestroy(graph);
                } else if (graph != nullptr) {
                    cudaGraphDestroy(graph);
                }
            }
            graph_mark_dead(lane);
            // Fail closed: recompute the step eager instead of sampling the
            // stale hidden. Captured work never executes (see PLATFORM note
            // above), so the layers call applies state exactly once here.
            card_->forward_serve_decode_layers(plan, batch, view,
                                               batch.seq_offsets.data(), row_slot.data(),
                                               tensors, x, envelope_);
            auto out = card_->sample_decode_rows(tensors.hidden, plan, batch, stream);
            vlog_step("capture-fail-eager", out);
            return out;
        }
        // Replay: embed stays eager (fresh token in, same arena address as
        // the capture run), then relaunch the recorded layers and sample.
        Tensor x = card_->embed_serve_input(tensors);
        (void)x;  // consumed by the graph at its baked arena address
        CUDA_CHECK(cudaGraphLaunch(hit->exec, stream));
        {
            auto out = card_->sample_decode_rows(tensors.hidden, plan, batch, stream);
            vlog_step("replay", out);
            plog_step("replay", x);
            return out;
        }
    }
};

class Engine::Impl {
public:
    using GenerationCore = runtime::EngineCore<runtime::ModelInstance>;
    using ScoringCore    = runtime::CausalScoreCore<runtime::ModelInstance>;
    using Core =
        std::variant<std::monostate, std::unique_ptr<GenerationCore>, std::unique_ptr<ScoringCore>>;

    explicit Impl(EngineOptions engine_options)
        : options(runtime::normalize_engine_options(std::move(engine_options))),
          device(initialize_device(options)) {
        nvtx::ScopedRange load_range(nvtx::Name::EngineLoad, nvtx::Category::Runtime);
        auto constructed  = runtime::construct_model(options, device);
        active            = std::move(constructed.instance);
        load              = std::move(constructed.load);
        sampling_defaults = active->frontend.sampling_defaults();
        StartupPhaseScope finalize_phase(options.startup_observer, StartupPhase::EngineFinalize);
        if (options.purpose == EnginePurpose::CausalScoring) {
            core = std::make_unique<ScoringCore>(*active, device);
        } else {
            core = std::make_unique<GenerationCore>(*active, device, options,
                                                    std::move(constructed.context_cost));
        }
        finalize_phase.complete();
    }

    ~Impl() noexcept {
        device.bind_to_current_thread_noexcept();
        core.emplace<std::monostate>();
        try {
            device.synchronize();
        } catch (...) {}
    }

    EngineOptions options;
    DeviceContext device;
    std::unique_ptr<runtime::ModelInstance> active;
    LoadSummary load;
    ModelSamplingDefaults sampling_defaults;
    Core core;
    // Serve mode: set by GenerationService. While true, submit() throws so
    // the serve path cannot accidentally use per-client generate loops.
    bool serve_mode = false;
    // Serve forward residency (single slot, Engine-owned): built lazily on
    // the first run_batch_step call. The scheduler owns page ids; GDN slots
    // ride the dispatch rows only.
    std::unique_ptr<ServeForwardContext> serve_forward;
    std::once_flag serve_forward_once;
};

Engine::Engine(EngineOptions options) {
    StartupObserver startup_observer = options.startup_observer;
    StartupPhaseScope startup_phase(startup_observer, StartupPhase::EngineStartup);
    impl_ = std::make_shared<Impl>(std::move(options));
    startup_phase.complete();
}

Engine::~Engine()                            = default;
Engine::Engine(Engine&&) noexcept            = default;
Engine& Engine::operator=(Engine&&) noexcept = default;

PreparedPrompt Engine::prepare(PromptInput input, const PreparationControl& control) const {
    nvtx::ScopedRange prepare_range(nvtx::Name::FrontendPrepare, nvtx::Category::Runtime);
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    auto prepared      = impl_->active->frontend.prepare(std::move(input), control);
    PromptSummary info = prepared.summary();
    const SamplingMode sampling_mode =
        info.starts_in_reasoning ? SamplingMode::Thinking : SamplingMode::NonThinking;
    if (info.prompt_tokens > impl_->active->capacity) {
        throw std::logic_error("target Frontend admitted a prompt beyond Engine capacity");
    }
    const PromptPreparationStats preparation = prepared.preparation_stats();
    return PreparedPrompt(std::make_unique<PreparedPrompt::Impl>(info, preparation, sampling_mode,
                                                                 std::move(prepared)));
}

PreparedPrompt Engine::prepare_tokens(std::vector<TokenId> token_ids,
                                      bool allow_prefix_identity) const {
    nvtx::ScopedRange prepare_range(nvtx::Name::FrontendPrepare, nvtx::Category::Runtime,
                                    static_cast<std::uint64_t>(token_ids.size()));
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    if (token_ids.size() > impl_->active->capacity) {
        throw RequestError(RequestErrorKind::ContextLengthExceeded,
                           context_capacity_error(token_ids.size(), impl_->active->capacity));
    }
    auto prepared =
        impl_->active->frontend.prepare_tokens(std::move(token_ids), allow_prefix_identity);
    PromptSummary info = prepared.summary();
    if (info.prompt_tokens > impl_->active->capacity) {
        throw std::logic_error("target Frontend admitted prompt tokens beyond capacity");
    }
    const PromptPreparationStats preparation = prepared.preparation_stats();
    return PreparedPrompt(std::make_unique<PreparedPrompt::Impl>(
        info, preparation, SamplingMode::Thinking, std::move(prepared)));
}

std::vector<TokenId> Engine::tokenize_text(std::string_view text) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->active->frontend.tokenize_text(text);
}

std::vector<float> Engine::score_tokens(std::vector<TokenId> tokens, std::uint32_t first_target) {
    nvtx::ScopedRange score_range(nvtx::Name::Score, nvtx::Category::Scoring,
                                  static_cast<std::uint64_t>(tokens.size()));
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    if (impl_->options.purpose != EnginePurpose::CausalScoring) {
        throw std::logic_error("score_tokens requires a CausalScoring Engine");
    }
    if (tokens.size() < 2 || tokens.size() > impl_->options.max_context) {
        throw std::invalid_argument("score_tokens token count must be in [2,max_context]");
    }
    if (first_target == 0 || first_target >= tokens.size()) {
        throw std::invalid_argument("score_tokens first_target must be in [1,token_count-1]");
    }
    PreparedPrompt prompt      = prepare_tokens(std::move(tokens), false);
    const std::size_t expected = prompt.summary().prompt_tokens - first_target;
    std::vector<float> result  = std::visit(
        [&](auto& core) -> std::vector<float> {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::unique_ptr<Impl::ScoringCore>>) {
                return core->score(std::move(prompt.impl_->value), first_target);
            } else {
                throw std::logic_error("Engine scoring core is unavailable");
            }
        },
        impl_->core);
    if (result.size() != expected) {
        throw std::logic_error("target Program returned an invalid causal score count");
    }
    return result;
}

std::uint32_t Engine::count_tokens(PromptInput input, const PreparationControl& control) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->active->frontend.count_tokens(std::move(input), control);
}

ModelSamplingDefaults Engine::sampling_defaults() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->sampling_defaults;
}

GenerationHandle Engine::submit(PreparedPrompt prompt, RequestOptions options,
                                OutputConsumerMode consumer_mode,
                                GenerationObservationOptions observation,
                                std::chrono::steady_clock::time_point pending_deadline) {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    // Committee A+B single scheduler: the serve path admits via hook_loop
    // EngineHooks::on_new_request and steps via Engine::run_batch_step. A
    // serve-mode submit is a programming error (second generate loop), so it
    // throws here. CLI never sets serve mode, keeping generate() usable.
    if (impl_->serve_mode) {
        throw std::logic_error("Engine::submit is disabled in serve mode: admit via hook_loop "
                               "EngineHooks::on_new_request and step via Engine::run_batch_step");
    }
    if (impl_->options.purpose != EnginePurpose::Generation) {
        throw std::logic_error("submit requires a Generation Engine");
    }
    if (prompt.impl_ == nullptr) { throw std::invalid_argument("PreparedPrompt is empty"); }
    if (observation.live_timings) { observation.phase_timings = true; }
    if (consumer_mode != OutputConsumerMode::Streaming &&
        (observation.live_timings || observation.prompt_progress)) {
        throw std::invalid_argument("live generation observations require a Streaming consumer");
    }

    runtime::ResolvedRequestOptions resolved_options = resolve_request_options(
        impl_->sampling_defaults, prompt.impl_->sampling_mode, std::move(options));
    const ResolvedSamplingParameters resolved_sampling = resolved_options.execution.sampling;

    const PromptSummary prompt_summary = prompt.impl_->summary;
    if (prompt_summary.prompt_tokens > impl_->options.max_context) {
        throw RequestError(
            RequestErrorKind::ContextLengthExceeded,
            context_capacity_error(prompt_summary.prompt_tokens, impl_->options.max_context));
    }
    const double prepare_seconds = prompt.impl_->prepare.seconds;
    if (resolved_options.execution.requested_output_tokens == 0) {
        struct ImmediateSubmission {
            GenerationResult result;
            OutputConsumerMode consumer_mode = OutputConsumerMode::Aggregate;

            GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) {
                const bool streaming = consumer_mode == OutputConsumerMode::Streaming;
                if (streaming != (sink != nullptr)) {
                    throw std::invalid_argument(
                        "GenerationHandle wait sink does not match its submitted consumer mode");
                }
                if (cancellation.requested()) { result.finish_reason = FinishReason::Cancelled; }
                return std::move(result);
            }
        } immediate{.consumer_mode = consumer_mode};

        immediate.result.prompt                     = prompt_summary;
        immediate.result.finish_reason              = FinishReason::OutputLimit;
        immediate.result.thinking.configured_budget = resolved_options.execution.thinking.budget;
        immediate.result.timings.prepare_seconds    = prepare_seconds;
        immediate.result.timings.total_seconds      = prepare_seconds;
        prompt.impl_.reset();
        return GenerationHandle(std::make_unique<GenerationHandle::Impl>(
            impl_, std::move(immediate), resolved_sampling));
    }

    return std::visit(
        [&](auto& core) -> GenerationHandle {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::monostate>) {
                throw std::logic_error("Engine core is unavailable");
            } else if constexpr (std::is_same_v<CoreState, std::unique_ptr<Impl::ScoringCore>>) {
                throw std::logic_error("Engine generation core is unavailable");
            } else {
                auto submission = core->submit(std::move(prompt.impl_->value), prompt_summary,
                                               prepare_seconds, std::move(resolved_options),
                                               consumer_mode, observation, pending_deadline);
                return GenerationHandle(std::make_unique<GenerationHandle::Impl>(
                    impl_, std::move(submission), resolved_sampling));
            }
        },
        impl_->core);
}

GenerationResult Engine::generate(PreparedPrompt prompt, RequestOptions options, OutputSink* sink,
                                  const CancellationView& cancellation) {
    const OutputConsumerMode consumer_mode =
        sink != nullptr ? OutputConsumerMode::Streaming : OutputConsumerMode::Aggregate;
    return submit(std::move(prompt), std::move(options), consumer_mode, {})
        .wait(sink, cancellation);
}

void Engine::set_serve_mode(bool serve) {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    impl_->serve_mode = serve;
}

std::vector<TokenId> Engine::prompt_token_ids(const PreparedPrompt& prompt) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    if (prompt.impl_ == nullptr) { throw std::invalid_argument("PreparedPrompt is empty"); }
    const models::qwen3_5::PreparedPromptData& data =
        models::qwen3_5::PreparedPromptAccess::view(prompt.impl_->value);
    return data.token_ids;
}

std::string Engine::decode_tokens(std::span<const TokenId> ids) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->active->frontend.decode_tokens(ids);
}

std::vector<TokenId> Engine::default_stop_token_ids() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->active->frontend.default_stop_policy().token_ids;
}

ResolvedSamplingParameters Engine::resolved_sampling(const PreparedPrompt& prompt,
                                                     const RequestOptions& options) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    if (prompt.impl_ == nullptr) { throw std::invalid_argument("PreparedPrompt is empty"); }
    runtime::ResolvedRequestOptions resolved = resolve_request_options(
        impl_->sampling_defaults, prompt.impl_->sampling_mode, options);
    return resolved.execution.sampling;
}

std::vector<std::pair<std::uint64_t, TokenId>>
Engine::run_batch_step(const batch::StepPlan& plan, const batch::StepDispatch& dispatch) {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    // Production forward (single slot, Engine-owned): embed dispatch tokens
    // at m == tokens.size(), per-layer production
    // causal_softmax_attention_ragged over the batch device view, EXL3 linear
    // per layer through the existing wrappers, then
    // TextContext::sample_decode_rows for decode rows only. The scheduler
    // owns page ids; the hook loop owns no TextContext/KV/GDN bytes.
    std::call_once(impl_->serve_forward_once, [&] {
        impl_->device.bind_to_current_thread();
        impl_->serve_forward = std::make_unique<ServeForwardContext>(
            impl_->device, impl_->active->parameters, impl_->options);
    });
    return impl_->serve_forward->step(plan, dispatch);
}

const EngineOptions& Engine::options() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->options;
}

LoadSummary Engine::load_summary() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->load;
}

MemorySummary Engine::memory_summary() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [](const auto& core) -> MemorySummary {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::monostate>) {
                throw std::logic_error("Engine core is unavailable");
            } else {
                return core->memory_summary();
            }
        },
        impl_->core);
}

MediaCacheSummary Engine::media_cache_summary() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->active->frontend.media_cache_summary();
}

RuntimeStats Engine::runtime_stats() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [](const auto& core) -> RuntimeStats {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::monostate>) {
                throw std::logic_error("Engine core is unavailable");
            } else {
                return core->runtime_stats();
            }
        },
        impl_->core);
}

bool Engine::is_available() const {
    if (impl_ == nullptr) { return false; }
    return std::visit(
        [](const auto& core) {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::monostate>) {
                return false;
            } else {
                return core != nullptr && core->is_available();
            }
        },
        impl_->core);
}

void Engine::reset_memory_peaks() noexcept {
    if (impl_ == nullptr) { return; }
    std::visit(
        [](auto& core) {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (!std::is_same_v<CoreState, std::monostate>) {
                core->reset_memory_peaks();
            }
        },
        impl_->core);
}

} // namespace ninfer
