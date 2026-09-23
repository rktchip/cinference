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

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
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
                    .mtp_layers                = 0,
                    .capacity                  = max_context_,
                    .kv_heads                  = dimension(text_config.attention->num_key_value_heads),
                    .attention_head_dim        = dimension(text_config.attention->head_dim),
                    .kv_storage                = options.kv_cache,
                    .enable_mtp                = false,
                    .kv_table_rows             = static_cast<std::int32_t>(max_seqs_),
                    .text_physical_page_groups = physical_pages,
                    .mtp_physical_page_groups  = 0,
                });
        kv_store_ = DeviceBuffer(kv_builder.finish(256));
        kv_       = std::make_unique<models::qwen3_5::PagedKVCache>(
            DeviceSpan{kv_store_.p, kv_store_.bytes}, kv_layout.text_kv);

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
                    .slot_count     = static_cast<std::int32_t>(max_seqs_),
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
                    .draft_window    = 0,
                    .backend         = SpeculativeBackend::None,
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
        scratch_ = DeviceBuffer(off);
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
            models::qwen3_5::PagedKVCacheView{}, kv_.get(), nullptr);
        envelope_ = ops::CausalAttentionExecutionEnvelope{1, max_context_};
        {
            // Explicit T=0 sampling config: temperature 0 resolves to greedy.
            ops::SamplingConfig explicit_argmax;
            explicit_argmax.temperature = 0.0F;
            explicit_argmax.top_k = 20;
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
        slots_.resize(max_seqs_);
        device_.synchronize();
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
                std::vector<DeviceKVPageHandle> pages;
                pages.reserve(pages_per_slot_);
                for (std::uint32_t l = 0; l < pages_per_slot_; ++l) {
                    pages.push_back(
                        page_leases_[static_cast<std::size_t>(found) * pages_per_slot_ + l]
                            .handle());
                }
                kv_->page_pool().zero_pages(pages, device_.stream);
                pool_->zero_slot(found, device_.stream);
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
        auto decoded = card_->forward_serve_step(plan, batch, view, batch.seq_offsets.data(),
                                                 row_slot.data(), tensors, envelope_);
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

private:
    struct ServeSlot {
        bool in_use             = false;
        std::uint64_t seq_id    = 0;
        std::uint32_t next_pos  = 0;
        bool touched_as_prefill = false;
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
