#pragma once
#include "models/qwen3_5/program/internal.h"

#include "batch/batch.h"


#include "core/arena.h"
#include "core/device.h"
#include "core/gdn_replay_records.h"
#include "core/linear_attention_state.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/softmax_attention.h"
#include "ninfer/ops/sparse_moe.h"
#include "models/qwen3_5/state/decoder_state.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/program/round_buffers.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

using Phase = qwen3_5::TextPhase;

enum class GdnStateAction : std::uint8_t {
    UpdateInPlace,
    RecordForReplay,
};

struct NullTap {
    static constexpr bool enabled = false;
};

struct PrefillChunkResult {
    std::uint32_t processed_tokens = 0;
    bool finalized                 = false;
    runtime::ExecutionTiming timing;
};

struct DFlashFeatureSink {
    static constexpr bool enabled = true;
    using PrefillConsumer         = std::function<void(const Tensor&, const Tensor&, bool)>;

    Tensor* features                  = nullptr;
    Tensor* positions                 = nullptr;
    Tensor* batch_features            = nullptr;
    const Tensor* batch_lanes         = nullptr;
    const Tensor* batch_valid_columns = nullptr;
    std::int32_t batch_width          = 0;
    std::int32_t batch_size           = 0;
    std::span<const std::uint32_t> layers;
    PrefillConsumer consume_prefill;
    std::uint32_t captured_mask = 0;
    std::int32_t active_tokens  = 0;

    void begin(const Tensor& value);
    void capture_layer(int layer, const Tensor& value, cudaStream_t stream);
    void capture_positions(const Tensor& source, cudaStream_t stream);
    void consume_prefill_chunk(std::int32_t tokens, bool rewrite_checkpoint);
};

class VisionPrefillSession;

class TextContext {
public:
    TextContext(DeviceContext& ctx, const execution::Parameters& weights, WorkspaceArena& work,
                qwen3_5::PagedKVCacheView kv, LinearAttentionStatePool& state,
                qwen3_5::RoundState& io, Tensor& prefill_hidden, std::uint32_t prefill_chunk,
                std::uint32_t text_kv_base,
                qwen3_5::PagedKVCacheView mtp_kv           = qwen3_5::PagedKVCacheView(),
                const qwen3_5::PagedKVCache* batch_text_kv = nullptr,
                const qwen3_5::PagedKVCache* batch_mtp_kv  = nullptr);
    ~TextContext();

    TextContext(const TextContext&)            = delete;
    TextContext& operator=(const TextContext&) = delete;

    void set_proposal_head(const LinearParameters* weight, const std::int32_t* ids,
                           int count) noexcept {
        proposal_head_     = weight;
        proposal_head_ids_ = ids;
        proposal_head_n_   = count;
    }

    void set_sampling(const ops::SamplingConfig* config) noexcept { sampling_config_ = config; }

    void set_prefill_split_frontier(std::int64_t position) noexcept {
        prefill_split_frontier_ = position;
    }

    void set_rewrite_checkpoint_hidden_output(Tensor* output) noexcept {
        rewrite_checkpoint_hidden_output_ = output;
    }

    void set_mtp_proposal_extent(std::uint32_t extent) noexcept { mtp_proposal_extent_ = extent; }

    void set_linear_state_slots(std::int32_t source_slot, std::int32_t destination_slot);
    void set_gdn_state_action(GdnStateAction action, const GdnReplayRecords* replay_records);
    // Row 20b slice B (revised): commit verify column states into the lane.
    // Table is {k+1} = {lane, s1..sk} (static per lane, init-filled):
    // col c reads t[c-1]-as-source (c==0: the verify source/spare) and both
    // the recurrent chain and the per-column snapshot write t[c]. Accept-a
    // state is then t[a] (t[0]=lane, so a==0 is a no-op). a>=1 copies
    // t[a] -> lane via pool copy_slot (conv+recurrent together — no gather:
    // each column slot holds its own exact post-column window from the real
    // width-1 snapshot kernel). Throws fail-closed on shape surprise.
    void commit_verify_slots(std::int32_t lane, const std::int32_t* col_slots,
                             std::uint32_t accepted, std::int32_t width);
    // E2 ragged serve-batch binding: device gather output plus the host
    // offsets mirror. Set for batch-mode steps (flat T live tokens); cleared
    // for single-request paths. Never reads host block tables.
    void set_ragged_batch(const batch::DeviceRaggedBatch* view,
                          const std::uint32_t* host_seq_offsets) noexcept {
        active_ragged_batch_   = view;
        active_ragged_offsets_ = host_seq_offsets;
    }
    void clear_ragged_batch() noexcept { set_ragged_batch(nullptr, nullptr); }

    [[nodiscard]] const LinearParameters* proposal_head() const noexcept { return proposal_head_; }

    [[nodiscard]] const std::int32_t* proposal_head_ids() const noexcept {
        return proposal_head_ids_;
    }

    [[nodiscard]] int proposal_head_n() const noexcept { return proposal_head_n_; }

    [[nodiscard]] PrefillChunkResult prefill_chunk(std::span<const int> full_ids,
                                                   std::uint32_t begin,
                                                   std::uint32_t nominal_length,
                                                   bool finalize_at_end);
    [[nodiscard]] PrefillChunkResult prefill_chunk(std::span<const int> full_ids,
                                                   std::uint32_t begin,
                                                   std::uint32_t nominal_length,
                                                   bool finalize_at_end, DFlashFeatureSink& sink);
    [[nodiscard]] PrefillChunkResult
    prefill_chunk(const qwen3_5::PreparedPromptData& input, std::uint32_t begin,
                  std::uint32_t nominal_length, VisionPrefillSession& vision, bool finalize_at_end);
    [[nodiscard]] PrefillChunkResult prefill_chunk(const qwen3_5::PreparedPromptData& input,
                                                   std::uint32_t begin,
                                                   std::uint32_t nominal_length,
                                                   VisionPrefillSession& vision,
                                                   bool finalize_at_end, DFlashFeatureSink& sink);
    void ordinary_decode_batch(const Tensor& ids, const Tensor& cache_positions,
                               const Tensor& rope_positions, const Tensor& kv_table_rows,
                               const Tensor& linear_state_source_slots,
                               const Tensor& linear_state_destination_slots,
                               ops::CausalAttentionExecutionEnvelope envelope, Tensor& hidden,
                               Tensor& logits);
    // Slot C: sample decode rows only. hidden is the post-final-norm flat-T
    // hidden matrix [H, T]; plan/batch give the ragged layout (prefill slices
    // first in plan order, then one 1-token row per decode seq). Prefill rows
    // are never sampled: only columns [prefill_tokens, prefill_tokens+n_dec)
    // are projected and sampled, one token per plan.decode_seq_ids entry.
    [[nodiscard]] std::vector<std::pair<std::uint64_t, TokenId>>
    sample_decode_rows(const Tensor& hidden, const batch::StepPlan& plan,
                       const batch::RaggedBatch& batch, cudaStream_t stream);
    // Slot B: production ragged serve forward for one StepPlan. All step
    // tensors are Engine-owned device buffers (this method allocates nothing
    // persistent): ids/positions are flat-T I32, kv_table_rows is
    // I32[num_seqs] over the owning cache execution rows, hidden is BF16
    // [H, T] scratch that sample_decode_rows consumes below. row_slots
    // carries one persistent GDN state slot per ragged row (prefill rows
    // first in plan order, then one 1-token row per decode seq). The ragged
    // device view plus the host offsets mirror bind the flat-T path in
    // attn_mix; GDN layers run one Prefill-phase update per prefill row plus
    // one width-1 Verify batch over the decode suffix (Verify GDN is
    // uniform-only). Returns sample_decode_rows(...) for decode rows only.
    struct ServeStepTensors {
        Tensor ids;                       // I32 [T]
        Tensor cache_positions;           // I32 [T]
        Tensor rope_positions;            // I32 [T]
        Tensor kv_table_rows;             // I32 [num_seqs]
        Tensor decode_source_slots;       // I32 [n_dec]; empty when n_dec == 0
        Tensor decode_destination_slots;  // I32 [n_dec]; empty when n_dec == 0
        Tensor hidden;                    // BF16 [H, T]
    };

    [[nodiscard]] std::vector<std::pair<std::uint64_t, TokenId>>
    forward_serve_step(const batch::StepPlan& plan, const batch::RaggedBatch& batch,
                       const batch::DeviceRaggedBatch& ragged,
                       const std::uint32_t* host_seq_offsets, const std::int32_t* row_slots,
                       ServeStepTensors& tensors,
                       ops::CausalAttentionExecutionEnvelope envelope);
    // Slot D: graph-safe layers-only decode forward. Embed and sample stay
    // with the caller (eager, outside any capture): embed_serve_input runs
    // the embedding into arena x, forward_serve_decode_layers runs bindings,
    // the 64-layer loop and the final norm into tensors.hidden. Decode-only
    // (n_pref == 0, n_dec > 0): no prefill branches, no sample, no D2H, no
    // persistent allocs. Safe under cudaStreamBeginCapture on ctx_.stream
    // with static tensor addresses: every per-step input arrives as device
    // contents (ids/positions/tables/slots uploaded eagerly beforehand).
    Tensor embed_serve_input(ServeStepTensors& tensors);
    void forward_serve_decode_layers(const batch::StepPlan& plan,
                                     const batch::RaggedBatch& batch,
                                     const batch::DeviceRaggedBatch& ragged,
                                     const std::uint32_t* host_seq_offsets,
                                     const std::int32_t* row_slots,
                                     ServeStepTensors& tensors, Tensor& x,
                                     ops::CausalAttentionExecutionEnvelope envelope);
    void reset_work() { work_.reset(); }
    void target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                             const Tensor& rope_positions, const Tensor& valid_columns,
                             const Tensor& kv_table_rows, const Tensor& linear_state_source_slots,
                             ops::CausalAttentionExecutionEnvelope envelope, Tensor& hidden,
                             Tensor& logits, Tensor& target_tokens);
    void target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                             const Tensor& rope_positions, const Tensor& valid_columns,
                             const Tensor& kv_table_rows, const Tensor& linear_state_source_slots,
                             ops::CausalAttentionExecutionEnvelope envelope, Tensor& hidden,
                             Tensor& logits, Tensor& target_tokens, DFlashFeatureSink& sink);
    // Lane B: snapshot verify with an explicit destination slot (serve MTP-3
    // batched verify: source = spare; the destination slot is inert when
    // the column table is set). Width > 1 chains across the window inside
    // the snapshot kernels; the lane is untouched until the commit.
    // Record path untouched (no sink on this arm).
    void target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                             const Tensor& rope_positions, const Tensor& valid_columns,
                             const Tensor& kv_table_rows, const Tensor& linear_state_source_slots,
                             const Tensor& linear_state_destination_slots,
                             ops::CausalAttentionExecutionEnvelope envelope, Tensor& hidden,
                             Tensor& logits, Tensor& target_tokens,
                             const Tensor* linear_state_column_slots = nullptr);
    void mtp_forward_decode_batch(const Tensor& ids, const Tensor& hidden,
                                  const Tensor& cache_positions, const Tensor& rope_positions,
                                  const Tensor& valid_columns, const Tensor& kv_table_rows,
                                  ops::CausalAttentionExecutionEnvelope envelope,
                                  Tensor& mtp_hidden);
    void mtp_propose_batch(const Tensor& hidden, Tensor& logits, Tensor& draft_tokens);
    void mtp_forward_batch(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                           ops::CausalAttentionExecutionEnvelope envelope, Tensor& mtp_hidden,
                           int logits_column, Tensor* logits, Tensor* draft_token,
                           const Tensor* explicit_rope_positions = nullptr,
                           const Tensor* input_embeddings        = nullptr);
    void mtp_forward_ar_step(const Tensor& token, const Tensor& previous_hidden,
                             const Tensor& position, ops::CausalAttentionExecutionEnvelope envelope,
                             Tensor& mtp_hidden, Tensor& logits, Tensor& draft_token);
    // Row-23 rung-1: project one target hidden column through lm_head_.
    // Warming fills MTP KV only, so per-row target logits are skipped;
    // the slice tail still needs its anchor logits (bonus token), via
    // exactly one projection per request.
    void project_target_tail(const Tensor& hidden_col, Tensor& logits);
private:
    [[nodiscard]] bool mtp_enabled() const noexcept {
        return mtp_kv_.valid() || batch_mtp_kv_ != nullptr;
    }

    void attn_mix(const BlockParameters& weights, Tensor& x, int index, Phase phase);
    void gdn_mix(const BlockParameters& weights, Tensor& x, int index, Phase phase);
    void mlp_tail(const BlockParameters& weights, Tensor& x, Phase phase,
                  const ops::SparseMoeHints& hints);
    [[nodiscard]] ops::SparseMoeHints next_projection_hints(int layer) const;
    void run_layers(Tensor& x, Phase phase);
    template <class Tap>
    void run_layers(Tensor& x, Phase phase, Tap& tap);
    template <class Tap>
    void target_verify_batch_impl(const Tensor& ids, const Tensor& cache_positions,
                                  const Tensor& rope_positions, const Tensor& valid_columns,
                                  const Tensor& kv_table_rows,
                                  const Tensor& linear_state_source_slots,
                                  const Tensor* linear_state_destination_slots,
                                  const Tensor* linear_state_column_slots,
                                  ops::CausalAttentionExecutionEnvelope envelope, Tensor& hidden,
                                  Tensor& logits, Tensor& target_tokens, Tap& tap);

    void mtp_forward_stem(const Tensor& ids, const Tensor& hidden, const Tensor* input_embeddings,
                          Tensor& x, Tensor& ah);
    void mtp_forward_tail(Tensor& x, const Tensor& ah, const Tensor& positions,
                          const Tensor& rope_positions,
                          ops::CausalAttentionExecutionEnvelope envelope, Tensor& mtp_hidden);
    void mtp_forward_core(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                          const Tensor& rope_positions,
                          ops::CausalAttentionExecutionEnvelope envelope, Tensor& mtp_hidden,
                          const Tensor* input_embeddings);
    void mtp_prefill_chunk(const Tensor& ids, const Tensor& hidden, const Tensor* input_embeddings,
                           const Tensor& positions, const Tensor& rope_positions,
                           ops::CausalAttentionExecutionEnvelope envelope, bool final_chunk,
                           Tensor* final_hidden, Tensor* logits, Tensor* draft_token);
    void proposal_argmax(const Tensor& hidden, Tensor& logits, Tensor& proposal_tokens);

    struct MultimodalPrefill {
        std::span<const int> token_ids;
        std::span<const std::int32_t> positions;
        VisionPrefillSession* vision = nullptr;
        std::uint32_t begin          = 0;
        std::int32_t rope_delta      = 0;
    };

    struct TextPrefill {
        std::span<const int> token_ids;
        std::uint32_t begin = 0;
    };

    template <class Tap>
    [[nodiscard]] PrefillChunkResult
    prefill_impl(std::span<const int> ids, const TextPrefill* text_prefill,
                 const MultimodalPrefill* multimodal, Tap& tap, bool finalize_at_end);
    DeviceContext& ctx_;
    const Parameters& parameters_;
    const TextConfig& config_;
    WorkspaceArena& work_;
    qwen3_5::PagedKVCacheView kv_;
    qwen3_5::PagedKVCacheView mtp_kv_;
    const qwen3_5::PagedKVCache* batch_text_kv_ = nullptr;
    const qwen3_5::PagedKVCache* batch_mtp_kv_  = nullptr;
    LinearAttentionStatePool& state_;
    qwen3_5::RoundState& io_;
    Tensor& prefill_hidden_;
    std::uint32_t prefill_chunk_;
    std::uint32_t text_kv_base_;
    const Tensor* active_cache_positions_                                          = nullptr;
    const Tensor* active_rope_positions_                                           = nullptr;
    const Tensor* active_kv_table_rows_                                            = nullptr;
    const Tensor* active_linear_state_source_slots_                                = nullptr;
    const Tensor* active_linear_state_destination_slots_                           = nullptr;
    // Row 20b layout-B column slots: optional {k+1} int32 device array
    // (batch==1 only): t[0]=lane (col-0 destination, in place for a==0),
    // t[1..k]=column slots. Col c reads (c==0 ? source : t[c-1]) and writes
    // t[c], for BOTH the recurrent chain and the per-column snapshot below.
    // Accept-a state is t[a]. Null = legacy ping/pong fallback.
    const Tensor* active_linear_state_column_slots_                                = nullptr;
    const Tensor* active_valid_columns_                                            = nullptr;
    const Tensor* active_backend_kv_table_rows_                                    = nullptr;
    const ops::CausalAttentionExecutionEnvelope* active_causal_attention_envelope_ = nullptr;
    // E2 ragged serve-batch binding (see set_ragged_batch): when non-null,
    // attn_mix takes the flat-T ragged path and never the uniform [W,B] path.
    const batch::DeviceRaggedBatch* active_ragged_batch_   = nullptr;
    const std::uint32_t* active_ragged_offsets_            = nullptr;
    std::int32_t active_sequence_batch_                                            = 0;
    std::int32_t active_sequence_width_                                            = 0;
    std::int32_t rope_delta_                                                       = 0;
    std::int32_t linear_state_source_slot_                                         = 0;
    std::int32_t linear_state_destination_slot_                                    = 0;
    GdnStateAction gdn_state_action_          = GdnStateAction::UpdateInPlace;
    const GdnReplayRecords* replay_records_   = nullptr;
    std::int64_t prefill_split_frontier_      = -1;
    Tensor* rewrite_checkpoint_hidden_output_ = nullptr;
    std::uint32_t mtp_proposal_extent_        = 0;

    const Weight* embed_                        = nullptr;
    const Tensor* final_norm_                   = nullptr;
    const LinearParameters* lm_head_            = nullptr;
    const LinearParameters* proposal_head_      = nullptr;
    const std::int32_t* proposal_head_ids_      = nullptr;
    int proposal_head_n_                        = 0;
    const ops::SamplingConfig* sampling_config_ = nullptr;
    const MtpParameters* mtp_                   = nullptr;
};

} // namespace ninfer::models::qwen3_5::execution
