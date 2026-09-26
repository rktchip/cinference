#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/execution/text.h"
#include "models/qwen3_5/execution/ladder_trace.h"
#include "models/qwen3_5/execution/attention.h"
#include "models/qwen3_5/execution/gdn.h"
#include "models/qwen3_5/execution/ffn.h"
#include "models/qwen3_5/execution/mtp.h"
#include "models/qwen3_5/execution/workspace.h"

#include "core/nvtx.h"
#include "ops/linear/exl3/exl3_op.h"
#include "models/qwen3_5/execution/visual_scatter.h"
#include "models/qwen3_5/execution/vision.h"
#include "models/qwen3_5/program/vision_control.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/gdn_gating.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/speculative_round.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/mtp_pack.h"
#include "ninfer/ops/position.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/sparse_moe.h"
#include "ninfer/ops/scatter.h"
#include "ops/wrapper/qg_scatter.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/sigmoid_mul.h"
#include "ninfer/ops/silu_mul.h"
#include <atomic>
#include "ninfer/ops/softmax_attention.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::execution {
namespace {

void project(const Tensor& x, const LinearParameters& parameters, Tensor& out,
             WorkspaceArena& workspace, cudaStream_t stream) {
    ops::linear(x, parameters.weight, out, parameters.policy, workspace, stream);
}

void copy_i32(const std::int32_t* source, Tensor& destination, cudaStream_t stream) {
    if (source == nullptr || destination.dtype != DType::I32 || !destination.is_contiguous() ||
        destination.data == nullptr) {
        throw std::invalid_argument("copy_i32: invalid host source or I32 destination");
    }
    CUDA_CHECK(cudaMemcpyAsync(destination.data, source, destination.bytes(),
                               cudaMemcpyHostToDevice, stream));
}

void require_tensor_shape(const Tensor& t, DType dtype, std::initializer_list<std::int32_t> shape,
                          const char* label) {
    if (t.dtype != dtype) { throw std::invalid_argument(std::string(label) + " dtype mismatch"); }
    int i = 0;
    for (const std::int32_t dim : shape) {
        if (t.ne[i] != dim) { throw std::invalid_argument(std::string(label) + " shape mismatch"); }
        ++i;
    }
    for (; i < 4; ++i) {
        if (t.ne[i] != 1) { throw std::invalid_argument(std::string(label) + " shape mismatch"); }
    }
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string(label) + " must be contiguous");
    }
    if (t.data == nullptr) { throw std::invalid_argument(std::string(label) + " data is null"); }
}

void require_tensor_window(const Tensor& t, DType dtype, std::int32_t rows, std::int32_t cols,
                           const char* label) {
    if (cols <= 0) { throw std::invalid_argument(std::string(label) + " cols must be positive"); }
    if (t.dtype != dtype) { throw std::invalid_argument(std::string(label) + " dtype mismatch"); }
    if (t.ne[0] != rows || t.ne[1] < cols || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument(std::string(label) + " shape mismatch");
    }
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string(label) + " must be contiguous");
    }
    if (t.data == nullptr) { throw std::invalid_argument(std::string(label) + " data is null"); }
}

Tensor matrix_window(Tensor& t, std::int32_t cols) {
    if (cols <= 0) { throw std::invalid_argument("matrix_window cols must be positive"); }
    if (t.ne[1] < cols || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument("matrix_window shape mismatch");
    }
    return t.slice(1, 0, cols);
}

class ScopedPositions {
public:
    ScopedPositions(const Tensor*& slot, const Tensor& positions) : slot_(slot) {
        slot_ = &positions;
    }

    ScopedPositions(const ScopedPositions&)            = delete;
    ScopedPositions& operator=(const ScopedPositions&) = delete;

    ~ScopedPositions() { slot_ = nullptr; }

private:
    const Tensor*& slot_;
};

class ScopedEnvelope {
public:
    ScopedEnvelope(const ops::CausalAttentionExecutionEnvelope*& slot,
                   const ops::CausalAttentionExecutionEnvelope& envelope)
        : slot_(slot) {
        slot_ = &envelope;
    }

    ScopedEnvelope(const ScopedEnvelope&)            = delete;
    ScopedEnvelope& operator=(const ScopedEnvelope&) = delete;

    ~ScopedEnvelope() { slot_ = nullptr; }

private:
    const ops::CausalAttentionExecutionEnvelope*& slot_;
};

template <class T>
class ScopedValue {
public:
    ScopedValue(T& slot, T value) : slot_(slot), previous_(slot) { slot_ = value; }

    ScopedValue(const ScopedValue&)            = delete;
    ScopedValue& operator=(const ScopedValue&) = delete;

    ~ScopedValue() { slot_ = previous_; }

private:
    T& slot_;
    T previous_;
};

} // namespace

// Row 20b pulse: per-column execution counters for the coltab branches.
// The oracle reads + resets them around the slots verify (expect 192/192
// conv/rec per verify = 48 GDN layers x 4 cols). Zero = legacy ran.
std::atomic<std::uint64_t> g_coltab_conv_cols{0};
std::atomic<std::uint64_t> g_coltab_rec_cols{0};

void DFlashFeatureSink::begin(const Tensor& value) {
    const bool prefill = features != nullptr && positions != nullptr && batch_features == nullptr;
    const bool batch   = batch_features != nullptr && batch_lanes != nullptr &&
                       batch_valid_columns != nullptr && batch_width > 0 && batch_size > 0;
    if ((!prefill && !batch) || layers.empty()) {
        throw std::logic_error("DFlash feature sink is incomplete");
    }
    captured_mask = 0;
    active_tokens = batch ? batch_width * batch_size : value.ne[1];
    if (value.ne[1] != active_tokens) {
        throw std::logic_error("DFlash batch feature source has an invalid width");
    }
}

void DFlashFeatureSink::capture_layer(int layer, const Tensor& value, cudaStream_t stream) {
    const auto it = std::find(layers.begin(), layers.end(), layer);
    if (it == layers.end()) { return; }
    const std::size_t index = static_cast<std::size_t>(it - layers.begin());
    Tensor* destination     = batch_features != nullptr ? batch_features : features;
    if (layers.size() > 32 || active_tokens <= 0 || value.dtype != DType::BF16 ||
        destination == nullptr ||
        value.ne[0] * static_cast<std::int32_t>(layers.size()) != destination->ne[0] ||
        value.ne[1] != active_tokens) {
        throw std::logic_error("DFlash feature capture shape is invalid");
    }
    if (batch_features != nullptr) {
        Tensor source = value.view({value.ne[0], batch_width, batch_size});
        Tensor target =
            batch_features->slice(0, static_cast<std::int32_t>(index) * value.ne[0], value.ne[0]);
        ops::scatter_bf16_batch(source, *batch_lanes, *batch_valid_columns, target, stream);
        captured_mask |= 1U << index;
        return;
    }
    if (active_tokens > features->ne[1]) {
        throw std::logic_error("DFlash prefill feature capture exceeds its buffer");
    }
    const std::size_t element_bytes = dtype_size(DType::BF16);
    const std::size_t width_bytes   = static_cast<std::size_t>(value.ne[0]) * element_bytes;
    const std::size_t source_pitch  = static_cast<std::size_t>(value.nb[1]);
    const std::size_t target_pitch  = static_cast<std::size_t>(features->nb[1]);
    auto* target                    = static_cast<std::byte*>(features->data) + index * width_bytes;
    CUDA_CHECK(cudaMemcpy2DAsync(target, target_pitch, value.data, source_pitch, width_bytes,
                                 static_cast<std::size_t>(active_tokens), cudaMemcpyDeviceToDevice,
                                 stream));
    captured_mask |= 1U << index;
}

void DFlashFeatureSink::capture_positions(const Tensor& source, cudaStream_t stream) {
    const std::uint32_t complete_mask = layers.size() == 32 ? ~0U : ((1U << layers.size()) - 1U);
    if (captured_mask != complete_mask) {
        throw std::logic_error("DFlash target call did not publish every feature layer");
    }
    if (batch_features != nullptr) {
        if (source.dtype != DType::I32 || source.ne[0] != batch_width ||
            source.ne[1] != batch_size) {
            throw std::logic_error("DFlash batch feature positions are invalid");
        }
        return;
    }
    if (active_tokens <= 0 || source.dtype != DType::I32 || source.ne[0] != active_tokens ||
        positions == nullptr || active_tokens > positions->ne[0]) {
        throw std::logic_error("DFlash feature positions are invalid");
    }
    CUDA_CHECK(cudaMemcpyAsync(positions->data, source.data,
                               static_cast<std::size_t>(active_tokens) * sizeof(std::int32_t),
                               cudaMemcpyDeviceToDevice, stream));
}

void DFlashFeatureSink::consume_prefill_chunk(std::int32_t tokens, bool rewrite_checkpoint) {
    if (!consume_prefill || tokens != active_tokens) {
        throw std::logic_error("DFlash prefill feature consumer is unavailable");
    }
    Tensor feature_window  = features->slice(1, 0, tokens);
    Tensor position_window = positions->slice(0, 0, tokens);
    consume_prefill(feature_window, position_window, rewrite_checkpoint);
}

TextContext::TextContext(DeviceContext& ctx, const execution::Parameters& weights,
                         WorkspaceArena& work, qwen3_5::PagedKVCacheView kv,
                         LinearAttentionStatePool& state, qwen3_5::RoundState& io,
                         Tensor& prefill_hidden, std::uint32_t prefill_chunk,
                         std::uint32_t text_kv_base, qwen3_5::PagedKVCacheView mtp_kv,
                         const qwen3_5::PagedKVCache* batch_text_kv,
                         const qwen3_5::PagedKVCache* batch_mtp_kv)
    : ctx_(ctx), parameters_(weights), config_(weights.model.config().text), work_(work), kv_(kv),
      mtp_kv_(mtp_kv), state_(state), io_(io), prefill_hidden_(prefill_hidden),
      prefill_chunk_(prefill_chunk), text_kv_base_(text_kv_base), batch_text_kv_(batch_text_kv),
      batch_mtp_kv_(batch_mtp_kv) {
    if (prefill_chunk_ == 0 ||
        prefill_chunk_ > static_cast<std::uint32_t>((std::numeric_limits<std::int32_t>::max)())) {
        throw std::invalid_argument("TextContext effective prefill chunk must fit positive int32");
    }
    if (mtp_enabled() && !io_.mtp_decode && !io_.mtp) {
        throw std::invalid_argument("MTP TextContext requires MTP round state");
    }
    set_linear_state_slots(0, 0);
    embed_      = &parameters_.text.token_embedding;
    final_norm_ = &parameters_.text.final_norm;
    lm_head_    = &parameters_.text.output_head;
    mtp_        = parameters_.mtp ? &*parameters_.mtp : nullptr;
    if (mtp_enabled() && mtp_ == nullptr) {
        throw std::invalid_argument("MTP state requires selected MTP parameters");
    }
    if (parameters_.proposal) {
        const auto& p = *parameters_.proposal;
        set_proposal_head(
            &p.head, p.token_ids ? static_cast<const std::int32_t*>(p.token_ids->data) : nullptr,
            dimension(p.rows));
    }
}

TextContext::~TextContext() = default;

void TextContext::set_linear_state_slots(std::int32_t source_slot, std::int32_t destination_slot) {
    if (source_slot < 0 || source_slot >= state_.slot_count() || destination_slot < 0 ||
        destination_slot >= state_.slot_count()) {
        throw std::invalid_argument("TextContext Linear Attention slots are invalid");
    }
    linear_state_source_slot_      = source_slot;
    linear_state_destination_slot_ = destination_slot;
}

void TextContext::set_gdn_state_action(GdnStateAction action,
                                       const GdnReplayRecords* replay_records) {
    if ((action == GdnStateAction::RecordForReplay) != (replay_records != nullptr)) {
        throw std::invalid_argument("TextContext GDN state action has inconsistent records");
    }
    gdn_state_action_ = action;
    replay_records_   = replay_records;
}

void TextContext::commit_verify_slots(std::int32_t lane, const std::int32_t* col_slots,
                                      std::uint32_t accepted, std::int32_t width) {
    cudaStream_t s = ctx_.stream;
    if (lane < 0 || lane >= state_.slot_count() || col_slots == nullptr || width < 2 ||
        accepted > static_cast<std::uint32_t>(width - 1)) {
        throw std::invalid_argument("commit_verify_slots: bad lane/table/shape");
    }
    // Layout A: accept-a state is t[a] in the consecutive column block;
    // copy it back into the lane at EVERY accept value, including a==0
    // (the lane holds the pre-step window; conv runs every step, so the
    // lane is never written during verify).
    const std::int32_t src = col_slots[accepted];
    if (src < 0 || src >= state_.slot_count()) {
        throw std::invalid_argument("commit_verify_slots: column slot OOB");
    }
    state_.copy_slot(src, lane, s);
}

void TextContext::mtp_forward_stem(const Tensor& ids, const Tensor& hidden,
                                   const Tensor* input_embeddings, Tensor& x, Tensor& ah) {
    cudaStream_t s     = ctx_.stream;
    const int T        = ids.ne[0] * ids.ne[1];
    Tensor flat_ids    = ids.view({T});
    Tensor flat_hidden = hidden.view({dimension(config_.hidden_size), T});

    auto roots = workspace::mtp_stem(work_, config_, T, input_embeddings == nullptr);
    Tensor emb;
    if (input_embeddings != nullptr) {
        if (input_embeddings->dtype != DType::BF16 ||
            input_embeddings->ne[0] != dimension(config_.hidden_size) ||
            input_embeddings->numel() !=
                static_cast<std::int64_t>(dimension(config_.hidden_size)) * T ||
            !input_embeddings->is_contiguous() || input_embeddings->data == nullptr) {
            throw std::invalid_argument("MTP input embeddings shape mismatch");
        }
        emb = input_embeddings->view({dimension(config_.hidden_size), T});
    } else {
        emb = roots.embedding;
        ops::embedding(flat_ids, *embed_, emb, s);
    }

    Tensor e = roots.normalized_embedding;
    Tensor h = roots.normalized_hidden;
    ops::rmsnorm(emb, mtp_->embedding_norm, config_.rms_norm_eps, true, e, s);
    ops::rmsnorm(flat_hidden, mtp_->hidden_norm, config_.rms_norm_eps, true, h, s);

    Tensor fc_in = roots.packed_input;
    ops::mtp_pack_fc_input(e, h, fc_in, s);

    x = roots.residual;
    project(fc_in, mtp_->input_projection, x, work_, s);

    ah = roots.attention_hidden;
    ops::rmsnorm(x, mtp_->input_norm, config_.rms_norm_eps, true, ah, s);
}

void TextContext::mtp_forward_tail(Tensor& x, const Tensor& ah, const Tensor& positions,
                                   const Tensor& rope_positions,
                                   ops::CausalAttentionExecutionEnvelope envelope,
                                   Tensor& mtp_hidden) {
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];

    const auto projection = workspace::mtp_attention_projection(work_, config_, T);
    Tensor q              = projection.query.view({dimension(config_.attention->head_dim),
                                                   dimension(config_.attention->num_attention_heads), T});
    Tensor k              = projection.key.view({dimension(config_.attention->head_dim),
                                                 dimension(config_.attention->num_key_value_heads), T});
    Tensor gate           = projection.gate.view({dimension(config_.attention->head_dim),
                                                  dimension(config_.attention->num_attention_heads), T});
    Tensor v              = projection.value.view({dimension(config_.attention->head_dim),
                                                   dimension(config_.attention->num_key_value_heads), T});
    Tensor q_flat         = q.view({dimension(config_.attention->query_width()), T});
    Tensor gate_flat      = gate.view({dimension(config_.attention->query_width()), T});
    Tensor k_flat         = k.view({dimension(config_.attention->key_width()), T});
    Tensor v_flat         = v.view({dimension(config_.attention->key_width()), T});
    mtp_projection(ah, mtp_->projection, *config_.attention, q_flat, gate_flat, k_flat, v_flat,
                   work_, s);

    const auto results = workspace::mtp_attention_results(work_, config_, T);
    Tensor qn =
        results.normalized_query.view({dimension(config_.attention->head_dim),
                                       dimension(config_.attention->num_attention_heads), T});
    Tensor kn = results.normalized_key.view({dimension(config_.attention->head_dim),
                                             dimension(config_.attention->num_key_value_heads), T});
    ops::rmsnorm(q, mtp_->query_norm, config_.rms_norm_eps, true, qn, s);
    ops::rmsnorm(k, mtp_->key_norm, config_.rms_norm_eps, true, kn, s);
    Tensor rope_for_op = active_sequence_batch_ != 0 ? rope_positions.view({T}) : rope_positions;
    text_rope(rope_for_op, *config_.rope_parameters, qn, kn, s);

    Tensor a = results.attention.view({dimension(config_.attention->head_dim),
                                       dimension(config_.attention->num_attention_heads), T});
    if (active_sequence_batch_ != 0) {
        const std::int32_t width = active_sequence_width_;
        if (width <= 0 || width * active_sequence_batch_ != T ||
            active_backend_kv_table_rows_ == nullptr || active_valid_columns_ == nullptr) {
            throw std::logic_error("MTP sequence batch binding is incomplete");
        }
        Tensor q_batch        = qn.view({dimension(config_.attention->head_dim),
                                         dimension(config_.attention->num_attention_heads), width,
                                         active_sequence_batch_});
        Tensor k_batch        = kn.view({dimension(config_.attention->head_dim),
                                         dimension(config_.attention->num_key_value_heads), width,
                                         active_sequence_batch_});
        Tensor v_batch        = v.view({dimension(config_.attention->head_dim),
                                        dimension(config_.attention->num_key_value_heads), width,
                                        active_sequence_batch_});
        Tensor a_batch        = a.view({dimension(config_.attention->head_dim),
                                        dimension(config_.attention->num_attention_heads), width,
                                        active_sequence_batch_});
        Tensor position_batch = positions.view({width, active_sequence_batch_});
        ops::causal_softmax_attention(
            q_batch, k_batch, v_batch, position_batch, *active_valid_columns_,
            *active_backend_kv_table_rows_,
            {dimension(config_.attention->head_dim),
             dimension(config_.attention->num_attention_heads),
             dimension(config_.attention->num_key_value_heads)},
            static_cast<float>(1.0 / std::sqrt(static_cast<double>(config_.attention->head_dim))),
            batch_mtp_kv_->batch_layer_view(0), envelope, work_, a_batch, s);
    } else {
        ops::causal_softmax_attention(
            qn, kn, v, positions, Tensor{}, io_.backend_kv_table_row,
            {dimension(config_.attention->head_dim),
             dimension(config_.attention->num_attention_heads),
             dimension(config_.attention->num_key_value_heads)},
            static_cast<float>(1.0 / std::sqrt(static_cast<double>(config_.attention->head_dim))),
            batch_mtp_kv_->batch_layer_view(0), envelope, work_, a, s);
    }
    ops::sigmoid_mul(gate, a, s);

    const auto post = workspace::mtp_post_attention(work_, config_, T);
    Tensor o        = post.output;
    project(a.view({dimension(config_.attention->query_width()), T}), mtp_->output, o, work_, s);
    ops::residual_add(o, x, s);

    Tensor mh = post.post_mixer_hidden;
    ops::rmsnorm(x, mtp_->post_attention_norm, config_.rms_norm_eps, true, mh, s);

    {
        auto post_mixer_scope = work_.scope();
        ffn(mh, mtp_->ffn, x, {}, work_, s, true);
    }

    Tensor flat_mtp_hidden = mtp_hidden.view({dimension(config_.hidden_size), T});
    ops::rmsnorm(x, mtp_->final_norm, config_.rms_norm_eps, true, flat_mtp_hidden, s);
}

void TextContext::mtp_forward_core(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                                   const Tensor& rope_positions,
                                   ops::CausalAttentionExecutionEnvelope envelope,
                                   Tensor& mtp_hidden, const Tensor* input_embeddings) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    nvtx::ScopedRange forward_range(nvtx::Name::MtpForward, nvtx::Category::Mtp,
                                    static_cast<std::uint64_t>(ids.numel()));
    auto scratch_scope = work_.scope();
    Tensor x;
    Tensor ah;
    mtp_forward_stem(ids, hidden, input_embeddings, x, ah);
    mtp_forward_tail(x, ah, positions, rope_positions, envelope, mtp_hidden);
}

void TextContext::mtp_prefill_chunk(const Tensor& ids, const Tensor& hidden,
                                    const Tensor* input_embeddings, const Tensor& positions,
                                    const Tensor& rope_positions,
                                    ops::CausalAttentionExecutionEnvelope envelope,
                                    bool final_chunk, Tensor* final_hidden, Tensor* logits,
                                    Tensor* draft_token) {
    if (!mtp_kv_.valid()) { throw std::runtime_error("MTP prefill is not enabled"); }
    const int T = ids.ne[0];
    if (T <= 0 || static_cast<std::uint32_t>(T) > prefill_chunk_) {
        throw std::invalid_argument("MTP prefill chunk T must be in [1,prefill_chunk]");
    }
    nvtx::ScopedRange mtp_prefill_range(nvtx::Name::PrefillMtpChunk, nvtx::Category::Mtp,
                                        static_cast<std::uint64_t>(T));
    require_tensor_shape(ids, DType::I32, {T}, "MTP prefill ids");
    require_tensor_shape(hidden, DType::BF16, {dimension(config_.hidden_size), T},
                         "MTP prefill hidden");
    require_tensor_shape(positions, DType::I32, {T}, "MTP prefill positions");
    if (rope_positions.dtype != DType::I32 || rope_positions.ne[0] != T ||
        (rope_positions.ne[1] != 1 && rope_positions.ne[1] != 3) || rope_positions.ne[2] != 1 ||
        rope_positions.ne[3] != 1 || !rope_positions.is_contiguous() ||
        rope_positions.data == nullptr) {
        throw std::invalid_argument("MTP prefill rope positions must be [T] or [T,3]");
    }
    if (final_chunk) {
        if (final_hidden == nullptr || logits == nullptr || draft_token == nullptr) {
            throw std::invalid_argument("MTP final prefill outputs are required");
        }
        require_tensor_shape(*final_hidden, DType::BF16, {dimension(config_.hidden_size), 1},
                             "MTP final prefill hidden");
        require_tensor_shape(*logits, DType::BF16, {dimension(config_.vocab_size), 1},
                             "MTP final prefill logits");
        require_tensor_shape(*draft_token, DType::I32, {1}, "MTP final prefill draft token");
    }

    cudaStream_t s     = ctx_.stream;
    auto scratch_scope = work_.scope();
    Tensor x_last;
    Tensor ah_last;
    if (final_chunk) {
        x_last  = work_.alloc(DType::BF16, {dimension(config_.hidden_size), 1});
        ah_last = work_.alloc(DType::BF16, {dimension(config_.hidden_size), 1});
    }

    {
        auto bulk_scope = work_.scope();
        Tensor x;
        Tensor ah;
        mtp_forward_stem(ids, hidden, input_embeddings, x, ah);

        Tensor k_flat = work_.alloc(DType::BF16, {dimension(config_.attention->key_width()), T});
        Tensor v_flat = work_.alloc(DType::BF16, {dimension(config_.attention->key_width()), T});
        mtp_kv_projection(ah, mtp_->projection, *config_.attention, k_flat, v_flat, work_, s);
        Tensor k = k_flat.view({dimension(config_.attention->head_dim),
                                dimension(config_.attention->num_key_value_heads), T});
        Tensor v = v_flat.view({dimension(config_.attention->head_dim),
                                dimension(config_.attention->num_key_value_heads), T});
        Tensor kn =
            work_.alloc(DType::BF16, {dimension(config_.attention->head_dim),
                                      dimension(config_.attention->num_key_value_heads), T});
        ops::rmsnorm(k, mtp_->key_norm, config_.rms_norm_eps, true, kn, s);
        text_rope(rope_positions, *config_.rope_parameters, kn, s);
        ops::kv_cache_append(kn, v, positions, mtp_kv_.layer_view(0), s);

        if (final_chunk) {
            const std::size_t column_bytes =
                static_cast<std::size_t>(dimension(config_.hidden_size)) * dtype_size(DType::BF16);
            const auto* x_src = static_cast<const unsigned char*>(x.data) +
                                static_cast<std::size_t>(T - 1) * column_bytes;
            const auto* ah_src = static_cast<const unsigned char*>(ah.data) +
                                 static_cast<std::size_t>(T - 1) * column_bytes;
            CUDA_CHECK(
                cudaMemcpyAsync(x_last.data, x_src, column_bytes, cudaMemcpyDeviceToDevice, s));
            CUDA_CHECK(
                cudaMemcpyAsync(ah_last.data, ah_src, column_bytes, cudaMemcpyDeviceToDevice, s));
        }
    }

    if (final_chunk) {
        Tensor q_flat = work_.alloc(DType::BF16, {dimension(config_.attention->query_width()), 1});
        Tensor gate_flat =
            work_.alloc(DType::BF16, {dimension(config_.attention->query_width()), 1});
        mtp_query_gate_projection(ah_last, mtp_->projection, *config_.attention, q_flat, gate_flat,
                                  work_, s);
        Tensor q    = q_flat.view({dimension(config_.attention->head_dim),
                                   dimension(config_.attention->num_attention_heads), 1});
        Tensor gate = gate_flat.view({dimension(config_.attention->head_dim),
                                      dimension(config_.attention->num_attention_heads), 1});
        Tensor qn =
            work_.alloc(DType::BF16, {dimension(config_.attention->head_dim),
                                      dimension(config_.attention->num_attention_heads), 1});
        ops::rmsnorm(q, mtp_->query_norm, config_.rms_norm_eps, true, qn, s);
        Tensor last_position = positions.slice(0, T - 1, 1);
        Tensor last_rope_position;
        if (rope_positions.ne[1] == 1) {
            last_rope_position = rope_positions.slice(0, T - 1, 1);
        } else {
            last_rope_position = work_.alloc(DType::I32, {1, 3});
            for (int axis = 0; axis < 3; ++axis) {
                const auto* src = static_cast<const std::int32_t*>(rope_positions.data) +
                                  static_cast<std::size_t>(axis) * T + (T - 1);
                auto* dst = static_cast<std::int32_t*>(last_rope_position.data) + axis;
                CUDA_CHECK(
                    cudaMemcpyAsync(dst, src, sizeof(std::int32_t), cudaMemcpyDeviceToDevice, s));
            }
        }
        text_rope(last_rope_position, *config_.rope_parameters, qn, s);

        Tensor a = work_.alloc(DType::BF16, {dimension(config_.attention->head_dim),
                                             dimension(config_.attention->num_attention_heads), 1});
        ops::causal_softmax_attention_cached(
            qn, last_position,
            {dimension(config_.attention->head_dim),
             dimension(config_.attention->num_attention_heads),
             dimension(config_.attention->num_key_value_heads)},
            static_cast<float>(1.0 / std::sqrt(static_cast<double>(config_.attention->head_dim))),
            mtp_kv_.layer_view(0), envelope, work_, a, s);
        ops::sigmoid_mul(gate, a, s);

        Tensor o = work_.alloc(DType::BF16, {dimension(config_.hidden_size), 1});
        project(a.view({dimension(config_.attention->query_width()), 1}), mtp_->output, o, work_,
                s);
        ops::residual_add(o, x_last, s);

        Tensor mh = work_.alloc(DType::BF16, {dimension(config_.hidden_size), 1});
        ops::rmsnorm(x_last, mtp_->post_attention_norm, config_.rms_norm_eps, true, mh, s);
        {
            auto post_mixer_scope = work_.scope();
            ffn(mh, mtp_->ffn, x_last, {}, work_, s, true);
        }
        ops::rmsnorm(x_last, mtp_->final_norm, config_.rms_norm_eps, true, *final_hidden, s);
        proposal_argmax(*final_hidden, *logits, *draft_token);
    }
}

void TextContext::proposal_argmax(const Tensor& hidden, Tensor& logits, Tensor& proposal_tokens) {
    auto proposal_scope = work_.scope();
    const int T         = hidden.ne[1];
    require_tensor_shape(hidden, DType::BF16, {dimension(config_.hidden_size), T},
                         "proposal hidden");
    require_tensor_shape(proposal_tokens, DType::I32, {T}, "proposal tokens");
    require_tensor_window(logits, DType::BF16, dimension(config_.vocab_size), T, "proposal logits");
    nvtx::ScopedRange proposal_range(nvtx::Name::MtpProposal, nvtx::Category::Mtp,
                                     static_cast<std::uint64_t>(T));
    if (proposal_head_ != nullptr) {
        Tensor proposal_logits = work_.alloc(DType::BF16, {proposal_head_n_, T});
        project(hidden, *proposal_head_, proposal_logits, work_, ctx_.stream);
        ops::argmax(proposal_logits, proposal_tokens,
                    proposal_head_ids_
                        ? proposal_head_n_
                        : dimension(parameters_.model.resources().public_token_count),
                    ctx_.stream);
        if (proposal_head_ids_ != nullptr) {
            ops::proposal_remap_token_ids(proposal_tokens, proposal_head_ids_, proposal_head_n_,
                                          ctx_.stream);
        }
    } else {
        Tensor output_logits = matrix_window(logits, T);
        project(hidden, mtp_->output_head, output_logits, work_, ctx_.stream);
        ops::argmax(output_logits, proposal_tokens,
                    dimension(parameters_.model.resources().public_token_count), ctx_.stream);
    }
}

void TextContext::mtp_forward_batch(const Tensor& ids, const Tensor& hidden,
                                    const Tensor& positions,
                                    ops::CausalAttentionExecutionEnvelope envelope,
                                    Tensor& mtp_hidden, int logits_column, Tensor* logits,
                                    Tensor* draft_token, const Tensor* explicit_rope_positions,
                                    const Tensor* input_embeddings) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    const int T = ids.ne[0];
    if (T <= 0 || static_cast<std::uint32_t>(T) > prefill_chunk_) {
        throw std::invalid_argument("MTP batch T must be in [1,prefill_chunk]");
    }
    require_tensor_shape(ids, DType::I32, {T}, "MTP ids");
    require_tensor_shape(positions, DType::I32, {T}, "MTP positions");
    require_tensor_shape(hidden, DType::BF16, {dimension(config_.hidden_size), T}, "MTP hidden");
    require_tensor_shape(mtp_hidden, DType::BF16, {dimension(config_.hidden_size), T},
                         "MTP output hidden");
    if (logits_column >= T) { throw std::invalid_argument("MTP logits column out of range"); }
    if (logits_column >= 0) {
        if (logits == nullptr || draft_token == nullptr) {
            throw std::invalid_argument("MTP logits and draft_token outputs are required");
        }
        require_tensor_shape(*logits, DType::BF16, {dimension(config_.vocab_size), 1},
                             "MTP logits");
        require_tensor_shape(*draft_token, DType::I32, {1}, "MTP draft token");
    }

    auto position_scope = work_.scope();
    Tensor generated_rope_positions;
    const Tensor* rope_positions = explicit_rope_positions;
    if (rope_positions == nullptr) {
        generated_rope_positions = work_.alloc(DType::I32, {T});
        ops::offset_i32_positions(positions, io_.rope_delta, generated_rope_positions, ctx_.stream);
        rope_positions = &generated_rope_positions;
    } else if (rope_positions->dtype != DType::I32 || rope_positions->ne[0] != T ||
               (rope_positions->ne[1] != 1 && rope_positions->ne[1] != 3) ||
               rope_positions->ne[2] != 1 || rope_positions->ne[3] != 1 ||
               !rope_positions->is_contiguous() || rope_positions->data == nullptr) {
        throw std::invalid_argument("MTP explicit rope positions must be [T] or [T,3]");
    }
    mtp_forward_core(ids, hidden, positions, *rope_positions, envelope, mtp_hidden,
                     input_embeddings);

    if (logits_column >= 0) {
        auto logits_scope = work_.scope();
        Tensor col        = mtp_hidden.slice(1, logits_column, 1);
        proposal_argmax(col, *logits, *draft_token);
    }
}

void TextContext::mtp_forward_ar_step(const Tensor& token, const Tensor& previous_hidden,
                                      const Tensor& position,
                                      ops::CausalAttentionExecutionEnvelope envelope,
                                      Tensor& mtp_hidden, Tensor& logits, Tensor& draft_token) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    require_tensor_shape(token, DType::I32, {1}, "MTP AR token");
    require_tensor_shape(position, DType::I32, {1}, "MTP AR position");
    require_tensor_shape(previous_hidden, DType::BF16, {dimension(config_.hidden_size), 1},
                         "MTP AR previous hidden");
    require_tensor_shape(mtp_hidden, DType::BF16, {dimension(config_.hidden_size), 1},
                         "MTP AR output hidden");
    require_tensor_shape(logits, DType::BF16, {dimension(config_.vocab_size), 1}, "MTP AR logits");
    require_tensor_shape(draft_token, DType::I32, {1}, "MTP AR draft token");

    auto position_scope  = work_.scope();
    Tensor rope_position = work_.alloc(DType::I32, {1});
    ops::offset_i32_positions(position, io_.rope_delta, rope_position, ctx_.stream);
    mtp_forward_core(token, previous_hidden, position, rope_position, envelope, mtp_hidden,
                     nullptr);
    auto logits_scope = work_.scope();
    proposal_argmax(mtp_hidden, logits, draft_token);
}

void TextContext::ordinary_decode_batch(const Tensor& ids, const Tensor& cache_positions,
                                        const Tensor& rope_positions, const Tensor& kv_table_rows,
                                        const Tensor& linear_state_source_slots,
                                        const Tensor& linear_state_destination_slots,
                                        ops::CausalAttentionExecutionEnvelope envelope,
                                        Tensor& hidden, Tensor& logits) {
    const std::int32_t batch = ids.ne[0];
    if (batch <= 0 || batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("ordinary decode batch size must be in [1,8]");
    }
    require_tensor_shape(ids, DType::I32, {batch}, "ordinary decode ids");
    require_tensor_shape(cache_positions, DType::I32, {batch}, "ordinary decode cache positions");
    require_tensor_shape(rope_positions, DType::I32, {batch}, "ordinary decode RoPE positions");
    require_tensor_shape(kv_table_rows, DType::I32, {batch}, "ordinary decode KV rows");
    require_tensor_shape(linear_state_source_slots, DType::I32, {batch},
                         "ordinary decode Linear Attention source slots");
    require_tensor_shape(linear_state_destination_slots, DType::I32, {batch},
                         "ordinary decode Linear Attention destination slots");
    require_tensor_shape(hidden, DType::BF16, {dimension(config_.hidden_size), batch},
                         "ordinary decode hidden");
    require_tensor_shape(logits, DType::BF16, {dimension(config_.vocab_size), batch},
                         "ordinary decode logits");

    cudaStream_t stream = ctx_.stream;
    work_.reset();
    {
        ScopedPositions cache_binding(active_cache_positions_, cache_positions);
        ScopedPositions rope_binding(active_rope_positions_, rope_positions);
        ScopedEnvelope envelope_binding(active_causal_attention_envelope_, envelope);
        ScopedValue<const Tensor*> kv_binding(active_kv_table_rows_, &kv_table_rows);
        ScopedValue<const Tensor*> source_binding(active_linear_state_source_slots_,
                                                  &linear_state_source_slots);
        ScopedValue<const Tensor*> destination_binding(active_linear_state_destination_slots_,
                                                       &linear_state_destination_slots);
        ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
        ScopedValue<std::int32_t> width_binding(active_sequence_width_, 1);

        Tensor x = work_.alloc(DType::BF16, {dimension(config_.hidden_size), batch});
        ops::embedding(ids, *embed_, x, stream);
        NullTap tap;
        run_layers(x, Phase::Verify, tap);
        ops::rmsnorm(x, *final_norm_, config_.rms_norm_eps, true, hidden, stream);
        project(hidden, *lm_head_, logits, work_, stream);
    }
    work_.reset();
}

std::vector<std::pair<std::uint64_t, TokenId>> TextContext::sample_decode_rows(
    const Tensor& hidden, const batch::StepPlan& plan, const batch::RaggedBatch& batch,
    cudaStream_t stream) {
    const std::size_t n_dec = plan.decode_seq_ids.size();
    if (n_dec == 0) { return {}; }
    if (lm_head_ == nullptr) {
        throw std::logic_error("sample_decode_rows requires a bound lm_head");
    }
    if (n_dec > static_cast<std::size_t>((std::numeric_limits<std::int32_t>::max)())) {
        throw std::invalid_argument("sample_decode_rows decode count overflows int32");
    }
    const std::size_t n_pref_seqs = plan.prefill.size();
    const std::size_t num_seqs    = batch.num_seqs();
    if (n_pref_seqs + n_dec != num_seqs) {
        throw std::logic_error("sample_decode_rows plan/batch sequence count mismatch");
    }
    if (batch.seq_offsets.size() != num_seqs + 1 || batch.seq_ids.size() != num_seqs) {
        throw std::logic_error("sample_decode_rows ragged offsets/ids are inconsistent");
    }
    if (batch.seq_offsets[0] != 0) {
        throw std::logic_error("sample_decode_rows ragged offsets must start at 0");
    }
    // Prefill block first in plan order: row s must carry plan.prefill[s].
    for (std::size_t s = 0; s < n_pref_seqs; ++s) {
        if (batch.seq_ids[s] != plan.prefill[s].seq_id) {
            throw std::logic_error("sample_decode_rows prefill row seq_id mismatch");
        }
        if (batch.seq_offsets[s + 1] < batch.seq_offsets[s] ||
            batch.seq_offsets[s + 1] - batch.seq_offsets[s] != plan.prefill[s].count) {
            throw std::logic_error("sample_decode_rows prefill row span mismatch");
        }
    }
    const std::uint32_t prefill_tokens = batch.seq_offsets[n_pref_seqs];
    // Decode rows follow: each is exactly one token and must carry the
    // matching plan.decode_seq_ids entry. Prefill columns [0, prefill_tokens)
    // are verified here and never projected or sampled below.
    for (std::size_t i = 0; i < n_dec; ++i) {
        const std::size_t s = n_pref_seqs + i;
        if (batch.seq_ids[s] != plan.decode_seq_ids[i]) {
            throw std::logic_error("sample_decode_rows decode row seq_id mismatch");
        }
        if (batch.seq_offsets[s + 1] != batch.seq_offsets[s] + 1) {
            throw std::logic_error("sample_decode_rows decode row must span exactly one token");
        }
    }
    if (batch.tokens.size() != batch.seq_offsets[num_seqs]) {
        throw std::logic_error("sample_decode_rows ragged token count mismatch");
    }
    const std::uint64_t total_needed = static_cast<std::uint64_t>(prefill_tokens) + n_dec;
    if (total_needed >
        static_cast<std::uint64_t>((std::numeric_limits<std::int32_t>::max)())) {
        throw std::invalid_argument("sample_decode_rows token range overflows int32");
    }
    const std::int32_t hidden_dim = dimension(config_.hidden_size);
    if (hidden.dtype != DType::BF16 || hidden.ne[0] != hidden_dim || hidden.ne[2] != 1 ||
        hidden.ne[3] != 1 || !hidden.is_contiguous() || hidden.data == nullptr ||
        hidden.ne[1] < static_cast<std::int32_t>(total_needed)) {
        throw std::invalid_argument("sample_decode_rows hidden must be contiguous BF16 [H,T]");
    }

    const std::int32_t first_col = static_cast<std::int32_t>(prefill_tokens);
    const std::int32_t ndec_cols = static_cast<std::int32_t>(n_dec);
    const std::int32_t vocab     = dimension(config_.vocab_size);
    const std::int32_t domain    = dimension(parameters_.model.resources().public_token_count);
    if (vocab <= 0 || domain <= 0 || domain > vocab) {
        throw std::logic_error("sample_decode_rows vocabulary domain is invalid");
    }

    auto arena_scope = work_.scope();
    // Decode-only window: prefill columns [0, prefill_tokens) are excluded by
    // construction, never the old prefill tail (last-column) slice.
    Tensor dec_h  = hidden.slice(1, first_col, ndec_cols);
    Tensor logits = work_.alloc(DType::BF16, {vocab, ndec_cols});
    project(dec_h, *lm_head_, logits, work_, stream);
    ladder::ladder_dump(logits, "logits", -1, stream);
    Tensor out = work_.alloc(DType::I32, {ndec_cols});
    if (sampling_config_ != nullptr) {
        // O1: one batched sample for all decode rows (was: one ops::sample
        // per row). Batched entry is proven in-tree (decode.cpp:55); row b's
        // RNG key (seed, logical_positions[b], purpose) is independent of
        // compact row index, so stochastic rows draw identically. Serve
        // runs greedy (temperature<=0: RNG skipped) with penalties 0 and
        // null token_counts (engine.cpp serve_sampling_), i.e. no per-row
        // side effects at all — batching is bit-identical there. Callers
        // with penalties/concurrency-sensitive counts keep per-lane configs
        // on the program path; this site broadcasts the one shared config.
        std::vector<std::int32_t> host_positions(n_dec);
        for (std::size_t i = 0; i < n_dec; ++i) {
            host_positions[i] = first_col + static_cast<std::int32_t>(i);
        }
        Tensor positions = work_.alloc(DType::I32, {ndec_cols});
        copy_i32(host_positions.data(), positions, stream);
        const ops::SamplingConfig* configs = sampling_config_;
        Tensor configs_array;
        if (n_dec > 1) {
            configs_array = work_.alloc(DType::U8, {static_cast<std::int32_t>(
                n_dec * sizeof(ops::SamplingConfig))});
            for (std::size_t i = 0; i < n_dec; ++i) {
                CUDA_CHECK(cudaMemcpyAsync(
                    static_cast<char*>(configs_array.data) + i * sizeof(ops::SamplingConfig),
                    sampling_config_, sizeof(ops::SamplingConfig),
                    cudaMemcpyDeviceToDevice, stream));
            }
            configs = static_cast<const ops::SamplingConfig*>(configs_array.data);
        }
        ops::sample(logits, out, domain, configs, positions,
                    ops::kSamplePurposeDecode, work_, stream);
    } else {
        ops::argmax(logits, out, domain, stream);
    }
    std::vector<TokenId> host_tokens(n_dec);
    CUDA_CHECK(cudaMemcpyAsync(host_tokens.data(), out.data,
                               n_dec * sizeof(TokenId), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    // DIAGNOSTIC-ONLY trace (env-gated, no numeric effect): per-decode-row
    // top-8 logits + distribution stats; optional full-domain u16 append to
    // NINFER_LOGITS_DUMP_PATH for offline solo-vs-mixed maxAbs. Reads the
    // already-projected logits; never writes device state.
    if (std::getenv("NINFER_LOGITS_DUMP") != nullptr) {
        const std::size_t dump_elems = static_cast<std::size_t>(vocab) * n_dec;
        std::vector<std::uint16_t> dump_host(dump_elems);
        CUDA_CHECK(cudaMemcpyAsync(dump_host.data(), logits.data,
                                   dump_elems * sizeof(std::uint16_t),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (std::size_t dump_i = 0; dump_i < n_dec; ++dump_i) {
            const std::uint16_t* dump_col =
                dump_host.data() + dump_i * static_cast<std::size_t>(vocab);
            const std::size_t dump_srow = n_pref_seqs + dump_i;
            const TokenId dump_in_tok = batch.tokens[batch.seq_offsets[dump_srow]];
            int dump_top_id[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
            float dump_top_val[8];
            for (int dump_k = 0; dump_k < 8; ++dump_k) {
                dump_top_val[dump_k] = -std::numeric_limits<float>::infinity();
            }
            float dump_max = -std::numeric_limits<float>::infinity();
            double dump_sum = 0.0;
            long dump_nan = 0;
            for (std::int32_t dump_v = 0; dump_v < domain; ++dump_v) {
                const std::uint32_t dump_bits =
                    static_cast<std::uint32_t>(dump_col[dump_v]) << 16;
                float dump_x = 0.0F;
                std::memcpy(&dump_x, &dump_bits, sizeof(dump_x));
                if (dump_x != dump_x) { ++dump_nan; continue; }
                dump_sum += static_cast<double>(dump_x);
                if (dump_x > dump_max) { dump_max = dump_x; }
                for (int dump_k = 0; dump_k < 8; ++dump_k) {
                    if (dump_x > dump_top_val[dump_k] ||
                        (dump_x == dump_top_val[dump_k] && dump_v < dump_top_id[dump_k])) {
                        for (int dump_j = 7; dump_j > dump_k; --dump_j) {
                            dump_top_val[dump_j] = dump_top_val[dump_j - 1];
                            dump_top_id[dump_j] = dump_top_id[dump_j - 1];
                        }
                        dump_top_val[dump_k] = dump_x;
                        dump_top_id[dump_k] = dump_v;
                        break;
                    }
                }
            }
            double dump_lse_shift = 0.0;
            for (std::int32_t dump_v = 0; dump_v < domain; ++dump_v) {
                const std::uint32_t dump_bits =
                    static_cast<std::uint32_t>(dump_col[dump_v]) << 16;
                float dump_x = 0.0F;
                std::memcpy(&dump_x, &dump_bits, sizeof(dump_x));
                if (dump_x != dump_x) { continue; }
                dump_lse_shift += std::exp(static_cast<double>(dump_x - dump_max));
            }
            const double dump_lse = static_cast<double>(dump_max) + std::log(dump_lse_shift);
            const double dump_mean_logprob = dump_sum / static_cast<double>(domain) - dump_lse;
            float dump_samp_logit = std::numeric_limits<float>::quiet_NaN();
            if (host_tokens[dump_i] >= 0 && host_tokens[dump_i] < domain) {
                const std::uint32_t dump_bits =
                    static_cast<std::uint32_t>(dump_col[host_tokens[dump_i]]) << 16;
                std::memcpy(&dump_samp_logit, &dump_bits, sizeof(dump_samp_logit));
            }
            std::fprintf(stderr,
                         "[logits-dump] seq=%llu col=%zu in_tok=%d first_col=%d ndec=%d "
                         "vocab=%d domain=%d tok=%d nan=%ld "
                         "top8=%d:%.4f,%d:%.4f,%d:%.4f,%d:%.4f,%d:%.4f,%d:%.4f,%d:%.4f,%d:%.4f "
                         "max=%.4f meanlogp=%.4f samplogit=%.4f\n",
                         (unsigned long long)plan.decode_seq_ids[dump_i], dump_i,
                         (int)dump_in_tok, first_col, ndec_cols, vocab, domain,
                         (int)host_tokens[dump_i], dump_nan,
                         dump_top_id[0], (double)dump_top_val[0],
                         dump_top_id[1], (double)dump_top_val[1],
                         dump_top_id[2], (double)dump_top_val[2],
                         dump_top_id[3], (double)dump_top_val[3],
                         dump_top_id[4], (double)dump_top_val[4],
                         dump_top_id[5], (double)dump_top_val[5],
                         dump_top_id[6], (double)dump_top_val[6],
                         dump_top_id[7], (double)dump_top_val[7],
                         (double)dump_max, dump_mean_logprob, (double)dump_samp_logit);
            std::fflush(stderr);
            const char* dump_path = std::getenv("NINFER_LOGITS_DUMP_PATH");
            if (dump_path != nullptr && dump_path[0] != 0) {
                std::FILE* dump_fp = std::fopen(dump_path, "ab");
                if (dump_fp != nullptr) {
                    std::fprintf(dump_fp, "SEQ %llu COL %zu INTOK %d FIRSTCOL %d NDEC %d DOMAIN %d TOK %d\n",
                                 (unsigned long long)plan.decode_seq_ids[dump_i], dump_i,
                                 (int)dump_in_tok, first_col, ndec_cols, domain,
                                 (int)host_tokens[dump_i]);
                    std::fwrite(dump_col, sizeof(std::uint16_t),
                                static_cast<std::size_t>(domain), dump_fp);
                    std::fclose(dump_fp);
                } else {
                    std::fprintf(stderr, "[logits-dump] WARN cannot append %s\n", dump_path);
                    std::fflush(stderr);
                }
            }
        }
    }
    std::vector<std::pair<std::uint64_t, TokenId>> decoded;
    decoded.reserve(n_dec);
    for (std::size_t i = 0; i < n_dec; ++i) {
        decoded.emplace_back(plan.decode_seq_ids[i], host_tokens[i]);
    }
    return decoded;
}

std::vector<std::pair<std::uint64_t, TokenId>> TextContext::forward_serve_step(
    const batch::StepPlan& plan, const batch::RaggedBatch& batch,
    const batch::DeviceRaggedBatch& ragged, const std::uint32_t* host_seq_offsets,
    const std::int32_t* row_slots, ServeStepTensors& tensors,
    ops::CausalAttentionExecutionEnvelope envelope) {
    const std::size_t n_dec    = plan.decode_seq_ids.size();
    const std::size_t n_pref   = plan.prefill.size();
    const std::size_t num_seqs = batch.num_seqs();
    if (num_seqs == 0 || n_pref + n_dec != num_seqs) {
        throw std::logic_error("forward_serve_step plan/batch sequence count mismatch");
    }
    const std::uint64_t total = batch.tokens.size();
    if (total == 0 ||
        total > static_cast<std::uint64_t>((std::numeric_limits<std::int32_t>::max)())) {
        throw std::invalid_argument("forward_serve_step token count out of range");
    }
    if (host_seq_offsets == nullptr || row_slots == nullptr) {
        throw std::invalid_argument("forward_serve_step requires host offsets and row slots");
    }
    for (std::size_t r = 0; r < num_seqs; ++r) {
        if (row_slots[r] < 0) {
            throw std::invalid_argument("forward_serve_step requires a bound slot per ragged row");
        }
    }
    const std::int32_t T    = static_cast<std::int32_t>(total);
    const std::int32_t nseq = static_cast<std::int32_t>(num_seqs);
    const std::int32_t ndec = static_cast<std::int32_t>(n_dec);
    require_tensor_shape(tensors.ids, DType::I32, {T}, "serve step ids");
    require_tensor_shape(tensors.cache_positions, DType::I32, {T}, "serve step cache positions");
    require_tensor_shape(tensors.rope_positions, DType::I32, {T}, "serve step rope positions");
    require_tensor_shape(tensors.kv_table_rows, DType::I32, {nseq}, "serve step KV rows");
    require_tensor_shape(tensors.hidden, DType::BF16, {dimension(config_.hidden_size), T},
                         "serve step hidden");
    if (ndec > 0) {
        require_tensor_shape(tensors.decode_source_slots, DType::I32, {ndec},
                             "serve step decode source slots");
        require_tensor_shape(tensors.decode_destination_slots, DType::I32, {ndec},
                             "serve step decode destination slots");
    }
    if (config_.layer_types.size() != parameters_.text.layers.size()) {
        throw std::logic_error("forward_serve_step layer inventory is inconsistent");
    }
    const std::int32_t prefill_tokens = static_cast<std::int32_t>(host_seq_offsets[n_pref]);

    cudaStream_t stream = ctx_.stream;
    work_.reset();
    set_ragged_batch(&ragged, host_seq_offsets);
    {
        ScopedPositions cache_binding(active_cache_positions_, tensors.cache_positions);
        ScopedPositions rope_binding(active_rope_positions_, tensors.rope_positions);
        ScopedEnvelope envelope_binding(active_causal_attention_envelope_, envelope);
        ScopedValue<const Tensor*> kv_binding(active_kv_table_rows_, &tensors.kv_table_rows);

        Tensor x = work_.alloc(DType::BF16, {dimension(config_.hidden_size), T});
        ops::embedding(tensors.ids, *embed_, x, stream);
        ladder::ladder_dump(x, "embed", -1, stream);
        ladder::ladder_dump_raw(x, "bisect_embed", -1, stream);
        for (std::size_t layer = 0; layer < parameters_.text.layers.size(); ++layer) {
            const auto& block = parameters_.text.layers[layer];
            const bool full   = config_.layer_types[layer] == MixerKind::FullAttention;
            const auto compact = dimension(config_.compact_layer_indices[layer]);
            try {
                if (layer <= 1) {
                    ladder::ladder_dump_raw(x, layer == 0 ? "bisect_b0_in" : "bisect_b1_in",
                                            static_cast<int>(layer), stream);
                }
                {
                    auto scope = work_.scope();
                    if (full) {
                        attn_mix(block, x, compact, Phase::Verify);
                    } else {
                        // GDN rows: one Prefill-phase update per prefill row
                        // over its slice with its persistent slot, then one
                        // width-1 Verify batch over the decode suffix.
                        if (n_pref > 0) {
                            ScopedValue<std::int32_t> batch_off(active_sequence_batch_, 0);
                            ScopedValue<std::int32_t> width_off(active_sequence_width_, 0);
                            ScopedValue<const Tensor*> src_off(active_linear_state_source_slots_,
                                                               nullptr);
                            ScopedValue<const Tensor*> dst_off(
                                active_linear_state_destination_slots_, nullptr);
                            for (std::size_t r = 0; r < n_pref; ++r) {
                                const std::int32_t begin =
                                    static_cast<std::int32_t>(host_seq_offsets[r]);
                                const std::int32_t span =
                                    static_cast<std::int32_t>(host_seq_offsets[r + 1]) - begin;
                                if (span <= 0) {
                                    throw std::logic_error(
                                        "forward_serve_step ragged row is empty");
                                }
                                set_linear_state_slots(row_slots[r], row_slots[r]);
                                Tensor xs = x.slice(1, begin, span);
                                gdn_mix(block, xs, compact, Phase::Prefill);
                            }
                        }
                        if (ndec > 0) {
                            ScopedValue<const Tensor*> src_on(active_linear_state_source_slots_,
                                                              &tensors.decode_source_slots);
                            ScopedValue<const Tensor*> dst_on(
                                active_linear_state_destination_slots_,
                                &tensors.decode_destination_slots);
                            ScopedValue<std::int32_t> batch_on(active_sequence_batch_, ndec);
                            ScopedValue<std::int32_t> width_on(active_sequence_width_, 1);
                            Tensor xd = x.slice(1, prefill_tokens, ndec);
                            gdn_mix(block, xd, compact, Phase::Verify);
                        }
                    }
                }
                if (layer <= 1) {
                    ladder::ladder_dump_raw(x, layer == 0 ? "bisect_b0_attn" : "bisect_b1_attn",
                                            static_cast<int>(layer), stream);
                }
                {
                    auto scope = work_.scope();
                    mlp_tail(block, x, Phase::Verify,
                             next_projection_hints(static_cast<int>(layer)));
                    ladder::ladder_dump(x, full ? "layer-full" : "layer-gdn",
                                        static_cast<int>(layer), stream);
                    if (layer <= 1) {
                        ladder::ladder_dump_raw(x, layer == 0 ? "bisect_b0_out" : "bisect_b1_out",
                                                static_cast<int>(layer), stream);
                    }
                }
            } catch (const std::exception& error) {
                throw std::runtime_error("text/serve/layers/" + std::to_string(layer) +
                                         " columns=" + std::to_string(T) + ": " + error.what());
            }
        }
        ops::rmsnorm(x, *final_norm_, config_.rms_norm_eps, true, tensors.hidden, stream);
        ladder::ladder_dump(tensors.hidden, "final_norm", -1, stream);
        ladder::ladder_dump_raw(tensors.hidden, "bisect_final_norm", -1, stream);
    }
    clear_ragged_batch();
    auto decoded = sample_decode_rows(tensors.hidden, plan, batch, stream);
    work_.reset();
    return decoded;
}

Tensor TextContext::embed_serve_input(ServeStepTensors& tensors) {
    const std::int32_t T = tensors.ids.ne[0];
    require_tensor_shape(tensors.ids, DType::I32, {T}, "serve embed ids");
    cudaStream_t stream = ctx_.stream;
    work_.reset();
    Tensor x = work_.alloc(DType::BF16, {dimension(config_.hidden_size), T});
    ops::embedding(tensors.ids, *embed_, x, stream);
    return x;
}

void TextContext::forward_serve_decode_layers(
    const batch::StepPlan& plan, const batch::RaggedBatch& batch,
    const batch::DeviceRaggedBatch& ragged, const std::uint32_t* host_seq_offsets,
    const std::int32_t* row_slots, ServeStepTensors& tensors, Tensor& x,
    ops::CausalAttentionExecutionEnvelope envelope) {
    const std::size_t n_dec    = plan.decode_seq_ids.size();
    const std::size_t n_pref   = plan.prefill.size();
    const std::size_t num_seqs = batch.num_seqs();
    if (num_seqs == 0 || n_pref != 0 || n_dec == 0 || n_pref + n_dec != num_seqs) {
        throw std::logic_error("forward_serve_decode_layers needs decode-only plans");
    }
    const std::int32_t T    = static_cast<std::int32_t>(batch.tokens.size());
    const std::int32_t nseq = static_cast<std::int32_t>(num_seqs);
    const std::int32_t ndec = static_cast<std::int32_t>(n_dec);
    require_tensor_shape(tensors.cache_positions, DType::I32, {T}, "serve layers cache positions");
    require_tensor_shape(tensors.rope_positions, DType::I32, {T}, "serve layers RoPE positions");
    require_tensor_shape(tensors.kv_table_rows, DType::I32, {nseq}, "serve layers KV rows");
    require_tensor_shape(tensors.decode_source_slots, DType::I32, {ndec},
                         "serve layers decode source slots");
    require_tensor_shape(tensors.decode_destination_slots, DType::I32, {ndec},
                         "serve layers decode destination slots");
    require_tensor_shape(tensors.hidden, DType::BF16, {dimension(config_.hidden_size), T},
                         "serve layers hidden");
    require_tensor_shape(x, DType::BF16, {dimension(config_.hidden_size), T},
                         "serve layers embed output");
    if (host_seq_offsets == nullptr || row_slots == nullptr) {
        throw std::invalid_argument("forward_serve_decode_layers requires host offsets and row slots");
    }
    (void)row_slots;  // decode rows ride the src/dst slot bindings, not per-row slots
    if (config_.layer_types.size() != parameters_.text.layers.size()) {
        throw std::logic_error("forward_serve_decode_layers layer inventory is inconsistent");
    }
    const std::int32_t prefill_tokens = static_cast<std::int32_t>(host_seq_offsets[0]);

    cudaStream_t stream = ctx_.stream;
    set_ragged_batch(&ragged, host_seq_offsets);
    {
        ScopedPositions cache_binding(active_cache_positions_, tensors.cache_positions);
        ScopedPositions rope_binding(active_rope_positions_, tensors.rope_positions);
        ScopedEnvelope envelope_binding(active_causal_attention_envelope_, envelope);
        ScopedValue<const Tensor*> kv_binding(active_kv_table_rows_, &tensors.kv_table_rows);
        ScopedValue<const Tensor*> src_on(active_linear_state_source_slots_,
                                          &tensors.decode_source_slots);
        ScopedValue<const Tensor*> dst_on(active_linear_state_destination_slots_,
                                          &tensors.decode_destination_slots);
        ScopedValue<std::int32_t> batch_on(active_sequence_batch_, ndec);
        ScopedValue<std::int32_t> width_on(active_sequence_width_, 1);
        ladder::ladder_dump(x, "embed", -1, stream);
        ladder::ladder_dump_raw(x, "bisect_embed", -1, stream);
        for (std::size_t layer = 0; layer < parameters_.text.layers.size(); ++layer) {
            const auto& block = parameters_.text.layers[layer];
            const bool full   = config_.layer_types[layer] == MixerKind::FullAttention;
            const auto compact = dimension(config_.compact_layer_indices[layer]);
            try {
                if (layer <= 1) {
                    ladder::ladder_dump_raw(x, layer == 0 ? "bisect_b0_in" : "bisect_b1_in",
                                            static_cast<int>(layer), stream);
                }
                {
                    auto scope = work_.scope();
                    if (full) {
                        attn_mix(block, x, compact, Phase::Verify);
                    } else {
                        Tensor xd = x.slice(1, prefill_tokens, ndec);
                        gdn_mix(block, xd, compact, Phase::Verify);
                    }
                }
                if (layer <= 1) {
                    ladder::ladder_dump_raw(x, layer == 0 ? "bisect_b0_attn" : "bisect_b1_attn",
                                            static_cast<int>(layer), stream);
                }
                {
                    auto scope = work_.scope();
                    mlp_tail(block, x, Phase::Verify,
                             next_projection_hints(static_cast<int>(layer)));
                    ladder::ladder_dump(x, full ? "layer-full" : "layer-gdn",
                                        static_cast<int>(layer), stream);
                    if (layer <= 1) {
                        ladder::ladder_dump_raw(x, layer == 0 ? "bisect_b0_out" : "bisect_b1_out",
                                                static_cast<int>(layer), stream);
                    }
                }
            } catch (const std::exception& error) {
                throw std::runtime_error("text/serve/layers-decode/" + std::to_string(layer) +
                                         " columns=" + std::to_string(T) + ": " + error.what());
            }
        }
        ops::rmsnorm(x, *final_norm_, config_.rms_norm_eps, true, tensors.hidden, stream);
        ladder::ladder_dump(tensors.hidden, "final_norm", -1, stream);
        ladder::ladder_dump_raw(tensors.hidden, "bisect_final_norm", -1, stream);
    }
    clear_ragged_batch();
    work_.reset();
}

template <class Tap>
void TextContext::target_verify_batch_impl(const Tensor& ids, const Tensor& cache_positions,
                                           const Tensor& rope_positions,
                                           const Tensor& valid_columns, const Tensor& kv_table_rows,
                                           const Tensor& linear_state_source_slots,
                                           const Tensor* linear_state_destination_slots,
                                           const Tensor* linear_state_column_slots,
                                           ops::CausalAttentionExecutionEnvelope envelope,
                                           Tensor& hidden, Tensor& logits, Tensor& target_tokens,
                                           Tap& tap) {
    const std::int32_t width = ids.ne[0];
    const std::int32_t batch = ids.ne[1];
    if (width <= 0 || width > static_cast<std::int32_t>(kDFlashDecodeMaximumWidth) || batch <= 0 ||
        batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("target verify batch shape is outside the supported domain");
    }
    const std::int32_t columns = width * batch;
    require_tensor_shape(ids, DType::I32, {width, batch}, "target verify batch ids");
    require_tensor_shape(cache_positions, DType::I32, {width, batch},
                         "target verify batch cache positions");
    require_tensor_shape(rope_positions, DType::I32, {width, batch},
                         "target verify batch RoPE positions");
    require_tensor_shape(valid_columns, DType::I32, {batch}, "target verify batch valid columns");
    require_tensor_shape(kv_table_rows, DType::I32, {batch}, "target verify batch KV rows");
    require_tensor_shape(linear_state_source_slots, DType::I32, {batch},
                         "target verify batch Linear Attention slots");
    require_tensor_shape(hidden, DType::BF16, {dimension(config_.hidden_size), width, batch},
                         "target verify batch hidden");
    require_tensor_shape(logits, DType::BF16, {dimension(config_.vocab_size), width, batch},
                         "target verify batch logits");
    require_tensor_shape(target_tokens, DType::I32, {width, batch}, "target verify batch tokens");

    cudaStream_t stream = ctx_.stream;
    work_.reset();
    {
        ScopedPositions cache_binding(active_cache_positions_, cache_positions);
        ScopedPositions rope_binding(active_rope_positions_, rope_positions);
        ScopedEnvelope envelope_binding(active_causal_attention_envelope_, envelope);
        ScopedValue<const Tensor*> kv_binding(active_kv_table_rows_, &kv_table_rows);
        ScopedValue<const Tensor*> state_binding(active_linear_state_source_slots_,
                                                 &linear_state_source_slots);
        ScopedValue<const Tensor*> destination_binding(active_linear_state_destination_slots_,
                                                       linear_state_destination_slots);
        ScopedValue<const Tensor*> column_binding(active_linear_state_column_slots_,
                                                  linear_state_column_slots);
        ScopedValue<const Tensor*> valid_binding(active_valid_columns_, &valid_columns);
        ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
        ScopedValue<std::int32_t> width_binding(active_sequence_width_, width);

        Tensor x        = work_.alloc(DType::BF16, {dimension(config_.hidden_size), columns});
        Tensor flat_ids = ids.view({columns});
        ops::embedding(flat_ids, *embed_, x, stream);
        if constexpr (Tap::enabled) { tap.begin(x); }
        run_layers(x, Phase::Verify, tap);
        if constexpr (requires { tap.capture_positions(cache_positions, stream); }) {
            tap.capture_positions(cache_positions, stream);
        }
        Tensor flat_hidden = hidden.view({dimension(config_.hidden_size), columns});
        Tensor flat_logits = logits.view({dimension(config_.vocab_size), columns});
        Tensor flat_tokens = target_tokens.view({columns});
        ops::rmsnorm(x, *final_norm_, config_.rms_norm_eps, true, flat_hidden, stream);
        project(flat_hidden, *lm_head_, flat_logits, work_, stream);
        ops::argmax(flat_logits, flat_tokens,
                    dimension(parameters_.model.resources().public_token_count), stream);
    }
    work_.reset();
}

void TextContext::target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                                      const Tensor& rope_positions, const Tensor& valid_columns,
                                      const Tensor& kv_table_rows,
                                      const Tensor& linear_state_source_slots,
                                      ops::CausalAttentionExecutionEnvelope envelope,
                                      Tensor& hidden, Tensor& logits, Tensor& target_tokens) {
    NullTap tap;
    target_verify_batch_impl(ids, cache_positions, rope_positions, valid_columns, kv_table_rows,
                             linear_state_source_slots, nullptr, nullptr, envelope, hidden, logits,
                             target_tokens, tap);
}

void TextContext::target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                                      const Tensor& rope_positions, const Tensor& valid_columns,
                                      const Tensor& kv_table_rows,
                                      const Tensor& linear_state_source_slots,
                                      ops::CausalAttentionExecutionEnvelope envelope,
                                      Tensor& hidden, Tensor& logits, Tensor& target_tokens,
                                      DFlashFeatureSink& sink) {
    target_verify_batch_impl(ids, cache_positions, rope_positions, valid_columns, kv_table_rows,
                             linear_state_source_slots, nullptr, nullptr, envelope, hidden, logits,
                             target_tokens, sink);
}

// Lane B: snapshot verify with explicit destination slot (serve MTP-3 batched
// verify, width 4). No feature sink on this arm; the record path is untouched.
void TextContext::target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                                      const Tensor& rope_positions, const Tensor& valid_columns,
                                      const Tensor& kv_table_rows,
                                      const Tensor& linear_state_source_slots,
                                      const Tensor& linear_state_destination_slots,
                                      ops::CausalAttentionExecutionEnvelope envelope,
                                      Tensor& hidden, Tensor& logits, Tensor& target_tokens,
                                      const Tensor* linear_state_column_slots) {
    NullTap tap;
    target_verify_batch_impl(ids, cache_positions, rope_positions, valid_columns, kv_table_rows,
                             linear_state_source_slots, &linear_state_destination_slots,
                             linear_state_column_slots, envelope,
                             hidden, logits, target_tokens, tap);
}

void TextContext::mtp_forward_decode_batch(const Tensor& ids, const Tensor& hidden,
                                           const Tensor& cache_positions,
                                           const Tensor& rope_positions,
                                           const Tensor& valid_columns, const Tensor& kv_table_rows,
                                           ops::CausalAttentionExecutionEnvelope envelope,
                                           Tensor& mtp_hidden) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    const std::int32_t width = ids.ne[0];
    const std::int32_t batch = ids.ne[1];
    if (width <= 0 || width > static_cast<std::int32_t>(kMaximumMtpDraftTokens + 1) || batch <= 0 ||
        batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("MTP decode batch shape is outside the supported domain");
    }
    require_tensor_shape(ids, DType::I32, {width, batch}, "MTP decode batch ids");
    require_tensor_shape(hidden, DType::BF16, {dimension(config_.hidden_size), width, batch},
                         "MTP decode batch target hidden");
    require_tensor_shape(cache_positions, DType::I32, {width, batch},
                         "MTP decode batch cache positions");
    require_tensor_shape(rope_positions, DType::I32, {width, batch},
                         "MTP decode batch RoPE positions");
    require_tensor_shape(valid_columns, DType::I32, {batch}, "MTP decode batch valid columns");
    require_tensor_shape(kv_table_rows, DType::I32, {batch}, "MTP decode batch KV rows");
    require_tensor_shape(mtp_hidden, DType::BF16, {dimension(config_.hidden_size), width, batch},
                         "MTP decode batch hidden");

    ScopedValue<const Tensor*> backend_binding(active_backend_kv_table_rows_, &kv_table_rows);
    ScopedValue<const Tensor*> valid_binding(active_valid_columns_, &valid_columns);
    ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
    ScopedValue<std::int32_t> width_binding(active_sequence_width_, width);
    mtp_forward_core(ids, hidden, cache_positions, rope_positions, envelope, mtp_hidden, nullptr);
}

void TextContext::mtp_propose_batch(const Tensor& hidden, Tensor& logits, Tensor& draft_tokens) {
    const std::int32_t batch = hidden.ne[1];
    require_tensor_shape(hidden, DType::BF16, {dimension(config_.hidden_size), batch},
                         "MTP proposal batch hidden");
    require_tensor_shape(logits, DType::BF16, {dimension(config_.vocab_size), batch},
                         "MTP proposal batch logits");
    require_tensor_shape(draft_tokens, DType::I32, {batch}, "MTP proposal batch tokens");
    proposal_argmax(hidden, logits, draft_tokens);
}

void TextContext::attn_mix(const BlockParameters& w, Tensor& x, int fidx, Phase ph) {
    const auto& p  = std::get<AttentionParameters>(w.mixer);
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];
    if (active_causal_attention_envelope_ == nullptr) {
        throw std::logic_error("Text GQA execution envelope is not set");
    }

    const auto projection = workspace::text_attention_projection(work_, config_, T);
    Tensor h              = projection.hidden;
    ops::rmsnorm(x, w.input_norm, config_.rms_norm_eps, true, h, s);

    Tensor q         = projection.query.view({dimension(config_.attention->head_dim),
                                              dimension(config_.attention->num_attention_heads), T});
    Tensor gate      = projection.gate.view({dimension(config_.attention->head_dim),
                                             dimension(config_.attention->num_attention_heads), T});
    Tensor k         = projection.key.view({dimension(config_.attention->head_dim),
                                            dimension(config_.attention->num_key_value_heads), T});
    Tensor v         = projection.value.view({dimension(config_.attention->head_dim),
                                              dimension(config_.attention->num_key_value_heads), T});
    Tensor q_flat    = q.view({dimension(config_.attention->query_width()), T});
    Tensor gate_flat = gate.view({dimension(config_.attention->query_width()), T});
    Tensor k_flat    = k.view({dimension(config_.attention->key_width()), T});
    Tensor v_flat    = v.view({dimension(config_.attention->key_width()), T});
    // EXL3 fused-QKV scatter (mixed-step correctness): the single-parent
    // EXL3 wrapper materializes a column-major [N,T] temp but copies row
    // ranges out of it as if row-major, which is exact only at T==1. For
    // the fused EXL3 single parent, dispatch the GEMM here and scatter the
    // temp column-wise (2D D2D), exact at every T. All other parents keep
    // the shared projection path unchanged.
    if (const auto* fused_single = std::get_if<LinearParameters>(&p.projection)) {
        if (fused_single->weight.qtype == QType::EXL3 &&
            fused_single->weight.n ==
                2 * dimension(config_.attention->query_width()) +
                    2 * dimension(config_.attention->key_width()) &&
            fused_single->weight.k == h.ne[0]) {
            const std::int32_t qr = dimension(config_.attention->query_width());
            const std::int32_t kr = dimension(config_.attention->key_width());
            const std::int32_t nr = 2 * qr + 2 * kr;
            auto proj_scope = work_.scope();
            Tensor full = work_.alloc(DType::BF16, {nr, T});
            ops::detail::exl3_dispatch(h, fused_single->weight, full, fused_single->policy,
                                       &work_, s);
            // Fused full-attention qkv scatter: single owner
            // ops::detail::scatter_qg_heads (C1 consolidation). Guards stay.
            const std::int32_t hd = dimension(config_.attention->head_dim);
            const std::int32_t nh = dimension(config_.attention->num_attention_heads);
            if (hd <= 0 || nh <= 0 || qr != hd * nh) {
                throw std::logic_error("text attn q/g interleave geometry mismatch");
            }
            ops::detail::scatter_qg_heads(
                q_flat.data, gate_flat.data, k_flat.data, v_flat.data,
                full.data, static_cast<std::size_t>(nr), qr, kr, nh, hd,
                static_cast<std::size_t>(T), s);
        } else {
            attention_projection(h, p, q_flat, gate_flat, k_flat, v_flat, work_, s);
        }
    } else {
        attention_projection(h, p, q_flat, gate_flat, k_flat, v_flat, work_, s);
    }

    const auto results = workspace::text_attention_results(work_, config_, T);
    Tensor qn =
        results.normalized_query.view({dimension(config_.attention->head_dim),
                                       dimension(config_.attention->num_attention_heads), T});
    Tensor kn = results.normalized_key.view({dimension(config_.attention->head_dim),
                                             dimension(config_.attention->num_key_value_heads), T});
    ops::rmsnorm(q, p.query_norm, config_.rms_norm_eps, true, qn, s);
    ops::rmsnorm(k, p.key_norm, config_.rms_norm_eps, true, kn, s);
    const Tensor& cache_positions =
        active_cache_positions_ != nullptr ? *active_cache_positions_ : io_.pos;
    const Tensor& rope_positions =
        active_rope_positions_ != nullptr ? *active_rope_positions_ : io_.rope_pos;
    Tensor rope_for_op = active_sequence_batch_ != 0 ? rope_positions.view({T}) : rope_positions;
    text_rope(rope_for_op, *config_.rope_parameters, qn, kn, s);

    Tensor a = results.attention.view({dimension(config_.attention->head_dim),
                                       dimension(config_.attention->num_attention_heads), T});
    const Tensor& kv_table_rows =
        active_kv_table_rows_ != nullptr ? *active_kv_table_rows_ : io_.text_kv_table_row;
    if (active_ragged_batch_ != nullptr) {
        // E2 serve batch path (ragged): flat T live tokens with variable spans
        // per row. The uniform [W,B] reshape (width*batch==T) must NOT be
        // required here: mixed prefill+decode steps are never uniform.
        // kv_table_rows is device I32 [num_seqs] in ragged row order; the
        // compact device tables (block_tables + seq_offsets) arrive via
        // BatchDeviceBuffers, never via host tables (empty on this path).
        const batch::DeviceRaggedBatch& ragged = *active_ragged_batch_;
        if (ragged.block_tables == nullptr || ragged.seq_offsets == nullptr ||
            active_ragged_offsets_ == nullptr) {
            throw std::logic_error("Ragged serve attention requires device block_tables + "
                                   "seq_offsets; host-only fallback is not allowed in serve");
        }
        if (ragged.total_tokens != static_cast<std::uint32_t>(T)) {
            throw std::logic_error("Ragged serve batch binding does not match aggregate columns");
        }
        ops::causal_softmax_attention_ragged(
            qn, kn, v, cache_positions, kv_table_rows, ragged, active_ragged_offsets_,
            {dimension(config_.attention->head_dim),
             dimension(config_.attention->num_attention_heads),
             dimension(config_.attention->num_key_value_heads)},
            static_cast<float>(1.0 / std::sqrt(static_cast<double>(config_.attention->head_dim))),
            batch_text_kv_->batch_layer_view(fidx), *active_causal_attention_envelope_, work_, a,
            s);
    } else if (active_sequence_batch_ != 0) {
        // Legacy uniform [W,B] path: single-request CLI compat and MTP verify
        // (uniform draft width by construction). NOT the serve batch path.
        const std::int32_t width = active_sequence_width_;
        if (width <= 0 || width * active_sequence_batch_ != T) {
            throw std::logic_error("Text sequence batch binding does not match aggregate columns");
        }
        Tensor q_batch        = qn.view({dimension(config_.attention->head_dim),
                                         dimension(config_.attention->num_attention_heads), width,
                                         active_sequence_batch_});
        Tensor k_batch        = kn.view({dimension(config_.attention->head_dim),
                                         dimension(config_.attention->num_key_value_heads), width,
                                         active_sequence_batch_});
        Tensor v_batch        = v.view({dimension(config_.attention->head_dim),
                                        dimension(config_.attention->num_key_value_heads), width,
                                        active_sequence_batch_});
        Tensor a_batch        = a.view({dimension(config_.attention->head_dim),
                                        dimension(config_.attention->num_attention_heads), width,
                                        active_sequence_batch_});
        Tensor position_batch = cache_positions.view({width, active_sequence_batch_});
        const Tensor valid = active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
        ops::causal_softmax_attention(
            q_batch, k_batch, v_batch, position_batch, valid, kv_table_rows,
            {dimension(config_.attention->head_dim),
             dimension(config_.attention->num_attention_heads),
             dimension(config_.attention->num_key_value_heads)},
            static_cast<float>(1.0 / std::sqrt(static_cast<double>(config_.attention->head_dim))),
            batch_text_kv_->batch_layer_view(fidx), *active_causal_attention_envelope_, work_,
            a_batch, s);
    } else {
        // Legacy single-seq lane only (single-request CLI compat / prefill):
        // ragged == null with more than one bound KV row is a multi-seq batch
        // missing its ragged serve binding and must throw here, never silently
        // run single dense attention over flat T.
        if (kv_table_rows.ne[0] > 1) {
            throw std::logic_error("Null-ragged multi-seq batch cannot run single dense "
                                   "attention: missing ragged serve binding");
        }
        ops::causal_softmax_attention(
            qn, kn, v, cache_positions, Tensor{}, kv_table_rows,
            {dimension(config_.attention->head_dim),
             dimension(config_.attention->num_attention_heads),
             dimension(config_.attention->num_key_value_heads)},
            static_cast<float>(1.0 / std::sqrt(static_cast<double>(config_.attention->head_dim))),
            batch_text_kv_->batch_layer_view(fidx), *active_causal_attention_envelope_, work_, a,
            s);
    }
    ops::sigmoid_mul(gate, a, s);

    ops::linear_add(a.view({dimension(config_.attention->query_width()), T}), p.output.weight, x,
                    p.output.policy, work_, s);
}

void TextContext::gdn_mix(const BlockParameters& w, Tensor& x, int gidx, Phase ph) {
    const auto& p  = std::get<GdnParameters>(w.mixer);
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];

    const auto control = workspace::gdn_control(work_, config_, T);
    Tensor h           = control.hidden;
    Tensor g           = control.g;
    Tensor beta        = control.beta;
    gdn_norm_control(x, w.input_norm, config_.rms_norm_eps, p, h, g, beta, work_,
                     ctx_.execution_view());
    ladder::ladder_dump_raw(h, "bisect_postnorm", gidx, s);

    const auto projection = workspace::gdn_projection(work_, config_, T);
    Tensor z  = projection.output_gate.view({dimension(config_.gdn->linear_value_head_dim),
                                             dimension(config_.gdn->linear_num_value_heads), T});
    Tensor qc = projection.query;
    Tensor kc = projection.key;
    Tensor vc = projection.value;
    if (ph == Phase::Verify) {
        if (active_sequence_batch_ == 0 || active_linear_state_source_slots_ == nullptr) {
            throw std::logic_error(
                "Verify GDN requires an explicit sequence batch and state slots");
        }
        const std::int32_t width = active_sequence_width_;
        if (width <= 0 || width * active_sequence_batch_ != T) {
            throw std::logic_error("GDN sequence batch binding does not match aggregate columns");
        }
        // Row 20b layout-B table ({k+1}, batch==1, length==width): shared by
        // the per-column snapshot below and the recurrent chain after it.
        // Null = legacy single snapshot + ping/pong chain.
        const Tensor* coltab_verify = active_linear_state_column_slots_;
        if (coltab_verify != nullptr) {
            if (active_sequence_batch_ != 1 || coltab_verify->ne[0] != width) {
                throw std::logic_error(
                    "Row 20b column slots need batch==1 and table length == width");
            }
        }
        if (gdn_state_action_ == GdnStateAction::UpdateInPlace && width != 1) {
            // Lane B batched snapshot verify only: single row, width in
            // (1, 4] with an explicit destination slot (spare source, shadow
            // destination). Every other in-place caller is width-1; the
            // record path below keeps its own domain. Anything else throws.
            if (active_linear_state_destination_slots_ == nullptr || width <= 1 || width > 4 ||
                active_sequence_batch_ != 1) {
                throw std::logic_error("In-place batched GDN update requires width one");
            }
        }
        Tensor projection_input =
            h.view({dimension(config_.hidden_size), width, active_sequence_batch_});
        Tensor query_output =
            qc.view({dimension(config_.gdn->key_width()), width, active_sequence_batch_});
        Tensor key_output =
            kc.view({dimension(config_.gdn->key_width()), width, active_sequence_batch_});
        Tensor value_output =
            vc.view({dimension(config_.gdn->value_width()), width, active_sequence_batch_});
        Tensor gate_output =
            z.view({dimension(config_.gdn->value_width()), width, active_sequence_batch_});
        Tensor conv_states = state_.layer_view(static_cast<std::uint32_t>(gidx)).conv;
        const Tensor valid = active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
        if (gdn_state_action_ == GdnStateAction::RecordForReplay) {
            if (replay_records_ == nullptr) {
                throw std::logic_error("Replay-record GDN has no record storage");
            }
            GdnReplayRecordLayer records = replay_records_->layer(gidx, active_sequence_batch_);
            gdn_projection_record(projection_input, p, *config_.gdn, conv_states, valid,
                                  *active_linear_state_source_slots_, records.conv, query_output,
                                  key_output, value_output, gate_output, work_, s);
        } else if (coltab_verify != nullptr) {
            // Row 20b layout A: ONE width-4 snapshot. The kernel publishes
            // col c -> base+c internally (same kernel as legacy), so base =
            // t[0] lands every column in its own consecutive block slot.
            // q/k/v/z come out width-4 directly; no per-column slicing, no
            // weight re-reads. The recurrent chain below still walks the
            // block column by column (it needs per-column state).
            Tensor dst_base = coltab_verify->slice(0, 0, 1);
            gdn_projection_snapshot(projection_input, p, *config_.gdn, conv_states, valid,
                                    *active_linear_state_source_slots_, dst_base, query_output,
                                    key_output, value_output, gate_output, work_, s);
            g_coltab_conv_cols.fetch_add(1, std::memory_order_relaxed);
        } else {
            gdn_projection_snapshot(projection_input, p, *config_.gdn, conv_states, valid,
                                    *active_linear_state_source_slots_,
                                    *active_linear_state_destination_slots_, query_output,
                                    key_output, value_output, gate_output, work_, s);
        }
    } else {
        Tensor qkv    = workspace::gdn_prefill_conv(work_, config_, T);
        Tensor z_flat = z.view({dimension(config_.gdn->value_width()), T});
        gdn_projection(h, p, qkv, z_flat, work_, s);
        Tensor conv_state_in =
            state_.conv_slot(static_cast<std::uint32_t>(gidx), linear_state_source_slot_);
        Tensor conv_state_out =
            state_.conv_slot(static_cast<std::uint32_t>(gidx), linear_state_destination_slot_);
        ops::causal_conv1d_silu_split(qkv, p.convolution, conv_state_in, conv_state_out, qc, kc, vc,
                                      s);
    }

    Tensor q_recurrent = qc.view({dimension(config_.gdn->linear_key_head_dim),
                                  dimension(config_.gdn->linear_num_key_heads), T});
    Tensor k_recurrent = kc.view({dimension(config_.gdn->linear_key_head_dim),
                                  dimension(config_.gdn->linear_num_key_heads), T});

    Tensor vv = vc.view({dimension(config_.gdn->linear_value_head_dim),
                         dimension(config_.gdn->linear_num_value_heads), T});
    Tensor o  = workspace::gdn_recurrent_output(work_, config_, T)
                   .view({dimension(config_.gdn->linear_value_head_dim),
                          dimension(config_.gdn->linear_num_value_heads), T});
    if (ph == Phase::Verify) {
        Tensor recurrent_states  = state_.layer_view(static_cast<std::uint32_t>(gidx)).recurrent;
        const std::int32_t width = active_sequence_width_;
        Tensor q_batch           = q_recurrent.view({dimension(config_.gdn->linear_key_head_dim),
                                                     dimension(config_.gdn->linear_num_key_heads), width,
                                                     active_sequence_batch_});
        Tensor k_batch           = k_recurrent.view({dimension(config_.gdn->linear_key_head_dim),
                                                     dimension(config_.gdn->linear_num_key_heads), width,
                                                     active_sequence_batch_});
        Tensor v_batch           = vv.view({dimension(config_.gdn->linear_value_head_dim),
                                            dimension(config_.gdn->linear_num_value_heads), width,
                                            active_sequence_batch_});
        Tensor g_batch =
            g.view({dimension(config_.gdn->linear_num_value_heads), width, active_sequence_batch_});
        Tensor beta_batch = beta.view(
            {dimension(config_.gdn->linear_num_value_heads), width, active_sequence_batch_});
        Tensor out_batch =
            o.view({dimension(config_.gdn->linear_value_head_dim),
                    dimension(config_.gdn->linear_num_value_heads), width, active_sequence_batch_});
        const Tensor valid = active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
        if (gdn_state_action_ == GdnStateAction::RecordForReplay) {
            GdnReplayRecordLayer records = replay_records_->layer(gidx, active_sequence_batch_);
            ops::gated_delta_net_replay_record(
                q_batch, k_batch, v_batch, g_batch, beta_batch,
                static_cast<float>(
                    1.0 / std::sqrt(static_cast<double>(config_.gdn->linear_key_head_dim))),
                recurrent_states, valid, *active_linear_state_source_slots_, records.key,
                records.value, records.gate, out_batch, s);
        } else if (width == 1) {
            ops::gated_delta_net_batch_update(
                q_batch, k_batch, v_batch, g_batch, beta_batch,
                static_cast<float>(
                    1.0 / std::sqrt(static_cast<double>(config_.gdn->linear_key_head_dim))),
                /*normalize_qk=*/true, recurrent_states, *active_linear_state_source_slots_,
                *active_linear_state_destination_slots_, out_batch, s);
        } else {
            // Lane B width-4 snapshot (serve batched verify): the recurrent
            // kernel is width-1, so chain the window column by column,
            // alternating source/destination between the spare and shadow
            // slots. Column c reads the previous column's published state
            // and publishes the next, exactly like the serial width-1 rows
            // this replaces (final state lands in the source slot after an
            // even width; both slots are scratch). All other paths above.
            const Tensor& ping = *active_linear_state_source_slots_;
            const Tensor& pong = *active_linear_state_destination_slots_;
            // Row 20b layout B ({k+1} table): col c reads
            // (c==0 ? source : t[c-1]) and writes t[c]. Accept-a = t[a].
            // Validated here (this block is outside the snapshot section's
            // scope); null keeps the legacy ping/pong scratch behavior.
            const Tensor* coltab = active_linear_state_column_slots_;
            if (coltab != nullptr) {
                if (active_sequence_batch_ != 1 || coltab->ne[0] != width) {
                    throw std::logic_error(
                        "Row 20b column slots need batch==1 and table length == width");
                }
            }
            const float scale  = static_cast<float>(
                1.0 / std::sqrt(static_cast<double>(config_.gdn->linear_key_head_dim)));
            for (std::int32_t c = 0; c < width; ++c) {
                Tensor qc = q_batch.slice(2, c, 1);
                Tensor kc = k_batch.slice(2, c, 1);
                Tensor vc = v_batch.slice(2, c, 1);
                Tensor gc = g_batch.slice(1, c, 1);
                Tensor bc = beta_batch.slice(1, c, 1);
                Tensor oc = out_batch.slice(2, c, 1);
                // Layout B {k+1}: col c reads (c==0 ? ping : t[c-1]), writes
                // t[c]. Legacy: even/odd ping/pong alternation (scratch).
                Tensor t_scol;
                Tensor t_dcol;
                const Tensor* scol_p = nullptr;
                const Tensor* dcol_p = nullptr;
                if (coltab != nullptr) {
                    t_dcol = coltab->slice(0, c, 1);
                    if (c == 0) {
                        scol_p = &ping;
                    } else {
                        t_scol = coltab->slice(0, c - 1, 1);
                        scol_p = &t_scol;
                    }
                    dcol_p = &t_dcol;
                } else {
                    scol_p = &((c % 2 == 0) ? ping : pong);
                    dcol_p = &((c % 2 == 0) ? pong : ping);
                }
                const Tensor& scol = *scol_p;
                const Tensor& dcol = *dcol_p;
                ops::gated_delta_net_batch_update(qc, kc, vc, gc, bc, scale,
                                                  /*normalize_qk=*/true, recurrent_states, scol,
                                                  dcol, oc, s);
                if (coltab != nullptr) {
                    g_coltab_rec_cols.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    } else {
        Tensor recurrent_state_in =
            state_.recurrent_slot(static_cast<std::uint32_t>(gidx), linear_state_source_slot_);
        Tensor recurrent_state_out =
            state_.recurrent_slot(static_cast<std::uint32_t>(gidx), linear_state_destination_slot_);
        ops::gated_delta_net(
            q_recurrent, k_recurrent, vv, g, beta,
            static_cast<float>(1.0 /
                               std::sqrt(static_cast<double>(config_.gdn->linear_key_head_dim))),
            /*normalize_qk=*/true, work_, recurrent_state_in, recurrent_state_out, o, s);
    }

    Tensor on = workspace::gdn_normalized_output(work_, config_, T)
                    .view({dimension(config_.gdn->linear_value_head_dim),
                           dimension(config_.gdn->linear_num_value_heads), T});
    ops::gated_rmsnorm(o, p.norm, z, config_.rms_norm_eps, on, s);

    ops::linear_add(on.view({dimension(config_.gdn->value_width()), T}), p.output.weight, x,
                    p.output.policy, work_, s);
}

ops::SparseMoeHints TextContext::next_projection_hints(int layer) const {
    const auto next = static_cast<std::size_t>(layer) + 1;
    return next < parameters_.text.layers.size() ? parameters_.text.layers[next].projection_prefetch
                                                 : ops::SparseMoeHints{};
}

void TextContext::mlp_tail(const BlockParameters& weights, Tensor& x, Phase,
                           const ops::SparseMoeHints& hints) {
    Tensor h = workspace::post_mixer_hidden(work_, config_, x.ne[1]);
    ops::rmsnorm(x, weights.post_attention_norm, config_.rms_norm_eps, true, h, ctx_.stream);
    ffn(h, weights.ffn, x, hints, work_, ctx_.stream);
}

template <class Tap>
void TextContext::run_layers(Tensor& x, Phase ph, Tap& tap) {
    const bool prefill = ph == Phase::Prefill;
    for (std::size_t layer = 0; layer < parameters_.text.layers.size(); ++layer) {
        const auto& block  = parameters_.text.layers[layer];
        const bool full    = config_.layer_types[layer] == MixerKind::FullAttention;
        const auto compact = dimension(config_.compact_layer_indices[layer]);
        nvtx::ScopedRange layer_range(
            full ? (prefill ? nvtx::Name::PrefillLayerFull : nvtx::Name::VerifyLayerFull)
                 : (prefill ? nvtx::Name::PrefillLayerGdn : nvtx::Name::VerifyLayerGdn),
            full ? nvtx::Category::Attention : nvtx::Category::Gdn, layer);
        try {
            {
                nvtx::ScopedRange mixer_range(
                    full ? (prefill ? nvtx::Name::PrefillAttention : nvtx::Name::VerifyAttention)
                         : (prefill ? nvtx::Name::PrefillGdn : nvtx::Name::VerifyGdn),
                    full ? nvtx::Category::Attention : nvtx::Category::Gdn, layer);
                auto scope = work_.scope();
                if (full) {
                    attn_mix(block, x, compact, ph);
                } else {
                    gdn_mix(block, x, compact, ph);
                }
            }
            {
                nvtx::ScopedRange range(prefill ? nvtx::Name::PrefillPostMixer
                                                : nvtx::Name::VerifyPostMixer,
                                        nvtx::Category::PostMixer, layer);
                auto scope = work_.scope();
                mlp_tail(block, x, ph, next_projection_hints(static_cast<int>(layer)));
            }
            if constexpr (Tap::enabled) {
                tap.capture_layer(static_cast<int>(layer), x, ctx_.stream);
            }
        } catch (const std::exception& error) {
            throw std::runtime_error("text/layers/" + std::to_string(layer) +
                                     (prefill ? " prefill" : " verify") +
                                     " columns=" + std::to_string(x.ne[1]) + ": " + error.what());
        }
    }
}

void TextContext::run_layers(Tensor& x, Phase ph) {
    NullTap tap;
    run_layers(x, ph, tap);
}

template <class Tap>
PrefillChunkResult
TextContext::prefill_impl(std::span<const int> ids, const TextPrefill* text_prefill,
                          const MultimodalPrefill* multimodal, Tap& tap, bool finalize_at_end) {
    runtime::ExecutionTimingRecorder timing;
    if (ids.empty()) { throw std::invalid_argument("TextContext::prefill requires tokens"); }
    if (ids.size() > static_cast<std::size_t>((std::numeric_limits<std::int32_t>::max)())) {
        throw std::overflow_error("TextContext::prefill token count exceeds int32");
    }
    cudaStream_t s           = ctx_.stream;
    const int T              = static_cast<int>(ids.size());
    const int chunk          = static_cast<int>(prefill_chunk_);
    const std::uint32_t base = text_kv_base_;

    if (text_prefill != nullptr) {
        if (multimodal != nullptr || base != text_prefill->begin ||
            text_prefill->token_ids.size() < static_cast<std::size_t>(base) + ids.size()) {
            throw std::invalid_argument("text prefill chunk does not match its full prompt");
        }
    }
    if (multimodal != nullptr) {
        if (base != multimodal->begin ||
            multimodal->token_ids.size() < static_cast<std::size_t>(base) + ids.size()) {
            throw std::invalid_argument("multimodal prefill suffix does not match its cache base");
        }
        if (multimodal->positions.size() != 3 * multimodal->token_ids.size()) {
            throw std::invalid_argument("multimodal positions must have shape [3,T]");
        }
        if (multimodal->vision == nullptr) {
            throw std::invalid_argument("multimodal prefill requires a Vision session");
        }
        rope_delta_ = multimodal->rope_delta;
    } else if (text_kv_base_ == 0) {
        rope_delta_ = 0;
    }
    ops::set_i32_scalar(io_.rope_delta, rope_delta_, s);

    // Prefix-append prefill continues an existing cache: positions are absolute (start at the
    // resident length) and KV/GDN state is not reset. For a reset prefill base == 0.
    if (static_cast<std::uint64_t>(base) + static_cast<std::uint64_t>(T) >
        static_cast<std::uint64_t>((std::numeric_limits<std::int32_t>::max)())) {
        throw std::overflow_error("TextContext::prefill absolute position exceeds int32");
    }
    const int base_i = static_cast<int>(base);

    const std::int64_t base64    = static_cast<std::int64_t>(base);
    const std::int64_t split_abs = prefill_split_frontier_;
    const bool has_split = split_abs > base64 && split_abs <= base64 + static_cast<std::int64_t>(T);
    const int split_rel  = has_split ? static_cast<int>(split_abs - base64) : -1;
    const bool prepare_mtp_prompt = mtp_enabled() && io_.mtp.has_value();
    if (prepare_mtp_prompt &&
        mtp_proposal_extent_ > static_cast<std::uint32_t>(io_.mtp->draft_tokens.ne[0])) {
        throw std::logic_error("MTP proposal extent exceeds the configured draft window");
    }
    int t0 = 0;
    for (; t0 < T;) {
        int len = (std::min)(chunk, T - t0);
        if (split_rel > 0 && t0 < split_rel && t0 + len > split_rel) { len = split_rel - t0; }
        work_.reset();

        VisionChunk vision_chunk;
        const std::uint32_t prompt_t0 = base + static_cast<std::uint32_t>(t0);
        if (multimodal != nullptr) {
            if (multimodal->vision == nullptr) {
                throw std::logic_error("multimodal prefill has no Vision session");
            }
            vision_chunk =
                multimodal->vision->prepare_chunk(prompt_t0, static_cast<std::uint32_t>(len));
            len = vision_chunk.length;
        }
        const bool is_last = finalize_at_end && (t0 + len == T);
        nvtx::ScopedRange chunk_range(nvtx::Name::PrefillChunk, nvtx::Category::Prefill,
                                      static_cast<std::uint64_t>(len));

        {
            std::vector<std::int32_t> local_scatter_indices;
            std::int32_t visual_begin = 0;
            if (vision_chunk.control != nullptr) {
                const auto scatter =
                    std::span<const std::int32_t>(vision_chunk.control->scatter_indices);
                const auto begin = std::lower_bound(scatter.begin(), scatter.end(), prompt_t0);
                const auto end   = std::lower_bound(begin, scatter.end(), prompt_t0 + len);
                const auto count = static_cast<std::int32_t>(end - begin);
                visual_begin     = static_cast<std::int32_t>(begin - scatter.begin());
                local_scatter_indices.resize(static_cast<std::size_t>(count));
                for (std::int32_t i = 0; i < count; ++i) {
                    local_scatter_indices[static_cast<std::size_t>(i)] =
                        begin[i] - static_cast<std::int32_t>(prompt_t0);
                }
            }

            const std::int32_t rope_axes = multimodal != nullptr ? 3 : (rope_delta_ != 0 ? 1 : 0);
            const auto roots             = workspace::text_prefill_roots(
                work_, config_, len, rope_axes,
                static_cast<std::int32_t>(local_scatter_indices.size()));
            Tensor ids_device = roots.ids;
            copy_i32(ids.data() + t0, ids_device, s);

            Tensor positions = roots.positions;
            ops::fill_i32_positions(positions, base_i + t0, s);

            Tensor rope_positions = positions;
            std::vector<std::int32_t> rope_positions_host;
            if (multimodal != nullptr) {
                rope_positions = roots.rope_positions;
                rope_positions_host.resize(static_cast<std::size_t>(3) * len);
                const std::size_t prompt_tokens = multimodal->token_ids.size();
                for (int axis = 0; axis < 3; ++axis) {
                    const auto* src = multimodal->positions.data() +
                                      static_cast<std::size_t>(axis) * prompt_tokens + prompt_t0;
                    std::copy_n(src, len,
                                rope_positions_host.data() + static_cast<std::size_t>(axis) * len);
                }
                copy_i32(rope_positions_host.data(), rope_positions, s);
            } else if (rope_delta_ != 0) {
                rope_positions = roots.rope_positions;
                ops::offset_i32_positions(positions, io_.rope_delta, rope_positions, s);
            }
            ScopedPositions scoped_cache(active_cache_positions_, positions);
            ScopedPositions scoped_rope(active_rope_positions_, rope_positions);
            const auto visible = static_cast<std::uint32_t>(base_i + t0 + len);
            const ops::CausalAttentionExecutionEnvelope chunk_envelope{visible, visible};
            ScopedEnvelope scoped_envelope(active_causal_attention_envelope_, chunk_envelope);

            Tensor x = roots.residual;
            ops::embedding(ids_device, *embed_, x, s);
            if (!local_scatter_indices.empty()) {
                Tensor indices_device = roots.scatter_indices;
                copy_i32(local_scatter_indices.data(), indices_device, s);
                Tensor embeddings = vision_chunk.embeddings.slice(
                    1, visual_begin, static_cast<std::int32_t>(local_scatter_indices.size()));
                ops::scatter(embeddings, indices_device, x, s);
            }
            if constexpr (Tap::enabled) { tap.begin(x); }
            run_layers(x, Phase::Prefill, tap);
            if constexpr (requires { tap.capture_positions(positions, s); }) {
                tap.capture_positions(positions, s);
            }

            Tensor xf = prefill_hidden_.data != nullptr
                            ? matrix_window(prefill_hidden_, len)
                            : work_.alloc(DType::BF16, {dimension(config_.hidden_size), len});
            ops::rmsnorm(x, *final_norm_, config_.rms_norm_eps, true, xf, s);

            if (is_last) {
                Tensor last_xf = xf.slice(1, len - 1, 1);
                Tensor logits  = matrix_window(io_.logits, 1);
                project(last_xf, *lm_head_, logits, work_, s);
                // Set io_.pos to the bonus token's absolute position (base + T) before picking so
                // the sampler RNG is keyed by it (prefill purpose keeps it distinct from the first
                // decode step, which reuses the same io_.pos).
                ops::set_i32_scalar(io_.pos, base_i + T, s);
                ops::set_i32_scalar(io_.rope_pos, base_i + T + rope_delta_, s);
                if (sampling_config_ != nullptr) {
                    ops::sample(logits, io_.token,
                                dimension(parameters_.model.resources().public_token_count),
                                sampling_config_, io_.pos, ops::kSamplePurposePrefill, work_, s);
                } else {
                    ops::argmax(logits, io_.token,
                                dimension(parameters_.model.resources().public_token_count), s);
                }
            }

            if (prepare_mtp_prompt) {
                const std::uint32_t alignment_tokens =
                    multimodal != nullptr ? static_cast<std::uint32_t>(multimodal->token_ids.size())
                    : text_prefill != nullptr
                        ? static_cast<std::uint32_t>(text_prefill->token_ids.size())
                        : static_cast<std::uint32_t>(T);
                const std::uint32_t alignment_begin =
                    multimodal != nullptr || text_prefill != nullptr
                        ? prompt_t0
                        : static_cast<std::uint32_t>(t0);
                const qwen3_5::MtpAlignmentWindow mtp_window = qwen3_5::plan_mtp_alignment_window(
                    alignment_tokens, alignment_begin, static_cast<std::uint32_t>(len));
                const std::span<const int> alignment_ids =
                    multimodal != nullptr     ? multimodal->token_ids
                    : text_prefill != nullptr ? text_prefill->token_ids
                                              : ids;
                const int prompt_columns =
                    len - static_cast<int>(mtp_window.final_column_uses_generated_token);
                Tensor mtp_ids = work_.alloc(DType::I32, {len});
                if (prompt_columns != 0) {
                    Tensor prompt_mtp_ids = mtp_ids.slice(0, 0, prompt_columns);
                    copy_i32(alignment_ids.data() + mtp_window.shifted_embedding_begin,
                             prompt_mtp_ids, s);
                }
                if (mtp_window.final_column_uses_generated_token) {
                    Tensor generated_mtp_id = mtp_ids.slice(0, len - 1, 1);
                    CUDA_CHECK(cudaMemcpyAsync(generated_mtp_id.data, io_.token.data,
                                               sizeof(std::int32_t), cudaMemcpyDeviceToDevice, s));
                }

                Tensor mtp_input_embeddings;
                const Tensor* mtp_input_embeddings_ptr = nullptr;
                if (multimodal != nullptr) {
                    mtp_input_embeddings =
                        work_.alloc(DType::BF16, {dimension(config_.hidden_size), len});
                    ops::embedding(mtp_ids, *embed_, mtp_input_embeddings, s);
                    if (vision_chunk.control != nullptr) {
                        const qwen3_5::MtpVisualOverlap overlap = qwen3_5::shifted_visual_overlap(
                            vision_chunk.control->scatter_indices, alignment_tokens, mtp_window);
                        if (!overlap.empty()) {
                            Tensor shifted_indices = workspace::visual_scatter_indices(
                                work_, static_cast<std::int32_t>(overlap.size()));
                            qwen3_5::detail::scatter_shifted_visual_embeddings(
                                mtp_input_embeddings, vision_chunk.embeddings, overlap,
                                shifted_indices, s);
                        }
                    }
                    mtp_input_embeddings_ptr = &mtp_input_embeddings;
                }
                if (is_last && mtp_proposal_extent_ != 0) {
                    Tensor logits = matrix_window(io_.logits, 1);
                    Tensor draft0 = io_.mtp->draft_tokens.slice(0, 0, 1);
                    mtp_prefill_chunk(mtp_ids, xf, mtp_input_embeddings_ptr, positions,
                                      rope_positions, chunk_envelope, true, &io_.mtp->ar_hidden,
                                      &logits, &draft0);

                    Tensor ar_position = io_.mtp->position.slice(0, 0, 1);
                    ops::set_i32_scalar(ar_position, base_i + T, s);
                    for (int i = 1; i < static_cast<int>(mtp_proposal_extent_); ++i) {
                        Tensor prev_token = io_.mtp->draft_tokens.slice(0, i - 1, 1);
                        Tensor next_token = io_.mtp->draft_tokens.slice(0, i, 1);
                        Tensor next_hidden =
                            work_.alloc(DType::BF16, {dimension(config_.hidden_size), 1});
                        const auto ar_visible = static_cast<std::uint32_t>(base_i + T + i);
                        const ops::CausalAttentionExecutionEnvelope ar_envelope{ar_visible,
                                                                                ar_visible};
                        mtp_forward_ar_step(prev_token, io_.mtp->ar_hidden, ar_position,
                                            ar_envelope, next_hidden, logits, next_token);
                        CUDA_CHECK(cudaMemcpyAsync(io_.mtp->ar_hidden.data, next_hidden.data,
                                                   io_.mtp->ar_hidden.bytes(),
                                                   cudaMemcpyDeviceToDevice, s));
                        ops::increment_i32_scalar(ar_position, s);
                    }
                } else {
                    mtp_prefill_chunk(mtp_ids, xf, mtp_input_embeddings_ptr, positions,
                                      rope_positions, chunk_envelope, false, nullptr, nullptr,
                                      nullptr);
                }
            }

            if (split_rel > 0 && t0 + len == split_rel &&
                rewrite_checkpoint_hidden_output_ != nullptr) {
                require_tensor_shape(*rewrite_checkpoint_hidden_output_, DType::BF16,
                                     {dimension(config_.hidden_size), 1},
                                     "rewrite checkpoint hidden output");
                const Tensor checkpoint_hidden = xf.slice(1, len - 1, 1);
                CUDA_CHECK(cudaMemcpyAsync(rewrite_checkpoint_hidden_output_->data,
                                           checkpoint_hidden.data, checkpoint_hidden.bytes(),
                                           cudaMemcpyDeviceToDevice, s));
            }
        }

        if constexpr (requires { tap.consume_prefill_chunk(len, false); }) {
            work_.reset();
            tap.consume_prefill_chunk(len, split_rel > 0 && t0 + len == split_rel);
        }

        t0 += len;
        break;
    }

    prefill_split_frontier_ = -1;

    timing.begin_wait();
    ctx_.synchronize();
    timing.end_wait();
    work_.reset();
    return PrefillChunkResult{.processed_tokens = static_cast<std::uint32_t>(t0),
                              .finalized        = finalize_at_end && t0 == T,
                              .timing           = timing.finish()};
}

PrefillChunkResult TextContext::prefill_chunk(std::span<const int> full_ids, std::uint32_t begin,
                                              std::uint32_t nominal_length, bool finalize_at_end) {
    if (begin >= full_ids.size() || nominal_length == 0 ||
        nominal_length > full_ids.size() - begin) {
        throw std::invalid_argument("text prefill chunk is outside the prompt");
    }
    const TextPrefill text_prefill{full_ids, begin};
    NullTap tap;
    return prefill_impl(full_ids.subspan(begin, nominal_length), &text_prefill, nullptr, tap,
                        finalize_at_end);
}

PrefillChunkResult TextContext::prefill_chunk(std::span<const int> full_ids, std::uint32_t begin,
                                              std::uint32_t nominal_length, bool finalize_at_end,
                                              DFlashFeatureSink& sink) {
    if (begin >= full_ids.size() || nominal_length == 0 ||
        nominal_length > full_ids.size() - begin) {
        throw std::invalid_argument("text prefill chunk is outside the prompt");
    }
    const TextPrefill text_prefill{full_ids, begin};
    return prefill_impl(full_ids.subspan(begin, nominal_length), &text_prefill, nullptr, sink,
                        finalize_at_end);
}

PrefillChunkResult TextContext::prefill_chunk(const qwen3_5::PreparedPromptData& input,
                                              std::uint32_t begin, std::uint32_t nominal_length,
                                              VisionPrefillSession& vision, bool finalize_at_end) {
    if (begin >= input.token_ids.size() || nominal_length == 0 ||
        nominal_length > input.token_ids.size() - begin) {
        throw std::invalid_argument("multimodal prefill chunk is outside the prompt");
    }
    const std::span<const int> tokens(input.token_ids);
    const MultimodalPrefill multimodal{tokens, input.positions, &vision, begin, input.rope_delta};
    NullTap tap;
    return prefill_impl(tokens.subspan(begin, nominal_length), nullptr, &multimodal, tap,
                        finalize_at_end);
}

PrefillChunkResult TextContext::prefill_chunk(const qwen3_5::PreparedPromptData& input,
                                              std::uint32_t begin, std::uint32_t nominal_length,
                                              VisionPrefillSession& vision, bool finalize_at_end,
                                              DFlashFeatureSink& sink) {
    if (begin >= input.token_ids.size() || nominal_length == 0 ||
        nominal_length > input.token_ids.size() - begin) {
        throw std::invalid_argument("multimodal prefill chunk is outside the prompt");
    }
    const std::span<const int> tokens(input.token_ids);
    const MultimodalPrefill multimodal{tokens, input.positions, &vision, begin, input.rope_delta};
    return prefill_impl(tokens.subspan(begin, nominal_length), nullptr, &multimodal, sink,
                        finalize_at_end);
}

} // namespace ninfer::models::qwen3_5::execution
