// exl3_program.cpp
// EXL3 serve-startup program binding for qwen3_5 (single slot). See
// exl3_program.h. Host-compiled; CUDA use is limited to cudaMalloc /
// cudaMemcpyAsync / stream sync for the BF16/FP32 resident tensors (norms,
// gating tables, token embedding). EXL3 linears upload through
// exl3_engine_load_dir (ops/linear/exl3) and are bound by logical name.
#include "runtime/engine/exl3_program.h"

#include "core/device.h"
#include "core/weight_view.h"
#include "models/load_options.h"
#include "models/qwen3_5/frontend/resources.h"
#include "models/qwen3_5/load/exl3_weights.h"
#include "models/qwen3_5/model.h"

#include <cuda_runtime.h>

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::exl3 {
struct Exl3EngineStore;
void exl3_engine_load_dir(const std::string& dir, cudaStream_t stream, Exl3EngineStore& out);
void exl3_engine_free_store(Exl3EngineStore& store) noexcept;
Exl3EngineStore& exl3_process_store() noexcept;
} // namespace ninfer::exl3

namespace ninfer::runtime {
namespace {

// ------ opaque EXL3 store entries (signatures must match exl3_bind.h) ------
// Opaque forward declarations: signatures must match exl3_bind.h exactly
// (full types are __CUDACC__-guarded there, so a host TU borrows the
// mangled ninfer::exl3:: names -- same pattern as generation_service.cpp).

// ------ float bit utilities (host side, F16 shard words -> BF16/FP32) ------
std::uint32_t half_to_float_bits(std::uint16_t h)
{
    const std::uint32_t sign = (std::uint32_t(h & 0x8000u) << 16);
    const std::uint32_t exp  = (h & 0x7C00u) >> 10;
    const std::uint32_t mant = h & 0x03FFu;
    if (exp == 0) {
        if (mant == 0) return sign; // signed zero
        // Subnormal: normalize.
        std::uint32_t m = mant;
        std::uint32_t e = 0;
        while ((m & 0x0400u) == 0) {
            m <<= 1;
            ++e;
        }
        m &= 0x03FFu;
        return sign | ((127 - 14 - e) << 23) | (m << 13);
    }
    if (exp == 0x1F) return sign | 0x7F800000u | (mant << 13); // inf/nan
    return sign | ((exp + 112) << 23) | (mant << 13);
}

std::uint16_t float_bits_to_bf16(std::uint32_t f)
{
    if ((f & 0x7F800000u) == 0x7F800000u) {
        // Preserve inf/nan payloads instead of rounding them.
        const std::uint32_t top = f >> 16;
        return static_cast<std::uint16_t>(top | ((f & 0x007FFFFFu) ? 0x0040u : 0u));
    }
    const std::uint32_t lsb = (f >> 16) & 1u;
    return static_cast<std::uint16_t>((f + 0x7FFFu + lsb) >> 16);
}

// ------ minimal safetensors reader (header JSON + verbatim tensor bytes) ---
struct ShardFile {
    std::string path;
    std::uint64_t data_base = 0; // file offset of the data section
    nlohmann::json header;
};

struct ShardTensor {
    std::string dtype; // "BF16" | "F16"
    std::vector<std::uint64_t> shape;
    std::vector<std::byte> bytes; // verbatim payload
};

class ShardReader {
public:
    explicit ShardReader(const std::string& dir) : dir_(dir)
    {
        std::ifstream index(dir + "/model.safetensors.index.json", std::ios::binary);
        if (!index) {
            throw std::runtime_error("EXL3 dir is missing model.safetensors.index.json: " + dir);
        }
        std::ostringstream text;
        text << index.rdbuf();
        const auto parsed = nlohmann::json::parse(text.str());
        for (const auto& [tensor, file] : parsed.at("weight_map").items()) {
            weight_map_[tensor] = file.get<std::string>();
        }
    }

    ShardTensor read(const std::string& tensor)
    {
        const auto it = weight_map_.find(tensor);
        if (it == weight_map_.end()) {
            throw std::invalid_argument("EXL3 checkpoint is missing tensor: " + tensor);
        }
        ShardFile& shard = open(it->second);
        const auto entry  = shard.header.find(tensor);
        if (entry == shard.header.end()) {
            throw std::runtime_error("safetensors header is missing tensor: " + tensor);
        }
        ShardTensor out;
        out.dtype = entry->at("dtype").get<std::string>();
        for (const auto& dim : entry->at("shape")) {
            out.shape.push_back(dim.get<std::uint64_t>());
        }
        const auto& offsets = entry->at("data_offsets");
        const std::uint64_t begin =
            static_cast<std::uint64_t>(offsets.at(0).get<std::uint64_t>());
        const std::uint64_t end = static_cast<std::uint64_t>(offsets.at(1).get<std::uint64_t>());
        if (end < begin) {
            throw std::runtime_error("safetensors offsets are inverted for: " + tensor);
        }
        std::ifstream file(shard.path, std::ios::binary);
        if (!file) { throw std::runtime_error("cannot reopen shard: " + shard.path); }
        file.seekg(static_cast<std::streamoff>(shard.data_base + begin));
        out.bytes.resize(static_cast<std::size_t>(end - begin));
        if (!out.bytes.empty()) {
            file.read(reinterpret_cast<char*>(out.bytes.data()),
                      static_cast<std::streamsize>(out.bytes.size()));
            if (!file) { throw std::runtime_error("short shard read for: " + tensor); }
        }
        return out;
    }

private:
    ShardFile& open(const std::string& file)
    {
        auto it = shards_.find(file);
        if (it != shards_.end()) return it->second;
        ShardFile shard;
        shard.path = dir_ + "/" + file;
        std::ifstream stream(shard.path, std::ios::binary);
        if (!stream) { throw std::runtime_error("cannot open shard: " + shard.path); }
        std::uint64_t header_len = 0;
        stream.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
        if (!stream) { throw std::runtime_error("cannot read shard header: " + shard.path); }
        std::string header_text(static_cast<std::size_t>(header_len), '\0');
        stream.read(header_text.data(), static_cast<std::streamsize>(header_len));
        if (!stream) { throw std::runtime_error("short shard header: " + shard.path); }
        shard.header    = nlohmann::json::parse(header_text);
        shard.data_base = sizeof(header_len) + header_len;
        return shards_.emplace(file, std::move(shard)).first->second;
    }

    std::string dir_;
    std::map<std::string, std::string> weight_map_;
    std::map<std::string, ShardFile> shards_;
};

// ------ backing owner: parents, device uploads, frontend blobs ------------
struct Backing final : Exl3Backing {
    ~Backing() override
    {
        for (void* blob : device_blobs) {
            if (blob != nullptr) (void)cudaFree(blob);
        }
    }

    std::vector<std::unique_ptr<WeightParent>> parents;
    std::vector<void*> device_blobs;
    std::vector<std::vector<std::byte>> host_blobs;
    std::vector<std::string> resources;
};

std::string read_text_file(const std::string& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream) { throw std::runtime_error("EXL3 dir is missing file: " + path); }
    std::ostringstream text;
    text << stream.rdbuf();
    return text.str();
}

} // namespace

Exl3ModelBundle build_exl3_model(const EngineOptions& options, DeviceContext& device)
{
    using models::qwen3_5::loading::exl3_find_payload;
    using Json = nlohmann::json;

    const std::string dir = options.artifact_path.string();
    if (options.enable_vision) {
        throw std::invalid_argument("EXL3 checkpoint dir serves the text backbone only "
                                    "(vision is not bound in single-slot)");
    }
    if (options.speculative.backend != SpeculativeBackend::None) {
        throw std::invalid_argument("EXL3 checkpoint dir serves without speculative decoding "
                                    "in single-slot");
    }

    // (1) Side-car store: packed trellis/suh/svh upload into the
    // process-lifetime store that materialization consumes by logical name.
    ninfer::exl3::exl3_engine_free_store(ninfer::exl3::exl3_process_store());
    ninfer::exl3::exl3_engine_load_dir(dir, device.transfer_stream, ninfer::exl3::exl3_process_store());

    // (2) Config: shape HF text_config into the artifact schema and reuse the
    // validated text parser (architectures/model_type take the .ninfer
    // packer spellings, which differ from the HF root spellings).
    const Json root     = Json::parse(read_text_file(dir + "/config.json"));
    const Json hf_text  = root.at("text_config");
    for (const char* moe : {"num_experts", "num_experts_per_tok", "moe_intermediate_size",
                            "shared_expert_intermediate_size"}) {
        if (hf_text.contains(moe)) {
            throw std::invalid_argument("EXL3 checkpoint dir has MoE fields (unsupported "
                                        "in single-slot)");
        }
    }
    Json shaped = Json::object();
    shaped["architectures"] = {"Qwen3_5ForCausalLM"};
    shaped["model_type"]    = "qwen3_5_text";
    for (const char* key :
         {"hidden_size", "vocab_size", "num_hidden_layers", "max_position_embeddings",
          "tie_word_embeddings", "rms_norm_eps", "layer_types", "num_attention_heads",
          "num_key_value_heads", "head_dim", "rope_parameters", "linear_num_key_heads",
          "linear_key_head_dim", "linear_num_value_heads", "linear_value_head_dim",
          "linear_conv_kernel_dim", "intermediate_size"}) {
        if (hf_text.contains(key)) shaped[key] = hf_text.at(key);
    }
    const models::qwen3_5::Config config = models::qwen3_5::parse_text_config(shaped, false);
    const auto& text                     = config.text;
    const std::uint64_t hidden           = text.hidden_size;
    const std::uint64_t intermediate =
        std::get<models::qwen3_5::DenseConfig>(text.ffn).intermediate_size;
    const std::uint64_t query_width = text.attention ? text.attention->query_width() : 0;
    const std::uint64_t key_width   = text.attention ? text.attention->key_width() : 0;

    auto backing = std::make_shared<Backing>();
    ShardReader shards(dir);

    // (3) Frontend resources: owned JSON blobs; views borrow them.
    models::qwen3_5::FrontendResources resources;
    backing->resources.push_back(read_text_file(dir + "/tokenizer.json"));
    backing->resources.push_back(read_text_file(dir + "/tokenizer_config.json"));
    backing->resources.push_back(read_text_file(dir + "/chat_template.jinja"));
    backing->resources.push_back(read_text_file(dir + "/generation_config.json"));
    resources.tokenizer_json         = backing->resources[0];
    resources.tokenizer_config_json  = backing->resources[1];
    resources.chat_template_jinja    = backing->resources[2];
    resources.generation_config_json = backing->resources[3];
    models::qwen3_5::parse_resources(resources, config);

    // (4) Weight binding helpers.
    std::vector<models::qwen3_5::BoundWeight> bound;
    bound.reserve(16 + text.num_hidden_layers * 24);
    backing->parents.reserve(16 + text.num_hidden_layers * 24);

    // EXL3 member binding. The process store holds FUSED logicals
    // (attention/qkv, gdn/qkv, mlp/gate_up); program members alias the
    // fused side-car payload while keeping their artifact-convention
    // member name and geometry (serve resolves the fused Single by store
    // name in parameters.cpp).
    auto add_exl3 = [&](std::string store, std::string member, std::uint64_t n,
                        std::uint64_t k, std::string input) {
        const void* payload = exl3_find_payload(store);
        if (payload == nullptr) {
            throw std::invalid_argument("EXL3 side-car is missing from the process store: " +
                                        store);
        }
        auto parent            = std::make_unique<WeightParent>();
        parent->geometry.format = QType::EXL3;
        parent->geometry.layout = QuantLayout::Contiguous;
        parent->geometry.shape  = {n, k};
        parent->geometry.elements = n * k;
        parent->data              = static_cast<const std::byte*>(payload);
        WeightView view;
        view.shape = {n, k};
        view.parts.push_back({parent.get(), 0, n * k});
        models::qwen3_5::WeightUse use;
        use.input  = std::move(input);
        use.policy = ops::LinearPolicy::A16Only;
        bound.push_back({member, {member}, std::move(view), {std::move(use)}});
        backing->parents.push_back(std::move(parent));
        return models::qwen3_5::WeightId{bound.size() - 1};
    };
    // Dense (unfused) EXL3 logical: store and member names coincide.
    auto add_single = [&](std::string logical, std::uint64_t n, std::uint64_t k,
                          std::string input) {
        return add_exl3(logical, logical, n, k, std::move(input));
    };
    // Fused-or-dense member: the binder fuses gate/up and q/k/v only when
    // the pair/triple is complete AND uniform (k, bits, codebook). A layer
    // whose members differ in K stays dense, so resolve the fused logical
    // first and fall back to the member logical.
    auto add_flex = [&](std::string fused, std::string member, std::uint64_t n,
                        std::uint64_t k, std::string input) {
        if (exl3_find_payload(fused) != nullptr) {
            return add_exl3(fused, member, n, k, std::move(input));
        }
        return add_single(member, n, k, std::move(input));
    };

    // Resident BF16/FP32 tensor from a shard tensor name, with optional
    // dtype conversion (F16 shard words -> BF16, BF16 -> FP32). View shape
    // may reframe the flat payload; the element count must match exactly.
    // The single rank-changing reframe is conv1d [C,1,K] -> [K,C]: the
    // depthwise kernels index weight tap-major (weight[k*C+c]) while the
    // shard stores channel-major [C][K], so the bytes must be TRANSPOSED
    // here. A flat memcpy deals each channel taps stolen from other
    // channels (only element (0,0) lands correctly).
    auto add_resident = [&](std::string logical, std::string tensor, QType format,
                            std::vector<std::uint64_t> view_shape, std::string input) {
        ShardTensor shard = shards.read(tensor);
        std::uint64_t count = 1;
        for (auto dim : shard.shape) count *= dim;
        std::uint64_t want = 1;
        for (auto dim : view_shape) want *= dim;
        if (count != want || count == 0) {
            throw std::invalid_argument("EXL3 resident tensor shape differs for: " + tensor);
        }
        const std::string& dtype = shard.dtype;
        std::size_t word = 0;
        if (format == QType::BF16) {
            if (dtype != "BF16" && dtype != "F16") {
                throw std::invalid_argument("EXL3 resident tensor needs BF16/F16 bytes: " + tensor);
            }
            word = 2;
        } else if (format == QType::FP32) {
            if (dtype != "BF16") {
                throw std::invalid_argument("EXL3 resident tensor needs BF16 bytes: " + tensor);
            }
            word = 4;
        } else {
            throw std::invalid_argument("EXL3 resident tensor needs BF16/FP32 format: " + logical);
        }
        backing->host_blobs.push_back({});
        std::vector<std::byte>& host = backing->host_blobs.back();
        host.resize(static_cast<std::size_t>(count * word));
        if (shard.shape.size() == 3) {
            // Conv reframe: [C,1,K] shard -> tap-major [K,C] view.
            if (shard.shape[1] != 1 || view_shape.size() != 2 ||
                view_shape[0] != shard.shape[2] || view_shape[1] != shard.shape[0]) {
                throw std::invalid_argument("EXL3 resident 3-D reframe must be [C,1,K] -> [K,C]: " +
                                            tensor);
            }
            if (format != QType::BF16 || dtype != "BF16") {
                throw std::invalid_argument("EXL3 resident 3-D reframe needs BF16 bytes: " + tensor);
            }
            const std::uint64_t channels = shard.shape[0];
            const std::uint64_t taps     = shard.shape[2];
            const auto* src = reinterpret_cast<const std::uint16_t*>(shard.bytes.data());
            auto* dst       = reinterpret_cast<std::uint16_t*>(host.data());
            for (std::uint64_t c = 0; c < channels; ++c) {
                for (std::uint64_t k = 0; k < taps; ++k) { dst[k * channels + c] = src[c * taps + k]; }
            }
        } else if (format == QType::BF16 && dtype == "BF16") {
            std::memcpy(host.data(), shard.bytes.data(), host.size());
        } else if (format == QType::BF16) {
            const auto* src = reinterpret_cast<const std::uint16_t*>(shard.bytes.data());
            auto* dst       = reinterpret_cast<std::uint16_t*>(host.data());
            for (std::uint64_t i = 0; i < count; ++i) {
                dst[i] = float_bits_to_bf16(half_to_float_bits(src[i]));
            }
        } else {
            // BF16 -> FP32 is exact: the bf16 word becomes the top half.
            const auto* src = reinterpret_cast<const std::uint16_t*>(shard.bytes.data());
            auto* dst       = reinterpret_cast<std::uint32_t*>(host.data());
            for (std::uint64_t i = 0; i < count; ++i) {
                dst[i] = static_cast<std::uint32_t>(src[i]) << 16;
            }
        }
        void* device_ptr = nullptr;
        CUDA_CHECK(cudaMalloc(&device_ptr, host.size()));
        backing->device_blobs.push_back(device_ptr);
        CUDA_CHECK(cudaMemcpyAsync(device_ptr, host.data(), host.size(), cudaMemcpyHostToDevice,
                                   device.transfer_stream));
        auto parent             = std::make_unique<WeightParent>();
        parent->geometry.format = format;
        parent->geometry.layout = QuantLayout::Contiguous;
        parent->geometry.shape  = view_shape;
        parent->geometry.elements = count;
        parent->geometry.bytes    = host.size();
        parent->data              = static_cast<const std::byte*>(device_ptr);
        WeightView view;
        view.shape = std::move(view_shape);
        view.parts.push_back({parent.get(), 0, count});
        models::qwen3_5::WeightUse use;
        use.input  = std::move(input);
        use.policy = ops::LinearPolicy::A16Only;
        bound.push_back({std::move(logical), {std::move(tensor)}, std::move(view),
                         {std::move(use)}});
        backing->parents.push_back(std::move(parent));
        return models::qwen3_5::WeightId{bound.size() - 1};
    };

    // (5) Text weights in bind_text order.
    using models::qwen3_5::WeightId;
    models::qwen3_5::ModelWeights weights;
    const std::string lm = "model.language_model.";
    weights.text.token_embedding = add_resident("text/token_embedding", lm + "embed_tokens.weight",
                                                QType::BF16, {text.vocab_size, hidden}, "");
    weights.text.output_head =
        add_single("text/output_head", text.vocab_size, hidden, "text/final_hidden");
    weights.text.final_norm =
        add_resident("text/final_norm", lm + "norm.weight", QType::BF16, {hidden}, "");
    weights.text.output_head_use = {weights.text.output_head, 0};

    std::uint64_t fused = 0;
    for (std::uint32_t i = 0; i < text.num_hidden_layers; ++i) {
        const std::string p  = "text/layers/" + std::to_string(i) + "/";
        const std::string hf = lm + "layers." + std::to_string(i) + ".";
        models::qwen3_5::BlockWeights block;
        block.input_norm = add_resident(p + "input_norm", hf + "input_layernorm.weight",
                                        QType::BF16, {hidden}, "");
        block.post_attention_norm =
            add_resident(p + "post_attention_norm", hf + "post_attention_layernorm.weight",
                         QType::BF16, {hidden}, "");
        if (text.layer_types[i] == models::qwen3_5::MixerKind::FullAttention) {
            // The checkpoint stores q+gate stacked as q_proj; the binder fuses
            // q/k/v into attention/qkv (groups=3). Members alias the whole
            // side-car; parameters.cpp resolves the fused Single by name.
            const std::string qkv_store = p + "attention/qkv";
            models::qwen3_5::AttentionWeights attn;
            attn.query = add_flex(qkv_store, p + "attention/query", query_width, hidden,
                                  p + "mixer_input");
            attn.key   = add_flex(qkv_store, p + "attention/key", key_width, hidden,
                                  p + "mixer_input");
            attn.gate  = add_flex(qkv_store, p + "attention/gate", query_width, hidden,
                                  p + "mixer_input");
            attn.value = add_flex(qkv_store, p + "attention/value", key_width, hidden,
                                  p + "mixer_input");
            attn.query_norm =
                add_resident(p + "attention/query_norm", hf + "self_attn.q_norm.weight",
                             QType::BF16, {text.attention->head_dim}, "");
            attn.key_norm =
                add_resident(p + "attention/key_norm", hf + "self_attn.k_norm.weight",
                             QType::BF16, {text.attention->head_dim}, "");
            attn.output = add_single(p + "attention/output", hidden, query_width,
                                     p + "attention/gated_output");
            block.mixer = std::move(attn);
            ++fused;
        } else {
            const auto& g     = text.gdn.value();
            const std::string gqv = p + "gdn/qkv";
            models::qwen3_5::GdnWeights gdn;
            gdn.query = add_flex(gqv, p + "gdn/query", g.key_width(), hidden,
                                 p + "mixer_input");
            gdn.key   = add_flex(gqv, p + "gdn/key", g.key_width(), hidden,
                                 p + "mixer_input");
            gdn.value = add_flex(gqv, p + "gdn/value", g.value_width(), hidden,
                                 p + "mixer_input");
            gdn.z     = add_single(p + "gdn/z", g.value_width(), hidden, p + "mixer_input");
            gdn.a_projection =
                add_resident(p + "gdn/a_projection", hf + "linear_attn.in_proj_a.weight",
                             QType::BF16, {g.linear_num_value_heads, hidden}, p + "mixer_input");
            gdn.b_projection =
                add_resident(p + "gdn/b_projection", hf + "linear_attn.in_proj_b.weight",
                             QType::BF16, {g.linear_num_value_heads, hidden}, p + "mixer_input");
            gdn.a_log =
                add_resident(p + "gdn/a_log", hf + "linear_attn.A_log", QType::FP32,
                             {g.linear_num_value_heads}, "");
            gdn.dt_bias =
                add_resident(p + "gdn/dt_bias", hf + "linear_attn.dt_bias", QType::FP32,
                             {g.linear_num_value_heads}, "");
            gdn.convolution = add_resident(p + "gdn/convolution", hf + "linear_attn.conv1d.weight",
                                           QType::BF16,
                                           {g.linear_conv_kernel_dim, g.conv_channels()}, "");
            gdn.norm = add_resident(p + "gdn/norm", hf + "linear_attn.norm.weight", QType::BF16,
                                    {g.linear_value_head_dim}, "");
            gdn.output = add_single(p + "gdn/output", hidden, g.value_width(),
                                    p + "gdn/gated_output");
            block.mixer = std::move(gdn);
        }
        models::qwen3_5::DenseWeights mlp;
        const std::string gu = p + "mlp/gate_up";
        mlp.gate = add_flex(gu, p + "mlp/gate", intermediate, hidden, p + "ffn_input");
        mlp.up   = add_flex(gu, p + "mlp/up", intermediate, hidden, p + "ffn_input");
        mlp.down = add_single(p + "mlp/down", hidden, intermediate, p + "mlp/product");
        block.ffn = std::move(mlp);
        ++fused;
        weights.text.layers.push_back(std::move(block));
    }

    // (6) Publish: uploads complete before the Model is visible.
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    std::cout << "exl3: bound " << bound.size() << " logical weights (" << fused
              << " fused) from " << dir << std::endl;

    models::qwen3_5::InstanceInfo info;
    info.name           = "exl3:" + options.artifact_path.filename().string();
    info.metadata_json  = "{}";
    info.provenance_json = "{}";

    Exl3ModelBundle bundle;
    bundle.model = models::qwen3_5::Model::create(config, models::load_options(options),
                                                 std::move(weights), std::move(bound),
                                                 std::move(resources), std::move(info),
                                                 artifact::MaterializedArtifact{});
    bundle.backing = std::move(backing);
    return bundle;
}

} // namespace ninfer::runtime
