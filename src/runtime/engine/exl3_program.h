// exl3_program.h
// EXL3 serve-startup program binding for qwen3_5 (single slot).
//
// Builds a serving Model directly from an HF EXL3 checkpoint directory
// (config.json + tokenizer files + safetensors shards + the EXL3 side-car
// process store), with no .ninfer Reader. Text backbone only: vision, MTP,
// draft, and proposal weights are not bound; requesting them (vision or
// speculative serve options) fails fast here instead of crashing later.
//
// Ownership: the returned bundle keeps every WeightParent and device upload
// alive (BoundWeight views borrow them) plus the frontend JSON blobs
// (FrontendResources string_views borrow those). The caller (construct_model)
// must keep bundle.backing alive as long as bundle.model.
#pragma once

#include <memory>

namespace ninfer {

struct DeviceContext;
struct EngineOptions;

namespace models::qwen3_5 {
class Model;
}

namespace runtime {

struct Exl3Backing {
    virtual ~Exl3Backing() = default;
};

struct Exl3ModelBundle {
    std::unique_ptr<models::qwen3_5::Model> model;
    std::shared_ptr<Exl3Backing> backing; // owns parents, uploads, resources
};

// Loads the EXL3 side-car store from options.artifact_path (an EXL3
// checkpoint directory), binds the qwen3_5 text program against it, and
// returns the serving Model plus its backing owner. Throws
// std::invalid_argument on config/tokenizer mismatch and std::runtime_error
// on shard I/O or CUDA failure. Uploads on device.transfer_stream and
// synchronizes it before returning.
[[nodiscard]] Exl3ModelBundle build_exl3_model(const EngineOptions& options,
                                               DeviceContext& device);

} // namespace runtime
} // namespace ninfer
