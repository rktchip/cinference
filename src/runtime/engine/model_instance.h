#pragma once

#include "models/qwen3_5/model.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/program/runtime_types.h"
#include "runtime/engine/context_cache/context_cost.h"
#include "runtime/engine/kv_capacity.h"

#include <memory>

namespace ninfer::runtime {

[[nodiscard]] EngineOptions normalize_engine_options(EngineOptions options);

struct ModelInstance {
    using ModelContract = models::qwen3_5::RuntimeTypes;

    std::unique_ptr<models::qwen3_5::Model> model;
    const models::qwen3_5::execution::Parameters parameters;
    models::qwen3_5::Frontend frontend;
    KvCapacityResolution kv_capacity_resolution;
    const std::uint32_t capacity;
    std::unique_ptr<models::qwen3_5::Program> program;
    // EXL3 serve-startup backing (single slot): owns the WeightParents and
    // device uploads the Model's BoundWeight views borrow. Null on the
    // artifact path (the MaterializedArtifact inside Model owns them there).
    // Assigned by construct_model right after construction, before Parameters
    // or Program dereference the views.
    std::shared_ptr<void> exl3_backing;

    ModelInstance(std::unique_ptr<models::qwen3_5::Model> model, const EngineOptions& options);
    ~ModelInstance();
    ModelInstance(const ModelInstance&)            = delete;
    ModelInstance& operator=(const ModelInstance&) = delete;
};

struct ConstructedModel {
    std::unique_ptr<ModelInstance> instance;
    LoadSummary load;
    ContextMachineCostModel context_cost;
};

[[nodiscard]] ConstructedModel construct_model(const EngineOptions& options, DeviceContext& device);

} // namespace ninfer::runtime
