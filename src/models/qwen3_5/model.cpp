#include "models/qwen3_5/model.h"

#include <stdexcept>
#include <utility>

namespace ninfer::models::qwen3_5 {

Model::Model(Config config, LoadOptions options, ModelWeights weights,
             std::vector<BoundWeight> bound, FrontendResources resources, InstanceInfo info,
             artifact::MaterializedArtifact backing)
    : backing_(std::move(backing)), config_(std::move(config)), options_(options),
      weights_(std::move(weights)), bound_(std::move(bound)), resources_(std::move(resources)),
      info_(std::move(info)) {}

Model::~Model() = default;

std::unique_ptr<Model> Model::create(Config config, LoadOptions options, ModelWeights weights,
                                     std::vector<BoundWeight> bound, FrontendResources resources,
                                     InstanceInfo info, artifact::MaterializedArtifact backing)
{
    return std::unique_ptr<Model>(new Model(std::move(config), options, std::move(weights),
                                            std::move(bound), std::move(resources), std::move(info),
                                            std::move(backing)));
}

ops::WeightInput Model::input(WeightUseId id) const {
    const auto& parameter = weight(id.parameter);
    const auto& use       = parameter.uses.at(id.use_index);
    return {parameter.view, use.policy, use.activation_input_divisor};
}

ops::WeightInput Model::input(WeightId id) const {
    if (weight(id).uses.size() != 1) {
        throw std::invalid_argument("weight input requires an explicit mathematical use");
    }
    return input(WeightUseId{id, 0});
}

} // namespace ninfer::models::qwen3_5
