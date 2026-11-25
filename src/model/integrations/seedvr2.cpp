#include "model/diffusion/seedvr2.hpp"
#include "model_build_context.h"
#include "model_definition.h"

static bool build_seedvr2(const sd::model_builders::Context& ctx, const TokenizerConfig&, sd::model_builders::CoreRunners& runners) {
    if (!ctx.build_diffusion || ctx.build_conditioner)
        return false;
    const auto& weights = ctx.tensor_storage_map;
    const char* prefix  = weights.count("model.diffusion_model.vid_in.proj.weight") ? "model.diffusion_model" : "diffusion";
    runners.diffusion   = std::make_shared<SeedVR2::SeedVR2Runner>(ctx.backends.runtime_backend(SDBackendModule::DIFFUSION),
                                                                   weights, ctx.weight_manager, prefix);
    return true;
}

const std::vector<ModelDefinition>& seedvr2_definitions() {
    static const std::vector<ModelDefinition> definitions = [] {
        ModelDefinition definition;
        definition.version                        = VERSION_SEEDVR2;
        definition.name                           = "SeedVR2";
        definition.id                             = "seedvr2";
        definition.geometry                       = {2, 16, 3};
        definition.traits.is_dit                  = true;
        definition.sampling                       = {FLOW_PRED, 1.f};
        definition.sampling.default_sample_method = EULER_SAMPLE_METHOD;
        definition.build                          = build_seedvr2;
        return std::vector<ModelDefinition>{definition};
    }();
    return definitions;
}
