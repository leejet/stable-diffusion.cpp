#include "model/diffusion/minimax_h3.hpp"
#include "conditioning/conditioner.hpp"
#include "model_build_context.h"
#include "model_definition.h"

static bool build_minimax_h3(const sd::model_builders::Context& ctx, const TokenizerConfig& tokenizers, sd::model_builders::CoreRunners& result) {
    const auto* sd_ctx_params      = &ctx.params;
    const auto& tensor_storage_map = ctx.tensor_storage_map;
    const auto version             = ctx.version;
    const auto& weight_manager     = ctx.weight_manager;
    if (ctx.build_conditioner) {
        result.conditioner = std::make_shared<LLMEmbedder>(ctx.backends.runtime_backend(SDBackendModule::TE),
                                                           tensor_storage_map,
                                                           version,
                                                           "",
                                                           true,
                                                           weight_manager,
                                                           tokenizers);
    }
    if (ctx.build_diffusion) {
        result.diffusion = std::make_shared<MiniMaxH3::MiniMaxH3Runner>(ctx.backends.runtime_backend(SDBackendModule::DIFFUSION),
                                                                        tensor_storage_map,
                                                                        "model.diffusion_model",
                                                                        weight_manager,
                                                                        sd_ctx_params->model_args.c_str());
    }
    return true;
}

const std::vector<ModelDefinition>& minimax_h3_definitions() {
    static const std::vector<ModelDefinition> definitions = [] {
        std::vector<ModelDefinition> result;
        {
            ModelDefinition definition;
            definition.version                          = VERSION_MINIMAX_H3;
            definition.geometry                         = {2, 24, 3};
            definition.name                             = "MiniMax-H3";
            definition.id                               = "minimax_h3";
            definition.traits.is_dit                    = true;
            definition.traits.supports_video_generation = true;
            definition.traits.vae_kind                  = ModelVAEKind::MINIMAX_H3;
            definition.traits.video_tae                 = true;
            definition.sampling                         = {H3_AV_FLOW_PRED, 12.f};
            definition.sampling.default_sample_method   = EULER_SAMPLE_METHOD;
            definition.sampling.default_scheduler       = DISCRETE_SCHEDULER;
            definition.build                            = build_minimax_h3;
            result.push_back(std::move(definition));
        }
        {
            ModelDefinition definition                  = result.front();
            definition.version                          = VERSION_MINIMAX_H3_IMAGE;
            definition.name                             = "MiniMax-H3-Image";
            definition.id                               = "minimax_h3_image";
            definition.traits.supports_video_generation = false;
            definition.sampling.default_flow_shift      = 1.f;
            result.push_back(std::move(definition));
        }
        {
            ModelDefinition definition                  = result.front();
            definition.version                          = VERSION_P_H3_IMAGE;
            definition.name                             = "P-H3-Image";
            definition.id                               = "p_h3_image";
            definition.traits.supports_video_generation = false;
            definition.sampling.default_flow_shift      = 1.f;
            result.push_back(std::move(definition));
        }
        return result;
    }();
    return definitions;
}
