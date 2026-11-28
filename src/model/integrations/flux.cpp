#include "model/diffusion/flux.hpp"
#include "conditioning/conditioner.hpp"
#include "model/integrations/flux.hpp"
#include "model_build_context.h"
#include "model_definition.h"

static bool build_flux(const sd::model_builders::Context& ctx, const TokenizerConfig& tokenizers, sd::model_builders::CoreRunners& result) {
    const auto* sd_ctx_params      = &ctx.params;
    const auto& tensor_storage_map = ctx.tensor_storage_map;
    const auto version             = ctx.version;
    const auto& weight_manager     = ctx.weight_manager;
    if (version == VERSION_CHROMA || version == VERSION_CHROMA_RADIANCE) {
        if (ctx.build_conditioner) {
            result.conditioner = std::make_shared<T5CLIPEmbedder>(ctx.backends.runtime_backend(SDBackendModule::TE),
                                                                  tensor_storage_map,
                                                                  false,
                                                                  1,
                                                                  false,
                                                                  weight_manager,
                                                                  sd_ctx_params->model_args.c_str());
        }
    } else if (version == VERSION_OVIS_IMAGE) {
        if (ctx.build_conditioner) {
            result.conditioner = std::make_shared<LLMEmbedder>(ctx.backends.runtime_backend(SDBackendModule::TE),
                                                               tensor_storage_map,
                                                               version,
                                                               "",
                                                               false,
                                                               weight_manager,
                                                               tokenizers);
        }
    } else {
        if (ctx.build_conditioner) {
            result.conditioner = std::make_shared<FluxCLIPEmbedder>(ctx.backends.runtime_backend(SDBackendModule::TE),
                                                                    tensor_storage_map,
                                                                    weight_manager,
                                                                    tokenizers);
        }
    }
    if (ctx.build_diffusion) {
        result.diffusion = std::make_shared<Flux::FluxRunner>(ctx.backends.runtime_backend(SDBackendModule::DIFFUSION),
                                                              tensor_storage_map,
                                                              "model.diffusion_model",
                                                              version,
                                                              weight_manager,
                                                              sd_ctx_params->model_args.c_str());
    }
    return true;
}

const std::vector<ModelDefinition>& flux_definitions() {
    static const std::vector<ModelDefinition> definitions = [] {
        std::vector<ModelDefinition> result;
        {
            ModelDefinition definition;
            definition.version                          = VERSION_FLUX;
            definition.geometry                         = {1, 16, 3};
            definition.name                             = "Flux";
            definition.id                               = "flux";
            definition.traits.is_dit                    = true;
            definition.traits.vae_kind                  = ModelVAEKind::FLUX;
            definition.sampling                         = {FLUX_FLOW_PRED, 1.0f};
            definition.sampling.default_sample_method   = EULER_SAMPLE_METHOD;
            definition.sampling.default_scheduler       = FLUX_SCHEDULER;
            definition.build                            = build_flux;
            definition.sampling.flow_shift_from_weights = flux_guidance_flow_shift;
            result.push_back(std::move(definition));
        }
        {
            ModelDefinition definition;
            definition.version                          = VERSION_FLUX_FILL;
            definition.geometry                         = {1, 16, 3};
            definition.name                             = "Flux Fill";
            definition.id                               = "flux_fill";
            definition.traits.is_dit                    = true;
            definition.traits.vae_kind                  = ModelVAEKind::FLUX;
            definition.traits.is_inpaint                = true;
            definition.sampling                         = {FLUX_FLOW_PRED, 1.0f};
            definition.sampling.default_sample_method   = EULER_SAMPLE_METHOD;
            definition.sampling.default_scheduler       = FLUX_SCHEDULER;
            definition.build                            = build_flux;
            definition.sampling.flow_shift_from_weights = flux_guidance_flow_shift;
            result.push_back(std::move(definition));
        }
        {
            ModelDefinition definition;
            definition.version                          = VERSION_FLUX_CONTROLS;
            definition.geometry                         = {1, 16, 3};
            definition.name                             = "Flux Control";
            definition.id                               = "flux_control";
            definition.traits.is_dit                    = true;
            definition.traits.vae_kind                  = ModelVAEKind::FLUX;
            definition.traits.is_control                = true;
            definition.sampling                         = {FLUX_FLOW_PRED, 1.0f};
            definition.sampling.default_sample_method   = EULER_SAMPLE_METHOD;
            definition.sampling.default_scheduler       = FLUX_SCHEDULER;
            definition.build                            = build_flux;
            definition.sampling.flow_shift_from_weights = flux_guidance_flow_shift;
            result.push_back(std::move(definition));
        }
        {
            ModelDefinition definition;
            definition.version                          = VERSION_FLEX_2;
            definition.geometry                         = {1, 16, 3};
            definition.name                             = "Flex.2";
            definition.id                               = "flex2";
            definition.traits.is_dit                    = true;
            definition.traits.vae_kind                  = ModelVAEKind::FLUX;
            definition.traits.is_inpaint                = true;
            definition.traits.is_control                = true;
            definition.sampling                         = {FLUX_FLOW_PRED, 1.0f};
            definition.sampling.default_sample_method   = EULER_SAMPLE_METHOD;
            definition.sampling.default_scheduler       = FLUX_SCHEDULER;
            definition.build                            = build_flux;
            definition.sampling.flow_shift_from_weights = flux_guidance_flow_shift;
            result.push_back(std::move(definition));
        }
        {
            ModelDefinition definition;
            definition.version                          = VERSION_CHROMA_RADIANCE;
            definition.geometry                         = {1, 3, 3};
            definition.name                             = "Chroma Radiance";
            definition.id                               = "chroma_radiance";
            definition.traits.is_dit                    = true;
            definition.traits.vae_kind                  = ModelVAEKind::FLUX;
            definition.traits.fake_vae                  = true;
            definition.sampling                         = {FLUX_FLOW_PRED, 1.0f};
            definition.sampling.default_sample_method   = EULER_SAMPLE_METHOD;
            definition.sampling.default_scheduler       = FLUX_SCHEDULER;
            definition.build                            = build_flux;
            definition.sampling.flow_shift_from_weights = flux_guidance_flow_shift;
            result.push_back(std::move(definition));
        }
        {
            ModelDefinition definition;
            definition.version                          = VERSION_CHROMA;
            definition.geometry                         = {1, 16, 3};
            definition.name                             = "Chroma";
            definition.id                               = "chroma";
            definition.traits.is_dit                    = true;
            definition.traits.vae_kind                  = ModelVAEKind::FLUX;
            definition.sampling                         = {FLUX_FLOW_PRED, 1.0f};
            definition.sampling.default_sample_method   = EULER_SAMPLE_METHOD;
            definition.sampling.default_scheduler       = FLUX_SCHEDULER;
            definition.build                            = build_flux;
            definition.sampling.flow_shift_from_weights = flux_guidance_flow_shift;
            result.push_back(std::move(definition));
        }
        {
            ModelDefinition definition;
            definition.version                          = VERSION_OVIS_IMAGE;
            definition.geometry                         = {1, 16, 3};
            definition.name                             = "Ovis Image";
            definition.id                               = "ovis_image";
            definition.traits.is_dit                    = true;
            definition.traits.vae_kind                  = ModelVAEKind::FLUX;
            definition.sampling                         = {FLUX_FLOW_PRED, 1.0f};
            definition.sampling.default_sample_method   = EULER_SAMPLE_METHOD;
            definition.sampling.default_scheduler       = FLUX_SCHEDULER;
            definition.build                            = build_flux;
            definition.sampling.flow_shift_from_weights = flux_guidance_flow_shift;
            result.push_back(std::move(definition));
        }
        return result;
    }();
    return definitions;
}

float flux_guidance_flow_shift(const String2TensorStorage& weights) {
    for (const auto& [name, tensor] : weights) {
        if (starts_with(name, "model.diffusion_model.guidance_in.in_layer.weight"))
            return 1.15f;
    }
    return 1.f;
}
