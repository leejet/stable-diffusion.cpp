#include "model_definition.h"

#include "conditioning/conditioner.hpp"
#include "model/diffusion/nucleus_image.hpp"
#include "pipeline/model_builders.h"

static bool build_nucleus_image(const sd::model_builders::Context& ctx,
                                const TokenizerConfig& tokenizers,
                                sd::model_builders::CoreRunners& runners) {
    // NOTE: Nucleus trunk activations run at RMS 100-600 with outliers up to ~2e5, which
    // overflows every F16-accumulated matmul kernel (Vulkan MMQ / CPU llamafile; F16 max
    // 65504). This is handled inside nucleus_image.hpp with exact power-of-two rescales at
    // each large-input matmul (see NucleusFeedForward / NucleusAttention), so the weights
    // stay quantized as shipped and the memory footprint is unchanged.
    runners.conditioner = std::make_shared<LLMEmbedder>(ctx.backends.runtime_backend(SDBackendModule::TE),
                                                        ctx.tensor_storage_map, ctx.version, "", false,
                                                        ctx.weight_manager, tokenizers);
    runners.diffusion   = std::make_shared<NucleusImage::NucleusImageRunner>(ctx.backends.runtime_backend(SDBackendModule::DIFFUSION),
                                                                             ctx.tensor_storage_map, "model.diffusion_model", ctx.weight_manager);
    return true;
}

const std::vector<ModelDefinition>& nucleus_image_definitions() {
    static const std::vector<ModelDefinition> definitions = [] {
        ModelDefinition definition;
        definition.version                        = VERSION_NUCLEUS_IMAGE;
        definition.geometry                       = {1, 16, 3};  // Qwen-Image VAE latents (z=16, 8x)
        definition.name                           = "Nucleus-Image";
        definition.traits.is_dit                  = true;
        definition.traits.vae_kind                = ModelVAEKind::WAN;  // AutoencoderKLQwenImage == WAN kind
        definition.traits.video_tae               = true;
        definition.sampling                       = {FLOW_PRED, 1.f};  // FlowMatchEuler, static shift 1.0
        definition.sampling.default_sample_method = EULER_SAMPLE_METHOD;
        definition.sampling.default_scheduler     = DISCRETE_SCHEDULER;
        definition.build                          = build_nucleus_image;
        return std::vector<ModelDefinition>{std::move(definition)};
    }();
    return definitions;
}
