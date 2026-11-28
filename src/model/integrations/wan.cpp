#include "model/diffusion/wan.hpp"
#include "conditioning/conditioner.hpp"
#include "model/audio/wav2vec2.hpp"
#include "model/diffusion/relay_vsr.hpp"
#include "model_build_context.h"
#include "model_definition.h"

static bool build_wan(const sd::model_builders::Context& ctx, const TokenizerConfig& tokenizers, sd::model_builders::CoreRunners& result) {
    const auto* sd_ctx_params      = &ctx.params;
    const auto& tensor_storage_map = ctx.tensor_storage_map;
    const auto version             = ctx.version;
    const auto& weight_manager     = ctx.weight_manager;
    if (ctx.build_conditioner) {
        result.conditioner = std::make_shared<T5CLIPEmbedder>(ctx.backends.runtime_backend(SDBackendModule::TE),
                                                              tensor_storage_map,
                                                              true,
                                                              0,
                                                              true,
                                                              weight_manager);
    }
    if (ctx.build_diffusion) {
        result.diffusion = std::make_shared<WAN::WanRunner>(ctx.backends.runtime_backend(SDBackendModule::DIFFUSION),
                                                            tensor_storage_map,
                                                            "model.diffusion_model",
                                                            version,
                                                            weight_manager);
    }
    if (ctx.build_auxiliary && sd_ctx_params->build_high_noise) {
        result.high_noise_diffusion = std::make_shared<WAN::WanRunner>(ctx.backends.runtime_backend(SDBackendModule::DIFFUSION),
                                                                       tensor_storage_map,
                                                                       "model.high_noise_diffusion_model",
                                                                       version,
                                                                       weight_manager);
    }
    if (ctx.build_auxiliary && result.diffusion &&
        (result.diffusion->get_desc() == "Wan2.1-I2V-14B" ||
         result.diffusion->get_desc() == "Wan2.1-FLF2V-14B" ||
         result.diffusion->get_desc() == "Wan2.1-I2V-1.3B")) {
        if (!sd::model_builders::ensure_model_backend_pair(ctx.backends, SDBackendModule::CLIP_VISION)) {
            return false;
        }
        result.clip_vision = std::make_shared<FrozenCLIPVisionEmbedder>(ctx.backends.runtime_backend(SDBackendModule::CLIP_VISION),
                                                                        tensor_storage_map,
                                                                        weight_manager);
    }
    if (ctx.build_auxiliary && version == VERSION_WAN2_2_S2V &&
        tensor_storage_map.count("wav2vec2.encoder.layer_norm.bias") > 0) {
        if (!sd::model_builders::ensure_model_backend_pair(ctx.backends, SDBackendModule::AUDIO_ENCODER)) {
            return false;
        }
        result.audio_encoder = std::make_shared<Wav2Vec2::Wav2Vec2ModelRunner>(ctx.backends.runtime_backend(SDBackendModule::AUDIO_ENCODER),
                                                                               tensor_storage_map,
                                                                               "wav2vec2.",
                                                                               weight_manager);
    }
    return true;
}

const std::vector<ModelDefinition>& wan_definitions() {
    static const std::vector<ModelDefinition> definitions = [] {
        std::vector<ModelDefinition> result;
        {
            ModelDefinition definition;
            definition.version                        = VERSION_RELAY_VSR;
            definition.name                           = "RelayVSR";
            definition.id                             = "relayvsr";
            definition.geometry                       = {2, 48, 3};
            definition.traits.is_dit                  = true;
            definition.sampling                       = {FLOW_PRED, 1.f};
            definition.sampling.default_sample_method = EULER_SAMPLE_METHOD;
            definition.build                          = [](const sd::model_builders::Context& ctx, const TokenizerConfig&, sd::model_builders::CoreRunners& runners) {
                if (!ctx.build_diffusion || ctx.build_conditioner)
                    return false;
                const auto channels = ctx.tensor_storage_map.at("wan.lq_proj.conv1.weight").ne[2];
                const int scale     = channels == 192 ? 2 : channels == 768 ? 4
                                                                            : 0;
                runners.diffusion   = std::make_shared<RelayVSR::WanVSRRunner>(ctx.backends.runtime_backend(SDBackendModule::DIFFUSION),
                                                                               ctx.tensor_storage_map, scale, ctx.weight_manager);
                return runners.diffusion != nullptr;
            };
            result.push_back(std::move(definition));
        }
        {
            ModelDefinition definition;
            definition.version                          = VERSION_WAN2;
            definition.geometry                         = {2, 16, 3};
            definition.name                             = "Wan 2.x";
            definition.id                               = "wan2";
            definition.traits.is_dit                    = true;
            definition.traits.supports_video_generation = true;
            definition.traits.vae_kind                  = ModelVAEKind::WAN;
            definition.traits.video_tae                 = true;
            definition.sampling                         = {FLOW_PRED, 5.f};
            definition.sampling.default_sample_method   = EULER_SAMPLE_METHOD;
            definition.sampling.default_scheduler       = DISCRETE_SCHEDULER;
            definition.build                            = build_wan;
            result.push_back(std::move(definition));
        }
        {
            ModelDefinition definition;
            definition.version                          = VERSION_WAN2_2_I2V;
            definition.geometry                         = {2, 16, 3};
            definition.name                             = "Wan 2.2 I2V";
            definition.id                               = "wan2_2_i2v";
            definition.traits.is_dit                    = true;
            definition.traits.supports_video_generation = true;
            definition.traits.vae_kind                  = ModelVAEKind::WAN;
            definition.traits.video_tae                 = true;
            definition.sampling                         = {FLOW_PRED, 5.f};
            definition.sampling.default_sample_method   = EULER_SAMPLE_METHOD;
            definition.sampling.default_scheduler       = DISCRETE_SCHEDULER;
            definition.build                            = build_wan;
            result.push_back(std::move(definition));
        }
        {
            ModelDefinition definition;
            definition.version                          = VERSION_WAN2_2_TI2V;
            definition.geometry                         = {2, 48, 3};
            definition.name                             = "Wan 2.2 TI2V";
            definition.id                               = "wan2_2_ti2v";
            definition.traits.is_dit                    = true;
            definition.traits.supports_video_generation = true;
            definition.traits.vae_kind                  = ModelVAEKind::WAN;
            definition.traits.video_tae                 = true;
            definition.sampling                         = {FLOW_PRED, 5.f};
            definition.sampling.default_sample_method   = EULER_SAMPLE_METHOD;
            definition.sampling.default_scheduler       = DISCRETE_SCHEDULER;
            definition.build                            = build_wan;
            result.push_back(std::move(definition));
        }
        {
            ModelDefinition definition;
            definition.version                          = VERSION_WAN2_2_S2V;
            definition.geometry                         = {2, 16, 3};
            definition.name                             = "Wan 2.2 S2V";
            definition.id                               = "wan2_2_s2v";
            definition.traits.is_dit                    = true;
            definition.traits.supports_video_generation = true;
            definition.traits.vae_kind                  = ModelVAEKind::WAN;
            definition.traits.video_tae                 = true;
            definition.sampling                         = {FLOW_PRED, 5.f};
            definition.sampling.default_sample_method   = EULER_SAMPLE_METHOD;
            definition.sampling.default_scheduler       = DISCRETE_SCHEDULER;
            definition.build                            = build_wan;
            result.push_back(std::move(definition));
        }
        return result;
    }();
    return definitions;
}
