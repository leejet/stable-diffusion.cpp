#include "dgl/models/component_builders.h"

#include "model/vae/geometry.hpp"

#include <cmath>
#include <cstring>
#include <utility>

#include "conditioning/conditioner.hpp"
#include "core/ggml_extend_backend.h"
#include "core/util.h"
#include "dgl/models/control.h"
#include "dgl/models/relay_vsr.h"
#include "dgl/models/vae.h"
#include "model/diffusion/control.hpp"
#include "model/diffusion/model.hpp"
#include "model/vae/auto_encoder_kl.hpp"
#include "model/vae/hunyuan_vae.hpp"
#include "model/vae/ltx_audio_vae.hpp"
#include "model/vae/ltx_vae.hpp"
#include "model/vae/mage_vae.hpp"
#include "model/vae/minimax_h3_audio_vae.hpp"
#include "model/vae/minimax_h3_vae.hpp"
#include "model/vae/seedvr2_vae.hpp"
#include "model/vae/tae.hpp"
#include "model/vae/vae.hpp"
#include "model/vae/wan_vae.hpp"
#include "model_definition.h"

namespace sd::dgl::models::component_builders {

    bool build_family_runners(const Context& ctx, CoreRunners& runners, TokenizerConfig tokenizers) try {
        const auto* sd_ctx_params = &ctx.params;
        const auto version        = ctx.version;
        CoreRunners result;
        if ((ctx.build_conditioner && !sd::model_builders::ensure_model_backend_pair(ctx.backends, SDBackendModule::TE)) ||
            (ctx.build_diffusion && !sd::model_builders::ensure_model_backend_pair(ctx.backends, SDBackendModule::DIFFUSION))) {
            return false;
        }

        const auto* definition = find_model_definition(version);
        if (!definition || !definition->build || !definition->build(ctx, tokenizers, result)) {
            LOG_ERROR("failed to build model runners for %s", sd_version_name(version));
            return false;
        }

        if (result.conditioner) {
            result.conditioner->set_scale_overrides(sd_ctx_params->linear_scale, sd_ctx_params->attn_scale);
        }
        if (result.diffusion) {
            result.diffusion->set_scale_overrides(sd_ctx_params->linear_scale, sd_ctx_params->attn_scale);
        }
        if (result.high_noise_diffusion) {
            result.high_noise_diffusion->set_scale_overrides(sd_ctx_params->linear_scale, sd_ctx_params->attn_scale);
        }
        tokenizers.validate_usage();
        runners = std::move(result);
        return true;
    } catch (const std::exception& error) {
        LOG_ERROR("failed to build model runners: %s", error.what());
        return false;
    }

    std::shared_ptr<AudioVAERunner> build_audio_vae_runner(const Context& ctx) {
        if (!sd::model_builders::ensure_model_backend_pair(ctx.backends, SDBackendModule::VAE)) {
            return nullptr;
        }
        const auto backend = ctx.backends.runtime_backend(SDBackendModule::VAE);
        std::shared_ptr<AudioVAERunner> audio;
        if (sd_version_is_minimax_h3(ctx.version)) {
            audio = std::make_shared<MiniMaxH3::AudioVAERunner>(backend, ctx.tensor_storage_map, "", ctx.weight_manager);
        } else {
            audio = std::make_shared<LTXV::LTXAudioVAERunner>(backend, ctx.tensor_storage_map, "", ctx.weight_manager);
        }
        audio->set_scale_overrides(ctx.params.linear_scale, ctx.params.attn_scale);
        return audio;
    }

    std::shared_ptr<::VAE> build_tae_runner(const Context& ctx, bool decode_only) {
        const auto backend = ctx.backends.runtime_backend(SDBackendModule::VAE);
        std::shared_ptr<::VAE> model;
        if (find_model_traits(ctx.version)->video_tae) {
            model = std::make_shared<TinyVideoAutoEncoder>(backend, ctx.tensor_storage_map, "decoder", decode_only,
                                                           ctx.version, ctx.weight_manager);
        } else {
            model = std::make_shared<TinyImageAutoEncoder>(backend, ctx.tensor_storage_map, "decoder.layers", decode_only,
                                                           ctx.version, ctx.weight_manager);
        }
        model->set_scale_overrides(ctx.params.linear_scale, ctx.params.attn_scale);
        return model;
    }

    std::shared_ptr<::VAE> build_vae_runner(const Context& ctx, const ModelLoader& loader) try {
        const auto* sd_ctx_params      = &ctx.params;
        const auto& tensor_storage_map = ctx.tensor_storage_map;
        const auto version             = ctx.version;
        const auto& weight_manager     = ctx.weight_manager;
        if (!sd::model_builders::ensure_model_backend_pair(ctx.backends, SDBackendModule::VAE))
            return {};

        sd_vae_format_t vae_format = sd_ctx_params->vae_format;
        if (vae_format < SD_VAE_FORMAT_AUTO || vae_format >= SD_VAE_FORMAT_COUNT) {
            LOG_WARN("invalid VAE format override, using auto");
            vae_format = SD_VAE_FORMAT_AUTO;
        }
        SDVersion vae_version = version;
        if (find_model_traits(version)->vae_format_override && vae_format != SD_VAE_FORMAT_AUTO) {
            vae_version = sd::vae::version_for_format(vae_format, vae_version);
        }

        auto create_vae = [&]() -> std::shared_ptr<::VAE> {
            if (vae_version == VERSION_RELAY_VSR) {
                return build_relay_vsr_vae(ctx, loader);
            } else if (vae_version == VERSION_SEEDVR2) {
                const char* prefix = tensor_storage_map.count("first_stage_model.encoder.conv_out.bias") ? "first_stage_model" : "vae";
                return std::make_shared<SeedVR2::SeedVR2VAERunner>(ctx.backends.runtime_backend(SDBackendModule::VAE),
                                                                   tensor_storage_map, weight_manager, prefix);
            } else if (find_model_traits(vae_version)->vae_kind == ModelVAEKind::LTXAV) {
                return std::make_shared<LTXVideoVAE>(ctx.backends.runtime_backend(SDBackendModule::VAE),
                                                     tensor_storage_map,
                                                     "first_stage_model",
                                                     false,
                                                     version,
                                                     weight_manager,
                                                     false);
            } else if (find_model_traits(vae_version)->vae_kind == ModelVAEKind::MINIMAX_H3) {
                if (version == VERSION_MINIMAX_H3_IMAGE) {
                    auto proj     = tensor_storage_map.find("first_stage_model.decoder.proj_out.weight");
                    auto embedder = tensor_storage_map.find("first_stage_model.decoder.x_embedder.weight");
                    if (proj == tensor_storage_map.end() || proj->second.ne[1] != 768 ||
                        embedder == tensor_storage_map.end() || embedder->second.ne[1] != 1024) {
                        throw std::runtime_error("MiniMax-H3-Image requires its 2D image VAE; pass --vae minimax_h3_image_vae_fp16.safetensors");
                    }
                }
                if (version == VERSION_P_H3_IMAGE) {
                    // The video VAE also loads successfully, but uses different temporal decoding and posterior sampling.
                    auto proj = tensor_storage_map.find("first_stage_model.decoder.proj_out.weight");
                    if (proj == tensor_storage_map.end() || proj->second.ne[1] != 768) {
                        throw std::runtime_error("P-H3-Image requires its single-frame VAE; pass --vae p-h3-image_vae_bf16.safetensors");
                    }
                }
                return std::make_shared<MiniMaxH3VAE::MiniMaxH3VideoVAERunner>(ctx.backends.runtime_backend(SDBackendModule::VAE),
                                                                               tensor_storage_map,
                                                                               "first_stage_model",
                                                                               weight_manager);
            } else if (find_model_traits(vae_version)->vae_kind == ModelVAEKind::MAGE) {
                return std::make_shared<MageVAE::MageVAERunner>(ctx.backends.runtime_backend(SDBackendModule::VAE),
                                                                tensor_storage_map,
                                                                "first_stage_model",
                                                                weight_manager);
            } else if (find_model_traits(vae_version)->vae_kind == ModelVAEKind::HUNYUAN) {
                return std::make_shared<Hunyuan::HunyuanVideoVAERunner>(ctx.backends.runtime_backend(SDBackendModule::VAE),
                                                                        tensor_storage_map,
                                                                        "first_stage_model",
                                                                        false,
                                                                        vae_version,
                                                                        weight_manager);
            } else if (find_model_traits(vae_version)->vae_kind == ModelVAEKind::WAN) {
                return std::make_shared<WAN::WanVAERunner>(ctx.backends.runtime_backend(SDBackendModule::VAE),
                                                           tensor_storage_map,
                                                           "first_stage_model",
                                                           false,
                                                           vae_version,
                                                           weight_manager);
            } else {
                auto model             = std::make_shared<AutoEncoderKL>(ctx.backends.runtime_backend(SDBackendModule::VAE),
                                                                         tensor_storage_map,
                                                                         "first_stage_model",
                                                                         false,
                                                                         false,
                                                                         vae_version,
                                                                         weight_manager);
                const auto* definition = find_model_definition(version);
                if (definition && definition->configure_vae) {
                    sd::model_builders::VAEOptions options;
                    definition->configure_vae(ctx, options, *model);
                }
                return model;
            }
        };

        std::shared_ptr<::VAE> model;
        if (find_model_traits(version)->fake_vae) {
            LOG_INFO("using FakeVAE");
            model = std::make_shared<FakeVAE>(version, ctx.backends.runtime_backend(SDBackendModule::VAE), weight_manager);
        } else {
            LOG_INFO("using VAE for encoding / decoding");
            model = create_vae();
        }
        if (model)
            model->set_scale_overrides(sd_ctx_params->linear_scale, sd_ctx_params->attn_scale);
        return model;
    } catch (const std::exception& error) {
        LOG_ERROR("failed to build VAE runner: %s", error.what());
        return {};
    }

    bool build_control_net_runner(const Context& ctx, std::shared_ptr<::ControlNet>& runner) {
        const auto* sd_ctx_params      = &ctx.params;
        const auto& tensor_storage_map = ctx.tensor_storage_map;
        const auto version             = ctx.version;
        const auto& weight_manager     = ctx.weight_manager;
        if (!sd::model_builders::ensure_model_backend_pair(ctx.backends, SDBackendModule::CONTROL_NET)) {
            return false;
        }
        auto control_net = std::make_shared<::ControlNet>(ctx.backends.runtime_backend(SDBackendModule::CONTROL_NET),
                                                          tensor_storage_map,
                                                          version,
                                                          "",
                                                          weight_manager);
        control_net->set_scale_overrides(sd_ctx_params->linear_scale, sd_ctx_params->attn_scale);
        runner = std::move(control_net);
        return true;
    }

}  // namespace sd::dgl::models::component_builders
