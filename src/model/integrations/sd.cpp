#include "conditioning/conditioner.hpp"
#include "core/option_types.h"
#include "model/diffusion/unet.hpp"
#include "model/vae/auto_encoder_kl.hpp"
#include "model_build_context.h"
#include "model_definition.h"

static void configure_sdxl_vae(const sd::model_builders::Context& ctx,
                               const sd::model_builders::VAEOptions& options,
                               VAE& vae) {
    if (options.apply_default_conv_scale) {
        float vae_conv_2d_scale = 1.f / 32.f;
        LOG_WARN(
            "No valid VAE specified with --vae or --force-sdxl-vae-conv-scale flag set, "
            "using Conv2D scale %.3f",
            vae_conv_2d_scale);
        static_cast<AutoEncoderKL&>(vae).set_conv2d_scale(vae_conv_2d_scale);
    }
}

static bool build_sd(const sd::model_builders::Context& ctx, const TokenizerConfig& tokenizers, sd::model_builders::CoreRunners& result) {
    const auto* sd_ctx_params      = &ctx.params;
    const auto& tensor_storage_map = ctx.tensor_storage_map;
    const auto version             = ctx.version;
    const auto& weight_manager     = ctx.weight_manager;
    // SD1.x SD2.x SDXL
    if (ctx.build_conditioner) {
        result.conditioner = std::make_shared<FrozenCLIPEmbedderWithCustomWords>(ctx.backends.runtime_backend(SDBackendModule::TE),
                                                                                 tensor_storage_map,
                                                                                 sd_ctx_params->embeddings,
                                                                                 version,
                                                                                 weight_manager,
                                                                                 tokenizers);
    }
    if (ctx.build_diffusion) {
        result.diffusion = std::make_shared<UNetModelRunner>(ctx.backends.runtime_backend(SDBackendModule::DIFFUSION),
                                                             tensor_storage_map,
                                                             "model.diffusion_model",
                                                             version,
                                                             weight_manager);
    }
    return true;
}
static prediction_t detect_sdxl_prediction(const String2TensorStorage& weights) {
    if (weights.count("edm_vpred.sigma_max"))
        return EDM_V_PRED;
    return weights.count("v_pred") ? V_PRED : EPS_PRED;
}

const std::vector<ModelDefinition>& sd_definitions() {
    static const std::vector<ModelDefinition> definitions = [] {
        std::vector<ModelDefinition> result;
        {
            ModelDefinition definition;
            definition.version  = VERSION_SD1;
            definition.geometry = {8, 4, 3};
            definition.name     = "SD 1.x";
            definition.id       = "sd1";

            definition.sampling                       = {EPS_PRED, INFINITY};
            definition.sampling.default_sample_method = EULER_A_SAMPLE_METHOD;
            definition.sampling.default_scheduler     = DISCRETE_SCHEDULER;
            definition.build                          = build_sd;
            result.push_back(std::move(definition));
        }
        {
            ModelDefinition definition;
            definition.version                        = VERSION_SD1_INPAINT;
            definition.geometry                       = {8, 4, 3};
            definition.name                           = "SD 1.x Inpaint";
            definition.id                             = "sd1_inpaint";
            definition.traits.is_inpaint              = true;
            definition.sampling                       = {EPS_PRED, INFINITY};
            definition.sampling.default_sample_method = EULER_A_SAMPLE_METHOD;
            definition.sampling.default_scheduler     = DISCRETE_SCHEDULER;
            definition.build                          = build_sd;
            result.push_back(std::move(definition));
        }
        {
            ModelDefinition definition;
            definition.version                        = VERSION_SD1_PIX2PIX;
            definition.geometry                       = {8, 4, 3};
            definition.name                           = "Instruct-Pix2Pix";
            definition.id                             = "sd1_pix2pix";
            definition.traits.is_unet_edit            = true;
            definition.sampling                       = {EPS_PRED, INFINITY};
            definition.sampling.default_sample_method = EULER_A_SAMPLE_METHOD;
            definition.sampling.default_scheduler     = DISCRETE_SCHEDULER;
            definition.build                          = build_sd;
            result.push_back(std::move(definition));
        }
        {
            ModelDefinition definition;
            definition.version  = VERSION_SD1_TINY_UNET;
            definition.geometry = {8, 4, 3};
            definition.name     = "SD 1.x Tiny UNet";
            definition.id       = "sd1_tiny_unet";

            definition.sampling                       = {EPS_PRED, INFINITY};
            definition.sampling.default_sample_method = EULER_A_SAMPLE_METHOD;
            definition.sampling.default_scheduler     = DISCRETE_SCHEDULER;
            definition.build                          = build_sd;
            result.push_back(std::move(definition));
        }
        {
            ModelDefinition definition;
            definition.version  = VERSION_SD2;
            definition.geometry = {8, 4, 3};
            definition.name     = "SD 2.x";
            definition.id       = "sd2";

            definition.sampling                       = {EPS_PRED, INFINITY};
            definition.sampling.default_sample_method = EULER_A_SAMPLE_METHOD;
            definition.sampling.default_scheduler     = DISCRETE_SCHEDULER;
            definition.build                          = build_sd;
            definition.sampling.probe_v_prediction    = true;
            result.push_back(std::move(definition));
        }
        {
            ModelDefinition definition;
            definition.version                        = VERSION_SD2_INPAINT;
            definition.geometry                       = {8, 4, 3};
            definition.name                           = "SD 2.x Inpaint";
            definition.id                             = "sd2_inpaint";
            definition.traits.is_inpaint              = true;
            definition.sampling                       = {EPS_PRED, INFINITY};
            definition.sampling.default_sample_method = EULER_A_SAMPLE_METHOD;
            definition.sampling.default_scheduler     = DISCRETE_SCHEDULER;
            definition.build                          = build_sd;
            definition.sampling.probe_v_prediction    = true;
            result.push_back(std::move(definition));
        }
        {
            ModelDefinition definition;
            definition.version  = VERSION_SD2_TINY_UNET;
            definition.geometry = {8, 4, 3};
            definition.name     = "SD 2.x Tiny UNet";
            definition.id       = "sd2_tiny_unet";

            definition.sampling                       = {EPS_PRED, INFINITY};
            definition.sampling.default_sample_method = EULER_A_SAMPLE_METHOD;
            definition.sampling.default_scheduler     = DISCRETE_SCHEDULER;
            definition.build                          = build_sd;
            definition.sampling.probe_v_prediction    = true;
            result.push_back(std::move(definition));
        }
        {
            ModelDefinition definition;
            definition.version  = VERSION_SDXS_512_DS;
            definition.geometry = {8, 4, 3};
            definition.name     = "SDXS (512-DS)";
            definition.id       = "sdxs_512_ds";

            definition.sampling                       = {EPS_PRED, INFINITY};
            definition.sampling.default_sample_method = EULER_A_SAMPLE_METHOD;
            definition.sampling.default_scheduler     = DISCRETE_SCHEDULER;
            definition.build                          = build_sd;
            result.push_back(std::move(definition));
        }
        {
            ModelDefinition definition;
            definition.version  = VERSION_SDXS_09;
            definition.geometry = {8, 4, 3};
            definition.name     = "SDXS (09)";
            definition.id       = "sdxs_09";

            definition.sampling                       = {EPS_PRED, INFINITY};
            definition.sampling.default_sample_method = EULER_A_SAMPLE_METHOD;
            definition.sampling.default_scheduler     = DISCRETE_SCHEDULER;
            definition.build                          = build_sd;
            definition.sampling.probe_v_prediction    = true;
            result.push_back(std::move(definition));
        }
        {
            ModelDefinition definition;
            definition.version  = VERSION_SDXL;
            definition.geometry = {8, 4, 3};
            definition.name     = "SDXL";
            definition.id       = "sdxl";

            definition.sampling                         = {EPS_PRED, INFINITY};
            definition.sampling.default_sample_method   = EULER_A_SAMPLE_METHOD;
            definition.sampling.default_scheduler       = DISCRETE_SCHEDULER;
            definition.build                            = build_sd;
            definition.sampling.prediction_from_weights = detect_sdxl_prediction;
            definition.configure_vae                    = configure_sdxl_vae;
            result.push_back(std::move(definition));
        }
        {
            ModelDefinition definition;
            definition.version                          = VERSION_SDXL_INPAINT;
            definition.geometry                         = {8, 4, 3};
            definition.name                             = "SDXL Inpaint";
            definition.id                               = "sdxl_inpaint";
            definition.traits.is_inpaint                = true;
            definition.sampling                         = {EPS_PRED, INFINITY};
            definition.sampling.default_sample_method   = EULER_A_SAMPLE_METHOD;
            definition.sampling.default_scheduler       = DISCRETE_SCHEDULER;
            definition.build                            = build_sd;
            definition.sampling.prediction_from_weights = detect_sdxl_prediction;
            definition.configure_vae                    = configure_sdxl_vae;
            result.push_back(std::move(definition));
        }
        {
            ModelDefinition definition;
            definition.version                          = VERSION_SDXL_PIX2PIX;
            definition.geometry                         = {8, 4, 3};
            definition.name                             = "SDXL Instruct-Pix2Pix";
            definition.id                               = "sdxl_pix2pix";
            definition.traits.is_unet_edit              = true;
            definition.sampling                         = {EPS_PRED, INFINITY};
            definition.sampling.default_sample_method   = EULER_A_SAMPLE_METHOD;
            definition.sampling.default_scheduler       = DISCRETE_SCHEDULER;
            definition.build                            = build_sd;
            definition.sampling.prediction_from_weights = detect_sdxl_prediction;
            definition.configure_vae                    = configure_sdxl_vae;
            result.push_back(std::move(definition));
        }
        {
            ModelDefinition definition;
            definition.version  = VERSION_SDXL_VEGA;
            definition.geometry = {8, 4, 3};
            definition.name     = "SDXL (Vega)";
            definition.id       = "sdxl_vega";

            definition.sampling                         = {EPS_PRED, INFINITY};
            definition.sampling.default_sample_method   = EULER_A_SAMPLE_METHOD;
            definition.sampling.default_scheduler       = DISCRETE_SCHEDULER;
            definition.build                            = build_sd;
            definition.sampling.prediction_from_weights = detect_sdxl_prediction;
            definition.configure_vae                    = configure_sdxl_vae;
            result.push_back(std::move(definition));
        }
        {
            ModelDefinition definition;
            definition.version  = VERSION_SDXL_SSD1B;
            definition.geometry = {8, 4, 3};
            definition.name     = "SDXL (SSD1B)";
            definition.id       = "sdxl_ssd1b";

            definition.sampling                         = {EPS_PRED, INFINITY};
            definition.sampling.default_sample_method   = EULER_A_SAMPLE_METHOD;
            definition.sampling.default_scheduler       = DISCRETE_SCHEDULER;
            definition.build                            = build_sd;
            definition.sampling.prediction_from_weights = detect_sdxl_prediction;
            definition.configure_vae                    = configure_sdxl_vae;
            result.push_back(std::move(definition));
        }
        {
            ModelDefinition definition;
            definition.version                          = VERSION_SVD;
            definition.geometry                         = {8, 4, 3};
            definition.name                             = "SVD";
            definition.id                               = "svd";
            definition.traits.supports_video_generation = true;
            definition.sampling                         = {EPS_PRED, INFINITY};
            definition.sampling.default_sample_method   = EULER_A_SAMPLE_METHOD;
            definition.sampling.default_scheduler       = DISCRETE_SCHEDULER;
            definition.build                            = build_sd;
            result.push_back(std::move(definition));
        }
        {
            ModelDefinition definition;
            definition.version  = VERSION_ESRGAN;
            definition.geometry = {8, 4, 3};
            definition.name     = "ESRGAN";
            definition.id       = "esrgan";

            definition.sampling                       = {EPS_PRED, INFINITY};
            definition.sampling.default_sample_method = EULER_A_SAMPLE_METHOD;
            definition.sampling.default_scheduler     = DISCRETE_SCHEDULER;
            definition.build                          = build_sd;
            result.push_back(std::move(definition));
        }
        return result;
    }();
    return definitions;
}
