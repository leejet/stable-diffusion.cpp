#include "api/generation_program.h"

#include <algorithm>
#include <cmath>

#include "core/image_geometry.h"
#include "model/vae/geometry.hpp"

namespace sd::api {
    void prepare_hires_program(GenerationTemplate& request) {
        const auto& hires      = request.image.hires;
        const auto mode        = text(hires.upscaler);
        const bool enabled     = hires.enabled && mode != "none";
        request.flags["hires"] = enabled;
        if (!enabled)
            return;
        static const std::map<std::string, std::string> modes{
            {"latent", "bilinear"}, {"latent_nearest", "nearest"}, {"latent_nearest_exact", "nearest_exact"}, {"latent_antialiased", "bilinear"}, {"latent_bicubic", "bicubic"}, {"latent_bicubic_antialiased", "bicubic"}, {"lanczos", "lanczos"}, {"nearest", "nearest"}, {"model", "nearest"}};
        const auto found = modes.find(mode);
        if (found == modes.end())
            throw dgl::Error(SD_STATUS_INVALID_ARGUMENT, "unknown hires upscaler '" + mode + "'");
        const bool model   = mode == "model";
        const bool pixels  = !request.video && (model || mode == "lanczos" || mode == "nearest");
        const auto version = request.definition.version;
        const bool ltx     = sd_version_is_ltxav(version);
        if (request.video && !ltx && version != VERSION_MINIMAX_H3)
            throw dgl::Error(SD_STATUS_UNSUPPORTED, "video hires supports LTX and MiniMax-H3; other workflows can be written in DGL");
        if (request.video && (mode == "lanczos" || mode == "nearest"))
            throw dgl::Error(SD_STATUS_INVALID_ARGUMENT, "video hires requires a latent interpolation mode or a latent upscaler model");
        if (model && text(hires.model_path).empty())
            throw dgl::Error(SD_STATUS_INVALID_ARGUMENT, "hires model upscaling requires hires.model_path");
        const auto& settings   = request.context.model.config.context;
        const auto vae_version = request.definition.traits.vae_format_override && request.fields.at("vae_loader") != "load_tae"
                                     ? sd::vae::version_for_format(settings.vae_format, version)
                                     : version;
        const int scale        = sd_version_is_pid(version) || sd_version_is_minit2i(version) || version == VERSION_IRIS ? 1 : sd::vae::spatial_scale(vae_version);
        const int multiple     = scale * request.definition.geometry.diffusion_down_factor;
        const double width     = hires.target_width ? hires.target_width : hires.target_height ? hires.target_height
                                                                                               : std::round(request.values.at("width").get<int>() * double(hires.scale));
        const double height    = hires.target_height ? hires.target_height : hires.target_width ? hires.target_width
                                                                                                : std::round(request.values.at("height").get<int>() * double(hires.scale));
        if (width < 1 || height < 1 || width > 16384 || height > 16384)
            throw dgl::Error(SD_STATUS_OUT_OF_RANGE, "hires target dimensions must be in 1..16384");
        const auto& sampling         = request.image.sample_params;
        const int default_steps      = sampling.custom_sigmas_count ? sampling.custom_sigmas_count - 1 : sampling.sample_steps;
        const int steps              = hires.steps > 0 ? hires.steps : default_steps;
        const double scheduler_steps = steps / double(hires.denoising_strength);
        if (!hires.custom_sigmas_count && scheduler_steps > 10000)
            throw dgl::Error(SD_STATUS_OUT_OF_RANGE, "hires steps / denoising_strength exceeds 10000; supply hires.sigmas or increase the strength");
        request.values.update({{"hires_target_width", sd::align_image_dimension(int(width), multiple)},
                               {"hires_target_height", sd::align_image_dimension(int(height), multiple)},
                               {"hires_steps", steps},
                               {"hires_scheduler_steps", std::max(1, int(std::min(scheduler_steps, 10000.0)))},
                               {"hires_mode", found->second},
                               {"hires_antialias", mode == "latent_antialiased" || mode == "latent_bicubic_antialiased"},
                               {"hires_vae_scale", scale}});
        if (hires.custom_sigmas_count)
            request.values["hires_custom_sigmas"] = std::vector<float>(hires.custom_sigmas, hires.custom_sigmas + hires.custom_sigmas_count);
        if (model) {
            request.values["hires_model_path"] = text(hires.model_path);
            auto options                       = loader_options(settings, request.video ? SDBackendModule::DIFFUSION : SDBackendModule::UPSCALER, request.definition);
            if (!request.video) {
                auto placement = nlohmann::json::object();
                for (const auto* key : {"backend", "params_backend"})
                    if (options.contains(key))
                        placement[key] = options.at(key);
                options              = std::move(placement);
                options["tile_size"] = hires.upscale_tile_size;
            } else
                options["family"] = ltx ? "ltxav" : "minimax_h3";
            request.fields["hires_model_options"] = arguments(options);
        }
        request.flags.update({{"hires_model", model}, {"hires_pixels", pixels}, {"hires_ltx", ltx}, {"hires_pixel_model", request.fields.at("entry") == "image_pixel"}, {"hires_sefi", sd_version_is_sefi_image(version)}, {"hires_h3_image", sd_version_is_minimax_h3_image(version)}, {"hires_custom_sigmas", hires.custom_sigmas_count > 0}});
        request.fields["result_latent"]            = "hires_latent";
        request.fields["hires_sample_options"]     = request.fields.at("sample_core");
        request.fields["hires_schedule_predictor"] = request.fields.at("schedule_predictor");
        if (request.video) {
            request.fields["output_video_frames"]      = "hires_video_shape[2]";
            request.fields["output_audio_shape"]       = "hires_audio_shape";
            request.fields["hires_schedule_predictor"] = ltx ? "hires_guided" : "guided";
            request.fields["hires_positions_option"]   = request.flags.at("last").get<bool>() ? ", video_positions = hires_positions" : "";
            if (ltx && request.flags.at("conditioned").get<bool>())
                request.fields["hires_sample_options"] += ", base = hires_base, denoise_mask = hires_packed_mask, mask_input = true";
            return;
        }
        const bool rebuild                         = request.flags.value("inpaint_model", false) || request.flags.value("unet_edit", false) || request.flags.value("native_control", false) || request.flags.value("z_control", false);
        request.flags["hires_rebuild"]             = rebuild;
        request.fields["hires_positive_condition"] = request.fields.at("positive_condition");
        request.fields["hires_negative_condition"] = request.fields.at("negative_condition");
        request.fields["hires_guidance_options"]   = request.fields.at("guidance_core");
        if (request.flags.value("image_guidance", false)) {
            request.fields["hires_guidance_options"] += ", image_only = hires_image_only, image_scale = image_cfg_scale";
            request.fields["hires_image_only_condition"] = request.fields.at("image_only_condition");
        }
        if (request.flags.value("z_control", false)) {
            request.fields["hires_positive_condition"] = "hires_controlled_positive";
            request.fields["hires_negative_condition"] = "hires_controlled_negative";
            if (request.flags.value("reference_image_guidance", false)) {
                request.fields["hires_negative_condition"]   = "hires_controlled_unconditional";
                request.fields["hires_image_only_condition"] = "hires_controlled_negative";
            }
        }
        if (rebuild) {
            auto options = request.fields.at("predictor_core") + request.fields.at("reference_predictor_options");
            if (request.flags.value("inpaint_model", false))
                options += ", concat = hires_inpaint_condition";
            else if (request.flags.value("unet_edit", false))
                options += ", concat = hires_edit_latent";
            else if (version == VERSION_FLUX_CONTROLS)
                options += ", concat = hires_control_latent";
            request.fields["hires_predictor_options"]          = options;
            request.fields["hires_image_only_options"]         = options;
            request.fields["hires_negative_predictor_options"] = request.flags.value("image_guidance", false)
                                                                     ? request.fields.at("predictor_core") + (request.flags.value("reference_image_guidance", false) ? "" : ", concat = hires_image_unconditional_concat")
                                                                     : options;
            request.fields["hires_schedule_predictor"]         = "hires_guided";
        }
        if (request.flags.value("mask_stage", false) || request.flags.value("has_mask", false)) {
            request.fields["hires_mask"] = request.flags.value("inpaint_model", false) ? "denoise_mask" : "mask_latent";
            request.fields["hires_sample_options"] += ", base = hires_base, denoise_mask = hires_mask";
        }
    }
}
