#include "api/generation_program.h"

#include <algorithm>
#include <cmath>

#include "model/vae/audio_geometry.hpp"

namespace sd::api {
    static void predictor_option(GenerationTemplate& request, const std::string& option) {
        request.fields["predictor_options"] += option;
        request.fields["negative_predictor_options"] += option;
    }

    static bool keyframe(GenerationTemplate& request, const char* key, const char* name) {
        const auto* tensor = bound_tensor(request.options, key);
        if (!tensor)
            return false;
        bind_tensor(request.program, name, tensor);
        return true;
    }

    static void driving_audio(GenerationTemplate& request) {
        const auto tracks = bound_audios(request.options, "ref_audios");
        if (tracks.size() > 1)
            throw dgl::Error(SD_STATUS_INVALID_ARGUMENT, "Wan S2V accepts one driving audio track");
        request.flags["driving_audio"] = !tracks.empty();
        if (tracks.empty())
            return;
        const auto& config = request.context.model.config.context;
        if (!config.audio_encoder_path || !*config.audio_encoder_path)
            throw dgl::Error(SD_STATUS_INVALID_ARGUMENT, "Wan S2V audio requires audio_encoder_path");
        bind_tensor(request.program, "driving_audio", tracks[0].data);
        request.values["audio_encoder_path"]    = config.audio_encoder_path;
        request.values["sample_rate"]           = tracks[0].sample_rate;
        request.fields["audio_encoder_options"] = arguments(loader_options(config, SDBackendModule::TE, request.definition));
        request.fields["audio"]                 = "{samples = driving_audio[:, :math::floor(sample_rate * frames / fps)], sample_rate = sample_rate}";
        predictor_option(request, ", audio_embedding = audio_embedding");
    }

    static void audiovisual(GenerationTemplate& request) {
        const bool h3        = sd_version_is_minimax_h3(request.definition.version);
        const auto& settings = request.context.model.config.context;
        auto path            = text(settings.audio_vae_path);
        if (path.empty())
            path = request.context.model.embedded_audio_vae;
        request.flags["decode_audio"] = !path.empty();
        if (!path.empty())
            request.values["audio_vae_path"] = path;
        else
            request.values["empty_audio_shape"] = sd::vae::audio_latent_shape(request.definition.version, request.values.at("frames").get<int>(), request.values.at("fps").get<int>());
        const auto video_shift = std::isfinite(request.image.sample_params.flow_shift) ? request.image.sample_params.flow_shift : request.definition.sampling.default_flow_shift;
        if (h3 && video_shift <= 0)
            throw dgl::Error(SD_STATUS_INVALID_ARGUMENT, "MiniMax-H3 flow shift must be positive");
        request.values["audio_scale"]       = h3 ? video_shift / 3.f : 1.f;
        request.fields["audio_vae_options"] = arguments(loader_options(settings, SDBackendModule::VAE, request.definition));
        predictor_option(request, h3 ? ", audio_length = audio_shape[0], fps = fps" : ", audio_length = audio_shape[1], fps = fps");
        if (h3)
            predictor_option(request, ", video_sigma_shift = " + literal(video_shift));
    }

    static void dual_expert(GenerationTemplate& request) {
        const auto& settings  = request.context.model.config.context;
        request.flags["dual"] = !text(settings.high_noise_diffusion_model_path).empty();
        if (!request.flags.at("dual").get<bool>())
            return;
        if (!sd_version_is_wan(request.definition.version) && !sd_version_is_lingbot_video(request.definition.version))
            throw dgl::Error(SD_STATUS_UNSUPPORTED, "two-model sampling requires Wan or LingBot; use DGL for other compositions");
        request.values["high_model_path"] = settings.high_noise_diffusion_model_path;
        request.values["moe_boundary"]    = request.video->moe_boundary;
        request.flags["fixed_high_steps"] = request.video->high_noise_steps >= 0;
        if (request.video->high_noise_steps >= 0) {
            const int64_t high_steps  = request.video->high_noise_steps;
            const int64_t total_steps = request.image.sample_params.sample_steps + high_steps;
            if (!request.image.sample_params.custom_sigmas_count && total_steps > 10000)
                throw dgl::Error(SD_STATUS_OUT_OF_RANGE, "combined low/high-noise steps exceed 10000");
            request.values["steps"]            = total_steps;
            request.values["high_noise_steps"] = high_steps;
            if (request.image.sample_params.custom_sigmas_count) {
                request.values["steps"] = request.image.sample_params.custom_sigmas_count - 1;
                if (high_steps > 0)
                    request.values["high_noise_steps"] = std::max<int64_t>(0, request.image.sample_params.custom_sigmas_count - 1 - request.image.sample_params.sample_steps);
            }
        }
        std::string high = "high_base", patches;
        for (uint32_t i = 0; i < request.video->lora_count; ++i) {
            const auto& lora = request.video->loras[i];
            if (!lora.is_high_noise)
                continue;
            patches += request.render("lora", {{"name", "high_lora_" + std::to_string(i)}, {"path", text(lora.path)}, {"diffusion", true}, {"conditioning", false}, {"model_result", "high_model_" + std::to_string(i)}, {"model_source", high}, {"weight", lora.multiplier}, {"mode", sd_lora_apply_mode_name(settings.lora_apply_mode)}});
            high = "high_model_" + std::to_string(i);
        }
        request.fields["high_loras"] = patches;
        request.fields["high_model"] = high;
    }

    void prepare_video_program(GenerationTemplate& request) {
        const auto version   = request.definition.version;
        const auto& settings = request.context.model.config.context;
        const bool first     = keyframe(request, "init_image", "first_frame");
        const bool last      = keyframe(request, "end_image", "last_frame");
        const bool vace      = request.context.model.wan_vace;
        const bool s2v       = version == VERSION_WAN2_2_S2V;
        const bool frozen    = (version == VERSION_WAN2_2_TI2V || sd_version_is_lingbot_video(version)) && (first || last);
        const bool ltx       = sd_version_is_ltxav(version);
        const bool h3        = version == VERSION_MINIMAX_H3;
        if (!h3 && !bound_videos(request.options, "ref_videos").empty())
            throw dgl::Error(SD_STATUS_UNSUPPORTED, "reference videos require MiniMax-H3");
        if (!h3 && !s2v && !bound_audios(request.options, "ref_audios").empty())
            throw dgl::Error(SD_STATUS_UNSUPPORTED, "reference audio requires MiniMax-H3 or Wan2.2 S2V");
        if (s2v && last)
            throw dgl::Error(SD_STATUS_UNSUPPORTED, "Wan2.2 S2V does not support an end image");
        const bool image_condition = !vace && !s2v && request.context.model.wan_image_condition;
        request.flags              = {{"first", first}, {"last", last}, {"conditioned", first || last}, {"vace", vace}, {"vace_reference", vace && first}, {"s2v_reference", s2v && first}, {"driving_audio", false}, {"image_condition", image_condition}, {"vision", image_condition && request.context.model.wan_vision_condition}, {"end_vision", request.context.model.wan_end_vision}, {"frozen", frozen}, {"hunyuan", sd_version_is_hunyuan_video(version) && (first || last)}, {"controls", false}, {"dual", false}, {"custom_sigmas", request.image.sample_params.custom_sigmas_count > 0}};
        request.fields["entry"]    = ltx ? "video_ltx" : h3 ? "video_h3"
                                                            : "video_wan";
        if (image_condition || request.flags.at("hunyuan").get<bool>())
            predictor_option(request, ", concat = video_concat");
        if (frozen)
            request.fields["sample_options"] += ", base = base, denoise_mask = video_mask, mask_input = true";
        if (vace) {
            const auto* frames = bound_tensor(request.options, "control_frames");
            if (frames) {
                bind_tensor(request.program, "control_video", frames);
                request.values["control_count"] = std::min(sd_tensor_dim(frames, 3), request.values.at("frames").get<int64_t>());
            }
            request.flags["controls"] = frames != nullptr;
            predictor_option(request, ", vace_context = vace_context, vace_strength = " + literal(request.video->vace_strength));
        } else if (bound_tensor(request.options, "control_frames")) {
            throw dgl::Error(SD_STATUS_UNSUPPORTED, "control frames require a VACE model or an explicit DGL workflow");
        }
        if (s2v) {
            if (first)
                predictor_option(request, ", ref_latents = [reference_latent]");
            driving_audio(request);
        }
        if (request.flags.at("vision").get<bool>()) {
            if (!settings.clip_vision_path || !*settings.clip_vision_path)
                throw dgl::Error(SD_STATUS_INVALID_ARGUMENT, "Wan image conditioning requires clip_vision_path");
            request.values["vision_path"]    = settings.clip_vision_path;
            request.fields["vision_options"] = arguments(loader_options(settings, SDBackendModule::TE, request.definition));
        }
        if (ltx || h3) {
            audiovisual(request);
            if (ltx && (first || last)) {
                request.values["strength"] = request.image.strength;
                request.fields["sample_options"] += ", base = base, denoise_mask = packed_mask, mask_input = true";
                if (last)
                    predictor_option(request, ", video_positions = positions");
            }
            if (h3)
                prepare_h3_references(request, first ? "first_frame_pixels" : "", last ? "last_frame_pixels" : "");
        }
        dual_expert(request);
    }
}
