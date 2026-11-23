#include "api/generation_program.h"

#include <algorithm>
#include <cmath>

#include "conditioning/prompt_input.h"
#include "core/image_geometry.h"
#include "model/vae/geometry.hpp"
#include "name_conversion.h"
#include "runtime/guidance.h"

namespace sd::api {
    std::string GenerationTemplate::render(const std::string& name, const nlohmann::json& overrides) const {
        nlohmann::json parameters     = fields;
        parameters["flags"]           = flags;
        parameters["vae_run_options"] = vae_options;
        for (auto it = overrides.begin(); it != overrides.end(); ++it)
            parameters[it.key()] = it.value();
        return render_template(name, parameters);
    }

    static std::string variables(const nlohmann::json& values) {
        std::string result;
        for (auto it = values.begin(); it != values.end(); ++it)
            result += "    " + it.key() + " = " + literal(it.value()) + "\n";
        return result;
    }

    std::string text(const char* value) {
        return value ? value : "";
    }

    static std::string module_option(const char* value, SDBackendModule module, bool (*parse)(const std::string&, SDBackendAssignment*, std::string*) = sd_parse_backend_assignment) {
        if (!value || !*value)
            return "";
        SDBackendAssignment assignment;
        std::string error;
        if (!parse(value, &assignment, &error))
            throw dgl::Error(SD_STATUS_INVALID_ARGUMENT, error);
        return assignment.get(module);
    }

    nlohmann::json loader_options(const ContextConfig& settings, SDBackendModule module, const ModelDefinition& definition) {
        nlohmann::json options{{"family", definition.id}};
        for (const auto& [key, value] : std::map<std::string, const char*>{{"backend", settings.backend}, {"params_backend", settings.params_backend}, {"split_mode", settings.split_mode}}) {
            const auto selected = module_option(value, module);
            if (!selected.empty())
                options[key] = selected;
        }
        auto flash = module_option(settings.flash_attn_mode, module, sd_parse_flash_attention_assignment);
        if (flash.empty()) {
            if (settings.flash_attn)
                flash = "on";
            else if (settings.diffusion_flash_attn)
                flash = module == SDBackendModule::DIFFUSION ? "on" : "off";
        }
        if (!flash.empty())
            options["flash_attn_mode"] = flash;
        auto conv = module_option(settings.conv_direct_mode, module, sd_parse_conv_direct_assignment);
        if (conv.empty() && (module == SDBackendModule::VAE ? settings.vae_conv_direct : settings.diffusion_conv_direct))
            conv = "on";
        if (!conv.empty())
            options["conv_direct_mode"] = conv;
        if (settings.wtype != SD_TYPE_COUNT)
            options["wtype"] = sd_type_name(settings.wtype);
        for (const auto& [key, value] : std::map<std::string, const char*>{{"model_args", settings.model_args}, {"tensor_type_rules", settings.tensor_type_rules}})
            if (value && *value)
                options[key] = value;
        if (settings.linear_scale)
            options["linear_scale"] = settings.linear_scale;
        if (settings.attn_scale)
            options["attn_scale"] = settings.attn_scale;
        if (module == SDBackendModule::DIFFUSION && settings.sage_attn)
            options["sage_attn"] = true;
        return options;
    }

    static std::string tiling_options(const TilingConfig& config, bool circular_x, bool circular_y) {
        std::string options;
        if (circular_x)
            options += ", circular_x = true";
        if (circular_y)
            options += ", circular_y = true";
        if (config.enabled || config.temporal_tiling) {
            auto fields = nlohmann::json{{"enabled", config.enabled}, {"temporal_tiling", config.temporal_tiling}, {"tile_size_w", config.tile_size_w}, {"tile_size_h", config.tile_size_h}, {"target_overlap", config.target_overlap}, {"rel_size_w", config.rel_size_w}, {"rel_size_h", config.rel_size_h}};
            if (config.extra_tiling_args && *config.extra_tiling_args)
                fields["extra_args"] = config.extra_tiling_args;
            const auto args = arguments(fields);
            options += ", tiling = tiling(" + args.substr(2) + ")";
        }
        return options;
    }

    static std::string cache_options(const sd_opt_t* options) {
        const auto values = scalar_values(options);
        if (!values.contains("cache.mode") || values.at("cache.mode") == "disabled")
            return "";
        auto fields = nlohmann::json::object();
        for (auto it = values.begin(); it != values.end(); ++it) {
            if (it.key().rfind("cache.", 0) != 0)
                continue;
            const auto name = it.key().substr(6);
            if (name == "fn_compute_blocks" || name == "bn_compute_blocks")
                continue;
            fields[name] = it.value();
        }
        const auto args = arguments(fields);
        return ", cache = cache(" + args.substr(2) + ")";
    }

    void bind_tensor(GenerationProgram& program, const std::string& name, const sd_tensor_t* tensor) {
        if (!tensor)
            throw dgl::Error(SD_STATUS_INVALID_ARGUMENT, "missing host tensor '" + name + "'");
        program.inputs.emplace(name, dgl::Value::object(std::shared_ptr<sd_tensor_t>(sd_tensor_retain(tensor), sd_tensor_free)));
    }

    static std::string input_names(const GenerationProgram& program) {
        std::string names;
        for (const auto& [name, value] : program.inputs)
            names += (names.empty() ? "" : ", ") + name;
        return names;
    }

    struct LoraBindings {
        std::string source;
        std::string model = "base_model";
        std::string text  = "base_text";
    };

    static LoraBindings lora_bindings(const ContextConfig& settings, SDVersion family, const sd_lora_t* loras, uint32_t count) {
        LoraBindings result;
        for (uint32_t i = 0; i < count; ++i) {
            if (loras[i].is_high_noise)
                continue;
            ModelLoader metadata;
            if (!metadata.init_from_file_and_convert_name(text(loras[i].path), "lora.", family))
                throw dgl::Error(SD_STATUS_LOAD_FAILED, "cannot inspect LoRA '" + text(loras[i].path) + "'");
            bool diffusion = false, conditioning = false;
            for (const auto& name : metadata.get_tensor_names()) {
                diffusion |= is_diffusion_model_name(name);
                conditioning |= is_cond_stage_model_name(name);
            }
            if (!diffusion && !conditioning)
                throw dgl::Error(SD_STATUS_UNSUPPORTED, "LoRA has no diffusion or text-encoder targets");
            const auto number = std::to_string(i);
            const auto model = "model_" + number, encoder = "text_" + number;
            result.source += render_template("lora", {{"name", "lora_" + number}, {"path", text(loras[i].path)}, {"diffusion", diffusion}, {"conditioning", conditioning}, {"model_result", model}, {"text_result", encoder}, {"model_source", result.model}, {"text_source", result.text}, {"weight", loras[i].multiplier}, {"mode", sd_lora_apply_mode_name(settings.lora_apply_mode)}});
            if (diffusion)
                result.model = model;
            if (conditioning)
                result.text = encoder;
        }
        return result;
    }

    struct ComponentBindings {
        bool tae;
        nlohmann::json values;
        std::map<std::string, std::string> fields;
    };

    static ComponentBindings component_bindings(const ContextConfig& settings, const ModelDefinition& definition) {
        const auto primary = text(settings.diffusion_model_path).empty() ? text(settings.model_path) : text(settings.diffusion_model_path);
        auto text_path     = text(settings.llm_path);
        if (text_path.empty())
            text_path = text(settings.model_path);
        if (text_path.empty())
            for (const auto* path : {settings.clip_l_path, settings.t5xxl_path, settings.clip_g_path})
                if (path && *path) {
                    text_path = path;
                    break;
                }
        if (text_path.empty())
            throw dgl::Error(SD_STATUS_INVALID_ARGUMENT, "a checkpoint or text encoder path is required");
        const bool mandatory_tae = definition.version == VERSION_SDXS_512_DS || definition.version == VERSION_SDXS_09;
        const bool tae           = settings.taesd_path && *settings.taesd_path && (!settings.tae_preview_only || mandatory_tae);
        auto vae_path            = tae ? text(settings.taesd_path) : text(settings.vae_path);
        if (vae_path.empty())
            vae_path = text(settings.model_path).empty() ? primary : text(settings.model_path);
        auto text_options = loader_options(settings, SDBackendModule::TE, definition);
        for (const auto& [key, path] : std::map<std::string, const char*>{{"tokenizer", settings.tokenizer}, {"vision_path", settings.llm_vision_path}, {"connectors_path", settings.embeddings_connectors_path}, {"t5_path", settings.t5xxl_path}, {"clip_l_path", settings.clip_l_path}, {"clip_g_path", settings.clip_g_path}})
            if (path && *path)
                text_options[key] = path;
        if (sd_version_is_ltxav(definition.version) || definition.version == VERSION_P_H3_IMAGE)
            text_options["projection_path"] = primary;
        auto vae_options = loader_options(settings, SDBackendModule::VAE, definition);
        if (sd_version_is_sdxl(definition.version) && !tae &&
            (settings.force_sdxl_vae_conv_scale || text(settings.vae_path).empty()))
            vae_options["conv_scale"] = 0.03125;
        if (!tae && settings.vae_format != SD_VAE_FORMAT_AUTO) {
            static const char* formats[]{"flux", "sd3", "flux2", "wan"};
            vae_options["vae_format"] = formats[int(settings.vae_format)];
        }
        return {tae,
                {{"model_path", primary}, {"text_path", text_path}, {"vae_path", vae_path}},
                {{"model_loader", sd_version_is_unet(definition.version) ? "load_unet" : "load_dit"},
                 {"model_options", arguments(loader_options(settings, SDBackendModule::DIFFUSION, definition))},
                 {"text_options", arguments(text_options)},
                 {"vae_loader", tae ? "load_tae" : "load_vae"},
                 {"vae_options", arguments(vae_options)}}};
    }

    static GenerationProgram generation_program(GenerationContext& context, const ImageConfig& image, const sd_opt_t* explicit_options, int64_t seed, const VideoConfig* video) {
        auto& model            = context.model;
        const auto& definition = model.definition(context);
        const auto& settings   = model.config.context;
        if (!supports_generation(definition, video != nullptr))
            throw dgl::Error(SD_STATUS_UNSUPPORTED, "this model requires a video or dedicated DGL workflow");
        const auto components = component_bindings(settings, definition);
        const auto method     = image.sample_params.sample_method == SAMPLE_METHOD_COUNT ? definition.sampling.default_sample_method : image.sample_params.sample_method;
        auto scheduler        = image.sample_params.scheduler;
        if (scheduler == SCHEDULER_COUNT) {
            scheduler = (settings.prediction == EDM_V_PRED || (settings.prediction == PREDICTION_COUNT && model.default_prediction == EDM_V_PRED)) ? EXPONENTIAL_SCHEDULER : method == LCM_SAMPLE_METHOD || method == TCD_SAMPLE_METHOD ? LCM_SCHEDULER
                                                                                                                                                                         : method == DDIM_TRAILING_SAMPLE_METHOD                        ? SIMPLE_SCHEDULER
                                                                                                                                                                                                                                        : definition.sampling.default_scheduler;
        }
        const auto predictor_options = arguments({{"prediction", settings.prediction == PREDICTION_COUNT ? "auto" : sd_prediction_name(settings.prediction)},
                                                  {"guidance", image.sample_params.guidance.distilled_guidance},
                                                  {"circular_x", image.circular_x},
                                                  {"circular_y", image.circular_y}});
        const double eta             = std::isfinite(image.sample_params.eta) ? image.sample_params.eta : definition.version == VERSION_HIDREAM_O1                                                                                                                                                                                        ? 8.0
                                                                                                      : method == EULER_A_SAMPLE_METHOD || method == DPMPP2S_A_SAMPLE_METHOD || method == ER_SDE_SAMPLE_METHOD || method == EULER_A_CFG_PP_SAMPLE_METHOD || method == DPMPP2M_SDE_SAMPLE_METHOD || method == DPMPP2M_SDE_BT_SAMPLE_METHOD ? 1.0
                                                                                                                                                                                                                                                                                                                                          : 0.0;
        auto values                  = components.values;
        values.update({{"prompt", text(image.prompt)}, {"negative_prompt", text(image.negative_prompt)}, {"seed", seed}, {"width", image.width}, {"height", image.height}, {"steps", image.sample_params.sample_steps}, {"cfg_scale", image.sample_params.guidance.txt_cfg}, {"method", sd_sample_method_name(method)}, {"scheduler", sd_scheduler_name(scheduler)}, {"eta", eta}, {"rng_type", sd_rng_type_name(settings.rng_type)}});
        const auto vae_version = definition.traits.vae_format_override && !components.tae ? sd::vae::version_for_format(settings.vae_format, definition.version) : definition.version;
        const int scale        = sd_version_is_pid(definition.version) ? 1 : sd::vae::spatial_scale(vae_version);
        const int multiple     = scale * definition.geometry.diffusion_down_factor;
        values["width"]        = sd_version_is_minit2i(definition.version) ? 512 : sd::align_image_dimension(image.width, multiple);
        values["height"]       = sd_version_is_minit2i(definition.version) ? 512 : sd::align_image_dimension(image.height, multiple);
        const auto axes        = tiling_options(image.vae_tiling_params, image.circular_x, image.circular_y);
        auto schedule_options  = nlohmann::json::object();
        if (std::isfinite(image.sample_params.flow_shift))
            schedule_options["flow_shift"] = image.sample_params.flow_shift;
        if (image.sample_params.extra_sample_args && *image.sample_params.extra_sample_args)
            schedule_options["extra_args"] = image.sample_params.extra_sample_args;
        if (image.sample_params.custom_sigmas_count) {
            values["custom_sigmas"] = std::vector<float>(image.sample_params.custom_sigmas, image.sample_params.custom_sigmas + image.sample_params.custom_sigmas_count);
        }
        GenerationProgram program;
        if (video) {
            values["frames"] = sd::vae::aligned_frames(definition.version, video->video_frames);
            values["fps"]    = sd_version_is_minimax_h3(definition.version) ? 24 : definition.version == VERSION_WAN2_2_S2V ? 16
                                                                                                                            : video->fps;
        }
        const bool init = image.init_image.data;
        if (!video && init) {
            bind_tensor(program, "input_image", bound_tensor(explicit_options, "init_image"));
            values["strength"] = image.strength;
        }
        const auto sampler_type     = settings.sampler_rng_type == RNG_TYPE_COUNT ? settings.rng_type : settings.sampler_rng_type;
        std::string preview_decoder = "vae";
        if (settings.tae_preview_only && settings.taesd_path && *settings.taesd_path) {
            auto options           = loader_options(settings, SDBackendModule::VAE, definition);
            options["decode_only"] = true;
            preview_decoder        = "load_tae(" + literal(settings.taesd_path) + arguments(options) + ")";
        }
        auto sample_options = cache_options(explicit_options) + ", observer = preview(decoder = " + preview_decoder + arguments({{"circular_x", image.circular_x}, {"circular_y", image.circular_y}}) + ")";
        if (image.sample_params.extra_sample_args && *image.sample_params.extra_sample_args)
            sample_options += ", extra_args = " + literal(image.sample_params.extra_sample_args);
        const auto to_vae            = sd_version_is_unet(definition.version) ? "unet_to_vae_latents" : "dit_to_vae_latents";
        const auto patches           = lora_bindings(settings, definition.version, image.loras, image.lora_count);
        const auto apg               = sd::guidance::parse_adaptive_projected_guidance_args(image.sample_params.extra_sample_args);
        const bool use_apg           = sd::guidance::is_adaptive_projected_guidance_enabled(apg);
        const auto guidance_schedule = sd::guidance::parse_guidance_schedule(image.sample_params.extra_sample_args);
        auto guidance_options        = nlohmann::json::object();
        if (!guidance_schedule.empty())
            guidance_options["schedule"] = guidance_schedule;
        if (use_apg)
            guidance_options.update({{"eta", apg.eta}, {"momentum", apg.momentum}, {"norm_threshold", apg.norm_threshold}, {"norm_threshold_smoothing", apg.norm_threshold_smoothing}});
        std::map<std::string, std::string> fields{{"patched_model", patches.model},
                                                  {"patched_text", patches.text},
                                                  {"references", ""},
                                                  {"loras", patches.source},
                                                  {"positive_condition", "positive"},
                                                  {"negative_condition", "negative"},
                                                  {"negative_model", "model"},
                                                  {"prompt_options", ", clip_skip = " + std::to_string(image.clip_skip)},
                                                  {"negative_prompt_options", ", clip_skip = " + std::to_string(image.clip_skip)},
                                                  {"negative_prompt_input", "negative_prompt"},
                                                  {"predictor_options", predictor_options},
                                                  {"predictor_core", predictor_options},
                                                  {"reference_predictor_options", ""},
                                                  {"negative_predictor_options", predictor_options},
                                                  {"guidance_options", arguments(guidance_options)},
                                                  {"guidance_core", arguments(guidance_options)},
                                                  {"guidance_function", use_apg ? "apg" : "cfg"},
                                                  {"sample_options", sample_options},
                                                  {"sample_core", sample_options},
                                                  {"result_latent", "latent"},
                                                  {"output_audio_shape", "audio_shape"},
                                                  {"output_video_frames", "tensor::shape(video_base)[2]"},
                                                  {"to_vae", to_vae},
                                                  {"channels", std::to_string(definition.geometry.image_channels)},
                                                  {"from_vae", sd_version_is_unet(definition.version) ? "vae_to_unet_latents" : "vae_to_dit_latents"},
                                                  {"schedule_options", arguments(schedule_options)},
                                                  {"schedule_predictor", "guided"},
                                                  {"audio", "{}"}};
        fields.insert(components.fields.begin(), components.fields.end());
        GenerationTemplate request{context, image, explicit_options, definition, program, values, fields, video, axes, seed};
        if (video)
            prepare_video_program(request);
        else
            prepare_image_program(request);
        prepare_prompt_images(request);
        if (sd_version_is_sensenova_u1(definition.version)) {
            const auto input = sd::parse_prompt_input(values.at("negative_prompt").get<std::string>(), true);
            if (std::any_of(input.begin(), input.end(), [](const auto& part) { return part.type != sd::PromptPart::Type::TEXT; }))
                throw dgl::Error(SD_STATUS_UNSUPPORTED, "SenseNova U1.5 does not support inline prompt images or embeddings");
            values["negative_prompt_raw"] = "<|im_start|>user\n" + sd::prompt_input_text(input) + "<|im_end|>\n<|im_start|>assistant\n<img>";
        }
        request.flags["encode_negative"] = image.sample_params.guidance.txt_cfg != 1.f || use_apg || request.flags.value("image_guidance", false) ||
                                           method == EULER_CFG_PP_SAMPLE_METHOD || method == EULER_A_CFG_PP_SAMPLE_METHOD ||
                                           std::any_of(guidance_schedule.begin(), guidance_schedule.end(), [](float scale) { return scale != 1.f; });
        request.flags["shared_rng"]      = sampler_type == settings.rng_type;
        if (sampler_type != settings.rng_type)
            values["sampler_rng_type"] = sd_rng_type_name(sampler_type);
        prepare_hires_program(request);
        fields["variables"] = variables(values);
        fields["inputs"]    = input_names(program);
        program.source      = request.render(fields.at("entry"));
        return program;
    }

    GenerationProgram image_program(GenerationContext& context, const ImageConfig& image, const sd_opt_t* explicit_options, int64_t seed) {
        return generation_program(context, image, explicit_options, seed, nullptr);
    }

    GenerationProgram video_program(GenerationContext& context, const VideoConfig& video, const sd_opt_t* explicit_options, int64_t seed) {
        ImageConfig common;
        common.loras             = video.loras;
        common.lora_count        = video.lora_count;
        common.prompt            = video.prompt;
        common.negative_prompt   = video.negative_prompt;
        common.clip_skip         = video.clip_skip;
        common.init_image        = video.init_image;
        common.ref_images        = video.ref_images;
        common.ref_images_count  = video.ref_images_count;
        common.width             = video.width;
        common.height            = video.height;
        common.sample_params     = video.sample_params;
        common.strength          = video.strength;
        common.vae_tiling_params = video.vae_tiling_params;
        common.cache             = video.cache;
        common.hires             = video.hires;
        common.circular_x        = video.circular_x;
        common.circular_y        = video.circular_y;
        return generation_program(context, common, explicit_options, seed, &video);
    }

}
