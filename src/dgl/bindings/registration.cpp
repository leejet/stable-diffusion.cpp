#include "dgl/bindings/registration.h"

#include <climits>
#include <limits>

#include "core/util.h"
#include "dgl/bindings/audio.h"
#include "dgl/bindings/component_loader.h"
#include "dgl/bindings/components.h"
#include "dgl/bindings/condition.h"
#include "dgl/bindings/detailer.h"
#include "dgl/bindings/detections.h"
#include "dgl/bindings/image.h"
#include "dgl/bindings/latents.h"
#include "dgl/bindings/lora.h"
#include "dgl/bindings/media.h"
#include "dgl/bindings/model.h"
#include "dgl/bindings/postprocess.h"
#include "dgl/bindings/predictor.h"
#include "dgl/bindings/preview.h"
#include "dgl/bindings/regions.h"
#include "dgl/bindings/relay_vsr.h"
#include "dgl/bindings/rng.h"
#include "dgl/bindings/seedvr2.h"
#include "dgl/bindings/sefi.h"
#include "dgl/bindings/tensor_ops.h"
#include "dgl/bindings/values.h"
#include "dgl/bindings/video.h"
#include "dgl/bindings/video_windows.h"

#include "dgl/bindings/conversion.h"
#include "model_definition.h"
#include "runtime/denoiser.hpp"

namespace sd::dgl {

    static Argument backend_option(const char* name) {
        auto argument     = optional(name, Kind::String, "");
        argument.validate = [name](const Value& value) {
            const auto& spec = value.scalar.get_ref<const std::string&>();
            if (spec.find_first_of("=,") != std::string::npos)
                throw std::runtime_error(std::string(name) + " expects a value for this model");
        };
        return argument;
    }

    static Argument operator_mode(const char* name) {
        auto argument     = optional(name, Kind::String, "auto");
        argument.validate = [name](const Value& value) {
            const auto mode = trim(value.scalar.get_ref<const std::string&>());
            if (mode != "auto" && mode != "on" && mode != "off")
                throw std::runtime_error(std::string(name) + " expects auto, on, or off");
        };
        return argument;
    }

    static Operation configuration(std::vector<Argument> arguments) {
        OutputSignature output{Kind::Dict};
        for (const auto& argument : arguments)
            output.fields.emplace(argument.name, argument.type);
        return make_operation(std::move(arguments), std::move(output), execute_configuration);
    }

    static Argument int_vector_argument(const std::string& name) {
        return {name, {Kind::IntVector}, true, std::nullopt, [](const Value& value) {
                    if (value.type().kind == Kind::Tensor)
                        i64_vector(value);
                }};
    }

    static void register_value_operations(Registry& registry) {
        registry["list::replace"] = make_operation(
            {list_argument("items", Kind::Any), integer("index", 0, INT64_MAX), required("value", Kind::Any)},
            {Kind::List}, execute_list_replace);
        registry["math::product"] = make_operation(
            {int_vector_argument("items")}, {Kind::Integer}, execute_math_product);
        registry["math::ceil_div"] = make_operation(
            {integer("numerator", 0, INT64_MAX), integer("denominator", 1, INT64_MAX)},
            {Kind::Integer}, execute_math_ceil_div);
        const std::vector<Argument> integers{required("a", Kind::Integer), required("b", Kind::Integer)};
        registry["math::min"] = make_operation(
            integers, {Kind::Integer}, execute_math_min);
        const std::vector<Argument> numbers{required("a", Kind::Number), required("b", Kind::Number)};
        registry["math::add"] = make_operation(
            numbers, {Kind::Number}, execute_math_add);
        registry["math::sub"] = make_operation(
            numbers, {Kind::Number}, execute_math_sub);
        registry["math::mul"] = make_operation(
            {required("a", Kind::Number), required("b", Kind::Number)}, {Kind::Number}, execute_math_mul);
        registry["math::div"] = make_operation(
            {required("a", Kind::Number), required("b", Kind::Number)}, {Kind::Number}, execute_math_div);
        const std::vector<Argument> rounding_args{required("value", Kind::Number)};
        registry["math::floor"] = make_operation(rounding_args, {Kind::Integer}, execute_math_floor);
        registry["math::ceil"]  = make_operation(rounding_args, {Kind::Integer}, execute_math_ceil);
        registry["math::round"] = make_operation(rounding_args, {Kind::Integer}, execute_math_round);
        registry["tiling"]      = configuration({
            optional("enabled", Kind::Boolean, true),
            optional("temporal_tiling", Kind::Boolean, false),
            integer("tile_size_w", 0, 16384, 0),
            integer("tile_size_h", 0, 16384, 0),
            number("target_overlap", 0, 0.99, 0.5),
            number("rel_size_w", 0, 16384, 0),
            number("rel_size_h", 0, 16384, 0),
            optional("extra_args", Kind::String, ""),
        });
        std::vector<Argument> cache_args{choice("mode", {"disabled", "easycache", "ucache", "dbcache", "taylorseer", "cache_dit", "spectrum"}, "disabled")};
        auto cache_arg = [&](Argument argument) { argument.required=false; cache_args.push_back(std::move(argument)); };
        cache_arg(number("reuse_threshold", 0, 100000));
        cache_arg(number("start_percent", 0, 1));
        cache_arg(number("end_percent", 0, 1));
        cache_arg(number("error_decay_rate", 0, 100000));
        cache_arg(optional("use_relative_threshold", Kind::Boolean));
        cache_arg(optional("reset_error_on_compute", Kind::Boolean));
        cache_arg(number("residual_diff_threshold", 0, 100000));
        cache_arg(integer("max_warmup_steps", 0, 100000));
        cache_arg(integer("max_cached_steps", INT32_MIN, INT32_MAX));
        cache_arg(integer("max_continuous_cached_steps", INT32_MIN, INT32_MAX));
        cache_arg(integer("taylorseer_n_derivatives", 0, 100000));
        cache_arg(integer("taylorseer_skip_interval", 0, 100000));
        cache_arg(optional("scm_mask", Kind::String));
        cache_arg(number("spectrum_w", 0, 100000));
        cache_arg(integer("spectrum_m", 0, 100000));
        cache_arg(number("spectrum_lam", 0, 100000));
        cache_arg(integer("spectrum_window_size", 0, 100000));
        cache_arg(number("spectrum_flex_window", 0, 100000));
        cache_arg(integer("spectrum_warmup_steps", 0, 100000));
        cache_arg(number("spectrum_stop_percent", 0, 1));
        registry["cache"] = configuration(std::move(cache_args));
    }

    static void register_rng_operations(Registry& registry) {
        registry["make_rng"] = make_operation(
            {choice("algorithm", {"cuda", "cpu", "std", "std_default"}, "cuda"), integer("seed", 0, INT64_MAX, 0)}, {Kind::Rng},
            execute_make_rng);
    }

    static void register_tensor_operations(Registry& registry) {
        const auto tensor        = required("input", Kind::Tensor);
        const auto shape         = int_vector_argument("shape");
        const auto axis          = integer("axis", 0, 7);
        const auto dtype         = choice("dtype", {"u8", "i32", "i64", "f32", "f64"});
        const auto default_dtype = choice("dtype", {"u8", "i32", "i64", "f32", "f64"}, "f32");
        const auto start         = integer("start", 0, INT64_MAX);
        const auto scalar        = [](const char* name) { return required(name, Kind::Number); };
        registry["tensor::load"] = make_operation(
            {required("path", Kind::String)}, {Kind::Tensor}, execute_load_tensor);
        registry["tensor::save"] = make_operation(
            {required("path", Kind::String), required("tensor", Kind::Tensor)}, {Kind::Void}, execute_save_tensor);
        registry["tensor::literal"] = make_operation(
            {dtype, shape, required("data", Kind::String)}, {Kind::Tensor}, execute_tensor_literal);
        registry["tensor::shape"] = make_operation(
            {tensor}, {Kind::Tensor}, execute_tensor_shape);
        registry["tensor::index"] = make_operation(
            {tensor, list_argument("indices", Kind::Dict)}, {Kind::Any}, execute_tensor_index);
        registry["tensor::vector"] = make_operation(
            {list_argument("values", Kind::Number), default_dtype}, {Kind::Tensor}, execute_tensor_vector);
        registry["tensor::to_list"] = make_operation(
            {tensor, choice("element", {"auto", "int", "number"}, "auto")}, {Kind::List}, execute_tensor_to_list);
        registry["tensor::reshape"] = make_operation(
            {tensor, shape}, {Kind::Tensor}, execute_tensor_reshape);
        registry["tensor::unsqueeze"] = make_operation(
            {tensor, integer("axis", -8, 7)}, {Kind::Tensor}, execute_tensor_unsqueeze);
        registry["tensor::pad"] = make_operation(
            {tensor, int_vector_argument("pad"), choice("mode", {"constant"}, "constant"), optional("value", Kind::Number, 0)},
            {Kind::Tensor}, execute_tensor_pad);
        registry["tensor::permute"] = make_operation(
            {tensor, int_vector_argument("axes")}, {Kind::Tensor}, execute_tensor_permute);
        registry["tensor::crop"] = make_operation(
            {tensor, int_vector_argument("axes"), int_vector_argument("start"), int_vector_argument("size")},
            {Kind::Tensor}, execute_tensor_crop);
        registry["tensor::numel"] = make_operation(
            {tensor}, {Kind::Integer}, execute_tensor_numel);
        registry["tensor::count_nonzero"] = make_operation(
            {tensor}, {Kind::Integer}, execute_tensor_count_nonzero);
        registry["tensor::replace_slice"] = make_operation(
            {tensor, required("value", Kind::Tensor), axis, start}, {Kind::Tensor},
            replace_slice);
        registry["tensor::concat"] = make_operation(
            {list_argument("items", Kind::Tensor), axis}, {Kind::Tensor},
            concat_tensors);
        registry["tensor::cast"] = make_operation(
            {tensor, dtype}, {Kind::Tensor}, cast_tensor);
        registry["tensor::broadcast_to"] = make_operation(
            {tensor, shape}, {Kind::Tensor}, execute_tensor_broadcast_to);
        registry["tensor::zeros"] = make_operation(
            {shape, default_dtype}, {Kind::Tensor}, execute_tensor_zeros);
        registry["tensor::full"] = make_operation(
            {shape, scalar("value"), default_dtype}, {Kind::Tensor}, execute_tensor_full);
        registry["tensor::zeros_like"] = make_operation(
            {required("like", Kind::Tensor)}, {Kind::Tensor}, execute_tensor_zeros_like);
        registry["tensor::arange"] = make_operation(
            {integer("start", INT64_MIN, INT64_MAX), integer("end", INT64_MIN, INT64_MAX), integer("step", INT64_MIN, INT64_MAX, 1)},
            {Kind::Tensor}, arange_tensor);
        registry["tensor::index_select"] = make_operation(
            {tensor, axis, int_vector_argument("indices")}, {Kind::Tensor}, index_select_tensor);
        registry["tensor::randn_like"] = make_operation(
            {required("like", Kind::Tensor), required("rng", Kind::Rng)},
            {Kind::Tensor}, execute_tensor_randn_like);
        const std::vector<Argument> binary_args{required("a", Kind::Any), required("b", Kind::Any)};
        registry["tensor::add"]  = make_operation(binary_args, {Kind::Tensor}, execute_tensor_add);
        registry["tensor::mul"]  = make_operation(binary_args, {Kind::Tensor}, execute_tensor_mul);
        registry["tensor::div"]  = make_operation(binary_args, {Kind::Tensor}, execute_tensor_div);
        registry["tensor::lerp"] = make_operation(
            {required("a", Kind::Any), required("b", Kind::Any), required("weight", Kind::Any)}, {Kind::Tensor},
            execute_tensor_lerp);
        const auto numeric_limit   = std::numeric_limits<double>::max();
        registry["tensor::affine"] = make_operation(
            {tensor, number("scale", -numeric_limit, numeric_limit, 1), number("offset", -numeric_limit, numeric_limit, 0)},
            {Kind::Tensor}, execute_tensor_affine);
        registry["tensor::clamp"] = make_operation(
            {tensor, scalar("min"), scalar("max")}, {Kind::Tensor}, execute_tensor_clamp);
        registry["tensor::translate"] = make_operation(
            {tensor, int_vector_argument("axes"), int_vector_argument("offset"), number("value", -numeric_limit, numeric_limit, 0)},
            {Kind::Tensor}, translate_tensor);
        registry["tensor::index_fill"] = make_operation(
            {tensor, axis, int_vector_argument("indices"), scalar("value")}, {Kind::Tensor}, execute_tensor_index_fill);
        registry["tensor::grayscale"] = make_operation(
            {tensor, integer("axis", 0, 7, 0)}, {Kind::Tensor}, grayscale_tensor);
        registry["tensor::threshold"] = make_operation(
            {tensor, scalar("threshold"), number("low", -numeric_limit, numeric_limit, 0), number("high", -numeric_limit, numeric_limit, 1)},
            {Kind::Tensor}, threshold_tensor);
        registry["tensor::maximum"] = make_operation(
            {required("a", Kind::Tensor), required("b", Kind::Tensor)}, {Kind::Tensor}, maximum_tensor);
        const auto optional_vector = [](const char* name) {
            auto argument     = int_vector_argument(name);
            argument.required = false;
            return argument;
        };
        const std::vector<Argument> pool_args{
            tensor,
            int_vector_argument("kernel"),
            optional_vector("axes"),
            optional_vector("stride"),
            optional_vector("padding_before"),
            optional_vector("padding_after"),
            optional("pad_value", Kind::Number),
        };
        registry["tensor::max_pool2d"] = make_operation(
            pool_args, {Kind::Tensor}, execute_tensor_max_pool2d);
        registry["tensor::min_pool2d"] = make_operation(
            pool_args, {Kind::Tensor}, execute_tensor_min_pool2d);
        registry["tensor::gaussian_blur"] = make_operation(
            {tensor, number("sigma", 0, numeric_limit), integer("radius", 0, 16384), optional_vector("axes"),
             choice("precision", {"auto", "f32", "f64"}, "auto")},
            {Kind::Tensor}, blur_tensor);
        auto axes                       = int_vector_argument("axes");
        axes.required                   = false;
        registry["tensor::interpolate"] = make_operation(
            {tensor, int_vector_argument("size"), std::move(axes),
             choice("mode", {"nearest", "nearest_exact", "nearest_max", "bilinear", "bicubic", "lanczos"}, "bilinear"),
             optional("antialias", Kind::Boolean, false)},
            {Kind::Tensor}, execute_tensor_interpolate);
    }

    static void register_image_operations(Registry& registry) {
        registry["image::to_f32"] = make_operation(
            {required("image", Kind::Tensor), integer("channels", 0, 4, 0)}, {Kind::Tensor}, execute_image_to_f32);
        registry["image::to_u8"] = make_operation(
            {required("pixels", Kind::Tensor)}, {Kind::Tensor}, execute_image_to_u8);
        registry["image::load"] = make_operation(
            {required("path", Kind::String)}, {Kind::Tensor}, execute_load_image);
        registry["image::load_dir"] = make_operation(
            {required("path", Kind::String), integer("max_images", 0, 100000, 0)},
            {Kind::List, Kind::Tensor}, execute_load_images);
        registry["image::save"] = make_operation(
            {required("path", Kind::String), required("image", Kind::Tensor)}, {Kind::Void}, execute_save_image);
        registry["image::encode"] = make_operation(
            {required("image", Kind::Tensor), choice("format", {"png"}, "png")}, media_signature(), execute_encode_image);
        registry["image::canny"] = make_operation(
            {
                required("image", Kind::Tensor),
                number("high_threshold", 0, 1, 0.08),
                number("low_threshold", 0, 1, 0.04),
                number("weak", 0, 1, 0.5),
                number("strong", 0, 1, 1),
                optional("invert", Kind::Boolean, false),
            },
            {Kind::Tensor},
            execute_canny);
    }

    static void register_region_operations(Registry& registry) {
        registry["regions::from_boxes"] = make_operation(
            {integer("width", 1, 16384), integer("height", 1, 16384), required("boxes", Kind::Any)},
            regions_signature(), regions_from_boxes);
        registry["regions::crop"] = make_operation(
            {required("image", Kind::Tensor), required("regions", Kind::Dict)},
            {Kind::List, Kind::Tensor}, crop_regions);
        registry["regions::paste"] = make_operation(
            {required("background", Kind::Tensor), list_argument("foregrounds", Kind::Tensor), required("regions", Kind::Dict)},
            {Kind::Tensor}, paste_regions);
    }

    static void register_video_operations(Registry& registry) {
        registry["video::encode"] = make_operation(
            {required("video", Kind::Dict), optional("audio", Kind::Dict), choice("format", {"webm", "avi"}, "webm")},
            media_signature(), execute_encode_video);
        registry["video::load"] = make_operation(
            {
                required("path", Kind::String),
                integer("max_frames", 0, 100000, 0),
                integer("fps_num", 0, 1000000, 0),
                integer("fps_den", 1, 1000000, 1),
                optional("audio", Kind::Boolean, true),
            },
            {{Kind::Dict}, {{"video", video_signature()}, {"audio", audio_signature()}}},
            execute_load_video);
        registry["video::save"] = make_operation(
            {
                required("path", Kind::String),
                required("video", Kind::Dict),
                optional("audio", Kind::Dict),
                optional("trim_audio", Kind::Boolean, false),
                choice("codec", {"auto", "mjpeg", "rgb"}, "auto"),
                integer("quality", 1, 90, 90),
                choice("audio_format", {"f32", "pcm16"}, "f32"),
            },
            {Kind::Void},
            execute_save_video);
        registry["video::split_windows"] = make_operation(
            {required("frames", Kind::Tensor), integer("size", 1, 4097), integer("overlap", 0, 4096, 0)},
            {Kind::List, Kind::Tensor}, split_video_frames);
        registry["video::join_windows"] = make_operation(
            {list_argument("windows", Kind::Tensor), integer("overlap", 0, 4096, 0)},
            {Kind::Tensor}, join_video_windows);
    }

    static void register_audio_operations(Registry& registry) {
        registry["audio::load"] = make_operation(
            {
                required("path", Kind::String),
            },
            audio_signature(),
            execute_load_audio);
        registry["audio::save"] = make_operation(
            {
                required("path", Kind::String),
                required("audio", Kind::Dict),
                choice("format", {"f32", "pcm16"}, "f32"),
            },
            {Kind::Void},
            execute_save_audio);
    }

    static void register_model_operations(Registry& registry) {
        registry["load_upscaler"] = make_resource_operation(
            {
                model_path("path"),
                integer("tile_size", 0, 16384, 128),
                backend_option("backend"),
                backend_option("params_backend"),
                optional("direct", Kind::Boolean, false),
            },
            {Kind::Upscaler},
            execute_load_upscaler);
        auto scale_hint     = integer("factor", 1, 16);
        scale_hint.required = false;
        registry["upscale"] = make_operation(
            {
                required("upscaler", Kind::Upscaler),
                required("image", Kind::Tensor),
                std::move(scale_hint),
            },
            {Kind::Tensor},
            execute_upscale);
        registry["load_detector"] = make_resource_operation(
            {
                model_path("path"),
                backend_option("backend"),
                backend_option("params_backend"),
            },
            {Kind::Detector},
            execute_load_detector);
    }

    template <class T, class Load>
    static Operation component_loader(const Argument& family, Load load, std::vector<Argument> options = {}, bool require_path = true) {
        std::vector<Argument> args{
            model_path("path", require_path),
            family,
            optional("wtype", Kind::String, "auto"),
            number("conv_scale", 0, 100000, 0),
            number("linear_scale", 0, 100000, 0),
            number("attn_scale", 0, 100000, 0),
        };
        for (const char* name : {"backend", "params_backend", "split_mode"})
            args.push_back(backend_option(name));
        for (const char* name : {"model_args", "tensor_type_rules"})
            args.push_back(optional(name, Kind::String, ""));
        args.push_back(operator_mode("flash_attn_mode"));
        args.push_back(operator_mode("conv_direct_mode"));
        args.push_back(optional("sage_attn", Kind::Boolean, false));
        for (auto& option : options)
            args.push_back(std::move(option));
        return make_resource_operation(std::move(args), {ObjectKind<T>::value}, load);
    }

    static void register_lora_operations(Registry& registry) {
        registry["load_lora"] = make_resource_operation({model_path("path")}, {Kind::Lora}, load_lora);
        OutputSignature output{Kind::Model};
        output.same_type_as    = "model";
        auto mode              = choice("mode", {"auto", "immediately", "at_runtime"});
        mode.required          = false;
        registry["apply_lora"] = make_operation(
            {required("model", Kind::Model), required("lora", Kind::Lora), optional("weight", Kind::Number, 1.0), std::move(mode)},
            std::move(output), apply_lora);
    }

    static void register_component_operations(Registry& registry) {
        std::vector<std::string> family_ids{"auto"};
        for (int version = 0; version < VERSION_COUNT; ++version)
            family_ids.emplace_back(find_model_definition(static_cast<SDVersion>(version))->id);
        const auto family      = choice("family", std::move(family_ids), "auto");
        const auto family_from = model_path("family_from", false);
        register_lora_operations(registry);

        registry["load_dit"] = component_loader<models::DiffusionModel>(family, load_dit);
        registry.alias("load_unet", "load_dit");

        auto tokenizer                = optional("tokenizer", Kind::String, "");
        registry["load_text_encoder"] = component_loader<models::TextEncoder>(
            family,
            load_text_encoder,
            {
                family_from,
                tokenizer,
                model_path("vision_path", false),
                model_path("projection_path", false),
                model_path("connectors_path", false),
                model_path("t5_path", false),
                model_path("clip_l_path", false),
                model_path("clip_g_path", false),
            });
        registry["load_vae"] = component_loader<models::VAE>(
            family,
            load_vae,
            {
                family_from,
                choice("vae_format", {"auto", "flux", "sd3", "flux2", "wan"}, "auto"),
            });
        registry["load_tae"] = component_loader<models::VAE>(
            family, load_tae, {family_from, optional("decode_only", Kind::Boolean, false)});
        registry["controlnet::load"]     = component_loader<models::ControlNet>(family, load_controlnet);
        registry["load_vision_encoder"]  = component_loader<models::VisionEncoder>(family, load_vision_encoder);
        registry["load_audio_encoder"]   = component_loader<models::AudioEncoder>(family, load_audio_encoder);
        registry["load_audio_vae"]       = component_loader<models::AudioVAE>(family, load_audio_vae, {family_from});
        registry["ip_adapter::load"]     = component_loader<models::IPAdapter>(family, load_ip_adapter);
        registry["photomaker::load"]     = component_loader<models::PhotoMaker>(family, load_photomaker);
        registry["pulid::load"]          = component_loader<models::PuLID>(family, load_pulid);
        registry["animatediff::load"]    = component_loader<models::AnimateDiff>(family, load_animatediff);
        registry["load_latent_upscaler"] = component_loader<models::LatentUpscaler>(
            choice("family", {"auto", "ltxav", "minimax_h3"}, "auto"), load_latent_upscaler);
        registry["pulid::load_embedding"] = make_operation(
            {required("path", Kind::String)}, {Kind::Tensor}, execute_load_pulid_embedding);
        registry["photomaker::encode_images"] = make_operation(
            {required("photomaker", Kind::PhotoMaker),
             list_argument("images", Kind::Tensor),
             optional("id_embedding", Kind::Tensor)},
            {Kind::Tensor},
            photomaker_tokens);
        registry["photomaker::encode_prompt"] = make_operation(
            {required("encoder", Kind::TextEncoder), required("text", Kind::String), required("tokens", Kind::Tensor),
             integer("width", 1, 16384, 1024), integer("height", 1, 16384, 1024), integer("clip_skip", -1, 1024, -1)},
            {{Kind::Dict}, {{"plain", condition_signature()}, {"condition", condition_signature()}, {"mask", {Kind::Tensor}}}}, encode_photomaker_prompt);
        OutputSignature photomaker_condition{{Kind::Dict}, {{"crossattn", {Kind::Tensor}}}};
        photomaker_condition.same_type_as = "condition";
        registry["photomaker::apply"]     = make_operation(
            {required("condition", Kind::Dict), required("photomaker", Kind::PhotoMaker),
             required("tokens", Kind::Tensor), required("mask", Kind::Tensor)},
            std::move(photomaker_condition), apply_photomaker);
        registry["encode_prompt"] = make_operation(
            {required("encoder", Kind::TextEncoder), required("text", Kind::String),
             integer("width", 1, 16384, 512), integer("height", 1, 16384, 512),
             integer("clip_skip", INT32_MIN, INT32_MAX, -1), optional("system", Kind::String),
             optional("media", Kind::List), optional("image_bindings", Kind::Dict), optional("raw", Kind::Boolean, false)},
            condition_signature(),
            encode_prompt);
        OutputSignature zeroed_condition{Kind::Dict};
        zeroed_condition.same_type_as = "condition";
        auto zero_fields              = list_argument("fields", Kind::String);
        zero_fields.required          = false;
        registry["zero_out"]          = make_operation(
            {required("condition", Kind::Dict), std::move(zero_fields)},
            std::move(zeroed_condition), zero_out_condition);
        registry["vae_encode"] = make_operation(
            {required("vae", Kind::VAE), required("pixels", Kind::Tensor), required("rng", Kind::Rng),
             optional("tiling", Kind::Dict), optional("circular_x", Kind::Boolean, false), optional("circular_y", Kind::Boolean, false),
             choice("posterior", {"sample", "mode"}, "sample")},
            {Kind::Tensor}, vae_encode);
        registry["vae_decode"] = make_operation(
            {required("vae", Kind::VAE), required("latent", Kind::Tensor), optional("tiling", Kind::Dict),
             optional("circular_x", Kind::Boolean, false), optional("circular_y", Kind::Boolean, false),
             choice("dtype", {"u8", "f32"}, "u8")},
            {Kind::Tensor}, vae_decode);
        const auto conversion_args     = {required("vae", Kind::VAE), required("latent", Kind::Tensor)};
        registry["vae_to_dit_latents"] = make_operation(
            conversion_args, {Kind::Tensor}, execute_vae_to_dit_latents);
        registry["dit_to_vae_latents"] = make_operation(
            conversion_args, {Kind::Tensor}, execute_dit_to_vae_latents);
        registry.alias("vae_to_unet_latents", "vae_to_dit_latents");
        registry.alias("unet_to_vae_latents", "dit_to_vae_latents");
        registry["latent_shape"] = make_operation(
            {required("vae", Kind::VAE), integer("width", 1, 16384), integer("height", 1, 16384),
             integer("batch", 1, 65536, 1), integer("frames", 1, 65536)},
            {Kind::Tensor}, latent_shape);
        registry["latent_shape"].arguments.back().required = false;
        auto sequence_length                               = integer("sequence_length", 1, INT_MAX);
        sequence_length.required                           = false;
        std::vector<std::string> schedulers;
        for (int i = 0; i < SCHEDULER_COUNT; ++i)
            schedulers.emplace_back(sd_scheduler_name(static_cast<scheduler_t>(i)));
        registry["make_sigmas"] = make_operation(
            {required("predict", Kind::Predictor), required("latent", Kind::Tensor), choice("scheduler", std::move(schedulers), "discrete"),
             integer("steps", 1, 10000), number("flow_shift", 0, 10000, 0), sequence_length,
             optional("extra_args", Kind::String, "")},
            {Kind::Tensor}, make_sigmas);
        registry["t_to_sigma"] = make_operation(
            {required("predict", Kind::Predictor), number("t", 0, 1), number("flow_shift", 0, 10000, 0),
             optional("latent", Kind::Tensor), std::move(sequence_length)},
            {Kind::Number}, t_to_sigma);
        const auto sigma_limit = std::numeric_limits<float>::max();
        const std::map<std::string, Type> scaling_fields{
            {"input_scale", {Kind::Number}}, {"skip_scale", {Kind::Number}}, {"output_scale", {Kind::Number}}};
        OutputSignature scaling_output{Kind::Dict};
        for (const auto& [name, type] : scaling_fields)
            scaling_output.fields.emplace(name, type);
        registry["denoiser_scalings"] = make_operation(
            {required("predict", Kind::Predictor), number("sigma", 0, sigma_limit)},
            std::move(scaling_output), denoiser_scalings);
        auto start_sigma     = number("start_sigma", 0, sigma_limit);
        auto end_sigma       = number("end_sigma", 0, sigma_limit);
        start_sigma.required = end_sigma.required = false;
        registry["add_noise"]                     = make_operation(
            {required("predict", Kind::Predictor), required("base", Kind::Tensor), required("noise", Kind::Tensor),
             number("sigma", 0, 100000)},
            {Kind::Tensor}, execute_add_noise);
        auto optional_list = [](const char* name, Kind type) {
            auto arg     = list_argument(name, type);
            arg.required = false;
            return arg;
        };
        registry["sefi::make_sigmas"] = make_operation(
            {integer("steps", 1, 10000), number("alpha", 0.001, 100, 1), number("delta", 0, 1, 0.1)},
            {{Kind::Dict}, {{"sigmas", {Kind::Tensor}}, {"semantic_sigmas", {Kind::Tensor}}}}, sefi_sigmas);
        OutputSignature controlled_condition{{Kind::Dict}, {{"controls", {Kind::ControlBindings}}}};
        controlled_condition.same_type_as = "condition";
        registry["controlnet::apply"]     = make_operation(
            {required("condition", Kind::Dict), required("model", Kind::ControlNet), required("hint", Kind::Tensor),
             number("strength", 0, 100, 1), start_sigma, end_sigma,
             optional("type", Kind::String, "none")},
            std::move(controlled_condition), apply_controlnet);
        OutputSignature adapted_condition{{Kind::Dict}, {{"ip_adapters", {Kind::IPAdapterBindings}}}};
        adapted_condition.same_type_as = "condition";
        registry["ip_adapter::apply"]  = make_operation(
            {required("condition", Kind::Dict), required("adapter", Kind::IPAdapter), required("tokens", Kind::Tensor),
             number("strength", 0, 100, 1), start_sigma, end_sigma},
            std::move(adapted_condition), apply_ip_adapter);
        OutputSignature pulid_condition{{Kind::Dict}, {{"pulid", {Kind::PuLIDBindings}}}};
        pulid_condition.same_type_as = "condition";
        registry["pulid::apply"]     = make_operation(
            {required("condition", Kind::Dict), required("pulid", Kind::PuLID), required("embedding", Kind::Tensor),
             number("strength", 0, 100, 1), start_sigma, end_sigma},
            std::move(pulid_condition), apply_pulid);
        registry["animatediff::apply"] = make_operation(
            {required("model", Kind::Denoiser), required("motion", Kind::AnimateDiff)},
            {Kind::Denoiser}, apply_animatediff);
        auto transform_fields = scaling_fields;
        transform_fields.emplace("timestep", Type{Kind::Number});
        auto step_fields = transform_fields;
        step_fields.emplace("sigma", Type{Kind::Number});
        auto transform     = optional("transform", Kind::Function);
        transform.callback = CallbackSignature{"transform_args", std::move(step_fields), std::move(transform_fields)};
        std::vector<std::string> prediction_ids{"auto"};
        for (int i = 0; i < PREDICTION_COUNT; ++i)
            prediction_ids.emplace_back(sd_prediction_name(static_cast<prediction_t>(i)));
        registry["predictor"] = make_operation(
            {required("model", Kind::Denoiser), required("condition", Kind::Dict), choice("prediction", std::move(prediction_ids), "auto"), optional("concat", Kind::Tensor),
             optional_list("ref_latents", Kind::Tensor), optional_list("ref_audios", Kind::Tensor),
             optional_list("reference_blocks", Kind::Dict), optional_list("keyframe_indices", Kind::Integer),
             choice("ref_index_mode", {"fixed", "increase", "decrease"}, "fixed"), optional("force_ref_timestep_zero", Kind::Boolean, false),
             choice("ref_rope_anchor", {"default", "image_start", "text_start"}, "default"),
             number("guidance", 0, 1000, 3.5), optional_list("skip_layers", Kind::Integer),
             optional("vace_context", Kind::Tensor), number("vace_strength", 0, 100, 1),
             optional("audio_embedding", Kind::Tensor), optional("video_positions", Kind::Tensor),
             integer("audio_length", 0, INT_MAX, 0),
             number("fps", 0.01, 100000, 24), number("video_sigma_shift", 0.001, 1000, 12), number("audio_sigma_shift", 0.001, 1000, 3),
             optional("circular_x", Kind::Boolean, false), optional("circular_y", Kind::Boolean, false),
             std::move(transform), optional("transform_args", Kind::Dict)},
            {Kind::Predictor}, make_predictor);
        const std::vector<Argument> guidance_branches{
            required("positive", Kind::Predictor), required("negative", Kind::Predictor), number("scale", 0, 1000),
            optional("image_only", Kind::Predictor), number("image_scale", 0, 1000, 1), optional_list("schedule", Kind::Number)};
        registry["cfg"] = make_operation(guidance_branches, {Kind::Predictor}, make_cfg);
        auto apg        = guidance_branches;
        apg.insert(apg.end(),
                   {number("eta", -sigma_limit, sigma_limit, 1), number("momentum", -sigma_limit, sigma_limit, 0),
                    number("norm_threshold", 0, sigma_limit, 0), number("norm_threshold_smoothing", 0, sigma_limit, 0)});
        registry["apg"]              = make_operation(std::move(apg), {Kind::Predictor}, make_apg);
        registry["switch_predictor"] = make_operation(
            {required("before", Kind::Predictor), required("after", Kind::Predictor), integer("after_steps", 0, 10000)},
            {Kind::Predictor}, switch_predictor);
        auto start_step_fraction     = number("start_step_fraction", 0, 1);
        auto stop_step_fraction      = number("stop_step_fraction", 0, 1);
        start_step_fraction.required = stop_step_fraction.required = false;
        registry["slg"]                                            = make_operation(
            {required("base", Kind::Predictor), required("positive", Kind::Predictor), required("skipped", Kind::Predictor),
             number("scale", 0, 1000), start_sigma, end_sigma, std::move(start_step_fraction), std::move(stop_step_fraction)},
            {Kind::Predictor}, make_slg);
        registry["preview"] = make_operation(
            {optional("decoder", Kind::VAE), optional("circular_x", Kind::Boolean, false), optional("circular_y", Kind::Boolean, false)},
            {Kind::SamplingObserver}, make_preview);
        std::vector<std::string> sample_methods;
        for (int i = 0; i < SAMPLE_METHOD_COUNT; ++i)
            sample_methods.emplace_back(sd_sample_method_name(static_cast<sample_method_t>(i)));
        std::vector<Argument> sampling_arguments{
            required("start", Kind::Tensor), required("sigmas", Kind::Tensor), required("predict", Kind::Predictor),
            required("rng", Kind::Rng), choice("method", std::move(sample_methods), "euler"), number("eta", 0, 100, 1),
            optional("base", Kind::Tensor), optional("denoise_mask", Kind::Tensor), optional("mask_input", Kind::Boolean, false),
            optional("extra_args", Kind::String, ""),
            optional("finish", Kind::Boolean),
            optional("cache", Kind::Dict), optional("observer", Kind::SamplingObserver)};
        registry["sample"]          = make_operation(sampling_arguments, {Kind::Tensor}, sample_tensor);
        registry["sample"].sampling = true;
        sampling_arguments.insert(sampling_arguments.begin() + 2, required("semantic_sigmas", Kind::Tensor));
        registry["sefi::sample"]          = make_operation(std::move(sampling_arguments), {Kind::Tensor}, sample_sefi_tensor);
        registry["sefi::sample"].sampling = true;
    }

    static void register_latent_operations(Registry& registry) {
        registry["audio_latent_shape"] = make_operation(
            {required("vae", Kind::AudioVAE), integer("frames", 1, 65536), integer("fps", 1, 1000)},
            {Kind::Tensor}, execute_audio_latent_shape);
        registry["ltx::video_positions"] = make_operation(
            {required("like", Kind::Tensor), integer("keyframe_latent_frames", 1, 65536),
             integer("keyframe_frame_index", 0, 65536), integer("keyframe_pixel_frames", 1, 65536),
             integer("fps", 1, 1000, 24), integer("spatial_scale", 1, 128, 32),
             integer("temporal_scale", 1, 128, 8), optional("causal", Kind::Boolean, true)},
            {Kind::Tensor}, execute_ltx_video_positions);
        registry["wan::s2v_audio_window"] = make_operation(
            {required("embedding", Kind::Tensor), integer("latent_frames", 1, 65536)}, {Kind::Tensor}, execute_s2v_audio_window);
        registry["encode_vision"] = make_operation(
            {required("encoder", Kind::VisionEncoder), required("image", Kind::Tensor), optional("pooled", Kind::Boolean, false),
             integer("clip_skip", -16, 1024, -2), optional("zero", Kind::Boolean, false),
             optional("zero_input", Kind::Boolean, false)},
            {Kind::Tensor}, encode_vision);
        registry["encode_audio"] = make_operation(
            {required("encoder", Kind::AudioEncoder), required("samples", Kind::Tensor), integer("sample_rate", 1000, 384000)}, {Kind::Tensor}, encode_audio);
        registry["ip_adapter::project"] = make_operation(
            {required("adapter", Kind::IPAdapter), required("embedding", Kind::Tensor)}, {Kind::Tensor}, encode_ip_adapter);
        auto width = integer("width", 1, 16384), height = integer("height", 1, 16384);
        width.required = height.required = false;
        registry["upscale_latent"]       = make_operation(
            {required("upscaler", Kind::LatentUpscaler), required("latent", Kind::Tensor), std::move(width), std::move(height)},
            {Kind::Tensor}, upscale_latent);
        registry["audio_vae_encode"] = make_operation(
            {required("vae", Kind::AudioVAE), required("samples", Kind::Tensor), integer("sample_rate", 1000, 384000)}, {Kind::Tensor}, audio_vae_encode);
        registry["audio_vae_decode"] = make_operation(
            {required("vae", Kind::AudioVAE), required("latent", Kind::Tensor)}, audio_signature(), audio_vae_decode);
    }

    static void register_video_restoration_operations(Registry& registry) {
        registry["seedvr2::positive_embedding"] = make_operation(
            {required("dit", Kind::Denoiser)}, {Kind::Tensor}, seedvr2_positive_embedding);
        registry["relayvsr::positive_embedding"] = make_operation(
            {required("dit", Kind::Denoiser)}, {Kind::Tensor}, positive_embedding);
        registry["relayvsr::latent_shape"] = make_operation(
            {required("dit", Kind::Denoiser), required("keyframes", Kind::Tensor)}, {Kind::Tensor}, relay_latent_shape);
        registry["relayvsr::decode"] = make_operation(
            {required("flash", Kind::VAE), required("latent", Kind::Tensor), required("frames", Kind::Tensor),
             required("keyframe_indices", Kind::Tensor), number("detail", 0, 1, 1)},
            {Kind::Tensor}, relay_decode);
    }

    static void register_detailer_operations(Registry& registry) {
        auto input_size     = optional("input_size", Kind::Integer, 640);
        input_size.validate = [](const Value& value) {
            auto size = value.scalar.get<int64_t>();
            if (size < 1 || size > 4096)
                throw std::runtime_error("input_size must be between 1 and 4096");
        };
        registry["detections::detect"] = make_operation(
            {
                required("detector", Kind::Detector),
                required("image", Kind::Tensor),
                std::move(input_size),
                number("confidence", 0, 1, 0.3),
                number("nms", 0, 1, 0.45),
                integer("max_detections", 1, 4096, 100),
            },
            detections_signature(),
            detect_objects);
        registry["detections::filter"] = make_operation(
            {
                required("image", Kind::Tensor),
                required("detections", Kind::Dict),
                integer("top_k", 0, INT_MAX, 0),
                choice("top_k_by", {"area", "confidence"}, "area"),
                number("min_area_ratio", 0, 1, 0),
                number("max_area_ratio", 0, 1, 1),
                choice("sort_by", {"none", "left_to_right", "center_to_edge", "area"}, "none"),
            },
            detections_signature(), filter_detections);
        registry["detections::box_masks"] = make_operation(
            {required("image", Kind::Tensor), required("detections", Kind::Dict)},
            {Kind::List, Kind::Tensor}, detection_box_masks);
        registry["detections::to_regions"] = make_operation(
            {required("image", Kind::Tensor), required("detections", Kind::Dict)},
            regions_signature(), detections_to_regions);
        auto aspect                          = int_vector_argument("aspect");
        aspect.required                      = false;
        aspect.default_value                 = Value::list({Value::literal(1), Value::literal(1)});
        registry["regions::plan_mask_crops"] = make_operation(
            {
                list_argument("masks", Kind::Tensor),
                std::move(aspect),
                integer("padding", 0, 16384, 32),
                optional("only_masked", Kind::Boolean, true),
            },
            {
                {Kind::List, Kind::Dict},
                {
                    {"box", {Kind::Tensor}},
                    {"mask", {Kind::Tensor}},
                    {"index", {Kind::Integer}},
                },
            },
            plan_mask_crops);
    }

    void register_builtin_operations(Registry& registry) {
        registry["media::save"] = make_operation(
            {required("path", Kind::String), required("media", Kind::Dict)}, {Kind::Void}, execute_save_media);
        register_value_operations(registry);
        register_rng_operations(registry);
        register_tensor_operations(registry);
        register_image_operations(registry);
        register_region_operations(registry);
        register_video_operations(registry);
        register_audio_operations(registry);
        register_model_operations(registry);
        register_component_operations(registry);
        register_latent_operations(registry);
        register_video_restoration_operations(registry);
        register_detailer_operations(registry);
        for (const auto* name : {"latent_shape", "audio_latent_shape", "make_sigmas", "t_to_sigma", "denoiser_scalings", "relayvsr::latent_shape",
                                 "add_noise", "vae_to_dit_latents", "dit_to_vae_latents", "list::replace", "zero_out"})
            if (auto found = registry.find(name); found != registry.end())
                found->second.model_use = Operation::ModelUse::Metadata;
        for (const auto* name : {"controlnet::apply", "ip_adapter::apply", "pulid::apply"})
            registry.at(name).model_use = Operation::ModelUse::Capture;
        registry.at("preview").model_use = Operation::ModelUse::Preview;
    }

}  // namespace sd::dgl
