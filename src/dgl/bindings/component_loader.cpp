#include "dgl/bindings/component_loader.h"

#include "core/util.h"
#include "dgl/context.h"
#include "dgl/models/animatediff.h"
#include "dgl/models/audio_encoder.h"
#include "dgl/models/audio_vae.h"
#include "dgl/models/control.h"
#include "dgl/models/diffusion_model.h"
#include "dgl/models/dit.h"
#include "dgl/models/ip_adapter.h"
#include "dgl/models/latent_upscaler.h"
#include "dgl/models/photomaker.h"
#include "dgl/models/pulid.h"
#include "dgl/models/text_encoder.h"
#include "dgl/models/vae.h"
#include "dgl/models/vision_encoder.h"
#include "model_definition.h"
#include "tokenizers/tokenizer_config.h"

namespace sd::dgl {
    static const char* string_option(const Values& args, const char* name) {
        auto found = args.find(name);
        return found == args.end() ? "" : found->second.scalar.get_ref<const std::string&>().c_str();
    }

    static models::ModelLoadParams read_model_params(const Values& args) {
        models::ModelLoadParams params;
        const auto family = get_string(args, "family");
        if (family != "auto") {
            const auto* definition = find_model_definition(family);
            if (!definition)
                throw std::runtime_error("unknown component family '" + family + "'");
            params.family = definition->version;
        }
        params.family_from       = string_option(args, "family_from");
        params.path              = get_string(args, "path");
        params.backend           = get_string(args, "backend");
        params.params_backend    = get_string(args, "params_backend");
        params.split_mode        = get_string(args, "split_mode");
        params.model_args        = get_string(args, "model_args");
        params.tensor_type_rules = get_string(args, "tensor_type_rules");
        static const std::map<std::string, models::OperatorMode> modes{
            {"auto", models::OperatorMode::Auto}, {"on", models::OperatorMode::On}, {"off", models::OperatorMode::Off}};
        params.flash_attn_mode  = modes.at(trim(get_string(args, "flash_attn_mode")));
        params.conv_direct_mode = modes.at(trim(get_string(args, "conv_direct_mode")));
        params.sage_attn        = get_bool(args, "sage_attn");
        params.linear_i8        = get_string(args, "linear_type") == "i8";
        params.conv_scale       = float(get_number(args, "conv_scale"));
        params.linear_scale     = float(get_number(args, "linear_scale"));
        params.attn_scale       = float(get_number(args, "attn_scale"));
        if (get_string(args, "wtype") != "auto") {
            params.wtype = str_to_sd_type(string_option(args, "wtype"));
            if (params.wtype == SD_TYPE_COUNT)
                throw std::runtime_error("unknown weight type");
        }
        return params;
    }

    static nlohmann::json normalized_model_params(models::ModelLoadParams& params, PreparedResource& prepared) {
        prepared.revision     = nlohmann::json::object();
        params.path           = file_key(params.path, prepared.revision);
        params.family_from    = file_key(params.family_from, prepared.revision);
        params.backend        = trim(params.backend);
        params.params_backend = trim(params.params_backend);
        params.split_mode     = trim(params.split_mode);
        for (float* scale : {&params.conv_scale, &params.linear_scale, &params.attn_scale})
            if (*scale == 0.f)
                *scale = 0.f;
        return {
            {"path", params.path},
            {"family", int(params.family)},
            {"family_from", params.family_from},
            {"backend", params.backend},
            {"params_backend", params.params_backend},
            {"split_mode", params.split_mode},
            {"wtype", int(params.wtype)},
            {"tensor_type_rules", params.tensor_type_rules},
            {"model_args", params.model_args},
            {"flash_attn_mode", int(params.flash_attn_mode)},
            {"conv_direct_mode", int(params.conv_direct_mode)},
            {"sage_attn", params.sage_attn},
            {"linear_i8", params.linear_i8},
            {"conv_scale", params.conv_scale},
            {"linear_scale", params.linear_scale},
            {"attn_scale", params.attn_scale},
        };
    }

    template <class T, class Params>
    static std::shared_ptr<T> load_model(ResourceLoadRequest& request, Params params, const std::string& path) {
        auto model = request.context.make_model<T>(request.weights_key);
        if (!model->init(std::move(params)))
            throw std::runtime_error("cannot load component from '" + path + "'");
        return model;
    }

    template <class T, class Params>
    static Value make_model(ResourceLoadRequest& request, Params params, const std::string& path) {
        return Value::object(load_model<T>(request, std::move(params), path));
    }

    PreparedResource load_dit(Values& args) {
        models::DiTLoadParams params{
            read_model_params(args),
        };
        PreparedResource prepared;
        prepared.key  = normalized_model_params(params.model, prepared);
        prepared.load = [params = std::move(params)](ResourceLoadRequest& request) mutable {
            const auto path = params.model.path;
            auto model      = load_model<models::DiT>(request, std::move(params), path);
            return Value::object(std::make_shared<models::DiffusionModel>(models::DiffusionModel{std::move(model)}));
        };
        return prepared;
    }

    PreparedResource load_text_encoder(Values& args) {
        models::TextEncoderLoadParams params{
            read_model_params(args),
            TokenizerConfig(string_option(args, "tokenizer")),
            string_option(args, "vision_path"),
            string_option(args, "projection_path"),
            string_option(args, "connectors_path"),
            string_option(args, "t5_path"),
            string_option(args, "clip_l_path"),
            string_option(args, "clip_g_path"),
        };
        PreparedResource prepared;
        prepared.key = normalized_model_params(params.model, prepared);
        for (const auto& [name, member] : {std::pair{"vision_path", &models::TextEncoderLoadParams::vision_path},
                                           {"projection_path", &models::TextEncoderLoadParams::projection_path},
                                           {"connectors_path", &models::TextEncoderLoadParams::connectors_path},
                                           {"t5_path", &models::TextEncoderLoadParams::t5_path},
                                           {"clip_l_path", &models::TextEncoderLoadParams::clip_l_path},
                                           {"clip_g_path", &models::TextEncoderLoadParams::clip_g_path}}) {
            auto& path         = params.*member;
            path               = file_key(path, prepared.revision);
            prepared.key[name] = path;
        }
        auto paths = params.tokenizer.paths();
        for (auto& path : paths)
            path = file_key(path, prepared.revision);
        prepared.key["tokenizer"] = paths;
        params.tokenizer          = TokenizerConfig(std::move(paths));
        prepared.load             = [params = std::move(params)](ResourceLoadRequest& request) mutable {
            const auto path = params.model.path;
            return make_model<models::TextEncoder>(request, std::move(params), path);
        };
        return prepared;
    }

    PreparedResource load_vae(Values& args) {
        static const std::map<std::string, sd_vae_format_t> formats{
            {"auto", SD_VAE_FORMAT_AUTO},
            {"flux", SD_VAE_FORMAT_FLUX},
            {"sd3", SD_VAE_FORMAT_SD3},
            {"flux2", SD_VAE_FORMAT_FLUX2},
            {"wan", SD_VAE_FORMAT_WAN},
        };
        models::VAELoadParams params{
            read_model_params(args),
            formats.at(get_string(args, "vae_format")),
        };
        PreparedResource prepared;
        prepared.key           = normalized_model_params(params.model, prepared);
        prepared.key["format"] = int(params.format);
        prepared.load          = [params = std::move(params)](ResourceLoadRequest& request) mutable {
            const auto path = params.model.path;
            return make_model<models::VAE>(request, std::move(params), path);
        };
        return prepared;
    }

    PreparedResource load_tae(Values& args) {
        auto params            = read_model_params(args);
        const bool decode_only = get_bool(args, "decode_only");
        PreparedResource prepared;
        prepared.key                = normalized_model_params(params, prepared);
        prepared.key["decode_only"] = decode_only;
        prepared.load               = [params = std::move(params), decode_only](ResourceLoadRequest& request) mutable {
            auto model = request.context.make_model<models::VAE>(request.weights_key);
            if (!model->init_tae(std::move(params), decode_only))
                throw std::runtime_error("cannot load TAE");
            return Value::object(std::move(model));
        };
        return prepared;
    }

    template <class T>
    static PreparedResource prepare_model(Values& args) {
        auto params = read_model_params(args);
        PreparedResource prepared;
        prepared.key  = normalized_model_params(params, prepared);
        prepared.load = [params = std::move(params)](ResourceLoadRequest& request) mutable {
            const auto path = params.path;
            return make_model<T>(request, std::move(params), path);
        };
        return prepared;
    }

    PreparedResource load_controlnet(Values& args) {
        return prepare_model<models::ControlNet>(args);
    }

    PreparedResource load_vision_encoder(Values& args) {
        return prepare_model<models::VisionEncoder>(args);
    }

    PreparedResource load_audio_encoder(Values& args) {
        return prepare_model<models::AudioEncoder>(args);
    }

    PreparedResource load_audio_vae(Values& args) {
        return prepare_model<models::AudioVAE>(args);
    }

    PreparedResource load_ip_adapter(Values& args) {
        return prepare_model<models::IPAdapter>(args);
    }

    PreparedResource load_photomaker(Values& args) {
        return prepare_model<models::PhotoMaker>(args);
    }

    PreparedResource load_pulid(Values& args) {
        return prepare_model<models::PuLID>(args);
    }

    PreparedResource load_animatediff(Values& args) {
        return prepare_model<models::AnimateDiff>(args);
    }

    PreparedResource load_latent_upscaler(Values& args) {
        return prepare_model<models::LatentUpscaler>(args);
    }
}  // namespace sd::dgl
