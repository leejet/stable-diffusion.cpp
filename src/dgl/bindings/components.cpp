#include "dgl/bindings/components.h"

#include <climits>
#include <cmath>
#include <type_traits>

#include "conditioning/conditioner.hpp"
#include "core/rng.hpp"
#include "dgl/bindings/arguments.h"
#include "dgl/bindings/audio.h"
#include "dgl/bindings/condition.h"
#include "dgl/bindings/values.h"
#include "dgl/bindings/video.h"
#include "dgl/context.h"
#include "dgl/models/dit.h"
#include "dgl/models/photomaker.h"
#include "dgl/models/photomaker_prompt.h"
#include "dgl/models/text_encoder.h"
#include "dgl/models/vae.h"
#include "dgl/ops/media.h"
#include "dgl/ops/predictor.h"
#include "dgl/ops/preview.h"
#include "dgl/ops/prompt.h"
#include "dgl/ops/sampling.h"
#include "dgl/ops/vae.h"
#include "extensions/pulid_embedding.h"
#include "model/vae/vae.hpp"
#include "model_definition.h"
#include "model_loader.h"
#include "runtime/denoiser.hpp"

namespace sd::dgl {
    static ops::PromptMedia prompt_media(const Value& input) {
        ops::PromptMedia result;
        if (input.type().kind != Kind::Dict) {
            result.media = pixels(input);
            return result;
        }
        const Value* media = nullptr;
        std::string kind;
        for (const char* name : {"image", "video", "audio"}) {
            if (const auto found = input.fields.find(name); found != input.fields.end()) {
                if (media)
                    throw std::runtime_error("a media dict requires exactly one image, video or audio");
                media = &found->second;
                kind  = name;
            }
        }
        if (!media)
            throw std::runtime_error("a media dict requires exactly one image, video or audio");
        for (const auto& field : input.fields)
            if (field.first != kind && field.first != "stride")
                throw std::runtime_error("unknown media field '" + field.first + "'");
        if (input.fields.count("stride")) {
            if (kind != "video" || input.fields.at("stride").type().kind != Kind::Integer)
                throw std::runtime_error("media stride requires a video and an integer value");
            result.stride = get_int(input.fields, "stride");
            if (result.stride < 1 || result.stride > 65536)
                throw std::runtime_error("media stride must be in [1,65536]");
        }
        if (kind == "video") {
            auto video     = read_video_dict(*media);
            result.media   = std::move(video.frames);
            result.fps_num = video.fps_num;
            result.fps_den = video.fps_den;
        } else if (kind == "audio") {
            result.media = read_audio_dict(*media);
        } else {
            result.media = pixels(*media);
        }
        return result;
    }

    Value encode_prompt(Values& args, RunState&) {
        ops::PromptEncodingOptions options;
        options.text      = get_string(args, "text");
        options.width     = int(get_int(args, "width"));
        options.height    = int(get_int(args, "height"));
        options.clip_skip = int(get_int(args, "clip_skip"));
        options.raw       = get_bool(args, "raw");
        if (args.count("system"))
            options.system = get_string(args, "system");
        if (options.raw && (args.count("media") || args.count("image_bindings")))
            throw std::runtime_error("raw prompt encoding accepts text only; omit media and image_bindings");
        if (args.count("media")) {
            options.media.emplace();
            for (const auto& input : args.at("media").items) {
                try {
                    options.media->push_back(prompt_media(input));
                } catch (const std::exception& error) {
                    throw std::runtime_error("media[" + std::to_string(options.media->size()) + "]: " + error.what());
                }
            }
        }
        if (args.count("image_bindings")) {
            options.image_bindings = [&bindings = args.at("image_bindings").fields](const std::string& source) {
                auto found = bindings.find(source);
                if (found == bindings.end())
                    return sd::Tensor<float>{};
                if (found->second.type().kind != Kind::Tensor)
                    throw std::runtime_error("image binding '" + source + "' must be a tensor");
                auto image = pixels(found->second);
                if (image.empty())
                    throw std::runtime_error("image binding '" + source + "' must not be empty");
                return image;
            };
        }
        return condition_value(ops::encode_prompt(*object<TextEncoder>(args, "encoder"), std::move(options)));
    }

    Value photomaker_tokens(Values& args, RunState&) {
        auto model = object<PhotoMaker>(args, "photomaker");
        Model::ExecutionScope scope(*model);
        scope.require_ready();
        std::vector<sd::Tensor<float>> images;
        for (const auto& value : args.at("images").items)
            images.push_back(pixels(value));
        const auto* embedding = args.count("id_embedding") ? &float_tensor(args, "id_embedding") : nullptr;
        auto tokens           = model->project_images(images, embedding);
        if (tokens.empty())
            throw std::runtime_error("PhotoMaker image encoding failed");
        return tensor_value(std::move(tokens));
    }

    Value encode_photomaker_prompt(Values& args, RunState&) {
        auto encoder = object<TextEncoder>(args, "encoder");
        ConditionerParams condition;
        condition.text      = get_string(args, "text");
        condition.width     = int(get_int(args, "width"));
        condition.height    = int(get_int(args, "height"));
        condition.clip_skip = int(get_int(args, "clip_skip"));
        auto output         = encoder->encode_photomaker_prompt(std::move(condition), float_tensor(args, "tokens"));
        return Value::dict({{"plain", condition_value(std::move(output.plain))},
                            {"condition", condition_value(std::move(output.condition))},
                            {"mask", Value::object(make_tensor(std::move(output.mask)))}});
    }

    Value apply_photomaker(Values& args, RunState&) {
        auto fields = args.at("condition").fields;
        if (!fields.count("crossattn"))
            throw std::runtime_error("PhotoMaker requires a condition with crossattn");
        auto model = object<PhotoMaker>(args, "photomaker");
        Model::ExecutionScope scope(*model);
        scope.require_ready();
        auto fused = model->fuse_prompt(float_tensor(fields, "crossattn"), float_tensor(args, "tokens"),
                                        read_tensor<uint8_t>(args.at("mask")));
        if (fused.empty())
            throw std::runtime_error("PhotoMaker condition fusion failed");
        fields.insert_or_assign("crossattn", tensor_value(std::move(fused)));
        return Value::dict(std::move(fields));
    }

    static ops::VAEOptions vae_options(const Values& args) {
        ops::VAEOptions options;
        options.circular_x = get_bool(args, "circular_x");
        options.circular_y = get_bool(args, "circular_y");
        if (args.count("tiling"))
            options.tiling = tiling_options(args.at("tiling").fields);
        return options;
    }

    Value vae_encode(Values& args, RunState&) {
        const auto rng = object<RNG>(args, "rng");
        auto& engine   = *object<VAE>(args, "vae");
        sd::Tensor<float> decoded;
        const auto& input = pixels(args.at("pixels"), decoded);
        auto output       = ops::encode_vae(engine, input, rng, get_string(args, "posterior") == "mode", vae_options(args));
        return tensor_value(std::move(output));
    }

    static Value convert_vae_latent(Values& args, bool to_dit) {
        auto& engine = *object<VAE>(args, "vae");
        Model::ExecutionScope scope(engine);
        scope.require_ready();
        const auto& input = float_tensor(args, "latent");
        engine.validate_vae_latent(input);
        auto output = to_dit ? engine.vae_to_dit_latents(input) : engine.dit_to_vae_latents(input);
        if (output.empty()) {
            throw std::runtime_error("VAE latent conversion failed");
        }
        return tensor_value(std::move(output));
    }

    Value vae_decode(Values& args, RunState&) {
        auto& engine = *object<VAE>(args, "vae");
        auto output  = ops::decode_vae(engine, float_tensor(args, "latent"), vae_options(args));
        return get_string(args, "dtype") == "f32" ? tensor_value(std::move(output)) : image_to_u8(output);
    }

    Value latent_shape(Values& args, RunState&) {
        auto& engine = *object<VAE>(args, "vae");
        Model::ExecutionScope scope(engine);
        scope.require_ready();
        return shape_value(engine.latent_shape(get_int(args, "width"), get_int(args, "height"),
                                               args.count("frames") ? int(get_int(args, "frames")) : 0,
                                               int(get_int(args, "batch"))));
    }

    static int schedule_sequence_length(const Values& args, const DiT& model) {
        const auto sequence_length = args.count("sequence_length") ? int(get_int(args, "sequence_length")) : 0;
        if (!args.count("latent"))
            return sequence_length;
        const auto& shape = object<sd_tensor_t>(args, "latent")->shape();
        if (shape.size() != 4 && shape.size() != 5)
            throw std::runtime_error("expected an image or video latent");
        validate_float_shape(shape);
        if (sequence_length > 0)
            return sequence_length;
        const auto patch = model.diffusion_model->get_patch_size();
        auto length      = ((shape[0] + patch - 1) / patch) * ((shape[1] + patch - 1) / patch);
        if (shape.size() == 5 && (sd_version_is_ltxav(model.version()) || sd_version_is_minimax_h3(model.version())))
            length *= shape[2];
        if (length <= 0 || length > INT_MAX)
            throw std::runtime_error("invalid diffusion sequence length");
        return int(length);
    }

    Value t_to_sigma(Values& args, RunState&) {
        const auto predict = object<ops::Predictor>(args, "predict");
        const auto& leaf   = ops::schedule_predictor(*predict);
        auto& model        = leaf.dit();
        const auto sigma   = leaf.resolved_parameterization().schedule_sigma(get_number(args, "t"), float(get_number(args, "flow_shift")),
                                                                             schedule_sequence_length(args, model));
        if (!std::isfinite(sigma) || sigma < 0.f)
            throw std::runtime_error("model time mapping did not yield a finite nonnegative sigma");
        return Value::literal(sigma);
    }

    Value denoiser_scalings(Values& args, RunState&) {
        const auto predict = object<ops::Predictor>(args, "predict");
        const auto scales  = ops::scaling_parameterization(*predict).scalings(float(get_number(args, "sigma")));
        return Value::dict({{"skip_scale", Value::literal(scales[0])},
                            {"output_scale", Value::literal(scales[1])},
                            {"input_scale", Value::literal(scales[2])}});
    }

    Value make_sigmas(Values& args, RunState&) {
        const auto predict   = object<ops::Predictor>(args, "predict");
        const auto& leaf     = ops::schedule_predictor(*predict);
        auto& engine         = leaf.dit();
        const auto seq_len   = schedule_sequence_length(args, engine);
        const auto name      = get_string(args, "scheduler");
        const auto scheduler = str_to_scheduler(name.c_str());
        if (scheduler == SCHEDULER_COUNT) {
            throw std::runtime_error("unknown scheduler '" + name + "'");
        }
        const auto extra = get_string(args, "extra_args");
        auto schedule    = leaf.resolved_parameterization().schedule(float(get_number(args, "flow_shift")));
        auto sigmas      = schedule->get_sigmas(int(get_int(args, "steps")), seq_len, scheduler, engine.version(), extra.c_str());
        return tensor_value(sd::Tensor<float>::from_vector(std::move(sigmas)));
    }

    Value sample_tensor(Values& args, RunState& state) {
        const auto rng       = object<RNG>(args, "rng");
        const auto predictor = object<ops::Predictor>(args, "predict");
        const auto& input    = float_tensor(args, "start");
        const auto& sigmas   = ops::sampling_schedule(sigma_tensor(args, "sigmas"));
        const auto options   = sigmas.size() == 1 ? ops::SamplingOptions{} : sampling_options(args, state.context);
        return tensor_value(ops::sample(input, sigmas, *predictor, rng, options, state.events));
    }

    Value execute_load_pulid_embedding(Values& args, RunState&) {
        auto path = get_string(args, "path");
        require_file_access(path, SD_DGL_FILE_READ);
        auto result = load_pulid_id_embedding(path.c_str());
        if (result.empty()) {
            throw std::runtime_error("cannot load PuLID identity embedding");
        }
        return tensor_value(std::move(result));
    }

    Value execute_vae_to_dit_latents(Values& args, RunState&) {
        return convert_vae_latent(args, true);
    }

    Value execute_dit_to_vae_latents(Values& args, RunState&) {
        return convert_vae_latent(args, false);
    }

    Value execute_add_noise(Values& args, RunState&) {
        const auto predict = object<ops::Predictor>(args, "predict");
        const auto& base   = float_tensor(args, "base");
        const auto& noise  = float_tensor(args, "noise");
        if (base.shape() != noise.shape()) {
            throw std::runtime_error("base and noise shapes differ");
        }
        return tensor_value(ops::sampling_parameterization(*predict).add_noise(float(get_number(args, "sigma")), noise, base));
    }

}  // namespace sd::dgl
