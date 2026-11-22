#include "dgl/bindings/latents.h"

#include "dgl/bindings/arguments.h"
#include "dgl/bindings/audio.h"
#include "dgl/bindings/values.h"
#include "dgl/models/audio_encoder.h"
#include "dgl/models/audio_vae.h"
#include "dgl/models/ip_adapter.h"
#include "dgl/models/latent_upscaler.h"
#include "dgl/models/vision_encoder.h"
#include "dgl/ops/media.h"
#include "model/vae/audio_vae.hpp"

namespace sd::dgl {

    Value encode_vision(Values& args, RunState&) {
        auto& engine = *object<VisionEncoder>(args, "encoder");
        Model::ExecutionScope scope(engine);
        scope.require_ready();
        sd::Tensor<float> decoded;
        const auto& image = pixels(args.at("image"), decoded);
        if (image.dim() != 4 || image.shape()[2] == 2) {
            throw std::runtime_error("vision encoding requires an image");
        }
        const auto& rgb       = image.shape()[2] == 3
                                    ? image
                                    : (decoded = sd::ensure_image_tensor_channels(image, 3));
        const auto zero       = get_bool(args, "zero");
        const auto zero_input = get_bool(args, "zero_input");
        if (zero && zero_input)
            throw std::runtime_error("encode_vision accepts either zero or zero_input");
        auto result = engine.get_clip_vision_output(rgb, get_bool(args, "pooled"),
                                                    int(get_int(args, "clip_skip")), zero, zero_input);
        if (result.empty()) {
            throw std::runtime_error("vision encoding failed");
        }
        return tensor_value(std::move(result));
    }

    Value encode_audio(Values& args, RunState&) {
        const auto audio = audio_view(*object<sd_tensor_t>(args, "samples"), get_int(args, "sample_rate"));
        auto& engine     = *object<AudioEncoder>(args, "encoder");
        Model::ExecutionScope scope(engine);
        scope.require_ready();
        auto result = engine.get_audio_embedding(audio);
        if (result.empty()) {
            throw std::runtime_error("audio encoding failed");
        }
        return tensor_value(std::move(result));
    }

    Value encode_ip_adapter(Values& args, RunState&) {
        auto& engine = *object<IPAdapter>(args, "adapter");
        Model::ExecutionScope scope(engine);
        scope.require_ready();
        auto result = engine.project_image(float_tensor(args, "embedding"));
        if (result.empty()) {
            throw std::runtime_error("IP-Adapter projection failed");
        }
        return tensor_value(std::move(result));
    }

    Value upscale_latent(Values& args, RunState&) {
        auto upscaler = object<models::LatentUpscaler>(args, "upscaler");
        Model::ExecutionScope scope(*upscaler);
        scope.require_ready();
        const auto& latent = float_tensor(args, "latent");
        const auto width   = args.count("width") ? get_int(args, "width") : 0;
        const auto height  = args.count("height") ? get_int(args, "height") : 0;
        auto result        = upscaler->upscale(latent, width, height);
        if (result.empty()) {
            throw std::runtime_error("learned latent upscaling failed");
        }
        return tensor_value(std::move(result));
    }

    Value audio_vae_encode(Values& args, RunState&) {
        const auto audio = audio_view(*object<sd_tensor_t>(args, "samples"), get_int(args, "sample_rate"));
        auto& engine     = *object<AudioVAE>(args, "vae");
        Model::ExecutionScope scope(engine);
        scope.require_ready();
        auto waveform = sd::dgl::ops::prepare_minimax_h3_reference_waveform(audio, engine.audio_vae_model->input_sample_rate());
        auto result   = engine.audio_vae_model->encode(engine.n_threads, waveform);
        if (result.empty()) {
            throw std::runtime_error("audio VAE encoding failed");
        }
        return tensor_value(std::move(result));
    }

    Value audio_vae_decode(Values& args, RunState&) {
        auto& engine = *object<AudioVAE>(args, "vae");
        Model::ExecutionScope scope(engine);
        scope.require_ready();
        auto waveform = engine.decode_ltx_audio_latent(float_tensor(args, "latent"));
        std::unique_ptr<sd::Audio, decltype(&sd::free_audio)> audio(sd::dgl::ops::waveform_to_sd_audio(&engine, waveform), sd::free_audio);
        if (!audio) {
            throw std::runtime_error("audio VAE decoding failed");
        }
        return audio_result(copy_audio(*audio));
    }

    Value execute_audio_latent_shape(Values& args, RunState&) {
        const auto& vae = *object<AudioVAE>(args, "vae");
        return shape_value(vae.latent_shape(int(get_int(args, "frames")), int(get_int(args, "fps"))));
    }

    Value execute_ltx_video_positions(Values& args, RunState&) {
        const auto& s   = object<sd_tensor_t>(args, "like")->shape();
        const auto keys = get_int(args, "keyframe_latent_frames");
        if (s.size() != 5 || s[0] <= 0 || s[1] <= 0 || s[2] <= 0 || s[0] > 16384 || s[1] > 16384 || s[2] > 65536) {
            throw std::runtime_error("expected a video latent for positions");
        }
        validate_float_shape({2, 3, s[0] * s[1] * (s[2] + keys), 1});
        return tensor_value(sd::dgl::ops::build_ltxv_video_positions(
            s[0], s[1], s[2], keys, int(get_int(args, "keyframe_frame_index")), int(get_int(args, "keyframe_pixel_frames")),
            int(get_int(args, "fps")), int(get_int(args, "spatial_scale")), int(get_int(args, "temporal_scale")), get_bool(args, "causal")));
    }

    Value execute_s2v_audio_window(Values& args, RunState&) {
        const auto& embedding = float_tensor(args, "embedding");
        if (embedding.dim() != 3) {
            throw std::runtime_error("audio embedding must have feature, frame and layer axes");
        }
        const auto frames = 4 * get_int(args, "latent_frames");
        validate_float_shape({embedding.shape()[0], frames, embedding.shape()[2]});
        auto result = sd::dgl::ops::build_s2v_audio_window(embedding, frames);
        if (result.empty()) {
            throw std::runtime_error("could not build the S2V audio window");
        }
        return tensor_value(std::move(result));
    }

}
