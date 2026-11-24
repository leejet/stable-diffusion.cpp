#include "dgl/models/audio_encoder.h"

#include <algorithm>

#include "core/util.h"
#include "dgl/models/component_sources.h"
#include "model/audio/wav2vec2.hpp"
#include "runtime/audio_processing.h"

namespace sd::dgl::models {
    AudioEncoder::AudioEncoder(std::shared_ptr<ModelManager> manager, std::shared_ptr<SDBackendPool> backends)
        : Model(std::move(manager)),
          runtime_(*this, std::move(backends), ModelComponent::AudioEncoder, SDBackendModule::AUDIO_ENCODER, [this] { return build_runners(); }) {}

    bool AudioEncoder::init(ModelLoadParams params) {
        return runtime_.init(std::move(params), load_component_catalog<ModelComponent::AudioEncoder>);
    }

    bool AudioEncoder::build_runners() {
        audio_encoder = std::make_shared<Wav2Vec2::Wav2Vec2ModelRunner>(runtime_.backend_for(), model_weights->loader().get_tensor_storage_map(), "wav2vec2.", model_weights);
        runtime_.configure_runner(*audio_encoder);
        return runtime_.register_runner_params(runtime_.component, audio_encoder);
    }

    sd::Tensor<float> AudioEncoder::get_audio_embedding(const sd::Audio& audio) {
        if (audio_encoder == nullptr) {
            LOG_ERROR("audio encoder model is not loaded");
            return {};
        }
        if (audio.data == nullptr || audio.sample_count == 0 || audio.channels == 0 || audio.sample_rate == 0) {
            LOG_ERROR("invalid driving audio");
            return {};
        }
        auto mono = sd::audio::downmix_to_mono(audio.data, audio.sample_count, audio.channels);
        if (mono.empty()) {
            LOG_ERROR("audio mono downmix failed");
            return {};
        }
        mono = sd::audio::resample_audio(mono.data(), mono.size(), audio.sample_rate, 16000);
        if (mono.empty()) {
            LOG_ERROR("audio resample to 16 kHz failed");
            return {};
        }
        return audio_encoder->compute(n_threads, mono);
    }
}
