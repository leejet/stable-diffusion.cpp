#include "dgl/models/audio_vae.h"

#include <algorithm>
#include <cmath>

#include "dgl/models/component_builders.h"
#include "dgl/models/component_sources.h"
#include "dgl/models/vae.h"
#include "dgl/run_events.h"
#include "model/vae/audio_geometry.hpp"
#include "model/vae/audio_vae.hpp"

namespace sd::dgl::models {
    AudioVAE::AudioVAE(std::shared_ptr<ModelManager> manager, std::shared_ptr<SDBackendPool> backends)
        : Model(std::move(manager)),
          runtime_(*this, std::move(backends), ModelComponent::AudioVAE, SDBackendModule::VAE, [this] { return build_runners(); }) {}

    bool AudioVAE::init(ModelLoadParams params) {
        return runtime_.init(std::move(params), load_component_catalog<ModelComponent::AudioVAE>);
    }

    bool AudioVAE::build_runners() {
        audio_vae_model = component_builders::build_audio_vae_runner(runtime_.model_build_context());
        if (!audio_vae_model) {
            return false;
        }
        runtime_.configure_runner(*audio_vae_model, true);
        return runtime_.register_runner_params(runtime_.component, audio_vae_model);
    }

    std::vector<int64_t> AudioVAE::latent_shape(int frames, int fps) const {
        return sd::vae::audio_latent_shape(version(), frames, fps);
    }

    sd::Tensor<float> AudioVAE::decode_ltx_audio_latent(const sd::Tensor<float>& audio_latent) {
        if (audio_vae_model == nullptr || audio_latent.empty()) {
            return {};
        }
        report_run_progress("audio_decode");
        auto waveform = audio_vae_model->decode(n_threads, audio_latent);
        return waveform;
    }
}
