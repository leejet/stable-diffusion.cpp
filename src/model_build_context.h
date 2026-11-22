#ifndef __SD_MODEL_BUILD_CONTEXT_H__
#define __SD_MODEL_BUILD_CONTEXT_H__

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "core/ggml_extend_backend.h"
#include "core/option_types.h"
#include "model.h"
#include "stable-diffusion.h"

class SDBackendManager;
struct DeviceResidencyManager;
struct Conditioner;
struct FrozenCLIPVisionEmbedder;
struct DiffusionModelRunner;
struct VAE;
struct AudioVAERunner;
struct ControlNet;
namespace Wav2Vec2 {
    class Wav2Vec2ModelRunner;
}
namespace IPAdapter {
    struct IPAdapterRunner;
}

namespace sd::model_builders {

    struct ModelBuildParams {
        std::string model_args;
        std::string tokenizer;
        std::map<std::string, std::string> embeddings;
        float linear_scale         = 0.f;
        float attn_scale           = 0.f;
        sd_vae_format_t vae_format = SD_VAE_FORMAT_AUTO;
        bool build_high_noise      = false;
        bool build_ip_adapter      = false;
    };

    struct Context {
        ModelBuildParams params;
        SDVersion version;
        const String2TensorStorage& tensor_storage_map;
        SDBackendManager& backends;
        std::shared_ptr<DeviceResidencyManager> weight_manager;
        bool build_conditioner = true;
        bool build_diffusion   = true;
        bool build_auxiliary   = true;
    };

    struct CoreRunners {
        std::shared_ptr<Conditioner> conditioner;
        std::shared_ptr<DiffusionModelRunner> diffusion;
        std::shared_ptr<DiffusionModelRunner> high_noise_diffusion;
        std::shared_ptr<FrozenCLIPVisionEmbedder> clip_vision;
        std::shared_ptr<IPAdapter::IPAdapterRunner> ip_adapter;
        std::shared_ptr<Wav2Vec2::Wav2Vec2ModelRunner> audio_encoder;
    };

    struct VAEOptions {
        bool use_tae                  = false;
        bool tae_preview_only         = false;
        bool use_audio_vae            = false;
        bool external_vae_is_invalid  = false;
        bool apply_default_conv_scale = false;
    };

    struct VAERunners {
        std::shared_ptr<VAE> vae;
        std::shared_ptr<VAE> preview;
        std::shared_ptr<AudioVAERunner> audio;
    };

    bool ensure_model_backend_pair(SDBackendManager& backends, SDBackendModule module);

}  // namespace sd::model_builders

#endif  // __SD_MODEL_BUILD_CONTEXT_H__
