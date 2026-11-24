#ifndef __SD_DGL_MODELS_COMPONENT_BUILDERS_H__
#define __SD_DGL_MODELS_COMPONENT_BUILDERS_H__

#include "model_build_context.h"
#include "tokenizers/tokenizer_config.h"

class ModelLoader;

namespace sd::dgl::models::component_builders {
    using sd::model_builders::Context;
    using sd::model_builders::CoreRunners;

    bool build_family_runners(const Context& ctx, CoreRunners& runners, TokenizerConfig tokenizers = {});
    std::shared_ptr<::VAE> build_vae_runner(const Context& ctx, const ModelLoader& loader);
    std::shared_ptr<::VAE> build_tae_runner(const Context& ctx, bool decode_only);
    std::shared_ptr<AudioVAERunner> build_audio_vae_runner(const Context& ctx);
    bool build_control_net_runner(const Context& ctx, std::shared_ptr<::ControlNet>& runner);

}  // namespace sd::dgl::models::component_builders

#endif  // __SD_DGL_MODELS_COMPONENT_BUILDERS_H__
