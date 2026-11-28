#ifndef __SD_DGL_MODELS_RELAY_VSR_H__
#define __SD_DGL_MODELS_RELAY_VSR_H__

#include "core/tensor.hpp"
#include "model_build_context.h"
#include "model_component.h"
#include "model_manager.h"

namespace sd::dgl::models {
    std::shared_ptr<::VAE> build_relay_vsr_vae(const model_builders::Context& ctx, const ModelLoader& loader);
}

#endif
