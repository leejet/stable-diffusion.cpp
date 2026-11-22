#ifndef __SD_DGL_MODELS_DIFFUSION_MODEL_H__
#define __SD_DGL_MODELS_DIFFUSION_MODEL_H__

#include <memory>

namespace sd::dgl::models {
    class AnimateDiff;
    class DiT;

    struct DiffusionModel {
        std::shared_ptr<DiT> dit;
        std::shared_ptr<AnimateDiff> motion;
        // Keep the original resource reachable through the context's weak cache.
        std::shared_ptr<const DiffusionModel> source;
    };
}

#endif  // __SD_DGL_MODELS_DIFFUSION_MODEL_H__
