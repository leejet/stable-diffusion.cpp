#ifndef __SD_DGL_MODELS_PULID_H__
#define __SD_DGL_MODELS_PULID_H__

#include "core/tensor.hpp"
#include "dgl/models/fwd.h"
#include "dgl/models/model_runtime.h"

class PuLIDRunner;
struct GGMLRunnerContext;

namespace sd::dgl::models {
    class PuLID final : public Model {
        ModelRuntime runtime_;
        std::shared_ptr<PuLIDRunner> runner_;

        bool refresh_weights() override { return runtime_.refresh_weights(); }
        bool attach_loras() override { return runtime_.attach_loras(); }
        bool build_variant(const Model& source, ModelLoader loader) override;
        bool build_runners();

    public:
        PuLID(std::shared_ptr<ModelManager> manager, std::shared_ptr<SDBackendPool> backends);
        bool init(ModelLoadParams params);
        void validate_target(const DiT& target) const;
        void validate_embedding(const Tensor<float>& embedding) const;
        std::vector<Tensor<float>> compute_kv(const Tensor<float>& embedding);
        size_t extra_graph_size() const;
        ggml_tensor* attend(GGMLRunnerContext& ctx, size_t layer, const Tensor<float>& kv, ggml_tensor* image) const;
    };
}

#endif  // __SD_DGL_MODELS_PULID_H__
