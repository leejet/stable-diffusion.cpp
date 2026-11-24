#ifndef __SD_DGL_OPS_PULID_H__
#define __SD_DGL_OPS_PULID_H__

#include "dgl/models/fwd.h"

#include <map>
#include <memory>

#include "core/tensor.hpp"
#include "model/diffusion/model.hpp"

namespace sd::dgl::ops {

    struct PuLIDBinding {
        std::shared_ptr<models::PuLID> model;
        std::shared_ptr<const Tensor<float>> embedding;
        float strength;
        float start_sigma;
        float end_sigma;
    };

    struct PuLIDBindings {
        std::vector<PuLIDBinding> entries;
    };

    using PuLIDCache = std::map<const models::PuLID*, std::map<const Tensor<float>*, std::vector<Tensor<float>>>>;

    class PuLIDAttention final : public FluxImageBlockProcessor {
        struct Projected {
            const models::PuLID* model;
            const std::vector<Tensor<float>>* kv;
            float strength;
        };
        std::vector<Projected> adapters_;

    public:
        PuLIDAttention(const std::vector<const PuLIDBinding*>& bindings, PuLIDCache& cache);
        bool applies(bool single_stream, int block) const override;
        ggml_tensor* apply(GGMLRunnerContext& ctx, bool single_stream, int block, ggml_tensor* image) const override;
        size_t extra_graph_size() const override;
    };
}

#endif  // __SD_DGL_OPS_PULID_H__
