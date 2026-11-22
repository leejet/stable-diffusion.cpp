#include "dgl/ops/seedvr2.h"

#include "dgl/models/dit.h"
#include "dgl/models/fwd.h"
#include "model/diffusion/seedvr2.hpp"

namespace sd::dgl::ops {
    Tensor<float> seedvr2_positive_embedding(models::DiT& dit) {
        if (dit.version() != VERSION_SEEDVR2)
            throw std::runtime_error("expected a SeedVR2 DiT");
        const auto& model = static_cast<const SeedVR2::SeedVR2Runner&>(*dit.diffusion_model);
        std::vector<float> values;
        if (!dit.model_weights->load_float_tensor(model.positive_embedding_weight(), values))
            throw std::runtime_error("cannot read SeedVR2 positive embedding");
        return Tensor<float>({model.config.text_dim, model.config.text_len}, std::move(values));
    }
}
