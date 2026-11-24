#ifndef __SD_DGL_MODELS_CONTROL_H__
#define __SD_DGL_MODELS_CONTROL_H__

#include "core/option_types.h"
#include "core/tensor.hpp"
#include "dgl/models/fwd.h"
#include "dgl/models/model_runtime.h"

struct SDCondition;
struct ZImageControlInput;
namespace ZImage {
    struct ControlNetRunner;
}

namespace sd::dgl::models {
    class ControlNet;

    struct ControlBinding {
        std::shared_ptr<ControlNet> model;
        std::shared_ptr<const Tensor<float>> hint;
        sd_control_type_t type;
        float strength;
        float start_sigma;
        float end_sigma;
    };

    struct ControlBindings {
        std::vector<ControlBinding> entries;
    };

    struct UNetControls {
        std::vector<Tensor<float>> residuals;
        float strength = 1.f;
    };

    class ControlNet final : public Model {
        ModelRuntime runtime_;

        bool refresh_weights() override { return runtime_.refresh_weights(); }
        bool attach_loras() override { return runtime_.attach_loras(); }

        std::shared_ptr<::ControlNet> control_net;
        std::shared_ptr<ZImage::ControlNetRunner> z_image_control_;

        bool build_variant(const Model& base, ModelLoader loader) override;
        bool build_runners();

    public:
        SDVersion version() const { return runtime_.version; }
        bool circular_x = false;
        bool circular_y = false;
        ControlNet(std::shared_ptr<ModelManager> manager, std::shared_ptr<SDBackendPool> backends);
        bool init(ModelLoadParams params);
        void validate_hint(const Tensor<float>& hint, sd_control_type_t type) const;
        void validate_target(const DiT& target) const;
        void apply_circular_axes(bool x, bool y);
        std::vector<Tensor<float>> compute_sample_controls(const ControlBinding& binding, const Tensor<float>& input, const Tensor<float>& timesteps, const SDCondition& condition);
        ZImageControlInput z_image_input(const ControlBinding& binding, const Tensor<float>& input) const;
    };
}

#endif
