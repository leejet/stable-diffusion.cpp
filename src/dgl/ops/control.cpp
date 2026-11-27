#include "dgl/ops/control.h"

#include "conditioning/conditioner.hpp"
#include "dgl/models/fwd.h"

namespace sd::dgl::ops {
    models::UNetControls compute_unet_controls(const std::vector<const models::ControlBinding*>& bindings,
                                               const Tensor<float>& input,
                                               const Tensor<float>& timesteps,
                                               const SDCondition& condition) {
        if (bindings.size() == 1) {
            const auto& binding = *bindings.front();
            return {binding.model->compute_sample_controls(binding, input, timesteps, condition), binding.strength};
        }

        models::UNetControls controls;
        for (const auto* binding : bindings) {
            auto residuals = binding->model->compute_sample_controls(*binding, input, timesteps, condition);
            if (controls.residuals.empty()) {
                controls.residuals = std::move(residuals);
                for (auto& residual : controls.residuals)
                    residual *= binding->strength;
                continue;
            }
            if (controls.residuals.size() != residuals.size())
                throw std::runtime_error("ControlNet residual counts differ");
            for (size_t i = 0; i < controls.residuals.size(); ++i) {
                if (controls.residuals[i].shape() != residuals[i].shape())
                    throw std::runtime_error("ControlNet residual shapes differ");
                controls.residuals[i] += residuals[i] * binding->strength;
            }
        }
        return controls;
    }
}  // namespace sd::dgl::ops
