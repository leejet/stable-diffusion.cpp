#ifndef __SD_DGL_MODELS_DENOISING_PARAMETERIZATION_H__
#define __SD_DGL_MODELS_DENOISING_PARAMETERIZATION_H__

#include <memory>
#include <vector>

#include "core/option_types.h"
#include "core/tensor.hpp"
#include "model.h"

struct Denoiser;

namespace sd::dgl::models {
    class DenoisingParameterization {
        SDVersion version_;
        prediction_t prediction_;
        float default_flow_shift_;
        std::shared_ptr<Denoiser> denoiser_;
        std::shared_ptr<Denoiser> (*copy_)(const Denoiser&) = nullptr;

    public:
        DenoisingParameterization(SDVersion version, prediction_t prediction, const String2TensorStorage& tensors, const std::vector<float>& alphas_cumprod);
        prediction_t prediction() const { return prediction_; }
        Denoiser* native() const { return denoiser_.get(); }
        bool is_flow() const;
        bool compatible_noise(const DenoisingParameterization& other) const;
        bool compatible_forward(const DenoisingParameterization& other) const;
        bool compatible_scalings(const DenoisingParameterization& other) const;
        float timestep(float sigma) const;
        std::vector<float> scalings(float sigma) const;
        std::shared_ptr<Denoiser> schedule(float flow_shift, int sequence_length = 0) const;
        float schedule_sigma(double t, float flow_shift, int sequence_length) const;
        Tensor<float> add_noise(float sigma, const Tensor<float>& noise, const Tensor<float>& base) const;
        Tensor<float> finish(float sigma, const Tensor<float>& latent) const;
    };
}

#endif  // __SD_DGL_MODELS_DENOISING_PARAMETERIZATION_H__
