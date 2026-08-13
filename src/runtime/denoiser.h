#ifndef __SD_RUNTIME_DENOISER_H__
#define __SD_RUNTIME_DENOISER_H__

#include "stable-diffusion.h"

#include "src/core/rng.hpp"
#include "src/core/tensor.hpp"
#include "src/model.h"
#include "src/runtime/guidance.h"

#include <functional>
#include <vector>

const int TIMESTEPS = 1000;

struct Denoiser {
    virtual float sigma_min()                                                        = 0;
    virtual float sigma_max()                                                        = 0;
    virtual float sigma_to_t(float sigma)                                            = 0;
    virtual float t_to_sigma(float t)                                                = 0;
    virtual std::vector<float> get_scalings(float sigma)                             = 0;
    virtual sd::Tensor<float> noise_scaling(float sigma,
                                            const sd::Tensor<float>& noise,
                                            const sd::Tensor<float>& latent)         = 0;
    virtual sd::Tensor<float> inverse_noise_scaling(float sigma,
                                                    const sd::Tensor<float>& latent) = 0;
    virtual float noise_level_to_sigma(float noise_level)                            = 0;

    virtual std::vector<float> get_sigmas(uint32_t n, int image_seq_len, scheduler_t scheduler_type, SDVersion version, const char* extra_sample_args = nullptr);
    virtual void refresh_compvis_denoiser(const std::vector<float>& file_alphas_cumprod);
    virtual scheduler_t get_default_scheduler();
    virtual bool is_flow_denoiser();
    virtual void set_shift(float shift);
    virtual std::vector<float> get_timesteps(int step);
};

std::shared_ptr<Denoiser> make_denoiser(prediction_t pred_type);

typedef std::function<sd::guidance::GuiderOutput(const sd::Tensor<float>&, float, int)> denoise_cb_t;

sd::Tensor<float> sample_k_diffusion(sample_method_t method,
                                     denoise_cb_t model,
                                     sd::Tensor<float> x,
                                     const std::vector<float>& sigmas,
                                     std::shared_ptr<RNG> rng,
                                     float eta,
                                     bool is_flow_denoiser,
                                     const char* extra_sample_args,
                                     std::shared_ptr<Denoiser> denoiser_for_dispatch = nullptr);

#endif  // __SD_RUNTIME_DENOISER_H__
