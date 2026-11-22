#ifndef __SD_DGL_MODELS_DIT_H__
#define __SD_DGL_MODELS_DIT_H__

#include <cmath>
#include <map>
#include <mutex>
#include "core/tensor.hpp"
#include "dgl/models/fwd.h"
#include "dgl/models/model_runtime.h"

struct SDCondition;

namespace sd::dgl::models {
    class DenoisingParameterization;

    struct DiTLoadParams {
        ModelLoadParams model;
    };

    class DiT final : public Model {
        ModelRuntime runtime_;
        prediction_t default_prediction_ = PREDICTION_COUNT;
        std::recursive_mutex prediction_mutex_;
        std::map<prediction_t, std::shared_ptr<const DenoisingParameterization>> parameterizations_;

        bool refresh_weights() override { return runtime_.refresh_weights(); }
        bool attach_loras() override { return runtime_.attach_loras(); }

        std::shared_ptr<DiffusionModelRunner> high_noise_diffusion_model;
        std::vector<float> file_alphas_cumprod;
        bool reuse_file_alphas_ = false;
        bool build_variant(const Model& base, ModelLoader loader) override;
        bool build_runners();
        bool is_using_v_parameterization_for_sd2(bool is_inpaint);
        bool set_sage_attention_enabled(bool enabled);
        void load_alphas_cumprod();
        int get_diffusion_model_down_factor();
        std::vector<float> process_timesteps(const std::vector<float>& timesteps, const Tensor<float>& init_latent, const Tensor<float>& denoise_mask);
        std::vector<float> process_ltxav_video_timesteps(const std::vector<float>& timesteps,
                                                         const Tensor<float>& init_latent,
                                                         const Tensor<float>& denoise_mask);

    public:
        SDVersion version() const { return runtime_.version; }
        int get_latent_channel() const;
        bool circular_x = false;
        bool circular_y = false;
        std::shared_ptr<DiffusionModelRunner> diffusion_model;

        DiT(std::shared_ptr<ModelManager> manager, std::shared_ptr<SDBackendPool> backends);
        bool init(DiTLoadParams params);
        void apply_circular_axes(bool x, bool y);
        std::shared_ptr<const DenoisingParameterization> make_parameterization(prediction_t prediction = PREDICTION_COUNT);
        DiffusionPrediction predict_diffusion(const Tensor<float>& x, const DenoisingParameterization& parameterization, float sigma, const SDCondition& condition, const std::vector<Tensor<float>>& references, const DiffusionPredictionOptions& options, const DiffusionPredictionStep& step);
    };
}

#endif  // __SD_DGL_MODELS_DIT_H__
