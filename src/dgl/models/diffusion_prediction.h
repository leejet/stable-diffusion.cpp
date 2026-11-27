#ifndef __SD_DGL_MODELS_DIFFUSION_PREDICTION_H__
#define __SD_DGL_MODELS_DIFFUSION_PREDICTION_H__

#include <cstdint>
#include <functional>
#include <limits>

#include "core/tensor.hpp"
#include "model/common/rope_circular.hpp"

struct AttentionProcessor;
struct FluxImageBlockProcessor;
struct UNetBlockProcessor;
struct SDCondition;

namespace sd_sample {
    struct SampleStepCacheDispatcher;
}

namespace sd::dgl::models {

    struct ControlBinding;
    struct UNetControls;

    struct DiffusionPrediction {
        sd::Tensor<float> output;
        float skip_scale   = 0.f;
        float output_scale = 0.f;
    };

    struct PredictionParameters {
        float sigma, timestep, input_scale, skip_scale, output_scale;
    };

    struct DiffusionPredictionOptions {
        sd::Tensor<float> vace_context, audio_embedding, video_positions;
        std::vector<int> skip_layers;
        float guidance                      = 3.5f;
        float vace_strength                 = 1.f;
        float fps                           = 24.f;
        float video_sigma_shift             = 12.f;
        float audio_sigma_shift             = 3.f;
        int audio_length                    = 0;
        Rope::RefIndexMode ref_index_mode   = Rope::RefIndexMode::FIXED;
        Rope::RefRoPEAnchor ref_rope_anchor = Rope::RefRoPEAnchor::MODEL_DEFAULT;
        bool force_ref_timestep_zero        = false;
        std::function<PredictionParameters(PredictionParameters)> transform;
    };

    struct DiffusionPredictionStep {
        const sd::Tensor<float>& base;
        const sd::Tensor<float>& mask;
        int index            = 1;
        size_t schedule_size = 0;
        // Native caches use the signed solver marker to identify intermediate evaluations.
        int solver_index                            = 1;
        bool completes_step                         = true;
        float target_sigma                          = std::numeric_limits<float>::quiet_NaN();
        float lookahead_sigma                       = std::numeric_limits<float>::quiet_NaN();
        float semantic_sigma                        = std::numeric_limits<float>::quiet_NaN();
        bool mask_input                             = false;
        sd_sample::SampleStepCacheDispatcher* cache = nullptr;
        std::vector<const ControlBinding*> controls;
        std::function<UNetControls(const Tensor<float>&, const Tensor<float>&, const SDCondition&)> compute_controls;
        const AttentionProcessor* attn2_processor           = nullptr;
        const FluxImageBlockProcessor* flux_image_processor = nullptr;
        const UNetBlockProcessor* unet_block_processor      = nullptr;
        uint64_t qwen_prefix_id                             = 0;

        void blend_mask(Tensor<float>& latent) const {
            if (!mask.empty())
                latent = latent * mask + base * (1.f - mask);
        }
    };

}  // namespace sd::dgl::models

#endif  // __SD_DGL_MODELS_DIFFUSION_PREDICTION_H__
