#include "guidance.h"

#include <utility>

namespace sd::guidance {

    static bool has_tensor(const sd::Tensor<float>* tensor) {
        return tensor != nullptr && !tensor->empty();
    }

    ClassifierFreeGuidance::ClassifierFreeGuidance(float guidance_scale,
                                                   float image_guidance_scale)
        : guidance_scale_(guidance_scale),
          image_guidance_scale_(image_guidance_scale) {
    }

    GuiderOutput ClassifierFreeGuidance::forward(const GuidanceInput& input,
                                                 GuiderOutput previous) const {
        (void)previous;

        GuiderOutput output;
        if (!has_tensor(input.pred_cond)) {
            return output;
        }

        const sd::Tensor<float>& pred_cond = *input.pred_cond;
        output.pred                        = pred_cond;
        if (has_tensor(input.pred_uncond)) {
            const sd::Tensor<float>& pred_uncond = *input.pred_uncond;
            if (has_tensor(input.pred_img_cond)) {
                const sd::Tensor<float>& pred_img_cond = *input.pred_img_cond;
                output.pred                            = pred_uncond +
                              image_guidance_scale_ * (pred_img_cond - pred_uncond) +
                              guidance_scale_ * (pred_cond - pred_img_cond);
            } else {
                output.pred = pred_uncond + guidance_scale_ * (pred_cond - pred_uncond);
            }
        } else if (has_tensor(input.pred_img_cond)) {
            const sd::Tensor<float>& pred_img_cond = *input.pred_img_cond;
            output.pred                            = pred_img_cond + guidance_scale_ * (pred_cond - pred_img_cond);
        }

        return output;
    }

    SkipLayerGuidance::SkipLayerGuidance(std::vector<int> layers,
                                         float scale,
                                         float start,
                                         float stop)
        : layers_(std::move(layers)),
          scale_(scale),
          start_(start),
          stop_(stop) {
    }

    bool SkipLayerGuidance::is_enabled_for_step(const GuidanceInput& input) const {
        if (scale_ == 0.0f || layers_.empty() || input.schedule_size == 0) {
            return false;
        }

        int start_step = static_cast<int>(start_ * static_cast<float>(input.schedule_size));
        int stop_step  = static_cast<int>(stop_ * static_cast<float>(input.schedule_size));
        return input.step > start_step && input.step < stop_step;
    }

    const std::vector<int>& SkipLayerGuidance::layers() const {
        return layers_;
    }

    GuiderOutput SkipLayerGuidance::forward(const GuidanceInput& input,
                                            GuiderOutput output) const {
        if (!is_enabled_for_step(input) || !input.predict_skip_layer) {
            return output;
        }

        if (output.pred.empty() || !has_tensor(input.pred_cond)) {
            return GuiderOutput();
        }

        output.pred_skip_layer = input.predict_skip_layer();
        if (output.pred_skip_layer.empty()) {
            return GuiderOutput();
        }

        output.pred += (*input.pred_cond - output.pred_skip_layer) * scale_;
        return output;
    }

}  // namespace sd::guidance
