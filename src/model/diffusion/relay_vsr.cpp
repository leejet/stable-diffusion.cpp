#include "relay_vsr.hpp"

namespace RelayVSR {

    static int64_t aligned_vsr_size(int64_t size, int scale) {
        const int alignment = 32 / scale;
        return (size + alignment - 1) / alignment * alignment;
    }

    static sd::Tensor<float> pad_vsr_frame(sd::Tensor<float> result, int64_t width, int64_t height) {
        const auto w = result.shape()[0], h = result.shape()[1];
        if (width > w) {
            auto edge    = sd::ops::slice(result, 0, w - 1, w);
            auto padding = sd::zeros<float>({width - w, h, 3, 1}) + edge;
            result       = sd::ops::concat(result, padding, 0);
        }
        if (height > h) {
            auto edge    = sd::ops::slice(result, 1, h - 1, h);
            auto padding = sd::zeros<float>({width, height - h, 3, 1}) + edge;
            result       = sd::ops::concat(result, padding, 1);
        }
        return result;
    }

    static sd::Tensor<float> prepare_vsr_lq(const sd::Tensor<float>& frame, int scale) {
        auto input = frame;
        if (scale == 4)
            input = sd::ops::interpolate(frame, {frame.shape()[0] * 4, frame.shape()[1] * 4, 3, 1}, sd::ops::InterpolateMode::Bilinear);
        const int patch = 4 * scale;
        const int width = int(input.shape()[0]), height = int(input.shape()[1]);
        input.reshape_({patch, width / patch, patch, height / patch, 3, 1});
        input.permute_({1, 3, 0, 2, 4, 5});
        input.reshape_({width / patch, height / patch, 3 * patch * patch, 1});
        return input;
    }

    static std::vector<int64_t> vsr_latent_shape(int scale, int64_t width, int64_t height, int64_t frames) {
        const int stride = 16 / scale;
        return {aligned_vsr_size(width, scale) / stride, aligned_vsr_size(height, scale) / stride, frames, 48, 1};
    }

    sd::Tensor<float> WanVSRRunner::compute(int n_threads, const DiffusionParams& params) {
        const auto& noise = *params.x;
        if (!params.ref_latents || params.ref_latents->size() != 1)
            throw std::runtime_error("RelayVSR requires one low-quality keyframe tensor");
        const auto& lq = params.ref_latents->front();
        if (lq.dim() != 5 || lq.shape()[3] != 3 || lq.shape()[4] != 1)
            throw std::runtime_error("RelayVSR requires lq_frames with shape [width, height, keyframes, 3, 1]");
        const auto w = lq.shape()[0], h = lq.shape()[1], count = lq.shape()[2];
        const int scale = config.upscale_factor;
        if (noise.shape() != vsr_latent_shape(scale, w, h, count))
            throw std::runtime_error("RelayVSR noise shape does not match the low-quality keyframes");
        if (!params.context || params.context->shape() != std::vector<int64_t>{config.text_dim, config.text_len})
            throw std::runtime_error("RelayVSR requires a compatible positive embedding");
        reset();
        struct Reset {
            WanVSRRunner& model;
            ~Reset() { model.reset(); }
        } cleanup{*this};
        const auto time = time_embedding((*params.timesteps)[0]);
        const auto pw = aligned_vsr_size(w, scale), ph = aligned_vsr_size(h, scale);
        sd::Tensor<float> output(noise.shape());
        for (int64_t i = 0; i < count; ++i) {
            auto input = sd::ops::slice(lq, 2, i, i + 1);
            input.reshape_({w, h, 3, 1});
            input *= 2.f;
            input -= 1.f;
            input       = pad_vsr_frame(std::move(input), pw, ph);
            auto sample = sd::ops::slice(noise, 2, i, i + 1);
            sample.reshape_({noise.shape()[0], noise.shape()[1], 1, 48});
            auto velocity = predict_frame(n_threads, sample, prepare_vsr_lq(input, scale), time, params.context);
            if (velocity.empty())
                return {};
            velocity.reshape_({noise.shape()[0], noise.shape()[1], 1, 48, 1});
            sd::ops::slice_assign(&output, 2, i, i + 1, velocity);
        }
        return output;
    }

}  // namespace RelayVSR
