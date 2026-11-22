#include "dgl/ops/relay_vsr.h"

#include "dgl/models/fwd.h"
#include "dgl/ops/tensor.h"
#include "dgl/run_events.h"
#include "json.hpp"
#include "model/diffusion/relay_vsr.hpp"
#include "model/upscaler/flash_vsr.hpp"

namespace sd {
    namespace dgl::ops {

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

        static sd::Tensor<float> prepare_vsr_frame(const sd::Image& frame, int64_t width, int64_t height) {
            auto result = sd_image_to_tensor(frame, -1, -1, false);
            result /= 127.5f;
            result -= 1.f;
            return pad_vsr_frame(std::move(result), width, height);
        }

        static sd::Tensor<float> finish_vsr_frame(const sd::Tensor<float>& input, const sd::Tensor<float>& residual, int scale, int64_t width, int64_t height, float detail) {
            auto output = sd::ops::interpolate(input, {input.shape()[0] * scale, input.shape()[1] * scale, 3, 1}, sd::ops::InterpolateMode::Bicubic);
            if (detail == 1.f)
                output += residual;
            else
                output += residual * detail;
            output += 1.f;
            output *= 127.5f;
            output = sd::ops::slice(sd::ops::slice(output, 0, 0, width), 1, 0, height);
            if (!std::all_of(output.values().begin(), output.values().end(), [](float value) { return std::isfinite(value); }))
                throw std::runtime_error("RelayVSR produced non-finite pixels");
            output = sd::ops::clamp(output, 0.f, 255.f);
            output.permute_({2, 0, 1, 3});
            return output;
        }

        static std::vector<int64_t> vsr_latent_shape(int scale, int64_t width, int64_t height, int64_t frames) {
            const int stride = 16 / scale;
            return {aligned_vsr_size(width, scale) / stride, aligned_vsr_size(height, scale) / stride, frames, 48, 1};
        }

        std::vector<int64_t> relay_vsr_latent_shape(const DiffusionModelRunner& model, int64_t width, int64_t height, int64_t frames) {
            const auto* wan = dynamic_cast<const RelayVSR::WanVSRRunner*>(&model);
            if (!wan)
                throw std::runtime_error("expected a RelayVSR DiT");
            return vsr_latent_shape(wan->config.upscale_factor, width, height, frames);
        }

        Tensor<float> relay_vsr_positive_embedding(const DiffusionModelRunner& model, ModelManager::ResourceHandle& weights) {
            const auto* wan = dynamic_cast<const RelayVSR::WanVSRRunner*>(&model);
            if (!wan)
                throw std::runtime_error("expected a RelayVSR DiT");
            std::vector<float> values;
            if (!weights.load_float_tensor("wan.prompt_embeds", values))
                throw std::runtime_error("cannot read RelayVSR positive embedding");
            auto tensor = Tensor<float>::from_vector(std::move(values));
            tensor.reshape_({wan->config.text_dim, wan->config.text_len});
            return tensor;
        }

        Tensor<uint8_t> relay_vsr_decode(::VAE& model, int threads, const Tensor<float>& latent, const Tensor<uint8_t>& frames, const Tensor<int64_t>& indices, float detail) {
            auto* flash = dynamic_cast<RelayVSR::FlashVSRRunner*>(&model);
            if (!flash)
                throw std::runtime_error("expected a RelayVSR Flash VAE");
            const int scale = flash->config.upscale_factor;
            if (frames.dim() != 4 || frames.shape()[0] != 3 || frames.empty())
                throw std::runtime_error("RelayVSR expects nonempty U8 RGB frames");
            const auto w = frames.shape()[1], h = frames.shape()[2], count = frames.shape()[3];
            const auto& anchors = indices.values();
            if (indices.dim() != 1 || anchors.empty() || anchors.front() != 0 || anchors.back() != count - 1 ||
                std::adjacent_find(anchors.begin(), anchors.end(), std::greater_equal<int64_t>()) != anchors.end())
                throw std::runtime_error("RelayVSR keyframe indices must increase from the first to the last frame");
            if (latent.shape() != vsr_latent_shape(scale, w, h, indices.numel()))
                throw std::runtime_error("RelayVSR keyframe latents do not match the frames, keyframe indices or decoder scale");
            const auto ow = w * scale, oh = h * scale;
            if (ow > 16384 || oh > 16384)
                throw std::runtime_error("RelayVSR output dimensions exceed 16384 pixels");
            element_count({3, ow, oh, count}, SD_TENSOR_U8);

            flash->reset();
            struct Reset {
                RelayVSR::FlashVSRRunner& model;
                ~Reset() { model.reset(); }
            } reset{*flash};
            Tensor<uint8_t> output({3, ow, oh, count});
            const auto pw = aligned_vsr_size(w, scale), ph = aligned_vsr_size(h, scale);
            auto cache_anchor = [&](int64_t i, int slot) {
                auto frame = sd::ops::slice(latent, 2, i, i + 1);
                frame.reshape_({latent.shape()[0], latent.shape()[1], 48, 1});
                if (flash->compute(threads, frame, slot).empty())
                    throw std::runtime_error("RelayVSR Flash could not encode a keyframe");
            };
            int start_slot = 0;
            cache_anchor(0, start_slot);
            for (size_t i = 0; i < std::max<size_t>(1, anchors.size() - 1); ++i) {
                const bool has_next = i + 1 < anchors.size();
                const auto first = anchors[i], last = has_next ? anchors[i + 1] : first;
                const int end_slot = has_next ? 1 - start_slot : start_slot;
                if (has_next)
                    cache_anchor(int64_t(i + 1), end_slot);
                flash->history = false;
                for (auto j = first; j <= last; ++j) {
                    check_run_cancelled();
                    sd::Image frame{uint32_t(w), uint32_t(h), 3, const_cast<uint8_t*>(frames.data() + j * 3 * w * h)};
                    auto input    = prepare_vsr_frame(frame, pw, ph);
                    auto residual = flash->compute(threads, input, -1, start_slot, end_slot);
                    if (residual.empty())
                        throw std::runtime_error("RelayVSR Flash decoding failed");
                    if (i != 0 && j == first)
                        continue;
                    auto pixels = finish_vsr_frame(input, residual, scale, ow, oh, detail);
                    std::transform(pixels.values().begin(), pixels.values().end(), output.data() + j * 3 * ow * oh,
                                   [](float pixel) { return uint8_t(pixel); });
                    report_run_frames("video_upscale", int(j + 1), int(count));
                }
                start_slot = end_slot;
            }
            return output;
        }

    }  // namespace dgl::ops
}  // namespace sd
