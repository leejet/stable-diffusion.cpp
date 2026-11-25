#include "dgl/ops/vae.h"

#include "core/image_tensor.h"
#include "dgl/models/circular_axes.h"
#include "dgl/models/vae.h"

namespace sd::dgl::ops {

    class VAETilingScope {
        models::VAE& model_;
        sd::TilingConfig previous_;

    public:
        VAETilingScope(models::VAE& model, const VAEOptions& options)
            : model_(model), previous_(model.vae_tiling_params) {
            if (options.tiling)
                model_.vae_tiling_params = *options.tiling;
        }
        ~VAETilingScope() { model_.vae_tiling_params = previous_; }
    };

    Tensor<float> encode_vae(models::VAE& model, const Tensor<float>& pixels, const std::shared_ptr<RNG>& rng, bool posterior_mode, const VAEOptions& options) {
        models::Model::ExecutionScope scope(model);
        scope.require_ready();
        models::CircularAxesScope axes(model, options.circular_x, options.circular_y);
        VAETilingScope tiling(model, options);
        if (pixels.empty() || (pixels.dim() != 4 && pixels.dim() != 5) || pixels.shape().back() != 1)
            throw std::runtime_error("expected pixels with width, height, optional time, channels and batch");
        const auto channels = pixels.shape()[pixels.dim() == 5 ? 3 : 2];
        const auto expected = model.get_image_channels();
        if (channels != expected && ((expected != 3 && expected != 4) || channels == 2))
            throw std::runtime_error("pixel channels do not match the VAE");
        const auto adjusted = channels == expected ? Tensor<float>{} : sd::ensure_image_tensor_channels(pixels, expected);
        auto output         = model.encode_vae_latents(channels == expected ? pixels : adjusted, posterior_mode, rng);
        if (output.empty())
            throw std::runtime_error("VAE encode failed");
        return output;
    }

    Tensor<float> decode_vae(models::VAE& model, const Tensor<float>& latent, const VAEOptions& options) {
        models::Model::ExecutionScope scope(model);
        scope.require_ready();
        models::CircularAxesScope axes(model, options.circular_x, options.circular_y);
        VAETilingScope tiling(model, options);
        model.validate_vae_latent(latent);
        if (latent.dim() == 4 && latent.shape().back() > 1) {
            Tensor<float> video;
            for (int64_t i = 0; i < latent.shape().back(); ++i) {
                auto frame = model.decode_vae_latents(sd::ops::slice(latent, 3, i, i + 1), false);
                if (frame.empty() || frame.dim() != 4 || frame.shape().back() != 1)
                    throw std::runtime_error("VAE did not return an individual frame");
                if (video.empty()) {
                    auto shape = frame.shape();
                    shape.insert(shape.begin() + 2, latent.shape().back());
                    size_t remaining = (256 * 1024 * 1024) / sizeof(float);
                    for (const auto dim : shape) {
                        if (dim <= 0 || static_cast<uint64_t>(dim) > remaining)
                            throw std::runtime_error("decoded video exceeds 256 MiB");
                        remaining /= static_cast<size_t>(dim);
                    }
                    video = Tensor<float>(std::move(shape));
                }
                sd::ops::slice_assign(&video, 2, i, i + 1, frame.unsqueeze(2));
            }
            return video;
        }
        auto output = model.decode_vae_latents(latent, latent.dim() == 5);
        if (output.empty())
            throw std::runtime_error("VAE did not return pixels");
        return output;
    }

}  // namespace sd::dgl::ops
