#include "dgl/ops/preview.h"

#include "dgl/models/denoising_parameterization.h"

#include <cstdint>

#include "api/tensor.h"
#include "core/ggml_tensor_utils.h"
#include "dgl/models/circular_axes.h"
#include "dgl/models/diffusion_prediction.h"
#include "dgl/models/dit.h"
#include "dgl/models/fwd.h"
#include "dgl/models/vae.h"
#include "dgl/run_events.h"
#include "model/vae/vae.hpp"
#include "model_definition.h"
#include "runtime/denoiser.hpp"
#include "runtime/latent-preview.h"

namespace sd::dgl::ops {
    static void preview_image(int step, const Tensor<float>& latents, SDVersion version, preview_t preview_mode, models::VAE* decoder, sd_preview_cb_t step_callback, void* step_callback_data, bool is_noisy) {
        bool is_video           = preview_latent_tensor_is_video(latents);
        uint32_t dim            = is_video ? static_cast<uint32_t>(latents.shape()[3]) : static_cast<uint32_t>(latents.shape()[2]);
        const bool sefi         = sd_version_is_sefi_image(version);
        const int channel_start = sefi ? SefiFlowDenoiser::kSemChannels : 0;
        const int channels      = sefi                                      ? SefiFlowDenoiser::kTotalChannels - channel_start
                                  : decoder && preview_mode != PREVIEW_PROJ ? decoder->get_latent_channel()
                                                                            : find_model_definition(version)->geometry.latent_channels;
        auto _latents           = channels != dim || channel_start
                                      ? sd::ops::slice(latents, is_video ? 3 : 2, channel_start, channel_start + channels)
                                      : latents;
        if (preview_mode == PREVIEW_PROJ) {
            int patch_sz                      = 1;
            const float (*latent_rgb_proj)[3] = nullptr;
            float* latent_rgb_bias            = nullptr;
            const float* latent_alpha_proj    = nullptr;
            float latent_alpha_bias           = 1.f;

            if (channels == 128) {
                if (sd_version_uses_flux2_vae(version)) {
                    latent_rgb_proj = flux2_latent_rgb_proj;
                    latent_rgb_bias = flux2_latent_rgb_bias;
                    patch_sz        = 2;
                } else if (version == VERSION_LTXAV) {
                    latent_rgb_proj = ltxav_latent_rgb_proj;
                    latent_rgb_bias = ltxav_latent_rgb_bias;
                } else {
                    LOG_WARN("No latent to RGB projection known for this model");
                    return;
                }
            } else if (channels == 64) {
                if (version == VERSION_QWEN_IMAGE_2_1) {
                    latent_rgb_proj   = qwen21_latent_rgb_proj;
                    latent_rgb_bias   = qwen21_latent_rgb_bias;
                    latent_alpha_proj = qwen21_latent_alpha_proj;
                    latent_alpha_bias = qwen21_latent_alpha_bias;
                } else {
                    LOG_WARN("No latent to RGB projection known for this model");
                    return;
                }
            } else if (channels == 48) {
                if (sd_version_is_wan(version)) {
                    latent_rgb_proj = wan_22_latent_rgb_proj;
                    latent_rgb_bias = wan_22_latent_rgb_bias;
                } else {
                    LOG_WARN("No latent to RGB projection known for this model");
                    return;
                }
            } else if (channels == 24) {
                if (sd_version_is_minimax_h3(version)) {
                    latent_rgb_proj = minimax_latent_rgb_proj;
                    latent_rgb_bias = minimax_latent_rgb_bias;
                } else {
                    LOG_WARN("No latent to RGB projection known for this model");
                    return;
                }
            } else if (channels == 16) {
                if (sd_version_is_sd3(version)) {
                    latent_rgb_proj = sd3_latent_rgb_proj;
                    latent_rgb_bias = sd3_latent_rgb_bias;
                } else if (sd_version_uses_flux_vae(version)) {
                    latent_rgb_proj = flux_latent_rgb_proj;
                    latent_rgb_bias = flux_latent_rgb_bias;
                } else if (sd_version_uses_wan_vae(version)) {
                    latent_rgb_proj = wan_21_latent_rgb_proj;
                    latent_rgb_bias = wan_21_latent_rgb_bias;
                } else {
                    LOG_WARN("No latent to RGB projection known for this model");
                    return;
                }
            } else if (channels == 4) {
                if (sd_version_is_sdxl(version)) {
                    latent_rgb_proj = sdxl_latent_rgb_proj;
                    latent_rgb_bias = sdxl_latent_rgb_bias;
                } else if (sd_version_is_sd1(version) || sd_version_is_sd2(version)) {
                    latent_rgb_proj = sd_latent_rgb_proj;
                    latent_rgb_bias = sd_latent_rgb_bias;
                } else {
                    LOG_WARN("No latent to RGB projection known for this model");
                    return;
                }
            } else if (channels != 3) {
                LOG_WARN("No latent to RGB projection known for this model (dim = %d)", dim);
                return;
            }

            uint32_t frames     = is_video ? static_cast<uint32_t>(_latents.shape()[2]) : 1;
            uint32_t img_width  = static_cast<uint32_t>(_latents.shape()[0]) * patch_sz;
            uint32_t img_height = static_cast<uint32_t>(_latents.shape()[1]) * patch_sz;

            uint32_t img_channels = latent_alpha_proj != nullptr ? 4 : 3;
            const size_t stride   = size_t(img_width) * img_height * img_channels;
            std::vector<uint8_t> data(frames * stride);
            preview_latent_video(data.data(), _latents, latent_rgb_proj, latent_rgb_bias, patch_sz, latent_alpha_proj, latent_alpha_bias);
            std::vector<sd::Image> images(frames);
            for (uint32_t i = 0; i < frames; i++) {
                images[i] = {img_width, img_height, img_channels, data.data() + i * stride};
            }
            sd::api::preview_images(step_callback, step, frames, images.data(), is_noisy, step_callback_data);
            return;
        }

        if (decoder && (preview_mode == PREVIEW_VAE || preview_mode == PREVIEW_TAE)) {
            auto vae_latents = decoder->dit_to_vae_latents(_latents);
            auto decoded     = decoder->first_stage_model->decode(decoder->n_threads, vae_latents, decoder->vae_tiling_params, is_video,
                                                                  decoder->circular_x, decoder->circular_y, true);
            if (decoded.empty()) {
                LOG_ERROR("preview decode failed at step %d", step);
                return;
            }

            is_video        = preview_latent_tensor_is_video(decoded);
            uint32_t frames = is_video ? static_cast<uint32_t>(decoded.shape()[2]) : 1;
            std::vector<sd::Image> images(frames);
            std::vector<std::unique_ptr<uint8_t, decltype(&std::free)>> pixels;
            pixels.reserve(frames);
            for (uint32_t i = 0; i < frames; ++i) {
                images[i] = tensor_to_sd_image(decoded, static_cast<int>(i));
                pixels.emplace_back(images[i].data, std::free);
            }

            sd::api::preview_images(step_callback, step, frames, images.data(), is_noisy, step_callback_data);
            return;
        }

        if (preview_mode != PREVIEW_NONE) {
            LOG_WARN("Unsupported preview mode: %d", static_cast<int>(preview_mode));
        }
    }

    class PreviewSession final : public SamplingObserverSession {
        SamplingRun run_;
        std::shared_ptr<models::VAE> decoder_;
        preview_t mode_;
        bool circular_x_;
        bool circular_y_;

        bool selected(const SamplingStep& step) const {
            if (!step.completes_step)
                return false;
            const auto interval = run_.events.callbacks().interval;
            if (interval > 0)
                return step.index % interval == 0;
            if (interval < 0)
                return step.index == -static_cast<int64_t>(interval);
            return step.index == run_.total_steps &&
                   (run_.events.uses_legacy_preview() || run_.events.preview_info().sample_pass == 1);
        }

        void show(const SamplingStep& step, const Tensor<float>& latent, bool noisy) {
            auto& events = run_.events;
            events.check_cancelled();
            const auto mode = mode_;
            if (mode == PREVIEW_PROJ) {
                sd::ParallelScope scope(run_.dit.tensor_executor.get());
                preview_image(step.index, latent, run_.dit.version(), mode, nullptr,
                              events.preview_callback(), &events, noisy);
            } else {
                models::Model::ExecutionScope model_scope(*decoder_);
                model_scope.require_ready();
                models::CircularAxesScope axes(*decoder_, circular_x_, circular_y_);
                preview_image(step.index, latent, run_.dit.version(), mode, decoder_.get(),
                              events.preview_callback(), &events, noisy);
            }
            events.check_cancelled();
        }

    public:
        PreviewSession(SamplingRun run, std::shared_ptr<models::VAE> decoder, preview_t mode, bool circular_x, bool circular_y)
            : run_(run), decoder_(std::move(decoder)), mode_(mode), circular_x_(circular_x), circular_y_(circular_y) {}

        void before_prediction(const SamplingStep& step, const Tensor<float>& x) override {
            if (!selected(step) || !run_.events.callbacks().noisy)
                return;
            auto noisy = x * run_.parameterization.scalings(step.sigma)[2];
            if (step.prediction.mask_input)
                step.prediction.blend_mask(noisy);
            show(step, noisy, true);
        }

        void after_prediction(const SamplingStep& step, const Tensor<float>&, const sd::guidance::GuiderOutput& prediction) override {
            if (selected(step) && run_.events.callbacks().denoised)
                show(step, prediction.pred, false);
        }
    };

    PreviewObserver::PreviewObserver(std::shared_ptr<models::VAE> decoder, bool circular_x, bool circular_y)
        : decoder_(std::move(decoder)), circular_x_(circular_x), circular_y_(circular_y) {}

    std::unique_ptr<SamplingObserverSession> PreviewObserver::start(const SamplingRun& run) const {
        const auto& callbacks = run.events.callbacks();
        if (callbacks.mode == PREVIEW_NONE || !run.events.preview_callback() || (!callbacks.noisy && !callbacks.denoised))
            return {};
        return std::make_unique<PreviewSession>(run, decoder_, decoder_ ? callbacks.mode : PREVIEW_PROJ, circular_x_, circular_y_);
    }
}  // namespace sd::dgl::ops
