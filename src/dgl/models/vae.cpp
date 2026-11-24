#include "dgl/models/vae.h"

#include "model/vae/geometry.hpp"

#include "core/backend_fit.h"
#include "core/ggml_tensor_utils.h"
#include "core/image_geometry.h"
#include "core/util.h"
#include "dgl/models/component_builders.h"
#include "dgl/models/component_sources.h"
#include "dgl/run_events.h"
#include "model/vae/auto_encoder_kl.hpp"
#include "model/vae/ltx_vae.hpp"
#include "model/vae/vae.hpp"
#include "model_definition.h"

namespace sd::dgl::models {

    VAE::VAE(std::shared_ptr<ModelManager> manager, std::shared_ptr<SDBackendPool> backends)
        : Model(std::move(manager)),
          runtime_(*this, std::move(backends), ModelComponent::VAE, SDBackendModule::VAE, [this] { return build_runners(); }) {}

    int VAE::get_latent_channel() const {
        return find_model_definition(version())->geometry.latent_channels;
    }

    void VAE::apply_circular_axes(bool x, bool y) {
        circular_x = x;
        circular_y = y;
        first_stage_model->set_circular_axes(x, y);
        if (x || y)
            LOG_DEBUG("Using circular padding for convolutions (x=%s, y=%s)", x ? "true" : "false", y ? "true" : "false");
    }

    bool VAE::init(VAELoadParams params) {
        format_ = params.format;
        return runtime_.init(std::move(params.model), load_component_catalog<ModelComponent::VAE>);
    }

    static bool load_tae_catalog(ModelLoader& loader, const ModelLoadParams& params, SDVersion& family, FamilyProbeCache&) {
        if (family == VERSION_COUNT)
            throw std::runtime_error("load_tae requires family or family_from");
        if (!loader.init_from_file_and_convert_name(params.path, "tae.", family))
            return false;
        loader.set_wtype_override(sd_type_to_ggml_type(params.wtype), params.tensor_type_rules.c_str());
        return true;
    }

    bool VAE::init_tae(ModelLoadParams params, bool decode_only) {
        tae_         = true;
        decode_only_ = decode_only;
        return runtime_.init(std::move(params), load_tae_catalog);
    }

    bool VAE::build_variant(const Model& source, ModelLoader loader) {
        const auto& base = static_cast<const VAE&>(source);
        format_          = base.format_;
        tae_             = base.tae_;
        decode_only_     = base.decode_only_;
        return runtime_.build_variant(base.runtime_, std::move(loader));
    }

    bool VAE::build_runners() {
        auto ctx              = runtime_.model_build_context();
        ctx.params.vae_format = format_;
        first_stage_model     = uses_tae() ? component_builders::build_tae_runner(ctx, decode_only_)
                                           : component_builders::build_vae_runner(ctx, model_weights->loader());
        if (!first_stage_model)
            return false;
        runtime_.configure_runner(*first_stage_model, true);
        return runtime_.register_runner_params(runtime_.component, first_stage_model);
    }

    bool VAE::uses_tae() const {
        return tae_ || version() == VERSION_SDXS_512_DS || version() == VERSION_SDXS_09;
    }

    int VAE::get_vae_scale_factor() {
        if (sd_version_is_pid(version())) {
            return 1;
        }
        return first_stage_model->get_scale_factor();
    }

    int VAE::get_image_channels() const {
        return find_model_definition(version())->geometry.image_channels;
    }

    int VAE::video_frames_to_latent_frames(int frames) {
        int latent_frames = frames;
        if (sd_version_is_ltxav(version())) {
            latent_frames = ((frames - 1) / 8) + 1;
        } else if (sd_version_is_minimax_h3(version())) {
            latent_frames = frames <= 5 ? 2 : ((frames - 5) / 17) * 5 + 2;
        } else if (sd_version_is_wan(version()) || sd_version_is_lingbot_video(version()) || sd_version_is_hunyuan_video(version()) || version() == VERSION_SEEDVR2) {
            latent_frames = ((frames - 1) / 4) + 1;
        }
        return latent_frames;
    }

    int VAE::align_video_frames(int frames) {
        return sd::vae::aligned_frames(version(), frames);
    }

    std::vector<int64_t> VAE::latent_shape(int64_t width, int64_t height, int frames, int batch) {
        if (frames < 0 || batch <= 0)
            throw std::invalid_argument("invalid latent frame or batch count");
        const auto scale          = get_vae_scale_factor();
        const auto multiple       = scale * find_model_definition(version())->geometry.diffusion_down_factor;
        const auto aligned_width  = sd::align_image_dimension(width, multiple);
        const auto aligned_height = sd::align_image_dimension(height, multiple);
        if (width != aligned_width || height != aligned_height)
            LOG_VERBOSE("latent_shape: aligning pixel dimensions from %lldx%lld to %lldx%lld", (long long)width, (long long)height,
                        (long long)aligned_width, (long long)aligned_height);
        std::vector<int64_t> shape{aligned_width / scale, aligned_height / scale};
        if (frames) {
            const auto aligned_frames = align_video_frames(frames);
            if (aligned_frames != frames)
                LOG_VERBOSE("latent_shape: aligning frame count from %d to %d", frames, aligned_frames);
            shape.push_back(video_frames_to_latent_frames(aligned_frames));
        }
        shape.push_back(get_latent_channel());
        shape.push_back(batch);
        return shape;
    }

    void VAE::validate_vae_latent(const Tensor<float>& latent) const {
        const auto* kl      = dynamic_cast<const AutoEncoderKL*>(first_stage_model.get());
        const auto* ltx     = dynamic_cast<const LTXVideoVAE*>(first_stage_model.get());
        const auto rank     = latent.dim();
        const auto channels = kl ? kl->ae.get_latent_channels() : get_latent_channel();
        if ((rank != 4 && rank != 5) || (kl && rank != 4) || (ltx && rank != 5) ||
            latent.shape()[rank == 5 ? 3 : 2] != channels ||
            (rank == 5 && latent.shape().back() != 1)) {
            throw std::runtime_error("latent shape does not match the VAE");
        }
    }

    sd::Tensor<float> VAE::encode_vae_latents(const sd::Tensor<float>& x, bool posterior_mode, const std::shared_ptr<RNG>& rng) {
        if (decode_only_)
            throw std::runtime_error("this TAE was loaded with decode_only and cannot encode");
        report_run_progress("vae_encode");
        // Qwen 2.1 expands each vision slot into four reference latent tokens.
        const auto alignment = version() == VERSION_QWEN_IMAGE_2_1 ? 32 : get_vae_scale_factor();
        auto adjusted        = sd::fit_image_tensor(x, alignment, "vae_encode");
        const auto* input    = adjusted.empty() ? &x : &adjusted;
        if (input->dim() == 5 && input->shape()[2] > 1 && !sd_version_is_minimax_h3_image(version())) {
            const auto frames = align_video_frames(int(input->shape()[2]));
            if (frames != input->shape()[2]) {
                LOG_VERBOSE("vae_encode: padding video from %lld to %d frames by repeating the last frame", (long long)input->shape()[2], frames);
                adjusted = sd::pad_video_tensor(*input, frames);
                input    = &adjusted;
            }
        }
        auto latents = first_stage_model->encode(n_threads, *input, vae_tiling_params, circular_x, circular_y);
        if (latents.empty()) {
            return {};
        }
        return first_stage_model->vae_output_to_latents(latents, rng,
                                                        posterior_mode ? VAEPosterior::Mode : VAEPosterior::Sample,
                                                        true);
    }

    sd::Tensor<float> VAE::vae_to_dit_latents(const sd::Tensor<float>& latent) {
        auto* ltx = dynamic_cast<LTXVideoVAE*>(first_stage_model.get());
        return ltx ? ltx->normalize_latents(n_threads, latent)
                   : first_stage_model->vae_to_diffusion_latents(latent);
    }

    sd::Tensor<float> VAE::dit_to_vae_latents(const sd::Tensor<float>& latent) {
        auto* ltx = dynamic_cast<LTXVideoVAE*>(first_stage_model.get());
        return ltx ? ltx->un_normalize_latents(n_threads, latent)
                   : first_stage_model->diffusion_to_vae_latents(latent);
    }

    sd::Tensor<float> VAE::decode_vae_latents(const sd::Tensor<float>& latents, bool decode_video) {
        report_run_progress("vae_decode");
        auto tiling_params                = first_stage_model->resolve_tiling_params(vae_tiling_params);
        const bool prefer_temporal_tiling = decode_video && latents.dim() == 5 && latents.shape()[2] > 1 &&
                                            first_stage_model->can_temporal_tile_decode();
        for (;;) {
            int tile_size_w = static_cast<int>(latents.shape()[0]);
            int tile_size_h = static_cast<int>(latents.shape()[1]);
            float tile_overlap;
            if (tiling_params.enabled &&
                !first_stage_model->get_tile_sizes(tile_size_w, tile_size_h, tile_overlap, tiling_params,
                                                   latents.shape()[0], latents.shape()[1])) {
                return {};
            }
            auto decoded = first_stage_model->decode(n_threads, latents, tiling_params, decode_video, circular_x, circular_y);
            if (sd_version_is_minimax_h3_image(version()) && !decoded.empty() && decoded.dim() == 5) {
                decoded = decoded.squeeze(2);
            }
            if (!decoded.empty() ||
                !sd::backend_fit::prepare_vae_decode_retry_tiling(tiling_params, prefer_temporal_tiling,
                                                                  first_stage_model->last_compute_status(),
                                                                  tile_size_w, tile_size_h, first_stage_model->get_scale_factor())) {
                return decoded;
            }
        }
    }

}  // namespace sd::dgl::models
