#include "dgl/models/dit.h"

#include <utility>

#include "conditioning/conditioner.hpp"
#include "core/ggml_tensor_utils.h"
#include "core/util.h"
#include "dgl/diagnostic.h"
#include "dgl/models/component_builders.h"
#include "dgl/models/component_sources.h"
#include "dgl/models/control.h"
#include "dgl/models/denoising_parameterization.h"
#include "dgl/models/diffusion_prediction.h"
#include "model/diffusion/animatediff.hpp"
#include "model/diffusion/minimax_h3.hpp"
#include "model/diffusion/z_image.hpp"
#include "model_definition.h"
#include "runtime/denoiser.hpp"
#include "runtime/guidance.h"
#include "runtime/sample-cache.h"

namespace sd::dgl::models {

    DiT::DiT(std::shared_ptr<ModelManager> manager, std::shared_ptr<SDBackendPool> backends)
        : Model(std::move(manager)),
          runtime_(*this, std::move(backends), ModelComponent::Diffusion, SDBackendModule::DIFFUSION, [this] { return build_runners(); }) {}

    int DiT::get_latent_channel() const {
        return find_model_definition(version())->geometry.latent_channels;
    }

    bool DiT::init(DiTLoadParams params) {
        return runtime_.init(std::move(params.model),
                             [](ModelLoader& loader, const ModelLoadParams& model, SDVersion& family, FamilyProbeCache& family_probes) {
                                 return init_component_loader(loader, model, ModelComponent::Diffusion, family, {}, &family_probes);
                             });
    }

    bool DiT::build_variant(const Model& source, ModelLoader loader) {
        const auto& base    = static_cast<const DiT&>(source);
        default_prediction_ = base.default_prediction_;
        file_alphas_cumprod = base.file_alphas_cumprod;
        reuse_file_alphas_  = true;
        return runtime_.build_variant(base.runtime_, std::move(loader));
    }

    bool DiT::build_runners() {
        parameterizations_.clear();
        if (!std::exchange(reuse_file_alphas_, false))
            load_alphas_cumprod();
        auto context              = runtime_.model_build_context();
        context.build_conditioner = false;
        context.build_diffusion   = true;
        context.build_auxiliary   = false;
        sd::model_builders::CoreRunners runners;
        if (!component_builders::build_family_runners(context, runners) || !runners.diffusion) {
            return false;
        }
        diffusion_model            = std::move(runners.diffusion);
        high_noise_diffusion_model = std::move(runners.high_noise_diffusion);
        runtime_.configure_runner(*diffusion_model, version() != VERSION_SEEDVR2);
        if (high_noise_diffusion_model) {
            runtime_.configure_runner(*high_noise_diffusion_model, true);
        }
        if (runtime_.params.sage_attn && !set_sage_attention_enabled(true)) {
            return false;
        }
        return runtime_.register_runner_params(runtime_.component, diffusion_model) &&
               runtime_.register_runner_params(ModelComponent::HighNoiseDiffusion, high_noise_diffusion_model);
    }

    void DiT::apply_circular_axes(bool x, bool y) {
        circular_x = x;
        circular_y = y;
        if (x || y)
            LOG_DEBUG("Using circular padding for convolutions (x=%s, y=%s)", x ? "true" : "false", y ? "true" : "false");
        diffusion_model->set_circular_axes(x, y);
        if (high_noise_diffusion_model) {
            high_noise_diffusion_model->set_circular_axes(x, y);
        }
    }

    static sd::Tensor<float> batch_two_condition_tensors(const sd::Tensor<float>& a, const sd::Tensor<float>& b) {
        if (a.empty() || b.empty() || a.dim() != b.dim()) {
            return {};
        }
        if (a.dim() == 1) {
            if (a.shape() != b.shape()) {
                return {};
            }
            auto batched = sd::ops::concat(a, b, 0);
            batched.reshape_({a.shape()[0], 2});
            return batched;
        }
        const int64_t batch_dim = a.dim() - 1;
        for (int64_t d = 0; d < batch_dim; d++) {
            if (a.shape()[d] != b.shape()[d]) {
                return {};
            }
        }
        return sd::ops::concat(a, b, static_cast<size_t>(batch_dim));
    }

    void DiT::load_alphas_cumprod() {
        file_alphas_cumprod.clear();

        std::vector<float> loaded_alphas;
        if (!model_weights->load_float_tensor("alphas_cumprod", loaded_alphas)) {
            return;
        }
        if (loaded_alphas.size() != TIMESTEPS) {
            LOG_WARN("ignore alphas_cumprod from model file: expected %d values, got %zu",
                     TIMESTEPS,
                     loaded_alphas.size());
            return;
        }
        for (float alpha : loaded_alphas) {
            if (!std::isfinite(alpha) || alpha <= 0.0f || alpha > 1.0f) {
                LOG_WARN("ignore invalid alphas_cumprod from model file");
                return;
            }
        }

        file_alphas_cumprod = std::move(loaded_alphas);
        LOG_VERBOSE("loaded alphas_cumprod from model file");
    }

    bool DiT::set_sage_attention_enabled(bool enabled) {
        if (!diffusion_model) {
            return false;
        }
        if (enabled) {
#ifndef SD_USE_UPSTREAM_GGML
            auto* ctx = ggml_init({4 * ggml_tensor_overhead(), nullptr, true});
            if (ctx == nullptr) {
                return false;
            }
            auto* q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 128, 1, 1);
            auto* k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 128, 1, 1);
            auto* v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 128, 128, 1, 1);
            GGMLRunnerContext probe_ctx;
            probe_ctx.ggml_ctx = ctx;
            auto* op           = ggml_ext_sage_attn(&probe_ctx, q, k, v, 1.f / sqrtf(128.f), GGML_SAGE_ATTN_AUTO);
            bool supported     = true;
            for (auto backend : runtime_.backend_manager.runtime_backends(SDBackendModule::DIFFUSION)) {
                if (!ggml_backend_supports_op(backend, op)) {
                    LOG_ERROR("SageAttention is unavailable on %s; it requires patched GGML, CUDA Toolkit 12.0 or newer, and SM80 or newer kernels",
                              ggml_backend_name(backend));
                    supported = false;
                }
            }
            ggml_free(ctx);
            if (!supported) {
                return false;
            }
#else
            LOG_ERROR("SageAttention requires -DSD_USE_UPSTREAM_GGML=OFF and a CUDA backend");
            return false;
#endif
        }
        diffusion_model->set_sage_attention_enabled(enabled);
        if (high_noise_diffusion_model) {
            high_noise_diffusion_model->set_sage_attention_enabled(enabled);
        }
        if (enabled) {
            LOG_INFO("Using SageAttention in the diffusion model; CUDA selects the supported kernel, unsupported layers use flash/default attention");
        }
        return true;
    }

    std::shared_ptr<const DenoisingParameterization> DiT::make_parameterization(prediction_t prediction) {
        std::lock_guard<std::recursive_mutex> lock(prediction_mutex_);
        if (prediction == PREDICTION_COUNT) {
            if (default_prediction_ == PREDICTION_COUNT) {
                const auto& policy = find_model_definition(version())->sampling;
                auto resolved      = policy.default_prediction;
                if (policy.probe_v_prediction) {
                    Model::ExecutionScope scope(*this);
                    scope.require_ready();
                    resolved = is_using_v_parameterization_for_sd2(sd_version_is_inpaint(version())) ? V_PRED : EPS_PRED;
                } else if (policy.prediction_from_weights) {
                    resolved = policy.prediction_from_weights(model_weights->loader().get_tensor_storage_map());
                }
                default_prediction_ = resolved;
            }
            prediction = default_prediction_;
        }
        auto found = parameterizations_.find(prediction);
        if (found == parameterizations_.end()) {
            auto value = std::make_shared<const DenoisingParameterization>(version(), prediction, model_weights->loader().get_tensor_storage_map(), file_alphas_cumprod);
            found      = parameterizations_.emplace(prediction, std::move(value)).first;
        }
        return found->second;
    }

    bool DiT::is_using_v_parameterization_for_sd2(bool is_inpaint) {
        struct RunnerEndOnExit {
            GGMLRunner* runner = nullptr;
            ~RunnerEndOnExit() {
                if (runner != nullptr) {
                    runner->runner_end();
                }
            }
        };
        RunnerEndOnExit diffusion_runner_end{diffusion_model.get()};

        sd::Tensor<float> x_t   = sd::full<float>({8, 8, 4, 1}, 0.5f);
        sd::Tensor<float> c     = sd::full<float>({1024, 2, 1, 1}, 0.5f);
        sd::Tensor<float> steps = sd::full<float>({1}, 999.0f);
        sd::Tensor<float> concat;
        if (is_inpaint) {
            concat = sd::zeros<float>({8, 8, 5, 1});
        }

        int64_t t0 = ggml_time_ms();
        sd::Tensor<float> out;
        DiffusionParams diffusion_params;
        diffusion_params.x         = &x_t;
        diffusion_params.timesteps = &steps;
        diffusion_params.context   = &c;
        diffusion_params.extra     = UNetDiffusionExtra{};
        if (!concat.empty()) {
            diffusion_params.c_concat = &concat;
        }
        auto out_opt = diffusion_model->compute(n_threads, diffusion_params);
        GGML_ASSERT(!out_opt.empty());
        out = std::move(out_opt);

        double result = static_cast<double>((out - x_t).mean());
        int64_t t1    = ggml_time_ms();
        LOG_VERBOSE("check is_using_v_parameterization_for_sd2, taking %.2fs", (t1 - t0) * 1.0f / 1000);
        return result < -1;
    }

    std::vector<float> DiT::process_timesteps(const std::vector<float>& timesteps,
                                              const sd::Tensor<float>& init_latent,
                                              const sd::Tensor<float>& denoise_mask) {
        if (diffusion_model->get_desc() == "Wan2.2-TI2V-5B") {
            int64_t frame_count = init_latent.shape()[2];
            auto new_timesteps  = std::vector<float>(static_cast<size_t>(frame_count), timesteps[0]);

            if (!denoise_mask.empty() && denoise_mask.dim() >= 4 && denoise_mask.shape()[2] == frame_count) {
                for (int64_t frame = 0; frame < frame_count; ++frame) {
                    float value = denoise_mask.dim() == 5 ? denoise_mask.index(0, 0, frame, 0, 0) : denoise_mask.index(0, 0, frame, 0);
                    if (value == 0.f) {
                        new_timesteps[static_cast<size_t>(frame)] = 0.f;
                    }
                }
            }
            return new_timesteps;
        }
        if (diffusion_model->get_desc() == "Wan2.2-S2V-14B") {
            int64_t frame_count = init_latent.shape()[2];
            return std::vector<float>(static_cast<size_t>(frame_count), timesteps[0]);
        } else {
            return timesteps;
        }
    }

    std::vector<float> DiT::process_ltxav_video_timesteps(const std::vector<float>& timesteps,
                                                          const sd::Tensor<float>& init_latent,
                                                          const sd::Tensor<float>& denoise_mask) {
        if (timesteps.empty() || denoise_mask.empty() || init_latent.dim() < 4 || denoise_mask.dim() < 4) {
            return timesteps;
        }

        int64_t width  = init_latent.shape()[0];
        int64_t height = init_latent.shape()[1];
        int64_t frames = init_latent.shape()[2];
        if (denoise_mask.shape()[0] != width ||
            denoise_mask.shape()[1] != height ||
            denoise_mask.shape()[2] != frames ||
            denoise_mask.shape()[3] < 1) {
            LOG_WARN("unexpected LTXAV denoise mask shape for timestep processing");
            return timesteps;
        }

        std::vector<float> video_timesteps(static_cast<size_t>(width * height * frames));
        size_t idx = 0;
        for (int64_t t = 0; t < frames; ++t) {
            for (int64_t h = 0; h < height; ++h) {
                for (int64_t w = 0; w < width; ++w) {
                    float mask             = denoise_mask.dim() == 5 ? denoise_mask.index(w, h, t, 0, 0)
                                                                     : denoise_mask.index(w, h, t, 0);
                    video_timesteps[idx++] = mask * timesteps[0];
                }
            }
        }
        return video_timesteps;
    }

    int DiT::get_diffusion_model_down_factor() {
        return find_model_definition(version())->geometry.diffusion_down_factor;
    }

    DiffusionPrediction DiT::predict_diffusion(const sd::Tensor<float>& x,
                                               const DenoisingParameterization& parameterization,
                                               float sigma,
                                               const SDCondition& condition,
                                               const std::vector<sd::Tensor<float>>& references,
                                               const DiffusionPredictionOptions& options,
                                               const DiffusionPredictionStep& step) {
        const auto& shape       = x.shape();
        const auto channel_axis = shape.size() == 5 ? 3 : 2;
        const auto down_factor  = get_diffusion_model_down_factor();
        const bool packed       = sd_version_is_ltxav(version()) || sd_version_is_minimax_h3(version());
        const bool unet_batch   = shape.size() == 4 && sd_version_is_unet(version());
        if ((shape.size() != 4 && shape.size() != 5) ||
            (packed ? shape[channel_axis] < get_latent_channel() : shape[channel_axis] != get_latent_channel()) ||
            (!unet_batch && shape.back() != 1) || shape[0] % down_factor || shape[1] % down_factor) {
            throw std::runtime_error("latent shape does not match the diffusion model");
        }
        if (!condition.c_concat.empty()) {
            auto concat_shape = condition.c_concat.shape();
            if (concat_shape.size() == shape.size()) {
                concat_shape[channel_axis] = shape[channel_axis];
            }
            if (concat_shape != shape) {
                throw std::runtime_error("concat conditioning must match the latent outside its channel axis");
            }
        }

        auto scaling                 = parameterization.scalings(sigma);
        const bool has_semantic_time = std::isfinite(step.semantic_sigma);
        if (parameterization.prediction() == SEFI_FLOW_PRED && !has_semantic_time)
            throw std::runtime_error("SeFi predictions require both noise times; use sefi::sample");
        auto base_times = has_semantic_time ? std::vector<float>{sigma * TIMESTEPS} : std::vector<float>{parameterization.timestep(sigma)};
        if (options.transform) {
            const auto transformed = options.transform({sigma, base_times.front(), scaling[2], scaling[0], scaling[1]});
            base_times.front()     = transformed.timestep;
            scaling[2]             = transformed.input_scale;
            scaling[0]             = transformed.skip_scale;
            scaling[1]             = transformed.output_scale;
        }
        const bool separate_audio_time = sd_version_is_ltxav(version()) && !step.mask.empty();
        auto times                     = has_semantic_time
                                             ? std::vector<float>{step.semantic_sigma * TIMESTEPS, base_times.front()}
                                         : separate_audio_time ? process_ltxav_video_timesteps(base_times, step.base, step.mask)
                                                               : process_timesteps(base_times, step.base, step.mask);
        auto input                     = x * scaling[2];
        if (step.mask_input)
            step.blend_mask(input);
        sd::Tensor<float> cached;
        if (step.cache && step.cache->before_condition(&condition, input, &cached)) {
            return {std::move(cached), scaling[0], scaling[1]};
        }
        auto timesteps      = sd::Tensor<float>::from_vector(std::move(times));
        auto audio_time     = separate_audio_time ? sd::Tensor<float>::from_vector(base_times) : sd::Tensor<float>{};
        auto target_time    = std::isfinite(step.target_sigma) ? sd::Tensor<float>::from_vector({step.target_sigma}) : sd::Tensor<float>{};
        auto lookahead_time = std::isfinite(step.lookahead_sigma) ? sd::Tensor<float>::from_vector({step.lookahead_sigma}) : sd::Tensor<float>{};
        auto guidance       = sd::Tensor<float>::from_vector({options.guidance});
        auto present        = [](const auto& tensor) { return tensor.empty() ? nullptr : &tensor; };
        auto first          = [](const auto& tensors) { return tensors.empty() ? nullptr : &tensors.front(); };

        DiffusionParams params;
        params.x                       = &input;
        params.timesteps               = &timesteps;
        params.context                 = present(condition.c_crossattn);
        params.y                       = present(condition.c_vector);
        params.c_concat                = present(condition.c_concat);
        params.ref_latents             = references.empty() ? &condition.c_ref_images : &references;
        params.ref_index_mode          = options.ref_index_mode;
        params.ref_rope_anchor         = options.ref_rope_anchor;
        params.force_ref_timestep_zero = options.force_ref_timestep_zero;
        UNetControls unet_controls;
        std::vector<ZImageControlInput> z_image_controls;
        if (sd_version_is_unet(version())) {
            if (step.compute_controls)
                unet_controls = step.compute_controls(input, timesteps, condition);
            params.extra = UNetDiffusionExtra{-1, &unet_controls.residuals, unet_controls.strength,
                                              nullptr, 1.f, step.attn2_processor, step.unet_block_processor};
        } else if (sd_version_is_sd3(version()) || version() == VERSION_MOSS_IMAGE) {
            params.extra = SkipLayerDiffusionExtra{present(options.skip_layers)};
        } else if (sd_version_is_flux(version()) || sd_version_is_flux2(version()) || sd_version_is_longcat(version()) || sd_version_is_sefi_image(version())) {
            params.extra = FluxDiffusionExtra{&guidance, present(options.skip_layers), nullptr, 1.f, step.flux_image_processor};
        } else if (version() == VERSION_FLUX3) {
            params.extra = Flux3DiffusionExtra{present(target_time)};
        } else if (version() == VERSION_QWEN_IMAGE_2_1) {
            params.extra = QwenImage21DiffusionExtra{&condition.c_token_types, step.qwen_prefix_id};
        } else if (sd_version_is_anima(version())) {
            params.extra = AnimaDiffusionExtra{present(condition.c_t5_ids), present(condition.c_t5_weights)};
        } else if (sd_version_is_wan(version())) {
            params.extra = WanDiffusionExtra{present(options.vace_context), options.vace_strength, present(options.audio_embedding)};
        } else if (sd_version_is_hunyuan_video(version())) {
            params.extra = HunyuanVideoDiffusionExtra{&guidance, first(condition.extra_c_crossattns), params.y, present(lookahead_time)};
        } else if (version() == VERSION_HIDREAM_O1) {
            params.extra = HiDreamO1DiffusionExtra{present(condition.c_input_ids), present(condition.c_position_ids),
                                                   present(condition.c_token_types), present(condition.c_vinput_mask),
                                                   present(condition.c_image_embeds)};
        } else if (version() == VERSION_MING_IMAGE) {
            params.extra = MingImageDiffusionExtra{first(condition.extra_c_crossattns)};
        } else if (sd_version_is_llada_image(version())) {
            params.extra = LLaDAImageDiffusionExtra{first(condition.extra_c_crossattns)};
        } else if (sd_version_is_minimax_h3(version())) {
            params.extra = MiniMaxH3DiffusionExtra{present(condition.c_token_types), present(condition.c_position_ids),
                                                   present(condition.c_ref_audios), present(condition.c_reference_blocks),
                                                   options.audio_length, options.video_sigma_shift, options.audio_sigma_shift, step.target_sigma};
        } else if (sd_version_is_ltxav(version())) {
            params.extra = LTXAVDiffusionExtra{nullptr, present(audio_time), options.audio_length, options.fps, present(options.video_positions), first(condition.extra_c_crossattns)};
        } else if (sd_version_is_minit2i(version())) {
            params.extra = MiniT2IDiffusionExtra{params.y};
        } else if (sd_version_is_sensenova_u1(version())) {
            params.extra = SenseNovaU1DiffusionExtra{present(condition.c_input_ids)};
        } else if (sd_version_is_z_image(version()) && !step.controls.empty()) {
            z_image_controls.reserve(step.controls.size());
            for (const auto* binding : step.controls)
                z_image_controls.push_back(binding->model->z_image_input(*binding, input));
            params.extra = ZImageDiffusionExtra{nullptr, 1.f, &z_image_controls};
        }
        diffusion_model->clear_device_memory_error();
        auto result = diffusion_model->compute(n_threads, params);
        if (result.empty()) {
            const auto& error = diffusion_model->device_memory_error();
            if (!error.empty()) {
                throw Error(SD_STATUS_OUT_OF_MEMORY, error);
            }
            throw std::runtime_error("diffusion compute failed");
        }
        if (result.shape() != shape) {
            throw std::runtime_error("diffusion output shape " + sd::tensor_shape_to_string(result.shape()) +
                                     " does not match input " + sd::tensor_shape_to_string(shape));
        }
        if (step.cache) {
            step.cache->after_condition(&condition, input, result);
        }
        return {std::move(result), scaling[0], scaling[1]};
    }

}  // namespace sd::dgl::models
