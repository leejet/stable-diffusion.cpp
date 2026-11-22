#include "config.h"

#include <cmath>

namespace sd {

    SampleConfig::SampleConfig() {
        guidance.txt_cfg            = 7.0f;
        guidance.img_cfg            = INFINITY;
        guidance.distilled_guidance = 3.5f;
        scheduler                   = SCHEDULER_COUNT;
        sample_method               = SAMPLE_METHOD_COUNT;
        sample_steps                = 20;
        eta                         = INFINITY;
        custom_sigmas               = nullptr;
        custom_sigmas_count         = 0;
        flow_shift                  = INFINITY;
        extra_sample_args           = nullptr;
    }

    CacheConfig::CacheConfig() {
        mode                        = SD_CACHE_DISABLED;
        reuse_threshold             = INFINITY;
        start_percent               = 0.15f;
        end_percent                 = 0.95f;
        error_decay_rate            = 1.0f;
        use_relative_threshold      = true;
        reset_error_on_compute      = true;
        Fn_compute_blocks           = 8;
        Bn_compute_blocks           = 0;
        residual_diff_threshold     = 0.08f;
        max_warmup_steps            = 8;
        max_cached_steps            = -1;
        max_continuous_cached_steps = -1;
        taylorseer_n_derivatives    = 1;
        taylorseer_skip_interval    = 1;
        scm_mask                    = nullptr;
        scm_policy_dynamic          = true;
        spectrum_w                  = 0.40f;
        spectrum_m                  = 3;
        spectrum_lam                = 1.0f;
        spectrum_window_size        = 2;
        spectrum_flex_window        = 0.50f;
        spectrum_warmup_steps       = 4;
        spectrum_stop_percent       = 0.9f;
    }

    ContextConfig::ContextConfig() {
        n_threads                 = sd_get_num_physical_cores();
        conditioning_cache_size   = 4;
        wtype                     = SD_TYPE_COUNT;
        rng_type                  = CUDA_RNG;
        sampler_rng_type          = RNG_TYPE_COUNT;
        prediction                = PREDICTION_COUNT;
        lora_apply_mode           = LORA_APPLY_AUTO;
        max_vram                  = nullptr;
        disable_prefetch          = false;
        disable_segmented_compute = false;
        batched_cfg               = true;
        eager_load                = false;
        enable_mmap               = false;
        flash_attn_mode           = nullptr;
        conv_direct_mode          = nullptr;
        diffusion_flash_attn      = false;
        sage_attn                 = false;
        linear_scale              = 0.f;
        attn_scale                = 0.f;
        vae_format                = SD_VAE_FORMAT_AUTO;
        backend                   = nullptr;
        params_backend            = nullptr;
        split_mode                = nullptr;
        auto_fit                  = true;
        rpc_servers               = nullptr;
        model_args                = nullptr;
    }

    ImageConfig::ImageConfig() {
        clip_skip         = -1;
        ref_images_count  = 0;
        width             = 512;
        height            = 512;
        strength          = 0.75f;
        seed              = -1;
        batch_count       = 1;
        control_strength  = 0.9f;
        qwen_image_layers = 3;
        circular_x        = false;
        circular_y        = false;
        vae_tiling_params = {false, false, 0, 0, 0.5f, 0.0f, 0.0f, nullptr};
    }

    VideoConfig::VideoConfig() {
        width             = 512;
        height            = 512;
        strength          = 0.75f;
        seed              = -1;
        video_frames      = 6;
        fps               = 16;
        moe_boundary      = 0.875f;
        vace_strength     = 1.f;
        vae_tiling_params = {false, false, 0, 0, 0.5f, 0.0f, 0.0f, nullptr};
        circular_x        = false;
        circular_y        = false;
    }

}  // namespace sd
