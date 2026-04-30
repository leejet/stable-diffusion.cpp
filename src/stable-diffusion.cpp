#include "ggml_extend.hpp"

#include "model.h"
#include "rng.hpp"
#include "rng_mt19937.hpp"
#include "rng_philox.hpp"
#include "stable-diffusion.h"
#include "util.h"

#include "auto_encoder_kl.hpp"
#include "backend_fit.hpp"
#include "conditioner.hpp"
#include "control.hpp"
#include "denoiser.hpp"
#include "diffusion_model.hpp"
#include "esrgan.hpp"
#include "lora.hpp"
#include "pmid.hpp"
#include "sample-cache.h"
#include "ltxvae.hpp"
#include "tae.hpp"
#include "vae.hpp"

#include "latent-preview.h"
#include "name_conversion.h"

const char* model_version_to_str[] = {
    "SD 1.x",
    "SD 1.x Inpaint",
    "Instruct-Pix2Pix",
    "SD 1.x Tiny UNet",
    "SD 2.x",
    "SD 2.x Inpaint",
    "SD 2.x Tiny UNet",
    "SDXS (512-DS)",
    "SDXS (09)",
    "SDXL",
    "SDXL Inpaint",
    "SDXL Instruct-Pix2Pix",
    "SDXL (Vega)",
    "SDXL (SSD1B)",
    "SVD",
    "SD3.x",
    "Flux",
    "Flux Fill",
    "Flux Control",
    "Flex.2",
    "Chroma Radiance",
    "Wan 2.x",
    "Wan 2.2 I2V",
    "Wan 2.2 TI2V",
    "Qwen Image",
    "Anima",
    "Flux.2",
    "Flux.2 klein",
    "Z-Image",
    "Ovis Image",
    "Ernie Image",
    "LTX-2",
};

const char* sampling_methods_str[] = {
    "Euler",
    "Euler A",
    "Heun",
    "DPM2",
    "DPM++ (2s)",
    "DPM++ (2M)",
    "modified DPM++ (2M)",
    "iPNDM",
    "iPNDM_v",
    "LCM",
    "DDIM \"trailing\"",
    "TCD",
    "Res Multistep",
    "Res 2s",
    "ER-SDE",
};

/*================================================== Helper Functions ================================================*/

void calculate_alphas_cumprod(float* alphas_cumprod,
                              float linear_start = 0.00085f,
                              float linear_end   = 0.0120f,
                              int timesteps      = TIMESTEPS) {
    float ls_sqrt = sqrtf(linear_start);
    float le_sqrt = sqrtf(linear_end);
    float amount  = le_sqrt - ls_sqrt;
    float product = 1.0f;
    for (int i = 0; i < timesteps; i++) {
        float beta = ls_sqrt + amount * ((float)i / (timesteps - 1));
        product *= 1.0f - powf(beta, 2.0f);
        alphas_cumprod[i] = product;
    }
}

static float get_cache_reuse_threshold(const sd_cache_params_t& params) {
    float reuse_threshold = params.reuse_threshold;
    if (reuse_threshold == INFINITY) {
        if (params.mode == SD_CACHE_EASYCACHE) {
            reuse_threshold = 0.2f;
        } else if (params.mode == SD_CACHE_UCACHE) {
            reuse_threshold = 1.0f;
        }
    }
    return std::max(0.0f, reuse_threshold);
}

/*=============================================== StableDiffusionGGML ================================================*/

class StableDiffusionGGML {
public:
    ggml_backend_t backend             = nullptr;  // general backend
    ggml_backend_t clip_backend        = nullptr;
    ggml_backend_t control_net_backend = nullptr;
    ggml_backend_t vae_backend         = nullptr;
    // Actual device id that `backend` points at. `SD_CUDA_DEVICE` can go stale
    // when auto-fit re-initialises `backend` onto a different GPU. We track the
    // live value so per-component resolution can decide "same device as main"
    // correctly. -1 means the main backend is CPU.
    int            backend_device_id   = -1;
    static constexpr int BACKEND_DEVICE_CPU = -1;

    SDVersion version;
    bool vae_decode_only         = false;
    bool external_vae_is_invalid = false;
    bool free_params_immediately = false;

    bool circular_x = false;
    bool circular_y = false;

    std::shared_ptr<RNG> rng         = std::make_shared<PhiloxRNG>();
    std::shared_ptr<RNG> sampler_rng = nullptr;
    int n_threads                    = -1;
    float default_flow_shift         = INFINITY;

    std::shared_ptr<Conditioner> cond_stage_model;
    std::shared_ptr<FrozenCLIPVisionEmbedder> clip_vision;  // for svd or wan2.1 i2v
    std::shared_ptr<DiffusionModel> diffusion_model;
    std::shared_ptr<DiffusionModel> high_noise_diffusion_model;
    std::shared_ptr<VAE> first_stage_model;
    std::shared_ptr<VAE> preview_vae;
    std::shared_ptr<ControlNet> control_net;
    std::shared_ptr<PhotoMakerIDEncoder> pmid_model;
    std::shared_ptr<LoraModel> pmid_lora;
    std::shared_ptr<PhotoMakerIDEmbed> pmid_id_embeds;
    std::vector<std::shared_ptr<LoraModel>> cond_stage_lora_models;
    std::vector<std::shared_ptr<LoraModel>> diffusion_lora_models;
    std::vector<std::shared_ptr<LoraModel>> first_stage_lora_models;
    bool apply_lora_immediately = false;

    std::string taesd_path;
    sd_tiling_params_t vae_tiling_params = {false, 0, 0, 0.5f, 0, 0};
    bool offload_params_to_cpu           = false;
    bool use_pmid                        = false;

    bool is_using_v_parameterization     = false;
    bool is_using_edm_v_parameterization = false;

    // Populated by auto-fit (when --auto-fit is passed). When enabled, this
    // overrides env-var based per-component placement. device_id == -1 means
    // "no override" (fall through to env vars / defaults).
    struct FitOverride {
        bool enabled             = false;
        int  dit_device_id       = -1;   // -1 = keep main backend
        int  vae_device_id       = -2;   // -2 = no override (distinguishes from "force CPU")
        int  cond_device_id      = -2;
        bool dit_offload_params  = false;  // force offload_params_to_cpu for DiT only
        bool cond_offload_params = false;  // force offload_params_to_cpu for Conditioner only
        bool vae_offload_params  = false;
        bool vae_on_cpu          = false;
        bool cond_on_cpu         = false;
    };
    FitOverride fit_override;

    // Auto-fit VAE tiling: when auto_fit is enabled, generate_video /
    // generate_image consult this budget at gen time to decide whether the
    // current resolution needs VAE tiling (and at what tile size). Captured
    // from sd_ctx_params at init so it survives past sd_ctx_params_t scope.
    bool    auto_fit_enabled                = false;
    int64_t auto_fit_vae_compute_reserve_bytes = 0;
    bool    auto_fit_vae_on_cpu             = false;

    // Pending tensor-split decisions from auto-fit (consumed by init_tensor_split).
    // When true, the corresponding component will be configured for split mode
    // even when SD_CUDA_TENSOR_SPLIT_DIT / _COND env vars are unset.
    bool    pending_split_dit  = false;
    bool    pending_split_cond = false;

    // Heuristic: VAE peak compute scales with the largest activation tensor,
    // which is roughly proportional to the number of latent voxels times a
    // per-voxel byte cost dominated by the deepest decoder block (1024-channel
    // up_block in LTX-2). 1 MiB/voxel is an empirical fit that lets a typical
    // 480x320x25 LTX-2 decode (15*10*4=600 voxels) decode in one pass under
    // the default 1024 MiB reserve while triggering tiling at HD+ (e.g.
    // 1280x720x49 → 40*22*7=6160 voxels, tile side ≈ 12).
    static constexpr int64_t LATENT_VOXEL_PEAK_BYTES = 1 * backend_fit::MiB;

    // Compute auto-tiling parameters for the VAE based on the captured
    // auto-fit budget and the latent grid (lw × lh × t_latent). Modifies
    // `vae_tiling_params` in place ONLY when:
    //   - auto-fit is enabled
    //   - the user did NOT explicitly enable tiling (we don't override)
    //   - the predicted peak exceeds the reserved budget
    // Logs the chosen tile size; no-op otherwise.
    void maybe_auto_set_vae_tiling(int64_t lw, int64_t lh, int64_t t_latent) {
        if (!auto_fit_enabled) return;
        if (vae_tiling_params.enabled) return;  // user override: don't touch
        if (auto_fit_vae_on_cpu) return;        // CPU VAE: no VRAM budget
        if (auto_fit_vae_compute_reserve_bytes <= 0) return;
        if (lw <= 0 || lh <= 0 || t_latent <= 0) return;

        const int64_t total_voxels = lw * lh * t_latent;
        const int64_t budget_voxels =
            auto_fit_vae_compute_reserve_bytes / LATENT_VOXEL_PEAK_BYTES;
        if (total_voxels <= budget_voxels) {
            // Fits in a single decode pass — leave tiling disabled.
            return;
        }

        // Time stays untiled (the LTX-2 VAE has temporal coupling); spread
        // the budget across spatial tiles. Pick a square tile_w × tile_h.
        const int64_t voxels_per_tile = std::max<int64_t>(budget_voxels, 1);
        const int64_t tile_area_max   = std::max<int64_t>(voxels_per_tile / t_latent, 16);
        int           tile_side       = static_cast<int>(std::round(std::sqrt(double(tile_area_max))));
        // Clamp to [8, max(lw, lh)] so the VAE tiler doesn't get a degenerate
        // tile size, and never go above the actual latent dim.
        tile_side = std::max(8, tile_side);
        int tile_x = std::min<int>(tile_side, static_cast<int>(lw));
        int tile_y = std::min<int>(tile_side, static_cast<int>(lh));

        vae_tiling_params.enabled        = true;
        vae_tiling_params.tile_size_x    = tile_x;
        vae_tiling_params.tile_size_y    = tile_y;
        vae_tiling_params.target_overlap = 0.5f;
        vae_tiling_params.rel_size_x     = 0.f;
        vae_tiling_params.rel_size_y     = 0.f;

        LOG_INFO("auto-fit: VAE tiling enabled (latent %lldx%lldx%lld = %lld voxels > "
                 "budget %lld voxels @ %lld MiB reserve); tile %dx%d (latent), overlap %.2f",
                 (long long)lw, (long long)lh, (long long)t_latent, (long long)total_voxels,
                 (long long)budget_voxels,
                 (long long)(auto_fit_vae_compute_reserve_bytes / backend_fit::MiB),
                 tile_x, tile_y, vae_tiling_params.target_overlap);
    }

    std::map<std::string, ggml_tensor*> tensors;

    // lora_name => multiplier
    std::unordered_map<std::string, float> curr_lora_state;

    std::shared_ptr<Denoiser> denoiser = std::make_shared<CompVisDenoiser>();

    // ModelLoader kept alive for the lifetime of the SD context. Lazy-load
    // callbacks (registered on DiT / LLM runners) call back into this loader
    // on the first compute() of each component to read weights from disk
    // sequentially — keeps peak VRAM at max-across-phases instead of
    // sum-of-components. Toggled per-component via SD_LAZY_LOAD_DIT /
    // SD_LAZY_LOAD_COND env vars.
    std::unique_ptr<ModelLoader> model_loader_;
    std::set<std::string>        load_ignore_tensors;
    bool                         lazy_load_dit  = false;
    bool                         lazy_load_cond = false;

    // Multi-GPU tensor split state. Populated when SD_CUDA_TENSOR_SPLIT is set.
    // The extra GPU backends and the CPU fallback are owned here so they live
    // as long as any GGMLRunner that references them via MultiBackendSpec.
    struct TensorSplitState {
        bool                        enabled = false;
        std::vector<float>          ratios;             // per-device row-split ratios
        int                         main_device = 0;
        std::vector<ggml_backend_t> extra_backends;     // additional GPU backends (excluding main)
        ggml_backend_t              cpu_fallback = nullptr;
        bool                        split_dit    = false;  // apply to LTX-2 DiT
        bool                        split_cond   = false;  // apply to conditioner (LLM/Gemma)
    };
    TensorSplitState tensor_split_state;

    StableDiffusionGGML() = default;

    ~StableDiffusionGGML() {
        if (clip_backend != backend) {
            ggml_backend_free(clip_backend);
        }
        if (control_net_backend != backend) {
            ggml_backend_free(control_net_backend);
        }
        if (vae_backend != backend) {
            ggml_backend_free(vae_backend);
        }
        // Tensor-split extra GPUs + CPU fallback: free after the runners (they
        // refer to these via MultiBackendSpec). Order: extras first, then CPU
        // fallback, then main `backend`.
        for (auto* b : tensor_split_state.extra_backends) {
            if (b != nullptr && b != backend) {
                ggml_backend_free(b);
            }
        }
        tensor_split_state.extra_backends.clear();
        if (tensor_split_state.cpu_fallback != nullptr) {
            ggml_backend_free(tensor_split_state.cpu_fallback);
            tensor_split_state.cpu_fallback = nullptr;
        }
        ggml_backend_free(backend);
    }

    // Read an integer environment variable, returning `def` if unset or malformed.
    static int get_env_int(const char* name, int def) {
        const char* v = getenv(name);
        if (v == nullptr || *v == '\0') return def;
        try {
            return std::stoi(v);
        } catch (...) {
            LOG_WARN("env %s: '%s' is not a valid integer, using default %d", name, v, def);
            return def;
        }
    }

    // Initialize a GPU backend for the given device id, or fall back to CPU.
    // For CUDA, device_id < 0 means "CPU only"; otherwise clamp to available count.
    // `component_name` is used for log messages (e.g. "DiT", "Gemma", "VAE").
    static ggml_backend_t init_device_backend(int device_id, const char* component_name) {
        if (device_id < 0) {
            LOG_INFO("%s: using CPU backend (device=-1)", component_name);
            return ggml_backend_cpu_init();
        }
#ifdef SD_USE_CUDA
        int count = ggml_backend_cuda_get_device_count();
        if (count <= 0) {
            LOG_WARN("%s: no CUDA devices available, falling back to CPU", component_name);
            return ggml_backend_cpu_init();
        }
        if (device_id >= count) {
            LOG_WARN("%s: CUDA device %d requested but only %d available, falling back to device 0",
                     component_name, device_id, count);
            device_id = 0;
        }
        auto b = ggml_backend_cuda_init(device_id);
        if (b != nullptr) {
            LOG_INFO("%s: using CUDA device %d", component_name, device_id);
            return b;
        }
        LOG_WARN("%s: CUDA device %d init failed, falling back to CPU", component_name, device_id);
        return ggml_backend_cpu_init();
#elif defined(SD_USE_VULKAN)
        int count = ggml_backend_vk_get_device_count();
        if (count <= 0) {
            LOG_WARN("%s: no Vulkan devices available, falling back to CPU", component_name);
            return ggml_backend_cpu_init();
        }
        if (device_id >= count) {
            LOG_WARN("%s: Vulkan device %d requested but only %d available, falling back to device 0",
                     component_name, device_id, count);
            device_id = 0;
        }
        auto b = ggml_backend_vk_init((size_t)device_id);
        if (b != nullptr) {
            LOG_INFO("%s: using Vulkan device %d", component_name, device_id);
            return b;
        }
        LOG_WARN("%s: Vulkan device %d init failed, falling back to CPU", component_name, device_id);
        return ggml_backend_cpu_init();
#elif defined(SD_USE_SYCL)
        auto b = ggml_backend_sycl_init(device_id);
        if (b != nullptr) {
            LOG_INFO("%s: using SYCL device %d", component_name, device_id);
            return b;
        }
        LOG_WARN("%s: SYCL init failed, falling back to CPU", component_name);
        return ggml_backend_cpu_init();
#else
        (void)device_id;
        LOG_INFO("%s: using CPU backend", component_name);
        return ggml_backend_cpu_init();
#endif
    }

    // Main backend init. Honours these env vars for per-component device placement
    // (used by the init path below):
    //   SD_CUDA_DEVICE          default CUDA device id (default 0) — also used for DiT
    //   SD_CUDA_DEVICE_CLIP     text encoder / conditioner (falls back to SD_CUDA_DEVICE)
    //   SD_CUDA_DEVICE_VAE      VAE                          (falls back to SD_CUDA_DEVICE)
    //   SD_CUDA_DEVICE_CONTROL  ControlNet                    (falls back to SD_CUDA_DEVICE)
    //   SD_VK_DEVICE            same pattern for the Vulkan build
    // Setting any of these to -1 forces CPU for that component.
    //
    // `keep_clip_on_cpu` / `keep_vae_on_cpu` still take precedence and force CPU.
    // For weights too big even for a dedicated device, use offload_params_to_cpu
    // (keeps weights on CPU and streams per-step to GPU).
    void init_backend() {
#ifdef SD_USE_CUDA
        int main_dev       = get_env_int("SD_CUDA_DEVICE", 0);
        backend            = init_device_backend(main_dev, "main");
        backend_device_id  = ggml_backend_is_cpu(backend) ? BACKEND_DEVICE_CPU : main_dev;
#endif
#ifdef SD_USE_METAL
        LOG_DEBUG("Using Metal backend");
        backend = ggml_backend_metal_init();
#endif
#ifdef SD_USE_VULKAN
        LOG_DEBUG("Using Vulkan backend");
        size_t device          = 0;
        const int device_count = ggml_backend_vk_get_device_count();
        if (device_count) {
            const char* SD_VK_DEVICE = getenv("SD_VK_DEVICE");
            if (SD_VK_DEVICE != nullptr) {
                std::string sd_vk_device_str = SD_VK_DEVICE;
                try {
                    device = std::stoull(sd_vk_device_str);
                } catch (const std::invalid_argument&) {
                    LOG_WARN("SD_VK_DEVICE environment variable is not a valid integer (%s). Falling back to device 0.", SD_VK_DEVICE);
                    device = 0;
                } catch (const std::out_of_range&) {
                    LOG_WARN("SD_VK_DEVICE environment variable value is out of range for `unsigned long long` type (%s). Falling back to device 0.", SD_VK_DEVICE);
                    device = 0;
                }
                if (device >= device_count) {
                    LOG_WARN("Cannot find targeted vulkan device (%llu). Falling back to device 0.", device);
                    device = 0;
                }
            }
            LOG_INFO("Vulkan: Using device %llu", device);
            backend = ggml_backend_vk_init(device);
        }
        if (!backend) {
            LOG_WARN("Failed to initialize Vulkan backend");
        }
#endif
#ifdef SD_USE_OPENCL
        LOG_DEBUG("Using OpenCL backend");
        // ggml_log_set(ggml_log_callback_default, nullptr); // Optional ggml logs
        backend = ggml_backend_opencl_init();
        if (!backend) {
            LOG_WARN("Failed to initialize OpenCL backend");
        }
#endif
#ifdef SD_USE_SYCL
        LOG_DEBUG("Using SYCL backend");
        backend = ggml_backend_sycl_init(0);
#endif

        if (!backend) {
            LOG_DEBUG("Using CPU backend");
            backend = ggml_backend_cpu_init();
        }
    }

    // Resolve the backend for a sub-component by reading its env override (if set),
    // otherwise reusing the main backend. Returns the main `backend` unchanged if
    // the override matches the main device; otherwise creates a new backend (which
    // the caller is responsible for freeing via the existing `!= backend` dtor check).
    // `force_cpu` short-circuits to CPU regardless of the env var.
    // `fit_device_id` is the auto-fit override: -2 means "no override", -1 means
    // "force CPU", >=0 names a specific GPU.
    ggml_backend_t resolve_component_backend(const char* env_name,
                                              const char* component_name,
                                              bool        force_cpu,
                                              int         fit_device_id = -2) {
        if (force_cpu) {
            if (ggml_backend_is_cpu(backend)) {
                return backend;
            }
            LOG_INFO("%s: forced CPU backend", component_name);
            return ggml_backend_cpu_init();
        }
#if defined(SD_USE_CUDA) || defined(SD_USE_VULKAN) || defined(SD_USE_SYCL)
        // Reuse the main backend iff this component resolves to the same
        // physical device. After auto-fit re-initialises the main backend
        // onto a different GPU, `SD_CUDA_DEVICE` no longer reflects reality,
        // so we compare against `backend_device_id` instead.
        int override_dev;
        if (fit_override.enabled && fit_device_id != -2) {
            override_dev = fit_device_id;
        } else {
            override_dev = get_env_int(env_name, get_env_int("SD_CUDA_DEVICE", 0));
        }
        if (override_dev == backend_device_id && !ggml_backend_is_cpu(backend)) {
            return backend;
        }
        return init_device_backend(override_dev, component_name);
#else
        (void)env_name;
        (void)component_name;
        (void)fit_device_id;
        return backend;
#endif
    }

    // Configure tensor split for multi-GPU systems.
    //
    // Source of ratios (in priority order):
    //   1. SD_CUDA_TENSOR_SPLIT="W0,W1,..." — explicit user override.
    //   2. sd_ctx_params->auto_tensor_split (default true): when >1 CUDA
    //      device is detected and the env is unset, compute ratios from the
    //      free VRAM of each device and apply them to the DiT.
    //
    // SD_CUDA_TENSOR_SPLIT_DIT=1 enables split for the LTX-2 DiT.
    // SD_CUDA_TENSOR_SPLIT_COND=1 enables split for the conditioner (Gemma).
    // Both default off when only the ratios env is set, so the user can dial
    // in just one component.
    void init_tensor_split(bool auto_tensor_split) {
#ifdef SD_USE_CUDA
        if (tensor_split_state.enabled) return;
        const int dev_count = ggml_backend_cuda_get_device_count();

        std::vector<float> ratios;
        bool from_env = false;
        if (const char* split_env = getenv("SD_CUDA_TENSOR_SPLIT");
            split_env != nullptr && *split_env != '\0') {
            from_env = true;
            std::string s(split_env);
            size_t i = 0;
            while (i < s.size()) {
                size_t j = s.find(',', i);
                std::string tok = s.substr(i, (j == std::string::npos) ? std::string::npos : j - i);
                try {
                    float v = std::stof(tok);
                    ratios.push_back(v);
                } catch (...) {
                    LOG_WARN("SD_CUDA_TENSOR_SPLIT: bad token '%s' — disabling tensor split",
                             tok.c_str());
                    return;
                }
                if (j == std::string::npos) break;
                i = j + 1;
            }
        } else if (auto_tensor_split && dev_count >= 2) {
            // Auto-derive: ratios proportional to free VRAM, one per device.
            ratios.reserve(dev_count);
            for (int d = 0; d < dev_count; d++) {
                size_t free_b = 0, total_b = 0;
                ggml_backend_cuda_get_device_memory(d, &free_b, &total_b);
                ratios.push_back(static_cast<float>(free_b) / float(backend_fit::MiB));
            }
            LOG_INFO("auto tensor split: deriving DiT ratios from free VRAM "
                     "across %d CUDA device(s)",
                     dev_count);
        } else {
            return;  // single GPU, env unset, auto disabled, or no CUDA
        }

        if (dev_count < 2) {
            if (from_env) {
                LOG_WARN("SD_CUDA_TENSOR_SPLIT set but only %d CUDA device(s) available; ignoring",
                         dev_count);
            }
            return;
        }
        if ((int)ratios.size() > dev_count) {
            LOG_WARN("SD_CUDA_TENSOR_SPLIT has %zu ratios but only %d CUDA devices; truncating",
                     ratios.size(), dev_count);
            ratios.resize(dev_count);
        }
        // Pad with zeros so downstream code can index by device id without bounds checks.
        while ((int)ratios.size() < dev_count) ratios.push_back(0.f);

        int main_dev = backend_device_id;
        if (main_dev < 0) {
            LOG_WARN("SD_CUDA_TENSOR_SPLIT: main backend is not CUDA; ignoring tensor split");
            return;
        }
        // Determine which non-main devices have non-zero ratio — those need
        // their own ggml_backend_cuda instance for sched routing.
        std::vector<int> participating_devices;
        for (int d = 0; d < dev_count; d++) {
            if (ratios[d] > 0.f && d != main_dev) {
                participating_devices.push_back(d);
            }
        }
        if (participating_devices.empty()) {
            LOG_WARN("SD_CUDA_TENSOR_SPLIT: no non-main device has nonzero ratio; ignoring");
            return;
        }

        tensor_split_state.enabled     = true;
        tensor_split_state.ratios      = std::move(ratios);
        tensor_split_state.main_device = main_dev;

        for (int d : participating_devices) {
            ggml_backend_t b = ggml_backend_cuda_init(d);
            if (b == nullptr) {
                LOG_WARN("SD_CUDA_TENSOR_SPLIT: failed to init CUDA device %d; skipping", d);
                continue;
            }
            tensor_split_state.extra_backends.push_back(b);
        }
        if (tensor_split_state.extra_backends.empty()) {
            LOG_WARN("SD_CUDA_TENSOR_SPLIT: no extra CUDA backends could be initialised; disabling");
            tensor_split_state.enabled = false;
            return;
        }

        tensor_split_state.cpu_fallback = ggml_backend_cpu_init();
        // Env vars take priority; otherwise honour what auto-fit decided
        // (pending_split_dit / pending_split_cond). If neither: default to
        // DiT-only split for the multi-GPU case.
        tensor_split_state.split_dit  = get_env_int("SD_CUDA_TENSOR_SPLIT_DIT",  0) != 0
                                         || pending_split_dit;
        tensor_split_state.split_cond = get_env_int("SD_CUDA_TENSOR_SPLIT_COND", 0) != 0
                                         || pending_split_cond;
        if (!tensor_split_state.split_dit && !tensor_split_state.split_cond) {
            tensor_split_state.split_dit = true;
            LOG_INFO("SD_CUDA_TENSOR_SPLIT: neither SD_CUDA_TENSOR_SPLIT_DIT nor _COND set; "
                     "defaulting to DiT-only split");
        }

        std::string ratio_str;
        for (size_t k = 0; k < tensor_split_state.ratios.size(); k++) {
            if (k > 0) ratio_str += ",";
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%.2f", tensor_split_state.ratios[k]);
            ratio_str += buf;
        }
        LOG_INFO("tensor split enabled: ratios=[%s] main_dev=%d extras=%zu DiT=%s Cond=%s",
                 ratio_str.c_str(), main_dev, tensor_split_state.extra_backends.size(),
                 tensor_split_state.split_dit ? "yes" : "no",
                 tensor_split_state.split_cond ? "yes" : "no");
#endif
    }

    // Populate g_pending_multi_backend_spec() with this runner's split config
    // if `apply` is true. The pending spec is consumed (cleared) by the next
    // GGMLRunner ctor. Returns a guard struct that clears the pending spec on
    // destruction (in case construction throws and the runner ctor doesn't
    // run).
    struct PendingSplitGuard {
        MultiBackendSpec spec;
        bool             active = false;
        ~PendingSplitGuard() {
            if (active && g_pending_multi_backend_spec() == &spec) {
                g_pending_multi_backend_spec() = nullptr;
            }
        }
    };
    std::unique_ptr<PendingSplitGuard> begin_pending_split(bool apply) {
        if (!apply || !tensor_split_state.enabled) return nullptr;
        auto g = std::make_unique<PendingSplitGuard>();
        g->spec.extra_backends = tensor_split_state.extra_backends;
        g->spec.tensor_split   = tensor_split_state.ratios;
        g->spec.main_device    = tensor_split_state.main_device;
        g->spec.cpu_fallback   = tensor_split_state.cpu_fallback;
        g->active              = true;
        g_pending_multi_backend_spec() = &g->spec;
        return g;
    }

    std::shared_ptr<RNG> get_rng(rng_type_t rng_type) {
        if (rng_type == STD_DEFAULT_RNG) {
            return std::make_shared<STDDefaultRNG>();
        } else if (rng_type == CPU_RNG) {
            return std::make_shared<MT19937RNG>();
        } else {  // default: CUDA_RNG
            return std::make_shared<PhiloxRNG>();
        }
    }

    bool init(const sd_ctx_params_t* sd_ctx_params) {
        n_threads               = sd_ctx_params->n_threads;
        vae_decode_only         = sd_ctx_params->vae_decode_only;
        free_params_immediately = sd_ctx_params->free_params_immediately;
        offload_params_to_cpu   = sd_ctx_params->offload_params_to_cpu;

        bool use_tae = false;

        rng = get_rng(sd_ctx_params->rng_type);
        if (sd_ctx_params->sampler_rng_type != RNG_TYPE_COUNT && sd_ctx_params->sampler_rng_type != sd_ctx_params->rng_type) {
            sampler_rng = get_rng(sd_ctx_params->sampler_rng_type);
        } else {
            sampler_rng = rng;
        }

        ggml_log_set(ggml_log_callback_default, nullptr);

        init_backend();
        // tensor split is initialised AFTER auto-fit, since auto-fit may decide
        // to place a component in tensor-split mode.

        // Lazy load — let DiT and Conditioner-LLM lazy-load weights on first
        // compute() instead of all-at-once at init. Required when sum-of-
        // components exceeds combined VRAM (e.g. Q6_K LTX-2 + Q8_K_XL Gemma
        // + connector + VAE on a 24 GB combined system). Defaults to ON via
        // sd_ctx_params; env vars (SD_LAZY_LOAD_DIT / SD_LAZY_LOAD_COND) act
        // as force-on overrides (they do not disable).
        lazy_load_dit  = sd_ctx_params->lazy_load_dit  || get_env_int("SD_LAZY_LOAD_DIT",  0) != 0;
        lazy_load_cond = sd_ctx_params->lazy_load_cond || get_env_int("SD_LAZY_LOAD_COND", 0) != 0;
        if (lazy_load_dit)  LOG_INFO("lazy load: DiT  (alloc + read on first compute, free after free_params_immediately)");
        if (lazy_load_cond) LOG_INFO("lazy load: LLM  (Gemma allocs on first encode; connector stays eager)");

        model_loader_ = std::make_unique<ModelLoader>();
        ModelLoader& model_loader = *model_loader_;

        if (strlen(SAFE_STR(sd_ctx_params->model_path)) > 0) {
            LOG_INFO("loading model from '%s'", sd_ctx_params->model_path);
            if (!model_loader.init_from_file(sd_ctx_params->model_path)) {
                LOG_ERROR("init model loader from file failed: '%s'", sd_ctx_params->model_path);
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->diffusion_model_path)) > 0) {
            LOG_INFO("loading diffusion model from '%s'", sd_ctx_params->diffusion_model_path);
            if (!model_loader.init_from_file(sd_ctx_params->diffusion_model_path, "model.diffusion_model.")) {
                LOG_WARN("loading diffusion model from '%s' failed", sd_ctx_params->diffusion_model_path);
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->high_noise_diffusion_model_path)) > 0) {
            LOG_INFO("loading high noise diffusion model from '%s'", sd_ctx_params->high_noise_diffusion_model_path);
            if (!model_loader.init_from_file(sd_ctx_params->high_noise_diffusion_model_path, "model.high_noise_diffusion_model.")) {
                LOG_WARN("loading diffusion model from '%s' failed", sd_ctx_params->high_noise_diffusion_model_path);
            }
        }

        bool is_unet = sd_version_is_unet(model_loader.get_sd_version());

        if (strlen(SAFE_STR(sd_ctx_params->clip_l_path)) > 0) {
            LOG_INFO("loading clip_l from '%s'", sd_ctx_params->clip_l_path);
            std::string prefix = is_unet ? "cond_stage_model.transformer." : "text_encoders.clip_l.transformer.";
            if (!model_loader.init_from_file(sd_ctx_params->clip_l_path, prefix)) {
                LOG_WARN("loading clip_l from '%s' failed", sd_ctx_params->clip_l_path);
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->clip_g_path)) > 0) {
            LOG_INFO("loading clip_g from '%s'", sd_ctx_params->clip_g_path);
            std::string prefix = is_unet ? "cond_stage_model.1.transformer." : "text_encoders.clip_g.transformer.";
            if (!model_loader.init_from_file(sd_ctx_params->clip_g_path, prefix)) {
                LOG_WARN("loading clip_g from '%s' failed", sd_ctx_params->clip_g_path);
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->clip_vision_path)) > 0) {
            LOG_INFO("loading clip_vision from '%s'", sd_ctx_params->clip_vision_path);
            std::string prefix = "cond_stage_model.transformer.";
            if (!model_loader.init_from_file(sd_ctx_params->clip_vision_path, prefix)) {
                LOG_WARN("loading clip_vision from '%s' failed", sd_ctx_params->clip_vision_path);
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->t5xxl_path)) > 0) {
            LOG_INFO("loading t5xxl from '%s'", sd_ctx_params->t5xxl_path);
            if (!model_loader.init_from_file(sd_ctx_params->t5xxl_path, "text_encoders.t5xxl.transformer.")) {
                LOG_WARN("loading t5xxl from '%s' failed", sd_ctx_params->t5xxl_path);
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->llm_path)) > 0) {
            LOG_INFO("loading llm from '%s'", sd_ctx_params->llm_path);
            if (!model_loader.init_from_file(sd_ctx_params->llm_path, "text_encoders.llm.")) {
                LOG_WARN("loading llm from '%s' failed", sd_ctx_params->llm_path);
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->llm_vision_path)) > 0) {
            LOG_INFO("loading llm vision from '%s'", sd_ctx_params->llm_vision_path);
            if (!model_loader.init_from_file(sd_ctx_params->llm_vision_path, "text_encoders.llm.visual.")) {
                LOG_WARN("loading llm vision from '%s' failed", sd_ctx_params->llm_vision_path);
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->vae_path)) > 0) {
            LOG_INFO("loading vae from '%s'", sd_ctx_params->vae_path);
            if (!model_loader.init_from_file(sd_ctx_params->vae_path, "vae.")) {
                LOG_WARN("loading vae from '%s' failed", sd_ctx_params->vae_path);
                external_vae_is_invalid = true;
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->taesd_path)) > 0) {
            LOG_INFO("loading tae from '%s'", sd_ctx_params->taesd_path);
            if (!model_loader.init_from_file(sd_ctx_params->taesd_path, "tae.")) {
                LOG_WARN("loading tae from '%s' failed", sd_ctx_params->taesd_path);
            }
            use_tae = true;
        }

        model_loader.convert_tensors_name();

        version = model_loader.get_sd_version();
        if (version == VERSION_COUNT) {
            LOG_ERROR("get sd version from file failed: '%s'", SAFE_STR(sd_ctx_params->model_path));
            return false;
        }

        auto& tensor_storage_map = model_loader.get_tensor_storage_map();

        // LTX-2 prefix + Gemma sandwich-norm fixup: the conditioner expects Gemma at
        // `text_encoder.model.*`, but `--llm-path` prepends `text_encoders.llm.*`
        // (convert_tensors_name then maps gguf llama names to HF names, yielding
        // `text_encoders.llm.model.*`).
        //
        // Additionally, Gemma 3 has 4 layernorms per block (sandwich norms) that the
        // shared llm_name_map only partly translates. The raw GGUF names blk.N.{attn_norm,
        // post_attention_norm, ffn_norm, post_ffw_norm} end up as HF-style
        // input_layernorm + post_attention_norm + post_attention_layernorm + post_ffw_norm
        // after the generic map (where ffn_norm→post_attention_layernorm is Qwen-correct
        // but wrong for Gemma). We rename here once version is LTX-2:
        //   post_attention_layernorm → pre_feedforward_layernorm  (was actually ffn_norm)
        //   post_attention_norm      → post_attention_layernorm   (append _layernorm)
        //   post_ffw_norm            → post_feedforward_layernorm
        // Order matters: do the first rename first so the second can safely write to
        // the now-vacated post_attention_layernorm slot.
        if (sd_version_is_ltx2(version)) {
            // Step 1: prefix rewrite text_encoders.llm. → text_encoder.
            const std::string from = "text_encoders.llm.";
            const std::string to   = "text_encoder.";
            {
                String2TensorStorage renamed;
                size_t renames = 0;
                for (auto& kv : tensor_storage_map) {
                    const std::string& k = kv.first;
                    std::string new_k    = k;
                    if (k.rfind(from, 0) == 0) {
                        new_k = to + k.substr(from.size());
                        kv.second.name = new_k;
                        renames++;
                    }
                    renamed[new_k] = std::move(kv.second);
                }
                if (renames > 0) {
                    tensor_storage_map.swap(renamed);
                    LOG_INFO("LTX-2: renamed %zu '%s*' tensors → '%s*' (Gemma text encoder path)",
                             renames, from.c_str(), to.c_str());
                }
            }

            // Step 2: Gemma 3 sandwich-norm renames, applied in the order documented
            // above. Each pass rebuilds the storage map because std::map keys are const.
            auto rename_suffix = [&](const std::string& old_suffix, const std::string& new_suffix) -> size_t {
                String2TensorStorage renamed;
                size_t renames = 0;
                for (auto& kv : tensor_storage_map) {
                    const std::string& k = kv.first;
                    std::string new_k    = k;
                    size_t p = k.rfind(old_suffix);
                    if (p != std::string::npos && p + old_suffix.size() == k.size()) {
                        // Only rename if prefix looks like a Gemma layer key.
                        if (k.find("text_encoder.model.layers.") != std::string::npos) {
                            new_k = k.substr(0, p) + new_suffix;
                            kv.second.name = new_k;
                            renames++;
                        }
                    }
                    renamed[new_k] = std::move(kv.second);
                }
                tensor_storage_map.swap(renamed);
                return renames;
            };
            size_t r1 = rename_suffix(".post_attention_layernorm.weight", ".pre_feedforward_layernorm.weight");
            size_t r2 = rename_suffix(".post_attention_norm.weight",      ".post_attention_layernorm.weight");
            size_t r3 = rename_suffix(".post_ffw_norm.weight",            ".post_feedforward_layernorm.weight");
            if (r1 + r2 + r3 > 0) {
                LOG_INFO("LTX-2: Gemma sandwich-norm rename: %zu pre_ff, %zu post_attn, %zu post_ff",
                         r1, r2, r3);
            }

            // Step 3: Duplicate `first_stage_model.per_channel_statistics.*` into the
            // `first_stage_model.encoder.per_channel_statistics.*` path expected by
            // VideoEncoder's child block tree. VideoDecoder also expects these under
            // its `decoder.per_channel_statistics` subprefix. Real LTX-2 checkpoints
            // only ship the top-level buffer (mean-of-means, std-of-means).
            {
                const std::string top_pre = "first_stage_model.per_channel_statistics.";
                size_t copied = 0;
                // Snapshot keys with top_pre first (iteration + insertion is unsafe).
                std::vector<std::pair<std::string, std::string>> to_copy;
                for (auto& kv : tensor_storage_map) {
                    const std::string& k = kv.first;
                    if (k.rfind(top_pre, 0) == 0) {
                        std::string suffix = k.substr(top_pre.size());
                        to_copy.push_back({k, suffix});
                    }
                }
                for (auto& pair : to_copy) {
                    const std::string& src_key = pair.first;
                    const std::string& suffix  = pair.second;
                    auto src_it = tensor_storage_map.find(src_key);
                    if (src_it == tensor_storage_map.end()) continue;
                    for (const char* sub : {"encoder", "decoder"}) {
                        std::string dst_key = "first_stage_model." + std::string(sub) +
                                              ".per_channel_statistics." + suffix;
                        if (tensor_storage_map.find(dst_key) != tensor_storage_map.end()) continue;
                        TensorStorage dup = src_it->second;
                        dup.name          = dst_key;
                        tensor_storage_map[dst_key] = dup;
                        copied++;
                    }
                }
                if (copied > 0) {
                    LOG_INFO("LTX-2: duplicated %zu PerChannelStatistics entries to encoder/decoder subprefixes",
                             copied);
                }
            }
        }

        LOG_INFO("Version: %s ", model_version_to_str[version]);
        ggml_type wtype               = (int)sd_ctx_params->wtype < std::min<int>(SD_TYPE_COUNT, GGML_TYPE_COUNT)
                                            ? (ggml_type)sd_ctx_params->wtype
                                            : GGML_TYPE_COUNT;
        std::string tensor_type_rules = SAFE_STR(sd_ctx_params->tensor_type_rules);
        if (wtype != GGML_TYPE_COUNT || tensor_type_rules.size() > 0) {
            model_loader.set_wtype_override(wtype, tensor_type_rules);
        }


        std::map<ggml_type, uint32_t> wtype_stat                 = model_loader.get_wtype_stat();
        std::map<ggml_type, uint32_t> conditioner_wtype_stat     = model_loader.get_conditioner_wtype_stat();
        std::map<ggml_type, uint32_t> diffusion_model_wtype_stat = model_loader.get_diffusion_model_wtype_stat();
        std::map<ggml_type, uint32_t> vae_wtype_stat             = model_loader.get_vae_wtype_stat();

        auto wtype_stat_to_str = [](const std::map<ggml_type, uint32_t>& m, int key_width = 8, int value_width = 5) -> std::string {
            std::ostringstream oss;
            bool first = true;
            for (const auto& [type, count] : m) {
                if (!first)
                    oss << "|";
                first = false;
                oss << std::right << std::setw(key_width) << ggml_type_name(type)
                    << ": "
                    << std::left << std::setw(value_width) << count;
            }
            return oss.str();
        };

        LOG_INFO("Weight type stat:                 %s", wtype_stat_to_str(wtype_stat).c_str());
        LOG_INFO("Conditioner weight type stat:     %s", wtype_stat_to_str(conditioner_wtype_stat).c_str());
        LOG_INFO("Diffusion model weight type stat: %s", wtype_stat_to_str(diffusion_model_wtype_stat).c_str());
        LOG_INFO("VAE weight type stat:             %s", wtype_stat_to_str(vae_wtype_stat).c_str());

        LOG_DEBUG("ggml tensor size = %d bytes", (int)sizeof(ggml_tensor));

        // ------------------------------------------------------------------
        // Auto-fit: compute per-component GPU/CPU placement plan based on
        // currently free device memory. Runs before backend resolution so we
        // can redirect the DiT backend and set per-component placement flags.
        // Only affects the run when sd_ctx_params->auto_fit is true.
        if (sd_ctx_params->auto_fit) {
            backend_fit::ComputeReserves reserves;
            if (sd_ctx_params->auto_fit_compute_reserve_dit_mb > 0) {
                reserves.dit_bytes =
                    int64_t(sd_ctx_params->auto_fit_compute_reserve_dit_mb) * backend_fit::MiB;
            }
            if (sd_ctx_params->auto_fit_compute_reserve_vae_mb > 0) {
                reserves.vae_bytes =
                    int64_t(sd_ctx_params->auto_fit_compute_reserve_vae_mb) * backend_fit::MiB;
            }
            if (sd_ctx_params->auto_fit_compute_reserve_cond_mb > 0) {
                reserves.conditioner_bytes =
                    int64_t(sd_ctx_params->auto_fit_compute_reserve_cond_mb) * backend_fit::MiB;
            }

            const int64_t alignment_guess = 256;
            auto components = backend_fit::estimate_components(
                model_loader, wtype, alignment_guess, reserves);
            auto devices = backend_fit::enumerate_gpu_devices();
            int64_t margin_bytes =
                int64_t(std::max(0, sd_ctx_params->auto_fit_target_mb)) * backend_fit::MiB;
            const bool allow_split = sd_ctx_params->auto_tensor_split &&
                                     devices.size() >= 2 &&
                                     getenv("SD_CUDA_TENSOR_SPLIT") == nullptr;
            auto plan = backend_fit::compute_plan(components, devices, margin_bytes, allow_split);
            backend_fit::print_plan(plan, components, devices, margin_bytes);

            if (sd_ctx_params->auto_fit_dry_run) {
                LOG_INFO("auto-fit: --fit-dry-run set, aborting init before loading models");
                return false;
            }

            // Apply plan to fit_override.
            fit_override.enabled = true;
            auto dit_d  = backend_fit::find_decision(plan, backend_fit::ComponentKind::DIT);
            auto vae_d  = backend_fit::find_decision(plan, backend_fit::ComponentKind::VAE);
            auto cond_d = backend_fit::find_decision(plan, backend_fit::ComponentKind::CONDITIONER);

            if (dit_d) {
                fit_override.dit_device_id      = dit_d->device_id;
                fit_override.dit_offload_params =
                    (dit_d->placement == backend_fit::Placement::GPU_OFFLOAD_PARAMS);
                // Re-init the main backend if the chosen DiT device differs from
                // whatever init_backend() picked. Keep `backend_device_id` in
                // sync — it's what resolve_component_backend compares against.
                const int current_dev = backend_device_id;
                if (!ggml_backend_is_cpu(backend) && dit_d->placement == backend_fit::Placement::CPU) {
                    LOG_INFO("auto-fit: switching DiT backend from GPU %d to CPU", current_dev);
                    ggml_backend_free(backend);
                    backend            = ggml_backend_cpu_init();
                    backend_device_id  = BACKEND_DEVICE_CPU;
                } else if (dit_d->placement != backend_fit::Placement::CPU &&
                           dit_d->device_id != current_dev) {
                    LOG_INFO("auto-fit: switching DiT backend from GPU %d to GPU %d",
                             current_dev, dit_d->device_id);
                    ggml_backend_free(backend);
                    backend            = init_device_backend(dit_d->device_id, "DiT (auto-fit)");
                    backend_device_id  = dit_d->device_id;
                }
            }
            if (vae_d) {
                fit_override.vae_device_id      = vae_d->device_id;
                fit_override.vae_on_cpu         = (vae_d->placement == backend_fit::Placement::CPU);
                fit_override.vae_offload_params =
                    (vae_d->placement == backend_fit::Placement::GPU_OFFLOAD_PARAMS);
            }
            if (cond_d) {
                fit_override.cond_device_id      = cond_d->device_id;
                fit_override.cond_on_cpu         = (cond_d->placement == backend_fit::Placement::CPU);
                fit_override.cond_offload_params =
                    (cond_d->placement == backend_fit::Placement::GPU_OFFLOAD_PARAMS);
            }

            // Capture state for auto-VAE-tiling at gen time. We can't read
            // sd_ctx_params after this scope, so store what we'll need.
            auto_fit_enabled                   = true;
            auto_fit_vae_compute_reserve_bytes = reserves.vae_bytes;
            auto_fit_vae_on_cpu                = vae_d && vae_d->placement == backend_fit::Placement::CPU;

            // If auto-fit placed any component in tensor-split mode, capture
            // that here so init_tensor_split below configures the right flags.
            const bool dit_split  = dit_d  && dit_d->placement  == backend_fit::Placement::GPU_TENSOR_SPLIT;
            const bool cond_split = cond_d && cond_d->placement == backend_fit::Placement::GPU_TENSOR_SPLIT;
            if (dit_split)  pending_split_dit  = true;
            if (cond_split) pending_split_cond = true;

            // For tensor-split components, the device id semantically means
            // "main GPU + extras"; pin that to whatever main backend ended up
            // active so resolve_component_backend doesn't try to forward to
            // a non-main GPU.
            if (dit_split) {
                fit_override.dit_device_id      = backend_device_id;
                fit_override.dit_offload_params = false;
            }
            if (cond_split) {
                fit_override.cond_device_id      = backend_device_id;
                fit_override.cond_on_cpu         = false;
                fit_override.cond_offload_params = false;
            }
        }
        init_tensor_split(sd_ctx_params->auto_tensor_split);

        if (sd_ctx_params->lora_apply_mode == LORA_APPLY_AUTO) {
            bool have_quantized_weight = false;
            if (wtype != GGML_TYPE_COUNT && ggml_is_quantized(wtype)) {
                have_quantized_weight = true;
            } else {
                for (const auto& [type, _] : wtype_stat) {
                    if (ggml_is_quantized(type)) {
                        have_quantized_weight = true;
                        break;
                    }
                }
            }
            if (have_quantized_weight) {
                apply_lora_immediately = false;
            } else {
                apply_lora_immediately = true;
            }
        } else if (sd_ctx_params->lora_apply_mode == LORA_APPLY_IMMEDIATELY) {
            apply_lora_immediately = true;
        } else {
            apply_lora_immediately = false;
        }

        if (sd_version_is_control(version)) {
            // Might need vae encode for control cond
            vae_decode_only = false;
        }

        bool tae_preview_only = sd_ctx_params->tae_preview_only;
        if (version == VERSION_SDXS_512_DS || version == VERSION_SDXS_09) {
            tae_preview_only = false;
            use_tae          = true;
        }

        if (sd_ctx_params->circular_x || sd_ctx_params->circular_y) {
            LOG_INFO("Using circular padding for convolutions");
        }

        bool clip_on_cpu = sd_ctx_params->keep_clip_on_cpu;
        if (fit_override.enabled && fit_override.cond_on_cpu) {
            clip_on_cpu = true;
        }

        // LTX-2 Gemma 3: default the text encoder to CPU. Per-layer F32
        // reduction-order disagreement between cuBLAS's tile reduction and
        // CPU's SIMD horizontal reduction seeds ~6e-5 drift at the first
        // RMSNorm and compounds across 48 transformer layers; the final
        // hidden state ends up ~10× off in absolute terms, which is enough
        // to alter prompt semantics in the LTX-2 conditioner (e.g. drop the
        // subject — "cat on beach" → "person on beach"). Allow the user to
        // override via SD_CUDA_DEVICE_CLIP=N or via auto-fit; warn loudly
        // when they do, especially on quantized weights.
        if (sd_version_is_ltx2(version) && !clip_on_cpu) {
            const char* explicit_clip   = getenv("SD_CUDA_DEVICE_CLIP");
            const bool  user_set_cpu    = (explicit_clip != nullptr && std::atoi(explicit_clip) < 0);
            const bool  user_set_cuda   = (explicit_clip != nullptr && std::atoi(explicit_clip) >= 0);
            const bool  autofit_picks   = fit_override.enabled && fit_override.cond_device_id >= 0;
            if (user_set_cpu) {
                clip_on_cpu = true;  // explicit -1 → CPU, no message needed
            } else if (!user_set_cuda && !autofit_picks) {
                clip_on_cpu = true;
                LOG_INFO("LTX-2: defaulting Gemma 3 text encoder to CPU "
                         "(CUDA path has cumulative F32 drift that can alter prompt semantics). "
                         "Set SD_CUDA_DEVICE_CLIP=N to run on CUDA device N anyway.");
            } else {
                auto q_proj_it = tensor_storage_map.find("text_encoder.model.layers.0.self_attn.q_proj.weight");
                const bool gemma_quantized = (q_proj_it != tensor_storage_map.end() &&
                                              ggml_is_quantized(q_proj_it->second.type));
                if (gemma_quantized) {
                    LOG_WARN("LTX-2: running QUANTIZED Gemma 3 text encoder on CUDA. "
                             "Cumulative F32 reduction-order drift across 48 layers can shift "
                             "the prompt embedding enough to lose subject/style cues "
                             "(e.g. \"cat on a beach\" → \"person on a beach\"). "
                             "Unset SD_CUDA_DEVICE_CLIP (or set it to -1) to use the CPU "
                             "encoder for full prompt fidelity.");
                } else {
                    LOG_WARN("LTX-2: running Gemma 3 text encoder on CUDA. "
                             "Cumulative F32 reduction-order drift across 48 layers may alter "
                             "prompt semantics. Unset SD_CUDA_DEVICE_CLIP to use CPU.");
                }
            }
        }

        // Per-component offload flags. `offload_params_to_cpu` (the user's
        // global --offload-to-cpu) applies to every component. Auto-fit may
        // additionally force DiT-only offload when the DiT doesn't fit in
        // VRAM; that MUST NOT be propagated to the Conditioner/VAE, otherwise
        // their weights get pinned in RAM and the system can OOM (e.g. an
        // LTX-2 run pinning Gemma 9.5 GB + DiT 13 GB + VAE 1.4 GB in 32 GB RAM).
        const bool dit_offload  = offload_params_to_cpu ||
                                  (fit_override.enabled && fit_override.dit_offload_params);
        const bool cond_offload = offload_params_to_cpu ||
                                  (fit_override.enabled && fit_override.cond_offload_params);
        const bool vae_offload  = offload_params_to_cpu ||
                                  (fit_override.enabled && fit_override.vae_offload_params);

        {
            // Pick a device for the text-encoder stack. SD_CUDA_DEVICE_CLIP overrides
            // (set to -1 for CPU); `keep_clip_on_cpu` still forces CPU regardless.
            // When auto-fit is active, fit_override.cond_device_id wins.
            clip_backend = resolve_component_backend(
                "SD_CUDA_DEVICE_CLIP", "CLIP/TextEncoder", clip_on_cpu,
                fit_override.enabled ? fit_override.cond_device_id : -2);
            if (sd_version_is_sd3(version)) {
                cond_stage_model = std::make_shared<SD3CLIPEmbedder>(clip_backend,
                                                                     cond_offload,
                                                                     tensor_storage_map);
                diffusion_model  = std::make_shared<MMDiTModel>(backend,
                                                               dit_offload,
                                                               tensor_storage_map);
            } else if (sd_version_is_flux(version)) {
                bool is_chroma = false;
                for (auto pair : tensor_storage_map) {
                    if (pair.first.find("distilled_guidance_layer.in_proj.weight") != std::string::npos) {
                        is_chroma = true;
                        break;
                    }
                }
                if (is_chroma) {
                    if ((sd_ctx_params->flash_attn || sd_ctx_params->diffusion_flash_attn) && sd_ctx_params->chroma_use_dit_mask) {
                        LOG_WARN(
                            "!!!It looks like you are using Chroma with flash attention. "
                            "This is currently unsupported. "
                            "If you find that the generated images are broken, "
                            "try either disabling flash attention or specifying "
                            "--chroma-disable-dit-mask as a workaround.");
                    }

                    cond_stage_model = std::make_shared<T5CLIPEmbedder>(clip_backend,
                                                                        cond_offload,
                                                                        tensor_storage_map,
                                                                        sd_ctx_params->chroma_use_t5_mask,
                                                                        sd_ctx_params->chroma_t5_mask_pad);
                } else if (version == VERSION_OVIS_IMAGE) {
                    cond_stage_model = std::make_shared<LLMEmbedder>(clip_backend,
                                                                     cond_offload,
                                                                     tensor_storage_map,
                                                                     version,
                                                                     "",
                                                                     false);
                } else {
                    cond_stage_model = std::make_shared<FluxCLIPEmbedder>(clip_backend,
                                                                          cond_offload,
                                                                          tensor_storage_map);
                }
                diffusion_model = std::make_shared<FluxModel>(backend,
                                                              dit_offload,
                                                              tensor_storage_map,
                                                              version,
                                                              sd_ctx_params->chroma_use_dit_mask);
            } else if (sd_version_is_flux2(version)) {
                bool is_chroma   = false;
                cond_stage_model = std::make_shared<LLMEmbedder>(clip_backend,
                                                                 cond_offload,
                                                                 tensor_storage_map,
                                                                 version);
                diffusion_model  = std::make_shared<FluxModel>(backend,
                                                              dit_offload,
                                                              tensor_storage_map,
                                                              version,
                                                              sd_ctx_params->chroma_use_dit_mask);
            } else if (sd_version_is_wan(version)) {
                cond_stage_model = std::make_shared<T5CLIPEmbedder>(clip_backend,
                                                                    cond_offload,
                                                                    tensor_storage_map,
                                                                    true,
                                                                    0,
                                                                    true);
                diffusion_model  = std::make_shared<WanModel>(backend,
                                                             dit_offload,
                                                             tensor_storage_map,
                                                             "model.diffusion_model",
                                                             version);
                if (strlen(SAFE_STR(sd_ctx_params->high_noise_diffusion_model_path)) > 0) {
                    high_noise_diffusion_model = std::make_shared<WanModel>(backend,
                                                                            dit_offload,
                                                                            tensor_storage_map,
                                                                            "model.high_noise_diffusion_model",
                                                                            version);
                }
                if (diffusion_model->get_desc() == "Wan2.1-I2V-14B" ||
                    diffusion_model->get_desc() == "Wan2.1-FLF2V-14B" ||
                    diffusion_model->get_desc() == "Wan2.1-I2V-1.3B") {
                    clip_vision = std::make_shared<FrozenCLIPVisionEmbedder>(backend,
                                                                             dit_offload,
                                                                             tensor_storage_map);
                    clip_vision->alloc_params_buffer();
                    clip_vision->get_param_tensors(tensors);
                }
            } else if (sd_version_is_qwen_image(version)) {
                bool enable_vision = false;
                if (!vae_decode_only) {
                    enable_vision = true;
                }
                cond_stage_model = std::make_shared<LLMEmbedder>(clip_backend,
                                                                 cond_offload,
                                                                 tensor_storage_map,
                                                                 version,
                                                                 "",
                                                                 enable_vision);
                diffusion_model  = std::make_shared<QwenImageModel>(backend,
                                                                   dit_offload,
                                                                   tensor_storage_map,
                                                                   "model.diffusion_model",
                                                                   version,
                                                                   sd_ctx_params->qwen_image_zero_cond_t);
            } else if (sd_version_is_anima(version)) {
                cond_stage_model = std::make_shared<AnimaConditioner>(clip_backend,
                                                                      cond_offload,
                                                                      tensor_storage_map);
                diffusion_model  = std::make_shared<AnimaModel>(backend,
                                                               dit_offload,
                                                               tensor_storage_map,
                                                               "model.diffusion_model");
            } else if (sd_version_is_z_image(version)) {
                cond_stage_model = std::make_shared<LLMEmbedder>(clip_backend,
                                                                 cond_offload,
                                                                 tensor_storage_map,
                                                                 version);
                diffusion_model  = std::make_shared<ZImageModel>(backend,
                                                                dit_offload,
                                                                tensor_storage_map,
                                                                "model.diffusion_model",
                                                                version);
            } else if (sd_version_is_ernie_image(version)) {
                cond_stage_model = std::make_shared<LLMEmbedder>(clip_backend,
                                                                 cond_offload,
                                                                 tensor_storage_map,
                                                                 version);
                diffusion_model  = std::make_shared<ErnieImageModel>(backend,
                                                                    dit_offload,
                                                                    tensor_storage_map,
                                                                    "model.diffusion_model");
            } else if (sd_version_is_ltx2(version)) {
                // LTX-2: Gemma 3 text encoder (Phase 8), 1D embeddings connector + DiT
                // caption_projection (Phase 9), and LTX-2 causal 3D VAE (Phase 11) are all
                // landed. LTX2GemmaConditioner auto-detects connector presence from the
                // tensor map; if absent it falls back to Gemma's last_hidden_state.
                // The tokenizer.json path is required — prompts can't be encoded without
                // it. Any HuggingFace-format `tokenizer.json` for Gemma 3 works.
                {
                    auto split_guard = begin_pending_split(tensor_split_state.split_cond);
                    cond_stage_model = std::make_shared<LTX2GemmaConditioner>(clip_backend,
                                                                              cond_offload,
                                                                              tensor_storage_map,
                                                                              "text_encoder",
                                                                              SAFE_STR(sd_ctx_params->gemma_tokenizer_path));
                }
                {
                    auto split_guard = begin_pending_split(tensor_split_state.split_dit);
                    diffusion_model  = std::make_shared<LTXDiffusionModel>(backend,
                                                                          dit_offload,
                                                                          tensor_storage_map,
                                                                          "model.diffusion_model",
                                                                          version);
                }
            } else {  // SD1.x SD2.x SDXL
                std::map<std::string, std::string> embbeding_map;
                for (uint32_t i = 0; i < sd_ctx_params->embedding_count; i++) {
                    embbeding_map.emplace(SAFE_STR(sd_ctx_params->embeddings[i].name), SAFE_STR(sd_ctx_params->embeddings[i].path));
                }
                if (strstr(SAFE_STR(sd_ctx_params->photo_maker_path), "v2")) {
                    cond_stage_model = std::make_shared<FrozenCLIPEmbedderWithCustomWords>(clip_backend,
                                                                                           cond_offload,
                                                                                           tensor_storage_map,
                                                                                           embbeding_map,
                                                                                           version,
                                                                                           PM_VERSION_2);
                } else {
                    cond_stage_model = std::make_shared<FrozenCLIPEmbedderWithCustomWords>(clip_backend,
                                                                                           cond_offload,
                                                                                           tensor_storage_map,
                                                                                           embbeding_map,
                                                                                           version);
                }
                diffusion_model = std::make_shared<UNetModel>(backend,
                                                              dit_offload,
                                                              tensor_storage_map,
                                                              version);
                if (sd_ctx_params->diffusion_conv_direct) {
                    LOG_INFO("Using Conv2d direct in the diffusion model");
                    std::dynamic_pointer_cast<UNetModel>(diffusion_model)->unet.set_conv2d_direct_enabled(true);
                }
            }

            // ---- Conditioner: optionally lazy-load the LLM (e.g. Gemma) ----
            // The connector stays eager because it's tiny relative to the LLM.
            // Lazy mode skips adding LLM tensors to the global eager-load map
            // and registers a callback that reads them from disk on first encode.
            if (lazy_load_cond) {
                std::map<std::string, ggml_tensor*> llm_lazy_tensors;
                cond_stage_model->get_llm_param_tensors(llm_lazy_tensors);
                int n_threads_local = n_threads;
                bool enable_mmap    = sd_ctx_params->enable_mmap;
                cond_stage_model->set_llm_lazy_load([this, llm_lazy_tensors, n_threads_local, enable_mmap]() mutable {
                    return model_loader_->load_tensors(llm_lazy_tensors, load_ignore_tensors, n_threads_local, enable_mmap);
                });
                size_t before = tensors.size();
                cond_stage_model->get_non_llm_param_tensors(tensors);
                LOG_INFO("lazy_cond: %zu LLM tensors (lazy), +%zu non-LLM tensors (eager)",
                         llm_lazy_tensors.size(), tensors.size() - before);
            } else {
                cond_stage_model->get_param_tensors(tensors);
            }
            cond_stage_model->alloc_params_buffer();

            // ---- DiT: optionally lazy-load entirely (single inner runner) ----
            if (lazy_load_dit) {
                std::map<std::string, ggml_tensor*> dit_lazy_tensors;
                diffusion_model->get_param_tensors(dit_lazy_tensors);
                int n_threads_local = n_threads;
                bool enable_mmap    = sd_ctx_params->enable_mmap;
                diffusion_model->set_lazy_load([this, dit_lazy_tensors, n_threads_local, enable_mmap]() mutable {
                    return model_loader_->load_tensors(dit_lazy_tensors, load_ignore_tensors, n_threads_local, enable_mmap);
                });
            } else {
                diffusion_model->get_param_tensors(tensors);
            }
            diffusion_model->alloc_params_buffer();

            if (sd_version_is_unet_edit(version)) {
                vae_decode_only = false;
            }

            if (high_noise_diffusion_model) {
                high_noise_diffusion_model->alloc_params_buffer();
                high_noise_diffusion_model->get_param_tensors(tensors);
            }

            // Pick a device for the VAE. SD_CUDA_DEVICE_VAE overrides (set to -1 for CPU);
            // `keep_vae_on_cpu` still forces CPU regardless. Auto-fit, when active,
            // supplies fit_override.vae_device_id which takes precedence over env.
            bool vae_on_cpu = sd_ctx_params->keep_vae_on_cpu;
            if (fit_override.enabled && fit_override.vae_on_cpu) {
                vae_on_cpu = true;
            }
            vae_backend = resolve_component_backend(
                "SD_CUDA_DEVICE_VAE", "VAE", vae_on_cpu,
                fit_override.enabled ? fit_override.vae_device_id : -2);

            auto create_tae = [&]() -> std::shared_ptr<VAE> {
                if (sd_version_is_wan(version) ||
                    sd_version_is_qwen_image(version) ||
                    sd_version_is_anima(version)) {
                    return std::make_shared<TinyVideoAutoEncoder>(vae_backend,
                                                                  vae_offload,
                                                                  tensor_storage_map,
                                                                  "decoder",
                                                                  vae_decode_only,
                                                                  version);

                } else {
                    auto model = std::make_shared<TinyImageAutoEncoder>(vae_backend,
                                                                        vae_offload,
                                                                        tensor_storage_map,
                                                                        "decoder.layers",
                                                                        vae_decode_only,
                                                                        version);
                    return model;
                }
            };

            auto create_vae = [&]() -> std::shared_ptr<VAE> {
                if (sd_version_is_wan(version) ||
                    sd_version_is_qwen_image(version) ||
                    sd_version_is_anima(version)) {
                    return std::make_shared<WAN::WanVAERunner>(vae_backend,
                                                               vae_offload,
                                                               tensor_storage_map,
                                                               "first_stage_model",
                                                               vae_decode_only,
                                                               version);
                } else if (sd_version_is_ltx2(version)) {
                    // LTX-2 VAE: in the real checkpoint after convert_tensors_name,
                    // the `vae.` → `first_stage_model.` rename from name_conversion.cpp
                    // puts weights under the standard `first_stage_model.` prefix. The
                    // sd-vae-parity test uses a pre-named `vae.` state dict directly so
                    // it can run on the parity dumper's output without going through the
                    // conversion pass.
                    //
                    // The 22B checkpoint (see `ltx2_22b_{enc,dec}_specs`) has a 9-block
                    // encoder/decoder with mixed RES_X and COMPRESS_* blocks — much deeper
                    // than the 4-block tiny-test default. We hardcode the 22B spec here for
                    // the smoke test; a proper auto-detect from tensor shapes is a follow-up.
                    return std::make_shared<LTXVAE::LTX2VAERunner>(vae_backend,
                                                                   vae_offload,
                                                                   tensor_storage_map,
                                                                   "first_stage_model",
                                                                   version,
                                                                   /*in_ch=*/3,
                                                                   /*latent_ch=*/128,
                                                                   /*patch=*/4,
                                                                   /*decoder_base_ch=*/128,
                                                                   /*timestep_cond=*/false,
                                                                   LTXVAE::LTX2VAERunner::ltx2_22b_enc_specs(),
                                                                   LTXVAE::LTX2VAERunner::ltx2_22b_dec_specs());
                } else {
                    auto model = std::make_shared<AutoEncoderKL>(vae_backend,
                                                                 vae_offload,
                                                                 tensor_storage_map,
                                                                 "first_stage_model",
                                                                 vae_decode_only,
                                                                 false,
                                                                 version);
                    if (sd_version_is_sdxl(version) &&
                        (strlen(SAFE_STR(sd_ctx_params->vae_path)) == 0 || sd_ctx_params->force_sdxl_vae_conv_scale || external_vae_is_invalid)) {
                        float vae_conv_2d_scale = 1.f / 32.f;
                        LOG_WARN(
                            "No valid VAE specified with --vae or --force-sdxl-vae-conv-scale flag set, "
                            "using Conv2D scale %.3f",
                            vae_conv_2d_scale);
                        model->set_conv2d_scale(vae_conv_2d_scale);
                    }
                    return model;
                }
            };

            if (version == VERSION_CHROMA_RADIANCE) {
                LOG_INFO("using FakeVAE");
                first_stage_model = std::make_shared<FakeVAE>(version,
                                                              vae_backend,
                                                              vae_offload);
            } else if (use_tae && !tae_preview_only) {
                LOG_INFO("using TAE for encoding / decoding");
                first_stage_model = create_tae();
                first_stage_model->alloc_params_buffer();
                first_stage_model->get_param_tensors(tensors, "tae");
            } else {
                LOG_INFO("using VAE for encoding / decoding");
                first_stage_model = create_vae();
                first_stage_model->alloc_params_buffer();
                first_stage_model->get_param_tensors(tensors, "first_stage_model");
                if (use_tae && tae_preview_only) {
                    LOG_INFO("using TAE for preview");
                    preview_vae = create_tae();
                    preview_vae->alloc_params_buffer();
                    preview_vae->get_param_tensors(tensors, "tae");
                }
            }

            if (sd_ctx_params->vae_conv_direct) {
                LOG_INFO("Using Conv2d direct in the vae model");
                first_stage_model->set_conv2d_direct_enabled(true);
                if (preview_vae) {
                    preview_vae->set_conv2d_direct_enabled(true);
                }
            }

            if (strlen(SAFE_STR(sd_ctx_params->control_net_path)) > 0) {
                ggml_backend_t controlnet_backend = nullptr;
                if (sd_ctx_params->keep_control_net_on_cpu && !ggml_backend_is_cpu(backend)) {
                    LOG_DEBUG("ControlNet: Using CPU backend");
                    controlnet_backend = ggml_backend_cpu_init();
                } else {
                    controlnet_backend = backend;
                }
                control_net = std::make_shared<ControlNet>(controlnet_backend,
                                                           offload_params_to_cpu,
                                                           tensor_storage_map,
                                                           version);
                if (sd_ctx_params->diffusion_conv_direct) {
                    LOG_INFO("Using Conv2d direct in the control net");
                    control_net->set_conv2d_direct_enabled(true);
                }
            }

            if (strstr(SAFE_STR(sd_ctx_params->photo_maker_path), "v2")) {
                pmid_model = std::make_shared<PhotoMakerIDEncoder>(backend,
                                                                   dit_offload,
                                                                   tensor_storage_map,
                                                                   "pmid",
                                                                   version,
                                                                   PM_VERSION_2);
                LOG_INFO("using PhotoMaker Version 2");
            } else {
                pmid_model = std::make_shared<PhotoMakerIDEncoder>(backend,
                                                                   dit_offload,
                                                                   tensor_storage_map,
                                                                   "pmid",
                                                                   version);
            }
            if (strlen(SAFE_STR(sd_ctx_params->photo_maker_path)) > 0) {
                pmid_lora               = std::make_shared<LoraModel>("pmid", backend, sd_ctx_params->photo_maker_path, "", version);
                auto lora_tensor_filter = [&](const std::string& tensor_name) {
                    if (starts_with(tensor_name, "lora.model")) {
                        return true;
                    }
                    return false;
                };
                if (!pmid_lora->load_from_file(n_threads, lora_tensor_filter)) {
                    LOG_WARN("load photomaker lora tensors from %s failed", sd_ctx_params->photo_maker_path);
                    return false;
                }
                LOG_INFO("loading stacked ID embedding (PHOTOMAKER) model file from '%s'", sd_ctx_params->photo_maker_path);
                if (!model_loader.init_from_file_and_convert_name(sd_ctx_params->photo_maker_path, "pmid.")) {
                    LOG_WARN("loading stacked ID embedding from '%s' failed", sd_ctx_params->photo_maker_path);
                } else {
                    use_pmid = true;
                }
            }
            if (use_pmid) {
                if (!pmid_model->alloc_params_buffer()) {
                    LOG_ERROR(" pmid model params buffer allocation failed");
                    return false;
                }
                pmid_model->get_param_tensors(tensors, "pmid");
            }

            if (sd_ctx_params->flash_attn) {
                LOG_INFO("Using flash attention");
                cond_stage_model->set_flash_attention_enabled(true);
                if (clip_vision) {
                    clip_vision->set_flash_attention_enabled(true);
                }
                if (first_stage_model) {
                    first_stage_model->set_flash_attention_enabled(true);
                }
                if (preview_vae) {
                    preview_vae->set_flash_attention_enabled(true);
                }
            }

            if (sd_ctx_params->flash_attn || sd_ctx_params->diffusion_flash_attn) {
                LOG_INFO("Using flash attention in the diffusion model");
                diffusion_model->set_flash_attention_enabled(true);
                if (high_noise_diffusion_model) {
                    high_noise_diffusion_model->set_flash_attention_enabled(true);
                }
            }

            diffusion_model->set_circular_axes(sd_ctx_params->circular_x, sd_ctx_params->circular_y);
            if (high_noise_diffusion_model) {
                high_noise_diffusion_model->set_circular_axes(sd_ctx_params->circular_x, sd_ctx_params->circular_y);
            }
            if (control_net) {
                control_net->set_circular_axes(sd_ctx_params->circular_x, sd_ctx_params->circular_y);
            }
            circular_x = sd_ctx_params->circular_x;
            circular_y = sd_ctx_params->circular_y;
        }

        ggml_init_params params;
        params.mem_size   = static_cast<size_t>(10 * 1024) * 1024;  // 10M
        params.mem_buffer = nullptr;
        params.no_alloc   = false;
        // LOG_DEBUG("mem_size %u ", params.mem_size);
        ggml_context* ctx = ggml_init(params);  // for  alphas_cumprod and is_using_v_parameterization check
        GGML_ASSERT(ctx != nullptr);
        ggml_tensor* alphas_cumprod_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, TIMESTEPS);
        calculate_alphas_cumprod((float*)alphas_cumprod_tensor->data);

        // load weights
        LOG_DEBUG("loading weights");

        std::set<std::string> ignore_tensors;
        tensors["alphas_cumprod"] = alphas_cumprod_tensor;
        if (use_tae && !tae_preview_only) {
            ignore_tensors.insert("first_stage_model.");
        }
        if (use_pmid) {
            ignore_tensors.insert("pmid.unet.");
        }
        ignore_tensors.insert("model.diffusion_model.__x0__");
        ignore_tensors.insert("model.diffusion_model.__32x32__");
        ignore_tensors.insert("model.diffusion_model.__index_timestep_zero__");

        if (vae_decode_only) {
            ignore_tensors.insert("first_stage_model.encoder");
            ignore_tensors.insert("first_stage_model.conv1");
            ignore_tensors.insert("first_stage_model.quant");
            ignore_tensors.insert("tae.encoder");
            ignore_tensors.insert("text_encoders.llm.visual.");
        }
        if (version == VERSION_OVIS_IMAGE) {
            ignore_tensors.insert("text_encoders.llm.vision_model.");
            ignore_tensors.insert("text_encoders.llm.visual_tokenizer.");
            ignore_tensors.insert("text_encoders.llm.vte.");
        }
        if (version == VERSION_SVD) {
            ignore_tensors.insert("conditioner.embedders.3");
        }
        if (sd_version_is_ernie_image(version)) {
            ignore_tensors.insert("text_encoders.llm.vision_tower.");
            ignore_tensors.insert("text_encoders.llm.multi_modal_projector.");
        }
        // Stash ignore_tensors so lazy-load callbacks (registered earlier) can
        // re-pass it when ModelLoader::load_tensors fires per-component.
        load_ignore_tensors = ignore_tensors;
        // Debug: dump tensor names containing video_embeddings_connector to verify
        // they're registered (only in lazy_load_cond mode).
        bool success = model_loader.load_tensors(tensors, ignore_tensors, n_threads, sd_ctx_params->enable_mmap);
        if (!success) {
            LOG_ERROR("load tensors from model loader failed");
            ggml_free(ctx);
            return false;
        }

        LOG_DEBUG("finished loaded file");

        {
            size_t clip_params_mem_size = cond_stage_model->get_params_buffer_size();
            size_t unet_params_mem_size = diffusion_model->get_params_buffer_size();
            if (high_noise_diffusion_model) {
                unet_params_mem_size += high_noise_diffusion_model->get_params_buffer_size();
            }
            size_t vae_params_mem_size = 0;
            vae_params_mem_size        = first_stage_model->get_params_buffer_size();
            if (preview_vae) {
                vae_params_mem_size += preview_vae->get_params_buffer_size();
            }
            size_t control_net_params_mem_size = 0;
            if (control_net) {
                if (!control_net->load_from_file(SAFE_STR(sd_ctx_params->control_net_path), n_threads)) {
                    return false;
                }
                control_net_params_mem_size = control_net->get_params_buffer_size();
            }
            size_t pmid_params_mem_size = 0;
            if (use_pmid) {
                pmid_params_mem_size = pmid_model->get_params_buffer_size();
            }

            size_t total_params_ram_size  = 0;
            size_t total_params_vram_size = 0;
            if (ggml_backend_is_cpu(clip_backend)) {
                total_params_ram_size += clip_params_mem_size + pmid_params_mem_size;
            } else {
                total_params_vram_size += clip_params_mem_size + pmid_params_mem_size;
            }

            if (ggml_backend_is_cpu(backend)) {
                total_params_ram_size += unet_params_mem_size;
            } else {
                total_params_vram_size += unet_params_mem_size;
            }

            if (ggml_backend_is_cpu(vae_backend)) {
                total_params_ram_size += vae_params_mem_size;
            } else {
                total_params_vram_size += vae_params_mem_size;
            }

            if (ggml_backend_is_cpu(control_net_backend)) {
                total_params_ram_size += control_net_params_mem_size;
            } else {
                total_params_vram_size += control_net_params_mem_size;
            }

            size_t total_params_size = total_params_ram_size + total_params_vram_size;
            LOG_INFO(
                "total params memory size = %.2fMB (VRAM %.2fMB, RAM %.2fMB): "
                "text_encoders %.2fMB(%s), diffusion_model %.2fMB(%s), vae %.2fMB(%s), controlnet %.2fMB(%s), pmid %.2fMB(%s)",
                total_params_size / 1024.0 / 1024.0,
                total_params_vram_size / 1024.0 / 1024.0,
                total_params_ram_size / 1024.0 / 1024.0,
                clip_params_mem_size / 1024.0 / 1024.0,
                ggml_backend_is_cpu(clip_backend) ? "RAM" : "VRAM",
                unet_params_mem_size / 1024.0 / 1024.0,
                ggml_backend_is_cpu(backend) ? "RAM" : "VRAM",
                vae_params_mem_size / 1024.0 / 1024.0,
                ggml_backend_is_cpu(vae_backend) ? "RAM" : "VRAM",
                control_net_params_mem_size / 1024.0 / 1024.0,
                ggml_backend_is_cpu(control_net_backend) ? "RAM" : "VRAM",
                pmid_params_mem_size / 1024.0 / 1024.0,
                ggml_backend_is_cpu(clip_backend) ? "RAM" : "VRAM");
        }

        // init denoiser
        {
            prediction_t pred_type = sd_ctx_params->prediction;

            if (pred_type == PREDICTION_COUNT) {
                if (sd_version_is_sd2(version)) {
                    // check is_using_v_parameterization_for_sd2
                    if (is_using_v_parameterization_for_sd2(sd_version_is_inpaint(version))) {
                        pred_type = V_PRED;
                    } else {
                        pred_type = EPS_PRED;
                    }
                } else if (sd_version_is_sdxl(version)) {
                    if (tensor_storage_map.find("edm_vpred.sigma_max") != tensor_storage_map.end()) {
                        // CosXL models
                        // TODO: get sigma_min and sigma_max values from file
                        pred_type = EDM_V_PRED;
                    } else if (tensor_storage_map.find("v_pred") != tensor_storage_map.end()) {
                        pred_type = V_PRED;
                    } else {
                        pred_type = EPS_PRED;
                    }
                } else if (sd_version_is_sd3(version) ||
                           sd_version_is_wan(version) ||
                           sd_version_is_qwen_image(version) ||
                           sd_version_is_anima(version) ||
                           sd_version_is_ernie_image(version) ||
                           sd_version_is_z_image(version)) {
                    pred_type = FLOW_PRED;
                    if (sd_version_is_wan(version)) {
                        default_flow_shift = 5.f;
                    } else if (sd_version_is_ernie_image(version)) {
                        default_flow_shift = 4.f;
                    } else {
                        default_flow_shift = 3.f;
                    }
                } else if (sd_version_is_flux(version)) {
                    pred_type = FLUX_FLOW_PRED;

                    default_flow_shift = 1.0f;  // TODO: validate
                    for (const auto& [name, tensor_storage] : tensor_storage_map) {
                        if (starts_with(name, "model.diffusion_model.guidance_in.in_layer.weight")) {
                            default_flow_shift = 1.15f;
                            break;
                        }
                    }
                } else if (sd_version_is_flux2(version)) {
                    pred_type = FLUX2_FLOW_PRED;
                } else if (sd_version_is_ltx2(version)) {
                    pred_type = LTX2_FLOW_PRED;
                } else {
                    pred_type = EPS_PRED;
                }
            }

            switch (pred_type) {
                case EPS_PRED:
                    LOG_INFO("running in eps-prediction mode");
                    break;
                case V_PRED:
                    LOG_INFO("running in v-prediction mode");
                    denoiser = std::make_shared<CompVisVDenoiser>();
                    break;
                case EDM_V_PRED:
                    LOG_INFO("running in v-prediction EDM mode");
                    denoiser = std::make_shared<EDMVDenoiser>();
                    break;
                case FLOW_PRED: {
                    LOG_INFO("running in FLOW mode");
                    denoiser = std::make_shared<DiscreteFlowDenoiser>();
                    break;
                }
                case FLUX_FLOW_PRED: {
                    LOG_INFO("running in Flux FLOW mode");
                    denoiser = std::make_shared<FluxFlowDenoiser>();
                    break;
                }
                case FLUX2_FLOW_PRED: {
                    LOG_INFO("running in Flux2 FLOW mode");
                    denoiser = std::make_shared<Flux2FlowDenoiser>();
                    break;
                }
                case LTX2_FLOW_PRED: {
                    LOG_INFO("running in LTX-2 FLOW mode");
                    denoiser = std::make_shared<LTX2FlowDenoiser>();
                    break;
                }
                default: {
                    LOG_ERROR("Unknown predition type %i", pred_type);
                    ggml_free(ctx);
                    return false;
                }
            }

            auto comp_vis_denoiser = std::dynamic_pointer_cast<CompVisDenoiser>(denoiser);
            if (comp_vis_denoiser) {
                for (int i = 0; i < TIMESTEPS; i++) {
                    comp_vis_denoiser->sigmas[i]     = std::sqrt((1 - ((float*)alphas_cumprod_tensor->data)[i]) / ((float*)alphas_cumprod_tensor->data)[i]);
                    comp_vis_denoiser->log_sigmas[i] = std::log(comp_vis_denoiser->sigmas[i]);
                }
            }
        }

        ggml_free(ctx);
        return true;
    }

    bool is_using_v_parameterization_for_sd2(bool is_inpaint = false) {
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
        if (!concat.empty()) {
            diffusion_params.c_concat = &concat;
        }
        auto out_opt = diffusion_model->compute(n_threads, diffusion_params);
        GGML_ASSERT(!out_opt.empty());
        out = std::move(out_opt);
        diffusion_model->free_compute_buffer();

        double result = static_cast<double>((out - x_t).mean());
        int64_t t1    = ggml_time_ms();
        LOG_DEBUG("check is_using_v_parameterization_for_sd2, taking %.2fs", (t1 - t0) * 1.0f / 1000);
        return result < -1;
    }

    std::shared_ptr<LoraModel> load_lora_model_from_file(const std::string& lora_id,
                                                         float multiplier,
                                                         ggml_backend_t backend,
                                                         LoraModel::filter_t lora_tensor_filter = nullptr) {
        std::string lora_path             = lora_id;
        static std::string high_noise_tag = "|high_noise|";
        bool is_high_noise                = false;
        if (starts_with(lora_path, high_noise_tag)) {
            lora_path     = lora_path.substr(high_noise_tag.size());
            is_high_noise = true;
            LOG_DEBUG("high noise lora: %s", lora_path.c_str());
        }
        auto lora = std::make_shared<LoraModel>(lora_id, backend, lora_path, is_high_noise ? "model.high_noise_" : "", version);
        if (!lora->load_from_file(n_threads, lora_tensor_filter)) {
            LOG_WARN("load lora tensors from %s failed", lora_path.c_str());
            return nullptr;
        }

        lora->multiplier = multiplier;
        return lora;
    }

    void apply_loras_immediately(const std::unordered_map<std::string, float>& lora_state) {
        std::unordered_map<std::string, float> lora_state_diff;
        for (auto& kv : lora_state) {
            const std::string& lora_name = kv.first;
            float multiplier             = kv.second;
            lora_state_diff[lora_name] += multiplier;
        }
        for (auto& kv : curr_lora_state) {
            const std::string& lora_name = kv.first;
            float curr_multiplier        = kv.second;
            lora_state_diff[lora_name] -= curr_multiplier;
        }

        if (lora_state_diff.empty()) {
            return;
        }

        LOG_INFO("apply lora immediately");

        size_t rm = lora_state_diff.size() - lora_state.size();
        if (rm != 0) {
            LOG_INFO("attempting to apply %lu LoRAs (removing %lu applied LoRAs)", lora_state.size(), rm);
        } else {
            LOG_INFO("attempting to apply %lu LoRAs", lora_state.size());
        }

        for (auto& kv : lora_state_diff) {
            int64_t t0 = ggml_time_ms();

            auto lora = load_lora_model_from_file(kv.first, kv.second, backend);
            if (!lora || lora->lora_tensors.empty()) {
                continue;
            }
            lora->apply(tensors, version, n_threads);
            lora->free_params_buffer();

            int64_t t1 = ggml_time_ms();

            LOG_INFO("lora '%s' applied, taking %.2fs", kv.first.c_str(), (t1 - t0) * 1.0f / 1000);
        }

        curr_lora_state = lora_state;
    }

    void apply_loras_at_runtime(const std::unordered_map<std::string, float>& lora_state) {
        cond_stage_lora_models.clear();
        diffusion_lora_models.clear();
        first_stage_lora_models.clear();
        if (cond_stage_model) {
            cond_stage_model->set_weight_adapter(nullptr);
        }
        if (diffusion_model) {
            diffusion_model->set_weight_adapter(nullptr);
        }
        if (high_noise_diffusion_model) {
            high_noise_diffusion_model->set_weight_adapter(nullptr);
        }
        if (first_stage_model) {
            first_stage_model->set_weight_adapter(nullptr);
        }
        if (lora_state.empty()) {
            return;
        }
        LOG_INFO("apply lora at runtime");
        if (cond_stage_model) {
            std::vector<std::shared_ptr<LoraModel>> lora_models;
            auto lora_state_diff = lora_state;
            for (auto& lora_model : cond_stage_lora_models) {
                auto iter = lora_state_diff.find(lora_model->lora_id);

                if (iter != lora_state_diff.end()) {
                    lora_model->multiplier = iter->second;
                    lora_models.push_back(lora_model);
                    lora_state_diff.erase(iter);
                }
            }
            cond_stage_lora_models  = lora_models;
            auto lora_tensor_filter = [&](const std::string& tensor_name) {
                if (is_cond_stage_model_name(tensor_name)) {
                    return true;
                }
                return false;
            };
            for (auto& kv : lora_state_diff) {
                const std::string& lora_id = kv.first;
                float multiplier           = kv.second;

                auto lora = load_lora_model_from_file(lora_id, multiplier, clip_backend, lora_tensor_filter);
                if (lora && !lora->lora_tensors.empty()) {
                    lora->preprocess_lora_tensors(tensors);
                    cond_stage_lora_models.push_back(lora);
                }
            }
            auto multi_lora_adapter = std::make_shared<MultiLoraAdapter>(cond_stage_lora_models);
            cond_stage_model->set_weight_adapter(multi_lora_adapter);
        }
        if (diffusion_model) {
            std::vector<std::shared_ptr<LoraModel>> lora_models;
            auto lora_state_diff = lora_state;
            for (auto& lora_model : diffusion_lora_models) {
                auto iter = lora_state_diff.find(lora_model->lora_id);

                if (iter != lora_state_diff.end()) {
                    lora_model->multiplier = iter->second;
                    lora_models.push_back(lora_model);
                    lora_state_diff.erase(iter);
                }
            }
            diffusion_lora_models   = lora_models;
            auto lora_tensor_filter = [&](const std::string& tensor_name) {
                if (is_diffusion_model_name(tensor_name)) {
                    return true;
                }
                return false;
            };
            for (auto& kv : lora_state_diff) {
                const std::string& lora_name = kv.first;
                float multiplier             = kv.second;

                auto lora = load_lora_model_from_file(lora_name, multiplier, backend, lora_tensor_filter);
                if (lora && !lora->lora_tensors.empty()) {
                    lora->preprocess_lora_tensors(tensors);
                    diffusion_lora_models.push_back(lora);
                }
            }
            auto multi_lora_adapter = std::make_shared<MultiLoraAdapter>(diffusion_lora_models);
            diffusion_model->set_weight_adapter(multi_lora_adapter);
            if (high_noise_diffusion_model) {
                high_noise_diffusion_model->set_weight_adapter(multi_lora_adapter);
            }
        }

        if (first_stage_model) {
            std::vector<std::shared_ptr<LoraModel>> lora_models;
            auto lora_state_diff = lora_state;
            for (auto& lora_model : first_stage_lora_models) {
                auto iter = lora_state_diff.find(lora_model->lora_id);

                if (iter != lora_state_diff.end()) {
                    lora_model->multiplier = iter->second;
                    lora_models.push_back(lora_model);
                    lora_state_diff.erase(iter);
                }
            }
            first_stage_lora_models = lora_models;
            auto lora_tensor_filter = [&](const std::string& tensor_name) {
                if (is_first_stage_model_name(tensor_name)) {
                    return true;
                }
                return false;
            };
            for (auto& kv : lora_state_diff) {
                const std::string& lora_name = kv.first;
                float multiplier             = kv.second;

                auto lora = load_lora_model_from_file(lora_name, multiplier, vae_backend, lora_tensor_filter);
                if (lora && !lora->lora_tensors.empty()) {
                    lora->preprocess_lora_tensors(tensors);
                    first_stage_lora_models.push_back(lora);
                }
            }
            auto multi_lora_adapter = std::make_shared<MultiLoraAdapter>(first_stage_lora_models);
            first_stage_model->set_weight_adapter(multi_lora_adapter);
        }
    }

    void lora_stat() {
        if (!cond_stage_lora_models.empty()) {
            LOG_INFO("cond_stage_lora_models:");
            for (auto& lora_model : cond_stage_lora_models) {
                lora_model->stat();
            }
        }

        if (!diffusion_lora_models.empty()) {
            LOG_INFO("diffusion_lora_models:");
            for (auto& lora_model : diffusion_lora_models) {
                lora_model->stat();
            }
        }

        if (!first_stage_lora_models.empty()) {
            LOG_INFO("first_stage_lora_models:");
            for (auto& lora_model : first_stage_lora_models) {
                lora_model->stat();
            }
        }
    }

    void apply_loras(const sd_lora_t* loras, uint32_t lora_count) {
        std::unordered_map<std::string, float> lora_f2m;
        for (uint32_t i = 0; i < lora_count; i++) {
            std::string lora_id = SAFE_STR(loras[i].path);
            if (loras[i].is_high_noise) {
                lora_id = "|high_noise|" + lora_id;
            }
            lora_f2m[lora_id] = loras[i].multiplier;
            LOG_DEBUG("lora %s:%.2f", lora_id.c_str(), loras[i].multiplier);
        }
        int64_t t0 = ggml_time_ms();
        if (apply_lora_immediately) {
            apply_loras_immediately(lora_f2m);
        } else {
            apply_loras_at_runtime(lora_f2m);
        }
        int64_t t1 = ggml_time_ms();
        if (!lora_f2m.empty()) {
            LOG_INFO("apply_loras completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);
        }
    }

    SDCondition get_pmid_conditon(sd_pm_params_t pm_params,
                                  ConditionerParams& condition_params) {
        SDCondition id_cond;
        if (use_pmid) {
            if (!pmid_lora->applied) {
                int64_t t0 = ggml_time_ms();
                pmid_lora->apply(tensors, version, n_threads);
                int64_t t1         = ggml_time_ms();
                pmid_lora->applied = true;
                LOG_INFO("pmid_lora apply completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);
                if (free_params_immediately) {
                    pmid_lora->free_params_buffer();
                }
            }
            // preprocess input id images
            bool pmv2 = pmid_model->get_version() == PM_VERSION_2;
            if (pm_params.id_images_count > 0) {
                int clip_image_size        = 224;
                pmid_model->style_strength = pm_params.style_strength;
                sd::Tensor<float> id_image_tensor;
                for (int i = 0; i < pm_params.id_images_count; i++) {
                    auto id_image           = sd_image_to_tensor(pm_params.id_images[i]);
                    auto processed_id_image = clip_preprocess(id_image, clip_image_size, clip_image_size);
                    if (id_image_tensor.empty()) {
                        id_image_tensor = processed_id_image;
                    } else {
                        id_image_tensor = sd::ops::concat(id_image_tensor, processed_id_image, 3);
                    }
                }

                int64_t t0                      = ggml_time_ms();
                condition_params.num_input_imgs = pm_params.id_images_count;
                auto cond_tup                   = cond_stage_model->get_learned_condition_with_trigger(n_threads,
                                                                                                       condition_params);
                id_cond                         = std::get<0>(cond_tup);
                auto class_tokens_mask          = std::get<1>(cond_tup);
                sd::Tensor<float> id_embeds;
                if (pmv2 && pm_params.id_embed_path != nullptr) {
                    try {
                        id_embeds = sd::load_tensor_from_file_as_tensor<float>(pm_params.id_embed_path);
                    } catch (const std::exception&) {
                        id_embeds = {};
                    }
                }
                if (pmv2 && id_embeds.empty()) {
                    LOG_WARN("Provided PhotoMaker images, but NO valid ID embeds file for PM v2");
                    LOG_WARN("Turn off PhotoMaker");
                    use_pmid = false;
                } else {
                    if (pmv2 && pm_params.id_images_count != id_embeds.shape()[1]) {
                        LOG_WARN("PhotoMaker image count (%d) does NOT match ID embeds (%d). You should run face_detect.py again.", pm_params.id_images_count, static_cast<int>(id_embeds.shape()[1]));
                        LOG_WARN("Turn off PhotoMaker");
                        use_pmid = false;
                    } else {
                        auto res = pmid_model->compute(n_threads,
                                                       id_image_tensor,
                                                       id_cond.c_crossattn,
                                                       id_embeds,
                                                       class_tokens_mask);
                        if (res.empty()) {
                            LOG_ERROR("Photomaker ID Stacking failed");
                            LOG_WARN("Turn off PhotoMaker");
                            use_pmid = false;
                        } else {
                            id_cond.c_crossattn = std::move(res);
                            int64_t t1          = ggml_time_ms();
                            LOG_INFO("Photomaker ID Stacking, taking %" PRId64 " ms", t1 - t0);
                            // Encode input prompt without the trigger word for delayed conditioning
                            condition_params.text = cond_stage_model->remove_trigger_from_prompt(condition_params.text);
                        }
                        if (free_params_immediately) {
                            pmid_model->free_params_buffer();
                        }
                    }
                }
            } else {
                LOG_WARN("Provided PhotoMaker model file, but NO input ID images");
                LOG_WARN("Turn off PhotoMaker");
                use_pmid = false;
            }
        }
        return id_cond;
    }

    sd::Tensor<float> get_clip_vision_output(const sd::Tensor<float>& image,
                                             bool return_pooled   = true,
                                             int clip_skip        = -1,
                                             bool zero_out_masked = false) {
        sd::Tensor<float> output;
        if (zero_out_masked) {
            if (return_pooled) {
                output = sd::zeros<float>({clip_vision->vision_model.projection_dim});
            } else {
                output = sd::zeros<float>({clip_vision->vision_model.hidden_size, 257});
            }
        } else {
            auto pixel_values = clip_preprocess(image, clip_vision->vision_model.image_size, clip_vision->vision_model.image_size);
            auto output_opt   = clip_vision->compute(n_threads, pixel_values, return_pooled, clip_skip);
            if (output_opt.empty()) {
                LOG_ERROR("clip_vision compute failed");
                return {};
            }
            output = std::move(output_opt);
        }
        return output;
    }

    std::vector<float> process_timesteps(const std::vector<float>& timesteps,
                                         const sd::Tensor<float>& init_latent,
                                         const sd::Tensor<float>& denoise_mask) {
        if (diffusion_model->get_desc() == "Wan2.2-TI2V-5B") {
            auto new_timesteps = std::vector<float>(static_cast<size_t>(init_latent.shape()[2]), timesteps[0]);

            if (!denoise_mask.empty()) {
                float value = denoise_mask.dim() == 5 ? denoise_mask.index(0, 0, 0, 0, 0) : denoise_mask.index(0, 0, 0, 0);
                if (value == 0.f) {
                    new_timesteps[0] = 0.f;
                }
            }
            return new_timesteps;
        } else {
            return timesteps;
        }
    }

    void preview_image(int step,
                       const sd::Tensor<float>& latents,
                       enum SDVersion version,
                       preview_t preview_mode,
                       std::function<void(int, int, sd_image_t*, bool, void*)> step_callback,
                       void* step_callback_data,
                       bool is_noisy) {
        if (preview_mode == PREVIEW_PROJ) {
            int patch_sz                     = 1;
            const float(*latent_rgb_proj)[3] = nullptr;
            float* latent_rgb_bias           = nullptr;
            bool is_video                    = preview_latent_tensor_is_video(latents);
            uint32_t dim                     = is_video ? static_cast<uint32_t>(latents.shape()[3]) : static_cast<uint32_t>(latents.shape()[2]);

            if (dim == 128) {
                if (sd_version_uses_flux2_vae(version)) {
                    latent_rgb_proj = flux2_latent_rgb_proj;
                    latent_rgb_bias = flux2_latent_rgb_bias;
                    patch_sz        = 2;
                }
            } else if (dim == 48) {
                if (sd_version_is_wan(version)) {
                    latent_rgb_proj = wan_22_latent_rgb_proj;
                    latent_rgb_bias = wan_22_latent_rgb_bias;
                } else {
                    LOG_WARN("No latent to RGB projection known for this model");
                    return;
                }
            } else if (dim == 16) {
                if (sd_version_is_sd3(version)) {
                    latent_rgb_proj = sd3_latent_rgb_proj;
                    latent_rgb_bias = sd3_latent_rgb_bias;
                } else if (sd_version_is_flux(version) || sd_version_is_z_image(version)) {
                    latent_rgb_proj = flux_latent_rgb_proj;
                    latent_rgb_bias = flux_latent_rgb_bias;
                } else if (sd_version_is_wan(version) || sd_version_is_qwen_image(version) || sd_version_is_anima(version)) {
                    latent_rgb_proj = wan_21_latent_rgb_proj;
                    latent_rgb_bias = wan_21_latent_rgb_bias;
                } else {
                    LOG_WARN("No latent to RGB projection known for this model");
                    return;
                }
            } else if (dim == 4) {
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
            } else if (dim != 3) {
                LOG_WARN("No latent to RGB projection known for this model");
                return;
            }

            uint32_t frames     = is_video ? static_cast<uint32_t>(latents.shape()[2]) : 1;
            uint32_t img_width  = static_cast<uint32_t>(latents.shape()[0]) * patch_sz;
            uint32_t img_height = static_cast<uint32_t>(latents.shape()[1]) * patch_sz;

            uint8_t* data = (uint8_t*)malloc(frames * img_width * img_height * 3 * sizeof(uint8_t));
            GGML_ASSERT(data != nullptr);
            preview_latent_video(data, latents, latent_rgb_proj, latent_rgb_bias, patch_sz);
            sd_image_t* images = (sd_image_t*)malloc(frames * sizeof(sd_image_t));
            GGML_ASSERT(images != nullptr);
            for (uint32_t i = 0; i < frames; i++) {
                images[i] = {img_width, img_height, 3, data + i * img_width * img_height * 3};
            }
            step_callback(step, frames, images, is_noisy, step_callback_data);
            free(data);
            free(images);
            return;
        }

        if (preview_mode == PREVIEW_VAE || preview_mode == PREVIEW_TAE) {
            sd::Tensor<float> vae_latents;
            sd::Tensor<float> decoded;
            bool is_video = preview_latent_tensor_is_video(latents);
            if (preview_vae) {
                vae_latents = preview_vae->diffusion_to_vae_latents(latents);
                decoded     = preview_vae->decode(n_threads, vae_latents, vae_tiling_params, is_video, circular_x, circular_y, true);
            } else {
                vae_latents = first_stage_model->diffusion_to_vae_latents(latents);
                decoded     = first_stage_model->decode(n_threads, vae_latents, vae_tiling_params, is_video, circular_x, circular_y, true);
            }
            if (decoded.empty()) {
                LOG_ERROR("preview decode failed at step %d", step);
                return;
            }

            is_video           = preview_latent_tensor_is_video(decoded);
            uint32_t frames    = is_video ? static_cast<uint32_t>(decoded.shape()[2]) : 1;
            sd_image_t* images = (sd_image_t*)malloc(frames * sizeof(sd_image_t));
            GGML_ASSERT(images != nullptr);
            for (uint32_t i = 0; i < frames; ++i) {
                images[i] = tensor_to_sd_image(decoded, static_cast<int>(i));
            }

            step_callback(step, frames, images, is_noisy, step_callback_data);
            for (uint32_t i = 0; i < frames; ++i) {
                free(images[i].data);
            }
            free(images);
            return;
        }

        if (preview_mode != PREVIEW_NONE) {
            LOG_WARN("Unsupported preview mode: %d", static_cast<int>(preview_mode));
        }
    }

    std::vector<float> prepare_sample_timesteps(float sigma,
                                                int shifted_timestep) {
        float t = denoiser->sigma_to_t(sigma);
        if (shifted_timestep > 0) {
            float shifted_t_float = t * (float(shifted_timestep) / float(TIMESTEPS));
            int64_t shifted_t     = static_cast<int64_t>(roundf(shifted_t_float));
            shifted_t             = std::max((int64_t)0, std::min((int64_t)(TIMESTEPS - 1), shifted_t));
            LOG_DEBUG("shifting timestep from %.2f to %" PRId64 " (sigma: %.4f)", t, shifted_t, sigma);
            return std::vector<float>{(float)shifted_t};
        }
        if (sd_version_is_anima(version)) {
            return std::vector<float>{t / static_cast<float>(TIMESTEPS)};
        }
        if (sd_version_is_z_image(version)) {
            return std::vector<float>{1000.f - t};
        }
        return std::vector<float>{t};
    }

    void adjust_sample_step_scalings(int shifted_timestep,
                                     const std::vector<float>& timesteps_vec,
                                     float c_in,
                                     float* c_skip,
                                     float* c_out) {
        GGML_ASSERT(c_skip != nullptr);
        GGML_ASSERT(c_out != nullptr);
        if (shifted_timestep <= 0) {
            return;
        }

        int64_t shifted_t_idx              = static_cast<int64_t>(roundf(timesteps_vec[0]));
        float shifted_sigma                = denoiser->t_to_sigma((float)shifted_t_idx);
        std::vector<float> shifted_scaling = denoiser->get_scalings(shifted_sigma);
        float shifted_c_skip               = shifted_scaling[0];
        float shifted_c_out                = shifted_scaling[1];
        float shifted_c_in                 = shifted_scaling[2];

        *c_skip = shifted_c_skip * c_in / shifted_c_in;
        *c_out  = shifted_c_out;
    }

    struct SamplePreviewContext {
        sd_preview_cb_t callback = nullptr;
        void* data               = nullptr;
        preview_t mode           = PREVIEW_NONE;
    };

    SamplePreviewContext prepare_sample_preview_context() {
        return SamplePreviewContext{sd_get_preview_callback(),
                                    sd_get_preview_callback_data(),
                                    sd_get_preview_mode()};
    }

    void report_sample_progress(int step, size_t total_steps, int64_t t0) {
        int64_t t1 = ggml_time_us();
        if (step > 0 || step == -(int)total_steps) {
            int showstep = std::abs(step);
            pretty_progress(showstep, (int)total_steps, (t1 - t0) / 1000000.f / showstep);
        }
    }

    void compute_sample_controls(const sd::Tensor<float>& control_image,
                                 const sd::Tensor<float>& noised_input,
                                 const sd::Tensor<float>& timesteps_tensor,
                                 const SDCondition& condition,
                                 std::vector<sd::Tensor<float>>* controls) {
        GGML_ASSERT(controls != nullptr);
        controls->clear();
        if (control_image.empty() || control_net == nullptr) {
            return;
        }

        auto control_result = control_net->compute(n_threads,
                                                   noised_input,
                                                   control_image,
                                                   timesteps_tensor,
                                                   condition.c_crossattn,
                                                   condition.c_vector);
        if (!control_result.has_value()) {
            LOG_ERROR("controlnet compute failed");
            return;
        }

        *controls = std::move(*control_result);
    }

    sd::Tensor<float> sample(const std::shared_ptr<DiffusionModel>& work_diffusion_model,
                             bool inverse_noise_scaling,
                             const sd::Tensor<float>& init_latent,
                             sd::Tensor<float> noise,
                             const SDCondition& cond,
                             const SDCondition& uncond,
                             const SDCondition& img_cond,
                             const SDCondition& id_cond,
                             const sd::Tensor<float>& control_image,
                             float control_strength,
                             const sd_guidance_params_t& guidance,
                             float eta,
                             int shifted_timestep,
                             sample_method_t method,
                             bool is_flow_denoiser,
                             const std::vector<float>& sigmas,
                             int start_merge_step,
                             const std::vector<sd::Tensor<float>>& ref_latents,
                             bool increase_ref_index,
                             const sd::Tensor<float>& denoise_mask,
                             const sd::Tensor<float>& vace_context,
                             float vace_strength,
                             const sd_cache_params_t* cache_params) {
        std::vector<int> skip_layers(guidance.slg.layers, guidance.slg.layers + guidance.slg.layer_count);
        float cfg_scale     = guidance.txt_cfg;
        float img_cfg_scale = guidance.img_cfg;
        float slg_scale     = guidance.slg.scale;
        float rescale_scale = guidance.rescale_scale;
        float stg_scale     = guidance.stg_scale;
        std::vector<int> stg_blocks(guidance.stg_blocks,
                                    guidance.stg_blocks + guidance.stg_blocks_count);
        bool has_stg        = stg_scale != 0.f && !stg_blocks.empty();

        sd_sample::SampleCacheRuntime cache_runtime = sd_sample::init_sample_cache_runtime(version,
                                                                                           cache_params,
                                                                                           denoiser.get(),
                                                                                           sigmas);
        size_t steps                                = sigmas.size() - 1;
        bool has_skiplayer                          = slg_scale != 0.0f && !skip_layers.empty();
        if (has_skiplayer && !sd_version_is_dit(version)) {
            has_skiplayer = false;
            LOG_WARN("SLG is incompatible with this model type");
        }

        int64_t t0                   = ggml_time_us();
        sd::Tensor<float> x_t        = !noise.empty()
                                           ? denoiser->noise_scaling(sigmas[0], noise, init_latent)
                                           : init_latent;
        sd::Tensor<float> denoised   = x_t;
        SamplePreviewContext preview = prepare_sample_preview_context();

        auto denoise = [&](const sd::Tensor<float>& x, float sigma, int step) -> sd::Tensor<float> {
            if (step == 1 || step == -1) {
                pretty_progress(0, (int)steps, 0);
            }

            std::vector<float> scaling = denoiser->get_scalings(sigma);
            GGML_ASSERT(scaling.size() == 3);
            float c_skip = scaling[0];
            float c_out  = scaling[1];
            float c_in   = scaling[2];

            std::vector<float> timesteps_vec = prepare_sample_timesteps(sigma, shifted_timestep);
            timesteps_vec                    = process_timesteps(timesteps_vec, init_latent, denoise_mask);
            adjust_sample_step_scalings(shifted_timestep, timesteps_vec, c_in, &c_skip, &c_out);

            sd::Tensor<float> timesteps_tensor({static_cast<int64_t>(timesteps_vec.size())}, timesteps_vec);
            sd::Tensor<float> guidance_tensor({1}, std::vector<float>{guidance.distilled_guidance});
            sd::Tensor<float> noised_input = x * c_in;
            if (!denoise_mask.empty() && version == VERSION_WAN2_2_TI2V) {
                noised_input = noised_input * denoise_mask + init_latent * (1.0f - denoise_mask);
            }

            if (cache_runtime.spectrum_enabled && cache_runtime.spectrum.should_predict()) {
                cache_runtime.spectrum.predict(&denoised);
                if (!denoise_mask.empty()) {
                    denoised = denoised * denoise_mask + init_latent * (1.0f - denoise_mask);
                }
                if (sd_should_preview_denoised() && preview.callback != nullptr) {
                    preview_image(step, denoised, version, preview.mode, preview.callback, preview.data, false);
                }
                report_sample_progress(step, steps, t0);
                return denoised;
            }

            if (sd_should_preview_noisy() && preview.callback != nullptr) {
                preview_image(step, noised_input, version, preview.mode, preview.callback, preview.data, true);
            }

            sd::Tensor<float> cond_out;
            sd::Tensor<float> uncond_out;
            sd::Tensor<float> img_cond_out;
            sd::Tensor<float> skip_cond_out;
            sd::Tensor<float> stg_cond_out;
            sd_sample::SampleStepCacheDispatcher step_cache(cache_runtime, step, sigma);
            std::vector<sd::Tensor<float>> controls;
            DiffusionParams diffusion_params;
            diffusion_params.x                  = &noised_input;
            diffusion_params.timesteps          = &timesteps_tensor;
            diffusion_params.guidance           = &guidance_tensor;
            diffusion_params.ref_latents        = &ref_latents;
            diffusion_params.increase_ref_index = increase_ref_index;
            diffusion_params.controls           = &controls;
            diffusion_params.control_strength   = control_strength;
            diffusion_params.vace_context       = vace_context.empty() ? nullptr : &vace_context;
            diffusion_params.vace_strength      = vace_strength;
            diffusion_params.skip_layers        = nullptr;
            diffusion_params.stg_skip_blocks    = nullptr;

            compute_sample_controls(control_image,
                                    noised_input,
                                    timesteps_tensor,
                                    cond,
                                    &controls);

            auto run_condition = [&](const SDCondition& condition,
                                     const sd::Tensor<float>* c_concat_override        = nullptr,
                                     const std::vector<int>* local_skip_layers         = nullptr,
                                     const std::vector<int>* local_stg_skip_blocks     = nullptr) -> sd::Tensor<float> {
                diffusion_params.context         = condition.c_crossattn.empty() ? nullptr : &condition.c_crossattn;
                diffusion_params.c_concat        = c_concat_override != nullptr ? c_concat_override : (condition.c_concat.empty() ? nullptr : &condition.c_concat);
                diffusion_params.y               = condition.c_vector.empty() ? nullptr : &condition.c_vector;
                diffusion_params.t5_ids          = condition.c_t5_ids.empty() ? nullptr : &condition.c_t5_ids;
                diffusion_params.t5_weights      = condition.c_t5_weights.empty() ? nullptr : &condition.c_t5_weights;
                diffusion_params.skip_layers     = local_skip_layers;
                diffusion_params.stg_skip_blocks = local_stg_skip_blocks;

                sd::Tensor<float> cached_output;
                if (step_cache.before_condition(&condition, noised_input, &cached_output)) {
                    return std::move(cached_output);
                }

                auto output_opt = work_diffusion_model->compute(n_threads, diffusion_params);
                if (output_opt.empty()) {
                    LOG_ERROR("diffusion model compute failed");
                    return sd::Tensor<float>();
                }

                step_cache.after_condition(&condition, noised_input, output_opt);
                return output_opt;
            };

            if (start_merge_step == -1 || step <= start_merge_step) {
                cond_out = run_condition(cond);
                if (cond_out.empty()) {
                    return {};
                }
            } else {
                GGML_ASSERT(!id_cond.empty());
                cond_out = run_condition(id_cond,
                                         cond.c_concat.empty() ? nullptr : &cond.c_concat);
                if (cond_out.empty()) {
                    return {};
                }
            }

            if (!uncond.empty()) {
                if (!step_cache.is_step_skipped()) {
                    compute_sample_controls(control_image,
                                            noised_input,
                                            timesteps_tensor,
                                            uncond,
                                            &controls);
                }
                uncond_out = run_condition(uncond);
                if (uncond_out.empty()) {
                    return {};
                }
            }
            if (!img_cond.empty()) {
                img_cond_out = run_condition(img_cond,
                                             cond.c_concat.empty() ? nullptr : &cond.c_concat);
                if (img_cond_out.empty()) {
                    return {};
                }
            }
            bool is_skiplayer_step = has_skiplayer &&
                                     step > (int)(guidance.slg.layer_start * static_cast<int>(sigmas.size())) &&
                                     step < (int)(guidance.slg.layer_end * static_cast<int>(sigmas.size()));
            if (is_skiplayer_step) {
                LOG_DEBUG("Skipping layers at step %d\n", step);
                if (!step_cache.is_step_skipped()) {
                    skip_cond_out = run_condition(cond,
                                                  cond.c_concat.empty() ? nullptr : &cond.c_concat,
                                                  &skip_layers);
                    if (skip_cond_out.empty()) {
                        return {};
                    }
                }
            }

            // STG (Spatio-Temporal Guidance): third forward pass with self-attention
            // skipped on stg_blocks. The "weakened" prediction is mixed into the
            // guided pred:  pred += stg_scale * (cond - perturbed). Reference:
            // ltx_core/components/guiders.py::calculate (perturbed term).
            if (has_stg && !step_cache.is_step_skipped()) {
                stg_cond_out = run_condition(cond,
                                             cond.c_concat.empty() ? nullptr : &cond.c_concat,
                                             /*local_skip_layers=*/nullptr,
                                             &stg_blocks);
                if (stg_cond_out.empty()) {
                    return {};
                }
            }

            GGML_ASSERT(!cond_out.empty());
            sd::Tensor<float> latent_result = cond_out;
            if (!uncond_out.empty()) {
                if (!img_cond_out.empty()) {
                    latent_result = uncond_out +
                                    img_cfg_scale * (img_cond_out - uncond_out) +
                                    cfg_scale * (cond_out - img_cond_out);
                } else {
                    latent_result = uncond_out + cfg_scale * (cond_out - uncond_out);
                }
            } else if (!img_cond_out.empty()) {
                latent_result = img_cond_out + cfg_scale * (cond_out - img_cond_out);
            }

            if (is_skiplayer_step && !skip_cond_out.empty()) {
                latent_result += (cond_out - skip_cond_out) * slg_scale;
            }

            // STG perturbed-pass mixing: pred += stg_scale * (cond - perturbed).
            if (has_stg && !stg_cond_out.empty()) {
                latent_result += (cond_out - stg_cond_out) * stg_scale;
            }

            denoised = latent_result * c_out + x * c_skip;

            // CFG-rescale: pull pred.std() back toward cond.std() to combat oversaturation
            // introduced by CFG amplitude. Reference: ltx_core/components/guiders.py::calculate.
            // Python operates on DENOISED predictions (X0Model returns denoised, then guider
            // does CFG on denoised) — so we must compute std and multiply on the denoised
            // tensor, not on the velocity-domain `latent_result`. Skip when rescale_scale==0
            // (default for non-LTX-2 models) or when only cond_out is present (no CFG mix
            // happened — pred would equal cond_only and rescale would be a no-op).
            if (rescale_scale != 0.f && (!uncond_out.empty() || !img_cond_out.empty())) {
                auto t_std = [](const float* d, int64_t n) -> double {
                    if (n <= 1) return 0.0;
                    double s = 0.0, sq = 0.0;
                    for (int64_t i = 0; i < n; ++i) {
                        double v = static_cast<double>(d[i]);
                        s += v;
                        sq += v * v;
                    }
                    double mean = s / n;
                    double var  = sq / n - mean * mean;
                    return std::sqrt(std::max(0.0, var));
                };
                // Denoised(cond_alone) = c_out * cond_out + c_skip * x. Materialize it just
                // for the std computation; we don't want to apply CFG to it.
                sd::Tensor<float> denoised_cond_only = cond_out * c_out + x * c_skip;
                double cond_std = t_std(denoised_cond_only.data(), denoised_cond_only.numel());
                double pred_std = t_std(denoised.data(), denoised.numel());
                if (pred_std > 1e-12) {
                    double factor = cond_std / pred_std;
                    factor = rescale_scale * factor + (1.0 - rescale_scale);
                    denoised *= static_cast<float>(factor);
                }
            }
            if (cache_runtime.spectrum_enabled) {
                cache_runtime.spectrum.update(denoised);
            }
            if (!denoise_mask.empty()) {
                denoised = denoised * denoise_mask + init_latent * (1.0f - denoise_mask);
            }
            if (sd_should_preview_denoised() && preview.callback != nullptr) {
                preview_image(step, denoised, version, preview.mode, preview.callback, preview.data, false);
            }
            report_sample_progress(step, steps, t0);
            return denoised;
        };

        auto x0_opt = sample_k_diffusion(method, denoise, x_t, sigmas, sampler_rng, eta, is_flow_denoiser);
        if (x0_opt.empty()) {
            LOG_ERROR("Diffusion model sampling failed");
            if (control_net) {
                control_net->free_control_ctx();
                control_net->free_compute_buffer();
            }
            if (work_diffusion_model) {
                work_diffusion_model->free_compute_buffer();
            }
            return {};
        }

        auto x0 = std::move(x0_opt);
        sd_sample::log_sample_cache_summary(cache_runtime, steps);
        if (inverse_noise_scaling) {
            x0 = denoiser->inverse_noise_scaling(sigmas[sigmas.size() - 1], x0);
        }

        if (control_net) {
            control_net->free_control_ctx();
            control_net->free_compute_buffer();
        }
        if (work_diffusion_model) {
            work_diffusion_model->free_compute_buffer();
        }
        return x0;
    }

    int get_vae_scale_factor() {
        return first_stage_model->get_scale_factor();
    }

    int get_diffusion_model_down_factor() {
        int down_factor = 8;  // unet
        if (sd_version_is_dit(version)) {
            if (sd_version_is_wan(version)) {
                down_factor = 2;
            } else {
                down_factor = 1;
            }
        }
        return down_factor;
    }

    int get_latent_channel() {
        int latent_channel = 4;
        if (sd_version_is_dit(version)) {
            if (version == VERSION_WAN2_2_TI2V) {
                latent_channel = 48;
            } else if (version == VERSION_CHROMA_RADIANCE) {
                latent_channel = 3;
            } else if (sd_version_uses_flux2_vae(version)) {
                latent_channel = 128;
            } else if (sd_version_is_ltx2(version)) {
                // LTX-2 VAE latent dim (matches DiT patchify_proj in_channels).
                latent_channel = 128;
            } else {
                latent_channel = 16;
            }
        }
        return latent_channel;
    }

    int get_image_seq_len(int h, int w, int frames = 1) {
        int vae_scale_factor = get_vae_scale_factor();
        int spatial_tokens   = (h / vae_scale_factor) * (w / vae_scale_factor);
        // For video flow-match schedulers (LTX-2, Wan), `tokens` in the shift
        // formula is math.prod(latent.shape[2:]) = T_latent * H_latent * W_latent.
        // Earlier we only passed the spatial count (H*W), which under-shifted
        // the LTX-2 schedule because the 22B run has 25-frame inputs →
        // T_latent = 4, so the real token count is 4× the spatial count.
        // Python reference: ltx_core/components/schedulers.py::LTX2Scheduler.execute.
        if (frames > 1 && sd_version_is_ltx2(version)) {
            int T_latent = ((frames - 1) / 8) + 1;  // LTX-2 VAE: 8× temporal compression.
            return spatial_tokens * T_latent;
        }
        if (frames > 1 && sd_version_is_wan(version)) {
            int T_latent = ((frames - 1) / 4) + 1;  // Wan VAE: 4× temporal compression.
            return spatial_tokens * T_latent;
        }
        return spatial_tokens;
    }

    sd::Tensor<float> generate_init_latent(int width,
                                           int height,
                                           int frames = 1,
                                           bool video = false) {
        int vae_scale_factor = get_vae_scale_factor();
        int W                = width / vae_scale_factor;
        int H                = height / vae_scale_factor;
        int T                = frames;
        if (sd_version_is_wan(version)) {
            T = ((T - 1) / 4) + 1;
        } else if (sd_version_is_ltx2(version)) {
            // LTX-2 VAE: 8× temporal compression.
            T = ((T - 1) / 8) + 1;
        }
        int C = get_latent_channel();
        if (video) {
            return sd::zeros<float>({W, H, T, C, 1});
        }
        return sd::zeros<float>({W, H, C, 1});
    }

    sd::Tensor<float> encode_to_vae_latents(const sd::Tensor<float>& x) {
        auto latents = first_stage_model->encode(n_threads, x, vae_tiling_params, circular_x, circular_y);
        if (latents.empty()) {
            return {};
        }
        latents = first_stage_model->vae_output_to_latents(latents, rng);
        return latents;
    }

    sd::Tensor<float> encode_first_stage(const sd::Tensor<float>& x) {
        auto latents = encode_to_vae_latents(x);
        if (latents.empty()) {
            return {};
        }
        if (version != VERSION_SD1_PIX2PIX) {
            latents = first_stage_model->vae_to_diffusion_latents(latents);
        }
        return latents;
    }

    sd::Tensor<float> decode_first_stage(const sd::Tensor<float>& x, bool decode_video = false) {
        auto latents = first_stage_model->diffusion_to_vae_latents(x);
        return first_stage_model->decode(n_threads, latents, vae_tiling_params, decode_video, circular_x, circular_y);
    }

    void set_flow_shift(float flow_shift = INFINITY) {
        auto flow_denoiser = std::dynamic_pointer_cast<DiscreteFlowDenoiser>(denoiser);
        if (flow_denoiser) {
            if (flow_shift == INFINITY) {
                flow_shift = default_flow_shift;
            }
            flow_denoiser->set_shift(flow_shift);
        }
    }

    bool is_flow_denoiser() {
        auto flow_denoiser = std::dynamic_pointer_cast<DiscreteFlowDenoiser>(denoiser);
        return !!flow_denoiser;
    }
};

/*================================================= SD API ==================================================*/

#define NONE_STR "NONE"

const char* sd_type_name(enum sd_type_t type) {
    if ((int)type < std::min<int>(SD_TYPE_COUNT, GGML_TYPE_COUNT)) {
        return ggml_type_name((ggml_type)type);
    }
    return NONE_STR;
}

enum sd_type_t str_to_sd_type(const char* str) {
    for (int i = 0; i < std::min<int>(SD_TYPE_COUNT, GGML_TYPE_COUNT); i++) {
        auto trait = ggml_get_type_traits((ggml_type)i);
        if (!strcmp(str, trait->type_name)) {
            return (enum sd_type_t)i;
        }
    }
    return SD_TYPE_COUNT;
}

const char* rng_type_to_str[] = {
    "std_default",
    "cuda",
    "cpu",
};

const char* sd_rng_type_name(enum rng_type_t rng_type) {
    if (rng_type < RNG_TYPE_COUNT) {
        return rng_type_to_str[rng_type];
    }
    return NONE_STR;
}

enum rng_type_t str_to_rng_type(const char* str) {
    for (int i = 0; i < RNG_TYPE_COUNT; i++) {
        if (!strcmp(str, rng_type_to_str[i])) {
            return (enum rng_type_t)i;
        }
    }
    return RNG_TYPE_COUNT;
}

const char* sample_method_to_str[] = {
    "euler",
    "euler_a",
    "heun",
    "dpm2",
    "dpm++2s_a",
    "dpm++2m",
    "dpm++2mv2",
    "ipndm",
    "ipndm_v",
    "lcm",
    "ddim_trailing",
    "tcd",
    "res_multistep",
    "res_2s",
    "er_sde",
};

const char* sd_sample_method_name(enum sample_method_t sample_method) {
    if (sample_method < SAMPLE_METHOD_COUNT) {
        return sample_method_to_str[sample_method];
    }
    return NONE_STR;
}

enum sample_method_t str_to_sample_method(const char* str) {
    for (int i = 0; i < SAMPLE_METHOD_COUNT; i++) {
        if (!strcmp(str, sample_method_to_str[i])) {
            return (enum sample_method_t)i;
        }
    }
    return SAMPLE_METHOD_COUNT;
}

const char* scheduler_to_str[] = {
    "discrete",
    "karras",
    "exponential",
    "ays",
    "gits",
    "sgm_uniform",
    "simple",
    "smoothstep",
    "kl_optimal",
    "lcm",
    "bong_tangent",
};

const char* sd_scheduler_name(enum scheduler_t scheduler) {
    if (scheduler < SCHEDULER_COUNT) {
        return scheduler_to_str[scheduler];
    }
    return NONE_STR;
}

enum scheduler_t str_to_scheduler(const char* str) {
    for (int i = 0; i < SCHEDULER_COUNT; i++) {
        if (!strcmp(str, scheduler_to_str[i])) {
            return (enum scheduler_t)i;
        }
    }
    return SCHEDULER_COUNT;
}

const char* prediction_to_str[] = {
    "eps",
    "v",
    "edm_v",
    "sd3_flow",
    "flux_flow",
    "flux2_flow",
    "ltx2_flow",
};

const char* sd_prediction_name(enum prediction_t prediction) {
    if (prediction < PREDICTION_COUNT) {
        return prediction_to_str[prediction];
    }
    return NONE_STR;
}

enum prediction_t str_to_prediction(const char* str) {
    for (int i = 0; i < PREDICTION_COUNT; i++) {
        if (!strcmp(str, prediction_to_str[i])) {
            return (enum prediction_t)i;
        }
    }
    return PREDICTION_COUNT;
}

const char* preview_to_str[] = {
    "none",
    "proj",
    "tae",
    "vae",
};

const char* sd_preview_name(enum preview_t preview) {
    if (preview < PREVIEW_COUNT) {
        return preview_to_str[preview];
    }
    return NONE_STR;
}

enum preview_t str_to_preview(const char* str) {
    for (int i = 0; i < PREVIEW_COUNT; i++) {
        if (!strcmp(str, preview_to_str[i])) {
            return (enum preview_t)i;
        }
    }
    return PREVIEW_COUNT;
}

const char* lora_apply_mode_to_str[] = {
    "auto",
    "immediately",
    "at_runtime",
};

const char* sd_lora_apply_mode_name(enum lora_apply_mode_t mode) {
    if (mode < LORA_APPLY_MODE_COUNT) {
        return lora_apply_mode_to_str[mode];
    }
    return NONE_STR;
}

enum lora_apply_mode_t str_to_lora_apply_mode(const char* str) {
    for (int i = 0; i < LORA_APPLY_MODE_COUNT; i++) {
        if (!strcmp(str, lora_apply_mode_to_str[i])) {
            return (enum lora_apply_mode_t)i;
        }
    }
    return LORA_APPLY_MODE_COUNT;
}

void sd_cache_params_init(sd_cache_params_t* cache_params) {
    *cache_params                             = {};
    cache_params->mode                        = SD_CACHE_DISABLED;
    cache_params->reuse_threshold             = INFINITY;
    cache_params->start_percent               = 0.15f;
    cache_params->end_percent                 = 0.95f;
    cache_params->error_decay_rate            = 1.0f;
    cache_params->use_relative_threshold      = true;
    cache_params->reset_error_on_compute      = true;
    cache_params->Fn_compute_blocks           = 8;
    cache_params->Bn_compute_blocks           = 0;
    cache_params->residual_diff_threshold     = 0.08f;
    cache_params->max_warmup_steps            = 8;
    cache_params->max_cached_steps            = -1;
    cache_params->max_continuous_cached_steps = -1;
    cache_params->taylorseer_n_derivatives    = 1;
    cache_params->taylorseer_skip_interval    = 1;
    cache_params->scm_mask                    = nullptr;
    cache_params->scm_policy_dynamic          = true;
    cache_params->spectrum_w                  = 0.40f;
    cache_params->spectrum_m                  = 3;
    cache_params->spectrum_lam                = 1.0f;
    cache_params->spectrum_window_size        = 2;
    cache_params->spectrum_flex_window        = 0.50f;
    cache_params->spectrum_warmup_steps       = 4;
    cache_params->spectrum_stop_percent       = 0.9f;
}

void sd_ctx_params_init(sd_ctx_params_t* sd_ctx_params) {
    *sd_ctx_params                         = {};
    sd_ctx_params->vae_decode_only         = true;
    sd_ctx_params->free_params_immediately = true;
    sd_ctx_params->n_threads               = sd_get_num_physical_cores();
    sd_ctx_params->wtype                   = SD_TYPE_COUNT;
    sd_ctx_params->rng_type                = CUDA_RNG;
    sd_ctx_params->sampler_rng_type        = RNG_TYPE_COUNT;
    sd_ctx_params->prediction              = PREDICTION_COUNT;
    sd_ctx_params->lora_apply_mode         = LORA_APPLY_AUTO;
    sd_ctx_params->offload_params_to_cpu   = false;
    sd_ctx_params->enable_mmap             = false;
    sd_ctx_params->keep_clip_on_cpu        = false;
    sd_ctx_params->keep_control_net_on_cpu = false;
    sd_ctx_params->keep_vae_on_cpu         = false;
    sd_ctx_params->diffusion_flash_attn    = false;
    sd_ctx_params->circular_x              = false;
    sd_ctx_params->circular_y              = false;
    sd_ctx_params->chroma_use_dit_mask     = true;
    sd_ctx_params->chroma_use_t5_mask      = false;
    sd_ctx_params->chroma_t5_mask_pad      = 1;

    sd_ctx_params->auto_fit                           = true;
    sd_ctx_params->auto_fit_target_mb                 = 512;
    sd_ctx_params->auto_fit_dry_run                   = false;
    sd_ctx_params->auto_fit_compute_reserve_dit_mb    = 0;
    sd_ctx_params->auto_fit_compute_reserve_vae_mb    = 0;
    sd_ctx_params->auto_fit_compute_reserve_cond_mb   = 0;
    sd_ctx_params->lazy_load_dit                      = true;
    sd_ctx_params->lazy_load_cond                     = true;
    sd_ctx_params->auto_tensor_split                  = true;
}

char* sd_ctx_params_to_str(const sd_ctx_params_t* sd_ctx_params) {
    char* buf = (char*)malloc(4096);
    if (!buf)
        return nullptr;
    buf[0] = '\0';

    snprintf(buf + strlen(buf), 4096 - strlen(buf),
             "model_path: %s\n"
             "clip_l_path: %s\n"
             "clip_g_path: %s\n"
             "clip_vision_path: %s\n"
             "t5xxl_path: %s\n"
             "llm_path: %s\n"
             "llm_vision_path: %s\n"
             "gemma_tokenizer_path: %s\n"
             "diffusion_model_path: %s\n"
             "high_noise_diffusion_model_path: %s\n"
             "vae_path: %s\n"
             "taesd_path: %s\n"
             "control_net_path: %s\n"
             "photo_maker_path: %s\n"
             "tensor_type_rules: %s\n"
             "vae_decode_only: %s\n"
             "free_params_immediately: %s\n"
             "n_threads: %d\n"
             "wtype: %s\n"
             "rng_type: %s\n"
             "sampler_rng_type: %s\n"
             "prediction: %s\n"
             "offload_params_to_cpu: %s\n"
             "keep_clip_on_cpu: %s\n"
             "keep_control_net_on_cpu: %s\n"
             "keep_vae_on_cpu: %s\n"
             "flash_attn: %s\n"
             "diffusion_flash_attn: %s\n"
             "circular_x: %s\n"
             "circular_y: %s\n"
             "chroma_use_dit_mask: %s\n"
             "chroma_use_t5_mask: %s\n"
             "chroma_t5_mask_pad: %d\n",
             SAFE_STR(sd_ctx_params->model_path),
             SAFE_STR(sd_ctx_params->clip_l_path),
             SAFE_STR(sd_ctx_params->clip_g_path),
             SAFE_STR(sd_ctx_params->clip_vision_path),
             SAFE_STR(sd_ctx_params->t5xxl_path),
             SAFE_STR(sd_ctx_params->llm_path),
             SAFE_STR(sd_ctx_params->llm_vision_path),
             SAFE_STR(sd_ctx_params->gemma_tokenizer_path),
             SAFE_STR(sd_ctx_params->diffusion_model_path),
             SAFE_STR(sd_ctx_params->high_noise_diffusion_model_path),
             SAFE_STR(sd_ctx_params->vae_path),
             SAFE_STR(sd_ctx_params->taesd_path),
             SAFE_STR(sd_ctx_params->control_net_path),
             SAFE_STR(sd_ctx_params->photo_maker_path),
             SAFE_STR(sd_ctx_params->tensor_type_rules),
             BOOL_STR(sd_ctx_params->vae_decode_only),
             BOOL_STR(sd_ctx_params->free_params_immediately),
             sd_ctx_params->n_threads,
             sd_type_name(sd_ctx_params->wtype),
             sd_rng_type_name(sd_ctx_params->rng_type),
             sd_rng_type_name(sd_ctx_params->sampler_rng_type),
             sd_prediction_name(sd_ctx_params->prediction),
             BOOL_STR(sd_ctx_params->offload_params_to_cpu),
             BOOL_STR(sd_ctx_params->keep_clip_on_cpu),
             BOOL_STR(sd_ctx_params->keep_control_net_on_cpu),
             BOOL_STR(sd_ctx_params->keep_vae_on_cpu),
             BOOL_STR(sd_ctx_params->flash_attn),
             BOOL_STR(sd_ctx_params->diffusion_flash_attn),
             BOOL_STR(sd_ctx_params->circular_x),
             BOOL_STR(sd_ctx_params->circular_y),
             BOOL_STR(sd_ctx_params->chroma_use_dit_mask),
             BOOL_STR(sd_ctx_params->chroma_use_t5_mask),
             sd_ctx_params->chroma_t5_mask_pad);

    return buf;
}

void sd_sample_params_init(sd_sample_params_t* sample_params) {
    *sample_params                             = {};
    sample_params->guidance.txt_cfg            = 7.0f;
    sample_params->guidance.img_cfg            = INFINITY;
    sample_params->guidance.distilled_guidance = 3.5f;
    sample_params->guidance.rescale_scale      = 0.f;  // LTX-2.3 expects 0.7
    sample_params->guidance.stg_scale          = 0.f;  // LTX-2.3 expects 1.0 with stg_blocks=[28]
    sample_params->guidance.stg_blocks         = nullptr;
    sample_params->guidance.stg_blocks_count   = 0;
    sample_params->guidance.slg.layer_count    = 0;
    sample_params->guidance.slg.layer_start    = 0.01f;
    sample_params->guidance.slg.layer_end      = 0.2f;
    sample_params->guidance.slg.scale          = 0.f;
    sample_params->scheduler                   = SCHEDULER_COUNT;
    sample_params->sample_method               = SAMPLE_METHOD_COUNT;
    sample_params->sample_steps                = 20;
    sample_params->eta                         = INFINITY;
    sample_params->custom_sigmas               = nullptr;
    sample_params->custom_sigmas_count         = 0;
    sample_params->flow_shift                  = INFINITY;
}

char* sd_sample_params_to_str(const sd_sample_params_t* sample_params) {
    char* buf = (char*)malloc(4096);
    if (!buf)
        return nullptr;
    buf[0] = '\0';

    snprintf(buf + strlen(buf), 4096 - strlen(buf),
             "(txt_cfg: %.2f, "
             "img_cfg: %.2f, "
             "distilled_guidance: %.2f, "
             "slg.layer_count: %zu, "
             "slg.layer_start: %.2f, "
             "slg.layer_end: %.2f, "
             "slg.scale: %.2f, "
             "scheduler: %s, "
             "sample_method: %s, "
             "sample_steps: %d, "
             "eta: %.2f, "
             "shifted_timestep: %d, "
             "flow_shift: %.2f)",
             sample_params->guidance.txt_cfg,
             std::isfinite(sample_params->guidance.img_cfg)
                 ? sample_params->guidance.img_cfg
                 : sample_params->guidance.txt_cfg,
             sample_params->guidance.distilled_guidance,
             sample_params->guidance.slg.layer_count,
             sample_params->guidance.slg.layer_start,
             sample_params->guidance.slg.layer_end,
             sample_params->guidance.slg.scale,
             sd_scheduler_name(sample_params->scheduler),
             sd_sample_method_name(sample_params->sample_method),
             sample_params->sample_steps,
             sample_params->eta,
             sample_params->shifted_timestep,
             sample_params->flow_shift);

    return buf;
}

void sd_img_gen_params_init(sd_img_gen_params_t* sd_img_gen_params) {
    *sd_img_gen_params = {};
    sd_sample_params_init(&sd_img_gen_params->sample_params);
    sd_img_gen_params->clip_skip         = -1;
    sd_img_gen_params->ref_images_count  = 0;
    sd_img_gen_params->width             = 512;
    sd_img_gen_params->height            = 512;
    sd_img_gen_params->strength          = 0.75f;
    sd_img_gen_params->seed              = -1;
    sd_img_gen_params->batch_count       = 1;
    sd_img_gen_params->control_strength  = 0.9f;
    sd_img_gen_params->pm_params         = {nullptr, 0, nullptr, 20.f};
    sd_img_gen_params->vae_tiling_params = {false, 0, 0, 0.5f, 0.0f, 0.0f};
    sd_cache_params_init(&sd_img_gen_params->cache);
}

char* sd_img_gen_params_to_str(const sd_img_gen_params_t* sd_img_gen_params) {
    char* buf = (char*)malloc(4096);
    if (!buf)
        return nullptr;
    buf[0] = '\0';

    char* sample_params_str = sd_sample_params_to_str(&sd_img_gen_params->sample_params);

    snprintf(buf + strlen(buf), 4096 - strlen(buf),
             "prompt: %s\n"
             "negative_prompt: %s\n"
             "clip_skip: %d\n"
             "width: %d\n"
             "height: %d\n"
             "sample_params: %s\n"
             "strength: %.2f\n"
             "seed: %" PRId64
             "\n"
             "batch_count: %d\n"
             "ref_images_count: %d\n"
             "auto_resize_ref_image: %s\n"
             "increase_ref_index: %s\n"
             "control_strength: %.2f\n"
             "photo maker: {style_strength = %.2f, id_images_count = %d, id_embed_path = %s}\n"
             "VAE tiling: %s\n",
             SAFE_STR(sd_img_gen_params->prompt),
             SAFE_STR(sd_img_gen_params->negative_prompt),
             sd_img_gen_params->clip_skip,
             sd_img_gen_params->width,
             sd_img_gen_params->height,
             SAFE_STR(sample_params_str),
             sd_img_gen_params->strength,
             sd_img_gen_params->seed,
             sd_img_gen_params->batch_count,
             sd_img_gen_params->ref_images_count,
             BOOL_STR(sd_img_gen_params->auto_resize_ref_image),
             BOOL_STR(sd_img_gen_params->increase_ref_index),
             sd_img_gen_params->control_strength,
             sd_img_gen_params->pm_params.style_strength,
             sd_img_gen_params->pm_params.id_images_count,
             SAFE_STR(sd_img_gen_params->pm_params.id_embed_path),
             BOOL_STR(sd_img_gen_params->vae_tiling_params.enabled));
    const char* cache_mode_str = "disabled";
    if (sd_img_gen_params->cache.mode == SD_CACHE_EASYCACHE) {
        cache_mode_str = "easycache";
    } else if (sd_img_gen_params->cache.mode == SD_CACHE_UCACHE) {
        cache_mode_str = "ucache";
    }
    snprintf(buf + strlen(buf), 4096 - strlen(buf),
             "cache: %s (threshold=%.3f, start=%.2f, end=%.2f)\n",
             cache_mode_str,
             get_cache_reuse_threshold(sd_img_gen_params->cache),
             sd_img_gen_params->cache.start_percent,
             sd_img_gen_params->cache.end_percent);
    free(sample_params_str);
    return buf;
}

void sd_vid_gen_params_init(sd_vid_gen_params_t* sd_vid_gen_params) {
    *sd_vid_gen_params = {};
    sd_sample_params_init(&sd_vid_gen_params->sample_params);
    sd_sample_params_init(&sd_vid_gen_params->high_noise_sample_params);
    sd_vid_gen_params->high_noise_sample_params.sample_steps = -1;
    sd_vid_gen_params->width                                 = 512;
    sd_vid_gen_params->height                                = 512;
    sd_vid_gen_params->strength                              = 0.75f;
    sd_vid_gen_params->seed                                  = -1;
    sd_vid_gen_params->video_frames                          = 6;
    sd_vid_gen_params->moe_boundary                          = 0.875f;
    sd_vid_gen_params->vace_strength                         = 1.f;
    sd_vid_gen_params->fps                                   = 24.f;
    sd_vid_gen_params->vae_tiling_params                     = {false, 0, 0, 0.5f, 0.0f, 0.0f};
    sd_cache_params_init(&sd_vid_gen_params->cache);
}

struct sd_ctx_t {
    StableDiffusionGGML* sd = nullptr;
};

static bool sd_version_supports_video_generation(SDVersion version) {
    return version == VERSION_SVD || sd_version_is_wan(version);
}

static bool sd_version_supports_image_generation(SDVersion version) {
    return !sd_version_supports_video_generation(version);
}

sd_ctx_t* new_sd_ctx(const sd_ctx_params_t* sd_ctx_params) {
    sd_ctx_t* sd_ctx = (sd_ctx_t*)malloc(sizeof(sd_ctx_t));
    if (sd_ctx == nullptr) {
        return nullptr;
    }

    sd_ctx->sd = new StableDiffusionGGML();
    if (sd_ctx->sd == nullptr) {
        free(sd_ctx);
        return nullptr;
    }

    if (!sd_ctx->sd->init(sd_ctx_params)) {
        delete sd_ctx->sd;
        sd_ctx->sd = nullptr;
        free(sd_ctx);
        return nullptr;
    }
    return sd_ctx;
}

void free_sd_ctx(sd_ctx_t* sd_ctx) {
    if (sd_ctx->sd != nullptr) {
        delete sd_ctx->sd;
        sd_ctx->sd = nullptr;
    }
    free(sd_ctx);
}

SD_API bool sd_ctx_supports_image_generation(const sd_ctx_t* sd_ctx) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr) {
        return false;
    }
    return sd_version_supports_image_generation(sd_ctx->sd->version);
}

SD_API bool sd_ctx_supports_video_generation(const sd_ctx_t* sd_ctx) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr) {
        return false;
    }
    return sd_version_supports_video_generation(sd_ctx->sd->version);
}

enum sample_method_t sd_get_default_sample_method(const sd_ctx_t* sd_ctx) {
    if (sd_ctx != nullptr && sd_ctx->sd != nullptr) {
        if (sd_version_is_dit(sd_ctx->sd->version)) {
            return EULER_SAMPLE_METHOD;
        }
    }
    return EULER_A_SAMPLE_METHOD;
}

enum scheduler_t sd_get_default_scheduler(const sd_ctx_t* sd_ctx, enum sample_method_t sample_method) {
    if (sd_ctx != nullptr && sd_ctx->sd != nullptr) {
        auto edm_v_denoiser = std::dynamic_pointer_cast<EDMVDenoiser>(sd_ctx->sd->denoiser);
        if (edm_v_denoiser) {
            return EXPONENTIAL_SCHEDULER;
        }
    }
    if (sample_method == LCM_SAMPLE_METHOD || sample_method == TCD_SAMPLE_METHOD) {
        return LCM_SCHEDULER;
    } else if (sample_method == DDIM_TRAILING_SAMPLE_METHOD) {
        return SIMPLE_SCHEDULER;
    }
    return DISCRETE_SCHEDULER;
}

static int64_t resolve_seed(int64_t seed) {
    if (seed >= 0) {
        return seed;
    }
    srand((int)time(nullptr));
    return rand();
}

static enum sample_method_t resolve_sample_method(sd_ctx_t* sd_ctx, enum sample_method_t sample_method) {
    if (sample_method == SAMPLE_METHOD_COUNT) {
        return sd_get_default_sample_method(sd_ctx);
    }
    return sample_method;
}

static scheduler_t resolve_scheduler(sd_ctx_t* sd_ctx,
                                     scheduler_t scheduler,
                                     enum sample_method_t sample_method) {
    if (scheduler == SCHEDULER_COUNT) {
        return sd_get_default_scheduler(sd_ctx, sample_method);
    }
    return scheduler;
}

static float resolve_eta(sd_ctx_t* sd_ctx,
                         float eta,
                         enum sample_method_t sample_method) {
    if (eta == INFINITY) {
        switch (sample_method) {
            case DDIM_TRAILING_SAMPLE_METHOD:
            case TCD_SAMPLE_METHOD:
            case RES_MULTISTEP_SAMPLE_METHOD:
            case RES_2S_SAMPLE_METHOD:
                return 0.0f;
            case EULER_A_SAMPLE_METHOD:
            case DPMPP2S_A_SAMPLE_METHOD:
            case ER_SDE_SAMPLE_METHOD:
                return 1.0f;
            default:;
        }
        return 0.0f;
    }
    return eta;
}

// Mirrors the Python LTX-2 reference's DEFAULT_NEGATIVE_PROMPT
// (ltx-pipelines/utils/constants.py:135). Used when the caller does not pass
// a negative prompt for an LTX-2 video gen — empty negative + CFG ≥ 5 was
// observed to over-push attention into broken/dark scenes for some seeds.
static const char* LTX2_DEFAULT_NEGATIVE_PROMPT =
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
    "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
    "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
    "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of "
    "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent "
    "lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
    "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
    "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, "
    "off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
    "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
    "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts.";

struct GenerationRequest {
    std::string prompt;
    std::string negative_prompt;
    int width                                = -1;
    int height                               = -1;
    int clip_skip                            = -1;
    int vae_scale_factor                     = -1;
    int diffusion_model_down_factor          = -1;
    int64_t seed                             = -1;
    bool use_uncond                          = false;
    bool use_img_cond                        = false;
    bool use_high_noise_uncond               = false;
    bool use_high_noise_img_cond             = false;
    const sd_cache_params_t* cache_params    = nullptr;
    int batch_count                          = 1;
    int shifted_timestep                     = 0;
    float strength                           = 1.f;
    float control_strength                   = 0.f;
    float eta                                = 0.f;
    bool increase_ref_index                  = false;
    bool auto_resize_ref_image               = false;
    sd_guidance_params_t guidance            = {};
    sd_guidance_params_t high_noise_guidance = {};
    sd_pm_params_t pm_params                 = {};
    int frames                               = -1;
    float fps                                = 0.f;  // 0 = keep diffusion model's default
    float vace_strength                      = 1.f;

    GenerationRequest(sd_ctx_t* sd_ctx, const sd_img_gen_params_t* sd_img_gen_params) {
        prompt                      = SAFE_STR(sd_img_gen_params->prompt);
        negative_prompt             = SAFE_STR(sd_img_gen_params->negative_prompt);
        width                       = sd_img_gen_params->width;
        height                      = sd_img_gen_params->height;
        vae_scale_factor            = sd_ctx->sd->get_vae_scale_factor();
        diffusion_model_down_factor = sd_ctx->sd->get_diffusion_model_down_factor();
        seed                        = sd_img_gen_params->seed;
        batch_count                 = sd_img_gen_params->batch_count;
        clip_skip                   = sd_img_gen_params->clip_skip;
        shifted_timestep            = sd_img_gen_params->sample_params.shifted_timestep;
        strength                    = sd_img_gen_params->strength;
        control_strength            = sd_img_gen_params->control_strength;
        eta                         = sd_img_gen_params->sample_params.eta;
        increase_ref_index          = sd_img_gen_params->increase_ref_index;
        auto_resize_ref_image       = sd_img_gen_params->auto_resize_ref_image;
        guidance                    = sd_img_gen_params->sample_params.guidance;
        pm_params                   = sd_img_gen_params->pm_params;
        cache_params                = &sd_img_gen_params->cache;
        resolve(sd_ctx);
    }

    GenerationRequest(sd_ctx_t* sd_ctx, const sd_vid_gen_params_t* sd_vid_gen_params) {
        prompt                      = SAFE_STR(sd_vid_gen_params->prompt);
        negative_prompt             = SAFE_STR(sd_vid_gen_params->negative_prompt);
        const SDVersion version     = sd_ctx->sd->version;
        const bool is_ltx2          = sd_version_is_ltx2(version);
        // LTX-2: default to the curated negative prompt from the Python
        // reference (ltx-pipelines/utils/constants.py:135) when the caller
        // didn't supply one. Empty negative + CFG ≥ 5 over-pushes attention
        // and produces dark/distorted scenes for some seeds.
        if (is_ltx2 && negative_prompt.empty()) {
            negative_prompt = LTX2_DEFAULT_NEGATIVE_PROMPT;
            LOG_INFO("LTX-2: using default negative prompt (caller passed empty). "
                     "Pass --negative-prompt to override.");
        }
        width                       = sd_vid_gen_params->width;
        height                      = sd_vid_gen_params->height;
        // LTX-2's VAE has 8× temporal compression, so the output frame count
        // must satisfy (frames - 1) %% 8 == 0; other video models (Wan etc.)
        // use 4× compression. Snap DOWN to the nearest valid value.
        const int frame_stride      = is_ltx2 ? 8 : 4;
        const int requested_frames  = sd_vid_gen_params->video_frames;
        frames                      = (requested_frames - 1) / frame_stride * frame_stride + 1;
        if (frames != requested_frames) {
            LOG_WARN("%s: requested %d frames is not (N - 1) %% %d == 0; snapping to %d",
                     is_ltx2 ? "LTX-2" : "video", requested_frames, frame_stride, frames);
        }
        fps                         = sd_vid_gen_params->fps;
        clip_skip                   = sd_vid_gen_params->clip_skip;
        vae_scale_factor            = sd_ctx->sd->get_vae_scale_factor();
        diffusion_model_down_factor = sd_ctx->sd->get_diffusion_model_down_factor();
        seed                        = sd_vid_gen_params->seed;
        cache_params                = &sd_vid_gen_params->cache;
        vace_strength               = sd_vid_gen_params->vace_strength;
        guidance                    = sd_vid_gen_params->sample_params.guidance;
        high_noise_guidance         = sd_vid_gen_params->high_noise_sample_params.guidance;
        resolve(sd_ctx);
    }

    void align_generation_request_size() {
        int spatial_multiple = vae_scale_factor * diffusion_model_down_factor;
        int width_offset     = align_up_offset(width, spatial_multiple);
        int height_offset    = align_up_offset(height, spatial_multiple);
        if (width_offset <= 0 && height_offset <= 0) {
            return;
        }

        int original_width  = width;
        int original_height = height;

        width += width_offset;
        height += height_offset;
        LOG_WARN("align up %dx%d to %dx%d (multiple=%d)",
                 original_width,
                 original_height,
                 width,
                 height,
                 spatial_multiple);
    }

    static void resolve_guidance(sd_ctx_t* sd_ctx,
                                 sd_guidance_params_t* guidance,
                                 bool* use_uncond,
                                 bool* use_img_cond,
                                 const char* stage_name = nullptr) {
        GGML_ASSERT(guidance != nullptr);
        GGML_ASSERT(use_uncond != nullptr);
        GGML_ASSERT(use_img_cond != nullptr);
        // out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)
        // img_cfg == txt_cfg means that img_cfg is not used
        if (!std::isfinite(guidance->img_cfg)) {
            guidance->img_cfg = guidance->txt_cfg;
        }

        if (!sd_version_is_inpaint_or_unet_edit(sd_ctx->sd->version)) {
            guidance->img_cfg = guidance->txt_cfg;
        }

        if (guidance->txt_cfg != 1.f) {
            *use_uncond = true;
        }

        if (guidance->img_cfg != guidance->txt_cfg) {
            *use_img_cond = true;
            *use_uncond   = true;
        }

        if (guidance->txt_cfg < 1.f) {
            const char* prefix = stage_name == nullptr ? "" : stage_name;
            if (guidance->txt_cfg == 0.f) {
                LOG_WARN("%sunconditioned mode, images won't follow the prompt (use cfg-scale=1 for distilled models)",
                         prefix);
            } else {
                LOG_WARN("%scfg value out of expected range may produce unexpected results", prefix);
            }
        }
    }

    void resolve(sd_ctx_t* sd_ctx) {
        align_generation_request_size();
        seed = resolve_seed(seed);

        resolve_guidance(sd_ctx, &guidance, &use_uncond, &use_img_cond);
        if (sd_ctx->sd->high_noise_diffusion_model) {
            resolve_guidance(sd_ctx,
                             &high_noise_guidance,
                             &use_high_noise_uncond,
                             &use_high_noise_img_cond,
                             "high noise: ");
        }

        if (shifted_timestep > 0 && !sd_version_is_sdxl(sd_ctx->sd->version)) {
            LOG_WARN("timestep shifting is only supported for SDXL models!");
            shifted_timestep = 0;
        }
    }
};

struct SamplePlan {
    enum sample_method_t sample_method            = SAMPLE_METHOD_COUNT;
    enum sample_method_t high_noise_sample_method = SAMPLE_METHOD_COUNT;
    float eta                                     = 0.f;
    float high_noise_eta                          = 0.f;
    int sample_steps                              = 0;
    int high_noise_sample_steps                   = 0;
    int total_steps                               = 0;
    float moe_boundary                            = 0.f;
    int start_merge_step                          = -1;
    std::vector<float> sigmas;

    SamplePlan(sd_ctx_t* sd_ctx,
               const sd_img_gen_params_t* sd_img_gen_params,
               const GenerationRequest& request) {
        sample_method = sd_img_gen_params->sample_params.sample_method;
        eta           = sd_img_gen_params->sample_params.eta;
        sample_steps  = sd_img_gen_params->sample_params.sample_steps;
        resolve(sd_ctx, &request, &sd_img_gen_params->sample_params);
    }

    SamplePlan(sd_ctx_t* sd_ctx,
               const sd_vid_gen_params_t* sd_vid_gen_params,
               const GenerationRequest& request) {
        sample_method = sd_vid_gen_params->sample_params.sample_method;
        eta           = sd_vid_gen_params->sample_params.eta;
        sample_steps  = sd_vid_gen_params->sample_params.sample_steps;
        if (sd_ctx->sd->high_noise_diffusion_model) {
            high_noise_sample_steps  = sd_vid_gen_params->high_noise_sample_params.sample_steps;
            high_noise_sample_method = sd_vid_gen_params->high_noise_sample_params.sample_method;
            high_noise_eta           = sd_vid_gen_params->high_noise_sample_params.eta;
        }
        moe_boundary = sd_vid_gen_params->moe_boundary;
        resolve(sd_ctx, &request, &sd_vid_gen_params->sample_params);
    }

    void resolve(sd_ctx_t* sd_ctx,
                 const GenerationRequest* request,
                 const sd_sample_params_t* sample_params) {
        sample_method = resolve_sample_method(sd_ctx, sample_method);

        total_steps = sample_steps + std::max(0, high_noise_sample_steps);

        if (sample_params->custom_sigmas_count > 0) {
            sigmas      = std::vector<float>(sample_params->custom_sigmas,
                                        sample_params->custom_sigmas + sample_params->custom_sigmas_count);
            total_steps = static_cast<int>(sigmas.size()) - 1;
            LOG_WARN("total_steps != custom_sigmas_count - 1, set total_steps to %d", total_steps);
            if (sample_steps >= total_steps) {
                sample_steps = total_steps;
                LOG_WARN("total_steps != custom_sigmas_count - 1, set sample_steps to %d", sample_steps);
            }
            if (high_noise_sample_steps > 0) {
                high_noise_sample_steps = total_steps - sample_steps;
                LOG_WARN("total_steps != custom_sigmas_count - 1, set high_noise_sample_steps to %d", high_noise_sample_steps);
            }
        } else {
            scheduler_t scheduler = resolve_scheduler(sd_ctx,
                                                      sample_params->scheduler,
                                                      sample_method);
            sigmas                = sd_ctx->sd->denoiser->get_sigmas(total_steps,
                                                                     sd_ctx->sd->get_image_seq_len(request->height, request->width, request->frames),
                                                                     scheduler,
                                                                     sd_ctx->sd->version);
        }

        eta = resolve_eta(sd_ctx, eta, sample_method);

        if (high_noise_sample_steps < 0) {
            for (size_t i = 0; i < sigmas.size(); ++i) {
                if (sigmas[i] < moe_boundary) {
                    high_noise_sample_steps = static_cast<int>(i);
                    break;
                }
            }
            LOG_DEBUG("switching from high noise model at step %d", high_noise_sample_steps);
        }

        LOG_INFO("sampling using %s method", sampling_methods_str[sample_method]);
        if (high_noise_sample_steps > 0) {
            high_noise_sample_method = resolve_sample_method(sd_ctx,
                                                             high_noise_sample_method);
            high_noise_eta           = resolve_eta(sd_ctx, high_noise_eta, high_noise_sample_method);
            LOG_INFO("sampling(high noise) using %s method", sampling_methods_str[high_noise_sample_method]);
        }

        if (sd_ctx->sd->use_pmid) {
            start_merge_step = int(sd_ctx->sd->pmid_model->style_strength / 100.f * total_steps);
            LOG_INFO("PHOTOMAKER: start_merge_step: %d", start_merge_step);
        }
    }
};

struct ImageGenerationLatents {
    sd::Tensor<float> init_latent;
    sd::Tensor<float> concat_latent;
    sd::Tensor<float> uncond_concat_latent;
    sd::Tensor<float> control_image;
    std::vector<sd::Tensor<float>> ref_images;
    std::vector<sd::Tensor<float>> ref_latents;
    sd::Tensor<float> denoise_mask;
    sd::Tensor<float> clip_vision_output;
    sd::Tensor<float> vace_context;
    int64_t ref_image_num = 0;
};

struct ImageGenerationEmbeds {
    SDCondition cond;
    SDCondition uncond;
    SDCondition img_cond;
    SDCondition id_cond;
};

struct CircularAxesState {
    bool circular_x = false;
    bool circular_y = false;
};

static CircularAxesState configure_image_vae_axes(sd_ctx_t* sd_ctx,
                                                  const sd_img_gen_params_t* sd_img_gen_params,
                                                  const GenerationRequest& request) {
    CircularAxesState original_axes = {sd_ctx->sd->circular_x, sd_ctx->sd->circular_y};

    if (!sd_img_gen_params->vae_tiling_params.enabled) {
        if (sd_ctx->sd->first_stage_model) {
            sd_ctx->sd->first_stage_model->set_circular_axes(sd_ctx->sd->circular_x, sd_ctx->sd->circular_y);
        }
        if (sd_ctx->sd->preview_vae) {
            sd_ctx->sd->preview_vae->set_circular_axes(sd_ctx->sd->circular_x, sd_ctx->sd->circular_y);
        }
        return original_axes;
    }

    int tile_size_x, tile_size_y;
    float overlap;
    int latent_size_x = request.width / request.vae_scale_factor;
    int latent_size_y = request.height / request.vae_scale_factor;
    sd_ctx->sd->first_stage_model->get_tile_sizes(tile_size_x,
                                                  tile_size_y,
                                                  overlap,
                                                  sd_img_gen_params->vae_tiling_params,
                                                  latent_size_x,
                                                  latent_size_y);

    sd_ctx->sd->circular_x = sd_ctx->sd->circular_x && (tile_size_x >= latent_size_x);
    sd_ctx->sd->circular_y = sd_ctx->sd->circular_y && (tile_size_y >= latent_size_y);

    if (sd_ctx->sd->first_stage_model) {
        sd_ctx->sd->first_stage_model->set_circular_axes(sd_ctx->sd->circular_x, sd_ctx->sd->circular_y);
    }
    if (sd_ctx->sd->preview_vae) {
        sd_ctx->sd->preview_vae->set_circular_axes(sd_ctx->sd->circular_x, sd_ctx->sd->circular_y);
    }

    sd_ctx->sd->circular_x = original_axes.circular_x && (tile_size_x < latent_size_x);
    sd_ctx->sd->circular_y = original_axes.circular_y && (tile_size_y < latent_size_y);

    return original_axes;
}

static void restore_image_vae_axes(sd_ctx_t* sd_ctx, const CircularAxesState& original_axes) {
    sd_ctx->sd->circular_x = original_axes.circular_x;
    sd_ctx->sd->circular_y = original_axes.circular_y;
}

class ImageVaeAxesGuard {
private:
    sd_ctx_t* sd_ctx = nullptr;
    CircularAxesState original_axes;

public:
    ImageVaeAxesGuard(sd_ctx_t* sd_ctx,
                      const sd_img_gen_params_t* sd_img_gen_params,
                      const GenerationRequest& request)
        : sd_ctx(sd_ctx),
          original_axes(configure_image_vae_axes(sd_ctx, sd_img_gen_params, request)) {}

    ~ImageVaeAxesGuard() {
        restore_image_vae_axes(sd_ctx, original_axes);
    }

    ImageVaeAxesGuard(const ImageVaeAxesGuard&)            = delete;
    ImageVaeAxesGuard& operator=(const ImageVaeAxesGuard&) = delete;
};

static std::optional<ImageGenerationLatents> prepare_image_generation_latents(sd_ctx_t* sd_ctx,
                                                                              const sd_img_gen_params_t* sd_img_gen_params,
                                                                              GenerationRequest* request,
                                                                              SamplePlan* plan) {
    int64_t prepare_start_ms = ggml_time_ms();

    sd::Tensor<float> init_image_tensor;
    sd::Tensor<float> control_image_tensor;
    sd::Tensor<float> mask_image_tensor;

    if (sd_img_gen_params->init_image.data != nullptr) {
        LOG_INFO("IMG2IMG");

        if (request->strength < 1.f) {
            size_t t_enc = static_cast<size_t>(plan->sample_steps * request->strength);
            if (t_enc == static_cast<size_t>(plan->sample_steps)) {
                t_enc--;
            }
            LOG_INFO("target t_enc is %zu steps", t_enc);
            std::vector<float> sigma_sched;
            sigma_sched.assign(plan->sigmas.begin() + plan->sample_steps - t_enc - 1, plan->sigmas.end());
            plan->sigmas       = std::move(sigma_sched);
            plan->sample_steps = static_cast<int>(plan->sigmas.size() - 1);
        }

        init_image_tensor = sd_image_to_tensor(sd_img_gen_params->init_image, request->width, request->height);
    }

    if (sd_img_gen_params->mask_image.data != nullptr) {
        mask_image_tensor = sd_image_to_tensor(sd_img_gen_params->mask_image, request->width, request->height);
        mask_image_tensor = sd::ops::round(mask_image_tensor);
    }

    if (sd_img_gen_params->control_image.data != nullptr) {
        control_image_tensor = sd_image_to_tensor(sd_img_gen_params->control_image, request->width, request->height);
    }

    if (init_image_tensor.empty() || mask_image_tensor.empty()) {
        if (sd_version_is_inpaint(sd_ctx->sd->version)) {
            LOG_WARN("inpainting model requires both an init image and a mask image.");
        }
    }

    if (mask_image_tensor.empty()) {
        mask_image_tensor = sd::full<float>({request->width, request->height, 1, 1}, 1.f);
    }

    sd::Tensor<float> latent_mask = sd::ops::interpolate(mask_image_tensor,
                                                         {request->width / request->vae_scale_factor,
                                                          request->height / request->vae_scale_factor,
                                                          1,
                                                          1},
                                                         sd::ops::InterpolateMode::NearestMax);

    sd::Tensor<float> init_latent;
    sd::Tensor<float> control_latent;
    if (init_image_tensor.empty()) {
        init_latent = sd_ctx->sd->generate_init_latent(request->width, request->height);
    } else {
        init_latent = sd_ctx->sd->encode_first_stage(init_image_tensor);
        if (init_latent.empty()) {
            LOG_ERROR("failed to encode init image");
            return std::nullopt;
        }
    }

    if (!control_image_tensor.empty() && !sd_ctx->sd->vae_decode_only) {
        control_latent = sd_ctx->sd->encode_first_stage(control_image_tensor);
        if (control_latent.empty()) {
            LOG_ERROR("failed to encode control image");
            return std::nullopt;
        }
    }

    std::vector<sd::Tensor<float>> ref_images;
    for (int i = 0; i < sd_img_gen_params->ref_images_count; i++) {
        ref_images.push_back(sd_image_to_tensor(sd_img_gen_params->ref_images[i]));
    }

    if (ref_images.empty() && sd_version_is_unet_edit(sd_ctx->sd->version)) {
        LOG_WARN("This model needs at least one reference image; using an empty reference");
        ref_images.push_back(sd::zeros<float>({request->width, request->height, 3, 1}));
        request->guidance.img_cfg = request->guidance.txt_cfg;
    }

    if (!ref_images.empty()) {
        LOG_INFO("EDIT mode");
    }

    std::vector<sd::Tensor<float>> ref_latents;
    for (size_t i = 0; i < ref_images.size(); i++) {
        sd::Tensor<float> ref_latent;
        if (request->auto_resize_ref_image) {
            LOG_DEBUG("auto resize ref images");
            int vae_image_size = std::min(1024 * 1024, request->width * request->height);
            double vae_width   = sqrt(vae_image_size * ref_images[i].shape()[0] / ref_images[i].shape()[1]);
            double vae_height  = vae_width * ref_images[i].shape()[1] / ref_images[i].shape()[0];

            int factor = sd_version_is_qwen_image(sd_ctx->sd->version) ? 32 : 16;
            vae_height = round(vae_height / factor) * factor;
            vae_width  = round(vae_width / factor) * factor;

            auto resized_ref_img = sd::ops::interpolate(ref_images[i],
                                                        {static_cast<int>(vae_width), static_cast<int>(vae_height), 3, 1});

            LOG_DEBUG("resize vae ref image %d from %" PRId64 "x%" PRId64 " to %" PRId64 "x%" PRId64,
                      static_cast<int>(i),
                      ref_images[i].shape()[1],
                      ref_images[i].shape()[0],
                      resized_ref_img.shape()[1],
                      resized_ref_img.shape()[0]);

            ref_latent = sd_ctx->sd->encode_first_stage(resized_ref_img);
        } else {
            ref_latent = sd_ctx->sd->encode_first_stage(ref_images[i]);
        }
        if (ref_latent.empty()) {
            LOG_ERROR("failed to encode reference image %d", static_cast<int>(i));
            return std::nullopt;
        }

        ref_latents.push_back(std::move(ref_latent));
    }

    sd::Tensor<float> concat_latent;
    sd::Tensor<float> uncond_concat_latent;
    if (sd_version_is_inpaint(sd_ctx->sd->version)) {
        sd::Tensor<float> masked_init_latent;

        if (sd_ctx->sd->version != VERSION_FLEX_2) {
            if (!init_image_tensor.empty()) {
                auto masked_image  = ((1.0f - mask_image_tensor) * (init_image_tensor - 0.5f)) + 0.5f;
                masked_init_latent = sd_ctx->sd->encode_first_stage(masked_image);
                if (masked_init_latent.empty()) {
                    LOG_ERROR("failed to encode masked init image");
                    return std::nullopt;
                }
            } else {
                masked_init_latent = sd::Tensor<float>::zeros_like(init_latent);
            }
        } else {
            masked_init_latent = ((1.0f - latent_mask) * init_latent);
        }

        auto uncond_masked_init_latent = sd::Tensor<float>::zeros_like(masked_init_latent);

        if (sd_ctx->sd->version == VERSION_FLUX_FILL) {
            auto mask = mask_image_tensor.reshape({request->vae_scale_factor,
                                                   request->width / request->vae_scale_factor,
                                                   request->vae_scale_factor,
                                                   request->height / request->vae_scale_factor});
            mask      = mask.permute({1, 3, 0, 2}).reshape({request->width / request->vae_scale_factor, request->height / request->vae_scale_factor, request->vae_scale_factor * request->vae_scale_factor, 1});

            concat_latent        = sd::ops::concat(masked_init_latent, mask, 2);
            uncond_concat_latent = sd::ops::concat(uncond_masked_init_latent, mask, 2);
        } else if (sd_ctx->sd->version == VERSION_FLEX_2) {
            concat_latent = sd::ops::concat(masked_init_latent, latent_mask, 2);
            if (!control_latent.empty()) {
                concat_latent = sd::ops::concat(concat_latent, control_latent, 2);
            } else {
                concat_latent = sd::ops::concat(concat_latent, sd::Tensor<float>::zeros_like(masked_init_latent), 2);
            }

            uncond_concat_latent = sd::ops::concat(uncond_masked_init_latent, latent_mask, 2);
            uncond_concat_latent = sd::ops::concat(uncond_concat_latent, sd::Tensor<float>::zeros_like(masked_init_latent), 2);
        } else {  // SD1.x SD2.x SDXL inpaint
            concat_latent        = sd::ops::concat(latent_mask, masked_init_latent, 2);
            uncond_concat_latent = sd::ops::concat(latent_mask, uncond_masked_init_latent, 2);
        }
    }
    if (sd_version_is_unet_edit(sd_ctx->sd->version)) {
        concat_latent        = sd::ops::interpolate<float>(ref_latents[0], init_latent.shape());
        uncond_concat_latent = sd::Tensor<float>::zeros_like(concat_latent);
    }
    if (sd_version_is_control(sd_ctx->sd->version)) {
        if (!control_latent.empty()) {
            concat_latent = control_latent;
        } else {
            concat_latent = sd::Tensor<float>::zeros_like(init_latent);
        }
        uncond_concat_latent = sd::Tensor<float>::zeros_like(concat_latent);
    }

    if (sd_img_gen_params->init_image.data != nullptr || sd_img_gen_params->ref_images_count > 0) {
        int64_t t1 = ggml_time_ms();
        LOG_INFO("encode_first_stage completed, taking %.2fs", (t1 - prepare_start_ms) * 1.0f / 1000);
    }

    ImageGenerationLatents latents;
    latents.init_latent          = std::move(init_latent);
    latents.concat_latent        = std::move(concat_latent);
    latents.uncond_concat_latent = std::move(uncond_concat_latent);
    latents.control_image        = std::move(control_image_tensor);
    latents.ref_images           = std::move(ref_images);
    latents.ref_latents          = std::move(ref_latents);

    if (sd_version_is_inpaint(sd_ctx->sd->version)) {
        latent_mask = sd::ops::max_pool_2d(latent_mask,
                                           {3, 3},
                                           {1, 1},
                                           {1, 1});
    }
    latents.denoise_mask = std::move(latent_mask);

    return latents;
}

static std::optional<ImageGenerationEmbeds> prepare_image_generation_embeds(sd_ctx_t* sd_ctx,
                                                                            const sd_img_gen_params_t* sd_img_gen_params,
                                                                            GenerationRequest* request,
                                                                            SamplePlan* plan,
                                                                            ImageGenerationLatents* latents) {
    ConditionerParams condition_params;
    condition_params.text            = request->prompt;
    condition_params.clip_skip       = request->clip_skip;
    condition_params.width           = request->width;
    condition_params.height          = request->height;
    condition_params.ref_images      = &latents->ref_images;
    condition_params.adm_in_channels = static_cast<int>(sd_ctx->sd->diffusion_model->get_adm_in_channels());

    auto id_cond                     = sd_ctx->sd->get_pmid_conditon(request->pm_params, condition_params);
    int64_t prepare_start_ms         = ggml_time_ms();
    condition_params.zero_out_masked = false;
    auto cond                        = sd_ctx->sd->cond_stage_model->get_learned_condition(sd_ctx->sd->n_threads,
                                                                                           condition_params);
    if (cond.c_concat.empty()) {
        cond.c_concat = latents->concat_latent;  // TODO: optimize
    }

    SDCondition uncond;
    if (request->use_uncond || request->use_high_noise_uncond) {
        bool zero_out_masked = false;
        if (sd_version_is_sdxl(sd_ctx->sd->version) &&
            request->negative_prompt.empty() &&
            !sd_ctx->sd->is_using_edm_v_parameterization) {
            zero_out_masked = true;
        }
        condition_params.text            = request->negative_prompt;
        condition_params.zero_out_masked = zero_out_masked;
        uncond                           = sd_ctx->sd->cond_stage_model->get_learned_condition(sd_ctx->sd->n_threads,
                                                                                               condition_params);
        if (uncond.c_concat.empty()) {
            uncond.c_concat = latents->uncond_concat_latent;  // TODO: optimize
        }
    }

    int64_t t1 = ggml_time_ms();
    LOG_INFO("get_learned_condition completed, taking %.2fs", (t1 - prepare_start_ms) * 1.0f / 1000);

    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->cond_stage_model->free_params_buffer();
    }

    ImageGenerationEmbeds embeds;
    if (request->use_img_cond) {
        embeds.img_cond = SDCondition(uncond.c_crossattn, uncond.c_vector, cond.c_concat);
    }
    embeds.cond    = std::move(cond);
    embeds.uncond  = std::move(uncond);
    embeds.id_cond = std::move(id_cond);

    return embeds;
}

static sd_image_t* decode_image_outputs(sd_ctx_t* sd_ctx,
                                        const GenerationRequest& request,
                                        const std::vector<sd::Tensor<float>>& final_latents) {
    if (final_latents.size() != static_cast<size_t>(request.batch_count)) {
        LOG_ERROR("expected %d latents, got %zu", request.batch_count, final_latents.size());
        return nullptr;
    }
    LOG_INFO("decoding %zu latents", final_latents.size());
    std::vector<sd::Tensor<float>> decoded_images;
    int64_t t0 = ggml_time_ms();

    for (size_t i = 0; i < final_latents.size(); i++) {
        int64_t t1              = ggml_time_ms();
        sd::Tensor<float> image = sd_ctx->sd->decode_first_stage(final_latents[i]);
        if (image.empty()) {
            LOG_ERROR("decode_first_stage failed for latent %" PRId64, i + 1);
            if (sd_ctx->sd->free_params_immediately) {
                sd_ctx->sd->first_stage_model->free_params_buffer();
            }
            return nullptr;
        }
        decoded_images.push_back(std::move(image));
        int64_t t2 = ggml_time_ms();
        LOG_INFO("latent %" PRId64 " decoded, taking %.2fs", i + 1, (t2 - t1) * 1.0f / 1000);
    }

    int64_t t4 = ggml_time_ms();
    LOG_INFO("decode_first_stage completed, taking %.2fs", (t4 - t0) * 1.0f / 1000);
    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->first_stage_model->free_params_buffer();
    }

    sd_image_t* result_images = (sd_image_t*)calloc(request.batch_count, sizeof(sd_image_t));
    if (result_images == nullptr) {
        return nullptr;
    }
    memset(result_images, 0, request.batch_count * sizeof(sd_image_t));

    for (size_t i = 0; i < decoded_images.size(); i++) {
        result_images[i] = tensor_to_sd_image(decoded_images[i]);
    }

    return result_images;
}

SD_API sd_image_t* generate_image(sd_ctx_t* sd_ctx, const sd_img_gen_params_t* sd_img_gen_params) {
    if (sd_ctx == nullptr || sd_img_gen_params == nullptr) {
        return nullptr;
    }

    int64_t t0                    = ggml_time_ms();
    sd_ctx->sd->vae_tiling_params = sd_img_gen_params->vae_tiling_params;
    GenerationRequest request(sd_ctx, sd_img_gen_params);
    {
        const int     vsf = std::max(1, request.vae_scale_factor);
        const int64_t lw  = request.width  / vsf;
        const int64_t lh  = request.height / vsf;
        sd_ctx->sd->maybe_auto_set_vae_tiling(lw, lh, /*t_latent=*/1);
    }
    LOG_INFO("generate_image %dx%d", request.width, request.height);

    sd_ctx->sd->rng->manual_seed(request.seed);
    sd_ctx->sd->sampler_rng->manual_seed(request.seed);
    sd_ctx->sd->set_flow_shift(sd_img_gen_params->sample_params.flow_shift);
    sd_ctx->sd->apply_loras(sd_img_gen_params->loras, sd_img_gen_params->lora_count);

    ImageVaeAxesGuard axes_guard(sd_ctx, sd_img_gen_params, request);

    SamplePlan plan(sd_ctx, sd_img_gen_params, request);
    auto latents_opt = prepare_image_generation_latents(sd_ctx,
                                                        sd_img_gen_params,
                                                        &request,
                                                        &plan);
    if (!latents_opt.has_value()) {
        return nullptr;
    }
    ImageGenerationLatents latents = std::move(*latents_opt);

    auto embeds_opt = prepare_image_generation_embeds(sd_ctx,
                                                      sd_img_gen_params,
                                                      &request,
                                                      &plan,
                                                      &latents);
    if (!embeds_opt.has_value()) {
        return nullptr;
    }
    ImageGenerationEmbeds embeds = std::move(*embeds_opt);

    std::vector<sd::Tensor<float>> final_latents;
    int64_t denoise_start = ggml_time_ms();
    for (int b = 0; b < request.batch_count; b++) {
        int64_t sampling_start = ggml_time_ms();
        int64_t cur_seed       = request.seed + b;
        LOG_INFO("generating image: %i/%i - seed %" PRId64, b + 1, request.batch_count, cur_seed);

        sd_ctx->sd->rng->manual_seed(cur_seed);
        sd_ctx->sd->sampler_rng->manual_seed(cur_seed);
        sd::Tensor<float> noise = sd::randn_like<float>(latents.init_latent, sd_ctx->sd->rng);

        sd::Tensor<float> x_0 = sd_ctx->sd->sample(sd_ctx->sd->diffusion_model,
                                                   true,
                                                   latents.init_latent,
                                                   std::move(noise),
                                                   embeds.cond,
                                                   embeds.uncond,
                                                   embeds.img_cond,
                                                   embeds.id_cond,
                                                   latents.control_image,
                                                   request.control_strength,
                                                   request.guidance,
                                                   plan.eta,
                                                   request.shifted_timestep,
                                                   plan.sample_method,
                                                   sd_ctx->sd->is_flow_denoiser(),
                                                   plan.sigmas,
                                                   plan.start_merge_step,
                                                   latents.ref_latents,
                                                   request.increase_ref_index,
                                                   latents.denoise_mask,
                                                   sd::Tensor<float>(),
                                                   1.f,
                                                   request.cache_params);
        int64_t sampling_end  = ggml_time_ms();
        if (!x_0.empty()) {
            LOG_INFO("sampling completed, taking %.2fs", (sampling_end - sampling_start) * 1.0f / 1000);
            final_latents.push_back(std::move(x_0));
            continue;
        }

        LOG_ERROR("sampling for image %d/%d failed after %.2fs",
                  b + 1,
                  request.batch_count,
                  (sampling_end - sampling_start) * 1.0f / 1000);
        if (sd_ctx->sd->free_params_immediately) {
            sd_ctx->sd->diffusion_model->free_params_buffer();
        }
        return nullptr;
    }
    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->diffusion_model->free_params_buffer();
    }
    int64_t denoise_end = ggml_time_ms();
    LOG_INFO("generating %" PRId64 " latent images completed, taking %.2fs",
             final_latents.size(),
             (denoise_end - denoise_start) * 1.0f / 1000);

    auto result = decode_image_outputs(sd_ctx, request, final_latents);
    if (result == nullptr) {
        return nullptr;
    }

    sd_ctx->sd->lora_stat();

    int64_t t1 = ggml_time_ms();
    LOG_INFO("generate_image completed in %.2fs", (t1 - t0) * 1.0f / 1000);
    return result;
}

static std::optional<ImageGenerationLatents> prepare_video_generation_latents(sd_ctx_t* sd_ctx,
                                                                              const sd_vid_gen_params_t* sd_vid_gen_params,
                                                                              GenerationRequest* request) {
    ImageGenerationLatents latents;
    int64_t prepare_start_ms = ggml_time_ms();

    sd::Tensor<float> start_image;
    sd::Tensor<float> end_image;

    if (sd_vid_gen_params->init_image.data) {
        start_image = sd_image_to_tensor(sd_vid_gen_params->init_image, request->width, request->height);
    }

    if (sd_vid_gen_params->end_image.data) {
        end_image = sd_image_to_tensor(sd_vid_gen_params->end_image, request->width, request->height);
    }

    if (sd_ctx->sd->diffusion_model->get_desc() == "Wan2.1-I2V-14B" ||
        sd_ctx->sd->diffusion_model->get_desc() == "Wan2.2-I2V-14B" ||
        sd_ctx->sd->diffusion_model->get_desc() == "Wan2.1-I2V-1.3B" ||
        sd_ctx->sd->diffusion_model->get_desc() == "Wan2.1-FLF2V-14B") {
        LOG_INFO("IMG2VID");

        if (sd_ctx->sd->diffusion_model->get_desc() == "Wan2.1-I2V-14B" ||
            sd_ctx->sd->diffusion_model->get_desc() == "Wan2.1-I2V-1.3B" ||
            sd_ctx->sd->diffusion_model->get_desc() == "Wan2.1-FLF2V-14B") {
            if (!start_image.empty()) {
                auto clip_vision_output = sd_ctx->sd->get_clip_vision_output(start_image, false, -2);
                if (clip_vision_output.empty()) {
                    LOG_ERROR("failed to compute clip vision output for init image");
                    return std::nullopt;
                }
                latents.clip_vision_output = std::move(clip_vision_output);
            } else {
                latents.clip_vision_output = sd_ctx->sd->get_clip_vision_output(start_image, false, -2, true);
            }

            if (sd_ctx->sd->diffusion_model->get_desc() == "Wan2.1-FLF2V-14B") {
                sd::Tensor<float> end_image_clip_vision_output;
                if (!end_image.empty()) {
                    end_image_clip_vision_output = sd_ctx->sd->get_clip_vision_output(end_image, false, -2);
                    if (end_image_clip_vision_output.empty()) {
                        LOG_ERROR("failed to compute clip vision output for end image");
                        return std::nullopt;
                    }
                } else {
                    end_image_clip_vision_output = sd_ctx->sd->get_clip_vision_output(end_image, false, -2, true);
                }
                latents.clip_vision_output = sd::ops::concat(latents.clip_vision_output, end_image_clip_vision_output, 1);
            }

            int64_t t1 = ggml_time_ms();
            LOG_INFO("get_clip_vision_output completed, taking %" PRId64 " ms", t1 - prepare_start_ms);
        }

        int64_t t1              = ggml_time_ms();
        sd::Tensor<float> image = sd::full<float>({request->width, request->height, request->frames, 3, 1}, 0.5f);
        if (!start_image.empty()) {
            sd::ops::slice_assign(&image, 2, 0, 1, start_image.unsqueeze(2));
        }
        if (!end_image.empty()) {
            sd::ops::slice_assign(&image, 2, request->frames - 1, request->frames, end_image.unsqueeze(2));
        }

        auto concat_latent = sd_ctx->sd->encode_first_stage(image);  // [b, c, t, h/vae_scale_factor, w/vae_scale_factor]
        if (concat_latent.empty()) {
            LOG_ERROR("failed to encode video conditioning frames");
            return std::nullopt;
        }
        latents.concat_latent = std::move(concat_latent);

        int64_t t2 = ggml_time_ms();
        LOG_INFO("encode_first_stage completed, taking %" PRId64 " ms", t2 - t1);

        sd::Tensor<float> concat_mask = sd::zeros<float>({latents.concat_latent.shape()[0],
                                                          latents.concat_latent.shape()[1],
                                                          latents.concat_latent.shape()[2],
                                                          4,
                                                          1});  // [b, 4, t, h/vae_scale_factor, w/vae_scale_factor]
        if (!start_image.empty()) {
            sd::ops::fill_slice(&concat_mask, 2, 0, 1, 1.0f);
        }
        if (!end_image.empty()) {
            auto last_channel = sd::ops::slice(concat_mask, 3, 3, 4);
            sd::ops::fill_slice(&last_channel, 2, last_channel.shape()[2] - 1, last_channel.shape()[2], 1.0f);
            sd::ops::slice_assign(&concat_mask, 3, 3, 4, last_channel);
        }
        latents.concat_latent = sd::ops::concat(concat_mask, latents.concat_latent, 3);  // [b, 4+c, t, h/vae_scale_factor, w/vae_scale_factor]
    } else if (sd_ctx->sd->diffusion_model->get_desc() == "Wan2.2-TI2V-5B" && !start_image.empty()) {
        LOG_INFO("IMG2VID");

        int64_t t1             = ggml_time_ms();
        auto init_img          = start_image.reshape({start_image.shape()[0], start_image.shape()[1], 1, start_image.shape()[2], 1});
        auto init_image_latent = sd_ctx->sd->encode_first_stage(init_img);  // [b, c, 1, h/vae_scale_factor, w/vae_scale_factor]
        if (init_image_latent.empty()) {
            LOG_ERROR("failed to encode init video frame");
            return std::nullopt;
        }

        latents.init_latent = sd_ctx->sd->generate_init_latent(request->width, request->height, request->frames, true);  // [b, c, t, h/vae_scale_factor, w/vae_scale_factor]
        sd::ops::slice_assign(&latents.init_latent, 2, 0, init_image_latent.shape()[2], init_image_latent);

        latents.denoise_mask = sd::full<float>({latents.init_latent.shape()[0], latents.init_latent.shape()[1], latents.init_latent.shape()[2], 1, 1}, 1.f);
        sd::ops::fill_slice(&latents.denoise_mask, 2, 0, init_image_latent.shape()[2], 0.0f);

        int64_t t2 = ggml_time_ms();
        LOG_INFO("encode_first_stage completed, taking %" PRId64 " ms", t2 - t1);
    } else if (sd_ctx->sd->diffusion_model->get_desc() == "Wan2.1-VACE-1.3B" ||
               sd_ctx->sd->diffusion_model->get_desc() == "Wan2.x-VACE-14B") {
        LOG_INFO("VACE");
        int64_t t1 = ggml_time_ms();
        sd::Tensor<float> ref_image_latent;
        if (!start_image.empty()) {
            auto ref_img     = start_image.reshape({start_image.shape()[0], start_image.shape()[1], 1, start_image.shape()[2], 1});
            auto encoded_ref = sd_ctx->sd->encode_first_stage(ref_img);  // [b, c, 1, h/vae_scale_factor, w/vae_scale_factor]
            if (encoded_ref.empty()) {
                LOG_ERROR("failed to encode VACE reference image");
                return std::nullopt;
            }
            ref_image_latent = sd::ops::concat(encoded_ref, sd::zeros<float>(encoded_ref.shape()), 3);  // [b, 2*c, 1, h/vae_scale_factor, w/vae_scale_factor]
        }

        sd::Tensor<float> control_video = sd::full<float>({request->width, request->height, request->frames, 3, 1}, 0.5f);
        int64_t control_frame_count     = std::min<int64_t>(request->frames, sd_vid_gen_params->control_frames_size);
        for (int64_t i = 0; i < control_frame_count; ++i) {
            auto control_frame = sd_image_to_tensor(sd_vid_gen_params->control_frames[i], request->width, request->height);
            sd::ops::slice_assign(&control_video, 2, i, i + 1, control_frame.unsqueeze(2));
        }

        sd::Tensor<float> mask = sd::full<float>({request->width, request->height, request->frames, 1, 1}, 1.0f);

        control_video              = control_video - 0.5f;
        sd::Tensor<float> inactive = control_video * (1.0f - mask) + 0.5f;
        sd::Tensor<float> reactive = control_video * mask + 0.5f;

        inactive = sd_ctx->sd->encode_first_stage(inactive);  // [b, c, t, h/vae_scale_factor, w/vae_scale_factor]
        if (inactive.empty()) {
            LOG_ERROR("failed to encode VACE inactive context");
            return std::nullopt;
        }

        reactive = sd_ctx->sd->encode_first_stage(reactive);  // [b, c, t, h/vae_scale_factor, w/vae_scale_factor]
        if (reactive.empty()) {
            LOG_ERROR("failed to encode VACE reactive context");
            return std::nullopt;
        }

        int64_t length = inactive.shape()[2];
        if (!ref_image_latent.empty()) {
            length += 1;
            request->frames       = static_cast<int>((length - 1) * 4 + 1);
            latents.ref_image_num = 1;
        }
        auto vace_context = sd::ops::concat(inactive, reactive, 3);  // [b, 2*c, t, h/vae_scale_factor, w/vae_scale_factor]

        mask              = sd::full<float>({request->width, request->height, inactive.shape()[2], 1, 1}, 1.0f);
        auto mask_context = mask.reshape({request->vae_scale_factor,
                                          inactive.shape()[0],
                                          request->vae_scale_factor,
                                          inactive.shape()[1],
                                          inactive.shape()[2]});   // [t, h/vae_scale_factor, vae_scale_factor, w/vae_scale_factor, vae_scale_factor]
        mask_context      = mask_context.permute({1, 3, 4, 0, 2})  // [vae_scale_factor, vae_scale_factor, t, h/vae_scale_factor, w/vae_scale_factor]
                           .reshape({inactive.shape()[0],
                                     inactive.shape()[1],
                                     inactive.shape()[2],
                                     request->vae_scale_factor * request->vae_scale_factor});  // [vae_scale_factor*vae_scale_factor, t, h/vae_scale_factor, w/vae_scale_factor]

        if (!ref_image_latent.empty()) {
            vace_context  = sd::ops::concat(ref_image_latent, vace_context, 2);  // [b, 2*c, t+1, h/vae_scale_factor, w/vae_scale_factor]
            auto mask_pad = sd::zeros<float>({mask_context.shape()[0],
                                              mask_context.shape()[1],
                                              1,
                                              mask_context.shape()[3]});  // [vae_scale_factor*vae_scale_factor, 1, h/vae_scale_factor, w/vae_scale_factor]
            mask_context  = sd::ops::concat(mask_pad, mask_context, 2);   // [vae_scale_factor*vae_scale_factor, t + 1, h/vae_scale_factor, w/vae_scale_factor]
        }

        mask_context.unsqueeze_(mask_context.dim());  // [b, vae_scale_factor*vae_scale_factor, t + 1 or t, h/vae_scale_factor, w/vae_scale_factor]

        latents.vace_context = sd::ops::concat(vace_context, mask_context, 3);  // [b, 2*c + vae_scale_factor*vae_scale_factor, t + 1 or t, h/vae_scale_factor, w/vae_scale_factor]
        int64_t t2           = ggml_time_ms();
        LOG_INFO("encode_first_stage completed, taking %" PRId64 " ms", t2 - t1);
    }

    if (latents.init_latent.empty()) {
        latents.init_latent = sd_ctx->sd->generate_init_latent(request->width, request->height, request->frames, true);
    }

    return latents;
}

static ImageGenerationEmbeds prepare_video_generation_embeds(sd_ctx_t* sd_ctx,
                                                             const sd_vid_gen_params_t* sd_vid_gen_params,
                                                             const GenerationRequest& request,
                                                             const ImageGenerationLatents& latents) {
    ImageGenerationEmbeds embeds;
    ConditionerParams condition_params;
    condition_params.clip_skip       = request.clip_skip;
    condition_params.text            = request.prompt;
    condition_params.zero_out_masked = true;

    int64_t prepare_start_ms = ggml_time_ms();
    embeds.cond              = sd_ctx->sd->cond_stage_model->get_learned_condition(sd_ctx->sd->n_threads,
                                                                                   condition_params);
    embeds.cond.c_concat     = latents.concat_latent;
    embeds.cond.c_vector     = latents.clip_vision_output;
    if (request.use_uncond) {
        condition_params.text  = request.negative_prompt;
        embeds.uncond          = sd_ctx->sd->cond_stage_model->get_learned_condition(sd_ctx->sd->n_threads,
                                                                                     condition_params);
        embeds.uncond.c_concat = latents.concat_latent;
        embeds.uncond.c_vector = latents.clip_vision_output;
    }

    int64_t t1 = ggml_time_ms();
    LOG_INFO("get_learned_condition completed, taking %.2fs", (t1 - prepare_start_ms) * 1.0f / 1000);

    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->cond_stage_model->free_params_buffer();
    }
    return embeds;
}

static sd_image_t* decode_video_outputs(sd_ctx_t* sd_ctx,
                                        const sd::Tensor<float>& final_latent,
                                        int* num_frames_out) {
    if (final_latent.empty()) {
        LOG_ERROR("no latent video to decode");
        return nullptr;
    }
    int64_t t4            = ggml_time_ms();
    sd::Tensor<float> vid = sd_ctx->sd->decode_first_stage(final_latent, true);
    int64_t t5            = ggml_time_ms();
    LOG_INFO("decode_first_stage completed, taking %.2fs", (t5 - t4) * 1.0f / 1000);
    // Diagnostic (gated by SD_DUMP_GEN_STATS=1): post-decode video stats.
    // Should be in [0, 1] after the scale_output rescale; mean ~0.3-0.5 for a
    // natural image. If mean is near 0, decoder output is being clamped.
    if (std::getenv("SD_DUMP_GEN_STATS") && !vid.empty()) {
        const auto& sh = vid.shape();
        const float* d = vid.data();
        int64_t n = vid.numel();
        double s = 0, sq = 0; float lo = d[0], hi = d[0];
        for (int64_t i = 0; i < n; ++i) {
            double v = d[i]; s += v; sq += v*v;
            if (d[i] < lo) lo = d[i]; if (d[i] > hi) hi = d[i];
        }
        double m = s / n; double v = sq / n - m * m;
        std::printf("=== decoded video stats: shape=[");
        for (size_t i = 0; i < sh.size(); ++i) std::printf("%s%lld", i ? "," : "", (long long)sh[i]);
        std::printf("] overall mean=%.3f std=%.3f min=%.3f max=%.3f ===\n",
                    m, std::sqrt(std::max(0.0, v)), lo, hi);
        // Per-channel breakdown if 5D [W,H,T,C,B] or 4D [W,H,T,C]
        if (sh.size() >= 4) {
            int64_t W = sh[0], H = sh[1], T = sh[2], C = sh[3];
            for (int64_t c = 0; c < std::min<int64_t>(C, 3); ++c) {
                double cs = 0, csq = 0; int64_t cn = W * H * T;
                for (int64_t t = 0; t < T; ++t)
                    for (int64_t h = 0; h < H; ++h)
                        for (int64_t w = 0; w < W; ++w) {
                            double vv = d[((c * T + t) * H + h) * W + w];
                            cs += vv; csq += vv*vv;
                        }
                double cm = cs / cn; double cv = csq / cn - cm * cm;
                std::printf("  channel %lld: mean=%.3f std=%.3f\n", (long long)c,
                            cm, std::sqrt(std::max(0.0, cv)));
            }
        }
    }
    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->first_stage_model->free_params_buffer();
    }
    if (vid.empty()) {
        LOG_ERROR("decode_first_stage failed for video");
        return nullptr;
    }

    sd_image_t* result_images = (sd_image_t*)calloc(vid.shape()[2], sizeof(sd_image_t));
    if (result_images == nullptr) {
        return nullptr;
    }
    if (num_frames_out != nullptr) {
        *num_frames_out = static_cast<int>(vid.shape()[2]);
    }

    for (int64_t i = 0; i < vid.shape()[2]; i++) {
        result_images[i] = tensor_to_sd_image(vid, static_cast<int>(i));
    }

    return result_images;
}

SD_API sd_image_t* generate_video(sd_ctx_t* sd_ctx, const sd_vid_gen_params_t* sd_vid_gen_params, int* num_frames_out) {
    if (sd_ctx == nullptr || sd_vid_gen_params == nullptr) {
        return nullptr;
    }
    if (num_frames_out != nullptr) {
        *num_frames_out = 0;
    }
    int64_t t0                    = ggml_time_ms();
    sd_ctx->sd->vae_tiling_params = sd_vid_gen_params->vae_tiling_params;
    GenerationRequest request(sd_ctx, sd_vid_gen_params);
    {
        const int      vsf      = std::max(1, request.vae_scale_factor);
        const int64_t  lw       = request.width  / vsf;
        const int64_t  lh       = request.height / vsf;
        // Temporal compression: WAN /4, LTX-2 /8, otherwise no compression.
        // GenerationRequest::frames already reflects the user request; the
        // latent T is computed the same way StableDiffusionGGML does it.
        int64_t        t_latent = request.frames;
        if (sd_version_is_wan(sd_ctx->sd->version))        t_latent = ((t_latent - 1) / 4) + 1;
        else if (sd_version_is_ltx2(sd_ctx->sd->version))  t_latent = ((t_latent - 1) / 8) + 1;
        sd_ctx->sd->maybe_auto_set_vae_tiling(lw, lh, t_latent);
    }
    sd_ctx->sd->rng->manual_seed(request.seed);
    sd_ctx->sd->sampler_rng->manual_seed(request.seed);
    sd_ctx->sd->set_flow_shift(sd_vid_gen_params->sample_params.flow_shift);
    sd_ctx->sd->apply_loras(sd_vid_gen_params->loras, sd_vid_gen_params->lora_count);

    // Propagate output fps to diffusion models that need it for temporal RoPE
    // (LTX-2 divides time positions by fps; see LTXRope::gen_video_positions).
    if (request.fps > 0.f && sd_ctx->sd->diffusion_model) {
        sd_ctx->sd->diffusion_model->set_fps(request.fps);
    }
    if (request.fps > 0.f && sd_ctx->sd->high_noise_diffusion_model) {
        sd_ctx->sd->high_noise_diffusion_model->set_fps(request.fps);
    }

    SamplePlan plan(sd_ctx, sd_vid_gen_params, request);
    auto latent_inputs_opt = prepare_video_generation_latents(sd_ctx, sd_vid_gen_params, &request);
    if (!latent_inputs_opt.has_value()) {
        return nullptr;
    }
    ImageGenerationLatents latents = std::move(*latent_inputs_opt);
    ImageGenerationEmbeds embeds   = prepare_video_generation_embeds(sd_ctx,
                                                                     sd_vid_gen_params,
                                                                     request,
                                                                     latents);
    LOG_INFO("generate_video %dx%dx%d",
             request.width,
             request.height,
             request.frames);

    int64_t latent_start = ggml_time_ms();
    int W                = request.width / request.vae_scale_factor;
    int H                = request.height / request.vae_scale_factor;
    int T                = static_cast<int>(latents.init_latent.shape()[2]);

    sd::Tensor<float> x_t   = latents.init_latent;
    sd::Tensor<float> noise = sd::Tensor<float>::randn_like(x_t, sd_ctx->sd->rng);

    if (plan.high_noise_sample_steps > 0) {
        LOG_DEBUG("sample(high noise) %dx%dx%d", W, H, T);

        int64_t sampling_start = ggml_time_ms();
        std::vector<float> high_noise_sigmas(plan.sigmas.begin(), plan.sigmas.begin() + plan.high_noise_sample_steps + 1);
        plan.sigmas = std::vector<float>(plan.sigmas.begin() + plan.high_noise_sample_steps, plan.sigmas.end());

        sd::Tensor<float> x_t_sampled = sd_ctx->sd->sample(sd_ctx->sd->high_noise_diffusion_model,
                                                           false,
                                                           x_t,
                                                           std::move(noise),
                                                           embeds.cond,
                                                           request.use_high_noise_uncond ? embeds.uncond : SDCondition(),
                                                           embeds.img_cond,
                                                           embeds.id_cond,
                                                           sd::Tensor<float>(),
                                                           0.f,
                                                           request.high_noise_guidance,
                                                           plan.high_noise_eta,
                                                           request.shifted_timestep,
                                                           plan.high_noise_sample_method,
                                                           sd_ctx->sd->is_flow_denoiser(),
                                                           high_noise_sigmas,
                                                           -1,
                                                           std::vector<sd::Tensor<float>>{},
                                                           false,
                                                           latents.denoise_mask,
                                                           latents.vace_context,
                                                           request.vace_strength,
                                                           request.cache_params);
        int64_t sampling_end          = ggml_time_ms();
        if (x_t_sampled.empty()) {
            LOG_ERROR("sampling(high noise) failed after %.2fs", (sampling_end - sampling_start) * 1.0f / 1000);
            if (sd_ctx->sd->free_params_immediately) {
                sd_ctx->sd->high_noise_diffusion_model->free_params_buffer();
            }
            return nullptr;
        }

        x_t   = std::move(x_t_sampled);
        noise = {};
        LOG_INFO("sampling(high noise) completed, taking %.2fs", (sampling_end - sampling_start) * 1.0f / 1000);
        if (sd_ctx->sd->free_params_immediately) {
            sd_ctx->sd->high_noise_diffusion_model->free_params_buffer();
        }
    }

    LOG_DEBUG("sample %dx%dx%d", W, H, T);
    int64_t sampling_start         = ggml_time_ms();
    sd::Tensor<float> final_latent = sd_ctx->sd->sample(sd_ctx->sd->diffusion_model,
                                                        true,
                                                        x_t,
                                                        std::move(noise),
                                                        embeds.cond,
                                                        request.use_uncond ? embeds.uncond : SDCondition(),
                                                        embeds.img_cond,
                                                        embeds.id_cond,
                                                        sd::Tensor<float>(),
                                                        0.f,
                                                        sd_vid_gen_params->sample_params.guidance,
                                                        plan.eta,
                                                        sd_vid_gen_params->sample_params.shifted_timestep,
                                                        plan.sample_method,
                                                        sd_ctx->sd->is_flow_denoiser(),
                                                        plan.sigmas,
                                                        -1,
                                                        std::vector<sd::Tensor<float>>{},
                                                        false,
                                                        latents.denoise_mask,
                                                        latents.vace_context,
                                                        request.vace_strength,
                                                        request.cache_params);

    int64_t sampling_end = ggml_time_ms();
    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->diffusion_model->free_params_buffer();
    }
    if (final_latent.empty()) {
        LOG_ERROR("sampling failed after %.2fs", (sampling_end - sampling_start) * 1.0f / 1000);
        return nullptr;
    }
    LOG_INFO("sampling completed, taking %.2fs", (sampling_end - sampling_start) * 1.0f / 1000);

    if (latents.ref_image_num > 0) {
        final_latent = sd::ops::slice(final_latent, 2, latents.ref_image_num, final_latent.shape()[2]);
    }

    int64_t latent_end = ggml_time_ms();
    LOG_INFO("generating latent video completed, taking %.2fs", (latent_end - latent_start) * 1.0f / 1000);

    // Diagnostic (gated by SD_DUMP_GEN_STATS=1): per-channel mean/std of the
    // final latent. Useful when VAE output looks off — confirms the latent is
    // in-distribution (~zero mean, unit std per channel post-PCS-normalize).
    if (std::getenv("SD_DUMP_GEN_STATS")) {
        const auto& sh = final_latent.shape();
        if (sh.size() >= 4) {
            int64_t W = sh[0], H = sh[1], T = sh[2], C = sh[3];
            const float* d = final_latent.data();
            double overall_s = 0, overall_sq = 0;
            int64_t overall_n = 0;
            std::printf("=== final_latent stats (W=%lld H=%lld T=%lld C=%lld) ===\n",
                        (long long)W, (long long)H, (long long)T, (long long)C);
            std::printf("first 4 channels (mean/std):");
            for (int64_t c = 0; c < std::min<int64_t>(C, 4); ++c) {
                double s = 0, sq = 0; int64_t n = W * H * T;
                for (int64_t t = 0; t < T; ++t)
                    for (int64_t h = 0; h < H; ++h)
                        for (int64_t w = 0; w < W; ++w) {
                            double v = d[((c * T + t) * H + h) * W + w];
                            s += v; sq += v*v;
                        }
                double mean = s / n;
                double var  = sq / n - mean * mean;
                std::printf(" c%lld=%.3f/%.3f", (long long)c, mean, std::sqrt(std::max(0.0, var)));
            }
            for (int64_t i = 0; i < final_latent.numel(); ++i) {
                double v = d[i]; overall_s += v; overall_sq += v*v;
            }
            overall_n = final_latent.numel();
            double om = overall_s / overall_n;
            double ov = overall_sq / overall_n - om * om;
            std::printf("\noverall: mean=%.3f std=%.3f min/max=", om, std::sqrt(std::max(0.0, ov)));
            const float* d2 = final_latent.data();
            float lo = d2[0], hi = d2[0];
            for (int64_t i = 1; i < overall_n; ++i) { if (d2[i] < lo) lo = d2[i]; if (d2[i] > hi) hi = d2[i]; }
            std::printf("%.3f/%.3f\n", lo, hi);
        }
    }

    auto result = decode_video_outputs(sd_ctx, final_latent, num_frames_out);
    if (result == nullptr) {
        return nullptr;
    }

    sd_ctx->sd->lora_stat();

    int64_t t1 = ggml_time_ms();
    LOG_INFO("generate_video completed in %.2fs", (t1 - t0) * 1.0f / 1000);
    return result;
}
