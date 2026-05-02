#include "ggml_extend.hpp"

#include "backend_fit.hpp"
#include "model.h"
#include "rng.hpp"
#include "rng_mt19937.hpp"
#include "rng_philox.hpp"
#include "stable-diffusion.h"
#include "util.h"

#include "auto_encoder_kl.hpp"
#include "conditioner.hpp"
#include "control.hpp"
#include "denoiser.hpp"
#include "diffusion_model.hpp"
#include "esrgan.hpp"
#include "lora.hpp"
#include "pmid.hpp"
#include "sample-cache.h"
#include "tae.hpp"
#include "upscaler.h"
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
    ggml_backend_t backend             = nullptr;  // general / main backend
    ggml_backend_t clip_backend        = nullptr;
    ggml_backend_t control_net_backend = nullptr;
    ggml_backend_t vae_backend         = nullptr;
    ggml_backend_t diffusion_backend   = nullptr;

    // Auto-fit decisions resolved into device-name strings. When non-empty,
    // these win over the user-provided sd_ctx_params->*_backend_device.
    // When empty, the explicit param (or `backend` fallback) is used.
    std::string fit_diffusion_device;
    std::string fit_clip_device;
    std::string fit_vae_device;
    // Per-component offload-params override coming from auto-fit. Forces
    // offload_params_to_cpu for that component even when global flag is off.
    bool fit_dit_offload_params  = false;
    bool fit_cond_offload_params = false;
    bool fit_vae_offload_params  = false;

    // Multi-GPU split state (LAYER_SPLIT or ROW_SPLIT). Holds the ordered
    // list of device names and per-device share bytes; the actual backend
    // handles are init'd at construction time and stored in *_extra_backends
    // so the destructor can free them. fit_*_row_split=true means use the
    // CUDA row-split path (matmul weights split row-wise via cuda_split_buft);
    // false means layer-split (per-block backend assignment via sched).
    std::vector<std::string>    fit_dit_split_device_names;
    std::vector<int64_t>        fit_dit_split_share_bytes;
    std::vector<ggml_backend_t> fit_dit_extra_backends;
    bool                        fit_dit_row_split  = false;
    std::vector<std::string>    fit_cond_split_device_names;
    std::vector<int64_t>        fit_cond_split_share_bytes;
    std::vector<ggml_backend_t> fit_cond_extra_backends;
    bool                        fit_cond_row_split = false;

    // Owned model loader: kept alive across init() so lazy_load callbacks
    // can re-read tensor data from disk on demand. Only set when at least
    // one component is configured for lazy load.
    std::unique_ptr<ModelLoader> owned_model_loader;
    // Auto-fit decided init-time SUM exceeds device cap; defer cond + DiT
    // allocation until first compute() so peaks don't pile up.
    bool auto_lazy_load = false;
    bool enable_mmap_member = false;

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

    std::map<std::string, ggml_tensor*> tensors;

    // lora_name => multiplier
    std::unordered_map<std::string, float> curr_lora_state;

    std::shared_ptr<Denoiser> denoiser = std::make_shared<CompVisDenoiser>();

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
        if (diffusion_backend != backend) {
            ggml_backend_free(diffusion_backend);
        }
        for (auto* b : fit_dit_extra_backends) {
            if (b != backend && b != diffusion_backend && b != clip_backend &&
                b != vae_backend && b != control_net_backend) {
                ggml_backend_free(b);
            }
        }
        for (auto* b : fit_cond_extra_backends) {
            if (b != backend && b != diffusion_backend && b != clip_backend &&
                b != vae_backend && b != control_net_backend) {
                ggml_backend_free(b);
            }
        }
        ggml_backend_free(backend);
    }

    void init_backend(const char* main_device_name) {
        if (main_device_name != nullptr && main_device_name[0] != '\0') {
            backend = init_named_backend(main_device_name);
            if (backend == nullptr) {
                LOG_WARN("main backend device '%s' init failed; falling back to default",
                         main_device_name);
            }
        }
        if (backend == nullptr) {
            backend = sd_get_default_backend();
        }
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

        init_backend(sd_ctx_params->main_backend_device);

        // Use a stack-local handle that points into `owned_model_loader` if we
        // need lazy callbacks (decided after auto-fit), otherwise a temp local
        // is fine. Defer the unique_ptr decision; for now always own it so the
        // pointer is stable even if lazy load is enabled later in this init().
        owned_model_loader = std::make_unique<ModelLoader>();
        ModelLoader& model_loader = *owned_model_loader;

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
            auto components = backend_fit::estimate_components(
                model_loader, wtype, /*alignment=*/64, reserves);
            auto    devices = backend_fit::enumerate_gpu_devices();
            int64_t margin_bytes =
                int64_t(std::max(0, sd_ctx_params->auto_fit_target_mb)) * backend_fit::MiB;
            backend_fit::MultiGpuMode mode = backend_fit::str_to_multi_gpu_mode(
                SAFE_STR(sd_ctx_params->multi_gpu_mode));
            auto plan = backend_fit::compute_plan(
                components, devices, margin_bytes,
                sd_ctx_params->auto_multi_gpu, mode);
            backend_fit::print_plan(plan, components, devices, margin_bytes);

            if (sd_ctx_params->auto_fit_dry_run) {
                LOG_INFO("auto-fit: --fit-dry-run set, aborting init before loading models");
                return false;
            }

            // Find the CPU device's ggml name (so we can route "CPU"
            // placements through init_named_backend uniformly).
            std::string cpu_device_name;
            for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                    cpu_device_name = ggml_backend_dev_name(dev);
                    break;
                }
            }
            auto device_id_to_name = [&](int dev_id) -> std::string {
                for (const auto& dev : devices) {
                    if (dev.id == dev_id) return dev.name;
                }
                return {};
            };
            auto resolve = [&](const backend_fit::Decision*  d,
                               std::string&                  out_device,
                               bool&                         out_offload,
                               std::vector<std::string>&     out_split_devices,
                               std::vector<int64_t>&         out_split_shares,
                               bool&                         out_row_split) {
                out_split_devices.clear();
                out_split_shares.clear();
                out_row_split = false;
                if (d == nullptr) {
                    out_device.clear();
                    out_offload = false;
                    return;
                }
                if (d->placement == backend_fit::Placement::CPU) {
                    out_device = cpu_device_name;
                    out_offload = false;
                    return;
                }
                if (d->placement == backend_fit::Placement::GPU_LAYER_SPLIT ||
                    d->placement == backend_fit::Placement::GPU_TENSOR_SPLIT) {
                    // Primary device drives main_backend choice for the model;
                    // the rest become additional backends in the spec.
                    for (size_t k = 0; k < d->split_device_ids.size(); k++) {
                        out_split_devices.push_back(device_id_to_name(d->split_device_ids[k]));
                        out_split_shares.push_back(d->split_share_bytes[k]);
                    }
                    if (!out_split_devices.empty()) out_device = out_split_devices[0];
                    out_offload = false;
                    out_row_split = (d->placement == backend_fit::Placement::GPU_TENSOR_SPLIT);
                    return;
                }
                out_device  = device_id_to_name(d->device_id);
                out_offload = (d->placement == backend_fit::Placement::GPU_OFFLOAD_PARAMS);
            };
            std::vector<std::string> dummy_devs;
            std::vector<int64_t>     dummy_shares;
            bool                     dummy_row_split = false;
            resolve(backend_fit::find_decision(plan, backend_fit::ComponentKind::DIT),
                    fit_diffusion_device, fit_dit_offload_params,
                    fit_dit_split_device_names, fit_dit_split_share_bytes,
                    fit_dit_row_split);
            resolve(backend_fit::find_decision(plan, backend_fit::ComponentKind::VAE),
                    fit_vae_device, fit_vae_offload_params, dummy_devs, dummy_shares,
                    dummy_row_split);
            resolve(backend_fit::find_decision(plan, backend_fit::ComponentKind::CONDITIONER),
                    fit_clip_device, fit_cond_offload_params,
                    fit_cond_split_device_names, fit_cond_split_share_bytes,
                    fit_cond_row_split);

            // CPU placements: leave fit_*_device empty AND remember they're
            // CPU so the resolver below picks ggml_backend_cpu_init().

            // Decide auto-lazy-load: if the per-component MAX-based plan fits
            // but the SUM-of-components on any device would exceed cap, defer
            // alloc until first compute() so peaks don't pile up. Heuristic:
            // sum the per-device on_device_bytes across all GPU decisions
            // (excluding VAE which is small) and compare to free_bytes.
            std::map<int, int64_t> sum_per_device;
            auto add_sum = [&](const backend_fit::Decision* d) {
                if (!d) return;
                if (d->placement == backend_fit::Placement::GPU_LAYER_SPLIT ||
                    d->placement == backend_fit::Placement::GPU_TENSOR_SPLIT) {
                    for (size_t k = 0; k < d->split_device_ids.size(); k++) {
                        sum_per_device[d->split_device_ids[k]] += d->split_share_bytes[k];
                    }
                } else if (d->placement == backend_fit::Placement::GPU ||
                           d->placement == backend_fit::Placement::GPU_OFFLOAD_PARAMS) {
                    sum_per_device[d->device_id] += d->on_device_bytes;
                }
            };
            add_sum(backend_fit::find_decision(plan, backend_fit::ComponentKind::DIT));
            add_sum(backend_fit::find_decision(plan, backend_fit::ComponentKind::VAE));
            add_sum(backend_fit::find_decision(plan, backend_fit::ComponentKind::CONDITIONER));
            for (const auto& dev : devices) {
                int64_t cap = dev.free_bytes - margin_bytes;
                int64_t sum = sum_per_device.count(dev.id) ? sum_per_device[dev.id] : 0;
                if (sum > cap) {
                    LOG_INFO("auto-fit: enabling lazy load (init-time SUM %lld MiB on %s "
                             "exceeds cap %lld MiB; per-component MAX plan needs lazy alloc)",
                             (long long)(sum / backend_fit::MiB),
                             dev.name.c_str(),
                             (long long)(cap / backend_fit::MiB));
                    auto_lazy_load = true;
                    break;
                }
            }
        }

        LOG_INFO("Weight type stat:                 %s", wtype_stat_to_str(wtype_stat).c_str());
        LOG_INFO("Conditioner weight type stat:     %s", wtype_stat_to_str(conditioner_wtype_stat).c_str());
        LOG_INFO("Diffusion model weight type stat: %s", wtype_stat_to_str(diffusion_model_wtype_stat).c_str());
        LOG_INFO("VAE weight type stat:             %s", wtype_stat_to_str(vae_wtype_stat).c_str());

        LOG_DEBUG("ggml tensor size = %d bytes", (int)sizeof(ggml_tensor));

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

        // If auto-fit decided ANY component must offload params, force the
        // global flag on. This is a coarsening: one component needing offload
        // forces all to offload (safer, just slower for non-offload ones).
        if (fit_dit_offload_params || fit_cond_offload_params || fit_vae_offload_params) {
            if (!offload_params_to_cpu) {
                LOG_INFO("auto-fit: enabling offload_params_to_cpu (one or more "
                         "components don't fit without param streaming)");
                offload_params_to_cpu = true;
            }
        }

        // Pick the effective device name for each component: the auto-fit
        // override (if any) wins; otherwise the user-provided string; nullptr
        // falls back to `backend` (the main).
        auto effective_device = [&](const std::string& fit_str, const char* user_str) -> const char* {
            if (!fit_str.empty()) return fit_str.c_str();
            return user_str;
        };
        const char* diffusion_dev_name = effective_device(fit_diffusion_device,
                                                          sd_ctx_params->diffusion_backend_device);
        const char* clip_dev_name      = effective_device(fit_clip_device,
                                                          sd_ctx_params->clip_backend_device);
        const char* vae_dev_name       = effective_device(fit_vae_device,
                                                          sd_ctx_params->vae_backend_device);

        // Build the row-split MultiBackendSpec for a component (ROW_SPLIT
        // mode). Unlike layer-split, the runner uses a SINGLE CUDA backend;
        // matmul weights are row-split across all CUDA devices internally
        // by cuda_split_buffer_type. extra_backends stays empty.
        // - share_devices/share_bytes: per-device share order from auto-fit
        //   (largest first by descending share). The first device is the
        //   "main" CUDA device, where the compute buffer lives.
        // Returns true on success; populates out_spec.tensor_split_ratios
        // with a vector of length total CUDA device count.
        auto prepare_row_split_spec = [&](const std::vector<std::string>&     share_devices,
                                          const std::vector<int64_t>&         share_bytes,
                                          std::vector<ggml_backend_t>&        out_extra_backends,
                                          MultiBackendSpec&                   out_spec) -> bool {
            if (share_devices.size() < 2) return false;

            // Derive the backend registry from the device-name prefix (e.g.
            // "CUDA0" -> reg "CUDA", "SYCL1" -> reg "SYCL"). This keeps the
            // code backend-agnostic: any backend whose registry publishes
            // `ggml_backend_split_buffer_type` via reg_get_proc_address can
            // drive row-split, not just CUDA.
            auto reg_prefix_of = [](const std::string& name) -> std::string {
                size_t i = 0;
                while (i < name.size() && (std::isalpha((unsigned char)name[i]) || name[i] == '_')) i++;
                return name.substr(0, i);
            };
            const std::string reg_name = reg_prefix_of(share_devices[0]);
            ggml_backend_reg_t reg     = ggml_backend_reg_by_name(reg_name.c_str());
            if (reg == nullptr) return false;
            const int dev_count = (int)ggml_backend_reg_dev_count(reg);
            if (dev_count <= 0) return false;

            auto reg_index_of = [&](const std::string& name) -> int {
                if (name.rfind(reg_name, 0) != 0) return -1;
                try { return std::stoi(name.substr(reg_name.size())); } catch (...) { return -1; }
            };

            std::vector<float> ratios(dev_count, 0.0f);
            int64_t total = 0;
            for (auto b : share_bytes) total += b;
            if (total <= 0) return false;
            int main_dev = -1;
            int64_t max_share = -1;
            for (size_t k = 0; k < share_devices.size(); k++) {
                int idx = reg_index_of(share_devices[k]);
                if (idx < 0 || idx >= dev_count) continue;
                ratios[idx] = float(double(share_bytes[k]) / double(total));
                if (share_bytes[k] > max_share) {
                    max_share = share_bytes[k];
                    main_dev  = idx;
                }
            }
            if (main_dev < 0) return false;

            // Init extra backends for the non-main devices so sched can
            // route ops across them (row-split tensors are dispatched by the
            // primary backend; ggml-sched still needs all participating
            // backends in its list to schedule cross-device copies).
            for (size_t k = 0; k < share_devices.size(); k++) {
                int idx = reg_index_of(share_devices[k]);
                if (idx == main_dev || idx < 0) continue;
                ggml_backend_t b = init_named_backend(share_devices[k]);
                if (b != nullptr) {
                    out_extra_backends.push_back(b);
                } else {
                    LOG_WARN("row-split: failed to init backend %s",
                             share_devices[k].c_str());
                }
            }
            out_spec.mode                = MultiBackendMode::ROW_SPLIT;
            out_spec.tensor_split_ratios = ratios;
            out_spec.main_device         = main_dev;
            out_spec.additional_backends.assign(out_extra_backends.begin(),
                                                out_extra_backends.end());
            out_spec.tensor_backend_fn   = nullptr;
            out_spec.cpu_fallback        = nullptr;

            std::string ratio_str;
            for (int i = 0; i < dev_count; i++) {
                if (i > 0) ratio_str += ",";
                char buf[16]; std::snprintf(buf, sizeof(buf), "%.2f", ratios[i]);
                ratio_str += buf;
            }
            LOG_INFO("row-split spec: ratios=[%s] main_device=%d",
                     ratio_str.c_str(), main_dev);
            return true;
        };

        // Build the layer-split MultiBackendSpec for a component. Only used
        // when auto-fit picked GPU_LAYER_SPLIT for this component.
        // - main_backend: the runner's primary backend (also first in the spec)
        // - extra_device_names: additional device names to span
        // - share_bytes: per-device share (for proportional block partition)
        // - tensor_prefix: the model's weight name prefix (e.g.,
        //   "model.diffusion_model.") — used to locate block-indexed tensors
        // Returns true if a spec was prepared and pending_spec_storage was
        // populated; the caller must set g_pending_multi_backend_spec()
        // immediately before constructing the model.
        auto prepare_layer_split_spec = [&](ggml_backend_t                       main_backend,
                                            const std::vector<std::string>&      extra_device_names,
                                            const std::vector<int64_t>&          share_bytes,
                                            const std::string&                   tensor_prefix,
                                            std::vector<ggml_backend_t>&         out_extra_backends,
                                            MultiBackendSpec&                    out_spec) -> bool {
            if (extra_device_names.size() < 2) return false;  // only [main] -> single GPU
            // Init the additional backends (skip [0] which is main_backend).
            std::vector<ggml_backend_t> all_backends;
            all_backends.push_back(main_backend);
            for (size_t k = 1; k < extra_device_names.size(); k++) {
                ggml_backend_t b = init_named_backend(extra_device_names[k]);
                if (b == nullptr) {
                    LOG_WARN("layer-split: failed to init extra backend %s; falling back to single backend",
                             extra_device_names[k].c_str());
                    return false;
                }
                out_extra_backends.push_back(b);
                all_backends.push_back(b);
            }

            // Walk tensor_storage_map to get per-block byte sizes and the
            // total non-block bytes that will land on backend[0]. Then
            // greedy-partition blocks by byte budget to balance per-backend
            // bytes (accounting for non-block fixed load on backend[0]).
            int max_block_idx = -1;
            static const std::regex block_re(
                R"((?:transformer_blocks|joint_blocks|double_blocks|single_blocks|blocks|layers)\.([0-9]+)\.)");
            std::map<int, int64_t> block_bytes;        // block idx -> bytes
            int64_t                non_block_bytes = 0;
            for (const auto& kv : tensor_storage_map) {
                if (!tensor_prefix.empty() && kv.first.compare(0, tensor_prefix.size(), tensor_prefix) != 0) {
                    continue;
                }
                int64_t bytes = (int64_t)kv.second.nbytes();
                std::smatch m;
                if (std::regex_search(kv.first, m, block_re)) {
                    int idx = std::stoi(m[1]);
                    if (idx > max_block_idx) max_block_idx = idx;
                    block_bytes[idx] += bytes;
                } else {
                    non_block_bytes += bytes;
                }
            }
            if (max_block_idx < 0) {
                LOG_WARN("layer-split: no blocks found under prefix '%s'; aborting split",
                         tensor_prefix.c_str());
                return false;
            }
            const int n_blocks = max_block_idx + 1;

            // Build per-backend byte budgets from share_bytes (ratios). The
            // first backend absorbs `non_block_bytes` as a fixed load, so we
            // SHRINK its remaining budget for blocks accordingly.
            int64_t total_share = 0;
            for (auto s : share_bytes) total_share += s;
            int64_t total_block_bytes = 0;
            for (const auto& kv : block_bytes) total_block_bytes += kv.second;
            std::vector<int64_t> backend_block_budgets(share_bytes.size(), 0);
            for (size_t k = 0; k < share_bytes.size(); k++) {
                int64_t share = int64_t(double(total_block_bytes + non_block_bytes) *
                                        double(share_bytes[k]) / double(total_share));
                if (k == 0) share = std::max<int64_t>(share - non_block_bytes, 0);
                backend_block_budgets[k] = share;
            }
            // Greedy assign each block (in order) to the current backend
            // until its budget is filled, then move to the next.
            std::vector<int> boundaries(share_bytes.size(), 0);
            size_t            cur_backend = 0;
            int64_t           cur_used    = 0;
            for (int b = 0; b < n_blocks; b++) {
                int64_t bb = block_bytes[b];
                if (cur_backend + 1 < share_bytes.size() &&
                    cur_used + bb > backend_block_budgets[cur_backend] &&
                    cur_used > 0) {
                    boundaries[cur_backend] = b;
                    cur_backend++;
                    cur_used = 0;
                }
                cur_used += bb;
            }
            // The remaining backends get the rest, terminating at n_blocks.
            for (size_t k = cur_backend; k < boundaries.size(); k++) {
                boundaries[k] = n_blocks;
            }
            // Safety: ensure each backend has at least one block.
            for (size_t k = 0; k < boundaries.size(); k++) {
                int min_bound = (k > 0 ? boundaries[k - 1] : 0) + 1;
                if (boundaries[k] < min_bound) boundaries[k] = std::min(min_bound, n_blocks);
            }
            std::string boundary_log = "layer-split [" + tensor_prefix + "] " +
                                       std::to_string(n_blocks) + " blocks: ";
            int prev = 0;
            for (size_t k = 0; k < all_backends.size() && k < boundaries.size(); k++) {
                if (k > 0) boundary_log += ", ";
                boundary_log += std::string(ggml_backend_name(all_backends[k])) + "=[" +
                                std::to_string(prev) + ".." + std::to_string(boundaries[k]) + ")";
                prev = boundaries[k];
            }
            LOG_INFO("%s", boundary_log.c_str());

            // Build the tensor_backend_fn closure.
            std::vector<ggml_backend_t> backends_capture = all_backends;
            std::vector<int>            boundaries_capture = boundaries;
            std::string                 prefix_capture     = tensor_prefix;
            out_spec.tensor_backend_fn =
                [backends_capture, boundaries_capture, prefix_capture](const std::string& name) -> ggml_backend_t {
                    if (!prefix_capture.empty() &&
                        name.compare(0, prefix_capture.size(), prefix_capture) != 0) {
                        return backends_capture[0];
                    }
                    std::smatch m;
                    if (!std::regex_search(name, m, block_re)) {
                        return backends_capture[0];
                    }
                    int idx = std::stoi(m[1]);
                    for (size_t k = 0; k < boundaries_capture.size(); k++) {
                        if (idx < boundaries_capture[k]) {
                            return backends_capture[std::min(k, backends_capture.size() - 1)];
                        }
                    }
                    return backends_capture.back();
                };
            // Spec contains the additional backends only (main is implicit).
            out_spec.additional_backends.assign(out_extra_backends.begin(), out_extra_backends.end());
            out_spec.cpu_fallback = nullptr;
            return true;
        };

        // Helper: init a named backend if name is non-null/non-empty,
        // returns nullptr on missing/failed name (caller falls back to main).
        auto init_named_or_null = [](const char* name) -> ggml_backend_t {
            if (name == nullptr || name[0] == '\0') return nullptr;
            return init_named_backend(name);
        };

        diffusion_backend = init_named_or_null(diffusion_dev_name);
        if (!diffusion_backend) {
            diffusion_backend = backend;
        } else {
            LOG_INFO("Diffusion model: using device %s", diffusion_dev_name);
        }

        // Tensor name sets for components that are configured for lazy load.
        // Populated below right before/after the cond + DiT construction;
        // consumed by the bulk-load step's ignore_tensors.
        std::set<std::string> cond_lazy_tensor_names;
        std::set<std::string> dit_lazy_tensor_names;

        // Build the layer-split MultiBackendSpec for DiT (when auto-fit picked
        // GPU_LAYER_SPLIT). The spec is consumed by the diffusion_model's
        // GGMLRunner ctor when we set g_pending_multi_backend_spec() to it.
        MultiBackendSpec dit_spec;
        bool             dit_spec_active = false;
        if (!fit_dit_split_device_names.empty()) {
            if (fit_dit_row_split) {
                dit_spec_active = prepare_row_split_spec(fit_dit_split_device_names,
                                                         fit_dit_split_share_bytes,
                                                         fit_dit_extra_backends,
                                                         dit_spec);
            } else {
                dit_spec_active = prepare_layer_split_spec(diffusion_backend,
                                                           fit_dit_split_device_names,
                                                           fit_dit_split_share_bytes,
                                                           "model.diffusion_model.",
                                                           fit_dit_extra_backends,
                                                           dit_spec);
            }
        }
        // Lambda to set the pending spec immediately before constructing the
        // diffusion model. Caller must invoke this on the same line / right
        // before the std::make_shared<...Model>(diffusion_backend, ...) call.
        auto prime_dit_spec = [&]() {
            if (dit_spec_active) {
                g_pending_multi_backend_spec() = &dit_spec;
            }
        };

        // Same dance for the conditioner. The conditioner uses clip_backend as
        // its main backend; we need to set up the spec BEFORE the cond_stage
        // ctor runs (which is BEFORE the DiT ctor). Each cond model wraps one
        // or more sub-runners; the spec's tensor_backend_fn handles all of
        // them since it's keyed on tensor names with a generic block regex.
        // (Some conditioners construct multiple sub-runners — only the FIRST
        // ggml runner ctor consumes the pending spec, so we re-prime between
        // sub-runners' allocs by leaving cond_spec_active true; the runner's
        // multi_backend_mode is per-runner.)
        // For LTX-2 specifically: LTXAVEmbedder constructs LLMRunner first
        // (consumes spec), then LTXAVTextProjectionRunner (no spec consumed).
        // The LLM has block-named tensors so layer-split applies; the
        // projector has only 4 tensors and they should ride along on its
        // single backend (clip_backend = main). Auto-fit's cond share counts
        // both, so the share is over-counted on backend[0] for the projector.
        // Acceptable for now — small correction.
        ggml_backend_t   clip_main_backend_for_spec = nullptr;  // resolved below
        MultiBackendSpec cond_spec;
        bool             cond_spec_active = false;
        auto prime_cond_spec = [&]() {
            if (cond_spec_active) {
                g_pending_multi_backend_spec() = &cond_spec;
            }
        };

        {
            clip_backend = init_named_or_null(clip_dev_name);
            if (!clip_backend) {
                clip_backend = backend;
            } else {
                LOG_INFO("CLIP: using device %s", clip_dev_name);
            }
            // Now that clip_backend is resolved, build the conditioner's
            // multi-GPU spec if auto-fit picked one (row-split or layer-split).
            if (!fit_cond_split_device_names.empty()) {
                if (fit_cond_row_split) {
                    cond_spec_active = prepare_row_split_spec(fit_cond_split_device_names,
                                                              fit_cond_split_share_bytes,
                                                              fit_cond_extra_backends,
                                                              cond_spec);
                } else {
                    cond_spec_active = prepare_layer_split_spec(clip_backend,
                                                                fit_cond_split_device_names,
                                                                fit_cond_split_share_bytes,
                                                                "text_encoders.",
                                                                fit_cond_extra_backends,
                                                                cond_spec);
                }
            }
            if (sd_version_is_sd3(version)) {
                prime_cond_spec();
                cond_stage_model = std::make_shared<SD3CLIPEmbedder>(clip_backend,
                                                                     offload_params_to_cpu,
                                                                     tensor_storage_map);
                prime_dit_spec();
                diffusion_model  = std::make_shared<MMDiTModel>(diffusion_backend,
                                                               offload_params_to_cpu,
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

                    prime_cond_spec();
                    cond_stage_model = std::make_shared<T5CLIPEmbedder>(clip_backend,
                                                                        offload_params_to_cpu,
                                                                        tensor_storage_map,
                                                                        sd_ctx_params->chroma_use_t5_mask,
                                                                        sd_ctx_params->chroma_t5_mask_pad);
                } else if (version == VERSION_OVIS_IMAGE) {
                    prime_cond_spec();
                    cond_stage_model = std::make_shared<LLMEmbedder>(clip_backend,
                                                                     offload_params_to_cpu,
                                                                     tensor_storage_map,
                                                                     version,
                                                                     "",
                                                                     false);
                } else {
                    prime_cond_spec();
                    cond_stage_model = std::make_shared<FluxCLIPEmbedder>(clip_backend,
                                                                          offload_params_to_cpu,
                                                                          tensor_storage_map);
                }
                prime_dit_spec();
                diffusion_model = std::make_shared<FluxModel>(diffusion_backend,
                                                              offload_params_to_cpu,
                                                              tensor_storage_map,
                                                              version,
                                                              sd_ctx_params->chroma_use_dit_mask);
            } else if (sd_version_is_flux2(version)) {
                bool is_chroma   = false;
                prime_cond_spec();
                cond_stage_model = std::make_shared<LLMEmbedder>(clip_backend,
                                                                 offload_params_to_cpu,
                                                                 tensor_storage_map,
                                                                 version);
                prime_dit_spec();
                diffusion_model  = std::make_shared<FluxModel>(diffusion_backend,
                                                              offload_params_to_cpu,
                                                              tensor_storage_map,
                                                              version,
                                                              sd_ctx_params->chroma_use_dit_mask);
            } else if (sd_version_is_wan(version)) {
                prime_cond_spec();
                cond_stage_model = std::make_shared<T5CLIPEmbedder>(clip_backend,
                                                                    offload_params_to_cpu,
                                                                    tensor_storage_map,
                                                                    true,
                                                                    0,
                                                                    true);
                prime_dit_spec();
                diffusion_model  = std::make_shared<WanModel>(diffusion_backend,
                                                             offload_params_to_cpu,
                                                             tensor_storage_map,
                                                             "model.diffusion_model",
                                                             version);
                if (strlen(SAFE_STR(sd_ctx_params->high_noise_diffusion_model_path)) > 0) {
                    prime_dit_spec();
                    high_noise_diffusion_model = std::make_shared<WanModel>(diffusion_backend,
                                                                            offload_params_to_cpu,
                                                                            tensor_storage_map,
                                                                            "model.high_noise_diffusion_model",
                                                                            version);
                }
                if (diffusion_model->get_desc() == "Wan2.1-I2V-14B" ||
                    diffusion_model->get_desc() == "Wan2.1-FLF2V-14B" ||
                    diffusion_model->get_desc() == "Wan2.1-I2V-1.3B") {
                    clip_vision = std::make_shared<FrozenCLIPVisionEmbedder>(backend,
                                                                             offload_params_to_cpu,
                                                                             tensor_storage_map);
                    clip_vision->alloc_params_buffer();
                    clip_vision->get_param_tensors(tensors);
                }
            } else if (sd_version_is_qwen_image(version)) {
                bool enable_vision = false;
                if (!vae_decode_only) {
                    enable_vision = true;
                }
                prime_cond_spec();
                cond_stage_model = std::make_shared<LLMEmbedder>(clip_backend,
                                                                 offload_params_to_cpu,
                                                                 tensor_storage_map,
                                                                 version,
                                                                 "",
                                                                 enable_vision);
                prime_dit_spec();
                diffusion_model  = std::make_shared<QwenImageModel>(diffusion_backend,
                                                                   offload_params_to_cpu,
                                                                   tensor_storage_map,
                                                                   "model.diffusion_model",
                                                                   version,
                                                                   sd_ctx_params->qwen_image_zero_cond_t);
            } else if (sd_version_is_anima(version)) {
                prime_cond_spec();
                cond_stage_model = std::make_shared<AnimaConditioner>(clip_backend,
                                                                      offload_params_to_cpu,
                                                                      tensor_storage_map);
                prime_dit_spec();
                diffusion_model  = std::make_shared<AnimaModel>(diffusion_backend,
                                                               offload_params_to_cpu,
                                                               tensor_storage_map,
                                                               "model.diffusion_model");
            } else if (sd_version_is_z_image(version)) {
                prime_cond_spec();
                cond_stage_model = std::make_shared<LLMEmbedder>(clip_backend,
                                                                 offload_params_to_cpu,
                                                                 tensor_storage_map,
                                                                 version);
                prime_dit_spec();
                diffusion_model  = std::make_shared<ZImageModel>(diffusion_backend,
                                                                offload_params_to_cpu,
                                                                tensor_storage_map,
                                                                "model.diffusion_model",
                                                                version);
            } else if (sd_version_is_ernie_image(version)) {
                prime_cond_spec();
                cond_stage_model = std::make_shared<LLMEmbedder>(clip_backend,
                                                                 offload_params_to_cpu,
                                                                 tensor_storage_map,
                                                                 version);
                prime_dit_spec();
                diffusion_model  = std::make_shared<ErnieImageModel>(diffusion_backend,
                                                                    offload_params_to_cpu,
                                                                    tensor_storage_map,
                                                                    "model.diffusion_model");
            } else {  // SD1.x SD2.x SDXL
                std::map<std::string, std::string> embbeding_map;
                for (uint32_t i = 0; i < sd_ctx_params->embedding_count; i++) {
                    embbeding_map.emplace(SAFE_STR(sd_ctx_params->embeddings[i].name), SAFE_STR(sd_ctx_params->embeddings[i].path));
                }
                if (strstr(SAFE_STR(sd_ctx_params->photo_maker_path), "v2")) {
                    prime_cond_spec();
                    cond_stage_model = std::make_shared<FrozenCLIPEmbedderWithCustomWords>(clip_backend,
                                                                                           offload_params_to_cpu,
                                                                                           tensor_storage_map,
                                                                                           embbeding_map,
                                                                                           version,
                                                                                           PM_VERSION_2);
                } else {
                    prime_cond_spec();
                    cond_stage_model = std::make_shared<FrozenCLIPEmbedderWithCustomWords>(clip_backend,
                                                                                           offload_params_to_cpu,
                                                                                           tensor_storage_map,
                                                                                           embbeding_map,
                                                                                           version);
                }
                prime_dit_spec();
                diffusion_model = std::make_shared<UNetModel>(diffusion_backend,
                                                              offload_params_to_cpu,
                                                              tensor_storage_map,
                                                              version);
                if (sd_ctx_params->diffusion_conv_direct) {
                    LOG_INFO("Using Conv2d direct in the diffusion model");
                    std::dynamic_pointer_cast<UNetModel>(diffusion_model)->unet.set_conv2d_direct_enabled(true);
                }
            }

            // Conditioner: publish its tensors to the global map, EXCEPT the
            // ones that are about to be configured for lazy load (we want the
            // bulk loader to skip them — they have no buffer yet).
            std::map<std::string, ggml_tensor*> cond_only_tensors;
            cond_stage_model->get_param_tensors(cond_only_tensors);
            std::map<std::string, ggml_tensor*> llm_lazy_map;
            if (auto_lazy_load) {
                for (const auto& kv : cond_only_tensors) {
                    if (kv.first.rfind("text_encoders.llm.", 0) == 0) {
                        llm_lazy_map[kv.first] = kv.second;
                        cond_lazy_tensor_names.insert(kv.first);
                    }
                }
            }
            for (const auto& kv : cond_only_tensors) {
                if (cond_lazy_tensor_names.find(kv.first) == cond_lazy_tensor_names.end()) {
                    tensors[kv.first] = kv.second;  // eager — bulk loader will fill
                }
            }
            if (auto_lazy_load && !llm_lazy_map.empty()) {
                ModelLoader* loader_ptr        = owned_model_loader.get();
                // Bound lazy-load threads to keep the per-thread staging
                // buffer footprint small. The default n_threads = nproc gives
                // ~nproc × max_tensor_bytes (up to several GB total) of
                // CPU-side staging; for RAM-constrained systems running large
                // models that's enough to trigger the OOM-killer even with
                // mmap enabled. 2 threads still keep the disk-read pipeline
                // fed while keeping staging bounded to ~2 × max_tensor_bytes.
                int          n_threads_capture = std::min(sd_ctx_params->n_threads > 0
                                                             ? sd_ctx_params->n_threads : 2,
                                                          2);
                bool         mmap_capture      = sd_ctx_params->enable_mmap;
                bool         quiet_capture     = sd_ctx_params->quiet_unknown_tensors;
                cond_stage_model->set_llm_lazy_load([=]() -> bool {
                    auto local_map = llm_lazy_map;
                    return loader_ptr->load_tensors(local_map, /*ignore=*/{},
                                                    n_threads_capture, mmap_capture,
                                                    quiet_capture);
                });
                LOG_INFO("auto-fit: conditioner LLM is lazy (defer alloc until first compute, %zu tensors)",
                         llm_lazy_map.size());
            }
            cond_stage_model->alloc_params_buffer();  // no-op for the lazy LLM

            std::map<std::string, ggml_tensor*> dit_only_tensors;
            diffusion_model->get_param_tensors(dit_only_tensors);
            if (auto_lazy_load) {
                for (const auto& kv : dit_only_tensors) {
                    dit_lazy_tensor_names.insert(kv.first);
                }
                ModelLoader* loader_ptr        = owned_model_loader.get();
                // Bound lazy-load threads to keep the per-thread staging
                // buffer footprint small. The default n_threads = nproc gives
                // ~nproc × max_tensor_bytes (up to several GB total) of
                // CPU-side staging; for RAM-constrained systems running large
                // models that's enough to trigger the OOM-killer even with
                // mmap enabled. 2 threads still keep the disk-read pipeline
                // fed while keeping staging bounded to ~2 × max_tensor_bytes.
                int          n_threads_capture = std::min(sd_ctx_params->n_threads > 0
                                                             ? sd_ctx_params->n_threads : 2,
                                                          2);
                bool         mmap_capture      = sd_ctx_params->enable_mmap;
                bool         quiet_capture     = sd_ctx_params->quiet_unknown_tensors;
                diffusion_model->set_lazy_load([=]() -> bool {
                    auto local_map = dit_only_tensors;
                    return loader_ptr->load_tensors(local_map, /*ignore=*/{},
                                                    n_threads_capture, mmap_capture,
                                                    quiet_capture);
                });
                LOG_INFO("auto-fit: diffusion_model is lazy (defer alloc until first compute, %zu tensors)",
                         dit_only_tensors.size());
            } else {
                for (const auto& kv : dit_only_tensors) {
                    tensors[kv.first] = kv.second;
                }
            }
            diffusion_model->alloc_params_buffer();  // no-op when lazy_load_fn is set

            if (sd_version_is_unet_edit(version)) {
                vae_decode_only = false;
            }

            if (high_noise_diffusion_model) {
                high_noise_diffusion_model->alloc_params_buffer();
                high_noise_diffusion_model->get_param_tensors(tensors);
            }

            if (vae_dev_name != nullptr && vae_dev_name[0] != '\0') {
                vae_backend = init_named_backend(vae_dev_name);
            }
            if (!vae_backend) {
                vae_backend = backend;
            } else {
                LOG_INFO("VAE: using device %s", vae_dev_name);
            }

            auto create_tae = [&]() -> std::shared_ptr<VAE> {
                if (sd_version_is_wan(version) ||
                    sd_version_is_qwen_image(version) ||
                    sd_version_is_anima(version)) {
                    return std::make_shared<TinyVideoAutoEncoder>(vae_backend,
                                                                  offload_params_to_cpu,
                                                                  tensor_storage_map,
                                                                  "decoder",
                                                                  vae_decode_only,
                                                                  version);

                } else {
                    auto model = std::make_shared<TinyImageAutoEncoder>(vae_backend,
                                                                        offload_params_to_cpu,
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
                                                               offload_params_to_cpu,
                                                               tensor_storage_map,
                                                               "first_stage_model",
                                                               vae_decode_only,
                                                               version);
                } else {
                    auto model = std::make_shared<AutoEncoderKL>(vae_backend,
                                                                 offload_params_to_cpu,
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
                                                              offload_params_to_cpu);
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
                const char* cn_dev_name = sd_ctx_params->control_net_backend_device;
                if (cn_dev_name != nullptr && cn_dev_name[0] != '\0') {
                    controlnet_backend = init_named_backend(cn_dev_name);
                }
                if (!controlnet_backend) {
                    controlnet_backend = backend;
                } else {
                    LOG_INFO("ControlNet: using device %s", cn_dev_name);
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
                                                                   offload_params_to_cpu,
                                                                   tensor_storage_map,
                                                                   "pmid",
                                                                   version,
                                                                   PM_VERSION_2);
                LOG_INFO("using PhotoMaker Version 2");
            } else {
                pmid_model = std::make_shared<PhotoMakerIDEncoder>(backend,
                                                                   offload_params_to_cpu,
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
        // Lazy-loaded components: skip them in the bulk load; their lazy
        // callbacks will load them on first compute().
        for (const auto& name : cond_lazy_tensor_names) {
            ignore_tensors.insert(name);
        }
        for (const auto& name : dit_lazy_tensor_names) {
            ignore_tensors.insert(name);
        }
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
        bool success = model_loader.load_tensors(tensors, ignore_tensors, n_threads,
                                                  sd_ctx_params->enable_mmap,
                                                  sd_ctx_params->quiet_unknown_tensors);
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

            compute_sample_controls(control_image,
                                    noised_input,
                                    timesteps_tensor,
                                    cond,
                                    &controls);

            auto run_condition = [&](const SDCondition& condition,
                                     const sd::Tensor<float>* c_concat_override = nullptr,
                                     const std::vector<int>* local_skip_layers  = nullptr) -> sd::Tensor<float> {
                diffusion_params.context     = condition.c_crossattn.empty() ? nullptr : &condition.c_crossattn;
                diffusion_params.c_concat    = c_concat_override != nullptr ? c_concat_override : (condition.c_concat.empty() ? nullptr : &condition.c_concat);
                diffusion_params.y           = condition.c_vector.empty() ? nullptr : &condition.c_vector;
                diffusion_params.t5_ids      = condition.c_t5_ids.empty() ? nullptr : &condition.c_t5_ids;
                diffusion_params.t5_weights  = condition.c_t5_weights.empty() ? nullptr : &condition.c_t5_weights;
                diffusion_params.skip_layers = local_skip_layers;

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
            denoised = latent_result * c_out + x * c_skip;
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
            } else {
                latent_channel = 16;
            }
        }
        return latent_channel;
    }

    int get_image_seq_len(int h, int w) {
        int vae_scale_factor = get_vae_scale_factor();
        return (h / vae_scale_factor) * (w / vae_scale_factor);
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

const char* hires_upscaler_to_str[] = {
    "None",
    "Latent",
    "Latent (nearest)",
    "Latent (nearest-exact)",
    "Latent (antialiased)",
    "Latent (bicubic)",
    "Latent (bicubic antialiased)",
    "Lanczos",
    "Nearest",
    "Model",
};

const char* sd_hires_upscaler_name(enum sd_hires_upscaler_t upscaler) {
    if (upscaler >= SD_HIRES_UPSCALER_NONE && upscaler < SD_HIRES_UPSCALER_COUNT) {
        return hires_upscaler_to_str[upscaler];
    }
    return NONE_STR;
}

enum sd_hires_upscaler_t str_to_sd_hires_upscaler(const char* str) {
    for (int i = 0; i < SD_HIRES_UPSCALER_COUNT; i++) {
        if (!strcmp(str, hires_upscaler_to_str[i])) {
            return (enum sd_hires_upscaler_t)i;
        }
    }
    return SD_HIRES_UPSCALER_COUNT;
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

void sd_hires_params_init(sd_hires_params_t* hires_params) {
    *hires_params                    = {};
    hires_params->enabled            = false;
    hires_params->upscaler           = SD_HIRES_UPSCALER_LATENT;
    hires_params->model_path         = nullptr;
    hires_params->scale              = 2.0f;
    hires_params->target_width       = 0;
    hires_params->target_height      = 0;
    hires_params->steps              = 0;
    hires_params->denoising_strength = 0.7f;
    hires_params->upscale_tile_size  = 128;
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
    sd_ctx_params->enable_mmap                  = false;
    sd_ctx_params->main_backend_device          = nullptr;
    sd_ctx_params->diffusion_backend_device     = nullptr;
    sd_ctx_params->clip_backend_device          = nullptr;
    sd_ctx_params->vae_backend_device           = nullptr;
    sd_ctx_params->control_net_backend_device   = nullptr;
    sd_ctx_params->tae_backend_device           = nullptr;
    sd_ctx_params->upscaler_backend_device      = nullptr;
    sd_ctx_params->photomaker_backend_device    = nullptr;
    sd_ctx_params->vision_backend_device        = nullptr;
    sd_ctx_params->diffusion_flash_attn         = false;
    sd_ctx_params->circular_x                   = false;
    sd_ctx_params->circular_y                   = false;
    sd_ctx_params->chroma_use_dit_mask          = true;
    sd_ctx_params->chroma_use_t5_mask           = false;
    sd_ctx_params->chroma_t5_mask_pad           = 1;
    sd_ctx_params->auto_fit                     = true;
    sd_ctx_params->auto_fit_target_mb           = 512;
    sd_ctx_params->auto_fit_dry_run             = false;
    sd_ctx_params->auto_fit_compute_reserve_dit_mb  = 0;
    sd_ctx_params->auto_fit_compute_reserve_vae_mb  = 0;
    sd_ctx_params->auto_fit_compute_reserve_cond_mb = 0;
    sd_ctx_params->auto_multi_gpu               = true;
    sd_ctx_params->multi_gpu_mode               = "row";
    sd_ctx_params->quiet_unknown_tensors        = false;
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
             "main_backend_device: %s\n"
             "diffusion_backend_device: %s\n"
             "clip_backend_device: %s\n"
             "vae_backend_device: %s\n"
             "control_net_backend_device: %s\n"
             "tae_backend_device: %s\n"
             "upscaler_backend_device: %s\n"
             "photomaker_backend_device: %s\n"
             "vision_backend_device: %s\n"
             "auto_fit: %s\n"
             "auto_fit_target_mb: %d\n"
             "auto_fit_dry_run: %s\n"
             "auto_fit_compute_reserve_dit_mb: %d\n"
             "auto_fit_compute_reserve_vae_mb: %d\n"
             "auto_fit_compute_reserve_cond_mb: %d\n"
             "auto_multi_gpu: %s\n"
             "multi_gpu_mode: %s\n"
             "quiet_unknown_tensors: %s\n"
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
             SAFE_STR(sd_ctx_params->main_backend_device),
             SAFE_STR(sd_ctx_params->diffusion_backend_device),
             SAFE_STR(sd_ctx_params->clip_backend_device),
             SAFE_STR(sd_ctx_params->vae_backend_device),
             SAFE_STR(sd_ctx_params->control_net_backend_device),
             SAFE_STR(sd_ctx_params->tae_backend_device),
             SAFE_STR(sd_ctx_params->upscaler_backend_device),
             SAFE_STR(sd_ctx_params->photomaker_backend_device),
             SAFE_STR(sd_ctx_params->vision_backend_device),
             BOOL_STR(sd_ctx_params->auto_fit),
             sd_ctx_params->auto_fit_target_mb,
             BOOL_STR(sd_ctx_params->auto_fit_dry_run),
             sd_ctx_params->auto_fit_compute_reserve_dit_mb,
             sd_ctx_params->auto_fit_compute_reserve_vae_mb,
             sd_ctx_params->auto_fit_compute_reserve_cond_mb,
             BOOL_STR(sd_ctx_params->auto_multi_gpu),
             SAFE_STR(sd_ctx_params->multi_gpu_mode),
             BOOL_STR(sd_ctx_params->quiet_unknown_tensors),
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
    sd_hires_params_init(&sd_img_gen_params->hires);
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
             "VAE tiling: %s\n"
             "hires: {enabled=%s, upscaler=%s, model_path=%s, scale=%.2f, target=%dx%d, steps=%d, denoising_strength=%.2f}\n",
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
             BOOL_STR(sd_img_gen_params->vae_tiling_params.enabled),
             BOOL_STR(sd_img_gen_params->hires.enabled),
             sd_hires_upscaler_name(sd_img_gen_params->hires.upscaler),
             SAFE_STR(sd_img_gen_params->hires.model_path),
             sd_img_gen_params->hires.scale,
             sd_img_gen_params->hires.target_width,
             sd_img_gen_params->hires.target_height,
             sd_img_gen_params->hires.steps,
             sd_img_gen_params->hires.denoising_strength);
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
    sd_hires_params_t hires                  = {};
    int frames                               = -1;
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
        hires                       = sd_img_gen_params->hires;
        cache_params                = &sd_img_gen_params->cache;
        resolve(sd_ctx);
    }

    GenerationRequest(sd_ctx_t* sd_ctx, const sd_vid_gen_params_t* sd_vid_gen_params) {
        prompt                      = SAFE_STR(sd_vid_gen_params->prompt);
        negative_prompt             = SAFE_STR(sd_vid_gen_params->negative_prompt);
        width                       = sd_vid_gen_params->width;
        height                      = sd_vid_gen_params->height;
        frames                      = (sd_vid_gen_params->video_frames - 1) / 4 * 4 + 1;
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
        align_image_size(&width, &height, "generation request");
    }

    void align_image_size(int* target_width, int* target_height, const char* label) {
        int spatial_multiple = vae_scale_factor * diffusion_model_down_factor;
        int width_offset     = align_up_offset(*target_width, spatial_multiple);
        int height_offset    = align_up_offset(*target_height, spatial_multiple);
        if (width_offset <= 0 && height_offset <= 0) {
            return;
        }

        int original_width  = *target_width;
        int original_height = *target_height;

        *target_width += width_offset;
        *target_height += height_offset;
        LOG_WARN("align %s up %dx%d to %dx%d (multiple=%d)",
                 label,
                 original_width,
                 original_height,
                 *target_width,
                 *target_height,
                 spatial_multiple);
    }

    void resolve_hires() {
        if (!hires.enabled) {
            return;
        }
        if (hires.upscaler == SD_HIRES_UPSCALER_NONE) {
            hires.enabled = false;
            return;
        }
        if (hires.upscaler < SD_HIRES_UPSCALER_NONE || hires.upscaler >= SD_HIRES_UPSCALER_COUNT) {
            LOG_WARN("hires upscaler '%d' is invalid, disabling hires", hires.upscaler);
            hires.enabled = false;
            return;
        }
        if (hires.upscaler == SD_HIRES_UPSCALER_MODEL && strlen(SAFE_STR(hires.model_path)) == 0) {
            LOG_WARN("hires model upscaler requires a model path, disabling hires");
            hires.enabled = false;
            return;
        }
        if (hires.scale <= 0.f && hires.target_width <= 0 && hires.target_height <= 0) {
            LOG_WARN("hires scale must be positive when no target size is set, disabling hires");
            hires.enabled = false;
            return;
        }
        hires.denoising_strength = std::clamp(hires.denoising_strength, 0.0001f, 1.f);
        hires.steps              = std::max(0, hires.steps);

        if (hires.target_width > 0 && hires.target_height > 0) {
            // pass
        } else if (hires.target_width > 0) {
            hires.target_height = hires.target_width;
        } else if (hires.target_height > 0) {
            hires.target_width = hires.target_height;
        } else {
            hires.target_width  = static_cast<int>(std::round(width * hires.scale));
            hires.target_height = static_cast<int>(std::round(height * hires.scale));
        }

        if (hires.target_width <= 0 || hires.target_height <= 0) {
            LOG_WARN("hires target size is not positive, disabling hires");
            hires.enabled = false;
            return;
        }
        align_image_size(&hires.target_width, &hires.target_height, "hires target");
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
        resolve_hires();
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
                                                                     sd_ctx->sd->get_image_seq_len(request->height, request->width),
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
        LOG_INFO("latent %zu decoded, taking %.2fs", i + 1, (t2 - t1) * 1.0f / 1000);
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

static sd::Tensor<float> upscale_hires_latent(sd_ctx_t* sd_ctx,
                                              const sd::Tensor<float>& latent,
                                              const GenerationRequest& request,
                                              UpscalerGGML* upscaler) {
    auto get_hires_latent_target_shape = [&]() {
        std::vector<int64_t> target_shape = latent.shape();
        if (target_shape.size() < 2) {
            target_shape.clear();
            return target_shape;
        }
        target_shape[0] = request.hires.target_width / request.vae_scale_factor;
        target_shape[1] = request.hires.target_height / request.vae_scale_factor;
        return target_shape;
    };

    if (request.hires.upscaler == SD_HIRES_UPSCALER_LATENT ||
        request.hires.upscaler == SD_HIRES_UPSCALER_LATENT_NEAREST ||
        request.hires.upscaler == SD_HIRES_UPSCALER_LATENT_NEAREST_EXACT ||
        request.hires.upscaler == SD_HIRES_UPSCALER_LATENT_ANTIALIASED ||
        request.hires.upscaler == SD_HIRES_UPSCALER_LATENT_BICUBIC ||
        request.hires.upscaler == SD_HIRES_UPSCALER_LATENT_BICUBIC_ANTIALIASED) {
        std::vector<int64_t> target_shape = get_hires_latent_target_shape();
        if (target_shape.empty()) {
            LOG_ERROR("latent has invalid shape for hires upscale");
            return {};
        }

        sd::ops::InterpolateMode mode = sd::ops::InterpolateMode::Nearest;
        bool antialias                = false;
        switch (request.hires.upscaler) {
            case SD_HIRES_UPSCALER_LATENT:
                mode = sd::ops::InterpolateMode::Bilinear;
                break;
            case SD_HIRES_UPSCALER_LATENT_NEAREST:
                mode = sd::ops::InterpolateMode::Nearest;
                break;
            case SD_HIRES_UPSCALER_LATENT_NEAREST_EXACT:
                mode = sd::ops::InterpolateMode::NearestExact;
                break;
            case SD_HIRES_UPSCALER_LATENT_ANTIALIASED:
                mode      = sd::ops::InterpolateMode::Bilinear;
                antialias = true;
                break;
            case SD_HIRES_UPSCALER_LATENT_BICUBIC:
                mode = sd::ops::InterpolateMode::Bicubic;
                break;
            case SD_HIRES_UPSCALER_LATENT_BICUBIC_ANTIALIASED:
                mode      = sd::ops::InterpolateMode::Bicubic;
                antialias = true;
                break;
            default:
                break;
        }

        LOG_INFO("hires %s upscale %" PRId64 "x%" PRId64 " -> %" PRId64 "x%" PRId64,
                 sd_hires_upscaler_name(request.hires.upscaler),
                 latent.shape()[0],
                 latent.shape()[1],
                 target_shape[0],
                 target_shape[1]);

        return sd::ops::interpolate(latent, target_shape, mode, false, antialias);
    } else if (request.hires.upscaler == SD_HIRES_UPSCALER_MODEL ||
               request.hires.upscaler == SD_HIRES_UPSCALER_LANCZOS ||
               request.hires.upscaler == SD_HIRES_UPSCALER_NEAREST) {
        if (sd_ctx->sd->vae_decode_only) {
            LOG_ERROR("hires %s upscaler requires VAE encoder weights; create the context with vae_decode_only=false",
                      sd_hires_upscaler_name(request.hires.upscaler));
            return {};
        }
        if (request.hires.upscaler == SD_HIRES_UPSCALER_MODEL && upscaler == nullptr) {
            LOG_ERROR("hires model upscaler context is null");
            return {};
        }

        sd::Tensor<float> decoded = sd_ctx->sd->decode_first_stage(latent);
        if (decoded.empty()) {
            LOG_ERROR("decode_first_stage failed before hires %s upscale",
                      sd_hires_upscaler_name(request.hires.upscaler));
            return {};
        }

        sd::Tensor<float> upscaled_tensor;
        if (request.hires.upscaler == SD_HIRES_UPSCALER_MODEL) {
            upscaled_tensor = upscaler->upscale_tensor(decoded);
            if (upscaled_tensor.empty()) {
                LOG_ERROR("hires model upscale failed");
                return {};
            }

            if (upscaled_tensor.shape()[0] != request.hires.target_width ||
                upscaled_tensor.shape()[1] != request.hires.target_height) {
                upscaled_tensor = sd::ops::interpolate(upscaled_tensor,
                                                       {request.hires.target_width,
                                                        request.hires.target_height,
                                                        upscaled_tensor.shape()[2],
                                                        upscaled_tensor.shape()[3]});
            }
        } else {
            sd::ops::InterpolateMode mode = request.hires.upscaler == SD_HIRES_UPSCALER_LANCZOS
                                                ? sd::ops::InterpolateMode::Lanczos
                                                : sd::ops::InterpolateMode::Nearest;
            LOG_INFO("hires %s image upscale %" PRId64 "x%" PRId64 " -> %dx%d",
                     sd_hires_upscaler_name(request.hires.upscaler),
                     decoded.shape()[0],
                     decoded.shape()[1],
                     request.hires.target_width,
                     request.hires.target_height);
            upscaled_tensor = sd::ops::interpolate(decoded,
                                                   {request.hires.target_width,
                                                    request.hires.target_height,
                                                    decoded.shape()[2],
                                                    decoded.shape()[3]},
                                                   mode);
            upscaled_tensor = sd::ops::clamp(upscaled_tensor, 0.0f, 1.0f);
        }

        sd::Tensor<float> upscaled_latent = sd_ctx->sd->encode_first_stage(upscaled_tensor);
        if (upscaled_latent.empty()) {
            LOG_ERROR("encode_first_stage failed after hires %s upscale",
                      sd_hires_upscaler_name(request.hires.upscaler));
        }
        return upscaled_latent;
    }

    LOG_ERROR("unsupported hires upscaler '%s'", sd_hires_upscaler_name(request.hires.upscaler));
    return {};
}

SD_API sd_image_t* generate_image(sd_ctx_t* sd_ctx, const sd_img_gen_params_t* sd_img_gen_params) {
    if (sd_ctx == nullptr || sd_img_gen_params == nullptr) {
        return nullptr;
    }

    int64_t t0                    = ggml_time_ms();
    sd_ctx->sd->vae_tiling_params = sd_img_gen_params->vae_tiling_params;
    GenerationRequest request(sd_ctx, sd_img_gen_params);
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
    if (sd_ctx->sd->free_params_immediately && !request.hires.enabled) {
        sd_ctx->sd->diffusion_model->free_params_buffer();
    }
    int64_t denoise_end = ggml_time_ms();
    LOG_INFO("generating %zu latent images completed, taking %.2fs",
             final_latents.size(),
             (denoise_end - denoise_start) * 1.0f / 1000);

    if (request.hires.enabled && request.hires.target_width > 0) {
        LOG_INFO("hires fix: upscaling to %dx%d", request.hires.target_width, request.hires.target_height);

        std::unique_ptr<UpscalerGGML> hires_upscaler;
        if (request.hires.upscaler == SD_HIRES_UPSCALER_MODEL) {
            LOG_INFO("hires fix: loading model upscaler from '%s'", request.hires.model_path);
            hires_upscaler = std::make_unique<UpscalerGGML>(sd_ctx->sd->n_threads,
                                                            false,
                                                            request.hires.upscale_tile_size);
            if (!hires_upscaler->load_from_file(request.hires.model_path,
                                                sd_ctx->sd->offload_params_to_cpu,
                                                sd_ctx->sd->n_threads)) {
                LOG_ERROR("load hires model upscaler failed");
                if (sd_ctx->sd->free_params_immediately) {
                    sd_ctx->sd->diffusion_model->free_params_buffer();
                }
                return nullptr;
            }
        }

        int hires_steps = request.hires.steps > 0 ? request.hires.steps : plan.sample_steps;

        // sd-webui behavior: scale up total steps so trimming by denoising_strength yields exactly hires_steps effective steps,
        // unlike img2img which trims from a fixed step count
        hires_steps = static_cast<int>(hires_steps / request.hires.denoising_strength);

        std::vector<float> hires_sigmas = sd_ctx->sd->denoiser->get_sigmas(
            hires_steps,
            sd_ctx->sd->get_image_seq_len(request.hires.target_height, request.hires.target_width),
            sd_img_gen_params->sample_params.scheduler,
            sd_ctx->sd->version);

        size_t t_enc = static_cast<size_t>(hires_steps * request.hires.denoising_strength);
        if (t_enc >= static_cast<size_t>(hires_steps)) {
            t_enc = static_cast<size_t>(hires_steps) - 1;
        }
        std::vector<float> hires_sigma_sched(hires_sigmas.begin() + hires_steps - static_cast<int>(t_enc) - 1,
                                             hires_sigmas.end());
        LOG_INFO("hires fix: %d steps, denoising_strength=%.2f, sigma_sched_size=%zu",
                 hires_steps,
                 request.hires.denoising_strength,
                 hires_sigma_sched.size());

        std::vector<sd::Tensor<float>> hires_final_latents;
        int64_t hires_denoise_start = ggml_time_ms();
        for (int b = 0; b < (int)final_latents.size(); b++) {
            int64_t cur_seed = request.seed + b;
            sd_ctx->sd->rng->manual_seed(cur_seed);
            sd_ctx->sd->sampler_rng->manual_seed(cur_seed);

            sd::Tensor<float> upscaled = upscale_hires_latent(sd_ctx,
                                                              final_latents[b],
                                                              request,
                                                              hires_upscaler.get());
            if (upscaled.empty()) {
                if (sd_ctx->sd->free_params_immediately) {
                    sd_ctx->sd->diffusion_model->free_params_buffer();
                }
                return nullptr;
            }

            sd::Tensor<float> noise = sd::randn_like<float>(upscaled, sd_ctx->sd->rng);

            sd::Tensor<float> hires_denoise_mask;
            if (!latents.denoise_mask.empty()) {
                std::vector<int64_t> mask_shape = latents.denoise_mask.shape();
                mask_shape[0]                   = upscaled.shape()[0];
                mask_shape[1]                   = upscaled.shape()[1];
                hires_denoise_mask              = sd::ops::interpolate(latents.denoise_mask,
                                                                       mask_shape,
                                                                       sd::ops::InterpolateMode::NearestMax);
            }

            int64_t hires_sample_start = ggml_time_ms();
            sd::Tensor<float> x_0      = sd_ctx->sd->sample(sd_ctx->sd->diffusion_model,
                                                            true,
                                                            upscaled,
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
                                                            hires_sigma_sched,
                                                            plan.start_merge_step,
                                                            latents.ref_latents,
                                                            request.increase_ref_index,
                                                            hires_denoise_mask,
                                                            sd::Tensor<float>(),
                                                            1.f,
                                                            request.cache_params);
            int64_t hires_sample_end   = ggml_time_ms();
            if (!x_0.empty()) {
                LOG_INFO("hires sampling %d/%d completed, taking %.2fs",
                         b + 1,
                         (int)final_latents.size(),
                         (hires_sample_end - hires_sample_start) * 1.0f / 1000);
                hires_final_latents.push_back(std::move(x_0));
                continue;
            }

            LOG_ERROR("hires sampling for image %d/%d failed after %.2fs",
                      b + 1,
                      (int)final_latents.size(),
                      (hires_sample_end - hires_sample_start) * 1.0f / 1000);
            if (sd_ctx->sd->free_params_immediately) {
                sd_ctx->sd->diffusion_model->free_params_buffer();
            }
            return nullptr;
        }
        if (sd_ctx->sd->free_params_immediately) {
            sd_ctx->sd->diffusion_model->free_params_buffer();
        }
        int64_t hires_denoise_end = ggml_time_ms();
        LOG_INFO("hires fix completed, taking %.2fs", (hires_denoise_end - hires_denoise_start) * 1.0f / 1000);

        final_latents = std::move(hires_final_latents);
    }

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
    sd_ctx->sd->rng->manual_seed(request.seed);
    sd_ctx->sd->sampler_rng->manual_seed(request.seed);
    sd_ctx->sd->set_flow_shift(sd_vid_gen_params->sample_params.flow_shift);
    sd_ctx->sd->apply_loras(sd_vid_gen_params->loras, sd_vid_gen_params->lora_count);

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

    auto result = decode_video_outputs(sd_ctx, final_latent, num_frames_out);
    if (result == nullptr) {
        return nullptr;
    }

    sd_ctx->sd->lora_stat();

    int64_t t1 = ggml_time_ms();
    LOG_INFO("generate_video completed in %.2fs", (t1 - t0) * 1.0f / 1000);
    return result;
}
