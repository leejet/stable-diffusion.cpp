#include "ggml-cpu.h"
#include "ggml_extend.hpp"

#include "model.h"
#include "rng.hpp"
#include "rng_mt19937.hpp"
#include "rng_philox.hpp"
#include "stable-diffusion.h"
#include <vector>
#include "util.h"

#include "auto_encoder_kl.hpp"
#include "cache_dit.hpp"
#include "conditioner.hpp"
#include "control.hpp"
#include "denoiser.hpp"
#include "diffusion_model.hpp"
#include "easycache.hpp"
#include "esrgan.hpp"
#include "lora.hpp"
#include "pmid.hpp"
#include "spectrum.hpp"
#include "tae.hpp"
#include "ucache.hpp"
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
    "SDXS",
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
};

/*================================================== Helper Functions ================================================*/

void calculate_alphas_cumprod(float* alphas_cumprod,
                              float linear_start = 0.00085f,
                              float linear_end   = 0.0120,
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
            reuse_threshold = 0.2;
        } else if (params.mode == SD_CACHE_UCACHE) {
            reuse_threshold = 1.0;
        }
    }
    return std::max(0.0f, reuse_threshold);
}

enum class SampleCacheMode {
    NONE,
    EASYCACHE,
    UCACHE,
    CACHEDIT,
};

struct SampleCacheRuntime {
    SampleCacheMode mode = SampleCacheMode::NONE;

    EasyCacheState easycache;
    UCacheState ucache;
    CacheDitConditionState cachedit;
    SpectrumState spectrum;

    bool spectrum_enabled = false;

    bool has_step_cache() const {
        return mode != SampleCacheMode::NONE;
    }

    bool easycache_enabled() const {
        return mode == SampleCacheMode::EASYCACHE;
    }

    bool ucache_enabled() const {
        return mode == SampleCacheMode::UCACHE;
    }

    bool cachedit_enabled() const {
        return mode == SampleCacheMode::CACHEDIT;
    }
};

static bool has_valid_cache_percent_range(const sd_cache_params_t& cache_params) {
    if (cache_params.mode != SD_CACHE_EASYCACHE && cache_params.mode != SD_CACHE_UCACHE) {
        return true;
    }

    return cache_params.start_percent >= 0.0f &&
           cache_params.start_percent < 1.0f &&
           cache_params.end_percent > 0.0f &&
           cache_params.end_percent <= 1.0f &&
           cache_params.start_percent < cache_params.end_percent;
}

static void init_easycache_runtime(SampleCacheRuntime& runtime,
                                   SDVersion version,
                                   const sd_cache_params_t& cache_params,
                                   Denoiser* denoiser) {
    if (!sd_version_is_dit(version)) {
        LOG_WARN("EasyCache requested but not supported for this model type");
        return;
    }

    EasyCacheConfig config;
    config.enabled         = true;
    config.reuse_threshold = get_cache_reuse_threshold(cache_params);
    config.start_percent   = cache_params.start_percent;
    config.end_percent     = cache_params.end_percent;

    runtime.easycache.init(config, denoiser);
    if (!runtime.easycache.enabled()) {
        LOG_WARN("EasyCache requested but could not be initialized for this run");
        return;
    }

    runtime.mode = SampleCacheMode::EASYCACHE;
    LOG_INFO("EasyCache enabled - threshold: %.3f, start: %.2f, end: %.2f",
             config.reuse_threshold,
             config.start_percent,
             config.end_percent);
}

static void init_ucache_runtime(SampleCacheRuntime& runtime,
                                SDVersion version,
                                const sd_cache_params_t& cache_params,
                                Denoiser* denoiser,
                                const std::vector<float>& sigmas) {
    if (!sd_version_is_unet(version)) {
        LOG_WARN("UCache requested but not supported for this model type (only UNET models)");
        return;
    }

    UCacheConfig config;
    config.enabled                = true;
    config.reuse_threshold        = get_cache_reuse_threshold(cache_params);
    config.start_percent          = cache_params.start_percent;
    config.end_percent            = cache_params.end_percent;
    config.error_decay_rate       = std::max(0.0f, std::min(1.0f, cache_params.error_decay_rate));
    config.use_relative_threshold = cache_params.use_relative_threshold;
    config.reset_error_on_compute = cache_params.reset_error_on_compute;

    runtime.ucache.init(config, denoiser);
    if (!runtime.ucache.enabled()) {
        LOG_WARN("UCache requested but could not be initialized for this run");
        return;
    }

    runtime.ucache.set_sigmas(sigmas);
    runtime.mode = SampleCacheMode::UCACHE;
    LOG_INFO("UCache enabled - threshold: %.3f, start: %.2f, end: %.2f, decay: %.2f, relative: %s, reset: %s",
             config.reuse_threshold,
             config.start_percent,
             config.end_percent,
             config.error_decay_rate,
             config.use_relative_threshold ? "true" : "false",
             config.reset_error_on_compute ? "true" : "false");
}

static void init_cachedit_runtime(SampleCacheRuntime& runtime,
                                  SDVersion version,
                                  const sd_cache_params_t& cache_params,
                                  const std::vector<float>& sigmas) {
    if (!sd_version_is_dit(version)) {
        LOG_WARN("CacheDIT requested but not supported for this model type (only DiT models)");
        return;
    }

    DBCacheConfig dbcfg;
    dbcfg.enabled                     = (cache_params.mode == SD_CACHE_DBCACHE ||
                     cache_params.mode == SD_CACHE_CACHE_DIT);
    dbcfg.Fn_compute_blocks           = cache_params.Fn_compute_blocks;
    dbcfg.Bn_compute_blocks           = cache_params.Bn_compute_blocks;
    dbcfg.residual_diff_threshold     = cache_params.residual_diff_threshold;
    dbcfg.max_warmup_steps            = cache_params.max_warmup_steps;
    dbcfg.max_cached_steps            = cache_params.max_cached_steps;
    dbcfg.max_continuous_cached_steps = cache_params.max_continuous_cached_steps;
    if (cache_params.scm_mask != nullptr && strlen(cache_params.scm_mask) > 0) {
        dbcfg.steps_computation_mask = parse_scm_mask(cache_params.scm_mask);
    }
    dbcfg.scm_policy_dynamic = cache_params.scm_policy_dynamic;

    TaylorSeerConfig tcfg;
    tcfg.enabled             = (cache_params.mode == SD_CACHE_TAYLORSEER ||
                    cache_params.mode == SD_CACHE_CACHE_DIT);
    tcfg.n_derivatives       = cache_params.taylorseer_n_derivatives;
    tcfg.skip_interval_steps = cache_params.taylorseer_skip_interval;

    runtime.cachedit.init(dbcfg, tcfg);
    if (!runtime.cachedit.enabled()) {
        LOG_WARN("CacheDIT requested but could not be initialized for this run");
        return;
    }

    runtime.cachedit.set_sigmas(sigmas);
    runtime.mode = SampleCacheMode::CACHEDIT;
    LOG_INFO("CacheDIT enabled - mode: %s, Fn: %d, Bn: %d, threshold: %.3f, warmup: %d",
             cache_params.mode == SD_CACHE_CACHE_DIT ? "DBCache+TaylorSeer" : (cache_params.mode == SD_CACHE_DBCACHE ? "DBCache" : "TaylorSeer"),
             dbcfg.Fn_compute_blocks,
             dbcfg.Bn_compute_blocks,
             dbcfg.residual_diff_threshold,
             dbcfg.max_warmup_steps);
}

static void init_spectrum_runtime(SampleCacheRuntime& runtime,
                                  SDVersion version,
                                  const sd_cache_params_t& cache_params,
                                  const std::vector<float>& sigmas) {
    if (!sd_version_is_unet(version) && !sd_version_is_dit(version)) {
        LOG_WARN("Spectrum requested but not supported for this model type (only UNET and DiT models)");
        return;
    }

    SpectrumConfig config;
    config.w            = cache_params.spectrum_w;
    config.m            = cache_params.spectrum_m;
    config.lam          = cache_params.spectrum_lam;
    config.window_size  = cache_params.spectrum_window_size;
    config.flex_window  = cache_params.spectrum_flex_window;
    config.warmup_steps = cache_params.spectrum_warmup_steps;
    config.stop_percent = cache_params.spectrum_stop_percent;

    size_t total_steps = sigmas.size() > 0 ? sigmas.size() - 1 : 0;
    runtime.spectrum.init(config, total_steps);
    runtime.spectrum_enabled = true;

    LOG_INFO("Spectrum enabled - w: %.2f, m: %d, lam: %.2f, window: %d, flex: %.2f, warmup: %d, stop: %.0f%%",
             config.w, config.m, config.lam,
             config.window_size, config.flex_window,
             config.warmup_steps, config.stop_percent * 100.0f);
}

static SampleCacheRuntime init_sample_cache_runtime(SDVersion version,
                                                    const sd_cache_params_t* cache_params,
                                                    Denoiser* denoiser,
                                                    const std::vector<float>& sigmas) {
    SampleCacheRuntime runtime;
    if (cache_params == nullptr || cache_params->mode == SD_CACHE_DISABLED) {
        return runtime;
    }

    if (!has_valid_cache_percent_range(*cache_params)) {
        LOG_WARN("Cache disabled due to invalid percent range (start=%.3f, end=%.3f)",
                 cache_params->start_percent,
                 cache_params->end_percent);
        return runtime;
    }

    switch (cache_params->mode) {
        case SD_CACHE_EASYCACHE:
            init_easycache_runtime(runtime, version, *cache_params, denoiser);
            break;
        case SD_CACHE_UCACHE:
            init_ucache_runtime(runtime, version, *cache_params, denoiser, sigmas);
            break;
        case SD_CACHE_DBCACHE:
        case SD_CACHE_TAYLORSEER:
        case SD_CACHE_CACHE_DIT:
            init_cachedit_runtime(runtime, version, *cache_params, sigmas);
            break;
        case SD_CACHE_SPECTRUM:
            init_spectrum_runtime(runtime, version, *cache_params, sigmas);
            break;
        default:
            break;
    }

    return runtime;
}

struct SampleStepCacheDispatcher {
    SampleCacheRuntime& runtime;
    int step;
    float sigma;
    int step_index;

    SampleStepCacheDispatcher(SampleCacheRuntime& runtime, int step, float sigma)
        : runtime(runtime), step(step), sigma(sigma), step_index(step > 0 ? (step - 1) : -1) {
        if (step_index < 0) {
            return;
        }

        switch (runtime.mode) {
            case SampleCacheMode::EASYCACHE:
                runtime.easycache.begin_step(step_index, sigma);
                break;
            case SampleCacheMode::UCACHE:
                runtime.ucache.begin_step(step_index, sigma);
                break;
            case SampleCacheMode::CACHEDIT:
                runtime.cachedit.begin_step(step_index, sigma);
                break;
            case SampleCacheMode::NONE:
                break;
        }
    }

    bool before_condition(const SDCondition* condition, ggml_tensor* input, ggml_tensor* output) {
        if (step_index < 0 || condition == nullptr || input == nullptr || output == nullptr) {
            return false;
        }

        switch (runtime.mode) {
            case SampleCacheMode::EASYCACHE:
                return runtime.easycache.before_condition(condition, input, output, sigma, step_index);
            case SampleCacheMode::UCACHE:
                return runtime.ucache.before_condition(condition, input, output, sigma, step_index);
            case SampleCacheMode::CACHEDIT:
                return runtime.cachedit.before_condition(condition, input, output, sigma, step_index);
            case SampleCacheMode::NONE:
                return false;
        }

        return false;
    }

    void after_condition(const SDCondition* condition, ggml_tensor* input, ggml_tensor* output) {
        if (step_index < 0 || condition == nullptr || input == nullptr || output == nullptr) {
            return;
        }

        switch (runtime.mode) {
            case SampleCacheMode::EASYCACHE:
                runtime.easycache.after_condition(condition, input, output);
                break;
            case SampleCacheMode::UCACHE:
                runtime.ucache.after_condition(condition, input, output);
                break;
            case SampleCacheMode::CACHEDIT:
                runtime.cachedit.after_condition(condition, input, output);
                break;
            case SampleCacheMode::NONE:
                break;
        }
    }

    bool is_step_skipped() const {
        switch (runtime.mode) {
            case SampleCacheMode::EASYCACHE:
                return runtime.easycache.is_step_skipped();
            case SampleCacheMode::UCACHE:
                return runtime.ucache.is_step_skipped();
            case SampleCacheMode::CACHEDIT:
                return runtime.cachedit.is_step_skipped();
            case SampleCacheMode::NONE:
                return false;
        }

        return false;
    }
};

static void log_sample_cache_summary(const SampleCacheRuntime& runtime, size_t total_steps) {
    if (runtime.easycache_enabled()) {
        if (runtime.easycache.total_steps_skipped > 0 && total_steps > 0) {
            if (runtime.easycache.total_steps_skipped < static_cast<int>(total_steps)) {
                double speedup = static_cast<double>(total_steps) /
                                 static_cast<double>(total_steps - runtime.easycache.total_steps_skipped);
                LOG_INFO("EasyCache skipped %d/%zu steps (%.2fx estimated speedup)",
                         runtime.easycache.total_steps_skipped,
                         total_steps,
                         speedup);
            } else {
                LOG_INFO("EasyCache skipped %d/%zu steps",
                         runtime.easycache.total_steps_skipped,
                         total_steps);
            }
        } else if (total_steps > 0) {
            LOG_INFO("EasyCache completed without skipping steps");
        }
    }

    if (runtime.ucache_enabled()) {
        if (runtime.ucache.total_steps_skipped > 0 && total_steps > 0) {
            if (runtime.ucache.total_steps_skipped < static_cast<int>(total_steps)) {
                double speedup = static_cast<double>(total_steps) /
                                 static_cast<double>(total_steps - runtime.ucache.total_steps_skipped);
                LOG_INFO("UCache skipped %d/%zu steps (%.2fx estimated speedup)",
                         runtime.ucache.total_steps_skipped,
                         total_steps,
                         speedup);
            } else {
                LOG_INFO("UCache skipped %d/%zu steps",
                         runtime.ucache.total_steps_skipped,
                         total_steps);
            }
        } else if (total_steps > 0) {
            LOG_INFO("UCache completed without skipping steps");
        }
    }

    if (runtime.cachedit_enabled()) {
        if (runtime.cachedit.total_steps_skipped > 0 && total_steps > 0) {
            if (runtime.cachedit.total_steps_skipped < static_cast<int>(total_steps)) {
                double speedup = static_cast<double>(total_steps) /
                                 static_cast<double>(total_steps - runtime.cachedit.total_steps_skipped);
                LOG_INFO("CacheDIT skipped %d/%zu steps (%.2fx estimated speedup), accum_diff: %.4f",
                         runtime.cachedit.total_steps_skipped,
                         total_steps,
                         speedup,
                         runtime.cachedit.accumulated_residual_diff);
            } else {
                LOG_INFO("CacheDIT skipped %d/%zu steps, accum_diff: %.4f",
                         runtime.cachedit.total_steps_skipped,
                         total_steps,
                         runtime.cachedit.accumulated_residual_diff);
            }
        } else if (total_steps > 0) {
            LOG_INFO("CacheDIT completed without skipping steps");
        }
    }

    if (runtime.spectrum_enabled && runtime.spectrum.total_steps_skipped > 0 && total_steps > 0) {
        double speedup = static_cast<double>(total_steps) /
                         static_cast<double>(total_steps - runtime.spectrum.total_steps_skipped);
        LOG_INFO("Spectrum skipped %d/%zu steps (%.2fx estimated speedup)",
                 runtime.spectrum.total_steps_skipped,
                 total_steps,
                 speedup);
    }
}

std::vector<std::string> string_split(const std::string & input, char separator)
{
    std::vector<std::string> parts;
    size_t begin_pos = 0;
    size_t separator_pos = input.find(separator);
    while (separator_pos != std::string::npos) {
        std::string part = input.substr(begin_pos, separator_pos - begin_pos);
        parts.emplace_back(part);
        begin_pos = separator_pos + 1;
        separator_pos = input.find(separator, begin_pos);
    }
    parts.emplace_back(input.substr(begin_pos, separator_pos - begin_pos));
    return parts;
}

static void add_rpc_devices(const std::string & servers) {
    auto rpc_servers = string_split(servers, ',');
    if (rpc_servers.empty()) {
        throw std::invalid_argument("no RPC servers specified");
    }
    ggml_backend_reg_t rpc_reg = ggml_backend_reg_by_name("RPC");
    if (!rpc_reg) {
        throw std::invalid_argument("failed to find RPC backend");
    }
    typedef ggml_backend_reg_t (*ggml_backend_rpc_add_server_t)(const char * endpoint);
    ggml_backend_rpc_add_server_t ggml_backend_rpc_add_server_fn = (ggml_backend_rpc_add_server_t) ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_add_server");
    if (!ggml_backend_rpc_add_server_fn) {
        throw std::invalid_argument("failed to find RPC add server function");
    }
    for (const auto & server : rpc_servers) {
        auto reg = ggml_backend_rpc_add_server_fn(server.c_str());
        ggml_backend_register(reg);
    }
}

void add_rpc_device(const char* servers_cstr){
    std::string servers(servers_cstr);
    add_rpc_devices(servers);
}

std::vector<std::string> sanitize_backend_name_list(std::string name) {
    std::vector<std::string> vec = {};
    if (name == "" || backend_name_exists(name)) {
        // single backend
        vec.push_back(name);
    } else if (name.find(",") != std::string::npos) {
        // comma-separated backend names
        std::stringstream ss(name);
        std::string token;
        while (std::getline(ss, token, ',')) {
            if (token == "" || backend_name_exists(token)) {
                vec.push_back(token);
            } else {
                LOG_WARN("backend name %s not found, using default", token.c_str());
                vec.push_back("");
            }
        }
    } else {
        vec.push_back("");
    }
    return vec;
}

std::vector<std::pair<std::string, std::string>> list_backends_vector() {
    std::vector<std::pair<std::string, std::string>> backends;
    const int device_count = ggml_backend_dev_count();
    for (int i = 0; i < device_count; i++) {
        auto dev = ggml_backend_dev_get(i);
        backends.push_back({ggml_backend_dev_name(dev), ggml_backend_dev_description(dev)});
    }
    return backends;
}

SD_API size_t backend_list_size(){
    // for C API
    size_t buffer_size = 0;
    auto backends = list_backends_vector();
    for (auto& backend : backends) {
        auto dev_name_size = backend.first.size();
        auto dev_desc_size = backend.second.size();
        buffer_size+=dev_name_size+dev_desc_size+2; // +2 for the separators
    }
    return buffer_size;
}

// devices are separated by \n and name and description are separated by \t
SD_API void list_backends_to_buffer(char* buffer, size_t buffer_size) {
    auto backends = list_backends_vector();
    size_t offset = 0;
    for (auto& backend : backends) {
        size_t name_size = backend.first.size();
        size_t desc_size = backend.second.size();
        if (offset + name_size + desc_size + 2 > buffer_size) {
            break; // Not enough space in the buffer
        }
        memcpy(buffer + offset, backend.first.c_str(), name_size);
        offset += name_size;
        buffer[offset++] = '\t';
        memcpy(buffer + offset, backend.second.c_str(), desc_size);
        offset += desc_size;
        buffer[offset++] = '\n'; 
    }
    if (offset < buffer_size) {
        buffer[offset] = '\0'; // Ensure the buffer is null-terminated at the end
    } else {
        LOG_WARN("Provided buffer size is too small to contain details of all devices.");
        buffer[buffer_size - 1] = '\0';  // Ensure the buffer is null-terminated at the end
    }
}

/*=============================================== StableDiffusionGGML ================================================*/

class StableDiffusionGGML {
public:
    ggml_backend_t backend             = nullptr;  // general backend
    ggml_backend_t diffusion_backend   = nullptr;
    ggml_backend_t control_net_backend = nullptr;
    ggml_backend_t vae_backend         = nullptr;
    ggml_backend_t tae_backend         = nullptr;
    ggml_backend_t pmid_backend        = nullptr;
    ggml_backend_t vision_backend      = nullptr;

    std::vector<ggml_backend_t> clip_backends        = {nullptr};

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
        if (diffusion_backend != backend) {
            ggml_backend_free(diffusion_backend);
        }
        for(auto clip_backend : clip_backends) {
            if (clip_backend != backend) {
                ggml_backend_free(clip_backend);
            }
        }
        if (control_net_backend != backend) {
            ggml_backend_free(control_net_backend);
        }
        if (tae_backend != vae_backend) {
            ggml_backend_free(tae_backend);
        }
        if (vae_backend != backend) {
            ggml_backend_free(vae_backend);
        }
        ggml_backend_free(backend);
    }


    void log_backends() {
        const int device_count = ggml_backend_dev_count();
        for (int i = 0; i < device_count; i++) {
            auto dev = ggml_backend_dev_get(i);
            LOG_INFO("%s (%s)", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
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

        log_backends();

        std::string default_backend_name = get_default_backend_name();

        std::string override_default_backend_name = sanitize_backend_name(SAFE_STR(sd_ctx_params->main_device));

        if (override_default_backend_name.size() > 0) {
            LOG_INFO("Setting default backend to %s", override_default_backend_name.c_str());
            default_backend_name = override_default_backend_name;
        }

        std::string diffusion_backend_name   = sanitize_backend_name(SAFE_STR(sd_ctx_params->diffusion_device));
        std::vector<std::string> clip_backend_names        = sanitize_backend_name_list(SAFE_STR(sd_ctx_params->clip_device));
        std::string control_net_backend_name = sanitize_backend_name(SAFE_STR(sd_ctx_params->control_net_device));
        std::string vae_backend_name         = sanitize_backend_name(SAFE_STR(sd_ctx_params->vae_device));
        std::string tae_backend_name         = sanitize_backend_name(SAFE_STR(sd_ctx_params->tae_device));
        std::string pmid_backend_name        = sanitize_backend_name(SAFE_STR(sd_ctx_params->photomaker_device));
        std::string vision_backend_name      = sanitize_backend_name(SAFE_STR(sd_ctx_params->vision_device));

        bool diffusion_backend_is_default   = diffusion_backend_name.empty() || diffusion_backend_name == default_backend_name;
        bool clip_backends_are_default = true;
        for (const auto& clip_backend_name : clip_backend_names) {
            if (!clip_backend_name.empty() && clip_backend_name != default_backend_name) {
                clip_backends_are_default = false;
                break;
            }
        }
        bool control_net_backend_is_default = (control_net_backend_name.empty() || control_net_backend_name == default_backend_name);
        bool vae_backend_is_default         = (vae_backend_name.empty() || vae_backend_name == default_backend_name);
        // if tae_backend_name is empty, it will use the same backend as vae
        bool tae_backend_is_default = (tae_backend_name.empty() && vae_backend_is_default) || tae_backend_name == default_backend_name;
        bool pmid_backend_is_default = (pmid_backend_name.empty() || pmid_backend_name == default_backend_name);
        bool vision_backend_is_default = (vision_backend_name.empty() || vision_backend_name == default_backend_name);

        // if some backend is not specified or is the same as the default backend, use the default backend
        bool use_default_backend = diffusion_backend_is_default || clip_backends_are_default || control_net_backend_is_default || vae_backend_is_default || tae_backend_is_default || pmid_backend_is_default || vision_backend_is_default;

        if (use_default_backend) {
            backend = init_named_backend(override_default_backend_name);
            LOG_DEBUG("Loaded default backend %s", ggml_backend_name(backend));
        }

        if (!diffusion_backend_is_default) {
            diffusion_backend = init_named_backend(diffusion_backend_name);
            LOG_INFO("Using diffusion backend: %s", ggml_backend_name(diffusion_backend));
        } else {
            diffusion_backend = backend;
        }

        ModelLoader model_loader;

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
        if (version == VERSION_SDXS) {
            tae_preview_only = false;
            use_tae          = true;
        }

        if (sd_ctx_params->circular_x || sd_ctx_params->circular_y) {
            LOG_INFO("Using circular padding for convolutions");
        }

        {
            if (!clip_backends_are_default) {
                clip_backends.clear();
                for(auto clip_backend_name : clip_backend_names){
                    auto clip_backend = init_named_backend(clip_backend_name);
                    LOG_INFO("CLIP: Using %s backend", ggml_backend_name(clip_backend));
                    clip_backends.push_back(clip_backend); 
                }
            }else{
                clip_backends = {backend};
            }
            if (sd_version_is_sd3(version)) {
                cond_stage_model = std::make_shared<SD3CLIPEmbedder>(clip_backends,
                                                                     offload_params_to_cpu,
                                                                     tensor_storage_map);
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

                    cond_stage_model = std::make_shared<T5CLIPEmbedder>(clip_backends[0],
                                                                        offload_params_to_cpu,
                                                                        tensor_storage_map,
                                                                        sd_ctx_params->chroma_use_t5_mask,
                                                                        sd_ctx_params->chroma_t5_mask_pad);
                } else if (version == VERSION_OVIS_IMAGE) {
                    cond_stage_model = std::make_shared<LLMEmbedder>(clip_backends[0],
                                                                     offload_params_to_cpu,
                                                                     tensor_storage_map,
                                                                     version,
                                                                     "",
                                                                     false);
                } else {
                    cond_stage_model = std::make_shared<FluxCLIPEmbedder>(clip_backends,
                                                                          offload_params_to_cpu,
                                                                          tensor_storage_map);
                }
                diffusion_model = std::make_shared<FluxModel>(diffusion_backend,
                                                              offload_params_to_cpu,
                                                              tensor_storage_map,
                                                              version,
                                                              sd_ctx_params->chroma_use_dit_mask);
            } else if (sd_version_is_flux2(version)) {
                bool is_chroma   = false;
                cond_stage_model = std::make_shared<LLMEmbedder>(clip_backends[0],
                                                                 offload_params_to_cpu,
                                                                 tensor_storage_map,
                                                                 version);
                diffusion_model  = std::make_shared<FluxModel>(diffusion_backend,
                                                               offload_params_to_cpu,
                                                               tensor_storage_map,
                                                               version,
                                                               sd_ctx_params->chroma_use_dit_mask);
            } else if (sd_version_is_wan(version)) {
                cond_stage_model = std::make_shared<T5CLIPEmbedder>(clip_backends[0],
                                                                    offload_params_to_cpu,
                                                                    tensor_storage_map,
                                                                    true,
                                                                    1,
                                                                    true);
                diffusion_model  = std::make_shared<WanModel>(diffusion_backend,
                                                              offload_params_to_cpu,
                                                              tensor_storage_map,
                                                              "model.diffusion_model",
                                                              version);
                if (strlen(SAFE_STR(sd_ctx_params->high_noise_diffusion_model_path)) > 0) {
                    high_noise_diffusion_model = std::make_shared<WanModel>(diffusion_backend,
                                                                            offload_params_to_cpu,
                                                                            tensor_storage_map,
                                                                            "model.high_noise_diffusion_model",
                                                                            version);
                }
                if (diffusion_model->get_desc() == "Wan2.1-I2V-14B" ||
                    diffusion_model->get_desc() == "Wan2.1-FLF2V-14B" ||
                    diffusion_model->get_desc() == "Wan2.1-I2V-1.3B") {
                    clip_vision = std::make_shared<FrozenCLIPVisionEmbedder>(vision_backend,
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
                cond_stage_model = std::make_shared<LLMEmbedder>(clip_backends[0],
                                                                 offload_params_to_cpu,
                                                                 tensor_storage_map,
                                                                 version,
                                                                 "",
                                                                 enable_vision);
                diffusion_model  = std::make_shared<QwenImageModel>(diffusion_backend,
                                                                    offload_params_to_cpu,
                                                                    tensor_storage_map,
                                                                    "model.diffusion_model",
                                                                    version,
                                                                    sd_ctx_params->qwen_image_zero_cond_t);
            } else if (sd_version_is_anima(version)) {
                cond_stage_model = std::make_shared<AnimaConditioner>(clip_backend,
                                                                      offload_params_to_cpu,
                                                                      tensor_storage_map);
                diffusion_model  = std::make_shared<AnimaModel>(backend,
                                                               offload_params_to_cpu,
                                                               tensor_storage_map,
                                                               "model.diffusion_model");
            } else if (sd_version_is_z_image(version)) {
                cond_stage_model = std::make_shared<LLMEmbedder>(clip_backends[0],
                                                                 offload_params_to_cpu,
                                                                 tensor_storage_map,
                                                                 version);
                diffusion_model  = std::make_shared<ZImageModel>(diffusion_backend,
                                                                 offload_params_to_cpu,
                                                                 tensor_storage_map,
                                                                 "model.diffusion_model",
                                                                 version);
            } else {  // SD1.x SD2.x SDXL
                std::map<std::string, std::string> embbeding_map;
                for (uint32_t i = 0; i < sd_ctx_params->embedding_count; i++) {
                    embbeding_map.emplace(SAFE_STR(sd_ctx_params->embeddings[i].name), SAFE_STR(sd_ctx_params->embeddings[i].path));
                }
                if (strstr(SAFE_STR(sd_ctx_params->photo_maker_path), "v2")) {
                    cond_stage_model = std::make_shared<FrozenCLIPEmbedderWithCustomWords>(clip_backends,
                                                                                           offload_params_to_cpu,
                                                                                           tensor_storage_map,
                                                                                           embbeding_map,
                                                                                           version,
                                                                                           PM_VERSION_2);
                } else {
                    cond_stage_model = std::make_shared<FrozenCLIPEmbedderWithCustomWords>(clip_backends,
                                                                                           offload_params_to_cpu,
                                                                                           tensor_storage_map,
                                                                                           embbeding_map,
                                                                                           version);
                }
                diffusion_model = std::make_shared<UNetModel>(diffusion_backend,
                                                              offload_params_to_cpu,
                                                              tensor_storage_map,
                                                              version);
                if (sd_ctx_params->diffusion_conv_direct) {
                    LOG_INFO("Using Conv2d direct in the diffusion model");
                    std::dynamic_pointer_cast<UNetModel>(diffusion_model)->unet.set_conv2d_direct_enabled(true);
                }
            }

            cond_stage_model->alloc_params_buffer();
            cond_stage_model->get_param_tensors(tensors);

            diffusion_model->alloc_params_buffer();
            diffusion_model->get_param_tensors(tensors);

            if (sd_version_is_unet_edit(version)) {
                vae_decode_only = false;
            }

            if (high_noise_diffusion_model) {
                high_noise_diffusion_model->alloc_params_buffer();
                high_noise_diffusion_model->get_param_tensors(tensors);
            }

            vae_backend = backend;
            if (!vae_backend_is_default) {
                vae_backend = init_named_backend(vae_backend_name);
                LOG_INFO("VAE Autoencoder: Using %s backend", ggml_backend_name(vae_backend));
            }
            tae_backend = vae_backend;
            if (tae_backend_name.length() > 0 && tae_backend_name != vae_backend_name) {
                tae_backend = init_named_backend(tae_backend_name);
                LOG_INFO("Tiny Autoencoder: Using %s backend", ggml_backend_name(tae_backend));
            }

            auto create_tae = [&]() -> std::shared_ptr<VAE> {
                if (sd_version_is_wan(version) ||
                    sd_version_is_qwen_image(version) ||
                    sd_version_is_anima(version)) {
                    return std::make_shared<TinyVideoAutoEncoder>(tae_backend,
                                                                  offload_params_to_cpu,
                                                                  tensor_storage_map,
                                                                  "decoder",
                                                                  vae_decode_only,
                                                                  version);

                } else {
                    auto model = std::make_shared<TinyImageAutoEncoder>(tae_backend,
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
                if (!control_net_backend_is_default) {
                    control_net_backend = init_named_backend(control_net_backend_name);
                    LOG_INFO("ControlNet: Using %s backend", ggml_backend_name(control_net_backend));
                } else {
                    control_net_backend = backend;
                }
                control_net = std::make_shared<ControlNet>(control_net_backend,
                                                           offload_params_to_cpu,
                                                           tensor_storage_map,
                                                           version);
                if (sd_ctx_params->diffusion_conv_direct) {
                    LOG_INFO("Using Conv2d direct in the control net");
                    control_net->set_conv2d_direct_enabled(true);
                }
            }
             pmid_backend = backend;
            if (!pmid_backend_is_default) {
                pmid_backend = init_named_backend(pmid_backend_name);
                LOG_INFO("PhotoMaker: Using %s backend", ggml_backend_name(pmid_backend));
            } else {
                pmid_backend = backend;
            }
            if (strstr(SAFE_STR(sd_ctx_params->photo_maker_path), "v2")) {
                pmid_model = std::make_shared<PhotoMakerIDEncoder>(pmid_backend,
                                                                   offload_params_to_cpu,
                                                                   tensor_storage_map,
                                                                   "pmid",
                                                                   version,
                                                                   PM_VERSION_2);
                LOG_INFO("using PhotoMaker Version 2");
            } else {
                pmid_model = std::make_shared<PhotoMakerIDEncoder>(pmid_backend,
                                                                   offload_params_to_cpu,
                                                                   tensor_storage_map,
                                                                   "pmid",
                                                                   version);
            }
            if (strlen(SAFE_STR(sd_ctx_params->photo_maker_path)) > 0) {
                pmid_lora               = std::make_shared<LoraModel>("pmid", diffusion_backend, sd_ctx_params->photo_maker_path, "", version);
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
            
            // TODO: split by individual text encoders
            if (ggml_backend_is_cpu(clip_backends[0])) {
                total_params_ram_size += clip_params_mem_size + pmid_params_mem_size;
            } else {
                total_params_vram_size += clip_params_mem_size + pmid_params_mem_size;
            }

            if (ggml_backend_is_cpu(diffusion_backend)) {
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
                // TODO: split
                ggml_backend_is_cpu(clip_backends[0]) ? "RAM" : "VRAM",
                unet_params_mem_size / 1024.0 / 1024.0,
                ggml_backend_is_cpu(backend) ? "RAM" : "VRAM",
                vae_params_mem_size / 1024.0 / 1024.0,
                ggml_backend_is_cpu(vae_backend) ? "RAM" : "VRAM",
                control_net_params_mem_size / 1024.0 / 1024.0,
                ggml_backend_is_cpu(control_net_backend) ? "RAM" : "VRAM",
                pmid_params_mem_size / 1024.0 / 1024.0,
                ggml_backend_is_cpu(pmid_backend) ? "RAM" : "VRAM");
        }

        // init denoiser
        {
            prediction_t pred_type = sd_ctx_params->prediction;

            if (pred_type == PREDICTION_COUNT) {
                if (sd_version_is_sd2(version)) {
                    // check is_using_v_parameterization_for_sd2
                    if (is_using_v_parameterization_for_sd2(ctx, sd_version_is_inpaint(version))) {
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
                           sd_version_is_z_image(version)) {
                    pred_type = FLOW_PRED;
                    if (sd_version_is_wan(version)) {
                        default_flow_shift = 5.f;
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

    bool is_using_v_parameterization_for_sd2(ggml_context* work_ctx, bool is_inpaint = false) {
        ggml_tensor* x_t = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, 8, 8, 4, 1);
        ggml_set_f32(x_t, 0.5);
        ggml_tensor* c = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, 1024, 2, 1, 1);
        ggml_set_f32(c, 0.5);

        ggml_tensor* timesteps = ggml_new_tensor_1d(work_ctx, GGML_TYPE_F32, 1);
        ggml_set_f32(timesteps, 999);

        ggml_tensor* concat = is_inpaint ? ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, 8, 8, 5, 1) : nullptr;
        if (concat != nullptr) {
            ggml_set_f32(concat, 0);
        }

        int64_t t0       = ggml_time_ms();
        ggml_tensor* out = ggml_dup_tensor(work_ctx, x_t);
        DiffusionParams diffusion_params;
        diffusion_params.x         = x_t;
        diffusion_params.timesteps = timesteps;
        diffusion_params.context   = c;
        diffusion_params.c_concat  = concat;
        diffusion_model->compute(n_threads, diffusion_params, &out);
        diffusion_model->free_compute_buffer();

        double result = 0.f;
        {
            float* vec_x   = (float*)x_t->data;
            float* vec_out = (float*)out->data;

            int64_t n = ggml_nelements(out);

            for (int i = 0; i < n; i++) {
                result += ((double)vec_out[i] - (double)vec_x[i]);
            }
            result /= n;
        }
        int64_t t1 = ggml_time_ms();
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
            bool applied = false;
            int64_t t0 = ggml_time_ms();
            // TODO: Fix that
            bool are_clip_backends_similar = true;
            for (auto backend: clip_backends){
                are_clip_backends_similar = are_clip_backends_similar && (clip_backends[0]==backend || ggml_backend_is_cpu(backend));
            }
            if(!are_clip_backends_similar){
                LOG_WARN("Text encoders are running on different backends. This may cause issues when immediately applying LoRAs.");
            }
            auto lora_tensor_filter_diff = [&](const std::string& tensor_name) {
                if (is_diffusion_model_name(tensor_name)) {
                    return true;
                }
                return false;
            };
            auto lora = load_lora_model_from_file(kv.first, kv.second, diffusion_backend, lora_tensor_filter_diff);
            if (lora && !lora->lora_tensors.empty()) {
                lora->apply(tensors, version, n_threads);
                lora->free_params_buffer();
                applied = true;
            }

            auto lora_tensor_filter_cond = [&](const std::string& tensor_name) {
                if (is_cond_stage_model_name(tensor_name)) {
                    return true;
                }
                return false;
            };
            // TODO: split by model
            lora = load_lora_model_from_file(kv.first, kv.second, clip_backends[0], lora_tensor_filter_cond);
            if (lora && !lora->lora_tensors.empty()) {
                lora->apply(tensors, version, n_threads);
                lora->free_params_buffer();
                applied = true;
            }

            auto lora_tensor_filter_first = [&](const std::string& tensor_name) {
                if (is_first_stage_model_name(tensor_name)) {
                    return true;
                }
                return false;
            };
            auto first_stage_backend = use_tiny_autoencoder ? tae_backend : vae_backend;
            lora                     = load_lora_model_from_file(kv.first, kv.second, first_stage_backend, lora_tensor_filter_first);
            if (lora && !lora->lora_tensors.empty()) {
                lora->apply(tensors, version, n_threads);
                lora->free_params_buffer();
                applied = true;
            }

            if (!applied) {
                continue;
            }
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
                //TODO: split by model
                auto lora = load_lora_model_from_file(lora_id, multiplier, clip_backends[0], lora_tensor_filter);
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

                auto lora = load_lora_model_from_file(lora_name, multiplier, diffusion_backend, lora_tensor_filter);
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

    SDCondition get_pmid_conditon(ggml_context* work_ctx,
                                  sd_pm_params_t pm_params,
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

                auto id_image_tensor = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, clip_image_size, clip_image_size, 3, pm_params.id_images_count);

                std::vector<sd_image_f32_t> processed_id_images;
                for (int i = 0; i < pm_params.id_images_count; i++) {
                    sd_image_f32_t id_image           = sd_image_t_to_sd_image_f32_t(pm_params.id_images[i]);
                    sd_image_f32_t processed_id_image = clip_preprocess(id_image, clip_image_size, clip_image_size);
                    free(id_image.data);
                    id_image.data = nullptr;
                    processed_id_images.push_back(processed_id_image);
                }

                ggml_ext_tensor_iter(id_image_tensor, [&](ggml_tensor* id_image_tensor, int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
                    float value = sd_image_get_f32(processed_id_images[i3], i0, i1, i2, false);
                    ggml_ext_tensor_set_f32(id_image_tensor, value, i0, i1, i2, i3);
                });

                for (auto& image : processed_id_images) {
                    free(image.data);
                    image.data = nullptr;
                }
                processed_id_images.clear();

                int64_t t0                      = ggml_time_ms();
                condition_params.num_input_imgs = pm_params.id_images_count;
                auto cond_tup                   = cond_stage_model->get_learned_condition_with_trigger(work_ctx,
                                                                                                       n_threads,
                                                                                                       condition_params);
                id_cond                         = std::get<0>(cond_tup);
                auto class_tokens_mask          = std::get<1>(cond_tup);
                ggml_tensor* id_embeds          = nullptr;
                if (pmv2 && pm_params.id_embed_path != nullptr) {
                    id_embeds = load_tensor_from_file(work_ctx, pm_params.id_embed_path);
                }
                if (pmv2 && id_embeds == nullptr) {
                    LOG_WARN("Provided PhotoMaker images, but NO valid ID embeds file for PM v2");
                    LOG_WARN("Turn off PhotoMaker");
                    use_pmid = false;
                } else {
                    if (pmv2 && pm_params.id_images_count != id_embeds->ne[1]) {
                        LOG_WARN("PhotoMaker image count (%d) does NOT match ID embeds (%d). You should run face_detect.py again.", pm_params.id_images_count, id_embeds->ne[1]);
                        LOG_WARN("Turn off PhotoMaker");
                        use_pmid = false;
                    } else {
                        ggml_tensor* res = nullptr;
                        pmid_model->compute(n_threads, id_image_tensor, id_cond.c_crossattn, id_embeds, class_tokens_mask, &res, work_ctx);
                        id_cond.c_crossattn = res;
                        int64_t t1          = ggml_time_ms();
                        LOG_INFO("Photomaker ID Stacking, taking %" PRId64 " ms", t1 - t0);
                        if (free_params_immediately) {
                            pmid_model->free_params_buffer();
                        }
                        // Encode input prompt without the trigger word for delayed conditioning
                        condition_params.text = cond_stage_model->remove_trigger_from_prompt(work_ctx, condition_params.text);
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

    ggml_tensor* get_clip_vision_output(ggml_context* work_ctx,
                                        sd_image_t init_image,
                                        bool return_pooled   = true,
                                        int clip_skip        = -1,
                                        bool zero_out_masked = false) {
        ggml_tensor* output = nullptr;
        if (zero_out_masked) {
            if (return_pooled) {
                output = ggml_new_tensor_1d(work_ctx,
                                            GGML_TYPE_F32,
                                            clip_vision->vision_model.projection_dim);
            } else {
                output = ggml_new_tensor_2d(work_ctx,
                                            GGML_TYPE_F32,
                                            clip_vision->vision_model.hidden_size,
                                            257);
            }

            ggml_set_f32(output, 0.f);
        } else {
            sd_image_f32_t image         = sd_image_t_to_sd_image_f32_t(init_image);
            sd_image_f32_t resized_image = clip_preprocess(image, clip_vision->vision_model.image_size, clip_vision->vision_model.image_size);
            free(image.data);
            image.data = nullptr;

            ggml_tensor* pixel_values = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, resized_image.width, resized_image.height, 3, 1);
            sd_image_f32_to_ggml_tensor(resized_image, pixel_values, false);
            free(resized_image.data);
            resized_image.data = nullptr;

            // print_ggml_tensor(pixel_values);
            clip_vision->compute(n_threads, pixel_values, return_pooled, clip_skip, &output, work_ctx);
            // print_ggml_tensor(c_crossattn);
        }
        return output;
    }

    SDCondition get_svd_condition(ggml_context* work_ctx,
                                  sd_image_t init_image,
                                  int width,
                                  int height,
                                  int fps                  = 6,
                                  int motion_bucket_id     = 127,
                                  float augmentation_level = 0.f,
                                  bool zero_out_masked     = false) {
        // c_crossattn
        int64_t t0               = ggml_time_ms();
        ggml_tensor* c_crossattn = get_clip_vision_output(work_ctx, init_image, true, -1, zero_out_masked);

        // c_concat
        ggml_tensor* c_concat = nullptr;
        {
            if (zero_out_masked) {
                c_concat = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, width / get_vae_scale_factor(), height / get_vae_scale_factor(), 4, 1);
                ggml_set_f32(c_concat, 0.f);
            } else {
                ggml_tensor* init_img = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, width, height, 3, 1);

                if (width != init_image.width || height != init_image.height) {
                    sd_image_f32_t image         = sd_image_t_to_sd_image_f32_t(init_image);
                    sd_image_f32_t resized_image = resize_sd_image_f32_t(image, width, height);
                    free(image.data);
                    image.data = nullptr;
                    sd_image_f32_to_ggml_tensor(resized_image, init_img, false);
                    free(resized_image.data);
                    resized_image.data = nullptr;
                } else {
                    sd_image_to_ggml_tensor(init_image, init_img);
                }
                if (augmentation_level > 0.f) {
                    ggml_tensor* noise = ggml_dup_tensor(work_ctx, init_img);
                    ggml_ext_im_set_randn_f32(noise, rng);
                    // encode_pixels += torch.randn_like(pixels) * augmentation_level
                    ggml_ext_tensor_scale_inplace(noise, augmentation_level);
                    ggml_ext_tensor_add_inplace(init_img, noise);
                }
                c_concat = encode_first_stage(work_ctx, init_img);
            }
        }

        // y
        ggml_tensor* y = nullptr;
        {
            y                            = ggml_new_tensor_1d(work_ctx, GGML_TYPE_F32, diffusion_model->get_adm_in_channels());
            int out_dim                  = 256;
            int fps_id                   = fps - 1;
            std::vector<float> timesteps = {(float)fps_id, (float)motion_bucket_id, augmentation_level};
            set_timestep_embedding(timesteps, y, out_dim);
        }
        int64_t t1 = ggml_time_ms();
        LOG_DEBUG("computing svd condition graph completed, taking %" PRId64 " ms", t1 - t0);
        return {c_crossattn, y, c_concat};
    }

    std::vector<float> process_timesteps(const std::vector<float>& timesteps,
                                         ggml_tensor* init_latent,
                                         ggml_tensor* denoise_mask) {
        if (diffusion_model->get_desc() == "Wan2.2-TI2V-5B") {
            auto new_timesteps = std::vector<float>(init_latent->ne[2], timesteps[0]);

            if (denoise_mask != nullptr) {
                float value = ggml_ext_tensor_get_f32(denoise_mask, 0, 0, 0, 0);
                if (value == 0.f) {
                    new_timesteps[0] = 0.f;
                }
            }
            return new_timesteps;
        } else {
            return timesteps;
        }
    }

    // a = a * mask + b * (1 - mask)
    void apply_mask(ggml_tensor* a, ggml_tensor* b, ggml_tensor* mask) {
        for (int64_t i0 = 0; i0 < a->ne[0]; i0++) {
            for (int64_t i1 = 0; i1 < a->ne[1]; i1++) {
                for (int64_t i2 = 0; i2 < a->ne[2]; i2++) {
                    for (int64_t i3 = 0; i3 < a->ne[3]; i3++) {
                        float a_value    = ggml_ext_tensor_get_f32(a, i0, i1, i2, i3);
                        float b_value    = ggml_ext_tensor_get_f32(b, i0, i1, i2, i3);
                        float mask_value = ggml_ext_tensor_get_f32(mask, i0 % mask->ne[0], i1 % mask->ne[1], i2 % mask->ne[2], i3 % mask->ne[3]);
                        ggml_ext_tensor_set_f32(a, a_value * mask_value + b_value * (1 - mask_value), i0, i1, i2, i3);
                    }
                }
            }
        }
    }

    void preview_image(ggml_context* work_ctx,
                       int step,
                       ggml_tensor* latents,
                       enum SDVersion version,
                       preview_t preview_mode,
                       ggml_tensor* result,
                       std::function<void(int, int, sd_image_t*, bool, void*)> step_callback,
                       void* step_callback_data,
                       bool is_noisy) {
        const uint32_t channel = 3;
        uint32_t width         = static_cast<uint32_t>(latents->ne[0]);
        uint32_t height        = static_cast<uint32_t>(latents->ne[1]);
        uint32_t dim           = static_cast<uint32_t>(latents->ne[ggml_n_dims(latents) - 1]);

        if (preview_mode == PREVIEW_PROJ) {
            int patch_sz                           = 1;
            const float(*latent_rgb_proj)[channel] = nullptr;
            float* latent_rgb_bias                 = nullptr;

            if (dim == 128) {
                if (sd_version_is_flux2(version)) {
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
                    // unknown model
                    return;
                }
            } else if (dim == 16) {
                // 16 channels VAE -> Flux or SD3

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
                    // unknown model
                    return;
                }

            } else if (dim == 4) {
                // 4 channels VAE
                if (sd_version_is_sdxl(version)) {
                    latent_rgb_proj = sdxl_latent_rgb_proj;
                    latent_rgb_bias = sdxl_latent_rgb_bias;
                } else if (sd_version_is_sd1(version) || sd_version_is_sd2(version)) {
                    latent_rgb_proj = sd_latent_rgb_proj;
                    latent_rgb_bias = sd_latent_rgb_bias;
                } else {
                    // unknown model
                    LOG_WARN("No latent to RGB projection known for this model");
                    return;
                }
            } else if (dim == 3) {
                // Do nothing, assuming already RGB latents
            } else {
                LOG_WARN("No latent to RGB projection known for this model");
                // unknown latent space
                return;
            }

            uint32_t frames = 1;
            if (ggml_n_dims(latents) == 4) {
                frames = static_cast<uint32_t>(latents->ne[2]);
            }

            uint32_t img_width  = width * patch_sz;
            uint32_t img_height = height * patch_sz;

            uint8_t* data = (uint8_t*)malloc(frames * img_width * img_height * channel * sizeof(uint8_t));

            preview_latent_video(data, latents, latent_rgb_proj, latent_rgb_bias, patch_sz);
            sd_image_t* images = (sd_image_t*)malloc(frames * sizeof(sd_image_t));
            for (uint32_t i = 0; i < frames; i++) {
                images[i] = {img_width, img_height, channel, data + i * img_width * img_height * channel};
            }
            step_callback(step, frames, images, is_noisy, step_callback_data);
            free(data);
            free(images);
        } else {
            if (preview_mode == PREVIEW_VAE || preview_mode == PREVIEW_TAE) {
                if (preview_vae) {
                    latents = preview_vae->diffusion_to_vae_latents(work_ctx, latents);
                    result  = preview_vae->decode(n_threads, work_ctx, latents, vae_tiling_params, false, circular_x, circular_y, result, true);
                } else {
                    latents = first_stage_model->diffusion_to_vae_latents(work_ctx, latents);
                    result  = first_stage_model->decode(n_threads, work_ctx, latents, vae_tiling_params, false, circular_x, circular_y, result, true);
                }
            } else {
                return;
            }

            ggml_ext_tensor_clamp_inplace(result, 0.0f, 1.0f);
            uint32_t frames = 1;
            if (ggml_n_dims(latents) == 4) {
                frames = static_cast<uint32_t>(result->ne[2]);
            }

            sd_image_t* images = (sd_image_t*)malloc(frames * sizeof(sd_image_t));
            // print_ggml_tensor(result,true);
            for (size_t i = 0; i < frames; i++) {
                images[i].width   = static_cast<uint32_t>(result->ne[0]);
                images[i].height  = static_cast<uint32_t>(result->ne[1]);
                images[i].channel = 3;
                images[i].data    = ggml_tensor_to_sd_image(result, static_cast<int>(i), ggml_n_dims(latents) == 4);
            }

            step_callback(step, frames, images, is_noisy, step_callback_data);

            ggml_ext_tensor_scale_inplace(result, 0);
            for (uint32_t i = 0; i < frames; i++) {
                free(images[i].data);
            }

            free(images);
        }
    }

    ggml_tensor* sample(ggml_context* work_ctx,
                        std::shared_ptr<DiffusionModel> work_diffusion_model,
                        bool inverse_noise_scaling,
                        ggml_tensor* init_latent,
                        ggml_tensor* noise,
                        SDCondition cond,
                        SDCondition uncond,
                        SDCondition img_cond,
                        ggml_tensor* control_hint,
                        float control_strength,
                        sd_guidance_params_t guidance,
                        float eta,
                        int shifted_timestep,
                        sample_method_t method,
                        const std::vector<float>& sigmas,
                        int start_merge_step,
                        SDCondition id_cond,
                        std::vector<ggml_tensor*> ref_latents = {},
                        bool increase_ref_index               = false,
                        ggml_tensor* denoise_mask             = nullptr,
                        ggml_tensor* vace_context             = nullptr,
                        float vace_strength                   = 1.f,
                        const sd_cache_params_t* cache_params = nullptr) {
        if (shifted_timestep > 0 && !sd_version_is_sdxl(version)) {
            LOG_WARN("timestep shifting is only supported for SDXL models!");
            shifted_timestep = 0;
        }
        std::vector<int> skip_layers(guidance.slg.layers, guidance.slg.layers + guidance.slg.layer_count);

        float cfg_scale = guidance.txt_cfg;
        if (cfg_scale < 1.f) {
            if (cfg_scale == 0.f) {
                // Diffusers follow the convention from the original paper
                // (https://arxiv.org/abs/2207.12598v1), so many distilled model docs
                // recommend 0 as guidance; warn the user that it'll disable prompt folowing
                LOG_WARN("unconditioned mode, images won't follow the prompt (use cfg-scale=1 for distilled models)");
            } else {
                LOG_WARN("cfg value out of expected range may produce unexpected results");
            }
        }

        float img_cfg_scale = std::isfinite(guidance.img_cfg) ? guidance.img_cfg : guidance.txt_cfg;
        float slg_scale     = guidance.slg.scale;

        if (img_cfg_scale != cfg_scale && !sd_version_is_inpaint_or_unet_edit(version)) {
            LOG_WARN("2-conditioning CFG is not supported with this model, disabling it for better performance...");
            img_cfg_scale = cfg_scale;
        }

        SampleCacheRuntime cache_runtime = init_sample_cache_runtime(version, cache_params, denoiser.get(), sigmas);

        size_t steps   = sigmas.size() - 1;
        ggml_tensor* x = ggml_ext_dup_and_cpy_tensor(work_ctx, init_latent);

        if (noise) {
            x = denoiser->noise_scaling(sigmas[0], noise, x);
        }

        ggml_tensor* noised_input = ggml_dup_tensor(work_ctx, x);

        bool has_unconditioned = img_cfg_scale != 1.0 && uncond.c_crossattn != nullptr;
        bool has_img_cond      = cfg_scale != img_cfg_scale && img_cond.c_crossattn != nullptr;
        bool has_skiplayer     = slg_scale != 0.0 && skip_layers.size() > 0;

        // denoise wrapper
        ggml_tensor* out_cond     = ggml_dup_tensor(work_ctx, x);
        ggml_tensor* out_uncond   = nullptr;
        ggml_tensor* out_skip     = nullptr;
        ggml_tensor* out_img_cond = nullptr;

        if (has_unconditioned) {
            out_uncond = ggml_dup_tensor(work_ctx, x);
        }
        if (has_skiplayer) {
            if (sd_version_is_dit(version)) {
                out_skip = ggml_dup_tensor(work_ctx, x);
            } else {
                has_skiplayer = false;
                LOG_WARN("SLG is incompatible with %s models", model_version_to_str[version]);
            }
        }
        if (has_img_cond) {
            out_img_cond = ggml_dup_tensor(work_ctx, x);
        }
        ggml_tensor* denoised = ggml_dup_tensor(work_ctx, x);

        int64_t t0 = ggml_time_us();

        ggml_tensor* preview_tensor = nullptr;
        auto sd_preview_mode        = sd_get_preview_mode();
        if (sd_preview_mode != PREVIEW_NONE && sd_preview_mode != PREVIEW_PROJ) {
            int64_t W = x->ne[0] * get_vae_scale_factor();
            int64_t H = x->ne[1] * get_vae_scale_factor();
            if (ggml_n_dims(x) == 4) {
                // assuming video mode (if batch processing gets implemented this will break)
                int64_t T = x->ne[2];
                if (sd_version_is_wan(version)) {
                    T = ((T - 1) * 4) + 1;
                }
                preview_tensor = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32,
                                                    W,
                                                    H,
                                                    T,
                                                    3);
            } else {
                preview_tensor = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32,
                                                    W,
                                                    H,
                                                    3,
                                                    x->ne[3]);
            }
        }

        auto denoise = [&](ggml_tensor* input, float sigma, int step) -> ggml_tensor* {
            auto sd_preview_cb      = sd_get_preview_callback();
            auto sd_preview_cb_data = sd_get_preview_callback_data();
            auto sd_preview_mode    = sd_get_preview_mode();
            if (step == 1 || step == -1) {
                pretty_progress(0, (int)steps, 0);
            }

            DiffusionParams diffusion_params;
            SampleStepCacheDispatcher step_cache(cache_runtime, step, sigma);

            std::vector<float> scaling = denoiser->get_scalings(sigma);
            GGML_ASSERT(scaling.size() == 3);
            float c_skip = scaling[0];
            float c_out  = scaling[1];
            float c_in   = scaling[2];

            float t = denoiser->sigma_to_t(sigma);
            std::vector<float> timesteps_vec;
            if (shifted_timestep > 0 && sd_version_is_sdxl(version)) {
                float shifted_t_float = t * (float(shifted_timestep) / float(TIMESTEPS));
                int64_t shifted_t     = static_cast<int64_t>(roundf(shifted_t_float));
                shifted_t             = std::max((int64_t)0, std::min((int64_t)(TIMESTEPS - 1), shifted_t));
                LOG_DEBUG("shifting timestep from %.2f to %" PRId64 " (sigma: %.4f)", t, shifted_t, sigma);
                timesteps_vec.assign(1, (float)shifted_t);
            } else if (sd_version_is_anima(version)) {
                // Anima uses normalized flow timesteps.
                timesteps_vec.assign(1, t / static_cast<float>(TIMESTEPS));
            } else if (sd_version_is_z_image(version)) {
                timesteps_vec.assign(1, 1000.f - t);
            } else {
                timesteps_vec.assign(1, t);
            }

            timesteps_vec = process_timesteps(timesteps_vec, init_latent, denoise_mask);

            if (cache_runtime.spectrum_enabled && cache_runtime.spectrum.should_predict()) {
                cache_runtime.spectrum.predict(denoised);

                if (denoise_mask != nullptr) {
                    apply_mask(denoised, init_latent, denoise_mask);
                }

                if (sd_preview_cb != nullptr && sd_should_preview_denoised()) {
                    if (step % sd_get_preview_interval() == 0) {
                        preview_image(work_ctx, step, denoised, version, sd_preview_mode, preview_tensor, sd_preview_cb, sd_preview_cb_data, false);
                    }
                }

                int64_t t1 = ggml_time_us();
                if (step > 0 || step == -(int)steps) {
                    int showstep = std::abs(step);
                    pretty_progress(showstep, (int)steps, (t1 - t0) / 1000000.f / showstep);
                }
                return denoised;
            }

            auto timesteps = vector_to_ggml_tensor(work_ctx, timesteps_vec);
            std::vector<float> guidance_vec(1, guidance.distilled_guidance);
            auto guidance_tensor = vector_to_ggml_tensor(work_ctx, guidance_vec);

            copy_ggml_tensor(noised_input, input);
            // noised_input = noised_input * c_in
            ggml_ext_tensor_scale_inplace(noised_input, c_in);

            if (denoise_mask != nullptr && version == VERSION_WAN2_2_TI2V) {
                apply_mask(noised_input, init_latent, denoise_mask);
            }
            if (sd_preview_cb != nullptr && sd_should_preview_noisy()) {
                if (step % sd_get_preview_interval() == 0) {
                    preview_image(work_ctx, step, noised_input, version, sd_preview_mode, preview_tensor, sd_preview_cb, sd_preview_cb_data, true);
                }
            }

            std::vector<ggml_tensor*> controls;

            if (control_hint != nullptr && control_net != nullptr) {
                if (control_net->compute(n_threads, noised_input, control_hint, timesteps, cond.c_crossattn, cond.c_vector)) {
                    controls = control_net->controls;
                } else {
                    LOG_ERROR("controlnet compute failed");
                }
                // print_ggml_tensor(controls[12]);
                // GGML_ASSERT(0);
            }

            diffusion_params.x                  = noised_input;
            diffusion_params.timesteps          = timesteps;
            diffusion_params.guidance           = guidance_tensor;
            diffusion_params.ref_latents        = ref_latents;
            diffusion_params.increase_ref_index = increase_ref_index;
            diffusion_params.controls           = controls;
            diffusion_params.control_strength   = control_strength;
            diffusion_params.vace_context       = vace_context;
            diffusion_params.vace_strength      = vace_strength;

            auto run_diffusion_condition = [&](const SDCondition* condition, ggml_tensor** output_tensor) -> bool {
                if (step_cache.before_condition(condition, diffusion_params.x, *output_tensor)) {
                    return true;
                }

                if (!work_diffusion_model->compute(n_threads,
                                                   diffusion_params,
                                                   output_tensor)) {
                    LOG_ERROR("diffusion model compute failed");
                    return false;
                }

                step_cache.after_condition(condition, diffusion_params.x, *output_tensor);
                return true;
            };

            const SDCondition* active_condition = nullptr;
            ggml_tensor** active_output         = &out_cond;
            if (start_merge_step == -1 || step <= start_merge_step) {
                // cond
                diffusion_params.context  = cond.c_crossattn;
                diffusion_params.c_concat = cond.c_concat;
                diffusion_params.y        = cond.c_vector;
                active_condition          = &cond;
            } else {
                diffusion_params.context  = id_cond.c_crossattn;
                diffusion_params.c_concat = cond.c_concat;
                diffusion_params.y        = id_cond.c_vector;
                active_condition          = &id_cond;
            }

            if (!run_diffusion_condition(active_condition, active_output)) {
                return nullptr;
            }

            bool current_step_skipped = step_cache.is_step_skipped();

            float* negative_data = nullptr;
            if (has_unconditioned) {
                // uncond
                if (!current_step_skipped && control_hint != nullptr && control_net != nullptr) {
                    if (control_net->compute(n_threads, noised_input, control_hint, timesteps, uncond.c_crossattn, uncond.c_vector)) {
                        controls = control_net->controls;
                    } else {
                        LOG_ERROR("controlnet compute failed");
                    }
                }
                current_step_skipped      = step_cache.is_step_skipped();
                diffusion_params.controls = controls;
                diffusion_params.context  = uncond.c_crossattn;
                diffusion_params.c_concat = uncond.c_concat;
                diffusion_params.y        = uncond.c_vector;
                if (!run_diffusion_condition(&uncond, &out_uncond)) {
                    return nullptr;
                }
                negative_data = (float*)out_uncond->data;
            }

            float* img_cond_data = nullptr;
            if (has_img_cond) {
                diffusion_params.context  = img_cond.c_crossattn;
                diffusion_params.c_concat = img_cond.c_concat;
                diffusion_params.y        = img_cond.c_vector;
                if (!run_diffusion_condition(&img_cond, &out_img_cond)) {
                    return nullptr;
                }
                img_cond_data = (float*)out_img_cond->data;
            }

            int step_count         = static_cast<int>(sigmas.size());
            bool is_skiplayer_step = has_skiplayer && step > (int)(guidance.slg.layer_start * step_count) && step < (int)(guidance.slg.layer_end * step_count);
            float* skip_layer_data = has_skiplayer ? (float*)out_skip->data : nullptr;
            if (is_skiplayer_step) {
                LOG_DEBUG("Skipping layers at step %d\n", step);
                if (!step_cache.is_step_skipped()) {
                    // skip layer (same as conditioned)
                    diffusion_params.context     = cond.c_crossattn;
                    diffusion_params.c_concat    = cond.c_concat;
                    diffusion_params.y           = cond.c_vector;
                    diffusion_params.skip_layers = skip_layers;
                    if (!work_diffusion_model->compute(n_threads,
                                                       diffusion_params,
                                                       &out_skip)) {
                        LOG_ERROR("diffusion model compute failed");
                        return nullptr;
                    }
                }
                skip_layer_data = (float*)out_skip->data;
            }
            float* vec_denoised  = (float*)denoised->data;
            float* vec_input     = (float*)input->data;
            float* positive_data = (float*)out_cond->data;
            int ne_elements      = (int)ggml_nelements(denoised);

            if (shifted_timestep > 0 && sd_version_is_sdxl(version)) {
                int64_t shifted_t_idx              = static_cast<int64_t>(roundf(timesteps_vec[0]));
                float shifted_sigma                = denoiser->t_to_sigma((float)shifted_t_idx);
                std::vector<float> shifted_scaling = denoiser->get_scalings(shifted_sigma);
                float shifted_c_skip               = shifted_scaling[0];
                float shifted_c_out                = shifted_scaling[1];
                float shifted_c_in                 = shifted_scaling[2];

                c_skip = shifted_c_skip * c_in / shifted_c_in;
                c_out  = shifted_c_out;
            }

            for (int i = 0; i < ne_elements; i++) {
                float latent_result = positive_data[i];
                if (has_unconditioned) {
                    // out_uncond + cfg_scale * (out_cond - out_uncond)
                    if (has_img_cond) {
                        // out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)
                        latent_result = negative_data[i] + img_cfg_scale * (img_cond_data[i] - negative_data[i]) + cfg_scale * (positive_data[i] - img_cond_data[i]);
                    } else {
                        // img_cfg_scale == cfg_scale
                        latent_result = negative_data[i] + cfg_scale * (positive_data[i] - negative_data[i]);
                    }
                } else if (has_img_cond) {
                    // img_cfg_scale == 1
                    latent_result = img_cond_data[i] + cfg_scale * (positive_data[i] - img_cond_data[i]);
                }
                if (is_skiplayer_step) {
                    latent_result = latent_result + (positive_data[i] - skip_layer_data[i]) * slg_scale;
                }
                // v = latent_result, eps = latent_result
                // denoised = (v * c_out + input * c_skip) or (input + eps * c_out)
                vec_denoised[i] = latent_result * c_out + vec_input[i] * c_skip;
            }

            if (cache_runtime.spectrum_enabled) {
                cache_runtime.spectrum.update(denoised);
            }

            if (denoise_mask != nullptr) {
                apply_mask(denoised, init_latent, denoise_mask);
            }

            if (sd_preview_cb != nullptr && sd_should_preview_denoised()) {
                if (step % sd_get_preview_interval() == 0) {
                    preview_image(work_ctx, step, denoised, version, sd_preview_mode, preview_tensor, sd_preview_cb, sd_preview_cb_data, false);
                }
            }

            int64_t t1 = ggml_time_us();
            if (step > 0 || step == -(int)steps) {
                int showstep = std::abs(step);
                pretty_progress(showstep, (int)steps, (t1 - t0) / 1000000.f / showstep);
                // LOG_INFO("step %d sampling completed taking %.2fs", step, (t1 - t0) * 1.0f / 1000000);
            }
            return denoised;
        };

        if (!sample_k_diffusion(method, denoise, work_ctx, x, sigmas, sampler_rng, eta)) {
            LOG_ERROR("Diffusion model sampling failed");
            if (control_net) {
                control_net->free_control_ctx();
                control_net->free_compute_buffer();
            }
            diffusion_model->free_compute_buffer();
            return NULL;
        }

        size_t total_steps = sigmas.size() > 0 ? sigmas.size() - 1 : 0;
        log_sample_cache_summary(cache_runtime, total_steps);

        if (inverse_noise_scaling) {
            x = denoiser->inverse_noise_scaling(sigmas[sigmas.size() - 1], x);
        }

        if (control_net) {
            control_net->free_control_ctx();
            control_net->free_compute_buffer();
        }
        work_diffusion_model->free_compute_buffer();
        return x;
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
            } else if (sd_version_is_flux2(version)) {
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

    ggml_tensor* generate_init_latent(ggml_context* work_ctx,
                                      int width,
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
        ggml_tensor* init_latent;
        if (video) {
            init_latent = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, W, H, T, C);
        } else {
            init_latent = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, W, H, C, 1);
        }
        ggml_set_f32(init_latent, 0.f);
        return init_latent;
    }

    ggml_tensor* encode_to_vae_latents(ggml_context* work_ctx, ggml_tensor* x) {
        ggml_tensor* vae_output = first_stage_model->encode(n_threads, work_ctx, x, vae_tiling_params, circular_x, circular_y);
        ggml_tensor* latents    = first_stage_model->vae_output_to_latents(work_ctx, vae_output, rng);
        return latents;
    }

    ggml_tensor* encode_first_stage(ggml_context* work_ctx, ggml_tensor* x) {
        ggml_tensor* latents = encode_to_vae_latents(work_ctx, x);
        if (version != VERSION_SD1_PIX2PIX) {
            latents = first_stage_model->vae_to_diffuison_latents(work_ctx, latents);
        }
        return latents;
    }

    ggml_tensor* decode_first_stage(ggml_context* work_ctx, ggml_tensor* x, bool decode_video = false) {
        x = first_stage_model->diffusion_to_vae_latents(work_ctx, x);
        x = first_stage_model->decode(n_threads, work_ctx, x, vae_tiling_params, decode_video, circular_x, circular_y);
        return x;
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
    sd_ctx_params->diffusion_flash_attn    = false;
    sd_ctx_params->circular_x              = false;
    sd_ctx_params->circular_y              = false;
    sd_ctx_params->chroma_use_dit_mask     = true;
    sd_ctx_params->chroma_use_t5_mask      = false;
    sd_ctx_params->chroma_t5_mask_pad      = 1;
}

char* sd_ctx_params_to_str(const sd_ctx_params_t* sd_ctx_params) {
    char* buf = (char*)malloc(4096);
    if (!buf)
        return nullptr;
    buf[0] = '\0';
    // TODO devices
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
    sd_vid_gen_params->vae_tiling_params                     = {false, 0, 0, 0.5f, 0.0f, 0.0f};
    sd_cache_params_init(&sd_vid_gen_params->cache);
}

struct sd_ctx_t {
    StableDiffusionGGML* sd = nullptr;
};

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
    if (sample_method == LCM_SAMPLE_METHOD) {
        return LCM_SCHEDULER;
    }
    return DISCRETE_SCHEDULER;
}

sd_image_t* generate_image_internal(sd_ctx_t* sd_ctx,
                                    ggml_context* work_ctx,
                                    ggml_tensor* init_latent,
                                    std::string prompt,
                                    std::string negative_prompt,
                                    int clip_skip,
                                    sd_guidance_params_t guidance,
                                    float eta,
                                    int shifted_timestep,
                                    int width,
                                    int height,
                                    enum sample_method_t sample_method,
                                    const std::vector<float>& sigmas,
                                    int64_t seed,
                                    int batch_count,
                                    sd_image_t control_image,
                                    float control_strength,
                                    sd_pm_params_t pm_params,
                                    std::vector<sd_image_t*> ref_images,
                                    std::vector<ggml_tensor*> ref_latents,
                                    bool increase_ref_index,
                                    ggml_tensor* concat_latent            = nullptr,
                                    ggml_tensor* denoise_mask             = nullptr,
                                    const sd_cache_params_t* cache_params = nullptr) {
    if (seed < 0) {
        // Generally, when using the provided command line, the seed is always >0.
        // However, to prevent potential issues if 'stable-diffusion.cpp' is invoked as a library
        // by a third party with a seed <0, let's incorporate randomization here.
        srand((int)time(nullptr));
        seed = rand();
    }

    if (!std::isfinite(guidance.img_cfg)) {
        guidance.img_cfg = guidance.txt_cfg;
    }

    int sample_steps = static_cast<int>(sigmas.size() - 1);

    int64_t t0 = ggml_time_ms();

    ConditionerParams condition_params;
    condition_params.text            = prompt;
    condition_params.clip_skip       = clip_skip;
    condition_params.width           = width;
    condition_params.height          = height;
    condition_params.ref_images      = ref_images;
    condition_params.adm_in_channels = static_cast<int>(sd_ctx->sd->diffusion_model->get_adm_in_channels());

    // Photo Maker
    SDCondition id_cond = sd_ctx->sd->get_pmid_conditon(work_ctx, pm_params, condition_params);

    // Get learned condition
    condition_params.zero_out_masked = false;
    SDCondition cond                 = sd_ctx->sd->cond_stage_model->get_learned_condition(work_ctx,
                                                                                           sd_ctx->sd->n_threads,
                                                                                           condition_params);

    SDCondition uncond;
    if (guidance.txt_cfg != 1.0 ||
        (sd_version_is_inpaint_or_unet_edit(sd_ctx->sd->version) && guidance.txt_cfg != guidance.img_cfg)) {
        bool zero_out_masked = false;
        if (sd_version_is_sdxl(sd_ctx->sd->version) && negative_prompt.size() == 0 && !sd_ctx->sd->is_using_edm_v_parameterization) {
            zero_out_masked = true;
        }
        condition_params.text            = negative_prompt;
        condition_params.zero_out_masked = zero_out_masked;
        uncond                           = sd_ctx->sd->cond_stage_model->get_learned_condition(work_ctx,
                                                                                               sd_ctx->sd->n_threads,
                                                                                               condition_params);
    }
    int64_t t1 = ggml_time_ms();
    LOG_INFO("get_learned_condition completed, taking %" PRId64 " ms", t1 - t0);

    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->cond_stage_model->free_params_buffer();
    }

    // Control net hint
    ggml_tensor* image_hint = nullptr;
    if (control_image.data != nullptr) {
        image_hint = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, width, height, 3, 1);
        sd_image_to_ggml_tensor(control_image, image_hint);
    }

    // Sample
    std::vector<ggml_tensor*> final_latents;  // collect latents to decode
    int C = sd_ctx->sd->get_latent_channel();
    int W = width / sd_ctx->sd->get_vae_scale_factor();
    int H = height / sd_ctx->sd->get_vae_scale_factor();

    ggml_tensor* control_latent = nullptr;
    if (sd_version_is_control(sd_ctx->sd->version) && image_hint != nullptr) {
        control_latent = sd_ctx->sd->encode_first_stage(work_ctx, image_hint);
        ggml_ext_tensor_scale_inplace(control_latent, control_strength);
    }

    if (sd_version_is_inpaint(sd_ctx->sd->version)) {
        int64_t mask_channels = 1;
        if (sd_ctx->sd->version == VERSION_FLUX_FILL) {
            mask_channels = 8 * 8;  // flatten the whole mask
        } else if (sd_ctx->sd->version == VERSION_FLEX_2) {
            mask_channels = 1 + init_latent->ne[2];
        }
        auto empty_latent = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, init_latent->ne[0], init_latent->ne[1], mask_channels + init_latent->ne[2], 1);
        // no mask, set the whole image as masked
        for (int64_t x = 0; x < empty_latent->ne[0]; x++) {
            for (int64_t y = 0; y < empty_latent->ne[1]; y++) {
                if (sd_ctx->sd->version == VERSION_FLUX_FILL) {
                    // TODO: this might be wrong
                    for (int64_t c = 0; c < init_latent->ne[2]; c++) {
                        ggml_ext_tensor_set_f32(empty_latent, 0, x, y, c);
                    }
                    for (int64_t c = init_latent->ne[2]; c < empty_latent->ne[2]; c++) {
                        ggml_ext_tensor_set_f32(empty_latent, 1, x, y, c);
                    }
                } else if (sd_ctx->sd->version == VERSION_FLEX_2) {
                    for (int64_t c = 0; c < empty_latent->ne[2]; c++) {
                        // 0x16,1x1,0x16
                        ggml_ext_tensor_set_f32(empty_latent, c == init_latent->ne[2], x, y, c);
                    }
                } else {
                    ggml_ext_tensor_set_f32(empty_latent, 1, x, y, 0);
                    for (int64_t c = 1; c < empty_latent->ne[2]; c++) {
                        ggml_ext_tensor_set_f32(empty_latent, 0, x, y, c);
                    }
                }
            }
        }

        if (sd_ctx->sd->version == VERSION_FLEX_2 && control_latent != nullptr && sd_ctx->sd->control_net == nullptr) {
            bool no_inpaint = concat_latent == nullptr;
            if (no_inpaint) {
                concat_latent = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, init_latent->ne[0], init_latent->ne[1], mask_channels + init_latent->ne[2], 1);
            }
            // fill in the control image here
            for (int64_t x = 0; x < control_latent->ne[0]; x++) {
                for (int64_t y = 0; y < control_latent->ne[1]; y++) {
                    if (no_inpaint) {
                        for (int64_t c = 0; c < concat_latent->ne[2] - control_latent->ne[2]; c++) {
                            // 0x16,1x1,0x16
                            ggml_ext_tensor_set_f32(concat_latent, c == init_latent->ne[2], x, y, c);
                        }
                    }
                    for (int64_t c = 0; c < control_latent->ne[2]; c++) {
                        float v = ggml_ext_tensor_get_f32(control_latent, x, y, c);
                        ggml_ext_tensor_set_f32(concat_latent, v, x, y, concat_latent->ne[2] - control_latent->ne[2] + c);
                    }
                }
            }
        } else if (concat_latent == nullptr) {
            concat_latent = empty_latent;
        }
        cond.c_concat   = concat_latent;
        uncond.c_concat = empty_latent;
        denoise_mask    = nullptr;
    } else if (sd_version_is_unet_edit(sd_ctx->sd->version)) {
        auto empty_latent = ggml_dup_tensor(work_ctx, init_latent);
        ggml_set_f32(empty_latent, 0);
        uncond.c_concat = empty_latent;
        cond.c_concat   = ref_latents[0];
        if (cond.c_concat == nullptr) {
            cond.c_concat = empty_latent;
        }
    } else if (sd_version_is_control(sd_ctx->sd->version)) {
        auto empty_latent = ggml_dup_tensor(work_ctx, init_latent);
        ggml_set_f32(empty_latent, 0);
        uncond.c_concat = empty_latent;
        if (sd_ctx->sd->control_net == nullptr) {
            cond.c_concat = control_latent;
        }
        if (cond.c_concat == nullptr) {
            cond.c_concat = empty_latent;
        }
    }
    SDCondition img_cond;
    if (uncond.c_crossattn != nullptr &&
        (sd_version_is_inpaint_or_unet_edit(sd_ctx->sd->version) && guidance.txt_cfg != guidance.img_cfg)) {
        img_cond = SDCondition(uncond.c_crossattn, uncond.c_vector, cond.c_concat);
    }
    for (int b = 0; b < batch_count; b++) {
        int64_t sampling_start = ggml_time_ms();
        int64_t cur_seed       = seed + b;
        LOG_INFO("generating image: %i/%i - seed %" PRId64, b + 1, batch_count, cur_seed);

        sd_ctx->sd->rng->manual_seed(cur_seed);
        sd_ctx->sd->sampler_rng->manual_seed(cur_seed);
        ggml_tensor* x_t   = init_latent;
        ggml_tensor* noise = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, W, H, C, 1);
        ggml_ext_im_set_randn_f32(noise, sd_ctx->sd->rng);

        int start_merge_step = -1;
        if (sd_ctx->sd->use_pmid) {
            start_merge_step = int(sd_ctx->sd->pmid_model->style_strength / 100.f * sample_steps);
            // if (start_merge_step > 30)
            //     start_merge_step = 30;
            LOG_INFO("PHOTOMAKER: start_merge_step: %d", start_merge_step);
        }

        ggml_tensor* x_0     = sd_ctx->sd->sample(work_ctx,
                                                  sd_ctx->sd->diffusion_model,
                                                  true,
                                                  x_t,
                                                  noise,
                                                  cond,
                                                  uncond,
                                                  img_cond,
                                                  image_hint,
                                                  control_strength,
                                                  guidance,
                                                  eta,
                                                  shifted_timestep,
                                                  sample_method,
                                                  sigmas,
                                                  start_merge_step,
                                                  id_cond,
                                                  ref_latents,
                                                  increase_ref_index,
                                                  denoise_mask,
                                                  nullptr,
                                                  1.0f,
                                                  cache_params);
        int64_t sampling_end = ggml_time_ms();
        if (x_0 != nullptr) {
            // print_ggml_tensor(x_0);
            LOG_INFO("sampling completed, taking %.2fs", (sampling_end - sampling_start) * 1.0f / 1000);
            final_latents.push_back(x_0);
        } else {
            LOG_ERROR("sampling for image %d/%d failed after %.2fs", b + 1, batch_count, (sampling_end - sampling_start) * 1.0f / 1000);
        }
    }

    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->diffusion_model->free_params_buffer();
    }
    int64_t t3 = ggml_time_ms();
    LOG_INFO("generating %" PRId64 " latent images completed, taking %.2fs", final_latents.size(), (t3 - t1) * 1.0f / 1000);

    // Decode to image
    LOG_INFO("decoding %zu latents", final_latents.size());
    std::vector<ggml_tensor*> decoded_images;  // collect decoded images
    for (size_t i = 0; i < final_latents.size(); i++) {
        t1               = ggml_time_ms();
        ggml_tensor* img = sd_ctx->sd->decode_first_stage(work_ctx, final_latents[i] /* x_0 */);
        // print_ggml_tensor(img);
        if (img != nullptr) {
            decoded_images.push_back(img);
        }
        int64_t t2 = ggml_time_ms();
        LOG_INFO("latent %" PRId64 " decoded, taking %.2fs", i + 1, (t2 - t1) * 1.0f / 1000);
    }

    int64_t t4 = ggml_time_ms();
    LOG_INFO("decode_first_stage completed, taking %.2fs", (t4 - t3) * 1.0f / 1000);
    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->first_stage_model->free_params_buffer();
    }

    sd_ctx->sd->lora_stat();

    sd_image_t* result_images = (sd_image_t*)calloc(batch_count, sizeof(sd_image_t));
    if (result_images == nullptr) {
        ggml_free(work_ctx);
        return nullptr;
    }
    memset(result_images, 0, batch_count * sizeof(sd_image_t));

    for (size_t i = 0; i < decoded_images.size(); i++) {
        result_images[i].width   = width;
        result_images[i].height  = height;
        result_images[i].channel = 3;
        result_images[i].data    = ggml_tensor_to_sd_image(decoded_images[i]);
    }
    ggml_free(work_ctx);

    return result_images;
}

sd_image_t* generate_image(sd_ctx_t* sd_ctx, const sd_img_gen_params_t* sd_img_gen_params) {
    sd_ctx->sd->vae_tiling_params = sd_img_gen_params->vae_tiling_params;

    int width  = sd_img_gen_params->width;
    int height = sd_img_gen_params->height;

    int vae_scale_factor            = sd_ctx->sd->get_vae_scale_factor();
    int diffusion_model_down_factor = sd_ctx->sd->get_diffusion_model_down_factor();
    int spatial_multiple            = vae_scale_factor * diffusion_model_down_factor;

    int width_offset  = align_up_offset(width, spatial_multiple);
    int height_offset = align_up_offset(height, spatial_multiple);
    if (width_offset > 0 || height_offset > 0) {
        width += width_offset;
        height += height_offset;
        LOG_WARN("align up %dx%d to %dx%d (multiple=%d)", sd_img_gen_params->width, sd_img_gen_params->height, width, height, spatial_multiple);
    }

    bool circular_x = sd_ctx->sd->circular_x;
    bool circular_y = sd_ctx->sd->circular_y;

    if (!sd_img_gen_params->vae_tiling_params.enabled) {
        if (sd_ctx->sd->first_stage_model) {
            sd_ctx->sd->first_stage_model->set_circular_axes(sd_ctx->sd->circular_x, sd_ctx->sd->circular_y);
        }
        if (sd_ctx->sd->preview_vae) {
            sd_ctx->sd->preview_vae->set_circular_axes(sd_ctx->sd->circular_x, sd_ctx->sd->circular_y);
        }
    } else {
        int tile_size_x, tile_size_y;
        float _overlap;
        int latent_size_x = width / sd_ctx->sd->get_vae_scale_factor();
        int latent_size_y = height / sd_ctx->sd->get_vae_scale_factor();
        sd_ctx->sd->first_stage_model->get_tile_sizes(tile_size_x, tile_size_y, _overlap, sd_img_gen_params->vae_tiling_params, latent_size_x, latent_size_y);

        // force disable circular padding for vae if tiling is enabled unless latent is smaller than tile size
        // otherwise it will cause artifacts at the edges of the tiles
        sd_ctx->sd->circular_x = sd_ctx->sd->circular_x && (tile_size_x >= latent_size_x);
        sd_ctx->sd->circular_y = sd_ctx->sd->circular_y && (tile_size_y >= latent_size_y);

        if (sd_ctx->sd->first_stage_model) {
            sd_ctx->sd->first_stage_model->set_circular_axes(sd_ctx->sd->circular_x, sd_ctx->sd->circular_y);
        }
        if (sd_ctx->sd->preview_vae) {
            sd_ctx->sd->preview_vae->set_circular_axes(sd_ctx->sd->circular_x, sd_ctx->sd->circular_y);
        }

        // disable circular tiling if it's enabled for the VAE
        sd_ctx->sd->circular_x = circular_x && (tile_size_x < latent_size_x);
        sd_ctx->sd->circular_y = circular_y && (tile_size_y < latent_size_y);
    }

    LOG_DEBUG("generate_image %dx%d", width, height);
    if (sd_ctx == nullptr || sd_img_gen_params == nullptr) {
        return nullptr;
    }

    ggml_init_params params;
    params.mem_size   = static_cast<size_t>(1024 * 1024) * 1024;  // 1G
    params.mem_buffer = nullptr;
    params.no_alloc   = false;
    // LOG_DEBUG("mem_size %u ", params.mem_size);

    ggml_context* work_ctx = ggml_init(params);
    if (!work_ctx) {
        LOG_ERROR("ggml_init() failed");
        return nullptr;
    }

    int64_t seed = sd_img_gen_params->seed;
    if (seed < 0) {
        srand((int)time(nullptr));
        seed = rand();
    }
    sd_ctx->sd->rng->manual_seed(seed);
    sd_ctx->sd->sampler_rng->manual_seed(seed);

    size_t t0 = ggml_time_ms();

    sd_ctx->sd->set_flow_shift(sd_img_gen_params->sample_params.flow_shift);

    // Apply lora
    sd_ctx->sd->apply_loras(sd_img_gen_params->loras, sd_img_gen_params->lora_count);

    enum sample_method_t sample_method = sd_img_gen_params->sample_params.sample_method;
    if (sample_method == SAMPLE_METHOD_COUNT) {
        sample_method = sd_get_default_sample_method(sd_ctx);
    }
    LOG_INFO("sampling using %s method", sampling_methods_str[sample_method]);

    int sample_steps = sd_img_gen_params->sample_params.sample_steps;
    std::vector<float> sigmas;
    if (sd_img_gen_params->sample_params.custom_sigmas_count > 0) {
        sigmas = std::vector<float>(sd_img_gen_params->sample_params.custom_sigmas,
                                    sd_img_gen_params->sample_params.custom_sigmas + sd_img_gen_params->sample_params.custom_sigmas_count);
        if (sample_steps != sigmas.size() - 1) {
            sample_steps = static_cast<int>(sigmas.size()) - 1;
            LOG_WARN("sample_steps != custom_sigmas_count - 1, set sample_steps to %d", sample_steps);
        }
    } else {
        scheduler_t scheduler = sd_img_gen_params->sample_params.scheduler;
        if (scheduler == SCHEDULER_COUNT) {
            scheduler = sd_get_default_scheduler(sd_ctx, sample_method);
        }
        sigmas = sd_ctx->sd->denoiser->get_sigmas(sample_steps,
                                                  sd_ctx->sd->get_image_seq_len(height, width),
                                                  scheduler,
                                                  sd_ctx->sd->version);
    }

    ggml_tensor* init_latent   = nullptr;
    ggml_tensor* concat_latent = nullptr;
    ggml_tensor* denoise_mask  = nullptr;
    if (sd_img_gen_params->init_image.data) {
        LOG_INFO("IMG2IMG");

        size_t t_enc = static_cast<size_t>(sample_steps * sd_img_gen_params->strength);
        if (t_enc == sample_steps)
            t_enc--;
        LOG_INFO("target t_enc is %zu steps", t_enc);
        std::vector<float> sigma_sched;
        sigma_sched.assign(sigmas.begin() + sample_steps - t_enc - 1, sigmas.end());
        sigmas = sigma_sched;

        ggml_tensor* init_img = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, width, height, 3, 1);
        ggml_tensor* mask_img = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, width, height, 1, 1);

        sd_image_to_ggml_tensor(sd_img_gen_params->mask_image, mask_img);
        sd_image_to_ggml_tensor(sd_img_gen_params->init_image, init_img);

        if (sd_version_is_inpaint(sd_ctx->sd->version)) {
            int64_t mask_channels = 1;
            if (sd_ctx->sd->version == VERSION_FLUX_FILL) {
                mask_channels = vae_scale_factor * vae_scale_factor;  // flatten the whole mask
            } else if (sd_ctx->sd->version == VERSION_FLEX_2) {
                mask_channels = 1 + sd_ctx->sd->get_latent_channel();
            }
            ggml_tensor* masked_latent = nullptr;

            if (sd_ctx->sd->version != VERSION_FLEX_2) {
                // most inpaint models mask before vae
                ggml_tensor* masked_img = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, width, height, 3, 1);
                ggml_ext_tensor_apply_mask(init_img, mask_img, masked_img);
                masked_latent = sd_ctx->sd->encode_first_stage(work_ctx, masked_img);
                init_latent   = sd_ctx->sd->encode_first_stage(work_ctx, init_img);
            } else {
                // mask after vae
                init_latent   = sd_ctx->sd->encode_first_stage(work_ctx, init_img);
                masked_latent = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, init_latent->ne[0], init_latent->ne[1], init_latent->ne[2], 1);
                ggml_ext_tensor_apply_mask(init_latent, mask_img, masked_latent, 0.);
            }
            concat_latent = ggml_new_tensor_4d(work_ctx,
                                               GGML_TYPE_F32,
                                               masked_latent->ne[0],
                                               masked_latent->ne[1],
                                               mask_channels + masked_latent->ne[2],
                                               1);
            for (int ix = 0; ix < masked_latent->ne[0]; ix++) {
                for (int iy = 0; iy < masked_latent->ne[1]; iy++) {
                    int mx = ix * vae_scale_factor;
                    int my = iy * vae_scale_factor;
                    if (sd_ctx->sd->version == VERSION_FLUX_FILL) {
                        for (int k = 0; k < masked_latent->ne[2]; k++) {
                            float v = ggml_ext_tensor_get_f32(masked_latent, ix, iy, k);
                            ggml_ext_tensor_set_f32(concat_latent, v, ix, iy, k);
                        }
                        // "Encode" 8x8 mask chunks into a flattened 1x64 vector, and concatenate to masked image
                        for (int x = 0; x < vae_scale_factor; x++) {
                            for (int y = 0; y < vae_scale_factor; y++) {
                                float m = ggml_ext_tensor_get_f32(mask_img, mx + x, my + y);
                                // TODO: check if the way the mask is flattened is correct (is it supposed to be x*vae_scale_factor+y or x+vae_scale_factor*y?)
                                // python code was using "b (h vae_scale_factor) (w vae_scale_factor) -> b (vae_scale_factor vae_scale_factor) h w"
                                ggml_ext_tensor_set_f32(concat_latent, m, ix, iy, masked_latent->ne[2] + x * vae_scale_factor + y);
                            }
                        }
                    } else if (sd_ctx->sd->version == VERSION_FLEX_2) {
                        float m = ggml_ext_tensor_get_f32(mask_img, mx, my);
                        // masked image
                        for (int k = 0; k < masked_latent->ne[2]; k++) {
                            float v = ggml_ext_tensor_get_f32(masked_latent, ix, iy, k);
                            ggml_ext_tensor_set_f32(concat_latent, v, ix, iy, k);
                        }
                        // downsampled mask
                        ggml_ext_tensor_set_f32(concat_latent, m, ix, iy, masked_latent->ne[2]);
                        // control (todo: support this)
                        for (int k = 0; k < masked_latent->ne[2]; k++) {
                            ggml_ext_tensor_set_f32(concat_latent, 0, ix, iy, masked_latent->ne[2] + 1 + k);
                        }
                    } else {
                        float m = ggml_ext_tensor_get_f32(mask_img, mx, my);
                        ggml_ext_tensor_set_f32(concat_latent, m, ix, iy, 0);
                        for (int k = 0; k < masked_latent->ne[2]; k++) {
                            float v = ggml_ext_tensor_get_f32(masked_latent, ix, iy, k);
                            ggml_ext_tensor_set_f32(concat_latent, v, ix, iy, k + mask_channels);
                        }
                    }
                }
            }
        } else {
            init_latent = sd_ctx->sd->encode_first_stage(work_ctx, init_img);
        }

        {
            // LOG_WARN("Inpainting with a base model is not great");
            denoise_mask = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, width / vae_scale_factor, height / vae_scale_factor, 1, 1);
            for (int ix = 0; ix < denoise_mask->ne[0]; ix++) {
                for (int iy = 0; iy < denoise_mask->ne[1]; iy++) {
                    int mx  = ix * vae_scale_factor;
                    int my  = iy * vae_scale_factor;
                    float m = ggml_ext_tensor_get_f32(mask_img, mx, my);
                    ggml_ext_tensor_set_f32(denoise_mask, m, ix, iy);
                }
            }
        }
    } else {
        LOG_INFO("TXT2IMG");
        if (sd_version_is_inpaint(sd_ctx->sd->version)) {
            LOG_WARN("This is an inpainting model, this should only be used in img2img mode with a mask");
        }
        init_latent = sd_ctx->sd->generate_init_latent(work_ctx, width, height);
    }

    sd_guidance_params_t guidance = sd_img_gen_params->sample_params.guidance;
    std::vector<sd_image_t*> ref_images;
    for (int i = 0; i < sd_img_gen_params->ref_images_count; i++) {
        ref_images.push_back(&sd_img_gen_params->ref_images[i]);
    }

    std::vector<uint8_t> empty_image_data;
    sd_image_t empty_image = {(uint32_t)width, (uint32_t)height, 3, nullptr};
    if (ref_images.empty() && sd_version_is_unet_edit(sd_ctx->sd->version)) {
        LOG_WARN("This model needs at least one reference image; using an empty reference");
        empty_image_data.resize(width * height * 3);
        ref_images.push_back(&empty_image);
        empty_image.data = empty_image_data.data();
        guidance.img_cfg = 0.f;
    }

    if (ref_images.size() > 0) {
        LOG_INFO("EDIT mode");
    }

    std::vector<ggml_tensor*> ref_latents;
    for (int i = 0; i < ref_images.size(); i++) {
        ggml_tensor* img;
        if (sd_img_gen_params->auto_resize_ref_image) {
            LOG_DEBUG("auto resize ref images");
            sd_image_f32_t ref_image = sd_image_t_to_sd_image_f32_t(*ref_images[i]);
            int VAE_IMAGE_SIZE       = std::min(1024 * 1024, width * height);
            double vae_width         = sqrt(VAE_IMAGE_SIZE * ref_image.width / ref_image.height);
            double vae_height        = vae_width * ref_image.height / ref_image.width;

            int factor = 16;
            if (sd_version_is_qwen_image(sd_ctx->sd->version)) {
                factor = 32;
            }

            vae_height = round(vae_height / factor) * factor;
            vae_width  = round(vae_width / factor) * factor;

            sd_image_f32_t resized_image = resize_sd_image_f32_t(ref_image, static_cast<int>(vae_width), static_cast<int>(vae_height));
            free(ref_image.data);
            ref_image.data = nullptr;

            LOG_DEBUG("resize vae ref image %d from %dx%d to %dx%d", i, ref_image.height, ref_image.width, resized_image.height, resized_image.width);

            img = ggml_new_tensor_4d(work_ctx,
                                     GGML_TYPE_F32,
                                     resized_image.width,
                                     resized_image.height,
                                     3,
                                     1);
            sd_image_f32_to_ggml_tensor(resized_image, img);
            free(resized_image.data);
            resized_image.data = nullptr;
        } else {
            img = ggml_new_tensor_4d(work_ctx,
                                     GGML_TYPE_F32,
                                     ref_images[i]->width,
                                     ref_images[i]->height,
                                     3,
                                     1);
            sd_image_to_ggml_tensor(*ref_images[i], img);
        }

        // print_ggml_tensor(img, false, "img");

        ggml_tensor* latent = sd_ctx->sd->encode_first_stage(work_ctx, img);
        ref_latents.push_back(latent);
    }

    if (sd_img_gen_params->init_image.data != nullptr || sd_img_gen_params->ref_images_count > 0) {
        size_t t1 = ggml_time_ms();
        LOG_INFO("encode_first_stage completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);
    }

    sd_image_t* result_images = generate_image_internal(sd_ctx,
                                                        work_ctx,
                                                        init_latent,
                                                        SAFE_STR(sd_img_gen_params->prompt),
                                                        SAFE_STR(sd_img_gen_params->negative_prompt),
                                                        sd_img_gen_params->clip_skip,
                                                        guidance,
                                                        sd_img_gen_params->sample_params.eta,
                                                        sd_img_gen_params->sample_params.shifted_timestep,
                                                        width,
                                                        height,
                                                        sample_method,
                                                        sigmas,
                                                        seed,
                                                        sd_img_gen_params->batch_count,
                                                        sd_img_gen_params->control_image,
                                                        sd_img_gen_params->control_strength,
                                                        sd_img_gen_params->pm_params,
                                                        ref_images,
                                                        ref_latents,
                                                        sd_img_gen_params->increase_ref_index,
                                                        concat_latent,
                                                        denoise_mask,
                                                        &sd_img_gen_params->cache);

    // restore circular params
    sd_ctx->sd->circular_x = circular_x;
    sd_ctx->sd->circular_y = circular_y;

    size_t t2 = ggml_time_ms();

    LOG_INFO("generate_image completed in %.2fs", (t2 - t0) * 1.0f / 1000);

    return result_images;
}

SD_API sd_image_t* generate_video(sd_ctx_t* sd_ctx, const sd_vid_gen_params_t* sd_vid_gen_params, int* num_frames_out) {
    if (sd_ctx == nullptr || sd_vid_gen_params == nullptr) {
        return nullptr;
    }
    sd_ctx->sd->vae_tiling_params = sd_vid_gen_params->vae_tiling_params;

    std::string prompt          = SAFE_STR(sd_vid_gen_params->prompt);
    std::string negative_prompt = SAFE_STR(sd_vid_gen_params->negative_prompt);

    int width        = sd_vid_gen_params->width;
    int height       = sd_vid_gen_params->height;
    int frames       = sd_vid_gen_params->video_frames;
    frames           = (frames - 1) / 4 * 4 + 1;
    int sample_steps = sd_vid_gen_params->sample_params.sample_steps;

    int vae_scale_factor            = sd_ctx->sd->get_vae_scale_factor();
    int diffusion_model_down_factor = sd_ctx->sd->get_diffusion_model_down_factor();
    int spatial_multiple            = vae_scale_factor * diffusion_model_down_factor;

    int width_offset  = align_up_offset(width, spatial_multiple);
    int height_offset = align_up_offset(height, spatial_multiple);
    if (width_offset > 0 || height_offset > 0) {
        width += width_offset;
        height += height_offset;
        LOG_WARN("align up %dx%d to %dx%d (multiple=%d)", sd_vid_gen_params->width, sd_vid_gen_params->height, width, height, spatial_multiple);
    }
    LOG_INFO("generate_video %dx%dx%d", width, height, frames);

    sd_ctx->sd->set_flow_shift(sd_vid_gen_params->sample_params.flow_shift);

    enum sample_method_t sample_method = sd_vid_gen_params->sample_params.sample_method;
    if (sample_method == SAMPLE_METHOD_COUNT) {
        sample_method = sd_get_default_sample_method(sd_ctx);
    }
    LOG_INFO("sampling using %s method", sampling_methods_str[sample_method]);

    int high_noise_sample_steps = 0;
    if (sd_ctx->sd->high_noise_diffusion_model) {
        high_noise_sample_steps = sd_vid_gen_params->high_noise_sample_params.sample_steps;
    }

    int total_steps = sample_steps;

    if (high_noise_sample_steps > 0) {
        total_steps += high_noise_sample_steps;
    }

    std::vector<float> sigmas;
    if (sd_vid_gen_params->sample_params.custom_sigmas_count > 0) {
        sigmas = std::vector<float>(sd_vid_gen_params->sample_params.custom_sigmas,
                                    sd_vid_gen_params->sample_params.custom_sigmas + sd_vid_gen_params->sample_params.custom_sigmas_count);
        if (total_steps != sigmas.size() - 1) {
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
        }
    } else {
        scheduler_t scheduler = sd_vid_gen_params->sample_params.scheduler;
        if (scheduler == SCHEDULER_COUNT) {
            scheduler = sd_get_default_scheduler(sd_ctx, sample_method);
        }
        sigmas = sd_ctx->sd->denoiser->get_sigmas(total_steps,
                                                  0,
                                                  scheduler,
                                                  sd_ctx->sd->version);
    }

    if (high_noise_sample_steps < 0) {
        // timesteps �?sigmas for Flow models (like wan2.2 a14b)
        for (size_t i = 0; i < sigmas.size(); ++i) {
            if (sigmas[i] < sd_vid_gen_params->moe_boundary) {
                high_noise_sample_steps = static_cast<int>(i);
                break;
            }
        }
        LOG_DEBUG("switching from high noise model at step %d", high_noise_sample_steps);
    }

    ggml_init_params params;
    params.mem_size   = static_cast<size_t>(1024 * 1024) * 1024;  // 1G
    params.mem_buffer = nullptr;
    params.no_alloc   = false;
    // LOG_DEBUG("mem_size %u ", params.mem_size);

    ggml_context* work_ctx = ggml_init(params);
    if (!work_ctx) {
        LOG_ERROR("ggml_init() failed");
        return nullptr;
    }

    int64_t seed = sd_vid_gen_params->seed;
    if (seed < 0) {
        seed = (int)time(nullptr);
    }

    sd_ctx->sd->rng->manual_seed(seed);
    sd_ctx->sd->sampler_rng->manual_seed(seed);

    int64_t t0 = ggml_time_ms();

    // Apply lora
    sd_ctx->sd->apply_loras(sd_vid_gen_params->loras, sd_vid_gen_params->lora_count);

    ggml_tensor* init_latent        = nullptr;
    ggml_tensor* clip_vision_output = nullptr;
    ggml_tensor* concat_latent      = nullptr;
    ggml_tensor* denoise_mask       = nullptr;
    ggml_tensor* vace_context       = nullptr;
    int64_t ref_image_num           = 0;  // for vace
    if (sd_ctx->sd->diffusion_model->get_desc() == "Wan2.1-I2V-14B" ||
        sd_ctx->sd->diffusion_model->get_desc() == "Wan2.2-I2V-14B" ||
        sd_ctx->sd->diffusion_model->get_desc() == "Wan2.1-I2V-1.3B" ||
        sd_ctx->sd->diffusion_model->get_desc() == "Wan2.1-FLF2V-14B") {
        LOG_INFO("IMG2VID");

        if (sd_ctx->sd->diffusion_model->get_desc() == "Wan2.1-I2V-14B" ||
            sd_ctx->sd->diffusion_model->get_desc() == "Wan2.1-I2V-1.3B" ||
            sd_ctx->sd->diffusion_model->get_desc() == "Wan2.1-FLF2V-14B") {
            if (sd_vid_gen_params->init_image.data) {
                clip_vision_output = sd_ctx->sd->get_clip_vision_output(work_ctx, sd_vid_gen_params->init_image, false, -2);
            } else {
                clip_vision_output = sd_ctx->sd->get_clip_vision_output(work_ctx, sd_vid_gen_params->init_image, false, -2, true);
            }

            if (sd_ctx->sd->diffusion_model->get_desc() == "Wan2.1-FLF2V-14B") {
                ggml_tensor* end_image_clip_vision_output = nullptr;
                if (sd_vid_gen_params->end_image.data) {
                    end_image_clip_vision_output = sd_ctx->sd->get_clip_vision_output(work_ctx, sd_vid_gen_params->end_image, false, -2);
                } else {
                    end_image_clip_vision_output = sd_ctx->sd->get_clip_vision_output(work_ctx, sd_vid_gen_params->end_image, false, -2, true);
                }
                clip_vision_output = ggml_ext_tensor_concat(work_ctx, clip_vision_output, end_image_clip_vision_output, 1);
            }

            int64_t t1 = ggml_time_ms();
            LOG_INFO("get_clip_vision_output completed, taking %" PRId64 " ms", t1 - t0);
        }

        int64_t t1         = ggml_time_ms();
        ggml_tensor* image = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, width, height, frames, 3);
        ggml_ext_tensor_iter(image, [&](ggml_tensor* image, int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
            float value = 0.5f;
            if (i2 == 0 && sd_vid_gen_params->init_image.data) {  // start image
                value = *(sd_vid_gen_params->init_image.data + i1 * width * 3 + i0 * 3 + i3);
                value /= 255.f;
            } else if (i2 == frames - 1 && sd_vid_gen_params->end_image.data) {
                value = *(sd_vid_gen_params->end_image.data + i1 * width * 3 + i0 * 3 + i3);
                value /= 255.f;
            }
            ggml_ext_tensor_set_f32(image, value, i0, i1, i2, i3);
        });

        concat_latent = sd_ctx->sd->encode_first_stage(work_ctx, image);  // [b*c, t, h/vae_scale_factor, w/vae_scale_factor]

        int64_t t2 = ggml_time_ms();
        LOG_INFO("encode_first_stage completed, taking %" PRId64 " ms", t2 - t1);

        ggml_tensor* concat_mask = ggml_new_tensor_4d(work_ctx,
                                                      GGML_TYPE_F32,
                                                      concat_latent->ne[0],
                                                      concat_latent->ne[1],
                                                      concat_latent->ne[2],
                                                      4);  // [b*4, t, w/vae_scale_factor, h/vae_scale_factor]
        ggml_ext_tensor_iter(concat_mask, [&](ggml_tensor* concat_mask, int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
            float value = 0.0f;
            if (i2 == 0 && sd_vid_gen_params->init_image.data) {  // start image
                value = 1.0f;
            } else if (i2 == frames - 1 && sd_vid_gen_params->end_image.data && i3 == 3) {
                value = 1.0f;
            }
            ggml_ext_tensor_set_f32(concat_mask, value, i0, i1, i2, i3);
        });

        concat_latent = ggml_ext_tensor_concat(work_ctx, concat_mask, concat_latent, 3);  // [b*(c+4), t, h/vae_scale_factor, w/vae_scale_factor]
    } else if (sd_ctx->sd->diffusion_model->get_desc() == "Wan2.2-TI2V-5B" && sd_vid_gen_params->init_image.data) {
        LOG_INFO("IMG2VID");

        int64_t t1            = ggml_time_ms();
        ggml_tensor* init_img = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, width, height, 3, 1);
        sd_image_to_ggml_tensor(sd_vid_gen_params->init_image, init_img);
        init_img = ggml_reshape_4d(work_ctx, init_img, width, height, 1, 3);

        auto init_image_latent = sd_ctx->sd->encode_to_vae_latents(work_ctx, init_img);  // [b*c, 1, h/16, w/16]

        init_latent  = sd_ctx->sd->generate_init_latent(work_ctx, width, height, frames, true);
        denoise_mask = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, init_latent->ne[0], init_latent->ne[1], init_latent->ne[2], 1);
        ggml_set_f32(denoise_mask, 1.f);

        init_latent = sd_ctx->sd->first_stage_model->diffusion_to_vae_latents(work_ctx, init_latent);

        ggml_ext_tensor_iter(init_image_latent, [&](ggml_tensor* t, int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
            float value = ggml_ext_tensor_get_f32(t, i0, i1, i2, i3);
            ggml_ext_tensor_set_f32(init_latent, value, i0, i1, i2, i3);
            if (i3 == 0) {
                ggml_ext_tensor_set_f32(denoise_mask, 0.f, i0, i1, i2, i3);
            }
        });

        init_latent = sd_ctx->sd->first_stage_model->vae_to_diffuison_latents(work_ctx, init_latent);

        int64_t t2 = ggml_time_ms();
        LOG_INFO("encode_first_stage completed, taking %" PRId64 " ms", t2 - t1);
    } else if (sd_ctx->sd->diffusion_model->get_desc() == "Wan2.1-VACE-1.3B" ||
               sd_ctx->sd->diffusion_model->get_desc() == "Wan2.x-VACE-14B") {
        LOG_INFO("VACE");
        int64_t t1                    = ggml_time_ms();
        ggml_tensor* ref_image_latent = nullptr;
        if (sd_vid_gen_params->init_image.data) {
            ggml_tensor* ref_img = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, width, height, 3, 1);
            sd_image_to_ggml_tensor(sd_vid_gen_params->init_image, ref_img);
            ref_img = ggml_reshape_4d(work_ctx, ref_img, width, height, 1, 3);

            ref_image_latent = sd_ctx->sd->encode_first_stage(work_ctx, ref_img);  // [b*c, 1, h/16, w/16]
            auto zero_latent = ggml_dup_tensor(work_ctx, ref_image_latent);
            ggml_set_f32(zero_latent, 0.f);
            ref_image_latent = ggml_ext_tensor_concat(work_ctx, ref_image_latent, zero_latent, 3);  // [b*2*c, 1, h/16, w/16]
        }

        ggml_tensor* control_video = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, width, height, frames, 3);
        ggml_ext_tensor_iter(control_video, [&](ggml_tensor* control_video, int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
            float value = 0.5f;
            if (i2 < sd_vid_gen_params->control_frames_size) {
                value = sd_image_get_f32(sd_vid_gen_params->control_frames[i2], i0, i1, i3);
            }
            ggml_ext_tensor_set_f32(control_video, value, i0, i1, i2, i3);
        });
        ggml_tensor* mask = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, width, height, frames, 1);
        ggml_set_f32(mask, 1.0f);
        ggml_tensor* inactive = ggml_dup_tensor(work_ctx, control_video);
        ggml_tensor* reactive = ggml_dup_tensor(work_ctx, control_video);

        ggml_ext_tensor_iter(control_video, [&](ggml_tensor* t, int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
            float control_video_value = ggml_ext_tensor_get_f32(t, i0, i1, i2, i3) - 0.5f;
            float mask_value          = ggml_ext_tensor_get_f32(mask, i0, i1, i2, 0);
            float inactive_value      = (control_video_value * (1.f - mask_value)) + 0.5f;
            float reactive_value      = (control_video_value * mask_value) + 0.5f;

            ggml_ext_tensor_set_f32(inactive, inactive_value, i0, i1, i2, i3);
            ggml_ext_tensor_set_f32(reactive, reactive_value, i0, i1, i2, i3);
        });

        inactive = sd_ctx->sd->encode_first_stage(work_ctx, inactive);  // [b*c, t, h/vae_scale_factor, w/vae_scale_factor]
        reactive = sd_ctx->sd->encode_first_stage(work_ctx, reactive);  // [b*c, t, h/vae_scale_factor, w/vae_scale_factor]

        int64_t length = inactive->ne[2];
        if (ref_image_latent) {
            length += 1;
            frames        = static_cast<int>((length - 1) * 4 + 1);
            ref_image_num = 1;
        }
        vace_context = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, inactive->ne[0], inactive->ne[1], length, 96);  // [b*96, t, h/vae_scale_factor, w/vae_scale_factor]
        ggml_ext_tensor_iter(vace_context, [&](ggml_tensor* vace_context, int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
            float value;
            if (i3 < 32) {
                if (ref_image_latent && i2 == 0) {
                    value = ggml_ext_tensor_get_f32(ref_image_latent, i0, i1, 0, i3);
                } else {
                    if (i3 < 16) {
                        value = ggml_ext_tensor_get_f32(inactive, i0, i1, i2 - ref_image_num, i3);
                    } else {
                        value = ggml_ext_tensor_get_f32(reactive, i0, i1, i2 - ref_image_num, i3 - 16);
                    }
                }
            } else {  // mask
                if (ref_image_latent && i2 == 0) {
                    value = 0.f;
                } else {
                    int64_t vae_stride        = vae_scale_factor;
                    int64_t mask_height_index = i1 * vae_stride + (i3 - 32) / vae_stride;
                    int64_t mask_width_index  = i0 * vae_stride + (i3 - 32) % vae_stride;
                    value                     = ggml_ext_tensor_get_f32(mask, mask_width_index, mask_height_index, i2 - ref_image_num, 0);
                }
            }
            ggml_ext_tensor_set_f32(vace_context, value, i0, i1, i2, i3);
        });
        int64_t t2 = ggml_time_ms();
        LOG_INFO("encode_first_stage completed, taking %" PRId64 " ms", t2 - t1);
    }

    if (init_latent == nullptr) {
        init_latent = sd_ctx->sd->generate_init_latent(work_ctx, width, height, frames, true);
    }

    // Get learned condition
    ConditionerParams condition_params;
    condition_params.clip_skip       = sd_vid_gen_params->clip_skip;
    condition_params.zero_out_masked = true;
    condition_params.text            = prompt;

    int64_t t1       = ggml_time_ms();
    SDCondition cond = sd_ctx->sd->cond_stage_model->get_learned_condition(work_ctx,
                                                                           sd_ctx->sd->n_threads,
                                                                           condition_params);
    cond.c_concat    = concat_latent;
    cond.c_vector    = clip_vision_output;
    SDCondition uncond;
    if (sd_vid_gen_params->sample_params.guidance.txt_cfg != 1.0 || sd_vid_gen_params->high_noise_sample_params.guidance.txt_cfg != 1.0) {
        condition_params.text = negative_prompt;
        uncond                = sd_ctx->sd->cond_stage_model->get_learned_condition(work_ctx,
                                                                                    sd_ctx->sd->n_threads,
                                                                                    condition_params);
        uncond.c_concat       = concat_latent;
        uncond.c_vector       = clip_vision_output;
    }
    int64_t t2 = ggml_time_ms();
    LOG_INFO("get_learned_condition completed, taking %" PRId64 " ms", t2 - t1);

    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->cond_stage_model->free_params_buffer();
    }

    int W = width / vae_scale_factor;
    int H = height / vae_scale_factor;
    int T = static_cast<int>(init_latent->ne[2]);
    int C = sd_ctx->sd->get_latent_channel();

    ggml_tensor* final_latent;
    ggml_tensor* x_t   = init_latent;
    ggml_tensor* noise = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, W, H, T, C);
    ggml_ext_im_set_randn_f32(noise, sd_ctx->sd->rng);
    // High Noise Sample
    if (high_noise_sample_steps > 0) {
        LOG_DEBUG("sample(high noise) %dx%dx%d", W, H, T);
        enum sample_method_t high_noise_sample_method = sd_vid_gen_params->high_noise_sample_params.sample_method;
        if (high_noise_sample_method == SAMPLE_METHOD_COUNT) {
            high_noise_sample_method = sd_get_default_sample_method(sd_ctx);
        }
        LOG_INFO("sampling(high noise) using %s method", sampling_methods_str[high_noise_sample_method]);

        int64_t sampling_start = ggml_time_ms();

        std::vector<float> high_noise_sigmas = std::vector<float>(sigmas.begin(), sigmas.begin() + high_noise_sample_steps + 1);
        sigmas                               = std::vector<float>(sigmas.begin() + high_noise_sample_steps, sigmas.end());

        x_t = sd_ctx->sd->sample(work_ctx,
                                 sd_ctx->sd->high_noise_diffusion_model,
                                 false,
                                 x_t,
                                 noise,
                                 cond,
                                 uncond,
                                 {},
                                 nullptr,
                                 0,
                                 sd_vid_gen_params->high_noise_sample_params.guidance,
                                 sd_vid_gen_params->high_noise_sample_params.eta,
                                 sd_vid_gen_params->high_noise_sample_params.shifted_timestep,
                                 high_noise_sample_method,
                                 high_noise_sigmas,
                                 -1,
                                 {},
                                 {},
                                 false,
                                 denoise_mask,
                                 vace_context,
                                 sd_vid_gen_params->vace_strength,
                                 &sd_vid_gen_params->cache);

        int64_t sampling_end = ggml_time_ms();
        LOG_INFO("sampling(high noise) completed, taking %.2fs", (sampling_end - sampling_start) * 1.0f / 1000);
        if (sd_ctx->sd->free_params_immediately) {
            sd_ctx->sd->high_noise_diffusion_model->free_params_buffer();
        }
        noise = nullptr;
    }

    // Sample
    {
        LOG_DEBUG("sample %dx%dx%d", W, H, T);
        int64_t sampling_start = ggml_time_ms();

        final_latent = sd_ctx->sd->sample(work_ctx,
                                          sd_ctx->sd->diffusion_model,
                                          true,
                                          x_t,
                                          noise,
                                          cond,
                                          uncond,
                                          {},
                                          nullptr,
                                          0,
                                          sd_vid_gen_params->sample_params.guidance,
                                          sd_vid_gen_params->sample_params.eta,
                                          sd_vid_gen_params->sample_params.shifted_timestep,
                                          sample_method,
                                          sigmas,
                                          -1,
                                          {},
                                          {},
                                          false,
                                          denoise_mask,
                                          vace_context,
                                          sd_vid_gen_params->vace_strength,
                                          &sd_vid_gen_params->cache);

        int64_t sampling_end = ggml_time_ms();
        LOG_INFO("sampling completed, taking %.2fs", (sampling_end - sampling_start) * 1.0f / 1000);
        if (sd_ctx->sd->free_params_immediately) {
            sd_ctx->sd->diffusion_model->free_params_buffer();
        }
    }

    if (ref_image_num > 0) {
        ggml_tensor* trim_latent = ggml_new_tensor_4d(work_ctx,
                                                      GGML_TYPE_F32,
                                                      final_latent->ne[0],
                                                      final_latent->ne[1],
                                                      final_latent->ne[2] - ref_image_num,
                                                      final_latent->ne[3]);
        ggml_ext_tensor_iter(trim_latent, [&](ggml_tensor* trim_latent, int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
            float value = ggml_ext_tensor_get_f32(final_latent, i0, i1, i2 + ref_image_num, i3);
            ggml_ext_tensor_set_f32(trim_latent, value, i0, i1, i2, i3);
        });
        final_latent = trim_latent;
    }

    int64_t t4 = ggml_time_ms();
    LOG_INFO("generating latent video completed, taking %.2fs", (t4 - t2) * 1.0f / 1000);
    ggml_tensor* vid = sd_ctx->sd->decode_first_stage(work_ctx, final_latent, true);
    int64_t t5       = ggml_time_ms();
    LOG_INFO("decode_first_stage completed, taking %.2fs", (t5 - t4) * 1.0f / 1000);
    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->first_stage_model->free_params_buffer();
    }

    sd_ctx->sd->lora_stat();

    sd_image_t* result_images = (sd_image_t*)calloc(vid->ne[2], sizeof(sd_image_t));
    if (result_images == nullptr) {
        ggml_free(work_ctx);
        return nullptr;
    }
    *num_frames_out = static_cast<int>(vid->ne[2]);

    for (int64_t i = 0; i < vid->ne[2]; i++) {
        result_images[i].width   = static_cast<uint32_t>(vid->ne[0]);
        result_images[i].height  = static_cast<uint32_t>(vid->ne[1]);
        result_images[i].channel = 3;
        result_images[i].data    = ggml_tensor_to_sd_image(vid, static_cast<int>(i), true);
    }
    ggml_free(work_ctx);

    LOG_INFO("generate_video completed in %.2fs", (t5 - t0) * 1.0f / 1000);

    return result_images;
}
