#include "ggml_extend.hpp"

#include "model.h"
#include "rng.hpp"
#include "rng_philox.hpp"
#include "stable-diffusion.h"
#include "util.h"

#include "clip.hpp"
#include "control.hpp"
#include "denoiser.hpp"
#include "esrgan.hpp"
#include "lora.hpp"
#include "tae.hpp"
#include "unet.hpp"
#include "vae.hpp"

const char* model_version_to_str[] = {
    "1.x",
    "2.x",
    "XL",
};

const char* sampling_methods_str[] = {
    "Euler A",
    "Euler",
    "Heun",
    "DPM2",
    "DPM++ (2s)",
    "DPM++ (2M)",
    "modified DPM++ (2M)",
    "LCM",
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

/*=============================================== StableDiffusionGGML ================================================*/

class StableDiffusionGGML {
public:
    SDVersion version;
    bool vae_decode_only         = false;
    bool free_params_immediately = false;

    std::shared_ptr<RNG> rng = std::make_shared<STDDefaultRNG>();
    int n_threads            = -1;
    float scale_factor       = 0.18215f;

    FrozenCLIPEmbedderWithCustomWords cond_stage_model;
    UNetModel diffusion_model;
    AutoEncoderKL first_stage_model;
    bool use_tiny_autoencoder = false;
    bool vae_tiling           = false;

    std::map<std::string, struct ggml_tensor*> tensors;

    std::string lora_model_dir;
    // lora_name => multiplier
    std::unordered_map<std::string, float> curr_lora_state;
    std::map<std::string, LoraModel> loras;

    std::shared_ptr<Denoiser> denoiser = std::make_shared<CompVisDenoiser>();
    ggml_backend_t backend             = NULL;  // general backend
    ggml_type model_data_type          = GGML_TYPE_COUNT;

    TinyAutoEncoder tae_first_stage;
    std::string taesd_path;

    ControlNet control_net;

    StableDiffusionGGML() = default;

    StableDiffusionGGML(int n_threads,
                        bool vae_decode_only,
                        bool free_params_immediately,
                        std::string lora_model_dir,
                        rng_type_t rng_type)
        : n_threads(n_threads),
          vae_decode_only(vae_decode_only),
          free_params_immediately(free_params_immediately),
          lora_model_dir(lora_model_dir) {
        first_stage_model.decode_only = vae_decode_only;
        tae_first_stage.decode_only   = vae_decode_only;
        if (rng_type == STD_DEFAULT_RNG) {
            rng = std::make_shared<STDDefaultRNG>();
        } else if (rng_type == CUDA_RNG) {
            rng = std::make_shared<PhiloxRNG>();
        }
    }

    ~StableDiffusionGGML() {
        ggml_backend_free(backend);
    }

    bool load_from_file(const std::string& model_path,
                        const std::string& vae_path,
                        const std::string control_net_path,
                        const std::string embeddings_path,
                        const std::string& taesd_path,
                        bool vae_tiling_,
                        ggml_type wtype,
                        schedule_t schedule,
                        bool control_net_cpu) {
        use_tiny_autoencoder = taesd_path.size() > 0;
#ifdef SD_USE_CUBLAS
        LOG_DEBUG("Using CUDA backend");
        backend = ggml_backend_cuda_init(0);
#endif
#ifdef SD_USE_METAL
        LOG_DEBUG("Using Metal backend");
        ggml_metal_log_set_callback(ggml_log_callback_default, nullptr);
        backend = ggml_backend_metal_init();
#endif

        if (!backend) {
            LOG_DEBUG("Using CPU backend");
            backend = ggml_backend_cpu_init();
        }
#ifdef SD_USE_FLASH_ATTENTION
#if defined(SD_USE_CUBLAS) || defined(SD_USE_METAL)
        LOG_WARN("Flash Attention not supported with GPU Backend");
#else
        LOG_INFO("Flash Attention enabled");
#endif
#endif
        LOG_INFO("loading model from '%s'", model_path.c_str());
        ModelLoader model_loader;

        vae_tiling = vae_tiling_;

        if (!model_loader.init_from_file(model_path)) {
            LOG_ERROR("init model loader from file failed: '%s'", model_path.c_str());
            return false;
        }

        if (vae_path.size() > 0) {
            LOG_INFO("loading vae from '%s'", vae_path.c_str());
            if (!model_loader.init_from_file(vae_path, "vae.")) {
                LOG_WARN("loading vae from '%s' failed", vae_path.c_str());
            }
        }

        version = model_loader.get_sd_version();
        if (version == VERSION_COUNT) {
            LOG_ERROR("get sd version from file failed: '%s'", model_path.c_str());
            return false;
        }
        if (version == VERSION_XL) {
            scale_factor = 0.13025f;
        }
        cond_stage_model = FrozenCLIPEmbedderWithCustomWords(version);
        diffusion_model  = UNetModel(version);

        LOG_INFO("Stable Diffusion %s ", model_version_to_str[version]);
        if (wtype == GGML_TYPE_COUNT) {
            model_data_type = model_loader.get_sd_wtype();
        } else {
            model_data_type = wtype;
        }
        LOG_INFO("Stable Diffusion weight type: %s", ggml_type_name(model_data_type));

        LOG_DEBUG("loading vocab");
        std::string merges_utf8_str = model_loader.load_merges();
        if (merges_utf8_str.size() == 0) {
            LOG_ERROR("get merges failed: '%s'", model_path.c_str());
            return false;
        }

        cond_stage_model.tokenizer.load_from_merges(merges_utf8_str);

        // create the ggml context for network params
        LOG_DEBUG("ggml tensor size = %d bytes", (int)sizeof(ggml_tensor));

        if (
            !cond_stage_model.alloc_params_buffer(backend, model_data_type) ||
            !diffusion_model.alloc_params_buffer(backend, model_data_type)) {
            return false;
        }

        cond_stage_model.text_model.embd_dir = embeddings_path;

        ggml_type vae_type = model_data_type;
        if (version == VERSION_XL) {
            vae_type = GGML_TYPE_F32;  // avoid nan, not work...
        }

        if (!use_tiny_autoencoder && !first_stage_model.alloc_params_buffer(backend, vae_type)) {
            return false;
        }

        LOG_DEBUG("preparing memory for the weights");
        // prepare memory for the weights
        {
            // cond_stage_model(FrozenCLIPEmbedder)
            cond_stage_model.init_params();
            cond_stage_model.map_by_name(tensors, "cond_stage_model.");

            // diffusion_model(UNetModel)
            diffusion_model.init_params();
            diffusion_model.map_by_name(tensors, "model.diffusion_model.");

            if (!use_tiny_autoencoder) {
                // firest_stage_model(AutoEncoderKL)
                first_stage_model.init_params();
            }
            first_stage_model.map_by_name(tensors, "first_stage_model.");
        }

        struct ggml_init_params params;
        params.mem_size   = static_cast<size_t>(10 * 1024) * 1024;  // 10M
        params.mem_buffer = NULL;
        params.no_alloc   = false;
        // LOG_DEBUG("mem_size %u ", params.mem_size);
        struct ggml_context* ctx = ggml_init(params);  // for  alphas_cumprod and is_using_v_parameterization check
        if (!ctx) {
            LOG_ERROR("ggml_init() failed");
            return false;
        }
        ggml_tensor* alphas_cumprod_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, TIMESTEPS);
        calculate_alphas_cumprod((float*)alphas_cumprod_tensor->data);

        // load weights
        LOG_DEBUG("loading weights");
        int64_t t0 = ggml_time_ms();

        std::map<std::string, struct ggml_tensor*> tensors_need_to_load;
        std::set<std::string> ignore_tensors;
        tensors_need_to_load["alphas_cumprod"] = alphas_cumprod_tensor;
        for (auto& pair : tensors) {
            const std::string& name = pair.first;

            if (use_tiny_autoencoder && starts_with(name, "first_stage_model.")) {
                ignore_tensors.insert(name);
                continue;
            }

            if (vae_decode_only && (starts_with(name, "first_stage_model.encoder") || starts_with(name, "first_stage_model.quant"))) {
                ignore_tensors.insert(name);
                continue;
            }

            tensors_need_to_load.insert(pair);
        }
        bool success = model_loader.load_tensors(tensors_need_to_load, backend, ignore_tensors);
        if (!success) {
            LOG_ERROR("load tensors from model loader failed");
            ggml_free(ctx);
            return false;
        }

        // LOG_DEBUG("model size = %.2fMB", total_size / 1024.0 / 1024.0);

        size_t total_params_size =
            cond_stage_model.params_buffer_size +
            diffusion_model.params_buffer_size +
            first_stage_model.params_buffer_size;
        LOG_INFO("total memory buffer size = %.2fMB (clip %.2fMB, unet %.2fMB, vae %.2fMB)",
                 total_params_size / 1024.0 / 1024.0,
                 cond_stage_model.params_buffer_size / 1024.0 / 1024.0,
                 diffusion_model.params_buffer_size / 1024.0 / 1024.0,
                 first_stage_model.params_buffer_size / 1024.0 / 1024.0);
        int64_t t1 = ggml_time_ms();
        LOG_INFO("loading model from '%s' completed, taking %.2fs", model_path.c_str(), (t1 - t0) * 1.0f / 1000);

        // check is_using_v_parameterization_for_sd2
        bool is_using_v_parameterization = false;
        if (version == VERSION_2_x) {
            if (is_using_v_parameterization_for_sd2(ctx)) {
                is_using_v_parameterization = true;
            }
        }

        if (is_using_v_parameterization) {
            denoiser = std::make_shared<CompVisVDenoiser>();
            LOG_INFO("running in v-prediction mode");
        } else {
            LOG_INFO("running in eps-prediction mode");
        }

        if (schedule != DEFAULT) {
            switch (schedule) {
                case DISCRETE:
                    LOG_INFO("running with discrete schedule");
                    denoiser->schedule = std::make_shared<DiscreteSchedule>();
                    break;
                case KARRAS:
                    LOG_INFO("running with Karras schedule");
                    denoiser->schedule = std::make_shared<KarrasSchedule>();
                    break;
                case DEFAULT:
                    // Don't touch anything.
                    break;
                default:
                    LOG_ERROR("Unknown schedule %i", schedule);
                    abort();
            }
        }

        for (int i = 0; i < TIMESTEPS; i++) {
            denoiser->schedule->alphas_cumprod[i] = ((float*)alphas_cumprod_tensor->data)[i];
            denoiser->schedule->sigmas[i]         = std::sqrt((1 - denoiser->schedule->alphas_cumprod[i]) / denoiser->schedule->alphas_cumprod[i]);
            denoiser->schedule->log_sigmas[i]     = std::log(denoiser->schedule->sigmas[i]);
        }

        LOG_DEBUG("finished loaded file");
        ggml_free(ctx);

        if (control_net_path.size() > 0) {
            ggml_backend_t cn_backend = NULL;
            if (control_net_cpu && !ggml_backend_is_cpu(backend)) {
                LOG_DEBUG("ControlNet: Using CPU backend");
                cn_backend = ggml_backend_cpu_init();
            } else {
                cn_backend = backend;
            }
            if (!control_net.load_from_file(control_net_path, cn_backend, GGML_TYPE_F16 /* just f16 controlnet models */)) {
                return false;
            }
        }

        if (use_tiny_autoencoder) {
            return tae_first_stage.load_from_file(taesd_path, backend);
        }
        return true;
    }

    bool is_using_v_parameterization_for_sd2(ggml_context* work_ctx) {
        struct ggml_tensor* x_t = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, 8, 8, 4, 1);
        ggml_set_f32(x_t, 0.5);
        struct ggml_tensor* c = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, 1024, 2, 1, 1);
        ggml_set_f32(c, 0.5);

        struct ggml_tensor* timesteps = ggml_new_tensor_1d(work_ctx, GGML_TYPE_F32, 1);                                     // [N, ]
        struct ggml_tensor* t_emb     = new_timestep_embedding(work_ctx, NULL, timesteps, diffusion_model.model_channels);  // [N, model_channels]

        int64_t t0 = ggml_time_ms();
        ggml_set_f32(timesteps, 999);
        set_timestep_embedding(timesteps, t_emb, diffusion_model.model_channels);
        struct ggml_tensor* out = ggml_dup_tensor(work_ctx, x_t);
        std::vector<struct ggml_tensor*> controls;
        diffusion_model.alloc_compute_buffer(x_t, c, controls, t_emb);
        diffusion_model.compute(out, n_threads, x_t, NULL, c, controls, 1.0f, t_emb);
        diffusion_model.free_compute_buffer();

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

    void apply_lora(const std::string& lora_name, float multiplier) {
        int64_t t0                 = ggml_time_ms();
        std::string st_file_path   = path_join(lora_model_dir, lora_name + ".safetensors");
        std::string ckpt_file_path = path_join(lora_model_dir, lora_name + ".ckpt");
        std::string file_path;
        if (file_exists(st_file_path)) {
            file_path = st_file_path;
        } else if (file_exists(ckpt_file_path)) {
            file_path = ckpt_file_path;
        } else {
            LOG_WARN("can not find %s or %s for lora %s", st_file_path.c_str(), ckpt_file_path.c_str(), lora_name.c_str());
            return;
        }
        LoraModel lora(file_path);
        if (!lora.load_from_file(backend)) {
            LOG_WARN("load lora tensors from %s failed", file_path.c_str());
            return;
        }

        lora.multiplier = multiplier;
        lora.apply(tensors, n_threads);
        loras[lora_name] = lora;
        lora.free_params_buffer();

        int64_t t1 = ggml_time_ms();

        LOG_INFO("lora '%s' applied, taking %.2fs",
                 lora_name.c_str(),
                 (t1 - t0) * 1.0f / 1000);
    }

    void apply_loras(const std::unordered_map<std::string, float>& lora_state) {
        if (lora_state.size() > 0 && model_data_type != GGML_TYPE_F16 && model_data_type != GGML_TYPE_F32) {
            LOG_WARN("In quantized models when applying LoRA, the images have poor quality.");
        }
        std::unordered_map<std::string, float> lora_state_diff;
        for (auto& kv : lora_state) {
            const std::string& lora_name = kv.first;
            float multiplier             = kv.second;

            if (curr_lora_state.find(lora_name) != curr_lora_state.end()) {
                float curr_multiplier = curr_lora_state[lora_name];
                float multiplier_diff = multiplier - curr_multiplier;
                if (multiplier_diff != 0.f) {
                    lora_state_diff[lora_name] = multiplier_diff;
                }
            } else {
                lora_state_diff[lora_name] = multiplier;
            }
        }

        for (auto& kv : lora_state_diff) {
            apply_lora(kv.first, kv.second);
        }

        curr_lora_state = lora_state;
    }

    std::pair<ggml_tensor*, ggml_tensor*> get_learned_condition(ggml_context* work_ctx,
                                                                const std::string& text,
                                                                int clip_skip,
                                                                int width,
                                                                int height,
                                                                bool force_zero_embeddings = false) {
        cond_stage_model.set_clip_skip(clip_skip);
        auto tokens_and_weights     = cond_stage_model.tokenize(text, true);
        std::vector<int>& tokens    = tokens_and_weights.first;
        std::vector<float>& weights = tokens_and_weights.second;
        int64_t t0                  = ggml_time_ms();
        struct ggml_tensor* pooled  = NULL;
        size_t total_hidden_size    = cond_stage_model.text_model.hidden_size;
        if (version == VERSION_XL) {
            total_hidden_size += cond_stage_model.text_model2.hidden_size;
            pooled = ggml_new_tensor_1d(work_ctx, GGML_TYPE_F32, cond_stage_model.text_model2.projection_dim);
        }
        struct ggml_tensor* hidden_states = ggml_new_tensor_2d(work_ctx,
                                                               GGML_TYPE_F32,
                                                               total_hidden_size,
                                                               cond_stage_model.text_model.max_position_embeddings);  // [N, n_token, hidden_size]
        cond_stage_model.alloc_compute_buffer(work_ctx, (int)tokens.size());
        cond_stage_model.compute(n_threads, tokens, hidden_states, pooled);
        cond_stage_model.free_compute_buffer();
        // if (pooled != NULL) {
        //     print_ggml_tensor(hidden_states);
        //     print_ggml_tensor(pooled);
        // }

        int64_t t1 = ggml_time_ms();
        LOG_DEBUG("computing condition graph completed, taking %" PRId64 " ms", t1 - t0);
        ggml_tensor* result = ggml_dup_tensor(work_ctx, hidden_states);
        {
            float original_mean = ggml_tensor_mean(hidden_states);
            for (int i2 = 0; i2 < hidden_states->ne[2]; i2++) {
                for (int i1 = 0; i1 < hidden_states->ne[1]; i1++) {
                    for (int i0 = 0; i0 < hidden_states->ne[0]; i0++) {
                        float value = ggml_tensor_get_f32(hidden_states, i0, i1, i2);
                        value *= weights[i1];
                        ggml_tensor_set_f32(result, value, i0, i1, i2);
                    }
                }
            }
            float new_mean = ggml_tensor_mean(result);
            ggml_tensor_scale(result, (original_mean / new_mean));
        }
        if (force_zero_embeddings) {
            float* vec = (float*)result->data;
            for (int i = 0; i < ggml_nelements(result); i++) {
                vec[i] = 0;
            }
        }

        ggml_tensor* vec = NULL;
        if (version == VERSION_XL) {
            int out_dim = 256;
            vec         = ggml_new_tensor_1d(work_ctx, GGML_TYPE_F32, diffusion_model.adm_in_channels);
            // [0:1280]
            size_t offset = 0;
            memcpy(vec->data, pooled->data, ggml_nbytes(pooled));
            offset += ggml_nbytes(pooled);

            struct ggml_tensor* timesteps = ggml_new_tensor_1d(work_ctx, GGML_TYPE_F32, 2);
            // original_size_as_tuple
            float orig_width  = (float)width;
            float orig_height = (float)height;
            ggml_tensor_set_f32(timesteps, orig_height, 0);
            ggml_tensor_set_f32(timesteps, orig_width, 1);
            ggml_tensor* embed_view = ggml_view_2d(work_ctx, vec, out_dim, 2, ggml_type_size(GGML_TYPE_F32) * out_dim, offset);
            offset += ggml_nbytes(embed_view);
            set_timestep_embedding(timesteps, embed_view, out_dim);
            // print_ggml_tensor(ggml_reshape_1d(work_ctx, embed_view, out_dim * 2));
            // crop_coords_top_left
            float crop_coord_top  = 0.f;
            float crop_coord_left = 0.f;
            ggml_tensor_set_f32(timesteps, crop_coord_top, 0);
            ggml_tensor_set_f32(timesteps, crop_coord_left, 1);
            embed_view = ggml_view_2d(work_ctx, vec, out_dim, 2, ggml_type_size(GGML_TYPE_F32) * out_dim, offset);
            offset += ggml_nbytes(embed_view);
            set_timestep_embedding(timesteps, embed_view, out_dim);
            // print_ggml_tensor(ggml_reshape_1d(work_ctx, embed_view, out_dim * 2));
            // target_size_as_tuple
            float target_width  = (float)width;
            float target_height = (float)height;
            ggml_tensor_set_f32(timesteps, target_height, 0);
            ggml_tensor_set_f32(timesteps, target_width, 1);
            embed_view = ggml_view_2d(work_ctx, vec, out_dim, 2, ggml_type_size(GGML_TYPE_F32) * out_dim, offset);
            offset += ggml_nbytes(embed_view);
            set_timestep_embedding(timesteps, embed_view, out_dim);
            // print_ggml_tensor(ggml_reshape_1d(work_ctx, embed_view, out_dim * 2));
            GGML_ASSERT(offset == ggml_nbytes(vec));
        }
        // print_ggml_tensor(result);
        return {result, vec};
    }

    ggml_tensor* sample(ggml_context* work_ctx,
                        ggml_tensor* x_t,
                        ggml_tensor* noise,
                        ggml_tensor* c,
                        ggml_tensor* c_vector,
                        ggml_tensor* uc,
                        ggml_tensor* uc_vector,
                        ggml_tensor* control_hint,
                        float cfg_scale,
                        sample_method_t method,
                        const std::vector<float>& sigmas,
                        float control_strength) {
        size_t steps = sigmas.size() - 1;
        // x_t = load_tensor_from_file(work_ctx, "./rand0.bin");
        // print_ggml_tensor(x_t);
        struct ggml_tensor* x = ggml_dup_tensor(work_ctx, x_t);
        copy_ggml_tensor(x, x_t);

        struct ggml_tensor* noised_input = ggml_dup_tensor(work_ctx, x_t);
        struct ggml_tensor* timesteps    = ggml_new_tensor_1d(work_ctx, GGML_TYPE_F32, 1);                                     // [N, ]
        struct ggml_tensor* t_emb        = new_timestep_embedding(work_ctx, NULL, timesteps, diffusion_model.model_channels);  // [N, model_channels]
        struct ggml_tensor* guided_hint  = NULL;
        if (control_hint != NULL) {
            guided_hint = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, noised_input->ne[0], noised_input->ne[1], diffusion_model.model_channels, 1);
            control_net.process_hint(guided_hint, n_threads, control_hint);
            control_net.alloc_compute_buffer(noised_input, guided_hint, c, t_emb);
        }

        diffusion_model.alloc_compute_buffer(noised_input, c, control_net.controls, t_emb, c_vector);

        bool has_unconditioned = cfg_scale != 1.0 && uc != NULL;

        if (noise == NULL) {
            // x = x * sigmas[0]
            ggml_tensor_scale(x, sigmas[0]);
        } else {
            // xi = x + noise * sigma_sched[0]
            ggml_tensor_scale(noise, sigmas[0]);
            ggml_tensor_add(x, noise);
        }

        // denoise wrapper
        struct ggml_tensor* out_cond   = ggml_dup_tensor(work_ctx, x);
        struct ggml_tensor* out_uncond = NULL;
        if (has_unconditioned) {
            out_uncond = ggml_dup_tensor(work_ctx, x);
        }
        struct ggml_tensor* denoised = ggml_dup_tensor(work_ctx, x);

        auto denoise = [&](ggml_tensor* input, float sigma, int step) {
            if (step == 1) {
                pretty_progress(0, (int)steps, 0);
            }
            int64_t t0 = ggml_time_us();

            float c_skip               = 1.0f;
            float c_out                = 1.0f;
            float c_in                 = 1.0f;
            std::vector<float> scaling = denoiser->get_scalings(sigma);

            if (scaling.size() == 3) {  // CompVisVDenoiser
                c_skip = scaling[0];
                c_out  = scaling[1];
                c_in   = scaling[2];
            } else {  // CompVisDenoiser
                c_out = scaling[0];
                c_in  = scaling[1];
            }

            float t = denoiser->schedule->sigma_to_t(sigma);
            ggml_set_f32(timesteps, t);
            set_timestep_embedding(timesteps, t_emb, diffusion_model.model_channels);

            copy_ggml_tensor(noised_input, input);
            // noised_input = noised_input * c_in
            ggml_tensor_scale(noised_input, c_in);

            // cond
            if (control_hint != NULL) {
                control_net.compute(n_threads, noised_input, guided_hint, c, t_emb);
            }
            diffusion_model.compute(out_cond, n_threads, noised_input, NULL, c, control_net.controls, control_strength, t_emb, c_vector);

            float* negative_data = NULL;
            if (has_unconditioned) {
                // uncond
                if (control_hint != NULL) {
                    control_net.compute(n_threads, noised_input, guided_hint, uc, t_emb);
                }

                diffusion_model.compute(out_uncond, n_threads, noised_input, NULL, uc, control_net.controls, control_strength, t_emb, uc_vector);
                negative_data = (float*)out_uncond->data;
            }
            float* vec_denoised  = (float*)denoised->data;
            float* vec_input     = (float*)input->data;
            float* positive_data = (float*)out_cond->data;
            int ne_elements      = (int)ggml_nelements(denoised);
            for (int i = 0; i < ne_elements; i++) {
                float latent_result = positive_data[i];
                if (has_unconditioned) {
                    // out_uncond + cfg_scale * (out_cond - out_uncond)
                    latent_result = negative_data[i] + cfg_scale * (positive_data[i] - negative_data[i]);
                }
                // v = latent_result, eps = latent_result
                // denoised = (v * c_out + input * c_skip) or (input + eps * c_out)
                vec_denoised[i] = latent_result * c_out + vec_input[i] * c_skip;
            }
            int64_t t1 = ggml_time_us();
            if (step > 0) {
                pretty_progress(step, (int)steps, (t1 - t0) / 1000000.f);
                // LOG_INFO("step %d sampling completed taking %.2fs", step, (t1 - t0) * 1.0f / 1000000);
            }
        };

        // sample_euler_ancestral
        switch (method) {
            case EULER_A: {
                struct ggml_tensor* noise = ggml_dup_tensor(work_ctx, x);
                struct ggml_tensor* d     = ggml_dup_tensor(work_ctx, x);

                for (int i = 0; i < steps; i++) {
                    float sigma = sigmas[i];

                    // denoise
                    denoise(x, sigma, i + 1);

                    // d = (x - denoised) / sigma
                    {
                        float* vec_d        = (float*)d->data;
                        float* vec_x        = (float*)x->data;
                        float* vec_denoised = (float*)denoised->data;

                        for (int i = 0; i < ggml_nelements(d); i++) {
                            vec_d[i] = (vec_x[i] - vec_denoised[i]) / sigma;
                        }
                    }

                    // get_ancestral_step
                    float sigma_up   = std::min(sigmas[i + 1],
                                                std::sqrt(sigmas[i + 1] * sigmas[i + 1] * (sigmas[i] * sigmas[i] - sigmas[i + 1] * sigmas[i + 1]) / (sigmas[i] * sigmas[i])));
                    float sigma_down = std::sqrt(sigmas[i + 1] * sigmas[i + 1] - sigma_up * sigma_up);

                    // Euler method
                    float dt = sigma_down - sigmas[i];
                    // x = x + d * dt
                    {
                        float* vec_d = (float*)d->data;
                        float* vec_x = (float*)x->data;

                        for (int i = 0; i < ggml_nelements(x); i++) {
                            vec_x[i] = vec_x[i] + vec_d[i] * dt;
                        }
                    }

                    if (sigmas[i + 1] > 0) {
                        // x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
                        ggml_tensor_set_f32_randn(noise, rng);
                        // noise = load_tensor_from_file(work_ctx, "./rand" + std::to_string(i+1) + ".bin");
                        {
                            float* vec_x     = (float*)x->data;
                            float* vec_noise = (float*)noise->data;

                            for (int i = 0; i < ggml_nelements(x); i++) {
                                vec_x[i] = vec_x[i] + vec_noise[i] * sigma_up;
                            }
                        }
                    }
                }
            } break;
            case EULER:  // Implemented without any sigma churn
            {
                struct ggml_tensor* d = ggml_dup_tensor(work_ctx, x);

                for (int i = 0; i < steps; i++) {
                    float sigma = sigmas[i];

                    // denoise
                    denoise(x, sigma, i + 1);

                    // d = (x - denoised) / sigma
                    {
                        float* vec_d        = (float*)d->data;
                        float* vec_x        = (float*)x->data;
                        float* vec_denoised = (float*)denoised->data;

                        for (int j = 0; j < ggml_nelements(d); j++) {
                            vec_d[j] = (vec_x[j] - vec_denoised[j]) / sigma;
                        }
                    }

                    float dt = sigmas[i + 1] - sigma;
                    // x = x + d * dt
                    {
                        float* vec_d = (float*)d->data;
                        float* vec_x = (float*)x->data;

                        for (int j = 0; j < ggml_nelements(x); j++) {
                            vec_x[j] = vec_x[j] + vec_d[j] * dt;
                        }
                    }
                }
            } break;
            case HEUN: {
                struct ggml_tensor* d  = ggml_dup_tensor(work_ctx, x);
                struct ggml_tensor* x2 = ggml_dup_tensor(work_ctx, x);

                for (int i = 0; i < steps; i++) {
                    // denoise
                    denoise(x, sigmas[i], -(i + 1));

                    // d = (x - denoised) / sigma
                    {
                        float* vec_d        = (float*)d->data;
                        float* vec_x        = (float*)x->data;
                        float* vec_denoised = (float*)denoised->data;

                        for (int j = 0; j < ggml_nelements(x); j++) {
                            vec_d[j] = (vec_x[j] - vec_denoised[j]) / sigmas[i];
                        }
                    }

                    float dt = sigmas[i + 1] - sigmas[i];
                    if (sigmas[i + 1] == 0) {
                        // Euler step
                        // x = x + d * dt
                        float* vec_d = (float*)d->data;
                        float* vec_x = (float*)x->data;

                        for (int j = 0; j < ggml_nelements(x); j++) {
                            vec_x[j] = vec_x[j] + vec_d[j] * dt;
                        }
                    } else {
                        // Heun step
                        float* vec_d  = (float*)d->data;
                        float* vec_d2 = (float*)d->data;
                        float* vec_x  = (float*)x->data;
                        float* vec_x2 = (float*)x2->data;

                        for (int j = 0; j < ggml_nelements(x); j++) {
                            vec_x2[j] = vec_x[j] + vec_d[j] * dt;
                        }

                        denoise(x2, sigmas[i + 1], i + 1);
                        float* vec_denoised = (float*)denoised->data;
                        for (int j = 0; j < ggml_nelements(x); j++) {
                            float d2 = (vec_x2[j] - vec_denoised[j]) / sigmas[i + 1];
                            vec_d[j] = (vec_d[j] + d2) / 2;
                            vec_x[j] = vec_x[j] + vec_d[j] * dt;
                        }
                    }
                }
            } break;
            case DPM2: {
                struct ggml_tensor* d  = ggml_dup_tensor(work_ctx, x);
                struct ggml_tensor* x2 = ggml_dup_tensor(work_ctx, x);

                for (int i = 0; i < steps; i++) {
                    // denoise
                    denoise(x, sigmas[i], i + 1);

                    // d = (x - denoised) / sigma
                    {
                        float* vec_d        = (float*)d->data;
                        float* vec_x        = (float*)x->data;
                        float* vec_denoised = (float*)denoised->data;

                        for (int j = 0; j < ggml_nelements(x); j++) {
                            vec_d[j] = (vec_x[j] - vec_denoised[j]) / sigmas[i];
                        }
                    }

                    if (sigmas[i + 1] == 0) {
                        // Euler step
                        // x = x + d * dt
                        float dt     = sigmas[i + 1] - sigmas[i];
                        float* vec_d = (float*)d->data;
                        float* vec_x = (float*)x->data;

                        for (int j = 0; j < ggml_nelements(x); j++) {
                            vec_x[j] = vec_x[j] + vec_d[j] * dt;
                        }
                    } else {
                        // DPM-Solver-2
                        float sigma_mid = exp(0.5f * (log(sigmas[i]) + log(sigmas[i + 1])));
                        float dt_1      = sigma_mid - sigmas[i];
                        float dt_2      = sigmas[i + 1] - sigmas[i];

                        float* vec_d  = (float*)d->data;
                        float* vec_x  = (float*)x->data;
                        float* vec_x2 = (float*)x2->data;
                        for (int j = 0; j < ggml_nelements(x); j++) {
                            vec_x2[j] = vec_x[j] + vec_d[j] * dt_1;
                        }

                        denoise(x2, sigma_mid, i + 1);
                        float* vec_denoised = (float*)denoised->data;
                        for (int j = 0; j < ggml_nelements(x); j++) {
                            float d2 = (vec_x2[j] - vec_denoised[j]) / sigma_mid;
                            vec_x[j] = vec_x[j] + d2 * dt_2;
                        }
                    }
                }

            } break;
            case DPMPP2S_A: {
                struct ggml_tensor* noise = ggml_dup_tensor(work_ctx, x);
                struct ggml_tensor* d     = ggml_dup_tensor(work_ctx, x);
                struct ggml_tensor* x2    = ggml_dup_tensor(work_ctx, x);

                for (int i = 0; i < steps; i++) {
                    // denoise
                    denoise(x, sigmas[i], i + 1);

                    // get_ancestral_step
                    float sigma_up   = std::min(sigmas[i + 1],
                                                std::sqrt(sigmas[i + 1] * sigmas[i + 1] * (sigmas[i] * sigmas[i] - sigmas[i + 1] * sigmas[i + 1]) / (sigmas[i] * sigmas[i])));
                    float sigma_down = std::sqrt(sigmas[i + 1] * sigmas[i + 1] - sigma_up * sigma_up);
                    auto t_fn        = [](float sigma) -> float { return -log(sigma); };
                    auto sigma_fn    = [](float t) -> float { return exp(-t); };

                    if (sigma_down == 0) {
                        // Euler step
                        float* vec_d        = (float*)d->data;
                        float* vec_x        = (float*)x->data;
                        float* vec_denoised = (float*)denoised->data;

                        for (int j = 0; j < ggml_nelements(d); j++) {
                            vec_d[j] = (vec_x[j] - vec_denoised[j]) / sigmas[i];
                        }

                        // TODO: If sigma_down == 0, isn't this wrong?
                        // But
                        // https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py#L525
                        // has this exactly the same way.
                        float dt = sigma_down - sigmas[i];
                        for (int j = 0; j < ggml_nelements(d); j++) {
                            vec_x[j] = vec_x[j] + vec_d[j] * dt;
                        }
                    } else {
                        // DPM-Solver++(2S)
                        float t      = t_fn(sigmas[i]);
                        float t_next = t_fn(sigma_down);
                        float h      = t_next - t;
                        float s      = t + 0.5f * h;

                        float* vec_d        = (float*)d->data;
                        float* vec_x        = (float*)x->data;
                        float* vec_x2       = (float*)x2->data;
                        float* vec_denoised = (float*)denoised->data;

                        // First half-step
                        for (int j = 0; j < ggml_nelements(x); j++) {
                            vec_x2[j] = (sigma_fn(s) / sigma_fn(t)) * vec_x[j] - (exp(-h * 0.5f) - 1) * vec_denoised[j];
                        }

                        denoise(x2, sigmas[i + 1], i + 1);

                        // Second half-step
                        for (int j = 0; j < ggml_nelements(x); j++) {
                            vec_x[j] = (sigma_fn(t_next) / sigma_fn(t)) * vec_x[j] - (exp(-h) - 1) * vec_denoised[j];
                        }
                    }

                    // Noise addition
                    if (sigmas[i + 1] > 0) {
                        ggml_tensor_set_f32_randn(noise, rng);
                        {
                            float* vec_x     = (float*)x->data;
                            float* vec_noise = (float*)noise->data;

                            for (int i = 0; i < ggml_nelements(x); i++) {
                                vec_x[i] = vec_x[i] + vec_noise[i] * sigma_up;
                            }
                        }
                    }
                }
            } break;
            case DPMPP2M:  // DPM++ (2M) from Karras et al (2022)
            {
                struct ggml_tensor* old_denoised = ggml_dup_tensor(work_ctx, x);

                auto t_fn = [](float sigma) -> float { return -log(sigma); };

                for (int i = 0; i < steps; i++) {
                    // denoise
                    denoise(x, sigmas[i], i + 1);

                    float t                 = t_fn(sigmas[i]);
                    float t_next            = t_fn(sigmas[i + 1]);
                    float h                 = t_next - t;
                    float a                 = sigmas[i + 1] / sigmas[i];
                    float b                 = exp(-h) - 1.f;
                    float* vec_x            = (float*)x->data;
                    float* vec_denoised     = (float*)denoised->data;
                    float* vec_old_denoised = (float*)old_denoised->data;

                    if (i == 0 || sigmas[i + 1] == 0) {
                        // Simpler step for the edge cases
                        for (int j = 0; j < ggml_nelements(x); j++) {
                            vec_x[j] = a * vec_x[j] - b * vec_denoised[j];
                        }
                    } else {
                        float h_last = t - t_fn(sigmas[i - 1]);
                        float r      = h_last / h;
                        for (int j = 0; j < ggml_nelements(x); j++) {
                            float denoised_d = (1.f + 1.f / (2.f * r)) * vec_denoised[j] - (1.f / (2.f * r)) * vec_old_denoised[j];
                            vec_x[j]         = a * vec_x[j] - b * denoised_d;
                        }
                    }

                    // old_denoised = denoised
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_old_denoised[j] = vec_denoised[j];
                    }
                }
            } break;
            case DPMPP2Mv2:  // Modified DPM++ (2M) from https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/8457
            {
                struct ggml_tensor* old_denoised = ggml_dup_tensor(work_ctx, x);

                auto t_fn = [](float sigma) -> float { return -log(sigma); };

                for (int i = 0; i < steps; i++) {
                    // denoise
                    denoise(x, sigmas[i], i + 1);

                    float t                 = t_fn(sigmas[i]);
                    float t_next            = t_fn(sigmas[i + 1]);
                    float h                 = t_next - t;
                    float a                 = sigmas[i + 1] / sigmas[i];
                    float* vec_x            = (float*)x->data;
                    float* vec_denoised     = (float*)denoised->data;
                    float* vec_old_denoised = (float*)old_denoised->data;

                    if (i == 0 || sigmas[i + 1] == 0) {
                        // Simpler step for the edge cases
                        float b = exp(-h) - 1.f;
                        for (int j = 0; j < ggml_nelements(x); j++) {
                            vec_x[j] = a * vec_x[j] - b * vec_denoised[j];
                        }
                    } else {
                        float h_last = t - t_fn(sigmas[i - 1]);
                        float h_min  = std::min(h_last, h);
                        float h_max  = std::max(h_last, h);
                        float r      = h_max / h_min;
                        float h_d    = (h_max + h_min) / 2.f;
                        float b      = exp(-h_d) - 1.f;
                        for (int j = 0; j < ggml_nelements(x); j++) {
                            float denoised_d = (1.f + 1.f / (2.f * r)) * vec_denoised[j] - (1.f / (2.f * r)) * vec_old_denoised[j];
                            vec_x[j]         = a * vec_x[j] - b * denoised_d;
                        }
                    }

                    // old_denoised = denoised
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_old_denoised[j] = vec_denoised[j];
                    }
                }
            } break;
            case LCM:  // Latent Consistency Models
            {
                struct ggml_tensor* noise = ggml_dup_tensor(work_ctx, x);
                struct ggml_tensor* d     = ggml_dup_tensor(work_ctx, x);

                for (int i = 0; i < steps; i++) {
                    float sigma = sigmas[i];

                    // denoise
                    denoise(x, sigma, i + 1);

                    // x = denoised
                    {
                        float* vec_x        = (float*)x->data;
                        float* vec_denoised = (float*)denoised->data;
                        for (int j = 0; j < ggml_nelements(x); j++) {
                            vec_x[j] = vec_denoised[j];
                        }
                    }

                    if (sigmas[i + 1] > 0) {
                        // x += sigmas[i + 1] * noise_sampler(sigmas[i], sigmas[i + 1])
                        ggml_tensor_set_f32_randn(noise, rng);
                        // noise = load_tensor_from_file(res_ctx, "./rand" + std::to_string(i+1) + ".bin");
                        {
                            float* vec_x     = (float*)x->data;
                            float* vec_noise = (float*)noise->data;

                            for (int j = 0; j < ggml_nelements(x); j++) {
                                vec_x[j] = vec_x[j] + sigmas[i + 1] * vec_noise[j];
                            }
                        }
                    }
                }
            } break;

            default:
                LOG_ERROR("Attempting to sample with nonexisting sample method %i", method);
                abort();
        }
        control_net.free_compute_buffer();
        diffusion_model.free_compute_buffer();
        return x;
    }

    // ldm.models.diffusion.ddpm.LatentDiffusion.get_first_stage_encoding
    ggml_tensor* get_first_stage_encoding(ggml_context* work_ctx, ggml_tensor* moments) {
        // ldm.modules.distributions.distributions.DiagonalGaussianDistribution.sample
        ggml_tensor* latent       = ggml_new_tensor_4d(work_ctx, moments->type, moments->ne[0], moments->ne[1], moments->ne[2] / 2, moments->ne[3]);
        struct ggml_tensor* noise = ggml_dup_tensor(work_ctx, latent);
        ggml_tensor_set_f32_randn(noise, rng);
        // noise = load_tensor_from_file(work_ctx, "noise.bin");
        {
            float mean   = 0;
            float logvar = 0;
            float value  = 0;
            float std_   = 0;
            for (int i = 0; i < latent->ne[3]; i++) {
                for (int j = 0; j < latent->ne[2]; j++) {
                    for (int k = 0; k < latent->ne[1]; k++) {
                        for (int l = 0; l < latent->ne[0]; l++) {
                            mean   = ggml_tensor_get_f32(moments, l, k, j, i);
                            logvar = ggml_tensor_get_f32(moments, l, k, j + (int)latent->ne[2], i);
                            logvar = std::max(-30.0f, std::min(logvar, 20.0f));
                            std_   = std::exp(0.5f * logvar);
                            value  = mean + std_ * ggml_tensor_get_f32(noise, l, k, j, i);
                            value  = value * scale_factor;
                            // printf("%d %d %d %d -> %f\n", i, j, k, l, value);
                            ggml_tensor_set_f32(latent, value, l, k, j, i);
                        }
                    }
                }
            }
        }
        return latent;
    }

    ggml_tensor* compute_first_stage(ggml_context* work_ctx, ggml_tensor* x, bool decode) {
        int64_t W           = x->ne[0];
        int64_t H           = x->ne[1];
        ggml_tensor* result = ggml_new_tensor_3d(work_ctx, GGML_TYPE_F32,
                                                 decode ? (W * 8) : (W / 8),                    // width
                                                 decode ? (H * 8) : (H / 8),                    // height
                                                 decode ? 3 : (use_tiny_autoencoder ? 4 : 8));  // channels
        int64_t t0          = ggml_time_ms();
        if (!use_tiny_autoencoder) {
            if (decode) {
                ggml_tensor_scale(x, 1.0f / scale_factor);
            } else {
                ggml_tensor_scale_input(x);
            }
            if (vae_tiling && decode) {  // TODO: support tiling vae encode
                // split latent in 32x32 tiles and compute in several steps
                auto on_tiling = [&](ggml_tensor* in, ggml_tensor* out, bool init) {
                    if (init) {
                        first_stage_model.alloc_compute_buffer(in, decode);
                    } else {
                        first_stage_model.compute(out, n_threads, in, decode);
                    }
                };
                sd_tiling(x, result, 8, 32, 0.5f, on_tiling);
            } else {
                first_stage_model.alloc_compute_buffer(x, decode);
                first_stage_model.compute(result, n_threads, x, decode);
            }
            first_stage_model.free_compute_buffer();
            if (decode) {
                ggml_tensor_scale_output(result);
            }
        } else {
            if (vae_tiling && decode) {  // TODO: support tiling vae encode
                // split latent in 64x64 tiles and compute in several steps
                auto on_tiling = [&](ggml_tensor* in, ggml_tensor* out, bool init) {
                    if (init) {
                        tae_first_stage.alloc_compute_buffer(in, decode);
                    } else {
                        tae_first_stage.compute(out, n_threads, in, decode);
                    }
                };
                sd_tiling(x, result, 8, 64, 0.5f, on_tiling);
            } else {
                tae_first_stage.alloc_compute_buffer(x, decode);
                tae_first_stage.compute(result, n_threads, x, decode);
            }
            tae_first_stage.free_compute_buffer();
        }
        int64_t t1 = ggml_time_ms();
        LOG_DEBUG("computing vae [mode: %s] graph completed, taking %.2fs", decode ? "DECODE" : "ENCODE", (t1 - t0) * 1.0f / 1000);
        if (decode) {
            ggml_tensor_clamp(result, 0.0f, 1.0f);
        }
        return result;
    }

    ggml_tensor* encode_first_stage(ggml_context* work_ctx, ggml_tensor* x) {
        return compute_first_stage(work_ctx, x, false);
    }

    ggml_tensor* decode_first_stage(ggml_context* work_ctx, ggml_tensor* x) {
        return compute_first_stage(work_ctx, x, true);
    }
};

/*================================================= SD API ==================================================*/

struct sd_ctx_t {
    StableDiffusionGGML* sd = NULL;
};

sd_ctx_t* new_sd_ctx(const char* model_path_c_str,
                     const char* vae_path_c_str,
                     const char* taesd_path_c_str,
                     const char* control_net_path_c_str,
                     const char* lora_model_dir_c_str,
                     const char* embed_dir_c_str,
                     bool vae_decode_only,
                     bool vae_tiling,
                     bool free_params_immediately,
                     int n_threads,
                     enum sd_type_t wtype,
                     enum rng_type_t rng_type,
                     enum schedule_t s,
                     bool keep_control_net_cpu) {
    sd_ctx_t* sd_ctx = (sd_ctx_t*)malloc(sizeof(sd_ctx_t));
    if (sd_ctx == NULL) {
        return NULL;
    }
    std::string model_path(model_path_c_str);
    std::string vae_path(vae_path_c_str);
    std::string taesd_path(taesd_path_c_str);
    std::string control_net_path(control_net_path_c_str);
    std::string embd_path(embed_dir_c_str);
    std::string lora_model_dir(lora_model_dir_c_str);

    sd_ctx->sd = new StableDiffusionGGML(n_threads,
                                         vae_decode_only,
                                         free_params_immediately,
                                         lora_model_dir,
                                         rng_type);
    if (sd_ctx->sd == NULL) {
        return NULL;
    }

    if (!sd_ctx->sd->load_from_file(model_path,
                                    vae_path,
                                    control_net_path,
                                    embd_path,
                                    taesd_path,
                                    vae_tiling,
                                    (ggml_type)wtype,
                                    s,
                                    keep_control_net_cpu)) {
        delete sd_ctx->sd;
        sd_ctx->sd = NULL;
        free(sd_ctx);
        return NULL;
    }
    return sd_ctx;
}

void free_sd_ctx(sd_ctx_t* sd_ctx) {
    if (sd_ctx->sd != NULL) {
        delete sd_ctx->sd;
        sd_ctx->sd = NULL;
    }
    free(sd_ctx);
}

sd_image_t* txt2img(sd_ctx_t* sd_ctx,
                    const char* prompt_c_str,
                    const char* negative_prompt_c_str,
                    int clip_skip,
                    float cfg_scale,
                    int width,
                    int height,
                    enum sample_method_t sample_method,
                    int sample_steps,
                    int64_t seed,
                    int batch_count,
                    const sd_image_t* control_cond,
                    float control_strength) {
    LOG_DEBUG("txt2img %dx%d", width, height);
    if (sd_ctx == NULL) {
        return NULL;
    }
    // LOG_DEBUG("%s %s %f %d %d %d", prompt_c_str, negative_prompt_c_str, cfg_scale, sample_steps, seed, batch_count);
    std::string prompt(prompt_c_str);
    std::string negative_prompt(negative_prompt_c_str);

    // extract and remove lora
    auto result_pair                                = extract_and_remove_lora(prompt);
    std::unordered_map<std::string, float> lora_f2m = result_pair.first;  // lora_name -> multiplier

    for (auto& kv : lora_f2m) {
        LOG_DEBUG("lora %s:%.2f", kv.first.c_str(), kv.second);
    }

    prompt = result_pair.second;
    LOG_DEBUG("prompt after extract and remove lora: \"%s\"", prompt.c_str());

    int64_t t0 = ggml_time_ms();
    sd_ctx->sd->apply_loras(lora_f2m);
    int64_t t1 = ggml_time_ms();
    LOG_INFO("apply_loras completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);
    struct ggml_init_params params;
    params.mem_size = static_cast<size_t>(10 * 1024 * 1024);  // 10 MB
    params.mem_size += width * height * 3 * sizeof(float);
    params.mem_size *= batch_count;
    params.mem_buffer = NULL;
    params.no_alloc   = false;
    // LOG_DEBUG("mem_size %u ", params.mem_size);

    struct ggml_context* work_ctx = ggml_init(params);
    if (!work_ctx) {
        LOG_ERROR("ggml_init() failed");
        return NULL;
    }

    if (seed < 0) {
        // Generally, when using the provided command line, the seed is always >0.
        // However, to prevent potential issues if 'stable-diffusion.cpp' is invoked as a library
        // by a third party with a seed <0, let's incorporate randomization here.
        srand((int)time(NULL));
        seed = rand();
    }

    t0                            = ggml_time_ms();
    auto cond_pair                = sd_ctx->sd->get_learned_condition(work_ctx, prompt, clip_skip, width, height);
    ggml_tensor* c                = cond_pair.first;
    ggml_tensor* c_vector         = cond_pair.second;  // [adm_in_channels, ]
    struct ggml_tensor* uc        = NULL;
    struct ggml_tensor* uc_vector = NULL;
    if (cfg_scale != 1.0) {
        bool force_zero_embeddings = false;
        if (sd_ctx->sd->version == VERSION_XL && negative_prompt.size() == 0) {
            force_zero_embeddings = true;
        }
        auto uncond_pair = sd_ctx->sd->get_learned_condition(work_ctx, negative_prompt, clip_skip, width, height, force_zero_embeddings);
        uc               = uncond_pair.first;
        uc_vector        = uncond_pair.second;  // [adm_in_channels, ]
    }
    t1 = ggml_time_ms();
    LOG_INFO("get_learned_condition completed, taking %" PRId64 " ms", t1 - t0);

    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->cond_stage_model.free_params_buffer();
    }

    struct ggml_tensor* image_hint = NULL;
    if (control_cond != NULL) {
        image_hint = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, width, height, 3, 1);
        sd_image_to_tensor(control_cond->data, image_hint);
    }

    std::vector<struct ggml_tensor*> final_latents;  // collect latents to decode
    int C = 4;
    int W = width / 8;
    int H = height / 8;
    LOG_INFO("sampling using %s method", sampling_methods_str[sample_method]);
    for (int b = 0; b < batch_count; b++) {
        int64_t sampling_start = ggml_time_ms();
        int64_t cur_seed       = seed + b;
        LOG_INFO("generating image: %i/%i - seed %i", b + 1, batch_count, cur_seed);

        sd_ctx->sd->rng->manual_seed(cur_seed);
        struct ggml_tensor* x_t = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, W, H, C, 1);
        ggml_tensor_set_f32_randn(x_t, sd_ctx->sd->rng);

        std::vector<float> sigmas = sd_ctx->sd->denoiser->schedule->get_sigmas(sample_steps);

        struct ggml_tensor* x_0 = sd_ctx->sd->sample(work_ctx, x_t, NULL, c, c_vector, uc, uc_vector, image_hint, cfg_scale, sample_method, sigmas, control_strength);
        // struct ggml_tensor* x_0 = load_tensor_from_file(ctx, "samples_ddim.bin");
        // print_ggml_tensor(x_0);
        int64_t sampling_end = ggml_time_ms();
        LOG_INFO("sampling completed, taking %.2fs", (sampling_end - sampling_start) * 1.0f / 1000);
        final_latents.push_back(x_0);
    }

    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->diffusion_model.free_params_buffer();
    }
    int64_t t3 = ggml_time_ms();
    LOG_INFO("generating %" PRId64 " latent images completed, taking %.2fs", final_latents.size(), (t3 - t1) * 1.0f / 1000);

    LOG_INFO("decoding %zu latents", final_latents.size());
    std::vector<struct ggml_tensor*> decoded_images;  // collect decoded images
    for (size_t i = 0; i < final_latents.size(); i++) {
        t1                      = ggml_time_ms();
        struct ggml_tensor* img = sd_ctx->sd->decode_first_stage(work_ctx, final_latents[i] /* x_0 */);
        // print_ggml_tensor(img);
        if (img != NULL) {
            decoded_images.push_back(img);
        }
        int64_t t2 = ggml_time_ms();
        LOG_INFO("latent %" PRId64 " decoded, taking %.2fs", i + 1, (t2 - t1) * 1.0f / 1000);
    }

    int64_t t4 = ggml_time_ms();
    LOG_INFO("decode_first_stage completed, taking %.2fs", (t4 - t3) * 1.0f / 1000);
    if (sd_ctx->sd->free_params_immediately && !sd_ctx->sd->use_tiny_autoencoder) {
        sd_ctx->sd->first_stage_model.free_params_buffer();
    }
    sd_image_t* result_images = (sd_image_t*)calloc(batch_count, sizeof(sd_image_t));
    if (result_images == NULL) {
        ggml_free(work_ctx);
        return NULL;
    }

    for (size_t i = 0; i < decoded_images.size(); i++) {
        result_images[i].width   = width;
        result_images[i].height  = height;
        result_images[i].channel = 3;
        result_images[i].data    = sd_tensor_to_image(decoded_images[i]);
    }
    ggml_free(work_ctx);
    LOG_INFO(
        "txt2img completed in %.2fs",
        (t4 - t0) * 1.0f / 1000);

    return result_images;
}

sd_image_t* img2img(sd_ctx_t* sd_ctx,
                    sd_image_t init_image,
                    const char* prompt_c_str,
                    const char* negative_prompt_c_str,
                    int clip_skip,
                    float cfg_scale,
                    int width,
                    int height,
                    sample_method_t sample_method,
                    int sample_steps,
                    float strength,
                    int64_t seed,
                    int batch_count) {
    if (sd_ctx == NULL) {
        return NULL;
    }
    std::string prompt(prompt_c_str);
    std::string negative_prompt(negative_prompt_c_str);

    LOG_INFO("img2img %dx%d", width, height);

    std::vector<float> sigmas = sd_ctx->sd->denoiser->schedule->get_sigmas(sample_steps);
    size_t t_enc              = static_cast<size_t>(sample_steps * strength);
    LOG_INFO("target t_enc is %zu steps", t_enc);
    std::vector<float> sigma_sched;
    sigma_sched.assign(sigmas.begin() + sample_steps - t_enc - 1, sigmas.end());

    struct ggml_init_params params;
    params.mem_size = static_cast<size_t>(10 * 1024) * 1024;  // 10 MB
    params.mem_size += width * height * 3 * sizeof(float) * 2;
    params.mem_buffer = NULL;
    params.no_alloc   = false;
    // LOG_DEBUG("mem_size %u ", params.mem_size);

    // draft context
    struct ggml_context* work_ctx = ggml_init(params);
    if (!work_ctx) {
        LOG_ERROR("ggml_init() failed");
        return NULL;
    }

    if (seed < 0) {
        seed = (int)time(NULL);
    }

    sd_ctx->sd->rng->manual_seed(seed);

    // extract and remove lora
    auto result_pair                                = extract_and_remove_lora(prompt);
    std::unordered_map<std::string, float> lora_f2m = result_pair.first;  // lora_name -> multiplier
    for (auto& kv : lora_f2m) {
        LOG_DEBUG("lora %s:%.2f", kv.first.c_str(), kv.second);
    }
    prompt = result_pair.second;
    LOG_DEBUG("prompt after extract and remove lora: \"%s\"", prompt.c_str());

    // load lora from file
    int64_t t0 = ggml_time_ms();
    sd_ctx->sd->apply_loras(lora_f2m);
    int64_t t1 = ggml_time_ms();
    LOG_INFO("apply_loras completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);

    ggml_tensor* init_img = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, width, height, 3, 1);
    sd_image_to_tensor(init_image.data, init_img);
    t0                       = ggml_time_ms();
    ggml_tensor* init_latent = NULL;
    if (!sd_ctx->sd->use_tiny_autoencoder) {
        ggml_tensor* moments = sd_ctx->sd->encode_first_stage(work_ctx, init_img);
        init_latent          = sd_ctx->sd->get_first_stage_encoding(work_ctx, moments);
    } else {
        init_latent = sd_ctx->sd->encode_first_stage(work_ctx, init_img);
    }
    // print_ggml_tensor(init_latent);
    t1 = ggml_time_ms();
    LOG_INFO("encode_first_stage completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);

    auto cond_pair                = sd_ctx->sd->get_learned_condition(work_ctx, prompt, clip_skip, width, height);
    ggml_tensor* c                = cond_pair.first;
    ggml_tensor* c_vector         = cond_pair.second;  // [adm_in_channels, ]
    struct ggml_tensor* uc        = NULL;
    struct ggml_tensor* uc_vector = NULL;
    if (cfg_scale != 1.0) {
        bool force_zero_embeddings = false;
        if (sd_ctx->sd->version == VERSION_XL && negative_prompt.size() == 0) {
            force_zero_embeddings = true;
        }
        auto uncond_pair = sd_ctx->sd->get_learned_condition(work_ctx, negative_prompt, clip_skip, width, height, force_zero_embeddings);
        uc               = uncond_pair.first;
        uc_vector        = uncond_pair.second;  // [adm_in_channels, ]
    }
    int64_t t2 = ggml_time_ms();
    LOG_INFO("get_learned_condition completed, taking %" PRId64 " ms", t2 - t1);
    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->cond_stage_model.free_params_buffer();
    }

    sd_ctx->sd->rng->manual_seed(seed);
    struct ggml_tensor* noise = ggml_dup_tensor(work_ctx, init_latent);
    ggml_tensor_set_f32_randn(noise, sd_ctx->sd->rng);

    LOG_INFO("sampling using %s method", sampling_methods_str[sample_method]);
    struct ggml_tensor* x_0 = sd_ctx->sd->sample(work_ctx, init_latent, noise, c, c_vector, uc,
                                                 uc_vector, NULL, cfg_scale, sample_method, sigma_sched, 1.0f);
    // struct ggml_tensor *x_0 = load_tensor_from_file(ctx, "samples_ddim.bin");
    // print_ggml_tensor(x_0);
    int64_t t3 = ggml_time_ms();
    LOG_INFO("sampling completed, taking %.2fs", (t3 - t2) * 1.0f / 1000);
    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->diffusion_model.free_params_buffer();
    }

    struct ggml_tensor* img = sd_ctx->sd->decode_first_stage(work_ctx, x_0);
    if (sd_ctx->sd->free_params_immediately && !sd_ctx->sd->use_tiny_autoencoder) {
        sd_ctx->sd->first_stage_model.free_params_buffer();
    }
    if (img == NULL) {
        ggml_free(work_ctx);
        return NULL;
    }

    sd_image_t* result_images = (sd_image_t*)calloc(1, sizeof(sd_image_t));
    if (result_images == NULL) {
        ggml_free(work_ctx);
        return NULL;
    }

    for (size_t i = 0; i < 1; i++) {
        result_images[i].width   = width;
        result_images[i].height  = height;
        result_images[i].channel = 3;
        result_images[i].data    = sd_tensor_to_image(img);
    }
    ggml_free(work_ctx);

    int64_t t4 = ggml_time_ms();
    LOG_INFO("decode_first_stage completed, taking %.2fs", (t4 - t3) * 1.0f / 1000);

    LOG_INFO("img2img completed in %.2fs", (t4 - t0) * 1.0f / 1000);

    return result_images;
}
