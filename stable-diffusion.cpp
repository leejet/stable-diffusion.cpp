#include "ggml_extend.hpp"

#include "model.h"
#include "rng.hpp"
#include "rng_philox.hpp"
#include "stable-diffusion.h"
#include "util.h"

#include "conditioner.hpp"
#include "control.hpp"
#include "denoiser.hpp"
#include "diffusion_model.hpp"
#include "esrgan.hpp"
#include "lora.hpp"
#include "pmid.hpp"
#include "tae.hpp"
#include "vae.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#include "stb_image.h"

// #define STB_IMAGE_WRITE_IMPLEMENTATION
// #define STB_IMAGE_WRITE_STATIC
// #include "stb_image_write.h"

const char* model_version_to_str[] = {
    "1.x",
    "2.x",
    "XL",
    "SVD",
    "3 2B"};

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
    ggml_backend_t backend             = NULL;  // general backend
    ggml_backend_t clip_backend        = NULL;
    ggml_backend_t control_net_backend = NULL;
    ggml_backend_t vae_backend         = NULL;
    ggml_type model_data_type          = GGML_TYPE_COUNT;

    SDVersion version;
    bool vae_decode_only         = false;
    bool free_params_immediately = false;

    std::shared_ptr<RNG> rng = std::make_shared<STDDefaultRNG>();
    int n_threads            = -1;
    float scale_factor       = 0.18215f;

    std::shared_ptr<Conditioner> cond_stage_model;
    std::shared_ptr<FrozenCLIPVisionEmbedder> clip_vision;  // for svd
    std::shared_ptr<DiffusionModel> diffusion_model;
    std::shared_ptr<AutoEncoderKL> first_stage_model;
    std::shared_ptr<TinyAutoEncoder> tae_first_stage;
    std::shared_ptr<ControlNet> control_net;
    std::shared_ptr<PhotoMakerIDEncoder> pmid_model;
    std::shared_ptr<LoraModel> pmid_lora;

    std::string taesd_path;
    bool use_tiny_autoencoder = false;
    bool vae_tiling           = false;
    bool stacked_id           = false;

    std::map<std::string, struct ggml_tensor*> tensors;

    std::string lora_model_dir;
    // lora_name => multiplier
    std::unordered_map<std::string, float> curr_lora_state;

    std::shared_ptr<Denoiser> denoiser = std::make_shared<CompVisDenoiser>();

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
        if (rng_type == STD_DEFAULT_RNG) {
            rng = std::make_shared<STDDefaultRNG>();
        } else if (rng_type == CUDA_RNG) {
            rng = std::make_shared<PhiloxRNG>();
        }
    }

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
        ggml_backend_free(backend);
    }

    bool load_from_file(const std::string& model_path,
                        const std::string& vae_path,
                        const std::string control_net_path,
                        const std::string embeddings_path,
                        const std::string id_embeddings_path,
                        const std::string& taesd_path,
                        bool vae_tiling_,
                        ggml_type wtype,
                        schedule_t schedule,
                        bool clip_on_cpu,
                        bool control_net_cpu,
                        bool vae_on_cpu) {
        use_tiny_autoencoder = taesd_path.size() > 0;
#ifdef SD_USE_CUBLAS
        LOG_DEBUG("Using CUDA backend");
        backend = ggml_backend_cuda_init(0);
#endif
#ifdef SD_USE_METAL
        LOG_DEBUG("Using Metal backend");
        ggml_backend_metal_log_set_callback(ggml_log_callback_default, nullptr);
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

        LOG_INFO("Stable Diffusion %s ", model_version_to_str[version]);
        if (wtype == GGML_TYPE_COUNT) {
            model_data_type = model_loader.get_sd_wtype();
        } else {
            model_data_type = wtype;
        }
        LOG_INFO("Stable Diffusion weight type: %s", ggml_type_name(model_data_type));
        LOG_DEBUG("ggml tensor size = %d bytes", (int)sizeof(ggml_tensor));

        if (version == VERSION_XL) {
            scale_factor = 0.13025f;
            if (vae_path.size() == 0 && taesd_path.size() == 0) {
                LOG_WARN(
                    "!!!It looks like you are using SDXL model. "
                    "If you find that the generated images are completely black, "
                    "try specifying SDXL VAE FP16 Fix with the --vae parameter. "
                    "You can find it here: https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/blob/main/sdxl_vae.safetensors");
            }
        } else if (version == VERSION_3_2B) {
            scale_factor = 1.5305f;
        }

        if (version == VERSION_SVD) {
            clip_vision = std::make_shared<FrozenCLIPVisionEmbedder>(backend, model_data_type);
            clip_vision->alloc_params_buffer();
            clip_vision->get_param_tensors(tensors);

            diffusion_model = std::make_shared<UNetModel>(backend, model_data_type, version);
            diffusion_model->alloc_params_buffer();
            diffusion_model->get_param_tensors(tensors);

            first_stage_model = std::make_shared<AutoEncoderKL>(backend, model_data_type, vae_decode_only, true, version);
            LOG_DEBUG("vae_decode_only %d", vae_decode_only);
            first_stage_model->alloc_params_buffer();
            first_stage_model->get_param_tensors(tensors, "first_stage_model");
        } else {
            clip_backend = backend;
            if (!ggml_backend_is_cpu(backend) && version == VERSION_3_2B && model_data_type != GGML_TYPE_F32) {
                clip_on_cpu = true;
                LOG_INFO("set clip_on_cpu to true");
            }
            if (clip_on_cpu && !ggml_backend_is_cpu(backend)) {
                LOG_INFO("CLIP: Using CPU backend");
                clip_backend = ggml_backend_cpu_init();
            }
            if (version == VERSION_3_2B) {
                cond_stage_model = std::make_shared<SD3CLIPEmbedder>(clip_backend, model_data_type);
                diffusion_model  = std::make_shared<MMDiTModel>(backend, model_data_type, version);
            } else {
                cond_stage_model = std::make_shared<FrozenCLIPEmbedderWithCustomWords>(clip_backend, model_data_type, embeddings_path, version);
                diffusion_model  = std::make_shared<UNetModel>(backend, model_data_type, version);
            }
            cond_stage_model->alloc_params_buffer();
            cond_stage_model->get_param_tensors(tensors);

            diffusion_model->alloc_params_buffer();
            diffusion_model->get_param_tensors(tensors);

            ggml_type vae_type = model_data_type;
            if (version == VERSION_XL) {
                vae_type = GGML_TYPE_F32;  // avoid nan, not work...
            }

            if (!use_tiny_autoencoder) {
                if (vae_on_cpu && !ggml_backend_is_cpu(backend)) {
                    LOG_INFO("VAE Autoencoder: Using CPU backend");
                    vae_backend = ggml_backend_cpu_init();
                } else {
                    vae_backend = backend;
                }
                first_stage_model = std::make_shared<AutoEncoderKL>(vae_backend, vae_type, vae_decode_only, false, version);
                first_stage_model->alloc_params_buffer();
                first_stage_model->get_param_tensors(tensors, "first_stage_model");
            } else {
                tae_first_stage = std::make_shared<TinyAutoEncoder>(backend, model_data_type, vae_decode_only);
            }
            // first_stage_model->get_param_tensors(tensors, "first_stage_model.");

            if (control_net_path.size() > 0) {
                ggml_backend_t controlnet_backend = NULL;
                if (control_net_cpu && !ggml_backend_is_cpu(backend)) {
                    LOG_DEBUG("ControlNet: Using CPU backend");
                    controlnet_backend = ggml_backend_cpu_init();
                } else {
                    controlnet_backend = backend;
                }
                control_net = std::make_shared<ControlNet>(controlnet_backend, model_data_type, version);
            }

            pmid_model = std::make_shared<PhotoMakerIDEncoder>(clip_backend, model_data_type, version);
            if (id_embeddings_path.size() > 0) {
                pmid_lora = std::make_shared<LoraModel>(backend, model_data_type, id_embeddings_path, "");
                if (!pmid_lora->load_from_file(true)) {
                    LOG_WARN("load photomaker lora tensors from %s failed", id_embeddings_path.c_str());
                    return false;
                }
                LOG_INFO("loading stacked ID embedding (PHOTOMAKER) model file from '%s'", id_embeddings_path.c_str());
                if (!model_loader.init_from_file(id_embeddings_path, "pmid.")) {
                    LOG_WARN("loading stacked ID embedding from '%s' failed", id_embeddings_path.c_str());
                } else {
                    stacked_id = true;
                }
            }
            if (stacked_id) {
                if (!pmid_model->alloc_params_buffer()) {
                    LOG_ERROR(" pmid model params buffer allocation failed");
                    return false;
                }
                // LOG_INFO("pmid param memory buffer size = %.2fMB ",
                //     pmid_model->params_buffer_size / 1024.0 / 1024.0);
                pmid_model->get_param_tensors(tensors, "pmid");
            }
            // if(stacked_id){
            //    pmid_model.init_params(GGML_TYPE_F32);
            //    pmid_model.map_by_name(tensors, "pmid.");
            // }
        }

        struct ggml_init_params params;
        params.mem_size   = static_cast<size_t>(10 * 1024) * 1024;  // 10M
        params.mem_buffer = NULL;
        params.no_alloc   = false;
        // LOG_DEBUG("mem_size %u ", params.mem_size);
        struct ggml_context* ctx = ggml_init(params);  // for  alphas_cumprod and is_using_v_parameterization check
        GGML_ASSERT(ctx != NULL);
        ggml_tensor* alphas_cumprod_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, TIMESTEPS);
        calculate_alphas_cumprod((float*)alphas_cumprod_tensor->data);

        // load weights
        LOG_DEBUG("loading weights");

        int64_t t0 = ggml_time_ms();

        std::set<std::string> ignore_tensors;
        tensors["alphas_cumprod"] = alphas_cumprod_tensor;
        if (use_tiny_autoencoder) {
            ignore_tensors.insert("first_stage_model.");
        }
        if (stacked_id) {
            ignore_tensors.insert("lora.");
        }

        if (vae_decode_only) {
            ignore_tensors.insert("first_stage_model.encoder");
            ignore_tensors.insert("first_stage_model.quant");
        }
        if (version == VERSION_SVD) {
            ignore_tensors.insert("conditioner.embedders.3");
        }
        bool success = model_loader.load_tensors(tensors, backend, ignore_tensors);
        if (!success) {
            LOG_ERROR("load tensors from model loader failed");
            ggml_free(ctx);
            return false;
        }

        // LOG_DEBUG("model size = %.2fMB", total_size / 1024.0 / 1024.0);

        if (version == VERSION_SVD) {
            // diffusion_model->test();
            // first_stage_model->test();
            // return false;
        } else {
            size_t clip_params_mem_size = cond_stage_model->get_params_buffer_size();
            size_t unet_params_mem_size = diffusion_model->get_params_buffer_size();
            size_t vae_params_mem_size  = 0;
            if (!use_tiny_autoencoder) {
                vae_params_mem_size = first_stage_model->get_params_buffer_size();
            } else {
                if (!tae_first_stage->load_from_file(taesd_path)) {
                    return false;
                }
                vae_params_mem_size = tae_first_stage->get_params_buffer_size();
            }
            size_t control_net_params_mem_size = 0;
            if (control_net) {
                if (!control_net->load_from_file(control_net_path)) {
                    return false;
                }
                control_net_params_mem_size = control_net->get_params_buffer_size();
            }
            size_t pmid_params_mem_size = 0;
            if (stacked_id) {
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
                "clip %.2fMB(%s), unet %.2fMB(%s), vae %.2fMB(%s), controlnet %.2fMB(%s), pmid %.2fMB(%s)",
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

        int64_t t1 = ggml_time_ms();
        LOG_INFO("loading model from '%s' completed, taking %.2fs", model_path.c_str(), (t1 - t0) * 1.0f / 1000);

        // check is_using_v_parameterization_for_sd2
        bool is_using_v_parameterization = false;
        if (version == VERSION_2_x) {
            if (is_using_v_parameterization_for_sd2(ctx)) {
                is_using_v_parameterization = true;
            }
        } else if (version == VERSION_SVD) {
            // TODO: V_PREDICTION_EDM
            is_using_v_parameterization = true;
        }

        if (version == VERSION_3_2B) {
            LOG_INFO("running in FLOW mode");
            denoiser = std::make_shared<DiscreteFlowDenoiser>();
        } else if (is_using_v_parameterization) {
            LOG_INFO("running in v-prediction mode");
            denoiser = std::make_shared<CompVisVDenoiser>();
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
                case AYS:
                    LOG_INFO("Running with Align-Your-Steps schedule");
                    denoiser->schedule          = std::make_shared<AYSSchedule>();
                    denoiser->schedule->version = version;
                    break;
                case DEFAULT:
                    // Don't touch anything.
                    break;
                default:
                    LOG_ERROR("Unknown schedule %i", schedule);
                    abort();
            }
        }

        auto comp_vis_denoiser = std::dynamic_pointer_cast<CompVisDenoiser>(denoiser);
        if (comp_vis_denoiser) {
            for (int i = 0; i < TIMESTEPS; i++) {
                comp_vis_denoiser->sigmas[i]     = std::sqrt((1 - ((float*)alphas_cumprod_tensor->data)[i]) / ((float*)alphas_cumprod_tensor->data)[i]);
                comp_vis_denoiser->log_sigmas[i] = std::log(comp_vis_denoiser->sigmas[i]);
            }
        }

        LOG_DEBUG("finished loaded file");
        ggml_free(ctx);
        return true;
    }

    bool is_using_v_parameterization_for_sd2(ggml_context* work_ctx) {
        struct ggml_tensor* x_t = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, 8, 8, 4, 1);
        ggml_set_f32(x_t, 0.5);
        struct ggml_tensor* c = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, 1024, 2, 1, 1);
        ggml_set_f32(c, 0.5);

        struct ggml_tensor* timesteps = ggml_new_tensor_1d(work_ctx, GGML_TYPE_F32, 1);
        ggml_set_f32(timesteps, 999);
        int64_t t0              = ggml_time_ms();
        struct ggml_tensor* out = ggml_dup_tensor(work_ctx, x_t);
        diffusion_model->compute(n_threads, x_t, timesteps, c, NULL, NULL, -1, {}, 0.f, &out);
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

    /**
     * 具体应用lora模型
     * @param lora_name lora模型名字,无后缀
     * @param multiplier 权重
     */
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
        // 加载lora模型
        // backend是类的成员变量, 用来选择采用哪种方式(cpu, gpu, ...)
        // model_data_type 定义了模型量化方式 fp16, int4这些
        // 注意这里lora对象是局部变量, 也就是会析构, 可能最终会把加载的内存数据统一交给backend管理.
        LoraModel lora(backend, model_data_type, file_path);
        if (!lora.load_from_file()) {
            LOG_WARN("load lora tensors from %s failed", file_path.c_str());
            return;
        }

        lora.multiplier = multiplier;
        // tensors是一个Map<String, Tensor *> 对象, 用来管理张量的.
        lora.apply(tensors, n_threads);
        lora.free_params_buffer();

        int64_t t1 = ggml_time_ms();

        LOG_INFO("lora '%s' applied, taking %.2fs", lora_name.c_str(), (t1 - t0) * 1.0f / 1000);
    }

    /**
     * 应用所有prompt里面找到的lora
     * @param lora_state 每一个key,value对都是 (lora名字, 权重) 对
     * 内部逻辑会从用户设定的lora权重中减掉已经应用的权重.
     */
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

        LOG_INFO("Attempting to apply %lu LoRAs", lora_state.size());

        for (auto& kv : lora_state_diff) {
            apply_lora(kv.first, kv.second);
        }

        curr_lora_state = lora_state;
    }

    ggml_tensor* id_encoder(ggml_context* work_ctx,
                            ggml_tensor* init_img,
                            ggml_tensor* prompts_embeds,
                            std::vector<bool>& class_tokens_mask) {
        ggml_tensor* res = NULL;
        pmid_model->compute(n_threads, init_img, prompts_embeds, class_tokens_mask, &res, work_ctx);

        return res;
    }

    SDCondition get_svd_condition(ggml_context* work_ctx,
                                  sd_image_t init_image,
                                  int width,
                                  int height,
                                  int fps                    = 6,
                                  int motion_bucket_id       = 127,
                                  float augmentation_level   = 0.f,
                                  bool force_zero_embeddings = false) {
        // c_crossattn
        int64_t t0                      = ggml_time_ms();
        struct ggml_tensor* c_crossattn = NULL;
        {
            if (force_zero_embeddings) {
                c_crossattn = ggml_new_tensor_1d(work_ctx, GGML_TYPE_F32, clip_vision->vision_model.projection_dim);
                ggml_set_f32(c_crossattn, 0.f);
            } else {
                sd_image_f32_t image         = sd_image_t_to_sd_image_f32_t(init_image);
                sd_image_f32_t resized_image = clip_preprocess(image, clip_vision->vision_model.image_size);
                free(image.data);
                image.data = NULL;

                ggml_tensor* pixel_values = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, resized_image.width, resized_image.height, 3, 1);
                sd_image_f32_to_tensor(resized_image.data, pixel_values, false);
                free(resized_image.data);
                resized_image.data = NULL;

                // print_ggml_tensor(pixel_values);
                clip_vision->compute(n_threads, pixel_values, &c_crossattn, work_ctx);
                // print_ggml_tensor(c_crossattn);
            }
        }

        // c_concat
        struct ggml_tensor* c_concat = NULL;
        {
            if (force_zero_embeddings) {
                c_concat = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, width / 8, height / 8, 4, 1);
                ggml_set_f32(c_concat, 0.f);
            } else {
                ggml_tensor* init_img = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, width, height, 3, 1);

                if (width != init_image.width || height != init_image.height) {
                    sd_image_f32_t image         = sd_image_t_to_sd_image_f32_t(init_image);
                    sd_image_f32_t resized_image = resize_sd_image_f32_t(image, width, height);
                    free(image.data);
                    image.data = NULL;
                    sd_image_f32_to_tensor(resized_image.data, init_img, false);
                    free(resized_image.data);
                    resized_image.data = NULL;
                } else {
                    sd_image_to_tensor(init_image.data, init_img);
                }
                if (augmentation_level > 0.f) {
                    struct ggml_tensor* noise = ggml_dup_tensor(work_ctx, init_img);
                    ggml_tensor_set_f32_randn(noise, rng);
                    // encode_pixels += torch.randn_like(pixels) * augmentation_level
                    ggml_tensor_scale(noise, augmentation_level);
                    ggml_tensor_add(init_img, noise);
                }
                ggml_tensor* moments = encode_first_stage(work_ctx, init_img);
                c_concat             = get_first_stage_encoding(work_ctx, moments);
            }
        }

        // y
        struct ggml_tensor* y = NULL;
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

    ggml_tensor* sample(ggml_context* work_ctx,
                        ggml_tensor* init_latent,
                        ggml_tensor* noise,
                        SDCondition cond,
                        SDCondition uncond,
                        ggml_tensor* control_hint,
                        float control_strength,
                        float min_cfg,
                        float cfg_scale,
                        sample_method_t method,
                        const std::vector<float>& sigmas,
                        int start_merge_step,
                        SDCondition id_cond) {
        size_t steps = sigmas.size() - 1;
        // noise = load_tensor_from_file(work_ctx, "./rand0.bin");
        // print_ggml_tensor(noise);
        struct ggml_tensor* x = ggml_dup_tensor(work_ctx, init_latent);
        copy_ggml_tensor(x, init_latent);
        x = denoiser->noise_scaling(sigmas[0], noise, x);

        struct ggml_tensor* noised_input = ggml_dup_tensor(work_ctx, noise);

        bool has_unconditioned = cfg_scale != 1.0 && uncond.c_crossattn != NULL;

        // denoise wrapper
        struct ggml_tensor* out_cond   = ggml_dup_tensor(work_ctx, x);
        struct ggml_tensor* out_uncond = NULL;
        if (has_unconditioned) {
            out_uncond = ggml_dup_tensor(work_ctx, x);
        }
        struct ggml_tensor* denoised = ggml_dup_tensor(work_ctx, x);

        auto denoise = [&](ggml_tensor* input, float sigma, int step) -> ggml_tensor* {
            if (step == 1) {
                pretty_progress(0, (int)steps, 0);
            }
            int64_t t0 = ggml_time_us();

            std::vector<float> scaling = denoiser->get_scalings(sigma);
            GGML_ASSERT(scaling.size() == 3);
            float c_skip = scaling[0];
            float c_out  = scaling[1];
            float c_in   = scaling[2];

            float t = denoiser->sigma_to_t(sigma);
            std::vector<float> timesteps_vec(x->ne[3], t);  // [N, ]
            auto timesteps = vector_to_ggml_tensor(work_ctx, timesteps_vec);

            copy_ggml_tensor(noised_input, input);
            // noised_input = noised_input * c_in
            ggml_tensor_scale(noised_input, c_in);

            std::vector<struct ggml_tensor*> controls;

            if (control_hint != NULL) {
                control_net->compute(n_threads, noised_input, control_hint, timesteps, cond.c_crossattn, cond.c_vector);
                controls = control_net->controls;
                // print_ggml_tensor(controls[12]);
                // GGML_ASSERT(0);
            }

            if (start_merge_step == -1 || step <= start_merge_step) {
                // cond
                diffusion_model->compute(n_threads,
                                         noised_input,
                                         timesteps,
                                         cond.c_crossattn,
                                         cond.c_concat,
                                         cond.c_vector,
                                         -1,
                                         controls,
                                         control_strength,
                                         &out_cond);
            } else {
                diffusion_model->compute(n_threads,
                                         noised_input,
                                         timesteps,
                                         id_cond.c_crossattn,
                                         cond.c_concat,
                                         id_cond.c_vector,
                                         -1,
                                         controls,
                                         control_strength,
                                         &out_cond);
            }

            float* negative_data = NULL;
            if (has_unconditioned) {
                // uncond
                if (control_hint != NULL) {
                    control_net->compute(n_threads, noised_input, control_hint, timesteps, uncond.c_crossattn, uncond.c_vector);
                    controls = control_net->controls;
                }
                diffusion_model->compute(n_threads,
                                         noised_input,
                                         timesteps,
                                         uncond.c_crossattn,
                                         uncond.c_concat,
                                         uncond.c_vector,
                                         -1,
                                         controls,
                                         control_strength,
                                         &out_uncond);
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
                    int64_t ne3 = out_cond->ne[3];
                    if (min_cfg != cfg_scale && ne3 != 1) {
                        int64_t i3  = i / out_cond->ne[0] * out_cond->ne[1] * out_cond->ne[2];
                        float scale = min_cfg + (cfg_scale - min_cfg) * (i3 * 1.0f / ne3);
                    } else {
                        latent_result = negative_data[i] + cfg_scale * (positive_data[i] - negative_data[i]);
                    }
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
            return denoised;
        };

        sample_k_diffusion(method, denoise, work_ctx, x, sigmas, rng);

        x = denoiser->inverse_noise_scaling(sigmas[sigmas.size() - 1], x);

        if (control_net) {
            control_net->free_control_ctx();
            control_net->free_compute_buffer();
        }
        diffusion_model->free_compute_buffer();
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
        int64_t W = x->ne[0];
        int64_t H = x->ne[1];
        int64_t C = 8;
        if (use_tiny_autoencoder) {
            C = 4;
        } else {
            if (version == VERSION_3_2B) {
                C = 32;
            }
        }
        ggml_tensor* result = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32,
                                                 decode ? (W * 8) : (W / 8),  // width
                                                 decode ? (H * 8) : (H / 8),  // height
                                                 decode ? 3 : C,
                                                 x->ne[3]);  // channels
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
                    first_stage_model->compute(n_threads, in, decode, &out);
                };
                sd_tiling(x, result, 8, 32, 0.5f, on_tiling);
            } else {
                first_stage_model->compute(n_threads, x, decode, &result);
            }
            first_stage_model->free_compute_buffer();
            if (decode) {
                ggml_tensor_scale_output(result);
            }
        } else {
            if (vae_tiling && decode) {  // TODO: support tiling vae encode
                // split latent in 64x64 tiles and compute in several steps
                auto on_tiling = [&](ggml_tensor* in, ggml_tensor* out, bool init) {
                    tae_first_stage->compute(n_threads, in, decode, &out);
                };
                sd_tiling(x, result, 8, 64, 0.5f, on_tiling);
            } else {
                tae_first_stage->compute(n_threads, x, decode, &result);
            }
            tae_first_stage->free_compute_buffer();
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
                     const char* id_embed_dir_c_str,
                     bool vae_decode_only,
                     bool vae_tiling,
                     bool free_params_immediately,
                     int n_threads,
                     enum sd_type_t wtype,
                     enum rng_type_t rng_type,
                     enum schedule_t s,
                     bool keep_clip_on_cpu,
                     bool keep_control_net_cpu,
                     bool keep_vae_on_cpu) {
    sd_ctx_t* sd_ctx = (sd_ctx_t*)malloc(sizeof(sd_ctx_t));
    if (sd_ctx == NULL) {
        return NULL;
    }
    std::string model_path(model_path_c_str);
    std::string vae_path(vae_path_c_str);
    std::string taesd_path(taesd_path_c_str);
    std::string control_net_path(control_net_path_c_str);
    std::string embd_path(embed_dir_c_str);
    std::string id_embd_path(id_embed_dir_c_str);
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
                                    id_embd_path,
                                    taesd_path,
                                    vae_tiling,
                                    (ggml_type)wtype,
                                    s,
                                    keep_clip_on_cpu,
                                    keep_control_net_cpu,
                                    keep_vae_on_cpu)) {
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

sd_image_t* generate_image(sd_ctx_t* sd_ctx,
                           struct ggml_context* work_ctx,
                           ggml_tensor* init_latent,
                           std::string prompt,
                           std::string negative_prompt,
                           int clip_skip,
                           float cfg_scale,
                           int width,
                           int height,
                           enum sample_method_t sample_method,
                           const std::vector<float>& sigmas,
                           int64_t seed,
                           int batch_count,
                           const sd_image_t* control_cond,
                           float control_strength,
                           float style_ratio,
                           bool normalize_input,
                           std::string input_id_images_path) {
    if (seed < 0) {
        // Generally, when using the provided command line, the seed is always >0.
        // However, to prevent potential issues if 'stable-diffusion.cpp' is invoked as a library
        // by a third party with a seed <0, let's incorporate randomization here.
        srand((int)time(NULL));
        seed = rand();
    }

    // for (auto v : sigmas) {
    //     std::cout << v << " ";
    // }
    // std::cout << std::endl;

    int sample_steps = sigmas.size() - 1;

    // Apply lora
    // 从正向提示词中抽取lora的限定词和权重, 保存为map对象, 放在pair.first中
    // pair.second存放排除掉lora相关项的提示词.
    auto result_pair                                = extract_and_remove_lora(prompt);
    std::unordered_map<std::string, float> lora_f2m = result_pair.first;  // lora_name -> multiplier

    for (auto& kv : lora_f2m) {
        LOG_DEBUG("lora %s:%.2f", kv.first.c_str(), kv.second);
    }

    prompt = result_pair.second;
    LOG_DEBUG("prompt after extract and remove lora: \"%s\"", prompt.c_str());

    int64_t t0 = ggml_time_ms();
    // 应用lora, 传入的map中有lora的模型名字, 权重.
    sd_ctx->sd->apply_loras(lora_f2m);
    int64_t t1 = ggml_time_ms();
    LOG_INFO("apply_loras completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);

    // Photo Maker
    // PhotoMaker功能允许用户通过提供参考图像来指导生成过程，从而创建出更加符合预期的图像。
    /*
     * 代码的主要流程和解释：

     * Lora应用:
        如果pmid_lora（假定是一个LoRA模型）还没有被应用过，则先应用它。
        LoRA（Low-Rank Adaptation）是一种轻量级的微调方法，用于对预训练模型进行适应性调整。
        应用LoRA之后，如果设置了free_params_immediately，则释放LoRA模型的参数缓冲区。
     * 读取并预处理ID图像:
        如果提供了PhotoMaker模型文件和输入ID图像路径，程序会遍历该路径下的所有图像文件。
        使用stbi_load函数加载图像数据。
        调用preprocess_id_image函数对图像进行预处理。
        预处理后的图像存储在input_id_images向量中。
     * 构建条件向量:
        如果存在预处理后的ID图像，则构建条件向量。
        设置style_strength（风格强度），这会影响生成图像与输入图像相似的程度。
        创建一个四维张量init_img，用于存储预处理后的图像数据。
        使用mean和std对图像数据进行归一化（如果normalize_input为真）。
        调用get_learned_condition_with_trigger函数生成条件向量id_cond以及class_tokens_mask。
     * ID编码:
        使用id_encoder函数对条件向量进行进一步处理。
        这一步可能会涉及注意力机制，以确保模型关注到输入图像的关键特征。
     * 处理文本提示:
        从原始文本提示中移除触发词（trigger word），得到prompt_text_only。
        这是为了避免在延迟条件生成阶段重复使用触发词。
        更新prompt变量为prompt_text_only。

     * 清理和后续处理:
        清理已加载的图像数据。
        如果没有提供输入ID图像，关闭PhotoMaker功能。
     */
    std::string prompt_text_only;
    ggml_tensor* init_img = NULL;
    SDCondition id_cond;
    std::vector<bool> class_tokens_mask;
    // stacked_id用来标记是否启用了 PhotoMaker 功能。
    // PhotoMaker 特性允许模型在生成图像时考虑额外的输入图像作为参考，以便生成的图像能够更好地匹配这些参考图像的风格或内容。
    if (sd_ctx->sd->stacked_id) {
        if (!sd_ctx->sd->pmid_lora->applied) {
            // 加载 LoRA 模型（如果尚未加载）并应用它。
            t0 = ggml_time_ms();
            sd_ctx->sd->pmid_lora->apply(sd_ctx->sd->tensors, sd_ctx->sd->n_threads);
            t1                             = ggml_time_ms();
            sd_ctx->sd->pmid_lora->applied = true;
            LOG_INFO("pmid_lora apply completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);
            if (sd_ctx->sd->free_params_immediately) {
                sd_ctx->sd->pmid_lora->free_params_buffer();
            }
        }
        // 加载和预处理输入的 ID 图像
        // preprocess input id images
        // id_image 通常指的是一个特定类型的输入图像，这种图像被用作生成新图像时的一种参考或者身份标识。
        // 在这个场景中，id_image 可能是指一种用于指导生成过程的图像，使得生成的图像能够具有与 id_image 类似的特征或风格。
        std::vector<sd_image_t*> input_id_images;
        if (sd_ctx->sd->pmid_model && input_id_images_path.size() > 0) {
            std::vector<std::string> img_files = get_files_from_dir(input_id_images_path);
            for (std::string img_file : img_files) {
                int c = 0;
                int width, height;
                uint8_t* input_image_buffer = stbi_load(img_file.c_str(), &width, &height, &c, 3);
                if (input_image_buffer == NULL) {
                    LOG_ERROR("PhotoMaker load image from '%s' failed", img_file.c_str());
                    continue;
                } else {
                    LOG_INFO("PhotoMaker loaded image from '%s'", img_file.c_str());
                }
                sd_image_t* input_image = NULL;
                input_image             = new sd_image_t{(uint32_t)width,
                                             (uint32_t)height,
                                             3,
                                             input_image_buffer};
                // 进行预处理
                input_image             = preprocess_id_image(input_image);
                if (input_image == NULL) {
                    LOG_ERROR("preprocess input id image from '%s' failed", img_file.c_str());
                    continue;
                }
                input_id_images.push_back(input_image);
            }
        }
        if (input_id_images.size() > 0) {
            // 为预处理后的ID图像, 构建条件变量
            // 设置模型的风格强度, 这影响输入与生成图像的相似度
            sd_ctx->sd->pmid_model->style_strength = style_ratio;
            int32_t w                              = input_id_images[0]->width;
            int32_t h                              = input_id_images[0]->height;
            int32_t channels                       = input_id_images[0]->channel;
            // 这是上一步中得到的id图像的个数
            int32_t num_input_images               = (int32_t)input_id_images.size();
            // init_img是一个四维张量, 用来存储处理后的图像数据
            // 第4维是图像数量
            init_img                               = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, w, h, channels, num_input_images);
            // 使用mean和std对图像数据进行归一化（如果normalize_input为真）
            // TODO: move these to somewhere else and be user settable
            float mean[] = {0.48145466f, 0.4578275f, 0.40821073f};
            float std[]  = {0.26862954f, 0.26130258f, 0.27577711f};
            for (int i = 0; i < num_input_images; i++) {
                sd_image_t* init_image = input_id_images[i];
                if (normalize_input)
                    sd_mul_images_to_tensor(init_image->data, init_img, i, mean, std);
                else
                    sd_mul_images_to_tensor(init_image->data, init_img, i, NULL, NULL);
            }
            // 调用get_learned_condition_with_trigger函数生成条件向量id_cond以及class_tokens_mask。
            // sd_ctx->sd->cond_stage_model 指向模型中负责处理条件阶段（conditional stage）的子模型
            // prompt: 文本提示，用于描述要生成的图像的内容
            // clip_skip: CLIP模型中可以跳过的层的数量。CLIP模型通常用于将文本提示转化为嵌入向量。
            // sd_ctx->sd->diffusion_model->get_adm_in_channels(): 获取扩散模型（diffusion model）的输入通道数。这通常是与模型架构相关的内部参数。
            // 整个函数调用的目的是为了获取一个条件元组（cond_tup），它将用于后续的图像生成过程。
            // 条件元组可能包含了文本嵌入、图像尺寸信息以及其他用于引导生成过程的信息。
            t0                = ggml_time_ms();
            auto cond_tup     = sd_ctx->sd->cond_stage_model->get_learned_condition_with_trigger(work_ctx,
                                                                                                 sd_ctx->sd->n_threads, prompt,
                                                                                                 clip_skip,
                                                                                                 width,
                                                                                                 height,
                                                                                                 num_input_images,
                                                                                                 sd_ctx->sd->diffusion_model->get_adm_in_channels());
            // cond_tup其结构为 std::tuple<SDCondition, std::vector<bool>>。
            //
            // SDCondition:
            // 这个类型应该是代表了条件信息的一个结构体或类。它包含了用于指导图像生成的重要数据，比如文本嵌入、图像尺寸等。
            // std::vector<bool>:
            // 这是一个布尔值的向量，通常用来表示某个条件是否适用，或者是一些标记信息，例如哪些类别的 token 应该被处理。
            id_cond           = std::get<0>(cond_tup);
            class_tokens_mask = std::get<1>(cond_tup);  //
            /*
             * id_encoder 函数被调用，它接收如下参数：
                work_ctx: 工作上下文，可能是一个计算上下文或资源管理器。
                init_img: 初始图像，可能是用于图像到图像任务的输入图像。
                id_cond.c_crossattn: 与交叉注意力相关的数据。
                class_tokens_mask: 之前从 cond_tup 中提取的布尔向量。
             * id_encoder 函数的作用可能是对 c_crossattn 进行编码，可能涉及更新交叉注意力的权重，
             * 以反映 class_tokens_mask 所指示的信息。
             */
            id_cond.c_crossattn = sd_ctx->sd->id_encoder(work_ctx, init_img, id_cond.c_crossattn, class_tokens_mask);
            t1                  = ggml_time_ms();
            LOG_INFO("Photomaker ID Stacking, taking %" PRId64 " ms", t1 - t0);
            if (sd_ctx->sd->free_params_immediately) {
                sd_ctx->sd->pmid_model->free_params_buffer();
            }
            // 将触发词干掉.
            // Encode input prompt without the trigger word for delayed conditioning
            prompt_text_only = sd_ctx->sd->cond_stage_model->remove_trigger_from_prompt(work_ctx, prompt);
            // printf("%s || %s \n", prompt.c_str(), prompt_text_only.c_str());
            prompt = prompt_text_only;  //
            // if (sample_steps < 50) {
            //     LOG_INFO("sampling steps increases from %d to 50 for PHOTOMAKER", sample_steps);
            //     sample_steps = 50;
            // }
        } else {
            LOG_WARN("Provided PhotoMaker model file, but NO input ID images");
            LOG_WARN("Turn off PhotoMaker");
            sd_ctx->sd->stacked_id = false;
        }
        for (sd_image_t* img : input_id_images) {
            free(img->data);
        }
        input_id_images.clear();
    }

    // 正向和负向的提示词转成向量, 注意这里get_learned_condition没有trigger
    // Get learned condition
    t0               = ggml_time_ms();
    SDCondition cond = sd_ctx->sd->cond_stage_model->get_learned_condition(work_ctx,
                                                                           sd_ctx->sd->n_threads,
                                                                           prompt,
                                                                           clip_skip,
                                                                           width,
                                                                           height,
                                                                           sd_ctx->sd->diffusion_model->get_adm_in_channels());

    SDCondition uncond;
    if (cfg_scale != 1.0) {
        bool force_zero_embeddings = false;
        if (sd_ctx->sd->version == VERSION_XL && negative_prompt.size() == 0) {
            force_zero_embeddings = true;
        }
        uncond = sd_ctx->sd->cond_stage_model->get_learned_condition(work_ctx,
                                                                     sd_ctx->sd->n_threads,
                                                                     negative_prompt,
                                                                     clip_skip,
                                                                     width,
                                                                     height,
                                                                     sd_ctx->sd->diffusion_model->get_adm_in_channels(),
                                                                     force_zero_embeddings);
    }
    t1 = ggml_time_ms();
    LOG_INFO("get_learned_condition completed, taking %" PRId64 " ms", t1 - t0);

    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->cond_stage_model->free_params_buffer();
    }

    // Control net hint
    struct ggml_tensor* image_hint = NULL; // 存储图像提示信息
    if (control_cond != NULL) {
        // ggml_new_tensor_4d 创建了一个新的四维张量，其尺寸为 width x height x 3 x 1。这里的 3 表示图像的通道数（假设是 RGB 图像），1 表示批量大小，这里只处理一张图像。
        image_hint = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, width, height, 3, 1); // 初始化hint
        // sd_image_to_tensor 函数将图像数据转换为张量形式
        // sd_image_to_tensor 函数负责将 control_cond->data 中的图像数据填充到新创建的张量 image_hint 中。
        sd_image_to_tensor(control_cond->data, image_hint); //
    }

    // Sample
    // Sample包括加噪音和去噪音两个过程. 算法例如DDIM (Denoising Diffusion Implicit Models)
    // Sample决定如何生成一系列潜在空间(latent space)的张量，这些张量随后可以被解码成图像
    // final_latents 是一个用于收集生成的潜在张量的向量，这些张量之后会被用来解码成图像。
    std::vector<struct ggml_tensor*> final_latents;  // collect latents to decode
    // C 表示通道数，对于版本 3.2B 的 Stable Diffusion 模型，通道数为 16；否则默认为 4。
    int C = 4;
    if (sd_ctx->sd->version == VERSION_3_2B) {
        C = 16;
    }
    int W = width / 8;
    int H = height / 8;
    LOG_INFO("sampling using %s method", sampling_methods_str[sample_method]);
    // batch_count是指生成图像的批量数量
    // 在一个训练或者生成过程中，不是一次只生成一张图像，而是同时生成多张图像，每张图像使用不同的随机种子。
    // batch_count 控制着循环的迭代次数，每次迭代都会生成一个新的图像。每个图像的生成过程都是独立的，并且使用不同的随机种子来初始化噪声张量，这有助于确保生成的图像具有多样性。
    for (int b = 0; b < batch_count; b++) {
        int64_t sampling_start = ggml_time_ms();
        int64_t cur_seed       = seed + b;
        LOG_INFO("generating image: %i/%i - seed %" PRId64, b + 1, batch_count, cur_seed);

        sd_ctx->sd->rng->manual_seed(cur_seed);
        // noise 张量是根据潜在空间的尺寸 (W, H, C) 创建的一个 4D 张量，类型为 GGML_TYPE_F32（32位浮点数），用于初始化每个潜在张量的噪声。
        // ggml_tensor_set_f32_randn(noise, sd_ctx->sd->rng)：用正态分布随机值填充噪声张量。
        struct ggml_tensor* x_t   = init_latent;
        struct ggml_tensor* noise = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, W, H, C, 1);
        ggml_tensor_set_f32_randn(noise, sd_ctx->sd->rng);

        // 如果使用了 stacked_id，则会计算 start_merge_step，这可能是为了混合不同的风格
        // stacked_id 表示 是否使用了id图像, 即参考图
        int start_merge_step = -1;
        if (sd_ctx->sd->stacked_id) {
            start_merge_step = int(sd_ctx->sd->pmid_model->style_strength / 100.f * sample_steps);
            // if (start_merge_step > 30)
            //     start_merge_step = 30;
            LOG_INFO("PHOTOMAKER: start_merge_step: %d", start_merge_step);
        }

        /**
         * sd_ctx->sd->sample 调用了采样函数，该函数负责执行从噪声到潜在张量的转换。
            x_t 是初始的潜在张量。
            noise 是前面创建的噪声张量。
            cond 和 uncond 可能分别代表条件和无条件的输入。
            image_hint 和 control_strength 用于控制生成过程。
            cfg_scale 用于控制条件自由引导的程度。
            sample_method 指定采样方法。
            sigmas 用于指定噪声水平。
            start_merge_step 和 id_cond 用于控制混合过程。
         */
        struct ggml_tensor* x_0 = sd_ctx->sd->sample(work_ctx,
                                                     x_t,
                                                     noise,
                                                     cond,
                                                     uncond,
                                                     image_hint,
                                                     control_strength,
                                                     cfg_scale,
                                                     cfg_scale,
                                                     sample_method,
                                                     sigmas,
                                                     start_merge_step,
                                                     id_cond);
        // struct ggml_tensor* x_0 = load_tensor_from_file(ctx, "samples_ddim.bin");
        // print_ggml_tensor(x_0);
        int64_t sampling_end = ggml_time_ms();
        LOG_INFO("sampling completed, taking %.2fs", (sampling_end - sampling_start) * 1.0f / 1000);
        final_latents.push_back(x_0);
    }

    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->diffusion_model->free_params_buffer();
    }
    int64_t t3 = ggml_time_ms();
    LOG_INFO("generating %" PRId64 " latent images completed, taking %.2fs", final_latents.size(), (t3 - t1) * 1.0f / 1000);

    // Decode to image
    LOG_INFO("decoding %zu latents", final_latents.size());
    std::vector<struct ggml_tensor*> decoded_images;  // collect decoded images
    for (size_t i = 0; i < final_latents.size(); i++) {
        t1                      = ggml_time_ms();
        // 从latent空间的张量 解码到 像素空间的张量
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
        sd_ctx->sd->first_stage_model->free_params_buffer();
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

    return result_images;
}

sd_image_t* txt2img(sd_ctx_t* sd_ctx,
                    const char* prompt_c_str,
                    const char* negative_prompt_c_str,
                    int clip_skip,  // CLIP 编码器的层数，用于控制文本嵌入的深度
                    float cfg_scale, // 条件缩放因子，用于调整提示的影响程度。
                    int width,
                    int height,
                    enum sample_method_t sample_method,
                    int sample_steps, // 采样步数，即迭代次数
                    int64_t seed,
                    int batch_count, // 批次数量，即一次生成多少张图像
                    const sd_image_t* control_cond, // 控制条件图像，可以为空，用于指导生成过程。
                    float control_strength,  // 控制条件图像的强度
                    float style_ratio,  // 风格混合比率
                    bool normalize_input, // 是否对输入进行归一化
                    const char* input_id_images_path_c_str) {
    LOG_DEBUG("txt2img %dx%d", width, height);
    if (sd_ctx == NULL) {
        return NULL;
    }

    // 初始化内存上下文:
    // 计算所需的内存大小，这取决于图像的大小、批次数量以及模型版本等因素。

    struct ggml_init_params params;
    // 这里首先分配了 10MB 的内存作为基础大小。这是为了保证有足够的内存来存储一些基础的数据结构和临时变量。
    params.mem_size = static_cast<size_t>(10 * 1024 * 1024);  // 10 MB
    // 如果模型版本是 VERSION_3_2B，则将基础内存大小乘以 3。这是因为不同版本的模型可能需要更多的内存来处理额外的数据结构或更大的中间张量。
    if (sd_ctx->sd->version == VERSION_3_2B) {
        params.mem_size *= 3;
    }
    // 为了处理与堆叠 ID 相关的额外数据。
    if (sd_ctx->sd->stacked_id) {
        params.mem_size += static_cast<size_t>(10 * 1024 * 1024);  // 10 MB
    }
    // 这一行代码是基于图像尺寸来计算潜空间张量的大小。通常情况下，稳定扩散模型会在每个像素位置上使用多个通道来表示潜在变量。
    // 这里使用 width * height * 3 是假设每个像素有 3 个通道（例如 RGB 图像），而每个通道使用单精度浮点数表示，因此需要乘以 sizeof(float)。
    params.mem_size += width * height * 3 * sizeof(float);
    // 如果生成多个图像（即 batch_count > 1），那么需要为每个批次分配相应的内存。这意味着如果需要同时生成多张图像，内存需求会相应地增加。
    params.mem_size *= batch_count;
    params.mem_buffer = NULL;
    params.no_alloc   = false;
    // LOG_DEBUG("mem_size %u ", params.mem_size);

    // 使用 ggml_init 初始化一个新的 ggml_context，用于存储中间计算结果。
    struct ggml_context* work_ctx = ggml_init(params);
    if (!work_ctx) {
        LOG_ERROR("ggml_init() failed");
        return NULL;
    }

    size_t t0 = ggml_time_ms();

    // 生成初始潜空间向量:
    // 根据模型版本设置初始潜空间向量的值。

    // sigmas 是一个浮点数向量，包含了采样步骤中每一阶段的噪声水平。
    std::vector<float> sigmas = sd_ctx->sd->denoiser->get_sigmas(sample_steps);

    // C 表示通道数，对于不同的模型版本有不同的默认值（4 或 16）。
    int C = 4;
    if (sd_ctx->sd->version == VERSION_3_2B) {
        C = 16;
    }
    // W: 图像宽度除以 8。这是因为稳定扩散模型通常会对图像进行下采样，即将图像尺寸缩小为原来的 1/8。这样做的目的是减少计算量，并且在潜在空间中处理较小的张量。
    // H: 图像高度除以 8，同样是为了适应模型的下采样要求。
    int W                    = width / 8;
    int H                    = height / 8;
    // work_ctx: 是一个指向 ggml_context 的指针，用于管理内存和计算图。在这个上下文中创建的张量将使用分配给 work_ctx 的内存。
    // GGML_TYPE_F32: 指定张量的数据类型为 32 位浮点数。
    // W 和 H: 分别是前面计算出的图像宽度和高度除以 8 的结果。
    // C: 通道数。根据模型版本的不同，C 的值可能为 4 或 16。在稳定扩散模型中，这个值表示每个位置上的通道数，通常对应于潜在空间中的特征图。
    // 1: 第四个维度。在这个上下文中，它通常是 1，表示这是一个二维的特征图（即使在四维张量中表示）。
    // 这段代码创建了一个四维张量 init_latent，它将用于表示图像的初始潜空间向量。
    // 这个张量的形状为 (W, H, C, 1)，其中 W 和 H 分别是原始图像宽度和高度除以 8，而 C 是通道数。这个张量将在后续的生成过程中用于表示潜在空间中的初始状态。
    //
    // 执行 sd 命令（假设是指稳定扩散模型的命令）时传入的量化类型，通常与创建张量时使用的数据类型是不同的概念。
    // 量化类型指的是模型权重如何被存储和计算，而张量的数据类型是指在运行时模型内部的计算所采用的数据精度。
    // 在创建张量时使用 GGML_TYPE_F32，意味着这些张量在内存中将以32位浮点数的形式存储。这通常用于确保计算精度，尤其是在神经网络的前向传播和反向传播过程中。
    //
    // 当你执行 sd 命令时，量化类型通常是在加载模型时指定的。量化类型会影响模型加载的方式以及模型在内存中的表示形式。
    ggml_tensor* init_latent = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, W, H, C, 1);
    if (sd_ctx->sd->version == VERSION_3_2B) {
        ggml_set_f32(init_latent, 0.0609f);
    } else {
        ggml_set_f32(init_latent, 0.f);
    }

    // 生成图像:
    // 调用 generate_image 函数来生成图像。这个函数负责核心的生成逻辑，包括条件编码、潜空间向量的采样、去噪等步骤。
    sd_image_t* result_images = generate_image(sd_ctx,
                                               work_ctx,
                                               init_latent,
                                               prompt_c_str,
                                               negative_prompt_c_str,
                                               clip_skip,
                                               cfg_scale,
                                               width,
                                               height,
                                               sample_method,
                                               sigmas,
                                               seed,
                                               batch_count,
                                               control_cond,
                                               control_strength,
                                               style_ratio,
                                               normalize_input,
                                               input_id_images_path_c_str);

    size_t t1 = ggml_time_ms();

    LOG_INFO("txt2img completed in %.2fs", (t1 - t0) * 1.0f / 1000);

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
                    int batch_count,
                    const sd_image_t* control_cond,
                    float control_strength,
                    float style_ratio,
                    bool normalize_input,
                    const char* input_id_images_path_c_str) {
    LOG_DEBUG("img2img %dx%d", width, height);
    if (sd_ctx == NULL) {
        return NULL;
    }

    struct ggml_init_params params;
    params.mem_size = static_cast<size_t>(10 * 1024 * 1024);  // 10 MB
    if (sd_ctx->sd->version == VERSION_3_2B) {
        params.mem_size *= 2;
    }
    if (sd_ctx->sd->stacked_id) {
        params.mem_size += static_cast<size_t>(10 * 1024 * 1024);  // 10 MB
    }
    params.mem_size += width * height * 3 * sizeof(float) * 2;
    params.mem_size *= batch_count;
    params.mem_buffer = NULL;
    params.no_alloc   = false;
    // LOG_DEBUG("mem_size %u ", params.mem_size);

    struct ggml_context* work_ctx = ggml_init(params);
    if (!work_ctx) {
        LOG_ERROR("ggml_init() failed");
        return NULL;
    }

    size_t t0 = ggml_time_ms();

    if (seed < 0) {
        srand((int)time(NULL));
        seed = rand();
    }
    sd_ctx->sd->rng->manual_seed(seed);

    ggml_tensor* init_img = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, width, height, 3, 1);
    sd_image_to_tensor(init_image.data, init_img);
    ggml_tensor* init_latent = NULL;
    if (!sd_ctx->sd->use_tiny_autoencoder) {
        ggml_tensor* moments = sd_ctx->sd->encode_first_stage(work_ctx, init_img);
        init_latent          = sd_ctx->sd->get_first_stage_encoding(work_ctx, moments);
    } else {
        init_latent = sd_ctx->sd->encode_first_stage(work_ctx, init_img);
    }
    print_ggml_tensor(init_latent, true);
    size_t t1 = ggml_time_ms();
    LOG_INFO("encode_first_stage completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);

    std::vector<float> sigmas = sd_ctx->sd->denoiser->get_sigmas(sample_steps);
    size_t t_enc              = static_cast<size_t>(sample_steps * strength);
    LOG_INFO("target t_enc is %zu steps", t_enc);
    std::vector<float> sigma_sched;
    sigma_sched.assign(sigmas.begin() + sample_steps - t_enc - 1, sigmas.end());

    sd_image_t* result_images = generate_image(sd_ctx,
                                               work_ctx,
                                               init_latent,
                                               prompt_c_str,
                                               negative_prompt_c_str,
                                               clip_skip,
                                               cfg_scale,
                                               width,
                                               height,
                                               sample_method,
                                               sigma_sched,
                                               seed,
                                               batch_count,
                                               control_cond,
                                               control_strength,
                                               style_ratio,
                                               normalize_input,
                                               input_id_images_path_c_str);

    size_t t2 = ggml_time_ms();

    LOG_INFO("img2img completed in %.2fs", (t1 - t0) * 1.0f / 1000);

    return result_images;
}

SD_API sd_image_t* img2vid(sd_ctx_t* sd_ctx,
                           sd_image_t init_image,
                           int width,
                           int height,
                           int video_frames,
                           int motion_bucket_id,
                           int fps,
                           float augmentation_level,
                           float min_cfg,
                           float cfg_scale,
                           enum sample_method_t sample_method,
                           int sample_steps,
                           float strength,
                           int64_t seed) {
    if (sd_ctx == NULL) {
        return NULL;
    }

    LOG_INFO("img2vid %dx%d", width, height);

    std::vector<float> sigmas = sd_ctx->sd->denoiser->get_sigmas(sample_steps);

    struct ggml_init_params params;
    params.mem_size = static_cast<size_t>(10 * 1024) * 1024;  // 10 MB
    params.mem_size += width * height * 3 * sizeof(float) * video_frames;
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

    int64_t t0 = ggml_time_ms();

    SDCondition cond = sd_ctx->sd->get_svd_condition(work_ctx,
                                                     init_image,
                                                     width,
                                                     height,
                                                     fps,
                                                     motion_bucket_id,
                                                     augmentation_level);

    auto uc_crossattn = ggml_dup_tensor(work_ctx, cond.c_crossattn);
    ggml_set_f32(uc_crossattn, 0.f);

    auto uc_concat = ggml_dup_tensor(work_ctx, cond.c_concat);
    ggml_set_f32(uc_concat, 0.f);

    auto uc_vector = ggml_dup_tensor(work_ctx, cond.c_vector);

    SDCondition uncond = SDCondition(uc_crossattn, uc_vector, uc_concat);

    int64_t t1 = ggml_time_ms();
    LOG_INFO("get_learned_condition completed, taking %" PRId64 " ms", t1 - t0);
    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->clip_vision->free_params_buffer();
    }

    sd_ctx->sd->rng->manual_seed(seed);
    int C                   = 4;
    int W                   = width / 8;
    int H                   = height / 8;
    struct ggml_tensor* x_t = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, W, H, C, video_frames);
    ggml_set_f32(x_t, 0.f);

    struct ggml_tensor* noise = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, W, H, C, video_frames);
    ggml_tensor_set_f32_randn(noise, sd_ctx->sd->rng);

    LOG_INFO("sampling using %s method", sampling_methods_str[sample_method]);
    struct ggml_tensor* x_0 = sd_ctx->sd->sample(work_ctx,
                                                 x_t,
                                                 noise,
                                                 cond,
                                                 uncond,
                                                 {},
                                                 0.f,
                                                 min_cfg,
                                                 cfg_scale,
                                                 sample_method,
                                                 sigmas,
                                                 -1,
                                                 SDCondition(NULL, NULL, NULL));

    int64_t t2 = ggml_time_ms();
    LOG_INFO("sampling completed, taking %.2fs", (t2 - t1) * 1.0f / 1000);
    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->diffusion_model->free_params_buffer();
    }

    struct ggml_tensor* img = sd_ctx->sd->decode_first_stage(work_ctx, x_0);
    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->first_stage_model->free_params_buffer();
    }
    if (img == NULL) {
        ggml_free(work_ctx);
        return NULL;
    }

    sd_image_t* result_images = (sd_image_t*)calloc(video_frames, sizeof(sd_image_t));
    if (result_images == NULL) {
        ggml_free(work_ctx);
        return NULL;
    }

    for (size_t i = 0; i < video_frames; i++) {
        auto img_i = ggml_view_3d(work_ctx, img, img->ne[0], img->ne[1], img->ne[2], img->nb[1], img->nb[2], img->nb[3] * i);

        result_images[i].width   = width;
        result_images[i].height  = height;
        result_images[i].channel = 3;
        result_images[i].data    = sd_tensor_to_image(img_i);
    }
    ggml_free(work_ctx);

    int64_t t3 = ggml_time_ms();

    LOG_INFO("img2vid completed in %.2fs", (t3 - t0) * 1.0f / 1000);

    return result_images;
}
