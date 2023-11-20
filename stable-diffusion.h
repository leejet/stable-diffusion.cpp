#ifndef __STABLE_DIFFUSION_H__
#define __STABLE_DIFFUSION_H__

#include <memory>
#include <vector>

enum sd_log_level {
    DEBUG,
    INFO,
    WARN,
    ERROR
};

enum sd_rng_type {
    STD_DEFAULT_RNG,
    CUDA_RNG
};

enum sd_sample_method {
    EULER_A,
    EULER,
    HEUN,
    DPM2,
    DPMPP2S_A,
    DPMPP2M,
    DPMPP2Mv2,
    LCM,
    N_SAMPLE_METHODS
};

enum sd_sample_schedule {
    DEFAULT,
    DISCRETE,
    KARRAS,
    N_SCHEDULES
};

class StableDiffusionGGML;

class StableDiffusion {
private:
    std::shared_ptr<StableDiffusionGGML> sd;

public:
    StableDiffusion(int n_threads                = -1,
                    bool vae_decode_only         = false,
                    bool free_params_immediately = false,
                    std::string lora_model_dir   = "",
                    sd_rng_type rng_type = STD_DEFAULT_RNG);
    bool load_from_file(const std::string& file_path, sd_sample_schedule d = DEFAULT);
    std::vector<uint8_t> txt2img(
        std::string prompt,
        std::string negative_prompt,
        float cfg_scale,
        int width,
        int height,
        sd_sample_method sample_method,
        int sample_steps,
        int64_t seed);

    std::vector<uint8_t> img2img(
        const std::vector<uint8_t>& init_img,
        std::string prompt,
        std::string negative_prompt,
        float cfg_scale,
        int width,
        int height,
        sd_sample_method sample_method,
        int sample_steps,
        float strength,
        int64_t seed);
};

void set_sd_log_level(sd_log_level level);
std::string sd_get_system_info();

#endif  // __STABLE_DIFFUSION_H__