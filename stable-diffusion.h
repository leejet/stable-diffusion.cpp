#ifndef __STABLE_DIFFUSION_H__
#define __STABLE_DIFFUSION_H__

#include <memory>
#include <string>
#include <vector>

enum RNGType {
    STD_DEFAULT_RNG,
    CUDA_RNG
};

enum SampleMethod {
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

enum Schedule {
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
                    std::string taesd_path       = "",
                    bool free_params_immediately = false,
                    std::string lora_model_dir   = "",
                    RNGType rng_type             = STD_DEFAULT_RNG);
    bool load_from_file(const std::string& model_path,
                        const std::string& vae_path,
                        ggml_type wtype,
                        Schedule d = DEFAULT);
    std::vector<uint8_t*> txt2img(
        std::string prompt,
        std::string negative_prompt,
        float cfg_scale,
        int width,
        int height,
        SampleMethod sample_method,
        int sample_steps,
        int64_t seed,
        int batch_count);

    std::vector<uint8_t*> img2img(
        const uint8_t* init_img_data,
        std::string prompt,
        std::string negative_prompt,
        float cfg_scale,
        int width,
        int height,
        SampleMethod sample_method,
        int sample_steps,
        float strength,
        int64_t seed);
};

std::string sd_get_system_info();

#endif  // __STABLE_DIFFUSION_H__