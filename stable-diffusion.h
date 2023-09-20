#ifndef __STABLE_DIFFUSION_H__
#define __STABLE_DIFFUSION_H__

#include <memory>
#include <vector>

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef STABLE_DIFFUSION_SHARED
#if defined(_WIN32) && !defined(__MINGW32__)
#ifdef STABLE_DIFFUSION_BUILD
#define STABLE_DIFFUSION_API __declspec(dllexport)
#else
#define STABLE_DIFFUSION_API __declspec(dllimport)
#endif
#else
#define STABLE_DIFFUSION_API __attribute__((visibility("default")))
#endif
#else
#define STABLE_DIFFUSION_API
#endif

#ifdef __cplusplus
extern "C"
{
#endif

    enum SDLogLevel
    {
        DEBUG,
        INFO,
        WARN,
        ERROR
    };

    enum RNGType
    {
        STD_DEFAULT_RNG,
        CUDA_RNG
    };

    enum SampleMethod
    {
        EULER_A,
        EULER,
        HEUN,
        DPM2,
        DPMPP2S_A,
        DPMPP2M,
        DPMPP2Mv2,
        N_SAMPLE_METHODS
    };

    enum Schedule
    {
        DEFAULT,
        DISCRETE,
        KARRAS,
        N_SCHEDULES
    };

    struct sd_txt2img_options;

    struct sd_img2img_options;

    // These method use for golang or other language binding

    //==============================sd_txt2img_options===============================
    STABLE_DIFFUSION_API void set_txt2img_prompt(sd_txt2img_options *opt, const char *prompt);
    STABLE_DIFFUSION_API void set_txt2img_negative_prompt(sd_txt2img_options *opt, const char *negative_prompt);
    STABLE_DIFFUSION_API void set_txt2img_cfg_scale(sd_txt2img_options *opt, const char *cfg_scale);
    STABLE_DIFFUSION_API void set_txt2img_size(sd_txt2img_options *opt, int width, int height);
    STABLE_DIFFUSION_API void set_txt2img_sample_method(sd_txt2img_options *opt, int sample_method);
    STABLE_DIFFUSION_API void set_txt2img_sample_steps(sd_txt2img_options *opt, int sample_steps);
    STABLE_DIFFUSION_API void set_txt2img_strength(sd_txt2img_options *opt, float strength);
    STABLE_DIFFUSION_API void set_txt2img_seed(sd_txt2img_options *opt, int64_t seed);
    //================================================================================

    //==============================sd_img2img_options===============================
    STABLE_DIFFUSION_API void set_img2img_init_img(sd_img2img_options *opt, const uint8_t *init_img);
    STABLE_DIFFUSION_API void set_img2img_prompt(sd_img2img_options *opt, const char *prompt);
    STABLE_DIFFUSION_API void set_img2img_negative_prompt(sd_img2img_options *opt, const char *negative_prompt);
    STABLE_DIFFUSION_API void set_img2img_cfg_scale(sd_img2img_options *opt, const char *cfg_scale);
    STABLE_DIFFUSION_API void set_img2img_size(sd_img2img_options *opt, int width, int height);
    STABLE_DIFFUSION_API void set_img2img_sample_method(sd_img2img_options *opt, int sample_method);
    STABLE_DIFFUSION_API void set_img2img_sample_steps(sd_img2img_options *opt, int sample_steps);
    STABLE_DIFFUSION_API void set_img2img_strength(sd_img2img_options *opt, float strength);
    STABLE_DIFFUSION_API void set_img2img_seed(sd_img2img_options *opt, int64_t seed);
    //================================================================================

    STABLE_DIFFUSION_API void *create_stable_diffusion(int n_threads, int vae_decode_only, int free_params_immediately, int rng_type);
    STABLE_DIFFUSION_API void destroy_stable_diffusion(void *sd);
    STABLE_DIFFUSION_API int load_from_file(void *sd, const char *file_path, int schedule);
    STABLE_DIFFUSION_API uint8_t *txt2img(void *sd, sd_txt2img_options *opt);
    STABLE_DIFFUSION_API uint8_t *img2img(void *sd, sd_img2img_options *opt);

    STABLE_DIFFUSION_API void set_sd_log_level(int level);
    STABLE_DIFFUSION_API const char *sd_get_system_info();

#ifdef __cplusplus
}
#endif

class StableDiffusionGGML;

class StableDiffusion
{
private:
    std::shared_ptr<StableDiffusionGGML> sd;

public:
    StableDiffusion(int n_threads = -1,
                    bool vae_decode_only = false,
                    bool free_params_immediately = false,
                    RNGType rng_type = STD_DEFAULT_RNG);
    bool load_from_file(const std::string &file_path, Schedule d = DEFAULT);
    std::vector<uint8_t> txt2img(
        const std::string &prompt,
        const std::string &negative_prompt,
        float cfg_scale,
        int width,
        int height,
        SampleMethod sample_method,
        int sample_steps,
        int64_t seed);
    std::vector<uint8_t> img2img(
        const std::vector<uint8_t> &init_img,
        const std::string &prompt,
        const std::string &negative_prompt,
        float cfg_scale,
        int width,
        int height,
        SampleMethod sample_method,
        int sample_steps,
        float strength,
        int64_t seed);
};

#endif // __STABLE_DIFFUSION_H__
