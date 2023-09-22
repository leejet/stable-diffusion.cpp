#ifndef STABLE_DIFFUSION_CPP_STABLE_DIFFUSION_C_H
#define STABLE_DIFFUSION_CPP_STABLE_DIFFUSION_C_H

#include "stable-diffusion.h"

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

#include <stddef.h>
#include <stdint.h>
#include <string.h>

struct sd_txt2img_options;

struct sd_img2img_options;

// These methods are used in binding in other languages,golang, python,etc.

//==============================sd_txt2img_options===============================
STABLE_DIFFUSION_API sd_txt2img_options *new_sd_txt2img_options();

STABLE_DIFFUSION_API void set_txt2img_prompt(sd_txt2img_options *opt, const char *prompt);

STABLE_DIFFUSION_API void set_txt2img_negative_prompt(sd_txt2img_options *opt, const char *negative_prompt);

STABLE_DIFFUSION_API void set_txt2img_cfg_scale(sd_txt2img_options *opt, float cfg_scale);

STABLE_DIFFUSION_API void set_txt2img_size(sd_txt2img_options *opt, int width, int height);

STABLE_DIFFUSION_API void set_txt2img_sample_method(sd_txt2img_options *opt, const char *sample_method);

STABLE_DIFFUSION_API void set_txt2img_sample_steps(sd_txt2img_options *opt, int sample_steps);

STABLE_DIFFUSION_API void set_txt2img_seed(sd_txt2img_options *opt, int64_t seed);
//================================================================================

//==============================sd_img2img_options===============================
STABLE_DIFFUSION_API sd_img2img_options *new_sd_img2img_options();

STABLE_DIFFUSION_API void set_img2img_init_img(sd_img2img_options *opt, const uint8_t *init_img, int64_t size);

STABLE_DIFFUSION_API void set_img2img_prompt(sd_img2img_options *opt, const char *prompt);

STABLE_DIFFUSION_API void set_img2img_negative_prompt(sd_img2img_options *opt, const char *negative_prompt);

STABLE_DIFFUSION_API void set_img2img_cfg_scale(sd_img2img_options *opt, float cfg_scale);

STABLE_DIFFUSION_API void set_img2img_size(sd_img2img_options *opt, int width, int height);

STABLE_DIFFUSION_API void set_img2img_sample_method(sd_img2img_options *opt, const char *sample_method);

STABLE_DIFFUSION_API void set_img2img_sample_steps(sd_img2img_options *opt, int sample_steps);

STABLE_DIFFUSION_API void set_img2img_strength(sd_img2img_options *opt, float strength);

STABLE_DIFFUSION_API void set_img2img_seed(sd_img2img_options *opt, int64_t seed);
//================================================================================

STABLE_DIFFUSION_API void *create_stable_diffusion(int n_threads,
                                                   bool vae_decode_only,
                                                   bool free_params_immediately,
                                                   const char *rng_type);

STABLE_DIFFUSION_API void destroy_stable_diffusion(void *sd);

STABLE_DIFFUSION_API bool load_from_file(void *sd, const char *file_path, const char *schedule);

STABLE_DIFFUSION_API uint8_t *txt2img(void *sd, sd_txt2img_options *opt);

STABLE_DIFFUSION_API uint8_t *img2img(void *sd, sd_img2img_options *opt);

STABLE_DIFFUSION_API void set_stable_diffusion_log_level(const char *level);

STABLE_DIFFUSION_API const char *get_stable_diffusion_system_info();

#ifdef __cplusplus
}
#endif

#endif //STABLE_DIFFUSION_CPP_STABLE_DIFFUSION_C_H
