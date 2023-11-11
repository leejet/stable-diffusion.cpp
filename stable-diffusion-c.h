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


	STABLE_DIFFUSION_API void* create_stable_diffusion(int n_threads,
		bool vae_decode_only,
		bool free_params_immediately,
		enum RNGType rng_type);

	STABLE_DIFFUSION_API void destroy_stable_diffusion(void* sd);

	STABLE_DIFFUSION_API bool load_from_file(void* sd, const char* file_path,enum Schedule schedule);

	STABLE_DIFFUSION_API uint8_t* txt2img(
	    void* sd,
	    const char* prompt;
        const char* negative_prompt;
	    float cfg_scale,
        int width,
        int height,
        enum SampleMethod sample_method,
        int sample_steps,
        int64_t seed,
        int64_t* output_size);

	STABLE_DIFFUSION_API uint8_t* img2img(
        void* sd,
        const uint8_t* init_img,
        const int64_t init_img_size,
        const char* prompt,
        const char* negative_prompt,
        float cfg_scale,
        int width,
        int height,
        enum SampleMethod sample_method,
        int sample_steps,
        float strength,
        int64_t seed
        int64_t* output_size);

	STABLE_DIFFUSION_API void set_stable_diffusion_log_level(enum SDLogLevel level);

	STABLE_DIFFUSION_API const char* get_stable_diffusion_system_info();

#ifdef __cplusplus
}
#endif

#endif //STABLE_DIFFUSION_CPP_STABLE_DIFFUSION_C_H
