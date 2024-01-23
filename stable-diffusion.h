#ifndef __STABLE_DIFFUSION_H__
#define __STABLE_DIFFUSION_H__

#if defined(_WIN32) || defined(__CYGWIN__)
#ifndef SD_BUILD_SHARED_LIB
#define SD_API
#else
#ifdef SD_BUILD_DLL
#define SD_API __declspec(dllexport)
#else
#define SD_API __declspec(dllimport)
#endif
#endif
#else
#if __GNUC__ >= 4
#define SD_API __attribute__((visibility("default")))
#else
#define SD_API
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

enum rng_type_t {
    STD_DEFAULT_RNG,
    CUDA_RNG
};

enum sample_method_t {
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

enum schedule_t {
    DEFAULT,
    DISCRETE,
    KARRAS,
    N_SCHEDULES
};

// same as enum ggml_type
enum sd_type_t {
    SD_TYPE_F32  = 0,
    SD_TYPE_F16  = 1,
    SD_TYPE_Q4_0 = 2,
    SD_TYPE_Q4_1 = 3,
    // SD_TYPE_Q4_2 = 4, support has been removed
    // SD_TYPE_Q4_3 (5) support has been removed
    SD_TYPE_Q5_0 = 6,
    SD_TYPE_Q5_1 = 7,
    SD_TYPE_Q8_0 = 8,
    SD_TYPE_Q8_1 = 9,
    // k-quantizations
    SD_TYPE_Q2_K    = 10,
    SD_TYPE_Q3_K    = 11,
    SD_TYPE_Q4_K    = 12,
    SD_TYPE_Q5_K    = 13,
    SD_TYPE_Q6_K    = 14,
    SD_TYPE_Q8_K    = 15,
    SD_TYPE_IQ2_XXS = 16,
    SD_TYPE_I8,
    SD_TYPE_I16,
    SD_TYPE_I32,
    SD_TYPE_COUNT,
};

SD_API const char* sd_type_name(enum sd_type_t type);

enum sd_log_level_t {
    SD_LOG_DEBUG,
    SD_LOG_INFO,
    SD_LOG_WARN,
    SD_LOG_ERROR
};

typedef void (*sd_log_cb_t)(enum sd_log_level_t level, const char* text, void* data);

SD_API void sd_set_log_callback(sd_log_cb_t sd_log_cb, void* data);
SD_API int32_t get_num_physical_cores();
SD_API const char* sd_get_system_info();

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t channel;
    uint8_t* data;
} sd_image_t;

typedef struct sd_ctx_t sd_ctx_t;

SD_API sd_ctx_t* new_sd_ctx(int n_threads,
                            bool vae_decode_only,
                            bool free_params_immediately,
                            const char* lora_model_dir_c_str,
                            enum rng_type_t rng_type,
                            bool vae_tiling,
                            enum sd_type_t wtype,
                            enum schedule_t s,
                            bool init_backend_immediately = true);

SD_API void free_sd_ctx(sd_ctx_t* sd_ctx);

SD_API sd_image_t* txt2img(sd_ctx_t* sd_ctx,
                           const char* prompt,
                           const char* negative_prompt,
                           int clip_skip,
                           float cfg_scale,
                           int width,
                           int height,
                           enum sample_method_t sample_method,
                           int sample_steps,
                           int64_t seed,
                           int batch_count);

SD_API sd_image_t* img2img(sd_ctx_t* sd_ctx,
                           sd_image_t init_image,
                           const char* prompt,
                           const char* negative_prompt,
                           int clip_skip,
                           float cfg_scale,
                           int width,
                           int height,
                           enum sample_method_t sample_method,
                           int sample_steps,
                           float strength,
                           int64_t seed,
                           int batch_count);

typedef struct upscaler_ctx_t upscaler_ctx_t;

SD_API upscaler_ctx_t* new_upscaler_ctx(const char* esrgan_path,
                                        int n_threads,
                                        enum sd_type_t wtype);
SD_API void free_upscaler_ctx(upscaler_ctx_t* upscaler_ctx);

SD_API sd_image_t upscale(upscaler_ctx_t* upscaler_ctx, sd_image_t input_image, uint32_t upscale_factor);

SD_API void init_backend(sd_ctx_t* sd_ctx);

SD_API void set_options(sd_ctx_t* sd_ctx,
                        int n_threads,
                        bool vae_decode_only,
                        bool free_params_immediately,
                        const char* lora_model_dir,
                        rng_type_t rng_type,
                        bool vae_tiling,
                        sd_type_t wtype,
                        schedule_t schedule);

SD_API bool load_clip_from_file(sd_ctx_t* sd_ctx, const char* model_path, const char* prefix = "te.");

SD_API void free_clip_params(sd_ctx_t* sd_ctx);

SD_API bool load_unet_from_file(sd_ctx_t* sd_ctx, const char* model_path, const char* prefix = "unet.");

SD_API void free_unet_params(sd_ctx_t* sd_ctx);

SD_API bool load_vae_from_file(sd_ctx_t* sd_ctx, const char* model_path, const char* prefix = "vae.");

SD_API void free_vae_params(sd_ctx_t* sd_ctx);

SD_API bool load_taesd_from_file(sd_ctx_t* sd_ctx, const char* model_path);

SD_API void free_taesd_params(sd_ctx_t* sd_ctx);

SD_API bool load_diffusions_from_file(sd_ctx_t* sd_ctx, const char* model_path);

SD_API void free_diffusions_params(sd_ctx_t* sd_ctx);

SD_API bool convert(const char* input_path, const char* vae_path, const char* output_path, sd_type_t output_type);

#ifdef __cplusplus
}
#endif

#endif  // __STABLE_DIFFUSION_H__