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

/* 随机数生成器 */
enum rng_type_t {
    STD_DEFAULT_RNG,
    /* CUDA RNG 能够高效地在 GPU 上生成大量的随机数，这对于并行化随机事件非常有用。
     * GPU 提供了比 CPU 更高的并行计算性能，因此在 GPU 上使用 CUDA RNG 可以显著加快涉及随机性的计算任务的速度。
     */
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

/**
 * 调度器（Scheduler）是一种控制噪声去除过程的策略，用于指导模型在生成图像时逐步去除噪声的过程。
 * 在生成阶段，Stable Diffusion 模型会从纯噪声开始，然后逐步减少噪声，最终得到一张清晰的图像。
 * 这一过程可以通过一系列离散的时间步长来模拟，每个时间步长对应一个噪声水平。Karras Scheduler 的关键在于如何选择这些时间步长。
 * KARRAS Scheduler 使用了一种非均匀的时间步长选择策略。这意味着在去噪过程中，某些步骤可能更长，而其他步骤可能更短。
 * 这种非均匀的步长选择可以更好地控制去噪过程，从而提高生成图像的质量。
 *
 */
enum schedule_t {
    DEFAULT,
    DISCRETE,
    KARRAS,
    AYS,
    N_SCHEDULES
};

/**
 * 张量中实际的数据量化类型.
 * SD_TYPE_Q4_0
 *
 * GGML_TYPE_Q4_0 特指一种4位量化的数据类型。这里的“Q4”意味着每个权重值被量化为4位（即可以存储16个不同的数值）。这种量化方式显著减少了模型的内存占用。
 * 4位量化：相比于全精度的32位浮点数，4位量化可以将模型的大小压缩到原来的1/8。这对于内存受限的设备非常有用。
 * 0：这里的"0"通常是指该量化方案的一个变体。在GGML中，可能会有多个针对同一量化位宽的不同实现版本。例如，可能存在GGML_TYPE_Q4_1，表示另一种4位量化的实现。
 */
// same as enum ggml_type
enum sd_type_t {
    SD_TYPE_F32     = 0,
    SD_TYPE_F16     = 1,
    SD_TYPE_Q4_0    = 2,
    SD_TYPE_Q4_1    = 3,
    // SD_TYPE_Q4_2 = 4, support has been removed
    // SD_TYPE_Q4_3 = 5, support has been removed
    SD_TYPE_Q5_0    = 6,
    SD_TYPE_Q5_1    = 7,
    SD_TYPE_Q8_0    = 8,
    SD_TYPE_Q8_1    = 9,
    SD_TYPE_Q2_K    = 10,
    SD_TYPE_Q3_K    = 11,
    SD_TYPE_Q4_K    = 12,
    SD_TYPE_Q5_K    = 13,
    SD_TYPE_Q6_K    = 14,
    SD_TYPE_Q8_K    = 15,
    SD_TYPE_IQ2_XXS = 16,
    SD_TYPE_IQ2_XS  = 17,
    SD_TYPE_IQ3_XXS = 18,
    SD_TYPE_IQ1_S   = 19,
    SD_TYPE_IQ4_NL  = 20,
    SD_TYPE_IQ3_S   = 21,
    SD_TYPE_IQ2_S   = 22,
    SD_TYPE_IQ4_XS  = 23,
    SD_TYPE_I8      = 24,
    SD_TYPE_I16     = 25,
    SD_TYPE_I32     = 26,
    SD_TYPE_I64     = 27,
    SD_TYPE_F64     = 28,
    SD_TYPE_IQ1_M   = 29,
    SD_TYPE_BF16    = 30,
    SD_TYPE_Q4_0_4_4 = 31,
    SD_TYPE_Q4_0_4_8 = 32,
    SD_TYPE_Q4_0_8_8 = 33,
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
typedef void (*sd_progress_cb_t)(int step, int steps, float time, void* data);

SD_API void sd_set_log_callback(sd_log_cb_t sd_log_cb, void* data);
SD_API void sd_set_progress_callback(sd_progress_cb_t cb, void* data);
SD_API int32_t get_num_physical_cores();
SD_API const char* sd_get_system_info();

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t channel; //  图像的颜色通道数。通常，对于RGB图像，channels 的值为 3；对于RGBA图像（包含透明度通道），channels 的值为 4。
    uint8_t* data; // 指向图像像素数据的指针。像素数据通常按照宽度优先的方式排列，即每一行的像素数据(R,G,B)或者(R,G,B,A)连续存储在内存中。
} sd_image_t;

// 内部是 StableDiffusionGGML* sd
typedef struct sd_ctx_t sd_ctx_t;

// 用来生成 sd_ctx
SD_API sd_ctx_t* new_sd_ctx(const char* model_path,
                            const char* vae_path,
                            const char* taesd_path,
                            const char* control_net_path_c_str,
                            const char* lora_model_dir,
                            const char* embed_dir_c_str,
                            const char* stacked_id_embed_dir_c_str,
                            bool vae_decode_only,
                            bool vae_tiling,
                            bool free_params_immediately,
                            int n_threads,
                            enum sd_type_t wtype,
                            enum rng_type_t rng_type,
                            enum schedule_t s,
                            bool keep_clip_on_cpu,
                            bool keep_control_net_cpu,
                            bool keep_vae_on_cpu);

SD_API void free_sd_ctx(sd_ctx_t* sd_ctx);

// 文本提示词 生 图片
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
                           int batch_count,
                           const sd_image_t* control_cond,
                           float control_strength,
                           float style_strength,
                           bool normalize_input,
                           const char* input_id_images_path);

// 图生图
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
                           int batch_count,
                           const sd_image_t* control_cond,
                           float control_strength,
                           float style_strength,
                           bool normalize_input,
                           const char* input_id_images_path);

// img2vid 功能通常是指将一系列图像文件转换成视频文件的功能
// 如果你的扩散模型是在一系列迭代过程中生成图像的，那么你可以将每个步骤的结果保存为图像，并使用 img2vid 将这些图像按顺序连接起来，以展示整个生成过程。
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
                           int64_t seed);

/**
 * 用于表示图像上采样（也称为图像放大或超分辨率重建）的上下文。这个上下文包含了进行图像上采样的所有必要信息和配置。
 */
typedef struct upscaler_ctx_t upscaler_ctx_t;

/**
 *
 * @param esrgan_path ESRGAN (Enhanced Super-Resolution Generative Adversarial Networks）模型文件路径.
 *                    ESRGAN 是一种深度学习模型，用于超分辨率图像重建，能够将低分辨率图像放大到高分辨率图像，并保持良好的视觉效果。
 * @param n_threads
 * @param wtype   这是一个枚举类型参数，指定了上采样器的工作类型。
 *                sd_type_t 枚举可能包含不同的上采样方法，例如：
 *                  SD_BICUBIC: 双三次插值
 *                  SD_BILINEAR: 双线性插值
 *                  SD_ESRGAN: 使用 ESRGAN 模型进行上采样
 *                  可能还有其他选项，具体取决于实现。
 * @return
 */
SD_API upscaler_ctx_t* new_upscaler_ctx(const char* esrgan_path,
                                        int n_threads,
                                        enum sd_type_t wtype);
SD_API void free_upscaler_ctx(upscaler_ctx_t* upscaler_ctx);

SD_API sd_image_t upscale(upscaler_ctx_t* upscaler_ctx, sd_image_t input_image, uint32_t upscale_factor);

/**
 * convert 函数用于将一个输入图像通过 VAE（Variational Autoencoder）模型进行转换，并将转换后的结果保存到指定的输出路径。
 * 这个函数通常用于图像的预处理或后处理阶段，尤其是在使用稳定扩散模型生成图像的过程中。
 * @param input_path 输入图像的文件路径
 * @param vae_path VAE 模型的文件路径,VAE 模型用于将输入图像转换为潜在空间表示，并从潜在空间重建图像。
 * @param output_path 输出图像的文件路径,输出图像通常是经过 VAE 转换后的图像。
 * @param output_type 输出图像的类型,SD_PNG,SD_JPEG,SD_TIFF
 * @return
 *  true: 成功转换图像。
 *  false: 转换失败，可能是因为输入文件不存在、模型文件无效等原因
 */
SD_API bool convert(const char* input_path, const char* vae_path, const char* output_path, sd_type_t output_type);

/**
 * 用于执行 Canny 边缘检测预处理的函数。Canny 边缘检测是一种广泛使用的边缘检测算法，它能够识别图像中的边界并突出显示这些边缘。
 * 此函数执行以下操作：
 * - 灰度化：如果输入的是彩色图像，首先需要将其转换为灰度图像。这是因为 Canny 边缘检测通常在灰度图像上进行。
 * - 高斯模糊：对灰度图像应用高斯滤波器以减少噪声。
 * - 计算梯度强度和方向：使用 Sobel 运算符来计算每个像素的梯度强度和方向。
 * - 非极大值抑制：消除不处于局部最大值的梯度值，以得到细线状的边缘。
 * - 双阈值和连接边缘：使用高低两个阈值来确定哪些边缘是真实的（强边缘），哪些可能是边缘的一部分（弱边缘）。通过连通性分析将弱边缘与强边缘连接起来。
 * - 反转图像（如果设置了 inverse 参数）：如果需要，反转输出图像的颜色。
 * @param img
 * @param width
 * @param height
 * @param high_threshold  Canny 边缘检测算法中的高阈值。这个阈值用于确定哪些边缘被认为是“强”边缘。
 * @param low_threshold   Canny 边缘检测算法中的低阈值。这个阈值用于确定哪些边缘被认为是“弱”边缘。
 * @param weak   “弱”边缘的像素值。通常是一个介于 0 和 255 之间的值，用来标记那些通过低阈值但未通过高阈值测试的像素。
 * @param strong “强”边缘的像素值。同样是一个介于 0 和 255 之间的值，用来标记那些通过高阈值测试的像素。
 * @param inverse 一个布尔标志，指示是否应该反转输出图像。如果为 true，则输出图像会是原图的反色版本；如果为 false，则输出是正常的边缘检测结果。
 * @return 返回一个指向处理后图像数据的指针。处理后的图像数据格式与输入相同，即一维数组，每个像素值依然是一个字节。
 */
SD_API uint8_t* preprocess_canny(uint8_t* img,
                                 int width,
                                 int height,
                                 float high_threshold,
                                 float low_threshold,
                                 float weak,
                                 float strong,
                                 bool inverse);

#ifdef __cplusplus
}
#endif

#endif  // __STABLE_DIFFUSION_H__
