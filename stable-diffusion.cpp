#include <assert.h>
#include <inttypes.h>
#include <stdarg.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "ggml/ggml-alloc.h"
#include "ggml/ggml-backend.h"
#include "ggml/ggml.h"

#ifdef SD_USE_CUBLAS
#include "ggml-cuda.h"
#endif

#ifdef SD_USE_METAL
#include "ggml-metal.h"
#endif

#include "model.h"
#include "rng.h"
#include "rng_philox.h"
#include "stable-diffusion.h"
#include "util.h"

#define EPS 1e-05f

#define UNET_GRAPH_SIZE 10240
#define LORA_GRAPH_SIZE 10240

#define TIMESTEPS 1000

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

std::string sd_get_system_info() {
    std::stringstream ss;
    ss << "System Info: \n";
    ss << "    BLAS = " << ggml_cpu_has_blas() << std::endl;
    ss << "    SSE3 = " << ggml_cpu_has_sse3() << std::endl;
    ss << "    AVX = " << ggml_cpu_has_avx() << std::endl;
    ss << "    AVX2 = " << ggml_cpu_has_avx2() << std::endl;
    ss << "    AVX512 = " << ggml_cpu_has_avx512() << std::endl;
    ss << "    AVX512_VBMI = " << ggml_cpu_has_avx512_vbmi() << std::endl;
    ss << "    AVX512_VNNI = " << ggml_cpu_has_avx512_vnni() << std::endl;
    ss << "    FMA = " << ggml_cpu_has_fma() << std::endl;
    ss << "    NEON = " << ggml_cpu_has_neon() << std::endl;
    ss << "    ARM_FMA = " << ggml_cpu_has_arm_fma() << std::endl;
    ss << "    F16C = " << ggml_cpu_has_f16c() << std::endl;
    ss << "    FP16_VA = " << ggml_cpu_has_fp16_va() << std::endl;
    ss << "    WASM_SIMD = " << ggml_cpu_has_wasm_simd() << std::endl;
    ss << "    VSX = " << ggml_cpu_has_vsx() << std::endl;
    return ss.str();
}

static void ggml_log_callback_default(ggml_log_level level, const char* text, void* user_data) {
    (void)level;
    (void)user_data;
    fputs(text, stderr);
    fflush(stderr);
}

void ggml_tensor_set_f32_randn(struct ggml_tensor* tensor, std::shared_ptr<RNG> rng) {
    uint32_t n                        = (uint32_t)ggml_nelements(tensor);
    std::vector<float> random_numbers = rng->randn(n);
    for (uint32_t i = 0; i < n; i++) {
        ggml_set_f32_1d(tensor, i, random_numbers[i]);
    }
}

void pretty_progress(int step, int steps, float time) {
    std::string progress = "  |";
    int max_progress     = 50;
    int32_t current      = (int32_t)(step * 1.f * max_progress / steps);
    for (int i = 0; i < 50; i++) {
        if (i > current) {
            progress += " ";
        } else if (i == current && i != max_progress - 1) {
            progress += ">";
        } else {
            progress += "=";
        }
    }
    progress += "|";
    printf(time > 1.0f ? "\r%s %i/%i - %.2fs/it" : "\r%s %i/%i - %.2fit/s",
           progress.c_str(), step, steps,
           time > 1.0f || time == 0 ? time : (1.0f / time));
    fflush(stdout);  // for linux
    if (step == steps) {
        printf("\n");
    }
}

// set tensor[i, j, k, l]
// set tensor[l]
// set tensor[k, l]
// set tensor[j, k, l]
void ggml_tensor_set_f32(struct ggml_tensor* tensor, float value, int l, int k = 0, int j = 0, int i = 0) {
    GGML_ASSERT(tensor->nb[0] == sizeof(float));
    *(float*)((char*)(tensor->data) + i * tensor->nb[3] + j * tensor->nb[2] + k * tensor->nb[1] + l * tensor->nb[0]) = value;
}

float ggml_tensor_get_f32(const ggml_tensor* tensor, int l, int k = 0, int j = 0, int i = 0) {
    // float value;
    // ggml_backend_tensor_get(tensor, &value, i * tensor->nb[3] + j * tensor->nb[2] + k * tensor->nb[1] + l * tensor->nb[0], sizeof(float));
    // return value;
    GGML_ASSERT(tensor->nb[0] == sizeof(float));
    return *(float*)((char*)(tensor->data) + i * tensor->nb[3] + j * tensor->nb[2] + k * tensor->nb[1] + l * tensor->nb[0]);
}

ggml_fp16_t ggml_tensor_get_f16(const ggml_tensor* tensor, int l, int k = 0, int j = 0, int i = 0) {
    GGML_ASSERT(tensor->nb[0] == sizeof(ggml_fp16_t));
    return *(ggml_fp16_t*)((char*)(tensor->data) + i * tensor->nb[3] + j * tensor->nb[2] + k * tensor->nb[1] + l * tensor->nb[0]);
}

void print_ggml_tensor(struct ggml_tensor* tensor, bool shape_only = false) {
    printf("shape(%zu, %zu, %zu, %zu)\n", tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    fflush(stdout);
    if (shape_only) {
        return;
    }
    int range = 3;
    for (int i = 0; i < tensor->ne[3]; i++) {
        if (i >= range && i + range < tensor->ne[3]) {
            continue;
        }
        for (int j = 0; j < tensor->ne[2]; j++) {
            if (j >= range && j + range < tensor->ne[2]) {
                continue;
            }
            for (int k = 0; k < tensor->ne[1]; k++) {
                if (k >= range && k + range < tensor->ne[1]) {
                    continue;
                }
                for (int l = 0; l < tensor->ne[0]; l++) {
                    if (l >= range && l + range < tensor->ne[0]) {
                        continue;
                    }
                    if (tensor->type == GGML_TYPE_F32) {
                        printf("  [%d, %d, %d, %d] = %f\n", i, j, k, l, ggml_tensor_get_f32(tensor, l, k, j, i));
                    } else if (tensor->type == GGML_TYPE_F16) {
                        printf("  [%d, %d, %d, %d] = %i\n", i, j, k, l, ggml_tensor_get_f16(tensor, l, k, j, i));
                    }
                    fflush(stdout);
                }
            }
        }
    }
}

ggml_tensor* load_tensor_from_file(ggml_context* ctx, const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERROR("failed to open '%s'", file_path.c_str());
        return NULL;
    }
    int32_t n_dims;
    int32_t length;
    int32_t ttype;

    file.read(reinterpret_cast<char*>(&n_dims), sizeof(n_dims));
    file.read(reinterpret_cast<char*>(&length), sizeof(length));
    file.read(reinterpret_cast<char*>(&ttype), sizeof(ttype));

    if (file.eof()) {
        LOG_ERROR("incomplete file '%s'", file_path.c_str());
        return NULL;
    }

    int32_t nelements = 1;
    int32_t ne[4]     = {1, 1, 1, 1};
    for (int i = 0; i < n_dims; ++i) {
        file.read(reinterpret_cast<char*>(&ne[i]), sizeof(ne[i]));
        nelements *= ne[i];
    }
    std::string name(length, 0);
    file.read(&name[0], length);
    ggml_tensor* tensor = ggml_new_tensor_4d(ctx, (ggml_type)ttype, ne[0], ne[1], ne[2], ne[3]);
    const size_t bpe    = ggml_type_size(ggml_type(ttype));
    file.read(reinterpret_cast<char*>(tensor->data), ggml_nbytes(tensor));
    return tensor;
}

// void save_tensor_to_file(const std::string& file_name, ggml_tensor* tensor, const std::string & name) {
//     std::string file_name_ = file_name + ".tensor";
//     std::string name_ = name;
//     std::ofstream file("./" + file_name_, std::ios::binary);
//     file.write(reinterpret_cast<char*>(&tensor->n_dims), sizeof(tensor->n_dims));
//     int len = (int)name_.size();
//     file.write(reinterpret_cast<char*>(&len), sizeof(len));
//     int ttype = (int)tensor->type;
//     file.write(reinterpret_cast<char*>(&ttype), sizeof(ttype));
//     for (int i = 0; i < tensor->n_dims; ++i) {
//         int ne_ = (int) tensor->ne[i];
//         file.write(reinterpret_cast<char*>(&ne_), sizeof(ne_));
//     }
//     file.write(&name_[0], len);
//     char* data = nullptr;
//     file.write((char*)tensor->data, ggml_nbytes(tensor));
//     file.close();
// }

void sd_fread(void* ptr, size_t size, size_t count, FILE* stream) {
    size_t ret = std::fread(ptr, size, count, stream);
    if (ret != count) {
        printf("Error: read from file failed");
        exit(1);
    }
}

void copy_ggml_tensor(struct ggml_tensor* dst, struct ggml_tensor* src) {
    if (dst->type == src->type) {
        dst->nb[0] = src->nb[0];
        dst->nb[1] = src->nb[1];
        dst->nb[2] = src->nb[2];
        dst->nb[3] = src->nb[3];

        memcpy(((char*)dst->data), ((char*)src->data), ggml_nbytes(dst));
        return;
    }
    struct ggml_init_params params;
    params.mem_size          = 10 * 1024 * 1024;  // for padding
    params.mem_buffer        = NULL;
    params.no_alloc          = false;
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        LOG_ERROR("ggml_init() failed");
        return;
    }
    ggml_tensor* final = ggml_cpy_inplace(ctx, src, dst);

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, final);
    ggml_graph_compute_with_ctx(ctx, graph, 1);
    ggml_free(ctx);
}

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

// Ref: https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/util.py#L151
void set_timestep_embedding(struct ggml_tensor* timesteps, struct ggml_tensor* embedding, int dim, int max_period = 10000) {
    // timesteps: [N,]
    // embedding: [dim, N]
    int half = dim / 2;
    std::vector<float> freqs(half);
    for (int i = 0; i < half; ++i) {
        freqs[i] = (float)std::exp(-std::log(max_period) * i / half);
    }
    for (int i = 0; i < timesteps->ne[0]; ++i) {
        for (int j = 0; j < half; ++j) {
            float arg = ggml_get_f32_1d(timesteps, i) * freqs[j];
            ggml_tensor_set_f32(embedding, std::cos(arg), j, i);
            ggml_tensor_set_f32(embedding, std::sin(arg), j + half, i);
        }
        if (dim % 2 != 0) {
            *(float*)((char*)embedding->data + i * embedding->nb[1] + dim * embedding->nb[0]) = 0;
        }
    }
}

struct ggml_tensor* new_timestep_embedding(struct ggml_context* ctx, struct ggml_allocr* allocr, struct ggml_tensor* timesteps, int dim, int max_period = 10000) {
    // timesteps: [N,]
    // embedding: [dim, N]
    int acutual_dim = dim;
    if (dim % 2 != 0) {
        acutual_dim = dim + 1;
    }
    struct ggml_tensor* embedding = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, acutual_dim, timesteps->ne[0]);
    if (allocr != NULL) {
        ggml_allocr_alloc(allocr, embedding);
    }
    if (allocr != NULL && !ggml_allocr_is_measure(allocr)) {
        set_timestep_embedding(timesteps, embedding, dim, max_period);
    }
    return embedding;
}

// SPECIAL OPERATIONS WITH TENSORS

uint8_t* sd_tensor_to_image(struct ggml_tensor* input) {
    int64_t width    = input->ne[0];
    int64_t height   = input->ne[1];
    int64_t channels = input->ne[2];
    GGML_ASSERT(channels == 3 && input->type == GGML_TYPE_F32);
    uint8_t* image_data = (uint8_t*)malloc(width * height * channels);
    for (int iy = 0; iy < height; iy++) {
        for (int ix = 0; ix < width; ix++) {
            for (int k = 0; k < channels; k++) {
                float value                                               = ggml_tensor_get_f32(input, ix, iy, k);
                *(image_data + iy * width * channels + ix * channels + k) = (uint8_t)(value * 255.0f);
            }
        }
    }
    return image_data;
}

void sd_image_to_tensor(const uint8_t* image_data,
                        struct ggml_tensor* output) {
    int64_t width    = output->ne[0];
    int64_t height   = output->ne[1];
    int64_t channels = output->ne[2];
    GGML_ASSERT(channels == 3 && output->type == GGML_TYPE_F32);
    for (int iy = 0; iy < height; iy++) {
        for (int ix = 0; ix < width; ix++) {
            for (int k = 0; k < channels; k++) {
                float value = *(image_data + iy * width * channels + ix * channels + k);
                ggml_tensor_set_f32(output, value / 255.0f, ix, iy, k);
            }
        }
    }
}

void ggml_split_tensor_2d(struct ggml_tensor* input,
                          struct ggml_tensor* output,
                          int x,
                          int y) {
    int64_t width    = output->ne[0];
    int64_t height   = output->ne[1];
    int64_t channels = output->ne[2];
    GGML_ASSERT(input->type == GGML_TYPE_F32 && output->type == GGML_TYPE_F32);
    for (int iy = 0; iy < height; iy++) {
        for (int ix = 0; ix < width; ix++) {
            for (int k = 0; k < channels; k++) {
                float value = ggml_tensor_get_f32(input, ix + x, iy + y, k);
                ggml_tensor_set_f32(output, value, ix, iy, k);
            }
        }
    }
}

void ggml_merge_tensor_2d(struct ggml_tensor* input,
                          struct ggml_tensor* output,
                          int x,
                          int y,
                          int overlap) {
    int64_t width    = input->ne[0];
    int64_t height   = input->ne[1];
    int64_t channels = input->ne[2];
    GGML_ASSERT(input->type == GGML_TYPE_F32 && output->type == GGML_TYPE_F32);
    for (int iy = 0; iy < height; iy++) {
        for (int ix = 0; ix < width; ix++) {
            for (int k = 0; k < channels; k++) {
                float new_value = ggml_tensor_get_f32(input, ix, iy, k);
                if (overlap > 0) {  // blend colors in overlapped area
                    float old_value = ggml_tensor_get_f32(output, x + ix, y + iy, k);
                    if (x > 0 && ix < overlap) {  // in overlapped horizontal
                        ggml_tensor_set_f32(output, old_value + (new_value - old_value) * (ix / (1.0f * overlap)), x + ix, y + iy, k);
                        continue;
                    }
                    if (y > 0 && iy < overlap) {  // in overlapped vertical
                        ggml_tensor_set_f32(output, old_value + (new_value - old_value) * (iy / (1.0f * overlap)), x + ix, y + iy, k);
                        continue;
                    }
                }
                ggml_tensor_set_f32(output, new_value, x + ix, y + iy, k);
            }
        }
    }
}

float ggml_tensor_mean(struct ggml_tensor* src) {
    float mean        = 0.0f;
    int64_t nelements = ggml_nelements(src);
    float* data       = (float*)src->data;
    for (int i = 0; i < nelements; i++) {
        mean += data[i] / nelements * 1.0f;
    }
    return mean;
}

// a = a+b
void ggml_tensor_add(struct ggml_tensor* a, struct ggml_tensor* b) {
    GGML_ASSERT(ggml_nelements(a) == ggml_nelements(b));
    int64_t nelements = ggml_nelements(a);
    float* vec_a      = (float*)a->data;
    float* vec_b      = (float*)b->data;
    for (int i = 0; i < nelements; i++) {
        vec_a[i] = vec_a[i] + vec_b[i];
    }
}

void ggml_tensor_scale(struct ggml_tensor* src, float scale) {
    int64_t nelements = ggml_nelements(src);
    float* data       = (float*)src->data;
    for (int i = 0; i < nelements; i++) {
        data[i] = data[i] * scale;
    }
}

void ggml_tensor_clamp(struct ggml_tensor* src, float min, float max) {
    int64_t nelements = ggml_nelements(src);
    float* data       = (float*)src->data;
    for (int i = 0; i < nelements; i++) {
        float val = data[i];
        data[i]   = val < min ? min : (val > max ? max : val);
    }
}

// convert values from [0, 1] to [-1, 1]
void ggml_tensor_scale_input(struct ggml_tensor* src) {
    int64_t nelements = ggml_nelements(src);
    float* data       = (float*)src->data;
    for (int i = 0; i < nelements; i++) {
        float val = data[i];
        data[i]   = val * 2.0f - 1.0f;
    }
}

// convert values from [-1, 1] to [0, 1]
void ggml_tensor_scale_output(struct ggml_tensor* src) {
    int64_t nelements = ggml_nelements(src);
    float* data       = (float*)src->data;
    for (int i = 0; i < nelements; i++) {
        float val = data[i];
        data[i]   = (val + 1.0f) * 0.5f;
    }
}

typedef std::function<void(ggml_tensor*, ggml_tensor*, bool)> on_tile_process;

// Tiling
void sd_tiling(ggml_tensor* input, ggml_tensor* output, const int scale, const int tile_size, const float tile_overlap_factor, on_tile_process on_processing) {
    int input_width   = input->ne[0];
    int input_height  = input->ne[1];
    int output_width  = output->ne[0];
    int output_height = output->ne[1];
    GGML_ASSERT(input_width % 2 == 0 && input_height % 2 == 0 && output_width % 2 == 0 && output_height % 2 == 0);  // should be multiple of 2

    int tile_overlap     = (int32_t)(tile_size * tile_overlap_factor);
    int non_tile_overlap = tile_size - tile_overlap;

    struct ggml_init_params params = {};
    params.mem_size += tile_size * tile_size * input->ne[2] * sizeof(float);                       // input chunk
    params.mem_size += (tile_size * scale) * (tile_size * scale) * output->ne[2] * sizeof(float);  // output chunk
    params.mem_size += 3 * ggml_tensor_overhead();
    params.mem_buffer = NULL;
    params.no_alloc   = false;

    LOG_DEBUG("tile work buffer size: %.2f MB", params.mem_size / 1024.f / 1024.f);

    // draft context
    struct ggml_context* tiles_ctx = ggml_init(params);
    if (!tiles_ctx) {
        LOG_ERROR("ggml_init() failed");
        return;
    }

    // tiling
    ggml_tensor* input_tile  = ggml_new_tensor_4d(tiles_ctx, GGML_TYPE_F32, tile_size, tile_size, input->ne[2], 1);
    ggml_tensor* output_tile = ggml_new_tensor_4d(tiles_ctx, GGML_TYPE_F32, tile_size * scale, tile_size * scale, output->ne[2], 1);
    on_processing(input_tile, NULL, true);
    int num_tiles = (input_width * input_height) / (non_tile_overlap * non_tile_overlap);
    LOG_INFO("processing %i tiles", num_tiles);
    pretty_progress(1, num_tiles, 0.0f);
    int tile_count = 1;
    bool last_y = false, last_x = false;
    float last_time = 0.0f;
    for (int y = 0; y < input_height && !last_y; y += non_tile_overlap) {
        if (y + tile_size >= input_height) {
            y      = input_height - tile_size;
            last_y = true;
        }
        for (int x = 0; x < input_width && !last_x; x += non_tile_overlap) {
            if (x + tile_size >= input_width) {
                x      = input_width - tile_size;
                last_x = true;
            }
            int64_t t1 = ggml_time_ms();
            ggml_split_tensor_2d(input, input_tile, x, y);
            on_processing(input_tile, output_tile, false);
            ggml_merge_tensor_2d(output_tile, output, x * scale, y * scale, tile_overlap * scale);
            int64_t t2 = ggml_time_ms();
            last_time  = (t2 - t1) / 1000.0f;
            pretty_progress(tile_count, num_tiles, last_time);
            tile_count++;
        }
        last_x = false;
    }
    if (tile_count < num_tiles) {
        pretty_progress(num_tiles, num_tiles, last_time);
    }
}

struct ggml_tensor* ggml_group_norm_32(struct ggml_context* ctx,
                                       struct ggml_tensor* a) {
    return ggml_group_norm(ctx, a, 32);
}

struct ggml_tensor* ggml_nn_linear(struct ggml_context* ctx,
                                   struct ggml_tensor* x,
                                   struct ggml_tensor* w,
                                   struct ggml_tensor* b) {
    x = ggml_mul_mat(ctx, w, x);
    x = ggml_add(ctx, x, b);
    return x;
}

// w: [OC，IC, KH, KW]
// x: [N, IC, IH, IW]
// b: [OC,]
// result: [N, OC, OH, OW]
struct ggml_tensor* ggml_nn_conv_2d(struct ggml_context* ctx,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* w,
                                    struct ggml_tensor* b,
                                    int s0 = 1,
                                    int s1 = 1,
                                    int p0 = 0,
                                    int p1 = 0,
                                    int d0 = 1,
                                    int d1 = 1) {
    x = ggml_conv_2d(ctx, w, x, s0, s1, p0, p1, d0, d1);
    if (b != NULL) {
        b = ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
        x = ggml_add(ctx, x, b);
    }
    return x;
}

struct ggml_tensor* ggml_nn_layer_norm(struct ggml_context* ctx,
                                       struct ggml_tensor* x,
                                       struct ggml_tensor* w,
                                       struct ggml_tensor* b,
                                       float eps = EPS) {
    x = ggml_norm(ctx, x, eps);
    x = ggml_mul(ctx, x, w);
    x = ggml_add(ctx, x, b);
    return x;
}

struct ggml_tensor* ggml_nn_group_norm(struct ggml_context* ctx,
                                       struct ggml_tensor* x,
                                       struct ggml_tensor* w,
                                       struct ggml_tensor* b,
                                       int num_groups = 32) {
    if (x->n_dims == 4) {
        w = ggml_reshape_4d(ctx, w, 1, 1, w->ne[0], 1);
        b = ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
    }

    x = ggml_group_norm(ctx, x, num_groups);
    x = ggml_mul(ctx, x, w);
    x = ggml_add(ctx, x, b);
    return x;
}

std::pair<std::unordered_map<std::string, float>, std::string> extract_and_remove_lora(std::string text) {
    std::regex re("<lora:([^:]+):([^>]+)>");
    std::smatch matches;
    std::unordered_map<std::string, float> filename2multiplier;

    while (std::regex_search(text, matches, re)) {
        std::string filename = matches[1].str();
        float multiplier     = std::stof(matches[2].str());

        text = std::regex_replace(text, re, "", std::regex_constants::format_first_only);

        if (multiplier == 0.f) {
            continue;
        }

        if (filename2multiplier.find(filename) == filename2multiplier.end()) {
            filename2multiplier[filename] = multiplier;
        } else {
            filename2multiplier[filename] += multiplier;
        }
    }

    return std::make_pair(filename2multiplier, text);
}

void ggml_backend_tensor_get_and_sync(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    #ifdef SD_USE_CUBLAS
        ggml_backend_tensor_get_async(backend, tensor, data, offset, size);
        ggml_backend_synchronize(backend);
    #else
        ggml_backend_tensor_get(tensor, data, offset, size);
    #endif
}

/*================================================== CLIPTokenizer ===================================================*/

const std::string UNK_TOKEN = "<|endoftext|>";
const std::string BOS_TOKEN = "<|startoftext|>";
const std::string EOS_TOKEN = "<|endoftext|>";
const std::string PAD_TOEKN = "<|endoftext|>";

const int UNK_TOKEN_ID = 49407;
const int BOS_TOKEN_ID = 49406;
const int EOS_TOKEN_ID = 49407;
const int PAD_TOKEN_ID = 49407;

std::vector<std::pair<int, std::u32string>> bytes_to_unicode() {
    std::vector<std::pair<int, std::u32string>> byte_unicode_pairs;
    std::set<int> byte_set;
    for (int b = static_cast<int>('!'); b <= static_cast<int>('~'); ++b) {
        byte_set.insert(b);
        byte_unicode_pairs.push_back(std::pair<int, std::u32string>(b, unicode_value_to_utf32(b)));
    }
    for (int b = 161; b <= 172; ++b) {
        byte_set.insert(b);
        byte_unicode_pairs.push_back(std::pair<int, std::u32string>(b, unicode_value_to_utf32(b)));
    }
    for (int b = 174; b <= 255; ++b) {
        byte_set.insert(b);
        byte_unicode_pairs.push_back(std::pair<int, std::u32string>(b, unicode_value_to_utf32(b)));
    }
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (byte_set.find(b) == byte_set.end()) {
            byte_unicode_pairs.push_back(std::pair<int, std::u32string>(b, unicode_value_to_utf32(n + 256)));
            ++n;
        }
    }
    // LOG_DEBUG("byte_unicode_pairs %d", byte_unicode_pairs.size());
    return byte_unicode_pairs;
}

// Ref: https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py
class CLIPTokenizer {
private:
    SDVersion version = VERSION_1_x;
    std::map<int, std::u32string> byte_encoder;
    std::map<std::u32string, int> encoder;
    std::map<std::pair<std::u32string, std::u32string>, int> bpe_ranks;
    std::regex pat;

    static std::string strip(const std::string& str) {
        std::string::size_type start = str.find_first_not_of(" \t\n\r\v\f");
        std::string::size_type end   = str.find_last_not_of(" \t\n\r\v\f");

        if (start == std::string::npos) {
            // String contains only whitespace characters
            return "";
        }

        return str.substr(start, end - start + 1);
    }

    static std::string whitespace_clean(std::string text) {
        text = std::regex_replace(text, std::regex(R"(\s+)"), " ");
        text = strip(text);
        return text;
    }

    static std::set<std::pair<std::u32string, std::u32string>> get_pairs(const std::vector<std::u32string>& subwords) {
        std::set<std::pair<std::u32string, std::u32string>> pairs;
        if (subwords.size() == 0) {
            return pairs;
        }
        std::u32string prev_subword = subwords[0];
        for (int i = 1; i < subwords.size(); i++) {
            std::u32string subword = subwords[i];
            std::pair<std::u32string, std::u32string> pair(prev_subword, subword);
            pairs.insert(pair);
            prev_subword = subword;
        }
        return pairs;
    }

public:
    CLIPTokenizer(SDVersion version = VERSION_1_x)
        : version(version) {}

    void load_from_merges(const std::string& merges_utf8_str) {
        auto byte_unicode_pairs = bytes_to_unicode();
        byte_encoder            = std::map<int, std::u32string>(byte_unicode_pairs.begin(), byte_unicode_pairs.end());
        // for (auto & pair: byte_unicode_pairs) {
        //     std::cout << pair.first << ": " << pair.second << std::endl;
        // }
        std::vector<std::u32string> merges;
        size_t start = 0;
        size_t pos;
        std::u32string merges_utf32_str = utf8_to_utf32(merges_utf8_str);
        while ((pos = merges_utf32_str.find('\n', start)) != std::string::npos) {
            merges.push_back(merges_utf32_str.substr(start, pos - start));
            start = pos + 1;
        }
        // LOG_DEBUG("merges size %llu", merges.size());
        GGML_ASSERT(merges.size() == 48895);
        merges = std::vector<std::u32string>(merges.begin() + 1, merges.end());
        std::vector<std::pair<std::u32string, std::u32string>> merge_pairs;
        for (const auto& merge : merges) {
            size_t space_pos = merge.find(' ');
            merge_pairs.emplace_back(merge.substr(0, space_pos), merge.substr(space_pos + 1));
            // LOG_DEBUG("%s", utf32_to_utf8(merge.substr(space_pos + 1)).c_str());
        }
        std::vector<std::u32string> vocab;
        for (const auto& pair : byte_unicode_pairs) {
            vocab.push_back(pair.second);
        }
        for (const auto& pair : byte_unicode_pairs) {
            vocab.push_back(pair.second + utf8_to_utf32("</w>"));
        }
        for (const auto& merge : merge_pairs) {
            vocab.push_back(merge.first + merge.second);
        }
        vocab.push_back(utf8_to_utf32("<|startoftext|>"));
        vocab.push_back(utf8_to_utf32("<|endoftext|>"));
        LOG_DEBUG("vocab size: %llu", vocab.size());
        int i = 0;
        for (const auto& token : vocab) {
            encoder[token] = i++;
        }

        int rank = 0;
        for (const auto& merge : merge_pairs) {
            bpe_ranks[merge] = rank++;
        }
    };

    std::u32string bpe(const std::u32string& token) {
        std::vector<std::u32string> word;

        for (int i = 0; i < token.size() - 1; i++) {
            word.emplace_back(1, token[i]);
        }
        word.push_back(token.substr(token.size() - 1) + utf8_to_utf32("</w>"));

        std::set<std::pair<std::u32string, std::u32string>> pairs = get_pairs(word);

        if (pairs.empty()) {
            return token + utf8_to_utf32("</w>");
        }

        while (true) {
            auto min_pair_iter = std::min_element(pairs.begin(),
                                                  pairs.end(),
                                                  [&](const std::pair<std::u32string, std::u32string>& a,
                                                      const std::pair<std::u32string, std::u32string>& b) {
                                                      if (bpe_ranks.find(a) == bpe_ranks.end()) {
                                                          return false;
                                                      } else if (bpe_ranks.find(b) == bpe_ranks.end()) {
                                                          return true;
                                                      }
                                                      return bpe_ranks.at(a) < bpe_ranks.at(b);
                                                  });

            const std::pair<std::u32string, std::u32string>& bigram = *min_pair_iter;

            if (bpe_ranks.find(bigram) == bpe_ranks.end()) {
                break;
            }

            std::u32string first  = bigram.first;
            std::u32string second = bigram.second;
            std::vector<std::u32string> new_word;
            int32_t i = 0;

            while (i < word.size()) {
                auto it = std::find(word.begin() + i, word.end(), first);
                if (it == word.end()) {
                    new_word.insert(new_word.end(), word.begin() + i, word.end());
                    break;
                }
                new_word.insert(new_word.end(), word.begin() + i, it);
                i = static_cast<int32_t>(std::distance(word.begin(), it));

                if (word[i] == first && i < static_cast<int32_t>(word.size()) - 1 && word[i + 1] == second) {
                    new_word.push_back(first + second);
                    i += 2;
                } else {
                    new_word.push_back(word[i]);
                    i += 1;
                }
            }

            word = new_word;

            if (word.size() == 1) {
                break;
            }
            pairs = get_pairs(word);
        }

        std::u32string result;
        for (int i = 0; i < word.size(); i++) {
            result += word[i];
            if (i != word.size() - 1) {
                result += utf8_to_utf32(" ");
            }
        }

        return result;
    }

    std::vector<int> tokenize(std::string text, size_t max_length = 0, bool padding = false) {
        std::vector<int32_t> tokens = encode(text);
        tokens.insert(tokens.begin(), BOS_TOKEN_ID);
        if (max_length > 0) {
            if (tokens.size() > max_length - 1) {
                tokens.resize(max_length - 1);
                tokens.push_back(EOS_TOKEN_ID);
            } else {
                tokens.push_back(EOS_TOKEN_ID);
                if (padding) {
                    int pad_token_id = PAD_TOKEN_ID;
                    if (version == VERSION_2_x) {
                        pad_token_id = 0;
                    }
                    tokens.insert(tokens.end(), max_length - tokens.size(), pad_token_id);
                }
            }
        }
        return tokens;
    }

    std::vector<int> encode(std::string text) {
        std::string original_text = text;
        std::vector<int32_t> bpe_tokens;
        text = whitespace_clean(text);
        std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return std::tolower(c); });

        std::regex pat(R"(<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[[:alpha:]]+|[[:digit:]]|[^[:space:][:alpha:][:digit:]]+)",
                       std::regex::icase);

        std::smatch matches;
        std::string str = text;
        std::vector<std::string> token_strs;
        while (std::regex_search(str, matches, pat)) {
            for (auto& token : matches) {
                std::string token_str = token.str();
                std::u32string utf32_token;
                for (int i = 0; i < token_str.length(); i++) {
                    char b = token_str[i];
                    utf32_token += byte_encoder[b];
                }
                auto bpe_strs = bpe(utf32_token);
                size_t start  = 0;
                size_t pos;
                while ((pos = bpe_strs.find(' ', start)) != std::u32string::npos) {
                    auto bpe_str = bpe_strs.substr(start, pos - start);
                    bpe_tokens.push_back(encoder[bpe_str]);
                    token_strs.push_back(utf32_to_utf8(bpe_str));

                    start = pos + 1;
                }
                auto bpe_str = bpe_strs.substr(start, bpe_strs.size() - start);
                bpe_tokens.push_back(encoder[bpe_str]);
                token_strs.push_back(utf32_to_utf8(bpe_str));
            }
            str = matches.suffix();
        }
        std::stringstream ss;
        ss << "[";
        for (auto token : token_strs) {
            ss << "\"" << token << "\", ";
        }
        ss << "]";
        LOG_DEBUG("split prompt \"%s\" to tokens %s", original_text.c_str(), ss.str().c_str());
        return bpe_tokens;
    }
};

// Ref: https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/cad87bf4e3e0b0a759afa94e933527c3123d59bc/modules/prompt_parser.py#L345
//
// Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
// Accepted tokens are:
//   (abc) - increases attention to abc by a multiplier of 1.1
//   (abc:3.12) - increases attention to abc by a multiplier of 3.12
//   [abc] - decreases attention to abc by a multiplier of 1.1
//   \( - literal character '('
//   \[ - literal character '['
//   \) - literal character ')'
//   \] - literal character ']'
//   \\ - literal character '\'
//   anything else - just text
//
// >>> parse_prompt_attention('normal text')
// [['normal text', 1.0]]
// >>> parse_prompt_attention('an (important) word')
// [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
// >>> parse_prompt_attention('(unbalanced')
// [['unbalanced', 1.1]]
// >>> parse_prompt_attention('\(literal\]')
// [['(literal]', 1.0]]
// >>> parse_prompt_attention('(unnecessary)(parens)')
// [['unnecessaryparens', 1.1]]
// >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
// [['a ', 1.0],
//  ['house', 1.5730000000000004],
//  [' ', 1.1],
//  ['on', 1.0],
//  [' a ', 1.1],
//  ['hill', 0.55],
//  [', sun, ', 1.1],
//  ['sky', 1.4641000000000006],
//  ['.', 1.1]]
std::vector<std::pair<std::string, float>> parse_prompt_attention(const std::string& text) {
    std::vector<std::pair<std::string, float>> res;
    std::vector<int> round_brackets;
    std::vector<int> square_brackets;

    float round_bracket_multiplier  = 1.1f;
    float square_bracket_multiplier = 1 / 1.1f;

    std::regex re_attention(R"(\\\(|\\\)|\\\[|\\\]|\\\\|\\|\(|\[|:([+-]?[.\d]+)\)|\)|\]|[^\\()\[\]:]+|:)");
    std::regex re_break(R"(\s*\bBREAK\b\s*)");

    auto multiply_range = [&](int start_position, float multiplier) {
        for (int p = start_position; p < res.size(); ++p) {
            res[p].second *= multiplier;
        }
    };

    std::smatch m;
    std::string remaining_text = text;

    while (std::regex_search(remaining_text, m, re_attention)) {
        std::string text   = m[0];
        std::string weight = m[1];

        if (text == "(") {
            round_brackets.push_back((int)res.size());
        } else if (text == "[") {
            square_brackets.push_back((int)res.size());
        } else if (!weight.empty()) {
            if (!round_brackets.empty()) {
                multiply_range(round_brackets.back(), std::stof(weight));
                round_brackets.pop_back();
            }
        } else if (text == ")" && !round_brackets.empty()) {
            multiply_range(round_brackets.back(), round_bracket_multiplier);
            round_brackets.pop_back();
        } else if (text == "]" && !square_brackets.empty()) {
            multiply_range(square_brackets.back(), square_bracket_multiplier);
            square_brackets.pop_back();
        } else if (text == "\\(") {
            res.push_back({text.substr(1), 1.0f});
        } else {
            res.push_back({text, 1.0f});
        }

        remaining_text = m.suffix();
    }

    for (int pos : round_brackets) {
        multiply_range(pos, round_bracket_multiplier);
    }

    for (int pos : square_brackets) {
        multiply_range(pos, square_bracket_multiplier);
    }

    if (res.empty()) {
        res.push_back({"", 1.0f});
    }

    int i = 0;
    while (i + 1 < res.size()) {
        if (res[i].second == res[i + 1].second) {
            res[i].first += res[i + 1].first;
            res.erase(res.begin() + i + 1);
        } else {
            ++i;
        }
    }

    return res;
}

/*================================================ FrozenCLIPEmbedder ================================================*/

struct ResidualAttentionBlock {
    int32_t n_head;
    int32_t d_model;
    int32_t hidden_size;  // n_head * d_model
    int32_t intermediate_size;

    // attention
    struct ggml_tensor* q_w;  // [hidden_size, hidden_size]
    struct ggml_tensor* q_b;  // [hidden_size, ]
    struct ggml_tensor* k_w;  // [hidden_size, hidden_size]
    struct ggml_tensor* k_b;  // [hidden_size, ]
    struct ggml_tensor* v_w;  // [hidden_size, hidden_size]
    struct ggml_tensor* v_b;  // [hidden_size, ]

    struct ggml_tensor* out_w;  // [hidden_size, hidden_size]
    struct ggml_tensor* out_b;  // [hidden_size, ]

    // layer norm 1
    struct ggml_tensor* ln1_w;  // [hidden_size, ]
    struct ggml_tensor* ln1_b;  // [hidden_size, ]

    // mlp
    struct ggml_tensor* fc1_w;  // [intermediate_size, hidden_size]
    struct ggml_tensor* fc1_b;  // [intermediate_size, ]

    struct ggml_tensor* fc2_w;  // [hidden_size, intermediate_size]
    struct ggml_tensor* fc2_b;  // [hidden_size, ]

    // layer norm 2
    struct ggml_tensor* ln2_w;  // [hidden_size, ]
    struct ggml_tensor* ln2_b;  // [hidden_size, ]

    struct ggml_tensor* attn_scale;  // [hidden_size, ]

    size_t calculate_mem_size(ggml_type wtype) {
        double mem_size = 0;
        mem_size += 4 * hidden_size * hidden_size * ggml_type_sizef(wtype);        // q_w/k_w/v_w/out_w
        mem_size += 8 * hidden_size * ggml_type_sizef(GGML_TYPE_F32);              // q_b/k_b/v_b/out_b/ln1_w/ln1_b/ln2_w/ln2_b
        mem_size += 2 * hidden_size * intermediate_size * ggml_type_sizef(wtype);  // fc1_w/fc2_w
        mem_size += intermediate_size * ggml_type_sizef(GGML_TYPE_F32);            // fc1_b
        mem_size += hidden_size * ggml_type_sizef(GGML_TYPE_F32);                  // fc2_b
        mem_size += ggml_type_sizef(GGML_TYPE_F32);                                // attn_scale
        return static_cast<size_t>(mem_size);
    }

    void init_params(struct ggml_context* ctx, ggml_allocr* alloc, ggml_type wtype) {
        ln1_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
        ln1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);

        q_w = ggml_new_tensor_2d(ctx, wtype, hidden_size, hidden_size);
        q_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
        k_w = ggml_new_tensor_2d(ctx, wtype, hidden_size, hidden_size);
        k_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
        v_w = ggml_new_tensor_2d(ctx, wtype, hidden_size, hidden_size);
        v_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);

        out_w = ggml_new_tensor_2d(ctx, wtype, hidden_size, hidden_size);
        out_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);

        fc1_w = ggml_new_tensor_2d(ctx, wtype, hidden_size, intermediate_size);
        fc1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, intermediate_size);

        fc2_w = ggml_new_tensor_2d(ctx, wtype, intermediate_size, hidden_size);
        fc2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);

        ln2_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
        ln2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);

        attn_scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        ggml_allocr_alloc(alloc, attn_scale);
        float scale = 1.0f / sqrt((float)d_model);
        ggml_backend_tensor_set(attn_scale, &scale, 0, sizeof(scale));
    }

    void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        tensors[prefix + "self_attn.q_proj.weight"]   = q_w;
        tensors[prefix + "self_attn.q_proj.bias"]     = q_b;
        tensors[prefix + "self_attn.k_proj.weight"]   = k_w;
        tensors[prefix + "self_attn.k_proj.bias"]     = k_b;
        tensors[prefix + "self_attn.v_proj.weight"]   = v_w;
        tensors[prefix + "self_attn.v_proj.bias"]     = v_b;
        tensors[prefix + "self_attn.out_proj.weight"] = out_w;
        tensors[prefix + "self_attn.out_proj.bias"]   = out_b;

        tensors[prefix + "layer_norm1.weight"] = ln1_w;
        tensors[prefix + "layer_norm1.bias"]   = ln1_b;

        tensors[prefix + "layer_norm2.weight"] = ln2_w;
        tensors[prefix + "layer_norm2.bias"]   = ln2_b;

        tensors[prefix + "mlp.fc1.weight"] = fc1_w;
        tensors[prefix + "mlp.fc1.bias"]   = fc1_b;

        tensors[prefix + "mlp.fc2.weight"] = fc2_w;
        tensors[prefix + "mlp.fc2.bias"]   = fc2_b;
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [N, n_token, hidden_size]
        int64_t N           = x->ne[2];
        int64_t n_token     = x->ne[1];
        int64_t hidden_size = n_head * d_model;

        struct ggml_tensor* r = x;

        // layer norm 1
        x = ggml_nn_layer_norm(ctx, x, ln1_w, ln1_b);
        // self-attention
        {
            struct ggml_tensor* q = ggml_nn_linear(ctx, x, q_w, q_b);
            q                     = ggml_scale_inplace(ctx, q, attn_scale);
            q                     = ggml_reshape_4d(ctx, q, d_model, n_head, n_token, N);   // [N, n_token, n_head, d_model]
            q                     = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));       // [N, n_head, n_token, d_model]
            q                     = ggml_reshape_3d(ctx, q, d_model, n_token, n_head * N);  // [N * n_head, n_token, d_model]

            struct ggml_tensor* k = ggml_nn_linear(ctx, x, k_w, k_b);
            k                     = ggml_reshape_4d(ctx, k, d_model, n_head, n_token, N);  // [N, n_token, n_head, d_model]
            k                     = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));      // [N, n_head, n_token, d_model]
            k                     = ggml_reshape_3d(ctx, k, d_model, n_token, n_head);     // [N * n_head, n_token, d_model]

            struct ggml_tensor* v = ggml_nn_linear(ctx, x, v_w, v_b);
            v                     = ggml_reshape_4d(ctx, v, d_model, n_head, n_token, N);   // [N, n_token, n_head, d_model]
            v                     = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));       // [N, n_head, d_model, n_token]
            v                     = ggml_reshape_3d(ctx, v, n_token, d_model, n_head * N);  // [N * n_head, d_model, n_token]

            struct ggml_tensor* kq = ggml_mul_mat(ctx, k, q);  // [N * n_head, n_token, n_token]

            kq = ggml_diag_mask_inf_inplace(ctx, kq, 0);
            kq = ggml_soft_max_inplace(ctx, kq);

            struct ggml_tensor* kqv = ggml_mul_mat(ctx, v, kq);  // [N * n_head, n_token, d_model]
            kqv                     = ggml_reshape_4d(ctx, kqv, d_model, n_token, n_head, N);
            kqv                     = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));  // [N, n_token, n_head, d_model]

            x = ggml_reshape_2d(ctx, kqv, d_model * n_head, n_token * N);  // // [N * n_token, d_model * n_head]
        }

        // attention output
        x = ggml_nn_linear(ctx, x, out_w, out_b);

        // residual
        x = ggml_add(ctx, x, r);
        r = x;

        // layer norm 2
        x = ggml_nn_layer_norm(ctx, x, ln2_w, ln2_b);

        // mlp
        x = ggml_nn_linear(ctx, x, fc1_w, fc1_b);

        if (hidden_size == 1024 || hidden_size == 1280) {  // SD 2.x
            x = ggml_gelu_inplace(ctx, x);
        } else {  // SD 1.x
            x = ggml_gelu_quick_inplace(ctx, x);
        }

        x = ggml_nn_linear(ctx, x, fc2_w, fc2_b);

        // residual 2
        x = ggml_add(ctx, x, r);
        return x;
    }
};

// OPENAI_CLIP_VIT_L_14: https://huggingface.co/openai/clip-vit-large-patch14/blob/main/config.json
// OPEN_CLIP_VIT_H_14: https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/blob/main/config.json
// OPEN_CLIP_VIT_BIGG_14: https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k/blob/main/config.json (CLIPTextModelWithProjection)
// SDXL CLIPModel
// CLIPTextModelWithProjection seems optional

enum CLIPVersion {
    OPENAI_CLIP_VIT_L_14,   // SD 1.x and SDXL
    OPEN_CLIP_VIT_H_14,     // SD 2.x
    OPEN_CLIP_VIT_BIGG_14,  // SDXL
};

struct CLIPTextModel {
    CLIPVersion version = OPENAI_CLIP_VIT_L_14;
    // network hparams
    int32_t vocab_size              = 49408;
    int32_t max_position_embeddings = 77;
    int32_t hidden_size             = 768;   // 1024 for OPEN_CLIP_VIT_H_14
    int32_t intermediate_size       = 3072;  // 4096 for OPEN_CLIP_VIT_H_14
    int32_t n_head                  = 12;    // num_attention_heads, 16 for OPEN_CLIP_VIT_H_14
    int32_t num_hidden_layers       = 12;    // 24 for OPEN_CLIP_VIT_H_14
    int32_t layer_idx               = 11;
    int32_t projection_dim          = 1280;  // only for OPEN_CLIP_VIT_BIGG_14
    bool with_final_ln              = true;

    // embeddings
    struct ggml_tensor* position_ids;
    struct ggml_tensor* token_embed_weight;
    struct ggml_tensor* position_embed_weight;

    // transformer
    std::vector<ResidualAttentionBlock> resblocks;
    struct ggml_tensor* final_ln_w;
    struct ggml_tensor* final_ln_b;

    struct ggml_tensor* text_projection;

    CLIPTextModel(CLIPVersion version = OPENAI_CLIP_VIT_L_14,
                  int clip_skip       = 1,
                  bool with_final_ln  = true)
        : version(version), with_final_ln(with_final_ln) {
        if (version == OPEN_CLIP_VIT_H_14) {
            hidden_size       = 1024;
            intermediate_size = 4096;
            n_head            = 16;
            num_hidden_layers = 24;
        } else if (version == OPEN_CLIP_VIT_BIGG_14) {  // CLIPTextModelWithProjection
            hidden_size       = 1280;
            intermediate_size = 5120;
            n_head            = 20;
            num_hidden_layers = 32;
        }
        layer_idx = num_hidden_layers - clip_skip;
        resblocks.resize(num_hidden_layers);
        set_resblocks_hp_params();
    }

    void set_resblocks_hp_params() {
        int d_model = hidden_size / n_head;  // 64 / SDXL is 40 for CLIPTextModelWithProjection
        for (int i = 0; i < num_hidden_layers; i++) {
            resblocks[i].d_model           = d_model;
            resblocks[i].n_head            = n_head;
            resblocks[i].hidden_size       = hidden_size;
            resblocks[i].intermediate_size = intermediate_size;
        }
    }

    size_t calculate_mem_size(ggml_type wtype) {
        double mem_size = 0;
        mem_size += hidden_size * max_position_embeddings * ggml_type_sizef(GGML_TYPE_I32);  // position_ids
        mem_size += hidden_size * vocab_size * ggml_type_sizef(wtype);                       // token_embed_weight
        mem_size += hidden_size * max_position_embeddings * ggml_type_sizef(wtype);          // position_embed_weight
        for (int i = 0; i < num_hidden_layers; i++) {
            mem_size += resblocks[i].calculate_mem_size(wtype);
        }
        mem_size += 2 * hidden_size * ggml_type_sizef(GGML_TYPE_F32);  // final_ln_w/b
        if (version == OPEN_CLIP_VIT_BIGG_14) {
            mem_size += hidden_size * projection_dim * ggml_type_sizef(GGML_TYPE_F32);  // text_projection
        }
        return static_cast<size_t>(mem_size);
    }

    void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        tensors[prefix + "embeddings.token_embedding.weight"]    = token_embed_weight;
        tensors[prefix + "embeddings.position_embedding.weight"] = position_embed_weight;
        tensors[prefix + "final_layer_norm.weight"]              = final_ln_w;
        tensors[prefix + "final_layer_norm.bias"]                = final_ln_b;
        for (int i = 0; i < num_hidden_layers; i++) {
            std::string name = prefix + "encoder.layers." + std::to_string(i) + ".";
            resblocks[i].map_by_name(tensors, prefix + "encoder.layers." + std::to_string(i) + ".");
        }
        if (version == OPEN_CLIP_VIT_BIGG_14) {
            tensors[prefix + "text_projection"] = text_projection;
        }
    }

    struct ggml_tensor* forward(struct ggml_context* ctx0, struct ggml_tensor* input_ids, uint32_t max_token_idx = 0, bool return_pooled = false) {
        // input_ids: [N, n_token]
        GGML_ASSERT(input_ids->ne[0] <= position_ids->ne[0]);

        // token_embedding + position_embedding
        struct ggml_tensor* x;
        x = ggml_add(ctx0,
                     ggml_get_rows(ctx0, token_embed_weight, input_ids),
                     ggml_get_rows(ctx0,
                                   position_embed_weight,
                                   ggml_view_1d(ctx0, position_ids, input_ids->ne[0], 0)));  // [N, n_token, hidden_size]

        // transformer
        for (int i = 0; i < num_hidden_layers; i++) {
            if (!return_pooled && i == layer_idx + 1) {
                // LOG_DEBUG("layer %d", i);
                break;
            }
            x = resblocks[i].forward(ctx0, x);  // [N, n_token, hidden_size]
        }

        // final layer norm
        if (return_pooled || with_final_ln) {
            x = ggml_nn_layer_norm(ctx0, x, final_ln_w, final_ln_b);
        }

        if (return_pooled) {
            // ggml_tensor* idx = ggml_argmax(ctx0, input_ids);
            // ggml_tensor* pooled = ggml_get_rows(ctx0, x, idx);
            // LOG_DEBUG("max_token_idx: %u %u", max_token_idx, x->nb[1]);
            ggml_tensor* pooled = ggml_view_1d(ctx0, x, hidden_size, x->nb[1] * max_token_idx);
            pooled              = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, text_projection)), pooled);
            return pooled;
        }

        return x;  // [N, n_token, hidden_size]
    }

    void alloc_params(ggml_context* ctx, ggml_backend_t backend, ggml_type wtype, ggml_allocr* alloc) {
        position_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, max_position_embeddings);

        token_embed_weight = ggml_new_tensor_2d(ctx, wtype, hidden_size, vocab_size);

        position_embed_weight = ggml_new_tensor_2d(ctx, wtype, hidden_size, max_position_embeddings);

        for (int i = 0; i < num_hidden_layers; i++) {
            resblocks[i].init_params(ctx, alloc, wtype);
        }

        final_ln_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);

        final_ln_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);

        if (version == OPEN_CLIP_VIT_BIGG_14) {
            text_projection = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, projection_dim, hidden_size);
        }

        // alloc all tensors linked to this context
        for (struct ggml_tensor* t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->data == NULL) {
                ggml_allocr_alloc(alloc, t);
            }
        }

        if (ggml_backend_is_cpu(backend)) {
            for (int i = 0; i < max_position_embeddings; i++) {
                ggml_set_i32_1d(position_ids, i, i);
            }
        } else {
            std::vector<int> pos_temp;
            for (int i = 0; i < max_position_embeddings; i++) {
                pos_temp.push_back(i);
            }
            ggml_backend_tensor_set(position_ids, pos_temp.data(), 0, ggml_nbytes(position_ids));
        }
    }
};

// ldm.modules.encoders.modules.FrozenCLIPEmbedder
struct FrozenCLIPEmbedder {
    CLIPTokenizer tokenizer;
    CLIPTextModel text_model;

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_allocr* allocr, const std::string& prompt) {
        std::vector<int32_t> tokens   = tokenizer.tokenize(prompt, text_model.max_position_embeddings, true);
        struct ggml_tensor* input_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tokens.size());
        memcpy(input_ids->data, tokens.data(), tokens.size() * ggml_element_size(input_ids));
        struct ggml_tensor* hidden_states = text_model.forward(ctx, input_ids);
        return hidden_states;
    }
};

// Ref: https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/cad87bf4e3e0b0a759afa94e933527c3123d59bc/modules/sd_hijack_clip.py#L283
struct FrozenCLIPEmbedderWithCustomWords {
    SDVersion version = VERSION_1_x;
    CLIPTokenizer tokenizer;
    CLIPTextModel text_model;
    CLIPTextModel text_model2;

    // context and memory buffers
    struct ggml_context* ctx;
    ggml_backend_buffer_t params_buffer;
    ggml_backend_buffer_t compute_buffer;  // for compute
    struct ggml_allocr* compute_alloc = NULL;
    size_t compute_memory_buffer_size = -1;

    size_t memory_buffer_size = 0;
    ggml_type wtype;
    ggml_backend_t backend           = NULL;
    ggml_tensor* hidden_state_output = NULL;
    ggml_tensor* pooled_output       = NULL;

    FrozenCLIPEmbedderWithCustomWords(SDVersion version = VERSION_1_x, int clip_skip = -1)
        : version(version), tokenizer(version) {
        if (clip_skip <= 0) {
            clip_skip = 1;
            if (version == VERSION_2_x || version == VERSION_XL) {
                clip_skip = 2;
            }
        }
        if (version == VERSION_1_x) {
            text_model = CLIPTextModel(OPENAI_CLIP_VIT_L_14, clip_skip);
        } else if (version == VERSION_2_x) {
            text_model = CLIPTextModel(OPEN_CLIP_VIT_H_14, clip_skip);
        } else if (version == VERSION_XL) {
            text_model  = CLIPTextModel(OPENAI_CLIP_VIT_L_14, clip_skip, false);
            text_model2 = CLIPTextModel(OPEN_CLIP_VIT_BIGG_14, clip_skip, false);
        }
    }

    size_t calculate_mem_size() {
        size_t mem_size = text_model.calculate_mem_size(wtype);
        if (version == VERSION_XL) {
            mem_size += text_model2.calculate_mem_size(wtype);
        }
        return mem_size;
    }

    void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        text_model.map_by_name(tensors, prefix + "transformer.text_model.");
        if (version == VERSION_XL) {
            text_model2.map_by_name(tensors, prefix + "1.transformer.text_model.");
        }
    }

    struct ggml_tensor* forward(struct ggml_context* ctx0, struct ggml_tensor* input_ids, struct ggml_tensor* input_ids2, uint32_t max_token_idx = 0, bool return_pooled = false) {
        if (return_pooled) {
            return text_model2.forward(ctx0, input_ids2, max_token_idx, return_pooled);
        }
        auto hidden_states = text_model.forward(ctx0, input_ids);  // [N, n_token, hidden_size]
        // LOG_DEBUG("hidden_states: %d %d %d %d %d", hidden_states->n_dims, hidden_states->ne[0], hidden_states->ne[1], hidden_states->ne[2], hidden_states->ne[3]);
        if (version == VERSION_XL) {
            hidden_states = ggml_reshape_4d(ctx0,
                                            hidden_states,
                                            hidden_states->ne[0],
                                            hidden_states->ne[1],
                                            hidden_states->ne[2],
                                            hidden_states->ne[3]);
            hidden_states = ggml_cont(ctx0, ggml_permute(ctx0, hidden_states, 2, 0, 1, 3));

            auto hidden_states2 = text_model2.forward(ctx0, input_ids2);  // [N, n_token, hidden_size2]
            hidden_states2      = ggml_reshape_4d(ctx0,
                                                  hidden_states2,
                                                  hidden_states2->ne[0],
                                                  hidden_states2->ne[1],
                                                  hidden_states2->ne[2],
                                                  hidden_states2->ne[3]);
            hidden_states2      = ggml_cont(ctx0, ggml_permute(ctx0, hidden_states2, 2, 0, 1, 3));

            hidden_states = ggml_concat(ctx0, hidden_states, hidden_states2);  // [N, n_token, hidden_size + hidden_size2]

            hidden_states = ggml_cont(ctx0, ggml_permute(ctx0, hidden_states, 1, 2, 0, 3));
        }
        // LOG_DEBUG("hidden_states: %d %d %d %d", hidden_states->ne[0], hidden_states->ne[1], hidden_states->ne[2], hidden_states->ne[3]);
        return hidden_states;
    }

    std::pair<std::vector<int>, std::vector<float>> tokenize(std::string text,
                                                             bool padding = false) {
        return tokenize(text, text_model.max_position_embeddings, padding);
    }

    std::pair<std::vector<int>, std::vector<float>> tokenize(std::string text,
                                                             size_t max_length = 0,
                                                             bool padding      = false) {
        auto parsed_attention = parse_prompt_attention(text);

        {
            std::stringstream ss;
            ss << "[";
            for (const auto& item : parsed_attention) {
                ss << "['" << item.first << "', " << item.second << "], ";
            }
            ss << "]";
            LOG_DEBUG("parse '%s' to %s", text.c_str(), ss.str().c_str());
        }

        std::vector<int> tokens;
        std::vector<float> weights;
        for (const auto& item : parsed_attention) {
            const std::string& curr_text = item.first;
            float curr_weight            = item.second;
            std::vector<int> curr_tokens = tokenizer.encode(curr_text);
            tokens.insert(tokens.end(), curr_tokens.begin(), curr_tokens.end());
            weights.insert(weights.end(), curr_tokens.size(), curr_weight);
        }
        tokens.insert(tokens.begin(), BOS_TOKEN_ID);
        weights.insert(weights.begin(), 1.0);

        if (max_length > 0) {
            if (tokens.size() > max_length - 1) {
                tokens.resize(max_length - 1);
                weights.resize(max_length - 1);
                tokens.push_back(EOS_TOKEN_ID);
                weights.push_back(1.0);
            } else {
                tokens.push_back(EOS_TOKEN_ID);
                weights.push_back(1.0);
                if (padding) {
                    int pad_token_id = PAD_TOKEN_ID;
                    if (version == VERSION_2_x) {
                        pad_token_id = 0;
                    }
                    tokens.insert(tokens.end(), max_length - tokens.size(), pad_token_id);
                    weights.insert(weights.end(), max_length - weights.size(), 1.0);
                }
            }
        }

        // for (int i = 0; i < tokens.size(); i++) {
        //     std::cout << tokens[i] << ":" << weights[i] << ", ";
        // }
        // std::cout << std::endl;

        return {tokens, weights};
    }

    bool initialize(ggml_backend_t backend_, ggml_type wtype_) {
        backend            = backend_;
        wtype              = wtype_;
        memory_buffer_size = 1 * 1024 * 1024;  // 1 MB, for padding
        memory_buffer_size += calculate_mem_size();

        int num_tensors = (3 + 2 + 37 * text_model.num_hidden_layers);
        if (version == VERSION_XL) {
            num_tensors += (3 + 2 + 37 * text_model2.num_hidden_layers);
        }
        LOG_DEBUG("clip params backend buffer size = % 6.2f MB (%i tensors)", memory_buffer_size / (1024.0 * 1024.0), num_tensors);

        struct ggml_init_params params;
        params.mem_size   = static_cast<size_t>(num_tensors * ggml_tensor_overhead());
        params.mem_buffer = NULL;
        params.no_alloc   = true;

        ctx = ggml_init(params);
        if (!ctx) {
            LOG_ERROR("ggml_init() failed");
            return false;
        }
        params_buffer = ggml_backend_alloc_buffer(backend, memory_buffer_size);
        return true;
    }

    void destroy() {
        if (ctx != NULL) {
            ggml_free(ctx);
            ctx = NULL;
        }

        if (params_buffer != NULL) {
            ggml_backend_buffer_free(params_buffer);
            params_buffer = NULL;
        }
    }

    void alloc_params() {
        ggml_allocr* alloc = ggml_allocr_new_from_buffer(params_buffer);
        text_model.alloc_params(ctx, backend, wtype, alloc);
        if (version == VERSION_XL) {
            text_model2.alloc_params(ctx, backend, wtype, alloc);
        }
        ggml_allocr_free(alloc);
    }

    struct ggml_cgraph* build_graph(struct ggml_allocr* allocr, std::vector<int> tokens, bool return_pooled = false) {
        // since we are using ggml-alloc, this buffer only needs enough space to hold the ggml_tensor and ggml_cgraph structs, but not the tensor data
        static size_t buf_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
        static std::vector<uint8_t> buf(buf_size);

        struct ggml_init_params params = {
            /*.mem_size   =*/buf_size,
            /*.mem_buffer =*/buf.data(),
            /*.no_alloc   =*/true,  // the tensors will be allocated later by ggml_allocr_alloc_graph()
        };

        struct ggml_context* ctx0 = ggml_init(params);

        struct ggml_cgraph* gf = ggml_new_graph(ctx0);

        struct ggml_tensor* input_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, tokens.size());
        ggml_allocr_alloc(allocr, input_ids);

        if (!ggml_allocr_is_measure(allocr)) {
            ggml_backend_tensor_set(input_ids, tokens.data(), 0, tokens.size() * ggml_element_size(input_ids));
        }

        struct ggml_tensor* input_ids2 = NULL;
        size_t max_token_idx           = 0;
        if (version == VERSION_XL) {
            input_ids2 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, tokens.size());
            ggml_allocr_alloc(allocr, input_ids2);

            auto it = std::find(tokens.begin(), tokens.end(), EOS_TOKEN_ID);
            if (it != tokens.end()) {
                std::fill(std::next(it), tokens.end(), 0);
            }

            max_token_idx = std::min<size_t>(std::distance(tokens.begin(), it), tokens.size() - 1);

            // for (int i = 0; i < tokens.size(); i++) {
            //     printf("%d ", tokens[i]);
            // }
            // printf("\n");

            if (!ggml_allocr_is_measure(allocr)) {
                ggml_backend_tensor_set(input_ids2, tokens.data(), 0, tokens.size() * ggml_element_size(input_ids2));
            }
        }

        struct ggml_tensor* hidden_states = forward(ctx0, input_ids, input_ids2, max_token_idx, return_pooled);

        ggml_build_forward_expand(gf, hidden_states);
        ggml_free(ctx0);

        return gf;
    }

    void begin(ggml_context* work_ctx, int max_tokens) {
        if (hidden_state_output == NULL) {
            size_t total_hidden_size = text_model.hidden_size;
            if (version == VERSION_XL) {
                total_hidden_size += text_model2.hidden_size;
            }
            hidden_state_output = ggml_new_tensor_2d(work_ctx, GGML_TYPE_F32, total_hidden_size, text_model.max_position_embeddings);
            pooled_output       = ggml_new_tensor_1d(work_ctx, GGML_TYPE_F32, text_model2.projection_dim);
        }
        // calculate the amount of memory required
        if (compute_memory_buffer_size == -1) {
            compute_alloc = ggml_allocr_new_measure_from_backend(backend);

            bool return_pooled = false;
            if (version == VERSION_XL) {
                return_pooled = true;
            }
            struct ggml_cgraph* gf = build_graph(compute_alloc, std::vector<int>(max_tokens), return_pooled);
            // compute the required memory
            compute_memory_buffer_size = ggml_allocr_alloc_graph(compute_alloc, gf) + 1024 * 1024;

            // recreate the allocator with the required memory
            ggml_allocr_free(compute_alloc);

            LOG_DEBUG("learned condition compute buffer size: %.2f MB", compute_memory_buffer_size / 1024.0 / 1024.0);
        }
        compute_buffer = ggml_backend_alloc_buffer(backend, compute_memory_buffer_size);
        compute_alloc  = ggml_allocr_new_from_buffer(compute_buffer);
    }

    std::pair<struct ggml_tensor*, struct ggml_tensor*> compute(const int n_threads, std::vector<int> tokens) {
        struct ggml_cgraph* gf = build_graph(compute_alloc, tokens);

        ggml_allocr_alloc_graph(compute_alloc, gf);

        if (ggml_backend_is_cpu(backend)) {
            ggml_backend_cpu_set_n_threads(backend, n_threads);
        }

#ifdef SD_USE_METAL
        if (ggml_backend_is_metal(backend)) {
            ggml_backend_metal_set_n_cb(backend, n_threads);
        }
#endif

        ggml_backend_graph_compute(backend, gf);

#ifdef GGML_PERF
        ggml_graph_print(gf);
#endif
        ggml_backend_tensor_get(gf->nodes[gf->n_nodes - 1], hidden_state_output->data, 0, ggml_nbytes(hidden_state_output));

        if (version == VERSION_XL) {
            struct ggml_cgraph* gf = build_graph(compute_alloc, tokens, true);

            ggml_allocr_alloc_graph(compute_alloc, gf);

            if (ggml_backend_is_cpu(backend)) {
                ggml_backend_cpu_set_n_threads(backend, n_threads);
            }

            ggml_backend_graph_compute(backend, gf);

#ifdef GGML_PERF
            ggml_graph_print(gf);
#endif
            ggml_backend_tensor_get(gf->nodes[gf->n_nodes - 1], pooled_output->data, 0, ggml_nbytes(pooled_output));
            return {hidden_state_output, pooled_output};
        }
        return {hidden_state_output, NULL};
    }

    void end() {
        ggml_allocr_free(compute_alloc);
        ggml_backend_buffer_free(compute_buffer);
        compute_alloc              = NULL;
        compute_memory_buffer_size = -1;
        hidden_state_output        = NULL;
        pooled_output              = NULL;
    }
};

/*==================================================== UnetModel =====================================================*/

struct ResBlock {
    // network hparams
    int channels;      // model_channels * (1, 1, 1, 2, 2, 4, 4, 4)
    int emb_channels;  // time_embed_dim
    int out_channels;  // mult * model_channels

    // network params
    // in_layers
    struct ggml_tensor* in_layer_0_w;  // [channels, ]
    struct ggml_tensor* in_layer_0_b;  // [channels, ]
    // in_layer_1 is nn.SILU()
    struct ggml_tensor* in_layer_2_w;  // [out_channels, channels, 3, 3]
    struct ggml_tensor* in_layer_2_b;  // [out_channels, ]

    // emb_layers
    // emb_layer_0 is nn.SILU()
    struct ggml_tensor* emb_layer_1_w;  // [out_channels, emb_channels]
    struct ggml_tensor* emb_layer_1_b;  // [out_channels, ]

    // out_layers
    struct ggml_tensor* out_layer_0_w;  // [out_channels, ]
    struct ggml_tensor* out_layer_0_b;  // [out_channels, ]
    // out_layer_1 is nn.SILU()
    // out_layer_2 is nn.Dropout(), p = 0 for inference
    struct ggml_tensor* out_layer_3_w;  // [out_channels, out_channels, 3, 3]
    struct ggml_tensor* out_layer_3_b;  // [out_channels, ]

    // skip connection, only if out_channels != channels
    struct ggml_tensor* skip_w;  // [out_channels, channels, 1, 1]
    struct ggml_tensor* skip_b;  // [out_channels, ]

    size_t calculate_mem_size(ggml_type wtype) {
        double mem_size = 0;
        mem_size += 2 * channels * ggml_type_sizef(GGML_TYPE_F32);                         // in_layer_0_w/b
        mem_size += out_channels * channels * 3 * 3 * ggml_type_sizef(GGML_TYPE_F16);      // in_layer_2_w
        mem_size += 5 * out_channels * ggml_type_sizef(GGML_TYPE_F32);                     // in_layer_2_b/emb_layer_1_b/out_layer_0_w/out_layer_0_b/out_layer_3_b
        mem_size += out_channels * emb_channels * ggml_type_sizef(wtype);                  // emb_layer_1_w
        mem_size += out_channels * out_channels * 3 * 3 * ggml_type_sizef(GGML_TYPE_F16);  // out_layer_3_w

        if (out_channels != channels) {
            mem_size += out_channels * channels * 1 * 1 * ggml_type_sizef(GGML_TYPE_F16);  // skip_w
            mem_size += out_channels * ggml_type_sizef(GGML_TYPE_F32);                     // skip_b
        }
        return static_cast<size_t>(mem_size);
    }

    void init_params(struct ggml_context* ctx, ggml_type wtype) {
        in_layer_0_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, channels);
        in_layer_0_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, channels);
        in_layer_2_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, channels, out_channels);
        in_layer_2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);

        emb_layer_1_w = ggml_new_tensor_2d(ctx, wtype, emb_channels, out_channels);
        emb_layer_1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);

        out_layer_0_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
        out_layer_0_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
        out_layer_3_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, out_channels, out_channels);
        out_layer_3_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);

        if (out_channels != channels) {
            skip_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 1, 1, channels, out_channels);
            skip_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
        }
    }

    void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        tensors[prefix + "in_layers.0.weight"] = in_layer_0_w;
        tensors[prefix + "in_layers.0.bias"]   = in_layer_0_b;
        tensors[prefix + "in_layers.2.weight"] = in_layer_2_w;
        tensors[prefix + "in_layers.2.bias"]   = in_layer_2_b;

        tensors[prefix + "emb_layers.1.weight"] = emb_layer_1_w;
        tensors[prefix + "emb_layers.1.bias"]   = emb_layer_1_b;

        tensors[prefix + "out_layers.0.weight"] = out_layer_0_w;
        tensors[prefix + "out_layers.0.bias"]   = out_layer_0_b;
        tensors[prefix + "out_layers.3.weight"] = out_layer_3_w;
        tensors[prefix + "out_layers.3.bias"]   = out_layer_3_b;

        if (out_channels != channels) {
            tensors[prefix + "skip_connection.weight"] = skip_w;
            tensors[prefix + "skip_connection.bias"]   = skip_b;
        }
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* emb) {
        // x: [N, channels, h, w]
        // emb: [N, emb_channels]

        // in_layers
        auto h = ggml_nn_group_norm(ctx, x, in_layer_0_w, in_layer_0_b);
        h      = ggml_silu_inplace(ctx, h);
        h      = ggml_nn_conv_2d(ctx, h, in_layer_2_w, in_layer_2_b, 1, 1, 1, 1);  // [N, out_channels, h, w]

        // emb_layers
        auto emb_out = ggml_silu(ctx, emb);
        emb_out      = ggml_nn_linear(ctx, emb_out, emb_layer_1_w, emb_layer_1_b);           // [N, out_channels]
        emb_out      = ggml_reshape_4d(ctx, emb_out, 1, 1, emb_out->ne[0], emb_out->ne[1]);  // [N, out_channels, 1, 1]

        // out_layers
        h = ggml_add(ctx, h, emb_out);
        h = ggml_nn_group_norm(ctx, h, out_layer_0_w, out_layer_0_b);
        h = ggml_silu_inplace(ctx, h);

        // dropout, skip for inference

        h = ggml_nn_conv_2d(ctx, h, out_layer_3_w, out_layer_3_b, 1, 1, 1, 1);  // [N, out_channels, h, w]

        // skip connection
        if (out_channels != channels) {
            x = ggml_nn_conv_2d(ctx, x, skip_w, skip_b);  // [N, out_channels, h, w]
        }

        h = ggml_add(ctx, h, x);
        return h;  // [N, out_channels, h, w]
    }
};

struct SpatialTransformer {
    int in_channels;        // mult * model_channels
    int n_head;             // num_heads
    int d_head;             // in_channels // n_heads
    int depth       = 1;    // 1
    int context_dim = 768;  // hidden_size, 1024 for VERSION_2_x

    // group norm
    struct ggml_tensor* norm_w;  // [in_channels,]
    struct ggml_tensor* norm_b;  // [in_channels,]

    // proj_in
    struct ggml_tensor* proj_in_w;  // [in_channels, in_channels, 1, 1]
    struct ggml_tensor* proj_in_b;  // [in_channels,]

    // transformer
    struct Transformer {
        // layer norm 1
        struct ggml_tensor* norm1_w;  // [in_channels, ]
        struct ggml_tensor* norm1_b;  // [in_channels, ]

        // attn1
        struct ggml_tensor* attn1_q_w;  // [in_channels, in_channels]
        struct ggml_tensor* attn1_k_w;  // [in_channels, in_channels]
        struct ggml_tensor* attn1_v_w;  // [in_channels, in_channels]

        struct ggml_tensor* attn1_out_w;  // [in_channels, in_channels]
        struct ggml_tensor* attn1_out_b;  // [in_channels, ]

        // layer norm 2
        struct ggml_tensor* norm2_w;  // [in_channels, ]
        struct ggml_tensor* norm2_b;  // [in_channels, ]

        // attn2
        struct ggml_tensor* attn2_q_w;  // [in_channels, in_channels]
        struct ggml_tensor* attn2_k_w;  // [in_channels, context_dim]
        struct ggml_tensor* attn2_v_w;  // [in_channels, context_dim]

        struct ggml_tensor* attn2_out_w;  // [in_channels, in_channels]
        struct ggml_tensor* attn2_out_b;  // [in_channels, ]

        // layer norm 3
        struct ggml_tensor* norm3_w;  // [in_channels, ]
        struct ggml_tensor* norm3_b;  // [in_channels, ]

        // ff
        struct ggml_tensor* ff_0_proj_w;  // [in_channels * 4 * 2, in_channels]
        struct ggml_tensor* ff_0_proj_b;  // [in_channels * 4 * 2]

        struct ggml_tensor* ff_2_w;  // [in_channels, in_channels * 4]
        struct ggml_tensor* ff_2_b;  // [in_channels,]
    };

    std::vector<Transformer> transformers;

    struct ggml_tensor* attn_scale;

    // proj_out
    struct ggml_tensor* proj_out_w;  // [in_channels, in_channels, 1, 1]
    struct ggml_tensor* proj_out_b;  // [in_channels,]

    SpatialTransformer(int depth = 1)
        : depth(depth) {
        transformers.resize(depth);
    }

    size_t get_num_tensors() {
        return depth * 20 + 7;
    }

    size_t calculate_mem_size(ggml_type wtype) {
        double mem_size = 0;
        mem_size += 2 * in_channels * ggml_type_sizef(GGML_TYPE_F32);                        // norm_w/norm_b
        mem_size += 2 * in_channels * in_channels * 1 * 1 * ggml_type_sizef(GGML_TYPE_F16);  // proj_in_w/proj_out_w
        mem_size += 2 * in_channels * ggml_type_sizef(GGML_TYPE_F32);                        // proj_in_b/proj_out_b
        mem_size += 1 * ggml_type_sizef(GGML_TYPE_F32);                                      // attn_scale

        // transformer
        for (auto& transformer : transformers) {
            mem_size += 6 * in_channels * ggml_type_sizef(GGML_TYPE_F32);            // norm1-3_w/b
            mem_size += 6 * in_channels * in_channels * ggml_type_sizef(wtype);      // attn1_q/k/v/out_w attn2_q/out_w
            mem_size += 2 * in_channels * context_dim * ggml_type_sizef(wtype);      // attn2_k/v_w
            mem_size += in_channels * 4 * 2 * in_channels * ggml_type_sizef(wtype);  // ff_0_proj_w
            mem_size += in_channels * 4 * 2 * ggml_type_sizef(GGML_TYPE_F32);        // ff_0_proj_b
            mem_size += in_channels * 4 * in_channels * ggml_type_sizef(wtype);      // ff_2_w
            mem_size += in_channels * ggml_type_sizef(GGML_TYPE_F32);                // ff_2_b
        }
        return static_cast<size_t>(mem_size);
    }

    void init_params(struct ggml_context* ctx, ggml_allocr* alloc, ggml_type wtype) {
        norm_w    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);
        norm_b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);
        proj_in_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 1, 1, in_channels, in_channels);
        proj_in_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);

        proj_out_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 1, 1, in_channels, in_channels);
        proj_out_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);

        attn_scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        ggml_allocr_alloc(alloc, attn_scale);
        float scale = 1.0f / sqrt((float)d_head);
        ggml_backend_tensor_set(attn_scale, &scale, 0, sizeof(scale));

        // transformer
        for (auto& transformer : transformers) {
            transformer.norm1_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);
            transformer.norm1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);

            transformer.attn1_q_w = ggml_new_tensor_2d(ctx, wtype, in_channels, in_channels);
            transformer.attn1_k_w = ggml_new_tensor_2d(ctx, wtype, in_channels, in_channels);
            transformer.attn1_v_w = ggml_new_tensor_2d(ctx, wtype, in_channels, in_channels);

            transformer.attn1_out_w = ggml_new_tensor_2d(ctx, wtype, in_channels, in_channels);
            transformer.attn1_out_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);

            transformer.norm2_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);
            transformer.norm2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);

            transformer.attn2_q_w = ggml_new_tensor_2d(ctx, wtype, in_channels, in_channels);
            transformer.attn2_k_w = ggml_new_tensor_2d(ctx, wtype, context_dim, in_channels);
            transformer.attn2_v_w = ggml_new_tensor_2d(ctx, wtype, context_dim, in_channels);

            transformer.attn2_out_w = ggml_new_tensor_2d(ctx, wtype, in_channels, in_channels);
            transformer.attn2_out_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);

            transformer.norm3_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);
            transformer.norm3_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);

            transformer.ff_0_proj_w = ggml_new_tensor_2d(ctx, wtype, in_channels, in_channels * 4 * 2);
            transformer.ff_0_proj_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels * 4 * 2);

            transformer.ff_2_w = ggml_new_tensor_2d(ctx, wtype, in_channels * 4, in_channels);
            transformer.ff_2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);
        }
    }

    void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        tensors[prefix + "norm.weight"]    = norm_w;
        tensors[prefix + "norm.bias"]      = norm_b;
        tensors[prefix + "proj_in.weight"] = proj_in_w;
        tensors[prefix + "proj_in.bias"]   = proj_in_b;

        // transformer
        for (int i = 0; i < transformers.size(); i++) {
            auto& transformer                                 = transformers[i];
            std::string transformer_prefix                    = prefix + "transformer_blocks." + std::to_string(i) + ".";
            tensors[transformer_prefix + "attn1.to_q.weight"] = transformer.attn1_q_w;
            tensors[transformer_prefix + "attn1.to_k.weight"] = transformer.attn1_k_w;
            tensors[transformer_prefix + "attn1.to_v.weight"] = transformer.attn1_v_w;

            tensors[transformer_prefix + "attn1.to_out.0.weight"] = transformer.attn1_out_w;
            tensors[transformer_prefix + "attn1.to_out.0.bias"]   = transformer.attn1_out_b;

            tensors[transformer_prefix + "ff.net.0.proj.weight"] = transformer.ff_0_proj_w;
            tensors[transformer_prefix + "ff.net.0.proj.bias"]   = transformer.ff_0_proj_b;
            tensors[transformer_prefix + "ff.net.2.weight"]      = transformer.ff_2_w;
            tensors[transformer_prefix + "ff.net.2.bias"]        = transformer.ff_2_b;

            tensors[transformer_prefix + "attn2.to_q.weight"] = transformer.attn2_q_w;
            tensors[transformer_prefix + "attn2.to_k.weight"] = transformer.attn2_k_w;
            tensors[transformer_prefix + "attn2.to_v.weight"] = transformer.attn2_v_w;

            tensors[transformer_prefix + "attn2.to_out.0.weight"] = transformer.attn2_out_w;
            tensors[transformer_prefix + "attn2.to_out.0.bias"]   = transformer.attn2_out_b;

            tensors[transformer_prefix + "norm1.weight"] = transformer.norm1_w;
            tensors[transformer_prefix + "norm1.bias"]   = transformer.norm1_b;
            tensors[transformer_prefix + "norm2.weight"] = transformer.norm2_w;
            tensors[transformer_prefix + "norm2.bias"]   = transformer.norm2_b;
            tensors[transformer_prefix + "norm3.weight"] = transformer.norm3_w;
            tensors[transformer_prefix + "norm3.bias"]   = transformer.norm3_b;
        }

        tensors[prefix + "proj_out.weight"] = proj_out_w;
        tensors[prefix + "proj_out.bias"]   = proj_out_b;
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* context) {
        // x: [N, in_channels, h, w]
        // context: [N, max_position, hidden_size(aka context_dim)]
        auto x_in = x;
        x         = ggml_nn_group_norm(ctx, x, norm_w, norm_b);
        // proj_in
        x = ggml_nn_conv_2d(ctx, x, proj_in_w, proj_in_b);  // [N, in_channels, h, w]

        // transformer
        const int64_t n            = x->ne[3];
        const int64_t c            = x->ne[2];
        const int64_t h            = x->ne[1];
        const int64_t w            = x->ne[0];
        const int64_t max_position = context->ne[1];
        x                          = ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3));  // [N, h, w, in_channels]

        for (auto& transformer : transformers) {
            auto r = x;
            // layer norm 1
            x = ggml_reshape_2d(ctx, x, c, w * h * n);
            x = ggml_nn_layer_norm(ctx, x, transformer.norm1_w, transformer.norm1_b);

            // self-attention
            {
                x                     = ggml_reshape_2d(ctx, x, c, h * w * n);        // [N * h * w, in_channels]
                struct ggml_tensor* q = ggml_mul_mat(ctx, transformer.attn1_q_w, x);  // [N * h * w, in_channels]
#if !defined(SD_USE_FLASH_ATTENTION) || defined(SD_USE_CUBLAS) || defined(SD_USE_METAL)
                q = ggml_scale_inplace(ctx, q, attn_scale);
#endif
                q = ggml_reshape_4d(ctx, q, d_head, n_head, h * w, n);   // [N, h * w, n_head, d_head]
                q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));    // [N, n_head, h * w, d_head]
                q = ggml_reshape_3d(ctx, q, d_head, h * w, n_head * n);  // [N * n_head, h * w, d_head]

                struct ggml_tensor* k = ggml_mul_mat(ctx, transformer.attn1_k_w, x);         // [N * h * w, in_channels]
                k                     = ggml_reshape_4d(ctx, k, d_head, n_head, h * w, n);   // [N, h * w, n_head, d_head]
                k                     = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));    // [N, n_head, h * w, d_head]
                k                     = ggml_reshape_3d(ctx, k, d_head, h * w, n_head * n);  // [N * n_head, h * w, d_head]

                struct ggml_tensor* v = ggml_mul_mat(ctx, transformer.attn1_v_w, x);         // [N * h * w, in_channels]
                v                     = ggml_reshape_4d(ctx, v, d_head, n_head, h * w, n);   // [N, h * w, n_head, d_head]
                v                     = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));    // [N, n_head, d_head, h * w]
                v                     = ggml_reshape_3d(ctx, v, h * w, d_head, n_head * n);  // [N * n_head, d_head, h * w]

#if defined(SD_USE_FLASH_ATTENTION) && !defined(SD_USE_CUBLAS) && !defined(SD_USE_METAL)
                struct ggml_tensor* kqv = ggml_flash_attn(ctx, q, k, v, false);  // [N * n_head, h * w, d_head]
#else
                struct ggml_tensor* kq = ggml_mul_mat(ctx, k, q);  // [N * n_head, h * w, h * w]
                // kq = ggml_diag_mask_inf_inplace(ctx, kq, 0);
                kq = ggml_soft_max_inplace(ctx, kq);

                struct ggml_tensor* kqv = ggml_mul_mat(ctx, v, kq);  // [N * n_head, h * w, d_head]
#endif
                kqv = ggml_reshape_4d(ctx, kqv, d_head, h * w, n_head, n);
                kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));  // [N, h * w, n_head, d_head]

                // x = ggml_cpy(ctx, kqv, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_head * n_head, h * w * n));
                x = ggml_reshape_2d(ctx, kqv, d_head * n_head, h * w * n);

                x = ggml_nn_linear(ctx, x, transformer.attn1_out_w, transformer.attn1_out_b);

                x = ggml_reshape_4d(ctx, x, c, w, h, n);
            }

            x = ggml_add(ctx, x, r);
            r = x;

            // layer norm 2
            x = ggml_nn_layer_norm(ctx, x, transformer.norm2_w, transformer.norm2_b);

            // cross-attention
            {
                x                     = ggml_reshape_2d(ctx, x, c, h * w * n);                                           // [N * h * w, in_channels]
                context               = ggml_reshape_2d(ctx, context, context->ne[0], context->ne[1] * context->ne[2]);  // [N * max_position, hidden_size]
                struct ggml_tensor* q = ggml_mul_mat(ctx, transformer.attn2_q_w, x);                                     // [N * h * w, in_channels]
#if !defined(SD_USE_FLASH_ATTENTION) || defined(SD_USE_CUBLAS) || defined(SD_USE_METAL)
                q = ggml_scale_inplace(ctx, q, attn_scale);
#endif
                q = ggml_reshape_4d(ctx, q, d_head, n_head, h * w, n);   // [N, h * w, n_head, d_head]
                q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));    // [N, n_head, h * w, d_head]
                q = ggml_reshape_3d(ctx, q, d_head, h * w, n_head * n);  // [N * n_head, h * w, d_head]

                struct ggml_tensor* k = ggml_mul_mat(ctx, transformer.attn2_k_w, context);          // [N * max_position, in_channels]
                k                     = ggml_reshape_4d(ctx, k, d_head, n_head, max_position, n);   // [N, max_position, n_head, d_head]
                k                     = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));           // [N, n_head, max_position, d_head]
                k                     = ggml_reshape_3d(ctx, k, d_head, max_position, n_head * n);  // [N * n_head, max_position, d_head]

                struct ggml_tensor* v = ggml_mul_mat(ctx, transformer.attn2_v_w, context);          // [N * max_position, in_channels]
                v                     = ggml_reshape_4d(ctx, v, d_head, n_head, max_position, n);   // [N, max_position, n_head, d_head]
                v                     = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));           // [N, n_head, d_head, max_position]
                v                     = ggml_reshape_3d(ctx, v, max_position, d_head, n_head * n);  // [N * n_head, d_head, max_position]
#if defined(SD_USE_FLASH_ATTENTION) && !defined(SD_USE_CUBLAS) && !defined(SD_USE_METAL)
                struct ggml_tensor* kqv = ggml_flash_attn(ctx, q, k, v, false);  // [N * n_head, h * w, d_head]
#else
                struct ggml_tensor* kq  = ggml_mul_mat(ctx, k, q);   // [N * n_head, h * w, max_position]
                // kq = ggml_diag_mask_inf_inplace(ctx, kq, 0);
                kq = ggml_soft_max_inplace(ctx, kq);

                struct ggml_tensor* kqv = ggml_mul_mat(ctx, v, kq);  // [N * n_head, h * w, d_head]
#endif
                kqv = ggml_reshape_4d(ctx, kqv, d_head, h * w, n_head, n);
                kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));

                // x = ggml_cpy(ctx, kqv, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_head * n_head, h * w * n)); // [N * h * w, in_channels]
                x = ggml_reshape_2d(ctx, kqv, d_head * n_head, h * w * n);  // [N * h * w, in_channels]

                x = ggml_nn_linear(ctx, x, transformer.attn2_out_w, transformer.attn2_out_b);

                x = ggml_reshape_4d(ctx, x, c, w, h, n);
            }

            x = ggml_add(ctx, x, r);
            r = x;

            // layer norm 3
            x = ggml_reshape_2d(ctx, x, c, h * w * n);  // [N * h * w, in_channels]
            x = ggml_nn_layer_norm(ctx, x, transformer.norm3_w, transformer.norm3_b);

            // ff
            {
                // GEGLU
                auto x_w    = ggml_view_2d(ctx,
                                           transformer.ff_0_proj_w,
                                           transformer.ff_0_proj_w->ne[0],
                                           transformer.ff_0_proj_w->ne[1] / 2,
                                           transformer.ff_0_proj_w->nb[1],
                                           0);  // [in_channels * 4, in_channels]
                auto x_b    = ggml_view_1d(ctx,
                                           transformer.ff_0_proj_b,
                                           transformer.ff_0_proj_b->ne[0] / 2,
                                           0);  // [in_channels * 4, in_channels]
                auto gate_w = ggml_view_2d(ctx,
                                           transformer.ff_0_proj_w,
                                           transformer.ff_0_proj_w->ne[0],
                                           transformer.ff_0_proj_w->ne[1] / 2,
                                           transformer.ff_0_proj_w->nb[1],
                                           transformer.ff_0_proj_w->nb[1] * transformer.ff_0_proj_w->ne[1] / 2);  // [in_channels * 4, ]
                auto gate_b = ggml_view_1d(ctx,
                                           transformer.ff_0_proj_b,
                                           transformer.ff_0_proj_b->ne[0] / 2,
                                           transformer.ff_0_proj_b->nb[0] * transformer.ff_0_proj_b->ne[0] / 2);  // [in_channels * 4, ]
                x           = ggml_reshape_2d(ctx, x, c, w * h * n);
                auto x_in   = x;
                x           = ggml_nn_linear(ctx, x_in, x_w, x_b);        // [N * h * w, in_channels * 4]
                auto gate   = ggml_nn_linear(ctx, x_in, gate_w, gate_b);  // [N * h * w, in_channels * 4]

                gate = ggml_gelu_inplace(ctx, gate);

                x = ggml_mul(ctx, x, gate);  // [N * h * w, in_channels * 4]
                // fc
                x = ggml_nn_linear(ctx, x, transformer.ff_2_w, transformer.ff_2_b);  // [N * h * w, in_channels]
            }

            x = ggml_reshape_4d(ctx, x, c, w, h, n);  // [N, h, w, in_channels]

            // residual
            x = ggml_add(ctx, x, r);
        }

        x = ggml_cont(ctx, ggml_permute(ctx, x, 2, 0, 1, 3));  // [N, in_channels, h, w]

        // proj_out
        x = ggml_nn_conv_2d(ctx, x, proj_out_w, proj_out_b);  // [N, in_channels, h, w]

        x = ggml_add(ctx, x, x_in);
        return x;
    }
};

struct DownSample {
    // hparams
    int channels;
    int out_channels;

    // conv2d params
    struct ggml_tensor* op_w;  // [out_channels, channels, 3, 3]
    struct ggml_tensor* op_b;  // [out_channels,]

    bool vae_downsample = false;

    size_t calculate_mem_size(ggml_type wtype) {
        double mem_size = 0;
        mem_size += out_channels * channels * 3 * 3 * ggml_type_sizef(GGML_TYPE_F16);  // op_w
        mem_size += out_channels * ggml_type_sizef(GGML_TYPE_F32);                     // op_b
        return static_cast<size_t>(mem_size);
    }

    void init_params(struct ggml_context* ctx, ggml_type wtype) {
        op_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, channels, out_channels);
        op_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
    }

    void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        if (vae_downsample) {
            tensors[prefix + "conv.weight"] = op_w;
            tensors[prefix + "conv.bias"]   = op_b;
        } else {
            tensors[prefix + "op.weight"] = op_w;
            tensors[prefix + "op.bias"]   = op_b;
        }
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [N, channels, h, w]
        struct ggml_tensor* c = NULL;
        if (vae_downsample) {
            c = ggml_pad(ctx, x, 1, 1, 0, 0);
            c = ggml_nn_conv_2d(ctx, c, op_w, op_b, 2, 2, 0, 0);
        } else {
            c = ggml_nn_conv_2d(ctx, x, op_w, op_b, 2, 2, 1, 1);
        }
        return c;  // [N, out_channels, h/2, w/2]
    }
};

struct UpSample {
    // hparams
    int channels;
    int out_channels;

    // conv2d params
    struct ggml_tensor* conv_w;  // [out_channels, channels, 3, 3]
    struct ggml_tensor* conv_b;  // [out_channels,]

    size_t calculate_mem_size(ggml_type wtype) {
        double mem_size = 0;
        mem_size += out_channels * channels * 3 * 3 * ggml_type_sizef(GGML_TYPE_F16);  // op_w
        mem_size += out_channels * ggml_type_sizef(GGML_TYPE_F32);                     // op_b
        return static_cast<size_t>(mem_size);
    }

    void init_params(struct ggml_context* ctx, ggml_type wtype) {
        conv_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, channels, out_channels);
        conv_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
    }

    void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        tensors[prefix + "conv.weight"] = conv_w;
        tensors[prefix + "conv.bias"]   = conv_b;
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [N, channels, h, w]
        x = ggml_upscale(ctx, x, 2);                              // [N, channels, h*2, w*2]
        x = ggml_nn_conv_2d(ctx, x, conv_w, conv_b, 1, 1, 1, 1);  // [N, out_channels, h*2, w*2]
        return x;
    }
};

// ldm.modules.diffusionmodules.openaimodel.UNetModel
struct UNetModel {
    SDVersion version = VERSION_1_x;
    // network hparams
    int in_channels                        = 4;
    int model_channels                     = 320;
    int out_channels                       = 4;
    int num_res_blocks                     = 2;
    std::vector<int> attention_resolutions = {4, 2, 1};
    std::vector<int> channel_mult          = {1, 2, 4, 4};
    std::vector<int> transformer_depth     = {1, 1, 1, 1};
    int time_embed_dim                     = 1280;  // model_channels*4
    int num_heads                          = 8;
    int num_head_channels                  = -1;    // channels // num_heads
    int context_dim                        = 768;   // 1024 for VERSION_2_x, 2048 for VERSION_XL
    int adm_in_channels                    = 2816;  // only for VERSION_XL

    // network params
    struct ggml_tensor* time_embed_0_w;  // [time_embed_dim, model_channels]
    struct ggml_tensor* time_embed_0_b;  // [time_embed_dim, ]
    // time_embed_1 is nn.SILU()
    struct ggml_tensor* time_embed_2_w;  // [time_embed_dim, time_embed_dim]
    struct ggml_tensor* time_embed_2_b;  // [time_embed_dim, ]

    struct ggml_tensor* label_embed_0_w;  // [time_embed_dim, adm_in_channels]
    struct ggml_tensor* label_embed_0_b;  // [time_embed_dim, ]
    // label_embed_1 is nn.SILU()
    struct ggml_tensor* label_embed_2_w;  // [time_embed_dim, time_embed_dim]
    struct ggml_tensor* label_embed_2_b;  // [time_embed_dim, ]

    struct ggml_tensor* input_block_0_w;  // [model_channels, in_channels, 3, 3]
    struct ggml_tensor* input_block_0_b;  // [model_channels, ]

    // input_blocks
    ResBlock input_res_blocks[4][2];
    SpatialTransformer input_transformers[3][2];
    DownSample input_down_samples[3];

    // middle_block
    ResBlock middle_block_0;
    SpatialTransformer middle_block_1;
    ResBlock middle_block_2;

    // output_blocks
    ResBlock output_res_blocks[4][3];
    SpatialTransformer output_transformers[3][3];
    UpSample output_up_samples[3];

    // out
    // group norm 32
    struct ggml_tensor* out_0_w;  // [model_channels, ]
    struct ggml_tensor* out_0_b;  // [model_channels, ]
    // out 1 is nn.SILU()
    struct ggml_tensor* out_2_w;  // [out_channels, model_channels, 3, 3]
    struct ggml_tensor* out_2_b;  // [out_channels, ]

    struct ggml_context* ctx;
    ggml_backend_buffer_t params_buffer;
    ggml_backend_buffer_t compute_buffer;  // for compute
    struct ggml_allocr* compute_alloc = NULL;
    size_t compute_memory_buffer_size = -1;

    size_t memory_buffer_size = 0;
    ggml_type wtype;
    ggml_backend_t backend = NULL;

    UNetModel(SDVersion version = VERSION_1_x)
        : version(version) {
        if (version == VERSION_2_x) {
            context_dim       = 1024;
            num_head_channels = 64;
            num_heads         = -1;
        } else if (version == VERSION_XL) {
            context_dim           = 2048;
            attention_resolutions = {4, 2};
            channel_mult          = {1, 2, 4};
            transformer_depth     = {1, 2, 10};
            num_head_channels     = 64;
            num_heads             = -1;
        }
        // set up hparams of blocks

        // input_blocks
        std::vector<int> input_block_chans;
        input_block_chans.push_back(model_channels);
        int ch = model_channels;
        int ds = 1;

        int len_mults = channel_mult.size();
        for (int i = 0; i < len_mults; i++) {
            int mult = channel_mult[i];
            for (int j = 0; j < num_res_blocks; j++) {
                input_res_blocks[i][j].channels     = ch;
                input_res_blocks[i][j].emb_channels = time_embed_dim;
                input_res_blocks[i][j].out_channels = mult * model_channels;

                ch = mult * model_channels;
                if (std::find(attention_resolutions.begin(), attention_resolutions.end(), ds) != attention_resolutions.end()) {
                    int n_head = num_heads;
                    int d_head = ch / num_heads;
                    if (num_head_channels != -1) {
                        d_head = num_head_channels;
                        n_head = ch / d_head;
                    }
                    input_transformers[i][j]             = SpatialTransformer(transformer_depth[i]);
                    input_transformers[i][j].in_channels = ch;
                    input_transformers[i][j].n_head      = n_head;
                    input_transformers[i][j].d_head      = d_head;
                    input_transformers[i][j].context_dim = context_dim;
                }
                input_block_chans.push_back(ch);
            }
            if (i != len_mults - 1) {
                input_down_samples[i].channels     = ch;
                input_down_samples[i].out_channels = ch;
                input_block_chans.push_back(ch);

                ds *= 2;
            }
        }

        // middle blocks
        middle_block_0.channels     = ch;
        middle_block_0.emb_channels = time_embed_dim;
        middle_block_0.out_channels = ch;

        int n_head = num_heads;
        int d_head = ch / num_heads;
        if (num_head_channels != -1) {
            d_head = num_head_channels;
            n_head = ch / d_head;
        }
        middle_block_1             = SpatialTransformer(transformer_depth[transformer_depth.size() - 1]);
        middle_block_1.in_channels = ch;
        middle_block_1.n_head      = n_head;
        middle_block_1.d_head      = d_head;
        middle_block_1.context_dim = context_dim;

        middle_block_2.channels     = ch;
        middle_block_2.emb_channels = time_embed_dim;
        middle_block_2.out_channels = ch;

        // output blocks
        for (int i = len_mults - 1; i >= 0; i--) {
            int mult = channel_mult[i];
            for (int j = 0; j < num_res_blocks + 1; j++) {
                int ich = input_block_chans.back();
                input_block_chans.pop_back();

                output_res_blocks[i][j].channels     = ch + ich;
                output_res_blocks[i][j].emb_channels = time_embed_dim;
                output_res_blocks[i][j].out_channels = mult * model_channels;

                ch = mult * model_channels;

                if (std::find(attention_resolutions.begin(), attention_resolutions.end(), ds) != attention_resolutions.end()) {
                    int n_head = num_heads;
                    int d_head = ch / num_heads;
                    if (num_head_channels != -1) {
                        d_head = num_head_channels;
                        n_head = ch / d_head;
                    }
                    output_transformers[i][j]             = SpatialTransformer(transformer_depth[i]);
                    output_transformers[i][j].in_channels = ch;
                    output_transformers[i][j].n_head      = n_head;
                    output_transformers[i][j].d_head      = d_head;
                    output_transformers[i][j].context_dim = context_dim;
                }

                if (i > 0 && j == num_res_blocks) {
                    output_up_samples[i - 1].channels     = ch;
                    output_up_samples[i - 1].out_channels = ch;

                    ds /= 2;
                }
            }
        }
    }

    size_t calculate_mem_size() {
        double mem_size = 0;
        mem_size += time_embed_dim * model_channels * ggml_type_sizef(wtype);  // time_embed_0_w
        mem_size += time_embed_dim * ggml_type_sizef(GGML_TYPE_F32);           // time_embed_0_b
        mem_size += time_embed_dim * time_embed_dim * ggml_type_sizef(wtype);  // time_embed_2_w
        mem_size += time_embed_dim * ggml_type_sizef(GGML_TYPE_F32);           // time_embed_2_b

        if (version == VERSION_XL) {
            mem_size += time_embed_dim * adm_in_channels * ggml_type_sizef(wtype);  // label_embed_0_w
            mem_size += time_embed_dim * ggml_type_sizef(GGML_TYPE_F32);            // label_embed_0_b
            mem_size += time_embed_dim * time_embed_dim * ggml_type_sizef(wtype);   // label_embed_2_w
            mem_size += time_embed_dim * ggml_type_sizef(GGML_TYPE_F32);            // label_embed_2_b
        }

        mem_size += model_channels * in_channels * 3 * 3 * ggml_type_sizef(GGML_TYPE_F16);  // input_block_0_w
        mem_size += model_channels * ggml_type_sizef(GGML_TYPE_F32);                        // input_block_0_b

        // input_blocks
        int ds        = 1;
        int len_mults = channel_mult.size();
        for (int i = 0; i < len_mults; i++) {
            for (int j = 0; j < num_res_blocks; j++) {
                mem_size += input_res_blocks[i][j].calculate_mem_size(wtype);
                if (std::find(attention_resolutions.begin(), attention_resolutions.end(), ds) != attention_resolutions.end()) {
                    mem_size += input_transformers[i][j].calculate_mem_size(wtype);
                }
            }
            if (i != len_mults - 1) {
                ds *= 2;
                mem_size += input_down_samples[i].calculate_mem_size(wtype);
            }
        }

        // middle_block
        mem_size += middle_block_0.calculate_mem_size(wtype);
        mem_size += middle_block_1.calculate_mem_size(wtype);
        mem_size += middle_block_2.calculate_mem_size(wtype);

        // output_blocks
        for (int i = len_mults - 1; i >= 0; i--) {
            for (int j = 0; j < num_res_blocks + 1; j++) {
                mem_size += output_res_blocks[i][j].calculate_mem_size(wtype);

                if (std::find(attention_resolutions.begin(), attention_resolutions.end(), ds) != attention_resolutions.end()) {
                    mem_size += output_transformers[i][j].calculate_mem_size(wtype);
                }

                if (i > 0 && j == num_res_blocks) {
                    mem_size += output_up_samples[i - 1].calculate_mem_size(wtype);

                    ds /= 2;
                }
            }
        }

        // out
        mem_size += 2 * model_channels * ggml_type_sizef(GGML_TYPE_F32);                     // out_0_w/b
        mem_size += out_channels * model_channels * 3 * 3 * ggml_type_sizef(GGML_TYPE_F16);  // out_2_w
        mem_size += out_channels * ggml_type_sizef(GGML_TYPE_F32);                           // out_2_b

        return static_cast<size_t>(mem_size);
    }

    int get_num_tensors() {
        // in
        int num_tensors = 6;
        if (version == VERSION_XL) {
            num_tensors += 4;
        }

        // input blocks
        int ds        = 1;
        int len_mults = channel_mult.size();
        for (int i = 0; i < len_mults; i++) {
            for (int j = 0; j < num_res_blocks; j++) {
                num_tensors += 12;
                if (std::find(attention_resolutions.begin(), attention_resolutions.end(), ds) != attention_resolutions.end()) {
                    num_tensors += input_transformers[i][j].get_num_tensors();
                }
            }
            if (i != len_mults - 1) {
                ds *= 2;
                num_tensors += 2;
            }
        }

        // middle blocks
        num_tensors += 13 * 2;
        num_tensors += middle_block_1.get_num_tensors();

        // output blocks
        for (int i = len_mults - 1; i >= 0; i--) {
            for (int j = 0; j < num_res_blocks + 1; j++) {
                num_tensors += 12;

                if (std::find(attention_resolutions.begin(), attention_resolutions.end(), ds) != attention_resolutions.end()) {
                    num_tensors += output_transformers[i][j].get_num_tensors();
                }

                if (i > 0 && j == num_res_blocks) {
                    num_tensors += 2;

                    ds /= 2;
                }
            }
        }

        // out
        num_tensors += 4;
        return num_tensors;
    }

    bool initialize(ggml_backend_t backend_, ggml_type wtype_) {
        backend            = backend_;
        wtype              = wtype_;
        memory_buffer_size = 10 * 1024 * 1024;  // 10 MB, for padding
        memory_buffer_size += calculate_mem_size();
        int num_tensors = get_num_tensors();

        LOG_DEBUG("unet params backend buffer size = % 6.2f MB (%i tensors)", memory_buffer_size / (1024.0 * 1024.0), num_tensors);

        struct ggml_init_params params;
        params.mem_size   = static_cast<size_t>(num_tensors * ggml_tensor_overhead()) + 1 * 1024 * 1024;
        params.mem_buffer = NULL;
        params.no_alloc   = true;
        // LOG_DEBUG("mem_size %u ", params.mem_size);

        ctx = ggml_init(params);
        if (!ctx) {
            LOG_ERROR("ggml_init() failed");
            return false;
        }

        params_buffer = ggml_backend_alloc_buffer(backend, memory_buffer_size);
        return true;
    }

    void destroy() {
        if (ctx != NULL) {
            ggml_free(ctx);
            ctx = NULL;
        }

        if (params_buffer != NULL) {
            ggml_backend_buffer_free(params_buffer);
            params_buffer = NULL;
        }
    }

    void alloc_params() {
        ggml_allocr* alloc = ggml_allocr_new_from_buffer(params_buffer);
        time_embed_0_w     = ggml_new_tensor_2d(ctx, wtype, model_channels, time_embed_dim);
        time_embed_0_b     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, time_embed_dim);
        time_embed_2_w     = ggml_new_tensor_2d(ctx, wtype, time_embed_dim, time_embed_dim);
        time_embed_2_b     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, time_embed_dim);

        // SDXL
        if (version == VERSION_XL) {
            label_embed_0_w = ggml_new_tensor_2d(ctx, wtype, adm_in_channels, time_embed_dim);
            label_embed_0_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, time_embed_dim);
            label_embed_2_w = ggml_new_tensor_2d(ctx, wtype, time_embed_dim, time_embed_dim);
            label_embed_2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, time_embed_dim);
        }

        // input_blocks
        input_block_0_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, in_channels, model_channels);
        input_block_0_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, model_channels);

        int ds        = 1;
        int len_mults = channel_mult.size();
        for (int i = 0; i < len_mults; i++) {
            for (int j = 0; j < num_res_blocks; j++) {
                input_res_blocks[i][j].init_params(ctx, wtype);
                if (std::find(attention_resolutions.begin(), attention_resolutions.end(), ds) != attention_resolutions.end()) {
                    input_transformers[i][j].init_params(ctx, alloc, wtype);
                }
            }
            if (i != len_mults - 1) {
                input_down_samples[i].init_params(ctx, wtype);
                ds *= 2;
            }
        }

        // middle_blocks
        middle_block_0.init_params(ctx, wtype);
        middle_block_1.init_params(ctx, alloc, wtype);
        middle_block_2.init_params(ctx, wtype);

        // output_blocks
        for (int i = len_mults - 1; i >= 0; i--) {
            for (int j = 0; j < num_res_blocks + 1; j++) {
                output_res_blocks[i][j].init_params(ctx, wtype);

                if (std::find(attention_resolutions.begin(), attention_resolutions.end(), ds) != attention_resolutions.end()) {
                    output_transformers[i][j].init_params(ctx, alloc, wtype);
                }

                if (i > 0 && j == num_res_blocks) {
                    output_up_samples[i - 1].init_params(ctx, wtype);

                    ds /= 2;
                }
            }
        }

        // out
        out_0_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, model_channels);
        out_0_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, model_channels);

        out_2_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, model_channels, out_channels);
        out_2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);

        // alloc all tensors linked to this context
        for (struct ggml_tensor* t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->data == NULL) {
                ggml_allocr_alloc(alloc, t);
            }
        }

        ggml_allocr_free(alloc);
    }

    void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        tensors[prefix + "time_embed.0.weight"] = time_embed_0_w;
        tensors[prefix + "time_embed.0.bias"]   = time_embed_0_b;
        tensors[prefix + "time_embed.2.weight"] = time_embed_2_w;
        tensors[prefix + "time_embed.2.bias"]   = time_embed_2_b;

        if (version == VERSION_XL) {
            tensors[prefix + "label_emb.0.0.weight"] = label_embed_0_w;
            tensors[prefix + "label_emb.0.0.bias"]   = label_embed_0_b;
            tensors[prefix + "label_emb.0.2.weight"] = label_embed_2_w;
            tensors[prefix + "label_emb.0.2.bias"]   = label_embed_2_b;
        }

        // input_blocks
        tensors[prefix + "input_blocks.0.0.weight"] = input_block_0_w;
        tensors[prefix + "input_blocks.0.0.bias"]   = input_block_0_b;

        int len_mults       = channel_mult.size();
        int input_block_idx = 0;
        int ds              = 1;
        for (int i = 0; i < len_mults; i++) {
            for (int j = 0; j < num_res_blocks; j++) {
                input_block_idx += 1;
                input_res_blocks[i][j].map_by_name(tensors, prefix + "input_blocks." + std::to_string(input_block_idx) + ".0.");
                if (std::find(attention_resolutions.begin(), attention_resolutions.end(), ds) != attention_resolutions.end()) {
                    input_transformers[i][j].map_by_name(tensors, prefix + "input_blocks." + std::to_string(input_block_idx) + ".1.");
                }
            }
            if (i != len_mults - 1) {
                input_block_idx += 1;
                input_down_samples[i].map_by_name(tensors, prefix + "input_blocks." + std::to_string(input_block_idx) + ".0.");
                ds *= 2;
            }
        }

        // middle_blocks
        middle_block_0.map_by_name(tensors, prefix + "middle_block.0.");
        middle_block_1.map_by_name(tensors, prefix + "middle_block.1.");
        middle_block_2.map_by_name(tensors, prefix + "middle_block.2.");

        // output_blocks
        int output_block_idx = 0;
        for (int i = len_mults - 1; i >= 0; i--) {
            for (int j = 0; j < num_res_blocks + 1; j++) {
                output_res_blocks[i][j].map_by_name(tensors, prefix + "output_blocks." + std::to_string(output_block_idx) + ".0.");

                int up_sample_idx = 1;
                if (std::find(attention_resolutions.begin(), attention_resolutions.end(), ds) != attention_resolutions.end()) {
                    output_transformers[i][j].map_by_name(tensors, prefix + "output_blocks." + std::to_string(output_block_idx) + ".1.");
                    up_sample_idx++;
                }

                if (i > 0 && j == num_res_blocks) {
                    output_up_samples[i - 1].map_by_name(tensors, prefix + "output_blocks." + std::to_string(output_block_idx) + "." + std::to_string(up_sample_idx) + ".");

                    ds /= 2;
                }
                output_block_idx += 1;
            }
        }

        // out
        tensors[prefix + "out.0.weight"] = out_0_w;
        tensors[prefix + "out.0.bias"]   = out_0_b;
        tensors[prefix + "out.2.weight"] = out_2_w;
        tensors[prefix + "out.2.bias"]   = out_2_b;
    }

    struct ggml_tensor* forward(struct ggml_context* ctx0,
                                struct ggml_tensor* x,
                                struct ggml_tensor* timesteps,
                                struct ggml_tensor* context,
                                struct ggml_tensor* t_emb = NULL,
                                struct ggml_tensor* y     = NULL) {
        // x: [N, in_channels, h, w]
        // timesteps: [N, ]
        // t_emb: [N, model_channels]
        // context: [N, max_position, hidden_size]([N, 77, 768])
        // y: [adm_in_channels]
        if (t_emb == NULL && timesteps != NULL) {
            t_emb = new_timestep_embedding(ctx0, compute_alloc, timesteps, model_channels);  // [N, model_channels]
        }

        // time_embed = nn.Sequential
        auto emb = ggml_nn_linear(ctx0, t_emb, time_embed_0_w, time_embed_0_b);
        emb      = ggml_silu_inplace(ctx0, emb);
        emb      = ggml_nn_linear(ctx0, emb, time_embed_2_w, time_embed_2_b);  // [N, time_embed_dim]

        // SDXL
        if (y != NULL) {
            auto label_emb = ggml_nn_linear(ctx0, y, label_embed_0_w, label_embed_0_b);
            label_emb      = ggml_silu_inplace(ctx0, label_emb);
            label_emb      = ggml_nn_linear(ctx0, label_emb, label_embed_2_w, label_embed_2_b);
            emb            = ggml_add(ctx, emb, label_emb);  // [N, time_embed_dim]
        }

        // input_blocks
        std::vector<struct ggml_tensor*> hs;

        // input block 0
        struct ggml_tensor* h = ggml_nn_conv_2d(ctx0, x, input_block_0_w, input_block_0_b, 1, 1, 1, 1);  // [N, model_channels, h, w]

        ggml_set_name(h, "bench-start");
        hs.push_back(h);
        // input block 1-11
        int len_mults = channel_mult.size();
        int ds        = 1;
        for (int i = 0; i < len_mults; i++) {
            int mult = channel_mult[i];
            for (int j = 0; j < num_res_blocks; j++) {
                h = input_res_blocks[i][j].forward(ctx0, h, emb);  // [N, mult*model_channels, h, w]
                if (std::find(attention_resolutions.begin(), attention_resolutions.end(), ds) != attention_resolutions.end()) {
                    h = input_transformers[i][j].forward(ctx0, h, context);  // [N, mult*model_channels, h, w]
                }
                hs.push_back(h);
            }
            if (i != len_mults - 1) {
                ds *= 2;
                h = input_down_samples[i].forward(ctx0, h);  // [N, mult*model_channels, h/(2^(i+1)), w/(2^(i+1))]
                hs.push_back(h);
            }
        }
        // [N, 4*model_channels, h/8, w/8]

        // middle_block
        h = middle_block_0.forward(ctx0, h, emb);      // [N, 4*model_channels, h/8, w/8]
        h = middle_block_1.forward(ctx0, h, context);  // [N, 4*model_channels, h/8, w/8]
        h = middle_block_2.forward(ctx0, h, emb);      // [N, 4*model_channels, h/8, w/8]

        // output_blocks
        for (int i = len_mults - 1; i >= 0; i--) {
            for (int j = 0; j < num_res_blocks + 1; j++) {
                auto h_skip = hs.back();
                hs.pop_back();

                h = ggml_concat(ctx0, h, h_skip);
                h = output_res_blocks[i][j].forward(ctx0, h, emb);

                if (std::find(attention_resolutions.begin(), attention_resolutions.end(), ds) != attention_resolutions.end()) {
                    h = output_transformers[i][j].forward(ctx0, h, context);
                }

                if (i > 0 && j == num_res_blocks) {
                    h = output_up_samples[i - 1].forward(ctx0, h);

                    ds /= 2;
                }
            }
        }

        // out
        h = ggml_nn_group_norm(ctx0, h, out_0_w, out_0_b);
        h = ggml_silu_inplace(ctx0, h);

        // conv2d
        h = ggml_nn_conv_2d(ctx0, h, out_2_w, out_2_b, 1, 1, 1, 1);  // [N, out_channels, h, w]
        ggml_set_name(h, "bench-end");
        return h;
    }

    struct ggml_cgraph* build_graph(struct ggml_tensor* x,
                                    struct ggml_tensor* timesteps,
                                    struct ggml_tensor* context,
                                    struct ggml_tensor* t_emb = NULL,
                                    struct ggml_tensor* y     = NULL) {
        // since we are using ggml-alloc, this buffer only needs enough space to hold the ggml_tensor and ggml_cgraph structs, but not the tensor data
        static size_t buf_size = ggml_tensor_overhead() * UNET_GRAPH_SIZE + ggml_graph_overhead();
        static std::vector<uint8_t> buf(buf_size);

        struct ggml_init_params params = {
            /*.mem_size   =*/buf_size,
            /*.mem_buffer =*/buf.data(),
            /*.no_alloc   =*/true,  // the tensors will be allocated later by ggml_allocr_alloc_graph()
        };
        // LOG_DEBUG("mem_size %u ", params.mem_size);

        struct ggml_context* ctx0 = ggml_init(params);

        struct ggml_cgraph* gf = ggml_new_graph_custom(ctx0, UNET_GRAPH_SIZE, false);

        // temporal tensors for transfer tensors from cpu to gpu if needed
        struct ggml_tensor* x_t         = NULL;
        struct ggml_tensor* timesteps_t = NULL;
        struct ggml_tensor* context_t   = NULL;
        struct ggml_tensor* t_emb_t     = NULL;
        struct ggml_tensor* y_t         = NULL;

        // it's performing a compute, check if backend isn't cpu
        if (!ggml_backend_is_cpu(backend)) {
            // pass input tensors to gpu memory
            x_t       = ggml_dup_tensor(ctx0, x);
            context_t = ggml_dup_tensor(ctx0, context);
            ggml_allocr_alloc(compute_alloc, x_t);
            if (timesteps != NULL) {
                timesteps_t = ggml_dup_tensor(ctx0, timesteps);
                ggml_allocr_alloc(compute_alloc, timesteps_t);
            }
            ggml_allocr_alloc(compute_alloc, context_t);
            if (t_emb != NULL) {
                t_emb_t = ggml_dup_tensor(ctx0, t_emb);
                ggml_allocr_alloc(compute_alloc, t_emb_t);
            }
            if (y != NULL) {
                y_t = ggml_dup_tensor(ctx0, y);
                ggml_allocr_alloc(compute_alloc, y_t);
            }
            // pass data to device backend
            if (!ggml_allocr_is_measure(compute_alloc)) {
                ggml_backend_tensor_set(x_t, x->data, 0, ggml_nbytes(x));
                ggml_backend_tensor_set(context_t, context->data, 0, ggml_nbytes(context));
                if (timesteps_t != NULL) {
                    ggml_backend_tensor_set(timesteps_t, timesteps->data, 0, ggml_nbytes(timesteps));
                }
                if (t_emb_t != NULL) {
                    ggml_backend_tensor_set(t_emb_t, t_emb->data, 0, ggml_nbytes(t_emb));
                }
                if (y != NULL) {
                    ggml_backend_tensor_set(y_t, y->data, 0, ggml_nbytes(y));
                }
            }
        } else {
            // if it's cpu backend just pass the same tensors
            x_t         = x;
            timesteps_t = timesteps;
            context_t   = context;
            t_emb_t     = t_emb;
            y_t         = y;
        }

        struct ggml_tensor* out = forward(ctx0, x_t, timesteps_t, context_t, t_emb_t, y_t);

        ggml_build_forward_expand(gf, out);
        ggml_free(ctx0);

        return gf;
    }

    void begin(struct ggml_tensor* x,
               struct ggml_tensor* context,
               struct ggml_tensor* t_emb = NULL,
               struct ggml_tensor* y     = NULL) {
        if (compute_memory_buffer_size == -1) {
            // alignment required by the backend
            compute_alloc = ggml_allocr_new_measure_from_backend(backend);

            struct ggml_cgraph* gf = build_graph(x, NULL, context, t_emb, y);

            // compute the required memory
            compute_memory_buffer_size = ggml_allocr_alloc_graph(compute_alloc, gf);

            // recreate the allocator with the required memory
            ggml_allocr_free(compute_alloc);

            LOG_DEBUG("diffusion compute buffer size: %.2f MB", compute_memory_buffer_size / 1024.0 / 1024.0);
        }

        compute_buffer = ggml_backend_alloc_buffer(backend, compute_memory_buffer_size);
        compute_alloc  = ggml_allocr_new_from_buffer(compute_buffer);
    }

    void compute(struct ggml_tensor* work_latent,
                 int n_threads,
                 struct ggml_tensor* x,
                 struct ggml_tensor* timesteps,
                 struct ggml_tensor* context,
                 struct ggml_tensor* t_emb = NULL,
                 struct ggml_tensor* y     = NULL) {
        ggml_allocr_reset(compute_alloc);

        // compute
        struct ggml_cgraph* gf = build_graph(x, timesteps, context, t_emb, y);

        ggml_allocr_alloc_graph(compute_alloc, gf);

        if (ggml_backend_is_cpu(backend)) {
            ggml_backend_cpu_set_n_threads(backend, n_threads);
        }

#ifdef SD_USE_METAL
        if (ggml_backend_is_metal(backend)) {
            ggml_backend_metal_set_n_cb(backend, n_threads);
        }
#endif

        ggml_backend_graph_compute(backend, gf);

#ifdef GGML_PERF
        ggml_graph_print(gf);
#endif

        ggml_backend_tensor_get_and_sync(backend, gf->nodes[gf->n_nodes - 1], work_latent->data, 0, ggml_nbytes(work_latent));
    }

    void end() {
        ggml_allocr_free(compute_alloc);
        ggml_backend_buffer_free(compute_buffer);
        compute_alloc              = NULL;
        compute_memory_buffer_size = -1;
    }
};

/*================================================== AutoEncoderKL ===================================================*/

struct ResnetBlock {
    // network hparams
    int in_channels;
    int out_channels;

    // network params
    struct ggml_tensor* norm1_w;  // [in_channels, ]
    struct ggml_tensor* norm1_b;  // [in_channels, ]

    struct ggml_tensor* conv1_w;  // [out_channels, in_channels, 3, 3]
    struct ggml_tensor* conv1_b;  // [out_channels, ]

    struct ggml_tensor* norm2_w;  // [out_channels, ]
    struct ggml_tensor* norm2_b;  // [out_channels, ]

    struct ggml_tensor* conv2_w;  // [out_channels, out_channels, 3, 3]
    struct ggml_tensor* conv2_b;  // [out_channels, ]

    // nin_shortcut, only if out_channels != in_channels
    struct ggml_tensor* nin_shortcut_w;  // [out_channels, in_channels, 1, 1]
    struct ggml_tensor* nin_shortcut_b;  // [out_channels, ]

    size_t calculate_mem_size(ggml_type wtype) {
        double mem_size = 0;
        mem_size += 2 * in_channels * ggml_type_sizef(GGML_TYPE_F32);                      // norm1_w/b
        mem_size += out_channels * in_channels * 3 * 3 * ggml_type_sizef(GGML_TYPE_F16);   // conv1_w
        mem_size += 4 * out_channels * ggml_type_sizef(GGML_TYPE_F32);                     // conv1_b/norm2_w/norm2_b/conv2_b
        mem_size += out_channels * out_channels * 3 * 3 * ggml_type_sizef(GGML_TYPE_F16);  // conv2_w

        if (out_channels != in_channels) {
            mem_size += out_channels * in_channels * 1 * 1 * ggml_type_sizef(GGML_TYPE_F16);  // nin_shortcut_w
            mem_size += out_channels * ggml_type_sizef(GGML_TYPE_F32);                        // nin_shortcut_b
        }
        return static_cast<size_t>(mem_size);
    }

    void init_params(struct ggml_context* ctx, ggml_type wtype) {
        norm1_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);
        norm1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);
        conv1_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, in_channels, out_channels);
        conv1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);

        norm2_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
        norm2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
        conv2_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, out_channels, out_channels);
        conv2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);

        if (out_channels != in_channels) {
            nin_shortcut_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 1, 1, in_channels, out_channels);
            nin_shortcut_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
        }
    }

    void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        tensors[prefix + "norm1.weight"] = norm1_w;
        tensors[prefix + "norm1.bias"]   = norm1_b;
        tensors[prefix + "conv1.weight"] = conv1_w;
        tensors[prefix + "conv1.bias"]   = conv1_b;

        tensors[prefix + "norm2.weight"] = norm2_w;
        tensors[prefix + "norm2.bias"]   = norm2_b;
        tensors[prefix + "conv2.weight"] = conv2_w;
        tensors[prefix + "conv2.bias"]   = conv2_b;

        if (out_channels != in_channels) {
            tensors[prefix + "nin_shortcut.weight"] = nin_shortcut_w;
            tensors[prefix + "nin_shortcut.bias"]   = nin_shortcut_b;
        }
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* z) {
        // z: [N, in_channels, h, w]

        auto h = ggml_nn_group_norm(ctx, z, norm1_w, norm1_b);
        h      = ggml_silu_inplace(ctx, h);
        h      = ggml_nn_conv_2d(ctx, h, conv1_w, conv1_b, 1, 1, 1, 1);  // [N, out_channels, h, w]
        h      = ggml_nn_group_norm(ctx, h, norm2_w, norm2_b);
        h      = ggml_silu_inplace(ctx, h);
        // dropout, skip for inference
        h = ggml_nn_conv_2d(ctx, h, conv2_w, conv2_b, 1, 1, 1, 1);  // [N, out_channels, h, w]

        // skip connection
        if (out_channels != in_channels) {
            z = ggml_nn_conv_2d(ctx, z, nin_shortcut_w, nin_shortcut_b);  // [N, out_channels, h, w]
        }

        h = ggml_add(ctx, h, z);
        return h;  // [N, out_channels, h, w]
    }
};

struct AttnBlock {
    int in_channels;  // mult * model_channels

    // group norm
    struct ggml_tensor* norm_w;  // [in_channels,]
    struct ggml_tensor* norm_b;  // [in_channels,]

    // q/k/v
    struct ggml_tensor* q_w;  // [in_channels, in_channels, 1, 1]
    struct ggml_tensor* q_b;  // [in_channels,]
    struct ggml_tensor* k_w;  // [in_channels, in_channels, 1, 1]
    struct ggml_tensor* k_b;  // [in_channels,]
    struct ggml_tensor* v_w;  // [in_channels, in_channels, 1, 1]
    struct ggml_tensor* v_b;  // [in_channels,]

    // proj_out
    struct ggml_tensor* proj_out_w;  // [in_channels, in_channels, 1, 1]
    struct ggml_tensor* proj_out_b;  // [in_channels,]

    struct ggml_tensor* attn_scale;

    size_t calculate_mem_size(ggml_type wtype) {
        double mem_size = 0;
        mem_size += 6 * in_channels * ggml_type_sizef(GGML_TYPE_F32);                        // norm_w/norm_b/q_b/k_v/v_b/proj_out_b
        mem_size += 4 * in_channels * in_channels * 1 * 1 * ggml_type_sizef(GGML_TYPE_F16);  // q_w/k_w/v_w/proj_out_w                                            // object overhead
        return static_cast<size_t>(mem_size);
    }

    void init_params(struct ggml_context* ctx, ggml_allocr* alloc, ggml_type wtype) {
        norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);
        norm_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);

        q_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 1, 1, in_channels, in_channels);
        q_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);
        k_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 1, 1, in_channels, in_channels);
        k_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);
        v_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 1, 1, in_channels, in_channels);
        v_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);

        proj_out_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 1, 1, in_channels, in_channels);
        proj_out_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);

        attn_scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        ggml_allocr_alloc(alloc, attn_scale);
        float scale = 1.0f / sqrt((float)in_channels);
        ggml_backend_tensor_set(attn_scale, &scale, 0, sizeof(scale));
    }

    void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        tensors[prefix + "norm.weight"]     = norm_w;
        tensors[prefix + "norm.bias"]       = norm_b;
        tensors[prefix + "q.weight"]        = q_w;
        tensors[prefix + "q.bias"]          = q_b;
        tensors[prefix + "k.weight"]        = k_w;
        tensors[prefix + "k.bias"]          = k_b;
        tensors[prefix + "v.weight"]        = v_w;
        tensors[prefix + "v.bias"]          = v_b;
        tensors[prefix + "proj_out.weight"] = proj_out_w;
        tensors[prefix + "proj_out.bias"]   = proj_out_b;
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [N, in_channels, h, w]

        auto h_ = ggml_nn_group_norm(ctx, x, norm_w, norm_b);

        const int64_t n = h_->ne[3];
        const int64_t c = h_->ne[2];
        const int64_t h = h_->ne[1];
        const int64_t w = h_->ne[0];

        auto q = ggml_nn_conv_2d(ctx, h_, q_w, q_b);  // [N, in_channels, h, w]
        auto k = ggml_nn_conv_2d(ctx, h_, k_w, k_b);  // [N, in_channels, h, w]
        auto v = ggml_nn_conv_2d(ctx, h_, v_w, v_b);  // [N, in_channels, h, w]

        q = ggml_cont(ctx, ggml_permute(ctx, q, 1, 2, 0, 3));  // [N, h, w, in_channels]
        q = ggml_reshape_3d(ctx, q, c, h * w, n);              // [N, h * w, in_channels]

        k = ggml_cont(ctx, ggml_permute(ctx, k, 1, 2, 0, 3));  // [N, h, w, in_channels]
        k = ggml_reshape_3d(ctx, k, c, h * w, n);              // [N, h * w, in_channels]

        auto w_ = ggml_mul_mat(ctx, k, q);  // [N, h * w, h * w]
        w_      = ggml_scale_inplace(ctx, w_, attn_scale);
        w_      = ggml_soft_max_inplace(ctx, w_);

        v  = ggml_reshape_3d(ctx, v, h * w, c, n);               // [N, in_channels, h * w]
        h_ = ggml_mul_mat(ctx, v, w_);                           // [N, h * w, in_channels]
        h_ = ggml_cont(ctx, ggml_permute(ctx, h_, 1, 0, 2, 3));  // [N, in_channels, h * w]
        h_ = ggml_reshape_4d(ctx, h_, w, h, c, n);               // [N, in_channels, h, w]

        // proj_out
        h_ = ggml_nn_conv_2d(ctx, h_, proj_out_w, proj_out_b);  // [N, in_channels, h, w]

        h_ = ggml_add(ctx, h_, x);
        return h_;
    }
};

// ldm.modules.diffusionmodules.model.Encoder
struct Encoder {
    int embed_dim      = 4;
    int ch             = 128;
    int z_channels     = 4;
    int in_channels    = 3;
    int num_res_blocks = 2;
    int ch_mult[4]     = {1, 2, 4, 4};

    struct ggml_tensor* conv_in_w;  // [ch, in_channels, 3, 3]
    struct ggml_tensor* conv_in_b;  // [ch, ]

    ResnetBlock down_blocks[4][2];
    DownSample down_samples[3];

    struct
    {
        ResnetBlock block_1;
        AttnBlock attn_1;
        ResnetBlock block_2;
    } mid;

    // block_in = ch * ch_mult[len_mults - 1]
    struct ggml_tensor* norm_out_w;  // [block_in, ]
    struct ggml_tensor* norm_out_b;  // [block_in, ]

    struct ggml_tensor* conv_out_w;  // [embed_dim*2, block_in, 3, 3]
    struct ggml_tensor* conv_out_b;  // [embed_dim*2, ]

    Encoder() {
        int len_mults = sizeof(ch_mult) / sizeof(int);

        int block_in = 1;
        for (int i = 0; i < len_mults; i++) {
            if (i == 0) {
                block_in = ch;
            } else {
                block_in = ch * ch_mult[i - 1];
            }
            int block_out = ch * ch_mult[i];
            for (int j = 0; j < num_res_blocks; j++) {
                down_blocks[i][j].in_channels  = block_in;
                down_blocks[i][j].out_channels = block_out;
                block_in                       = block_out;
            }
            if (i != len_mults - 1) {
                down_samples[i].channels       = block_in;
                down_samples[i].out_channels   = block_in;
                down_samples[i].vae_downsample = true;
            }
        }

        mid.block_1.in_channels  = block_in;
        mid.block_1.out_channels = block_in;
        mid.attn_1.in_channels   = block_in;
        mid.block_2.in_channels  = block_in;
        mid.block_2.out_channels = block_in;
    }

    size_t get_num_tensors() {
        int num_tensors = 6;

        // mid
        num_tensors += 10 * 3;

        int len_mults = sizeof(ch_mult) / sizeof(int);
        for (int i = len_mults - 1; i >= 0; i--) {
            for (int j = 0; j < num_res_blocks + 1; j++) {
                num_tensors += 10;
            }

            if (i != 0) {
                num_tensors += 2;
            }
        }
        return num_tensors;
    }

    size_t calculate_mem_size(ggml_type wtype) {
        double mem_size = 0;
        int len_mults   = sizeof(ch_mult) / sizeof(int);
        int block_in    = ch * ch_mult[len_mults - 1];

        mem_size += ch * in_channels * 3 * 3 * ggml_type_sizef(GGML_TYPE_F16);  // conv_in_w
        mem_size += ch * ggml_type_sizef(GGML_TYPE_F32);                        // conv_in_b

        mem_size += 2 * block_in * ggml_type_sizef(GGML_TYPE_F32);  // norm_out_w/b

        mem_size += z_channels * 2 * block_in * 3 * 3 * ggml_type_sizef(GGML_TYPE_F16);  // conv_out_w
        mem_size += z_channels * 2 * ggml_type_sizef(GGML_TYPE_F32);                     // conv_out_b

        mem_size += mid.block_1.calculate_mem_size(wtype);
        mem_size += mid.attn_1.calculate_mem_size(wtype);
        mem_size += mid.block_2.calculate_mem_size(wtype);

        for (int i = len_mults - 1; i >= 0; i--) {
            for (int j = 0; j < num_res_blocks + 1; j++) {
                mem_size += down_blocks[i][j].calculate_mem_size(wtype);
            }
            if (i != 0) {
                mem_size += down_samples[i - 1].calculate_mem_size(wtype);
            }
        }

        return static_cast<size_t>(mem_size);
    }

    void init_params(struct ggml_context* ctx, ggml_allocr* alloc, ggml_type wtype) {
        int len_mults = sizeof(ch_mult) / sizeof(int);
        int block_in  = ch * ch_mult[len_mults - 1];

        conv_in_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, in_channels, ch);
        conv_in_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ch);

        norm_out_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, block_in);
        norm_out_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, block_in);

        conv_out_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, block_in, z_channels * 2);
        conv_out_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, z_channels * 2);

        mid.block_1.init_params(ctx, wtype);
        mid.attn_1.init_params(ctx, alloc, wtype);
        mid.block_2.init_params(ctx, wtype);

        for (int i = 0; i < len_mults; i++) {
            for (int j = 0; j < num_res_blocks; j++) {
                down_blocks[i][j].init_params(ctx, wtype);
            }
            if (i != len_mults - 1) {
                down_samples[i].init_params(ctx, wtype);
            }
        }
    }

    void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        tensors[prefix + "norm_out.weight"] = norm_out_w;
        tensors[prefix + "norm_out.bias"]   = norm_out_b;
        tensors[prefix + "conv_in.weight"]  = conv_in_w;
        tensors[prefix + "conv_in.bias"]    = conv_in_b;
        tensors[prefix + "conv_out.weight"] = conv_out_w;
        tensors[prefix + "conv_out.bias"]   = conv_out_b;

        mid.block_1.map_by_name(tensors, prefix + "mid.block_1.");
        mid.attn_1.map_by_name(tensors, prefix + "mid.attn_1.");
        mid.block_2.map_by_name(tensors, prefix + "mid.block_2.");

        int len_mults = sizeof(ch_mult) / sizeof(int);
        for (int i = 0; i < len_mults; i++) {
            for (int j = 0; j < num_res_blocks; j++) {
                down_blocks[i][j].map_by_name(tensors, prefix + "down." + std::to_string(i) + ".block." + std::to_string(j) + ".");
            }
            if (i != len_mults - 1) {
                down_samples[i].map_by_name(tensors, prefix + "down." + std::to_string(i) + ".downsample.");
            }
        }
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [N, in_channels, h, w]

        // conv_in
        auto h = ggml_nn_conv_2d(ctx, x, conv_in_w, conv_in_b, 1, 1, 1, 1);  // [N, ch, h, w]
        ggml_set_name(h, "b-start");
        int len_mults = sizeof(ch_mult) / sizeof(int);
        for (int i = 0; i < len_mults; i++) {
            for (int j = 0; j < num_res_blocks; j++) {
                h = down_blocks[i][j].forward(ctx, h);
            }
            if (i != len_mults - 1) {
                h = down_samples[i].forward(ctx, h);
            }
        }

        h = mid.block_1.forward(ctx, h);
        h = mid.attn_1.forward(ctx, h);
        h = mid.block_2.forward(ctx, h);  // [N, block_in, h, w]

        h = ggml_nn_group_norm(ctx, h, norm_out_w, norm_out_b);
        h = ggml_silu_inplace(ctx, h);

        // conv_out
        h = ggml_nn_conv_2d(ctx, h, conv_out_w, conv_out_b, 1, 1, 1, 1);  // [N, z_channels*2, h, w]

        return h;
    }
};

// ldm.modules.diffusionmodules.model.Decoder
struct Decoder {
    int embed_dim      = 4;
    int ch             = 128;
    int z_channels     = 4;
    int out_ch         = 3;
    int num_res_blocks = 2;
    int ch_mult[4]     = {1, 2, 4, 4};

    // block_in = ch *  ch_mult[-1], 512
    struct ggml_tensor* conv_in_w;  // [block_in, z_channels, 3, 3]
    struct ggml_tensor* conv_in_b;  // [block_in, ]

    struct
    {
        ResnetBlock block_1;
        AttnBlock attn_1;
        ResnetBlock block_2;
    } mid;

    ResnetBlock up_blocks[4][3];
    UpSample up_samples[3];

    struct ggml_tensor* norm_out_w;  // [ch *  ch_mult[0], ]
    struct ggml_tensor* norm_out_b;  // [ch *  ch_mult[0], ]

    struct ggml_tensor* conv_out_w;  // [out_ch, ch *  ch_mult[0], 3, 3]
    struct ggml_tensor* conv_out_b;  // [out_ch, ]

    Decoder() {
        int len_mults = sizeof(ch_mult) / sizeof(int);
        int block_in  = ch * ch_mult[len_mults - 1];

        mid.block_1.in_channels  = block_in;
        mid.block_1.out_channels = block_in;
        mid.attn_1.in_channels   = block_in;
        mid.block_2.in_channels  = block_in;
        mid.block_2.out_channels = block_in;

        for (int i = len_mults - 1; i >= 0; i--) {
            int mult      = ch_mult[i];
            int block_out = ch * mult;
            for (int j = 0; j < num_res_blocks + 1; j++) {
                up_blocks[i][j].in_channels  = block_in;
                up_blocks[i][j].out_channels = block_out;
                block_in                     = block_out;
            }
            if (i != 0) {
                up_samples[i - 1].channels     = block_in;
                up_samples[i - 1].out_channels = block_in;
            }
        }
    }

    size_t calculate_mem_size(ggml_type wtype) {
        double mem_size = 0;
        int len_mults   = sizeof(ch_mult) / sizeof(int);
        int block_in    = ch * ch_mult[len_mults - 1];

        mem_size += block_in * z_channels * 3 * 3 * ggml_type_sizef(GGML_TYPE_F16);  // conv_in_w
        mem_size += block_in * ggml_type_sizef(GGML_TYPE_F32);                       // conv_in_b

        mem_size += 2 * (ch * ch_mult[0]) * ggml_type_sizef(GGML_TYPE_F32);  // norm_out_w/b

        mem_size += (ch * ch_mult[0]) * out_ch * 3 * 3 * ggml_type_sizef(GGML_TYPE_F16);  // conv_out_w
        mem_size += out_ch * ggml_type_sizef(GGML_TYPE_F32);                              // conv_out_b

        mem_size += mid.block_1.calculate_mem_size(wtype);
        mem_size += mid.attn_1.calculate_mem_size(wtype);
        mem_size += mid.block_2.calculate_mem_size(wtype);

        for (int i = len_mults - 1; i >= 0; i--) {
            for (int j = 0; j < num_res_blocks + 1; j++) {
                mem_size += up_blocks[i][j].calculate_mem_size(wtype);
            }
            if (i != 0) {
                mem_size += up_samples[i - 1].calculate_mem_size(wtype);
            }
        }

        return static_cast<size_t>(mem_size);
    }

    size_t get_num_tensors() {
        int num_tensors = 8;

        // mid
        num_tensors += 10 * 3;

        int len_mults = sizeof(ch_mult) / sizeof(int);
        for (int i = len_mults - 1; i >= 0; i--) {
            for (int j = 0; j < num_res_blocks + 1; j++) {
                num_tensors += 10;
            }

            if (i != 0) {
                num_tensors += 2;
            }
        }
        return num_tensors;
    }

    void init_params(struct ggml_context* ctx, ggml_allocr* alloc, ggml_type wtype) {
        int len_mults = sizeof(ch_mult) / sizeof(int);
        int block_in  = ch * ch_mult[len_mults - 1];

        norm_out_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ch * ch_mult[0]);
        norm_out_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ch * ch_mult[0]);

        conv_in_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, z_channels, block_in);
        conv_in_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, block_in);

        conv_out_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, ch * ch_mult[0], out_ch);
        conv_out_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_ch);

        mid.block_1.init_params(ctx, wtype);
        mid.attn_1.init_params(ctx, alloc, wtype);
        mid.block_2.init_params(ctx, wtype);

        for (int i = len_mults - 1; i >= 0; i--) {
            for (int j = 0; j < num_res_blocks + 1; j++) {
                up_blocks[i][j].init_params(ctx, wtype);
            }

            if (i != 0) {
                up_samples[i - 1].init_params(ctx, wtype);
            }
        }
    }

    void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        tensors[prefix + "norm_out.weight"] = norm_out_w;
        tensors[prefix + "norm_out.bias"]   = norm_out_b;
        tensors[prefix + "conv_in.weight"]  = conv_in_w;
        tensors[prefix + "conv_in.bias"]    = conv_in_b;
        tensors[prefix + "conv_out.weight"] = conv_out_w;
        tensors[prefix + "conv_out.bias"]   = conv_out_b;

        mid.block_1.map_by_name(tensors, prefix + "mid.block_1.");
        mid.attn_1.map_by_name(tensors, prefix + "mid.attn_1.");
        mid.block_2.map_by_name(tensors, prefix + "mid.block_2.");

        int len_mults = sizeof(ch_mult) / sizeof(int);
        for (int i = len_mults - 1; i >= 0; i--) {
            for (int j = 0; j < num_res_blocks + 1; j++) {
                up_blocks[i][j].map_by_name(tensors, prefix + "up." + std::to_string(i) + ".block." + std::to_string(j) + ".");
            }
            if (i != 0) {
                up_samples[i - 1].map_by_name(tensors, prefix + "up." + std::to_string(i) + ".upsample.");
            }
        }
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* z) {
        // z: [N, z_channels, h, w]
        // conv_in
        auto h = ggml_nn_conv_2d(ctx, z, conv_in_w, conv_in_b, 1, 1, 1, 1);  // [N, block_in, h, w]

        h = mid.block_1.forward(ctx, h);
        h = mid.attn_1.forward(ctx, h);
        h = mid.block_2.forward(ctx, h);  // [N, block_in, h, w]

        int len_mults = sizeof(ch_mult) / sizeof(int);
        for (int i = len_mults - 1; i >= 0; i--) {
            for (int j = 0; j < num_res_blocks + 1; j++) {
                h = up_blocks[i][j].forward(ctx, h);
            }
            if (i != 0) {
                h = up_samples[i - 1].forward(ctx, h);
            }
        }

        // group norm 32
        h = ggml_nn_group_norm(ctx, h, norm_out_w, norm_out_b);
        h = ggml_silu_inplace(ctx, h);

        // conv_out
        h = ggml_nn_conv_2d(ctx, h, conv_out_w, conv_out_b, 1, 1, 1, 1);  // [N, out_ch, h, w]
        return h;
    }
};

// ldm.models.autoencoder.AutoencoderKL
struct AutoEncoderKL {
    bool decode_only = true;
    int embed_dim    = 4;
    struct
    {
        int z_channels     = 4;
        int resolution     = 256;
        int in_channels    = 3;
        int out_ch         = 3;
        int ch             = 128;
        int ch_mult[4]     = {1, 2, 4, 4};
        int num_res_blocks = 2;
    } dd_config;

    struct ggml_tensor* quant_conv_w;  // [2*embed_dim, 2*z_channels, 1, 1]
    struct ggml_tensor* quant_conv_b;  // [2*embed_dim, ]

    struct ggml_tensor* post_quant_conv_w;  // [z_channels, embed_dim, 1, 1]
    struct ggml_tensor* post_quant_conv_b;  // [z_channels, ]

    Encoder encoder;
    Decoder decoder;

    struct ggml_context* ctx;
    ggml_backend_buffer_t params_buffer;
    ggml_backend_buffer_t compute_buffer;  // for compute
    struct ggml_allocr* compute_alloc = NULL;

    int memory_buffer_size = 0;
    ggml_type wtype;
    ggml_backend_t backend = NULL;

    AutoEncoderKL(bool decode_only = false)
        : decode_only(decode_only) {
        assert(sizeof(dd_config.ch_mult) == sizeof(encoder.ch_mult));
        assert(sizeof(dd_config.ch_mult) == sizeof(decoder.ch_mult));

        encoder.embed_dim      = embed_dim;
        decoder.embed_dim      = embed_dim;
        encoder.ch             = dd_config.ch;
        decoder.ch             = dd_config.ch;
        encoder.z_channels     = dd_config.z_channels;
        decoder.z_channels     = dd_config.z_channels;
        encoder.in_channels    = dd_config.in_channels;
        decoder.out_ch         = dd_config.out_ch;
        encoder.num_res_blocks = dd_config.num_res_blocks;

        int len_mults = sizeof(dd_config.ch_mult) / sizeof(int);
        for (int i = 0; i < len_mults; i++) {
            encoder.ch_mult[i] = dd_config.ch_mult[i];
            decoder.ch_mult[i] = dd_config.ch_mult[i];
        }
    }

    size_t calculate_mem_size() {
        double mem_size = 0;

        if (!decode_only) {
            mem_size += 2 * embed_dim * 2 * dd_config.z_channels * 1 * 1 * ggml_type_sizef(GGML_TYPE_F16);  // quant_conv_w
            mem_size += 2 * embed_dim * ggml_type_sizef(GGML_TYPE_F32);                                     // quant_conv_b
            mem_size += encoder.calculate_mem_size(wtype);
        }

        mem_size += dd_config.z_channels * embed_dim * 1 * 1 * ggml_type_sizef(GGML_TYPE_F16);  // post_quant_conv_w
        mem_size += dd_config.z_channels * ggml_type_sizef(GGML_TYPE_F32);                      // post_quant_conv_b

        mem_size += decoder.calculate_mem_size(wtype);
        return static_cast<size_t>(mem_size);
    }

    bool initialize(ggml_backend_t backend_, ggml_type wtype_) {
        backend            = backend_;
        wtype              = wtype_;
        memory_buffer_size = 1 * 1024 * 1024;  // 1 MB, for padding
        memory_buffer_size += (int)calculate_mem_size();
        int num_tensors = 0;
        if (!decode_only) {
            num_tensors += 2;
            num_tensors += (int)encoder.get_num_tensors();
        }

        num_tensors += (int)decoder.get_num_tensors();
        LOG_DEBUG("vae params backend buffer size = % 6.2f MB (%i tensors)", memory_buffer_size / (1024.0 * 1024.0), num_tensors);

        struct ggml_init_params params;
        params.mem_size   = static_cast<size_t>(num_tensors * ggml_tensor_overhead());
        params.mem_buffer = NULL;
        params.no_alloc   = true;
        // LOG_DEBUG("mem_size %u ", params.mem_size);

        params_buffer = ggml_backend_alloc_buffer(backend, memory_buffer_size);

        ctx = ggml_init(params);
        if (!ctx) {
            LOG_ERROR("ggml_init() failed");
            return false;
        }
        return true;
    }

    void destroy() {
        if (ctx != NULL) {
            ggml_free(ctx);
            ctx = NULL;
        }
    }

    void alloc_params() {
        ggml_allocr* alloc = ggml_allocr_new_from_buffer(params_buffer);

        if (!decode_only) {
            quant_conv_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 1, 1, 2 * dd_config.z_channels, 2 * embed_dim);
            quant_conv_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2 * embed_dim);
            encoder.init_params(ctx, alloc, wtype);
        }

        post_quant_conv_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 1, 1, embed_dim, dd_config.z_channels);
        post_quant_conv_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dd_config.z_channels);
        decoder.init_params(ctx, alloc, wtype);

        // alloc all tensors linked to this context
        for (struct ggml_tensor* t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->data == NULL) {
                ggml_allocr_alloc(alloc, t);
            }
        }
        ggml_allocr_free(alloc);
    }

    void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        if (!decode_only) {
            tensors[prefix + "quant_conv.weight"] = quant_conv_w;
            tensors[prefix + "quant_conv.bias"]   = quant_conv_b;
            encoder.map_by_name(tensors, prefix + "encoder.");
        }

        tensors[prefix + "post_quant_conv.weight"] = post_quant_conv_w;
        tensors[prefix + "post_quant_conv.bias"]   = post_quant_conv_b;
        decoder.map_by_name(tensors, prefix + "decoder.");
    }

    struct ggml_tensor* decode(struct ggml_context* ctx0, struct ggml_tensor* z) {
        // z: [N, z_channels, h, w]
        // post_quant_conv
        auto h = ggml_nn_conv_2d(ctx0, z, post_quant_conv_w, post_quant_conv_b);  // [N, z_channels, h, w]
        ggml_set_name(h, "bench-start");
        h = decoder.forward(ctx0, h);
        ggml_set_name(h, "bench-end");
        return h;
    }

    struct ggml_tensor* encode(struct ggml_context* ctx0, struct ggml_tensor* x) {
        // x: [N, in_channels, h, w]
        auto h = encoder.forward(ctx0, x);  // [N, 2*z_channels, h/8, w/8]
        // quant_conv
        h = ggml_nn_conv_2d(ctx0, h, quant_conv_w, quant_conv_b);  // [N, 2*embed_dim, h/8, w/8]
        ggml_set_name(h, "b-end");
        return h;
    }

    struct ggml_cgraph* build_graph(struct ggml_tensor* z, bool decode_graph) {
        // since we are using ggml-alloc, this buffer only needs enough space to hold the ggml_tensor and ggml_cgraph structs, but not the tensor data
        static size_t buf_size = ggml_tensor_overhead() * UNET_GRAPH_SIZE + ggml_graph_overhead();
        static std::vector<uint8_t> buf(buf_size);

        struct ggml_init_params params = {
            /*.mem_size   =*/buf_size,
            /*.mem_buffer =*/buf.data(),
            /*.no_alloc   =*/true,  // the tensors will be allocated later by ggml_allocr_alloc_graph()
        };
        // LOG_DEBUG("mem_size %u ", params.mem_size);

        struct ggml_context* ctx0 = ggml_init(params);

        struct ggml_cgraph* gf = ggml_new_graph(ctx0);

        struct ggml_tensor* z_ = NULL;

        // it's performing a compute, check if backend isn't cpu
        if (!ggml_backend_is_cpu(backend)) {
            // pass input tensors to gpu memory
            z_ = ggml_dup_tensor(ctx0, z);
            ggml_allocr_alloc(compute_alloc, z_);

            // pass data to device backend
            if (!ggml_allocr_is_measure(compute_alloc)) {
                ggml_backend_tensor_set(z_, z->data, 0, ggml_nbytes(z));
            }
        } else {
            z_ = z;
        }

        struct ggml_tensor* out = decode_graph ? decode(ctx0, z_) : encode(ctx0, z_);

        ggml_build_forward_expand(gf, out);
        ggml_free(ctx0);

        return gf;
    }

    void begin(struct ggml_tensor* x, bool decode) {
        // calculate the amount of memory required
        // alignment required by the backend
        compute_alloc = ggml_allocr_new_measure_from_backend(backend);

        struct ggml_cgraph* gf = build_graph(x, decode);

        // compute the required memory
        size_t compute_memory_buffer_size = ggml_allocr_alloc_graph(compute_alloc, gf);

        // recreate the allocator with the required memory
        ggml_allocr_free(compute_alloc);

        LOG_DEBUG("vae compute buffer size: %.2f MB", compute_memory_buffer_size / 1024.0 / 1024.0);

        compute_buffer = ggml_backend_alloc_buffer(backend, compute_memory_buffer_size);
        compute_alloc  = ggml_allocr_new_from_buffer(compute_buffer);
    }

    void compute(struct ggml_tensor* work_result, const int n_threads, struct ggml_tensor* z, bool decode_graph) {
        ggml_allocr_reset(compute_alloc);

        struct ggml_cgraph* gf = build_graph(z, decode_graph);
        ggml_allocr_alloc_graph(compute_alloc, gf);

        if (ggml_backend_is_cpu(backend)) {
            ggml_backend_cpu_set_n_threads(backend, n_threads);
        }

#ifdef SD_USE_METAL
        if (ggml_backend_is_metal(backend)) {
            ggml_backend_metal_set_n_cb(backend, n_threads);
        }
#endif

        ggml_backend_graph_compute(backend, gf);

#ifdef GGML_PERF
        ggml_graph_print(gf);
#endif

        ggml_backend_tensor_get_and_sync(backend, gf->nodes[gf->n_nodes - 1], work_result->data, 0, ggml_nbytes(work_result));
    }

    void end() {
        ggml_allocr_free(compute_alloc);
        ggml_backend_buffer_free(compute_buffer);
        compute_alloc = NULL;
    }
};

/*
    ===================================    TinyAutoEncoder  ===================================
    References:
    https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoder_tiny.py
    https://github.com/madebyollin/taesd/blob/main/taesd.py

*/
struct TAEBlock {
    int in_channels;
    int out_channels;

    // conv
    ggml_tensor* conv_0_w;  // [in_channels, out_channels, 3, 3]
    ggml_tensor* conv_0_b;  // [in_channels]
    ggml_tensor* conv_1_w;  // [out_channels, out_channels, 3, 3]
    ggml_tensor* conv_1_b;  // [out_channels]
    ggml_tensor* conv_2_w;  // [out_channels, out_channels, 3, 3]
    ggml_tensor* conv_2_b;  // [out_channels]

    // skip
    ggml_tensor* conv_skip_w;  // [in_channels, out_channels, 1, 1]

    size_t calculate_mem_size() {
        size_t mem_size = in_channels * out_channels * 3 * 3 * ggml_type_size(GGML_TYPE_F16);  // conv_0_w
        mem_size += in_channels * ggml_type_size(GGML_TYPE_F32);                               // conv_0_b
        mem_size += out_channels * out_channels * 3 * 3 * ggml_type_size(GGML_TYPE_F16);       // conv_1_w
        mem_size += out_channels * ggml_type_size(GGML_TYPE_F32);                              // conv_1_b
        mem_size += out_channels * out_channels * 3 * 3 * ggml_type_size(GGML_TYPE_F16);       // conv_1_w
        mem_size += out_channels * ggml_type_size(GGML_TYPE_F32);                              // conv_1_b
        mem_size += out_channels * out_channels * 3 * 3 * ggml_type_size(GGML_TYPE_F16);       // conv_2_w
        mem_size += out_channels * ggml_type_size(GGML_TYPE_F32);                              // conv_2_b

        if (in_channels != out_channels) {
            mem_size += in_channels * out_channels * ggml_type_size(GGML_TYPE_F16);  // conv_skip_w
        }
        return mem_size;
    }

    int get_num_tensors() {
        return 6 + (in_channels != out_channels ? 1 : 0);
    }

    void init_params(ggml_context* ctx) {
        conv_0_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, out_channels, in_channels);
        conv_0_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);

        conv_1_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, out_channels, out_channels);
        conv_1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);

        conv_2_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, out_channels, out_channels);
        conv_2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);

        if (in_channels != out_channels) {
            conv_skip_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 1, 1, out_channels, in_channels);
        }
    }

    void map_by_name(std::map<std::string, ggml_tensor*>& tensors, std::string prefix) {
        tensors[prefix + "conv.0.weight"] = conv_0_w;
        tensors[prefix + "conv.0.bias"]   = conv_0_b;

        tensors[prefix + "conv.2.weight"] = conv_1_w;
        tensors[prefix + "conv.2.bias"]   = conv_1_b;

        tensors[prefix + "conv.4.weight"] = conv_2_w;
        tensors[prefix + "conv.4.bias"]   = conv_2_b;

        if (in_channels != out_channels) {
            tensors[prefix + "skip.weight"] = conv_skip_w;
        }
    }

    ggml_tensor* forward(ggml_context* ctx, ggml_tensor* x) {
        // conv(n_in, n_out)
        ggml_tensor* h;
        h = ggml_nn_conv_2d(ctx, x, conv_0_w, conv_0_b, 1, 1, 1, 1);
        h = ggml_relu_inplace(ctx, h);
        h = ggml_nn_conv_2d(ctx, h, conv_1_w, conv_1_b, 1, 1, 1, 1);
        h = ggml_relu_inplace(ctx, h);
        h = ggml_nn_conv_2d(ctx, h, conv_2_w, conv_2_b, 1, 1, 1, 1);

        // skip connection
        if (in_channels != out_channels) {
            // skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
            x = ggml_nn_conv_2d(ctx, x, conv_skip_w, NULL, 1, 1, 1, 1);
        }

        h = ggml_add(ctx, h, x);
        h = ggml_relu_inplace(ctx, h);
        return h;
    }
};

struct TinyEncoder {
    int in_channels = 3;
    int z_channels  = 4;
    int channels    = 64;
    int num_blocks  = 3;

    // input
    ggml_tensor* conv_input_w;  // [channels, in_channels, 3, 3]
    ggml_tensor* conv_input_b;  // [channels]
    TAEBlock initial_block;

    ggml_tensor* conv_1_w;  // [channels, channels, 3, 3]
    TAEBlock input_blocks[3];

    // middle
    ggml_tensor* conv_2_w;  // [channels, channels, 3, 3]
    TAEBlock middle_blocks[3];

    // output
    ggml_tensor* conv_3_w;  // [channels, channels, 3, 3]
    TAEBlock output_blocks[3];

    // final
    ggml_tensor* conv_final_w;  // [z_channels, channels, 3, 3]
    ggml_tensor* conv_final_b;  // [z_channels]

    TinyEncoder() {
        for (int i = 0; i < num_blocks; i++) {
            input_blocks[i].in_channels  = channels;
            input_blocks[i].out_channels = channels;

            middle_blocks[i].in_channels  = channels;
            middle_blocks[i].out_channels = channels;

            output_blocks[i].in_channels  = channels;
            output_blocks[i].out_channels = channels;
        }

        initial_block.in_channels  = channels;
        initial_block.out_channels = channels;
    }

    size_t calculate_mem_size() {
        size_t mem_size = channels * in_channels * 3 * 3 * ggml_type_size(GGML_TYPE_F16);  // conv_input_w
        mem_size += channels * ggml_type_size(GGML_TYPE_F32);                              // conv_input_b

        mem_size += initial_block.calculate_mem_size();

        mem_size += channels * channels * 3 * 3 * ggml_type_size(GGML_TYPE_F16);  // conv_1_w
        mem_size += channels * channels * 3 * 3 * ggml_type_size(GGML_TYPE_F16);  // conv_2_w
        mem_size += channels * channels * 3 * 3 * ggml_type_size(GGML_TYPE_F16);  // conv_3_w

        for (int i = 0; i < num_blocks; i++) {
            mem_size += input_blocks[i].calculate_mem_size();
            mem_size += middle_blocks[i].calculate_mem_size();
            mem_size += output_blocks[i].calculate_mem_size();
        }
        mem_size += z_channels * channels * 3 * 3 * ggml_type_size(GGML_TYPE_F16);  // conv_input_w
        mem_size += z_channels * ggml_type_size(GGML_TYPE_F32);                     // conv_input_b
        return mem_size;
    }

    int get_num_tensors() {
        int num_tensors = 7;
        for (int i = 0; i < num_blocks; i++) {
            num_tensors += input_blocks[i].get_num_tensors();
            num_tensors += middle_blocks[i].get_num_tensors();
            num_tensors += output_blocks[i].get_num_tensors();
        }
        num_tensors += initial_block.get_num_tensors();
        return num_tensors;
    }

    void init_params(ggml_context* ctx) {
        conv_input_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, in_channels, channels);
        conv_input_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, channels);

        initial_block.init_params(ctx);

        conv_1_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, channels, channels);
        conv_2_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, channels, channels);
        conv_3_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, channels, channels);

        conv_final_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, channels, z_channels);
        conv_final_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, z_channels);

        for (int i = 0; i < num_blocks; i++) {
            input_blocks[i].init_params(ctx);
            middle_blocks[i].init_params(ctx);
            output_blocks[i].init_params(ctx);
        }
    }

    void map_by_name(std::map<std::string, ggml_tensor*>& tensors, std::string prefix) {
        tensors[prefix + "0.weight"] = conv_input_w;
        tensors[prefix + "0.bias"]   = conv_input_b;

        initial_block.map_by_name(tensors, prefix + "1.");

        tensors[prefix + "2.weight"] = conv_1_w;
        for (int i = 0; i < num_blocks; i++) {
            input_blocks[i].map_by_name(tensors, prefix + std::to_string(i + 3) + ".");
        }

        tensors[prefix + "6.weight"] = conv_2_w;
        for (int i = 0; i < num_blocks; i++) {
            middle_blocks[i].map_by_name(tensors, prefix + std::to_string(i + 7) + ".");
        }

        tensors[prefix + "10.weight"] = conv_3_w;
        for (int i = 0; i < num_blocks; i++) {
            output_blocks[i].map_by_name(tensors, prefix + std::to_string(i + 11) + ".");
        }

        tensors[prefix + "14.weight"] = conv_final_w;
        tensors[prefix + "14.bias"]   = conv_final_b;
    }

    ggml_tensor* forward(ggml_context* ctx, ggml_tensor* x) {
        // conv(3, 64)
        auto z = ggml_nn_conv_2d(ctx, x, conv_input_w, conv_input_b, 1, 1, 1, 1);

        // Block(64, 64)
        z = initial_block.forward(ctx, z);

        // conv(64, 64, stride=2, bias=False)
        z = ggml_nn_conv_2d(ctx, z, conv_1_w, NULL, 2, 2, 1, 1);

        // Block(64, 64), Block(64, 64), Block(64, 64)
        for (int i = 0; i < num_blocks; i++) {
            z = input_blocks[i].forward(ctx, z);
        }

        // conv(64, 64, stride=2, bias=False)
        z = ggml_nn_conv_2d(ctx, z, conv_2_w, NULL, 2, 2, 1, 1);

        // Block(64, 64), Block(64, 64), Block(64, 64)
        for (int i = 0; i < num_blocks; i++) {
            z = middle_blocks[i].forward(ctx, z);
        }

        // conv(64, 64, stride=2, bias=False)
        z = ggml_nn_conv_2d(ctx, z, conv_3_w, NULL, 2, 2, 1, 1);

        // Block(64, 64), Block(64, 64), Block(64, 64)
        for (int i = 0; i < num_blocks; i++) {
            z = output_blocks[i].forward(ctx, z);
        }

        // conv(64, 4)
        z = ggml_nn_conv_2d(ctx, z, conv_final_w, conv_final_b, 1, 1, 1, 1);
        return z;
    }
};

struct TinyDecoder {
    int z_channels      = 4;
    int channels        = 64;
    int output_channels = 3;
    int num_blocks      = 3;

    // input
    ggml_tensor* conv_input_w;  // [channels, z_channels, 3, 3]
    ggml_tensor* conv_input_b;  // [channels]
    TAEBlock input_blocks[3];
    ggml_tensor* conv_1_w;  // [channels, channels, 3, 3]

    // middle
    TAEBlock middle_blocks[3];
    ggml_tensor* conv_2_w;  // [channels, channels, 3, 3]

    // output
    TAEBlock output_blocks[3];
    ggml_tensor* conv_3_w;  // [channels, channels, 3, 3]

    // final
    TAEBlock final_block;
    ggml_tensor* conv_final_w;  // [output_channels, channels, 3, 3]
    ggml_tensor* conv_final_b;  // [output_channels]

    ggml_tensor* in_scale_1d3;  // [1]
    ggml_tensor* in_scale_3;    // [1]

    TinyDecoder() {
        for (int i = 0; i < num_blocks; i++) {
            input_blocks[i].in_channels  = channels;
            input_blocks[i].out_channels = channels;

            middle_blocks[i].in_channels  = channels;
            middle_blocks[i].out_channels = channels;

            output_blocks[i].in_channels  = channels;
            output_blocks[i].out_channels = channels;
        }

        final_block.in_channels  = channels;
        final_block.out_channels = channels;
    }

    size_t calculate_mem_size() {
        size_t mem_size = channels * z_channels * 3 * 3 * ggml_type_size(GGML_TYPE_F16);  // conv_input_w
        mem_size += channels * ggml_type_size(GGML_TYPE_F32);                             // conv_input_b

        for (int i = 0; i < num_blocks; i++) {
            mem_size += input_blocks[i].calculate_mem_size();
        }
        mem_size += channels * channels * 3 * 3 * ggml_type_size(GGML_TYPE_F16);  // conv_1_w

        for (int i = 0; i < num_blocks; i++) {
            mem_size += middle_blocks[i].calculate_mem_size();
        }
        mem_size += channels * channels * 3 * 3 * ggml_type_size(GGML_TYPE_F16);  // conv_2_w

        for (int i = 0; i < num_blocks; i++) {
            mem_size += output_blocks[i].calculate_mem_size();
        }
        mem_size += channels * channels * 3 * 3 * ggml_type_size(GGML_TYPE_F16);  // conv_3_w

        mem_size += final_block.calculate_mem_size();
        mem_size += output_channels * channels * 3 * 3 * ggml_type_size(GGML_TYPE_F16);  // conv_input_w
        mem_size += output_channels * ggml_type_size(GGML_TYPE_F32);                     // conv_input_b
        return mem_size;
    }

    int get_num_tensors() {
        int num_tensors = 9;
        for (int i = 0; i < num_blocks; i++) {
            num_tensors += input_blocks[i].get_num_tensors();
            num_tensors += middle_blocks[i].get_num_tensors();
            num_tensors += output_blocks[i].get_num_tensors();
        }
        num_tensors += final_block.get_num_tensors();
        return num_tensors;
    }

    void init_params(ggml_allocr* alloc, ggml_context* ctx) {
        conv_input_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, z_channels, channels);
        conv_input_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, channels);

        conv_1_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, channels, channels);
        conv_2_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, channels, channels);
        conv_3_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, channels, channels);

        conv_final_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, channels, output_channels);
        conv_final_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, output_channels);

        for (int i = 0; i < num_blocks; i++) {
            input_blocks[i].init_params(ctx);
            middle_blocks[i].init_params(ctx);
            output_blocks[i].init_params(ctx);
        }

        final_block.init_params(ctx);

        // initialize constants scales
        in_scale_1d3 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        in_scale_3   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        ggml_allocr_alloc(alloc, in_scale_1d3);
        float scale_1d3 = 1.0f / 3.0f;
        ggml_backend_tensor_set(in_scale_1d3, &scale_1d3, 0, sizeof(scale_1d3));
        ggml_allocr_alloc(alloc, in_scale_3);
        float scale_3 = 3.0f;
        ggml_backend_tensor_set(in_scale_3, &scale_3, 0, sizeof(scale_3));
    }

    void map_by_name(std::map<std::string, ggml_tensor*>& tensors, std::string prefix) {
        tensors[prefix + "0.weight"] = conv_input_w;
        tensors[prefix + "0.bias"]   = conv_input_b;

        for (int i = 0; i < num_blocks; i++) {
            input_blocks[i].map_by_name(tensors, prefix + std::to_string(i + 2) + ".");
        }

        tensors[prefix + "6.weight"] = conv_1_w;
        for (int i = 0; i < num_blocks; i++) {
            middle_blocks[i].map_by_name(tensors, prefix + std::to_string(i + 7) + ".");
        }

        tensors[prefix + "11.weight"] = conv_2_w;
        for (int i = 0; i < num_blocks; i++) {
            output_blocks[i].map_by_name(tensors, prefix + std::to_string(i + 12) + ".");
        }

        tensors[prefix + "16.weight"] = conv_3_w;

        final_block.map_by_name(tensors, prefix + "17.");

        tensors[prefix + "18.weight"] = conv_final_w;
        tensors[prefix + "18.bias"]   = conv_final_b;
    }

    ggml_tensor* forward(ggml_context* ctx, ggml_tensor* z) {
        // torch.tanh(x / 3) * 3
        auto h = ggml_scale(ctx, z, in_scale_1d3);
        h      = ggml_tanh_inplace(ctx, h);
        h      = ggml_scale(ctx, h, in_scale_3);

        // conv(4, 64)
        h = ggml_nn_conv_2d(ctx, h, conv_input_w, conv_input_b, 1, 1, 1, 1);

        // nn.ReLU()
        h = ggml_relu_inplace(ctx, h);

        // Block(64, 64), Block(64, 64), Block(64, 64)
        for (int i = 0; i < num_blocks; i++) {
            h = input_blocks[i].forward(ctx, h);
        }

        // nn.Upsample(scale_factor=2)
        h = ggml_upscale(ctx, h, 2);

        // conv(64, 64, bias=False)
        h = ggml_nn_conv_2d(ctx, h, conv_1_w, NULL, 1, 1, 1, 1);

        // Block(64, 64), Block(64, 64), Block(64, 64)
        for (int i = 0; i < num_blocks; i++) {
            h = middle_blocks[i].forward(ctx, h);
        }

        // nn.Upsample(scale_factor=2)
        h = ggml_upscale(ctx, h, 2);

        // conv(64, 64, bias=False)
        h = ggml_nn_conv_2d(ctx, h, conv_2_w, NULL, 1, 1, 1, 1);

        // Block(64, 64), Block(64, 64), Block(64, 64)
        for (int i = 0; i < num_blocks; i++) {
            h = output_blocks[i].forward(ctx, h);
        }

        // nn.Upsample(scale_factor=2)
        h = ggml_upscale(ctx, h, 2);

        // conv(64, 64, bias=False)
        h = ggml_nn_conv_2d(ctx, h, conv_3_w, NULL, 1, 1, 1, 1);

        // Block(64, 64)
        h = final_block.forward(ctx, h);

        // conv(64, 3)
        h = ggml_nn_conv_2d(ctx, h, conv_final_w, conv_final_b, 1, 1, 1, 1);
        return h;
    }
};

struct TinyAutoEncoder {
    TinyEncoder encoder;
    TinyDecoder decoder;

    ggml_context* ctx;
    bool decode_only = false;
    ggml_backend_buffer_t params_buffer;
    ggml_backend_buffer_t compute_buffer;  // for compute
    struct ggml_allocr* compute_alloc = NULL;

    int memory_buffer_size = 0;
    ggml_type wtype;
    ggml_backend_t backend = NULL;

    TinyAutoEncoder(bool decoder_only_ = true)
        : decode_only(decoder_only_) {
        decoder = TinyDecoder();
        if (!decoder_only_) {
            encoder = TinyEncoder();
        }
    }

    size_t calculate_mem_size() {
        size_t mem_size = decoder.calculate_mem_size();
        if (!decode_only) {
            mem_size += encoder.calculate_mem_size();
        }
        mem_size += 1024;  // padding
        return mem_size;
    }

    bool init(ggml_backend_t backend_) {
        backend            = backend_;
        memory_buffer_size = calculate_mem_size();
        int num_tensors    = decoder.get_num_tensors();
        if (!decode_only) {
            num_tensors += encoder.get_num_tensors();
        }

        LOG_DEBUG("TAE params backend buffer size = % 6.2f MB (%i tensors)", memory_buffer_size / (1024.0 * 1024.0), num_tensors);

        struct ggml_init_params params;
        params.mem_size   = static_cast<size_t>(num_tensors * ggml_tensor_overhead());
        params.mem_buffer = NULL;
        params.no_alloc   = true;
        // LOG_DEBUG("mem_size %u ", params.mem_size);

        params_buffer = ggml_backend_alloc_buffer(backend, memory_buffer_size);

        ctx = ggml_init(params);
        if (!ctx) {
            LOG_ERROR("ggml_init() failed");
            return false;
        }
        return true;
    }

    void alloc_params() {
        ggml_allocr* alloc = ggml_allocr_new_from_buffer(params_buffer);
        decoder.init_params(alloc, ctx);
        if (!decode_only) {
            encoder.init_params(ctx);
        }

        // alloc all tensors linked to this context
        for (struct ggml_tensor* t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->data == NULL) {
                ggml_allocr_alloc(alloc, t);
            }
        }
        ggml_allocr_free(alloc);
    }

    void map_by_name(std::map<std::string, ggml_tensor*>& tensors) {
        decoder.map_by_name(tensors, "decoder.layers.");
        if (!decode_only) {
            encoder.map_by_name(tensors, "encoder.layers.");
        }
    }

    bool load_from_file(const std::string& file_path, ggml_backend_t backend) {
        LOG_INFO("loading taesd from '%s'", file_path.c_str());

        if (!init(backend)) {
            return false;
        }

        std::map<std::string, ggml_tensor*> taesd_tensors;

        ModelLoader model_loader;
        if (!model_loader.init_from_file(file_path)) {
            LOG_ERROR("init taesd model loader from file failed: '%s'", file_path.c_str());
            return false;
        }

        // prepare memory for the weights
        {
            alloc_params();
            map_by_name(taesd_tensors);
        }

        std::set<std::string> tensor_names_in_file;

        auto on_new_tensor_cb = [&](const TensorStorage& tensor_storage, ggml_tensor** dst_tensor) -> bool {
            const std::string& name = tensor_storage.name;
            tensor_names_in_file.insert(name);

            struct ggml_tensor* real;
            if (taesd_tensors.find(name) != taesd_tensors.end()) {
                real = taesd_tensors[name];
            } else {
                if (name.find("encoder.") != std::string::npos && decode_only) {
                    return true;
                }
                LOG_ERROR("unknown tensor '%s' in model file", name.data());
                return true;
            }

            if (
                real->ne[0] != tensor_storage.ne[0] ||
                real->ne[1] != tensor_storage.ne[1] ||
                real->ne[2] != tensor_storage.ne[2] ||
                real->ne[3] != tensor_storage.ne[3]) {
                LOG_ERROR(
                    "tensor '%s' has wrong shape in model file: "
                    "got [%d, %d, %d, %d], expected [%d, %d, %d, %d]",
                    name.c_str(),
                    (int)tensor_storage.ne[0], (int)tensor_storage.ne[1], (int)tensor_storage.ne[2], (int)tensor_storage.ne[3],
                    (int)real->ne[0], (int)real->ne[1], (int)real->ne[2], (int)real->ne[3]);
                return false;
            }

            *dst_tensor = real;

            return true;
        };

        bool success = model_loader.load_tensors(on_new_tensor_cb, backend);

        bool some_tensor_not_init = false;

        for (auto pair : taesd_tensors) {
            if (tensor_names_in_file.find(pair.first) == tensor_names_in_file.end()) {
                LOG_ERROR("tensor '%s' not in model file", pair.first.c_str());
                some_tensor_not_init = true;
            }
        }

        if (some_tensor_not_init) {
            return false;
        }

        LOG_INFO("taesd model loaded");
        return success;
    }

    struct ggml_cgraph* build_graph(struct ggml_tensor* z, bool decode_graph) {
        // since we are using ggml-alloc, this buffer only needs enough space to hold the ggml_tensor and ggml_cgraph structs, but not the tensor data
        static size_t buf_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
        static std::vector<uint8_t> buf(buf_size);

        struct ggml_init_params params = {
            /*.mem_size   =*/buf_size,
            /*.mem_buffer =*/buf.data(),
            /*.no_alloc   =*/true,  // the tensors will be allocated later by ggml_allocr_alloc_graph()
        };
        // LOG_DEBUG("mem_size %u ", params.mem_size);

        struct ggml_context* ctx0 = ggml_init(params);

        struct ggml_cgraph* gf = ggml_new_graph(ctx0);

        struct ggml_tensor* z_ = NULL;

        // it's performing a compute, check if backend isn't cpu
        if (!ggml_backend_is_cpu(backend)) {
            // pass input tensors to gpu memory
            z_ = ggml_dup_tensor(ctx0, z);
            ggml_allocr_alloc(compute_alloc, z_);

            // pass data to device backend
            if (!ggml_allocr_is_measure(compute_alloc)) {
                ggml_backend_tensor_set(z_, z->data, 0, ggml_nbytes(z));
            }
        } else {
            z_ = z;
        }

        struct ggml_tensor* out = decode_graph ? decoder.forward(ctx0, z_) : encoder.forward(ctx0, z_);

        ggml_build_forward_expand(gf, out);
        ggml_free(ctx0);

        return gf;
    }

    void begin(struct ggml_tensor* x, bool decode) {
        // calculate the amount of memory required
        // alignment required by the backend
        compute_alloc = ggml_allocr_new_measure_from_backend(backend);

        struct ggml_cgraph* gf = build_graph(x, decode);

        // compute the required memory
        size_t compute_memory_buffer_size = ggml_allocr_alloc_graph(compute_alloc, gf);

        // recreate the allocator with the required memory
        ggml_allocr_free(compute_alloc);

        LOG_DEBUG("TAE compute buffer size: %.2f MB", compute_memory_buffer_size / 1024.0 / 1024.0);

        compute_buffer = ggml_backend_alloc_buffer(backend, compute_memory_buffer_size);
        compute_alloc  = ggml_allocr_new_from_buffer(compute_buffer);
    }

    void compute(struct ggml_tensor* work_result, const int n_threads, struct ggml_tensor* z, bool decode_graph) {
        ggml_allocr_reset(compute_alloc);

        struct ggml_cgraph* gf = build_graph(z, decode_graph);
        ggml_allocr_alloc_graph(compute_alloc, gf);

        if (ggml_backend_is_cpu(backend)) {
            ggml_backend_cpu_set_n_threads(backend, n_threads);
        }

#ifdef SD_USE_METAL
        if (ggml_backend_is_metal(backend)) {
            ggml_backend_metal_set_n_cb(backend, n_threads);
        }
#endif

        ggml_backend_graph_compute(backend, gf);

#ifdef GGML_PERF
        ggml_graph_print(gf);
#endif

        ggml_backend_tensor_get_and_sync(backend, gf->nodes[gf->n_nodes - 1], work_result->data, 0, ggml_nbytes(work_result));
    }

    void end() {
        ggml_allocr_free(compute_alloc);
        ggml_backend_buffer_free(compute_buffer);
        compute_alloc = NULL;
    }
};

/*
    ===================================    ESRGAN  ===================================
    References:
    https://github.com/xinntao/Real-ESRGAN/blob/master/inference_realesrgan.py
    https://github.com/XPixelGroup/BasicSR/blob/v1.4.2/basicsr/archs/rrdbnet_arch.py

*/

struct ResidualDenseBlock {
    int num_features;
    int num_grow_ch;
    ggml_tensor* conv1_w;  // [num_grow_ch, num_features, 3, 3]
    ggml_tensor* conv1_b;  // [num_grow_ch]

    ggml_tensor* conv2_w;  // [num_grow_ch, num_features + num_grow_ch, 3, 3]
    ggml_tensor* conv2_b;  // [num_grow_ch]

    ggml_tensor* conv3_w;  // [num_grow_ch, num_features + 2 * num_grow_ch, 3, 3]
    ggml_tensor* conv3_b;  // [num_grow_ch]

    ggml_tensor* conv4_w;  // [num_grow_ch, num_features + 3 * num_grow_ch, 3, 3]
    ggml_tensor* conv4_b;  // [num_grow_ch]

    ggml_tensor* conv5_w;  // [num_features, num_features + 4 * num_grow_ch, 3, 3]
    ggml_tensor* conv5_b;  // [num_features]

    ResidualDenseBlock() {}

    ResidualDenseBlock(int num_feat, int n_grow_ch) {
        num_features = num_feat;
        num_grow_ch  = n_grow_ch;
    }

    size_t calculate_mem_size() {
        size_t mem_size = num_features * num_grow_ch * 3 * 3 * ggml_type_size(GGML_TYPE_F16);  // conv1_w
        mem_size += num_grow_ch * ggml_type_size(GGML_TYPE_F32);                               // conv1_b

        mem_size += (num_features + num_grow_ch) * num_grow_ch * 3 * 3 * ggml_type_size(GGML_TYPE_F16);  // conv2_w
        mem_size += num_grow_ch * ggml_type_size(GGML_TYPE_F32);                                         // conv2_b

        mem_size += (num_features + 2 * num_grow_ch) * num_grow_ch * 3 * 3 * ggml_type_size(GGML_TYPE_F16);  // conv3_w
        mem_size += num_grow_ch * ggml_type_size(GGML_TYPE_F32);                                             // conv3_w

        mem_size += (num_features + 3 * num_grow_ch) * num_grow_ch * 3 * 3 * ggml_type_size(GGML_TYPE_F16);  // conv4_w
        mem_size += num_grow_ch * ggml_type_size(GGML_TYPE_F32);                                             // conv4_w

        mem_size += (num_features + 4 * num_grow_ch) * num_features * 3 * 3 * ggml_type_size(GGML_TYPE_F16);  // conv5_w
        mem_size += num_features * ggml_type_size(GGML_TYPE_F32);                                             // conv5_w

        return mem_size;
    }

    int get_num_tensors() {
        int num_tensors = 10;
        return num_tensors;
    }

    void init_params(ggml_context* ctx) {
        conv1_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, num_features, num_grow_ch);
        conv1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_grow_ch);
        conv2_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, num_features + num_grow_ch, num_grow_ch);
        conv2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_grow_ch);
        conv3_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, num_features + 2 * num_grow_ch, num_grow_ch);
        conv3_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_grow_ch);
        conv4_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, num_features + 3 * num_grow_ch, num_grow_ch);
        conv4_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_grow_ch);
        conv5_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, num_features + 4 * num_grow_ch, num_features);
        conv5_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_features);
    }

    void map_by_name(std::map<std::string, ggml_tensor*>& tensors, std::string prefix) {
        tensors[prefix + "conv1.weight"] = conv1_w;
        tensors[prefix + "conv1.bias"]   = conv1_b;

        tensors[prefix + "conv2.weight"] = conv2_w;
        tensors[prefix + "conv2.bias"]   = conv2_b;

        tensors[prefix + "conv3.weight"] = conv3_w;
        tensors[prefix + "conv3.bias"]   = conv3_b;

        tensors[prefix + "conv4.weight"] = conv4_w;
        tensors[prefix + "conv4.bias"]   = conv4_b;

        tensors[prefix + "conv5.weight"] = conv5_w;
        tensors[prefix + "conv5.bias"]   = conv5_b;
    }

    ggml_tensor* forward(ggml_context* ctx, ggml_tensor* out_scale, ggml_tensor* x /* feat */) {
        // x1 = self.lrelu(self.conv1(x))
        ggml_tensor* x1 = ggml_nn_conv_2d(ctx, x, conv1_w, conv1_b, 1, 1, 1, 1);
        x1              = ggml_leaky_relu(ctx, x1, 0.2f, true);

        // x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        ggml_tensor* x_cat = ggml_concat(ctx, x, x1);
        ggml_tensor* x2    = ggml_nn_conv_2d(ctx, x_cat, conv2_w, conv2_b, 1, 1, 1, 1);
        x2                 = ggml_leaky_relu(ctx, x2, 0.2f, true);

        // x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x_cat           = ggml_concat(ctx, x_cat, x2);
        ggml_tensor* x3 = ggml_nn_conv_2d(ctx, x_cat, conv3_w, conv3_b, 1, 1, 1, 1);
        x3              = ggml_leaky_relu(ctx, x3, 0.2f, true);

        // x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x_cat           = ggml_concat(ctx, x_cat, x3);
        ggml_tensor* x4 = ggml_nn_conv_2d(ctx, x_cat, conv4_w, conv4_b, 1, 1, 1, 1);
        x4              = ggml_leaky_relu(ctx, x4, 0.2f, true);

        // self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x_cat           = ggml_concat(ctx, x_cat, x4);
        ggml_tensor* x5 = ggml_nn_conv_2d(ctx, x_cat, conv5_w, conv5_b, 1, 1, 1, 1);

        // return x5 * 0.2 + x
        x5 = ggml_add(ctx, ggml_scale(ctx, x5, out_scale), x);
        return x5;
    }
};

struct EsrganBlock {
    ResidualDenseBlock rd_blocks[3];
    int num_residual_blocks = 3;

    EsrganBlock() {}

    EsrganBlock(int num_feat, int num_grow_ch) {
        for (int i = 0; i < num_residual_blocks; i++) {
            rd_blocks[i] = ResidualDenseBlock(num_feat, num_grow_ch);
        }
    }

    int get_num_tensors() {
        int num_tensors = 0;
        for (int i = 0; i < num_residual_blocks; i++) {
            num_tensors += rd_blocks[i].get_num_tensors();
        }
        return num_tensors;
    }

    size_t calculate_mem_size() {
        size_t mem_size = 0;
        for (int i = 0; i < num_residual_blocks; i++) {
            mem_size += rd_blocks[i].calculate_mem_size();
        }
        return mem_size;
    }

    void init_params(ggml_context* ctx) {
        for (int i = 0; i < num_residual_blocks; i++) {
            rd_blocks[i].init_params(ctx);
        }
    }

    void map_by_name(std::map<std::string, ggml_tensor*>& tensors, std::string prefix) {
        for (int i = 0; i < num_residual_blocks; i++) {
            rd_blocks[i].map_by_name(tensors, prefix + "rdb" + std::to_string(i + 1) + ".");
        }
    }

    ggml_tensor* forward(ggml_context* ctx, ggml_tensor* out_scale, ggml_tensor* x) {
        ggml_tensor* out = x;
        for (int i = 0; i < num_residual_blocks; i++) {
            // out = self.rdb...(x)
            out = rd_blocks[i].forward(ctx, out_scale, out);
        }
        // return out * 0.2 + x
        out = ggml_add(ctx, ggml_scale(ctx, out, out_scale), x);
        return out;
    }
};

struct ESRGAN {
    int scale        = 4;  // default RealESRGAN_x4plus_anime_6B
    int num_blocks   = 6;  // default RealESRGAN_x4plus_anime_6B
    int in_channels  = 3;
    int out_channels = 3;
    int num_features = 64;   // default RealESRGAN_x4plus_anime_6B
    int num_grow_ch  = 32;   // default RealESRGAN_x4plus_anime_6B
    int tile_size    = 128;  // avoid cuda OOM for 4gb VRAM

    ggml_tensor* conv_first_w;  // [num_features, in_channels, 3, 3]
    ggml_tensor* conv_first_b;  // [num_features]

    EsrganBlock body_blocks[6];
    ggml_tensor* conv_body_w;  // [num_features, num_features, 3, 3]
    ggml_tensor* conv_body_b;  // [num_features]

    // upsample
    ggml_tensor* conv_up1_w;  // [num_features, num_features, 3, 3]
    ggml_tensor* conv_up1_b;  // [num_features]
    ggml_tensor* conv_up2_w;  // [num_features, num_features, 3, 3]
    ggml_tensor* conv_up2_b;  // [num_features]

    ggml_tensor* conv_hr_w;    // [num_features, num_features, 3, 3]
    ggml_tensor* conv_hr_b;    // [num_features]
    ggml_tensor* conv_last_w;  // [out_channels, num_features, 3, 3]
    ggml_tensor* conv_last_b;  // [out_channels]

    ggml_context* ctx;
    bool decode_only = false;
    ggml_backend_buffer_t params_buffer;
    ggml_backend_buffer_t compute_buffer;  // for compute
    struct ggml_allocr* compute_alloc = NULL;

    int memory_buffer_size = 0;
    ggml_type wtype;
    ggml_backend_t backend = NULL;

    ESRGAN() {
        for (int i = 0; i < num_blocks; i++) {
            body_blocks[i] = EsrganBlock(num_features, num_grow_ch);
        }
    }

    size_t calculate_mem_size() {
        size_t mem_size = num_features * in_channels * 3 * 3 * ggml_type_size(GGML_TYPE_F16);  // conv_first_w
        mem_size += num_features * ggml_type_size(GGML_TYPE_F32);                              // conv_first_b

        for (int i = 0; i < num_blocks; i++) {
            mem_size += body_blocks[i].calculate_mem_size();
        }

        mem_size += num_features * num_features * 3 * 3 * ggml_type_size(GGML_TYPE_F16);  // conv_body_w
        mem_size += num_features * ggml_type_size(GGML_TYPE_F32);                         // conv_body_w

        // upsample
        mem_size += num_features * num_features * 3 * 3 * ggml_type_size(GGML_TYPE_F16);  // conv_up1_w
        mem_size += num_features * ggml_type_size(GGML_TYPE_F32);                         // conv_up1_b

        mem_size += num_features * num_features * 3 * 3 * ggml_type_size(GGML_TYPE_F16);  // conv_up2_w
        mem_size += num_features * ggml_type_size(GGML_TYPE_F32);                         // conv_up2_b

        mem_size += num_features * num_features * 3 * 3 * ggml_type_size(GGML_TYPE_F16);  // conv_hr_w
        mem_size += num_features * ggml_type_size(GGML_TYPE_F32);                         // conv_hr_b

        mem_size += out_channels * num_features * 3 * 3 * ggml_type_size(GGML_TYPE_F16);  // conv_last_w
        mem_size += out_channels * ggml_type_size(GGML_TYPE_F32);                         // conv_last_b
        return mem_size;
    }

    int get_num_tensors() {
        int num_tensors = 12;
        for (int i = 0; i < num_blocks; i++) {
            num_tensors += body_blocks[i].get_num_tensors();
        }
        return num_tensors;
    }

    bool init(ggml_backend_t backend_) {
        this->backend      = backend_;
        memory_buffer_size = calculate_mem_size();
        memory_buffer_size += 1024;  // overhead
        int num_tensors = get_num_tensors();

        LOG_DEBUG("ESRGAN params backend buffer size = % 6.2f MB (%i tensors)", memory_buffer_size / (1024.0 * 1024.0), num_tensors);

        struct ggml_init_params params;
        params.mem_size   = static_cast<size_t>(num_tensors * ggml_tensor_overhead());
        params.mem_buffer = NULL;
        params.no_alloc   = true;

        params_buffer = ggml_backend_alloc_buffer(backend, memory_buffer_size);

        ctx = ggml_init(params);
        if (!ctx) {
            LOG_ERROR("ggml_init() failed");
            return false;
        }
        return true;
    }

    void alloc_params() {
        ggml_allocr* alloc = ggml_allocr_new_from_buffer(params_buffer);
        conv_first_w       = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, in_channels, num_features);
        conv_first_b       = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_features);
        conv_body_w        = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, num_features, num_features);
        conv_body_b        = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_features);
        conv_up1_w         = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, num_features, num_features);
        conv_up1_b         = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_features);
        conv_up2_w         = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, num_features, num_features);
        conv_up2_b         = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_features);
        conv_hr_w          = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, num_features, num_features);
        conv_hr_b          = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_features);
        conv_last_w        = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, num_features, out_channels);
        conv_last_b        = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);

        for (int i = 0; i < num_blocks; i++) {
            body_blocks[i].init_params(ctx);
        }

        // alloc all tensors linked to this context
        for (struct ggml_tensor* t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->data == NULL) {
                ggml_allocr_alloc(alloc, t);
            }
        }
        ggml_allocr_free(alloc);
    }

    bool load_from_file(const std::string& file_path, ggml_backend_t backend) {
        LOG_INFO("loading esrgan from '%s'", file_path.c_str());

        if (!init(backend)) {
            return false;
        }

        std::map<std::string, ggml_tensor*> esrgan_tensors;

        ModelLoader model_loader;
        if (!model_loader.init_from_file(file_path)) {
            LOG_ERROR("init esrgan model loader from file failed: '%s'", file_path.c_str());
            return false;
        }

        // prepare memory for the weights
        {
            alloc_params();
            map_by_name(esrgan_tensors);
        }

        std::set<std::string> tensor_names_in_file;

        auto on_new_tensor_cb = [&](const TensorStorage& tensor_storage, ggml_tensor** dst_tensor) -> bool {
            const std::string& name = tensor_storage.name;
            tensor_names_in_file.insert(name);

            struct ggml_tensor* real;
            if (esrgan_tensors.find(name) != esrgan_tensors.end()) {
                real = esrgan_tensors[name];
            } else {
                LOG_ERROR("unknown tensor '%s' in model file", name.data());
                return true;
            }

            if (
                real->ne[0] != tensor_storage.ne[0] ||
                real->ne[1] != tensor_storage.ne[1] ||
                real->ne[2] != tensor_storage.ne[2] ||
                real->ne[3] != tensor_storage.ne[3]) {
                LOG_ERROR(
                    "tensor '%s' has wrong shape in model file: "
                    "got [%d, %d, %d, %d], expected [%d, %d, %d, %d]",
                    name.c_str(),
                    (int)tensor_storage.ne[0], (int)tensor_storage.ne[1], (int)tensor_storage.ne[2], (int)tensor_storage.ne[3],
                    (int)real->ne[0], (int)real->ne[1], (int)real->ne[2], (int)real->ne[3]);
                return false;
            }

            *dst_tensor = real;

            return true;
        };

        bool success = model_loader.load_tensors(on_new_tensor_cb, backend);

        bool some_tensor_not_init = false;

        for (auto pair : esrgan_tensors) {
            if (tensor_names_in_file.find(pair.first) == tensor_names_in_file.end()) {
                LOG_ERROR("tensor '%s' not in model file", pair.first.c_str());
                some_tensor_not_init = true;
            }
        }

        if (some_tensor_not_init) {
            return false;
        }

        LOG_INFO("esrgan model loaded");
        return success;
    }

    void map_by_name(std::map<std::string, ggml_tensor*>& tensors) {
        tensors["conv_first.weight"] = conv_first_w;
        tensors["conv_first.bias"]   = conv_first_b;

        for (int i = 0; i < num_blocks; i++) {
            body_blocks[i].map_by_name(tensors, "body." + std::to_string(i) + ".");
        }

        tensors["conv_body.weight"] = conv_body_w;
        tensors["conv_body.bias"]   = conv_body_b;

        tensors["conv_up1.weight"] = conv_up1_w;
        tensors["conv_up1.bias"]   = conv_up1_b;
        tensors["conv_up2.weight"] = conv_up2_w;
        tensors["conv_up2.bias"]   = conv_up2_b;
        tensors["conv_hr.weight"]  = conv_hr_w;
        tensors["conv_hr.bias"]    = conv_hr_b;

        tensors["conv_last.weight"] = conv_last_w;
        tensors["conv_last.bias"]   = conv_last_b;
    }

    ggml_tensor* forward(ggml_context* ctx0, ggml_tensor* out_scale, ggml_tensor* x /* feat */) {
        // feat = self.conv_first(feat)
        auto h = ggml_nn_conv_2d(ctx0, x, conv_first_w, conv_first_b, 1, 1, 1, 1);

        auto body_h = h;
        // self.body(feat)
        for (int i = 0; i < num_blocks; i++) {
            body_h = body_blocks[i].forward(ctx0, out_scale, body_h);
        }

        // body_feat = self.conv_body(self.body(feat))
        body_h = ggml_nn_conv_2d(ctx0, body_h, conv_body_w, conv_body_b, 1, 1, 1, 1);

        // feat = feat + body_feat
        h = ggml_add(ctx0, h, body_h);

        // upsample
        // feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        h = ggml_upscale(ctx0, h, 2);
        h = ggml_nn_conv_2d(ctx0, h, conv_up1_w, conv_up1_b, 1, 1, 1, 1);
        h = ggml_leaky_relu(ctx0, h, 0.2f, true);

        // feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        h = ggml_upscale(ctx0, h, 2);
        h = ggml_nn_conv_2d(ctx0, h, conv_up2_w, conv_up2_b, 1, 1, 1, 1);
        h = ggml_leaky_relu(ctx0, h, 0.2f, true);

        // out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        h = ggml_nn_conv_2d(ctx0, h, conv_hr_w, conv_hr_b, 1, 1, 1, 1);
        h = ggml_leaky_relu(ctx0, h, 0.2f, true);

        h = ggml_nn_conv_2d(ctx0, h, conv_last_w, conv_last_b, 1, 1, 1, 1);
        return h;
    }

    struct ggml_cgraph* build_graph(struct ggml_tensor* x) {
        // since we are using ggml-alloc, this buffer only needs enough space to hold the ggml_tensor and ggml_cgraph structs, but not the tensor data
        static size_t buf_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
        static std::vector<uint8_t> buf(buf_size);

        struct ggml_init_params params = {
            /*.mem_size   =*/buf_size,
            /*.mem_buffer =*/buf.data(),
            /*.no_alloc   =*/true,  // the tensors will be allocated later by ggml_allocr_alloc_graph()
        };

        struct ggml_context* ctx0 = ggml_init(params);

        struct ggml_cgraph* gf = ggml_new_graph(ctx0);

        struct ggml_tensor* x_ = NULL;
        struct ggml_tensor* os = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
        ggml_allocr_alloc(compute_alloc, os);
        if (!ggml_allocr_is_measure(compute_alloc)) {
            float scale = 0.2f;
            ggml_backend_tensor_set(os, &scale, 0, sizeof(scale));
        }

        // it's performing a compute, check if backend isn't cpu
        if (!ggml_backend_is_cpu(backend)) {
            // pass input tensors to gpu memory
            x_ = ggml_dup_tensor(ctx0, x);
            ggml_allocr_alloc(compute_alloc, x_);

            // pass data to device backend
            if (!ggml_allocr_is_measure(compute_alloc)) {
                ggml_backend_tensor_set(x_, x->data, 0, ggml_nbytes(x));
            }
        } else {
            x_ = x;
        }

        struct ggml_tensor* out = forward(ctx0, os, x);

        ggml_build_forward_expand(gf, out);
        ggml_free(ctx0);

        return gf;
    }

    void begin(struct ggml_tensor* x) {
        // calculate the amount of memory required
        // alignment required by the backend
        compute_alloc = ggml_allocr_new_measure_from_backend(backend);

        struct ggml_cgraph* gf = build_graph(x);

        // compute the required memory
        size_t compute_memory_buffer_size = ggml_allocr_alloc_graph(compute_alloc, gf);

        // recreate the allocator with the required memory
        ggml_allocr_free(compute_alloc);

        LOG_DEBUG("ESRGAN compute buffer size: %.2f MB", compute_memory_buffer_size / 1024.0 / 1024.0);

        compute_buffer = ggml_backend_alloc_buffer(backend, compute_memory_buffer_size);
        compute_alloc  = ggml_allocr_new_from_buffer(compute_buffer);
    }

    void compute(struct ggml_tensor* work_result, const int n_threads, struct ggml_tensor* x) {
        ggml_allocr_reset(compute_alloc);

        struct ggml_cgraph* gf = build_graph(x);
        ggml_allocr_alloc_graph(compute_alloc, gf);

        if (ggml_backend_is_cpu(backend)) {
            ggml_backend_cpu_set_n_threads(backend, n_threads);
        }

#ifdef SD_USE_METAL
        if (ggml_backend_is_metal(backend)) {
            ggml_backend_metal_set_n_cb(backend, n_threads);
        }
#endif

        ggml_backend_graph_compute(backend, gf);

#ifdef GGML_PERF
        ggml_graph_print(gf);
#endif
        ggml_tensor* out = gf->nodes[gf->n_nodes - 1];
        ggml_backend_tensor_get_and_sync(backend, out, work_result->data, 0, ggml_nbytes(out));
    }

    void end() {
        ggml_allocr_free(compute_alloc);
        ggml_backend_buffer_free(compute_buffer);
        compute_alloc = NULL;
    }
};

float ggml_backend_tensor_get_f32(ggml_tensor* tensor) {
    GGML_ASSERT(tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_F16);
    float value;
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(tensor, &value, 0, sizeof(value));
    } else {  // GGML_TYPE_F16
        ggml_fp16_t f16_value;
        ggml_backend_tensor_get(tensor, &f16_value, 0, sizeof(f16_value));
        value = ggml_fp16_to_fp32(f16_value);
    }
    return value;
}

struct LoraModel {
    float multiplier = 1.0f;
    std::map<std::string, struct ggml_tensor*> lora_tensors;

    struct ggml_context* ctx;
    ggml_backend_buffer_t params_buffer_lora;
    ggml_backend_t backend = NULL;

    bool load(ggml_backend_t backend_, std::string file_path) {
        backend = backend_;
        LOG_INFO("loading LoRA from '%s'", file_path.c_str());
        ModelLoader model_loader;

        if (!model_loader.init_from_file(file_path)) {
            LOG_ERROR("init lora model loader from file failed: '%s'", file_path.c_str());
            return false;
        }

        struct ggml_init_params params;
        params.mem_size   = static_cast<size_t>(LORA_GRAPH_SIZE * ggml_tensor_overhead());
        params.mem_buffer = NULL;
        params.no_alloc   = true;
        // LOG_DEBUG("mem_size %u ", params.mem_size);

        ctx = ggml_init(params);
        if (!ctx) {
            LOG_ERROR("ggml_init() failed");
            return false;
        }

        ggml_type wtype = model_loader.get_sd_wtype();

        LOG_DEBUG("calculating buffer size");
        int64_t memory_buffer_size = model_loader.cal_mem_size(backend);
        LOG_DEBUG("lora params backend buffer size = % 6.2f MB", memory_buffer_size / (1024.0 * 1024.0));

        params_buffer_lora = ggml_backend_alloc_buffer(backend, memory_buffer_size);
        ggml_allocr* alloc = ggml_allocr_new_from_buffer(params_buffer_lora);

        auto on_new_tensor_cb = [&](const TensorStorage& tensor_storage, ggml_tensor** dst_tensor) -> bool {
            const std::string& name = tensor_storage.name;

            struct ggml_tensor* real = ggml_new_tensor(ctx, tensor_storage.type, tensor_storage.n_dims, tensor_storage.ne);
            ggml_allocr_alloc(alloc, real);

            *dst_tensor = real;

            lora_tensors[name] = real;
            return true;
        };

        model_loader.load_tensors(on_new_tensor_cb, backend);

        LOG_DEBUG("finished loaded lora");
        ggml_allocr_free(alloc);
        return true;
    }

    struct ggml_cgraph* build_graph(struct ggml_allocr* compute_alloc, std::map<std::string, struct ggml_tensor*> model_tensors) {
        // make a graph to compute all lora, expected lora and models tensors are in the same backend
        // since we are using ggml-alloc, this buffer only needs enough space to hold the ggml_tensor and ggml_cgraph structs, but not the tensor data
        static size_t buf_size = ggml_tensor_overhead() * LORA_GRAPH_SIZE + ggml_graph_overhead();
        static std::vector<uint8_t> buf(buf_size);

        struct ggml_init_params params = {
            /*.mem_size   =*/buf_size,
            /*.mem_buffer =*/buf.data(),
            /*.no_alloc   =*/true,  // the tensors will be allocated later by ggml_allocr_alloc_graph()
        };
        // LOG_DEBUG("mem_size %u ", params.mem_size);

        struct ggml_context* ctx0 = ggml_init(params);
        struct ggml_cgraph* gf    = ggml_new_graph_custom(ctx0, LORA_GRAPH_SIZE, false);

        std::set<std::string> applied_lora_tensors;
        for (auto it : model_tensors) {
            std::string k_tensor       = it.first;
            struct ggml_tensor* weight = model_tensors[it.first];

            size_t k_pos = k_tensor.find(".weight");
            if (k_pos == std::string::npos) {
                continue;
            }
            k_tensor = k_tensor.substr(0, k_pos);
            replace_all_chars(k_tensor, '.', '_');
            std::string lora_up_name   = "lora." + k_tensor + ".lora_up.weight";
            std::string lora_down_name = "lora." + k_tensor + ".lora_down.weight";
            std::string alpha_name     = "lora." + k_tensor + ".alpha";
            std::string scale_name     = "lora." + k_tensor + ".scale";

            ggml_tensor* lora_up   = NULL;
            ggml_tensor* lora_down = NULL;

            if (lora_tensors.find(lora_up_name) != lora_tensors.end()) {
                lora_up = lora_tensors[lora_up_name];
            }

            if (lora_tensors.find(lora_down_name) != lora_tensors.end()) {
                lora_down = lora_tensors[lora_down_name];
            }

            if (lora_up == NULL || lora_down == NULL) {
                continue;
            }

            applied_lora_tensors.insert(lora_up_name);
            applied_lora_tensors.insert(lora_down_name);
            applied_lora_tensors.insert(alpha_name);
            applied_lora_tensors.insert(scale_name);

            // calc_cale
            int64_t dim       = lora_down->ne[lora_down->n_dims - 1];
            float scale_value = 1.0f;
            if (lora_tensors.find(scale_name) != lora_tensors.end()) {
                scale_value = ggml_backend_tensor_get_f32(lora_tensors[scale_name]);
            } else if (lora_tensors.find(alpha_name) != lora_tensors.end()) {
                float alpha = ggml_backend_tensor_get_f32(lora_tensors[alpha_name]);
                scale_value = alpha / dim;
            }
            scale_value *= multiplier;

            ggml_tensor* lora_scale = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);

            ggml_allocr_alloc(compute_alloc, lora_scale);
            if (!ggml_allocr_is_measure(compute_alloc)) {
                ggml_backend_tensor_set(lora_scale, &scale_value, 0, ggml_nbytes(lora_scale));
            }

            // flat lora tensors to multiply it
            int64_t lora_up_rows   = lora_up->ne[lora_up->n_dims - 1];
            lora_up                = ggml_reshape_2d(ctx0, lora_up, ggml_nelements(lora_up) / lora_up_rows, lora_up_rows);
            int64_t lora_down_rows = lora_down->ne[lora_down->n_dims - 1];
            lora_down              = ggml_reshape_2d(ctx0, lora_down, ggml_nelements(lora_down) / lora_down_rows, lora_down_rows);

            // ggml_mul_mat requires tensor b transposed
            lora_down                  = ggml_cont(ctx0, ggml_transpose(ctx0, lora_down));
            struct ggml_tensor* updown = ggml_mul_mat(ctx0, lora_up, lora_down);
            updown                     = ggml_cont(ctx0, ggml_transpose(ctx0, updown));
            updown                     = ggml_reshape(ctx0, updown, weight);
            GGML_ASSERT(ggml_nelements(updown) == ggml_nelements(weight));
            updown = ggml_scale_inplace(ctx0, updown, lora_scale);
            ggml_tensor* final_weight;
            // if (weight->type != GGML_TYPE_F32 && weight->type != GGML_TYPE_F16) {
            //     final_weight = ggml_new_tensor(ctx0, GGML_TYPE_F32, weight->n_dims, weight->ne);
            //     final_weight = ggml_cpy_inplace(ctx0, weight, final_weight);
            //     final_weight = ggml_add_inplace(ctx0, final_weight, updown);
            //     final_weight = ggml_cpy_inplace(ctx0, final_weight, weight);
            // } else {
            //     final_weight = ggml_add_inplace(ctx0, weight, updown);
            // }
            final_weight = ggml_add_inplace(ctx0, weight, updown);  // apply directly
            ggml_build_forward_expand(gf, final_weight);
        }

        for (auto& kv : lora_tensors) {
            if (applied_lora_tensors.find(kv.first) == applied_lora_tensors.end()) {
                LOG_WARN("unused lora tensor %s", kv.first.c_str());
            }
        }

        return gf;
    }

    void apply(std::map<std::string, struct ggml_tensor*> model_tensors, int n_threads) {
        struct ggml_allocr* compute_alloc         = NULL;
        ggml_backend_buffer_t buffer_compute_lora = NULL;

        // compute the required memory
        {
            compute_alloc                     = ggml_allocr_new_measure_from_backend(backend);
            struct ggml_cgraph* gf            = build_graph(compute_alloc, model_tensors);
            size_t compute_memory_buffer_size = ggml_allocr_alloc_graph(compute_alloc, gf);

            // recreate the allocator with the required memory
            ggml_allocr_free(compute_alloc);
            LOG_DEBUG("apply lora buffer size: %.2f MB", compute_memory_buffer_size / 1024.0 / 1024.0);
            buffer_compute_lora = ggml_backend_alloc_buffer(backend, compute_memory_buffer_size);
            compute_alloc       = ggml_allocr_new_from_buffer(buffer_compute_lora);
        }

        ggml_allocr_reset(compute_alloc);

        struct ggml_cgraph* gf = build_graph(compute_alloc, model_tensors);
        ggml_allocr_alloc_graph(compute_alloc, gf);

        if (ggml_backend_is_cpu(backend)) {
            ggml_backend_cpu_set_n_threads(backend, n_threads);
        }

#ifdef SD_USE_METAL
        if (ggml_backend_is_metal(backend)) {
            ggml_backend_metal_set_n_cb(backend, n_threads);
        }
#endif

        ggml_backend_graph_compute(backend, gf);
        ggml_allocr_free(compute_alloc);
        ggml_backend_buffer_free(buffer_compute_lora);
        compute_alloc = NULL;
    }

    void release() {
        if (ctx != NULL) {
            ggml_free(ctx);
            ctx = NULL;
        }

        if (params_buffer_lora != NULL) {
            ggml_backend_buffer_free(params_buffer_lora);
            params_buffer_lora = NULL;
        }
    }
};

/*================================================= CompVisDenoiser ==================================================*/

// Ref: https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/external.py

struct SigmaSchedule {
    float alphas_cumprod[TIMESTEPS];
    float sigmas[TIMESTEPS];
    float log_sigmas[TIMESTEPS];

    virtual std::vector<float> get_sigmas(uint32_t n) = 0;

    float sigma_to_t(float sigma) {
        float log_sigma = std::log(sigma);
        std::vector<float> dists;
        dists.reserve(TIMESTEPS);
        for (float log_sigma_val : log_sigmas) {
            dists.push_back(log_sigma - log_sigma_val);
        }

        int low_idx = 0;
        for (size_t i = 0; i < TIMESTEPS; i++) {
            if (dists[i] >= 0) {
                low_idx++;
            }
        }
        low_idx      = std::min(std::max(low_idx - 1, 0), TIMESTEPS - 2);
        int high_idx = low_idx + 1;

        float low  = log_sigmas[low_idx];
        float high = log_sigmas[high_idx];
        float w    = (low - log_sigma) / (low - high);
        w          = std::max(0.f, std::min(1.f, w));
        float t    = (1.0f - w) * low_idx + w * high_idx;

        return t;
    }

    float t_to_sigma(float t) {
        int low_idx     = static_cast<int>(std::floor(t));
        int high_idx    = static_cast<int>(std::ceil(t));
        float w         = t - static_cast<float>(low_idx);
        float log_sigma = (1.0f - w) * log_sigmas[low_idx] + w * log_sigmas[high_idx];
        return std::exp(log_sigma);
    }
};

struct DiscreteSchedule : SigmaSchedule {
    std::vector<float> get_sigmas(uint32_t n) {
        std::vector<float> result;

        int t_max = TIMESTEPS - 1;

        if (n == 0) {
            return result;
        } else if (n == 1) {
            result.push_back(t_to_sigma((float)t_max));
            result.push_back(0);
            return result;
        }

        float step = static_cast<float>(t_max) / static_cast<float>(n - 1);
        for (uint32_t i = 0; i < n; ++i) {
            float t = t_max - step * i;
            result.push_back(t_to_sigma(t));
        }
        result.push_back(0);
        return result;
    }
};

struct KarrasSchedule : SigmaSchedule {
    std::vector<float> get_sigmas(uint32_t n) {
        // These *COULD* be function arguments here,
        // but does anybody ever bother to touch them?
        float sigma_min = 0.1f;
        float sigma_max = 10.f;
        float rho       = 7.f;

        std::vector<float> result(n + 1);

        float min_inv_rho = pow(sigma_min, (1.f / rho));
        float max_inv_rho = pow(sigma_max, (1.f / rho));
        for (uint32_t i = 0; i < n; i++) {
            // Eq. (5) from Karras et al 2022
            result[i] = pow(max_inv_rho + (float)i / ((float)n - 1.f) * (min_inv_rho - max_inv_rho), rho);
        }
        result[n] = 0.;
        return result;
    }
};

struct Denoiser {
    std::shared_ptr<SigmaSchedule> schedule              = std::make_shared<DiscreteSchedule>();
    virtual std::vector<float> get_scalings(float sigma) = 0;
};

struct CompVisDenoiser : public Denoiser {
    float sigma_data = 1.0f;

    std::vector<float> get_scalings(float sigma) {
        float c_out = -sigma;
        float c_in  = 1.0f / std::sqrt(sigma * sigma + sigma_data * sigma_data);
        return {c_out, c_in};
    }
};

struct CompVisVDenoiser : public Denoiser {
    float sigma_data = 1.0f;

    std::vector<float> get_scalings(float sigma) {
        float c_skip = sigma_data * sigma_data / (sigma * sigma + sigma_data * sigma_data);
        float c_out  = -sigma * sigma_data / std::sqrt(sigma * sigma + sigma_data * sigma_data);
        float c_in   = 1.0f / std::sqrt(sigma * sigma + sigma_data * sigma_data);
        return {c_skip, c_out, c_in};
    }
};

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

    ESRGAN esrgan_upscaler;
    std::string esrgan_path;
    bool upscale_output = false;

    StableDiffusionGGML() = default;

    StableDiffusionGGML(int n_threads,
                        bool vae_decode_only,
                        bool free_params_immediately,
                        std::string lora_model_dir,
                        RNGType rng_type)
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
        this->lora_model_dir = lora_model_dir;
    }

    ~StableDiffusionGGML() {
        cond_stage_model.destroy();
        diffusion_model.destroy();
        if (!use_tiny_autoencoder) {
            first_stage_model.destroy();
        }
    }

    bool load_from_file(const std::string& model_path,
                        const std::string& vae_path,
                        ggml_type wtype,
                        Schedule schedule,
                        int clip_skip) {
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
        cond_stage_model = FrozenCLIPEmbedderWithCustomWords(version, clip_skip);
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
            !cond_stage_model.initialize(backend, model_data_type) ||
            !diffusion_model.initialize(backend, model_data_type)) {
            return false;
        }

        ggml_type vae_type = model_data_type;
        if (version == VERSION_XL) {
            vae_type = GGML_TYPE_F32;  // avoid nan, not work...
        }

        if (!use_tiny_autoencoder && !first_stage_model.initialize(backend, vae_type)) {
            return false;
        }

        LOG_DEBUG("preparing memory for the weights");
        // prepare memory for the weights
        {
            // cond_stage_model(FrozenCLIPEmbedder)
            cond_stage_model.alloc_params();
            cond_stage_model.map_by_name(tensors, "cond_stage_model.");

            // diffusion_model(UNetModel)
            diffusion_model.alloc_params();
            diffusion_model.map_by_name(tensors, "model.diffusion_model.");

            if (!use_tiny_autoencoder) {
                // firest_stage_model(AutoEncoderKL)
                first_stage_model.alloc_params();
                first_stage_model.map_by_name(tensors, "first_stage_model.");
            }
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
        std::set<std::string> tensor_names_in_file;
        int64_t t0 = ggml_time_ms();

        size_t total_size = 0;
        std::vector<char> read_buf;

        auto on_new_tensor_cb = [&](const TensorStorage& tensor_storage, ggml_tensor** dst_tensor) -> bool {
            const std::string& name = tensor_storage.name;
            tensor_names_in_file.insert(name);

            if (name == "alphas_cumprod") {
                *dst_tensor = alphas_cumprod_tensor;
                return true;
            }

            struct ggml_tensor* real;
            if (tensors.find(name) != tensors.end()) {
                real = tensors[name];
            } else {
                if (use_tiny_autoencoder && starts_with(name, "first_stage_model.")) {
                    return true;
                }
                if (name.find("quant") == std::string::npos && name.find("first_stage_model.encoder.") == std::string::npos) {
                    LOG_WARN("unknown tensor '%s' in model file", name.data());
                } else {
                    if (!vae_decode_only) {
                        LOG_WARN("unknown tensor '%s' in model file", name.data());
                    }
                }
                return true;
            }

            if (
                real->ne[0] != tensor_storage.ne[0] ||
                real->ne[1] != tensor_storage.ne[1] ||
                real->ne[2] != tensor_storage.ne[2] ||
                real->ne[3] != tensor_storage.ne[3]) {
                LOG_ERROR(
                    "tensor '%s' has wrong shape in model file: "
                    "got [%d, %d, %d, %d], expected [%d, %d, %d, %d]",
                    name.c_str(),
                    (int)tensor_storage.ne[0], (int)tensor_storage.ne[1], (int)tensor_storage.ne[2], (int)tensor_storage.ne[3],
                    (int)real->ne[0], (int)real->ne[1], (int)real->ne[2], (int)real->ne[3]);
                return false;
            }

            *dst_tensor = real;

            total_size += ggml_nbytes(real);
            return true;
        };

        // print_ggml_tensor(alphas_cumprod_tensor);

        bool success = model_loader.load_tensors(on_new_tensor_cb, backend);
        if (!success) {
            LOG_ERROR("load tensors from file failed");
            ggml_free(ctx);
            return false;
        }

        // print_ggml_tensor(alphas_cumprod_tensor);

        // calculate_alphas_cumprod((float*)alphas_cumprod_tensor->data);

        bool some_tensor_not_init = false;

        for (auto pair : tensors) {
            if (pair.first.find("cond_stage_model.transformer.text_model.encoder.layers.23") != std::string::npos) {
                continue;
            }

            if (use_tiny_autoencoder && starts_with(pair.first, "first_stage_model.")) {
                continue;
            }

            if (tensor_names_in_file.find(pair.first) == tensor_names_in_file.end()) {
                LOG_ERROR("tensor '%s' not in model file", pair.first.c_str());
                some_tensor_not_init = true;
            }
        }

        if (some_tensor_not_init) {
            ggml_free(ctx);
            return false;
        }

        LOG_DEBUG("model size = %.2fMB", total_size / 1024.0 / 1024.0);

        size_t total_params_size =
            cond_stage_model.memory_buffer_size +
            diffusion_model.memory_buffer_size +
            first_stage_model.memory_buffer_size;
        LOG_INFO("total memory buffer size = %.2fMB (clip %.2fMB, unet %.2fMB, vae %.2fMB)",
                 total_params_size / 1024.0 / 1024.0,
                 cond_stage_model.memory_buffer_size / 1024.0 / 1024.0,
                 diffusion_model.memory_buffer_size / 1024.0 / 1024.0,
                 first_stage_model.memory_buffer_size / 1024.0 / 1024.0);
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
        if (upscale_output) {
            if (!esrgan_upscaler.load_from_file(esrgan_path, backend)) {
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

        diffusion_model.begin(x_t, c, t_emb);

        int64_t t0 = ggml_time_ms();
        ggml_set_f32(timesteps, 999);
        set_timestep_embedding(timesteps, t_emb, diffusion_model.model_channels);
        struct ggml_tensor* out = ggml_dup_tensor(work_ctx, x_t);
        diffusion_model.compute(out, n_threads, x_t, NULL, c, t_emb);
        diffusion_model.end();

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
        int64_t t0 = ggml_time_ms();
        LoraModel lora;
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
        if (!lora.load(backend, file_path)) {
            LOG_WARN("load lora tensors from %s failed", file_path.c_str());
            return;
        }

        lora.multiplier = multiplier;
        lora.apply(tensors, n_threads);
        loras[lora_name] = lora;
        lora.release();

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

    std::pair<ggml_tensor*, ggml_tensor*> get_learned_condition(ggml_context* work_ctx, const std::string& text, int width, int height, bool force_zero_embeddings = false) {
        auto tokens_and_weights     = cond_stage_model.tokenize(text, true);
        std::vector<int>& tokens    = tokens_and_weights.first;
        std::vector<float>& weights = tokens_and_weights.second;
        int64_t t0                  = ggml_time_ms();
        cond_stage_model.begin(work_ctx, (int)tokens.size());
        auto result_pair                  = cond_stage_model.compute(n_threads, tokens);  // [N, n_token, hidden_size]
        struct ggml_tensor* hidden_states = result_pair.first;
        struct ggml_tensor* pooled        = result_pair.second;
        // if (pooled != NULL) {
        //     print_ggml_tensor(hidden_states);
        //     print_ggml_tensor(pooled);
        // }

        cond_stage_model.end();
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
            size_t out_dim = 256;
            vec            = ggml_new_tensor_1d(work_ctx, GGML_TYPE_F32, diffusion_model.adm_in_channels);
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
                        float cfg_scale,
                        SampleMethod method,
                        const std::vector<float>& sigmas) {
        size_t steps = sigmas.size() - 1;
        // x_t = load_tensor_from_file(work_ctx, "./rand0.bin");
        // print_ggml_tensor(x_t);
        struct ggml_tensor* x = ggml_dup_tensor(work_ctx, x_t);
        copy_ggml_tensor(x, x_t);

        struct ggml_tensor* noised_input = ggml_dup_tensor(work_ctx, x_t);
        struct ggml_tensor* timesteps    = ggml_new_tensor_1d(work_ctx, GGML_TYPE_F32, 1);                                     // [N, ]
        struct ggml_tensor* t_emb        = new_timestep_embedding(work_ctx, NULL, timesteps, diffusion_model.model_channels);  // [N, model_channels]
        diffusion_model.begin(noised_input, c, t_emb, c_vector);

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
            diffusion_model.compute(out_cond, n_threads, noised_input, NULL, c, t_emb, c_vector);

            float* negative_data = NULL;
            if (has_unconditioned) {
                // uncond
                diffusion_model.compute(out_uncond, n_threads, noised_input, NULL, uc, t_emb, uc_vector);
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
        diffusion_model.end();
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
                        first_stage_model.begin(in, decode);
                    } else {
                        first_stage_model.compute(out, n_threads, in, decode);
                    }
                };
                sd_tiling(x, result, 8, 32, 0.5f, on_tiling);
            } else {
                first_stage_model.begin(x, decode);
                first_stage_model.compute(result, n_threads, x, decode);
            }
            first_stage_model.end();
            if (decode) {
                ggml_tensor_scale_output(result);
            }
        } else {
            if (vae_tiling && decode) {  // TODO: support tiling vae encode
                // split latent in 64x64 tiles and compute in several steps
                auto on_tiling = [&](ggml_tensor* in, ggml_tensor* out, bool init) {
                    if (init) {
                        tae_first_stage.begin(in, decode);
                    } else {
                        tae_first_stage.compute(out, n_threads, in, decode);
                    }
                };
                sd_tiling(x, result, 8, 64, 0.5f, on_tiling);
            } else {
                tae_first_stage.begin(x, decode);
                tae_first_stage.compute(result, n_threads, x, decode);
            }
            tae_first_stage.end();
        }
        int64_t t1 = ggml_time_ms();
        LOG_DEBUG("computing vae [mode: %s] graph completed, taking %.2fs", decode ? "DECODE" : "ENCODE", (t1 - t0) * 1.0f / 1000);
        if (decode) {
            ggml_tensor_clamp(result, 0.0f, 1.0f);
        }
        return result;
    }

    uint8_t* upscale(ggml_tensor* image) {
        int output_width  = image->ne[0] * esrgan_upscaler.scale;
        int output_height = image->ne[1] * esrgan_upscaler.scale;
        LOG_INFO("upscaling from (%i x %i) to (%i x %i)", image->ne[0], image->ne[1], output_width, output_height);
        struct ggml_init_params params;
        params.mem_size = output_width * output_height * 3 * sizeof(float);  // upscaled
        params.mem_size += 1 * ggml_tensor_overhead();
        params.mem_buffer = NULL;
        params.no_alloc   = false;
        // draft context
        struct ggml_context* upscale_ctx = ggml_init(params);
        if (!upscale_ctx) {
            LOG_ERROR("ggml_init() failed");
            return NULL;
        }
        LOG_DEBUG("upscale work buffer size: %.2f MB", params.mem_size / 1024.f / 1024.f);
        ggml_tensor* upscaled = ggml_new_tensor_4d(upscale_ctx, GGML_TYPE_F32, output_width, output_height, image->ne[2], 1);
        auto on_tiling        = [&](ggml_tensor* in, ggml_tensor* out, bool init) {
            if (init) {
                esrgan_upscaler.begin(in);
            } else {
                esrgan_upscaler.compute(out, n_threads, in);
            }
        };
        int64_t t0 = ggml_time_ms();
        sd_tiling(image, upscaled, esrgan_upscaler.scale, esrgan_upscaler.tile_size, 0.25f, on_tiling);
        esrgan_upscaler.end();
        ggml_tensor_clamp(upscaled, 0.f, 1.f);
        uint8_t* upscaled_data = sd_tensor_to_image(upscaled);
        ggml_free(upscale_ctx);
        int64_t t3 = ggml_time_ms();
        LOG_INFO("image upscaled, taking %.2fs", (t3 - t0) / 1000.0f);
        return upscaled_data;
    }

    ggml_tensor* encode_first_stage(ggml_context* work_ctx, ggml_tensor* x) {
        return compute_first_stage(work_ctx, x, false);
    }

    ggml_tensor* decode_first_stage(ggml_context* work_ctx, ggml_tensor* x) {
        return compute_first_stage(work_ctx, x, true);
    }
};

/*================================================= StableDiffusion ==================================================*/

StableDiffusion::StableDiffusion(int n_threads,
                                 bool vae_decode_only,
                                 std::string taesd_path,
                                 std::string esrgan_path,
                                 bool free_params_immediately,
                                 bool vae_tiling,
                                 std::string lora_model_dir,
                                 RNGType rng_type) {
    sd                       = std::make_shared<StableDiffusionGGML>(n_threads,
                                               vae_decode_only,
                                               free_params_immediately,
                                               lora_model_dir,
                                               rng_type);
    sd->use_tiny_autoencoder = taesd_path.size() > 0;
    sd->taesd_path           = taesd_path;
    sd->upscale_output       = esrgan_path.size() > 0;
    sd->esrgan_path          = esrgan_path;
    sd->vae_tiling           = vae_tiling;
}

bool StableDiffusion::load_from_file(const std::string& model_path,
                                     const std::string& vae_path,
                                     ggml_type wtype,
                                     Schedule s,
                                     int clip_skip) {
    return sd->load_from_file(model_path, vae_path, wtype, s, clip_skip);
}

std::vector<uint8_t*> StableDiffusion::txt2img(std::string prompt,
                                               std::string negative_prompt,
                                               float cfg_scale,
                                               int width,
                                               int height,
                                               SampleMethod sample_method,
                                               int sample_steps,
                                               int64_t seed,
                                               int batch_count) {
    std::vector<uint8_t*> results;
    // if (width >= 1024 && height >= 1024) {  // 1024 x 1024 images
    //     LOG_WARN("Image too large, try a smaller size.");
    //     return results;
    // }
    // extract and remove lora
    auto result_pair                                = extract_and_remove_lora(prompt);
    std::unordered_map<std::string, float> lora_f2m = result_pair.first;  // lora_name -> multiplier

    for (auto& kv : lora_f2m) {
        LOG_DEBUG("lora %s:%.2f", kv.first.c_str(), kv.second);
    }

    prompt = result_pair.second;
    LOG_DEBUG("prompt after extract and remove lora: \"%s\"", prompt.c_str());

    int64_t t0 = ggml_time_ms();
    sd->apply_loras(lora_f2m);
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
        return results;
    }

    if (seed < 0) {
        // Generally, when using the provided command line, the seed is always >0.
        // However, to prevent potential issues if 'stable-diffusion.cpp' is invoked as a library
        // by a third party with a seed <0, let's incorporate randomization here.
        srand((int)time(NULL));
        seed = rand();
    }

    t0                            = ggml_time_ms();
    auto cond_pair                = sd->get_learned_condition(work_ctx, prompt, width, height);
    ggml_tensor* c                = cond_pair.first;
    ggml_tensor* c_vector         = cond_pair.second;  // [adm_in_channels, ]
    struct ggml_tensor* uc        = NULL;
    struct ggml_tensor* uc_vector = NULL;
    if (cfg_scale != 1.0) {
        bool force_zero_embeddings = false;
        if (sd->version == VERSION_XL && negative_prompt.size() == 0) {
            force_zero_embeddings = true;
        }
        auto uncond_pair = sd->get_learned_condition(work_ctx, negative_prompt, width, height, force_zero_embeddings);
        uc               = uncond_pair.first;
        uc_vector        = uncond_pair.second;  // [adm_in_channels, ]
    }
    t1 = ggml_time_ms();
    LOG_INFO("get_learned_condition completed, taking %" PRId64 " ms", t1 - t0);

    if (sd->free_params_immediately) {
        sd->cond_stage_model.destroy();
    }

    std::vector<struct ggml_tensor*> final_latents;  // collect latents to decode
    int C = 4;
    int W = width / 8;
    int H = height / 8;
    LOG_INFO("sampling using %s method", sampling_methods_str[sample_method]);
    for (int b = 0; b < batch_count; b++) {
        int64_t sampling_start = ggml_time_ms();
        int cur_seed           = seed + b;
        LOG_INFO("generating image: %i/%i - seed %i", b + 1, batch_count, cur_seed);

        sd->rng->manual_seed(cur_seed);
        struct ggml_tensor* x_t = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, W, H, C, 1);
        ggml_tensor_set_f32_randn(x_t, sd->rng);

        std::vector<float> sigmas = sd->denoiser->schedule->get_sigmas(sample_steps);

        struct ggml_tensor* x_0 = sd->sample(work_ctx, x_t, NULL, c, c_vector, uc, uc_vector, cfg_scale, sample_method, sigmas);
        // struct ggml_tensor* x_0 = load_tensor_from_file(ctx, "samples_ddim.bin");
        // print_ggml_tensor(x_0);
        int64_t sampling_end = ggml_time_ms();
        LOG_INFO("sampling completed, taking %.2fs", (sampling_end - sampling_start) * 1.0f / 1000);
        final_latents.push_back(x_0);
    }

    if (sd->free_params_immediately) {
        sd->diffusion_model.destroy();
    }
    int64_t t3 = ggml_time_ms();
    LOG_INFO("generating %" PRId64 " latent images completed, taking %.2fs", final_latents.size(), (t3 - t1) * 1.0f / 1000);

    LOG_INFO("decoding %zu latents", final_latents.size());
    std::vector<struct ggml_tensor*> decoded_images;  // collect decoded images
    for (size_t i = 0; i < final_latents.size(); i++) {
        t1                      = ggml_time_ms();
        struct ggml_tensor* img = sd->decode_first_stage(work_ctx, final_latents[i] /* x_0 */);
        // print_ggml_tensor(img);
        if (img != NULL) {
            decoded_images.push_back(img);
        }
        int64_t t2 = ggml_time_ms();
        LOG_INFO("latent %" PRId64 " decoded, taking %.2fs", i + 1, (t2 - t1) * 1.0f / 1000);
    }

    int64_t t4 = ggml_time_ms();
    LOG_INFO("decode_first_stage completed, taking %.2fs", (t4 - t3) * 1.0f / 1000);
    if (sd->free_params_immediately && !sd->use_tiny_autoencoder) {
        sd->first_stage_model.destroy();
    }
    if (sd->upscale_output) {
        LOG_INFO("upscaling %" PRId64 " images", decoded_images.size());
    }
    for (size_t i = 0; i < decoded_images.size(); i++) {
        if (sd->upscale_output) {
            results.push_back(sd->upscale(decoded_images[i]));
        } else {
            results.push_back(sd_tensor_to_image(decoded_images[i]));
        }
    }
    ggml_free(work_ctx);
    LOG_INFO(
        "txt2img completed in %.2fs",
        (t4 - t0) * 1.0f / 1000);

    return results;
}

std::vector<uint8_t*> StableDiffusion::img2img(const uint8_t* init_img_data,
                                               std::string prompt,
                                               std::string negative_prompt,
                                               float cfg_scale,
                                               int width,
                                               int height,
                                               SampleMethod sample_method,
                                               int sample_steps,
                                               float strength,
                                               int64_t seed) {
    std::vector<uint8_t*> result;
    LOG_INFO("img2img %dx%d", width, height);

    std::vector<float> sigmas = sd->denoiser->schedule->get_sigmas(sample_steps);
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
        return result;
    }

    if (seed < 0) {
        seed = (int)time(NULL);
    }

    sd->rng->manual_seed(seed);

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
    sd->apply_loras(lora_f2m);
    int64_t t1 = ggml_time_ms();
    LOG_INFO("apply_loras completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);

    ggml_tensor* init_img = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, width, height, 3, 1);
    sd_image_to_tensor(init_img_data, init_img);
    t0                       = ggml_time_ms();
    ggml_tensor* init_latent = NULL;
    if (!sd->use_tiny_autoencoder) {
        ggml_tensor* moments = sd->encode_first_stage(work_ctx, init_img);
        init_latent          = sd->get_first_stage_encoding(work_ctx, moments);
    } else {
        init_latent = sd->encode_first_stage(work_ctx, init_img);
    }
    // print_ggml_tensor(init_latent);
    t1 = ggml_time_ms();
    LOG_INFO("encode_first_stage completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);

    auto cond_pair                = sd->get_learned_condition(work_ctx, prompt, width, height);
    ggml_tensor* c                = cond_pair.first;
    ggml_tensor* c_vector         = cond_pair.second;  // [adm_in_channels, ]
    struct ggml_tensor* uc        = NULL;
    struct ggml_tensor* uc_vector = NULL;
    if (cfg_scale != 1.0) {
        bool force_zero_embeddings = false;
        if (sd->version == VERSION_XL && negative_prompt.size() == 0) {
            force_zero_embeddings = true;
        }
        auto uncond_pair = sd->get_learned_condition(work_ctx, negative_prompt, width, height, force_zero_embeddings);
        uc               = uncond_pair.first;
        uc_vector        = uncond_pair.second;  // [adm_in_channels, ]
    }
    int64_t t2 = ggml_time_ms();
    LOG_INFO("get_learned_condition completed, taking %" PRId64 " ms", t2 - t1);
    if (sd->free_params_immediately) {
        sd->cond_stage_model.destroy();
    }

    // SDXL
    // requires encode_adm
    // apply set_timestep_embedding with dim 256

    sd->rng->manual_seed(seed);
    struct ggml_tensor* noise = ggml_dup_tensor(work_ctx, init_latent);
    ggml_tensor_set_f32_randn(noise, sd->rng);

    LOG_INFO("sampling using %s method", sampling_methods_str[sample_method]);
    struct ggml_tensor* x_0 = sd->sample(work_ctx, init_latent, noise, c, c_vector, uc, uc_vector, cfg_scale, sample_method, sigma_sched);
    // struct ggml_tensor *x_0 = load_tensor_from_file(ctx, "samples_ddim.bin");
    // print_ggml_tensor(x_0);
    int64_t t3 = ggml_time_ms();
    LOG_INFO("sampling completed, taking %.2fs", (t3 - t2) * 1.0f / 1000);
    if (sd->free_params_immediately) {
        sd->diffusion_model.destroy();
    }

    struct ggml_tensor* img = sd->decode_first_stage(work_ctx, x_0);
    if (img != NULL) {
        if (sd->upscale_output) {
            result.push_back(sd->upscale(img));
        } else {
            result.push_back(sd_tensor_to_image(img));
        }
    }

    int64_t t4 = ggml_time_ms();
    LOG_INFO("decode_first_stage completed, taking %.2fs", (t4 - t3) * 1.0f / 1000);

    if (sd->free_params_immediately && !sd->use_tiny_autoencoder) {
        sd->first_stage_model.destroy();
    }

    LOG_INFO(
        "img2img completed in %.2fs",
        (t4 - t0) * 1.0f / 1000);

    ggml_free(work_ctx);

    return result;
}
