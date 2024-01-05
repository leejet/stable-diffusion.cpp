#ifndef __GGML_EXTEND_HPP__
#define __GGML_EXTEND_HPP__

#include <assert.h>
#include <inttypes.h>
#include <stdarg.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <functional>
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

#include "rng.hpp"
#include "util.h"

#define EPS 1e-05f

#ifndef __STATIC_INLINE__
#define __STATIC_INLINE__ static inline
#endif

__STATIC_INLINE__ void ggml_log_callback_default(ggml_log_level level, const char* text, void* user_data) {
    (void)level;
    (void)user_data;
    fputs(text, stderr);
    fflush(stderr);
}

__STATIC_INLINE__ void ggml_tensor_set_f32_randn(struct ggml_tensor* tensor, std::shared_ptr<RNG> rng) {
    uint32_t n                        = (uint32_t)ggml_nelements(tensor);
    std::vector<float> random_numbers = rng->randn(n);
    for (uint32_t i = 0; i < n; i++) {
        ggml_set_f32_1d(tensor, i, random_numbers[i]);
    }
}

// set tensor[i, j, k, l]
// set tensor[l]
// set tensor[k, l]
// set tensor[j, k, l]
__STATIC_INLINE__ void ggml_tensor_set_f32(struct ggml_tensor* tensor, float value, int l, int k = 0, int j = 0, int i = 0) {
    GGML_ASSERT(tensor->nb[0] == sizeof(float));
    *(float*)((char*)(tensor->data) + i * tensor->nb[3] + j * tensor->nb[2] + k * tensor->nb[1] + l * tensor->nb[0]) = value;
}

__STATIC_INLINE__ float ggml_tensor_get_f32(const ggml_tensor* tensor, int l, int k = 0, int j = 0, int i = 0) {
    // float value;
    // ggml_backend_tensor_get(tensor, &value, i * tensor->nb[3] + j * tensor->nb[2] + k * tensor->nb[1] + l * tensor->nb[0], sizeof(float));
    // return value;
    GGML_ASSERT(tensor->nb[0] == sizeof(float));
    return *(float*)((char*)(tensor->data) + i * tensor->nb[3] + j * tensor->nb[2] + k * tensor->nb[1] + l * tensor->nb[0]);
}

__STATIC_INLINE__ ggml_fp16_t ggml_tensor_get_f16(const ggml_tensor* tensor, int l, int k = 0, int j = 0, int i = 0) {
    GGML_ASSERT(tensor->nb[0] == sizeof(ggml_fp16_t));
    return *(ggml_fp16_t*)((char*)(tensor->data) + i * tensor->nb[3] + j * tensor->nb[2] + k * tensor->nb[1] + l * tensor->nb[0]);
}

__STATIC_INLINE__ void print_ggml_tensor(struct ggml_tensor* tensor, bool shape_only = false) {
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

__STATIC_INLINE__ ggml_tensor* load_tensor_from_file(ggml_context* ctx, const std::string& file_path) {
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

// __STATIC_INLINE__ void save_tensor_to_file(const std::string& file_name, ggml_tensor* tensor, const std::string & name) {
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

__STATIC_INLINE__ void copy_ggml_tensor(struct ggml_tensor* dst, struct ggml_tensor* src) {
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

// SPECIAL OPERATIONS WITH TENSORS

__STATIC_INLINE__ uint8_t* sd_tensor_to_image(struct ggml_tensor* input) {
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

__STATIC_INLINE__ void sd_image_to_tensor(const uint8_t* image_data,
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

__STATIC_INLINE__ void ggml_split_tensor_2d(struct ggml_tensor* input,
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

__STATIC_INLINE__ void ggml_merge_tensor_2d(struct ggml_tensor* input,
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

__STATIC_INLINE__ float ggml_tensor_mean(struct ggml_tensor* src) {
    float mean        = 0.0f;
    int64_t nelements = ggml_nelements(src);
    float* data       = (float*)src->data;
    for (int i = 0; i < nelements; i++) {
        mean += data[i] / nelements * 1.0f;
    }
    return mean;
}

// a = a+b
__STATIC_INLINE__ void ggml_tensor_add(struct ggml_tensor* a, struct ggml_tensor* b) {
    GGML_ASSERT(ggml_nelements(a) == ggml_nelements(b));
    int64_t nelements = ggml_nelements(a);
    float* vec_a      = (float*)a->data;
    float* vec_b      = (float*)b->data;
    for (int i = 0; i < nelements; i++) {
        vec_a[i] = vec_a[i] + vec_b[i];
    }
}

__STATIC_INLINE__ void ggml_tensor_scale(struct ggml_tensor* src, float scale) {
    int64_t nelements = ggml_nelements(src);
    float* data       = (float*)src->data;
    for (int i = 0; i < nelements; i++) {
        data[i] = data[i] * scale;
    }
}

__STATIC_INLINE__ void ggml_tensor_clamp(struct ggml_tensor* src, float min, float max) {
    int64_t nelements = ggml_nelements(src);
    float* data       = (float*)src->data;
    for (int i = 0; i < nelements; i++) {
        float val = data[i];
        data[i]   = val < min ? min : (val > max ? max : val);
    }
}

// convert values from [0, 1] to [-1, 1]
__STATIC_INLINE__ void ggml_tensor_scale_input(struct ggml_tensor* src) {
    int64_t nelements = ggml_nelements(src);
    float* data       = (float*)src->data;
    for (int i = 0; i < nelements; i++) {
        float val = data[i];
        data[i]   = val * 2.0f - 1.0f;
    }
}

// convert values from [-1, 1] to [0, 1]
__STATIC_INLINE__ void ggml_tensor_scale_output(struct ggml_tensor* src) {
    int64_t nelements = ggml_nelements(src);
    float* data       = (float*)src->data;
    for (int i = 0; i < nelements; i++) {
        float val = data[i];
        data[i]   = (val + 1.0f) * 0.5f;
    }
}

typedef std::function<void(ggml_tensor*, ggml_tensor*, bool)> on_tile_process;

// Tiling
__STATIC_INLINE__ void sd_tiling(ggml_tensor* input, ggml_tensor* output, const int scale, const int tile_size, const float tile_overlap_factor, on_tile_process on_processing) {
    int input_width   = (int)input->ne[0];
    int input_height  = (int)input->ne[1];
    int output_width  = (int)output->ne[0];
    int output_height = (int)output->ne[1];
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

__STATIC_INLINE__ struct ggml_tensor* ggml_group_norm_32(struct ggml_context* ctx,
                                                         struct ggml_tensor* a) {
    return ggml_group_norm(ctx, a, 32);
}

__STATIC_INLINE__ struct ggml_tensor* ggml_nn_linear(struct ggml_context* ctx,
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
__STATIC_INLINE__ struct ggml_tensor* ggml_nn_conv_2d(struct ggml_context* ctx,
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

__STATIC_INLINE__ struct ggml_tensor* ggml_nn_layer_norm(struct ggml_context* ctx,
                                                         struct ggml_tensor* x,
                                                         struct ggml_tensor* w,
                                                         struct ggml_tensor* b,
                                                         float eps = EPS) {
    x = ggml_norm(ctx, x, eps);
    x = ggml_mul(ctx, x, w);
    x = ggml_add(ctx, x, b);
    return x;
}

__STATIC_INLINE__ struct ggml_tensor* ggml_nn_group_norm(struct ggml_context* ctx,
                                                         struct ggml_tensor* x,
                                                         struct ggml_tensor* w,
                                                         struct ggml_tensor* b,
                                                         int num_groups = 32) {
    if (ggml_n_dims(x) >= 3) {
        w = ggml_reshape_4d(ctx, w, 1, 1, w->ne[0], 1);
        b = ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
    }

    x = ggml_group_norm(ctx, x, num_groups);
    x = ggml_mul(ctx, x, w);
    x = ggml_add(ctx, x, b);
    return x;
}

__STATIC_INLINE__ void ggml_backend_tensor_get_and_sync(ggml_backend_t backend, const struct ggml_tensor* tensor, void* data, size_t offset, size_t size) {
#ifdef SD_USE_CUBLAS
    ggml_backend_tensor_get_async(backend, tensor, data, offset, size);
    ggml_backend_synchronize(backend);
#else
    ggml_backend_tensor_get(tensor, data, offset, size);
#endif
}

__STATIC_INLINE__ float ggml_backend_tensor_get_f32(ggml_tensor* tensor) {
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

// Ref: https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/util.py#L151
__STATIC_INLINE__ void set_timestep_embedding(struct ggml_tensor* timesteps, struct ggml_tensor* embedding, int dim, int max_period = 10000) {
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

__STATIC_INLINE__ struct ggml_tensor* new_timestep_embedding(struct ggml_context* ctx,
                                                             struct ggml_allocr* allocr,
                                                             struct ggml_tensor* timesteps,
                                                             int dim,
                                                             int max_period = 10000) {
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

struct GGMLModule {
    typedef std::function<struct ggml_cgraph*()> get_graph_cb_t;

    std::string name                     = "ggml module";
    struct ggml_context* params_ctx      = NULL;
    size_t params_buffer_size            = 0;
    size_t compute_buffer_size           = 0;
    ggml_backend_buffer_t params_buffer  = NULL;
    ggml_backend_buffer_t compute_buffer = NULL;  // for compute
    struct ggml_allocr* compute_allocr   = NULL;

    ggml_type wtype        = GGML_TYPE_F32;
    ggml_backend_t backend = NULL;

    virtual size_t calculate_mem_size() = 0;
    virtual size_t get_num_tensors()    = 0;

    bool alloc_params_buffer(ggml_backend_t backend_, ggml_type wtype_ = GGML_TYPE_F32) {
        backend            = backend_;
        wtype              = wtype_;
        params_buffer_size = 10 * 1024 * 1024;  // 10 MB, for padding
        params_buffer_size += calculate_mem_size();
        size_t num_tensors = get_num_tensors();

        LOG_DEBUG("%s params backend buffer size = % 6.2f MB (%i tensors)",
                  name.c_str(), params_buffer_size / (1024.0 * 1024.0), num_tensors);

        struct ggml_init_params params;
        params.mem_size   = static_cast<size_t>(num_tensors * ggml_tensor_overhead()) + 1 * 1024 * 1024;
        params.mem_buffer = NULL;
        params.no_alloc   = true;
        // LOG_DEBUG("mem_size %u ", params.mem_size);

        params_ctx = ggml_init(params);
        if (!params_ctx) {
            LOG_ERROR("ggml_init() failed");
            return false;
        }

        params_buffer = ggml_backend_alloc_buffer(backend, params_buffer_size);
        return true;
    }

    void free_params_buffer() {
        if (params_ctx != NULL) {
            ggml_free(params_ctx);
            params_ctx = NULL;
        }

        if (params_buffer != NULL) {
            ggml_backend_buffer_free(params_buffer);
            params_buffer = NULL;
        }
    }

    ~GGMLModule() {
        free_params_buffer();
    }

    void alloc_compute_buffer(get_graph_cb_t get_graph) {
        if (compute_buffer_size == 0) {
            // alignment required by the backend
            compute_allocr = ggml_allocr_new_measure_from_backend(backend);

            struct ggml_cgraph* gf = get_graph();

            // compute the required memory
            compute_buffer_size = ggml_allocr_alloc_graph(compute_allocr, gf) + 1024 * 1024;

            // recreate the allocator with the required memory
            ggml_allocr_free(compute_allocr);

            LOG_DEBUG("%s compute buffer size: %.2f MB", name.c_str(), compute_buffer_size / 1024.0 / 1024.0);
        }

        compute_buffer = ggml_backend_alloc_buffer(backend, compute_buffer_size);
        compute_allocr = ggml_allocr_new_from_buffer(compute_buffer);
    }

    void compute(get_graph_cb_t get_graph, int n_threads, struct ggml_tensor* output = NULL) {
        ggml_allocr_reset(compute_allocr);

        struct ggml_cgraph* gf = get_graph();

        ggml_allocr_alloc_graph(compute_allocr, gf);

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

        if (output != NULL) {
            ggml_backend_tensor_get_and_sync(backend, gf->nodes[gf->n_nodes - 1], output->data, 0, ggml_nbytes(output));
        }
    }

    void free_compute_buffer() {
        ggml_allocr_free(compute_allocr);
        ggml_backend_buffer_free(compute_buffer);
        compute_allocr      = NULL;
        compute_buffer_size = 0;
    }
};

#endif  // __GGML_EXTEND__HPP__