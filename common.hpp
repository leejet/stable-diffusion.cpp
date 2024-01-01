#ifndef __COMMON_HPP__
#define __COMMON_HPP__

#include "ggml_extend.hpp"

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

#endif  // __COMMON_HPP__