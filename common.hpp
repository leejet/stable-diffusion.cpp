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
        size_t mem_size = 0;
        mem_size += ggml_row_size(GGML_TYPE_F16, out_channels * channels * 3 * 3);  // op_w
        mem_size += ggml_row_size(GGML_TYPE_F32, out_channels);                     // op_b
        return mem_size;
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
        size_t mem_size = 0;
        mem_size += ggml_row_size(GGML_TYPE_F16, out_channels * channels * 3 * 3);  // op_w
        mem_size += ggml_row_size(GGML_TYPE_F32, out_channels);                     // op_b
        return mem_size;
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
        size_t mem_size = 0;
        mem_size += 2 * ggml_row_size(GGML_TYPE_F32, channels);                         // in_layer_0_w/b
        mem_size += ggml_row_size(GGML_TYPE_F16, out_channels * channels * 3 * 3);      // in_layer_2_w
        mem_size += 5 * ggml_row_size(GGML_TYPE_F32, out_channels);                     // in_layer_2_b/emb_layer_1_b/out_layer_0_w/out_layer_0_b/out_layer_3_b
        mem_size += ggml_row_size(wtype, out_channels * emb_channels);                  // emb_layer_1_w
        mem_size += ggml_row_size(GGML_TYPE_F16, out_channels * out_channels * 3 * 3);  // out_layer_3_w

        if (out_channels != channels) {
            mem_size += ggml_row_size(GGML_TYPE_F16, out_channels * channels * 1 * 1);  // skip_w
            mem_size += ggml_row_size(GGML_TYPE_F32, out_channels);                     // skip_b
        }
        return mem_size;
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

    // proj_out
    struct ggml_tensor* proj_out_w;  // [in_channels, in_channels, 1, 1]
    struct ggml_tensor* proj_out_b;  // [in_channels,]

    SpatialTransformer(int depth = 1)
        : depth(depth) {
        transformers.resize(depth);
    }

    int get_num_tensors() {
        return depth * 20 + 7;
    }

    size_t calculate_mem_size(ggml_type wtype) {
        size_t mem_size = 0;
        mem_size += 2 * ggml_row_size(GGML_TYPE_F32, in_channels);                        // norm_w/norm_b
        mem_size += 2 * ggml_row_size(GGML_TYPE_F16, in_channels * in_channels * 1 * 1);  // proj_in_w/proj_out_w
        mem_size += 2 * ggml_row_size(GGML_TYPE_F32, in_channels);                        // proj_in_b/proj_out_b

        // transformer
        for (auto& transformer : transformers) {
            mem_size += 6 * ggml_row_size(GGML_TYPE_F32, in_channels);            // norm1-3_w/b
            mem_size += 6 * ggml_row_size(wtype, in_channels * in_channels);      // attn1_q/k/v/out_w attn2_q/out_w
            mem_size += 2 * ggml_row_size(wtype, in_channels * context_dim);      // attn2_k/v_w
            mem_size += ggml_row_size(wtype, in_channels * 4 * 2 * in_channels);  // ff_0_proj_w
            mem_size += ggml_row_size(GGML_TYPE_F32, in_channels * 4 * 2);        // ff_0_proj_b
            mem_size += ggml_row_size(wtype, in_channels * 4 * in_channels);      // ff_2_w
            mem_size += ggml_row_size(GGML_TYPE_F32, in_channels);                // ff_2_b
        }
        return mem_size;
    }

    void init_params(struct ggml_context* ctx, ggml_allocr* alloc, ggml_type wtype) {
        norm_w    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);
        norm_b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);
        proj_in_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 1, 1, in_channels, in_channels);
        proj_in_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);

        proj_out_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 1, 1, in_channels, in_channels);
        proj_out_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);

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
                q = ggml_scale_inplace(ctx, q, 1.0f / sqrt((float)d_head));
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
                q = ggml_scale_inplace(ctx, q, 1.0f / sqrt((float)d_head));
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

#endif  // __COMMON_HPP__