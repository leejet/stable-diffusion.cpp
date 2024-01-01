#ifndef __UNET_HPP__
#define __UNET_HPP__

#include "common.hpp"
#include "ggml_extend.hpp"

/*==================================================== UnetModel =====================================================*/

#define UNET_GRAPH_SIZE 10240

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
        double mem_size = 0;
        mem_size += 2 * in_channels * ggml_type_sizef(GGML_TYPE_F32);                        // norm_w/norm_b
        mem_size += 2 * in_channels * in_channels * 1 * 1 * ggml_type_sizef(GGML_TYPE_F16);  // proj_in_w/proj_out_w
        mem_size += 2 * in_channels * ggml_type_sizef(GGML_TYPE_F32);                        // proj_in_b/proj_out_b

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

// ldm.modules.diffusionmodules.openaimodel.UNetModel
struct UNetModel : public GGMLModule {
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

    UNetModel(SDVersion version = VERSION_1_x)
        : version(version) {
        name = "unet";
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

        size_t len_mults = channel_mult.size();
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
        for (int i = (int)len_mults - 1; i >= 0; i--) {
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
        int ds           = 1;
        size_t len_mults = channel_mult.size();
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
        for (int i = (int)len_mults - 1; i >= 0; i--) {
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

    size_t get_num_tensors() {
        // in
        int num_tensors = 6;
        if (version == VERSION_XL) {
            num_tensors += 4;
        }

        // input blocks
        int ds           = 1;
        size_t len_mults = channel_mult.size();
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
        for (int i = (int)len_mults - 1; i >= 0; i--) {
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

    void init_params() {
        ggml_allocr* alloc = ggml_allocr_new_from_buffer(params_buffer);
        time_embed_0_w     = ggml_new_tensor_2d(params_ctx, wtype, model_channels, time_embed_dim);
        time_embed_0_b     = ggml_new_tensor_1d(params_ctx, GGML_TYPE_F32, time_embed_dim);
        time_embed_2_w     = ggml_new_tensor_2d(params_ctx, wtype, time_embed_dim, time_embed_dim);
        time_embed_2_b     = ggml_new_tensor_1d(params_ctx, GGML_TYPE_F32, time_embed_dim);

        // SDXL
        if (version == VERSION_XL) {
            label_embed_0_w = ggml_new_tensor_2d(params_ctx, wtype, adm_in_channels, time_embed_dim);
            label_embed_0_b = ggml_new_tensor_1d(params_ctx, GGML_TYPE_F32, time_embed_dim);
            label_embed_2_w = ggml_new_tensor_2d(params_ctx, wtype, time_embed_dim, time_embed_dim);
            label_embed_2_b = ggml_new_tensor_1d(params_ctx, GGML_TYPE_F32, time_embed_dim);
        }

        // input_blocks
        input_block_0_w = ggml_new_tensor_4d(params_ctx, GGML_TYPE_F16, 3, 3, in_channels, model_channels);
        input_block_0_b = ggml_new_tensor_1d(params_ctx, GGML_TYPE_F32, model_channels);

        int ds           = 1;
        size_t len_mults = channel_mult.size();
        for (int i = 0; i < len_mults; i++) {
            for (int j = 0; j < num_res_blocks; j++) {
                input_res_blocks[i][j].init_params(params_ctx, wtype);
                if (std::find(attention_resolutions.begin(), attention_resolutions.end(), ds) != attention_resolutions.end()) {
                    input_transformers[i][j].init_params(params_ctx, alloc, wtype);
                }
            }
            if (i != len_mults - 1) {
                input_down_samples[i].init_params(params_ctx, wtype);
                ds *= 2;
            }
        }

        // middle_blocks
        middle_block_0.init_params(params_ctx, wtype);
        middle_block_1.init_params(params_ctx, alloc, wtype);
        middle_block_2.init_params(params_ctx, wtype);

        // output_blocks
        for (int i = (int)len_mults - 1; i >= 0; i--) {
            for (int j = 0; j < num_res_blocks + 1; j++) {
                output_res_blocks[i][j].init_params(params_ctx, wtype);

                if (std::find(attention_resolutions.begin(), attention_resolutions.end(), ds) != attention_resolutions.end()) {
                    output_transformers[i][j].init_params(params_ctx, alloc, wtype);
                }

                if (i > 0 && j == num_res_blocks) {
                    output_up_samples[i - 1].init_params(params_ctx, wtype);

                    ds /= 2;
                }
            }
        }

        // out
        out_0_w = ggml_new_tensor_1d(params_ctx, GGML_TYPE_F32, model_channels);
        out_0_b = ggml_new_tensor_1d(params_ctx, GGML_TYPE_F32, model_channels);

        out_2_w = ggml_new_tensor_4d(params_ctx, GGML_TYPE_F16, 3, 3, model_channels, out_channels);
        out_2_b = ggml_new_tensor_1d(params_ctx, GGML_TYPE_F32, out_channels);

        // alloc all tensors linked to this context
        for (struct ggml_tensor* t = ggml_get_first_tensor(params_ctx); t != NULL; t = ggml_get_next_tensor(params_ctx, t)) {
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

        size_t len_mults    = channel_mult.size();
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
        for (int i = (int)len_mults - 1; i >= 0; i--) {
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
            t_emb = new_timestep_embedding(ctx0, compute_allocr, timesteps, model_channels);  // [N, model_channels]
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
            emb            = ggml_add(params_ctx, emb, label_emb);  // [N, time_embed_dim]
        }

        // input_blocks
        std::vector<struct ggml_tensor*> hs;

        // input block 0
        struct ggml_tensor* h = ggml_nn_conv_2d(ctx0, x, input_block_0_w, input_block_0_b, 1, 1, 1, 1);  // [N, model_channels, h, w]

        ggml_set_name(h, "bench-start");
        hs.push_back(h);
        // input block 1-11
        size_t len_mults = channel_mult.size();
        int ds           = 1;
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
        for (int i = (int)len_mults - 1; i >= 0; i--) {
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
            ggml_allocr_alloc(compute_allocr, x_t);
            if (timesteps != NULL) {
                timesteps_t = ggml_dup_tensor(ctx0, timesteps);
                ggml_allocr_alloc(compute_allocr, timesteps_t);
            }
            ggml_allocr_alloc(compute_allocr, context_t);
            if (t_emb != NULL) {
                t_emb_t = ggml_dup_tensor(ctx0, t_emb);
                ggml_allocr_alloc(compute_allocr, t_emb_t);
            }
            if (y != NULL) {
                y_t = ggml_dup_tensor(ctx0, y);
                ggml_allocr_alloc(compute_allocr, y_t);
            }
            // pass data to device backend
            if (!ggml_allocr_is_measure(compute_allocr)) {
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

    void alloc_compute_buffer(struct ggml_tensor* x,
                              struct ggml_tensor* context,
                              struct ggml_tensor* t_emb = NULL,
                              struct ggml_tensor* y     = NULL) {
        auto get_graph = [&]() -> struct ggml_cgraph* {
            return build_graph(x, NULL, context, t_emb, y);
        };
        GGMLModule::alloc_compute_buffer(get_graph);
    }

    void compute(struct ggml_tensor* work_latent,
                 int n_threads,
                 struct ggml_tensor* x,
                 struct ggml_tensor* timesteps,
                 struct ggml_tensor* context,
                 struct ggml_tensor* t_emb = NULL,
                 struct ggml_tensor* y     = NULL) {
        auto get_graph = [&]() -> struct ggml_cgraph* {
            return build_graph(x, timesteps, context, t_emb, y);
        };

        GGMLModule::compute(get_graph, n_threads, work_latent);
    }
};

#endif  // __UNET_HPP__