#ifndef __Z_IMAGE_HPP__
#define __Z_IMAGE_HPP__

#include <algorithm>
#include <cmath>

#include "flux.hpp"
#include "ggml_extend.hpp"
#include "layer_streaming.hpp"
#include "mmdit.hpp"

// Ref: https://github.com/Alpha-VLLM/Lumina-Image-2.0/blob/main/models/model.py
// Ref: https://github.com/huggingface/diffusers/pull/12703

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

namespace ZImage {
    constexpr int Z_IMAGE_GRAPH_SIZE = 20480;
    constexpr int ADALN_EMBED_DIM    = 256;
    constexpr int SEQ_MULTI_OF       = 32;

    struct JointAttention : public GGMLBlock {
    protected:
        int64_t head_dim;
        int64_t num_heads;
        int64_t num_kv_heads;
        bool qk_norm;

    public:
        JointAttention(int64_t hidden_size, int64_t head_dim, int64_t num_heads, int64_t num_kv_heads, bool qk_norm)
            : head_dim(head_dim), num_heads(num_heads), num_kv_heads(num_kv_heads), qk_norm(qk_norm) {
            blocks["qkv"] = std::make_shared<Linear>(hidden_size, (num_heads + num_kv_heads * 2) * head_dim, false);
            float scale   = 1.f;
            blocks["out"] = std::make_shared<Linear>(num_heads * head_dim, hidden_size, false, false, false, scale);
            if (qk_norm) {
                blocks["q_norm"] = std::make_shared<RMSNorm>(head_dim);
                blocks["k_norm"] = std::make_shared<RMSNorm>(head_dim);
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* pe,
                             ggml_tensor* mask = nullptr) {
            // x: [N, n_token, hidden_size]
            int64_t n_token = x->ne[1];
            int64_t N       = x->ne[2];
            auto qkv_proj   = std::dynamic_pointer_cast<Linear>(blocks["qkv"]);
            auto out_proj   = std::dynamic_pointer_cast<Linear>(blocks["out"]);

            if (sd_backend_is(ctx->backend, "ROCm")) {
                out_proj->set_scale(1.f / 16.f);
            }

            auto qkv = qkv_proj->forward(ctx, x);                                                                            // [N, n_token, (num_heads + num_kv_heads*2)*head_dim]
            qkv      = ggml_reshape_4d(ctx->ggml_ctx, qkv, head_dim, num_heads + num_kv_heads * 2, qkv->ne[1], qkv->ne[2]);  // [N, n_token, num_heads + num_kv_heads*2, head_dim]

            auto q = ggml_view_4d(ctx->ggml_ctx,
                                  qkv,
                                  qkv->ne[0],
                                  num_heads,
                                  qkv->ne[2],
                                  qkv->ne[3],
                                  qkv->nb[1],
                                  qkv->nb[2],
                                  qkv->nb[3],
                                  0);  // [N, n_token, num_heads, head_dim]
            auto k = ggml_view_4d(ctx->ggml_ctx,
                                  qkv,
                                  qkv->ne[0],
                                  num_kv_heads,
                                  qkv->ne[2],
                                  qkv->ne[3],
                                  qkv->nb[1],
                                  qkv->nb[2],
                                  qkv->nb[3],
                                  num_heads * qkv->nb[1]);  // [N, n_token, num_kv_heads, head_dim]
            auto v = ggml_view_4d(ctx->ggml_ctx,
                                  qkv,
                                  qkv->ne[0],
                                  num_kv_heads,
                                  qkv->ne[2],
                                  qkv->ne[3],
                                  qkv->nb[1],
                                  qkv->nb[2],
                                  qkv->nb[3],
                                  (num_heads + num_kv_heads) * qkv->nb[1]);  // [N, n_token, num_kv_heads, head_dim]

            if (qk_norm) {
                auto q_norm = std::dynamic_pointer_cast<RMSNorm>(blocks["q_norm"]);
                auto k_norm = std::dynamic_pointer_cast<RMSNorm>(blocks["k_norm"]);

                q = q_norm->forward(ctx, q);
                k = k_norm->forward(ctx, k);
            }

            x = Rope::attention(ctx, q, k, v, pe, mask, 1.f / 128.f);  // [N, n_token, num_heads * head_dim]

            x = out_proj->forward(ctx, x);  // [N, n_token, hidden_size]
            return x;
        }
    };

    class FeedForward : public GGMLBlock {
    public:
        FeedForward(int64_t dim,
                    int64_t hidden_dim,
                    int64_t multiple_of,
                    float ffn_dim_multiplier = 0.f) {
            if (ffn_dim_multiplier > 0.f) {
                hidden_dim = static_cast<int64_t>(ffn_dim_multiplier * hidden_dim);
            }
            hidden_dim   = multiple_of * ((hidden_dim + multiple_of - 1) / multiple_of);
            blocks["w1"] = std::make_shared<Linear>(dim, hidden_dim, false);

            bool force_prec_f32 = false;
            float scale         = 1.f / 128.f;

            // The purpose of the scale here is to prevent NaN issues in certain situations.
            // For example, when using CUDA but the weights are k-quants.
            blocks["w2"] = std::make_shared<Linear>(hidden_dim, dim, false, false, force_prec_f32, scale);
            blocks["w3"] = std::make_shared<Linear>(dim, hidden_dim, false);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto w1 = std::dynamic_pointer_cast<Linear>(blocks["w1"]);
            auto w2 = std::dynamic_pointer_cast<Linear>(blocks["w2"]);
            auto w3 = std::dynamic_pointer_cast<Linear>(blocks["w3"]);

            if (sd_backend_is(ctx->backend, "Vulkan")) {
                w2->set_force_prec_f32(true);
            }

            auto x1 = w1->forward(ctx, x);
            auto x3 = w3->forward(ctx, x);
            x       = ggml_swiglu_split(ctx->ggml_ctx, x1, x3);
            x       = w2->forward(ctx, x);

            return x;
        }
    };

    __STATIC_INLINE__ ggml_tensor* modulate(ggml_context* ctx,
                                            ggml_tensor* x,
                                            ggml_tensor* scale) {
        // x: [N, L, C]
        // scale: [N, C]
        scale = ggml_reshape_3d(ctx, scale, scale->ne[0], 1, scale->ne[1]);  // [N, 1, C]
        x     = ggml_add(ctx, x, ggml_mul(ctx, x, scale));
        return x;
    }

    struct JointTransformerBlock : public GGMLBlock {
    protected:
        bool modulation;

    public:
        JointTransformerBlock(int layer_id,
                              int64_t hidden_size,
                              int64_t head_dim,
                              int64_t num_heads,
                              int64_t num_kv_heads,
                              int64_t multiple_of,
                              float ffn_dim_multiplier,
                              float norm_eps,
                              bool qk_norm,
                              bool modulation = true)
            : modulation(modulation) {
            blocks["attention"]       = std::make_shared<JointAttention>(hidden_size, head_dim, num_heads, num_kv_heads, qk_norm);
            blocks["feed_forward"]    = std::make_shared<FeedForward>(hidden_size, hidden_size, multiple_of, ffn_dim_multiplier);
            blocks["attention_norm1"] = std::make_shared<RMSNorm>(hidden_size, norm_eps);
            blocks["ffn_norm1"]       = std::make_shared<RMSNorm>(hidden_size, norm_eps);
            blocks["attention_norm2"] = std::make_shared<RMSNorm>(hidden_size, norm_eps);
            blocks["ffn_norm2"]       = std::make_shared<RMSNorm>(hidden_size, norm_eps);
            if (modulation) {
                blocks["adaLN_modulation.0"] = std::make_shared<Linear>(MIN(hidden_size, ADALN_EMBED_DIM), 4 * hidden_size);
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* pe,
                             ggml_tensor* mask        = nullptr,
                             ggml_tensor* adaln_input = nullptr) {
            auto attention       = std::dynamic_pointer_cast<JointAttention>(blocks["attention"]);
            auto feed_forward    = std::dynamic_pointer_cast<FeedForward>(blocks["feed_forward"]);
            auto attention_norm1 = std::dynamic_pointer_cast<RMSNorm>(blocks["attention_norm1"]);
            auto ffn_norm1       = std::dynamic_pointer_cast<RMSNorm>(blocks["ffn_norm1"]);
            auto attention_norm2 = std::dynamic_pointer_cast<RMSNorm>(blocks["attention_norm2"]);
            auto ffn_norm2       = std::dynamic_pointer_cast<RMSNorm>(blocks["ffn_norm2"]);

            if (modulation) {
                GGML_ASSERT(adaln_input != nullptr);
                auto adaLN_modulation_0 = std::dynamic_pointer_cast<Linear>(blocks["adaLN_modulation.0"]);

                auto m         = adaLN_modulation_0->forward(ctx, adaln_input);  // [N, 4 * hidden_size]
                auto mods      = ggml_ext_chunk(ctx->ggml_ctx, m, 4, 0);
                auto scale_msa = mods[0];
                auto gate_msa  = mods[1];
                auto scale_mlp = mods[2];
                auto gate_mlp  = mods[3];

                auto residual = x;
                x             = modulate(ctx->ggml_ctx, attention_norm1->forward(ctx, x), scale_msa);
                x             = attention->forward(ctx, x, pe, mask);
                x             = attention_norm2->forward(ctx, x);
                x             = ggml_mul(ctx->ggml_ctx, x, ggml_tanh(ctx->ggml_ctx, gate_msa));
                x             = ggml_add(ctx->ggml_ctx, x, residual);

                residual = x;
                x        = modulate(ctx->ggml_ctx, ffn_norm1->forward(ctx, x), scale_mlp);
                x        = feed_forward->forward(ctx, x);
                x        = ffn_norm2->forward(ctx, x);
                x        = ggml_mul(ctx->ggml_ctx, x, ggml_tanh(ctx->ggml_ctx, gate_mlp));
                x        = ggml_add(ctx->ggml_ctx, x, residual);
            } else {
                GGML_ASSERT(adaln_input == nullptr);

                auto residual = x;
                x             = attention_norm1->forward(ctx, x);
                x             = attention->forward(ctx, x, pe, mask);
                x             = attention_norm2->forward(ctx, x);
                x             = ggml_add(ctx->ggml_ctx, x, residual);

                residual = x;
                x        = ffn_norm1->forward(ctx, x);
                x        = feed_forward->forward(ctx, x);
                x        = ffn_norm2->forward(ctx, x);
                x        = ggml_add(ctx->ggml_ctx, x, residual);
            }

            return x;
        }
    };

    struct FinalLayer : public GGMLBlock {
    public:
        FinalLayer(int64_t hidden_size,
                   int64_t patch_size,
                   int64_t out_channels) {
            blocks["norm_final"]         = std::make_shared<LayerNorm>(hidden_size, 1e-06f, false);
            blocks["linear"]             = std::make_shared<Linear>(hidden_size, patch_size * patch_size * out_channels, true, true);
            blocks["adaLN_modulation.1"] = std::make_shared<Linear>(MIN(hidden_size, ADALN_EMBED_DIM), hidden_size);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* c) {
            // x: [N, n_token, hidden_size]
            // c: [N, hidden_size]
            // return: [N, n_token, patch_size * patch_size * out_channels]
            auto norm_final         = std::dynamic_pointer_cast<LayerNorm>(blocks["norm_final"]);
            auto linear             = std::dynamic_pointer_cast<Linear>(blocks["linear"]);
            auto adaLN_modulation_1 = std::dynamic_pointer_cast<Linear>(blocks["adaLN_modulation.1"]);

            auto scale = adaLN_modulation_1->forward(ctx, ggml_silu(ctx->ggml_ctx, c));  // [N, hidden_size]
            x          = norm_final->forward(ctx, x);
            x          = modulate(ctx->ggml_ctx, x, scale);
            x          = linear->forward(ctx, x);

            return x;
        }
    };

    struct ZImageParams {
        int patch_size             = 2;
        int64_t hidden_size        = 3840;
        int64_t in_channels        = 16;
        int64_t out_channels       = 16;
        int64_t num_layers         = 30;
        int64_t num_refiner_layers = 2;
        int64_t head_dim           = 128;
        int64_t num_heads          = 30;
        int64_t num_kv_heads       = 30;
        int64_t multiple_of        = 256;
        float ffn_dim_multiplier   = 8.0f / 3.0f;
        float norm_eps             = 1e-5f;
        bool qk_norm               = true;
        int64_t cap_feat_dim       = 2560;
        int theta                  = 256;
        std::vector<int> axes_dim  = {32, 48, 48};
        int64_t axes_dim_sum       = 128;
    };

    class ZImageModel : public GGMLBlock {
    protected:
        ZImageParams z_image_params;

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            params["cap_pad_token"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, z_image_params.hidden_size);
            params["x_pad_token"]   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, z_image_params.hidden_size);
        }

    public:
        ZImageModel() = default;
        ZImageModel(ZImageParams z_image_params)
            : z_image_params(z_image_params) {
            blocks["x_embedder"]     = std::make_shared<Linear>(z_image_params.patch_size * z_image_params.patch_size * z_image_params.in_channels, z_image_params.hidden_size);
            blocks["t_embedder"]     = std::make_shared<TimestepEmbedder>(MIN(z_image_params.hidden_size, 1024), 256, 256);
            blocks["cap_embedder.0"] = std::make_shared<RMSNorm>(z_image_params.cap_feat_dim, z_image_params.norm_eps);
            blocks["cap_embedder.1"] = std::make_shared<Linear>(z_image_params.cap_feat_dim, z_image_params.hidden_size);

            for (int i = 0; i < z_image_params.num_refiner_layers; i++) {
                auto block = std::make_shared<JointTransformerBlock>(i,
                                                                     z_image_params.hidden_size,
                                                                     z_image_params.head_dim,
                                                                     z_image_params.num_heads,
                                                                     z_image_params.num_kv_heads,
                                                                     z_image_params.multiple_of,
                                                                     z_image_params.ffn_dim_multiplier,
                                                                     z_image_params.norm_eps,
                                                                     z_image_params.qk_norm,
                                                                     true);

                blocks["noise_refiner." + std::to_string(i)] = block;
            }

            for (int i = 0; i < z_image_params.num_refiner_layers; i++) {
                auto block = std::make_shared<JointTransformerBlock>(i,
                                                                     z_image_params.hidden_size,
                                                                     z_image_params.head_dim,
                                                                     z_image_params.num_heads,
                                                                     z_image_params.num_kv_heads,
                                                                     z_image_params.multiple_of,
                                                                     z_image_params.ffn_dim_multiplier,
                                                                     z_image_params.norm_eps,
                                                                     z_image_params.qk_norm,
                                                                     false);

                blocks["context_refiner." + std::to_string(i)] = block;
            }

            for (int i = 0; i < z_image_params.num_layers; i++) {
                auto block = std::make_shared<JointTransformerBlock>(i,
                                                                     z_image_params.hidden_size,
                                                                     z_image_params.head_dim,
                                                                     z_image_params.num_heads,
                                                                     z_image_params.num_kv_heads,
                                                                     z_image_params.multiple_of,
                                                                     z_image_params.ffn_dim_multiplier,
                                                                     z_image_params.norm_eps,
                                                                     z_image_params.qk_norm,
                                                                     true);

                blocks["layers." + std::to_string(i)] = block;
            }

            blocks["final_layer"] = std::make_shared<FinalLayer>(z_image_params.hidden_size, z_image_params.patch_size, z_image_params.out_channels);
        }

        ggml_tensor* forward_core(GGMLRunnerContext* ctx,
                                  ggml_tensor* x,
                                  ggml_tensor* timestep,
                                  ggml_tensor* context,
                                  ggml_tensor* pe) {
            auto x_embedder     = std::dynamic_pointer_cast<Linear>(blocks["x_embedder"]);
            auto t_embedder     = std::dynamic_pointer_cast<TimestepEmbedder>(blocks["t_embedder"]);
            auto cap_embedder_0 = std::dynamic_pointer_cast<RMSNorm>(blocks["cap_embedder.0"]);
            auto cap_embedder_1 = std::dynamic_pointer_cast<Linear>(blocks["cap_embedder.1"]);
            auto norm_final     = std::dynamic_pointer_cast<RMSNorm>(blocks["norm_final"]);
            auto final_layer    = std::dynamic_pointer_cast<FinalLayer>(blocks["final_layer"]);

            auto txt_pad_token = params["cap_pad_token"];
            auto img_pad_token = params["x_pad_token"];

            int64_t N           = x->ne[2];
            int64_t n_img_token = x->ne[1];
            int64_t n_txt_token = context->ne[1];

            auto t_emb = t_embedder->forward(ctx, timestep);

            auto txt = cap_embedder_1->forward(ctx, cap_embedder_0->forward(ctx, context));  // [N, n_txt_token, hidden_size]
            auto img = x_embedder->forward(ctx, x);                                          // [N, n_img_token, hidden_size]

            int64_t n_txt_pad_token = Rope::bound_mod(static_cast<int>(n_txt_token), SEQ_MULTI_OF);
            if (n_txt_pad_token > 0) {
                auto txt_pad_tokens = ggml_repeat_4d(ctx->ggml_ctx, txt_pad_token, txt_pad_token->ne[0], n_txt_pad_token, N, 1);
                txt                 = ggml_concat(ctx->ggml_ctx, txt, txt_pad_tokens, 1);  // [N, n_txt_token + n_txt_pad_token, hidden_size]
            }

            int64_t n_img_pad_token = Rope::bound_mod(static_cast<int>(n_img_token), SEQ_MULTI_OF);
            if (n_img_pad_token > 0) {
                auto img_pad_tokens = ggml_repeat_4d(ctx->ggml_ctx, img_pad_token, img_pad_token->ne[0], n_img_pad_token, N, 1);
                img                 = ggml_concat(ctx->ggml_ctx, img, img_pad_tokens, 1);  // [N, n_img_token + n_img_pad_token, hidden_size]
            }

            GGML_ASSERT(txt->ne[1] + img->ne[1] == pe->ne[3]);

            auto txt_pe = ggml_ext_slice(ctx->ggml_ctx, pe, 3, 0, txt->ne[1]);
            auto img_pe = ggml_ext_slice(ctx->ggml_ctx, pe, 3, txt->ne[1], pe->ne[3]);

            for (int i = 0; i < z_image_params.num_refiner_layers; i++) {
                auto block = std::dynamic_pointer_cast<JointTransformerBlock>(blocks["context_refiner." + std::to_string(i)]);

                txt = block->forward(ctx, txt, txt_pe, nullptr, nullptr);
            }

            for (int i = 0; i < z_image_params.num_refiner_layers; i++) {
                auto block = std::dynamic_pointer_cast<JointTransformerBlock>(blocks["noise_refiner." + std::to_string(i)]);

                img = block->forward(ctx, img, img_pe, nullptr, t_emb);
            }

            auto txt_img = ggml_concat(ctx->ggml_ctx, txt, img, 1);  // [N, n_txt_token + n_txt_pad_token + n_img_token + n_img_pad_token, hidden_size]

            for (int i = 0; i < z_image_params.num_layers; i++) {
                auto block = std::dynamic_pointer_cast<JointTransformerBlock>(blocks["layers." + std::to_string(i)]);

                txt_img = block->forward(ctx, txt_img, pe, nullptr, t_emb);
            }

            txt_img = final_layer->forward(ctx, txt_img, t_emb);  // [N, n_txt_token + n_txt_pad_token + n_img_token + n_img_pad_token, ph*pw*C]

            img = ggml_ext_slice(ctx->ggml_ctx, txt_img, 1, n_txt_token + n_txt_pad_token, n_txt_token + n_txt_pad_token + n_img_token);  // [N, n_img_token, ph*pw*C]

            return img;
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* timestep,
                             ggml_tensor* context,
                             ggml_tensor* pe,
                             std::vector<ggml_tensor*> ref_latents = {}) {
            // Forward pass of DiT.
            // x: [N, C, H, W]
            // timestep: [N,]
            // context: [N, L, D]
            // pe: [L, d_head/2, 2, 2]
            // return: [N, C, H, W]

            int64_t W = x->ne[0];
            int64_t H = x->ne[1];
            int64_t C = x->ne[2];
            int64_t N = x->ne[3];

            int patch_size = z_image_params.patch_size;

            auto img             = DiT::pad_and_patchify(ctx, x, patch_size, patch_size, false);
            uint64_t n_img_token = img->ne[1];

            if (ref_latents.size() > 0) {
                for (ggml_tensor* ref : ref_latents) {
                    ref = DiT::pad_and_patchify(ctx, ref, patch_size, patch_size, false);
                    img = ggml_concat(ctx->ggml_ctx, img, ref, 1);
                }
            }

            auto out = forward_core(ctx, img, timestep, context, pe);

            out = ggml_ext_slice(ctx->ggml_ctx, out, 1, 0, n_img_token);                              // [N, n_img_token, ph*pw*C]
            out = DiT::unpatchify_and_crop(ctx->ggml_ctx, out, H, W, patch_size, patch_size, false);  // [N, C, H, W]

            out = ggml_ext_scale(ctx->ggml_ctx, out, -1.f);

            return out;
        }

        struct StreamingInputResult {
            ggml_tensor* txt;       // [N, n_txt_token + n_txt_pad_token, hidden_size]
            ggml_tensor* img;       // [N, n_img_token + n_img_pad_token, hidden_size]
            ggml_tensor* t_emb;     // [N, hidden_size]
            ggml_tensor* txt_pe;    // PE for txt
            ggml_tensor* img_pe;    // PE for img
            ggml_tensor* full_pe;   // Full PE for main layers
            int64_t n_txt_token;
            int64_t n_txt_pad_token;
            int64_t n_img_token;
        };

        StreamingInputResult forward_input_stage(GGMLRunnerContext* ctx,
                                                  struct ggml_tensor* x,
                                                  struct ggml_tensor* timestep,
                                                  struct ggml_tensor* context,
                                                  struct ggml_tensor* pe) {
            auto x_embedder     = std::dynamic_pointer_cast<Linear>(blocks["x_embedder"]);
            auto t_embedder     = std::dynamic_pointer_cast<TimestepEmbedder>(blocks["t_embedder"]);
            auto cap_embedder_0 = std::dynamic_pointer_cast<RMSNorm>(blocks["cap_embedder.0"]);
            auto cap_embedder_1 = std::dynamic_pointer_cast<Linear>(blocks["cap_embedder.1"]);

            auto txt_pad_token = params["cap_pad_token"];
            auto img_pad_token = params["x_pad_token"];

            int64_t N           = x->ne[2];
            int64_t n_img_token = x->ne[1];
            int64_t n_txt_token = context->ne[1];

            auto t_emb = t_embedder->forward(ctx, timestep);

            auto txt = cap_embedder_1->forward(ctx, cap_embedder_0->forward(ctx, context));  // [N, n_txt_token, hidden_size]
            auto img = x_embedder->forward(ctx, x);                                          // [N, n_img_token, hidden_size]

            int64_t n_txt_pad_token = Rope::bound_mod(static_cast<int>(n_txt_token), SEQ_MULTI_OF);
            if (n_txt_pad_token > 0) {
                auto txt_pad_tokens = ggml_repeat_4d(ctx->ggml_ctx, txt_pad_token, txt_pad_token->ne[0], n_txt_pad_token, N, 1);
                txt                 = ggml_concat(ctx->ggml_ctx, txt, txt_pad_tokens, 1);
            }

            int64_t n_img_pad_token = Rope::bound_mod(static_cast<int>(n_img_token), SEQ_MULTI_OF);
            if (n_img_pad_token > 0) {
                auto img_pad_tokens = ggml_repeat_4d(ctx->ggml_ctx, img_pad_token, img_pad_token->ne[0], n_img_pad_token, N, 1);
                img                 = ggml_concat(ctx->ggml_ctx, img, img_pad_tokens, 1);
            }

            auto txt_pe = ggml_ext_slice(ctx->ggml_ctx, pe, 3, 0, txt->ne[1]);
            auto img_pe = ggml_ext_slice(ctx->ggml_ctx, pe, 3, txt->ne[1], pe->ne[3]);

            return {txt, img, t_emb, txt_pe, img_pe, pe, n_txt_token, n_txt_pad_token, n_img_token};
        }

        ggml_tensor* forward_context_refiner_block(GGMLRunnerContext* ctx,
                                                    int block_idx,
                                                    struct ggml_tensor* txt,
                                                    struct ggml_tensor* txt_pe) {
            auto block = std::dynamic_pointer_cast<JointTransformerBlock>(blocks["context_refiner." + std::to_string(block_idx)]);
            return block->forward(ctx, txt, txt_pe, nullptr, nullptr);
        }

        ggml_tensor* forward_noise_refiner_block(GGMLRunnerContext* ctx,
                                                  int block_idx,
                                                  struct ggml_tensor* img,
                                                  struct ggml_tensor* img_pe,
                                                  struct ggml_tensor* t_emb) {
            auto block = std::dynamic_pointer_cast<JointTransformerBlock>(blocks["noise_refiner." + std::to_string(block_idx)]);
            return block->forward(ctx, img, img_pe, nullptr, t_emb);
        }

        ggml_tensor* forward_layer_block(GGMLRunnerContext* ctx,
                                          int block_idx,
                                          struct ggml_tensor* txt_img,
                                          struct ggml_tensor* pe,
                                          struct ggml_tensor* t_emb) {
            auto block = std::dynamic_pointer_cast<JointTransformerBlock>(blocks["layers." + std::to_string(block_idx)]);
            return block->forward(ctx, txt_img, pe, nullptr, t_emb);
        }

        ggml_tensor* forward_output_stage(GGMLRunnerContext* ctx,
                                           struct ggml_tensor* txt_img,
                                           struct ggml_tensor* t_emb) {
            auto final_layer = std::dynamic_pointer_cast<FinalLayer>(blocks["final_layer"]);
            return final_layer->forward(ctx, txt_img, t_emb);
        }

        int get_num_refiner_layers() const { return z_image_params.num_refiner_layers; }
        int get_num_layers() const { return z_image_params.num_layers; }
        int get_patch_size() const { return z_image_params.patch_size; }
    };

    struct ZImageRunner : public GGMLRunner {
    public:
        ZImageParams z_image_params;
        ZImageModel z_image;
        std::vector<float> pe_vec;
        std::vector<float> timestep_vec;
        SDVersion version;

        // Number of main layers kept resident on GPU across sampling steps.
        // -1 = uncomputed; set on the first compute_streaming_true() call once
        // refiners and _global are loaded so we know real free VRAM.
        int resident_layer_count_ = -1;

        // Pinned host buffer for persistent activations (txt_img, t_emb) used
        // across the per-layer streaming graphs. Pageable host buffers force
        // the CUDA backend to stage transfers through an internal bounce
        // buffer; pinning makes both ggml_backend_tensor_get and
        // copy_data_to_backend_tensor 3–4x faster.
        ggml_backend_buffer_t persistent_act_host_buf_ = nullptr;
        size_t persistent_act_host_size_               = 0;
        float* persistent_txt_img_ptr_                 = nullptr;
        float* persistent_t_emb_ptr_                   = nullptr;
        size_t persistent_txt_img_count_               = 0;
        size_t persistent_t_emb_count_                 = 0;

    public:

        ZImageRunner(ggml_backend_t backend,
                     bool offload_params_to_cpu,
                     const String2TensorStorage& tensor_storage_map = {},
                     const std::string prefix                       = "",
                     SDVersion version                              = VERSION_Z_IMAGE)
            : GGMLRunner(backend, offload_params_to_cpu) {
            z_image = ZImageModel(z_image_params);
            z_image.init(params_ctx, tensor_storage_map, prefix);
        }

        ~ZImageRunner() {
            if (persistent_act_host_buf_ != nullptr) {
                ggml_backend_buffer_free(persistent_act_host_buf_);
                persistent_act_host_buf_ = nullptr;
            }
        }

        std::string get_desc() override {
            return "z_image";
        }

        // Allocates (or reallocates if size grew) a single pinned host buffer
        // big enough to hold both persistent_txt_img and persistent_t_emb. The
        // pinned memory makes the per-layer ggml_backend_tensor_get and
        // copy_data_to_backend_tensor calls run at full PCIe bandwidth instead
        // of staging through CUDA's internal bounce buffer.
        bool ensure_pinned_act_buffers(size_t txt_img_count, size_t t_emb_count) {
            const size_t align = 256;
            size_t txt_img_bytes = ((txt_img_count * sizeof(float) + align - 1) / align) * align;
            size_t t_emb_bytes   = ((t_emb_count   * sizeof(float) + align - 1) / align) * align;
            size_t total         = txt_img_bytes + t_emb_bytes;

            if (persistent_act_host_buf_ != nullptr && persistent_act_host_size_ >= total) {
                persistent_txt_img_count_ = txt_img_count;
                persistent_t_emb_count_   = t_emb_count;
                persistent_t_emb_ptr_     = persistent_txt_img_ptr_ + (txt_img_bytes / sizeof(float));
                return true;
            }

            if (persistent_act_host_buf_ != nullptr) {
                ggml_backend_buffer_free(persistent_act_host_buf_);
                persistent_act_host_buf_ = nullptr;
            }

            ggml_backend_dev_t gpu_dev = runtime_backend ? ggml_backend_get_device(runtime_backend) : nullptr;
            ggml_backend_buffer_type_t host_buft = gpu_dev ? ggml_backend_dev_host_buffer_type(gpu_dev) : nullptr;
            if (host_buft != nullptr) {
                persistent_act_host_buf_ = ggml_backend_buft_alloc_buffer(host_buft, total);
            }
            if (persistent_act_host_buf_ == nullptr) {
                LOG_WARN("%s pinned activation buffer alloc failed (%.2f MB), "
                         "falling back to pageable",
                         get_desc().c_str(), total / (1024.0 * 1024.0));
                return false;
            }

            persistent_act_host_size_ = total;
            persistent_txt_img_ptr_   = static_cast<float*>(ggml_backend_buffer_get_base(persistent_act_host_buf_));
            persistent_t_emb_ptr_     = persistent_txt_img_ptr_ + (txt_img_bytes / sizeof(float));
            persistent_txt_img_count_ = txt_img_count;
            persistent_t_emb_count_   = t_emb_count;
            return true;
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string prefix) {
            z_image.get_param_tensors(tensors, prefix);
        }

        void enable_layer_streaming(const LayerStreaming::StreamingConfig& config = {}) {
            std::map<std::string, ggml_tensor*> tensor_map;
            z_image.get_param_tensors(tensor_map, "model.diffusion_model");
            init_streaming(config, tensor_map, LayerStreaming::zimage_layer_pattern);
            LOG_INFO("%s layer streaming enabled (%zu layers)",
                     get_desc().c_str(), streaming_engine_->get_registry().get_layer_count());
        }

        bool compute_streaming(int n_threads,
                               struct ggml_tensor* x,
                               struct ggml_tensor* timesteps,
                               struct ggml_tensor* context,
                               std::vector<ggml_tensor*> ref_latents = {},
                               bool increase_ref_index               = false,
                               struct ggml_tensor** output           = nullptr,
                               struct ggml_context* output_ctx       = nullptr) {
            if (!is_streaming_enabled()) {
                LOG_ERROR("%s streaming not enabled", get_desc().c_str());
                return false;
            }

            int64_t t0 = ggml_time_ms();
            auto analysis = analyze_vram_budget();

            if (analysis.fits_in_vram) {
                LOG_INFO("%s model fits in VRAM, using coarse-stage streaming", get_desc().c_str());
                load_all_layers_coarse();
                bool result = compute(n_threads, x, timesteps, context, ref_latents, increase_ref_index,
                                      output, output_ctx, true);
                int64_t t1 = ggml_time_ms();
                LOG_INFO("%s coarse-stage streaming completed in %.2fs", get_desc().c_str(), (t1 - t0) / 1000.0);
                free_compute_buffer();
                return result;
            }

            LOG_INFO("%s remaining %.2f GB exceeds available %.2f GB, using per-layer streaming",
                     get_desc().c_str(),
                     analysis.remaining_to_load / (1024.0 * 1024.0 * 1024.0),
                     analysis.available_vram / (1024.0 * 1024.0 * 1024.0));

            return compute_streaming_true(n_threads, x, timesteps, context, ref_latents, increase_ref_index,
                                          output, output_ctx);
        }

        bool compute_streaming_true(int n_threads,
                                     struct ggml_tensor* x,
                                     struct ggml_tensor* timesteps,
                                     struct ggml_tensor* context,
                                     std::vector<ggml_tensor*> ref_latents = {},
                                     bool increase_ref_index               = false,
                                     struct ggml_tensor** output           = nullptr,
                                     struct ggml_context* output_ctx       = nullptr) {
            auto& registry = streaming_engine_->get_registry();
            int64_t t_start = ggml_time_ms();

            const int num_refiner_layers = z_image.get_num_refiner_layers();
            const int num_layers = z_image.get_num_layers();
            const int patch_size = z_image.get_patch_size();
            const int64_t W = x->ne[0];
            const int64_t H = x->ne[1];

            LOG_INFO("TRUE per-layer streaming - %d refiners + %d layers",
                     num_refiner_layers, num_layers);

            // Load global layers
            if (!registry.move_layer_to_gpu("_global")) {
                LOG_ERROR("Failed to load _global to GPU");
                return false;
            }

            // Load refiner layers (context_refiner and noise_refiner)
            for (int i = 0; i < num_refiner_layers; i++) {
                std::string cr_name = "context_refiner." + std::to_string(i);
                std::string nr_name = "noise_refiner." + std::to_string(i);
                if (!registry.move_layer_to_gpu(cr_name)) {
                    LOG_ERROR("Failed to load %s to GPU", cr_name.c_str());
                    return false;
                }
                if (!registry.move_layer_to_gpu(nr_name)) {
                    LOG_ERROR("Failed to load %s to GPU", nr_name.c_str());
                    return false;
                }
            }
            // Generate PE
            pe_vec = Rope::gen_z_image_pe(static_cast<int>(H),
                                           static_cast<int>(W),
                                           z_image_params.patch_size,
                                           static_cast<int>(x->ne[3]),
                                           static_cast<int>(context->ne[1]),
                                           SEQ_MULTI_OF,
                                           ref_latents,
                                           increase_ref_index,
                                           z_image_params.theta,
                                           circular_y_enabled,
                                           circular_x_enabled,
                                           z_image_params.axes_dim);
            // For ZImage with refiners, we'll execute refiners with global,
            // then stream main layers one at a time
            // This is a simplified approach - refiners are usually small

            // Persistent storage. Pinned host buffer (member-scoped, reused
            // across sampling steps) so the per-layer ggml_backend_tensor_get
            // and copy_data_to_backend_tensor calls run at full PCIe bandwidth.
            // Falls back to pageable std::vector if pinned alloc fails.
            std::vector<float> persistent_txt_img_fallback;
            std::vector<float> persistent_t_emb_fallback;
            float* persistent_txt_img = nullptr;
            float* persistent_t_emb   = nullptr;
            int64_t txt_img_ne[4], t_emb_ne[4];
            int64_t n_txt_token = 0, n_txt_pad_token = 0, n_img_token_val = 0;

            // Stage 1: Input + Refiners (all in one graph since refiners are small)
            {
                ggml_tensor* txt_img_output = nullptr;
                ggml_tensor* t_emb_output = nullptr;

                auto get_refiner_graph = [&]() -> struct ggml_cgraph* {
                    struct ggml_cgraph* gf = new_graph_custom(Z_IMAGE_GRAPH_SIZE / 2);
                    auto runner_ctx = get_context();

                    ggml_tensor* x_backend = to_backend(x);
                    ggml_tensor* context_backend = to_backend(context);
                    ggml_tensor* timesteps_backend = to_backend(timesteps);

                    // Patchify
                    auto img = DiT::pad_and_patchify(&runner_ctx, x_backend, patch_size, patch_size, false);
                    n_img_token_val = img->ne[1];

                    // Handle ref_latents
                    for (auto& ref : ref_latents) {
                        auto ref_backend = to_backend(ref);
                        ref_backend = DiT::pad_and_patchify(&runner_ctx, ref_backend, patch_size, patch_size, false);
                        img = ggml_concat(compute_ctx, img, ref_backend, 1);
                    }

                    // PE tensor
                    int pos_len = static_cast<int>(pe_vec.size() / z_image_params.axes_dim_sum / 2);
                    auto pe = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, z_image_params.axes_dim_sum / 2, pos_len);
                    set_backend_tensor_data(pe, pe_vec.data());

                    // Input stage
                    auto input_result = z_image.forward_input_stage(&runner_ctx, img, timesteps_backend, context_backend, pe);
                    auto txt = input_result.txt;
                    img = input_result.img;
                    auto t_emb = input_result.t_emb;
                    auto txt_pe = input_result.txt_pe;
                    auto img_pe = input_result.img_pe;
                    n_txt_token = input_result.n_txt_token;
                    n_txt_pad_token = input_result.n_txt_pad_token;

                    // Verify PE size
                    int64_t total_tokens = txt->ne[1] + img->ne[1];
                    if (pe->ne[3] != total_tokens) {
                        LOG_ERROR("ZImage PE mismatch: PE has %ld positions but model needs %ld tokens",
                                  pe->ne[3], total_tokens);
                    }

                    // Context refiners
                    for (int i = 0; i < num_refiner_layers; i++) {
                        txt = z_image.forward_context_refiner_block(&runner_ctx, i, txt, txt_pe);
                    }

                    // Noise refiners
                    for (int i = 0; i < num_refiner_layers; i++) {
                        img = z_image.forward_noise_refiner_block(&runner_ctx, i, img, img_pe, t_emb);
                    }

                    // Concat for main layers
                    txt_img_output = ggml_concat(compute_ctx, txt, img, 1);

                    // Create explicit copy of t_emb to prevent buffer aliasing
                    // The allocator may reuse t_emb's buffer after noise refiners use it
                    auto t_emb_copy = ggml_new_tensor(compute_ctx, t_emb->type, ggml_n_dims(t_emb), t_emb->ne);
                    t_emb_copy = ggml_cpy(compute_ctx, t_emb, t_emb_copy);
                    ggml_set_name(t_emb_copy, "t_emb_output_copy");
                    t_emb_output = t_emb_copy;

                    ggml_build_forward_expand(gf, txt_img_output);
                    ggml_build_forward_expand(gf, t_emb_output);

                    return gf;
                };

                // Don't free compute buffer immediately - we need to read outputs first
                if (!GGMLRunner::compute(get_refiner_graph, n_threads, false, nullptr, nullptr, true)) {
                    LOG_ERROR("Refiner stage failed");
                    return false;
                }

                // Extract to persistent storage
                if (txt_img_output && t_emb_output) {
                    size_t txt_img_size = ggml_nelements(txt_img_output);
                    size_t t_emb_size = ggml_nelements(t_emb_output);

                    if (ensure_pinned_act_buffers(txt_img_size, t_emb_size)) {
                        persistent_txt_img = persistent_txt_img_ptr_;
                        persistent_t_emb   = persistent_t_emb_ptr_;
                    } else {
                        persistent_txt_img_fallback.resize(txt_img_size);
                        persistent_t_emb_fallback.resize(t_emb_size);
                        persistent_txt_img = persistent_txt_img_fallback.data();
                        persistent_t_emb   = persistent_t_emb_fallback.data();
                    }

                    ggml_backend_tensor_get(txt_img_output, persistent_txt_img, 0, txt_img_size * sizeof(float));
                    ggml_backend_tensor_get(t_emb_output, persistent_t_emb, 0, t_emb_size * sizeof(float));

                    for (int i = 0; i < 4; i++) {
                        txt_img_ne[i] = txt_img_output->ne[i];
                        t_emb_ne[i] = t_emb_output->ne[i];
                    }
                } else {
                    LOG_ERROR("Failed to get refiner stage outputs");
                    free_compute_buffer();
                    return false;
                }

                // Now safe to free compute buffer
                free_compute_buffer();
            }

            // Refiners stay resident across sampling steps. Their weights are
            // identical every step, so evicting and re-streaming them was
            // pure waste. They cost ~4 layers worth of VRAM (small).

            // On the first sampling step, decide how many main layers we can
            // keep permanently resident. Layers [0..K-1] become a static cache;
            // layers [K..N-1] continue to stream and evict each step.
            if (resident_layer_count_ < 0 && streaming_engine_) {
                resident_layer_count_ = streaming_engine_->compute_resident_block_count("layers.0", num_layers);
                LOG_INFO("%s layer cache: %d resident, %d streamed per step",
                         get_desc().c_str(),
                         resident_layer_count_,
                         num_layers - resident_layer_count_);
            }

            // Stage 2: Main layers (one at a time)
            // Debug: limit layers if env var set (to isolate where grid pattern appears)
            const char* limit_layers_env = std::getenv("SDCPP_LIMIT_MAIN_LAYERS");
            int layers_to_run = num_layers;
            if (limit_layers_env) {
                int limit = std::atoi(limit_layers_env);
                if (limit >= 0 && limit < num_layers) {
                    layers_to_run = limit;
                    LOG_WARN("SDCPP_LIMIT_MAIN_LAYERS=%d: Running only %d of %d main layers (debug mode)",
                             limit, layers_to_run, num_layers);
                }
            }

            auto layer_name_at = [](int i) { return "layers." + std::to_string(i); };

            // Begin prefetch at the first non-resident layer. On step 1 nothing
            // is loaded so this starts at 0; on later steps it skips the cache
            // prefix and queues the streamed tail directly.
            int prefetch_start = 0;
            while (prefetch_start < num_layers &&
                   registry.is_layer_on_gpu(layer_name_at(prefetch_start))) {
                prefetch_start++;
            }
            if (streaming_engine_) {
                streaming_engine_->prime_prefetch(layer_name_at, prefetch_start, num_layers);
            }

            // Phase 3 profiling: per-stage cumulative timings, dumped after the
            // main loop. Set SDCPP_STREAM_PROFILE=1 to enable.
            int64_t prof_wait_us    = 0;
            int64_t prof_load_us    = 0;
            int64_t prof_advance_us = 0;
            int64_t prof_build_us   = 0;
            int64_t prof_compute_us = 0;
            int64_t prof_get_us     = 0;
            int64_t prof_evict_us   = 0;
            const bool prof_enabled = std::getenv("SDCPP_STREAM_PROFILE") != nullptr;
            auto prof_now = []() { return ggml_time_us(); };

            // Phase 3c: build the per-layer graph ONCE (using layer 0's weight
            // tensors) and reuse it for every subsequent layer by swapping
            // the registered weight pointers between layer 0 and layer N.
            // All 30 ZImage main layers share an identical JointTransformerBlock
            // structure, so the cached graph is valid for any layer once its
            // weights are mapped behind layer 0's tensor pointers.
            //
            // Disabled when an at-runtime WeightAdapter (e.g. LoRA) is active —
            // the adapter's forward_with_lora() looks up adapter tensors by
            // a layer-specific prefix at graph-build time, so a cached graph
            // would always reference layer 0's adapter weights, applying
            // them to every layer. We could swap adapter tensors too, but
            // they're managed outside the streaming registry, so for now we
            // just fall back to per-layer graph rebuild.
            const bool graph_reuse_enabled = !has_weight_adapter();
            ggml_cgraph* cached_layer_gf = nullptr;
            ggml_tensor* cached_layer_out = nullptr;

            for (int layer_idx = 0; layer_idx < layers_to_run; layer_idx++) {
                std::string layer_name = layer_name_at(layer_idx);

                int64_t t0 = prof_enabled ? prof_now() : 0;

                // Wait for this layer's prefetch to complete (if async prefetch was started)
                if (streaming_engine_) {
                    streaming_engine_->wait_for_prefetch(layer_name);
                }
                int64_t t1 = prof_enabled ? prof_now() : 0;

                // Load this layer's weights (sync load if prefetch didn't happen)
                if (!registry.move_layer_to_gpu(layer_name)) {
                    LOG_ERROR("Failed to load %s", layer_name.c_str());
                    return false;
                }
                int64_t t2 = prof_enabled ? prof_now() : 0;

                // Keep the prefetch window full
                if (streaming_engine_) {
                    streaming_engine_->advance_prefetch(layer_name_at, layer_idx, num_layers);
                }
                int64_t t3 = prof_enabled ? prof_now() : 0;

                // Redirect the cached graph at this layer's weights. For
                // layer 0 the graph already references its own tensors, so no
                // swap is needed; for any other layer we swap the runtime
                // pointers between layer 0 and layer N before dispatch.
                bool swapped = false;
                if (graph_reuse_enabled && cached_layer_gf != nullptr && layer_idx != 0) {
                    swapped = registry.swap_layer_buffers("layers.0", layer_name);
                    if (!swapped) {
                        LOG_ERROR("Failed to swap weights into cached graph for %s", layer_name.c_str());
                        return false;
                    }
                }

                if (!graph_reuse_enabled || cached_layer_gf == nullptr) {
                    // First layer (or fallback path when graph reuse is disabled
                    // due to at-runtime weight adapters): build the per-layer
                    // graph and dispatch through GGMLRunner::compute() which
                    // creates / re-uses the gallocr.
                    ggml_tensor* current_layer_out = nullptr;
                    auto build_layer_graph = [&]() -> struct ggml_cgraph* {
                        struct ggml_cgraph* gf = new_graph_custom(Z_IMAGE_GRAPH_SIZE / 4);

                        ggml_tensor* txt_img_in = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32,
                                                                      txt_img_ne[0], txt_img_ne[1], txt_img_ne[2], txt_img_ne[3]);
                        ggml_tensor* t_emb_in = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32,
                                                                    t_emb_ne[0], t_emb_ne[1], t_emb_ne[2], t_emb_ne[3]);

                        set_backend_tensor_data(txt_img_in, persistent_txt_img);
                        set_backend_tensor_data(t_emb_in, persistent_t_emb);

                        int pos_len = static_cast<int>(pe_vec.size() / z_image_params.axes_dim_sum / 2);
                        auto pe = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, z_image_params.axes_dim_sum / 2, pos_len);
                        set_backend_tensor_data(pe, pe_vec.data());

                        auto runner_ctx = get_context();
                        current_layer_out = z_image.forward_layer_block(&runner_ctx, layer_idx, txt_img_in, pe, t_emb_in);

                        ggml_build_forward_expand(gf, current_layer_out);

                        if (graph_reuse_enabled) {
                            cached_layer_gf  = gf;
                            cached_layer_out = current_layer_out;
                        }
                        return gf;
                    };

                    if (!GGMLRunner::compute(build_layer_graph, n_threads, false, nullptr, nullptr, true)) {
                        LOG_ERROR("Layer %d execution failed", layer_idx);
                        return false;
                    }
                    if (!graph_reuse_enabled) {
                        cached_layer_out = current_layer_out;
                    }
                } else {
                    if (!dispatch_cached_graph(cached_layer_gf)) {
                        LOG_ERROR("Layer %d cached dispatch failed", layer_idx);
                        if (swapped) {
                            registry.swap_layer_buffers("layers.0", layer_name);
                        }
                        return false;
                    }
                }
                int64_t t4 = prof_enabled ? prof_now() : 0;

                // Read output back into the persistent host buffer (which is
                // the source for the next iteration's txt_img_in upload).
                if (cached_layer_out) {
                    ggml_backend_tensor_get(cached_layer_out, persistent_txt_img, 0, persistent_txt_img_count_ * sizeof(float));
                    for (int i = 0; i < 4; i++) {
                        txt_img_ne[i] = cached_layer_out->ne[i];
                    }
                }
                int64_t t5 = prof_enabled ? prof_now() : 0;

                // Restore layer 0's weight pointers BEFORE move_layer_to_cpu,
                // otherwise the registry's swap-back would move the wrong
                // bytes between CPU and GPU.
                if (swapped) {
                    registry.swap_layer_buffers("layers.0", layer_name);
                }

                if (prof_enabled) {
                    prof_wait_us    += t1 - t0;
                    prof_load_us    += t2 - t1;
                    prof_advance_us += t3 - t2;
                    prof_compute_us += t4 - t3;
                    prof_get_us     += t5 - t4;
                }

                // Don't free compute buffer here — every main layer has the same shape
                // so the gallocr can be reused for the entire sampling step. Freeing here
                // forces a destroy-and-recreate cycle that idles the GPU between layers.

                // Resident layers stay on GPU across sampling steps; only evict
                // streamed layers (idx >= resident_layer_count_).
                if (layer_idx >= resident_layer_count_) {
                    registry.move_layer_to_cpu(layer_name);
                }
            }

            if (prof_enabled) {
                int64_t total = prof_wait_us + prof_load_us + prof_advance_us +
                                prof_compute_us + prof_get_us;
                LOG_INFO("[stream-profile] %d layers: total=%.2fms wait=%.2fms load=%.2fms "
                         "advance=%.2fms compute=%.2fms tensor_get=%.2fms",
                         layers_to_run,
                         total / 1000.0,
                         prof_wait_us / 1000.0,
                         prof_load_us / 1000.0,
                         prof_advance_us / 1000.0,
                         prof_compute_us / 1000.0,
                         prof_get_us / 1000.0);
            }

            // After all main layers are done, free the compute buffer so the output stage
            // (different graph topology) can allocate a fresh one.
            free_compute_buffer();

            // Stage 3: Output
            {
                auto get_output_graph = [&]() -> struct ggml_cgraph* {
                    struct ggml_cgraph* gf = new_graph_custom(Z_IMAGE_GRAPH_SIZE / 4);

                    // Create input tensors in compute_ctx - no to_backend() needed
                    ggml_tensor* txt_img_in = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32,
                                                                  txt_img_ne[0], txt_img_ne[1], txt_img_ne[2], txt_img_ne[3]);
                    ggml_tensor* t_emb_in = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32,
                                                                t_emb_ne[0], t_emb_ne[1], t_emb_ne[2], t_emb_ne[3]);

                    // Schedule data copy from CPU to GPU
                    set_backend_tensor_data(txt_img_in, persistent_txt_img);
                    set_backend_tensor_data(t_emb_in, persistent_t_emb);

                    auto runner_ctx = get_context();
                    auto final_out = z_image.forward_output_stage(&runner_ctx, txt_img_in, t_emb_in);

                    // Extract img portion and unpatchify
                    int64_t n_img_token = n_img_token_val;
                    final_out = ggml_ext_slice(compute_ctx, final_out, 1,
                                               n_txt_token + n_txt_pad_token,
                                               n_txt_token + n_txt_pad_token + n_img_token);

                    final_out = DiT::unpatchify_and_crop(compute_ctx, final_out, H, W, patch_size, patch_size, false);
                    final_out = ggml_ext_scale(compute_ctx, final_out, -1.f);

                    ggml_build_forward_expand(gf, final_out);

                    return gf;
                };

                if (!GGMLRunner::compute(get_output_graph, n_threads, true, output, output_ctx, true)) {
                    LOG_ERROR("Output stage failed");
                    return false;
                }
            }

            int64_t t_end = ggml_time_ms();
            LOG_INFO("TRUE per-layer streaming completed in %.2fs (%d refiners + %d layers)",
                     (t_end - t_start) / 1000.0, num_refiner_layers, num_layers);

            return true;
        }

        // Raw pointer overload used by streaming code paths
        struct ggml_cgraph* build_graph(struct ggml_tensor* x,
                                        struct ggml_tensor* timesteps,
                                        struct ggml_tensor* context,
                                        std::vector<ggml_tensor*> ref_latents = {},
                                        bool increase_ref_index               = false) {
            GGML_ASSERT(x->ne[3] == 1);
            struct ggml_cgraph* gf = new_graph_custom(Z_IMAGE_GRAPH_SIZE);

            x         = to_backend(x);
            context   = to_backend(context);
            timesteps = to_backend(timesteps);

            for (size_t i = 0; i < ref_latents.size(); i++) {
                ref_latents[i] = to_backend(ref_latents[i]);
            }

            pe_vec      = Rope::gen_z_image_pe(static_cast<int>(x->ne[1]),
                                               static_cast<int>(x->ne[0]),
                                               z_image_params.patch_size,
                                               static_cast<int>(x->ne[3]),
                                               static_cast<int>(context->ne[1]),
                                               SEQ_MULTI_OF,
                                               ref_latents,
                                               increase_ref_index,
                                               z_image_params.theta,
                                               circular_y_enabled,
                                               circular_x_enabled,
                                               z_image_params.axes_dim);
            int pos_len = static_cast<int>(pe_vec.size() / z_image_params.axes_dim_sum / 2);
            auto pe = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, z_image_params.axes_dim_sum / 2, pos_len);
            set_backend_tensor_data(pe, pe_vec.data());
            auto runner_ctx = get_context();

            ggml_tensor* out = z_image.forward(&runner_ctx,
                                               x,
                                               timesteps,
                                               context,
                                               pe,
                                               ref_latents);

            ggml_build_forward_expand(gf, out);

            return gf;
        }

        // sd::Tensor overload used by upstream pipeline
        ggml_cgraph* build_graph(const sd::Tensor<float>& x_tensor,
                                 const sd::Tensor<float>& timesteps_tensor,
                                 const sd::Tensor<float>& context_tensor,
                                 const std::vector<sd::Tensor<float>>& ref_latents_tensor = {},
                                 bool increase_ref_index                                  = false) {
            ggml_cgraph* gf        = new_graph_custom(Z_IMAGE_GRAPH_SIZE);
            ggml_tensor* x         = make_input(x_tensor);
            ggml_tensor* timesteps = make_input(timesteps_tensor);
            GGML_ASSERT(x->ne[3] == 1);
            GGML_ASSERT(!context_tensor.empty());
            ggml_tensor* context = make_input(context_tensor);
            std::vector<ggml_tensor*> ref_latents;
            ref_latents.reserve(ref_latents_tensor.size());
            for (const auto& ref_latent_tensor : ref_latents_tensor) {
                ref_latents.push_back(make_input(ref_latent_tensor));
            }

            pe_vec      = Rope::gen_z_image_pe(static_cast<int>(x->ne[1]),
                                               static_cast<int>(x->ne[0]),
                                               z_image_params.patch_size,
                                               static_cast<int>(x->ne[3]),
                                               static_cast<int>(context->ne[1]),
                                               SEQ_MULTI_OF,
                                               ref_latents,
                                               increase_ref_index,
                                               z_image_params.theta,
                                               circular_y_enabled,
                                               circular_x_enabled,
                                               z_image_params.axes_dim);
            int pos_len = static_cast<int>(pe_vec.size() / z_image_params.axes_dim_sum / 2);
            // LOG_DEBUG("pos_len %d", pos_len);
            auto pe = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, z_image_params.axes_dim_sum / 2, pos_len);
            // pe->data = pe_vec.data();
            // print_ggml_tensor(pe, true, "pe");
            // pe->data = nullptr;
            set_backend_tensor_data(pe, pe_vec.data());
            auto runner_ctx = get_context();

            ggml_tensor* out = z_image.forward(&runner_ctx,
                                               x,
                                               timesteps,
                                               context,
                                               pe,
                                               ref_latents);

            ggml_build_forward_expand(gf, out);

            return gf;
        }

        // Raw pointer overload used by streaming/offloading code paths
        bool compute(int n_threads,
                     struct ggml_tensor* x,
                     struct ggml_tensor* timesteps,
                     struct ggml_tensor* context,
                     std::vector<ggml_tensor*> ref_latents = {},
                     bool increase_ref_index               = false,
                     struct ggml_tensor** output           = nullptr,
                     struct ggml_context* output_ctx       = nullptr,
                     bool skip_param_offload               = false) {
            // x: [N, in_channels, h, w]
            // timesteps: [N, ]
            // context: [N, max_position, hidden_size]
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(x, timesteps, context, ref_latents, increase_ref_index);
            };

            return GGMLRunner::compute(get_graph, n_threads, false, output, output_ctx, skip_param_offload);
        }

        // sd::Tensor overload used by upstream pipeline
        sd::Tensor<float> compute(int n_threads,
                                  const sd::Tensor<float>& x,
                                  const sd::Tensor<float>& timesteps,
                                  const sd::Tensor<float>& context,
                                  const std::vector<sd::Tensor<float>>& ref_latents = {},
                                  bool increase_ref_index                           = false) {
            // x: [N, in_channels, h, w]
            // timesteps: [N, ]
            // context: [N, max_position, hidden_size]
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(x, timesteps, context, ref_latents, increase_ref_index);
            };

            return restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, false), x.dim());
        }

        void test() {
            ggml_init_params params;
            params.mem_size   = static_cast<size_t>(1024 * 1024) * 1024;  // 1GB
            params.mem_buffer = nullptr;
            params.no_alloc   = false;

            ggml_context* ctx = ggml_init(params);
            GGML_ASSERT(ctx != nullptr);

            {
                // auto x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 16, 16, 1);
                // ggml_set_f32(x, 0.01f);
                auto x = sd::load_tensor_from_file_as_tensor<float>("./z_image_x.bin");
                print_sd_tensor(x);

                std::vector<float> timesteps_vec(1, 0.f);
                auto timesteps = sd::Tensor<float>::from_vector(timesteps_vec);

                // auto context = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2560, 256, 1);
                // ggml_set_f32(context, 0.01f);
                auto context = sd::load_tensor_from_file_as_tensor<float>("./z_image_context.bin");
                print_sd_tensor(context);

                sd::Tensor<float> out;

                int64_t t0   = ggml_time_ms();
                auto out_opt = compute(8,
                                       x,
                                       timesteps,
                                       context,
                                       {},
                                       false);
                int64_t t1   = ggml_time_ms();

                GGML_ASSERT(!out_opt.empty());
                out = std::move(out_opt);
                print_sd_tensor(out);
                LOG_DEBUG("z_image test done in %lldms", t1 - t0);
            }
        }

        static void load_from_file_and_test(const std::string& file_path) {
            // cuda q8: pass
            // cuda q8 fa: pass
            // ggml_backend_t backend = ggml_backend_cuda_init(0);
            ggml_backend_t backend    = ggml_backend_cpu_init();
            ggml_type model_data_type = GGML_TYPE_Q8_0;

            ModelLoader model_loader;
            if (!model_loader.init_from_file_and_convert_name(file_path, "model.diffusion_model.")) {
                LOG_ERROR("init model loader from file failed: '%s'", file_path.c_str());
                return;
            }

            auto& tensor_storage_map = model_loader.get_tensor_storage_map();
            if (model_data_type != GGML_TYPE_COUNT) {
                for (auto& [name, tensor_storage] : tensor_storage_map) {
                    if (ends_with(name, "weight")) {
                        tensor_storage.expected_type = model_data_type;
                    }
                }
            }

            std::shared_ptr<ZImageRunner> z_image = std::make_shared<ZImageRunner>(backend,
                                                                                   false,
                                                                                   tensor_storage_map,
                                                                                   "model.diffusion_model",
                                                                                   VERSION_QWEN_IMAGE);

            z_image->alloc_params_buffer();
            std::map<std::string, ggml_tensor*> tensors;
            z_image->get_param_tensors(tensors, "model.diffusion_model");

            bool success = model_loader.load_tensors(tensors);

            if (!success) {
                LOG_ERROR("load tensors from model loader failed");
                return;
            }

            LOG_INFO("z_image model loaded");
            z_image->test();
        }
    };

}  // namespace ZImage

#endif  // __Z_IMAGE_HPP__
