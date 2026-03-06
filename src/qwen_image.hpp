#ifndef __QWEN_IMAGE_HPP__
#define __QWEN_IMAGE_HPP__

#include <memory>

#include "common_block.hpp"
#include "flux.hpp"
#include "layer_streaming.hpp"

namespace Qwen {
    constexpr int QWEN_IMAGE_GRAPH_SIZE = 20480;

    struct TimestepEmbedding : public GGMLBlock {
    public:
        TimestepEmbedding(int64_t in_channels,
                          int64_t time_embed_dim,
                          int64_t out_dim       = 0,
                          int64_t cond_proj_dim = 0,
                          bool sample_proj_bias = true) {
            blocks["linear_1"] = std::shared_ptr<GGMLBlock>(new Linear(in_channels, time_embed_dim, sample_proj_bias));
            if (cond_proj_dim > 0) {
                blocks["cond_proj"] = std::shared_ptr<GGMLBlock>(new Linear(cond_proj_dim, in_channels, false));
            }
            if (out_dim <= 0) {
                out_dim = time_embed_dim;
            }
            blocks["linear_2"] = std::shared_ptr<GGMLBlock>(new Linear(time_embed_dim, out_dim, sample_proj_bias));
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* sample,
                                    struct ggml_tensor* condition = nullptr) {
            if (condition != nullptr) {
                auto cond_proj = std::dynamic_pointer_cast<Linear>(blocks["cond_proj"]);
                sample         = ggml_add(ctx->ggml_ctx, sample, cond_proj->forward(ctx, condition));
            }
            auto linear_1 = std::dynamic_pointer_cast<Linear>(blocks["linear_1"]);
            auto linear_2 = std::dynamic_pointer_cast<Linear>(blocks["linear_2"]);

            sample = linear_1->forward(ctx, sample);
            sample = ggml_silu_inplace(ctx->ggml_ctx, sample);
            sample = linear_2->forward(ctx, sample);
            return sample;
        }
    };

    struct QwenTimestepProjEmbeddings : public GGMLBlock {
    public:
        QwenTimestepProjEmbeddings(int64_t embedding_dim) {
            blocks["timestep_embedder"] = std::shared_ptr<GGMLBlock>(new TimestepEmbedding(256, embedding_dim));
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* timesteps) {
            // timesteps: [N,]
            // return: [N, embedding_dim]
            auto timestep_embedder = std::dynamic_pointer_cast<TimestepEmbedding>(blocks["timestep_embedder"]);

            auto timesteps_proj = ggml_ext_timestep_embedding(ctx->ggml_ctx, timesteps, 256, 10000, 1.f);
            auto timesteps_emb  = timestep_embedder->forward(ctx, timesteps_proj);
            return timesteps_emb;
        }
    };

    struct QwenImageAttention : public GGMLBlock {
    protected:
        int64_t dim_head;

    public:
        QwenImageAttention(int64_t query_dim,
                           int64_t dim_head,
                           int64_t num_heads,
                           int64_t out_dim         = 0,
                           int64_t out_context_dim = 0,
                           bool bias               = true,
                           bool out_bias           = true,
                           float eps               = 1e-6)
            : dim_head(dim_head) {
            int64_t inner_dim = out_dim > 0 ? out_dim : dim_head * num_heads;
            out_dim           = out_dim > 0 ? out_dim : query_dim;
            out_context_dim   = out_context_dim > 0 ? out_context_dim : query_dim;

            blocks["to_q"] = std::shared_ptr<GGMLBlock>(new Linear(query_dim, inner_dim, bias));
            blocks["to_k"] = std::shared_ptr<GGMLBlock>(new Linear(query_dim, inner_dim, bias));
            blocks["to_v"] = std::shared_ptr<GGMLBlock>(new Linear(query_dim, inner_dim, bias));

            blocks["norm_q"] = std::shared_ptr<GGMLBlock>(new RMSNorm(dim_head, eps));
            blocks["norm_k"] = std::shared_ptr<GGMLBlock>(new RMSNorm(dim_head, eps));

            blocks["add_q_proj"] = std::shared_ptr<GGMLBlock>(new Linear(query_dim, inner_dim, bias));
            blocks["add_k_proj"] = std::shared_ptr<GGMLBlock>(new Linear(query_dim, inner_dim, bias));
            blocks["add_v_proj"] = std::shared_ptr<GGMLBlock>(new Linear(query_dim, inner_dim, bias));

            blocks["norm_added_q"] = std::shared_ptr<GGMLBlock>(new RMSNorm(dim_head, eps));
            blocks["norm_added_k"] = std::shared_ptr<GGMLBlock>(new RMSNorm(dim_head, eps));

            float scale         = 1.f / 32.f;
            bool force_prec_f32 = false;
#ifdef SD_USE_VULKAN
            force_prec_f32 = true;
#endif
            // The purpose of the scale here is to prevent NaN issues in certain situations.
            // For example when using CUDA but the weights are k-quants (not all prompts).
            blocks["to_out.0"] = std::shared_ptr<GGMLBlock>(new Linear(inner_dim, out_dim, out_bias, false, force_prec_f32, scale));
            // to_out.1 is nn.Dropout

            blocks["to_add_out"] = std::shared_ptr<GGMLBlock>(new Linear(inner_dim, out_context_dim, out_bias, false, false, scale));
        }

        std::pair<ggml_tensor*, ggml_tensor*> forward(GGMLRunnerContext* ctx,
                                                      struct ggml_tensor* img,
                                                      struct ggml_tensor* txt,
                                                      struct ggml_tensor* pe,
                                                      struct ggml_tensor* mask = nullptr) {
            // img: [N, n_img_token, hidden_size]
            // txt: [N, n_txt_token, hidden_size]
            // pe: [n_img_token + n_txt_token, d_head/2, 2, 2]
            // return: ([N, n_img_token, hidden_size], [N, n_txt_token, hidden_size])

            auto norm_q = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_q"]);
            auto norm_k = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_k"]);

            auto to_q     = std::dynamic_pointer_cast<Linear>(blocks["to_q"]);
            auto to_k     = std::dynamic_pointer_cast<Linear>(blocks["to_k"]);
            auto to_v     = std::dynamic_pointer_cast<Linear>(blocks["to_v"]);
            auto to_out_0 = std::dynamic_pointer_cast<Linear>(blocks["to_out.0"]);

            auto norm_added_q = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_added_q"]);
            auto norm_added_k = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_added_k"]);

            auto add_q_proj = std::dynamic_pointer_cast<Linear>(blocks["add_q_proj"]);
            auto add_k_proj = std::dynamic_pointer_cast<Linear>(blocks["add_k_proj"]);
            auto add_v_proj = std::dynamic_pointer_cast<Linear>(blocks["add_v_proj"]);
            auto to_add_out = std::dynamic_pointer_cast<Linear>(blocks["to_add_out"]);

            int64_t N           = img->ne[2];
            int64_t n_img_token = img->ne[1];
            int64_t n_txt_token = txt->ne[1];

            auto img_q        = to_q->forward(ctx, img);
            int64_t num_heads = img_q->ne[0] / dim_head;
            img_q             = ggml_reshape_4d(ctx->ggml_ctx, img_q, dim_head, num_heads, n_img_token, N);  // [N, n_img_token, n_head, d_head]
            auto img_k        = to_k->forward(ctx, img);
            img_k             = ggml_reshape_4d(ctx->ggml_ctx, img_k, dim_head, num_heads, n_img_token, N);  // [N, n_img_token, n_head, d_head]
            auto img_v        = to_v->forward(ctx, img);
            img_v             = ggml_reshape_4d(ctx->ggml_ctx, img_v, dim_head, num_heads, n_img_token, N);  // [N, n_img_token, n_head, d_head]

            img_q = norm_q->forward(ctx, img_q);
            img_k = norm_k->forward(ctx, img_k);

            auto txt_q = add_q_proj->forward(ctx, txt);
            txt_q      = ggml_reshape_4d(ctx->ggml_ctx, txt_q, dim_head, num_heads, n_txt_token, N);  // [N, n_txt_token, n_head, d_head]
            auto txt_k = add_k_proj->forward(ctx, txt);
            txt_k      = ggml_reshape_4d(ctx->ggml_ctx, txt_k, dim_head, num_heads, n_txt_token, N);  // [N, n_txt_token, n_head, d_head]
            auto txt_v = add_v_proj->forward(ctx, txt);
            txt_v      = ggml_reshape_4d(ctx->ggml_ctx, txt_v, dim_head, num_heads, n_txt_token, N);  // [N, n_txt_token, n_head, d_head]

            txt_q = norm_added_q->forward(ctx, txt_q);
            txt_k = norm_added_k->forward(ctx, txt_k);

            auto q = ggml_concat(ctx->ggml_ctx, txt_q, img_q, 2);  // [N, n_txt_token + n_img_token, n_head, d_head]
            auto k = ggml_concat(ctx->ggml_ctx, txt_k, img_k, 2);  // [N, n_txt_token + n_img_token, n_head, d_head]
            auto v = ggml_concat(ctx->ggml_ctx, txt_v, img_v, 2);  // [N, n_txt_token + n_img_token, n_head, d_head]

            auto attn         = Rope::attention(ctx, q, k, v, pe, mask, (1.0f / 128.f));  // [N, n_txt_token + n_img_token, n_head*d_head]
            auto txt_attn_out = ggml_view_3d(ctx->ggml_ctx,
                                             attn,
                                             attn->ne[0],
                                             txt->ne[1],
                                             attn->ne[2],
                                             attn->nb[1],
                                             attn->nb[2],
                                             0);  // [N, n_txt_token, n_head*d_head]
            auto img_attn_out = ggml_view_3d(ctx->ggml_ctx,
                                             attn,
                                             attn->ne[0],
                                             img->ne[1],
                                             attn->ne[2],
                                             attn->nb[1],
                                             attn->nb[2],
                                             txt->ne[1] * attn->nb[1]);  // [N, n_img_token, n_head*d_head]
            img_attn_out      = ggml_cont(ctx->ggml_ctx, img_attn_out);
            txt_attn_out      = ggml_cont(ctx->ggml_ctx, txt_attn_out);

            img_attn_out = to_out_0->forward(ctx, img_attn_out);
            txt_attn_out = to_add_out->forward(ctx, txt_attn_out);

            return {img_attn_out, txt_attn_out};
        }
    };

    class QwenImageTransformerBlock : public GGMLBlock {
    protected:
        bool zero_cond_t;

    public:
        QwenImageTransformerBlock(int64_t dim,
                                  int64_t num_attention_heads,
                                  int64_t attention_head_dim,
                                  float eps        = 1e-6,
                                  bool zero_cond_t = false)
            : zero_cond_t(zero_cond_t) {
            // img_mod.0 is nn.SiLU()
            blocks["img_mod.1"] = std::shared_ptr<GGMLBlock>(new Linear(dim, 6 * dim, true));

            blocks["img_norm1"] = std::shared_ptr<GGMLBlock>(new LayerNorm(dim, eps, false));
            blocks["img_norm2"] = std::shared_ptr<GGMLBlock>(new LayerNorm(dim, eps, false));
            blocks["img_mlp"]   = std::shared_ptr<GGMLBlock>(new FeedForward(dim, dim, 4, FeedForward::Activation::GELU, true));

            // txt_mod.0 is nn.SiLU()
            blocks["txt_mod.1"] = std::shared_ptr<GGMLBlock>(new Linear(dim, 6 * dim, true));

            blocks["txt_norm1"] = std::shared_ptr<GGMLBlock>(new LayerNorm(dim, eps, false));
            blocks["txt_norm2"] = std::shared_ptr<GGMLBlock>(new LayerNorm(dim, eps, false));
            blocks["txt_mlp"]   = std::shared_ptr<GGMLBlock>(new FeedForward(dim, dim, 4, FeedForward::Activation::GELU, true));

            blocks["attn"] = std::shared_ptr<GGMLBlock>(new QwenImageAttention(dim,
                                                                               attention_head_dim,
                                                                               num_attention_heads,
                                                                               0,     // out_dim
                                                                               0,     // out_context-dim
                                                                               true,  // bias
                                                                               true,  // out_bias
                                                                               eps));
        }

        std::vector<ggml_tensor*> get_mod_params_vec(ggml_context* ctx, ggml_tensor* mod_params, ggml_tensor* index = nullptr) {
            // index: [N, n_img_token]
            // mod_params: [N, hidden_size * 12]
            if (index == nullptr) {
                return ggml_ext_chunk(ctx, mod_params, 6, 0);
            }
            mod_params          = ggml_reshape_1d(ctx, mod_params, ggml_nelements(mod_params));
            auto mod_params_vec = ggml_ext_chunk(ctx, mod_params, 12, 0);
            index               = ggml_reshape_3d(ctx, index, 1, index->ne[0], index->ne[1]);                                      // [N, n_img_token, 1]
            index               = ggml_repeat_4d(ctx, index, mod_params_vec[0]->ne[0], index->ne[1], index->ne[2], index->ne[3]);  // [N, n_img_token, hidden_size]
            std::vector<ggml_tensor*> mod_results;
            for (int i = 0; i < 6; i++) {
                auto mod_0 = mod_params_vec[i];
                auto mod_1 = mod_params_vec[i + 6];

                // mod_result = torch.where(index == 0, mod_0, mod_1)
                // mod_result = (1 - index)*mod_0 + index*mod_1
                mod_0           = ggml_sub(ctx, ggml_repeat(ctx, mod_0, index), ggml_mul(ctx, index, mod_0));  // [N, n_img_token, hidden_size]
                mod_1           = ggml_mul(ctx, index, mod_1);                                                 // [N, n_img_token, hidden_size]
                auto mod_result = ggml_add(ctx, mod_0, mod_1);
                mod_results.push_back(mod_result);
            }
            return mod_results;
        }

        virtual std::pair<ggml_tensor*, ggml_tensor*> forward(GGMLRunnerContext* ctx,
                                                              struct ggml_tensor* img,
                                                              struct ggml_tensor* txt,
                                                              struct ggml_tensor* t_emb,
                                                              struct ggml_tensor* pe,
                                                              struct ggml_tensor* modulate_index = nullptr) {
            // img: [N, n_img_token, hidden_size]
            // txt: [N, n_txt_token, hidden_size]
            // pe: [n_img_token + n_txt_token, d_head/2, 2, 2]
            // return: ([N, n_img_token, hidden_size], [N, n_txt_token, hidden_size])

            auto img_mod_1 = std::dynamic_pointer_cast<Linear>(blocks["img_mod.1"]);
            auto img_norm1 = std::dynamic_pointer_cast<LayerNorm>(blocks["img_norm1"]);
            auto img_norm2 = std::dynamic_pointer_cast<LayerNorm>(blocks["img_norm2"]);
            auto img_mlp   = std::dynamic_pointer_cast<FeedForward>(blocks["img_mlp"]);

            auto txt_mod_1 = std::dynamic_pointer_cast<Linear>(blocks["txt_mod.1"]);
            auto txt_norm1 = std::dynamic_pointer_cast<LayerNorm>(blocks["txt_norm1"]);
            auto txt_norm2 = std::dynamic_pointer_cast<LayerNorm>(blocks["txt_norm2"]);
            auto txt_mlp   = std::dynamic_pointer_cast<FeedForward>(blocks["txt_mlp"]);

            auto attn = std::dynamic_pointer_cast<QwenImageAttention>(blocks["attn"]);

            auto img_mod_params    = ggml_silu(ctx->ggml_ctx, t_emb);
            img_mod_params         = img_mod_1->forward(ctx, img_mod_params);
            auto img_mod_param_vec = get_mod_params_vec(ctx->ggml_ctx, img_mod_params, modulate_index);

            if (zero_cond_t) {
                t_emb = ggml_ext_chunk(ctx->ggml_ctx, t_emb, 2, 1)[0];
            }

            auto txt_mod_params    = ggml_silu(ctx->ggml_ctx, t_emb);
            txt_mod_params         = txt_mod_1->forward(ctx, txt_mod_params);
            auto txt_mod_param_vec = get_mod_params_vec(ctx->ggml_ctx, txt_mod_params);

            auto img_normed    = img_norm1->forward(ctx, img);
            auto img_modulated = Flux::modulate(ctx->ggml_ctx, img_normed, img_mod_param_vec[0], img_mod_param_vec[1], modulate_index != nullptr);
            auto img_gate1     = img_mod_param_vec[2];

            auto txt_normed    = txt_norm1->forward(ctx, txt);
            auto txt_modulated = Flux::modulate(ctx->ggml_ctx, txt_normed, txt_mod_param_vec[0], txt_mod_param_vec[1]);
            auto txt_gate1     = txt_mod_param_vec[2];

            auto [img_attn_output, txt_attn_output] = attn->forward(ctx, img_modulated, txt_modulated, pe);

            img = ggml_add(ctx->ggml_ctx, img, ggml_mul(ctx->ggml_ctx, img_attn_output, img_gate1));
            txt = ggml_add(ctx->ggml_ctx, txt, ggml_mul(ctx->ggml_ctx, txt_attn_output, txt_gate1));

            auto img_normed2    = img_norm2->forward(ctx, img);
            auto img_modulated2 = Flux::modulate(ctx->ggml_ctx, img_normed2, img_mod_param_vec[3], img_mod_param_vec[4], modulate_index != nullptr);
            auto img_gate2      = img_mod_param_vec[5];

            auto txt_normed2    = txt_norm2->forward(ctx, txt);
            auto txt_modulated2 = Flux::modulate(ctx->ggml_ctx, txt_normed2, txt_mod_param_vec[3], txt_mod_param_vec[4]);
            auto txt_gate2      = txt_mod_param_vec[5];

            auto img_mlp_out = img_mlp->forward(ctx, img_modulated2);
            auto txt_mlp_out = txt_mlp->forward(ctx, txt_modulated2);

            img = ggml_add(ctx->ggml_ctx, img, ggml_mul(ctx->ggml_ctx, img_mlp_out, img_gate2));
            txt = ggml_add(ctx->ggml_ctx, txt, ggml_mul(ctx->ggml_ctx, txt_mlp_out, txt_gate2));

            return {img, txt};
        }
    };

    struct AdaLayerNormContinuous : public GGMLBlock {
    public:
        AdaLayerNormContinuous(int64_t embedding_dim,
                               int64_t conditioning_embedding_dim,
                               bool elementwise_affine = true,
                               float eps               = 1e-5f,
                               bool bias               = true) {
            blocks["norm"]   = std::shared_ptr<GGMLBlock>(new LayerNorm(conditioning_embedding_dim, eps, elementwise_affine, bias));
            blocks["linear"] = std::shared_ptr<GGMLBlock>(new Linear(conditioning_embedding_dim, embedding_dim * 2, bias));
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* c) {
            // x: [N, n_token, hidden_size]
            // c: [N, hidden_size]
            // return: [N, n_token, patch_size * patch_size * out_channels]

            auto norm   = std::dynamic_pointer_cast<LayerNorm>(blocks["norm"]);
            auto linear = std::dynamic_pointer_cast<Linear>(blocks["linear"]);

            auto emb   = linear->forward(ctx, ggml_silu(ctx->ggml_ctx, c));
            auto mods  = ggml_ext_chunk(ctx->ggml_ctx, emb, 2, 0);
            auto scale = mods[0];
            auto shift = mods[1];

            x = norm->forward(ctx, x);
            x = Flux::modulate(ctx->ggml_ctx, x, shift, scale);

            return x;
        }
    };

    struct QwenImageParams {
        int patch_size              = 2;
        int64_t in_channels         = 64;
        int64_t out_channels        = 16;
        int num_layers              = 60;
        int64_t attention_head_dim  = 128;
        int64_t num_attention_heads = 24;
        int64_t joint_attention_dim = 3584;
        int theta                   = 10000;
        std::vector<int> axes_dim   = {16, 56, 56};
        int axes_dim_sum            = 128;
        bool zero_cond_t            = false;
    };

    class QwenImageModel : public GGMLBlock {
    protected:
        QwenImageParams params;

    public:
        QwenImageModel() {}
        QwenImageModel(QwenImageParams params)
            : params(params) {
            int64_t inner_dim         = params.num_attention_heads * params.attention_head_dim;
            blocks["time_text_embed"] = std::shared_ptr<GGMLBlock>(new QwenTimestepProjEmbeddings(inner_dim));
            blocks["txt_norm"]        = std::shared_ptr<GGMLBlock>(new RMSNorm(params.joint_attention_dim, 1e-6f));
            blocks["img_in"]          = std::shared_ptr<GGMLBlock>(new Linear(params.in_channels, inner_dim));
            blocks["txt_in"]          = std::shared_ptr<GGMLBlock>(new Linear(params.joint_attention_dim, inner_dim));

            // blocks
            for (int i = 0; i < params.num_layers; i++) {
                auto block                                        = std::shared_ptr<GGMLBlock>(new QwenImageTransformerBlock(inner_dim,
                                                                                                                             params.num_attention_heads,
                                                                                                                             params.attention_head_dim,
                                                                                                                             1e-6f,
                                                                                                                             params.zero_cond_t));
                blocks["transformer_blocks." + std::to_string(i)] = block;
            }

            blocks["norm_out"] = std::shared_ptr<GGMLBlock>(new AdaLayerNormContinuous(inner_dim, inner_dim, false, 1e-6f));
            blocks["proj_out"] = std::shared_ptr<GGMLBlock>(new Linear(inner_dim, params.patch_size * params.patch_size * params.out_channels));
        }

        struct ggml_tensor* forward_orig(GGMLRunnerContext* ctx,
                                         struct ggml_tensor* x,
                                         struct ggml_tensor* timestep,
                                         struct ggml_tensor* context,
                                         struct ggml_tensor* pe,
                                         struct ggml_tensor* modulate_index = nullptr) {
            auto time_text_embed = std::dynamic_pointer_cast<QwenTimestepProjEmbeddings>(blocks["time_text_embed"]);
            auto txt_norm        = std::dynamic_pointer_cast<RMSNorm>(blocks["txt_norm"]);
            auto img_in          = std::dynamic_pointer_cast<Linear>(blocks["img_in"]);
            auto txt_in          = std::dynamic_pointer_cast<Linear>(blocks["txt_in"]);
            auto norm_out        = std::dynamic_pointer_cast<AdaLayerNormContinuous>(blocks["norm_out"]);
            auto proj_out        = std::dynamic_pointer_cast<Linear>(blocks["proj_out"]);

            auto t_emb = time_text_embed->forward(ctx, timestep);
            if (params.zero_cond_t) {
                auto t_emb_0 = time_text_embed->forward(ctx, ggml_ext_zeros_like(ctx->ggml_ctx, timestep));
                t_emb        = ggml_concat(ctx->ggml_ctx, t_emb, t_emb_0, 1);
            }
            auto img = img_in->forward(ctx, x);
            auto txt = txt_norm->forward(ctx, context);
            txt      = txt_in->forward(ctx, txt);

            for (int i = 0; i < params.num_layers; i++) {
                auto block = std::dynamic_pointer_cast<QwenImageTransformerBlock>(blocks["transformer_blocks." + std::to_string(i)]);

                auto result = block->forward(ctx, img, txt, t_emb, pe, modulate_index);
                img         = result.first;
                txt         = result.second;
            }

            if (params.zero_cond_t) {
                t_emb = ggml_ext_chunk(ctx->ggml_ctx, t_emb, 2, 1)[0];
            }

            img = norm_out->forward(ctx, img, t_emb);
            img = proj_out->forward(ctx, img);

            return img;
        }

        struct StreamingInputResult {
            ggml_tensor* img;
            ggml_tensor* txt;
            ggml_tensor* t_emb;
        };

        StreamingInputResult forward_input_stage(GGMLRunnerContext* ctx,
                                                  struct ggml_tensor* x,
                                                  struct ggml_tensor* timestep,
                                                  struct ggml_tensor* context,
                                                  std::vector<ggml_tensor*> ref_latents = {},
                                                  int64_t* out_img_tokens = nullptr) {
            auto time_text_embed = std::dynamic_pointer_cast<QwenTimestepProjEmbeddings>(blocks["time_text_embed"]);
            auto txt_norm        = std::dynamic_pointer_cast<RMSNorm>(blocks["txt_norm"]);
            auto img_in          = std::dynamic_pointer_cast<Linear>(blocks["img_in"]);
            auto txt_in          = std::dynamic_pointer_cast<Linear>(blocks["txt_in"]);

            auto t_emb = time_text_embed->forward(ctx, timestep);
            if (params.zero_cond_t) {
                auto t_emb_0 = time_text_embed->forward(ctx, ggml_ext_zeros(ctx->ggml_ctx, timestep->ne[0], timestep->ne[1], timestep->ne[2], timestep->ne[3]));
                t_emb        = ggml_concat(ctx->ggml_ctx, t_emb, t_emb_0, 1);
            }

            // Patchify input (same as main forward())
            auto img_patched = DiT::pad_and_patchify(ctx, x, params.patch_size, params.patch_size);
            int64_t img_tokens = img_patched->ne[1];

            // Handle reference latents
            if (ref_latents.size() > 0) {
                for (ggml_tensor* ref : ref_latents) {
                    ref = DiT::pad_and_patchify(ctx, ref, params.patch_size, params.patch_size);
                    img_patched = ggml_concat(ctx->ggml_ctx, img_patched, ref, 1);
                }
            }

            auto img = img_in->forward(ctx, img_patched);
            auto txt = txt_norm->forward(ctx, context);
            txt      = txt_in->forward(ctx, txt);

            if (out_img_tokens) {
                *out_img_tokens = img_tokens;
            }

            return {img, txt, t_emb};
        }

        std::pair<ggml_tensor*, ggml_tensor*> forward_single_block(GGMLRunnerContext* ctx,
                                                                    int block_idx,
                                                                    struct ggml_tensor* img,
                                                                    struct ggml_tensor* txt,
                                                                    struct ggml_tensor* t_emb,
                                                                    struct ggml_tensor* pe,
                                                                    struct ggml_tensor* modulate_index = nullptr) {
            auto block = std::dynamic_pointer_cast<QwenImageTransformerBlock>(blocks["transformer_blocks." + std::to_string(block_idx)]);
            return block->forward(ctx, img, txt, t_emb, pe, modulate_index);
        }

        struct ggml_tensor* forward_output_stage(GGMLRunnerContext* ctx,
                                                  struct ggml_tensor* img,
                                                  struct ggml_tensor* t_emb,
                                                  int64_t img_tokens,
                                                  int64_t orig_H,
                                                  int64_t orig_W) {
            auto norm_out = std::dynamic_pointer_cast<AdaLayerNormContinuous>(blocks["norm_out"]);
            auto proj_out = std::dynamic_pointer_cast<Linear>(blocks["proj_out"]);

            if (params.zero_cond_t) {
                t_emb = ggml_ext_chunk(ctx->ggml_ctx, t_emb, 2, 1)[0];
            }

            // Trim to original img_tokens if ref_latents were used
            if (img->ne[1] > img_tokens) {
                img = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, img, 0, 2, 1, 3));
                img = ggml_view_3d(ctx->ggml_ctx, img, img->ne[0], img->ne[1], img_tokens, img->nb[1], img->nb[2], 0);
                img = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, img, 0, 2, 1, 3));
            }

            img = norm_out->forward(ctx, img, t_emb);
            img = proj_out->forward(ctx, img);

            // Unpatchify and crop
            img = DiT::unpatchify_and_crop(ctx->ggml_ctx, img, orig_H, orig_W, params.patch_size, params.patch_size);

            return img;
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* timestep,
                                    struct ggml_tensor* context,
                                    struct ggml_tensor* pe,
                                    std::vector<ggml_tensor*> ref_latents = {},
                                    struct ggml_tensor* modulate_index    = nullptr) {
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

            auto img           = DiT::pad_and_patchify(ctx, x, params.patch_size, params.patch_size);
            int64_t img_tokens = img->ne[1];

            if (ref_latents.size() > 0) {
                for (ggml_tensor* ref : ref_latents) {
                    ref = DiT::pad_and_patchify(ctx, ref, params.patch_size, params.patch_size);
                    img = ggml_concat(ctx->ggml_ctx, img, ref, 1);
                }
            }

            auto out = forward_orig(ctx, img, timestep, context, pe, modulate_index);  // [N, h_len*w_len, ph*pw*C]

            if (out->ne[1] > img_tokens) {
                out = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, out, 0, 2, 1, 3));  // [num_tokens, N, C * patch_size * patch_size]
                out = ggml_view_3d(ctx->ggml_ctx, out, out->ne[0], out->ne[1], img_tokens, out->nb[1], out->nb[2], 0);
                out = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, out, 0, 2, 1, 3));  // [N, h*w, C * patch_size * patch_size]
            }

            out = DiT::unpatchify_and_crop(ctx->ggml_ctx, out, H, W, params.patch_size, params.patch_size);  // [N, C, H, W]

            return out;
        }
    };

    struct QwenImageRunner : public GGMLRunner {
    public:
        QwenImageParams qwen_image_params;
        QwenImageModel qwen_image;
        std::vector<float> pe_vec;
        std::vector<float> modulate_index_vec;
        SDVersion version;

        QwenImageRunner(ggml_backend_t backend,
                        bool offload_params_to_cpu,
                        const String2TensorStorage& tensor_storage_map = {},
                        const std::string prefix                       = "",
                        SDVersion version                              = VERSION_QWEN_IMAGE,
                        bool zero_cond_t                               = false)
            : GGMLRunner(backend, offload_params_to_cpu) {
            qwen_image_params.num_layers  = 0;
            qwen_image_params.zero_cond_t = zero_cond_t;
            for (auto pair : tensor_storage_map) {
                std::string tensor_name = pair.first;
                if (tensor_name.find(prefix) == std::string::npos)
                    continue;
                if (tensor_name.find("__index_timestep_zero__") != std::string::npos) {
                    qwen_image_params.zero_cond_t = true;
                }
                size_t pos = tensor_name.find("transformer_blocks.");
                if (pos != std::string::npos) {
                    tensor_name = tensor_name.substr(pos);  // remove prefix
                    auto items  = split_string(tensor_name, '.');
                    if (items.size() > 1) {
                        int block_index = atoi(items[1].c_str());
                        if (block_index + 1 > qwen_image_params.num_layers) {
                            qwen_image_params.num_layers = block_index + 1;
                        }
                    }
                    continue;
                }
            }
            LOG_INFO("qwen_image_params.num_layers: %ld", qwen_image_params.num_layers);
            if (qwen_image_params.zero_cond_t) {
                LOG_INFO("use zero_cond_t");
            }
            qwen_image = QwenImageModel(qwen_image_params);
            qwen_image.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return "qwen_image";
        }

        void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
            qwen_image.get_param_tensors(tensors, prefix);
        }

    public:
        void enable_layer_streaming(const LayerStreaming::StreamingConfig& config = {}) {
            std::map<std::string, ggml_tensor*> tensor_map;
            qwen_image.get_param_tensors(tensor_map, "model.diffusion_model");
            init_streaming(config, tensor_map, LayerStreaming::qwen_image_layer_pattern);
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

            return compute_streaming_true(n_threads, x, timesteps, context, ref_latents, increase_ref_index, output, output_ctx);
        }

    private:
        // Persistent storage for intermediate tensors between layer executions
        struct StreamingState {
            std::vector<float> img_data;
            std::vector<float> txt_data;
            std::vector<float> t_emb_data;
            std::vector<float> pe_data;
            std::vector<float> modulate_index_data;

            // Tensor dimensions
            int64_t img_ne[4];
            int64_t txt_ne[4];
            int64_t t_emb_ne[4];
            int64_t pe_ne[4];
            int64_t modulate_index_ne[4];
            bool has_modulate_index = false;
        };

        void copy_tensor_to_storage(ggml_tensor* tensor, std::vector<float>& storage, int64_t* ne) {
            size_t nelements = ggml_nelements(tensor);
            storage.resize(nelements);

            // Copy to CPU if needed
            ggml_backend_tensor_get(tensor, storage.data(), 0, nelements * sizeof(float));

            // Store dimensions
            for (int i = 0; i < 4; i++) {
                ne[i] = tensor->ne[i];
            }
        }

        ggml_tensor* create_tensor_from_storage(ggml_context* ctx, const std::vector<float>& storage,
                                                 const int64_t* ne, const char* name) {
            ggml_tensor* tensor = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]);
            ggml_set_name(tensor, name);
            return tensor;
        }

        bool compute_streaming_true(int n_threads,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* timesteps,
                                    struct ggml_tensor* context,
                                    std::vector<ggml_tensor*> ref_latents,
                                    bool increase_ref_index,
                                    struct ggml_tensor** output,
                                    struct ggml_context* output_ctx) {
            auto& registry = streaming_engine_->get_registry();
            int64_t t_start = ggml_time_ms();

            const int num_layers = qwen_image_params.num_layers;
            LOG_INFO("TRUE per-layer streaming - %d blocks (one at a time)", num_layers);

            // Phase 1: Load global layers (_global contains input/output projections)
            LOG_DEBUG("Loading global layers");
            if (!registry.move_layer_to_gpu("_global")) {
                LOG_ERROR("Failed to load _global to GPU");
                return false;
            }

            // Pre-generate PE and modulate_index vectors (needed for all blocks)
            pe_vec = Rope::gen_qwen_image_pe(static_cast<int>(x->ne[1]),
                                              static_cast<int>(x->ne[0]),
                                              qwen_image_params.patch_size,
                                              static_cast<int>(x->ne[3]),
                                              static_cast<int>(context->ne[1]),
                                              ref_latents,
                                              increase_ref_index,
                                              qwen_image_params.theta,
                                              circular_y_enabled,
                                              circular_x_enabled,
                                              qwen_image_params.axes_dim);

            if (qwen_image_params.zero_cond_t) {
                modulate_index_vec.clear();
                int64_t h_len = ((x->ne[1] + (qwen_image_params.patch_size / 2)) / qwen_image_params.patch_size);
                int64_t w_len = ((x->ne[0] + (qwen_image_params.patch_size / 2)) / qwen_image_params.patch_size);
                int64_t num_img_tokens = h_len * w_len;
                modulate_index_vec.insert(modulate_index_vec.end(), num_img_tokens, 0.f);

                int64_t num_ref_img_tokens = 0;
                for (ggml_tensor* ref : ref_latents) {
                    int64_t rh_len = ((ref->ne[1] + (qwen_image_params.patch_size / 2)) / qwen_image_params.patch_size);
                    int64_t rw_len = ((ref->ne[0] + (qwen_image_params.patch_size / 2)) / qwen_image_params.patch_size);
                    num_ref_img_tokens += rh_len * rw_len;
                }
                if (num_ref_img_tokens > 0) {
                    modulate_index_vec.insert(modulate_index_vec.end(), num_ref_img_tokens, 1.f);
                }
            }

            // TRUE per-layer streaming with mini-graphs
            // Execute each block as a separate mini-graph to minimize activation memory

            int64_t t_blocks_start = ggml_time_ms();

            // Store original image dimensions for unpatchify
            int64_t orig_H = x->ne[1];
            int64_t orig_W = x->ne[0];

            // Persistent storage for intermediate img and txt tensors
            std::vector<float> persistent_img;
            std::vector<float> persistent_txt;
            std::vector<float> persistent_t_emb;
            int64_t img_ne[4], txt_ne[4], t_emb_ne[4];
            int64_t img_tokens_count = 0;

            LOG_DEBUG("Executing input stage");
            {
                // Build mini-graph for input projections only
                struct ggml_cgraph* input_graph = nullptr;
                ggml_tensor* img_output = nullptr;
                ggml_tensor* txt_output = nullptr;
                ggml_tensor* t_emb_output = nullptr;
                int64_t img_tokens_local = 0;

                auto get_input_graph = [&]() -> struct ggml_cgraph* {
                    struct ggml_cgraph* gf = new_graph_custom(QWEN_IMAGE_GRAPH_SIZE / 4);  // Smaller graph

                    ggml_tensor* x_backend = to_backend(x);
                    ggml_tensor* context_backend = to_backend(context);
                    ggml_tensor* timesteps_backend = to_backend(timesteps);

                    // Convert ref_latents to backend
                    std::vector<ggml_tensor*> ref_latents_backend;
                    for (auto& ref : ref_latents) {
                        ref_latents_backend.push_back(to_backend(ref));
                    }

                    auto runner_ctx = get_context();
                    auto result = qwen_image.forward_input_stage(&runner_ctx, x_backend, timesteps_backend, context_backend,
                                                                  ref_latents_backend, &img_tokens_local);

                    img_output = result.img;
                    txt_output = result.txt;
                    t_emb_output = result.t_emb;

                    // Concatenate outputs into single tensor for extraction
                    // We'll use img as the primary output and extract separately
                    ggml_build_forward_expand(gf, result.img);
                    ggml_build_forward_expand(gf, result.txt);
                    ggml_build_forward_expand(gf, result.t_emb);

                    return gf;
                };

                // Execute input stage - don't free compute buffer immediately
                if (!GGMLRunner::compute(get_input_graph, n_threads, false, nullptr, nullptr, true)) {
                    LOG_ERROR("Input stage failed");
                    return false;
                }

                img_tokens_count = img_tokens_local;

                // Extract computed tensors to persistent storage
                if (img_output && txt_output && t_emb_output) {
                    // Copy tensor data to CPU storage
                    size_t img_size = ggml_nelements(img_output);
                    size_t txt_size = ggml_nelements(txt_output);
                    size_t t_emb_size = ggml_nelements(t_emb_output);

                    persistent_img.resize(img_size);
                    persistent_txt.resize(txt_size);
                    persistent_t_emb.resize(t_emb_size);

                    ggml_backend_tensor_get(img_output, persistent_img.data(), 0, img_size * sizeof(float));
                    ggml_backend_tensor_get(txt_output, persistent_txt.data(), 0, txt_size * sizeof(float));
                    ggml_backend_tensor_get(t_emb_output, persistent_t_emb.data(), 0, t_emb_size * sizeof(float));

                    for (int i = 0; i < 4; i++) {
                        img_ne[i] = img_output->ne[i];
                        txt_ne[i] = txt_output->ne[i];
                        t_emb_ne[i] = t_emb_output->ne[i];
                    }
                } else {
                    LOG_ERROR("Failed to get input stage outputs");
                    free_compute_buffer();
                    return false;
                }

                // Now safe to free compute buffer
                free_compute_buffer();
            }

            LOG_DEBUG("Input stage done, img=%ldx%ldx%ldx%ld, txt=%ldx%ldx%ldx%ld",
                      img_ne[0], img_ne[1], img_ne[2], img_ne[3],
                      txt_ne[0], txt_ne[1], txt_ne[2], txt_ne[3]);

            // Start prefetching the first block
            std::string first_block_name = "transformer_blocks.0";
            streaming_engine_->prefetch_layer(first_block_name);

            for (int block_idx = 0; block_idx < num_layers; block_idx++) {
                std::string block_name = "transformer_blocks." + std::to_string(block_idx);
                int64_t t_block_start = ggml_time_ms();

                // Wait for this block's prefetch to complete (if it was prefetched)
                streaming_engine_->wait_for_prefetch(block_name);

                // Load this block's weights (sync load if prefetch didn't happen)
                if (!registry.move_layer_to_gpu(block_name)) {
                    LOG_ERROR("Failed to load block %d", block_idx);
                    return false;
                }

                // Start async prefetch of the NEXT block while we compute this one
                // This overlaps memory transfer with GPU computation
                if (block_idx + 1 < num_layers) {
                    std::string next_block_name = "transformer_blocks." + std::to_string(block_idx + 1);
                    streaming_engine_->prefetch_layer(next_block_name);
                }

                // Build and execute mini-graph for this block
                ggml_tensor* img_out = nullptr;
                ggml_tensor* txt_out = nullptr;

                auto get_block_graph = [&]() -> struct ggml_cgraph* {
                    struct ggml_cgraph* gf = new_graph_custom(QWEN_IMAGE_GRAPH_SIZE / 4);

                    // Create input tensors from persistent storage
                    ggml_tensor* img_in = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, img_ne[0], img_ne[1], img_ne[2], img_ne[3]);
                    ggml_tensor* txt_in = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, txt_ne[0], txt_ne[1], txt_ne[2], txt_ne[3]);
                    ggml_tensor* t_emb_in = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, t_emb_ne[0], t_emb_ne[1], t_emb_ne[2], t_emb_ne[3]);

                    // Copy to backend and set data
                    img_in = to_backend(img_in);
                    txt_in = to_backend(txt_in);
                    t_emb_in = to_backend(t_emb_in);

                    set_backend_tensor_data(img_in, persistent_img.data());
                    set_backend_tensor_data(txt_in, persistent_txt.data());
                    set_backend_tensor_data(t_emb_in, persistent_t_emb.data());

                    // Generate PE
                    int pos_len = static_cast<int>(pe_vec.size() / qwen_image_params.axes_dim_sum / 2);
                    ggml_tensor* pe = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, qwen_image_params.axes_dim_sum / 2, pos_len);
                    set_backend_tensor_data(pe, pe_vec.data());

                    // Modulate index
                    ggml_tensor* modulate_index = nullptr;
                    if (qwen_image_params.zero_cond_t && !modulate_index_vec.empty()) {
                        modulate_index = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_F32, modulate_index_vec.size());
                        set_backend_tensor_data(modulate_index, modulate_index_vec.data());
                    }

                    auto runner_ctx = get_context();
                    auto [img_result, txt_result] = qwen_image.forward_single_block(&runner_ctx, block_idx,
                                                                                     img_in, txt_in, t_emb_in, pe, modulate_index);

                    img_out = img_result;
                    txt_out = txt_result;

                    ggml_build_forward_expand(gf, img_out);
                    ggml_build_forward_expand(gf, txt_out);

                    return gf;
                };

                // Don't free compute buffer immediately - we need to read outputs first
                if (!GGMLRunner::compute(get_block_graph, n_threads, false, nullptr, nullptr, true)) {
                    LOG_ERROR("Block %d execution failed", block_idx);
                    return false;
                }

                // Extract outputs to persistent storage
                if (img_out && txt_out) {
                    ggml_backend_tensor_get(img_out, persistent_img.data(), 0, persistent_img.size() * sizeof(float));
                    ggml_backend_tensor_get(txt_out, persistent_txt.data(), 0, persistent_txt.size() * sizeof(float));

                    for (int i = 0; i < 4; i++) {
                        img_ne[i] = img_out->ne[i];
                        txt_ne[i] = txt_out->ne[i];
                    }
                }

                // Now safe to free compute buffer
                free_compute_buffer();

                // Offload this block
                registry.move_layer_to_cpu(block_name);

                LOG_DEBUG("Block %d/%d done (%.2fms)",
                          block_idx + 1, num_layers, (ggml_time_ms() - t_block_start) / 1.0);
            }

            LOG_DEBUG("Executing output stage");
            {
                ggml_tensor* final_out = nullptr;

                auto get_output_graph = [&]() -> struct ggml_cgraph* {
                    struct ggml_cgraph* gf = new_graph_custom(QWEN_IMAGE_GRAPH_SIZE / 4);

                    // Create input tensors
                    ggml_tensor* img_in = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, img_ne[0], img_ne[1], img_ne[2], img_ne[3]);
                    ggml_tensor* t_emb_in = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, t_emb_ne[0], t_emb_ne[1], t_emb_ne[2], t_emb_ne[3]);

                    img_in = to_backend(img_in);
                    t_emb_in = to_backend(t_emb_in);

                    set_backend_tensor_data(img_in, persistent_img.data());
                    set_backend_tensor_data(t_emb_in, persistent_t_emb.data());

                    auto runner_ctx = get_context();
                    final_out = qwen_image.forward_output_stage(&runner_ctx, img_in, t_emb_in,
                                                                 img_tokens_count, orig_H, orig_W);

                    ggml_build_forward_expand(gf, final_out);

                    return gf;
                };

                if (!GGMLRunner::compute(get_output_graph, n_threads, true, output, output_ctx, true)) {
                    LOG_ERROR("Output stage failed");
                    return false;
                }
            }

            int64_t t_end = ggml_time_ms();
            LOG_INFO("TRUE per-layer streaming completed in %.2fs (%d blocks)",
                     (t_end - t_start) / 1000.0, num_layers);

            return true;
        }

    public:

        struct ggml_cgraph* build_graph(struct ggml_tensor* x,
                                        struct ggml_tensor* timesteps,
                                        struct ggml_tensor* context,
                                        std::vector<ggml_tensor*> ref_latents = {},
                                        bool increase_ref_index               = false) {
            GGML_ASSERT(x->ne[3] == 1);
            struct ggml_cgraph* gf = new_graph_custom(QWEN_IMAGE_GRAPH_SIZE);

            x         = to_backend(x);
            context   = to_backend(context);
            timesteps = to_backend(timesteps);

            for (int i = 0; i < ref_latents.size(); i++) {
                ref_latents[i] = to_backend(ref_latents[i]);
            }

            pe_vec      = Rope::gen_qwen_image_pe(static_cast<int>(x->ne[1]),
                                                  static_cast<int>(x->ne[0]),
                                                  qwen_image_params.patch_size,
                                                  static_cast<int>(x->ne[3]),
                                                  static_cast<int>(context->ne[1]),
                                                  ref_latents,
                                                  increase_ref_index,
                                                  qwen_image_params.theta,
                                                  circular_y_enabled,
                                                  circular_x_enabled,
                                                  qwen_image_params.axes_dim);
            int pos_len = static_cast<int>(pe_vec.size() / qwen_image_params.axes_dim_sum / 2);
            // LOG_DEBUG("pos_len %d", pos_len);
            auto pe = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, qwen_image_params.axes_dim_sum / 2, pos_len);
            // pe->data = pe_vec.data();
            // print_ggml_tensor(pe, true, "pe");
            // pe->data = nullptr;
            set_backend_tensor_data(pe, pe_vec.data());

            ggml_tensor* modulate_index = nullptr;
            if (qwen_image_params.zero_cond_t) {
                modulate_index_vec.clear();

                int64_t h_len          = ((x->ne[1] + (qwen_image_params.patch_size / 2)) / qwen_image_params.patch_size);
                int64_t w_len          = ((x->ne[0] + (qwen_image_params.patch_size / 2)) / qwen_image_params.patch_size);
                int64_t num_img_tokens = h_len * w_len;

                modulate_index_vec.insert(modulate_index_vec.end(), num_img_tokens, 0.f);
                int64_t num_ref_img_tokens = 0;
                for (ggml_tensor* ref : ref_latents) {
                    int64_t h_len = ((ref->ne[1] + (qwen_image_params.patch_size / 2)) / qwen_image_params.patch_size);
                    int64_t w_len = ((ref->ne[0] + (qwen_image_params.patch_size / 2)) / qwen_image_params.patch_size);

                    num_ref_img_tokens += h_len * w_len;
                }

                if (num_ref_img_tokens > 0) {
                    modulate_index_vec.insert(modulate_index_vec.end(), num_ref_img_tokens, 1.f);
                }

                modulate_index = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_F32, modulate_index_vec.size());
                set_backend_tensor_data(modulate_index, modulate_index_vec.data());
            }

            auto runner_ctx = get_context();

            struct ggml_tensor* out = qwen_image.forward(&runner_ctx,
                                                         x,
                                                         timesteps,
                                                         context,
                                                         pe,
                                                         ref_latents,
                                                         modulate_index);

            ggml_build_forward_expand(gf, out);

            return gf;
        }

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
            auto get_graph = [&]() -> struct ggml_cgraph* {
                return build_graph(x, timesteps, context, ref_latents, increase_ref_index);
            };

            return GGMLRunner::compute(get_graph, n_threads, false, output, output_ctx, skip_param_offload);
        }

        void test() {
            struct ggml_init_params params;
            params.mem_size   = static_cast<size_t>(1024 * 1024) * 1024;  // 1GB
            params.mem_buffer = nullptr;
            params.no_alloc   = false;

            struct ggml_context* work_ctx = ggml_init(params);
            GGML_ASSERT(work_ctx != nullptr);

            {
                // auto x = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, 16, 16, 16, 1);
                // ggml_set_f32(x, 0.01f);
                auto x = load_tensor_from_file(work_ctx, "./qwen_image_x.bin");
                print_ggml_tensor(x);

                std::vector<float> timesteps_vec(1, 1000.f);
                auto timesteps = vector_to_ggml_tensor(work_ctx, timesteps_vec);

                // auto context = ggml_new_tensor_3d(work_ctx, GGML_TYPE_F32, 3584, 256, 1);
                // ggml_set_f32(context, 0.01f);
                auto context = load_tensor_from_file(work_ctx, "./qwen_image_context.bin");
                print_ggml_tensor(context);

                struct ggml_tensor* out = nullptr;

                int64_t t0 = ggml_time_ms();
                compute(8, x, timesteps, context, {}, false, &out, work_ctx);
                int64_t t1 = ggml_time_ms();

                print_ggml_tensor(out);
                LOG_DEBUG("qwen_image test done in %lldms", t1 - t0);
            }
        }

        static void load_from_file_and_test(const std::string& file_path) {
            // cuda q8: pass
            // cuda q8 fa: pass
            // ggml_backend_t backend    = ggml_backend_cuda_init(0);
            ggml_backend_t backend    = ggml_backend_cpu_init();
            ggml_type model_data_type = GGML_TYPE_Q8_0;

            ModelLoader model_loader;
            if (!model_loader.init_from_file_and_convert_name(file_path, "model.diffusion_model.")) {
                LOG_ERROR("init model loader from file failed: '%s'", file_path.c_str());
                return;
            }

            auto& tensor_storage_map = model_loader.get_tensor_storage_map();
            for (auto& [name, tensor_storage] : tensor_storage_map) {
                if (ends_with(name, "weight")) {
                    tensor_storage.expected_type = model_data_type;
                }
            }

            std::shared_ptr<QwenImageRunner> qwen_image = std::make_shared<QwenImageRunner>(backend,
                                                                                            false,
                                                                                            tensor_storage_map,
                                                                                            "model.diffusion_model",
                                                                                            VERSION_QWEN_IMAGE);

            qwen_image->alloc_params_buffer();
            std::map<std::string, ggml_tensor*> tensors;
            qwen_image->get_param_tensors(tensors, "model.diffusion_model");

            bool success = model_loader.load_tensors(tensors);

            if (!success) {
                LOG_ERROR("load tensors from model loader failed");
                return;
            }

            LOG_INFO("qwen_image model loaded");
            qwen_image->test();
        }
    };

}  // namespace name

#endif  // __QWEN_IMAGE_HPP__
