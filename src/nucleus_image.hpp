#ifndef __NUCLEUS_IMAGE_HPP__
#define __NUCLEUS_IMAGE_HPP__

#include "common_block.hpp"
#include "flux.hpp"
#include "ggml_extend.hpp"
#include "mmdit.hpp"
#include "z_image.hpp"

namespace NucleusImage {
    struct MLPEmbedder : public UnaryBlock {
    public:
        MLPEmbedder(int64_t in_dim, int64_t hidden_dim, bool bias = true) {
            blocks["linear_1"] = std::shared_ptr<GGMLBlock>(new Linear(in_dim, hidden_dim, bias));
            blocks["linear_2"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_dim, hidden_dim, bias));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            // x: [..., in_dim]
            // return: [..., hidden_dim]
            auto in_layer  = std::dynamic_pointer_cast<Linear>(blocks["linear_1"]);
            auto out_layer = std::dynamic_pointer_cast<Linear>(blocks["linear_2"]);

            x = in_layer->forward(ctx, x);
            x = ggml_silu_inplace(ctx->ggml_ctx, x);
            x = out_layer->forward(ctx, x);
            return x;
        }
    };

    struct TimestepEmbedder : public GGMLBlock {
        // Embeds scalar timesteps into vector representations.
    protected:
        int time_embed_dim;

    public:
        TimestepEmbedder(int64_t hidden_size,
                         int time_embed_dim = 256,
                         int64_t out_channels         = 0)
            : time_embed_dim(time_embed_dim) {
            if (out_channels <= 0) {
                out_channels = hidden_size;
            }
            blocks["linear_1"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, time_embed_dim, true, true));
            blocks["linear_2"] = std::shared_ptr<GGMLBlock>(new Linear(time_embed_dim, out_channels, true, true));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* t_freq) {
            // t: [N, ]
            // return: [N, hidden_size]
            auto linear_1 = std::dynamic_pointer_cast<Linear>(blocks["linear_1"]);
            auto linear_2 = std::dynamic_pointer_cast<Linear>(blocks["linear_2"]);

            auto t_emb = linear_1->forward(ctx, t_freq);
            t_emb      = ggml_silu_inplace(ctx->ggml_ctx, t_emb);
            t_emb      = linear_2->forward(ctx, t_emb);
            return t_emb;
        }
    };
    class TimeEmbed : public UnaryBlock {
        int64_t embedding_dim;
    public:
        TimeEmbed(int64_t embedding_dim): embedding_dim(embedding_dim) {
            blocks["timestep_embedder"] = std::make_shared<TimestepEmbedder>(embedding_dim, embedding_dim * 4, embedding_dim);
            blocks["norm"]              = std::make_shared<RMSNorm>(embedding_dim);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* timestep) override {
            // x: [N, in_dim]
            // return: [N, out_dim]
            auto timestep_embedder = std::dynamic_pointer_cast<TimestepEmbedder>(blocks["timestep_embedder"]);
            auto norm              = std::dynamic_pointer_cast<RMSNorm>(blocks["norm"]);


            auto timesteps_proj = ggml_ext_timestep_embedding(ctx->ggml_ctx, timestep, embedding_dim, 10000, 1000.f); 
            auto timesteps_emb = timestep_embedder->forward(ctx, timesteps_proj);

            return norm->forward(ctx, timesteps_emb);
        }
    };

    /***
     *     Packed SwiGLU feed-forward experts for MoE: ``gate, up = (x @ gate_up_proj).chunk(2); out = (silu(gate) * up) @
    down_proj``.

    Gate and up projections are fused into a single weight ``gate_up_proj`` so that only two grouped matmuls are needed
    at runtime (gate+up combined, then down).

    Weights are stored pre-transposed relative to the standard linear-layer convention so that matmuls can be issued
    without a transpose at runtime.

    Weight shapes:
        gate_up_proj: (num_experts, hidden_size, 2 * moe_intermediate_dim) -- fused gate + up projection down_proj:
        (num_experts, moe_intermediate_dim, hidden_size) -- down projection
     */
    class SwiGLUExperts : public GGMLBlock {
    protected:
        int64_t hidden_size;
        int64_t moe_intermediate_dim;
        int num_experts;
        bool use_grouped_mm;

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, std::string prefix = "") override {
            enum ggml_type wtype = get_type(prefix + "gate_up_proj", tensor_storage_map, GGML_TYPE_F32);
            if (hidden_size % ggml_blck_size(wtype) != 0) {
                wtype = GGML_TYPE_F32;
            }
            params["gate_up_proj"] = ggml_new_tensor_3d(ctx, wtype, moe_intermediate_dim * 2, hidden_size, num_experts);
            wtype                  = get_type(prefix + "down_proj", tensor_storage_map, GGML_TYPE_F32);
            if (moe_intermediate_dim % ggml_blck_size(wtype) != 0) {
                wtype = GGML_TYPE_F32;
            }
            params["down_proj"] = ggml_new_tensor_3d(ctx, wtype, hidden_size, moe_intermediate_dim, num_experts);
        }

    public:
        SwiGLUExperts(int64_t hidden_size, int64_t moe_intermediate_dim, int num_experts, bool use_grouped_mm) : hidden_size(hidden_size), moe_intermediate_dim(moe_intermediate_dim), num_experts(num_experts), use_grouped_mm(use_grouped_mm) {}

        ggml_tensor* run_experts_for_loop(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* num_tokens_per_expert) {
            ggml_tensor* gate_up_proj = params["gate_up_proj"];
            ggml_tensor* down_proj    = params["down_proj"];

            GGML_ASSERT(false && "TODO: implement for-loop version of SwiGLUExperts");
            return nullptr;
        }

        ggml_tensor* run_experts_grouped_mm(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* num_tokens_per_expert) {
            ggml_tensor* gate_up_proj = params["gate_up_proj"];
            ggml_tensor* down_proj    = params["down_proj"];

            GGML_ASSERT(false && "TODO: implement grouped mm for SWIGLUExperts");
            return nullptr;
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* num_tokens_per_expert) {
            if (use_grouped_mm) {
                return run_experts_grouped_mm(ctx, x, num_tokens_per_expert);
            } else {
                return run_experts_for_loop(ctx, x, num_tokens_per_expert);
            }
        }
    };

    class FeedForward : public GGMLBlock {
    public:
        FeedForward(int64_t dim,
                    int64_t dim_out,
                    int64_t inner_dim) {
            blocks["net.0.proj"] = std::make_shared<Linear>(dim, inner_dim * 2, false);
            blocks["net.2"]      = std::make_shared<Linear>(inner_dim, dim_out, false);
        }

        ggml_tensor* swiGLU(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto act_proj = std::dynamic_pointer_cast<Linear>(blocks["net.0.proj"]);

            x = act_proj->forward(ctx, x);

            auto x_vec = ggml_ext_chunk(ctx->ggml_ctx, x, 2, 0, false);
            x          = x_vec[0];
            auto gate  = x_vec[1];
            gate       = ggml_cont(ctx->ggml_ctx, gate);
            gate       = ggml_silu_inplace(ctx->ggml_ctx, gate);

            return ggml_mul(ctx->ggml_ctx, x, gate);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto out = std::dynamic_pointer_cast<Linear>(blocks["net.2"]);

            x = swiGLU(ctx, x);
            x = out->forward(ctx, x);

            return x;
        }
    };

    class NucleusMoELayer : public GGMLBlock {
    protected:
        int64_t hidden_size;
        int64_t moe_intermediate_dim;
        int num_experts;
        float capacity_factor;
        bool use_sigmoid;
        float route_scale;

    public:
        NucleusMoELayer(int64_t hidden_size,
                        int64_t moe_intermediate_dim,
                        int num_experts,
                        float capacity_factor,
                        bool use_sigmoid,
                        float route_scale,
                        bool use_grouped_mm)
            : hidden_size(hidden_size), moe_intermediate_dim(moe_intermediate_dim), num_experts(num_experts), capacity_factor(capacity_factor), use_sigmoid(use_sigmoid), route_scale(route_scale) {
            blocks["shared_expert"] = std::make_shared<FeedForward>(hidden_size, hidden_size, moe_intermediate_dim);
            blocks["gate"]          = std::make_shared<Linear>(hidden_size * 2, num_experts, false);
            blocks["experts"]       = std::make_shared<SwiGLUExperts>(hidden_size, moe_intermediate_dim, num_experts, use_grouped_mm);
#define NUCLEUS_DISABLE_EXPERTS 1
#if NUCLEUS_DISABLE_EXPERTS
            LOG_WARN("NucleusMoELayer: Experts are disabled for now, running shared expert only\n Poor performance is expected.");
#endif
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* x_unmodulated, ggml_tensor* timestep) {
            // x: [N, slen, hidden]
            auto shared_expert = std::dynamic_pointer_cast<FeedForward>(blocks["shared_expert"]);
            auto gate          = std::dynamic_pointer_cast<Linear>(blocks["gate"]);
            auto experts       = std::dynamic_pointer_cast<SwiGLUExperts>(blocks["experts"]);

#if NUCLEUS_DISABLE_EXPERTS
            return shared_expert->forward(ctx, x);
#else
            GGML_ASSERT(false && "NucleusMoELayer: MoE routing is not implemented yet");
            return x;
#endif
        }
    };

    class Attention : public GGMLBlock {
        int64_t dim;
        int64_t heads;
        int64_t kv_heads;
        int64_t head_dim;
        

        bool rmsnorm;

        int64_t query_dim;
        int64_t added_kv_proj_dim;
        int64_t out_dim;
        
        int64_t inner_dim;
        int64_t inner_kv_dim;

    public:
        Attention(int64_t dim,
                  int64_t heads,
                  int64_t kv_heads,
                  int64_t head_dim,
                  std::string qk_norm = "rms") : dim(dim), heads(heads), kv_heads(kv_heads), head_dim(head_dim) {
            query_dim         = dim;
            added_kv_proj_dim = dim;
            out_dim           = dim;

            inner_dim    = heads * head_dim;
            inner_kv_dim = kv_heads * head_dim;
            
            blocks["to_q"]       = std::make_shared<Linear>(query_dim, inner_dim, false);
            blocks["to_k"]       = std::make_shared<Linear>(query_dim, inner_kv_dim, false);
            blocks["to_v"]       = std::make_shared<Linear>(query_dim, inner_kv_dim, false);

            blocks["add_k_proj"] = std::make_shared<Linear>(added_kv_proj_dim, inner_kv_dim, false);
            blocks["add_v_proj"] = std::make_shared<Linear>(added_kv_proj_dim, inner_kv_dim, false);

            // only rms norm supported
            if (qk_norm == "rms") {
                rmsnorm          = true;
                blocks["norm_q"] = std::make_shared<RMSNorm>(head_dim);
                blocks["norm_k"] = std::make_shared<RMSNorm>(head_dim);

                blocks["norm_added_q"] = std::make_shared<RMSNorm>(head_dim); // unused?
                blocks["norm_added_k"] = std::make_shared<RMSNorm>(head_dim);
            }

            blocks["to_out.0"] = std::make_shared<Linear>(inner_dim, out_dim, false);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* y = nullptr, ggml_tensor* x_pe = nullptr, ggml_tensor* y_pe = nullptr) {
            int64_t kv_group = heads / kv_heads;

            auto to_q = std::dynamic_pointer_cast<Linear>(blocks["to_q"]);
            auto to_k = std::dynamic_pointer_cast<Linear>(blocks["to_k"]);
            auto to_v = std::dynamic_pointer_cast<Linear>(blocks["to_v"]);

            auto q = to_q->forward(ctx, x);
            auto k = to_k->forward(ctx, x);
            auto v = to_v->forward(ctx, x);

            if(rmsnorm) {
                auto norm_q = std::dynamic_pointer_cast<RMSNorm>(blocks["norm_q"]);
                auto norm_k = std::dynamic_pointer_cast<RMSNorm>(blocks["norm_k"]);

                q = norm_q->forward(ctx, q);
                k = norm_k->forward(ctx, k);
            }

            if (x_pe) {
                ggml_tensor* orig_q = ggml_dup_tensor(ctx->ggml_ctx, q);
                ggml_tensor* orig_k = ggml_dup_tensor(ctx->ggml_ctx, k);

                q = ggml_reshape_4d(ctx->ggml_ctx, q, head_dim, heads, x->ne[1], x->ne[2]);
                q = Rope::apply_rope(ctx->ggml_ctx, q, x_pe);
                k = ggml_reshape_4d(ctx->ggml_ctx, k, head_dim, kv_heads, x->ne[1], x->ne[2]);
                k = Rope::apply_rope(ctx->ggml_ctx, k, x_pe);

                q = ggml_reshape(ctx->ggml_ctx, q, orig_q);
                k = ggml_reshape(ctx->ggml_ctx, k, orig_k);
            }

            if (y) {
                auto add_k_proj = std::dynamic_pointer_cast<Linear>(blocks["add_k_proj"]);
                auto add_v_proj = std::dynamic_pointer_cast<Linear>(blocks["add_v_proj"]);
                auto k_added = add_k_proj->forward(ctx, y);
                auto v_added = add_v_proj->forward(ctx, y);

                if(rmsnorm) {
                    auto norm_added_k = std::dynamic_pointer_cast<RMSNorm>(blocks["norm_added_k"]);
                    
                    k_added = norm_added_k->forward(ctx, k_added);
                }

                if (y_pe) {
                    ggml_tensor* orig_k = ggml_dup_tensor(ctx->ggml_ctx, k_added);
                    k_added = ggml_reshape_4d(ctx->ggml_ctx, k_added, head_dim, kv_heads, y->ne[1], y->ne[2]);
                    k_added = Rope::apply_rope(ctx->ggml_ctx, k_added, y_pe);
                    k_added = ggml_reshape(ctx->ggml_ctx, k_added, orig_k);
                }
                
                k = ggml_concat(ctx->ggml_ctx, k, k_added, 1);
                v = ggml_concat(ctx->ggml_ctx, v, v_added, 1);
            }

            if (kv_group > 1) {
                int64_t ne0 = k->ne[0];
                int64_t ne1 = k->ne[1];
                int64_t ne2 = k->ne[2];
                int64_t ne3 = k->ne[3];

                struct ggml_tensor* target_shape = ggml_new_tensor_4d(ctx->ggml_ctx,
                                                                      GGML_TYPE_F32,
                                                                      ne0,
                                                                      kv_group,
                                                                      ne1,
                                                                      ne3 * ne2);

                // repeat_interleave
                k = ggml_reshape_4d(ctx->ggml_ctx,
                                    ggml_repeat(ctx->ggml_ctx,
                                                ggml_view_4d(ctx->ggml_ctx, k, ne0, 1, ne1, ne3 * ne2, k->nb[1], k->nb[1], k->nb[2], 0),
                                                target_shape),
                                    ne0,
                                    ne1 * kv_group,
                                    ne2,
                                    ne3);

                v = ggml_reshape_4d(ctx->ggml_ctx,
                                    ggml_repeat(ctx->ggml_ctx,
                                                ggml_view_4d(ctx->ggml_ctx, v, ne0, 1, ne1, ne3 * ne2, v->nb[1], v->nb[1], v->nb[2], 0),
                                                target_shape),
                                    ne0,
                                    ne1 * kv_group,
                                    ne2,
                                    ne3);
            }
            x = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v, heads);

            auto to_out = std::dynamic_pointer_cast<Linear>(blocks["to_out.0"]);
            x = to_out->forward(ctx, x);


            return x;
        }
    };

    class NucleusMoETransformerBlock : public GGMLBlock {
    protected:
        int64_t dim;
        bool moe_enabled;

    public:
        NucleusMoETransformerBlock(int dim                  = 4096,
                                   int num_attention_heads  = 16,
                                   int attention_head_dim   = 128,
                                   int num_key_value_heads  = 4,
                                   int joint_attention_dim  = 4096,
                                   std::string qk_norm      = "rms",
                                   float eps                = 1e-6f,
                                   float mlp_ratio          = 4.0f,
                                   bool moe_enabled         = false,
                                   int num_experts          = 64,
                                   int moe_intermediate_dim = 1344,
                                   float capacity_factor    = 4.0f,
                                   bool use_sigmoid         = false,
                                   float route_scale        = 2.5f,
                                   bool use_grouped_mm      = true):moe_enabled(moe_enabled) {
            blocks["img_mod.1"]    = std::make_shared<Linear>(dim, 4 * dim);
            blocks["encoder_proj"] = std::make_shared<Linear>(joint_attention_dim, dim);
            blocks["attn"]         = std::make_shared<Attention>(dim, num_attention_heads, num_key_value_heads, attention_head_dim, qk_norm);
            if (moe_enabled) {
                blocks["img_mlp"] = std::make_shared<NucleusMoELayer>(dim, moe_intermediate_dim, num_experts, capacity_factor, use_sigmoid, route_scale, use_grouped_mm);
            } else {
                int64_t mlp_inner_dim = 128 * ((int)((float)dim * mlp_ratio * 2.0f / 3.0f) / 128);
                blocks["img_mlp"]     = std::make_shared<FeedForward>(dim, dim, mlp_inner_dim);
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* y,
                             ggml_tensor* temb,
                             ggml_tensor* img_pe,
                             ggml_tensor* txt_pe) {
            // x: hidden_states: [N, n_token, dim]
            // y: encoder_hidden_states: [N, n_token, joint_attention_dim]
            // pe: image_rotary_emb: tuple of [N, n_token, attention_head_dim]
            // temb: [N, joint_attention_dim]

            auto img_mod_1    = std::dynamic_pointer_cast<Linear>(blocks["img_mod.1"]);
            auto encoder_proj = std::dynamic_pointer_cast<Linear>(blocks["encoder_proj"]);
            auto attn         = std::dynamic_pointer_cast<Attention>(blocks["attn"]);
            auto img_mlp      = std::dynamic_pointer_cast<GGMLBlock>(blocks["img_mlp"]);

            auto img_mod_params = img_mod_1->forward(ctx, ggml_silu(ctx->ggml_ctx, temb));  // [N, 4 * dim]
            auto img_mod_vec    = ggml_ext_chunk(ctx->ggml_ctx, img_mod_params, 4, 0);
            auto scale1         = img_mod_vec[0];
            auto gate1          = img_mod_vec[1];
            auto scale2         = img_mod_vec[2];
            auto gate2          = img_mod_vec[3];

            gate1 = ggml_clamp(ctx->ggml_ctx, gate1, -2.0f, 2.0f);
            gate2 = ggml_clamp(ctx->ggml_ctx, gate2, -2.0f, 2.0f);

            ggml_tensor* context = nullptr;
            if (y != nullptr) {
                context = encoder_proj->forward(ctx, y);  // [N, n_token, dim]
            }

            auto img_normed    = ggml_rms_norm(ctx->ggml_ctx, x, 1e-12f);                                           // [N, n_token, dim]
            auto img_modulated = ggml_add(ctx->ggml_ctx, img_normed, ggml_mul(ctx->ggml_ctx, img_normed, scale1));  // [N, n_token, dim]

            auto attn_output = attn->forward(ctx, img_modulated, context, img_pe, txt_pe);  // [N, n_token, dim]

            x = ggml_add(ctx->ggml_ctx, x, ggml_mul(ctx->ggml_ctx, attn_output, ggml_tanh(ctx->ggml_ctx, gate1)));  // [N, n_token, dim]

            auto img_normed2 = ggml_norm(ctx->ggml_ctx, x, 1e-6f);  // [N, n_token, dim]

            auto img_modulated2 = ggml_add(ctx->ggml_ctx, img_normed2, ggml_mul(ctx->ggml_ctx, img_normed2, scale2));  // [N, n_token, dim]

            ggml_tensor* img_mlp_output = nullptr;
            if (moe_enabled) {
                img_mlp_output = std::dynamic_pointer_cast<NucleusMoELayer>(img_mlp)->forward(ctx, img_modulated2, img_normed2, temb);  // [N, n_token, dim]
            } else {
                img_mlp_output = std::dynamic_pointer_cast<FeedForward>(img_mlp)->forward(ctx, img_modulated2);  // [N, n_token, dim]
            }

            x = ggml_add(ctx->ggml_ctx, x, ggml_mul(ctx->ggml_ctx, img_mlp_output, ggml_tanh(ctx->ggml_ctx, gate2)));  // [N, n_token, dim]
            return x;
        }
    };

    struct NucleusImageParams {
        int patch_size                      = 2;
        int64_t in_channels                 = 64;
        int64_t out_channels                = 16;
        int64_t num_layers                  = 32;
        int64_t attention_head_dim          = 128;
        int64_t num_attention_heads         = 16;
        int64_t num_key_value_heads         = 4;
        int64_t joint_attention_dim         = 4096;
        std::vector<int> axes_dims          = {16, 56, 56};
        int axes_dim_sum                    = 128;
        float mlp_ratio                     = 4.0f;
        bool moe_enabled                    = true;
        int num_first_dense_layers          = 3;
        int num_experts                     = 64;
        int64_t moe_intermediate_dim        = 1344;
        std::vector<float> capacity_factors = {0.0f, 0.0f, 0.0f, 4.0f, 4.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
        bool use_sigmoid                    = false;
        float route_scale                   = 2.5f;
        bool use_grouped_mm                 = true;
    };

    class NucleusImageModel : public GGMLBlock {
    protected:
        NucleusImageParams nucleusParams;
        int64_t inner_dim;


    public:
        NucleusImageModel() = default;
        NucleusImageModel(NucleusImageParams nucleusParams) : nucleusParams(nucleusParams) {
            inner_dim = nucleusParams.num_attention_heads * nucleusParams.attention_head_dim;

            blocks["txt_norm"] = std::make_shared<RMSNorm>(nucleusParams.joint_attention_dim);

            blocks["time_text_embed"] = std::make_shared<TimeEmbed>(inner_dim);
            blocks["img_in"]          = std::make_shared<Linear>(nucleusParams.in_channels, inner_dim, true);
            for (int i = 0; i < nucleusParams.num_layers; i++) {
                bool is_dense_layer   = i < nucleusParams.num_first_dense_layers;
                float capacity_factor = nucleusParams.moe_enabled ? nucleusParams.capacity_factors[i] : 0.0f;
                auto block = std::make_shared<NucleusMoETransformerBlock>(inner_dim,
                                                                          nucleusParams.num_attention_heads,
                                                                          nucleusParams.attention_head_dim,
                                                                          nucleusParams.num_key_value_heads,
                                                                          nucleusParams.joint_attention_dim,
                                                                          "rms",
                                                                          1e-6f,
                                                                          nucleusParams.mlp_ratio,
                                                                          nucleusParams.moe_enabled && !is_dense_layer,
                                                                          nucleusParams.num_experts,
                                                                          nucleusParams.moe_intermediate_dim,
                                                                          capacity_factor,
                                                                          nucleusParams.use_sigmoid,
                                                                          nucleusParams.route_scale,
                                                                          nucleusParams.use_grouped_mm);

                blocks["transformer_blocks." + std::to_string(i)] = block;
            }
            // AdaLayerNormContinuous
            blocks["norm_out.linear"] = std::make_shared<Linear>(inner_dim, inner_dim * 2, true);
            blocks["proj_out"]        = std::make_shared<Linear>(inner_dim, nucleusParams.out_channels * nucleusParams.patch_size * nucleusParams.patch_size, false);
        }

        ggml_tensor* adalnc(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* temb) {
            auto norm_out_linear = std::dynamic_pointer_cast<Linear>(blocks["norm_out.linear"]);
            auto emb             = norm_out_linear->forward(ctx, ggml_silu(ctx->ggml_ctx, temb));  // [N, inner_dim * 2]
            auto emb_vec         = ggml_ext_chunk(ctx->ggml_ctx, emb, 2, 0);
            auto scale           = emb_vec[0];
            auto shift           = emb_vec[1];
            x = ggml_norm(ctx->ggml_ctx, x, 1e-6f);
            x = ggml_add(ctx->ggml_ctx, x, ggml_mul(ctx->ggml_ctx, x, scale));  // [N, joint_attention_dim, h * w]
            x = ggml_add(ctx->ggml_ctx, x, shift);                                      // [N, joint_attention_dim, h * w]
            return x;
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* txt, ggml_tensor* timestep, ggml_tensor * pe) {
            int64_t W = x->ne[0];
            int64_t H = x->ne[1];
            int64_t C = x->ne[2];
            int64_t N = x->ne[3];

            // x: [N, in_channels, h, w]
            auto img_in = std::dynamic_pointer_cast<Linear>(blocks["img_in"]);

            auto img             = DiT::pad_and_patchify(ctx, x, nucleusParams.patch_size, nucleusParams.patch_size, false);
            img           = img_in->forward(ctx, img);  // [N, joint_attention_dim, h * w]

            auto txt_norm = std::dynamic_pointer_cast<RMSNorm>(blocks["txt_norm"]);
            txt = txt_norm->forward(ctx, txt);  // [N, max_position, joint_attention_dim]

            auto time_text_embed = std::dynamic_pointer_cast<TimeEmbed>(blocks["time_text_embed"]);
            auto temb           = time_text_embed->forward(ctx, timestep);  // [N, joint_attention_dim]

            // TODO: pe
            int img_len = H * W / (nucleusParams.patch_size * nucleusParams.patch_size);
            int txt_len = pe->ne[3] - img_len;
            auto img_pe = ggml_view_4d(ctx->ggml_ctx, pe, pe->ne[0], pe->ne[1], pe->ne[2], img_len, pe->nb[1], pe->nb[2], pe->nb[3], 0);
            auto txt_pe = ggml_view_4d(ctx->ggml_ctx, pe, pe->ne[0], pe->ne[1], pe->ne[2], txt_len, pe->nb[1], pe->nb[2], pe->nb[3], img_len * pe->nb[3]);


            for (int i = 0; i < nucleusParams.num_layers; i++) {
                auto block = std::dynamic_pointer_cast<NucleusMoETransformerBlock>(blocks["transformer_blocks." + std::to_string(i)]);
                img = block->forward(ctx, img, txt, temb, img_pe, txt_pe);  // [N, joint_attention_dim, h * w]
            }

            img = adalnc(ctx, img, temb);                                      // [N, joint_attention_dim, h * w]

            auto proj_out = std::dynamic_pointer_cast<Linear>(blocks["proj_out"]);
            img = proj_out->forward(ctx, img);  // [N, out_channels * patch_size * patch_size, h * w]

            img = DiT::unpatchify_and_crop(ctx->ggml_ctx, img, H, W, nucleusParams.patch_size, nucleusParams.patch_size, false);
            return img;
        }
    };

    struct NucleusImageRunner : public GGMLRunner {
        NucleusImageModel model;
        NucleusImageParams nucleus_params;
        std::vector<float> pe_vec;


        NucleusImageRunner(ggml_backend_t backend,
                           bool offload_params_to_cpu,
                           const String2TensorStorage& tensor_storage_map = {},
                           const std::string prefix                       = "",
                           SDVersion version                              = VERSION_NUCLEUS_IMAGE)
            : GGMLRunner(backend, offload_params_to_cpu) {
            model = NucleusImageModel(nucleus_params);
            model.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return "nucleus_image";
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string prefix) {
            model.get_param_tensors(tensors, prefix);
        }

        ggml_cgraph* build_graph(const sd::Tensor<float>& x_tensor,
                                 const sd::Tensor<float>& timesteps_tensor,
                                 const sd::Tensor<float>& context_tensor,
                                 const std::vector<sd::Tensor<float>>& ref_latents_tensor = {},
                                 bool increase_ref_index                                  = false) {
            ggml_cgraph* gf        = new_graph_custom(204800);
            ggml_tensor* x         = make_input(x_tensor);
            ggml_set_name(x,"x");
            ggml_tensor* timesteps = make_input(timesteps_tensor);
            ggml_set_name(timesteps,"timesteps");
            GGML_ASSERT(x->ne[3] == 1);
            GGML_ASSERT(!context_tensor.empty());
            ggml_tensor* context = make_input(context_tensor);
            ggml_set_name(context,"context");

            std::set<int> txt_arange_dims;

            pe_vec      = Rope::gen_nucleus_image_pe(static_cast<int>(x->ne[1]),
                                                  static_cast<int>(x->ne[0]),
                                                  nucleus_params.patch_size,
                                                  static_cast<int>(x->ne[3]),
                                                  static_cast<int>(context->ne[1]),
                                                  10000,
                                                  true, 
                                                  circular_y_enabled,
                                                  circular_x_enabled,
                                                  nucleus_params.axes_dims);
                                                  
            int pos_len = static_cast<int>(pe_vec.size() / nucleus_params.axes_dim_sum / 2);
            // LOG_DEBUG("pos_len %d", pos_len);
            auto pe = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, nucleus_params.axes_dim_sum / 2, pos_len);
            // pe->data = pe_vec.data();
            // print_ggml_tensor(pe,true);
            // pe->data = nullptr;
            set_backend_tensor_data(pe, pe_vec.data());
            ggml_set_name(pe,"pe");

            auto runner_ctx  = get_context();
            ggml_tensor* out = model.forward(&runner_ctx,
                                             x,
                                             context,
                                             timesteps,
                                             pe);

            ggml_build_forward_expand(gf, out);
            return gf;
        }

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
    };

}  // namespace NucleusImage

#endif