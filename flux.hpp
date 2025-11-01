#ifndef __FLUX_HPP__
#define __FLUX_HPP__

#include <memory>
#include <vector>

#include "ggml_extend.hpp"
#include "model.h"
#include "rope.hpp"

#define FLUX_GRAPH_SIZE 10240

namespace Flux {

    struct MLPEmbedder : public UnaryBlock {
    public:
        MLPEmbedder(int64_t in_dim, int64_t hidden_dim) {
            blocks["in_layer"]  = std::shared_ptr<GGMLBlock>(new Linear(in_dim, hidden_dim, true));
            blocks["out_layer"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_dim, hidden_dim, true));
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) override {
            // x: [..., in_dim]
            // return: [..., hidden_dim]
            auto in_layer  = std::dynamic_pointer_cast<Linear>(blocks["in_layer"]);
            auto out_layer = std::dynamic_pointer_cast<Linear>(blocks["out_layer"]);

            x = in_layer->forward(ctx, x);
            x = ggml_silu_inplace(ctx->ggml_ctx, x);
            x = out_layer->forward(ctx, x);
            return x;
        }
    };

    class RMSNorm : public UnaryBlock {
    protected:
        int64_t hidden_size;
        float eps;

        void init_params(struct ggml_context* ctx, const String2GGMLType& tensor_types = {}, const std::string prefix = "") override {
            ggml_type wtype = GGML_TYPE_F32;
            params["scale"] = ggml_new_tensor_1d(ctx, wtype, hidden_size);
        }

    public:
        RMSNorm(int64_t hidden_size,
                float eps = 1e-06f)
            : hidden_size(hidden_size),
              eps(eps) {}

        struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) override {
            struct ggml_tensor* w = params["scale"];
            x                     = ggml_rms_norm(ctx->ggml_ctx, x, eps);
            x                     = ggml_mul(ctx->ggml_ctx, x, w);
            return x;
        }
    };

    struct QKNorm : public GGMLBlock {
    public:
        QKNorm(int64_t dim) {
            blocks["query_norm"] = std::shared_ptr<GGMLBlock>(new RMSNorm(dim));
            blocks["key_norm"]   = std::shared_ptr<GGMLBlock>(new RMSNorm(dim));
        }

        struct ggml_tensor* query_norm(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
            // x: [..., dim]
            // return: [..., dim]
            auto norm = std::dynamic_pointer_cast<RMSNorm>(blocks["query_norm"]);

            x = norm->forward(ctx, x);
            return x;
        }

        struct ggml_tensor* key_norm(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
            // x: [..., dim]
            // return: [..., dim]
            auto norm = std::dynamic_pointer_cast<RMSNorm>(blocks["key_norm"]);

            x = norm->forward(ctx, x);
            return x;
        }
    };

    struct SelfAttention : public GGMLBlock {
    public:
        int64_t num_heads;

    public:
        SelfAttention(int64_t dim,
                      int64_t num_heads = 8,
                      bool qkv_bias     = false)
            : num_heads(num_heads) {
            int64_t head_dim = dim / num_heads;
            blocks["qkv"]    = std::shared_ptr<GGMLBlock>(new Linear(dim, dim * 3, qkv_bias));
            blocks["norm"]   = std::shared_ptr<GGMLBlock>(new QKNorm(head_dim));
            blocks["proj"]   = std::shared_ptr<GGMLBlock>(new Linear(dim, dim));
        }

        std::vector<struct ggml_tensor*> pre_attention(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
            auto qkv_proj = std::dynamic_pointer_cast<Linear>(blocks["qkv"]);
            auto norm     = std::dynamic_pointer_cast<QKNorm>(blocks["norm"]);

            auto qkv         = qkv_proj->forward(ctx, x);
            auto qkv_vec     = split_qkv(ctx->ggml_ctx, qkv);
            int64_t head_dim = qkv_vec[0]->ne[0] / num_heads;
            auto q           = ggml_reshape_4d(ctx->ggml_ctx, qkv_vec[0], head_dim, num_heads, qkv_vec[0]->ne[1], qkv_vec[0]->ne[2]);
            auto k           = ggml_reshape_4d(ctx->ggml_ctx, qkv_vec[1], head_dim, num_heads, qkv_vec[1]->ne[1], qkv_vec[1]->ne[2]);
            auto v           = ggml_reshape_4d(ctx->ggml_ctx, qkv_vec[2], head_dim, num_heads, qkv_vec[2]->ne[1], qkv_vec[2]->ne[2]);
            q                = norm->query_norm(ctx, q);
            k                = norm->key_norm(ctx, k);
            return {q, k, v};
        }

        struct ggml_tensor* post_attention(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
            auto proj = std::dynamic_pointer_cast<Linear>(blocks["proj"]);

            x = proj->forward(ctx, x);  // [N, n_token, dim]
            return x;
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* pe,
                                    struct ggml_tensor* mask) {
            // x: [N, n_token, dim]
            // pe: [n_token, d_head/2, 2, 2]
            // return [N, n_token, dim]
            auto qkv = pre_attention(ctx, x);                                   // q,k,v: [N, n_token, n_head, d_head]
            x        = Rope::attention(ctx, qkv[0], qkv[1], qkv[2], pe, mask);  // [N, n_token, dim]
            x        = post_attention(ctx, x);                                  // [N, n_token, dim]
            return x;
        }
    };

    struct ModulationOut {
        ggml_tensor* shift = nullptr;
        ggml_tensor* scale = nullptr;
        ggml_tensor* gate  = nullptr;

        ModulationOut(ggml_tensor* shift = nullptr, ggml_tensor* scale = nullptr, ggml_tensor* gate = nullptr)
            : shift(shift), scale(scale), gate(gate) {}

        ModulationOut(GGMLRunnerContext* ctx, ggml_tensor* vec, int64_t offset) {
            int64_t stride = vec->nb[1] * vec->ne[1];
            shift          = ggml_view_2d(ctx->ggml_ctx, vec, vec->ne[0], vec->ne[1], vec->nb[1], stride * (offset + 0));  // [N, dim]
            scale          = ggml_view_2d(ctx->ggml_ctx, vec, vec->ne[0], vec->ne[1], vec->nb[1], stride * (offset + 1));  // [N, dim]
            gate           = ggml_view_2d(ctx->ggml_ctx, vec, vec->ne[0], vec->ne[1], vec->nb[1], stride * (offset + 2));  // [N, dim]
        }
    };

    struct Modulation : public GGMLBlock {
    public:
        bool is_double;
        int multiplier;

    public:
        Modulation(int64_t dim, bool is_double)
            : is_double(is_double) {
            multiplier    = is_double ? 6 : 3;
            blocks["lin"] = std::shared_ptr<GGMLBlock>(new Linear(dim, dim * multiplier));
        }

        std::vector<ModulationOut> forward(GGMLRunnerContext* ctx, struct ggml_tensor* vec) {
            // x: [N, dim]
            // return: [ModulationOut, ModulationOut]
            auto lin = std::dynamic_pointer_cast<Linear>(blocks["lin"]);

            auto out = ggml_silu(ctx->ggml_ctx, vec);
            out      = lin->forward(ctx, out);  // [N, multiplier*dim]

            auto m = ggml_reshape_3d(ctx->ggml_ctx, out, vec->ne[0], multiplier, vec->ne[1]);  // [N, multiplier, dim]
            m      = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, m, 0, 2, 1, 3));     // [multiplier, N, dim]

            ModulationOut m_0 = ModulationOut(ctx, m, 0);
            if (is_double) {
                return {m_0, ModulationOut(ctx, m, 3)};
            }

            return {m_0, ModulationOut()};
        }
    };

    __STATIC_INLINE__ struct ggml_tensor* modulate(struct ggml_context* ctx,
                                                   struct ggml_tensor* x,
                                                   struct ggml_tensor* shift,
                                                   struct ggml_tensor* scale) {
        // x: [N, L, C]
        // scale: [N, C]
        // shift: [N, C]
        scale = ggml_reshape_3d(ctx, scale, scale->ne[0], 1, scale->ne[1]);  // [N, 1, C]
        shift = ggml_reshape_3d(ctx, shift, shift->ne[0], 1, shift->ne[1]);  // [N, 1, C]
        x     = ggml_add(ctx, x, ggml_mul(ctx, x, scale));
        x     = ggml_add(ctx, x, shift);
        return x;
    }

    struct DoubleStreamBlock : public GGMLBlock {
        bool prune_mod;
        int idx = 0;

    public:
        DoubleStreamBlock(int64_t hidden_size,
                          int64_t num_heads,
                          float mlp_ratio,
                          int idx        = 0,
                          bool qkv_bias  = false,
                          bool prune_mod = false)
            : idx(idx), prune_mod(prune_mod) {
            int64_t mlp_hidden_dim = hidden_size * mlp_ratio;
            if (!prune_mod) {
                blocks["img_mod"] = std::shared_ptr<GGMLBlock>(new Modulation(hidden_size, true));
            }
            blocks["img_norm1"] = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size, 1e-6f, false));
            blocks["img_attn"]  = std::shared_ptr<GGMLBlock>(new SelfAttention(hidden_size, num_heads, qkv_bias));

            blocks["img_norm2"] = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size, 1e-6f, false));
            blocks["img_mlp.0"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, mlp_hidden_dim));
            // img_mlp.1 is nn.GELU(approximate="tanh")
            blocks["img_mlp.2"] = std::shared_ptr<GGMLBlock>(new Linear(mlp_hidden_dim, hidden_size));

            if (!prune_mod) {
                blocks["txt_mod"] = std::shared_ptr<GGMLBlock>(new Modulation(hidden_size, true));
            }
            blocks["txt_norm1"] = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size, 1e-6f, false));
            blocks["txt_attn"]  = std::shared_ptr<GGMLBlock>(new SelfAttention(hidden_size, num_heads, qkv_bias));

            blocks["txt_norm2"] = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size, 1e-6f, false));
            blocks["txt_mlp.0"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, mlp_hidden_dim));
            // img_mlp.1 is nn.GELU(approximate="tanh")
            blocks["txt_mlp.2"] = std::shared_ptr<GGMLBlock>(new Linear(mlp_hidden_dim, hidden_size));
        }

        std::vector<ModulationOut> get_distil_img_mod(GGMLRunnerContext* ctx, struct ggml_tensor* vec) {
            // TODO: not hardcoded?
            const int single_blocks_count = 38;
            const int double_blocks_count = 19;

            int64_t offset = 6 * idx + 3 * single_blocks_count;
            return {ModulationOut(ctx, vec, offset), ModulationOut(ctx, vec, offset + 3)};
        }

        std::vector<ModulationOut> get_distil_txt_mod(GGMLRunnerContext* ctx, struct ggml_tensor* vec) {
            // TODO: not hardcoded?
            const int single_blocks_count = 38;
            const int double_blocks_count = 19;

            int64_t offset = 6 * idx + 6 * double_blocks_count + 3 * single_blocks_count;
            return {ModulationOut(ctx, vec, offset), ModulationOut(ctx, vec, offset + 3)};
        }

        std::pair<struct ggml_tensor*, struct ggml_tensor*> forward(GGMLRunnerContext* ctx,
                                                                    struct ggml_tensor* img,
                                                                    struct ggml_tensor* txt,
                                                                    struct ggml_tensor* vec,
                                                                    struct ggml_tensor* pe,
                                                                    struct ggml_tensor* mask = nullptr) {
            // img: [N, n_img_token, hidden_size]
            // txt: [N, n_txt_token, hidden_size]
            // pe: [n_img_token + n_txt_token, d_head/2, 2, 2]
            // return: ([N, n_img_token, hidden_size], [N, n_txt_token, hidden_size])
            auto img_norm1 = std::dynamic_pointer_cast<LayerNorm>(blocks["img_norm1"]);
            auto img_attn  = std::dynamic_pointer_cast<SelfAttention>(blocks["img_attn"]);

            auto img_norm2 = std::dynamic_pointer_cast<LayerNorm>(blocks["img_norm2"]);
            auto img_mlp_0 = std::dynamic_pointer_cast<Linear>(blocks["img_mlp.0"]);
            auto img_mlp_2 = std::dynamic_pointer_cast<Linear>(blocks["img_mlp.2"]);

            auto txt_norm1 = std::dynamic_pointer_cast<LayerNorm>(blocks["txt_norm1"]);
            auto txt_attn  = std::dynamic_pointer_cast<SelfAttention>(blocks["txt_attn"]);

            auto txt_norm2 = std::dynamic_pointer_cast<LayerNorm>(blocks["txt_norm2"]);
            auto txt_mlp_0 = std::dynamic_pointer_cast<Linear>(blocks["txt_mlp.0"]);
            auto txt_mlp_2 = std::dynamic_pointer_cast<Linear>(blocks["txt_mlp.2"]);

            std::vector<ModulationOut> img_mods;
            if (prune_mod) {
                img_mods = get_distil_img_mod(ctx, vec);
            } else {
                auto img_mod = std::dynamic_pointer_cast<Modulation>(blocks["img_mod"]);
                img_mods     = img_mod->forward(ctx, vec);
            }
            ModulationOut img_mod1 = img_mods[0];
            ModulationOut img_mod2 = img_mods[1];
            std::vector<ModulationOut> txt_mods;
            if (prune_mod) {
                txt_mods = get_distil_txt_mod(ctx, vec);
            } else {
                auto txt_mod = std::dynamic_pointer_cast<Modulation>(blocks["txt_mod"]);
                txt_mods     = txt_mod->forward(ctx, vec);
            }
            ModulationOut txt_mod1 = txt_mods[0];
            ModulationOut txt_mod2 = txt_mods[1];

            // prepare image for attention
            auto img_modulated = img_norm1->forward(ctx, img);
            img_modulated      = Flux::modulate(ctx->ggml_ctx, img_modulated, img_mod1.shift, img_mod1.scale);
            auto img_qkv       = img_attn->pre_attention(ctx, img_modulated);  // q,k,v: [N, n_img_token, n_head, d_head]
            auto img_q         = img_qkv[0];
            auto img_k         = img_qkv[1];
            auto img_v         = img_qkv[2];

            // prepare txt for attention
            auto txt_modulated = txt_norm1->forward(ctx, txt);
            txt_modulated      = Flux::modulate(ctx->ggml_ctx, txt_modulated, txt_mod1.shift, txt_mod1.scale);
            auto txt_qkv       = txt_attn->pre_attention(ctx, txt_modulated);  // q,k,v: [N, n_txt_token, n_head, d_head]
            auto txt_q         = txt_qkv[0];
            auto txt_k         = txt_qkv[1];
            auto txt_v         = txt_qkv[2];

            // run actual attention
            auto q = ggml_concat(ctx->ggml_ctx, txt_q, img_q, 2);  // [N, n_txt_token + n_img_token, n_head, d_head]
            auto k = ggml_concat(ctx->ggml_ctx, txt_k, img_k, 2);  // [N, n_txt_token + n_img_token, n_head, d_head]
            auto v = ggml_concat(ctx->ggml_ctx, txt_v, img_v, 2);  // [N, n_txt_token + n_img_token, n_head, d_head]

            auto attn         = Rope::attention(ctx, q, k, v, pe, mask);                                  // [N, n_txt_token + n_img_token, n_head*d_head]
            attn              = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, attn, 0, 2, 1, 3));  // [n_txt_token + n_img_token, N, hidden_size]
            auto txt_attn_out = ggml_view_3d(ctx->ggml_ctx,
                                             attn,
                                             attn->ne[0],
                                             attn->ne[1],
                                             txt->ne[1],
                                             attn->nb[1],
                                             attn->nb[2],
                                             0);                                                                  // [n_txt_token, N, hidden_size]
            txt_attn_out      = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, txt_attn_out, 0, 2, 1, 3));  // [N, n_txt_token, hidden_size]
            auto img_attn_out = ggml_view_3d(ctx->ggml_ctx,
                                             attn,
                                             attn->ne[0],
                                             attn->ne[1],
                                             img->ne[1],
                                             attn->nb[1],
                                             attn->nb[2],
                                             attn->nb[2] * txt->ne[1]);                                           // [n_img_token, N, hidden_size]
            img_attn_out      = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, img_attn_out, 0, 2, 1, 3));  // [N, n_img_token, hidden_size]

            // calculate the img bloks
            img = ggml_add(ctx->ggml_ctx, img, ggml_mul(ctx->ggml_ctx, img_attn->post_attention(ctx, img_attn_out), img_mod1.gate));

            auto img_mlp_out = img_mlp_0->forward(ctx, Flux::modulate(ctx->ggml_ctx, img_norm2->forward(ctx, img), img_mod2.shift, img_mod2.scale));
            img_mlp_out      = ggml_gelu_inplace(ctx->ggml_ctx, img_mlp_out);
            img_mlp_out      = img_mlp_2->forward(ctx, img_mlp_out);

            img = ggml_add(ctx->ggml_ctx, img, ggml_mul(ctx->ggml_ctx, img_mlp_out, img_mod2.gate));

            // calculate the txt bloks
            txt = ggml_add(ctx->ggml_ctx, txt, ggml_mul(ctx->ggml_ctx, txt_attn->post_attention(ctx, txt_attn_out), txt_mod1.gate));

            auto txt_mlp_out = txt_mlp_0->forward(ctx, Flux::modulate(ctx->ggml_ctx, txt_norm2->forward(ctx, txt), txt_mod2.shift, txt_mod2.scale));
            txt_mlp_out      = ggml_gelu_inplace(ctx->ggml_ctx, txt_mlp_out);
            txt_mlp_out      = txt_mlp_2->forward(ctx, txt_mlp_out);

            txt = ggml_add(ctx->ggml_ctx, txt, ggml_mul(ctx->ggml_ctx, txt_mlp_out, txt_mod2.gate));

            return {img, txt};
        }
    };

    struct SingleStreamBlock : public GGMLBlock {
    public:
        int64_t num_heads;
        int64_t hidden_size;
        int64_t mlp_hidden_dim;
        bool prune_mod;
        int idx = 0;

    public:
        SingleStreamBlock(int64_t hidden_size,
                          int64_t num_heads,
                          float mlp_ratio = 4.0f,
                          int idx         = 0,
                          float qk_scale  = 0.f,
                          bool prune_mod  = false)
            : hidden_size(hidden_size), num_heads(num_heads), idx(idx), prune_mod(prune_mod) {
            int64_t head_dim = hidden_size / num_heads;
            float scale      = qk_scale;
            if (scale <= 0.f) {
                scale = 1 / sqrt((float)head_dim);
            }
            mlp_hidden_dim = hidden_size * mlp_ratio;

            blocks["linear1"]  = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, hidden_size * 3 + mlp_hidden_dim));
            blocks["linear2"]  = std::shared_ptr<GGMLBlock>(new Linear(hidden_size + mlp_hidden_dim, hidden_size));
            blocks["norm"]     = std::shared_ptr<GGMLBlock>(new QKNorm(head_dim));
            blocks["pre_norm"] = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size, 1e-6f, false));
            // mlp_act is nn.GELU(approximate="tanh")
            if (!prune_mod) {
                blocks["modulation"] = std::shared_ptr<GGMLBlock>(new Modulation(hidden_size, false));
            }
        }

        ModulationOut get_distil_mod(GGMLRunnerContext* ctx, struct ggml_tensor* vec) {
            int64_t offset = 3 * idx;
            return ModulationOut(ctx, vec, offset);
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* vec,
                                    struct ggml_tensor* pe,
                                    struct ggml_tensor* mask = nullptr) {
            // x: [N, n_token, hidden_size]
            // pe: [n_token, d_head/2, 2, 2]
            // return: [N, n_token, hidden_size]

            auto linear1  = std::dynamic_pointer_cast<Linear>(blocks["linear1"]);
            auto linear2  = std::dynamic_pointer_cast<Linear>(blocks["linear2"]);
            auto norm     = std::dynamic_pointer_cast<QKNorm>(blocks["norm"]);
            auto pre_norm = std::dynamic_pointer_cast<LayerNorm>(blocks["pre_norm"]);
            ModulationOut mod;
            if (prune_mod) {
                mod = get_distil_mod(ctx, vec);
            } else {
                auto modulation = std::dynamic_pointer_cast<Modulation>(blocks["modulation"]);

                mod = modulation->forward(ctx, vec)[0];
            }
            auto x_mod   = Flux::modulate(ctx->ggml_ctx, pre_norm->forward(ctx, x), mod.shift, mod.scale);
            auto qkv_mlp = linear1->forward(ctx, x_mod);                                                // [N, n_token, hidden_size * 3 + mlp_hidden_dim]
            qkv_mlp      = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, qkv_mlp, 2, 0, 1, 3));  // [hidden_size * 3 + mlp_hidden_dim, N, n_token]

            auto qkv = ggml_view_3d(ctx->ggml_ctx,
                                    qkv_mlp,
                                    qkv_mlp->ne[0],
                                    qkv_mlp->ne[1],
                                    hidden_size * 3,
                                    qkv_mlp->nb[1],
                                    qkv_mlp->nb[2],
                                    0);                                                         // [hidden_size * 3 , N, n_token]
            qkv      = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, qkv, 1, 2, 0, 3));  // [N, n_token, hidden_size * 3]
            auto mlp = ggml_view_3d(ctx->ggml_ctx,
                                    qkv_mlp,
                                    qkv_mlp->ne[0],
                                    qkv_mlp->ne[1],
                                    mlp_hidden_dim,
                                    qkv_mlp->nb[1],
                                    qkv_mlp->nb[2],
                                    qkv_mlp->nb[2] * hidden_size * 3);                          // [mlp_hidden_dim , N, n_token]
            mlp      = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, mlp, 1, 2, 0, 3));  // [N, n_token, mlp_hidden_dim]

            auto qkv_vec     = split_qkv(ctx->ggml_ctx, qkv);  // q,k,v: [N, n_token, hidden_size]
            int64_t head_dim = hidden_size / num_heads;
            auto q           = ggml_reshape_4d(ctx->ggml_ctx, qkv_vec[0], head_dim, num_heads, qkv_vec[0]->ne[1], qkv_vec[0]->ne[2]);  // [N, n_token, n_head, d_head]
            auto k           = ggml_reshape_4d(ctx->ggml_ctx, qkv_vec[1], head_dim, num_heads, qkv_vec[1]->ne[1], qkv_vec[1]->ne[2]);  // [N, n_token, n_head, d_head]
            auto v           = ggml_reshape_4d(ctx->ggml_ctx, qkv_vec[2], head_dim, num_heads, qkv_vec[2]->ne[1], qkv_vec[2]->ne[2]);  // [N, n_token, n_head, d_head]
            q                = norm->query_norm(ctx, q);
            k                = norm->key_norm(ctx, k);
            auto attn        = Rope::attention(ctx, q, k, v, pe, mask);  // [N, n_token, hidden_size]

            auto attn_mlp = ggml_concat(ctx->ggml_ctx, attn, ggml_gelu_inplace(ctx->ggml_ctx, mlp), 0);  // [N, n_token, hidden_size + mlp_hidden_dim]
            auto output   = linear2->forward(ctx, attn_mlp);                                             // [N, n_token, hidden_size]

            output = ggml_add(ctx->ggml_ctx, x, ggml_mul(ctx->ggml_ctx, output, mod.gate));
            return output;
        }
    };

    struct LastLayer : public GGMLBlock {
        bool prune_mod;

    public:
        LastLayer(int64_t hidden_size,
                  int64_t patch_size,
                  int64_t out_channels,
                  bool prune_mod = false)
            : prune_mod(prune_mod) {
            blocks["norm_final"] = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size, 1e-06f, false));
            blocks["linear"]     = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, patch_size * patch_size * out_channels));
            if (!prune_mod) {
                blocks["adaLN_modulation.1"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, 2 * hidden_size));
            }
        }

        ModulationOut get_distil_mod(GGMLRunnerContext* ctx, struct ggml_tensor* vec) {
            int64_t offset = vec->ne[2] - 2;
            int64_t stride = vec->nb[1] * vec->ne[1];
            auto shift     = ggml_view_2d(ctx->ggml_ctx, vec, vec->ne[0], vec->ne[1], vec->nb[1], stride * (offset + 0));  // [N, dim]
            auto scale     = ggml_view_2d(ctx->ggml_ctx, vec, vec->ne[0], vec->ne[1], vec->nb[1], stride * (offset + 1));  // [N, dim]
            // No gate
            return {shift, scale, nullptr};
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* c) {
            // x: [N, n_token, hidden_size]
            // c: [N, hidden_size]
            // return: [N, n_token, patch_size * patch_size * out_channels]
            auto norm_final = std::dynamic_pointer_cast<LayerNorm>(blocks["norm_final"]);
            auto linear     = std::dynamic_pointer_cast<Linear>(blocks["linear"]);
            struct ggml_tensor *shift, *scale;
            if (prune_mod) {
                auto mod = get_distil_mod(ctx, c);
                shift    = mod.shift;
                scale    = mod.scale;
            } else {
                auto adaLN_modulation_1 = std::dynamic_pointer_cast<Linear>(blocks["adaLN_modulation.1"]);

                auto m = adaLN_modulation_1->forward(ctx, ggml_silu(ctx->ggml_ctx, c));         // [N, 2 * hidden_size]
                m      = ggml_reshape_3d(ctx->ggml_ctx, m, c->ne[0], 2, c->ne[1]);              // [N, 2, hidden_size]
                m      = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, m, 0, 2, 1, 3));  // [2, N, hidden_size]

                int64_t offset = m->nb[1] * m->ne[1];
                shift          = ggml_view_2d(ctx->ggml_ctx, m, m->ne[0], m->ne[1], m->nb[1], offset * 0);  // [N, hidden_size]
                scale          = ggml_view_2d(ctx->ggml_ctx, m, m->ne[0], m->ne[1], m->nb[1], offset * 1);  // [N, hidden_size]
            }

            x = Flux::modulate(ctx->ggml_ctx, norm_final->forward(ctx, x), shift, scale);
            x = linear->forward(ctx, x);

            return x;
        }
    };

    struct ChromaApproximator : public GGMLBlock {
        int64_t inner_size = 5120;
        int64_t n_layers   = 5;
        ChromaApproximator(int64_t in_channels = 64, int64_t hidden_size = 3072) {
            blocks["in_proj"] = std::shared_ptr<GGMLBlock>(new Linear(in_channels, inner_size, true));
            for (int i = 0; i < n_layers; i++) {
                blocks["norms." + std::to_string(i)]  = std::shared_ptr<GGMLBlock>(new RMSNorm(inner_size));
                blocks["layers." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new MLPEmbedder(inner_size, inner_size));
            }
            blocks["out_proj"] = std::shared_ptr<GGMLBlock>(new Linear(inner_size, hidden_size, true));
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
            auto in_proj  = std::dynamic_pointer_cast<Linear>(blocks["in_proj"]);
            auto out_proj = std::dynamic_pointer_cast<Linear>(blocks["out_proj"]);

            x = in_proj->forward(ctx, x);
            for (int i = 0; i < n_layers; i++) {
                auto norm  = std::dynamic_pointer_cast<RMSNorm>(blocks["norms." + std::to_string(i)]);
                auto embed = std::dynamic_pointer_cast<MLPEmbedder>(blocks["layers." + std::to_string(i)]);
                x          = ggml_add_inplace(ctx->ggml_ctx, x, embed->forward(ctx, norm->forward(ctx, x)));
            }
            x = out_proj->forward(ctx, x);

            return x;
        }
    };

    struct NerfEmbedder : public GGMLBlock {
        NerfEmbedder(int64_t in_channels,
                     int64_t hidden_size_input,
                     int64_t max_freqs) {
            blocks["embedder.0"] = std::make_shared<Linear>(in_channels + max_freqs * max_freqs, hidden_size_input);
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* dct) {
            // x: (B, P^2, C)
            // dct: (1, P^2, max_freqs^2)
            // return: (B, P^2, hidden_size_input)
            auto embedder = std::dynamic_pointer_cast<Linear>(blocks["embedder.0"]);

            dct = ggml_repeat_4d(ctx->ggml_ctx, dct, dct->ne[0], dct->ne[1], x->ne[2], x->ne[3]);
            x   = ggml_concat(ctx->ggml_ctx, x, dct, 0);
            x   = embedder->forward(ctx, x);

            return x;
        }
    };

    struct NerfGLUBlock : public GGMLBlock {
        int64_t mlp_ratio;
        NerfGLUBlock(int64_t hidden_size_s,
                     int64_t hidden_size_x,
                     int64_t mlp_ratio)
            : mlp_ratio(mlp_ratio) {
            int64_t total_params      = 3 * hidden_size_x * hidden_size_x * mlp_ratio;
            blocks["param_generator"] = std::make_shared<Linear>(hidden_size_s, total_params);
            blocks["norm"]            = std::make_shared<RMSNorm>(hidden_size_x);
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* s) {
            // x: (batch_size, n_token, hidden_size_x)
            // s: (batch_size, hidden_size_s)
            // return: (batch_size, n_token, hidden_size_x)
            auto param_generator = std::dynamic_pointer_cast<Linear>(blocks["param_generator"]);
            auto norm            = std::dynamic_pointer_cast<RMSNorm>(blocks["norm"]);

            int64_t batch_size    = x->ne[2];
            int64_t hidden_size_x = x->ne[0];

            auto mlp_params = param_generator->forward(ctx, s);
            auto fc_params  = ggml_ext_chunk(ctx->ggml_ctx, mlp_params, 3, 0);
            auto fc1_gate   = ggml_reshape_3d(ctx->ggml_ctx, fc_params[0], hidden_size_x * mlp_ratio, hidden_size_x, batch_size);
            auto fc1_value  = ggml_reshape_3d(ctx->ggml_ctx, fc_params[1], hidden_size_x * mlp_ratio, hidden_size_x, batch_size);
            auto fc2        = ggml_reshape_3d(ctx->ggml_ctx, fc_params[2], hidden_size_x, mlp_ratio * hidden_size_x, batch_size);

            fc1_gate  = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, fc1_gate, 1, 0, 2, 3));  // [batch_size, hidden_size_x*mlp_ratio, hidden_size_x]
            fc1_gate  = ggml_l2_norm(ctx->ggml_ctx, fc1_gate, 1e-12f);
            fc1_value = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, fc1_value, 1, 0, 2, 3));  // [batch_size, hidden_size_x*mlp_ratio, hidden_size_x]
            fc1_value = ggml_l2_norm(ctx->ggml_ctx, fc1_value, 1e-12f);
            fc2       = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, fc2, 1, 0, 2, 3));  // [batch_size, hidden_size_x, hidden_size_x*mlp_ratio]
            fc2       = ggml_l2_norm(ctx->ggml_ctx, fc2, 1e-12f);

            auto res_x = x;
            x          = norm->forward(ctx, x);  // [batch_size, n_token, hidden_size_x]

            auto x1 = ggml_mul_mat(ctx->ggml_ctx, fc1_gate, x);  // [batch_size, n_token, hidden_size_x*mlp_ratio]
            x1      = ggml_silu_inplace(ctx->ggml_ctx, x1);

            auto x2 = ggml_mul_mat(ctx->ggml_ctx, fc1_value, x);  // [batch_size, n_token, hidden_size_x*mlp_ratio]

            x = ggml_mul_inplace(ctx->ggml_ctx, x1, x2);  // [batch_size, n_token, hidden_size_x*mlp_ratio]

            x = ggml_mul_mat(ctx->ggml_ctx, fc2, x);  // [batch_size, n_token, hidden_size_x]

            x = ggml_add_inplace(ctx->ggml_ctx, x, res_x);

            return x;
        }
    };

    struct NerfFinalLayer : public GGMLBlock {
        NerfFinalLayer(int64_t hidden_size,
                       int64_t out_channels) {
            blocks["norm"]   = std::make_shared<RMSNorm>(hidden_size);
            blocks["linear"] = std::make_shared<Linear>(hidden_size, out_channels);
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x) {
            auto norm   = std::dynamic_pointer_cast<RMSNorm>(blocks["norm"]);
            auto linear = std::dynamic_pointer_cast<Linear>(blocks["linear"]);

            x = norm->forward(ctx, x);
            x = linear->forward(ctx, x);

            return x;
        }
    };

    struct NerfFinalLayerConv : public GGMLBlock {
        NerfFinalLayerConv(int64_t hidden_size,
                           int64_t out_channels) {
            blocks["norm"] = std::make_shared<RMSNorm>(hidden_size);
            blocks["conv"] = std::make_shared<Conv2d>(hidden_size, out_channels, std::pair{3, 3}, std::pair{1, 1}, std::pair{1, 1});
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x) {
            // x: [N, C, H, W]
            auto norm = std::dynamic_pointer_cast<RMSNorm>(blocks["norm"]);
            auto conv = std::dynamic_pointer_cast<Conv2d>(blocks["conv"]);

            x = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 2, 0, 1, 3));  // [N, H, W, C]
            x = norm->forward(ctx, x);
            x = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 1, 2, 0, 3));  // [N, C, H, W]
            x = conv->forward(ctx, x);

            return x;
        }
    };

    struct ChromaRadianceParams {
        int64_t nerf_hidden_size = 64;
        int64_t nerf_mlp_ratio   = 4;
        int64_t nerf_depth       = 4;
        int64_t nerf_max_freqs   = 8;
    };

    struct FluxParams {
        SDVersion version           = VERSION_FLUX;
        bool is_chroma              = false;
        int64_t patch_size          = 2;
        int64_t in_channels         = 64;
        int64_t out_channels        = 64;
        int64_t vec_in_dim          = 768;
        int64_t context_in_dim      = 4096;
        int64_t hidden_size         = 3072;
        float mlp_ratio             = 4.0f;
        int64_t num_heads           = 24;
        int64_t depth               = 19;
        int64_t depth_single_blocks = 38;
        std::vector<int> axes_dim   = {16, 56, 56};
        int64_t axes_dim_sum        = 128;
        int theta                   = 10000;
        bool qkv_bias               = true;
        bool guidance_embed         = true;
        int64_t in_dim              = 64;
        ChromaRadianceParams chroma_radiance_params;
    };

    struct Flux : public GGMLBlock {
    public:
        FluxParams params;
        Flux() {}
        Flux(FluxParams params)
            : params(params) {
            if (params.version == VERSION_CHROMA_RADIANCE) {
                std::pair<int, int> kernel_size = {(int)params.patch_size, (int)params.patch_size};
                std::pair<int, int> stride      = kernel_size;

                blocks["img_in_patch"] = std::make_shared<Conv2d>(params.in_channels,
                                                                  params.hidden_size,
                                                                  kernel_size,
                                                                  stride);
            } else {
                blocks["img_in"] = std::make_shared<Linear>(params.in_channels, params.hidden_size, true);
            }
            if (params.is_chroma) {
                blocks["distilled_guidance_layer"] = std::make_shared<ChromaApproximator>(params.in_dim, params.hidden_size);
            } else {
                blocks["time_in"]   = std::make_shared<MLPEmbedder>(256, params.hidden_size);
                blocks["vector_in"] = std::make_shared<MLPEmbedder>(params.vec_in_dim, params.hidden_size);
                if (params.guidance_embed) {
                    blocks["guidance_in"] = std::make_shared<MLPEmbedder>(256, params.hidden_size);
                }
            }
            blocks["txt_in"] = std::make_shared<Linear>(params.context_in_dim, params.hidden_size, true);

            for (int i = 0; i < params.depth; i++) {
                blocks["double_blocks." + std::to_string(i)] = std::make_shared<DoubleStreamBlock>(params.hidden_size,
                                                                                                   params.num_heads,
                                                                                                   params.mlp_ratio,
                                                                                                   i,
                                                                                                   params.qkv_bias,
                                                                                                   params.is_chroma);
            }

            for (int i = 0; i < params.depth_single_blocks; i++) {
                blocks["single_blocks." + std::to_string(i)] = std::make_shared<SingleStreamBlock>(params.hidden_size,
                                                                                                   params.num_heads,
                                                                                                   params.mlp_ratio,
                                                                                                   i,
                                                                                                   0.f,
                                                                                                   params.is_chroma);
            }

            if (params.version == VERSION_CHROMA_RADIANCE) {
                blocks["nerf_image_embedder"] = std::make_shared<NerfEmbedder>(params.in_channels,
                                                                               params.chroma_radiance_params.nerf_hidden_size,
                                                                               params.chroma_radiance_params.nerf_max_freqs);

                for (int i = 0; i < params.chroma_radiance_params.nerf_depth; i++) {
                    blocks["nerf_blocks." + std::to_string(i)] = std::make_shared<NerfGLUBlock>(params.hidden_size,
                                                                                                params.chroma_radiance_params.nerf_hidden_size,
                                                                                                params.chroma_radiance_params.nerf_mlp_ratio);
                }

                blocks["nerf_final_layer_conv"] = std::make_shared<NerfFinalLayerConv>(params.chroma_radiance_params.nerf_hidden_size,
                                                                                       params.in_channels);

            } else {
                blocks["final_layer"] = std::make_shared<LastLayer>(params.hidden_size, 1, params.out_channels, params.is_chroma);
            }
        }

        struct ggml_tensor* pad_to_patch_size(struct ggml_context* ctx,
                                              struct ggml_tensor* x) {
            int64_t W = x->ne[0];
            int64_t H = x->ne[1];

            int pad_h = (params.patch_size - H % params.patch_size) % params.patch_size;
            int pad_w = (params.patch_size - W % params.patch_size) % params.patch_size;
            x         = ggml_pad(ctx, x, pad_w, pad_h, 0, 0);  // [N, C, H + pad_h, W + pad_w]
            return x;
        }

        struct ggml_tensor* patchify(struct ggml_context* ctx,
                                     struct ggml_tensor* x) {
            // x: [N, C, H, W]
            // return: [N, h*w, C * patch_size * patch_size]
            int64_t N = x->ne[3];
            int64_t C = x->ne[2];
            int64_t H = x->ne[1];
            int64_t W = x->ne[0];
            int64_t p = params.patch_size;
            int64_t h = H / params.patch_size;
            int64_t w = W / params.patch_size;

            GGML_ASSERT(h * p == H && w * p == W);

            x = ggml_reshape_4d(ctx, x, p, w, p, h * C * N);       // [N*C*h, p, w, p]
            x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));  // [N*C*h, w, p, p]
            x = ggml_reshape_4d(ctx, x, p * p, w * h, C, N);       // [N, C, h*w, p*p]
            x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));  // [N, h*w, C, p*p]
            x = ggml_reshape_3d(ctx, x, p * p * C, w * h, N);      // [N, h*w, C*p*p]
            return x;
        }

        struct ggml_tensor* process_img(struct ggml_context* ctx,
                                        struct ggml_tensor* x) {
            // img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)
            x = pad_to_patch_size(ctx, x);
            x = patchify(ctx, x);
            return x;
        }

        struct ggml_tensor* unpatchify(struct ggml_context* ctx,
                                       struct ggml_tensor* x,
                                       int64_t h,
                                       int64_t w) {
            // x: [N, h*w, C*patch_size*patch_size]
            // return: [N, C, H, W]
            int64_t N = x->ne[2];
            int64_t C = x->ne[0] / params.patch_size / params.patch_size;
            int64_t H = h * params.patch_size;
            int64_t W = w * params.patch_size;
            int64_t p = params.patch_size;

            GGML_ASSERT(C * p * p == x->ne[0]);

            x = ggml_reshape_4d(ctx, x, p * p, C, w * h, N);       // [N, h*w, C, p*p]
            x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));  // [N, C, h*w, p*p]
            x = ggml_reshape_4d(ctx, x, p, p, w, h * C * N);       // [N*C*h, w, p, p]
            x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));  // [N*C*h, p, w, p]
            x = ggml_reshape_4d(ctx, x, W, H, C, N);               // [N, C, h*p, w*p]

            return x;
        }

        struct ggml_tensor* forward_orig(GGMLRunnerContext* ctx,
                                         struct ggml_tensor* img,
                                         struct ggml_tensor* txt,
                                         struct ggml_tensor* timesteps,
                                         struct ggml_tensor* y,
                                         struct ggml_tensor* guidance,
                                         struct ggml_tensor* pe,
                                         struct ggml_tensor* mod_index_arange = nullptr,
                                         std::vector<int> skip_layers         = {}) {
            auto img_in      = std::dynamic_pointer_cast<Linear>(blocks["img_in"]);
            auto txt_in      = std::dynamic_pointer_cast<Linear>(blocks["txt_in"]);
            auto final_layer = std::dynamic_pointer_cast<LastLayer>(blocks["final_layer"]);

            if (img_in) {
                img = img_in->forward(ctx, img);
            }

            struct ggml_tensor* vec;
            struct ggml_tensor* txt_img_mask = nullptr;
            if (params.is_chroma) {
                int64_t mod_index_length = 344;
                auto approx              = std::dynamic_pointer_cast<ChromaApproximator>(blocks["distilled_guidance_layer"]);
                auto distill_timestep    = ggml_ext_timestep_embedding(ctx->ggml_ctx, timesteps, 16, 10000, 1000.f);
                auto distill_guidance    = ggml_ext_timestep_embedding(ctx->ggml_ctx, guidance, 16, 10000, 1000.f);

                // auto mod_index_arange  = ggml_arange(ctx, 0, (float)mod_index_length, 1);
                // ggml_arange tot working on a lot of backends, precomputing it on CPU instead
                GGML_ASSERT(mod_index_arange != nullptr);
                auto modulation_index = ggml_ext_timestep_embedding(ctx->ggml_ctx, mod_index_arange, 32, 10000, 1000.f);  // [1, 344, 32]

                // Batch broadcast (will it ever be useful)
                modulation_index = ggml_repeat(ctx->ggml_ctx, modulation_index, ggml_new_tensor_3d(ctx->ggml_ctx, GGML_TYPE_F32, modulation_index->ne[0], modulation_index->ne[1], img->ne[2]));  // [N, 344, 32]

                auto timestep_guidance = ggml_concat(ctx->ggml_ctx, distill_timestep, distill_guidance, 0);  // [N, 1, 32]
                timestep_guidance      = ggml_repeat(ctx->ggml_ctx, timestep_guidance, modulation_index);    // [N, 344, 32]

                vec = ggml_concat(ctx->ggml_ctx, timestep_guidance, modulation_index, 0);  // [N, 344, 64]
                // Permute for consistency with non-distilled modulation implementation
                vec = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, vec, 0, 2, 1, 3));  // [344, N, 64]
                vec = approx->forward(ctx, vec);                                               // [344, N, hidden_size]

                if (y != nullptr) {
                    txt_img_mask = ggml_pad(ctx->ggml_ctx, y, img->ne[1], 0, 0, 0);
                }
            } else {
                auto time_in   = std::dynamic_pointer_cast<MLPEmbedder>(blocks["time_in"]);
                auto vector_in = std::dynamic_pointer_cast<MLPEmbedder>(blocks["vector_in"]);
                vec            = time_in->forward(ctx, ggml_ext_timestep_embedding(ctx->ggml_ctx, timesteps, 256, 10000, 1000.f));
                if (params.guidance_embed) {
                    GGML_ASSERT(guidance != nullptr);
                    auto guidance_in = std::dynamic_pointer_cast<MLPEmbedder>(blocks["guidance_in"]);
                    // bf16 and fp16 result is different
                    auto g_in = ggml_ext_timestep_embedding(ctx->ggml_ctx, guidance, 256, 10000, 1000.f);
                    vec       = ggml_add(ctx->ggml_ctx, vec, guidance_in->forward(ctx, g_in));
                }

                vec = ggml_add(ctx->ggml_ctx, vec, vector_in->forward(ctx, y));
            }

            txt = txt_in->forward(ctx, txt);

            for (int i = 0; i < params.depth; i++) {
                if (skip_layers.size() > 0 && std::find(skip_layers.begin(), skip_layers.end(), i) != skip_layers.end()) {
                    continue;
                }

                auto block = std::dynamic_pointer_cast<DoubleStreamBlock>(blocks["double_blocks." + std::to_string(i)]);

                auto img_txt = block->forward(ctx, img, txt, vec, pe, txt_img_mask);
                img          = img_txt.first;   // [N, n_img_token, hidden_size]
                txt          = img_txt.second;  // [N, n_txt_token, hidden_size]
            }

            auto txt_img = ggml_concat(ctx->ggml_ctx, txt, img, 1);  // [N, n_txt_token + n_img_token, hidden_size]
            for (int i = 0; i < params.depth_single_blocks; i++) {
                if (skip_layers.size() > 0 && std::find(skip_layers.begin(), skip_layers.end(), i + params.depth) != skip_layers.end()) {
                    continue;
                }
                auto block = std::dynamic_pointer_cast<SingleStreamBlock>(blocks["single_blocks." + std::to_string(i)]);

                txt_img = block->forward(ctx, txt_img, vec, pe, txt_img_mask);
            }

            txt_img = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, txt_img, 0, 2, 1, 3));  // [n_txt_token + n_img_token, N, hidden_size]
            img     = ggml_view_3d(ctx->ggml_ctx,
                                   txt_img,
                                   txt_img->ne[0],
                                   txt_img->ne[1],
                                   img->ne[1],
                                   txt_img->nb[1],
                                   txt_img->nb[2],
                                   txt_img->nb[2] * txt->ne[1]);                               // [n_img_token, N, hidden_size]
            img     = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, img, 0, 2, 1, 3));  // [N, n_img_token, hidden_size]

            if (final_layer) {
                img = final_layer->forward(ctx, img, vec);  // (N, T, patch_size ** 2 * out_channels)
            }

            return img;
        }

        struct ggml_tensor* forward_chroma_radiance(GGMLRunnerContext* ctx,
                                                    struct ggml_tensor* x,
                                                    struct ggml_tensor* timestep,
                                                    struct ggml_tensor* context,
                                                    struct ggml_tensor* c_concat,
                                                    struct ggml_tensor* y,
                                                    struct ggml_tensor* guidance,
                                                    struct ggml_tensor* pe,
                                                    struct ggml_tensor* mod_index_arange  = nullptr,
                                                    struct ggml_tensor* dct               = nullptr,
                                                    std::vector<ggml_tensor*> ref_latents = {},
                                                    std::vector<int> skip_layers          = {}) {
            GGML_ASSERT(x->ne[3] == 1);

            int64_t W          = x->ne[0];
            int64_t H          = x->ne[1];
            int64_t C          = x->ne[2];
            int64_t patch_size = params.patch_size;
            int pad_h          = (patch_size - H % patch_size) % patch_size;
            int pad_w          = (patch_size - W % patch_size) % patch_size;

            auto img      = pad_to_patch_size(ctx->ggml_ctx, x);
            auto orig_img = img;

            auto img_in_patch = std::dynamic_pointer_cast<Conv2d>(blocks["img_in_patch"]);

            img = img_in_patch->forward(ctx, img);                                                       // [N, hidden_size, H/patch_size, W/patch_size]
            img = ggml_reshape_3d(ctx->ggml_ctx, img, img->ne[0] * img->ne[1], img->ne[2], img->ne[3]);  // [N, hidden_size, H/patch_size*W/patch_size]
            img = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, img, 1, 0, 2, 3));      // [N, H/patch_size*W/patch_size, hidden_size]

            auto out = forward_orig(ctx, img, context, timestep, y, guidance, pe, mod_index_arange, skip_layers);  // [N, n_img_token, hidden_size]

            // nerf decode
            auto nerf_image_embedder   = std::dynamic_pointer_cast<NerfEmbedder>(blocks["nerf_image_embedder"]);
            auto nerf_final_layer_conv = std::dynamic_pointer_cast<NerfFinalLayerConv>(blocks["nerf_final_layer_conv"]);

            auto nerf_pixels    = patchify(ctx->ggml_ctx, orig_img);  // [N, num_patches, C * patch_size * patch_size]
            int64_t num_patches = nerf_pixels->ne[1];
            nerf_pixels         = ggml_reshape_3d(ctx->ggml_ctx,
                                                  nerf_pixels,
                                                  nerf_pixels->ne[0] / C,
                                                  C,
                                                  nerf_pixels->ne[1] * nerf_pixels->ne[2]);                                  // [N*num_patches, C, patch_size*patch_size]
            nerf_pixels         = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, nerf_pixels, 1, 0, 2, 3));  // [N*num_patches, patch_size*patch_size, C]

            auto nerf_hidden = ggml_reshape_2d(ctx->ggml_ctx, out, out->ne[0], out->ne[1] * out->ne[2]);  // [N*num_patches, hidden_size]
            auto img_dct     = nerf_image_embedder->forward(ctx, nerf_pixels, dct);                       // [N*num_patches, patch_size*patch_size, nerf_hidden_size]

            for (int i = 0; i < params.chroma_radiance_params.nerf_depth; i++) {
                auto block = std::dynamic_pointer_cast<NerfGLUBlock>(blocks["nerf_blocks." + std::to_string(i)]);

                img_dct = block->forward(ctx, img_dct, nerf_hidden);
            }

            img_dct = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, img_dct, 1, 0, 2, 3));                                 // [N*num_patches, nerf_hidden_size, patch_size*patch_size]
            img_dct = ggml_reshape_3d(ctx->ggml_ctx, img_dct, img_dct->ne[0] * img_dct->ne[1], num_patches, img_dct->ne[2] / num_patches);  // [N, num_patches, nerf_hidden_size*patch_size*patch_size]
            img_dct = unpatchify(ctx->ggml_ctx, img_dct, (H + pad_h) / patch_size, (W + pad_w) / patch_size);                               // [N, nerf_hidden_size, H, W]

            out = nerf_final_layer_conv->forward(ctx, img_dct);  // [N, C, H, W]

            return out;
        }

        struct ggml_tensor* forward_flux_chroma(GGMLRunnerContext* ctx,
                                                struct ggml_tensor* x,
                                                struct ggml_tensor* timestep,
                                                struct ggml_tensor* context,
                                                struct ggml_tensor* c_concat,
                                                struct ggml_tensor* y,
                                                struct ggml_tensor* guidance,
                                                struct ggml_tensor* pe,
                                                struct ggml_tensor* mod_index_arange  = nullptr,
                                                struct ggml_tensor* dct               = nullptr,
                                                std::vector<ggml_tensor*> ref_latents = {},
                                                std::vector<int> skip_layers          = {}) {
            GGML_ASSERT(x->ne[3] == 1);

            int64_t W          = x->ne[0];
            int64_t H          = x->ne[1];
            int64_t C          = x->ne[2];
            int64_t patch_size = params.patch_size;
            int pad_h          = (patch_size - H % patch_size) % patch_size;
            int pad_w          = (patch_size - W % patch_size) % patch_size;

            auto img            = process_img(ctx->ggml_ctx, x);
            uint64_t img_tokens = img->ne[1];

            if (params.version == VERSION_FLUX_FILL) {
                GGML_ASSERT(c_concat != nullptr);
                ggml_tensor* masked = ggml_view_4d(ctx->ggml_ctx, c_concat, c_concat->ne[0], c_concat->ne[1], C, 1, c_concat->nb[1], c_concat->nb[2], c_concat->nb[3], 0);
                ggml_tensor* mask   = ggml_view_4d(ctx->ggml_ctx, c_concat, c_concat->ne[0], c_concat->ne[1], 8 * 8, 1, c_concat->nb[1], c_concat->nb[2], c_concat->nb[3], c_concat->nb[2] * C);

                masked = process_img(ctx->ggml_ctx, masked);
                mask   = process_img(ctx->ggml_ctx, mask);

                img = ggml_concat(ctx->ggml_ctx, img, ggml_concat(ctx->ggml_ctx, masked, mask, 0), 0);
            } else if (params.version == VERSION_FLEX_2) {
                GGML_ASSERT(c_concat != nullptr);
                ggml_tensor* masked  = ggml_view_4d(ctx->ggml_ctx, c_concat, c_concat->ne[0], c_concat->ne[1], C, 1, c_concat->nb[1], c_concat->nb[2], c_concat->nb[3], 0);
                ggml_tensor* mask    = ggml_view_4d(ctx->ggml_ctx, c_concat, c_concat->ne[0], c_concat->ne[1], 1, 1, c_concat->nb[1], c_concat->nb[2], c_concat->nb[3], c_concat->nb[2] * C);
                ggml_tensor* control = ggml_view_4d(ctx->ggml_ctx, c_concat, c_concat->ne[0], c_concat->ne[1], C, 1, c_concat->nb[1], c_concat->nb[2], c_concat->nb[3], c_concat->nb[2] * (C + 1));

                masked  = process_img(ctx->ggml_ctx, masked);
                mask    = process_img(ctx->ggml_ctx, mask);
                control = process_img(ctx->ggml_ctx, control);

                img = ggml_concat(ctx->ggml_ctx, img, ggml_concat(ctx->ggml_ctx, ggml_concat(ctx->ggml_ctx, masked, mask, 0), control, 0), 0);
            } else if (params.version == VERSION_FLUX_CONTROLS) {
                GGML_ASSERT(c_concat != nullptr);

                auto control = process_img(ctx->ggml_ctx, c_concat);
                img          = ggml_concat(ctx->ggml_ctx, img, control, 0);
            }

            if (ref_latents.size() > 0) {
                for (ggml_tensor* ref : ref_latents) {
                    ref = process_img(ctx->ggml_ctx, ref);
                    img = ggml_concat(ctx->ggml_ctx, img, ref, 1);
                }
            }

            auto out = forward_orig(ctx, img, context, timestep, y, guidance, pe, mod_index_arange, skip_layers);  // [N, num_tokens, C * patch_size * patch_size]

            if (out->ne[1] > img_tokens) {
                out = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, out, 0, 2, 1, 3));  // [num_tokens, N, C * patch_size * patch_size]
                out = ggml_view_3d(ctx->ggml_ctx, out, out->ne[0], out->ne[1], img_tokens, out->nb[1], out->nb[2], 0);
                out = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, out, 0, 2, 1, 3));  // [N, h*w, C * patch_size * patch_size]
            }

            // rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=2, pw=2)
            out = unpatchify(ctx->ggml_ctx, out, (H + pad_h) / patch_size, (W + pad_w) / patch_size);  // [N, C, H + pad_h, W + pad_w]
            return out;
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* timestep,
                                    struct ggml_tensor* context,
                                    struct ggml_tensor* c_concat,
                                    struct ggml_tensor* y,
                                    struct ggml_tensor* guidance,
                                    struct ggml_tensor* pe,
                                    struct ggml_tensor* mod_index_arange  = nullptr,
                                    struct ggml_tensor* dct               = nullptr,
                                    std::vector<ggml_tensor*> ref_latents = {},
                                    std::vector<int> skip_layers          = {}) {
            // Forward pass of DiT.
            // x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
            // timestep: (N,) tensor of diffusion timesteps
            // context: (N, L, D)
            // c_concat: nullptr, or for (N,C+M, H, W) for Fill
            // y: (N, adm_in_channels) tensor of class labels
            // guidance: (N,)
            // pe: (L, d_head/2, 2, 2)
            // return: (N, C, H, W)

            if (params.version == VERSION_CHROMA_RADIANCE) {
                return forward_chroma_radiance(ctx,
                                               x,
                                               timestep,
                                               context,
                                               c_concat,
                                               y,
                                               guidance,
                                               pe,
                                               mod_index_arange,
                                               dct,
                                               ref_latents,
                                               skip_layers);
            } else {
                return forward_flux_chroma(ctx,
                                           x,
                                           timestep,
                                           context,
                                           c_concat,
                                           y,
                                           guidance,
                                           pe,
                                           mod_index_arange,
                                           dct,
                                           ref_latents,
                                           skip_layers);
            }
        }
    };

    struct FluxRunner : public GGMLRunner {
    public:
        FluxParams flux_params;
        Flux flux;
        std::vector<float> pe_vec;
        std::vector<float> mod_index_arange_vec;
        std::vector<float> dct_vec;
        SDVersion version;
        bool use_mask = false;

        FluxRunner(ggml_backend_t backend,
                   bool offload_params_to_cpu,
                   const String2GGMLType& tensor_types = {},
                   const std::string prefix            = "",
                   SDVersion version                   = VERSION_FLUX,
                   bool use_mask                       = false)
            : GGMLRunner(backend, offload_params_to_cpu), version(version), use_mask(use_mask) {
            flux_params.version             = version;
            flux_params.guidance_embed      = false;
            flux_params.depth               = 0;
            flux_params.depth_single_blocks = 0;
            if (version == VERSION_FLUX_FILL) {
                flux_params.in_channels = 384;
            } else if (version == VERSION_FLUX_CONTROLS) {
                flux_params.in_channels = 128;
            } else if (version == VERSION_FLEX_2) {
                flux_params.in_channels = 196;
            } else if (version == VERSION_CHROMA_RADIANCE) {
                flux_params.in_channels = 3;
                flux_params.patch_size  = 16;
            }
            for (auto pair : tensor_types) {
                std::string tensor_name = pair.first;
                if (!starts_with(tensor_name, prefix))
                    continue;
                if (tensor_name.find("guidance_in.in_layer.weight") != std::string::npos) {
                    // not schnell
                    flux_params.guidance_embed = true;
                }
                if (tensor_name.find("distilled_guidance_layer.in_proj.weight") != std::string::npos) {
                    // Chroma
                    flux_params.is_chroma = true;
                }
                size_t db = tensor_name.find("double_blocks.");
                if (db != std::string::npos) {
                    tensor_name     = tensor_name.substr(db);  // remove prefix
                    int block_depth = atoi(tensor_name.substr(14, tensor_name.find(".", 14)).c_str());
                    if (block_depth + 1 > flux_params.depth) {
                        flux_params.depth = block_depth + 1;
                    }
                }
                size_t sb = tensor_name.find("single_blocks.");
                if (sb != std::string::npos) {
                    tensor_name     = tensor_name.substr(sb);  // remove prefix
                    int block_depth = atoi(tensor_name.substr(14, tensor_name.find(".", 14)).c_str());
                    if (block_depth + 1 > flux_params.depth_single_blocks) {
                        flux_params.depth_single_blocks = block_depth + 1;
                    }
                }
            }

            LOG_INFO("Flux blocks: %d double, %d single", flux_params.depth, flux_params.depth_single_blocks);
            if (flux_params.is_chroma) {
                LOG_INFO("Using pruned modulation (Chroma)");
            } else if (!flux_params.guidance_embed) {
                LOG_INFO("Flux guidance is disabled (Schnell mode)");
            }

            flux = Flux(flux_params);
            flux.init(params_ctx, tensor_types, prefix);
        }

        std::string get_desc() override {
            return "flux";
        }

        void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
            flux.get_param_tensors(tensors, prefix);
        }

        std::vector<float> fetch_dct_pos(int patch_size, int max_freqs) {
            const float PI = 3.14159265358979323846f;

            std::vector<float> pos(patch_size);
            for (int i = 0; i < patch_size; ++i) {
                pos[i] = static_cast<float>(i) / static_cast<float>(patch_size - 1);
            }

            std::vector<float> pos_x(patch_size * patch_size);
            std::vector<float> pos_y(patch_size * patch_size);
            for (int i = 0; i < patch_size; ++i) {
                for (int j = 0; j < patch_size; ++j) {
                    pos_x[i * patch_size + j] = pos[j];
                    pos_y[i * patch_size + j] = pos[i];
                }
            }

            std::vector<float> freqs(max_freqs);
            for (int i = 0; i < max_freqs; ++i) {
                freqs[i] = static_cast<float>(i);
            }

            std::vector<float> coeffs(max_freqs * max_freqs);
            for (int fx = 0; fx < max_freqs; ++fx) {
                for (int fy = 0; fy < max_freqs; ++fy) {
                    coeffs[fx * max_freqs + fy] = 1.0f / (1.0f + freqs[fx] * freqs[fy]);
                }
            }

            int num_positions = patch_size * patch_size;
            int num_features  = max_freqs * max_freqs;
            std::vector<float> dct(num_positions * num_features);

            for (int p = 0; p < num_positions; ++p) {
                float px = pos_x[p];
                float py = pos_y[p];

                for (int fx = 0; fx < max_freqs; ++fx) {
                    float cx = std::cos(px * freqs[fx] * PI);
                    for (int fy = 0; fy < max_freqs; ++fy) {
                        float cy                                      = std::cos(py * freqs[fy] * PI);
                        float val                                     = cx * cy * coeffs[fx * max_freqs + fy];
                        dct[p * num_features + (fx * max_freqs + fy)] = val;
                    }
                }
            }

            return dct;
        }

        struct ggml_cgraph* build_graph(struct ggml_tensor* x,
                                        struct ggml_tensor* timesteps,
                                        struct ggml_tensor* context,
                                        struct ggml_tensor* c_concat,
                                        struct ggml_tensor* y,
                                        struct ggml_tensor* guidance,
                                        std::vector<ggml_tensor*> ref_latents = {},
                                        bool increase_ref_index               = false,
                                        std::vector<int> skip_layers          = {}) {
            GGML_ASSERT(x->ne[3] == 1);
            struct ggml_cgraph* gf = ggml_new_graph_custom(compute_ctx, FLUX_GRAPH_SIZE, false);

            struct ggml_tensor* mod_index_arange = nullptr;
            struct ggml_tensor* dct              = nullptr;  // for chroma radiance

            x       = to_backend(x);
            context = to_backend(context);
            if (c_concat != nullptr) {
                c_concat = to_backend(c_concat);
            }
            if (flux_params.is_chroma) {
                guidance = ggml_set_f32(guidance, 0);

                if (!use_mask) {
                    y = nullptr;
                }

                // ggml_arange is not working on some backends, precompute it
                mod_index_arange_vec = arange(0, 344);
                mod_index_arange     = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_F32, mod_index_arange_vec.size());
                set_backend_tensor_data(mod_index_arange, mod_index_arange_vec.data());
            }
            y = to_backend(y);

            timesteps = to_backend(timesteps);
            if (flux_params.guidance_embed || flux_params.is_chroma) {
                guidance = to_backend(guidance);
            }
            for (int i = 0; i < ref_latents.size(); i++) {
                ref_latents[i] = to_backend(ref_latents[i]);
            }

            pe_vec      = Rope::gen_flux_pe(x->ne[1],
                                            x->ne[0],
                                            flux_params.patch_size,
                                            x->ne[3],
                                            context->ne[1],
                                            ref_latents,
                                            increase_ref_index,
                                            flux_params.theta,
                                            flux_params.axes_dim);
            int pos_len = pe_vec.size() / flux_params.axes_dim_sum / 2;
            // LOG_DEBUG("pos_len %d", pos_len);
            auto pe = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, flux_params.axes_dim_sum / 2, pos_len);
            // pe->data = pe_vec.data();
            // print_ggml_tensor(pe);
            // pe->data = nullptr;
            set_backend_tensor_data(pe, pe_vec.data());

            if (version == VERSION_CHROMA_RADIANCE) {
                int64_t patch_size     = flux_params.patch_size;
                int64_t nerf_max_freqs = flux_params.chroma_radiance_params.nerf_max_freqs;
                dct_vec                = fetch_dct_pos(patch_size, nerf_max_freqs);
                dct                    = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, nerf_max_freqs * nerf_max_freqs, patch_size * patch_size);
                // dct->data = dct_vec.data();
                // print_ggml_tensor(dct);
                // dct->data = nullptr;
                set_backend_tensor_data(dct, dct_vec.data());
            }

            auto runner_ctx = get_context();

            struct ggml_tensor* out = flux.forward(&runner_ctx,
                                                   x,
                                                   timesteps,
                                                   context,
                                                   c_concat,
                                                   y,
                                                   guidance,
                                                   pe,
                                                   mod_index_arange,
                                                   dct,
                                                   ref_latents,
                                                   skip_layers);

            ggml_build_forward_expand(gf, out);

            return gf;
        }

        void compute(int n_threads,
                     struct ggml_tensor* x,
                     struct ggml_tensor* timesteps,
                     struct ggml_tensor* context,
                     struct ggml_tensor* c_concat,
                     struct ggml_tensor* y,
                     struct ggml_tensor* guidance,
                     std::vector<ggml_tensor*> ref_latents = {},
                     bool increase_ref_index               = false,
                     struct ggml_tensor** output           = nullptr,
                     struct ggml_context* output_ctx       = nullptr,
                     std::vector<int> skip_layers          = std::vector<int>()) {
            // x: [N, in_channels, h, w]
            // timesteps: [N, ]
            // context: [N, max_position, hidden_size]
            // y: [N, adm_in_channels] or [1, adm_in_channels]
            // guidance: [N, ]
            auto get_graph = [&]() -> struct ggml_cgraph* {
                return build_graph(x, timesteps, context, c_concat, y, guidance, ref_latents, increase_ref_index, skip_layers);
            };

            GGMLRunner::compute(get_graph, n_threads, false, output, output_ctx);
        }

        void test() {
            struct ggml_init_params params;
            params.mem_size   = static_cast<size_t>(1024 * 1024) * 1024;  // 1GB
            params.mem_buffer = nullptr;
            params.no_alloc   = false;

            struct ggml_context* work_ctx = ggml_init(params);
            GGML_ASSERT(work_ctx != nullptr);

            {
                // cpu f16:
                // cuda f16: nan
                // cuda q8_0: pass
                // auto x = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, 16, 16, 16, 1);
                // ggml_set_f32(x, 0.01f);
                auto x = load_tensor_from_file(work_ctx, "chroma_x.bin");
                // print_ggml_tensor(x);

                std::vector<float> timesteps_vec(1, 1.f);
                auto timesteps = vector_to_ggml_tensor(work_ctx, timesteps_vec);

                std::vector<float> guidance_vec(1, 0.f);
                auto guidance = vector_to_ggml_tensor(work_ctx, guidance_vec);

                // auto context = ggml_new_tensor_3d(work_ctx, GGML_TYPE_F32, 4096, 256, 1);
                // ggml_set_f32(context, 0.01f);
                auto context = load_tensor_from_file(work_ctx, "chroma_context.bin");
                // print_ggml_tensor(context);

                // auto y = ggml_new_tensor_2d(work_ctx, GGML_TYPE_F32, 768, 1);
                // ggml_set_f32(y, 0.01f);
                auto y = nullptr;
                // print_ggml_tensor(y);

                struct ggml_tensor* out = nullptr;

                int t0 = ggml_time_ms();
                compute(8, x, timesteps, context, nullptr, y, guidance, {}, false, &out, work_ctx);
                int t1 = ggml_time_ms();

                print_ggml_tensor(out);
                LOG_DEBUG("flux test done in %dms", t1 - t0);
            }
        }

        static void load_from_file_and_test(const std::string& file_path) {
            // ggml_backend_t backend = ggml_backend_cuda_init(0);
            ggml_backend_t backend    = ggml_backend_cpu_init();
            ggml_type model_data_type = GGML_TYPE_Q8_0;

            ModelLoader model_loader;
            if (!model_loader.init_from_file(file_path, "model.diffusion_model.")) {
                LOG_ERROR("init model loader from file failed: '%s'", file_path.c_str());
                return;
            }

            auto tensor_types = model_loader.tensor_storages_types;
            for (auto& item : tensor_types) {
                // LOG_DEBUG("%s %u", item.first.c_str(), item.second);
                if (ends_with(item.first, "weight")) {
                    // item.second = model_data_type;
                }
            }

            std::shared_ptr<FluxRunner> flux = std::make_shared<FluxRunner>(backend,
                                                                            false,
                                                                            tensor_types,
                                                                            "model.diffusion_model",
                                                                            VERSION_CHROMA_RADIANCE,
                                                                            false);

            flux->alloc_params_buffer();
            std::map<std::string, ggml_tensor*> tensors;
            flux->get_param_tensors(tensors, "model.diffusion_model");

            bool success = model_loader.load_tensors(tensors);

            if (!success) {
                LOG_ERROR("load tensors from model loader failed");
                return;
            }

            LOG_INFO("flux model loaded");
            flux->test();
        }
    };

}  // namespace Flux

#endif  // __FLUX_HPP__
