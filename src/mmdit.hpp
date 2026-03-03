#ifndef __MMDIT_HPP__
#define __MMDIT_HPP__

#include <memory>

#include "common_dit.hpp"
#include "ggml_extend.hpp"
#include "layer_streaming.hpp"
#include "model.h"

#define MMDIT_GRAPH_SIZE 10240

struct Mlp : public GGMLBlock {
public:
    Mlp(int64_t in_features,
        int64_t hidden_features = -1,
        int64_t out_features    = -1,
        bool bias               = true) {
        // act_layer is always lambda: nn.GELU(approximate="tanh")
        // norm_layer is always None
        // use_conv is always False
        if (hidden_features == -1) {
            hidden_features = in_features;
        }
        if (out_features == -1) {
            out_features = in_features;
        }
        blocks["fc1"] = std::shared_ptr<GGMLBlock>(new Linear(in_features, hidden_features, bias));
        blocks["fc2"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_features, out_features, bias));
    }

    struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
        // x: [N, n_token, in_features]
        auto fc1 = std::dynamic_pointer_cast<Linear>(blocks["fc1"]);
        auto fc2 = std::dynamic_pointer_cast<Linear>(blocks["fc2"]);

        x = fc1->forward(ctx, x);
        x = ggml_ext_gelu(ctx->ggml_ctx, x, true);
        x = fc2->forward(ctx, x);
        return x;
    }
};

struct PatchEmbed : public GGMLBlock {
    // 2D Image to Patch Embedding
protected:
    bool flatten;
    bool dynamic_img_pad;
    int patch_size;

public:
    PatchEmbed(int64_t img_size     = 224,
               int patch_size       = 16,
               int64_t in_chans     = 3,
               int64_t embed_dim    = 1536,
               bool bias            = true,
               bool flatten         = true,
               bool dynamic_img_pad = true)
        : patch_size(patch_size),
          flatten(flatten),
          dynamic_img_pad(dynamic_img_pad) {
        // img_size is always None
        // patch_size is always 2
        // in_chans is always 16
        // norm_layer is always False
        // strict_img_size is always true, but not used

        blocks["proj"] = std::shared_ptr<GGMLBlock>(new Conv2d(in_chans,
                                                               embed_dim,
                                                               {patch_size, patch_size},
                                                               {patch_size, patch_size},
                                                               {0, 0},
                                                               {1, 1},
                                                               bias));
    }

    struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
        // x: [N, C, H, W]
        // return: [N, H*W, embed_dim]
        auto proj = std::dynamic_pointer_cast<Conv2d>(blocks["proj"]);

        if (dynamic_img_pad) {
            int64_t W = x->ne[0];
            int64_t H = x->ne[1];
            int pad_h = (patch_size - H % patch_size) % patch_size;
            int pad_w = (patch_size - W % patch_size) % patch_size;
            x         = ggml_pad(ctx->ggml_ctx, x, pad_w, pad_h, 0, 0);  // TODO: reflect pad mode
        }
        x = proj->forward(ctx, x);

        if (flatten) {
            x = ggml_reshape_3d(ctx->ggml_ctx, x, x->ne[0] * x->ne[1], x->ne[2], x->ne[3]);
            x = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 1, 0, 2, 3));
        }
        return x;
    }
};

struct TimestepEmbedder : public GGMLBlock {
    // Embeds scalar timesteps into vector representations.
protected:
    int frequency_embedding_size;

public:
    TimestepEmbedder(int64_t hidden_size,
                     int frequency_embedding_size = 256,
                     int64_t out_channels         = 0)
        : frequency_embedding_size(frequency_embedding_size) {
        if (out_channels <= 0) {
            out_channels = hidden_size;
        }
        blocks["mlp.0"] = std::shared_ptr<GGMLBlock>(new Linear(frequency_embedding_size, hidden_size, true, true));
        blocks["mlp.2"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, out_channels, true, true));
    }

    struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* t) {
        // t: [N, ]
        // return: [N, hidden_size]
        auto mlp_0 = std::dynamic_pointer_cast<Linear>(blocks["mlp.0"]);
        auto mlp_2 = std::dynamic_pointer_cast<Linear>(blocks["mlp.2"]);

        auto t_freq = ggml_ext_timestep_embedding(ctx->ggml_ctx, t, frequency_embedding_size);  // [N, frequency_embedding_size]

        auto t_emb = mlp_0->forward(ctx, t_freq);
        t_emb      = ggml_silu_inplace(ctx->ggml_ctx, t_emb);
        t_emb      = mlp_2->forward(ctx, t_emb);
        return t_emb;
    }
};

struct VectorEmbedder : public GGMLBlock {
    // Embeds a flat vector of dimension input_dim
public:
    VectorEmbedder(int64_t input_dim,
                   int64_t hidden_size) {
        blocks["mlp.0"] = std::shared_ptr<GGMLBlock>(new Linear(input_dim, hidden_size, true, true));
        blocks["mlp.2"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, hidden_size, true, true));
    }

    struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
        // x: [N, input_dim]
        // return: [N, hidden_size]
        auto mlp_0 = std::dynamic_pointer_cast<Linear>(blocks["mlp.0"]);
        auto mlp_2 = std::dynamic_pointer_cast<Linear>(blocks["mlp.2"]);

        x = mlp_0->forward(ctx, x);
        x = ggml_silu_inplace(ctx->ggml_ctx, x);
        x = mlp_2->forward(ctx, x);
        return x;
    }
};

class SelfAttention : public GGMLBlock {
public:
    int64_t num_heads;
    bool pre_only;
    std::string qk_norm;

public:
    SelfAttention(int64_t dim,
                  int64_t num_heads   = 8,
                  std::string qk_norm = "",
                  bool qkv_bias       = false,
                  bool pre_only       = false)
        : num_heads(num_heads), pre_only(pre_only), qk_norm(qk_norm) {
        int64_t d_head = dim / num_heads;
        blocks["qkv"]  = std::shared_ptr<GGMLBlock>(new Linear(dim, dim * 3, qkv_bias));
        if (!pre_only) {
            blocks["proj"] = std::shared_ptr<GGMLBlock>(new Linear(dim, dim));
        }
        if (qk_norm == "rms") {
            blocks["ln_q"] = std::shared_ptr<GGMLBlock>(new RMSNorm(d_head, 1.0e-6f));
            blocks["ln_k"] = std::shared_ptr<GGMLBlock>(new RMSNorm(d_head, 1.0e-6f));
        } else if (qk_norm == "ln") {
            blocks["ln_q"] = std::shared_ptr<GGMLBlock>(new LayerNorm(d_head, 1.0e-6f));
            blocks["ln_k"] = std::shared_ptr<GGMLBlock>(new LayerNorm(d_head, 1.0e-6f));
        }
    }

    std::vector<struct ggml_tensor*> pre_attention(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
        auto qkv_proj = std::dynamic_pointer_cast<Linear>(blocks["qkv"]);

        auto qkv         = qkv_proj->forward(ctx, x);
        auto qkv_vec     = split_qkv(ctx->ggml_ctx, qkv);
        int64_t head_dim = qkv_vec[0]->ne[0] / num_heads;
        auto q           = ggml_reshape_4d(ctx->ggml_ctx, qkv_vec[0], head_dim, num_heads, qkv_vec[0]->ne[1], qkv_vec[0]->ne[2]);  // [N, n_token, n_head, d_head]
        auto k           = ggml_reshape_4d(ctx->ggml_ctx, qkv_vec[1], head_dim, num_heads, qkv_vec[1]->ne[1], qkv_vec[1]->ne[2]);  // [N, n_token, n_head, d_head]
        auto v           = qkv_vec[2];                                                                                             // [N, n_token, n_head*d_head]

        if (qk_norm == "rms" || qk_norm == "ln") {
            auto ln_q = std::dynamic_pointer_cast<UnaryBlock>(blocks["ln_q"]);
            auto ln_k = std::dynamic_pointer_cast<UnaryBlock>(blocks["ln_k"]);
            q         = ln_q->forward(ctx, q);
            k         = ln_k->forward(ctx, k);
        }

        q = ggml_reshape_3d(ctx->ggml_ctx, q, q->ne[0] * q->ne[1], q->ne[2], q->ne[3]);  // [N, n_token, n_head*d_head]
        k = ggml_reshape_3d(ctx->ggml_ctx, k, k->ne[0] * k->ne[1], k->ne[2], k->ne[3]);  // [N, n_token, n_head*d_head]

        return {q, k, v};
    }

    struct ggml_tensor* post_attention(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
        GGML_ASSERT(!pre_only);

        auto proj = std::dynamic_pointer_cast<Linear>(blocks["proj"]);

        x = proj->forward(ctx, x);  // [N, n_token, dim]
        return x;
    }

    // x: [N, n_token, dim]
    struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                struct ggml_tensor* x) {
        auto qkv = pre_attention(ctx, x);
        x        = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, qkv[0], qkv[1], qkv[2], num_heads, nullptr, false, ctx->flash_attn_enabled);  // [N, n_token, dim]
        x        = post_attention(ctx, x);                                                                                                           // [N, n_token, dim]
        return x;
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

struct DismantledBlock : public GGMLBlock {
    // A DiT block with gated adaptive layer norm (adaLN) conditioning.
public:
    int64_t num_heads;
    bool pre_only;
    bool self_attn;

public:
    DismantledBlock(int64_t hidden_size,
                    int64_t num_heads,
                    float mlp_ratio     = 4.0,
                    std::string qk_norm = "",
                    bool qkv_bias       = false,
                    bool pre_only       = false,
                    bool self_attn      = false)
        : num_heads(num_heads), pre_only(pre_only), self_attn(self_attn) {
        // rmsnorm is always Flase
        // scale_mod_only is always Flase
        // swiglu is always Flase
        blocks["norm1"] = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size, 1e-06f, false));
        blocks["attn"]  = std::shared_ptr<GGMLBlock>(new SelfAttention(hidden_size, num_heads, qk_norm, qkv_bias, pre_only));

        if (self_attn) {
            blocks["attn2"] = std::shared_ptr<GGMLBlock>(new SelfAttention(hidden_size, num_heads, qk_norm, qkv_bias, false));
        }

        if (!pre_only) {
            blocks["norm2"]        = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size, 1e-06f, false));
            int64_t mlp_hidden_dim = (int64_t)(hidden_size * mlp_ratio);
            blocks["mlp"]          = std::shared_ptr<GGMLBlock>(new Mlp(hidden_size, mlp_hidden_dim));
        }

        int64_t n_mods = 6;
        if (pre_only) {
            n_mods = 2;
        }
        if (self_attn) {
            n_mods = 9;
        }
        blocks["adaLN_modulation.1"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, n_mods * hidden_size));
    }

    std::tuple<std::vector<ggml_tensor*>, std::vector<ggml_tensor*>, std::vector<ggml_tensor*>> pre_attention_x(GGMLRunnerContext* ctx,
                                                                                                                struct ggml_tensor* x,
                                                                                                                struct ggml_tensor* c) {
        GGML_ASSERT(self_attn);
        // x: [N, n_token, hidden_size]
        // c: [N, hidden_size]
        auto norm1              = std::dynamic_pointer_cast<LayerNorm>(blocks["norm1"]);
        auto attn               = std::dynamic_pointer_cast<SelfAttention>(blocks["attn"]);
        auto attn2              = std::dynamic_pointer_cast<SelfAttention>(blocks["attn2"]);
        auto adaLN_modulation_1 = std::dynamic_pointer_cast<Linear>(blocks["adaLN_modulation.1"]);

        int n_mods = 9;
        auto m     = adaLN_modulation_1->forward(ctx, ggml_silu(ctx->ggml_ctx, c));  // [N, n_mods * hidden_size]
        auto m_vec = ggml_ext_chunk(ctx->ggml_ctx, m, n_mods, 0);

        auto shift_msa  = m_vec[0];  // [N, hidden_size]
        auto scale_msa  = m_vec[1];  // [N, hidden_size]
        auto gate_msa   = m_vec[2];  // [N, hidden_size]
        auto shift_mlp  = m_vec[3];  // [N, hidden_size]
        auto scale_mlp  = m_vec[4];  // [N, hidden_size]
        auto gate_mlp   = m_vec[5];  // [N, hidden_size]
        auto shift_msa2 = m_vec[6];  // [N, hidden_size]
        auto scale_msa2 = m_vec[7];  // [N, hidden_size]
        auto gate_msa2  = m_vec[8];  // [N, hidden_size]

        auto x_norm = norm1->forward(ctx, x);

        auto attn_in = modulate(ctx->ggml_ctx, x_norm, shift_msa, scale_msa);
        auto qkv     = attn->pre_attention(ctx, attn_in);

        auto attn2_in = modulate(ctx->ggml_ctx, x_norm, shift_msa2, scale_msa2);
        auto qkv2     = attn2->pre_attention(ctx, attn2_in);

        return {qkv, qkv2, {x, gate_msa, shift_mlp, scale_mlp, gate_mlp, gate_msa2}};
    }

    std::pair<std::vector<struct ggml_tensor*>, std::vector<struct ggml_tensor*>> pre_attention(GGMLRunnerContext* ctx,
                                                                                                struct ggml_tensor* x,
                                                                                                struct ggml_tensor* c) {
        // x: [N, n_token, hidden_size]
        // c: [N, hidden_size]
        auto norm1              = std::dynamic_pointer_cast<LayerNorm>(blocks["norm1"]);
        auto attn               = std::dynamic_pointer_cast<SelfAttention>(blocks["attn"]);
        auto adaLN_modulation_1 = std::dynamic_pointer_cast<Linear>(blocks["adaLN_modulation.1"]);

        int n_mods = 6;
        if (pre_only) {
            n_mods = 2;
        }
        auto m     = adaLN_modulation_1->forward(ctx, ggml_silu(ctx->ggml_ctx, c));  // [N, n_mods * hidden_size]
        auto m_vec = ggml_ext_chunk(ctx->ggml_ctx, m, n_mods, 0);

        auto shift_msa = m_vec[0];  // [N, hidden_size]
        auto scale_msa = m_vec[1];  // [N, hidden_size]
        if (!pre_only) {
            auto gate_msa  = m_vec[2];  // [N, hidden_size]
            auto shift_mlp = m_vec[3];  // [N, hidden_size]
            auto scale_mlp = m_vec[4];  // [N, hidden_size]
            auto gate_mlp  = m_vec[5];  // [N, hidden_size]

            auto attn_in = modulate(ctx->ggml_ctx, norm1->forward(ctx, x), shift_msa, scale_msa);

            auto qkv = attn->pre_attention(ctx, attn_in);

            return {qkv, {x, gate_msa, shift_mlp, scale_mlp, gate_mlp}};
        } else {
            auto attn_in = modulate(ctx->ggml_ctx, norm1->forward(ctx, x), shift_msa, scale_msa);
            auto qkv     = attn->pre_attention(ctx, attn_in);

            return {qkv, {nullptr, nullptr, nullptr, nullptr, nullptr}};
        }
    }

    struct ggml_tensor* post_attention_x(GGMLRunnerContext* ctx,
                                         struct ggml_tensor* attn_out,
                                         struct ggml_tensor* attn2_out,
                                         struct ggml_tensor* x,
                                         struct ggml_tensor* gate_msa,
                                         struct ggml_tensor* shift_mlp,
                                         struct ggml_tensor* scale_mlp,
                                         struct ggml_tensor* gate_mlp,
                                         struct ggml_tensor* gate_msa2) {
        // attn_out: [N, n_token, hidden_size]
        // x: [N, n_token, hidden_size]
        // gate_msa: [N, hidden_size]
        // shift_mlp: [N, hidden_size]
        // scale_mlp: [N, hidden_size]
        // gate_mlp: [N, hidden_size]
        // return: [N, n_token, hidden_size]
        GGML_ASSERT(!pre_only);

        auto attn  = std::dynamic_pointer_cast<SelfAttention>(blocks["attn"]);
        auto attn2 = std::dynamic_pointer_cast<SelfAttention>(blocks["attn2"]);
        auto norm2 = std::dynamic_pointer_cast<LayerNorm>(blocks["norm2"]);
        auto mlp   = std::dynamic_pointer_cast<Mlp>(blocks["mlp"]);

        gate_msa  = ggml_reshape_3d(ctx->ggml_ctx, gate_msa, gate_msa->ne[0], 1, gate_msa->ne[1]);     // [N, 1, hidden_size]
        gate_mlp  = ggml_reshape_3d(ctx->ggml_ctx, gate_mlp, gate_mlp->ne[0], 1, gate_mlp->ne[1]);     // [N, 1, hidden_size]
        gate_msa2 = ggml_reshape_3d(ctx->ggml_ctx, gate_msa2, gate_msa2->ne[0], 1, gate_msa2->ne[1]);  // [N, 1, hidden_size]

        attn_out  = attn->post_attention(ctx, attn_out);
        attn2_out = attn2->post_attention(ctx, attn2_out);

        x            = ggml_add(ctx->ggml_ctx, x, ggml_mul(ctx->ggml_ctx, attn_out, gate_msa));
        x            = ggml_add(ctx->ggml_ctx, x, ggml_mul(ctx->ggml_ctx, attn2_out, gate_msa2));
        auto mlp_out = mlp->forward(ctx, modulate(ctx->ggml_ctx, norm2->forward(ctx, x), shift_mlp, scale_mlp));
        x            = ggml_add(ctx->ggml_ctx, x, ggml_mul(ctx->ggml_ctx, mlp_out, gate_mlp));

        return x;
    }

    struct ggml_tensor* post_attention(GGMLRunnerContext* ctx,
                                       struct ggml_tensor* attn_out,
                                       struct ggml_tensor* x,
                                       struct ggml_tensor* gate_msa,
                                       struct ggml_tensor* shift_mlp,
                                       struct ggml_tensor* scale_mlp,
                                       struct ggml_tensor* gate_mlp) {
        // attn_out: [N, n_token, hidden_size]
        // x: [N, n_token, hidden_size]
        // gate_msa: [N, hidden_size]
        // shift_mlp: [N, hidden_size]
        // scale_mlp: [N, hidden_size]
        // gate_mlp: [N, hidden_size]
        // return: [N, n_token, hidden_size]
        GGML_ASSERT(!pre_only);

        auto attn  = std::dynamic_pointer_cast<SelfAttention>(blocks["attn"]);
        auto norm2 = std::dynamic_pointer_cast<LayerNorm>(blocks["norm2"]);
        auto mlp   = std::dynamic_pointer_cast<Mlp>(blocks["mlp"]);

        gate_msa = ggml_reshape_3d(ctx->ggml_ctx, gate_msa, gate_msa->ne[0], 1, gate_msa->ne[1]);  // [N, 1, hidden_size]
        gate_mlp = ggml_reshape_3d(ctx->ggml_ctx, gate_mlp, gate_mlp->ne[0], 1, gate_mlp->ne[1]);  // [N, 1, hidden_size]

        attn_out = attn->post_attention(ctx, attn_out);

        x            = ggml_add(ctx->ggml_ctx, x, ggml_mul(ctx->ggml_ctx, attn_out, gate_msa));
        auto mlp_out = mlp->forward(ctx, modulate(ctx->ggml_ctx, norm2->forward(ctx, x), shift_mlp, scale_mlp));
        x            = ggml_add(ctx->ggml_ctx, x, ggml_mul(ctx->ggml_ctx, mlp_out, gate_mlp));

        return x;
    }

    struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                struct ggml_tensor* x,
                                struct ggml_tensor* c) {
        // x: [N, n_token, hidden_size]
        // c: [N, hidden_size]
        // return: [N, n_token, hidden_size]

        auto attn = std::dynamic_pointer_cast<SelfAttention>(blocks["attn"]);
        if (self_attn) {
            auto qkv_intermediates = pre_attention_x(ctx, x, c);
            // auto qkv               = qkv_intermediates.first;
            // auto intermediates     = qkv_intermediates.second;
            // no longer a pair, but a tuple
            auto qkv           = std::get<0>(qkv_intermediates);
            auto qkv2          = std::get<1>(qkv_intermediates);
            auto intermediates = std::get<2>(qkv_intermediates);

            auto attn_out  = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, qkv[0], qkv[1], qkv[2], num_heads, nullptr, false, ctx->flash_attn_enabled);     // [N, n_token, dim]
            auto attn2_out = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, qkv2[0], qkv2[1], qkv2[2], num_heads, nullptr, false, ctx->flash_attn_enabled);  // [N, n_token, dim]
            x              = post_attention_x(ctx,
                                              attn_out,
                                              attn2_out,
                                              intermediates[0],
                                              intermediates[1],
                                              intermediates[2],
                                              intermediates[3],
                                              intermediates[4],
                                              intermediates[5]);
            return x;  // [N, n_token, dim]
        } else {
            auto qkv_intermediates = pre_attention(ctx, x, c);
            auto qkv               = qkv_intermediates.first;
            auto intermediates     = qkv_intermediates.second;

            auto attn_out = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, qkv[0], qkv[1], qkv[2], num_heads, nullptr, false, ctx->flash_attn_enabled);  // [N, n_token, dim]
            x             = post_attention(ctx,
                                           attn_out,
                                           intermediates[0],
                                           intermediates[1],
                                           intermediates[2],
                                           intermediates[3],
                                           intermediates[4]);
            return x;  // [N, n_token, dim]
        }
    }
};

__STATIC_INLINE__ std::pair<struct ggml_tensor*, struct ggml_tensor*>
block_mixing(GGMLRunnerContext* ctx,
             struct ggml_tensor* context,
             struct ggml_tensor* x,
             struct ggml_tensor* c,
             std::shared_ptr<DismantledBlock> context_block,
             std::shared_ptr<DismantledBlock> x_block) {
    // context: [N, n_context, hidden_size]
    // x: [N, n_token, hidden_size]
    // c: [N, hidden_size]
    auto context_qkv_intermediates = context_block->pre_attention(ctx, context, c);
    auto context_qkv               = context_qkv_intermediates.first;
    auto context_intermediates     = context_qkv_intermediates.second;

    std::vector<ggml_tensor*> x_qkv, x_qkv2, x_intermediates;

    if (x_block->self_attn) {
        auto x_qkv_intermediates = x_block->pre_attention_x(ctx, x, c);
        x_qkv                    = std::get<0>(x_qkv_intermediates);
        x_qkv2                   = std::get<1>(x_qkv_intermediates);
        x_intermediates          = std::get<2>(x_qkv_intermediates);
    } else {
        auto x_qkv_intermediates = x_block->pre_attention(ctx, x, c);
        x_qkv                    = x_qkv_intermediates.first;
        x_intermediates          = x_qkv_intermediates.second;
    }
    std::vector<struct ggml_tensor*> qkv;
    for (int i = 0; i < 3; i++) {
        qkv.push_back(ggml_concat(ctx->ggml_ctx, context_qkv[i], x_qkv[i], 1));
    }

    auto attn = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, qkv[0], qkv[1], qkv[2], x_block->num_heads, nullptr, false, ctx->flash_attn_enabled);  // [N, n_context + n_token, hidden_size]

    auto context_attn = ggml_view_3d(ctx->ggml_ctx,
                                     attn,
                                     attn->ne[0],
                                     context->ne[1],
                                     attn->ne[2],
                                     attn->nb[1],
                                     attn->nb[2],
                                     0);  // [N, n_context, hidden_size]
    auto x_attn       = ggml_view_3d(ctx->ggml_ctx,
                                     attn,
                                     attn->ne[0],
                                     x->ne[1],
                                     attn->ne[2],
                                     attn->nb[1],
                                     attn->nb[2],
                                     context->ne[1] * attn->nb[1]);  // [N, n_token, hidden_size]

    if (!context_block->pre_only) {
        context = context_block->post_attention(ctx,
                                                context_attn,
                                                context_intermediates[0],
                                                context_intermediates[1],
                                                context_intermediates[2],
                                                context_intermediates[3],
                                                context_intermediates[4]);
    } else {
        context = nullptr;
    }

    if (x_block->self_attn) {
        auto attn2 = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, x_qkv2[0], x_qkv2[1], x_qkv2[2], x_block->num_heads, nullptr, false, ctx->flash_attn_enabled);  // [N, n_token, hidden_size]

        x = x_block->post_attention_x(ctx,
                                      x_attn,
                                      attn2,
                                      x_intermediates[0],
                                      x_intermediates[1],
                                      x_intermediates[2],
                                      x_intermediates[3],
                                      x_intermediates[4],
                                      x_intermediates[5]);
    } else {
        x = x_block->post_attention(ctx,
                                    x_attn,
                                    x_intermediates[0],
                                    x_intermediates[1],
                                    x_intermediates[2],
                                    x_intermediates[3],
                                    x_intermediates[4]);
    }

    return {context, x};
}

struct JointBlock : public GGMLBlock {
public:
    JointBlock(int64_t hidden_size,
               int64_t num_heads,
               float mlp_ratio     = 4.0,
               std::string qk_norm = "",
               bool qkv_bias       = false,
               bool pre_only       = false,
               bool self_attn_x    = false) {
        blocks["context_block"] = std::shared_ptr<GGMLBlock>(new DismantledBlock(hidden_size, num_heads, mlp_ratio, qk_norm, qkv_bias, pre_only, false));
        blocks["x_block"]       = std::shared_ptr<GGMLBlock>(new DismantledBlock(hidden_size, num_heads, mlp_ratio, qk_norm, qkv_bias, false, self_attn_x));
    }

    std::pair<struct ggml_tensor*, struct ggml_tensor*> forward(GGMLRunnerContext* ctx,
                                                                struct ggml_tensor* context,
                                                                struct ggml_tensor* x,
                                                                struct ggml_tensor* c) {
        auto context_block = std::dynamic_pointer_cast<DismantledBlock>(blocks["context_block"]);
        auto x_block       = std::dynamic_pointer_cast<DismantledBlock>(blocks["x_block"]);

        return block_mixing(ctx, context, x, c, context_block, x_block);
    }
};

struct FinalLayer : public GGMLBlock {
    // The final layer of DiT.
public:
    FinalLayer(int64_t hidden_size,
               int64_t patch_size,
               int64_t out_channels) {
        // total_out_channels is always None
        blocks["norm_final"]         = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size, 1e-06f, false));
        blocks["linear"]             = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, patch_size * patch_size * out_channels, true, true));
        blocks["adaLN_modulation.1"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, 2 * hidden_size));
    }

    struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                struct ggml_tensor* x,
                                struct ggml_tensor* c) {
        // x: [N, n_token, hidden_size]
        // c: [N, hidden_size]
        // return: [N, n_token, patch_size * patch_size * out_channels]
        auto norm_final         = std::dynamic_pointer_cast<LayerNorm>(blocks["norm_final"]);
        auto linear             = std::dynamic_pointer_cast<Linear>(blocks["linear"]);
        auto adaLN_modulation_1 = std::dynamic_pointer_cast<Linear>(blocks["adaLN_modulation.1"]);

        auto m     = adaLN_modulation_1->forward(ctx, ggml_silu(ctx->ggml_ctx, c));  // [N, 2 * hidden_size]
        auto m_vec = ggml_ext_chunk(ctx->ggml_ctx, m, 2, 0);
        auto shift = m_vec[0];  // [N, hidden_size]
        auto scale = m_vec[1];  // [N, hidden_size]

        x = modulate(ctx->ggml_ctx, norm_final->forward(ctx, x), shift, scale);
        x = linear->forward(ctx, x);

        return x;
    }
};

struct MMDiT : public GGMLBlock {
    // Diffusion model with a Transformer backbone.
protected:
    int64_t input_size               = -1;
    int patch_size                   = 2;
    int64_t in_channels              = 16;
    int64_t d_self                   = -1;  // >=0 for MMdiT-X
    int64_t depth                    = 24;
    float mlp_ratio                  = 4.0f;
    int64_t adm_in_channels          = 2048;
    int64_t out_channels             = 16;
    int64_t pos_embed_max_size       = 192;
    int64_t num_patchs               = 36864;  // 192 * 192
    int64_t context_size             = 4096;
    int64_t context_embedder_out_dim = 1536;
    int64_t hidden_size;
    std::string qk_norm;

    void init_params(struct ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, std::string prefix = "") override {
        enum ggml_type wtype = GGML_TYPE_F32;
        params["pos_embed"]  = ggml_new_tensor_3d(ctx, wtype, hidden_size, num_patchs, 1);
    }

public:
    MMDiT(const String2TensorStorage& tensor_storage_map = {}) {
        // input_size is always None
        // learn_sigma is always False
        // register_length is alwalys 0
        // rmsnorm is alwalys False
        // scale_mod_only is alwalys False
        // swiglu is alwalys False
        // qkv_bias is always True
        // context_processor_layers is always None
        // pos_embed_scaling_factor is not used
        // pos_embed_offset is not used
        // context_embedder_config is always {'target': 'torch.nn.Linear', 'params': {'in_features': 4096, 'out_features': 1536}}

        for (auto pair : tensor_storage_map) {
            std::string tensor_name = pair.first;
            if (tensor_name.find("model.diffusion_model.") == std::string::npos)
                continue;
            size_t jb = tensor_name.find("joint_blocks.");
            if (jb != std::string::npos) {
                tensor_name     = tensor_name.substr(jb);  // remove prefix
                int block_depth = atoi(tensor_name.substr(13, tensor_name.find(".", 13)).c_str());
                if (block_depth + 1 > depth) {
                    depth = block_depth + 1;
                }
                if (tensor_name.find("attn.ln") != std::string::npos) {
                    if (tensor_name.find(".bias") != std::string::npos) {
                        qk_norm = "ln";
                    } else {
                        qk_norm = "rms";
                    }
                }
                if (tensor_name.find("attn2") != std::string::npos) {
                    if (block_depth > d_self) {
                        d_self = block_depth;
                    }
                }
            }
        }

        if (d_self >= 0) {
            pos_embed_max_size *= 2;
            num_patchs *= 4;
        }

        LOG_INFO("MMDiT layers: %d (including %d MMDiT-x layers)", depth, d_self + 1);

        int64_t default_out_channels = in_channels;
        hidden_size                  = 64 * depth;
        context_embedder_out_dim     = 64 * depth;
        int64_t num_heads            = depth;

        blocks["x_embedder"] = std::shared_ptr<GGMLBlock>(new PatchEmbed(input_size, patch_size, in_channels, hidden_size, true));
        blocks["t_embedder"] = std::shared_ptr<GGMLBlock>(new TimestepEmbedder(hidden_size));

        if (adm_in_channels != -1) {
            blocks["y_embedder"] = std::shared_ptr<GGMLBlock>(new VectorEmbedder(adm_in_channels, hidden_size));
        }

        blocks["context_embedder"] = std::shared_ptr<GGMLBlock>(new Linear(4096, context_embedder_out_dim, true, true));

        for (int i = 0; i < depth; i++) {
            blocks["joint_blocks." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new JointBlock(hidden_size,
                                                                                                    num_heads,
                                                                                                    mlp_ratio,
                                                                                                    qk_norm,
                                                                                                    true,
                                                                                                    i == depth - 1,
                                                                                                    i <= d_self));
        }

        blocks["final_layer"] = std::shared_ptr<GGMLBlock>(new FinalLayer(hidden_size, patch_size, out_channels));
    }

    struct ggml_tensor*
    cropped_pos_embed(struct ggml_context* ctx,
                      int64_t h,
                      int64_t w) {
        auto pos_embed = params["pos_embed"];

        h = (h + 1) / patch_size;
        w = (w + 1) / patch_size;

        GGML_ASSERT(h <= pos_embed_max_size && h > 0);
        GGML_ASSERT(w <= pos_embed_max_size && w > 0);

        int64_t top  = (pos_embed_max_size - h) / 2;
        int64_t left = (pos_embed_max_size - w) / 2;

        auto spatial_pos_embed = ggml_reshape_3d(ctx, pos_embed, hidden_size, pos_embed_max_size, pos_embed_max_size);

        // spatial_pos_embed = spatial_pos_embed[:, top : top + h, left : left + w, :]
        spatial_pos_embed = ggml_view_3d(ctx,
                                         spatial_pos_embed,
                                         hidden_size,
                                         pos_embed_max_size,
                                         h,
                                         spatial_pos_embed->nb[1],
                                         spatial_pos_embed->nb[2],
                                         spatial_pos_embed->nb[2] * top);                      // [h, pos_embed_max_size, hidden_size]
        spatial_pos_embed = ggml_cont(ctx, ggml_permute(ctx, spatial_pos_embed, 0, 2, 1, 3));  // [pos_embed_max_size, h, hidden_size]
        spatial_pos_embed = ggml_view_3d(ctx,
                                         spatial_pos_embed,
                                         hidden_size,
                                         h,
                                         w,
                                         spatial_pos_embed->nb[1],
                                         spatial_pos_embed->nb[2],
                                         spatial_pos_embed->nb[2] * left);                     // [w, h, hidden_size]
        spatial_pos_embed = ggml_cont(ctx, ggml_permute(ctx, spatial_pos_embed, 0, 2, 1, 3));  // [h, w, hidden_size]
        spatial_pos_embed = ggml_reshape_3d(ctx, spatial_pos_embed, hidden_size, h * w, 1);    // [1, h*w, hidden_size]
        return spatial_pos_embed;
    }

    // ============== Staged Forward Methods for True Per-Layer Streaming ==============

    /**
     * Input stage result structure
     */
    struct StreamingInputResult {
        ggml_tensor* x;        // [N, H*W, hidden_size]
        ggml_tensor* context;  // [N, L, hidden_size]
        ggml_tensor* c_mod;    // [N, hidden_size]
    };

    /**
     * Input stage: compute x_embed, t_embed, y_embed, context_embed
     * Returns: {x, context, c_mod}
     */
    StreamingInputResult forward_input_stage(GGMLRunnerContext* ctx,
                                              struct ggml_tensor* x,
                                              struct ggml_tensor* t,
                                              struct ggml_tensor* y,
                                              struct ggml_tensor* context,
                                              int64_t H, int64_t W) {
        auto x_embedder = std::dynamic_pointer_cast<PatchEmbed>(blocks["x_embedder"]);
        auto t_embedder = std::dynamic_pointer_cast<TimestepEmbedder>(blocks["t_embedder"]);

        // Patch embed + pos embed
        auto patch_embed = x_embedder->forward(ctx, x);              // [N, H*W, hidden_size]
        auto pos_embed_out = cropped_pos_embed(ctx->ggml_ctx, H, W); // [1, H*W, hidden_size]
        x = ggml_add(ctx->ggml_ctx, patch_embed, pos_embed_out);     // [N, H*W, hidden_size]

        // Timestep embedding
        auto c = t_embedder->forward(ctx, t);  // [N, hidden_size]

        // Y embedding (if present)
        if (y != nullptr && adm_in_channels != -1) {
            auto y_embedder = std::dynamic_pointer_cast<VectorEmbedder>(blocks["y_embedder"]);
            y = y_embedder->forward(ctx, y);   // [N, hidden_size]
            c = ggml_add(ctx->ggml_ctx, c, y);
        }

        // Context embedding
        if (context != nullptr) {
            auto context_embedder = std::dynamic_pointer_cast<Linear>(blocks["context_embedder"]);
            context = context_embedder->forward(ctx, context);  // [N, L, hidden_size]
        }

        return {x, context, c};
    }

    /**
     * Execute one joint_block
     * Returns: {context, x}
     */
    std::pair<ggml_tensor*, ggml_tensor*> forward_joint_block(GGMLRunnerContext* ctx,
                                                               int block_idx,
                                                               struct ggml_tensor* context,
                                                               struct ggml_tensor* x,
                                                               struct ggml_tensor* c_mod) {
        auto block = std::dynamic_pointer_cast<JointBlock>(blocks["joint_blocks." + std::to_string(block_idx)]);
        return block->forward(ctx, context, x, c_mod);
    }

    /**
     * Output stage: apply final_layer
     * Returns: final output tensor (before unpatchify)
     */
    ggml_tensor* forward_output_stage(GGMLRunnerContext* ctx,
                                       struct ggml_tensor* x,
                                       struct ggml_tensor* c_mod) {
        auto final_layer = std::dynamic_pointer_cast<FinalLayer>(blocks["final_layer"]);
        return final_layer->forward(ctx, x, c_mod);  // (N, H*W, patch_size ** 2 * out_channels)
    }

    int get_depth() const { return depth; }
    int get_patch_size() const { return patch_size; }

    struct ggml_tensor* forward_core_with_concat(GGMLRunnerContext* ctx,
                                                 struct ggml_tensor* x,
                                                 struct ggml_tensor* c_mod,
                                                 struct ggml_tensor* context,
                                                 std::vector<int> skip_layers = std::vector<int>()) {
        // x: [N, H*W, hidden_size]
        // context: [N, n_context, d_context]
        // c: [N, hidden_size]
        // return: [N, N*W, patch_size * patch_size * out_channels]
        auto final_layer = std::dynamic_pointer_cast<FinalLayer>(blocks["final_layer"]);

        for (int i = 0; i < depth; i++) {
            // skip iteration if i is in skip_layers
            if (skip_layers.size() > 0 && std::find(skip_layers.begin(), skip_layers.end(), i) != skip_layers.end()) {
                continue;
            }

            auto block = std::dynamic_pointer_cast<JointBlock>(blocks["joint_blocks." + std::to_string(i)]);

            auto context_x = block->forward(ctx, context, x, c_mod);
            context        = context_x.first;
            x              = context_x.second;
        }

        x = final_layer->forward(ctx, x, c_mod);  // (N, T, patch_size ** 2 * out_channels)

        return x;
    }

    struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                struct ggml_tensor* x,
                                struct ggml_tensor* t,
                                struct ggml_tensor* y        = nullptr,
                                struct ggml_tensor* context  = nullptr,
                                std::vector<int> skip_layers = std::vector<int>()) {
        // Forward pass of DiT.
        // x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        // t: (N,) tensor of diffusion timesteps
        // y: (N, adm_in_channels) tensor of class labels
        // context: (N, L, D)
        // return: (N, C, H, W)
        auto x_embedder = std::dynamic_pointer_cast<PatchEmbed>(blocks["x_embedder"]);
        auto t_embedder = std::dynamic_pointer_cast<TimestepEmbedder>(blocks["t_embedder"]);

        int64_t W = x->ne[0];
        int64_t H = x->ne[1];

        auto patch_embed = x_embedder->forward(ctx, x);                      // [N, H*W, hidden_size]
        auto pos_embed   = cropped_pos_embed(ctx->ggml_ctx, H, W);           // [1, H*W, hidden_size]
        x                = ggml_add(ctx->ggml_ctx, patch_embed, pos_embed);  // [N, H*W, hidden_size]

        auto c = t_embedder->forward(ctx, t);  // [N, hidden_size]
        if (y != nullptr && adm_in_channels != -1) {
            auto y_embedder = std::dynamic_pointer_cast<VectorEmbedder>(blocks["y_embedder"]);

            y = y_embedder->forward(ctx, y);  // [N, hidden_size]
            c = ggml_add(ctx->ggml_ctx, c, y);
        }

        if (context != nullptr) {
            auto context_embedder = std::dynamic_pointer_cast<Linear>(blocks["context_embedder"]);

            context = context_embedder->forward(ctx, context);  // [N, L, D] aka [N, L, 1536]
        }

        x = forward_core_with_concat(ctx, x, c, context, skip_layers);  // (N, H*W, patch_size ** 2 * out_channels)

        x = DiT::unpatchify_and_crop(ctx->ggml_ctx, x, H, W, patch_size, patch_size, /*patch_last*/ false);  // [N, C, H, W]

        return x;
    }
};
struct MMDiTRunner : public GGMLRunner {
    MMDiT mmdit;

    // Layer streaming support
    std::unique_ptr<LayerStreaming::LayerExecutionEngine> streaming_engine_;
    bool streaming_enabled_ = false;

    MMDiTRunner(ggml_backend_t backend,
                bool offload_params_to_cpu,
                const String2TensorStorage& tensor_storage_map = {},
                const std::string prefix                       = "")
        : GGMLRunner(backend, offload_params_to_cpu), mmdit(tensor_storage_map) {
        mmdit.init(params_ctx, tensor_storage_map, prefix);
    }

    std::string get_desc() override {
        return "mmdit";
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        mmdit.get_param_tensors(tensors, prefix);
    }

    // ============== Layer Streaming Support ==============

    /**
     * Enable layer streaming for MMDiT
     * MMDiT has no skip connections, so each joint_block is independent.
     * Uses coarse-stage streaming: load all weights before graph execution.
     */
    void enable_layer_streaming(const LayerStreaming::StreamingConfig& config = {}) {
        if (!params_backend || !runtime_backend) {
            LOG_WARN("MMDiTRunner: Cannot enable streaming without both CPU and GPU backends");
            return;
        }

        streaming_engine_ = std::make_unique<LayerStreaming::LayerExecutionEngine>(
            runtime_backend, params_backend);

        LayerStreaming::StreamingConfig cfg = config;
        cfg.enabled = true;
        // MMDiT has no skip connections, so we only need to keep the current layer
        cfg.keep_layers_behind = 0;
        streaming_engine_->set_config(cfg);

        // Register tensors with MMDiT layer pattern
        std::map<std::string, ggml_tensor*> tensor_map;
        mmdit.get_param_tensors(tensor_map, "model.diffusion_model");
        streaming_engine_->register_model_layers_from_map(tensor_map, LayerStreaming::mmdit_layer_pattern);

        streaming_enabled_ = true;
        LOG_INFO("MMDiTRunner: Layer streaming enabled (%zu layers)",
                 streaming_engine_->get_registry().get_layer_count());
    }

    void disable_layer_streaming() {
        streaming_enabled_ = false;
        streaming_engine_.reset();
        LOG_INFO("MMDiTRunner: Layer streaming disabled");
    }

    bool is_streaming_enabled() const {
        return streaming_enabled_ && streaming_engine_ != nullptr;
    }

    void offload_streaming_layers() {
        if (streaming_engine_) {
            auto& registry = streaming_engine_->get_registry();
            auto layers = registry.get_layer_names_sorted();
            size_t offloaded = 0;
            for (const auto& layer : layers) {
                if (registry.is_layer_on_gpu(layer)) {
                    registry.move_layer_to_cpu(layer);
                    offloaded++;
                }
            }
            if (offloaded > 0) {
                LOG_INFO("MMDiTRunner: Offloaded %zu streaming layers to CPU", offloaded);
            }
        }
    }

    /**
     * Streaming compute for MMDiT
     * Since MMDiT has no skip connections, we load all joint_blocks before execution.
     */
    bool compute_streaming(int n_threads,
                           struct ggml_tensor* x,
                           struct ggml_tensor* timesteps,
                           struct ggml_tensor* context,
                           struct ggml_tensor* y,
                           struct ggml_tensor** output     = nullptr,
                           struct ggml_context* output_ctx = nullptr,
                           std::vector<int> skip_layers    = std::vector<int>()) {
        if (!streaming_engine_) {
            LOG_ERROR("MMDiTRunner: Streaming not enabled");
            return false;
        }

        int64_t t0 = ggml_time_ms();

        auto& registry = streaming_engine_->get_registry();
        auto& budget = streaming_engine_->get_budget();

        // Calculate total model size
        size_t total_model_size = 0;
        auto all_layers = registry.get_layer_names_sorted();
        for (const auto& layer_name : all_layers) {
            total_model_size += registry.get_layer_size(layer_name);
        }

        // Get available VRAM
        size_t available_vram = budget.get_available_vram();

        // Check how much is already on GPU (for CFG - multiple calls per step)
        size_t already_on_gpu = 0;
        for (const auto& layer_name : all_layers) {
            if (registry.is_layer_on_gpu(layer_name)) {
                already_on_gpu += registry.get_layer_size(layer_name);
            }
        }

        // Effective model size = what still needs to be loaded
        size_t remaining_to_load = (total_model_size > already_on_gpu) ? (total_model_size - already_on_gpu) : 0;

        LOG_DEBUG("MMDiTRunner: Model size = %.2f GB, On GPU = %.2f GB, Remaining = %.2f GB, Available VRAM = %.2f GB",
                  total_model_size / (1024.0 * 1024.0 * 1024.0),
                  already_on_gpu / (1024.0 * 1024.0 * 1024.0),
                  remaining_to_load / (1024.0 * 1024.0 * 1024.0),
                  available_vram / (1024.0 * 1024.0 * 1024.0));

        // Check if model fits in VRAM (accounting for what's already loaded)
        if (remaining_to_load <= available_vram) {
            // Model fits - load all and compute
            LOG_INFO("MMDiTRunner: Model fits in VRAM, using coarse-stage streaming");
            for (const auto& layer_name : all_layers) {
                if (!registry.is_layer_on_gpu(layer_name)) {
                    if (!budget.ensure_vram_for_layer(layer_name, 0)) {
                        LOG_WARN("MMDiTRunner: Could not ensure VRAM for layer %s", layer_name.c_str());
                    }
                    registry.move_layer_to_gpu(layer_name);
                }
            }
            // Execute full graph
            bool result = compute(n_threads, x, timesteps, context, y, output, output_ctx, skip_layers,
                                  true /* skip_param_offload */);

            int64_t t1 = ggml_time_ms();
            LOG_INFO("MMDiTRunner: Coarse-stage streaming completed in %.2fs", (t1 - t0) / 1000.0);

            // Free compute buffer so next iteration can use different graph if needed
            free_compute_buffer();
            return result;
        }

        // Model doesn't fit - use TRUE per-layer streaming
        LOG_INFO("MMDiTRunner: Remaining to load (%.2f GB) exceeds available VRAM (%.2f GB), using TRUE per-layer streaming",
                 remaining_to_load / (1024.0 * 1024.0 * 1024.0),
                 available_vram / (1024.0 * 1024.0 * 1024.0));

        return compute_streaming_true(n_threads, x, timesteps, context, y, output, output_ctx, skip_layers);
    }

    /**
     * TRUE per-layer streaming for MMDiT
     * Executes each joint_block as a separate mini-graph to minimize VRAM usage
     */
    bool compute_streaming_true(int n_threads,
                                 struct ggml_tensor* x,
                                 struct ggml_tensor* timesteps,
                                 struct ggml_tensor* context,
                                 struct ggml_tensor* y,
                                 struct ggml_tensor** output     = nullptr,
                                 struct ggml_context* output_ctx = nullptr,
                                 std::vector<int> skip_layers    = std::vector<int>()) {
        auto& registry = streaming_engine_->get_registry();
        int64_t t_start = ggml_time_ms();

        const int num_blocks = mmdit.get_depth();
        const int patch_size = mmdit.get_patch_size();
        const int64_t W = x->ne[0];
        const int64_t H = x->ne[1];

        LOG_INFO("MMDiTRunner: TRUE per-layer streaming - %d joint_blocks", num_blocks);

        // Load global layers
        LOG_DEBUG("MMDiTRunner: Loading global layers");
        if (!registry.move_layer_to_gpu("_global")) {
            LOG_ERROR("MMDiTRunner: Failed to load _global to GPU");
            return false;
        }

        // Persistent storage for intermediate tensors
        std::vector<float> persistent_x;
        std::vector<float> persistent_context;
        std::vector<float> persistent_c_mod;
        int64_t x_ne[4], context_ne[4], c_mod_ne[4];

        // ============ STAGE 1: Input projections ============
        LOG_DEBUG("MMDiTRunner: Executing input stage");
        {
            ggml_tensor* x_output = nullptr;
            ggml_tensor* context_output = nullptr;
            ggml_tensor* c_mod_output = nullptr;

            auto get_input_graph = [&]() -> struct ggml_cgraph* {
                struct ggml_cgraph* gf = new_graph_custom(MMDIT_GRAPH_SIZE / 4);
                auto runner_ctx = get_context();

                ggml_tensor* x_backend = to_backend(x);
                ggml_tensor* timesteps_backend = to_backend(timesteps);
                ggml_tensor* y_backend = y ? to_backend(y) : nullptr;
                ggml_tensor* context_backend = context ? to_backend(context) : nullptr;

                auto result = mmdit.forward_input_stage(&runner_ctx, x_backend, timesteps_backend,
                                                         y_backend, context_backend, H, W);

                x_output = result.x;
                context_output = result.context;
                c_mod_output = result.c_mod;

                ggml_build_forward_expand(gf, x_output);
                if (context_output) ggml_build_forward_expand(gf, context_output);
                ggml_build_forward_expand(gf, c_mod_output);

                return gf;
            };

            // Don't free compute buffer immediately - we need to read outputs first
            if (!GGMLRunner::compute(get_input_graph, n_threads, false, nullptr, nullptr, true)) {
                LOG_ERROR("MMDiTRunner: Input stage failed");
                return false;
            }

            // Extract to persistent storage
            if (x_output && c_mod_output) {
                size_t x_size = ggml_nelements(x_output);
                size_t c_mod_size = ggml_nelements(c_mod_output);

                persistent_x.resize(x_size);
                persistent_c_mod.resize(c_mod_size);

                ggml_backend_tensor_get(x_output, persistent_x.data(), 0, x_size * sizeof(float));
                ggml_backend_tensor_get(c_mod_output, persistent_c_mod.data(), 0, c_mod_size * sizeof(float));

                for (int i = 0; i < 4; i++) {
                    x_ne[i] = x_output->ne[i];
                    c_mod_ne[i] = c_mod_output->ne[i];
                }

                if (context_output) {
                    size_t context_size = ggml_nelements(context_output);
                    persistent_context.resize(context_size);
                    ggml_backend_tensor_get(context_output, persistent_context.data(), 0, context_size * sizeof(float));
                    for (int i = 0; i < 4; i++) {
                        context_ne[i] = context_output->ne[i];
                    }
                }
            } else {
                LOG_ERROR("MMDiTRunner: Failed to get input stage outputs");
                free_compute_buffer();
                return false;
            }

            // Now safe to free compute buffer
            free_compute_buffer();
        }

        LOG_DEBUG("MMDiTRunner: Input stage done, x=%ldx%ldx%ld", x_ne[0], x_ne[1], x_ne[2]);

        // ============ STAGE 2: Joint blocks (one at a time) ============
        // Start async prefetch for first block
        if (num_blocks > 0 && streaming_engine_) {
            std::string first_block = "joint_blocks.0";
            streaming_engine_->prefetch_layer(first_block);
        }

        for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
            // Check skip_layers
            if (skip_layers.size() > 0 && std::find(skip_layers.begin(), skip_layers.end(), block_idx) != skip_layers.end()) {
                LOG_DEBUG("MMDiTRunner: Skipping joint_block %d", block_idx);
                continue;
            }

            std::string block_name = "joint_blocks." + std::to_string(block_idx);
            int64_t t_block_start = ggml_time_ms();

            // Wait for this block's prefetch to complete (if async prefetch was started)
            if (streaming_engine_) {
                streaming_engine_->wait_for_prefetch(block_name);
            }

            // Load this block's weights (sync load if prefetch didn't happen)
            if (!registry.move_layer_to_gpu(block_name)) {
                LOG_ERROR("MMDiTRunner: Failed to load %s", block_name.c_str());
                return false;
            }

            // Start async prefetch of NEXT block while we compute this one
            if (streaming_engine_ && block_idx + 1 < num_blocks) {
                std::string next_block = "joint_blocks." + std::to_string(block_idx + 1);
                streaming_engine_->prefetch_layer(next_block);
            }

            ggml_tensor* x_out = nullptr;
            ggml_tensor* context_out = nullptr;

            auto get_block_graph = [&]() -> struct ggml_cgraph* {
                struct ggml_cgraph* gf = new_graph_custom(MMDIT_GRAPH_SIZE / 4);

                // Create input tensors from persistent storage
                ggml_tensor* x_in = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, x_ne[0], x_ne[1], x_ne[2], x_ne[3]);
                ggml_tensor* c_mod_in = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, c_mod_ne[0], c_mod_ne[1], c_mod_ne[2], c_mod_ne[3]);

                x_in = to_backend(x_in);
                c_mod_in = to_backend(c_mod_in);

                set_backend_tensor_data(x_in, persistent_x.data());
                set_backend_tensor_data(c_mod_in, persistent_c_mod.data());

                ggml_tensor* context_in = nullptr;
                if (!persistent_context.empty()) {
                    context_in = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, context_ne[0], context_ne[1], context_ne[2], context_ne[3]);
                    context_in = to_backend(context_in);
                    set_backend_tensor_data(context_in, persistent_context.data());
                }

                auto runner_ctx = get_context();
                auto result = mmdit.forward_joint_block(&runner_ctx, block_idx, context_in, x_in, c_mod_in);

                context_out = result.first;
                x_out = result.second;

                if (context_out) ggml_build_forward_expand(gf, context_out);
                ggml_build_forward_expand(gf, x_out);

                return gf;
            };

            // Don't free compute buffer immediately - we need to read outputs first
            if (!GGMLRunner::compute(get_block_graph, n_threads, false, nullptr, nullptr, true)) {
                LOG_ERROR("MMDiTRunner: Joint block %d execution failed", block_idx);
                return false;
            }

            // Extract outputs to persistent storage
            if (x_out) {
                ggml_backend_tensor_get(x_out, persistent_x.data(), 0, persistent_x.size() * sizeof(float));
                for (int i = 0; i < 4; i++) {
                    x_ne[i] = x_out->ne[i];
                }
            }
            if (context_out && !persistent_context.empty()) {
                ggml_backend_tensor_get(context_out, persistent_context.data(), 0, persistent_context.size() * sizeof(float));
                for (int i = 0; i < 4; i++) {
                    context_ne[i] = context_out->ne[i];
                }
            }

            // Now safe to free compute buffer
            free_compute_buffer();

            // Offload this block
            registry.move_layer_to_cpu(block_name);

            LOG_DEBUG("MMDiTRunner: Joint block %d/%d done (%.2fms)",
                      block_idx + 1, num_blocks, (ggml_time_ms() - t_block_start) / 1.0);
        }

        // ============ STAGE 3: Output stage ============
        LOG_DEBUG("MMDiTRunner: Executing output stage");
        {
            auto get_output_graph = [&]() -> struct ggml_cgraph* {
                struct ggml_cgraph* gf = new_graph_custom(MMDIT_GRAPH_SIZE / 4);

                ggml_tensor* x_in = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, x_ne[0], x_ne[1], x_ne[2], x_ne[3]);
                ggml_tensor* c_mod_in = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, c_mod_ne[0], c_mod_ne[1], c_mod_ne[2], c_mod_ne[3]);

                x_in = to_backend(x_in);
                c_mod_in = to_backend(c_mod_in);

                set_backend_tensor_data(x_in, persistent_x.data());
                set_backend_tensor_data(c_mod_in, persistent_c_mod.data());

                auto runner_ctx = get_context();
                auto final_out = mmdit.forward_output_stage(&runner_ctx, x_in, c_mod_in);

                // Unpatchify
                final_out = DiT::unpatchify_and_crop(compute_ctx, final_out, H, W, patch_size, patch_size, /*patch_last*/ false);

                ggml_build_forward_expand(gf, final_out);

                return gf;
            };

            if (!GGMLRunner::compute(get_output_graph, n_threads, true, output, output_ctx, true)) {
                LOG_ERROR("MMDiTRunner: Output stage failed");
                return false;
            }
        }

        int64_t t_end = ggml_time_ms();
        LOG_INFO("MMDiTRunner: TRUE per-layer streaming completed in %.2fs (%d joint_blocks)",
                 (t_end - t_start) / 1000.0, num_blocks);

        return true;
    }

    struct ggml_cgraph* build_graph(struct ggml_tensor* x,
                                    struct ggml_tensor* timesteps,
                                    struct ggml_tensor* context,
                                    struct ggml_tensor* y,
                                    std::vector<int> skip_layers = std::vector<int>()) {
        struct ggml_cgraph* gf = new_graph_custom(MMDIT_GRAPH_SIZE);

        x         = to_backend(x);
        context   = to_backend(context);
        y         = to_backend(y);
        timesteps = to_backend(timesteps);

        auto runner_ctx         = get_context();
        struct ggml_tensor* out = mmdit.forward(&runner_ctx,
                                                x,
                                                timesteps,
                                                y,
                                                context,
                                                skip_layers);

        ggml_build_forward_expand(gf, out);

        return gf;
    }

    bool compute(int n_threads,
                 struct ggml_tensor* x,
                 struct ggml_tensor* timesteps,
                 struct ggml_tensor* context,
                 struct ggml_tensor* y,
                 struct ggml_tensor** output     = nullptr,
                 struct ggml_context* output_ctx = nullptr,
                 std::vector<int> skip_layers    = std::vector<int>(),
                 bool skip_param_offload         = false) {
        // x: [N, in_channels, h, w]
        // timesteps: [N, ]
        // context: [N, max_position, hidden_size]([N, 154, 4096]) or [1, max_position, hidden_size]
        // y: [N, adm_in_channels] or [1, adm_in_channels]
        auto get_graph = [&]() -> struct ggml_cgraph* {
            return build_graph(x, timesteps, context, y, skip_layers);
        };

        return GGMLRunner::compute(get_graph, n_threads, false, output, output_ctx, skip_param_offload);
    }

    void test() {
        struct ggml_init_params params;
        params.mem_size   = static_cast<size_t>(10 * 1024 * 1024);  // 10 MB
        params.mem_buffer = nullptr;
        params.no_alloc   = false;

        struct ggml_context* work_ctx = ggml_init(params);
        GGML_ASSERT(work_ctx != nullptr);

        {
            // cpu f16: pass
            // cpu f32: pass
            // cuda f16: pass
            // cuda f32: pass
            auto x = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, 128, 128, 16, 1);
            std::vector<float> timesteps_vec(1, 999.f);
            auto timesteps = vector_to_ggml_tensor(work_ctx, timesteps_vec);
            ggml_set_f32(x, 0.01f);
            // print_ggml_tensor(x);

            auto context = ggml_new_tensor_3d(work_ctx, GGML_TYPE_F32, 4096, 154, 1);
            ggml_set_f32(context, 0.01f);
            // print_ggml_tensor(context);

            auto y = ggml_new_tensor_2d(work_ctx, GGML_TYPE_F32, 2048, 1);
            ggml_set_f32(y, 0.01f);
            // print_ggml_tensor(y);

            struct ggml_tensor* out = nullptr;

            int64_t t0 = ggml_time_ms();
            compute(8, x, timesteps, context, y, &out, work_ctx);
            int64_t t1 = ggml_time_ms();

            print_ggml_tensor(out);
            LOG_DEBUG("mmdit test done in %lldms", t1 - t0);
        }
    }

    static void load_from_file_and_test(const std::string& file_path) {
        // ggml_backend_t backend    = ggml_backend_cuda_init(0);
        ggml_backend_t backend             = ggml_backend_cpu_init();
        ggml_type model_data_type          = GGML_TYPE_F16;
        std::shared_ptr<MMDiTRunner> mmdit = std::make_shared<MMDiTRunner>(backend, false);
        {
            LOG_INFO("loading from '%s'", file_path.c_str());

            mmdit->alloc_params_buffer();
            std::map<std::string, ggml_tensor*> tensors;
            mmdit->get_param_tensors(tensors, "model.diffusion_model");

            ModelLoader model_loader;
            if (!model_loader.init_from_file_and_convert_name(file_path)) {
                LOG_ERROR("init model loader from file failed: '%s'", file_path.c_str());
                return;
            }

            bool success = model_loader.load_tensors(tensors);

            if (!success) {
                LOG_ERROR("load tensors from model loader failed");
                return;
            }

            LOG_INFO("mmdit model loaded");
        }
        mmdit->test();
    }
};

#endif
