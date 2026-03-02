#ifndef __ACE_HPP__
#define __ACE_HPP__

#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "ggml_extend.hpp"
#include "model.h"

#define ACE_GRAPH_SIZE 81920

namespace ACE {

static inline ggml_tensor* repeat_like(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* like) {
    if (ggml_are_same_shape(x, like)) {
        if (ggml_is_contiguous(x)) {
            return x;
        }
        return ggml_cont(ctx->ggml_ctx, x);
    }
    return ggml_ext_repeat(ctx->ggml_ctx, x, like);
}

static inline ggml_tensor* cont_if_needed(GGMLRunnerContext* ctx, ggml_tensor* x) {
    if (ggml_is_contiguous(x)) {
        return x;
    }
    return ggml_cont(ctx->ggml_ctx, x);
}

static inline ggml_tensor* add_cont(GGMLRunnerContext* ctx, ggml_tensor* a, ggml_tensor* b) {
    return ggml_add(ctx->ggml_ctx, cont_if_needed(ctx, a), cont_if_needed(ctx, b));
}

static inline ggml_tensor* repeat_kv_heads(GGMLRunnerContext* ctx, ggml_tensor* x, int64_t num_kv_heads, int64_t num_heads) {
    if (num_kv_heads == num_heads) {
        return x;
    }
    GGML_ASSERT(num_kv_heads > 0 && num_heads % num_kv_heads == 0);

    int64_t n_rep = num_heads / num_kv_heads;
    int64_t d     = x->ne[0];
    int64_t L     = x->ne[2];
    int64_t B     = x->ne[3];

    auto x3 = ggml_reshape_3d(ctx->ggml_ctx, x, d, num_kv_heads, L * B);
    auto x4 = ggml_reshape_4d(ctx->ggml_ctx, x3, d, 1, num_kv_heads, L * B);
    auto repeat_target = ggml_new_tensor_4d(ctx->ggml_ctx, x->type, d, n_rep, num_kv_heads, L * B);
    x4 = ggml_ext_repeat(ctx->ggml_ctx, x4, repeat_target);
    auto x3r = ggml_reshape_3d(ctx->ggml_ctx, x4, d, num_heads, L * B);
    auto x4r = ggml_reshape_4d(ctx->ggml_ctx, x3r, d, num_heads, L, B);
    return x4r;
}

static inline ggml_tensor* slice_dim1(GGMLRunnerContext* ctx, ggml_tensor* x, int64_t idx) {
    return ggml_ext_slice(ctx->ggml_ctx, x, 1, idx, idx + 1, true);
}

static inline ggml_tensor* swap_dim0_dim1(GGMLRunnerContext* ctx, ggml_tensor* x) {
    return ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 1, 0, 2, 3));
}

struct AceMLP : public UnaryBlock {
public:
    AceMLP(int64_t hidden_size, int64_t intermediate_size) {
        blocks["gate_proj"] = std::make_shared<Linear>(hidden_size, intermediate_size, false);
        blocks["up_proj"]   = std::make_shared<Linear>(hidden_size, intermediate_size, false);
        blocks["down_proj"] = std::make_shared<Linear>(intermediate_size, hidden_size, false);
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        auto gate_proj = std::dynamic_pointer_cast<Linear>(blocks["gate_proj"]);
        auto up_proj   = std::dynamic_pointer_cast<Linear>(blocks["up_proj"]);
        auto down_proj = std::dynamic_pointer_cast<Linear>(blocks["down_proj"]);

        auto gate = gate_proj->forward(ctx, x);
        gate      = ggml_silu_inplace(ctx->ggml_ctx, gate);
        x         = up_proj->forward(ctx, x);
        x         = ggml_mul(ctx->ggml_ctx, x, gate);
        x         = down_proj->forward(ctx, x);
        return x;
    }
};

struct TimestepEmbedding : public GGMLBlock {
    int64_t in_channels;
    int64_t time_embed_dim;
    float time_factor;

    TimestepEmbedding(int64_t in_channels, int64_t time_embed_dim, float time_factor = 1000.f)
        : in_channels(in_channels), time_embed_dim(time_embed_dim), time_factor(time_factor) {
        blocks["linear_1"] = std::make_shared<Linear>(in_channels, time_embed_dim, true);
        blocks["linear_2"] = std::make_shared<Linear>(time_embed_dim, time_embed_dim, true);
        blocks["time_proj"] = std::make_shared<Linear>(time_embed_dim, time_embed_dim * 6, true);
    }

    std::pair<ggml_tensor*, ggml_tensor*> forward(GGMLRunnerContext* ctx, ggml_tensor* t) {
        auto linear_1  = std::dynamic_pointer_cast<Linear>(blocks["linear_1"]);
        auto linear_2  = std::dynamic_pointer_cast<Linear>(blocks["linear_2"]);
        auto time_proj = std::dynamic_pointer_cast<Linear>(blocks["time_proj"]);

        auto t_freq = ggml_ext_timestep_embedding(ctx->ggml_ctx, t, (int)in_channels, 10000, time_factor);
        auto temb   = linear_1->forward(ctx, t_freq);
        temb        = ggml_silu_inplace(ctx->ggml_ctx, temb);
        temb        = linear_2->forward(ctx, temb);

        // ggml_dup_tensor only allocates shape; we need ggml_dup to feed actual temb values into SiLU.
        auto temb_act = ggml_silu_inplace(ctx->ggml_ctx, ggml_dup(ctx->ggml_ctx, temb));
        auto proj     = time_proj->forward(ctx, temb_act);  // [hidden*6, B]
        proj          = ggml_reshape_3d(ctx->ggml_ctx, proj, time_embed_dim, 6, proj->ne[1]);

        return {temb, proj};
    }
};

struct AceStepAttention : public GGMLBlock {
    int64_t hidden_size;
    int64_t num_heads;
    int64_t num_kv_heads;
    int64_t head_dim;
    bool is_cross_attention;

    AceStepAttention(int64_t hidden_size,
                     int64_t num_heads,
                     int64_t num_kv_heads,
                     int64_t head_dim,
                     bool is_cross_attention = false)
        : hidden_size(hidden_size),
          num_heads(num_heads),
          num_kv_heads(num_kv_heads),
          head_dim(head_dim),
          is_cross_attention(is_cross_attention) {
        blocks["q_proj"] = std::make_shared<Linear>(hidden_size, num_heads * head_dim, false);
        blocks["k_proj"] = std::make_shared<Linear>(hidden_size, num_kv_heads * head_dim, false);
        blocks["v_proj"] = std::make_shared<Linear>(hidden_size, num_kv_heads * head_dim, false);
        blocks["o_proj"] = std::make_shared<Linear>(num_heads * head_dim, hidden_size, false);
        blocks["q_norm"] = std::make_shared<RMSNorm>(head_dim, 1e-6f);
        blocks["k_norm"] = std::make_shared<RMSNorm>(head_dim, 1e-6f);
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx,
                         ggml_tensor* hidden_states,
                         ggml_tensor* encoder_hidden_states,
                         ggml_tensor* attention_mask,
                         ggml_tensor* input_pos) {
        int64_t q_len = hidden_states->ne[1];
        int64_t B     = hidden_states->ne[2];

        auto q_proj = std::dynamic_pointer_cast<Linear>(blocks["q_proj"]);
        auto k_proj = std::dynamic_pointer_cast<Linear>(blocks["k_proj"]);
        auto v_proj = std::dynamic_pointer_cast<Linear>(blocks["v_proj"]);
        auto o_proj = std::dynamic_pointer_cast<Linear>(blocks["o_proj"]);
        auto q_norm = std::dynamic_pointer_cast<RMSNorm>(blocks["q_norm"]);
        auto k_norm = std::dynamic_pointer_cast<RMSNorm>(blocks["k_norm"]);

        auto q = q_proj->forward(ctx, hidden_states);  // [Hq, q_len, B]

        ggml_tensor* kv_states = hidden_states;
        if (is_cross_attention && encoder_hidden_states != nullptr) {
            kv_states = encoder_hidden_states;
        }

        int64_t kv_len = kv_states->ne[1];
        auto k         = k_proj->forward(ctx, kv_states);
        auto v         = v_proj->forward(ctx, kv_states);

        q = ggml_reshape_4d(ctx->ggml_ctx, q, head_dim, num_heads, q_len, B);
        k = ggml_reshape_4d(ctx->ggml_ctx, k, head_dim, num_kv_heads, kv_len, B);
        v = ggml_reshape_4d(ctx->ggml_ctx, v, head_dim, num_kv_heads, kv_len, B);

        q = q_norm->forward(ctx, q);
        k = k_norm->forward(ctx, k);

        if (!is_cross_attention && input_pos != nullptr) {
            // Match ACE 1.5 rotary config (base=1e6, max_position_embeddings=32768).
            q = ggml_rope_ext(ctx->ggml_ctx, q, input_pos, nullptr, (int)head_dim, GGML_ROPE_TYPE_NEOX, 32768, 1000000.f, 1.f, 0.f, 1.f, 32.f, 1.f);
            k = ggml_rope_ext(ctx->ggml_ctx, k, input_pos, nullptr, (int)head_dim, GGML_ROPE_TYPE_NEOX, 32768, 1000000.f, 1.f, 0.f, 1.f, 32.f, 1.f);
        }

        int64_t kv_heads_effective = num_kv_heads;
        if (num_kv_heads != num_heads) {
            // Match Comfy/PyTorch: repeat GQA K/V heads explicitly before attention.
            k = repeat_kv_heads(ctx, k, num_kv_heads, num_heads);
            v = repeat_kv_heads(ctx, v, num_kv_heads, num_heads);
            kv_heads_effective = num_heads;
        }

        q = ggml_cont(ctx->ggml_ctx, q);
        k = ggml_cont(ctx->ggml_ctx, k);
        q = ggml_reshape_3d(ctx->ggml_ctx, q, head_dim * num_heads, q_len, B);
        k = ggml_reshape_3d(ctx->ggml_ctx, k, head_dim * kv_heads_effective, kv_len, B);
        v = ggml_cont(ctx->ggml_ctx, v);
        v = ggml_reshape_3d(ctx->ggml_ctx, v, head_dim * kv_heads_effective, kv_len, B);

        const int64_t attn_batch = hidden_states->ne[2] * hidden_states->ne[3];
        const bool use_flash_attn = ctx->flash_attn_enabled && attn_batch == 1 && attention_mask == nullptr;

        auto attn = ggml_ext_attention_ext(ctx->ggml_ctx,
                                           ctx->backend,
                                           q,
                                           k,
                                           v,
                                           num_heads,
                                           attention_mask,
                                           false,
                                           use_flash_attn);
        attn = o_proj->forward(ctx, attn);
        return attn;
    }
};

struct AceStepDiTLayer : public GGMLBlock {
    int64_t hidden_size;
    bool use_sliding;

    AceStepDiTLayer(int64_t hidden_size,
                    int64_t num_heads,
                    int64_t num_kv_heads,
                    int64_t head_dim,
                    int64_t intermediate_size,
                    bool use_sliding)
        : hidden_size(hidden_size),
          use_sliding(use_sliding) {
        blocks["self_attn_norm"]  = std::make_shared<RMSNorm>(hidden_size, 1e-6f);
        blocks["self_attn"]       = std::make_shared<AceStepAttention>(hidden_size, num_heads, num_kv_heads, head_dim, false);
        blocks["cross_attn_norm"] = std::make_shared<RMSNorm>(hidden_size, 1e-6f);
        blocks["cross_attn"]      = std::make_shared<AceStepAttention>(hidden_size, num_heads, num_kv_heads, head_dim, true);
        blocks["mlp_norm"]        = std::make_shared<RMSNorm>(hidden_size, 1e-6f);
        blocks["mlp"]             = std::make_shared<AceMLP>(hidden_size, intermediate_size);
    }

protected:
    void init_params(struct ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
        enum ggml_type wtype   = get_type(prefix + "scale_shift_table", tensor_storage_map, GGML_TYPE_F32);
        params["scale_shift_table"] = ggml_new_tensor_3d(ctx, wtype, hidden_size, 6, 1);
        GGMLBlock::init_params(ctx, tensor_storage_map, prefix);
    }

public:
    ggml_tensor* forward(GGMLRunnerContext* ctx,
                         ggml_tensor* hidden_states,
                         ggml_tensor* temb,
                         ggml_tensor* encoder_hidden_states,
                         ggml_tensor* input_pos,
                         ggml_tensor* attention_mask) {
        auto self_attn_norm  = std::dynamic_pointer_cast<RMSNorm>(blocks["self_attn_norm"]);
        auto self_attn       = std::dynamic_pointer_cast<AceStepAttention>(blocks["self_attn"]);
        auto cross_attn_norm = std::dynamic_pointer_cast<RMSNorm>(blocks["cross_attn_norm"]);
        auto cross_attn      = std::dynamic_pointer_cast<AceStepAttention>(blocks["cross_attn"]);
        auto mlp_norm        = std::dynamic_pointer_cast<RMSNorm>(blocks["mlp_norm"]);
        auto mlp             = std::dynamic_pointer_cast<AceMLP>(blocks["mlp"]);

        auto scale_shift_table = params["scale_shift_table"];
        auto scale_shift        = ggml_ext_repeat(ctx->ggml_ctx, scale_shift_table, temb);
        auto modulation         = add_cont(ctx, scale_shift, temb);

        auto shift_msa   = slice_dim1(ctx, modulation, 0);
        auto scale_msa   = slice_dim1(ctx, modulation, 1);
        auto gate_msa    = slice_dim1(ctx, modulation, 2);
        auto c_shift_msa = slice_dim1(ctx, modulation, 3);
        auto c_scale_msa = slice_dim1(ctx, modulation, 4);
        auto c_gate_msa  = slice_dim1(ctx, modulation, 5);

        auto norm_hidden = self_attn_norm->forward(ctx, hidden_states);
        auto scale_b     = repeat_like(ctx, scale_msa, norm_hidden);
        auto shift_b     = repeat_like(ctx, shift_msa, norm_hidden);
        auto ones        = ggml_ext_ones(ctx->ggml_ctx, norm_hidden->ne[0], norm_hidden->ne[1], norm_hidden->ne[2], norm_hidden->ne[3]);
        auto scale_one   = add_cont(ctx, scale_b, ones);
        norm_hidden      = ggml_mul(ctx->ggml_ctx, norm_hidden, scale_one);
        norm_hidden      = add_cont(ctx, norm_hidden, shift_b);

        auto attn_out = self_attn->forward(ctx, norm_hidden, nullptr, use_sliding ? attention_mask : nullptr, input_pos);
        auto gate_b   = repeat_like(ctx, gate_msa, attn_out);
        attn_out      = ggml_mul(ctx->ggml_ctx, attn_out, gate_b);
        hidden_states = add_cont(ctx, hidden_states, attn_out);

        auto norm_hidden_cross = cross_attn_norm->forward(ctx, hidden_states);
        auto cross_out         = cross_attn->forward(ctx, norm_hidden_cross, encoder_hidden_states, nullptr, nullptr);
        hidden_states          = add_cont(ctx, hidden_states, cross_out);

        auto norm_hidden_mlp = mlp_norm->forward(ctx, hidden_states);
        auto c_scale_b       = repeat_like(ctx, c_scale_msa, norm_hidden_mlp);
        auto c_shift_b       = repeat_like(ctx, c_shift_msa, norm_hidden_mlp);
        auto c_scale_one     = add_cont(ctx, c_scale_b, ones);
        norm_hidden_mlp      = ggml_mul(ctx->ggml_ctx, norm_hidden_mlp, c_scale_one);
        norm_hidden_mlp      = add_cont(ctx, norm_hidden_mlp, c_shift_b);
        auto mlp_out         = mlp->forward(ctx, norm_hidden_mlp);
        auto c_gate_b        = repeat_like(ctx, c_gate_msa, mlp_out);
        mlp_out              = ggml_mul(ctx->ggml_ctx, mlp_out, c_gate_b);
        hidden_states        = add_cont(ctx, hidden_states, mlp_out);

        return hidden_states;
    }
};

struct AceStepEncoderLayer : public GGMLBlock {
    AceStepEncoderLayer(int64_t hidden_size,
                        int64_t num_heads,
                        int64_t num_kv_heads,
                        int64_t head_dim,
                        int64_t intermediate_size) {
        blocks["self_attn"] = std::make_shared<AceStepAttention>(hidden_size, num_heads, num_kv_heads, head_dim, false);
        blocks["input_layernorm"] = std::make_shared<RMSNorm>(hidden_size, 1e-6f);
        blocks["post_attention_layernorm"] = std::make_shared<RMSNorm>(hidden_size, 1e-6f);
        blocks["mlp"] = std::make_shared<AceMLP>(hidden_size, intermediate_size);
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx,
                         ggml_tensor* hidden_states,
                         ggml_tensor* input_pos,
                         ggml_tensor* attention_mask) {
        auto self_attn  = std::dynamic_pointer_cast<AceStepAttention>(blocks["self_attn"]);
        auto ln_in      = std::dynamic_pointer_cast<RMSNorm>(blocks["input_layernorm"]);
        auto ln_post    = std::dynamic_pointer_cast<RMSNorm>(blocks["post_attention_layernorm"]);
        auto mlp        = std::dynamic_pointer_cast<AceMLP>(blocks["mlp"]);

        auto residual = hidden_states;
        hidden_states = ln_in->forward(ctx, hidden_states);
        hidden_states = self_attn->forward(ctx, hidden_states, nullptr, attention_mask, input_pos);
        hidden_states = add_cont(ctx, hidden_states, residual);

        residual      = hidden_states;
        hidden_states = ln_post->forward(ctx, hidden_states);
        hidden_states = mlp->forward(ctx, hidden_states);
        hidden_states = add_cont(ctx, hidden_states, residual);
        return hidden_states;
    }
};

struct AceStepLyricEncoder : public GGMLBlock {
    int64_t num_layers;

    AceStepLyricEncoder(int64_t text_hidden_dim,
                        int64_t hidden_size,
                        int64_t num_layers,
                        int64_t num_heads,
                        int64_t num_kv_heads,
                        int64_t head_dim,
                        int64_t intermediate_size)
        : num_layers(num_layers) {
        blocks["embed_tokens"] = std::make_shared<Linear>(text_hidden_dim, hidden_size, true, false, false);
        blocks["norm"]         = std::make_shared<RMSNorm>(hidden_size, 1e-6f);
        for (int i = 0; i < num_layers; ++i) {
            blocks["layers." + std::to_string(i)] = std::make_shared<AceStepEncoderLayer>(hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size);
        }
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx,
                         ggml_tensor* inputs_embeds,
                         ggml_tensor* input_pos) {
        auto embed_tokens = std::dynamic_pointer_cast<Linear>(blocks["embed_tokens"]);
        auto norm         = std::dynamic_pointer_cast<RMSNorm>(blocks["norm"]);

        inputs_embeds = ggml_ext_cont(ctx->ggml_ctx, inputs_embeds);
        if (auto w = embed_tokens->get_weight(); w && inputs_embeds->type != w->type) {
            inputs_embeds = ggml_cast(ctx->ggml_ctx, inputs_embeds, w->type);
        }
        inputs_embeds = ggml_ext_cont(ctx->ggml_ctx, inputs_embeds);
        auto hidden_states = embed_tokens->forward(ctx, inputs_embeds);

        for (int i = 0; i < num_layers; ++i) {
            auto layer = std::dynamic_pointer_cast<AceStepEncoderLayer>(blocks["layers." + std::to_string(i)]);
            hidden_states = layer->forward(ctx, hidden_states, input_pos, nullptr);
        }

        hidden_states = norm->forward(ctx, hidden_states);
        return hidden_states;
    }
};

struct AceStepTimbreEncoder : public GGMLBlock {
    int64_t num_layers;
    int64_t hidden_size;

    AceStepTimbreEncoder(int64_t timbre_hidden_dim,
                         int64_t hidden_size,
                         int64_t num_layers,
                         int64_t num_heads,
                         int64_t num_kv_heads,
                         int64_t head_dim,
                         int64_t intermediate_size)
        : num_layers(num_layers),
          hidden_size(hidden_size) {
        // Use F32 accumulation to avoid CUDA NaNs in timbre projection.
        blocks["embed_tokens"] = std::make_shared<Linear>(timbre_hidden_dim, hidden_size, true, false, false);
        blocks["norm"]         = std::make_shared<RMSNorm>(hidden_size, 1e-6f);
        for (int i = 0; i < num_layers; ++i) {
            blocks["layers." + std::to_string(i)] = std::make_shared<AceStepEncoderLayer>(hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size);
        }
    }

protected:
    void init_params(struct ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
        enum ggml_type wtype = get_type(prefix + "special_token", tensor_storage_map, GGML_TYPE_F32);
        params["special_token"] = ggml_new_tensor_3d(ctx, wtype, hidden_size, 1, 1);
        GGMLBlock::init_params(ctx, tensor_storage_map, prefix);
    }

public:
    ggml_tensor* forward(GGMLRunnerContext* ctx,
                         ggml_tensor* refer_audio,
                         ggml_tensor* input_pos) {
        auto embed_tokens = std::dynamic_pointer_cast<Linear>(blocks["embed_tokens"]);
        auto norm         = std::dynamic_pointer_cast<RMSNorm>(blocks["norm"]);

        refer_audio = ggml_ext_cont(ctx->ggml_ctx, refer_audio);
        if (auto w = embed_tokens->get_weight(); w && refer_audio->type != w->type) {
            refer_audio = ggml_cast(ctx->ggml_ctx, refer_audio, w->type);
        }
        refer_audio = ggml_ext_cont(ctx->ggml_ctx, refer_audio);
        auto hidden_states = embed_tokens->forward(ctx, refer_audio);
        for (int i = 0; i < num_layers; ++i) {
            auto layer = std::dynamic_pointer_cast<AceStepEncoderLayer>(blocks["layers." + std::to_string(i)]);
            hidden_states = layer->forward(ctx, hidden_states, input_pos, nullptr);
        }
        hidden_states = norm->forward(ctx, hidden_states);

        // take first token as timbre embedding
        hidden_states = slice_dim1(ctx, hidden_states, 0);
        return hidden_states;
    }
};

struct AceStepConditionEncoder : public GGMLBlock {
    AceStepConditionEncoder(int64_t text_hidden_dim,
                            int64_t timbre_hidden_dim,
                            int64_t hidden_size,
                            int64_t num_lyric_layers,
                            int64_t num_timbre_layers,
                            int64_t num_heads,
                            int64_t num_kv_heads,
                            int64_t head_dim,
                            int64_t intermediate_size) {
        blocks["text_projector"] = std::make_shared<Linear>(text_hidden_dim, hidden_size, false, false, false);
        blocks["lyric_encoder"]  = std::make_shared<AceStepLyricEncoder>(text_hidden_dim, hidden_size, num_lyric_layers, num_heads, num_kv_heads, head_dim, intermediate_size);
        blocks["timbre_encoder"] = std::make_shared<AceStepTimbreEncoder>(timbre_hidden_dim, hidden_size, num_timbre_layers, num_heads, num_kv_heads, head_dim, intermediate_size);
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx,
                         ggml_tensor* text_hidden_states,
                         ggml_tensor* lyric_hidden_states,
                         ggml_tensor* refer_audio,
                         ggml_tensor* lyric_pos,
                         ggml_tensor* timbre_pos) {
        auto text_projector = std::dynamic_pointer_cast<Linear>(blocks["text_projector"]);
        auto lyric_encoder  = std::dynamic_pointer_cast<AceStepLyricEncoder>(blocks["lyric_encoder"]);
        auto timbre_encoder = std::dynamic_pointer_cast<AceStepTimbreEncoder>(blocks["timbre_encoder"]);

        text_hidden_states  = ggml_ext_cont(ctx->ggml_ctx, text_hidden_states);
        lyric_hidden_states = ggml_ext_cont(ctx->ggml_ctx, lyric_hidden_states);
        refer_audio         = ggml_ext_cont(ctx->ggml_ctx, refer_audio);
        ggml_tensor* text_input = text_hidden_states;
        int64_t text_len = text_hidden_states->ne[1];
        int64_t text_pad = (256 - (text_len % 256)) % 256;
        if (text_pad > 0) {
            auto pad_tensor = ggml_ext_full(ctx->ggml_ctx, 0.0f, text_hidden_states->ne[0], text_pad, text_hidden_states->ne[2], text_hidden_states->ne[3]);
            if (text_hidden_states->type != GGML_TYPE_F32) {
                pad_tensor = ggml_cast(ctx->ggml_ctx, pad_tensor, text_hidden_states->type);
            }
            text_input = ggml_concat(ctx->ggml_ctx, text_hidden_states, pad_tensor, 1);
        }
        if (auto w = text_projector->get_weight(); w && text_input->type != w->type) {
            text_input = ggml_cast(ctx->ggml_ctx, text_input, w->type);
        }
        text_input = ggml_ext_cont(ctx->ggml_ctx, text_input);
        auto text_emb  = text_projector->forward(ctx, text_input);
        if (text_pad > 0) {
            text_emb = ggml_ext_slice(ctx->ggml_ctx, text_emb, 1, 0, text_len, true);
        }
        auto lyric_emb = lyric_encoder->forward(ctx, lyric_hidden_states, lyric_pos);
        auto timbre_emb = timbre_encoder->forward(ctx, refer_audio, timbre_pos);

        // CUDA concat only supports f32 today; normalize to f32 for packing.
        ggml_type merged_type = GGML_TYPE_F32;
        if (text_emb->type != merged_type) {
            text_emb = ggml_cast(ctx->ggml_ctx, text_emb, merged_type);
        }
        if (lyric_emb->type != merged_type) {
            lyric_emb = ggml_cast(ctx->ggml_ctx, lyric_emb, merged_type);
        }
        if (timbre_emb->type != merged_type) {
            timbre_emb = ggml_cast(ctx->ggml_ctx, timbre_emb, merged_type);
        }
        text_emb   = ggml_ext_cont(ctx->ggml_ctx, text_emb);
        lyric_emb  = ggml_ext_cont(ctx->ggml_ctx, lyric_emb);
        timbre_emb = ggml_ext_cont(ctx->ggml_ctx, timbre_emb);

        auto merged = ggml_concat(ctx->ggml_ctx, lyric_emb, timbre_emb, 1);
        merged      = ggml_concat(ctx->ggml_ctx, merged, text_emb, 1);
        return merged;
    }
};

struct AceStepAttentionPooler : public GGMLBlock {
    int64_t hidden_size;
    int64_t num_layers;
    int64_t head_dim;
    int64_t num_heads;
    int64_t num_kv_heads;
    int64_t intermediate_size;
    float rms_norm_eps;

    AceStepAttentionPooler(int64_t hidden_size,
                           int64_t num_layers,
                           int64_t head_dim,
                           float rms_norm_eps = 1e-6f)
        : hidden_size(hidden_size),
          num_layers(num_layers),
          head_dim(head_dim),
          num_heads(16),
          num_kv_heads(8),
          intermediate_size(hidden_size * 3),
          rms_norm_eps(rms_norm_eps) {
        blocks["embed_tokens"] = std::make_shared<Linear>(hidden_size, hidden_size, true);
        blocks["norm"]         = std::make_shared<RMSNorm>(hidden_size, rms_norm_eps);
        for (int i = 0; i < num_layers; ++i) {
            blocks["layers." + std::to_string(i)] = std::make_shared<AceStepEncoderLayer>(hidden_size,
                                                                                          num_heads,
                                                                                          num_kv_heads,
                                                                                          head_dim,
                                                                                          intermediate_size);
        }
    }

protected:
    void init_params(struct ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
        enum ggml_type wtype = get_type(prefix + "special_token", tensor_storage_map, GGML_TYPE_F32);
        params["special_token"] = ggml_new_tensor_3d(ctx, wtype, hidden_size, 1, 1);
        GGMLBlock::init_params(ctx, tensor_storage_map, prefix);
    }

public:
    ggml_tensor* forward(GGMLRunnerContext* ctx,
                         ggml_tensor* x,
                         ggml_tensor* input_pos) {
        GGML_ASSERT(x != nullptr && ggml_n_dims(x) == 4);

        auto embed_tokens = std::dynamic_pointer_cast<Linear>(blocks["embed_tokens"]);
        auto norm         = std::dynamic_pointer_cast<RMSNorm>(blocks["norm"]);

        int64_t P = x->ne[1];
        int64_t T = x->ne[2];
        int64_t B = x->ne[3];

        auto x3 = ggml_reshape_3d(ctx->ggml_ctx, cont_if_needed(ctx, x), x->ne[0], P, T * B);
        x3      = embed_tokens->forward(ctx, x3);
        auto x4 = ggml_reshape_4d(ctx->ggml_ctx, x3, x3->ne[0], P, T, B);

        auto special = params["special_token"];
        special      = ggml_reshape_4d(ctx->ggml_ctx, special, special->ne[0], 1, 1, 1);
        auto repeat_target = ggml_new_tensor_4d(ctx->ggml_ctx, GGML_TYPE_F16, x4->ne[0], 1, T, B);
        auto special_rep   = ggml_ext_repeat(ctx->ggml_ctx, special, repeat_target);

        x4 = ggml_concat(ctx->ggml_ctx, special_rep, x4, 1);
        x4 = ggml_cont(ctx->ggml_ctx, x4);

        int64_t seq_len = x4->ne[1];
        auto x_seq = ggml_reshape_3d(ctx->ggml_ctx, x4, x4->ne[0], seq_len, T * B);

        for (int i = 0; i < num_layers; ++i) {
            auto layer = std::dynamic_pointer_cast<AceStepEncoderLayer>(blocks["layers." + std::to_string(i)]);
            x_seq = layer->forward(ctx, x_seq, input_pos, nullptr);
        }

        x_seq = norm->forward(ctx, x_seq);
        x_seq = slice_dim1(ctx, x_seq, 0);
        x_seq = ggml_cont(ctx->ggml_ctx, x_seq);
        x_seq = ggml_reshape_3d(ctx->ggml_ctx, x_seq, x_seq->ne[0], T, B);
        return x_seq;
    }
};

struct AceStepDiTModel : public GGMLBlock {
    int64_t hidden_size;
    int64_t patch_size;
    int64_t num_layers;
    int64_t num_heads;
    int64_t num_kv_heads;
    int64_t head_dim;
    int64_t intermediate_size;
    int64_t audio_acoustic_hidden_dim;
    std::vector<bool> sliding_layers;

    AceStepDiTModel(int64_t in_channels,
                    int64_t hidden_size,
                    int64_t num_layers,
                    int64_t num_heads,
                    int64_t num_kv_heads,
                    int64_t head_dim,
                    int64_t intermediate_size,
                    int64_t patch_size,
                    int64_t audio_acoustic_hidden_dim,
                    const std::vector<bool>& sliding_layers)
        : hidden_size(hidden_size),
          patch_size(patch_size),
          num_layers(num_layers),
          num_heads(num_heads),
          num_kv_heads(num_kv_heads),
          head_dim(head_dim),
          intermediate_size(intermediate_size),
          audio_acoustic_hidden_dim(audio_acoustic_hidden_dim),
          sliding_layers(sliding_layers) {
        blocks["proj_in.1"]        = std::make_shared<Conv1d>(in_channels, hidden_size, (int)patch_size, (int)patch_size, 0);
        blocks["time_embed"]       = std::make_shared<TimestepEmbedding>(256, hidden_size);
        blocks["time_embed_r"]     = std::make_shared<TimestepEmbedding>(256, hidden_size);
        blocks["condition_embedder"] = std::make_shared<Linear>(hidden_size, hidden_size, true);

        for (int i = 0; i < num_layers; ++i) {
            blocks["layers." + std::to_string(i)] = std::make_shared<AceStepDiTLayer>(hidden_size,
                                                                                      num_heads,
                                                                                      num_kv_heads,
                                                                                      head_dim,
                                                                                      intermediate_size,
                                                                                      sliding_layers[i]);
        }

        blocks["norm_out"]   = std::make_shared<RMSNorm>(hidden_size, 1e-6f);
        blocks["proj_out.1"] = std::make_shared<ConvTranspose1d>(hidden_size, audio_acoustic_hidden_dim, (int)patch_size, (int)patch_size, 0);
    }

protected:
    void init_params(struct ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
        enum ggml_type wtype = get_type(prefix + "scale_shift_table", tensor_storage_map, GGML_TYPE_F32);
        params["scale_shift_table"] = ggml_new_tensor_3d(ctx, wtype, hidden_size, 2, 1);
        GGMLBlock::init_params(ctx, tensor_storage_map, prefix);
    }

public:
    ggml_tensor* forward(GGMLRunnerContext* ctx,
                         ggml_tensor* hidden_states,
                         ggml_tensor* timesteps,
                         ggml_tensor* timesteps_r,
                         ggml_tensor* encoder_hidden_states,
                         ggml_tensor* context_latents,
                         ggml_tensor* input_pos,
                         ggml_tensor* attention_mask) {
        auto time_embed   = std::dynamic_pointer_cast<TimestepEmbedding>(blocks["time_embed"]);
        auto time_embed_r = std::dynamic_pointer_cast<TimestepEmbedding>(blocks["time_embed_r"]);
        auto condition_embedder = std::dynamic_pointer_cast<Linear>(blocks["condition_embedder"]);
        auto norm_out     = std::dynamic_pointer_cast<RMSNorm>(blocks["norm_out"]);
        auto proj_in      = std::dynamic_pointer_cast<Conv1d>(blocks["proj_in.1"]);
        auto proj_out     = std::dynamic_pointer_cast<ConvTranspose1d>(blocks["proj_out.1"]);

        ggml_tensor* t_delta = ggml_sub(ctx->ggml_ctx, timesteps, timesteps_r);

        auto temb_t_pair = time_embed->forward(ctx, timesteps);
        auto temb_r_pair = time_embed_r->forward(ctx, t_delta);

        auto temb         = add_cont(ctx, temb_t_pair.first, temb_r_pair.first);
        auto timestep_proj = add_cont(ctx, temb_t_pair.second, temb_r_pair.second);

        GGML_ASSERT(context_latents->ne[1] == hidden_states->ne[1]);
        GGML_ASSERT(context_latents->ne[2] == hidden_states->ne[2]);
        GGML_ASSERT(context_latents->ne[3] == hidden_states->ne[3]);

        auto lhs = context_latents;
        auto rhs = hidden_states;
        if (rhs->type != lhs->type) {
            rhs = ggml_cast(ctx->ggml_ctx, rhs, lhs->type);
        }

        ggml_tensor* x = ggml_concat(ctx->ggml_ctx, lhs, rhs, 0);

        int64_t original_seq_len = x->ne[1];
        if (original_seq_len % patch_size != 0) {
            int64_t pad_len = patch_size - (original_seq_len % patch_size);
            auto pad4 = ggml_ext_zeros(ctx->ggml_ctx, x->ne[0], pad_len, x->ne[2], 1);
            auto pad = ggml_reshape_3d(ctx->ggml_ctx, pad4, x->ne[0], pad_len, x->ne[2]);
            x = ggml_concat(ctx->ggml_ctx, x, pad, 1);
        }

        x = swap_dim0_dim1(ctx, x);
        x = proj_in->forward(ctx, x);
        x = swap_dim0_dim1(ctx, x);

        encoder_hidden_states = condition_embedder->forward(ctx, encoder_hidden_states);

        for (int i = 0; i < num_layers; ++i) {
            auto layer = std::dynamic_pointer_cast<AceStepDiTLayer>(blocks["layers." + std::to_string(i)]);
            x = layer->forward(ctx, x, timestep_proj, encoder_hidden_states, input_pos, attention_mask);
        }

        auto scale_shift_table = params["scale_shift_table"];
        auto temb_reshaped      = ggml_reshape_3d(ctx->ggml_ctx, temb, temb->ne[0], 1, temb->ne[1]);
        auto repeat_target      = ggml_new_tensor_3d(ctx->ggml_ctx, GGML_TYPE_F16, temb->ne[0], 2, temb->ne[1]);
        auto scale_shift        = ggml_ext_repeat(ctx->ggml_ctx, scale_shift_table, repeat_target);
        auto temb_rep           = ggml_ext_repeat(ctx->ggml_ctx, temb_reshaped, repeat_target);
        auto modulation         = add_cont(ctx, scale_shift, temb_rep);

        auto shift = slice_dim1(ctx, modulation, 0);
        auto scale = slice_dim1(ctx, modulation, 1);

        x = norm_out->forward(ctx, x);
        auto scale_b = repeat_like(ctx, scale, x);
        auto shift_b = repeat_like(ctx, shift, x);
        auto ones = ggml_ext_ones(ctx->ggml_ctx, x->ne[0], x->ne[1], x->ne[2], x->ne[3]);
        auto scale_one = add_cont(ctx, scale_b, ones);
        x = ggml_mul(ctx->ggml_ctx, x, scale_one);
        x = add_cont(ctx, x, shift_b);

        x = swap_dim0_dim1(ctx, x);
        x = proj_out->forward(ctx, x);
        x = swap_dim0_dim1(ctx, x);

        x = ggml_ext_slice(ctx->ggml_ctx, x, 1, 0, original_seq_len, true);
        return x;
    }
};

struct AceQuantizer : public GGMLBlock {
    std::vector<int> levels;
    std::vector<int> basis;
    int64_t codebook_dim;
    int64_t hidden_size;
    std::vector<float> codes_buffer;

    AceQuantizer(int64_t hidden_size, const std::vector<int>& levels)
        : levels(levels), codebook_dim(levels.size()), hidden_size(hidden_size) {
        blocks["project_in"]  = std::make_shared<Linear>(hidden_size, (int64_t)codebook_dim, true);
        blocks["project_out"] = std::make_shared<Linear>((int64_t)codebook_dim, hidden_size, true);

        basis.resize(codebook_dim);
        int accum = 1;
        for (size_t i = 0; i < codebook_dim; ++i) {
            basis[i] = accum;
            accum *= levels[i];
        }
    }

    ggml_tensor* get_output_from_indices(GGMLRunnerContext* ctx,
                                         const std::vector<int>& indices,
                                         int64_t T,
                                         int64_t B) {
        auto project_out = std::dynamic_pointer_cast<Linear>(blocks["project_out"]);

        const int64_t total = codebook_dim * T * B;
        if (total <= 0) {
            return nullptr;
        }
        codes_buffer.assign(static_cast<size_t>(total), 0.f);

        for (int64_t b = 0; b < B; ++b) {
            for (int64_t t = 0; t < T; ++t) {
                int idx = 35847;
                if (!indices.empty() && t < (int64_t)indices.size()) {
                    idx = indices[t];
                }
                for (int64_t d = 0; d < codebook_dim; ++d) {
                    int level = levels[d];
                    int value = (idx / basis[d]) % level;
                    float scaled = value * (2.f / (level - 1)) - 1.f;
                    int64_t offset = d + codebook_dim * (t + T * b);
                    codes_buffer[static_cast<size_t>(offset)] = scaled;
                }
            }
        }

        auto codes = ggml_new_tensor_3d(ctx->ggml_ctx, GGML_TYPE_F32, codebook_dim, T, B);
        if (codes->data != nullptr) {
            memcpy(codes->data, codes_buffer.data(), static_cast<size_t>(total) * sizeof(float));
        } else {
            ctx->set_backend_tensor_data(codes, codes_buffer.data());
        }

        if (auto w = project_out->get_weight(); w && codes->type != w->type) {
            codes = ggml_cast(ctx->ggml_ctx, codes, w->type);
        }
        auto out = project_out->forward(ctx, codes);
        return out;
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        auto project_in  = std::dynamic_pointer_cast<Linear>(blocks["project_in"]);
        auto project_out = std::dynamic_pointer_cast<Linear>(blocks["project_out"]);

        auto z = project_in->forward(ctx, x);
        z = ggml_cast(ctx->ggml_ctx, z, GGML_TYPE_F32);

        auto tanh_z = ggml_tanh(ctx->ggml_ctx, z);
        auto ones = ggml_ext_ones(ctx->ggml_ctx, tanh_z->ne[0], tanh_z->ne[1], tanh_z->ne[2], tanh_z->ne[3]);
        auto z_plus = ggml_add(ctx->ggml_ctx, tanh_z, ones);
        auto z_scaled = ggml_scale(ctx->ggml_ctx, z_plus, 0.5f);

        std::vector<float> levels_minus_1_vec(codebook_dim);
        std::vector<float> scales_vec(codebook_dim);
        for (int i = 0; i < codebook_dim; ++i) {
            float level_minus_1 = static_cast<float>(levels[i] - 1);
            levels_minus_1_vec[i] = level_minus_1;
            scales_vec[i] = 2.f / level_minus_1;
        }

        auto levels_minus_1 = ggml_new_tensor_3d(ctx->ggml_ctx, GGML_TYPE_F32, codebook_dim, 1, 1);
        ctx->set_backend_tensor_data(levels_minus_1, levels_minus_1_vec.data());
        auto levels_rep = ggml_repeat(ctx->ggml_ctx, levels_minus_1, z_scaled);
        auto scaled = ggml_mul(ctx->ggml_ctx, z_scaled, levels_rep);
        auto rounded = ggml_round(ctx->ggml_ctx, scaled);

        auto scales = ggml_new_tensor_3d(ctx->ggml_ctx, GGML_TYPE_F32, codebook_dim, 1, 1);
        ctx->set_backend_tensor_data(scales, scales_vec.data());
        auto scales_rep = ggml_repeat(ctx->ggml_ctx, scales, rounded);
        auto codes = ggml_mul(ctx->ggml_ctx, rounded, scales_rep);
        auto neg_ones = ggml_scale(ctx->ggml_ctx, ones, -1.f);
        codes = ggml_add(ctx->ggml_ctx, codes, neg_ones);

        return project_out->forward(ctx, codes);
    }
};

struct AudioTokenDetokenizer : public GGMLBlock {
    int64_t pool_window_size;
    int64_t hidden_size;
    int64_t audio_acoustic_hidden_dim;
    int64_t num_layers;
    int64_t num_heads;
    int64_t num_kv_heads;
    int64_t head_dim;
    int64_t intermediate_size;

    AudioTokenDetokenizer(int64_t hidden_size,
                          int64_t pool_window_size,
                          int64_t audio_acoustic_hidden_dim,
                          int64_t num_layers,
                          int64_t num_heads,
                          int64_t num_kv_heads,
                          int64_t head_dim,
                          int64_t intermediate_size)
        : pool_window_size(pool_window_size),
          hidden_size(hidden_size),
          audio_acoustic_hidden_dim(audio_acoustic_hidden_dim),
          num_layers(num_layers),
          num_heads(num_heads),
          num_kv_heads(num_kv_heads),
          head_dim(head_dim),
          intermediate_size(intermediate_size) {
        blocks["embed_tokens"] = std::make_shared<Linear>(hidden_size, hidden_size, true);
        blocks["norm"]         = std::make_shared<RMSNorm>(hidden_size, 1e-6f);
        blocks["proj_out"]     = std::make_shared<Linear>(hidden_size, audio_acoustic_hidden_dim, true);
        for (int i = 0; i < num_layers; ++i) {
            blocks["layers." + std::to_string(i)] = std::make_shared<AceStepEncoderLayer>(hidden_size,
                                                                                          num_heads,
                                                                                          num_kv_heads,
                                                                                          head_dim,
                                                                                          intermediate_size);
        }
    }

protected:
    void init_params(struct ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
        enum ggml_type wtype = get_type(prefix + "special_tokens", tensor_storage_map, GGML_TYPE_F32);
        params["special_tokens"] = ggml_new_tensor_3d(ctx, wtype, hidden_size, pool_window_size, 1);
        GGMLBlock::init_params(ctx, tensor_storage_map, prefix);
    }

public:
    ggml_tensor* forward(GGMLRunnerContext* ctx,
                         ggml_tensor* x,
                         ggml_tensor* input_pos) {
        auto embed_tokens = std::dynamic_pointer_cast<Linear>(blocks["embed_tokens"]);
        auto norm         = std::dynamic_pointer_cast<RMSNorm>(blocks["norm"]);
        auto proj_out     = std::dynamic_pointer_cast<Linear>(blocks["proj_out"]);

        int64_t T = x->ne[1];
        int64_t B = x->ne[2];

        x = embed_tokens->forward(ctx, x);
        x = ggml_cont(ctx->ggml_ctx, x);
        const int64_t D  = x->ne[0];
        const int64_t TB = x->ne[1] * x->ne[2];
        auto x3          = ggml_reshape_3d(ctx->ggml_ctx, x, D, TB, 1);
        x3               = ggml_dup(ctx->ggml_ctx, x3);
        auto repeat_target = ggml_new_tensor_3d(ctx->ggml_ctx, x3->type, D, TB, pool_window_size);
        auto x3r           = ggml_ext_repeat(ctx->ggml_ctx, x3, repeat_target);  // (D, TB, P)
        x3r                = ggml_permute(ctx->ggml_ctx, x3r, 0, 2, 1, 3);       // (D, P, TB)
        x3r                = ggml_cont(ctx->ggml_ctx, x3r);

        auto special = params["special_tokens"];
        special      = ggml_reshape_3d(ctx->ggml_ctx, special, special->ne[0], special->ne[1], 1);  // (D, P, 1)
        special      = ggml_dup(ctx->ggml_ctx, special);
        auto special_rep = ggml_ext_repeat(ctx->ggml_ctx, special, x3r);
        x                = add_cont(ctx, x3r, special_rep);

        for (int i = 0; i < num_layers; ++i) {
            auto layer = std::dynamic_pointer_cast<AceStepEncoderLayer>(blocks["layers." + std::to_string(i)]);
            x = layer->forward(ctx, x, input_pos, nullptr);
        }

        x = norm->forward(ctx, x);
        x = proj_out->forward(ctx, x);

        x = ggml_cont(ctx->ggml_ctx, x);
        x = ggml_reshape_4d(ctx->ggml_ctx, x, x->ne[0], pool_window_size, T, B);
        x = ggml_cont(ctx->ggml_ctx, x);
        x = ggml_reshape_3d(ctx->ggml_ctx, x, x->ne[0], pool_window_size * T, B);
        return x;
    }
};

struct AceStepConditionGenerationModel : public GGMLBlock {
    int64_t pool_window_size = 5;
    std::vector<int> fsq_levels = {8, 8, 8, 5, 5, 5};
    int64_t hidden_size = 2048;
    int64_t audio_acoustic_hidden_dim = 64;
    int64_t patch_size = 2;
    bool use_tokenizer_path = false;  // mirror Comfy: LM-only hints by default

    AceStepConditionGenerationModel() {
        std::vector<bool> layer_types;
        for (int i = 0; i < 24; ++i) {
            layer_types.push_back((i % 2) == 0);
        }
        blocks["decoder"] = std::make_shared<AceStepDiTModel>(192, hidden_size, 24, 16, 8, 128, 6144, patch_size, audio_acoustic_hidden_dim, layer_types);
        blocks["encoder"] = std::make_shared<AceStepConditionEncoder>(1024, audio_acoustic_hidden_dim, hidden_size, 8, 4, 16, 8, 128, 6144);
        blocks["tokenizer.audio_acoustic_proj"] = std::make_shared<Linear>(audio_acoustic_hidden_dim, hidden_size, true);
        blocks["tokenizer.attention_pooler"] = std::make_shared<AceStepAttentionPooler>(hidden_size, 2, 128, 1e-6f);
        blocks["tokenizer.quantizer"] = std::make_shared<AceQuantizer>(hidden_size, fsq_levels);
        blocks["detokenizer"] = std::make_shared<AudioTokenDetokenizer>(hidden_size, pool_window_size, audio_acoustic_hidden_dim, 2, 16, 8, 128, hidden_size * 3);
    }

protected:
    void init_params(struct ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
        enum ggml_type wtype = get_type(prefix + "null_condition_emb", tensor_storage_map, GGML_TYPE_F32);
        params["null_condition_emb"] = ggml_new_tensor_3d(ctx, wtype, hidden_size, 1, 1);
        GGMLBlock::init_params(ctx, tensor_storage_map, prefix);
    }

public:
    ggml_tensor* forward(GGMLRunnerContext* ctx,
                         ggml_tensor* x,
                         ggml_tensor* timesteps,
                         ggml_tensor* context,
                         ggml_tensor* lyric_embed,
                         ggml_tensor* refer_audio,
                         const std::shared_ptr<std::vector<int>>& audio_codes,
                         ggml_tensor* decoder_pos,
                         ggml_tensor* lyric_pos,
                         ggml_tensor* timbre_pos,
                         ggml_tensor* tokenizer_pos,
                         ggml_tensor* detok_pos,
                         ggml_tensor* sliding_mask) {
        auto encoder     = std::dynamic_pointer_cast<AceStepConditionEncoder>(blocks["encoder"]);
        auto decoder     = std::dynamic_pointer_cast<AceStepDiTModel>(blocks["decoder"]);
        auto audio_proj  = std::dynamic_pointer_cast<Linear>(blocks["tokenizer.audio_acoustic_proj"]);
        auto pooler      = std::dynamic_pointer_cast<AceStepAttentionPooler>(blocks["tokenizer.attention_pooler"]);
        auto quantizer   = std::dynamic_pointer_cast<AceQuantizer>(blocks["tokenizer.quantizer"]);
        auto detokenizer = std::dynamic_pointer_cast<AudioTokenDetokenizer>(blocks["detokenizer"]);

        ggml_tensor* enc_hidden = nullptr;
        if (lyric_embed == nullptr && refer_audio == nullptr && context && context->ne[0] == hidden_size) {
            enc_hidden = context;
        } else {
            enc_hidden = encoder->forward(ctx, context, lyric_embed, refer_audio, lyric_pos, timbre_pos);
        }

        auto src_latents = x;
        auto chunk_masks = ggml_ext_ones(ctx->ggml_ctx, x->ne[0], x->ne[1], x->ne[2], x->ne[3]);

        int64_t T = x->ne[1];
        int64_t B = x->ne[2];
        int64_t T_codes = (T + pool_window_size - 1) / pool_window_size;

        ggml_tensor* tokenizer_hints_5hz = nullptr;
        if (use_tokenizer_path && refer_audio && audio_proj && pooler && quantizer && tokenizer_pos) {
            int64_t target_len = T_codes * pool_window_size;
            ggml_tensor* tok_audio = refer_audio;
            if (tok_audio->ne[1] < target_len) {
                int64_t repeat_factor = (target_len + tok_audio->ne[1] - 1) / tok_audio->ne[1];
                auto repeat_target = ggml_new_tensor_3d(ctx->ggml_ctx, tok_audio->type, tok_audio->ne[0], tok_audio->ne[1] * repeat_factor, tok_audio->ne[2]);
                tok_audio = ggml_ext_repeat(ctx->ggml_ctx, tok_audio, repeat_target);
            }
            if (tok_audio->ne[1] > target_len) {
                tok_audio = ggml_ext_slice(ctx->ggml_ctx, tok_audio, 1, 0, target_len, true);
            }

            auto tok_hidden = audio_proj->forward(ctx, tok_audio);
            auto tok_hidden4 = ggml_reshape_4d(ctx->ggml_ctx, tok_hidden, tok_hidden->ne[0], pool_window_size, T_codes, tok_hidden->ne[2]);
            auto pooled = pooler->forward(ctx, tok_hidden4, tokenizer_pos);
            tokenizer_hints_5hz = quantizer->forward(ctx, pooled);
        }

        ggml_tensor* lm_hints_5hz = nullptr;
        if (audio_codes && !audio_codes->empty()) {
            lm_hints_5hz = quantizer->get_output_from_indices(ctx, *audio_codes, T_codes, B);
        }
        if (lm_hints_5hz == nullptr) {
            lm_hints_5hz = tokenizer_hints_5hz;
        } else if (tokenizer_hints_5hz != nullptr) {
            auto combined = add_cont(ctx, lm_hints_5hz, tokenizer_hints_5hz);
            lm_hints_5hz = ggml_scale(ctx->ggml_ctx, combined, 0.5f);
        }

        if (lm_hints_5hz != nullptr) {
            auto lm_hints = detokenizer->forward(ctx, lm_hints_5hz, detok_pos);
            lm_hints = ggml_ext_slice(ctx->ggml_ctx, lm_hints, 1, 0, T, true);
            src_latents = lm_hints;
        }

        auto context_latents = ggml_concat(ctx->ggml_ctx, src_latents, chunk_masks, 0);

        auto out = decoder->forward(ctx,
                                    x,
                                    timesteps,
                                    timesteps,
                                    enc_hidden,
                                    context_latents,
                                    decoder_pos,
                                    sliding_mask);

        return out;
    }
};

struct AceEncoderRunner : public GGMLRunner {
    AceStepConditionEncoder encoder;
    std::vector<int> lyric_pos_vec;
    std::vector<int> timbre_pos_vec;

    AceEncoderRunner(ggml_backend_t backend,
                     bool offload_params_to_cpu,
                     const String2TensorStorage& tensor_storage_map = {},
                     const std::string prefix = "model.diffusion_model.encoder")
        : GGMLRunner(backend, offload_params_to_cpu),
          encoder(1024, 64, 2048, 8, 4, 16, 8, 128, 6144) {
        encoder.init(params_ctx, tensor_storage_map, prefix);
    }

    std::string get_desc() override {
        return "ace_step_1_5_encoder";
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        encoder.get_param_tensors(tensors, prefix);
    }

    ggml_tensor* build_input_pos(int64_t seq_len, std::vector<int>& cache) {
        cache.resize(seq_len);
        for (int64_t i = 0; i < seq_len; ++i) {
            cache[i] = (int)i;
        }
        auto input_pos = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_I32, seq_len);
        set_backend_tensor_data(input_pos, cache.data());
        return input_pos;
    }

    struct ggml_cgraph* build_graph(ggml_tensor* text_hidden_states,
                                    ggml_tensor* lyric_hidden_states,
                                    ggml_tensor* refer_audio) {
        struct ggml_cgraph* gf = new_graph_custom(ACE_GRAPH_SIZE);

        text_hidden_states  = to_backend(text_hidden_states);
        lyric_hidden_states = to_backend(lyric_hidden_states);
        refer_audio         = to_backend(refer_audio);

        int64_t lyric_len  = lyric_hidden_states ? lyric_hidden_states->ne[1] : 1;
        auto lyric_pos     = build_input_pos(lyric_len, lyric_pos_vec);
        int64_t timbre_len = refer_audio ? refer_audio->ne[1] : 1;
        auto timbre_pos    = build_input_pos(timbre_len, timbre_pos_vec);

        auto runner_ctx = get_context();
        ggml_tensor* out = encoder.forward(&runner_ctx,
                                           text_hidden_states,
                                           lyric_hidden_states,
                                           refer_audio,
                                           lyric_pos,
                                           timbre_pos);
        ggml_build_forward_expand(gf, out);
        return gf;
    }

    bool compute(int n_threads,
                 ggml_tensor* text_hidden_states,
                 ggml_tensor* lyric_hidden_states,
                 ggml_tensor* refer_audio,
                 ggml_tensor** output = nullptr,
                 ggml_context* output_ctx = nullptr) {
        auto get_graph = [&]() -> ggml_cgraph* {
            return build_graph(text_hidden_states, lyric_hidden_states, refer_audio);
        };
        return GGMLRunner::compute(get_graph, n_threads, false, output, output_ctx);
    }
};

struct AceRunner : public GGMLRunner {
    AceStepConditionGenerationModel model;
    std::vector<int> input_pos_vec;
    std::vector<int> lyric_pos_vec;
    std::vector<int> timbre_pos_vec;
    std::vector<int> tokenizer_pos_vec;
    std::vector<int> detok_pos_vec;
    std::vector<float> sliding_mask_vec;
    std::unique_ptr<AceEncoderRunner> cpu_encoder;
    bool use_cpu_encoder = false;

    struct EncCacheEntry {
        const ggml_tensor* context = nullptr;
        const ggml_tensor* lyric = nullptr;
        const ggml_tensor* refer = nullptr;
        int n_dims = 0;
        int64_t ne[4] = {0, 0, 0, 0};
        std::vector<float> data;
    };

    std::vector<EncCacheEntry> enc_cache;
    const EncCacheEntry* active_enc_cache = nullptr;

    AceRunner(ggml_backend_t backend,
              bool offload_params_to_cpu,
              const String2TensorStorage& tensor_storage_map = {},
              const std::string prefix = "model.diffusion_model")
        : GGMLRunner(backend, offload_params_to_cpu) {
        model = AceStepConditionGenerationModel();
        model.init(params_ctx, tensor_storage_map, prefix);
    }

    std::string get_desc() override {
        return "ace_step_1_5";
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        model.get_param_tensors(tensors, prefix);
    }

    void reset_encoder_cache() {
        enc_cache.clear();
        active_enc_cache = nullptr;
    }

    void init_cpu_encoder(const String2TensorStorage& tensor_storage_map,
                          const std::map<std::string, struct ggml_tensor*>& model_tensors) {
        if (cpu_encoder) {
            use_cpu_encoder = true;
            return;
        }
        ggml_backend_t cpu_backend = ggml_backend_cpu_init();
        cpu_encoder = std::make_unique<AceEncoderRunner>(cpu_backend, false, tensor_storage_map, "model.diffusion_model.encoder");
        cpu_encoder->alloc_params_buffer();

        std::map<std::string, struct ggml_tensor*> cpu_tensors;
        cpu_encoder->get_param_tensors(cpu_tensors, "model.diffusion_model.encoder");
        int copied = 0;
        for (const auto& kv : cpu_tensors) {
            auto it = model_tensors.find(kv.first);
            if (it == model_tensors.end()) {
                LOG_WARN("ACE CPU encoder: missing tensor '%s'", kv.first.c_str());
                continue;
            }
            ggml_backend_tensor_copy(it->second, kv.second);
            copied++;
        }
        LOG_INFO("ACE CPU encoder: copied %d/%d tensors", copied, (int)cpu_tensors.size());
        use_cpu_encoder = true;
    }

    ggml_tensor* build_mask(int64_t seq_len, int64_t window) {
        sliding_mask_vec.resize(seq_len * seq_len);
        for (int64_t i0 = 0; i0 < seq_len; ++i0) {
            for (int64_t i1 = 0; i1 < seq_len; ++i1) {
                float value = 0.f;
                if (std::abs(i0 - i1) > window) {
                    // Match Comfy/PyTorch sliding window bias: torch.finfo(dtype).min.
                    // Sliding attention path does not use flash attention in our backend.
                    value = std::numeric_limits<float>::lowest();
                }
                sliding_mask_vec[i1 * seq_len + i0] = value;
            }
        }
        auto mask = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, seq_len, seq_len);
        if (mask->data != nullptr) {
            memcpy(mask->data, sliding_mask_vec.data(), sliding_mask_vec.size() * sizeof(float));
        } else {
            set_backend_tensor_data(mask, sliding_mask_vec.data());
        }
        return mask;
    }

    ggml_tensor* build_input_pos(int64_t seq_len, std::vector<int>& cache) {
        cache.resize(seq_len);
        for (int64_t i = 0; i < seq_len; ++i) {
            cache[i] = (int)i;
        }
        auto input_pos = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_I32, seq_len);
        set_backend_tensor_data(input_pos, cache.data());
        return input_pos;
    }

    const EncCacheEntry* get_or_create_enc_cache(int n_threads,
                                                 ggml_tensor* context,
                                                 ggml_tensor* lyric_embed,
                                                 ggml_tensor* refer_audio) {
        for (const auto& entry : enc_cache) {
            if (entry.context == context && entry.lyric == lyric_embed && entry.refer == refer_audio) {
                return &entry;
            }
        }
        if (!cpu_encoder || !context || !lyric_embed || !refer_audio) {
            return nullptr;
        }

        ggml_context* out_ctx = nullptr;
        {
            struct ggml_init_params params;
            params.mem_size   = static_cast<size_t>(256 * 1024 * 1024);
            params.mem_buffer = nullptr;
            params.no_alloc   = false;
            out_ctx = ggml_init(params);
        }
        if (!out_ctx) {
            LOG_ERROR("ACE CPU encoder: failed to allocate output context");
            return nullptr;
        }

        ggml_tensor* out = nullptr;
        if (!cpu_encoder->compute(n_threads, context, lyric_embed, refer_audio, &out, out_ctx) || out == nullptr) {
            LOG_ERROR("ACE CPU encoder: compute failed");
            ggml_free(out_ctx);
            return nullptr;
        }
        if (out->data == nullptr) {
            LOG_ERROR("ACE CPU encoder: output buffer is null (insufficient output context memory)");
            ggml_free(out_ctx);
            return nullptr;
        }

        EncCacheEntry entry;
        entry.context = context;
        entry.lyric   = lyric_embed;
        entry.refer   = refer_audio;
        entry.n_dims  = ggml_n_dims(out);
        entry.ne[0]   = out->ne[0];
        entry.ne[1]   = out->ne[1];
        entry.ne[2]   = out->ne[2];
        entry.ne[3]   = out->ne[3];

        int64_t n_elem = ggml_nelements(out);
        entry.data.resize(static_cast<size_t>(n_elem));
        const float* src = (const float*)out->data;
        if (n_elem > 0) {
            memcpy(entry.data.data(), src, sizeof(float) * n_elem);
        }

        ggml_free(out_ctx);
        enc_cache.push_back(std::move(entry));
        return &enc_cache.back();
    }

    struct ggml_cgraph* build_graph(ggml_tensor* x,
                                    ggml_tensor* timesteps,
                                    ggml_tensor* context,
                                    ggml_tensor* lyric_embed,
                                    ggml_tensor* refer_audio,
                                    const std::shared_ptr<std::vector<int>>& audio_codes) {
        struct ggml_cgraph* gf = new_graph_custom(ACE_GRAPH_SIZE);

        x         = to_backend(x);
        timesteps = to_backend(timesteps);
        ggml_tensor* enc_cached = nullptr;
        if (active_enc_cache != nullptr && active_enc_cache->n_dims > 0 && !active_enc_cache->data.empty()) {
            enc_cached = ggml_new_tensor(compute_ctx,
                                         GGML_TYPE_F32,
                                         active_enc_cache->n_dims,
                                         active_enc_cache->ne);
            ggml_set_name(enc_cached, "ace_enc_hidden_cached");
            set_backend_tensor_data(enc_cached, active_enc_cache->data.data());
        }
        bool use_cached = (enc_cached != nullptr);
        if (use_cached) {
            context     = enc_cached;
            lyric_embed = nullptr;
            refer_audio = nullptr;
        } else {
            context = to_backend(context);
        }

        if (lyric_embed) {
            lyric_embed = to_backend(lyric_embed);
        }
        if (refer_audio) {
            refer_audio = to_backend(refer_audio);
        }

        int64_t seq_len = x->ne[1];
        int64_t patch_len = (seq_len + model.patch_size - 1) / model.patch_size;
        auto decoder_pos = build_input_pos(patch_len, input_pos_vec);

        ggml_tensor* lyric_pos = nullptr;
        ggml_tensor* timbre_pos = nullptr;
        if (!use_cached) {
            int64_t lyric_len = lyric_embed ? lyric_embed->ne[1] : 1;
            lyric_pos = build_input_pos(lyric_len, lyric_pos_vec);

            int64_t timbre_len = refer_audio ? refer_audio->ne[1] : 1;
            timbre_pos = build_input_pos(timbre_len, timbre_pos_vec);
        }

        ggml_tensor* tokenizer_pos = nullptr;
        if (model.use_tokenizer_path) {
            tokenizer_pos = build_input_pos(model.pool_window_size + 1, tokenizer_pos_vec);
        }
        auto detok_pos = build_input_pos(model.pool_window_size, detok_pos_vec);

        auto sliding_mask = build_mask(patch_len, 128);

        auto runner_ctx = get_context();
        ggml_tensor* out = model.forward(&runner_ctx,
                                         x,
                                         timesteps,
                                         context,
                                         lyric_embed,
                                         refer_audio,
                                         audio_codes,
                                         decoder_pos,
                                         lyric_pos,
                                         timbre_pos,
                                         tokenizer_pos,
                                         detok_pos,
                                         sliding_mask);

        ggml_build_forward_expand(gf, out);
        return gf;
    }

    bool compute(int n_threads,
                 ggml_tensor* x,
                 ggml_tensor* timesteps,
                 ggml_tensor* context,
                 ggml_tensor* lyric_embed,
                 ggml_tensor* refer_audio,
                 const std::shared_ptr<std::vector<int>>& audio_codes,
                 ggml_tensor** output = nullptr,
                 ggml_context* output_ctx = nullptr) {
        active_enc_cache = nullptr;
        if (use_cpu_encoder) {
            active_enc_cache = get_or_create_enc_cache(n_threads, context, lyric_embed, refer_audio);
        }
        auto get_graph = [&]() -> ggml_cgraph* {
            return build_graph(x, timesteps, context, lyric_embed, refer_audio, audio_codes);
        };
        return GGMLRunner::compute(get_graph, n_threads, false, output, output_ctx);
    }
};

}  // namespace ACE

#endif  // __ACE_HPP__
