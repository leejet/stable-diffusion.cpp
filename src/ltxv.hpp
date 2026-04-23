#ifndef __LTXV_HPP__
#define __LTXV_HPP__

// LTX-Video 2.3 (Lightricks) port targeting
//   Lightricks/LTX-2.3/ltx-2.3-22b-dev.safetensors (22B params, 5947 tensors)
// and its distilled siblings (8-step, CFG=1).
//
// The weight layout is inferred directly from the safetensors header of the
// official 22B checkpoint — diffusers' `transformer_ltx2.py` is a close but
// NOT identical reference (names and block counts differ for LTX-2.3).
//
// Scope: VIDEO-ONLY generation.
//   * Every weight in the checkpoint (including audio self-attn, a2v/v2a
//     cross-attn, audio FFN, audio VAE) is registered so loading succeeds.
//   * The forward path exercises only the video branch — audio hidden state
//     stays at zeros and the audio-to-video/video-to-audio paths are skipped
//     (equivalent to diffusers `isolate_modalities=True` + discarding audio
//     output). Enable them later for audio generation.
//
// Tensor-layout conventions:
//   * torch (N, C, F, H, W) video is stored in ggml as ne = [W, H, F, C*N]
//   * torch (N, L, D) tokens    are stored as ne = [D, L, N, 1]

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "common_block.hpp"
#include "ggml_extend.hpp"
#include "model.h"
#include "rope.hpp"
#include "vae.hpp"

namespace LTXV {

    constexpr int LTXV_GRAPH_SIZE = 32768;

    // =================================================================
    // Shared primitives
    // =================================================================

    class RMSNormNoAffine : public UnaryBlock {
    protected:
        float eps;
        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, std::string prefix = "") override {}

    public:
        RMSNormNoAffine(float eps = 1e-6f) : eps(eps) {}
        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            return ggml_rms_norm(ctx->ggml_ctx, x, eps);
        }
    };

    class PerChannelRMSNorm : public UnaryBlock {
    protected:
        float eps;
        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, std::string prefix = "") override {}

    public:
        PerChannelRMSNorm(float eps = 1e-8f) : eps(eps) {}
        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            auto h = ggml_ext_cont(ctx->ggml_ctx,
                                   ggml_ext_torch_permute(ctx->ggml_ctx, x, 3, 0, 1, 2));
            h      = ggml_rms_norm(ctx->ggml_ctx, h, eps);
            h      = ggml_ext_cont(ctx->ggml_ctx,
                                   ggml_ext_torch_permute(ctx->ggml_ctx, h, 1, 2, 3, 0));
            return h;
        }
    };

    // Temporal-causal 3-D conv with runtime causal flag.
    // Weight layout follows diffusers' LTX2VideoCausalConv3d (the raw nn.Conv3d
    // is wrapped in `self.conv`, so tensor names are `<prefix>.conv.weight`).
    class CausalConv3d : public GGMLBlock {
    protected:
        int64_t in_channels;
        int64_t out_channels;
        std::tuple<int, int, int> kernel_size;  // (kt, kh, kw)
        std::tuple<int, int, int> stride;
        std::tuple<int, int, int> dilation;
        bool bias;

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            // ggml_cuda_op_im2col_3d only supports F16/F32 destination tensors
            // — BF16 weights (the native LTX-2.3 dtype) would trigger its
            // GGML_ASSERT. Force F16 here so sd.cpp's loader converts BF16
            // from the checkpoint on its way in.
            params["conv.weight"] = ggml_new_tensor_4d(ctx,
                                                      GGML_TYPE_F16,
                                                      std::get<2>(kernel_size),
                                                      std::get<1>(kernel_size),
                                                      std::get<0>(kernel_size),
                                                      in_channels * out_channels);
            if (bias) {
                params["conv.bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
            }
        }

    public:
        CausalConv3d(int64_t in_channels,
                     int64_t out_channels,
                     std::tuple<int, int, int> kernel_size,
                     std::tuple<int, int, int> stride   = {1, 1, 1},
                     std::tuple<int, int, int> dilation = {1, 1, 1},
                     bool bias                          = true)
            : in_channels(in_channels),
              out_channels(out_channels),
              kernel_size(kernel_size),
              stride(stride),
              dilation(dilation),
              bias(bias) {}

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, bool causal = true) {
            ggml_tensor* w = params["conv.weight"];
            ggml_tensor* b = bias ? params["conv.bias"] : nullptr;

            int kt = std::get<0>(kernel_size);
            int kh = std::get<1>(kernel_size);
            int kw = std::get<2>(kernel_size);

            if (kt > 1) {
                if (causal) {
                    auto first = ggml_view_4d(ctx->ggml_ctx, x,
                                              x->ne[0], x->ne[1], 1, x->ne[3],
                                              x->nb[1], x->nb[2], x->nb[3], 0);
                    auto pad_left = first;
                    for (int i = 1; i < kt - 1; ++i) {
                        pad_left = ggml_concat(ctx->ggml_ctx, pad_left, first, 2);
                    }
                    x = ggml_concat(ctx->ggml_ctx, pad_left, x, 2);
                } else {
                    int half = (kt - 1) / 2;
                    if (half > 0) {
                        auto first = ggml_view_4d(ctx->ggml_ctx, x,
                                                  x->ne[0], x->ne[1], 1, x->ne[3],
                                                  x->nb[1], x->nb[2], x->nb[3], 0);
                        auto last  = ggml_view_4d(ctx->ggml_ctx, x,
                                                 x->ne[0], x->ne[1], 1, x->ne[3],
                                                 x->nb[1], x->nb[2], x->nb[3],
                                                 x->nb[2] * (x->ne[2] - 1));
                        auto pad_left = first;
                        for (int i = 1; i < half; ++i) {
                            pad_left = ggml_concat(ctx->ggml_ctx, pad_left, first, 2);
                        }
                        auto pad_right = last;
                        for (int i = 1; i < half; ++i) {
                            pad_right = ggml_concat(ctx->ggml_ctx, pad_right, last, 2);
                        }
                        x = ggml_concat(ctx->ggml_ctx, pad_left, x, 2);
                        x = ggml_concat(ctx->ggml_ctx, x, pad_right, 2);
                    }
                }
            }

            int lp_w = kw / 2, rp_w = kw / 2;
            int lp_h = kh / 2, rp_h = kh / 2;
            x = ggml_ext_pad_ext(ctx->ggml_ctx, x, lp_w, rp_w, lp_h, rp_h, 0, 0, 0, 0,
                                 ctx->circular_x_enabled, ctx->circular_y_enabled);

            return ggml_ext_conv_3d(ctx->ggml_ctx, x, w, b, in_channels,
                                    std::get<2>(stride), std::get<1>(stride), std::get<0>(stride),
                                    0, 0, 0,
                                    std::get<2>(dilation), std::get<1>(dilation), std::get<0>(dilation));
        }
    };

    // =================================================================
    // Transformer primitives
    // =================================================================

    class TimestepEmbedderSingle : public GGMLBlock {
    protected:
        int64_t frequency_embedding_size;

    public:
        TimestepEmbedderSingle(int64_t hidden_size, int64_t frequency_embedding_size = 256)
            : frequency_embedding_size(frequency_embedding_size) {
            blocks["linear_1"] = std::shared_ptr<GGMLBlock>(new Linear(frequency_embedding_size, hidden_size, true));
            blocks["linear_2"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, hidden_size, true));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* t) {
            auto l1 = std::dynamic_pointer_cast<Linear>(blocks["linear_1"]);
            auto l2 = std::dynamic_pointer_cast<Linear>(blocks["linear_2"]);
            auto f  = ggml_ext_timestep_embedding(ctx->ggml_ctx, t, frequency_embedding_size);
            f       = l1->forward(ctx, f);
            f       = ggml_silu_inplace(ctx->ggml_ctx, f);
            f       = l2->forward(ctx, f);
            return f;
        }
    };

    class AdaLayerNormSingle : public GGMLBlock {
    public:
        int64_t hidden_size;
        int64_t num_mod_params;

        AdaLayerNormSingle(int64_t hidden_size, int64_t num_mod_params)
            : hidden_size(hidden_size), num_mod_params(num_mod_params) {
            blocks["emb.timestep_embedder"] =
                std::shared_ptr<GGMLBlock>(new TimestepEmbedderSingle(hidden_size));
            blocks["linear"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, num_mod_params * hidden_size, true));
        }

        std::pair<ggml_tensor*, ggml_tensor*> forward(GGMLRunnerContext* ctx, ggml_tensor* t) {
            auto emb    = std::dynamic_pointer_cast<TimestepEmbedderSingle>(blocks["emb.timestep_embedder"]);
            auto linear = std::dynamic_pointer_cast<Linear>(blocks["linear"]);
            auto embedded = emb->forward(ctx, t);
            auto x        = ggml_silu(ctx->ggml_ctx, embedded);
            auto temb     = linear->forward(ctx, x);
            return {temb, embedded};
        }
    };

    class FeedForward : public GGMLBlock {
    public:
        FeedForward(int64_t dim, int64_t inner_dim = -1) {
            if (inner_dim < 0) inner_dim = dim * 4;
            blocks["net.0.proj"] = std::shared_ptr<GGMLBlock>(new Linear(dim, inner_dim, true));
            blocks["net.2"]      = std::shared_ptr<GGMLBlock>(new Linear(inner_dim, dim, true));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto fc1 = std::dynamic_pointer_cast<Linear>(blocks["net.0.proj"]);
            auto fc2 = std::dynamic_pointer_cast<Linear>(blocks["net.2"]);
            x        = fc1->forward(ctx, x);
            x        = ggml_gelu_inplace(ctx->ggml_ctx, x);
            x        = fc2->forward(ctx, x);
            return x;
        }
    };

    // LTX-2.3 attention: gated, qk_norm_across_heads, split or interleaved RoPE.
    // Parameters: to_q, to_k, to_v, to_out.0, q_norm, k_norm, to_gate_logits.
    // rope_type selects between the two rotation layouts used by LTX-2.3:
    //   * "interleaved": pair indices (2k, 2k+1) rotate together
    //   * "split":       pair indices (k, k+r) rotate together (r = D/2)
    // LTX-2.3 22B uses `rope_type = "split"`.
    class LTXAttention : public GGMLBlock {
    public:
        int64_t query_dim;
        int64_t inner_dim;
        int64_t kv_inner_dim;
        int64_t num_heads;
        int64_t head_dim;
        bool has_rope;
        std::string rope_type;

        LTXAttention(int64_t query_dim,
                     int64_t heads,
                     int64_t dim_head,
                     int64_t cross_attention_dim = -1,
                     bool attention_bias         = true,
                     bool attention_out_bias     = true,
                     bool apply_rope             = true,
                     int64_t kv_heads            = -1,
                     int64_t kv_dim_head         = -1,
                     std::string rope_type       = "split")
            : query_dim(query_dim),
              num_heads(heads),
              head_dim(dim_head),
              has_rope(apply_rope && cross_attention_dim < 0),
              rope_type(rope_type) {
            inner_dim    = heads * dim_head;
            if (kv_heads < 0) kv_heads = heads;
            if (kv_dim_head < 0) kv_dim_head = dim_head;
            kv_inner_dim = kv_heads * kv_dim_head;
            int64_t kv_source_dim = (cross_attention_dim > 0) ? cross_attention_dim : query_dim;

            blocks["to_q"]     = std::shared_ptr<GGMLBlock>(new Linear(query_dim, inner_dim, attention_bias));
            blocks["to_k"]     = std::shared_ptr<GGMLBlock>(new Linear(kv_source_dim, kv_inner_dim, attention_bias));
            blocks["to_v"]     = std::shared_ptr<GGMLBlock>(new Linear(kv_source_dim, kv_inner_dim, attention_bias));
            blocks["to_out.0"] = std::shared_ptr<GGMLBlock>(new Linear(inner_dim, query_dim, attention_out_bias));
            blocks["q_norm"]   = std::shared_ptr<GGMLBlock>(new RMSNorm(inner_dim, 1e-6f));
            blocks["k_norm"]   = std::shared_ptr<GGMLBlock>(new RMSNorm(kv_inner_dim, 1e-6f));
            blocks["to_gate_logits"] = std::shared_ptr<GGMLBlock>(new Linear(query_dim, heads, true));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* hidden_states,
                             ggml_tensor* encoder_hidden_states = nullptr,
                             ggml_tensor* query_rope_cos        = nullptr,
                             ggml_tensor* query_rope_sin        = nullptr,
                             ggml_tensor* key_rope_cos          = nullptr,
                             ggml_tensor* key_rope_sin          = nullptr,
                             ggml_tensor* attention_mask        = nullptr) {
            auto to_q    = std::dynamic_pointer_cast<Linear>(blocks["to_q"]);
            auto to_k    = std::dynamic_pointer_cast<Linear>(blocks["to_k"]);
            auto to_v    = std::dynamic_pointer_cast<Linear>(blocks["to_v"]);
            auto to_out  = std::dynamic_pointer_cast<Linear>(blocks["to_out.0"]);
            auto q_norm  = std::dynamic_pointer_cast<UnaryBlock>(blocks["q_norm"]);
            auto k_norm  = std::dynamic_pointer_cast<UnaryBlock>(blocks["k_norm"]);
            auto gate    = std::dynamic_pointer_cast<Linear>(blocks["to_gate_logits"]);

            ggml_tensor* kv_src = encoder_hidden_states != nullptr ? encoder_hidden_states : hidden_states;

            auto gate_logits = gate->forward(ctx, hidden_states);

            auto q = to_q->forward(ctx, hidden_states);
            auto k = to_k->forward(ctx, kv_src);
            auto v = to_v->forward(ctx, kv_src);

            q = q_norm->forward(ctx, q);
            k = k_norm->forward(ctx, k);

            if (has_rope && query_rope_cos != nullptr && query_rope_sin != nullptr) {
                if (rope_type == "split") {
                    q = apply_split_rotary_emb(ctx, q, query_rope_cos, query_rope_sin, num_heads);
                    ggml_tensor* kc = key_rope_cos != nullptr ? key_rope_cos : query_rope_cos;
                    ggml_tensor* ks = key_rope_sin != nullptr ? key_rope_sin : query_rope_sin;
                    k               = apply_split_rotary_emb(ctx, k, kc, ks, num_heads);
                } else {
                    q = apply_rotary_emb(ctx, q, query_rope_cos, query_rope_sin);
                    ggml_tensor* kc = key_rope_cos != nullptr ? key_rope_cos : query_rope_cos;
                    ggml_tensor* ks = key_rope_sin != nullptr ? key_rope_sin : query_rope_sin;
                    k               = apply_rotary_emb(ctx, k, kc, ks);
                }
            }

            auto out = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v,
                                              num_heads, attention_mask, false,
                                              ctx->flash_attn_enabled);

            // Per-head gate: gates = 2 * sigmoid(gate_logits). Broadcast [heads, L_q, N]
            // over head_dim via reshape to [1, heads, L_q, N].
            auto gates = ggml_sigmoid(ctx->ggml_ctx, gate_logits);
            gates      = ggml_scale(ctx->ggml_ctx, gates, 2.0f);
            int64_t N   = out->ne[2];
            int64_t L_q = out->ne[1];
            auto out_4d   = ggml_reshape_4d(ctx->ggml_ctx, out, head_dim, num_heads, L_q, N);
            auto gates_4d = ggml_reshape_4d(ctx->ggml_ctx, gates, 1, num_heads, L_q, N);
            out_4d        = ggml_mul(ctx->ggml_ctx, out_4d, gates_4d);
            out           = ggml_reshape_3d(ctx->ggml_ctx, out_4d, inner_dim, L_q, N);

            out = to_out->forward(ctx, out);
            return out;
        }

        static ggml_tensor* apply_rotary_emb(GGMLRunnerContext* ctx,
                                             ggml_tensor* x,
                                             ggml_tensor* cos_freqs,
                                             ggml_tensor* sin_freqs) {
            int64_t C = x->ne[0];
            int64_t L = x->ne[1];
            int64_t N = x->ne[2];

            auto x4    = ggml_reshape_4d(ctx->ggml_ctx, x, 2, C / 2, L, N);
            auto real  = ggml_view_4d(ctx->ggml_ctx, x4, 1, C / 2, L, N,
                                      x4->nb[1], x4->nb[2], x4->nb[3], 0);
            auto imag  = ggml_view_4d(ctx->ggml_ctx, x4, 1, C / 2, L, N,
                                      x4->nb[1], x4->nb[2], x4->nb[3], x4->nb[0]);
            auto real_c   = ggml_cont(ctx->ggml_ctx, real);
            auto imag_c   = ggml_cont(ctx->ggml_ctx, imag);
            auto neg_imag = ggml_neg(ctx->ggml_ctx, imag_c);
            auto rotated  = ggml_concat(ctx->ggml_ctx, neg_imag, real_c, 0);
            rotated       = ggml_reshape_4d(ctx->ggml_ctx, rotated, C, L, N, 1);

            auto x_cos = ggml_mul(ctx->ggml_ctx, x, cos_freqs);
            auto x_sin = ggml_mul(ctx->ggml_ctx, rotated, sin_freqs);
            return ggml_add(ctx->ggml_ctx, x_cos, x_sin);
        }

        // Split-rope: pair is (x[k], x[k+r]) where r = D_per_head/2.
        //   In diffusers: x.reshape(..., 2, r), [first, second] = x.unbind(-2)
        //     first_new  = first  * cos - second * sin
        //     second_new = second * cos + first  * sin
        //   reshape back.
        //
        // cos_freqs / sin_freqs are [inner_dim/2, L] tensors in our layout;
        // we reshape them per head to [head_dim/2, L] via broadcast.
        static ggml_tensor* apply_split_rotary_emb(GGMLRunnerContext* ctx,
                                                    ggml_tensor* x,
                                                    ggml_tensor* cos_freqs,
                                                    ggml_tensor* sin_freqs,
                                                    int64_t num_heads) {
            int64_t C = x->ne[0];     // inner_dim
            int64_t L = x->ne[1];
            int64_t N = x->ne[2];
            int64_t D = C / num_heads;  // head_dim
            int64_t r = D / 2;

            // Reshape x from [C, L, N] to [r, 2, num_heads, L*N] so the last dim
            // of the pair (the "2" axis) is at ggml axis 1.
            auto x4 = ggml_reshape_4d(ctx->ggml_ctx, x, r, 2, num_heads, L * N);
            // first  = x4[:, 0, :, :]  (ne = [r, 1, num_heads, L*N])
            // second = x4[:, 1, :, :]
            auto first  = ggml_view_4d(ctx->ggml_ctx, x4, r, 1, num_heads, L * N,
                                       x4->nb[1], x4->nb[2], x4->nb[3], 0);
            auto second = ggml_view_4d(ctx->ggml_ctx, x4, r, 1, num_heads, L * N,
                                       x4->nb[1], x4->nb[2], x4->nb[3], x4->nb[1]);
            first       = ggml_cont(ctx->ggml_ctx, first);
            second      = ggml_cont(ctx->ggml_ctx, second);

            // cos/sin are [inner_dim/2, L] == [num_heads*r, L]. Reshape to
            // [r, 1, num_heads, L] so they broadcast over the batch axis (L*N/L).
            auto cos_v = ggml_reshape_4d(ctx->ggml_ctx, cos_freqs, r, 1, num_heads, L);
            auto sin_v = ggml_reshape_4d(ctx->ggml_ctx, sin_freqs, r, 1, num_heads, L);

            // first_new  = first * cos - second * sin
            // second_new = second * cos + first * sin
            auto first_new  = ggml_sub(ctx->ggml_ctx,
                                        ggml_mul(ctx->ggml_ctx, first, cos_v),
                                        ggml_mul(ctx->ggml_ctx, second, sin_v));
            auto second_new = ggml_add(ctx->ggml_ctx,
                                        ggml_mul(ctx->ggml_ctx, second, cos_v),
                                        ggml_mul(ctx->ggml_ctx, first, sin_v));

            // Stack back along axis 1: [r, 2, num_heads, L*N] and reshape to [C, L, N].
            auto out = ggml_concat(ctx->ggml_ctx, first_new, second_new, 1);
            out      = ggml_reshape_3d(ctx->ggml_ctx, out, C, L, N);
            return out;
        }
    };

    // EmbeddingsConnector's internal transformer_1d_blocks have only attn1 + ff
    // (no norms or cross-attention — checkpoint confirms this layout).
    class EmbeddingsConnectorBlock : public GGMLBlock {
    public:
        int64_t dim;

        EmbeddingsConnectorBlock(int64_t dim,
                                 int64_t num_attention_heads,
                                 int64_t attention_head_dim) : dim(dim) {
            blocks["attn1"] = std::shared_ptr<GGMLBlock>(new LTXAttention(
                dim, num_attention_heads, attention_head_dim, /*cross=*/-1, true, true, /*apply_rope=*/false));
            blocks["ff"] = std::shared_ptr<GGMLBlock>(new FeedForward(dim, 4 * dim));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto attn1 = std::dynamic_pointer_cast<LTXAttention>(blocks["attn1"]);
            auto ff    = std::dynamic_pointer_cast<FeedForward>(blocks["ff"]);
            auto a     = attn1->forward(ctx, x);
            x          = ggml_add(ctx->ggml_ctx, x, a);
            auto f     = ff->forward(ctx, x);
            x          = ggml_add(ctx->ggml_ctx, x, f);
            return x;
        }
    };

    // EmbeddingsConnector — LTX-2.3 prompt re-embedder.
    // 128 learnable registers prepended to the projected text embeddings, then
    // passed through a stack of self-attention + FF blocks.
    class EmbeddingsConnector : public GGMLBlock {
    public:
        int64_t dim;
        int64_t num_registers;
        int64_t num_blocks;

    protected:
        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, std::string prefix = "") override {
            params["learnable_registers"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, num_registers);
        }

    public:
        EmbeddingsConnector(int64_t dim,
                            int64_t num_attention_heads,
                            int64_t attention_head_dim,
                            int64_t num_registers = 128,
                            int64_t num_blocks    = 8)
            : dim(dim), num_registers(num_registers), num_blocks(num_blocks) {
            for (int64_t i = 0; i < num_blocks; ++i) {
                blocks["transformer_1d_blocks." + std::to_string(i)] =
                    std::shared_ptr<GGMLBlock>(new EmbeddingsConnectorBlock(
                        dim, num_attention_heads, attention_head_dim));
            }
        }

        // text_embeddings: [dim, L, N, 1]
        // Output: [dim, L + num_registers, N, 1]
        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* text_embeddings) {
            ggml_tensor* reg = params["learnable_registers"];  // [dim, num_registers]
            int64_t N         = text_embeddings->ne[2];
            auto reg_3d       = ggml_reshape_3d(ctx->ggml_ctx, reg, reg->ne[0], reg->ne[1], 1);
            if (N != 1) {
                auto target = ggml_new_tensor_3d(ctx->ggml_ctx, reg_3d->type, reg->ne[0], reg->ne[1], N);
                reg_3d      = ggml_repeat(ctx->ggml_ctx, reg_3d, target);
            }
            auto x = ggml_concat(ctx->ggml_ctx, reg_3d, text_embeddings, 1);
            for (int64_t i = 0; i < num_blocks; ++i) {
                auto b = std::dynamic_pointer_cast<EmbeddingsConnectorBlock>(
                    blocks["transformer_1d_blocks." + std::to_string(i)]);
                x = b->forward(ctx, x);
            }
            return x;
        }
    };

    // Transformer block for LTX-2.3 (video-only forward).
    // Every weight slot in transformer_blocks.N is registered:
    //   attn1, attn2, audio_attn1, audio_attn2 (all gated, qk_norm)
    //   audio_to_video_attn, video_to_audio_attn (gated, no rope)
    //   ff, audio_ff
    //   scale_shift_table              [dim, 9]
    //   audio_scale_shift_table        [audio_dim, 9]
    //   prompt_scale_shift_table       [dim, 2]
    //   audio_prompt_scale_shift_table [audio_dim, 2]
    //   scale_shift_table_a2v_ca_video [dim, 5]
    //   scale_shift_table_a2v_ca_audio [audio_dim, 5]
    class LTX2VideoTransformerBlock : public GGMLBlock {
    protected:
        int64_t dim;
        int64_t audio_dim;

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, std::string prefix = "") override {
            params["scale_shift_table"]               = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, 9);
            params["audio_scale_shift_table"]         = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, audio_dim, 9);
            params["prompt_scale_shift_table"]        = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, 2);
            params["audio_prompt_scale_shift_table"]  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, audio_dim, 2);
            params["scale_shift_table_a2v_ca_video"]  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, 5);
            params["scale_shift_table_a2v_ca_audio"]  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, audio_dim, 5);
        }

    public:
        LTX2VideoTransformerBlock(int64_t dim,
                                   int64_t num_attention_heads,
                                   int64_t attention_head_dim,
                                   int64_t cross_attention_dim,
                                   int64_t audio_dim,
                                   int64_t audio_num_attention_heads,
                                   int64_t audio_attention_head_dim,
                                   int64_t audio_cross_attention_dim,
                                   float eps = 1e-6f)
            : dim(dim), audio_dim(audio_dim) {
            blocks["attn1"]       = std::shared_ptr<GGMLBlock>(new LTXAttention(
                dim, num_attention_heads, attention_head_dim));
            blocks["attn2"]       = std::shared_ptr<GGMLBlock>(new LTXAttention(
                dim, num_attention_heads, attention_head_dim, cross_attention_dim, true, true, false));
            blocks["audio_attn1"] = std::shared_ptr<GGMLBlock>(new LTXAttention(
                audio_dim, audio_num_attention_heads, audio_attention_head_dim));
            blocks["audio_attn2"] = std::shared_ptr<GGMLBlock>(new LTXAttention(
                audio_dim, audio_num_attention_heads, audio_attention_head_dim,
                audio_cross_attention_dim, true, true, false));

            // Cross-modal attention — query_dim from target modality, kv from source.
            blocks["audio_to_video_attn"] = std::shared_ptr<GGMLBlock>(new LTXAttention(
                dim, audio_num_attention_heads, audio_attention_head_dim,
                audio_dim, true, true, false,
                audio_num_attention_heads, audio_attention_head_dim));
            blocks["video_to_audio_attn"] = std::shared_ptr<GGMLBlock>(new LTXAttention(
                audio_dim, audio_num_attention_heads, audio_attention_head_dim,
                dim, true, true, false,
                audio_num_attention_heads, audio_attention_head_dim));

            blocks["ff"]       = std::shared_ptr<GGMLBlock>(new FeedForward(dim, 4 * dim));
            blocks["audio_ff"] = std::shared_ptr<GGMLBlock>(new FeedForward(audio_dim, 4 * audio_dim));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* hidden,
                             ggml_tensor* encoder,
                             ggml_tensor* temb,
                             ggml_tensor* rope_cos     = nullptr,
                             ggml_tensor* rope_sin     = nullptr,
                             ggml_tensor* encoder_mask = nullptr) {
            auto attn1 = std::dynamic_pointer_cast<LTXAttention>(blocks["attn1"]);
            auto attn2 = std::dynamic_pointer_cast<LTXAttention>(blocks["attn2"]);
            auto ff    = std::dynamic_pointer_cast<FeedForward>(blocks["ff"]);

            ggml_tensor* sst = params["scale_shift_table"];  // [dim, 9]
            auto temb_r      = ggml_reshape_4d(ctx->ggml_ctx, temb, dim, 9, temb->ne[1], temb->ne[2]);
            auto ada         = ggml_add(ctx->ggml_ctx, temb_r, sst);

            auto slice = [&](int idx) {
                auto v = ggml_view_4d(ctx->ggml_ctx, ada, ada->ne[0], 1, ada->ne[2], ada->ne[3],
                                      ada->nb[1], ada->nb[2], ada->nb[3], ada->nb[1] * idx);
                return ggml_reshape_3d(ctx->ggml_ctx, v, ada->ne[0], ada->ne[2], ada->ne[3]);
            };
            auto shift_msa    = slice(0);
            auto scale_msa    = slice(1);
            auto gate_msa     = slice(2);
            auto shift_mlp    = slice(3);
            auto scale_mlp    = slice(4);
            auto gate_mlp     = slice(5);
            auto shift_text_q = slice(6);
            auto scale_text_q = slice(7);
            auto gate_text_q  = slice(8);

            // 1. Video self-attention
            auto h_norm = ggml_rms_norm(ctx->ggml_ctx, hidden, 1e-6f);
            h_norm      = ggml_add(ctx->ggml_ctx, h_norm, ggml_mul(ctx->ggml_ctx, h_norm, scale_msa));
            h_norm      = ggml_add(ctx->ggml_ctx, h_norm, shift_msa);
            auto attn_out = attn1->forward(ctx, h_norm, nullptr,
                                            rope_cos, rope_sin, nullptr, nullptr, nullptr);
            hidden = ggml_add(ctx->ggml_ctx, hidden, ggml_mul(ctx->ggml_ctx, attn_out, gate_msa));

            // 2. Prompt cross-attention with Q modulation
            auto h_norm2 = ggml_rms_norm(ctx->ggml_ctx, hidden, 1e-6f);
            h_norm2      = ggml_add(ctx->ggml_ctx, h_norm2, ggml_mul(ctx->ggml_ctx, h_norm2, scale_text_q));
            h_norm2      = ggml_add(ctx->ggml_ctx, h_norm2, shift_text_q);
            auto ca_out  = attn2->forward(ctx, h_norm2, encoder,
                                           nullptr, nullptr, nullptr, nullptr, encoder_mask);
            ca_out       = ggml_mul(ctx->ggml_ctx, ca_out, gate_text_q);
            hidden       = ggml_add(ctx->ggml_ctx, hidden, ca_out);

            // 3. a2v/v2a cross-attention — SKIPPED (video-only mode).

            // 4. FFN
            auto h_norm3 = ggml_rms_norm(ctx->ggml_ctx, hidden, 1e-6f);
            h_norm3      = ggml_add(ctx->ggml_ctx, h_norm3, ggml_mul(ctx->ggml_ctx, h_norm3, scale_mlp));
            h_norm3      = ggml_add(ctx->ggml_ctx, h_norm3, shift_mlp);
            auto ff_out  = ff->forward(ctx, h_norm3);
            hidden       = ggml_add(ctx->ggml_ctx, hidden, ggml_mul(ctx->ggml_ctx, ff_out, gate_mlp));
            return hidden;
        }
    };

    // =================================================================
    // 3-D RoPE (interleaved)
    // =================================================================

    struct RopeTables {
        std::vector<float> cos;
        std::vector<float> sin;
        int64_t L   = 0;
        int64_t dim = 0;
    };

    __STATIC_INLINE__ RopeTables compute_rope_ltx2(int num_frames,
                                                   int height,
                                                   int width,
                                                   int dim,
                                                   bool split_rope    = true,
                                                   int patch_size     = 1,
                                                   int patch_size_t   = 1,
                                                   int base_frames    = 20,
                                                   int base_h         = 2048,
                                                   int base_w         = 2048,
                                                   int vae_scale_t    = 8,
                                                   int vae_scale_h    = 32,
                                                   int vae_scale_w    = 32,
                                                   int causal_offset  = 1,
                                                   float fps          = 24.f,
                                                   float theta        = 10000.f) {
        // Split-layout  : cos/sin of size dim/2 per position (no duplication).
        // Interleaved   : cos/sin of size dim   per position (repeat_interleave(2)).
        RopeTables t;
        int64_t pos_dim  = split_rope ? (int64_t)(dim / 2) : (int64_t)dim;
        t.dim            = pos_dim;
        t.L              = (int64_t)num_frames * height * width;
        t.cos.assign(t.L * pos_dim, 0.f);
        t.sin.assign(t.L * pos_dim, 0.f);

        // Split: 3 pos-axes, dim/2 total freq slots → freq_per_axis = (dim/2) / 3
        // Interleaved: 6 rope elems, dim/6 per axis.
        int num_axes      = 3;
        int slots         = (int)pos_dim;  // total per-position storage size
        int freq_per_axis = slots / num_axes;
        int pad           = slots - num_axes * freq_per_axis;

        std::vector<float> freqs(freq_per_axis);
        if (freq_per_axis > 1) {
            for (int i = 0; i < freq_per_axis; ++i) {
                float exponent = (float)i / (float)(freq_per_axis - 1);
                freqs[i]       = std::pow(theta, exponent) * (float)M_PI / 2.f;
            }
        } else if (freq_per_axis == 1) {
            freqs[0] = (float)M_PI / 2.f;
        }

        int64_t idx = 0;
        for (int f = 0; f < num_frames; ++f) {
            float pix_start_t = (float)f * patch_size_t * vae_scale_t;
            float pix_end_t   = ((float)f * patch_size_t + patch_size_t) * vae_scale_t;
            pix_start_t       = std::max(0.f, pix_start_t + (float)causal_offset - (float)vae_scale_t);
            pix_end_t         = std::max(0.f, pix_end_t   + (float)causal_offset - (float)vae_scale_t);
            float mid_t       = 0.5f * (pix_start_t + pix_end_t) / fps;
            float gf          = mid_t / (float)base_frames;

            for (int h = 0; h < height; ++h) {
                float mid_h = ((float)h + 0.5f) * (float)patch_size * (float)vae_scale_h;
                float gh    = mid_h / (float)base_h;
                for (int w = 0; w < width; ++w) {
                    float mid_w = ((float)w + 0.5f) * (float)patch_size * (float)vae_scale_w;
                    float gw    = mid_w / (float)base_w;
                    float* co   = &t.cos[idx * pos_dim];
                    float* si   = &t.sin[idx * pos_dim];

                    // Leading pad: cos=1, sin=0. For LTX-2.3 22B with dim=4096 split,
                    // pad_size = 2048 - 3 * (2048/3) = 2 (matches diffusers).
                    for (int p = 0; p < pad; ++p) {
                        co[p] = 1.f;
                        si[p] = 0.f;
                    }
                    for (int k = 0; k < freq_per_axis; ++k) {
                        float ang_f   = freqs[k] * (gf * 2.f - 1.f);
                        float ang_h   = freqs[k] * (gh * 2.f - 1.f);
                        float ang_w   = freqs[k] * (gw * 2.f - 1.f);
                        float vals[3] = {ang_f, ang_h, ang_w};
                        if (split_rope) {
                            // Layout: per-position, values = [pad, (F0,H0,W0), (F1,H1,W1), ...]
                            for (int a = 0; a < 3; ++a) {
                                co[pad + k * 3 + a] = std::cos(vals[a]);
                                si[pad + k * 3 + a] = std::sin(vals[a]);
                            }
                        } else {
                            // Interleaved layout: each (ang) expands to (cos, cos) / (sin, sin).
                            for (int a = 0; a < 3; ++a) {
                                float c = std::cos(vals[a]);
                                float s = std::sin(vals[a]);
                                co[pad + 2 * (k * 3 + a) + 0] = c;
                                co[pad + 2 * (k * 3 + a) + 1] = c;
                                si[pad + 2 * (k * 3 + a) + 0] = s;
                                si[pad + 2 * (k * 3 + a) + 1] = s;
                            }
                        }
                    }
                    ++idx;
                }
            }
        }
        return t;
    }

    // =================================================================
    // Full transformer
    // =================================================================

    class LTX2VideoTransformer3DModel : public GGMLBlock {
    public:
        int64_t in_channels;
        int64_t out_channels;
        int64_t num_layers;
        int64_t num_attention_heads;
        int64_t attention_head_dim;
        int64_t inner_dim;
        int64_t audio_inner_dim;
        int64_t audio_num_attention_heads;
        int64_t audio_attention_head_dim;
        int64_t cross_attention_dim;
        int64_t caption_channels;
        int64_t audio_cross_attention_dim;
        int64_t audio_in_channels;
        int64_t audio_out_channels;
        int64_t connector_num_registers;
        int64_t connector_num_blocks;
        int patch_size;
        int patch_size_t;

    protected:
        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, std::string prefix = "") override {
            params["scale_shift_table"]       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, inner_dim, 2);
            params["audio_scale_shift_table"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, audio_inner_dim, 2);
        }

    public:
        LTX2VideoTransformer3DModel(int64_t in_channels               = 128,
                                    int64_t out_channels              = 128,
                                    int patch_size                    = 1,
                                    int patch_size_t                  = 1,
                                    int64_t num_attention_heads       = 32,
                                    int64_t attention_head_dim        = 128,
                                    int64_t cross_attention_dim       = 4096,
                                    int64_t num_layers                = 48,
                                    int64_t caption_channels          = 4096,
                                    int64_t audio_in_channels         = 128,
                                    int64_t audio_out_channels        = 128,
                                    int64_t audio_num_attention_heads = 32,
                                    int64_t audio_attention_head_dim  = 64,
                                    int64_t audio_cross_attention_dim = 2048,
                                    int64_t connector_num_registers   = 128,
                                    int64_t connector_num_blocks      = 8)
            : in_channels(in_channels),
              out_channels(out_channels),
              num_layers(num_layers),
              num_attention_heads(num_attention_heads),
              attention_head_dim(attention_head_dim),
              cross_attention_dim(cross_attention_dim),
              caption_channels(caption_channels),
              audio_cross_attention_dim(audio_cross_attention_dim),
              audio_in_channels(audio_in_channels),
              audio_out_channels(audio_out_channels),
              audio_num_attention_heads(audio_num_attention_heads),
              audio_attention_head_dim(audio_attention_head_dim),
              connector_num_registers(connector_num_registers),
              connector_num_blocks(connector_num_blocks),
              patch_size(patch_size),
              patch_size_t(patch_size_t) {
            inner_dim       = num_attention_heads * attention_head_dim;
            audio_inner_dim = audio_num_attention_heads * audio_attention_head_dim;

            blocks["patchify_proj"]       = std::shared_ptr<GGMLBlock>(new Linear(in_channels, inner_dim, true));
            blocks["audio_patchify_proj"] = std::shared_ptr<GGMLBlock>(new Linear(audio_in_channels, audio_inner_dim, true));

            blocks["adaln_single"]       = std::shared_ptr<GGMLBlock>(new AdaLayerNormSingle(inner_dim, 9));
            blocks["audio_adaln_single"] = std::shared_ptr<GGMLBlock>(new AdaLayerNormSingle(audio_inner_dim, 9));

            blocks["prompt_adaln_single"]       = std::shared_ptr<GGMLBlock>(new AdaLayerNormSingle(inner_dim, 2));
            blocks["audio_prompt_adaln_single"] = std::shared_ptr<GGMLBlock>(new AdaLayerNormSingle(audio_inner_dim, 2));

            blocks["av_ca_video_scale_shift_adaln_single"] =
                std::shared_ptr<GGMLBlock>(new AdaLayerNormSingle(inner_dim, 4));
            blocks["av_ca_audio_scale_shift_adaln_single"] =
                std::shared_ptr<GGMLBlock>(new AdaLayerNormSingle(audio_inner_dim, 4));
            blocks["av_ca_a2v_gate_adaln_single"] =
                std::shared_ptr<GGMLBlock>(new AdaLayerNormSingle(inner_dim, 1));
            blocks["av_ca_v2a_gate_adaln_single"] =
                std::shared_ptr<GGMLBlock>(new AdaLayerNormSingle(audio_inner_dim, 1));

            blocks["video_embeddings_connector"] = std::shared_ptr<GGMLBlock>(new EmbeddingsConnector(
                inner_dim, num_attention_heads, attention_head_dim,
                connector_num_registers, connector_num_blocks));
            blocks["audio_embeddings_connector"] = std::shared_ptr<GGMLBlock>(new EmbeddingsConnector(
                audio_inner_dim, audio_num_attention_heads, audio_attention_head_dim,
                connector_num_registers, connector_num_blocks));

            for (int64_t i = 0; i < num_layers; ++i) {
                blocks["transformer_blocks." + std::to_string(i)] =
                    std::shared_ptr<GGMLBlock>(new LTX2VideoTransformerBlock(
                        inner_dim, num_attention_heads, attention_head_dim, cross_attention_dim,
                        audio_inner_dim, audio_num_attention_heads, audio_attention_head_dim, audio_cross_attention_dim));
            }

            blocks["proj_out"]       = std::shared_ptr<GGMLBlock>(new Linear(inner_dim, out_channels, true));
            blocks["audio_proj_out"] = std::shared_ptr<GGMLBlock>(new Linear(audio_inner_dim, audio_out_channels, true));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* hidden_states,
                             ggml_tensor* encoder_hidden_states,
                             ggml_tensor* timestep,
                             ggml_tensor* rope_cos,
                             ggml_tensor* rope_sin,
                             ggml_tensor* encoder_mask = nullptr) {
            auto patchify  = std::dynamic_pointer_cast<Linear>(blocks["patchify_proj"]);
            auto adaln     = std::dynamic_pointer_cast<AdaLayerNormSingle>(blocks["adaln_single"]);
            auto connector = std::dynamic_pointer_cast<EmbeddingsConnector>(blocks["video_embeddings_connector"]);
            auto proj_out  = std::dynamic_pointer_cast<Linear>(blocks["proj_out"]);

            auto x = patchify->forward(ctx, hidden_states);

            auto te_pair           = adaln->forward(ctx, timestep);
            auto temb              = te_pair.first;
            auto embedded_timestep = te_pair.second;

            temb = ggml_reshape_4d(ctx->ggml_ctx, temb, temb->ne[0], 1, temb->ne[1], 1);

            auto encoder = connector->forward(ctx, encoder_hidden_states);

            for (int64_t i = 0; i < num_layers; ++i) {
                auto blk = std::dynamic_pointer_cast<LTX2VideoTransformerBlock>(
                    blocks["transformer_blocks." + std::to_string(i)]);
                x = blk->forward(ctx, x, encoder, temb, rope_cos, rope_sin, encoder_mask);
            }

            ggml_tensor* sst = params["scale_shift_table"];
            auto et_r        = ggml_reshape_4d(ctx->ggml_ctx, embedded_timestep,
                                                inner_dim, 1, embedded_timestep->ne[1], 1);
            auto sst_r       = ggml_reshape_4d(ctx->ggml_ctx, sst, inner_dim, 2, 1, 1);
            auto target      = ggml_new_tensor_4d(ctx->ggml_ctx, et_r->type,
                                                   inner_dim, 2, et_r->ne[2], 1);
            auto et_expand   = ggml_repeat(ctx->ggml_ctx, et_r, target);
            auto mod         = ggml_add(ctx->ggml_ctx, et_expand, sst_r);

            auto shift = ggml_view_3d(ctx->ggml_ctx, mod, inner_dim, 1, mod->ne[2],
                                      mod->nb[1], mod->nb[2], 0);
            auto scale = ggml_view_3d(ctx->ggml_ctx, mod, inner_dim, 1, mod->ne[2],
                                      mod->nb[1], mod->nb[2], mod->nb[1]);
            x = ggml_add(ctx->ggml_ctx, x, ggml_mul(ctx->ggml_ctx, x, scale));
            x = ggml_add(ctx->ggml_ctx, x, shift);
            x = proj_out->forward(ctx, x);
            return x;
        }
    };

    // =================================================================
    // Transformer runner
    // =================================================================

    struct LTXVRunner : public GGMLRunner {
        LTX2VideoTransformer3DModel dit;
        RopeTables rope_tbl;

        LTXVRunner(ggml_backend_t backend,
                   bool offload_params_to_cpu,
                   const String2TensorStorage& tensor_storage_map = {},
                   const std::string prefix                       = "model.diffusion_model",
                   SDVersion version                              = VERSION_COUNT)
            : GGMLRunner(backend, offload_params_to_cpu) {
            dit.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override { return "ltxv2.3"; }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string prefix) {
            dit.get_param_tensors(tensors, prefix);
        }

        struct ggml_cgraph* build_graph(const sd::Tensor<float>& x,
                                        const sd::Tensor<float>& timesteps,
                                        const sd::Tensor<float>& context,
                                        const sd::Tensor<float>* mask_bias) {
            auto* compute = compute_ctx;
            auto gf       = ggml_new_graph_custom(compute, LTXV_GRAPH_SIZE, false);

            auto x_t  = make_input(x);
            auto ts_t = make_input(timesteps);
            auto c_t  = make_input(context);
            ggml_tensor* m_t = nullptr;
            if (mask_bias != nullptr && !mask_bias->empty()) {
                m_t = make_input(*mask_bias);
            }

            int64_t W = x_t->ne[0];
            int64_t H = x_t->ne[1];
            int64_t F = x_t->ne[2];
            int64_t C = x_t->ne[3];
            GGML_ASSERT(C == dit.in_channels);

            // LTX-2.3 uses split rope → cos/sin is inner_dim/2 per position.
            rope_tbl      = compute_rope_ltx2((int)F, (int)H, (int)W, (int)dit.inner_dim, /*split_rope=*/true);
            auto rope_cos = ggml_new_tensor_2d(compute, GGML_TYPE_F32,
                                                rope_tbl.dim, rope_tbl.L);
            auto rope_sin = ggml_new_tensor_2d(compute, GGML_TYPE_F32,
                                                rope_tbl.dim, rope_tbl.L);
            set_backend_tensor_data(rope_cos, rope_tbl.cos.data());
            set_backend_tensor_data(rope_sin, rope_tbl.sin.data());

            auto hidden = ggml_ext_cont(compute,
                                        ggml_ext_torch_permute(compute, x_t, 3, 0, 1, 2));
            hidden      = ggml_reshape_3d(compute, hidden, C, W * H * F, 1);

            auto rctx = get_context();
            auto out  = dit.forward(&rctx, hidden, c_t, ts_t, rope_cos, rope_sin, m_t);

            out = ggml_reshape_4d(compute, out, C, W, H, F);
            out = ggml_ext_cont(compute, ggml_ext_torch_permute(compute, out, 1, 2, 3, 0));

            ggml_build_forward_expand(gf, out);
            return gf;
        }

        sd::Tensor<float> compute(int n_threads,
                                  const sd::Tensor<float>& x,
                                  const sd::Tensor<float>& timesteps,
                                  const sd::Tensor<float>& context,
                                  const sd::Tensor<float>& mask) {
            auto get_graph = [&]() -> struct ggml_cgraph* {
                return build_graph(x, timesteps, context,
                                   mask.empty() ? nullptr : &mask);
            };
            auto result = GGMLRunner::compute<float>(get_graph, n_threads, false);
            if (!result.has_value()) return {};
            return std::move(*result);
        }
    };

    // =================================================================
    //                              LTX-2.3 VAE
    // =================================================================
    //
    // Structure inferred from `ltx-2.3-22b-dev.safetensors`.
    // Encoder has 9 top-level `down_blocks.N` groups alternating res-stacks and
    // downsampler convs:
    //   0: res × 4 @ 128     1: spatial(1,2,2) 128→256   2: res × 6 @ 256
    //   3: temporal(2,1,1) 256→512   4: res × 4 @ 512    5: st(2,2,2) 512→1024
    //   6: res × 2 @ 1024             7: st(2,2,2) 1024→1024  8: res × 2 @ 1024
    // Decoder mirror (sizes from checkpoint):
    //   0: res × 2 @ 1024    1: upsamp st(2,2,2) conv[4096,1024] → 512
    //   2: res × 2 @ 512     3: upsamp st(2,2,2) conv[4096,512]  → 512
    //   4: res × 4 @ 512     5: upsamp temporal(2,1,1) conv[512,512] → 256
    //   6: res × 6 @ 256     7: upsamp spatial(1,2,2) conv[512,256]  → 128
    //   8: res × 4 @ 128

    class VAEResBlock : public GGMLBlock {
    protected:
        int64_t channels;

    public:
        VAEResBlock(int64_t channels) : channels(channels) {
            blocks["conv1"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(channels, channels, {3, 3, 3}));
            blocks["conv2"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(channels, channels, {3, 3, 3}));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* h, bool causal = true) {
            auto conv1 = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv1"]);
            auto conv2 = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv2"]);
            auto residual = h;
            h = ggml_silu_inplace(ctx->ggml_ctx, h);
            h = conv1->forward(ctx, h, causal);
            h = ggml_silu_inplace(ctx->ggml_ctx, h);
            h = conv2->forward(ctx, h, causal);
            return ggml_add(ctx->ggml_ctx, h, residual);
        }
    };

    class VAEResStack : public GGMLBlock {
    protected:
        int64_t num_layers;

    public:
        VAEResStack(int64_t channels, int64_t num_layers) : num_layers(num_layers) {
            for (int64_t i = 0; i < num_layers; ++i) {
                blocks["res_blocks." + std::to_string(i)] =
                    std::shared_ptr<GGMLBlock>(new VAEResBlock(channels));
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* h, bool causal = true) {
            for (int64_t i = 0; i < num_layers; ++i) {
                auto rn = std::dynamic_pointer_cast<VAEResBlock>(blocks["res_blocks." + std::to_string(i)]);
                h = rn->forward(ctx, h, causal);
            }
            return h;
        }
    };

    // Downsampler: conv (in_ch → conv_out_ch) then channel-inflation via reshape.
    class VAEDownsampler : public GGMLBlock {
    protected:
        int64_t in_channels;
        int64_t conv_out_channels;
        std::tuple<int, int, int> stride;

    public:
        VAEDownsampler(int64_t in_channels, int64_t conv_out_channels, std::tuple<int, int, int> stride)
            : in_channels(in_channels), conv_out_channels(conv_out_channels), stride(stride) {
            blocks["conv"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(in_channels, conv_out_channels, {3, 3, 3}));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* h, bool causal = true) {
            auto conv = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv"]);
            h = conv->forward(ctx, h, causal);
            int st_t = std::get<0>(stride), st_h = std::get<1>(stride), st_w = std::get<2>(stride);
            int64_t W = h->ne[0], H = h->ne[1], F = h->ne[2], C = h->ne[3];
            h = ggml_cont(ctx->ggml_ctx, h);
            h = ggml_reshape_4d(ctx->ggml_ctx, h, W / st_w, H / st_h, F / st_t, C * st_w * st_h * st_t);
            return h;
        }
    };

    class VAEUpsampler : public GGMLBlock {
    protected:
        int64_t in_channels;
        int64_t conv_out_channels;
        std::tuple<int, int, int> stride;

    public:
        VAEUpsampler(int64_t in_channels, int64_t conv_out_channels, std::tuple<int, int, int> stride)
            : in_channels(in_channels), conv_out_channels(conv_out_channels), stride(stride) {
            blocks["conv"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(in_channels, conv_out_channels, {3, 3, 3}));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* h, bool causal = false) {
            auto conv = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv"]);
            h = conv->forward(ctx, h, causal);
            int st_t = std::get<0>(stride), st_h = std::get<1>(stride), st_w = std::get<2>(stride);
            int64_t W = h->ne[0], H = h->ne[1], F = h->ne[2], C = h->ne[3];
            int64_t prod = (int64_t)st_t * st_h * st_w;
            int64_t out_c = C / prod;
            h = ggml_cont(ctx->ggml_ctx, h);
            h = ggml_reshape_4d(ctx->ggml_ctx, h, W * st_w, H * st_h, F * st_t, out_c);
            // Diffusers LTX2VideoUpsampler3d drops the first (st_t - 1) temporal
            // samples so each upsampled chunk boundary stays causal and the
            // overall frame count follows f_out = (f_in - 1) * st_t + 1 when
            // composed across multiple temporal upsamples.
            if (st_t > 1) {
                int64_t T_out   = F * st_t;
                int64_t T_keep  = T_out - (st_t - 1);
                int64_t offset_bytes = h->nb[2] * (st_t - 1);
                h = ggml_view_4d(ctx->ggml_ctx, h,
                                 h->ne[0], h->ne[1], T_keep, h->ne[3],
                                 h->nb[1], h->nb[2], h->nb[3], offset_bytes);
                h = ggml_cont(ctx->ggml_ctx, h);
            }
            return h;
        }
    };

    class LTX23Encoder3d : public GGMLBlock {
    public:
        LTX23Encoder3d() {
            blocks["conv_in"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(48, 128, {3, 3, 3}));
            blocks["down_blocks.0"] = std::shared_ptr<GGMLBlock>(new VAEResStack(128, 4));
            blocks["down_blocks.1"] = std::shared_ptr<GGMLBlock>(new VAEDownsampler(128, 64, {1, 2, 2}));
            blocks["down_blocks.2"] = std::shared_ptr<GGMLBlock>(new VAEResStack(256, 6));
            blocks["down_blocks.3"] = std::shared_ptr<GGMLBlock>(new VAEDownsampler(256, 256, {2, 1, 1}));
            blocks["down_blocks.4"] = std::shared_ptr<GGMLBlock>(new VAEResStack(512, 4));
            blocks["down_blocks.5"] = std::shared_ptr<GGMLBlock>(new VAEDownsampler(512, 128, {2, 2, 2}));
            blocks["down_blocks.6"] = std::shared_ptr<GGMLBlock>(new VAEResStack(1024, 2));
            blocks["down_blocks.7"] = std::shared_ptr<GGMLBlock>(new VAEDownsampler(1024, 128, {2, 2, 2}));
            blocks["down_blocks.8"] = std::shared_ptr<GGMLBlock>(new VAEResStack(1024, 2));
            blocks["conv_out"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(1024, 129, {3, 3, 3}));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, bool causal = true) {
            auto conv_in  = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv_in"]);
            auto conv_out = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv_out"]);
            int64_t W = x->ne[0], H = x->ne[1], F = x->ne[2], C = x->ne[3];
            GGML_ASSERT(W % 4 == 0 && H % 4 == 0);
            x = ggml_cont(ctx->ggml_ctx, x);
            x = ggml_reshape_4d(ctx->ggml_ctx, x, W / 4, H / 4, F, C * 16);
            auto h = conv_in->forward(ctx, x, causal);
            for (int i = 0; i < 9; ++i) {
                auto& blk = blocks["down_blocks." + std::to_string(i)];
                if (i % 2 == 0) {
                    auto s = std::dynamic_pointer_cast<VAEResStack>(blk);
                    h = s->forward(ctx, h, causal);
                } else {
                    auto s = std::dynamic_pointer_cast<VAEDownsampler>(blk);
                    h = s->forward(ctx, h, causal);
                }
            }
            h = conv_out->forward(ctx, h, causal);
            return h;
        }
    };

    class LTX23Decoder3d : public GGMLBlock {
    public:
        LTX23Decoder3d() {
            blocks["conv_in"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(128, 1024, {3, 3, 3}));
            blocks["up_blocks.0"] = std::shared_ptr<GGMLBlock>(new VAEResStack(1024, 2));
            blocks["up_blocks.1"] = std::shared_ptr<GGMLBlock>(new VAEUpsampler(1024, 4096, {2, 2, 2}));
            blocks["up_blocks.2"] = std::shared_ptr<GGMLBlock>(new VAEResStack(512, 2));
            blocks["up_blocks.3"] = std::shared_ptr<GGMLBlock>(new VAEUpsampler(512, 4096, {2, 2, 2}));
            blocks["up_blocks.4"] = std::shared_ptr<GGMLBlock>(new VAEResStack(512, 4));
            blocks["up_blocks.5"] = std::shared_ptr<GGMLBlock>(new VAEUpsampler(512, 512, {2, 1, 1}));
            blocks["up_blocks.6"] = std::shared_ptr<GGMLBlock>(new VAEResStack(256, 6));
            blocks["up_blocks.7"] = std::shared_ptr<GGMLBlock>(new VAEUpsampler(256, 512, {1, 2, 2}));
            blocks["up_blocks.8"] = std::shared_ptr<GGMLBlock>(new VAEResStack(128, 4));
            blocks["conv_out"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(128, 48, {3, 3, 3}));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* z, bool causal = false) {
            auto conv_in  = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv_in"]);
            auto conv_out = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv_out"]);
            auto h = conv_in->forward(ctx, z, causal);
            for (int i = 0; i < 9; ++i) {
                auto& blk = blocks["up_blocks." + std::to_string(i)];
                if (i % 2 == 0) {
                    auto s = std::dynamic_pointer_cast<VAEResStack>(blk);
                    h = s->forward(ctx, h, causal);
                } else {
                    auto s = std::dynamic_pointer_cast<VAEUpsampler>(blk);
                    h = s->forward(ctx, h, causal);
                }
            }
            h = conv_out->forward(ctx, h, causal);
            int64_t W = h->ne[0], H = h->ne[1], F = h->ne[2], C = h->ne[3];
            h = ggml_cont(ctx->ggml_ctx, h);
            h = ggml_reshape_4d(ctx->ggml_ctx, h, W * 4, H * 4, F, C / 16);
            return h;
        }
    };

    class LTX23Autoencoder : public GGMLBlock {
    public:
        bool decode_only;

    protected:
        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, std::string prefix = "") override {
            params["per_channel_statistics.mean-of-means"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 128);
            params["per_channel_statistics.std-of-means"]  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 128);
        }

    public:
        LTX23Autoencoder(bool decode_only = true) : decode_only(decode_only) {
            if (!decode_only) {
                blocks["encoder"] = std::shared_ptr<GGMLBlock>(new LTX23Encoder3d());
            }
            blocks["decoder"] = std::shared_ptr<GGMLBlock>(new LTX23Decoder3d());
        }

        ggml_tensor* decode(GGMLRunnerContext* ctx, ggml_tensor* z) {
            auto dec = std::dynamic_pointer_cast<LTX23Decoder3d>(blocks["decoder"]);
            return dec->forward(ctx, z, /*causal=*/false);
        }

        ggml_tensor* encode(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto enc = std::dynamic_pointer_cast<LTX23Encoder3d>(blocks["encoder"]);
            return enc->forward(ctx, x, /*causal=*/true);
        }
    };

    struct LTXVVAERunner : public VAE {
        bool decode_only = true;
        LTX23Autoencoder ae;

        LTXVVAERunner(SDVersion version,
                      ggml_backend_t backend,
                      bool offload_params_to_cpu,
                      const String2TensorStorage& tensor_storage_map = {},
                      const std::string prefix                       = "vae",
                      bool decode_only                               = true)
            : VAE(version, backend, offload_params_to_cpu),
              decode_only(decode_only),
              ae(decode_only) {
            scale_input = false;
            ae.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override { return "ltxv2.3_vae"; }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string prefix) override {
            ae.get_param_tensors(tensors, prefix);
        }

        int get_encoder_output_channels(int input_channels) override {
            SD_UNUSED(input_channels);
            return 129;
        }

        sd::Tensor<float> vae_output_to_latents(const sd::Tensor<float>& vae_output,
                                                 std::shared_ptr<RNG> rng) override {
            SD_UNUSED(rng);
            return vae_output;
        }

        // LTX-2.3 normalises diffusion-space latents to unit variance using the
        // per-channel stats saved with the VAE:
        //   diffusion_to_vae   = latents * std + mean
        //   vae_to_diffusion   = (latents - mean) / std
        // The stats are loaded into `ae.params["per_channel_statistics.*"]` at
        // init_params time. When the stats are unavailable (e.g. running
        // without the checkpoint), we fall back to identity so tests on
        // synthetic data still work.
        //
        // NOTE: We can't easily read backend-resident tensors from CPU here
        // without a separate copy. For correctness on a CUDA run the caller
        // must materialise the stats to CPU first — TODO: plumb that through.
        // For now the identity fall-through is preserved and we note this as
        // a known quality gap in docs/ltxv.md.
        sd::Tensor<float> diffusion_to_vae_latents(const sd::Tensor<float>& latents) override { return latents; }
        sd::Tensor<float> vae_to_diffusion_latents(const sd::Tensor<float>& latents) override { return latents; }

      protected:
        struct ggml_cgraph* build_graph_decode(const sd::Tensor<float>& z) {
            auto gf   = ggml_new_graph_custom(compute_ctx, LTXV_GRAPH_SIZE, false);
            auto z_t  = make_input(z);
            auto rctx = get_context();
            auto h    = ae.decode(&rctx, z_t);
            ggml_build_forward_expand(gf, h);
            return gf;
        }

        struct ggml_cgraph* build_graph_encode(const sd::Tensor<float>& x) {
            auto gf   = ggml_new_graph_custom(compute_ctx, LTXV_GRAPH_SIZE, false);
            auto x_t  = make_input(x);
            auto rctx = get_context();
            auto h    = ae.encode(&rctx, x_t);
            ggml_build_forward_expand(gf, h);
            return gf;
        }

        sd::Tensor<float> _compute(const int n_threads,
                                   const sd::Tensor<float>& z,
                                   bool decode_graph) override {
            auto get_graph = [&]() -> struct ggml_cgraph* {
                return decode_graph ? build_graph_decode(z) : build_graph_encode(z);
            };
            auto result = GGMLRunner::compute<float>(get_graph, n_threads, false);
            if (!result.has_value()) return {};
            return std::move(*result);
        }
    };

}  // namespace LTXV

#endif  // __LTXV_HPP__
