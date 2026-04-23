#ifndef __LTXV_HPP__
#define __LTXV_HPP__

// LTX-Video 2.0 (Lightricks) port — diffusers reference:
//   src/diffusers/models/transformers/transformer_ltx2.py
//   src/diffusers/models/autoencoders/autoencoder_kl_ltx2.py
//
// Scope for this port: VIDEO-ONLY generation.
//   * All audio-related parameters (audio_proj_in, audio_time_embed, audio_caption_projection,
//     audio_rope, cross_attn_audio_rope, per-block audio_*, av_cross_attn_audio_*,
//     audio_scale_shift_table, audio_norm_out, audio_proj_out) are loaded so LTX-2
//     checkpoints open cleanly, but the forward path SKIPS audio self-attention,
//     audio cross-attention, audio-to-video and video-to-audio cross attention,
//     audio FFN, and audio output projection (equivalent to
//     `isolate_modalities=True, return audio_output=None`).
//   * Audio VAE / vocoder are not ported. Add later if audio generation is needed.
//
// Tensor-layout conventions (match wan.hpp and ltxv.hpp for LTX-1):
//   * torch (N, C, F, H, W) video is stored in ggml as ne = [W, H, F, C*N]
//   * torch (N, L, D) tokens    are stored as ne = [D, L, N, 1]
//   * permutations use ggml_ext_torch_permute (takes torch-order axes)

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

    constexpr int LTXV_GRAPH_SIZE = 20480;

    // ------------------------------------------------------------------
    // RMSNorm with no elementwise-affine weight.
    // Used for block-level norm1/norm2/norm3 in LTX-2 (elementwise_affine=False).
    // ------------------------------------------------------------------
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

    // ------------------------------------------------------------------
    // PerChannelRMSNorm — diffusers LTX-2 `PerChannelRMSNorm`.
    //   y = x / sqrt(mean(x**2, dim=channel, keepdim=True) + eps)
    // No parameters. For ggml video tensors [W, H, F, C*N], C is at ne[3],
    // so we permute C to innermost, run rms_norm (which normalises ne[0]),
    // then permute back.
    // ------------------------------------------------------------------
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

    // ------------------------------------------------------------------
    // LTX-2 CausalConv3d — temporal-causal 3-D conv with RUNTIME causal flag
    // (diffusers LTX-2 moved `causal` from constructor to forward).
    // ------------------------------------------------------------------
    class CausalConv3d : public GGMLBlock {
    protected:
        int64_t in_channels;
        int64_t out_channels;
        std::tuple<int, int, int> kernel_size;   // (kt, kh, kw)
        std::tuple<int, int, int> stride;
        std::tuple<int, int, int> dilation;
        bool bias;

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            params["weight"] = ggml_new_tensor_4d(ctx,
                                                  GGML_TYPE_F16,
                                                  std::get<2>(kernel_size),
                                                  std::get<1>(kernel_size),
                                                  std::get<0>(kernel_size),
                                                  in_channels * out_channels);
            if (bias) {
                params["bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
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
            ggml_tensor* w = params["weight"];
            ggml_tensor* b = bias ? params["bias"] : nullptr;

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

    // ==================================================================
    //                           TRANSFORMER
    // ==================================================================

    // PixArtAlphaTextProjection — caption_projection block.
    // Parameters: linear_1, linear_2.  Act: GELU(tanh).
    class CaptionProjection : public GGMLBlock {
    public:
        CaptionProjection(int64_t in_features, int64_t hidden_size) {
            blocks["linear_1"] = std::shared_ptr<GGMLBlock>(new Linear(in_features, hidden_size, true));
            blocks["linear_2"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, hidden_size, true));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* caption) {
            auto l1 = std::dynamic_pointer_cast<Linear>(blocks["linear_1"]);
            auto l2 = std::dynamic_pointer_cast<Linear>(blocks["linear_2"]);
            auto x  = l1->forward(ctx, caption);
            x       = ggml_gelu_inplace(ctx->ggml_ctx, x);
            x       = l2->forward(ctx, x);
            return x;
        }
    };

    // PixArtAlphaCombinedTimestepSizeEmbeddings — used inside LTX2AdaLayerNormSingle.
    // With `use_additional_conditions=False` (the LTX-2 setting) this collapses to
    // just the timestep projection: ts_emb → linear_1 → SiLU → linear_2.
    // Parameters: `timestep_embedder.linear_1`, `timestep_embedder.linear_2`
    //            (size_embedder tensors are not loaded when additional_conditions=False).
    class CombinedTimestepSizeEmbeddings : public GGMLBlock {
    protected:
        int64_t frequency_embedding_size;

    public:
        CombinedTimestepSizeEmbeddings(int64_t hidden_size, int64_t frequency_embedding_size = 256)
            : frequency_embedding_size(frequency_embedding_size) {
            blocks["timestep_embedder.linear_1"] =
                std::shared_ptr<GGMLBlock>(new Linear(frequency_embedding_size, hidden_size, true));
            blocks["timestep_embedder.linear_2"] =
                std::shared_ptr<GGMLBlock>(new Linear(hidden_size, hidden_size, true));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* t) {
            auto l1 = std::dynamic_pointer_cast<Linear>(blocks["timestep_embedder.linear_1"]);
            auto l2 = std::dynamic_pointer_cast<Linear>(blocks["timestep_embedder.linear_2"]);
            auto f  = ggml_ext_timestep_embedding(ctx->ggml_ctx, t, frequency_embedding_size);
            f       = l1->forward(ctx, f);
            f       = ggml_silu_inplace(ctx->ggml_ctx, f);
            f       = l2->forward(ctx, f);
            return f;
        }
    };

    // LTX2AdaLayerNormSingle(hidden, num_mod_params, use_additional_conditions=False).
    // Structure:
    //   emb    : PixArtAlphaCombinedTimestepSizeEmbeddings(hidden)
    //   linear : hidden -> num_mod_params * hidden
    // Returns (temb_modulation, embedded_timestep).
    class LTX2AdaLayerNormSingle : public GGMLBlock {
    protected:
        int64_t hidden_size;
        int64_t num_mod_params;

    public:
        LTX2AdaLayerNormSingle(int64_t hidden_size, int64_t num_mod_params = 6)
            : hidden_size(hidden_size), num_mod_params(num_mod_params) {
            blocks["emb"]    = std::shared_ptr<GGMLBlock>(new CombinedTimestepSizeEmbeddings(hidden_size));
            blocks["linear"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, num_mod_params * hidden_size, true));
        }

        std::pair<ggml_tensor*, ggml_tensor*> forward(GGMLRunnerContext* ctx, ggml_tensor* t) {
            auto emb    = std::dynamic_pointer_cast<CombinedTimestepSizeEmbeddings>(blocks["emb"]);
            auto linear = std::dynamic_pointer_cast<Linear>(blocks["linear"]);

            auto embedded_timestep = emb->forward(ctx, t);
            auto x                 = ggml_silu(ctx->ggml_ctx, embedded_timestep);
            auto temb              = linear->forward(ctx, x);
            return {temb, embedded_timestep};
        }
    };

    // FeedForward(dim, "gelu-approximate"): net.0.proj, net.2 (inner_dim = 4*dim).
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

    // LTX2 Attention — diffusers.LTX2Attention.
    // Parameters: to_q, to_k, to_v, to_out.0 (+bias), norm_q, norm_k,
    //             plus optional to_gate_logits (Linear(dim, heads)) when `apply_gated_attention=True`.
    // qk_norm = rms_norm_across_heads → weight shape = (inner_dim,).
    // rope_type ∈ { "interleaved", "split" } — interleaved matches LTX-1.
    class LTX2Attention : public GGMLBlock {
    public:
        int64_t inner_dim;
        int64_t num_heads;
        int64_t head_dim;
        bool is_cross_attn;
        bool has_rope;
        bool apply_gated_attention;
        std::string rope_type;  // "interleaved" or "split"

    public:
        LTX2Attention(int64_t query_dim,
                      int64_t heads,
                      int64_t dim_head,
                      int64_t cross_attention_dim = -1,
                      bool attention_bias         = true,
                      bool attention_out_bias     = true,
                      bool apply_gated_attention  = false,
                      std::string rope_type       = "interleaved")
            : num_heads(heads),
              head_dim(dim_head),
              apply_gated_attention(apply_gated_attention),
              rope_type(rope_type) {
            inner_dim           = heads * dim_head;
            int64_t kv_dim      = (cross_attention_dim > 0) ? cross_attention_dim : query_dim;
            is_cross_attn       = cross_attention_dim > 0;
            has_rope            = !is_cross_attn;

            blocks["to_q"]     = std::shared_ptr<GGMLBlock>(new Linear(query_dim, inner_dim, attention_bias));
            blocks["to_k"]     = std::shared_ptr<GGMLBlock>(new Linear(kv_dim, inner_dim, attention_bias));
            blocks["to_v"]     = std::shared_ptr<GGMLBlock>(new Linear(kv_dim, inner_dim, attention_bias));
            blocks["to_out.0"] = std::shared_ptr<GGMLBlock>(new Linear(inner_dim, query_dim, attention_out_bias));

            blocks["norm_q"] = std::shared_ptr<GGMLBlock>(new RMSNorm(inner_dim, 1e-6f));
            blocks["norm_k"] = std::shared_ptr<GGMLBlock>(new RMSNorm(inner_dim, 1e-6f));

            if (apply_gated_attention) {
                // Per-head gate logits.
                blocks["to_gate_logits"] = std::shared_ptr<GGMLBlock>(new Linear(query_dim, heads, true));
            }
        }

        // hidden_states         : [N, L_q, query_dim]
        // encoder_hidden_states : [N, L_k, kv_dim] (cross-attn only)
        // query_rope_cos/sin    : [L_q, inner_dim] (rope applied to Q — and K if key_rope not provided)
        // key_rope_cos/sin      : optional separate rope for K (LTX-2 a2v/v2a cross-attn)
        // attention_mask        : additive bias
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
            auto norm_q  = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_q"]);
            auto norm_k  = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_k"]);

            ggml_tensor* kv_src = encoder_hidden_states != nullptr ? encoder_hidden_states : hidden_states;

            ggml_tensor* gate_logits = nullptr;
            if (apply_gated_attention) {
                auto gate_proj = std::dynamic_pointer_cast<Linear>(blocks["to_gate_logits"]);
                gate_logits    = gate_proj->forward(ctx, hidden_states);  // [N, L_q, num_heads]
            }

            auto q = to_q->forward(ctx, hidden_states);
            auto k = to_k->forward(ctx, kv_src);
            auto v = to_v->forward(ctx, kv_src);

            q = norm_q->forward(ctx, q);
            k = norm_k->forward(ctx, k);

            if (has_rope && query_rope_cos != nullptr && query_rope_sin != nullptr) {
                q = apply_rotary_emb(ctx, q, query_rope_cos, query_rope_sin);
                ggml_tensor* kc = key_rope_cos != nullptr ? key_rope_cos : query_rope_cos;
                ggml_tensor* ks = key_rope_sin != nullptr ? key_rope_sin : query_rope_sin;
                k               = apply_rotary_emb(ctx, k, kc, ks);
            }

            auto out = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v,
                                              num_heads, attention_mask, false,
                                              ctx->flash_attn_enabled);

            if (apply_gated_attention && gate_logits != nullptr) {
                // gates = 2.0 * sigmoid(gate_logits)  — shape [N, L_q, num_heads]
                // The factor of 2.0 makes zero-init gates identity.
                auto gates = ggml_sigmoid(ctx->ggml_ctx, gate_logits);
                gates      = ggml_scale(ctx->ggml_ctx, gates, 2.0f);

                // Unflatten `out` to [N, L_q, num_heads, head_dim] and multiply by gates.
                int64_t d_head = head_dim;
                int64_t N      = out->ne[2];
                int64_t L_q    = out->ne[1];
                auto out_4d    = ggml_reshape_4d(ctx->ggml_ctx, out, d_head, num_heads, L_q, N);
                // gates is [num_heads, L_q, N] (ggml ne ordering). Reshape to
                // [1, num_heads, L_q, N] so it broadcasts over d_head.
                auto gates_4d  = ggml_reshape_4d(ctx->ggml_ctx, gates, 1, num_heads, L_q, N);
                out_4d         = ggml_mul(ctx->ggml_ctx, out_4d, gates_4d);
                out            = ggml_reshape_3d(ctx->ggml_ctx, out_4d, d_head * num_heads, L_q, N);
            }

            out = to_out->forward(ctx, out);
            return out;
        }

        // pairs-of-two rotation (interleaved RoPE).
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
    };

    // Transformer block for LTX-2 (video-only forward path).
    //
    // Load-time: every attribute present in the diffusers `LTX2VideoTransformerBlock`
    // is registered so weights load correctly — including audio_*, audio_to_video_*,
    // video_to_audio_*, and audio_* cross-attn modulation tables.
    //
    // Runtime: only the video pathway is executed (self-attn + prompt cross-attn + FF).
    // This corresponds to `isolate_modalities=True` in diffusers.
    class LTX2VideoTransformerBlock : public GGMLBlock {
    protected:
        int64_t dim;
        int64_t audio_dim;
        int64_t video_mod_params;
        int64_t audio_mod_params;
        bool video_cross_attn_adaln;
        bool audio_cross_attn_adaln;
        bool cross_attn_adaln;  // OR of the two above

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, std::string prefix = "") override {
            params["scale_shift_table"]       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, video_mod_params);
            params["audio_scale_shift_table"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, audio_dim, audio_mod_params);

            params["video_a2v_cross_attn_scale_shift_table"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, 5);
            params["audio_a2v_cross_attn_scale_shift_table"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, audio_dim, 5);

            if (cross_attn_adaln) {
                params["prompt_scale_shift_table"]       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, 2);
                params["audio_prompt_scale_shift_table"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, audio_dim, 2);
            }
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
                                   bool video_gated_attn        = false,
                                   bool video_cross_attn_adaln  = false,
                                   bool audio_gated_attn        = false,
                                   bool audio_cross_attn_adaln  = false,
                                   bool attention_bias          = true,
                                   bool attention_out_bias      = true,
                                   float eps                    = 1e-6f,
                                   std::string rope_type        = "interleaved")
            : dim(dim),
              audio_dim(audio_dim),
              video_cross_attn_adaln(video_cross_attn_adaln),
              audio_cross_attn_adaln(audio_cross_attn_adaln) {
            video_mod_params = video_cross_attn_adaln ? 9 : 6;
            audio_mod_params = audio_cross_attn_adaln ? 9 : 6;
            cross_attn_adaln = video_cross_attn_adaln || audio_cross_attn_adaln;

            // 1. Self-attention
            blocks["norm1"]       = std::shared_ptr<GGMLBlock>(new RMSNormNoAffine(eps));
            blocks["attn1"]       = std::shared_ptr<GGMLBlock>(new LTX2Attention(
                dim, num_attention_heads, attention_head_dim,
                /*cross_attention_dim=*/-1, attention_bias, attention_out_bias,
                video_gated_attn, rope_type));
            blocks["audio_norm1"] = std::shared_ptr<GGMLBlock>(new RMSNormNoAffine(eps));
            blocks["audio_attn1"] = std::shared_ptr<GGMLBlock>(new LTX2Attention(
                audio_dim, audio_num_attention_heads, audio_attention_head_dim,
                /*cross_attention_dim=*/-1, attention_bias, attention_out_bias,
                audio_gated_attn, rope_type));

            // 2. Prompt cross-attention
            blocks["norm2"]       = std::shared_ptr<GGMLBlock>(new RMSNormNoAffine(eps));
            blocks["attn2"]       = std::shared_ptr<GGMLBlock>(new LTX2Attention(
                dim, num_attention_heads, attention_head_dim,
                cross_attention_dim, attention_bias, attention_out_bias,
                video_gated_attn, rope_type));
            blocks["audio_norm2"] = std::shared_ptr<GGMLBlock>(new RMSNormNoAffine(eps));
            blocks["audio_attn2"] = std::shared_ptr<GGMLBlock>(new LTX2Attention(
                audio_dim, audio_num_attention_heads, audio_attention_head_dim,
                audio_cross_attention_dim, attention_bias, attention_out_bias,
                audio_gated_attn, rope_type));

            // 3. Audio-Video cross-attention
            blocks["audio_to_video_norm"] = std::shared_ptr<GGMLBlock>(new RMSNormNoAffine(eps));
            blocks["audio_to_video_attn"] = std::shared_ptr<GGMLBlock>(new LTX2Attention(
                dim, audio_num_attention_heads, audio_attention_head_dim,
                audio_dim, attention_bias, attention_out_bias,
                video_gated_attn, rope_type));
            blocks["video_to_audio_norm"] = std::shared_ptr<GGMLBlock>(new RMSNormNoAffine(eps));
            blocks["video_to_audio_attn"] = std::shared_ptr<GGMLBlock>(new LTX2Attention(
                audio_dim, audio_num_attention_heads, audio_attention_head_dim,
                dim, attention_bias, attention_out_bias,
                audio_gated_attn, rope_type));

            // 4. Feedforward
            blocks["norm3"]       = std::shared_ptr<GGMLBlock>(new RMSNormNoAffine(eps));
            blocks["ff"]          = std::shared_ptr<GGMLBlock>(new FeedForward(dim, 4 * dim));
            blocks["audio_norm3"] = std::shared_ptr<GGMLBlock>(new RMSNormNoAffine(eps));
            blocks["audio_ff"]    = std::shared_ptr<GGMLBlock>(new FeedForward(audio_dim, 4 * audio_dim));
        }

        // Video-only forward path (isolate_modalities=True, no audio state).
        // hidden        : [N, L, dim]
        // encoder       : [N, L_enc, cross_attention_dim]
        // temb          : [N, T_temb, video_mod_params*dim] — broadcasted across tokens.
        //                 T_temb == 1 in LTX-2 unless per-token modulation is used.
        // rope_cos/sin  : [L, dim]
        // encoder_mask  : additive bias
        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* hidden,
                             ggml_tensor* encoder,
                             ggml_tensor* temb,
                             ggml_tensor* rope_cos     = nullptr,
                             ggml_tensor* rope_sin     = nullptr,
                             ggml_tensor* encoder_mask = nullptr) {
            auto norm1 = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm1"]);
            auto attn1 = std::dynamic_pointer_cast<LTX2Attention>(blocks["attn1"]);
            auto norm2 = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm2"]);
            auto attn2 = std::dynamic_pointer_cast<LTX2Attention>(blocks["attn2"]);
            auto norm3 = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm3"]);
            auto ff    = std::dynamic_pointer_cast<FeedForward>(blocks["ff"]);

            ggml_tensor* sst = params["scale_shift_table"];  // [dim, video_mod_params]

            // temb has shape [video_mod_params*dim, T_temb, N, 1] → reshape to
            // [dim, video_mod_params, T_temb, N].
            auto temb_r = ggml_reshape_4d(ctx->ggml_ctx, temb, dim, video_mod_params,
                                          temb->ne[1], temb->ne[2]);
            auto ada    = ggml_add(ctx->ggml_ctx, temb_r, sst);

            auto slice = [&](int idx) -> ggml_tensor* {
                auto v = ggml_view_4d(ctx->ggml_ctx, ada, ada->ne[0], 1, ada->ne[2], ada->ne[3],
                                      ada->nb[1], ada->nb[2], ada->nb[3], ada->nb[1] * idx);
                return ggml_reshape_3d(ctx->ggml_ctx, v, ada->ne[0], ada->ne[2], ada->ne[3]);
            };
            auto shift_msa = slice(0);
            auto scale_msa = slice(1);
            auto gate_msa  = slice(2);
            auto shift_mlp = slice(3);
            auto scale_mlp = slice(4);
            auto gate_mlp  = slice(5);
            // If video_cross_attn_adaln, indices 6,7,8 are shift_text_q, scale_text_q, gate_text_q.

            // 1. Video self-attention
            auto h_norm   = norm1->forward(ctx, hidden);
            h_norm        = ggml_add(ctx->ggml_ctx, h_norm, ggml_mul(ctx->ggml_ctx, h_norm, scale_msa));
            h_norm        = ggml_add(ctx->ggml_ctx, h_norm, shift_msa);
            auto attn_out = attn1->forward(ctx, h_norm, nullptr,
                                            rope_cos, rope_sin, nullptr, nullptr, nullptr);
            hidden        = ggml_add(ctx->ggml_ctx, hidden, ggml_mul(ctx->ggml_ctx, attn_out, gate_msa));

            // 2. Prompt cross-attention
            auto h_norm2 = norm2->forward(ctx, hidden);
            if (video_cross_attn_adaln) {
                auto shift_q = slice(6);
                auto scale_q = slice(7);
                h_norm2      = ggml_add(ctx->ggml_ctx, h_norm2, ggml_mul(ctx->ggml_ctx, h_norm2, scale_q));
                h_norm2      = ggml_add(ctx->ggml_ctx, h_norm2, shift_q);
            }
            auto ca_out = attn2->forward(ctx, h_norm2, encoder,
                                          nullptr, nullptr, nullptr, nullptr, encoder_mask);
            if (video_cross_attn_adaln) {
                auto gate_q = slice(8);
                ca_out      = ggml_mul(ctx->ggml_ctx, ca_out, gate_q);
            }
            hidden = ggml_add(ctx->ggml_ctx, hidden, ca_out);

            // 3. a2v cross-attention — SKIPPED (video-only mode).

            // 4. Feedforward
            auto h_norm3 = norm3->forward(ctx, hidden);
            h_norm3      = ggml_add(ctx->ggml_ctx, h_norm3, ggml_mul(ctx->ggml_ctx, h_norm3, scale_mlp));
            h_norm3      = ggml_add(ctx->ggml_ctx, h_norm3, shift_mlp);
            auto ff_out  = ff->forward(ctx, h_norm3);
            hidden       = ggml_add(ctx->ggml_ctx, hidden, ggml_mul(ctx->ggml_ctx, ff_out, gate_mlp));

            return hidden;
        }
    };

    // ------------------------------------------------------------------
    // LTX-2 rotary positional embedding.
    //
    // Compared to LTX-1:
    //   * Coords use patch-boundary midpoints (stride `patch_size` start + size/2 step).
    //   * vae_scale_factors = (8, 32, 32) applied per-axis, with causal_offset (=1)
    //     to clamp the first frame's timestamps.
    //   * FPS is applied to the temporal axis (coords / fps → seconds).
    //   * Two rope types: "interleaved" (matches LTX-1 layout) and "split"
    //     (Q and K reshaped to [B, H, T, D/2] before rotation — NOT supported here yet).
    //
    // Host-side CPU builder returns cos/sin tables of shape [L, dim] (interleaved layout).
    // ------------------------------------------------------------------
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
        RopeTables t;
        t.dim = dim;
        t.L   = (int64_t)num_frames * height * width;
        t.cos.assign(t.L * dim, 0.f);
        t.sin.assign(t.L * dim, 0.f);

        // num_pos_dims = 3 (video), num_rope_elems = 6.
        int num_rope_elems = 6;
        int freq_per_axis  = dim / num_rope_elems;
        int pad            = dim % num_rope_elems;  // prepended with cos=1, sin=0

        // Frequencies: pow(theta, linspace(0, 1, dim//num_rope_elems)) * pi/2
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
            // Latent coords: [f, f + patch_size_t) with step patch_size_t.
            // Pixel coords (mid): ((f + patch_size_t/2.0) * vae_scale_t + causal_offset - vae_scale_t)
            // clamped at min 0, then divided by fps.
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
                    float* co   = &t.cos[idx * dim];
                    float* si   = &t.sin[idx * dim];

                    for (int p = 0; p < pad; ++p) {
                        co[p] = 1.f;
                        si[p] = 0.f;
                    }

                    for (int k = 0; k < freq_per_axis; ++k) {
                        float ang_f   = freqs[k] * (gf * 2.f - 1.f);
                        float ang_h   = freqs[k] * (gh * 2.f - 1.f);
                        float ang_w   = freqs[k] * (gw * 2.f - 1.f);
                        float vals[3] = {ang_f, ang_h, ang_w};
                        for (int a = 0; a < 3; ++a) {
                            float c = std::cos(vals[a]);
                            float s = std::sin(vals[a]);
                            co[pad + 2 * (k * 3 + a) + 0] = c;
                            co[pad + 2 * (k * 3 + a) + 1] = c;
                            si[pad + 2 * (k * 3 + a) + 0] = s;
                            si[pad + 2 * (k * 3 + a) + 1] = s;
                        }
                    }
                    ++idx;
                }
            }
        }
        return t;
    }

    // Full LTX-2 transformer (video-only forward, all weights loaded).
    //
    // Default config (LTX-2.0 "Video"):
    //   in_channels=128, out_channels=128,
    //   num_attention_heads=32, attention_head_dim=128, inner_dim=4096,
    //   cross_attention_dim=4096, caption_channels=3840,
    //   num_layers=48, audio_inner_dim=32*64=2048, audio_cross_attention_dim=2048.
    class LTX2VideoTransformer3DModel : public GGMLBlock {
    public:
        int64_t in_channels;
        int64_t out_channels;
        int64_t num_layers;
        int64_t num_attention_heads;
        int64_t attention_head_dim;
        int64_t inner_dim;
        int64_t audio_inner_dim;
        int64_t cross_attention_dim;
        int64_t caption_channels;
        int patch_size;
        int patch_size_t;
        bool gated_attn;
        bool cross_attn_mod;       // adds 3 extra mod params to scale_shift_table
        bool use_prompt_embeddings;
        bool prompt_modulation;    // LTX-2.3 only
        std::string rope_type;     // "interleaved" (supported) or "split" (TODO)

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
                                    int64_t caption_channels          = 3840,
                                    int64_t audio_in_channels         = 128,
                                    int64_t audio_num_attention_heads = 32,
                                    int64_t audio_attention_head_dim  = 64,
                                    int64_t audio_cross_attention_dim = 2048,
                                    bool gated_attn                   = false,
                                    bool cross_attn_mod               = false,
                                    bool audio_gated_attn             = false,
                                    bool audio_cross_attn_mod         = false,
                                    bool use_prompt_embeddings        = true,
                                    float norm_eps                    = 1e-6f,
                                    std::string rope_type             = "interleaved")
            : in_channels(in_channels),
              out_channels(out_channels),
              num_layers(num_layers),
              num_attention_heads(num_attention_heads),
              attention_head_dim(attention_head_dim),
              cross_attention_dim(cross_attention_dim),
              caption_channels(caption_channels),
              patch_size(patch_size),
              patch_size_t(patch_size_t),
              gated_attn(gated_attn),
              cross_attn_mod(cross_attn_mod),
              use_prompt_embeddings(use_prompt_embeddings),
              rope_type(rope_type) {
            inner_dim          = num_attention_heads * attention_head_dim;
            audio_inner_dim    = audio_num_attention_heads * audio_attention_head_dim;
            prompt_modulation  = cross_attn_mod || audio_cross_attn_mod;

            int video_time_emb_mod_params = cross_attn_mod ? 9 : 6;
            int audio_time_emb_mod_params = audio_cross_attn_mod ? 9 : 6;

            // 1. Patchification projections
            blocks["proj_in"]       = std::shared_ptr<GGMLBlock>(new Linear(in_channels, inner_dim, true));
            blocks["audio_proj_in"] = std::shared_ptr<GGMLBlock>(new Linear(audio_in_channels, audio_inner_dim, true));

            // 2. Prompt embeddings
            if (use_prompt_embeddings) {
                blocks["caption_projection"]       = std::shared_ptr<GGMLBlock>(new CaptionProjection(caption_channels, inner_dim));
                blocks["audio_caption_projection"] = std::shared_ptr<GGMLBlock>(new CaptionProjection(caption_channels, audio_inner_dim));
            }

            // 3. Timestep modulation
            blocks["time_embed"]       = std::shared_ptr<GGMLBlock>(new LTX2AdaLayerNormSingle(inner_dim, video_time_emb_mod_params));
            blocks["audio_time_embed"] = std::shared_ptr<GGMLBlock>(new LTX2AdaLayerNormSingle(audio_inner_dim, audio_time_emb_mod_params));

            // Global cross-attention modulation (a2v / v2a)
            blocks["av_cross_attn_video_scale_shift"] =
                std::shared_ptr<GGMLBlock>(new LTX2AdaLayerNormSingle(inner_dim, 4));
            blocks["av_cross_attn_audio_scale_shift"] =
                std::shared_ptr<GGMLBlock>(new LTX2AdaLayerNormSingle(audio_inner_dim, 4));
            blocks["av_cross_attn_video_a2v_gate"] =
                std::shared_ptr<GGMLBlock>(new LTX2AdaLayerNormSingle(inner_dim, 1));
            blocks["av_cross_attn_audio_v2a_gate"] =
                std::shared_ptr<GGMLBlock>(new LTX2AdaLayerNormSingle(audio_inner_dim, 1));

            if (prompt_modulation) {
                blocks["prompt_adaln"]       = std::shared_ptr<GGMLBlock>(new LTX2AdaLayerNormSingle(inner_dim, 2));
                blocks["audio_prompt_adaln"] = std::shared_ptr<GGMLBlock>(new LTX2AdaLayerNormSingle(audio_inner_dim, 2));
            }

            // 5. Transformer blocks
            for (int64_t i = 0; i < num_layers; ++i) {
                blocks["transformer_blocks." + std::to_string(i)] =
                    std::shared_ptr<GGMLBlock>(new LTX2VideoTransformerBlock(
                        inner_dim, num_attention_heads, attention_head_dim, cross_attention_dim,
                        audio_inner_dim, audio_num_attention_heads, audio_attention_head_dim, audio_cross_attention_dim,
                        gated_attn, cross_attn_mod, audio_gated_attn, audio_cross_attn_mod,
                        true, true, norm_eps, rope_type));
            }

            // 6. Output layers
            blocks["norm_out"]       = std::shared_ptr<GGMLBlock>(new LayerNorm(inner_dim, norm_eps, false, false));
            blocks["proj_out"]       = std::shared_ptr<GGMLBlock>(new Linear(inner_dim, out_channels, true));
            blocks["audio_norm_out"] = std::shared_ptr<GGMLBlock>(new LayerNorm(audio_inner_dim, norm_eps, false, false));
            blocks["audio_proj_out"] = std::shared_ptr<GGMLBlock>(new Linear(audio_inner_dim, audio_in_channels, true));
        }

        // Video-only forward pass.
        // hidden_states         : [N, L, in_channels]
        // encoder_hidden_states : [N, L_enc, caption_channels]
        // timestep              : [N]
        // rope_cos / rope_sin   : [L, inner_dim]
        // encoder_mask          : additive bias
        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* hidden_states,
                             ggml_tensor* encoder_hidden_states,
                             ggml_tensor* timestep,
                             ggml_tensor* rope_cos,
                             ggml_tensor* rope_sin,
                             ggml_tensor* encoder_mask = nullptr) {
            auto proj_in   = std::dynamic_pointer_cast<Linear>(blocks["proj_in"]);
            auto te        = std::dynamic_pointer_cast<LTX2AdaLayerNormSingle>(blocks["time_embed"]);
            auto norm_out  = std::dynamic_pointer_cast<LayerNorm>(blocks["norm_out"]);
            auto proj_out  = std::dynamic_pointer_cast<Linear>(blocks["proj_out"]);

            // proj_in patches the latent into inner_dim tokens.
            auto x = proj_in->forward(ctx, hidden_states);

            auto te_pair           = te->forward(ctx, timestep);
            auto temb              = te_pair.first;     // [6*inner_dim or 9*inner_dim, N]
            auto embedded_timestep = te_pair.second;    // [inner_dim, N]

            // Reshape temb to [mod_params*inner_dim, 1, N, 1] for broadcasting.
            temb = ggml_reshape_4d(ctx->ggml_ctx, temb, temb->ne[0], 1, temb->ne[1], 1);

            // Caption projection
            ggml_tensor* encoder = encoder_hidden_states;
            if (use_prompt_embeddings) {
                auto cproj = std::dynamic_pointer_cast<CaptionProjection>(blocks["caption_projection"]);
                encoder    = cproj->forward(ctx, encoder);
            }

            for (int64_t i = 0; i < num_layers; ++i) {
                auto blk = std::dynamic_pointer_cast<LTX2VideoTransformerBlock>(
                    blocks["transformer_blocks." + std::to_string(i)]);
                x = blk->forward(ctx, x, encoder, temb, rope_cos, rope_sin, encoder_mask);
            }

            // Output modulation + projection.
            ggml_tensor* sst = params["scale_shift_table"];  // [inner_dim, 2]
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

            x = norm_out->forward(ctx, x);
            x = ggml_add(ctx->ggml_ctx, x, ggml_mul(ctx->ggml_ctx, x, scale));
            x = ggml_add(ctx->ggml_ctx, x, shift);
            x = proj_out->forward(ctx, x);
            return x;
        }
    };

    // Transformer runner.
    struct LTXVRunner : public GGMLRunner {
        LTX2VideoTransformer3DModel dit;
        RopeTables rope_tbl;

        LTXVRunner(ggml_backend_t backend,
                   bool offload_params_to_cpu,
                   const String2TensorStorage& tensor_storage_map = {},
                   const std::string prefix                       = "model.diffusion_model",
                   SDVersion version                              = VERSION_COUNT)
            : GGMLRunner(backend, offload_params_to_cpu),
              dit(/*in_channels=*/128,
                  /*out_channels=*/128,
                  /*patch_size=*/1,
                  /*patch_size_t=*/1,
                  /*num_attention_heads=*/32,
                  /*attention_head_dim=*/128,
                  /*cross_attention_dim=*/4096,
                  /*num_layers=*/48,
                  /*caption_channels=*/3840) {
            dit.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override { return "ltxv2"; }

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

            rope_tbl      = compute_rope_ltx2((int)F, (int)H, (int)W, (int)dit.inner_dim);
            auto rope_cos = ggml_new_tensor_2d(compute, GGML_TYPE_F32,
                                                (int64_t)dit.inner_dim, rope_tbl.L);
            auto rope_sin = ggml_new_tensor_2d(compute, GGML_TYPE_F32,
                                                (int64_t)dit.inner_dim, rope_tbl.L);
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

    // ==================================================================
    //                              LTX-2 VAE
    // ==================================================================

    // LTX-2 ResnetBlock3d.
    //   norm1: PerChannelRMSNorm  (no weight, runtime)
    //   conv1: CausalConv3d (runtime causal flag)
    //   norm2: PerChannelRMSNorm
    //   conv2: CausalConv3d
    //   shortcut (in != out): LayerNorm(in, elementwise_affine=True, bias=True)
    //                         + plain nn.Conv3d(1, bias=True)  — NO causal padding
    //   timestep_conditioning: scale_shift_table [4, in] applied in two stages.
    class LTX2ResnetBlock3d : public GGMLBlock {
    protected:
        int64_t in_channels;
        int64_t out_channels;
        bool timestep_conditioning;
        bool has_shortcut;

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, std::string prefix = "") override {
            if (timestep_conditioning) {
                params["scale_shift_table"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_channels, 4);
            }
        }

    public:
        LTX2ResnetBlock3d(int64_t in_channels,
                          int64_t out_channels             = -1,
                          bool timestep_conditioning        = false,
                          float eps                        = 1e-6f)
            : in_channels(in_channels), timestep_conditioning(timestep_conditioning) {
            if (out_channels < 0) out_channels = in_channels;
            this->out_channels = out_channels;
            has_shortcut       = (in_channels != out_channels);

            blocks["norm1"] = std::shared_ptr<GGMLBlock>(new PerChannelRMSNorm(1e-8f));
            blocks["conv1"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(in_channels, out_channels, {3, 3, 3}));

            blocks["norm2"] = std::shared_ptr<GGMLBlock>(new PerChannelRMSNorm(1e-8f));
            blocks["conv2"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(out_channels, out_channels, {3, 3, 3}));
            if (has_shortcut) {
                blocks["norm3"]         = std::shared_ptr<GGMLBlock>(new LayerNorm(in_channels, eps, true, true));
                // Plain Conv3d 1x1x1 — NO causal temporal padding (LTX-2 change).
                blocks["conv_shortcut"] = std::shared_ptr<GGMLBlock>(new Conv3d(in_channels, out_channels, {1, 1, 1}));
            }
        }

        // hidden : [W, H, F, C*N]
        // temb   : per-channel modulation (from decoder's time_embedder), or nullptr
        // causal : runtime causal flag
        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* hidden, ggml_tensor* temb = nullptr, bool causal = true) {
            auto norm1 = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm1"]);
            auto conv1 = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv1"]);
            auto norm2 = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm2"]);
            auto conv2 = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv2"]);

            auto residual = hidden;
            auto h        = norm1->forward(ctx, hidden);

            ggml_tensor* shift_1 = nullptr;
            ggml_tensor* scale_1 = nullptr;
            ggml_tensor* shift_2 = nullptr;
            ggml_tensor* scale_2 = nullptr;
            if (timestep_conditioning && temb != nullptr) {
                ggml_tensor* sst = params["scale_shift_table"];
                auto temb_r = ggml_reshape_4d(ctx->ggml_ctx, temb, in_channels, 4, temb->ne[1], 1);
                auto sst_r  = ggml_reshape_4d(ctx->ggml_ctx, sst, in_channels, 4, 1, 1);
                auto ada    = ggml_add(ctx->ggml_ctx, temb_r, sst_r);
                auto slice  = [&](int idx) {
                    auto v = ggml_view_4d(ctx->ggml_ctx, ada,
                                          ada->ne[0], 1, ada->ne[2], ada->ne[3],
                                          ada->nb[1], ada->nb[2], ada->nb[3], ada->nb[1] * idx);
                    return ggml_reshape_4d(ctx->ggml_ctx, v, 1, 1, 1, ada->ne[0] * ada->ne[2]);
                };
                shift_1 = slice(0);
                scale_1 = slice(1);
                shift_2 = slice(2);
                scale_2 = slice(3);
                h       = ggml_add(ctx->ggml_ctx, h, ggml_mul(ctx->ggml_ctx, h, scale_1));
                h       = ggml_add(ctx->ggml_ctx, h, shift_1);
            }

            h = ggml_silu_inplace(ctx->ggml_ctx, h);
            h = conv1->forward(ctx, h, causal);

            h = norm2->forward(ctx, h);
            if (timestep_conditioning && temb != nullptr) {
                h = ggml_add(ctx->ggml_ctx, h, ggml_mul(ctx->ggml_ctx, h, scale_2));
                h = ggml_add(ctx->ggml_ctx, h, shift_2);
            }
            h = ggml_silu_inplace(ctx->ggml_ctx, h);
            h = conv2->forward(ctx, h, causal);

            if (has_shortcut) {
                auto norm3   = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm3"]);
                auto shortct = std::dynamic_pointer_cast<Conv3d>(blocks["conv_shortcut"]);
                residual     = norm3->forward(ctx, residual);
                residual     = shortct->forward(ctx, residual);
            }
            return ggml_add(ctx->ggml_ctx, h, residual);
        }
    };

    // Downsampler3d — LTX-2 (spatial, temporal, spatiotemporal variants).
    // Output computed via a residual "pool" branch (mean of strided blocks)
    // plus the convolution branch. For "spatiotemporal" stride (2,2,2) only
    // the convolution is strided; the other variants rearrange channels to
    // achieve the effective stride.
    class LTX2Downsampler3d : public GGMLBlock {
    protected:
        int64_t in_channels;
        int64_t out_channels;
        std::tuple<int, int, int> stride;

    public:
        LTX2Downsampler3d(int64_t in_channels,
                          int64_t out_channels,
                          std::tuple<int, int, int> stride)
            : in_channels(in_channels), out_channels(out_channels), stride(stride) {
            int st = std::get<0>(stride), sh = std::get<1>(stride), sw = std::get<2>(stride);
            int64_t conv_out = out_channels / (st * sh * sw);
            blocks["conv"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(in_channels, conv_out, {3, 3, 3}));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* h, bool causal = true) {
            auto conv = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv"]);
            // Diffusers' LTX2VideoDownsampler3d is a pixel-shuffle-style operator.
            // The dedicated ggml implementation still needs pixel-shuffle ordering
            // verification against PyTorch outputs (TODO in README.ltxv.md).
            return conv->forward(ctx, h, causal);
        }
    };

    class LTX2DownBlock3D : public GGMLBlock {
    protected:
        int64_t in_channels;
        int64_t out_channels;
        int64_t num_layers;
        bool spatio_temporal_scale;
        std::string downsample_type;

    public:
        LTX2DownBlock3D(int64_t in_channels,
                        int64_t out_channels,
                        int64_t num_layers,
                        bool spatio_temporal_scale,
                        std::string downsample_type = "spatiotemporal")
            : in_channels(in_channels), out_channels(out_channels), num_layers(num_layers),
              spatio_temporal_scale(spatio_temporal_scale),
              downsample_type(downsample_type) {
            for (int64_t i = 0; i < num_layers; ++i) {
                blocks["resnets." + std::to_string(i)] =
                    std::shared_ptr<GGMLBlock>(new LTX2ResnetBlock3d(in_channels, in_channels, false));
            }
            if (spatio_temporal_scale) {
                if (downsample_type == "conv") {
                    blocks["downsamplers.0"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(
                        in_channels, in_channels, {3, 3, 3}, {2, 2, 2}));
                } else {
                    std::tuple<int, int, int> stride{2, 2, 2};
                    if (downsample_type == "spatial")       stride = {1, 2, 2};
                    else if (downsample_type == "temporal") stride = {2, 1, 1};
                    blocks["downsamplers.0"] = std::shared_ptr<GGMLBlock>(new LTX2Downsampler3d(
                        in_channels, out_channels, stride));
                }
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* h, bool causal = true) {
            for (int64_t i = 0; i < num_layers; ++i) {
                auto rn = std::dynamic_pointer_cast<LTX2ResnetBlock3d>(
                    blocks["resnets." + std::to_string(i)]);
                h = rn->forward(ctx, h, nullptr, causal);
            }
            if (spatio_temporal_scale) {
                if (downsample_type == "conv") {
                    auto ds = std::dynamic_pointer_cast<CausalConv3d>(blocks["downsamplers.0"]);
                    h       = ds->forward(ctx, h, causal);
                } else {
                    auto ds = std::dynamic_pointer_cast<LTX2Downsampler3d>(blocks["downsamplers.0"]);
                    h       = ds->forward(ctx, h, causal);
                }
            }
            return h;
        }
    };

    class LTX2MidBlock3d : public GGMLBlock {
    protected:
        int64_t channels;
        int64_t num_layers;
        bool timestep_conditioning;

    public:
        LTX2MidBlock3d(int64_t channels,
                       int64_t num_layers,
                       bool timestep_conditioning = false)
            : channels(channels), num_layers(num_layers), timestep_conditioning(timestep_conditioning) {
            if (timestep_conditioning) {
                blocks["time_embedder.timestep_embedder.linear_1"] =
                    std::shared_ptr<GGMLBlock>(new Linear(256, channels * 4, true));
                blocks["time_embedder.timestep_embedder.linear_2"] =
                    std::shared_ptr<GGMLBlock>(new Linear(channels * 4, channels * 4, true));
            }
            for (int64_t i = 0; i < num_layers; ++i) {
                blocks["resnets." + std::to_string(i)] =
                    std::shared_ptr<GGMLBlock>(new LTX2ResnetBlock3d(channels, channels, timestep_conditioning));
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* h, ggml_tensor* temb_in = nullptr, bool causal = true) {
            ggml_tensor* temb = nullptr;
            if (timestep_conditioning && temb_in != nullptr) {
                auto l1 = std::dynamic_pointer_cast<Linear>(blocks["time_embedder.timestep_embedder.linear_1"]);
                auto l2 = std::dynamic_pointer_cast<Linear>(blocks["time_embedder.timestep_embedder.linear_2"]);
                auto f  = ggml_ext_timestep_embedding(ctx->ggml_ctx, temb_in, 256);
                f       = l1->forward(ctx, f);
                f       = ggml_silu_inplace(ctx->ggml_ctx, f);
                f       = l2->forward(ctx, f);
                temb    = f;
            }
            for (int64_t i = 0; i < num_layers; ++i) {
                auto rn = std::dynamic_pointer_cast<LTX2ResnetBlock3d>(
                    blocks["resnets." + std::to_string(i)]);
                h = rn->forward(ctx, h, temb, causal);
            }
            return h;
        }
    };

    class LTX2UpBlock3d : public GGMLBlock {
    protected:
        int64_t in_channels;
        int64_t out_channels;
        int64_t num_layers;
        bool spatio_temporal_scale;
        bool timestep_conditioning;
        bool has_conv_in;

    public:
        LTX2UpBlock3d(int64_t in_channels,
                      int64_t out_channels,
                      int64_t num_layers,
                      bool spatio_temporal_scale,
                      bool timestep_conditioning)
            : in_channels(in_channels), out_channels(out_channels), num_layers(num_layers),
              spatio_temporal_scale(spatio_temporal_scale),
              timestep_conditioning(timestep_conditioning) {
            has_conv_in = (in_channels != out_channels);

            if (timestep_conditioning) {
                blocks["time_embedder.timestep_embedder.linear_1"] =
                    std::shared_ptr<GGMLBlock>(new Linear(256, in_channels * 4, true));
                blocks["time_embedder.timestep_embedder.linear_2"] =
                    std::shared_ptr<GGMLBlock>(new Linear(in_channels * 4, in_channels * 4, true));
            }
            if (has_conv_in) {
                blocks["conv_in"] = std::shared_ptr<GGMLBlock>(new LTX2ResnetBlock3d(
                    in_channels, out_channels, timestep_conditioning));
            }
            if (spatio_temporal_scale) {
                // Upsampler conv: (out_channels, out_channels*8)  — stride (2,2,2)
                blocks["upsamplers.0.conv"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(
                    out_channels, out_channels * 8, {3, 3, 3}));
            }
            for (int64_t i = 0; i < num_layers; ++i) {
                blocks["resnets." + std::to_string(i)] =
                    std::shared_ptr<GGMLBlock>(new LTX2ResnetBlock3d(
                        out_channels, out_channels, timestep_conditioning));
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* h, ggml_tensor* temb_in = nullptr, bool causal = true) {
            ggml_tensor* temb = nullptr;
            if (timestep_conditioning && temb_in != nullptr) {
                auto l1 = std::dynamic_pointer_cast<Linear>(blocks["time_embedder.timestep_embedder.linear_1"]);
                auto l2 = std::dynamic_pointer_cast<Linear>(blocks["time_embedder.timestep_embedder.linear_2"]);
                auto f  = ggml_ext_timestep_embedding(ctx->ggml_ctx, temb_in, 256);
                f       = l1->forward(ctx, f);
                f       = ggml_silu_inplace(ctx->ggml_ctx, f);
                temb    = l2->forward(ctx, f);
            }

            if (has_conv_in) {
                auto ci = std::dynamic_pointer_cast<LTX2ResnetBlock3d>(blocks["conv_in"]);
                h       = ci->forward(ctx, h, temb, causal);
            }

            if (spatio_temporal_scale) {
                auto up_conv = std::dynamic_pointer_cast<CausalConv3d>(blocks["upsamplers.0.conv"]);
                h            = up_conv->forward(ctx, h, causal);
                // Pixel-shuffle 3D expansion factor (2,2,2). See TODO in docs/ltxv.md
                // about matching diffusers' exact permute order.
                int64_t W = h->ne[0];
                int64_t H = h->ne[1];
                int64_t F = h->ne[2];
                int64_t C = h->ne[3];
                int64_t C_out_real = C / 8;
                h = ggml_cont(ctx->ggml_ctx, h);
                h = ggml_reshape_4d(ctx->ggml_ctx, h, W * 2, H * 2, F * 2, C_out_real);
            }

            for (int64_t i = 0; i < num_layers; ++i) {
                auto rn = std::dynamic_pointer_cast<LTX2ResnetBlock3d>(
                    blocks["resnets." + std::to_string(i)]);
                h = rn->forward(ctx, h, temb, causal);
            }
            return h;
        }
    };

    class LTX2VideoEncoder3d : public GGMLBlock {
    protected:
        int patch_size;
        int patch_size_t;
        int64_t in_channels_patched;
        std::vector<int64_t> block_out_channels;
        std::vector<bool> spatio_temporal_scaling;
        std::vector<int> layers_per_block;
        std::vector<std::string> downsample_type;

    public:
        LTX2VideoEncoder3d(int64_t in_channels_arg                          = 3,
                           int64_t latent_channels                          = 128,
                           std::vector<int64_t> block_out_channels          = {256, 512, 1024, 2048},
                           std::vector<bool> spatio_temporal_scaling         = {true, true, true, true},
                           std::vector<int> layers_per_block                = {4, 3, 3, 3, 4},
                           std::vector<std::string> downsample_type         = {"spatiotemporal", "spatiotemporal", "spatiotemporal", "spatiotemporal"},
                           int patch_size                                   = 4,
                           int patch_size_t                                 = 1)
            : patch_size(patch_size), patch_size_t(patch_size_t),
              block_out_channels(block_out_channels),
              spatio_temporal_scaling(spatio_temporal_scaling),
              layers_per_block(layers_per_block),
              downsample_type(downsample_type) {
            in_channels_patched = in_channels_arg * patch_size * patch_size;
            int64_t out_ch      = block_out_channels[0];

            blocks["conv_in"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(in_channels_patched, out_ch, {3, 3, 3}));
            int nb = (int)block_out_channels.size();
            for (int i = 0; i < nb; ++i) {
                int64_t ic = out_ch;
                int64_t oc = (i + 1 < nb) ? block_out_channels[i + 1] : block_out_channels[i];
                blocks["down_blocks." + std::to_string(i)] =
                    std::shared_ptr<GGMLBlock>(new LTX2DownBlock3D(ic, oc, layers_per_block[i],
                                                                     spatio_temporal_scaling[i], downsample_type[i]));
                out_ch = oc;
            }
            blocks["mid_block"] = std::shared_ptr<GGMLBlock>(new LTX2MidBlock3d(out_ch, layers_per_block.back(), false));
            blocks["norm_out"]  = std::shared_ptr<GGMLBlock>(new PerChannelRMSNorm(1e-8f));
            blocks["conv_out"]  = std::shared_ptr<GGMLBlock>(new CausalConv3d(
                out_ch, latent_channels + 1, {3, 3, 3}));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* h, bool causal = true) {
            int64_t W = h->ne[0];
            int64_t H = h->ne[1];
            int64_t F = h->ne[2];
            int64_t C = h->ne[3];
            if (patch_size > 1 || patch_size_t > 1) {
                int pw = patch_size, ph = patch_size, pt = patch_size_t;
                GGML_ASSERT(W % pw == 0 && H % ph == 0 && F % pt == 0);
                h = ggml_cont(ctx->ggml_ctx, h);
                h = ggml_reshape_4d(ctx->ggml_ctx, h, W / pw, H / ph, F / pt, C * pw * ph * pt);
            }

            auto conv_in = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv_in"]);
            h            = conv_in->forward(ctx, h, causal);

            int nb = (int)block_out_channels.size();
            for (int i = 0; i < nb; ++i) {
                auto db = std::dynamic_pointer_cast<LTX2DownBlock3D>(blocks["down_blocks." + std::to_string(i)]);
                h       = db->forward(ctx, h, causal);
            }

            auto mid      = std::dynamic_pointer_cast<LTX2MidBlock3d>(blocks["mid_block"]);
            h             = mid->forward(ctx, h, nullptr, causal);
            auto norm_out = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_out"]);
            h             = norm_out->forward(ctx, h);
            h             = ggml_silu_inplace(ctx->ggml_ctx, h);
            auto conv_out = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv_out"]);
            h             = conv_out->forward(ctx, h, causal);
            return h;
        }
    };

    class LTX2VideoDecoder3d : public GGMLBlock {
    protected:
        int patch_size;
        int patch_size_t;
        int64_t latent_channels;
        int64_t out_channels_patched;
        std::vector<int64_t> block_out_channels;
        std::vector<bool> spatio_temporal_scaling;
        std::vector<int> layers_per_block;
        bool timestep_conditioning;

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, std::string prefix = "") override {
            if (timestep_conditioning) {
                params["scale_shift_table"]         = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, block_out_channels.back(), 2);
                params["timestep_scale_multiplier"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
            }
        }

    public:
        LTX2VideoDecoder3d(int64_t latent_channels                  = 128,
                           int64_t out_channels_arg                  = 3,
                           std::vector<int64_t> block_out_channels  = {256, 512, 1024, 2048},
                           std::vector<bool> spatio_temporal_scaling = {true, true, true, true},
                           std::vector<int> layers_per_block         = {4, 3, 3, 3, 4},
                           int patch_size                            = 4,
                           int patch_size_t                          = 1,
                           bool timestep_conditioning                 = false)
            : patch_size(patch_size), patch_size_t(patch_size_t),
              latent_channels(latent_channels),
              timestep_conditioning(timestep_conditioning) {
            out_channels_patched = out_channels_arg * patch_size * patch_size;

            std::reverse(block_out_channels.begin(), block_out_channels.end());
            std::reverse(spatio_temporal_scaling.begin(), spatio_temporal_scaling.end());
            std::reverse(layers_per_block.begin(), layers_per_block.end());
            this->block_out_channels      = block_out_channels;
            this->spatio_temporal_scaling = spatio_temporal_scaling;
            this->layers_per_block        = layers_per_block;

            int64_t out_ch = block_out_channels[0];
            blocks["conv_in"]   = std::shared_ptr<GGMLBlock>(new CausalConv3d(latent_channels, out_ch, {3, 3, 3}));
            blocks["mid_block"] = std::shared_ptr<GGMLBlock>(new LTX2MidBlock3d(out_ch, layers_per_block[0], timestep_conditioning));

            int nb = (int)block_out_channels.size();
            for (int i = 0; i < nb; ++i) {
                int64_t ic = out_ch;
                int64_t oc = block_out_channels[i];
                blocks["up_blocks." + std::to_string(i)] =
                    std::shared_ptr<GGMLBlock>(new LTX2UpBlock3d(ic, oc, layers_per_block[i + 1],
                                                                   spatio_temporal_scaling[i], timestep_conditioning));
                out_ch = oc;
            }

            blocks["norm_out"] = std::shared_ptr<GGMLBlock>(new PerChannelRMSNorm(1e-8f));
            blocks["conv_out"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(out_ch, out_channels_patched, {3, 3, 3}));
            if (timestep_conditioning) {
                blocks["time_embedder.timestep_embedder.linear_1"] =
                    std::shared_ptr<GGMLBlock>(new Linear(256, out_ch * 2, true));
                blocks["time_embedder.timestep_embedder.linear_2"] =
                    std::shared_ptr<GGMLBlock>(new Linear(out_ch * 2, out_ch * 2, true));
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* z, ggml_tensor* temb_in = nullptr, bool causal = false) {
            auto conv_in = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv_in"]);
            auto h       = conv_in->forward(ctx, z, causal);

            ggml_tensor* temb_scaled = nullptr;
            if (timestep_conditioning && temb_in != nullptr) {
                ggml_tensor* mult = params["timestep_scale_multiplier"];
                temb_scaled       = ggml_mul(ctx->ggml_ctx, temb_in, mult);
            }

            auto mid = std::dynamic_pointer_cast<LTX2MidBlock3d>(blocks["mid_block"]);
            h        = mid->forward(ctx, h, temb_scaled, causal);

            int nb = (int)block_out_channels.size();
            for (int i = 0; i < nb; ++i) {
                auto ub = std::dynamic_pointer_cast<LTX2UpBlock3d>(blocks["up_blocks." + std::to_string(i)]);
                h       = ub->forward(ctx, h, temb_scaled, causal);
            }

            auto norm_out = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_out"]);
            h             = norm_out->forward(ctx, h);

            if (timestep_conditioning && temb_in != nullptr) {
                auto l1 = std::dynamic_pointer_cast<Linear>(blocks["time_embedder.timestep_embedder.linear_1"]);
                auto l2 = std::dynamic_pointer_cast<Linear>(blocks["time_embedder.timestep_embedder.linear_2"]);
                auto f  = ggml_ext_timestep_embedding(ctx->ggml_ctx, temb_scaled, 256);
                f       = l1->forward(ctx, f);
                f       = ggml_silu_inplace(ctx->ggml_ctx, f);
                f       = l2->forward(ctx, f);
                int64_t out_ch = block_out_channels.back();
                auto f_r   = ggml_reshape_4d(ctx->ggml_ctx, f, out_ch, 2, f->ne[1], 1);
                auto sst   = params["scale_shift_table"];
                auto sst_r = ggml_reshape_4d(ctx->ggml_ctx, sst, out_ch, 2, 1, 1);
                auto ada   = ggml_add(ctx->ggml_ctx, f_r, sst_r);
                auto slice = [&](int idx) {
                    auto v = ggml_view_4d(ctx->ggml_ctx, ada, ada->ne[0], 1, ada->ne[2], ada->ne[3],
                                          ada->nb[1], ada->nb[2], ada->nb[3], ada->nb[1] * idx);
                    return ggml_reshape_4d(ctx->ggml_ctx, v, 1, 1, 1, ada->ne[0] * ada->ne[2]);
                };
                auto shift = slice(0);
                auto scale = slice(1);
                h          = ggml_add(ctx->ggml_ctx, h, ggml_mul(ctx->ggml_ctx, h, scale));
                h          = ggml_add(ctx->ggml_ctx, h, shift);
            }

            h = ggml_silu_inplace(ctx->ggml_ctx, h);
            auto conv_out = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv_out"]);
            h             = conv_out->forward(ctx, h, causal);

            int64_t W = h->ne[0];
            int64_t H = h->ne[1];
            int64_t F = h->ne[2];
            int64_t C = h->ne[3];
            if (patch_size > 1 || patch_size_t > 1) {
                int pw = patch_size, ph = patch_size, pt = patch_size_t;
                int64_t C_out_real = C / (pw * ph * pt);
                h = ggml_cont(ctx->ggml_ctx, h);
                h = ggml_reshape_4d(ctx->ggml_ctx, h, W * pw, H * ph, F * pt, C_out_real);
            }
            return h;
        }
    };

    class LTX2CausalVideoAutoencoder : public GGMLBlock {
    public:
        int64_t latent_channels;

        LTX2CausalVideoAutoencoder(bool decode_only        = true,
                                    int64_t in_channels     = 3,
                                    int64_t out_channels    = 3,
                                    int64_t latent_channels = 128,
                                    bool timestep_conditioning = true)
            : latent_channels(latent_channels) {
            if (!decode_only) {
                blocks["encoder"] = std::shared_ptr<GGMLBlock>(new LTX2VideoEncoder3d(
                    in_channels, latent_channels));
            }
            blocks["decoder"] = std::shared_ptr<GGMLBlock>(new LTX2VideoDecoder3d(
                latent_channels, out_channels,
                {256, 512, 1024, 2048}, {true, true, true, true}, {4, 3, 3, 3, 4},
                4, 1, timestep_conditioning));
        }

        // Encoder is causal by default; decoder is non-causal.
        ggml_tensor* decode(GGMLRunnerContext* ctx, ggml_tensor* z, ggml_tensor* temb_in = nullptr) {
            auto dec = std::dynamic_pointer_cast<LTX2VideoDecoder3d>(blocks["decoder"]);
            return dec->forward(ctx, z, temb_in, /*causal=*/false);
        }

        ggml_tensor* encode(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto enc = std::dynamic_pointer_cast<LTX2VideoEncoder3d>(blocks["encoder"]);
            return enc->forward(ctx, x, /*causal=*/true);
        }
    };

    struct LTXVVAERunner : public VAE {
        bool decode_only = true;
        LTX2CausalVideoAutoencoder ae;

        LTXVVAERunner(SDVersion version,
                      ggml_backend_t backend,
                      bool offload_params_to_cpu,
                      const String2TensorStorage& tensor_storage_map = {},
                      const std::string prefix                       = "first_stage_model",
                      bool decode_only                               = true)
            : VAE(version, backend, offload_params_to_cpu),
              decode_only(decode_only),
              ae(decode_only) {
            scale_input = false;
            ae.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override { return "ltxv2_vae"; }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string prefix) override {
            ae.get_param_tensors(tensors, prefix);
        }

        int get_encoder_output_channels(int input_channels) override {
            SD_UNUSED(input_channels);
            return (int)(2 * ae.latent_channels - 1);
        }

        sd::Tensor<float> vae_output_to_latents(const sd::Tensor<float>& vae_output,
                                                 std::shared_ptr<RNG> rng) override {
            SD_UNUSED(rng);
            return vae_output;
        }
        sd::Tensor<float> diffusion_to_vae_latents(const sd::Tensor<float>& latents) override { return latents; }
        sd::Tensor<float> vae_to_diffusion_latents(const sd::Tensor<float>& latents) override { return latents; }

      protected:
        struct ggml_cgraph* build_graph_decode(const sd::Tensor<float>& z) {
            auto gf   = ggml_new_graph_custom(compute_ctx, LTXV_GRAPH_SIZE, false);
            auto z_t  = make_input(z);
            auto rctx = get_context();
            auto h    = ae.decode(&rctx, z_t, nullptr);
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
