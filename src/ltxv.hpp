#ifndef __LTXV_HPP__
#define __LTXV_HPP__

// LTX-Video (Lightricks) port — diffusers reference:
//   src/diffusers/models/transformers/transformer_ltx.py
//   src/diffusers/models/autoencoders/autoencoder_kl_ltx.py
//
// Two runners are exposed:
//   LTXV::LTXVRunner       — DiT transformer (28 layers, 32 heads × 64 head_dim,
//                            inner_dim=2048, T5-XXL cross-attention, 3D RoPE).
//   LTXV::LTXVVAERunner    — CausalVideoAutoencoder (128 latent channels,
//                            spatial compression 32, temporal compression 8).
//
// Tensor-layout conventions:
//   * torch (N, C, F, H, W) video is stored in ggml as ne = [W, H, F, C*N];
//   * torch (N, L, D) tokens    are stored as ne = [D, L, N, 1];
//   * permutations use ggml_ext_torch_permute which takes torch-order axes.

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

    constexpr int LTXV_GRAPH_SIZE = 10240;

    // RMSNorm with no elementwise-affine weight.
    // Used for block-level norm1/norm2 and VAE norms (`elementwise_affine=False`).
    class RMSNormNoAffine : public UnaryBlock {
    protected:
        float eps;

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, std::string prefix = "") override {
            // no parameters
        }

    public:
        RMSNormNoAffine(float eps = 1e-6f) : eps(eps) {}

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            return ggml_rms_norm(ctx->ggml_ctx, x, eps);
        }
    };

    // Channel-wise RMSNorm for 5-D video.
    // Input ne = [W, H, F, C*N]; permutes C to innermost, normalises, optionally
    // applies affine weight of shape [C], permutes back. Mirrors diffusers'
    // `RMSNorm(C, …).movedim(1,-1)` dance from autoencoder_kl_ltx.py.
    class VideoChannelRMSNorm : public UnaryBlock {
    protected:
        int64_t channels;
        float eps;
        bool elementwise_affine;
        std::string prefix;

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, std::string prefix = "") override {
            this->prefix = prefix;
            if (elementwise_affine) {
                params["weight"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, channels);
            }
        }

    public:
        VideoChannelRMSNorm(int64_t channels,
                            float eps               = 1e-8f,
                            bool elementwise_affine = false)
            : channels(channels), eps(eps), elementwise_affine(elementwise_affine) {}

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            // x: [W, H, F, C*N] (N == 1 inference path).
            auto h = ggml_ext_cont(ctx->ggml_ctx,
                                   ggml_ext_torch_permute(ctx->ggml_ctx, x, 3, 0, 1, 2));  // [C*N, W, H, F]
            h      = ggml_rms_norm(ctx->ggml_ctx, h, eps);
            if (elementwise_affine) {
                ggml_tensor* w = params["weight"];
                h              = ggml_mul(ctx->ggml_ctx, h, w);
            }
            h = ggml_ext_cont(ctx->ggml_ctx,
                              ggml_ext_torch_permute(ctx->ggml_ctx, h, 1, 2, 3, 0));  // [W, H, F, C*N]
            return h;
        }
    };

    // Temporal-causal 3-D convolution.
    // Spatial padding is k/2 (same-padding); temporal padding is:
    //   causal: (k_t - 1) frames left via first-frame replication, 0 right;
    //   non-causal: (k_t - 1)/2 each side via first/last-frame replication.
    class CausalConv3d : public GGMLBlock {
    protected:
        int64_t in_channels;
        int64_t out_channels;
        std::tuple<int, int, int> kernel_size;   // (kt, kh, kw)
        std::tuple<int, int, int> stride;
        std::tuple<int, int, int> dilation;
        bool bias;
        bool is_causal;

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
                     bool bias                          = true,
                     bool is_causal                     = true)
            : in_channels(in_channels),
              out_channels(out_channels),
              kernel_size(kernel_size),
              stride(stride),
              dilation(dilation),
              bias(bias),
              is_causal(is_causal) {}

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            ggml_tensor* w = params["weight"];
            ggml_tensor* b = bias ? params["bias"] : nullptr;

            int kt = std::get<0>(kernel_size);
            int kh = std::get<1>(kernel_size);
            int kw = std::get<2>(kernel_size);

            if (kt > 1) {
                if (is_causal) {
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

            int lp_w = kw / 2;
            int rp_w = kw / 2;
            int lp_h = kh / 2;
            int rp_h = kh / 2;
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

    // Caption projection (PixArt-Alpha): Linear → GELU(tanh) → Linear.
    // Parameters: linear_1, linear_2.
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

    // Timestep embedder used inside AdaLayerNormSingle.
    // Parameters: linear_1, linear_2.
    class TimestepEmbedder : public GGMLBlock {
    protected:
        int64_t frequency_embedding_size;

    public:
        TimestepEmbedder(int64_t hidden_size, int64_t frequency_embedding_size = 256)
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

    // AdaLayerNormSingle(hidden, use_additional_conditions=False).
    // emb.timestep_embedder + linear(hidden -> 6*hidden).
    // Returns (temb, embedded_timestep) as a pair.
    class AdaLayerNormSingle : public GGMLBlock {
    public:
        AdaLayerNormSingle(int64_t hidden_size, int64_t frequency_embedding_size = 256) {
            blocks["emb.timestep_embedder"] =
                std::shared_ptr<GGMLBlock>(new TimestepEmbedder(hidden_size, frequency_embedding_size));
            blocks["linear"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, 6 * hidden_size, true));
        }

        std::pair<ggml_tensor*, ggml_tensor*> forward(GGMLRunnerContext* ctx, ggml_tensor* t) {
            auto tse    = std::dynamic_pointer_cast<TimestepEmbedder>(blocks["emb.timestep_embedder"]);
            auto linear = std::dynamic_pointer_cast<Linear>(blocks["linear"]);

            auto embedded_timestep = tse->forward(ctx, t);                      // [hidden, N]
            auto x                 = ggml_silu(ctx->ggml_ctx, embedded_timestep);
            auto temb              = linear->forward(ctx, x);                   // [6*hidden, N]
            return {temb, embedded_timestep};
        }
    };

    // FeedForward(dim, "gelu-approximate"): net.0.proj, net.2.
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

    // LTXAttention — diffusers.LTXAttention.
    // Parameters: to_q, to_k, to_v, to_out.0 (+bias), norm_q, norm_k.
    // qk_norm = rms_norm_across_heads → weight shape = (inner_dim,), applied
    // to full Q/K before head split.
    // Self-attn uses 3-D RoPE on Q and K; cross-attn does not.
    class LTXAttention : public GGMLBlock {
    public:
        int64_t inner_dim;
        int64_t num_heads;
        int64_t head_dim;
        bool is_cross_attn;
        bool has_rope;

    public:
        LTXAttention(int64_t query_dim,
                     int64_t heads,
                     int64_t dim_head,
                     int64_t cross_attention_dim = -1,
                     bool attention_bias         = true,
                     bool attention_out_bias     = true)
            : num_heads(heads), head_dim(dim_head) {
            inner_dim           = heads * dim_head;
            int64_t kv_dim      = (cross_attention_dim > 0) ? cross_attention_dim : query_dim;
            is_cross_attn       = cross_attention_dim > 0;
            has_rope            = !is_cross_attn;

            blocks["to_q"]     = std::shared_ptr<GGMLBlock>(new Linear(query_dim, inner_dim, attention_bias));
            blocks["to_k"]     = std::shared_ptr<GGMLBlock>(new Linear(kv_dim, inner_dim, attention_bias));
            blocks["to_v"]     = std::shared_ptr<GGMLBlock>(new Linear(kv_dim, inner_dim, attention_bias));
            blocks["to_out.0"] = std::shared_ptr<GGMLBlock>(new Linear(inner_dim, query_dim, attention_out_bias));

            blocks["norm_q"] = std::shared_ptr<GGMLBlock>(new RMSNorm(inner_dim, 1e-5f));
            blocks["norm_k"] = std::shared_ptr<GGMLBlock>(new RMSNorm(inner_dim, 1e-5f));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* hidden_states,
                             ggml_tensor* encoder_hidden_states = nullptr,
                             ggml_tensor* rope_cos              = nullptr,
                             ggml_tensor* rope_sin              = nullptr,
                             ggml_tensor* attention_mask        = nullptr) {
            auto to_q    = std::dynamic_pointer_cast<Linear>(blocks["to_q"]);
            auto to_k    = std::dynamic_pointer_cast<Linear>(blocks["to_k"]);
            auto to_v    = std::dynamic_pointer_cast<Linear>(blocks["to_v"]);
            auto to_out  = std::dynamic_pointer_cast<Linear>(blocks["to_out.0"]);
            auto norm_q  = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_q"]);
            auto norm_k  = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_k"]);

            ggml_tensor* kv_src = encoder_hidden_states != nullptr ? encoder_hidden_states : hidden_states;

            auto q = to_q->forward(ctx, hidden_states);
            auto k = to_k->forward(ctx, kv_src);
            auto v = to_v->forward(ctx, kv_src);

            q = norm_q->forward(ctx, q);
            k = norm_k->forward(ctx, k);

            if (has_rope && rope_cos != nullptr && rope_sin != nullptr) {
                q = apply_rotary_emb(ctx, q, rope_cos, rope_sin);
                k = apply_rotary_emb(ctx, k, rope_cos, rope_sin);
            }

            auto out = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v,
                                              num_heads, attention_mask, false,
                                              ctx->flash_attn_enabled);
            out      = to_out->forward(ctx, out);
            return out;
        }

        // diffusers apply_rotary_emb: pairs-of-two rotation.
        //   x_real, x_imag = x.unflatten(2, (-1, 2)).unbind(-1)
        //   x_rotated      = stack([-x_imag, x_real], -1).flatten(2)
        //   out            = x * cos + x_rotated * sin
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

    // Transformer block.
    // Modulation (diffusers transformer_ltx.py:342-379):
    //   sst                                  : [6, dim]           (parameter)
    //   ada = sst[None,None] + temb.reshape(B, T_temb, 6, dim)
    //   shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada.unbind(2)
    //   h = norm1(h) * (1 + scale_msa) + shift_msa
    //   h = h + attn1(h, rope) * gate_msa
    //   h = h + attn2(h, encoder)        # cross-attn, no gate
    //   h = norm2(h) * (1 + scale_mlp) + shift_mlp
    //   h = h + ff(h) * gate_mlp
    class LTXVideoTransformerBlock : public GGMLBlock {
    protected:
        int64_t dim;

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, std::string prefix = "") override {
            params["scale_shift_table"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, 6);
        }

    public:
        LTXVideoTransformerBlock(int64_t dim,
                                 int64_t num_attention_heads,
                                 int64_t attention_head_dim,
                                 int64_t cross_attention_dim,
                                 bool attention_bias     = true,
                                 bool attention_out_bias = true,
                                 float eps                = 1e-6f)
            : dim(dim) {
            blocks["norm1"] = std::shared_ptr<GGMLBlock>(new RMSNormNoAffine(eps));
            blocks["attn1"] = std::shared_ptr<GGMLBlock>(new LTXAttention(
                dim, num_attention_heads, attention_head_dim,
                /*cross_attention_dim=*/-1, attention_bias, attention_out_bias));

            blocks["norm2"] = std::shared_ptr<GGMLBlock>(new RMSNormNoAffine(eps));
            blocks["attn2"] = std::shared_ptr<GGMLBlock>(new LTXAttention(
                dim, num_attention_heads, attention_head_dim,
                /*cross_attention_dim=*/cross_attention_dim, attention_bias, attention_out_bias));

            blocks["ff"] = std::shared_ptr<GGMLBlock>(new FeedForward(dim, 4 * dim));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* hidden,
                             ggml_tensor* encoder,
                             ggml_tensor* temb,
                             ggml_tensor* rope_cos     = nullptr,
                             ggml_tensor* rope_sin     = nullptr,
                             ggml_tensor* encoder_mask = nullptr) {
            auto norm1 = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm1"]);
            auto attn1 = std::dynamic_pointer_cast<LTXAttention>(blocks["attn1"]);
            auto norm2 = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm2"]);
            auto attn2 = std::dynamic_pointer_cast<LTXAttention>(blocks["attn2"]);
            auto ff    = std::dynamic_pointer_cast<FeedForward>(blocks["ff"]);

            ggml_tensor* sst = params["scale_shift_table"];  // [dim, 6]

            // temb has shape [6*dim, T_temb, N, 1]; reshape to [dim, 6, T_temb, N].
            auto temb_r = ggml_reshape_4d(ctx->ggml_ctx, temb, dim, 6, temb->ne[1], temb->ne[2]);
            // sst is [dim, 6, 1, 1] — broadcasts across T_temb and N.
            auto ada    = ggml_add(ctx->ggml_ctx, temb_r, sst);

            auto ada_slice = [&](int idx) -> ggml_tensor* {
                auto v = ggml_view_4d(ctx->ggml_ctx, ada,
                                      ada->ne[0], 1, ada->ne[2], ada->ne[3],
                                      ada->nb[1], ada->nb[2], ada->nb[3],
                                      ada->nb[1] * idx);
                return ggml_reshape_3d(ctx->ggml_ctx, v, ada->ne[0], ada->ne[2], ada->ne[3]);
            };
            auto shift_msa = ada_slice(0);
            auto scale_msa = ada_slice(1);
            auto gate_msa  = ada_slice(2);
            auto shift_mlp = ada_slice(3);
            auto scale_mlp = ada_slice(4);
            auto gate_mlp  = ada_slice(5);

            auto h_norm   = norm1->forward(ctx, hidden);
            h_norm        = ggml_add(ctx->ggml_ctx, h_norm,
                                     ggml_mul(ctx->ggml_ctx, h_norm, scale_msa));
            h_norm        = ggml_add(ctx->ggml_ctx, h_norm, shift_msa);
            auto attn_out = attn1->forward(ctx, h_norm, nullptr, rope_cos, rope_sin, nullptr);
            hidden        = ggml_add(ctx->ggml_ctx, hidden,
                                     ggml_mul(ctx->ggml_ctx, attn_out, gate_msa));

            auto cross_out = attn2->forward(ctx, hidden, encoder, nullptr, nullptr, encoder_mask);
            hidden         = ggml_add(ctx->ggml_ctx, hidden, cross_out);

            h_norm = norm2->forward(ctx, hidden);
            h_norm = ggml_add(ctx->ggml_ctx, h_norm,
                              ggml_mul(ctx->ggml_ctx, h_norm, scale_mlp));
            h_norm = ggml_add(ctx->ggml_ctx, h_norm, shift_mlp);
            auto ff_out = ff->forward(ctx, h_norm);
            hidden      = ggml_add(ctx->ggml_ctx, hidden,
                                   ggml_mul(ctx->ggml_ctx, ff_out, gate_mlp));
            return hidden;
        }
    };

    // 3-D rotary positional embedding.
    // Per-axis freqs = dim // 6. Applied to (F, H, W) grid.
    // diffusers reference: transformer_ltx.py lines 179-278.
    struct RopeTables {
        std::vector<float> cos;
        std::vector<float> sin;
        int64_t L   = 0;
        int64_t dim = 0;
    };

    __STATIC_INLINE__ RopeTables compute_rope(int num_frames,
                                              int height,
                                              int width,
                                              int dim,
                                              int base_frames = 20,
                                              int base_h      = 2048,
                                              int base_w      = 2048,
                                              int patch_size  = 1,
                                              int patch_t     = 1,
                                              float scale_f   = 1.f,
                                              float scale_h   = 1.f,
                                              float scale_w   = 1.f,
                                              float theta     = 10000.f) {
        RopeTables t;
        t.dim = dim;
        t.L   = (int64_t)num_frames * height * width;
        t.cos.assign(t.L * dim, 0.f);
        t.sin.assign(t.L * dim, 0.f);

        int freq_per_axis = dim / 6;
        int pad           = dim % 6;

        std::vector<float> omega(freq_per_axis);
        if (freq_per_axis > 1) {
            float start = 0.f;
            float end   = 1.f;
            float step  = (end - start) / (freq_per_axis - 1);
            for (int i = 0; i < freq_per_axis; ++i) {
                float exponent = start + i * step;
                omega[i]       = std::pow(theta, exponent) * (float)M_PI / 2.f;
            }
        } else if (freq_per_axis == 1) {
            omega[0] = 1.f * (float)M_PI / 2.f;
        }

        int64_t idx = 0;
        for (int f = 0; f < num_frames; ++f) {
            float gf = (float)f * scale_f * patch_t / (float)base_frames;
            for (int h = 0; h < height; ++h) {
                float gh = (float)h * scale_h * patch_size / (float)base_h;
                for (int w = 0; w < width; ++w) {
                    float gw = (float)w * scale_w * patch_size / (float)base_w;
                    float* co = &t.cos[idx * dim];
                    float* si = &t.sin[idx * dim];

                    for (int p = 0; p < pad; ++p) {
                        co[p] = 1.f;
                        si[p] = 0.f;
                    }

                    for (int k = 0; k < freq_per_axis; ++k) {
                        float ang_f   = omega[k] * (gf * 2.f - 1.f);
                        float ang_h   = omega[k] * (gh * 2.f - 1.f);
                        float ang_w   = omega[k] * (gw * 2.f - 1.f);
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

    // Full LTX transformer (LTXVideoTransformer3DModel).
    // Top-level parameters:
    //   proj_in              : Linear(in, inner_dim, bias)
    //   time_embed           : AdaLayerNormSingle(inner_dim)
    //   caption_projection   : CaptionProjection(caption_ch, inner_dim)
    //   transformer_blocks.N : LTXVideoTransformerBlock * num_layers
    //   norm_out             : LayerNorm(inner_dim, elementwise_affine=False)
    //   scale_shift_table    : [2, inner_dim]
    //   proj_out             : Linear(inner_dim, out, bias)
    class LTXVideoTransformer3DModel : public GGMLBlock {
    public:
        int64_t in_channels;
        int64_t out_channels;
        int64_t num_layers;
        int64_t num_attention_heads;
        int64_t attention_head_dim;
        int64_t inner_dim;
        int64_t cross_attention_dim;
        int64_t caption_channels;
        int patch_size;
        int patch_size_t;

    protected:
        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, std::string prefix = "") override {
            params["scale_shift_table"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, inner_dim, 2);
        }

    public:
        LTXVideoTransformer3DModel(int64_t in_channels         = 128,
                                   int64_t out_channels         = 128,
                                   int patch_size               = 1,
                                   int patch_size_t             = 1,
                                   int64_t num_attention_heads  = 32,
                                   int64_t attention_head_dim   = 64,
                                   int64_t cross_attention_dim  = 2048,
                                   int64_t num_layers           = 28,
                                   int64_t caption_channels     = 4096,
                                   bool attention_bias          = true,
                                   bool attention_out_bias      = true,
                                   float norm_eps               = 1e-6f)
            : in_channels(in_channels),
              out_channels(out_channels),
              num_layers(num_layers),
              num_attention_heads(num_attention_heads),
              attention_head_dim(attention_head_dim),
              cross_attention_dim(cross_attention_dim),
              caption_channels(caption_channels),
              patch_size(patch_size),
              patch_size_t(patch_size_t) {
            inner_dim = num_attention_heads * attention_head_dim;

            blocks["proj_in"]            = std::shared_ptr<GGMLBlock>(new Linear(in_channels, inner_dim, true));
            blocks["time_embed"]         = std::shared_ptr<GGMLBlock>(new AdaLayerNormSingle(inner_dim));
            blocks["caption_projection"] = std::shared_ptr<GGMLBlock>(new CaptionProjection(caption_channels, inner_dim));
            for (int64_t i = 0; i < num_layers; ++i) {
                blocks["transformer_blocks." + std::to_string(i)] =
                    std::shared_ptr<GGMLBlock>(new LTXVideoTransformerBlock(
                        inner_dim, num_attention_heads, attention_head_dim, cross_attention_dim,
                        attention_bias, attention_out_bias, norm_eps));
            }
            blocks["norm_out"] = std::shared_ptr<GGMLBlock>(new LayerNorm(inner_dim, norm_eps,
                                                                           /*elementwise_affine=*/false,
                                                                           /*bias=*/false));
            blocks["proj_out"] = std::shared_ptr<GGMLBlock>(new Linear(inner_dim, out_channels, true));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* hidden_states,
                             ggml_tensor* encoder_hidden_states,
                             ggml_tensor* timestep,
                             ggml_tensor* rope_cos,
                             ggml_tensor* rope_sin,
                             ggml_tensor* encoder_mask = nullptr) {
            auto proj_in  = std::dynamic_pointer_cast<Linear>(blocks["proj_in"]);
            auto te       = std::dynamic_pointer_cast<AdaLayerNormSingle>(blocks["time_embed"]);
            auto cproj    = std::dynamic_pointer_cast<CaptionProjection>(blocks["caption_projection"]);
            auto norm_out = std::dynamic_pointer_cast<LayerNorm>(blocks["norm_out"]);
            auto proj_out = std::dynamic_pointer_cast<Linear>(blocks["proj_out"]);

            auto x = proj_in->forward(ctx, hidden_states);  // [inner_dim, L, N]

            auto te_pair           = te->forward(ctx, timestep);
            auto temb              = te_pair.first;    // [6*inner_dim, N]
            auto embedded_timestep = te_pair.second;   // [inner_dim, N]

            // Reshape temb to [6*inner_dim, 1, N, 1] for broadcasting across L.
            temb = ggml_reshape_4d(ctx->ggml_ctx, temb, 6 * inner_dim, 1, temb->ne[1], 1);

            auto encoder = cproj->forward(ctx, encoder_hidden_states);

            for (int64_t i = 0; i < num_layers; ++i) {
                auto blk = std::dynamic_pointer_cast<LTXVideoTransformerBlock>(
                    blocks["transformer_blocks." + std::to_string(i)]);
                x = blk->forward(ctx, x, encoder, temb, rope_cos, rope_sin, encoder_mask);
            }

            // Final modulation + projection.
            ggml_tensor* sst = params["scale_shift_table"];                 // [inner_dim, 2]
            auto et_r        = ggml_reshape_4d(ctx->ggml_ctx, embedded_timestep,
                                                inner_dim, 1, embedded_timestep->ne[1], 1);
            auto sst_r       = ggml_reshape_4d(ctx->ggml_ctx, sst, inner_dim, 2, 1, 1);
            // Broadcast et_r to [inner_dim, 2, N, 1] via explicit repeat.
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

    // ==================================================================
    //                         TRANSFORMER RUNNER
    // ==================================================================

    struct LTXVRunner : public GGMLRunner {
        LTXVideoTransformer3DModel dit;
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
                  /*attention_head_dim=*/64,
                  /*cross_attention_dim=*/2048,
                  /*num_layers=*/28,
                  /*caption_channels=*/4096) {
            dit.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override { return "ltxv"; }

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

            // Build RoPE tables on host; rope_tbl member keeps data alive.
            rope_tbl      = compute_rope((int)F, (int)H, (int)W, (int)dit.inner_dim);
            auto rope_cos = ggml_new_tensor_2d(compute, GGML_TYPE_F32,
                                                (int64_t)dit.inner_dim, rope_tbl.L);
            auto rope_sin = ggml_new_tensor_2d(compute, GGML_TYPE_F32,
                                                (int64_t)dit.inner_dim, rope_tbl.L);
            set_backend_tensor_data(rope_cos, rope_tbl.cos.data());
            set_backend_tensor_data(rope_sin, rope_tbl.sin.data());

            // [W, H, F, C] -> [C, W*H*F, 1]
            auto hidden = ggml_ext_cont(compute,
                                        ggml_ext_torch_permute(compute, x_t, 3, 0, 1, 2));
            hidden      = ggml_reshape_3d(compute, hidden, C, W * H * F, 1);

            auto rctx = get_context();
            auto out  = dit.forward(&rctx, hidden, c_t, ts_t, rope_cos, rope_sin, m_t);

            // [C, W*H*F, 1] -> [W, H, F, C]
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
    //                               VAE
    // ==================================================================

    class LTXResnetBlock3d : public GGMLBlock {
    protected:
        int64_t in_channels;
        int64_t out_channels;
        bool is_causal;
        bool timestep_conditioning;
        bool has_shortcut;

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, std::string prefix = "") override {
            if (timestep_conditioning) {
                params["scale_shift_table"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_channels, 4);
            }
        }

    public:
        LTXResnetBlock3d(int64_t in_channels,
                         int64_t out_channels             = -1,
                         bool is_causal                   = true,
                         bool timestep_conditioning        = false,
                         float eps                        = 1e-6f)
            : in_channels(in_channels),
              timestep_conditioning(timestep_conditioning) {
            if (out_channels < 0) out_channels = in_channels;
            this->out_channels = out_channels;
            has_shortcut       = (in_channels != out_channels);

            blocks["norm1"] = std::shared_ptr<GGMLBlock>(new VideoChannelRMSNorm(in_channels, 1e-8f, false));
            blocks["conv1"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(in_channels, out_channels, {3, 3, 3},
                                                                           {1, 1, 1}, {1, 1, 1}, true, is_causal));

            blocks["norm2"] = std::shared_ptr<GGMLBlock>(new VideoChannelRMSNorm(out_channels, 1e-8f, false));
            blocks["conv2"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(out_channels, out_channels, {3, 3, 3},
                                                                           {1, 1, 1}, {1, 1, 1}, true, is_causal));
            if (has_shortcut) {
                blocks["norm3"]         = std::shared_ptr<GGMLBlock>(new VideoChannelRMSNorm(in_channels, eps, true));
                blocks["conv_shortcut"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(in_channels, out_channels, {1, 1, 1},
                                                                                       {1, 1, 1}, {1, 1, 1}, true, is_causal));
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* hidden, ggml_tensor* temb = nullptr) {
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
                ggml_tensor* sst = params["scale_shift_table"];                 // [C, 4]
                auto temb_r = ggml_reshape_4d(ctx->ggml_ctx, temb, in_channels, 4, temb->ne[1], 1);
                auto sst_r  = ggml_reshape_4d(ctx->ggml_ctx, sst, in_channels, 4, 1, 1);
                auto ada    = ggml_add(ctx->ggml_ctx, temb_r, sst_r);
                auto slice  = [&](int idx) {
                    auto v = ggml_view_4d(ctx->ggml_ctx, ada,
                                          ada->ne[0], 1, ada->ne[2], ada->ne[3],
                                          ada->nb[1], ada->nb[2], ada->nb[3], ada->nb[1] * idx);
                    // Make it broadcastable over [W, H, F]: reshape to [1,1,1,C*N].
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
            h = conv1->forward(ctx, h);

            h = norm2->forward(ctx, h);
            if (timestep_conditioning && temb != nullptr) {
                h = ggml_add(ctx->ggml_ctx, h, ggml_mul(ctx->ggml_ctx, h, scale_2));
                h = ggml_add(ctx->ggml_ctx, h, shift_2);
            }
            h = ggml_silu_inplace(ctx->ggml_ctx, h);
            h = conv2->forward(ctx, h);

            if (has_shortcut) {
                auto norm3   = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm3"]);
                auto shortct = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv_shortcut"]);
                residual     = norm3->forward(ctx, residual);
                residual     = shortct->forward(ctx, residual);
            }
            return ggml_add(ctx->ggml_ctx, h, residual);
        }
    };

    class LTXDownBlock3D : public GGMLBlock {
    protected:
        int64_t in_channels;
        int64_t out_channels;
        int64_t num_layers;
        bool spatio_temporal_scale;
        bool is_causal;
        bool has_out_proj;

    public:
        LTXDownBlock3D(int64_t in_channels,
                       int64_t out_channels,
                       int64_t num_layers,
                       bool spatio_temporal_scale,
                       bool is_causal)
            : in_channels(in_channels),
              out_channels(out_channels),
              num_layers(num_layers),
              spatio_temporal_scale(spatio_temporal_scale),
              is_causal(is_causal) {
            for (int64_t i = 0; i < num_layers; ++i) {
                blocks["resnets." + std::to_string(i)] =
                    std::shared_ptr<GGMLBlock>(new LTXResnetBlock3d(in_channels, in_channels, is_causal, false));
            }
            if (spatio_temporal_scale) {
                blocks["downsamplers.0"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(
                    in_channels, in_channels, {3, 3, 3}, {2, 2, 2}, {1, 1, 1}, true, is_causal));
            }
            has_out_proj = (in_channels != out_channels);
            if (has_out_proj) {
                blocks["conv_out"] = std::shared_ptr<GGMLBlock>(new LTXResnetBlock3d(
                    in_channels, out_channels, is_causal, false));
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* h) {
            for (int64_t i = 0; i < num_layers; ++i) {
                auto rn = std::dynamic_pointer_cast<LTXResnetBlock3d>(
                    blocks["resnets." + std::to_string(i)]);
                h = rn->forward(ctx, h, nullptr);
            }
            if (spatio_temporal_scale) {
                auto ds = std::dynamic_pointer_cast<CausalConv3d>(blocks["downsamplers.0"]);
                h       = ds->forward(ctx, h);
            }
            if (has_out_proj) {
                auto co = std::dynamic_pointer_cast<LTXResnetBlock3d>(blocks["conv_out"]);
                h       = co->forward(ctx, h, nullptr);
            }
            return h;
        }
    };

    class LTXMidBlock3d : public GGMLBlock {
    protected:
        int64_t channels;
        int64_t num_layers;
        bool timestep_conditioning;

    public:
        LTXMidBlock3d(int64_t channels,
                      int64_t num_layers,
                      bool is_causal             = true,
                      bool timestep_conditioning = false)
            : channels(channels),
              num_layers(num_layers),
              timestep_conditioning(timestep_conditioning) {
            if (timestep_conditioning) {
                blocks["time_embedder.timestep_embedder.linear_1"] =
                    std::shared_ptr<GGMLBlock>(new Linear(256, channels * 4, true));
                blocks["time_embedder.timestep_embedder.linear_2"] =
                    std::shared_ptr<GGMLBlock>(new Linear(channels * 4, channels * 4, true));
            }
            for (int64_t i = 0; i < num_layers; ++i) {
                blocks["resnets." + std::to_string(i)] =
                    std::shared_ptr<GGMLBlock>(new LTXResnetBlock3d(channels, channels, is_causal, timestep_conditioning));
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* h, ggml_tensor* temb_in = nullptr) {
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
                auto rn = std::dynamic_pointer_cast<LTXResnetBlock3d>(
                    blocks["resnets." + std::to_string(i)]);
                h = rn->forward(ctx, h, temb);
            }
            return h;
        }
    };

    class LTXUpBlock3d : public GGMLBlock {
    protected:
        int64_t in_channels;
        int64_t out_channels;
        int64_t num_layers;
        bool spatio_temporal_scale;
        bool is_causal;
        bool timestep_conditioning;
        bool has_conv_in;

    public:
        LTXUpBlock3d(int64_t in_channels,
                     int64_t out_channels,
                     int64_t num_layers,
                     bool spatio_temporal_scale,
                     bool is_causal,
                     bool timestep_conditioning)
            : in_channels(in_channels),
              out_channels(out_channels),
              num_layers(num_layers),
              spatio_temporal_scale(spatio_temporal_scale),
              is_causal(is_causal),
              timestep_conditioning(timestep_conditioning) {
            has_conv_in = (in_channels != out_channels);

            if (timestep_conditioning) {
                blocks["time_embedder.timestep_embedder.linear_1"] =
                    std::shared_ptr<GGMLBlock>(new Linear(256, in_channels * 4, true));
                blocks["time_embedder.timestep_embedder.linear_2"] =
                    std::shared_ptr<GGMLBlock>(new Linear(in_channels * 4, in_channels * 4, true));
            }
            if (has_conv_in) {
                blocks["conv_in"] = std::shared_ptr<GGMLBlock>(new LTXResnetBlock3d(
                    in_channels, out_channels, is_causal, timestep_conditioning));
            }
            if (spatio_temporal_scale) {
                // Upsampler's internal conv: (out_channels, out_channels*8) with stride 1.
                blocks["upsamplers.0.conv"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(
                    out_channels, out_channels * 8, {3, 3, 3},
                    {1, 1, 1}, {1, 1, 1}, true, is_causal));
            }
            for (int64_t i = 0; i < num_layers; ++i) {
                blocks["resnets." + std::to_string(i)] =
                    std::shared_ptr<GGMLBlock>(new LTXResnetBlock3d(
                        out_channels, out_channels, is_causal, timestep_conditioning));
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* h, ggml_tensor* temb_in = nullptr) {
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
                auto ci = std::dynamic_pointer_cast<LTXResnetBlock3d>(blocks["conv_in"]);
                h       = ci->forward(ctx, h, temb);
            }

            if (spatio_temporal_scale) {
                auto up_conv = std::dynamic_pointer_cast<CausalConv3d>(blocks["upsamplers.0.conv"]);
                h            = up_conv->forward(ctx, h);

                // Pixel-shuffle 3D with factor (2, 2, 2).
                // In ggml: ne = [W, H, F, 8*C_out]; we re-interpret as
                //   [W*2, H*2, F*2, C_out] (contiguous reshape).
                int64_t W = h->ne[0];
                int64_t H = h->ne[1];
                int64_t F = h->ne[2];
                int64_t C = h->ne[3];
                int64_t C_out_real = C / 8;
                h = ggml_cont(ctx->ggml_ctx, h);
                h = ggml_reshape_4d(ctx->ggml_ctx, h, W * 2, H * 2, F * 2, C_out_real);
            }

            for (int64_t i = 0; i < num_layers; ++i) {
                auto rn = std::dynamic_pointer_cast<LTXResnetBlock3d>(
                    blocks["resnets." + std::to_string(i)]);
                h = rn->forward(ctx, h, temb);
            }
            return h;
        }
    };

    // Encoder3d — diffusers' LTXVideoEncoder3d. Produces `latent_channels + 1`
    // channel outputs per position; the final row is replicated to form
    // `2*latent_channels - 1` posterior channels (see diffusers line 872-874).
    class LTXVideoEncoder3d : public GGMLBlock {
    protected:
        int patch_size;
        int patch_size_t;
        int64_t in_channels_patched;
        std::vector<int64_t> block_out_channels;
        std::vector<bool> spatio_temporal_scaling;
        std::vector<int> layers_per_block;

    public:
        LTXVideoEncoder3d(int64_t in_channels_arg = 3,
                          int64_t latent_channels  = 128,
                          std::vector<int64_t> block_out_channels = {128, 256, 512, 512},
                          std::vector<bool> spatio_temporal_scaling = {true, true, true, false},
                          std::vector<int> layers_per_block        = {4, 3, 3, 3, 4},
                          int patch_size                           = 4,
                          int patch_size_t                         = 1,
                          bool is_causal                           = true)
            : patch_size(patch_size),
              patch_size_t(patch_size_t),
              block_out_channels(block_out_channels),
              spatio_temporal_scaling(spatio_temporal_scaling),
              layers_per_block(layers_per_block) {
            in_channels_patched = in_channels_arg * patch_size * patch_size;
            int64_t out_ch      = block_out_channels[0];

            blocks["conv_in"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(
                in_channels_patched, out_ch, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, true, is_causal));
            int nb = (int)block_out_channels.size();
            for (int i = 0; i < nb; ++i) {
                int64_t ic = out_ch;
                int64_t oc = (i + 1 < nb) ? block_out_channels[i + 1] : block_out_channels[i];
                blocks["down_blocks." + std::to_string(i)] =
                    std::shared_ptr<GGMLBlock>(new LTXDownBlock3D(ic, oc, layers_per_block[i],
                                                                   spatio_temporal_scaling[i], is_causal));
                out_ch = oc;
            }
            blocks["mid_block"] = std::shared_ptr<GGMLBlock>(new LTXMidBlock3d(
                out_ch, layers_per_block.back(), is_causal, false));
            blocks["norm_out"]  = std::shared_ptr<GGMLBlock>(new VideoChannelRMSNorm(latent_channels, 1e-8f, false));
            blocks["conv_out"]  = std::shared_ptr<GGMLBlock>(new CausalConv3d(
                out_ch, latent_channels + 1, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, true, is_causal));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* h) {
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
            h            = conv_in->forward(ctx, h);

            int nb = (int)block_out_channels.size();
            for (int i = 0; i < nb; ++i) {
                auto db = std::dynamic_pointer_cast<LTXDownBlock3D>(blocks["down_blocks." + std::to_string(i)]);
                h       = db->forward(ctx, h);
            }

            auto mid      = std::dynamic_pointer_cast<LTXMidBlock3d>(blocks["mid_block"]);
            h             = mid->forward(ctx, h, nullptr);
            auto norm_out = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_out"]);
            h             = norm_out->forward(ctx, h);
            h             = ggml_silu_inplace(ctx->ggml_ctx, h);
            auto conv_out = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv_out"]);
            h             = conv_out->forward(ctx, h);
            return h;
        }
    };

    class LTXVideoDecoder3d : public GGMLBlock {
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
        LTXVideoDecoder3d(int64_t latent_channels = 128,
                          int64_t out_channels_arg = 3,
                          std::vector<int64_t> block_out_channels = {128, 256, 512, 512},
                          std::vector<bool> spatio_temporal_scaling = {true, true, true, false},
                          std::vector<int> layers_per_block         = {4, 3, 3, 3, 4},
                          int patch_size                            = 4,
                          int patch_size_t                          = 1,
                          bool is_causal                            = false,
                          bool timestep_conditioning                 = false)
            : patch_size(patch_size),
              patch_size_t(patch_size_t),
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
            blocks["conv_in"]   = std::shared_ptr<GGMLBlock>(new CausalConv3d(
                latent_channels, out_ch, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, true, is_causal));
            blocks["mid_block"] = std::shared_ptr<GGMLBlock>(new LTXMidBlock3d(
                out_ch, layers_per_block[0], is_causal, timestep_conditioning));

            int nb = (int)block_out_channels.size();
            for (int i = 0; i < nb; ++i) {
                int64_t ic = out_ch;
                int64_t oc = block_out_channels[i];
                blocks["up_blocks." + std::to_string(i)] =
                    std::shared_ptr<GGMLBlock>(new LTXUpBlock3d(ic, oc, layers_per_block[i + 1],
                                                                  spatio_temporal_scaling[i], is_causal,
                                                                  timestep_conditioning));
                out_ch = oc;
            }

            blocks["norm_out"] = std::shared_ptr<GGMLBlock>(new VideoChannelRMSNorm(out_ch, 1e-8f, false));
            blocks["conv_out"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(
                out_ch, out_channels_patched, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, true, is_causal));
            if (timestep_conditioning) {
                blocks["time_embedder.timestep_embedder.linear_1"] =
                    std::shared_ptr<GGMLBlock>(new Linear(256, out_ch * 2, true));
                blocks["time_embedder.timestep_embedder.linear_2"] =
                    std::shared_ptr<GGMLBlock>(new Linear(out_ch * 2, out_ch * 2, true));
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* z, ggml_tensor* temb_in = nullptr) {
            auto conv_in = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv_in"]);
            auto h       = conv_in->forward(ctx, z);

            ggml_tensor* temb_scaled = nullptr;
            if (timestep_conditioning && temb_in != nullptr) {
                ggml_tensor* mult = params["timestep_scale_multiplier"];
                temb_scaled       = ggml_mul(ctx->ggml_ctx, temb_in, mult);
            }

            auto mid = std::dynamic_pointer_cast<LTXMidBlock3d>(blocks["mid_block"]);
            h        = mid->forward(ctx, h, temb_scaled);

            int nb = (int)block_out_channels.size();
            for (int i = 0; i < nb; ++i) {
                auto ub = std::dynamic_pointer_cast<LTXUpBlock3d>(blocks["up_blocks." + std::to_string(i)]);
                h       = ub->forward(ctx, h, temb_scaled);
            }

            auto norm_out = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_out"]);
            h             = norm_out->forward(ctx, h);

            if (timestep_conditioning && temb_in != nullptr) {
                auto l1 = std::dynamic_pointer_cast<Linear>(blocks["time_embedder.timestep_embedder.linear_1"]);
                auto l2 = std::dynamic_pointer_cast<Linear>(blocks["time_embedder.timestep_embedder.linear_2"]);
                auto f  = ggml_ext_timestep_embedding(ctx->ggml_ctx, temb_scaled, 256);
                f       = l1->forward(ctx, f);
                f       = ggml_silu_inplace(ctx->ggml_ctx, f);
                f       = l2->forward(ctx, f);   // [out_ch*2, N]
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
            h             = conv_out->forward(ctx, h);

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

    class CausalVideoAutoencoder : public GGMLBlock {
    public:
        int64_t latent_channels;

        CausalVideoAutoencoder(bool decode_only       = true,
                                int64_t in_channels    = 3,
                                int64_t out_channels   = 3,
                                int64_t latent_channels = 128,
                                bool timestep_conditioning = true,
                                bool encoder_causal    = true,
                                bool decoder_causal    = false)
            : latent_channels(latent_channels) {
            if (!decode_only) {
                blocks["encoder"] = std::shared_ptr<GGMLBlock>(new LTXVideoEncoder3d(
                    in_channels, latent_channels,
                    {128, 256, 512, 512}, {true, true, true, false}, {4, 3, 3, 3, 4},
                    4, 1, encoder_causal));
            }
            blocks["decoder"] = std::shared_ptr<GGMLBlock>(new LTXVideoDecoder3d(
                latent_channels, out_channels,
                {128, 256, 512, 512}, {true, true, true, false}, {4, 3, 3, 3, 4},
                4, 1, decoder_causal, timestep_conditioning));
        }

        ggml_tensor* decode(GGMLRunnerContext* ctx, ggml_tensor* z, ggml_tensor* temb_in = nullptr) {
            auto dec = std::dynamic_pointer_cast<LTXVideoDecoder3d>(blocks["decoder"]);
            return dec->forward(ctx, z, temb_in);
        }

        ggml_tensor* encode(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto enc = std::dynamic_pointer_cast<LTXVideoEncoder3d>(blocks["encoder"]);
            return enc->forward(ctx, x);
        }
    };

    // VAE runner plugged into sd.cpp's VAE abstract class.
    struct LTXVVAERunner : public VAE {
        float scale_factor = 1.0f;
        bool decode_only   = true;
        CausalVideoAutoencoder ae;

        LTXVVAERunner(SDVersion version,
                      ggml_backend_t backend,
                      bool offload_params_to_cpu,
                      const String2TensorStorage& tensor_storage_map = {},
                      const std::string prefix                       = "first_stage_model",
                      bool decode_only                               = true)
            : VAE(version, backend, offload_params_to_cpu),
              decode_only(decode_only),
              ae(decode_only) {
            scale_input = false;  // LTX latents are not in [-1, 1] domain.
            ae.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override { return "ltxv_vae"; }

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

        sd::Tensor<float> diffusion_to_vae_latents(const sd::Tensor<float>& latents) override {
            return latents;
        }

        sd::Tensor<float> vae_to_diffusion_latents(const sd::Tensor<float>& latents) override {
            return latents;
        }

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
