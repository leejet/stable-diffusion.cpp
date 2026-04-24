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

    // Debug probe registry: block forwards add intermediate tensors here;
    // the Runner keeps them alive across compute and logs stats.
    struct DebugProbes {
        struct Entry {
            std::string name;
            ggml_tensor* tensor = nullptr;
        };
        std::vector<Entry> entries;
        void add(const std::string& n, ggml_tensor* t) {
            entries.push_back({n, t});
        }
        void clear() { entries.clear(); }
    };
    __STATIC_INLINE__ DebugProbes& debug_probes() {
        static DebugProbes p;
        return p;
    }

    // 3-D depth-to-space (pixel-shuffle) matching einops
    //   rearrange(x, "b (c p1 p2 p3) f h w -> b c (f p1) (h p2) (w p3)")
    // where the channel axis has structure (c outer, p1, p2, p3 inner). In
    // ggml ne order the input is [W, H, F, C*p1*p2*p3] and the output is
    // [W*p3, H*p2, F*p1, C]. Implemented as three separate passes that each
    // peel one sub-axis off the channel, route it to its destination and
    // merge it as the INNER sub-index — matching einops' conventions
    // exactly (naive ggml_reshape_4d alone produces swapped sub-indices and
    // causes the visible banding artefacts in decoded frames).
    __STATIC_INLINE__ ggml_tensor* depth_to_space_3d(ggml_context* ctx,
                                                     ggml_tensor* x,
                                                     int p1, int p2, int p3) {
        int64_t W = x->ne[0], H = x->ne[1], F = x->ne[2], Cb = x->ne[3];
        int64_t C = Cb / ((int64_t)p1 * p2 * p3);
        GGML_ASSERT(C * p1 * p2 * p3 == Cb);

        // ---- pass p3: merge into W as inner sub-index ----------------
        if (p3 > 1) {
            // Split p3 from channel into F*p3 (p3 outer within ne[2]).
            x = ggml_reshape_4d(ctx, x, W, H, F * p3, C * p1 * p2);
            // Isolate p3: ne=[W, H*F, p3, X].
            x = ggml_reshape_4d(ctx, x, W, H * F, p3, C * p1 * p2);
            // Bring p3 innermost: [p3, W, H*F, X].
            x = ggml_cont(ctx, ggml_ext_torch_permute(ctx, x, 2, 0, 1, 3));
            // Merge p3 with W (p3 inner, w outer) and restore H, F.
            x = ggml_reshape_4d(ctx, x, p3 * W, H, F, C * p1 * p2);
            W *= p3;
        }

        // ---- pass p2: merge into H as inner sub-index ----------------
        if (p2 > 1) {
            x = ggml_reshape_4d(ctx, x, W, H, F * p2, C * p1);
            // Isolate p2: ne=[W*H, F, p2, X].
            x = ggml_reshape_4d(ctx, x, W * H, F, p2, C * p1);
            // Bring p2 next to W*H: [W*H, p2, F, X].
            x = ggml_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 1, 3));
            // Split W*H → (W inner, H outer): ne=[W, H, p2, F*X].
            x = ggml_reshape_4d(ctx, x, W, H, p2, F * C * p1);
            // Swap H ↔ p2 so that the next merge puts p2 inner of H.
            x = ggml_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 1, 3));
            // Merge p2 and H (p2 inner, h outer) and restore F, C*p1.
            x = ggml_reshape_4d(ctx, x, W, p2 * H, F, C * p1);
            H *= p2;
        }

        // ---- pass p1: merge into F as inner sub-index ----------------
        if (p1 > 1) {
            x = ggml_reshape_4d(ctx, x, W, H, F * p1, C);
            // Split F*p1 into separate F and p1 axes: ne=[W*H, F, p1, C].
            x = ggml_reshape_4d(ctx, x, W * H, F, p1, C);
            // Swap so p1 is inner of the merged F*p1: [W*H, p1, F, C].
            x = ggml_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 1, 3));
            // Merge p1 with F (p1 inner, f outer) and restore W, H.
            x = ggml_reshape_4d(ctx, x, W, H, p1 * F, C);
            F *= p1;
        }

        return x;
    }

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
                             ggml_tensor* attention_mask        = nullptr,
                             const char* probe_prefix           = nullptr) {
            auto to_q    = std::dynamic_pointer_cast<Linear>(blocks["to_q"]);
            auto to_k    = std::dynamic_pointer_cast<Linear>(blocks["to_k"]);
            auto to_v    = std::dynamic_pointer_cast<Linear>(blocks["to_v"]);
            auto to_out  = std::dynamic_pointer_cast<Linear>(blocks["to_out.0"]);
            auto q_norm  = std::dynamic_pointer_cast<UnaryBlock>(blocks["q_norm"]);
            auto k_norm  = std::dynamic_pointer_cast<UnaryBlock>(blocks["k_norm"]);
            auto gate    = std::dynamic_pointer_cast<Linear>(blocks["to_gate_logits"]);

            auto probe_attn = [&](const char* suffix, ggml_tensor* t) {
                if (!probe_prefix) return;
                std::string full = std::string(probe_prefix) + "_" + suffix;
                auto dup = ggml_dup(ctx->ggml_ctx, t);
                ggml_set_name(dup, full.c_str());
                debug_probes().add(full, dup);
            };

            ggml_tensor* kv_src = encoder_hidden_states != nullptr ? encoder_hidden_states : hidden_states;
            probe_attn("kv_src", kv_src);
            probe_attn("q_src", hidden_states);

            auto gate_logits = gate->forward(ctx, hidden_states);
            probe_attn("gate_logits", gate_logits);

            auto q = to_q->forward(ctx, hidden_states);
            auto k = to_k->forward(ctx, kv_src);
            auto v = to_v->forward(ctx, kv_src);
            probe_attn("q_proj", q);
            probe_attn("k_proj", k);
            probe_attn("v_proj", v);

            q = q_norm->forward(ctx, q);
            k = k_norm->forward(ctx, k);
            probe_attn("q_norm", q);
            probe_attn("k_norm", k);

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
            probe_attn("raw_attn_out", out);

            // Per-head gate: gates = 2 * sigmoid(gate_logits). Broadcast
            // [heads, L_q, N] over head_dim via reshape to [1, heads, L_q, N].
            {
                auto gates  = ggml_sigmoid(ctx->ggml_ctx, gate_logits);
                gates       = ggml_scale(ctx->ggml_ctx, gates, 2.0f);
                int64_t N   = out->ne[2];
                int64_t L_q = out->ne[1];
                auto out_4d   = ggml_reshape_4d(ctx->ggml_ctx, out, head_dim, num_heads, L_q, N);
                auto gates_4d = ggml_reshape_4d(ctx->ggml_ctx, gates, 1, num_heads, L_q, N);
                out_4d        = ggml_mul(ctx->ggml_ctx, out_4d, gates_4d);
                out           = ggml_reshape_3d(ctx->ggml_ctx, out_4d, inner_dim, L_q, N);
            }
            probe_attn("after_gate", out);

            out = to_out->forward(ctx, out);
            probe_attn("to_out", out);
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

    // EmbeddingsConnector's internal transformer_1d_blocks: attn1 + ff with
    // PRE-NORM (stateless rms_norm) before each op. The reference is
    // Lightricks' Embeddings1DConnector._BasicTransformerBlock1D: it calls
    // `rms_norm(h)` before attn1 and before ff; residuals add the un-normed
    // input back. Without the pre-norms, residual magnitudes compound across
    // the 8 blocks and drive the connector output to ~1e12.
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
            auto xn    = ggml_rms_norm(ctx->ggml_ctx, x, 1e-6f);
            auto a     = attn1->forward(ctx, xn);
            x          = ggml_add(ctx->ggml_ctx, x, a);
            auto xn2   = ggml_rms_norm(ctx->ggml_ctx, x, 1e-6f);
            auto f     = ff->forward(ctx, xn2);
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
            // Final stateless rms_norm (matches reference).
            x = ggml_rms_norm(ctx->ggml_ctx, x, 1e-6f);
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
                             ggml_tensor* encoder_mask = nullptr,
                             int block_idx            = -1) {
            auto attn1 = std::dynamic_pointer_cast<LTXAttention>(blocks["attn1"]);
            auto attn2 = std::dynamic_pointer_cast<LTXAttention>(blocks["attn2"]);
            auto ff    = std::dynamic_pointer_cast<FeedForward>(blocks["ff"]);

            auto probe_tensor = [&](const char* name, ggml_tensor* t) {
                if (block_idx == 0) {
                    auto dup = ggml_dup(ctx->ggml_ctx, t);
                    ggml_set_name(dup, name);
                    debug_probes().add(name, dup);
                }
            };

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

            const char* dbg_mode = std::getenv("LTXV_DEBUG_MODE");
            bool skip_mod       = dbg_mode && std::strstr(dbg_mode, "no_mod");
            bool skip_attn1     = dbg_mode && std::strstr(dbg_mode, "no_attn1");
            bool skip_attn2     = dbg_mode && std::strstr(dbg_mode, "no_attn2");
            bool skip_ff        = dbg_mode && std::strstr(dbg_mode, "no_ff");
            bool skip_scale     = dbg_mode && std::strstr(dbg_mode, "no_scale");
            bool skip_shift     = dbg_mode && std::strstr(dbg_mode, "no_shift");
            bool skip_gate      = dbg_mode && std::strstr(dbg_mode, "no_gate");
            bool ret_h_norm1    = dbg_mode && std::strstr(dbg_mode, "ret=h_norm1");
            bool ret_scale_msa  = dbg_mode && std::strstr(dbg_mode, "ret=scale_msa");
            bool ret_attn1_out  = dbg_mode && std::strstr(dbg_mode, "ret=attn1_out");

            if (ret_scale_msa) {
                // Broadcast scale_msa to hidden shape so the caller's reshape works.
                // scale_msa is [dim, T_temb, N]; hidden is [dim, L, N]. Broadcast
                // the first axis with repeat.
                auto target = ggml_new_tensor_3d(ctx->ggml_ctx, scale_msa->type,
                                                  scale_msa->ne[0], hidden->ne[1], scale_msa->ne[2]);
                return ggml_repeat(ctx->ggml_ctx, scale_msa, target);
            }

            probe_tensor("blk0_hidden_in", hidden);
            probe_tensor("blk0_encoder_in", encoder);
            probe_tensor("blk0_scale_msa", scale_msa);
            probe_tensor("blk0_shift_msa", shift_msa);
            probe_tensor("blk0_gate_msa", gate_msa);
            probe_tensor("blk0_scale_text_q", scale_text_q);
            probe_tensor("blk0_shift_text_q", shift_text_q);
            probe_tensor("blk0_gate_text_q", gate_text_q);

            // 1. Video self-attention
            auto h_norm = ggml_rms_norm(ctx->ggml_ctx, hidden, 1e-6f);
            probe_tensor("blk0_after_norm1", h_norm);
            if (!skip_mod) {
                if (!skip_scale) {
                    h_norm = ggml_add(ctx->ggml_ctx, h_norm, ggml_mul(ctx->ggml_ctx, h_norm, scale_msa));
                }
                if (!skip_shift) {
                    h_norm = ggml_add(ctx->ggml_ctx, h_norm, shift_msa);
                }
            }
            probe_tensor("blk0_after_mod1", h_norm);
            if (ret_h_norm1) {
                return h_norm;
            }
            if (!skip_attn1) {
                auto attn_out = attn1->forward(ctx, h_norm, nullptr,
                                                rope_cos, rope_sin, nullptr, nullptr, nullptr);
                probe_tensor("blk0_after_attn1", attn_out);
                if (ret_attn1_out) {
                    return attn_out;
                }
                if (!skip_mod && !skip_gate) {
                    hidden = ggml_add(ctx->ggml_ctx, hidden, ggml_mul(ctx->ggml_ctx, attn_out, gate_msa));
                } else {
                    hidden = ggml_add(ctx->ggml_ctx, hidden, attn_out);
                }
                probe_tensor("blk0_after_attn1_residual", hidden);
            }

            // 2. Prompt cross-attention with Q modulation
            auto h_norm2 = ggml_rms_norm(ctx->ggml_ctx, hidden, 1e-6f);
            probe_tensor("blk0_after_norm2", h_norm2);
            if (!skip_mod) {
                h_norm2 = ggml_add(ctx->ggml_ctx, h_norm2, ggml_mul(ctx->ggml_ctx, h_norm2, scale_text_q));
                h_norm2 = ggml_add(ctx->ggml_ctx, h_norm2, shift_text_q);
            }
            probe_tensor("blk0_after_mod2", h_norm2);
            if (!skip_attn2) {
                const char* attn2_prefix = (block_idx == 0) ? "blk0_attn2" : nullptr;
                auto ca_out = attn2->forward(ctx, h_norm2, encoder,
                                              nullptr, nullptr, nullptr, nullptr, encoder_mask,
                                              attn2_prefix);
                probe_tensor("blk0_after_attn2", ca_out);
                if (!skip_mod) {
                    ca_out = ggml_mul(ctx->ggml_ctx, ca_out, gate_text_q);
                }
                hidden = ggml_add(ctx->ggml_ctx, hidden, ca_out);
                probe_tensor("blk0_after_attn2_residual", hidden);
            }

            // 3. a2v/v2a cross-attention — SKIPPED (video-only mode).

            // 4. FFN
            auto h_norm3 = ggml_rms_norm(ctx->ggml_ctx, hidden, 1e-6f);
            probe_tensor("blk0_after_norm3", h_norm3);
            if (!skip_mod) {
                h_norm3 = ggml_add(ctx->ggml_ctx, h_norm3, ggml_mul(ctx->ggml_ctx, h_norm3, scale_mlp));
                h_norm3 = ggml_add(ctx->ggml_ctx, h_norm3, shift_mlp);
            }
            probe_tensor("blk0_after_mod3", h_norm3);
            if (!skip_ff) {
                auto ff_out = ff->forward(ctx, h_norm3);
                probe_tensor("blk0_after_ff", ff_out);
                if (!skip_mod) {
                    hidden = ggml_add(ctx->ggml_ctx, hidden, ggml_mul(ctx->ggml_ctx, ff_out, gate_mlp));
                } else {
                    hidden = ggml_add(ctx->ggml_ctx, hidden, ff_out);
                }
                probe_tensor("blk0_after_ff_residual", hidden);
            }
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

            // Force F32 on patchify weights: the combination of tiny in_channels
            // (128) and BF16 storage triggers a matmul pathway that gives wildly
            // wrong magnitudes on some ggml backends (observed 6e9x explosion).
            blocks["patchify_proj"]       = std::shared_ptr<GGMLBlock>(new Linear(in_channels, inner_dim, true, /*force_f32=*/true));
            blocks["audio_patchify_proj"] = std::shared_ptr<GGMLBlock>(new Linear(audio_in_channels, audio_inner_dim, true, /*force_f32=*/true));

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

            const char* probe = std::getenv("LTXV_PROBE");
            const char* stage_env = std::getenv("LTXV_PROBE_STAGE");
            int stage = stage_env ? std::atoi(stage_env) : -1;

            (void)stage;
            (void)probe;

            auto& probes = debug_probes();
            probes.clear();

            auto dup_hs = ggml_dup(ctx->ggml_ctx, hidden_states);
            ggml_set_name(dup_hs, "dbg_patchify_in");
            probes.add("dbg_patchify_in", dup_hs);

            auto x = patchify->forward(ctx, hidden_states);

            auto dup_x = ggml_dup(ctx->ggml_ctx, x);
            ggml_set_name(dup_x, "dbg_after_patchify");
            probes.add("dbg_after_patchify", dup_x);
            if (probe && std::strcmp(probe, "after_proj_in") == 0) {
                ggml_set_name(x, "ltxv_probe_out");
                return ggml_cont(ctx->ggml_ctx, x);
            }

            auto te_pair           = adaln->forward(ctx, timestep);
            auto temb              = te_pair.first;
            auto embedded_timestep = te_pair.second;
            if (probe && std::strcmp(probe, "temb") == 0) {
                ggml_set_name(temb, "ltxv_probe_out");
                // temb shape doesn't match transformer output expected shape;
                // skip rest of forward by returning early — this corrupts the
                // sampler but is acceptable for diagnostic-only runs.
                return ggml_cont(ctx->ggml_ctx, temb);
            }
            if (probe && std::strcmp(probe, "embedded_timestep") == 0) {
                ggml_set_name(embedded_timestep, "ltxv_probe_out");
                return ggml_cont(ctx->ggml_ctx, embedded_timestep);
            }

            temb = ggml_reshape_4d(ctx->ggml_ctx, temb, temb->ne[0], 1, temb->ne[1], 1);

            auto encoder = connector->forward(ctx, encoder_hidden_states);

            int64_t max_i = num_layers;
            const char* dbg_env = std::getenv("LTXV_DEBUG_MAX_LAYERS");
            if (dbg_env) {
                int64_t dbg = std::atoi(dbg_env);
                if (dbg > 0 && dbg < max_i) max_i = dbg;
            }
            for (int64_t i = 0; i < max_i; ++i) {
                auto blk = std::dynamic_pointer_cast<LTX2VideoTransformerBlock>(
                    blocks["transformer_blocks." + std::to_string(i)]);
                x = blk->forward(ctx, x, encoder, temb, rope_cos, rope_sin, encoder_mask, (int)i);
                // Probe the first few block outputs.
                if (i < 3) {
                    auto dup = ggml_dup(ctx->ggml_ctx, x);
                    std::string name = "dbg_after_block" + std::to_string(i);
                    ggml_set_name(dup, name.c_str());
                    debug_probes().add(name, dup);
                }
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
            // norm_out (LayerNorm, elementwise_affine=False, eps=1e-6) —
            // matches reference LTXModel._process_output. Without this the
            // post-block activations (std≈200+ after 48 layers) leak into
            // the predicted velocity and the sampler diverges.
            x = ggml_norm(ctx->ggml_ctx, x, 1e-6f);
            x = ggml_add(ctx->ggml_ctx, x, ggml_mul(ctx->ggml_ctx, x, scale));
            x = ggml_add(ctx->ggml_ctx, x, shift);
            x = proj_out->forward(ctx, x);
            return x;
        }
    };

    // =================================================================
    // Transformer runner
    // =================================================================

    // Globally-mutable "probe table" that any forward path can push named
    // intermediates into. LTXVRunner::compute then reads them after running
    // the graph and logs stats. Keeps the probe infrastructure out of the
    // block forward signatures.
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

        // Debug: override to only run the first N transformer blocks during
        // build_graph. Set via LTXV_DEBUG_MAX_LAYERS env var (0 = all).
        int debug_max_layers() const {
            const char* e = std::getenv("LTXV_DEBUG_MAX_LAYERS");
            return e ? std::atoi(e) : 0;
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
            ggml_tensor* rope_cos = nullptr;
            ggml_tensor* rope_sin = nullptr;
            const char* probe_stage_env = std::getenv("LTXV_PROBE_STAGE");
            if (!probe_stage_env) {
                rope_tbl = compute_rope_ltx2((int)F, (int)H, (int)W, (int)dit.inner_dim, /*split_rope=*/true);
                rope_cos = ggml_new_tensor_2d(compute, GGML_TYPE_F32,
                                               rope_tbl.dim, rope_tbl.L);
                rope_sin = ggml_new_tensor_2d(compute, GGML_TYPE_F32,
                                               rope_tbl.dim, rope_tbl.L);
                set_backend_tensor_data(rope_cos, rope_tbl.cos.data());
                set_backend_tensor_data(rope_sin, rope_tbl.sin.data());
            }

            // Flatten the latent grid into tokens. Note: the exact (f, h, w)
            // order implied by this permute doesn't perfectly match RoPE's
            // meshgrid ordering — that's a TODO flagged in docs/ltxv.md.
            // Using the previously-validated permute that at least produces
            // a consistent round-trip shape.
            auto hidden = ggml_ext_cont(compute,
                                        ggml_ext_torch_permute(compute, x_t, 3, 0, 1, 2));
            hidden      = ggml_reshape_3d(compute, hidden, C, W * H * F, 1);

            const char* bypass    = std::getenv("LTXV_BYPASS");
            const char* stage_env = std::getenv("LTXV_PROBE_STAGE");
            bool skip_final_reshape = stage_env != nullptr;
            ggml_tensor* out;
            if (bypass && std::strlen(bypass) > 0) {
                out = ggml_cont(compute, x_t);
            } else {
                auto rctx = get_context();
                out = dit.forward(&rctx, hidden, c_t, ts_t, rope_cos, rope_sin, m_t);
                if (!skip_final_reshape) {
                    out = ggml_reshape_4d(compute, out, C, W, H, F);
                    out = ggml_ext_cont(compute, ggml_ext_torch_permute(compute, out, 1, 2, 3, 0));
                }
            }

            // Expand probes first, then `out` last so it remains the
            // graph's final node (which get_compute_graph names final_result).
            for (auto& p : debug_probes().entries) {
                if (p.tensor) ggml_build_forward_expand(gf, p.tensor);
            }
            ggml_build_forward_expand(gf, out);
            return gf;
        }

        // Dump min/max/mean/stddev of an sd::Tensor to the log.
        // Used to locate where the forward path becomes seed-invariant.
        template <typename T>
        static void log_tensor_stats(const char* label, const sd::Tensor<T>& t) {
            if (t.empty()) {
                LOG_INFO("[ltxv.stats] %s: EMPTY", label);
                return;
            }
            const int64_t n = t.numel();
            double mn = 1e30, mx = -1e30, sum = 0.0, sum_sq = 0.0;
            size_t nan_count = 0;
            const T* data = t.data();
            for (int64_t i = 0; i < n; ++i) {
                double v = static_cast<double>(data[i]);
                if (std::isnan(v)) {
                    ++nan_count;
                    continue;
                }
                if (v < mn) mn = v;
                if (v > mx) mx = v;
                sum += v;
                sum_sq += v * v;
            }
            int64_t valid = n - static_cast<int64_t>(nan_count);
            double mean = valid > 0 ? sum / valid : 0;
            double var  = valid > 0 ? sum_sq / valid - mean * mean : 0;
            double sd   = var > 0 ? std::sqrt(var) : 0;
            std::string shape_str;
            for (size_t i = 0; i < t.shape().size(); ++i) {
                if (i) shape_str += "x";
                shape_str += std::to_string(t.shape()[i]);
            }
            LOG_INFO("[ltxv.stats] %s: shape=[%s] n=%ld min=%.6g max=%.6g mean=%.6g std=%.6g nan=%zu",
                     label, shape_str.c_str(), (long)n, mn, mx, mean, sd, nan_count);
        }

        sd::Tensor<float> compute(int n_threads,
                                  const sd::Tensor<float>& x,
                                  const sd::Tensor<float>& timesteps,
                                  const sd::Tensor<float>& context,
                                  const sd::Tensor<float>& mask) {
            log_tensor_stats("transformer_in_x", x);
            log_tensor_stats("transformer_in_timesteps", timesteps);
            log_tensor_stats("transformer_in_context", context);

            const char* bypass = std::getenv("LTXV_BYPASS");
            if (bypass && std::strlen(bypass) > 0) {
                // Bypass the entire transformer compute: return the input
                // unchanged so the VAE sees seed-dependent data and we can
                // validate the rest of the pipeline.
                LOG_INFO("[ltxv.stats] transformer bypassed (LTXV_BYPASS set)");
                return x;
            }

            auto get_graph = [&]() -> struct ggml_cgraph* {
                return build_graph(x, timesteps, context,
                                   mask.empty() ? nullptr : &mask);
            };
            auto result = GGMLRunner::compute<float>(get_graph, n_threads, false);
            if (!result.has_value()) return {};
            sd::Tensor<float> out = std::move(*result);
            log_tensor_stats("transformer_out", out);
            // Dump any debug-tagged intermediate tensor from the graph so we
            // can compare against the PyTorch reference. We enumerate every
            // registered probe rather than a hardcoded list so new probes
            // (e.g. blk0_*) are picked up automatically.
            std::vector<std::string> probe_names;
            for (auto& p : debug_probes().entries) {
                probe_names.push_back(p.name);
            }
            for (const auto& nm : probe_names) {
                const char* name = nm.c_str();
                ggml_tensor* t = ggml_get_tensor(compute_ctx, name);
                if (!t) continue;
                const size_t nb = ggml_nbytes(t);
                std::vector<float> cpu(ggml_nelements(t));
                if (t->type == GGML_TYPE_F32) {
                    ggml_backend_tensor_get(t, cpu.data(), 0, nb);
                } else if (t->type == GGML_TYPE_F16) {
                    std::vector<uint16_t> tmp(ggml_nelements(t));
                    ggml_backend_tensor_get(t, tmp.data(), 0, nb);
                    for (size_t i = 0; i < cpu.size(); ++i) {
                        cpu[i] = ggml_fp16_to_fp32(tmp[i]);
                    }
                } else {
                    LOG_INFO("[ltxv.stats] %s: type=%d (skipping stats)", name, (int)t->type);
                    continue;
                }
                double mn = 1e30, mx = -1e30, sum = 0, sum_sq = 0;
                size_t nan_count = 0;
                for (float v : cpu) {
                    if (std::isnan(v)) { ++nan_count; continue; }
                    if (v < mn) mn = v;
                    if (v > mx) mx = v;
                    sum += v; sum_sq += v * v;
                }
                size_t valid = cpu.size() - nan_count;
                double mean = valid > 0 ? sum / valid : 0;
                double var  = valid > 0 ? sum_sq / valid - mean * mean : 0;
                double sd   = var > 0 ? std::sqrt(var) : 0;
                LOG_INFO("[ltxv.stats] %s: shape=[%lld,%lld,%lld,%lld] n=%zu min=%.4g max=%.4g mean=%.4g std=%.4g nan=%zu",
                         name,
                         (long long)t->ne[0], (long long)t->ne[1],
                         (long long)t->ne[2], (long long)t->ne[3],
                         cpu.size(), mn, mx, mean, sd, nan_count);
            }
            return out;
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

    // VAE residual block — diffusers' LTX2VideoResnetBlock3d simplified for
    // the LTX-2.3 checkpoint layout (no timestep conditioning, no shortcut
    // conv, no learned affine in the norms).
    //   norm1 (PerChannelRMSNorm stateless) → silu → conv1 →
    //   norm2 → silu → conv2 → + residual
    class VAEResBlock : public GGMLBlock {
    protected:
        int64_t channels;

    public:
        VAEResBlock(int64_t channels) : channels(channels) {
            // PerChannelRMSNorm is stateless (no weight), so the checkpoint
            // has no norm tensors for these — we just do the arithmetic
            // before each conv to keep activations bounded.
            blocks["conv1"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(channels, channels, {3, 3, 3}));
            blocks["conv2"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(channels, channels, {3, 3, 3}));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* h, bool causal = true) {
            auto conv1 = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv1"]);
            auto conv2 = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv2"]);
            auto residual = h;
            // Stateless per-channel RMS normalisation to bound activations.
            h = ggml_ext_cont(ctx->ggml_ctx,
                              ggml_ext_torch_permute(ctx->ggml_ctx, h, 3, 0, 1, 2));
            h = ggml_rms_norm(ctx->ggml_ctx, h, 1e-8f);
            h = ggml_ext_cont(ctx->ggml_ctx,
                              ggml_ext_torch_permute(ctx->ggml_ctx, h, 1, 2, 3, 0));
            h = ggml_silu_inplace(ctx->ggml_ctx, h);
            h = conv1->forward(ctx, h, causal);
            h = ggml_ext_cont(ctx->ggml_ctx,
                              ggml_ext_torch_permute(ctx->ggml_ctx, h, 3, 0, 1, 2));
            h = ggml_rms_norm(ctx->ggml_ctx, h, 1e-8f);
            h = ggml_ext_cont(ctx->ggml_ctx,
                              ggml_ext_torch_permute(ctx->ggml_ctx, h, 1, 2, 3, 0));
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
            h = ggml_cont(ctx->ggml_ctx, h);
            h = depth_to_space_3d(ctx->ggml_ctx, h, st_t, st_h, st_w);
            // Diffusers LTX2VideoUpsampler3d drops the first (st_t - 1) temporal
            // samples so each upsampled chunk boundary stays causal and the
            // overall frame count follows f_out = (f_in - 1) * st_t + 1 when
            // composed across multiple temporal upsamples.
            if (st_t > 1) {
                int64_t T_out  = h->ne[2];
                int64_t T_keep = T_out - (st_t - 1);
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
            // conv_norm_out (stateless PerChannelRMSNorm) + SiLU before conv_out,
            // matching the reference video_vae decoder. Without these the
            // output is O(1000) instead of O(1) per pixel.
            {
                PerChannelRMSNorm pn;
                h = pn.forward(ctx, h);
            }
            h = ggml_silu(ctx->ggml_ctx, h);
            h = conv_out->forward(ctx, h, causal);
            // Un-patchify 4×4 spatial pack: ne [W, H, F, C*16] → [W*4, H*4, F, C]
            h = ggml_cont(ctx->ggml_ctx, h);
            h = depth_to_space_3d(ctx->ggml_ctx, h, /*p1=*/1, /*p2=*/4, /*p3=*/4);
            // sd.cpp's decode_video_outputs expects the 5-D layout
            //   [W, H, T, C, N=1]
            // (batch last, time before channel). Our 4-D result is
            //   [W, H, T, C] — reinterpret by prepending N=1 to match.
            h = ggml_reshape_4d(ctx->ggml_ctx, h, h->ne[0], h->ne[1], h->ne[2], h->ne[3]);
            // NOTE: ggml tensors are 4-D max. sd.cpp's tensor_to_sd_image
            // reads the dimensionality from the sd::Tensor's shape vector
            // (not from ggml ne), so we need to ensure the C++ side sees a
            // 5-D shape. That happens in LTXVVAERunner::_compute by
            // unsqueezing the resulting sd::Tensor before returning.
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
            // Keep scale_input=true (the sd.cpp default): the VAE::decode
            // output is mapped (x + 1) / 2 into [0, 1] before the frame
            // extraction. LTX-2.3's VAE is trained to produce values in
            // roughly [-1, 1] per-channel so this is the correct range.
            // scale_input = false  // <-- was here, caused black frames
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
        //   diffusion_to_vae (un_normalize)   = latents * std + mean
        //   vae_to_diffusion (normalize)      = (latents - mean) / std
        // The stats live in the backend under `ae.params["per_channel_statistics.*"]`;
        // we materialise them to CPU lazily on the first call.
        std::vector<float> mean_of_means;
        std::vector<float> std_of_means;
        bool stats_loaded = false;
        void load_stats_cpu() {
            if (stats_loaded) return;
            std::map<std::string, ggml_tensor*> tensors;
            ae.get_param_tensors(tensors);
            auto mm  = tensors.find("per_channel_statistics.mean-of-means");
            auto sm  = tensors.find("per_channel_statistics.std-of-means");
            if (mm == tensors.end() || sm == tensors.end() || !mm->second || !sm->second) return;
            ggml_tensor* m = mm->second;
            ggml_tensor* s = sm->second;
            int64_t C = m->ne[0];
            mean_of_means.resize(C);
            std_of_means.resize(C);
            ggml_backend_tensor_get(m, mean_of_means.data(), 0, C * sizeof(float));
            ggml_backend_tensor_get(s, std_of_means.data(),  0, C * sizeof(float));
            stats_loaded = true;
            LOG_INFO("[ltxv.stats] per-channel stats loaded: C=%lld mean[0..3]=%g %g %g std[0..3]=%g %g %g",
                     (long long)C,
                     mean_of_means[0], mean_of_means[1], mean_of_means[2],
                     std_of_means[0],  std_of_means[1],  std_of_means[2]);
        }
        // latents shape: [W, H, F, C, N] or [W, H, F, C] (missing batch axis).
        // The data layout is row-major with shape[0] the fastest-varying dim,
        // so index(w,h,f,c,n) = n*W*H*F*C + c*W*H*F + f*W*H + h*W + w.
        sd::Tensor<float> diffusion_to_vae_latents(const sd::Tensor<float>& latents) override {
            load_stats_cpu();
            if (!stats_loaded) return latents;
            sd::Tensor<float> out(latents.shape());
            const auto& sh = latents.shape();
            int64_t W = sh.size() > 0 ? sh[0] : 1;
            int64_t H = sh.size() > 1 ? sh[1] : 1;
            int64_t F = sh.size() > 2 ? sh[2] : 1;
            int64_t C = sh.size() > 3 ? sh[3] : 1;
            int64_t N = sh.size() > 4 ? sh[4] : 1;
            if ((size_t)C != mean_of_means.size()) return latents;
            const float* src = latents.data();
            float* dst       = out.data();
            int64_t plane    = W * H * F;
            for (int64_t n = 0; n < N; ++n) {
                for (int64_t c = 0; c < C; ++c) {
                    float mu = mean_of_means[c];
                    float sg = std_of_means[c];
                    int64_t off = (n * C + c) * plane;
                    for (int64_t i = 0; i < plane; ++i) {
                        dst[off + i] = src[off + i] * sg + mu;
                    }
                }
            }
            return out;
        }
        sd::Tensor<float> vae_to_diffusion_latents(const sd::Tensor<float>& latents) override {
            load_stats_cpu();
            if (!stats_loaded) return latents;
            sd::Tensor<float> out(latents.shape());
            const auto& sh = latents.shape();
            int64_t W = sh.size() > 0 ? sh[0] : 1;
            int64_t H = sh.size() > 1 ? sh[1] : 1;
            int64_t F = sh.size() > 2 ? sh[2] : 1;
            int64_t C = sh.size() > 3 ? sh[3] : 1;
            int64_t N = sh.size() > 4 ? sh[4] : 1;
            if ((size_t)C != mean_of_means.size()) return latents;
            const float* src = latents.data();
            float* dst       = out.data();
            int64_t plane    = W * H * F;
            for (int64_t n = 0; n < N; ++n) {
                for (int64_t c = 0; c < C; ++c) {
                    float mu = mean_of_means[c];
                    float sg = std_of_means[c];
                    int64_t off = (n * C + c) * plane;
                    for (int64_t i = 0; i < plane; ++i) {
                        dst[off + i] = (src[off + i] - mu) / sg;
                    }
                }
            }
            return out;
        }

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
            LTXVRunner::log_tensor_stats(decode_graph ? "vae_in_decode_z" : "vae_in_encode_x", z);
            auto get_graph = [&]() -> struct ggml_cgraph* {
                return decode_graph ? build_graph_decode(z) : build_graph_encode(z);
            };
            auto result = GGMLRunner::compute<float>(get_graph, n_threads, false);
            if (!result.has_value()) return {};
            sd::Tensor<float> out = std::move(*result);
            LTXVRunner::log_tensor_stats(decode_graph ? "vae_out_decode" : "vae_out_encode", out);
            if (decode_graph && out.dim() == 4) {
                out.unsqueeze_(out.dim());
            }
            return out;
        }
    };

}  // namespace LTXV

#endif  // __LTXV_HPP__
