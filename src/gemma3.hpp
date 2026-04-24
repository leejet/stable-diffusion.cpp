// Gemma-3 text encoder for LTX-Video 2.3 conditioning.
//
// Architecture reference: llama.cpp src/models/gemma3.cpp (LLM_ARCH_GEMMA3)
// and HuggingFace transformers modeling_gemma3.py.
//
// Only the *text* sub-model is implemented — LTX-2.3 feeds the prompt
// through Gemma-3-12B-it's 48 transformer layers and concatenates the 49
// resulting hidden states (input embedding + 48 layer outputs) along the
// last dim, then runs them through a per-modality linear (baked into the
// LTX-2.3 safetensors under `text_embedding_projection.*`) and through
// `video_embeddings_connector` to produce the cross-attention keys used
// by every block of the LTX video DiT.
//
// This file covers the GGML architecture + forward pass. Tokenizer and
// weight loading live in gemma3_tokenizer.{h,cpp} and gemma3_loader.{h,cpp}.
//
// Gemma-3-12B hyperparameters (from the model's config.json):
//   hidden_size = 3840       intermediate_size = 15360
//   num_attention_heads = 16 num_key_value_heads = 8   (GQA, 2:1 ratio)
//   head_dim = 256           num_hidden_layers = 48
//   rope_theta (global) = 1e6     rope_local_base_freq = 1e4
//   rope_scaling = linear factor 8    sliding_window = 1024
//   sliding_window_pattern = 6   (every 6th layer is full-attention)
//   rms_norm_eps = 1e-6
//   query_pre_attn_scalar = 256  (attn_scale = 1 / sqrt(256) = 0.0625)
//   hidden_activation = gelu_pytorch_tanh
//   vocab_size = 262144 (tokens) + 64 special = 262208

#ifndef __GEMMA3_HPP__
#define __GEMMA3_HPP__

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "common_block.hpp"
#include "ggml_extend.hpp"

namespace GEMMA3 {

    constexpr int GEMMA3_GRAPH_SIZE = 32768;

    struct Gemma3Params {
        int64_t hidden_size          = 3840;
        int64_t intermediate_size    = 15360;
        int64_t num_heads            = 16;
        int64_t num_kv_heads         = 8;
        int64_t head_dim             = 256;
        int64_t num_layers           = 48;
        int64_t vocab_size           = 262208;
        float rms_norm_eps           = 1e-6f;
        float rope_theta_global      = 1e6f;
        float rope_theta_local       = 1e4f;
        float rope_scaling_factor    = 8.0f;     // applied to GLOBAL rope only
        int sliding_window           = 1024;
        int sliding_window_pattern   = 6;        // global attn every Nth layer
        float query_pre_attn_scalar  = 256.0f;   // attn_scale = 1/sqrt(q_pre)
        float embed_scale_sqrt_embd  = 1.0f;     // filled in ctor (sqrt(hidden_size))
    };

    // Gemma-3 RMSNorm: applies `(1 + weight)` rather than `weight`, so the
    // checkpoint stores weights initialised at 0. Equivalent to
    //   out = x * rsqrt(mean(x^2) + eps) * (1 + w)
    class Gemma3RMSNorm : public UnaryBlock {
    protected:
        int64_t hidden_size;
        float eps;
        std::string prefix;

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, std::string prefix = "") override {
            this->prefix     = prefix;
            params["weight"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
        }

    public:
        Gemma3RMSNorm(int64_t hidden_size, float eps = 1e-6f)
            : hidden_size(hidden_size), eps(eps) {}

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            ggml_tensor* w = params["weight"];
            if (ctx->weight_adapter) {
                w = ctx->weight_adapter->patch_weight(ctx->ggml_ctx, w, prefix + "weight");
            }
            x = ggml_rms_norm(ctx->ggml_ctx, x, eps);
            // Equivalent to `x * (1 + w)` — add a fresh f32 "1" tensor of
            // matching shape, or use ggml_add with a constant. ggml_scale
            // on `x` would need two ops; cleanest is to materialise
            // `(1 + w)` at graph-build time, but `w` lives on the backend.
            // So we do it with two ggml ops: tmp = x * w + x  == x * (1+w).
            auto mul = ggml_mul(ctx->ggml_ctx, x, w);
            return ggml_add(ctx->ggml_ctx, mul, x);
        }
    };

    // Gemma-3 MLP: SwiGLU variant using GELU (pytorch_tanh approximation).
    //   out = down(gelu_tanh(gate(x)) * up(x))
    class Gemma3MLP : public GGMLBlock {
    public:
        Gemma3MLP(int64_t hidden_size, int64_t intermediate_size) {
            blocks["gate_proj"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, intermediate_size, /*bias=*/false));
            blocks["up_proj"]   = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, intermediate_size, /*bias=*/false));
            blocks["down_proj"] = std::shared_ptr<GGMLBlock>(new Linear(intermediate_size, hidden_size, /*bias=*/false));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto gate = std::dynamic_pointer_cast<Linear>(blocks["gate_proj"]);
            auto up   = std::dynamic_pointer_cast<Linear>(blocks["up_proj"]);
            auto down = std::dynamic_pointer_cast<Linear>(blocks["down_proj"]);

            auto g = gate->forward(ctx, x);
            g      = ggml_gelu_inplace(ctx->ggml_ctx, g);  // tanh-approx
            auto u = up->forward(ctx, x);
            auto h = ggml_mul_inplace(ctx->ggml_ctx, g, u);
            return down->forward(ctx, h);
        }
    };

    // Single Gemma-3 decoder block.
    //   attn_branch : pre_attn_norm -> (Q,K,V) -> q_norm/k_norm -> RoPE
    //                 -> GQA (sliding-window or global) -> post_attn_norm
    //                 -> residual
    //   ffn_branch  : pre_ffn_norm -> Gemma3MLP -> post_ffn_norm -> residual
    class Gemma3Block : public GGMLBlock {
    protected:
        Gemma3Params params_;
        int layer_idx;

    public:
        Gemma3Block(const Gemma3Params& p, int layer_idx) : params_(p), layer_idx(layer_idx) {
            int64_t q_dim  = p.num_heads * p.head_dim;
            int64_t kv_dim = p.num_kv_heads * p.head_dim;

            blocks["input_layernorm"]             = std::shared_ptr<GGMLBlock>(new Gemma3RMSNorm(p.hidden_size, p.rms_norm_eps));
            blocks["post_attention_layernorm"]    = std::shared_ptr<GGMLBlock>(new Gemma3RMSNorm(p.hidden_size, p.rms_norm_eps));
            blocks["pre_feedforward_layernorm"]   = std::shared_ptr<GGMLBlock>(new Gemma3RMSNorm(p.hidden_size, p.rms_norm_eps));
            blocks["post_feedforward_layernorm"]  = std::shared_ptr<GGMLBlock>(new Gemma3RMSNorm(p.hidden_size, p.rms_norm_eps));

            blocks["self_attn.q_proj"]            = std::shared_ptr<GGMLBlock>(new Linear(p.hidden_size, q_dim,  /*bias=*/false));
            blocks["self_attn.k_proj"]            = std::shared_ptr<GGMLBlock>(new Linear(p.hidden_size, kv_dim, /*bias=*/false));
            blocks["self_attn.v_proj"]            = std::shared_ptr<GGMLBlock>(new Linear(p.hidden_size, kv_dim, /*bias=*/false));
            blocks["self_attn.o_proj"]            = std::shared_ptr<GGMLBlock>(new Linear(q_dim, p.hidden_size, /*bias=*/false));
            blocks["self_attn.q_norm"]            = std::shared_ptr<GGMLBlock>(new Gemma3RMSNorm(p.head_dim, p.rms_norm_eps));
            blocks["self_attn.k_norm"]            = std::shared_ptr<GGMLBlock>(new Gemma3RMSNorm(p.head_dim, p.rms_norm_eps));

            blocks["mlp"] = std::shared_ptr<GGMLBlock>(new Gemma3MLP(p.hidden_size, p.intermediate_size));
        }

        // Returns (layer_output, residual_after_attn) — the latter is useful
        // for the final hidden-state list. We concatenate per-layer outputs
        // outside this class.
        //
        // rope_cos/rope_sin: precomputed per-token cos/sin tables. The caller
        // picks the right one (local for sliding layers, global for full).
        // attn_mask: [L, L] additive mask; caller builds the sliding-window
        // band or leaves nullptr for full-attention layers.
        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* rope_cos,
                             ggml_tensor* rope_sin,
                             ggml_tensor* attn_mask /* may be nullptr */) {
            auto in_norm   = std::dynamic_pointer_cast<Gemma3RMSNorm>(blocks["input_layernorm"]);
            auto post_attn = std::dynamic_pointer_cast<Gemma3RMSNorm>(blocks["post_attention_layernorm"]);
            auto pre_ffn   = std::dynamic_pointer_cast<Gemma3RMSNorm>(blocks["pre_feedforward_layernorm"]);
            auto post_ffn  = std::dynamic_pointer_cast<Gemma3RMSNorm>(blocks["post_feedforward_layernorm"]);

            auto q_proj    = std::dynamic_pointer_cast<Linear>(blocks["self_attn.q_proj"]);
            auto k_proj    = std::dynamic_pointer_cast<Linear>(blocks["self_attn.k_proj"]);
            auto v_proj    = std::dynamic_pointer_cast<Linear>(blocks["self_attn.v_proj"]);
            auto o_proj    = std::dynamic_pointer_cast<Linear>(blocks["self_attn.o_proj"]);
            auto q_norm    = std::dynamic_pointer_cast<Gemma3RMSNorm>(blocks["self_attn.q_norm"]);
            auto k_norm    = std::dynamic_pointer_cast<Gemma3RMSNorm>(blocks["self_attn.k_norm"]);
            auto mlp       = std::dynamic_pointer_cast<Gemma3MLP>(blocks["mlp"]);

            auto residual = x;

            // --- attention branch ---
            auto h = in_norm->forward(ctx, x);
            auto q = q_proj->forward(ctx, h);  // [q_dim, L, N]
            auto k = k_proj->forward(ctx, h);  // [kv_dim, L, N]
            auto v = v_proj->forward(ctx, h);  // [kv_dim, L, N]

            int64_t L = q->ne[1];
            int64_t N = q->ne[2];

            // q_norm / k_norm are PER-HEAD — reshape to expose head_dim on
            // the inner axis, apply RMSNorm, reshape back.
            q = ggml_reshape_4d(ctx->ggml_ctx, q, params_.head_dim, params_.num_heads,    L, N);
            k = ggml_reshape_4d(ctx->ggml_ctx, k, params_.head_dim, params_.num_kv_heads, L, N);
            v = ggml_reshape_4d(ctx->ggml_ctx, v, params_.head_dim, params_.num_kv_heads, L, N);

            q = q_norm->forward(ctx, q);
            k = k_norm->forward(ctx, k);

            // Apply RoPE to Q and K. Q has num_heads heads; K has num_kv_heads.
            // RoPE tables are shape [head_dim/2 or head_dim, L] depending on
            // variant — here we use interleaved (standard Gemma-3).
            q = apply_rotary_emb(ctx, q, rope_cos, rope_sin);
            k = apply_rotary_emb(ctx, k, rope_cos, rope_sin);

            // Scale Q by 1 / sqrt(query_pre_attn_scalar) — Gemma-3 applies
            // the scale to Q, not inside softmax.
            q = ggml_scale(ctx->ggml_ctx, q, 1.0f / std::sqrt(params_.query_pre_attn_scalar));

            // GQA: K and V each map to num_heads by repeat (num_heads /
            // num_kv_heads copies). ggml's attention helper handles this
            // when we pass K/V with num_kv_heads directly if the backend
            // supports broadcasting; otherwise we tile.
            auto attn_out = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend,
                                                    ggml_reshape_3d(ctx->ggml_ctx, q,
                                                                    params_.head_dim * params_.num_heads, L, N),
                                                    ggml_reshape_3d(ctx->ggml_ctx, k,
                                                                    params_.head_dim * params_.num_kv_heads, L, N),
                                                    ggml_reshape_3d(ctx->ggml_ctx, v,
                                                                    params_.head_dim * params_.num_kv_heads, L, N),
                                                    params_.num_heads,
                                                    attn_mask,
                                                    /*scale_for_sdp=*/false,
                                                    ctx->flash_attn_enabled);
            auto attn = o_proj->forward(ctx, attn_out);
            attn      = post_attn->forward(ctx, attn);
            x         = ggml_add(ctx->ggml_ctx, residual, attn);

            // --- FFN branch ---
            residual = x;
            auto ff  = pre_ffn->forward(ctx, x);
            ff       = mlp->forward(ctx, ff);
            ff       = post_ffn->forward(ctx, ff);
            return ggml_add(ctx->ggml_ctx, residual, ff);
        }

    private:
        // NEOX-style RoPE (matches Gemma-3 in llama.cpp): pair is
        //   (x[k], x[k + D/2])   for k in [0, D/2)
        // rotation:
        //   x_new[k]       = x[k]       * cos[k] - x[k + D/2] * sin[k]
        //   x_new[k + D/2] = x[k + D/2] * cos[k] + x[k]       * sin[k]
        // cos/sin are [D, L] (duplicated: cos[k] == cos[k+D/2], same for sin)
        // so we can apply via element-wise multiplies.
        static ggml_tensor* apply_rotary_emb(GGMLRunnerContext* ctx,
                                             ggml_tensor* x,
                                             ggml_tensor* cos,
                                             ggml_tensor* sin) {
            // x: [head_dim, n_heads, L, N]
            int64_t D = x->ne[0];
            int64_t H = x->ne[1];
            int64_t L = x->ne[2];
            int64_t N = x->ne[3];
            int64_t r = D / 2;

            // Split x along the head_dim axis into first half (k=0..r-1)
            // and second half (k=r..D-1), both shape [r, H, L, N].
            // In ggml ne order, ne[0] is innermost; use views into the
            // contiguous memory.
            auto first  = ggml_view_4d(ctx->ggml_ctx, x, r, H, L, N,
                                       x->nb[1], x->nb[2], x->nb[3], 0);
            auto second = ggml_view_4d(ctx->ggml_ctx, x, r, H, L, N,
                                       x->nb[1], x->nb[2], x->nb[3],
                                       x->nb[0] * r);
            first  = ggml_cont(ctx->ggml_ctx, first);
            second = ggml_cont(ctx->ggml_ctx, second);

            // cos / sin broadcast over (H, N).
            auto cos_b = ggml_reshape_4d(ctx->ggml_ctx, cos, D, 1, L, 1);
            auto sin_b = ggml_reshape_4d(ctx->ggml_ctx, sin, D, 1, L, 1);
            auto cos_first  = ggml_view_4d(ctx->ggml_ctx, cos_b, r, 1, L, 1,
                                           cos_b->nb[1], cos_b->nb[2], cos_b->nb[3], 0);
            auto sin_first  = ggml_view_4d(ctx->ggml_ctx, sin_b, r, 1, L, 1,
                                           sin_b->nb[1], sin_b->nb[2], sin_b->nb[3], 0);
            cos_first = ggml_cont(ctx->ggml_ctx, cos_first);
            sin_first = ggml_cont(ctx->ggml_ctx, sin_first);

            // first_new  = first * cos - second * sin
            // second_new = second * cos + first * sin
            auto first_new  = ggml_sub(ctx->ggml_ctx,
                                        ggml_mul(ctx->ggml_ctx, first, cos_first),
                                        ggml_mul(ctx->ggml_ctx, second, sin_first));
            auto second_new = ggml_add(ctx->ggml_ctx,
                                        ggml_mul(ctx->ggml_ctx, second, cos_first),
                                        ggml_mul(ctx->ggml_ctx, first, sin_first));

            // Concatenate back along head_dim.
            return ggml_concat(ctx->ggml_ctx, first_new, second_new, 0);
        }
    };

    // Full Gemma-3 text model: embedding + 48 decoder blocks + final RMSNorm.
    // Exposes `forward_with_hidden_states` that returns all 49 intermediate
    // hidden states (post-embedding + each of 48 layer outputs) so the LTX
    // embeddings processor can concatenate them.
    class Gemma3TextModel : public GGMLBlock {
    public:
        Gemma3Params params_;

        Gemma3TextModel(const Gemma3Params& p) : params_(p) {
            blocks["embed_tokens"] = std::shared_ptr<GGMLBlock>(new Embedding(p.vocab_size, p.hidden_size));
            for (int64_t i = 0; i < p.num_layers; ++i) {
                blocks["layers." + std::to_string(i)] =
                    std::shared_ptr<GGMLBlock>(new Gemma3Block(p, (int)i));
            }
            blocks["norm"] = std::shared_ptr<GGMLBlock>(new Gemma3RMSNorm(p.hidden_size, p.rms_norm_eps));
        }

        // input_ids: [L, N=1] int32
        // rope_cos_global / rope_sin_global: [head_dim, L] (global θ+scaling)
        // rope_cos_local  / rope_sin_local:  [head_dim, L] (local θ, no scaling)
        // sliding_mask: [L, L] additive mask with the 1024-band; full layers
        //               use nullptr.
        // hidden_out: caller-provided vector to receive intermediate tensors.
        //             After a full forward it will have num_layers+1
        //             entries: [post-embed, layer0_out, ..., layer_{N-1}_out].
        //             The last entry is REPLACED with the post-final-norm
        //             result on return.
        //
        // max_layers: run at most this many decoder blocks; -1 = all.
        //             When < num_layers, the final norm is NOT applied and
        //             hidden_out contains [post-embed, layer0_out, ...,
        //             layer_{max_layers-1}_out].
        ggml_tensor* forward_with_hidden_states(GGMLRunnerContext* ctx,
                                                ggml_tensor* input_ids,
                                                ggml_tensor* rope_cos_global,
                                                ggml_tensor* rope_sin_global,
                                                ggml_tensor* rope_cos_local,
                                                ggml_tensor* rope_sin_local,
                                                ggml_tensor* sliding_mask,
                                                ggml_tensor* full_mask,
                                                std::vector<ggml_tensor*>& hidden_out,
                                                int64_t max_layers = -1) {
            auto embed = std::dynamic_pointer_cast<Embedding>(blocks["embed_tokens"]);
            auto fnorm = std::dynamic_pointer_cast<Gemma3RMSNorm>(blocks["norm"]);

            auto x = embed->forward(ctx, input_ids);
            // Gemma paper: embeddings scaled by sqrt(hidden_size).
            x = ggml_scale(ctx->ggml_ctx, x, std::sqrt((float)params_.hidden_size));

            int64_t lim = max_layers < 0 ? params_.num_layers
                                          : std::min<int64_t>(max_layers, params_.num_layers);
            hidden_out.clear();
            hidden_out.reserve(lim + 1);
            hidden_out.push_back(x);

            for (int64_t i = 0; i < lim; ++i) {
                auto blk = std::dynamic_pointer_cast<Gemma3Block>(
                    blocks["layers." + std::to_string(i)]);
                bool is_global = ((i + 1) % params_.sliding_window_pattern) == 0;
                auto* cos = is_global ? rope_cos_global : rope_cos_local;
                auto* sin = is_global ? rope_sin_global : rope_sin_local;
                // Gemma-3 uses CAUSAL attention everywhere. Full-attention
                // layers get a plain causal mask; sliding layers get a
                // windowed causal mask. Caller provides both under the
                // `full_mask` / `sliding_mask` names.
                auto* msk = is_global ? full_mask : sliding_mask;
                x = blk->forward(ctx, x, cos, sin, msk);
                hidden_out.push_back(x);
            }
            if (lim == params_.num_layers) {
                x = fnorm->forward(ctx, x);
                // Replace last entry with post-final-norm.
                hidden_out.back() = x;
            }
            return x;
        }
    };

    // Precompute interleaved RoPE tables on CPU. The LTX pipeline encodes a
    // single short prompt (max ~256 tokens); we materialise the full
    // [head_dim, L] cos/sin once per run.
    struct RopeTables {
        std::vector<float> cos;
        std::vector<float> sin;
        int64_t L      = 0;
        int64_t dim    = 0;
    };

    __STATIC_INLINE__ RopeTables compute_gemma3_rope(int64_t L,
                                                      int64_t head_dim,
                                                      float theta,
                                                      float scaling_factor) {
        RopeTables t;
        t.L   = L;
        t.dim = head_dim;
        t.cos.assign(L * head_dim, 0.f);
        t.sin.assign(L * head_dim, 0.f);
        // NEOX RoPE layout: pairs are (x[k], x[k+D/2]).
        //   cos[pos*D + k]   = cos[pos*D + k + D/2] = cos(scaled_pos * freq_k)
        // i.e. the first half of the dim holds the values and the second
        // half is a duplicate — so `apply_rotary_emb` can just broadcast.
        // freq_k = 1 / theta^(2k / head_dim) for k in [0, D/2).
        int64_t half = head_dim / 2;
        for (int64_t pos = 0; pos < L; ++pos) {
            float scaled_pos = (float)pos / scaling_factor;
            for (int64_t k = 0; k < half; ++k) {
                float freq = 1.0f / std::pow(theta, (float)(2 * k) / (float)head_dim);
                float ang  = scaled_pos * freq;
                float c    = std::cos(ang);
                float s    = std::sin(ang);
                t.cos[pos * head_dim + k]        = c;
                t.cos[pos * head_dim + k + half] = c;
                t.sin[pos * head_dim + k]        = s;
                t.sin[pos * head_dim + k + half] = s;
            }
        }
        return t;
    }

    // Build an additive causal sliding-window mask of shape [L, L]:
    //   mask[i, j] = 0       if j <= i && i - j < window
    //              = -inf    otherwise
    // Gemma-3 uses causal attention for both sliding and full layers
    // (`use_bidirectional_attention = False` in the text_config). For full-
    // attention layers, pass `window = L` to get a plain causal mask.
    __STATIC_INLINE__ std::vector<float> build_causal_mask(int64_t L, int window) {
        std::vector<float> m(L * L, -INFINITY);
        for (int64_t i = 0; i < L; ++i) {
            int64_t lo = std::max<int64_t>(0, i - window + 1);
            for (int64_t j = lo; j <= i; ++j) {
                m[i * L + j] = 0.0f;
            }
        }
        return m;
    }

    // Back-compat shim.
    __STATIC_INLINE__ std::vector<float> build_sliding_mask(int64_t L, int window) {
        return build_causal_mask(L, window);
    }

    // GGMLRunner wrapper: allocates params_buffer, builds graph per call.
    // Owns two sets of precomputed RoPE tables (local + global) and the
    // sliding mask, uploaded to the backend per compute() invocation.
    struct Gemma3Runner : public GGMLRunner {
        Gemma3Params params;
        Gemma3TextModel model;
        RopeTables rope_global;
        RopeTables rope_local;
        std::vector<float> sliding_mask;
        std::vector<float> full_mask;

        Gemma3Runner(ggml_backend_t backend,
                     bool offload_params_to_cpu,
                     const String2TensorStorage& tensor_storage_map,
                     const std::string prefix = "model")
            : GGMLRunner(backend, offload_params_to_cpu), model(params) {
            model.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override { return "gemma3_12b"; }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors,
                               const std::string prefix) {
            model.get_param_tensors(tensors, prefix);
        }

        // Build graph, set RoPE + mask tensors, run, return the last layer's
        // hidden state (shape [hidden, L, 1]) as sd::Tensor. For LTX we will
        // also need the INTERMEDIATE hidden states — see compute_all_layers.
        ggml_cgraph* build_graph(const sd::Tensor<int32_t>& input_ids,
                                 const std::vector<ggml_tensor**>& hidden_out_slots,
                                 bool want_final = true) {
            auto gf = ggml_new_graph_custom(compute_ctx, GEMMA3_GRAPH_SIZE, false);
            auto ids_t = make_input(input_ids);
            int64_t L  = ids_t->ne[0];

            // Lazily rebuild rope / mask to match L.
            if (rope_global.L != L) {
                rope_global  = compute_gemma3_rope(L, params.head_dim, params.rope_theta_global, params.rope_scaling_factor);
                rope_local   = compute_gemma3_rope(L, params.head_dim, params.rope_theta_local,  /*scaling=*/1.0f);
                sliding_mask = build_causal_mask(L, params.sliding_window);
                full_mask    = build_causal_mask(L, (int)L);
            }

            auto rope_cos_g = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, params.head_dim, L);
            auto rope_sin_g = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, params.head_dim, L);
            auto rope_cos_l = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, params.head_dim, L);
            auto rope_sin_l = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, params.head_dim, L);
            auto mask_s     = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, L, L);
            auto mask_f     = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, L, L);
            set_backend_tensor_data(rope_cos_g, rope_global.cos.data());
            set_backend_tensor_data(rope_sin_g, rope_global.sin.data());
            set_backend_tensor_data(rope_cos_l, rope_local.cos.data());
            set_backend_tensor_data(rope_sin_l, rope_local.sin.data());
            set_backend_tensor_data(mask_s,     sliding_mask.data());
            set_backend_tensor_data(mask_f,     full_mask.data());

            auto rctx = get_context();
            std::vector<ggml_tensor*> hidden_all;
            auto out = model.forward_with_hidden_states(&rctx, ids_t,
                                                         rope_cos_g, rope_sin_g,
                                                         rope_cos_l, rope_sin_l,
                                                         mask_s, mask_f, hidden_all);
            // Publish the hidden states the caller asked for.
            GGML_ASSERT(hidden_out_slots.size() <= hidden_all.size());
            for (size_t i = 0; i < hidden_out_slots.size(); ++i) {
                if (hidden_out_slots[i]) *hidden_out_slots[i] = hidden_all[i];
            }
            // Expand all requested hidden states first so graph scheduling
            // keeps them reachable, then `out` last so it remains final.
            for (auto* h : hidden_all) ggml_build_forward_expand(gf, h);
            if (want_final) ggml_build_forward_expand(gf, out);
            return gf;
        }

        // Compute and return ONE specific hidden state by index.
        // layer_idx=0 → post-embed; 1..num_layers-1 → post-block i-1;
        // num_layers → post-final-norm (full model output).
        //
        // Implementation note: we inline the forward pass here and STOP
        // when we reach the target layer, so the graph's last node is
        // exactly the tensor we want. This bypasses the gallocr buffer-
        // reuse surprise that makes hidden_out entries unreadable after
        // later layers overwrite them.
        sd::Tensor<float> compute_layer_hidden(int n_threads,
                                               const sd::Tensor<int32_t>& input_ids,
                                               int layer_idx) {
            auto get_graph = [&]() -> ggml_cgraph* {
                auto* gf  = ggml_new_graph_custom(compute_ctx, GEMMA3_GRAPH_SIZE, false);
                auto ids_t = make_input(input_ids);
                int64_t L  = ids_t->ne[0];
                if (rope_global.L != L) {
                    rope_global  = compute_gemma3_rope(L, params.head_dim,
                                                       params.rope_theta_global,
                                                       params.rope_scaling_factor);
                    rope_local   = compute_gemma3_rope(L, params.head_dim,
                                                       params.rope_theta_local, 1.0f);
                    sliding_mask = build_causal_mask(L, params.sliding_window);
                    full_mask    = build_causal_mask(L, (int)L);
                }
                auto rctx = get_context();
                // Conditionally create the input tensors we'll actually
                // use. `set_backend_tensor_data` is only called for tensors
                // we DEFINITELY put in the graph — otherwise compute<>
                // tries to upload data to unallocated tensors and asserts.
                //
                // For layer_idx=N (num_layers or -1), all layers run, so we
                // need both global and local RoPE + mask. For a truncated
                // forward we compute which RoPE families are required.
                int64_t max_layers = (layer_idx < 0) ? params.num_layers
                                                      : (int64_t)layer_idx;
                bool need_global = false;
                bool need_local  = false;
                for (int64_t i = 0; i < max_layers; ++i) {
                    bool is_global = ((i + 1) % params.sliding_window_pattern) == 0;
                    if (is_global) need_global = true;
                    else           need_local  = true;
                }

                ggml_tensor* rope_cos_g = nullptr;
                ggml_tensor* rope_sin_g = nullptr;
                ggml_tensor* rope_cos_l = nullptr;
                ggml_tensor* rope_sin_l = nullptr;
                ggml_tensor* mask_s     = nullptr;
                ggml_tensor* mask_f     = nullptr;
                if (need_global) {
                    rope_cos_g = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, params.head_dim, L);
                    rope_sin_g = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, params.head_dim, L);
                    mask_f     = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, L, L);
                    set_backend_tensor_data(rope_cos_g, rope_global.cos.data());
                    set_backend_tensor_data(rope_sin_g, rope_global.sin.data());
                    set_backend_tensor_data(mask_f,     full_mask.data());
                }
                if (need_local) {
                    rope_cos_l = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, params.head_dim, L);
                    rope_sin_l = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, params.head_dim, L);
                    mask_s     = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, L, L);
                    set_backend_tensor_data(rope_cos_l, rope_local.cos.data());
                    set_backend_tensor_data(rope_sin_l, rope_local.sin.data());
                    set_backend_tensor_data(mask_s,     sliding_mask.data());
                }

                std::vector<ggml_tensor*> hidden_all;
                model.forward_with_hidden_states(&rctx, ids_t,
                                                 rope_cos_g, rope_sin_g,
                                                 rope_cos_l, rope_sin_l,
                                                 mask_s, mask_f,
                                                 hidden_all, max_layers);
                ggml_tensor* pick = hidden_all.back();
                auto pick_out = ggml_cont(compute_ctx, pick);
                ggml_build_forward_expand(gf, pick_out);
                return gf;
            };
            auto result = GGMLRunner::compute<float>(get_graph, n_threads, false);
            if (!result.has_value()) return {};
            return std::move(*result);
        }

        // Back-compat shim: previous callers asked for the "concatenated"
        // hidden; for now just return the last hidden state (post-final-
        // norm) so they still compile. Phase 5 will replace this with a
        // real concat over all layers.
        sd::Tensor<float> compute_concatenated_hiddens(int n_threads,
                                                       const sd::Tensor<int32_t>& input_ids) {
            return compute_layer_hidden(n_threads, input_ids, (int)params.num_layers);
        }
    };

}  // namespace GEMMA3

#endif  // __GEMMA3_HPP__
