#ifndef __LTX_CONNECTOR_HPP__
#define __LTX_CONNECTOR_HPP__

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "ggml_extend.hpp"
#include "ltx.hpp"
#include "ltx_rope.hpp"
#include "model.h"

// 1D position generator for the connector's RoPE (n_pos_dims=1, max_pos=[1],
// positions[t] = t). Lives here so it sits next to its only caller, but stays
// in the LTXRope namespace.
namespace LTXRope {
    __STATIC_INLINE__ std::vector<std::vector<float>> gen_1d_positions(int T) {
        std::vector<std::vector<float>> pos(1, std::vector<float>(T, 0.f));
        for (int t = 0; t < T; ++t) pos[0][t] = static_cast<float>(t);
        return pos;
    }
}  // namespace LTXRope

// LTX-2 text connector (Phase 9.1, V1 / 19B).
//
// Python reference:
//   ltx_core/text_encoders/gemma/feature_extractor.py   (FeatureExtractorV1)
//   ltx_core/text_encoders/gemma/embeddings_connector.py (Embeddings1DConnector)
//   ltx_core/model/transformer/text_projection.py        (PixArtAlphaTextProjection)
//
// Pipeline (Gemma 49-layer stack → DiT cross-attention context):
//   stacked[B, T, D, L]  → feature_extractor_normalize() (CPU, per-(B,L) masked
//       mean/range → normed[B, T, D*L])
//   normed[B, T, D*L]    → FeatureExtractorV1::forward (aggregate_embed Linear)
//                         → video_features[B, T, inner_dim]
//   video_features       → Embeddings1DConnector::forward (2× BasicTransformerBlock1D
//                         + final rms_norm) → [B, T, inner_dim]
//   connector_out        → PixArtAlphaTextProjection::forward (linear, gelu_tanh,
//                         linear) → [B, T, caption_out_dim]  (= DiT inner_dim)

namespace LTXConnector {

    // Compute FeatureExtractorV1's _norm_and_concat_padded_batch on the CPU.
    // Python reference: _norm_and_concat_padded_batch in feature_extractor.py.
    //
    // Input:
    //   stacked:     [B*T*D*L] contiguous, logical shape [B, T, D, L]
    //   seq_lengths: [B] — valid (non-pad) token count per batch
    //   padding_side: "left" or "right"
    // Output:
    //   normed:      [B*T*(D*L)] contiguous, logical shape [B, T, D*L]
    //
    // Padded positions (outside [0, seq_len) for "right", outside [T - seq_len, T) for "left")
    // are zero'd after the normalization.
    __STATIC_INLINE__ void feature_extractor_normalize(const float* stacked,
                                                       const int* seq_lengths,
                                                       float* normed,
                                                       int B, int T, int D, int L,
                                                       const std::string& padding_side = "left",
                                                       float eps                       = 1e-6f) {
        const float FINF = std::numeric_limits<float>::infinity();
        const float NINF = -FINF;
        const bool is_left = (padding_side == "left");

        for (int b = 0; b < B; ++b) {
            int seq_len = seq_lengths[b];
            int valid_start = is_left ? (T - seq_len) : 0;
            int valid_end   = is_left ? T : seq_len;

            for (int l = 0; l < L; ++l) {
                // Compute per-(b,l) masked mean, min, max over (t, d) where mask == 1.
                double sum = 0.0;
                float vmin = FINF;
                float vmax = NINF;
                for (int t = valid_start; t < valid_end; ++t) {
                    for (int d = 0; d < D; ++d) {
                        // Python layout: encoded[b, t, d, l]
                        // Flat index with ne [L, D, T, B] order would be ((b*T + t)*D + d)*L + l.
                        int64_t idx = ((static_cast<int64_t>(b) * T + t) * D + d) * L + l;
                        float v = stacked[idx];
                        sum += v;
                        if (v < vmin) vmin = v;
                        if (v > vmax) vmax = v;
                    }
                }
                double denom = static_cast<double>(seq_len) * D;
                float mean   = static_cast<float>(sum / (denom + eps));
                float range  = vmax - vmin;
                float inv    = 8.0f / (range + eps);

                // Apply normalization over all T positions; zero out padded ones.
                for (int t = 0; t < T; ++t) {
                    bool in_valid = (t >= valid_start && t < valid_end);
                    for (int d = 0; d < D; ++d) {
                        int64_t src_idx = ((static_cast<int64_t>(b) * T + t) * D + d) * L + l;
                        // normed layout: [B, T, D*L] with flat index (b*T + t)*(D*L) + (d*L + l).
                        int64_t dst_idx = (static_cast<int64_t>(b) * T + t) * (D * L) + (d * L + l);
                        if (in_valid) {
                            normed[dst_idx] = (stacked[src_idx] - mean) * inv;
                        } else {
                            normed[dst_idx] = 0.0f;
                        }
                    }
                }
            }
        }
    }

    // Per-token RMSNorm used by FeatureExtractorV2 (22B / V2 text path). Mirrors
    // norm_and_concat_per_token_rms in Python feature_extractor.py.
    //
    // Input layout (ggml ne order, matches llm->compute_all_hidden_states):
    //   stacked[l + L*(d + D*(t + T*b))] — logical shape [B, T, D, L]
    //
    // Output layout (ggml ne order):
    //   normed[k + (D*L)*(t + T*b)]        — logical shape [B, T, D*L] with k = d*L + l
    //
    // Per-(B, T, L) variance is computed over D; every entry is scaled by the
    // corresponding rsqrt(var + eps). Padded positions (per `attention_mask`) get
    // zeroed out post-reshape, matching Python's `torch.where(mask_3d, normed, 0)`.
    //
    // The result is NOT yet rescaled by sqrt(target/source) — that's applied as a
    // `ggml_scale` in the graph immediately before the aggregate_embed Linear so
    // video and audio branches (with different target dims) can share this buffer.
    __STATIC_INLINE__ void feature_extractor_normalize_v2(const float* stacked,
                                                          const int* seq_lengths,
                                                          float* normed,
                                                          int B, int T, int D, int L,
                                                          const std::string& padding_side = "left",
                                                          float eps                       = 1e-6f) {
        const bool is_left = (padding_side == "left");
        for (int b = 0; b < B; ++b) {
            int seq_len     = seq_lengths[b];
            int valid_start = is_left ? (T - seq_len) : 0;
            int valid_end   = is_left ? T : seq_len;

            for (int t = 0; t < T; ++t) {
                bool in_valid = (t >= valid_start && t < valid_end);
                // Per-layer rsqrt factor for the (b, t, *, l) row.
                for (int l = 0; l < L; ++l) {
                    double sum_sq = 0.0;
                    for (int d = 0; d < D; ++d) {
                        int64_t idx = ((static_cast<int64_t>(b) * T + t) * D + d) * L + l;
                        double v    = stacked[idx];
                        sum_sq += v * v;
                    }
                    double variance = sum_sq / static_cast<double>(D);
                    float rsq       = static_cast<float>(1.0 / std::sqrt(variance + eps));

                    for (int d = 0; d < D; ++d) {
                        int64_t src_idx = ((static_cast<int64_t>(b) * T + t) * D + d) * L + l;
                        int64_t dst_idx = (static_cast<int64_t>(b) * T + t) * (D * L) + (d * L + l);
                        if (in_valid) {
                            normed[dst_idx] = stacked[src_idx] * rsq;
                        } else {
                            normed[dst_idx] = 0.0f;
                        }
                    }
                }
            }
        }
    }

    // FeatureExtractorV1 block — just wraps the aggregate_embed Linear
    // (feature_extractor.aggregate_embed.weight).
    //
    // The CPU-side normalization lives in feature_extractor_normalize(); this block
    // expects an already-normalized [B, T, D*L] tensor as input.
    struct FeatureExtractorV1 : public GGMLBlock {
    protected:
        int64_t flat_dim;
        int64_t inner_dim;

    public:
        FeatureExtractorV1() = default;
        FeatureExtractorV1(int64_t flat_dim, int64_t inner_dim)
            : flat_dim(flat_dim), inner_dim(inner_dim) {
            // Python: aggregate_embed = Linear(flat_dim, inner_dim, bias=False).
            // flat_dim is huge (49 × 3840 = 188160 for 22B); F16 matmul accumulator
            // can't hold that many sums at full precision. Match V2's force_prec_f32.
            blocks["aggregate_embed"] = std::make_shared<Linear>(flat_dim, inner_dim, /*bias=*/false,
                                                                 /*force_f32=*/false, /*force_prec_f32=*/true);
        }

        // x: ne [flat_dim, T, B] (already normalized via feature_extractor_normalize).
        // returns:  ne [inner_dim, T, B].
        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto agg = std::dynamic_pointer_cast<Linear>(blocks["aggregate_embed"]);
            return agg->forward(ctx, x);
        }
    };

    // FeatureExtractorV2 block — V2 / 22B text path. Two parallel Linears
    // (video + optional audio) WITH bias on a per-token RMS-normalized input.
    // Python: ltx_core/text_encoders/gemma/feature_extractor.py::FeatureExtractorV2.
    //
    // The CPU-side normalization lives in feature_extractor_normalize_v2(); this
    // block applies the in-graph rescale factor sqrt(target/source_dim) and the
    // video_aggregate_embed Linear. Audio path is declared optional — if audio
    // weights are absent the block skips it and is video-only.
    struct FeatureExtractorV2 : public GGMLBlock {
    protected:
        int64_t flat_dim;       // D * L (Gemma hidden × num_layers)
        int64_t source_dim;     // Gemma hidden size (D)
        int64_t video_out_dim;  // DiT inner_dim
        int64_t audio_out_dim;  // optional; 0 when no audio aggregate_embed
        float   video_scale;    // sqrt(video_out_dim / source_dim)
        float   audio_scale;    // sqrt(audio_out_dim / source_dim)

    public:
        FeatureExtractorV2() = default;
        FeatureExtractorV2(int64_t flat_dim, int64_t source_dim,
                           int64_t video_out_dim,
                           int64_t audio_out_dim = 0)
            : flat_dim(flat_dim), source_dim(source_dim),
              video_out_dim(video_out_dim), audio_out_dim(audio_out_dim) {
            video_scale = std::sqrt(static_cast<float>(video_out_dim) / static_cast<float>(source_dim));
            audio_scale = audio_out_dim > 0
                              ? std::sqrt(static_cast<float>(audio_out_dim) / static_cast<float>(source_dim))
                              : 0.f;
            // Force FP32 matmul precision: flat_dim=188160 sums easily exceed F16's
            // mantissa precision and produce direction-rotating drift on later
            // tokens. Comfy runs this in BF16 which has full FP32 range; we need
            // explicit F32 precision when running on CUDA/F16 backends.
            blocks["video_aggregate_embed"] = std::make_shared<Linear>(flat_dim, video_out_dim, /*bias=*/true,
                                                                       /*force_f32=*/false, /*force_prec_f32=*/true);
            if (audio_out_dim > 0) {
                blocks["audio_aggregate_embed"] = std::make_shared<Linear>(flat_dim, audio_out_dim, /*bias=*/true,
                                                                            /*force_f32=*/false, /*force_prec_f32=*/true);
            }
        }

        bool has_audio() const { return audio_out_dim > 0; }
        int64_t get_video_out_dim() const { return video_out_dim; }
        int64_t get_audio_out_dim() const { return audio_out_dim; }

        // x: ne [flat_dim, T, B] (already per-token RMS-normalized via feature_extractor_normalize_v2).
        // Returns video_features ne [video_out_dim, T, B]. Audio branch unused for video-only smoke tests.
        ggml_tensor* forward_video(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto agg     = std::dynamic_pointer_cast<Linear>(blocks["video_aggregate_embed"]);
            auto scaled  = ggml_scale(ctx->ggml_ctx, x, video_scale);
            return agg->forward(ctx, scaled);
        }

        ggml_tensor* forward_audio(GGMLRunnerContext* ctx, ggml_tensor* x) {
            GGML_ASSERT(has_audio() && "FeatureExtractorV2: audio_aggregate_embed not allocated");
            auto agg     = std::dynamic_pointer_cast<Linear>(blocks["audio_aggregate_embed"]);
            auto scaled  = ggml_scale(ctx->ggml_ctx, x, audio_scale);
            return agg->forward(ctx, scaled);
        }
    };

    // A single 1D transformer block in the connector.
    // Python: _BasicTransformerBlock1D in embeddings_connector.py.
    //
    // Self-attention only (no cross-attention, no AdaLN). Parameter-free rms_norm
    // before attention and before the feed-forward.
    struct BasicTransformerBlock1D : public GGMLBlock {
    protected:
        int64_t dim;
        int num_heads;
        int head_dim;
        bool apply_gated_attention;
        float norm_eps;

    public:
        BasicTransformerBlock1D() = default;
        BasicTransformerBlock1D(int64_t dim, int num_heads, int head_dim,
                                bool apply_gated_attention = false,
                                float norm_eps             = 1e-6f)
            : dim(dim), num_heads(num_heads), head_dim(head_dim),
              apply_gated_attention(apply_gated_attention), norm_eps(norm_eps) {
            // Self-attention: context_dim = query_dim = dim.  The connector's 1D RoPE
            // uses INTERLEAVED layout (Python embeddings_connector.py calls
            // precompute_freqs_cis with default rope_type=INTERLEAVED); only the DiT
            // was switched to SPLIT in LTX-2.3.
            blocks["attn1"] = std::make_shared<LTX::LTXAttention>(dim, dim, num_heads, head_dim,
                                                                   apply_gated_attention, norm_eps,
                                                                   LTX::RopeType::INTERLEAVED);
            blocks["ff"]    = std::make_shared<LTX::FeedForward>(dim, dim);
        }

        // hidden_states: ne [dim, T, B]
        // pe:            ne [dim, T, 2] packed cos/sin (or nullptr)
        // mask:          additive attention mask (or nullptr)
        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* hidden_states,
                             ggml_tensor* pe,
                             ggml_tensor* mask = nullptr) {
            auto attn1 = std::dynamic_pointer_cast<LTX::LTXAttention>(blocks["attn1"]);
            auto ff    = std::dynamic_pointer_cast<LTX::FeedForward>(blocks["ff"]);

            // Pre-norm + self-attention + residual.
            auto norm1 = LTX::parameterless_rms_norm(ctx->ggml_ctx, hidden_states, norm_eps);
            auto a_out = attn1->forward(ctx, norm1, /*context=*/nullptr, pe, mask);
            hidden_states = ggml_add(ctx->ggml_ctx, hidden_states, a_out);

            // Pre-norm + feed-forward + residual.
            auto norm2 = LTX::parameterless_rms_norm(ctx->ggml_ctx, hidden_states, norm_eps);
            auto f_out = ff->forward(ctx, norm2);
            hidden_states = ggml_add(ctx->ggml_ctx, hidden_states, f_out);

            return hidden_states;
        }
    };

    // Embeddings1DConnector: 2-layer 1D transformer with learnable registers +
    // final parameter-free rms_norm. 1D RoPE with max_pos=[1], theta=10000.0.
    struct Embeddings1DConnector : public GGMLBlock {
    protected:
        int num_heads;
        int head_dim;
        int64_t inner_dim;
        int num_layers;
        int num_registers;  // 0 disables the learnable-registers path.
        float theta;
        std::vector<int> max_pos;
        bool apply_gated_attention;
        float norm_eps;

        void init_params(ggml_context* ctx, const String2TensorStorage&, const std::string prefix = "") override {
            if (num_registers > 0) {
                // Python: learnable_registers = Parameter(rand(num_registers, inner_dim) * 2 - 1)
                // ggml ne layout: innermost = inner_dim, so [inner_dim, num_registers].
                params["learnable_registers"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, inner_dim, num_registers);
            }
        }

    public:
        Embeddings1DConnector() = default;
        Embeddings1DConnector(int num_heads, int head_dim, int num_layers,
                              int num_registers               = 128,
                              float theta                     = 10000.0f,
                              const std::vector<int>& max_pos = {1},
                              bool apply_gated_attention      = false,
                              float norm_eps                  = 1e-6f)
            : num_heads(num_heads), head_dim(head_dim),
              inner_dim(static_cast<int64_t>(num_heads) * head_dim),
              num_layers(num_layers), num_registers(num_registers),
              theta(theta), max_pos(max_pos),
              apply_gated_attention(apply_gated_attention), norm_eps(norm_eps) {
            for (int i = 0; i < num_layers; ++i) {
                blocks["transformer_1d_blocks." + std::to_string(i)] =
                    std::make_shared<BasicTransformerBlock1D>(inner_dim, num_heads, head_dim,
                                                              apply_gated_attention, norm_eps);
            }
        }

        int64_t get_inner_dim() const { return inner_dim; }
        int get_num_registers() const { return num_registers; }
        int get_num_layers() const { return num_layers; }

        ggml_tensor* get_learnable_registers() {
            auto it = params.find("learnable_registers");
            return it == params.end() ? nullptr : it->second;
        }

        std::shared_ptr<BasicTransformerBlock1D> get_block(int i) {
            return std::dynamic_pointer_cast<BasicTransformerBlock1D>(
                blocks["transformer_1d_blocks." + std::to_string(i)]);
        }

        // hidden_states: ne [inner_dim, T, B]
        // pe:            ne [inner_dim, T, 2] packed cos/sin
        // mask:          additive attention mask (or nullptr)
        //
        // NOTE: this currently skips `_replace_padded_with_learnable_registers` —
        // callers must guarantee the input is already register-substituted (or no
        // padding is present). Handling the register replacement in ggml requires
        // boolean indexing/scatter semantics that we defer.
        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* hidden_states,
                             ggml_tensor* pe,
                             ggml_tensor* mask = nullptr) {
            for (int i = 0; i < num_layers; ++i) {
                auto block = std::dynamic_pointer_cast<BasicTransformerBlock1D>(
                    blocks["transformer_1d_blocks." + std::to_string(i)]);
                hidden_states = block->forward(ctx, hidden_states, pe, mask);
            }
            hidden_states = LTX::parameterless_rms_norm(ctx->ggml_ctx, hidden_states, norm_eps);
            return hidden_states;
        }
    };

    // Which feature-extractor flavor the runner uses. V1 (19B) has a single
    // Linear(flat_dim → inner_dim, bias=False) named `aggregate_embed.weight`, with
    // CPU pre-norm via _norm_and_concat_padded_batch. V2 (22B) has two parallel
    // Linears with bias (`video_aggregate_embed`, `audio_aggregate_embed`) on a
    // per-token RMS-normalized input; we currently wire only the video path.
    enum class FeatureExtractorVersion { V1, V2 };

    // Runner that bundles feature_extractor + Embeddings1DConnector (and optionally
    // caption_projection for end-to-end parity testing). Used both by the parity
    // test (default ctor args match dump_connector.py) and by LTX2GemmaConditioner
    // (which passes real-checkpoint prefixes and sets include_caption_projection=false
    // because the DiT owns caption_projection).
    //
    // Input is the already-normalized [B, T, flat_dim] tensor (see
    // feature_extractor_normalize[_v2] for the CPU pre-processing).
    struct LTX2ConnectorRunner : public GGMLRunner {
        int64_t flat_dim;
        int64_t connector_inner_dim;
        int num_heads;
        int head_dim;
        int num_layers;
        int num_registers;
        int64_t caption_channels;
        int64_t caption_hidden;
        int64_t caption_out;
        float theta;
        std::vector<int> max_pos;
        bool include_caption_projection;
        FeatureExtractorVersion fe_version;
        int64_t source_dim;  // V2 only: Gemma hidden_size used for rescale

        std::string feat_ext_prefix;
        std::string connector_prefix;
        std::string caption_proj_prefix;

        FeatureExtractorV1 feature_extractor_v1;
        FeatureExtractorV2 feature_extractor_v2;
        Embeddings1DConnector connector;
        LTX::PixArtAlphaTextProjection caption_projection;

        std::vector<float> pe_vec;

        // probe_stage selects the returned tensor. Stages <1 and >2 are shared
        // between V1 and V2; 1 and 2 are legacy V1 parity probes (after block 0/1)
        // and only work when num_layers >= 2. For V2 (production use), stage 3
        // (final rms_norm) is what the conditioner calls.
        //   0 = after feature_extractor (+ graph-side rescale for V2)
        //   1 = after connector block 0
        //   2 = after connector block 1
        //   3 = after all blocks + final rms_norm (connector output)
        //   4 = after caption_projection (requires include_caption_projection)
        int probe_stage = 3;

        // Target sequence length fed into the 1D connector. Python's
        // LTXVGemmaTokenizer pads to max_length=1024 so the connector always sees
        // 1024 tokens with learnable_registers tiled max_length/num_registers times.
        // A value of 0 falls back to num_registers (the old, compact behaviour used
        // by the parity dumper). Real inference MUST set this to match the Python
        // tokenizer max_length (1024) — see LTX-2 ti2vid pipelines.
        int target_seq_len = 0;
        void set_target_seq_len(int len) { target_seq_len = len; }

        LTX2ConnectorRunner(ggml_backend_t backend,
                            bool offload_params_to_cpu,
                            int64_t flat_dim,
                            int num_heads,
                            int head_dim,
                            int num_layers,
                            int num_registers,
                            int64_t caption_channels                = 0,
                            int64_t caption_hidden                  = 0,
                            int64_t caption_out                     = 0,
                            float theta                             = 10000.0f,
                            const std::vector<int>& max_pos         = {1},
                            const String2TensorStorage& tsm         = {},
                            bool include_caption_projection         = true,
                            const std::string& feat_ext_prefix      = "feature_extractor",
                            const std::string& connector_prefix     = "connector",
                            const std::string& caption_proj_prefix  = "caption_projection",
                            FeatureExtractorVersion fe_version      = FeatureExtractorVersion::V1,
                            int64_t source_dim                      = 0,
                            bool apply_gated_attention              = false)
            : GGMLRunner(backend, offload_params_to_cpu),
              flat_dim(flat_dim),
              connector_inner_dim(static_cast<int64_t>(num_heads) * head_dim),
              num_heads(num_heads), head_dim(head_dim), num_layers(num_layers),
              num_registers(num_registers),
              caption_channels(caption_channels),
              caption_hidden(caption_hidden),
              caption_out(caption_out),
              theta(theta), max_pos(max_pos),
              include_caption_projection(include_caption_projection),
              fe_version(fe_version),
              source_dim(source_dim),
              feat_ext_prefix(feat_ext_prefix),
              connector_prefix(connector_prefix),
              caption_proj_prefix(caption_proj_prefix) {
            if (fe_version == FeatureExtractorVersion::V2) {
                GGML_ASSERT(source_dim > 0 && "FeatureExtractorV2 needs Gemma source_dim for the sqrt-rescale");
                feature_extractor_v2 = FeatureExtractorV2(flat_dim, source_dim, connector_inner_dim);
                feature_extractor_v2.init(params_ctx, tsm, feat_ext_prefix);
            } else {
                feature_extractor_v1 = FeatureExtractorV1(flat_dim, connector_inner_dim);
                feature_extractor_v1.init(params_ctx, tsm, feat_ext_prefix);
            }
            connector = Embeddings1DConnector(num_heads, head_dim, num_layers,
                                              num_registers, theta, max_pos,
                                              apply_gated_attention);
            connector.init(params_ctx, tsm, connector_prefix);
            if (include_caption_projection) {
                caption_projection = LTX::PixArtAlphaTextProjection(caption_channels, caption_hidden, caption_out);
                caption_projection.init(params_ctx, tsm, caption_proj_prefix);
            }
        }

        std::string get_desc() override { return "ltx2-connector"; }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors,
                               const std::string /*unused*/ = "") {
            if (fe_version == FeatureExtractorVersion::V2) {
                feature_extractor_v2.get_param_tensors(tensors, feat_ext_prefix);
            } else {
                feature_extractor_v1.get_param_tensors(tensors, feat_ext_prefix);
            }
            connector.get_param_tensors(tensors, connector_prefix);
            if (include_caption_projection) {
                caption_projection.get_param_tensors(tensors, caption_proj_prefix);
            }
        }

        // Build the full graph. probe_stage selects the final returned tensor:
        //   0: after feature_extractor (shape [connector_inner_dim, T, B])
        //   1: after connector block 0  (V1 parity probe, legacy)
        //   2: after connector block 1  (V1 parity probe, legacy)
        //   3: after all connector blocks + final rms_norm
        //   4: after caption_projection (needs include_caption_projection=true)
        ggml_cgraph* build_graph(const sd::Tensor<float>& normed_in) {
            ggml_cgraph* gf = new_graph_custom(LTX::LTX_GRAPH_SIZE);

            ggml_tensor* x = make_input(normed_in);  // ne [flat_dim, T, B]
            int64_t T = x->ne[1];

            auto runner_ctx = get_context();

            // Step 1: feature_extractor → [inner_dim, T, B].
            ggml_tensor* feat = nullptr;
            if (fe_version == FeatureExtractorVersion::V2) {
                feat = feature_extractor_v2.forward_video(&runner_ctx, x);
            } else {
                feat = feature_extractor_v1.forward(&runner_ctx, x);
            }

            // Step 1.5: Pad to the target length by filling the tail with
            // learnable_registers (tiled when target > num_registers).
            //
            // Python reference: `_replace_padded_with_learnable_registers` in
            // ltx_core/text_encoders/gemma/embeddings_connector.py. It:
            //   1. tiles learnable_registers by (seq_len / num_registers) so the tiled
            //      buffer covers the whole sequence (seq_len == tokenizer max_length),
            //   2. moves real tokens to [0, T_real),
            //   3. fills [T_real, seq_len) with tiled_registers[T_real, seq_len).
            //
            // The caller (conditioner.hpp) already does step 2 on CPU and passes feat
            // as [inner_dim, T_real, B]. We pick the target length in this order of
            // preference: (a) explicit target_seq_len (set by the conditioner to
            // Gemma's max_length), (b) num_registers (legacy/parity default).
            //
            // Tiling is implemented with a ggml_repeat into a [inner_dim, target, B]
            // destination — cheap on GPU and matches torch.tile semantics for the
            // innermost tiling axis.
            const int num_registers = connector.get_num_registers();
            int64_t target_len =
                target_seq_len > 0 ? static_cast<int64_t>(target_seq_len)
                                   : static_cast<int64_t>(num_registers);
            if (num_registers > 0 && target_len > 0 && T < target_len) {
                GGML_ASSERT(target_len % num_registers == 0 &&
                            "target_seq_len must be a multiple of num_registers "
                            "(Embeddings1DConnector tiles learnable_registers).");
                auto regs = connector.get_learnable_registers();  // [inner_dim, num_registers]
                GGML_ASSERT(regs != nullptr && "learnable_registers not initialized");

                // Build the tiled registers tensor [inner_dim, target_len] by
                // repeating learnable_registers along axis 1.
                ggml_tensor* tiled = regs;
                if (target_len > num_registers) {
                    auto repeat_tgt = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32,
                                                         connector_inner_dim, target_len);
                    tiled = ggml_repeat(compute_ctx, regs, repeat_tgt);
                }

                // Slice rows [T : target_len] along axis 1 to get the padding tail.
                auto regs_slice = ggml_ext_slice(compute_ctx, tiled, 1,
                                                 static_cast<int>(T),
                                                 static_cast<int>(target_len));  // [inner_dim, target-T]
                regs_slice = ggml_reshape_3d(compute_ctx, ggml_cont(compute_ctx, regs_slice),
                                             connector_inner_dim,
                                             target_len - T,
                                             1);
                feat = ggml_concat(compute_ctx, feat, regs_slice, 1);  // [inner_dim, target, B]
                T = target_len;
            }

            // Build only the subgraph up to the selected probe stage. The final
            // named result is the LAST node added (GGMLRunner::get_compute_graph
            // picks `ggml_graph_node(gf, -1)`).
            ggml_tensor* out = feat;
            if (probe_stage >= 1) {
                // Precompute 1D RoPE for connector.
                auto positions = LTXRope::gen_1d_positions(static_cast<int>(T));
                pe_vec         = LTXRope::precompute_freqs_cis_interleaved(positions,
                                                                           static_cast<int>(connector_inner_dim),
                                                                           theta, max_pos);
                auto pe = ggml_new_tensor_3d(compute_ctx, GGML_TYPE_F32, connector_inner_dim, T, 2);
                set_backend_tensor_data(pe, pe_vec.data());

                // Stages 1, 2: legacy V1 parity probes — stop after block 0/1.
                // Stages 3+: production path — run all blocks and the final rms_norm.
                if (probe_stage == 1 || probe_stage == 2) {
                    int blocks_to_run = probe_stage;  // 1 → block 0 only; 2 → blocks 0 and 1
                    for (int i = 0; i < blocks_to_run && i < num_layers; ++i) {
                        out = connector.get_block(i)->forward(&runner_ctx, out, pe, nullptr);
                    }
                } else {
                    for (int i = 0; i < num_layers; ++i) {
                        out = connector.get_block(i)->forward(&runner_ctx, out, pe, nullptr);
                    }
                    out = LTX::parameterless_rms_norm(compute_ctx, out, 1e-6f);
                }
            }
            if (probe_stage >= 4 && include_caption_projection) {
                out = caption_projection.forward(&runner_ctx, out);
            }

            ggml_build_forward_expand(gf, out);
            return gf;
        }

        sd::Tensor<float> compute(int n_threads, const sd::Tensor<float>& normed_in, int stage = 4) {
            probe_stage    = stage;
            auto get_graph = [&]() -> ggml_cgraph* { return build_graph(normed_in); };
            return take_or_empty(GGMLRunner::compute<float>(get_graph, n_threads, true));
        }
    };

}  // namespace LTXConnector

#endif  // __LTX_CONNECTOR_HPP__
