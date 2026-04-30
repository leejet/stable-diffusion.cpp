#ifndef __LTX_HPP__
#define __LTX_HPP__

#include <algorithm>
#include <memory>
#include <tuple>
#include <vector>

#include "ggml_extend.hpp"
#include "ltx_rope.hpp"
#include "model.h"

// LTX-2 video DiT.
// Reference: /devel/tools/diffusion/LTX-2/packages/ltx-core/src/ltx_core/model/transformer/
//
// Scope (first landing): text-conditioned video-only (LTXModelType.VideoOnly), rope_type=INTERLEAVED,
// cross_attention_adaln=false, apply_gated_attention=false. Audio pathway and AV cross-attention are
// deferred (stubbed out) — the weights are just not instantiated.

namespace LTX {
    // 32768 was enough for the 2-layer parity-test DiT. The 22B V2 has 48 layers
    // + cross_attention_adaln + prompt_adaln_single, roughly 2-3× the op count
    // per block vs. V1. Bump generously so graph construction never fails the
    // `cgraph->n_nodes < cgraph->size` assert in ggml's append path.
    constexpr int LTX_GRAPH_SIZE  = 131072;
    constexpr int TIME_PROJ_DIM   = 256;
    constexpr int ADALN_BASE      = 6;
    constexpr int ADALN_WITH_CA   = 9;

    // Python: ltx_core.model.transformer.rope.LTXRopeType.  Real LTX-2.3 config uses
    // SPLIT; earlier LTX variants (and our parity test's old default) were INTERLEAVED.
    enum class RopeType { INTERLEAVED, SPLIT };

    // Parameter-free RMSNorm helper.
    __STATIC_INLINE__ ggml_tensor* parameterless_rms_norm(ggml_context* ctx, ggml_tensor* x, float eps = 1e-6f) {
        return ggml_rms_norm(ctx, x, eps);
    }

    struct AdaLayerNormSingle : public GGMLBlock {
    protected:
        int embedding_dim;
        int embedding_coefficient;

    public:
        AdaLayerNormSingle() = default;
        AdaLayerNormSingle(int embedding_dim, int embedding_coefficient)
            : embedding_dim(embedding_dim), embedding_coefficient(embedding_coefficient) {
            // Python: self.emb = PixArtAlphaCombinedTimestepSizeEmbeddings(embedding_dim, size_emb_dim=embedding_dim // 3)
            //   -> time_proj: sinusoidal (no weights)
            //   -> timestep_embedder.linear_1: Linear(256, embedding_dim)
            //   -> timestep_embedder.linear_2: Linear(embedding_dim, embedding_dim)
            // Python: self.linear = Linear(embedding_dim, coefficient * embedding_dim)
            blocks["emb.timestep_embedder.linear_1"] = std::make_shared<Linear>(TIME_PROJ_DIM, embedding_dim, true);
            blocks["emb.timestep_embedder.linear_2"] = std::make_shared<Linear>(embedding_dim, embedding_dim, true);
            blocks["linear"]                         = std::make_shared<Linear>(embedding_dim, embedding_coefficient * embedding_dim, true);
        }

        // timestep: [B] — caller MUST pass the pre-scaled timestep (σ * timestep_scale_multiplier).
        // Python applies the scaling in TransformerArgsPreprocessor._prepare_timestep; we mirror that
        // boundary so the denoiser (sigma_to_t) is the single place that owns the 1000× factor.
        // Double-scaling (denoiser + AdaLN) would drive sinusoidal embedding args to σ·1e6, which is
        // numerical nonsense and was a real risk before this refactor.
        //
        // Returns {modulation, embedded_timestep}.
        // modulation ne: [embedding_dim, coefficient, B]
        // embedded_timestep ne: [embedding_dim, B]
        std::pair<ggml_tensor*, ggml_tensor*> forward(GGMLRunnerContext* ctx,
                                                      ggml_tensor* timestep) {
            auto l1   = std::dynamic_pointer_cast<Linear>(blocks["emb.timestep_embedder.linear_1"]);
            auto l2   = std::dynamic_pointer_cast<Linear>(blocks["emb.timestep_embedder.linear_2"]);
            auto proj = std::dynamic_pointer_cast<Linear>(blocks["linear"]);

            auto t_proj   = ggml_ext_timestep_embedding(ctx->ggml_ctx, timestep, TIME_PROJ_DIM, 10000, 1.0f);
            auto hidden   = l1->forward(ctx, t_proj);
            hidden        = ggml_silu_inplace(ctx->ggml_ctx, hidden);
            auto embedded = l2->forward(ctx, hidden);  // [embedding_dim, B]

            auto modulation = ggml_silu(ctx->ggml_ctx, embedded);
            modulation      = proj->forward(ctx, modulation);  // [coeff*embedding_dim, B]

            int64_t B   = modulation->ne[1];
            modulation  = ggml_reshape_3d(ctx->ggml_ctx, modulation, embedding_dim, embedding_coefficient, B);
            return {modulation, embedded};
        }
    };

    // GELUApprox block: Linear(dim_in → dim_out) + gelu(tanh approximation).
    // Python: GELUApprox uses torch.nn.functional.gelu(..., approximate="tanh") which matches ggml_gelu.
    struct GELUApprox : public GGMLBlock {
    public:
        GELUApprox() = default;
        GELUApprox(int64_t dim_in, int64_t dim_out) {
            blocks["proj"] = std::make_shared<Linear>(dim_in, dim_out, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto proj = std::dynamic_pointer_cast<Linear>(blocks["proj"]);
            x         = proj->forward(ctx, x);
            return ggml_ext_gelu(ctx->ggml_ctx, x, true);
        }
    };

    struct FeedForward : public GGMLBlock {
    public:
        FeedForward() = default;
        FeedForward(int64_t dim, int64_t dim_out, int mult = 4) {
            int64_t inner = dim * mult;
            // Python: self.net = Sequential(GELUApprox(dim, inner), Identity(), Linear(inner, dim_out))
            blocks["net.0"] = std::make_shared<GELUApprox>(dim, inner);
            blocks["net.2"] = std::make_shared<Linear>(inner, dim_out, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto gelu_proj = std::dynamic_pointer_cast<GELUApprox>(blocks["net.0"]);
            auto out_proj  = std::dynamic_pointer_cast<Linear>(blocks["net.2"]);
            x              = gelu_proj->forward(ctx, x);
            x              = out_proj->forward(ctx, x);
            return x;
        }
    };

    struct LTXAttention : public GGMLBlock {
    protected:
        int64_t query_dim;
        int64_t context_dim;
        int num_heads;
        int head_dim;
        int64_t inner_dim;
        float norm_eps;
        bool apply_gated_attention;
        RopeType rope_type;

    public:
        LTXAttention() = default;
        LTXAttention(int64_t query_dim, int64_t context_dim, int num_heads, int head_dim,
                     bool apply_gated_attention = false, float norm_eps = 1e-6f,
                     RopeType rope_type         = RopeType::SPLIT)
            : query_dim(query_dim), context_dim(context_dim), num_heads(num_heads),
              head_dim(head_dim), inner_dim(static_cast<int64_t>(num_heads) * head_dim),
              norm_eps(norm_eps), apply_gated_attention(apply_gated_attention),
              rope_type(rope_type) {
            blocks["to_q"]     = std::make_shared<Linear>(query_dim, inner_dim, true);
            blocks["to_k"]     = std::make_shared<Linear>(context_dim, inner_dim, true);
            blocks["to_v"]     = std::make_shared<Linear>(context_dim, inner_dim, true);
            blocks["q_norm"]   = std::make_shared<RMSNorm>(inner_dim, norm_eps);
            blocks["k_norm"]   = std::make_shared<RMSNorm>(inner_dim, norm_eps);
            blocks["to_out.0"] = std::make_shared<Linear>(inner_dim, query_dim, true);
            if (apply_gated_attention) {
                blocks["to_gate_logits"] = std::make_shared<Linear>(query_dim, num_heads, true);
            }
        }

        // x: [query_dim, L_q, B]
        // context: [context_dim, L_kv, B] (defaults to x for self-attn)
        // pe: optional packed cos/sin [inner_dim, L_q, 2] applied to Q
        // mask: optional additive attention mask
        // k_pe: optional separate cos/sin [inner_dim, L_kv, 2] applied to K. Null →
        //       K uses `pe` (same length as Q). Used for cross-modal attention where
        //       Q and K have different sequence lengths and per-modality positional
        //       embeddings (audio_to_video_attn / video_to_audio_attn).
        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* context,
                             ggml_tensor* pe,
                             ggml_tensor* mask = nullptr,
                             ggml_tensor* k_pe = nullptr) {
            if (context == nullptr) {
                context = x;
            }
            auto to_q   = std::dynamic_pointer_cast<Linear>(blocks["to_q"]);
            auto to_k   = std::dynamic_pointer_cast<Linear>(blocks["to_k"]);
            auto to_v   = std::dynamic_pointer_cast<Linear>(blocks["to_v"]);
            auto q_norm = std::dynamic_pointer_cast<RMSNorm>(blocks["q_norm"]);
            auto k_norm = std::dynamic_pointer_cast<RMSNorm>(blocks["k_norm"]);
            auto to_out = std::dynamic_pointer_cast<Linear>(blocks["to_out.0"]);

            auto q = to_q->forward(ctx, x);        // [inner_dim, L_q, B]
            auto k = to_k->forward(ctx, context);  // [inner_dim, L_kv, B]
            auto v = to_v->forward(ctx, context);  // [inner_dim, L_kv, B]

            q = q_norm->forward(ctx, q);
            k = k_norm->forward(ctx, k);

            if (pe != nullptr) {
                ggml_tensor* k_pe_eff = (k_pe != nullptr) ? k_pe : pe;
                if (rope_type == RopeType::SPLIT) {
                    auto q_cs = LTXRope::split_pe_split(ctx->ggml_ctx, pe);
                    q = LTXRope::apply_rotary_emb_split(ctx->ggml_ctx, q, q_cs.first, q_cs.second, num_heads);
                    auto k_cs = LTXRope::split_pe_split(ctx->ggml_ctx, k_pe_eff);
                    k = LTXRope::apply_rotary_emb_split(ctx->ggml_ctx, k, k_cs.first, k_cs.second, num_heads);
                } else {
                    auto q_cs = LTXRope::split_pe(ctx->ggml_ctx, pe);
                    q = LTXRope::apply_rotary_emb_interleaved(ctx->ggml_ctx, q, q_cs.first, q_cs.second);
                    auto k_cs = LTXRope::split_pe(ctx->ggml_ctx, k_pe_eff);
                    k = LTXRope::apply_rotary_emb_interleaved(ctx->ggml_ctx, k, k_cs.first, k_cs.second);
                }
            }

            auto out = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v,
                                              num_heads, mask, false, ctx->flash_attn_enabled);
            // out: [inner_dim, L_q, B]

            if (apply_gated_attention) {
                auto gate_proj  = std::dynamic_pointer_cast<Linear>(blocks["to_gate_logits"]);
                auto gate_logits = gate_proj->forward(ctx, x);  // [num_heads, L_q, B]
                auto gates       = ggml_sigmoid(ctx->ggml_ctx, gate_logits);
                gates            = ggml_scale(ctx->ggml_ctx, gates, 2.f);
                // out is [inner_dim, L_q, B]; reshape to [head_dim, num_heads, L_q, B], multiply gates as [1, num_heads, L_q, B] broadcast.
                int64_t L_q = out->ne[1];
                int64_t B   = out->ne[2];
                auto out4 = ggml_reshape_4d(ctx->ggml_ctx, out, head_dim, num_heads, L_q, B);
                auto g4   = ggml_reshape_4d(ctx->ggml_ctx, gates, 1, num_heads, L_q, B);
                out4 = ggml_mul(ctx->ggml_ctx, out4, g4);
                out  = ggml_reshape_3d(ctx->ggml_ctx, out4, inner_dim, L_q, B);
            }

            out = to_out->forward(ctx, out);  // [query_dim, L_q, B]
            return out;
        }
    };

    // PixArtAlphaTextProjection — caption_projection inside the DiT.
    // Python: ltx_core/model/transformer/text_projection.py.
    // linear_1 (caption_channels → hidden) → GELU(tanh) → linear_2 (hidden → out).
    // Used in V1 / 19B to bring the connector's 3840-dim output up to the DiT's
    // 4096-dim inner space. In config the `caption_proj_before_connector` flag
    // distinguishes V1 (True, used here) from V2 (False, handled separately).
    struct PixArtAlphaTextProjection : public GGMLBlock {
    protected:
        int64_t in_features;
        int64_t hidden_size;
        int64_t out_features;

    public:
        PixArtAlphaTextProjection() = default;
        PixArtAlphaTextProjection(int64_t in_features, int64_t hidden_size, int64_t out_features = 0)
            : in_features(in_features), hidden_size(hidden_size),
              out_features(out_features == 0 ? hidden_size : out_features) {
            blocks["linear_1"] = std::make_shared<Linear>(in_features, hidden_size, true);
            blocks["linear_2"] = std::make_shared<Linear>(hidden_size, this->out_features, true);
        }

        int64_t get_in_features() const { return in_features; }
        int64_t get_out_features() const { return out_features; }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto l1 = std::dynamic_pointer_cast<Linear>(blocks["linear_1"]);
            auto l2 = std::dynamic_pointer_cast<Linear>(blocks["linear_2"]);
            x       = l1->forward(ctx, x);
            x       = ggml_ext_gelu(ctx->ggml_ctx, x, /*approximate_tanh=*/true);
            x       = l2->forward(ctx, x);
            return x;
        }
    };

    // Args for one modality (video or audio) flowing through an AV transformer
    // block. Mirrors python TransformerArgs but only the fields the block needs.
    struct LTX2AVModalityArgs {
        ggml_tensor* x                            = nullptr;  // [dim, L, B] — set null to skip this modality
        ggml_tensor* context                      = nullptr;  // [ctx_dim, L_ctx, B]
        ggml_tensor* modulation                   = nullptr;  // [dim, 6_or_9, B]
        ggml_tensor* pe                           = nullptr;  // [inner_dim, L, 2]
        ggml_tensor* prompt_modulation            = nullptr;  // [dim, 2, B] or null (cross_attention_adaln only)
        ggml_tensor* context_mask                 = nullptr;  // [L_ctx, L, 1, B] or null
        // Cross-modal modulation tensors. Each block uses table[0:2] + cross_scale_shift_modulation[:,0:2,:]
        // for its a2v slot, table[2:4] + cross_scale_shift_modulation[:,2:4,:] for its v2a slot,
        // and table[4:5] + cross_gate_modulation for the gate.
        ggml_tensor* cross_scale_shift_modulation = nullptr;  // [dim, 4, B]
        ggml_tensor* cross_gate_modulation        = nullptr;  // [dim, 1, B]
        // Cross-modal RoPE positional embeddings. Computed at the model level
        // from the union of video+audio max_pos so both sides share scale.
        ggml_tensor* cross_pe                     = nullptr;  // [inner_dim_cross, L_cross, 2]
    };

    struct LTXTransformerBlock : public GGMLBlock {
    protected:
        int64_t dim;
        int num_heads;
        int head_dim;
        int64_t context_dim;
        bool cross_attention_adaln;
        bool apply_gated_attention;
        float norm_eps;

        // --- audio-video extension ---
        // When `has_audio_video == true`, the block additionally carries the
        // audio-side self-attn / text-CA / FFN, and the cross-modal a2v / v2a
        // attentions plus their scale_shift_table_a2v_ca_{audio,video} tables.
        // This mirrors python BasicAVTransformerBlock when `audio is not None
        // and video is not None`.
        bool    has_audio_video    = false;
        int64_t audio_dim          = 0;
        int     audio_num_heads    = 0;
        int     audio_head_dim     = 0;
        int64_t audio_context_dim  = 0;

        void init_params(ggml_context* ctx, const String2TensorStorage&, const std::string prefix = "") override {
            int num_params              = cross_attention_adaln ? ADALN_WITH_CA : ADALN_BASE;
            params["scale_shift_table"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, num_params);
            if (cross_attention_adaln) {
                params["prompt_scale_shift_table"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, 2);
            }
            if (has_audio_video) {
                // audio_scale_shift_table mirrors video's: 6 rows (or 9 with cross_attention_adaln)
                params["audio_scale_shift_table"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, audio_dim, num_params);
                if (cross_attention_adaln) {
                    params["audio_prompt_scale_shift_table"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, audio_dim, 2);
                }
                // 5-row tables: rows 0-1 = a2v scale/shift, rows 2-3 = v2a scale/shift, row 4 = gate.
                params["scale_shift_table_a2v_ca_audio"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, audio_dim, 5);
                params["scale_shift_table_a2v_ca_video"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, 5);
            }
        }

    public:
        LTXTransformerBlock() = default;
        LTXTransformerBlock(int64_t dim, int num_heads, int head_dim, int64_t context_dim,
                            bool cross_attention_adaln = false, bool apply_gated_attention = false,
                            float norm_eps = 1e-6f,
                            RopeType rope_type = RopeType::SPLIT,
                            // Audio-video config — set audio_dim > 0 to enable. Defaults disable
                            // the audio path so existing video-only construction is unchanged.
                            int64_t audio_dim         = 0,
                            int     audio_num_heads   = 0,
                            int     audio_head_dim    = 0,
                            int64_t audio_context_dim = 0)
            : dim(dim), num_heads(num_heads), head_dim(head_dim), context_dim(context_dim),
              cross_attention_adaln(cross_attention_adaln),
              apply_gated_attention(apply_gated_attention), norm_eps(norm_eps),
              has_audio_video(audio_dim > 0),
              audio_dim(audio_dim), audio_num_heads(audio_num_heads),
              audio_head_dim(audio_head_dim), audio_context_dim(audio_context_dim) {
            blocks["attn1"] = std::make_shared<LTXAttention>(dim, dim, num_heads, head_dim, apply_gated_attention, norm_eps, rope_type);
            blocks["attn2"] = std::make_shared<LTXAttention>(dim, context_dim, num_heads, head_dim, apply_gated_attention, norm_eps, rope_type);
            blocks["ff"]    = std::make_shared<FeedForward>(dim, dim);
            if (has_audio_video) {
                // Audio self-attention + audio text cross-attention + audio FFN.
                // Heads/d_head are AUDIO's (typically 32×64 vs video's 32×128).
                blocks["audio_attn1"] = std::make_shared<LTXAttention>(
                    audio_dim, audio_dim, audio_num_heads, audio_head_dim,
                    apply_gated_attention, norm_eps, rope_type);
                blocks["audio_attn2"] = std::make_shared<LTXAttention>(
                    audio_dim, audio_context_dim, audio_num_heads, audio_head_dim,
                    apply_gated_attention, norm_eps, rope_type);
                blocks["audio_ff"] = std::make_shared<FeedForward>(audio_dim, audio_dim);

                // Cross-modal: query_dim is the QUERYING modality's; context_dim is
                // the OTHER modality's; heads/d_head come from AUDIO config (for both,
                // matching python BasicAVTransformerBlock).
                blocks["audio_to_video_attn"] = std::make_shared<LTXAttention>(
                    /*query_dim=*/dim, /*context_dim=*/audio_dim,
                    audio_num_heads, audio_head_dim,
                    apply_gated_attention, norm_eps, rope_type);
                blocks["video_to_audio_attn"] = std::make_shared<LTXAttention>(
                    /*query_dim=*/audio_dim, /*context_dim=*/dim,
                    audio_num_heads, audio_head_dim,
                    apply_gated_attention, norm_eps, rope_type);
            }
        }

        // Helper — returns a triple (a, b, c) from scale_shift_table[start:start+3] + modulation[:, start:start+3, :]
        // scale_shift_table: ne [dim, num_params]
        // modulation: ne [dim, num_params, B]
        // Returns three tensors each ne [dim, 1, B].
        std::tuple<ggml_tensor*, ggml_tensor*, ggml_tensor*> extract_triple(ggml_context* ctx,
                                                                            ggml_tensor* sst,
                                                                            ggml_tensor* modulation,
                                                                            int start) {
            int64_t B = modulation->ne[2];

            // Slice scale_shift_table rows [start, start+3).
            auto sst_slice = ggml_ext_slice(ctx, sst, 1, start, start + 3);  // ne [dim, 3]

            // Slice modulation along dim 1 [start, start+3).
            auto mod_slice = ggml_ext_slice(ctx, modulation, 1, start, start + 3);  // ne [dim, 3, B]

            // Broadcast add: sst_slice [dim, 3] + mod_slice [dim, 3, B] → [dim, 3, B].
            auto combined = ggml_add(ctx, mod_slice, sst_slice);

            auto chunks = ggml_ext_chunk(ctx, combined, 3, 1);
            // Each chunk ne [dim, 1, B]
            return std::make_tuple(chunks[0], chunks[1], chunks[2]);
        }

        // Extract (shift, scale) from prompt_scale_shift_table [dim, 2] + prompt_modulation [dim, 2, B].
        // Python: `(prompt_scale_shift_table[None, None] + prompt_timestep.reshape(...,2,-1)).unbind(2)`.
        std::pair<ggml_tensor*, ggml_tensor*> extract_kv_pair(ggml_context* ctx,
                                                              ggml_tensor* psst,
                                                              ggml_tensor* prompt_mod) {
            auto combined = ggml_add(ctx, prompt_mod, psst);                   // [dim, 2, B]
            auto chunks   = ggml_ext_chunk(ctx, combined, 2, 1);
            return {chunks[0], chunks[1]};  // (shift_kv, scale_kv), each [dim, 1, B]
        }

        // Extract (scale, shift, gate) for an AV cross-modal slot.
        //   table:    [dim, 5] — row 0/1 a2v scale/shift, row 2/3 v2a scale/shift, row 4 gate
        //   ss_mod:   [dim, 4, B] — modulation matching the 4 scale/shift rows
        //   gate_mod: [dim, 1, B] — modulation for the gate row
        //   start:    0 for the a2v slot, 2 for the v2a slot
        // Returns three tensors each ne [dim, 1, B].
        std::tuple<ggml_tensor*, ggml_tensor*, ggml_tensor*> extract_av_modulation(
            ggml_context* ctx, ggml_tensor* table, ggml_tensor* ss_mod, ggml_tensor* gate_mod, int start) {
            // scale,shift = table[start:start+2] + ss_mod[:, start:start+2, :]
            auto sst_ss   = ggml_ext_slice(ctx, table, 1, start, start + 2);    // [dim, 2]
            auto mod_ss   = ggml_ext_slice(ctx, ss_mod, 1, start, start + 2);   // [dim, 2, B]
            auto sum_ss   = ggml_add(ctx, mod_ss, sst_ss);                       // [dim, 2, B]
            auto chunks_ss = ggml_ext_chunk(ctx, sum_ss, 2, 1);                  // 2× [dim, 1, B]
            auto scale    = chunks_ss[0];
            auto shift    = chunks_ss[1];

            // gate = table[4:5] + gate_mod
            auto sst_g  = ggml_ext_slice(ctx, table, 1, 4, 5);                   // [dim, 1]
            auto sum_g  = ggml_add(ctx, gate_mod, sst_g);                         // [dim, 1, B]
            auto chunks_g = ggml_ext_chunk(ctx, sum_g, 1, 1);                     // 1× [dim, 1, B]
            auto gate   = chunks_g[0];
            return std::make_tuple(scale, shift, gate);
        }

        // Apply text cross-attention path (V1 or V2 per cross_attention_adaln).
        // Mirrors the inner block of forward() so audio-side text-CA can reuse it.
        ggml_tensor* apply_text_ca(GGMLRunnerContext* ctx,
                                    ggml_tensor* x,
                                    ggml_tensor* context,
                                    ggml_tensor* context_mask,
                                    std::shared_ptr<LTXAttention> attn,
                                    ggml_tensor* sst,
                                    ggml_tensor* modulation,
                                    ggml_tensor* prompt_modulation,
                                    ggml_tensor* prompt_sst) {
            if (cross_attention_adaln) {
                GGML_ASSERT(prompt_modulation != nullptr && prompt_sst != nullptr);
                auto triple_ca = extract_triple(ctx->ggml_ctx, sst, modulation, 6);
                auto shift_q   = std::get<0>(triple_ca);
                auto scale_q   = std::get<1>(triple_ca);
                auto gate_q    = std::get<2>(triple_ca);

                auto kv_pair      = extract_kv_pair(ctx->ggml_ctx, prompt_sst, prompt_modulation);
                auto shift_kv     = kv_pair.first;
                auto scale_kv     = kv_pair.second;

                auto norm_x_ca     = parameterless_rms_norm(ctx->ggml_ctx, x, norm_eps);
                auto q_scaled      = ggml_add(ctx->ggml_ctx, norm_x_ca, ggml_mul(ctx->ggml_ctx, norm_x_ca, scale_q));
                auto q_modulated   = ggml_add(ctx->ggml_ctx, q_scaled, shift_q);
                auto ctx_scaled    = ggml_add(ctx->ggml_ctx, context, ggml_mul(ctx->ggml_ctx, context, scale_kv));
                auto ctx_modulated = ggml_add(ctx->ggml_ctx, ctx_scaled, shift_kv);

                auto ca_out = attn->forward(ctx, q_modulated, ctx_modulated, nullptr, context_mask);
                return ggml_add(ctx->ggml_ctx, x, ggml_mul(ctx->ggml_ctx, ca_out, gate_q));
            } else {
                auto norm_x_ca = parameterless_rms_norm(ctx->ggml_ctx, x, norm_eps);
                auto ca_out    = attn->forward(ctx, norm_x_ca, context, nullptr, context_mask);
                return ggml_add(ctx->ggml_ctx, x, ca_out);
            }
        }

        // x: [dim, L_q, B]
        // context: [context_dim, L_kv, B]
        // modulation: [dim, num_params, B]  (num_params = 6 for V1, 9 for V2)
        // pe: packed cos/sin tensor [dim, L_q, 2]
        // prompt_modulation: [dim, 2, B] — required when cross_attention_adaln=true, else nullptr
        // context_mask: [L_kv, L_q, 1, B] additive mask (or nullptr)
        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* context,
                             ggml_tensor* modulation,
                             ggml_tensor* pe,
                             ggml_tensor* prompt_modulation = nullptr,
                             ggml_tensor* context_mask      = nullptr,
                             // STG (Spatio-Temporal Guidance) perturbation: when true,
                             // bypass video self-attention entirely. Mirrors python's
                             // SKIP_VIDEO_SELF_ATTN with all_perturbed=True. The block's
                             // residual passes through unchanged for the self-attn step.
                             bool skip_self_attn            = false) {
            auto attn1 = std::dynamic_pointer_cast<LTXAttention>(blocks["attn1"]);
            auto attn2 = std::dynamic_pointer_cast<LTXAttention>(blocks["attn2"]);
            auto ff    = std::dynamic_pointer_cast<FeedForward>(blocks["ff"]);

            auto sst = params["scale_shift_table"];

            // --- Self-attention path (modulation slice 0:3 → shift, scale, gate) ---
            // Skipped entirely when skip_self_attn is set (STG perturbation pass).
            if (!skip_self_attn) {
                auto triple1       = extract_triple(ctx->ggml_ctx, sst, modulation, 0);
                auto shift_msa     = std::get<0>(triple1);
                auto scale_msa     = std::get<1>(triple1);
                auto gate_msa      = std::get<2>(triple1);

                auto norm_x        = parameterless_rms_norm(ctx->ggml_ctx, x, norm_eps);
                auto scaled        = ggml_add(ctx->ggml_ctx, norm_x, ggml_mul(ctx->ggml_ctx, norm_x, scale_msa));
                auto modulated     = ggml_add(ctx->ggml_ctx, scaled, shift_msa);
                auto attn_out      = attn1->forward(ctx, modulated, nullptr, pe, nullptr);
                x                  = ggml_add(ctx->ggml_ctx, x, ggml_mul(ctx->ggml_ctx, attn_out, gate_msa));
            }

            // --- Cross-attention ---
            // V1 (cross_attention_adaln=false): plain rms_norm → attn2 → residual.
            // V2 (cross_attention_adaln=true):
            //   modulation[6:9] → (q_shift, q_scale, q_gate) for the query path
            //   prompt_scale_shift_table + prompt_modulation → (kv_shift, kv_scale) for the context
            //   attn_input = rms_norm(x) * (1 + q_scale) + q_shift
            //   context_mod = context * (1 + kv_scale) + kv_shift
            //   x = x + attn2(attn_input, context_mod) * q_gate
            if (cross_attention_adaln) {
                GGML_ASSERT(prompt_modulation != nullptr && "cross_attention_adaln requires prompt_modulation");
                auto triple_ca = extract_triple(ctx->ggml_ctx, sst, modulation, 6);
                auto shift_q   = std::get<0>(triple_ca);
                auto scale_q   = std::get<1>(triple_ca);
                auto gate_q    = std::get<2>(triple_ca);

                auto psst       = params["prompt_scale_shift_table"];  // [dim, 2]
                auto kv_pair    = extract_kv_pair(ctx->ggml_ctx, psst, prompt_modulation);
                auto shift_kv   = kv_pair.first;
                auto scale_kv   = kv_pair.second;

                auto norm_x_ca    = parameterless_rms_norm(ctx->ggml_ctx, x, norm_eps);
                auto q_scaled     = ggml_add(ctx->ggml_ctx, norm_x_ca, ggml_mul(ctx->ggml_ctx, norm_x_ca, scale_q));
                auto q_modulated  = ggml_add(ctx->ggml_ctx, q_scaled, shift_q);
                auto ctx_scaled   = ggml_add(ctx->ggml_ctx, context, ggml_mul(ctx->ggml_ctx, context, scale_kv));
                auto ctx_modulated = ggml_add(ctx->ggml_ctx, ctx_scaled, shift_kv);

                auto ca_out = attn2->forward(ctx, q_modulated, ctx_modulated, nullptr, context_mask);
                x           = ggml_add(ctx->ggml_ctx, x, ggml_mul(ctx->ggml_ctx, ca_out, gate_q));
            } else {
                auto norm_x_ca = parameterless_rms_norm(ctx->ggml_ctx, x, norm_eps);
                auto ca_out    = attn2->forward(ctx, norm_x_ca, context, nullptr, context_mask);
                x              = ggml_add(ctx->ggml_ctx, x, ca_out);
            }

            // --- FeedForward path (modulation slice 3:6 → shift, scale, gate) ---
            auto triple2      = extract_triple(ctx->ggml_ctx, sst, modulation, 3);
            auto shift_mlp    = std::get<0>(triple2);
            auto scale_mlp    = std::get<1>(triple2);
            auto gate_mlp     = std::get<2>(triple2);

            auto norm_x2      = parameterless_rms_norm(ctx->ggml_ctx, x, norm_eps);
            auto scaled_mlp   = ggml_add(ctx->ggml_ctx, norm_x2, ggml_mul(ctx->ggml_ctx, norm_x2, scale_mlp));
            auto modulated_mlp = ggml_add(ctx->ggml_ctx, scaled_mlp, shift_mlp);
            auto ff_out       = ff->forward(ctx, modulated_mlp);
            x                 = ggml_add(ctx->ggml_ctx, x, ggml_mul(ctx->ggml_ctx, ff_out, gate_mlp));

            return x;
        }

        // Audio-video forward. Mirrors python BasicAVTransformerBlock.forward.
        // Either modality may be skipped by setting its `x = nullptr`. When
        // both are provided, the cross-modal a2v / v2a paths run.
        // Returns {video_out, audio_out}; either may be null when skipped.
        // NOTE: perturbation masks (PerturbationType.SKIP_*) are not modeled —
        // the block is always all-on. Parity tests should leave perturbations
        // unset on the python side too.
        std::pair<ggml_tensor*, ggml_tensor*> forward_av(GGMLRunnerContext* ctx,
                                                          LTX2AVModalityArgs vargs,
                                                          LTX2AVModalityArgs aargs) {
            GGML_ASSERT(has_audio_video && "block lacks audio-video weights");

            const bool run_vx  = (vargs.x != nullptr);
            const bool run_ax  = (aargs.x != nullptr);
            const bool run_a2v = run_vx && run_ax;
            const bool run_v2a = run_vx && run_ax;  // mirrors python: both modalities present

            ggml_tensor* vx = vargs.x;
            ggml_tensor* ax = aargs.x;

            auto v_sst        = params["scale_shift_table"];
            auto v_prompt_sst = cross_attention_adaln ? params["prompt_scale_shift_table"] : nullptr;
            auto a_sst        = params["audio_scale_shift_table"];
            auto a_prompt_sst = cross_attention_adaln ? params["audio_prompt_scale_shift_table"] : nullptr;
            auto sst_av_v     = params["scale_shift_table_a2v_ca_video"];  // [dim, 5]
            auto sst_av_a     = params["scale_shift_table_a2v_ca_audio"];  // [audio_dim, 5]

            auto v_attn1 = std::dynamic_pointer_cast<LTXAttention>(blocks["attn1"]);
            auto v_attn2 = std::dynamic_pointer_cast<LTXAttention>(blocks["attn2"]);
            auto v_ff    = std::dynamic_pointer_cast<FeedForward>(blocks["ff"]);
            auto a_attn1 = std::dynamic_pointer_cast<LTXAttention>(blocks["audio_attn1"]);
            auto a_attn2 = std::dynamic_pointer_cast<LTXAttention>(blocks["audio_attn2"]);
            auto a_ff    = std::dynamic_pointer_cast<FeedForward>(blocks["audio_ff"]);
            auto a2v_attn = std::dynamic_pointer_cast<LTXAttention>(blocks["audio_to_video_attn"]);
            auto v2a_attn = std::dynamic_pointer_cast<LTXAttention>(blocks["video_to_audio_attn"]);

            // === Video self-attn + text-CA ===
            if (run_vx) {
                auto t1 = extract_triple(ctx->ggml_ctx, v_sst, vargs.modulation, 0);
                auto sh = std::get<0>(t1), sc = std::get<1>(t1), ga = std::get<2>(t1);
                auto norm_v = parameterless_rms_norm(ctx->ggml_ctx, vx, norm_eps);
                auto scaled = ggml_add(ctx->ggml_ctx, norm_v, ggml_mul(ctx->ggml_ctx, norm_v, sc));
                auto modul  = ggml_add(ctx->ggml_ctx, scaled, sh);
                auto out    = v_attn1->forward(ctx, modul, nullptr, vargs.pe, nullptr);
                vx          = ggml_add(ctx->ggml_ctx, vx, ggml_mul(ctx->ggml_ctx, out, ga));
                vx = apply_text_ca(ctx, vx, vargs.context, vargs.context_mask,
                                   v_attn2, v_sst, vargs.modulation, vargs.prompt_modulation, v_prompt_sst);
            }

            // === Audio self-attn + text-CA ===
            if (run_ax) {
                auto t1 = extract_triple(ctx->ggml_ctx, a_sst, aargs.modulation, 0);
                auto sh = std::get<0>(t1), sc = std::get<1>(t1), ga = std::get<2>(t1);
                auto norm_a = parameterless_rms_norm(ctx->ggml_ctx, ax, norm_eps);
                auto scaled = ggml_add(ctx->ggml_ctx, norm_a, ggml_mul(ctx->ggml_ctx, norm_a, sc));
                auto modul  = ggml_add(ctx->ggml_ctx, scaled, sh);
                auto out    = a_attn1->forward(ctx, modul, nullptr, aargs.pe, nullptr);
                ax          = ggml_add(ctx->ggml_ctx, ax, ggml_mul(ctx->ggml_ctx, out, ga));
                ax = apply_text_ca(ctx, ax, aargs.context, aargs.context_mask,
                                   a_attn2, a_sst, aargs.modulation, aargs.prompt_modulation, a_prompt_sst);
            }

            // === Audio-Video cross-attention ===
            if (run_a2v || run_v2a) {
                auto vx_norm3 = run_vx ? parameterless_rms_norm(ctx->ggml_ctx, vx, norm_eps) : nullptr;
                auto ax_norm3 = run_ax ? parameterless_rms_norm(ctx->ggml_ctx, ax, norm_eps) : nullptr;

                if (run_a2v) {
                    // Q from video, K/V from audio.
                    auto v_av = extract_av_modulation(ctx->ggml_ctx, sst_av_v,
                                                       vargs.cross_scale_shift_modulation,
                                                       vargs.cross_gate_modulation, /*start=*/0);
                    auto v_scale = std::get<0>(v_av), v_shift = std::get<1>(v_av), gate_a2v = std::get<2>(v_av);
                    auto vx_scaled = ggml_add(ctx->ggml_ctx, vx_norm3, ggml_mul(ctx->ggml_ctx, vx_norm3, v_scale));
                    vx_scaled      = ggml_add(ctx->ggml_ctx, vx_scaled, v_shift);

                    auto a_av = extract_av_modulation(ctx->ggml_ctx, sst_av_a,
                                                       aargs.cross_scale_shift_modulation,
                                                       aargs.cross_gate_modulation, /*start=*/0);
                    auto a_scale = std::get<0>(a_av), a_shift = std::get<1>(a_av);
                    auto ax_scaled = ggml_add(ctx->ggml_ctx, ax_norm3, ggml_mul(ctx->ggml_ctx, ax_norm3, a_scale));
                    ax_scaled      = ggml_add(ctx->ggml_ctx, ax_scaled, a_shift);

                    // Cross-modal RoPE: Q (video) uses video.cross_pe, K (audio) uses audio.cross_pe.
                    auto out = a2v_attn->forward(ctx, vx_scaled, ax_scaled,
                                                  vargs.cross_pe, /*mask=*/nullptr,
                                                  /*k_pe=*/aargs.cross_pe);
                    vx = ggml_add(ctx->ggml_ctx, vx, ggml_mul(ctx->ggml_ctx, out, gate_a2v));
                }

                if (run_v2a) {
                    // Q from audio, K/V from video.
                    auto a_av = extract_av_modulation(ctx->ggml_ctx, sst_av_a,
                                                       aargs.cross_scale_shift_modulation,
                                                       aargs.cross_gate_modulation, /*start=*/2);
                    auto a_scale = std::get<0>(a_av), a_shift = std::get<1>(a_av), gate_v2a = std::get<2>(a_av);
                    auto ax_scaled = ggml_add(ctx->ggml_ctx, ax_norm3, ggml_mul(ctx->ggml_ctx, ax_norm3, a_scale));
                    ax_scaled      = ggml_add(ctx->ggml_ctx, ax_scaled, a_shift);

                    auto v_av = extract_av_modulation(ctx->ggml_ctx, sst_av_v,
                                                       vargs.cross_scale_shift_modulation,
                                                       vargs.cross_gate_modulation, /*start=*/2);
                    auto v_scale = std::get<0>(v_av), v_shift = std::get<1>(v_av);
                    auto vx_scaled = ggml_add(ctx->ggml_ctx, vx_norm3, ggml_mul(ctx->ggml_ctx, vx_norm3, v_scale));
                    vx_scaled      = ggml_add(ctx->ggml_ctx, vx_scaled, v_shift);

                    auto out = v2a_attn->forward(ctx, ax_scaled, vx_scaled,
                                                  aargs.cross_pe, /*mask=*/nullptr,
                                                  /*k_pe=*/vargs.cross_pe);
                    ax = ggml_add(ctx->ggml_ctx, ax, ggml_mul(ctx->ggml_ctx, out, gate_v2a));
                }
            }

            // === Video FF ===
            if (run_vx) {
                auto t = extract_triple(ctx->ggml_ctx, v_sst, vargs.modulation, 3);
                auto sh = std::get<0>(t), sc = std::get<1>(t), ga = std::get<2>(t);
                auto norm = parameterless_rms_norm(ctx->ggml_ctx, vx, norm_eps);
                auto scaled = ggml_add(ctx->ggml_ctx, norm, ggml_mul(ctx->ggml_ctx, norm, sc));
                auto modul  = ggml_add(ctx->ggml_ctx, scaled, sh);
                auto out    = v_ff->forward(ctx, modul);
                vx          = ggml_add(ctx->ggml_ctx, vx, ggml_mul(ctx->ggml_ctx, out, ga));
            }

            // === Audio FF ===
            if (run_ax) {
                auto t = extract_triple(ctx->ggml_ctx, a_sst, aargs.modulation, 3);
                auto sh = std::get<0>(t), sc = std::get<1>(t), ga = std::get<2>(t);
                auto norm = parameterless_rms_norm(ctx->ggml_ctx, ax, norm_eps);
                auto scaled = ggml_add(ctx->ggml_ctx, norm, ggml_mul(ctx->ggml_ctx, norm, sc));
                auto modul  = ggml_add(ctx->ggml_ctx, scaled, sh);
                auto out    = a_ff->forward(ctx, modul);
                ax          = ggml_add(ctx->ggml_ctx, ax, ggml_mul(ctx->ggml_ctx, out, ga));
            }

            return {vx, ax};
        }
    };

    struct LTXParams {
        int64_t in_channels                         = 128;
        int64_t out_channels                        = 128;
        int64_t inner_dim                           = 4096;
        int num_heads                               = 32;
        int head_dim                                = 128;
        int num_layers                              = 48;
        int64_t cross_attention_dim                 = 4096;
        bool cross_attention_adaln                  = false;
        bool apply_gated_attention                  = false;
        float norm_eps                              = 1e-6f;
        float positional_embedding_theta            = 10000.f;
        std::vector<int> positional_embedding_max_pos = {20, 2048, 2048};
        float timestep_scale_multiplier             = 1000.f;
        bool use_middle_indices_grid                = true;
        RopeType rope_type                          = RopeType::SPLIT;  // real LTX-2.3 default
        // Optional caption_projection sitting on the DiT side (V1 / 19B); absent for
        // tiny parity tests that feed context in DiT inner_dim already. When enabled,
        // `caption_channels` is the input dim (connector output) and `caption_hidden`
        // / `caption_out` follow the PixArtAlphaTextProjection defaults.
        bool has_caption_projection                 = false;
        int64_t caption_channels                    = 0;
        int64_t caption_hidden                      = 0;
        int64_t caption_out                         = 0;

        // ---- Audio-video extension (model_type=AudioVideo). ----
        // Set `has_audio_video=true` to enable the audio side end-to-end:
        // audio_patchify_proj, audio_adaln_single, audio_caption_projection,
        // audio_norm_out / audio_scale_shift_table / audio_proj_out, plus the
        // four cross-modal AdaLN modules (av_ca_*). Each transformer block
        // also gets its audio-side weights and cross-modal a2v / v2a tables.
        bool    has_audio_video                          = false;
        int64_t audio_in_channels                        = 128;
        int64_t audio_out_channels                       = 128;
        int64_t audio_inner_dim                          = 2048;       // 32 × 64 for 22B
        int     audio_num_heads                          = 32;
        int     audio_head_dim                           = 64;
        int64_t audio_cross_attention_dim                = 2048;
        std::vector<int> audio_positional_embedding_max_pos = {20};
        float   av_ca_timestep_scale_multiplier          = 1.f;
        // Audio-side caption projection (rare — most checkpoints don't carry it).
        bool    has_audio_caption_projection             = false;
        int64_t audio_caption_channels                   = 0;
        int64_t audio_caption_hidden                     = 0;
        int64_t audio_caption_out                        = 0;
    };

    struct LTXModel : public GGMLBlock {
        LTXParams p;

    protected:
        void init_params(ggml_context* ctx, const String2TensorStorage&, const std::string prefix = "") override {
            params["scale_shift_table"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, p.inner_dim, 2);
            if (p.has_audio_video) {
                params["audio_scale_shift_table"] =
                    ggml_new_tensor_2d(ctx, GGML_TYPE_F32, p.audio_inner_dim, 2);
            }
        }

    public:
        LTXModel() = default;
        LTXModel(LTXParams p) : p(p) {
            blocks["patchify_proj"] = std::make_shared<Linear>(p.in_channels, p.inner_dim, true);
            int coeff               = p.cross_attention_adaln ? ADALN_WITH_CA : ADALN_BASE;
            blocks["adaln_single"]  = std::make_shared<AdaLayerNormSingle>(p.inner_dim, coeff);
            blocks["proj_out"]      = std::make_shared<Linear>(p.inner_dim, p.out_channels, true);

            // V2: a second AdaLayerNormSingle that generates modulation for the
            // context path inside cross-attention. Python:
            // `prompt_adaln_single = AdaLayerNormSingle(inner_dim, embedding_coefficient=2)`.
            if (p.cross_attention_adaln) {
                blocks["prompt_adaln_single"] = std::make_shared<AdaLayerNormSingle>(p.inner_dim, 2);
            }

            // Audio-video components (model_type=AudioVideo).
            if (p.has_audio_video) {
                blocks["audio_patchify_proj"] =
                    std::make_shared<Linear>(p.audio_in_channels, p.audio_inner_dim, true);
                blocks["audio_adaln_single"] =
                    std::make_shared<AdaLayerNormSingle>(p.audio_inner_dim, coeff);
                blocks["audio_proj_out"] =
                    std::make_shared<Linear>(p.audio_inner_dim, p.audio_out_channels, true);
                if (p.cross_attention_adaln) {
                    blocks["audio_prompt_adaln_single"] =
                        std::make_shared<AdaLayerNormSingle>(p.audio_inner_dim, 2);
                }
                if (p.has_audio_caption_projection) {
                    blocks["audio_caption_projection"] = std::make_shared<PixArtAlphaTextProjection>(
                        p.audio_caption_channels, p.audio_caption_hidden, p.audio_caption_out);
                }
                // Cross-modal AdaLN modules. Coefficients per python LTXModel._init_audio_video:
                //   av_ca_video_scale_shift: 4 for the (a2v_scale, a2v_shift, v2a_scale, v2a_shift) row pack.
                //   av_ca_audio_scale_shift: same shape on audio side.
                //   av_ca_a2v_gate: 1 (single gate for video Q in a2v).
                //   av_ca_v2a_gate: 1 (single gate for audio Q in v2a).
                blocks["av_ca_video_scale_shift_adaln_single"] =
                    std::make_shared<AdaLayerNormSingle>(p.inner_dim, 4);
                blocks["av_ca_audio_scale_shift_adaln_single"] =
                    std::make_shared<AdaLayerNormSingle>(p.audio_inner_dim, 4);
                blocks["av_ca_a2v_gate_adaln_single"] =
                    std::make_shared<AdaLayerNormSingle>(p.inner_dim, 1);
                blocks["av_ca_v2a_gate_adaln_single"] =
                    std::make_shared<AdaLayerNormSingle>(p.audio_inner_dim, 1);
            }

            for (int i = 0; i < p.num_layers; ++i) {
                if (p.has_audio_video) {
                    blocks["transformer_blocks." + std::to_string(i)] =
                        std::make_shared<LTXTransformerBlock>(p.inner_dim, p.num_heads, p.head_dim,
                                                              p.cross_attention_dim,
                                                              p.cross_attention_adaln,
                                                              p.apply_gated_attention,
                                                              p.norm_eps,
                                                              p.rope_type,
                                                              p.audio_inner_dim,
                                                              p.audio_num_heads,
                                                              p.audio_head_dim,
                                                              p.audio_cross_attention_dim);
                } else {
                    blocks["transformer_blocks." + std::to_string(i)] =
                        std::make_shared<LTXTransformerBlock>(p.inner_dim, p.num_heads, p.head_dim,
                                                              p.cross_attention_dim,
                                                              p.cross_attention_adaln,
                                                              p.apply_gated_attention,
                                                              p.norm_eps,
                                                              p.rope_type);
                }
            }

            if (p.has_caption_projection) {
                blocks["caption_projection"] = std::make_shared<PixArtAlphaTextProjection>(
                    p.caption_channels, p.caption_hidden, p.caption_out);
            }
        }

        // latent: ne [in_channels, T*H*W, B]  (already patchified by caller)
        // timestep: ne [B]
        // context: ne [cross_attention_dim, S, B]
        // pe: ne [inner_dim, T*H*W, 2]  (interleaved cos/sin)
        // context_mask: ne [S, T*H*W, 1, B] or nullptr
        // Returns: ne [out_channels, T*H*W, B]
        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* latent,
                             ggml_tensor* timestep,
                             ggml_tensor* context,
                             ggml_tensor* pe,
                             ggml_tensor* context_mask                  = nullptr,
                             // STG (Spatio-Temporal Guidance): block indices whose video
                             // self-attention is bypassed during the perturbed pass.
                             // Empty by default — passing a non-empty set produces a
                             // weakened prediction used by the guider's stg_scale term.
                             const std::vector<int>* stg_skip_blocks    = nullptr) {
            auto patchify_proj = std::dynamic_pointer_cast<Linear>(blocks["patchify_proj"]);
            auto adaln_single  = std::dynamic_pointer_cast<AdaLayerNormSingle>(blocks["adaln_single"]);
            auto proj_out      = std::dynamic_pointer_cast<Linear>(blocks["proj_out"]);

            // Apply caption_projection (V1 / 19B) to lift context from connector dim
            // to DiT inner_dim. Python: TransformerArgs._prepare_context.
            if (p.has_caption_projection && context != nullptr) {
                auto caption_proj = std::dynamic_pointer_cast<PixArtAlphaTextProjection>(blocks["caption_projection"]);
                context           = caption_proj->forward(ctx, context);
            }

            auto x = patchify_proj->forward(ctx, latent);  // [inner_dim, T*H*W, B]

            // Caller must feed the already-scaled timestep (σ * 1000). The LTX2 denoiser's sigma_to_t
            // is the single source of truth for that scaling — see LTXParams::timestep_scale_multiplier
            // which is kept as documentation/config only, not applied here.
            auto adaln_res      = adaln_single->forward(ctx, timestep);
            auto modulation     = adaln_res.first;   // [inner_dim, coeff, B]  (coeff = 6 or 9)
            auto embedded_t     = adaln_res.second;  // [inner_dim, B]

            // V2: prompt_adaln_single takes the same σ (raw timestep before AdaLN-scaling)
            // and emits a [inner_dim, 2, B] modulation that's shared across all blocks'
            // cross-attention kv path. In Python video_args_preprocessor passes
            // `modality.sigma`; for our single-prompt inference sigma == timestep. We reuse
            // the same timestep tensor here.
            ggml_tensor* prompt_modulation = nullptr;
            if (p.cross_attention_adaln) {
                auto prompt_adaln = std::dynamic_pointer_cast<AdaLayerNormSingle>(blocks["prompt_adaln_single"]);
                auto prompt_res   = prompt_adaln->forward(ctx, timestep);
                prompt_modulation = prompt_res.first;  // [inner_dim, 2, B]
            }

            for (int i = 0; i < p.num_layers; ++i) {
                auto block = std::dynamic_pointer_cast<LTXTransformerBlock>(
                    blocks["transformer_blocks." + std::to_string(i)]);
                bool skip_self_attn = false;
                if (stg_skip_blocks != nullptr) {
                    for (int b : *stg_skip_blocks) {
                        if (b == i) { skip_self_attn = true; break; }
                    }
                }
                x = block->forward(ctx, x, context, modulation, pe, prompt_modulation,
                                   context_mask, skip_self_attn);
            }

            // Output modulation: python has `sst[None,None] + embedded[:,:,None]` giving (B, 1, 2, dim).
            // In ggml ne that's [dim, 2, 1, B]. For B>1 we'd need to broadcast sst over B explicitly;
            // current parity test uses B=1 so we pick the direct add path here and rely on ggml's
            // ggml_can_repeat(b, a) — `a` must be >= `b` in every dim so we put sst first.
            //   sst ne:     [inner_dim, 2, 1, 1]
            //   embedded:   [inner_dim, 1, 1, B]  (after reshape_4d from [inner_dim, B])
            //   sum:        [inner_dim, 2, 1, B]  (provided B == 1; see TODO for B>1)
            int64_t B = x->ne[2];
            GGML_ASSERT(B == 1 && "LTXModel output modulation currently assumes batch=1");
            auto sst      = params["scale_shift_table"];                                                 // ne [inner_dim, 2]
            auto emb_view = ggml_reshape_4d(ctx->ggml_ctx, embedded_t, p.inner_dim, 1, 1, B);             // ne [inner_dim, 1, 1, B]
            auto ss_sum   = ggml_add(ctx->ggml_ctx, sst, emb_view);                                      // ne [inner_dim, 2, 1, 1]
            auto chunks   = ggml_ext_chunk(ctx->ggml_ctx, ss_sum, 2, 1);                                 // 2× ne [inner_dim, 1, 1, 1]
            auto shift    = ggml_reshape_3d(ctx->ggml_ctx, chunks[0], p.inner_dim, 1, 1);                // ne [inner_dim, 1, 1]
            auto scale    = ggml_reshape_3d(ctx->ggml_ctx, chunks[1], p.inner_dim, 1, 1);                // ne [inner_dim, 1, 1]

            x = ggml_ext_layer_norm(ctx->ggml_ctx, x, nullptr, nullptr, p.norm_eps);                     // param-less LN

            // x ne: [inner_dim, T, 1]; scale/shift ne: [inner_dim, 1, 1] — second arg broadcasts ok.
            x = ggml_add(ctx->ggml_ctx, ggml_add(ctx->ggml_ctx, x, ggml_mul(ctx->ggml_ctx, x, scale)), shift);
            x = proj_out->forward(ctx, x);                                                               // [out_channels, T*H*W, B]
            return x;
        }

        // Helper for the output head: param-less LayerNorm + (sst[None, None] +
        // embedded[..., None]) modulation + linear projection. Mirrors python
        // LTXModel._process_output. B==1 only (matches existing forward()).
        ggml_tensor* process_output_head(GGMLRunnerContext* ctx,
                                          ggml_tensor* sst,         // [dim, 2]
                                          ggml_tensor* x,           // [dim, T, B=1]
                                          ggml_tensor* embedded_t,  // [dim, B=1]
                                          std::shared_ptr<Linear> proj_out,
                                          int64_t dim) {
            int64_t B = x->ne[2];
            GGML_ASSERT(B == 1 && "LTXModel output modulation currently assumes batch=1");
            auto emb_view = ggml_reshape_4d(ctx->ggml_ctx, embedded_t, dim, 1, 1, B);
            auto ss_sum   = ggml_add(ctx->ggml_ctx, sst, emb_view);
            auto chunks   = ggml_ext_chunk(ctx->ggml_ctx, ss_sum, 2, 1);
            auto shift    = ggml_reshape_3d(ctx->ggml_ctx, chunks[0], dim, 1, 1);
            auto scale    = ggml_reshape_3d(ctx->ggml_ctx, chunks[1], dim, 1, 1);
            x = ggml_ext_layer_norm(ctx->ggml_ctx, x, nullptr, nullptr, p.norm_eps);
            x = ggml_add(ctx->ggml_ctx, ggml_add(ctx->ggml_ctx, x, ggml_mul(ctx->ggml_ctx, x, scale)), shift);
            x = proj_out->forward(ctx, x);
            return x;
        }

        // Audio-video forward — mirrors python LTXModel.forward when both video
        // and audio modalities are provided. Returns {video_out, audio_out},
        // each in shape [out_channels, T, B].
        //
        // Inputs:
        //   *_latent: pre-patchify modality latent ([in_channels, T, B])
        //   *_t_self: σ·timestep_scale_multiplier — fed to {video,audio}_adaln_single
        //   *_t_prompt_self: same scaling — fed to *_prompt_adaln_single (only when
        //     cross_attention_adaln=true; pass nullptr otherwise)
        //   *_t_cross_ss: cross-modality σ·timestep_scale_multiplier — fed to
        //     av_ca_{video,audio}_scale_shift_adaln_single
        //   *_t_cross_gate: cross-modality σ·av_ca_timestep_scale_multiplier — fed
        //     to av_ca_{a2v,v2a}_gate_adaln_single
        //   *_context: text encoder output (if has_caption_projection, this is
        //     the unprojected version; otherwise must already be in inner_dim space)
        //   *_pe: per-modality positional embeddings ([inner_dim, T, 2])
        //   *_cross_pe: per-modality cross-modal positional embeddings sized to
        //     audio_inner_dim ([audio_inner_dim, T, 2])
        //   *_context_mask: optional additive log-bias mask
        std::pair<ggml_tensor*, ggml_tensor*> forward_av(
            GGMLRunnerContext* ctx,
            ggml_tensor* v_latent, ggml_tensor* a_latent,
            ggml_tensor* v_t_self, ggml_tensor* a_t_self,
            ggml_tensor* v_t_prompt_self, ggml_tensor* a_t_prompt_self,
            ggml_tensor* v_t_cross_ss, ggml_tensor* a_t_cross_ss,
            ggml_tensor* v_t_cross_gate, ggml_tensor* a_t_cross_gate,
            ggml_tensor* v_context, ggml_tensor* a_context,
            ggml_tensor* v_pe, ggml_tensor* a_pe,
            ggml_tensor* v_cross_pe, ggml_tensor* a_cross_pe,
            ggml_tensor* v_context_mask = nullptr,
            ggml_tensor* a_context_mask = nullptr) {
            GGML_ASSERT(p.has_audio_video && "LTXModel was not configured for audio-video");

            // ---- Video patchify + caption projection ----
            auto v_patchify = std::dynamic_pointer_cast<Linear>(blocks["patchify_proj"]);
            auto v_adaln    = std::dynamic_pointer_cast<AdaLayerNormSingle>(blocks["adaln_single"]);
            auto v_proj_out = std::dynamic_pointer_cast<Linear>(blocks["proj_out"]);
            if (p.has_caption_projection && v_context != nullptr) {
                auto cp = std::dynamic_pointer_cast<PixArtAlphaTextProjection>(blocks["caption_projection"]);
                v_context = cp->forward(ctx, v_context);
            }
            auto vx = v_patchify->forward(ctx, v_latent);
            auto v_adaln_res = v_adaln->forward(ctx, v_t_self);
            auto v_modulation       = v_adaln_res.first;
            auto v_embedded_timestep = v_adaln_res.second;
            ggml_tensor* v_prompt_modulation = nullptr;
            if (p.cross_attention_adaln) {
                GGML_ASSERT(v_t_prompt_self != nullptr);
                auto v_prompt_adaln = std::dynamic_pointer_cast<AdaLayerNormSingle>(blocks["prompt_adaln_single"]);
                v_prompt_modulation = v_prompt_adaln->forward(ctx, v_t_prompt_self).first;
            }

            // ---- Audio patchify + caption projection ----
            auto a_patchify = std::dynamic_pointer_cast<Linear>(blocks["audio_patchify_proj"]);
            auto a_adaln    = std::dynamic_pointer_cast<AdaLayerNormSingle>(blocks["audio_adaln_single"]);
            auto a_proj_out = std::dynamic_pointer_cast<Linear>(blocks["audio_proj_out"]);
            if (p.has_audio_caption_projection && a_context != nullptr) {
                auto cp = std::dynamic_pointer_cast<PixArtAlphaTextProjection>(blocks["audio_caption_projection"]);
                a_context = cp->forward(ctx, a_context);
            }
            auto ax = a_patchify->forward(ctx, a_latent);
            auto a_adaln_res = a_adaln->forward(ctx, a_t_self);
            auto a_modulation        = a_adaln_res.first;
            auto a_embedded_timestep = a_adaln_res.second;
            ggml_tensor* a_prompt_modulation = nullptr;
            if (p.cross_attention_adaln) {
                GGML_ASSERT(a_t_prompt_self != nullptr);
                auto a_prompt_adaln = std::dynamic_pointer_cast<AdaLayerNormSingle>(blocks["audio_prompt_adaln_single"]);
                a_prompt_modulation = a_prompt_adaln->forward(ctx, a_t_prompt_self).first;
            }

            // ---- Cross-modal AdaLN modulations (one set per modality) ----
            auto v_cross_ss_adaln = std::dynamic_pointer_cast<AdaLayerNormSingle>(blocks["av_ca_video_scale_shift_adaln_single"]);
            auto a_cross_ss_adaln = std::dynamic_pointer_cast<AdaLayerNormSingle>(blocks["av_ca_audio_scale_shift_adaln_single"]);
            auto v_cross_gate_adaln = std::dynamic_pointer_cast<AdaLayerNormSingle>(blocks["av_ca_a2v_gate_adaln_single"]);
            auto a_cross_gate_adaln = std::dynamic_pointer_cast<AdaLayerNormSingle>(blocks["av_ca_v2a_gate_adaln_single"]);
            auto v_cross_ss_mod   = v_cross_ss_adaln->forward(ctx, v_t_cross_ss).first;     // [inner_dim, 4, B]
            auto a_cross_ss_mod   = a_cross_ss_adaln->forward(ctx, a_t_cross_ss).first;     // [audio_inner_dim, 4, B]
            auto v_cross_gate_mod = v_cross_gate_adaln->forward(ctx, v_t_cross_gate).first; // [inner_dim, 1, B]
            auto a_cross_gate_mod = a_cross_gate_adaln->forward(ctx, a_t_cross_gate).first; // [audio_inner_dim, 1, B]

            // ---- Run all transformer blocks ----
            for (int i = 0; i < p.num_layers; ++i) {
                auto block = std::dynamic_pointer_cast<LTXTransformerBlock>(
                    blocks["transformer_blocks." + std::to_string(i)]);
                LTX2AVModalityArgs vargs;
                vargs.x = vx; vargs.context = v_context; vargs.modulation = v_modulation;
                vargs.pe = v_pe; vargs.cross_pe = v_cross_pe;
                vargs.prompt_modulation = v_prompt_modulation;
                vargs.context_mask = v_context_mask;
                vargs.cross_scale_shift_modulation = v_cross_ss_mod;
                vargs.cross_gate_modulation        = v_cross_gate_mod;

                LTX2AVModalityArgs aargs;
                aargs.x = ax; aargs.context = a_context; aargs.modulation = a_modulation;
                aargs.pe = a_pe; aargs.cross_pe = a_cross_pe;
                aargs.prompt_modulation = a_prompt_modulation;
                aargs.context_mask = a_context_mask;
                aargs.cross_scale_shift_modulation = a_cross_ss_mod;
                aargs.cross_gate_modulation        = a_cross_gate_mod;

                auto outs = block->forward_av(ctx, vargs, aargs);
                vx = outs.first;
                ax = outs.second;
            }

            // ---- Output heads ----
            auto v_sst = params["scale_shift_table"];
            auto a_sst = params["audio_scale_shift_table"];
            auto v_out = process_output_head(ctx, v_sst, vx, v_embedded_timestep, v_proj_out, p.inner_dim);
            auto a_out = process_output_head(ctx, a_sst, ax, a_embedded_timestep, a_proj_out, p.audio_inner_dim);
            return {v_out, a_out};
        }
    };

    struct LTXRunner : public GGMLRunner {
    public:
        LTXParams ltx_params;
        LTXModel ltx;
        std::vector<float> pe_vec;
        SDVersion version;
        // fps used for temporal RoPE normalisation — see LTXRope::gen_video_positions.
        // Defaults to 24 (LTX-2's canonical output fps); callers can override before compute().
        float fps = 24.0f;
        // VAE spatiotemporal compression factors (time, height, width) applied to latent
        // coordinates to reconstruct the pixel-space positions used for RoPE. Defaults match
        // the LTX-2 22B VAE: 8× temporal, 32× spatial. The parity tests feed the Python model
        // simplified positions (f/fps, h, w) — set scale_factors={1,1,1} and causal_fix=false
        // in that path to keep parity assertions valid.
        std::vector<int> scale_factors = {8, 32, 32};
        bool causal_fix                = true;

        void set_fps(float new_fps) { fps = new_fps; }
        void set_scale_factors(int time, int height, int width) {
            scale_factors = {time, height, width};
        }
        void set_causal_fix(bool enable) { causal_fix = enable; }

        // params_override forces the given LTXParams instead of auto-detecting from the tensor map.
        // Useful for parity tests and for cases where metadata pins the head_dim / num_heads to
        // values that can't be inferred from weight shapes alone (q_norm etc. are inner_dim-wide).
        LTXRunner(ggml_backend_t backend,
                  bool offload_params_to_cpu,
                  const String2TensorStorage& tensor_storage_map = {},
                  const std::string prefix                       = "model.diffusion_model",
                  SDVersion version                              = VERSION_LTX2,
                  const LTXParams* params_override               = nullptr)
            : GGMLRunner(backend, offload_params_to_cpu), version(version) {
            if (params_override != nullptr) {
                ltx_params = *params_override;
            } else {
                detect_params(tensor_storage_map, prefix);
            }
            ltx = LTXModel(ltx_params);
            ltx.init(params_ctx, tensor_storage_map, prefix);
        }

        void detect_params(const String2TensorStorage& tensor_storage_map, const std::string& prefix) {
            std::string pre = prefix.empty() ? "" : prefix + ".";

            auto patchify_it = tensor_storage_map.find(pre + "patchify_proj.weight");
            if (patchify_it != tensor_storage_map.end()) {
                const auto& ts = patchify_it->second;
                if (ts.n_dims >= 2) {
                    ltx_params.in_channels = ts.ne[0];
                    ltx_params.inner_dim   = ts.ne[1];
                }
            }

            auto proj_out_it = tensor_storage_map.find(pre + "proj_out.weight");
            if (proj_out_it != tensor_storage_map.end()) {
                const auto& ts = proj_out_it->second;
                if (ts.n_dims >= 2) {
                    ltx_params.out_channels = ts.ne[1];
                }
            }

            // Infer num_layers from highest transformer_blocks index.
            int max_layer = -1;
            std::string block_prefix = pre + "transformer_blocks.";
            for (auto& pair : tensor_storage_map) {
                const std::string& name = pair.first;
                if (name.rfind(block_prefix, 0) != 0) {
                    continue;
                }
                size_t start = block_prefix.size();
                size_t end   = name.find('.', start);
                if (end == std::string::npos) {
                    continue;
                }
                try {
                    int idx   = std::stoi(name.substr(start, end - start));
                    max_layer = std::max(max_layer, idx);
                } catch (...) {
                }
            }
            if (max_layer >= 0) {
                ltx_params.num_layers = max_layer + 1;
            }

            // Detect cross_attention_adaln from the size of scale_shift_table (9 if CA-AdaLN, 6 otherwise).
            auto sst_it = tensor_storage_map.find(pre + "transformer_blocks.0.scale_shift_table");
            if (sst_it != tensor_storage_map.end()) {
                const auto& ts = sst_it->second;
                if (ts.n_dims >= 2 && ts.ne[1] == ADALN_WITH_CA) {
                    ltx_params.cross_attention_adaln = true;
                }
            }

            // Infer head_dim × num_heads from attn1.to_q.weight shape.
            auto q_it = tensor_storage_map.find(pre + "transformer_blocks.0.attn1.to_q.weight");
            if (q_it != tensor_storage_map.end()) {
                const auto& ts = q_it->second;
                if (ts.n_dims >= 2) {
                    ltx_params.inner_dim = ts.ne[1];
                }
            }
            // head_dim is a fixed LTX-2 hyperparam (128) unless a config tensor overrides.
            ltx_params.head_dim  = 128;
            ltx_params.num_heads = static_cast<int>(ltx_params.inner_dim / ltx_params.head_dim);

            // Infer cross_attention_dim from attn2.to_k weight shape.
            auto k_it = tensor_storage_map.find(pre + "transformer_blocks.0.attn2.to_k.weight");
            if (k_it != tensor_storage_map.end()) {
                const auto& ts = k_it->second;
                if (ts.n_dims >= 2) {
                    ltx_params.cross_attention_dim = ts.ne[0];
                }
            }

            // Detect gated attention from presence of to_gate_logits.
            auto gate_it = tensor_storage_map.find(pre + "transformer_blocks.0.attn1.to_gate_logits.weight");
            if (gate_it != tensor_storage_map.end()) {
                ltx_params.apply_gated_attention = true;
            }

            // Detect optional caption_projection (V1 / 19B).
            // linear_1 weight shape [in_features, hidden_size]; linear_2 shape [hidden_size, out_features].
            // (ggml ne[0] = innermost dim = PyTorch's in_features / hidden_size.)
            auto cap1_it = tensor_storage_map.find(pre + "caption_projection.linear_1.weight");
            auto cap2_it = tensor_storage_map.find(pre + "caption_projection.linear_2.weight");
            if (cap1_it != tensor_storage_map.end() && cap2_it != tensor_storage_map.end()) {
                const auto& l1 = cap1_it->second;
                const auto& l2 = cap2_it->second;
                if (l1.n_dims >= 2 && l2.n_dims >= 2) {
                    ltx_params.has_caption_projection = true;
                    ltx_params.caption_channels       = l1.ne[0];
                    ltx_params.caption_hidden         = l1.ne[1];
                    ltx_params.caption_out            = l2.ne[1];
                }
            }
        }

        std::string get_desc() override {
            return "ltx2";
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string prefix) {
            ltx.get_param_tensors(tensors, prefix);
        }

        // Build the diffusion graph.
        // x_tensor layout (ggml ne order): [W, H, T, in_channels] — follows the Wan / video convention with implicit batch N=1.
        // timesteps: ne [N]
        // context: ne [cross_attention_dim, S, N]
        // context_mask: empty (not yet wired through)
        ggml_cgraph* build_graph(const sd::Tensor<float>& x_tensor,
                                 const sd::Tensor<float>& timesteps_tensor,
                                 const sd::Tensor<float>& context_tensor,
                                 const sd::Tensor<float>& context_mask_tensor,
                                 const std::vector<int>* stg_skip_blocks = nullptr) {
            ggml_cgraph* gf = new_graph_custom(LTX_GRAPH_SIZE);

            ggml_tensor* x         = make_input(x_tensor);
            ggml_tensor* timesteps = make_input(timesteps_tensor);
            ggml_tensor* context   = make_input(context_tensor);
            ggml_tensor* ctx_mask  = context_mask_tensor.empty() ? nullptr : make_input(context_mask_tensor);

            int64_t W = x->ne[0];
            int64_t H = x->ne[1];
            int64_t T = x->ne[2];
            int64_t C = x->ne[3];

            LOG_DEBUG("LTX build_graph: x=[%lld,%lld,%lld,%lld] timesteps=[%lld] context=[%lld,%lld,%lld] inner_dim=%lld cross_attn_dim=%lld has_cap_proj=%d ca_adaln=%d gated=%d",
                      (long long)x->ne[0], (long long)x->ne[1], (long long)x->ne[2], (long long)x->ne[3],
                      (long long)timesteps->ne[0],
                      (long long)context->ne[0], (long long)context->ne[1], (long long)context->ne[2],
                      (long long)ltx_params.inner_dim, (long long)ltx_params.cross_attention_dim,
                      ltx_params.has_caption_projection ? 1 : 0,
                      ltx_params.cross_attention_adaln ? 1 : 0,
                      ltx_params.apply_gated_attention ? 1 : 0);

            // Flatten spatiotemporal dims into a sequence and move channels to ne[0].
            auto latent = ggml_reshape_3d(compute_ctx, x, W * H * T, C, 1);               // [W*H*T, C, 1]
            latent      = ggml_cont(compute_ctx, ggml_permute(compute_ctx, latent, 1, 0, 2, 3));  // [C, W*H*T, 1]

            auto positions = LTXRope::gen_video_positions(static_cast<int>(T), static_cast<int>(H), static_cast<int>(W),
                                                          ltx_params.use_middle_indices_grid, fps,
                                                          scale_factors, causal_fix);
            ggml_tensor* pe = nullptr;
            if (ltx_params.rope_type == RopeType::SPLIT) {
                pe_vec = LTXRope::precompute_freqs_cis_split(positions,
                                                             static_cast<int>(ltx_params.inner_dim),
                                                             ltx_params.num_heads,
                                                             ltx_params.positional_embedding_theta,
                                                             ltx_params.positional_embedding_max_pos);
                // Split layout ne: [head_dim/2, num_heads, T*H*W, 2].
                int64_t half = ltx_params.inner_dim / 2;
                int64_t per_head_half = half / ltx_params.num_heads;
                pe = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32,
                                        per_head_half, ltx_params.num_heads, T * H * W, 2);
            } else {
                pe_vec = LTXRope::precompute_freqs_cis_interleaved(positions,
                                                                   static_cast<int>(ltx_params.inner_dim),
                                                                   ltx_params.positional_embedding_theta,
                                                                   ltx_params.positional_embedding_max_pos);
                pe = ggml_new_tensor_3d(compute_ctx, GGML_TYPE_F32, ltx_params.inner_dim, T * H * W, 2);
            }
            set_backend_tensor_data(pe, pe_vec.data());

            auto runner_ctx  = get_context();
            ggml_tensor* out = ltx.forward(&runner_ctx, latent, timesteps, context, pe, ctx_mask,
                                            stg_skip_blocks);

            // out: [out_channels, T*H*W, 1] → [W, H, T, out_channels] to match Wan-style output.
            out = ggml_cont(compute_ctx, ggml_permute(compute_ctx, out, 1, 0, 2, 3));  // [T*H*W, out_channels, 1]
            out = ggml_reshape_4d(compute_ctx, out, W, H, T, ltx_params.out_channels);

            ggml_build_forward_expand(gf, out);
            return gf;
        }

        sd::Tensor<float> compute(int n_threads,
                                  const sd::Tensor<float>& x,
                                  const sd::Tensor<float>& timesteps,
                                  const sd::Tensor<float>& context,
                                  const sd::Tensor<float>& context_mask,
                                  const std::vector<int>* stg_skip_blocks = nullptr) {
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(x, timesteps, context, context_mask, stg_skip_blocks);
            };
            return take_or_empty(GGMLRunner::compute<float>(get_graph, n_threads, true));
        }
    };

}  // namespace LTX

#endif  // __LTX_HPP__
