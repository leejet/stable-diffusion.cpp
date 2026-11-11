#ifndef __SD_MODEL_DIFFUSION_NUCLEUS_IMAGE_HPP__
#define __SD_MODEL_DIFFUSION_NUCLEUS_IMAGE_HPP__

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/ggml_graph_cut.h"
#include "core/util.h"
#include "model/common/ggml_block.hpp"
#include "model/common/rope.hpp"
#include "model/diffusion/dit.hpp"
#include "model/diffusion/model.hpp"
#include "model_loader.h"

// Nucleus-Image: sparse mixture-of-experts diffusion transformer with expert-choice routing
// (diffusers: NucleusMoEImageTransformer2DModel, text encoder Qwen3-VL-8B, Qwen-Image VAE).
//
// Reference math: diffusers transformer_nucleusmoe_image.py (see NucleusMoELayer / NucleusMoEAttnProcessor2_0 /
// NucleusMoEImageTransformerBlock). The GGUF checkpoint ships no metadata, so all hyper-parameters are hardcoded
// in NucleusImageConfig and cross-checked against weight shapes in detect_from_weights.
//
// MoE expert weights are stored in the file pre-transposed per expert (torch layout (E, in, out), i.e. ggml
// ne = [out, in, E]) and are declared with exactly that layout in NucleusMoELayer::init_params, so they load
// through the regular ModelManager path like every other weight. ggml_mul_mat contracts ne[0], so each expert
// slice is dequantized to F32 and transposed in the graph (see NucleusMoELayer::expert_weight).
//
// Expert-choice routing keeps every (token, expert) contribution: each expert processes its own top-C tokens
// and scatter-adds the result, so a token picked by k experts receives all k contributions (no per-token
// expert slot limit).
namespace NucleusImage {

struct NucleusImageConfig {
    int patch_size              = 2;
    int64_t in_channels         = 64;   // patchified input dim = patch_size^2 * 16
    int64_t out_channels        = 16;   // Qwen-Image VAE latent channels
    int64_t hidden_size         = 2048; // H
    int64_t num_attention_heads = 16;
    int64_t num_kv_heads        = 4;    // GQA
    int64_t head_dim            = 128;
    int64_t depth               = 32;
    int64_t joint_attention_dim = 4096;  // Qwen3-VL-8B hidden size
    int64_t dense_inner_dim     = 5376;  // dense SwiGLU inner dim (mlp_ratio 4, rounded to 128)
    int64_t moe_inner_dim       = 1344;  // I: expert / shared-expert inner dim
    int64_t num_experts         = 64;    // E
    float capacity_factor_first = 4.f;   // MoE layers 3-4
    float capacity_factor_rest  = 2.f;   // MoE layers 5+
    float route_scale           = 2.5f;
    bool use_sigmoid            = false;  // softmax router
    int num_first_dense_layers  = 3;
    int theta                   = 10000;
    std::vector<int> axes_dim   = {16, 56, 56};
    int axes_dim_sum            = 128;
    float norm_eps              = 1e-6f;
    std::vector<int> moe_layers;

    static NucleusImageConfig detect_from_weights(const String2TensorStorage& weights, const std::string& prefix) {
        NucleusImageConfig config;
        config.depth     = 0;
        config.moe_layers.clear();
        const std::string p = prefix + ".";
        for (const auto& [name, tensor] : weights) {
            if (!starts_with(name, p)) {
                continue;
            }
            auto key = name.substr(p.size());
            if (key == "img_in.weight" && tensor.n_dims == 2) {
                // [in, out] ggml layout: in = patchified channels, out = hidden
                config.in_channels = tensor.ne[0];
                config.hidden_size = tensor.ne[1];
            } else if (key == "txt_norm.weight") {
                config.joint_attention_dim = tensor.ne[0];
            } else if (key == "proj_out.weight" && tensor.n_dims == 2) {
                config.out_channels = tensor.ne[1] / (2 * 2);  // patch_size * patch_size
            } else if (key == "transformer_blocks.0.img_mlp.net.0.proj.weight" && tensor.n_dims == 2) {
                config.dense_inner_dim = tensor.ne[1] / 2;  // fused gate+up
            } else if (key == "transformer_blocks.0.attn.norm_q.weight") {
                config.head_dim = tensor.ne[0];
            } else if (key == "transformer_blocks.0.img_mlp.experts.gate_up_proj" && tensor.n_dims == 3) {
                // file layout [2I, H, E] -> E experts, I = moe inner dim
                config.num_experts   = tensor.ne[2];
                config.moe_inner_dim = tensor.ne[0] / 2;
            }
            if (starts_with(key, "transformer_blocks.")) {
                auto items = split_string(key, '.');
                if (items.size() > 1) {
                    int block_index = atoi(items[1].c_str());
                    config.depth    = std::max<int64_t>(config.depth, block_index + 1);
                    if (key == "transformer_blocks." + std::to_string(block_index) + ".img_mlp.experts.gate_up_proj") {
                        config.moe_layers.push_back(block_index);
                    }
                }
            }
        }
        std::sort(config.moe_layers.begin(), config.moe_layers.end());
        config.num_attention_heads = config.hidden_size / config.head_dim;
        config.axes_dim_sum        = 0;
        for (int axis_dim : config.axes_dim) {
            config.axes_dim_sum += axis_dim;
        }
        GGML_ASSERT(config.head_dim * config.num_attention_heads == config.hidden_size);
        GGML_ASSERT(config.hidden_size == 2048 && config.depth == 32 && config.num_experts == 64 &&
                    config.moe_inner_dim == 1344);
        LOG_VERBOSE("nucleus_image: hidden = %" PRId64 ", depth = %" PRId64 ", heads = %" PRId64 ", text_dim = %" PRId64
                    ", experts = %" PRId64 ", moe_inner = %" PRId64 ", moe_layers = %zu",
                    config.hidden_size, config.depth, config.num_attention_heads, config.joint_attention_dim,
                    config.num_experts, config.moe_inner_dim, config.moe_layers.size());
        return config;
    }

    bool is_moe_layer(int layer) const {
        return std::find(moe_layers.begin(), moe_layers.end(), layer) != moe_layers.end();
    }

    float capacity_factor_for(int layer) const {
        // Nucleus-Image config: layers 3-4 use cf=4, layers 5+ use cf=2
        // (capacity_factors list from the released checkpoint config).
        return layer <= 4 ? capacity_factor_first : capacity_factor_rest;
    }
};

// diffusers FeedForward(activation_fn="swiglu", bias=False):
// net.0.proj: H -> 2*inner (fused gate/up), net.2: inner -> H.
// Used for the dense blocks (inner = 5376) and the MoE shared expert (inner = 1344).
struct NucleusFeedForward : public UnaryBlock {
    NucleusFeedForward(int64_t dim, int64_t inner_dim) {
        blocks["net.0.proj"] = std::make_shared<Linear>(dim, inner_dim * 2, false);
        blocks["net.2"]      = std::make_shared<Linear>(inner_dim, dim, false);
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        auto c = ctx->ggml_ctx;
        auto net2 = std::dynamic_pointer_cast<Linear>(blocks["net.2"]);
        // Same Vulkan/ROCm precision guard as the shared FeedForward: quantized down-projections
        // accumulate in fp16 there without it.
        if (sd_backend_is(ctx->backend, "Vulkan") || sd_backend_is(ctx->backend, "ROCm")) {
            net2->set_force_prec_f32(true);
        }
        // Exact power-of-two rescales: Nucleus activations reach |x| ~ 1e3 (net0 input) and
        // ~2e5 (net2 input); matmul kernels on this stack accumulate in fp16 (max 65504) and
        // overflow. Linear ops, so pre-scale + post-scale is bit-exact in F32.
        auto net0 = std::dynamic_pointer_cast<Linear>(blocks["net.0.proj"]);
        x         = net0->forward(ctx, ggml_scale(c, x, 1.f / 64.f));
        x         = ggml_scale(c, x, 64.f);
        // diffusers FeedForward(activation_fn="swiglu") -> SwiGLU module:
        //   hidden, gate = proj(x).chunk(2, dim=-1); out = hidden * silu(gate)
        // i.e. the FIRST half is the VALUE and silu applies to the SECOND half.
        // (Mini-test vs diffusers NucleusMoELayer.shared_expert: cos 0.79 before, 1.0 after.)
        // NOTE: the routed MoE experts (SwiGLUExperts) are the opposite convention
        // (gate first, silu(gate) * up) and are implemented as such below.
        auto up   = ggml_ext_slice(ctx, x, 0, 0, x->ne[0] / 2);
        auto gate = ggml_ext_slice(ctx, x, 0, x->ne[0] / 2, x->ne[0]);
        gate      = ggml_silu(c, gate);
        x = ggml_mul(c, gate, up);
        x = net2->forward(ctx, ggml_scale(c, x, 1.f / 4096.f));
        return ggml_scale(c, x, 4096.f);
    }
};

// diffusers NucleusMoELayer: expert-choice routing with a shared expert.
//
// Expert-choice routing (per recipe, verified against diffusers):
//  1) router_in = concat(temb, x_unmodulated) [2H, S, B]; P = softmax over experts of
//     gate(router_in) [E, S, B].
//  2) Each expert e keeps its top-C tokens: IDX = argsort_top_k(P^T [S, E, B], C) (descending).
//  3) Gating matrix M [1, S, E, B]: scatter P[e, t] into (t, e) for the selected pairs.
//  4) Per-token denominator: denom[t, b] = sum_e M[:, t, e, b]; renormalized gating
//     G = M / max(denom, 1e-12) * route_scale. A token picked by k experts keeps all k
//     contributions, each weighted by its share of the (unnormalized) router mass.
//  5) Routed sum: for each expert, gather its C tokens (ggml_get_rows), run the SwiGLU
//     expert, weight by G and scatter-add back (ggml_set_rows into zeros + ggml_add).
//  6) out = routed_sum + shared_expert(x_mod).
struct NucleusMoELayer : public GGMLBlock {
    NucleusImageConfig config;
    float capacity_factor;
    int64_t hidden_size;
    int64_t inner_dim;
    int64_t num_experts;

    NucleusMoELayer(const NucleusImageConfig& config_, float capacity_factor_)
        : config(config_),
          capacity_factor(capacity_factor_),
          hidden_size(config_.hidden_size),
          inner_dim(config_.moe_inner_dim),
          num_experts(config_.num_experts) {
        blocks["gate"]          = std::make_shared<Linear>(hidden_size * 2, num_experts, false);
        blocks["shared_expert"] = std::make_shared<NucleusFeedForward>(hidden_size, inner_dim);
    }

    void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
        // NOTE: the router Linear lives in blocks["gate"] and registers/loads its own
        // "...img_mlp.gate.weight" param - do NOT create a manual param with the same full name
        // here: get_param_tensors would collide and the graph would reference the unloaded tensor.
        //
        // Expert tensors are declared in the checkpoint's own layout ([out, in, E]) so the
        // model manager loads them as-is; expert_weight() handles the transpose in the graph.
        ggml_type gate_up_type = get_type(prefix + "experts.gate_up_proj", tensor_storage_map, GGML_TYPE_F32);
        if (inner_dim * 2 % ggml_blck_size(gate_up_type) != 0) {
            gate_up_type = GGML_TYPE_F32;
        }
        params["experts.gate_up_proj"] = ggml_new_tensor_3d(ctx, gate_up_type, inner_dim * 2, hidden_size, num_experts);
        ggml_type down_type = get_type(prefix + "experts.down_proj", tensor_storage_map, GGML_TYPE_F32);
        if (hidden_size % ggml_blck_size(down_type) != 0) {
            down_type = GGML_TYPE_F32;
        }
        params["experts.down_proj"] = ggml_new_tensor_3d(ctx, down_type, hidden_size, inner_dim, num_experts);
    }

    // Expert e's weight matrix in the [in, out] layout ggml_mul_mat contracts (ne[0] = in).
    // The stored slice is [out, in]; dequantize to F32, then permute + cont. F32 also keeps
    // the expert matmuls clear of the fp16 accumulation ceiling of the quantized kernels.
    ggml_tensor* expert_weight(GGMLRunnerContext* ctx, const char* param, int64_t e, int64_t in, int64_t out) {
        auto c   = ctx->ggml_ctx;
        auto w   = params[param];  // [out, in, E]
        auto w_e = ggml_view_2d(c, w, out, in, w->nb[1], e * w->nb[2]);  // [out, in]
        w_e      = ggml_cast(c, w_e, GGML_TYPE_F32);
        return ggml_ext_cont(ctx, ggml_permute(c, w_e, 1, 0, 2, 3));  // [in, out]
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x_mod, ggml_tensor* x_unmodulated, ggml_tensor* temb) {
        auto c             = ctx->ggml_ctx;
        auto gate          = std::dynamic_pointer_cast<Linear>(blocks["gate"]);
        auto shared_expert = std::dynamic_pointer_cast<NucleusFeedForward>(blocks["shared_expert"]);

        const int64_t H = hidden_size;
        const int64_t E = num_experts;
        const int64_t S = x_mod->ne[1];
        const int64_t B = x_mod->ne[2];
        // capacity C = max(1, ceil(cf * S / E)) (per batch item, per expert)
        const int64_t C = std::max<int64_t>(1, (int64_t)std::ceil(capacity_factor * S / (double)E));

        // 1) router: concat(temb, x_unmodulated) -> logits [E, S, B] -> softmax
        auto temb_expanded = ggml_repeat(c, ggml_reshape_3d(c, temb, H, 1, B), x_unmodulated);  // [H, S, B]
        auto router_in     = ggml_concat(c, temb_expanded, x_unmodulated, 0);                   // [2H, S, B]
        auto probs         = ggml_soft_max(c, gate->forward(ctx, router_in));                   // [E, S, B]

        // 2) affinity [S, E, B] and per-expert top-C token selection
        auto affinity = ggml_cont(c, ggml_permute(c, probs, 1, 0, 2, 3));           // [S, E, B]
        auto top_idx  = ggml_cont(c, ggml_argsort_top_k(c, affinity, (int)C));      // I32 [C, E, B], descending

        // 3) scatter the selected affinities into M [1, S, E, B]
        //    (ggml_get_rows batches as a [1, S, E, B] / top_idx [C, E, B])
        auto affinity_rows = ggml_reshape_4d(c, affinity, 1, S, E, B);  // [1, S, E, B]
        auto gating_sel    = ggml_get_rows(c, affinity_rows, top_idx);  // [1, C, E, B]
        auto zeros         = ggml_ext_zeros(ctx, 1, S, E, B);
        auto m             = ggml_set_rows(c, zeros, gating_sel, top_idx);  // view of `zeros` [1, S, E, B]

        // 4) per-token renormalization: denom[t, b] = sum_e M[., t, e, b]; G = M / max(denom, 1e-12) * 2.5
        //    (diffusers: gating_flat / (token_score_sums + 1e-12) * route_scale - DIVISION, not multiplication)
        //    ggml_sum_rows requires unit-stride rows, so materialize the permuted view first.
        auto expert_major = ggml_cont(c, ggml_permute(c, m, 2, 1, 0, 3));  // [E, S, 1, B]
        auto denom        = ggml_sum_rows(c, expert_major);                // [1, S, 1, B]
        denom             = ggml_clamp(c, denom, 1e-12f, INFINITY);
        auto g            = ggml_div(c, m, denom);  // broadcast over E
        if (config.route_scale != 1.f) {
            g = ggml_scale(c, g, config.route_scale);
        }

        // 5) routed computation: one pass per expert over its own C tokens; every (token, expert)
        //    contribution survives (a hot token can be picked by any number of experts)
        auto x_rows = ggml_reshape_4d(c, x_mod, H, S, B, 1);  // row source for batched ggml_get_rows
        ggml_tensor* routed = nullptr;
        for (int64_t e = 0; e < E; ++e) {
            // expert e's token ids, [C, B, 1] (rows along ne[0], batch along ne[1])
            auto ids = ggml_view_3d(c, top_idx, C, B, 1, top_idx->nb[2], top_idx->nb[3], e * top_idx->nb[1]);
            auto xs  = ggml_get_rows(c, x_rows, ids);  // [H, C, B, 1] selected tokens
            // gating weight per (slot, batch): G sliced at expert e as [1, S, B, 1]
            auto g_e = ggml_view_4d(c, g, 1, S, B, 1, g->nb[1], g->nb[3], g->nb[3], e * g->nb[2]);
            auto w   = ggml_get_rows(c, g_e, ids);  // [1, C, B, 1]
            // SwiGLU expert: silu(gate) * up with the gate in the FIRST half (SwiGLUExperts convention)
            auto gate_up = ggml_mul_mat(c, expert_weight(ctx, "experts.gate_up_proj", e, H, inner_dim * 2), xs);
            auto h       = ggml_mul(c,
                                    ggml_silu(c, ggml_ext_slice(ctx, gate_up, 0, 0, inner_dim)),
                                    ggml_ext_slice(ctx, gate_up, 0, inner_dim, inner_dim * 2));  // [I, C, B, 1]
            auto out     = ggml_mul(c, ggml_mul_mat(c, expert_weight(ctx, "experts.down_proj", e, inner_dim, H), h), w);
            // scatter the weighted contribution back onto its token rows
            auto z = ggml_set_rows(c, ggml_ext_zeros(ctx, H, S, B, 1), out, ids);  // [H, S, B, 1]
            routed = routed == nullptr ? z : ggml_add(c, routed, z);
        }

        auto routed_sum = ggml_reshape_3d(c, routed, H, S, B);
        return ggml_add(c, routed_sum, shared_expert->forward(ctx, x_mod));
    }
};

// diffusers NucleusMoEAttnProcessor2_0: image queries attend to concatenated image+text K/V
// (cross-attention style, no text query, no text MLP). GQA 16Q/4KV. Qwen-Image style RoPE
// (interleaved, centered coordinates) applied to img q/k and txt k.
struct NucleusAttention : public GGMLBlock {
    int64_t num_heads;
    int64_t kv_heads;
    int64_t head_dim;

    NucleusAttention(const NucleusImageConfig& config)
        : num_heads(config.num_attention_heads),
          kv_heads(config.num_kv_heads),
          head_dim(config.head_dim) {
        const int64_t inner    = num_heads * head_dim;
        const int64_t inner_kv = kv_heads * head_dim;
        blocks["to_q"]         = std::make_shared<Linear>(config.hidden_size, inner, false);
        blocks["to_k"]         = std::make_shared<Linear>(config.hidden_size, inner_kv, false);
        blocks["to_v"]         = std::make_shared<Linear>(config.hidden_size, inner_kv, false);
        blocks["add_k_proj"]   = std::make_shared<Linear>(config.hidden_size, inner_kv, false);
        blocks["add_v_proj"]   = std::make_shared<Linear>(config.hidden_size, inner_kv, false);
        blocks["norm_q"]       = std::make_shared<RMSNorm>(head_dim, config.norm_eps);
        blocks["norm_k"]       = std::make_shared<RMSNorm>(head_dim, config.norm_eps);
        // Present in the checkpoint; the diffusers processor never applies norm_added_q
        // (text produces no queries) - loaded but unused.
        blocks["norm_added_q"] = std::make_shared<RMSNorm>(head_dim, config.norm_eps);
        blocks["norm_added_k"] = std::make_shared<RMSNorm>(head_dim, config.norm_eps);
        blocks["to_out.0"]     = std::make_shared<Linear>(inner, config.hidden_size, false);
    }

    // pe_full: [2, 2, d/2, Lt + S] with txt tokens first (Rope::gen_qwen_image_pe order);
    // img_pe: [2, 2, d/2, S] slice for the image-only queries.
    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* y, ggml_tensor* pe_full, ggml_tensor* img_pe) {
        auto c           = ctx->ggml_ctx;
        const int64_t S  = x->ne[1];
        const int64_t Lt = y->ne[1];
        const int64_t N  = x->ne[2];

        auto to_q         = std::dynamic_pointer_cast<Linear>(blocks["to_q"]);
        auto to_k         = std::dynamic_pointer_cast<Linear>(blocks["to_k"]);
        auto to_v         = std::dynamic_pointer_cast<Linear>(blocks["to_v"]);
        auto add_k_proj   = std::dynamic_pointer_cast<Linear>(blocks["add_k_proj"]);
        auto add_v_proj   = std::dynamic_pointer_cast<Linear>(blocks["add_v_proj"]);
        auto norm_q       = std::dynamic_pointer_cast<RMSNorm>(blocks["norm_q"]);
        auto norm_k       = std::dynamic_pointer_cast<RMSNorm>(blocks["norm_k"]);
        auto norm_added_k = std::dynamic_pointer_cast<RMSNorm>(blocks["norm_added_k"]);
        auto to_out       = std::dynamic_pointer_cast<Linear>(blocks["to_out.0"]);
        if (sd_backend_is(ctx->backend, "Vulkan") || sd_backend_is(ctx->backend, "ROCm")) {
            to_out->set_force_prec_f32(true);
        }

        // Exact power-of-two down-scale of the attention input: q/k pass through RMSNorm
        // (scale invariant), v is rescaled back exactly after to_v. Keeps the to_q/to_k/to_v
        // matmul sums below the fp16 accumulation ceiling of this stack.
        auto x_scaled = ggml_scale(c, x, 1.f / 64.f);
        auto q = norm_q->forward(ctx, ggml_reshape_4d(c, to_q->forward(ctx, x_scaled), head_dim, num_heads, S, N));
        auto v = ggml_reshape_4d(c, to_v->forward(ctx, x_scaled), head_dim, kv_heads, S, N);
        v      = ggml_scale(c, v, 64.f);

        // Concat K/V BEFORE rope (4D, txt first to match the pe token order), then apply rope with the
        // full pe. Rope::apply_rope returns 3D [d, L, heads*N], so concatenating after rope would need
        // a dim-1 concat of 3D tensors; the concat-first order mirrors the verified qwen_image path.
        auto k_img  = norm_k->forward(ctx, ggml_reshape_4d(c, to_k->forward(ctx, x_scaled), head_dim, kv_heads, S, N));
        auto k_txt  = norm_added_k->forward(ctx, ggml_reshape_4d(c, add_k_proj->forward(ctx, y), head_dim, kv_heads, Lt, N));
        auto k      = ggml_concat(c, k_txt, k_img, 2);  // [d, kv_heads, Lt + S, N]
        auto v_txt  = ggml_reshape_4d(c, add_v_proj->forward(ctx, y), head_dim, kv_heads, Lt, N);
        v           = ggml_concat(c, v_txt, v, 2);      // [d, kv_heads, Lt + S, N]

        q = Rope::apply_rope(ctx, q, img_pe);    // image queries: img positions only
        k = Rope::apply_rope(ctx, k, pe_full);   // txt-first pe order matches the concat order

        // GQA handled by ggml_ext_attention_ext (mul_mat batch broadcast / flash-attn GQA).
        // Nucleus hidden states legitimately reach RMS ~1e3 (stable across layers 4..31), and to_v
        // amplifies further; values beyond ~6.5e4 overflow the F16 casts inside the attention helper.
        // The reference runs in bf16 (fp32 range), so it never sees this. Attention is linear in v,
        // so an exact power-of-two pre-scale of v with a matching post-scale of the projected output
        // keeps the math bit-stable while giving the F16 casts 1024x headroom.
        v = ggml_scale(c, v, 1.f / 1024.f);
        auto attn = ggml_ext_attention_ext(ctx, q, k, v, num_heads, nullptr, true, ctx->flash_attn_enabled, 1.f);
        // to_out input down-scale (exact) + combined post-scale 8 * 1024 * 64 keeps the
        // attention output pipeline bit-exact while staying under the fp16 dot ceiling.
        auto out = to_out->forward(ctx, ggml_scale(c, attn, 1.f / 8.f));
        return ggml_scale(c, out, 8.f * 1024.f);
    }
};

// diffusers NucleusMoEImageTransformerBlock (single image stream + per-block text projection):
// mod = img_mod(silu(temb)) -> (scale1, gate1, scale2, gate2); gates clamped to +-2 and tanh'ed;
// norms are affine-free LayerNorm(eps=1e-6); modulation is scale-only (no shift).
struct NucleusBlock : public GGMLBlock {
    NucleusImageConfig config;
    bool moe;
    float capacity_factor;

    NucleusBlock(const NucleusImageConfig& config_, bool moe_, float capacity_factor_)
        : config(config_),
          moe(moe_),
          capacity_factor(capacity_factor_) {
        blocks["img_mod.1"]    = std::make_shared<Linear>(config.hidden_size, config.hidden_size * 4, true);
        blocks["encoder_proj"] = std::make_shared<Linear>(config.joint_attention_dim, config.hidden_size, true);
        blocks["attn"]         = std::make_shared<NucleusAttention>(config);
        if (moe) {
            blocks["img_mlp"] = std::make_shared<NucleusMoELayer>(config, capacity_factor);
        } else {
            blocks["img_mlp"] = std::make_shared<NucleusFeedForward>(config.hidden_size, config.dense_inner_dim);
        }
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* txt, ggml_tensor* temb, ggml_tensor* pe_full, ggml_tensor* img_pe) {
        auto c        = ctx->ggml_ctx;
        auto img_mod  = std::dynamic_pointer_cast<Linear>(blocks["img_mod.1"]);
        auto attn     = std::dynamic_pointer_cast<NucleusAttention>(blocks["attn"]);
        auto encoder_proj = std::dynamic_pointer_cast<Linear>(blocks["encoder_proj"]);

        // scale1, gate1, scale2, gate2 = chunk(img_mod(silu(temb)), 4); only the gates are clamped to +-2.
        auto mod_params = img_mod->forward(ctx, ggml_silu(c, temb));  // [4H, 1, B]
        auto mods       = ggml_ext_chunk(ctx, mod_params, 4, 0, false);
        auto scale1     = mods[0];
        auto gate1      = ggml_clamp(c, mods[1], -2.f, 2.f);
        auto scale2     = mods[2];
        auto gate2      = ggml_clamp(c, mods[3], -2.f, 2.f);

        auto context = encoder_proj->forward(ctx, txt);  // [H, Lt, B]

        auto normed    = ggml_norm(c, x, config.norm_eps);                 // affine-free LayerNorm
        auto modulated = ggml_add(c, normed, ggml_mul(c, normed, scale1));  // normed * (1 + scale1)
        auto attn_out  = attn->forward(ctx, modulated, context, pe_full, img_pe);
        x = ggml_add(c, x, ggml_mul(c, attn_out, ggml_tanh(c, gate1)));

        auto normed2    = ggml_norm(c, x, config.norm_eps);
        auto modulated2 = ggml_add(c, normed2, ggml_mul(c, normed2, scale2));
        ggml_tensor* mlp_out;
        if (moe) {
            mlp_out = std::dynamic_pointer_cast<NucleusMoELayer>(blocks["img_mlp"])->forward(ctx, modulated2, normed2, temb);
        } else {
            mlp_out = std::dynamic_pointer_cast<NucleusFeedForward>(blocks["img_mlp"])->forward(ctx, modulated2);
        }
        return ggml_add(c, x, ggml_mul(c, mlp_out, ggml_tanh(c, gate2)));
    }
};

// diffusers NucleusMoETimestepProjEmbeddings.
struct NucleusTimeTextEmbed : public GGMLBlock {
    int64_t freq_dim;

    NucleusTimeTextEmbed(int64_t embedding_dim, int64_t out_dim)
        : freq_dim(embedding_dim) {
        // Timesteps(num_channels=embedding_dim, flip_sin_to_cos=True, scale=1000) ->
        // TimestepEmbedding(in=embedding_dim, time_embed_dim=4*embedding_dim, out=out_dim) -> RMSNorm
        blocks["timestep_embedder.linear_1"] = std::make_shared<Linear>(embedding_dim, embedding_dim * 4, true);
        blocks["timestep_embedder.linear_2"] = std::make_shared<Linear>(embedding_dim * 4, out_dim, true);
        blocks["norm"]                       = std::make_shared<RMSNorm>(out_dim, 1e-6f);
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* timestep) {
        auto c = ctx->ggml_ctx;
        // time_factor 1.0: the sd.cpp flow sampler already feeds t = sigma * 1000 (denoiser
        // sigma_to_t), which matches diffusers' total of raw_sigma * Timesteps(scale=1000).
        // Qwen-Image uses the same 1.0 factor for the same reason.
        auto t  = ggml_ext_timestep_embedding(ctx, timestep, (int)freq_dim, 10000, 1.f);
        auto l1 = std::dynamic_pointer_cast<Linear>(blocks["timestep_embedder.linear_1"]);
        auto l2 = std::dynamic_pointer_cast<Linear>(blocks["timestep_embedder.linear_2"]);
        auto n  = std::dynamic_pointer_cast<RMSNorm>(blocks["norm"]);
        t       = l1->forward(ctx, t);
        t       = ggml_silu(c, t);
        t       = l2->forward(ctx, t);
        return n->forward(ctx, t);
    }
};

struct NucleusImageModel : public GGMLBlock {
    NucleusImageConfig config;

    NucleusImageModel(const NucleusImageConfig& config_)
        : config(config_) {
        const int64_t H = config.hidden_size;
        blocks["txt_norm"] = std::make_shared<RMSNorm>(config.joint_attention_dim, 1e-6f);
        blocks["time_text_embed"] = std::make_shared<NucleusTimeTextEmbed>(H, H);
        blocks["img_in"]            = std::make_shared<Linear>(config.in_channels, H, true);
        for (int64_t i = 0; i < config.depth; ++i) {
            const bool moe = config.is_moe_layer((int)i);
            blocks["transformer_blocks." + std::to_string(i)] =
                std::make_shared<NucleusBlock>(config, moe, config.capacity_factor_for((int)i));
        }
        // AdaLayerNormContinuous(elementwise_affine=False, eps=1e-6): emb = linear(silu(temb));
        // scale, shift = chunk(2); x = LN(x) * (1 + scale) + shift
        blocks["norm_out.linear"] = std::make_shared<Linear>(H, H * 2, true);
        blocks["proj_out"]        = std::make_shared<Linear>(H, config.out_channels * config.patch_size * config.patch_size, false);
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* context, ggml_tensor* timestep, ggml_tensor* pe) {
        auto c          = ctx->ggml_ctx;
        const int64_t H = config.hidden_size;
        // x is the raw latent [W, H, C, N]; patch token count S must use the same rounding as
        // Rope::gen_qwen_image_ids ((dim + patch/2) / patch), NOT x->ne[0]/x->ne[1] (latents, not tokens).
        const int64_t patch = config.patch_size;
        const int64_t h_len = (x->ne[1] + (patch / 2)) / patch;
        const int64_t w_len = (x->ne[0] + (patch / 2)) / patch;
        const int64_t S  = h_len * w_len;        // img tokens
        const int64_t Lt = pe->ne[3] - S;        // txt tokens (pe covers txt + img, txt first)

        // A/B RESOLVED: diffusers _pack_latents (pipeline_nucleusmoe_image.py) is channel-major
        // (permute(0, 2, 4, 1, 3, 5) over (B, h, w, C, ph, pw)) => patch_last=true. The earlier
        // false (patch-major) pair silently permuted the img_in/proj_out feature dims.
        auto img = std::dynamic_pointer_cast<Linear>(blocks["img_in"])->forward(
            ctx, DiT::pad_and_patchify(ctx, x, config.patch_size, config.patch_size, true));  // [H, S, N]

        auto txt = std::dynamic_pointer_cast<RMSNorm>(blocks["txt_norm"])->forward(ctx, context);  // [4096, Lt, N]

        auto temb = std::dynamic_pointer_cast<NucleusTimeTextEmbed>(blocks["time_text_embed"])->forward(ctx, timestep);
        temb      = ggml_reshape_3d(c, temb, H, 1, temb->ne[1]);  // [H, 1, N]

        // pe layout: [2, 2, axes/2, Lt + S] with txt tokens first (gen_qwen_image_pe order).
        // K/V are concatenated txt-first and roped with the full pe; queries use the img slice.
        auto img_pe = ggml_ext_slice(ctx, pe, 3, Lt, Lt + S);

        for (int64_t i = 0; i < config.depth; ++i) {
            img = std::dynamic_pointer_cast<NucleusBlock>(blocks["transformer_blocks." + std::to_string(i)])
                      ->forward(ctx, img, txt, temb, pe, img_pe);
            sd::ggml_graph_cut::mark_graph_cut(img, "nucleus_image.transformer_blocks." + std::to_string(i), "img");
        }

        // AdaLayerNormContinuous
        auto emb   = std::dynamic_pointer_cast<Linear>(blocks["norm_out.linear"])->forward(ctx, ggml_silu(c, temb));
        auto mods  = ggml_ext_chunk(ctx, emb, 2, 0, false);
        auto scale = ggml_reshape_3d(c, mods[0], H, 1, emb->ne[2]);
        auto shift = ggml_reshape_3d(c, mods[1], H, 1, emb->ne[2]);
        img        = ggml_norm(c, img, config.norm_eps);
        // AdaLayerNormContinuous: x = LN(x) * (1 + scale) + shift  (diffusers; cf. Flux::modulate)
        img        = ggml_add(c, ggml_add(c, ggml_mul(c, img, scale), img), shift);

        img = std::dynamic_pointer_cast<Linear>(blocks["proj_out"])->forward(ctx, img);  // [4*out_ch, S, N]

        // diffusers NucleusMoE pipeline negates the transformer output before the scheduler
        // step (noise_pred = -noise_pred); fold the negation into the model output.
        auto out = DiT::unpatchify_and_crop(ctx, img, x->ne[1], x->ne[0], config.patch_size, config.patch_size, true);
        return ggml_scale(c, out, -1.f);
    }
};

struct NucleusImageRunner : public DiffusionModelRunner {
    NucleusImageConfig config;
    NucleusImageModel model;
    std::vector<float> pe_vec;
    bool params_sanitized = false;

    NucleusImageRunner(ggml_backend_t backend,
                       const String2TensorStorage& weights,
                       const std::string& prefix,
                       std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
        : DiffusionModelRunner(backend, prefix, weight_manager),
          config(NucleusImageConfig::detect_from_weights(weights, prefix)),
          model(config) {
        model.init(params_ctx, weights, prefix);
    }

    std::string get_desc() override { return "Nucleus-Image"; }

    // The released stduhpf Q5_1 checkpoint carries corrupted tensor data (e.g. all five
    // transformer_blocks.9 img_mod/img_mlp tensors have 3-12% non-finite/huge quant scales,
    // plus scattered bad blocks in layers 21/30/31). Any non-finite weight poisons the whole
    // denoiser output (uniform-noise images). Zero the bad blocks/elements once at load so the
    // model runs until a clean re-download; disable with NUCLEUS_NO_SANITIZE.
    // Clean tensors in this checkpoint have |quant scale| <= 3.6 and |weight| <= ~1.3, so the
    // thresholds below never touch healthy data.
    static void sanitize_weight_bytes(uint8_t* data, size_t nbytes, ggml_type type, int64_t& patched_blocks, int64_t& patched_elems) {
        if (type == GGML_TYPE_F32) {
            float* v = reinterpret_cast<float*>(data);
            for (size_t i = 0; i < nbytes / sizeof(float); ++i) {
                if (!std::isfinite(v[i]) || std::fabs(v[i]) > 100.f) {
                    v[i] = 0.f;
                    ++patched_elems;
                }
            }
        } else if (type == GGML_TYPE_F16 || type == GGML_TYPE_BF16) {
            uint16_t* v = reinterpret_cast<uint16_t*>(data);
            for (size_t i = 0; i < nbytes / 2; ++i) {
                float x;
                if (type == GGML_TYPE_F16) {
                    x = ggml_fp16_to_fp32(*reinterpret_cast<ggml_fp16_t*>(&v[i]));
                } else {
                    uint32_t u = static_cast<uint32_t>(v[i]) << 16;
                    std::memcpy(&x, &u, sizeof(x));
                }
                if (!std::isfinite(x) || std::fabs(x) > 100.f) {
                    v[i] = 0;
                    ++patched_elems;
                }
            }
        } else if (type == GGML_TYPE_Q8_0) {
            // block: d f16, qs 32B (34 bytes). d = 0 zeroes the block's 32 values.
            for (size_t off = 0; off + 34 <= nbytes; off += 34) {
                float d = ggml_fp16_to_fp32(*reinterpret_cast<ggml_fp16_t*>(data + off));
                if (!std::isfinite(d) || std::fabs(d) > 4.f) {
                    std::memset(data + off, 0, 2);
                    ++patched_blocks;
                }
            }
        } else if (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q5_0) {
            // Q4_0: d f16 + qs 16B (18B); Q5_0: d f16 + qh 4B + qs 16B (22B).
            const size_t block_size = type == GGML_TYPE_Q4_0 ? 18 : 22;
            for (size_t off = 0; off + block_size <= nbytes; off += block_size) {
                float d = ggml_fp16_to_fp32(*reinterpret_cast<ggml_fp16_t*>(data + off));
                if (!std::isfinite(d) || std::fabs(d) > 4.f) {
                    std::memset(data + off, 0, 2);
                    ++patched_blocks;
                }
            }
        } else if (type == GGML_TYPE_Q4_1) {
            // Q4_1: d f16, m f16, qs 16B (20B).
            for (size_t off = 0; off + 20 <= nbytes; off += 20) {
                float d = ggml_fp16_to_fp32(*reinterpret_cast<ggml_fp16_t*>(data + off));
                float m = ggml_fp16_to_fp32(*reinterpret_cast<ggml_fp16_t*>(data + off + 2));
                if (!std::isfinite(d) || !std::isfinite(m) || std::fabs(d) > 4.f || std::fabs(m) > 4.f) {
                    std::memset(data + off, 0, 4);
                    ++patched_blocks;
                }
            }
        } else if (type == GGML_TYPE_Q5_1) {
            // block: d f16, m f16, qh 4B, qs 16B. d = m = 0 zeroes the block's 32 values.
            for (size_t off = 0; off + 24 <= nbytes; off += 24) {
                float d = ggml_fp16_to_fp32(*reinterpret_cast<ggml_fp16_t*>(data + off));
                float m = ggml_fp16_to_fp32(*reinterpret_cast<ggml_fp16_t*>(data + off + 2));
                if (!std::isfinite(d) || !std::isfinite(m) || std::fabs(d) > 4.f || std::fabs(m) > 4.f) {
                    std::memset(data + off, 0, 4);
                    ++patched_blocks;
                }
            }
        }
    }

    void ensure_params_sanitized() {
        if (params_sanitized) {
            return;
        }
        params_sanitized = true;
        if (std::getenv("NUCLEUS_NO_SANITIZE") != nullptr) {
            return;
        }
        std::map<std::string, ggml_tensor*> tensors;
        model.get_param_tensors(tensors, prefix);
        int64_t patched_blocks = 0;
        int64_t patched_elems  = 0;
        int64_t scanned        = 0;
        int64_t skipped_nobuf  = 0;
        int64_t skipped_type   = 0;
        for (const auto& [name, tensor] : tensors) {
            if (tensor == nullptr) {
                continue;
            }
            if (tensor->buffer == nullptr) {
                ++skipped_nobuf;
                continue;
            }
            const ggml_type type   = tensor->type;
            const size_t  nbytes   = ggml_nbytes(tensor);
            if (type != GGML_TYPE_F32 && type != GGML_TYPE_F16 && type != GGML_TYPE_BF16 &&
                type != GGML_TYPE_Q5_1 && type != GGML_TYPE_Q8_0 && type != GGML_TYPE_Q5_0 &&
                type != GGML_TYPE_Q4_0 && type != GGML_TYPE_Q4_1) {
                ++skipped_type;
                continue;
            }
            ++scanned;
            std::vector<uint8_t> buf(nbytes);
            ggml_backend_tensor_get(tensor, buf.data(), 0, nbytes);
            const int64_t before_blocks = patched_blocks;
            const int64_t before_elems  = patched_elems;
            sanitize_weight_bytes(buf.data(), nbytes, type, patched_blocks, patched_elems);
            if (patched_blocks != before_blocks || patched_elems != before_elems) {
                ggml_backend_tensor_set(tensor, buf.data(), 0, nbytes);
                LOG_WARN("Nucleus-Image: sanitized corrupt tensor '%s' (%lld blocks, %lld elements zeroed)",
                         name.c_str(), (long long)(patched_blocks - before_blocks),
                         (long long)(patched_elems - before_elems));
            }
        }
        LOG_VERBOSE("Nucleus-Image: sanitize pass scanned %lld tensors (skipped: %lld no-buffer, %lld unsupported-type), "
                    "patched %lld blocks + %lld elements",
                    (long long)scanned, (long long)skipped_nobuf, (long long)skipped_type,
                    (long long)patched_blocks, (long long)patched_elems);
        if (patched_blocks > 0 || patched_elems > 0) {
            LOG_WARN("Nucleus-Image: checkpoint corruption repaired: %lld quant blocks + %lld elements zeroed in total",
                     (long long)patched_blocks, (long long)patched_elems);
        }
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix) override {
        model.get_param_tensors(tensors, prefix);
    }

    sd::Tensor<float> compute(int n_threads, const DiffusionParams& inputs_) override {
        DiffusionParams inputs = inputs_;
        if (!inputs.x || !inputs.timesteps || !inputs.context || inputs.context->empty()) {
            LOG_ERROR("Nucleus-Image requires latents, timesteps and text features");
            return {};
        }
        ensure_params_sanitized();
        auto graph = [&]() {
            auto gf   = new_graph_custom(131072);
            auto x    = make_input(*inputs.x);
            auto text = make_input(*inputs.context);
            int64_t h = x->ne[1], w = x->ne[0];
            int64_t nt = text->ne[1];

            // Qwen-Image style RoPE (theta=10000, axes [16, 56, 56], scale_rope centered coords,
            // text positions offset by max(h_len, w_len)/2) - identical to NucleusMoEEmbedRope.
            // Positions are identical across the batch, so pe is generated for bs=1 and broadcast:
            // the txt/img pe slices then stay valid whether CFG batches both passes (B=2) or runs
            // them separately (B=1). ggml_rope_apply broadcasts pe over heads x batch.
            auto embedding = Rope::gen_qwen_image_pe(1, (int)h, (int)w, config.patch_size,
                                                     1, (int)nt, {},
                                                     Rope::RefIndexMode::FIXED, config.theta, config.axes_dim);
            pe_vec         = finish_rope_pe(std::move(embedding));
            int pos_len    = static_cast<int>(pe_vec.size() / config.axes_dim_sum / 2);
            auto pe        = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, config.axes_dim_sum / 2, pos_len);
            set_backend_tensor_data(pe, pe_vec.data());

            auto timesteps = make_input(*inputs.timesteps);
            auto ctx2      = get_context();
            ggml_tensor* out = model.forward(&ctx2, x, text, timesteps, pe);
            ggml_build_forward_expand(gf, out);
            // Nucleus trunk activations legitimately run at RMS 100-600 (trained operating
            // point; the diffusers reference runs bf16). Vulkan/CUDA quantized matmuls
            // accumulate in F16 by default, which overflows (inf -> NaN) at these magnitudes.
            // Force F32 accumulation on every matmul in this model's graphs.
            for (int ni = 0; ni < ggml_graph_n_nodes(gf); ++ni) {
                ggml_tensor* node = ggml_graph_node(gf, ni);
                if (node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID) {
                    ggml_prec_set_acc(node, GGML_PREC_F32);
                }
            }
            return gf;
        };
        return restore_trailing_singleton_dims(GGMLRunner::compute(graph, n_threads, false, false),
                                               inputs.x->dim());
    }
};

}  // namespace NucleusImage

#endif  // __SD_MODEL_DIFFUSION_NUCLEUS_IMAGE_HPP__
