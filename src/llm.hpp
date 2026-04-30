#ifndef __LLM_HPP__
#define __LLM_HPP__

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "ggml_extend.hpp"
#include "json.hpp"
#include "rope.hpp"
#include "tokenizers/bpe_tokenizer.h"
#include "tokenizers/mistral_tokenizer.h"
#include "tokenizers/qwen2_tokenizer.h"

// Debug tap: when non-null, Gemma layer-0 forward paths push intermediate
// tensors here (tagged "DBG:<name>"). Definition lives as `inline` to keep
// this file header-only. Set from LLMRunner::compute_all_hidden_states when
// the SD_DUMP_LAYER0 env var is present.
inline std::vector<ggml_tensor*>* g_layer0_taps = nullptr;

// Helper: preserve a tap's value by routing the graph THROUGH a ggml_cont
// copy. Returning the cont'd tensor (instead of the original) means the
// next op in the graph consumes the cont, so the allocator has to keep the
// cont's buffer live. Mathematically a bitwise copy — no graph change.
// The cont's name starts with "DBG:<name>" so the dumper can find it.
inline ggml_tensor* tap_tensor(ggml_context* ctx, ggml_tensor* t, const char* name) {
    if (::g_layer0_taps == nullptr) return t;
    ggml_tensor* keep = ggml_cont(ctx, t);
    ggml_set_output(keep);  // tell allocator: don't reuse my buffer
    ggml_set_name(keep, (std::string("DBG:") + name).c_str());
    ::g_layer0_taps->push_back(keep);
    return keep;
}

namespace LLM {
    // Bumped aggressively for the 22B LTX-2 smoke test where Gemma 3 12B runs with
    // compute_all_hidden_states (49-layer concat stack over 48 layers of sandwich-
    // norm + attn + MLP). The assert at ggml.c:6877 fired at 40960; 200000 leaves
    // ample headroom while we diagnose whether real op count or hash dedup is the
    // issue.
    constexpr int LLM_GRAPH_SIZE = 200000;

    enum class LLMArch {
        QWEN2_5_VL,
        QWEN3,
        MISTRAL_SMALL_3_2,
        MINISTRAL_3_3B,
        GEMMA3,
        ARCH_COUNT,
    };

    static const char* llm_arch_to_str[] = {
        "qwen2.5vl",
        "qwen3",
        "mistral_small3.2",
        "ministral3.3b",
        "gemma3",
    };

    struct LLMVisionParams {
        int num_layers                      = 32;
        int64_t hidden_size                 = 1280;
        int64_t intermediate_size           = 3420;
        int num_heads                       = 16;
        int64_t in_channels                 = 3;
        int64_t out_hidden_size             = 3584;
        int temporal_patch_size             = 2;
        int patch_size                      = 14;
        int spatial_merge_size              = 2;
        int window_size                     = 112;
        std::set<int> fullatt_block_indexes = {7, 15, 23, 31};
    };

    struct LLMParams {
        LLMArch arch              = LLMArch::QWEN2_5_VL;
        int64_t num_layers        = 28;
        int64_t hidden_size       = 3584;
        int64_t intermediate_size = 18944;
        int num_heads             = 28;
        int num_kv_heads          = 4;
        int head_dim              = 128;
        bool qkv_bias             = true;
        bool qk_norm              = false;
        int64_t vocab_size        = 152064;
        float rms_norm_eps        = 1e-06f;

        // Gemma 3 additions (unused by other archs).
        // Pattern: layers where (idx % sliding_window_pattern == 0) use global attention
        // with rope_theta_global; other layers use sliding-window attention of size
        // sliding_window with rope_theta_local. has_post_norms adds a second RMSNorm after
        // attn and after MLP inside each block. embed_scale multiplies token embeddings
        // once before the first layer.
        int sliding_window            = 0;      // 0 = disabled
        int sliding_window_pattern    = 0;      // 0 = disabled
        float rope_theta_global       = 0.f;    // 0 = use legacy hardcoded theta
        float rope_theta_local        = 0.f;
        // Gemma 3 rope_scaling: linear RoPE scaling applied only to full-attention
        // (global) layers. HuggingFace config.json: rope_scaling={factor: F, rope_type: linear}.
        // Sliding layers are unscaled. 1.0 = disabled. For the 12B model this is 8.0.
        float rope_scaling_factor_global = 1.0f;
        bool has_post_norms           = false;
        float embed_scale             = 1.0f;

        // When true, Linear layers inside this model force GGML_PREC_F32 on
        // their mul_mat ops. ggml-cuda defaults to F16 accumulation for
        // quantized matmul, which drifts ~2% per layer vs the CPU/F32 path.
        // For Gemma 3 used as a fixed embedding encoder (LTX-2) the compound
        // drift across 48 layers corrupts the final embedding to uselessness
        // on CUDA. Set true for Gemma 3; leave false for generative LLMs
        // where the drift is acceptable and speed matters more.
        bool force_matmul_prec_f32    = false;

        LLMVisionParams vision;
    };

    // Gemma 3 RMSNorm variant: scale by (1 + w) instead of w. The PyTorch original
    // Gemma3RMSNorm stores weights centered around 0 (so init scale is 1.0), and the
    // forward applies `x * (1 + w)`. This class implements that math.
    //
    // IMPORTANT: this is currently UNUSED for production Gemma3, because our only
    // supported Gemma3 source is GGUF. llama.cpp's `convert_hf_to_gguf.py`
    // (`Gemma3Model.norm_shift`) bakes the +1 INTO the weights at convert time
    // (so `w_gguf = w_pytorch + 1`), letting llama.cpp's runtime use the simpler
    // `x * w` form. We therefore consume those GGUF weights with plain `RMSNorm`
    // and the +1 is implicit in the weight values. If a non-GGUF Gemma3 loader
    // is ever added, swap to this class for those code paths.
    class RMSNormPlus1 : public UnaryBlock {
    protected:
        int64_t hidden_size;
        float eps;
        std::string prefix;

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, std::string prefix = "") override {
            this->prefix     = prefix;
            params["weight"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
        }

    public:
        RMSNormPlus1(int64_t hidden_size, float eps = 1e-06f)
            : hidden_size(hidden_size), eps(eps) {}

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            ggml_tensor* w = params["weight"];
            if (ctx->weight_adapter) {
                w = ctx->weight_adapter->patch_weight(ctx->ggml_ctx, w, prefix + "weight");
            }
            x           = ggml_rms_norm(ctx->ggml_ctx, x, eps);
            auto scaled = ggml_mul(ctx->ggml_ctx, x, w);     // rms(x) * w
            x           = ggml_add_inplace(ctx->ggml_ctx, x, scaled);  // rms(x) * (1 + w)
            return x;
        }
    };

    struct MLP : public GGMLBlock {
    protected:
        bool use_gelu_tanh;

    public:
        MLP(int64_t hidden_size, int64_t intermediate_size, bool bias = false,
            bool use_gelu_tanh = false, bool force_prec_f32 = false)
            : use_gelu_tanh(use_gelu_tanh) {
            blocks["gate_proj"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, intermediate_size, bias, /*force_f32=*/false, force_prec_f32));
            blocks["up_proj"]   = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, intermediate_size, bias, /*force_f32=*/false, force_prec_f32));
            blocks["down_proj"] = std::shared_ptr<GGMLBlock>(new Linear(intermediate_size, hidden_size, bias, /*force_f32=*/false, force_prec_f32));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            // x: [N, n_token, hidden_size]
            auto gate_proj = std::dynamic_pointer_cast<Linear>(blocks["gate_proj"]);
            auto up_proj   = std::dynamic_pointer_cast<Linear>(blocks["up_proj"]);
            auto down_proj = std::dynamic_pointer_cast<Linear>(blocks["down_proj"]);

            auto h = gate_proj->forward(ctx, x);
            if (use_gelu_tanh) {
                h = ggml_gelu_inplace(ctx->ggml_ctx, h);
            } else {
                h = ggml_silu_inplace(ctx->ggml_ctx, h);
            }
            h = ggml_mul_inplace(ctx->ggml_ctx, h, up_proj->forward(ctx, x));
            h = down_proj->forward(ctx, h);
            return h;
        }
    };

    struct VisionPatchEmbed : public GGMLBlock {
    protected:
        bool llama_cpp_style;
        int patch_size;
        int temporal_patch_size;
        int64_t in_channels;
        int64_t embed_dim;

    public:
        VisionPatchEmbed(bool llama_cpp_style,
                         int patch_size          = 14,
                         int temporal_patch_size = 2,
                         int64_t in_channels     = 3,
                         int64_t embed_dim       = 1152)
            : llama_cpp_style(llama_cpp_style),
              patch_size(patch_size),
              temporal_patch_size(temporal_patch_size),
              in_channels(in_channels),
              embed_dim(embed_dim) {
            if (llama_cpp_style) {
                blocks["proj.0"] = std::shared_ptr<GGMLBlock>(new Conv2d(in_channels,
                                                                         embed_dim,
                                                                         {patch_size, patch_size},
                                                                         {patch_size, patch_size},  // stride
                                                                         {0, 0},                    // padding
                                                                         {1, 1},                    // dilation
                                                                         false));
                blocks["proj.1"] = std::shared_ptr<GGMLBlock>(new Conv2d(in_channels,
                                                                         embed_dim,
                                                                         {patch_size, patch_size},
                                                                         {patch_size, patch_size},  // stride
                                                                         {0, 0},                    // padding
                                                                         {1, 1},                    // dilation
                                                                         false));
            } else {
                std::tuple<int, int, int> kernel_size = {(int)temporal_patch_size, (int)patch_size, (int)patch_size};
                blocks["proj"]                        = std::shared_ptr<GGMLBlock>(new Conv3d(in_channels,
                                                                                              embed_dim,
                                                                                              kernel_size,
                                                                                              kernel_size,  // stride
                                                                                              {0, 0, 0},    // padding
                                                                                              {1, 1, 1},    // dilation
                                                                                              false));
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            // x: [N*grid_t*grid_h*grid_w, in_channels, temporal_patch_size*patch_size*patch_size]
            // return: [N*grid_t*grid_h*grid_w, embed_dim]
            x = ggml_reshape_4d(ctx->ggml_ctx,
                                x,
                                patch_size,
                                patch_size,
                                temporal_patch_size,
                                ggml_nelements(x) / (temporal_patch_size * patch_size * patch_size));

            if (llama_cpp_style) {
                auto proj_0 = std::dynamic_pointer_cast<Conv2d>(blocks["proj.0"]);
                auto proj_1 = std::dynamic_pointer_cast<Conv2d>(blocks["proj.1"]);

                auto x0 = ggml_ext_slice(ctx->ggml_ctx, x, 2, 0, 1);
                x0      = ggml_reshape_4d(ctx->ggml_ctx, x0, x0->ne[0], x0->ne[1], in_channels, x0->ne[3] / in_channels);
                x0      = proj_0->forward(ctx, x0);

                auto x1 = ggml_ext_slice(ctx->ggml_ctx, x, 2, 1, 2);
                x1      = ggml_reshape_4d(ctx->ggml_ctx, x1, x1->ne[0], x1->ne[1], in_channels, x1->ne[3] / in_channels);
                x1      = proj_1->forward(ctx, x1);

                x = ggml_add(ctx->ggml_ctx, x0, x1);
            } else {
                auto proj = std::dynamic_pointer_cast<Conv3d>(blocks["proj"]);

                x = proj->forward(ctx, x);
            }

            x = ggml_reshape_2d(ctx->ggml_ctx, x, embed_dim, ggml_nelements(x) / embed_dim);
            return x;
        }
    };

    struct PatchMerger : public GGMLBlock {
    protected:
        int64_t hidden_size;

    public:
        PatchMerger(int64_t dim,
                    int64_t context_dim,
                    int64_t spatial_merge_size) {
            hidden_size     = context_dim * spatial_merge_size * spatial_merge_size;
            blocks["ln_q"]  = std::shared_ptr<GGMLBlock>(new RMSNorm(context_dim, 1e-6f));
            blocks["mlp.0"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, hidden_size));
            // mlp.1 is nn.GELU()
            blocks["mlp.2"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, dim));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto ln_q  = std::dynamic_pointer_cast<RMSNorm>(blocks["ln_q"]);
            auto mlp_0 = std::dynamic_pointer_cast<Linear>(blocks["mlp.0"]);
            auto mlp_2 = std::dynamic_pointer_cast<Linear>(blocks["mlp.2"]);

            x = ln_q->forward(ctx, x);
            x = ggml_reshape_2d(ctx->ggml_ctx, x, hidden_size, ggml_nelements(x) / hidden_size);
            x = mlp_0->forward(ctx, x);
            x = ggml_ext_gelu(ctx->ggml_ctx, x);
            x = mlp_2->forward(ctx, x);
            return x;
        }
    };

    struct VisionAttention : public GGMLBlock {
    protected:
        bool llama_cpp_style;
        int head_dim;
        int num_heads;

    public:
        VisionAttention(bool llama_cpp_style,
                        int64_t hidden_size,
                        int num_heads)
            : llama_cpp_style(llama_cpp_style), num_heads(num_heads) {
            head_dim = static_cast<int>(hidden_size / num_heads);
            GGML_ASSERT(num_heads * head_dim == hidden_size);
            if (llama_cpp_style) {
                blocks["q_proj"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, hidden_size));
                blocks["k_proj"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, hidden_size));
                blocks["v_proj"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, hidden_size));
            } else {
                blocks["qkv"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, hidden_size * 3));
            }
            blocks["proj"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, hidden_size));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* pe,
                             ggml_tensor* mask = nullptr) {
            // x: [N, n_token, hidden_size]
            int64_t n_token = x->ne[1];
            int64_t N       = x->ne[2];
            auto proj       = std::dynamic_pointer_cast<Linear>(blocks["proj"]);

            std::vector<ggml_tensor*> qkv_vec;
            if (llama_cpp_style) {
                auto q_proj = std::dynamic_pointer_cast<Linear>(blocks["q_proj"]);
                auto k_proj = std::dynamic_pointer_cast<Linear>(blocks["k_proj"]);
                auto v_proj = std::dynamic_pointer_cast<Linear>(blocks["v_proj"]);

                auto q = q_proj->forward(ctx, x);
                auto k = k_proj->forward(ctx, x);
                auto v = v_proj->forward(ctx, x);

                qkv_vec = {q, k, v};
            } else {
                auto qkv_proj = std::dynamic_pointer_cast<Linear>(blocks["qkv"]);
                auto qkv      = qkv_proj->forward(ctx, x);
                qkv_vec       = split_qkv(ctx->ggml_ctx, qkv);
            }

            auto q = ggml_reshape_4d(ctx->ggml_ctx, qkv_vec[0], head_dim, num_heads, qkv_vec[0]->ne[1], qkv_vec[0]->ne[2]);  // [N, n_token, n_head, d_head]
            auto k = ggml_reshape_4d(ctx->ggml_ctx, qkv_vec[1], head_dim, num_heads, qkv_vec[1]->ne[1], qkv_vec[1]->ne[2]);  // [N, n_token, n_head, d_head]
            auto v = ggml_reshape_4d(ctx->ggml_ctx, qkv_vec[2], head_dim, num_heads, qkv_vec[2]->ne[1], qkv_vec[2]->ne[2]);  // [N, n_token, n_head, d_head]

            x = Rope::attention(ctx, q, k, v, pe, mask, 1.f, false);  // [N, n_token, hidden_size]

            x = proj->forward(ctx, x);  // [N, n_token, hidden_size]
            return x;
        }
    };

    struct VisionBlock : public GGMLBlock {
    public:
        VisionBlock(bool llama_cpp_style,
                    int64_t hidden_size,
                    int64_t intermediate_size,
                    int num_heads,
                    float eps = 1e-6f) {
            blocks["attn"]  = std::shared_ptr<GGMLBlock>(new VisionAttention(llama_cpp_style, hidden_size, num_heads));
            blocks["mlp"]   = std::shared_ptr<GGMLBlock>(new MLP(hidden_size, intermediate_size, true));
            blocks["norm1"] = std::shared_ptr<GGMLBlock>(new RMSNorm(hidden_size, eps));
            blocks["norm2"] = std::shared_ptr<GGMLBlock>(new RMSNorm(hidden_size, eps));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* pe,
                             ggml_tensor* mask = nullptr) {
            // x: [N, n_token, hidden_size]
            auto attn  = std::dynamic_pointer_cast<VisionAttention>(blocks["attn"]);
            auto mlp   = std::dynamic_pointer_cast<MLP>(blocks["mlp"]);
            auto norm1 = std::dynamic_pointer_cast<RMSNorm>(blocks["norm1"]);
            auto norm2 = std::dynamic_pointer_cast<RMSNorm>(blocks["norm2"]);

            auto residual = x;
            x             = norm1->forward(ctx, x);
            x             = attn->forward(ctx, x, pe, mask);
            x             = ggml_add_inplace(ctx->ggml_ctx, x, residual);

            residual = x;
            x        = norm2->forward(ctx, x);
            x        = mlp->forward(ctx, x);
            x        = ggml_add_inplace(ctx->ggml_ctx, x, residual);

            return x;
        }
    };

    struct VisionModel : public GGMLBlock {
    protected:
        int num_layers;
        int spatial_merge_size;
        std::set<int> fullatt_block_indexes;

    public:
        VisionModel(bool llama_cpp_style,
                    int num_layers,
                    int64_t in_channels,
                    int64_t hidden_size,
                    int64_t out_hidden_size,
                    int64_t intermediate_size,
                    int num_heads,
                    int spatial_merge_size,
                    int patch_size,
                    int temporal_patch_size,
                    int window_size,
                    std::set<int> fullatt_block_indexes = {7, 15, 23, 31},
                    float eps                           = 1e-6f)
            : num_layers(num_layers), fullatt_block_indexes(std::move(fullatt_block_indexes)), spatial_merge_size(spatial_merge_size) {
            blocks["patch_embed"] = std::shared_ptr<GGMLBlock>(new VisionPatchEmbed(llama_cpp_style,
                                                                                    patch_size,
                                                                                    temporal_patch_size,
                                                                                    in_channels,
                                                                                    hidden_size));
            for (int i = 0; i < num_layers; i++) {
                blocks["blocks." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new VisionBlock(llama_cpp_style,
                                                                                                   hidden_size,
                                                                                                   intermediate_size,
                                                                                                   num_heads,
                                                                                                   eps));
            }
            blocks["merger"] = std::shared_ptr<GGMLBlock>(new PatchMerger(out_hidden_size, hidden_size, spatial_merge_size));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* pixel_values,
                             ggml_tensor* pe,
                             ggml_tensor* window_index,
                             ggml_tensor* window_inverse_index,
                             ggml_tensor* window_mask) {
            // pixel_values: [grid_t*(H/mh/ph)*(W/mw/pw)*mh*mw, C*pt*ph*pw]
            // window_index: [grid_t*(H/mh/ph)*(W/mw/pw)]
            // window_inverse_index: [grid_t*(H/mh/ph)*(W/mw/pw)]
            // window_mask: [grid_h*grid_w, grid_h*grid_w]
            auto patch_embed = std::dynamic_pointer_cast<VisionPatchEmbed>(blocks["patch_embed"]);
            auto merger      = std::dynamic_pointer_cast<PatchMerger>(blocks["merger"]);

            auto x = patch_embed->forward(ctx, pixel_values);

            x = ggml_reshape_4d(ctx->ggml_ctx, x, x->ne[0] * spatial_merge_size * spatial_merge_size, x->ne[1] / spatial_merge_size / spatial_merge_size, x->ne[2], x->ne[3]);
            x = ggml_get_rows(ctx->ggml_ctx, x, window_index);
            x = ggml_reshape_4d(ctx->ggml_ctx, x, x->ne[0] / spatial_merge_size / spatial_merge_size, x->ne[1] * spatial_merge_size * spatial_merge_size, x->ne[2], x->ne[3]);

            for (int i = 0; i < num_layers; i++) {
                auto block = std::dynamic_pointer_cast<VisionBlock>(blocks["blocks." + std::to_string(i)]);

                auto mask = window_mask;
                if (fullatt_block_indexes.find(i) != fullatt_block_indexes.end()) {
                    mask = nullptr;
                }
                x = block->forward(ctx, x, pe, mask);
            }

            x = merger->forward(ctx, x);

            x = ggml_get_rows(ctx->ggml_ctx, x, window_inverse_index);

            return x;
        }
    };

    struct Attention : public GGMLBlock {
    protected:
        LLMArch arch;
        int head_dim;
        int64_t num_heads;
        int64_t num_kv_heads;
        bool qk_norm;
        int layer_idx;
        int sliding_window_pattern;
        float rope_theta_global;
        float rope_theta_local;
        float rope_scaling_factor_global;

    public:
        Attention(const LLMParams& params, int layer_idx = 0)
            : arch(params.arch),
              num_heads(params.num_heads),
              num_kv_heads(params.num_kv_heads),
              head_dim(params.head_dim),
              qk_norm(params.qk_norm),
              layer_idx(layer_idx),
              sliding_window_pattern(params.sliding_window_pattern),
              rope_theta_global(params.rope_theta_global),
              rope_theta_local(params.rope_theta_local),
              rope_scaling_factor_global(params.rope_scaling_factor_global) {
            const bool fp = params.force_matmul_prec_f32;
            blocks["q_proj"] = std::make_shared<Linear>(params.hidden_size, num_heads    * head_dim, params.qkv_bias, /*force_f32=*/false, fp);
            blocks["k_proj"] = std::make_shared<Linear>(params.hidden_size, num_kv_heads * head_dim, params.qkv_bias, /*force_f32=*/false, fp);
            blocks["v_proj"] = std::make_shared<Linear>(params.hidden_size, num_kv_heads * head_dim, params.qkv_bias, /*force_f32=*/false, fp);
            blocks["o_proj"] = std::make_shared<Linear>(num_heads * head_dim, params.hidden_size,                 false, /*force_f32=*/false, fp);
            if (params.qk_norm) {
                // Gemma3 GGUF: weights have +1 baked in (see `RMSNormPlus1` comment),
                // so plain `RMSNorm` produces `x * w_gguf == x * (w_pytorch + 1)` which
                // matches the PyTorch reference's `x * (1 + w_pytorch)`.
                blocks["q_norm"] = std::make_shared<RMSNorm>(head_dim, params.rms_norm_eps);
                blocks["k_norm"] = std::make_shared<RMSNorm>(head_dim, params.rms_norm_eps);
            }
        }

        bool is_gemma_sliding_layer() const {
            return arch == LLMArch::GEMMA3
                   && sliding_window_pattern > 0
                   && ((layer_idx + 1) % sliding_window_pattern) != 0;
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* input_pos,
                             ggml_tensor* attention_mask         = nullptr,
                             ggml_tensor* attention_mask_sliding = nullptr) {
            // x: [N, n_token, hidden_size]
            int64_t n_token = x->ne[1];
            int64_t N       = x->ne[2];
            auto q_proj     = std::dynamic_pointer_cast<Linear>(blocks["q_proj"]);
            auto k_proj     = std::dynamic_pointer_cast<Linear>(blocks["k_proj"]);
            auto v_proj     = std::dynamic_pointer_cast<Linear>(blocks["v_proj"]);
            auto out_proj   = std::dynamic_pointer_cast<Linear>(blocks["o_proj"]);

            const bool trace = (layer_idx == 0);
            auto tag = [&](ggml_tensor* t, const char* name) {
                return trace ? tap_tensor(ctx->ggml_ctx, t, name) : t;
            };

            auto q = tag(q_proj->forward(ctx, x), "q_proj");
            auto k = tag(k_proj->forward(ctx, x), "k_proj");
            auto v = tag(v_proj->forward(ctx, x), "v_proj");

            q = ggml_reshape_4d(ctx->ggml_ctx, q, head_dim, num_heads, n_token, N);     // [N, n_token, num_heads, head_dim]
            k = ggml_reshape_4d(ctx->ggml_ctx, k, head_dim, num_kv_heads, n_token, N);  // [N, n_token, num_kv_heads, head_dim]
            v = ggml_reshape_4d(ctx->ggml_ctx, v, head_dim, num_kv_heads, n_token, N);  // [N, n_token, num_kv_heads, head_dim]

            if (qk_norm) {
                auto q_norm = std::dynamic_pointer_cast<UnaryBlock>(blocks["q_norm"]);
                auto k_norm = std::dynamic_pointer_cast<UnaryBlock>(blocks["k_norm"]);

                q = tag(q_norm->forward(ctx, q), "q_norm");
                k = tag(k_norm->forward(ctx, k), "k_norm");
            }

            if (arch == LLMArch::MISTRAL_SMALL_3_2) {
                q = ggml_rope_ext(ctx->ggml_ctx, q, input_pos, nullptr, 128, GGML_ROPE_TYPE_NORMAL, 8192, 1000000000.f, 1.f, 0.f, 1.f, 32.f, 1.f);
                k = ggml_rope_ext(ctx->ggml_ctx, k, input_pos, nullptr, 128, GGML_ROPE_TYPE_NORMAL, 8192, 1000000000.f, 1.f, 0.f, 1.f, 32.f, 1.f);
            } else if (arch == LLMArch::MINISTRAL_3_3B) {
                q = ggml_rope_ext(ctx->ggml_ctx, q, input_pos, nullptr, 128, GGML_ROPE_TYPE_NEOX, 262144, 1000000.f, 1.f, 0.f, 1.f, 32.f, 1.f);
                k = ggml_rope_ext(ctx->ggml_ctx, k, input_pos, nullptr, 128, GGML_ROPE_TYPE_NEOX, 262144, 1000000.f, 1.f, 0.f, 1.f, 32.f, 1.f);
            } else if (arch == LLMArch::QWEN3) {
                q = ggml_rope_ext(ctx->ggml_ctx, q, input_pos, nullptr, 128, GGML_ROPE_TYPE_NEOX, 40960, 1000000.f, 1.f, 0.f, 1.f, 32.f, 1.f);
                k = ggml_rope_ext(ctx->ggml_ctx, k, input_pos, nullptr, 128, GGML_ROPE_TYPE_NEOX, 40960, 1000000.f, 1.f, 0.f, 1.f, 32.f, 1.f);
            } else if (arch == LLMArch::GEMMA3) {
                // Per-layer theta: global (full attention) layers use rope_theta_global,
                // sliding layers use rope_theta_local. Pattern: is_global = ((l+1)%p == 0).
                // Real Gemma 3 12B config also sets linear rope_scaling with factor=8.0
                // on full_attention only. HuggingFace divides inv_freq by factor, which
                // ggml_rope_ext expresses as freq_scale = 1 / factor.
                bool is_sliding = is_gemma_sliding_layer();
                float theta      = is_sliding ? rope_theta_local : rope_theta_global;
                float freq_scale = is_sliding ? 1.0f : (1.0f / rope_scaling_factor_global);
                q = tag(ggml_rope_ext(ctx->ggml_ctx, q, input_pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 1024, theta, freq_scale, 0.f, 1.f, 32.f, 1.f), "q_rope");
                k = tag(ggml_rope_ext(ctx->ggml_ctx, k, input_pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 1024, theta, freq_scale, 0.f, 1.f, 32.f, 1.f), "k_rope");
            } else {
                int sections[4] = {16, 24, 24, 0};
                q               = ggml_rope_multi(ctx->ggml_ctx, q, input_pos, nullptr, head_dim, sections, GGML_ROPE_TYPE_MROPE, 128000, 1000000.f, 1.f, 0.f, 1.f, 32.f, 1.f);
                k               = ggml_rope_multi(ctx->ggml_ctx, k, input_pos, nullptr, head_dim, sections, GGML_ROPE_TYPE_MROPE, 128000, 1000000.f, 1.f, 0.f, 1.f, 32.f, 1.f);
            }

            // Gemma 3: pick the sliding-window mask for local layers.
            if (is_gemma_sliding_layer() && attention_mask_sliding != nullptr) {
                attention_mask = attention_mask_sliding;
            }

            q = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, q, 0, 2, 1, 3));  // [N, num_heads, n_token, head_dim]
            q = ggml_reshape_3d(ctx->ggml_ctx, q, q->ne[0], q->ne[1], q->ne[2] * q->ne[3]);      // [N*num_heads, n_token, head_dim]

            k = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, k, 0, 2, 1, 3));  // [N, num_kv_heads, n_token, head_dim]
            k = ggml_reshape_3d(ctx->ggml_ctx, k, k->ne[0], k->ne[1], k->ne[2] * k->ne[3]);      // [N*num_kv_heads, n_token, head_dim]

            x = tag(ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v, num_heads, attention_mask, true, false), "attn_out");  // [N, n_token, hidden_size]

            x = tag(out_proj->forward(ctx, x), "o_proj");  // [N, n_token, hidden_size]
            return x;
        }
    };

    struct TransformerBlock : public GGMLBlock {
    protected:
        bool has_post_norms;
        int  layer_idx;

    public:
        TransformerBlock(const LLMParams& params, int layer_idx = 0)
            : has_post_norms(params.has_post_norms), layer_idx(layer_idx) {
            bool gemma               = (params.arch == LLMArch::GEMMA3);
            blocks["self_attn"]      = std::make_shared<Attention>(params, layer_idx);
            blocks["mlp"]            = std::make_shared<MLP>(params.hidden_size, params.intermediate_size, false, gemma, params.force_matmul_prec_f32);

            if (gemma) {
                // GGUF Gemma3: weights have +1 baked in by llama.cpp's convert script,
                // so plain `RMSNorm` is the right form. See `RMSNormPlus1` class comment.
                blocks["input_layernorm"]             = std::make_shared<RMSNorm>(params.hidden_size, params.rms_norm_eps);
                blocks["post_attention_layernorm"]    = std::make_shared<RMSNorm>(params.hidden_size, params.rms_norm_eps);
                blocks["pre_feedforward_layernorm"]   = std::make_shared<RMSNorm>(params.hidden_size, params.rms_norm_eps);
                blocks["post_feedforward_layernorm"]  = std::make_shared<RMSNorm>(params.hidden_size, params.rms_norm_eps);
            } else {
                blocks["input_layernorm"]          = std::make_shared<RMSNorm>(params.hidden_size, params.rms_norm_eps);
                blocks["post_attention_layernorm"] = std::make_shared<RMSNorm>(params.hidden_size, params.rms_norm_eps);
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* input_pos,
                             ggml_tensor* attention_mask         = nullptr,
                             ggml_tensor* attention_mask_sliding = nullptr) {
            // x: [N, n_token, hidden_size]
            auto self_attn = std::dynamic_pointer_cast<Attention>(blocks["self_attn"]);
            auto mlp       = std::dynamic_pointer_cast<MLP>(blocks["mlp"]);

            if (has_post_norms) {
                // Gemma 3 sandwich: pre-attn-norm → attn → post-attn-norm → +res
                //                   → pre-ff-norm  → mlp  → post-ff-norm  → +res.
                auto input_ln        = std::dynamic_pointer_cast<UnaryBlock>(blocks["input_layernorm"]);
                auto post_attn_ln    = std::dynamic_pointer_cast<UnaryBlock>(blocks["post_attention_layernorm"]);
                auto pre_ff_ln       = std::dynamic_pointer_cast<UnaryBlock>(blocks["pre_feedforward_layernorm"]);
                auto post_ff_ln      = std::dynamic_pointer_cast<UnaryBlock>(blocks["post_feedforward_layernorm"]);

                auto residual = x;
                const bool trace_block = (layer_idx == 0);
                auto tag = [&](ggml_tensor* t, const char* name) {
                    return trace_block ? tap_tensor(ctx->ggml_ctx, t, name) : t;
                };
                if (trace_block) {
                    x = tag(x, "x_embed_in");
                    residual = x;  // residual must match the post-tap tensor
                }

                x = tag(input_ln->forward(ctx, x), "input_ln");
                x = self_attn->forward(ctx, x, input_pos, attention_mask, attention_mask_sliding);
                x = tag(post_attn_ln->forward(ctx, x), "post_attn_ln");
                x = tag(ggml_add_inplace(ctx->ggml_ctx, x, residual), "after_attn_res");

                residual = x;
                x = tag(pre_ff_ln->forward(ctx, x), "pre_ff_ln");
                x = tag(mlp->forward(ctx, x), "mlp_out");
                x = tag(post_ff_ln->forward(ctx, x), "post_ff_ln");
                x = tag(ggml_add_inplace(ctx->ggml_ctx, x, residual), "after_ff_res");
                return x;
            }

            auto input_layernorm          = std::dynamic_pointer_cast<RMSNorm>(blocks["input_layernorm"]);
            auto post_attention_layernorm = std::dynamic_pointer_cast<RMSNorm>(blocks["post_attention_layernorm"]);

            auto residual = x;
            x             = input_layernorm->forward(ctx, x);
            x             = self_attn->forward(ctx, x, input_pos, attention_mask);
            x             = ggml_add_inplace(ctx->ggml_ctx, x, residual);

            residual = x;
            x        = post_attention_layernorm->forward(ctx, x);
            x        = mlp->forward(ctx, x);
            x        = ggml_add_inplace(ctx->ggml_ctx, x, residual);

            return x;
        }
    };

    struct TextModel : public GGMLBlock {
    protected:
        int64_t num_layers;
        float embed_scale;
        bool has_post_norms;

    public:
        TextModel(const LLMParams& params)
            : num_layers(params.num_layers),
              embed_scale(params.embed_scale),
              has_post_norms(params.has_post_norms) {
            blocks["embed_tokens"] = std::shared_ptr<GGMLBlock>(new Embedding(params.vocab_size, params.hidden_size));
            for (int i = 0; i < num_layers; i++) {
                blocks["layers." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new TransformerBlock(params, i));
            }
            // GGUF Gemma3 norm weights have +1 baked in (per llama.cpp convert), so plain
            // RMSNorm is correct for both Gemma3 and other archs.
            blocks["norm"] = std::shared_ptr<GGMLBlock>(new RMSNorm(params.hidden_size, params.rms_norm_eps));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* input_ids,
                             ggml_tensor* input_pos,
                             ggml_tensor* attention_mask,
                             std::vector<std::pair<int, ggml_tensor*>> image_embeds,
                             std::set<int> out_layers,
                             ggml_tensor* attention_mask_sliding = nullptr,
                             std::vector<ggml_tensor*>* all_hidden_states = nullptr) {
            // input_ids: [N, n_token]
            // return: [N, n_token, hidden_size]

            auto embed_tokens = std::dynamic_pointer_cast<Embedding>(blocks["embed_tokens"]);
            auto norm         = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm"]);

            auto x = embed_tokens->forward(ctx, input_ids);
            x = tap_tensor(ctx->ggml_ctx, x, "embed_raw");
            if (embed_scale != 1.0f) {
                x = ggml_scale(ctx->ggml_ctx, x, embed_scale);
                x = tap_tensor(ctx->ggml_ctx, x, "embed_scaled");
            }
            if (all_hidden_states) {
                all_hidden_states->push_back(x);
            }

            std::vector<ggml_tensor*> intermediate_outputs;

            if (image_embeds.size() > 0) {
                GGML_ASSERT(x->ne[2] == 1);  // N == 1

                auto raw_x              = ggml_cast(ctx->ggml_ctx, x, image_embeds[0].second->type);
                int64_t txt_token_start = 0;
                int64_t txt_token_end   = 0;

                ggml_tensor* input_embed = nullptr;

                for (int i = 0; i < image_embeds.size(); i++) {
                    if (i == 0) {
                        txt_token_start = 0;
                    } else {
                        txt_token_start = image_embeds[i - 1].first + image_embeds[i - 1].second->ne[1];
                    }
                    txt_token_end = image_embeds[i].first;

                    auto txt_embed = ggml_ext_slice(ctx->ggml_ctx, raw_x, 1, txt_token_start, txt_token_end);
                    if (input_embed == nullptr) {
                        input_embed = txt_embed;
                    } else {
                        input_embed = ggml_concat(ctx->ggml_ctx, input_embed, txt_embed, 1);
                    }

                    auto image_embed = image_embeds[i].second;
                    input_embed      = ggml_concat(ctx->ggml_ctx, input_embed, image_embed, 1);
                }

                txt_token_start = image_embeds[image_embeds.size() - 1].first + image_embeds[image_embeds.size() - 1].second->ne[1];
                txt_token_end   = raw_x->ne[1];

                auto final_txt_embed = ggml_ext_slice(ctx->ggml_ctx, raw_x, 1, txt_token_start, txt_token_end);

                input_embed = ggml_concat(ctx->ggml_ctx, input_embed, final_txt_embed, 1);
                GGML_ASSERT(raw_x->ne[1] == input_embed->ne[1]);

                x = input_embed;
            }

            for (int i = 0; i < num_layers; i++) {
                auto block = std::dynamic_pointer_cast<TransformerBlock>(blocks["layers." + std::to_string(i)]);

                x = block->forward(ctx, x, input_pos, attention_mask, attention_mask_sliding);
                if (all_hidden_states) {
                    all_hidden_states->push_back(x);
                }
                if (out_layers.find(i + 1) != out_layers.end()) {
                    intermediate_outputs.push_back(x);
                }
            }

            if (!intermediate_outputs.empty()) {
                x = intermediate_outputs[0];
                for (int i = 1; i < intermediate_outputs.size(); i++) {
                    x = ggml_concat(ctx->ggml_ctx, x, intermediate_outputs[i], 0);
                }
            } else {
                x = norm->forward(ctx, x);
            }
            // HF Gemma 3 (and most HF causal-LM models): hidden_states[-1] is the
            // POST-final-norm state. Replace the last pre-norm entry we stored with
            // the normed version so downstream stacking matches exactly.
            if (all_hidden_states && !all_hidden_states->empty()) {
                all_hidden_states->back() = x;
            }
            return x;
        }
    };

    struct LLM : public GGMLBlock {
        bool enable_vision;
        LLMParams params;

    public:
        LLM() = default;
        LLM(LLMParams params, bool enable_vision = false, bool llama_cpp_style = false)
            : enable_vision(enable_vision), params(params) {
            blocks["model"] = std::shared_ptr<GGMLBlock>(new TextModel(params));
            if (enable_vision) {
                blocks["visual"] = std::shared_ptr<GGMLBlock>(new VisionModel(llama_cpp_style,
                                                                              params.vision.num_layers,
                                                                              params.vision.in_channels,
                                                                              params.vision.hidden_size,
                                                                              params.vision.out_hidden_size,
                                                                              params.vision.intermediate_size,
                                                                              params.vision.num_heads,
                                                                              params.vision.spatial_merge_size,
                                                                              params.vision.patch_size,
                                                                              params.vision.temporal_patch_size,
                                                                              params.vision.window_size,
                                                                              params.vision.fullatt_block_indexes));
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* input_ids,
                             ggml_tensor* input_pos,
                             ggml_tensor* attention_mask,
                             std::vector<std::pair<int, ggml_tensor*>> image_embeds,
                             std::set<int> out_layers,
                             ggml_tensor* attention_mask_sliding = nullptr,
                             std::vector<ggml_tensor*>* all_hidden_states = nullptr) {
            // input_ids: [N, n_token]
            auto model = std::dynamic_pointer_cast<TextModel>(blocks["model"]);

            auto x = model->forward(ctx, input_ids, input_pos, attention_mask, image_embeds, out_layers,
                                    attention_mask_sliding, all_hidden_states);
            return x;
        }

        ggml_tensor* vision_forward(GGMLRunnerContext* ctx,
                                    ggml_tensor* pixel_values,
                                    ggml_tensor* pe,
                                    ggml_tensor* window_index,
                                    ggml_tensor* window_inverse_index,
                                    ggml_tensor* window_mask) {
            GGML_ASSERT(enable_vision);
            auto vision_model = std::dynamic_pointer_cast<VisionModel>(blocks["visual"]);
            return vision_model->forward(ctx, pixel_values, pe, window_index, window_inverse_index, window_mask);
        }
    };

    struct LLMRunner : public GGMLRunner {
        LLMParams params;
        bool enable_vision;
        LLM model;

        std::vector<int> input_pos_vec;
        std::vector<float> attention_mask_vec;
        std::vector<float> window_mask_vec;
        std::vector<int> window_index_vec;
        std::vector<int> window_inverse_index_vec;
        std::vector<float> pe_vec;

        LLMRunner(LLMArch arch,
                  ggml_backend_t backend,
                  bool offload_params_to_cpu,
                  const String2TensorStorage& tensor_storage_map,
                  const std::string prefix,
                  bool enable_vision_ = false)
            : GGMLRunner(backend, offload_params_to_cpu), enable_vision(enable_vision_) {
            params.arch = arch;
            if (arch == LLMArch::MISTRAL_SMALL_3_2 || arch == LLMArch::MINISTRAL_3_3B) {
                params.head_dim     = 128;
                params.num_heads    = 32;
                params.num_kv_heads = 8;
                params.qkv_bias     = false;
                params.rms_norm_eps = 1e-5f;
            } else if (arch == LLMArch::QWEN3) {
                params.head_dim     = 128;
                params.num_heads    = 32;
                params.num_kv_heads = 8;
                params.qkv_bias     = false;
                params.qk_norm      = true;
                params.rms_norm_eps = 1e-6f;
            } else if (arch == LLMArch::GEMMA3) {
                // Gemma 3 12B (LTX-2 text encoder). See memory file
                // .opencode/memories/2026-04-22_1000_gemma3-delta-note.md for derivation.
                params.head_dim              = 256;
                params.num_heads             = 16;
                params.num_kv_heads          = 8;
                params.qkv_bias              = false;
                params.qk_norm               = true;
                params.rms_norm_eps          = 1e-6f;
                params.sliding_window        = 1024;
                params.sliding_window_pattern = 6;
                params.rope_theta_global     = 1000000.f;
                params.rope_theta_local      = 10000.f;
                // Real Gemma 3 12B config.json sets rope_scaling={factor: 8.0,
                // rope_type: linear} on full_attention layers.  HuggingFace divides
                // inv_freq by factor, which corresponds to ggml_rope_ext freq_scale
                // = 1/factor.  Sliding-attention layers stay unscaled.
                params.rope_scaling_factor_global = 8.f;
                params.has_post_norms        = true;
                // Gemma 3 has narrow weight scales; the CUDA mmvq/mmq kernels
                // quantize activations to q8_1 (block-32 fp16 scale) while the
                // CPU iq4_xs kernel uses q8_K (block-256 fp32 scale). That
                // format mismatch causes ~5% per-layer drift and ruins the
                // embedding. Requesting GGML_PREC_F32 routes matmul through
                // cuBLAS dequant+GEMM, which matches CPU bit-for-bit. Even with
                // Q8_0 weights this disables TF32 to keep prompt fidelity —
                // without it the cumulative reduction-order drift across 48
                // layers shifts subject identity (cat → person on beach).
                params.force_matmul_prec_f32 = true;
                // embed_scale is sqrt(hidden_size); hidden_size is autodetected below,
                // so defer setting embed_scale until after the tensor-storage scan.
            }
            bool have_vision_weight = false;
            bool llama_cpp_style    = false;
            params.num_layers       = 0;
            for (auto pair : tensor_storage_map) {
                std::string tensor_name = pair.first;
                // Use prefix-boundary match (must be followed by '.') rather than bare
                // substring: otherwise e.g. prefix "text_encoder" would also match
                // "text_encoder_deep.*" tensors and inflate auto-detected num_layers.
                if (tensor_name.rfind(prefix + ".", 0) != 0)
                    continue;
                size_t pos = tensor_name.find("visual.");
                if (pos != std::string::npos) {
                    have_vision_weight = true;
                    if (contains(tensor_name, "attn.q_proj")) {
                        llama_cpp_style = true;
                    }
                    continue;
                }
                pos = tensor_name.find("layers.");
                if (pos != std::string::npos) {
                    tensor_name = tensor_name.substr(pos);  // remove prefix
                    auto items  = split_string(tensor_name, '.');
                    if (items.size() > 1) {
                        int block_index = atoi(items[1].c_str());
                        if (block_index + 1 > params.num_layers) {
                            params.num_layers = block_index + 1;
                        }
                    }
                }
                if (contains(tensor_name, "embed_tokens.weight")) {
                    params.hidden_size = pair.second.ne[0];
                    params.vocab_size  = pair.second.ne[1];
                }
                if (contains(tensor_name, "layers.0.mlp.gate_proj.weight")) {
                    params.intermediate_size = pair.second.ne[1];
                }
                if (arch == LLMArch::GEMMA3) {
                    // Gemma 3 has configurable head_dim (256 for 12B, 32 in our tiny test).
                    // q_norm.weight has shape [head_dim]; q_proj.weight is [hidden_size, num_heads*head_dim]
                    // and stored in GGML with ne[1]=num_heads*head_dim; likewise k_proj gives num_kv_heads.
                    if (contains(tensor_name, "layers.0.self_attn.q_norm.weight")) {
                        params.head_dim = (int)pair.second.ne[0];
                    }
                }
            }
            if (arch == LLMArch::QWEN3 && params.num_layers == 28) {  // Qwen3 2B
                params.num_heads = 16;
            }
            if (arch == LLMArch::GEMMA3) {
                // Second pass: derive num_heads / num_kv_heads once head_dim is known.
                for (auto pair : tensor_storage_map) {
                    std::string tn = pair.first;
                    if (tn.rfind(prefix + ".", 0) != 0) continue;
                    if (contains(tn, "layers.0.self_attn.q_proj.weight") && params.head_dim > 0) {
                        params.num_heads = (int)(pair.second.ne[1] / params.head_dim);
                    }
                    if (contains(tn, "layers.0.self_attn.k_proj.weight") && params.head_dim > 0) {
                        params.num_kv_heads = (int)(pair.second.ne[1] / params.head_dim);
                    }
                }
                params.embed_scale = sqrtf((float)params.hidden_size);
            }
            LOG_DEBUG("llm: num_layers = %" PRId64 ", vocab_size = %" PRId64 ", hidden_size = %" PRId64 ", intermediate_size = %" PRId64,
                      params.num_layers,
                      params.vocab_size,
                      params.hidden_size,
                      params.intermediate_size);
            if (enable_vision && !have_vision_weight) {
                LOG_WARN("no vision weights detected, vision disabled");
                enable_vision = false;
            }
            if (enable_vision) {
                LOG_DEBUG("enable llm vision");
                if (llama_cpp_style) {
                    LOG_DEBUG("llama.cpp style vision weight");
                }
            }
            model = LLM(params, enable_vision, llama_cpp_style);
            model.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return llm_arch_to_str[static_cast<int>(params.arch)];
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string prefix) {
            model.get_param_tensors(tensors, prefix);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* input_ids,
                             ggml_tensor* input_pos,
                             ggml_tensor* attention_mask,
                             std::vector<std::pair<int, ggml_tensor*>> image_embeds,
                             std::set<int> out_layers,
                             ggml_tensor* attention_mask_sliding     = nullptr,
                             std::vector<ggml_tensor*>* all_hidden_states = nullptr) {
            auto hidden_states = model.forward(ctx, input_ids, input_pos, attention_mask, image_embeds, out_layers,
                                               attention_mask_sliding, all_hidden_states);  // [N, n_token, hidden_size]
            return hidden_states;
        }

        ggml_tensor* vision_forward(GGMLRunnerContext* ctx,
                                    ggml_tensor* pixel_values,
                                    ggml_tensor* input_pos,
                                    ggml_tensor* window_index,
                                    ggml_tensor* window_inverse_index,
                                    ggml_tensor* window_mask) {
            auto hidden_states = model.vision_forward(ctx, pixel_values, input_pos, window_index, window_inverse_index, window_mask);
            return hidden_states;
        }

        // Scratch storage for the Gemma sliding-window mask.
        std::vector<float> sliding_attention_mask_vec;

        ggml_cgraph* build_graph(const sd::Tensor<int32_t>& input_ids_tensor,
                                 const sd::Tensor<float>& attention_mask_tensor,
                                 const std::vector<std::pair<int, sd::Tensor<float>>>& image_embeds_tensor,
                                 std::set<int> out_layers,
                                 std::vector<ggml_tensor*>* all_hidden_states = nullptr,
                                 int pad_count = 0) {
            ggml_cgraph* gf        = new_graph_custom(LLM_GRAPH_SIZE);
            ggml_tensor* input_ids = make_input(input_ids_tensor);
            std::vector<std::pair<int, ggml_tensor*>> image_embeds;
            image_embeds.reserve(image_embeds_tensor.size());
            for (const auto& [idx, embed_tensor] : image_embeds_tensor) {
                ggml_tensor* embed = make_input(embed_tensor);
                image_embeds.emplace_back(idx, embed);
            }

            int64_t n_tokens = input_ids->ne[0];
            if (params.arch == LLMArch::MISTRAL_SMALL_3_2 || params.arch == LLMArch::MINISTRAL_3_3B || params.arch == LLMArch::QWEN3 || params.arch == LLMArch::GEMMA3) {
                input_pos_vec.resize(n_tokens);
                for (int i = 0; i < n_tokens; ++i) {
                    input_pos_vec[i] = i;
                }
            } else {
                input_pos_vec.resize(n_tokens * 4);
                for (int i = 0; i < n_tokens; ++i) {
                    input_pos_vec[i]                = i;
                    input_pos_vec[n_tokens + i]     = i;
                    input_pos_vec[2 * n_tokens + i] = i;
                    input_pos_vec[3 * n_tokens + i] = 0;
                }
            }

            auto input_pos = ggml_new_tensor_1d(compute_ctx,
                                                GGML_TYPE_I32,
                                                input_pos_vec.size());
            set_backend_tensor_data(input_pos, input_pos_vec.data());

            ggml_tensor* attention_mask = nullptr;
            if (!attention_mask_tensor.empty()) {
                attention_mask = make_input(attention_mask_tensor);
            } else {
                // Causal AND (when pad_count > 0) "real query cannot attend to a pad key"
                // — required when the input is left-padded (pad tokens occupy [0, pad_count)).
                // Pad-as-query rows still attend causally to earlier pads so softmax stays
                // finite; pad outputs are discarded downstream.
                attention_mask_vec.resize(n_tokens * n_tokens);
                for (int i0 = 0; i0 < n_tokens; i0++) {       // i0 = key
                    for (int i1 = 0; i1 < n_tokens; i1++) {   // i1 = query
                        float value = 0.f;
                        if (i0 > i1) {
                            value = -INFINITY;
                        }
                        if (pad_count > 0 && i0 < pad_count && i1 >= pad_count) {
                            value = -INFINITY;
                        }
                        attention_mask_vec[i1 * n_tokens + i0] = value;
                    }
                }
                attention_mask = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, n_tokens, n_tokens);
                set_backend_tensor_data(attention_mask, attention_mask_vec.data());
            }

            // Gemma 3 sliding-window mask: causal AND (q - k < window_size), with the
            // same pad-as-key restriction applied so real queries don't pick up pad keys
            // even within the sliding window.
            ggml_tensor* attention_mask_sliding = nullptr;
            if (params.arch == LLMArch::GEMMA3 && params.sliding_window > 0) {
                sliding_attention_mask_vec.resize(n_tokens * n_tokens);
                for (int q = 0; q < n_tokens; q++) {
                    for (int k = 0; k < n_tokens; k++) {
                        float value = 0.f;
                        if (k > q || (q - k) >= params.sliding_window) {
                            value = -INFINITY;
                        }
                        if (pad_count > 0 && k < pad_count && q >= pad_count) {
                            value = -INFINITY;
                        }
                        sliding_attention_mask_vec[q * n_tokens + k] = value;
                    }
                }
                attention_mask_sliding = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, n_tokens, n_tokens);
                set_backend_tensor_data(attention_mask_sliding, sliding_attention_mask_vec.data());
            }

            auto runner_ctx = get_context();

            ggml_tensor* hidden_states = forward(&runner_ctx, input_ids, input_pos, attention_mask, image_embeds, out_layers,
                                                 attention_mask_sliding, all_hidden_states);

            ggml_build_forward_expand(gf, hidden_states);

            return gf;
        }

        sd::Tensor<float> compute(const int n_threads,
                                  const sd::Tensor<int32_t>& input_ids,
                                  const sd::Tensor<float>& attention_mask,
                                  const std::vector<std::pair<int, sd::Tensor<float>>>& image_embeds,
                                  std::set<int> out_layers) {
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(input_ids, attention_mask, image_embeds, out_layers);
            };
            return take_or_empty(GGMLRunner::compute<float>(get_graph, n_threads, true));
        }

        // Returns all N+1 hidden states (embedding + each transformer layer, with
        // final layer's output post-model.norm). Stacked along a new innermost axis,
        // shape in GGML: ne=[num_layers+1, hidden_size, n_tokens, batch] which matches
        // PyTorch `torch.stack(hidden_states, dim=-1)` layout of [B, T, H, N+1].
        sd::Tensor<float> compute_all_hidden_states(const int n_threads,
                                                    const sd::Tensor<int32_t>& input_ids,
                                                    const sd::Tensor<float>& attention_mask,
                                                    int pad_count = 0) {
            // Debug hook: capture layer-0 intermediates via the global tap vector.
            // Forward paths push tensors here when ::g_layer0_taps != nullptr.
            std::vector<ggml_tensor*> taps;
            const char* dump_dir = std::getenv("SD_DUMP_LAYER0");
            if (dump_dir != nullptr) ::g_layer0_taps = &taps;
            struct TapGuard {
                ~TapGuard() { ::g_layer0_taps = nullptr; }
            } guard;

            auto get_graph = [&]() -> ggml_cgraph* {
                // GGMLRunner::compute calls this lambda TWICE — once to measure
                // the allocator, then reset_compute_ctx wipes all tensors and
                // it's called again to actually compute. Both builds need to
                // re-fire attn_tap so the second build's tensors get our names
                // (otherwise ggml auto-names them "node_X" since the first
                // build's set_name applied to since-dead tensors).
                ::g_attn_tap_count = 0;
                taps.clear();  // also clear the OUTER taps so we only collect
                               // pointers from the latest (compute-pass) build
                std::vector<ggml_tensor*> hidden_states;
                ggml_cgraph* gf = build_graph(input_ids, attention_mask, {}, {}, &hidden_states, pad_count);

                // Keep taps alive through the allocator: mark each as an output
                // (prevents buffer aliasing) and expand into the graph.
                for (auto* t : taps) {
                    ggml_set_output(t);
                    ggml_build_forward_expand(gf, t);
                }

                GGML_ASSERT(!hidden_states.empty());
                // Reshape each [H, T, B] -> [1, H, T, B] so we can concat along axis 0.
                ggml_tensor* stacked = nullptr;
                for (auto* h : hidden_states) {
                    auto h_cont = ggml_cont(compute_ctx, h);
                    auto h_4d   = ggml_reshape_4d(compute_ctx, h_cont, 1, h_cont->ne[0], h_cont->ne[1], h_cont->ne[2]);
                    if (stacked == nullptr) {
                        stacked = h_4d;
                    } else {
                        stacked = ggml_concat(compute_ctx, stacked, h_4d, 0);
                    }
                }
                ggml_build_forward_expand(gf, stacked);
                return gf;
            };
            auto result = take_or_empty(GGMLRunner::compute<float>(get_graph, n_threads, /*free_compute_buffer_immediately=*/false));

            if (dump_dir != nullptr && !taps.empty()) {
                LOG_INFO("SD_DUMP_LAYER0: dumping %zu tensors to %s/", taps.size(), dump_dir);
                for (auto* t : taps) {
                    const char* full_name = ggml_get_name(t);
                    if (std::strncmp(full_name, "DBG:", 4) != 0) continue;
                    const char* name = full_name + 4;
                    size_t nbytes    = ggml_nbytes(t);
                    std::vector<uint8_t> buf(nbytes);
                    ggml_backend_tensor_get(t, buf.data(), 0, nbytes);
                    std::string path = std::string(dump_dir) + "/" + name + ".bin";
                    FILE* f = std::fopen(path.c_str(), "wb");
                    if (f) {
                        std::fwrite(buf.data(), 1, nbytes, f);
                        std::fclose(f);
                        LOG_INFO("  %-22s ne=[%ld,%ld,%ld,%ld] type=%s bytes=%zu -> %s",
                                 name, (long)t->ne[0], (long)t->ne[1], (long)t->ne[2], (long)t->ne[3],
                                 ggml_type_name(t->type), nbytes, path.c_str());
                    }
                }
                // Free now so we don't leak the compute buffer.
                free_compute_buffer();
            }

            return result;
        }

        int64_t get_num_image_tokens(int64_t t, int64_t h, int64_t w) {
            int64_t grid_t     = 1;
            int64_t grid_h     = h / params.vision.patch_size;
            int64_t grid_w     = w / params.vision.patch_size;
            int64_t llm_grid_h = grid_h / params.vision.spatial_merge_size;
            int64_t llm_grid_w = grid_w / params.vision.spatial_merge_size;
            return grid_t * grid_h * grid_w;
        }

        ggml_tensor* process_image(ggml_context* ctx, ggml_tensor* image) {
            // image: [C, H, W]
            // return: [grid_t*(H/mh/ph)*(W/mw/pw)*mh*mw, C*pt*ph*pw], grid_t == 1
            int64_t C  = image->ne[2];
            int64_t H  = image->ne[1];
            int64_t W  = image->ne[0];
            int64_t mh = params.vision.spatial_merge_size;
            int64_t mw = params.vision.spatial_merge_size;
            int64_t pt = params.vision.temporal_patch_size;
            int64_t ph = params.vision.patch_size;
            int64_t pw = params.vision.patch_size;

            image = ggml_reshape_4d(ctx, image, pw, mw, (W / mw / pw), H * C);                               // [C*H, (W/mw/pw), mw, pw]
            image = ggml_cont(ctx, ggml_ext_torch_permute(ctx, image, 0, 2, 3, 1));                          // [mw, C*H, (W/mw/pw), pw]
            image = ggml_reshape_4d(ctx, image, pw * (W / mw / pw), H, C, mw);                               // [mw, C, H, (W/mw/pw)*pw]
            image = ggml_cont(ctx, ggml_ext_torch_permute(ctx, image, 0, 2, 3, 1));                          // [H, mw, C, (W/mw/pw)*pw]
            image = ggml_reshape_4d(ctx, image, pw, (W / mw / pw) * C * mw, ph, mh * (H / mh / ph));         // [(H/mh/ph)*mh, ph, mw*C*(W/mw/pw), pw]
            image = ggml_cont(ctx, ggml_ext_torch_permute(ctx, image, 0, 2, 1, 3));                          // [(H/mh/ph)*mh, mw*C*(W/mw/pw), ph, pw]
            image = ggml_reshape_4d(ctx, image, pw * ph, (W / mw / pw), C, mw * mh * (H / mh / ph));         // [(H/mh/ph)*mh*mw, C, (W/mw/pw), ph*pw]
            image = ggml_concat(ctx, image, image, 0);                                                       // [(H/mh/ph)*mh*mw, C, (W/mw/pw), pt*ph*pw]
            image = ggml_cont(ctx, ggml_ext_torch_permute(ctx, image, 0, 2, 1, 3));                          // [(H/mh/ph)*mh*mw, (W/mw/pw), C, pt*ph*pw]
            image = ggml_reshape_4d(ctx, image, pw * ph * pt * C, (W / mw / pw), mw * mh, (H / mh / ph));    // [(H/mh/ph), mh*mw, (W/mw/pw), C*pt*ph*pw]
            image = ggml_cont(ctx, ggml_ext_torch_permute(ctx, image, 0, 2, 1, 3));                          // [(H/mh/ph), (W/mw/pw), mh*mw, C*pt*ph*pw]
            image = ggml_reshape_2d(ctx, image, pw * ph * pt * C, mw * mh * (W / mw / pw) * (H / mh / ph));  // [(H/mh/ph)*(W/mw/pw)*mh*mw, C*pt*ph*pw]
            return image;
        }

        ggml_cgraph* build_encode_image_graph(const sd::Tensor<float>& image_tensor) {
            ggml_cgraph* gf    = new_graph_custom(LLM_GRAPH_SIZE);
            ggml_tensor* image = make_input(image_tensor);

            GGML_ASSERT(image->ne[1] % (params.vision.patch_size * params.vision.spatial_merge_size) == 0);
            GGML_ASSERT(image->ne[0] % (params.vision.patch_size * params.vision.spatial_merge_size) == 0);

            int grid_t                 = 1;
            int grid_h                 = static_cast<int>(image->ne[1]) / params.vision.patch_size;
            int grid_w                 = static_cast<int>(image->ne[0]) / params.vision.patch_size;
            int llm_grid_h             = grid_h / params.vision.spatial_merge_size;
            int llm_grid_w             = grid_w / params.vision.spatial_merge_size;
            int vit_merger_window_size = params.vision.window_size / params.vision.patch_size / params.vision.spatial_merge_size;

            auto pixel_values = process_image(compute_ctx, image);

            // window index
            int inverse_index = 0;
            window_index_vec.resize(llm_grid_h * llm_grid_w);
            window_inverse_index_vec.resize(llm_grid_h * llm_grid_w);
            std::vector<int> seqlens;
            for (int ih = 0; ih < llm_grid_h; ih += vit_merger_window_size) {
                for (int iw = 0; iw < llm_grid_w; iw += vit_merger_window_size) {
                    int win_h = std::min(vit_merger_window_size, llm_grid_h - ih);
                    int win_w = std::min(vit_merger_window_size, llm_grid_w - iw);
                    for (int iy = 0; iy < win_h; iy++) {
                        for (int ix = 0; ix < win_w; ix++) {
                            int index                       = (ih + iy) * llm_grid_w + iw + ix;
                            window_index_vec[inverse_index] = index;
                            window_inverse_index_vec[index] = inverse_index;
                            inverse_index++;
                        }
                    }
                    seqlens.push_back(win_h * win_w * params.vision.spatial_merge_size * params.vision.spatial_merge_size);
                }
            }
            // printf("window_index: ");
            // for (int i : window_index_vec) {
            //     printf("%d ", i);
            // }
            // printf("\n");
            // printf("window_inverse_index: ");
            // for (int i : window_inverse_index_vec) {
            //     printf("%d ", i);
            // }
            // printf("\n");
            // printf("seqlens: ");
            // for (int i : seqlens) {
            //     printf("%d ", i);
            // }
            // printf("\n");
            auto window_index         = ggml_new_tensor_1d(compute_ctx,
                                                           GGML_TYPE_I32,
                                                           llm_grid_h * llm_grid_w);
            auto window_inverse_index = ggml_new_tensor_1d(compute_ctx,
                                                           GGML_TYPE_I32,
                                                           llm_grid_h * llm_grid_w);
            set_backend_tensor_data(window_index, window_index_vec.data());
            set_backend_tensor_data(window_inverse_index, window_inverse_index_vec.data());

            // window mask
            int seq_window_size = (vit_merger_window_size * params.vision.spatial_merge_size) * (vit_merger_window_size * params.vision.spatial_merge_size);
            window_mask_vec.resize((grid_h * grid_w) * (grid_h * grid_w));
            int window_start_index = 0;
            for (int seq_index = 0; seq_index < seqlens.size(); seq_index++) {
                int window_end_index = window_start_index + seqlens[seq_index];
                // LOG_DEBUG("%d %d", window_start_index, window_end_index);
                GGML_ASSERT(window_end_index <= grid_h * grid_w);
                for (int i = window_start_index; i < window_end_index; i++) {
                    for (int j = 0; j < grid_h * grid_w; j++) {
                        float mask_value = -INFINITY;
                        if (j >= window_start_index && j < window_end_index) {
                            mask_value = 0;
                        }
                        GGML_ASSERT((i * (grid_h * grid_w) + j) < window_mask_vec.size());
                        window_mask_vec[i * (grid_h * grid_w) + j] = mask_value;
                    }
                }
                window_start_index = window_end_index;
                // printf("\n");
            }
            // printf("window_mask: \n");
            // for (int i = 0; i < grid_h*grid_w; i++) {
            //     for (int j = 0; j < grid_h*grid_w; j++) {
            //         printf("%f ", window_mask_vec[i * (grid_h * grid_w) + j]);
            //     }
            //     printf("\n");
            // }
            auto window_mask = ggml_new_tensor_2d(compute_ctx,
                                                  GGML_TYPE_F32,
                                                  grid_h * grid_w,
                                                  grid_h * grid_w);
            set_backend_tensor_data(window_mask, window_mask_vec.data());

            // pe
            int head_dim = static_cast<int>(params.vision.hidden_size / params.vision.num_heads);
            pe_vec       = Rope::gen_qwen2vl_pe(grid_h,
                                                grid_w,
                                                params.vision.spatial_merge_size,
                                                window_inverse_index_vec,
                                                10000,
                                                {head_dim / 2, head_dim / 2});
            int pos_len  = static_cast<int>(pe_vec.size() / head_dim / 2);
            // LOG_DEBUG("pos_len %d", pos_len);
            auto pe = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, head_dim / 2, pos_len);
            // pe->data = pe_vec.data();
            // print_ggml_tensor(pe);
            // pe->data = nullptr;
            set_backend_tensor_data(pe, pe_vec.data());

            auto runnter_ctx           = get_context();
            ggml_tensor* hidden_states = vision_forward(&runnter_ctx,
                                                        pixel_values,
                                                        pe,
                                                        window_index,
                                                        window_inverse_index,
                                                        window_mask);
            ggml_build_forward_expand(gf, hidden_states);

            return gf;
        }

        sd::Tensor<float> encode_image(const int n_threads,
                                       const sd::Tensor<float>& image) {
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_encode_image_graph(image);
            };
            return take_or_empty(GGMLRunner::compute<float>(get_graph, n_threads, false));
        }
    };

    struct LLMEmbedder {
        std::shared_ptr<BPETokenizer> tokenizer;
        LLMRunner model;

        LLMEmbedder(LLMArch arch,
                    ggml_backend_t backend,
                    bool offload_params_to_cpu,
                    const String2TensorStorage& tensor_storage_map = {},
                    const std::string prefix                       = "",
                    bool enable_vision                             = false)
            : model(arch, backend, offload_params_to_cpu, tensor_storage_map, prefix, enable_vision) {
            if (arch == LLMArch::MISTRAL_SMALL_3_2 || arch == LLMArch::MINISTRAL_3_3B) {
                tokenizer = std::make_shared<MistralTokenizer>();
            } else if (arch == LLMArch::GEMMA3) {
                // Gemma 3 uses SentencePiece (vocab 262208). A SentencePiece loader is
                // not yet implemented in this repo; tokenization path lands in task #25.
                GGML_ABORT("Gemma 3 SentencePiece tokenizer not implemented yet");
            } else {
                tokenizer = std::make_shared<Qwen2Tokenizer>();
            }
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string prefix) {
            model.get_param_tensors(tensors, prefix);
        }

        void alloc_params_buffer() {
            model.alloc_params_buffer();
        }

        std::tuple<std::vector<int>, std::vector<float>> tokenize(std::string text,
                                                                  std::pair<int, int> attn_range,
                                                                  size_t max_length = 0,
                                                                  bool padding      = false) {
            std::vector<std::pair<std::string, float>> parsed_attention;
            parsed_attention.emplace_back(text.substr(0, attn_range.first), 1.f);
            if (attn_range.second - attn_range.first > 0) {
                auto new_parsed_attention = parse_prompt_attention(text.substr(attn_range.first, attn_range.second - attn_range.first));
                parsed_attention.insert(parsed_attention.end(),
                                        new_parsed_attention.begin(),
                                        new_parsed_attention.end());
            }
            parsed_attention.emplace_back(text.substr(attn_range.second), 1.f);
            {
                std::stringstream ss;
                ss << "[";
                for (const auto& item : parsed_attention) {
                    ss << "['" << item.first << "', " << item.second << "], ";
                }
                ss << "]";
                LOG_DEBUG("parse '%s' to %s", text.c_str(), ss.str().c_str());
            }

            std::vector<int> tokens;
            std::vector<float> weights;
            for (const auto& item : parsed_attention) {
                const std::string& curr_text = item.first;
                float curr_weight            = item.second;
                std::vector<int> curr_tokens = tokenizer->tokenize(curr_text, nullptr);
                tokens.insert(tokens.end(), curr_tokens.begin(), curr_tokens.end());
                weights.insert(weights.end(), curr_tokens.size(), curr_weight);
            }

            tokenizer->pad_tokens(tokens, &weights, nullptr, padding ? max_length : 0, padding ? max_length : 100000000, padding);

            // for (int i = 0; i < tokens.size(); i++) {
            //     std::cout << tokens[i] << ":" << weights[i] << ", ";
            // }
            // std::cout << std::endl;

            return {tokens, weights};
        }

        void test() {
            ggml_init_params params;
            params.mem_size   = static_cast<size_t>(1024 * 1024) * 1024;  // 1GB
            params.mem_buffer = nullptr;
            params.no_alloc   = false;

            ggml_context* ctx = ggml_init(params);
            GGML_ASSERT(ctx != nullptr);
            bool test_mistral          = false;
            bool test_qwen3            = true;
            bool test_vit              = false;
            bool test_decoder_with_vit = false;

            if (test_decoder_with_vit) {
                sd::Tensor<float> image_embed;
                {
                    auto image = sd::load_tensor_from_file_as_tensor<float>("qwen2vl_normalized.bin");
                    print_sd_tensor(image, false, "image");
                    sd::Tensor<float> out;

                    int64_t t0   = ggml_time_ms();
                    auto out_opt = model.encode_image(8, image);
                    int64_t t1   = ggml_time_ms();

                    GGML_ASSERT(!out_opt.empty());
                    out = std::move(out_opt);
                    print_sd_tensor(out, false, "image_embed");
                    image_embed = out;
                    LOG_DEBUG("llm encode_image test done in %lldms", t1 - t0);
                }

                std::string placeholder  = "<|image_pad|>";
                std::string img_prompt   = "Picture 1: <|vision_start|>";  // [24669, 220, 16, 25, 220, 151652]
                int64_t num_image_tokens = image_embed.shape()[1];
                img_prompt.reserve(num_image_tokens * placeholder.size());
                for (int i = 0; i < num_image_tokens; i++) {
                    img_prompt += placeholder;
                }
                img_prompt += "<|vision_end|>";

                std::vector<std::pair<int, sd::Tensor<float>>> image_embeds;
                image_embeds.emplace_back(64, image_embed);

                std::pair<int, int> prompt_attn_range;
                std::string text = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n";
                text += img_prompt;
                prompt_attn_range.first = static_cast<int>(text.size());
                text += "change 'flux.cpp' to 'edit.cpp'";
                prompt_attn_range.second = static_cast<int>(text.size());
                text += "<|im_end|>\n<|im_start|>assistant\n";

                auto tokens_and_weights     = tokenize(text, prompt_attn_range, 0, false);
                std::vector<int>& tokens    = std::get<0>(tokens_and_weights);
                std::vector<float>& weights = std::get<1>(tokens_and_weights);
                for (auto token : tokens) {
                    printf("%d ", token);
                }
                printf("\n");
                auto input_ids = sd::Tensor<int32_t>::from_vector(tokens);
                sd::Tensor<float> out;

                int64_t t0   = ggml_time_ms();
                auto out_opt = model.compute(8, input_ids, sd::Tensor<float>(), image_embeds, {});
                int64_t t1   = ggml_time_ms();

                GGML_ASSERT(!out_opt.empty());
                out = std::move(out_opt);
                print_sd_tensor(out);
                LOG_DEBUG("llm test done in %lldms", t1 - t0);
            } else if (test_vit) {
                // auto image = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 280, 280, 3);
                // ggml_set_f32(image, 0.f);
                auto image = sd::load_tensor_from_file_as_tensor<float>("qwen2vl_normalized.bin");
                print_sd_tensor(image, false, "image");
                sd::Tensor<float> out;

                int64_t t0   = ggml_time_ms();
                auto out_opt = model.encode_image(8, image);
                int64_t t1   = ggml_time_ms();

                GGML_ASSERT(!out_opt.empty());
                out = std::move(out_opt);
                print_sd_tensor(out, false, "out");

                // auto ref_out = load_tensor_from_file(ctx, "qwen2vl.bin");
                // ggml_ext_tensor_diff(ref_out, out, 0.01f);

                LOG_DEBUG("llm test done in %lldms", t1 - t0);
            } else if (test_mistral) {
                std::pair<int, int> prompt_attn_range;
                std::string text        = "[SYSTEM_PROMPT]You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object\nattribution and actions without speculation.[/SYSTEM_PROMPT][INST]";
                prompt_attn_range.first = static_cast<int>(text.size());
                text += "a lovely cat";
                prompt_attn_range.second = static_cast<int>(text.size());
                text += "[/INST]";
                auto tokens_and_weights     = tokenize(text, prompt_attn_range, 0, false);
                std::vector<int>& tokens    = std::get<0>(tokens_and_weights);
                std::vector<float>& weights = std::get<1>(tokens_and_weights);
                for (auto token : tokens) {
                    printf("%d ", token);
                }
                printf("\n");
                auto input_ids = sd::Tensor<int32_t>::from_vector(tokens);
                sd::Tensor<float> out;

                int64_t t0   = ggml_time_ms();
                auto out_opt = model.compute(8, input_ids, sd::Tensor<float>(), {}, {10, 20, 30});
                int64_t t1   = ggml_time_ms();

                GGML_ASSERT(!out_opt.empty());
                out = std::move(out_opt);
                print_sd_tensor(out);
                LOG_DEBUG("llm test done in %lldms", t1 - t0);
            } else if (test_qwen3) {
                std::pair<int, int> prompt_attn_range;
                std::string text        = "<|im_start|>user\n";
                prompt_attn_range.first = static_cast<int>(text.size());
                text += "a lovely cat";
                prompt_attn_range.second = static_cast<int>(text.size());
                text += "<|im_end|>\n<|im_start|>assistant\n";
                auto tokens_and_weights     = tokenize(text, prompt_attn_range, 0, false);
                std::vector<int>& tokens    = std::get<0>(tokens_and_weights);
                std::vector<float>& weights = std::get<1>(tokens_and_weights);
                for (auto token : tokens) {
                    printf("%d ", token);
                }
                printf("\n");
                auto input_ids = sd::Tensor<int32_t>::from_vector(tokens);
                sd::Tensor<float> out;

                int64_t t0   = ggml_time_ms();
                auto out_opt = model.compute(8, input_ids, sd::Tensor<float>(), {}, {35});
                int64_t t1   = ggml_time_ms();

                GGML_ASSERT(!out_opt.empty());
                out = std::move(out_opt);
                print_sd_tensor(out);
                LOG_DEBUG("llm test done in %lldms", t1 - t0);
            } else {
                std::pair<int, int> prompt_attn_range;
                std::string text        = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n";
                prompt_attn_range.first = static_cast<int>(text.size());
                text += "a lovely cat";
                prompt_attn_range.second = static_cast<int>(text.size());
                text += "<|im_end|>\n<|im_start|>assistant\n";
                auto tokens_and_weights     = tokenize(text, prompt_attn_range, 0, false);
                std::vector<int>& tokens    = std::get<0>(tokens_and_weights);
                std::vector<float>& weights = std::get<1>(tokens_and_weights);
                for (auto token : tokens) {
                    printf("%d ", token);
                }
                printf("\n");
                auto input_ids = sd::Tensor<int32_t>::from_vector(tokens);
                sd::Tensor<float> out;

                int64_t t0   = ggml_time_ms();
                auto out_opt = model.compute(8, input_ids, sd::Tensor<float>(), {}, {});
                int64_t t1   = ggml_time_ms();

                GGML_ASSERT(!out_opt.empty());
                out = std::move(out_opt);
                print_sd_tensor(out);
                LOG_DEBUG("llm test done in %lldms", t1 - t0);
            }
        }

        static void load_from_file_and_test(const std::string& file_path) {
            // cpu f16: pass
            // ggml_backend_t backend = ggml_backend_cuda_init(0);
            ggml_backend_t backend    = ggml_backend_cpu_init();
            ggml_type model_data_type = GGML_TYPE_COUNT;

            ModelLoader model_loader;
            if (!model_loader.init_from_file_and_convert_name(file_path, "text_encoders.llm.")) {
                LOG_ERROR("init model loader from file failed: '%s'", file_path.c_str());
                return;
            }

            auto& tensor_storage_map = model_loader.get_tensor_storage_map();
            if (model_data_type != GGML_TYPE_COUNT) {
                for (auto& [name, tensor_storage] : tensor_storage_map) {
                    if (ends_with(name, "weight")) {
                        tensor_storage.expected_type = model_data_type;
                    }
                }
            }

            LLMArch arch = LLMArch::QWEN3;

            std::shared_ptr<LLMEmbedder> llm = std::make_shared<LLMEmbedder>(arch,
                                                                             backend,
                                                                             true,
                                                                             tensor_storage_map,
                                                                             "text_encoders.llm",
                                                                             true);

            llm->alloc_params_buffer();
            std::map<std::string, ggml_tensor*> tensors;
            llm->get_param_tensors(tensors, "text_encoders.llm");

            bool success = model_loader.load_tensors(tensors);

            if (!success) {
                LOG_ERROR("load tensors from model loader failed");
                return;
            }

            LOG_INFO("llm model loaded");
            llm->test();
        }
    };
};  // LLM

#endif  // __LLM_HPP__
