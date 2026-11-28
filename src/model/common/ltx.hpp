#ifndef __SD_MODEL_COMMON_LTX_HPP__
#define __SD_MODEL_COMMON_LTX_HPP__

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/ggml_tensor_utils.h"
#include "model/common/block.hpp"
#include "model/common/rope.hpp"
#include "model_loader.h"

namespace LTXV {

    constexpr int LTXAV_GRAPH_SIZE = 102400;

    __STATIC_INLINE__ ggml_tensor* rms_norm(ggml_context* ctx,
                                            ggml_tensor* x,
                                            float eps = 1e-6f) {
        return ggml_rms_norm(ctx, x, eps);
    }

    __STATIC_INLINE__ int count_prefix_blocks(const String2TensorStorage& tensor_storage_map,
                                              const std::string& prefix,
                                              const std::string& marker) {
        int max_block = -1;
        for (const auto& [name, _] : tensor_storage_map) {
            if (!starts_with(name, prefix)) {
                continue;
            }
            size_t pos = name.find(marker);
            if (pos == std::string::npos) {
                continue;
            }
            pos += marker.size();
            size_t end = name.find(".", pos);
            if (end == std::string::npos) {
                continue;
            }
            int block = atoi(name.substr(pos, end - pos).c_str());
            max_block = std::max(max_block, block);
        }
        return max_block + 1;
    }

    __STATIC_INLINE__ std::pair<int64_t, int64_t> infer_attention_layout(int64_t hidden_size,
                                                                         int64_t preferred_heads = -1) {
        if (preferred_heads > 0 && hidden_size % preferred_heads == 0) {
            return {preferred_heads, hidden_size / preferred_heads};
        }
        const int candidates[] = {128, 96, 80, 64, 48, 40, 32};
        for (int head_dim : candidates) {
            if (hidden_size % head_dim == 0) {
                int64_t heads = hidden_size / head_dim;
                if (heads >= 8 && heads <= 64) {
                    return {heads, head_dim};
                }
            }
        }
        return {32, hidden_size / 32};
    }

    __STATIC_INLINE__ int64_t infer_gate_heads(const String2TensorStorage& tensor_storage_map,
                                               const std::string& bias_name,
                                               int64_t fallback_heads) {
        auto it = tensor_storage_map.find(bias_name);
        if (it != tensor_storage_map.end()) {
            return it->second.ne[0];
        }
        return fallback_heads;
    }

    __STATIC_INLINE__ std::vector<float> generate_freq_grid(float theta,
                                                            int positional_dims,
                                                            int dim) {
        const int n_elem     = 2 * positional_dims;
        const int freq_count = dim / n_elem;

        std::vector<float> out(freq_count);
        if (freq_count <= 0) {
            return out;
        }
        if (freq_count == 1) {
            out[0] = 1.5707963267948966f;
            return out;
        }

        const float half_pi   = 1.5707963267948966f;
        const float log_theta = std::log(theta);
        for (int i = 0; i < freq_count; i++) {
            float ratio = static_cast<float>(i) / static_cast<float>(freq_count - 1);
            out[i]      = std::exp(log_theta * ratio) * half_pi;
        }
        return out;
    }

    __STATIC_INLINE__ std::vector<double> generate_freq_grid_double(double theta,
                                                                    int positional_dims,
                                                                    int dim) {
        const int n_elem     = 2 * positional_dims;
        const int freq_count = dim / n_elem;

        std::vector<double> out(freq_count);
        if (freq_count <= 0) {
            return out;
        }
        if (freq_count == 1) {
            out[0] = 1.5707963267948966;
            return out;
        }

        const double half_pi   = 1.5707963267948966;
        const double log_theta = std::log(theta);
        for (int i = 0; i < freq_count; i++) {
            double ratio = static_cast<double>(i) / static_cast<double>(freq_count - 1);
            out[i]       = std::exp(log_theta * ratio) * half_pi;
        }
        return out;
    }

    __STATIC_INLINE__ std::vector<float> build_rope_matrix_from_frequencies(
        const std::vector<std::vector<float>>& frequencies,
        int dim) {
        const int half_dim = dim / 2;
        std::vector<float> out(static_cast<size_t>(frequencies.size()) * static_cast<size_t>(half_dim) * 4, 0.f);

        for (size_t token = 0; token < frequencies.size(); token++) {
            for (int i = 0; i < half_dim; i++) {
                float angle = i < static_cast<int>(frequencies[token].size()) ? frequencies[token][i] : 0.f;
                float c     = std::cos(angle);
                float s     = std::sin(angle);

                size_t base   = (token * static_cast<size_t>(half_dim) + static_cast<size_t>(i)) * 4;
                out[base + 0] = c;
                out[base + 1] = -s;
                out[base + 2] = s;
                out[base + 3] = c;
            }
        }

        return out;
    }

    __STATIC_INLINE__ std::vector<std::vector<float>> split_frequencies_by_heads(
        const std::vector<std::vector<float>>& frequencies,
        int inner_dim,
        int num_heads) {
        GGML_ASSERT(num_heads > 0);
        GGML_ASSERT(inner_dim % num_heads == 0);
        const int inner_half_dim    = inner_dim / 2;
        const int per_head_half_dim = inner_half_dim / num_heads;
        GGML_ASSERT(inner_half_dim % num_heads == 0);

        std::vector<std::vector<float>> out(
            frequencies.size() * static_cast<size_t>(num_heads),
            std::vector<float>(per_head_half_dim, 0.f));

        for (size_t token = 0; token < frequencies.size(); token++) {
            GGML_ASSERT(static_cast<int>(frequencies[token].size()) == inner_half_dim);
            for (int head = 0; head < num_heads; head++) {
                auto& dst = out[token * static_cast<size_t>(num_heads) + static_cast<size_t>(head)];
                std::copy_n(frequencies[token].begin() + head * per_head_half_dim, per_head_half_dim, dst.begin());
            }
        }
        return out;
    }

    __STATIC_INLINE__ ggml_tensor* apply_hidden_rope(GGMLRunnerContext* runner_ctx,
                                                     ggml_tensor* x,
                                                     ggml_tensor* pe,
                                                     int64_t heads,
                                                     int64_t dim_head,
                                                     bool rope_interleaved) {
        auto ctx = runner_ctx->ggml_ctx;
        GGML_ASSERT(x->ne[0] == heads * dim_head);
        if (pe != nullptr && pe->ne[3] == x->ne[1] * heads) {
            auto x_flat   = ggml_reshape_4d(ctx, x, dim_head, 1, x->ne[1] * heads, x->ne[2]);
            auto out_flat = Rope::apply_rope(runner_ctx, x_flat, pe, rope_interleaved);
            auto out4     = ggml_reshape_4d(ctx, out_flat, dim_head, heads, x->ne[1], x->ne[2]);
            return ggml_reshape_3d(ctx, out4, heads * dim_head, x->ne[1], x->ne[2]);
        }
        auto x4 = ggml_reshape_4d(ctx, x, dim_head, heads, x->ne[1], x->ne[2]);
        return Rope::apply_rope(runner_ctx, x4, pe, rope_interleaved);
    }

    struct TimestepEmbedder : public GGMLBlock {
        int frequency_embedding_size;

        TimestepEmbedder(int64_t hidden_size,
                         int frequency_embedding_size = 256)
            : frequency_embedding_size(frequency_embedding_size) {
            blocks["linear_1"] = std::make_shared<Linear>(frequency_embedding_size, hidden_size, true, true);
            blocks["linear_2"] = std::make_shared<Linear>(hidden_size, hidden_size, true, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* timestep) {
            auto linear_1 = std::dynamic_pointer_cast<Linear>(blocks["linear_1"]);
            auto linear_2 = std::dynamic_pointer_cast<Linear>(blocks["linear_2"]);

            auto t_emb = ggml_ext_timestep_embedding(ctx, timestep, frequency_embedding_size);
            t_emb      = linear_1->forward(ctx, t_emb);
            t_emb      = ggml_silu_inplace(ctx->ggml_ctx, t_emb);
            t_emb      = linear_2->forward(ctx, t_emb);
            return t_emb;
        }
    };

    struct CrossAttention : public GGMLBlock {
        int64_t heads;
        int64_t dim_head;
        bool rope_interleaved;

        CrossAttention(int64_t query_dim,
                       int64_t context_dim,
                       int64_t heads,
                       int64_t dim_head,
                       bool apply_gated_attention = false,
                       bool rope_interleaved      = true)
            : heads(heads), dim_head(dim_head), rope_interleaved(rope_interleaved) {
            int64_t inner_dim = heads * dim_head;
            blocks["q_norm"]  = std::make_shared<RMSNorm>(inner_dim, 1e-5f);
            blocks["k_norm"]  = std::make_shared<RMSNorm>(inner_dim, 1e-5f);
            blocks["to_q"]    = std::make_shared<Linear>(query_dim, inner_dim, true);
            blocks["to_k"]    = std::make_shared<Linear>(context_dim, inner_dim, true);
            blocks["to_v"]    = std::make_shared<Linear>(context_dim, inner_dim, true);
            if (apply_gated_attention) {
                blocks["to_gate_logits"] = std::make_shared<Linear>(query_dim, heads, true);
            }
            blocks["to_out.0"] = std::make_shared<Linear>(inner_dim, query_dim, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* context = nullptr,
                             ggml_tensor* mask    = nullptr,
                             ggml_tensor* pe      = nullptr,
                             ggml_tensor* k_pe    = nullptr) {
            if (context == nullptr) {
                context = x;
            }

            auto to_q     = std::dynamic_pointer_cast<Linear>(blocks["to_q"]);
            auto to_k     = std::dynamic_pointer_cast<Linear>(blocks["to_k"]);
            auto to_v     = std::dynamic_pointer_cast<Linear>(blocks["to_v"]);
            auto q_norm   = std::dynamic_pointer_cast<RMSNorm>(blocks["q_norm"]);
            auto k_norm   = std::dynamic_pointer_cast<RMSNorm>(blocks["k_norm"]);
            auto to_out_0 = std::dynamic_pointer_cast<Linear>(blocks["to_out.0"]);

            auto q = to_q->forward(ctx, x);
            auto k = to_k->forward(ctx, context);
            auto v = to_v->forward(ctx, context);

            q = q_norm->forward(ctx, q);
            k = k_norm->forward(ctx, k);

            if (pe != nullptr) {
                if (k_pe == nullptr) {
                    k_pe = pe;
                }
                q = apply_hidden_rope(ctx, q, pe, heads, dim_head, rope_interleaved);
                k = apply_hidden_rope(ctx, k, k_pe, heads, dim_head, rope_interleaved);
            }

            const bool strided_qk = ctx->flash_attn_enabled && !ctx->sage_attn_enabled &&
                                    q->ne[2] == 1 && k->ne[2] == 1 && v->ne[2] == 1 &&
                                    sd_backend_is_cuda_family(ctx->backend);
            if (strided_qk) {
                auto split_heads = [&](ggml_tensor* x) {
                    return ggml_reshape_4d(ctx->ggml_ctx, x, dim_head, heads, x->ne[1], 1);
                };
                q = ggml_permute(ctx->ggml_ctx, split_heads(q), 0, 2, 1, 3);
                k = ggml_permute(ctx->ggml_ctx, split_heads(k), 0, 2, 1, 3);
                v = split_heads(v);
            }
            auto out = ggml_ext_attention_ext(ctx, q, k, v, heads, mask, strided_qk, ctx->flash_attn_enabled);

            if (blocks.count("to_gate_logits") > 0) {
                auto to_gate_logits = std::dynamic_pointer_cast<Linear>(blocks["to_gate_logits"]);
                auto gate_logits    = to_gate_logits->forward(ctx, x);
                auto gates          = ggml_sigmoid(ctx->ggml_ctx, gate_logits);
                gates               = ggml_ext_scale(ctx, gates, 2.0f, true);
                gates               = ggml_reshape_4d(ctx->ggml_ctx, gates, 1, heads, gate_logits->ne[1], gate_logits->ne[2]);

                auto out4 = ggml_reshape_4d(ctx->ggml_ctx, out, dim_head, heads, out->ne[1], out->ne[2]);
                out4      = ggml_mul(ctx->ggml_ctx, out4, gates);
                out       = ggml_reshape_3d(ctx->ggml_ctx, out4, heads * dim_head, out4->ne[2], out4->ne[3]);
            }

            return to_out_0->forward(ctx, out);
        }
    };

}  // namespace LTXV

#endif  // __SD_MODEL_COMMON_LTX_HPP__
