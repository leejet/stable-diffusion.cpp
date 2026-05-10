#ifndef __SD_HIDREAM_O1_H__
#define __SD_HIDREAM_O1_H__

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common_dit.hpp"
#include "conditioner.hpp"
#include "llm.hpp"
#include "util.h"

namespace HiDreamO1 {
    constexpr int HIDREAM_O1_GRAPH_SIZE = 32768;
    constexpr int PATCH_SIZE            = 32;
    constexpr int TIMESTEP_TOKEN_NUM    = 1;
    constexpr int IMAGE_TOKEN_ID        = 151655;
    constexpr int VISION_START_TOKEN_ID = 151652;

    static inline std::string repeat_special_token(const std::string& token, int64_t count) {
        std::string out;
        out.reserve(static_cast<size_t>(count) * token.size());
        for (int64_t i = 0; i < count; ++i) {
            out += token;
        }
        return out;
    }

    static inline std::pair<int, int> calculate_dimensions(int max_size, double ratio) {
        int width  = static_cast<int>(std::sqrt(max_size * max_size * ratio));
        int height = static_cast<int>(width / ratio);
        width      = (width / PATCH_SIZE) * PATCH_SIZE;
        height     = (height / PATCH_SIZE) * PATCH_SIZE;
        width      = std::max(width, PATCH_SIZE);
        height     = std::max(height, PATCH_SIZE);
        return {width, height};
    }

    static inline sd::Tensor<float> resize_to_area(const sd::Tensor<float>& image, int image_size) {
        int64_t width  = image.shape()[0];
        int64_t height = image.shape()[1];
        int64_t s_max  = static_cast<int64_t>(image_size) * image_size;
        double scale   = std::sqrt(static_cast<double>(s_max) / static_cast<double>(width * height));

        std::vector<std::pair<int64_t, int64_t>> sizes = {
            {(static_cast<int64_t>(std::llround(width * scale)) / PATCH_SIZE) * PATCH_SIZE, (static_cast<int64_t>(std::llround(height * scale)) / PATCH_SIZE) * PATCH_SIZE},
            {(static_cast<int64_t>(std::llround(width * scale)) / PATCH_SIZE) * PATCH_SIZE, (static_cast<int64_t>(std::floor(height * scale)) / PATCH_SIZE) * PATCH_SIZE},
            {(static_cast<int64_t>(std::floor(width * scale)) / PATCH_SIZE) * PATCH_SIZE, (static_cast<int64_t>(std::llround(height * scale)) / PATCH_SIZE) * PATCH_SIZE},
            {(static_cast<int64_t>(std::floor(width * scale)) / PATCH_SIZE) * PATCH_SIZE, (static_cast<int64_t>(std::floor(height * scale)) / PATCH_SIZE) * PATCH_SIZE},
        };
        std::sort(sizes.begin(), sizes.end(), [](const auto& a, const auto& b) {
            return a.first * a.second > b.first * b.second;
        });

        std::pair<int64_t, int64_t> new_size = sizes.back();
        for (const auto& size : sizes) {
            if (size.first > 0 && size.second > 0 && size.first * size.second <= s_max) {
                new_size = size;
                break;
            }
        }

        double s1 = static_cast<double>(width) / static_cast<double>(new_size.first);
        double s2 = static_cast<double>(height) / static_cast<double>(new_size.second);
        sd::Tensor<float> resized;
        if (s1 < s2) {
            int64_t resized_h = static_cast<int64_t>(std::llround(height / s1));
            resized           = sd::ops::interpolate(image, {new_size.first, resized_h, image.shape()[2], image.shape()[3]});
            int64_t top       = (resized_h - new_size.second) / 2;
            resized           = sd::ops::slice(resized, 1, top, top + new_size.second);
        } else {
            int64_t resized_w = static_cast<int64_t>(std::llround(width / s2));
            resized           = sd::ops::interpolate(image, {resized_w, new_size.second, image.shape()[2], image.shape()[3]});
            int64_t left      = (resized_w - new_size.first) / 2;
            resized           = sd::ops::slice(resized, 0, left, left + new_size.first);
        }
        return resized;
    }

    static inline std::vector<int32_t> build_position_ids(const std::vector<int32_t>& input_ids,
                                                          const std::vector<std::array<int32_t, 3>>& image_grids,
                                                          const std::vector<int32_t>& skip_vision_start_token) {
        std::vector<int32_t> position_ids(4 * input_ids.size(), 0);
        int image_index = 0;
        int st          = 0;
        int fix_point   = 4096;
        std::vector<int32_t> out_t;
        std::vector<int32_t> out_h;
        std::vector<int32_t> out_w;

        while (st < static_cast<int>(input_ids.size())) {
            int ed = st;
            while (ed < static_cast<int>(input_ids.size()) && input_ids[ed] != IMAGE_TOKEN_ID) {
                ed++;
            }

            if (ed >= static_cast<int>(input_ids.size())) {
                int st_idx = out_t.empty() ? 0 : (*std::max_element(out_t.begin(), out_t.end()) + 1);
                for (int i = 0; i < static_cast<int>(input_ids.size()) - st; ++i) {
                    out_t.push_back(st_idx + i);
                    out_h.push_back(st_idx + i);
                    out_w.push_back(st_idx + i);
                }
                break;
            }

            int text_len = std::max(0, ed - st - skip_vision_start_token[image_index]);
            int st_idx   = out_t.empty() ? 0 : (*std::max_element(out_t.begin(), out_t.end()) + 1);
            for (int i = 0; i < text_len; ++i) {
                out_t.push_back(st_idx + i);
                out_h.push_back(st_idx + i);
                out_w.push_back(st_idx + i);
            }

            auto grid = image_grids[image_index];
            int base;
            if (skip_vision_start_token[image_index]) {
                if (fix_point > 0) {
                    base      = fix_point;
                    fix_point = 0;
                } else {
                    base = st_idx;
                }
            } else {
                base = text_len + st_idx;
            }
            for (int32_t ti = 0; ti < grid[0]; ++ti) {
                for (int32_t hi = 0; hi < grid[1]; ++hi) {
                    for (int32_t wi = 0; wi < grid[2]; ++wi) {
                        out_t.push_back(base + ti);
                        out_h.push_back(base + hi);
                        out_w.push_back(base + wi);
                    }
                }
            }

            st = ed + grid[0] * grid[1] * grid[2];
            image_index++;
        }

        GGML_ASSERT(out_t.size() == input_ids.size());
        for (size_t i = 0; i < input_ids.size(); ++i) {
            // ggml IMROPE consumes 4 flattened position streams:
            //   [t, h, w, e]
            // llama.cpp's generic Qwen-VL fallback expands text positions as
            // [pos, pos, pos, 0]. Keep the extra stream zeroed here too.
            position_ids[i]                        = out_t[i];
            position_ids[input_ids.size() + i]     = out_h[i];
            position_ids[input_ids.size() * 2 + i] = out_w[i];
            position_ids[input_ids.size() * 3 + i] = 0;
        }
        return position_ids;
    }

    struct TimestepEmbedder : public GGMLBlock {
        int frequency_embedding_size = 256;

        TimestepEmbedder(int64_t hidden_size) {
            blocks["mlp.0"] = std::make_shared<Linear>(frequency_embedding_size, hidden_size, true);
            blocks["mlp.2"] = std::make_shared<Linear>(hidden_size, hidden_size, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* t) {
            auto mlp_0 = std::dynamic_pointer_cast<Linear>(blocks["mlp.0"]);
            auto mlp_2 = std::dynamic_pointer_cast<Linear>(blocks["mlp.2"]);
            auto emb   = ggml_ext_timestep_embedding(ctx->ggml_ctx, t, frequency_embedding_size, 10000, 1000.0f);
            emb        = mlp_0->forward(ctx, emb);
            emb        = ggml_silu_inplace(ctx->ggml_ctx, emb);
            emb        = mlp_2->forward(ctx, emb);
            return emb;
        }
    };

    struct BottleneckPatchEmbed : public GGMLBlock {
        BottleneckPatchEmbed(int64_t in_dim, int64_t pca_dim, int64_t embed_dim) {
            blocks["proj1"] = std::make_shared<Linear>(in_dim, pca_dim, false);
            blocks["proj2"] = std::make_shared<Linear>(pca_dim, embed_dim, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto proj1 = std::dynamic_pointer_cast<Linear>(blocks["proj1"]);
            auto proj2 = std::dynamic_pointer_cast<Linear>(blocks["proj2"]);
            return proj2->forward(ctx, proj1->forward(ctx, x));
        }
    };

    struct FinalLayer : public GGMLBlock {
        FinalLayer(int64_t hidden_size, int64_t out_dim) {
            blocks["linear"] = std::make_shared<Linear>(hidden_size, out_dim, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto linear = std::dynamic_pointer_cast<Linear>(blocks["linear"]);
            return linear->forward(ctx, x);
        }
    };

    struct HiDreamO1Params {
        LLM::LLMParams llm;
        int patch_size              = PATCH_SIZE;
        int num_position_embeddings = 2304;
        std::vector<int> deepstack_visual_indexes;
    };

    struct VisionMLP : public GGMLBlock {
        VisionMLP(int64_t hidden_size, int64_t intermediate_size) {
            blocks["linear_fc1"] = std::make_shared<Linear>(hidden_size, intermediate_size, true);
            blocks["linear_fc2"] = std::make_shared<Linear>(intermediate_size, hidden_size, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto linear_fc1 = std::dynamic_pointer_cast<Linear>(blocks["linear_fc1"]);
            auto linear_fc2 = std::dynamic_pointer_cast<Linear>(blocks["linear_fc2"]);

            x = linear_fc1->forward(ctx, x);
            x = ggml_ext_gelu(ctx->ggml_ctx, x);
            x = linear_fc2->forward(ctx, x);
            return x;
        }
    };

    struct VisionPatchEmbed : public GGMLBlock {
        int patch_size;
        int temporal_patch_size;
        int64_t in_channels;
        int64_t embed_dim;

        VisionPatchEmbed(int patch_size,
                         int temporal_patch_size,
                         int64_t in_channels,
                         int64_t embed_dim)
            : patch_size(patch_size),
              temporal_patch_size(temporal_patch_size),
              in_channels(in_channels),
              embed_dim(embed_dim) {
            blocks["proj"] = std::make_shared<Conv3d>(in_channels,
                                                      embed_dim,
                                                      std::tuple<int, int, int>{temporal_patch_size, patch_size, patch_size},
                                                      std::tuple<int, int, int>{temporal_patch_size, patch_size, patch_size},
                                                      std::tuple<int, int, int>{0, 0, 0},
                                                      std::tuple<int, int, int>{1, 1, 1},
                                                      true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto proj = std::dynamic_pointer_cast<Conv3d>(blocks["proj"]);
            x         = ggml_reshape_4d(ctx->ggml_ctx,
                                        x,
                                        patch_size,
                                        patch_size,
                                        temporal_patch_size,
                                        ggml_nelements(x) / (temporal_patch_size * patch_size * patch_size));
            x         = proj->forward(ctx, x);
            x         = ggml_reshape_2d(ctx->ggml_ctx, x, embed_dim, ggml_nelements(x) / embed_dim);
            return x;
        }
    };

    struct VisionPatchMerger : public GGMLBlock {
        int64_t hidden_size;
        bool use_postshuffle_norm;

        VisionPatchMerger(int64_t dim,
                          int64_t context_dim,
                          int spatial_merge_size,
                          bool use_postshuffle_norm)
            : hidden_size(context_dim * spatial_merge_size * spatial_merge_size),
              use_postshuffle_norm(use_postshuffle_norm) {
            blocks["norm"]       = std::make_shared<LayerNorm>(use_postshuffle_norm ? hidden_size : context_dim, 1e-6f);
            blocks["linear_fc1"] = std::make_shared<Linear>(hidden_size, hidden_size, true);
            blocks["linear_fc2"] = std::make_shared<Linear>(hidden_size, dim, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto norm       = std::dynamic_pointer_cast<LayerNorm>(blocks["norm"]);
            auto linear_fc1 = std::dynamic_pointer_cast<Linear>(blocks["linear_fc1"]);
            auto linear_fc2 = std::dynamic_pointer_cast<Linear>(blocks["linear_fc2"]);

            x = norm->forward(ctx, x);
            x = ggml_reshape_2d(ctx->ggml_ctx, x, hidden_size, ggml_nelements(x) / hidden_size);
            x = linear_fc1->forward(ctx, x);
            x = ggml_ext_gelu(ctx->ggml_ctx, x);
            x = linear_fc2->forward(ctx, x);
            return x;
        }
    };

    struct VisionAttention : public GGMLBlock {
        int head_dim;
        int num_heads;

        VisionAttention(int64_t hidden_size, int num_heads)
            : num_heads(num_heads) {
            head_dim = static_cast<int>(hidden_size / num_heads);
            GGML_ASSERT(num_heads * head_dim == hidden_size);
            blocks["qkv"]  = std::make_shared<Linear>(hidden_size, hidden_size * 3, true);
            blocks["proj"] = std::make_shared<Linear>(hidden_size, hidden_size, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* pe) {
            auto qkv_proj = std::dynamic_pointer_cast<Linear>(blocks["qkv"]);
            auto proj     = std::dynamic_pointer_cast<Linear>(blocks["proj"]);

            auto qkv     = qkv_proj->forward(ctx, x);
            auto qkv_vec = split_qkv(ctx->ggml_ctx, qkv);

            auto q = ggml_reshape_4d(ctx->ggml_ctx, qkv_vec[0], head_dim, num_heads, qkv_vec[0]->ne[1], qkv_vec[0]->ne[2]);
            auto k = ggml_reshape_4d(ctx->ggml_ctx, qkv_vec[1], head_dim, num_heads, qkv_vec[1]->ne[1], qkv_vec[1]->ne[2]);
            auto v = ggml_reshape_4d(ctx->ggml_ctx, qkv_vec[2], head_dim, num_heads, qkv_vec[2]->ne[1], qkv_vec[2]->ne[2]);

            x = Rope::attention(ctx, q, k, v, pe, nullptr, 1.f, false);
            x = proj->forward(ctx, x);
            return x;
        }
    };

    struct VisionBlock : public GGMLBlock {
        VisionBlock(int64_t hidden_size,
                    int64_t intermediate_size,
                    int num_heads) {
            blocks["norm1"] = std::make_shared<LayerNorm>(hidden_size, 1e-6f);
            blocks["norm2"] = std::make_shared<LayerNorm>(hidden_size, 1e-6f);
            blocks["attn"]  = std::make_shared<VisionAttention>(hidden_size, num_heads);
            blocks["mlp"]   = std::make_shared<VisionMLP>(hidden_size, intermediate_size);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* pe) {
            auto norm1 = std::dynamic_pointer_cast<LayerNorm>(blocks["norm1"]);
            auto norm2 = std::dynamic_pointer_cast<LayerNorm>(blocks["norm2"]);
            auto attn  = std::dynamic_pointer_cast<VisionAttention>(blocks["attn"]);
            auto mlp   = std::dynamic_pointer_cast<VisionMLP>(blocks["mlp"]);

            auto residual = x;
            x             = norm1->forward(ctx, x);
            x             = attn->forward(ctx, x, pe);
            x             = ggml_add_inplace(ctx->ggml_ctx, x, residual);

            residual = x;
            x        = norm2->forward(ctx, x);
            x        = mlp->forward(ctx, x);
            x        = ggml_add_inplace(ctx->ggml_ctx, x, residual);
            return x;
        }
    };

    struct VisionOutput {
        ggml_tensor* hidden_states = nullptr;
        std::vector<ggml_tensor*> deepstack_hidden_states;
    };

    struct VisionModel : public GGMLBlock {
        int num_layers;
        int spatial_merge_size;
        int num_grid_per_side;
        std::vector<int> deepstack_visual_indexes;

        VisionModel(int num_layers,
                    int64_t in_channels,
                    int64_t hidden_size,
                    int64_t out_hidden_size,
                    int64_t intermediate_size,
                    int num_heads,
                    int spatial_merge_size,
                    int patch_size,
                    int temporal_patch_size,
                    int num_position_embeddings,
                    std::vector<int> deepstack_visual_indexes)
            : num_layers(num_layers),
              spatial_merge_size(spatial_merge_size),
              num_grid_per_side(static_cast<int>(std::sqrt(num_position_embeddings))),
              deepstack_visual_indexes(std::move(deepstack_visual_indexes)) {
            blocks["patch_embed"] = std::make_shared<VisionPatchEmbed>(patch_size,
                                                                       temporal_patch_size,
                                                                       in_channels,
                                                                       hidden_size);
            blocks["pos_embed"]   = std::make_shared<Embedding>(num_position_embeddings, hidden_size);
            for (int i = 0; i < num_layers; ++i) {
                blocks["blocks." + std::to_string(i)] = std::make_shared<VisionBlock>(hidden_size,
                                                                                      intermediate_size,
                                                                                      num_heads);
            }
            blocks["merger"] = std::make_shared<VisionPatchMerger>(out_hidden_size,
                                                                   hidden_size,
                                                                   spatial_merge_size,
                                                                   false);
            for (int i = 0; i < static_cast<int>(this->deepstack_visual_indexes.size()); ++i) {
                blocks["deepstack_merger_list." + std::to_string(i)] = std::make_shared<VisionPatchMerger>(out_hidden_size,
                                                                                                           hidden_size,
                                                                                                           spatial_merge_size,
                                                                                                           true);
            }
        }

        ggml_tensor* fast_pos_embed_interpolate(GGMLRunnerContext* ctx,
                                                int grid_h,
                                                int grid_w) {
            auto pos_embed = std::dynamic_pointer_cast<Embedding>(blocks["pos_embed"]);
            std::vector<int32_t> idx_list[4];
            std::vector<float> weight_list[4];
            idx_list[0].reserve(static_cast<size_t>(grid_h * grid_w));
            idx_list[1].reserve(static_cast<size_t>(grid_h * grid_w));
            idx_list[2].reserve(static_cast<size_t>(grid_h * grid_w));
            idx_list[3].reserve(static_cast<size_t>(grid_h * grid_w));
            weight_list[0].reserve(static_cast<size_t>(grid_h * grid_w));
            weight_list[1].reserve(static_cast<size_t>(grid_h * grid_w));
            weight_list[2].reserve(static_cast<size_t>(grid_h * grid_w));
            weight_list[3].reserve(static_cast<size_t>(grid_h * grid_w));

            double max_index = static_cast<double>(num_grid_per_side - 1);
            for (int h = 0; h < grid_h; ++h) {
                double h_pos = grid_h == 1 ? 0.0 : max_index * h / static_cast<double>(grid_h - 1);
                int h_floor  = static_cast<int>(std::floor(h_pos));
                int h_ceil   = std::min(h_floor + 1, num_grid_per_side - 1);
                double dh    = h_pos - h_floor;
                for (int w = 0; w < grid_w; ++w) {
                    double w_pos = grid_w == 1 ? 0.0 : max_index * w / static_cast<double>(grid_w - 1);
                    int w_floor  = static_cast<int>(std::floor(w_pos));
                    int w_ceil   = std::min(w_floor + 1, num_grid_per_side - 1);
                    double dw    = w_pos - w_floor;

                    idx_list[0].push_back(h_floor * num_grid_per_side + w_floor);
                    idx_list[1].push_back(h_floor * num_grid_per_side + w_ceil);
                    idx_list[2].push_back(h_ceil * num_grid_per_side + w_floor);
                    idx_list[3].push_back(h_ceil * num_grid_per_side + w_ceil);

                    weight_list[0].push_back(static_cast<float>((1.0 - dh) * (1.0 - dw)));
                    weight_list[1].push_back(static_cast<float>((1.0 - dh) * dw));
                    weight_list[2].push_back(static_cast<float>(dh * (1.0 - dw)));
                    weight_list[3].push_back(static_cast<float>(dh * dw));
                }
            }

            ggml_tensor* patch_pos_embeds = nullptr;
            for (int i = 0; i < 4; ++i) {
                auto idx_tensor = ggml_new_tensor_1d(ctx->ggml_ctx, GGML_TYPE_I32, static_cast<int64_t>(idx_list[i].size()));
                std::memcpy(idx_tensor->data, idx_list[i].data(), idx_list[i].size() * sizeof(int32_t));
                auto embed         = pos_embed->forward(ctx, idx_tensor);
                auto weight_tensor = ggml_new_tensor_2d(ctx->ggml_ctx, GGML_TYPE_F32, 1, static_cast<int64_t>(weight_list[i].size()));
                std::memcpy(weight_tensor->data, weight_list[i].data(), weight_list[i].size() * sizeof(float));
                embed            = ggml_mul(ctx->ggml_ctx, embed, weight_tensor);
                patch_pos_embeds = patch_pos_embeds == nullptr ? embed : ggml_add(ctx->ggml_ctx, patch_pos_embeds, embed);
            }

            patch_pos_embeds = ggml_reshape_4d(ctx->ggml_ctx,
                                               patch_pos_embeds,
                                               patch_pos_embeds->ne[0],
                                               spatial_merge_size,
                                               grid_w / spatial_merge_size,
                                               grid_h * spatial_merge_size);
            patch_pos_embeds = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, patch_pos_embeds, 0, 1, 3, 2));
            patch_pos_embeds = ggml_reshape_2d(ctx->ggml_ctx,
                                               patch_pos_embeds,
                                               patch_pos_embeds->ne[0],
                                               ggml_nelements(patch_pos_embeds) / patch_pos_embeds->ne[0]);
            return patch_pos_embeds;
        }

        VisionOutput forward(GGMLRunnerContext* ctx,
                             ggml_tensor* pixel_values,
                             ggml_tensor* pe,
                             int grid_h,
                             int grid_w) {
            auto patch_embed = std::dynamic_pointer_cast<VisionPatchEmbed>(blocks["patch_embed"]);
            auto merger      = std::dynamic_pointer_cast<VisionPatchMerger>(blocks["merger"]);

            auto x          = patch_embed->forward(ctx, pixel_values);
            auto pos_embeds = fast_pos_embed_interpolate(ctx, grid_h, grid_w);
            x               = ggml_add(ctx->ggml_ctx, x, pos_embeds);
            x               = ggml_reshape_3d(ctx->ggml_ctx, x, x->ne[0], x->ne[1], 1);

            VisionOutput out;
            for (int i = 0; i < num_layers; ++i) {
                auto block = std::dynamic_pointer_cast<VisionBlock>(blocks["blocks." + std::to_string(i)]);
                x          = block->forward(ctx, x, pe);
                for (int j = 0; j < static_cast<int>(deepstack_visual_indexes.size()); ++j) {
                    if (deepstack_visual_indexes[j] == i) {
                        auto deepstack_merger = std::dynamic_pointer_cast<VisionPatchMerger>(blocks["deepstack_merger_list." + std::to_string(j)]);
                        out.deepstack_hidden_states.push_back(deepstack_merger->forward(ctx, x));
                        break;
                    }
                }
            }

            out.hidden_states = merger->forward(ctx, x);
            return out;
        }
    };

    struct HiDreamO1Model : public GGMLBlock {
        HiDreamO1Params params;

        HiDreamO1Model() = default;
        explicit HiDreamO1Model(HiDreamO1Params params)
            : params(std::move(params)) {
            blocks["language_model"] = std::make_shared<LLM::TextModel>(this->params.llm);
            blocks["visual"]         = std::make_shared<VisionModel>(this->params.llm.vision.num_layers,
                                                             this->params.llm.vision.in_channels,
                                                             this->params.llm.vision.hidden_size,
                                                             this->params.llm.vision.out_hidden_size,
                                                             this->params.llm.vision.intermediate_size,
                                                             this->params.llm.vision.num_heads,
                                                             this->params.llm.vision.spatial_merge_size,
                                                             this->params.llm.vision.patch_size,
                                                             this->params.llm.vision.temporal_patch_size,
                                                             this->params.num_position_embeddings,
                                                             this->params.deepstack_visual_indexes);
            blocks["t_embedder1"]    = std::make_shared<TimestepEmbedder>(this->params.llm.hidden_size);
            blocks["x_embedder"]     = std::make_shared<BottleneckPatchEmbed>(this->params.patch_size * this->params.patch_size * 3,
                                                                          this->params.llm.hidden_size / 4,
                                                                          this->params.llm.hidden_size);
            blocks["final_layer2"]   = std::make_shared<FinalLayer>(this->params.llm.hidden_size,
                                                                  this->params.patch_size * this->params.patch_size * 3);
        }

        std::shared_ptr<LLM::TextModel> text_model() {
            return std::dynamic_pointer_cast<LLM::TextModel>(blocks["language_model"]);
        }

        std::shared_ptr<VisionModel> vision_model() {
            return std::dynamic_pointer_cast<VisionModel>(blocks["visual"]);
        }

        std::shared_ptr<TimestepEmbedder> timestep_embedder() {
            return std::dynamic_pointer_cast<TimestepEmbedder>(blocks["t_embedder1"]);
        }

        std::shared_ptr<BottleneckPatchEmbed> patch_embedder() {
            return std::dynamic_pointer_cast<BottleneckPatchEmbed>(blocks["x_embedder"]);
        }

        std::shared_ptr<FinalLayer> final_layer() {
            return std::dynamic_pointer_cast<FinalLayer>(blocks["final_layer2"]);
        }
    };

    struct HiDreamO1Runner : public GGMLRunner {
        HiDreamO1Params params;
        HiDreamO1Model model;

        std::vector<int> window_index_vec;
        std::vector<int> window_inverse_index_vec;
        std::vector<float> window_mask_vec;
        std::vector<float> pe_vec;
        std::vector<float> attention_mask_vec;

        HiDreamO1Runner(ggml_backend_t backend,
                        bool offload_params_to_cpu,
                        const String2TensorStorage& tensor_storage_map = {},
                        const std::string& prefix                      = "model")
            : GGMLRunner(backend, offload_params_to_cpu) {
            params.llm.arch                       = LLM::LLMArch::QWEN3_VL;
            params.llm.hidden_size                = 4096;
            params.llm.intermediate_size          = 12288;
            params.llm.num_layers                 = 36;
            params.llm.num_heads                  = 32;
            params.llm.num_kv_heads               = 8;
            params.llm.head_dim                   = 128;
            params.llm.qkv_bias                   = false;
            params.llm.qk_norm                    = true;
            params.llm.vocab_size                 = 151936;
            params.llm.rms_norm_eps               = 1e-6f;
            params.llm.vision.num_layers          = 27;
            params.llm.vision.hidden_size         = 1152;
            params.llm.vision.intermediate_size   = 4304;
            params.llm.vision.num_heads           = 16;
            params.llm.vision.out_hidden_size     = 4096;
            params.llm.vision.patch_size          = 16;
            params.llm.vision.spatial_merge_size  = 2;
            params.llm.vision.temporal_patch_size = 2;
            params.num_position_embeddings        = 2304;
            params.deepstack_visual_indexes       = {8, 16, 24};

            model = HiDreamO1Model(params);
            model.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return "hidream_o1";
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix) {
            model.get_param_tensors(tensors, prefix);
        }

        ggml_tensor* process_image(ggml_context* ctx, ggml_tensor* image) {
            int64_t C  = image->ne[2];
            int64_t H  = image->ne[1];
            int64_t W  = image->ne[0];
            int64_t mh = params.llm.vision.spatial_merge_size;
            int64_t mw = params.llm.vision.spatial_merge_size;
            int64_t pt = params.llm.vision.temporal_patch_size;
            int64_t ph = params.llm.vision.patch_size;
            int64_t pw = params.llm.vision.patch_size;

            image = ggml_reshape_4d(ctx, image, pw, mw, (W / mw / pw), H * C);
            image = ggml_cont(ctx, ggml_ext_torch_permute(ctx, image, 0, 2, 3, 1));
            image = ggml_reshape_4d(ctx, image, pw * (W / mw / pw), H, C, mw);
            image = ggml_cont(ctx, ggml_ext_torch_permute(ctx, image, 0, 2, 3, 1));
            image = ggml_reshape_4d(ctx, image, pw, (W / mw / pw) * C * mw, ph, mh * (H / mh / ph));
            image = ggml_cont(ctx, ggml_ext_torch_permute(ctx, image, 0, 2, 1, 3));
            image = ggml_reshape_4d(ctx, image, pw * ph, (W / mw / pw), C, mw * mh * (H / mh / ph));
            image = ggml_concat(ctx, image, image, 0);
            image = ggml_cont(ctx, ggml_ext_torch_permute(ctx, image, 0, 2, 1, 3));
            image = ggml_reshape_4d(ctx, image, pw * ph * pt * C, (W / mw / pw), mw * mh, (H / mh / ph));
            image = ggml_cont(ctx, ggml_ext_torch_permute(ctx, image, 0, 2, 1, 3));
            image = ggml_reshape_2d(ctx, image, pw * ph * pt * C, mw * mh * (W / mw / pw) * (H / mh / ph));
            return image;
        }

        ggml_tensor* concat_seq(GGMLRunnerContext* ctx, ggml_tensor* a, ggml_tensor* b) {
            if (a == nullptr) {
                return b;
            }
            if (b == nullptr) {
                return a;
            }
            return ggml_concat(ctx->ggml_ctx, a, b, 1);
        }

        ggml_tensor* scatter_visual_embeds(GGMLRunnerContext* ctx,
                                           ggml_tensor* inputs_embeds,
                                           const sd::Tensor<int32_t>& image_embed_ranges_tensor,
                                           ggml_tensor* visual_embeds) {
            if (visual_embeds == nullptr || image_embed_ranges_tensor.empty()) {
                return inputs_embeds;
            }

            ggml_tensor* output = nullptr;
            int prev_end        = 0;
            int n_ranges        = static_cast<int>(image_embed_ranges_tensor.shape()[1]);
            int visual_offset   = 0;
            for (int i = 0; i < n_ranges; ++i) {
                int start = image_embed_ranges_tensor.values()[i * 2];
                int len   = image_embed_ranges_tensor.values()[i * 2 + 1];

                if (start > prev_end) {
                    output = concat_seq(ctx, output, ggml_ext_slice(ctx->ggml_ctx, inputs_embeds, 1, prev_end, start));
                }

                output   = concat_seq(ctx,
                                      output,
                                      ggml_ext_slice(ctx->ggml_ctx, visual_embeds, 1, visual_offset, visual_offset + len));
                prev_end = start + len;
                visual_offset += len;
            }

            if (prev_end < inputs_embeds->ne[1]) {
                output = concat_seq(ctx, output, ggml_ext_slice(ctx->ggml_ctx, inputs_embeds, 1, prev_end, inputs_embeds->ne[1]));
            }
            return output == nullptr ? inputs_embeds : output;
        }

        VisionOutput encode_image(GGMLRunnerContext* runner_ctx, ggml_tensor* image) {
            auto vision = model.vision_model();
            GGML_ASSERT(image->ne[1] % (params.llm.vision.patch_size * params.llm.vision.spatial_merge_size) == 0);
            GGML_ASSERT(image->ne[0] % (params.llm.vision.patch_size * params.llm.vision.spatial_merge_size) == 0);

            int grid_h = static_cast<int>(image->ne[1]) / params.llm.vision.patch_size;
            int grid_w = static_cast<int>(image->ne[0]) / params.llm.vision.patch_size;

            auto pixel_values = process_image(compute_ctx, image);

            int head_dim = static_cast<int>(params.llm.vision.hidden_size / params.llm.vision.num_heads);
            std::vector<int> window_index_vec(static_cast<size_t>((grid_h / params.llm.vision.spatial_merge_size) * (grid_w / params.llm.vision.spatial_merge_size)));
            for (int i = 0; i < static_cast<int>(window_index_vec.size()); ++i) {
                window_index_vec[static_cast<size_t>(i)] = i;
            }
            pe_vec      = Rope::gen_qwen2vl_pe(grid_h, grid_w, params.llm.vision.spatial_merge_size, window_index_vec, 10000, {head_dim / 2, head_dim / 2});
            int pos_len = static_cast<int>(pe_vec.size() / head_dim / 2);
            auto pe     = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, head_dim / 2, pos_len);
            set_backend_tensor_data(pe, pe_vec.data());

            return vision->forward(runner_ctx, pixel_values, pe, grid_h, grid_w);
        }

        ggml_cgraph* build_graph(const sd::Tensor<float>& x_tensor,
                                 const sd::Tensor<float>& timestep_tensor,
                                 const sd::Tensor<int32_t>& input_ids_tensor,
                                 const sd::Tensor<int32_t>& input_pos_tensor,
                                 const sd::Tensor<int32_t>& token_types_tensor,
                                 const sd::Tensor<int32_t>& image_embed_ranges_tensor,
                                 const sd::Tensor<int32_t>& vinput_mask_tensor,
                                 const std::vector<sd::Tensor<float>>& vlm_images,
                                 const std::vector<sd::Tensor<float>>& ref_images) {
            ggml_cgraph* gf        = new_graph_custom(HIDREAM_O1_GRAPH_SIZE);
            ggml_tensor* x         = make_input(x_tensor);
            ggml_tensor* timestep  = make_input(timestep_tensor);
            ggml_tensor* input_ids = make_input(input_ids_tensor);
            ggml_tensor* input_pos = make_input(input_pos_tensor);

            auto text_model   = model.text_model();
            auto t_embedder1  = model.timestep_embedder();
            auto x_embedder   = model.patch_embedder();
            auto final_layer2 = model.final_layer();

            std::vector<ggml_tensor*> vlm_image_tensors;
            for (const auto& image : vlm_images) {
                vlm_image_tensors.push_back(make_input(image));
            }

            std::vector<ggml_tensor*> ref_image_tensors;
            for (const auto& image : ref_images) {
                ref_image_tensors.push_back(make_input(image));
            }

            attention_mask_vec    = std::vector<float>(static_cast<size_t>(token_types_tensor.shape()[0] * token_types_tensor.shape()[0]), 0.0f);
            int64_t total_seq_len = token_types_tensor.shape()[0];
            for (int64_t query = 0; query < total_seq_len; ++query) {
                bool is_gen = token_types_tensor.values()[static_cast<size_t>(query)] > 0;
                for (int64_t key = 0; key < total_seq_len; ++key) {
                    if (!is_gen && key > query) {
                        attention_mask_vec[static_cast<size_t>(query * total_seq_len + key)] = -INFINITY;
                    }
                }
            }
            auto attention_mask = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, total_seq_len, total_seq_len);
            set_backend_tensor_data(attention_mask, attention_mask_vec.data());

            auto runner_ctx            = get_context();
            ggml_tensor* visual_embeds = nullptr;
            for (size_t i = 0; i < vlm_image_tensors.size(); ++i) {
                auto image_output = encode_image(&runner_ctx, vlm_image_tensors[i]);
                visual_embeds     = visual_embeds == nullptr ? image_output.hidden_states : ggml_concat(compute_ctx, visual_embeds, image_output.hidden_states, 1);
            }

            auto txt = text_model->embed(&runner_ctx, input_ids);
            txt      = scatter_visual_embeds(&runner_ctx, txt, image_embed_ranges_tensor, visual_embeds);

            auto t_emb          = t_embedder1->forward(&runner_ctx, timestep);
            int64_t txt_seq_len = input_ids->ne[0];
            if (txt_seq_len > 1) {
                auto prefix = ggml_ext_slice(compute_ctx, txt, 1, 0, txt_seq_len - 1);
                txt         = ggml_concat(compute_ctx, prefix, ggml_reshape_3d(compute_ctx, t_emb, t_emb->ne[0], 1, 1), 1);
            } else {
                txt = ggml_reshape_3d(compute_ctx, t_emb, t_emb->ne[0], 1, 1);
            }

            auto vinputs          = DiT::pad_and_patchify(&runner_ctx, x, PATCH_SIZE, PATCH_SIZE);
            int64_t target_tokens = vinputs->ne[1];
            for (ggml_tensor* ref_image : ref_image_tensors) {
                auto ref = DiT::pad_and_patchify(&runner_ctx, ref_image, PATCH_SIZE, PATCH_SIZE);
                vinputs  = ggml_concat(compute_ctx, vinputs, ref, 1);
            }
            auto vis = x_embedder->forward(&runner_ctx, vinputs);

            auto inputs_embeds = ggml_concat(compute_ctx, txt, vis, 1);
            auto hidden_states = text_model->forward_embeds(&runner_ctx, inputs_embeds, input_pos, attention_mask, {});
            auto x_pred_all    = final_layer2->forward(&runner_ctx, hidden_states);

            int64_t x_pred_start = txt_seq_len;
            if (!vinput_mask_tensor.empty()) {
                int64_t seq_len      = static_cast<int64_t>(vinput_mask_tensor.shape()[0]);
                int64_t first_vinput = 0;
                while (first_vinput < seq_len && vinput_mask_tensor.values()[static_cast<size_t>(first_vinput)] == 0) {
                    first_vinput++;
                }
                x_pred_start = first_vinput;
            }
            auto x_pred = ggml_view_3d(compute_ctx,
                                       x_pred_all,
                                       x_pred_all->ne[0],
                                       target_tokens,
                                       x_pred_all->ne[2],
                                       x_pred_all->nb[1],
                                       x_pred_all->nb[2],
                                       x_pred_start * x_pred_all->nb[1]);
            x_pred      = ggml_cont(compute_ctx, x_pred);
            x_pred      = DiT::unpatchify_and_crop(compute_ctx, x_pred, x->ne[1], x->ne[0], PATCH_SIZE, PATCH_SIZE);

            float sigma = 1.0f - timestep_tensor.values()[0];
            sigma       = std::max(1e-6f, sigma);
            auto out    = ggml_scale(compute_ctx, ggml_sub(compute_ctx, x, x_pred), 1.0f / sigma);

            ggml_build_forward_expand(gf, out);
            return gf;
        }

        sd::Tensor<float> compute(int n_threads,
                                  const sd::Tensor<float>& x,
                                  const sd::Tensor<float>& timestep,
                                  const sd::Tensor<int32_t>& input_ids,
                                  const sd::Tensor<int32_t>& input_pos,
                                  const sd::Tensor<int32_t>& token_types,
                                  const sd::Tensor<int32_t>& image_embed_ranges,
                                  const sd::Tensor<int32_t>& vinput_mask,
                                  const std::vector<sd::Tensor<float>>& vlm_images,
                                  const std::vector<sd::Tensor<float>>& ref_images) {
            auto get_graph = [&]() {
                return build_graph(x, timestep, input_ids, input_pos, token_types, image_embed_ranges, vinput_mask, vlm_images, ref_images);
            };
            return restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, false), x.dim());
        }
    };

    struct HiDreamO1Conditioner : public Conditioner {
        Qwen2Tokenizer tokenizer;

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
            SD_UNUSED(tensors);
        }

        void alloc_params_buffer() override {}
        void free_params_buffer() override {}
        size_t get_params_buffer_size() override { return 0; }
        void set_flash_attention_enabled(bool enabled) override { SD_UNUSED(enabled); }

        SDCondition get_learned_condition(int n_threads,
                                          const ConditionerParams& conditioner_params) override {
            SD_UNUSED(n_threads);
            SDCondition result;

            int width                = conditioner_params.width;
            int height               = conditioner_params.height;
            int64_t target_image_len = static_cast<int64_t>(width / PATCH_SIZE) * static_cast<int64_t>(height / PATCH_SIZE);

            std::vector<sd::Tensor<float>> ref_images;
            if (conditioner_params.ref_images != nullptr) {
                ref_images = *conditioner_params.ref_images;
            }

            std::vector<sd::Tensor<float>> vlm_images;
            std::vector<std::array<int32_t, 3>> image_grids;
            std::vector<int32_t> skip_vision_start;

            std::string prompt = "<|im_start|>user\n";
            std::vector<int32_t> image_ranges;

            if (ref_images.empty()) {
                prompt += conditioner_params.text;
                prompt += "<|im_end|>\n<|im_start|>assistant\n<|boi_token|><|tms_token|>";
                auto input_ids = tokenizer.encode(prompt, nullptr);

                std::vector<int32_t> input_ids_pad = input_ids;
                input_ids_pad.push_back(VISION_START_TOKEN_ID);
                input_ids_pad.insert(input_ids_pad.end(), target_image_len - 1, IMAGE_TOKEN_ID);

                image_grids.push_back({1, static_cast<int32_t>(height / PATCH_SIZE), static_cast<int32_t>(width / PATCH_SIZE)});
                skip_vision_start.push_back(1);

                std::vector<int32_t> token_types(input_ids_pad.size(), 0);
                int txt_seq_len = static_cast<int>(input_ids.size());
                int bgn         = txt_seq_len - TIMESTEP_TOKEN_NUM;
                for (int i = bgn; i < bgn + target_image_len + TIMESTEP_TOKEN_NUM; ++i) {
                    token_types[i] = 1;
                }
                for (int i = txt_seq_len - TIMESTEP_TOKEN_NUM; i < txt_seq_len; ++i) {
                    token_types[i] = 3;
                }

                auto position_ids = build_position_ids(input_ids_pad, image_grids, skip_vision_start);

                std::vector<int64_t> input_shape{static_cast<int64_t>(input_ids.size())};
                std::vector<int64_t> position_shape{static_cast<int64_t>(input_ids_pad.size() * 4)};
                std::vector<int64_t> token_type_shape{static_cast<int64_t>(token_types.size())};
                std::vector<int32_t> vinput_mask(token_types.size(), 0);
                for (int64_t i = txt_seq_len; i < static_cast<int64_t>(vinput_mask.size()); ++i) {
                    vinput_mask[static_cast<size_t>(i)] = 1;
                }
                std::vector<int64_t> vinput_mask_shape{static_cast<int64_t>(vinput_mask.size())};

                result.c_input_ids          = sd::Tensor<int32_t>(input_shape, std::move(input_ids));
                result.c_position_ids       = sd::Tensor<int32_t>(position_shape, position_ids);
                result.c_token_types        = sd::Tensor<int32_t>(token_type_shape, std::move(token_types));
                result.c_vinput_mask        = sd::Tensor<int32_t>(vinput_mask_shape, std::move(vinput_mask));
                result.c_image_embed_ranges = sd::Tensor<int32_t>();
                return result;
            }

            int K = static_cast<int>(ref_images.size());
            int max_size;
            if (K == 1) {
                max_size = std::max(height, width);
            } else if (K == 2) {
                max_size = std::max(height, width) * 48 / 64;
            } else if (K <= 4) {
                max_size = std::max(height, width) / 2;
            } else if (K <= 8) {
                max_size = std::max(height, width) * 24 / 64;
            } else {
                max_size = std::max(height, width) / 4;
            }

            int cond_img_size;
            if (K <= 4) {
                cond_img_size = 384;
            } else if (K <= 8) {
                cond_img_size = 384 * 48 / 64;
            } else {
                cond_img_size = 384 / 2;
            }

            for (const auto& ref_image : ref_images) {
                auto patch_img = resize_to_area(ref_image, max_size);
                patch_img      = sd::ops::clamp(patch_img, 0.0f, 1.0f);
                patch_img      = patch_img * 2.0f - 1.0f;
                result.c_ref_images.push_back(std::move(patch_img));

                auto dims            = calculate_dimensions(cond_img_size, static_cast<double>(ref_image.shape()[0]) / static_cast<double>(ref_image.shape()[1]));
                auto vlm_image       = clip_preprocess(ref_image, dims.first, dims.second);
                int64_t image_tokens = static_cast<int64_t>(dims.first / PATCH_SIZE) * static_cast<int64_t>(dims.second / PATCH_SIZE);
                int64_t prompt_start = static_cast<int64_t>(tokenizer.encode(prompt + "<|vision_start|>", nullptr).size());
                prompt += "<|vision_start|>";
                prompt += repeat_special_token("<|image_pad|>", image_tokens);
                prompt += "<|vision_end|>";
                image_ranges.push_back(static_cast<int32_t>(prompt_start));
                image_ranges.push_back(static_cast<int32_t>(image_tokens));
                result.c_vlm_images.push_back(std::move(vlm_image));
                image_grids.push_back({1, dims.second / PATCH_SIZE, dims.first / PATCH_SIZE});
                skip_vision_start.push_back(0);
            }

            prompt += conditioner_params.text;
            prompt += "<|im_end|>\n<|im_start|>assistant\n<|boi_token|><|tms_token|>";
            auto input_ids = tokenizer.encode(prompt, nullptr);

            std::vector<int32_t> input_ids_pad = input_ids;
            input_ids_pad.push_back(VISION_START_TOKEN_ID);
            input_ids_pad.insert(input_ids_pad.end(), target_image_len - 1, IMAGE_TOKEN_ID);
            image_grids.push_back({1, static_cast<int32_t>(height / PATCH_SIZE), static_cast<int32_t>(width / PATCH_SIZE)});
            skip_vision_start.push_back(1);

            int64_t total_ref_len = 0;
            for (const auto& ref_image : result.c_ref_images) {
                int64_t ref_len = static_cast<int64_t>(ref_image.shape()[0] / PATCH_SIZE) * static_cast<int64_t>(ref_image.shape()[1] / PATCH_SIZE);
                total_ref_len += ref_len;
                input_ids_pad.push_back(VISION_START_TOKEN_ID);
                input_ids_pad.insert(input_ids_pad.end(), ref_len - 1, IMAGE_TOKEN_ID);
                image_grids.push_back({1, static_cast<int32_t>(ref_image.shape()[1] / PATCH_SIZE), static_cast<int32_t>(ref_image.shape()[0] / PATCH_SIZE)});
                skip_vision_start.push_back(1);
            }

            std::vector<int32_t> token_types(input_ids_pad.size(), 0);
            int txt_seq_len = static_cast<int>(input_ids.size());
            int bgn         = txt_seq_len - TIMESTEP_TOKEN_NUM;
            int end         = bgn + static_cast<int>(target_image_len) + TIMESTEP_TOKEN_NUM;
            for (int i = bgn; i < end; ++i) {
                token_types[i] = 1;
            }
            for (int i = end; i < end + total_ref_len; ++i) {
                token_types[i] = 2;
            }
            for (int i = txt_seq_len - TIMESTEP_TOKEN_NUM; i < txt_seq_len; ++i) {
                token_types[i] = 3;
            }

            std::vector<int64_t> input_shape{static_cast<int64_t>(input_ids.size())};
            std::vector<int64_t> position_shape{static_cast<int64_t>(input_ids_pad.size() * 4)};
            std::vector<int64_t> token_type_shape{static_cast<int64_t>(token_types.size())};
            std::vector<int64_t> image_range_shape{2, static_cast<int64_t>(image_ranges.size() / 2)};
            std::vector<int32_t> vinput_mask(token_types.size(), 0);
            for (int i = txt_seq_len; i < static_cast<int>(vinput_mask.size()); ++i) {
                vinput_mask[static_cast<size_t>(i)] = 1;
            }
            std::vector<int64_t> vinput_mask_shape{static_cast<int64_t>(vinput_mask.size())};

            result.c_input_ids          = sd::Tensor<int32_t>(input_shape, std::move(input_ids));
            result.c_position_ids       = sd::Tensor<int32_t>(position_shape, build_position_ids(input_ids_pad, image_grids, skip_vision_start));
            result.c_token_types        = sd::Tensor<int32_t>(token_type_shape, std::move(token_types));
            result.c_image_embed_ranges = sd::Tensor<int32_t>(image_range_shape, std::move(image_ranges));
            result.c_vinput_mask        = sd::Tensor<int32_t>(vinput_mask_shape, std::move(vinput_mask));
            return result;
        }
    };
}  // namespace HiDreamO1

#endif  // __SD_HIDREAM_O1_H__
