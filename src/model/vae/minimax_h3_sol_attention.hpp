#ifndef __SD_MODEL_VAE_MINIMAX_H3_SOL_ATTENTION_HPP__
#define __SD_MODEL_VAE_MINIMAX_H3_SOL_ATTENTION_HPP__

#include <algorithm>
#include <map>
#include <utility>
#include <vector>

#include "core/ggml_extend.h"
#include "core/tensor.hpp"

namespace MiniMaxH3VAE {

    constexpr int64_t sol_window_size    = 16;
    constexpr int64_t sol_block_size     = 5;
    constexpr int64_t sol_global_rows    = 5;
    constexpr int64_t sol_spatial_shift  = sol_window_size / 2;
    constexpr int64_t sol_temporal_shift = 2;

    struct SolAttentionPlan {
        struct Batch {
            sd::Tensor<int32_t> queries;
            sd::Tensor<int32_t> keys;
        };

        std::vector<Batch> batches;

        SolAttentionPlan(int64_t width, int64_t height, int64_t frames, bool shifted) {
            const int64_t frame_size = width * height;
            const int64_t pixels     = frames * frame_size;
            const int64_t blocks     = (frames + sol_block_size - 1) / sol_block_size;
            const int64_t shift      = shifted ? sol_spatial_shift : 0;
            const int64_t time_shift = shifted ? sol_temporal_shift : 0;

            auto global_rows = [&](int64_t block, std::vector<int32_t>& rows) {
                for (int64_t i = 0; i < sol_global_rows; ++i) {
                    rows.push_back(static_cast<int32_t>(pixels + block * sol_global_rows + i));
                }
            };
            using Window = std::pair<std::vector<int32_t>, std::vector<int32_t>>;
            std::map<std::pair<int64_t, int64_t>, std::vector<Window>> groups;
            auto add = [&](std::vector<int32_t> queries, std::vector<int32_t> keys) {
                groups[{queries.size(), keys.size()}].emplace_back(std::move(queries), std::move(keys));
            };

            for (int64_t t = -time_shift; t < frames; t += sol_block_size) {
                const int64_t first = std::max<int64_t>(0, t);
                const int64_t end   = std::min(frames, t + sol_block_size);
                for (int64_t y = -shift; y < height; y += sol_window_size) {
                    for (int64_t x = -shift; x < width; x += sol_window_size) {
                        std::vector<int32_t> queries;
                        for (int64_t i = first; i < end; ++i) {
                            for (int64_t j = std::max<int64_t>(0, y); j < std::min(height, y + sol_window_size); ++j) {
                                for (int64_t k = std::max<int64_t>(0, x); k < std::min(width, x + sol_window_size); ++k) {
                                    queries.push_back(static_cast<int32_t>((i * height + j) * width + k));
                                }
                            }
                        }
                        auto keys = queries;
                        // A shifted window can overlap two native blocks and read both sets of global tokens.
                        for (int64_t block = first / sol_block_size; block <= (end - 1) / sol_block_size; ++block) {
                            global_rows(block, keys);
                        }
                        add(std::move(queries), std::move(keys));
                    }
                }
            }
            for (int64_t block = 0; block < blocks; ++block) {
                std::vector<int32_t> queries;
                global_rows(block, queries);
                const int64_t first = block * sol_block_size * frame_size;
                const int64_t end   = std::min(pixels, first + sol_block_size * frame_size);
                std::vector<int32_t> keys;
                keys.reserve(static_cast<size_t>(end - first + sol_global_rows));
                for (int64_t i = first; i < end; ++i) {
                    keys.push_back(static_cast<int32_t>(i));
                }
                keys.insert(keys.end(), queries.begin(), queries.end());
                add(std::move(queries), std::move(keys));
            }

            // Equal-length windows can share an attention launch without padding or a mask.
            // Bound the batch so the unfused attention fallback also has bounded workspace.
            constexpr size_t max_windows = 4;
            for (const auto& [lengths, windows] : groups) {
                const auto [query_count, key_count] = lengths;
                for (size_t first = 0; first < windows.size(); first += max_windows) {
                    const int64_t count = static_cast<int64_t>(std::min(max_windows, windows.size() - first));
                    Batch batch{sd::Tensor<int32_t>({query_count, count}), sd::Tensor<int32_t>({key_count, count})};
                    for (int64_t i = 0; i < count; ++i) {
                        const auto& [queries, keys] = windows[first + i];
                        std::copy(queries.begin(), queries.end(), batch.queries.data() + i * query_count);
                        std::copy(keys.begin(), keys.end(), batch.keys.data() + i * key_count);
                    }
                    batches.push_back(std::move(batch));
                }
            }
        }

        struct Inputs {
            std::vector<std::pair<ggml_tensor*, ggml_tensor*>> batches;

            ggml_tensor* forward(GGMLRunnerContext* runner_ctx,
                                 ggml_tensor* q,
                                 ggml_tensor* k,
                                 ggml_tensor* v,
                                 int64_t heads) const {
                auto ctx          = runner_ctx->ggml_ctx;
                const int64_t dim = q->ne[0];
                // The batches write each query row exactly once.
                auto output = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, q->ne[1]);
                auto gather = [&](ggml_tensor* value, ggml_tensor* indices) {
                    auto rows = ggml_reshape_1d(ctx, indices, ggml_nelements(indices));
                    return ggml_reshape_3d(ctx, ggml_get_rows(ctx, value, rows), dim, indices->ne[0], indices->ne[1]);
                };
                for (const auto& [queries, keys] : batches) {
                    auto attended = ggml_ext_attention_ext(runner_ctx,
                                                           gather(q, queries),
                                                           gather(k, keys),
                                                           gather(v, keys),
                                                           heads,
                                                           nullptr,
                                                           false,
                                                           runner_ctx->flash_attn_enabled);
                    auto indices  = ggml_reshape_1d(ctx, queries, ggml_nelements(queries));
                    attended      = ggml_reshape_2d(ctx, attended, dim, indices->ne[0]);
                    output        = ggml_set_rows(ctx, output, attended, indices);
                }
                return output;
            }
        };

        template <typename BindInput>
        Inputs bind(BindInput bind_input) const {
            Inputs inputs;
            for (const auto& batch : batches) {
                inputs.batches.emplace_back(bind_input(batch.queries), bind_input(batch.keys));
            }
            return inputs;
        }
    };

}  // namespace MiniMaxH3VAE

#endif  // __SD_MODEL_VAE_MINIMAX_H3_SOL_ATTENTION_HPP__
