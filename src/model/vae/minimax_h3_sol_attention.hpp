#ifndef __SD_MODEL_VAE_MINIMAX_H3_SOL_ATTENTION_HPP__
#define __SD_MODEL_VAE_MINIMAX_H3_SOL_ATTENTION_HPP__

#include <algorithm>
#include <map>
#include <tuple>
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
            int64_t query_offset;
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
            std::map<std::tuple<int64_t, int64_t, int64_t>, std::vector<Window>> groups;
            auto add = [&](std::vector<int32_t> queries, std::vector<int32_t> keys, int64_t query_offset) {
                groups[{queries.size(), keys.size(), query_offset}].emplace_back(std::move(queries), std::move(keys));
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
                        add(std::move(queries), std::move(keys), 0);
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
                add(std::move(queries), std::move(keys), end - first);
            }

            // Equal-length windows can share an attention launch without padding or a mask.
            // Bound the batch so the unfused attention fallback also has bounded workspace.
            for (const auto& [lengths, windows] : groups) {
                const auto [query_count, key_count, query_offset] = lengths;
                const size_t max_windows                          = query_offset == 0 ? 4 : 1;
                for (size_t first = 0; first < windows.size(); first += max_windows) {
                    const int64_t count = static_cast<int64_t>(std::min(max_windows, windows.size() - first));
                    Batch batch{sd::Tensor<int32_t>({query_count, count}), sd::Tensor<int32_t>({key_count, count}), query_offset};
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
            struct Batch {
                ggml_tensor* queries;
                ggml_tensor* keys;
                int64_t query_offset;
            };
            std::vector<Batch> batches;
            ggml_tensor* row_ids = nullptr;

            template <typename Project>
            ggml_tensor* forward(GGMLRunnerContext* runner_ctx,
                                 ggml_tensor* x,
                                 ggml_tensor* pe,
                                 int64_t heads,
                                 Project project) const {
                auto ctx          = runner_ctx->ggml_ctx;
                const int64_t dim = x->ne[0];
                // The batches write each query row exactly once.
                auto output    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, x->ne[1]);
                auto positions = ggml_reshape_2d(ctx, pe, pe->ne[0] * pe->ne[1] * pe->ne[2], pe->ne[3]);
                for (const auto& [queries, keys, query_offset] : batches) {
                    auto key_rows = ggml_reshape_1d(ctx, keys, ggml_nelements(keys));
                    auto local_pe = ggml_reshape_4d(ctx, ggml_get_rows(ctx, positions, key_rows),
                                                    pe->ne[0], pe->ne[1], pe->ne[2], key_rows->ne[0]);
                    // Only the rows read by this batch need projected attention data.
                    auto [q, k, v]  = project(ggml_get_rows(ctx, x, key_rows), local_pe);
                    auto as_windows = [&](ggml_tensor* value) {
                        return ggml_reshape_3d(ctx, value, dim, keys->ne[0], keys->ne[1]);
                    };
                    q             = ggml_ext_slice(runner_ctx, as_windows(q), 1, query_offset, query_offset + queries->ne[0]);
                    auto attended = ggml_ext_attention_ext(runner_ctx,
                                                           q,
                                                           as_windows(k),
                                                           as_windows(v),
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
                inputs.batches.push_back({bind_input(batch.queries), bind_input(batch.keys), batch.query_offset});
            }
            return inputs;
        }
    };

}  // namespace MiniMaxH3VAE

#endif  // __SD_MODEL_VAE_MINIMAX_H3_SOL_ATTENTION_HPP__
