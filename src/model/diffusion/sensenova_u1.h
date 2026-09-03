#ifndef __SD_MODEL_DIFFUSION_SENSENOVA_U1_H__
#define __SD_MODEL_DIFFUSION_SENSENOVA_U1_H__

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>

#include "core/ggml_extend.hpp"
#include "model/diffusion/dit.hpp"
#include "model_loader.h"

namespace SenseNovaU1 {
    constexpr int SENSENOVA_U1_GRAPH_SIZE = 327680;

    struct SenseNovaU1Config {
        int64_t hidden_size                  = 4096;
        int64_t intermediate_size            = 12288;
        int64_t num_layers                   = 42;
        int64_t num_heads                    = 32;
        int64_t num_kv_heads                 = 8;
        int64_t head_dim                     = 128;
        int64_t vocab_size                   = 151936;
        int64_t max_position_embeddings      = 262144;
        int64_t max_position_embeddings_hw   = 10000;
        int64_t vision_hidden_size           = 1024;
        int64_t patch_size                   = 16;
        int64_t vision_downsample_factor     = 2;
        int64_t in_channels                  = 3;
        int64_t timestep_embedding_size      = 256;
        float rms_norm_eps                   = 1e-6f;
        float rope_theta                     = 5000000.f;
        float rope_theta_hw                  = 10000.f;
        float noise_scale_base_image_seq_len = 64.f;
        float noise_scale_max_value          = 16.f;
        float t_eps                          = 0.02f;
        bool add_noise_scale_embedding       = true;

        int64_t image_token_stride() const {
            return patch_size * vision_downsample_factor;
        }

        static SenseNovaU1Config detect_from_weights(const String2TensorStorage& tensor_storage_map,
                                                     const std::string& prefix) {
            SenseNovaU1Config config;
            config.num_layers      = 0;
            const std::string root = prefix.empty() ? "" : prefix + ".";

            for (const auto& [name, tensor_storage] : tensor_storage_map) {
                if (!starts_with(name, root)) {
                    continue;
                }
                if (ends_with(name, "language_model.model.embed_tokens.weight") && tensor_storage.n_dims == 2) {
                    config.hidden_size = tensor_storage.ne[0];
                    config.vocab_size  = tensor_storage.ne[1];
                } else if (ends_with(name, "language_model.model.layers.0.mlp.gate_proj.weight") && tensor_storage.n_dims == 2) {
                    config.intermediate_size = tensor_storage.ne[1];
                } else if (ends_with(name, "language_model.model.layers.0.self_attn.q_proj.weight") && tensor_storage.n_dims == 2) {
                    config.num_heads = tensor_storage.ne[1] / config.head_dim;
                } else if (ends_with(name, "language_model.model.layers.0.self_attn.k_proj.weight") && tensor_storage.n_dims == 2) {
                    config.num_kv_heads = tensor_storage.ne[1] / config.head_dim;
                } else if (ends_with(name, "fm_modules.vision_model_mot_gen.embeddings.patch_embedding.weight") && tensor_storage.n_dims == 4) {
                    config.patch_size         = tensor_storage.ne[0];
                    config.in_channels        = tensor_storage.ne[2];
                    config.vision_hidden_size = tensor_storage.ne[3];
                } else if (ends_with(name, "fm_modules.vision_model_mot_gen.embeddings.dense_embedding.weight") && tensor_storage.n_dims == 4) {
                    config.vision_downsample_factor = tensor_storage.ne[0];
                }

                const std::string layer_prefix = root + "language_model.model.layers.";
                if (starts_with(name, layer_prefix)) {
                    const char* index_begin = name.c_str() + layer_prefix.size();
                    config.num_layers       = std::max<int64_t>(config.num_layers, std::strtoll(index_begin, nullptr, 10) + 1);
                }
            }

            if (config.num_layers == 0) {
                config.num_layers = 42;
            }
            config.add_noise_scale_embedding = tensor_storage_map.find(root + "fm_modules.noise_scale_embedder.mlp.0.weight") != tensor_storage_map.end();

            LOG_DEBUG("sensenova-u1.5: layers=%" PRId64 ", hidden=%" PRId64 ", intermediate=%" PRId64 ", heads=%" PRId64 ", kv_heads=%" PRId64 ", patch=%" PRId64 "x%" PRId64,
                      config.num_layers,
                      config.hidden_size,
                      config.intermediate_size,
                      config.num_heads,
                      config.num_kv_heads,
                      config.patch_size,
                      config.vision_downsample_factor);
            return config;
        }
    };

    class StorageConv2d : public Conv2d {
    protected:
        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            this->prefix     = prefix;
            ggml_type wtype  = get_type(prefix + "weight", tensor_storage_map, GGML_TYPE_F16);
            params["weight"] = ggml_new_tensor_4d(ctx,
                                                  wtype,
                                                  kernel_size.second,
                                                  kernel_size.first,
                                                  in_channels,
                                                  out_channels);
            if (bias) {
                params["bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
            }
        }

    public:
        StorageConv2d(int64_t in_channels,
                      int64_t out_channels,
                      std::pair<int, int> kernel_size,
                      std::pair<int, int> stride  = {1, 1},
                      std::pair<int, int> padding = {0, 0},
                      bool bias                   = true)
            : Conv2d(in_channels,
                     out_channels,
                     kernel_size,
                     stride,
                     padding,
                     {1, 1},
                     bias) {}
    };

    struct TimestepEmbedder : public GGMLBlock {
        int64_t frequency_embedding_size;

        TimestepEmbedder(int64_t hidden_size, int64_t frequency_embedding_size = 256)
            : frequency_embedding_size(frequency_embedding_size) {
            blocks["mlp.0"] = std::make_shared<Linear>(frequency_embedding_size, hidden_size, true);
            blocks["mlp.2"] = std::make_shared<Linear>(hidden_size, hidden_size, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* timesteps) {
            auto mlp_0 = std::dynamic_pointer_cast<Linear>(blocks["mlp.0"]);
            auto mlp_2 = std::dynamic_pointer_cast<Linear>(blocks["mlp.2"]);
            auto x     = ggml_ext_timestep_embedding(ctx->ggml_ctx,
                                                     timesteps,
                                                     static_cast<int>(frequency_embedding_size),
                                                     10000.f,
                                                     1.f);
            x          = mlp_0->forward(ctx, x);
            x          = ggml_silu_inplace(ctx->ggml_ctx, x);
            return mlp_2->forward(ctx, x);
        }
    };

    inline ggml_tensor* apply_vision_rope(GGMLRunnerContext* ctx,
                                          ggml_tensor* x,
                                          ggml_tensor* position_x,
                                          ggml_tensor* position_y,
                                          float theta,
                                          int max_position) {
        GGML_ASSERT(x->ne[0] % 2 == 0);
        const int64_t half = x->ne[0] / 2;
        auto x_part        = ggml_ext_slice(ctx->ggml_ctx, x, 0, 0, half);
        auto y_part        = ggml_ext_slice(ctx->ggml_ctx, x, 0, half, x->ne[0]);
        x_part             = ggml_rope_ext(ctx->ggml_ctx,
                                           x_part,
                                           position_x,
                                           nullptr,
                                           static_cast<int>(half),
                                           GGML_ROPE_TYPE_NORMAL,
                                           max_position,
                                           theta,
                                           1.f,
                                           0.f,
                                           1.f,
                                           32.f,
                                           1.f);
        y_part             = ggml_rope_ext(ctx->ggml_ctx,
                                           y_part,
                                           position_y,
                                           nullptr,
                                           static_cast<int>(half),
                                           GGML_ROPE_TYPE_NORMAL,
                                           max_position,
                                           theta,
                                           1.f,
                                           0.f,
                                           1.f,
                                           32.f,
                                           1.f);
        return ggml_concat(ctx->ggml_ctx, x_part, y_part, 0);
    }

    struct VisionEmbeddings : public GGMLBlock {
        SenseNovaU1Config config;

        explicit VisionEmbeddings(const SenseNovaU1Config& config)
            : config(config) {
            blocks["patch_embedding"] = std::make_shared<StorageConv2d>(config.in_channels,
                                                                        config.vision_hidden_size,
                                                                        std::pair<int, int>{static_cast<int>(config.patch_size), static_cast<int>(config.patch_size)},
                                                                        std::pair<int, int>{static_cast<int>(config.patch_size), static_cast<int>(config.patch_size)},
                                                                        std::pair<int, int>{0, 0},
                                                                        true);
            blocks["dense_embedding"] = std::make_shared<StorageConv2d>(config.vision_hidden_size,
                                                                        config.hidden_size,
                                                                        std::pair<int, int>{static_cast<int>(config.vision_downsample_factor), static_cast<int>(config.vision_downsample_factor)},
                                                                        std::pair<int, int>{static_cast<int>(config.vision_downsample_factor), static_cast<int>(config.vision_downsample_factor)},
                                                                        std::pair<int, int>{0, 0},
                                                                        true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* image,
                             ggml_tensor* position_x,
                             ggml_tensor* position_y) {
            auto patch_embedding = std::dynamic_pointer_cast<StorageConv2d>(blocks["patch_embedding"]);
            auto dense_embedding = std::dynamic_pointer_cast<StorageConv2d>(blocks["dense_embedding"]);

            auto x = patch_embedding->forward(ctx, image);
            x      = ggml_gelu_erf(ctx->ggml_ctx, x);

            const int64_t grid_w = x->ne[0];
            const int64_t grid_h = x->ne[1];
            const int64_t batch  = x->ne[3];
            x                    = ggml_reshape_3d(ctx->ggml_ctx, x, grid_w * grid_h, x->ne[2], batch);
            x                    = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 1, 0, 2, 3));
            x                    = apply_vision_rope(ctx,
                                                     x,
                                                     position_x,
                                                     position_y,
                                                     config.rope_theta_hw,
                                                     static_cast<int>(config.max_position_embeddings_hw));
            x                    = ggml_reshape_4d(ctx->ggml_ctx, x, config.vision_hidden_size, grid_w, grid_h, batch);
            x                    = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 1, 2, 0, 3));
            x                    = dense_embedding->forward(ctx, x);

            const int64_t token_w = x->ne[0];
            const int64_t token_h = x->ne[1];
            x                     = ggml_reshape_3d(ctx->ggml_ctx, x, token_w * token_h, x->ne[2], x->ne[3]);
            return ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 1, 0, 2, 3));
        }
    };

    inline ggml_tensor* pixel_shuffle(GGMLRunnerContext* ctx,
                                      ggml_tensor* x,
                                      int upscale_factor) {
        GGML_ASSERT(upscale_factor > 0);
        const int64_t h = x->ne[1];
        const int64_t w = x->ne[0];
        GGML_ASSERT(x->ne[2] % (upscale_factor * upscale_factor) == 0);
        x = ggml_ext_cont(ctx->ggml_ctx,
                          ggml_ext_torch_permute(ctx->ggml_ctx, x, 2, 0, 1, 3));
        x = ggml_reshape_3d(ctx->ggml_ctx, x, x->ne[0], x->ne[1] * x->ne[2], x->ne[3]);
        return DiT::unpatchify(ctx->ggml_ctx, x, h, w, upscale_factor, upscale_factor, true);
    }

    struct PixelDecoder : public GGMLBlock {
        explicit PixelDecoder(const SenseNovaU1Config& config) {
            blocks["conv1"] = std::make_shared<StorageConv2d>(config.hidden_size / 4,
                                                              1024,
                                                              std::pair<int, int>{3, 3},
                                                              std::pair<int, int>{1, 1},
                                                              std::pair<int, int>{1, 1},
                                                              true);
            blocks["conv2"] = std::make_shared<StorageConv2d>(256,
                                                              192,
                                                              std::pair<int, int>{3, 3},
                                                              std::pair<int, int>{1, 1},
                                                              std::pair<int, int>{1, 1},
                                                              true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto conv1 = std::dynamic_pointer_cast<StorageConv2d>(blocks["conv1"]);
            auto conv2 = std::dynamic_pointer_cast<StorageConv2d>(blocks["conv2"]);
            x          = pixel_shuffle(ctx, x, 2);
            x          = conv1->forward(ctx, x);
            x          = ggml_gelu_erf(ctx->ggml_ctx, x);
            x          = pixel_shuffle(ctx, x, 2);
            x          = conv2->forward(ctx, x);
            return pixel_shuffle(ctx, x, 8);
        }
    };
}  // namespace SenseNovaU1

#endif  // __SD_MODEL_DIFFUSION_SENSENOVA_U1_H__
