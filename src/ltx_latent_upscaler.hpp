#ifndef __SD_LTX_LATENT_UPSCALER_HPP__
#define __SD_LTX_LATENT_UPSCALER_HPP__

#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "common_dit.hpp"
#include "ggml_extend.hpp"
#include "ggml_graph_cut.h"
#include "model.h"
#include "util.h"

namespace LTXVUpsampler {
    constexpr int LTX_UPSAMPLER_GRAPH_SIZE = 10240;

    struct LatentUpsamplerConfig {
        int64_t in_channels      = 128;
        int64_t mid_channels     = 1024;
        int num_blocks_per_stage = 4;
        int dims                 = 3;
        bool spatial_upsample    = true;
        bool temporal_upsample   = false;
        bool rational_resampler  = false;
    };

    static inline bool has_tensor(const String2TensorStorage& tensor_storage_map,
                                  const std::string& name) {
        return tensor_storage_map.find(name) != tensor_storage_map.end();
    }

    static inline int64_t get_tensor_ne0(const String2TensorStorage& tensor_storage_map,
                                         const std::string& name,
                                         int64_t fallback) {
        auto it = tensor_storage_map.find(name);
        if (it == tensor_storage_map.end()) {
            return fallback;
        }
        return it->second.ne[0];
    }

    static inline int count_module_blocks(const String2TensorStorage& tensor_storage_map,
                                          const std::string& module_name) {
        int max_block            = -1;
        const std::string prefix = module_name + ".";
        for (const auto& pair : tensor_storage_map) {
            const std::string& name = pair.first;
            if (name.find(prefix) != 0) {
                continue;
            }
            size_t begin = prefix.size();
            size_t end   = name.find('.', begin);
            if (end == std::string::npos) {
                continue;
            }
            int index = atoi(name.substr(begin, end - begin).c_str());
            max_block = std::max(max_block, index);
        }
        return max_block + 1;
    }

    static inline LatentUpsamplerConfig detect_config_from_weights(const String2TensorStorage& tensor_storage_map) {
        LatentUpsamplerConfig config;
        config.mid_channels = get_tensor_ne0(tensor_storage_map, "initial_norm.weight", config.mid_channels);
        config.in_channels  = get_tensor_ne0(tensor_storage_map, "final_conv.bias", config.in_channels);
        int detected_blocks = count_module_blocks(tensor_storage_map, "res_blocks");
        if (detected_blocks > 0) {
            config.num_blocks_per_stage = detected_blocks;
        }
        config.spatial_upsample  = has_tensor(tensor_storage_map, "upsampler.0.weight");
        config.temporal_upsample = has_tensor(tensor_storage_map, "temporal_upsampler.0.weight");
        return config;
    }

    class VideoGroupNorm : public GGMLBlock {
    protected:
        int num_groups;
        int64_t num_channels;
        float eps;
        std::string prefix;

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            SD_UNUSED(tensor_storage_map);
            this->prefix     = prefix;
            params["weight"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_channels);
            params["bias"]   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_channels);
        }

    public:
        VideoGroupNorm(int num_groups, int64_t num_channels, float eps = 1e-05f)
            : num_groups(num_groups),
              num_channels(num_channels),
              eps(eps) {}

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            // LTX video latent layout is [W, H, T, C]. ggml_group_norm treats ne[2]
            // as channels, so fold only H/T internally and restore the same layout.
            GGML_ASSERT(x->ne[3] == num_channels);
            const int64_t W = x->ne[0];
            const int64_t H = x->ne[1];
            const int64_t T = x->ne[2];
            x               = ggml_ext_cont(ctx->ggml_ctx, x);
            x               = ggml_reshape_4d(ctx->ggml_ctx, x, W, H * T, num_channels, 1);
            x               = ggml_group_norm(ctx->ggml_ctx, x, num_groups, eps);

            ggml_tensor* weight = params["weight"];
            ggml_tensor* bias   = params["bias"];
            if (ctx->weight_adapter) {
                weight = ctx->weight_adapter->patch_weight(ctx->ggml_ctx, ctx->backend, weight, prefix + "weight");
                bias   = ctx->weight_adapter->patch_weight(ctx->ggml_ctx, ctx->backend, bias, prefix + "bias");
            }
            weight = ggml_reshape_4d(ctx->ggml_ctx, weight, 1, 1, num_channels, 1);
            bias   = ggml_reshape_4d(ctx->ggml_ctx, bias, 1, 1, num_channels, 1);
            x      = ggml_mul_inplace(ctx->ggml_ctx, x, weight);
            x      = ggml_add_inplace(ctx->ggml_ctx, x, bias);
            return ggml_reshape_4d(ctx->ggml_ctx, x, W, H, T, num_channels);
        }
    };

    class ResBlock : public GGMLBlock {
    public:
        ResBlock(int64_t channels, int dims = 3) {
            GGML_ASSERT(dims == 3);
            blocks["conv1"] = std::shared_ptr<GGMLBlock>(new Conv3d(channels, channels, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}));
            blocks["norm1"] = std::shared_ptr<GGMLBlock>(new VideoGroupNorm(32, channels));
            blocks["conv2"] = std::shared_ptr<GGMLBlock>(new Conv3d(channels, channels, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}));
            blocks["norm2"] = std::shared_ptr<GGMLBlock>(new VideoGroupNorm(32, channels));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto conv1 = std::dynamic_pointer_cast<Conv3d>(blocks["conv1"]);
            auto norm1 = std::dynamic_pointer_cast<VideoGroupNorm>(blocks["norm1"]);
            auto conv2 = std::dynamic_pointer_cast<Conv3d>(blocks["conv2"]);
            auto norm2 = std::dynamic_pointer_cast<VideoGroupNorm>(blocks["norm2"]);

            ggml_tensor* residual = x;

            x = conv1->forward(ctx, x);
            x = norm1->forward(ctx, x);
            x = ggml_silu_inplace(ctx->ggml_ctx, x);
            x = conv2->forward(ctx, x);
            x = norm2->forward(ctx, x);
            x = ggml_add(ctx->ggml_ctx, x, residual);
            return ggml_silu(ctx->ggml_ctx, x);
        }
    };

    class PixelShuffleND : public UnaryBlock {
    protected:
        int upscale_factor;

    public:
        explicit PixelShuffleND(int upscale_factor)
            : upscale_factor(upscale_factor) {}

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            GGML_ASSERT(upscale_factor == 2);
            int64_t h = x->ne[1];
            int64_t w = x->ne[0];
            // x: [b*f, c*4, h, w] -> [b*f, c, h*2, w*2]
            x = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 2, 0, 1, 3));  // [b*f, h, w, c*4]
            x = ggml_reshape_3d(ctx->ggml_ctx, x, x->ne[0], x->ne[1] * x->ne[2], x->ne[3]);          // [b*f, h*w, c*4]
            return DiT::unpatchify(ctx->ggml_ctx, x, h, w, upscale_factor, upscale_factor, true);
        }
    };

    class LatentUpsampler : public GGMLBlock {
    public:
        LatentUpsamplerConfig config;

        explicit LatentUpsampler(LatentUpsamplerConfig config)
            : config(std::move(config)) {
            GGML_ASSERT(this->config.dims == 3);
            GGML_ASSERT(this->config.spatial_upsample);
            GGML_ASSERT(!this->config.temporal_upsample);
            GGML_ASSERT(!this->config.rational_resampler);

            blocks["initial_conv"] = std::shared_ptr<GGMLBlock>(new Conv3d(this->config.in_channels,
                                                                           this->config.mid_channels,
                                                                           {3, 3, 3},
                                                                           {1, 1, 1},
                                                                           {1, 1, 1}));
            blocks["initial_norm"] = std::shared_ptr<GGMLBlock>(new VideoGroupNorm(32, this->config.mid_channels));
            for (int i = 0; i < this->config.num_blocks_per_stage; ++i) {
                blocks["res_blocks." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new ResBlock(this->config.mid_channels, this->config.dims));
            }
            blocks["upsampler.0"] = std::shared_ptr<GGMLBlock>(new Conv2d(this->config.mid_channels,
                                                                          4 * this->config.mid_channels,
                                                                          {3, 3},
                                                                          {1, 1},
                                                                          {1, 1}));
            blocks["upsampler.1"] = std::shared_ptr<GGMLBlock>(new PixelShuffleND(2));
            for (int i = 0; i < this->config.num_blocks_per_stage; ++i) {
                blocks["post_upsample_res_blocks." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new ResBlock(this->config.mid_channels, this->config.dims));
            }
            blocks["final_conv"] = std::shared_ptr<GGMLBlock>(new Conv3d(this->config.mid_channels,
                                                                         this->config.in_channels,
                                                                         {3, 3, 3},
                                                                         {1, 1, 1},
                                                                         {1, 1, 1}));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            // x: [b*c, f, h, w]
            // return: [b*c, f, h*2, w*2]
            auto initial_conv  = std::dynamic_pointer_cast<Conv3d>(blocks["initial_conv"]);
            auto initial_norm  = std::dynamic_pointer_cast<VideoGroupNorm>(blocks["initial_norm"]);
            auto upsample_conv = std::dynamic_pointer_cast<Conv2d>(blocks["upsampler.0"]);
            auto pixel_shuffle = std::dynamic_pointer_cast<PixelShuffleND>(blocks["upsampler.1"]);
            auto final_conv    = std::dynamic_pointer_cast<Conv3d>(blocks["final_conv"]);

            x = initial_conv->forward(ctx, x);
            x = initial_norm->forward(ctx, x);
            x = ggml_silu(ctx->ggml_ctx, x);
            sd::ggml_graph_cut::mark_graph_cut(x, "ltx_latent_upsampler.initial", "x");

            for (int i = 0; i < config.num_blocks_per_stage; ++i) {
                auto block = std::dynamic_pointer_cast<ResBlock>(blocks["res_blocks." + std::to_string(i)]);
                x          = block->forward(ctx, x);
                sd::ggml_graph_cut::mark_graph_cut(x, "ltx_latent_upsampler.res_blocks." + std::to_string(i), "x");
            }

            // rearrange(x, "b c f h w -> (b f) c h w"),
            x = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 0, 1, 3, 2));  // [b*f, c, h, w]
            x = upsample_conv->forward(ctx, x);                                                      // [b*f, c*4, h, w]
            x = pixel_shuffle->forward(ctx, x);                                                      // [b*f, c, h*2, w*2]
            x = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 0, 1, 3, 2));  // [b*c, f, h, w]
            sd::ggml_graph_cut::mark_graph_cut(x, "ltx_latent_upsampler.spatial_up", "x");

            for (int i = 0; i < config.num_blocks_per_stage; ++i) {
                auto block = std::dynamic_pointer_cast<ResBlock>(blocks["post_upsample_res_blocks." + std::to_string(i)]);
                x          = block->forward(ctx, x);
                sd::ggml_graph_cut::mark_graph_cut(x, "ltx_latent_upsampler.post_blocks." + std::to_string(i), "x");
            }

            x = final_conv->forward(ctx, x);
            sd::ggml_graph_cut::mark_graph_cut(x, "ltx_latent_upsampler.final", "x");
            return x;
        }
    };

    struct LatentUpsamplerRunner : public GGMLRunner {
        std::unique_ptr<LatentUpsampler> model;

        LatentUpsamplerRunner(ggml_backend_t backend,
                              ggml_backend_t params_backend)
            : GGMLRunner(backend, params_backend) {}

        std::string get_desc() override {
            return "ltx_latent_upsampler";
        }

        bool load_from_file(const std::string& file_path, int n_threads) {
            LOG_INFO("loading LTX latent upsampler from '%s'", file_path.c_str());
            ModelLoader model_loader;
            if (!model_loader.init_from_file(file_path)) {
                LOG_ERROR("init LTX latent upsampler model loader from file failed: '%s'", file_path.c_str());
                return false;
            }

            const auto& tensor_storage_map = model_loader.get_tensor_storage_map();
            if (!has_tensor(tensor_storage_map, "post_upsample_res_blocks.0.conv2.bias") ||
                !has_tensor(tensor_storage_map, "upsampler.0.weight")) {
                LOG_ERROR("unsupported LTX latent upsampler weights: expected spatial upsampler tensors");
                return false;
            }

            LatentUpsamplerConfig config = detect_config_from_weights(tensor_storage_map);
            if (config.dims != 3 || !config.spatial_upsample || config.temporal_upsample ||
                config.rational_resampler) {
                LOG_ERROR("unsupported LTX latent upsampler config: dims=%d spatial=%d temporal=%d rational=%d",
                          config.dims,
                          config.spatial_upsample,
                          config.temporal_upsample,
                          config.rational_resampler);
                return false;
            }

            model = std::make_unique<LatentUpsampler>(config);
            model->init(params_ctx, tensor_storage_map, "");
            if (!alloc_params_buffer()) {
                LOG_ERROR("LTX latent upsampler params buffer allocation failed");
                return false;
            }

            std::map<std::string, ggml_tensor*> tensors;
            model->get_param_tensors(tensors);
            if (!model_loader.load_tensors(tensors, {}, n_threads)) {
                LOG_ERROR("load LTX latent upsampler tensors failed");
                return false;
            }

            LOG_INFO("LTX latent upsampler loaded: in_channels=%" PRId64 ", mid_channels=%" PRId64 ", blocks=%d",
                     config.in_channels,
                     config.mid_channels,
                     config.num_blocks_per_stage);
            return true;
        }

        ggml_cgraph* build_graph(const sd::Tensor<float>& x_tensor) {
            if (!model) {
                return nullptr;
            }
            ggml_cgraph* gf  = new_graph_custom(LTX_UPSAMPLER_GRAPH_SIZE);
            ggml_tensor* x   = make_input(x_tensor);
            auto runner_ctx  = get_context();
            ggml_tensor* out = model->forward(&runner_ctx, x);
            ggml_build_forward_expand(gf, out);
            return gf;
        }

        sd::Tensor<float> compute(const int n_threads,
                                  const sd::Tensor<float>& x) {
            if (!model) {
                LOG_ERROR("LTX latent upsampler is not loaded");
                return {};
            }
            if (x.dim() != 4 && x.dim() != 5) {
                LOG_ERROR("LTX latent upsampler expects 4D or 5D video latent, got dim=%lld",
                          (long long)x.dim());
                return {};
            }
            if (x.dim() == 5 && x.shape()[4] != 1) {
                LOG_ERROR("LTX latent upsampler currently supports batch size 1, got batch=%lld",
                          (long long)x.shape()[4]);
                return {};
            }
            if (x.shape()[3] != model->config.in_channels) {
                LOG_ERROR("LTX latent upsampler expected %" PRId64 " channels, got %lld",
                          model->config.in_channels,
                          (long long)x.shape()[3]);
                return {};
            }
            size_t expected_dim = static_cast<size_t>(x.dim());
            auto get_graph      = [&]() -> ggml_cgraph* { return build_graph(x); };
            return restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, false), expected_dim);
        }
    };

}  // namespace LTXVUpsampler

#endif  // __SD_LTX_LATENT_UPSCALER_HPP__
