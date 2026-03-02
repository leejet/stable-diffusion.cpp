#ifndef __ACE_VAE_HPP__
#define __ACE_VAE_HPP__

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "ggml_extend.hpp"
#include "vae.hpp"

class AudioResidualUnit : public UnaryBlock {
protected:
    int64_t in_channels;
    int64_t out_channels;
    int dilation;
    bool use_snake;

public:
    AudioResidualUnit(int64_t in_channels,
                      int64_t out_channels,
                      int dilation,
                      bool use_snake = true)
        : in_channels(in_channels),
          out_channels(out_channels),
          dilation(dilation),
          use_snake(use_snake) {
        (void)use_snake;
        int padding = (dilation * (7 - 1)) / 2;

        blocks["layers.0"] = std::make_shared<Snake1d>(out_channels, true);
        blocks["layers.1"] = std::make_shared<WNConv1d>(in_channels, out_channels, 7, 1, padding, dilation, true);
        blocks["layers.2"] = std::make_shared<Snake1d>(out_channels, true);
        blocks["layers.3"] = std::make_shared<WNConv1d>(out_channels, out_channels, 1, 1, 0, 1, true);
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        auto act1 = std::dynamic_pointer_cast<Snake1d>(blocks["layers.0"]);
        auto conv1 = std::dynamic_pointer_cast<WNConv1d>(blocks["layers.1"]);
        auto act2 = std::dynamic_pointer_cast<Snake1d>(blocks["layers.2"]);
        auto conv2 = std::dynamic_pointer_cast<WNConv1d>(blocks["layers.3"]);

        auto residual = x;
        x = act1->forward(ctx, x);
        x = conv1->forward(ctx, x);
        x = act2->forward(ctx, x);
        x = conv2->forward(ctx, x);
        x = ggml_add(ctx->ggml_ctx, x, residual);
        return x;
    }
};

class AudioEncoderBlock : public UnaryBlock {
public:
    AudioEncoderBlock(int64_t in_channels,
                      int64_t out_channels,
                      int stride,
                      bool use_snake = true) {
        int padding = static_cast<int>(std::ceil(stride / 2.0));
        int kernel_size = 2 * stride;

        blocks["layers.0"] = std::make_shared<AudioResidualUnit>(in_channels, in_channels, 1, use_snake);
        blocks["layers.1"] = std::make_shared<AudioResidualUnit>(in_channels, in_channels, 3, use_snake);
        blocks["layers.2"] = std::make_shared<AudioResidualUnit>(in_channels, in_channels, 9, use_snake);
        blocks["layers.3"] = std::make_shared<Snake1d>(in_channels, true);
        blocks["layers.4"] = std::make_shared<WNConv1d>(in_channels, out_channels, kernel_size, stride, padding, 1, true);
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        for (int i = 0; i < 5; ++i) {
            auto layer = std::dynamic_pointer_cast<UnaryBlock>(blocks["layers." + std::to_string(i)]);
            x = layer->forward(ctx, x);
        }
        return x;
    }
};

class AudioDecoderBlock : public UnaryBlock {
public:
    AudioDecoderBlock(int64_t in_channels,
                      int64_t out_channels,
                      int stride,
                      bool use_snake = true) {
        int padding = static_cast<int>(std::ceil(stride / 2.0));
        int kernel_size = 2 * stride;

        blocks["layers.0"] = std::make_shared<Snake1d>(in_channels, true);
        blocks["layers.1"] = std::make_shared<WNConvTranspose1d>(in_channels, out_channels, kernel_size, stride, padding, 1, true);
        blocks["layers.2"] = std::make_shared<AudioResidualUnit>(out_channels, out_channels, 1, use_snake);
        blocks["layers.3"] = std::make_shared<AudioResidualUnit>(out_channels, out_channels, 3, use_snake);
        blocks["layers.4"] = std::make_shared<AudioResidualUnit>(out_channels, out_channels, 9, use_snake);
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        for (int i = 0; i < 5; ++i) {
            auto layer = std::dynamic_pointer_cast<UnaryBlock>(blocks["layers." + std::to_string(i)]);
            x = layer->forward(ctx, x);
        }
        return x;
    }
};

class AudioOobleckEncoder : public UnaryBlock {
protected:
    int64_t in_channels;
    int64_t channels;
    int64_t latent_dim;
    std::vector<int> c_mults;
    std::vector<int> strides;
    int depth;
    int num_layers;

public:
    AudioOobleckEncoder(int64_t in_channels,
                        int64_t channels,
                        int64_t latent_dim,
                        const std::vector<int>& c_mults,
                        const std::vector<int>& strides,
                        bool use_snake = true)
        : in_channels(in_channels),
          channels(channels),
          latent_dim(latent_dim),
          c_mults(c_mults),
          strides(strides) {
        std::vector<int> c_mults_local = c_mults;
        c_mults_local.insert(c_mults_local.begin(), 1);
        depth = static_cast<int>(c_mults_local.size());

        int layer_idx = 0;
        blocks["layers." + std::to_string(layer_idx++)] = std::make_shared<WNConv1d>(in_channels, c_mults_local[0] * channels, 7, 1, 3, 1, true);

        for (int i = 0; i < depth - 1; ++i) {
            blocks["layers." + std::to_string(layer_idx++)] = std::make_shared<AudioEncoderBlock>(c_mults_local[i] * channels,
                                                                                                c_mults_local[i + 1] * channels,
                                                                                                strides[i],
                                                                                                use_snake);
        }

        blocks["layers." + std::to_string(layer_idx++)] = std::make_shared<Snake1d>(c_mults_local.back() * channels, true);
        blocks["layers." + std::to_string(layer_idx++)] = std::make_shared<WNConv1d>(c_mults_local.back() * channels, latent_dim, 3, 1, 1, 1, true);

        num_layers = layer_idx;
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        for (int i = 0; i < num_layers; ++i) {
            auto layer = std::dynamic_pointer_cast<UnaryBlock>(blocks["layers." + std::to_string(i)]);
            x = layer->forward(ctx, x);
        }
        return x;
    }
};

class AudioOobleckDecoder : public UnaryBlock {
protected:
    int64_t out_channels;
    int64_t channels;
    int64_t latent_dim;
    std::vector<int> c_mults;
    std::vector<int> strides;
    int depth;
    int num_layers;

public:
    AudioOobleckDecoder(int64_t out_channels,
                        int64_t channels,
                        int64_t latent_dim,
                        const std::vector<int>& c_mults,
                        const std::vector<int>& strides,
                        bool use_snake = true,
                        bool final_tanh = false)
        : out_channels(out_channels),
          channels(channels),
          latent_dim(latent_dim),
          c_mults(c_mults),
          strides(strides) {
        (void)final_tanh;
        std::vector<int> c_mults_local = c_mults;
        c_mults_local.insert(c_mults_local.begin(), 1);
        depth = static_cast<int>(c_mults_local.size());

        int layer_idx = 0;
        blocks["layers." + std::to_string(layer_idx++)] = std::make_shared<WNConv1d>(latent_dim, c_mults_local.back() * channels, 7, 1, 3, 1, true);

        for (int i = depth - 1; i > 0; --i) {
            blocks["layers." + std::to_string(layer_idx++)] = std::make_shared<AudioDecoderBlock>(c_mults_local[i] * channels,
                                                                                                c_mults_local[i - 1] * channels,
                                                                                                strides[i - 1],
                                                                                                use_snake);
        }

        blocks["layers." + std::to_string(layer_idx++)] = std::make_shared<Snake1d>(c_mults_local.front() * channels, true);
        blocks["layers." + std::to_string(layer_idx++)] = std::make_shared<WNConv1d>(c_mults_local.front() * channels, out_channels, 7, 1, 3, 1, false);
        blocks["layers." + std::to_string(layer_idx++)] = std::make_shared<Identity>();

        num_layers = layer_idx;
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        for (int i = 0; i < num_layers; ++i) {
            auto layer = std::dynamic_pointer_cast<UnaryBlock>(blocks["layers." + std::to_string(i)]);
            x = layer->forward(ctx, x);
        }
        return x;
    }
};

class AudioOobleckVAEModel : public GGMLBlock {
public:
    AudioOobleckVAEModel(int64_t in_channels,
                         int64_t channels,
                         int64_t latent_dim,
                         const std::vector<int>& c_mults,
                         const std::vector<int>& strides,
                         bool use_snake = true,
                         bool final_tanh = false) {
        blocks["encoder"] = std::make_shared<AudioOobleckEncoder>(in_channels, channels, latent_dim * 2, c_mults, strides, use_snake);
        blocks["decoder"] = std::make_shared<AudioOobleckDecoder>(in_channels, channels, latent_dim, c_mults, strides, use_snake, final_tanh);
    }

    ggml_tensor* encode(GGMLRunnerContext* ctx, ggml_tensor* x) {
        auto encoder = std::dynamic_pointer_cast<AudioOobleckEncoder>(blocks["encoder"]);
        return encoder->forward(ctx, x);
    }

    ggml_tensor* decode(GGMLRunnerContext* ctx, ggml_tensor* z) {
        auto decoder = std::dynamic_pointer_cast<AudioOobleckDecoder>(blocks["decoder"]);
        return decoder->forward(ctx, z);
    }
};

struct AudioOobleckVAE : public VAE {
    bool decode_only = true;
    std::shared_ptr<AudioOobleckVAEModel> vae;

    AudioOobleckVAE(ggml_backend_t backend,
                    bool offload_params_to_cpu,
                    const String2TensorStorage& tensor_storage_map,
                    const std::string& prefix,
                    bool decode_only = true)
        : decode_only(decode_only),
          VAE(backend, offload_params_to_cpu) {
        std::vector<int> strides = {2, 4, 4, 8, 8};
        std::string stride_key = prefix + ".decoder.layers.2.layers.1.weight_v";
        auto iter = tensor_storage_map.find(stride_key);
        if (iter == tensor_storage_map.end()) {
            stride_key = prefix + ".decoder.layers.2.layers.1.parametrizations.weight.original1";
            iter       = tensor_storage_map.find(stride_key);
        }
        if (iter != tensor_storage_map.end()) {
            int64_t k0 = iter->second.ne[0];
            int64_t k1 = iter->second.ne[1];
            int64_t k2 = iter->second.ne[2];
            int kernel_size = static_cast<int>(std::min(std::min(k0, k1), k2));
            if (kernel_size == 12) {
                strides = {2, 4, 4, 6, 10};
            }
        }
        vae = std::make_shared<AudioOobleckVAEModel>(2, 128, 64, std::vector<int>{1, 2, 4, 8, 16}, strides, true, false);
        vae->init(params_ctx, tensor_storage_map, prefix);
    }

    std::string get_desc() override {
        return "audio_oobleck_vae";
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) override {
        if (vae) {
            vae->get_param_tensors(tensors, prefix);
        }
    }

    struct ggml_cgraph* build_graph(struct ggml_tensor* z, bool decode_graph) {
        struct ggml_cgraph* gf = ggml_new_graph(compute_ctx);

        z = to_backend(z);
        auto runner_ctx = get_context();

        ggml_tensor* x = z;
        if (decode_graph) {
            // input: [C, T, B] -> [T, C, B]
            x = ggml_cont(compute_ctx, ggml_permute(compute_ctx, x, 1, 0, 2, 3));
            x = vae->decode(&runner_ctx, x);
        } else {
            x = ggml_cont(compute_ctx, ggml_permute(compute_ctx, x, 1, 0, 2, 3));
            x = vae->encode(&runner_ctx, x);
        }

        ggml_build_forward_expand(gf, x);
        return gf;
    }

    bool compute(const int n_threads,
                 struct ggml_tensor* z,
                 bool decode_graph,
                 struct ggml_tensor** output,
                 struct ggml_context* output_ctx = nullptr) override {
        GGML_ASSERT(!decode_only || decode_graph);
        auto get_graph = [&]() -> struct ggml_cgraph* {
            return build_graph(z, decode_graph);
        };
        return GGMLRunner::compute(get_graph, n_threads, false, output, output_ctx);
    }
};

#endif  // __ACE_VAE_HPP__
