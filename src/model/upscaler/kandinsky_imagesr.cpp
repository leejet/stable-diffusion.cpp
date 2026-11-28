#include "model/upscaler/kandinsky_imagesr.hpp"

#include <fstream>
#include "json.hpp"
#include "model/vae/auto_encoder_kl.hpp"
#include "model_io/safetensors_io.h"
#include "model_loader.h"
#include "runtime/tiling.h"

namespace KandinskyImageSR {
    static constexpr int tile_side     = 1024;
    static constexpr float timestep_x2 = 0.15f;
    static constexpr float timestep_x4 = 0.244f;

    static int timestep_row(const std::vector<float>& steps, float t) {
        for (size_t i = 0; i < steps.size(); ++i) {
            if (std::abs(steps[i] - t) <= 1e-6f) {
                return static_cast<int>(i);
            }
        }
        throw std::runtime_error("missing baked timestep " + std::to_string(t));
    }

    struct Bundle {
        std::string transformer_path;
        std::string vae_path;
        std::vector<int> axes;
        int layers;
        int64_t rows;
        int row_x2, row_x4;
    };

    static Bundle read_bundle(const std::string& path) {
        std::ifstream input(std::filesystem::u8path(path) / "transformer" / "config.json");
        const auto json  = nlohmann::json::parse(input);
        const auto steps = json.at("baked_timesteps").get<std::vector<float>>();
        const auto axes  = json.at("axes_dims").get<std::vector<int>>();
        if (json.at("_class_name") != "KandinskyImageSRBakedTransformer3DModel" ||
            json.at("attention_type") != "regular" || json.at("visual_cond") != false ||
            json.at("patch_size") != nlohmann::json::array({1, 2, 2}) || json.at("baked_latent_size") != tile_side / 8 ||
            json.at("in_visual_dim") != 16 || json.at("out_visual_dim") != 16 ||
            axes != std::vector<int>{32, 48, 48} || json.at("num_visual_blocks") != 50) {
            throw std::runtime_error("unsupported ImageSR transformer configuration");
        }
        auto transformer_path = path_join(path, "transformer/diffusion_pytorch_model.safetensors");
        if (file_exists(transformer_path + ".index.json")) {
            transformer_path += ".index.json";
        }
        return {transformer_path, path_join(path, "vae/diffusion_pytorch_model.safetensors"), axes,
                json.at("num_visual_blocks").get<int>(), static_cast<int64_t>(steps.size()),
                timestep_row(steps, timestep_x2), timestep_row(steps, timestep_x4)};
    }

    bool is_bundle(const std::string& path) {
        try {
            const auto bundle = read_bundle(path);
            if (!is_safetensors_file(bundle.vae_path)) {
                return false;
            }
            if (ends_with(bundle.transformer_path, ".index.json")) {
                std::vector<std::string> shards;
                return read_safetensors_index_file(bundle.transformer_path, shards) &&
                       std::all_of(shards.begin(), shards.end(), is_safetensors_file);
            }
            return is_safetensors_file(bundle.transformer_path);
        } catch (const std::exception&) {
            return false;
        }
    }

    std::unique_ptr<Upscaler> Upscaler::load(const std::string& path, ModelLoader& loader, ggml_backend_t backend, std::shared_ptr<RunnerWeightManager> manager, bool direct) {
        try {
            const auto bundle = read_bundle(path);
            if (!loader.init_from_file_and_convert_name(bundle.vae_path, "vae.", VERSION_FLUX) ||
                !loader.add_file(bundle.transformer_path)) {
                throw std::runtime_error("could not load the transformer or VAE weights");
            }
            const auto& tensors = loader.get_tensor_storage_map();
            const auto config   = KandinskyImageSRConfig::detect_from_weights(tensors, bundle.layers, bundle.axes);
            if (config.dim != 2560 || config.ff_dim != 10240 || config.time_dim != 512 || config.rows != bundle.rows) {
                throw std::runtime_error("incompatible transformer weight dimensions");
            }
            auto upscaler         = std::make_unique<Upscaler>();
            upscaler->row_x2      = bundle.row_x2;
            upscaler->row_x4      = bundle.row_x4;
            upscaler->transformer = std::make_unique<Runner>(backend, tensors, config, tile_side / 8, manager);
            upscaler->vae         = std::make_shared<AutoEncoderKL>(backend, tensors, "first_stage_model", false, false, VERSION_FLUX, manager);
            upscaler->vae->set_conv2d_direct_enabled(direct);
            upscaler->vae->set_flash_attention_enabled(true);
            return upscaler;
        } catch (const std::exception& e) {
            LOG_ERROR("cannot load Kandinsky ImageSR bundle '%s': %s", path.c_str(), e.what());
            return nullptr;
        }
    }

    void Upscaler::get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) {
        transformer->model.get_param_tensors(tensors);
        vae->get_param_tensors(tensors);
    }

    void Upscaler::set_max_graph_vram_bytes(size_t bytes) {
        transformer->set_max_graph_vram_bytes(bytes);
        vae->set_max_graph_vram_bytes(bytes);
    }

    static std::vector<int64_t> tile_starts(int64_t length, int side) {
        int64_t last = std::max<int64_t>(0, length - side);
        std::vector<int64_t> starts;
        for (int64_t i = 0; i < last; i += side * 3 / 4) {
            starts.push_back(i);
        }
        starts.push_back(last);
        return starts;
    }

    static std::vector<float> hann_window(bool first, bool last) {
        std::vector<float> weights(tile_side, 1.f);
        for (int i = first ? tile_side / 2 : 0; i < (last ? tile_side / 2 : tile_side); ++i) {
            weights[i] = 0.5f - 0.5f * std::cos(6.283185307179586f * i / tile_side);
        }
        return weights;
    }

    static sd::Tensor<float> prepare_tile(const sd::Tensor<float>& image, int64_t left, int64_t top, int side) {
        const auto width = image.shape()[0], height = image.shape()[1];
        sd::Tensor<float> tile({side, side, 3, 1});
        const int64_t copy_width = std::min<int64_t>(side, width - left);
        for (int c = 0; c < 3; ++c) {
            for (int y = 0; y < side; ++y) {
                auto src = image.data() + (c * height + std::min(height - 1, top + y)) * width + left;
                auto dst = tile.data() + (c * side + y) * side;
                std::copy_n(src, copy_width, dst);
                std::fill(dst + copy_width, dst + side, src[copy_width - 1]);
            }
        }
        return sd::ops::clamp(sd::ops::interpolate(tile, {tile_side, tile_side, 3, 1}, sd::ops::InterpolateMode::Bicubic), 0.f, 1.f);
    }

    static sd::Tensor<float> upscale_tiles(const sd::Tensor<float>& image, int factor, const TileProcessCallback& predict) {
        const auto width = image.shape()[0] * factor, height = image.shape()[1] * factor;
        const int side = tile_side / factor;
        const auto xs = tile_starts(image.shape()[0], side), ys = tile_starts(image.shape()[1], side);
        sd::Tensor<float> output({width, height, 3, 1});
        sd::Tensor<float> weights({width, height});
        for (auto top : ys) {
            const auto wy = hann_window(top == 0, top == ys.back());
            for (auto left : xs) {
                auto tile = predict(prepare_tile(image, left, top, side));
                if (tile.empty()) {
                    return {};
                }
                const auto wx = hann_window(left == 0, left == xs.back());
                for (int y = 0; y < std::min<int64_t>(tile_side, height - top * factor); ++y) {
                    auto dst = (top * factor + y) * width + left * factor;
                    for (int x = 0; x < std::min<int64_t>(tile_side, width - left * factor); ++x) {
                        const float weight = wy[y] * wx[x];
                        weights[dst + x] += weight;
                        for (int c = 0; c < 3; ++c) {
                            output[c * width * height + dst + x] += weight * tile[(c * tile_side + y) * tile_side + x];
                        }
                    }
                }
            }
        }
        output /= weights;
        return output;
    }

    sd::Tensor<float> Upscaler::upscale(int threads, const sd::Tensor<float>& image, int factor) {
        if ((factor != 2 && factor != 4) || image.shape()[3] != 1) {
            LOG_ERROR("Kandinsky ImageSR expects one image and a scale factor of 2 or 4");
            return {};
        }
        const float t = factor == 2 ? timestep_x2 : timestep_x4;
        return upscale_tiles(image, factor, [&](const sd::Tensor<float>& pixels) -> sd::Tensor<float> {
            auto moments = vae->encode(threads, pixels, {});
            if (moments.empty()) {
                return {};
            }
            // ImageSR takes the posterior mode and applies only the FLUX VAE scale.
            auto latent = sd::ops::slice(moments, 2, 0, 16) * vae->scale_factor;
            auto v      = transformer->predict(threads, latent, factor == 2 ? row_x2 : row_x4);
            if (v.empty()) {
                return {};
            }
            return vae->decode(threads, (latent - t * v) / vae->scale_factor, {});
        });
    }
}  // namespace KandinskyImageSR
