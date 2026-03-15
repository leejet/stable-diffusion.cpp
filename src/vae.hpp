#ifndef __VAE_HPP__
#define __VAE_HPP__

#include "common_block.hpp"

struct VAE : public GGMLRunner {
protected:
    SDVersion version;
    bool scale_input                                       = true;
    virtual bool _compute(const int n_threads,
                          struct ggml_tensor* z,
                          bool decode_graph,
                          struct ggml_tensor** output,
                          struct ggml_context* output_ctx) = 0;

public:
    VAE(SDVersion version, ggml_backend_t backend, bool offload_params_to_cpu)
        : version(version), GGMLRunner(backend, offload_params_to_cpu) {}

    int get_scale_factor() {
        int scale_factor = 8;
        if (version == VERSION_WAN2_2_TI2V) {
            scale_factor = 16;
        } else if (sd_version_is_flux2(version)) {
            scale_factor = 16;
        } else if (version == VERSION_CHROMA_RADIANCE) {
            scale_factor = 1;
        }
        return scale_factor;
    }

    virtual int get_encoder_output_channels(int input_channels) = 0;

    void get_tile_sizes(int& tile_size_x,
                        int& tile_size_y,
                        float& tile_overlap,
                        const sd_tiling_params_t& params,
                        int64_t latent_x,
                        int64_t latent_y,
                        float encoding_factor = 1.0f) {
        tile_overlap       = std::max(std::min(params.target_overlap, 0.5f), 0.0f);
        auto get_tile_size = [&](int requested_size, float factor, int64_t latent_size) {
            const int default_tile_size  = 32;
            const int min_tile_dimension = 4;
            int tile_size                = default_tile_size;
            // factor <= 1 means simple fraction of the latent dimension
            // factor > 1 means number of tiles across that dimension
            if (factor > 0.f) {
                if (factor > 1.0)
                    factor = 1 / (factor - factor * tile_overlap + tile_overlap);
                tile_size = static_cast<int>(std::round(latent_size * factor));
            } else if (requested_size >= min_tile_dimension) {
                tile_size = requested_size;
            }
            tile_size = static_cast<int>(tile_size * encoding_factor);
            return std::max(std::min(tile_size, static_cast<int>(latent_size)), min_tile_dimension);
        };

        tile_size_x = get_tile_size(params.tile_size_x, params.rel_size_x, latent_x);
        tile_size_y = get_tile_size(params.tile_size_y, params.rel_size_y, latent_y);
    }

    ggml_tensor* encode(int n_threads,
                        ggml_context* work_ctx,
                        ggml_tensor* x,
                        sd_tiling_params_t tiling_params,
                        bool circular_x = false,
                        bool circular_y = false) {
        int64_t t0             = ggml_time_ms();
        ggml_tensor* result    = nullptr;
        const int scale_factor = get_scale_factor();
        int64_t W              = x->ne[0] / scale_factor;
        int64_t H              = x->ne[1] / scale_factor;
        int channel_dim        = sd_version_is_wan(version) ? 3 : 2;
        int64_t C              = get_encoder_output_channels(static_cast<int>(x->ne[channel_dim]));
        int64_t ne2;
        int64_t ne3;
        if (sd_version_is_wan(version)) {
            int64_t T = x->ne[2];
            ne2       = (T - 1) / 4 + 1;
            ne3       = C;
        } else {
            ne2 = C;
            ne3 = x->ne[3];
        }
        result = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, W, H, ne2, ne3);

        if (scale_input) {
            scale_to_minus1_1(x);
        }

        if (sd_version_is_qwen_image(version) || sd_version_is_anima(version)) {
            x = ggml_reshape_4d(work_ctx, x, x->ne[0], x->ne[1], 1, x->ne[2] * x->ne[3]);
        }

        if (tiling_params.enabled) {
            float tile_overlap;
            int tile_size_x, tile_size_y;
            // multiply tile size for encode to keep the compute buffer size consistent
            get_tile_sizes(tile_size_x, tile_size_y, tile_overlap, tiling_params, W, H, 1.30539f);

            LOG_DEBUG("VAE Tile size: %dx%d", tile_size_x, tile_size_y);

            auto on_tiling = [&](ggml_tensor* in, ggml_tensor* out, bool init) {
                return _compute(n_threads, in, false, &out, work_ctx);
            };
            sd_tiling_non_square(x, result, scale_factor, tile_size_x, tile_size_y, tile_overlap, circular_x, circular_y, on_tiling);
        } else {
            _compute(n_threads, x, false, &result, work_ctx);
        }
        free_compute_buffer();

        int64_t t1 = ggml_time_ms();
        LOG_DEBUG("computing vae encode graph completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);
        return result;
    }

    ggml_tensor* decode(int n_threads,
                        ggml_context* work_ctx,
                        ggml_tensor* x,
                        sd_tiling_params_t tiling_params,
                        bool decode_video   = false,
                        bool circular_x     = false,
                        bool circular_y     = false,
                        ggml_tensor* result = nullptr,
                        bool silent         = false) {
        const int scale_factor = get_scale_factor();
        int64_t W              = x->ne[0] * scale_factor;
        int64_t H              = x->ne[1] * scale_factor;
        int64_t C              = 3;
        if (result == nullptr) {
            if (decode_video) {
                int64_t T = x->ne[2];
                if (sd_version_is_wan(version)) {
                    T = ((T - 1) * 4) + 1;
                }
                result = ggml_new_tensor_4d(work_ctx,
                                            GGML_TYPE_F32,
                                            W,
                                            H,
                                            T,
                                            3);
            } else {
                result = ggml_new_tensor_4d(work_ctx,
                                            GGML_TYPE_F32,
                                            W,
                                            H,
                                            C,
                                            x->ne[3]);
            }
        }
        int64_t t0 = ggml_time_ms();
        if (sd_version_is_qwen_image(version) || sd_version_is_anima(version)) {
            x = ggml_reshape_4d(work_ctx, x, x->ne[0], x->ne[1], 1, x->ne[2] * x->ne[3]);
        }
        if (tiling_params.enabled) {
            float tile_overlap;
            int tile_size_x, tile_size_y;
            get_tile_sizes(tile_size_x, tile_size_y, tile_overlap, tiling_params, x->ne[0], x->ne[1]);

            if (!silent) {
                LOG_DEBUG("VAE Tile size: %dx%d", tile_size_x, tile_size_y);
            }

            auto on_tiling = [&](ggml_tensor* in, ggml_tensor* out, bool init) {
                return _compute(n_threads, in, true, &out, nullptr);
            };
            sd_tiling_non_square(x, result, scale_factor, tile_size_x, tile_size_y, tile_overlap, circular_x, circular_y, on_tiling, silent);
        } else {
            if (!_compute(n_threads, x, true, &result, work_ctx)) {
                LOG_ERROR("Failed to decode latetnts");
                free_compute_buffer();
                return nullptr;
            }
        }
        free_compute_buffer();
        if (scale_input) {
            scale_to_0_1(result);
        }
        int64_t t1 = ggml_time_ms();
        LOG_DEBUG("computing vae decode graph completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);
        ggml_ext_tensor_clamp_inplace(result, 0.0f, 1.0f);
        return result;
    }

    virtual ggml_tensor* vae_output_to_latents(ggml_context* work_ctx, ggml_tensor* vae_output, std::shared_ptr<RNG> rng) = 0;
    virtual ggml_tensor* diffusion_to_vae_latents(ggml_context* work_ctx, ggml_tensor* latents)                           = 0;
    virtual ggml_tensor* vae_to_diffuison_latents(ggml_context* work_ctx, ggml_tensor* latents)                           = 0;
    virtual void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix)         = 0;
    virtual void set_conv2d_scale(float scale) { SD_UNUSED(scale); };
};

struct FakeVAE : public VAE {
    FakeVAE(SDVersion version, ggml_backend_t backend, bool offload_params_to_cpu)
        : VAE(version, backend, offload_params_to_cpu) {}

    int get_encoder_output_channels(int input_channels) {
        return input_channels;
    }

    bool _compute(const int n_threads,
                  struct ggml_tensor* z,
                  bool decode_graph,
                  struct ggml_tensor** output,
                  struct ggml_context* output_ctx) override {
        if (*output == nullptr && output_ctx != nullptr) {
            *output = ggml_dup_tensor(output_ctx, z);
        }
        ggml_ext_tensor_iter(z, [&](ggml_tensor* z, int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
            float value = ggml_ext_tensor_get_f32(z, i0, i1, i2, i3);
            ggml_ext_tensor_set_f32(*output, value, i0, i1, i2, i3);
        });
        return true;
    }

    ggml_tensor* vae_output_to_latents(ggml_context* work_ctx, ggml_tensor* vae_output, std::shared_ptr<RNG> rng) {
        return vae_output;
    }

    ggml_tensor* diffusion_to_vae_latents(ggml_context* work_ctx, ggml_tensor* latents) {
        return ggml_ext_dup_and_cpy_tensor(work_ctx, latents);
    }

    ggml_tensor* vae_to_diffuison_latents(ggml_context* work_ctx, ggml_tensor* latents) {
        return ggml_ext_dup_and_cpy_tensor(work_ctx, latents);
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) override {}

    std::string get_desc() override {
        return "fake_vae";
    }
};

#endif  // __VAE_HPP__
