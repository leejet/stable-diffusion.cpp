#ifndef __SD_MODEL_UPSCALER_H3_LATENT_UPSCALER_HPP__
#define __SD_MODEL_UPSCALER_H3_LATENT_UPSCALER_HPP__

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "core/ggml_extend.h"
#include "core/ggml_graph_cut.h"
#include "core/ggml_runner.h"
#include "core/tensor.hpp"
#include "core/util.h"
#include "model/common/ggml_block.hpp"
#include "model/upscaler/ltx_latent_upscaler.hpp"
#include "model_loader.h"

// Ported from the community ComfyUI node LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler
// (MIT) with checkpoint LBH-123-AI/Minimax_h3_latent_Upscaler (Apache-2.0).

namespace H3VUpsampler {
    constexpr int H3_UPSAMPLER_GRAPH_SIZE = 8192;

    // Training applies the VAE normalization a second time to diffusion latents, and both
    // upscaler generations were trained in that domain. sd.cpp's sampler already lives
    // there, so the transform is applied on entry and undone on the way out.
    inline const sd::Tensor<float>& h3_latent_mean() {
        static const sd::Tensor<float> mean({1, 1, 1, 24},
                                            {0.858090341091156f, -0.9606591463088989f, 1.0661640167236328f, -0.5090325474739075f,
                                             -0.2727581858634949f, -1.3675414323806763f, -0.2553254961967468f, -0.26907554268836975f,
                                             -0.5376840829849243f, -0.0464097298681736f, 0.6657370328903198f, 0.19690127670764923f,
                                             -0.5460608005523682f, -0.4035342037677765f, -0.23683024942874908f, 0.25928452610969543f,
                                             -0.30133944749832153f, 0.211341992020607f, -1.1206848621368408f, 0.3581933379173279f,
                                             -0.04225143790245056f, 0.2604829967021942f, 0.22864092886447906f, 0.7056031823158264f});
        return mean;
    }

    inline const sd::Tensor<float>& h3_latent_stddev() {
        static const sd::Tensor<float> stddev({1, 1, 1, 24},
                                              {1.2223774194717407f, 1.2767263650894165f, 1.6831774711608887f, 1.7549455165863037f,
                                               1.5636216402053833f, 2.194143533706665f, 0.9653137922286987f, 1.0569885969161987f,
                                               0.841948926448822f, 0.7729952931404114f, 1.8955937623977661f, 0.946841835975647f,
                                               0.7996809482574463f, 0.44988900423049927f, 0.7197399735450745f, 0.6936293244361877f,
                                               2.961095094680786f, 2.7694199085235596f, 3.0496184825897217f, 2.1088054180145264f,
                                               3.276226282119751f, 3.1627357006073f, 2.2816812992095947f, 2.6127843856811523f});
        return stddev;
    }

    // Input contract shared by both generations: an H3 video latent with a single batch.
    inline bool h3_check_upsampler_input(const sd::Tensor<float>& x, int64_t in_channels, const char* label) {
        if (x.dim() != 4 && x.dim() != 5) {
            LOG_ERROR("%s expects a 4D or 5D video latent, got dim=%lld", label, (long long)x.dim());
            return false;
        }
        if (x.dim() == 5 && x.shape()[4] != 1) {
            LOG_ERROR("%s currently supports batch size 1, got batch=%lld", label, (long long)x.shape()[4]);
            return false;
        }
        if (x.shape()[3] != in_channels) {
            LOG_ERROR("%s expects %" PRId64 " channels, got %lld", label, in_channels, (long long)x.shape()[3]);
            return false;
        }
        return true;
    }

    // An even latent size keeps the decoded pixel size a multiple of 32.
    inline bool h3_check_upsampler_target(int64_t target_w, int64_t target_h, const char* label) {
        if (target_w % 2 != 0 || target_h % 2 != 0) {
            LOG_ERROR("%s target latent size must be even (pixel size a multiple of 32), got %" PRId64 "x%" PRId64,
                      label,
                      target_w,
                      target_h);
            return false;
        }
        return true;
    }

    // Both checkpoint generations expose the same entry point, so the video pipeline can
    // hold a single runner pointer and stay unaware of which one it loaded.
    struct H3LatentUpscalerRunnerBase : public GGMLRunner {
        // A stage maps the latent it receives, plus a stage-specific graph leaf, to the
        // next latent inside the graph.
        using StageFn = std::function<ggml_tensor*(GGMLRunnerContext*, ggml_tensor*, ggml_tensor*)>;

        using GGMLRunner::GGMLRunner;

        virtual ~H3LatentUpscalerRunnerBase() = default;

        // False when the checkpoint did not bind; the caller must abort instead of running.
        virtual bool ready() const = 0;

        virtual void collect_params(std::map<std::string, ggml_tensor*>& tensors) = 0;

        // x: H3 diffusion latent [W, H, T, C] (5D with batch 1 accepted). The result stays
        // in the same normalized domain for the next sampling pass.
        virtual sd::Tensor<float> compute(int n_threads,
                                          const sd::Tensor<float>& x,
                                          int64_t target_w,
                                          int64_t target_h) = 0;

    protected:
        // Both generations run the same shape of chain: stage A at the source resolution,
        // a bilinear resize, then stage B at the target. `extra` is the stage-specific
        // leaf: a scale scalar for the 3D checkpoint, the Fourier features for the 2D one.
        sd::Tensor<float> run_stages(const sd::Tensor<float>& normalized,
                                     const sd::Tensor<float>& extra,
                                     int64_t target_w,
                                     int64_t target_h,
                                     int n_threads,
                                     const StageFn& stage_a,
                                     const StageFn& stage_b,
                                     const char* label) {
            auto get_graph_a = [&]() -> ggml_cgraph* {
                ggml_cgraph* gf  = new_graph_custom(H3_UPSAMPLER_GRAPH_SIZE);
                ggml_tensor* x_t = make_input(normalized);
                ggml_tensor* e_t = make_input(extra);
                auto runner_ctx  = get_context(gf);
                ggml_tensor* out = stage_a(&runner_ctx, x_t, e_t);
                ggml_build_forward_expand(gf, out);
                return gf;
            };
            std::optional<sd::Tensor<float>> mid_opt = GGMLRunner::compute(get_graph_a, n_threads, false, false);
            runner_end();
            if (!mid_opt.has_value() || mid_opt->empty()) {
                LOG_ERROR("%s stage A (in-block chain) failed", label);
                return {};
            }
            sd::Tensor<float> mid = std::move(*mid_opt);

            const int64_t in_w             = normalized.shape()[0];
            const int64_t in_h             = normalized.shape()[1];
            std::vector<int64_t> mid_shape = mid.shape();
            mid_shape[0]                   = target_w;
            mid_shape[1]                   = target_h;
            sd::Tensor<float> mid_up       = sd::ops::interpolate(mid, mid_shape, sd::ops::InterpolateMode::Bilinear, false, false);
            if (mid_up.empty()) {
                LOG_ERROR("%s middle spatial interpolate failed: %" PRId64 "x%" PRId64 " -> %" PRId64 "x%" PRId64,
                          label,
                          in_w,
                          in_h,
                          target_w,
                          target_h);
                return {};
            }

            auto get_graph_b = [&]() -> ggml_cgraph* {
                ggml_cgraph* gf  = new_graph_custom(H3_UPSAMPLER_GRAPH_SIZE);
                ggml_tensor* m_t = make_input(mid_up);
                ggml_tensor* e_t = make_input(extra);
                auto runner_ctx  = get_context(gf);
                ggml_tensor* out = stage_b(&runner_ctx, m_t, e_t);
                ggml_build_forward_expand(gf, out);
                return gf;
            };
            std::optional<sd::Tensor<float>> out_opt = GGMLRunner::compute(get_graph_b, n_threads, false, false);
            runner_end();
            if (!out_opt.has_value() || out_opt->empty()) {
                LOG_ERROR("%s stage B (out-block chain) failed", label);
                return {};
            }
            return std::move(*out_opt);
        }
    };

    struct H3LatentUpscalerConfig {
        int64_t in_channels  = 24;
        int64_t mid_channels = 512;
        int embed_dim        = 64;
        int temporal_kernel  = 5;
        // Residual and temporal blocks share interleaved checkpoint indices.
        std::vector<int> in_res, in_temporal, out_res, out_temporal;
        bool valid = false;

        static H3LatentUpscalerConfig detect_from_weights(const String2TensorStorage& tensor_storage_map,
                                                          const std::string& prefix = "") {
            H3LatentUpscalerConfig config;
            std::string base = prefix;
            if (!base.empty()) {
                base += ".";
            }
            auto find_weight = [&](const std::string& suffix) -> const TensorStorage* {
                auto iter = tensor_storage_map.find(base + suffix);
                if (iter == tensor_storage_map.end()) {
                    return nullptr;
                }
                return &iter->second;
            };

            const TensorStorage* conv_in = find_weight("conv_in.weight");
            if (conv_in == nullptr) {
                LOG_ERROR("h3 latent upsampler: 'conv_in.weight' not found under prefix '%s'", prefix.c_str());
                return config;
            }
            // Bias shapes preserve channel counts when the loader folds 5D conv weights.
            const TensorStorage* conv_in_bias = find_weight("conv_in.bias");
            if (conv_in_bias != nullptr) {
                config.mid_channels = conv_in_bias->ne[0];
            } else if (conv_in->n_dims == 5) {
                config.mid_channels = conv_in->ne[0];
            }
            const TensorStorage* conv_out_bias = find_weight("conv_out.bias");
            if (conv_out_bias != nullptr) {
                config.in_channels = conv_out_bias->ne[0];
            } else if (conv_in->n_dims == 5) {
                config.in_channels = conv_in->ne[1];
            }
            if (config.mid_channels <= 0 || config.in_channels <= 0) {
                LOG_ERROR("h3 latent upsampler: could not infer channel counts from the checkpoint");
                return config;
            }

            const TensorStorage* embed_mid = find_weight("embed.2.weight");
            if (embed_mid != nullptr && embed_mid->n_dims >= 1) {
                config.embed_dim = static_cast<int>(embed_mid->logical_ne0());
            }

            std::set<int> in_res_idx, in_temporal_idx, out_res_idx, out_temporal_idx;
            for (const auto& [name, _] : tensor_storage_map) {
                if (!starts_with(name, base)) {
                    continue;
                }
                const std::string rel = name.substr(base.size());
                const bool is_in      = starts_with(rel, "in_blocks.");
                const bool is_out     = starts_with(rel, "out_blocks.");
                if (!is_in && !is_out) {
                    continue;
                }
                const std::string rest = rel.substr(is_in ? 10 : 11);
                const size_t dot       = rest.find('.');
                if (dot == std::string::npos) {
                    continue;
                }
                int idx = 0;
                try {
                    idx = std::stoi(rest.substr(0, dot));
                } catch (...) {
                    continue;
                }
                const std::string part = rest.substr(dot + 1);
                if (starts_with(part, "dwconv.") || starts_with(part, "pwconv.")) {
                    (is_in ? in_temporal_idx : out_temporal_idx).insert(idx);
                } else if (starts_with(part, "in_layers.") || starts_with(part, "out_norm.") ||
                           starts_with(part, "out_layers.") || starts_with(part, "emb_layers.")) {
                    (is_in ? in_res_idx : out_res_idx).insert(idx);
                } else if (starts_with(part, "attn.")) {
                    LOG_ERROR("h3 latent upsampler: attention blocks are not supported by the v1 runner");
                    return config;
                }
            }
            config.in_res       = std::vector<int>(in_res_idx.begin(), in_res_idx.end());
            config.in_temporal  = std::vector<int>(in_temporal_idx.begin(), in_temporal_idx.end());
            config.out_res      = std::vector<int>(out_res_idx.begin(), out_res_idx.end());
            config.out_temporal = std::vector<int>(out_temporal_idx.begin(), out_temporal_idx.end());
            if (config.in_res.empty() || config.out_res.empty()) {
                LOG_ERROR("h3 latent upsampler: no res blocks found in the checkpoint");
                return config;
            }

            auto temporal_kernel_of = [&](std::vector<int> idxs, const char* stage) -> int {
                for (int idx : idxs) {
                    const TensorStorage* dw = find_weight(std::string(stage) + "." + std::to_string(idx) + ".dwconv.weight");
                    if (dw != nullptr) {
                        // kD stays at ne[2] in both [OC, IC, kD, kH, kW] and [kW, kH, kD, OC*IC].
                        return static_cast<int>(dw->ne[2]);
                    }
                }
                return 0;
            };
            const int k_in  = temporal_kernel_of(config.in_temporal, "in_blocks");
            const int k_out = k_in > 0 ? k_in : temporal_kernel_of(config.out_temporal, "out_blocks");
            if (k_out > 0) {
                config.temporal_kernel = k_out;
            } else {
                config.temporal_kernel = 1;
            }

            config.valid = true;
            LOG_VERBOSE("h3 latent upsampler: in_channels = %" PRId64 ", mid_channels = %" PRId64
                        ", in_res = %zu, in_temporal = %zu, out_res = %zu, out_temporal = %zu, temporal_kernel = %d, embed_dim = %d",
                        config.in_channels,
                        config.mid_channels,
                        config.in_res.size(),
                        config.in_temporal.size(),
                        config.out_res.size(),
                        config.out_temporal.size(),
                        config.temporal_kernel,
                        config.embed_dim);
            return config;
        }
    };

    class H3ResBlock : public GGMLBlock {
    protected:
        int64_t channels;
        int64_t embed_dim;

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map,
                         const std::string prefix) override {
            auto make_conv = [&](const std::string& name) {
                blocks[name] = std::shared_ptr<GGMLBlock>(new Conv3d(channels,
                                                                     channels,
                                                                     {3, 3, 3},
                                                                     {1, 1, 1},
                                                                     {1, 1, 1}));
            };
            make_conv("in_layers.2");
            make_conv("out_layers.2");
            blocks["in_layers.0"]  = std::shared_ptr<GGMLBlock>(new LTXVUpsampler::VideoGroupNorm(32, channels));
            blocks["out_norm"]     = std::shared_ptr<GGMLBlock>(new LTXVUpsampler::VideoGroupNorm(32, channels));
            blocks["emb_layers.1"] = std::shared_ptr<GGMLBlock>(new Linear(embed_dim, 2 * channels, true));
        }

    public:
        H3ResBlock(int64_t channels, int64_t embed_dim)
            : channels(channels),
              embed_dim(embed_dim) {}

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* emb) {
            auto in_layers  = std::dynamic_pointer_cast<Conv3d>(blocks["in_layers.2"]);
            auto in_norm    = std::dynamic_pointer_cast<LTXVUpsampler::VideoGroupNorm>(blocks["in_layers.0"]);
            auto out_layers = std::dynamic_pointer_cast<Conv3d>(blocks["out_layers.2"]);
            auto out_norm   = std::dynamic_pointer_cast<LTXVUpsampler::VideoGroupNorm>(blocks["out_norm"]);
            auto emb_layers = std::dynamic_pointer_cast<Linear>(blocks["emb_layers.1"]);

            ggml_tensor* h = in_norm->forward(ctx, x);
            h              = ggml_silu_inplace(ctx->ggml_ctx, h);
            h              = in_layers->forward(ctx, h);

            // Keep emb intact: every residual block uses the same embedding.
            ggml_tensor* emb_silu = ggml_silu(ctx->ggml_ctx, emb);
            ggml_tensor* emb_out  = emb_layers->forward(ctx, emb_silu);
            ggml_tensor* emb_1d   = ggml_reshape_1d(ctx->ggml_ctx, ggml_cont(ctx->ggml_ctx, emb_out), 2 * channels);
            ggml_tensor* scale    = ggml_reshape_4d(ctx->ggml_ctx, ggml_ext_slice(ctx, emb_1d, 0, 0, channels), 1, 1, 1, channels);
            ggml_tensor* shift    = ggml_reshape_4d(ctx->ggml_ctx, ggml_ext_slice(ctx, emb_1d, 0, channels, 2 * channels), 1, 1, 1, channels);
            ggml_tensor* gain     = ggml_scale_bias(ctx->ggml_ctx, scale, 1.f, 1.f);

            h = out_norm->forward(ctx, h);
            h = ggml_mul_inplace(ctx->ggml_ctx, h, gain);
            h = ggml_add_inplace(ctx->ggml_ctx, h, shift);
            h = ggml_silu_inplace(ctx->ggml_ctx, h);
            h = out_layers->forward(ctx, h);
            return ggml_add(ctx->ggml_ctx, h, x);
        }
    };

    // Shifted multiply-adds preserve depthwise weights [1, 1, k, C] without
    // expanding them into the dense kernels required by Conv3d.
    class H3TemporalDWConv3d : public UnaryBlock {
    protected:
        int64_t channels;
        int kernel;
        bool has_bias;
        std::string prefix;

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map,
                         const std::string prefix) override {
            this->prefix     = prefix;
            params["weight"] = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 1, 1, kernel, channels);
            if (has_bias) {
                params["bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, channels);
            }
        }

    public:
        H3TemporalDWConv3d(int64_t channels, int kernel, bool has_bias = true)
            : channels(channels),
              kernel(kernel),
              has_bias(has_bias) {}

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            ggml_tensor* w = params["weight"];
            if (ctx->weight_adapter) {
                w = ctx->weight_adapter->patch_weight(ctx, w, prefix + "weight");
            }
            ggml_tensor* w32 = ggml_cast(ctx->ggml_ctx, w, GGML_TYPE_F32);
            ggml_tensor* xa  = x;
            if (xa->type != GGML_TYPE_F32) {
                xa = ggml_cast(ctx->ggml_ctx, xa, GGML_TYPE_F32);
            }

            const int64_t T = xa->ne[2];
            const int pad   = kernel / 2;

            // ggml_pad only pads the end; concatenate zeros for symmetric padding.
            ggml_tensor* xp = xa;
            if (pad > 0) {
                ggml_tensor* zero = ggml_scale(ctx->ggml_ctx, ggml_ext_slice(ctx, xa, 2, 0, 1), 0.0f);
                ggml_tensor* lead = zero;
                for (int i = 1; i < pad; ++i) {
                    lead = ggml_concat(ctx->ggml_ctx, lead, zero, 2);
                }
                xp = ggml_concat(ctx->ggml_ctx, lead, xa, 2);
                xp = ggml_concat(ctx->ggml_ctx, xp, lead, 2);
            }

            ggml_tensor* acc = nullptr;
            for (int j = 0; j < kernel; ++j) {
                ggml_tensor* xs   = ggml_ext_slice(ctx, xp, 2, j, j + T);
                ggml_tensor* wj   = ggml_ext_slice(ctx, w32, 2, j, j + 1);  // [1, 1, 1, C]
                ggml_tensor* term = ggml_mul(ctx->ggml_ctx, xs, wj);
                acc               = acc == nullptr ? term : ggml_add(ctx->ggml_ctx, acc, term);
            }
            if (has_bias) {
                ggml_tensor* b = params["bias"];
                if (ctx->weight_adapter) {
                    b = ctx->weight_adapter->patch_weight(ctx, b, prefix + "bias");
                }
                acc = ggml_add(ctx->ggml_ctx, acc, ggml_reshape_4d(ctx->ggml_ctx, b, 1, 1, 1, channels));
            }
            return acc;
        }
    };

    class H3TemporalConv : public GGMLBlock {
    protected:
        int64_t channels;
        int temporal_kernel;

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map,
                         const std::string prefix) override {
            blocks["dwconv"] = std::shared_ptr<GGMLBlock>(new H3TemporalDWConv3d(channels, temporal_kernel));
            blocks["pwconv"] = std::shared_ptr<GGMLBlock>(new Conv3d(channels,
                                                                     channels,
                                                                     {1, 1, 1},
                                                                     {1, 1, 1},
                                                                     {0, 0, 0}));
            blocks["norm"]   = std::shared_ptr<GGMLBlock>(new LTXVUpsampler::VideoGroupNorm(32, channels));
        }

    public:
        H3TemporalConv(int64_t channels, int temporal_kernel)
            : channels(channels),
              temporal_kernel(temporal_kernel) {}

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto dwconv = std::dynamic_pointer_cast<H3TemporalDWConv3d>(blocks["dwconv"]);
            auto pwconv = std::dynamic_pointer_cast<Conv3d>(blocks["pwconv"]);
            auto norm   = std::dynamic_pointer_cast<LTXVUpsampler::VideoGroupNorm>(blocks["norm"]);

            ggml_tensor* h = norm->forward(ctx, x);
            h              = ggml_silu_inplace(ctx->ggml_ctx, h);
            h              = dwconv->forward(ctx, h);
            h              = pwconv->forward(ctx, h);
            return ggml_add(ctx->ggml_ctx, h, x);
        }
    };

    class H3Embed : public GGMLBlock {
    protected:
        int embed_dim;

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map,
                         const std::string prefix) override {
            blocks["0"] = std::shared_ptr<GGMLBlock>(new Linear(1, embed_dim, true));
            blocks["2"] = std::shared_ptr<GGMLBlock>(new Linear(embed_dim, embed_dim, true));
        }

    public:
        explicit H3Embed(int embed_dim)
            : embed_dim(embed_dim) {}

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* scale_input) {
            auto l0        = std::dynamic_pointer_cast<Linear>(blocks["0"]);
            auto l2        = std::dynamic_pointer_cast<Linear>(blocks["2"]);
            ggml_tensor* h = l0->forward(ctx, ggml_reshape_2d(ctx->ggml_ctx, scale_input, 1, 1));
            h              = ggml_silu_inplace(ctx->ggml_ctx, h);
            return l2->forward(ctx, h);
        }
    };

    class H3LatentUpscaler : public GGMLBlock {
    public:
        H3LatentUpscalerConfig config;

        explicit H3LatentUpscaler(H3LatentUpscalerConfig config_)
            : config(std::move(config_)) {
            for (int idx : config.in_res) {
                blocks["in_blocks." + std::to_string(idx)] = std::shared_ptr<GGMLBlock>(new H3ResBlock(config.mid_channels,
                                                                                                       config.embed_dim));
            }
            for (int idx : config.in_temporal) {
                blocks["in_blocks." + std::to_string(idx)] = std::shared_ptr<GGMLBlock>(new H3TemporalConv(config.mid_channels,
                                                                                                           config.temporal_kernel));
            }
            for (int idx : config.out_res) {
                blocks["out_blocks." + std::to_string(idx)] = std::shared_ptr<GGMLBlock>(new H3ResBlock(config.mid_channels,
                                                                                                        config.embed_dim));
            }
            for (int idx : config.out_temporal) {
                blocks["out_blocks." + std::to_string(idx)] = std::shared_ptr<GGMLBlock>(new H3TemporalConv(config.mid_channels,
                                                                                                            config.temporal_kernel));
            }
            blocks["conv_in"]  = std::shared_ptr<GGMLBlock>(new Conv3d(config.in_channels,
                                                                       config.mid_channels,
                                                                       {3, 3, 3},
                                                                       {1, 1, 1},
                                                                       {1, 1, 1}));
            blocks["norm_out"] = std::shared_ptr<GGMLBlock>(new LTXVUpsampler::VideoGroupNorm(32, config.mid_channels));
            blocks["conv_out"] = std::shared_ptr<GGMLBlock>(new Conv3d(config.mid_channels,
                                                                       config.in_channels,
                                                                       {3, 3, 3},
                                                                       {1, 1, 1},
                                                                       {1, 1, 1}));
            blocks["embed"]    = std::shared_ptr<GGMLBlock>(new H3Embed(config.embed_dim));
        }

        ggml_tensor* forward_stage_a(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* scale_input) {
            auto conv_in = std::dynamic_pointer_cast<Conv3d>(blocks["conv_in"]);
            auto embed   = std::dynamic_pointer_cast<H3Embed>(blocks["embed"]);

            ggml_tensor* h = conv_in->forward(ctx, x);
            sd::ggml_graph_cut::mark_graph_cut(h, "h3_latent_upscaler.conv_in", "x");

            ggml_tensor* emb = embed->forward(ctx, scale_input);
            std::set<int> temporal_set(config.in_temporal.begin(), config.in_temporal.end());
            int max_idx = 0;
            for (int idx : config.in_res) {
                max_idx = std::max(max_idx, idx);
            }
            for (int idx : config.in_temporal) {
                max_idx = std::max(max_idx, idx);
            }
            for (int i = 0; i <= max_idx; ++i) {
                if (temporal_set.count(i)) {
                    auto block = std::dynamic_pointer_cast<H3TemporalConv>(blocks["in_blocks." + std::to_string(i)]);
                    h          = block->forward(ctx, h);
                } else {
                    auto block = std::dynamic_pointer_cast<H3ResBlock>(blocks["in_blocks." + std::to_string(i)]);
                    h          = block->forward(ctx, h, emb);
                }
                sd::ggml_graph_cut::mark_graph_cut(h, "h3_latent_upscaler.in_blocks." + std::to_string(i), "x");
            }
            return h;
        }

        ggml_tensor* forward_stage_b(GGMLRunnerContext* ctx, ggml_tensor* m, ggml_tensor* scale_input) {
            auto conv_out = std::dynamic_pointer_cast<Conv3d>(blocks["conv_out"]);
            auto norm_out = std::dynamic_pointer_cast<LTXVUpsampler::VideoGroupNorm>(blocks["norm_out"]);
            auto embed    = std::dynamic_pointer_cast<H3Embed>(blocks["embed"]);

            ggml_tensor* g   = m;
            ggml_tensor* emb = embed->forward(ctx, scale_input);
            std::set<int> temporal_set(config.out_temporal.begin(), config.out_temporal.end());
            int max_idx = 0;
            for (int idx : config.out_res) {
                max_idx = std::max(max_idx, idx);
            }
            for (int idx : config.out_temporal) {
                max_idx = std::max(max_idx, idx);
            }
            for (int i = 0; i <= max_idx; ++i) {
                if (temporal_set.count(i)) {
                    auto block = std::dynamic_pointer_cast<H3TemporalConv>(blocks["out_blocks." + std::to_string(i)]);
                    g          = block->forward(ctx, g);
                } else {
                    auto block = std::dynamic_pointer_cast<H3ResBlock>(blocks["out_blocks." + std::to_string(i)]);
                    g          = block->forward(ctx, g, emb);
                }
                sd::ggml_graph_cut::mark_graph_cut(g, "h3_latent_upscaler.out_blocks." + std::to_string(i), "x");
            }

            g = norm_out->forward(ctx, g);
            g = ggml_silu_inplace(ctx->ggml_ctx, g);
            g = conv_out->forward(ctx, g);
            sd::ggml_graph_cut::mark_graph_cut(g, "h3_latent_upscaler.conv_out", "x");
            return g;
        }
    };

    struct H3LatentUpscalerRunner : public H3LatentUpscalerRunnerBase {
        H3LatentUpscalerConfig config;
        std::unique_ptr<H3LatentUpscaler> model;
        std::string weight_prefix;

        H3LatentUpscalerRunner(ggml_backend_t backend,
                               const String2TensorStorage& tensor_storage_map,
                               const std::string& prefix                           = "",
                               std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
            : H3LatentUpscalerRunnerBase(backend, weight_manager),
              config(H3LatentUpscalerConfig::detect_from_weights(tensor_storage_map, prefix)),
              weight_prefix(prefix) {
            if (!config.valid || config.embed_dim <= 0 || config.mid_channels <= 0) {
                LOG_ERROR("unsupported H3 latent upsampler config: in_channels=%" PRId64 " mid_channels=%" PRId64 " embed_dim=%d temporal_kernel=%d",
                          config.in_channels,
                          config.mid_channels,
                          config.embed_dim,
                          config.temporal_kernel);
                return;
            }
            model = std::make_unique<H3LatentUpscaler>(config);
            model->init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return "h3_latent_upscaler";
        }

        bool ready() const override {
            return model != nullptr;
        }

        void collect_params(std::map<std::string, ggml_tensor*>& tensors) override {
            if (model) {
                model->get_param_tensors(tensors, weight_prefix);
            }
        }

        sd::Tensor<float> compute(const int n_threads,
                                  const sd::Tensor<float>& x,
                                  int64_t target_w,
                                  int64_t target_h) override {
            constexpr const char* label = "H3 latent upsampler";
            if (!h3_check_upsampler_input(x, config.in_channels, label) ||
                !h3_check_upsampler_target(target_w, target_h, label)) {
                return {};
            }
            if (config.in_channels != 24) {
                LOG_ERROR("%s requires 24 channels, got model=%" PRId64, label, config.in_channels);
                return {};
            }

            const int64_t in_w = x.shape()[0];
            const int64_t in_h = x.shape()[1];
            const int64_t T    = x.shape()[2];
            // The 3D checkpoint takes one isotropic scale; the node front-ends feed it the
            // mean of the two axis ratios.
            const float scale = (static_cast<float>(target_w) / static_cast<float>(in_w) +
                                 static_cast<float>(target_h) / static_cast<float>(in_h)) /
                                2.0f;
            if (scale < 1.0f - 1e-3f) {
                LOG_ERROR("%s only supports upscaling (scale >= 1.0), got %.3f", label, scale);
                return {};
            }

            const size_t expected_dim            = static_cast<size_t>(x.dim());
            const sd::Tensor<float> normalized   = (x - h3_latent_mean()) / h3_latent_stddev();
            const sd::Tensor<float> scale_tensor = sd::Tensor<float>::from_vector({scale - 1.0f});

            sd::Tensor<float> out = run_stages(
                normalized,
                scale_tensor,
                target_w,
                target_h,
                n_threads,
                [this](GGMLRunnerContext* ctx, ggml_tensor* latent, ggml_tensor* scale_in) {
                    return model->forward_stage_a(ctx, latent, scale_in);
                },
                [this](GGMLRunnerContext* ctx, ggml_tensor* latent, ggml_tensor* scale_in) {
                    return model->forward_stage_b(ctx, latent, scale_in);
                },
                label);
            if (out.empty()) {
                return {};
            }
            if (out.shape()[0] != target_w || out.shape()[1] != target_h || out.shape()[2] != T) {
                LOG_ERROR("%s output shape mismatch: got %" PRId64 "x%" PRId64 "x%" PRId64 " expected %" PRId64 "x%" PRId64 "x%" PRId64,
                          label,
                          out.shape()[0],
                          out.shape()[1],
                          out.shape()[2],
                          target_w,
                          target_h,
                          T);
                return {};
            }
            return restore_trailing_singleton_dims(out * h3_latent_stddev() + h3_latent_mean(), expected_dim);
        }
    };

    // ---------------------------------------------------------------------------
    // 2D-conv variant (minimax_h3_latent_upscaler_2d_conv_v1)
    //
    // The lean sibling of the 3D checkpoint: a 2D conv backbone with two *shared*
    // temporal blocks, no attention, no squeeze-excite, no large-kernel branch, no
    // post-upsample refine and no learned pixel-shuffle stage. Spatial upscaling in the
    // middle of the network is plain bilinear and the time axis is never resampled, so
    // the packed audio latent length stays valid.
    //
    // State-dict layout note: the backbone lives under a "resizer." prefix while the two
    // temporal blocks sit at the top level, so the module tree mirrors that nesting
    // instead of flattening everything like the 3D variant does.
    // ---------------------------------------------------------------------------

    struct H3LatentUpscaler2dConfig {
        int64_t in_channels  = 24;
        int64_t mid_channels = 640;
        int embed_dim        = 128;
        int embed_freq       = 16;
        int tier_dim         = 32;
        int temporal_kernel  = 5;
        int in_blocks        = 12;
        int out_blocks       = 12;
        int temporal_every   = 2;
        int temporal_blocks  = 2;
        bool valid           = false;

        // True when any key under `key_prefix` carries a block index at or past `count`,
        // which means the numbering has a hole before that index.
        static bool has_block_beyond(const String2TensorStorage& tensor_storage_map,
                                     const std::string& base,
                                     const std::string& key_prefix,
                                     int count) {
            const std::string prefix = base + key_prefix;
            for (const auto& entry : tensor_storage_map) {
                if (entry.first.compare(0, prefix.size(), prefix) != 0) {
                    continue;
                }
                int index      = 0;
                bool has_digit = false;
                for (const char* p = entry.first.c_str() + prefix.size(); *p >= '0' && *p <= '9'; ++p) {
                    index     = index * 10 + (*p - '0');
                    has_digit = true;
                }
                if (has_digit && index >= count) {
                    return true;
                }
            }
            return false;
        }

        static bool is_2d_checkpoint(const String2TensorStorage& tensor_storage_map, const std::string& prefix = "") {
            std::string base = prefix;
            if (!base.empty()) {
                base += ".";
            }
            auto iter = tensor_storage_map.find(base + "resizer.conv_in.weight");
            if (iter == tensor_storage_map.end()) {
                return false;
            }
            // The 2D checkpoint nests its network under a `resizer` module and keeps a plain
            // 4D [OC, IC, kH, kW] kernel there, while the 3D checkpoint stores 5D kernels at
            // the root. Either signal alone is enough; both are checked to stay explicit.
            return iter->second.n_dims == 4;
        }

        static H3LatentUpscaler2dConfig detect_from_weights(const String2TensorStorage& tensor_storage_map,
                                                            const std::string& prefix = "") {
            H3LatentUpscaler2dConfig config;
            std::string base = prefix;
            if (!base.empty()) {
                base += ".";
            }
            auto find_weight = [&](const std::string& suffix) -> const TensorStorage* {
                auto iter = tensor_storage_map.find(base + suffix);
                if (iter == tensor_storage_map.end()) {
                    return nullptr;
                }
                return &iter->second;
            };

            const TensorStorage* conv_in = find_weight("resizer.conv_in.weight");
            if (conv_in == nullptr) {
                LOG_ERROR("h3 latent upsampler 2d: 'resizer.conv_in.weight' not found under prefix '%s'", prefix.c_str());
                return config;
            }
            if (conv_in->n_dims != 4) {
                LOG_ERROR("h3 latent upsampler 2d: conv_in.weight must be a 4D 2D-conv kernel, got n_dims=%d", conv_in->n_dims);
                return config;
            }

            // ne is the torch shape reversed: [kW, kH, in, out].
            config.in_channels  = conv_in->ne[2];
            config.mid_channels = conv_in->ne[3];
            if (config.mid_channels <= 0 || config.in_channels <= 0) {
                LOG_ERROR("h3 latent upsampler 2d: could not infer channel counts from the checkpoint");
                return config;
            }

            const TensorStorage* embed_in = find_weight("resizer.embed.linear.0.weight");
            if (embed_in == nullptr || embed_in->n_dims != 2) {
                LOG_ERROR("h3 latent upsampler 2d: 'resizer.embed.linear.0.weight' not found or not 2D");
                return config;
            }
            // Linear weight is [out, in] in torch, so [in, out] once reversed.
            config.embed_dim     = static_cast<int>(embed_in->ne[1]);
            const int64_t in_dim = embed_in->ne[0];
            if (in_dim != 4 * 16 && in_dim != 4 * 16 + 32) {
                LOG_ERROR("h3 latent upsampler 2d: unsupported embedding input width %lld", (long long)in_dim);
                return config;
            }
            config.embed_freq = 16;
            config.tier_dim   = static_cast<int>(in_dim - 4 * config.embed_freq);

            const TensorStorage* dwconv = find_weight("temporal_blocks.0.dwconv.weight");
            if (dwconv == nullptr) {
                LOG_ERROR("h3 latent upsampler 2d: 'temporal_blocks.0.dwconv.weight' not found");
                return config;
            }
            // Depthwise temporal kernel [C, 1, k, 1, 1]: k is ne[2] in every layout that can
            // reach this point, the same axis the 3D variant reads.
            config.temporal_kernel = static_cast<int>(dwconv->ne[2]);
            config.temporal_kernel = std::max(1, config.temporal_kernel);

            int in_res = 0;
            while (find_weight("resizer.in_blocks." + std::to_string(in_res) + ".in_layers.0.weight") != nullptr) {
                ++in_res;
            }
            int out_res = 0;
            while (find_weight("resizer.out_blocks." + std::to_string(out_res) + ".in_layers.0.weight") != nullptr) {
                ++out_res;
            }
            int temporal_blocks = 0;
            while (find_weight("temporal_blocks." + std::to_string(temporal_blocks) + ".dwconv.weight") != nullptr) {
                ++temporal_blocks;
            }
            if (in_res <= 0 || out_res <= 0 || temporal_blocks < 2) {
                LOG_ERROR("h3 latent upsampler 2d: unexpected block layout (in_res=%d out_res=%d temporal=%d)",
                          in_res,
                          out_res,
                          temporal_blocks);
                return config;
            }
            // The counts above stop at the first missing index, so a checkpoint that skips one
            // would silently leave the later blocks unbound. Reject the layout instead.
            if (has_block_beyond(tensor_storage_map, base, "resizer.in_blocks.", in_res) ||
                has_block_beyond(tensor_storage_map, base, "resizer.out_blocks.", out_res) ||
                has_block_beyond(tensor_storage_map, base, "temporal_blocks.", temporal_blocks)) {
                LOG_ERROR("h3 latent upsampler 2d: block indices are not contiguous (in_res=%d out_res=%d temporal=%d)",
                          in_res,
                          out_res,
                          temporal_blocks);
                return config;
            }
            config.in_blocks       = in_res;
            config.out_blocks      = out_res;
            config.temporal_blocks = temporal_blocks;
            config.temporal_every  = 2;

            // Modules the lean checkpoint does not carry. Refuse instead of silently
            // dropping them, since each one changes the forward pass. Every key under the
            // resizer is scanned rather than probing one block index, so a checkpoint that
            // only adds the module on a later block is still rejected.
            const std::string resizer_prefix = base + "resizer.";
            for (const auto& entry : tensor_storage_map) {
                const std::string& key = entry.first;
                if (key.compare(0, resizer_prefix.size(), resizer_prefix) != 0) {
                    continue;
                }
                const std::string rest = key.substr(resizer_prefix.size());
                if (rest.rfind("post_up.", 0) == 0) {
                    LOG_ERROR("h3 latent upsampler 2d: this checkpoint has a post-upsample refine stage, which is not supported");
                    return config;
                }
                if (rest.find(".se.") != std::string::npos) {
                    LOG_ERROR("h3 latent upsampler 2d: this checkpoint uses squeeze-excite blocks, which are not supported");
                    return config;
                }
                if (rest.find(".dw.") != std::string::npos || rest.find(".lk.") != std::string::npos) {
                    LOG_ERROR("h3 latent upsampler 2d: this checkpoint uses large-kernel blocks, which are not supported");
                    return config;
                }
                if (rest.rfind("learned_up.", 0) == 0) {
                    LOG_ERROR("h3 latent upsampler 2d: this checkpoint uses learned pixel-shuffle upsampling, which is not supported");
                    return config;
                }
            }

            config.valid = true;
            LOG_VERBOSE("h3 latent upsampler 2d: in_channels = %" PRId64 ", mid_channels = %" PRId64
                        ", embed_dim = %d (freq = %d, tier = %d), temporal_kernel = %d, in_res = %d, out_res = %d, temporal_blocks = %d",
                        config.in_channels,
                        config.mid_channels,
                        config.embed_dim,
                        config.embed_freq,
                        config.tier_dim,
                        config.temporal_kernel,
                        config.in_blocks,
                        config.out_blocks,
                        config.temporal_blocks);
            return config;
        }
    };

    // GroupNorm over the spatial axes only, i.e. every frame is normalized on its own.
    // x is [W, H, T, C] and ggml_group_norm expects the channel axis at ne[2], so the
    // layout is swapped around the op to keep T out of the statistics.
    class H3FrameGroupNorm : public GGMLBlock {
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
        H3FrameGroupNorm(int num_groups, int64_t num_channels, float eps = 1e-05f)
            : num_groups(num_groups),
              num_channels(num_channels),
              eps(eps) {}

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            GGML_ASSERT(x->ne[3] == num_channels);

            ggml_tensor* xc = ggml_cont(ctx->ggml_ctx, x);
            ggml_tensor* xt = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, xc, 0, 1, 3, 2));
            ggml_tensor* y  = ggml_group_norm(ctx->ggml_ctx, xt, num_groups, eps);
            y               = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, y, 0, 1, 3, 2));

            ggml_tensor* weight = params["weight"];
            ggml_tensor* bias   = params["bias"];
            if (ctx->weight_adapter) {
                weight = ctx->weight_adapter->patch_weight(ctx, weight, prefix + "weight");
                bias   = ctx->weight_adapter->patch_weight(ctx, bias, prefix + "bias");
            }
            weight = ggml_reshape_4d(ctx->ggml_ctx, weight, 1, 1, 1, num_channels);
            bias   = ggml_reshape_4d(ctx->ggml_ctx, bias, 1, 1, 1, num_channels);
            y      = ggml_mul(ctx->ggml_ctx, y, weight);
            return ggml_add(ctx->ggml_ctx, y, bias);
        }
    };

    // 2D convolution applied to every frame of a [W, H, T, C] tensor. Conv2d only
    // understands [W, H, C, N], so the time axis is moved into the batch slot and back.
    class H3Conv2d3d : public Conv2d {
    public:
        using Conv2d::Conv2d;

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            ggml_tensor* xs = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 0, 1, 3, 2));
            ggml_tensor* ys = Conv2d::forward(ctx, xs);
            return ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, ys, 0, 1, 3, 2));
        }
    };

    class H3ResBlock2d : public GGMLBlock {
    protected:
        int64_t channels;
        int64_t embed_dim;

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map,
                         const std::string prefix) override {
            blocks["in_layers.0"]  = std::shared_ptr<GGMLBlock>(new H3FrameGroupNorm(32, channels));
            blocks["in_layers.2"]  = std::shared_ptr<GGMLBlock>(new H3Conv2d3d(channels,
                                                                               channels,
                                                                               std::pair<int, int>{3, 3},
                                                                               std::pair<int, int>{1, 1},
                                                                               std::pair<int, int>{1, 1}));
            blocks["emb_layers.1"] = std::shared_ptr<GGMLBlock>(new Linear(embed_dim, 2 * channels, true));
            blocks["out_norm"]     = std::shared_ptr<GGMLBlock>(new H3FrameGroupNorm(32, channels));
            blocks["out_layers.2"] = std::shared_ptr<GGMLBlock>(new H3Conv2d3d(channels,
                                                                               channels,
                                                                               std::pair<int, int>{3, 3},
                                                                               std::pair<int, int>{1, 1},
                                                                               std::pair<int, int>{1, 1}));
        }

    public:
        H3ResBlock2d(int64_t channels, int64_t embed_dim)
            : channels(channels),
              embed_dim(embed_dim) {}

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* emb) {
            auto in_norm    = std::dynamic_pointer_cast<H3FrameGroupNorm>(blocks["in_layers.0"]);
            auto in_layers  = std::dynamic_pointer_cast<H3Conv2d3d>(blocks["in_layers.2"]);
            auto emb_layers = std::dynamic_pointer_cast<Linear>(blocks["emb_layers.1"]);
            auto out_norm   = std::dynamic_pointer_cast<H3FrameGroupNorm>(blocks["out_norm"]);
            auto out_layers = std::dynamic_pointer_cast<H3Conv2d3d>(blocks["out_layers.2"]);

            ggml_tensor* h = in_norm->forward(ctx, x);
            h              = ggml_silu_inplace(ctx->ggml_ctx, h);
            h              = in_layers->forward(ctx, h);

            // The embedding is shared by every residual block, so it must stay intact.
            ggml_tensor* emb_silu = ggml_silu(ctx->ggml_ctx, emb);
            ggml_tensor* emb_out  = emb_layers->forward(ctx, emb_silu);
            ggml_tensor* emb_1d   = ggml_reshape_1d(ctx->ggml_ctx, ggml_cont(ctx->ggml_ctx, emb_out), 2 * channels);
            ggml_tensor* scale    = ggml_reshape_4d(ctx->ggml_ctx, ggml_ext_slice(ctx, emb_1d, 0, 0, channels), 1, 1, 1, channels);
            ggml_tensor* shift    = ggml_reshape_4d(ctx->ggml_ctx, ggml_ext_slice(ctx, emb_1d, 0, channels, 2 * channels), 1, 1, 1, channels);
            ggml_tensor* gain     = ggml_scale_bias(ctx->ggml_ctx, scale, 1.f, 1.f);

            h = out_norm->forward(ctx, h);
            h = ggml_mul_inplace(ctx->ggml_ctx, h, gain);
            h = ggml_add_inplace(ctx->ggml_ctx, h, shift);
            h = ggml_silu_inplace(ctx->ggml_ctx, h);
            h = out_layers->forward(ctx, h);
            return ggml_add(ctx->ggml_ctx, h, x);
        }
    };

    // Depthwise temporal convolution as shifted multiply-adds, elementwise only so every
    // backend supports it.
    class H3TemporalDWConv2d : public UnaryBlock {
    protected:
        int64_t channels;
        int kernel;
        bool has_bias;
        std::string prefix;

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map,
                         const std::string prefix) override {
            this->prefix     = prefix;
            params["weight"] = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 1, 1, kernel, channels);
            if (has_bias) {
                params["bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, channels);
            }
        }

    public:
        H3TemporalDWConv2d(int64_t channels, int kernel, bool has_bias = true)
            : channels(channels),
              kernel(kernel),
              has_bias(has_bias) {}

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            ggml_tensor* w = params["weight"];
            if (ctx->weight_adapter) {
                w = ctx->weight_adapter->patch_weight(ctx, w, prefix + "weight");
            }
            ggml_tensor* w32 = ggml_cast(ctx->ggml_ctx, w, GGML_TYPE_F32);
            ggml_tensor* xa  = x;
            if (xa->type != GGML_TYPE_F32) {
                xa = ggml_cast(ctx->ggml_ctx, xa, GGML_TYPE_F32);
            }

            const int64_t T = xa->ne[2];
            const int pad   = kernel / 2;

            // ggml_pad only extends the end, so build the symmetric zero padding by hand.
            ggml_tensor* xp = xa;
            if (pad > 0) {
                ggml_tensor* zero = ggml_scale(ctx->ggml_ctx, ggml_ext_slice(ctx, xa, 2, 0, 1), 0.0f);
                ggml_tensor* lead = zero;
                for (int i = 1; i < pad; ++i) {
                    lead = ggml_concat(ctx->ggml_ctx, lead, zero, 2);
                }
                xp = ggml_concat(ctx->ggml_ctx, lead, xa, 2);
                xp = ggml_concat(ctx->ggml_ctx, xp, lead, 2);
            }

            ggml_tensor* acc = nullptr;
            for (int j = 0; j < kernel; ++j) {
                ggml_tensor* xs   = ggml_ext_slice(ctx, xp, 2, j, j + T);
                ggml_tensor* wj   = ggml_ext_slice(ctx, w32, 2, j, j + 1);  // [1, 1, 1, C]
                ggml_tensor* term = ggml_mul(ctx->ggml_ctx, xs, wj);
                acc               = acc == nullptr ? term : ggml_add(ctx->ggml_ctx, acc, term);
            }
            if (has_bias) {
                ggml_tensor* b = params["bias"];
                if (ctx->weight_adapter) {
                    b = ctx->weight_adapter->patch_weight(ctx, b, prefix + "bias");
                }
                acc = ggml_add(ctx->ggml_ctx, acc, ggml_reshape_4d(ctx->ggml_ctx, b, 1, 1, 1, channels));
            }
            return acc;
        }
    };

    class H3TemporalConv2d : public GGMLBlock {
    protected:
        int64_t channels;
        int temporal_kernel;

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map,
                         const std::string prefix) override {
            blocks["norm"]   = std::shared_ptr<GGMLBlock>(new H3FrameGroupNorm(32, channels));
            blocks["dwconv"] = std::shared_ptr<GGMLBlock>(new H3TemporalDWConv2d(channels, temporal_kernel));
            blocks["pwconv"] = std::shared_ptr<GGMLBlock>(new Conv3d(channels,
                                                                     channels,
                                                                     {1, 1, 1},
                                                                     {1, 1, 1},
                                                                     {0, 0, 0}));
        }

    public:
        H3TemporalConv2d(int64_t channels, int temporal_kernel)
            : channels(channels),
              temporal_kernel(temporal_kernel) {}

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto norm   = std::dynamic_pointer_cast<H3FrameGroupNorm>(blocks["norm"]);
            auto dwconv = std::dynamic_pointer_cast<H3TemporalDWConv2d>(blocks["dwconv"]);
            auto pwconv = std::dynamic_pointer_cast<Conv3d>(blocks["pwconv"]);

            ggml_tensor* h = norm->forward(ctx, x);
            h              = ggml_silu_inplace(ctx->ggml_ctx, h);
            h              = dwconv->forward(ctx, h);
            h              = pwconv->forward(ctx, h);
            return ggml_add(ctx->ggml_ctx, h, x);
        }
    };

    // Fourier scale embedding: 4*freq (+tier) features are built on the host and fed in as
    // a graph leaf, then passed through the two-layer MLP.
    class H3FourierEmbed : public GGMLBlock {
    protected:
        int embed_dim;
        int in_dim;

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map,
                         const std::string prefix) override {
            blocks["linear.0"] = std::shared_ptr<GGMLBlock>(new Linear(in_dim, embed_dim, true));
            blocks["linear.2"] = std::shared_ptr<GGMLBlock>(new Linear(embed_dim, embed_dim, true));
        }

    public:
        H3FourierEmbed(int embed_dim, int in_dim)
            : embed_dim(embed_dim),
              in_dim(in_dim) {}

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* features) {
            auto l0        = std::dynamic_pointer_cast<Linear>(blocks["linear.0"]);
            auto l2        = std::dynamic_pointer_cast<Linear>(blocks["linear.2"]);
            ggml_tensor* h = l0->forward(ctx, features);
            h              = ggml_silu_inplace(ctx->ggml_ctx, h);
            return l2->forward(ctx, h);
        }
    };

    // Backbone: everything the checkpoint stores under "resizer.".
    class H3Resizer2d : public GGMLBlock {
    public:
        H3LatentUpscaler2dConfig config;

        explicit H3Resizer2d(H3LatentUpscaler2dConfig config_)
            : config(std::move(config_)) {
            blocks["conv_in"]  = std::shared_ptr<GGMLBlock>(new H3Conv2d3d(config.in_channels,
                                                                           config.mid_channels,
                                                                           std::pair<int, int>{3, 3},
                                                                           std::pair<int, int>{1, 1},
                                                                           std::pair<int, int>{1, 1}));
            blocks["conv_out"] = std::shared_ptr<GGMLBlock>(new H3Conv2d3d(config.mid_channels,
                                                                           config.in_channels,
                                                                           std::pair<int, int>{3, 3},
                                                                           std::pair<int, int>{1, 1},
                                                                           std::pair<int, int>{1, 1}));
            blocks["norm_out"] = std::shared_ptr<GGMLBlock>(new H3FrameGroupNorm(32, config.mid_channels));
            blocks["embed"]    = std::shared_ptr<GGMLBlock>(new H3FourierEmbed(config.embed_dim,
                                                                               4 * config.embed_freq + config.tier_dim));
            for (int i = 0; i < config.in_blocks; ++i) {
                blocks["in_blocks." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new H3ResBlock2d(config.mid_channels,
                                                                                                       config.embed_dim));
            }
            for (int i = 0; i < config.out_blocks; ++i) {
                blocks["out_blocks." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new H3ResBlock2d(config.mid_channels,
                                                                                                        config.embed_dim));
            }
        }

        ggml_tensor* embed_forward(GGMLRunnerContext* ctx, ggml_tensor* features) {
            return std::dynamic_pointer_cast<H3FourierEmbed>(blocks["embed"])->forward(ctx, features);
        }

        ggml_tensor* conv_in_forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            return std::dynamic_pointer_cast<H3Conv2d3d>(blocks["conv_in"])->forward(ctx, x);
        }

        ggml_tensor* in_block_forward(GGMLRunnerContext* ctx, int i, ggml_tensor* h, ggml_tensor* emb) {
            return std::dynamic_pointer_cast<H3ResBlock2d>(blocks["in_blocks." + std::to_string(i)])->forward(ctx, h, emb);
        }

        ggml_tensor* out_block_forward(GGMLRunnerContext* ctx, int i, ggml_tensor* h, ggml_tensor* emb) {
            return std::dynamic_pointer_cast<H3ResBlock2d>(blocks["out_blocks." + std::to_string(i)])->forward(ctx, h, emb);
        }

        ggml_tensor* tail_forward(GGMLRunnerContext* ctx, ggml_tensor* h) {
            h = std::dynamic_pointer_cast<H3FrameGroupNorm>(blocks["norm_out"])->forward(ctx, h);
            h = ggml_silu_inplace(ctx->ggml_ctx, h);
            return std::dynamic_pointer_cast<H3Conv2d3d>(blocks["conv_out"])->forward(ctx, h);
        }
    };

    class H3LatentUpscaler2d : public GGMLBlock {
    public:
        H3LatentUpscaler2dConfig config;

        explicit H3LatentUpscaler2d(H3LatentUpscaler2dConfig config_)
            : config(std::move(config_)) {
            blocks["resizer"] = std::shared_ptr<GGMLBlock>(new H3Resizer2d(config));
            for (int i = 0; i < config.temporal_blocks; ++i) {
                blocks["temporal_blocks." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new H3TemporalConv2d(config.mid_channels,
                                                                                                                 config.temporal_kernel));
            }
        }

        ggml_tensor* forward_stage_a(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* features) {
            auto resizer = std::dynamic_pointer_cast<H3Resizer2d>(blocks["resizer"]);
            auto t0      = std::dynamic_pointer_cast<H3TemporalConv2d>(blocks["temporal_blocks.0"]);

            ggml_tensor* h   = resizer->conv_in_forward(ctx, x);
            ggml_tensor* emb = resizer->embed_forward(ctx, features);

            for (int i = 0; i < config.in_blocks; ++i) {
                h = resizer->in_block_forward(ctx, i, h, emb);
                if (i % config.temporal_every == 0) {
                    h = t0->forward(ctx, h);
                }
                sd::ggml_graph_cut::mark_graph_cut(h, "h3_latent_upscaler_2d.in_blocks." + std::to_string(i), "x");
            }
            return h;
        }

        ggml_tensor* forward_stage_b(GGMLRunnerContext* ctx, ggml_tensor* m, ggml_tensor* features) {
            auto resizer = std::dynamic_pointer_cast<H3Resizer2d>(blocks["resizer"]);
            auto t1      = std::dynamic_pointer_cast<H3TemporalConv2d>(blocks["temporal_blocks.1"]);

            ggml_tensor* g   = m;
            ggml_tensor* emb = resizer->embed_forward(ctx, features);

            for (int i = 0; i < config.out_blocks; ++i) {
                g = resizer->out_block_forward(ctx, i, g, emb);
                if (i % config.temporal_every == 0) {
                    g = t1->forward(ctx, g);
                }
                sd::ggml_graph_cut::mark_graph_cut(g, "h3_latent_upscaler_2d.out_blocks." + std::to_string(i), "x");
            }

            g = resizer->tail_forward(ctx, g);
            sd::ggml_graph_cut::mark_graph_cut(g, "h3_latent_upscaler_2d.conv_out", "x");
            return g;
        }
    };

    struct H3LatentUpscaler2dRunner : public H3LatentUpscalerRunnerBase {
        H3LatentUpscaler2dConfig config;
        std::unique_ptr<H3LatentUpscaler2d> model;
        std::string weight_prefix;

        H3LatentUpscaler2dRunner(ggml_backend_t backend,
                                 const String2TensorStorage& tensor_storage_map,
                                 const std::string& prefix                           = "",
                                 std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
            : H3LatentUpscalerRunnerBase(backend, weight_manager),
              config(H3LatentUpscaler2dConfig::detect_from_weights(tensor_storage_map, prefix)),
              weight_prefix(prefix) {
            if (!config.valid || config.embed_dim <= 0 || config.mid_channels <= 0) {
                LOG_ERROR("unsupported H3 2D latent upsampler config: in_channels=%" PRId64
                          " mid_channels=%" PRId64 " embed_dim=%d temporal_kernel=%d",
                          config.in_channels,
                          config.mid_channels,
                          config.embed_dim,
                          config.temporal_kernel);
                return;
            }
            model = std::make_unique<H3LatentUpscaler2d>(config);
            model->init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return "h3_latent_upscaler_2d";
        }

        bool ready() const override {
            return model != nullptr;
        }

        void collect_params(std::map<std::string, ggml_tensor*>& tensors) override {
            if (model) {
                model->get_param_tensors(tensors, weight_prefix);
            }
        }

        // Fourier features for (scale_h, scale_w, tier), matching the reference
        // EnhancedScaleEmbedding: no pi in the phase, sin/cos in fp32.
        static std::vector<float> make_features(float scale_h, float scale_w, float tier, int freq, int tier_dim) {
            std::vector<float> out(static_cast<size_t>(4 * freq + tier_dim), 0.f);
            for (int i = 0; i < freq; ++i) {
                const float f     = std::pow(2.f, static_cast<float>(i));
                out[i]            = std::sin(scale_h * f);
                out[freq + i]     = std::cos(scale_h * f);
                out[2 * freq + i] = std::sin(scale_w * f);
                out[3 * freq + i] = std::cos(scale_w * f);
                if (tier_dim > 0) {
                    // sin block then cos block, same order as the scale features.
                    out[4 * freq + i]        = std::sin(tier * f);
                    out[4 * freq + freq + i] = std::cos(tier * f);
                }
            }
            return out;
        }

        sd::Tensor<float> compute(const int n_threads,
                                  const sd::Tensor<float>& x,
                                  int64_t target_w,
                                  int64_t target_h) override {
            constexpr const char* label = "H3 2D latent upsampler";
            if (!h3_check_upsampler_input(x, config.in_channels, label) ||
                !h3_check_upsampler_target(target_w, target_h, label)) {
                return {};
            }

            const int64_t in_w  = x.shape()[0];
            const int64_t in_h  = x.shape()[1];
            const int64_t T     = x.shape()[2];
            const float scale_h = static_cast<float>(target_h) / static_cast<float>(in_h);
            const float scale_w = static_cast<float>(target_w) / static_cast<float>(in_w);
            if (scale_h < 1.0f - 1e-3f || scale_w < 1.0f - 1e-3f) {
                LOG_ERROR("%s only supports upscaling (scale >= 1.0), got %.3fx%.3f", label, scale_h, scale_w);
                return {};
            }

            const size_t expected_dim          = static_cast<size_t>(x.dim());
            const sd::Tensor<float> normalized = (x - h3_latent_mean()) / h3_latent_stddev();

            // The checkpoint is trained with a sharpness tier in 1..6, and tier 1 is the
            // zero point of the phase, so the neutral setting contributes nothing here.
            constexpr float tier     = 0.0f;
            std::vector<float> feats = make_features(scale_h, scale_w, tier, config.embed_freq, config.tier_dim);
            const sd::Tensor<float> features({static_cast<int64_t>(feats.size()), 1}, feats);

            sd::Tensor<float> out = run_stages(
                normalized,
                features,
                target_w,
                target_h,
                n_threads,
                [this](GGMLRunnerContext* ctx, ggml_tensor* latent, ggml_tensor* feat) {
                    return model->forward_stage_a(ctx, latent, feat);
                },
                [this](GGMLRunnerContext* ctx, ggml_tensor* latent, ggml_tensor* feat) {
                    return model->forward_stage_b(ctx, latent, feat);
                },
                label);
            if (out.empty()) {
                return {};
            }
            if (out.shape()[0] != target_w || out.shape()[1] != target_h || out.shape()[2] != T) {
                LOG_ERROR("%s output shape mismatch: got %" PRId64 "x%" PRId64 "x%" PRId64 " expected %" PRId64 "x%" PRId64 "x%" PRId64,
                          label,
                          out.shape()[0],
                          out.shape()[1],
                          out.shape()[2],
                          target_w,
                          target_h,
                          T);
                return {};
            }
            return restore_trailing_singleton_dims(out * h3_latent_stddev() + h3_latent_mean(), expected_dim);
        }
    };

    // Builds the runner matching the checkpoint layout. The two generations differ in the
    // rank of their conv_in kernel: the 3D one is stored 5D and folded by the loader, the
    // 2D one stays at 4D and also nests its backbone under a "resizer." prefix.
    inline std::unique_ptr<H3LatentUpscalerRunnerBase> create_upscaler_runner(
        ggml_backend_t backend,
        const String2TensorStorage& tensor_storage_map,
        const std::string& prefix,
        std::shared_ptr<RunnerWeightManager> weight_manager) {
        if (H3LatentUpscaler2dConfig::is_2d_checkpoint(tensor_storage_map, prefix)) {
            return std::make_unique<H3LatentUpscaler2dRunner>(backend, tensor_storage_map, prefix, weight_manager);
        }
        return std::make_unique<H3LatentUpscalerRunner>(backend, tensor_storage_map, prefix, weight_manager);
    }

}  // namespace H3VUpsampler

#endif  // __SD_MODEL_UPSCALER_H3_LATENT_UPSCALER_HPP__
