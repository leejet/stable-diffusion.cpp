#ifndef __SD_MODEL_DIFFUSION_MOSS_IMAGE_HPP__
#define __SD_MODEL_DIFFUSION_MOSS_IMAGE_HPP__

#include <algorithm>
#include <cinttypes>
#include <stdexcept>

#include "core/pe_cache.h"

#include "model/common/ggml_block.hpp"
#include "model/common/rope.hpp"
#include "model/diffusion/model.hpp"
#include "model_loader.h"

namespace MOSSImage {
    struct MOSSImageConfig {
        int64_t dim      = 3072;
        int64_t ffn_dim  = 8192;
        int64_t text_dim = 4096;
        int64_t channels = 128;
        int64_t head_dim = 128;
        int64_t depth    = 42;
        int64_t freq_dim = 256;

        static MOSSImageConfig detect_from_weights(const String2TensorStorage& weights, const std::string& prefix) {
            MOSSImageConfig config;
            int64_t depth = 0;
            for (const auto& [name, tensor] : weights) {
                if (!starts_with(name, prefix + ".")) {
                    continue;
                }
                auto key = name.substr(prefix.size() + 1);
                if (key == "video_embedder.weight") {
                    config.channels = tensor.logical_ne0();
                    config.dim      = tensor.ne[1];
                } else if (key == "text_embedder.proj.weight") {
                    config.text_dim = tensor.logical_ne0();
                } else if (key == "layers.0.proj.ffn_down.weight") {
                    config.ffn_dim = tensor.logical_ne0();
                } else if (key == "layers.0.q_norm.weight") {
                    config.head_dim = tensor.ne[0];
                } else if (key == "time_embedding.0.weight") {
                    config.freq_dim = tensor.logical_ne0();
                }
                if (starts_with(key, "layers.")) {
                    depth = std::max<int64_t>(depth, std::stoi(key.substr(7)) + 1);
                }
            }
            if (depth) {
                config.depth = depth;
                LOG_VERBOSE("moss_image: dim = %" PRId64 ", ffn_dim = %" PRId64 ", depth = %" PRId64 ", head_dim = %" PRId64,
                            config.dim, config.ffn_dim, config.depth, config.head_dim);
            }
            return config;
        }
    };

    struct TextBlock : public GGMLBlock {
        MOSSImageConfig config;

        TextBlock(const MOSSImageConfig& config)
            : config(config) {
            blocks["norm1"]    = std::make_shared<RMSNorm>(config.dim, 1e-6f);
            blocks["norm2"]    = std::make_shared<RMSNorm>(config.dim, 1e-6f);
            blocks["q_norm"]   = std::make_shared<RMSNorm>(config.head_dim, 1e-6f);
            blocks["k_norm"]   = std::make_shared<RMSNorm>(config.head_dim, 1e-6f);
            blocks["qkv"]      = std::make_shared<Linear>(config.dim, config.dim * 3, true);
            blocks["proj"]     = std::make_shared<Linear>(config.dim, config.dim, true);
            blocks["ffn_up"]   = std::make_shared<Linear>(config.dim, config.ffn_dim, true);
            blocks["ffn_gate"] = std::make_shared<Linear>(config.dim, config.ffn_dim, true);
            blocks["ffn_down"] = std::make_shared<Linear>(config.ffn_dim, config.dim, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* pe) {
            auto c      = ctx->ggml_ctx;
            auto linear = [&](const char* name, ggml_tensor* h) { return std::dynamic_pointer_cast<Linear>(blocks[name])->forward(ctx, h); };
            auto norm   = [&](const char* name, ggml_tensor* h) { return std::dynamic_pointer_cast<RMSNorm>(blocks[name])->forward(ctx, h); };
            auto y      = norm("norm1", x);
            auto qkv    = linear("qkv", y);
            auto split  = [&](int i) {
                // q/k feed rms_norm and v a permuted copy, which read these strided views in place.
                return ggml_view_4d(c, qkv, config.head_dim, config.dim / config.head_dim, x->ne[1], x->ne[2], config.head_dim * qkv->nb[0], qkv->nb[1], qkv->nb[2], i * config.dim * qkv->nb[0]);
            };
            auto q = norm("q_norm", split(0));
            auto k = norm("k_norm", split(1));
            auto v = split(2);
            y      = Rope::attention(ctx, q, k, v, pe, nullptr, 1.f, false);
            y      = linear("proj", y);
            x      = ggml_add(c, x, y);
            y      = norm("norm2", x);
            y      = ggml_ext_situ(ctx, linear("ffn_up", y), linear("ffn_gate", y));
            y      = linear("ffn_down", y);
            return ggml_add(c, x, y);
        }
    };

    static ggml_tensor* attend(GGMLRunnerContext* ctx, ggml_tensor* q, ggml_tensor* k, ggml_tensor* v, ggml_tensor* pe, int64_t ref_start) {
        const int64_t tokens = q->ne[2];
        if (ref_start == tokens) {
            return Rope::attention(ctx, q, k, v, pe, nullptr, 1.f, false);
        }

        const int64_t heads = q->ne[1];
        q                   = Rope::apply_rope(ctx, q, pe, false);
        k                   = Rope::apply_rope(ctx, k, pe, false);
        // Image and text queries see the whole sequence. Reference queries see only references.
        auto attention = [&](ggml_tensor* query, ggml_tensor* key, ggml_tensor* value) {
            return ggml_ext_attention_ext(ctx, query, key, value, heads, nullptr, true, ctx->flash_attn_enabled);
        };
        auto body = attention(ggml_ext_slice(ctx, q, 1, 0, ref_start, false), k, v);
        auto refs = attention(ggml_ext_slice(ctx, q, 1, ref_start, tokens, false),
                              ggml_ext_slice(ctx, k, 1, ref_start, tokens, false),
                              ggml_ext_slice(ctx, v, 2, ref_start, tokens, false));
        return ggml_concat(ctx->ggml_ctx, body, refs, 1);
    }

    struct Block : public GGMLBlock {
        MOSSImageConfig config;

        Block(const MOSSImageConfig& config)
            : config(config) {
            blocks["norm1"]            = std::make_shared<RMSNorm>(config.dim, 1e-6f);
            blocks["norm2"]            = std::make_shared<RMSNorm>(config.dim, 1e-6f);
            blocks["q_norm"]           = std::make_shared<RMSNorm>(config.head_dim, 1e-6f);
            blocks["k_norm"]           = std::make_shared<RMSNorm>(config.head_dim, 1e-6f);
            blocks["proj.qkv"]         = std::make_shared<Linear>(config.dim, config.dim * 3, false);
            blocks["proj.o"]           = std::make_shared<Linear>(config.dim, config.dim, false);
            blocks["proj.ffn_up_gate"] = std::make_shared<Linear>(config.dim, config.ffn_dim * 2, false);
            blocks["proj.ffn_down"]    = std::make_shared<Linear>(config.ffn_dim, config.dim, false);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* pe, const std::vector<ggml_tensor*>& mod, int64_t ref_start) {
            auto c      = ctx->ggml_ctx;
            auto linear = [&](const char* name, ggml_tensor* h) { return std::dynamic_pointer_cast<Linear>(blocks[name])->forward(ctx, h); };
            auto norm   = [&](const char* name, ggml_tensor* h) { return std::dynamic_pointer_cast<RMSNorm>(blocks[name])->forward(ctx, h); };
            auto y      = norm("norm1", x);
            y           = ggml_ext_modulate(ctx, y, mod[0], mod[1]);
            auto qkv    = linear("proj.qkv", y);
            auto split  = [&](int i) {
                return ggml_view_4d(c, qkv, config.head_dim, config.dim / config.head_dim, x->ne[1], x->ne[2],
                                    config.head_dim * qkv->nb[0], qkv->nb[1], qkv->nb[2], i * config.dim * qkv->nb[0]);
            };
            auto q       = norm("q_norm", split(0));
            auto k       = norm("k_norm", split(1));
            auto v       = split(2);
            y            = attend(ctx, q, k, v, pe, ref_start);
            y            = linear("proj.o", y);
            y            = ggml_mul(c, y, mod[2]);
            x            = ggml_add(c, x, y);
            y            = norm("norm2", x);
            y            = ggml_ext_modulate(ctx, y, mod[3], mod[4]);
            auto up_gate = linear("proj.ffn_up_gate", y);
            y            = ggml_ext_situ(ctx, up_gate);
            y            = linear("proj.ffn_down", y);
            y            = ggml_mul(c, y, mod[5]);
            return ggml_add(c, x, y);
        }
    };

    struct Model : public GGMLBlock {
        MOSSImageConfig config;

        Model(const MOSSImageConfig& config)
            : config(config) {
            blocks["video_embedder"]     = std::make_shared<Linear>(config.channels, config.dim, true);
            blocks["text_embedder.proj"] = std::make_shared<Linear>(config.text_dim, config.dim, true);
            for (int i = 0; i < 2; ++i) {
                blocks["text_embedder.blocks." + std::to_string(i)] = std::make_shared<TextBlock>(config);
            }
            blocks["time_embedding.0"]        = std::make_shared<Linear>(config.freq_dim, config.dim, true);
            blocks["time_embedding.2"]        = std::make_shared<Linear>(config.dim, config.dim, true);
            blocks["shared_block_modulation"] = std::make_shared<Linear>(config.dim, 6 * config.dim, true);
            blocks["video_head_modulation"]   = std::make_shared<Linear>(config.dim, 2 * config.dim, true);
            for (int64_t i = 0; i < config.depth; ++i) {
                blocks["layers." + std::to_string(i)] = std::make_shared<Block>(config);
            }
            blocks["final_norm_video"]   = std::make_shared<RMSNorm>(config.dim, 1e-6f);
            blocks["final_linear_video"] = std::make_shared<Linear>(config.dim, config.channels, false);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* image, ggml_tensor* time, ggml_tensor* text, ggml_tensor* pe, const std::vector<int>& skip, const std::vector<ggml_tensor*>& refs, ggml_tensor* modulation_rows) {
            auto c           = ctx->ggml_ctx;
            auto linear      = [&](const char* name, ggml_tensor* h) { return std::dynamic_pointer_cast<Linear>(blocks[name])->forward(ctx, h); };
            auto embed_image = [&](ggml_tensor* latent) {
                auto pixels = ggml_reshape_3d(c, latent, latent->ne[0] * latent->ne[1], latent->ne[2], latent->ne[3]);
                return linear("video_embedder", ggml_cont(c, ggml_transpose(c, pixels)));
            };
            auto x                  = embed_image(image);
            const int64_t image_len = x->ne[1];
            text                    = linear("text_embedder.proj", text);
            auto text_pe            = ggml_ext_slice(ctx, pe, 3, image_len, image_len + text->ne[1], false);
            for (int i = 0; i < 2; ++i) {
                text = std::dynamic_pointer_cast<TextBlock>(blocks["text_embedder.blocks." + std::to_string(i)])->forward(ctx, text, text_pe);
            }
            x                       = ggml_concat(c, x, text, 1);
            const int64_t ref_start = x->ne[1];
            for (auto* ref : refs) {
                x = ggml_concat(c, x, embed_image(ref), 1);
            }

            time       = linear("time_embedding.2", ggml_silu(c, linear("time_embedding.0", time)));
            auto head  = linear("video_head_modulation", ggml_ext_slice(ctx, time, 1, 0, 1, false));
            auto parts = ggml_ext_chunk(ctx, linear("shared_block_modulation", time), 6, 0, false);
            for (auto*& part : parts) {
                if (modulation_rows) {
                    part = ggml_get_rows(c, part, modulation_rows);
                }
                if (ctx->graph != nullptr) {
                    ggml_build_forward_expand(ctx->graph, part);
                }
            }

            for (int64_t i = 0; i < config.depth; ++i) {
                if (std::find(skip.begin(), skip.end(), i) == skip.end()) {
                    x = std::dynamic_pointer_cast<Block>(blocks["layers." + std::to_string(i)])->forward(ctx, x, pe, parts, ref_start);
                    sd::ggml_graph_cut::mark_graph_cut(x, "moss_image.layers." + std::to_string(i), "x");
                }
            }
            x          = ggml_ext_slice(ctx, x, 1, 0, image_len, false);
            x          = std::dynamic_pointer_cast<RMSNorm>(blocks["final_norm_video"])->forward(ctx, x);
            auto shift = ggml_ext_slice(ctx, head, 0, 0, config.dim, false);
            auto scale = ggml_ext_slice(ctx, head, 0, config.dim, 2 * config.dim, false);
            x          = ggml_ext_modulate(ctx, x, shift, scale);
            x          = linear("final_linear_video", x);
            return ggml_cont_4d(c, ggml_transpose(c, x), image->ne[0], image->ne[1], image->ne[2], image->ne[3]);
        }
    };

    struct MOSSImageRunner : public DiffusionModelRunner {
        sd::PECache pe_cache_{2};
        MOSSImageConfig config;
        Model model;
        std::vector<float> time_data;

        MOSSImageRunner(ggml_backend_t backend, const String2TensorStorage& weights, const std::string& prefix, std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
            : DiffusionModelRunner(backend, prefix, weight_manager), config(MOSSImageConfig::detect_from_weights(weights, prefix)), model(config) {
            model.init(params_ctx, weights, prefix);
        }

        std::string get_desc() override { return "MOSS-Image"; }
        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix) override {
            model.get_param_tensors(tensors, prefix);
        }

        sd::Tensor<float> compute(int n_threads, const DiffusionParams& inputs) override {
            if (!inputs.x || !inputs.context || !inputs.timesteps || inputs.context->empty()) {
                LOG_ERROR("MOSS-Image requires latents, timesteps and text features");
                return {};
            }
            const auto* extra = std::get_if<SkipLayerDiffusionExtra>(&inputs.extra);
            const std::vector<int> no_skips;
            const auto& skip       = extra && extra->skip_layers ? *extra->skip_layers : no_skips;
            const auto* references = inputs.ref_latents;
            if (references) {
                for (const auto& ref : *references) {
                    if (ref.empty() || ref.dim() != 4 || ref.shape()[2] != config.channels || ref.shape()[3] != 1) {
                        throw std::runtime_error("MOSS-Image reference latents must have shape [width, height, channels, 1] with the model's latent channel count");
                    }
                }
            }
            sd::Tensor<int32_t> modulation_rows;
            auto graph = [&]() {
                auto gf   = new_graph_custom(65536);
                auto x    = make_input(*inputs.x);
                auto text = make_input(*inputs.context);
                GGML_ASSERT(x->ne[3] == 1 && x->ne[2] == config.channels && text->ne[0] == config.text_dim);
                const auto h = x->ne[1], w = x->ne[0], nt = text->ne[1];
                const auto ref_start = h * w + nt;
                auto tokens          = ref_start;
                // ai-toolkit aligns references with image positions. Chinoll aligns them with the start of the text.
                const auto text_offset = references && !references->empty() && inputs.ref_rope_anchor != Rope::RefRoPEAnchor::TEXT_START ? -nt : 0;
                sd::PECacheKey pe_key("moss_image.pe");
                pe_key.add(h).add(w).add(nt).add(text_offset).add(config.head_dim).add(circular_x_enabled).add(circular_y_enabled);
                std::vector<ggml_tensor*> refs;
                if (references) {
                    for (const auto& ref : *references) {
                        refs.push_back(make_input(ref));
                        pe_key.add(ref.shape()[1]).add(ref.shape()[0]);
                        tokens += ref.shape()[1] * ref.shape()[0];
                    }
                }
                auto pe = make_input(pe_cache_.get_or_create(pe_key, [&]() {
                    Rope::Embedding embedding;
                    embedding.ids.reserve(tokens);
                    auto append_image = [&](int64_t height, int64_t width, float ref_index, float text_position) {
                        embedding.positions.append_image(static_cast<int>(height), static_cast<int>(width));
                        auto& region       = embedding.positions.images.back();
                        region.height_axis = 2;
                        region.width_axis  = 3;
                        for (int64_t i = 0; i < height * width; ++i) {
                            embedding.ids.push_back({ref_index, text_position,
                                                     float(i / width) - float(height - 1) / 2.f,
                                                     float(i % width) - float(width - 1) / 2.f});
                        }
                    };
                    append_image(h, w, 0.f, float(nt + text_offset));
                    embedding.positions.append_tokens(nt);
                    for (int64_t i = 0; i < nt; ++i) {
                        embedding.ids.push_back({0.f, float(i + text_offset), 0.f, 0.f});
                    }
                    for (size_t i = 0; i < refs.size(); ++i) {
                        append_image(refs[i]->ne[1], refs[i]->ne[0], float(i + 1), 0.f);
                    }
                    embedding.values = Rope::embed_nd(embedding.ids, 1, 10000.f, std::vector<int>(4, static_cast<int>(config.head_dim / 4)), embedding.layout, &embedding.frequencies);
                    return sd::Tensor<float>({2, 2, config.head_dim / 2, tokens}, finish_rope_pe(std::move(embedding)));
                }));
                const auto half      = config.freq_dim / 2;
                const int time_count = refs.empty() ? 1 : 2;
                time_data.assign(time_count * config.freq_dim, 0.f);
                for (int64_t i = 0; i < half; ++i) {
                    const double angle  = inputs.timesteps->values()[0] * std::pow(10000.0, -double(i) / double(half));
                    time_data[i]        = float(std::cos(angle));
                    time_data[i + half] = float(std::sin(angle));
                }
                ggml_tensor* rows = nullptr;
                if (!refs.empty()) {
                    // The second row is the timestep-zero embedding for reference tokens.
                    std::fill_n(time_data.begin() + config.freq_dim, half, 1.f);
                    modulation_rows = sd::Tensor<int32_t>::zeros({tokens});
                    std::fill_n(modulation_rows.data() + ref_start, tokens - ref_start, 1);
                    rows = make_input(modulation_rows);
                }
                auto time = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, config.freq_dim, time_count);
                set_backend_tensor_data(time, time_data.data());
                auto ctx = get_context(gf);
                ggml_build_forward_expand(gf, model.forward(&ctx, x, time, text, pe, skip, refs, rows));
                return gf;
            };
            return restore_trailing_singleton_dims(GGMLRunner::compute(graph, n_threads, false), inputs.x->dim());
        }
    };
}  // namespace MOSSImage

#endif  // __SD_MODEL_DIFFUSION_MOSS_IMAGE_HPP__
