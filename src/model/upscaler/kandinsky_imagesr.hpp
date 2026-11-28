#ifndef __SD_MODEL_UPSCALER_KANDINSKY_IMAGESR_HPP__
#define __SD_MODEL_UPSCALER_KANDINSKY_IMAGESR_HPP__

#include <cinttypes>
#include <limits>
#include <numeric>

#include "model/common/ggml_block.hpp"
#include "model/common/rope.hpp"
#include "model/diffusion/dit.hpp"

struct AutoEncoderKL;
class ModelLoader;

namespace KandinskyImageSR {
    struct KandinskyImageSRConfig {
        int64_t dim, time_dim, ff_dim, rows;
        int layers;
        std::vector<int> axes;
        int head_dim() const { return std::accumulate(axes.begin(), axes.end(), 0); }

        static KandinskyImageSRConfig detect_from_weights(const String2TensorStorage& tensors, int layers, std::vector<int> axes) {
            const auto& table = tensors.at("time_embed_table");
            const auto& ff    = tensors.at("visual_transformer_blocks.0.feed_forward.in_layer.weight");
            KandinskyImageSRConfig config{ff.ne[0], table.ne[0], ff.ne[1], table.ne[1], layers, std::move(axes)};
            LOG_VERBOSE("Kandinsky ImageSR: dim = %" PRId64 ", ff_dim = %" PRId64 ", time_dim = %" PRId64 ", layers = %d",
                        config.dim, config.ff_dim, config.time_dim, layers);
            return config;
        }
    };

    inline ggml_tensor* modulate(ggml_context* ctx, ggml_tensor* x, ggml_tensor* shift, ggml_tensor* scale) {
        return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, 1e-5f), ggml_scale_bias(ctx, scale, 1.f, 1.f)), shift);
    }

    class VisualBlock : public GGMLBlock {
        KandinskyImageSRConfig config;

        void init_params(ggml_context* ctx, const String2TensorStorage&, std::string) override {
            params["xattn_bias_table"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, config.dim, config.rows);
        }

        ggml_tensor* project(GGMLRunnerContext* ctx, const char* name, ggml_tensor* x) {
            return std::static_pointer_cast<UnaryBlock>(blocks.at(name))->forward(ctx, x);
        }

    public:
        explicit VisualBlock(const KandinskyImageSRConfig& config) : config(config) {
            blocks["visual_modulation.out_layer"] = std::make_shared<Linear>(config.time_dim, 9 * config.dim);
            for (const char* name : {"to_query", "to_key", "to_value", "out_layer"}) {
                blocks[std::string("self_attention.") + name] = std::make_shared<Linear>(config.dim, config.dim);
            }
            for (const char* name : {"query_norm", "key_norm"}) {
                blocks[std::string("self_attention.") + name] = std::make_shared<RMSNorm>(config.head_dim(), std::numeric_limits<float>::epsilon());
            }
            blocks["feed_forward.in_layer"]  = std::make_shared<Linear>(config.dim, config.ff_dim, false);
            blocks["feed_forward.out_layer"] = std::make_shared<Linear>(config.ff_dim, config.dim, false);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* time, ggml_tensor* pe, int row) {
            auto g          = ctx->ggml_ctx;
            auto modulation = project(ctx, "visual_modulation.out_layer", time);
            auto part       = [&](int index) {
                return ggml_view_1d(g, modulation, config.dim, index * config.dim * modulation->nb[0]);
            };
            auto h     = modulate(g, x, part(0), part(1));
            auto heads = [&](const char* name) {
                return ggml_reshape_3d(g, project(ctx, name, h), config.head_dim(), config.dim / config.head_dim(), x->ne[1]);
            };
            auto q = project(ctx, "self_attention.query_norm", heads("self_attention.to_query"));
            auto k = project(ctx, "self_attention.key_norm", heads("self_attention.to_key"));
            h      = Rope::attention(ctx, q, k, heads("self_attention.to_value"), pe, nullptr);
            x      = ggml_add(g, x, ggml_mul(g, project(ctx, "self_attention.out_layer", h), part(2)));

            auto table = params.at("xattn_bias_table");
            x          = ggml_add(g, x, ggml_view_1d(g, table, config.dim, row * table->nb[1]));
            h          = modulate(g, x, part(6), part(7));
            h          = ggml_gelu_erf(g, project(ctx, "feed_forward.in_layer", h));
            return ggml_add(g, x, ggml_mul(g, project(ctx, "feed_forward.out_layer", h), part(8)));
        }
    };

    class Transformer : public GGMLBlock {
        KandinskyImageSRConfig config;
        std::vector<std::shared_ptr<VisualBlock>> layers;
        std::shared_ptr<Linear> input, modulation, output;

        void init_params(ggml_context* ctx, const String2TensorStorage&, std::string) override {
            params["time_embed_table"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, config.time_dim, config.rows);
        }

    public:
        explicit Transformer(const KandinskyImageSRConfig& config) : config(config) {
            blocks["visual_embeddings.in_layer"] = input = std::make_shared<Linear>(64, config.dim);
            blocks["out_layer.modulation.out_layer"] = modulation = std::make_shared<Linear>(config.time_dim, 2 * config.dim);
            blocks["out_layer.out_layer"] = output = std::make_shared<Linear>(config.dim, 64);
            for (int i = 0; i < config.layers; ++i) {
                layers.push_back(std::make_shared<VisualBlock>(config));
                blocks["visual_transformer_blocks." + std::to_string(i)] = layers.back();
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* latent, ggml_tensor* pe, int row) {
            auto g     = ctx->ggml_ctx;
            auto table = params.at("time_embed_table");
            auto time  = ggml_silu(g, ggml_view_1d(g, table, config.time_dim, row * table->nb[1]));
            // The input projection packs (patch_y, patch_x, channel), the output packs (channel, patch_y, patch_x).
            auto x = input->forward(ctx, DiT::patchify(g, latent, 2, 2, false));
            for (size_t i = 0; i < layers.size(); ++i) {
                x = layers[i]->forward(ctx, x, time, pe, row);
                sd::ggml_graph_cut::mark_graph_cut(x, "imagesr.block." + std::to_string(i), "x");
            }
            auto m = modulation->forward(ctx, time);
            x      = modulate(g, x, ggml_view_1d(g, m, config.dim, 0), ggml_view_1d(g, m, config.dim, config.dim * m->nb[0]));
            return DiT::unpatchify(g, output->forward(ctx, x), latent->ne[1] / 2, latent->ne[0] / 2, 2, 2, true);
        }
    };

    struct Runner : GGMLRunner {
        Transformer model;
        sd::Tensor<float> pe;

        Runner(ggml_backend_t backend,
               const String2TensorStorage& tensors,
               const KandinskyImageSRConfig& config,
               int latent_side,
               std::shared_ptr<RunnerWeightManager> manager)
            : GGMLRunner(backend, manager), model(config) {
            const auto ids = Rope::gen_flux_img_ids(latent_side, latent_side, 2, 1, 3);
            pe             = sd::Tensor<float>({2, 2, config.head_dim() / 2, static_cast<int64_t>(ids.size())},
                                               Rope::embed_nd(ids, 1, 10000.f, config.axes));
            model.init(params_ctx, tensors);
            set_flash_attention_enabled(true);
        }

        std::string get_desc() override { return "Kandinsky ImageSR"; }

        sd::Tensor<float> predict(int threads, const sd::Tensor<float>& latent, int row) {
            auto graph = [&]() {
                auto gf  = new_graph_custom(1 << 16);
                auto ctx = get_context();
                auto out = model.forward(&ctx, make_input(latent), make_input(pe), row);
                ggml_build_forward_expand(gf, out);
                return gf;
            };
            return restore_trailing_singleton_dims(GGMLRunner::compute(graph, threads), latent.dim());
        }
    };
    bool is_bundle(const std::string& path);

    struct Upscaler {
        std::unique_ptr<Runner> transformer;
        std::shared_ptr<AutoEncoderKL> vae;
        int row_x2, row_x4;

        static std::unique_ptr<Upscaler> load(const std::string& path, ModelLoader& loader, ggml_backend_t backend, std::shared_ptr<RunnerWeightManager> manager, bool direct);
        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors);
        void set_max_graph_vram_bytes(size_t bytes);
        sd::Tensor<float> upscale(int threads, const sd::Tensor<float>& image, int factor);
    };
}  // namespace KandinskyImageSR

#endif  // __SD_MODEL_UPSCALER_KANDINSKY_IMAGESR_HPP__
