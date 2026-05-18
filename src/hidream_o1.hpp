#ifndef __SD_HIDREAM_O1_H__
#define __SD_HIDREAM_O1_H__

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common_dit.hpp"
#include "conditioner.hpp"
#include "layer_streaming_executor.hpp"
#include "llm.hpp"
#include "util.h"

namespace HiDreamO1 {
    constexpr int HIDREAM_O1_GRAPH_SIZE = 32768;
    constexpr int PATCH_SIZE            = 32;
    constexpr int TIMESTEP_TOKEN_NUM    = 1;
    constexpr int IMAGE_TOKEN_ID        = 151655;
    constexpr int VISION_START_TOKEN_ID = 151652;

    static inline std::string repeat_special_token(const std::string& token, int64_t count) {
        std::string out;
        out.reserve(static_cast<size_t>(count) * token.size());
        for (int64_t i = 0; i < count; ++i) {
            out += token;
        }
        return out;
    }

    static inline std::pair<int, int> calculate_dimensions(int max_size, double ratio) {
        int width  = static_cast<int>(std::sqrt(max_size * max_size * ratio));
        int height = static_cast<int>(width / ratio);
        width      = (width / PATCH_SIZE) * PATCH_SIZE;
        height     = (height / PATCH_SIZE) * PATCH_SIZE;
        width      = std::max(width, PATCH_SIZE);
        height     = std::max(height, PATCH_SIZE);
        return {width, height};
    }

    static inline sd::Tensor<float> resize_to_area(const sd::Tensor<float>& image, int image_size) {
        int64_t width  = image.shape()[0];
        int64_t height = image.shape()[1];
        int64_t s_max  = static_cast<int64_t>(image_size) * image_size;
        double scale   = std::sqrt(static_cast<double>(s_max) / static_cast<double>(width * height));

        std::vector<std::pair<int64_t, int64_t>> sizes = {
            {(static_cast<int64_t>(std::llround(width * scale)) / PATCH_SIZE) * PATCH_SIZE, (static_cast<int64_t>(std::llround(height * scale)) / PATCH_SIZE) * PATCH_SIZE},
            {(static_cast<int64_t>(std::llround(width * scale)) / PATCH_SIZE) * PATCH_SIZE, (static_cast<int64_t>(std::floor(height * scale)) / PATCH_SIZE) * PATCH_SIZE},
            {(static_cast<int64_t>(std::floor(width * scale)) / PATCH_SIZE) * PATCH_SIZE, (static_cast<int64_t>(std::llround(height * scale)) / PATCH_SIZE) * PATCH_SIZE},
            {(static_cast<int64_t>(std::floor(width * scale)) / PATCH_SIZE) * PATCH_SIZE, (static_cast<int64_t>(std::floor(height * scale)) / PATCH_SIZE) * PATCH_SIZE},
        };
        std::sort(sizes.begin(), sizes.end(), [](const auto& a, const auto& b) {
            return a.first * a.second > b.first * b.second;
        });

        std::pair<int64_t, int64_t> new_size = sizes.back();
        for (const auto& size : sizes) {
            if (size.first > 0 && size.second > 0 && size.first * size.second <= s_max) {
                new_size = size;
                break;
            }
        }

        double s1 = static_cast<double>(width) / static_cast<double>(new_size.first);
        double s2 = static_cast<double>(height) / static_cast<double>(new_size.second);
        sd::Tensor<float> resized;
        if (s1 < s2) {
            int64_t resized_h = static_cast<int64_t>(std::llround(height / s1));
            resized           = sd::ops::interpolate(image,
                                                     {new_size.first, resized_h, image.shape()[2], image.shape()[3]},
                                                     sd::ops::InterpolateMode::Bicubic);
            int64_t top       = (resized_h - new_size.second) / 2;
            resized           = sd::ops::slice(resized, 1, top, top + new_size.second);
        } else {
            int64_t resized_w = static_cast<int64_t>(std::llround(width / s2));
            resized           = sd::ops::interpolate(image,
                                                     {resized_w, new_size.second, image.shape()[2], image.shape()[3]},
                                                     sd::ops::InterpolateMode::Bicubic);
            int64_t left      = (resized_w - new_size.first) / 2;
            resized           = sd::ops::slice(resized, 0, left, left + new_size.first);
        }
        return resized;
    }

    static inline std::vector<int32_t> build_position_ids(const std::vector<int32_t>& input_ids,
                                                          const std::vector<std::array<int32_t, 3>>& image_grids,
                                                          const std::vector<int32_t>& skip_vision_start_token) {
        std::vector<int32_t> position_ids(4 * input_ids.size(), 0);
        int image_index = 0;
        int st          = 0;
        int fix_point   = 4096;
        std::vector<int32_t> out_t;
        std::vector<int32_t> out_h;
        std::vector<int32_t> out_w;

        while (st < static_cast<int>(input_ids.size())) {
            int ed = st;
            while (ed < static_cast<int>(input_ids.size()) && input_ids[ed] != IMAGE_TOKEN_ID) {
                ed++;
            }

            if (ed >= static_cast<int>(input_ids.size())) {
                int st_idx = out_t.empty() ? 0 : (*std::max_element(out_t.begin(), out_t.end()) + 1);
                for (int i = 0; i < static_cast<int>(input_ids.size()) - st; ++i) {
                    out_t.push_back(st_idx + i);
                    out_h.push_back(st_idx + i);
                    out_w.push_back(st_idx + i);
                }
                break;
            }

            int text_len = std::max(0, ed - st - skip_vision_start_token[image_index]);
            int st_idx   = out_t.empty() ? 0 : (*std::max_element(out_t.begin(), out_t.end()) + 1);
            for (int i = 0; i < text_len; ++i) {
                out_t.push_back(st_idx + i);
                out_h.push_back(st_idx + i);
                out_w.push_back(st_idx + i);
            }

            auto grid = image_grids[image_index];
            int base;
            if (skip_vision_start_token[image_index]) {
                if (fix_point > 0) {
                    base      = fix_point;
                    fix_point = 0;
                } else {
                    base = st_idx;
                }
            } else {
                base = text_len + st_idx;
            }
            for (int32_t ti = 0; ti < grid[0]; ++ti) {
                for (int32_t hi = 0; hi < grid[1]; ++hi) {
                    for (int32_t wi = 0; wi < grid[2]; ++wi) {
                        out_t.push_back(base + ti);
                        out_h.push_back(base + hi);
                        out_w.push_back(base + wi);
                    }
                }
            }

            st = ed + grid[0] * grid[1] * grid[2];
            image_index++;
        }

        GGML_ASSERT(out_t.size() == input_ids.size());
        for (size_t i = 0; i < input_ids.size(); ++i) {
            // ggml IMROPE consumes 4 flattened position streams:
            //   [t, h, w, e]
            // llama.cpp's generic Qwen-VL fallback expands text positions as
            // [pos, pos, pos, 0]. Keep the extra stream zeroed here too.
            position_ids[i]                        = out_t[i];
            position_ids[input_ids.size() + i]     = out_h[i];
            position_ids[input_ids.size() * 2 + i] = out_w[i];
            position_ids[input_ids.size() * 3 + i] = 0;
        }
        return position_ids;
    }

    struct TimestepEmbedder : public GGMLBlock {
        int frequency_embedding_size = 256;

        TimestepEmbedder(int64_t hidden_size) {
            blocks["mlp.0"] = std::make_shared<Linear>(frequency_embedding_size, hidden_size, true);
            blocks["mlp.2"] = std::make_shared<Linear>(hidden_size, hidden_size, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* t) {
            auto mlp_0 = std::dynamic_pointer_cast<Linear>(blocks["mlp.0"]);
            auto mlp_2 = std::dynamic_pointer_cast<Linear>(blocks["mlp.2"]);
            auto emb   = ggml_ext_timestep_embedding(ctx->ggml_ctx, t, frequency_embedding_size, 10000, 1000.0f);
            emb        = mlp_0->forward(ctx, emb);
            emb        = ggml_silu_inplace(ctx->ggml_ctx, emb);
            emb        = mlp_2->forward(ctx, emb);
            return emb;
        }
    };

    struct BottleneckPatchEmbed : public GGMLBlock {
        BottleneckPatchEmbed(int64_t in_dim, int64_t pca_dim, int64_t embed_dim) {
            blocks["proj1"] = std::make_shared<Linear>(in_dim, pca_dim, false);
            blocks["proj2"] = std::make_shared<Linear>(pca_dim, embed_dim, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto proj1 = std::dynamic_pointer_cast<Linear>(blocks["proj1"]);
            auto proj2 = std::dynamic_pointer_cast<Linear>(blocks["proj2"]);
            return proj2->forward(ctx, proj1->forward(ctx, x));
        }
    };

    struct FinalLayer : public GGMLBlock {
        FinalLayer(int64_t hidden_size, int64_t out_dim) {
            blocks["linear"] = std::make_shared<Linear>(hidden_size, out_dim, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto linear = std::dynamic_pointer_cast<Linear>(blocks["linear"]);
            return linear->forward(ctx, x);
        }
    };

    struct HiDreamO1Params {
        LLM::LLMParams llm;
        int patch_size = PATCH_SIZE;
    };

    static inline HiDreamO1Params make_hidream_o1_params() {
        HiDreamO1Params params;
        params.llm.arch                           = LLM::LLMArch::QWEN3_VL;
        params.llm.hidden_size                    = 4096;
        params.llm.intermediate_size              = 12288;
        params.llm.num_layers                     = 36;
        params.llm.num_heads                      = 32;
        params.llm.num_kv_heads                   = 8;
        params.llm.head_dim                       = 128;
        params.llm.qkv_bias                       = false;
        params.llm.qk_norm                        = true;
        params.llm.vocab_size                     = 151936;
        params.llm.rms_norm_eps                   = 1e-6f;
        params.llm.vision.arch                    = LLM::LLMVisionArch::QWEN3_VL;
        params.llm.vision.num_layers              = 27;
        params.llm.vision.hidden_size             = 1152;
        params.llm.vision.intermediate_size       = 4304;
        params.llm.vision.num_heads               = 16;
        params.llm.vision.out_hidden_size         = 4096;
        params.llm.vision.patch_size              = 16;
        params.llm.vision.spatial_merge_size      = 2;
        params.llm.vision.temporal_patch_size     = 2;
        params.llm.vision.num_position_embeddings = 2304;
        return params;
    }

    struct HiDreamO1Model : public GGMLBlock {
        HiDreamO1Params params;

        HiDreamO1Model() = default;
        explicit HiDreamO1Model(HiDreamO1Params params)
            : params(std::move(params)) {
            blocks["language_model"] = std::make_shared<LLM::TextModel>(this->params.llm);
            blocks["t_embedder1"]    = std::make_shared<TimestepEmbedder>(this->params.llm.hidden_size);
            blocks["x_embedder"]     = std::make_shared<BottleneckPatchEmbed>(this->params.patch_size * this->params.patch_size * 3,
                                                                          this->params.llm.hidden_size / 4,
                                                                          this->params.llm.hidden_size);
            blocks["final_layer2"]   = std::make_shared<FinalLayer>(this->params.llm.hidden_size,
                                                                  this->params.patch_size * this->params.patch_size * 3);
        }

        std::shared_ptr<LLM::TextModel> text_model() {
            return std::dynamic_pointer_cast<LLM::TextModel>(blocks["language_model"]);
        }

        std::shared_ptr<TimestepEmbedder> timestep_embedder() {
            return std::dynamic_pointer_cast<TimestepEmbedder>(blocks["t_embedder1"]);
        }

        std::shared_ptr<BottleneckPatchEmbed> patch_embedder() {
            return std::dynamic_pointer_cast<BottleneckPatchEmbed>(blocks["x_embedder"]);
        }

        std::shared_ptr<FinalLayer> final_layer() {
            return std::dynamic_pointer_cast<FinalLayer>(blocks["final_layer2"]);
        }
    };

    struct HiDreamO1VisionRunner : public GGMLRunner {
        HiDreamO1Params params;
        std::shared_ptr<LLM::VisionModel> model;

        std::vector<int> window_index_vec;
        std::vector<int> window_inverse_index_vec;
        std::vector<float> window_mask_vec;
        std::vector<float> pe_vec;
        std::array<std::vector<int32_t>, 4> pos_embed_idx_data_;
        std::array<std::vector<float>, 4> pos_embed_weight_data_;

        HiDreamO1VisionRunner(ggml_backend_t backend,
                              ggml_backend_t params_backend,
                              const String2TensorStorage& tensor_storage_map = {},
                              const std::string& prefix                      = "model.visual")
            : GGMLRunner(backend, params_backend),
              params(make_hidream_o1_params()),
              model(std::make_shared<LLM::VisionModel>(false, params.llm.vision)) {
            model->init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return "hidream_o1_vision";
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix = "model.visual") {
            model->get_param_tensors(tensors, prefix);
        }

        ggml_tensor* encode_image(GGMLRunnerContext* runner_ctx, ggml_tensor* image) {
            return LLM::LLMRunner::encode_image_common(this,
                                                       compute_ctx,
                                                       runner_ctx,
                                                       image,
                                                       params.llm.vision,
                                                       model,
                                                       window_index_vec,
                                                       window_inverse_index_vec,
                                                       window_mask_vec,
                                                       pe_vec,
                                                       pos_embed_idx_data_,
                                                       pos_embed_weight_data_);
        }

        ggml_cgraph* build_graph(const sd::Tensor<float>& image_tensor) {
            ggml_cgraph* gf    = new_graph_custom(HIDREAM_O1_GRAPH_SIZE);
            ggml_tensor* image = make_input(image_tensor);
            auto runner_ctx    = get_context();
            auto image_embeds  = encode_image(&runner_ctx, image);
            ggml_build_forward_expand(gf, image_embeds);
            return gf;
        }

        sd::Tensor<float> compute(int n_threads, const sd::Tensor<float>& image) {
            auto get_graph = [&]() {
                return build_graph(image);
            };
            auto output = GGMLRunner::compute<float>(get_graph, n_threads, false);
            return output.has_value() ? std::move(output.value()) : sd::Tensor<float>();
        }
    };

    struct HiDreamO1Runner : public GGMLRunner {
        HiDreamO1Params params;
        HiDreamO1Model model;
        std::string params_prefix;

        std::vector<float> attention_mask_vec;
        std::vector<int32_t> persistent_input_pos_vec;
        std::vector<float> persistent_hidden_fallback;

        // Captured output handles for the three executor stages. build_graph
        // lambdas write these; post_compute lambdas read them.
        ggml_tensor* stage1_hidden_out_ = nullptr;
        ggml_tensor* layer_hidden_out_  = nullptr;
        int64_t      hidden_ne_[4]      = {0, 0, 0, 0};
        float*       persistent_hidden_ = nullptr;

        HiDreamO1Runner(ggml_backend_t backend,
                        ggml_backend_t params_backend,
                        const String2TensorStorage& tensor_storage_map = {},
                        const std::string& prefix                      = "model")
            : GGMLRunner(backend, params_backend),
              params(make_hidream_o1_params()),
              params_prefix(prefix) {
            model = HiDreamO1Model(params);
            model.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return "hidream_o1";
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix) {
            model.get_param_tensors(tensors, prefix);
        }

        int get_num_layers() const {
            return static_cast<int>(params.llm.num_layers);
        }

        ggml_cgraph* build_graph(const sd::Tensor<float>& x_tensor,
                                 const sd::Tensor<float>& timestep_tensor,
                                 const sd::Tensor<int32_t>& input_ids_tensor,
                                 const sd::Tensor<int32_t>& input_pos_tensor,
                                 const sd::Tensor<int32_t>& token_types_tensor,
                                 const sd::Tensor<int32_t>& vinput_mask_tensor,
                                 const std::vector<std::pair<int, sd::Tensor<float>>>& image_embeds_tensor,
                                 const std::vector<sd::Tensor<float>>& ref_images) {
            ggml_cgraph* gf        = new_graph_custom(HIDREAM_O1_GRAPH_SIZE);
            ggml_tensor* x         = make_input(x_tensor);
            ggml_tensor* timestep  = make_input(timestep_tensor);
            ggml_tensor* input_ids = make_input(input_ids_tensor);
            ggml_tensor* input_pos = make_input(input_pos_tensor);

            auto text_model   = model.text_model();
            auto t_embedder1  = model.timestep_embedder();
            auto x_embedder   = model.patch_embedder();
            auto final_layer2 = model.final_layer();

            std::vector<ggml_tensor*> ref_image_tensors;
            for (const auto& image : ref_images) {
                ref_image_tensors.push_back(make_input(image));
            }

            attention_mask_vec    = std::vector<float>(static_cast<size_t>(token_types_tensor.shape()[0] * token_types_tensor.shape()[0]), 0.0f);
            int64_t total_seq_len = token_types_tensor.shape()[0];
            for (int64_t query = 0; query < total_seq_len; ++query) {
                bool is_gen = token_types_tensor.values()[static_cast<size_t>(query)] > 0;
                for (int64_t key = 0; key < total_seq_len; ++key) {
                    if (!is_gen && key > query) {
                        attention_mask_vec[static_cast<size_t>(query * total_seq_len + key)] = -INFINITY;
                    }
                }
            }
            auto attention_mask = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, total_seq_len, total_seq_len);
            set_backend_tensor_data(attention_mask, attention_mask_vec.data());

            auto runner_ctx = get_context();
            auto txt        = text_model->embed(&runner_ctx, input_ids);
            std::vector<std::pair<int, ggml_tensor*>> image_embeds;
            image_embeds.reserve(image_embeds_tensor.size());
            for (const auto& image_embed : image_embeds_tensor) {
                image_embeds.emplace_back(image_embed.first, make_input(image_embed.second));
            }
            txt = LLM::splice_image_embeds(&runner_ctx, txt, image_embeds);

            auto t_emb          = t_embedder1->forward(&runner_ctx, timestep);
            int64_t txt_seq_len = input_ids->ne[0];
            if (txt_seq_len > 1) {
                auto prefix = ggml_ext_slice(compute_ctx, txt, 1, 0, txt_seq_len - 1);
                txt         = ggml_concat(compute_ctx, prefix, ggml_reshape_3d(compute_ctx, t_emb, t_emb->ne[0], 1, 1), 1);
            } else {
                txt = ggml_reshape_3d(compute_ctx, t_emb, t_emb->ne[0], 1, 1);
            }

            auto vinputs          = DiT::pad_and_patchify(&runner_ctx, x, PATCH_SIZE, PATCH_SIZE);
            int64_t target_tokens = vinputs->ne[1];
            for (ggml_tensor* ref_image : ref_image_tensors) {
                auto ref = DiT::pad_and_patchify(&runner_ctx, ref_image, PATCH_SIZE, PATCH_SIZE);
                vinputs  = ggml_concat(compute_ctx, vinputs, ref, 1);
            }
            auto vis = x_embedder->forward(&runner_ctx, vinputs);

            auto inputs_embeds = ggml_concat(compute_ctx, txt, vis, 1);
            auto hidden_states = text_model->forward_embeds(&runner_ctx, inputs_embeds, input_pos, attention_mask, {});
            auto x_pred_all    = final_layer2->forward(&runner_ctx, hidden_states);

            int64_t x_pred_start = txt_seq_len;
            if (!vinput_mask_tensor.empty()) {
                int64_t seq_len      = static_cast<int64_t>(vinput_mask_tensor.shape()[0]);
                int64_t first_vinput = 0;
                while (first_vinput < seq_len && vinput_mask_tensor.values()[static_cast<size_t>(first_vinput)] == 0) {
                    first_vinput++;
                }
                x_pred_start = first_vinput;
            }
            auto x_pred = ggml_ext_slice(compute_ctx, x_pred_all, 1, x_pred_start, x_pred_start + target_tokens);
            x_pred      = DiT::unpatchify_and_crop(compute_ctx, x_pred, x->ne[1], x->ne[0], PATCH_SIZE, PATCH_SIZE);

            float sigma = 1.0f - timestep_tensor.values()[0];
            sigma       = std::max(1e-6f, sigma);
            auto out    = ggml_scale(compute_ctx, ggml_sub(compute_ctx, x, x_pred), 1.0f / sigma);

            ggml_build_forward_expand(gf, out);
            return gf;
        }

        sd::Tensor<float> compute(int n_threads,
                                  const sd::Tensor<float>& x,
                                  const sd::Tensor<float>& timestep,
                                  const sd::Tensor<int32_t>& input_ids,
                                  const sd::Tensor<int32_t>& input_pos,
                                  const sd::Tensor<int32_t>& token_types,
                                  const sd::Tensor<int32_t>& vinput_mask,
                                  const std::vector<std::pair<int, sd::Tensor<float>>>& image_embeds,
                                  const std::vector<sd::Tensor<float>>& ref_images) {
            auto get_graph = [&]() {
                return build_graph(x, timestep, input_ids, input_pos, token_types, vinput_mask, image_embeds, ref_images);
            };
            return restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, false), x.dim());
        }

        // ---- layer streaming -------------------------------------------------

        void enable_layer_streaming(const LayerStreaming::StreamingConfig& config = {}) {
            std::map<std::string, ggml_tensor*> tensor_map;
            model.get_param_tensors(tensor_map, params_prefix);
            init_streaming(config, tensor_map, LayerStreaming::hidream_o1_layer_pattern);
            LOG_INFO("%s layer streaming enabled (%zu layers)",
                     get_desc().c_str(),
                     streaming_engine_->get_registry().get_layer_count());
        }

        // Streaming entry point. Coarse-stage path when everything fits, true
        // per-layer path otherwise. Inputs are sd::Tensors converted to
        // ggml_tensor* metadata (data ptrs point into the original sd::Tensor
        // buffers) by the diffusion-model wrapper's StreamingParamConverter.
        bool compute_streaming(int n_threads,
                               ggml_tensor* x,
                               ggml_tensor* timestep,
                               ggml_tensor* input_ids,
                               ggml_tensor* input_pos,
                               ggml_tensor* token_types,
                               ggml_tensor* vinput_mask,
                               const std::vector<std::pair<int, ggml_tensor*>>& image_embeds,
                               const std::vector<ggml_tensor*>& ref_images,
                               ggml_tensor** output,
                               ggml_context* output_ctx) {
            if (!is_streaming_enabled()) {
                LOG_ERROR("%s streaming not enabled", get_desc().c_str());
                return false;
            }

            int64_t t0    = ggml_time_ms();
            auto analysis = analyze_vram_budget();

            if (analysis.fits_in_vram) {
                LOG_INFO("%s model fits in VRAM, using coarse-stage streaming",
                         get_desc().c_str());
                load_all_layers_coarse();

                // Coarse path delegates to the regular single-graph compute().
                // Materialize ggml inputs back into sd::Tensors that compute()
                // expects. The metadata ggml_tensors here carry data pointers
                // into the original sd::Tensor buffers, so we can just rewrap.
                auto rewrap_float = [](ggml_tensor* t) -> sd::Tensor<float> {
                    if (t == nullptr) return {};
                    sd::Tensor<float> r(sd::shape_from_ggml(t));
                    std::memcpy(r.data(), t->data, r.numel() * sizeof(float));
                    return r;
                };
                auto rewrap_int32 = [](ggml_tensor* t) -> sd::Tensor<int32_t> {
                    if (t == nullptr) return {};
                    sd::Tensor<int32_t> r(sd::shape_from_ggml(t));
                    std::memcpy(r.data(), t->data, r.numel() * sizeof(int32_t));
                    return r;
                };

                std::vector<std::pair<int, sd::Tensor<float>>> image_embeds_sd;
                image_embeds_sd.reserve(image_embeds.size());
                for (auto& ie : image_embeds) {
                    image_embeds_sd.emplace_back(ie.first, rewrap_float(ie.second));
                }
                std::vector<sd::Tensor<float>> ref_images_sd;
                ref_images_sd.reserve(ref_images.size());
                for (auto* r : ref_images) {
                    ref_images_sd.push_back(rewrap_float(r));
                }

                auto result = compute(n_threads,
                                      rewrap_float(x),
                                      rewrap_float(timestep),
                                      rewrap_int32(input_ids),
                                      rewrap_int32(input_pos),
                                      rewrap_int32(token_types),
                                      rewrap_int32(vinput_mask),
                                      image_embeds_sd,
                                      ref_images_sd);

                if (output != nullptr && result.numel() > 0) {
                    if (*output == nullptr && output_ctx != nullptr) {
                        auto shape = result.shape();
                        int n_dims = std::min(static_cast<int>(shape.size()), GGML_MAX_DIMS);
                        std::array<int64_t, GGML_MAX_DIMS> ne = {1, 1, 1, 1};
                        for (int i = 0; i < n_dims; i++) ne[i] = shape[i];
                        *output = ggml_new_tensor(output_ctx, GGML_TYPE_F32, n_dims, ne.data());
                    }
                    if (*output != nullptr) {
                        std::memcpy((*output)->data, result.data(), result.numel() * sizeof(float));
                    }
                }
                int64_t t1 = ggml_time_ms();
                LOG_INFO("%s coarse-stage streaming completed in %.2fs",
                         get_desc().c_str(), (t1 - t0) / 1000.0);
                free_compute_buffer();
                return result.numel() > 0;
            }

            LOG_INFO("%s remaining %.2f GB exceeds available %.2f GB, using per-layer streaming",
                     get_desc().c_str(),
                     analysis.remaining_to_load / (1024.0 * 1024.0 * 1024.0),
                     analysis.available_vram / (1024.0 * 1024.0 * 1024.0));

            return compute_streaming_true(n_threads, x, timestep, input_ids, input_pos,
                                          token_types, vinput_mask, image_embeds, ref_images,
                                          output, output_ctx);
        }

        // True per-layer streaming for HiDream O1.
        //
        // HiDream O1's diffusion forward pass is unusual: it is an LLM
        // transformer applied to a concatenated [text-tokens | image-tokens]
        // sequence, with three small heads on either side (t_embedder,
        // x_embedder up front; final_layer2 + slice + unpatchify at the end).
        //
        // We exploit this by streaming only the language_model.layers.N
        // weights, which dominate the param size. The three stages:
        //
        //   Stage 1  build inputs_embeds (token embed + image splice + t_emb
        //            concat + patchify x + ref concat + x_embedder + final
        //            concat). Output read back to pinned host buffer.
        //   Stage 2  for each LLM layer i: build a graph that runs only
        //            text_model->forward_layer_block(i, x, input_pos, mask).
        //            Stream layer.N weights in, compute, read x out, evict.
        //   Stage 3  build final_norm + final_layer2 + slice + unpatchify +
        //            velocity-prediction graph using the resident _global
        //            weights and the streamed-in head weights.
        //
        // attention_mask and input_pos stay constant for every layer (they
        // depend only on token_types/input_pos inputs), so we precompute them
        // once on host and re-bind their CPU buffer into each per-layer graph.
        bool compute_streaming_true(int n_threads,
                                    ggml_tensor* x_in,
                                    ggml_tensor* timestep_in,
                                    ggml_tensor* input_ids_in,
                                    ggml_tensor* input_pos_in,
                                    ggml_tensor* token_types_in,
                                    ggml_tensor* vinput_mask_in,
                                    const std::vector<std::pair<int, ggml_tensor*>>& image_embeds_in,
                                    const std::vector<ggml_tensor*>& ref_images_in,
                                    ggml_tensor** output,
                                    ggml_context* output_ctx) {
            const int num_layers = get_num_layers();

            // ---- Precompute CPU-side inputs --------------------------------

            // attention_mask: token_types > 0 marks a "gen" token (image side).
            // Non-gen tokens get a causal mask (no attention beyond self);
            // gen tokens see the full sequence. This matches build_graph.
            const int32_t* token_types_data = static_cast<const int32_t*>(token_types_in->data);
            const int64_t total_seq_len     = token_types_in->ne[0];
            attention_mask_vec.assign(static_cast<size_t>(total_seq_len * total_seq_len), 0.0f);
            for (int64_t q = 0; q < total_seq_len; ++q) {
                bool is_gen = token_types_data[q] > 0;
                for (int64_t k = 0; k < total_seq_len; ++k) {
                    if (!is_gen && k > q) {
                        attention_mask_vec[static_cast<size_t>(q * total_seq_len + k)] = -INFINITY;
                    }
                }
            }

            // input_pos: copy CPU-side so per-layer graphs can re-bind it.
            const int32_t* input_pos_data = static_cast<const int32_t*>(input_pos_in->data);
            persistent_input_pos_vec.assign(input_pos_data,
                                            input_pos_data + input_pos_in->ne[0]);

            // Scalars derived from inputs that we need in Stage 3.
            const int64_t txt_seq_len    = input_ids_in->ne[0];
            const int64_t H              = x_in->ne[1];
            const int64_t W              = x_in->ne[0];
            const int64_t target_tokens  = (W / PATCH_SIZE) * (H / PATCH_SIZE);
            int64_t x_pred_start         = txt_seq_len;
            if (vinput_mask_in != nullptr && vinput_mask_in->ne[0] > 0) {
                const int32_t* vmask = static_cast<const int32_t*>(vinput_mask_in->data);
                int64_t seq_len      = vinput_mask_in->ne[0];
                int64_t first_vinput = 0;
                while (first_vinput < seq_len && vmask[first_vinput] == 0) {
                    first_vinput++;
                }
                x_pred_start = first_vinput;
            }
            float sigma = 1.0f - static_cast<const float*>(timestep_in->data)[0];
            sigma       = std::max(1e-6f, sigma);

            // ---- Stage 1: inputs_embeds prelude ----------------------------

            LayerStreaming::Stage input_stage;
            input_stage.build_graph = [&]() -> ggml_cgraph* {
                ggml_cgraph* gf = new_graph_custom(HIDREAM_O1_GRAPH_SIZE / 2);
                auto runner_ctx = get_context();

                ggml_tensor* x_be         = to_backend(x_in);
                ggml_tensor* timestep_be  = to_backend(timestep_in);
                ggml_tensor* input_ids_be = to_backend(input_ids_in);

                auto text_model   = model.text_model();
                auto t_embedder1  = model.timestep_embedder();
                auto x_embedder   = model.patch_embedder();

                std::vector<std::pair<int, ggml_tensor*>> image_embeds_be;
                image_embeds_be.reserve(image_embeds_in.size());
                for (const auto& ie : image_embeds_in) {
                    image_embeds_be.emplace_back(ie.first, to_backend(ie.second));
                }
                std::vector<ggml_tensor*> ref_images_be;
                ref_images_be.reserve(ref_images_in.size());
                for (auto* r : ref_images_in) {
                    ref_images_be.push_back(to_backend(r));
                }

                auto txt = text_model->embed(&runner_ctx, input_ids_be);
                txt      = LLM::splice_image_embeds(&runner_ctx, txt, image_embeds_be);

                auto t_emb = t_embedder1->forward(&runner_ctx, timestep_be);
                if (txt_seq_len > 1) {
                    auto prefix = ggml_ext_slice(compute_ctx, txt, 1, 0, txt_seq_len - 1);
                    txt         = ggml_concat(compute_ctx,
                                              prefix,
                                              ggml_reshape_3d(compute_ctx, t_emb, t_emb->ne[0], 1, 1),
                                              1);
                } else {
                    txt = ggml_reshape_3d(compute_ctx, t_emb, t_emb->ne[0], 1, 1);
                }

                auto vinputs = DiT::pad_and_patchify(&runner_ctx, x_be, PATCH_SIZE, PATCH_SIZE);
                for (ggml_tensor* ref : ref_images_be) {
                    auto rp = DiT::pad_and_patchify(&runner_ctx, ref, PATCH_SIZE, PATCH_SIZE);
                    vinputs = ggml_concat(compute_ctx, vinputs, rp, 1);
                }
                auto vis           = x_embedder->forward(&runner_ctx, vinputs);
                auto inputs_embeds = ggml_concat(compute_ctx, txt, vis, 1);

                // Match LLM::TextModel::forward_embeds: scale the input at the
                // start of the layer loop when the arch requests it. HiDream O1
                // uses Qwen3-VL which has normalize_input=false (default), so
                // this is a no-op today, but keep it for parity with
                // forward_embeds in case the param ever changes.
                if (params.llm.normalize_input) {
                    inputs_embeds = ggml_ext_scale(compute_ctx, inputs_embeds,
                                                    std::sqrt(static_cast<float>(params.llm.hidden_size)),
                                                    true);
                }
                ggml_set_name(inputs_embeds, "inputs_embeds_out");
                stage1_hidden_out_ = inputs_embeds;
                ggml_build_forward_expand(gf, stage1_hidden_out_);
                return gf;
            };
            input_stage.post_compute = [&]() {
                const size_t inputs_embeds_n = ggml_nelements(stage1_hidden_out_);
                for (int i = 0; i < 4; ++i) hidden_ne_[i] = stage1_hidden_out_->ne[i];

                std::vector<float*> ptrs;
                if (ensure_pinned_act_buffers({inputs_embeds_n * sizeof(float)}, ptrs)) {
                    persistent_hidden_ = ptrs[0];
                } else {
                    persistent_hidden_fallback.resize(inputs_embeds_n);
                    persistent_hidden_ = persistent_hidden_fallback.data();
                }
                ggml_backend_tensor_get(stage1_hidden_out_, persistent_hidden_, 0,
                                        inputs_embeds_n * sizeof(float));
            };

            // ---- Stage 2: per-layer factory --------------------------------

            const int64_t total_seq_len_capture = total_seq_len;
            auto layer_factory = [&, total_seq_len_capture](int layer_idx,
                                                            ggml_tensor* prev_gpu_output) -> LayerStreaming::Stage {
                LayerStreaming::Stage s;
                s.build_graph = [&, layer_idx, prev_gpu_output, total_seq_len_capture]() -> ggml_cgraph* {
                    ggml_cgraph* gf = new_graph_custom(HIDREAM_O1_GRAPH_SIZE / 4);
                    auto runner_ctx = get_context();

                    ggml_tensor* hidden_in;
                    if (prev_gpu_output != nullptr) {
                        hidden_in = prev_gpu_output;
                    } else {
                        hidden_in = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32,
                                                       hidden_ne_[0], hidden_ne_[1],
                                                       hidden_ne_[2], hidden_ne_[3]);
                        set_backend_tensor_data(hidden_in, persistent_hidden_);
                    }

                    ggml_tensor* input_pos_t = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_I32,
                                                                  persistent_input_pos_vec.size());
                    set_backend_tensor_data(input_pos_t, persistent_input_pos_vec.data());

                    ggml_tensor* attention_mask_t = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32,
                                                                       total_seq_len_capture, total_seq_len_capture);
                    set_backend_tensor_data(attention_mask_t, attention_mask_vec.data());

                    auto text_model = model.text_model();
                    layer_hidden_out_ = text_model->forward_layer_block(&runner_ctx,
                                                                         layer_idx,
                                                                         hidden_in,
                                                                         input_pos_t,
                                                                         attention_mask_t);
                    ggml_build_forward_expand(gf, layer_hidden_out_);
                    return gf;
                };
                s.post_compute = [&]() {
                    ggml_backend_tensor_get(layer_hidden_out_, persistent_hidden_, 0,
                                            ggml_nbytes(layer_hidden_out_));
                };
                return s;
            };

            // ---- Stage 3: final norm + final_layer2 + slice + unpatchify ---

            LayerStreaming::Stage output_stage;
            output_stage.build_graph = [&]() -> ggml_cgraph* {
                ggml_cgraph* gf = new_graph_custom(HIDREAM_O1_GRAPH_SIZE / 2);
                auto runner_ctx = get_context();

                ggml_tensor* hidden_in = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32,
                                                            hidden_ne_[0], hidden_ne_[1],
                                                            hidden_ne_[2], hidden_ne_[3]);
                set_backend_tensor_data(hidden_in, persistent_hidden_);

                ggml_tensor* x_be = to_backend(x_in);

                auto text_model   = model.text_model();
                auto final_layer2 = model.final_layer();

                auto hs         = text_model->forward_final_norm(&runner_ctx, hidden_in);
                auto x_pred_all = final_layer2->forward(&runner_ctx, hs);
                auto x_pred     = ggml_ext_slice(compute_ctx, x_pred_all, 1,
                                                  x_pred_start, x_pred_start + target_tokens);
                x_pred          = DiT::unpatchify_and_crop(compute_ctx, x_pred, H, W,
                                                            PATCH_SIZE, PATCH_SIZE);

                auto out = ggml_scale(compute_ctx, ggml_sub(compute_ctx, x_be, x_pred), 1.0f / sigma);
                ggml_build_forward_expand(gf, out);
                return gf;
            };
            // output_stage.post_compute left unset; executor writes directly via *output.

            return LayerStreaming::run_streaming(
                this, n_threads, streaming_engine_->get_config(),
                input_stage, layer_factory, output_stage,
                num_layers,
                [](int i) { return "language_model.layers." + std::to_string(i); },
                /*start_layer_idx=*/0,
                output, output_ctx);
        }
    };

    struct HiDreamO1Conditioner : public Conditioner {
        Qwen2Tokenizer tokenizer;
        std::shared_ptr<HiDreamO1VisionRunner> vision_runner;

        HiDreamO1Conditioner(ggml_backend_t backend,
                             ggml_backend_t params_backend,
                             const String2TensorStorage& tensor_storage_map = {})
            : vision_runner(std::make_shared<HiDreamO1VisionRunner>(backend, params_backend, tensor_storage_map)) {}

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
            vision_runner->get_param_tensors(tensors);
        }

        void alloc_params_buffer() override {
            vision_runner->alloc_params_buffer();
        }

        void free_params_buffer() override {
            vision_runner->free_params_buffer();
        }

        size_t get_params_buffer_size() override {
            return vision_runner->get_params_buffer_size();
        }

        void set_max_graph_vram_bytes(size_t max_graph_vram_bytes) override {
            vision_runner->set_max_graph_vram_bytes(max_graph_vram_bytes);
        }

        void set_flash_attention_enabled(bool enabled) override {
            vision_runner->set_flash_attention_enabled(enabled);
        }

        void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override {
            vision_runner->set_weight_adapter(adapter);
        }

        SDCondition get_learned_condition(int n_threads,
                                          const ConditionerParams& conditioner_params) override {
            SDCondition result;

            int width                = conditioner_params.width;
            int height               = conditioner_params.height;
            int64_t target_image_len = static_cast<int64_t>(width / PATCH_SIZE) * static_cast<int64_t>(height / PATCH_SIZE);

            std::vector<sd::Tensor<float>> ref_images;
            if (conditioner_params.ref_images != nullptr) {
                ref_images = *conditioner_params.ref_images;
            }

            std::vector<std::pair<int, sd::Tensor<float>>> vlm_images;
            std::vector<std::array<int32_t, 3>> image_grids;
            std::vector<int32_t> skip_vision_start;

            std::string prompt = "<|im_start|>user\n";

            if (ref_images.empty()) {
                prompt += conditioner_params.text;
                prompt += "<|im_end|>\n<|im_start|>assistant\n<|boi_token|><|tms_token|>";
                auto input_ids = tokenizer.encode(prompt, nullptr);

                std::vector<int32_t> input_ids_pad = input_ids;
                input_ids_pad.push_back(VISION_START_TOKEN_ID);
                input_ids_pad.insert(input_ids_pad.end(), target_image_len - 1, IMAGE_TOKEN_ID);

                image_grids.push_back({1, static_cast<int32_t>(height / PATCH_SIZE), static_cast<int32_t>(width / PATCH_SIZE)});
                skip_vision_start.push_back(1);

                std::vector<int32_t> token_types(input_ids_pad.size(), 0);
                int txt_seq_len = static_cast<int>(input_ids.size());
                int bgn         = txt_seq_len - TIMESTEP_TOKEN_NUM;
                for (int i = bgn; i < static_cast<int>(token_types.size()); ++i) {
                    token_types[i] = 1;
                }

                auto position_ids = build_position_ids(input_ids_pad, image_grids, skip_vision_start);

                std::vector<int64_t> input_shape{static_cast<int64_t>(input_ids.size())};
                std::vector<int64_t> position_shape{static_cast<int64_t>(input_ids_pad.size() * 4)};
                std::vector<int64_t> token_type_shape{static_cast<int64_t>(token_types.size())};
                std::vector<int32_t> vinput_mask(token_types.size(), 0);
                for (int64_t i = txt_seq_len; i < static_cast<int64_t>(vinput_mask.size()); ++i) {
                    vinput_mask[static_cast<size_t>(i)] = 1;
                }
                std::vector<int64_t> vinput_mask_shape{static_cast<int64_t>(vinput_mask.size())};

                result.c_input_ids    = sd::Tensor<int32_t>(input_shape, std::move(input_ids));
                result.c_position_ids = sd::Tensor<int32_t>(position_shape, position_ids);
                result.c_token_types  = sd::Tensor<int32_t>(token_type_shape, std::move(token_types));
                result.c_vinput_mask  = sd::Tensor<int32_t>(vinput_mask_shape, std::move(vinput_mask));
                return result;
            }

            int K = static_cast<int>(ref_images.size());
            int max_size;
            if (K == 1) {
                max_size = std::max(height, width);
            } else if (K == 2) {
                max_size = std::max(height, width) * 48 / 64;
            } else if (K <= 4) {
                max_size = std::max(height, width) / 2;
            } else if (K <= 8) {
                max_size = std::max(height, width) * 24 / 64;
            } else {
                max_size = std::max(height, width) / 4;
            }

            int cond_img_size;
            if (K <= 4) {
                cond_img_size = 384;
            } else if (K <= 8) {
                cond_img_size = 384 * 48 / 64;
            } else {
                cond_img_size = 384 / 2;
            }

            for (const auto& ref_image : ref_images) {
                auto resized_ref = resize_to_area(ref_image, max_size);
                resized_ref      = sd::ops::clamp(resized_ref, 0.0f, 1.0f);

                // VLM image: Qwen3-VL expects mean=[0.5]/std=[0.5] (i.e. range [-1,1]),
                // not CLIP normalization. Resize the already-resized ref directly to
                // (cond_w, cond_h) to match the Python pipeline's pil_r.resize().
                auto dims                   = calculate_dimensions(cond_img_size,
                                                                   static_cast<double>(resized_ref.shape()[0]) / static_cast<double>(resized_ref.shape()[1]));
                sd::Tensor<float> vlm_image = sd::ops::interpolate(
                    resized_ref,
                    {dims.first, dims.second, resized_ref.shape()[2], resized_ref.shape()[3]});
                vlm_image            = vlm_image * 2.0f - 1.0f;
                int64_t image_tokens = static_cast<int64_t>(dims.first / PATCH_SIZE) * static_cast<int64_t>(dims.second / PATCH_SIZE);

                auto patch_img = resized_ref * 2.0f - 1.0f;
                result.c_ref_images.push_back(std::move(patch_img));
                int64_t prompt_start = static_cast<int64_t>(tokenizer.encode(prompt + "<|vision_start|>", nullptr).size());
                prompt += "<|vision_start|>";
                prompt += repeat_special_token("<|image_pad|>", image_tokens);
                prompt += "<|vision_end|>";
                vlm_images.emplace_back(static_cast<int>(prompt_start), std::move(vlm_image));
                image_grids.push_back({1, dims.second / PATCH_SIZE, dims.first / PATCH_SIZE});
                skip_vision_start.push_back(0);
            }

            prompt += conditioner_params.text;
            prompt += "<|im_end|>\n<|im_start|>assistant\n<|boi_token|><|tms_token|>";
            auto input_ids = tokenizer.encode(prompt, nullptr);

            std::vector<int32_t> input_ids_pad = input_ids;
            input_ids_pad.push_back(VISION_START_TOKEN_ID);
            input_ids_pad.insert(input_ids_pad.end(), target_image_len - 1, IMAGE_TOKEN_ID);
            image_grids.push_back({1, static_cast<int32_t>(height / PATCH_SIZE), static_cast<int32_t>(width / PATCH_SIZE)});
            skip_vision_start.push_back(1);

            for (const auto& ref_image : result.c_ref_images) {
                int64_t ref_len = static_cast<int64_t>(ref_image.shape()[0] / PATCH_SIZE) * static_cast<int64_t>(ref_image.shape()[1] / PATCH_SIZE);
                input_ids_pad.push_back(VISION_START_TOKEN_ID);
                input_ids_pad.insert(input_ids_pad.end(), ref_len - 1, IMAGE_TOKEN_ID);
                image_grids.push_back({1, static_cast<int32_t>(ref_image.shape()[1] / PATCH_SIZE), static_cast<int32_t>(ref_image.shape()[0] / PATCH_SIZE)});
                skip_vision_start.push_back(1);
            }

            std::vector<int32_t> token_types(input_ids_pad.size(), 0);
            int txt_seq_len = static_cast<int>(input_ids.size());
            int bgn         = txt_seq_len - TIMESTEP_TOKEN_NUM;
            for (int i = bgn; i < static_cast<int>(token_types.size()); ++i) {
                token_types[i] = 1;
            }

            std::vector<int64_t> input_shape{static_cast<int64_t>(input_ids.size())};
            std::vector<int64_t> position_shape{static_cast<int64_t>(input_ids_pad.size() * 4)};
            std::vector<int64_t> token_type_shape{static_cast<int64_t>(token_types.size())};
            std::vector<int32_t> vinput_mask(token_types.size(), 0);
            for (int i = txt_seq_len; i < static_cast<int>(vinput_mask.size()); ++i) {
                vinput_mask[static_cast<size_t>(i)] = 1;
            }
            std::vector<int64_t> vinput_mask_shape{static_cast<int64_t>(vinput_mask.size())};

            result.c_input_ids    = sd::Tensor<int32_t>(input_shape, std::move(input_ids));
            result.c_position_ids = sd::Tensor<int32_t>(position_shape, build_position_ids(input_ids_pad, image_grids, skip_vision_start));
            result.c_token_types  = sd::Tensor<int32_t>(token_type_shape, std::move(token_types));
            result.c_vinput_mask  = sd::Tensor<int32_t>(vinput_mask_shape, std::move(vinput_mask));
            result.c_image_embeds.reserve(vlm_images.size());
            for (const auto& vlm_image : vlm_images) {
                auto image_embed = vision_runner->compute(n_threads, vlm_image.second);
                if (image_embed.empty()) {
                    LOG_ERROR("hidream_o1 conditioner: encode VLM image failed");
                    return SDCondition();
                }
                result.c_image_embeds.emplace_back(vlm_image.first, std::move(image_embed));
            }
            return result;
        }
    };
}  // namespace HiDreamO1

#endif  // __SD_HIDREAM_O1_H__
