#ifndef __SD_MODEL_TE_LTX_CONNECTOR_HPP__
#define __SD_MODEL_TE_LTX_CONNECTOR_HPP__

#include <cinttypes>

#include "core/pe_cache.h"
#include "core/tensor_ggml.hpp"
#include "model/common/ltx.hpp"

namespace LTXV {
    struct LTXConditioningConfig {
        int64_t hidden_size       = 3840;
        int64_t audio_hidden_size = 2048;
        int64_t caption_channels  = 3840;

        bool use_connector                   = false;
        int64_t connector_hidden_size        = 3840;
        int64_t connector_num_heads          = 30;
        int64_t connector_head_dim           = 128;
        int64_t connector_num_layers         = 2;
        int64_t connector_num_registers      = 128;
        bool connector_rope_interleaved      = false;
        bool connector_apply_gated_attention = false;

        bool use_audio_connector                   = false;
        int64_t audio_connector_hidden_size        = 2048;
        int64_t audio_connector_num_heads          = 32;
        int64_t audio_connector_head_dim           = 64;
        int64_t audio_connector_num_layers         = 2;
        int64_t audio_connector_num_registers      = 128;
        bool audio_connector_rope_interleaved      = false;
        bool audio_connector_apply_gated_attention = false;

        bool use_caption_projection          = true;
        bool use_audio_caption_projection    = true;
        bool caption_proj_before_connector   = true;
        bool caption_projection_first_linear = false;

        static LTXConditioningConfig detect_from_weights(const String2TensorStorage& tensor_storage_map, const std::string& prefix) {
            LTXConditioningConfig config;
            auto connector_iter = tensor_storage_map.find(prefix + ".video_embeddings_connector.transformer_1d_blocks.0.attn1.to_q.weight");
            if (connector_iter != tensor_storage_map.end()) {
                config.use_connector         = true;
                config.connector_hidden_size = connector_iter->second.ne[1];
                int64_t connector_heads      = infer_gate_heads(tensor_storage_map,
                                                                prefix + ".video_embeddings_connector.transformer_1d_blocks.0.attn1.to_gate_logits.bias",
                                                                32);
                auto connector_layout        = infer_attention_layout(config.connector_hidden_size, connector_heads);
                config.connector_num_heads   = connector_layout.first;
                config.connector_head_dim    = connector_layout.second;
                config.connector_num_layers  = count_prefix_blocks(tensor_storage_map, prefix + ".video_embeddings_connector.", "transformer_1d_blocks.");
                auto register_iter           = tensor_storage_map.find(prefix + ".video_embeddings_connector.learnable_registers");
                if (register_iter != tensor_storage_map.end()) {
                    config.connector_num_registers = register_iter->second.ne[1];
                }
                if (tensor_storage_map.find(prefix + ".video_embeddings_connector.transformer_1d_blocks.0.attn1.to_gate_logits.weight") != tensor_storage_map.end()) {
                    config.connector_apply_gated_attention = true;
                }
            }

            auto audio_connector_iter = tensor_storage_map.find(prefix + ".audio_embeddings_connector.transformer_1d_blocks.0.attn1.to_q.weight");
            if (audio_connector_iter != tensor_storage_map.end()) {
                config.use_audio_connector         = true;
                config.audio_connector_hidden_size = audio_connector_iter->second.ne[1];
                int64_t connector_heads            = infer_gate_heads(tensor_storage_map,
                                                                      prefix + ".audio_embeddings_connector.transformer_1d_blocks.0.attn1.to_gate_logits.bias",
                                                                      32);
                auto connector_layout              = infer_attention_layout(config.audio_connector_hidden_size, connector_heads);
                config.audio_connector_num_heads   = connector_layout.first;
                config.audio_connector_head_dim    = connector_layout.second;
                config.audio_connector_num_layers  = count_prefix_blocks(tensor_storage_map, prefix + ".audio_embeddings_connector.", "transformer_1d_blocks.");
                auto register_iter                 = tensor_storage_map.find(prefix + ".audio_embeddings_connector.learnable_registers");
                if (register_iter != tensor_storage_map.end()) {
                    config.audio_connector_num_registers = register_iter->second.ne[1];
                }
                if (tensor_storage_map.find(prefix + ".audio_embeddings_connector.transformer_1d_blocks.0.attn1.to_gate_logits.weight") != tensor_storage_map.end()) {
                    config.audio_connector_apply_gated_attention = true;
                }
            }

            const auto video_first              = tensor_storage_map.find(prefix + ".caption_projection.linear_1.weight");
            const auto video_last               = tensor_storage_map.find(prefix + ".caption_projection.linear_2.weight");
            const auto audio_first              = tensor_storage_map.find(prefix + ".audio_caption_projection.linear_1.weight");
            const auto audio_last               = tensor_storage_map.find(prefix + ".audio_caption_projection.linear_2.weight");
            config.use_caption_projection       = video_first != tensor_storage_map.end();
            config.use_audio_caption_projection = audio_first != tensor_storage_map.end();
            const bool has_video_last           = video_last != tensor_storage_map.end();
            const bool has_audio_last           = audio_last != tensor_storage_map.end();
            if ((has_video_last && !config.use_caption_projection) || (has_audio_last && !config.use_audio_caption_projection))
                throw std::runtime_error("LTX caption projection requires its first linear layer");
            if (config.use_caption_projection && config.use_audio_caption_projection &&
                (has_video_last != has_audio_last || video_first->second.logical_ne0() != audio_first->second.logical_ne0()))
                throw std::runtime_error("LTX video and audio caption projections require compatible layouts");
            config.caption_proj_before_connector   = !(has_video_last || has_audio_last);
            config.caption_projection_first_linear = config.use_caption_projection || config.use_audio_caption_projection;
            if (config.use_caption_projection) {
                config.caption_channels = video_first->second.logical_ne0();
                config.hidden_size      = (has_video_last ? video_last : video_first)->second.ne[1];
            } else {
                config.hidden_size = config.connector_hidden_size;
            }
            if (config.use_audio_caption_projection) {
                config.caption_channels  = audio_first->second.logical_ne0();
                config.audio_hidden_size = (has_audio_last ? audio_last : audio_first)->second.ne[1];
            } else {
                config.audio_hidden_size = config.audio_connector_hidden_size;
            }
            LOG_VERBOSE("ltxav_conditioning: video_hidden_size = %" PRId64 ", audio_hidden_size = %" PRId64 ", video_layers = %" PRId64 ", audio_layers = %" PRId64,
                        config.connector_hidden_size, config.audio_connector_hidden_size,
                        config.connector_num_layers, config.audio_connector_num_layers);
            return config;
        }
    };

    __STATIC_INLINE__ std::vector<float> build_1d_rope_matrix(int64_t seq_len,
                                                              int dim,
                                                              int num_heads          = 1,
                                                              float theta            = 10000.f,
                                                              float positional_scale = 4096.f,
                                                              bool double_precision  = false) {
        GGML_ASSERT(dim % num_heads == 0);
        const std::vector<float> indices = double_precision ? std::vector<float>() : generate_freq_grid(theta, 1, dim);
        const std::vector<double> indices_d =
            double_precision ? generate_freq_grid_double(static_cast<double>(theta), 1, dim) : std::vector<double>();
        const int half_dim = dim / 2;
        const int pad_size = half_dim - static_cast<int>(double_precision ? indices_d.size() : indices.size());

        std::vector<std::vector<float>> freqs(static_cast<size_t>(seq_len), std::vector<float>(half_dim, 0.f));
        for (int64_t pos = 0; pos < seq_len; pos++) {
            int out_idx = 0;
            for (int i = 0; i < pad_size; i++) {
                freqs[static_cast<size_t>(pos)][out_idx++] = 0.f;
            }

            if (double_precision) {
                double coord = static_cast<double>(pos) / static_cast<double>(positional_scale);
                for (double index : indices_d) {
                    freqs[static_cast<size_t>(pos)][out_idx++] = static_cast<float>(index * (coord * 2.0 - 1.0));
                }
            } else {
                float coord = static_cast<float>(pos) / positional_scale;
                for (float index : indices) {
                    freqs[static_cast<size_t>(pos)][out_idx++] = index * (coord * 2.f - 1.f);
                }
            }
        }

        if (num_heads > 1) {
            return build_rope_matrix_from_frequencies(split_frequencies_by_heads(freqs, dim, num_heads), dim / num_heads);
        }
        return build_rope_matrix_from_frequencies(freqs, dim);
    }

    struct PixArtAlphaTextProjection : public GGMLBlock {
        PixArtAlphaTextProjection(int64_t in_features,
                                  int64_t hidden_size,
                                  int64_t out_features = -1) {
            if (out_features < 0) {
                out_features = hidden_size;
            }
            blocks["linear_1"] = std::make_shared<Linear>(in_features, hidden_size, true, true);
            blocks["linear_2"] = std::make_shared<Linear>(hidden_size, out_features, true, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* caption) {
            auto linear_1 = std::dynamic_pointer_cast<Linear>(blocks["linear_1"]);
            auto linear_2 = std::dynamic_pointer_cast<Linear>(blocks["linear_2"]);

            caption = linear_1->forward(ctx, caption);
            caption = ggml_ext_gelu(ctx, caption, true);
            caption = linear_2->forward(ctx, caption);
            return caption;
        }
    };

    struct NormSingleLinearTextProjection : public GGMLBlock {
        int64_t in_features;
        int64_t hidden_size;

        NormSingleLinearTextProjection(int64_t in_features,
                                       int64_t hidden_size)
            : in_features(in_features), hidden_size(hidden_size) {
            blocks["linear_1"] = std::make_shared<Linear>(in_features, hidden_size, true, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* caption) {
            auto linear_1 = std::dynamic_pointer_cast<Linear>(blocks["linear_1"]);
            caption       = ggml_rms_norm(ctx->ggml_ctx, caption, 1e-6f);
            caption       = ggml_ext_scale(ctx, caption, std::sqrt(static_cast<float>(hidden_size) / static_cast<float>(in_features)));
            return linear_1->forward(ctx, caption);
        }
    };

    struct BasicTransformerBlock1D : public GGMLBlock {
        BasicTransformerBlock1D(int64_t dim,
                                int64_t n_heads,
                                int64_t d_head,
                                bool rope_interleaved,
                                bool apply_gated_attention = false) {
            blocks["attn1"] = std::make_shared<CrossAttention>(dim, dim, n_heads, d_head, apply_gated_attention, rope_interleaved);
            blocks["ff"]    = std::make_shared<FeedForward>(dim, dim, 4, FeedForward::Activation::GELU);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* pe,
                             ggml_tensor* attention_mask = nullptr) {
            auto attn1 = std::dynamic_pointer_cast<CrossAttention>(blocks["attn1"]);
            auto ff    = std::dynamic_pointer_cast<FeedForward>(blocks["ff"]);

            auto h = rms_norm(ctx->ggml_ctx, x);
            h      = attn1->forward(ctx, h, nullptr, attention_mask, pe);
            x      = ggml_add(ctx->ggml_ctx, x, h);

            h = rms_norm(ctx->ggml_ctx, x);
            h = ff->forward(ctx, h);
            x = ggml_add(ctx->ggml_ctx, x, h);
            return x;
        }
    };

    struct Embeddings1DConnector : public GGMLBlock {
        int64_t hidden_size;
        int64_t num_attention_heads;
        int64_t attention_head_dim;
        int64_t num_layers;
        int64_t num_learnable_registers;
        bool rope_interleaved;
        bool apply_gated_attention;

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            if (num_learnable_registers > 0) {
                params["learnable_registers"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, num_learnable_registers);
            }
        }

        Embeddings1DConnector(int64_t hidden_size,
                              int64_t num_attention_heads     = 30,
                              int64_t attention_head_dim      = 128,
                              int64_t num_layers              = 2,
                              int64_t num_learnable_registers = 128,
                              bool rope_interleaved           = false,
                              bool apply_gated_attention      = false)
            : hidden_size(hidden_size),
              num_attention_heads(num_attention_heads),
              attention_head_dim(attention_head_dim),
              num_layers(num_layers),
              num_learnable_registers(num_learnable_registers),
              rope_interleaved(rope_interleaved),
              apply_gated_attention(apply_gated_attention) {
            for (int i = 0; i < num_layers; i++) {
                blocks["transformer_1d_blocks." + std::to_string(i)] =
                    std::make_shared<BasicTransformerBlock1D>(hidden_size,
                                                              num_attention_heads,
                                                              attention_head_dim,
                                                              rope_interleaved,
                                                              apply_gated_attention);
            }
        }

        ggml_tensor* append_registers(GGMLRunnerContext* ctx,
                                      ggml_tensor* hidden_states) {
            if (num_learnable_registers <= 0 || params.count("learnable_registers") == 0) {
                return hidden_states;
            }

            int64_t seq_len       = hidden_states->ne[1];
            int64_t target_len    = std::max<int64_t>(1024, seq_len);
            int64_t duplications  = (target_len + num_learnable_registers - 1) / num_learnable_registers;
            int64_t total_to_keep = duplications * num_learnable_registers - seq_len;
            if (total_to_keep <= 0) {
                return hidden_states;
            }

            auto regs = ggml_reshape_3d(ctx->ggml_ctx, params["learnable_registers"], hidden_size, num_learnable_registers, 1);
            auto temp = ggml_new_tensor_3d(ctx->ggml_ctx, regs->type, regs->ne[0], regs->ne[1], hidden_states->ne[2]);
            regs      = ggml_repeat(ctx->ggml_ctx, regs, temp);

            auto regs_full = regs;
            for (int64_t i = 1; i < duplications; i++) {
                regs_full = ggml_concat(ctx->ggml_ctx, regs_full, regs, 1);
            }
            regs_full = ggml_ext_slice(ctx, regs_full, 1, seq_len, seq_len + total_to_keep);
            return ggml_concat(ctx->ggml_ctx, hidden_states, regs_full, 1);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* hidden_states,
                             ggml_tensor* pe,
                             ggml_tensor* attention_mask = nullptr) {
            hidden_states = append_registers(ctx, hidden_states);

            for (int i = 0; i < num_layers; i++) {
                auto block    = std::dynamic_pointer_cast<BasicTransformerBlock1D>(blocks["transformer_1d_blocks." + std::to_string(i)]);
                hidden_states = block->forward(ctx, hidden_states, pe, attention_mask);
            }

            return ggml_rms_norm(ctx->ggml_ctx, hidden_states, 1e-6f);
        }
    };

    struct LTXAVConditioningBlock : public GGMLBlock {
        LTXConditioningConfig config;

        LTXAVConditioningBlock(const LTXConditioningConfig& config)
            : config(config) {
            if (config.use_caption_projection) {
                if (config.caption_proj_before_connector) {
                    if (config.caption_projection_first_linear) {
                        blocks["caption_projection"] = std::make_shared<NormSingleLinearTextProjection>(config.caption_channels, config.hidden_size);
                    }
                } else {
                    blocks["caption_projection"] = std::make_shared<PixArtAlphaTextProjection>(config.caption_channels, config.hidden_size, config.hidden_size);
                }
            }
            if (config.use_audio_caption_projection) {
                if (config.caption_proj_before_connector) {
                    if (config.caption_projection_first_linear) {
                        blocks["audio_caption_projection"] = std::make_shared<NormSingleLinearTextProjection>(config.caption_channels, config.audio_hidden_size);
                    }
                } else {
                    blocks["audio_caption_projection"] = std::make_shared<PixArtAlphaTextProjection>(config.caption_channels, config.audio_hidden_size, config.audio_hidden_size);
                }
            }

            if (config.use_connector) {
                blocks["video_embeddings_connector"] = std::make_shared<Embeddings1DConnector>(config.connector_hidden_size,
                                                                                               config.connector_num_heads,
                                                                                               config.connector_head_dim,
                                                                                               config.connector_num_layers,
                                                                                               config.connector_num_registers,
                                                                                               config.connector_rope_interleaved,
                                                                                               config.connector_apply_gated_attention);
            }
            if (config.use_audio_connector) {
                blocks["audio_embeddings_connector"] = std::make_shared<Embeddings1DConnector>(config.audio_connector_hidden_size,
                                                                                               config.audio_connector_num_heads,
                                                                                               config.audio_connector_head_dim,
                                                                                               config.audio_connector_num_layers,
                                                                                               config.audio_connector_num_registers,
                                                                                               config.audio_connector_rope_interleaved,
                                                                                               config.audio_connector_apply_gated_attention);
            }
        }

        std::pair<ggml_tensor*, ggml_tensor*> process_contexts(GGMLRunnerContext* ctx,
                                                               ggml_tensor* v_context,
                                                               ggml_tensor* a_context,
                                                               ggml_tensor* video_connector_pe,
                                                               ggml_tensor* audio_connector_pe) {
            const bool process_audio_context = a_context != nullptr;
            if (config.caption_proj_before_connector) {
                if (config.use_caption_projection &&
                    blocks.count("caption_projection") > 0 &&
                    v_context != nullptr &&
                    v_context->ne[0] == config.caption_channels) {
                    auto caption_projection = std::dynamic_pointer_cast<NormSingleLinearTextProjection>(blocks["caption_projection"]);
                    if (caption_projection != nullptr) {
                        v_context = caption_projection->forward(ctx, v_context);
                    }
                }
                if (process_audio_context &&
                    config.use_audio_caption_projection &&
                    blocks.count("audio_caption_projection") > 0 &&
                    a_context != nullptr &&
                    a_context->ne[0] == config.caption_channels) {
                    auto caption_projection = std::dynamic_pointer_cast<NormSingleLinearTextProjection>(blocks["audio_caption_projection"]);
                    if (caption_projection != nullptr) {
                        a_context = caption_projection->forward(ctx, a_context);
                    }
                }
            }

            if (config.use_connector && v_context != nullptr && v_context->ne[0] == config.connector_hidden_size) {
                auto connector = std::dynamic_pointer_cast<Embeddings1DConnector>(blocks["video_embeddings_connector"]);
                v_context      = connector->forward(ctx, v_context, video_connector_pe);
            }
            if (process_audio_context &&
                config.use_audio_connector &&
                a_context != nullptr &&
                a_context->ne[0] == config.audio_connector_hidden_size) {
                auto connector = std::dynamic_pointer_cast<Embeddings1DConnector>(blocks["audio_embeddings_connector"]);
                a_context      = connector->forward(ctx, a_context, audio_connector_pe);
            }

            if (!config.caption_proj_before_connector &&
                config.use_caption_projection &&
                blocks.count("caption_projection") > 0 &&
                v_context != nullptr &&
                v_context->ne[0] == config.caption_channels) {
                auto caption_projection = std::dynamic_pointer_cast<PixArtAlphaTextProjection>(blocks["caption_projection"]);
                if (caption_projection != nullptr) {
                    v_context = caption_projection->forward(ctx, v_context);
                }
            }
            if (process_audio_context &&
                !config.caption_proj_before_connector &&
                config.use_audio_caption_projection &&
                blocks.count("audio_caption_projection") > 0 &&
                a_context != nullptr &&
                a_context->ne[0] == config.caption_channels) {
                auto caption_projection = std::dynamic_pointer_cast<PixArtAlphaTextProjection>(blocks["audio_caption_projection"]);
                if (caption_projection != nullptr) {
                    a_context = caption_projection->forward(ctx, a_context);
                }
            }

            return {v_context, a_context};
        }
    };

    struct LTXConditioningRunner : public GGMLRunner {
        static constexpr const char* prefix = "text_encoders.connector";
        LTXAVConditioningBlock model;
        sd::PECache pe_cache;
        int64_t video_projection_size = 0;
        int64_t audio_projection_size = 0;

        LTXConditioningRunner(ggml_backend_t backend, const String2TensorStorage& tensors, std::shared_ptr<RunnerWeightManager> manager, const std::string& projector_prefix = "text_encoders.connector.text_embedding_projection")
            : GGMLRunner(backend, std::move(manager)),
              model(LTXConditioningConfig::detect_from_weights(tensors, prefix)) {
            if (!model.config.use_connector || !model.config.use_audio_connector)
                throw std::runtime_error("LTX text encoding requires video and audio connector weights in the model sources");
            auto video = tensors.find(projector_prefix + ".video_aggregate_embed.weight");
            if (video != tensors.end()) {
                auto audio = tensors.find(projector_prefix + ".audio_aggregate_embed.weight");
                if (audio == tensors.end())
                    throw std::runtime_error("LTX dual text projection requires both video and audio weights");
                video_projection_size = video->second.ne[1];
                audio_projection_size = audio->second.ne[1];
            }
            model.init(params_ctx, tensors, prefix);
        }

        std::string get_desc() override { return "ltxav_conditioning"; }

        ggml_tensor* connector_pe(int64_t length, bool audio) {
            const auto& config = model.config;
            if (!(audio ? config.use_audio_connector : config.use_connector))
                return nullptr;
            const auto registers   = audio ? config.audio_connector_num_registers : config.connector_num_registers;
            const auto hidden      = audio ? config.audio_connector_hidden_size : config.connector_hidden_size;
            const auto heads       = audio ? config.audio_connector_num_heads : config.connector_num_heads;
            const auto head_dim    = audio ? config.audio_connector_head_dim : config.connector_head_dim;
            const auto full_length = registers > 0 ? ((std::max<int64_t>(1024, length) + registers - 1) / registers) * registers : length;
            sd::PECacheKey key(audio ? "ltxav_audio_connector_pe" : "ltxav_video_connector_pe");
            key.add(full_length).add(hidden).add(heads).add(head_dim);
            return make_input(pe_cache.get_or_create(key, [&] {
                return sd::Tensor<float>({2, 2, head_dim / 2, full_length * heads},
                                         build_1d_rope_matrix(full_length, static_cast<int>(hidden), static_cast<int>(heads), 10000.f, 4096.f, true));
            }));
        }

        std::pair<sd::Tensor<float>, sd::Tensor<float>> compute(int n_threads, const sd::Tensor<float>& projected) {
            if (projected.empty() || projected.dim() > 3)
                throw std::runtime_error("LTX text projection must produce a sequence of embeddings");
            std::pair<sd::Tensor<float>, sd::Tensor<float>> result;
            std::pair<ggml_tensor*, ggml_tensor*> outputs;
            const auto get_graph = [&] {
                auto graph   = new_graph_custom(LTXAV_GRAPH_SIZE);
                auto input   = make_input(projected);
                auto video   = input;
                auto audio   = input;
                auto context = get_context();
                if (video_projection_size) {
                    if (input->ne[0] != video_projection_size + audio_projection_size)
                        throw std::runtime_error("LTX dual text projection dimensions do not match its weights");
                    video = ggml_ext_slice(&context, input, 0, 0, video_projection_size);
                    audio = ggml_ext_slice(&context, input, 0, video_projection_size, video_projection_size + audio_projection_size);
                }
                outputs = model.process_contexts(&context, video, audio, connector_pe(input->ne[1], false), connector_pe(input->ne[1], true));
                if (outputs.first->ne[0] != model.config.hidden_size || outputs.second->ne[0] != model.config.audio_hidden_size)
                    throw std::runtime_error("LTX conditioning output dimensions do not match its projections");
                for (auto output : {outputs.first, outputs.second}) {
                    ggml_set_output(output);
                    ggml_build_forward_expand(graph, output);
                }
                return graph;
            };
            const auto read_outputs = [&] {
                result.first  = sd::make_sd_tensor_from_ggml<float>(outputs.first);
                result.second = sd::make_sd_tensor_from_ggml<float>(outputs.second);
                return !result.first.empty() && !result.second.empty();
            };
            if (!GGMLRunner::compute(get_graph, n_threads, false, true, read_outputs))
                return {};
            return result;
        }
    };

}  // namespace LTXV

#endif  // __SD_MODEL_TE_LTX_CONNECTOR_HPP__
