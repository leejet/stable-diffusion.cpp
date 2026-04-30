#ifndef __CONDITIONER_HPP__
#define __CONDITIONER_HPP__

#include <optional>

#include "clip.hpp"
#include "llm.hpp"
#include "ltx_connector.hpp"
#include "t5.hpp"
#include "tensor_ggml.hpp"
#include "tokenizers/gemma_tokenizer.h"

struct SDCondition {
    sd::Tensor<float> c_crossattn;
    sd::Tensor<float> c_vector;
    sd::Tensor<float> c_concat;
    sd::Tensor<int32_t> c_t5_ids;
    sd::Tensor<float> c_t5_weights;

    std::vector<sd::Tensor<float>> extra_c_crossattns;

    SDCondition() = default;

    SDCondition(sd::Tensor<float> c_crossattn,
                sd::Tensor<float> c_vector,
                sd::Tensor<float> c_concat)
        : c_crossattn(std::move(c_crossattn)), c_vector(std::move(c_vector)), c_concat(std::move(c_concat)) {}

    bool empty() const {
        if (!c_crossattn.empty() || !c_vector.empty() || !c_concat.empty() ||
            !c_t5_ids.empty() || !c_t5_weights.empty()) {
            return false;
        }

        for (const auto& tensor : extra_c_crossattns) {
            if (!tensor.empty()) {
                return false;
            }
        }

        return true;
    }
};

static inline sd::Tensor<float> apply_token_weights(sd::Tensor<float> hidden_states,
                                                    const std::vector<float>& weights) {
    if (hidden_states.empty()) {
        return hidden_states;
    }

    if (hidden_states.dim() == 1) {
        hidden_states.unsqueeze_(1);
    }

    GGML_ASSERT(static_cast<size_t>(hidden_states.shape()[1]) == weights.size());

    float original_mean = hidden_states.mean();
    auto chunk_weights  = sd::Tensor<float>::from_vector(weights);
    chunk_weights.reshape_({1, static_cast<int64_t>(weights.size())});
    hidden_states *= chunk_weights;
    float new_mean = hidden_states.mean();
    if (new_mean != 0.0f) {
        hidden_states *= (original_mean / new_mean);
    }

    return hidden_states;
}

struct ConditionerParams {
    std::string text;
    int clip_skip                                    = -1;
    int width                                        = -1;
    int height                                       = -1;
    int adm_in_channels                              = -1;
    bool zero_out_masked                             = false;
    int num_input_imgs                               = 0;        // for photomaker
    const std::vector<sd::Tensor<float>>* ref_images = nullptr;  // for qwen image edit
};

struct Conditioner {
    virtual ~Conditioner() = default;

public:
    virtual SDCondition get_learned_condition(int n_threads,
                                              const ConditionerParams& conditioner_params) = 0;
    virtual void alloc_params_buffer()                                                     = 0;
    virtual void free_params_buffer()                                                      = 0;
    virtual void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors)           = 0;
    virtual size_t get_params_buffer_size()                                                = 0;
    virtual void set_flash_attention_enabled(bool enabled)                                 = 0;
    virtual void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) {}
    virtual std::tuple<SDCondition, std::vector<bool>> get_learned_condition_with_trigger(int n_threads,
                                                                                          const ConditionerParams& conditioner_params) {
        GGML_ABORT("Not implemented yet!");
    }
    virtual std::string remove_trigger_from_prompt(const std::string& prompt) {
        GGML_ABORT("Not implemented yet!");
    }
    // Lazy-load hook on the LLM/Gemma side. Default no-op for conditioners
    // whose state is too small to need it. Overridden by LTX2GemmaConditioner.
    virtual void set_llm_lazy_load(std::function<bool()> /*fn*/) {}
    // Tensor-map split for the lazy LLM path: populate `tensors` with ONLY the
    // LLM's tensors (so the lazy callback knows what to load) or with
    // EVERYTHING EXCEPT the LLM (so the global eager load skips them). Default
    // no-op / delegates to get_param_tensors for conditioners without an LLM
    // split point.
    virtual void get_llm_param_tensors(std::map<std::string, ggml_tensor*>& /*tensors*/) {}
    virtual void get_non_llm_param_tensors(std::map<std::string, ggml_tensor*>& tensors) {
        get_param_tensors(tensors);
    }
};

// ldm.modules.encoders.modules.FrozenCLIPEmbedder
// Ref: https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/cad87bf4e3e0b0a759afa94e933527c3123d59bc/modules/sd_hijack_clip.py#L283
struct FrozenCLIPEmbedderWithCustomWords : public Conditioner {
    SDVersion version    = VERSION_SD1;
    PMVersion pm_version = PM_VERSION_1;
    CLIPTokenizer tokenizer;
    std::shared_ptr<CLIPTextModelRunner> text_model;
    std::shared_ptr<CLIPTextModelRunner> text_model2;

    std::string trigger_word = "img";  // should be user settable
    std::map<std::string, std::string> embedding_map;
    int32_t num_custom_embeddings   = 0;
    int32_t num_custom_embeddings_2 = 0;
    std::vector<uint8_t> token_embed_custom;
    std::map<std::string, std::pair<int, int>> embedding_pos_map;

    FrozenCLIPEmbedderWithCustomWords(ggml_backend_t backend,
                                      bool offload_params_to_cpu,
                                      const String2TensorStorage& tensor_storage_map,
                                      const std::map<std::string, std::string>& orig_embedding_map,
                                      SDVersion version = VERSION_SD1,
                                      PMVersion pv      = PM_VERSION_1)
        : version(version), pm_version(pv), tokenizer(sd_version_is_sd2(version) ? 0 : 49407) {
        for (const auto& kv : orig_embedding_map) {
            std::string name = kv.first;
            std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return std::tolower(c); });
            embedding_map[name] = kv.second;
            tokenizer.add_special_token(name);
        }
        bool force_clip_f32 = !embedding_map.empty();
        if (sd_version_is_sd1(version)) {
            text_model = std::make_shared<CLIPTextModelRunner>(backend, offload_params_to_cpu, tensor_storage_map, "cond_stage_model.transformer.text_model", OPENAI_CLIP_VIT_L_14, true, force_clip_f32);
        } else if (sd_version_is_sd2(version)) {
            text_model = std::make_shared<CLIPTextModelRunner>(backend, offload_params_to_cpu, tensor_storage_map, "cond_stage_model.transformer.text_model", OPEN_CLIP_VIT_H_14, true, force_clip_f32);
        } else if (sd_version_is_sdxl(version)) {
            text_model  = std::make_shared<CLIPTextModelRunner>(backend, offload_params_to_cpu, tensor_storage_map, "cond_stage_model.transformer.text_model", OPENAI_CLIP_VIT_L_14, false, force_clip_f32);
            text_model2 = std::make_shared<CLIPTextModelRunner>(backend, offload_params_to_cpu, tensor_storage_map, "cond_stage_model.1.transformer.text_model", OPEN_CLIP_VIT_BIGG_14, false, force_clip_f32);
        }
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
        text_model->get_param_tensors(tensors, "cond_stage_model.transformer.text_model");
        if (sd_version_is_sdxl(version)) {
            text_model2->get_param_tensors(tensors, "cond_stage_model.1.transformer.text_model");
        }
    }

    void alloc_params_buffer() override {
        text_model->alloc_params_buffer();
        if (sd_version_is_sdxl(version)) {
            text_model2->alloc_params_buffer();
        }
    }

    void free_params_buffer() override {
        text_model->free_params_buffer();
        if (sd_version_is_sdxl(version)) {
            text_model2->free_params_buffer();
        }
    }

    size_t get_params_buffer_size() override {
        size_t buffer_size = text_model->get_params_buffer_size();
        if (sd_version_is_sdxl(version)) {
            buffer_size += text_model2->get_params_buffer_size();
        }
        return buffer_size;
    }

    void set_flash_attention_enabled(bool enabled) override {
        text_model->set_flash_attention_enabled(enabled);
        if (sd_version_is_sdxl(version)) {
            text_model2->set_flash_attention_enabled(enabled);
        }
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override {
        text_model->set_weight_adapter(adapter);
        if (sd_version_is_sdxl(version)) {
            text_model2->set_weight_adapter(adapter);
        }
    }

    bool load_embedding(std::string embd_name, std::string embd_path, std::vector<int32_t>& bpe_tokens) {
        ModelLoader model_loader;
        if (!model_loader.init_from_file_and_convert_name(embd_path)) {
            LOG_ERROR("embedding '%s' failed", embd_name.c_str());
            return false;
        }
        auto iter = embedding_pos_map.find(embd_name);
        if (iter != embedding_pos_map.end()) {
            LOG_DEBUG("embedding already read in: %s", embd_name.c_str());
            for (int i = iter->second.first; i < iter->second.second; i++) {
                bpe_tokens.push_back(text_model->model.vocab_size + i);
            }
            return true;
        }
        ggml_init_params params;
        params.mem_size        = 100 * 1024 * 1024;  // max for custom embeddings 100 MB
        params.mem_buffer      = nullptr;
        params.no_alloc        = false;
        ggml_context* embd_ctx = ggml_init(params);
        ggml_tensor* embd      = nullptr;
        ggml_tensor* embd2     = nullptr;
        auto on_load           = [&](const TensorStorage& tensor_storage, ggml_tensor** dst_tensor) {
            if (tensor_storage.ne[0] != text_model->model.hidden_size) {
                if (text_model2) {
                    if (tensor_storage.ne[0] == text_model2->model.hidden_size) {
                        embd2       = ggml_new_tensor_2d(embd_ctx, tensor_storage.type, text_model2->model.hidden_size, tensor_storage.n_dims > 1 ? tensor_storage.ne[1] : 1);
                        *dst_tensor = embd2;
                    } else {
                        LOG_DEBUG("embedding wrong hidden size, got %i, expected %i or %i", tensor_storage.ne[0], text_model->model.hidden_size, text_model2->model.hidden_size);
                        return false;
                    }
                } else {
                    LOG_DEBUG("embedding wrong hidden size, got %i, expected %i", tensor_storage.ne[0], text_model->model.hidden_size);
                    return false;
                }
            } else {
                embd        = ggml_new_tensor_2d(embd_ctx, tensor_storage.type, text_model->model.hidden_size, tensor_storage.n_dims > 1 ? tensor_storage.ne[1] : 1);
                *dst_tensor = embd;
            }
            return true;
        };
        model_loader.load_tensors(on_load, 1);
        int pos_start = num_custom_embeddings;
        if (embd) {
            int64_t hidden_size = text_model->model.hidden_size;
            token_embed_custom.resize(token_embed_custom.size() + ggml_nbytes(embd));
            memcpy((void*)(token_embed_custom.data() + num_custom_embeddings * hidden_size * ggml_type_size(embd->type)),
                   embd->data,
                   ggml_nbytes(embd));
            for (int i = 0; i < embd->ne[1]; i++) {
                bpe_tokens.push_back(text_model->model.vocab_size + num_custom_embeddings);
                // LOG_DEBUG("new custom token: %i", text_model.vocab_size + num_custom_embeddings);
                num_custom_embeddings++;
            }
            LOG_DEBUG("embedding '%s' applied, custom embeddings: %i", embd_name.c_str(), num_custom_embeddings);
        }
        if (embd2) {
            int64_t hidden_size = text_model2->model.hidden_size;
            token_embed_custom.resize(token_embed_custom.size() + ggml_nbytes(embd2));
            memcpy((void*)(token_embed_custom.data() + num_custom_embeddings_2 * hidden_size * ggml_type_size(embd2->type)),
                   embd2->data,
                   ggml_nbytes(embd2));
            for (int i = 0; i < embd2->ne[1]; i++) {
                bpe_tokens.push_back(text_model2->model.vocab_size + num_custom_embeddings_2);
                // LOG_DEBUG("new custom token: %i", text_model.vocab_size + num_custom_embeddings);
                num_custom_embeddings_2++;
            }
            LOG_DEBUG("embedding '%s' applied, custom embeddings: %i (text model 2)", embd_name.c_str(), num_custom_embeddings_2);
        }
        int pos_end = num_custom_embeddings;
        if (pos_end == pos_start) {
            return false;
        }
        embedding_pos_map[embd_name] = std::pair{pos_start, pos_end};
        return true;
    }

    std::vector<int> convert_token_to_id(std::string text) {
        auto on_new_token_cb = [&](std::string& str, std::vector<int32_t>& bpe_tokens) -> bool {
            auto iter = embedding_map.find(str);
            if (iter == embedding_map.end()) {
                return false;
            }
            std::string embedding_path = iter->second;
            if (load_embedding(str, embedding_path, bpe_tokens)) {
                return true;
            }
            return false;
        };
        std::vector<int> curr_tokens = tokenizer.encode(text, on_new_token_cb);
        return curr_tokens;
    }

    std::string decode(const std::vector<int>& tokens) {
        return tokenizer.decode(tokens);
    }

    std::tuple<std::vector<int>, std::vector<float>, std::vector<bool>>
    tokenize_with_trigger_token(std::string text,
                                int num_input_imgs,
                                int32_t image_token) {
        auto parsed_attention = parse_prompt_attention(text);

        {
            std::stringstream ss;
            ss << "[";
            for (const auto& item : parsed_attention) {
                ss << "['" << item.first << "', " << item.second << "], ";
            }
            ss << "]";
            LOG_DEBUG("parse '%s' to %s", text.c_str(), ss.str().c_str());
        }

        auto on_new_token_cb = [&](std::string& str, std::vector<int32_t>& bpe_tokens) -> bool {
            auto iter = embedding_map.find(str);
            if (iter == embedding_map.end()) {
                return false;
            }
            std::string embedding_path = iter->second;
            if (load_embedding(str, embedding_path, bpe_tokens)) {
                return true;
            }
            return false;
        };

        std::vector<int> tokens;
        std::vector<float> weights;
        std::vector<bool> class_token_mask;
        int32_t class_idx = -1, tokens_acc = 0;
        for (const auto& item : parsed_attention) {
            std::vector<int> class_token_index;
            std::vector<int> clean_input_ids;
            const std::string& curr_text = item.first;
            float curr_weight            = item.second;
            // printf(" %s: %f \n", curr_text.c_str(), curr_weight);
            int32_t clean_index = 0;
            if (curr_text == "BREAK" && curr_weight == -1.0f) {
                // Pad token array up to chunk size at this point.
                // TODO: This is a hardcoded chunk_len, like in stable-diffusion.cpp, make it a parameter for the future?
                // Also, this is 75 instead of 77 to leave room for BOS and EOS tokens.
                int padding_size = 75 - (tokens_acc % 75);
                for (int j = 0; j < padding_size; j++) {
                    clean_input_ids.push_back(tokenizer.EOS_TOKEN_ID);
                    clean_index++;
                }

                // After padding, continue to the next iteration to process the following text as a new segment
                tokens.insert(tokens.end(), clean_input_ids.begin(), clean_input_ids.end());
                weights.insert(weights.end(), padding_size, curr_weight);
                continue;
            }

            // Regular token, process normally
            std::vector<int> curr_tokens = tokenizer.encode(curr_text, on_new_token_cb);
            for (uint32_t i = 0; i < curr_tokens.size(); i++) {
                int token_id = curr_tokens[i];
                if (token_id == image_token) {
                    class_token_index.push_back(clean_index - 1);
                } else {
                    clean_input_ids.push_back(token_id);
                    clean_index++;
                }
            }
            // GGML_ASSERT(class_token_index.size() == 1); // PhotoMaker currently does not support multiple
            //     trigger words in a single prompt.
            if (class_token_index.size() == 1) {
                // Expand the class word token and corresponding mask
                int class_token = clean_input_ids[class_token_index[0]];
                class_idx       = tokens_acc + class_token_index[0];
                std::vector<int> clean_input_ids_tmp;
                for (int i = 0; i < class_token_index[0]; i++)
                    clean_input_ids_tmp.push_back(clean_input_ids[i]);
                for (int i = 0; i < (pm_version == PM_VERSION_2 ? 2 * num_input_imgs : num_input_imgs); i++)
                    clean_input_ids_tmp.push_back(class_token);
                for (int i = class_token_index[0] + 1; i < clean_input_ids.size(); i++)
                    clean_input_ids_tmp.push_back(clean_input_ids[i]);
                clean_input_ids.clear();
                clean_input_ids = clean_input_ids_tmp;
            }
            tokens_acc += clean_index;
            tokens.insert(tokens.end(), clean_input_ids.begin(), clean_input_ids.end());
            weights.insert(weights.end(), clean_input_ids.size(), curr_weight);
        }
        // BUG!! double couting, pad_tokens will add BOS at the beginning
        // tokens.insert(tokens.begin(), tokenizer.BOS_TOKEN_ID);
        // weights.insert(weights.begin(), 1.0);

        tokenizer.pad_tokens(tokens, &weights, nullptr, text_model->model.n_token, text_model->model.n_token, true);
        int offset = pm_version == PM_VERSION_2 ? 2 * num_input_imgs : num_input_imgs;
        for (int i = 0; i < tokens.size(); i++) {
            // if (class_idx + 1 <= i && i < class_idx + 1 + 2*num_input_imgs) // photomaker V2 has num_tokens(=2)*num_input_imgs
            if (class_idx + 1 <= i && i < class_idx + 1 + offset)  // photomaker V2 has num_tokens(=2)*num_input_imgs
                                                                   // hardcode for now
                class_token_mask.push_back(true);
            else
                class_token_mask.push_back(false);
        }

        // printf("[");
        // for (int i = 0; i < tokens.size(); i++) {
        //     printf("%d, ", class_token_mask[i] ? 1 : 0);
        // }
        // printf("]\n");

        // for (int i = 0; i < tokens.size(); i++) {
        //     std::cout << tokens[i] << ":" << weights[i] << ", ";
        // }
        // std::cout << std::endl;

        return std::make_tuple(tokens, weights, class_token_mask);
    }

    std::pair<std::vector<int>, std::vector<float>> tokenize(std::string text,
                                                             size_t min_length          = 0,
                                                             size_t max_length          = 0,
                                                             bool allow_overflow_expand = true) {
        auto parsed_attention = parse_prompt_attention(text);

        {
            std::stringstream ss;
            ss << "[";
            for (const auto& item : parsed_attention) {
                ss << "['" << item.first << "', " << item.second << "], ";
            }
            ss << "]";
            LOG_DEBUG("parse '%s' to %s", text.c_str(), ss.str().c_str());
        }

        auto on_new_token_cb = [&](std::string& str, std::vector<int32_t>& bpe_tokens) -> bool {
            auto iter = embedding_map.find(str);
            if (iter == embedding_map.end()) {
                return false;
            }
            std::string embedding_path = iter->second;
            if (load_embedding(str, embedding_path, bpe_tokens)) {
                return true;
            }
            return false;
        };

        std::vector<int> tokens;
        std::vector<float> weights;
        for (const auto& item : parsed_attention) {
            const std::string& curr_text = item.first;
            float curr_weight            = item.second;

            if (curr_text == "BREAK" && curr_weight == -1.0f) {
                // Pad token array up to chunk size at this point.
                // TODO: This is a hardcoded chunk_len, like in stable-diffusion.cpp, make it a parameter for the future?
                // Also, this is 75 instead of 77 to leave room for BOS and EOS tokens.
                size_t current_size = tokens.size();
                size_t padding_size = (75 - (current_size % 75)) % 75;  // Ensure no negative padding

                if (padding_size > 0) {
                    LOG_DEBUG("BREAK token encountered, padding current chunk by %zu tokens.", padding_size);
                    tokens.insert(tokens.end(), padding_size, tokenizer.EOS_TOKEN_ID);
                    weights.insert(weights.end(), padding_size, 1.0f);
                }
                continue;  // Skip to the next item after handling BREAK
            }

            std::vector<int> curr_tokens = tokenizer.encode(curr_text, on_new_token_cb);
            tokens.insert(tokens.end(), curr_tokens.begin(), curr_tokens.end());
            weights.insert(weights.end(), curr_tokens.size(), curr_weight);
        }

        tokenizer.pad_tokens(tokens, &weights, nullptr, min_length, max_length, allow_overflow_expand);

        // for (int i = 0; i < tokens.size(); i++) {
        //     std::cout << tokens[i] << ":" << weights[i] << ", ";
        // }
        // std::cout << std::endl;

        return {tokens, weights};
    }

    SDCondition get_learned_condition_common(int n_threads,
                                             std::vector<int>& tokens,
                                             std::vector<float>& weights,
                                             int clip_skip,
                                             int width,
                                             int height,
                                             int adm_in_channels  = -1,
                                             bool zero_out_masked = false) {
        int64_t t0 = ggml_time_ms();
        sd::Tensor<float> hidden_states;  // [n_token, hidden_size] or [n_token, hidden_size + hidden_size2]
        sd::Tensor<float> pooled;

        if (clip_skip <= 0) {
            clip_skip = (sd_version_is_sd2(version) || sd_version_is_sdxl(version)) ? 2 : 1;
        }

        size_t chunk_len   = 77;
        size_t chunk_count = tokens.size() / chunk_len;
        for (int chunk_idx = 0; chunk_idx < chunk_count; chunk_idx++) {
            std::vector<int> chunk_tokens(tokens.begin() + chunk_idx * chunk_len,
                                          tokens.begin() + (chunk_idx + 1) * chunk_len);
            std::vector<float> chunk_weights(weights.begin() + chunk_idx * chunk_len,
                                             weights.begin() + (chunk_idx + 1) * chunk_len);

            sd::Tensor<int32_t> input_ids({static_cast<int64_t>(chunk_tokens.size())}, chunk_tokens);
            sd::Tensor<int32_t> input_ids2;
            size_t max_token_idx = 0;
            if (sd_version_is_sdxl(version)) {
                auto it = std::find(chunk_tokens.begin(), chunk_tokens.end(), tokenizer.EOS_TOKEN_ID);
                if (it != chunk_tokens.end()) {
                    std::fill(std::next(it), chunk_tokens.end(), 0);
                }

                max_token_idx = std::min<size_t>(std::distance(chunk_tokens.begin(), it), chunk_tokens.size() - 1);

                input_ids2 = sd::Tensor<int32_t>({static_cast<int64_t>(chunk_tokens.size())}, chunk_tokens);

                // for (int i = 0; i < chunk_tokens.size(); i++) {
                //     printf("%d ", chunk_tokens[i]);
                // }
                // printf("\n");
            }

            {
                auto chunk_hidden_states = text_model->compute(n_threads,
                                                               input_ids,
                                                               num_custom_embeddings,
                                                               token_embed_custom.data(),
                                                               max_token_idx,
                                                               false,
                                                               clip_skip);
                GGML_ASSERT(!chunk_hidden_states.empty());
                if (sd_version_is_sdxl(version)) {
                    auto chunk_hidden_states2 = text_model2->compute(n_threads,
                                                                     input_ids2,
                                                                     num_custom_embeddings,
                                                                     token_embed_custom.data(),
                                                                     max_token_idx,
                                                                     false,
                                                                     clip_skip);
                    GGML_ASSERT(!chunk_hidden_states2.empty());
                    chunk_hidden_states = sd::ops::concat(chunk_hidden_states, chunk_hidden_states2, 0);

                    if (chunk_idx == 0) {
                        pooled = text_model2->compute(n_threads,
                                                      input_ids2,
                                                      num_custom_embeddings,
                                                      token_embed_custom.data(),
                                                      max_token_idx,
                                                      true,
                                                      clip_skip);
                        GGML_ASSERT(!pooled.empty());
                    }
                }
                int64_t t1 = ggml_time_ms();
                LOG_DEBUG("computing condition graph completed, taking %" PRId64 " ms", t1 - t0);

                chunk_hidden_states = apply_token_weights(std::move(chunk_hidden_states), chunk_weights);

                if (zero_out_masked) {
                    chunk_hidden_states.fill_(0.0f);
                }
                if (!hidden_states.empty()) {
                    hidden_states = sd::ops::concat(hidden_states, chunk_hidden_states, 1);
                } else {
                    hidden_states = std::move(chunk_hidden_states);
                }
            }
        }

        sd::Tensor<float> vec;
        if (sd_version_is_sdxl(version)) {
            int out_dim = 256;
            GGML_ASSERT(!pooled.empty());
            vec = sd::Tensor<float>({adm_in_channels});
            vec.fill_(0.0f);
            size_t offset = 0;
            std::copy(pooled.values().begin(), pooled.values().end(), vec.values().begin());
            offset += pooled.values().size();

            auto append_embedding = [&](const std::vector<float>& timesteps) {
                sd::Tensor<float> embedding;
                set_timestep_embedding(timesteps, &embedding, out_dim);
                std::copy(embedding.values().begin(), embedding.values().end(), vec.values().begin() + static_cast<int64_t>(offset));
                offset += embedding.values().size();
            };

            append_embedding({static_cast<float>(height), static_cast<float>(width)});
            append_embedding({0.0f, 0.0f});
            append_embedding({static_cast<float>(height), static_cast<float>(width)});
            GGML_ASSERT(offset == vec.values().size());
        }
        SDCondition result;
        if (!hidden_states.empty()) {
            result.c_crossattn = std::move(hidden_states);
        }

        if (!vec.empty()) {
            result.c_vector = std::move(vec);
        }
        return result;
    }

    std::tuple<SDCondition, std::vector<bool>>
    get_learned_condition_with_trigger(int n_threads,
                                       const ConditionerParams& conditioner_params) override {
        auto image_tokens = convert_token_to_id(trigger_word);
        // if(image_tokens.size() == 1){
        //     printf(" image token id is: %d \n", image_tokens[0]);
        // }
        GGML_ASSERT(image_tokens.size() == 1);
        auto tokens_and_weights     = tokenize_with_trigger_token(conditioner_params.text,
                                                                  conditioner_params.num_input_imgs,
                                                                  image_tokens[0]);
        std::vector<int>& tokens    = std::get<0>(tokens_and_weights);
        std::vector<float>& weights = std::get<1>(tokens_and_weights);
        std::vector<bool>& clsm     = std::get<2>(tokens_and_weights);
        // printf("tokens: \n");
        // for(int i = 0; i < tokens.size(); ++i)
        //    printf("%d ", tokens[i]);
        // printf("\n");
        // printf("clsm: \n");
        // for(int i = 0; i < clsm.size(); ++i)
        //    printf("%d ", clsm[i]?1:0);
        // printf("\n");
        auto cond = get_learned_condition_common(n_threads,
                                                 tokens,
                                                 weights,
                                                 conditioner_params.clip_skip,
                                                 conditioner_params.width,
                                                 conditioner_params.height,
                                                 conditioner_params.adm_in_channels,
                                                 conditioner_params.zero_out_masked);
        return std::make_tuple(cond, clsm);
    }

    std::string remove_trigger_from_prompt(const std::string& prompt) override {
        auto image_tokens = convert_token_to_id(trigger_word);
        GGML_ASSERT(image_tokens.size() == 1);
        auto tokens_and_weights  = tokenize(prompt);
        std::vector<int>& tokens = tokens_and_weights.first;
        auto it                  = std::find(tokens.begin(), tokens.end(), image_tokens[0]);
        GGML_ASSERT(it != tokens.end());  // prompt must have trigger word
        tokens.erase(it);
        return decode(tokens);
    }

    SDCondition get_learned_condition(int n_threads,
                                      const ConditionerParams& conditioner_params) override {
        auto tokens_and_weights     = tokenize(conditioner_params.text, text_model->model.n_token, text_model->model.n_token, true);
        std::vector<int>& tokens    = tokens_and_weights.first;
        std::vector<float>& weights = tokens_and_weights.second;
        return get_learned_condition_common(n_threads,
                                            tokens,
                                            weights,
                                            conditioner_params.clip_skip,
                                            conditioner_params.width,
                                            conditioner_params.height,
                                            conditioner_params.adm_in_channels,
                                            conditioner_params.zero_out_masked);
    }
};

struct FrozenCLIPVisionEmbedder : public GGMLRunner {
    CLIPVisionModelProjection vision_model;

    FrozenCLIPVisionEmbedder(ggml_backend_t backend,
                             bool offload_params_to_cpu,
                             const String2TensorStorage& tensor_storage_map = {})
        : GGMLRunner(backend, offload_params_to_cpu) {
        std::string prefix = "cond_stage_model.transformer";
        bool proj_in       = false;
        for (const auto& [name, tensor_storage] : tensor_storage_map) {
            if (!starts_with(name, prefix)) {
                continue;
            }
            if (contains(name, "self_attn.in_proj")) {
                proj_in = true;
                break;
            }
        }
        vision_model = CLIPVisionModelProjection(OPEN_CLIP_VIT_H_14, false, proj_in);
        vision_model.init(params_ctx, tensor_storage_map, prefix);
    }

    std::string get_desc() override {
        return "clip_vision";
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) {
        vision_model.get_param_tensors(tensors, "cond_stage_model.transformer");
    }

    ggml_cgraph* build_graph(const sd::Tensor<float>& pixel_values_tensor, bool return_pooled, int clip_skip) {
        ggml_cgraph* gf           = ggml_new_graph(compute_ctx);
        ggml_tensor* pixel_values = make_input(pixel_values_tensor);

        auto runner_ctx = get_context();

        ggml_tensor* hidden_states = vision_model.forward(&runner_ctx, pixel_values, return_pooled, clip_skip);

        ggml_build_forward_expand(gf, hidden_states);

        return gf;
    }

    sd::Tensor<float> compute(const int n_threads,
                              const sd::Tensor<float>& pixel_values,
                              bool return_pooled,
                              int clip_skip) {
        auto get_graph = [&]() -> ggml_cgraph* {
            return build_graph(pixel_values, return_pooled, clip_skip);
        };
        return take_or_empty(GGMLRunner::compute<float>(get_graph, n_threads, true));
    }
};

struct SD3CLIPEmbedder : public Conditioner {
    CLIPTokenizer clip_l_tokenizer;
    CLIPTokenizer clip_g_tokenizer;
    T5UniGramTokenizer t5_tokenizer;
    std::shared_ptr<CLIPTextModelRunner> clip_l;
    std::shared_ptr<CLIPTextModelRunner> clip_g;
    std::shared_ptr<T5Runner> t5;

    SD3CLIPEmbedder(ggml_backend_t backend,
                    bool offload_params_to_cpu,
                    const String2TensorStorage& tensor_storage_map = {})
        : clip_g_tokenizer(0) {
        bool use_clip_l = false;
        bool use_clip_g = false;
        bool use_t5     = false;
        for (auto pair : tensor_storage_map) {
            if (pair.first.find("text_encoders.clip_l") != std::string::npos) {
                use_clip_l = true;
            } else if (pair.first.find("text_encoders.clip_g") != std::string::npos) {
                use_clip_g = true;
            } else if (pair.first.find("text_encoders.t5xxl") != std::string::npos) {
                use_t5 = true;
            }
        }
        if (!use_clip_l && !use_clip_g && !use_t5) {
            LOG_WARN("IMPORTANT NOTICE: No text encoders provided, cannot process prompts!");
            return;
        }
        if (use_clip_l) {
            clip_l = std::make_shared<CLIPTextModelRunner>(backend, offload_params_to_cpu, tensor_storage_map, "text_encoders.clip_l.transformer.text_model", OPENAI_CLIP_VIT_L_14, false);
        }
        if (use_clip_g) {
            clip_g = std::make_shared<CLIPTextModelRunner>(backend, offload_params_to_cpu, tensor_storage_map, "text_encoders.clip_g.transformer.text_model", OPEN_CLIP_VIT_BIGG_14, false);
        }
        if (use_t5) {
            t5 = std::make_shared<T5Runner>(backend, offload_params_to_cpu, tensor_storage_map, "text_encoders.t5xxl.transformer");
        }
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
        if (clip_l) {
            clip_l->get_param_tensors(tensors, "text_encoders.clip_l.transformer.text_model");
        }
        if (clip_g) {
            clip_g->get_param_tensors(tensors, "text_encoders.clip_g.transformer.text_model");
        }
        if (t5) {
            t5->get_param_tensors(tensors, "text_encoders.t5xxl.transformer");
        }
    }

    void alloc_params_buffer() override {
        if (clip_l) {
            clip_l->alloc_params_buffer();
        }
        if (clip_g) {
            clip_g->alloc_params_buffer();
        }
        if (t5) {
            t5->alloc_params_buffer();
        }
    }

    void free_params_buffer() override {
        if (clip_l) {
            clip_l->free_params_buffer();
        }
        if (clip_g) {
            clip_g->free_params_buffer();
        }
        if (t5) {
            t5->free_params_buffer();
        }
    }

    size_t get_params_buffer_size() override {
        size_t buffer_size = 0;
        if (clip_l) {
            buffer_size += clip_l->get_params_buffer_size();
        }
        if (clip_g) {
            buffer_size += clip_g->get_params_buffer_size();
        }
        if (t5) {
            buffer_size += t5->get_params_buffer_size();
        }
        return buffer_size;
    }

    void set_flash_attention_enabled(bool enabled) override {
        if (clip_l) {
            clip_l->set_flash_attention_enabled(enabled);
        }
        if (clip_g) {
            clip_g->set_flash_attention_enabled(enabled);
        }
        if (t5) {
            t5->set_flash_attention_enabled(enabled);
        }
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override {
        if (clip_l) {
            clip_l->set_weight_adapter(adapter);
        }
        if (clip_g) {
            clip_g->set_weight_adapter(adapter);
        }
        if (t5) {
            t5->set_weight_adapter(adapter);
        }
    }

    std::vector<std::pair<std::vector<int>, std::vector<float>>> tokenize(std::string text,
                                                                          size_t min_length          = 0,
                                                                          size_t max_length          = 0,
                                                                          bool allow_overflow_expand = true) {
        auto parsed_attention = parse_prompt_attention(text);

        {
            std::stringstream ss;
            ss << "[";
            for (const auto& item : parsed_attention) {
                ss << "['" << item.first << "', " << item.second << "], ";
            }
            ss << "]";
            LOG_DEBUG("parse '%s' to %s", text.c_str(), ss.str().c_str());
        }

        auto on_new_token_cb = [&](std::string& str, std::vector<int32_t>& bpe_tokens) -> bool {
            return false;
        };

        std::vector<int> clip_l_tokens;
        std::vector<float> clip_l_weights;
        std::vector<int> clip_g_tokens;
        std::vector<float> clip_g_weights;
        std::vector<int> t5_tokens;
        std::vector<float> t5_weights;
        for (const auto& item : parsed_attention) {
            const std::string& curr_text = item.first;
            float curr_weight            = item.second;
            if (clip_l) {
                std::vector<int> curr_tokens = clip_l_tokenizer.encode(curr_text, on_new_token_cb);
                clip_l_tokens.insert(clip_l_tokens.end(), curr_tokens.begin(), curr_tokens.end());
                clip_l_weights.insert(clip_l_weights.end(), curr_tokens.size(), curr_weight);
            }
            if (clip_g) {
                std::vector<int> curr_tokens = clip_g_tokenizer.encode(curr_text, on_new_token_cb);
                clip_g_tokens.insert(clip_g_tokens.end(), curr_tokens.begin(), curr_tokens.end());
                clip_g_weights.insert(clip_g_weights.end(), curr_tokens.size(), curr_weight);
            }
            if (t5) {
                std::vector<int> curr_tokens = t5_tokenizer.encode(curr_text);
                t5_tokens.insert(t5_tokens.end(), curr_tokens.begin(), curr_tokens.end());
                t5_weights.insert(t5_weights.end(), curr_tokens.size(), curr_weight);
            }
        }

        if (clip_l) {
            clip_l_tokenizer.pad_tokens(clip_l_tokens, &clip_l_weights, nullptr, min_length, max_length, allow_overflow_expand);
        }
        if (clip_g) {
            clip_g_tokenizer.pad_tokens(clip_g_tokens, &clip_g_weights, nullptr, min_length, max_length, allow_overflow_expand);
        }
        if (t5) {
            t5_tokenizer.pad_tokens(t5_tokens, &t5_weights, nullptr, min_length, max_length, true);
        }

        // for (int i = 0; i < clip_l_tokens.size(); i++) {
        //     std::cout << clip_l_tokens[i] << ":" << clip_l_weights[i] << ", ";
        // }
        // std::cout << std::endl;

        // for (int i = 0; i < clip_g_tokens.size(); i++) {
        //     std::cout << clip_g_tokens[i] << ":" << clip_g_weights[i] << ", ";
        // }
        // std::cout << std::endl;

        // for (int i = 0; i < t5_tokens.size(); i++) {
        //     std::cout << t5_tokens[i] << ":" << t5_weights[i] << ", ";
        // }
        // std::cout << std::endl;

        return {{clip_l_tokens, clip_l_weights}, {clip_g_tokens, clip_g_weights}, {t5_tokens, t5_weights}};
    }

    SDCondition get_learned_condition_common(int n_threads,
                                             std::vector<std::pair<std::vector<int>, std::vector<float>>> token_and_weights,
                                             int clip_skip,
                                             bool zero_out_masked = false) {
        auto& clip_l_tokens  = token_and_weights[0].first;
        auto& clip_l_weights = token_and_weights[0].second;
        auto& clip_g_tokens  = token_and_weights[1].first;
        auto& clip_g_weights = token_and_weights[1].second;
        auto& t5_tokens      = token_and_weights[2].first;
        auto& t5_weights     = token_and_weights[2].second;

        if (clip_skip <= 0) {
            clip_skip = 2;
        }

        size_t chunk_len = 77;
        int64_t t0       = ggml_time_ms();
        sd::Tensor<float> hidden_states;
        sd::Tensor<float> pooled;

        size_t chunk_count = std::max(std::max(clip_l_tokens.size(), clip_g_tokens.size()), t5_tokens.size()) / chunk_len;

        for (int chunk_idx = 0; chunk_idx < chunk_count; chunk_idx++) {
            // clip_l
            sd::Tensor<float> chunk_hidden_states_l;
            sd::Tensor<float> pooled_l;
            if (clip_l) {
                std::vector<int> chunk_tokens(clip_l_tokens.begin() + chunk_idx * chunk_len,
                                              clip_l_tokens.begin() + (chunk_idx + 1) * chunk_len);
                std::vector<float> chunk_weights(clip_l_weights.begin() + chunk_idx * chunk_len,
                                                 clip_l_weights.begin() + (chunk_idx + 1) * chunk_len);

                sd::Tensor<int32_t> input_ids({static_cast<int64_t>(chunk_tokens.size())}, chunk_tokens);
                size_t max_token_idx = 0;

                chunk_hidden_states_l = clip_l->compute(n_threads,
                                                        input_ids,
                                                        0,
                                                        nullptr,
                                                        max_token_idx,
                                                        false,
                                                        clip_skip);
                GGML_ASSERT(!chunk_hidden_states_l.empty());
                chunk_hidden_states_l = ::apply_token_weights(std::move(chunk_hidden_states_l), chunk_weights);

                if (chunk_idx == 0) {
                    auto it       = std::find(chunk_tokens.begin(), chunk_tokens.end(), clip_l_tokenizer.EOS_TOKEN_ID);
                    max_token_idx = std::min<size_t>(std::distance(chunk_tokens.begin(), it), chunk_tokens.size() - 1);
                    pooled_l      = clip_l->compute(n_threads,
                                                    input_ids,
                                                    0,
                                                    nullptr,
                                                    max_token_idx,
                                                    true,
                                                    clip_skip);
                    GGML_ASSERT(!pooled_l.empty());
                }
            } else {
                chunk_hidden_states_l = sd::Tensor<float>::zeros({768, static_cast<int64_t>(chunk_len), 1});
                if (chunk_idx == 0) {
                    pooled = sd::Tensor<float>::zeros({768, 1});
                }
            }

            // clip_g
            sd::Tensor<float> chunk_hidden_states_g;
            sd::Tensor<float> pooled_g;
            if (clip_g) {
                std::vector<int> chunk_tokens(clip_g_tokens.begin() + chunk_idx * chunk_len,
                                              clip_g_tokens.begin() + (chunk_idx + 1) * chunk_len);
                std::vector<float> chunk_weights(clip_g_weights.begin() + chunk_idx * chunk_len,
                                                 clip_g_weights.begin() + (chunk_idx + 1) * chunk_len);

                sd::Tensor<int32_t> input_ids({static_cast<int64_t>(chunk_tokens.size())}, chunk_tokens);
                size_t max_token_idx = 0;

                chunk_hidden_states_g = clip_g->compute(n_threads,
                                                        input_ids,
                                                        0,
                                                        nullptr,
                                                        max_token_idx,
                                                        false,
                                                        clip_skip);
                GGML_ASSERT(!chunk_hidden_states_g.empty());
                chunk_hidden_states_g = ::apply_token_weights(std::move(chunk_hidden_states_g), chunk_weights);

                if (chunk_idx == 0) {
                    auto it       = std::find(chunk_tokens.begin(), chunk_tokens.end(), clip_g_tokenizer.EOS_TOKEN_ID);
                    max_token_idx = std::min<size_t>(std::distance(chunk_tokens.begin(), it), chunk_tokens.size() - 1);
                    pooled_g      = clip_g->compute(n_threads,
                                                    input_ids,
                                                    0,
                                                    nullptr,
                                                    max_token_idx,
                                                    true,
                                                    clip_skip);
                    GGML_ASSERT(!pooled_g.empty());
                }
            } else {
                chunk_hidden_states_g = sd::Tensor<float>::zeros({1280, static_cast<int64_t>(chunk_len), 1});
                if (chunk_idx == 0) {
                    pooled_g = sd::Tensor<float>::zeros({1280, 1});
                }
            }

            // t5
            sd::Tensor<float> chunk_hidden_states_t5;
            if (t5) {
                std::vector<int> chunk_tokens(t5_tokens.begin() + chunk_idx * chunk_len,
                                              t5_tokens.begin() + (chunk_idx + 1) * chunk_len);
                std::vector<float> chunk_weights(t5_weights.begin() + chunk_idx * chunk_len,
                                                 t5_weights.begin() + (chunk_idx + 1) * chunk_len);

                sd::Tensor<int32_t> input_ids({static_cast<int64_t>(chunk_tokens.size())}, chunk_tokens);

                chunk_hidden_states_t5 = t5->compute(n_threads,
                                                     input_ids,
                                                     sd::Tensor<float>());
                GGML_ASSERT(!chunk_hidden_states_t5.empty());
                chunk_hidden_states_t5 = ::apply_token_weights(std::move(chunk_hidden_states_t5), chunk_weights);
            } else {
                chunk_hidden_states_t5 = sd::Tensor<float>::zeros({4096, static_cast<int64_t>(chunk_len), 1});
            }

            sd::Tensor<float> chunk_hidden_states_lg = sd::ops::concat(chunk_hidden_states_l, chunk_hidden_states_g, 0);
            if (chunk_hidden_states_lg.shape()[0] < 4096) {
                auto pad_shape         = chunk_hidden_states_lg.shape();
                pad_shape[0]           = 4096 - chunk_hidden_states_lg.shape()[0];
                chunk_hidden_states_lg = sd::ops::concat(chunk_hidden_states_lg,
                                                         sd::Tensor<float>::zeros(pad_shape),
                                                         0);
            }

            sd::Tensor<float> chunk_hidden_states = sd::ops::concat(chunk_hidden_states_lg,
                                                                    chunk_hidden_states_t5,
                                                                    1);  // [n_token*2, 4096]

            if (chunk_idx == 0) {
                pooled = sd::ops::concat(pooled_l, pooled_g, 0);  // [768 + 1280]
            }

            int64_t t1 = ggml_time_ms();
            LOG_DEBUG("computing condition graph completed, taking %" PRId64 " ms", t1 - t0);
            if (zero_out_masked) {
                chunk_hidden_states.fill_(0.0f);
            }

            if (!hidden_states.empty()) {
                hidden_states = sd::ops::concat(hidden_states, chunk_hidden_states, 1);
            } else {
                hidden_states = std::move(chunk_hidden_states);
            }
        }

        SDCondition result;
        result.c_crossattn = std::move(hidden_states);
        result.c_vector    = std::move(pooled);
        return result;
    }

    SDCondition get_learned_condition(int n_threads,
                                      const ConditionerParams& conditioner_params) override {
        auto tokens_and_weights = tokenize(conditioner_params.text, 77, 77, true);
        return get_learned_condition_common(n_threads,
                                            tokens_and_weights,
                                            conditioner_params.clip_skip,
                                            conditioner_params.zero_out_masked);
    }
};

struct FluxCLIPEmbedder : public Conditioner {
    CLIPTokenizer clip_l_tokenizer;
    T5UniGramTokenizer t5_tokenizer;
    std::shared_ptr<CLIPTextModelRunner> clip_l;
    std::shared_ptr<T5Runner> t5;
    size_t chunk_len = 256;

    FluxCLIPEmbedder(ggml_backend_t backend,
                     bool offload_params_to_cpu,
                     const String2TensorStorage& tensor_storage_map = {}) {
        bool use_clip_l = false;
        bool use_t5     = false;
        for (auto pair : tensor_storage_map) {
            if (pair.first.find("text_encoders.clip_l") != std::string::npos) {
                use_clip_l = true;
            } else if (pair.first.find("text_encoders.t5xxl") != std::string::npos) {
                use_t5 = true;
            }
        }

        if (!use_clip_l && !use_t5) {
            LOG_WARN("IMPORTANT NOTICE: No text encoders provided, cannot process prompts!");
            return;
        }

        if (use_clip_l) {
            clip_l = std::make_shared<CLIPTextModelRunner>(backend, offload_params_to_cpu, tensor_storage_map, "text_encoders.clip_l.transformer.text_model", OPENAI_CLIP_VIT_L_14, true);
        } else {
            LOG_WARN("clip_l text encoder not found! Prompt adherence might be degraded.");
        }
        if (use_t5) {
            t5 = std::make_shared<T5Runner>(backend, offload_params_to_cpu, tensor_storage_map, "text_encoders.t5xxl.transformer");
        } else {
            LOG_WARN("t5xxl text encoder not found! Prompt adherence might be degraded.");
        }
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
        if (clip_l) {
            clip_l->get_param_tensors(tensors, "text_encoders.clip_l.transformer.text_model");
        }
        if (t5) {
            t5->get_param_tensors(tensors, "text_encoders.t5xxl.transformer");
        }
    }

    void alloc_params_buffer() override {
        if (clip_l) {
            clip_l->alloc_params_buffer();
        }
        if (t5) {
            t5->alloc_params_buffer();
        }
    }

    void free_params_buffer() override {
        if (clip_l) {
            clip_l->free_params_buffer();
        }
        if (t5) {
            t5->free_params_buffer();
        }
    }

    size_t get_params_buffer_size() override {
        size_t buffer_size = 0;
        if (clip_l) {
            buffer_size += clip_l->get_params_buffer_size();
        }
        if (t5) {
            buffer_size += t5->get_params_buffer_size();
        }
        return buffer_size;
    }

    void set_flash_attention_enabled(bool enabled) override {
        if (clip_l) {
            clip_l->set_flash_attention_enabled(enabled);
        }
        if (t5) {
            t5->set_flash_attention_enabled(enabled);
        }
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) {
        if (clip_l) {
            clip_l->set_weight_adapter(adapter);
        }
        if (t5) {
            t5->set_weight_adapter(adapter);
        }
    }

    std::vector<std::pair<std::vector<int>, std::vector<float>>> tokenize(std::string text,
                                                                          size_t min_length = 0,
                                                                          size_t max_length = 0) {
        auto parsed_attention = parse_prompt_attention(text);

        {
            std::stringstream ss;
            ss << "[";
            for (const auto& item : parsed_attention) {
                ss << "['" << item.first << "', " << item.second << "], ";
            }
            ss << "]";
            LOG_DEBUG("parse '%s' to %s", text.c_str(), ss.str().c_str());
        }

        auto on_new_token_cb = [&](std::string& str, std::vector<int32_t>& bpe_tokens) -> bool {
            return false;
        };

        std::vector<int> clip_l_tokens;
        std::vector<float> clip_l_weights;
        std::vector<int> t5_tokens;
        std::vector<float> t5_weights;
        for (const auto& item : parsed_attention) {
            const std::string& curr_text = item.first;
            float curr_weight            = item.second;
            if (clip_l) {
                std::vector<int> curr_tokens = clip_l_tokenizer.encode(curr_text, on_new_token_cb);
                clip_l_tokens.insert(clip_l_tokens.end(), curr_tokens.begin(), curr_tokens.end());
                clip_l_weights.insert(clip_l_weights.end(), curr_tokens.size(), curr_weight);
            }
            if (t5) {
                std::vector<int> curr_tokens = t5_tokenizer.encode(curr_text);
                t5_tokens.insert(t5_tokens.end(), curr_tokens.begin(), curr_tokens.end());
                t5_weights.insert(t5_weights.end(), curr_tokens.size(), curr_weight);
            }
        }

        if (clip_l) {
            clip_l_tokenizer.pad_tokens(clip_l_tokens, &clip_l_weights, nullptr, 77, 77, true);
        }
        if (t5) {
            t5_tokenizer.pad_tokens(t5_tokens, &t5_weights, nullptr, min_length, max_length, true);
        }

        // for (int i = 0; i < clip_l_tokens.size(); i++) {
        //     std::cout << clip_l_tokens[i] << ":" << clip_l_weights[i] << ", ";
        // }
        // std::cout << std::endl;

        // for (int i = 0; i < t5_tokens.size(); i++) {
        //     std::cout << t5_tokens[i] << ":" << t5_weights[i] << ", ";
        // }
        // std::cout << std::endl;

        return {{clip_l_tokens, clip_l_weights}, {t5_tokens, t5_weights}};
    }

    SDCondition get_learned_condition_common(int n_threads,
                                             std::vector<std::pair<std::vector<int>, std::vector<float>>> token_and_weights,
                                             int clip_skip,
                                             bool zero_out_masked = false) {
        auto& clip_l_tokens  = token_and_weights[0].first;
        auto& clip_l_weights = token_and_weights[0].second;
        auto& t5_tokens      = token_and_weights[1].first;
        auto& t5_weights     = token_and_weights[1].second;

        if (clip_skip <= 0) {
            clip_skip = 2;
        }

        int64_t t0 = ggml_time_ms();
        sd::Tensor<float> hidden_states;  // [N, n_token, 4096]
        sd::Tensor<float> pooled;         // [768,]

        size_t chunk_count = std::max(clip_l_tokens.size() > 0 ? chunk_len : 0, t5_tokens.size()) / chunk_len;
        for (int chunk_idx = 0; chunk_idx < chunk_count; chunk_idx++) {
            // clip_l
            if (chunk_idx == 0) {
                if (clip_l) {
                    size_t chunk_len_l = 77;
                    std::vector<int> chunk_tokens(clip_l_tokens.begin(),
                                                  clip_l_tokens.begin() + chunk_len_l);
                    std::vector<float> chunk_weights(clip_l_weights.begin(),
                                                     clip_l_weights.begin() + chunk_len_l);

                    sd::Tensor<int32_t> input_ids({static_cast<int64_t>(chunk_tokens.size())}, chunk_tokens);
                    size_t max_token_idx = 0;

                    auto it       = std::find(chunk_tokens.begin(), chunk_tokens.end(), clip_l_tokenizer.EOS_TOKEN_ID);
                    max_token_idx = std::min<size_t>(std::distance(chunk_tokens.begin(), it), chunk_tokens.size() - 1);

                    pooled = clip_l->compute(n_threads,
                                             input_ids,
                                             0,
                                             nullptr,
                                             max_token_idx,
                                             true,
                                             clip_skip);
                    GGML_ASSERT(!pooled.empty());
                } else {
                    pooled = sd::Tensor<float>::zeros({768});
                }
            }

            // t5
            sd::Tensor<float> chunk_hidden_states;
            if (t5) {
                std::vector<int> chunk_tokens(t5_tokens.begin() + chunk_idx * chunk_len,
                                              t5_tokens.begin() + (chunk_idx + 1) * chunk_len);
                std::vector<float> chunk_weights(t5_weights.begin() + chunk_idx * chunk_len,
                                                 t5_weights.begin() + (chunk_idx + 1) * chunk_len);

                sd::Tensor<int32_t> input_ids({static_cast<int64_t>(chunk_tokens.size())}, chunk_tokens);
                chunk_hidden_states = t5->compute(n_threads,
                                                  input_ids,
                                                  sd::Tensor<float>());
                GGML_ASSERT(!chunk_hidden_states.empty());
                chunk_hidden_states = ::apply_token_weights(std::move(chunk_hidden_states), chunk_weights);
                if (zero_out_masked) {
                    chunk_hidden_states.fill_(0.0f);
                }
            } else {
                chunk_hidden_states = sd::Tensor<float>::zeros({4096, static_cast<int64_t>(chunk_len)});
            }

            int64_t t1 = ggml_time_ms();
            LOG_DEBUG("computing condition graph completed, taking %" PRId64 " ms", t1 - t0);
            if (!hidden_states.empty()) {
                hidden_states = sd::ops::concat(hidden_states, chunk_hidden_states, 1);
            } else {
                hidden_states = std::move(chunk_hidden_states);
            }
        }

        SDCondition result;
        result.c_crossattn = std::move(hidden_states);
        result.c_vector    = std::move(pooled);
        return result;
    }

    SDCondition get_learned_condition(int n_threads,
                                      const ConditionerParams& conditioner_params) override {
        auto tokens_and_weights = tokenize(conditioner_params.text, chunk_len, chunk_len);
        return get_learned_condition_common(n_threads,
                                            tokens_and_weights,
                                            conditioner_params.clip_skip,
                                            conditioner_params.zero_out_masked);
    }
};

struct T5CLIPEmbedder : public Conditioner {
    T5UniGramTokenizer t5_tokenizer;
    std::shared_ptr<T5Runner> t5;
    size_t chunk_len = 512;
    bool use_mask    = false;
    int mask_pad     = 0;
    bool is_umt5     = false;

    T5CLIPEmbedder(ggml_backend_t backend,
                   bool offload_params_to_cpu,
                   const String2TensorStorage& tensor_storage_map = {},
                   bool use_mask                                  = false,
                   int mask_pad                                   = 0,
                   bool is_umt5                                   = false)
        : use_mask(use_mask), mask_pad(mask_pad), t5_tokenizer(is_umt5) {
        bool use_t5 = false;
        for (auto pair : tensor_storage_map) {
            if (pair.first.find("text_encoders.t5xxl") != std::string::npos) {
                use_t5 = true;
            }
        }

        if (!use_t5) {
            LOG_WARN("IMPORTANT NOTICE: No text encoders provided, cannot process prompts!");
            return;
        } else {
            t5 = std::make_shared<T5Runner>(backend, offload_params_to_cpu, tensor_storage_map, "text_encoders.t5xxl.transformer", is_umt5);
        }
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
        if (t5) {
            t5->get_param_tensors(tensors, "text_encoders.t5xxl.transformer");
        }
    }

    void alloc_params_buffer() override {
        if (t5) {
            t5->alloc_params_buffer();
        }
    }

    void free_params_buffer() override {
        if (t5) {
            t5->free_params_buffer();
        }
    }

    size_t get_params_buffer_size() override {
        size_t buffer_size = 0;
        if (t5) {
            buffer_size += t5->get_params_buffer_size();
        }
        return buffer_size;
    }

    void set_flash_attention_enabled(bool enabled) override {
        if (t5) {
            t5->set_flash_attention_enabled(enabled);
        }
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override {
        if (t5) {
            t5->set_weight_adapter(adapter);
        }
    }

    std::tuple<std::vector<int>, std::vector<float>, std::vector<float>> tokenize(std::string text,
                                                                                  size_t min_length = 0,
                                                                                  size_t max_length = 0) {
        auto parsed_attention = parse_prompt_attention(text);

        {
            std::stringstream ss;
            ss << "[";
            for (const auto& item : parsed_attention) {
                ss << "['" << item.first << "', " << item.second << "], ";
            }
            ss << "]";
            LOG_DEBUG("parse '%s' to %s", text.c_str(), ss.str().c_str());
        }

        auto on_new_token_cb = [&](std::string& str, std::vector<int32_t>& bpe_tokens) -> bool {
            return false;
        };

        std::vector<int> t5_tokens;
        std::vector<float> t5_weights;
        std::vector<float> t5_mask;
        if (t5) {
            for (const auto& item : parsed_attention) {
                const std::string& curr_text = item.first;
                float curr_weight            = item.second;

                std::vector<int> curr_tokens = t5_tokenizer.encode(curr_text);
                t5_tokens.insert(t5_tokens.end(), curr_tokens.begin(), curr_tokens.end());
                t5_weights.insert(t5_weights.end(), curr_tokens.size(), curr_weight);
            }

            t5_tokenizer.pad_tokens(t5_tokens, &t5_weights, &t5_mask, min_length, max_length, true);
            for (auto& mask_value : t5_mask) {
                mask_value = mask_value > 0.0f ? 0.0f : -HUGE_VALF;
            }
        }
        return {t5_tokens, t5_weights, t5_mask};
    }

    void modify_mask_to_attend_padding(sd::Tensor<float>* mask, int max_seq_length, int num_extra_padding = 8) {
        GGML_ASSERT(mask != nullptr);
        float* mask_data = mask->data();
        int num_pad      = 0;
        for (int64_t i = 0; i < max_seq_length; i++) {
            if (num_pad >= num_extra_padding) {
                break;
            }
            if (std::isinf(mask_data[i])) {
                mask_data[i] = 0;
                ++num_pad;
            }
        }
        // LOG_DEBUG("PAD: %d", num_pad);
    }

    SDCondition get_learned_condition_common(int n_threads,
                                             std::tuple<std::vector<int>, std::vector<float>, std::vector<float>> token_and_weights,
                                             int clip_skip,
                                             bool zero_out_masked = false) {
        if (!t5) {
            SDCondition result;
            result.c_crossattn = sd::Tensor<float>::zeros({4096, 256});
            result.c_vector    = sd::Tensor<float>::full({256}, -HUGE_VALF);
            return result;
        }
        auto& t5_tokens        = std::get<0>(token_and_weights);
        auto& t5_weights       = std::get<1>(token_and_weights);
        auto& t5_attn_mask_vec = std::get<2>(token_and_weights);

        int64_t t0                     = ggml_time_ms();
        sd::Tensor<float> t5_attn_mask = sd::Tensor<float>::from_vector(t5_attn_mask_vec);
        sd::Tensor<float> hidden_states;

        size_t chunk_count = t5_tokens.size() / chunk_len;

        for (int chunk_idx = 0; chunk_idx < chunk_count; chunk_idx++) {
            // t5
            std::vector<int> chunk_tokens(t5_tokens.begin() + chunk_idx * chunk_len,
                                          t5_tokens.begin() + (chunk_idx + 1) * chunk_len);
            std::vector<float> chunk_weights(t5_weights.begin() + chunk_idx * chunk_len,
                                             t5_weights.begin() + (chunk_idx + 1) * chunk_len);
            std::vector<float> chunk_mask(t5_attn_mask_vec.begin() + chunk_idx * chunk_len,
                                          t5_attn_mask_vec.begin() + (chunk_idx + 1) * chunk_len);

            sd::Tensor<int32_t> input_ids({static_cast<int64_t>(chunk_tokens.size())}, chunk_tokens);
            sd::Tensor<float> t5_attn_mask_chunk;
            if (use_mask) {
                t5_attn_mask_chunk = sd::Tensor<float>({static_cast<int64_t>(chunk_mask.size())}, chunk_mask);
            }

            auto chunk_hidden_states = t5->compute(n_threads,
                                                   input_ids,
                                                   t5_attn_mask_chunk);
            GGML_ASSERT(!chunk_hidden_states.empty());
            chunk_hidden_states = apply_token_weights(std::move(chunk_hidden_states), chunk_weights);

            if (zero_out_masked) {
                auto chunk_mask_tensor = sd::Tensor<float>::from_vector(chunk_mask)
                                             .reshape_({1, static_cast<int64_t>(chunk_mask.size())});
                chunk_hidden_states.masked_fill_(chunk_mask_tensor < 0.0f, 0.0f);
            }

            int64_t t1 = ggml_time_ms();
            LOG_DEBUG("computing condition graph completed, taking %" PRId64 " ms", t1 - t0);

            if (!hidden_states.empty()) {
                hidden_states = sd::ops::concat(hidden_states, chunk_hidden_states, 1);
            } else {
                hidden_states = std::move(chunk_hidden_states);
            }
        }

        modify_mask_to_attend_padding(&t5_attn_mask, static_cast<int>(t5_attn_mask.numel()), mask_pad);

        SDCondition result;
        result.c_crossattn = std::move(hidden_states);
        result.c_vector    = std::move(t5_attn_mask);
        return result;
    }

    SDCondition get_learned_condition(int n_threads,
                                      const ConditionerParams& conditioner_params) override {
        auto tokens_and_weights = tokenize(conditioner_params.text, chunk_len, chunk_len);
        return get_learned_condition_common(n_threads,
                                            tokens_and_weights,
                                            conditioner_params.clip_skip,
                                            conditioner_params.zero_out_masked);
    }
};

struct AnimaConditioner : public Conditioner {
    std::shared_ptr<BPETokenizer> qwen_tokenizer;
    T5UniGramTokenizer t5_tokenizer;
    std::shared_ptr<LLM::LLMRunner> llm;

    AnimaConditioner(ggml_backend_t backend,
                     bool offload_params_to_cpu,
                     const String2TensorStorage& tensor_storage_map = {}) {
        qwen_tokenizer = std::make_shared<Qwen2Tokenizer>();
        llm            = std::make_shared<LLM::LLMRunner>(LLM::LLMArch::QWEN3,
                                               backend,
                                               offload_params_to_cpu,
                                               tensor_storage_map,
                                               "text_encoders.llm",
                                               false);
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
        llm->get_param_tensors(tensors, "text_encoders.llm");
    }

    void alloc_params_buffer() override {
        llm->alloc_params_buffer();
    }

    void free_params_buffer() override {
        llm->free_params_buffer();
    }

    size_t get_params_buffer_size() override {
        return llm->get_params_buffer_size();
    }

    void set_flash_attention_enabled(bool enabled) override {
        llm->set_flash_attention_enabled(enabled);
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override {
        llm->set_weight_adapter(adapter);
    }

    std::tuple<std::vector<int>, std::vector<float>, std::vector<int>, std::vector<float>> tokenize(std::string text) {
        auto parsed_attention = parse_prompt_attention(text);

        {
            std::stringstream ss;
            ss << "[";
            for (const auto& item : parsed_attention) {
                ss << "['" << item.first << "', " << item.second << "], ";
            }
            ss << "]";
            LOG_DEBUG("parse '%s' to %s", text.c_str(), ss.str().c_str());
        }

        std::vector<int> qwen_tokens;
        std::vector<float> qwen_weights;
        std::vector<int> t5_tokens;
        std::vector<float> t5_weights;

        for (const auto& item : parsed_attention) {
            const std::string& curr_text = item.first;
            std::vector<int> curr_tokens = qwen_tokenizer->tokenize(curr_text, nullptr);
            qwen_tokens.insert(qwen_tokens.end(), curr_tokens.begin(), curr_tokens.end());
            // Anima uses uniform Qwen token weights.
            qwen_weights.insert(qwen_weights.end(), curr_tokens.size(), 1.f);
        }
        if (qwen_tokens.empty()) {
            qwen_tokens.push_back(151643);  // qwen3 pad token
            qwen_weights.push_back(1.f);
        }

        for (const auto& item : parsed_attention) {
            const std::string& curr_text = item.first;
            float curr_weight            = item.second;
            std::vector<int> curr_tokens = t5_tokenizer.tokenize(curr_text, nullptr, true);
            t5_tokens.insert(t5_tokens.end(), curr_tokens.begin(), curr_tokens.end());
            t5_weights.insert(t5_weights.end(), curr_tokens.size(), curr_weight);
        }

        return {qwen_tokens, qwen_weights, t5_tokens, t5_weights};
    }

    SDCondition get_learned_condition(int n_threads,
                                      const ConditionerParams& conditioner_params) override {
        int64_t t0 = ggml_time_ms();

        auto tokenized     = tokenize(conditioner_params.text);
        auto& qwen_tokens  = std::get<0>(tokenized);
        auto& qwen_weights = std::get<1>(tokenized);
        auto& t5_tokens    = std::get<2>(tokenized);
        auto& t5_weights   = std::get<3>(tokenized);

        sd::Tensor<int32_t> input_ids({static_cast<int64_t>(qwen_tokens.size()), 1}, qwen_tokens);
        auto hidden_states = llm->compute(n_threads,
                                          input_ids,
                                          sd::Tensor<float>(),
                                          {},
                                          {});
        GGML_ASSERT(!hidden_states.empty());
        hidden_states         = apply_token_weights(std::move(hidden_states), qwen_weights);
        auto t5_ids_tensor    = sd::Tensor<int32_t>::from_vector(t5_tokens);
        auto t5_weight_tensor = sd::Tensor<float>::from_vector(t5_weights);

        int64_t t1 = ggml_time_ms();
        LOG_DEBUG("computing condition graph completed, taking %" PRId64 " ms", t1 - t0);

        SDCondition result;
        result.c_crossattn  = std::move(hidden_states);
        result.c_t5_ids     = std::move(t5_ids_tensor);
        result.c_t5_weights = std::move(t5_weight_tensor);
        return result;
    }
};

struct LLMEmbedder : public Conditioner {
    SDVersion version;
    std::shared_ptr<BPETokenizer> tokenizer;
    std::shared_ptr<LLM::LLMRunner> llm;

    LLMEmbedder(ggml_backend_t backend,
                bool offload_params_to_cpu,
                const String2TensorStorage& tensor_storage_map = {},
                SDVersion version                              = VERSION_QWEN_IMAGE,
                const std::string prefix                       = "",
                bool enable_vision                             = false)
        : version(version) {
        LLM::LLMArch arch = LLM::LLMArch::QWEN2_5_VL;
        if (version == VERSION_FLUX2) {
            arch = LLM::LLMArch::MISTRAL_SMALL_3_2;
        } else if (sd_version_is_ernie_image(version)) {
            arch = LLM::LLMArch::MINISTRAL_3_3B;
        } else if (sd_version_is_z_image(version) || version == VERSION_OVIS_IMAGE || version == VERSION_FLUX2_KLEIN) {
            arch = LLM::LLMArch::QWEN3;
        }
        if (arch == LLM::LLMArch::MISTRAL_SMALL_3_2 || arch == LLM::LLMArch::MINISTRAL_3_3B) {
            tokenizer = std::make_shared<MistralTokenizer>();
        } else {
            tokenizer = std::make_shared<Qwen2Tokenizer>();
        }
        llm = std::make_shared<LLM::LLMRunner>(arch,
                                               backend,
                                               offload_params_to_cpu,
                                               tensor_storage_map,
                                               "text_encoders.llm",
                                               enable_vision);
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
        llm->get_param_tensors(tensors, "text_encoders.llm");
    }

    void alloc_params_buffer() override {
        llm->alloc_params_buffer();
    }

    void free_params_buffer() override {
        llm->free_params_buffer();
    }

    size_t get_params_buffer_size() override {
        size_t buffer_size = 0;
        buffer_size += llm->get_params_buffer_size();
        return buffer_size;
    }

    void set_flash_attention_enabled(bool enabled) override {
        llm->set_flash_attention_enabled(enabled);
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override {
        if (llm) {
            llm->set_weight_adapter(adapter);
        }
    }

    std::tuple<std::vector<int>, std::vector<float>, std::vector<float>> tokenize(std::string text,
                                                                                  const std::pair<int, int>& attn_range,
                                                                                  size_t min_length = 0,
                                                                                  size_t max_length = 100000000) {
        std::vector<std::pair<std::string, float>> parsed_attention;
        if (attn_range.first >= 0 && attn_range.second > 0) {
            if (attn_range.first > 0) {
                parsed_attention.emplace_back(text.substr(0, attn_range.first), 1.f);
            }
            if (attn_range.second - attn_range.first > 0) {
                auto new_parsed_attention = parse_prompt_attention(text.substr(attn_range.first, attn_range.second - attn_range.first));
                parsed_attention.insert(parsed_attention.end(),
                                        new_parsed_attention.begin(),
                                        new_parsed_attention.end());
            }
            if (attn_range.second < text.size()) {
                parsed_attention.emplace_back(text.substr(attn_range.second), 1.f);
            }
        } else {
            parsed_attention.emplace_back(text, 1.f);
        }

        {
            std::stringstream ss;
            ss << "[";
            for (const auto& item : parsed_attention) {
                ss << "['" << item.first << "', " << item.second << "], ";
            }
            ss << "]";
            LOG_DEBUG("parse '%s' to %s", text.c_str(), ss.str().c_str());
        }

        std::vector<int> tokens;
        std::vector<float> weights;
        for (const auto& item : parsed_attention) {
            const std::string& curr_text = item.first;
            float curr_weight            = item.second;
            std::vector<int> curr_tokens = tokenizer->encode(curr_text, nullptr);
            tokens.insert(tokens.end(), curr_tokens.begin(), curr_tokens.end());
            weights.insert(weights.end(), curr_tokens.size(), curr_weight);
        }

        std::vector<float> mask;
        tokenizer->pad_tokens(tokens, &weights, &mask, min_length, max_length);

        // for (int i = 0; i < tokens.size(); i++) {
        //     std::cout << tokens[i] << ":" << weights[i] << ", " << i << std::endl;
        // }
        // std::cout << std::endl;

        return {tokens, weights, mask};
    }

    sd::Tensor<float> encode_prompt(int n_threads,
                                    const std::string prompt,
                                    const std::pair<int, int>& prompt_attn_range,
                                    int min_length,
                                    int hidden_states_min_length,
                                    const std::vector<std::pair<int, sd::Tensor<float>>>& image_embeds,
                                    const std::set<int>& out_layers,
                                    int prompt_template_encode_start_idx) {
        auto tokens_weights_mask = tokenize(prompt, prompt_attn_range, min_length);
        auto& tokens             = std::get<0>(tokens_weights_mask);
        auto& weights            = std::get<1>(tokens_weights_mask);
        auto& mask               = std::get<2>(tokens_weights_mask);

        sd::Tensor<int32_t> input_ids({static_cast<int64_t>(tokens.size())}, tokens);
        sd::Tensor<float> attention_mask;
        if (!mask.empty()) {
            attention_mask = sd::Tensor<float>({static_cast<int64_t>(mask.size()), static_cast<int64_t>(mask.size())});
            for (size_t i1 = 0; i1 < mask.size(); ++i1) {
                for (size_t i0 = 0; i0 < mask.size(); ++i0) {
                    float value = 0.0f;
                    if (mask[i0] == 0.0f || i0 > i1) {
                        value = -INFINITY;
                    }
                    attention_mask[static_cast<int64_t>(i0 + mask.size() * i1)] = value;
                }
            }
        }

        auto hidden_states = llm->compute(n_threads,
                                          input_ids,
                                          attention_mask,
                                          image_embeds,
                                          out_layers);
        GGML_ASSERT(!hidden_states.empty());
        hidden_states = apply_token_weights(std::move(hidden_states), weights);
        GGML_ASSERT(hidden_states.shape()[1] > prompt_template_encode_start_idx);

        int64_t zero_pad_len = 0;
        if (hidden_states_min_length > 0) {
            if (hidden_states.shape()[1] - prompt_template_encode_start_idx < hidden_states_min_length) {
                zero_pad_len = hidden_states_min_length - hidden_states.shape()[1] + prompt_template_encode_start_idx;
            }
        }

        sd::Tensor<float> new_hidden_states = sd::ops::slice(hidden_states,
                                                             1,
                                                             prompt_template_encode_start_idx,
                                                             hidden_states.shape()[1]);
        if (zero_pad_len > 0) {
            auto pad_shape    = new_hidden_states.shape();
            pad_shape[1]      = zero_pad_len;
            new_hidden_states = sd::ops::concat(new_hidden_states,
                                                sd::Tensor<float>::zeros(std::move(pad_shape)),
                                                1);
        }

        return new_hidden_states;
    }

    SDCondition get_learned_condition(int n_threads,
                                      const ConditionerParams& conditioner_params) override {
        std::string prompt;
        std::pair<int, int> prompt_attn_range;
        std::vector<std::string> extra_prompts;
        std::vector<std::pair<int, int>> extra_prompts_attn_range;
        std::vector<std::pair<int, sd::Tensor<float>>> image_embeds;
        int prompt_template_encode_start_idx = 34;
        int min_length                       = 0;  // pad tokens
        int hidden_states_min_length         = 0;  // zero pad hidden_states
        std::set<int> out_layers;

        int64_t t0 = ggml_time_ms();

        if (sd_version_is_qwen_image(version)) {
            if (llm->enable_vision && conditioner_params.ref_images != nullptr && !conditioner_params.ref_images->empty()) {
                LOG_INFO("QwenImageEditPlusPipeline");
                prompt_template_encode_start_idx = 64;
                int image_embed_idx              = 64 + 6;

                int min_pixels          = 384 * 384;
                int max_pixels          = 560 * 560;
                std::string placeholder = "<|image_pad|>";
                std::string img_prompt;

                for (int i = 0; i < conditioner_params.ref_images->size(); i++) {
                    const auto& image = (*conditioner_params.ref_images)[i];
                    double factor     = llm->params.vision.patch_size * llm->params.vision.spatial_merge_size;
                    int height        = static_cast<int>(image.shape()[1]);
                    int width         = static_cast<int>(image.shape()[0]);
                    int h_bar         = static_cast<int>(std::round(height / factor) * factor);
                    int w_bar         = static_cast<int>(std::round(width / factor) * factor);

                    if (static_cast<double>(h_bar) * w_bar > max_pixels) {
                        double beta = std::sqrt((height * width) / static_cast<double>(max_pixels));
                        h_bar       = std::max(static_cast<int>(factor),
                                               static_cast<int>(std::floor(height / beta / factor)) * static_cast<int>(factor));
                        w_bar       = std::max(static_cast<int>(factor),
                                               static_cast<int>(std::floor(width / beta / factor)) * static_cast<int>(factor));
                    } else if (static_cast<double>(h_bar) * w_bar < min_pixels) {
                        double beta = std::sqrt(static_cast<double>(min_pixels) / (height * width));
                        h_bar       = static_cast<int>(std::ceil(height * beta / factor)) * static_cast<int>(factor);
                        w_bar       = static_cast<int>(std::ceil(width * beta / factor)) * static_cast<int>(factor);
                    }

                    LOG_DEBUG("resize conditioner ref image %d from %dx%d to %dx%d", i, height, width, h_bar, w_bar);

                    auto resized_image = clip_preprocess(image, w_bar, h_bar);

                    auto image_embed = llm->encode_image(n_threads, resized_image);
                    GGML_ASSERT(!image_embed.empty());
                    image_embeds.emplace_back(image_embed_idx, image_embed);
                    image_embed_idx += 1 + static_cast<int>(image_embed.shape()[1]) + 6;

                    img_prompt += "Picture " + std::to_string(i + 1) + ": <|vision_start|>";  // [24669, 220, index, 25, 220, 151652]
                    int64_t num_image_tokens = image_embed.shape()[1];
                    img_prompt.reserve(num_image_tokens * placeholder.size());
                    for (int j = 0; j < num_image_tokens; j++) {
                        img_prompt += placeholder;
                    }
                    img_prompt += "<|vision_end|>";
                }

                prompt = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n";
                prompt += img_prompt;

                prompt_attn_range.first = static_cast<int>(prompt.size());
                prompt += conditioner_params.text;
                prompt_attn_range.second = static_cast<int>(prompt.size());

                prompt += "<|im_end|>\n<|im_start|>assistant\n";
            } else {
                prompt_template_encode_start_idx = 34;

                prompt = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n";

                prompt_attn_range.first = static_cast<int>(prompt.size());
                prompt += conditioner_params.text;
                prompt_attn_range.second = static_cast<int>(prompt.size());

                prompt += "<|im_end|>\n<|im_start|>assistant\n";
            }
        } else if (version == VERSION_FLUX2) {
            prompt_template_encode_start_idx = 0;
            hidden_states_min_length         = 512;
            out_layers                       = {10, 20, 30};

            prompt = "[SYSTEM_PROMPT]You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object\nattribution and actions without speculation.[/SYSTEM_PROMPT][INST]";

            prompt_attn_range.first = static_cast<int>(prompt.size());
            prompt += conditioner_params.text;
            prompt_attn_range.second = static_cast<int>(prompt.size());

            prompt += "[/INST]";
        } else if (sd_version_is_ernie_image(version)) {
            prompt_template_encode_start_idx = 0;
            out_layers                       = {25};  // -2

            prompt_attn_range.first = 0;
            prompt += conditioner_params.text;
            prompt_attn_range.second = static_cast<int>(prompt.size());
        } else if (sd_version_is_z_image(version)) {
            prompt_template_encode_start_idx = 0;
            out_layers                       = {35};  // -2

            if (conditioner_params.ref_images != nullptr && !conditioner_params.ref_images->empty()) {
                LOG_INFO("ZImageOmniPipeline");
                prompt = "<|im_start|>user\n<|vision_start|>";
                for (int i = 0; i < conditioner_params.ref_images->size() - 1; i++) {
                    extra_prompts.push_back("<|vision_end|><|vision_start|>");
                }
                extra_prompts.push_back("<|vision_end|>" + conditioner_params.text + "<|im_end|>\n<|im_start|>assistant\n<|vision_start|>");
                extra_prompts.push_back("<|vision_end|><|im_end|>");
            } else {
                prompt = "<|im_start|>user\n";

                prompt_attn_range.first = static_cast<int>(prompt.size());
                prompt += conditioner_params.text;
                prompt_attn_range.second = static_cast<int>(prompt.size());

                prompt += "<|im_end|>\n<|im_start|>assistant\n";
            }
        } else if (version == VERSION_FLUX2_KLEIN) {
            prompt_template_encode_start_idx = 0;
            min_length                       = 512;
            out_layers                       = {9, 18, 27};

            prompt = "<|im_start|>user\n";

            prompt_attn_range.first = static_cast<int>(prompt.size());
            prompt += conditioner_params.text;
            prompt_attn_range.second = static_cast<int>(prompt.size());

            prompt += "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        } else if (version == VERSION_OVIS_IMAGE) {
            prompt_template_encode_start_idx = 28;
            min_length                       = prompt_template_encode_start_idx + 256;

            prompt = "<|im_start|>user\nDescribe the image by detailing the color, quantity, text, shape, size, texture, spatial relationships of the objects and background:";

            prompt_attn_range.first = static_cast<int>(prompt.size());
            prompt += " " + conditioner_params.text;
            prompt_attn_range.second = static_cast<int>(prompt.size());

            prompt += "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        } else {
            GGML_ABORT("unknown version %d", version);
        }

        auto hidden_states = encode_prompt(n_threads,
                                           prompt,
                                           prompt_attn_range,
                                           min_length,
                                           hidden_states_min_length,
                                           image_embeds,
                                           out_layers,
                                           prompt_template_encode_start_idx);
        std::vector<sd::Tensor<float>> extra_hidden_states_vec;
        for (int i = 0; i < extra_prompts.size(); i++) {
            auto extra_hidden_states = encode_prompt(n_threads,
                                                     extra_prompts[i],
                                                     extra_prompts_attn_range[i],
                                                     min_length,
                                                     hidden_states_min_length,
                                                     image_embeds,
                                                     out_layers,
                                                     prompt_template_encode_start_idx);
            extra_hidden_states_vec.push_back(std::move(extra_hidden_states));
        }

        int64_t t1 = ggml_time_ms();
        LOG_DEBUG("computing condition graph completed, taking %" PRId64 " ms", t1 - t0);
        SDCondition result;
        result.c_crossattn        = std::move(hidden_states);
        result.extra_c_crossattns = std::move(extra_hidden_states_vec);
        return result;
    }
};

// LTX-2 conditioner: Gemma 3 text encoder → feature extractor → 1D connector →
// DiT cross-attention context. Supports both V1 (19B) and V2 (22B) feature
// extractor variants, auto-detected from the tensor map.
//
// Key prefixes (native LTX-2 checkpoint layout, no name-conversion applied):
//   text_encoder.model.*                                  Gemma weights
//   text_embedding_projection.aggregate_embed.*           V1 FeatureExtractorV1 (19B)
//   text_embedding_projection.video_aggregate_embed.*     V2 FeatureExtractorV2 video branch (22B)
//   text_embedding_projection.audio_aggregate_embed.*     V2 audio branch (22B, currently unused)
//   model.diffusion_model.embeddings_connector.*          V1 Embeddings1DConnector (19B)
//   model.diffusion_model.video_embeddings_connector.*    V2 video connector (22B)
//   model.diffusion_model.caption_projection.*            V1 PixArt caption_projection (on DiT)
//                                                          (V2 has no caption_projection — feature
//                                                           extractor already outputs DiT's inner_dim)
//
// If neither V1 nor V2 connector weights are present (e.g. Gemma-only test
// checkpoints), the conditioner falls back to returning the final post-norm
// hidden state — the same cheap path we had before Phase 9 landed.
struct LTX2GemmaConditioner : public Conditioner {
    std::shared_ptr<LLM::LLMRunner> llm;
    std::shared_ptr<GemmaTokenizer> tokenizer;
    std::shared_ptr<LTXConnector::LTX2ConnectorRunner> connector_runner;
    std::string prefix;
    std::string tokenizer_path;
    int64_t gemma_hidden_size   = 0;
    int gemma_num_hidden_layers = 0;
    // True when using the V2 (22B) feature extractor; used by get_learned_condition
    // to pick the right CPU normalization path.
    bool use_v2_feature_extractor = false;

    LTX2GemmaConditioner(ggml_backend_t backend,
                         bool offload_params_to_cpu,
                         const String2TensorStorage& tensor_storage_map,
                         const std::string prefix            = "text_encoder",
                         const std::string tokenizer_path    = "",
                         const std::string feat_ext_prefix   = "text_embedding_projection",
                         const std::string connector_prefix_arg = "")
        : prefix(prefix), tokenizer_path(tokenizer_path) {
        llm = std::make_shared<LLM::LLMRunner>(LLM::LLMArch::GEMMA3,
                                               backend,
                                               offload_params_to_cpu,
                                               tensor_storage_map,
                                               prefix,
                                               /*enable_vision=*/false);
        gemma_hidden_size       = llm->params.hidden_size;
        gemma_num_hidden_layers = static_cast<int>(llm->params.num_layers);

        if (!tokenizer_path.empty()) {
            tokenizer = std::make_shared<GemmaTokenizer>();
            if (!tokenizer->load_from_file(tokenizer_path)) {
                LOG_WARN("LTX2GemmaConditioner: failed to load Gemma tokenizer from '%s'", tokenizer_path.c_str());
                tokenizer.reset();
            }
        }

        // Auto-detect V1 vs V2 feature extractor + connector prefix variant.
        //   V2 (22B): text_embedding_projection.video_aggregate_embed.{weight,bias} +
        //             model.diffusion_model.video_embeddings_connector.*
        //   V1 (19B): text_embedding_projection.aggregate_embed.weight +
        //             model.diffusion_model.embeddings_connector.*
        // `connector_prefix_arg` is honored when non-empty, otherwise we probe both.
        const std::string& feat_ext_pre = feat_ext_prefix;

        auto agg_v1_it = tensor_storage_map.find(feat_ext_pre + ".aggregate_embed.weight");
        auto agg_v2_it = tensor_storage_map.find(feat_ext_pre + ".video_aggregate_embed.weight");

        std::string connector_pre;
        LTXConnector::FeatureExtractorVersion fe_version = LTXConnector::FeatureExtractorVersion::V1;
        int64_t flat_dim  = 0;
        int64_t inner_dim = 0;

        if (agg_v2_it != tensor_storage_map.end()) {
            fe_version = LTXConnector::FeatureExtractorVersion::V2;
            flat_dim   = agg_v2_it->second.ne[0];
            inner_dim  = agg_v2_it->second.ne[1];
            use_v2_feature_extractor = true;
            connector_pre = connector_prefix_arg.empty()
                                ? "model.diffusion_model.video_embeddings_connector"
                                : connector_prefix_arg;
        } else if (agg_v1_it != tensor_storage_map.end()) {
            fe_version = LTXConnector::FeatureExtractorVersion::V1;
            flat_dim   = agg_v1_it->second.ne[0];
            inner_dim  = agg_v1_it->second.ne[1];
            connector_pre = connector_prefix_arg.empty()
                                ? "model.diffusion_model.embeddings_connector"
                                : connector_prefix_arg;
        } else {
            LOG_INFO("LTX2GemmaConditioner: no feature_extractor weights found — falling back to "
                     "last_hidden_state pass-through (Gemma-only mode)");
            return;
        }

        auto conn0_it = tensor_storage_map.find(connector_pre + ".transformer_1d_blocks.0.attn1.to_q.weight");
        if (conn0_it == tensor_storage_map.end()) {
            LOG_WARN("LTX2GemmaConditioner: feature_extractor weights present but connector at '%s' is missing; "
                     "falling back to last_hidden_state",
                     connector_pre.c_str());
            return;
        }
        if (conn0_it->second.ne[1] != inner_dim) {
            LOG_WARN("LTX2GemmaConditioner: connector to_q out_features=%lld does not match "
                     "feature_extractor inner_dim=%lld; skipping connector.",
                     (long long)conn0_it->second.ne[1], (long long)inner_dim);
            return;
        }

        // Count connector layers by probing to_q presence.
        int num_layers = 0;
        while (tensor_storage_map.find(connector_pre + ".transformer_1d_blocks." +
                                       std::to_string(num_layers) + ".attn1.to_q.weight") !=
               tensor_storage_map.end()) {
            num_layers++;
        }

        // num_registers from learnable_registers.ne (ne[0]=inner_dim, ne[1]=num_registers).
        int num_registers = 0;
        auto reg_it = tensor_storage_map.find(connector_pre + ".learnable_registers");
        if (reg_it != tensor_storage_map.end() && reg_it->second.n_dims >= 2) {
            num_registers = static_cast<int>(reg_it->second.ne[1]);
        }

        // Detect gated attention inside the connector (V2 / 22B has this).
        bool apply_gated = tensor_storage_map.find(
            connector_pre + ".transformer_1d_blocks.0.attn1.to_gate_logits.weight") !=
                           tensor_storage_map.end();

        // LTX-2 fixes head_dim=128 across both variants.
        int head_dim  = 128;
        int num_heads = static_cast<int>(inner_dim / head_dim);

        // We do NOT include caption_projection here — V1 has it on the DiT side,
        // V2 has none. Pass source_dim=Gemma hidden so V2's sqrt(target/source)
        // rescale is applied correctly.
        connector_runner = std::make_shared<LTXConnector::LTX2ConnectorRunner>(
            backend, offload_params_to_cpu,
            flat_dim, num_heads, head_dim, num_layers, num_registers,
            /*caption_channels=*/0, /*caption_hidden=*/0, /*caption_out=*/0,
            /*theta=*/10000.0f, /*max_pos=*/std::vector<int>{1},
            tensor_storage_map,
            /*include_caption_projection=*/false,
            feat_ext_pre, connector_pre, /*caption_proj_prefix=*/"",
            fe_version, /*source_dim=*/gemma_hidden_size, apply_gated);
        LOG_INFO("LTX2GemmaConditioner: wired %s connector (flat_dim=%lld inner_dim=%lld "
                 "num_layers=%d num_registers=%d gated=%d)",
                 fe_version == LTXConnector::FeatureExtractorVersion::V2 ? "V2" : "V1",
                 (long long)flat_dim, (long long)inner_dim, num_layers, num_registers,
                 apply_gated ? 1 : 0);
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
        llm->get_param_tensors(tensors, prefix);
        if (connector_runner) {
            connector_runner->get_param_tensors(tensors);
        }
    }
    void get_llm_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
        if (llm) llm->get_param_tensors(tensors, prefix);
    }
    void get_non_llm_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
        if (connector_runner) connector_runner->get_param_tensors(tensors);
    }
    void alloc_params_buffer() override {
        llm->alloc_params_buffer();
        if (connector_runner) connector_runner->alloc_params_buffer();
    }
    void free_params_buffer() override {
        llm->free_params_buffer();
        if (connector_runner) connector_runner->free_params_buffer();
    }
    size_t get_params_buffer_size() override {
        size_t s = llm->get_params_buffer_size();
        if (connector_runner) s += connector_runner->get_params_buffer_size();
        return s;
    }
    void set_flash_attention_enabled(bool enabled) override {
        llm->set_flash_attention_enabled(enabled);
        if (connector_runner) connector_runner->set_flash_attention_enabled(enabled);
    }

    void set_llm_lazy_load(std::function<bool()> fn) override {
        if (llm) llm->set_lazy_load(std::move(fn));
    }

    SDCondition get_learned_condition(int n_threads,
                                      const ConditionerParams& p) override {
        if (!tokenizer) {
            LOG_ERROR("LTX2GemmaConditioner: no tokenizer loaded. Construct the conditioner "
                      "with a path to Gemma's tokenizer.json.");
            GGML_ABORT("Gemma tokenizer missing");
        }
        // HuggingFace Gemma tokenizer always prepends <bos>; we replicate that here
        // so the encoder sees the same sequence the Python reference does.
        std::vector<int> real_ids = tokenizer->tokenize(p.text, nullptr, /*padding=*/false);
        real_ids.insert(real_ids.begin(), tokenizer->BOS_TOKEN_ID);
        const int64_t T_real = static_cast<int64_t>(real_ids.size());
        LOG_DEBUG("LTX2GemmaConditioner: tokenized prompt '%s' -> %lld real tokens",
                  p.text.c_str(), (long long)T_real);
        sd::Tensor<float> empty_mask;

        if (!connector_runner) {
            // No connector weights: behave like before Phase 9 landed (no padding).
            sd::Tensor<int32_t> ids_tensor({T_real, 1});
            for (int64_t i = 0; i < T_real; ++i) ids_tensor.data()[i] = real_ids[i];
            auto last_hidden = llm->compute(n_threads, ids_tensor, empty_mask, {}, {});
            SDCondition cond;
            cond.c_crossattn = last_hidden;
            return cond;
        }

        // Python LTX-2 tokenizer pads to max_length=1024 with padding_side="left"
        // and pad_token = EOS:
        //   ltx_core/text_encoders/gemma/tokenizer.py:21-24 (padding_side="left",
        //   pad_token=EOS) and ltx_core/text_encoders/gemma/encoders/base_encoder.py:182
        //   (`LTXVGemmaTokenizer(tokenizer_root, 1024)`).
        // Gemma processes the full max_length, and the connector then sees a
        // max_length-long sequence with learnable_registers tiled max_length/num_reg
        // times (8× on the 22B V2 path, where num_reg=128). Padding only to
        // num_registers produces the wrong Gemma RoPE positions for the real tokens
        // and cuts the DiT cross-attention context by the same factor; both regress
        // output quality from recognisable subjects to colored-blob textures.
        const int num_registers   = connector_runner->num_registers;
        // The 1024-pad-with-register-tile path (Python ref's
        // `Embeddings1DConnector._replace_padded_with_learnable_registers`) produces
        // wrong subjects on LTX-2.3 22B (e.g. "fiery sword" → vintage farm scene),
        // while feeding T_real real tokens through the connector blocks produces
        // correctly prompt-locked output. Until we localize the discrepancy in the
        // tile path, default to the no-pad/no-tile path. Set LTX2_PAD=1 to opt back
        // into the Python-ref tile path (e.g. for parity dumps).
        const bool no_pad = std::getenv("LTX2_PAD") == nullptr;
        const int64_t max_length = no_pad ? T_real : 1024;
        int64_t T_pad            = 0;
        int64_t T                = T_real;
        if (T_real < max_length) {
            T_pad = max_length - T_real;
            T     = max_length;
        } else if (T_real > max_length) {
            // Prompt already exceeds max_length — truncate to match tokenizer
            // behaviour (`truncation=True` in LTXVGemmaTokenizer).
            LOG_WARN("LTX2GemmaConditioner: prompt tokenised to %lld >= max_length=%lld; truncating.",
                     (long long)T_real, (long long)max_length);
            real_ids.resize(static_cast<size_t>(max_length));
            T      = max_length;
            T_pad  = 0;
        }
        sd::Tensor<int32_t> input_ids({T, 1});
        for (int64_t i = 0; i < T_pad; ++i) input_ids.data()[i] = tokenizer->EOS_TOKEN_ID;
        const int64_t real_to_write = std::min<int64_t>(T_real, max_length);
        for (int64_t i = 0; i < real_to_write; ++i) input_ids.data()[T_pad + i] = real_ids[i];
        // In no_pad mode the connector skips the register tile (target_seq_len = T_real).
        // In padded mode (default) we tile to 1024 to match Python's Embeddings1DConnector.
        if (no_pad) {
            connector_runner->set_target_seq_len(static_cast<int>(T_real));
        } else {
            GGML_ASSERT(num_registers == 0 || max_length % num_registers == 0);
            connector_runner->set_target_seq_len(static_cast<int>(max_length));
        }

        // 1. Gemma: compute all N+1 hidden states on the padded sequence. Passing
        //    T_pad as pad_count tells build_graph to mask out positions [0, T_pad)
        //    as keys for any real query — without this the real tokens at [T_pad, T)
        //    attend across all 1024-T_real left-padded EOS tokens and lose subject
        //    information. HF transformers does this implicitly when given an
        //    attention_mask=[0..0,1..1] alongside left-padded input_ids.
        //    Layout returned by compute_all_hidden_states: ne [N+1, H, T, B] =
        //    PyTorch [B, T, H, N+1] (stack of per-layer hidden states).
        auto stacked   = llm->compute_all_hidden_states(n_threads, input_ids, empty_mask,
                                                        /*pad_count=*/static_cast<int>(T_pad));
        const int64_t B = 1;
        const int64_t D = gemma_hidden_size;
        const int64_t L = gemma_num_hidden_layers + 1;
        GGML_ASSERT(stacked.numel() == L * D * T * B);

        if (const char* dump_path = std::getenv("SD_DUMP_COND_STACKED")) {
            FILE* f = std::fopen(dump_path, "wb");
            if (f) {
                std::fwrite(stacked.data(), sizeof(float), stacked.numel(), f);
                std::fclose(f);
                LOG_INFO("SD_DUMP_COND_STACKED: wrote %ld floats to %s (ne=[%ld,%ld,%ld,%ld])",
                         (long)stacked.numel(), dump_path, (long)L, (long)D, (long)T, (long)B);
            }
        }

        // 2. CPU normalize → [B, T, D*L]. seq_lens=[T_real_eff] + left-padding tells
        //    the normalizer to zero out the pad positions (which live at [0, T_pad)).
        //    T_real_eff caps at max_length to handle the truncated-prompt branch above.
        const int64_t T_real_eff = std::min<int64_t>(T_real, max_length);
        std::vector<int> seq_lens(B, static_cast<int>(T_real_eff));
        sd::Tensor<float> normed({D * L, T, B});
        if (use_v2_feature_extractor) {
            LTXConnector::feature_extractor_normalize_v2(
                stacked.data(), seq_lens.data(), normed.data(),
                static_cast<int>(B), static_cast<int>(T), static_cast<int>(D), static_cast<int>(L),
                "left", 1e-6f);
        } else {
            LTXConnector::feature_extractor_normalize(
                stacked.data(), seq_lens.data(), normed.data(),
                static_cast<int>(B), static_cast<int>(T), static_cast<int>(D), static_cast<int>(L),
                "left", 1e-6f);
        }

        // Python's Embeddings1DConnector._replace_padded_with_learnable_registers moves
        // the real-token rows from [T_pad, T) to the START of the sequence and replaces
        // the now-empty tail with learnable_registers[T_real:]. Equivalent CPU-side shift:
        // after normalize, [0,T_pad) holds zeros (masked pad), [T_pad,T) holds real.
        // Slide the real rows down to [0,T_real) and re-zero the tail — the connector
        // runner then tiles/slices learnable_registers[T_real:max_length] and concats.
        if (T_pad > 0) {
            const int64_t flat_dim = D * L;
            sd::Tensor<float> reals({flat_dim, T_real_eff, B});
            for (int64_t b = 0; b < B; ++b) {
                std::memcpy(reals.data() + b * T_real_eff * flat_dim,
                            normed.data() + b * T * flat_dim + T_pad * flat_dim,
                            static_cast<size_t>(T_real_eff * flat_dim) * sizeof(float));
            }
            normed = std::move(reals);
        }

        // 3. Run connector. Stage selection:
        //    0 = stop right after the feature_extractor projection (no register
        //        tiling, no transformer_1d_blocks, no final rms_norm). This is
        //        what PR #1459 / #1463 do — they treat the embeddings_connector
        //        as a refinement layer and skip it.
        //    3 = full Embeddings1DConnector (Python-faithful: register-tile to
        //        max_length, run all blocks, final rms_norm).
        // Defaulting to stage=3 (faithful). Override with LTX2_COND_STAGE=N for
        // diagnostics.
        int cond_stage = 3;
        if (const char* cs = std::getenv("LTX2_COND_STAGE")) {
            cond_stage = std::atoi(cs);
        }
        auto context = connector_runner->compute(n_threads, normed, cond_stage);

        if (const char* dump_path = std::getenv("SD_DUMP_COND_CONTEXT")) {
            FILE* f = std::fopen(dump_path, "wb");
            if (f) {
                std::fwrite(context.data(), sizeof(float), context.numel(), f);
                std::fclose(f);
                LOG_INFO("SD_DUMP_COND_CONTEXT: wrote %ld floats to %s",
                         (long)context.numel(), dump_path);
            }
        }

        SDCondition cond;
        cond.c_crossattn = context;
        return cond;
    }
};

#endif
