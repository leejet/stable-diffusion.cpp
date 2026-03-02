#ifndef __CONDITIONER_HPP__
#define __CONDITIONER_HPP__

#include "clip.hpp"
#include "llm.hpp"
#include "t5.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <random>

struct SDCondition {
    struct ggml_tensor* c_crossattn = nullptr;  // aka context
    struct ggml_tensor* c_vector    = nullptr;  // aka y
    struct ggml_tensor* c_concat    = nullptr;
    struct ggml_tensor* c_lyrics    = nullptr;  // ace: lyric embedding
    struct ggml_tensor* refer_audio = nullptr;  // ace: reference audio (acoustic hidden states)
    std::shared_ptr<std::vector<int>> audio_codes;  // ace: semantic audio codes

    std::vector<struct ggml_tensor*> extra_c_crossattns;

    SDCondition() = default;
    SDCondition(struct ggml_tensor* c_crossattn,
                struct ggml_tensor* c_vector,
                struct ggml_tensor* c_concat,
                const std::vector<struct ggml_tensor*>& extra_c_crossattns = {})
        : c_crossattn(c_crossattn), c_vector(c_vector), c_concat(c_concat), extra_c_crossattns(extra_c_crossattns) {}
};

struct ConditionerParams {
    std::string text;
    std::string lyrics;
    std::string keyscale = "C major";
    std::string language = "en";
    float bpm            = 120.f;
    float duration       = 120.f;
    int timesignature    = 2;
    int lm_seed          = 0;
    int clip_skip                       = -1;
    int width                           = -1;
    int height                          = -1;
    int adm_in_channels                 = -1;
    bool zero_out_masked                = false;
    int num_input_imgs                  = 0;   // for photomaker
    std::vector<sd_image_t*> ref_images = {};  // for qwen image edit
};

struct Conditioner {
    virtual SDCondition get_learned_condition(ggml_context* work_ctx,
                                              int n_threads,
                                              const ConditionerParams& conditioner_params) = 0;
    virtual void alloc_params_buffer()                                                     = 0;
    virtual void free_params_buffer()                                                      = 0;
    virtual void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors)    = 0;
    virtual size_t get_params_buffer_size()                                                = 0;
    virtual void set_flash_attention_enabled(bool enabled)                                 = 0;
    virtual void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) {}
    virtual std::tuple<SDCondition, std::vector<bool>> get_learned_condition_with_trigger(ggml_context* work_ctx,
                                                                                          int n_threads,
                                                                                          const ConditionerParams& conditioner_params) {
        GGML_ABORT("Not implemented yet!");
    }
    virtual std::string remove_trigger_from_prompt(ggml_context* work_ctx,
                                                   const std::string& prompt) {
        GGML_ABORT("Not implemented yet!");
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

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors) override {
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
        struct ggml_init_params params;
        params.mem_size               = 100 * 1024 * 1024;  // max for custom embeddings 100 MB
        params.mem_buffer             = nullptr;
        params.no_alloc               = false;
        struct ggml_context* embd_ctx = ggml_init(params);
        struct ggml_tensor* embd      = nullptr;
        struct ggml_tensor* embd2     = nullptr;
        auto on_load                  = [&](const TensorStorage& tensor_storage, ggml_tensor** dst_tensor) {
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

    std::tuple<std::vector<int>, std::vector<float>, std::vector<bool>>
    tokenize_with_trigger_token(std::string text,
                                int num_input_imgs,
                                int32_t image_token,
                                bool padding = false) {
        return tokenize_with_trigger_token(text, num_input_imgs, image_token,
                                           text_model->model.n_token, padding);
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
                                int32_t image_token,
                                size_t max_length = 0,
                                bool padding      = false) {
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

        tokenizer.pad_tokens(tokens, weights, max_length, padding);
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
                                                             bool padding = false) {
        return tokenize(text, text_model->model.n_token, padding);
    }

    std::pair<std::vector<int>, std::vector<float>> tokenize(std::string text,
                                                             size_t max_length = 0,
                                                             bool padding      = false) {
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

        tokenizer.pad_tokens(tokens, weights, max_length, padding);

        // for (int i = 0; i < tokens.size(); i++) {
        //     std::cout << tokens[i] << ":" << weights[i] << ", ";
        // }
        // std::cout << std::endl;

        return {tokens, weights};
    }

    SDCondition get_learned_condition_common(ggml_context* work_ctx,
                                             int n_threads,
                                             std::vector<int>& tokens,
                                             std::vector<float>& weights,
                                             int clip_skip,
                                             int width,
                                             int height,
                                             int adm_in_channels  = -1,
                                             bool zero_out_masked = false) {
        int64_t t0                               = ggml_time_ms();
        struct ggml_tensor* hidden_states        = nullptr;  // [N, n_token, hidden_size]
        struct ggml_tensor* chunk_hidden_states  = nullptr;  // [n_token, hidden_size] or [n_token, hidden_size + hidden_size2]
        struct ggml_tensor* chunk_hidden_states1 = nullptr;  // [n_token, hidden_size]
        struct ggml_tensor* chunk_hidden_states2 = nullptr;  // [n_token, hidden_size2]
        struct ggml_tensor* pooled               = nullptr;
        std::vector<float> hidden_states_vec;

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

            auto input_ids                 = vector_to_ggml_tensor_i32(work_ctx, chunk_tokens);
            struct ggml_tensor* input_ids2 = nullptr;
            size_t max_token_idx           = 0;
            if (sd_version_is_sdxl(version)) {
                auto it = std::find(chunk_tokens.begin(), chunk_tokens.end(), tokenizer.EOS_TOKEN_ID);
                if (it != chunk_tokens.end()) {
                    std::fill(std::next(it), chunk_tokens.end(), 0);
                }

                max_token_idx = std::min<size_t>(std::distance(chunk_tokens.begin(), it), chunk_tokens.size() - 1);

                input_ids2 = vector_to_ggml_tensor_i32(work_ctx, chunk_tokens);

                // for (int i = 0; i < chunk_tokens.size(); i++) {
                //     printf("%d ", chunk_tokens[i]);
                // }
                // printf("\n");
            }

            {
                text_model->compute(n_threads,
                                    input_ids,
                                    num_custom_embeddings,
                                    token_embed_custom.data(),
                                    max_token_idx,
                                    false,
                                    clip_skip,
                                    &chunk_hidden_states1,
                                    work_ctx);
                if (sd_version_is_sdxl(version)) {
                    text_model2->compute(n_threads,
                                         input_ids2,
                                         num_custom_embeddings,
                                         token_embed_custom.data(),
                                         max_token_idx,
                                         false,
                                         clip_skip,
                                         &chunk_hidden_states2, work_ctx);
                    // concat
                    chunk_hidden_states = ggml_ext_tensor_concat(work_ctx, chunk_hidden_states1, chunk_hidden_states2, 0);

                    if (chunk_idx == 0) {
                        text_model2->compute(n_threads,
                                             input_ids2,
                                             num_custom_embeddings,
                                             token_embed_custom.data(),
                                             max_token_idx,
                                             true,
                                             clip_skip,
                                             &pooled,
                                             work_ctx);
                    }
                } else {
                    chunk_hidden_states = chunk_hidden_states1;
                }
            }

            int64_t t1 = ggml_time_ms();
            LOG_DEBUG("computing condition graph completed, taking %" PRId64 " ms", t1 - t0);
            ggml_tensor* result = ggml_dup_tensor(work_ctx, chunk_hidden_states);
            {
                float original_mean = ggml_ext_tensor_mean(chunk_hidden_states);
                for (int i2 = 0; i2 < chunk_hidden_states->ne[2]; i2++) {
                    for (int i1 = 0; i1 < chunk_hidden_states->ne[1]; i1++) {
                        for (int i0 = 0; i0 < chunk_hidden_states->ne[0]; i0++) {
                            float value = ggml_ext_tensor_get_f32(chunk_hidden_states, i0, i1, i2);
                            value *= chunk_weights[i1];
                            ggml_ext_tensor_set_f32(result, value, i0, i1, i2);
                        }
                    }
                }
                float new_mean = ggml_ext_tensor_mean(result);
                ggml_ext_tensor_scale_inplace(result, (original_mean / new_mean));
            }
            if (zero_out_masked) {
                float* vec = (float*)result->data;
                for (int i = 0; i < ggml_nelements(result); i++) {
                    vec[i] = 0;
                }
            }
            hidden_states_vec.insert(hidden_states_vec.end(), (float*)result->data, ((float*)result->data) + ggml_nelements(result));
        }

        hidden_states = vector_to_ggml_tensor(work_ctx, hidden_states_vec);
        hidden_states = ggml_reshape_2d(work_ctx,
                                        hidden_states,
                                        chunk_hidden_states->ne[0],
                                        ggml_nelements(hidden_states) / chunk_hidden_states->ne[0]);

        ggml_tensor* vec = nullptr;
        if (sd_version_is_sdxl(version)) {
            int out_dim = 256;
            vec         = ggml_new_tensor_1d(work_ctx, GGML_TYPE_F32, adm_in_channels);
            // [0:1280]
            size_t offset = 0;
            memcpy(vec->data, pooled->data, ggml_nbytes(pooled));
            offset += ggml_nbytes(pooled);

            // original_size_as_tuple
            float orig_width             = (float)width;
            float orig_height            = (float)height;
            std::vector<float> timesteps = {orig_height, orig_width};

            ggml_tensor* embed_view = ggml_view_2d(work_ctx, vec, out_dim, 2, ggml_type_size(GGML_TYPE_F32) * out_dim, offset);
            offset += ggml_nbytes(embed_view);
            set_timestep_embedding(timesteps, embed_view, out_dim);
            // print_ggml_tensor(ggml_reshape_1d(work_ctx, embed_view, out_dim * 2));
            // crop_coords_top_left
            float crop_coord_top  = 0.f;
            float crop_coord_left = 0.f;
            timesteps             = {crop_coord_top, crop_coord_left};
            embed_view            = ggml_view_2d(work_ctx, vec, out_dim, 2, ggml_type_size(GGML_TYPE_F32) * out_dim, offset);
            offset += ggml_nbytes(embed_view);
            set_timestep_embedding(timesteps, embed_view, out_dim);
            // print_ggml_tensor(ggml_reshape_1d(work_ctx, embed_view, out_dim * 2));
            // target_size_as_tuple
            float target_width  = (float)width;
            float target_height = (float)height;
            timesteps           = {target_height, target_width};
            embed_view          = ggml_view_2d(work_ctx, vec, out_dim, 2, ggml_type_size(GGML_TYPE_F32) * out_dim, offset);
            offset += ggml_nbytes(embed_view);
            set_timestep_embedding(timesteps, embed_view, out_dim);
            // print_ggml_tensor(ggml_reshape_1d(work_ctx, embed_view, out_dim * 2));
            GGML_ASSERT(offset == ggml_nbytes(vec));
        }
        // print_ggml_tensor(result);
        return {hidden_states, vec, nullptr};
    }

    std::tuple<SDCondition, std::vector<bool>>
    get_learned_condition_with_trigger(ggml_context* work_ctx,
                                       int n_threads,
                                       const ConditionerParams& conditioner_params) override {
        auto image_tokens = convert_token_to_id(trigger_word);
        // if(image_tokens.size() == 1){
        //     printf(" image token id is: %d \n", image_tokens[0]);
        // }
        GGML_ASSERT(image_tokens.size() == 1);
        auto tokens_and_weights     = tokenize_with_trigger_token(conditioner_params.text,
                                                                  conditioner_params.num_input_imgs,
                                                                  image_tokens[0],
                                                                  true);
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
        auto cond = get_learned_condition_common(work_ctx,
                                                 n_threads,
                                                 tokens,
                                                 weights,
                                                 conditioner_params.clip_skip,
                                                 conditioner_params.width,
                                                 conditioner_params.height,
                                                 conditioner_params.adm_in_channels,
                                                 conditioner_params.zero_out_masked);
        return std::make_tuple(cond, clsm);
    }

    std::string remove_trigger_from_prompt(ggml_context* work_ctx,
                                           const std::string& prompt) override {
        auto image_tokens = convert_token_to_id(trigger_word);
        GGML_ASSERT(image_tokens.size() == 1);
        auto tokens_and_weights  = tokenize(prompt, false);
        std::vector<int>& tokens = tokens_and_weights.first;
        auto it                  = std::find(tokens.begin(), tokens.end(), image_tokens[0]);
        GGML_ASSERT(it != tokens.end());  // prompt must have trigger word
        tokens.erase(it);
        return decode(tokens);
    }

    SDCondition get_learned_condition(ggml_context* work_ctx,
                                      int n_threads,
                                      const ConditionerParams& conditioner_params) override {
        auto tokens_and_weights     = tokenize(conditioner_params.text, true);
        std::vector<int>& tokens    = tokens_and_weights.first;
        std::vector<float>& weights = tokens_and_weights.second;
        return get_learned_condition_common(work_ctx,
                                            n_threads,
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

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors) {
        vision_model.get_param_tensors(tensors, "cond_stage_model.transformer");
    }

    struct ggml_cgraph* build_graph(struct ggml_tensor* pixel_values, bool return_pooled, int clip_skip) {
        struct ggml_cgraph* gf = ggml_new_graph(compute_ctx);

        pixel_values = to_backend(pixel_values);

        auto runner_ctx = get_context();

        struct ggml_tensor* hidden_states = vision_model.forward(&runner_ctx, pixel_values, return_pooled, clip_skip);

        ggml_build_forward_expand(gf, hidden_states);

        return gf;
    }

    bool compute(const int n_threads,
                 ggml_tensor* pixel_values,
                 bool return_pooled,
                 int clip_skip,
                 ggml_tensor** output,
                 ggml_context* output_ctx) {
        auto get_graph = [&]() -> struct ggml_cgraph* {
            return build_graph(pixel_values, return_pooled, clip_skip);
        };
        return GGMLRunner::compute(get_graph, n_threads, true, output, output_ctx);
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

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors) override {
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
                                                                          size_t max_length = 0,
                                                                          bool padding      = false) {
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
                std::vector<int> curr_tokens = t5_tokenizer.Encode(curr_text, true);
                t5_tokens.insert(t5_tokens.end(), curr_tokens.begin(), curr_tokens.end());
                t5_weights.insert(t5_weights.end(), curr_tokens.size(), curr_weight);
            }
        }

        if (clip_l) {
            clip_l_tokenizer.pad_tokens(clip_l_tokens, clip_l_weights, max_length, padding);
        }
        if (clip_g) {
            clip_g_tokenizer.pad_tokens(clip_g_tokens, clip_g_weights, max_length, padding);
        }
        if (t5) {
            t5_tokenizer.pad_tokens(t5_tokens, t5_weights, nullptr, max_length, padding);
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

    SDCondition get_learned_condition_common(ggml_context* work_ctx,
                                             int n_threads,
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

        int64_t t0                                 = ggml_time_ms();
        struct ggml_tensor* hidden_states          = nullptr;  // [N, n_token*2, 4096]
        struct ggml_tensor* chunk_hidden_states    = nullptr;  // [n_token*2, 4096]
        struct ggml_tensor* chunk_hidden_states_l  = nullptr;  // [n_token, hidden_size_l]
        struct ggml_tensor* chunk_hidden_states_g  = nullptr;  // [n_token, hidden_size_g]
        struct ggml_tensor* chunk_hidden_states_t5 = nullptr;  // [n_token, hidden_size_t5]
        struct ggml_tensor* pooled                 = nullptr;
        struct ggml_tensor* pooled_l               = nullptr;  // [768,]
        struct ggml_tensor* pooled_g               = nullptr;  // [1280,]
        std::vector<float> hidden_states_vec;

        size_t chunk_len   = 77;
        size_t chunk_count = std::max(std::max(clip_l_tokens.size(), clip_g_tokens.size()), t5_tokens.size()) / chunk_len;
        for (int chunk_idx = 0; chunk_idx < chunk_count; chunk_idx++) {
            // clip_l
            if (clip_l) {
                std::vector<int> chunk_tokens(clip_l_tokens.begin() + chunk_idx * chunk_len,
                                              clip_l_tokens.begin() + (chunk_idx + 1) * chunk_len);
                std::vector<float> chunk_weights(clip_l_weights.begin() + chunk_idx * chunk_len,
                                                 clip_l_weights.begin() + (chunk_idx + 1) * chunk_len);

                auto input_ids       = vector_to_ggml_tensor_i32(work_ctx, chunk_tokens);
                size_t max_token_idx = 0;

                clip_l->compute(n_threads,
                                input_ids,
                                0,
                                nullptr,
                                max_token_idx,
                                false,
                                clip_skip,
                                &chunk_hidden_states_l,
                                work_ctx);
                {
                    auto tensor         = chunk_hidden_states_l;
                    float original_mean = ggml_ext_tensor_mean(tensor);
                    for (int i2 = 0; i2 < tensor->ne[2]; i2++) {
                        for (int i1 = 0; i1 < tensor->ne[1]; i1++) {
                            for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                                float value = ggml_ext_tensor_get_f32(tensor, i0, i1, i2);
                                value *= chunk_weights[i1];
                                ggml_ext_tensor_set_f32(tensor, value, i0, i1, i2);
                            }
                        }
                    }
                    float new_mean = ggml_ext_tensor_mean(tensor);
                    ggml_ext_tensor_scale_inplace(tensor, (original_mean / new_mean));
                }

                if (chunk_idx == 0) {
                    auto it       = std::find(chunk_tokens.begin(), chunk_tokens.end(), clip_l_tokenizer.EOS_TOKEN_ID);
                    max_token_idx = std::min<size_t>(std::distance(chunk_tokens.begin(), it), chunk_tokens.size() - 1);
                    clip_l->compute(n_threads,
                                    input_ids,
                                    0,
                                    nullptr,
                                    max_token_idx,
                                    true,
                                    clip_skip,
                                    &pooled_l,
                                    work_ctx);
                }
            } else {
                chunk_hidden_states_l = ggml_new_tensor_2d(work_ctx, GGML_TYPE_F32, 768, chunk_len);
                ggml_set_f32(chunk_hidden_states_l, 0.f);
                if (chunk_idx == 0) {
                    pooled_l = ggml_new_tensor_1d(work_ctx, GGML_TYPE_F32, 768);
                    ggml_set_f32(pooled_l, 0.f);
                }
            }

            // clip_g
            if (clip_g) {
                std::vector<int> chunk_tokens(clip_g_tokens.begin() + chunk_idx * chunk_len,
                                              clip_g_tokens.begin() + (chunk_idx + 1) * chunk_len);
                std::vector<float> chunk_weights(clip_g_weights.begin() + chunk_idx * chunk_len,
                                                 clip_g_weights.begin() + (chunk_idx + 1) * chunk_len);

                auto input_ids       = vector_to_ggml_tensor_i32(work_ctx, chunk_tokens);
                size_t max_token_idx = 0;

                clip_g->compute(n_threads,
                                input_ids,
                                0,
                                nullptr,
                                max_token_idx,
                                false,
                                clip_skip,
                                &chunk_hidden_states_g,
                                work_ctx);

                {
                    auto tensor         = chunk_hidden_states_g;
                    float original_mean = ggml_ext_tensor_mean(tensor);
                    for (int i2 = 0; i2 < tensor->ne[2]; i2++) {
                        for (int i1 = 0; i1 < tensor->ne[1]; i1++) {
                            for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                                float value = ggml_ext_tensor_get_f32(tensor, i0, i1, i2);
                                value *= chunk_weights[i1];
                                ggml_ext_tensor_set_f32(tensor, value, i0, i1, i2);
                            }
                        }
                    }
                    float new_mean = ggml_ext_tensor_mean(tensor);
                    ggml_ext_tensor_scale_inplace(tensor, (original_mean / new_mean));
                }

                if (chunk_idx == 0) {
                    auto it       = std::find(chunk_tokens.begin(), chunk_tokens.end(), clip_g_tokenizer.EOS_TOKEN_ID);
                    max_token_idx = std::min<size_t>(std::distance(chunk_tokens.begin(), it), chunk_tokens.size() - 1);
                    clip_g->compute(n_threads,
                                    input_ids,
                                    0,
                                    nullptr,
                                    max_token_idx,
                                    true,
                                    clip_skip,
                                    &pooled_g,
                                    work_ctx);
                }
            } else {
                chunk_hidden_states_g = ggml_new_tensor_2d(work_ctx, GGML_TYPE_F32, 1280, chunk_len);
                ggml_set_f32(chunk_hidden_states_g, 0.f);
                if (chunk_idx == 0) {
                    pooled_g = ggml_new_tensor_1d(work_ctx, GGML_TYPE_F32, 1280);
                    ggml_set_f32(pooled_g, 0.f);
                }
            }

            // t5
            if (t5) {
                std::vector<int> chunk_tokens(t5_tokens.begin() + chunk_idx * chunk_len,
                                              t5_tokens.begin() + (chunk_idx + 1) * chunk_len);
                std::vector<float> chunk_weights(t5_weights.begin() + chunk_idx * chunk_len,
                                                 t5_weights.begin() + (chunk_idx + 1) * chunk_len);

                auto input_ids = vector_to_ggml_tensor_i32(work_ctx, chunk_tokens);

                t5->compute(n_threads,
                            input_ids,
                            nullptr,
                            &chunk_hidden_states_t5,
                            work_ctx);
                {
                    auto tensor         = chunk_hidden_states_t5;
                    float original_mean = ggml_ext_tensor_mean(tensor);
                    for (int i2 = 0; i2 < tensor->ne[2]; i2++) {
                        for (int i1 = 0; i1 < tensor->ne[1]; i1++) {
                            for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                                float value = ggml_ext_tensor_get_f32(tensor, i0, i1, i2);
                                value *= chunk_weights[i1];
                                ggml_ext_tensor_set_f32(tensor, value, i0, i1, i2);
                            }
                        }
                    }
                    float new_mean = ggml_ext_tensor_mean(tensor);
                    ggml_ext_tensor_scale_inplace(tensor, (original_mean / new_mean));
                }
            } else {
                chunk_hidden_states_t5 = ggml_new_tensor_2d(work_ctx, GGML_TYPE_F32, 4096, chunk_len);
                ggml_set_f32(chunk_hidden_states_t5, 0.f);
            }

            auto chunk_hidden_states_lg_pad = ggml_new_tensor_3d(work_ctx,
                                                                 chunk_hidden_states_l->type,
                                                                 4096,
                                                                 chunk_hidden_states_l->ne[1],
                                                                 chunk_hidden_states_l->ne[2]);  // [n_token, 4096]

            for (int i2 = 0; i2 < chunk_hidden_states_lg_pad->ne[2]; i2++) {
                for (int i1 = 0; i1 < chunk_hidden_states_lg_pad->ne[1]; i1++) {
                    for (int i0 = 0; i0 < chunk_hidden_states_lg_pad->ne[0]; i0++) {
                        float value = 0.f;
                        if (i0 < chunk_hidden_states_l->ne[0]) {
                            value = ggml_ext_tensor_get_f32(chunk_hidden_states_l, i0, i1, i2);
                        } else if (i0 < chunk_hidden_states_l->ne[0] + chunk_hidden_states_g->ne[0]) {
                            value = ggml_ext_tensor_get_f32(chunk_hidden_states_g, i0 - chunk_hidden_states_l->ne[0], i1, i2);
                        }
                        ggml_ext_tensor_set_f32(chunk_hidden_states_lg_pad, value, i0, i1, i2);
                    }
                }
            }

            chunk_hidden_states = ggml_ext_tensor_concat(work_ctx, chunk_hidden_states_lg_pad, chunk_hidden_states_t5, 1);  // [n_token*2, 4096]

            if (chunk_idx == 0) {
                pooled = ggml_ext_tensor_concat(work_ctx, pooled_l, pooled_g, 0);  // [768 + 1280]
            }

            int64_t t1 = ggml_time_ms();
            LOG_DEBUG("computing condition graph completed, taking %" PRId64 " ms", t1 - t0);
            if (zero_out_masked) {
                float* vec = (float*)chunk_hidden_states->data;
                for (int i = 0; i < ggml_nelements(chunk_hidden_states); i++) {
                    vec[i] = 0;
                }
            }

            hidden_states_vec.insert(hidden_states_vec.end(),
                                     (float*)chunk_hidden_states->data,
                                     ((float*)chunk_hidden_states->data) + ggml_nelements(chunk_hidden_states));
        }

        if (hidden_states_vec.size() > 0) {
            hidden_states = vector_to_ggml_tensor(work_ctx, hidden_states_vec);
            hidden_states = ggml_reshape_2d(work_ctx,
                                            hidden_states,
                                            chunk_hidden_states->ne[0],
                                            ggml_nelements(hidden_states) / chunk_hidden_states->ne[0]);
        } else {
            hidden_states = ggml_new_tensor_2d(work_ctx, GGML_TYPE_F32, 4096, 256);
            ggml_set_f32(hidden_states, 0.f);
        }
        if (pooled == nullptr) {
            pooled = ggml_new_tensor_1d(work_ctx, GGML_TYPE_F32, 2048);
            ggml_set_f32(pooled, 0.f);
        }
        return {hidden_states, pooled, nullptr};
    }

    SDCondition get_learned_condition(ggml_context* work_ctx,
                                      int n_threads,
                                      const ConditionerParams& conditioner_params) override {
        auto tokens_and_weights = tokenize(conditioner_params.text, 77, true);
        return get_learned_condition_common(work_ctx,
                                            n_threads,
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

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors) override {
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
                                                                          size_t max_length = 0,
                                                                          bool padding      = false) {
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
                std::vector<int> curr_tokens = t5_tokenizer.Encode(curr_text, true);
                t5_tokens.insert(t5_tokens.end(), curr_tokens.begin(), curr_tokens.end());
                t5_weights.insert(t5_weights.end(), curr_tokens.size(), curr_weight);
            }
        }

        if (clip_l) {
            clip_l_tokenizer.pad_tokens(clip_l_tokens, clip_l_weights, 77, padding);
        }
        if (t5) {
            t5_tokenizer.pad_tokens(t5_tokens, t5_weights, nullptr, max_length, padding);
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

    SDCondition get_learned_condition_common(ggml_context* work_ctx,
                                             int n_threads,
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

        int64_t t0                              = ggml_time_ms();
        struct ggml_tensor* hidden_states       = nullptr;  // [N, n_token, 4096]
        struct ggml_tensor* chunk_hidden_states = nullptr;  // [n_token, 4096]
        struct ggml_tensor* pooled              = nullptr;  // [768,]
        std::vector<float> hidden_states_vec;

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

                    auto input_ids       = vector_to_ggml_tensor_i32(work_ctx, chunk_tokens);
                    size_t max_token_idx = 0;

                    auto it       = std::find(chunk_tokens.begin(), chunk_tokens.end(), clip_l_tokenizer.EOS_TOKEN_ID);
                    max_token_idx = std::min<size_t>(std::distance(chunk_tokens.begin(), it), chunk_tokens.size() - 1);

                    clip_l->compute(n_threads,
                                    input_ids,
                                    0,
                                    nullptr,
                                    max_token_idx,
                                    true,
                                    clip_skip,
                                    &pooled,
                                    work_ctx);
                }
            }

            // t5
            if (t5) {
                std::vector<int> chunk_tokens(t5_tokens.begin() + chunk_idx * chunk_len,
                                              t5_tokens.begin() + (chunk_idx + 1) * chunk_len);
                std::vector<float> chunk_weights(t5_weights.begin() + chunk_idx * chunk_len,
                                                 t5_weights.begin() + (chunk_idx + 1) * chunk_len);

                auto input_ids = vector_to_ggml_tensor_i32(work_ctx, chunk_tokens);

                t5->compute(n_threads,
                            input_ids,
                            nullptr,
                            &chunk_hidden_states,
                            work_ctx);
                {
                    auto tensor         = chunk_hidden_states;
                    float original_mean = ggml_ext_tensor_mean(tensor);
                    for (int i2 = 0; i2 < tensor->ne[2]; i2++) {
                        for (int i1 = 0; i1 < tensor->ne[1]; i1++) {
                            for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                                float value = ggml_ext_tensor_get_f32(tensor, i0, i1, i2);
                                value *= chunk_weights[i1];
                                ggml_ext_tensor_set_f32(tensor, value, i0, i1, i2);
                            }
                        }
                    }
                    float new_mean = ggml_ext_tensor_mean(tensor);
                    ggml_ext_tensor_scale_inplace(tensor, (original_mean / new_mean));
                }
            } else {
                chunk_hidden_states = ggml_new_tensor_2d(work_ctx, GGML_TYPE_F32, 4096, chunk_len);
                ggml_set_f32(chunk_hidden_states, 0.f);
            }

            int64_t t1 = ggml_time_ms();
            LOG_DEBUG("computing condition graph completed, taking %" PRId64 " ms", t1 - t0);
            if (zero_out_masked) {
                float* vec = (float*)chunk_hidden_states->data;
                for (int i = 0; i < ggml_nelements(chunk_hidden_states); i++) {
                    vec[i] = 0;
                }
            }

            hidden_states_vec.insert(hidden_states_vec.end(),
                                     (float*)chunk_hidden_states->data,
                                     ((float*)chunk_hidden_states->data) + ggml_nelements(chunk_hidden_states));
        }

        if (hidden_states_vec.size() > 0) {
            hidden_states = vector_to_ggml_tensor(work_ctx, hidden_states_vec);
            hidden_states = ggml_reshape_2d(work_ctx,
                                            hidden_states,
                                            chunk_hidden_states->ne[0],
                                            ggml_nelements(hidden_states) / chunk_hidden_states->ne[0]);
        } else {
            hidden_states = ggml_new_tensor_2d(work_ctx, GGML_TYPE_F32, 4096, 256);
            ggml_set_f32(hidden_states, 0.f);
        }
        if (pooled == nullptr) {
            pooled = ggml_new_tensor_1d(work_ctx, GGML_TYPE_F32, 768);
            ggml_set_f32(pooled, 0.f);
        }
        return {hidden_states, pooled, nullptr};
    }

    SDCondition get_learned_condition(ggml_context* work_ctx,
                                      int n_threads,
                                      const ConditionerParams& conditioner_params) override {
        auto tokens_and_weights = tokenize(conditioner_params.text, chunk_len, true);
        return get_learned_condition_common(work_ctx,
                                            n_threads,
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
    int mask_pad     = 1;
    bool is_umt5     = false;

    T5CLIPEmbedder(ggml_backend_t backend,
                   bool offload_params_to_cpu,
                   const String2TensorStorage& tensor_storage_map = {},
                   bool use_mask                                  = false,
                   int mask_pad                                   = 1,
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

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors) override {
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
                                                                                  size_t max_length = 0,
                                                                                  bool padding      = false) {
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

                std::vector<int> curr_tokens = t5_tokenizer.Encode(curr_text, true);
                t5_tokens.insert(t5_tokens.end(), curr_tokens.begin(), curr_tokens.end());
                t5_weights.insert(t5_weights.end(), curr_tokens.size(), curr_weight);
            }

            t5_tokenizer.pad_tokens(t5_tokens, t5_weights, &t5_mask, max_length, padding);
        }
        return {t5_tokens, t5_weights, t5_mask};
    }

    void modify_mask_to_attend_padding(struct ggml_tensor* mask, int max_seq_length, int num_extra_padding = 8) {
        float* mask_data = (float*)mask->data;
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

    SDCondition get_learned_condition_common(ggml_context* work_ctx,
                                             int n_threads,
                                             std::tuple<std::vector<int>, std::vector<float>, std::vector<float>> token_and_weights,
                                             int clip_skip,
                                             bool zero_out_masked = false) {
        if (!t5) {
            auto hidden_states = ggml_new_tensor_2d(work_ctx, GGML_TYPE_F32, 4096, 256);
            ggml_set_f32(hidden_states, 0.f);
            auto t5_attn_mask = ggml_new_tensor_1d(work_ctx, GGML_TYPE_F32, 256);
            ggml_set_f32(t5_attn_mask, -HUGE_VALF);
            return {hidden_states, t5_attn_mask, nullptr};
        }
        auto& t5_tokens        = std::get<0>(token_and_weights);
        auto& t5_weights       = std::get<1>(token_and_weights);
        auto& t5_attn_mask_vec = std::get<2>(token_and_weights);

        int64_t t0                              = ggml_time_ms();
        struct ggml_tensor* hidden_states       = nullptr;  // [N, n_token, 4096]
        struct ggml_tensor* chunk_hidden_states = nullptr;  // [n_token, 4096]
        struct ggml_tensor* pooled              = nullptr;
        struct ggml_tensor* t5_attn_mask        = vector_to_ggml_tensor(work_ctx, t5_attn_mask_vec);  // [n_token]

        std::vector<float> hidden_states_vec;

        size_t chunk_count = t5_tokens.size() / chunk_len;

        for (int chunk_idx = 0; chunk_idx < chunk_count; chunk_idx++) {
            // t5
            std::vector<int> chunk_tokens(t5_tokens.begin() + chunk_idx * chunk_len,
                                          t5_tokens.begin() + (chunk_idx + 1) * chunk_len);
            std::vector<float> chunk_weights(t5_weights.begin() + chunk_idx * chunk_len,
                                             t5_weights.begin() + (chunk_idx + 1) * chunk_len);
            std::vector<float> chunk_mask(t5_attn_mask_vec.begin() + chunk_idx * chunk_len,
                                          t5_attn_mask_vec.begin() + (chunk_idx + 1) * chunk_len);

            auto input_ids          = vector_to_ggml_tensor_i32(work_ctx, chunk_tokens);
            auto t5_attn_mask_chunk = use_mask ? vector_to_ggml_tensor(work_ctx, chunk_mask) : nullptr;

            t5->compute(n_threads,
                        input_ids,
                        t5_attn_mask_chunk,
                        &chunk_hidden_states,
                        work_ctx);
            {
                auto tensor         = chunk_hidden_states;
                float original_mean = ggml_ext_tensor_mean(tensor);
                for (int i2 = 0; i2 < tensor->ne[2]; i2++) {
                    for (int i1 = 0; i1 < tensor->ne[1]; i1++) {
                        for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                            float value = ggml_ext_tensor_get_f32(tensor, i0, i1, i2);
                            value *= chunk_weights[i1];
                            ggml_ext_tensor_set_f32(tensor, value, i0, i1, i2);
                        }
                    }
                }
                float new_mean = ggml_ext_tensor_mean(tensor);
                ggml_ext_tensor_scale_inplace(tensor, (original_mean / new_mean));
            }

            int64_t t1 = ggml_time_ms();
            LOG_DEBUG("computing condition graph completed, taking %" PRId64 " ms", t1 - t0);
            if (zero_out_masked) {
                auto tensor = chunk_hidden_states;
                for (int i2 = 0; i2 < tensor->ne[2]; i2++) {
                    for (int i1 = 0; i1 < tensor->ne[1]; i1++) {
                        for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                            if (chunk_mask[i1] < 0.f) {
                                ggml_ext_tensor_set_f32(tensor, 0.f, i0, i1, i2);
                            }
                        }
                    }
                }
            }

            hidden_states_vec.insert(hidden_states_vec.end(),
                                     (float*)chunk_hidden_states->data,
                                     ((float*)chunk_hidden_states->data) + ggml_nelements(chunk_hidden_states));
        }

        GGML_ASSERT(hidden_states_vec.size() > 0);
        hidden_states = vector_to_ggml_tensor(work_ctx, hidden_states_vec);
        hidden_states = ggml_reshape_2d(work_ctx,
                                        hidden_states,
                                        chunk_hidden_states->ne[0],
                                        ggml_nelements(hidden_states) / chunk_hidden_states->ne[0]);

        modify_mask_to_attend_padding(t5_attn_mask, static_cast<int>(ggml_nelements(t5_attn_mask)), mask_pad);

        return {hidden_states, t5_attn_mask, nullptr};
    }

    SDCondition get_learned_condition(ggml_context* work_ctx,
                                      int n_threads,
                                      const ConditionerParams& conditioner_params) override {
        auto tokens_and_weights = tokenize(conditioner_params.text, chunk_len, true);
        return get_learned_condition_common(work_ctx,
                                            n_threads,
                                            tokens_and_weights,
                                            conditioner_params.clip_skip,
                                            conditioner_params.zero_out_masked);
    }
};

struct AnimaConditioner : public Conditioner {
    std::shared_ptr<LLM::BPETokenizer> qwen_tokenizer;
    T5UniGramTokenizer t5_tokenizer;
    std::shared_ptr<LLM::LLMRunner> llm;

    AnimaConditioner(ggml_backend_t backend,
                     bool offload_params_to_cpu,
                     const String2TensorStorage& tensor_storage_map = {}) {
        qwen_tokenizer = std::make_shared<LLM::Qwen2Tokenizer>();
        llm            = std::make_shared<LLM::LLMRunner>(LLM::LLMArch::QWEN3,
                                               backend,
                                               offload_params_to_cpu,
                                               tensor_storage_map,
                                               "text_encoders.llm",
                                               false);
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors) override {
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
            std::vector<int> curr_tokens = t5_tokenizer.Encode(curr_text, true);
            t5_tokens.insert(t5_tokens.end(), curr_tokens.begin(), curr_tokens.end());
            t5_weights.insert(t5_weights.end(), curr_tokens.size(), curr_weight);
        }

        return {qwen_tokens, qwen_weights, t5_tokens, t5_weights};
    }

    SDCondition get_learned_condition(ggml_context* work_ctx,
                                      int n_threads,
                                      const ConditionerParams& conditioner_params) override {
        int64_t t0 = ggml_time_ms();

        auto tokenized     = tokenize(conditioner_params.text);
        auto& qwen_tokens  = std::get<0>(tokenized);
        auto& qwen_weights = std::get<1>(tokenized);
        auto& t5_tokens    = std::get<2>(tokenized);
        auto& t5_weights   = std::get<3>(tokenized);

        auto input_ids = vector_to_ggml_tensor_i32(work_ctx, qwen_tokens);

        struct ggml_tensor* hidden_states = nullptr;  // [N, n_token, 1024]
        llm->compute(n_threads,
                     input_ids,
                     nullptr,
                     {},
                     {},
                     &hidden_states,
                     work_ctx);

        {
            auto tensor         = hidden_states;
            float original_mean = ggml_ext_tensor_mean(tensor);
            for (int i2 = 0; i2 < tensor->ne[2]; i2++) {
                for (int i1 = 0; i1 < tensor->ne[1]; i1++) {
                    for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                        float value = ggml_ext_tensor_get_f32(tensor, i0, i1, i2);
                        value *= qwen_weights[i1];
                        ggml_ext_tensor_set_f32(tensor, value, i0, i1, i2);
                    }
                }
            }
            float new_mean = ggml_ext_tensor_mean(tensor);
            if (new_mean != 0.f) {
                ggml_ext_tensor_scale_inplace(tensor, (original_mean / new_mean));
            }
        }

        struct ggml_tensor* t5_ids_tensor    = nullptr;
        struct ggml_tensor* t5_weight_tensor = nullptr;
        if (!t5_tokens.empty()) {
            t5_ids_tensor    = vector_to_ggml_tensor_i32(work_ctx, t5_tokens);
            t5_weight_tensor = vector_to_ggml_tensor(work_ctx, t5_weights);
        }

        int64_t t1 = ggml_time_ms();
        LOG_DEBUG("computing condition graph completed, taking %" PRId64 " ms", t1 - t0);

        return {hidden_states, t5_weight_tensor, t5_ids_tensor};
    }
};

struct LLMEmbedder : public Conditioner {
    SDVersion version;
    std::shared_ptr<LLM::BPETokenizer> tokenizer;
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
        } else if (sd_version_is_z_image(version) || version == VERSION_OVIS_IMAGE || version == VERSION_FLUX2_KLEIN) {
            arch = LLM::LLMArch::QWEN3;
        }
        if (arch == LLM::LLMArch::MISTRAL_SMALL_3_2) {
            tokenizer = std::make_shared<LLM::MistralTokenizer>();
        } else {
            tokenizer = std::make_shared<LLM::Qwen2Tokenizer>();
        }
        llm = std::make_shared<LLM::LLMRunner>(arch,
                                               backend,
                                               offload_params_to_cpu,
                                               tensor_storage_map,
                                               "text_encoders.llm",
                                               enable_vision);
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors) override {
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

    std::tuple<std::vector<int>, std::vector<float>> tokenize(std::string text,
                                                              const std::pair<int, int>& attn_range,
                                                              size_t max_length = 0,
                                                              bool padding      = false) {
        std::vector<std::pair<std::string, float>> parsed_attention;
        if (attn_range.first >= 0 && attn_range.second > 0) {
            parsed_attention.emplace_back(text.substr(0, attn_range.first), 1.f);
            if (attn_range.second - attn_range.first > 0) {
                auto new_parsed_attention = parse_prompt_attention(text.substr(attn_range.first, attn_range.second - attn_range.first));
                parsed_attention.insert(parsed_attention.end(),
                                        new_parsed_attention.begin(),
                                        new_parsed_attention.end());
            }
            parsed_attention.emplace_back(text.substr(attn_range.second), 1.f);
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
            std::vector<int> curr_tokens = tokenizer->tokenize(curr_text, nullptr);
            tokens.insert(tokens.end(), curr_tokens.begin(), curr_tokens.end());
            weights.insert(weights.end(), curr_tokens.size(), curr_weight);
        }

        tokenizer->pad_tokens(tokens, weights, max_length, padding);

        // for (int i = 0; i < tokens.size(); i++) {
        //     std::cout << tokens[i] << ":" << weights[i] << ", " << i << std::endl;
        // }
        // std::cout << std::endl;

        return {tokens, weights};
    }

    ggml_tensor* encode_prompt(ggml_context* work_ctx,
                               int n_threads,
                               const std::string prompt,
                               const std::pair<int, int>& prompt_attn_range,
                               int max_length,
                               int min_length,
                               std::vector<std::pair<int, ggml_tensor*>> image_embeds,
                               const std::set<int>& out_layers,
                               int prompt_template_encode_start_idx) {
        auto tokens_and_weights = tokenize(prompt, prompt_attn_range);
        auto& tokens            = std::get<0>(tokens_and_weights);
        auto& weights           = std::get<1>(tokens_and_weights);
        std::vector<float> mask;

        if (max_length > 0 && tokens.size() < max_length) {
            mask.insert(mask.end(), tokens.size(), 1.f);
            mask.insert(mask.end(), max_length - tokens.size(), 0.f);
            tokenizer->pad_tokens(tokens, weights, max_length, true);
        }

        struct ggml_tensor* hidden_states = nullptr;  // [N, n_token, hidden_size]

        auto input_ids = vector_to_ggml_tensor_i32(work_ctx, tokens);

        ggml_tensor* attention_mask = nullptr;
        if (!mask.empty()) {
            attention_mask = ggml_new_tensor_2d(work_ctx, GGML_TYPE_F32, mask.size(), mask.size());
            ggml_ext_tensor_iter(attention_mask, [&](ggml_tensor* attention_mask, int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
                float value = 0.f;
                if (mask[i0] == 0.f) {
                    value = -INFINITY;
                } else if (i0 > i1) {
                    value = -INFINITY;
                }
                ggml_ext_tensor_set_f32(attention_mask, value, i0, i1, i2, i3);
            });
        }

        llm->compute(n_threads,
                     input_ids,
                     attention_mask,
                     image_embeds,
                     out_layers,
                     &hidden_states,
                     work_ctx);
        {
            auto tensor         = hidden_states;
            float original_mean = ggml_ext_tensor_mean(tensor);
            for (int i2 = 0; i2 < tensor->ne[2]; i2++) {
                for (int i1 = 0; i1 < tensor->ne[1]; i1++) {
                    for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                        float value = ggml_ext_tensor_get_f32(tensor, i0, i1, i2);
                        value *= weights[i1];
                        ggml_ext_tensor_set_f32(tensor, value, i0, i1, i2);
                    }
                }
            }
            float new_mean = ggml_ext_tensor_mean(tensor);
            ggml_ext_tensor_scale_inplace(tensor, (original_mean / new_mean));
        }

        GGML_ASSERT(hidden_states->ne[1] > prompt_template_encode_start_idx);

        int64_t zero_pad_len = 0;
        if (min_length > 0) {
            if (hidden_states->ne[1] - prompt_template_encode_start_idx < min_length) {
                zero_pad_len = min_length - hidden_states->ne[1] + prompt_template_encode_start_idx;
            }
        }

        ggml_tensor* new_hidden_states = ggml_new_tensor_3d(work_ctx,
                                                            GGML_TYPE_F32,
                                                            hidden_states->ne[0],
                                                            hidden_states->ne[1] - prompt_template_encode_start_idx + zero_pad_len,
                                                            hidden_states->ne[2]);

        ggml_ext_tensor_iter(new_hidden_states, [&](ggml_tensor* new_hidden_states, int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
            float value = 0.f;
            if (i1 + prompt_template_encode_start_idx < hidden_states->ne[1]) {
                value = ggml_ext_tensor_get_f32(hidden_states, i0, i1 + prompt_template_encode_start_idx, i2, i3);
            }
            ggml_ext_tensor_set_f32(new_hidden_states, value, i0, i1, i2, i3);
        });

        return new_hidden_states;
    }

    SDCondition get_learned_condition(ggml_context* work_ctx,
                                      int n_threads,
                                      const ConditionerParams& conditioner_params) override {
        std::string prompt;
        std::pair<int, int> prompt_attn_range;
        std::vector<std::string> extra_prompts;
        std::vector<std::pair<int, int>> extra_prompts_attn_range;
        std::vector<std::pair<int, ggml_tensor*>> image_embeds;
        int prompt_template_encode_start_idx = 34;
        int max_length                       = 0;  // pad tokens
        int min_length                       = 0;  // zero pad hidden_states
        std::set<int> out_layers;

        int64_t t0 = ggml_time_ms();

        if (sd_version_is_qwen_image(version)) {
            if (llm->enable_vision && !conditioner_params.ref_images.empty()) {
                LOG_INFO("QwenImageEditPlusPipeline");
                prompt_template_encode_start_idx = 64;
                int image_embed_idx              = 64 + 6;

                int min_pixels          = 384 * 384;
                int max_pixels          = 560 * 560;
                std::string placeholder = "<|image_pad|>";
                std::string img_prompt;

                for (int i = 0; i < conditioner_params.ref_images.size(); i++) {
                    sd_image_f32_t image = sd_image_t_to_sd_image_f32_t(*conditioner_params.ref_images[i]);
                    double factor        = llm->params.vision.patch_size * llm->params.vision.spatial_merge_size;
                    int height           = image.height;
                    int width            = image.width;
                    int h_bar            = static_cast<int>(std::round(height / factor) * factor);
                    int w_bar            = static_cast<int>(std::round(width / factor) * factor);

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

                    LOG_DEBUG("resize conditioner ref image %d from %dx%d to %dx%d", i, image.height, image.width, h_bar, w_bar);

                    sd_image_f32_t resized_image = clip_preprocess(image, w_bar, h_bar);
                    free(image.data);
                    image.data = nullptr;

                    ggml_tensor* image_tensor = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, resized_image.width, resized_image.height, 3, 1);
                    sd_image_f32_to_ggml_tensor(resized_image, image_tensor, false);
                    free(resized_image.data);
                    resized_image.data = nullptr;

                    ggml_tensor* image_embed = nullptr;
                    llm->encode_image(n_threads, image_tensor, &image_embed, work_ctx);
                    image_embeds.emplace_back(image_embed_idx, image_embed);
                    image_embed_idx += 1 + static_cast<int>(image_embed->ne[1]) + 6;

                    img_prompt += "Picture " + std::to_string(i + 1) + ": <|vision_start|>";  // [24669, 220, index, 25, 220, 151652]
                    int64_t num_image_tokens = image_embed->ne[1];
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
            min_length                       = 512;
            out_layers                       = {10, 20, 30};

            prompt = "[SYSTEM_PROMPT]You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object\nattribution and actions without speculation.[/SYSTEM_PROMPT][INST]";

            prompt_attn_range.first = static_cast<int>(prompt.size());
            prompt += conditioner_params.text;
            prompt_attn_range.second = static_cast<int>(prompt.size());

            prompt += "[/INST]";
        } else if (sd_version_is_z_image(version)) {
            prompt_template_encode_start_idx = 0;
            out_layers                       = {35};  // -2

            if (!conditioner_params.ref_images.empty()) {
                LOG_INFO("ZImageOmniPipeline");
                prompt = "<|im_start|>user\n<|vision_start|>";
                for (int i = 0; i < conditioner_params.ref_images.size() - 1; i++) {
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
            max_length                       = 512;
            out_layers                       = {9, 18, 27};

            prompt = "<|im_start|>user\n";

            prompt_attn_range.first = static_cast<int>(prompt.size());
            prompt += conditioner_params.text;
            prompt_attn_range.second = static_cast<int>(prompt.size());

            prompt += "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        } else if (version == VERSION_OVIS_IMAGE) {
            prompt_template_encode_start_idx = 28;
            max_length                       = prompt_template_encode_start_idx + 256;

            prompt = "<|im_start|>user\nDescribe the image by detailing the color, quantity, text, shape, size, texture, spatial relationships of the objects and background:";

            prompt_attn_range.first = static_cast<int>(prompt.size());
            prompt += " " + conditioner_params.text;
            prompt_attn_range.second = static_cast<int>(prompt.size());

            prompt += "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        } else {
            GGML_ABORT("unknown version %d", version);
        }

        auto hidden_states = encode_prompt(work_ctx,
                                           n_threads,
                                           prompt,
                                           prompt_attn_range,
                                           max_length,
                                           min_length,
                                           image_embeds,
                                           out_layers,
                                           prompt_template_encode_start_idx);

        std::vector<ggml_tensor*> extra_hidden_states_vec;
        for (int i = 0; i < extra_prompts.size(); i++) {
            auto extra_hidden_states = encode_prompt(work_ctx,
                                                     n_threads,
                                                     extra_prompts[i],
                                                     extra_prompts_attn_range[i],
                                                     max_length,
                                                     min_length,
                                                     image_embeds,
                                                     out_layers,
                                                     prompt_template_encode_start_idx);
            extra_hidden_states_vec.push_back(extra_hidden_states);
        }

        int64_t t1 = ggml_time_ms();
        LOG_DEBUG("computing condition graph completed, taking %" PRId64 " ms", t1 - t0);
        return {hidden_states, nullptr, nullptr, extra_hidden_states_vec};
    }
};

struct AceConditioner : public Conditioner {
    std::shared_ptr<LLM::Qwen3Tokenizer> tokenizer;
    std::shared_ptr<LLM::LLMRunner> base_llm;
    std::shared_ptr<LLM::LLMRunner> lm_llm;
    std::string base_llm_prefix;
    std::string lm_llm_prefix;

    static bool has_prefix(const String2TensorStorage& tensor_storage_map, const std::string& prefix) {
        for (const auto& kv : tensor_storage_map) {
            if (kv.first.rfind(prefix, 0) == 0) {
                return true;
            }
        }
        return false;
    }

    static std::string resolve_prefix(const String2TensorStorage& tensor_storage_map, const std::string& base_prefix) {
        // LLM blocks already add ".model" in their submodule names. Pick a prefix that
        // results in "<prefix>.model.*" matching the weight file.
        std::string transformer_model_prefix = base_prefix + ".transformer.model";
        if (has_prefix(tensor_storage_map, transformer_model_prefix + ".")) {
            return base_prefix + ".transformer";
        }
        std::string transformer_prefix = base_prefix + ".transformer";
        if (has_prefix(tensor_storage_map, transformer_prefix + ".")) {
            return transformer_prefix;
        }
        std::string model_prefix = base_prefix + ".model";
        if (has_prefix(tensor_storage_map, model_prefix + ".")) {
            return base_prefix;
        }
        return base_prefix;
    }

    static float parse_qwen3_size(const std::string& name) {
        std::string s = name;
        if (s.rfind("qwen3_", 0) == 0) {
            s = s.substr(6);
        }
        if (!s.empty() && s.back() == 'b') {
            s.pop_back();
        }
        if (s.empty()) {
            return 0.f;
        }
        if (s.find('.') != std::string::npos) {
            return std::stof(s);
        }
        if (s.size() > 1 && s[0] == '0') {
            return std::stof("0." + s.substr(1));
        }
        return std::stof(s);
    }

    static std::vector<std::pair<float, std::string>> find_qwen3_variants(const String2TensorStorage& tensor_storage_map) {
        std::map<std::string, float> found;
        for (const auto& kv : tensor_storage_map) {
            if (kv.first.rfind("text_encoders.qwen3_", 0) != 0) {
                continue;
            }
            auto end = kv.first.find('.', strlen("text_encoders."));
            if (end == std::string::npos) {
                continue;
            }
            std::string model_name = kv.first.substr(strlen("text_encoders."), end - strlen("text_encoders."));
            if (found.find(model_name) == found.end()) {
                found[model_name] = parse_qwen3_size(model_name);
            }
        }
        std::vector<std::pair<float, std::string>> variants;
        variants.reserve(found.size());
        for (const auto& kv : found) {
            variants.emplace_back(kv.second, "text_encoders." + kv.first);
        }
        std::sort(variants.begin(), variants.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        return variants;
    }

    static bool has_qwen3_variant(const String2TensorStorage& tensor_storage_map, const std::string& name) {
        std::string prefix = "text_encoders." + name + ".";
        return has_prefix(tensor_storage_map, prefix);
    }

    static std::string resolve_qwen3_variant(const String2TensorStorage& tensor_storage_map, const std::string& name) {
        std::string prefix = "text_encoders." + name;
        return resolve_prefix(tensor_storage_map, prefix);
    }

    AceConditioner(ggml_backend_t backend,
                   bool offload_params_to_cpu,
                   const String2TensorStorage& tensor_storage_map = {}) {
        tokenizer = std::make_shared<LLM::Qwen3Tokenizer>();

        auto variants = find_qwen3_variants(tensor_storage_map);
        std::string base_prefix = "text_encoders.qwen3_06b";
        if (!variants.empty()) {
            base_prefix = variants.front().second;
        }

        base_llm_prefix = resolve_prefix(tensor_storage_map, base_prefix);
        base_llm = std::make_shared<LLM::LLMRunner>(LLM::LLMArch::QWEN3,
                                                    backend,
                                                    offload_params_to_cpu,
                                                    tensor_storage_map,
                                                    base_llm_prefix,
                                                    false);

        bool has_lm = has_prefix(tensor_storage_map, "text_encoders.llm.");
        if (has_lm) {
            lm_llm_prefix = resolve_prefix(tensor_storage_map, "text_encoders.llm");
        } else if (has_qwen3_variant(tensor_storage_map, "qwen3_2b")) {
            lm_llm_prefix = resolve_qwen3_variant(tensor_storage_map, "qwen3_2b");
        } else if (has_qwen3_variant(tensor_storage_map, "qwen3_4b")) {
            lm_llm_prefix = resolve_qwen3_variant(tensor_storage_map, "qwen3_4b");
        } else if (variants.size() > 1) {
            std::string lm_prefix = variants.back().second;
            if (lm_prefix != base_prefix) {
                lm_llm_prefix = resolve_prefix(tensor_storage_map, lm_prefix);
            }
        }

        if (!lm_llm_prefix.empty()) {
            lm_llm = std::make_shared<LLM::LLMRunner>(LLM::LLMArch::QWEN3,
                                                      backend,
                                                      offload_params_to_cpu,
                                                      tensor_storage_map,
                                                      lm_llm_prefix,
                                                      false);
        }
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors) override {
        if (base_llm) {
            base_llm->get_param_tensors(tensors, base_llm_prefix);
        }
        if (lm_llm) {
            lm_llm->get_param_tensors(tensors, lm_llm_prefix);
        }
    }

    void alloc_params_buffer() override {
        if (base_llm) {
            base_llm->alloc_params_buffer();
        }
        if (lm_llm) {
            lm_llm->alloc_params_buffer();
        }
    }

    void free_params_buffer() override {
        if (base_llm) {
            base_llm->free_params_buffer();
        }
        if (lm_llm) {
            lm_llm->free_params_buffer();
        }
    }

    size_t get_params_buffer_size() override {
        size_t size = 0;
        if (base_llm) {
            size += base_llm->get_params_buffer_size();
        }
        if (lm_llm) {
            size += lm_llm->get_params_buffer_size();
        }
        return size;
    }

    void set_flash_attention_enabled(bool enabled) override {
        if (base_llm) {
            base_llm->set_flash_attention_enabled(false);
        }
        if (lm_llm) {
            lm_llm->set_flash_attention_enabled(enabled);
        }
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override {
        if (base_llm) {
            base_llm->set_weight_adapter(adapter);
        }
        if (lm_llm) {
            lm_llm->set_weight_adapter(adapter);
        }
    }

    static std::string format_meta_cap(int bpm, int timesignature, const std::string& keyscale, int duration) {
        std::ostringstream oss;
        oss << "- bpm: " << bpm << "\n";
        oss << "- timesignature: " << timesignature << "\n";
        oss << "- keyscale: " << keyscale << "\n";
        oss << "- duration: " << duration << "\n";
        return oss.str();
    }

    static std::string format_meta_lm(int bpm, int timesignature, const std::string& keyscale, int duration) {
        std::ostringstream oss;
        oss << "bpm: " << bpm << "\n";
        oss << "duration: " << duration << "\n";
        oss << "keyscale: " << keyscale << "\n";
        oss << "timesignature: " << timesignature;
        return oss.str();
    }

        std::vector<float> compute_logits(int n_threads,
                                          LLM::LLMRunner* runner,
                                          const std::vector<int>& tokens,
                                          int pad_len = 0) {
        std::vector<float> logits;
        if (!runner) {
            return logits;
        }

        size_t vocab_size = static_cast<size_t>(runner->params.vocab_size);
        size_t logits_vocab_size = vocab_size;
        int64_t logits_start = runner->get_logits_range_start();
        int64_t logits_end = runner->get_logits_range_end();
        if (logits_end > logits_start) {
            logits_vocab_size = static_cast<size_t>(logits_end - logits_start);
        }
        if (logits_vocab_size == 0 || tokens.empty()) {
            return logits;
        }

        size_t n_tokens = tokens.size();
        size_t mask_bytes = n_tokens * n_tokens * sizeof(float);
        size_t mem_size = std::max<size_t>({logits_vocab_size * sizeof(float) * 2,
                                            8 * 1024 * 1024,
                                            mask_bytes + 1024 * 1024});
        struct ggml_init_params params;
        params.mem_size   = mem_size;
        params.mem_buffer = nullptr;
        params.no_alloc   = false;
        struct ggml_context* ctx = ggml_init(params);
        if (!ctx) {
            return logits;
        }

        ggml_tensor* input_ids = vector_to_ggml_tensor_i32(ctx, tokens);
        ggml_tensor* logits_tensor = nullptr;
        std::vector<std::pair<int, ggml_tensor*>> image_embeds;

        ggml_tensor* attention_mask = nullptr;
        {
            // Match Comfy: use finite negative mask values to avoid NaNs in some backends.
            constexpr float kMaskNeg = -65504.0f;
            int64_t n_tokens_i = static_cast<int64_t>(tokens.size());
            std::vector<float> attention_mask_vec(static_cast<size_t>(n_tokens_i) * static_cast<size_t>(n_tokens_i), 0.f);
            for (int64_t i0 = 0; i0 < n_tokens_i; ++i0) {
                for (int64_t i1 = 0; i1 < n_tokens_i; ++i1) {
                    float value = 0.f;
                    if (i0 < pad_len) {  // mask out pad tokens as keys (left padding)
                        value = kMaskNeg;
                    } else if (i0 > i1) {  // causal mask
                        value = kMaskNeg;
                    }
                    attention_mask_vec[i1 * n_tokens_i + i0] = value;
                }
            }
            attention_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_tokens_i, n_tokens_i);
            if (attention_mask->data != nullptr) {
                memcpy(attention_mask->data, attention_mask_vec.data(), attention_mask_vec.size() * sizeof(float));
            } else {
                ggml_ext_tensor_iter(attention_mask, [&](ggml_tensor* mask, int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
                    size_t idx = static_cast<size_t>(i1 * n_tokens_i + i0);
                    ggml_ext_tensor_set_f32(mask, attention_mask_vec[idx], i0, i1, i2, i3);
                });
            }
        }

        if (!runner->compute_logits(n_threads, input_ids, attention_mask, image_embeds, &logits_tensor, ctx) || logits_tensor == nullptr) {
            ggml_free(ctx);
            return logits;
        }

        size_t n = static_cast<size_t>(ggml_nelements(logits_tensor));
        logits.resize(std::min(n, logits_vocab_size));
        if (logits_tensor->type == GGML_TYPE_F32 && logits_tensor->buffer == nullptr) {
            memcpy(logits.data(), logits_tensor->data, logits.size() * sizeof(float));
        } else {
            for (size_t i = 0; i < logits.size(); ++i) {
                logits[i] = ggml_ext_tensor_get_f32(logits_tensor, i);
            }
        }
        ggml_free(ctx);
        return logits;
    }

    bool compute_logits_kv_cfg(int n_threads,
                               LLM::LLMRunner* runner,
                               const std::vector<int>& cond_tokens,
                               const std::vector<int>& uncond_tokens,
                               int64_t n_past,
                               int cond_pad_len,
                               int uncond_pad_len,
                               std::vector<float>& cond_logits,
                               std::vector<float>& uncond_logits) {
        cond_logits.clear();
        uncond_logits.clear();
        if (!runner || cond_tokens.empty() || cond_tokens.size() != uncond_tokens.size()) {
            return false;
        }

        size_t vocab_size = static_cast<size_t>(runner->params.vocab_size);
        size_t logits_vocab_size = vocab_size;
        int64_t logits_start = runner->get_logits_range_start();
        int64_t logits_end = runner->get_logits_range_end();
        if (logits_end > logits_start) {
            logits_vocab_size = static_cast<size_t>(logits_end - logits_start);
        }
        if (logits_vocab_size == 0) {
            return false;
        }

        const size_t n_tokens = cond_tokens.size();
        ggml_tensor* logits_tensor = nullptr;
        bool ok = false;

        if (n_tokens == 1 && n_past > 0) {
            size_t mem_size = std::max<size_t>(logits_vocab_size * 2 * sizeof(float), 2 * 1024 * 1024);
            struct ggml_init_params params;
            params.mem_size   = mem_size;
            params.mem_buffer = nullptr;
            params.no_alloc   = false;
            struct ggml_context* ctx = ggml_init(params);
            if (!ctx) {
                return false;
            }

            std::vector<int> ids = {cond_tokens[0], uncond_tokens[0]};
            std::vector<int> pad_lens = {cond_pad_len, uncond_pad_len};
            ok = runner->compute_logits_kv_decode_1token(n_threads, ids, n_past, pad_lens, &logits_tensor, ctx);
            if (!ok || logits_tensor == nullptr) {
                ggml_free(ctx);
                return false;
            }

            const size_t n_vocab = std::min<size_t>(logits_vocab_size, static_cast<size_t>(logits_tensor->ne[0]));
            cond_logits.resize(n_vocab);
            uncond_logits.resize(n_vocab);
            for (size_t i = 0; i < n_vocab; ++i) {
                cond_logits[i] = ggml_ext_tensor_get_f32(logits_tensor, static_cast<int64_t>(i), 0, 0, 0);
                uncond_logits[i] = ggml_ext_tensor_get_f32(logits_tensor, static_cast<int64_t>(i), 0, 1, 0);
            }

            ggml_free(ctx);
            return true;
        }

        const size_t n_kv = static_cast<size_t>(n_past) + n_tokens;
        size_t mask_bytes = n_kv * n_tokens * 2 * sizeof(float);
        size_t mem_size = std::max<size_t>({logits_vocab_size * 2 * sizeof(float),
                                            8 * 1024 * 1024,
                                            mask_bytes + 1024 * 1024});
        struct ggml_init_params params;
        params.mem_size   = mem_size;
        params.mem_buffer = nullptr;
        params.no_alloc   = false;
        struct ggml_context* ctx = ggml_init(params);
        if (!ctx) {
            return false;
        }

        ggml_tensor* input_ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_tokens, 2);
        {
            int32_t* ids = static_cast<int32_t*>(input_ids->data);
            for (size_t i = 0; i < n_tokens; ++i) {
                ids[i] = cond_tokens[i];
                ids[n_tokens + i] = uncond_tokens[i];
            }
        }

        constexpr float kMaskNeg = -65504.0f;
        ggml_tensor* attention_mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_kv, n_tokens, 2);
        std::vector<float> attention_mask_vec(n_kv * n_tokens * 2, 0.f);
        for (int b = 0; b < 2; ++b) {
            int pad_len = b == 0 ? cond_pad_len : uncond_pad_len;
            size_t offset = static_cast<size_t>(b) * n_kv * n_tokens;
            for (size_t q = 0; q < n_tokens; ++q) {
                int64_t abs_q = n_past + static_cast<int64_t>(q);
                for (size_t k = 0; k < n_kv; ++k) {
                    float value = 0.f;
                    if (static_cast<int>(k) < pad_len || static_cast<int64_t>(k) > abs_q) {
                        value = kMaskNeg;
                    }
                    attention_mask_vec[offset + q * n_kv + k] = value;
                }
            }
        }
        if (attention_mask->data != nullptr) {
            memcpy(attention_mask->data, attention_mask_vec.data(), attention_mask_vec.size() * sizeof(float));
        } else {
            for (int b = 0; b < 2; ++b) {
                size_t offset = static_cast<size_t>(b) * n_kv * n_tokens;
                for (size_t q = 0; q < n_tokens; ++q) {
                    for (size_t k = 0; k < n_kv; ++k) {
                        ggml_ext_tensor_set_f32(attention_mask,
                                                attention_mask_vec[offset + q * n_kv + k],
                                                static_cast<int64_t>(k),
                                                static_cast<int64_t>(q),
                                                b,
                                                0);
                    }
                }
            }
        }

        std::vector<std::pair<int, ggml_tensor*>> image_embeds;
        ok = runner->compute_logits_kv(n_threads,
                                       input_ids,
                                       attention_mask,
                                       image_embeds,
                                       n_past,
                                       &logits_tensor,
                                       ctx);
        if (!ok || logits_tensor == nullptr) {
            ggml_free(ctx);
            return false;
        }

        const size_t n_vocab = std::min<size_t>(logits_vocab_size, static_cast<size_t>(logits_tensor->ne[0]));
        cond_logits.resize(n_vocab);
        uncond_logits.resize(n_vocab);
        for (size_t i = 0; i < n_vocab; ++i) {
            cond_logits[i] = ggml_ext_tensor_get_f32(logits_tensor, static_cast<int64_t>(i), 0, 0, 0);
            uncond_logits[i] = ggml_ext_tensor_get_f32(logits_tensor, static_cast<int64_t>(i), 0, 1, 0);
        }

        ggml_free(ctx);
        return true;
    }

    int sample_from_logits(const std::vector<float>& logits,
                           float temperature,
                           float top_p,
                           std::mt19937_64& rng) {
        if (logits.empty()) {
            return -1;
        }

        int vocab_size = static_cast<int>(logits.size());
        if (temperature <= 0.f) {
            int best = 0;
            float best_val = logits[0];
            for (int i = 1; i < vocab_size; ++i) {
                if (logits[i] > best_val) {
                    best_val = logits[i];
                    best = i;
                }
            }
            return best;
        }

        float max_logit = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < vocab_size; ++i) {
            if (logits[i] > max_logit) {
                max_logit = logits[i];
            }
        }

        std::vector<float> probs(vocab_size, 0.f);
        double sum = 0.0;
        for (int i = 0; i < vocab_size; ++i) {
            float val = (logits[i] - max_logit) / temperature;
            if (std::isinf(logits[i]) && logits[i] < 0) {
                probs[i] = 0.f;
                continue;
            }
            float p = std::exp(val);
            probs[i] = p;
            sum += p;
        }

        if (sum <= 0.0) {
            int best = 0;
            float best_val = logits[0];
            for (int i = 1; i < vocab_size; ++i) {
                if (logits[i] > best_val) {
                    best_val = logits[i];
                    best = i;
                }
            }
            return best;
        }

        if (top_p < 1.0f) {
            std::vector<int> indices(vocab_size);
            for (int i = 0; i < vocab_size; ++i) {
                indices[i] = i;
            }
            std::sort(indices.begin(), indices.end(), [&](int a, int b) { return probs[a] > probs[b]; });

            double cumulative = 0.0;
            std::vector<char> keep(vocab_size, 0);
            for (int idx : indices) {
                cumulative += probs[idx] / sum;
                keep[idx] = 1;
                if (cumulative >= top_p) {
                    break;
                }
            }

            double new_sum = 0.0;
            for (int i = 0; i < vocab_size; ++i) {
                if (!keep[i]) {
                    probs[i] = 0.f;
                } else {
                    new_sum += probs[i];
                }
            }
            sum = new_sum > 0.0 ? new_sum : sum;
        }

        std::uniform_real_distribution<double> dist(0.0, sum);
        double r = dist(rng);
        double acc = 0.0;
        for (int i = 0; i < vocab_size; ++i) {
            acc += probs[i];
            if (acc >= r) {
                return i;
            }
        }
        return vocab_size - 1;
    }

    std::shared_ptr<std::vector<int>> generate_audio_codes(int n_threads,
                                                           const std::string& lm_prompt,
                                                           const std::string& lm_prompt_negative,
                                                           int min_tokens,
                                                           int lm_seed) {
        const int audio_start_id = 151669;
        const float cfg_scale = 2.0f;
        const float temperature = 0.85f;
        const float top_p = 0.9f;

        std::shared_ptr<std::vector<int>> codes = std::make_shared<std::vector<int>>();
        auto runner = lm_llm ? lm_llm.get() : base_llm.get();
        if (!runner || !tokenizer) {
            return codes;
        }

        int64_t t0 = ggml_time_ms();
        LOG_INFO("ACE LM: generating %d audio tokens (seed=%d)", min_tokens, lm_seed);

        std::vector<int> cond_tokens = tokenizer->tokenize(lm_prompt, nullptr);
        std::vector<int> uncond_tokens = tokenizer->tokenize(lm_prompt_negative, nullptr);

        const int pad_token_id = 151643;
        int pos_pad = 0;
        int neg_pad = 0;
        if (uncond_tokens.size() < cond_tokens.size()) {
            neg_pad = static_cast<int>(cond_tokens.size() - uncond_tokens.size());
            uncond_tokens.insert(uncond_tokens.begin(), neg_pad, pad_token_id);
        } else if (cond_tokens.size() < uncond_tokens.size()) {
            pos_pad = static_cast<int>(uncond_tokens.size() - cond_tokens.size());
            cond_tokens.insert(cond_tokens.begin(), pos_pad, pad_token_id);
        }

        const int num_tokens_to_generate = min_tokens;
        std::mt19937_64 rng(static_cast<uint64_t>(lm_seed));
        bool use_kv_cache = true;
        int64_t n_past = 0;
        std::vector<int> cond_tokens_full = cond_tokens;
        std::vector<int> uncond_tokens_full = uncond_tokens;
        std::vector<int> cond_step_tokens = cond_tokens;
        std::vector<int> uncond_step_tokens = uncond_tokens;

        const int64_t full_vocab_end = runner->params.vocab_size;
        runner->set_logits_range(audio_start_id, full_vocab_end);
        const int logits_id_offset = static_cast<int>(runner->get_logits_range_start());

        runner->reset_kv_cache();
        if (use_kv_cache) {
            int64_t kv_capacity = static_cast<int64_t>(cond_tokens.size()) + static_cast<int64_t>(num_tokens_to_generate);
            if (!runner->prepare_kv_cache(kv_capacity, 2)) {
                use_kv_cache = false;
                LOG_WARN("ACE LM: KV-cache allocation failed, falling back to full-sequence logits");
            }
        }

        for (int step = 0; step < num_tokens_to_generate; ++step) {
            std::vector<float> cond_logits;
            std::vector<float> uncond_logits;
            if (use_kv_cache) {
                bool ok = compute_logits_kv_cfg(n_threads,
                                                runner,
                                                cond_step_tokens,
                                                uncond_step_tokens,
                                                n_past,
                                                pos_pad,
                                                neg_pad,
                                                cond_logits,
                                                uncond_logits);
                if (ok) {
                    n_past += static_cast<int64_t>(cond_step_tokens.size());
                }
                if (!ok) {
                    use_kv_cache = false;
                    runner->reset_kv_cache();
                    LOG_WARN("ACE LM: KV-cache decode unavailable, falling back to full-sequence logits");
                }
            }
            if (!use_kv_cache) {
                cond_logits = compute_logits(n_threads, runner, cond_tokens_full, pos_pad);
                uncond_logits = compute_logits(n_threads, runner, uncond_tokens_full, neg_pad);
            }
            if (cond_logits.empty() || uncond_logits.empty() || cond_logits.size() != uncond_logits.size()) {
                break;
            }
            std::vector<float> cfg_logits(cond_logits.size(), 0.f);
            for (size_t i = 0; i < cond_logits.size(); ++i) {
                cfg_logits[i] = uncond_logits[i] + cfg_scale * (cond_logits[i] - uncond_logits[i]);
            }

            const int mask_upto = std::max(0, audio_start_id - logits_id_offset);
            for (int i = 0; i < mask_upto && i < static_cast<int>(cfg_logits.size()); ++i) {
                cfg_logits[i] = -std::numeric_limits<float>::infinity();
            }

            if (top_p < 1.0f) {
                std::vector<int> indices(cfg_logits.size());
                for (size_t i = 0; i < indices.size(); ++i) {
                    indices[i] = static_cast<int>(i);
                }
                std::sort(indices.begin(), indices.end(),
                          [&](int a, int b) { return cfg_logits[a] > cfg_logits[b]; });

                float max_logit = -std::numeric_limits<float>::infinity();
                for (float v : cfg_logits) {
                    if (v > max_logit) {
                        max_logit = v;
                    }
                }
                double sum = 0.0;
                std::vector<double> sorted_probs(indices.size(), 0.0);
                for (size_t i = 0; i < indices.size(); ++i) {
                    float v = cfg_logits[indices[i]];
                    if (std::isinf(v) && v < 0) {
                        sorted_probs[i] = 0.0;
                        continue;
                    }
                    double p = std::exp((double)(v - max_logit));
                    sorted_probs[i] = p;
                    sum += p;
                }
                if (sum > 0.0) {
                    double cumulative = 0.0;
                    std::vector<char> remove(indices.size(), 0);
                    for (size_t i = 0; i < indices.size(); ++i) {
                        cumulative += sorted_probs[i] / sum;
                        if (cumulative > top_p) {
                            remove[i] = 1;
                        }
                    }
                    for (int i = static_cast<int>(remove.size()) - 1; i >= 1; --i) {
                        remove[i] = remove[i - 1];
                    }
                    if (!remove.empty()) {
                        remove[0] = 0;
                    }
                    for (size_t i = 0; i < indices.size(); ++i) {
                        if (remove[i]) {
                            cfg_logits[indices[i]] = -std::numeric_limits<float>::infinity();
                        }
                    }
                }
            }

            int next_token = logits_id_offset + sample_from_logits(cfg_logits, temperature, 1.0f, rng);
            if (next_token < audio_start_id) {
                next_token = audio_start_id;
            }

            codes->push_back(next_token - audio_start_id);
            cond_tokens_full.push_back(next_token);
            uncond_tokens_full.push_back(next_token);
            cond_step_tokens.assign(1, next_token);
            uncond_step_tokens.assign(1, next_token);

            if ((step + 1) % 10 == 0 || step + 1 == num_tokens_to_generate) {
                int64_t t = ggml_time_ms();
                LOG_INFO("ACE LM: generated %d/%d tokens (elapsed %.2fs)", step + 1, num_tokens_to_generate, (t - t0) / 1000.0);
            }
        }
        runner->reset_kv_cache();
        runner->set_logits_range(0, full_vocab_end);
        int64_t t1 = ggml_time_ms();
        LOG_INFO("ACE LM: audio token generation done in %.2fs", (t1 - t0) / 1000.0);

        return codes;
    }

    SDCondition get_learned_condition(ggml_context* work_ctx,
                                      int n_threads,
                                      const ConditionerParams& conditioner_params) override {
        SDCondition cond;

        std::string caption = conditioner_params.text;
        std::string lyrics  = conditioner_params.lyrics;
        std::string language = conditioner_params.language;
        std::string keyscale = conditioner_params.keyscale;
        int bpm = static_cast<int>(std::round(conditioner_params.bpm));
        int timesignature = conditioner_params.timesignature;
        int duration = std::max(1, static_cast<int>(std::ceil(conditioner_params.duration)));

        std::string meta_lm = format_meta_lm(bpm, timesignature, keyscale, duration);
        std::string meta_cap = format_meta_cap(bpm, timesignature, keyscale, duration);

        std::string lm_template = "<|im_start|>system\n# Instruction\nGenerate audio semantic tokens based on the given conditions:\n\n<|im_end|>\n<|im_start|>user\n# Caption\n";
        std::string lm_prompt = lm_template + caption + "\n" + lyrics + "\n<|im_end|>\n<|im_start|>assistant\n<think>\n" + meta_lm + "\n</think>\n\n<|im_end|>\n";
        std::string lm_prompt_negative = lm_template + caption + "\n" + lyrics + "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<|im_end|>\n";

        std::string lyric_prompt = "# Languages\n" + language + "\n\n# Lyric" + lyrics + "<|endoftext|><|endoftext|>";
        std::string qwen_prompt = "# Instruction\nGenerate audio semantic tokens based on the given conditions:\n\n# Caption\n" + caption + "# Metas\n" + meta_cap + "<|endoftext|>\n<|endoftext|>";

        auto tokens_and_weights = tokenize_with_weights(qwen_prompt);
        auto tokens = tokens_and_weights.first;
        auto weights = tokens_and_weights.second;
        auto lyric_tokens = tokenizer->tokenize(lyric_prompt, nullptr);

        ggml_tensor* context = nullptr;
        ggml_tensor* lyric_embed = nullptr;
        if (base_llm) {
            std::set<int> out_layers;
            auto input_ids = vector_to_ggml_tensor_i32(work_ctx, tokens);
            std::vector<std::pair<int, ggml_tensor*>> image_embeds;
            base_llm->compute(n_threads, input_ids, nullptr, image_embeds, out_layers, &context, work_ctx);
            if (context && !weights.empty()) {
                float original_mean = ggml_ext_tensor_mean(context);
                int64_t n_tokens = context->ne[1];
                int64_t weight_len = static_cast<int64_t>(weights.size());
                int64_t limit = std::min(n_tokens, weight_len);
                for (int i2 = 0; i2 < context->ne[2]; ++i2) {
                    for (int i1 = 0; i1 < limit; ++i1) {
                        for (int i0 = 0; i0 < context->ne[0]; ++i0) {
                            float value = ggml_ext_tensor_get_f32(context, i0, i1, i2);
                            value *= weights[i1];
                            ggml_ext_tensor_set_f32(context, value, i0, i1, i2);
                        }
                    }
                }
                float new_mean = ggml_ext_tensor_mean(context);
                if (new_mean != 0.f) {
                    ggml_ext_tensor_scale_inplace(context, (original_mean / new_mean));
                }
            }

            std::set<int> lyric_layers = {0};
            auto lyric_ids = vector_to_ggml_tensor_i32(work_ctx, lyric_tokens);
            base_llm->compute(n_threads, lyric_ids, nullptr, image_embeds, lyric_layers, &lyric_embed, work_ctx);
        }

        static const float kReferAudioVec[64] = {
            -1.3672e-01f, -1.5820e-01f,  5.8594e-01f, -5.7422e-01f,  3.0273e-02f,
             2.7930e-01f, -2.5940e-03f, -2.0703e-01f, -1.6113e-01f, -1.4746e-01f,
            -2.7710e-02f, -1.8066e-01f, -2.9688e-01f,  1.6016e+00f, -2.6719e+00f,
             7.7734e-01f, -1.3516e+00f, -1.9434e-01f, -7.1289e-02f, -5.0938e+00f,
             2.4316e-01f,  4.7266e-01f,  4.6387e-02f, -6.6406e-01f, -2.1973e-01f,
            -6.7578e-01f, -1.5723e-01f,  9.5312e-01f, -2.0020e-01f, -1.7109e+00f,
             5.8984e-01f, -5.7422e-01f,  5.1562e-01f,  2.8320e-01f,  1.4551e-01f,
            -1.8750e-01f, -5.9814e-02f,  3.6719e-01f, -1.0059e-01f, -1.5723e-01f,
             2.0605e-01f, -4.3359e-01f, -8.2812e-01f,  4.5654e-02f, -6.6016e-01f,
             1.4844e-01f,  9.4727e-02f,  3.8477e-01f, -1.2578e+00f, -3.3203e-01f,
            -8.5547e-01f,  4.3359e-01f,  4.2383e-01f, -8.9453e-01f, -5.0391e-01f,
            -5.6152e-02f, -2.9219e+00f, -2.4658e-02f,  5.0391e-01f,  9.8438e-01f,
             7.2754e-02f, -2.1582e-01f,  6.3672e-01f,  1.0000e+00f
        };

        ggml_tensor* refer_audio = ggml_new_tensor_3d(work_ctx, GGML_TYPE_F32, 64, 750, 1);
        for (int64_t t = 0; t < refer_audio->ne[1]; ++t) {
            for (int64_t c = 0; c < refer_audio->ne[0]; ++c) {
                ggml_ext_tensor_set_f32(refer_audio, kReferAudioVec[c], c, t, 0);
            }
        }

        int min_tokens = std::max(1, duration * 5);
        std::shared_ptr<std::vector<int>> audio_codes =
            generate_audio_codes(n_threads, lm_prompt, lm_prompt_negative, min_tokens, conditioner_params.lm_seed);

        cond.c_crossattn = context;
        cond.c_lyrics    = lyric_embed;
        cond.refer_audio = refer_audio;
        cond.audio_codes = audio_codes;
        return cond;
    }

private:
    std::pair<std::vector<int>, std::vector<float>> tokenize_with_weights(const std::string& text) {
        auto parsed_attention = parse_prompt_attention(text);
        std::vector<int> tokens;
        std::vector<float> weights;
        for (const auto& item : parsed_attention) {
            const std::string& curr_text = item.first;
            float curr_weight = item.second;
            auto curr_tokens = tokenizer->tokenize(curr_text, nullptr);
            tokens.insert(tokens.end(), curr_tokens.begin(), curr_tokens.end());
            weights.insert(weights.end(), curr_tokens.size(), curr_weight);
        }
        tokenizer->pad_tokens(tokens, weights, 0, false);
        return {tokens, weights};
    }
};

#endif
