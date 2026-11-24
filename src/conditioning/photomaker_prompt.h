#ifndef __SD_PHOTOMAKER_PROMPT_H__
#define __SD_PHOTOMAKER_PROMPT_H__

#include <algorithm>
#include <tuple>
#include <utility>

#include "conditioning/conditioner.hpp"

namespace sd::photomaker {
    inline std::tuple<std::vector<int>, std::vector<float>, std::vector<bool>>
    tokenize_photomaker_trigger(FrozenCLIPEmbedderWithCustomWords& clip_conditioner,
                                const std::string& text,
                                int trigger_token_count,
                                int32_t image_token) {
        auto tokens_and_weights           = clip_conditioner.tokenize(text);
        std::vector<int> source_tokens    = std::move(tokens_and_weights.first);
        std::vector<float> source_weights = std::move(tokens_and_weights.second);
        if (source_tokens.empty()) {
            return {};
        }

        if (!source_tokens.empty() && source_tokens.front() == clip_conditioner.tokenizer->BOS_TOKEN_ID) {
            source_tokens.erase(source_tokens.begin());
            source_weights.erase(source_weights.begin());
        }
        if (!source_tokens.empty() && source_tokens.back() == clip_conditioner.tokenizer->EOS_TOKEN_ID) {
            source_tokens.pop_back();
            source_weights.pop_back();
        }

        std::vector<int> tokens;
        std::vector<float> weights;
        int32_t class_idx = -1;
        for (size_t i = 0; i < source_tokens.size(); i++) {
            int token = source_tokens[i];
            if (token == image_token) {
                if (!tokens.empty()) {
                    class_idx          = static_cast<int32_t>(tokens.size()) - 1;
                    int class_token    = tokens.back();
                    float class_weight = weights.back();
                    for (int j = 1; j < trigger_token_count; j++) {
                        tokens.push_back(class_token);
                        weights.push_back(class_weight);
                    }
                }
                continue;
            }
            tokens.push_back(token);
            weights.push_back(source_weights[i]);
        }

        clip_conditioner.tokenizer->pad_tokens(tokens,
                                               &weights,
                                               nullptr,
                                               clip_conditioner.text_model->model.n_token,
                                               clip_conditioner.text_model->model.n_token,
                                               true);
        std::vector<bool> class_token_mask;
        for (int i = 0; i < tokens.size(); i++) {
            class_token_mask.push_back(class_idx >= 0 && class_idx + 1 <= i && i < class_idx + 1 + trigger_token_count);
        }

        return std::make_tuple(tokens, weights, class_token_mask);
    }

    inline std::tuple<SDCondition, std::vector<bool>>
    encode_prompt(FrozenCLIPEmbedderWithCustomWords& clip_conditioner,
                  int n_threads,
                  const ConditionerParams& conditioner_params,
                  const std::string& trigger_word,
                  int trigger_token_count) {
        std::vector<int> image_tokens;
        if (!clip_conditioner.convert_token_to_id(trigger_word, image_tokens)) {
            return {};
        }
        if (image_tokens.size() != 1) {
            LOG_ERROR("PhotoMaker trigger word must encode to one token");
            return {};
        }
        auto tokens_and_weights         = tokenize_photomaker_trigger(clip_conditioner,
                                                                      conditioner_params.text,
                                                                      trigger_token_count,
                                                                      image_tokens[0]);
        std::vector<int>& tokens        = std::get<0>(tokens_and_weights);
        std::vector<float>& weights     = std::get<1>(tokens_and_weights);
        std::vector<bool>& trigger_mask = std::get<2>(tokens_and_weights);
        if (tokens.empty()) {
            return {};
        }
        auto cond = clip_conditioner.get_learned_condition_common(n_threads,
                                                                  tokens,
                                                                  weights,
                                                                  conditioner_params.clip_skip,
                                                                  conditioner_params.width,
                                                                  conditioner_params.height,
                                                                  conditioner_params.zero_out_masked);
        return std::make_tuple(std::move(cond), trigger_mask);
    }

    inline bool remove_trigger(FrozenCLIPEmbedderWithCustomWords& clip_conditioner,
                               const std::string& prompt,
                               const std::string& trigger_word,
                               std::string& result) {
        std::vector<int> image_tokens;
        if (!clip_conditioner.convert_token_to_id(trigger_word, image_tokens)) {
            return false;
        }
        if (image_tokens.size() != 1) {
            LOG_ERROR("PhotoMaker trigger word must encode to one token");
            return false;
        }
        auto tokens_and_weights  = clip_conditioner.tokenize(prompt);
        std::vector<int>& tokens = tokens_and_weights.first;
        if (tokens.empty()) {
            return false;
        }
        auto it = std::find(tokens.begin(), tokens.end(), image_tokens[0]);
        if (it == tokens.end()) {
            LOG_ERROR("PhotoMaker trigger word was not found in tokenized prompt");
            return false;
        }
        tokens.erase(it);
        return clip_conditioner.decode(tokens, result);
    }
}  // namespace sd::photomaker

#endif  // __SD_PHOTOMAKER_PROMPT_H__
