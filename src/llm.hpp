#ifndef __LLM_HPP__
#define __LLM_HPP__

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "clip.hpp"
#include "ggml_extend.hpp"
#include "json.hpp"
#include "rope.hpp"
#include "tokenize_util.h"
#include "vocab/vocab.h"

namespace LLM {
    constexpr int LLM_GRAPH_SIZE = 10240;

    class BPETokenizer {
    protected:
        std::map<int, std::u32string> byte_encoder;
        std::map<std::u32string, int> byte_decoder;
        std::map<std::u32string, int> encoder;
        std::map<int, std::u32string> decoder;
        std::map<std::pair<std::u32string, std::u32string>, int> bpe_ranks;
        std::regex pat;
        int encoder_len;
        int bpe_len;

        std::string UNK_TOKEN;
        std::string BOS_TOKEN;
        std::string EOS_TOKEN;
        std::string PAD_TOKEN;

        int UNK_TOKEN_ID;
        int BOS_TOKEN_ID;
        int EOS_TOKEN_ID;
        int PAD_TOKEN_ID;

        std::vector<std::string> special_tokens;

        bool add_bos_token = false;

    protected:
        static std::string strip(const std::string& str) {
            std::string::size_type start = str.find_first_not_of(" \t\n\r\v\f");
            std::string::size_type end   = str.find_last_not_of(" \t\n\r\v\f");

            if (start == std::string::npos) {
                // String contains only whitespace characters
                return "";
            }

            return str.substr(start, end - start + 1);
        }

        static std::string whitespace_clean(std::string text) {
            text = std::regex_replace(text, std::regex(R"(\s+)"), " ");
            text = strip(text);
            return text;
        }

        static std::set<std::pair<std::u32string, std::u32string>> get_pairs(const std::vector<std::u32string>& subwords) {
            std::set<std::pair<std::u32string, std::u32string>> pairs;
            if (subwords.size() == 0) {
                return pairs;
            }
            std::u32string prev_subword = subwords[0];
            for (int i = 1; i < subwords.size(); i++) {
                std::u32string subword = subwords[i];
                std::pair<std::u32string, std::u32string> pair(prev_subword, subword);
                pairs.insert(pair);
                prev_subword = subword;
            }
            return pairs;
        }

        bool is_special_token(const std::string& token) {
            for (auto& special_token : special_tokens) {
                if (special_token == token) {
                    return true;
                }
            }
            return false;
        }

    public:
        BPETokenizer() = default;

        std::u32string bpe(const std::u32string& token) {
            std::vector<std::u32string> word;

            for (int i = 0; i < token.size(); i++) {
                word.emplace_back(1, token[i]);
            }

            std::set<std::pair<std::u32string, std::u32string>> pairs = get_pairs(word);

            if (pairs.empty()) {
                return token;
            }

            while (true) {
                auto min_pair_iter = std::min_element(pairs.begin(),
                                                      pairs.end(),
                                                      [&](const std::pair<std::u32string, std::u32string>& a,
                                                          const std::pair<std::u32string, std::u32string>& b) {
                                                          if (bpe_ranks.find(a) == bpe_ranks.end()) {
                                                              return false;
                                                          } else if (bpe_ranks.find(b) == bpe_ranks.end()) {
                                                              return true;
                                                          }
                                                          return bpe_ranks.at(a) < bpe_ranks.at(b);
                                                      });

                const std::pair<std::u32string, std::u32string>& bigram = *min_pair_iter;

                if (bpe_ranks.find(bigram) == bpe_ranks.end()) {
                    break;
                }

                std::u32string first  = bigram.first;
                std::u32string second = bigram.second;
                std::vector<std::u32string> new_word;
                int32_t i = 0;

                while (i < word.size()) {
                    auto it = std::find(word.begin() + i, word.end(), first);
                    if (it == word.end()) {
                        new_word.insert(new_word.end(), word.begin() + i, word.end());
                        break;
                    }
                    new_word.insert(new_word.end(), word.begin() + i, it);
                    i = static_cast<int32_t>(std::distance(word.begin(), it));

                    if (word[i] == first && i < static_cast<int32_t>(word.size()) - 1 && word[i + 1] == second) {
                        new_word.push_back(first + second);
                        i += 2;
                    } else {
                        new_word.push_back(word[i]);
                        i += 1;
                    }
                }

                word = new_word;

                if (word.size() == 1) {
                    break;
                }
                pairs = get_pairs(word);
            }

            std::u32string result;
            for (int i = 0; i < word.size(); i++) {
                result += word[i];
                if (i != word.size() - 1) {
                    result += utf8_to_utf32(" ");
                }
            }

            return result;
        }

        std::vector<int> tokenize(std::string text,
                                  on_new_token_cb_t on_new_token_cb = nullptr,
                                  size_t max_length                 = 0,
                                  bool padding                      = false) {
            std::vector<int32_t> tokens = encode(text, on_new_token_cb);

            if (max_length > 0) {
                if (tokens.size() < max_length) {
                    tokens.resize(max_length);
                } else {
                    if (padding) {
                        tokens.insert(tokens.end(), max_length - tokens.size(), PAD_TOKEN_ID);
                    }
                }
            }

            return tokens;
        }

        void pad_tokens(std::vector<int>& tokens,
                        std::vector<float>& weights,
                        size_t max_length = 0,
                        bool padding      = false) {
            if (add_bos_token) {
                tokens.insert(tokens.begin(), BOS_TOKEN_ID);
            }
            if (max_length > 0 && padding) {
                size_t n = static_cast<size_t>(std::ceil(tokens.size() * 1.f / max_length));
                if (n == 0) {
                    n = 1;
                }
                size_t length = max_length * n;
                LOG_DEBUG("token length: %llu", length);
                tokens.insert(tokens.end(), length - tokens.size(), PAD_TOKEN_ID);
                weights.insert(weights.end(), length - weights.size(), 1.f);
            }
        }

        std::vector<int> encode(std::string text, on_new_token_cb_t on_new_token_cb = nullptr) {
            std::string original_text = text;
            std::vector<int32_t> bpe_tokens;
            std::vector<std::string> token_strs;

            auto splited_texts = split_with_special_tokens(text, special_tokens);

            for (auto& splited_text : splited_texts) {
                if (is_special_token(splited_text)) {
                    bpe_tokens.push_back(encoder[utf8_to_utf32(splited_text)]);
                    token_strs.push_back(splited_text);
                    continue;
                }
                auto tokens = token_split(splited_text);
                for (auto& token : tokens) {
                    if (on_new_token_cb != nullptr) {
                        bool skip = on_new_token_cb(token, bpe_tokens);
                        if (skip) {
                            continue;
                        }
                    }

                    std::string token_str = token;
                    std::u32string utf32_token;
                    for (int i = 0; i < token_str.length(); i++) {
                        unsigned char b = token_str[i];
                        utf32_token += byte_encoder[b];
                    }
                    auto bpe_strs = bpe(utf32_token);
                    size_t start  = 0;
                    size_t pos;
                    while ((pos = bpe_strs.find(' ', start)) != std::u32string::npos) {
                        auto bpe_str = bpe_strs.substr(start, pos - start);
                        bpe_tokens.push_back(encoder[bpe_str]);
                        token_strs.push_back(utf32_to_utf8(bpe_str));

                        start = pos + 1;
                    }
                    auto bpe_str = bpe_strs.substr(start, bpe_strs.size() - start);
                    bpe_tokens.push_back(encoder[bpe_str]);
                    token_strs.push_back(utf32_to_utf8(bpe_str));
                }
            }

            std::stringstream ss;
            ss << "[";
            for (auto token : token_strs) {
                ss << "\"" << token << "\", ";
            }
            ss << "]";
            LOG_DEBUG("split prompt \"%s\" to tokens %s", original_text.c_str(), ss.str().c_str());
            // printf("split prompt \"%s\" to tokens %s \n", original_text.c_str(), ss.str().c_str());
            return bpe_tokens;
        }
    };

    class Qwen2Tokenizer : public BPETokenizer {
    protected:
        void load_from_merges(const std::string& merges_utf8_str) {
            auto byte_unicode_pairs = bytes_to_unicode();
            // printf("byte_unicode_pairs have %lu pairs \n", byte_unicode_pairs.size());
            byte_encoder = std::map<int, std::u32string>(byte_unicode_pairs.begin(), byte_unicode_pairs.end());
            for (auto& pair : byte_unicode_pairs) {
                byte_decoder[pair.second] = pair.first;
            }
            // for (auto & pair: byte_unicode_pairs) {
            //     std::cout << pair.first << ": " << pair.second << std::endl;
            // }
            std::vector<std::u32string> merges;
            size_t start = 0;
            size_t pos;
            std::u32string merges_utf32_str = utf8_to_utf32(merges_utf8_str);
            while ((pos = merges_utf32_str.find('\n', start)) != std::string::npos) {
                merges.push_back(merges_utf32_str.substr(start, pos - start));
                start = pos + 1;
            }
            LOG_DEBUG("merges size %llu", merges.size());
            merges = std::vector<std::u32string>(merges.begin(), merges.end());
            std::vector<std::pair<std::u32string, std::u32string>> merge_pairs;
            // int print_num = 10;
            for (const auto& merge : merges) {
                size_t space_pos = merge.find(' ');
                merge_pairs.emplace_back(merge.substr(0, space_pos), merge.substr(space_pos + 1));
                // if (print_num > 0) {
                //     print_num--;
                //     printf("%s :: %s | %s \n", utf32_to_utf8(merge).c_str(), utf32_to_utf8(merge.substr(0, space_pos)).c_str(),
                //                     utf32_to_utf8(merge.substr(space_pos + 1)).c_str());
                // }
            }

            std::vector<std::u32string> tokens;
            for (const auto& pair : byte_unicode_pairs) {
                tokens.push_back(pair.second);
            }
            for (const auto& merge : merge_pairs) {
                tokens.push_back(merge.first + merge.second);
            }
            for (auto& special_token : special_tokens) {
                tokens.push_back(utf8_to_utf32(special_token));
            }

            int i = 0;
            for (const auto& token : tokens) {
                encoder[token] = i;
                decoder[i]     = token;
                i++;
            }
            encoder_len = i;
            LOG_DEBUG("vocab size: %d", encoder_len);

            int rank = 0;
            for (const auto& merge : merge_pairs) {
                bpe_ranks[merge] = rank++;
            }
            bpe_len = rank;
        };

    public:
        explicit Qwen2Tokenizer(const std::string& merges_utf8_str = "") {
            UNK_TOKEN = "<|endoftext|>";
            EOS_TOKEN = "<|endoftext|>";
            PAD_TOKEN = "<|endoftext|>";

            UNK_TOKEN_ID = 151643;
            EOS_TOKEN_ID = 151643;
            PAD_TOKEN_ID = 151643;

            special_tokens = {
                "<|endoftext|>",
                "<|im_start|>",
                "<|im_end|>",
                "<|object_ref_start|>",
                "<|object_ref_end|>",
                "<|box_start|>",
                "<|box_end|>",
                "<|quad_start|>",
                "<|quad_end|>",
                "<|vision_start|>",
                "<|vision_end|>",
                "<|vision_pad|>",
                "<|image_pad|>",
                "<|video_pad|>",
                "<tool_call>",
                "</tool_call>",
                "<|fim_prefix|>",
                "<|fim_middle|>",
                "<|fim_suffix|>",
                "<|fim_pad|>",
                "<|repo_name|>",
                "<|file_sep|>",
                "<tool_response>",
                "</tool_response>",
                "<think>",
                "</think>",
            };

            if (merges_utf8_str.size() > 0) {
                load_from_merges(merges_utf8_str);
            } else {
                load_from_merges(load_qwen2_merges());
            }
        }
    };

    class Qwen3Tokenizer : public BPETokenizer {
    protected:
        static std::string read_file_to_string(const std::string& path) {
            std::ifstream ifs(path, std::ios::binary);
            if (!ifs.good()) {
                return "";
            }
            std::ostringstream oss;
            oss << ifs.rdbuf();
            return oss.str();
        }

        void load_from_merges(const std::string& merges_utf8_str) {
            auto byte_unicode_pairs = bytes_to_unicode();
            byte_encoder            = std::map<int, std::u32string>(byte_unicode_pairs.begin(), byte_unicode_pairs.end());
            for (auto& pair : byte_unicode_pairs) {
                byte_decoder[pair.second] = pair.first;
            }
            std::vector<std::u32string> merges;
            size_t start = 0;
            size_t pos;
            std::u32string merges_utf32_str = utf8_to_utf32(merges_utf8_str);
            while ((pos = merges_utf32_str.find('\n', start)) != std::string::npos) {
                merges.push_back(merges_utf32_str.substr(start, pos - start));
                start = pos + 1;
            }
            LOG_DEBUG("merges size %llu", merges.size());
            std::vector<std::pair<std::u32string, std::u32string>> merge_pairs;
            merge_pairs.reserve(merges.size());
            for (const auto& merge : merges) {
                size_t space_pos = merge.find(' ');
                if (space_pos == std::u32string::npos) {
                    continue;
                }
                merge_pairs.emplace_back(merge.substr(0, space_pos), merge.substr(space_pos + 1));
            }

            std::vector<std::u32string> tokens;
            for (const auto& pair : byte_unicode_pairs) {
                tokens.push_back(pair.second);
            }
            for (const auto& merge : merge_pairs) {
                tokens.push_back(merge.first + merge.second);
            }
            for (auto& special_token : special_tokens) {
                tokens.push_back(utf8_to_utf32(special_token));
            }

            int i = 0;
            for (const auto& token : tokens) {
                encoder[token] = i;
                decoder[i]     = token;
                i++;
            }
            encoder_len = i;
            LOG_DEBUG("vocab size: %d", encoder_len);

            int rank = 0;
            for (const auto& merge : merge_pairs) {
                bpe_ranks[merge] = rank++;
            }
            bpe_len = rank;
        }

        void load_from_tokenizer_json(const std::string& json_str) {
            nlohmann::json tok;
            try {
                tok = nlohmann::json::parse(json_str);
            } catch (const nlohmann::json::parse_error&) {
                GGML_ABORT("invalid qwen3 tokenizer json");
            }

            if (!tok.contains("model") || !tok["model"].contains("vocab") || !tok["model"].contains("merges")) {
                GGML_ABORT("qwen3 tokenizer json missing vocab/merges");
            }

            auto vocab = tok["model"]["vocab"];
            int max_id = -1;
            for (auto it = vocab.begin(); it != vocab.end(); ++it) {
                const std::string token = it.key();
                int id                  = it.value();
                std::u32string token_u  = utf8_to_utf32(token);
                encoder[token_u]        = id;
                decoder[id]             = token_u;
                max_id                  = std::max(max_id, id);
            }

            special_tokens.clear();
            if (tok.contains("added_tokens")) {
                for (auto& item : tok["added_tokens"]) {
                    if (!item.contains("content") || !item.contains("id")) {
                        continue;
                    }
                    std::string content = item["content"];
                    int id              = item["id"];
                    std::u32string u    = utf8_to_utf32(content);
                    encoder[u]          = id;
                    decoder[id]         = u;
                    special_tokens.push_back(content);
                    if (content == "<|endoftext|>") {
                        UNK_TOKEN_ID = id;
                        EOS_TOKEN_ID = id;
                        PAD_TOKEN_ID = id;
                    }
                    max_id = std::max(max_id, id);
                }
            }

            encoder_len = max_id + 1;
            LOG_DEBUG("qwen3 vocab size: %d", encoder_len);

            auto byte_unicode_pairs = bytes_to_unicode();
            byte_encoder            = std::map<int, std::u32string>(byte_unicode_pairs.begin(), byte_unicode_pairs.end());
            for (auto& pair : byte_unicode_pairs) {
                byte_decoder[pair.second] = pair.first;
            }

            std::vector<std::pair<std::u32string, std::u32string>> merge_pairs;
            auto merges_json = tok["model"]["merges"];
            merge_pairs.reserve(merges_json.size());
            for (auto& merge_item : merges_json) {
                if (merge_item.is_string()) {
                    std::string merge_str = merge_item.get<std::string>();
                    std::u32string merge_u32 = utf8_to_utf32(merge_str);
                    size_t space_pos = merge_u32.find(' ');
                    if (space_pos == std::u32string::npos) {
                        continue;
                    }
                    merge_pairs.emplace_back(merge_u32.substr(0, space_pos), merge_u32.substr(space_pos + 1));
                } else if (merge_item.is_array() && merge_item.size() == 2) {
                    std::string first = merge_item[0].get<std::string>();
                    std::string second = merge_item[1].get<std::string>();
                    merge_pairs.emplace_back(utf8_to_utf32(first), utf8_to_utf32(second));
                }
            }

            int rank = 0;
            for (const auto& merge : merge_pairs) {
                bpe_ranks[merge] = rank++;
            }
            bpe_len = rank;
        }

    public:
        explicit Qwen3Tokenizer(const std::string& tokenizer_json_str = "") {
            UNK_TOKEN = "<|endoftext|>";
            EOS_TOKEN = "<|endoftext|>";
            PAD_TOKEN = "<|endoftext|>";

            UNK_TOKEN_ID = 151643;
            EOS_TOKEN_ID = 151643;
            PAD_TOKEN_ID = 151643;

            std::string json_str = tokenizer_json_str;
            if (json_str.empty()) {
                if (const char* env_path = std::getenv("QWEN3_TOKENIZER_PATH"); env_path && *env_path) {
                    json_str = read_file_to_string(env_path);
                } else if (const char* env_path = std::getenv("ACE_QWEN3_TOKENIZER_PATH"); env_path && *env_path) {
                    json_str = read_file_to_string(env_path);
                } else if (const char* env_root = std::getenv("ACE_STEP_HOME"); env_root && *env_root) {
                    std::string p = std::string(env_root) + "/checkpoints/Qwen3-Embedding-0.6B/tokenizer.json";
                    json_str = read_file_to_string(p);
                } else if (const char* env_root = std::getenv("ACE_STEP_PATH"); env_root && *env_root) {
                    std::string p = std::string(env_root) + "/checkpoints/Qwen3-Embedding-0.6B/tokenizer.json";
                    json_str = read_file_to_string(p);
                } else {
                    // common local fallback paths
                    const std::vector<std::string> fallback_paths = {
                        "./ACE-Step-1.5/checkpoints/Qwen3-Embedding-0.6B/tokenizer.json",
                        "../ACE-Step-1.5/checkpoints/Qwen3-Embedding-0.6B/tokenizer.json",
                        "../../ACE-Step-1.5/checkpoints/Qwen3-Embedding-0.6B/tokenizer.json",
                        "./tokenizer.json",
                    };
                    for (const auto& p : fallback_paths) {
                        json_str = read_file_to_string(p);
                        if (!json_str.empty()) {
                            break;
                        }
                    }
                }
            }

            if (!json_str.empty()) {
                load_from_tokenizer_json(json_str);
            } else {
                LOG_WARN("Qwen3 tokenizer json not found, falling back to Qwen2 merges. Set QWEN3_TOKENIZER_PATH for correct tokenization.");
                special_tokens = {
                    "<|endoftext|>",
                    "<|im_start|>",
                    "<|im_end|>",
                    "<|object_ref_start|>",
                    "<|object_ref_end|>",
                    "<|box_start|>",
                    "<|box_end|>",
                    "<|quad_start|>",
                    "<|quad_end|>",
                    "<|vision_start|>",
                    "<|vision_end|>",
                    "<|vision_pad|>",
                    "<|image_pad|>",
                    "<|video_pad|>",
                    "<tool_call>",
                    "</tool_call>",
                    "<|fim_prefix|>",
                    "<|fim_middle|>",
                    "<|fim_suffix|>",
                    "<|fim_pad|>",
                    "<|repo_name|>",
                    "<|file_sep|>",
                    "<tool_response>",
                    "</tool_response>",
                    "<think>",
                    "</think>",
                };
                load_from_merges(load_qwen2_merges());
            }
        }
    };

    class MistralTokenizer : public BPETokenizer {
    protected:
        void load_from_merges(const std::string& merges_utf8_str, const std::string& vocab_utf8_str) {
            nlohmann::json vocab;

            try {
                vocab = nlohmann::json::parse(vocab_utf8_str);
            } catch (const nlohmann::json::parse_error&) {
                GGML_ABORT("invalid vocab json str");
            }
            for (const auto& [key, value] : vocab.items()) {
                std::u32string token = utf8_to_utf32(key);
                int i                = value;
                encoder[token]       = i;
                decoder[i]           = token;
            }
            encoder_len = static_cast<int>(vocab.size());
            LOG_DEBUG("vocab size: %d", encoder_len);

            auto byte_unicode_pairs = bytes_to_unicode();
            byte_encoder            = std::map<int, std::u32string>(byte_unicode_pairs.begin(), byte_unicode_pairs.end());
            for (auto& pair : byte_unicode_pairs) {
                byte_decoder[pair.second] = pair.first;
            }
            std::vector<std::u32string> merges;
            size_t start = 0;
            size_t pos;
            std::u32string merges_utf32_str = utf8_to_utf32(merges_utf8_str);
            while ((pos = merges_utf32_str.find('\n', start)) != std::string::npos) {
                merges.push_back(merges_utf32_str.substr(start, pos - start));
                start = pos + 1;
            }
            LOG_DEBUG("merges size %llu", merges.size());
            merges = std::vector<std::u32string>(merges.begin(), merges.end());
            std::vector<std::pair<std::u32string, std::u32string>> merge_pairs;
            // int print_num = 10;
            for (const auto& merge : merges) {
                size_t space_pos = merge.find(' ');
                merge_pairs.emplace_back(merge.substr(0, space_pos), merge.substr(space_pos + 1));
                // if (print_num > 0) {
                //     print_num--;
                //     printf("%s :: %s | %s \n", utf32_to_utf8(merge).c_str(), utf32_to_utf8(merge.substr(0, space_pos)).c_str(),
                //                     utf32_to_utf8(merge.substr(space_pos + 1)).c_str());
                // }
            }

            int rank = 0;
            for (const auto& merge : merge_pairs) {
                bpe_ranks[merge] = rank++;
            }
            bpe_len = rank;
        };

    public:
        explicit MistralTokenizer(const std::string& merges_utf8_str = "", const std::string& vocab_utf8_str = "") {
            add_bos_token = true;

            UNK_TOKEN = "<unk>";
            BOS_TOKEN = "<s>";
            EOS_TOKEN = "</s>";
            PAD_TOKEN = "<pad>";

            UNK_TOKEN_ID = 0;
            BOS_TOKEN_ID = 1;
            EOS_TOKEN_ID = 2;
            PAD_TOKEN_ID = 11;

            special_tokens = {
                "<unk>",
                "<s>",
                "</s>",
                "[INST]",
                "[/INST]",
                "[AVAILABLE_TOOLS]",
                "[/AVAILABLE_TOOLS]",
                "[TOOL_RESULTS]",
                "[/TOOL_RESULTS]",
                "[TOOL_CALLS]",
                "[IMG]",
                "<pad>",
                "[IMG_BREAK]",
                "[IMG_END]",
                "[PREFIX]",
                "[MIDDLE]",
                "[SUFFIX]",
                "[SYSTEM_PROMPT]",
                "[/SYSTEM_PROMPT]",
                "[TOOL_CONTENT]",
            };
            for (int i = 20; i < 1000; i++) {
                special_tokens.push_back("<SPECIAL_" + std::to_string(i) + ">");
            }

            if (merges_utf8_str.size() > 0 && vocab_utf8_str.size() > 0) {
                load_from_merges(merges_utf8_str, vocab_utf8_str);
            } else {
                load_from_merges(load_mistral_merges(), load_mistral_vocab_json());
            }
        }
    };

    enum class LLMArch {
        QWEN2_5_VL,
        QWEN3,
        MISTRAL_SMALL_3_2,
        ARCH_COUNT,
    };

    static const char* llm_arch_to_str[] = {
        "qwen2.5vl",
        "qwen3",
        "mistral_small3.2",
    };

    struct LLMVisionParams {
        int num_layers                      = 32;
        int64_t hidden_size                 = 1280;
        int64_t intermediate_size           = 3420;
        int num_heads                       = 16;
        int64_t in_channels                 = 3;
        int64_t out_hidden_size             = 3584;
        int temporal_patch_size             = 2;
        int patch_size                      = 14;
        int spatial_merge_size              = 2;
        int window_size                     = 112;
        std::set<int> fullatt_block_indexes = {7, 15, 23, 31};
    };

    struct LLMParams {
        LLMArch arch              = LLMArch::QWEN2_5_VL;
        int64_t num_layers        = 28;
        int64_t hidden_size       = 3584;
        int64_t intermediate_size = 18944;
        int num_heads             = 28;
        int num_kv_heads          = 4;
        int head_dim              = 128;
        bool qkv_bias             = true;
        bool qk_norm              = false;
        int64_t vocab_size        = 152064;
        float rms_norm_eps        = 1e-06f;
        LLMVisionParams vision;
    };

    struct MLP : public GGMLBlock {
    public:
        MLP(int64_t hidden_size, int64_t intermediate_size, bool bias = false) {
            blocks["gate_proj"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, intermediate_size, bias));
            blocks["up_proj"]   = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, intermediate_size, bias));
            blocks["down_proj"] = std::shared_ptr<GGMLBlock>(new Linear(intermediate_size, hidden_size, bias));
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
            // x: [N, n_token, hidden_size]
            auto gate_proj = std::dynamic_pointer_cast<Linear>(blocks["gate_proj"]);
            auto up_proj   = std::dynamic_pointer_cast<Linear>(blocks["up_proj"]);
            auto down_proj = std::dynamic_pointer_cast<Linear>(blocks["down_proj"]);

            auto h = gate_proj->forward(ctx, x);
            h      = ggml_silu_inplace(ctx->ggml_ctx, h);
            h      = ggml_mul_inplace(ctx->ggml_ctx, h, up_proj->forward(ctx, x));
            h      = down_proj->forward(ctx, h);
            return h;
        }
    };

    struct VisionPatchEmbed : public GGMLBlock {
    protected:
        bool llama_cpp_style;
        int patch_size;
        int temporal_patch_size;
        int64_t in_channels;
        int64_t embed_dim;

    public:
        VisionPatchEmbed(bool llama_cpp_style,
                         int patch_size          = 14,
                         int temporal_patch_size = 2,
                         int64_t in_channels     = 3,
                         int64_t embed_dim       = 1152)
            : llama_cpp_style(llama_cpp_style),
              patch_size(patch_size),
              temporal_patch_size(temporal_patch_size),
              in_channels(in_channels),
              embed_dim(embed_dim) {
            if (llama_cpp_style) {
                blocks["proj.0"] = std::shared_ptr<GGMLBlock>(new Conv2d(in_channels,
                                                                         embed_dim,
                                                                         {patch_size, patch_size},
                                                                         {patch_size, patch_size},  // stride
                                                                         {0, 0},                    // padding
                                                                         {1, 1},                    // dilation
                                                                         false));
                blocks["proj.1"] = std::shared_ptr<GGMLBlock>(new Conv2d(in_channels,
                                                                         embed_dim,
                                                                         {patch_size, patch_size},
                                                                         {patch_size, patch_size},  // stride
                                                                         {0, 0},                    // padding
                                                                         {1, 1},                    // dilation
                                                                         false));
            } else {
                std::tuple<int, int, int> kernel_size = {(int)temporal_patch_size, (int)patch_size, (int)patch_size};
                blocks["proj"]                        = std::shared_ptr<GGMLBlock>(new Conv3d(in_channels,
                                                                                              embed_dim,
                                                                                              kernel_size,
                                                                                              kernel_size,  // stride
                                                                                              {0, 0, 0},    // padding
                                                                                              {1, 1, 1},    // dilation
                                                                                              false));
            }
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
            // x: [N*grid_t*grid_h*grid_w, in_channels, temporal_patch_size*patch_size*patch_size]
            // return: [N*grid_t*grid_h*grid_w, embed_dim]
            x = ggml_reshape_4d(ctx->ggml_ctx,
                                x,
                                patch_size,
                                patch_size,
                                temporal_patch_size,
                                ggml_nelements(x) / (temporal_patch_size * patch_size * patch_size));

            if (llama_cpp_style) {
                auto proj_0 = std::dynamic_pointer_cast<Conv2d>(blocks["proj.0"]);
                auto proj_1 = std::dynamic_pointer_cast<Conv2d>(blocks["proj.1"]);

                auto x0 = ggml_ext_slice(ctx->ggml_ctx, x, 2, 0, 1);
                x0      = ggml_reshape_4d(ctx->ggml_ctx, x0, x0->ne[0], x0->ne[1], in_channels, x0->ne[3] / in_channels);
                x0      = proj_0->forward(ctx, x0);

                auto x1 = ggml_ext_slice(ctx->ggml_ctx, x, 2, 1, 2);
                x1      = ggml_reshape_4d(ctx->ggml_ctx, x1, x1->ne[0], x1->ne[1], in_channels, x1->ne[3] / in_channels);
                x1      = proj_1->forward(ctx, x1);

                x = ggml_add(ctx->ggml_ctx, x0, x1);
            } else {
                auto proj = std::dynamic_pointer_cast<Conv3d>(blocks["proj"]);

                x = proj->forward(ctx, x);
            }

            x = ggml_reshape_2d(ctx->ggml_ctx, x, embed_dim, ggml_nelements(x) / embed_dim);
            return x;
        }
    };

    struct PatchMerger : public GGMLBlock {
    protected:
        int64_t hidden_size;

    public:
        PatchMerger(int64_t dim,
                    int64_t context_dim,
                    int64_t spatial_merge_size) {
            hidden_size     = context_dim * spatial_merge_size * spatial_merge_size;
            blocks["ln_q"]  = std::shared_ptr<GGMLBlock>(new RMSNorm(context_dim, 1e-6f));
            blocks["mlp.0"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, hidden_size));
            // mlp.1 is nn.GELU()
            blocks["mlp.2"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, dim));
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
            auto ln_q  = std::dynamic_pointer_cast<RMSNorm>(blocks["ln_q"]);
            auto mlp_0 = std::dynamic_pointer_cast<Linear>(blocks["mlp.0"]);
            auto mlp_2 = std::dynamic_pointer_cast<Linear>(blocks["mlp.2"]);

            x = ln_q->forward(ctx, x);
            x = ggml_reshape_2d(ctx->ggml_ctx, x, hidden_size, ggml_nelements(x) / hidden_size);
            x = mlp_0->forward(ctx, x);
            x = ggml_ext_gelu(ctx->ggml_ctx, x);
            x = mlp_2->forward(ctx, x);
            return x;
        }
    };

    struct VisionAttention : public GGMLBlock {
    protected:
        bool llama_cpp_style;
        int head_dim;
        int num_heads;

    public:
        VisionAttention(bool llama_cpp_style,
                        int64_t hidden_size,
                        int num_heads)
            : llama_cpp_style(llama_cpp_style), num_heads(num_heads) {
            head_dim = static_cast<int>(hidden_size / num_heads);
            GGML_ASSERT(num_heads * head_dim == hidden_size);
            if (llama_cpp_style) {
                blocks["q_proj"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, hidden_size));
                blocks["k_proj"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, hidden_size));
                blocks["v_proj"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, hidden_size));
            } else {
                blocks["qkv"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, hidden_size * 3));
            }
            blocks["proj"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, hidden_size));
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* pe,
                                    struct ggml_tensor* mask = nullptr) {
            // x: [N, n_token, hidden_size]
            int64_t n_token = x->ne[1];
            int64_t N       = x->ne[2];
            auto proj       = std::dynamic_pointer_cast<Linear>(blocks["proj"]);

            std::vector<ggml_tensor*> qkv_vec;
            if (llama_cpp_style) {
                auto q_proj = std::dynamic_pointer_cast<Linear>(blocks["q_proj"]);
                auto k_proj = std::dynamic_pointer_cast<Linear>(blocks["k_proj"]);
                auto v_proj = std::dynamic_pointer_cast<Linear>(blocks["v_proj"]);

                auto q = q_proj->forward(ctx, x);
                auto k = k_proj->forward(ctx, x);
                auto v = v_proj->forward(ctx, x);

                qkv_vec = {q, k, v};
            } else {
                auto qkv_proj = std::dynamic_pointer_cast<Linear>(blocks["qkv"]);
                auto qkv      = qkv_proj->forward(ctx, x);
                qkv_vec       = split_qkv(ctx->ggml_ctx, qkv);
            }

            auto q = ggml_reshape_4d(ctx->ggml_ctx, qkv_vec[0], head_dim, num_heads, qkv_vec[0]->ne[1], qkv_vec[0]->ne[2]);  // [N, n_token, n_head, d_head]
            auto k = ggml_reshape_4d(ctx->ggml_ctx, qkv_vec[1], head_dim, num_heads, qkv_vec[1]->ne[1], qkv_vec[1]->ne[2]);  // [N, n_token, n_head, d_head]
            auto v = ggml_reshape_4d(ctx->ggml_ctx, qkv_vec[2], head_dim, num_heads, qkv_vec[2]->ne[1], qkv_vec[2]->ne[2]);  // [N, n_token, n_head, d_head]

            x = Rope::attention(ctx, q, k, v, pe, mask, 1.f, false);  // [N, n_token, hidden_size]

            x = proj->forward(ctx, x);  // [N, n_token, hidden_size]
            return x;
        }
    };

    struct VisionBlock : public GGMLBlock {
    public:
        VisionBlock(bool llama_cpp_style,
                    int64_t hidden_size,
                    int64_t intermediate_size,
                    int num_heads,
                    float eps = 1e-6f) {
            blocks["attn"]  = std::shared_ptr<GGMLBlock>(new VisionAttention(llama_cpp_style, hidden_size, num_heads));
            blocks["mlp"]   = std::shared_ptr<GGMLBlock>(new MLP(hidden_size, intermediate_size, true));
            blocks["norm1"] = std::shared_ptr<GGMLBlock>(new RMSNorm(hidden_size, eps));
            blocks["norm2"] = std::shared_ptr<GGMLBlock>(new RMSNorm(hidden_size, eps));
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* pe,
                                    struct ggml_tensor* mask = nullptr) {
            // x: [N, n_token, hidden_size]
            auto attn  = std::dynamic_pointer_cast<VisionAttention>(blocks["attn"]);
            auto mlp   = std::dynamic_pointer_cast<MLP>(blocks["mlp"]);
            auto norm1 = std::dynamic_pointer_cast<RMSNorm>(blocks["norm1"]);
            auto norm2 = std::dynamic_pointer_cast<RMSNorm>(blocks["norm2"]);

            auto residual = x;
            x             = norm1->forward(ctx, x);
            x             = attn->forward(ctx, x, pe, mask);
            x             = ggml_add_inplace(ctx->ggml_ctx, x, residual);

            residual = x;
            x        = norm2->forward(ctx, x);
            x        = mlp->forward(ctx, x);
            x        = ggml_add_inplace(ctx->ggml_ctx, x, residual);

            return x;
        }
    };

    struct VisionModel : public GGMLBlock {
    protected:
        int num_layers;
        int spatial_merge_size;
        std::set<int> fullatt_block_indexes;

    public:
        VisionModel(bool llama_cpp_style,
                    int num_layers,
                    int64_t in_channels,
                    int64_t hidden_size,
                    int64_t out_hidden_size,
                    int64_t intermediate_size,
                    int num_heads,
                    int spatial_merge_size,
                    int patch_size,
                    int temporal_patch_size,
                    int window_size,
                    std::set<int> fullatt_block_indexes = {7, 15, 23, 31},
                    float eps                           = 1e-6f)
            : num_layers(num_layers), fullatt_block_indexes(std::move(fullatt_block_indexes)), spatial_merge_size(spatial_merge_size) {
            blocks["patch_embed"] = std::shared_ptr<GGMLBlock>(new VisionPatchEmbed(llama_cpp_style,
                                                                                    patch_size,
                                                                                    temporal_patch_size,
                                                                                    in_channels,
                                                                                    hidden_size));
            for (int i = 0; i < num_layers; i++) {
                blocks["blocks." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new VisionBlock(llama_cpp_style,
                                                                                                   hidden_size,
                                                                                                   intermediate_size,
                                                                                                   num_heads,
                                                                                                   eps));
            }
            blocks["merger"] = std::shared_ptr<GGMLBlock>(new PatchMerger(out_hidden_size, hidden_size, spatial_merge_size));
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* pixel_values,
                                    struct ggml_tensor* pe,
                                    struct ggml_tensor* window_index,
                                    struct ggml_tensor* window_inverse_index,
                                    struct ggml_tensor* window_mask) {
            // pixel_values: [grid_t*(H/mh/ph)*(W/mw/pw)*mh*mw, C*pt*ph*pw]
            // window_index: [grid_t*(H/mh/ph)*(W/mw/pw)]
            // window_inverse_index: [grid_t*(H/mh/ph)*(W/mw/pw)]
            // window_mask: [grid_h*grid_w, grid_h*grid_w]
            auto patch_embed = std::dynamic_pointer_cast<VisionPatchEmbed>(blocks["patch_embed"]);
            auto merger      = std::dynamic_pointer_cast<PatchMerger>(blocks["merger"]);

            auto x = patch_embed->forward(ctx, pixel_values);

            x = ggml_reshape_4d(ctx->ggml_ctx, x, x->ne[0] * spatial_merge_size * spatial_merge_size, x->ne[1] / spatial_merge_size / spatial_merge_size, x->ne[2], x->ne[3]);
            x = ggml_get_rows(ctx->ggml_ctx, x, window_index);
            x = ggml_reshape_4d(ctx->ggml_ctx, x, x->ne[0] / spatial_merge_size / spatial_merge_size, x->ne[1] * spatial_merge_size * spatial_merge_size, x->ne[2], x->ne[3]);

            for (int i = 0; i < num_layers; i++) {
                auto block = std::dynamic_pointer_cast<VisionBlock>(blocks["blocks." + std::to_string(i)]);

                auto mask = window_mask;
                if (fullatt_block_indexes.find(i) != fullatt_block_indexes.end()) {
                    mask = nullptr;
                }
                x = block->forward(ctx, x, pe, mask);
            }

            x = merger->forward(ctx, x);

            x = ggml_get_rows(ctx->ggml_ctx, x, window_inverse_index);

            return x;
        }
    };

    struct Attention : public GGMLBlock {
    protected:
        LLMArch arch;
        int head_dim;
        int64_t num_heads;
        int64_t num_kv_heads;
        bool qk_norm;

    public:
        Attention(const LLMParams& params)
            : arch(params.arch), num_heads(params.num_heads), num_kv_heads(params.num_kv_heads), head_dim(params.head_dim), qk_norm(params.qk_norm) {
            blocks["q_proj"] = std::make_shared<Linear>(params.hidden_size, num_heads * head_dim, params.qkv_bias);
            blocks["k_proj"] = std::make_shared<Linear>(params.hidden_size, num_kv_heads * head_dim, params.qkv_bias);
            blocks["v_proj"] = std::make_shared<Linear>(params.hidden_size, num_kv_heads * head_dim, params.qkv_bias);
            blocks["o_proj"] = std::make_shared<Linear>(num_heads * head_dim, params.hidden_size, false);
            if (params.qk_norm) {
                blocks["q_norm"] = std::make_shared<RMSNorm>(head_dim, params.rms_norm_eps);
                blocks["k_norm"] = std::make_shared<RMSNorm>(head_dim, params.rms_norm_eps);
            }
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* input_pos,
                                    struct ggml_tensor* attention_mask = nullptr,
                                    struct ggml_tensor* past_k         = nullptr,
                                    struct ggml_tensor* past_v         = nullptr,
                                    struct ggml_tensor* kv_row_indices = nullptr,
                                    int64_t kv_cache_len               = -1,
                                    struct ggml_tensor** present_k     = nullptr,
                                    struct ggml_tensor** present_v     = nullptr) {
            // x: [N, n_token, hidden_size]
            int64_t n_token = x->ne[1];
            int64_t N       = x->ne[2];
            auto q_proj     = std::dynamic_pointer_cast<Linear>(blocks["q_proj"]);
            auto k_proj     = std::dynamic_pointer_cast<Linear>(blocks["k_proj"]);
            auto v_proj     = std::dynamic_pointer_cast<Linear>(blocks["v_proj"]);
            auto out_proj   = std::dynamic_pointer_cast<Linear>(blocks["o_proj"]);

            auto q = q_proj->forward(ctx, x);  // [N, n_token, num_heads*head_dim]
            auto k = k_proj->forward(ctx, x);  // [N, n_token, num_kv_heads*head_dim]
            auto v = v_proj->forward(ctx, x);  // [N, n_token, num_kv_heads*head_dim]

            q = ggml_reshape_4d(ctx->ggml_ctx, q, head_dim, num_heads, n_token, N);     // [N, n_token, num_heads, head_dim]
            k = ggml_reshape_4d(ctx->ggml_ctx, k, head_dim, num_kv_heads, n_token, N);  // [N, n_token, num_kv_heads, head_dim]
            v = ggml_reshape_4d(ctx->ggml_ctx, v, head_dim, num_kv_heads, n_token, N);  // [N, n_token, num_kv_heads, head_dim]

            if (qk_norm) {
                auto q_norm = std::dynamic_pointer_cast<RMSNorm>(blocks["q_norm"]);
                auto k_norm = std::dynamic_pointer_cast<RMSNorm>(blocks["k_norm"]);

                q = q_norm->forward(ctx, q);
                k = k_norm->forward(ctx, k);
            }

            if (arch == LLMArch::MISTRAL_SMALL_3_2) {
                q = ggml_rope_ext(ctx->ggml_ctx, q, input_pos, nullptr, 128, GGML_ROPE_TYPE_NORMAL, 8192, 1000000000.f, 1.f, 0.f, 1.f, 32.f, 1.f);
                k = ggml_rope_ext(ctx->ggml_ctx, k, input_pos, nullptr, 128, GGML_ROPE_TYPE_NORMAL, 8192, 1000000000.f, 1.f, 0.f, 1.f, 32.f, 1.f);
            } else if (arch == LLMArch::QWEN3) {
                q = ggml_rope_ext(ctx->ggml_ctx, q, input_pos, nullptr, 128, GGML_ROPE_TYPE_NEOX, 40960, 1000000.f, 1.f, 0.f, 1.f, 32.f, 1.f);
                k = ggml_rope_ext(ctx->ggml_ctx, k, input_pos, nullptr, 128, GGML_ROPE_TYPE_NEOX, 40960, 1000000.f, 1.f, 0.f, 1.f, 32.f, 1.f);
            } else {
                int sections[4] = {16, 24, 24, 0};
                q               = ggml_rope_multi(ctx->ggml_ctx, q, input_pos, nullptr, head_dim, sections, GGML_ROPE_TYPE_MROPE, 128000, 1000000.f, 1.f, 0.f, 1.f, 32.f, 1.f);
                k               = ggml_rope_multi(ctx->ggml_ctx, k, input_pos, nullptr, head_dim, sections, GGML_ROPE_TYPE_MROPE, 128000, 1000000.f, 1.f, 0.f, 1.f, 32.f, 1.f);
            }

            if (past_k != nullptr && past_v != nullptr) {
                if (kv_row_indices != nullptr) {
                    auto k_flat = ggml_reshape_3d(ctx->ggml_ctx, k, head_dim * num_kv_heads, n_token, N);
                    auto v_flat = ggml_reshape_3d(ctx->ggml_ctx, v, head_dim * num_kv_heads, n_token, N);
                    if (k_flat->type != GGML_TYPE_F32) {
                        k_flat = ggml_cast(ctx->ggml_ctx, k_flat, GGML_TYPE_F32);
                    }
                    if (v_flat->type != GGML_TYPE_F32) {
                        v_flat = ggml_cast(ctx->ggml_ctx, v_flat, GGML_TYPE_F32);
                    }
                    k_flat = ggml_cont(ctx->ggml_ctx, k_flat);
                    v_flat = ggml_cont(ctx->ggml_ctx, v_flat);

                    const int64_t cache_tokens = past_k->ne[1];
                    const int64_t use_tokens   = kv_cache_len > 0 ? kv_cache_len : cache_tokens;
                    GGML_ASSERT(use_tokens <= cache_tokens);
                    const int64_t start_token = std::max<int64_t>(0, use_tokens - n_token);
                    GGML_ASSERT(start_token + n_token <= cache_tokens);

                    auto k_cache = ggml_set(ctx->ggml_ctx,
                                            past_k,
                                            k_flat,
                                            past_k->nb[1],
                                            past_k->nb[2],
                                            past_k->nb[3],
                                            static_cast<size_t>(start_token) * past_k->nb[1]);
                    auto v_cache = ggml_set(ctx->ggml_ctx,
                                            past_v,
                                            v_flat,
                                            past_v->nb[1],
                                            past_v->nb[2],
                                            past_v->nb[3],
                                            static_cast<size_t>(start_token) * past_v->nb[1]);

                    auto k_cache_4d = ggml_reshape_4d(ctx->ggml_ctx, k_cache, head_dim, num_kv_heads, cache_tokens, N);
                    auto v_cache_4d = ggml_reshape_4d(ctx->ggml_ctx, v_cache, head_dim, num_kv_heads, cache_tokens, N);

                    k = ggml_view_4d(ctx->ggml_ctx, k_cache_4d, head_dim, num_kv_heads, use_tokens, N, k_cache_4d->nb[1], k_cache_4d->nb[2], k_cache_4d->nb[3], 0);
                    v = ggml_view_4d(ctx->ggml_ctx, v_cache_4d, head_dim, num_kv_heads, use_tokens, N, v_cache_4d->nb[1], v_cache_4d->nb[2], v_cache_4d->nb[3], 0);
                } else {
                    k = ggml_concat(ctx->ggml_ctx, past_k, k, 2);
                    v = ggml_concat(ctx->ggml_ctx, past_v, v, 2);
                }
            }

            if (present_k != nullptr) {
                *present_k = k;
            }
            if (present_v != nullptr) {
                *present_v = v;
            }

            q = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, q, 0, 2, 1, 3));  // [N, num_heads, n_token, head_dim]
            q = ggml_reshape_3d(ctx->ggml_ctx, q, q->ne[0], q->ne[1], q->ne[2] * q->ne[3]);      // [N*num_heads, n_token, head_dim]

            k = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, k, 0, 2, 1, 3));  // [N, num_kv_heads, n_token, head_dim]
            k = ggml_reshape_3d(ctx->ggml_ctx, k, k->ne[0], k->ne[1], k->ne[2] * k->ne[3]);      // [N*num_kv_heads, n_token, head_dim]

            x = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v, num_heads, attention_mask, true, ctx->flash_attn_enabled);  // [N, n_token, hidden_size]

            x = out_proj->forward(ctx, x);  // [N, n_token, hidden_size]
            return x;
        }
    };

    struct TransformerBlock : public GGMLBlock {
    public:
        TransformerBlock(const LLMParams& params) {
            blocks["self_attn"]                = std::make_shared<Attention>(params);
            blocks["mlp"]                      = std::make_shared<MLP>(params.hidden_size, params.intermediate_size);
            blocks["input_layernorm"]          = std::make_shared<RMSNorm>(params.hidden_size, params.rms_norm_eps);
            blocks["post_attention_layernorm"] = std::make_shared<RMSNorm>(params.hidden_size, params.rms_norm_eps);
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* input_pos,
                                    struct ggml_tensor* attention_mask = nullptr,
                                    struct ggml_tensor* past_k         = nullptr,
                                    struct ggml_tensor* past_v         = nullptr,
                                    struct ggml_tensor* kv_row_indices = nullptr,
                                    int64_t kv_cache_len               = -1,
                                    struct ggml_tensor** present_k     = nullptr,
                                    struct ggml_tensor** present_v     = nullptr) {
            // x: [N, n_token, hidden_size]
            auto self_attn                = std::dynamic_pointer_cast<Attention>(blocks["self_attn"]);
            auto mlp                      = std::dynamic_pointer_cast<MLP>(blocks["mlp"]);
            auto input_layernorm          = std::dynamic_pointer_cast<RMSNorm>(blocks["input_layernorm"]);
            auto post_attention_layernorm = std::dynamic_pointer_cast<RMSNorm>(blocks["post_attention_layernorm"]);

            auto residual = x;
            x             = input_layernorm->forward(ctx, x);
            x             = self_attn->forward(ctx, x, input_pos, attention_mask, past_k, past_v, kv_row_indices, kv_cache_len, present_k, present_v);
            x             = ggml_add_inplace(ctx->ggml_ctx, x, residual);

            residual = x;
            x        = post_attention_layernorm->forward(ctx, x);
            x        = mlp->forward(ctx, x);
            x        = ggml_add_inplace(ctx->ggml_ctx, x, residual);

            return x;
        }
    };

    struct TextModel : public GGMLBlock {
    protected:
        int64_t num_layers;

    public:
        TextModel(const LLMParams& params)
            : num_layers(params.num_layers) {
            blocks["embed_tokens"] = std::shared_ptr<GGMLBlock>(new Embedding(params.vocab_size, params.hidden_size));
            for (int i = 0; i < num_layers; i++) {
                blocks["layers." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new TransformerBlock(params));
            }
            blocks["norm"] = std::shared_ptr<GGMLBlock>(new RMSNorm(params.hidden_size, params.rms_norm_eps));
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* input_ids,
                                    struct ggml_tensor* input_pos,
                                    struct ggml_tensor* attention_mask,
                                    std::vector<std::pair<int, ggml_tensor*>> image_embeds,
                                    std::set<int> out_layers,
                                    const std::vector<ggml_tensor*>* past_k_cache = nullptr,
                                    const std::vector<ggml_tensor*>* past_v_cache = nullptr,
                                    struct ggml_tensor* kv_row_indices             = nullptr,
                                    int64_t kv_cache_len                           = -1,
                                    std::vector<ggml_tensor*>* present_k_cache    = nullptr,
                                    std::vector<ggml_tensor*>* present_v_cache    = nullptr) {
            // input_ids: [N, n_token]
            // return: [N, n_token, hidden_size]

            auto embed_tokens = std::dynamic_pointer_cast<Embedding>(blocks["embed_tokens"]);
            auto norm         = std::dynamic_pointer_cast<RMSNorm>(blocks["norm"]);

            auto x = embed_tokens->forward(ctx, input_ids);

            std::vector<ggml_tensor*> intermediate_outputs;

            if (image_embeds.size() > 0) {
                GGML_ASSERT(x->ne[2] == 1);  // N == 1

                auto raw_x              = ggml_cast(ctx->ggml_ctx, x, image_embeds[0].second->type);
                int64_t txt_token_start = 0;
                int64_t txt_token_end   = 0;

                ggml_tensor* input_embed = nullptr;

                for (int i = 0; i < image_embeds.size(); i++) {
                    if (i == 0) {
                        txt_token_start = 0;
                    } else {
                        txt_token_start = image_embeds[i - 1].first + image_embeds[i - 1].second->ne[1];
                    }
                    txt_token_end = image_embeds[i].first;

                    auto txt_embed = ggml_ext_slice(ctx->ggml_ctx, raw_x, 1, txt_token_start, txt_token_end);
                    if (input_embed == nullptr) {
                        input_embed = txt_embed;
                    } else {
                        input_embed = ggml_concat(ctx->ggml_ctx, input_embed, txt_embed, 1);
                    }

                    auto image_embed = image_embeds[i].second;
                    input_embed      = ggml_concat(ctx->ggml_ctx, input_embed, image_embed, 1);
                }

                txt_token_start = image_embeds[image_embeds.size() - 1].first + image_embeds[image_embeds.size() - 1].second->ne[1];
                txt_token_end   = raw_x->ne[1];

                auto final_txt_embed = ggml_ext_slice(ctx->ggml_ctx, raw_x, 1, txt_token_start, txt_token_end);

                input_embed = ggml_concat(ctx->ggml_ctx, input_embed, final_txt_embed, 1);
                GGML_ASSERT(raw_x->ne[1] == input_embed->ne[1]);

                x = input_embed;
            }

            if (out_layers.find(0) != out_layers.end()) {
                intermediate_outputs.push_back(x);
            }

            if (present_k_cache != nullptr) {
                present_k_cache->clear();
                present_k_cache->reserve(num_layers);
            }
            if (present_v_cache != nullptr) {
                present_v_cache->clear();
                present_v_cache->reserve(num_layers);
            }

            for (int i = 0; i < num_layers; i++) {
                auto block = std::dynamic_pointer_cast<TransformerBlock>(blocks["layers." + std::to_string(i)]);
                ggml_tensor* layer_past_k = nullptr;
                ggml_tensor* layer_past_v = nullptr;
                if (past_k_cache != nullptr && i < static_cast<int>(past_k_cache->size())) {
                    layer_past_k = (*past_k_cache)[i];
                }
                if (past_v_cache != nullptr && i < static_cast<int>(past_v_cache->size())) {
                    layer_past_v = (*past_v_cache)[i];
                }
                ggml_tensor* layer_present_k = nullptr;
                ggml_tensor* layer_present_v = nullptr;
                x = block->forward(ctx, x, input_pos, attention_mask, layer_past_k, layer_past_v, kv_row_indices, kv_cache_len, &layer_present_k, &layer_present_v);
                if (present_k_cache != nullptr) {
                    present_k_cache->push_back(layer_present_k);
                }
                if (present_v_cache != nullptr) {
                    present_v_cache->push_back(layer_present_v);
                }
                if (out_layers.find(i + 1) != out_layers.end()) {
                    intermediate_outputs.push_back(x);
                }
            }

            if (!intermediate_outputs.empty()) {
                x = intermediate_outputs[0];
                for (int i = 1; i < intermediate_outputs.size(); i++) {
                    x = ggml_concat(ctx->ggml_ctx, x, intermediate_outputs[i], 0);
                }
            } else {
                x = norm->forward(ctx, x);
            }
            return x;
        }

        struct ggml_tensor* get_embedding_weight() const {
            auto embed_tokens = std::dynamic_pointer_cast<Embedding>(blocks.at("embed_tokens"));
            return embed_tokens ? embed_tokens->get_weight() : nullptr;
        }
    };

    struct LLM : public GGMLBlock {
        bool enable_vision;
        LLMParams params;

    public:
        LLM() = default;
        LLM(LLMParams params, bool enable_vision = false, bool llama_cpp_style = false)
            : enable_vision(enable_vision), params(params) {
            blocks["model"] = std::shared_ptr<GGMLBlock>(new TextModel(params));
            if (enable_vision) {
                blocks["visual"] = std::shared_ptr<GGMLBlock>(new VisionModel(llama_cpp_style,
                                                                              params.vision.num_layers,
                                                                              params.vision.in_channels,
                                                                              params.vision.hidden_size,
                                                                              params.vision.out_hidden_size,
                                                                              params.vision.intermediate_size,
                                                                              params.vision.num_heads,
                                                                              params.vision.spatial_merge_size,
                                                                              params.vision.patch_size,
                                                                              params.vision.temporal_patch_size,
                                                                              params.vision.window_size,
                                                                              params.vision.fullatt_block_indexes));
            }
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* input_ids,
                                    struct ggml_tensor* input_pos,
                                    struct ggml_tensor* attention_mask,
                                    std::vector<std::pair<int, ggml_tensor*>> image_embeds,
                                    std::set<int> out_layers,
                                    const std::vector<ggml_tensor*>* past_k_cache = nullptr,
                                    const std::vector<ggml_tensor*>* past_v_cache = nullptr,
                                    struct ggml_tensor* kv_row_indices             = nullptr,
                                    int64_t kv_cache_len                           = -1,
                                    std::vector<ggml_tensor*>* present_k_cache    = nullptr,
                                    std::vector<ggml_tensor*>* present_v_cache    = nullptr) {
            // input_ids: [N, n_token]
            auto model = std::dynamic_pointer_cast<TextModel>(blocks["model"]);

            auto x = model->forward(ctx,
                                    input_ids,
                                    input_pos,
                                    attention_mask,
                                    image_embeds,
                                    out_layers,
                                    past_k_cache,
                                    past_v_cache,
                                    kv_row_indices,
                                    kv_cache_len,
                                    present_k_cache,
                                    present_v_cache);
            return x;
        }

        struct ggml_tensor* get_embedding_weight() const {
            auto model = std::dynamic_pointer_cast<TextModel>(blocks.at("model"));
            return model ? model->get_embedding_weight() : nullptr;
        }

        struct ggml_tensor* vision_forward(GGMLRunnerContext* ctx,
                                           struct ggml_tensor* pixel_values,
                                           struct ggml_tensor* pe,
                                           struct ggml_tensor* window_index,
                                           struct ggml_tensor* window_inverse_index,
                                           struct ggml_tensor* window_mask) {
            GGML_ASSERT(enable_vision);
            auto vision_model = std::dynamic_pointer_cast<VisionModel>(blocks["visual"]);
            return vision_model->forward(ctx, pixel_values, pe, window_index, window_inverse_index, window_mask);
        }
    };

    struct LLMRunner : public GGMLRunner {
        LLMParams params;
        bool enable_vision;
        LLM model;
        ggml_tensor* logit_scale = nullptr;
        std::string logit_scale_name;

        std::vector<int> input_pos_vec;
        std::vector<int64_t> kv_row_indices_vec;
        std::vector<float> attention_mask_vec;
        std::vector<float> window_mask_vec;
        std::vector<int> window_index_vec;
        std::vector<int> window_inverse_index_vec;
        std::vector<float> pe_vec;

        int64_t kv_cache_capacity   = 0;
        int64_t kv_cache_batch_size = 0;
        int64_t logits_range_start  = 0;
        int64_t logits_range_end    = -1;

        struct Decode1TokenGraphState {
            bool ready               = false;
            int64_t batch_size       = 0;
            int64_t kv_capacity      = 0;
            struct ggml_cgraph* graph = nullptr;
            ggml_tensor* input_ids   = nullptr;
            ggml_tensor* input_pos   = nullptr;
            ggml_tensor* attention_mask = nullptr;
            ggml_tensor* kv_row_indices = nullptr;
            ggml_tensor* logits      = nullptr;
        } decode_graph_state;

        std::vector<int32_t> decode_input_ids_vec;
        std::vector<int> decode_input_pos_vec;
        std::vector<float> decode_attention_mask_vec;

        LLMRunner(LLMArch arch,
                  ggml_backend_t backend,
                  bool offload_params_to_cpu,
                  const String2TensorStorage& tensor_storage_map,
                  const std::string prefix,
                  bool enable_vision_ = false)
            : GGMLRunner(backend, offload_params_to_cpu), enable_vision(enable_vision_) {
            params.arch = arch;
            if (arch == LLMArch::MISTRAL_SMALL_3_2) {
                params.head_dim     = 128;
                params.num_heads    = 32;
                params.num_kv_heads = 8;
                params.qkv_bias     = false;
                params.rms_norm_eps = 1e-5f;
            } else if (arch == LLMArch::QWEN3) {
                params.head_dim     = 128;
                params.num_heads    = 32;
                params.num_kv_heads = 8;
                params.qkv_bias     = false;
                params.qk_norm      = true;
                params.rms_norm_eps = 1e-6f;
            }
            bool have_vision_weight = false;
            bool llama_cpp_style    = false;
            params.num_layers       = 0;
            for (auto pair : tensor_storage_map) {
                std::string tensor_name = pair.first;
                if (tensor_name.find(prefix) == std::string::npos)
                    continue;
                size_t pos = tensor_name.find("visual.");
                if (pos != std::string::npos) {
                    have_vision_weight = true;
                    if (contains(tensor_name, "attn.q_proj")) {
                        llama_cpp_style = true;
                    }
                    continue;
                }
                pos = tensor_name.find("layers.");
                if (pos != std::string::npos) {
                    tensor_name = tensor_name.substr(pos);  // remove prefix
                    auto items  = split_string(tensor_name, '.');
                    if (items.size() > 1) {
                        int block_index = atoi(items[1].c_str());
                        if (block_index + 1 > params.num_layers) {
                            params.num_layers = block_index + 1;
                        }
                    }
                }
                if (contains(tensor_name, "embed_tokens.weight")) {
                    params.hidden_size = pair.second.ne[0];
                    params.vocab_size  = pair.second.ne[1];
                }
                if (contains(tensor_name, "layers.0.mlp.gate_proj.weight")) {
                    params.intermediate_size = pair.second.ne[1];
                }
            }
            if (arch == LLMArch::QWEN3 && params.num_layers == 28) {  // Qwen3 2B
                params.num_heads = 16;
            }
            LOG_DEBUG("llm: num_layers = %" PRId64 ", vocab_size = %" PRId64 ", hidden_size = %" PRId64 ", intermediate_size = %" PRId64,
                      params.num_layers,
                      params.vocab_size,
                      params.hidden_size,
                      params.intermediate_size);
            if (enable_vision && !have_vision_weight) {
                LOG_WARN("no vision weights detected, vision disabled");
                enable_vision = false;
            }
            if (enable_vision) {
                LOG_DEBUG("enable llm vision");
                if (llama_cpp_style) {
                    LOG_DEBUG("llama.cpp style vision weight");
                }
            }
            model = LLM(params, enable_vision, llama_cpp_style);
            model.init(params_ctx, tensor_storage_map, prefix);

            std::string root_prefix = prefix;
            if (ends_with(root_prefix, ".transformer.model")) {
                root_prefix = root_prefix.substr(0, root_prefix.size() - strlen(".transformer.model"));
            } else if (ends_with(root_prefix, ".transformer")) {
                root_prefix = root_prefix.substr(0, root_prefix.size() - strlen(".transformer"));
            } else if (ends_with(root_prefix, ".model")) {
                root_prefix = root_prefix.substr(0, root_prefix.size() - strlen(".model"));
            }
            std::string candidate = root_prefix + ".logit_scale";
            auto it = tensor_storage_map.find(candidate);
            if (it != tensor_storage_map.end()) {
                logit_scale_name = candidate;
                enum ggml_type wtype = GGML_TYPE_F32;
                if (it->second.expected_type != GGML_TYPE_COUNT) {
                    wtype = it->second.expected_type;
                } else {
                    wtype = it->second.type;
                }
                logit_scale = ggml_new_tensor_1d(params_ctx, wtype, 1);
            }

            logits_range_start = 0;
            logits_range_end   = params.vocab_size;
        }

        std::string get_desc() override {
            return llm_arch_to_str[static_cast<int>(params.arch)];
        }

        void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
            model.get_param_tensors(tensors, prefix);
            if (logit_scale != nullptr && !logit_scale_name.empty()) {
                tensors[logit_scale_name] = logit_scale;
            }
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* input_ids,
                                    struct ggml_tensor* input_pos,
                                    struct ggml_tensor* attention_mask,
                                    std::vector<std::pair<int, ggml_tensor*>> image_embeds,
                                    std::set<int> out_layers,
                                    const std::vector<ggml_tensor*>* past_k_cache = nullptr,
                                    const std::vector<ggml_tensor*>* past_v_cache = nullptr,
                                    struct ggml_tensor* kv_row_indices             = nullptr,
                                    int64_t kv_cache_len                           = -1,
                                    std::vector<ggml_tensor*>* present_k_cache    = nullptr,
                                    std::vector<ggml_tensor*>* present_v_cache    = nullptr) {
            auto hidden_states = model.forward(ctx,
                                               input_ids,
                                               input_pos,
                                               attention_mask,
                                               image_embeds,
                                               out_layers,
                                               past_k_cache,
                                               past_v_cache,
                                               kv_row_indices,
                                               kv_cache_len,
                                               present_k_cache,
                                               present_v_cache);  // [N, n_token, hidden_size]
            return hidden_states;
        }

        struct ggml_tensor* vision_forward(GGMLRunnerContext* ctx,
                                           struct ggml_tensor* pixel_values,
                                           struct ggml_tensor* input_pos,
                                           struct ggml_tensor* window_index,
                                           struct ggml_tensor* window_inverse_index,
                                           struct ggml_tensor* window_mask) {
            auto hidden_states = model.vision_forward(ctx, pixel_values, input_pos, window_index, window_inverse_index, window_mask);
            return hidden_states;
        }

        struct ggml_tensor* get_embedding_weight() const {
            return model.get_embedding_weight();
        }

        bool ensure_kv_cache(int64_t required_tokens, int64_t batch_size) {
            if (required_tokens <= 0 || batch_size <= 0) {
                return false;
            }

            int64_t target_capacity = required_tokens;
            if (kv_cache_capacity > 0 && kv_cache_batch_size == batch_size && kv_cache_capacity < required_tokens) {
                target_capacity = std::max<int64_t>(required_tokens, kv_cache_capacity * 2);
            } else if (kv_cache_capacity == 0) {
                target_capacity = std::max<int64_t>(required_tokens, 1024);
            }

            bool need_realloc = cache_ctx == nullptr ||
                                cache_buffer == nullptr ||
                                kv_cache_batch_size != batch_size ||
                                kv_cache_capacity < required_tokens;
            if (!need_realloc) {
                return true;
            }

            free_cache_ctx_and_buffer();
            alloc_cache_ctx();

            for (int i = 0; i < params.num_layers; ++i) {
                std::string k_name = "llm.k." + std::to_string(i);
                std::string v_name = "llm.v." + std::to_string(i);
                auto k_cache = ggml_new_tensor_3d(cache_ctx,
                                                  GGML_TYPE_F32,
                                                  params.head_dim * params.num_kv_heads,
                                                  target_capacity,
                                                  batch_size);
                auto v_cache = ggml_new_tensor_3d(cache_ctx,
                                                  GGML_TYPE_F32,
                                                  params.head_dim * params.num_kv_heads,
                                                  target_capacity,
                                                  batch_size);
                ggml_set_name(k_cache, k_name.c_str());
                ggml_set_name(v_cache, v_name.c_str());
            }

            cache_buffer = ggml_backend_alloc_ctx_tensors(cache_ctx, runtime_backend);
            if (cache_buffer == nullptr) {
                LOG_ERROR("%s alloc kv cache backend buffer failed", get_desc().c_str());
                kv_cache_capacity   = 0;
                kv_cache_batch_size = 0;
                return false;
            }

            ggml_backend_buffer_clear(cache_buffer, 0);
            kv_cache_capacity   = target_capacity;
            kv_cache_batch_size = batch_size;
            return true;
        }

        void set_logits_range(int64_t start, int64_t end) {
            int64_t vocab = params.vocab_size;
            start = std::max<int64_t>(0, start);
            end   = std::min<int64_t>(vocab, end);
            if (end <= start) {
                start = 0;
                end   = vocab;
            }
            logits_range_start = start;
            logits_range_end   = end;
        }

        int64_t get_logits_range_start() const {
            return logits_range_start;
        }

        int64_t get_logits_range_end() const {
            return logits_range_end;
        }

        void invalidate_decode_graph_state() {
            if (decode_graph_state.ready) {
                free_compute_buffer();
            }
            decode_graph_state = Decode1TokenGraphState{};
        }

        struct ggml_cgraph* build_graph_logits_kv_decode_1token(int64_t batch_size, int64_t kv_capacity) {
            struct ggml_cgraph* gf = ggml_new_graph(compute_ctx);

            decode_input_ids_vec.resize(static_cast<size_t>(batch_size), 0);
            decode_input_pos_vec.resize((params.arch == LLMArch::MISTRAL_SMALL_3_2 || params.arch == LLMArch::QWEN3) ? 1 : 4, 0);
            decode_attention_mask_vec.resize(static_cast<size_t>(kv_capacity * batch_size), 0.f);
            kv_row_indices_vec.resize(static_cast<size_t>(batch_size), 0);

            auto input_ids = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_I32, 1, batch_size);
            set_backend_tensor_data(input_ids, decode_input_ids_vec.data());

            auto input_pos = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_I32, decode_input_pos_vec.size());
            set_backend_tensor_data(input_pos, decode_input_pos_vec.data());

            auto attention_mask = ggml_new_tensor_3d(compute_ctx, GGML_TYPE_F32, kv_capacity, 1, batch_size);
            set_backend_tensor_data(attention_mask, decode_attention_mask_vec.data());

            auto kv_row_indices = ggml_new_tensor_3d(compute_ctx, GGML_TYPE_I64, 1, batch_size, 1);
            set_backend_tensor_data(kv_row_indices, kv_row_indices_vec.data());

            std::vector<ggml_tensor*> past_k_cache(params.num_layers, nullptr);
            std::vector<ggml_tensor*> past_v_cache(params.num_layers, nullptr);
            for (int i = 0; i < params.num_layers; ++i) {
                std::string k_name = "llm.k." + std::to_string(i);
                std::string v_name = "llm.v." + std::to_string(i);
                auto k_cache       = get_cache_tensor_by_name(k_name);
                auto v_cache       = get_cache_tensor_by_name(v_name);
                GGML_ASSERT(k_cache != nullptr && v_cache != nullptr);
                past_k_cache[i] = k_cache;
                past_v_cache[i] = v_cache;
            }

            std::vector<std::pair<int, ggml_tensor*>> image_embeds;
            std::set<int> out_layers;
            auto runner_ctx = get_context();
            auto hidden_states = forward(&runner_ctx,
                                         input_ids,
                                         input_pos,
                                         attention_mask,
                                         image_embeds,
                                         out_layers,
                                         &past_k_cache,
                                         &past_v_cache,
                                         kv_row_indices,
                                         kv_capacity);

            auto weight = get_embedding_weight();
            GGML_ASSERT(weight != nullptr);
            int64_t start = std::max<int64_t>(0, logits_range_start);
            int64_t end   = logits_range_end > 0 ? std::min<int64_t>(logits_range_end, weight->ne[1]) : weight->ne[1];
            if (end <= start) {
                start = 0;
                end   = weight->ne[1];
            }
            auto weight_slice = ggml_ext_slice(compute_ctx, weight, 1, start, end);
            auto logits       = ggml_mul_mat(compute_ctx, weight_slice, hidden_states);

            ggml_build_forward_expand(gf, input_ids);
            ggml_build_forward_expand(gf, input_pos);
            ggml_build_forward_expand(gf, attention_mask);
            ggml_build_forward_expand(gf, kv_row_indices);
            ggml_build_forward_expand(gf, logits);

            decode_graph_state.graph          = gf;
            decode_graph_state.input_ids      = input_ids;
            decode_graph_state.input_pos      = input_pos;
            decode_graph_state.attention_mask = attention_mask;
            decode_graph_state.kv_row_indices = kv_row_indices;
            decode_graph_state.logits         = logits;

            return gf;
        }

        bool prepare_decode_graph_1token(int64_t batch_size, int64_t kv_capacity) {
            if (decode_graph_state.ready &&
                decode_graph_state.batch_size == batch_size &&
                decode_graph_state.kv_capacity == kv_capacity &&
                decode_graph_state.graph != nullptr) {
                return true;
            }

            if (!ensure_kv_cache(kv_capacity, batch_size)) {
                return false;
            }

            auto get_graph = [&]() -> struct ggml_cgraph* {
                return build_graph_logits_kv_decode_1token(batch_size, kv_capacity);
            };

            if (!alloc_compute_buffer(get_graph)) {
                return false;
            }

            reset_compute_ctx();
            struct ggml_cgraph* gf = get_compute_graph(get_graph);
            if (!ggml_gallocr_alloc_graph(compute_allocr, gf)) {
                LOG_ERROR("%s alloc decode graph failed", get_desc().c_str());
                invalidate_decode_graph_state();
                return false;
            }
            copy_data_to_backend_tensor();

            decode_graph_state.graph       = gf;
            decode_graph_state.batch_size  = batch_size;
            decode_graph_state.kv_capacity = kv_capacity;
            decode_graph_state.ready       = true;
            return true;
        }

        bool compute_logits_kv_decode_1token(const int n_threads,
                                             const std::vector<int>& token_ids,
                                             int64_t n_past,
                                             const std::vector<int>& pad_lens,
                                             ggml_tensor** output,
                                             ggml_context* output_ctx = nullptr) {
            const int64_t batch_size = static_cast<int64_t>(token_ids.size());
            if (batch_size <= 0 || kv_cache_capacity <= 0 || kv_cache_batch_size != batch_size) {
                return false;
            }
            if (static_cast<size_t>(batch_size) != pad_lens.size()) {
                return false;
            }

            if (!prepare_decode_graph_1token(batch_size, kv_cache_capacity)) {
                return false;
            }

            decode_input_ids_vec.resize(static_cast<size_t>(batch_size));
            for (int64_t b = 0; b < batch_size; ++b) {
                decode_input_ids_vec[static_cast<size_t>(b)] = token_ids[static_cast<size_t>(b)];
            }

            if (params.arch == LLMArch::MISTRAL_SMALL_3_2 || params.arch == LLMArch::QWEN3) {
                decode_input_pos_vec.resize(1);
                decode_input_pos_vec[0] = static_cast<int>(n_past);
            } else {
                decode_input_pos_vec.resize(4);
                decode_input_pos_vec[0] = static_cast<int>(n_past);
                decode_input_pos_vec[1] = static_cast<int>(n_past);
                decode_input_pos_vec[2] = static_cast<int>(n_past);
                decode_input_pos_vec[3] = 0;
            }

            constexpr float kMaskNeg = -65504.0f;
            decode_attention_mask_vec.resize(static_cast<size_t>(kv_cache_capacity * batch_size));
            for (int64_t b = 0; b < batch_size; ++b) {
                const int pad_len = pad_lens[static_cast<size_t>(b)];
                size_t batch_offset = static_cast<size_t>(b * kv_cache_capacity);
                for (int64_t k = 0; k < kv_cache_capacity; ++k) {
                    float value = 0.f;
                    if (k < pad_len || k > n_past) {
                        value = kMaskNeg;
                    }
                    decode_attention_mask_vec[batch_offset + static_cast<size_t>(k)] = value;
                }
            }

            kv_row_indices_vec.resize(static_cast<size_t>(batch_size));
            for (int64_t b = 0; b < batch_size; ++b) {
                kv_row_indices_vec[static_cast<size_t>(b)] = n_past;
            }

            ggml_backend_tensor_set(decode_graph_state.input_ids, decode_input_ids_vec.data(), 0, ggml_nbytes(decode_graph_state.input_ids));
            ggml_backend_tensor_set(decode_graph_state.input_pos, decode_input_pos_vec.data(), 0, ggml_nbytes(decode_graph_state.input_pos));
            ggml_backend_tensor_set(decode_graph_state.attention_mask, decode_attention_mask_vec.data(), 0, ggml_nbytes(decode_graph_state.attention_mask));
            ggml_backend_tensor_set(decode_graph_state.kv_row_indices, kv_row_indices_vec.data(), 0, ggml_nbytes(decode_graph_state.kv_row_indices));

            if (ggml_backend_is_cpu(runtime_backend)) {
                ggml_backend_cpu_set_n_threads(runtime_backend, n_threads);
            }

            ggml_status status = ggml_backend_graph_compute(runtime_backend, decode_graph_state.graph);
            if (status != GGML_STATUS_SUCCESS) {
                LOG_ERROR("%s decode compute failed: %s", get_desc().c_str(), ggml_status_to_string(status));
                return false;
            }

            if (output != nullptr) {
                auto result = decode_graph_state.logits;
                if (*output == nullptr && output_ctx != nullptr) {
                    *output = ggml_dup_tensor(output_ctx, result);
                }
                if (*output != nullptr) {
                    ggml_ext_backend_tensor_get_and_sync(runtime_backend, result, (*output)->data, 0, ggml_nbytes(*output));
                }
            }
            return true;
        }

        struct ggml_cgraph* build_graph(struct ggml_tensor* input_ids,
                                        struct ggml_tensor* attention_mask,
                                        std::vector<std::pair<int, ggml_tensor*>> image_embeds,
                                        std::set<int> out_layers) {
            struct ggml_cgraph* gf = ggml_new_graph(compute_ctx);

            input_ids = to_backend(input_ids);

            for (auto& image_embed : image_embeds) {
                image_embed.second = to_backend(image_embed.second);
            }

            int64_t n_tokens = input_ids->ne[0];
            if (params.arch == LLMArch::MISTRAL_SMALL_3_2 || params.arch == LLMArch::QWEN3) {
                input_pos_vec.resize(n_tokens);
                for (int i = 0; i < n_tokens; ++i) {
                    input_pos_vec[i] = i;
                }
            } else {
                input_pos_vec.resize(n_tokens * 4);
                for (int i = 0; i < n_tokens; ++i) {
                    input_pos_vec[i]                = i;
                    input_pos_vec[n_tokens + i]     = i;
                    input_pos_vec[2 * n_tokens + i] = i;
                    input_pos_vec[3 * n_tokens + i] = 0;
                }
            }

            auto input_pos = ggml_new_tensor_1d(compute_ctx,
                                                GGML_TYPE_I32,
                                                input_pos_vec.size());
            set_backend_tensor_data(input_pos, input_pos_vec.data());

            if (attention_mask != nullptr) {
                attention_mask = to_backend(attention_mask);
            } else {
                constexpr float kMaskNeg = -65504.0f;
                attention_mask_vec.resize(n_tokens * n_tokens);
                for (int i0 = 0; i0 < n_tokens; i0++) {
                    for (int i1 = 0; i1 < n_tokens; i1++) {
                        float value = 0.f;
                        if (i0 > i1) {
                            value = kMaskNeg;
                        }
                        attention_mask_vec[i1 * n_tokens + i0] = value;
                    }
                }
                attention_mask = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, n_tokens, n_tokens);
                set_backend_tensor_data(attention_mask, attention_mask_vec.data());
            }

            auto runner_ctx = get_context();

            struct ggml_tensor* hidden_states = forward(&runner_ctx, input_ids, input_pos, attention_mask, image_embeds, out_layers);

            ggml_build_forward_expand(gf, input_pos);
            if (attention_mask != nullptr) {
                ggml_build_forward_expand(gf, attention_mask);
            }
            ggml_build_forward_expand(gf, hidden_states);

            return gf;
        }

        struct ggml_cgraph* build_graph_logits(struct ggml_tensor* input_ids,
                                               struct ggml_tensor* attention_mask,
                                               std::vector<std::pair<int, ggml_tensor*>> image_embeds) {
            struct ggml_cgraph* gf = ggml_new_graph(compute_ctx);

            input_ids = to_backend(input_ids);

            for (auto& image_embed : image_embeds) {
                image_embed.second = to_backend(image_embed.second);
            }

            int64_t n_tokens = input_ids->ne[0];
            if (params.arch == LLMArch::MISTRAL_SMALL_3_2 || params.arch == LLMArch::QWEN3) {
                input_pos_vec.resize(n_tokens);
                for (int i = 0; i < n_tokens; ++i) {
                    input_pos_vec[i] = i;
                }
            } else {
                input_pos_vec.resize(n_tokens * 4);
                for (int i = 0; i < n_tokens; ++i) {
                    input_pos_vec[i]                = i;
                    input_pos_vec[n_tokens + i]     = i;
                    input_pos_vec[2 * n_tokens + i] = i;
                    input_pos_vec[3 * n_tokens + i] = 0;
                }
            }

            auto input_pos = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_I32, input_pos_vec.size());
            set_backend_tensor_data(input_pos, input_pos_vec.data());

            if (attention_mask != nullptr) {
                attention_mask = to_backend(attention_mask);
            } else {
                constexpr float kMaskNeg = -65504.0f;
                attention_mask_vec.resize(n_tokens * n_tokens);
                for (int i0 = 0; i0 < n_tokens; i0++) {
                    for (int i1 = 0; i1 < n_tokens; i1++) {
                        float value = 0.f;
                        if (i0 > i1) {
                            value = kMaskNeg;
                        }
                        attention_mask_vec[i1 * n_tokens + i0] = value;
                    }
                }
                attention_mask = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, n_tokens, n_tokens);
                set_backend_tensor_data(attention_mask, attention_mask_vec.data());
            }

            auto runner_ctx = get_context();
            std::set<int> out_layers;
            struct ggml_tensor* hidden_states = forward(&runner_ctx, input_ids, input_pos, attention_mask, image_embeds, out_layers);

            auto last = ggml_ext_slice(compute_ctx, hidden_states, 1, n_tokens - 1, n_tokens, true);

            auto weight = get_embedding_weight();
            GGML_ASSERT(weight != nullptr);
            int64_t start = std::max<int64_t>(0, logits_range_start);
            int64_t end   = logits_range_end > 0 ? std::min<int64_t>(logits_range_end, weight->ne[1]) : weight->ne[1];
            if (end <= start) {
                start = 0;
                end   = weight->ne[1];
            }
            auto weight_slice = ggml_ext_slice(compute_ctx, weight, 1, start, end);
            auto logits       = ggml_mul_mat(compute_ctx, weight_slice, last);

            ggml_build_forward_expand(gf, input_pos);
            if (attention_mask != nullptr) {
                ggml_build_forward_expand(gf, attention_mask);
            }
            ggml_build_forward_expand(gf, logits);

            return gf;
        }

        struct ggml_cgraph* build_graph_logits_kv(struct ggml_tensor* input_ids,
                                                  struct ggml_tensor* attention_mask,
                                                  std::vector<std::pair<int, ggml_tensor*>> image_embeds,
                                                  int64_t n_past) {
            struct ggml_cgraph* gf = ggml_new_graph(compute_ctx);

            input_ids = to_backend(input_ids);
            for (auto& image_embed : image_embeds) {
                image_embed.second = to_backend(image_embed.second);
            }

            const int64_t n_tokens = input_ids->ne[0];
            const int64_t batch_size = std::max<int64_t>(1, input_ids->ne[1]);
            const int64_t n_kv = n_past + n_tokens;

            if (!ensure_kv_cache(n_kv, batch_size)) {
                LOG_ERROR("%s failed to allocate kv cache (tokens=%" PRId64 ", batch=%" PRId64 ")",
                          get_desc().c_str(),
                          n_kv,
                          batch_size);
                return gf;
            }

            if (params.arch == LLMArch::MISTRAL_SMALL_3_2 || params.arch == LLMArch::QWEN3) {
                input_pos_vec.resize(n_tokens);
                for (int64_t i = 0; i < n_tokens; ++i) {
                    input_pos_vec[i] = static_cast<int>(n_past + i);
                }
            } else {
                input_pos_vec.resize(n_tokens * 4);
                for (int64_t i = 0; i < n_tokens; ++i) {
                    int p                           = static_cast<int>(n_past + i);
                    input_pos_vec[i]                = p;
                    input_pos_vec[n_tokens + i]     = p;
                    input_pos_vec[2 * n_tokens + i] = p;
                    input_pos_vec[3 * n_tokens + i] = 0;
                }
            }

            auto input_pos = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_I32, input_pos_vec.size());
            set_backend_tensor_data(input_pos, input_pos_vec.data());

            if (attention_mask != nullptr) {
                attention_mask = to_backend(attention_mask);
            } else {
                constexpr float kMaskNeg = -65504.0f;
                attention_mask_vec.resize(static_cast<size_t>(n_kv * n_tokens * batch_size));
                for (int64_t b = 0; b < batch_size; ++b) {
                    size_t batch_offset = static_cast<size_t>(b * n_kv * n_tokens);
                    for (int64_t q = 0; q < n_tokens; ++q) {
                        const int64_t abs_q = n_past + q;
                        for (int64_t k = 0; k < n_kv; ++k) {
                            float value = 0.f;
                            if (k > abs_q) {
                                value = kMaskNeg;
                            }
                            attention_mask_vec[batch_offset + q * n_kv + k] = value;
                        }
                    }
                }
                if (batch_size == 1) {
                    attention_mask = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, n_kv, n_tokens);
                } else {
                    attention_mask = ggml_new_tensor_3d(compute_ctx, GGML_TYPE_F32, n_kv, n_tokens, batch_size);
                }
                set_backend_tensor_data(attention_mask, attention_mask_vec.data());
            }

            kv_row_indices_vec.resize(static_cast<size_t>(n_tokens * batch_size));
            for (int64_t b = 0; b < batch_size; ++b) {
                size_t batch_offset = static_cast<size_t>(b * n_tokens);
                for (int64_t t = 0; t < n_tokens; ++t) {
                    kv_row_indices_vec[batch_offset + t] = n_past + t;
                }
            }
            auto kv_row_indices = ggml_new_tensor_3d(compute_ctx, GGML_TYPE_I64, n_tokens, batch_size, 1);
            set_backend_tensor_data(kv_row_indices, kv_row_indices_vec.data());

            std::vector<ggml_tensor*> past_k_cache(params.num_layers, nullptr);
            std::vector<ggml_tensor*> past_v_cache(params.num_layers, nullptr);
            for (int i = 0; i < params.num_layers; ++i) {
                std::string k_name = "llm.k." + std::to_string(i);
                std::string v_name = "llm.v." + std::to_string(i);
                auto k_cache       = get_cache_tensor_by_name(k_name);
                auto v_cache       = get_cache_tensor_by_name(v_name);
                if (k_cache == nullptr || v_cache == nullptr) {
                    LOG_ERROR("%s kv cache tensor missing for layer %d", get_desc().c_str(), i);
                    return gf;
                }
                past_k_cache[i] = k_cache;
                past_v_cache[i] = v_cache;
            }

            auto runner_ctx = get_context();
            std::set<int> out_layers;
            struct ggml_tensor* hidden_states = forward(&runner_ctx,
                                                        input_ids,
                                                        input_pos,
                                                        attention_mask,
                                                        image_embeds,
                                                        out_layers,
                                                        &past_k_cache,
                                                        &past_v_cache,
                                                        kv_row_indices,
                                                        n_kv);

            auto last = ggml_ext_slice(compute_ctx, hidden_states, 1, n_tokens - 1, n_tokens, true);
            auto weight = get_embedding_weight();
            GGML_ASSERT(weight != nullptr);
            int64_t start = std::max<int64_t>(0, logits_range_start);
            int64_t end   = logits_range_end > 0 ? std::min<int64_t>(logits_range_end, weight->ne[1]) : weight->ne[1];
            if (end <= start) {
                start = 0;
                end   = weight->ne[1];
            }
            auto weight_slice = ggml_ext_slice(compute_ctx, weight, 1, start, end);
            auto logits       = ggml_mul_mat(compute_ctx, weight_slice, last);

            ggml_build_forward_expand(gf, input_pos);
            ggml_build_forward_expand(gf, attention_mask);
            ggml_build_forward_expand(gf, kv_row_indices);
            ggml_build_forward_expand(gf, logits);

            return gf;
        }

        bool compute(const int n_threads,
                     struct ggml_tensor* input_ids,
                     struct ggml_tensor* attention_mask,
                     std::vector<std::pair<int, ggml_tensor*>> image_embeds,
                     std::set<int> out_layers,
                     ggml_tensor** output,
                     ggml_context* output_ctx = nullptr) {
            invalidate_decode_graph_state();
            auto get_graph = [&]() -> struct ggml_cgraph* {
                return build_graph(input_ids, attention_mask, image_embeds, out_layers);
            };
            return GGMLRunner::compute(get_graph, n_threads, true, output, output_ctx);
        }

        bool compute_logits(const int n_threads,
                            struct ggml_tensor* input_ids,
                            struct ggml_tensor* attention_mask,
                            std::vector<std::pair<int, ggml_tensor*>> image_embeds,
                            ggml_tensor** output,
                            ggml_context* output_ctx = nullptr) {
            invalidate_decode_graph_state();
            auto get_graph = [&]() -> struct ggml_cgraph* {
                return build_graph_logits(input_ids, attention_mask, image_embeds);
            };
            return GGMLRunner::compute(get_graph, n_threads, true, output, output_ctx);
        }

        bool compute_logits_kv(const int n_threads,
                               struct ggml_tensor* input_ids,
                               struct ggml_tensor* attention_mask,
                               std::vector<std::pair<int, ggml_tensor*>> image_embeds,
                               int64_t n_past,
                               ggml_tensor** output,
                               ggml_context* output_ctx = nullptr) {
            invalidate_decode_graph_state();
            auto get_graph = [&]() -> struct ggml_cgraph* {
                return build_graph_logits_kv(input_ids, attention_mask, image_embeds, n_past);
            };
            return GGMLRunner::compute(get_graph, n_threads, true, output, output_ctx);
        }

        bool prepare_kv_cache(int64_t max_tokens, int64_t batch_size) {
            invalidate_decode_graph_state();
            if (!ensure_kv_cache(max_tokens, batch_size)) {
                return false;
            }
            if (cache_buffer != nullptr) {
                ggml_backend_buffer_clear(cache_buffer, 0);
            }
            return true;
        }

        void reset_kv_cache() {
            invalidate_decode_graph_state();
            free_cache_ctx_and_buffer();
            kv_cache_capacity   = 0;
            kv_cache_batch_size = 0;
        }

        int64_t get_num_image_tokens(int64_t t, int64_t h, int64_t w) {
            int64_t grid_t     = 1;
            int64_t grid_h     = h / params.vision.patch_size;
            int64_t grid_w     = w / params.vision.patch_size;
            int64_t llm_grid_h = grid_h / params.vision.spatial_merge_size;
            int64_t llm_grid_w = grid_w / params.vision.spatial_merge_size;
            return grid_t * grid_h * grid_w;
        }

        struct ggml_tensor* process_image(struct ggml_context* ctx, struct ggml_tensor* image) {
            // image: [C, H, W]
            // return: [grid_t*(H/mh/ph)*(W/mw/pw)*mh*mw, C*pt*ph*pw], grid_t == 1
            int64_t C  = image->ne[2];
            int64_t H  = image->ne[1];
            int64_t W  = image->ne[0];
            int64_t mh = params.vision.spatial_merge_size;
            int64_t mw = params.vision.spatial_merge_size;
            int64_t pt = params.vision.temporal_patch_size;
            int64_t ph = params.vision.patch_size;
            int64_t pw = params.vision.patch_size;

            image = ggml_reshape_4d(ctx, image, pw, mw, (W / mw / pw), H * C);                               // [C*H, (W/mw/pw), mw, pw]
            image = ggml_cont(ctx, ggml_ext_torch_permute(ctx, image, 0, 2, 3, 1));                          // [mw, C*H, (W/mw/pw), pw]
            image = ggml_reshape_4d(ctx, image, pw * (W / mw / pw), H, C, mw);                               // [mw, C, H, (W/mw/pw)*pw]
            image = ggml_cont(ctx, ggml_ext_torch_permute(ctx, image, 0, 2, 3, 1));                          // [H, mw, C, (W/mw/pw)*pw]
            image = ggml_reshape_4d(ctx, image, pw, (W / mw / pw) * C * mw, ph, mh * (H / mh / ph));         // [(H/mh/ph)*mh, ph, mw*C*(W/mw/pw), pw]
            image = ggml_cont(ctx, ggml_ext_torch_permute(ctx, image, 0, 2, 1, 3));                          // [(H/mh/ph)*mh, mw*C*(W/mw/pw), ph, pw]
            image = ggml_reshape_4d(ctx, image, pw * ph, (W / mw / pw), C, mw * mh * (H / mh / ph));         // [(H/mh/ph)*mh*mw, C, (W/mw/pw), ph*pw]
            image = ggml_concat(ctx, image, image, 0);                                                       // [(H/mh/ph)*mh*mw, C, (W/mw/pw), pt*ph*pw]
            image = ggml_cont(ctx, ggml_ext_torch_permute(ctx, image, 0, 2, 1, 3));                          // [(H/mh/ph)*mh*mw, (W/mw/pw), C, pt*ph*pw]
            image = ggml_reshape_4d(ctx, image, pw * ph * pt * C, (W / mw / pw), mw * mh, (H / mh / ph));    // [(H/mh/ph), mh*mw, (W/mw/pw), C*pt*ph*pw]
            image = ggml_cont(ctx, ggml_ext_torch_permute(ctx, image, 0, 2, 1, 3));                          // [(H/mh/ph), (W/mw/pw), mh*mw, C*pt*ph*pw]
            image = ggml_reshape_2d(ctx, image, pw * ph * pt * C, mw * mh * (W / mw / pw) * (H / mh / ph));  // [(H/mh/ph)*(W/mw/pw)*mh*mw, C*pt*ph*pw]
            return image;
        }

        struct ggml_cgraph* build_encode_image_graph(struct ggml_tensor* image) {
            struct ggml_cgraph* gf = new_graph_custom(LLM_GRAPH_SIZE);

            GGML_ASSERT(image->ne[1] % (params.vision.patch_size * params.vision.spatial_merge_size) == 0);
            GGML_ASSERT(image->ne[0] % (params.vision.patch_size * params.vision.spatial_merge_size) == 0);

            int grid_t                 = 1;
            int grid_h                 = static_cast<int>(image->ne[1]) / params.vision.patch_size;
            int grid_w                 = static_cast<int>(image->ne[0]) / params.vision.patch_size;
            int llm_grid_h             = grid_h / params.vision.spatial_merge_size;
            int llm_grid_w             = grid_w / params.vision.spatial_merge_size;
            int vit_merger_window_size = params.vision.window_size / params.vision.patch_size / params.vision.spatial_merge_size;

            image = to_backend(image);

            auto pixel_values = process_image(compute_ctx, image);

            // window index
            int inverse_index = 0;
            window_index_vec.resize(llm_grid_h * llm_grid_w);
            window_inverse_index_vec.resize(llm_grid_h * llm_grid_w);
            std::vector<int> seqlens;
            for (int ih = 0; ih < llm_grid_h; ih += vit_merger_window_size) {
                for (int iw = 0; iw < llm_grid_w; iw += vit_merger_window_size) {
                    int win_h = std::min(vit_merger_window_size, llm_grid_h - ih);
                    int win_w = std::min(vit_merger_window_size, llm_grid_w - iw);
                    for (int iy = 0; iy < win_h; iy++) {
                        for (int ix = 0; ix < win_w; ix++) {
                            int index                       = (ih + iy) * llm_grid_w + iw + ix;
                            window_index_vec[inverse_index] = index;
                            window_inverse_index_vec[index] = inverse_index;
                            inverse_index++;
                        }
                    }
                    seqlens.push_back(win_h * win_w * params.vision.spatial_merge_size * params.vision.spatial_merge_size);
                }
            }
            // printf("window_index: ");
            // for (int i : window_index_vec) {
            //     printf("%d ", i);
            // }
            // printf("\n");
            // printf("window_inverse_index: ");
            // for (int i : window_inverse_index_vec) {
            //     printf("%d ", i);
            // }
            // printf("\n");
            // printf("seqlens: ");
            // for (int i : seqlens) {
            //     printf("%d ", i);
            // }
            // printf("\n");
            auto window_index         = ggml_new_tensor_1d(compute_ctx,
                                                           GGML_TYPE_I32,
                                                           llm_grid_h * llm_grid_w);
            auto window_inverse_index = ggml_new_tensor_1d(compute_ctx,
                                                           GGML_TYPE_I32,
                                                           llm_grid_h * llm_grid_w);
            set_backend_tensor_data(window_index, window_index_vec.data());
            set_backend_tensor_data(window_inverse_index, window_inverse_index_vec.data());

            // window mask
            int seq_window_size = (vit_merger_window_size * params.vision.spatial_merge_size) * (vit_merger_window_size * params.vision.spatial_merge_size);
            window_mask_vec.resize((grid_h * grid_w) * (grid_h * grid_w));
            int window_start_index = 0;
            for (int seq_index = 0; seq_index < seqlens.size(); seq_index++) {
                int window_end_index = window_start_index + seqlens[seq_index];
                // LOG_DEBUG("%d %d", window_start_index, window_end_index);
                GGML_ASSERT(window_end_index <= grid_h * grid_w);
                for (int i = window_start_index; i < window_end_index; i++) {
                    for (int j = 0; j < grid_h * grid_w; j++) {
                        float mask_value = -INFINITY;
                        if (j >= window_start_index && j < window_end_index) {
                            mask_value = 0;
                        }
                        GGML_ASSERT((i * (grid_h * grid_w) + j) < window_mask_vec.size());
                        window_mask_vec[i * (grid_h * grid_w) + j] = mask_value;
                    }
                }
                window_start_index = window_end_index;
                // printf("\n");
            }
            // printf("window_mask: \n");
            // for (int i = 0; i < grid_h*grid_w; i++) {
            //     for (int j = 0; j < grid_h*grid_w; j++) {
            //         printf("%f ", window_mask_vec[i * (grid_h * grid_w) + j]);
            //     }
            //     printf("\n");
            // }
            auto window_mask = ggml_new_tensor_2d(compute_ctx,
                                                  GGML_TYPE_F32,
                                                  grid_h * grid_w,
                                                  grid_h * grid_w);
            set_backend_tensor_data(window_mask, window_mask_vec.data());

            // pe
            int head_dim = static_cast<int>(params.vision.hidden_size / params.vision.num_heads);
            pe_vec       = Rope::gen_qwen2vl_pe(grid_h,
                                                grid_w,
                                                params.vision.spatial_merge_size,
                                                window_inverse_index_vec,
                                                10000,
                                                {head_dim / 2, head_dim / 2});
            int pos_len  = static_cast<int>(pe_vec.size() / head_dim / 2);
            // LOG_DEBUG("pos_len %d", pos_len);
            auto pe = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, head_dim / 2, pos_len);
            // pe->data = pe_vec.data();
            // print_ggml_tensor(pe);
            // pe->data = nullptr;
            set_backend_tensor_data(pe, pe_vec.data());

            auto runnter_ctx                  = get_context();
            struct ggml_tensor* hidden_states = vision_forward(&runnter_ctx,
                                                               pixel_values,
                                                               pe,
                                                               window_index,
                                                               window_inverse_index,
                                                               window_mask);
            ggml_build_forward_expand(gf, hidden_states);

            return gf;
        }

        void encode_image(const int n_threads,
                          struct ggml_tensor* image,
                          ggml_tensor** output,
                          ggml_context* output_ctx = nullptr) {
            auto get_graph = [&]() -> struct ggml_cgraph* {
                return build_encode_image_graph(image);
            };
            GGMLRunner::compute(get_graph, n_threads, false, output, output_ctx);
        }
    };

    struct LLMEmbedder {
        std::shared_ptr<BPETokenizer> tokenizer;
        LLMRunner model;

        LLMEmbedder(LLMArch arch,
                    ggml_backend_t backend,
                    bool offload_params_to_cpu,
                    const String2TensorStorage& tensor_storage_map = {},
                    const std::string prefix                       = "",
                    bool enable_vision                             = false)
            : model(arch, backend, offload_params_to_cpu, tensor_storage_map, prefix, enable_vision) {
            if (arch == LLMArch::MISTRAL_SMALL_3_2) {
                tokenizer = std::make_shared<MistralTokenizer>();
            } else if (arch == LLMArch::QWEN3) {
                tokenizer = std::make_shared<Qwen2Tokenizer>();
            } else {
                tokenizer = std::make_shared<Qwen2Tokenizer>();
            }
        }

        void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
            model.get_param_tensors(tensors, prefix);
        }

        void alloc_params_buffer() {
            model.alloc_params_buffer();
        }

        std::tuple<std::vector<int>, std::vector<float>> tokenize(std::string text,
                                                                  std::pair<int, int> attn_range,
                                                                  size_t max_length = 0,
                                                                  bool padding      = false) {
            std::vector<std::pair<std::string, float>> parsed_attention;
            parsed_attention.emplace_back(text.substr(0, attn_range.first), 1.f);
            if (attn_range.second - attn_range.first > 0) {
                auto new_parsed_attention = parse_prompt_attention(text.substr(attn_range.first, attn_range.second - attn_range.first));
                parsed_attention.insert(parsed_attention.end(),
                                        new_parsed_attention.begin(),
                                        new_parsed_attention.end());
            }
            parsed_attention.emplace_back(text.substr(attn_range.second), 1.f);
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
            //     std::cout << tokens[i] << ":" << weights[i] << ", ";
            // }
            // std::cout << std::endl;

            return {tokens, weights};
        }

        void test() {
            struct ggml_init_params params;
            params.mem_size   = static_cast<size_t>(1024 * 1024) * 1024;  // 1GB
            params.mem_buffer = nullptr;
            params.no_alloc   = false;

            struct ggml_context* work_ctx = ggml_init(params);
            GGML_ASSERT(work_ctx != nullptr);
            bool test_mistral          = false;
            bool test_qwen3            = true;
            bool test_vit              = false;
            bool test_decoder_with_vit = false;

            if (test_decoder_with_vit) {
                ggml_tensor* image_embed = nullptr;
                {
                    auto image = load_tensor_from_file(work_ctx, "qwen2vl_normalized.bin");
                    print_ggml_tensor(image, false, "image");
                    struct ggml_tensor* out = nullptr;

                    int64_t t0 = ggml_time_ms();
                    model.encode_image(8, image, &out, work_ctx);
                    int64_t t1 = ggml_time_ms();

                    print_ggml_tensor(out, false, "image_embed");
                    image_embed = out;
                    LOG_DEBUG("llm encode_image test done in %lldms", t1 - t0);
                }

                std::string placeholder  = "<|image_pad|>";
                std::string img_prompt   = "Picture 1: <|vision_start|>";  // [24669, 220, 16, 25, 220, 151652]
                int64_t num_image_tokens = image_embed->ne[1];
                img_prompt.reserve(num_image_tokens * placeholder.size());
                for (int i = 0; i < num_image_tokens; i++) {
                    img_prompt += placeholder;
                }
                img_prompt += "<|vision_end|>";

                std::vector<std::pair<int, ggml_tensor*>> image_embeds;
                image_embeds.emplace_back(64, image_embed);

                std::pair<int, int> prompt_attn_range;
                std::string text = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n";
                text += img_prompt;
                prompt_attn_range.first = static_cast<int>(text.size());
                text += "change 'flux.cpp' to 'edit.cpp'";
                prompt_attn_range.second = static_cast<int>(text.size());
                text += "<|im_end|>\n<|im_start|>assistant\n";

                auto tokens_and_weights     = tokenize(text, prompt_attn_range, 0, false);
                std::vector<int>& tokens    = std::get<0>(tokens_and_weights);
                std::vector<float>& weights = std::get<1>(tokens_and_weights);
                for (auto token : tokens) {
                    printf("%d ", token);
                }
                printf("\n");
                auto input_ids          = vector_to_ggml_tensor_i32(work_ctx, tokens);
                struct ggml_tensor* out = nullptr;

                int64_t t0 = ggml_time_ms();
                model.compute(8, input_ids, nullptr, image_embeds, {}, &out, work_ctx);
                int64_t t1 = ggml_time_ms();

                print_ggml_tensor(out);
                LOG_DEBUG("llm test done in %lldms", t1 - t0);
            } else if (test_vit) {
                // auto image = ggml_new_tensor_3d(work_ctx, GGML_TYPE_F32, 280, 280, 3);
                // ggml_set_f32(image, 0.f);
                auto image = load_tensor_from_file(work_ctx, "qwen2vl_normalized.bin");
                print_ggml_tensor(image, false, "image");
                struct ggml_tensor* out = nullptr;

                int64_t t0 = ggml_time_ms();
                model.encode_image(8, image, &out, work_ctx);
                int64_t t1 = ggml_time_ms();

                print_ggml_tensor(out, false, "out");

                // auto ref_out = load_tensor_from_file(work_ctx, "qwen2vl.bin");
                // ggml_ext_tensor_diff(ref_out, out, 0.01f);

                LOG_DEBUG("llm test done in %lldms", t1 - t0);
            } else if (test_mistral) {
                std::pair<int, int> prompt_attn_range;
                std::string text        = "[SYSTEM_PROMPT]You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object\nattribution and actions without speculation.[/SYSTEM_PROMPT][INST]";
                prompt_attn_range.first = static_cast<int>(text.size());
                text += "a lovely cat";
                prompt_attn_range.second = static_cast<int>(text.size());
                text += "[/INST]";
                auto tokens_and_weights     = tokenize(text, prompt_attn_range, 0, false);
                std::vector<int>& tokens    = std::get<0>(tokens_and_weights);
                std::vector<float>& weights = std::get<1>(tokens_and_weights);
                for (auto token : tokens) {
                    printf("%d ", token);
                }
                printf("\n");
                auto input_ids          = vector_to_ggml_tensor_i32(work_ctx, tokens);
                struct ggml_tensor* out = nullptr;

                int64_t t0 = ggml_time_ms();
                model.compute(8, input_ids, nullptr, {}, {10, 20, 30}, &out, work_ctx);
                int64_t t1 = ggml_time_ms();

                print_ggml_tensor(out);
                LOG_DEBUG("llm test done in %lldms", t1 - t0);
            } else if (test_qwen3) {
                std::pair<int, int> prompt_attn_range;
                std::string text        = "<|im_start|>user\n";
                prompt_attn_range.first = static_cast<int>(text.size());
                text += "a lovely cat";
                prompt_attn_range.second = static_cast<int>(text.size());
                text += "<|im_end|>\n<|im_start|>assistant\n";
                auto tokens_and_weights     = tokenize(text, prompt_attn_range, 0, false);
                std::vector<int>& tokens    = std::get<0>(tokens_and_weights);
                std::vector<float>& weights = std::get<1>(tokens_and_weights);
                for (auto token : tokens) {
                    printf("%d ", token);
                }
                printf("\n");
                auto input_ids          = vector_to_ggml_tensor_i32(work_ctx, tokens);
                struct ggml_tensor* out = nullptr;

                int64_t t0 = ggml_time_ms();
                model.compute(8, input_ids, nullptr, {}, {35}, &out, work_ctx);
                int64_t t1 = ggml_time_ms();

                print_ggml_tensor(out);
                LOG_DEBUG("llm test done in %lldms", t1 - t0);
            } else {
                std::pair<int, int> prompt_attn_range;
                std::string text        = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n";
                prompt_attn_range.first = static_cast<int>(text.size());
                text += "a lovely cat";
                prompt_attn_range.second = static_cast<int>(text.size());
                text += "<|im_end|>\n<|im_start|>assistant\n";
                auto tokens_and_weights     = tokenize(text, prompt_attn_range, 0, false);
                std::vector<int>& tokens    = std::get<0>(tokens_and_weights);
                std::vector<float>& weights = std::get<1>(tokens_and_weights);
                for (auto token : tokens) {
                    printf("%d ", token);
                }
                printf("\n");
                auto input_ids          = vector_to_ggml_tensor_i32(work_ctx, tokens);
                struct ggml_tensor* out = nullptr;

                int64_t t0 = ggml_time_ms();
                model.compute(8, input_ids, nullptr, {}, {}, &out, work_ctx);
                int64_t t1 = ggml_time_ms();

                print_ggml_tensor(out);
                LOG_DEBUG("llm test done in %lldms", t1 - t0);
            }
        }

        static void load_from_file_and_test(const std::string& file_path) {
            // cpu f16: pass
            // ggml_backend_t backend = ggml_backend_cuda_init(0);
            ggml_backend_t backend    = ggml_backend_cpu_init();
            ggml_type model_data_type = GGML_TYPE_COUNT;

            ModelLoader model_loader;
            if (!model_loader.init_from_file_and_convert_name(file_path, "text_encoders.llm.")) {
                LOG_ERROR("init model loader from file failed: '%s'", file_path.c_str());
                return;
            }

            auto& tensor_storage_map = model_loader.get_tensor_storage_map();
            if (model_data_type != GGML_TYPE_COUNT) {
                for (auto& [name, tensor_storage] : tensor_storage_map) {
                    if (ends_with(name, "weight")) {
                        tensor_storage.expected_type = model_data_type;
                    }
                }
            }

            LLMArch arch = LLMArch::QWEN3;

            std::shared_ptr<LLMEmbedder> llm = std::make_shared<LLMEmbedder>(arch,
                                                                             backend,
                                                                             true,
                                                                             tensor_storage_map,
                                                                             "text_encoders.llm",
                                                                             true);

            llm->alloc_params_buffer();
            std::map<std::string, ggml_tensor*> tensors;
            llm->get_param_tensors(tensors, "text_encoders.llm");

            bool success = model_loader.load_tensors(tensors);

            if (!success) {
                LOG_ERROR("load tensors from model loader failed");
                return;
            }

            LOG_INFO("llm model loaded");
            llm->test();
        }
    };
};  // LLM

#endif  // __LLM_HPP__
