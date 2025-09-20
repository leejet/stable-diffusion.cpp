#ifndef __QWEN_HPP__
#define __QWEN_HPP__

#include "ggml_extend.hpp"

#include <string>
#include <vector>
#include <map>
#include <set>
#include <regex>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <optional>
#include <iostream>

#include "json.hpp"
#include "clip.hpp"
#include "tokenize_util.h"

namespace Qwen {

class Qwen2Tokenizer {
private:
    std::map<int, std::u32string> byte_encoder;
    std::map<std::u32string, int> byte_decoder;
    std::map<std::u32string, int> encoder;
    std::map<int, std::u32string> decoder;
    std::map<std::pair<std::u32string, std::u32string>, int> bpe_ranks;
    std::regex pat;
    int encoder_len;
    int bpe_len;

public:
    const std::string UNK_TOKEN = "<|endoftext|>";
    const std::string EOS_TOKEN = "<|endoftext|>";
    const std::string PAD_TOKEN = "<|endoftext|>";

    const int UNK_TOKEN_ID = 151643;
    const int EOS_TOKEN_ID = 151643;
    const int PAD_TOKEN_ID = 151643;

private:
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

public:
    explicit Qwen2Tokenizer(const std::string& merges_utf8_str = "") {
        if (merges_utf8_str.size() > 0) {
            load_from_merges(merges_utf8_str);
        } else {
            load_from_merges(ModelLoader::load_qwen2_merges());
        }
    }

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
        // GGML_ASSERT(merges.size() == 48895);
        merges = std::vector<std::u32string>(merges.begin(), merges.end());
        std::vector<std::pair<std::u32string, std::u32string>> merge_pairs;
        for (const auto& merge : merges) {
            size_t space_pos = merge.find(' ');
            merge_pairs.emplace_back(merge.substr(0, space_pos), merge.substr(space_pos + 1));
            // LOG_DEBUG("%s", utf32_to_utf8(merge.substr(space_pos + 1)).c_str());
            // printf("%s :: %s | %s \n", utf32_to_utf8(merge).c_str(), utf32_to_utf8(merge.substr(0, space_pos)).c_str(),
            //                     utf32_to_utf8(merge.substr(space_pos + 1)).c_str());
        }

        std::vector<std::u32string> vocab;
        for (const auto& pair : byte_unicode_pairs) {
            vocab.push_back(pair.second);
        }
        for (const auto& merge : merge_pairs) {
            vocab.push_back(merge.first + merge.second);
        }
        vocab.push_back(utf8_to_utf32("<|endoftext|>"));
        vocab.push_back(utf8_to_utf32("<|im_start|>"));
        vocab.push_back(utf8_to_utf32("<|im_end|>"));
        vocab.push_back(utf8_to_utf32("<|object_ref_start|>"));
        vocab.push_back(utf8_to_utf32("<|object_ref_end|>"));
        vocab.push_back(utf8_to_utf32("<|box_start|>"));
        vocab.push_back(utf8_to_utf32("<|box_end|>"));
        vocab.push_back(utf8_to_utf32("<|quad_start|>"));
        vocab.push_back(utf8_to_utf32("<|quad_end|>"));
        vocab.push_back(utf8_to_utf32("<|vision_start|>"));
        vocab.push_back(utf8_to_utf32("<|vision_end|>"));
        vocab.push_back(utf8_to_utf32("<|vision_pad|>"));
        vocab.push_back(utf8_to_utf32("<|image_pad|>"));
        vocab.push_back(utf8_to_utf32("<|video_pad|>"));
        vocab.push_back(utf8_to_utf32("<tool_call>"));
        vocab.push_back(utf8_to_utf32("</tool_call>"));
        vocab.push_back(utf8_to_utf32("<|fim_prefix|>"));
        vocab.push_back(utf8_to_utf32("<|fim_middle|>"));
        vocab.push_back(utf8_to_utf32("<|fim_suffix|>"));
        vocab.push_back(utf8_to_utf32("<|fim_pad|>"));
        vocab.push_back(utf8_to_utf32("<|repo_name|>"));
        vocab.push_back(utf8_to_utf32("<|file_sep|>"));

        LOG_DEBUG("vocab size: %llu", vocab.size());
        int i = 0;
        for (const auto& token : vocab) {
            encoder[token] = i;
            decoder[i]     = token;
            i++;
        }
        encoder_len = i;

        int rank = 0;
        for (const auto& merge : merge_pairs) {
            bpe_ranks[merge] = rank++;
        }
        bpe_len = rank;
    };

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
                              on_new_token_cb_t on_new_token_cb,
                              size_t max_length = 0,
                              bool padding      = false) {
        std::vector<int32_t> tokens = encode(text, on_new_token_cb);

        if (max_length > 0) {
            tokens.resize(max_length);
            if (padding) {
                tokens.insert(tokens.end(), max_length - tokens.size(), PAD_TOKEN_ID);
            }
        }

        return tokens;
    }

    void pad_tokens(std::vector<int>& tokens,
                    std::vector<float>& weights,
                    size_t max_length = 0,
                    bool padding      = false) {
        if (max_length > 0 && padding) {
            size_t n = std::ceil(tokens.size() * 1.0 / max_length);
            if (n == 0) {
                n = 1;
            }
            size_t length = max_length * n;
            LOG_DEBUG("token length: %llu", length);
            tokens.insert(tokens.end(), length - tokens.size(), PAD_TOKEN_ID);
            weights.insert(weights.end(), length - weights.size(), 1.0);
        }
    }

    std::vector<int> encode(std::string text, on_new_token_cb_t on_new_token_cb) {
        std::string original_text = text;
        std::vector<int32_t> bpe_tokens;
        
        auto tokens = token_split(text);
        std::vector<std::string> token_strs;
        for (auto& token : tokens) {
            bool skip = on_new_token_cb(token, bpe_tokens);
            if (skip) {
                continue;
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

};



#endif // __QWEN_HPP__
