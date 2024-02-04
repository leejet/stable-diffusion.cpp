#ifndef __CLIP_HPP__
#define __CLIP_HPP__

#include "ggml_extend.hpp"
#include "model.h"

/*================================================== CLIPTokenizer ===================================================*/

std::pair<std::unordered_map<std::string, float>, std::string> extract_and_remove_lora(std::string text) {
    std::regex re("<lora:([^:]+):([^>]+)>");
    std::smatch matches;
    std::unordered_map<std::string, float> filename2multiplier;

    while (std::regex_search(text, matches, re)) {
        std::string filename = matches[1].str();
        float multiplier     = std::stof(matches[2].str());

        text = std::regex_replace(text, re, "", std::regex_constants::format_first_only);

        if (multiplier == 0.f) {
            continue;
        }

        if (filename2multiplier.find(filename) == filename2multiplier.end()) {
            filename2multiplier[filename] = multiplier;
        } else {
            filename2multiplier[filename] += multiplier;
        }
    }

    return std::make_pair(filename2multiplier, text);
}

const std::string UNK_TOKEN = "<|endoftext|>";
const std::string BOS_TOKEN = "<|startoftext|>";
const std::string EOS_TOKEN = "<|endoftext|>";
const std::string PAD_TOEKN = "<|endoftext|>";

const int UNK_TOKEN_ID = 49407;
const int BOS_TOKEN_ID = 49406;
const int EOS_TOKEN_ID = 49407;
const int PAD_TOKEN_ID = 49407;

std::vector<std::pair<int, std::u32string>> bytes_to_unicode() {
    std::vector<std::pair<int, std::u32string>> byte_unicode_pairs;
    std::set<int> byte_set;
    for (int b = static_cast<int>('!'); b <= static_cast<int>('~'); ++b) {
        byte_set.insert(b);
        byte_unicode_pairs.push_back(std::pair<int, std::u32string>(b, unicode_value_to_utf32(b)));
    }
    for (int b = 161; b <= 172; ++b) {
        byte_set.insert(b);
        byte_unicode_pairs.push_back(std::pair<int, std::u32string>(b, unicode_value_to_utf32(b)));
    }
    for (int b = 174; b <= 255; ++b) {
        byte_set.insert(b);
        byte_unicode_pairs.push_back(std::pair<int, std::u32string>(b, unicode_value_to_utf32(b)));
    }
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (byte_set.find(b) == byte_set.end()) {
            byte_unicode_pairs.push_back(std::pair<int, std::u32string>(b, unicode_value_to_utf32(n + 256)));
            ++n;
        }
    }
    // LOG_DEBUG("byte_unicode_pairs %d", byte_unicode_pairs.size());
    return byte_unicode_pairs;
}

// Ref: https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py

typedef std::function<bool(std::string&, std::vector<int32_t>&)> on_new_token_cb_t;

class CLIPTokenizer {
private:
    SDVersion version = VERSION_1_x;
    std::map<int, std::u32string> byte_encoder;
    std::map<std::u32string, int> encoder;
    std::map<std::pair<std::u32string, std::u32string>, int> bpe_ranks;
    std::regex pat;

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
    CLIPTokenizer(SDVersion version = VERSION_1_x)
        : version(version) {}

    void load_from_merges(const std::string& merges_utf8_str) {
        auto byte_unicode_pairs = bytes_to_unicode();
        byte_encoder            = std::map<int, std::u32string>(byte_unicode_pairs.begin(), byte_unicode_pairs.end());
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
        // LOG_DEBUG("merges size %llu", merges.size());
        GGML_ASSERT(merges.size() == 48895);
        merges = std::vector<std::u32string>(merges.begin() + 1, merges.end());
        std::vector<std::pair<std::u32string, std::u32string>> merge_pairs;
        for (const auto& merge : merges) {
            size_t space_pos = merge.find(' ');
            merge_pairs.emplace_back(merge.substr(0, space_pos), merge.substr(space_pos + 1));
            // LOG_DEBUG("%s", utf32_to_utf8(merge.substr(space_pos + 1)).c_str());
        }
        std::vector<std::u32string> vocab;
        for (const auto& pair : byte_unicode_pairs) {
            vocab.push_back(pair.second);
        }
        for (const auto& pair : byte_unicode_pairs) {
            vocab.push_back(pair.second + utf8_to_utf32("</w>"));
        }
        for (const auto& merge : merge_pairs) {
            vocab.push_back(merge.first + merge.second);
        }
        vocab.push_back(utf8_to_utf32("<|startoftext|>"));
        vocab.push_back(utf8_to_utf32("<|endoftext|>"));
        LOG_DEBUG("vocab size: %llu", vocab.size());
        int i = 0;
        for (const auto& token : vocab) {
            encoder[token] = i++;
        }

        int rank = 0;
        for (const auto& merge : merge_pairs) {
            bpe_ranks[merge] = rank++;
        }
    };

    std::u32string bpe(const std::u32string& token) {
        std::vector<std::u32string> word;

        for (int i = 0; i < token.size() - 1; i++) {
            word.emplace_back(1, token[i]);
        }
        word.push_back(token.substr(token.size() - 1) + utf8_to_utf32("</w>"));

        std::set<std::pair<std::u32string, std::u32string>> pairs = get_pairs(word);

        if (pairs.empty()) {
            return token + utf8_to_utf32("</w>");
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
        tokens.insert(tokens.begin(), BOS_TOKEN_ID);
        if (max_length > 0) {
            if (tokens.size() > max_length - 1) {
                tokens.resize(max_length - 1);
                tokens.push_back(EOS_TOKEN_ID);
            } else {
                tokens.push_back(EOS_TOKEN_ID);
                if (padding) {
                    int pad_token_id = PAD_TOKEN_ID;
                    if (version == VERSION_2_x) {
                        pad_token_id = 0;
                    }
                    tokens.insert(tokens.end(), max_length - tokens.size(), pad_token_id);
                }
            }
        }
        return tokens;
    }

    std::vector<int> encode(std::string text, on_new_token_cb_t on_new_token_cb) {
        std::string original_text = text;
        std::vector<int32_t> bpe_tokens;
        text = whitespace_clean(text);
        std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return std::tolower(c); });

        std::regex pat(R"(<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[[:alpha:]]+|[[:digit:]]|[^[:space:][:alpha:][:digit:]]+)",
                       std::regex::icase);

        std::smatch matches;
        std::string str = text;
        std::vector<std::string> token_strs;
        while (std::regex_search(str, matches, pat)) {
            bool skip = on_new_token_cb(str, bpe_tokens);
            if (skip) {
                continue;
            }
            for (auto& token : matches) {
                std::string token_str = token.str();
                std::u32string utf32_token;
                for (int i = 0; i < token_str.length(); i++) {
                    char b = token_str[i];
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
            str = matches.suffix();
        }
        std::stringstream ss;
        ss << "[";
        for (auto token : token_strs) {
            ss << "\"" << token << "\", ";
        }
        ss << "]";
        LOG_DEBUG("split prompt \"%s\" to tokens %s", original_text.c_str(), ss.str().c_str());
        return bpe_tokens;
    }
};

// Ref: https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/cad87bf4e3e0b0a759afa94e933527c3123d59bc/modules/prompt_parser.py#L345
//
// Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
// Accepted tokens are:
//   (abc) - increases attention to abc by a multiplier of 1.1
//   (abc:3.12) - increases attention to abc by a multiplier of 3.12
//   [abc] - decreases attention to abc by a multiplier of 1.1
//   \( - literal character '('
//   \[ - literal character '['
//   \) - literal character ')'
//   \] - literal character ']'
//   \\ - literal character '\'
//   anything else - just text
//
// >>> parse_prompt_attention('normal text')
// [['normal text', 1.0]]
// >>> parse_prompt_attention('an (important) word')
// [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
// >>> parse_prompt_attention('(unbalanced')
// [['unbalanced', 1.1]]
// >>> parse_prompt_attention('\(literal\]')
// [['(literal]', 1.0]]
// >>> parse_prompt_attention('(unnecessary)(parens)')
// [['unnecessaryparens', 1.1]]
// >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
// [['a ', 1.0],
//  ['house', 1.5730000000000004],
//  [' ', 1.1],
//  ['on', 1.0],
//  [' a ', 1.1],
//  ['hill', 0.55],
//  [', sun, ', 1.1],
//  ['sky', 1.4641000000000006],
//  ['.', 1.1]]
std::vector<std::pair<std::string, float>> parse_prompt_attention(const std::string& text) {
    std::vector<std::pair<std::string, float>> res;
    std::vector<int> round_brackets;
    std::vector<int> square_brackets;

    float round_bracket_multiplier  = 1.1f;
    float square_bracket_multiplier = 1 / 1.1f;

    std::regex re_attention(R"(\\\(|\\\)|\\\[|\\\]|\\\\|\\|\(|\[|:([+-]?[.\d]+)\)|\)|\]|[^\\()\[\]:]+|:)");
    std::regex re_break(R"(\s*\bBREAK\b\s*)");

    auto multiply_range = [&](int start_position, float multiplier) {
        for (int p = start_position; p < res.size(); ++p) {
            res[p].second *= multiplier;
        }
    };

    std::smatch m;
    std::string remaining_text = text;

    while (std::regex_search(remaining_text, m, re_attention)) {
        std::string text   = m[0];
        std::string weight = m[1];

        if (text == "(") {
            round_brackets.push_back((int)res.size());
        } else if (text == "[") {
            square_brackets.push_back((int)res.size());
        } else if (!weight.empty()) {
            if (!round_brackets.empty()) {
                multiply_range(round_brackets.back(), std::stof(weight));
                round_brackets.pop_back();
            }
        } else if (text == ")" && !round_brackets.empty()) {
            multiply_range(round_brackets.back(), round_bracket_multiplier);
            round_brackets.pop_back();
        } else if (text == "]" && !square_brackets.empty()) {
            multiply_range(square_brackets.back(), square_bracket_multiplier);
            square_brackets.pop_back();
        } else if (text == "\\(") {
            res.push_back({text.substr(1), 1.0f});
        } else {
            res.push_back({text, 1.0f});
        }

        remaining_text = m.suffix();
    }

    for (int pos : round_brackets) {
        multiply_range(pos, round_bracket_multiplier);
    }

    for (int pos : square_brackets) {
        multiply_range(pos, square_bracket_multiplier);
    }

    if (res.empty()) {
        res.push_back({"", 1.0f});
    }

    int i = 0;
    while (i + 1 < res.size()) {
        if (res[i].second == res[i + 1].second) {
            res[i].first += res[i + 1].first;
            res.erase(res.begin() + i + 1);
        } else {
            ++i;
        }
    }

    return res;
}

/*================================================ FrozenCLIPEmbedder ================================================*/

struct ResidualAttentionBlock {
    int32_t n_head;
    int32_t d_model;
    int32_t hidden_size;  // n_head * d_model
    int32_t intermediate_size;

    // attention
    struct ggml_tensor* q_w;  // [hidden_size, hidden_size]
    struct ggml_tensor* q_b;  // [hidden_size, ]
    struct ggml_tensor* k_w;  // [hidden_size, hidden_size]
    struct ggml_tensor* k_b;  // [hidden_size, ]
    struct ggml_tensor* v_w;  // [hidden_size, hidden_size]
    struct ggml_tensor* v_b;  // [hidden_size, ]

    struct ggml_tensor* out_w;  // [hidden_size, hidden_size]
    struct ggml_tensor* out_b;  // [hidden_size, ]

    // layer norm 1
    struct ggml_tensor* ln1_w;  // [hidden_size, ]
    struct ggml_tensor* ln1_b;  // [hidden_size, ]

    // mlp
    struct ggml_tensor* fc1_w;  // [intermediate_size, hidden_size]
    struct ggml_tensor* fc1_b;  // [intermediate_size, ]

    struct ggml_tensor* fc2_w;  // [hidden_size, intermediate_size]
    struct ggml_tensor* fc2_b;  // [hidden_size, ]

    // layer norm 2
    struct ggml_tensor* ln2_w;  // [hidden_size, ]
    struct ggml_tensor* ln2_b;  // [hidden_size, ]

    size_t calculate_mem_size(ggml_type wtype) {
        size_t mem_size = 0;
        mem_size += 4 * ggml_row_size(wtype, hidden_size * hidden_size);        // q_w/k_w/v_w/out_w
        mem_size += 8 * ggml_row_size(GGML_TYPE_F32, hidden_size);              // q_b/k_b/v_b/out_b/ln1_w/ln1_b/ln2_w/ln2_b
        mem_size += 2 * ggml_row_size(wtype, hidden_size * intermediate_size);  // fc1_w/fc2_w
        mem_size += ggml_row_size(GGML_TYPE_F32, intermediate_size);            // fc1_b
        mem_size += ggml_row_size(GGML_TYPE_F32, hidden_size);                  // fc2_b
        return mem_size;
    }

    void init_params(struct ggml_context* ctx, ggml_allocr* alloc, ggml_type wtype) {
        ln1_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
        ln1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);

        q_w = ggml_new_tensor_2d(ctx, wtype, hidden_size, hidden_size);
        q_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
        k_w = ggml_new_tensor_2d(ctx, wtype, hidden_size, hidden_size);
        k_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
        v_w = ggml_new_tensor_2d(ctx, wtype, hidden_size, hidden_size);
        v_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);

        out_w = ggml_new_tensor_2d(ctx, wtype, hidden_size, hidden_size);
        out_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);

        fc1_w = ggml_new_tensor_2d(ctx, wtype, hidden_size, intermediate_size);
        fc1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, intermediate_size);

        fc2_w = ggml_new_tensor_2d(ctx, wtype, intermediate_size, hidden_size);
        fc2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);

        ln2_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
        ln2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
    }

    void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        tensors[prefix + "self_attn.q_proj.weight"]   = q_w;
        tensors[prefix + "self_attn.q_proj.bias"]     = q_b;
        tensors[prefix + "self_attn.k_proj.weight"]   = k_w;
        tensors[prefix + "self_attn.k_proj.bias"]     = k_b;
        tensors[prefix + "self_attn.v_proj.weight"]   = v_w;
        tensors[prefix + "self_attn.v_proj.bias"]     = v_b;
        tensors[prefix + "self_attn.out_proj.weight"] = out_w;
        tensors[prefix + "self_attn.out_proj.bias"]   = out_b;

        tensors[prefix + "layer_norm1.weight"] = ln1_w;
        tensors[prefix + "layer_norm1.bias"]   = ln1_b;

        tensors[prefix + "layer_norm2.weight"] = ln2_w;
        tensors[prefix + "layer_norm2.bias"]   = ln2_b;

        tensors[prefix + "mlp.fc1.weight"] = fc1_w;
        tensors[prefix + "mlp.fc1.bias"]   = fc1_b;

        tensors[prefix + "mlp.fc2.weight"] = fc2_w;
        tensors[prefix + "mlp.fc2.bias"]   = fc2_b;
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [N, n_token, hidden_size]
        int64_t N           = x->ne[2];
        int64_t n_token     = x->ne[1];
        int64_t hidden_size = n_head * d_model;

        struct ggml_tensor* r = x;

        // layer norm 1
        x = ggml_nn_layer_norm(ctx, x, ln1_w, ln1_b);
        // self-attention
        {
            struct ggml_tensor* q = ggml_nn_linear(ctx, x, q_w, q_b);
            q                     = ggml_scale_inplace(ctx, q, 1.0f / sqrt((float)d_model));
            q                     = ggml_reshape_4d(ctx, q, d_model, n_head, n_token, N);   // [N, n_token, n_head, d_model]
            q                     = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));       // [N, n_head, n_token, d_model]
            q                     = ggml_reshape_3d(ctx, q, d_model, n_token, n_head * N);  // [N * n_head, n_token, d_model]

            struct ggml_tensor* k = ggml_nn_linear(ctx, x, k_w, k_b);
            k                     = ggml_reshape_4d(ctx, k, d_model, n_head, n_token, N);  // [N, n_token, n_head, d_model]
            k                     = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));      // [N, n_head, n_token, d_model]
            k                     = ggml_reshape_3d(ctx, k, d_model, n_token, n_head);     // [N * n_head, n_token, d_model]

            struct ggml_tensor* v = ggml_nn_linear(ctx, x, v_w, v_b);
            v                     = ggml_reshape_4d(ctx, v, d_model, n_head, n_token, N);   // [N, n_token, n_head, d_model]
            v                     = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));       // [N, n_head, d_model, n_token]
            v                     = ggml_reshape_3d(ctx, v, n_token, d_model, n_head * N);  // [N * n_head, d_model, n_token]

            struct ggml_tensor* kq = ggml_mul_mat(ctx, k, q);  // [N * n_head, n_token, n_token]

            kq = ggml_diag_mask_inf_inplace(ctx, kq, 0);
            kq = ggml_soft_max_inplace(ctx, kq);

            struct ggml_tensor* kqv = ggml_mul_mat(ctx, v, kq);  // [N * n_head, n_token, d_model]
            kqv                     = ggml_reshape_4d(ctx, kqv, d_model, n_token, n_head, N);
            kqv                     = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));  // [N, n_token, n_head, d_model]

            x = ggml_reshape_2d(ctx, kqv, d_model * n_head, n_token * N);  // // [N * n_token, d_model * n_head]
        }

        // attention output
        x = ggml_nn_linear(ctx, x, out_w, out_b);

        // residual
        x = ggml_add(ctx, x, r);
        r = x;

        // layer norm 2
        x = ggml_nn_layer_norm(ctx, x, ln2_w, ln2_b);

        // mlp
        x = ggml_nn_linear(ctx, x, fc1_w, fc1_b);

        if (hidden_size == 1024 || hidden_size == 1280) {  // SD 2.x
            x = ggml_gelu_inplace(ctx, x);
        } else {  // SD 1.x
            x = ggml_gelu_quick_inplace(ctx, x);
        }

        x = ggml_nn_linear(ctx, x, fc2_w, fc2_b);

        // residual 2
        x = ggml_add(ctx, x, r);
        return x;
    }
};

// OPENAI_CLIP_VIT_L_14: https://huggingface.co/openai/clip-vit-large-patch14/blob/main/config.json
// OPEN_CLIP_VIT_H_14: https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/blob/main/config.json
// OPEN_CLIP_VIT_BIGG_14: https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k/blob/main/config.json (CLIPTextModelWithProjection)
// SDXL CLIPModel
// CLIPTextModelWithProjection seems optional

enum CLIPVersion {
    OPENAI_CLIP_VIT_L_14,   // SD 1.x and SDXL
    OPEN_CLIP_VIT_H_14,     // SD 2.x
    OPEN_CLIP_VIT_BIGG_14,  // SDXL
};

struct CLIPTextModel {
    CLIPVersion version = OPENAI_CLIP_VIT_L_14;
    // network hparams
    int32_t vocab_size              = 49408;
    int32_t max_position_embeddings = 77;
    int32_t hidden_size             = 768;   // 1024 for OPEN_CLIP_VIT_H_14
    int32_t intermediate_size       = 3072;  // 4096 for OPEN_CLIP_VIT_H_14
    int32_t n_head                  = 12;    // num_attention_heads, 16 for OPEN_CLIP_VIT_H_14
    int32_t num_hidden_layers       = 12;    // 24 for OPEN_CLIP_VIT_H_14
    int32_t layer_idx               = 11;
    int32_t projection_dim          = 1280;  // only for OPEN_CLIP_VIT_BIGG_14
    bool with_final_ln              = true;

    // embeddings
    struct ggml_tensor* position_ids;
    struct ggml_tensor* token_embed_weight;
    struct ggml_tensor* position_embed_weight;
    struct ggml_tensor* token_embed_custom;

    // transformer
    std::vector<ResidualAttentionBlock> resblocks;
    struct ggml_tensor* final_ln_w;
    struct ggml_tensor* final_ln_b;

    struct ggml_tensor* text_projection;
    std::string embd_dir;
    int32_t num_custom_embeddings = 0;
    std::vector<std::string> readed_embeddings;

    CLIPTextModel(CLIPVersion version = OPENAI_CLIP_VIT_L_14,
                  int clip_skip       = -1,
                  bool with_final_ln  = true)
        : version(version), with_final_ln(with_final_ln) {
        if (version == OPEN_CLIP_VIT_H_14) {
            hidden_size       = 1024;
            intermediate_size = 4096;
            n_head            = 16;
            num_hidden_layers = 24;
        } else if (version == OPEN_CLIP_VIT_BIGG_14) {  // CLIPTextModelWithProjection
            hidden_size       = 1280;
            intermediate_size = 5120;
            n_head            = 20;
            num_hidden_layers = 32;
        }
        set_clip_skip(clip_skip);
        resblocks.resize(num_hidden_layers);
        set_resblocks_hp_params();
    }

    void set_clip_skip(int clip_skip) {
        if (clip_skip > 0) {
            layer_idx = num_hidden_layers - clip_skip;
        }
    }

    void set_resblocks_hp_params() {
        int d_model = hidden_size / n_head;  // 64 / SDXL is 40 for CLIPTextModelWithProjection
        for (int i = 0; i < num_hidden_layers; i++) {
            resblocks[i].d_model           = d_model;
            resblocks[i].n_head            = n_head;
            resblocks[i].hidden_size       = hidden_size;
            resblocks[i].intermediate_size = intermediate_size;
        }
    }

    size_t calculate_mem_size(ggml_type wtype) {
        size_t mem_size = 0;
        mem_size += ggml_row_size(GGML_TYPE_I32, hidden_size * max_position_embeddings);  // position_ids
        mem_size += ggml_row_size(wtype, hidden_size * vocab_size);                       // token_embed_weight
        mem_size += ggml_row_size(wtype, hidden_size * max_position_embeddings);          // position_embed_weight
        if (version == OPENAI_CLIP_VIT_L_14) {
            mem_size += ggml_row_size(wtype, hidden_size * max_position_embeddings);  // token_embed_custom
        }
        for (int i = 0; i < num_hidden_layers; i++) {
            mem_size += resblocks[i].calculate_mem_size(wtype);
        }
        mem_size += 2 * ggml_row_size(GGML_TYPE_F32, hidden_size);  // final_ln_w/b
        if (version == OPEN_CLIP_VIT_BIGG_14) {
            mem_size += ggml_row_size(GGML_TYPE_F32, hidden_size * projection_dim);  // text_projection
        }
        return mem_size;
    }

    void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        tensors[prefix + "embeddings.token_embedding.weight"]    = token_embed_weight;
        tensors[prefix + "embeddings.position_embedding.weight"] = position_embed_weight;
        tensors[prefix + "final_layer_norm.weight"]              = final_ln_w;
        tensors[prefix + "final_layer_norm.bias"]                = final_ln_b;
        for (int i = 0; i < num_hidden_layers; i++) {
            std::string name = prefix + "encoder.layers." + std::to_string(i) + ".";
            resblocks[i].map_by_name(tensors, prefix + "encoder.layers." + std::to_string(i) + ".");
        }
        if (version == OPEN_CLIP_VIT_BIGG_14) {
            tensors[prefix + "text_projection"] = text_projection;
        }
    }

    bool load_embedding(std::string embd_name, std::string embd_path, std::vector<int32_t>& bpe_tokens) {
        // the order matters
        ModelLoader model_loader;
        if (!model_loader.init_from_file(embd_path)) {
            LOG_ERROR("embedding '%s' failed", embd_name.c_str());
            return false;
        }
        struct ggml_init_params params;
        params.mem_size               = 32 * 1024;  // max for custom embeddings 32 KB
        params.mem_buffer             = NULL;
        params.no_alloc               = false;
        struct ggml_context* embd_ctx = ggml_init(params);
        struct ggml_tensor* embd      = NULL;
        auto on_load                  = [&](const TensorStorage& tensor_storage, ggml_tensor** dst_tensor) {
            if (tensor_storage.ne[0] != hidden_size) {
                LOG_DEBUG("embedding wrong hidden size, got %i, expected %i", tensor_storage.ne[0], hidden_size);
                return false;
            }
            embd        = ggml_new_tensor_2d(embd_ctx, token_embed_weight->type, hidden_size, tensor_storage.n_dims > 1 ? tensor_storage.ne[1] : 1);
            *dst_tensor = embd;
            return true;
        };
        model_loader.load_tensors(on_load, NULL);
        ggml_backend_tensor_set(token_embed_custom, embd->data, num_custom_embeddings * hidden_size * ggml_type_size(token_embed_custom->type), ggml_nbytes(embd));
        readed_embeddings.push_back(embd_name);
        for (int i = 0; i < embd->ne[1]; i++) {
            bpe_tokens.push_back(vocab_size + num_custom_embeddings);
            // LOG_DEBUG("new custom token: %i", vocab_size + num_custom_embeddings);
            num_custom_embeddings++;
        }
        LOG_DEBUG("embedding '%s' applied, custom embeddings: %i", embd_name.c_str(), num_custom_embeddings);
        return true;
    }

    struct ggml_tensor* forward(struct ggml_context* ctx0, struct ggml_tensor* input_ids, struct ggml_tensor* tkn_embeddings, uint32_t max_token_idx = 0, bool return_pooled = false) {
        // input_ids: [N, n_token]
        GGML_ASSERT(input_ids->ne[0] <= position_ids->ne[0]);

        // token_embedding + position_embedding
        struct ggml_tensor* x;
        x = ggml_add(ctx0,
                     ggml_get_rows(ctx0, tkn_embeddings == NULL ? token_embed_weight : tkn_embeddings, input_ids),
                     ggml_get_rows(ctx0,
                                   position_embed_weight,
                                   ggml_view_1d(ctx0, position_ids, input_ids->ne[0], 0)));  // [N, n_token, hidden_size]

        // transformer
        for (int i = 0; i < num_hidden_layers; i++) {
            if (!return_pooled && i == layer_idx + 1) {
                // LOG_DEBUG("layer %d", i);
                break;
            }
            x = resblocks[i].forward(ctx0, x);  // [N, n_token, hidden_size]
        }

        // final layer norm
        if (return_pooled || with_final_ln) {
            x = ggml_nn_layer_norm(ctx0, x, final_ln_w, final_ln_b);
        }

        if (return_pooled) {
            // ggml_tensor* idx = ggml_argmax(ctx0, input_ids);
            // ggml_tensor* pooled = ggml_get_rows(ctx0, x, idx);
            // LOG_DEBUG("max_token_idx: %u %u", max_token_idx, x->nb[1]);
            ggml_tensor* pooled = ggml_view_1d(ctx0, x, hidden_size, x->nb[1] * max_token_idx);
            pooled              = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, text_projection)), pooled);
            return pooled;
        }

        return x;  // [N, n_token, hidden_size]
    }

    void init_params(ggml_context* ctx, ggml_backend_t backend, ggml_type wtype, ggml_allocr* alloc) {
        position_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, max_position_embeddings);

        token_embed_weight = ggml_new_tensor_2d(ctx, wtype, hidden_size, vocab_size);

        position_embed_weight = ggml_new_tensor_2d(ctx, wtype, hidden_size, max_position_embeddings);

        for (int i = 0; i < num_hidden_layers; i++) {
            resblocks[i].init_params(ctx, alloc, wtype);
        }

        final_ln_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);

        final_ln_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);

        if (version == OPENAI_CLIP_VIT_L_14) {
            token_embed_custom = ggml_new_tensor_2d(ctx, wtype, hidden_size, max_position_embeddings);
        }

        if (version == OPEN_CLIP_VIT_BIGG_14) {
            text_projection = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, projection_dim, hidden_size);
        }

        // alloc all tensors linked to this context
        for (struct ggml_tensor* t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->data == NULL) {
                ggml_allocr_alloc(alloc, t);
            }
        }

        if (ggml_backend_is_cpu(backend)) {
            for (int i = 0; i < max_position_embeddings; i++) {
                ggml_set_i32_1d(position_ids, i, i);
            }
        } else {
            std::vector<int> pos_temp;
            for (int i = 0; i < max_position_embeddings; i++) {
                pos_temp.push_back(i);
            }
            ggml_backend_tensor_set(position_ids, pos_temp.data(), 0, ggml_nbytes(position_ids));
        }
    }
};

// ldm.modules.encoders.modules.FrozenCLIPEmbedder
// Ref: https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/cad87bf4e3e0b0a759afa94e933527c3123d59bc/modules/sd_hijack_clip.py#L283
struct FrozenCLIPEmbedderWithCustomWords : public GGMLModule {
    SDVersion version = VERSION_1_x;
    CLIPTokenizer tokenizer;
    CLIPTextModel text_model;
    CLIPTextModel text_model2;

    FrozenCLIPEmbedderWithCustomWords(SDVersion version = VERSION_1_x, int clip_skip = -1)
        : version(version), tokenizer(version) {
        name = "clip";
        if (clip_skip <= 0) {
            clip_skip = 1;
            if (version == VERSION_2_x || version == VERSION_XL) {
                clip_skip = 2;
            }
        }
        if (version == VERSION_1_x) {
            text_model = CLIPTextModel(OPENAI_CLIP_VIT_L_14, clip_skip);
        } else if (version == VERSION_2_x) {
            text_model = CLIPTextModel(OPEN_CLIP_VIT_H_14, clip_skip);
        } else if (version == VERSION_XL) {
            text_model  = CLIPTextModel(OPENAI_CLIP_VIT_L_14, clip_skip, false);
            text_model2 = CLIPTextModel(OPEN_CLIP_VIT_BIGG_14, clip_skip, false);
        }
    }

    void set_clip_skip(int clip_skip) {
        text_model.set_clip_skip(clip_skip);
        if (version == VERSION_XL) {
            text_model2.set_clip_skip(clip_skip);
        }
    }

    size_t calculate_mem_size() {
        size_t mem_size = text_model.calculate_mem_size(wtype);
        if (version == VERSION_XL) {
            mem_size += text_model2.calculate_mem_size(wtype);
        }
        return mem_size;
    }

    size_t get_num_tensors() {
        size_t num_tensors = (3 + 2 + 37 * text_model.num_hidden_layers);
        if (version == VERSION_XL) {
            num_tensors += (3 + 2 + 37 * text_model2.num_hidden_layers);
        }
        return num_tensors;
    }

    void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        text_model.map_by_name(tensors, prefix + "transformer.text_model.");
        if (version == VERSION_XL) {
            text_model2.map_by_name(tensors, prefix + "1.transformer.text_model.");
        }
    }

    struct ggml_tensor* forward(struct ggml_context* ctx0, struct ggml_tensor* input_ids, struct ggml_tensor* input_ids2, struct ggml_tensor* embeddings, size_t max_token_idx = 0, bool return_pooled = false) {
        if (return_pooled) {
            return text_model2.forward(ctx0, input_ids2, NULL, max_token_idx, return_pooled);
        }
        auto hidden_states = text_model.forward(ctx0, input_ids, embeddings);  // [N, n_token, hidden_size]
        // LOG_DEBUG("hidden_states: %d %d %d %d %d", hidden_states->n_dims, hidden_states->ne[0], hidden_states->ne[1], hidden_states->ne[2], hidden_states->ne[3]);
        if (version == VERSION_XL) {
            hidden_states = ggml_reshape_4d(ctx0,
                                            hidden_states,
                                            hidden_states->ne[0],
                                            hidden_states->ne[1],
                                            hidden_states->ne[2],
                                            hidden_states->ne[3]);
            hidden_states = ggml_cont(ctx0, ggml_permute(ctx0, hidden_states, 2, 0, 1, 3));

            auto hidden_states2 = text_model2.forward(ctx0, input_ids2, NULL);  // [N, n_token, hidden_size2]
            hidden_states2      = ggml_reshape_4d(ctx0,
                                                  hidden_states2,
                                                  hidden_states2->ne[0],
                                                  hidden_states2->ne[1],
                                                  hidden_states2->ne[2],
                                                  hidden_states2->ne[3]);
            hidden_states2      = ggml_cont(ctx0, ggml_permute(ctx0, hidden_states2, 2, 0, 1, 3));

            hidden_states = ggml_concat(ctx0, hidden_states, hidden_states2);  // [N, n_token, hidden_size + hidden_size2]

            hidden_states = ggml_cont(ctx0, ggml_permute(ctx0, hidden_states, 1, 2, 0, 3));
        }
        // LOG_DEBUG("hidden_states: %d %d %d %d", hidden_states->ne[0], hidden_states->ne[1], hidden_states->ne[2], hidden_states->ne[3]);
        return hidden_states;
    }

    std::pair<std::vector<int>, std::vector<float>> tokenize(std::string text,
                                                             bool padding = false) {
        return tokenize(text, text_model.max_position_embeddings, padding);
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
            size_t word_end       = str.find(",");
            std::string embd_name = word_end == std::string::npos ? str : str.substr(0, word_end);
            embd_name             = trim(embd_name);
            std::string embd_path = get_full_path(text_model.embd_dir, embd_name + ".pt");
            if (embd_path.size() == 0) {
                embd_path = get_full_path(text_model.embd_dir, embd_name + ".ckpt");
            }
            if (embd_path.size() == 0) {
                embd_path = get_full_path(text_model.embd_dir, embd_name + ".safetensors");
            }
            if (embd_path.size() > 0) {
                if (text_model.load_embedding(embd_name, embd_path, bpe_tokens)) {
                    if (word_end != std::string::npos) {
                        str = str.substr(word_end);
                    } else {
                        str = "";
                    }
                    return true;
                }
            }
            return false;
        };

        std::vector<int> tokens;
        std::vector<float> weights;
        for (const auto& item : parsed_attention) {
            const std::string& curr_text = item.first;
            float curr_weight            = item.second;
            std::vector<int> curr_tokens = tokenizer.encode(curr_text, on_new_token_cb);
            tokens.insert(tokens.end(), curr_tokens.begin(), curr_tokens.end());
            weights.insert(weights.end(), curr_tokens.size(), curr_weight);
        }
        tokens.insert(tokens.begin(), BOS_TOKEN_ID);
        weights.insert(weights.begin(), 1.0);

        if (max_length > 0) {
            if (tokens.size() > max_length - 1) {
                tokens.resize(max_length - 1);
                weights.resize(max_length - 1);
                tokens.push_back(EOS_TOKEN_ID);
                weights.push_back(1.0);
            } else {
                tokens.push_back(EOS_TOKEN_ID);
                weights.push_back(1.0);
                if (padding) {
                    int pad_token_id = PAD_TOKEN_ID;
                    if (version == VERSION_2_x) {
                        pad_token_id = 0;
                    }
                    tokens.insert(tokens.end(), max_length - tokens.size(), pad_token_id);
                    weights.insert(weights.end(), max_length - weights.size(), 1.0);
                }
            }
        }

        // for (int i = 0; i < tokens.size(); i++) {
        //     std::cout << tokens[i] << ":" << weights[i] << ", ";
        // }
        // std::cout << std::endl;

        return {tokens, weights};
    }

    void init_params() {
        ggml_allocr* alloc = ggml_allocr_new_from_buffer(params_buffer);
        text_model.init_params(params_ctx, backend, wtype, alloc);
        if (version == VERSION_XL) {
            text_model2.init_params(params_ctx, backend, wtype, alloc);
        }
        ggml_allocr_free(alloc);
    }

    struct ggml_cgraph* build_graph(struct ggml_allocr* allocr, std::vector<int> tokens, bool return_pooled = false) {
        // since we are using ggml-alloc, this buffer only needs enough space to hold the ggml_tensor and ggml_cgraph structs, but not the tensor data
        static size_t buf_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
        static std::vector<uint8_t> buf(buf_size);

        struct ggml_init_params params = {
            /*.mem_size   =*/buf_size,
            /*.mem_buffer =*/buf.data(),
            /*.no_alloc   =*/true,  // the tensors will be allocated later by ggml_allocr_alloc_graph()
        };

        struct ggml_context* ctx0 = ggml_init(params);

        struct ggml_cgraph* gf = ggml_new_graph(ctx0);

        struct ggml_tensor* input_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, tokens.size());
        ggml_allocr_alloc(allocr, input_ids);

        if (!ggml_allocr_is_measure(allocr)) {
            ggml_backend_tensor_set(input_ids, tokens.data(), 0, tokens.size() * ggml_element_size(input_ids));
        }

        struct ggml_tensor* input_ids2 = NULL;
        size_t max_token_idx           = 0;
        if (version == VERSION_XL) {
            input_ids2 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, tokens.size());
            ggml_allocr_alloc(allocr, input_ids2);

            auto it = std::find(tokens.begin(), tokens.end(), EOS_TOKEN_ID);
            if (it != tokens.end()) {
                std::fill(std::next(it), tokens.end(), 0);
            }

            max_token_idx = std::min<size_t>(std::distance(tokens.begin(), it), tokens.size() - 1);

            // for (int i = 0; i < tokens.size(); i++) {
            //     printf("%d ", tokens[i]);
            // }
            // printf("\n");

            if (!ggml_allocr_is_measure(allocr)) {
                ggml_backend_tensor_set(input_ids2, tokens.data(), 0, tokens.size() * ggml_element_size(input_ids2));
            }
        }

        struct ggml_tensor* embeddings = NULL;

        if (text_model.num_custom_embeddings > 0 && version != VERSION_XL) {
            embeddings = ggml_new_tensor_2d(ctx0, wtype, text_model.hidden_size, text_model.vocab_size + text_model.num_custom_embeddings /* custom placeholder */);
            ggml_allocr_alloc(allocr, embeddings);
            if (!ggml_allocr_is_measure(allocr)) {
                // really bad, there is memory inflexibility (this is for host<->device memory conflicts)
                void* freeze_data = malloc(ggml_nbytes(text_model.token_embed_weight));
                ggml_backend_tensor_get_and_sync(backend, text_model.token_embed_weight, freeze_data, 0, ggml_nbytes(text_model.token_embed_weight));
                ggml_backend_tensor_set(embeddings, freeze_data, 0, ggml_nbytes(text_model.token_embed_weight));
                free(freeze_data);
                // concatenate custom embeddings
                void* custom_data = malloc(ggml_nbytes(text_model.token_embed_custom));
                ggml_backend_tensor_get_and_sync(backend, text_model.token_embed_custom, custom_data, 0, ggml_nbytes(text_model.token_embed_custom));
                ggml_backend_tensor_set(embeddings, custom_data, ggml_nbytes(text_model.token_embed_weight), text_model.num_custom_embeddings * text_model.hidden_size * ggml_type_size(wtype));
                free(custom_data);
            }
        }

        struct ggml_tensor* hidden_states = forward(ctx0, input_ids, input_ids2, embeddings, max_token_idx, return_pooled);

        ggml_build_forward_expand(gf, hidden_states);
        ggml_free(ctx0);

        return gf;
    }

    void alloc_compute_buffer(ggml_context* work_ctx, int max_tokens) {
        auto get_graph = [&]() -> struct ggml_cgraph* {
            bool return_pooled = false;
            if (version == VERSION_XL) {
                return_pooled = true;
            }
            return build_graph(compute_allocr, std::vector<int>(max_tokens), return_pooled);
        };
        GGMLModule::alloc_compute_buffer(get_graph);
    }

    void compute(const int n_threads,
                 std::vector<int> tokens,
                 ggml_tensor* hidden_state_output,
                 ggml_tensor* pooled_output = NULL) {
        auto get_graph = [&]() -> struct ggml_cgraph* {
            return build_graph(compute_allocr, tokens, false);
        };
        GGMLModule::compute(get_graph, n_threads, hidden_state_output);

        if (version == VERSION_XL && pooled_output != NULL) {
            auto get_graph = [&]() -> struct ggml_cgraph* {
                return build_graph(compute_allocr, tokens, true);
            };
            GGMLModule::compute(get_graph, n_threads, pooled_output);
        }
    }
};

#endif  // __CLIP_HPP__