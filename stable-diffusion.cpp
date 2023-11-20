#include <assert.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "ggml/ggml.h"
#include "rng.h"
#include "rng_philox.h"
#include "stable-diffusion.h"

#define EPS 1e-05f

static SDLogLevel log_level = SDLogLevel::INFO;

#define __FILENAME__ "stable-diffusion.cpp"
#define SD_LOG(level, format, ...)                                                                    \
    do {                                                                                              \
        if (level < log_level) {                                                                      \
            break;                                                                                    \
        }                                                                                             \
        if (level == SDLogLevel::DEBUG) {                                                             \
            printf("[DEBUG] %s:%-4d - " format "\n", __FILENAME__, __LINE__, ##__VA_ARGS__);          \
            fflush(stdout);                                                                           \
        } else if (level == SDLogLevel::INFO) {                                                       \
            printf("[INFO]  %s:%-4d - " format "\n", __FILENAME__, __LINE__, ##__VA_ARGS__);          \
            fflush(stdout);                                                                           \
        } else if (level == SDLogLevel::WARN) {                                                       \
            fprintf(stderr, "[WARN]  %s:%-4d - " format "\n", __FILENAME__, __LINE__, ##__VA_ARGS__); \
            fflush(stdout);                                                                           \
        } else if (level == SDLogLevel::ERROR) {                                                      \
            fprintf(stderr, "[ERROR] %s:%-4d - " format "\n", __FILENAME__, __LINE__, ##__VA_ARGS__); \
            fflush(stdout);                                                                           \
        }                                                                                             \
    } while (0)

#define LOG_DEBUG(format, ...) SD_LOG(SDLogLevel::DEBUG, format, ##__VA_ARGS__)
#define LOG_INFO(format, ...) SD_LOG(SDLogLevel::INFO, format, ##__VA_ARGS__)
#define LOG_WARN(format, ...) SD_LOG(SDLogLevel::WARN, format, ##__VA_ARGS__)
#define LOG_ERROR(format, ...) SD_LOG(SDLogLevel::ERROR, format, ##__VA_ARGS__)

#define GGML_FILE_MAGIC 0x67676d6c

#define TIMESTEPS 1000

enum ModelType {
    SD1 = 0,
    SD2 = 1,
    MODEL_TYPE_COUNT,
};

const char* model_type_to_str[] = {
    "SD1.x",
    "SD2.x"};

/*================================================== Helper Functions ================================================*/

void set_sd_log_level(SDLogLevel level) {
    log_level = level;
}

std::string sd_get_system_info() {
    std::stringstream ss;
    ss << "System Info: \n";
    ss << "    BLAS = " << ggml_cpu_has_blas() << std::endl;
    ss << "    SSE3 = " << ggml_cpu_has_sse3() << std::endl;
    ss << "    AVX = " << ggml_cpu_has_avx() << std::endl;
    ss << "    AVX2 = " << ggml_cpu_has_avx2() << std::endl;
    ss << "    AVX512 = " << ggml_cpu_has_avx512() << std::endl;
    ss << "    AVX512_VBMI = " << ggml_cpu_has_avx512_vbmi() << std::endl;
    ss << "    AVX512_VNNI = " << ggml_cpu_has_avx512_vnni() << std::endl;
    ss << "    FMA = " << ggml_cpu_has_fma() << std::endl;
    ss << "    NEON = " << ggml_cpu_has_neon() << std::endl;
    ss << "    ARM_FMA = " << ggml_cpu_has_arm_fma() << std::endl;
    ss << "    F16C = " << ggml_cpu_has_f16c() << std::endl;
    ss << "    FP16_VA = " << ggml_cpu_has_fp16_va() << std::endl;
    ss << "    WASM_SIMD = " << ggml_cpu_has_wasm_simd() << std::endl;
    ss << "    VSX = " << ggml_cpu_has_vsx() << std::endl;
    return ss.str();
}

ggml_tensor* load_tensor_from_file(ggml_context* ctx, const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERROR("failed to open '%s'", file_path.c_str());
        return NULL;
    }
    int32_t n_dims;
    int32_t length;
    int32_t ttype;

    file.read(reinterpret_cast<char*>(&n_dims), sizeof(n_dims));
    file.read(reinterpret_cast<char*>(&length), sizeof(length));
    file.read(reinterpret_cast<char*>(&ttype), sizeof(ttype));

    if (file.eof()) {
        LOG_ERROR("incomplete file '%s'", file_path.c_str());
        return NULL;
    }

    int32_t nelements = 1;
    int32_t ne[4]     = {1, 1, 1, 1};
    for (int i = 0; i < n_dims; ++i) {
        file.read(reinterpret_cast<char*>(&ne[i]), sizeof(ne[i]));
        nelements *= ne[i];
    }
    std::string name(length, 0);
    file.read(&name[0], length);
    ggml_tensor* tensor = ggml_new_tensor_4d(ctx, (ggml_type)ttype, ne[0], ne[1], ne[2], ne[3]);
    const size_t bpe    = ggml_type_size(ggml_type(ttype));
    file.read(reinterpret_cast<char*>(tensor->data), ggml_nbytes(tensor));
    return tensor;
}

void ggml_tensor_set_f32_randn(struct ggml_tensor* tensor, std::shared_ptr<RNG> rng) {
    uint32_t n                        = (uint32_t)ggml_nelements(tensor);
    std::vector<float> random_numbers = rng->randn(n);
    for (uint32_t i = 0; i < n; i++) {
        ggml_set_f32_1d(tensor, i, random_numbers[i]);
    }
}

// set tensor[i, j, k, l]
// set tensor[l]
// set tensor[k, l]
// set tensor[j, k, l]
void ggml_tensor_set_f32(struct ggml_tensor* tensor, float value, int l, int k = 0, int j = 0, int i = 0) {
    GGML_ASSERT(tensor->nb[0] == sizeof(float));
    *(float*)((char*)(tensor->data) + i * tensor->nb[3] + j * tensor->nb[2] + k * tensor->nb[1] + l * tensor->nb[0]) = value;
}

float ggml_tensor_get_f32(const ggml_tensor* tensor, int l, int k = 0, int j = 0, int i = 0) {
    GGML_ASSERT(tensor->nb[0] == sizeof(float));
    return *(float*)((char*)(tensor->data) + i * tensor->nb[3] + j * tensor->nb[2] + k * tensor->nb[1] + l * tensor->nb[0]);
}

void print_ggml_tensor(struct ggml_tensor* tensor, bool shape_only = false) {
    printf("shape(%zu, %zu, %zu, %zu)\n", tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    fflush(stdout);
    if (shape_only) {
        return;
    }
    int range = 3;
    for (int i = 0; i < tensor->ne[3]; i++) {
        if (i >= range && i + range < tensor->ne[3]) {
            continue;
        }
        for (int j = 0; j < tensor->ne[2]; j++) {
            if (j >= range && j + range < tensor->ne[2]) {
                continue;
            }
            for (int k = 0; k < tensor->ne[1]; k++) {
                if (k >= range && k + range < tensor->ne[1]) {
                    continue;
                }
                for (int l = 0; l < tensor->ne[0]; l++) {
                    if (l >= range && l + range < tensor->ne[0]) {
                        continue;
                    }
                    printf("  [%d, %d, %d, %d] = %f\n", i, j, k, l, ggml_tensor_get_f32(tensor, l, k, j, i));
                    fflush(stdout);
                }
            }
        }
    }
}

void copy_ggml_tensor(
    struct ggml_tensor* dst,
    const struct ggml_tensor* src) {
    dst->nb[0] = src->nb[0];
    dst->nb[1] = src->nb[1];
    dst->nb[2] = src->nb[2];
    dst->nb[3] = src->nb[3];

    memcpy(((char*)dst->data), ((char*)src->data), ggml_nbytes(dst));
}

// Ref: https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/util.py#L151
void set_timestep_embedding(struct ggml_tensor* timesteps, struct ggml_tensor* embedding, int dim, int max_period = 10000) {
    // timesteps: [N,]
    // embedding: [(dim + 1)/2, N]
    int half = dim / 2;
    std::vector<float> freqs(half);
    for (int i = 0; i < half; ++i) {
        freqs[i] = (float)std::exp(-std::log(max_period) * i / half);
    }
    for (int i = 0; i < timesteps->ne[0]; ++i) {
        for (int j = 0; j < half; ++j) {
            float arg = ggml_get_f32_1d(timesteps, i) * freqs[j];
            ggml_tensor_set_f32(embedding, std::cos(arg), j, i);
            ggml_tensor_set_f32(embedding, std::sin(arg), j + half, i);
        }
        if (dim % 2 != 0) {
            *(float*)((char*)embedding->data + i * embedding->nb[1] + dim * embedding->nb[0]) = 0;
        }
    }
}

struct ggml_tensor* new_timestep_embedding(struct ggml_context* ctx, struct ggml_tensor* timesteps, int dim, int max_period = 10000) {
    // timesteps: [N,]
    // embedding: [(dim + 1)/2, N]
    int acutual_dim = dim;
    if (dim % 2 != 0) {
        acutual_dim = dim + 1;
    }
    struct ggml_tensor* embedding = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, acutual_dim, timesteps->ne[0]);
    if (!ggml_get_no_alloc(ctx)) {
        set_timestep_embedding(timesteps, embedding, dim, max_period);
    }
    return embedding;
}

std::vector<uint8_t> ggml_to_image_vec(struct ggml_tensor* t) {
    int64_t w = t->ne[0];
    int64_t h = t->ne[1];
    int64_t c = t->ne[2];
    std::vector<uint8_t> vec;
    vec.resize(w * h * c);
    uint8_t* data = (uint8_t*)vec.data();
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            for (int k = 0; k < c; k++) {
                float value = ggml_tensor_get_f32(t, j, i, k);
                value       = (value + 1.0f) * 0.5f;
                if (value < 0) {
                    value = 0;
                } else if (value > 1) {
                    value = 1;
                }
                value *= 255.f;
                *(data + i * w * c + j * c + k) = (uint8_t)value;
            }
        }
    }
    return vec;
}

void image_vec_to_ggml(const std::vector<uint8_t>& vec,
                       struct ggml_tensor* t) {
    int64_t w     = t->ne[0];
    int64_t h     = t->ne[1];
    int64_t c     = t->ne[2];
    uint8_t* data = (uint8_t*)vec.data();
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            for (int k = 0; k < c; k++) {
                float value = *(data + i * w * c + j * c + k);
                value       = value / 255.f;
                value       = 2 * value - 1;
                ggml_tensor_set_f32(t, value, j, i, k);
            }
        }
    }
}

struct ggml_tensor* ggml_group_norm_32(struct ggml_context* ctx,
                                       struct ggml_tensor* a) {
    return ggml_group_norm(ctx, a, 32);
}

std::pair<std::unordered_map<std::string, float>, std::string> extract_and_remove_lora(std::string text) {
    std::regex re("<lora:([^:]+):([^>]+)>");
    std::smatch matches;
    std::unordered_map<std::string, float> filename2multiplier;

    while (std::regex_search(text, matches, re)) {
        std::string filename = matches[1].str();
        float multiplier     = std::stof(matches[2].str());

        if (multiplier == 0.f) {
            continue;
        }

        if (filename2multiplier.find(filename) == filename2multiplier.end()) {
            filename2multiplier[filename] = multiplier;
        } else {
            filename2multiplier[filename] += multiplier;
        }

        text = std::regex_replace(text, re, "", std::regex_constants::format_first_only);
    }

    return std::make_pair(filename2multiplier, text);
}

bool ends_with(const std::string& str, const std::string& ending) {
    if (str.length() >= ending.length()) {
        return (str.compare(str.length() - ending.length(), ending.length(), ending) == 0);
    } else {
        return false;
    }
}

void replace_all_chars(std::string& str, char target, char replacement) {
    for (size_t i = 0; i < str.length(); ++i) {
        if (str[i] == target) {
            str[i] = replacement;
        }
    }
}

/*================================================== CLIPTokenizer ===================================================*/

const std::string UNK_TOKEN = "<|endoftext|>";
const std::string BOS_TOKEN = "<|startoftext|>";
const std::string EOS_TOKEN = "<|endoftext|>";
const std::string PAD_TOEKN = "<|endoftext|>";

const int UNK_TOKEN_ID = 49407;
const int BOS_TOKEN_ID = 49406;
const int EOS_TOKEN_ID = 49407;
const int PAD_TOKEN_ID = 49407;

// Ref: https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py
// TODO: implement bpe
class CLIPTokenizer {
private:
    ModelType model_type = SD1;
    std::map<std::string, int32_t> encoder;
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

public:
    CLIPTokenizer(ModelType model_type = SD1)
        : model_type(model_type){};
    std::string bpe(std::string token) {
        std::string word = token + "</w>";
        if (encoder.find(word) != encoder.end()) {
            return word;
        } else if (encoder.find(token) != encoder.end()) {
            return token;
        }
        return UNK_TOKEN;
    }

    void add_token(std::string token, int32_t token_id) {
        encoder[token] = token_id;
    }

    std::vector<int> tokenize(std::string text, size_t max_length = 0, bool padding = false) {
        std::vector<int32_t> tokens = encode(text);
        tokens.insert(tokens.begin(), BOS_TOKEN_ID);
        if (max_length > 0) {
            if (tokens.size() > max_length - 1) {
                tokens.resize(max_length - 1);
                tokens.push_back(EOS_TOKEN_ID);
            } else {
                tokens.push_back(EOS_TOKEN_ID);
                if (padding) {
                    int pad_token_id = PAD_TOKEN_ID;
                    if (model_type == SD2) {
                        pad_token_id = 0;
                    }
                    tokens.insert(tokens.end(), max_length - tokens.size(), pad_token_id);
                }
            }
        }
        return tokens;
    }

    std::vector<int> encode(std::string text) {
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
            for (auto& token : matches) {
                std::istringstream iss(bpe(token));
                std::vector<std::string> tokens{std::istream_iterator<std::string>{iss},
                                                std::istream_iterator<std::string>{}};
                for (const auto& bpe_token : tokens) {
                    bpe_tokens.push_back(encoder[bpe_token]);
                    token_strs.push_back(bpe_token);
                }
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

    size_t compute_params_mem_size(ggml_type wtype) {
        double mem_size = 0;
        mem_size += 4 * hidden_size * hidden_size * ggml_type_sizef(wtype);        // q_w/k_w/v_w/out_w
        mem_size += 8 * hidden_size * ggml_type_sizef(GGML_TYPE_F32);              // q_b/k_b/v_b/out_b/ln1_w/ln1_b/ln2_w/ln2_b
        mem_size += 2 * hidden_size * intermediate_size * ggml_type_sizef(wtype);  // fc1_w/fc2_w
        mem_size += intermediate_size * ggml_type_sizef(GGML_TYPE_F32);            // fc1_b
        mem_size += hidden_size * ggml_type_sizef(GGML_TYPE_F32);                  // fc2_b
        mem_size += 16 * ggml_tensor_overhead();                                   // tensor overhead
        return static_cast<size_t>(mem_size);
    }

    void init_params(struct ggml_context* ctx, ggml_type wtype) {
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
        {
            x = ggml_norm(ctx, x, EPS);
            x = ggml_add(ctx,
                         ggml_mul(ctx, ggml_repeat(ctx, ln1_w, x), x),
                         ggml_repeat(ctx, ln1_b, x));
        }
        // self-attention
        {
            struct ggml_tensor* q = ggml_add(ctx,
                                             ggml_repeat(ctx, q_b, x),
                                             ggml_mul_mat(ctx, q_w, x));
            q                     = ggml_scale_inplace(ctx, q, ggml_new_f32(ctx, 1.0f / sqrt((float)d_model)));
            q                     = ggml_reshape_4d(ctx, q, d_model, n_head, n_token, N);   // [N, n_token, n_head, d_model]
            q                     = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));       // [N, n_head, n_token, d_model]
            q                     = ggml_reshape_3d(ctx, q, d_model, n_token, n_head * N);  // [N * n_head, n_token, d_model]

            struct ggml_tensor* k = ggml_add(ctx,
                                             ggml_repeat(ctx, k_b, x),
                                             ggml_mul_mat(ctx, k_w, x));
            k                     = ggml_reshape_4d(ctx, k, d_model, n_head, n_token, N);  // [N, n_token, n_head, d_model]
            k                     = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));      // [N, n_head, n_token, d_model]
            k                     = ggml_reshape_3d(ctx, k, d_model, n_token, n_head);     // [N * n_head, n_token, d_model]

            struct ggml_tensor* v = ggml_add(ctx,
                                             ggml_repeat(ctx, v_b, x),
                                             ggml_mul_mat(ctx, v_w, x));
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
        x = ggml_add(ctx, ggml_repeat(ctx, out_b, x), ggml_mul_mat(ctx, out_w, x));

        // residual
        x = ggml_add(ctx, x, r);
        r = x;

        // layer norm 2
        {
            x = ggml_norm(ctx, x, EPS);

            x = ggml_add(ctx, ggml_mul(ctx, ggml_repeat(ctx, ln2_w, x), x),
                         ggml_repeat(ctx, ln2_b, x));
        }

        // mlp
        x = ggml_mul_mat(ctx, fc1_w, x);
        x = ggml_add(ctx, ggml_repeat(ctx, fc1_b, x), x);

        if (hidden_size == 1024) {  // SD 2.x
            x = ggml_gelu_inplace(ctx, x);
        } else {  // SD 1.x
            x = ggml_gelu_quick_inplace(ctx, x);
        }

        x = ggml_mul_mat(ctx, fc2_w, x);
        x = ggml_add(ctx, ggml_repeat(ctx, fc2_b, x), x);

        // residual 2
        x = ggml_add(ctx, x, r);

        return x;
    }
};

// SD1.x: https://huggingface.co/openai/clip-vit-large-patch14/blob/main/config.json
// SD2.x: https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/blob/main/config.json
struct CLIPTextModel {
    ModelType model_type = SD1;
    // network hparams
    int32_t vocab_size              = 49408;
    int32_t max_position_embeddings = 77;
    int32_t hidden_size             = 768;   // 1024 for SD 2.x
    int32_t intermediate_size       = 3072;  // 4096 for SD 2.x
    int32_t n_head                  = 12;    // num_attention_heads, 16 for SD 2.x
    int32_t num_hidden_layers       = 12;    // 24 for SD 2.x

    // embeddings
    struct ggml_tensor* position_ids;
    struct ggml_tensor* token_embed_weight;
    struct ggml_tensor* position_embed_weight;
    // transformer
    std::vector<ResidualAttentionBlock> resblocks;
    struct ggml_tensor* final_ln_w;
    struct ggml_tensor* final_ln_b;

    CLIPTextModel(ModelType model_type = SD1)
        : model_type(model_type) {
        if (model_type == SD2) {
            hidden_size       = 1024;
            intermediate_size = 4096;
            n_head            = 16;
            num_hidden_layers = 24;
        }
        resblocks.resize(num_hidden_layers);
        set_resblocks_hp_params();
    }

    void set_resblocks_hp_params() {
        int d_model = hidden_size / n_head;  // 64
        for (int i = 0; i < num_hidden_layers; i++) {
            resblocks[i].d_model           = d_model;
            resblocks[i].n_head            = n_head;
            resblocks[i].hidden_size       = hidden_size;
            resblocks[i].intermediate_size = intermediate_size;
        }
    }

    size_t compute_params_mem_size(ggml_type wtype) {
        double mem_size = 0;
        mem_size += hidden_size * max_position_embeddings * ggml_type_sizef(GGML_TYPE_I32);  // position_ids
        mem_size += hidden_size * vocab_size * ggml_type_sizef(wtype);                       // token_embed_weight
        mem_size += hidden_size * max_position_embeddings * ggml_type_sizef(wtype);          // position_embed_weight
        for (int i = 0; i < num_hidden_layers; i++) {
            mem_size += resblocks[i].compute_params_mem_size(wtype);
        }
        mem_size += 2 * hidden_size * ggml_type_sizef(GGML_TYPE_F32);  // final_ln_w/b
        mem_size += ggml_tensor_overhead();                            // object overhead
        return static_cast<size_t>(mem_size);
    }

    void init_params(struct ggml_context* ctx, ggml_type wtype) {
        position_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, max_position_embeddings);
        for (int i = 0; i < max_position_embeddings; i++) {
            ggml_set_i32_1d(position_ids, i, i);
        }
        token_embed_weight    = ggml_new_tensor_2d(ctx, wtype, hidden_size, vocab_size);
        position_embed_weight = ggml_new_tensor_2d(ctx, wtype, hidden_size, max_position_embeddings);

        for (int i = 0; i < num_hidden_layers; i++) {
            resblocks[i].init_params(ctx, wtype);
        }

        final_ln_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
        final_ln_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
    }

    void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        tensors[prefix + "embeddings.token_embedding.weight"]    = token_embed_weight;
        tensors[prefix + "embeddings.position_embedding.weight"] = position_embed_weight;
        tensors[prefix + "final_layer_norm.weight"]              = final_ln_w;
        tensors[prefix + "final_layer_norm.bias"]                = final_ln_b;
        for (int i = 0; i < num_hidden_layers; i++) {
            resblocks[i].map_by_name(tensors, prefix + "encoder.layers." + std::to_string(i) + ".");
        }
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* input_ids) {
        // input_ids: [N, n_token]
        GGML_ASSERT(input_ids->ne[0] <= position_ids->ne[0]);

        // token_embedding + position_embedding
        struct ggml_tensor* x;
        x = ggml_add(ctx,
                     ggml_get_rows(ctx, token_embed_weight, input_ids),
                     ggml_get_rows(ctx,
                                   position_embed_weight,
                                   ggml_view_1d(ctx, position_ids, input_ids->ne[0], 0)));  // [N, n_token, hidden_size]

        // transformer
        for (int i = 0; i < num_hidden_layers; i++) {
            if (model_type == SD2 && i == num_hidden_layers - 1) {  // layer: "penultimate"
                break;
            }
            x = resblocks[i].forward(ctx, x);  // [N, n_token, hidden_size]
        }

        // final layer norm
        {
            x = ggml_norm(ctx, x, EPS);

            x = ggml_add(ctx, ggml_mul(ctx, ggml_repeat(ctx, final_ln_w, x), x),
                         ggml_repeat(ctx, final_ln_b, x));
        }

        return x;  // [N, n_token, hidden_size]
    }
};

// ldm.modules.encoders.modules.FrozenCLIPEmbedder
struct FrozenCLIPEmbedder {
    CLIPTokenizer tokenizer;
    CLIPTextModel text_model;
    struct ggml_tensor* forward(struct ggml_context* ctx, const std::string& prompt) {
        std::vector<int32_t> tokens   = tokenizer.tokenize(prompt, text_model.max_position_embeddings, true);
        struct ggml_tensor* input_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tokens.size());
        memcpy(input_ids->data, tokens.data(), tokens.size() * ggml_element_size(input_ids));
        struct ggml_tensor* hidden_states = text_model.forward(ctx, input_ids);
        return hidden_states;
    }
};

// Ref: https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/cad87bf4e3e0b0a759afa94e933527c3123d59bc/modules/sd_hijack_clip.py#L283
struct FrozenCLIPEmbedderWithCustomWords {
    ModelType model_type = SD1;
    CLIPTokenizer tokenizer;
    CLIPTextModel text_model;

    FrozenCLIPEmbedderWithCustomWords(ModelType model_type = SD1)
        : model_type(model_type), tokenizer(model_type), text_model(model_type) {}

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

        std::vector<int> tokens;
        std::vector<float> weights;
        for (const auto& item : parsed_attention) {
            const std::string& curr_text = item.first;
            float curr_weight            = item.second;
            std::vector<int> curr_tokens = tokenizer.encode(curr_text);
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
                    if (model_type == SD2) {
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
};

/*==================================================== UnetModel =====================================================*/

struct ResBlock {
    // network hparams
    int channels;      // model_channels * (1, 1, 1, 2, 2, 4, 4, 4)
    int emb_channels;  // time_embed_dim
    int out_channels;  // mult * model_channels

    // network params
    // in_layers
    struct ggml_tensor* in_layer_0_w;  // [channels, ]
    struct ggml_tensor* in_layer_0_b;  // [channels, ]
    // in_layer_1 is nn.SILU()
    struct ggml_tensor* in_layer_2_w;  // [out_channels, channels, 3, 3]
    struct ggml_tensor* in_layer_2_b;  // [out_channels, ]

    // emb_layers
    // emb_layer_0 is nn.SILU()
    struct ggml_tensor* emb_layer_1_w;  // [out_channels, emb_channels]
    struct ggml_tensor* emb_layer_1_b;  // [out_channels, ]

    // out_layers
    struct ggml_tensor* out_layer_0_w;  // [out_channels, ]
    struct ggml_tensor* out_layer_0_b;  // [out_channels, ]
    // out_layer_1 is nn.SILU()
    // out_layer_2 is nn.Dropout(), p = 0 for inference
    struct ggml_tensor* out_layer_3_w;  // [out_channels, out_channels, 3, 3]
    struct ggml_tensor* out_layer_3_b;  // [out_channels, ]

    // skip connection, only if out_channels != channels
    struct ggml_tensor* skip_w;  // [out_channels, channels, 1, 1]
    struct ggml_tensor* skip_b;  // [out_channels, ]

    size_t compute_params_mem_size(ggml_type wtype) {
        double mem_size = 0;
        mem_size += 2 * channels * ggml_type_sizef(GGML_TYPE_F32);                         // in_layer_0_w/b
        mem_size += out_channels * channels * 3 * 3 * ggml_type_sizef(GGML_TYPE_F16);      // in_layer_2_w
        mem_size += 5 * out_channels * ggml_type_sizef(GGML_TYPE_F32);                     // in_layer_2_b/emb_layer_1_b/out_layer_0_w/out_layer_0_b/out_layer_3_b
        mem_size += out_channels * emb_channels * ggml_type_sizef(wtype);                  // emb_layer_1_w
        mem_size += out_channels * out_channels * 3 * 3 * ggml_type_sizef(GGML_TYPE_F16);  // out_layer_3_w

        mem_size += 10 * ggml_tensor_overhead();  // object overhead

        if (out_channels != channels) {
            mem_size += out_channels * channels * 1 * 1 * ggml_type_sizef(GGML_TYPE_F16);  // skip_w
            mem_size += out_channels * ggml_type_sizef(GGML_TYPE_F32);                     // skip_b

            mem_size += 2 * ggml_tensor_overhead();  // object overhead
        }
        return static_cast<size_t>(mem_size);
    }

    void init_params(struct ggml_context* ctx, ggml_type wtype) {
        in_layer_0_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, channels);
        in_layer_0_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, channels);
        in_layer_2_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, channels, out_channels);
        in_layer_2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);

        emb_layer_1_w = ggml_new_tensor_2d(ctx, wtype, emb_channels, out_channels);
        emb_layer_1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);

        out_layer_0_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
        out_layer_0_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
        out_layer_3_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, out_channels, out_channels);
        out_layer_3_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);

        if (out_channels != channels) {
            skip_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 1, 1, channels, out_channels);
            skip_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
        }
    }

    void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        tensors[prefix + "in_layers.0.weight"] = in_layer_0_w;
        tensors[prefix + "in_layers.0.bias"]   = in_layer_0_b;
        tensors[prefix + "in_layers.2.weight"] = in_layer_2_w;
        tensors[prefix + "in_layers.2.bias"]   = in_layer_2_b;

        tensors[prefix + "emb_layers.1.weight"] = emb_layer_1_w;
        tensors[prefix + "emb_layers.1.bias"]   = emb_layer_1_b;

        tensors[prefix + "out_layers.0.weight"] = out_layer_0_w;
        tensors[prefix + "out_layers.0.bias"]   = out_layer_0_b;
        tensors[prefix + "out_layers.3.weight"] = out_layer_3_w;
        tensors[prefix + "out_layers.3.bias"]   = out_layer_3_b;

        if (out_channels != channels) {
            tensors[prefix + "skip_connection.weight"] = skip_w;
            tensors[prefix + "skip_connection.bias"]   = skip_b;
        }
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* emb) {
        // x: [N, channels, h, w]
        // emb: [N, emb_channels]

        // in_layers
        // group norm 32
        auto h = ggml_group_norm_32(ctx, x);
        h      = ggml_add(ctx,
                          ggml_mul(ctx,
                                   ggml_repeat(ctx,
                                               ggml_reshape_4d(ctx, in_layer_0_w, 1, 1, in_layer_0_w->ne[0], 1),
                                               h),
                                   h),
                          ggml_repeat(ctx,
                                      ggml_reshape_4d(ctx, in_layer_0_b, 1, 1, in_layer_0_b->ne[0], 1),
                                      h));
        // silu
        h = ggml_silu_inplace(ctx, h);
        // conv2d
        h = ggml_conv_2d(ctx, in_layer_2_w, h, 1, 1, 1, 1, 1, 1);
        h = ggml_add(ctx,
                     h,
                     ggml_repeat(ctx,
                                 ggml_reshape_4d(ctx, in_layer_2_b, 1, 1, in_layer_2_b->ne[0], 1),
                                 h));  // [N, out_channels, h, w]

        // emb_layers
        auto emb_out = ggml_silu(ctx, emb);
        emb_out      = ggml_mul_mat(ctx, emb_layer_1_w, emb_out);
        emb_out      = ggml_add(ctx, ggml_repeat(ctx, emb_layer_1_b, emb_out), emb_out);     // [N, out_channels]
        emb_out      = ggml_reshape_4d(ctx, emb_out, 1, 1, emb_out->ne[0], emb_out->ne[1]);  // [N, out_channels, 1, 1]
        emb_out      = ggml_repeat(ctx, emb_out, h);                                         // [N, out_channels, h, w]

        // out_layers
        h = ggml_add(ctx, h, emb_out);
        // group norm 32
        h = ggml_group_norm_inplace(ctx, h, 32);
        h = ggml_add(ctx,
                     ggml_mul(ctx, ggml_repeat(ctx, ggml_reshape_4d(ctx, out_layer_0_w, 1, 1, out_layer_0_w->ne[0], 1), h), h),
                     ggml_repeat(ctx, ggml_reshape_4d(ctx, out_layer_0_b, 1, 1, out_layer_0_b->ne[0], 1), h));
        // silu
        h = ggml_silu_inplace(ctx, h);
        // dropout, skip for inference
        // conv2d
        h = ggml_conv_2d(ctx, out_layer_3_w, h, 1, 1, 1, 1, 1, 1);
        h = ggml_add(ctx,
                     h,
                     ggml_repeat(ctx,
                                 ggml_reshape_4d(ctx, out_layer_3_b, 1, 1, out_layer_3_b->ne[0], 1),
                                 h));  // [N, out_channels, h, w

        // skip connection
        if (out_channels != channels) {
            x = ggml_conv_2d(ctx, skip_w, x, 1, 1, 0, 0, 1, 1);
            x = ggml_add(ctx,
                         x,
                         ggml_repeat(ctx,
                                     ggml_reshape_4d(ctx, skip_b, 1, 1, skip_b->ne[0], 1),
                                     x));  // [N, out_channels, h, w]
        }
        h = ggml_add(ctx, h, x);
        return h;  // [N, out_channels, h, w]
    }
};

struct SpatialTransformer {
    int in_channels;        // mult * model_channels
    int n_head;             // num_heads
    int d_head;             // in_channels // n_heads
    int depth       = 1;    // 1
    int context_dim = 768;  // hidden_size, 1024 for SD2.x

    // group norm
    struct ggml_tensor* norm_w;  // [in_channels,]
    struct ggml_tensor* norm_b;  // [in_channels,]

    // proj_in
    struct ggml_tensor* proj_in_w;  // [in_channels, in_channels, 1, 1]
    struct ggml_tensor* proj_in_b;  // [in_channels,]

    // transformer
    struct
    {
        // layer norm 1
        struct ggml_tensor* norm1_w;  // [in_channels, ]
        struct ggml_tensor* norm1_b;  // [in_channels, ]

        // attn1
        struct ggml_tensor* attn1_q_w;  // [in_channels, in_channels]
        struct ggml_tensor* attn1_k_w;  // [in_channels, in_channels]
        struct ggml_tensor* attn1_v_w;  // [in_channels, in_channels]

        struct ggml_tensor* attn1_out_w;  // [in_channels, in_channels]
        struct ggml_tensor* attn1_out_b;  // [in_channels, ]

        // layer norm 2
        struct ggml_tensor* norm2_w;  // [in_channels, ]
        struct ggml_tensor* norm2_b;  // [in_channels, ]

        // attn2
        struct ggml_tensor* attn2_q_w;  // [in_channels, in_channels]
        struct ggml_tensor* attn2_k_w;  // [in_channels, context_dim]
        struct ggml_tensor* attn2_v_w;  // [in_channels, context_dim]

        struct ggml_tensor* attn2_out_w;  // [in_channels, in_channels]
        struct ggml_tensor* attn2_out_b;  // [in_channels, ]

        // layer norm 3
        struct ggml_tensor* norm3_w;  // [in_channels, ]
        struct ggml_tensor* norm3_b;  // [in_channels, ]

        // ff
        struct ggml_tensor* ff_0_proj_w;  // [in_channels * 4 * 2, in_channels]
        struct ggml_tensor* ff_0_proj_b;  // [in_channels * 4 * 2]

        struct ggml_tensor* ff_2_w;  // [in_channels, in_channels * 4]
        struct ggml_tensor* ff_2_b;  // [in_channels,]
    } transformer;

    // proj_out
    struct ggml_tensor* proj_out_w;  // [in_channels, in_channels, 1, 1]
    struct ggml_tensor* proj_out_b;  // [in_channels,]

    size_t compute_params_mem_size(ggml_type wtype) {
        double mem_size = 0;
        mem_size += 2 * in_channels * ggml_type_sizef(GGML_TYPE_F32);                        // norm_w/norm_b
        mem_size += 2 * in_channels * in_channels * 1 * 1 * ggml_type_sizef(GGML_TYPE_F16);  // proj_in_w/proj_out_w
        mem_size += 2 * in_channels * ggml_type_sizef(GGML_TYPE_F32);                        // proj_in_b/proj_out_b

        // transformer
        {
            mem_size += 6 * in_channels * ggml_type_sizef(GGML_TYPE_F32);            // norm1-3_w/b
            mem_size += 6 * in_channels * in_channels * ggml_type_sizef(wtype);      // attn1_q/k/v/out_w attn2_q/out_w
            mem_size += 2 * in_channels * context_dim * ggml_type_sizef(wtype);      // attn2_k/v_w
            mem_size += in_channels * 4 * 2 * in_channels * ggml_type_sizef(wtype);  // ff_0_proj_w
            mem_size += in_channels * 4 * 2 * ggml_type_sizef(GGML_TYPE_F32);        // ff_0_proj_b
            mem_size += in_channels * 4 * in_channels * ggml_type_sizef(wtype);      // ff_2_w
            mem_size += in_channels * ggml_type_sizef(GGML_TYPE_F32);                // ff_2_b
        }
        mem_size += 26 * ggml_tensor_overhead();  // object overhead
        return static_cast<size_t>(mem_size);
    }

    void init_params(struct ggml_context* ctx, ggml_type wtype) {
        norm_w    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);
        norm_b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);
        proj_in_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 1, 1, in_channels, in_channels);
        proj_in_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);

        proj_out_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 1, 1, in_channels, in_channels);
        proj_out_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);

        // transformer
        transformer.norm1_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);
        transformer.norm1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);

        transformer.attn1_q_w = ggml_new_tensor_2d(ctx, wtype, in_channels, in_channels);
        transformer.attn1_k_w = ggml_new_tensor_2d(ctx, wtype, in_channels, in_channels);
        transformer.attn1_v_w = ggml_new_tensor_2d(ctx, wtype, in_channels, in_channels);

        transformer.attn1_out_w = ggml_new_tensor_2d(ctx, wtype, in_channels, in_channels);
        transformer.attn1_out_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);

        transformer.norm2_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);
        transformer.norm2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);

        transformer.attn2_q_w = ggml_new_tensor_2d(ctx, wtype, in_channels, in_channels);
        transformer.attn2_k_w = ggml_new_tensor_2d(ctx, wtype, context_dim, in_channels);
        transformer.attn2_v_w = ggml_new_tensor_2d(ctx, wtype, context_dim, in_channels);

        transformer.attn2_out_w = ggml_new_tensor_2d(ctx, wtype, in_channels, in_channels);
        transformer.attn2_out_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);

        transformer.norm3_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);
        transformer.norm3_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);

        transformer.ff_0_proj_w = ggml_new_tensor_2d(ctx, wtype, in_channels, in_channels * 4 * 2);
        transformer.ff_0_proj_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels * 4 * 2);

        transformer.ff_2_w = ggml_new_tensor_2d(ctx, wtype, in_channels * 4, in_channels);
        transformer.ff_2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);
    }

    void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        tensors[prefix + "norm.weight"]    = norm_w;
        tensors[prefix + "norm.bias"]      = norm_b;
        tensors[prefix + "proj_in.weight"] = proj_in_w;
        tensors[prefix + "proj_in.bias"]   = proj_in_b;

        // transformer
        {
            std::string transformer_prefix                    = prefix + "transformer_blocks.0.";
            tensors[transformer_prefix + "attn1.to_q.weight"] = transformer.attn1_q_w;
            tensors[transformer_prefix + "attn1.to_k.weight"] = transformer.attn1_k_w;
            tensors[transformer_prefix + "attn1.to_v.weight"] = transformer.attn1_v_w;

            tensors[transformer_prefix + "attn1.to_out.0.weight"] = transformer.attn1_out_w;
            tensors[transformer_prefix + "attn1.to_out.0.bias"]   = transformer.attn1_out_b;

            tensors[transformer_prefix + "ff.net.0.proj.weight"] = transformer.ff_0_proj_w;
            tensors[transformer_prefix + "ff.net.0.proj.bias"]   = transformer.ff_0_proj_b;
            tensors[transformer_prefix + "ff.net.2.weight"]      = transformer.ff_2_w;
            tensors[transformer_prefix + "ff.net.2.bias"]        = transformer.ff_2_b;

            tensors[transformer_prefix + "attn2.to_q.weight"] = transformer.attn2_q_w;
            tensors[transformer_prefix + "attn2.to_k.weight"] = transformer.attn2_k_w;
            tensors[transformer_prefix + "attn2.to_v.weight"] = transformer.attn2_v_w;

            tensors[transformer_prefix + "attn2.to_out.0.weight"] = transformer.attn2_out_w;
            tensors[transformer_prefix + "attn2.to_out.0.bias"]   = transformer.attn2_out_b;

            tensors[transformer_prefix + "norm1.weight"] = transformer.norm1_w;
            tensors[transformer_prefix + "norm1.bias"]   = transformer.norm1_b;
            tensors[transformer_prefix + "norm2.weight"] = transformer.norm2_w;
            tensors[transformer_prefix + "norm2.bias"]   = transformer.norm2_b;
            tensors[transformer_prefix + "norm3.weight"] = transformer.norm3_w;
            tensors[transformer_prefix + "norm3.bias"]   = transformer.norm3_b;
        }

        tensors[prefix + "proj_out.weight"] = proj_out_w;
        tensors[prefix + "proj_out.bias"]   = proj_out_b;
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* context) {
        // x: [N, in_channels, h, w]
        // context: [N, max_position, hidden_size(aka context_dim)]

        auto x_in = x;
        // group norm 32
        x = ggml_group_norm_32(ctx, x);
        x = ggml_add(ctx,
                     ggml_mul(ctx, ggml_repeat(ctx, ggml_reshape_4d(ctx, norm_w, 1, 1, norm_w->ne[0], 1), x), x),
                     ggml_repeat(ctx, ggml_reshape_4d(ctx, norm_b, 1, 1, norm_b->ne[0], 1), x));
        // proj_in
        x = ggml_conv_2d(ctx, proj_in_w, x, 1, 1, 0, 0, 1, 1);
        x = ggml_add(ctx,
                     x,
                     ggml_repeat(ctx,
                                 ggml_reshape_4d(ctx, proj_in_b, 1, 1, proj_in_b->ne[0], 1),
                                 x));  // [N, in_channels, h, w]

        // transformer
        const int64_t n            = x->ne[3];
        const int64_t c            = x->ne[2];
        const int64_t h            = x->ne[1];
        const int64_t w            = x->ne[0];
        const int64_t max_position = context->ne[1];
        x                          = ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3));  // [N, h, w, in_channels]

        {
            auto r = x;
            // layer norm 1
            {
                x = ggml_reshape_2d(ctx, x, c, w * h * n);
                x = ggml_norm(ctx, x, EPS);
                x = ggml_add(ctx,
                             ggml_mul(ctx,
                                      ggml_repeat(ctx, transformer.norm1_w, x),
                                      x),
                             ggml_repeat(ctx, transformer.norm1_b, x));
            }

            // self-attention
            {
                x                     = ggml_reshape_2d(ctx, x, c, h * w * n);        // [N * h * w, in_channels]
                struct ggml_tensor* q = ggml_mul_mat(ctx, transformer.attn1_q_w, x);  // [N * h * w, in_channels]
                q                     = ggml_scale_inplace(ctx, q, ggml_new_f32(ctx, 1.0f / sqrt((float)d_head)));
                q                     = ggml_reshape_4d(ctx, q, d_head, n_head, h * w, n);   // [N, h * w, n_head, d_head]
                q                     = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));    // [N, n_head, h * w, d_head]
                q                     = ggml_reshape_3d(ctx, q, d_head, h * w, n_head * n);  // [N * n_head, h * w, d_head]

                struct ggml_tensor* k = ggml_mul_mat(ctx, transformer.attn1_k_w, x);         // [N * h * w, in_channels]
                k                     = ggml_reshape_4d(ctx, k, d_head, n_head, h * w, n);   // [N, h * w, n_head, d_head]
                k                     = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));    // [N, n_head, h * w, d_head]
                k                     = ggml_reshape_3d(ctx, k, d_head, h * w, n_head * n);  // [N * n_head, h * w, d_head]

                struct ggml_tensor* v = ggml_mul_mat(ctx, transformer.attn1_v_w, x);         // [N * h * w, in_channels]
                v                     = ggml_reshape_4d(ctx, v, d_head, n_head, h * w, n);   // [N, h * w, n_head, d_head]
                v                     = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));    // [N, n_head, d_head, h * w]
                v                     = ggml_reshape_3d(ctx, v, h * w, d_head, n_head * n);  // [N * n_head, d_head, h * w]

                struct ggml_tensor* kq = ggml_mul_mat(ctx, k, q);  // [N * n_head, h * w, h * w]
                // kq = ggml_diag_mask_inf_inplace(ctx, kq, 0);
                kq = ggml_soft_max_inplace(ctx, kq);

                struct ggml_tensor* kqv = ggml_mul_mat(ctx, v, kq);  // [N * n_head, h * w, d_head]
                kqv                     = ggml_reshape_4d(ctx, kqv, d_head, h * w, n_head, n);
                kqv                     = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));  // [N, h * w, n_head, d_head]

                // x = ggml_cpy(ctx, kqv, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_head * n_head, h * w * n));
                x = ggml_reshape_2d(ctx, kqv, d_head * n_head, h * w * n);

                x = ggml_add(ctx, ggml_repeat(ctx, transformer.attn1_out_b, x), ggml_mul_mat(ctx, transformer.attn1_out_w, x));

                x = ggml_reshape_4d(ctx, x, c, w, h, n);
            }

            x = ggml_add(ctx, x, r);
            r = x;

            // layer norm 2
            {
                x = ggml_norm(ctx, x, EPS);
                x = ggml_add(ctx,
                             ggml_mul(ctx,
                                      ggml_repeat(ctx, transformer.norm2_w, x), x),
                             ggml_repeat(ctx, transformer.norm2_b, x));
            }

            // cross-attention
            {
                x                     = ggml_reshape_2d(ctx, x, c, h * w * n);                                           // [N * h * w, in_channels]
                context               = ggml_reshape_2d(ctx, context, context->ne[0], context->ne[1] * context->ne[2]);  // [N * max_position, hidden_size]
                struct ggml_tensor* q = ggml_mul_mat(ctx, transformer.attn2_q_w, x);                                     // [N * h * w, in_channels]

                q = ggml_scale_inplace(ctx, q, ggml_new_f32(ctx, 1.0f / sqrt((float)d_head)));
                q = ggml_reshape_4d(ctx, q, d_head, n_head, h * w, n);   // [N, h * w, n_head, d_head]
                q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));    // [N, n_head, h * w, d_head]
                q = ggml_reshape_3d(ctx, q, d_head, h * w, n_head * n);  // [N * n_head, h * w, d_head]

                struct ggml_tensor* k = ggml_mul_mat(ctx, transformer.attn2_k_w, context);          // [N * max_position, in_channels]
                k                     = ggml_reshape_4d(ctx, k, d_head, n_head, max_position, n);   // [N, max_position, n_head, d_head]
                k                     = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));           // [N, n_head, max_position, d_head]
                k                     = ggml_reshape_3d(ctx, k, d_head, max_position, n_head * n);  // [N * n_head, max_position, d_head]

                struct ggml_tensor* v = ggml_mul_mat(ctx, transformer.attn2_v_w, context);          // [N * max_position, in_channels]
                v                     = ggml_reshape_4d(ctx, v, d_head, n_head, max_position, n);   // [N, max_position, n_head, d_head]
                v                     = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));           // [N, n_head, d_head, max_position]
                v                     = ggml_reshape_3d(ctx, v, max_position, d_head, n_head * n);  // [N * n_head, d_head, max_position]

                struct ggml_tensor* kq = ggml_mul_mat(ctx, k, q);  // [N * n_head, h * w, max_position]
                // kq = ggml_diag_mask_inf_inplace(ctx, kq, 0);
                kq = ggml_soft_max_inplace(ctx, kq);

                struct ggml_tensor* kqv = ggml_mul_mat(ctx, v, kq);  // [N * n_head, h * w, d_head]

                kqv = ggml_reshape_4d(ctx, kqv, d_head, h * w, n_head, n);
                kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));

                // x = ggml_cpy(ctx, kqv, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_head * n_head, h * w * n)); // [N * h * w, in_channels]
                x = ggml_reshape_2d(ctx, kqv, d_head * n_head, h * w * n);  // [N * h * w, in_channels]

                x = ggml_add(ctx, ggml_repeat(ctx, transformer.attn2_out_b, x), ggml_mul_mat(ctx, transformer.attn2_out_w, x));

                x = ggml_reshape_4d(ctx, x, c, w, h, n);
            }

            x = ggml_add(ctx, x, r);
            r = x;

            // layer norm 3
            {
                x = ggml_reshape_2d(ctx, x, c, h * w * n);  // [N * h * w, in_channels]
                x = ggml_norm(ctx, x, EPS);
                x = ggml_add(ctx,
                             ggml_mul(ctx,
                                      ggml_repeat(ctx, transformer.norm3_w, x), x),
                             ggml_repeat(ctx, transformer.norm3_b, x));
            }

            // ff
            {
                // GEGLU
                auto x_w    = ggml_view_2d(ctx,
                                           transformer.ff_0_proj_w,
                                           transformer.ff_0_proj_w->ne[0],
                                           transformer.ff_0_proj_w->ne[1] / 2,
                                           transformer.ff_0_proj_w->nb[1],
                                           0);  // [in_channels * 4, in_channels]
                auto x_b    = ggml_view_1d(ctx,
                                           transformer.ff_0_proj_b,
                                           transformer.ff_0_proj_b->ne[0] / 2,
                                           0);  // [in_channels * 4, in_channels]
                auto gate_w = ggml_view_2d(ctx,
                                           transformer.ff_0_proj_w,
                                           transformer.ff_0_proj_w->ne[0],
                                           transformer.ff_0_proj_w->ne[1] / 2,
                                           transformer.ff_0_proj_w->nb[1],
                                           transformer.ff_0_proj_w->nb[1] * transformer.ff_0_proj_w->ne[1] / 2);  // [in_channels * 4, ]
                auto gate_b = ggml_view_1d(ctx,
                                           transformer.ff_0_proj_b,
                                           transformer.ff_0_proj_b->ne[0] / 2,
                                           transformer.ff_0_proj_b->nb[0] * transformer.ff_0_proj_b->ne[0] / 2);  // [in_channels * 4, ]
                x           = ggml_reshape_2d(ctx, x, c, w * h * n);
                auto x_in   = x;
                x           = ggml_mul_mat(ctx, x_w, x_in);  // [N * h * w, in_channels * 4]
                x           = ggml_add(ctx, ggml_repeat(ctx, x_b, x), x);
                auto gate   = ggml_mul_mat(ctx, gate_w, x_in);  // [N * h * w, in_channels * 4]
                gate        = ggml_add(ctx, ggml_repeat(ctx, gate_b, gate), gate);

                gate = ggml_gelu_inplace(ctx, gate);

                x = ggml_mul(ctx, x, gate);  // [N * h * w, in_channels * 4]
                // fc
                x = ggml_mul_mat(ctx, transformer.ff_2_w, x);  // [N * h * w, in_channels]
                x = ggml_add(ctx, ggml_repeat(ctx, transformer.ff_2_b, x), x);
            }

            x = ggml_reshape_4d(ctx, x, c, w, h, n);  // [N, h, w, in_channels]

            // residual
            x = ggml_add(ctx, x, r);
        }
        x = ggml_cont(ctx, ggml_permute(ctx, x, 2, 0, 1, 3));  // // [N, in_channels, h, w]

        // proj_out
        x = ggml_conv_2d(ctx, proj_out_w, x, 1, 1, 0, 0, 1, 1);
        x = ggml_add(ctx,
                     x,
                     ggml_repeat(ctx,
                                 ggml_reshape_4d(ctx, proj_out_b, 1, 1, proj_out_b->ne[0], 1),
                                 x));  // [N, in_channels, h, w]
        x = ggml_add(ctx, x, x_in);
        return x;
    }
};

struct DownSample {
    // hparams
    int channels;
    int out_channels;

    // conv2d params
    struct ggml_tensor* op_w;  // [out_channels, channels, 3, 3]
    struct ggml_tensor* op_b;  // [out_channels,]

    bool vae_downsample = false;

    size_t compute_params_mem_size(ggml_type wtype) {
        double mem_size = 0;
        mem_size += out_channels * channels * 3 * 3 * ggml_type_sizef(GGML_TYPE_F16);  // op_w
        mem_size += out_channels * ggml_type_sizef(GGML_TYPE_F32);                     // op_b
        mem_size += 2 * ggml_tensor_overhead();                                        // object overhead
        return static_cast<size_t>(mem_size);
    }

    void init_params(struct ggml_context* ctx, ggml_type wtype) {
        op_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, channels, out_channels);
        op_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
    }

    void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        if (vae_downsample) {
            tensors[prefix + "conv.weight"] = op_w;
            tensors[prefix + "conv.bias"]   = op_b;
        } else {
            tensors[prefix + "op.weight"] = op_w;
            tensors[prefix + "op.bias"]   = op_b;
        }
    }

    // TODO: making it parallel
    static void asymmetric_pad(struct ggml_tensor* dst,
                               const struct ggml_tensor* a,
                               const struct ggml_tensor* b,
                               int ith,
                               int nth,
                               void* userdata) {
        assert(sizeof(dst->nb[0]) == sizeof(float));
        assert(sizeof(a->nb[0]) == sizeof(float));
        assert(sizeof(b->nb[0]) == sizeof(float));
        float value = 0;

        for (int i = 0; i < dst->ne[3]; i++) {
            for (int j = 0; j < dst->ne[2]; j++) {
                for (int k = 0; k < dst->ne[1]; k++) {
                    for (int l = 0; l < dst->ne[0]; l++) {
                        if (k == dst->ne[1] - 1 || l == dst->ne[0] - 1) {
                            value = 0;
                        } else {
                            value = ggml_tensor_get_f32(b, l, k, j, i);
                        }
                        // printf("%d %d %d %d -> %f\n", i, j, k, l, value);
                        ggml_tensor_set_f32(dst, value, l, k, j, i);
                    }
                }
            }
        }
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [N, channels, h, w]
        if (vae_downsample) {
            bool dynamic = ggml_get_dynamic(ctx);
            ggml_set_dynamic(ctx, false);
            auto pad_x = ggml_new_tensor_4d(ctx, x->type, x->ne[0] + 1, x->ne[1] + 1, x->ne[2], x->ne[3]);
            ggml_set_dynamic(ctx, dynamic);

            x = ggml_map_custom2_inplace(ctx, pad_x, x, asymmetric_pad, 1, NULL);
            x = ggml_conv_2d(ctx, op_w, x, 2, 2, 0, 0, 1, 1);
        } else {
            x = ggml_conv_2d(ctx, op_w, x, 2, 2, 1, 1, 1, 1);
        }
        x = ggml_add(ctx,
                     x,
                     ggml_repeat(ctx,
                                 ggml_reshape_4d(ctx, op_b, 1, 1, op_b->ne[0], 1),
                                 x));  // [N, out_channels, h/2, w/2]
        return x;
    }
};

struct UpSample {
    // hparams
    int channels;
    int out_channels;

    // conv2d params
    struct ggml_tensor* conv_w;  // [out_channels, channels, 3, 3]
    struct ggml_tensor* conv_b;  // [out_channels,]

    size_t compute_params_mem_size(ggml_type wtype) {
        double mem_size = 0;
        mem_size += out_channels * channels * 3 * 3 * ggml_type_sizef(GGML_TYPE_F16);  // op_w
        mem_size += out_channels * ggml_type_sizef(GGML_TYPE_F32);                     // op_b
        mem_size += 2 * ggml_tensor_overhead();                                        // object overhead
        return static_cast<size_t>(mem_size);
    }

    void init_params(struct ggml_context* ctx, ggml_type wtype) {
        conv_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, channels, out_channels);
        conv_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
    }

    void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        tensors[prefix + "conv.weight"] = conv_w;
        tensors[prefix + "conv.bias"]   = conv_b;
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [N, channels, h, w]
        x = ggml_upscale(ctx, x, 2);  // [N, channels, h*2, w*2]
        x = ggml_conv_2d(ctx, conv_w, x, 1, 1, 1, 1, 1, 1);

        x = ggml_add(ctx,
                     x,
                     ggml_repeat(ctx,
                                 ggml_reshape_4d(ctx, conv_b, 1, 1, conv_b->ne[0], 1),
                                 x));  // [N, out_channels, h*2, w*2]
        return x;
    }
};

// ldm.modules.diffusionmodules.openaimodel.UNetModel
struct UNetModel {
    // network hparams
    int in_channels              = 4;
    int model_channels           = 320;
    int out_channels             = 4;
    int num_res_blocks           = 2;
    int attention_resolutions[3] = {4, 2, 1};
    int channel_mult[4]          = {1, 2, 4, 4};
    int time_embed_dim           = 1280;  // model_channels*4
    int num_heads                = 8;
    int num_head_channels        = -1;   // channels // num_heads
    int context_dim              = 768;  // 1024 for SD2.x

    // network params
    struct ggml_tensor* time_embed_0_w;  // [time_embed_dim, model_channels]
    struct ggml_tensor* time_embed_0_b;  // [time_embed_dim, ]
    // time_embed_1 is nn.SILU()
    struct ggml_tensor* time_embed_2_w;  // [time_embed_dim, time_embed_dim]
    struct ggml_tensor* time_embed_2_b;  // [time_embed_dim, ]

    struct ggml_tensor* input_block_0_w;  // [model_channels, in_channels, 3, 3]
    struct ggml_tensor* input_block_0_b;  // [model_channels, ]

    // input_blocks
    ResBlock input_res_blocks[4][2];
    SpatialTransformer input_transformers[3][2];
    DownSample input_down_samples[3];

    // middle_block
    ResBlock middle_block_0;
    SpatialTransformer middle_block_1;
    ResBlock middle_block_2;

    // output_blocks
    ResBlock output_res_blocks[4][3];
    SpatialTransformer output_transformers[3][3];
    UpSample output_up_samples[3];

    // out
    // group norm 32
    struct ggml_tensor* out_0_w;  // [model_channels, ]
    struct ggml_tensor* out_0_b;  // [model_channels, ]
    // out 1 is nn.SILU()
    struct ggml_tensor* out_2_w;  // [out_channels, model_channels, 3, 3]
    struct ggml_tensor* out_2_b;  // [out_channels, ]

    UNetModel(ModelType model_type = SD1) {
        if (model_type == SD2) {
            context_dim       = 1024;
            num_head_channels = 64;
            num_heads         = -1;
        }
        // set up hparams of blocks

        // input_blocks
        std::vector<int> input_block_chans;
        input_block_chans.push_back(model_channels);
        int ch = model_channels;
        int ds = 1;

        int len_mults = sizeof(channel_mult) / sizeof(int);
        for (int i = 0; i < len_mults; i++) {
            int mult = channel_mult[i];
            for (int j = 0; j < num_res_blocks; j++) {
                input_res_blocks[i][j].channels     = ch;
                input_res_blocks[i][j].emb_channels = time_embed_dim;
                input_res_blocks[i][j].out_channels = mult * model_channels;

                ch = mult * model_channels;

                if (ds == attention_resolutions[0] || ds == attention_resolutions[1] || ds == attention_resolutions[2]) {
                    int n_head = num_heads;
                    int d_head = ch / num_heads;
                    if (num_head_channels != -1) {
                        d_head = num_head_channels;
                        n_head = ch / d_head;
                    }
                    input_transformers[i][j].in_channels = ch;
                    input_transformers[i][j].n_head      = n_head;
                    input_transformers[i][j].d_head      = d_head;
                    input_transformers[i][j].context_dim = context_dim;
                }
                input_block_chans.push_back(ch);
            }
            if (i != len_mults - 1) {
                input_down_samples[i].channels     = ch;
                input_down_samples[i].out_channels = ch;
                input_block_chans.push_back(ch);

                ds *= 2;
            }
        }

        // middle blocks
        middle_block_0.channels     = ch;
        middle_block_0.emb_channels = time_embed_dim;
        middle_block_0.out_channels = ch;

        int n_head = num_heads;
        int d_head = ch / num_heads;
        if (num_head_channels != -1) {
            d_head = num_head_channels;
            n_head = ch / d_head;
        }
        middle_block_1.in_channels = ch;
        middle_block_1.n_head      = n_head;
        middle_block_1.d_head      = d_head;
        middle_block_1.context_dim = context_dim;

        middle_block_2.channels     = ch;
        middle_block_2.emb_channels = time_embed_dim;
        middle_block_2.out_channels = ch;

        // output blocks
        for (int i = len_mults - 1; i >= 0; i--) {
            int mult = channel_mult[i];
            for (int j = 0; j < num_res_blocks + 1; j++) {
                int ich = input_block_chans.back();
                input_block_chans.pop_back();

                output_res_blocks[i][j].channels     = ch + ich;
                output_res_blocks[i][j].emb_channels = time_embed_dim;
                output_res_blocks[i][j].out_channels = mult * model_channels;

                ch = mult * model_channels;

                if (ds == attention_resolutions[0] || ds == attention_resolutions[1] || ds == attention_resolutions[2]) {
                    int n_head = num_heads;
                    int d_head = ch / num_heads;
                    if (num_head_channels != -1) {
                        d_head = num_head_channels;
                        n_head = ch / d_head;
                    }
                    output_transformers[i][j].in_channels = ch;
                    output_transformers[i][j].n_head      = n_head;
                    output_transformers[i][j].d_head      = d_head;
                    output_transformers[i][j].context_dim = context_dim;
                }

                if (i > 0 && j == num_res_blocks) {
                    output_up_samples[i - 1].channels     = ch;
                    output_up_samples[i - 1].out_channels = ch;

                    ds /= 2;
                }
            }
        }
    }

    size_t compute_params_mem_size(ggml_type wtype) {
        double mem_size = 0;
        mem_size += time_embed_dim * model_channels * ggml_type_sizef(wtype);  // time_embed_0_w
        mem_size += time_embed_dim * ggml_type_sizef(GGML_TYPE_F32);           // time_embed_0_b
        mem_size += time_embed_dim * time_embed_dim * ggml_type_sizef(wtype);  // time_embed_2_w
        mem_size += time_embed_dim * ggml_type_sizef(GGML_TYPE_F32);           // time_embed_2_b

        mem_size += model_channels * in_channels * 3 * 3 * ggml_type_sizef(GGML_TYPE_F16);  // input_block_0_w
        mem_size += model_channels * ggml_type_sizef(GGML_TYPE_F32);                        // input_block_0_b

        mem_size += 6 * ggml_tensor_overhead();  // object overhead

        // input_blocks
        int ds        = 1;
        int len_mults = sizeof(channel_mult) / sizeof(int);
        for (int i = 0; i < len_mults; i++) {
            for (int j = 0; j < num_res_blocks; j++) {
                mem_size += input_res_blocks[i][j].compute_params_mem_size(wtype);
                if (ds == attention_resolutions[0] || ds == attention_resolutions[1] || ds == attention_resolutions[2]) {
                    mem_size += input_transformers[i][j].compute_params_mem_size(wtype);
                }
            }
            if (i != len_mults - 1) {
                ds *= 2;
                mem_size += input_down_samples[i].compute_params_mem_size(wtype);
            }
        }

        // middle_block
        mem_size += middle_block_0.compute_params_mem_size(wtype);
        mem_size += middle_block_1.compute_params_mem_size(wtype);
        mem_size += middle_block_2.compute_params_mem_size(wtype);

        // output_blocks
        for (int i = len_mults - 1; i >= 0; i--) {
            for (int j = 0; j < num_res_blocks + 1; j++) {
                mem_size += output_res_blocks[i][j].compute_params_mem_size(wtype);

                if (ds == attention_resolutions[0] || ds == attention_resolutions[1] || ds == attention_resolutions[2]) {
                    mem_size += output_transformers[i][j].compute_params_mem_size(wtype);
                }

                if (i > 0 && j == num_res_blocks) {
                    mem_size += output_up_samples[i - 1].compute_params_mem_size(wtype);

                    ds /= 2;
                }
            }
        }

        // out
        mem_size += 2 * model_channels * ggml_type_sizef(GGML_TYPE_F32);                     // out_0_w/b
        mem_size += out_channels * model_channels * 3 * 3 * ggml_type_sizef(GGML_TYPE_F16);  // out_2_w
        mem_size += out_channels * ggml_type_sizef(GGML_TYPE_F32);                           // out_2_b

        mem_size += 4 * ggml_tensor_overhead();

        return static_cast<size_t>(mem_size);
    }

    void init_params(struct ggml_context* ctx, ggml_type wtype) {
        time_embed_0_w = ggml_new_tensor_2d(ctx, wtype, model_channels, time_embed_dim);
        time_embed_0_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, time_embed_dim);

        time_embed_2_w = ggml_new_tensor_2d(ctx, wtype, time_embed_dim, time_embed_dim);
        time_embed_2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, time_embed_dim);

        // input_blocks
        input_block_0_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, in_channels, model_channels);
        input_block_0_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, model_channels);
        int ds          = 1;
        int len_mults   = sizeof(channel_mult) / sizeof(int);
        for (int i = 0; i < len_mults; i++) {
            for (int j = 0; j < num_res_blocks; j++) {
                input_res_blocks[i][j].init_params(ctx, wtype);
                if (ds == attention_resolutions[0] || ds == attention_resolutions[1] || ds == attention_resolutions[2]) {
                    input_transformers[i][j].init_params(ctx, wtype);
                }
            }
            if (i != len_mults - 1) {
                input_down_samples[i].init_params(ctx, wtype);
                ds *= 2;
            }
        }

        // middle_blocks
        middle_block_0.init_params(ctx, wtype);
        middle_block_1.init_params(ctx, wtype);
        middle_block_2.init_params(ctx, wtype);

        // output_blocks
        for (int i = len_mults - 1; i >= 0; i--) {
            for (int j = 0; j < num_res_blocks + 1; j++) {
                output_res_blocks[i][j].init_params(ctx, wtype);

                if (ds == attention_resolutions[0] || ds == attention_resolutions[1] || ds == attention_resolutions[2]) {
                    output_transformers[i][j].init_params(ctx, wtype);
                }

                if (i > 0 && j == num_res_blocks) {
                    output_up_samples[i - 1].init_params(ctx, wtype);

                    ds /= 2;
                }
            }
        }

        // out
        out_0_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, model_channels);
        out_0_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, model_channels);

        out_2_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, model_channels, out_channels);
        out_2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
    }

    void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        tensors[prefix + "time_embed.0.weight"] = time_embed_0_w;
        tensors[prefix + "time_embed.0.bias"]   = time_embed_0_b;

        tensors[prefix + "time_embed.2.weight"] = time_embed_2_w;
        tensors[prefix + "time_embed.2.bias"]   = time_embed_2_b;

        // input_blocks
        tensors[prefix + "input_blocks.0.0.weight"] = input_block_0_w;
        tensors[prefix + "input_blocks.0.0.bias"]   = input_block_0_b;

        int len_mults       = sizeof(channel_mult) / sizeof(int);
        int input_block_idx = 0;
        int ds              = 1;
        for (int i = 0; i < len_mults; i++) {
            for (int j = 0; j < num_res_blocks; j++) {
                input_block_idx += 1;

                input_res_blocks[i][j].map_by_name(tensors, prefix + "input_blocks." + std::to_string(input_block_idx) + ".0.");
                if (ds == attention_resolutions[0] || ds == attention_resolutions[1] || ds == attention_resolutions[2]) {
                    input_transformers[i][j].map_by_name(tensors, prefix + "input_blocks." + std::to_string(input_block_idx) + ".1.");
                }
            }
            if (i != len_mults - 1) {
                input_block_idx += 1;
                input_down_samples[i].map_by_name(tensors, prefix + "input_blocks." + std::to_string(input_block_idx) + ".0.");
                ds *= 2;
            }
        }

        // middle_blocks
        middle_block_0.map_by_name(tensors, prefix + "middle_block.0.");
        middle_block_1.map_by_name(tensors, prefix + "middle_block.1.");
        middle_block_2.map_by_name(tensors, prefix + "middle_block.2.");

        // output_blocks
        int output_block_idx = 0;
        for (int i = len_mults - 1; i >= 0; i--) {
            for (int j = 0; j < num_res_blocks + 1; j++) {
                output_res_blocks[i][j].map_by_name(tensors, prefix + "output_blocks." + std::to_string(output_block_idx) + ".0.");

                int up_sample_idx = 1;
                if (ds == attention_resolutions[0] || ds == attention_resolutions[1] || ds == attention_resolutions[2]) {
                    output_transformers[i][j].map_by_name(tensors, prefix + "output_blocks." + std::to_string(output_block_idx) + ".1.");
                    up_sample_idx++;
                }

                if (i > 0 && j == num_res_blocks) {
                    output_up_samples[i - 1].map_by_name(tensors, prefix + "output_blocks." + std::to_string(output_block_idx) + "." + std::to_string(up_sample_idx) + ".");

                    ds /= 2;
                }
                output_block_idx += 1;
            }
        }

        // out
        tensors[prefix + "out.0.weight"] = out_0_w;
        tensors[prefix + "out.0.bias"]   = out_0_b;
        tensors[prefix + "out.2.weight"] = out_2_w;
        tensors[prefix + "out.2.bias"]   = out_2_b;
    }

    struct ggml_tensor* forward(struct ggml_context* ctx,
                                struct ggml_tensor* x,
                                struct ggml_tensor* timesteps,
                                struct ggml_tensor* context,
                                struct ggml_tensor* t_emb = NULL) {
        // x: [N, in_channels, h, w]
        // timesteps: [N, ]
        // t_emb: [N, model_channels]
        // context: [N, max_position, hidden_size]([N, 77, 768])
        if (t_emb == NULL && timesteps != NULL) {
            t_emb = new_timestep_embedding(ctx, timesteps, model_channels);  // [N, model_channels]
        }

        // time_embed
        auto emb = ggml_mul_mat(ctx, time_embed_0_w, t_emb);
        emb      = ggml_add(ctx, ggml_repeat(ctx, time_embed_0_b, emb), emb);
        emb      = ggml_silu_inplace(ctx, emb);
        emb      = ggml_mul_mat(ctx, time_embed_2_w, emb);
        emb      = ggml_add(ctx, ggml_repeat(ctx, time_embed_2_b, emb), emb);  // [N, time_embed_dim]

        // input_blocks
        std::vector<struct ggml_tensor*> hs;
        // input block 0
        auto h = ggml_conv_2d(ctx, input_block_0_w, x, 1, 1, 1, 1, 1, 1);  // [N, model_channels, h, w]
        h      = ggml_add(ctx,
                          h,
                          ggml_repeat(ctx,
                                      ggml_reshape_4d(ctx, input_block_0_b, 1, 1, input_block_0_b->ne[0], 1),
                                      h));  // [N, model_channels, h, w]
        hs.push_back(h);
        // input block 1-11
        int len_mults = sizeof(channel_mult) / sizeof(int);
        int ds        = 1;
        for (int i = 0; i < len_mults; i++) {
            int mult = channel_mult[i];
            for (int j = 0; j < num_res_blocks; j++) {
                h = input_res_blocks[i][j].forward(ctx, h, emb);  // [N, mult*model_channels, h, w]
                if (ds == attention_resolutions[0] || ds == attention_resolutions[1] || ds == attention_resolutions[2]) {
                    h = input_transformers[i][j].forward(ctx, h, context);  // [N, mult*model_channels, h, w]
                }
                hs.push_back(h);
            }
            if (i != len_mults - 1) {
                ds *= 2;
                h = input_down_samples[i].forward(ctx, h);  // [N, mult*model_channels, h/(2^(i+1)), w/(2^(i+1))]
                hs.push_back(h);
            }
        }
        // [N, 4*model_channels, h/8, w/8]

        // middle_block
        h = middle_block_0.forward(ctx, h, emb);      // [N, 4*model_channels, h/8, w/8]
        h = middle_block_1.forward(ctx, h, context);  // [N, 4*model_channels, h/8, w/8]
        h = middle_block_2.forward(ctx, h, emb);      // [N, 4*model_channels, h/8, w/8]

        // output_blocks
        for (int i = len_mults - 1; i >= 0; i--) {
            for (int j = 0; j < num_res_blocks + 1; j++) {
                auto h_skip = hs.back();
                hs.pop_back();

                h = ggml_concat(ctx, h, h_skip);
                h = output_res_blocks[i][j].forward(ctx, h, emb);

                if (ds == attention_resolutions[0] || ds == attention_resolutions[1] || ds == attention_resolutions[2]) {
                    h = output_transformers[i][j].forward(ctx, h, context);
                }

                if (i > 0 && j == num_res_blocks) {
                    h = output_up_samples[i - 1].forward(ctx, h);

                    ds /= 2;
                }
            }
        }

        // out
        // group norm 32
        h = ggml_group_norm_32(ctx, h);
        h = ggml_add(ctx,
                     ggml_mul(ctx,
                              ggml_repeat(ctx,
                                          ggml_reshape_4d(ctx, out_0_w, 1, 1, out_0_w->ne[0], 1),
                                          h),
                              h),
                     ggml_repeat(ctx,
                                 ggml_reshape_4d(ctx, out_0_b, 1, 1, out_0_b->ne[0], 1),
                                 h));
        // silu
        h = ggml_silu_inplace(ctx, h);
        // conv2d
        h = ggml_conv_2d(ctx, out_2_w, h, 1, 1, 1, 1, 1, 1);
        h = ggml_add(ctx,
                     h,
                     ggml_repeat(ctx,
                                 ggml_reshape_4d(ctx, out_2_b, 1, 1, out_2_b->ne[0], 1),
                                 h));  // [N, out_channels, h, w]

        return h;
    }
};

/*================================================== AutoEncoderKL ===================================================*/

struct ResnetBlock {
    // network hparams
    int in_channels;
    int out_channels;

    // network params
    struct ggml_tensor* norm1_w;  // [in_channels, ]
    struct ggml_tensor* norm1_b;  // [in_channels, ]

    struct ggml_tensor* conv1_w;  // [out_channels, in_channels, 3, 3]
    struct ggml_tensor* conv1_b;  // [out_channels, ]

    struct ggml_tensor* norm2_w;  // [out_channels, ]
    struct ggml_tensor* norm2_b;  // [out_channels, ]

    struct ggml_tensor* conv2_w;  // [out_channels, out_channels, 3, 3]
    struct ggml_tensor* conv2_b;  // [out_channels, ]

    // nin_shortcut, only if out_channels != in_channels
    struct ggml_tensor* nin_shortcut_w;  // [out_channels, in_channels, 1, 1]
    struct ggml_tensor* nin_shortcut_b;  // [out_channels, ]

    size_t compute_params_mem_size(ggml_type wtype) {
        double mem_size = 0;
        mem_size += 2 * in_channels * ggml_type_sizef(GGML_TYPE_F32);                      // norm1_w/b
        mem_size += out_channels * in_channels * 3 * 3 * ggml_type_sizef(GGML_TYPE_F16);   // conv1_w
        mem_size += 4 * out_channels * ggml_type_sizef(GGML_TYPE_F32);                     // conv1_b/norm2_w/norm2_b/conv2_b
        mem_size += out_channels * out_channels * 3 * 3 * ggml_type_sizef(GGML_TYPE_F16);  // conv2_w

        mem_size += 8 * ggml_tensor_overhead();  // object overhead

        if (out_channels != in_channels) {
            mem_size += out_channels * in_channels * 1 * 1 * ggml_type_sizef(GGML_TYPE_F16);  // nin_shortcut_w
            mem_size += out_channels * ggml_type_sizef(GGML_TYPE_F32);                        // nin_shortcut_b

            mem_size += 2 * ggml_tensor_overhead();  // object overhead
        }
        return static_cast<size_t>(mem_size);
    }

    void init_params(struct ggml_context* ctx, ggml_type wtype) {
        norm1_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);
        norm1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);
        conv1_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, in_channels, out_channels);
        conv1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);

        norm2_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
        norm2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
        conv2_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, out_channels, out_channels);
        conv2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);

        if (out_channels != in_channels) {
            nin_shortcut_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 1, 1, in_channels, out_channels);
            nin_shortcut_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
        }
    }

    void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        tensors[prefix + "norm1.weight"] = norm1_w;
        tensors[prefix + "norm1.bias"]   = norm1_b;
        tensors[prefix + "conv1.weight"] = conv1_w;
        tensors[prefix + "conv1.bias"]   = conv1_b;

        tensors[prefix + "norm2.weight"] = norm2_w;
        tensors[prefix + "norm2.bias"]   = norm2_b;
        tensors[prefix + "conv2.weight"] = conv2_w;
        tensors[prefix + "conv2.bias"]   = conv2_b;

        if (out_channels != in_channels) {
            tensors[prefix + "nin_shortcut.weight"] = nin_shortcut_w;
            tensors[prefix + "nin_shortcut.bias"]   = nin_shortcut_b;
        }
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* z) {
        // z: [N, in_channels, h, w]

        // group norm 32
        auto h = ggml_group_norm_32(ctx, z);
        h      = ggml_mul(ctx,
                          ggml_repeat(ctx,
                                      ggml_reshape_4d(ctx, norm1_w, 1, 1, norm1_w->ne[0], 1),
                                      h),
                          h);
        h      = ggml_add(ctx,
                          h,
                          ggml_repeat(ctx,
                                      ggml_reshape_4d(ctx, norm1_b, 1, 1, norm1_b->ne[0], 1),
                                      h));
        // silu
        h = ggml_silu_inplace(ctx, h);
        // conv2d
        h = ggml_conv_2d(ctx, conv1_w, h, 1, 1, 1, 1, 1, 1);
        h = ggml_add(ctx,
                     h,
                     ggml_repeat(ctx,
                                 ggml_reshape_4d(ctx, conv1_b, 1, 1, conv1_b->ne[0], 1),
                                 h));  // [N, out_channels, h, w]

        // group norm 32
        h = ggml_group_norm_32(ctx, h);
        h = ggml_add(ctx,
                     ggml_mul(ctx, ggml_repeat(ctx, ggml_reshape_4d(ctx, norm2_w, 1, 1, norm2_w->ne[0], 1), h), h),
                     ggml_repeat(ctx, ggml_reshape_4d(ctx, norm2_b, 1, 1, norm2_b->ne[0], 1), h));
        // silu
        h = ggml_silu_inplace(ctx, h);
        // dropout, skip for inference
        // conv2d
        h = ggml_conv_2d(ctx, conv2_w, h, 1, 1, 1, 1, 1, 1);
        h = ggml_add(ctx,
                     h,
                     ggml_repeat(ctx,
                                 ggml_reshape_4d(ctx, conv2_b, 1, 1, conv2_b->ne[0], 1),
                                 h));  // [N, out_channels, h, w

        // skip connection
        if (out_channels != in_channels) {
            z = ggml_conv_2d(ctx, nin_shortcut_w, z, 1, 1, 0, 0, 1, 1);
            z = ggml_add(ctx,
                         z,
                         ggml_repeat(ctx,
                                     ggml_reshape_4d(ctx, nin_shortcut_b, 1, 1, nin_shortcut_b->ne[0], 1),
                                     z));  // [N, out_channels, h, w]
        }
        h = ggml_add(ctx, h, z);
        return h;  // [N, out_channels, h, w]
    }
};

struct AttnBlock {
    int in_channels;  // mult * model_channels

    // group norm
    struct ggml_tensor* norm_w;  // [in_channels,]
    struct ggml_tensor* norm_b;  // [in_channels,]

    // q/k/v
    struct ggml_tensor* q_w;  // [in_channels, in_channels, 1, 1]
    struct ggml_tensor* q_b;  // [in_channels,]
    struct ggml_tensor* k_w;  // [in_channels, in_channels, 1, 1]
    struct ggml_tensor* k_b;  // [in_channels,]
    struct ggml_tensor* v_w;  // [in_channels, in_channels, 1, 1]
    struct ggml_tensor* v_b;  // [in_channels,]

    // proj_out
    struct ggml_tensor* proj_out_w;  // [in_channels, in_channels, 1, 1]
    struct ggml_tensor* proj_out_b;  // [in_channels,]

    size_t compute_params_mem_size(ggml_type wtype) {
        double mem_size = 0;
        mem_size += 6 * in_channels * ggml_type_sizef(GGML_TYPE_F32);                        // norm_w/norm_b/q_b/k_v/v_b/proj_out_b
        mem_size += 4 * in_channels * in_channels * 1 * 1 * ggml_type_sizef(GGML_TYPE_F16);  // q_w/k_w/v_w/proj_out_w
        mem_size += 10 * ggml_tensor_overhead();                                             // object overhead
        return static_cast<size_t>(mem_size);
    }

    void init_params(struct ggml_context* ctx, ggml_type wtype) {
        norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);
        norm_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);
        q_w    = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 1, 1, in_channels, in_channels);
        q_b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);
        k_w    = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 1, 1, in_channels, in_channels);
        k_b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);
        v_w    = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 1, 1, in_channels, in_channels);
        v_b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);

        proj_out_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 1, 1, in_channels, in_channels);
        proj_out_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);
    }

    void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        tensors[prefix + "norm.weight"]     = norm_w;
        tensors[prefix + "norm.bias"]       = norm_b;
        tensors[prefix + "q.weight"]        = q_w;
        tensors[prefix + "q.bias"]          = q_b;
        tensors[prefix + "k.weight"]        = k_w;
        tensors[prefix + "k.bias"]          = k_b;
        tensors[prefix + "v.weight"]        = v_w;
        tensors[prefix + "v.bias"]          = v_b;
        tensors[prefix + "proj_out.weight"] = proj_out_w;
        tensors[prefix + "proj_out.bias"]   = proj_out_b;
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [N, in_channels, h, w]

        // group norm 32
        auto h_ = ggml_group_norm_32(ctx, x);
        h_      = ggml_add(ctx,
                           ggml_mul(ctx, ggml_repeat(ctx, ggml_reshape_4d(ctx, norm_w, 1, 1, norm_w->ne[0], 1), h_), h_),
                           ggml_repeat(ctx, ggml_reshape_4d(ctx, norm_b, 1, 1, norm_b->ne[0], 1), h_));

        const int64_t n = h_->ne[3];
        const int64_t c = h_->ne[2];
        const int64_t h = h_->ne[1];
        const int64_t w = h_->ne[0];
        // q
        auto q = ggml_conv_2d(ctx, q_w, h_, 1, 1, 0, 0, 1, 1);
        q      = ggml_add(ctx,
                          q,
                          ggml_repeat(ctx,
                                      ggml_reshape_4d(ctx, q_b, 1, 1, q_b->ne[0], 1),
                                      q));  // [N, in_channels, h, w]

        // k
        auto k = ggml_conv_2d(ctx, k_w, h_, 1, 1, 0, 0, 1, 1);
        k      = ggml_add(ctx,
                          k,
                          ggml_repeat(ctx,
                                      ggml_reshape_4d(ctx, k_b, 1, 1, k_b->ne[0], 1),
                                      k));  // [N, in_channels, h, w]

        // v
        auto v = ggml_conv_2d(ctx, v_w, h_, 1, 1, 0, 0, 1, 1);
        v      = ggml_add(ctx,
                          v,
                          ggml_repeat(ctx,
                                      ggml_reshape_4d(ctx, v_b, 1, 1, v_b->ne[0], 1),
                                      v));  // [N, in_channels, h, w]

        q = ggml_cont(ctx, ggml_permute(ctx, q, 1, 2, 0, 3));  // [N, h, w, in_channels]
        q = ggml_reshape_3d(ctx, q, c, h * w, n);              // [N, h * w, in_channels]

        k = ggml_cont(ctx, ggml_permute(ctx, k, 1, 2, 0, 3));  // [N, h, w, in_channels]
        k = ggml_reshape_3d(ctx, k, c, h * w, n);              // [N, h * w, in_channels]

        auto w_ = ggml_mul_mat(ctx, k, q);  // [N, h * w, h * w]
        w_      = ggml_scale_inplace(ctx, w_, ggml_new_f32(ctx, 1.0f / sqrt((float)c)));
        w_      = ggml_soft_max_inplace(ctx, w_);

        v  = ggml_reshape_3d(ctx, v, h * w, c, n);               // [N, in_channels, h * w]
        h_ = ggml_mul_mat(ctx, v, w_);                           // [N, h * w, in_channels]
        h_ = ggml_cont(ctx, ggml_permute(ctx, h_, 1, 0, 2, 3));  // [N, in_channels, h * w]
        h_ = ggml_reshape_4d(ctx, h_, w, h, c, n);               // [N, in_channels, h, w]

        // proj_out
        h_ = ggml_conv_2d(ctx, proj_out_w, h_, 1, 1, 0, 0, 1, 1);
        h_ = ggml_add(ctx,
                      h_,
                      ggml_repeat(ctx,
                                  ggml_reshape_4d(ctx, proj_out_b, 1, 1, proj_out_b->ne[0], 1),
                                  h_));  // [N, in_channels, h, w]
        h_ = ggml_add(ctx, h_, x);
        return h_;
    }
};

// ldm.modules.diffusionmodules.model.Encoder
struct Encoder {
    int embed_dim      = 4;
    int ch             = 128;
    int z_channels     = 4;
    int in_channels    = 3;
    int num_res_blocks = 2;
    int ch_mult[4]     = {1, 2, 4, 4};

    struct ggml_tensor* conv_in_w;  // [ch, in_channels, 3, 3]
    struct ggml_tensor* conv_in_b;  // [ch, ]

    ResnetBlock down_blocks[4][2];
    DownSample down_samples[3];

    struct
    {
        ResnetBlock block_1;
        AttnBlock attn_1;
        ResnetBlock block_2;
    } mid;

    // block_in = ch * ch_mult[len_mults - 1]
    struct ggml_tensor* norm_out_w;  // [block_in, ]
    struct ggml_tensor* norm_out_b;  // [block_in, ]

    struct ggml_tensor* conv_out_w;  // [embed_dim*2, block_in, 3, 3]
    struct ggml_tensor* conv_out_b;  // [embed_dim*2, ]

    Encoder() {
        int len_mults = sizeof(ch_mult) / sizeof(int);

        int block_in = 1;
        for (int i = 0; i < len_mults; i++) {
            if (i == 0) {
                block_in = ch;
            } else {
                block_in = ch * ch_mult[i - 1];
            }
            int block_out = ch * ch_mult[i];
            for (int j = 0; j < num_res_blocks; j++) {
                down_blocks[i][j].in_channels  = block_in;
                down_blocks[i][j].out_channels = block_out;
                block_in                       = block_out;
            }
            if (i != len_mults - 1) {
                down_samples[i].channels       = block_in;
                down_samples[i].out_channels   = block_in;
                down_samples[i].vae_downsample = true;
            }
        }

        mid.block_1.in_channels  = block_in;
        mid.block_1.out_channels = block_in;
        mid.attn_1.in_channels   = block_in;
        mid.block_2.in_channels  = block_in;
        mid.block_2.out_channels = block_in;
    }

    size_t compute_params_mem_size(ggml_type wtype) {
        double mem_size = 0;
        int len_mults   = sizeof(ch_mult) / sizeof(int);
        int block_in    = ch * ch_mult[len_mults - 1];

        mem_size += ch * in_channels * 3 * 3 * ggml_type_sizef(GGML_TYPE_F16);  // conv_in_w
        mem_size += ch * ggml_type_sizef(GGML_TYPE_F32);                        // conv_in_b

        mem_size += 2 * block_in * ggml_type_sizef(GGML_TYPE_F32);  // norm_out_w/b

        mem_size += z_channels * 2 * block_in * 3 * 3 * ggml_type_sizef(GGML_TYPE_F16);  // conv_out_w
        mem_size += z_channels * 2 * ggml_type_sizef(GGML_TYPE_F32);                     // conv_out_b

        mem_size += 6 * ggml_tensor_overhead();  // object overhead

        mem_size += mid.block_1.compute_params_mem_size(wtype);
        mem_size += mid.attn_1.compute_params_mem_size(wtype);
        mem_size += mid.block_2.compute_params_mem_size(wtype);

        for (int i = len_mults - 1; i >= 0; i--) {
            for (int j = 0; j < num_res_blocks + 1; j++) {
                mem_size += down_blocks[i][j].compute_params_mem_size(wtype);
            }
            if (i != 0) {
                mem_size += down_samples[i - 1].compute_params_mem_size(wtype);
            }
        }

        return static_cast<size_t>(mem_size);
    }

    void init_params(struct ggml_context* ctx, ggml_type wtype) {
        int len_mults = sizeof(ch_mult) / sizeof(int);
        int block_in  = ch * ch_mult[len_mults - 1];

        conv_in_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, in_channels, ch);
        conv_in_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ch);

        norm_out_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, block_in);
        norm_out_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, block_in);

        conv_out_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, block_in, z_channels * 2);
        conv_out_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, z_channels * 2);

        mid.block_1.init_params(ctx, wtype);
        mid.attn_1.init_params(ctx, wtype);
        mid.block_2.init_params(ctx, wtype);

        for (int i = 0; i < len_mults; i++) {
            for (int j = 0; j < num_res_blocks; j++) {
                down_blocks[i][j].init_params(ctx, wtype);
            }
            if (i != len_mults - 1) {
                down_samples[i].init_params(ctx, wtype);
            }
        }
    }

    void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        tensors[prefix + "norm_out.weight"] = norm_out_w;
        tensors[prefix + "norm_out.bias"]   = norm_out_b;
        tensors[prefix + "conv_in.weight"]  = conv_in_w;
        tensors[prefix + "conv_in.bias"]    = conv_in_b;
        tensors[prefix + "conv_out.weight"] = conv_out_w;
        tensors[prefix + "conv_out.bias"]   = conv_out_b;

        mid.block_1.map_by_name(tensors, prefix + "mid.block_1.");
        mid.attn_1.map_by_name(tensors, prefix + "mid.attn_1.");
        mid.block_2.map_by_name(tensors, prefix + "mid.block_2.");

        int len_mults = sizeof(ch_mult) / sizeof(int);
        for (int i = 0; i < len_mults; i++) {
            for (int j = 0; j < num_res_blocks; j++) {
                down_blocks[i][j].map_by_name(tensors, prefix + "down." + std::to_string(i) + ".block." + std::to_string(j) + ".");
            }
            if (i != len_mults - 1) {
                down_samples[i].map_by_name(tensors, prefix + "down." + std::to_string(i) + ".downsample.");
            }
        }
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [N, in_channels, h, w]

        // conv_in
        auto h        = ggml_conv_2d(ctx, conv_in_w, x, 1, 1, 1, 1, 1, 1);
        h             = ggml_add(ctx,
                                 h,
                                 ggml_repeat(ctx,
                                             ggml_reshape_4d(ctx, conv_in_b, 1, 1, conv_in_b->ne[0], 1),
                                             h));  // [N, ch, h, w]
        int len_mults = sizeof(ch_mult) / sizeof(int);
        for (int i = 0; i < len_mults; i++) {
            for (int j = 0; j < num_res_blocks; j++) {
                h = down_blocks[i][j].forward(ctx, h);
            }
            if (i != len_mults - 1) {
                h = down_samples[i].forward(ctx, h);
            }
        }

        h = mid.block_1.forward(ctx, h);
        h = mid.attn_1.forward(ctx, h);
        h = mid.block_2.forward(ctx, h);  // [N, block_in, h, w]

        // group norm 32
        h = ggml_group_norm_32(ctx, h);
        h = ggml_add(ctx,
                     ggml_mul(ctx, ggml_repeat(ctx, ggml_reshape_4d(ctx, norm_out_w, 1, 1, norm_out_w->ne[0], 1), h), h),
                     ggml_repeat(ctx, ggml_reshape_4d(ctx, norm_out_b, 1, 1, norm_out_b->ne[0], 1), h));

        // silu
        // silu
        h = ggml_silu_inplace(ctx, h);

        // conv_out
        h = ggml_conv_2d(ctx, conv_out_w, h, 1, 1, 1, 1, 1, 1);
        h = ggml_add(ctx,
                     h,
                     ggml_repeat(ctx,
                                 ggml_reshape_4d(ctx, conv_out_b, 1, 1, conv_out_b->ne[0], 1),
                                 h));  // [N, z_channels*2, h, w]

        return h;
    }
};

// ldm.modules.diffusionmodules.model.Decoder
struct Decoder {
    int embed_dim      = 4;
    int ch             = 128;
    int z_channels     = 4;
    int out_ch         = 3;
    int num_res_blocks = 2;
    int ch_mult[4]     = {1, 2, 4, 4};

    // block_in = ch *  ch_mult[-1], 512
    struct ggml_tensor* conv_in_w;  // [block_in, z_channels, 3, 3]
    struct ggml_tensor* conv_in_b;  // [block_in, ]

    struct
    {
        ResnetBlock block_1;
        AttnBlock attn_1;
        ResnetBlock block_2;
    } mid;

    ResnetBlock up_blocks[4][3];
    UpSample up_samples[3];

    struct ggml_tensor* norm_out_w;  // [ch *  ch_mult[0], ]
    struct ggml_tensor* norm_out_b;  // [ch *  ch_mult[0], ]

    struct ggml_tensor* conv_out_w;  // [out_ch, ch *  ch_mult[0], 3, 3]
    struct ggml_tensor* conv_out_b;  // [out_ch, ]

    Decoder() {
        int len_mults = sizeof(ch_mult) / sizeof(int);
        int block_in  = ch * ch_mult[len_mults - 1];

        mid.block_1.in_channels  = block_in;
        mid.block_1.out_channels = block_in;
        mid.attn_1.in_channels   = block_in;
        mid.block_2.in_channels  = block_in;
        mid.block_2.out_channels = block_in;

        for (int i = len_mults - 1; i >= 0; i--) {
            int mult      = ch_mult[i];
            int block_out = ch * mult;
            for (int j = 0; j < num_res_blocks + 1; j++) {
                up_blocks[i][j].in_channels  = block_in;
                up_blocks[i][j].out_channels = block_out;
                block_in                     = block_out;
            }
            if (i != 0) {
                up_samples[i - 1].channels     = block_in;
                up_samples[i - 1].out_channels = block_in;
            }
        }
    }

    size_t compute_params_mem_size(ggml_type wtype) {
        double mem_size = 0;
        int len_mults   = sizeof(ch_mult) / sizeof(int);
        int block_in    = ch * ch_mult[len_mults - 1];

        mem_size += block_in * z_channels * 3 * 3 * ggml_type_sizef(GGML_TYPE_F16);  // conv_in_w
        mem_size += block_in * ggml_type_sizef(GGML_TYPE_F32);                       // conv_in_b

        mem_size += 2 * (ch * ch_mult[0]) * ggml_type_sizef(GGML_TYPE_F32);  // norm_out_w/b

        mem_size += (ch * ch_mult[0]) * out_ch * 3 * 3 * ggml_type_sizef(GGML_TYPE_F16);  // conv_out_w
        mem_size += out_ch * ggml_type_sizef(GGML_TYPE_F32);                              // conv_out_b

        mem_size += 8 * ggml_tensor_overhead();  // object overhead

        mem_size += mid.block_1.compute_params_mem_size(wtype);
        mem_size += mid.attn_1.compute_params_mem_size(wtype);
        mem_size += mid.block_2.compute_params_mem_size(wtype);

        for (int i = len_mults - 1; i >= 0; i--) {
            for (int j = 0; j < num_res_blocks + 1; j++) {
                mem_size += up_blocks[i][j].compute_params_mem_size(wtype);
            }
            if (i != 0) {
                mem_size += up_samples[i - 1].compute_params_mem_size(wtype);
            }
        }

        return static_cast<size_t>(mem_size);
    }

    void init_params(struct ggml_context* ctx, ggml_type wtype) {
        int len_mults = sizeof(ch_mult) / sizeof(int);
        int block_in  = ch * ch_mult[len_mults - 1];

        norm_out_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ch * ch_mult[0]);
        norm_out_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ch * ch_mult[0]);

        conv_in_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, z_channels, block_in);
        conv_in_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, block_in);

        conv_out_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, ch * ch_mult[0], out_ch);
        conv_out_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_ch);

        mid.block_1.init_params(ctx, wtype);
        mid.attn_1.init_params(ctx, wtype);
        mid.block_2.init_params(ctx, wtype);

        for (int i = len_mults - 1; i >= 0; i--) {
            for (int j = 0; j < num_res_blocks + 1; j++) {
                up_blocks[i][j].init_params(ctx, wtype);
            }
            if (i != 0) {
                up_samples[i - 1].init_params(ctx, wtype);
            }
        }
    }

    void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        tensors[prefix + "norm_out.weight"] = norm_out_w;
        tensors[prefix + "norm_out.bias"]   = norm_out_b;
        tensors[prefix + "conv_in.weight"]  = conv_in_w;
        tensors[prefix + "conv_in.bias"]    = conv_in_b;
        tensors[prefix + "conv_out.weight"] = conv_out_w;
        tensors[prefix + "conv_out.bias"]   = conv_out_b;

        mid.block_1.map_by_name(tensors, prefix + "mid.block_1.");
        mid.attn_1.map_by_name(tensors, prefix + "mid.attn_1.");
        mid.block_2.map_by_name(tensors, prefix + "mid.block_2.");

        int len_mults = sizeof(ch_mult) / sizeof(int);
        for (int i = len_mults - 1; i >= 0; i--) {
            for (int j = 0; j < num_res_blocks + 1; j++) {
                up_blocks[i][j].map_by_name(tensors, prefix + "up." + std::to_string(i) + ".block." + std::to_string(j) + ".");
            }
            if (i != 0) {
                up_samples[i - 1].map_by_name(tensors, prefix + "up." + std::to_string(i) + ".upsample.");
            }
        }
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* z) {
        // z: [N, z_channels, h, w]

        // conv_in
        auto h = ggml_conv_2d(ctx, conv_in_w, z, 1, 1, 1, 1, 1, 1);
        h      = ggml_add(ctx,
                          h,
                          ggml_repeat(ctx,
                                      ggml_reshape_4d(ctx, conv_in_b, 1, 1, conv_in_b->ne[0], 1),
                                      h));  // [N, block_in, h, w]

        h = mid.block_1.forward(ctx, h);
        h = mid.attn_1.forward(ctx, h);
        h = mid.block_2.forward(ctx, h);  // [N, block_in, h, w]

        int len_mults = sizeof(ch_mult) / sizeof(int);
        for (int i = len_mults - 1; i >= 0; i--) {
            for (int j = 0; j < num_res_blocks + 1; j++) {
                h = up_blocks[i][j].forward(ctx, h);
            }
            if (i != 0) {
                h = up_samples[i - 1].forward(ctx, h);
            }
        }

        // group norm 32
        h = ggml_group_norm_32(ctx, h);
        h = ggml_add(ctx,
                     ggml_mul(ctx, ggml_repeat(ctx, ggml_reshape_4d(ctx, norm_out_w, 1, 1, norm_out_w->ne[0], 1), h), h),
                     ggml_repeat(ctx, ggml_reshape_4d(ctx, norm_out_b, 1, 1, norm_out_b->ne[0], 1), h));

        // silu
        // silu
        h = ggml_silu_inplace(ctx, h);

        // conv_out
        h = ggml_conv_2d(ctx, conv_out_w, h, 1, 1, 1, 1, 1, 1);
        h = ggml_add(ctx,
                     h,
                     ggml_repeat(ctx,
                                 ggml_reshape_4d(ctx, conv_out_b, 1, 1, conv_out_b->ne[0], 1),
                                 h));  // [N, out_ch, h, w]

        return h;
    }
};

// ldm.models.autoencoder.AutoencoderKL
struct AutoEncoderKL {
    bool decode_only = true;
    int embed_dim    = 4;
    struct
    {
        int z_channels     = 4;
        int resolution     = 256;
        int in_channels    = 3;
        int out_ch         = 3;
        int ch             = 128;
        int ch_mult[4]     = {1, 2, 4, 4};
        int num_res_blocks = 2;
    } dd_config;

    struct ggml_tensor* quant_conv_w;  // [2*embed_dim, 2*z_channels, 1, 1]
    struct ggml_tensor* quant_conv_b;  // [2*embed_dim, ]

    struct ggml_tensor* post_quant_conv_w;  // [z_channels, embed_dim, 1, 1]
    struct ggml_tensor* post_quant_conv_b;  // [z_channels, ]

    Encoder encoder;
    Decoder decoder;

    AutoEncoderKL(bool decode_only = false)
        : decode_only(decode_only) {
        assert(sizeof(dd_config.ch_mult) == sizeof(encoder.ch_mult));
        assert(sizeof(dd_config.ch_mult) == sizeof(decoder.ch_mult));

        encoder.embed_dim      = embed_dim;
        decoder.embed_dim      = embed_dim;
        encoder.ch             = dd_config.ch;
        decoder.ch             = dd_config.ch;
        encoder.z_channels     = dd_config.z_channels;
        decoder.z_channels     = dd_config.z_channels;
        encoder.in_channels    = dd_config.in_channels;
        decoder.out_ch         = dd_config.out_ch;
        encoder.num_res_blocks = dd_config.num_res_blocks;

        int len_mults = sizeof(dd_config.ch_mult) / sizeof(int);
        for (int i = 0; i < len_mults; i++) {
            encoder.ch_mult[i] = dd_config.ch_mult[i];
            decoder.ch_mult[i] = dd_config.ch_mult[i];
        }
    }

    size_t compute_params_mem_size(ggml_type wtype) {
        double mem_size = 0;

        if (!decode_only) {
            mem_size += 2 * embed_dim * 2 * dd_config.z_channels * 1 * 1 * ggml_type_sizef(GGML_TYPE_F16);  // quant_conv_w
            mem_size += 2 * embed_dim * ggml_type_sizef(GGML_TYPE_F32);                                     // quant_conv_b
            mem_size += encoder.compute_params_mem_size(wtype);
        }

        mem_size += dd_config.z_channels * embed_dim * 1 * 1 * ggml_type_sizef(GGML_TYPE_F16);  // post_quant_conv_w
        mem_size += dd_config.z_channels * ggml_type_sizef(GGML_TYPE_F32);                      // post_quant_conv_b

        mem_size += decoder.compute_params_mem_size(wtype);
        return static_cast<size_t>(mem_size);
    }

    void init_params(struct ggml_context* ctx, ggml_type wtype) {
        if (!decode_only) {
            quant_conv_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 1, 1, 2 * dd_config.z_channels, 2 * embed_dim);
            quant_conv_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2 * embed_dim);
            encoder.init_params(ctx, wtype);
        }

        post_quant_conv_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 1, 1, embed_dim, dd_config.z_channels);
        post_quant_conv_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dd_config.z_channels);
        decoder.init_params(ctx, wtype);
    }

    void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        if (!decode_only) {
            tensors[prefix + "quant_conv.weight"] = quant_conv_w;
            tensors[prefix + "quant_conv.bias"]   = quant_conv_b;
            encoder.map_by_name(tensors, prefix + "encoder.");
        }

        tensors[prefix + "post_quant_conv.weight"] = post_quant_conv_w;
        tensors[prefix + "post_quant_conv.bias"]   = post_quant_conv_b;
        decoder.map_by_name(tensors, prefix + "decoder.");
    }

    struct ggml_tensor* decode(struct ggml_context* ctx, struct ggml_tensor* z) {
        // z: [N, z_channels, h, w]

        // post_quant_conv
        auto h = ggml_conv_2d(ctx, post_quant_conv_w, z, 1, 1, 0, 0, 1, 1);
        h      = ggml_add(ctx,
                          h,
                          ggml_repeat(ctx,
                                      ggml_reshape_4d(ctx, post_quant_conv_b, 1, 1, post_quant_conv_b->ne[0], 1),
                                      h));  // [N, z_channels, h, w]
        h      = decoder.forward(ctx, h);
        return h;
    }

    struct ggml_tensor* encode(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [N, in_channels, h, w]
        auto h = encoder.forward(ctx, x);  // [N, 2*z_channels, h/8, w/8]
        // quant_conv
        h = ggml_conv_2d(ctx, quant_conv_w, h, 1, 1, 0, 0, 1, 1);
        h = ggml_add(ctx,
                     h,
                     ggml_repeat(ctx,
                                 ggml_reshape_4d(ctx, quant_conv_b, 1, 1, quant_conv_b->ne[0], 1),
                                 h));  // [N, 2*embed_dim, h/8, w/8]
        return h;
    }
};

/*================================================= CompVisDenoiser ==================================================*/

// Ref: https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/external.py

struct SigmaSchedule {
    float alphas_cumprod[TIMESTEPS];
    float sigmas[TIMESTEPS];
    float log_sigmas[TIMESTEPS];

    virtual std::vector<float> get_sigmas(uint32_t n) = 0;

    float sigma_to_t(float sigma) {
        float log_sigma = std::log(sigma);
        std::vector<float> dists;
        dists.reserve(TIMESTEPS);
        for (float log_sigma_val : log_sigmas) {
            dists.push_back(log_sigma - log_sigma_val);
        }

        int low_idx = 0;
        for (size_t i = 0; i < TIMESTEPS; i++) {
            if (dists[i] >= 0) {
                low_idx++;
            }
        }
        low_idx      = std::min(std::max(low_idx - 1, 0), TIMESTEPS - 2);
        int high_idx = low_idx + 1;

        float low  = log_sigmas[low_idx];
        float high = log_sigmas[high_idx];
        float w    = (low - log_sigma) / (low - high);
        w          = std::max(0.f, std::min(1.f, w));
        float t    = (1.0f - w) * low_idx + w * high_idx;

        return t;
    }

    float t_to_sigma(float t) {
        int low_idx     = static_cast<int>(std::floor(t));
        int high_idx    = static_cast<int>(std::ceil(t));
        float w         = t - static_cast<float>(low_idx);
        float log_sigma = (1.0f - w) * log_sigmas[low_idx] + w * log_sigmas[high_idx];
        return std::exp(log_sigma);
    }
};

struct DiscreteSchedule : SigmaSchedule {
    std::vector<float> get_sigmas(uint32_t n) {
        std::vector<float> result;

        int t_max = TIMESTEPS - 1;

        if (n == 0) {
            return result;
        } else if (n == 1) {
            result.push_back(t_to_sigma((float)t_max));
            result.push_back(0);
            return result;
        }

        float step = static_cast<float>(t_max) / static_cast<float>(n - 1);
        for (uint32_t i = 0; i < n; ++i) {
            float t = t_max - step * i;
            result.push_back(t_to_sigma(t));
        }
        result.push_back(0);
        return result;
    }
};

struct KarrasSchedule : SigmaSchedule {
    std::vector<float> get_sigmas(uint32_t n) {
        // These *COULD* be function arguments here,
        // but does anybody ever bother to touch them?
        float sigma_min = 0.1f;
        float sigma_max = 10.f;
        float rho       = 7.f;

        std::vector<float> result(n + 1);

        float min_inv_rho = pow(sigma_min, (1.f / rho));
        float max_inv_rho = pow(sigma_max, (1.f / rho));
        for (uint32_t i = 0; i < n; i++) {
            // Eq. (5) from Karras et al 2022
            result[i] = pow(max_inv_rho + (float)i / ((float)n - 1.f) * (min_inv_rho - max_inv_rho), rho);
        }
        result[n] = 0.;
        return result;
    }
};

struct Denoiser {
    std::shared_ptr<SigmaSchedule> schedule              = std::make_shared<DiscreteSchedule>();
    virtual std::vector<float> get_scalings(float sigma) = 0;
};

struct CompVisDenoiser : public Denoiser {
    float sigma_data = 1.0f;

    std::vector<float> get_scalings(float sigma) {
        float c_out = -sigma;
        float c_in  = 1.0f / std::sqrt(sigma * sigma + sigma_data * sigma_data);
        return {c_out, c_in};
    }
};

struct CompVisVDenoiser : public Denoiser {
    float sigma_data = 1.0f;

    std::vector<float> get_scalings(float sigma) {
        float c_skip = sigma_data * sigma_data / (sigma * sigma + sigma_data * sigma_data);
        float c_out  = -sigma * sigma_data / std::sqrt(sigma * sigma + sigma_data * sigma_data);
        float c_in   = 1.0f / std::sqrt(sigma * sigma + sigma_data * sigma_data);
        return {c_skip, c_out, c_in};
    }
};

/*=============================================== StableDiffusionGGML ================================================*/

class StableDiffusionGGML {
public:
    ggml_context* clip_params_ctx = NULL;
    ggml_context* unet_params_ctx = NULL;
    ggml_context* vae_params_ctx  = NULL;

    bool dynamic                 = true;
    bool vae_decode_only         = false;
    bool free_params_immediately = false;

    std::shared_ptr<RNG> rng    = std::make_shared<STDDefaultRNG>();
    int32_t ftype               = 1;
    int n_threads               = -1;
    float scale_factor          = 0.18215f;
    size_t max_mem_size         = 0;
    size_t curr_params_mem_size = 0;
    size_t max_params_mem_size  = 0;
    size_t max_rt_mem_size      = 0;

    FrozenCLIPEmbedderWithCustomWords cond_stage_model;
    UNetModel diffusion_model;
    AutoEncoderKL first_stage_model;

    std::map<std::string, struct ggml_tensor*> tensors;

    std::string lora_model_dir;
    // lora_name => lora_tensor_name => tensor
    std::map<std::string, std::map<std::string, struct ggml_tensor*>> lora_tensors;
    // lora_name => lora_params_ctx
    std::map<std::string, ggml_context*> lora_params_ctxs;
    // lora_name => multiplier
    std::unordered_map<std::string, float> curr_lora_state;

    std::shared_ptr<Denoiser> denoiser = std::make_shared<CompVisDenoiser>();

    StableDiffusionGGML() = default;

    StableDiffusionGGML(int n_threads,
                        bool vae_decode_only,
                        bool free_params_immediately,
                        std::string lora_model_dir,
                        RNGType rng_type)
        : n_threads(n_threads),
          vae_decode_only(vae_decode_only),
          free_params_immediately(free_params_immediately),
          lora_model_dir(lora_model_dir) {
        first_stage_model.decode_only = vae_decode_only;
        if (rng_type == STD_DEFAULT_RNG) {
            rng = std::make_shared<STDDefaultRNG>();
        } else if (rng_type == CUDA_RNG) {
            rng = std::make_shared<PhiloxRNG>();
        }
        if (lora_model_dir.size() > 0) {
            if (lora_model_dir[lora_model_dir.size() - 1] != '/' && lora_model_dir[lora_model_dir.size() - 1] != '\\') {
                this->lora_model_dir = lora_model_dir + "/";
            }
        }
    }

    ~StableDiffusionGGML() {
        if (clip_params_ctx != NULL) {
            ggml_free(clip_params_ctx);
            clip_params_ctx = NULL;
        }
        if (unet_params_ctx != NULL) {
            ggml_free(unet_params_ctx);
            unet_params_ctx = NULL;
        }
        if (vae_params_ctx != NULL) {
            ggml_free(vae_params_ctx);
            vae_params_ctx = NULL;
        }
        for (auto& kv : lora_params_ctxs) {
            ggml_free(kv.second);
        }
        lora_params_ctxs.clear();

        tensors.clear();
        lora_tensors.clear();
    }

    bool load_from_file(const std::string& file_path, Schedule schedule) {
        LOG_INFO("loading model from '%s'", file_path.c_str());

        std::ifstream file(file_path, std::ios::binary);
        if (!file.is_open()) {
            LOG_ERROR("failed to open '%s'", file_path.c_str());
            return false;
        }

        LOG_DEBUG("verifying magic");
        // verify magic
        {
            uint32_t magic;
            file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
            if (magic != GGML_FILE_MAGIC) {
                LOG_ERROR("invalid model file '%s' (bad magic)", file_path.c_str());
                return false;
            }
        }

        LOG_DEBUG("loading hparams");
        // load hparams
        file.read(reinterpret_cast<char*>(&ftype), sizeof(ftype));

        int model_type = (ftype >> 16) & 0xFFFF;
        if (model_type >= MODEL_TYPE_COUNT) {
            LOG_ERROR("invalid model file '%s' (bad model type value %d)", file_path.c_str(), ftype);
            return false;
        }
        LOG_INFO("model type: %s", model_type_to_str[model_type]);

        if (model_type == SD2) {
            cond_stage_model = FrozenCLIPEmbedderWithCustomWords((ModelType)model_type);
            diffusion_model  = UNetModel((ModelType)model_type);
        }

        ggml_type wtype = ggml_ftype_to_ggml_type((ggml_ftype)(ftype & 0xFFFF));
        LOG_INFO("ftype: %s", ggml_type_name(wtype));
        if (wtype == GGML_TYPE_COUNT) {
            LOG_ERROR("invalid model file '%s' (bad ftype value %d)", file_path.c_str(), ftype);
            return false;
        }

        LOG_DEBUG("loading vocab");
        // load vocab
        {
            int32_t n_vocab = 0;
            file.read(reinterpret_cast<char*>(&n_vocab), sizeof(n_vocab));

            if (n_vocab != cond_stage_model.text_model.vocab_size) {
                LOG_ERROR("invalid model file '%s' (bad vocab size %d != %d)",
                          file_path.c_str(), n_vocab, cond_stage_model.text_model.vocab_size);
                return false;
            }

            std::string word;
            std::vector<char> buf(128);

            for (int i = 0; i < n_vocab; i++) {
                uint32_t len;
                file.read((char*)&len, sizeof(len));

                buf.resize(len);
                file.read((char*)buf.data(), len);
                word.assign(buf.data(), len);

                cond_stage_model.tokenizer.add_token(word, i);
            }
        }

        // create the ggml context for network params
        LOG_DEBUG("ggml tensor size = %d bytes", (int)sizeof(ggml_tensor));
        {
            // cond_stage_model(FrozenCLIPEmbedder)
            double ctx_size = 1 * 1024 * 1024;  // 1 MB, for padding
            ctx_size += cond_stage_model.text_model.compute_params_mem_size(wtype);
            LOG_DEBUG("clip params ctx size = % 6.2f MB", ctx_size / (1024.0 * 1024.0));

            struct ggml_init_params params;
            params.mem_size   = static_cast<size_t>(ctx_size);
            params.mem_buffer = NULL;
            params.no_alloc   = false;
            params.dynamic    = false;

            clip_params_ctx = ggml_init(params);
            if (!clip_params_ctx) {
                LOG_ERROR("ggml_init() failed");
                return false;
            }
        }

        {
            // diffusion_model(UNetModel)
            double ctx_size = 1 * 1024 * 1024;  // 1 MB, for padding
            ctx_size += diffusion_model.compute_params_mem_size(wtype);
            LOG_DEBUG("unet params ctx size = % 6.2f MB", ctx_size / (1024.0 * 1024.0));

            struct ggml_init_params params;
            params.mem_size   = static_cast<size_t>(ctx_size);
            params.mem_buffer = NULL;
            params.no_alloc   = false;
            params.dynamic    = false;

            unet_params_ctx = ggml_init(params);
            if (!unet_params_ctx) {
                LOG_ERROR("ggml_init() failed");
                ggml_free(clip_params_ctx);
                clip_params_ctx = NULL;
                return false;
            }
        }

        {
            // first_stage_model(AutoEncoderKL)
            double ctx_size = 1 * 1024 * 1024;  // 1 MB, for padding
            ctx_size += first_stage_model.compute_params_mem_size(wtype);
            LOG_DEBUG("vae params ctx size = % 6.2f MB", ctx_size / (1024.0 * 1024.0));

            struct ggml_init_params params;
            params.mem_size   = static_cast<size_t>(ctx_size);
            params.mem_buffer = NULL;
            params.no_alloc   = false;
            params.dynamic    = false;

            vae_params_ctx = ggml_init(params);
            if (!vae_params_ctx) {
                LOG_ERROR("ggml_init() failed");
                ggml_free(clip_params_ctx);
                clip_params_ctx = NULL;
                ggml_free(unet_params_ctx);
                unet_params_ctx = NULL;
                return false;
            }
        }

        LOG_DEBUG("preparing memory for the weights");
        // prepare memory for the weights
        {
            // cond_stage_model(FrozenCLIPEmbedder)
            cond_stage_model.text_model.init_params(clip_params_ctx, wtype);
            cond_stage_model.text_model.map_by_name(tensors, "cond_stage_model.transformer.text_model.");

            // diffusion_model(UNetModel)
            diffusion_model.init_params(unet_params_ctx, wtype);
            diffusion_model.map_by_name(tensors, "model.diffusion_model.");

            // firest_stage_model(AutoEncoderKL)
            first_stage_model.init_params(vae_params_ctx, wtype);
            first_stage_model.map_by_name(tensors, "first_stage_model.");
        }

        LOG_DEBUG("loading weights");
        std::set<std::string> tensor_names_in_file;
        int64_t t0 = ggml_time_ms();
        // load weights
        float alphas_cumprod[TIMESTEPS];
        {
            int n_tensors     = 0;
            size_t total_size = 0;

            while (true) {
                int32_t n_dims;
                int32_t length;
                int32_t ttype;

                file.read(reinterpret_cast<char*>(&n_dims), sizeof(n_dims));
                file.read(reinterpret_cast<char*>(&length), sizeof(length));
                file.read(reinterpret_cast<char*>(&ttype), sizeof(ttype));

                if (file.eof()) {
                    break;
                }

                int32_t nelements = 1;
                int32_t ne[4]     = {1, 1, 1, 1};
                for (int i = 0; i < n_dims; ++i) {
                    file.read(reinterpret_cast<char*>(&ne[i]), sizeof(ne[i]));
                    nelements *= ne[i];
                }

                const size_t num_bytes = nelements / ggml_blck_size(ggml_type(ttype)) * ggml_type_size(ggml_type(ttype));

                std::string name(length, 0);
                file.read(&name[0], length);

                tensor_names_in_file.insert(std::string(name.data()));

                if (std::string(name.data()) == "alphas_cumprod") {
                    file.read(reinterpret_cast<char*>(alphas_cumprod), nelements * ggml_type_size((ggml_type)ttype));
                    continue;
                }

                struct ggml_tensor* tensor;
                if (tensors.find(name.data()) != tensors.end()) {
                    tensor = tensors[name.data()];
                } else {
                    if (name.find("quant") == std::string::npos && name.find("first_stage_model.encoder.") == std::string::npos) {
                        LOG_WARN("unknown tensor '%s' in model file", name.data());
                    } else {
                        if (!vae_decode_only) {
                            LOG_WARN("unknown tensor '%s' in model file", name.data());
                            return false;
                        }
                    }
                    file.ignore(num_bytes);
                    continue;
                }

                if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1] || tensor->ne[2] != ne[2] || tensor->ne[3] != ne[3]) {
                    LOG_ERROR(
                        "tensor '%s' has wrong shape in model file: "
                        "got [%d, %d, %d, %d], expected [%d, %d, %d, %d]",
                        name.data(),
                        ne[0], ne[1], ne[2], ne[3],
                        (int)tensor->ne[0], (int)tensor->ne[1], (int)tensor->ne[2], (int)tensor->ne[3]);
                    return false;
                }

                if (ggml_nelements(tensor) != nelements) {
                    LOG_ERROR(
                        "tensor '%s' has wrong number of elements in model file: "
                        "got %u, expert %zu",
                        name.data(), nelements, ggml_nelements(tensor));
                    return false;
                }

                if (tensor->type != ttype) {
                    LOG_ERROR("tensor '%s' has wrong type in model file: got %s, expect %s",
                              name.data(), ggml_type_name(ggml_type(ttype)), ggml_type_name(tensor->type));
                    return false;
                }

                file.read(reinterpret_cast<char*>(tensor->data), num_bytes);

                total_size += ggml_nbytes(tensor);
            }
            bool some_tensor_not_init = false;
            for (auto pair : tensors) {
                if (pair.first.find("cond_stage_model.transformer.text_model.encoder.layers.23") != std::string::npos) {
                    continue;
                }
                if (tensor_names_in_file.find(pair.first) == tensor_names_in_file.end()) {
                    LOG_ERROR("tensor '%s' not in model file", pair.first.c_str());
                    some_tensor_not_init = true;
                }
            }
            if (tensor_names_in_file.find("alphas_cumprod") == tensor_names_in_file.end()) {
                LOG_ERROR("tensor alphas_cumprod not in model file");
                some_tensor_not_init = true;
            }
            if (some_tensor_not_init) {
                file.close();
                return false;
            }
            LOG_DEBUG("model size = %.2fMB", total_size / 1024.0 / 1024.0);
        }
        max_params_mem_size  = ggml_used_mem(clip_params_ctx) + ggml_used_mem(unet_params_ctx) + ggml_used_mem(vae_params_ctx);
        max_mem_size         = max_params_mem_size;
        curr_params_mem_size = max_params_mem_size;
        LOG_INFO("total params size = %.2fMB (clip %.2fMB, unet %.2fMB, vae %.2fMB)",
                 max_params_mem_size / 1024.0 / 1024.0,
                 ggml_used_mem(clip_params_ctx) / 1024.0 / 1024.0,
                 ggml_used_mem(unet_params_ctx) / 1024.0 / 1024.0,
                 ggml_used_mem(vae_params_ctx) / 1024.0 / 1024.0);
        int64_t t1 = ggml_time_ms();
        LOG_INFO("loading model from '%s' completed, taking %.2fs", file_path.c_str(), (t1 - t0) * 1.0f / 1000);
        file.close();

        // check is_using_v_parameterization_for_sd2
        bool is_using_v_parameterization = false;
        if (model_type == SD2) {
            struct ggml_init_params params;
            params.mem_size          = static_cast<size_t>(10 * 1024) * 1024;  // 10M
            params.mem_buffer        = NULL;
            params.no_alloc          = false;
            params.dynamic           = false;
            struct ggml_context* ctx = ggml_init(params);
            if (!ctx) {
                LOG_ERROR("ggml_init() failed");
                return false;
            }
            if (is_using_v_parameterization_for_sd2(ctx)) {
                is_using_v_parameterization = true;
            }
        }

        if (is_using_v_parameterization) {
            denoiser = std::make_shared<CompVisVDenoiser>();
            LOG_INFO("running in v-prediction mode");
        } else {
            LOG_INFO("running in eps-prediction mode");
        }

        if (schedule != DEFAULT) {
            switch (schedule) {
                case DISCRETE:
                    LOG_INFO("running with discrete schedule");
                    denoiser->schedule = std::make_shared<DiscreteSchedule>();
                    break;
                case KARRAS:
                    LOG_INFO("running with Karras schedule");
                    denoiser->schedule = std::make_shared<KarrasSchedule>();
                    break;
                case DEFAULT:
                    // Don't touch anything.
                    break;
                default:
                    LOG_ERROR("Unknown schedule %i", schedule);
                    abort();
            }
        }

        for (int i = 0; i < TIMESTEPS; i++) {
            denoiser->schedule->alphas_cumprod[i] = alphas_cumprod[i];
            denoiser->schedule->sigmas[i]         = std::sqrt((1 - denoiser->schedule->alphas_cumprod[i]) / denoiser->schedule->alphas_cumprod[i]);
            denoiser->schedule->log_sigmas[i]     = std::log(denoiser->schedule->sigmas[i]);
        }

        return true;
    }

    bool is_using_v_parameterization_for_sd2(ggml_context* res_ctx) {
        struct ggml_tensor* x_t = ggml_new_tensor_4d(res_ctx, GGML_TYPE_F32, 8, 8, 4, 1);
        ggml_set_f32(x_t, 0.5);
        struct ggml_tensor* c = ggml_new_tensor_4d(res_ctx, GGML_TYPE_F32, 1024, 2, 1, 1);
        ggml_set_f32(c, 0.5);

        struct ggml_cplan cplan;

        size_t ctx_size = 10 * 1024 * 1024;  // 10MB
        // calculate the amount of memory required
        {
            struct ggml_init_params params;
            params.mem_size   = ctx_size;
            params.mem_buffer = NULL;
            params.no_alloc   = true;
            params.dynamic    = dynamic;

            struct ggml_context* ctx = ggml_init(params);
            if (!ctx) {
                LOG_ERROR("ggml_init() failed");
                return false;
            }

            ggml_set_dynamic(ctx, false);
            struct ggml_tensor* timesteps = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);                               // [N, ]
            struct ggml_tensor* t_emb     = new_timestep_embedding(ctx, timesteps, diffusion_model.model_channels);  // [N, model_channels]
            ggml_set_dynamic(ctx, params.dynamic);

            struct ggml_tensor* out = diffusion_model.forward(ctx, x_t, NULL, c, t_emb);
            ctx_size += ggml_used_mem(ctx) + ggml_used_mem_of_data(ctx);

            struct ggml_cgraph* diffusion_graph = ggml_build_forward_ctx(ctx, out);
            cplan                               = ggml_graph_plan(diffusion_graph, n_threads);

            ctx_size += cplan.work_size;
            LOG_DEBUG("diffusion context need %.2fMB static memory, with work_size needing %.2fMB",
                      ctx_size * 1.0f / 1024 / 1024,
                      cplan.work_size * 1.0f / 1024 / 1024);

            ggml_free(ctx);
        }

        struct ggml_init_params params;
        params.mem_size   = ctx_size;
        params.mem_buffer = NULL;
        params.no_alloc   = false;
        params.dynamic    = dynamic;

        struct ggml_context* ctx = ggml_init(params);
        if (!ctx) {
            LOG_ERROR("ggml_init() failed");
            return false;
        }

        ggml_set_dynamic(ctx, false);
        struct ggml_tensor* timesteps = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);                               // [N, ]
        struct ggml_tensor* t_emb     = new_timestep_embedding(ctx, timesteps, diffusion_model.model_channels);  // [N, model_channels]
        ggml_set_dynamic(ctx, params.dynamic);
        ggml_set_f32(timesteps, 999);
        set_timestep_embedding(timesteps, t_emb, diffusion_model.model_channels);

        struct ggml_tensor* out = diffusion_model.forward(ctx, x_t, NULL, c, t_emb);
        ggml_hold_dynamic_tensor(out);

        struct ggml_cgraph* diffusion_graph = ggml_build_forward_ctx(ctx, out);
        cplan                               = ggml_graph_plan(diffusion_graph, n_threads);

        ggml_set_dynamic(ctx, false);
        struct ggml_tensor* buf = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, cplan.work_size);
        ggml_set_dynamic(ctx, params.dynamic);

        cplan.work_data = (uint8_t*)buf->data;

        int64_t t0 = ggml_time_ms();
        ggml_graph_compute(diffusion_graph, &cplan);

        double result = 0.f;

        {
            float* vec_x   = (float*)x_t->data;
            float* vec_out = (float*)out->data;

            int64_t n = ggml_nelements(out);

            for (int i = 0; i < n; i++) {
                result += ((double)vec_out[i] - (double)vec_x[i]);
            }
            result /= n;
        }

#ifdef GGML_PERF
        ggml_graph_print(&diffusion_graph);
#endif
        int64_t t1 = ggml_time_ms();
        LOG_INFO("check is_using_v_parameterization_for_sd2 completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);
        LOG_DEBUG("diffusion graph use %.2fMB runtime memory: static %.2fMB, dynamic %.2fMB",
                  (ctx_size + ggml_curr_max_dynamic_size()) * 1.0f / 1024 / 1024,
                  ctx_size * 1.0f / 1024 / 1024,
                  ggml_curr_max_dynamic_size() * 1.0f / 1024 / 1024);
        LOG_DEBUG("%zu bytes of dynamic memory has not been released yet", ggml_dynamic_size());

        return result < -1;
    }

    bool load_lora_from_file(const std::string& lora_name) {
        if (lora_tensors.find(lora_name) != lora_tensors.end()) {
            return true;
        }
        std::string file_path = lora_model_dir + lora_name + "-ggml-lora.bin";
        LOG_INFO("loading lora '%s' from '%s'", lora_name.c_str(), file_path.c_str());

        std::ifstream file(file_path, std::ios::binary);
        if (!file.is_open()) {
            LOG_ERROR("failed to open '%s'", file_path.c_str());
            return false;
        }

        // get file size
        file.seekg(0, file.end);
        int file_size = (int)file.tellg();
        file.seekg(0, file.beg);

        LOG_DEBUG("'%s': %.2fMB", file_path.c_str(), file_size * 1.f / 1024 / 1024);

        LOG_DEBUG("verifying magic");
        // verify magic
        {
            uint32_t magic;
            file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
            if (magic != GGML_FILE_MAGIC) {
                LOG_ERROR("invalid model file '%s' (bad magic)", file_path.c_str());
                return false;
            }
        }

        LOG_DEBUG("loading hparams");
        // load hparams
        file.read(reinterpret_cast<char*>(&ftype), sizeof(ftype));

        int model_type = (ftype >> 16) & 0xFFFF;
        if (model_type >= MODEL_TYPE_COUNT) {
            LOG_ERROR("invalid model file '%s' (bad model type value %d)", file_path.c_str(), ftype);
            return false;
        }
        LOG_INFO("lora model type: %s", model_type_to_str[model_type]);

        ggml_type wtype = ggml_ftype_to_ggml_type((ggml_ftype)(ftype & 0xFFFF));
        LOG_INFO("ftype: %s", ggml_type_name(wtype));
        if (wtype == GGML_TYPE_COUNT) {
            LOG_ERROR("invalid model file '%s' (bad ftype value %d)", file_path.c_str(), ftype);
            return false;
        }

        // create the ggml context for network params
        struct ggml_init_params params;
        size_t ctx_size = 10 * 1024 * 1024;  // 10 MB, for padding
        ctx_size += file_size;
        params.mem_size   = ctx_size;
        params.mem_buffer = NULL;
        params.no_alloc   = false;
        params.dynamic    = false;
        LOG_DEBUG("lora '%s' params ctx size = % 6.2f MB", lora_name.c_str(), ctx_size / (1024.0 * 1024.0));
        ggml_context* lora_params_ctx = ggml_init(params);
        if (!lora_params_ctx) {
            LOG_ERROR("ggml_init() failed");
            return false;
        }
        lora_params_ctxs[lora_name] = lora_params_ctx;

        std::map<std::string, struct ggml_tensor*> lora_tensor_map;
        int64_t t0 = ggml_time_ms();
        // load weights
        {
            int n_tensors     = 0;
            size_t total_size = 0;

            while (true) {
                int32_t n_dims;
                int32_t length;
                int32_t ttype;

                file.read(reinterpret_cast<char*>(&n_dims), sizeof(n_dims));
                file.read(reinterpret_cast<char*>(&length), sizeof(length));
                file.read(reinterpret_cast<char*>(&ttype), sizeof(ttype));

                if (file.eof()) {
                    break;
                }

                int32_t nelements = 1;
                int32_t ne[4]     = {1, 1, 1, 1};
                for (int i = 0; i < n_dims; ++i) {
                    file.read(reinterpret_cast<char*>(&ne[i]), sizeof(ne[i]));
                    nelements *= ne[i];
                }

                const size_t num_bytes = nelements / ggml_blck_size(ggml_type(ttype)) * ggml_type_size(ggml_type(ttype));

                std::string name_buf(length, 0);
                file.read(&name_buf[0], length);
                std::string name = std::string(name_buf.data());

                // LOG_DEBUG("load lora tensor %s", name.c_str());

                int64_t ne64[4]            = {ne[0], ne[1], ne[2], ne[3]};
                struct ggml_tensor* tensor = ggml_new_tensor(lora_params_ctx, (ggml_type)ttype, n_dims, ne64);
                file.read(reinterpret_cast<char*>(tensor->data), num_bytes);

                lora_tensor_map[name] = tensor;

                total_size += ggml_nbytes(tensor);
            }
        }
        lora_tensors[lora_name] = lora_tensor_map;
        int64_t t1              = ggml_time_ms();
        LOG_INFO("lora '%s' params size = %.2fMB",
                 lora_name.c_str(),
                 ggml_used_mem(lora_params_ctx) / 1024.0 / 1024.0);
        LOG_INFO("loading lora from '%s' completed, taking %.2fs", file_path.c_str(), (t1 - t0) * 1.0f / 1000);
        file.close();
        return true;
    }

    void remove_lora_params(const std::string& lora_name) {
        if (lora_params_ctxs.find(lora_name) == lora_params_ctxs.end()) {
            return;
        }
        ggml_free(lora_params_ctxs[lora_name]);
        lora_params_ctxs.erase(lora_name);
        lora_tensors.erase(lora_name);
    }

    void apply_lora(const std::string& lora_name, float multiplier) {
        int64_t t0 = ggml_time_ms();
        if (!load_lora_from_file(lora_name)) {
            std::string file_path = lora_model_dir + lora_name + "-ggml-lora.bin";
            LOG_WARN("apply lora '%s' failed", lora_name.c_str());
            return;
        }

        size_t ctx_size  = 500 * 1024 * 1024;  // 500MB
        void* mem_buffer = malloc(ctx_size);
        if (!mem_buffer) {
            if (free_params_immediately) {
                remove_lora_params(lora_name);
            }
            LOG_ERROR("malloc() failed");
            return;
        }

        std::map<std::string, struct ggml_tensor*>& lora_tensor_map = lora_tensors[lora_name];
        std::set<std::string> applied_lora_tensors;
        for (auto& kv : tensors) {
            const std::string name = kv.first;
            ggml_tensor* weight    = kv.second;
            std::string ending     = ".weight";
            if (!ends_with(name, ending)) {
                continue;
            }

            // find corresponding lora tensors
            std::string network_name = name.substr(0, name.size() - ending.size());  // remove .weight
            replace_all_chars(network_name, '.', '_');
            std::string lora_up_name   = network_name + ".lora_up.weight";
            std::string lora_down_name = network_name + ".lora_down.weight";
            std::string alpha_name     = network_name + ".alpha";
            std::string scale_name     = network_name + ".scale";

            ggml_tensor* lora_up   = NULL;
            ggml_tensor* lora_down = NULL;

            float scale = 1.0f;

            if (lora_tensor_map.find(lora_up_name) != lora_tensor_map.end()) {
                lora_up = lora_tensor_map[lora_up_name];
            }

            if (lora_tensor_map.find(lora_down_name) != lora_tensor_map.end()) {
                lora_down = lora_tensor_map[lora_down_name];
            }

            if (lora_up == NULL || lora_down == NULL) {
                continue;
            }

            // LOG_DEBUG("apply lora tensor %s [%ld %ld %ld %ld]", network_name.c_str(), weight->ne[0], weight->ne[1], weight->ne[2], weight->ne[3]);

            applied_lora_tensors.insert(lora_up_name);
            applied_lora_tensors.insert(lora_down_name);
            applied_lora_tensors.insert(alpha_name);
            applied_lora_tensors.insert(scale_name);

            // calc_scale
            int64_t dim = lora_down->ne[lora_down->n_dims - 1];
            if (lora_tensor_map.find(scale_name) != lora_tensor_map.end()) {
                ggml_tensor* t = lora_tensor_map[scale_name];
                scale          = ggml_get_f32_1d(t, 0);
            } else if (lora_tensor_map.find(alpha_name) != lora_tensor_map.end()) {
                ggml_tensor* t = lora_tensor_map[alpha_name];
                scale          = ggml_get_f32_1d(t, 0) / dim;
            }

            // LOG_DEBUG("scale: %f %ld", scale, dim);

            scale = scale * multiplier;

            // apply
            {
                struct ggml_init_params params;
                params.mem_size   = ctx_size;
                params.mem_buffer = mem_buffer;
                params.no_alloc   = false;
                params.dynamic    = false;

                struct ggml_context* ctx = ggml_init(params);
                if (!ctx) {
                    LOG_ERROR("ggml_init() failed");
                    free(mem_buffer);
                    if (free_params_immediately) {
                        remove_lora_params(lora_name);
                    }
                    return;
                }

                ggml_tensor* scale_factor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
                ggml_set_f32_1d(scale_factor, 0, scale);
                int64_t lora_up_size_0   = lora_up->ne[lora_up->n_dims - 1];
                lora_up                  = ggml_reshape_2d(ctx, lora_up, ggml_nelements(lora_up) / lora_up_size_0, lora_up_size_0);
                int64_t lora_down_size_0 = lora_down->ne[lora_down->n_dims - 1];
                lora_down                = ggml_reshape_2d(ctx, lora_down, ggml_nelements(lora_down) / lora_down_size_0, lora_down_size_0);

                lora_down = ggml_cont(ctx, ggml_transpose(ctx, lora_down));

                if (lora_down->type != GGML_TYPE_F32) {
                    ggml_tensor* lora_down_f32 = ggml_new_tensor(ctx, GGML_TYPE_F32, lora_down->n_dims, lora_down->ne);
                    lora_down                  = ggml_cpy_inplace(ctx, lora_down, lora_down_f32);
                }

                ggml_tensor* updown = ggml_mul_mat(ctx, lora_up, lora_down);
                updown              = ggml_cont(ctx, ggml_transpose(ctx, updown));
                updown              = ggml_reshape(ctx, updown, weight);

                GGML_ASSERT(ggml_nelements(updown) == ggml_nelements(weight));

                updown = ggml_scale_inplace(ctx, updown, scale_factor);
                ggml_tensor* final_weight;
                final_weight = ggml_add_inplace(ctx, weight, updown);
                final_weight = ggml_cpy_inplace(ctx, final_weight, weight);

                struct ggml_cgraph* graph = ggml_build_forward_ctx(ctx, final_weight);

                ggml_graph_compute_with_ctx(ctx, graph, n_threads);

                // LOG_INFO("network_name '%s' ggml_used_mem size = %.2fMB",
                //           network_name.c_str(),
                //           ggml_used_mem(ctx) / 1024.0 / 1024.0);

                ggml_free(ctx);
            }
        }
        free(mem_buffer);

        for (auto& kv : lora_tensor_map) {
            if (applied_lora_tensors.find(kv.first) == applied_lora_tensors.end()) {
                LOG_WARN("unused lora tensor %s", kv.first.c_str());
            }
        }

        if (free_params_immediately) {
            remove_lora_params(lora_name);
        }

        int64_t t1 = ggml_time_ms();

        LOG_INFO("apply lora '%s:%f' completed, taking %.2fs",
                 lora_name.c_str(),
                 multiplier,
                 (t1 - t0) * 1.0f / 1000);
    }

    void apply_loras(const std::unordered_map<std::string, float>& lora_state) {
        std::unordered_map<std::string, float> lora_state_diff;
        for (auto& kv : lora_state) {
            const std::string& lora_name = kv.first;
            float multiplier             = kv.second;

            if (curr_lora_state.find(lora_name) != curr_lora_state.end()) {
                float curr_multiplier = curr_lora_state[lora_name];
                float multiplier_diff = multiplier - curr_multiplier;
                if (multiplier_diff != 0.f) {
                    lora_state_diff[lora_name] = multiplier_diff;
                }
            } else {
                lora_state_diff[lora_name] = multiplier;
            }
        }

        for (auto& kv : lora_state_diff) {
            apply_lora(kv.first, kv.second);
        }

        curr_lora_state = lora_state;
    }

    ggml_tensor* get_learned_condition(ggml_context* res_ctx, const std::string& text) {
        auto tokens_and_weights     = cond_stage_model.tokenize(text,
                                                                cond_stage_model.text_model.max_position_embeddings,
                                                                true);
        std::vector<int>& tokens    = tokens_and_weights.first;
        std::vector<float>& weights = tokens_and_weights.second;
        struct ggml_cplan cplan;
        size_t ctx_size = 10 * 1024 * 1024;  // 10MB
        // calculate the amount of memory required
        {
            struct ggml_init_params params;
            params.mem_size   = ctx_size;
            params.mem_buffer = NULL;
            params.no_alloc   = true;
            params.dynamic    = dynamic;

            struct ggml_context* ctx = ggml_init(params);
            if (!ctx) {
                LOG_ERROR("ggml_init() failed");
                return NULL;
            }

            ggml_set_dynamic(ctx, false);
            struct ggml_tensor* input_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tokens.size());
            ggml_set_dynamic(ctx, params.dynamic);

            struct ggml_tensor* hidden_states = cond_stage_model.text_model.forward(ctx, input_ids);

            struct ggml_cgraph* cond_graph = ggml_build_forward_ctx(ctx, hidden_states);
            cplan                          = ggml_graph_plan(cond_graph, n_threads);
            ctx_size += cplan.work_size;

            ctx_size += ggml_used_mem(ctx) + ggml_used_mem_of_data(ctx);
            LOG_DEBUG("condition context need %.2fMB static memory, with work_size needing %.2fMB",
                      ctx_size * 1.0f / 1024 / 1024,
                      cplan.work_size * 1.0f / 1024 / 1024);
            ggml_free(ctx);
        }

        // allocate the required memory and compute forward
        struct ggml_init_params params;
        params.mem_size   = ctx_size;
        params.mem_buffer = NULL;
        params.no_alloc   = false;
        params.dynamic    = dynamic;

        struct ggml_context* ctx = ggml_init(params);
        if (!ctx) {
            LOG_ERROR("ggml_init() failed");
            return NULL;
        }

        ggml_set_dynamic(ctx, false);
        struct ggml_tensor* input_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tokens.size());
        ggml_set_dynamic(ctx, params.dynamic);

        struct ggml_tensor* hidden_states = cond_stage_model.text_model.forward(ctx, input_ids);
        struct ggml_cgraph* cond_graph    = ggml_build_forward_ctx(ctx, hidden_states);
        LOG_DEBUG("building condition graph completed: %d nodes, %d leafs",
                  cond_graph->n_nodes, cond_graph->n_leafs);

        memcpy(input_ids->data, tokens.data(), tokens.size() * ggml_element_size(input_ids));

        int64_t t0 = ggml_time_ms();
        ggml_graph_compute_with_ctx(ctx, cond_graph, n_threads);
        int64_t t1 = ggml_time_ms();
        LOG_DEBUG("computing condition graph completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);

        ggml_tensor* result = ggml_dup_tensor(res_ctx, hidden_states);  // [N, n_token, hidden_size]

        {
            int64_t nelements   = ggml_nelements(hidden_states);
            float original_mean = 0.f;
            float new_mean      = 0.f;
            float* vec          = (float*)hidden_states->data;
            for (int i = 0; i < nelements; i++) {
                original_mean += vec[i] / nelements * 1.0f;
            }

            for (int i2 = 0; i2 < hidden_states->ne[2]; i2++) {
                for (int i1 = 0; i1 < hidden_states->ne[1]; i1++) {
                    for (int i0 = 0; i0 < hidden_states->ne[0]; i0++) {
                        float value = ggml_tensor_get_f32(hidden_states, i0, i1, i2);
                        value *= weights[i1];
                        ggml_tensor_set_f32(result, value, i0, i1, i2);
                    }
                }
            }

            vec = (float*)result->data;
            for (int i = 0; i < nelements; i++) {
                new_mean += vec[i] / nelements * 1.0f;
            }

            for (int i = 0; i < nelements; i++) {
                vec[i] = vec[i] * (original_mean / new_mean);
            }
        }

        // print_ggml_tensor(result);

        size_t rt_mem_size = ctx_size + ggml_curr_max_dynamic_size();
        if (rt_mem_size > max_rt_mem_size) {
            max_rt_mem_size = rt_mem_size;
        }
        size_t graph_mem_size = ggml_used_mem(clip_params_ctx) + rt_mem_size;

        size_t curr_mem_size = curr_params_mem_size + rt_mem_size;
        if (curr_mem_size > max_mem_size) {
            max_mem_size = curr_mem_size;
        }

        LOG_INFO(
            "condition graph use %.2fMB of memory: params %.2fMB, "
            "runtime %.2fMB (static %.2fMB, dynamic %.2fMB)",
            graph_mem_size * 1.0f / 1024 / 1024,
            ggml_used_mem(clip_params_ctx) * 1.0f / 1024 / 1024,
            rt_mem_size * 1.0f / 1024 / 1024,
            ctx_size * 1.0f / 1024 / 1024,
            ggml_curr_max_dynamic_size() * 1.0f / 1024 / 1024);

        LOG_DEBUG("%zu bytes of dynamic memory has not been released yet", ggml_dynamic_size());

        ggml_free(ctx);

        return result;  // [1, 77, 768]
    }

    ggml_tensor* sample(ggml_context* res_ctx,
                        ggml_tensor* x_t,
                        ggml_tensor* c,
                        ggml_tensor* uc,
                        float cfg_scale,
                        SampleMethod method,
                        const std::vector<float>& sigmas) {
        size_t steps = sigmas.size() - 1;
        // x_t = load_tensor_from_file(res_ctx, "./rand0.bin");
        // print_ggml_tensor(x_t);
        struct ggml_tensor* x = ggml_dup_tensor(res_ctx, x_t);
        copy_ggml_tensor(x, x_t);
        struct ggml_cplan cplan;

        size_t ctx_size = 10 * 1024 * 1024;  // 10MB
        // calculate the amount of memory required
        {
            struct ggml_init_params params;
            params.mem_size   = ctx_size;
            params.mem_buffer = NULL;
            params.no_alloc   = true;
            params.dynamic    = dynamic;

            struct ggml_context* ctx = ggml_init(params);
            if (!ctx) {
                LOG_ERROR("ggml_init() failed");
                return NULL;
            }

            ggml_set_dynamic(ctx, false);
            struct ggml_tensor* noised_input = ggml_dup_tensor(ctx, x_t);
            struct ggml_tensor* context      = ggml_dup_tensor(ctx, c);
            struct ggml_tensor* timesteps    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);                               // [N, ]
            struct ggml_tensor* t_emb        = new_timestep_embedding(ctx, timesteps, diffusion_model.model_channels);  // [N, model_channels]
            ggml_set_dynamic(ctx, params.dynamic);

            struct ggml_tensor* out = diffusion_model.forward(ctx, noised_input, NULL, context, t_emb);
            ctx_size += ggml_used_mem(ctx) + ggml_used_mem_of_data(ctx);

            struct ggml_cgraph* diffusion_graph = ggml_build_forward_ctx(ctx, out);
            cplan                               = ggml_graph_plan(diffusion_graph, n_threads);

            ctx_size += cplan.work_size;
            LOG_DEBUG("diffusion context need %.2fMB static memory, with work_size needing %.2fMB",
                      ctx_size * 1.0f / 1024 / 1024,
                      cplan.work_size * 1.0f / 1024 / 1024);

            ggml_free(ctx);
        }

        struct ggml_init_params params;
        params.mem_size   = ctx_size;
        params.mem_buffer = NULL;
        params.no_alloc   = false;
        params.dynamic    = dynamic;

        struct ggml_context* ctx = ggml_init(params);
        if (!ctx) {
            LOG_ERROR("ggml_init() failed");
            return NULL;
        }

        ggml_set_dynamic(ctx, false);
        struct ggml_tensor* noised_input = ggml_dup_tensor(ctx, x_t);
        struct ggml_tensor* context      = ggml_dup_tensor(ctx, c);
        struct ggml_tensor* timesteps    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);                               // [N, ]
        struct ggml_tensor* t_emb        = new_timestep_embedding(ctx, timesteps, diffusion_model.model_channels);  // [N, model_channels]
        ggml_set_dynamic(ctx, params.dynamic);

        struct ggml_tensor* out = diffusion_model.forward(ctx, noised_input, NULL, context, t_emb);
        ggml_hold_dynamic_tensor(out);

        struct ggml_cgraph* diffusion_graph = ggml_new_graph(ctx);
        diffusion_graph->order              = GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT;
        ggml_build_forward_expand(diffusion_graph, out);
        cplan = ggml_graph_plan(diffusion_graph, n_threads);

        ggml_set_dynamic(ctx, false);
        struct ggml_tensor* buf = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, cplan.work_size);
        ggml_set_dynamic(ctx, params.dynamic);

        cplan.work_data = (uint8_t*)buf->data;

        // x = x * sigmas[0]
        {
            float* vec = (float*)x->data;
            for (int i = 0; i < ggml_nelements(x); i++) {
                vec[i] = vec[i] * sigmas[0];
            }
        }

        // denoise wrapper
        ggml_set_dynamic(ctx, false);
        struct ggml_tensor* out_cond   = NULL;
        struct ggml_tensor* out_uncond = NULL;
        if (cfg_scale != 1.0f && uc != NULL) {
            out_uncond = ggml_dup_tensor(ctx, x);
        }
        struct ggml_tensor* denoised = ggml_dup_tensor(ctx, x);
        ggml_set_dynamic(ctx, params.dynamic);

        auto denoise = [&](ggml_tensor* input, float sigma, int step) {
            int64_t t0 = ggml_time_ms();

            float c_skip               = 1.0f;
            float c_out                = 1.0f;
            float c_in                 = 1.0f;
            std::vector<float> scaling = denoiser->get_scalings(sigma);
            if (scaling.size() == 3) {  // CompVisVDenoiser
                c_skip = scaling[0];
                c_out  = scaling[1];
                c_in   = scaling[2];
            } else {  // CompVisDenoiser
                c_out = scaling[0];
                c_in  = scaling[1];
            }

            float t = denoiser->schedule->sigma_to_t(sigma);
            ggml_set_f32(timesteps, t);
            set_timestep_embedding(timesteps, t_emb, diffusion_model.model_channels);

            copy_ggml_tensor(noised_input, input);
            // noised_input = noised_input * c_in
            {
                float* vec = (float*)noised_input->data;
                for (int i = 0; i < ggml_nelements(noised_input); i++) {
                    vec[i] = vec[i] * c_in;
                }
            }

            if (cfg_scale != 1.0 && uc != NULL) {
                // uncond
                copy_ggml_tensor(context, uc);
                ggml_graph_compute(diffusion_graph, &cplan);
                copy_ggml_tensor(out_uncond, out);

                // cond
                copy_ggml_tensor(context, c);
                ggml_graph_compute(diffusion_graph, &cplan);

                out_cond = out;

                // out_uncond + cfg_scale * (out_cond - out_uncond)
                {
                    float* vec_out        = (float*)out->data;
                    float* vec_out_uncond = (float*)out_uncond->data;
                    float* vec_out_cond   = (float*)out_cond->data;

                    for (int i = 0; i < ggml_nelements(out); i++) {
                        vec_out[i] = vec_out_uncond[i] + cfg_scale * (vec_out_cond[i] - vec_out_uncond[i]);
                    }
                }
            } else {
                // cond
                copy_ggml_tensor(context, c);
                ggml_graph_compute(diffusion_graph, &cplan);
            }

            // v = out, eps = out
            // denoised = (v * c_out + input * c_skip) or (input + eps * c_out)
            {
                float* vec_denoised = (float*)denoised->data;
                float* vec_input    = (float*)input->data;
                float* vec_out      = (float*)out->data;

                for (int i = 0; i < ggml_nelements(denoised); i++) {
                    vec_denoised[i] = vec_out[i] * c_out + vec_input[i] * c_skip;
                }
            }

#ifdef GGML_PERF
            ggml_graph_print(&diffusion_graph);
#endif
            int64_t t1 = ggml_time_ms();
            if (step > 0) {
                LOG_INFO("step %d sampling completed, taking %.2fs", step, (t1 - t0) * 1.0f / 1000);
                LOG_DEBUG("diffusion graph use %.2fMB runtime memory: static %.2fMB, dynamic %.2fMB",
                          (ctx_size + ggml_curr_max_dynamic_size()) * 1.0f / 1024 / 1024,
                          ctx_size * 1.0f / 1024 / 1024,
                          ggml_curr_max_dynamic_size() * 1.0f / 1024 / 1024);
                LOG_DEBUG("%zu bytes of dynamic memory has not been released yet", ggml_dynamic_size());
            }
        };

        // sample_euler_ancestral
        switch (method) {
            case EULER_A: {
                LOG_INFO("sampling using Euler A method");
                ggml_set_dynamic(ctx, false);
                struct ggml_tensor* noise = ggml_dup_tensor(ctx, x);
                struct ggml_tensor* d     = ggml_dup_tensor(ctx, x);
                ggml_set_dynamic(ctx, params.dynamic);

                for (int i = 0; i < steps; i++) {
                    float sigma = sigmas[i];

                    // denoise
                    denoise(x, sigma, i + 1);

                    // d = (x - denoised) / sigma
                    {
                        float* vec_d        = (float*)d->data;
                        float* vec_x        = (float*)x->data;
                        float* vec_denoised = (float*)denoised->data;

                        for (int i = 0; i < ggml_nelements(d); i++) {
                            vec_d[i] = (vec_x[i] - vec_denoised[i]) / sigma;
                        }
                    }

                    // get_ancestral_step
                    float sigma_up   = std::min(sigmas[i + 1],
                                                std::sqrt(sigmas[i + 1] * sigmas[i + 1] * (sigmas[i] * sigmas[i] - sigmas[i + 1] * sigmas[i + 1]) / (sigmas[i] * sigmas[i])));
                    float sigma_down = std::sqrt(sigmas[i + 1] * sigmas[i + 1] - sigma_up * sigma_up);

                    // Euler method
                    float dt = sigma_down - sigmas[i];
                    // x = x + d * dt
                    {
                        float* vec_d = (float*)d->data;
                        float* vec_x = (float*)x->data;

                        for (int i = 0; i < ggml_nelements(x); i++) {
                            vec_x[i] = vec_x[i] + vec_d[i] * dt;
                        }
                    }

                    if (sigmas[i + 1] > 0) {
                        // x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
                        ggml_tensor_set_f32_randn(noise, rng);
                        // noise = load_tensor_from_file(res_ctx, "./rand" + std::to_string(i+1) + ".bin");
                        {
                            float* vec_x     = (float*)x->data;
                            float* vec_noise = (float*)noise->data;

                            for (int i = 0; i < ggml_nelements(x); i++) {
                                vec_x[i] = vec_x[i] + vec_noise[i] * sigma_up;
                            }
                        }
                    }
                }
            } break;
            case EULER:  // Implemented without any sigma churn
            {
                LOG_INFO("sampling using Euler method");
                ggml_set_dynamic(ctx, false);
                struct ggml_tensor* d = ggml_dup_tensor(ctx, x);
                ggml_set_dynamic(ctx, params.dynamic);

                for (int i = 0; i < steps; i++) {
                    float sigma = sigmas[i];

                    // denoise
                    denoise(x, sigma, i + 1);

                    // d = (x - denoised) / sigma
                    {
                        float* vec_d        = (float*)d->data;
                        float* vec_x        = (float*)x->data;
                        float* vec_denoised = (float*)denoised->data;

                        for (int j = 0; j < ggml_nelements(d); j++) {
                            vec_d[j] = (vec_x[j] - vec_denoised[j]) / sigma;
                        }
                    }

                    float dt = sigmas[i + 1] - sigma;
                    // x = x + d * dt
                    {
                        float* vec_d = (float*)d->data;
                        float* vec_x = (float*)x->data;

                        for (int j = 0; j < ggml_nelements(x); j++) {
                            vec_x[j] = vec_x[j] + vec_d[j] * dt;
                        }
                    }
                }
            } break;
            case HEUN: {
                LOG_INFO("sampling using Heun method");
                ggml_set_dynamic(ctx, false);
                struct ggml_tensor* d  = ggml_dup_tensor(ctx, x);
                struct ggml_tensor* x2 = ggml_dup_tensor(ctx, x);
                ggml_set_dynamic(ctx, params.dynamic);

                for (int i = 0; i < steps; i++) {
                    // denoise
                    denoise(x, sigmas[i], -(i + 1));

                    // d = (x - denoised) / sigma
                    {
                        float* vec_d        = (float*)d->data;
                        float* vec_x        = (float*)x->data;
                        float* vec_denoised = (float*)denoised->data;

                        for (int j = 0; j < ggml_nelements(x); j++) {
                            vec_d[j] = (vec_x[j] - vec_denoised[j]) / sigmas[i];
                        }
                    }

                    float dt = sigmas[i + 1] - sigmas[i];
                    if (sigmas[i + 1] == 0) {
                        // Euler step
                        // x = x + d * dt
                        float* vec_d = (float*)d->data;
                        float* vec_x = (float*)x->data;

                        for (int j = 0; j < ggml_nelements(x); j++) {
                            vec_x[j] = vec_x[j] + vec_d[j] * dt;
                        }
                    } else {
                        // Heun step
                        float* vec_d  = (float*)d->data;
                        float* vec_d2 = (float*)d->data;
                        float* vec_x  = (float*)x->data;
                        float* vec_x2 = (float*)x2->data;

                        for (int j = 0; j < ggml_nelements(x); j++) {
                            vec_x2[j] = vec_x[j] + vec_d[j] * dt;
                        }

                        denoise(x2, sigmas[i + 1], i + 1);
                        float* vec_denoised = (float*)denoised->data;
                        for (int j = 0; j < ggml_nelements(x); j++) {
                            float d2 = (vec_x2[j] - vec_denoised[j]) / sigmas[i + 1];
                            vec_d[j] = (vec_d[j] + d2) / 2;
                            vec_x[j] = vec_x[j] + vec_d[j] * dt;
                        }
                    }
                }
            } break;
            case DPM2: {
                LOG_INFO("sampling using DPM2 method");
                ggml_set_dynamic(ctx, false);
                struct ggml_tensor* d  = ggml_dup_tensor(ctx, x);
                struct ggml_tensor* x2 = ggml_dup_tensor(ctx, x);
                ggml_set_dynamic(ctx, params.dynamic);

                for (int i = 0; i < steps; i++) {
                    // denoise
                    denoise(x, sigmas[i], i + 1);

                    // d = (x - denoised) / sigma
                    {
                        float* vec_d        = (float*)d->data;
                        float* vec_x        = (float*)x->data;
                        float* vec_denoised = (float*)denoised->data;

                        for (int j = 0; j < ggml_nelements(x); j++) {
                            vec_d[j] = (vec_x[j] - vec_denoised[j]) / sigmas[i];
                        }
                    }

                    if (sigmas[i + 1] == 0) {
                        // Euler step
                        // x = x + d * dt
                        float dt     = sigmas[i + 1] - sigmas[i];
                        float* vec_d = (float*)d->data;
                        float* vec_x = (float*)x->data;

                        for (int j = 0; j < ggml_nelements(x); j++) {
                            vec_x[j] = vec_x[j] + vec_d[j] * dt;
                        }
                    } else {
                        // DPM-Solver-2
                        float sigma_mid = exp(0.5f * (log(sigmas[i]) + log(sigmas[i + 1])));
                        float dt_1      = sigma_mid - sigmas[i];
                        float dt_2      = sigmas[i + 1] - sigmas[i];

                        float* vec_d  = (float*)d->data;
                        float* vec_x  = (float*)x->data;
                        float* vec_x2 = (float*)x2->data;
                        for (int j = 0; j < ggml_nelements(x); j++) {
                            vec_x2[j] = vec_x[j] + vec_d[j] * dt_1;
                        }

                        denoise(x2, sigma_mid, i + 1);
                        float* vec_denoised = (float*)denoised->data;
                        for (int j = 0; j < ggml_nelements(x); j++) {
                            float d2 = (vec_x2[j] - vec_denoised[j]) / sigma_mid;
                            vec_x[j] = vec_x[j] + d2 * dt_2;
                        }
                    }
                }

            } break;
            case DPMPP2S_A: {
                LOG_INFO("sampling using DPM++ (2s) a method");
                ggml_set_dynamic(ctx, false);
                struct ggml_tensor* noise = ggml_dup_tensor(ctx, x);
                struct ggml_tensor* d     = ggml_dup_tensor(ctx, x);
                struct ggml_tensor* x2    = ggml_dup_tensor(ctx, x);
                ggml_set_dynamic(ctx, params.dynamic);

                for (int i = 0; i < steps; i++) {
                    // denoise
                    denoise(x, sigmas[i], i + 1);

                    // get_ancestral_step
                    float sigma_up   = std::min(sigmas[i + 1],
                                                std::sqrt(sigmas[i + 1] * sigmas[i + 1] * (sigmas[i] * sigmas[i] - sigmas[i + 1] * sigmas[i + 1]) / (sigmas[i] * sigmas[i])));
                    float sigma_down = std::sqrt(sigmas[i + 1] * sigmas[i + 1] - sigma_up * sigma_up);
                    auto t_fn        = [](float sigma) -> float { return -log(sigma); };
                    auto sigma_fn    = [](float t) -> float { return exp(-t); };

                    if (sigma_down == 0) {
                        // Euler step
                        float* vec_d        = (float*)d->data;
                        float* vec_x        = (float*)x->data;
                        float* vec_denoised = (float*)denoised->data;

                        for (int j = 0; j < ggml_nelements(d); j++) {
                            vec_d[j] = (vec_x[j] - vec_denoised[j]) / sigmas[i];
                        }

                        // TODO: If sigma_down == 0, isn't this wrong?
                        // But
                        // https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py#L525
                        // has this exactly the same way.
                        float dt = sigma_down - sigmas[i];
                        for (int j = 0; j < ggml_nelements(d); j++) {
                            vec_x[j] = vec_x[j] + vec_d[j] * dt;
                        }
                    } else {
                        // DPM-Solver++(2S)
                        float t      = t_fn(sigmas[i]);
                        float t_next = t_fn(sigma_down);
                        float h      = t_next - t;
                        float s      = t + 0.5f * h;

                        float* vec_d        = (float*)d->data;
                        float* vec_x        = (float*)x->data;
                        float* vec_x2       = (float*)x2->data;
                        float* vec_denoised = (float*)denoised->data;

                        // First half-step
                        for (int j = 0; j < ggml_nelements(x); j++) {
                            vec_x2[j] = (sigma_fn(s) / sigma_fn(t)) * vec_x[j] - (exp(-h * 0.5f) - 1) * vec_denoised[j];
                        }

                        denoise(x2, sigmas[i + 1], i + 1);

                        // Second half-step
                        for (int j = 0; j < ggml_nelements(x); j++) {
                            vec_x[j] = (sigma_fn(t_next) / sigma_fn(t)) * vec_x[j] - (exp(-h) - 1) * vec_denoised[j];
                        }
                    }

                    // Noise addition
                    if (sigmas[i + 1] > 0) {
                        ggml_tensor_set_f32_randn(noise, rng);
                        {
                            float* vec_x     = (float*)x->data;
                            float* vec_noise = (float*)noise->data;

                            for (int i = 0; i < ggml_nelements(x); i++) {
                                vec_x[i] = vec_x[i] + vec_noise[i] * sigma_up;
                            }
                        }
                    }
                }
            } break;
            case DPMPP2M:  // DPM++ (2M) from Karras et al (2022)
            {
                LOG_INFO("sampling using DPM++ (2M) method");
                ggml_set_dynamic(ctx, false);
                struct ggml_tensor* old_denoised = ggml_dup_tensor(ctx, x);
                ggml_set_dynamic(ctx, params.dynamic);

                auto t_fn = [](float sigma) -> float { return -log(sigma); };

                for (int i = 0; i < steps; i++) {
                    // denoise
                    denoise(x, sigmas[i], i + 1);

                    float t                 = t_fn(sigmas[i]);
                    float t_next            = t_fn(sigmas[i + 1]);
                    float h                 = t_next - t;
                    float a                 = sigmas[i + 1] / sigmas[i];
                    float b                 = exp(-h) - 1.f;
                    float* vec_x            = (float*)x->data;
                    float* vec_denoised     = (float*)denoised->data;
                    float* vec_old_denoised = (float*)old_denoised->data;

                    if (i == 0 || sigmas[i + 1] == 0) {
                        // Simpler step for the edge cases
                        for (int j = 0; j < ggml_nelements(x); j++) {
                            vec_x[j] = a * vec_x[j] - b * vec_denoised[j];
                        }
                    } else {
                        float h_last = t - t_fn(sigmas[i - 1]);
                        float r      = h_last / h;
                        for (int j = 0; j < ggml_nelements(x); j++) {
                            float denoised_d = (1.f + 1.f / (2.f * r)) * vec_denoised[j] - (1.f / (2.f * r)) * vec_old_denoised[j];
                            vec_x[j]         = a * vec_x[j] - b * denoised_d;
                        }
                    }

                    // old_denoised = denoised
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_old_denoised[j] = vec_denoised[j];
                    }
                }
            } break;
            case DPMPP2Mv2:  // Modified DPM++ (2M) from https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/8457
            {
                LOG_INFO("sampling using modified DPM++ (2M) method");
                ggml_set_dynamic(ctx, false);
                struct ggml_tensor* old_denoised = ggml_dup_tensor(ctx, x);
                ggml_set_dynamic(ctx, params.dynamic);

                auto t_fn = [](float sigma) -> float { return -log(sigma); };

                for (int i = 0; i < steps; i++) {
                    // denoise
                    denoise(x, sigmas[i], i + 1);

                    float t                 = t_fn(sigmas[i]);
                    float t_next            = t_fn(sigmas[i + 1]);
                    float h                 = t_next - t;
                    float a                 = sigmas[i + 1] / sigmas[i];
                    float* vec_x            = (float*)x->data;
                    float* vec_denoised     = (float*)denoised->data;
                    float* vec_old_denoised = (float*)old_denoised->data;

                    if (i == 0 || sigmas[i + 1] == 0) {
                        // Simpler step for the edge cases
                        float b = exp(-h) - 1.f;
                        for (int j = 0; j < ggml_nelements(x); j++) {
                            vec_x[j] = a * vec_x[j] - b * vec_denoised[j];
                        }
                    } else {
                        float h_last = t - t_fn(sigmas[i - 1]);
                        float h_min  = std::min(h_last, h);
                        float h_max  = std::max(h_last, h);
                        float r      = h_max / h_min;
                        float h_d    = (h_max + h_min) / 2.f;
                        float b      = exp(-h_d) - 1.f;
                        for (int j = 0; j < ggml_nelements(x); j++) {
                            float denoised_d = (1.f + 1.f / (2.f * r)) * vec_denoised[j] - (1.f / (2.f * r)) * vec_old_denoised[j];
                            vec_x[j]         = a * vec_x[j] - b * denoised_d;
                        }
                    }

                    // old_denoised = denoised
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_old_denoised[j] = vec_denoised[j];
                    }
                }
            } break;
            case LCM:  // Latent Consistency Models
            {
                LOG_INFO("sampling using LCM method");
                ggml_set_dynamic(ctx, false);
                struct ggml_tensor* noise = ggml_dup_tensor(ctx, x);
                struct ggml_tensor* d     = ggml_dup_tensor(ctx, x);
                ggml_set_dynamic(ctx, params.dynamic);

                for (int i = 0; i < steps; i++) {
                    float sigma = sigmas[i];

                    // denoise
                    denoise(x, sigma, i + 1);

                    // x = denoised
                    {
                        float* vec_x        = (float*)x->data;
                        float* vec_denoised = (float*)denoised->data;
                        for (int j = 0; j < ggml_nelements(x); j++) {
                            vec_x[j] = vec_denoised[j];
                        }
                    }

                    if (sigmas[i + 1] > 0) {
                        // x += sigmas[i + 1] * noise_sampler(sigmas[i], sigmas[i + 1])
                        ggml_tensor_set_f32_randn(noise, rng);
                        // noise = load_tensor_from_file(res_ctx, "./rand" + std::to_string(i+1) + ".bin");
                        {
                            float* vec_x     = (float*)x->data;
                            float* vec_noise = (float*)noise->data;

                            for (int j = 0; j < ggml_nelements(x); j++) {
                                vec_x[j] = vec_x[j] + sigmas[i + 1] * vec_noise[j];
                            }
                        }
                    }
                }
            } break;

            default:
                LOG_ERROR("Attempting to sample with nonexisting sample method %i", method);
                abort();
        }

        size_t rt_mem_size = ctx_size + ggml_curr_max_dynamic_size();
        if (rt_mem_size > max_rt_mem_size) {
            max_rt_mem_size = rt_mem_size;
        }
        size_t graph_mem_size = ggml_used_mem(unet_params_ctx) + rt_mem_size;

        size_t curr_mem_size = curr_params_mem_size + rt_mem_size;
        if (curr_mem_size > max_mem_size) {
            max_mem_size = curr_mem_size;
        }

        LOG_INFO(
            "diffusion graph use %.2fMB of memory: params %.2fMB, "
            "runtime %.2fMB (static %.2fMB, dynamic %.2fMB)",
            graph_mem_size * 1.0f / 1024 / 1024,
            ggml_used_mem(unet_params_ctx) * 1.0f / 1024 / 1024,
            rt_mem_size * 1.0f / 1024 / 1024,
            ctx_size * 1.0f / 1024 / 1024,
            ggml_curr_max_dynamic_size() * 1.0f / 1024 / 1024);
        LOG_DEBUG("%zu bytes of dynamic memory has not been released yet", ggml_dynamic_size());

        ggml_free(ctx);

        return x;
    }

    ggml_tensor* encode_first_stage(ggml_context* res_ctx, ggml_tensor* x) {
        int64_t W                  = x->ne[0];
        int64_t H                  = x->ne[1];
        struct ggml_tensor* result = NULL;
        struct ggml_cplan cplan;

        // calculate the amount of memory required
        size_t ctx_size = 10 * 1024 * 1024;  // 10MB
        {
            struct ggml_init_params params;
            params.mem_size   = ctx_size;
            params.mem_buffer = NULL;
            params.no_alloc   = true;
            params.dynamic    = dynamic;

            struct ggml_context* ctx = ggml_init(params);
            if (!ctx) {
                LOG_ERROR("ggml_init() failed");
                return NULL;
            }

            struct ggml_tensor* moments = first_stage_model.encode(ctx, x);
            ctx_size += ggml_used_mem(ctx) + ggml_used_mem_of_data(ctx);

            struct ggml_cgraph* vae_graph = ggml_build_forward_ctx(ctx, moments);
            cplan                         = ggml_graph_plan(vae_graph, n_threads);

            ctx_size += cplan.work_size;
            LOG_DEBUG("vae context need %.2fMB static memory, with work_size needing %.2fMB",
                      ctx_size * 1.0f / 1024 / 1024,
                      cplan.work_size * 1.0f / 1024 / 1024);

            ggml_free(ctx);
        }

        {
            struct ggml_init_params params;
            params.mem_size   = ctx_size;
            params.mem_buffer = NULL;
            params.no_alloc   = false;
            params.dynamic    = dynamic;

            struct ggml_context* ctx = ggml_init(params);
            if (!ctx) {
                LOG_ERROR("ggml_init() failed");
                return NULL;
            }

            struct ggml_tensor* moments = first_stage_model.encode(ctx, x);

            struct ggml_cgraph* vae_graph = ggml_new_graph(ctx);
            vae_graph->order              = GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT;
            ggml_build_forward_expand(vae_graph, moments);

            int64_t t0 = ggml_time_ms();
            ggml_graph_compute_with_ctx(ctx, vae_graph, n_threads);
            int64_t t1 = ggml_time_ms();

#ifdef GGML_PERF
            ggml_graph_print(&vae_graph);
#endif
            LOG_DEBUG("computing vae graph completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);

            result = ggml_dup_tensor(res_ctx, moments);
            copy_ggml_tensor(result, moments);

            size_t rt_mem_size = ctx_size + ggml_curr_max_dynamic_size();
            if (rt_mem_size > max_rt_mem_size) {
                max_rt_mem_size = rt_mem_size;
            }
            size_t graph_mem_size = ggml_used_mem(vae_params_ctx) + rt_mem_size;

            size_t curr_mem_size = curr_params_mem_size + rt_mem_size;
            if (curr_mem_size > max_mem_size) {
                max_mem_size = curr_mem_size;
            }

            LOG_INFO(
                "vae graph use %.2fMB of memory: params %.2fMB, "
                "runtime %.2fMB (static %.2fMB, dynamic %.2fMB)",
                graph_mem_size * 1.0f / 1024 / 1024,
                ggml_used_mem(vae_params_ctx) * 1.0f / 1024 / 1024,
                rt_mem_size * 1.0f / 1024 / 1024,
                ctx_size * 1.0f / 1024 / 1024,
                ggml_curr_max_dynamic_size() * 1.0f / 1024 / 1024);
            LOG_DEBUG("%zu bytes of dynamic memory has not been released yet", ggml_dynamic_size());

            ggml_free(ctx);
        }

        return result;
    }

    // ldm.models.diffusion.ddpm.LatentDiffusion.get_first_stage_encoding
    ggml_tensor* get_first_stage_encoding(ggml_context* res_ctx, ggml_tensor* moments) {
        // ldm.modules.distributions.distributions.DiagonalGaussianDistribution.sample
        ggml_tensor* latent       = ggml_new_tensor_4d(res_ctx, moments->type, moments->ne[0],
                                                       moments->ne[1], moments->ne[2] / 2, moments->ne[3]);
        struct ggml_tensor* noise = ggml_dup_tensor(res_ctx, latent);
        ggml_tensor_set_f32_randn(noise, rng);
        // noise = load_tensor_from_file(res_ctx, "noise.bin");
        {
            float mean   = 0;
            float logvar = 0;
            float value  = 0;
            float std_   = 0;
            for (int i = 0; i < latent->ne[3]; i++) {
                for (int j = 0; j < latent->ne[2]; j++) {
                    for (int k = 0; k < latent->ne[1]; k++) {
                        for (int l = 0; l < latent->ne[0]; l++) {
                            mean   = ggml_tensor_get_f32(moments, l, k, j, i);
                            logvar = ggml_tensor_get_f32(moments, l, k, j + (int)latent->ne[2], i);
                            logvar = std::max(-30.0f, std::min(logvar, 20.0f));
                            std_   = std::exp(0.5f * logvar);
                            value  = mean + std_ * ggml_tensor_get_f32(noise, l, k, j, i);
                            value  = value * scale_factor;
                            // printf("%d %d %d %d -> %f\n", i, j, k, l, value);
                            ggml_tensor_set_f32(latent, value, l, k, j, i);
                        }
                    }
                }
            }
        }
        return latent;
    }

    ggml_tensor* decode_first_stage(ggml_context* res_ctx, ggml_tensor* z) {
        int64_t W                      = z->ne[0];
        int64_t H                      = z->ne[1];
        struct ggml_tensor* result_img = NULL;
        struct ggml_cplan cplan;

        {
            float* vec = (float*)z->data;
            for (int i = 0; i < ggml_nelements(z); i++) {
                vec[i] = 1.0f / scale_factor * vec[i];
            }
        }

        // calculate the amount of memory required
        size_t ctx_size = 10 * 1024 * 1024;  // 10MB
        {
            struct ggml_init_params params;
            params.mem_size   = ctx_size;
            params.mem_buffer = NULL;
            params.no_alloc   = true;
            params.dynamic    = dynamic;

            struct ggml_context* ctx = ggml_init(params);
            if (!ctx) {
                LOG_ERROR("ggml_init() failed");
                return NULL;
            }

            struct ggml_tensor* img = first_stage_model.decoder.forward(ctx, z);
            ctx_size += ggml_used_mem(ctx) + ggml_used_mem_of_data(ctx);

            struct ggml_cgraph* vae_graph = ggml_build_forward_ctx(ctx, img);
            cplan                         = ggml_graph_plan(vae_graph, n_threads);

            ctx_size += cplan.work_size;
            LOG_DEBUG("vae context need %.2fMB static memory, with work_size needing %.2fMB",
                      ctx_size * 1.0f / 1024 / 1024,
                      cplan.work_size * 1.0f / 1024 / 1024);

            ggml_free(ctx);
        }

        {
            struct ggml_init_params params;
            params.mem_size   = ctx_size;
            params.mem_buffer = NULL;
            params.no_alloc   = false;
            params.dynamic    = dynamic;

            struct ggml_context* ctx = ggml_init(params);
            if (!ctx) {
                LOG_ERROR("ggml_init() failed");
                return NULL;
            }

            struct ggml_tensor* img = first_stage_model.decode(ctx, z);

            struct ggml_cgraph* vae_graph = ggml_new_graph(ctx);
            vae_graph->order              = GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT;
            ggml_build_forward_expand(vae_graph, img);

            int64_t t0 = ggml_time_ms();
            ggml_graph_compute_with_ctx(ctx, vae_graph, n_threads);
            int64_t t1 = ggml_time_ms();

#ifdef GGML_PERF
            ggml_graph_print(&vae_graph);
#endif
            LOG_DEBUG("computing vae graph completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);

            result_img = ggml_dup_tensor(res_ctx, img);
            copy_ggml_tensor(result_img, img);

            size_t rt_mem_size = ctx_size + ggml_curr_max_dynamic_size();
            if (rt_mem_size > max_rt_mem_size) {
                max_rt_mem_size = rt_mem_size;
            }
            size_t graph_mem_size = ggml_used_mem(vae_params_ctx) + rt_mem_size;

            size_t curr_mem_size = curr_params_mem_size + rt_mem_size;
            if (curr_mem_size > max_mem_size) {
                max_mem_size = curr_mem_size;
            }

            LOG_INFO(
                "vae graph use %.2fMB of memory: params %.2fMB, "
                "runtime %.2fMB (static %.2fMB, dynamic %.2fMB)",
                graph_mem_size * 1.0f / 1024 / 1024,
                ggml_used_mem(vae_params_ctx) * 1.0f / 1024 / 1024,
                rt_mem_size * 1.0f / 1024 / 1024,
                ctx_size * 1.0f / 1024 / 1024,
                ggml_curr_max_dynamic_size() * 1.0f / 1024 / 1024);
            LOG_DEBUG("%zu bytes of dynamic memory has not been released yet", ggml_dynamic_size());

            ggml_free(ctx);
        }

        return result_img;
    }
};

/*================================================= StableDiffusion ==================================================*/

StableDiffusion::StableDiffusion(int n_threads,
                                 bool vae_decode_only,
                                 bool free_params_immediately,
                                 std::string lora_model_dir,
                                 RNGType rng_type) {
    sd = std::make_shared<StableDiffusionGGML>(n_threads,
                                               vae_decode_only,
                                               free_params_immediately,
                                               lora_model_dir,
                                               rng_type);
}

bool StableDiffusion::load_from_file(const std::string& file_path, Schedule s) {
    return sd->load_from_file(file_path, s);
}

std::vector<uint8_t> StableDiffusion::txt2img(std::string prompt,
                                              std::string negative_prompt,
                                              float cfg_scale,
                                              int width,
                                              int height,
                                              SampleMethod sample_method,
                                              int sample_steps,
                                              int64_t seed) {
    std::vector<uint8_t> result;
    struct ggml_init_params params;
    params.mem_size = static_cast<size_t>(10 * 1024) * 1024;  // 10M
    params.mem_size += width * height * 3 * sizeof(float) * 2;
    params.mem_buffer        = NULL;
    params.no_alloc          = false;
    params.dynamic           = false;
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        LOG_ERROR("ggml_init() failed");
        return result;
    }

    if (seed < 0) {
        seed = (int)time(NULL);
    }
    sd->rng->manual_seed(seed);

    // extract and remove lora
    auto result_pair                                = extract_and_remove_lora(prompt);
    std::unordered_map<std::string, float> lora_f2m = result_pair.first;  // lora_name -> multiplier
    for (auto& kv : lora_f2m) {
        LOG_DEBUG("lora %s:%.2f", kv.first.c_str(), kv.second);
    }
    prompt = result_pair.second;
    LOG_DEBUG("prompt after extract and remove lora: \"%s\"", prompt.c_str());

    // load lora from file
    int64_t t0 = ggml_time_ms();
    sd->apply_loras(lora_f2m);
    int64_t t1 = ggml_time_ms();
    LOG_INFO("apply_loras completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);

    t0                     = ggml_time_ms();
    ggml_tensor* c         = sd->get_learned_condition(ctx, prompt);
    struct ggml_tensor* uc = NULL;
    if (cfg_scale != 1.0) {
        uc = sd->get_learned_condition(ctx, negative_prompt);
    }
    t1 = ggml_time_ms();
    LOG_INFO("get_learned_condition completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);

    if (sd->free_params_immediately) {
        sd->curr_params_mem_size -= ggml_used_mem(sd->clip_params_ctx);
        ggml_free(sd->clip_params_ctx);
        sd->clip_params_ctx = NULL;
    }

    int C                   = 4;
    int W                   = width / 8;
    int H                   = height / 8;
    struct ggml_tensor* x_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, W, H, C, 1);
    ggml_tensor_set_f32_randn(x_t, sd->rng);

    std::vector<float> sigmas = sd->denoiser->schedule->get_sigmas(sample_steps);

    LOG_INFO("start sampling");
    struct ggml_tensor* x_0 = sd->sample(ctx, x_t, c, uc, cfg_scale, sample_method, sigmas);
    // struct ggml_tensor* x_0 = load_tensor_from_file(ctx, "samples_ddim.bin");
    // print_ggml_tensor(x_0);
    int64_t t2 = ggml_time_ms();
    LOG_INFO("sampling completed, taking %.2fs", (t2 - t1) * 1.0f / 1000);

    if (sd->free_params_immediately) {
        sd->curr_params_mem_size -= ggml_used_mem(sd->unet_params_ctx);
        ggml_free(sd->unet_params_ctx);
        sd->unet_params_ctx = NULL;
    }

    struct ggml_tensor* img = sd->decode_first_stage(ctx, x_0);
    if (img != NULL) {
        result = ggml_to_image_vec(img);
    }
    int64_t t3 = ggml_time_ms();
    LOG_INFO("decode_first_stage completed, taking %.2fs", (t3 - t2) * 1.0f / 1000);

    if (sd->free_params_immediately) {
        sd->curr_params_mem_size -= ggml_used_mem(sd->vae_params_ctx);
        ggml_free(sd->vae_params_ctx);
        sd->vae_params_ctx = NULL;
    }

    LOG_INFO(
        "txt2img completed in %.2fs, use %.2fMB of memory: peak params memory %.2fMB, "
        "peak runtime memory %.2fMB",
        (t3 - t0) * 1.0f / 1000,
        sd->max_mem_size * 1.0f / 1024 / 1024,
        sd->max_params_mem_size * 1.0f / 1024 / 1024,
        sd->max_rt_mem_size * 1.0f / 1024 / 1024);

    ggml_free(ctx);
    return result;
}

std::vector<uint8_t> StableDiffusion::img2img(const std::vector<uint8_t>& init_img_vec,
                                              std::string prompt,
                                              std::string negative_prompt,
                                              float cfg_scale,
                                              int width,
                                              int height,
                                              SampleMethod sample_method,
                                              int sample_steps,
                                              float strength,
                                              int64_t seed) {
    std::vector<uint8_t> result;
    if (init_img_vec.size() != width * height * 3) {
        return result;
    }
    LOG_INFO("img2img %dx%d", width, height);

    std::vector<float> sigmas = sd->denoiser->schedule->get_sigmas(sample_steps);
    size_t t_enc              = static_cast<size_t>(sample_steps * strength);
    LOG_INFO("target t_enc is %zu steps", t_enc);
    std::vector<float> sigma_sched;
    sigma_sched.assign(sigmas.begin() + sample_steps - t_enc - 1, sigmas.end());

    struct ggml_init_params params;
    params.mem_size = static_cast<size_t>(10 * 1024) * 1024;  // 10M
    params.mem_size += width * height * 3 * sizeof(float) * 2;
    params.mem_buffer        = NULL;
    params.no_alloc          = false;
    params.dynamic           = false;
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        LOG_ERROR("ggml_init() failed");
        return result;
    }

    if (seed < 0) {
        seed = (int)time(NULL);
    }
    sd->rng->manual_seed(seed);

    // extract and remove lora
    auto result_pair                                = extract_and_remove_lora(prompt);
    std::unordered_map<std::string, float> lora_f2m = result_pair.first;  // lora_name -> multiplier
    for (auto& kv : lora_f2m) {
        LOG_DEBUG("lora %s:%.2f", kv.first.c_str(), kv.second);
    }
    prompt = result_pair.second;
    LOG_DEBUG("prompt after extract and remove lora: \"%s\"", prompt.c_str());

    // load lora from file
    int64_t t0 = ggml_time_ms();
    sd->apply_loras(lora_f2m);
    int64_t t1 = ggml_time_ms();
    LOG_INFO("apply_loras completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);

    ggml_tensor* init_img = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, width, height, 3, 1);
    image_vec_to_ggml(init_img_vec, init_img);

    t0                       = ggml_time_ms();
    ggml_tensor* moments     = sd->encode_first_stage(ctx, init_img);
    ggml_tensor* init_latent = sd->get_first_stage_encoding(ctx, moments);
    // print_ggml_tensor(init_latent);
    t1 = ggml_time_ms();
    LOG_INFO("encode_first_stage completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);

    ggml_reset_curr_max_dynamic_size();  // reset counter

    ggml_tensor* c         = sd->get_learned_condition(ctx, prompt);
    struct ggml_tensor* uc = NULL;
    if (cfg_scale != 1.0) {
        uc = sd->get_learned_condition(ctx, negative_prompt);
    }
    int64_t t2 = ggml_time_ms();
    LOG_INFO("get_learned_condition completed, taking %.2fs", (t2 - t1) * 1.0f / 1000);
    if (sd->free_params_immediately) {
        sd->curr_params_mem_size -= ggml_used_mem(sd->clip_params_ctx);
        ggml_free(sd->clip_params_ctx);
        sd->clip_params_ctx = NULL;
    }

    LOG_INFO("start sampling");
    struct ggml_tensor* x_0 = sd->sample(ctx, init_latent, c, uc, cfg_scale, sample_method, sigma_sched);
    // struct ggml_tensor *x_0 = load_tensor_from_file(ctx, "samples_ddim.bin");
    // print_ggml_tensor(x_0);
    int64_t t3 = ggml_time_ms();
    LOG_INFO("sampling completed, taking %.2fs", (t3 - t2) * 1.0f / 1000);
    if (sd->free_params_immediately) {
        sd->curr_params_mem_size -= ggml_used_mem(sd->unet_params_ctx);
        ggml_free(sd->unet_params_ctx);
        sd->unet_params_ctx = NULL;
    }

    struct ggml_tensor* img = sd->decode_first_stage(ctx, x_0);
    if (img != NULL) {
        result = ggml_to_image_vec(img);
    }
    int64_t t4 = ggml_time_ms();
    LOG_INFO("decode_first_stage completed, taking %.2fs", (t4 - t3) * 1.0f / 1000);

    if (sd->free_params_immediately) {
        sd->curr_params_mem_size -= ggml_used_mem(sd->vae_params_ctx);
        ggml_free(sd->vae_params_ctx);
        sd->vae_params_ctx = NULL;
    }

    LOG_INFO(
        "img2img completed in %.2fs, use %.2fMB of memory: peak params memory %.2fMB, "
        "peak runtime memory %.2fMB",
        (t4 - t0) * 1.0f / 1000,
        sd->max_mem_size * 1.0f / 1024 / 1024,
        sd->max_params_mem_size * 1.0f / 1024 / 1024,
        sd->max_rt_mem_size * 1.0f / 1024 / 1024);

    ggml_free(ctx);

    return result;
}
