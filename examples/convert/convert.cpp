#include "ggml/ggml.h"

// third-party libraries
#include "json.hpp"
#include "zip.h"

#include <stdio.h>
#include <cstdlib>
#include <string>
#include <vector>

#include "vocab.hpp"

using json = nlohmann::json;

#define MAX_STRING_BUFFER 90
#define UNUSED_MODEL_TENSORS 19
#define TIMESTEPS 1000

const char* unused_tensors[UNUSED_MODEL_TENSORS] = {
    "betas",
    "alphas_cumprod_prev",
    "sqrt_alphas_cumprod",
    "sqrt_one_minus_alphas_cumprod",
    "log_one_minus_alphas_cumprod",
    "sqrt_recip_alphas_cumprod",
    "sqrt_recipm1_alphas_cumprod",
    "posterior_variance",
    "posterior_log_variance_clipped",
    "posterior_mean_coef1",
    "posterior_mean_coef2",
    "cond_stage_model.transformer.text_model.embeddings.position_ids",
    "cond_stage_model.model.logit_scale",
    "cond_stage_model.model.text_projection",
    "model_ema.decay",
    "model_ema.num_updates",
    "model_ema.diffusion_model",
    "control_model",
    "embedding_manager"
};

std::string kqv_self[6] = {
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",

    "self_attn.q_proj.bias",
    "self_attn.k_proj.bias",
    "self_attn.v_proj.bias"
};

enum sd_version {
    VERSION_1_x,
    VERSION_2_x,
    VERSION_XL
};

enum read_phase {
    READ_NAME,
    READ_DATA,
    CHECK_SIZE,
    READ_DIMENS
};

enum sd_lora_type {
    LORA_NONE,
    LORA_REGULAR,
    LORA_DIFFUSERS,
    LORA_TRANSFORMERS
};

struct convert_params {
    ggml_type out_type = GGML_TYPE_F32;
    sd_version version = VERSION_1_x;
    std::string model_name = "";
    std::string model_path = "";
    std::string output_path = "";
    std::string vocab_path = "";
    bool verbose = false;
    bool generate_alphas_cumprod = false;

    // LoRA
    bool lora = false;
    std::map<std::string, float> lora_alphas;
    sd_lora_type lora_type = LORA_NONE;

    // VAE
    bool vae = false;
};

struct Tensor {
    std::string name;
    size_t data_offset;
    ggml_type dtype;
    size_t data_size = 0;
    int32_t shape[4];
    int32_t n_dims = 0;
    read_phase t_phase;
    int32_t num_elements = 0;
    bool is_view = false;
    void* data = NULL;

    void dump() {
        printf("Tensor: %30s | n_dim: %i | [%i, %i, %i, %i] | %s \n", name.c_str(), n_dims, shape[0], shape[1], shape[2], shape[3], ggml_type_name(dtype));
    }

    int64_t* inverse_shape() {
        int64_t* v = new int64_t[4];
        for(int i = 0;i < 4; i++) {
            v[i] = (i < n_dims) ? shape[n_dims - 1 - i] : 1;
        }
        return v;
    }
};

/*

    UTILS FUNTIONS

*/

int64_t read_long(uint8_t* buffer) {
    // little endian
    int64_t value = 0;
    value |= buffer[7] << 56;
    value |= buffer[6] << 48;
    value |= buffer[5] << 40;
    value |= buffer[4] << 32;
    value |= buffer[3] << 24;
    value |= buffer[2] << 16;
    value |= buffer[1] << 8;
    value |= buffer[0];
    return value;
}

int32_t read_int(uint8_t* buffer) {
    // little endian
    int value = 0;
    value |= buffer[3] << 24;
    value |= buffer[2] << 16;
    value |= buffer[1] << 8;
    value |= buffer[0];
    return value;
}

uint16_t read_short(uint8_t* buffer) {
     // little endian
    uint16_t value = 0;
    value |= buffer[1] << 8;
    value |= buffer[0];
    return value;
}

int8_t findChar(uint8_t* buffer, char c) {
    for(int8_t len = 0; len < MAX_STRING_BUFFER; len++) {
        if(buffer[len] == c) {
            return len;
        }
    }
    return -1;
}

// ported from https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py#L16
std::map<char, int> unicode_to_byte() {
    std::map<int, char> byteToUnicode;
    
    // List of utf-8 byte ranges
    for (int b = static_cast<int>('!'); b <= static_cast<int>('~'); ++b) {
        byteToUnicode[b] = static_cast<char>(b);
    }
    
    for (int b = static_cast<int>('¡'); b <= static_cast<int>('¬'); ++b) {
        byteToUnicode[b] = static_cast<char>(b);
    }
    
    for (int b = static_cast<int>('®'); b <= static_cast<int>('ÿ'); ++b) {
        byteToUnicode[b] = static_cast<char>(b);
    }
    
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (byteToUnicode.find(b) == byteToUnicode.end()) {
            byteToUnicode[b] = static_cast<char>(256 + n);
            n++;
        }
    }

    // byte_encoder = bytes_to_unicode()
    // byte_decoder = {v: k for k, v in byte_encoder.items()}
    std::map<char, int> byteDecoder;

     for (const auto& entry : byteToUnicode) {
        byteDecoder[entry.second] = entry.first;
    }

    byteToUnicode.clear();
    
    return byteDecoder;
}

bool is_unused_tensor(std::string name) {
    for(int i = 0; i < UNUSED_MODEL_TENSORS;i++) {
        if(name.find(unused_tensors[i]) == 0) {
            return true;
        }
    }
    return false;
}

float* calculate_alpha_cumprod(float linear_start = 0.00085f, float linear_end = 0.0120,  int timesteps = TIMESTEPS) {
    float* ac = (float*)malloc(timesteps * 4);
    float ls_sqrt = sqrtf(linear_start);
    float le_sqrt = sqrtf(linear_end);
    float amount = le_sqrt - ls_sqrt;
    float product = 1.0f;
    for(int i = 0; i < timesteps; i++) {
        float beta = ls_sqrt + amount * ((float)i / (timesteps - 1));
        product *= 1.0f - powf(beta, 2.0f);
        ac[i] = product;
    }
    return ac;
}

/*

    READ PYTORCH CHECKPOINT MODEL

*/

static ggml_type global_type = GGML_TYPE_F32; // all tensors data type
static bool read_global_type = false;

void exist_in_zip(struct zip_t *zip, const char* f_test, Tensor & tensor) {
    size_t i, n = zip_entries_total(zip);
    for (i = 0; i < n; ++i) {
        zip_entry_openbyindex(zip, i);
        {
            const char *name = zip_entry_name(zip);
            if(strcmp(name, f_test) == 0) {
                tensor.data_offset = i;
                tensor.data_size = zip_entry_size(zip);
                zip_entry_close(zip);
                return;
            }
        }
        zip_entry_close(zip);
    }
}

bool set_pkl_tensor_props(uint32_t value,struct Tensor & tensor) {
    if(tensor.t_phase == CHECK_SIZE) {
        if(tensor.data_size == value * ggml_type_size(tensor.dtype)) {
            tensor.num_elements = value;
            tensor.t_phase = READ_DIMENS;
            return true;
        } else {
            tensor.t_phase = READ_NAME;
        }
    } else if (tensor.t_phase == READ_DIMENS) {
        if(tensor.n_dims + 1 > 4) { // too many dimens
            tensor.t_phase = READ_NAME;
            tensor.n_dims = 0;
        }
        if(tensor.num_elements % value == 0) {
            tensor.shape[tensor.n_dims] = value;
            tensor.n_dims++;
        }
    }
    return false;
}

void read_pkl_data_type(char* _name,struct Tensor & tensor) {
    if(!strcmp(_name, "FloatStorage")) {
        if(read_global_type) {
            global_type = GGML_TYPE_F32;
            read_global_type = false;
        }
        tensor.dtype = GGML_TYPE_F32;
    } else if(!strcmp(_name, "HalfStorage")) {
        if(read_global_type) {
            global_type = GGML_TYPE_F16;
            read_global_type = false;
        }
        tensor.dtype = GGML_TYPE_F16;
    }
}

void read_pkl_string(char* text_str,struct zip_t *zip, std::string dir, struct Tensor & tensor) {
    if(!strcmp(text_str, "storage")) {
        read_global_type = true;
    } else if(strcmp(text_str, "state_dict")) { // no state_dict
        if(tensor.t_phase == READ_DATA) {
            std::string zip_entry_name = dir + "data/" + std::string(text_str);
            exist_in_zip(zip, zip_entry_name.c_str(), tensor);
            tensor.t_phase = tensor.data_size > 0 ? CHECK_SIZE : READ_NAME;
        }
        if(!read_global_type && tensor.t_phase == READ_NAME) {
            tensor.name = text_str;
            tensor.t_phase = READ_DATA;
            tensor.dtype = global_type;
        }
    }
}

void read_pkl_props(uint8_t*  buffer, zip_t *zip, std::string dir,std::unordered_map<std::string, Tensor> & tensors, convert_params & params) {
    if(buffer[0] == 0x80) { // proto
        if(buffer[1] != 2) {
            printf("Unsupported protocol\n");
            return;
        }
        buffer += 2; // 0x80 and version
        char string_buffer [MAX_STRING_BUFFER];
        bool finish = false;
        Tensor tensor = Tensor{"", 0, GGML_TYPE_F32, 0, {1, 1, 1, 1}, 0, READ_NAME, 0};
        // read pickle binary file
        while(!finish) {
            uint8_t type = *buffer;
            buffer++;
            switch (type)
            {
            case '}':
                break;
            case ']':
                break;
            // skip unused sections
            case 'h':
            case 'q':
            case 'Q': // 0
                buffer++;
                break;
            case 'r':
                buffer += 4;
                break;
            case 0x95:
                buffer += 8;
            break;
            case 0x94:
            break;
            case '(':
                break;
            case 'K':
                {
                    uint8_t value =  *buffer;
                    if(set_pkl_tensor_props(value, tensor)) {
                        buffer++;
                    }
                    buffer++;
                }
                break;
             case 'M':
                {
                    uint16_t value = read_short(buffer);
                    if(set_pkl_tensor_props(value, tensor)) {
                        buffer++;
                    }
                    buffer += 2;
                }
                break;
            case 'J':
                {
                    const int32_t value = read_int(buffer);
                    if(set_pkl_tensor_props(value, tensor)) {
                        buffer++; // skip tuple after read num_elements
                    }
                    buffer += 4;
                }
                break;
            case 'X':
                {
                    const int32_t len = read_int(buffer);
                    buffer += 4;
                    memset(string_buffer, 0, MAX_STRING_BUFFER);
                    if(len > MAX_STRING_BUFFER) {
                        printf("Tensor name very large\n");
                    }
                    memcpy(string_buffer, buffer,len < MAX_STRING_BUFFER ? len : (MAX_STRING_BUFFER - 1));
                    buffer += len;
                    read_pkl_string(string_buffer, zip, dir, tensor);
                    if(params.verbose) {
                        printf("pickle str: %s\n", string_buffer);
                    }
                }
                break;
            case 0x8C:
                {
                    const int8_t len = *buffer;
                    buffer ++;
                    memset(string_buffer, 0, MAX_STRING_BUFFER);
                    memcpy(string_buffer, buffer, len);
                    buffer += len;
                    //printf("String: '%s'\n", string_buffer);
                }
                break;
            case 'c':
                {
                    int8_t len = findChar(buffer, '\n');
                    buffer += len + 1;
                    len = findChar(buffer, '\n');
                    memset(string_buffer, 0, MAX_STRING_BUFFER);
                    memcpy(string_buffer, buffer, len);
                    buffer += len + 1;
                    read_pkl_data_type(string_buffer, tensor);
                    //printf("Global: %s\n", string_buffer);

                }
                break;
            case 0x86: // tuple 2
            case 0x85: // tuple 1
            case 't':
                if(tensor.t_phase == READ_DIMENS) {
                    if(!is_unused_tensor(tensor.name)) { // ignore unused tensors
                        tensors[tensor.name] = tensor;
                    }
                    // reset
                    tensor = Tensor{"", 0, GGML_TYPE_F32, 0, {1, 1, 1, 1}, 0, READ_NAME, 0};
                }
                break;
            case '.':
                finish = true;
                break;
            default:
                break;
            }
        }
        printf("Num Tensors readed: %i\n", tensors.size());
    }
}

void read_vocab_json(std::map<int, std::string> & vocab_map, convert_params params) {
    char* vocab_buffer = NULL;
    if(!params.vocab_path.empty()) {
        FILE* fv = std::fopen(params.vocab_path.c_str(), "r");
        if(fv == NULL) {
            printf("Error: failed to open vocab file '%s'\n", params.vocab_path.c_str());
            exit(0);
            return;
        }
        fseek(fv, 0, SEEK_END);
        size_t file_size = ftell(fv);
        // return to begin
        fseek(fv, 0, SEEK_SET);
        vocab_buffer = (char*)malloc(file_size);
        fread(vocab_buffer, 1, file_size, fv);
        fclose(fv);
    } else {
        // read embedded vocab
        printf("using embedded vocab\n");
        vocab_buffer = reinterpret_cast<char*>(vocab_json);
    }
    json vocab = json::parse(vocab_buffer);
    std::map<char, int> decoder = unicode_to_byte();
    for(auto& it : vocab.items()) {
        std::string token_str = it.key();
        std::string result = "";
        int id = it.value();
        for(char c : token_str) {
            result += decoder[c];
        }
        vocab_map[id] = result;
    }
}

/*

    PREPROCESS TENSORS

*/

std::string replace_name_by_map(const std::string full_name, std::unordered_map<std::string,std::string> ft_map) {
    std::string result = full_name;
    for(auto it : ft_map) {
        size_t pos = result.find(it.first);
        if(pos != std::string::npos) {
            result = result.replace(pos, it.first.size(), it.second);
        }
    }
    return result;
}

 // hugging face pipeline to legacy stable diffusion
std::unordered_map<std::string, std::string> unet_convert_map;
std::unordered_map<std::string, std::string> unet_convert_map_resnet;
std::unordered_map<std::string, std::string> unet_convert_map_layers;
std::unordered_map<std::string, std::string> vae_convert_map;
std::unordered_map<std::string, std::string> clip_convert_map;
std::unordered_map<std::string, std::string> lora_fix_map;
std::unordered_map<std::string, std::string> shorten_name_map;

std::string convert_unet_to_original(std::string name, convert_params params) {
    bool resnet_tensor = name.find("resnets") != std::string::npos;
    std::string separator = params.lora ? "." : "_";
    if(unet_convert_map.empty()) {
        unet_convert_map["time" + separator + "embedding.linear" + separator + "1.weight"] = "time_embed.0.weight";
        unet_convert_map["time" + separator + "embedding.linear" + separator + "1.bias"] = "time_embed.0.bias";
        unet_convert_map["time" + separator + "embedding.linear" + separator + "2.weight"] = "time_embed.2.weight";
        unet_convert_map["time" + separator + "embedding.linear" + separator + "2.bias"] = "time_embed.2.bias";
        unet_convert_map["conv" + separator + "in.weight"] = "input_blocks.0.0.weight";
        unet_convert_map["conv" + separator + "in.bias"] = "input_blocks.0.0.bias";
        unet_convert_map["conv" + separator + "norm" + separator + "out.weight"] ="out.0.weight";
        unet_convert_map["conv" + separator + "norm" + separator + "out.bias"] = "out.0.bias";
        unet_convert_map["conv" + separator + "out.weight"] = "out.2.weight";
        unet_convert_map["conv" + separator + "out.bias"] = "out.2.bias";
    }

    // resnet
    if(unet_convert_map_resnet.empty() && resnet_tensor) {
        unet_convert_map_resnet["norm1"] = "in_layers.0";
        unet_convert_map_resnet["conv1"] = "in_layers.2";
        unet_convert_map_resnet["norm2"] = "out_layers.0";
        unet_convert_map_resnet["conv2"] = "out_layers.2";
        unet_convert_map_resnet["time" + separator + "emb" + separator + "proj"] = "emb_layers.1";
        unet_convert_map_resnet["conv" + separator + "shortcut"] = "skip_connection";
    }

    if(unet_convert_map_layers.empty()) {
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                unet_convert_map_layers["down" + separator + "blocks." + std::to_string(i) + ".resnets." + std::to_string(j) + "."] = "input_blocks." + std::to_string(3 * i + j + 1) + ".0.";
                if (i < 3)
                {
                    unet_convert_map_layers["down" + separator + "blocks." + std::to_string(i) + ".attentions." + std::to_string(j) + "."] = "input_blocks." + std::to_string(3 * i + j + 1) + ".1.";
                }
            }
            for (int j = 0; j < 3; j++)
            {
                unet_convert_map_layers["up" + separator + "blocks." + std::to_string(i) + ".resnets." + std::to_string(j) + "."] = "output_blocks." + std::to_string(3 * i + j) + ".0.";
                if (i > 0)
                {
                    unet_convert_map_layers["up" + separator + "blocks." + std::to_string(i) + ".attentions." + std::to_string(j) + "."] = "output_blocks." + std::to_string(3 * i + j) + ".1.";
                }
            }
            if (i < 3)
            {
                unet_convert_map_layers["down" + separator + "blocks." + std::to_string(i) + ".downsamplers.0.conv."] = "input_blocks." + std::to_string(3 * (i + 1)) + ".0.op.";
                unet_convert_map_layers["up" + separator + "blocks." + std::to_string(i) + ".upsamplers.0."] = "output_blocks." + std::to_string(3 * i + 2) + "." + std::to_string(i + 2) + ".";
            }
        }
        unet_convert_map_layers["mid" + separator + "block.attentions.0."] = "middle_block.1.";
        for (int j = 0; j < 2; j++)
        {
            unet_convert_map_layers["mid" + separator + "block.resnets." + std::to_string(j) + "."] = "middle_block." + std::to_string(2 * j) + ".";
        }
    }
    
    std::string result = replace_name_by_map(name, unet_convert_map);
    result = replace_name_by_map(result, unet_convert_map_layers);
    if(resnet_tensor) {
        result = replace_name_by_map(result, unet_convert_map_resnet);
    }
    return result;
}

std::string convert_vae_to_original(std::string name, convert_params params) {
    std::unordered_map<std::string, std::string> vae_map;
    bool hf_attention = name.find("attentions") != std::string::npos;
    if(vae_convert_map.empty()) {
        vae_convert_map[ "conv_shortcut"] = "nin_shortcut";
        vae_convert_map[ "conv_norm_out"] = "norm_out";
        vae_convert_map[ "mid_block.attentions.0."] = "mid.attn_1.";
        for(int i = 0; i < 4; i++) {
            for(int j = 0; j < 2; j++) {
                vae_convert_map["encoder.down_blocks."+std::to_string(i)+".resnets."+std::to_string(j)+"."] = "encoder.down."+std::to_string(i)+".block."+std::to_string(j)+".";
            }
            if(i < 2) {
                vae_convert_map["mid_block.resnets."+std::to_string(i)+"."] = "mid.block_"+std::to_string(i + 1)+".";
            }
            if(i < 3) {
                vae_convert_map["down_blocks."+std::to_string(i)+".downsamplers.0."] = "down."+std::to_string(i)+".downsample.";
                vae_convert_map["up_blocks."+std::to_string(i)+".upsamplers.0."] = "up."+std::to_string(3 - i)+".upsample.";
            }
            for(int j = 0; j < 3; j++) {
                vae_convert_map["decoder.up_blocks."+std::to_string(i)+".resnets."+std::to_string(j)+"."] = "decoder.up."+std::to_string(3 - i)+".block."+std::to_string(j)+".";
            }
        }
    }

    if(hf_attention || params.version == VERSION_2_x) {
        vae_convert_map["to_k."] = "k.";
        vae_convert_map["to_q."] = "q.";
        vae_convert_map["to_v."] = "v.";
        vae_convert_map["to_out.0."] = "proj_out.";
    }

    if(hf_attention) {
        vae_convert_map["key."] = "k.";
        vae_convert_map["query."] = "q.";
        vae_convert_map["value."] = "v.";
        vae_convert_map["group_norm."] = "norm.";
        vae_convert_map["proj_attn."] = "proj_out.";
    }

    return replace_name_by_map(name, vae_convert_map);
}

std::string convert_clip_to_hf_clip(std::string name, convert_params params) {
    std::string separator = params.lora ? "." : "_";
    if(clip_convert_map.empty()) {
        if(params.version == VERSION_2_x) {
            clip_convert_map[".model."] = ".transformer.text_model.";
            clip_convert_map["transformer.resblocks."] = "encoder.layers.";
            clip_convert_map["attn.out_proj"] = "self_attn.out_proj";
            clip_convert_map["ln_final."] = "final_layer_norm.";
            clip_convert_map["token_embedding.weight"] =
                "embeddings.token_embedding.weight";
            clip_convert_map["positional_embedding"] =
                "embeddings.position_embedding.weight";
        } else {
            clip_convert_map["resblocks."] = "text_model.encoder.layers.";
            clip_convert_map[".attn."] = ".self_attn.";
            clip_convert_map["ln_final."] = "transformer.text_model.final_layer_norm.";
            if(name == "token_embedding.weight") {
                return "transformer.text_model.embeddings.token_embedding.weight";
            } else if(name == "positional_embedding") {
                return "transformer.text_model.embeddings.position_embedding.weight";
            }
        }
        clip_convert_map["ln_1."] = "layer_norm1.";
        clip_convert_map["ln_2."] = "layer_norm2.";
        clip_convert_map[".c_fc."] = ".fc1.";
        clip_convert_map[".c_proj."] = ".fc2.";
    }
    if(params.lora) {
        clip_convert_map["te.text.model"] = "cond_stage_model.transformer.text_model";
    }
    // SD XL to SD normal
    if(params.version == VERSION_XL) {
        clip_convert_map["conditioner.embedders.0.transformer.text_model"] = "cond_stage_model.transformer.text_model";
        clip_convert_map["conditioner.embedders.1.model"] = "cond_stage_model.g.transformer.text_model";
    }
    return replace_name_by_map(name, clip_convert_map);
}

std::string fix_lora_names(std::string name) {
    // lora fix names
    if(lora_fix_map.empty()) {
        lora_fix_map["self.attn"] = "self_attn";
        lora_fix_map["proj.in"] = "proj_in";
        lora_fix_map["proj.out"] = "proj_out";
        lora_fix_map["out.proj"] = "out_proj";
        lora_fix_map["transformer.blocks"] = "transformer_blocks";
        lora_fix_map["q.proj"] = "q_proj";
        lora_fix_map["k.proj"] = "k_proj";
        lora_fix_map["v.proj"] = "v_proj";
        lora_fix_map["to.q"] = "to_q";
        lora_fix_map["to.k"] = "to_k";
        lora_fix_map["to.v"] = "to_v";
        lora_fix_map[".to.out"] = ".to_out";
    }
    return replace_name_by_map(name, lora_fix_map);
}

std::string shorten_name(std::string name, convert_params params) {
    if(shorten_name_map.empty()) {
        shorten_name_map["model.diffusion_model"] = "unet";
        shorten_name_map["first_stage_model"] = "vae";
        shorten_name_map["cond_stage_model"] = "clip";

        // transformer blocks
        shorten_name_map["transformer_blocks"] = "t_blks";
        shorten_name_map["output_blocks"] = "out_blks";
        shorten_name_map["input_blocks"] = "in_blks";
        shorten_name_map["middle_block"] = "mddl_blk";

        // clip
        shorten_name_map["transformer.text_model"] = "txt_mdl";

        // vae
        shorten_name_map["encoder"] = "enc";
        shorten_name_map["decoder"] = "dec";

        if(params.lora) {
            // Lora
            shorten_name_map["lora"] = "lr";
            shorten_name_map["lora.up"] = "lu";
            shorten_name_map["lora.down"] = "ld";
        }
    }
    return replace_name_by_map(name, shorten_name_map);
}

void* fetch_data(Tensor tensor, zip_t* zip, FILE* fp) {
    if(!tensor.data) { // fetch tensor data from zip (.ckpt) or file stream (.safetensors)
        if(zip) {
            zip_entry_openbyindex(zip, tensor.data_offset);
            size_t buf_sz;
            if(zip_entry_read(zip, &tensor.data, &buf_sz) < 0) {
                return NULL;
            }
        } else {
#ifdef _WIN32
            _fseeki64(fp, (__int64) tensor.data_offset, SEEK_SET);
#else
            std::fseek(fp, (long) tensor.data_offset, SEEK_SET);
#endif
            tensor.data = malloc(tensor.data_size);
            std::fread(tensor.data, 1, tensor.data_size, fp);
        }
    }
    return tensor.data;
}

std::tuple<Tensor, Tensor, Tensor> split_qkv_tensor(Tensor qkv_tensor, zip_t* zip, FILE* fp) {
    const int ne0 = qkv_tensor.shape[0] / 3; // split in 3 tensors: query, key, value
    const int ne1 = qkv_tensor.shape[1];
    const int32_t num_elements = ne0 * ne1;
    ggml_type dtype = qkv_tensor.dtype;
    const int n_dims = qkv_tensor.n_dims;

    size_t chunk_size = (size_t)num_elements * ggml_type_size(qkv_tensor.dtype);
    void* qkv_data = fetch_data(qkv_tensor, zip, fp);

    Tensor q = Tensor{"", 0, dtype, chunk_size, {ne0, ne1, 1, 1}, n_dims, READ_NAME, num_elements, true}; // query
    Tensor k = Tensor{"", 0, dtype, chunk_size, {ne0, ne1, 1, 1}, n_dims, READ_NAME, num_elements, true}; // key
    Tensor v = Tensor{"", 0, dtype, chunk_size, {ne0, ne1, 1, 1}, n_dims, READ_NAME, num_elements, true}; // value

    // make a view of original tensor data
    q.data = qkv_data;
    k.data = ((char*)qkv_data) + chunk_size;
    v.data = ((char*)qkv_data) + chunk_size * 2;
    return {q, k, v};
}

void preprocess_tensors(
    std::unordered_map<std::string, Tensor> & src,
    std::vector<Tensor> & dst,
    zip_t * zip, FILE * fp, convert_params & params) {
    printf("preprocessing %zu tensors\n", src.size());

    for(auto & it : src) {
        std::string name = it.first;
        Tensor tensor = it.second;

        if(name.find("first_stage_model.") == 0 || params.vae) {
            tensor.name = convert_vae_to_original(name, params);

            // convert vae attn block linear to conv2d 1x1
            if(params.vae && name.find("first_stage_model.") == std::string::npos) {
                tensor.name = "first_stage_model." + name;
            }

            if(tensor.name.find("attn_1") != std::string::npos) {
                if(tensor.n_dims == 2) {
                    tensor.n_dims += 2;
                    if(params.verbose) {
                        printf("linear to conv2d %s\n", tensor.name.c_str());
                    }
                }
            }
        }

        if(name.find("cond_stage_model.") == 0 ||
            name.find("conditioner.") == 0  ||
            params.lora && name.find("text.model.") != std::string::npos) {
            tensor.name = convert_clip_to_hf_clip(name, params);
            if(params.version == VERSION_2_x) {
                std::string preffix = shorten_name(tensor.name, params);
                size_t fw = preffix.find("attn.in_proj_weight");
                size_t fb = preffix.find("attn.in_proj_bias");
                if(fw != std::string::npos) {
                    Tensor q, k, v;
                    std::tie(q, k, v) = split_qkv_tensor(tensor, zip, fp);
                    for(int i = 0; i< 3;i++) {
                        Tensor attn_t = i == 0 ? q : (i == 1 ? k : v);
                        attn_t.name = preffix.substr(0, fw) + kqv_self[i];
                        dst.push_back(attn_t);
                        if(params.verbose) {
                            printf("split %s => %s\n", it.first.c_str(), attn_t.name.c_str());
                        }
                    }
                    continue;
                } else if(fb != std::string::npos) {
                    Tensor q, k, v;
                    std::tie(q, k, v) = split_qkv_tensor(tensor, zip, fp);
                    for(int i = 0; i< 3;i++) {
                        Tensor attn_t = i == 0 ? q : (i == 1 ? k : v);
                        attn_t.name = preffix.substr(0, fb) + kqv_self[i + 3];
                        dst.push_back(attn_t);
                        if(params.verbose) {
                            printf("split %s => %s\n", it.first.c_str(), attn_t.name.c_str());
                        }
                    }
                    continue;
                }
            }
        } else if(name.find("model.diffusion_model.") == 0 || params.lora) {
            tensor.name = convert_unet_to_original(name, params);
            if(tensor.name.find("proj_in.weight") != std::string::npos ||
                tensor.name.find("proj_out.weight") != std::string::npos) {
                if(tensor.n_dims == 2) {
                    tensor.n_dims += 2;
                    if(params.verbose) {
                        printf("linear to conv2d %s\n", tensor.name.c_str());
                    }
                }
            }
        }

        if(params.lora) {
            tensor.name = fix_lora_names(tensor.name);
        }

        if(is_unused_tensor(tensor.name)) { // discard tensors
            continue;
        }

        tensor.name = shorten_name(tensor.name, params);
        if(params.lora) {
            int pos = name.find("lora.down");
            if(pos != std::string::npos) {
                std::string key = name.substr(0, pos) + "alpha";
                if(params.lora_alphas.find(key) != params.lora_alphas.end()) {
                    int kpos = tensor.name.find("ld.");
                    std::string target = tensor.name.substr(0, kpos) + "alpha";
                    params.lora_alphas[target] = params.lora_alphas[key];
                    params.lora_alphas.erase(key);
                } else {
                    printf("WARNING: missing alpha '%s'\n", key.c_str());
                }
            }
        }
        dst.push_back(tensor);
    }
}

void *convert_tensor(void * source, Tensor tensor, ggml_type dst_type) {
    if(tensor.dtype == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
        ggml_fp16_t* dest = (ggml_fp16_t*)malloc(tensor.num_elements * sizeof(ggml_fp16_t));
        ggml_fp32_to_fp16_row((float*)source, dest, tensor.num_elements);
        return dest;
    } else if(tensor.dtype == GGML_TYPE_F16 && dst_type == GGML_TYPE_F32) {
        float* dest = (float*)malloc(tensor.num_elements * sizeof(float));
        ggml_fp16_to_fp32_row((ggml_fp16_t*)source, dest, tensor.num_elements);
        return dest;
    } else if(
        dst_type == GGML_TYPE_Q4_0 ||
        dst_type == GGML_TYPE_Q4_1 ||
        dst_type == GGML_TYPE_Q5_0 ||
        dst_type == GGML_TYPE_Q5_1 ||
        dst_type == GGML_TYPE_Q8_0) {
        // in development
        int num_blocks = tensor.shape[0] * tensor.shape[1] / ggml_blck_size(dst_type);
        float* src = nullptr;
        if(tensor.dtype == GGML_TYPE_F16) {
            src = (float*)malloc(tensor.num_elements * sizeof(float));
            ggml_fp16_to_fp32_row((ggml_fp16_t*)source, src, tensor.num_elements);
        } else {
            src = (float*)source;
        }
        int64_t* hist = new int64_t[16];
        void* quantized = malloc(ggml_type_size(dst_type) * num_blocks);
        ggml_quantize_chunk(dst_type, src, quantized, 0, tensor.num_elements, hist);
        if(tensor.dtype == GGML_TYPE_F16) {
            free(src);
        }
        delete[] hist;
        return quantized;
    } else {
        throw std::invalid_argument("unsupported conversion");
    }
    return NULL;
}

void convert_to_gguf(std::unordered_map<std::string, Tensor> & model_tensors, zip_t * zip, FILE * fp, convert_params & params) {
    if(!params.vae &&
        model_tensors.find("first_stage_model.post_quant_conv.bias") == model_tensors.end() && // is not a stable diffusion model
        model_tensors.find("post_quant_conv.bias") != model_tensors.end()) { // has a tensor of VAE
        params.vae = true;
        printf("VAE detected\n");
    }

    if(!params.lora && model_tensors.find("alphas_cumprod") == model_tensors.end()) {
        params.generate_alphas_cumprod = true;
    }

    std::vector<Tensor> processed_tensors;

    if(!params.lora) {
        if(model_tensors.find("cond_stage_model.model.token_embedding.weight") != model_tensors.end()) {
            params.version = VERSION_2_x;
            printf("Stable Diffusion 2.x - %s\n", params.model_name.c_str());
        } else if(model_tensors.find("conditioner.embedders.0.transformer.text_model.embeddings.position_embedding.weight") != model_tensors.end()) {
            params.version = VERSION_XL;
            printf("Stable Diffusion XL - %s\n", params.model_name.c_str());
        } else {
            printf("Stable Diffusion 1.x - %s\n", params.model_name.c_str());
        }
    }

    preprocess_tensors(model_tensors, processed_tensors, zip, fp, params);

    gguf_context* g_ctx = gguf_init_empty();

    if(params.lora) {
        gguf_set_val_str(g_ctx, "sd.lora.name", params.model_name.c_str());
        gguf_set_val_i32(g_ctx, "sd.lora.dtype", (int)params.out_type);
        gguf_set_val_i32(g_ctx, "sd.lora.type", (int)params.lora_type);

        // process alphas
        std::vector<float> alpha_values;
        std::vector<const char*> alpha_key;

        for(auto k : params.lora_alphas) {
            alpha_key.push_back(k.first.c_str());
            alpha_values.push_back(k.second);
        }

        printf("writing %zu lora alphas\n", alpha_key.size());
        gguf_set_arr_str(g_ctx, "sd.lora.alphas_k", alpha_key.data(), alpha_key.size());
        gguf_set_arr_data(g_ctx, "sd.lora.alphas_v", GGUF_TYPE_FLOAT32, alpha_values.data(), alpha_values.size());
    } else if(params.vae) {
        gguf_set_val_str(g_ctx, "sd.vae.name", params.model_name.c_str());
        gguf_set_val_i32(g_ctx, "sd.vae.dtype", (int)params.out_type);
        gguf_set_val_i32(g_ctx, "sd.vae.type", (int)params.lora_type);
    } else {
        // process vocab
        std::map<int, std::string> vocab_map;
        std::vector<const char*> vocab_data;
        read_vocab_json(vocab_map, params);

        for(int i = 0; i < vocab_map.size(); i++) {
            vocab_data.push_back(vocab_map[i].c_str());
        }

        gguf_set_val_str(g_ctx, "sd.model.name", params.model_name.c_str());
        gguf_set_val_i32(g_ctx, "sd.model.dtype", (int)params.out_type);
        gguf_set_val_i8(g_ctx, "sd.model.version", (int)params.version);

        // write vocab
        if(params.verbose) {
            printf("writing vocab: %zu tokens\n", vocab_data.size());
        }
        gguf_set_arr_str(g_ctx, "sd.vocab.tokens", vocab_data.data(), vocab_data.size());
    }
    printf("converting %zu tensors\n", processed_tensors.size());
    // write tensors
    ggml_context* ctx = ggml_init({ (processed_tensors.size() + (params.generate_alphas_cumprod ? 1 : 0)) * ggml_tensor_overhead(), NULL, true }); // no alloc data
    int num_clip_tensors = 0, num_unet_tensors = 0, num_vae_tensors = 0;
    size_t total_org_model = 0, total_conv_model = 0;
    for(Tensor & tensor : processed_tensors) {
        if(tensor.name.size() >= 64) {
            printf("Error: tensor name very large '%s', might not be supported anyway by stable-diffusion.cpp\n", tensor.name.c_str());
            exit(0);
            return;
        }
        if(tensor.name.find("clip.") != std::string::npos) {
             num_clip_tensors++;
        } else if(tensor.name.find("unet.") != std::string::npos) {
            num_unet_tensors++;
        } else if(tensor.name.find("vae.") != std::string::npos) {
            num_vae_tensors++;
        }
        ggml_type dest_type = GGML_TYPE_F32;
        if(tensor.name.find(".weight") && tensor.n_dims == 2) { // allow quantize only weights
            dest_type = params.out_type;
        } else if(tensor.n_dims == 4) {
            dest_type = GGML_TYPE_F16;
        }
        ggml_tensor* gg_tensor = ggml_new_tensor(ctx, dest_type, tensor.n_dims, tensor.inverse_shape());
        ggml_set_name(gg_tensor, tensor.name.c_str());
        void* source = fetch_data(tensor, zip, fp);
        void* dest = NULL;
        if(params.verbose) {
            printf("converting: %s | %s => %s\n", tensor.name.c_str(), ggml_type_name(tensor.dtype), ggml_type_name(dest_type));
        }
        if(tensor.dtype == dest_type) {
            dest = source;
        } else {
            // convert
            dest = convert_tensor(source, tensor, dest_type);
            if(!tensor.is_view) {
                free(source);
            }
        }
        gguf_add_tensor(g_ctx, gg_tensor);
        gguf_set_tensor_data(g_ctx, tensor.name.c_str(), dest, ggml_nbytes(gg_tensor));
        total_org_model += tensor.data_size;
        total_conv_model += ggml_nbytes(gg_tensor);
    }
    if(params.generate_alphas_cumprod) {
        ggml_tensor* gg_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, TIMESTEPS);
        ggml_set_name(gg_tensor, "alphas_cumprod");
        gguf_add_tensor(g_ctx, gg_tensor);
        float* dest = calculate_alpha_cumprod();
        gguf_set_tensor_data(g_ctx, "alphas_cumprod", dest, ggml_nbytes(gg_tensor));
        printf("alphas_cumprod computed\n");
    }
    printf("\nCLIP Model Tensor count: %i\nUNET Model Tensor count: %i\nVAE Model Tensor count: %i\n\nsaving gguf file\n",
        num_clip_tensors,
        num_unet_tensors,
        num_vae_tensors);
    gguf_write_to_file(g_ctx, params.output_path.c_str(), false);
    printf("model saved '%s' correctly.", params.output_path.c_str());
    ggml_free(ctx);
    gguf_free(g_ctx);
}

void convert_checkpoint_file(const char * file_name, convert_params params) {
    struct zip_t *zip = zip_open(file_name, 0, 'r');
    std::unordered_map<std::string, Tensor> tensors;
    {
        int i, n = zip_entries_total(zip);
        for (i = 0; i < n; ++i) {
            zip_entry_openbyindex(zip, i);
            {
                std::string name = zip_entry_name(zip);
                int isdir = zip_entry_isdir(zip);
                unsigned long long size = zip_entry_size(zip);
                unsigned int crc32 = zip_entry_crc32(zip);
                size_t res = name.find( "data.pkl");
                if(res != std::string::npos) {
                    std::string dir_ = name.substr(0, res);
                    void* pkl_data = NULL;
                    size_t pkl_size;
                    zip_entry_read(zip, &pkl_data, &pkl_size);
                    printf("reading tensors metadata\n");
                    read_pkl_props((uint8_t*)pkl_data, zip, dir_, tensors, params);
                }
            }
            zip_entry_close(zip);
        }
        // convert tensors
        if(tensors.size() > 0) {
            convert_to_gguf(tensors, zip, NULL, params);
        } else {
            printf("Error: failed to load tensors\n");
        }
    }
    zip_close(zip);
}

void convert_safetensor_file(FILE * f, int64_t metadata_size, convert_params params) {
    std::fseek(f, 8, SEEK_SET); // from begin

    char* metadata_buffer = new char[metadata_size + 1];
    memset(metadata_buffer, 0, metadata_size + 1);
    std::fread(metadata_buffer, 1, metadata_size, f);
    json sf_mt = json::parse(metadata_buffer);
    std::unordered_map<std::string, Tensor> tensors;

    printf("reading safetensors metadata\n");
    int data_begin = 8 + metadata_size;
    for (json::iterator it = sf_mt.begin(); it != sf_mt.end(); ++it) {
        std::string tensor_name = it.key();
        json tensor_props = it.value();

        // auto detect lora
        if(!params.lora) {
            if(tensor_name == "__metadata__" && tensor_props.contains("ss_network_module")) {
                params.lora = tensor_props["ss_network_module"] == "networks.lora";
                if(params.lora) {
                    printf("LoRA detected\n");
                }
            }
        }

        if(tensor_props.contains("dtype") && !is_unused_tensor(tensor_name)) { // check if there dtype param
            int n_dims = tensor_props["shape"].size();
            std::string dtype = tensor_props["dtype"];
            size_t start_data = tensor_props["data_offsets"][0].get<size_t>();
            size_t end_data = tensor_props["data_offsets"][1].get<size_t>();

            if(params.lora) {
                if(params.lora_type == LORA_NONE) {
                    if(tensor_name.find("lora_up.weight") != std::string::npos) {
                        params.lora_type = LORA_REGULAR;
                    } else if(tensor_name.find("lora.up.weight") != std::string::npos) {
                        params.lora_type = LORA_DIFFUSERS;
                    } else if(tensor_name.find("lora_linear_layer.up.weight") != std::string::npos) {
                        params.lora_type = LORA_TRANSFORMERS;
                    }
                }
                // replace all '_' to '.'
                for (char &c : tensor_name) { if (c == '_') { c = '.'; } }
            }

            // collect alphas
            if(params.lora &&
                n_dims == 0 &&
                tensor_name.find(".alpha") != std::string::npos) {
                    std::fseek(f, data_begin + start_data, SEEK_SET);
                    if(dtype == "F16") {
                        ggml_fp16_t val;
                        std::fread(&val, 1, sizeof(val), f);
                        params.lora_alphas[tensor_name] = ggml_fp16_to_fp32(val);
                    } else if(dtype == "F32") {
                        float val;
                        std::fread(&val, 1, sizeof(val), f);
                        params.lora_alphas[tensor_name] = val;
                    }
                    continue;
            }

            Tensor tensor = Tensor{tensor_name.c_str(), 0, GGML_TYPE_F32, 0, {1, 1, 1, 1}, n_dims, READ_NAME, 0};
            if(dtype == "F16") {
                tensor.dtype = GGML_TYPE_F16;
            } else if(dtype != "F32") {
                printf("unsupported model data type: %s", dtype.c_str());
                return;
            }

            for(uint8_t i = 0;i < n_dims; i++) {
                tensor.shape[i] = tensor_props["shape"][i];
            }

            tensor.data_size = end_data - start_data;
            tensor.num_elements = tensor.data_size / ggml_type_size(tensor.dtype);
            tensor.data_offset = data_begin + start_data;
            tensors[tensor_name] = tensor;
        }
    }

    if(tensors.size() > 0) {
        convert_to_gguf(tensors, NULL, f, params);
    } else {
        printf("Error: failed to load tensors\n");
    }
}

void convert_model(convert_params & params) {
    // check if the model is safetensor or pytorch checkpoint
    FILE* fp = std::fopen(params.model_path.c_str(), "rb");
    if(!fp) {
        printf("Fail to open file: %s", params.model_path.c_str());
        return;
    }
    std::fseek(fp, 0, SEEK_END);
    size_t file_size = ftell(fp);
    // return to begin
    std::fseek(fp, 0, SEEK_SET);
    // read first 9 bytes
    uint8_t buffer_[9];
    std::fread(buffer_, 1, 9, fp);
    int64_t safe_tensor_metadata_size = read_long(buffer_);
    bool safe_tensor = false;
    if(
        buffer_[8] == '{' &&
        safe_tensor_metadata_size > 0 &&
        safe_tensor_metadata_size < file_size) { // begin safetensor metadata
        size_t offset = safe_tensor_metadata_size + /* long */ 8L - 1L;
#ifdef _WIN32
        _fseeki64(fp, (__int64) offset, SEEK_SET);
#else
        std::fseek(fp, (long) offset, SEEK_SET);
#endif
        std::fread(buffer_, 1, 1, fp);
        safe_tensor = buffer_[0] == '}' || buffer_[0] == ' ';
    } else {
        std::fclose(fp);
    }
    size_t last = params.model_path.find_last_of("/\\");
    params.model_name = params.model_path.substr(last + 1);
    if(params.output_path.empty()) {
        last = params.model_name.find_last_of(".");
        params.output_path = params.model_name.substr(0, last) + "-" + ggml_type_name(params.out_type) + ".gguf";
    }
    printf("model type: %s\n", safe_tensor ? "safetensors" : "checkpoint");
    if(safe_tensor) {
        convert_safetensor_file(fp, safe_tensor_metadata_size, params);
    } else {
        convert_checkpoint_file(params.model_path.c_str(), params);
    }
}

void print_usage(int argc, const char* argv[]) {
    printf("usage: %s [MODEL_PATH] --type [OUT_TYPE] [arguments]\n", argv[0]);
    printf("Model supported for conversion: .safetensors models or .ckpt checkpoints models\n");
    printf("\n");
    printf("arguments:\n");
    printf("  -h, --help                         show this help message and exit\n");
    printf("  -o, --out [FILENAME]               path or name to converted model\n");
    printf("  --vocab [FILENAME]                 path to custom vocab.json (usually unnecessary)\n");
    printf("  -v, --verbose                      print processing info - dev info\n");
    printf("  -l, --lora                         force read the model as a LoRA\n");
    printf("  -tp, --type [OUT_TYPE]             output format (f32, f16, q4_0, q4_1, q5_0, q5_1, q8_0)\n");
}

bool parse_params(int argc, const char* argv[], convert_params & params) {
    params.model_path = argv[1];
    for(int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if(arg == "-o" || arg == "--out") {
            if (++i >= argc) {
                break;
            }
            params.output_path = argv[i];
            if(params.output_path.find(".gguf") == std::string::npos) {
                params.output_path = params.output_path + ".gguf";
            }
        } else if(arg == "--vocab") {
            if (++i >= argc) {
                break;
            }
            params.vocab_path = argv[i];
        } else if(arg == "-l" || arg == "--lora") {
            params.lora = true;
        } else if(arg == "-v" || arg == "--verbose") {
            params.verbose = true;
        } else if(arg == "--type" || arg == "-tp") {
            if (++i >= argc) {
                printf("specify the output format\n");
                exit(1);
            }
            std::string fmt_select = argv[i];
            if(fmt_select == "f32") {
                params.out_type = GGML_TYPE_F32;
            } else if(fmt_select == "f16") {
                params.out_type = GGML_TYPE_F16;
            } else if(fmt_select == "q4_0") {
                params.out_type = GGML_TYPE_Q4_0;
            } else if(fmt_select == "q4_1") {
                params.out_type = GGML_TYPE_Q4_1;
            } else if(fmt_select == "q5_0") {
                params.out_type = GGML_TYPE_Q5_0;
            } else if(fmt_select == "q5_1") {
                params.out_type = GGML_TYPE_Q5_1;
            } else if(fmt_select == "q8_0") {
                params.out_type = GGML_TYPE_Q8_0;
            } else {
                fprintf(stderr, "error: invalid output format %s, must be one of [f32, f16, q4_0, q4_1, q5_0, q5_1, q8_0]\n",
                        fmt_select.c_str());
                exit(1);
            }
        } else if(arg == "-h" || arg == "--help") {
            print_usage(argc, argv);
            return false;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            print_usage(argc, argv);
            exit(1);
        }
    }
    if(params.model_path.empty()) {
        fprintf(stderr, "error: missing model input path\n");
        print_usage(argc, argv);
        exit(1);
    }
    return true;
}

// support safetensors and ckpt (pikle)

int main(int argc, const char* argv[]) {
    convert_params params;
    if(argc > 2) {
        // needed to initialize f16 tables
        {
            struct ggml_init_params params = { 0, NULL, false };
            struct ggml_context * ctx = ggml_init(params);
            ggml_free(ctx);
        }
        // parse params
        if(parse_params(argc, argv, params)) {
            convert_model(params);
        }
    } else {
        print_usage(argc, argv);
    }
}