#include <stdio.h>
#include <string.h>
#include <time.h>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "preprocessing.hpp"
#include "stable-diffusion.h"

#define STB_IMAGE_IMPLEMENTATION

#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC

#include "stb_image_write.h"

const char* rng_type_to_str[] = {
    "std_default",
    "cuda",
};

// Names of the sampler method, same order as enum sample_method in stable-diffusion.h
const char* sample_method_str[] = {
    "euler_a",
    "euler",
    "heun",
    "dpm2",
    "dpm++2s_a",
    "dpm++2m",
    "dpm++2mv2",
    "lcm",
};

// Names of the sigma schedule overrides, same order as sample_schedule in stable-diffusion.h
const char* schedule_str[] = {
    "default",
    "discrete",
    "karras",
};

const char* modes_str[] = {
    "txt2img",
    "img2img",
    "convert",
};

enum SDMode {
    TXT2IMG,
    IMG2IMG,
    CONVERT,
    STREAM,
    MODE_COUNT
};

struct SDParams {
    int n_threads = -1;
    SDMode mode   = TXT2IMG;

    std::string model_path;
    std::string vae_path;
    std::string clip_path;
    std::string unet_path;
    std::string taesd_path;
    std::string esrgan_path;
    std::string controlnet_path;
    std::string embeddings_path;
    sd_type_t wtype = SD_TYPE_COUNT;
    std::string lora_model_dir;
    std::string output_path = "output.png";
    std::string input_path;
    std::string control_image_path;

    std::string prompt;
    std::string negative_prompt;
    float cfg_scale = 7.0f;
    int clip_skip   = -1;  // <= 0 represents unspecified
    int width       = 512;
    int height      = 512;
    int batch_count = 1;

    sample_method_t sample_method = EULER_A;
    schedule_t schedule           = DEFAULT;
    int sample_steps              = 20;
    float strength                = 0.75f;
    float control_strength        = 0.9f;
    rng_type_t rng_type           = CUDA_RNG;
    int64_t seed                  = 42;
    bool verbose                  = false;
    bool vae_tiling               = false;
    bool vae_decode_only          = false;
    bool control_net_cpu          = false;
    bool canny_preprocess         = false;
};

static std::string sd_basename(const std::string& path) {
    size_t pos = path.find_last_of('/');
    if (pos != std::string::npos) {
        return path.substr(pos + 1);
    }
    pos = path.find_last_of('\\');
    if (pos != std::string::npos) {
        return path.substr(pos + 1);
    }
    return path;
}

void print_params(SDParams params) {
    printf("Option: \n");
    printf("    n_threads:         %d\n", params.n_threads);
    printf("    mode:              %s\n", modes_str[params.mode]);
    printf("    model_path:        %s\n", params.model_path.c_str());
    printf("    wtype:             %s\n", params.wtype < SD_TYPE_COUNT ? sd_type_name(params.wtype) : "unspecified");
    printf("    vae_path:          %s\n", params.vae_path.c_str());
    printf("    clip_path:         %s\n", params.clip_path.c_str());
    printf("    unet_path:         %s\n", params.unet_path.c_str());
    printf("    taesd_path:        %s\n", params.taesd_path.c_str());
    printf("    esrgan_path:       %s\n", params.esrgan_path.c_str());
    printf("    controlnet_path:   %s\n", params.controlnet_path.c_str());
    printf("    embeddings_path:   %s\n", params.embeddings_path.c_str());
    printf("    output_path:       %s\n", params.output_path.c_str());
    printf("    init_img:          %s\n", params.input_path.c_str());
    printf("    control_image:     %s\n", params.control_image_path.c_str());
    printf("    controlnet cpu:    %s\n", params.control_net_cpu ? "true" : "false");
    printf("    strength(control): %.2f\n", params.control_strength);
    printf("    prompt:            %s\n", params.prompt.c_str());
    printf("    negative_prompt:   %s\n", params.negative_prompt.c_str());
    printf("    cfg_scale:         %.2f\n", params.cfg_scale);
    printf("    clip_skip:         %d\n", params.clip_skip);
    printf("    width:             %d\n", params.width);
    printf("    height:            %d\n", params.height);
    printf("    sample_method:     %s\n", sample_method_str[params.sample_method]);
    printf("    schedule:          %s\n", schedule_str[params.schedule]);
    printf("    sample_steps:      %d\n", params.sample_steps);
    printf("    strength(img2img): %.2f\n", params.strength);
    printf("    rng:               %s\n", rng_type_to_str[params.rng_type]);
    printf("    seed:              %ld\n", params.seed);
    printf("    batch_count:       %d\n", params.batch_count);
    printf("    vae_tiling:        %s\n", params.vae_tiling ? "true" : "false");
}

void print_usage(int argc, const char* argv[]) {
    printf("usage: %s [arguments]\n", argv[0]);
    printf("\n");
    printf("arguments:\n");
    printf("  -h, --help                         show this help message and exit\n");
    printf("  -M, --mode [MODEL]                 run mode (txt2img or img2img or convert or stream, default: txt2img)\n");
    printf("  -t, --threads N                    number of threads to use during computation (default: -1).\n");
    printf("                                     If threads <= 0, then threads will be set to the number of CPU physical cores\n");
    printf("  -m, --model [MODEL]                path to model\n");
    printf("                                     If the path is directory, support load model from \"unet/diffusion_pytorch_model.safetensors\", \"vae/diffusion_pytorch_model.safetensors\",\"text_encoder/model.safetensors\"\n");
    printf("  --vae [VAE]                        path to vae\n");
    printf("  --clip [CLIP]                      path to clip\n");
    printf("  --unet [UNET]                      path to unet\n");
    printf("  --taesd [TAESD_PATH]               path to taesd. Using Tiny AutoEncoder for fast decoding (low quality)\n");
    printf("  --control-net [CONTROL_PATH]       path to control net model\n");
    printf("  --embd-dir [EMBEDDING_PATH]        path to embeddings.\n");
    printf("  --upscale-model [ESRGAN_PATH]      path to esrgan model. Upscale images after generate, just RealESRGAN_x4plus_anime_6B supported by now.\n");
    printf("  --type [TYPE]                      weight type (f32, f16, q4_0, q4_1, q5_0, q5_1, q8_0)\n");
    printf("                                     If not specified, the default is the type of the weight file.\n");
    printf("  --lora-model-dir [DIR]             lora model directory\n");
    printf("  -i, --init-img [IMAGE]             path to the input image, required by img2img\n");
    printf("  --control-image [IMAGE]            path to image condition, control net\n");
    printf("  -o, --output OUTPUT                path to write result image to (default: ./output.png)\n");
    printf("  -p, --prompt [PROMPT]              the prompt to render\n");
    printf("  -n, --negative-prompt PROMPT       the negative prompt (default: \"\")\n");
    printf("  --cfg-scale SCALE                  unconditional guidance scale: (default: 7.0)\n");
    printf("  --strength STRENGTH                strength for noising/unnoising (default: 0.75)\n");
    printf("  --control-strength STRENGTH        strength to apply Control Net (default: 0.9)\n");
    printf("                                     1.0 corresponds to full destruction of information in init image\n");
    printf("  -H, --height H                     image height, in pixel space (default: 512)\n");
    printf("  -W, --width W                      image width, in pixel space (default: 512)\n");
    printf("  --sampling-method                  {euler, euler_a, heun, dpm2, dpm++2s_a, dpm++2m, dpm++2mv2, lcm}\n");
    printf("                                     sampling method (default: \"euler_a\")\n");
    printf("  --steps  STEPS                     number of sample steps (default: 20)\n");
    printf("  --rng {std_default, cuda}          RNG (default: cuda)\n");
    printf("  -s SEED, --seed SEED               RNG seed (default: 42, use random seed for < 0)\n");
    printf("  -b, --batch-count COUNT            number of images to generate.\n");
    printf("  --schedule {discrete, karras}      Denoiser sigma schedule (default: discrete)\n");
    printf("  --clip-skip N                      ignore last layers of CLIP network; 1 ignores none, 2 ignores one layer (default: -1)\n");
    printf("                                     <= 0 represents unspecified, will be 1 for SD1.x, 2 for SD2.x\n");
    printf("  --vae-tiling                       process vae in tiles to reduce memory usage\n");
    printf("  --control-net-cpu                  keep controlnet in cpu (for low vram)\n");
    printf("  --canny                            apply canny preprocessor (edge detection)\n");
    printf("  -v, --verbose                      print extra info\n");
}

void parse_args(int argc, const char** argv, SDParams& params) {
    bool invalid_arg = false;
    std::string arg;
    for (int i = 1; i < argc; i++) {
        arg = argv[i];

        if (arg == "-t" || arg == "--threads") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.n_threads = std::stoi(argv[i]);
        } else if (arg == "-M" || arg == "--mode") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            const char* mode_selected = argv[i];
            int mode_found            = -1;
            for (int d = 0; d < MODE_COUNT; d++) {
                if (!strcmp(mode_selected, modes_str[d])) {
                    mode_found = d;
                }
            }
            if (mode_found == -1) {
                fprintf(stderr, "error: invalid mode %s, must be one of [txt2img, img2img]\n",
                        mode_selected);
                exit(1);
            }
            params.mode = (SDMode)mode_found;
        } else if (arg == "-m" || arg == "--model") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.model_path = argv[i];
        } else if (arg == "--vae") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.vae_path = argv[i];
        } else if (arg == "--clip") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.vae_path = argv[i];
        } else if (arg == "--unet") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.vae_path = argv[i];
        } else if (arg == "--taesd") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.taesd_path = argv[i];
        } else if (arg == "--control-net") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.controlnet_path = argv[i];
        } else if (arg == "--upscale-model") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.esrgan_path = argv[i];
        } else if (arg == "--embd-dir") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.embeddings_path = argv[i];
        } else if (arg == "--type") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            std::string type = argv[i];
            if (type == "f32") {
                params.wtype = SD_TYPE_F32;
            } else if (type == "f16") {
                params.wtype = SD_TYPE_F16;
            } else if (type == "q4_0") {
                params.wtype = SD_TYPE_Q4_0;
            } else if (type == "q4_1") {
                params.wtype = SD_TYPE_Q4_1;
            } else if (type == "q5_0") {
                params.wtype = SD_TYPE_Q5_0;
            } else if (type == "q5_1") {
                params.wtype = SD_TYPE_Q5_1;
            } else if (type == "q8_0") {
                params.wtype = SD_TYPE_Q8_0;
            } else {
                fprintf(stderr, "error: invalid weight format %s, must be one of [f32, f16, q4_0, q4_1, q5_0, q5_1, q8_0]\n",
                        type.c_str());
                exit(1);
            }
        } else if (arg == "--lora-model-dir") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.lora_model_dir = argv[i];
        } else if (arg == "-i" || arg == "--init-img") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.input_path = argv[i];
        } else if (arg == "--control-image") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.control_image_path = argv[i];
        } else if (arg == "-o" || arg == "--output") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.output_path = argv[i];
        } else if (arg == "-p" || arg == "--prompt") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.prompt = argv[i];
        } else if (arg == "-n" || arg == "--negative-prompt") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.negative_prompt = argv[i];
        } else if (arg == "--cfg-scale") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.cfg_scale = std::stof(argv[i]);
        } else if (arg == "--strength") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.strength = std::stof(argv[i]);
        } else if (arg == "--control-strength") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.control_strength = std::stof(argv[i]);
        } else if (arg == "-H" || arg == "--height") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.height = std::stoi(argv[i]);
        } else if (arg == "-W" || arg == "--width") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.width = std::stoi(argv[i]);
        } else if (arg == "--steps") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.sample_steps = std::stoi(argv[i]);
        } else if (arg == "--clip-skip") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.clip_skip = std::stoi(argv[i]);
        } else if (arg == "--vae-tiling") {
            params.vae_tiling = true;
        } else if (arg == "--control-net-cpu") {
            params.control_net_cpu = true;
        } else if (arg == "--canny") {
            params.canny_preprocess = true;
        } else if (arg == "-b" || arg == "--batch-count") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.batch_count = std::stoi(argv[i]);
        } else if (arg == "--rng") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            std::string rng_type_str = argv[i];
            if (rng_type_str == "std_default") {
                params.rng_type = STD_DEFAULT_RNG;
            } else if (rng_type_str == "cuda") {
                params.rng_type = CUDA_RNG;
            } else {
                invalid_arg = true;
                break;
            }
        } else if (arg == "--schedule") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            const char* schedule_selected = argv[i];
            int schedule_found            = -1;
            for (int d = 0; d < N_SCHEDULES; d++) {
                if (!strcmp(schedule_selected, schedule_str[d])) {
                    schedule_found = d;
                }
            }
            if (schedule_found == -1) {
                invalid_arg = true;
                break;
            }
            params.schedule = (schedule_t)schedule_found;
        } else if (arg == "-s" || arg == "--seed") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.seed = std::stoll(argv[i]);
        } else if (arg == "--sampling-method") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            const char* sample_method_selected = argv[i];
            int sample_method_found            = -1;
            for (int m = 0; m < N_SAMPLE_METHODS; m++) {
                if (!strcmp(sample_method_selected, sample_method_str[m])) {
                    sample_method_found = m;
                }
            }
            if (sample_method_found == -1) {
                invalid_arg = true;
                break;
            }
            params.sample_method = (sample_method_t)sample_method_found;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argc, argv);
            exit(0);
        } else if (arg == "-v" || arg == "--verbose") {
            params.verbose = true;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            print_usage(argc, argv);
            exit(1);
        }
    }
    if (invalid_arg) {
        fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
        print_usage(argc, argv);
        exit(1);
    }
}

bool check_params(SDParams params) {
    std::vector<std::string> required_args;
    std::vector<std::string> invalid_args;

    if (params.n_threads <= 0) {
        params.n_threads = get_num_physical_cores();
    }

    if (params.mode != CONVERT && params.prompt.length() == 0) {
        required_args.emplace_back("prompt");
    }

    if (params.model_path.length() == 0) {
        required_args.emplace_back("model_path");
    }

    if (params.mode == IMG2IMG && params.input_path.length() == 0) {
        required_args.emplace_back("init-img");
    }

    if (params.output_path.length() == 0) {
        required_args.emplace_back("output_path");
    }

    if (params.width <= 0 || params.width % 64 != 0) {
        invalid_args.emplace_back("the width must be a multiple of 64");
    }

    if (params.height <= 0 || params.height % 64 != 0) {
        invalid_args.emplace_back("the height must be a multiple of 64");
    }

    if (params.sample_steps <= 0) {
        invalid_args.emplace_back("the sample_steps must be greater than 0");
    }

    if (params.strength < 0.f || params.strength > 1.f) {
        invalid_args.emplace_back("can only work with strength in [0.0, 1.0]");
    }

    if (params.seed < 0) {
        srand((int)time(NULL));
        params.seed = rand();
    }

    if (params.mode == CONVERT) {
        if (params.output_path == "output.png") {
            params.output_path = "output.gguf";
        }
    }

    if ((!invalid_args.empty()) || (!required_args.empty())) {
        if (!invalid_args.empty()) {
            std::ostringstream oss;
            for (int i = 0; i < invalid_args.size(); i++) {
                if (i > 0) {
                    oss << ",\n";
                }
                oss << invalid_args[i];
            }
            std::string invalid_args_str = oss.str();
            std::cout << "error: " << invalid_args_str << std::endl;
        }

        if (!required_args.empty()) {
            std::ostringstream oss;
            for (int i = 0; i < required_args.size(); i++) {
                if (i > 0) {
                    oss << ",";
                }
                oss << required_args[i];
            }
            std::string required_args_str = oss.str();
            std::cout << "require: " << required_args_str << std::endl;
        }

        return false;
    }

    return true;
}

std::string get_image_params(SDParams params, int64_t seed) {
    std::string parameter_string = params.prompt + "\n";
    if (params.negative_prompt.size() != 0) {
        parameter_string += "Negative prompt: " + params.negative_prompt + "\n";
    }
    parameter_string += "Steps: " + std::to_string(params.sample_steps) + ", ";
    parameter_string += "CFG scale: " + std::to_string(params.cfg_scale) + ", ";
    parameter_string += "Seed: " + std::to_string(seed) + ", ";
    parameter_string += "Size: " + std::to_string(params.width) + "x" + std::to_string(params.height) + ", ";
    parameter_string += "Model: " + sd_basename(params.model_path) + ", ";
    parameter_string += "RNG: " + std::string(rng_type_to_str[params.rng_type]) + ", ";
    parameter_string += "Sampler: " + std::string(sample_method_str[params.sample_method]);
    if (params.schedule == KARRAS) {
        parameter_string += " karras";
    }
    parameter_string += ", ";
    parameter_string += "Version: stable-diffusion.cpp";
    return parameter_string;
}

void sd_log_cb(enum sd_log_level_t level, const char* log, void* data) {
    SDParams* params = (SDParams*)data;
    if (!params->verbose && level <= SD_LOG_DEBUG) {
        return;
    }
    if (level <= SD_LOG_INFO) {
        fputs(log, stdout);
        fflush(stdout);
    } else {
        fputs(log, stderr);
        fflush(stderr);
    }
}

std::vector<std::string> parse_cin(std::string& input, std::set<std::string> ignore_args) {
    std::vector<std::string> inputTokens;
    std::string token;
    std::istringstream iss(input);

    std::string word;
    bool in_stmt = false;
    std::string stmt;
    inputTokens.emplace_back("fake run path, no use!");
    while (iss >> word) {
        if (word[0] == '"') {
            in_stmt = true;
        }

        if (word[word.length() - 1] == '"') {
            stmt += word;
            word    = stmt.substr(1, stmt.length() - 2);
            stmt    = "";
            in_stmt = false;
        }

        if (in_stmt) {
            stmt += word;
            stmt += " ";
            continue;
        }
        inputTokens.push_back(word);
    }

    std::vector<std::string> commands;
    for (int i = 0; i < inputTokens.size(); i++) {
        if (ignore_args.find(inputTokens[i]) != ignore_args.end()) {
            i++;
            continue;
        }
        commands.push_back(inputTokens[i]);
    }
    return commands;
}

SDParams merge_params(SDParams dst, SDParams src) {
    if (dst.n_threads != src.n_threads) {
        if (src.n_threads > 0) {
            dst.n_threads = src.n_threads;
        }
    }

    if (dst.mode != src.mode) {
        if (src.mode == TXT2IMG || src.mode == IMG2IMG) {
            dst.mode = src.mode;
            if (dst.mode == IMG2IMG) {
                dst.vae_decode_only = false;
            }
        }
    }

    if (dst.model_path != src.model_path) {
        if (!src.model_path.empty()) {
            dst.model_path = src.model_path;
        }
    }

    if (dst.vae_path != src.vae_path) {
        if (!src.vae_path.empty()) {
            dst.vae_path = src.vae_path;
        }
    }

    if (dst.clip_path != src.clip_path) {
        if (!src.clip_path.empty()) {
            dst.clip_path = src.clip_path;
        }
    }

    if (dst.unet_path != src.unet_path) {
        if (!src.unet_path.empty()) {
            dst.unet_path = src.unet_path;
        }
    }

    if (dst.taesd_path != src.taesd_path) {
        if (!src.taesd_path.empty()) {
            dst.taesd_path = src.taesd_path;
        }
    }

    if (dst.esrgan_path != src.esrgan_path) {
        if (!src.esrgan_path.empty()) {
            dst.esrgan_path = src.esrgan_path;
        }
    }

    if (dst.wtype != src.wtype) {
        dst.wtype = src.wtype;
    }

    if (dst.lora_model_dir != src.lora_model_dir) {
        if (!src.lora_model_dir.empty()) {
            dst.lora_model_dir = src.lora_model_dir;
        }
    }

    if (dst.output_path != src.output_path) {
        if (!src.output_path.empty()) {
            dst.output_path = src.output_path;
        }
    }

    if (dst.prompt != src.prompt) {
        if (!src.prompt.empty()) {
            dst.prompt = src.prompt;
        }
    }

    if (dst.negative_prompt != src.negative_prompt) {
        if (!src.negative_prompt.empty()) {
            dst.negative_prompt = src.negative_prompt;
        }
    }

    if (dst.cfg_scale != src.cfg_scale) {
        if (src.cfg_scale >= 0) {
            dst.cfg_scale = src.cfg_scale;
        }
    }

    if (dst.clip_skip != src.clip_skip) {
        dst.clip_skip = src.clip_skip;
    }

    if (dst.width != src.width) {
        if (src.width > 0 || src.width % 64 == 0) {
            dst.width = src.width;
        }
    }

    if (dst.height != src.height) {
        if (src.height > 0 || src.height % 64 == 0) {
            dst.height = src.height;
        }
    }

    if (dst.sample_steps != src.sample_steps) {
        if (src.sample_steps > 0) {
            dst.sample_steps = src.sample_steps;
        }
    }

    if (dst.strength != src.strength) {
        if (src.strength >= 0.f && src.strength <= 1.f) {
            dst.strength = src.strength;
        }
    }

    if (dst.seed != src.seed) {
        if (src.seed > 0) {
            dst.seed = src.seed;
        }
    }
    return dst;
}

class CliInstance {
public:
    sd_ctx_t* sd_ctx;

    ~CliInstance() {
        free_sd_ctx(sd_ctx);
    }

    CliInstance(const SDParams& params) {
        sd_ctx = new_sd_ctx(
            params.n_threads,
            params.vae_decode_only,
            false,
            params.lora_model_dir.c_str(),
            params.rng_type,
            params.vae_tiling,
            params.wtype,
            params.schedule,
            params.control_net_cpu,
            true);
    }

    bool load_from_file(SDParams& params) {
        // free api always check if the following methods can free, so we can always free the model before load it.
        free_diffusions_params(sd_ctx);
        auto load_status = load_diffusions_from_file(sd_ctx, params.model_path.c_str());

        if (load_status && !params.clip_path.empty()) {
            free_clip_params(sd_ctx);
            load_status = load_clip_from_file(sd_ctx, params.clip_path.c_str());
        }

        if (load_status && !params.vae_path.empty()) {
            free_vae_params(sd_ctx);
            load_status = load_vae_from_file(sd_ctx, params.vae_path.c_str());
        }

        if (load_status && !params.unet_path.empty()) {
            free_unet_params(sd_ctx);
            load_status = load_unet_from_file(sd_ctx, params.unet_path.c_str());
        }

        return load_status;
    }

    void txtimg(SDParams& params) {
        set_options(sd_ctx, params.n_threads,
                    params.vae_decode_only,
                    true,
                    params.lora_model_dir.c_str(),
                    params.rng_type,
                    params.vae_tiling,
                    params.wtype,
                    params.schedule);
        int c                       = 0;
        uint8_t* input_image_buffer = stbi_load(params.control_image_path.c_str(), &params.width, &params.height, &c, 3);
        if (input_image_buffer == NULL) {
            fprintf(stderr, "load image from '%s' failed\n", params.control_image_path.c_str());
            return;
        }
        if (c != 3) {
            fprintf(stderr, "input image must be a 3 channels RGB image, but got %d channels\n", c);
            free(input_image_buffer);
            return;
        }

        sd_image_t input_image = {(uint32_t)params.width,
                                  (uint32_t)params.height,
                                  3,
                                  input_image_buffer};

        sd_image_t* results = txt2img(sd_ctx,
                                      params.prompt.c_str(),
                                      params.negative_prompt.c_str(),
                                      params.clip_skip,
                                      params.cfg_scale,
                                      params.width,
                                      params.height,
                                      params.sample_method,
                                      params.sample_steps,
                                      params.seed,
                                      params.batch_count,
                                      &input_image,
                                      params.control_strength);

        results = upscaler(params, results);
        save_image(params, results);
    }

    void imgimg(SDParams& params) {
        set_options(sd_ctx, params.n_threads,
                    params.vae_decode_only,
                    true,
                    params.lora_model_dir.c_str(),
                    params.rng_type,
                    params.vae_tiling,
                    params.wtype,
                    params.schedule);
        uint8_t* input_image_buffer = NULL;

        int c              = 0;
        input_image_buffer = stbi_load(params.input_path.c_str(), &params.width, &params.height, &c, 3);
        if (input_image_buffer == NULL) {
            fprintf(stderr, "load image from '%s' failed\n", params.input_path.c_str());
            return;
        }
        if (c != 3) {
            fprintf(stderr, "input image must be a 3 channels RGB image, but got %d channels\n", c);
            free(input_image_buffer);
            return;
        }
        if (params.width <= 0 || params.width % 64 != 0) {
            fprintf(stderr, "error: the width of image must be a multiple of 64\n");
            free(input_image_buffer);
            return;
        }

        if (params.height <= 0 || params.height % 64 != 0) {
            fprintf(stderr, "error: the height of image must be a multiple of 64\n");
            free(input_image_buffer);
            return;
        }

        sd_image_t input_image = {(uint32_t)params.width,
                                  (uint32_t)params.height,
                                  3,
                                  input_image_buffer};

        sd_image_t* results = img2img(sd_ctx,
                                      input_image,
                                      params.prompt.c_str(),
                                      params.negative_prompt.c_str(),
                                      params.clip_skip,
                                      params.cfg_scale,
                                      params.width,
                                      params.height,
                                      params.sample_method,
                                      params.sample_steps,
                                      params.strength,
                                      params.seed,
                                      params.batch_count);
        results             = upscaler(params, results);
        save_image(params, results);
    }

protected:
    void save_image(const SDParams& params, sd_image_t* results) {
        size_t last            = params.output_path.find_last_of(".");
        std::string dummy_name = last != std::string::npos ? params.output_path.substr(0, last) : params.output_path;
        for (int i = 0; i < params.batch_count; i++) {
            if (results[i].data == NULL) {
                continue;
            }
            std::string final_image_path =
                i > 0 ? dummy_name + "_" + std::to_string(i + 1) + ".png" : dummy_name + ".png";
            stbi_write_png(final_image_path.c_str(), results[i].width, results[i].height, results[i].channel,
                           results[i].data, 0, get_image_params(params, params.seed + i).c_str());
            printf("save result image to '%s'\n", final_image_path.c_str());
            free(results[i].data);
            results[i].data = NULL;
        }
        free(results);
    }

    sd_image_t* upscaler(const SDParams& params, sd_image_t* results) {
        int upscale_factor = 4;  // unused for RealESRGAN_x4plus_anime_6B.pth
        if (params.esrgan_path.size() > 0) {
            upscaler_ctx_t* upscaler_ctx = new_upscaler_ctx(params.esrgan_path.c_str(),
                                                            params.n_threads,
                                                            params.wtype);
            if (upscaler_ctx == NULL) {
                printf("new_upscaler_ctx failed\n");
            } else {
                for (int i = 0; i < params.batch_count; i++) {
                    if (results[i].data == NULL) {
                        continue;
                    }
                    sd_image_t upscaled_image = upscale(upscaler_ctx, results[i], upscale_factor);
                    if (upscaled_image.data == NULL) {
                        printf("upscale failed\n");
                        continue;
                    }
                    free(results[i].data);
                    results[i] = upscaled_image;
                }
                free_upscaler_ctx(upscaler_ctx);
            }
        }
        return results;
    }
};

int main(int argc, const char* argv[]) {
    SDParams params;

    parse_args(argc, argv, params);

    if (params.mode != STREAM && !check_params(params)) {
        return 1;
    }

    sd_set_log_callback(sd_log_cb, (void*)&params);

    if (params.verbose) {
        print_params(params);
        printf("%s", sd_get_system_info());
    }

    if (params.mode == CONVERT) {
        bool success = convert(params.model_path.c_str(),
                               params.vae_path.c_str(),
                               params.output_path.c_str(),
                               params.wtype);
        if (!success) {
            fprintf(stderr,
                    "convert '%s'/'%s' to '%s' failed\n",
                    params.model_path.c_str(),
                    params.vae_path.c_str(),
                    params.output_path.c_str());
            return 1;
        } else {
            printf("convert '%s'/'%s' to '%s' success\n",
                   params.model_path.c_str(),
                   params.vae_path.c_str(),
                   params.output_path.c_str());
            return 0;
        }
    }

    auto instance = new CliInstance(params);

    if (params.mode == STREAM) {
        std::cout << "you are in stream model, feel free to use txt2img or img2img" << std::endl;
        while (true) {
            std::string input;
            std::cout << "please input args: " << std::endl;
            std::getline(std::cin, input);
            // hold an ignore cmd for feature to ignore the cmd not support
            std::set<std::string> ignore_cmd = {""};
            std::vector<std::string> args    = parse_cin(input, ignore_cmd);
            SDParams stream_params;
            const char** args_c_arr = new const char*[args.size()];
            for (int i = 0; i < args.size(); i++) {
                std::string arg = args[i];
                char* c_str     = new char[args[i].length() + 1];
                std::strcpy(c_str, arg.c_str());
                args_c_arr[i] = c_str;
            }
            parse_args(args.size(), args_c_arr, stream_params);
            if (params.model_path != stream_params.model_path ||
                params.clip_path != stream_params.clip_path ||
                params.vae_path != stream_params.vae_path ||
                params.unet_path != stream_params.unet_path) {
                instance->load_from_file(stream_params);
            }
            params = merge_params(params, stream_params);
            if (!check_params(params)) {
                continue;
            }
            if (params.mode == TXT2IMG) {
                instance->txtimg(params);
            } else if (params.mode == IMG2IMG) {
                instance->imgimg(params);
            } else {
                return 1;
            }
        }
    } else {
        if (!params.model_path.empty()) {
            if (!instance->load_from_file(params)) {
                return 1;
            }
        } else {
            if (!params.clip_path.empty() && !params.vae_path.empty() && !params.unet_path.empty()) {
                if (!instance->load_from_file(params)) {
                    return 1;
                }
            }
        }
        if (params.mode == TXT2IMG) {
            instance->txtimg(params);
        } else if (params.mode == IMG2IMG) {
            instance->imgimg(params);
        } else {
            return 0;
        }
    }
    return 0;
}
