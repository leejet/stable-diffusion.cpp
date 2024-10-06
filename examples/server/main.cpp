#include <stdio.h>
#include <string.h>
#include <time.h>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// #include "preprocessing.hpp"
#include "flux.hpp"
#include "stable-diffusion.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include "stb_image_write.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_STATIC
#include "stb_image_resize.h"

#include "b64.cpp"
#include "httplib.h"
#include "json.hpp"

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
    "ays",
};

enum SDMode {
    TXT2IMG,
    IMG2IMG,
    IMG2VID,
    CONVERT,
    MODE_COUNT
};

struct SDParams {
    int n_threads = -1;
    SDMode mode   = TXT2IMG;

    std::string model_path;
    std::string clip_l_path;
    std::string t5xxl_path;
    std::string diffusion_model_path;
    std::string vae_path;
    // std::string taesd_path;
    std::string embeddings_path;
    std::string stacked_id_embeddings_path;
    sd_type_t wtype = SD_TYPE_COUNT;
    std::string lora_model_dir;
    std::string output_path = "output.png";
    std::string input_path;

    std::string prompt;
    std::string negative_prompt;
    float min_cfg     = 1.0f;
    float cfg_scale   = 7.0f;
    float guidance    = 3.5f;
    float style_ratio = 20.f;
    int clip_skip     = -1;  // <= 0 represents unspecified
    int width         = 512;
    int height        = 512;
    int batch_count   = 1;

    sample_method_t sample_method = EULER_A;
    schedule_t schedule           = DEFAULT;
    int sample_steps              = 20;
    float strength                = 0.75f;
    rng_type_t rng_type           = CUDA_RNG;
    int64_t seed                  = 42;
    bool verbose                  = false;
    bool vae_tiling               = false;
    bool normalize_input          = false;
    bool clip_on_cpu              = false;
    bool vae_on_cpu               = false;
    bool color                    = false;

    // server things
    int port         = 8080;
    std::string host = "127.0.0.1";
};

void print_params(SDParams params) {
    printf("Option: \n");
    printf("    n_threads:         %d\n", params.n_threads);
    printf("    model_path:        %s\n", params.model_path.c_str());
    printf("    wtype:             %s\n", params.wtype < SD_TYPE_COUNT ? sd_type_name(params.wtype) : "unspecified");
    printf("    clip_l_path:       %s\n", params.clip_l_path.c_str());
    printf("    t5xxl_path:        %s\n", params.t5xxl_path.c_str());
    printf("    diffusion_model_path:   %s\n", params.diffusion_model_path.c_str());
    printf("    vae_path:          %s\n", params.vae_path.c_str());
    // printf("    taesd_path:        %s\n", params.taesd_path.c_str());
    printf("    embeddings_path:   %s\n", params.embeddings_path.c_str());
    printf("    stacked_id_embeddings_path:   %s\n", params.stacked_id_embeddings_path.c_str());
    printf("    style ratio:       %.2f\n", params.style_ratio);
    printf("    normzalize input image :  %s\n", params.normalize_input ? "true" : "false");
    printf("    output_path:       %s\n", params.output_path.c_str());
    printf("    clip on cpu:       %s\n", params.clip_on_cpu ? "true" : "false");
    printf("    vae decoder on cpu:%s\n", params.vae_on_cpu ? "true" : "false");
    printf("    prompt:            %s\n", params.prompt.c_str());
    printf("    negative_prompt:   %s\n", params.negative_prompt.c_str());
    printf("    min_cfg:           %.2f\n", params.min_cfg);
    printf("    cfg_scale:         %.2f\n", params.cfg_scale);
    printf("    guidance:          %.2f\n", params.guidance);
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
    printf("  -M, --mode [MODEL]                 run mode (txt2img or img2img or convert, default: txt2img)\n");
    printf("  -t, --threads N                    number of threads to use during computation (default: -1).\n");
    printf("                                     If threads <= 0, then threads will be set to the number of CPU physical cores\n");
    printf("  -m, --model [MODEL]                path to full model\n");
    printf("  --diffusion-model                  path to the standalone diffusion model\n");
    printf("  --clip_l                           path to the clip-l text encoder\n");
    printf("  --t5xxl                            path to the the t5xxl text encoder.\n");
    printf("  --vae [VAE]                        path to vae\n");
    printf("  --embd-dir [EMBEDDING_PATH]        path to embeddings.\n");
    printf("  --type [TYPE]                      weight type (f32, f16, q4_0, q4_1, q5_0, q5_1, q8_0, q2_k, q3_k, q4_k)\n");
    printf("                                     If not specified, the default is the type of the weight file.\n");
    printf("  --lora-model-dir [DIR]             lora model directory\n");
    printf("  -o, --output OUTPUT                path to write result image to (default: ./output.png)\n");
    printf("  -p, --prompt [PROMPT]              the prompt to render\n");
    printf("  -n, --negative-prompt PROMPT       the negative prompt (default: \"\")\n");
    printf("  --cfg-scale SCALE                  unconditional guidance scale: (default: 7.0)\n");
    printf("  --strength STRENGTH                strength for noising/unnoising (default: 0.75)\n");
    printf("  --style-ratio STYLE-RATIO          strength for keeping input identity (default: 20%%)\n");
    printf("  --control-strength STRENGTH        strength to apply Control Net (default: 0.9)\n");
    printf("                                     1.0 corresponds to full destruction of information in init image\n");
    printf("  -H, --height H                     image height, in pixel space (default: 512)\n");
    printf("  -W, --width W                      image width, in pixel space (default: 512)\n");
    printf("  --sampling-method {euler, euler_a, heun, dpm2, dpm++2s_a, dpm++2m, dpm++2mv2, lcm}\n");
    printf("                                     sampling method (default: \"euler_a\")\n");
    printf("  --steps  STEPS                     number of sample steps (default: 20)\n");
    printf("  --rng {std_default, cuda}          RNG (default: cuda)\n");
    printf("  -s SEED, --seed SEED               RNG seed (default: 42, use random seed for < 0)\n");
    printf("  -b, --batch-count COUNT            number of images to generate.\n");
    printf("  --schedule {discrete, karras, ays} Denoiser sigma schedule (default: discrete)\n");
    printf("  --clip-skip N                      ignore last layers of CLIP network; 1 ignores none, 2 ignores one layer (default: -1)\n");
    printf("                                     <= 0 represents unspecified, will be 1 for SD1.x, 2 for SD2.x\n");
    printf("  --vae-tiling                       process vae in tiles to reduce memory usage\n");
    printf("  --vae-on-cpu                       keep vae in cpu (for low vram)\n");
    printf("  --clip-on-cpu                      keep clip in cpu (for low vram).\n");
    printf("  --color                            Colors the logging tags according to level\n");
    printf("  -v, --verbose                      print extra info\n");
    printf("  --port                             port used for server (default: 8080)\n");
    printf("  --host                             IP address used for server. Use 0.0.0.0 to expose server to LAN (default: localhost)\n");
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
        } else if (arg == "-m" || arg == "--model") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.model_path = argv[i];
        } else if (arg == "--clip_l") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.clip_l_path = argv[i];
        } else if (arg == "--t5xxl") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.t5xxl_path = argv[i];
        } else if (arg == "--diffusion-model") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.diffusion_model_path = argv[i];
        } else if (arg == "--vae") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.vae_path = argv[i];
            // TODO Tiny AE
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
            } else if (type == "q2_k") {
                params.wtype = SD_TYPE_Q2_K;
            } else if (type == "q3_k") {
                params.wtype = SD_TYPE_Q3_K;
            } else if (type == "q4_k") {
                params.wtype = SD_TYPE_Q4_K;
            } else {
                fprintf(stderr, "error: invalid weight format %s, must be one of [f32, f16, q4_0, q4_1, q5_0, q5_1, q8_0, q2_k, q3_k, q4_k]\n",
                        type.c_str());
                exit(1);
            }
        } else if (arg == "--lora-model-dir") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.lora_model_dir = argv[i];
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
        } else if (arg == "--guidance") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.guidance = std::stof(argv[i]);
        } else if (arg == "--strength") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.strength = std::stof(argv[i]);
        } else if (arg == "--style-ratio") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.style_ratio = std::stof(argv[i]);
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
        } else if (arg == "--normalize-input") {
            params.normalize_input = true;
        } else if (arg == "--clip-on-cpu") {
            params.clip_on_cpu = true;  // will slow down get_learned_condiotion but necessary for low MEM GPUs
        } else if (arg == "--vae-on-cpu") {
            params.vae_on_cpu = true;  // will slow down latent decoding but necessary for low MEM GPUs
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
        } else if (arg == "--color") {
            params.color = true;
        } else if (arg == "--port") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.port = std::stoi(argv[i]);
        } else if (arg == "--host") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.host = argv[i];
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
    if (params.n_threads <= 0) {
        params.n_threads = get_num_physical_cores();
    }

    if (params.mode != CONVERT && params.mode != IMG2VID && params.prompt.length() == 0) {
        fprintf(stderr, "error: the following arguments are required: prompt\n");
        print_usage(argc, argv);
        exit(1);
    }

    if (params.model_path.length() == 0 && params.diffusion_model_path.length() == 0) {
        fprintf(stderr, "error: the following arguments are required: model_path/diffusion_model\n");
        print_usage(argc, argv);
        exit(1);
    }

    if ((params.mode == IMG2IMG || params.mode == IMG2VID) && params.input_path.length() == 0) {
        fprintf(stderr, "error: when using the img2img mode, the following arguments are required: init-img\n");
        print_usage(argc, argv);
        exit(1);
    }

    if (params.output_path.length() == 0) {
        fprintf(stderr, "error: the following arguments are required: output_path\n");
        print_usage(argc, argv);
        exit(1);
    }

    if (params.width <= 0 || params.width % 64 != 0) {
        fprintf(stderr, "error: the width must be a multiple of 64\n");
        exit(1);
    }

    if (params.height <= 0 || params.height % 64 != 0) {
        fprintf(stderr, "error: the height must be a multiple of 64\n");
        exit(1);
    }

    if (params.sample_steps <= 0) {
        fprintf(stderr, "error: the sample_steps must be greater than 0\n");
        exit(1);
    }

    if (params.strength < 0.f || params.strength > 1.f) {
        fprintf(stderr, "error: can only work with strength in [0.0, 1.0]\n");
        exit(1);
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
}

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

std::string get_image_params(SDParams params, int64_t seed) {
    std::string parameter_string = params.prompt + "\n";
    if (params.negative_prompt.size() != 0) {
        parameter_string += "Negative prompt: " + params.negative_prompt + "\n";
    }
    parameter_string += "Steps: " + std::to_string(params.sample_steps) + ", ";
    parameter_string += "CFG scale: " + std::to_string(params.cfg_scale) + ", ";
    parameter_string += "Guidance: " + std::to_string(params.guidance) + ", ";
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

/* Enables Printing the log level tag in color using ANSI escape codes */
void sd_log_cb(enum sd_log_level_t level, const char* log, void* data) {
    SDParams* params = (SDParams*)data;
    int tag_color;
    const char* level_str;
    FILE* out_stream = (level == SD_LOG_ERROR) ? stderr : stdout;

    if (!log || (!params->verbose && level <= SD_LOG_DEBUG)) {
        return;
    }

    switch (level) {
        case SD_LOG_DEBUG:
            tag_color = 37;
            level_str = "DEBUG";
            break;
        case SD_LOG_INFO:
            tag_color = 34;
            level_str = "INFO";
            break;
        case SD_LOG_WARN:
            tag_color = 35;
            level_str = "WARN";
            break;
        case SD_LOG_ERROR:
            tag_color = 31;
            level_str = "ERROR";
            break;
        default: /* Potential future-proofing */
            tag_color = 33;
            level_str = "?????";
            break;
    }

    if (params->color == true) {
        fprintf(out_stream, "\033[%d;1m[%-5s]\033[0m ", tag_color, level_str);
    } else {
        fprintf(out_stream, "[%-5s] ", level_str);
    }
    fputs(log, out_stream);
    fflush(out_stream);
}

void* server_log_params = NULL;

// enable logging in the server
#define LOG_BUFFER_SIZE 1024
void sd_log(enum sd_log_level_t level, const char* format, ...) {
    va_list args;
    va_start(args, format);

    char log[LOG_BUFFER_SIZE];
    vsnprintf(log, 1024, format, args);
    strncat(log, "\n", LOG_BUFFER_SIZE - strlen(log));

    sd_log_cb(level, log, server_log_params);
    va_end(args);
}

static void log_server_request(const httplib::Request& req, const httplib::Response& res) {
    printf("request: %s %s (%s)\n", req.method.c_str(), req.path.c_str(), req.body.c_str());
}

void parseJsonPrompt(std::string json_str, SDParams* params) {
    using namespace nlohmann;
    json payload = json::parse(json_str);
    // if no exception, the request is a json object
    // now we try to get the new param values from the payload object
    // const char *prompt, const char *negative_prompt, int clip_skip, float cfg_scale, float guidance, int width, int height, sample_method_t sample_method, int sample_steps, int64_t seed, int batch_count, const sd_image_t *control_cond, float control_strength, float style_strength, bool normalize_input, const char *input_id_images_path
    try {
        std::string prompt = payload["prompt"];
        params->prompt     = prompt;
    } catch (...) {
    }
    try {
        std::string negative_prompt = payload["negative_prompt"];
        params->negative_prompt     = negative_prompt;
    } catch (...) {
    }
    try {
        int clip_skip     = payload["clip_skip"];
        params->clip_skip = clip_skip;
    } catch (...) {
    }
    try {
        float cfg_scale   = payload["cfg_scale"];
        params->cfg_scale = cfg_scale;
    } catch (...) {
    }
    try {
        float guidance   = payload["guidance"];
        params->guidance = guidance;
    } catch (...) {
    }
    try {
        int width     = payload["width"];
        params->width = width;
    } catch (...) {
    }
    try {
        int height     = payload["height"];
        params->height = height;
    } catch (...) {
    }
    try {
        std::string sample_method = payload["sample_method"];

        int sample_method_found = -1;
        for (int m = 0; m < N_SAMPLE_METHODS; m++) {
            if (!strcmp(sample_method.c_str(), sample_method_str[m])) {
                sample_method_found = m;
            }
        }
        if (sample_method_found >= 0) {
            params->sample_method = (sample_method_t)sample_method_found;
        } else {
            sd_log(sd_log_level_t::SD_LOG_WARN, "Unknown sampling method: %s\n", sample_method.c_str());
        }
    } catch (...) {
    }
    try {
        int sample_steps     = payload["sample_steps"];
        params->sample_steps = sample_steps;
    } catch (...) {
    }
    try {
        int64_t seed = payload["seed"];
        params->seed = seed;
    } catch (...) {
    }
    try {
        int batch_count     = payload["batch_count"];
        params->batch_count = batch_count;
    } catch (...) {
    }

    try {
        std::string control_cond = payload["control_cond"];
        // TODO map to enum value
        // LOG_WARN("control_cond is not supported yet\n");
        sd_log(sd_log_level_t::SD_LOG_WARN, "control_cond is not supported yet\n");
    } catch (...) {
    }
    try {
        float control_strength = payload["control_strength"];
    } catch (...) {
    }
    try {
        float style_strength = payload["style_strength"];
    } catch (...) {
    }
    try {
        bool normalize_input    = payload["normalize_input"];
        params->normalize_input = normalize_input;
    } catch (...) {
    }
    try {
        std::string input_id_images_path = payload["input_id_images_path"];
        // TODO replace with b64 image maybe?
    } catch (...) {
    }
}

int main(int argc, const char* argv[]) {
    SDParams params;

    parse_args(argc, argv, params);

    sd_set_log_callback(sd_log_cb, (void*)&params);

    server_log_params = (void*)&params;

    if (params.verbose) {
        print_params(params);
        printf("%s", sd_get_system_info());
    }

    bool vae_decode_only = true;

    sd_ctx_t* sd_ctx = new_sd_ctx(params.model_path.c_str(),
                                  params.clip_l_path.c_str(),
                                  params.t5xxl_path.c_str(),
                                  params.diffusion_model_path.c_str(),
                                  params.vae_path.c_str(),
                                  "",
                                  "",
                                  params.lora_model_dir.c_str(),
                                  params.embeddings_path.c_str(),
                                  params.stacked_id_embeddings_path.c_str(),
                                  vae_decode_only,
                                  params.vae_tiling,
                                  false,
                                  params.n_threads,
                                  params.wtype,
                                  params.rng_type,
                                  params.schedule,
                                  params.clip_on_cpu,
                                  true,
                                  params.vae_on_cpu,
								  true
								  );

    if (sd_ctx == NULL) {
        printf("new_sd_ctx_t failed\n");
        return 1;
    }

    int n_prompts = 0;

    const auto txt2imgRequest = [&sd_ctx, &params, &n_prompts](const httplib::Request& req, httplib::Response& res) {
        // LOG_DEBUG("raw body is: %s\n", req.body.c_str());
        sd_log(sd_log_level_t::SD_LOG_DEBUG, "raw body is: %s\n", req.body.c_str());
        // parse req.body as json using jsoncpp
        using json = nlohmann::json;

        try {
            std::string json_str = req.body;
            parseJsonPrompt(json_str, &params);
        } catch (json::parse_error& e) {
            // assume the request is just a prompt
            // LOG_WARN("Failed to parse json: %s\n Assuming it's just a prompt...\n", e.what());
            sd_log(sd_log_level_t::SD_LOG_WARN, "Failed to parse json: %s\n Assuming it's just a prompt...\n", e.what());
            std::string prompt = req.body;
            if (!prompt.empty()) {
                params.prompt = prompt;
            } else {
                params.seed += 1;
            }
        } catch (...) {
            // Handle any other type of exception
            // LOG_ERROR("An unexpected error occurred\n");
            sd_log(sd_log_level_t::SD_LOG_ERROR, "An unexpected error occurred\n");
        }
        // LOG_DEBUG("prompt is: %s\n", params.prompt.c_str());
        sd_log(sd_log_level_t::SD_LOG_DEBUG, "prompt is: %s\n", params.prompt.c_str());

        {
            sd_image_t* results;
            results = txt2img(sd_ctx,
                              params.prompt.c_str(),
                              params.negative_prompt.c_str(),
                              params.clip_skip,
                              params.cfg_scale,
                              params.guidance,
                              params.width,
                              params.height,
                              params.sample_method,
                              params.sample_steps,
                              params.seed,
                              params.batch_count,
                              NULL,
                              1,
                              params.style_ratio,
                              params.normalize_input,
                              "");

            if (results == NULL) {
                printf("generate failed\n");
                free_sd_ctx(sd_ctx);
                return 1;
            }

            size_t last            = params.output_path.find_last_of(".");
            std::string dummy_name = last != std::string::npos ? params.output_path.substr(0, last) : params.output_path;
            json images_json       = json::array();
            for (int i = 0; i < params.batch_count; i++) {
                if (results[i].data == NULL) {
                    continue;
                }
                // TODO allow disable save to disk
                std::string final_image_path = i > 0 ? dummy_name + "_" + std::to_string(i + 1 + n_prompts * params.batch_count) + ".png" : dummy_name + ".png";
                stbi_write_png(final_image_path.c_str(), results[i].width, results[i].height, results[i].channel,
                               results[i].data, 0, get_image_params(params, params.seed + i).c_str());
                printf("save result image to '%s'\n", final_image_path.c_str());
                // Todo: return base64 encoded image via httplib::Response& res

                int len;
                unsigned char* png = stbi_write_png_to_mem((const unsigned char*)results[i].data, 0, results[i].width, results[i].height, results[i].channel, &len, get_image_params(params, params.seed + i).c_str());

                std::string data_str(png, png + len);
                std::string encoded_img = base64_encode(data_str);

                images_json.push_back({{"width", results[i].width},
                                       {"height", results[i].height},
                                       {"channel", results[i].channel},
                                       {"data", encoded_img},
                                       {"encoding", "png"}});

                free(results[i].data);
                results[i].data = NULL;
            }
            free(results);
            n_prompts++;
            res.set_content(images_json.dump(), "application/json");
        }
        return 0;
    };

    std::unique_ptr<httplib::Server> svr;
    svr.reset(new httplib::Server());
    svr->set_default_headers({{"Server", "sd.cpp"}});
    // CORS preflight
    svr->Options(R"(.*)", [](const httplib::Request&, httplib::Response& res) {
        // Access-Control-Allow-Origin is already set by middleware
        res.set_header("Access-Control-Allow-Credentials", "true");
        res.set_header("Access-Control-Allow-Methods", "POST");
        res.set_header("Access-Control-Allow-Headers", "*");
        return res.set_content("", "text/html");  // blank response, no data
    });
    svr->set_logger(log_server_request);

    svr->Post("/txt2img", txt2imgRequest);

    // bind HTTP listen port, run the HTTP server in a thread
    if (!svr->bind_to_port(params.host, params.port)) {
        // TODO: Error message
        return 1;
    }
    std::thread t([&]() { svr->listen_after_bind(); });
    svr->wait_until_ready();

    printf("Server listening at %s:%d\n", params.host.c_str(), params.port);

    t.join();

    free_sd_ctx(sd_ctx);

    return 0;
}
