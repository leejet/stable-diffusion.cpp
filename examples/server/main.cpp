#include <stdio.h>
#include <string.h>
#include <time.h>
#include <filesystem>
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

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>

#include "frontend.cpp"

struct SDCtxParams {
    std::string model_path;
    std::string clip_l_path;
    std::string clip_g_path;
    std::string t5xxl_path;
    std::string diffusion_model_path;
    std::string vae_path;
    std::string taesd_path;

    std::string control_net_path;
    std::string lora_model_dir;
    std::string embeddings_path;
    std::string stacked_id_embeddings_path;

    bool vae_decode_only = false;
    bool vae_tiling      = false;

    int n_threads   = -1;
    sd_type_t wtype = SD_TYPE_COUNT;

    rng_type_t rng_type = CUDA_RNG;
    schedule_t schedule = DEFAULT;

    bool keep_control_net_on_cpu = false;
    bool keep_clip_on_cpu     = false;
    bool keep_vae_on_cpu      = false;

    bool diffusion_flash_attn = false;
};

struct SDRequestParams {
    // TODO set to true if esrgan_path is specified in args
    bool upscale = false;

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
    int sample_steps              = 20;
    float strength                = 0.75f;
    float control_strength        = 0.9f;
    int64_t seed                  = 42;

    std::vector<int> skip_layers = {7, 8, 9};
    float slg_scale              = 0.;
    float skip_layer_start       = 0.01;
    float skip_layer_end         = 0.2;
    bool normalize_input         = false;
};

struct SDParams {
    SDCtxParams ctxParams;
    SDRequestParams lastRequest;

    std::string esrgan_path;

    std::string output_path        = "./server/output.png";
    std::string input_path         = "./server/input.png";
    std::string control_image_path = "./server/control.png";

    std::string models_dir;
    std::string diffusion_models_dir;
    std::string clip_dir;
    std::string vae_dir;
    std::string tae_dir;

    // external dir
    std::string input_id_images_path;

    bool verbose = false;

    bool color = false;

    // server things
    int port         = 8080;
    std::string host = "127.0.0.1";
};

void print_params(SDParams params) {
    printf("Starting Options: \n");
    printf("    n_threads:         %d\n", params.ctxParams.n_threads);
    printf("    mode:              server\n");
    printf("    model_path:        %s\n", params.ctxParams.model_path.c_str());
    printf("    wtype:             %s\n", params.ctxParams.wtype < SD_TYPE_COUNT ? sd_type_name(params.ctxParams.wtype) : "unspecified");
    printf("    clip_l_path:       %s\n", params.ctxParams.clip_l_path.c_str());
    printf("    clip_g_path:       %s\n", params.ctxParams.clip_g_path.c_str());
    printf("    t5xxl_path:        %s\n", params.ctxParams.t5xxl_path.c_str());
    printf("    diffusion_model_path:   %s\n", params.ctxParams.diffusion_model_path.c_str());
    printf("    vae_path:          %s\n", params.ctxParams.vae_path.c_str());
    printf("    taesd_path:        %s\n", params.ctxParams.taesd_path.c_str());
    printf("    control_net_path:   %s\n", params.ctxParams.control_net_path.c_str());
    printf("    embeddings_path:   %s\n", params.ctxParams.embeddings_path.c_str());
    printf("    stacked_id_embeddings_path:   %s\n", params.ctxParams.stacked_id_embeddings_path.c_str());
    printf("    input_id_images_path:   %s\n", params.input_id_images_path.c_str());
    printf("    style ratio:       %.2f\n", params.lastRequest.style_ratio);
    printf("    normalize input image :  %s\n", params.lastRequest.normalize_input ? "true" : "false");
    printf("    output_path:       %s\n", params.output_path.c_str());
    printf("    init_img:          %s\n", params.input_path.c_str());
    printf("    control_image:     %s\n", params.control_image_path.c_str());
    printf("    clip on cpu:       %s\n", params.ctxParams.keep_clip_on_cpu ? "true" : "false");
    printf("    control_net cpu:    %s\n", params.ctxParams.keep_control_net_on_cpu ? "true" : "false");
    printf("    vae decoder on cpu:%s\n", params.ctxParams.keep_vae_on_cpu ? "true" : "false");
    printf("    diffusion flash attention:%s\n", params.ctxParams.diffusion_flash_attn ? "true" : "false");
    printf("    strength(control): %.2f\n", params.lastRequest.control_strength);
    printf("    prompt:            %s\n", params.lastRequest.prompt.c_str());
    printf("    negative_prompt:   %s\n", params.lastRequest.negative_prompt.c_str());
    printf("    min_cfg:           %.2f\n", params.lastRequest.min_cfg);
    printf("    cfg_scale:         %.2f\n", params.lastRequest.cfg_scale);
    printf("    slg_scale:         %.2f\n", params.lastRequest.slg_scale);
    printf("    guidance:          %.2f\n", params.lastRequest.guidance);
    printf("    clip_skip:         %d\n", params.lastRequest.clip_skip);
    printf("    width:             %d\n", params.lastRequest.width);
    printf("    height:            %d\n", params.lastRequest.height);
    printf("    sample_method:     %s\n", sd_sample_method_name(params.lastRequest.sample_method));
    printf("    schedule:          %s\n", sd_schedule_name(params.ctxParams.schedule));
    printf("    sample_steps:      %d\n", params.lastRequest.sample_steps);
    printf("    strength(img2img): %.2f\n", params.lastRequest.strength);
    printf("    rng:               %s\n", sd_rng_type_name(params.ctxParams.rng_type));
    printf("    seed:              %ld\n", params.lastRequest.seed);
    printf("    batch_count:       %d\n", params.lastRequest.batch_count);
    printf("    vae_tiling:        %s\n", params.ctxParams.vae_tiling ? "true" : "false");
}

void print_usage(int argc, const char* argv[]) {
    printf("usage: %s [arguments]\n", argv[0]);
    printf("\n");
    printf("arguments:\n");
    printf("  -h, --help                         show this help message and exit\n");
    printf("  -t, --threads N                    number of threads to use during computation (default: -1)\n");
    printf("                                     If threads <= 0, then threads will be set to the number of CPU physical cores\n");
    printf("  -m, --model [MODEL]                path to full model\n");
    printf("  --diffusion-model                  path to the standalone diffusion model\n");
    printf("  --clip_l                           path to the clip-l text encoder\n");
    printf("  --clip_g                           path to the clip-g text encoder\n");
    printf("  --t5xxl                            path to the the t5xxl text encoder\n");
    printf("  --vae [VAE]                        path to vae\n");
    printf("  --taesd [TAESD_PATH]               path to taesd. Using Tiny AutoEncoder for fast decoding (low quality)\n");
    printf("  --control-net [CONTROL_PATH]       path to control net model\n");
    printf("  --embd-dir [EMBEDDING_PATH]        path to embeddings\n");
    printf("  --stacked-id-embd-dir [DIR]        path to PHOTOMAKER stacked id embeddings\n");
    printf("  --input-id-images-dir [DIR]        path to PHOTOMAKER input id images dir\n");
    printf("  --normalize-input                  normalize PHOTOMAKER input id images\n");
    // printf("  --upscale-model [ESRGAN_PATH]      path to esrgan model. Upscale images after generate, just RealESRGAN_x4plus_anime_6B supported by now\n");
    // printf("  --upscale-repeats                  Run the ESRGAN upscaler this many times (default 1)\n");
    printf("  --type [TYPE]                      weight type (f32, f16, q4_0, q4_1, q5_0, q5_1, q8_0, q2_k, q3_k, q4_k)\n");
    printf("                                     If not specified, the default is the type of the weight file\n");
    printf("  --lora-model-dir [DIR]             lora model directory\n");
    printf("  --control-image [IMAGE]            path to image condition, control net\n");
    printf("  -o, --output OUTPUT                path to write result image to (default: ./output.png)\n");
    printf("  -p, --prompt [PROMPT]              the prompt to render\n");
    printf("  -n, --negative-prompt PROMPT       the negative prompt (default: \"\")\n");
    printf("  --cfg-scale SCALE                  unconditional guidance scale: (default: 7.0)\n");
    printf("  --slg-scale SCALE                  skip layer guidance (SLG) scale, only for DiT models: (default: 0)\n");
    printf("                                     0 means disabled, a value of 2.5 is nice for sd3.5 medium\n");
    printf("  --skip_layers LAYERS               Layers to skip for SLG steps: (default: [7,8,9])\n");
    printf("  --skip_layer_start START           SLG enabling point: (default: 0.01)\n");
    printf("  --skip_layer_end END               SLG disabling point: (default: 0.2)\n");
    printf("                                     SLG will be enabled at step int([STEPS]*[START]) and disabled at int([STEPS]*[END])\n");
    printf("  --strength STRENGTH                strength for noising/unnoising (default: 0.75)\n");
    printf("  --style-ratio STYLE-RATIO          strength for keeping input identity (default: 20%%)\n");
    printf("  --control-strength STRENGTH        strength to apply Control Net (default: 0.9)\n");
    printf("                                     1.0 corresponds to full destruction of information in init image\n");
    printf("  -H, --height H                     image height, in pixel space (default: 512)\n");
    printf("  -W, --width W                      image width, in pixel space (default: 512)\n");
    printf("  --sampling-method {euler, euler_a, heun, dpm2, dpm++2s_a, dpm++2m, dpm++2mv2, ipndm, ipndm_v, lcm}\n");
    printf("                                     sampling method (default: \"euler_a\")\n");
    printf("  --steps  STEPS                     number of sample steps (default: 20)\n");
    printf("  --rng {std_default, cuda}          RNG (default: cuda)\n");
    printf("  -s SEED, --seed SEED               RNG seed (default: 42, use random seed for < 0)\n");
    printf("  -b, --batch-count COUNT            number of images to generate\n");
    printf("  --schedule {discrete, karras, exponential, ays, gits} Denoiser sigma schedule (default: discrete)\n");
    printf("  --clip-skip N                      ignore last layers of CLIP network; 1 ignores none, 2 ignores one layer (default: -1)\n");
    printf("                                     <= 0 represents unspecified, will be 1 for SD1.x, 2 for SD2.x\n");
    printf("  --vae-tiling                       process vae in tiles to reduce memory usage\n");
    printf("  --vae-on-cpu                       keep vae in cpu (for low vram)\n");
    printf("  --clip-on-cpu                      keep clip in cpu (for low vram)\n");
    printf("  --diffusion-fa                     use flash attention in the diffusion model (for low vram)\n");
    printf("                                     Might lower quality, since it implies converting k and v to f16.\n");
    printf("                                     This might crash if it is not supported by the backend.\n");
    printf("  --control-net-cpu                  keep control_net in cpu (for low vram)\n");
    printf("  --canny                            apply canny preprocessor (edge detection)\n");
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
            params.ctxParams.n_threads = std::stoi(argv[i]);
        } else if (arg == "-m" || arg == "--model") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.ctxParams.model_path = argv[i];
        } else if (arg == "--clip_l") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.ctxParams.clip_l_path = argv[i];
        } else if (arg == "--clip_g") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.ctxParams.clip_g_path = argv[i];
        } else if (arg == "--t5xxl") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.ctxParams.t5xxl_path = argv[i];
        } else if (arg == "--diffusion-model") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.ctxParams.diffusion_model_path = argv[i];
        } else if (arg == "--vae") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.ctxParams.vae_path = argv[i];
        } else if (arg == "--taesd") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.ctxParams.taesd_path = argv[i];
        } else if (arg == "--control-net") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.ctxParams.control_net_path = argv[i];
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
            params.ctxParams.embeddings_path = argv[i];
        } else if (arg == "--stacked-id-embd-dir") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.ctxParams.stacked_id_embeddings_path = argv[i];
        } else if (arg == "--input-id-images-dir") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.input_id_images_path = argv[i];
        } else if (arg == "--type") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            std::string type        = argv[i];
            bool found              = false;
            std::string valid_types = "";
            for (size_t i = 0; i < SD_TYPE_COUNT; i++) {
                auto trait = ggml_get_type_traits((ggml_type)i);
                std::string name(trait->type_name);
                if (name == "f32" || trait->to_float && trait->type_size) {
                    if (i)
                        valid_types += ", ";
                    valid_types += name;
                    if (type == name) {
                        if (ggml_quantize_requires_imatrix((ggml_type)i)) {
                            printf("\033[35;1m[WARNING]\033[0m: type %s requires imatrix to work properly. A dummy imatrix will be used, expect poor quality.\n", trait->type_name);
                        }
                        params.ctxParams.wtype = (enum sd_type_t)i;
                        found                  = true;
                        break;
                    }
                }
            }
            if (!found) {
                fprintf(stderr, "error: invalid weight format %s, must be one of [%s]\n",
                        type.c_str(),
                        valid_types.c_str());
                exit(1);
            }
        } else if (arg == "--lora-model-dir") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.ctxParams.lora_model_dir = argv[i];
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
            params.lastRequest.prompt = argv[i];
        } else if (arg == "-n" || arg == "--negative-prompt") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.lastRequest.negative_prompt = argv[i];
        } else if (arg == "--cfg-scale") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.lastRequest.cfg_scale = std::stof(argv[i]);
        } else if (arg == "--guidance") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.lastRequest.guidance = std::stof(argv[i]);
        } else if (arg == "--strength") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.lastRequest.strength = std::stof(argv[i]);
        } else if (arg == "--style-ratio") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.lastRequest.style_ratio = std::stof(argv[i]);
        } else if (arg == "--control-strength") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.lastRequest.control_strength = std::stof(argv[i]);
        } else if (arg == "-H" || arg == "--height") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.lastRequest.height = std::stoi(argv[i]);
        } else if (arg == "-W" || arg == "--width") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.lastRequest.width = std::stoi(argv[i]);
        } else if (arg == "--steps") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.lastRequest.sample_steps = std::stoi(argv[i]);
        } else if (arg == "--clip-skip") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.lastRequest.clip_skip = std::stoi(argv[i]);
        } else if (arg == "--vae-tiling") {
            params.ctxParams.vae_tiling = true;
        } else if (arg == "--control-net-cpu") {
            params.ctxParams.keep_control_net_on_cpu = true;
        } else if (arg == "--normalize-input") {
            params.lastRequest.normalize_input = true;
        } else if (arg == "--clip-on-cpu") {
            params.ctxParams.keep_clip_on_cpu = true;  // will slow down get_learned_condiotion but necessary for low MEM GPUs
        } else if (arg == "--vae-on-cpu") {
            params.ctxParams.keep_vae_on_cpu = true;  // will slow down latent decoding but necessary for low MEM GPUs
        } else if (arg == "--diffusion-fa") {
            params.ctxParams.diffusion_flash_attn = true;  // can reduce MEM significantly
        } else if (arg == "-b" || arg == "--batch-count") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.lastRequest.batch_count = std::stoi(argv[i]);
        } else if (arg == "--rng") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            std::string rng_type_str = argv[i];
            if (rng_type_str == "std_default") {
                params.ctxParams.rng_type = STD_DEFAULT_RNG;
            } else if (rng_type_str == "cuda") {
                params.ctxParams.rng_type = CUDA_RNG;
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
            schedule_t schedule_found     = str_to_schedule(schedule_selected);
            if (schedule_found == SCHEDULE_COUNT) {
                invalid_arg = true;
                break;
            }
            params.ctxParams.schedule = schedule_found;
        } else if (arg == "-s" || arg == "--seed") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.lastRequest.seed = std::stoll(argv[i]);
        } else if (arg == "--sampling-method") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            const char* sample_method_selected = argv[i];
            int sample_method_found            = str_to_sample_method(sample_method_selected);
            if (sample_method_found == SAMPLE_METHOD_COUNT) {
                invalid_arg = true;
                break;
            }
            params.lastRequest.sample_method = (sample_method_t)sample_method_found;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argc, argv);
            exit(0);
        } else if (arg == "-v" || arg == "--verbose") {
            params.verbose = true;
        } else if (arg == "--color") {
            params.color = true;
        } else if (arg == "--slg-scale") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.lastRequest.slg_scale = std::stof(argv[i]);
        } else if (arg == "--skip-layers") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            if (argv[i][0] != '[') {
                invalid_arg = true;
                break;
            }
            std::string layers_str = argv[i];
            while (layers_str.back() != ']') {
                if (++i >= argc) {
                    invalid_arg = true;
                    break;
                }
                layers_str += " " + std::string(argv[i]);
            }
            layers_str = layers_str.substr(1, layers_str.size() - 2);

            std::regex regex("[, ]+");
            std::sregex_token_iterator iter(layers_str.begin(), layers_str.end(), regex, -1);
            std::sregex_token_iterator end;
            std::vector<std::string> tokens(iter, end);
            std::vector<int> layers;
            for (const auto& token : tokens) {
                try {
                    layers.push_back(std::stoi(token));
                } catch (const std::invalid_argument& e) {
                    invalid_arg = true;
                    break;
                }
            }
            params.lastRequest.skip_layers = layers;

            if (invalid_arg) {
                break;
            }
        } else if (arg == "--skip-layer-start") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.lastRequest.skip_layer_start = std::stof(argv[i]);
        } else if (arg == "--skip-layer-end") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.lastRequest.skip_layer_end = std::stof(argv[i]);
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
        } else if (arg == "--models-dir") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.models_dir = argv[i];
        } else if (arg == "--diffusion-models-dir") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.diffusion_models_dir = argv[i];
        } else if (arg == "--encoders-dir") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.clip_dir = argv[i];
        } else if (arg == "--vae-dir") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.vae_dir = argv[i];
        } else if (arg == "--tae-dir") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.tae_dir = argv[i];
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
    if (params.ctxParams.n_threads <= 0) {
        params.ctxParams.n_threads = get_num_physical_cores();
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
    std::string parameter_string = params.lastRequest.prompt + "\n";
    if (params.lastRequest.negative_prompt.size() != 0) {
        parameter_string += "Negative prompt: " + params.lastRequest.negative_prompt + "\n";
    }
    parameter_string += "Steps: " + std::to_string(params.lastRequest.sample_steps) + ", ";
    parameter_string += "CFG scale: " + std::to_string(params.lastRequest.cfg_scale) + ", ";
    parameter_string += "Guidance: " + std::to_string(params.lastRequest.guidance) + ", ";
    parameter_string += "Seed: " + std::to_string(seed) + ", ";
    parameter_string += "Size: " + std::to_string(params.lastRequest.width) + "x" + std::to_string(params.lastRequest.height) + ", ";
    parameter_string += "Model: " + sd_basename(params.ctxParams.model_path) + ", ";
    parameter_string += "RNG: " + std::string(sd_rng_type_name(params.ctxParams.rng_type)) + ", ";
    parameter_string += "Sampler: " + std::string(sd_sample_method_name(params.lastRequest.sample_method));
    if (params.ctxParams.schedule == KARRAS) {
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

bool parseJsonPrompt(std::string json_str, SDParams* params) {
    bool updatectx = false;
    using namespace nlohmann;
    json payload = json::parse(json_str);
    // if no exception, the request is a json object
    // now we try to get the new param values from the payload object
    // const char *prompt, const char *negative_prompt, int clip_skip, float cfg_scale, float guidance, int width, int height, sample_method_t sample_method, int sample_steps, int64_t seed, int batch_count, const sd_image_t *control_cond, float control_strength, float style_strength, bool normalize_input, const char *input_id_images_path
    try {
        std::string prompt         = payload["prompt"];
        params->lastRequest.prompt = prompt;
    } catch (...) {
    }
    try {
        std::string negative_prompt         = payload["negative_prompt"];
        params->lastRequest.negative_prompt = negative_prompt;
    } catch (...) {
    }
    try {
        int clip_skip                 = payload["clip_skip"];
        params->lastRequest.clip_skip = clip_skip;
    } catch (...) {
    }
    try {
        float cfg_scale               = payload["cfg_scale"];
        params->lastRequest.cfg_scale = cfg_scale;
    } catch (...) {
    }
    try {
        float guidance               = payload["guidance"];
        params->lastRequest.guidance = guidance;
    } catch (...) {
    }
    try {
        int width                 = payload["width"];
        params->lastRequest.width = width;
    } catch (...) {
    }
    try {
        int height                 = payload["height"];
        params->lastRequest.height = height;
    } catch (...) {
    }
    try {
        std::string sample_method = payload["sample_method"];

        sample_method_t sample_method_found = str_to_sample_method(sample_method.c_str());
        if (sample_method_found != SAMPLE_METHOD_COUNT) {
            params->lastRequest.sample_method = sample_method_found;
        } else {
            sd_log(sd_log_level_t::SD_LOG_WARN, "Unknown sampling method: %s\n", sample_method.c_str());
        }
    } catch (...) {
    }
    try {
        int sample_steps                 = payload["sample_steps"];
        params->lastRequest.sample_steps = sample_steps;
    } catch (...) {
    }
    try {
        int64_t seed             = payload["seed"];
        params->lastRequest.seed = seed;
    } catch (...) {
    }
    try {
        int batch_count                 = payload["batch_count"];
        params->lastRequest.batch_count = batch_count;
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
        // params->control_strength = control_strength;
        // LOG_WARN("control_strength is not supported yet\n");
        sd_log(sd_log_level_t::SD_LOG_WARN, "control_strength is not supported yet\n", params);
    } catch (...) {
    }
    try {
        float style_strength = payload["style_strength"];
        // params->style_strength = style_strength;
        // LOG_WARN("style_strength is not supported yet\n");
        sd_log(sd_log_level_t::SD_LOG_WARN, "style_strength is not supported yet\n", params);
    } catch (...) {
    }
    try {
        bool normalize_input                = payload["normalize_input"];
        params->lastRequest.normalize_input = normalize_input;
    } catch (...) {
    }
    try {
        std::string input_id_images_path = payload["input_id_images_path"];
        // TODO replace with b64 image maybe?
        params->input_id_images_path = input_id_images_path;
    } catch (...) {
    }

    try {
        bool vae_cpu = payload["keep_vae_on_cpu"];
        if (params->ctxParams.keep_vae_on_cpu != vae_cpu) {
            params->ctxParams.keep_vae_on_cpu = vae_cpu;
            updatectx                    = true;
        }
    } catch (...) {
    }
    try {
        bool clip_cpu = payload["keep_clip_on_cpu"];
        if (params->ctxParams.keep_clip_on_cpu != clip_cpu) {
            params->ctxParams.keep_clip_on_cpu = clip_cpu;
            updatectx                     = true;
        }
    } catch (...) {
    }
    try {
        bool vae_tiling = payload["vae_tiling"];
        if (params->ctxParams.vae_tiling != vae_tiling) {
            params->ctxParams.vae_tiling = vae_tiling;
            updatectx                    = true;
        }
    } catch (...) {
    }

    try {
        std::string model = payload["model"];
        if (model == "none") {
            if (params->ctxParams.model_path != "") {
                updatectx = true;
            }
            params->ctxParams.model_path = "";
        } else {
            std::string new_path = params->models_dir + model;
            if (params->ctxParams.model_path != new_path) {
                params->ctxParams.model_path           = new_path;
                params->ctxParams.diffusion_model_path = "";
                updatectx                              = true;
            }
        }
    } catch (...) {
    }
    try {
        std::string diffusion_model = payload["diffusion_model"];
        if (diffusion_model == "none") {
            if (params->ctxParams.diffusion_model_path != "") {
                updatectx = true;
            }
            params->ctxParams.diffusion_model_path = "";
        } else {
            std::string new_path = params->diffusion_models_dir + diffusion_model;
            if (params->ctxParams.diffusion_model_path != new_path) {
                params->ctxParams.diffusion_model_path = new_path;
                params->ctxParams.model_path           = "";
                updatectx                              = true;
            }
        }
    } catch (...) {
    }
    try {
        std::string clip_l = payload["clip_l"];
        if (clip_l == "none") {
            if (params->ctxParams.clip_l_path != "") {
                updatectx = true;
            }
            params->ctxParams.clip_l_path = "";
        } else {
            std::string new_path = params->clip_dir + clip_l;
            if (params->ctxParams.clip_l_path != new_path) {
                params->ctxParams.clip_l_path = new_path;
                updatectx                     = true;
            }
        }
    } catch (...) {
    }
    try {
        std::string clip_g = payload["clip_g"];
        if (clip_g == "none") {
            if (params->ctxParams.clip_g_path != "") {
                updatectx = true;
            }
            params->ctxParams.clip_g_path = "";
        } else {
            std::string new_path = params->clip_dir + clip_g;
            if (params->ctxParams.clip_g_path != new_path) {
                params->ctxParams.clip_g_path = new_path;
                updatectx                     = true;
            }
        }
    } catch (...) {
    }
    try {
        std::string t5xxl = payload["t5xxl"];
        if (t5xxl == "none") {
            if (params->ctxParams.t5xxl_path != "") {
                updatectx = true;
            }
            params->ctxParams.t5xxl_path = "";
        } else {
            std::string new_path = params->clip_dir + t5xxl;
            if (params->ctxParams.t5xxl_path != new_path) {
                params->ctxParams.t5xxl_path = new_path;
                updatectx                    = true;
            }
        }
    } catch (...) {
    }
    try {
        std::string vae = payload["vae"];
        if (vae == "none") {
            if (params->ctxParams.vae_path != "") {
                updatectx = true;
            }
            params->ctxParams.vae_path = "";
        } else {
            std::string new_path = params->vae_dir + vae;
            if (params->ctxParams.vae_path != new_path) {
                params->ctxParams.vae_path = new_path;
                updatectx                  = true;
            }
        }
    } catch (...) {
    }
    try {
        std::string tae = payload["tae"];
        if (tae == "none") {
            if (params->ctxParams.taesd_path != "") {
                updatectx = true;
            }
            params->ctxParams.taesd_path = "";
        } else {
            std::string new_path = params->tae_dir + tae;
            if (params->ctxParams.taesd_path != new_path) {
                params->ctxParams.taesd_path = new_path;
                updatectx                    = true;
            }
        }
    } catch (...) {
    }

    try {
        std::string schedule      = payload["schedule"];
        schedule_t schedule_found = str_to_schedule(schedule.c_str());
        if (schedule_found != SCHEDULE_COUNT) {
            if (params->ctxParams.schedule != schedule_found) {
                params->ctxParams.schedule = schedule_found;
                updatectx                  = true;
            }
        } else {
            sd_log(sd_log_level_t::SD_LOG_WARN, "Unknown schedule: %s\n", schedule.c_str());
        }
    } catch (...) {
    }

    return updatectx;
}

std::vector<std::string> list_files(const std::string& dir_path) {
    namespace fs = std::filesystem;
    std::vector<std::string> files;
    if (dir_path != "")
        for (const auto& entry : fs::recursive_directory_iterator(dir_path)) {
            if (entry.is_regular_file()) {
                auto relative_path   = fs::relative(entry.path(), dir_path);
                std::string path_str = relative_path.string();
                std::replace(path_str.begin(), path_str.end(), '\\', '/');
                files.push_back(path_str);
            }
        }
    return files;
}

//--------------------------------------//
// Thread-safe queue
std::queue<std::function<void()>> task_queue;
std::mutex queue_mutex;
std::condition_variable queue_cond;
bool stop_worker = false;
std::atomic<bool> is_busy(false);
std::string running_task_id("");

std::unordered_map<std::string, nlohmann::json> task_results;
std::mutex results_mutex;

void worker_thread() {
    while (!stop_worker) {
        std::unique_lock<std::mutex> lock(queue_mutex);
        queue_cond.wait(lock, [] { return !task_queue.empty() || stop_worker; });

        if (!task_queue.empty()) {
            is_busy   = true;
            auto task = task_queue.front();
            task_queue.pop();
            lock.unlock();
            task();
            is_busy         = false;
            running_task_id = "";
        }
    }
}

void add_task(std::string task_id, std::function<void()> task) {
    std::lock_guard<std::mutex> lock(queue_mutex);
    task_queue.push([task_id, task]() {
        task();
    });
    queue_cond.notify_one();
}

void update_progress_cb(int step, int steps, float time, void* _data) {
    using json = nlohmann::json;
    if (running_task_id != "") {
        std::lock_guard<std::mutex> results_lock(results_mutex);
        json running_task_json = task_results[running_task_id];
        if (running_task_json["status"] == "Working" && running_task_json["step"] == running_task_json["steps"]) {
            running_task_json["status"] = "Decoding";
        }
        running_task_json["step"]     = step;
        running_task_json["steps"]    = steps;
        task_results[running_task_id] = running_task_json;
    }
}

void start_server(SDParams params) {
    sd_set_log_callback(sd_log_cb, (void*)&params);
    sd_set_progress_callback(update_progress_cb, NULL);

    server_log_params = (void*)&params;

    if (params.verbose) {
        print_params(params);
        printf("%s", sd_get_system_info());
    }

    sd_ctx_t* sd_ctx = NULL;

    int n_prompts = 0;

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
    if (params.verbose) {
        svr->set_logger(log_server_request);
    }

    svr->Post("/txt2img", [&sd_ctx, &params, &n_prompts](const httplib::Request& req, httplib::Response& res) {
        using json          = nlohmann::json;
        std::string task_id = std::to_string(std::chrono::system_clock::now().time_since_epoch().count());

        {
            json pending_task_json      = json::object();
            pending_task_json["status"] = "Pending";
            pending_task_json["data"]   = json::array();
            pending_task_json["step"]   = -1;
            pending_task_json["steps"]  = 0;
            pending_task_json["eta"]    = "?";

            std::lock_guard<std::mutex> results_lock(results_mutex);
            task_results[task_id] = pending_task_json;
        }

        auto task = [req, &sd_ctx, &params, &n_prompts, task_id]() {
            running_task_id = task_id;
            // LOG_DEBUG("raw body is: %s\n", req.body.c_str());
            sd_log(sd_log_level_t::SD_LOG_DEBUG, "raw body is: %s\n", req.body.c_str());
            // parse req.body as json using jsoncpp
            bool updateCTX = false;
            try {
                std::string json_str = req.body;
                updateCTX            = parseJsonPrompt(json_str, &params);
            } catch (json::parse_error& e) {
                // assume the request is just a prompt
                // LOG_WARN("Failed to parse json: %s\n Assuming it's just a prompt...\n", e.what());
                sd_log(sd_log_level_t::SD_LOG_WARN, "Failed to parse json: %s\n Assuming it's just a prompt...\n", e.what());
                std::string prompt = req.body;
                if (!prompt.empty()) {
                    params.lastRequest.prompt = prompt;
                } else {
                    params.lastRequest.seed += 1;
                }
            } catch (...) {
                // Handle any other type of exception
                // LOG_ERROR("An unexpected error occurred\n");
                sd_log(sd_log_level_t::SD_LOG_ERROR, "An unexpected error occurred\n");
            }
            // LOG_DEBUG("prompt is: %s\n", params.prompt.c_str());
            sd_log(sd_log_level_t::SD_LOG_INFO, "prompt is: %s\n", params.lastRequest.prompt.c_str());

            if (updateCTX && sd_ctx != NULL) {
                free_sd_ctx(sd_ctx);
                sd_ctx = NULL;
            }

            if (sd_ctx == NULL) {
                printf("Loading sd_ctx\n");
                {
                    json task_json      = json::object();
                    task_json["status"] = "Loading";
                    task_json["data"]   = json::array();
                    task_json["step"]   = -1;
                    task_json["steps"]  = 0;
                    task_json["eta"]    = "?";

                    std::lock_guard<std::mutex> results_lock(results_mutex);
                    task_results[task_id] = task_json;
                }
                sd_ctx_params_t sd_ctx_params = {
                    params.ctxParams.model_path.c_str(),
                    params.ctxParams.clip_l_path.c_str(),
                    params.ctxParams.clip_g_path.c_str(),
                    params.ctxParams.t5xxl_path.c_str(),
                    params.ctxParams.diffusion_model_path.c_str(),
                    params.ctxParams.vae_path.c_str(),
                    params.ctxParams.taesd_path.c_str(),
                    params.ctxParams.control_net_path.c_str(),
                    params.ctxParams.lora_model_dir.c_str(),
                    params.ctxParams.embeddings_path.c_str(),
                    params.ctxParams.stacked_id_embeddings_path.c_str(),
                    params.ctxParams.vae_decode_only,
                    params.ctxParams.vae_tiling,
                    false,
                    params.ctxParams.n_threads,
                    params.ctxParams.wtype,
                    params.ctxParams.rng_type,
                    params.ctxParams.schedule,
                    params.ctxParams.keep_clip_on_cpu,
                    params.ctxParams.keep_control_net_on_cpu,
                    params.ctxParams.keep_vae_on_cpu,
                    params.ctxParams.diffusion_flash_attn,
                    true, false, 1};
                sd_ctx = new_sd_ctx(&sd_ctx_params);
                if (sd_ctx == NULL) {
                    printf("new_sd_ctx_t failed\n");
                    std::lock_guard<std::mutex> results_lock(results_mutex);
                    task_results[task_id]["status"] = "Failed";
                    return;
                }
            }

            {
                json started_task_json      = json::object();
                started_task_json["status"] = "Working";
                started_task_json["data"]   = json::array();
                started_task_json["step"]   = 0;
                started_task_json["steps"]  = params.lastRequest.sample_steps;
                started_task_json["eta"]    = "?";

                std::lock_guard<std::mutex> results_lock(results_mutex);
                task_results[task_id] = started_task_json;
            }

            {
                sd_image_t* results;
                sd_slg_params_t slg = {
                    params.lastRequest.skip_layers.data(),
                    params.lastRequest.skip_layers.size(),
                    params.lastRequest.skip_layer_start,
                    params.lastRequest.skip_layer_end,
                    params.lastRequest.slg_scale};
                sd_guidance_params_t guidance = {
                    params.lastRequest.cfg_scale,
                    params.lastRequest.cfg_scale,
                    params.lastRequest.cfg_scale,
                    params.lastRequest.guidance,
                    slg};
                sd_image_t input_image = {
                    (uint32_t)params.lastRequest.width,
                    (uint32_t)params.lastRequest.height,
                    3,
                    NULL};
                sd_image_t mask_img            = input_image;
                sd_img_gen_params_t gen_params = {
                    params.lastRequest.prompt.c_str(),
                    params.lastRequest.negative_prompt.c_str(),
                    params.lastRequest.clip_skip,
                    guidance,
                    input_image,
                    NULL,  // ref images
                    0,     // ref images count
                    mask_img,
                    params.lastRequest.width,
                    params.lastRequest.height,
                    params.lastRequest.sample_method,
                    params.lastRequest.sample_steps,
                    0.f,  // eta
                    1.f,  // denoise strength
                    params.lastRequest.seed,
                    params.lastRequest.batch_count,
                    NULL,  // control image ptr
                    1.f,   // control strength
                    params.lastRequest.style_ratio,
                    params.lastRequest.normalize_input,
                    params.input_id_images_path.c_str()};
                results = generate_image(sd_ctx, &gen_params);

                if (results == NULL) {
                    printf("generate failed\n");
                    free_sd_ctx(sd_ctx);
                    std::lock_guard<std::mutex> results_lock(results_mutex);
                    task_results[task_id]["status"] = "Failed";
                    return;
                }

                size_t last            = params.output_path.find_last_of(".");
                std::string dummy_name = last != std::string::npos ? params.output_path.substr(0, last) : params.output_path;
                json images_json       = json::array();
                for (int i = 0; i < params.lastRequest.batch_count; i++) {
                    if (results[i].data == NULL) {
                        continue;
                    }
                    // TODO allow disable save to disk
                    std::string final_image_path = i > 0 ? dummy_name + "_" + std::to_string(i + 1 + n_prompts * params.lastRequest.batch_count) + ".png" : dummy_name + ".png";
                    stbi_write_png(final_image_path.c_str(), results[i].width, results[i].height, results[i].channel,
                                   results[i].data, 0, get_image_params(params, params.lastRequest.seed + i).c_str());
                    printf("save result image to '%s'\n", final_image_path.c_str());
                    // Todo: return base64 encoded image via httplib::Response& res

                    int len;
                    unsigned char* png = stbi_write_png_to_mem((const unsigned char*)results[i].data, 0, results[i].width, results[i].height, results[i].channel, &len, get_image_params(params, params.lastRequest.seed + i).c_str());

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
                // res.set_content(images_json.dump(), "application/json");
                json end_task_json      = json::object();
                end_task_json["status"] = "Done";
                end_task_json["data"]   = images_json;
                end_task_json["step"]   = -1;
                end_task_json["steps"]  = 0;
                end_task_json["eta"]    = "?";
                std::lock_guard<std::mutex> results_lock(results_mutex);
                task_results[task_id] = end_task_json;
            }
        };
        // Add the task to the queue
        add_task(task_id, task);

        json response       = json::object();
        response["task_id"] = task_id;
        res.set_content(response.dump(), "application/json");
    });

    svr->Get("/params", [&params](const httplib::Request& req, httplib::Response& res) {
        using json = nlohmann::json;
        json response;
        json params_json               = json::object();
        params_json["prompt"]          = params.lastRequest.prompt;
        params_json["negative_prompt"] = params.lastRequest.negative_prompt;
        params_json["clip_skip"]       = params.lastRequest.clip_skip;
        params_json["cfg_scale"]       = params.lastRequest.cfg_scale;
        params_json["guidance"]        = params.lastRequest.guidance;
        params_json["width"]           = params.lastRequest.width;
        params_json["height"]          = params.lastRequest.height;
        params_json["sample_method"]   = sd_sample_method_name(params.lastRequest.sample_method);
        params_json["sample_steps"]    = params.lastRequest.sample_steps;
        params_json["seed"]            = params.lastRequest.seed;
        params_json["batch_count"]     = params.lastRequest.batch_count;
        params_json["normalize_input"] = params.lastRequest.normalize_input;
        // params_json["input_id_images_path"] = params.input_id_images_path;

        json context_params = json::object();
        // Do not expose paths
        // context_params["model_path"] = params.ctxParams.model_path;
        // context_params["clip_l_path"] = params.ctxParams.clip_l_path;
        // context_params["clip_g_path"] = params.ctxParams.clip_g_path;
        // context_params["t5xxl_path"] = params.ctxParams.t5xxl_path;
        // context_params["diffusion_model_path"] = params.ctxParams.diffusion_model_path;
        // context_params["vae_path"] = params.ctxParams.vae_path;
        // context_params["control_net_path"] = params.ctxParams.control_net_path;
        context_params["lora_model_dir"] = params.ctxParams.lora_model_dir;
        // context_params["embeddings_path"] = params.ctxParams.embeddings_path;
        // context_params["stacked_id_embeddings_path"] = params.ctxParams.stacked_id_embeddings_path;
        context_params["vae_decode_only"]      = params.ctxParams.vae_decode_only;
        context_params["vae_tiling"]           = params.ctxParams.vae_tiling;
        context_params["n_threads"]            = params.ctxParams.n_threads;
        context_params["wtype"]                = params.ctxParams.wtype;
        context_params["rng_type"]             = params.ctxParams.rng_type;
        context_params["schedule"]             = sd_schedule_name(params.ctxParams.schedule);
        context_params["keep_clip_on_cpu"]          = params.ctxParams.keep_clip_on_cpu;
        context_params["keep_control_net_on_cpu"]      = params.ctxParams.keep_control_net_on_cpu;
        context_params["keep_vae_on_cpu"]           = params.ctxParams.keep_vae_on_cpu;
        context_params["diffusion_flash_attn"] = params.ctxParams.diffusion_flash_attn;

        // response["taesd_preview"]       = params.taesd_preview;
        // params_json["preview_method"]   = previews_str[params.lastRequest.preview_method];
        // params_json["preview_interval"] = params.lastRequest.preview_interval;

        response["generation_params"] = params_json;
        response["context_params"]    = context_params;
        res.set_content(response.dump(), "application/json");
    });

    svr->Get("/result", [](const httplib::Request& req, httplib::Response& res) {
        using json = nlohmann::json;
        // Parse task ID from query parameters
        try {
            std::string task_id = req.get_param_value("task_id");
            std::lock_guard<std::mutex> lock(results_mutex);
            if (task_results.find(task_id) != task_results.end()) {
                json result = task_results[task_id];
                res.set_content(result.dump(), "application/json");
                // Erase data after sending
                result["data"]        = json::array();
                task_results[task_id] = result;
            } else {
                res.set_content("Cannot find task " + task_id + " in queue", "text/plain");
            }
        } catch (...) {
            sd_log(sd_log_level_t::SD_LOG_WARN, "Error when fetching result");
        }
    });

    svr->Get("/sample_methods", [](const httplib::Request& req, httplib::Response& res) {
        using json = nlohmann::json;
        json response;
        for (int m = 0; m < SAMPLE_METHOD_COUNT; m++) {
            response.push_back(sd_sample_method_name((sample_method_t)m));
        }
        res.set_content(response.dump(), "application/json");
    });

    svr->Get("/schedules", [](const httplib::Request& req, httplib::Response& res) {
        using json = nlohmann::json;
        json response;
        for (int s = 0; s < SCHEDULE_COUNT; s++) {
            response.push_back(sd_schedule_name((schedule_t)s));
        }
        res.set_content(response.dump(), "application/json");
    });

    svr->Get("/previews", [](const httplib::Request& req, httplib::Response& res) {
        using json = nlohmann::json;
        json response;
        // unsupported
        // for (int s = 0; s < N_PREVIEWS; s++) {
        //     response.push_back(previews_str[s]);
        // }
        res.set_content(response.dump(), "application/json");
    });

    svr->Get("/models", [&params](const httplib::Request& req, httplib::Response& res) {
        using json = nlohmann::json;
        json response;
        response["models"] = json::array();
        for (auto file : list_files(params.models_dir)) {
            response["models"].push_back(file);
        }

        response["diffusion_models"] = json::array();
        for (auto file : list_files(params.diffusion_models_dir)) {
            response["diffusion_models"].push_back(file);
        }

        response["text_encoders"] = json::array();
        for (auto file : list_files(params.clip_dir)) {
            response["text_encoders"].push_back(file);
        }

        response["vaes"] = json::array();
        for (auto file : list_files(params.vae_dir)) {
            response["vaes"].push_back(file);
        }

        response["taes"] = json::array();
        for (auto file : list_files(params.tae_dir)) {
            response["taes"].push_back(file);
        }

        res.set_content(response.dump(), "application/json");
    });
    svr->Get("/model", [&params](const httplib::Request& req, httplib::Response& res) {
        using json = nlohmann::json;
        json response;
        if (!params.ctxParams.model_path.empty()) {
            response["model"] = sd_basename(params.ctxParams.model_path);
        }
        if (!params.ctxParams.diffusion_model_path.empty()) {
            response["diffusion_model"] = sd_basename(params.ctxParams.diffusion_model_path);
        }

        if (!params.ctxParams.clip_l_path.empty()) {
            response["clip_l"] = sd_basename(params.ctxParams.clip_l_path);
        }
        if (!params.ctxParams.clip_g_path.empty()) {
            response["clip_g"] = sd_basename(params.ctxParams.clip_g_path);
        }
        if (!params.ctxParams.t5xxl_path.empty()) {
            response["t5xxl"] = sd_basename(params.ctxParams.t5xxl_path);
        }

        if (!params.ctxParams.vae_path.empty()) {
            response["vae"] = sd_basename(params.ctxParams.vae_path);
        }
        if (!params.ctxParams.taesd_path.empty()) {
            response["tae"] = sd_basename(params.ctxParams.taesd_path);
        }
        res.set_content(response.dump(), "application/json");
    });

    svr->Get("/index.html", [](const httplib::Request& req, httplib::Response& res) {
        try {
            res.set_content(html_content, "text/html");
        } catch (const std::exception& e) {
            res.set_content("Error loading page", "text/plain");
        }
    });

    // bind HTTP listen port, run the HTTP server in a thread
    if (!svr->bind_to_port(params.host, params.port)) {
        // TODO: Error message
        return;
    }
    std::thread t([&]() { svr->listen_after_bind(); });
    svr->wait_until_ready();

    printf("Server listening at %s:%d\n", params.host.c_str(), params.port);

    t.join();

    free_sd_ctx(sd_ctx);
}

int main(int argc, const char* argv[]) {
    SDParams params;
    // Setup default args
    parse_args(argc, argv, params);

    std::thread worker(worker_thread);
    // Start the HTTP server
    start_server(params);

    // Cleanup
    stop_worker = true;
    queue_cond.notify_one();
    worker.join();

    return 0;
}