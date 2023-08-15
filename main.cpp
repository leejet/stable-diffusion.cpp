#include <stdio.h>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <unordered_set>

#include "stable-diffusion.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include "stb_image_write.h"

#if defined(__APPLE__) && defined(__MACH__)
#include <sys/sysctl.h>
#include <sys/types.h>
#endif

#if !defined(_WIN32)
#include <sys/ioctl.h>
#include <unistd.h>
#endif

#define TXT2IMG "txt2img"
#define IMG2IMG "img2img"

// get_num_physical_cores is copy from 
// https://github.com/ggerganov/llama.cpp/blob/master/examples/common.cpp
// LICENSE: https://github.com/ggerganov/llama.cpp/blob/master/LICENSE
int32_t get_num_physical_cores() {
#ifdef __linux__
    // enumerate the set of thread siblings, num entries is num cores
    std::unordered_set<std::string> siblings;
    for (uint32_t cpu=0; cpu < UINT32_MAX; ++cpu) {
        std::ifstream thread_siblings("/sys/devices/system/cpu"
            + std::to_string(cpu) + "/topology/thread_siblings");
        if (!thread_siblings.is_open()) {
            break; // no more cpus
        }
        std::string line;
        if (std::getline(thread_siblings, line)) {
            siblings.insert(line);
        }
    }
    if (siblings.size() > 0) {
        return static_cast<int32_t>(siblings.size());
    }
#elif defined(__APPLE__) && defined(__MACH__)
    int32_t num_physical_cores;
    size_t len = sizeof(num_physical_cores);
    int result = sysctlbyname("hw.perflevel0.physicalcpu", &num_physical_cores, &len, NULL, 0);
    if (result == 0) {
        return num_physical_cores;
    }
    result = sysctlbyname("hw.physicalcpu", &num_physical_cores, &len, NULL, 0);
    if (result == 0) {
        return num_physical_cores;
    }
#elif defined(_WIN32)
    //TODO: Implement
#endif
    unsigned int n_threads = std::thread::hardware_concurrency();
    return n_threads > 0 ? (n_threads <= 4 ? n_threads : n_threads / 2) : 4;
}

struct Option {
    int n_threads = -1;
    std::string mode = TXT2IMG;
    std::string model_path;
    std::string output_path = "output.png";
    std::string init_img;
    std::string prompt;
    std::string negative_prompt;
    float cfg_scale = 7.0f;
    int w = 512;
    int h = 512;
    SampleMethod sample_method = EULAR_A;
    int sample_steps = 20;
    float strength = 0.75f;
    int seed = 42;
    bool verbose = false;

    void print() {
        printf("Option: \n");
        printf("    n_threads:       %d\n", n_threads);
        printf("    mode:            %s\n", mode.c_str());
        printf("    model_path:      %s\n", model_path.c_str());
        printf("    output_path:     %s\n", output_path.c_str());
        printf("    init_img:        %s\n", init_img.c_str());
        printf("    prompt:          %s\n", prompt.c_str());
        printf("    negative_prompt: %s\n", negative_prompt.c_str());
        printf("    cfg_scale:       %.2f\n", cfg_scale);
        printf("    width:           %d\n", w);
        printf("    height:          %d\n", h);
        printf("    sample_method:   %s\n", "eular a");
        printf("    sample_steps:    %d\n", sample_steps);
        printf("    strength:        %.2f\n", strength);
        printf("    seed:            %d\n", seed);
    }
};

void print_usage(int argc, const char* argv[]) {
    printf("usage: %s [arguments]\n", argv[0]);
    printf("\n");
    printf("arguments:\n");
    printf("  -h, --help                         show this help message and exit\n");
    printf("  -M, --mode [txt2img or img2img]    generation mode (default: txt2img)\n");
    printf("  -t, --threads N                    number of threads to use during computation (default: -1).\n");
    printf("                                     If threads <= 0, then threads will be set to the number of CPU physical cores\n");
    printf("  -m, --model [MODEL]                path to model\n");
    printf("  -i, --init-img [IMAGE]             path to the input image, required by img2img\n");
    printf("  -o, --output OUTPUT                path to write result image to (default: .\\output.png)\n");
    printf("  -p, --prompt [PROMPT]              the prompt to render\n");
    printf("  -n, --negative-prompt PROMPT       the negative prompt (default: \"\")\n");
    printf("  --cfg-scale SCALE                  unconditional guidance scale: (default: 7.0)\n");
    printf("  --strength STRENGTH                strength for noising/unnoising (default: 0.75)\n");
    printf("                                     1.0 corresponds to full destruction of information in init image\n");
    printf("  -H, --height H                     image height, in pixel space (default: 512)\n");
    printf("  -W, --width W                      image width, in pixel space (default: 512)\n");
    printf("  --sample-method SAMPLE_METHOD      sample method (default: \"eular a\")\n");
    printf("  --steps  STEPS                     number of sample steps (default: 20)\n");
    printf("  -s SEED, --seed SEED               RNG seed (default: 42, use random seed for < 0)\n");
    printf("  -v, --verbose                      print extra info\n");
}

void parse_args(int argc, const char* argv[], Option* opt) {
    bool invalid_arg = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-t" || arg == "--threads") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            opt->n_threads = std::stoi(argv[i]);
        } else if (arg == "-M" || arg == "--mode") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            opt->mode = argv[i];

        } else if (arg == "-m" || arg == "--model") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            opt->model_path = argv[i];
        } else if (arg == "-i" || arg == "--init-img") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            opt->init_img = argv[i];
        } else if (arg == "-o" || arg == "--output") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            opt->output_path = argv[i];
        } else if (arg == "-p" || arg == "--prompt") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            opt->prompt = argv[i];
        } else if (arg == "-n" || arg == "--negative-prompt") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            opt->negative_prompt = argv[i];
        } else if (arg == "--cfg-scale") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            opt->cfg_scale = std::stof(argv[i]);
        } else if (arg == "--strength") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            opt->strength = std::stof(argv[i]);
        } else if (arg == "-H" || arg == "--height") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            opt->h = std::stoi(argv[i]);
        } else if (arg == "-W" || arg == "--width") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            opt->w = std::stoi(argv[i]);
        } else if (arg == "--steps") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            opt->sample_steps = std::stoi(argv[i]);
        } else if (arg == "-s" || arg == "--seed") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            opt->seed = std::stoi(argv[i]);
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argc, argv);
            exit(0);
        } else if (arg == "-v" || arg == "--verbose") {
            opt->verbose = true;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            print_usage(argc, argv);
            exit(1);
        }
        if (invalid_arg) {
            fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
            print_usage(argc, argv);
            exit(1);
        }
    }

    if (opt->n_threads <= 0) {
        opt->n_threads = get_num_physical_cores();
    }

    if (opt->mode != TXT2IMG && opt->mode != IMG2IMG) {
        fprintf(stderr, "error: invalid mode %s, must be one of ['%s', '%s']\n",
                opt->mode.c_str(), TXT2IMG, IMG2IMG);
        exit(1);
    }

    if (opt->prompt.length() == 0) {
        fprintf(stderr, "error: the following arguments are required: prompt\n");
        print_usage(argc, argv);
        exit(1);
    }

    if (opt->model_path.length() == 0) {
        fprintf(stderr, "error: the following arguments are required: model_path\n");
        print_usage(argc, argv);
        exit(1);
    }

    if (opt->mode == IMG2IMG && opt->init_img.length() == 0) {
        fprintf(stderr, "error: when using the img2img mode, the following arguments are required: init-img\n");
        print_usage(argc, argv);
        exit(1);
    }

    if (opt->output_path.length() == 0) {
        fprintf(stderr, "error: the following arguments are required: output_path\n");
        print_usage(argc, argv);
        exit(1);
    }

    if (opt->w <= 0 || opt->w % 32 != 0) {
        fprintf(stderr, "error: the width must be a multiple of 32\n");
        exit(1);
    }

    if (opt->h <= 0 || opt->h % 32 != 0) {
        fprintf(stderr, "error: the height must be a multiple of 32\n");
        exit(1);
    }

    if (opt->sample_steps <= 0) {
        fprintf(stderr, "error: the sample_steps must be greater than 0\n");
        exit(1);
    }

    if (opt->strength < 0.f || opt->strength > 1.f) {
        fprintf(stderr, "error: can only work with strength in [0.0, 1.0]\n");
        exit(1);
    }
}

int main(int argc, const char* argv[]) {
    Option opt;
    parse_args(argc, argv, &opt);

    if (opt.verbose) {
        opt.print();
        printf("%s", sd_get_system_info().c_str());
        set_sd_log_level(SDLogLevel::DEBUG);
    }

    bool vae_decode_only = true;
    std::vector<uint8_t> init_img;
    if (opt.mode == IMG2IMG) {
        vae_decode_only = false;

        int c = 0;
        unsigned char* img_data = stbi_load(opt.init_img.c_str(), &opt.w, &opt.h, &c, 3);
        if (img_data == NULL) {
            fprintf(stderr, "load image from '%s' failed\n", opt.init_img.c_str());
            return 1;
        }
        if (c != 3) {
            fprintf(stderr, "input image must be a 3 channels RGB image, but got %d channels\n", c);
            free(img_data);
            return 1;
        }
        if (opt.w <= 0 || opt.w % 32 != 0) {
            fprintf(stderr, "error: the width of image must be a multiple of 32\n");
            free(img_data);
            return 1;
        }
        if (opt.h <= 0 || opt.h % 32 != 0) {
            fprintf(stderr, "error: the height of image must be a multiple of 32\n");
            free(img_data);
            return 1;
        }
        init_img.assign(img_data, img_data + (opt.w * opt.h * c));
    }
    StableDiffusion sd(opt.n_threads, vae_decode_only);
    if (!sd.load_from_file(opt.model_path)) {
        return 1;
    }

    std::vector<uint8_t> img;
    if (opt.mode == TXT2IMG) {
        img = sd.txt2img(opt.prompt,
                         opt.negative_prompt,
                         opt.cfg_scale,
                         opt.w,
                         opt.h,
                         opt.sample_method,
                         opt.sample_steps,
                         opt.seed);
    } else {
        img = sd.img2img(init_img,
                         opt.prompt,
                         opt.negative_prompt,
                         opt.cfg_scale,
                         opt.w,
                         opt.h,
                         opt.sample_method,
                         opt.sample_steps,
                         opt.strength,
                         opt.seed);
    }

    if (img.size() == 0) {
        fprintf(stderr, "generate failed\n");
        return 1;
    }

    stbi_write_png(opt.output_path.c_str(), opt.w, opt.h, 3, img.data(), 0);
    printf("save result image to '%s'\n", opt.output_path.c_str());

    return 0;
}