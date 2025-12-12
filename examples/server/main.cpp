#include <stdio.h>
#include <string.h>
#include <time.h>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <fstream>
#include <unordered_set>

#ifdef _WIN32
#include <process.h>
#else
#include <sys/wait.h>
#include <unistd.h>
#endif

// #include "preprocessing.hpp"
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
    // sd_ctx_params_t but with strings
    std::string model_path;
    std::string clip_l_path;
    std::string clip_g_path;
    std::string clip_vision_path;
    std::string t5xxl_path;
    std::string llm_path;
    std::string llm_vision_path;
    std::string diffusion_model_path;
    std::string high_noise_diffusion_model_path;
    std::string vae_path;
    std::string taesd_path;

    std::string control_net_path;
    std::string lora_model_dir;
    std::string embeddings_path;
    std::string photo_maker_path;
    std::string tensor_type_rules;
    std::string upscale_model_path;

    bool vae_decode_only         = false;  // Does it ever make sense to set it to true?
    bool free_params_immediately = false;  // has to be false for server too keep ctx alive between prompts
    int n_threads                = -1;
    sd_type_t wtype              = SD_TYPE_COUNT;

    rng_type_t rng_type               = CUDA_RNG;
    rng_type_t sampler_rng_type       = CUDA_RNG;
    prediction_t prediction           = PREDICTION_COUNT;
    lora_apply_mode_t lora_apply_mode = LORA_APPLY_AUTO;

    bool offload_params_to_cpu   = false;
    bool keep_clip_on_cpu        = false;
    bool keep_control_net_on_cpu = false;
    bool keep_vae_on_cpu         = false;

    bool diffusion_flash_attn = false;
    // Don't use TAE decoding by default
    bool taesd_preview             = true;
    bool diffusion_conv_direct     = false;
    bool vae_conv_direct           = false;
    bool force_sdxl_vae_conv_scale = false;
    bool chroma_use_dit_mask       = true;
    bool chroma_use_t5_mask        = false;
    int chroma_t5_mask_pad         = 1;
    float flow_shift               = INFINITY;  // inf means auto
};

struct SDRequestParams {
    std::string prompt;
    std::string negative_prompt;
    int clip_skip = -1;  // <= 0 represents unspecified

    sd_image_t init_image              = {512, 512, 3, NULL};
    std::vector<sd_image_t> ref_images = {};
    bool auto_resize_ref_image         = true;
    bool increase_ref_index            = false;

    sd_image_t mask_image = {512, 512, 1, NULL};

    int width  = 512;
    int height = 512;

    // skip_layers should be turned into an array of ints at the beginning of sd_slg_params
    std::vector<int> skip_layers     = {7, 8, 9};
    sd_sample_params_t sample_params = {
        sd_guidance_params_t{
            7,
            1,
            3.5,
            sd_slg_params_t{NULL, 0, 0.01, 0.2, 0}},
        SCHEDULER_COUNT,
        SAMPLE_METHOD_COUNT,
        20,
        0, 0};
    float strength  = 1.f;
    int64_t seed    = 42;
    int batch_count = 1;

    sd_image_t control_image = {512, 512, 3, NULL};
    float control_strength   = 0.9f;

    // pm_images_vec should be turned into a vector of sd_image_t at the beginning of sd_pm_params
    std::vector<sd_image_t> pm_images_vec = {};
    // same but a char ptr
    std::string pm_id_embed_path     = "";
    sd_pm_params_t pm_params         = {NULL, 0, NULL, 20.f};
    sd_tiling_params_t tiling_params = {false, 0, 0, 0.5f, 0.0f, 0.0f};

    // sd_img_gen_params_t covered, extras below
    // TODO set to true if esrgan_path is specified in args and upscale in request
    bool upscale = false;

    // float apg_eta            = 1.0f;
    // float apg_momentum       = 0.0f;
    // float apg_norm_threshold = 0.0f;
    // float apg_norm_smoothing = 0.0f;

    preview_t preview_method = PREVIEW_NONE;
    int preview_interval     = 1;
    bool preview_noisy       = false;
};

struct SDParams {
    SDCtxParams ctxParams;
    SDRequestParams lastRequest;

    std::string esrgan_path;

    std::string output_path        = "./server/output.png";
    std::string input_path         = "./server/input.png";
    std::string control_image_path = "./server/control.png";

    std::string preview_path = "./server/preview.png";

    std::string models_dir;
    std::string diffusion_models_dir;
    std::string clip_dir;
    std::string clip_vision_dir;
    std::string vae_dir;
    std::string tae_dir;
    std::string controlnet_dir;
    std::string photomaker_dir;
    std::string upscaler_dir;

    std::vector<std::string> models_files;
    std::vector<std::string> diffusion_models_files;
    std::vector<std::string> clip_files;
    std::vector<std::string> clip_vision_files;
    std::vector<std::string> vae_files;
    std::vector<std::string> tae_files;
    std::vector<std::string> controlnet_files;
    std::vector<std::string> photomaker_files;
    std::vector<std::string> upscaler_files;

    // external dir
    std::string input_id_images_path;

    bool verbose = false;

    bool color = false;

    // server things
    int port         = 8080;
    std::string host = "127.0.0.1";

    std::string custom_frontend_path = "";

    bool restore_state = false;
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
    printf("    photo_maker_path:   %s\n", params.ctxParams.photo_maker_path.c_str());
    printf("    input_id_images_path:   %s\n", params.input_id_images_path.c_str());
    printf("    style ratio:       %.2f\n", params.lastRequest.pm_params.style_strength);
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
    printf("    cfg_scale:         %.2f\n", params.lastRequest.sample_params.guidance.txt_cfg);
    printf("    slg_scale:         %.2f\n", params.lastRequest.sample_params.guidance.slg.scale);
    printf("    guidance:          %.2f\n", params.lastRequest.sample_params.guidance.distilled_guidance);
    printf("    clip_skip:         %d\n", params.lastRequest.clip_skip);
    printf("    width:             %d\n", params.lastRequest.width);
    printf("    height:            %d\n", params.lastRequest.height);
    printf("    sample_method:     %s\n", sd_sample_method_name(params.lastRequest.sample_params.sample_method));
    printf("    schedule:          %s\n", sd_scheduler_name(params.lastRequest.sample_params.scheduler));
    printf("    sample_steps:      %d\n", params.lastRequest.sample_params.sample_steps);
    printf("    strength(img2img): %.2f\n", params.lastRequest.strength);
    printf("    rng:               %s\n", sd_rng_type_name(params.ctxParams.rng_type));
    printf("    seed:              %ld\n", params.lastRequest.seed);
    printf("    batch_count:       %d\n", params.lastRequest.batch_count);
    printf("    vae_tiling:        %s\n", params.lastRequest.tiling_params.enabled ? "true" : "false");
}

#if defined(_WIN32)
static std::string utf16_to_utf8(const std::wstring& wstr) {
    if (wstr.empty())
        return {};
    int size_needed = WideCharToMultiByte(CP_UTF8, 0, wstr.data(), (int)wstr.size(),
                                          nullptr, 0, nullptr, nullptr);
    if (size_needed <= 0)
        throw std::runtime_error("UTF-16 to UTF-8 conversion failed");

    std::string utf8(size_needed, 0);
    WideCharToMultiByte(CP_UTF8, 0, wstr.data(), (int)wstr.size(),
                        (char*)utf8.data(), size_needed, nullptr, nullptr);
    return utf8;
}

static std::string argv_to_utf8(int index, const char** argv) {
    int argc;
    wchar_t** argv_w = CommandLineToArgvW(GetCommandLineW(), &argc);
    if (!argv_w)
        throw std::runtime_error("Failed to parse command line");

    std::string result;
    if (index < argc) {
        result = utf16_to_utf8(argv_w[index]);
    }
    LocalFree(argv_w);
    return result;
}

#else  // Linux / macOS
static std::string argv_to_utf8(int index, const char** argv) {
    return std::string(argv[index]);
}

#endif

struct StringOption {
    std::string short_name;
    std::string long_name;
    std::string desc;
    std::string* target;
};

struct IntOption {
    std::string short_name;
    std::string long_name;
    std::string desc;
    int* target;
};

struct FloatOption {
    std::string short_name;
    std::string long_name;
    std::string desc;
    float* target;
};

struct BoolOption {
    std::string short_name;
    std::string long_name;
    std::string desc;
    bool keep_true;
    bool* target;
};

struct ManualOption {
    std::string short_name;
    std::string long_name;
    std::string desc;
    std::function<int(int argc, const char** argv, int index)> cb;
};

struct ArgOptions {
    std::vector<StringOption> string_options;
    std::vector<IntOption> int_options;
    std::vector<FloatOption> float_options;
    std::vector<BoolOption> bool_options;
    std::vector<ManualOption> manual_options;
};

bool parse_options(int argc, const char** argv, ArgOptions& options) {
    bool invalid_arg = false;
    std::string arg;
    for (int i = 1; i < argc; i++) {
        bool found_arg = false;
        arg            = argv[i];

        for (auto& option : options.string_options) {
            if ((option.short_name.size() > 0 && arg == option.short_name) || (option.long_name.size() > 0 && arg == option.long_name)) {
                found_arg = true;
                if (++i >= argc) {
                    invalid_arg = true;
                    break;
                }
                *option.target = argv_to_utf8(i, argv);
            }
        }
        if (invalid_arg) {
            break;
        }

        for (auto& option : options.int_options) {
            if ((option.short_name.size() > 0 && arg == option.short_name) || (option.long_name.size() > 0 && arg == option.long_name)) {
                found_arg = true;
                if (++i >= argc) {
                    invalid_arg = true;
                    break;
                }
                *option.target = std::stoi(argv[i]);
            }
        }
        if (invalid_arg) {
            break;
        }

        for (auto& option : options.float_options) {
            if ((option.short_name.size() > 0 && arg == option.short_name) || (option.long_name.size() > 0 && arg == option.long_name)) {
                found_arg = true;
                if (++i >= argc) {
                    invalid_arg = true;
                    break;
                }
                *option.target = std::stof(argv[i]);
            }
        }
        if (invalid_arg) {
            break;
        }

        for (auto& option : options.bool_options) {
            if ((option.short_name.size() > 0 && arg == option.short_name) || (option.long_name.size() > 0 && arg == option.long_name)) {
                found_arg = true;
                if (option.keep_true) {
                    *option.target = true;
                } else {
                    *option.target = false;
                }
            }
        }
        if (invalid_arg) {
            break;
        }

        for (auto& option : options.manual_options) {
            if ((option.short_name.size() > 0 && arg == option.short_name) || (option.long_name.size() > 0 && arg == option.long_name)) {
                found_arg = true;
                int ret   = option.cb(argc, argv, i);
                if (ret < 0) {
                    invalid_arg = true;
                    break;
                }
                i += ret;
            }
        }
        if (invalid_arg) {
            break;
        }
        if (!found_arg) {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            return false;
        }
    }
    if (invalid_arg) {
        fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
        return false;
    }
    return true;
}

static std::string wrap_text(const std::string& text, size_t width, size_t indent) {
    std::ostringstream oss;
    size_t line_len = 0;
    size_t pos      = 0;

    while (pos < text.size()) {
        // Preserve manual newlines
        if (text[pos] == '\n') {
            oss << '\n'
                << std::string(indent, ' ');
            line_len = indent;
            ++pos;
            continue;
        }

        // Add the character
        oss << text[pos];
        ++line_len;
        ++pos;

        // If the current line exceeds width, try to break at the last space
        if (line_len >= width) {
            std::string current = oss.str();
            size_t back         = current.size();

            // Find the last space (for a clean break)
            while (back > 0 && current[back - 1] != ' ' && current[back - 1] != '\n')
                --back;

            // If found a space to break on
            if (back > 0 && current[back - 1] != '\n') {
                std::string before = current.substr(0, back - 1);
                std::string after  = current.substr(back);
                oss.str("");
                oss.clear();
                oss << before << "\n"
                    << std::string(indent, ' ') << after;
            } else {
                // If no space found, just break at width
                oss << "\n"
                    << std::string(indent, ' ');
            }
            line_len = indent;
        }
    }

    return oss.str();
}

void print_usage(int argc, const char* argv[], const ArgOptions& options) {
    constexpr size_t max_line_width = 120;

    std::cout << "Usage: " << argv[0] << " [options]\n\n";
    std::cout << "Options:\n";

    struct Entry {
        std::string names;
        std::string desc;
    };
    std::vector<Entry> entries;

    auto add_entry = [&](const std::string& s, const std::string& l,
                         const std::string& desc, const std::string& hint = "") {
        std::ostringstream ss;
        if (!s.empty())
            ss << s;
        if (!s.empty() && !l.empty())
            ss << ", ";
        if (!l.empty())
            ss << l;
        if (!hint.empty())
            ss << " " << hint;
        entries.push_back({ss.str(), desc});
    };

    for (auto& o : options.string_options)
        add_entry(o.short_name, o.long_name, o.desc, "<string>");
    for (auto& o : options.int_options)
        add_entry(o.short_name, o.long_name, o.desc, "<int>");
    for (auto& o : options.float_options)
        add_entry(o.short_name, o.long_name, o.desc, "<float>");
    for (auto& o : options.bool_options)
        add_entry(o.short_name, o.long_name, o.desc, "");
    for (auto& o : options.manual_options)
        add_entry(o.short_name, o.long_name, o.desc);

    size_t max_name_width = 0;
    for (auto& e : entries)
        max_name_width = std::max(max_name_width, e.names.size());

    for (auto& e : entries) {
        size_t indent            = 2 + max_name_width + 4;
        size_t desc_width        = (max_line_width > indent ? max_line_width - indent : 40);
        std::string wrapped_desc = wrap_text(e.desc, max_line_width, indent);
        std::cout << "  " << std::left << std::setw(static_cast<int>(max_name_width) + 4)
                  << e.names << wrapped_desc << "\n";
    }
}

void parse_args(int argc, const char** argv, SDParams& params) {
    ArgOptions options;

    options.string_options = {
        {"-m", "--model", "path to full model", &params.ctxParams.model_path},
        {"", "--clip_l", "path to the clip-l text encoder", &params.ctxParams.clip_l_path},
        {"", "--clip_g", "path to the clip-g text encoder", &params.ctxParams.clip_g_path},
        {"", "--clip_vision", "path to the clip-vision encoder", &params.ctxParams.clip_vision_path},
        {"", "--t5xxl", "path to the t5xxl text encoder", &params.ctxParams.t5xxl_path},
        {"", "--llm", "path to the llm text encoder", &params.ctxParams.llm_path},
        {"", "--llm_vision", "path to the llm vit", &params.ctxParams.llm_vision_path},
        {"", "--diffusion-model", "path to the standalone diffusion model", &params.ctxParams.diffusion_model_path},
        {"", "--high-noise-diffusion-model", "path to the standalone high noise diffusion model", &params.ctxParams.high_noise_diffusion_model_path},
        {"", "--vae", "path to standalone vae model", &params.ctxParams.vae_path},
        {"", "--taesd", "path to taesd. Using Tiny AutoEncoder for fast decoding (low quality)", &params.ctxParams.taesd_path},
        {"", "--embd-dir", "embeddings directory", &params.ctxParams.embeddings_path},
        {"", "--lora-model-dir", "lora model directory", &params.ctxParams.lora_model_dir},
        {"-o", "--output", "path to write result image to (default: ./server/output.png)", &params.output_path},
        {"-p", "--prompt", "the prompt to render", &params.lastRequest.prompt},
        {"-n", "--negative-prompt", "the negative prompt (default: \"\")", &params.lastRequest.negative_prompt},
        {"", "--preview-path", "path to write preview image to (default: ./server/preview.png)", &params.preview_path},
        // {"", "--pm-id-images-dir", "input id images directory", &params.input_id_images_path}, // maybe shouldn't be set via cli
        {"", "--models-dir", "path to models directory", &params.models_dir},
        {"", "--diffusion-models-dir", "path to diffusion models directory", &params.diffusion_models_dir},
        {"", "--encoders-dir", "path to text encoders directory", &params.clip_dir},
        {"", "--vision-model-dir", "path to vision encoders directory", &params.clip_vision_dir},
        {"", "--vae-dir", "path to vae directory", &params.vae_dir},
        {"", "--tae-dir", "path to tae directory", &params.tae_dir},
        {"", "--control-net-dir", "path to controlnet models directory", &params.controlnet_dir},
        {"", "--photo-maker-dir", "path to PHOTOMAKER models directory", &params.photomaker_dir},
        {"", "--upscaler-dir", "path to upscaler models directory", &params.upscaler_dir},
        {"", "--host", "host to listen on (default: 0.0.0.0)", &params.host},
        {"", "--custom-frontend-path", "path to custom frontend directory", &params.custom_frontend_path},
    };

    options.int_options = {
        {"-t", "--threads", "number of threads to use during computation (default: -1). If threads <= 0, then threads will be set to the number of CPU physical cores", &params.ctxParams.n_threads},
        {"-H", "--height", "image height, in pixel space (default: 512)", &params.lastRequest.height},
        {"-W", "--width", "image width, in pixel space (default: 512)", &params.lastRequest.width},
        {"", "--steps", "number of sample steps (default: 20)", &params.lastRequest.sample_params.sample_steps},
        {"", "--clip-skip", "ignore last layers of CLIP network; 1 ignores none, 2 ignores one layer (default: -1). <= 0 represents unspecified, will be 1 for SD1.x, 2 for SD2.x", &params.lastRequest.clip_skip},
        {"-b", "--batch-count", "batch count", &params.lastRequest.batch_count},
        {"", "--port", "port to listen on", &params.port}};

    options.float_options = {
        {"", "--cfg-scale", "unconditional guidance scale: (default: 7.0)", &params.lastRequest.sample_params.guidance.txt_cfg},
        {"", "--guidance", "distilled guidance scale for models with guidance input (default: 3.5)", &params.lastRequest.sample_params.guidance.distilled_guidance}};

    static bool dummy_worker_mode = false;
    options.bool_options          = {
        {"", "--vae-tiling", "process vae in tiles to reduce memory usage", true, &params.lastRequest.tiling_params.enabled},
        {"", "--force-sdxl-vae-conv-scale", "force use of conv scale on sdxl vae", true, &params.ctxParams.force_sdxl_vae_conv_scale},
        {"", "--offload-to-cpu", "place the weights in RAM to save VRAM, and automatically load them into VRAM when needed", true, &params.ctxParams.offload_params_to_cpu},
        {"", "--control-net-cpu", "keep controlnet in cpu (for low vram)", true, &params.ctxParams.keep_control_net_on_cpu},
        {"", "--clip-on-cpu", "keep clip in cpu (for low vram)", true, &params.ctxParams.keep_clip_on_cpu},
        {"", "--vae-on-cpu", "keep vae in cpu (for low vram)", true, &params.ctxParams.keep_vae_on_cpu},
        {"", "--diffusion-fa", "use flash attention in the diffusion model", true, &params.ctxParams.diffusion_flash_attn},
        {"", "--diffusion-conv-direct", "use ggml_conv2d_direct in the diffusion model", true, &params.ctxParams.diffusion_conv_direct},
        {"", "--vae-conv-direct", "use ggml_conv2d_direct in the vae model", true, &params.ctxParams.vae_conv_direct},
        {"-v", "--verbose", "print extra info", true, &params.verbose},
        {"", "--color", "colors the logging tags according to level", true, &params.color},
        {"", "--preview-noisy", "enables previewing noisy inputs of the models rather than the denoised outputs", true, &params.lastRequest.preview_noisy},
        {"", "--worker-process-mode", "[internal] run in worker process mode (no supervisor process)", true, &dummy_worker_mode},
        {"", "--restore-state", "[internal] restore server state from disk on startup (file: server_state_dump.json)", true, &params.restore_state},
    };

    auto on_rng_arg = [&](int argc, const char** argv, int index) {
        if (++index >= argc) {
            return -1;
        }
        const char* arg = argv[index];
        if (strcmp(arg, "std_default") == 0) {
            params.ctxParams.rng_type = STD_DEFAULT_RNG;
        } else if (strcmp(arg, "cuda") == 0) {
            params.ctxParams.rng_type = CUDA_RNG;
        } else {
            fprintf(stderr, "error: invalid rng type %s\n",
                    arg);
            return -1;
        }
        return 1;
    };

    auto on_schedule_arg = [&](int argc, const char** argv, int index) {
        if (++index >= argc) {
            return -1;
        }
        const char* arg                            = argv[index];
        params.lastRequest.sample_params.scheduler = str_to_scheduler(arg);
        if (params.lastRequest.sample_params.scheduler == SCHEDULER_COUNT) {
            fprintf(stderr, "error: invalid scheduler %s\n",
                    arg);
            return -1;
        }
        return 1;
    };

    auto on_seed_arg = [&](int argc, const char** argv, int index) {
        if (++index >= argc) {
            return -1;
        }
        params.lastRequest.seed = std::stoll(argv[index]);
        return 1;
    };

    auto on_sample_method_arg = [&](int argc, const char** argv, int index) {
        if (++index >= argc) {
            return -1;
        }
        const char* arg   = argv[index];
        int sample_method = str_to_sample_method(arg);
        if (sample_method == SAMPLE_METHOD_COUNT) {
            fprintf(stderr, "error: invalid sample method %s\n",
                    arg);
            return -1;
        }
        params.lastRequest.sample_params.sample_method = (sample_method_t)sample_method;
        return 1;
    };

    auto on_help_arg = [&](int argc, const char** argv, int index) {
        print_usage(argc, argv, options);
        exit(0);
        return 0;
    };

    options.manual_options = {
        {"", "--rng", "RNG, one of [std_default, cuda, cpu], default: cuda(sd-webui), cpu(comfyui)", on_rng_arg},
        {"-s", "--seed", "RNG seed (default: 42, use random seed for < 0)", on_seed_arg},
        {"", "--sampling-method", "sampling method, one of [euler, euler_a, heun, dpm2, dpm++2s_a, dpm++2m, dpm++2mv2, ipndm, ipndm_v, lcm, ddim_trailing, tcd] (default: euler for Flux/SD3/Wan, euler_a otherwise)", on_sample_method_arg},
        {"", "--schedule", "denoiser sigma scheduler, one of [discrete, karras, exponential, ays, gits, smoothstep, sgm_uniform, simple], default: discrete", on_schedule_arg},
        {"-h", "--help", "show this help message and exit", on_help_arg}};

    if (!parse_options(argc, argv, options)) {
        print_usage(argc, argv, options);
        exit(1);
    }

    if (params.ctxParams.n_threads <= 0) {
        params.ctxParams.n_threads = sd_get_num_physical_cores();
    }

    if (params.lastRequest.prompt.length() == 0) {
        fprintf(stderr, "error: the following arguments are required: prompt\n");
        print_usage(argc, argv, options);
        exit(1);
    }

    if (params.ctxParams.model_path.length() == 0 && params.ctxParams.diffusion_model_path.length() == 0) {
        fprintf(stderr, "error: the following arguments are required: model_path/diffusion_model\n");
        print_usage(argc, argv, options);
        exit(1);
    }

    if (params.output_path.length() == 0) {
        fprintf(stderr, "error: the following arguments are required: output_path\n");
        print_usage(argc, argv, options);
        exit(1);
    }

    if (params.lastRequest.height <= 0) {
        fprintf(stderr, "error: the height must be greater than 0\n");
        exit(1);
    }

    if (params.lastRequest.width <= 0) {
        fprintf(stderr, "error: the width must be greater than 0\n");
        exit(1);
    }

    if (params.lastRequest.sample_params.sample_steps <= 0) {
        fprintf(stderr, "error: the sample_steps must be greater than 0\n");
        exit(1);
    }

    if (params.lastRequest.strength < 0.f || params.lastRequest.strength > 1.f) {
        fprintf(stderr, "error: can only work with strength in [0.0, 1.0]\n");
        exit(1);
    }

    if (params.lastRequest.seed < 0) {
        srand((int)time(nullptr));
        params.lastRequest.seed = rand();
    }

    if (params.ctxParams.n_threads <= 0) {
        params.ctxParams.n_threads = sd_get_num_physical_cores();
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
    parameter_string += "Steps: " + std::to_string(params.lastRequest.sample_params.sample_steps) + ", ";
    parameter_string += "CFG scale: " + std::to_string(params.lastRequest.sample_params.guidance.txt_cfg) + ", ";
    parameter_string += "Guidance: " + std::to_string(params.lastRequest.sample_params.guidance.distilled_guidance) + ", ";
    parameter_string += "Seed: " + std::to_string(seed) + ", ";
    parameter_string += "Size: " + std::to_string(params.lastRequest.width) + "x" + std::to_string(params.lastRequest.height) + ", ";
    parameter_string += "Model: " + sd_basename(params.ctxParams.model_path) + ", ";
    parameter_string += "RNG: " + std::string(sd_rng_type_name(params.ctxParams.rng_type)) + ", ";
    parameter_string += "Sampler: " + std::string(sd_sample_method_name(params.lastRequest.sample_params.sample_method));
    if (params.lastRequest.sample_params.scheduler == KARRAS_SCHEDULER) {
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

    bool verbose = params ? params->verbose : false;
    bool color   = params ? params->color : false;

    if (!log || (!verbose && level <= SD_LOG_DEBUG)) {
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

    if (color == true) {
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

uint8_t* load_image_from_memory(const std::string image_bin, int& width, int& height, int& channel, int expected_width = 0, int expected_height = 0, int expected_channel = 3) {
    uint8_t* image_buffer = stbi_load_from_memory((const stbi_uc*)image_bin.c_str(), image_bin.size(), &width, &height, &channel, expected_channel);
    if (image_buffer == nullptr) {
        sd_log(sd_log_level_t::SD_LOG_ERROR, "load image from binary data failed\n");
        return nullptr;
    }
    if (channel < expected_channel) {
        sd_log(sd_log_level_t::SD_LOG_ERROR,
               "the number of channels for the input image must be >= %d,"
               "but got %d channels",
               expected_channel,
               channel);
        free(image_buffer);
        return nullptr;
    }
    if (width <= 0) {
        sd_log(sd_log_level_t::SD_LOG_ERROR, "error: the width of image must be greater than 0\n");
        free(image_buffer);
        return nullptr;
    }
    if (height <= 0) {
        sd_log(sd_log_level_t::SD_LOG_ERROR, "error: the height of image must be greater than 0\n");
        free(image_buffer);
        return nullptr;
    }

    // Resize input image ...
    if ((expected_width > 0 && expected_height > 0) && (height != expected_height || width != expected_width)) {
        float dst_aspect = (float)expected_width / (float)expected_height;
        float src_aspect = (float)width / (float)height;

        int crop_x = 0, crop_y = 0;
        int crop_w = width, crop_h = height;

        if (src_aspect > dst_aspect) {
            crop_w = (int)(height * dst_aspect);
            crop_x = (width - crop_w) / 2;
        } else if (src_aspect < dst_aspect) {
            crop_h = (int)(width / dst_aspect);
            crop_y = (height - crop_h) / 2;
        }

        if (crop_x != 0 || crop_y != 0) {
            sd_log(sd_log_level_t::SD_LOG_INFO, "crop input image from %dx%d to %dx%d\n", width, height, crop_w, crop_h);
            uint8_t* cropped_image_buffer = (uint8_t*)malloc(crop_w * crop_h * expected_channel);
            if (cropped_image_buffer == nullptr) {
                sd_log(sd_log_level_t::SD_LOG_ERROR, "error: allocate memory for crop\n");
                free(image_buffer);
                return nullptr;
            }
            for (int row = 0; row < crop_h; row++) {
                uint8_t* src = image_buffer + ((crop_y + row) * width + crop_x) * expected_channel;
                uint8_t* dst = cropped_image_buffer + (row * crop_w) * expected_channel;
                memcpy(dst, src, crop_w * expected_channel);
            }

            width  = crop_w;
            height = crop_h;
            free(image_buffer);
            image_buffer = cropped_image_buffer;
        }

        sd_log(sd_log_level_t::SD_LOG_INFO, "resize input image from %dx%d to %dx%d\n", width, height, expected_width, expected_height);
        int resized_height = expected_height;
        int resized_width  = expected_width;

        uint8_t* resized_image_buffer = (uint8_t*)malloc(resized_height * resized_width * expected_channel);
        if (resized_image_buffer == nullptr) {
            sd_log(sd_log_level_t::SD_LOG_ERROR, "error: allocate memory for resize input image\n");
            free(image_buffer);
            return nullptr;
        }
        stbir_resize(image_buffer, width, height, 0,
                     resized_image_buffer, resized_width, resized_height, 0, STBIR_TYPE_UINT8,
                     expected_channel, STBIR_ALPHA_CHANNEL_NONE, 0,
                     STBIR_EDGE_CLAMP, STBIR_EDGE_CLAMP,
                     STBIR_FILTER_BOX, STBIR_FILTER_BOX,
                     STBIR_COLORSPACE_SRGB, nullptr);
        width  = resized_width;
        height = resized_height;
        free(image_buffer);
        image_buffer = resized_image_buffer;
        channel      = expected_channel;
    }
    return image_buffer;
}

struct StringOptionJson {
    std::string name;
    std::string* target;
};

struct IntOptionJson {
    std::string name;
    int* target;
};

struct LongOptionJson {
    std::string name;
    int64_t* target;
};

struct FloatOptionJson {
    std::string name;
    float* target;
};

struct BoolOptionJson {
    std::string name;
    bool* target;
};

struct ManualOptionJson {
    std::string name;
    std::function<bool(const nlohmann::json&)> cb;
};

struct JsonOptions {
    std::vector<StringOptionJson> string_options;
    std::vector<IntOptionJson> int_options;
    std::vector<LongOptionJson> long_options;
    std::vector<FloatOptionJson> float_options;
    std::vector<BoolOptionJson> bool_options;
    std::vector<ManualOptionJson> manual_options;
};

bool parse_json_options(nlohmann::json payload, JsonOptions& options) {
    bool has_anything_changed = false;

    for (auto& option : options.string_options) {
        try {
            if (payload.contains(option.name)) {
                if (option.target->compare(payload[option.name].get<std::string>()) != 0) {
                    has_anything_changed = true;
                }
                option.target->assign(payload[option.name].get<std::string>());
            }
        } catch (...) {
            sd_log(sd_log_level_t::SD_LOG_WARN, "Failed to parse string option: %s\n", option.name.c_str());
        }
    }
    for (auto& option : options.int_options) {
        try {
            if (payload.contains(option.name)) {
                if (*option.target != payload[option.name].get<int>()) {
                    has_anything_changed = true;
                }
                *option.target = payload[option.name].get<int>();
            }
        } catch (...) {
            sd_log(sd_log_level_t::SD_LOG_WARN, "Failed to parse int option: %s\n", option.name.c_str());
        }
    }
    for (auto& option : options.long_options) {
        try {
            if (payload.contains(option.name)) {
                if (*option.target != payload[option.name].get<int64_t>()) {
                    has_anything_changed = true;
                }
                *option.target = payload[option.name].get<int64_t>();
            }
        } catch (...) {
            sd_log(sd_log_level_t::SD_LOG_WARN, "Failed to parse long option: %s\n", option.name.c_str());
        }
    }
    for (auto& option : options.float_options) {
        try {
            if (payload.contains(option.name)) {
                if (*option.target != payload[option.name].get<float>()) {
                    has_anything_changed = true;
                }
                *option.target = payload[option.name].get<float>();
            }
        } catch (...) {
            sd_log(sd_log_level_t::SD_LOG_WARN, "Failed to parse float option: %s\n", option.name.c_str());
        }
    }

    for (auto& option : options.bool_options) {
        try {
            if (payload.contains(option.name)) {
                if (*option.target != payload[option.name].get<bool>()) {
                    has_anything_changed = true;
                }
                *option.target = payload[option.name].get<bool>();
            }
        } catch (...) {
            sd_log(sd_log_level_t::SD_LOG_WARN, "Failed to parse bool option: %s\n", option.name.c_str());
        }
    }

    for (auto& option : options.manual_options) {
        try {
            if (payload.contains(option.name)) {
                has_anything_changed = option.cb(payload[option.name]) || has_anything_changed;
            }
        } catch (...) {
            sd_log(sd_log_level_t::SD_LOG_WARN, "Failed to parse bool option: %s\n", option.name.c_str());
        }
    }

    return has_anything_changed;
}

bool parseJsonPrompt(std::string json_str, SDParams* params) {
    bool updatectx = false;
    using namespace nlohmann;
    json payload = json::parse(json_str);

    // if no exception, the request is a json object
    // now we try to get the new param values from the payload object
    // const char *prompt, const char *negative_prompt, int clip_skip, float cfg_scale, float guidance, int width, int height, sample_method_t sample_method, int sample_steps, int64_t seed, int batch_count, const sd_image_t *control_cond, float control_strength, float style_strength, const char *input_id_images_path

    JsonOptions guidance_options = {
        {},
        {},
        {},
        // float_options
        {
            {"txt_cfg", &params->lastRequest.sample_params.guidance.txt_cfg},
            {"img_cfg", &params->lastRequest.sample_params.guidance.img_cfg},
            {"distilled_guidance", &params->lastRequest.sample_params.guidance.distilled_guidance},
        },
        {},
        // manual_options
        {
            {"slg",
             [&](const json& o) -> bool {
                 JsonOptions slg_options = {
                     {},
                     {},
                     {},
                     // float_options
                     {
                         {"layer_start", &params->lastRequest.sample_params.guidance.slg.layer_start},
                         {"layer_end", &params->lastRequest.sample_params.guidance.slg.layer_end},
                         {"scale", &params->lastRequest.sample_params.guidance.slg.scale},
                     },
                     {},
                     // manual_options
                     {{"layers", [&](const json& o) -> bool {
                           std::vector<int> layers         = o.get<std::vector<int>>();
                           bool change                     = params->lastRequest.skip_layers != layers;
                           params->lastRequest.skip_layers = layers;
                           return change;
                       }}},
                 };
                 return parse_json_options(o, slg_options);
             }},
        },
    };

    JsonOptions sample_params_options = {
        {},
        // int_options
        {
            {"sample_steps", &params->lastRequest.sample_params.sample_steps},
            {"shifted_timestep", &params->lastRequest.sample_params.shifted_timestep},
        },
        {},
        // float_options
        {{"eta", &params->lastRequest.sample_params.eta}},
        {},
        // manual_options
        {
            {"guidance", [&](const json& o) -> bool {
                 return parse_json_options(o, guidance_options);
             }},
            {"scheduler", [&](const json& o) -> bool {
                 std::string schedule       = o.get<std::string>();
                 scheduler_t schedule_found = str_to_scheduler(schedule.c_str());
                 bool change                = false;
                 if (schedule_found != SCHEDULER_COUNT) {
                     if (params->lastRequest.sample_params.scheduler != schedule_found) {
                         params->lastRequest.sample_params.scheduler = schedule_found;
                         change                                      = true;
                     }
                 } else {
                     sd_log(sd_log_level_t::SD_LOG_WARN, "Unknown schedule: %s\n", schedule.c_str());
                 }
                 return change;
             }},
            {"sample_method", [&](const json& o) -> bool {
                 std::string method           = o.get<std::string>();
                 sample_method_t method_found = str_to_sample_method(method.c_str());
                 bool change                  = false;
                 if (method_found != SAMPLE_METHOD_COUNT) {
                     if (params->lastRequest.sample_params.sample_method != method_found) {
                         params->lastRequest.sample_params.sample_method = method_found;
                         change                                          = true;
                     }
                 } else {
                     sd_log(sd_log_level_t::SD_LOG_WARN, "Unknown method: %s\n", method.c_str());
                 }
                 return change;
             }},
        }};

    JsonOptions photomaker_options = {
        {},
        {},
        {},
        {{"style_strength", &params->lastRequest.pm_params.style_strength}},
        {},
        {
            {"id_images", [&](const json& o) -> bool {
                 // fill up params->lastRequest.pm_images_vec
                 std::vector<std::string> b64_data = o.get<std::vector<std::string>>();

                 // empty the vector if the new data is empty
                 if (b64_data.empty()) {
                     if (params->lastRequest.pm_images_vec.size() > 0) {
                         for (auto& img : params->lastRequest.pm_images_vec) {
                             free(img.data);
                         }
                         params->lastRequest.pm_images_vec.clear();
                         return true;
                     }
                     return false;
                 }

                 for (auto& b64_image : b64_data) {
                     // decode the base64 image
                     std::string bin_image = base64_decode(b64_image);
                     int width, height, channels;
                     uint8_t* image = load_image_from_memory(bin_image, width, height, channels);
                     if (image == nullptr) {
                         sd_log(sd_log_level_t::SD_LOG_WARN, "Failed to load image from memory\n");
                         continue;
                     }
                     sd_image_t img = {(uint32_t)width, (uint32_t)height, 3, image};
                     params->lastRequest.pm_images_vec.push_back(img);
                 }
                 return true;
             }},
            {"id_embed_path", [&](const json& o) -> bool {
                 // TODO: avoid parsing paths, rather use ids and convert to path server-side
                 std::string new_path                 = o.get<std::string>();
                 bool change                          = params->lastRequest.pm_id_embed_path != new_path;
                 params->lastRequest.pm_id_embed_path = new_path;
                 return change;
             }},
        }};

    JsonOptions tiling_options = {
        {},
        // int_options
        {
            {"tile_size_x", &params->lastRequest.tiling_params.tile_size_x},
            {"tile_size_y", &params->lastRequest.tiling_params.tile_size_y},
        },
        {},
        // float_options
        {
            {"target_overlap", &params->lastRequest.tiling_params.target_overlap},
            {"rel_size_x", &params->lastRequest.tiling_params.rel_size_x},
            {"rel_size_y", &params->lastRequest.tiling_params.rel_size_y},
        },
        {{"enabled", &params->lastRequest.tiling_params.enabled}},
        // manual_options
        {
            {"tile_size", [&](const json& o) -> bool {
                 bool change = false;
                 // try parsing as single int, as list of ints, or as formated string
                 if (o.is_number_integer()) {
                     int new_size = o.get<int>();

                     change = params->lastRequest.tiling_params.tile_size_x != new_size || params->lastRequest.tiling_params.tile_size_y != new_size;

                     params->lastRequest.tiling_params.tile_size_x = new_size;
                     params->lastRequest.tiling_params.tile_size_y = new_size;
                 } else if (o.is_array()) {
                     std::vector<int> tile_size = o.get<std::vector<int>>();
                     if (tile_size.size() == 2) {
                         change = params->lastRequest.tiling_params.tile_size_x != tile_size[0] || params->lastRequest.tiling_params.tile_size_y != tile_size[1];

                         params->lastRequest.tiling_params.tile_size_x = tile_size[0];
                         params->lastRequest.tiling_params.tile_size_y = tile_size[1];
                     } else {
                         sd_log(sd_log_level_t::SD_LOG_WARN, "tile_size array must have 2 elements\n");
                     }
                 } else {
                     std::string tile_size_str = o.get<std::string>();
                     // parse string as "WxH" or "W,H" or "W H"
                     size_t x_pos = tile_size_str.find_first_of("xX, ");
                     if (x_pos != std::string::npos) {
                         int x  = std::stoi(tile_size_str.substr(0, x_pos));
                         int y  = std::stoi(tile_size_str.substr(x_pos + 1));
                         change = params->lastRequest.tiling_params.tile_size_x != x || params->lastRequest.tiling_params.tile_size_y != y;

                         params->lastRequest.tiling_params.tile_size_x = x;
                         params->lastRequest.tiling_params.tile_size_y = y;
                     } else {
                         sd_log(sd_log_level_t::SD_LOG_WARN, "tile_size string must be in format WxH or W,H or W H\n");
                     }
                 }
                 return change;
             }},
        }};

    JsonOptions request_options = {
        // string_options
        {
            {"prompt", &params->lastRequest.prompt},
            {"negative_prompt", &params->lastRequest.negative_prompt},
        },
        // int_options
        {
            {"clip_skip", &params->lastRequest.clip_skip},
            {"width", &params->lastRequest.width},
            {"height", &params->lastRequest.height},
            {"batch_count", &params->lastRequest.batch_count},
            {"preview_interval", &params->lastRequest.preview_interval},
        },
        // long_options
        {
            {"seed", &params->lastRequest.seed},
        },
        // float_options
        {
            {"strength", &params->lastRequest.strength},
            {"control_strength", &params->lastRequest.control_strength},
        },
        // bool_options
        {
            {"auto_resize_ref_image", &params->lastRequest.auto_resize_ref_image},
            {"increase_ref_index", &params->lastRequest.increase_ref_index},
            {"upscale", &params->lastRequest.upscale},
            {"preview_noisy", &params->lastRequest.preview_noisy},
        },
        // manual_options
        {
            {"sample_params", [&](const json& o) -> bool {
                 return parse_json_options(o, sample_params_options);
             }},
            {"pm_params", [&](const json& o) -> bool {
                 return parse_json_options(o, photomaker_options);
             }},
            {"tiling_params", [&](const json& o) -> bool {
                 return parse_json_options(o, tiling_options);
             }},
            {"init_image", [&](const json& o) -> bool {
                 // assumes base64 encoded png or jpg image
                 std::string b64_data = o.get<std::string>();
                 // empty string means no init image, cleanup previous one if exists
                 if (b64_data.empty()) {
                     if (params->lastRequest.init_image.data) {
                         free(params->lastRequest.init_image.data);
                         params->lastRequest.init_image.data = NULL;
                         return true;
                     }
                     return false;
                 }

                 std::string bin_data = base64_decode(b64_data);
                 int width = 0, height = 0, c = 0;
                 // int_options are processed before manual_options, so we can use the width and height here
                 uint8_t* image_buffer = load_image_from_memory(bin_data, width, height, c, params->lastRequest.width, params->lastRequest.height);
                 if (image_buffer == NULL) {
                     sd_log(sd_log_level_t::SD_LOG_WARN, "Failed to load image from memory\n");
                 }
                 if (params->lastRequest.init_image.data) {
                     free(params->lastRequest.init_image.data);
                 }
                 params->lastRequest.init_image = sd_image_t{(uint32_t)width, (uint32_t)height, 3, image_buffer};
                 sd_log(sd_log_level_t::SD_LOG_INFO, "Loaded image from memory: %dx%d, %d channels\n", width, height, c);
                 return true;
             }},
            {"mask_image", [&](const json& o) -> bool {
                 // base64 encoded png or jpg monochrome image
                 std::string b64_data = o.get<std::string>();
                 // empty string means no mask image, cleanup previous one if exists
                 if (b64_data.empty()) {
                     if (params->lastRequest.mask_image.data) {
                         free(params->lastRequest.mask_image.data);
                         params->lastRequest.mask_image.data = NULL;
                         return true;
                     }
                     return false;
                 }

                 std::string bin_data = base64_decode(b64_data);
                 int width = 0, height = 0, c = 0;
                 uint8_t* image_buffer = load_image_from_memory(bin_data, width, height, c, params->lastRequest.width, params->lastRequest.height, 1);  // force 1 channel
                 if (image_buffer == NULL) {
                     sd_log(sd_log_level_t::SD_LOG_WARN, "Failed to load image from memory\n");
                 }
                 if (params->lastRequest.mask_image.data) {
                     free(params->lastRequest.mask_image.data);
                 }
                 params->lastRequest.mask_image = sd_image_t{(uint32_t)width, (uint32_t)height, 1, image_buffer};
                 sd_log(sd_log_level_t::SD_LOG_INFO, "Loaded image from memory: %dx%d, %d channels\n", width, height, c);
                 return true;
             }},
            {"control_image", [&](const json& o) -> bool {
                 // base64 encoded png or jpg rgb image
                 std::string b64_data = o.get<std::string>();
                 // empty string means no control image, cleanup previous one if exists
                 if (b64_data.empty()) {
                     if (params->lastRequest.control_image.data) {
                         free(params->lastRequest.control_image.data);
                         params->lastRequest.control_image.data = NULL;
                         return true;
                     }
                     return false;
                 }

                 std::string bin_data = base64_decode(b64_data);
                 int width = 0, height = 0, c = 0;
                 // int_options are processed before manual_options, so we can use the width and height here
                 uint8_t* image_buffer = load_image_from_memory(bin_data, width, height, c, params->lastRequest.width, params->lastRequest.height);
                 if (image_buffer == NULL) {
                     sd_log(sd_log_level_t::SD_LOG_WARN, "Failed to load image from memory\n");
                 }
                 if (params->lastRequest.control_image.data) {
                     free(params->lastRequest.control_image.data);
                 }
                 params->lastRequest.control_image = sd_image_t{(uint32_t)width, (uint32_t)height, (uint32_t)3, image_buffer};
                 sd_log(sd_log_level_t::SD_LOG_INFO, "Loaded image from memory: %dx%d, %d channels\n", width, height, c);
                 return true;
             }},
            // ref images
            {"ref_images", [&](const json& o) -> bool {
                 // base64 encoded png or jpg rgb images
                 std::vector<std::string> b64_data = o.get<std::vector<std::string>>();
                 // empty array means no ref images, cleanup previous ones if exists
                 if (b64_data.empty()) {
                     if (params->lastRequest.ref_images.size() > 0) {
                         for (auto& ref_image : params->lastRequest.ref_images) {
                             free(ref_image.data);
                         }
                         params->lastRequest.ref_images.clear();
                         return true;
                     }
                     return false;
                 }

                 for (auto& b64_image : b64_data) {
                     std::string bin_data = base64_decode(b64_image);
                     int width = 0, height = 0, c = 0;
                     uint8_t* image_buffer = load_image_from_memory(bin_data, width, height, c);
                     if (image_buffer == NULL) {
                         sd_log(sd_log_level_t::SD_LOG_WARN, "Failed to load image from memory\n");
                     }
                     params->lastRequest.ref_images.push_back(sd_image_t{(uint32_t)width, (uint32_t)height, 3, image_buffer});
                 }
                 return true;
             }},
            // preview_method
            {"preview_method", [&](const json& o) -> bool {
                 std::string preview = o.get<std::string>();
                 int preview_found   = -1;
                 for (int m = 0; m < PREVIEW_COUNT; m++) {
                     if (!strcmp(preview.c_str(), sd_preview_name((preview_t)m))) {
                         preview_found = m;
                     }
                 }
                 bool change = false;
                 if (preview_found >= 0) {
                     if (params->lastRequest.preview_method != (preview_t)preview_found) {
                         params->lastRequest.preview_method = (preview_t)preview_found;
                         change                             = true;
                     }
                 } else {
                     sd_log(sd_log_level_t::SD_LOG_WARN, "Unknown preview: %s\n", preview.c_str());
                 }
                 return change;
             }},
        }};

    parse_json_options(payload, request_options);

    // CTX

    const int MODEL_UNLOAD = -2;
    const int MODEL_KEEP   = -1;

    auto parse_model_part = [&](const json& o, std::vector<std::string> model_part_files, std::string model_part_dir, std::string& model_part_path) -> bool {
        bool change = false;
        int index   = o.get<int>();
        if (index >= 0 && index < model_part_files.size()) {
            std::filesystem::path new_path = std::filesystem::path(model_part_dir) / model_part_files[index];
            std::string new_path_str       = new_path.string();
            if (model_part_path != new_path_str) {
                model_part_path = new_path_str;
                change          = true;
            }
        } else if (index == MODEL_UNLOAD) {
            if (model_part_path != "") {
                change = true;
            }
            model_part_path = "";
        } else if (index != MODEL_KEEP) {
            sd_log(sd_log_level_t::SD_LOG_WARN, "Invalid model index: %d out of %d\n", index, model_part_files.size());
        }
        return change;
    };

    JsonOptions ctx_options = {
        // string_options (empty, we use ids to avoid exposing paths)
        {},
        // int_options
        {
            {"n_threads", &params->ctxParams.n_threads},
            {"chroma_t5_mask_pad", &params->ctxParams.chroma_t5_mask_pad},
        },
        // long_options
        {},
        // float_options
        {
            {"flow_shift", &params->ctxParams.flow_shift},
        },
        // bool_options
        {
            {"vae_decode_only", &params->ctxParams.vae_decode_only},
            {"free_params_immediately", &params->ctxParams.free_params_immediately},
            {"offload_params_to_cpu", &params->ctxParams.offload_params_to_cpu},
            {"keep_clip_on_cpu", &params->ctxParams.keep_clip_on_cpu},
            {"keep_control_net_on_cpu", &params->ctxParams.keep_control_net_on_cpu},
            {"keep_vae_on_cpu", &params->ctxParams.keep_vae_on_cpu},
            {"diffusion_flash_attn", &params->ctxParams.diffusion_flash_attn},
            {"taesd_preview", &params->ctxParams.taesd_preview},
            {"diffusion_conv_direct", &params->ctxParams.diffusion_conv_direct},
            {"vae_conv_direct", &params->ctxParams.vae_conv_direct},
            {"force_sdxl_vae_conv_scale", &params->ctxParams.force_sdxl_vae_conv_scale},
            {"chroma_use_dit_mask", &params->ctxParams.chroma_use_dit_mask},
            {"chroma_use_t5_mask", &params->ctxParams.chroma_use_t5_mask},
        },
        // manual_options (oh boy there are a lot)
        {
            {"model", [&](const json& o) -> bool {
                 bool change     = false;
                 int model_index = o.get<int>();
                 if (model_index >= 0 && model_index < params->models_files.size()) {
                     std::filesystem::path new_path = std::filesystem::path(params->models_dir) / params->models_files[model_index];
                     std::string new_path_str       = new_path.string();
                     if (params->ctxParams.model_path != new_path_str) {
                         params->ctxParams.model_path           = new_path_str;
                         params->ctxParams.diffusion_model_path = "";
                         change                                 = true;
                     }
                 } else {
                     if (model_index == MODEL_UNLOAD) {
                         if (params->ctxParams.model_path != "") {
                             change = true;
                         }
                         params->ctxParams.model_path = "";
                     } else if (model_index != MODEL_KEEP) {
                         sd_log(sd_log_level_t::SD_LOG_WARN, "Invalid model index: %d\n", model_index);
                     }
                 }
                 return change;
             }},
            {"diffusion_model", [&](const json& o) -> bool {
                 bool change     = false;
                 int model_index = o.get<int>();
                 if (model_index >= 0 && model_index < params->diffusion_models_files.size()) {
                     std::filesystem::path new_path = std::filesystem::path(params->diffusion_models_dir) / params->diffusion_models_files[model_index];
                     std::string new_path_str       = new_path.string();
                     if (params->ctxParams.diffusion_model_path != new_path_str) {
                         params->ctxParams.diffusion_model_path = new_path_str;
                         params->ctxParams.model_path           = "";
                         change                                 = true;
                     }
                 } else {
                     if (model_index == MODEL_UNLOAD) {
                         if (params->ctxParams.diffusion_model_path != "") {
                             change = true;
                         }
                         params->ctxParams.diffusion_model_path = "";
                     } else if (model_index != MODEL_KEEP) {
                         sd_log(sd_log_level_t::SD_LOG_WARN, "Invalid diffusion_model index: %d\n", model_index);
                     }
                 }
                 return change;
             }},
            {"high_noise_diffusion_model", [&](const json& o) -> bool {
                 bool change     = false;
                 int model_index = o.get<int>();
                 if (model_index >= 0 && model_index < params->diffusion_models_files.size()) {
                     std::filesystem::path new_path = std::filesystem::path(params->diffusion_models_dir) / params->diffusion_models_files[model_index];
                     std::string new_path_str       = new_path.string();
                     if (params->ctxParams.high_noise_diffusion_model_path != new_path_str) {
                         params->ctxParams.high_noise_diffusion_model_path = new_path_str;
                         change                                            = true;
                     }
                 } else {
                     if (model_index == MODEL_UNLOAD) {
                         if (params->ctxParams.high_noise_diffusion_model_path != "") {
                             change = true;
                         }
                         params->ctxParams.diffusion_model_path = "";
                     } else if (model_index != MODEL_KEEP) {
                         sd_log(sd_log_level_t::SD_LOG_WARN, "Invalid diffusion_model index: %d\n", model_index);
                     }
                 }
                 return change;
             }},
            {"clip_l", [&](const json& o) -> bool {
                 return parse_model_part(o, params->clip_files, params->clip_dir, params->ctxParams.clip_l_path);
             }},
            {"clip_g", [&](const json& o) -> bool {
                 return parse_model_part(o, params->clip_files, params->clip_dir, params->ctxParams.clip_g_path);
             }},
            {"clip_vision", [&](const json& o) -> bool {
                 return parse_model_part(o, params->clip_vision_files, params->clip_vision_dir, params->ctxParams.clip_vision_path);
             }},
            {"t5xxl", [&](const json& o) -> bool {
                 return parse_model_part(o, params->clip_files, params->clip_dir, params->ctxParams.t5xxl_path);
             }},
            {"vae", [&](const json& o) -> bool {
                 return parse_model_part(o, params->vae_files, params->vae_dir, params->ctxParams.vae_path);
             }},
            {"tae", [&](const json& o) -> bool {
                 return parse_model_part(o, params->tae_files, params->tae_dir, params->ctxParams.taesd_path);
             }},
            {"llm", [&](const json& o) -> bool {
                 return parse_model_part(o, params->clip_files, params->clip_dir, params->ctxParams.llm_path);
             }},
            {"llm_vision", [&](const json& o) -> bool {
                 return parse_model_part(o, params->clip_vision_files, params->clip_vision_dir, params->ctxParams.llm_vision_path);
             }},
            {"control_net", [&](const json& o) -> bool {
                 return parse_model_part(o, params->controlnet_files, params->controlnet_dir, params->ctxParams.control_net_path);
             }},
            // skip lora_model_dir and embeddings (only set via cli args)
            {"photo_maker", [&](const json& o) -> bool {
                 return parse_model_part(o, params->photomaker_files, params->photomaker_dir, params->ctxParams.photo_maker_path);
             }},
            {"upscale_model", [&](const json& o) -> bool {
                 return parse_model_part(o, params->upscaler_files, params->upscaler_dir, params->ctxParams.upscale_model_path);
             }},
            {"tensor_type_rules", [&](const json& o) -> bool {
                 // TODO
                 sd_log(sd_log_level_t::SD_LOG_WARN, "tensor_type_rules not implemented yet\n");
                 return false;
             }},
            {"wtype", [&](const json& o) -> bool {
                 bool change      = false;
                 std::string type = o.get<std::string>();
                 if (type != "") {
                     bool found = false;
                     auto wtype = str_to_sd_type(type.c_str());
                     if (wtype != SD_TYPE_COUNT) {
                         found                   = true;
                         params->ctxParams.wtype = wtype;
                     }
                     if (!found) {
                         sd_log(sd_log_level_t::SD_LOG_WARN, "Unknown wtype: %s\n", type.c_str());
                     }
                 }
                 return change;
             }},
            {"rng_type", [&](const json& o) -> bool {
                 bool change          = false;
                 std::string type_str = o.get<std::string>();
                 if (type_str != "") {
                     bool found = false;
                     for (size_t i = 0; i < RNG_TYPE_COUNT; i++) {
                         enum rng_type_t type = (enum rng_type_t)i;
                         if (type_str == std::string(sd_rng_type_name(type))) {
                             if (type != params->ctxParams.rng_type) {
                                 change = true;
                             }
                             params->ctxParams.rng_type = type;
                             found                      = true;
                             break;
                         }
                     }
                     if (!found) {
                         sd_log(sd_log_level_t::SD_LOG_WARN, "Unknown rng_type: %s\n", type_str.c_str());
                     }
                 }
                 return change;
             }},
            {"sampler_rng_type", [&](const json& o) -> bool {
                 bool change          = false;
                 std::string type_str = o.get<std::string>();
                 if (type_str != "") {
                     bool found = false;
                     for (size_t i = 0; i < RNG_TYPE_COUNT; i++) {
                         enum rng_type_t type = (enum rng_type_t)i;
                         if (type_str == std::string(sd_rng_type_name(type))) {
                             if (type != params->ctxParams.sampler_rng_type) {
                                 change = true;
                             }
                             params->ctxParams.sampler_rng_type = type;
                             found                              = true;
                             break;
                         }
                     }
                     if (!found) {
                         sd_log(sd_log_level_t::SD_LOG_WARN, "Unknown sampler_rng_type: %s\n", type_str.c_str());
                     }
                 }
                 return change;
             }},
            {"prediction", [&](const json& o) -> bool {
                 bool change          = false;
                 std::string pred_str = o.get<std::string>();
                 if (pred_str != "") {
                     bool found = false;
                     for (size_t i = 0; i < PREDICTION_COUNT; i++) {
                         enum prediction_t pred = (enum prediction_t)i;
                         if (pred_str == std::string(sd_prediction_name(pred))) {
                             if (pred != params->ctxParams.prediction) {
                                 change = true;
                             }
                             params->ctxParams.prediction = pred;
                             found                        = true;
                             break;
                         }
                     }
                     if (!found) {
                         sd_log(sd_log_level_t::SD_LOG_WARN, "Unknown prediction: %s\n", pred_str.c_str());
                     }
                 }
                 return change;
             }},
            {"lora_apply_mode", [&](const json& o) -> bool {
                 bool change          = false;
                 std::string mode_str = o.get<std::string>();
                 if (mode_str != "") {
                     bool found = false;
                     for (size_t i = 0; i < LORA_APPLY_MODE_COUNT; i++) {
                         enum lora_apply_mode_t mode = (enum lora_apply_mode_t)i;
                         if (mode_str == std::string(sd_lora_apply_mode_name(mode))) {
                             if (mode != params->ctxParams.lora_apply_mode) {
                                 change = true;
                             }
                             params->ctxParams.lora_apply_mode = mode;
                             found                             = true;
                             break;
                         }
                     }
                     if (!found) {
                         sd_log(sd_log_level_t::SD_LOG_WARN, "Unknown lora_apply_mode: %s\n", mode_str.c_str());
                     }
                 }
                 return change;
             }},
        }};

    updatectx = parse_json_options(payload, ctx_options);

    // Legacy stuff (to keep webui working for now)
    // TODO: remove

    try {
        json guidance_params = payload["guidance_params"];
        try {
            float cfg_scale                                    = guidance_params["cfg_scale"];
            params->lastRequest.sample_params.guidance.txt_cfg = cfg_scale;
        } catch (...) {
        }
        try {
            float guidance                                                = guidance_params["guidance"];
            params->lastRequest.sample_params.guidance.distilled_guidance = guidance;
        } catch (...) {
        }
        try {
            json slg = guidance_params["slg"];
            try {
                params->lastRequest.skip_layers = slg["layers"].get<std::vector<int>>();
            } catch (...) {
            }
            try {
                float slg_scale                                      = slg["scale"];
                params->lastRequest.sample_params.guidance.slg.scale = slg_scale;
            } catch (...) {
            }
            try {
                float skip_layer_start                                     = slg["start"];
                params->lastRequest.sample_params.guidance.slg.layer_start = skip_layer_start;
            } catch (...) {
            }
            try {
                float skip_layer_end                                     = slg["end"];
                params->lastRequest.sample_params.guidance.slg.layer_end = skip_layer_end;
            } catch (...) {
            }
        } catch (...) {
        }
        // try {
        //     json apg = guidance_params["apg"];
        //     try {
        //         float apg_eta               = apg["eta"];
        //         params->lastRequest.apg_eta = apg_eta;
        //     } catch (...) {
        //     }
        //     try {
        //         float apg_momentum               = apg["momentum"];
        //         params->lastRequest.apg_momentum = apg_momentum;
        //     } catch (...) {
        //     }
        //     try {
        //         float apg_norm_threshold               = apg["norm_threshold"];
        //         params->lastRequest.apg_norm_threshold = apg_norm_threshold;
        //     } catch (...) {
        //     }
        //     try {
        //         float apg_norm_smoothing               = apg["norm_smoothing"];
        //         params->lastRequest.apg_norm_smoothing = apg_norm_smoothing;
        //     } catch (...) {
        //     }
        // } catch (...) {
        // }
    } catch (...) {
    }

    try {
        std::string sample_method = payload["sample_method"];

        sample_method_t sample_method_found = str_to_sample_method(sample_method.c_str());
        if (sample_method_found != SAMPLE_METHOD_COUNT) {
            params->lastRequest.sample_params.sample_method = sample_method_found;
        } else {
            sd_log(sd_log_level_t::SD_LOG_WARN, "Unknown sampling method: %s\n", sample_method.c_str());
        }
    } catch (...) {
    }
    try {
        std::string schedule       = payload["schedule"];
        scheduler_t schedule_found = str_to_scheduler(schedule.c_str());
        if (schedule_found != SCHEDULER_COUNT) {
            if (params->lastRequest.sample_params.scheduler != schedule_found) {
                params->lastRequest.sample_params.scheduler = schedule_found;
            }
        } else {
            sd_log(sd_log_level_t::SD_LOG_WARN, "Unknown schedule: %s\n", schedule.c_str());
        }
    } catch (...) {
    }
    try {
        int sample_steps                               = payload["sample_steps"];
        params->lastRequest.sample_params.sample_steps = sample_steps;
    } catch (...) {
    }

    try {
        std::string input_id_images_path = payload["input_id_images_path"];
        // TODO replace with b64 image maybe?
        params->input_id_images_path = input_id_images_path;
    } catch (...) {
    }
    try {
        bool vae_tiling = payload["vae_tiling"];
        if (params->lastRequest.tiling_params.enabled != vae_tiling) {
            params->lastRequest.tiling_params.enabled = vae_tiling;
        }
    } catch (...) {
    }

    try {
        std::string preview = payload["preview_mode"];
        int preview_found   = -1;
        for (int m = 0; m < PREVIEW_COUNT; m++) {
            if (!strcmp(preview.c_str(), sd_preview_name((preview_t)m))) {
                preview_found = m;
            }
        }
        if (preview_found >= 0) {
            if (params->lastRequest.preview_method != (preview_t)preview_found) {
                params->lastRequest.preview_method = (preview_t)preview_found;
            }
        } else {
            sd_log(sd_log_level_t::SD_LOG_WARN, "Unknown preview: %s\n", preview.c_str());
        }
    } catch (...) {
    }

    // ctxParams zone

    try {
        // renamed to wtype in new API
        std::string type = payload["type"];
        if (type != "") {
            bool found = false;
            auto wtype = str_to_sd_type(type.c_str());
            if (wtype != SD_TYPE_COUNT) {
                found                   = true;
                params->ctxParams.wtype = wtype;
            }
            if (!found) {
                sd_log(sd_log_level_t::SD_LOG_WARN, "Unknown type: %s\n", type.c_str());
            }
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
                auto relative_path   = entry.path().lexically_relative(dir_path);
                std::string path_str = relative_path.string();
                std::replace(path_str.begin(), path_str.end(), '\\', '/');
                files.push_back(path_str);
            }
        }
    return files;
}

// 1. Data Structure for persistent queueing
struct TaskData {
    std::string task_id;
    std::string req_body;  // Raw JSON to be parsed later
};

// 2. Global Contexts (Must exist globally to be freed/reloaded)
SDParams g_params;
sd_ctx_t* g_sd_ctx = NULL;
int g_n_prompts    = 0;

// 3. Queue & Thread Synchronization
std::queue<TaskData> g_task_queue;
std::mutex g_queue_mutex;
std::condition_variable g_queue_cond;
std::atomic<bool> g_stop_worker(false);
std::atomic<bool> g_is_busy(false);
std::string g_running_task_id("");

// 4. Results & Cancellation
std::unordered_map<std::string, nlohmann::json> g_task_results;
std::mutex g_results_mutex;
std::unordered_set<std::string> g_cancelled_tasks;  // To skip queued tasks
std::mutex g_cancel_mutex;
std::atomic<bool> g_abort_flag(false);  // To signal running task to stop

const char* g_preview_path;
float g_preview_fps = 24;  // TODO : video

void step_callback(int step, int frame_count, sd_image_t* image, bool is_noisy, void* data) {
    (void)data;
    using json = nlohmann::json;
    if (frame_count > 1) {
        return;
    }

    if (g_preview_path) {
        stbi_write_png(g_preview_path, image->width, image->height, image->channel, image->data, 0);
    }

    json task_json = g_task_results[g_running_task_id];
    if (task_json == NULL) {
        // shouldn't happen
        task_json = json::object();
    }
    task_json["status"] = "Working";
    task_json["data"]   = json::array();

    int len;
    unsigned char* png = stbi_write_png_to_mem((const unsigned char*)image->data, 0, image->width, image->height, image->channel, &len, NULL);
    std::string data_str(png, png + len);
    std::string encoded_img = base64_encode(data_str);
    task_json["data"].push_back({{"width", image->width},
                                 {"height", image->height},
                                 {"channel", image->channel},
                                 {"data", encoded_img},
                                 {"encoding", "png"}});

    std::lock_guard<std::mutex> results_lock(g_results_mutex);
    g_task_results[g_running_task_id] = task_json;
}

void process_generation_task(TaskData task) {
    g_running_task_id = task.task_id;
    g_abort_flag      = false;
    using json        = nlohmann::json;

    bool updateCTX = g_params.ctxParams.free_params_immediately;
    try {
        updateCTX = parseJsonPrompt(task.req_body, &g_params) || updateCTX;
    } catch (json::parse_error& e) {
        sd_log(sd_log_level_t::SD_LOG_WARN, "Failed to parse json: %s\n Assuming it's just a prompt...\n", e.what());
        std::string prompt = task.req_body;
        if (!prompt.empty()) {
            g_params.lastRequest.prompt = prompt;
        } else {
            g_params.lastRequest.seed += 1;
        }
    } catch (...) {
        sd_log(sd_log_level_t::SD_LOG_ERROR, "An unexpected error occurred\n");
    }

    {
        std::lock_guard<std::mutex> lock(g_cancel_mutex);
        if (g_cancelled_tasks.count(task.task_id)) {
            g_cancelled_tasks.erase(task.task_id);
            std::lock_guard<std::mutex> res_lock(g_results_mutex);
            g_task_results[task.task_id]["status"] = "Cancelled";
            g_running_task_id                      = "";
            if (updateCTX) {
                // ctx g_params have been changed, must cleanup before next uncancelled task
                free_sd_ctx(g_sd_ctx);
                g_sd_ctx = NULL;
            }
            sd_log(sd_log_level_t::SD_LOG_INFO, "Task %s was cancelled before execution, skipping\n", task.task_id.c_str());
            return;
        }
    }

    sd_log(sd_log_level_t::SD_LOG_INFO, "prompt is: %s\n", g_params.lastRequest.prompt.c_str());

    // 3. Reload Context if needed
    if (updateCTX && g_sd_ctx != NULL) {
        free_sd_ctx(g_sd_ctx);
        g_sd_ctx = NULL;
    }
    if (g_sd_ctx == NULL) {
        printf("Loading sd_ctx\n");
        {
            json task_json      = json::object();
            task_json["status"] = "Loading";
            task_json["data"]   = json::array();
            task_json["step"]   = -1;
            task_json["steps"]  = 0;
            task_json["eta"]    = "?";

            std::lock_guard<std::mutex> results_lock(g_results_mutex);
            g_task_results[task.task_id] = task_json;
        }
        sd_ctx_params_t sd_ctx_params = {
            g_params.ctxParams.model_path.c_str(),
            g_params.ctxParams.clip_l_path.c_str(),
            g_params.ctxParams.clip_g_path.c_str(),
            g_params.ctxParams.clip_vision_path.c_str(),
            g_params.ctxParams.t5xxl_path.c_str(),
            g_params.ctxParams.llm_path.c_str(),
            g_params.ctxParams.llm_vision_path.c_str(),
            g_params.ctxParams.diffusion_model_path.c_str(),
            g_params.ctxParams.high_noise_diffusion_model_path.c_str(),
            g_params.ctxParams.vae_path.c_str(),
            g_params.ctxParams.taesd_path.c_str(),
            g_params.ctxParams.control_net_path.c_str(),
            g_params.ctxParams.lora_model_dir.c_str(),
            g_params.ctxParams.embeddings_path.c_str(),
            g_params.ctxParams.photo_maker_path.c_str(),
            g_params.ctxParams.tensor_type_rules.c_str(),
            g_params.ctxParams.vae_decode_only,
            g_params.ctxParams.free_params_immediately,
            g_params.ctxParams.n_threads,
            g_params.ctxParams.wtype,
            g_params.ctxParams.rng_type,
            g_params.ctxParams.sampler_rng_type,
            g_params.ctxParams.prediction,
            g_params.ctxParams.lora_apply_mode,
            g_params.ctxParams.offload_params_to_cpu,
            g_params.ctxParams.keep_clip_on_cpu,
            g_params.ctxParams.keep_control_net_on_cpu,
            g_params.ctxParams.keep_vae_on_cpu,
            g_params.ctxParams.diffusion_flash_attn,
            g_params.ctxParams.taesd_preview,
            g_params.ctxParams.diffusion_conv_direct,
            g_params.ctxParams.vae_conv_direct,
            g_params.ctxParams.force_sdxl_vae_conv_scale,
            g_params.ctxParams.chroma_use_dit_mask,
            g_params.ctxParams.chroma_use_t5_mask,
            g_params.ctxParams.chroma_t5_mask_pad,
            g_params.ctxParams.flow_shift};

        g_sd_ctx = new_sd_ctx(&sd_ctx_params);
        if (g_sd_ctx == NULL) {
            printf("new_sd_ctx_t failed\n");
            std::lock_guard<std::mutex> results_lock(g_results_mutex);
            g_task_results[task.task_id]["status"] = "Failed";
            return;
        }
    }

    // 4. Update Status
    {
        std::lock_guard<std::mutex> lock(g_results_mutex);
        g_task_results[task.task_id]["status"] = "Working";
        g_task_results[task.task_id]["step"]   = 0;
        g_task_results[task.task_id]["steps"]  = g_params.lastRequest.sample_params.sample_steps;
        g_task_results[task.task_id]["eta"]    = "?";
        g_task_results[task.task_id]["data"]   = json::array();
    }

    {
        sd_image_t* results;
        g_params.lastRequest.sample_params.guidance.slg.layers      = g_params.lastRequest.skip_layers.data();
        g_params.lastRequest.sample_params.guidance.slg.layer_count = g_params.lastRequest.skip_layers.size();

        sd_image_t init_image = g_params.lastRequest.init_image;

        sd_image_t mask_img = g_params.lastRequest.mask_image;
        std::vector<uint8_t> ones(g_params.lastRequest.width * g_params.lastRequest.height, 0xFF);
        if (init_image.data != NULL && g_params.lastRequest.mask_image.data == NULL) {
            mask_img = {
                (uint32_t)g_params.lastRequest.width,
                (uint32_t)g_params.lastRequest.height,
                1,
                ones.data()};
        }

        sd_image_t control_img = g_params.lastRequest.control_image;

        g_params.lastRequest.pm_params.id_embed_path   = g_params.input_id_images_path.c_str();
        g_params.lastRequest.pm_params.id_images       = g_params.lastRequest.pm_images_vec.data();
        g_params.lastRequest.pm_params.id_images_count = g_params.lastRequest.pm_images_vec.size();

        sd_img_gen_params_t gen_params = {
            g_params.lastRequest.prompt.c_str(),
            g_params.lastRequest.negative_prompt.c_str(),
            g_params.lastRequest.clip_skip,
            init_image,
            g_params.lastRequest.ref_images.data(),
            (int)g_params.lastRequest.ref_images.size(),
            g_params.lastRequest.auto_resize_ref_image,
            g_params.lastRequest.increase_ref_index,
            mask_img,
            g_params.lastRequest.width,
            g_params.lastRequest.height,
            g_params.lastRequest.sample_params,
            g_params.lastRequest.strength,
            g_params.lastRequest.seed,
            g_params.lastRequest.batch_count,
            control_img,
            g_params.lastRequest.control_strength,
            g_params.lastRequest.pm_params,
            g_params.lastRequest.tiling_params};
        sd_set_preview_callback((sd_preview_cb_t)step_callback, g_params.lastRequest.preview_method, g_params.lastRequest.preview_interval, !g_params.lastRequest.preview_noisy, g_params.lastRequest.preview_noisy, NULL);

        results = generate_image(g_sd_ctx, &gen_params);

        if (results == NULL) {
            printf("generate failed\n");
            free_sd_ctx(g_sd_ctx);
            std::lock_guard<std::mutex> g_results_lock(g_results_mutex);
            g_task_results[task.task_id]["status"] = "Failed";
            return;
        }

        size_t last            = g_params.output_path.find_last_of(".");
        std::string dummy_name = last != std::string::npos ? g_params.output_path.substr(0, last) : g_params.output_path;
        json images_json       = json::array();
        for (int i = 0; i < g_params.lastRequest.batch_count; i++) {
            if (results[i].data == NULL) {
                continue;
            }
            // TODO allow disable save to disk
            std::string final_image_path = i > 0 ? dummy_name + "_" + std::to_string(i + 1 + g_n_prompts * g_params.lastRequest.batch_count) + ".png" : dummy_name + ".png";
            stbi_write_png(final_image_path.c_str(), results[i].width, results[i].height, results[i].channel,
                           results[i].data, 0, get_image_params(g_params, g_params.lastRequest.seed + i).c_str());
            printf("save result image to '%s'\n", final_image_path.c_str());
            // Todo: return base64 encoded image via httplib::Response& res

            int len;
            unsigned char* png = stbi_write_png_to_mem((const unsigned char*)results[i].data, 0, results[i].width, results[i].height, results[i].channel, &len, get_image_params(g_params, g_params.lastRequest.seed + i).c_str());

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
        g_n_prompts++;
        // res.set_content(images_json.dump(), "application/json");
        json end_task_json      = json::object();
        end_task_json["status"] = "Done";
        end_task_json["data"]   = images_json;
        end_task_json["step"]   = -1;
        end_task_json["steps"]  = 0;
        end_task_json["eta"]    = "?";
        std::lock_guard<std::mutex> results_lock(g_results_mutex);
        g_task_results[task.task_id] = end_task_json;
    }
    g_running_task_id = "";
}

void worker_thread() {
    while (!g_stop_worker) {
        TaskData task;
        bool has_task = false;
        {
            std::unique_lock<std::mutex> lock(g_queue_mutex);
            g_queue_cond.wait(lock, [] { return !g_task_queue.empty() || g_stop_worker; });
            if (!g_task_queue.empty() && !g_stop_worker) {
                task = g_task_queue.front();
                g_task_queue.pop();
                has_task  = true;
                g_is_busy = true;
            }
        }
        if (has_task) {
            process_generation_task(task);
            g_is_busy = false;
        }
    }
}

void update_progress_cb(int step, int steps, float time, void* _data) {
    using json = nlohmann::json;
    if (g_running_task_id != "") {
        std::lock_guard<std::mutex> results_lock(g_results_mutex);
        json running_task_json = g_task_results[g_running_task_id];
        if (running_task_json["status"] == "Working" && running_task_json["step"] == running_task_json["steps"]) {
            running_task_json["status"] = "Decoding";
        }
        running_task_json["step"]         = step;
        running_task_json["steps"]        = steps;
        g_task_results[g_running_task_id] = running_task_json;
    }
}

bool is_model_file(const std::string& path) {
    size_t name_start = path.find_last_of("/\\");
    if (name_start == std::string::npos) {
        name_start = 0;
    }
    size_t extension_start = path.substr(name_start).find_last_of(".");
    if (extension_start == std::string::npos) {
        return false;  // No extension
    }
    std::string file_extension = path.substr(name_start + extension_start + 1);
    return (file_extension == "gguf" || file_extension == "safetensors" || file_extension == "sft" || file_extension == "ckpt");
}

nlohmann::json serv_generate_image(const httplib::Request& req) {
    using json          = nlohmann::json;
    std::string task_id = std::to_string(std::chrono::system_clock::now().time_since_epoch().count());

    {
        json pending_task_json      = json::object();
        pending_task_json["status"] = "Pending";
        pending_task_json["data"]   = json::array();
        pending_task_json["step"]   = -1;
        pending_task_json["steps"]  = 0;
        pending_task_json["eta"]    = "?";

        std::lock_guard<std::mutex> results_lock(g_results_mutex);
        g_task_results[task_id] = pending_task_json;
    }

    // Add the task to the queue
    {
        std::lock_guard<std::mutex> lock(g_queue_mutex);
        g_task_queue.push({task_id, req.body});
    }
    g_queue_cond.notify_one();

    json response       = json::object();
    response["task_id"] = task_id;
    return response;
}

// Helper to safely get a string from json
std::string j_get_str(const nlohmann::json& j, const std::string& key, const std::string& def) {
    if (j.contains(key) && j[key].is_string())
        return j[key].get<std::string>();
    return def;
}

// Helper to safely get primitives
template <typename T>
T j_get(const nlohmann::json& j, const std::string& key, T def) {
    if (j.contains(key))
        return j[key].get<T>();
    return def;
}

nlohmann::json serialize_server_state(const SDParams& p) {
    using json = nlohmann::json;
    json j;

    j["prompt"]          = p.lastRequest.prompt;
    j["negative_prompt"] = p.lastRequest.negative_prompt;
    j["clip_skip"]       = p.lastRequest.clip_skip;

    j["init_image"] = {
        {"width", p.lastRequest.init_image.width},
        {"height", p.lastRequest.init_image.height},
        {"channel", p.lastRequest.init_image.channel}};

    j["auto_resize_ref_image"] = p.lastRequest.auto_resize_ref_image;
    j["increase_ref_index"]    = p.lastRequest.increase_ref_index;

    j["mask_image"] = {
        {"width", p.lastRequest.mask_image.width},
        {"height", p.lastRequest.mask_image.height},
        {"channel", p.lastRequest.mask_image.channel}};

    j["width"]       = p.lastRequest.width;
    j["height"]      = p.lastRequest.height;
    j["skip_layers"] = p.lastRequest.skip_layers;

    j["sample_params"] = {
        {"guidance", {{"txt_cfg", p.lastRequest.sample_params.guidance.txt_cfg}, {"img_cfg", p.lastRequest.sample_params.guidance.img_cfg}, {"distilled_guidance", p.lastRequest.sample_params.guidance.distilled_guidance}, {"slg", {
                                                                                                                                                                                                                                      {"layer_count", p.lastRequest.sample_params.guidance.slg.layer_count},
                                                                                                                                                                                                                                      {"layer_start", p.lastRequest.sample_params.guidance.slg.layer_start},
                                                                                                                                                                                                                                      {"layer_end", p.lastRequest.sample_params.guidance.slg.layer_end},
                                                                                                                                                                                                                                      {"scale", p.lastRequest.sample_params.guidance.slg.scale}}}}},
        {"scheduler", sd_scheduler_name(p.lastRequest.sample_params.scheduler)},
        {"sample_method", sd_sample_method_name(p.lastRequest.sample_params.sample_method)},
        {"sample_steps", p.lastRequest.sample_params.sample_steps},
        {"eta", p.lastRequest.sample_params.eta},
        {"shifted_timestep", p.lastRequest.sample_params.shifted_timestep}};

    j["strength"]    = p.lastRequest.strength;
    j["seed"]        = p.lastRequest.seed;
    j["batch_count"] = p.lastRequest.batch_count;

    j["control_image"] = {
        {"width", p.lastRequest.control_image.width},
        {"height", p.lastRequest.control_image.height},
        {"channel", p.lastRequest.control_image.channel}};
    j["control_strength"] = p.lastRequest.control_strength;


    j["pm_id_embed_path"] = p.lastRequest.pm_id_embed_path;
    j["pm_params"]        = {
        {"id_images_count", p.lastRequest.pm_params.id_images_count},
        {"style_strength", p.lastRequest.pm_params.style_strength}};

    j["tiling_params"] = {
        {"enabled", p.lastRequest.tiling_params.enabled},
        {"tile_size_x", p.lastRequest.tiling_params.tile_size_x},
        {"tile_size_y", p.lastRequest.tiling_params.tile_size_y},
        {"target_overlap", p.lastRequest.tiling_params.target_overlap},
        {"rel_size_x", p.lastRequest.tiling_params.rel_size_x},
        {"rel_size_y", p.lastRequest.tiling_params.rel_size_y}};

    j["upscale"]          = p.lastRequest.upscale;
    j["preview_method"]   = sd_preview_name(p.lastRequest.preview_method);
    j["preview_interval"] = p.lastRequest.preview_interval;
    j["preview_noisy"]    = p.lastRequest.preview_noisy;

    // --- Context Params (Models & Hardware) ---

    j["model"]                      = p.ctxParams.model_path;
    j["clip_l"]                     = p.ctxParams.clip_l_path;
    j["clip_g"]                     = p.ctxParams.clip_g_path;
    j["clip_vision"]                = p.ctxParams.clip_vision_path;
    j["t5xxl"]                      = p.ctxParams.t5xxl_path;
    j["llm"]                        = p.ctxParams.llm_path;
    j["llm_vision"]                 = p.ctxParams.llm_vision_path;
    j["diffusion_model"]            = p.ctxParams.diffusion_model_path;
    j["high_noise_diffusion_model"] = p.ctxParams.high_noise_diffusion_model_path;
    j["vae"]                        = p.ctxParams.vae_path;
    j["tae"]                        = p.ctxParams.taesd_path;

    j["control_net"]   = p.ctxParams.control_net_path;
    j["photo_maker"]   = p.ctxParams.photo_maker_path;
    j["upscale_model"] = p.ctxParams.upscale_model_path;

    j["vae_decode_only"]         = p.ctxParams.vae_decode_only;
    j["free_params_immediately"] = p.ctxParams.free_params_immediately;
    j["n_threads"]               = p.ctxParams.n_threads;
    if (p.ctxParams.wtype < SD_TYPE_COUNT) {
        j["wtype"] = sd_type_name(p.ctxParams.wtype);
    }

    j["rng_type"]         = sd_rng_type_name(p.ctxParams.rng_type);
    j["sampler_rng_type"] = sd_rng_type_name(p.ctxParams.sampler_rng_type);
    j["prediction"]       = sd_prediction_name(p.ctxParams.prediction);
    j["lora_apply_mode"]  = sd_lora_apply_mode_name(p.ctxParams.lora_apply_mode);

    j["offload_params_to_cpu"]   = p.ctxParams.offload_params_to_cpu;
    j["keep_clip_on_cpu"]        = p.ctxParams.keep_clip_on_cpu;
    j["keep_control_net_on_cpu"] = p.ctxParams.keep_control_net_on_cpu;
    j["keep_vae_on_cpu"]         = p.ctxParams.keep_vae_on_cpu;

    j["diffusion_flash_attn"]      = p.ctxParams.diffusion_flash_attn;
    j["taesd_preview"]             = p.ctxParams.taesd_preview;
    j["diffusion_conv_direct"]     = p.ctxParams.diffusion_conv_direct;
    j["force_sdxl_vae_conv_scale"] = p.ctxParams.force_sdxl_vae_conv_scale;
    j["vae_conv_direct"]           = p.ctxParams.vae_conv_direct;
    j["chroma_use_dit_mask"]       = p.ctxParams.chroma_use_dit_mask;
    j["chroma_use_t5_mask"]        = p.ctxParams.chroma_use_t5_mask;
    j["chroma_t5_mask_pad"]        = p.ctxParams.chroma_t5_mask_pad;
    j["flow_shift"]                = p.ctxParams.flow_shift;
    return j;
}

void deserialize_server_state(SDParams& p, const nlohmann::json& j) {
    // --- Request Params ---
    p.lastRequest.prompt          = j_get_str(j, "prompt", p.lastRequest.prompt);
    p.lastRequest.negative_prompt = j_get_str(j, "negative_prompt", p.lastRequest.negative_prompt);
    p.lastRequest.clip_skip       = j_get(j, "clip_skip", p.lastRequest.clip_skip);

    if (j.contains("init_image")) {
        p.lastRequest.init_image.width   = j_get(j["init_image"], "width", p.lastRequest.init_image.width);
        p.lastRequest.init_image.height  = j_get(j["init_image"], "height", p.lastRequest.init_image.height);
        p.lastRequest.init_image.channel = j_get(j["init_image"], "channel", p.lastRequest.init_image.channel);
        p.lastRequest.init_image.data    = NULL;
    }

    p.lastRequest.auto_resize_ref_image = j_get(j, "auto_resize_ref_image", p.lastRequest.auto_resize_ref_image);
    p.lastRequest.increase_ref_index    = j_get(j, "increase_ref_index", p.lastRequest.increase_ref_index);

    if (j.contains("mask_image")) {
        p.lastRequest.mask_image.width   = j_get(j["mask_image"], "width", p.lastRequest.mask_image.width);
        p.lastRequest.mask_image.height  = j_get(j["mask_image"], "height", p.lastRequest.mask_image.height);
        p.lastRequest.mask_image.channel = j_get(j["mask_image"], "channel", p.lastRequest.mask_image.channel);
        p.lastRequest.mask_image.data    = NULL;
    }

    p.lastRequest.width       = j_get(j, "width", p.lastRequest.width);
    p.lastRequest.height      = j_get(j, "height", p.lastRequest.height);
    p.lastRequest.skip_layers = j_get<std::vector<int>>(j, "skip_layers", p.lastRequest.skip_layers);

    if (j.contains("sample_params")) {
        if (j["sample_params"].contains("guidance")) {
            p.lastRequest.sample_params.guidance.txt_cfg            = j_get(j["sample_params"]["guidance"], "txt_cfg", p.lastRequest.sample_params.guidance.txt_cfg);
            p.lastRequest.sample_params.guidance.img_cfg            = j_get(j["sample_params"]["guidance"], "img_cfg", p.lastRequest.sample_params.guidance.img_cfg);
            p.lastRequest.sample_params.guidance.distilled_guidance = j_get(j["sample_params"]["guidance"], "distilled_guidance", p.lastRequest.sample_params.guidance.distilled_guidance);
            if (j["sample_params"]["guidance"].contains("slg")) {
                p.lastRequest.sample_params.guidance.slg.layers      = NULL;
                p.lastRequest.sample_params.guidance.slg.layer_count = j_get(j["sample_params"]["guidance"]["slg"], "layer_count", p.lastRequest.sample_params.guidance.slg.layer_count);
                p.lastRequest.sample_params.guidance.slg.layer_start = j_get(j["sample_params"]["guidance"]["slg"], "layer_start", p.lastRequest.sample_params.guidance.slg.layer_start);
                p.lastRequest.sample_params.guidance.slg.layer_end   = j_get(j["sample_params"]["guidance"]["slg"], "layer_end", p.lastRequest.sample_params.guidance.slg.layer_end);
                p.lastRequest.sample_params.guidance.slg.scale       = j_get(j["sample_params"]["guidance"]["slg"], "scale", p.lastRequest.sample_params.guidance.slg.scale);
            }
        }
        if (j["sample_params"].contains("scheduler")) {
            p.lastRequest.sample_params.scheduler = str_to_scheduler(j_get_str(j["sample_params"], "scheduler", sd_scheduler_name(p.lastRequest.sample_params.scheduler)).c_str());
        }
        if (j["sample_params"].contains("sample_method")) {
            p.lastRequest.sample_params.sample_method = str_to_sample_method(j_get_str(j["sample_params"], "sample_method", sd_sample_method_name(p.lastRequest.sample_params.sample_method)).c_str());
        }
        p.lastRequest.sample_params.sample_steps     = j_get(j["sample_params"], "sample_steps", p.lastRequest.sample_params.sample_steps);
        p.lastRequest.sample_params.eta              = j_get(j["sample_params"], "eta", p.lastRequest.sample_params.eta);
        p.lastRequest.sample_params.shifted_timestep = j_get(j["sample_params"], "shifted_timestep", p.lastRequest.sample_params.shifted_timestep);
    }

    p.lastRequest.strength    = j_get(j, "strength", p.lastRequest.strength);
    p.lastRequest.seed        = j_get(j, "seed", p.lastRequest.seed);
    p.lastRequest.batch_count = j_get(j, "batch_count", p.lastRequest.batch_count);

    if (j.contains("control_image")) {
        p.lastRequest.control_image.width   = j_get(j["control_image"], "width", p.lastRequest.control_image.width);
        p.lastRequest.control_image.height  = j_get(j["control_image"], "height", p.lastRequest.control_image.height);
        p.lastRequest.control_image.channel = j_get(j["control_image"], "channel", p.lastRequest.control_image.channel);
        p.lastRequest.control_image.data    = NULL;
    }

    p.lastRequest.control_strength = j_get(j, "control_strength", p.lastRequest.control_strength);
    p.lastRequest.pm_id_embed_path = j_get_str(j, "pm_id_embed_path", p.lastRequest.pm_id_embed_path);

    if (j.contains("pm_params")) {
        p.lastRequest.pm_params.id_images_count = j_get(j["pm_params"], "id_images_count", p.lastRequest.pm_params.id_images_count);
        p.lastRequest.pm_params.style_strength  = j_get(j["pm_params"], "style_strength", p.lastRequest.pm_params.style_strength);
    }

    if (j.contains("tiling_params")) {
        p.lastRequest.tiling_params.enabled        = j_get(j["tiling_params"], "enabled", p.lastRequest.tiling_params.enabled);
        p.lastRequest.tiling_params.tile_size_x    = j_get(j["tiling_params"], "tile_size_x", p.lastRequest.tiling_params.tile_size_x);
        p.lastRequest.tiling_params.tile_size_y    = j_get(j["tiling_params"], "tile_size_y", p.lastRequest.tiling_params.tile_size_y);
        p.lastRequest.tiling_params.target_overlap = j_get(j["tiling_params"], "overlap", p.lastRequest.tiling_params.target_overlap);
        p.lastRequest.tiling_params.rel_size_x     = j_get(j["tiling_params"], "rel_size_x", p.lastRequest.tiling_params.rel_size_x);
        p.lastRequest.tiling_params.rel_size_y     = j_get(j["tiling_params"], "rel_size_y", p.lastRequest.tiling_params.rel_size_y);
    }

    p.lastRequest.upscale          = j_get(j, "upscale", p.lastRequest.upscale);
    p.lastRequest.preview_method   = str_to_preview(j_get_str(j, "preview_method", sd_preview_name(p.lastRequest.preview_method)).c_str());
    p.lastRequest.preview_interval = j_get(j, "preview_interval", p.lastRequest.preview_interval);
    p.lastRequest.preview_noisy    = j_get(j, "preview_noisy", p.lastRequest.preview_noisy);

    // --- Context Params (Models & Hardware) ---
    p.ctxParams.model_path                      = j_get_str(j, "model", p.ctxParams.model_path);
    p.ctxParams.clip_l_path                     = j_get_str(j, "clip_l", p.ctxParams.clip_l_path);
    p.ctxParams.clip_g_path                     = j_get_str(j, "clip_g", p.ctxParams.clip_g_path);
    p.ctxParams.clip_vision_path                = j_get_str(j, "clip_vision", p.ctxParams.clip_vision_path);
    p.ctxParams.t5xxl_path                      = j_get_str(j, "t5xxl", p.ctxParams.t5xxl_path);
    p.ctxParams.llm_path                        = j_get_str(j, "llm", p.ctxParams.llm_path);
    p.ctxParams.llm_vision_path                 = j_get_str(j, "llm_vision", p.ctxParams.llm_vision_path);
    p.ctxParams.diffusion_model_path            = j_get_str(j, "diffusion_model", p.ctxParams.diffusion_model_path);
    p.ctxParams.high_noise_diffusion_model_path = j_get_str(j, "high_noise_diffusion_model", p.ctxParams.high_noise_diffusion_model_path);
    p.ctxParams.vae_path                        = j_get_str(j, "vae", p.ctxParams.vae_path);
    p.ctxParams.taesd_path                      = j_get_str(j, "tae", p.ctxParams.taesd_path);
    p.ctxParams.control_net_path                = j_get_str(j, "control_net", p.ctxParams.control_net_path);
    p.ctxParams.photo_maker_path                = j_get_str(j, "photo_maker", p.ctxParams.photo_maker_path);
    p.ctxParams.upscale_model_path              = j_get_str(j, "upscale_model", p.ctxParams.upscale_model_path);

    p.ctxParams.vae_decode_only         = j_get(j, "vae_decode_only", p.ctxParams.vae_decode_only);
    p.ctxParams.free_params_immediately = j_get(j, "free_params_immediately", p.ctxParams.free_params_immediately);
    p.ctxParams.n_threads               = j_get(j, "n_threads", p.ctxParams.n_threads);

    if (j.contains("wtype")) {
        p.ctxParams.wtype = str_to_sd_type(j_get_str(j, "wtype", sd_type_name(p.ctxParams.wtype)).c_str());
    }

    if (j.contains("rng_type")) {
        p.ctxParams.rng_type = str_to_rng_type(j_get_str(j, "rng_type", sd_rng_type_name(p.ctxParams.rng_type)).c_str());
    }

    if (j.contains("sampler_rng_type")) {
        p.ctxParams.sampler_rng_type = str_to_rng_type(j_get_str(j, "sampler_rng_type", sd_rng_type_name(p.ctxParams.sampler_rng_type)).c_str());
    }

    if (j.contains("prediction")) {
        p.ctxParams.prediction = str_to_prediction(j_get_str(j, "prediction", sd_prediction_name(p.ctxParams.prediction)).c_str());
    }

    if (j.contains("lora_apply_mode")) {
        p.ctxParams.lora_apply_mode = str_to_lora_apply_mode(j_get_str(j, "lora_apply_mode", sd_lora_apply_mode_name(p.ctxParams.lora_apply_mode)).c_str());
    }

    p.ctxParams.offload_params_to_cpu     = j_get(j, "offload_params_to_cpu", p.ctxParams.offload_params_to_cpu);
    p.ctxParams.keep_clip_on_cpu          = j_get(j, "keep_clip_on_cpu", p.ctxParams.keep_clip_on_cpu);
    p.ctxParams.keep_control_net_on_cpu   = j_get(j, "keep_control_net_on_cpu", p.ctxParams.keep_control_net_on_cpu);
    p.ctxParams.keep_vae_on_cpu           = j_get(j, "keep_vae_on_cpu", p.ctxParams.keep_vae_on_cpu);
    p.ctxParams.diffusion_flash_attn      = j_get(j, "diffusion_flash_attn", p.ctxParams.diffusion_flash_attn);
    p.ctxParams.taesd_preview             = j_get(j, "taesd_preview", p.ctxParams.taesd_preview);
    p.ctxParams.diffusion_conv_direct     = j_get(j, "diffusion_conv_direct", p.ctxParams.diffusion_conv_direct);
    p.ctxParams.force_sdxl_vae_conv_scale = j_get(j, "force_sdxl_vae_conv_scale", p.ctxParams.force_sdxl_vae_conv_scale);
    p.ctxParams.vae_conv_direct           = j_get(j, "vae_conv_direct", p.ctxParams.vae_conv_direct);
    p.ctxParams.chroma_use_dit_mask       = j_get(j, "chroma_use_dit_mask", p.ctxParams.chroma_use_dit_mask);
    p.ctxParams.chroma_use_t5_mask        = j_get(j, "chroma_use_t5_mask", p.ctxParams.chroma_use_t5_mask);
    p.ctxParams.chroma_t5_mask_pad        = j_get(j, "chroma_t5_mask_pad", p.ctxParams.chroma_t5_mask_pad);
    p.ctxParams.flow_shift                = j_get(j, "flow_shift", p.ctxParams.flow_shift);
}

void save_state_to_disk() {
    using json    = nlohmann::json;
    json root_obj = json::object();

    json queue_arr = json::array();
    std::lock_guard<std::mutex> lock(g_queue_mutex);
    std::queue<TaskData> temp_q = g_task_queue;
    while (!temp_q.empty()) {
        TaskData t = temp_q.front();
        temp_q.pop();
        queue_arr.push_back({{"task_id", t.task_id}, {"req_body", t.req_body}});
    }
    root_obj["queue"] = queue_arr;

    root_obj["system_state"] = serialize_server_state(g_params);

    std::ofstream o("server_state_dump.json");
    o << root_obj.dump(4) << std::endl;  // Pretty print
    o.flush();
    o.close();
    printf("Saved queue and system state to disk\n");
}

void load_state_from_disk() {
    using json = nlohmann::json;
    if (!std::filesystem::exists("server_state_dump.json"))
        return;

    std::ifstream i("server_state_dump.json");
    if (!i.good())
        return;

    try {
        json root_obj;
        i >> root_obj;

        if (root_obj.contains("system_state")) {
            deserialize_server_state(g_params, root_obj["system_state"]);
            sd_log(SD_LOG_INFO, "Restored system parameters (model: %s)\n",
                   sd_basename(g_params.ctxParams.model_path.empty() ? g_params.ctxParams.diffusion_model_path : g_params.ctxParams.model_path).c_str());
        }

        if (root_obj.contains("queue") && root_obj["queue"].is_array()) {
            std::lock_guard<std::mutex> lock(g_queue_mutex);
            for (auto& element : root_obj["queue"]) {
                if (element.contains("task_id") && element.contains("req_body")) {
                    TaskData t;
                    t.task_id  = element["task_id"].get<std::string>();
                    t.req_body = element["req_body"].get<std::string>();
                    g_task_queue.push(t);
                    g_task_results[t.task_id] = {{"status", "Pending (Restored)"}};
                }
            }
            sd_log(SD_LOG_INFO, "Restored %d tasks from disk.\n", (int)root_obj["queue"].size());
        }

    } catch (const std::exception& e) {
        sd_log(SD_LOG_WARN, "Failed to load queue dump: %s\n", e.what());
    } catch (...) {
        sd_log(SD_LOG_WARN, "Failed to load queue dump (unknown error).\n");
    }

    i.close();
    std::filesystem::remove("server_state_dump.json");
}

void trigger_hard_restart() {
    sd_log(SD_LOG_WARN, "Triggering HARD RESTART...\n");
    save_state_to_disk();
    // Force exit with code 111 to indicate a hard restart
#ifdef _WIN32
    TerminateProcess(GetCurrentProcess(), 111);
#else
    _exit(111);
    kill(getpid(), SIGKILL);
#endif
}

void start_server(SDParams& params) {
    g_preview_path = params.preview_path.c_str();
    sd_set_log_callback(sd_log_cb, (void*)&params);
    sd_set_progress_callback(update_progress_cb, NULL);

    params.models_files                  = list_files(params.models_dir);
    params.diffusion_models_files        = list_files(params.diffusion_models_dir);
    params.clip_files                    = list_files(params.clip_dir);
    params.clip_vision_files             = list_files(params.clip_vision_dir);
    params.vae_files                     = list_files(params.vae_dir);
    params.tae_files                     = list_files(params.tae_dir);
    params.controlnet_files              = list_files(params.controlnet_dir);
    params.photomaker_files              = list_files(params.photomaker_dir);
    params.upscaler_files                = list_files(params.upscaler_dir);
    std::vector<std::string> lora_files  = list_files(params.ctxParams.lora_model_dir);
    std::vector<std::string> embed_files = list_files(params.ctxParams.embeddings_path);

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

    // TODO new flag for extra verbose to log all requests
    // if (params.verbose) {
    //     svr->set_logger(log_server_request);
    // }

    svr->Post("/txt2img", [](const httplib::Request& req, httplib::Response& res) {
        // Deprecated
        sd_log(SD_LOG_WARN, "/txt2img endpoint is soon to be deprecated, use /generate_image instead");
        using json    = nlohmann::json;
        json response = serv_generate_image(req);
        res.set_content(response.dump(), "application/json");
    });

    svr->Post("/generate_image", [](const httplib::Request& req, httplib::Response& res) {
        using json    = nlohmann::json;
        json response = serv_generate_image(req);
        res.set_content(response.dump(), "application/json");
    });

    svr->Get("/params", [&params](const httplib::Request& req, httplib::Response& res) {
        using json = nlohmann::json;
        json response;
        json params_json               = json::object();
        params_json["prompt"]          = params.lastRequest.prompt;
        params_json["negative_prompt"] = params.lastRequest.negative_prompt;
        params_json["clip_skip"]       = params.lastRequest.clip_skip;
        params_json["cfg_scale"]       = params.lastRequest.sample_params.guidance.txt_cfg;
        params_json["guidance"]        = params.lastRequest.sample_params.guidance.distilled_guidance;
        params_json["width"]           = params.lastRequest.width;
        params_json["height"]          = params.lastRequest.height;
        params_json["sample_method"]   = sd_sample_method_name(params.lastRequest.sample_params.sample_method);
        params_json["sample_steps"]    = params.lastRequest.sample_params.sample_steps;
        params_json["seed"]            = params.lastRequest.seed;
        params_json["batch_count"]     = params.lastRequest.batch_count;
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
        // context_params["lora_model_dir"] = params.ctxParams.lora_model_dir;
        // context_params["embeddings_path"] = params.ctxParams.embeddings_path;
        // context_params["photo_maker_path"] = params.ctxParams.photo_maker_path;
        context_params["vae_decode_only"]         = params.ctxParams.vae_decode_only;
        context_params["vae_tiling"]              = params.lastRequest.tiling_params.enabled;
        context_params["n_threads"]               = params.ctxParams.n_threads;
        context_params["wtype"]                   = params.ctxParams.wtype;
        context_params["rng_type"]                = params.ctxParams.rng_type;
        context_params["schedule"]                = sd_scheduler_name(params.lastRequest.sample_params.scheduler);
        context_params["keep_clip_on_cpu"]        = params.ctxParams.keep_clip_on_cpu;
        context_params["keep_control_net_on_cpu"] = params.ctxParams.keep_control_net_on_cpu;
        context_params["keep_vae_on_cpu"]         = params.ctxParams.keep_vae_on_cpu;
        context_params["diffusion_flash_attn"]    = params.ctxParams.diffusion_flash_attn;

        response["taesd_preview"]       = params.ctxParams.taesd_preview;
        params_json["preview_method"]   = sd_preview_name(params.lastRequest.preview_method);
        params_json["preview_interval"] = params.lastRequest.preview_interval;

        response["generation_params"] = params_json;
        response["context_params"]    = context_params;
        res.set_content(response.dump(), "application/json");
    });

    svr->Get("/result", [](const httplib::Request& req, httplib::Response& res) {
        using json = nlohmann::json;
        // Parse task ID from query parameters
        try {
            std::string task_id = req.get_param_value("task_id");
            std::lock_guard<std::mutex> lock(g_results_mutex);
            if (g_task_results.find(task_id) != g_task_results.end()) {
                json result = g_task_results[task_id];
                res.set_content(result.dump(), "application/json");
                // Erase data after sending
                result["data"]          = json::array();
                g_task_results[task_id] = result;
            } else {
                res.set_content("Cannot find task " + task_id + " in queue", "text/plain");
                res.status = 404;
            }
        } catch (...) {
            sd_log(sd_log_level_t::SD_LOG_WARN, "Error when fetching result");
        }
    });

    svr->Get("/types", [](const httplib::Request& req, httplib::Response& res) {
        // Deprecated
        sd_log(SD_LOG_WARN, "/types endpoint is soon to be deprecated, use /wtypes instead");
        using json = nlohmann::json;
        json response;

        for (size_t i = 0; i < SD_TYPE_COUNT; i++) {
            std::string name = sd_type_name((sd_type_t)i);
            response.push_back(name);
        }
        res.set_content(response.dump(), "application/json");
    });

    svr->Get("/wtypes", [](const httplib::Request& req, httplib::Response& res) {
        using json = nlohmann::json;
        json response;

        for (size_t i = 0; i < SD_TYPE_COUNT; i++) {
            std::string name = sd_type_name((sd_type_t)i);
            response.push_back(name);
        }
        res.set_content(response.dump(), "application/json");
    });

    svr->Get("/rngs", [](const httplib::Request& req, httplib::Response& res) {
        using json = nlohmann::json;
        json response;
        for (size_t i = 0; i < RNG_TYPE_COUNT; i++) {
            response.push_back(sd_rng_type_name((rng_type_t)i));
        }
        res.set_content(response.dump(), "application/json");
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
        // Deprecated
        sd_log(SD_LOG_WARN, "/schedules endpoint is soon to be deprecated, use /schedulers instead");
        using json = nlohmann::json;
        json response;
        for (int s = 0; s < SCHEDULER_COUNT; s++) {
            response.push_back(sd_scheduler_name((scheduler_t)s));
        }
        res.set_content(response.dump(), "application/json");
    });

    svr->Get("/schedulers", [](const httplib::Request& req, httplib::Response& res) {
        using json = nlohmann::json;
        json response;
        for (int s = 0; s < SCHEDULER_COUNT; s++) {
            response.push_back(sd_scheduler_name((scheduler_t)s));
        }
        res.set_content(response.dump(), "application/json");
    });

    svr->Get("/predictions", [](const httplib::Request& req, httplib::Response& res) {
        using json = nlohmann::json;
        json response;
        for (int s = 0; s < PREDICTION_COUNT; s++) {
            response.push_back(sd_prediction_name((prediction_t)s));
        }
        res.set_content(response.dump(), "application/json");
    });

    svr->Get("/previews", [](const httplib::Request& req, httplib::Response& res) {
        using json = nlohmann::json;
        json response;
        for (int s = 0; s < PREVIEW_COUNT; s++) {
            response.push_back(sd_preview_name((preview_t)s));
        }
        res.set_content(response.dump(), "application/json");
    });

    svr->Get("/lora_apply_modes", [](const httplib::Request& req, httplib::Response& res) {
        using json = nlohmann::json;
        json response;
        for (int s = 0; s < LORA_APPLY_MODE_COUNT; s++) {
            response.push_back(sd_lora_apply_mode_name((lora_apply_mode_t)s));
        }
        res.set_content(response.dump(), "application/json");
    });

    svr->Get("/models", [&params, &lora_files, &embed_files](const httplib::Request& req, httplib::Response& res) {
        using json = nlohmann::json;
        json response;

        json models;
        json diffusion_models;
        json text_encoders;
        json vision_models;
        json vaes;
        json taes;
        json controlnets;
        json photomakers;

        for (size_t i = 0; i < params.models_files.size(); i++) {
            if (is_model_file(params.models_files[i])) {
                models.push_back({{"id", i}, {"name", params.models_files[i]}});
            }
        }
        for (size_t i = 0; i < params.diffusion_models_files.size(); i++) {
            if (is_model_file(params.diffusion_models_files[i])) {
                diffusion_models.push_back({{"id", i}, {"name", params.diffusion_models_files[i]}});
            }
        }
        for (size_t i = 0; i < params.clip_files.size(); i++) {
            if (is_model_file(params.clip_files[i])) {
                text_encoders.push_back({{"id", i}, {"name", params.clip_files[i]}});
            }
        }
        for (size_t i = 0; i < params.clip_vision_files.size(); i++) {
            if (is_model_file(params.clip_vision_files[i])) {
                vision_models.push_back({{"id", i}, {"name", params.clip_vision_files[i]}});
            }
        }
        for (size_t i = 0; i < params.vae_files.size(); i++) {
            if (is_model_file(params.vae_files[i])) {
                vaes.push_back({{"id", i}, {"name", params.vae_files[i]}});
            }
        }
        for (size_t i = 0; i < params.tae_files.size(); i++) {
            if (is_model_file(params.tae_files[i])) {
                taes.push_back({{"id", i}, {"name", params.tae_files[i]}});
            }
        }
        for (size_t i = 0; i < params.controlnet_files.size(); i++) {
            if (is_model_file(params.controlnet_files[i])) {
                controlnets.push_back({{"id", i}, {"name", params.controlnet_files[i]}});
            }
        }
        for (size_t i = 0; i < params.photomaker_files.size(); i++) {
            if (is_model_file(params.photomaker_files[i])) {
                photomakers.push_back({{"id", i}, {"name", params.photomaker_files[i]}});
            }
        }

        response["models"]           = models;
        response["diffusion_models"] = diffusion_models;
        response["text_encoders"]    = text_encoders;
        response["vision_models"]    = vision_models;
        response["vaes"]             = vaes;
        response["taes"]             = taes;
        response["control_nets"]     = controlnets;
        response["photo_makers"]     = photomakers;

        for (size_t i = 0; i < lora_files.size(); i++) {
            std::string lora_name = lora_files[i];
            // Remove file extension
            size_t pos = lora_name.find_last_of(".");
            if (pos != std::string::npos) {
                // Check if extension was either ".safetensors" or ".ckpt"
                std::string extension = lora_name.substr(pos + 1);
                lora_name             = lora_name.substr(0, pos);
                if (extension == "safetensors" || extension == "ckpt" || extension == "pt") {
                    response["loras"].push_back(lora_name);
                }
            }
        }

        for (size_t i = 0; i < embed_files.size(); i++) {
            std::string full_embedding_name = embed_files[i];
            // Remove file extension
            size_t pos = full_embedding_name.find_last_of(".");
            if (pos != std::string::npos) {
                std::string extension = full_embedding_name.substr(pos + 1);
                full_embedding_name   = full_embedding_name.substr(0, pos);
                if (extension == "safetensors" || extension == "ckpt" || extension == "pt") {
                    // split into subdirectory and embedding name
                    std::string subdir         = "/";
                    std::string embedding_name = full_embedding_name;
                    size_t last_slash          = full_embedding_name.find_last_of("/\\");
                    if (last_slash != std::string::npos) {
                        subdir         = full_embedding_name.substr(0, last_slash);
                        embedding_name = full_embedding_name.substr(last_slash + 1);
                    }
                    // add to dict
                    response["embeddings"][subdir].push_back(embedding_name);
                }
            }
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
        if (!params.ctxParams.high_noise_diffusion_model_path.empty()) {
            response["high_noise_diffusion_model"] = sd_basename(params.ctxParams.high_noise_diffusion_model_path);
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
        if (!params.ctxParams.llm_path.empty()) {
            response["llm"] = sd_basename(params.ctxParams.llm_path);
        }

        if (!params.ctxParams.clip_vision_path.empty()) {
            response["clip_vision"] = sd_basename(params.ctxParams.clip_vision_path);
        }
        if (!params.ctxParams.llm_vision_path.empty()) {
            response["llm_vision"] = sd_basename(params.ctxParams.llm_vision_path);
        }

        if (!params.ctxParams.vae_path.empty()) {
            response["vae"] = sd_basename(params.ctxParams.vae_path);
        }
        if (!params.ctxParams.taesd_path.empty()) {
            response["tae"] = sd_basename(params.ctxParams.taesd_path);
        }
        if (!params.ctxParams.control_net_path.empty()) {
            response["control_net"] = sd_basename(params.ctxParams.control_net_path);
        }
        if (!params.ctxParams.photo_maker_path.empty()) {
            response["photo_maker"] = sd_basename(params.ctxParams.photo_maker_path);
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

    // redirect base url to index
    svr->Get("/", [&params](const httplib::Request& req, httplib::Response& res) {
        if (params.custom_frontend_path != "") {
            res.set_redirect("/frontend-custom.html");
        } else {
            res.set_redirect("/index.html");
        }
    });

    svr->Get("/frontend-custom.html", [&params](const httplib::Request& req, httplib::Response& res) {
        try {
            std::string def_frontend_path = params.custom_frontend_path;
            std::string html              = "";
            std::ifstream file(def_frontend_path);
            if (file.is_open()) {
                std::stringstream buffer;
                buffer << file.rdbuf();
                html = buffer.str();
                file.close();
            } else {
                html = "Error: Unable to open file " + def_frontend_path;
            }
            res.set_content(html, "text/html");
        } catch (const std::exception& e) {
            res.set_content("Error loading page", "text/plain");
        }
    });

    svr->Post("/cancel_task", [](const httplib::Request& req, httplib::Response& res) {
        try {
            std::string target_id = nlohmann::json::parse(req.body)["task_id"];
            bool is_running_task  = (g_running_task_id == target_id);

            if (is_running_task) {
                if (g_is_busy) {
                    printf("-------------\n Cancelling running task %s\n-------------\n", target_id.c_str());
                    g_abort_flag = true;  // Try nice stop
                    // Spawn Watchdog
                    std::thread watchdog([target_id]() {
                        std::this_thread::sleep_for(std::chrono::seconds(10));
                        if (g_running_task_id == target_id && g_is_busy) {
                            printf("-------------\n Task %s did not stop gracefully, hard restarting\n-------------\n", target_id.c_str());
                            trigger_hard_restart();
                        }
                    });
                    watchdog.detach();
                    res.set_content("{\"status\":\"cancelling\"}", "application/json");
                } else {
                    printf("-------------\n Task %s already completed\n-------------\n", target_id.c_str());
                    res.set_content("{\"status\":\"completed\"}", "application/json");
                }
            } else {
                // Check if task is queued
                bool is_queued_task = false;

                // find task in queue
                {
                    std::lock_guard<std::mutex> lock(g_queue_mutex);
                    // iterate through queue (should not affect queue order)
                    for (auto it = 0; it < g_task_queue.size(); it++) {
                        auto task = g_task_queue.front();
                        if (task.task_id == target_id) {
                            is_queued_task = true;
                        }
                        g_task_queue.pop();       // Remove task from queue
                        g_task_queue.push(task);  // Put task back in front of queue
                    }
                }
                if (is_queued_task) {
                    printf("-------------\n Cancelling queued task %s\n-------------\n", target_id.c_str());
                    std::lock_guard<std::mutex> lock(g_cancel_mutex);
                    g_cancelled_tasks.insert(target_id);
                    res.set_content("{\"status\":\"cancelled\"}", "application/json");
                } else {
                    res.set_content("{\"error\":\"Task not found\"}", "application/json");
                    res.status = 404;
                }
            }
            return;
        } catch (const std::exception& e) {
            printf("Error cancelling task: %s\nbody: %s\n", e.what(), req.body.c_str());
            res.set_content("{\"error\":\"Invalid request\"}", "application/json");
            res.status = 400;
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

void run_worker_mode(int argc, const char* argv[]) {
    parse_args(argc, argv, g_params);
    server_log_params = (void*)&g_params;

    if (g_params.restore_state) {
        sd_log(SD_LOG_INFO, "Restoring server state from disk...\n");
        load_state_from_disk();
    }
    std::thread worker(worker_thread);
    start_server(g_params); // This will block until the server is stopped
    g_stop_worker = true;
    g_queue_cond.notify_all();
    if (worker.joinable())
        worker.join();
}

// --- Supervisor Logic (Cross-Platform Wrapper) ---

std::string win_escape_arg(const std::string& arg) {
    if (arg.empty())
        return "\"\"";

    if (arg.find_first_of(" \t\"") == std::string::npos)
        return arg;

    // Needs escaping
    std::string escaped = "\"";
    for (size_t i = 0; i < arg.length(); ++i) {
        if (arg[i] == '\"') {
            escaped += "\\\"";
        } else if (arg[i] == '\\') {
            // Handle backslashes at the end of the string or before a quote
            size_t backslash_count = 1;
            while (i + 1 < arg.length() && arg[i + 1] == '\\') {
                backslash_count++;
                i++;
            }
            // If followed by a quote, double the slashes
            if (i + 1 < arg.length() && arg[i + 1] == '\"') {
                escaped.append(backslash_count * 2, '\\');
            } else if (i + 1 == arg.length()) {
                // If at end of string, double them so the closing quote isn't escaped
                escaped.append(backslash_count * 2, '\\');
            } else {
                // Otherwise just literal backslashes
                escaped.append(backslash_count, '\\');
            }
        } else {
            escaped += arg[i];
        }
    }
    escaped += "\"";
    return escaped;
}

int run_child_process(const std::vector<std::string>& args) {
#ifdef _WIN32
    std::vector<std::string> escaped_args_storage;
    std::vector<char*> c_args;

    for (const auto& arg : args) {
        escaped_args_storage.push_back(win_escape_arg(arg));
    }

    for (const auto& s : escaped_args_storage) {
        c_args.push_back(const_cast<char*>(s.c_str()));
    }
    c_args.push_back(nullptr);


    intptr_t result = _spawnvp(_P_WAIT, args[0].c_str(), c_args.data());
    return (int)result;

#else
    std::vector<char*> c_args;
    for (const auto& arg : args)
        c_args.push_back(const_cast<char*>(arg.c_str()));
    c_args.push_back(nullptr);

    pid_t pid = fork();
    if (pid == 0) {
        execvp(c_args[0], c_args.data());
        _exit(1);
    } else if (pid > 0) {
        int status;
        waitpid(pid, &status, 0);
        return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
    }
    return -1;
#endif
}

int main(int argc, const char* argv[]) {
    printf("--- Starting ---\n");
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--worker-process-mode") == 0) {
            run_worker_mode(argc, argv);
            return 0;
        }
    }

    std::vector<std::string> args;
    for (int i = 0; i < argc; i++)
        args.push_back(argv[i]);
    args.push_back("--worker-process-mode");

    printf("--- Supervisor Started ---\n");
    bool first_run = true;
    while (true) {
        int exit_code = run_child_process(args);
        printf("Worker exited with code %d\n", exit_code);
        if (exit_code == 111) {
            if (first_run) {
                args.push_back("--restore-state");
                first_run = false;
            }
            printf(">>> Hard Restarting...\n");
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
            // restart if worker crashed (not killed)
#ifdef WIN32
        } else if (exit_code == 0xC000013A) {  // Ctrl+C
#else
        } else if (exit_code == 130 || exit_code == 143) {  // SIGINT/SIGTERM
#endif
            // exit if worker was killed by user, do nothing
            return exit_code;
        } else if (exit_code == 0) {
            // exit if worker exited normally
            return 0;
        } else {
            // probably a crash, restart
            printf(">>> Restarting worker after crash...\n");
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }

        break;
    }
    return 0;
}