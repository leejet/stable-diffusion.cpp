#include <stdio.h>
#include <ctime>
#include <random>
#include "common.h"
#include "stable-diffusion.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include "stb_image_write.h"

int main(int argc, const char* argv[]) {
    SDParams params;
    parse_args(argc, argv, params);

    if (params.verbose) {
        print_params(params);
        printf("%s", sd_get_system_info().c_str());
        set_sd_log_level(SDLogLevel::DEBUG);
    }

    bool vae_decode_only        = true;
    uint8_t* input_image_buffer = NULL;
    if (params.mode == IMG2IMG) {
        vae_decode_only = false;

        int c              = 0;
        input_image_buffer = stbi_load(params.input_path.c_str(), &params.width, &params.height, &c, 3);
        if (input_image_buffer == NULL) {
            fprintf(stderr, "load image from '%s' failed\n", params.input_path.c_str());
            return 1;
        }
        if (c != 3) {
            fprintf(stderr, "input image must be a 3 channels RGB image, but got %d channels\n", c);
            free(input_image_buffer);
            return 1;
        }
        if (params.width <= 0 || params.width % 64 != 0) {
            fprintf(stderr, "error: the width of image must be a multiple of 64\n");
            free(input_image_buffer);
            return 1;
        }
        if (params.height <= 0 || params.height % 64 != 0) {
            fprintf(stderr, "error: the height of image must be a multiple of 64\n");
            free(input_image_buffer);
            return 1;
        }
    }

    StableDiffusion sd(params.n_threads, vae_decode_only, true, params.lora_model_dir, params.rng_type);
    if (!sd.load_from_file(params.model_path, params.schedule)) {
        return 1;
    }

    std::vector<uint8_t*> results;
    if (params.mode == TXT2IMG) {
        results = sd.txt2img(params.prompt,
                             params.negative_prompt,
                             params.cfg_scale,
                             params.width,
                             params.height,
                             params.sample_method,
                             params.sample_steps,
                             params.seed,
                             params.batch_count);
    } else {
        results = sd.img2img(input_image_buffer,
                             params.prompt,
                             params.negative_prompt,
                             params.cfg_scale,
                             params.width,
                             params.height,
                             params.sample_method,
                             params.sample_steps,
                             params.strength,
                             params.seed);
    }

    if (results.size() == 0 || results.size() != params.batch_count) {
        fprintf(stderr, "generate failed\n");
        return 1;
    }

    size_t last            = params.output_path.find_last_of(".");
    std::string dummy_name = last != std::string::npos ? params.output_path.substr(0, last) : params.output_path;
    for (int i = 0; i < params.batch_count; i++) {
        std::string final_image_path = i > 0 ? dummy_name + "_" + std::to_string(i + 1) + ".png" : dummy_name + ".png";
        stbi_write_png(final_image_path.c_str(), params.width, params.height, 3, results[i], 0, get_image_params(params, params.seed + i).c_str());
        printf("save result image to '%s'\n", final_image_path.c_str());
    }

    return 0;
}
