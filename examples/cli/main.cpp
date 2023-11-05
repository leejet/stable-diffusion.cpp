#include <stdio.h>
#include <ctime>
#include <fstream>
#include <iostream>
#include <random>
#include <unordered_set>

#include "stable-diffusion.h"
#include "common.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include "stb_image_write.h"


int main(int argc, const char* argv[]) {
    sd_params params;
    parse_args(argc, argv, params);

    if (params.verbose) {
        print_params(params);
        printf("%s", sd_get_system_info().c_str());
        set_sd_log_level(sd_log_level::DEBUG);
    }

    bool vae_decode_only = true;
    std::vector<uint8_t> input_image_buffer;
    if (params.mode == IMG2IMG) {
        vae_decode_only = false;

        int c = 0;
        unsigned char* img_data = stbi_load(params.input_path.c_str(), &params.width, &params.height, &c, 3);
        if (img_data == NULL) {
            fprintf(stderr, "load image from '%s' failed\n", params.input_path.c_str());
            return 1;
        }
        if (c != 3) {
            fprintf(stderr, "input image must be a 3 channels RGB image, but got %d channels\n", c);
            free(img_data);
            return 1;
        }
        if (params.width <= 0 || params.width % 64 != 0) {
            fprintf(stderr, "error: the width of image must be a multiple of 64\n");
            free(img_data);
            return 1;
        }
        if (params.height <= 0 || params.height % 64 != 0) {
            fprintf(stderr, "error: the height of image must be a multiple of 64\n");
            free(img_data);
            return 1;
        }
        input_image_buffer.assign(img_data, img_data + (params.width * params.height * c));
    }

    StableDiffusion sd(params.n_threads, vae_decode_only, true, params.lora_model_dir, params.rng_type);
    if (!sd.load_from_file(params.model_path, params.schedule)) {
        return 1;
    }

    std::vector<uint8_t> img;
    if (params.mode == TXT2IMG) {
        img = sd.txt2img(params.prompt,
                         params.negative_prompt,
                         params.cfg_scale,
                         params.width,
                         params.height,
                         params.sample_method,
                         params.sample_steps,
                         params.seed);
    } else {
        img = sd.img2img(input_image_buffer,
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

    if (img.size() == 0) {
        fprintf(stderr, "generate failed\n");
        return 1;
    }

    stbi_write_png(params.output_path.c_str(), params.width, params.height, 3, img.data(), 0, get_image_params(params));
    printf("save result image to '%s'\n", params.output_path.c_str());

    return 0;
}
