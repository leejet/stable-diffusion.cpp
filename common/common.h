#pragma once

#include <string>
#include "stable-diffusion.h"

enum sd_mode {
    TXT2IMG,
    IMG2IMG,
    MODE_COUNT
};

struct SDParams {
    int n_threads = -1;
    sd_mode mode  = TXT2IMG;

    std::string model_path;
    std::string lora_model_dir;
    std::string output_path = "output.png";
    std::string input_path;

    std::string prompt;
    std::string negative_prompt;
    float cfg_scale = 7.0f;
    int width       = 512;
    int height      = 512;
    int batch_count = 1;

    SampleMethod sample_method = EULER_A;
    Schedule schedule          = DEFAULT;
    int sample_steps           = 20;
    float strength             = 0.75f;
    RNGType rng_type           = CUDA_RNG;
    int64_t seed               = 42;
    bool verbose               = false;
};

void print_params(SDParams params);

void print_usage(int argc, const char* argv[]);

void parse_args(int argc, const char** argv, SDParams& params);

std::string get_image_params(SDParams params, int seed);
