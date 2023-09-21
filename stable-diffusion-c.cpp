#include "stable-diffusion-c.h"
/*================================================= StableDiffusion C API  =============================================*/

// Use setter to handle purego max args limit less than 9
// see https://github.com/ebitengine/purego/pull/7
struct sd_txt2img_options {
    const char *prompt;
    const char *negative_prompt;
    float cfg_scale;
    int width;
    int height;
    SampleMethod sample_method;
    int sample_steps;
    int64_t seed;
};

struct sd_img2img_options {
    const uint8_t *init_img;
    int64_t init_img_size;
    const char *prompt;
    const char *negative_prompt;
    float cfg_scale;
    int width;
    int height;
    SampleMethod sample_method;
    int sample_steps;
    float strength;
    int64_t seed;
};

sd_txt2img_options *new_sd_txt2img_options() {
    sd_txt2img_options *opt;
    opt = new sd_txt2img_options;
    return opt;
};

sd_img2img_options *new_sd_img2img_options() {
    sd_img2img_options *opt;
    opt = new sd_img2img_options;
    return opt;
};

// Implementation for txt2img options setters
void set_txt2img_prompt(sd_txt2img_options *opt, const char *prompt) {
    opt->prompt = prompt;
}

void set_txt2img_negative_prompt(sd_txt2img_options *opt, const char *negative_prompt) {
    opt->negative_prompt = negative_prompt;
}

void set_txt2img_cfg_scale(sd_txt2img_options *opt, const char *cfg_scale) {
    // Assuming cfg_scale is a floating point number in string format
    opt->cfg_scale = strtof(cfg_scale, nullptr);
}

void set_txt2img_size(sd_txt2img_options *opt, int width, int height) {
    opt->width = width;
    opt->height = height;
}

void set_txt2img_sample_method(sd_txt2img_options *opt, SampleMethod sample_method) {
    opt->sample_method = sample_method;
}

void set_txt2img_sample_steps(sd_txt2img_options *opt, int sample_steps) {
    opt->sample_steps = sample_steps;
}

void set_txt2img_seed(sd_txt2img_options *opt, int64_t seed) {
    opt->seed = seed;
}

// Implementation for img2img options setters
void set_img2img_init_img(sd_img2img_options *opt, const uint8_t *init_img, int64_t init_img_size) {
    // Assuming init_img is a pointer to image data
    // Depending on the actual image data representation, you may need to handle it accordingly
    // For simplicity, let's assume init_img is a pointer to a memory block containing image data
    opt->init_img = init_img;
    opt->init_img_size = init_img_size;
}

void set_img2img_prompt(sd_img2img_options *opt, const char *prompt) {
    opt->prompt = prompt;
}

void set_img2img_negative_prompt(sd_img2img_options *opt, const char *negative_prompt) {
    opt->negative_prompt = negative_prompt;
}

void set_img2img_cfg_scale(sd_img2img_options *opt, const char *cfg_scale) {
    // Assuming cfg_scale is a floating point number in string format
    opt->cfg_scale = strtof(cfg_scale, nullptr);
}

void set_img2img_size(sd_img2img_options *opt, int width, int height) {
    opt->width = width;
    opt->height = height;
}

void set_img2img_sample_method(sd_img2img_options *opt, SampleMethod sample_method) {
    opt->sample_method = sample_method;
}

void set_img2img_sample_steps(sd_img2img_options *opt, int sample_steps) {
    opt->sample_steps = sample_steps;
}

void set_img2img_strength(sd_img2img_options *opt, float strength) {
    // Assuming strength is a floating point number
    opt->strength = strength;
}

void set_img2img_seed(sd_img2img_options *opt, int64_t seed) {
    opt->seed = seed;
}

void *create_stable_diffusion(int n_threads, bool vae_decode_only, bool free_params_immediately, RNGType rng_type) {
    return new StableDiffusion(n_threads, vae_decode_only, free_params_immediately, rng_type);
};

void destroy_stable_diffusion(void *sd) {
    auto *s = (StableDiffusion *) sd;
    delete s;
};

int load_from_file(void *sd, const char *file_path, Schedule schedule) {
    auto *s = (StableDiffusion *) sd;
    s->load_from_file(std::string(file_path), schedule);
};

uint8_t *txt2img(void *sd, sd_txt2img_options *opt) {
    auto *s = (StableDiffusion *) sd;
    std::vector<uint8_t> result;
    result = s->txt2img(
            /* const std::string &prompt */             std::string(opt->prompt),
            /* const std::string &negative_prompt */    std::string(opt->negative_prompt),
            /* float cfg_scale */                       opt->cfg_scale,
            /* int width */                             opt->width,
            /* int height */                            opt->height,
            /* SampleMethod sample_method */            opt->sample_method,
            /* int sample_steps */                      opt->sample_steps,
            /* int64_t seed */                          opt->seed
    );
    static auto *data = (uint8_t *) result.data();
    return data;
};

uint8_t *img2img(void *sd, sd_img2img_options *opt) {
    auto *s = (StableDiffusion *) sd;
    std::vector<uint8_t> result;
    std::vector<uint8_t> vec;
    std::memcpy(vec.data(), opt->init_img, opt->init_img_size);
    result = s->img2img(
            /* const std::vector<uint8_t>& init_img */ vec,
            /* const std::string &prompt */             std::string(opt->prompt),
            /* const std::string &negative_prompt */    std::string(opt->negative_prompt),
            /* float cfg_scale */                       opt->cfg_scale,
            /* int width */                             opt->width,
            /* int height */                            opt->height,
            /* SampleMethod sample_method */            opt->sample_method,
            /* int sample_steps */                      opt->sample_steps,
            /* float strength */                        opt->strength,
            /* int64_t seed */                          opt->seed
    );
    static auto *data = (uint8_t *) result.data();
    return data;
};

void set_stable_diffusion_log_level(SDLogLevel level) {
    set_sd_log_level(level);
};

const char *get_stable_diffusion_system_info() {
    static std::string info = sd_get_system_info();
    return info.c_str();
};
