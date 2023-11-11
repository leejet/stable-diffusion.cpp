#include "stable-diffusion-c.h"

/*================================================= StableDiffusion C API  =============================================*/

void* create_stable_diffusion(int n_threads, bool vae_decode_only, bool free_params_immediately,  enum RNGType rng_type) {
    return new StableDiffusion(n_threads, vae_decode_only, free_params_immediately, rng_type);
};

void destroy_stable_diffusion(void* sd) {
	auto s = (StableDiffusion*)sd;
	delete s;
};

bool load_from_file(void* sd, const char* file_path, Schedule schedule) {
	auto s = (StableDiffusion*)sd;
	return s->load_from_file(std::string(file_path), schedule);
};

uint8_t* txt2img(
    void* sd,
    const char* prompt;
    const char* negative_prompt;
    float cfg_scale,
    int width,
    int height,
    enum SampleMethod sample_method,
    int sample_steps,
    int64_t seed,
    int64_t* output_size)
{
    auto s = (StableDiffusion*)sd;
    const auto result = s->txt2img(std::string(prompt), std::string(negative_prompt),cfg_scale,width,height,second,sample_steps,seed);
    *output_size = result.size();
    return result.data();
};

uint8_t* img2img(
    void* sd,
    const uint8_t* init_img,
    const int64_t init_img_size,
    const char* prompt,
    const char* negative_prompt,
    float cfg_scale,
    int width,
    int height,
    enum SampleMethod sample_method,
    int sample_steps,
    float strength,
    int64_t seed
    int64_t* output_size)
{
    auto s = (StableDiffusion*)sd;
    std::vector<uint8_t> vec;
    std::memcpy(vec.data(), init_img, init_img_size);
    std::vector<uint8_t>  result = s->img2img(vec,std::string(prompt),std::string(negative_prompt),cfg_scale,width,height,second,sample_steps,strength,seed);
    *output_size = result.size();
    return result.data();
};

void set_stable_diffusion_log_level(enum SDLogLevel level) {
	set_sd_log_level(level)
};

const char* get_stable_diffusion_system_info() {
    std::string info = sd_get_system_info();
    size_t length = info.size() + 1;
    char* buffer = new char[length];
    memcpy(buffer, info.c_str(), length);
    return buffer;
};
