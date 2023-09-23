#include "stable-diffusion-c.h"

/*================================================= StableDiffusion C API  =============================================*/

const static std::map<std::string, enum SDLogLevel> SDLogLevelMap = {
		{"DEBUG", DEBUG},
		{"INFO", INFO},
		{"WARN", WARN},
		{"ERROR", ERROR},
};

const static std::map<std::string, enum RNGType> RNGTypeMap = {
		{"STD_DEFAULT_RNG", STD_DEFAULT_RNG},
		{"CUDA_RNG", CUDA_RNG},
};

const static std::map<std::string, enum SampleMethod> SampleMethodMap = {
		{"EULER_A",          EULER_A},
		{"EULER",            EULER},
		{"HEUN",             HEUN},
		{"DPM2",             DPM2},
		{"DPMPP2S_A",        DPMPP2S_A},
		{"DPMPP2M",          DPMPP2M},
		{"DPMPP2Mv2",        DPMPP2Mv2},
		{"N_SAMPLE_METHODS", N_SAMPLE_METHODS},
};

const static std::map<std::string, enum Schedule> ScheduleMap = {
		{"DEFAULT", DEFAULT},
		{"DISCRETE", DISCRETE},
		{"KARRAS", KARRAS},
		{"N_SCHEDULES", N_SCHEDULES},
};

// Use setter to handle purego max args limit less than 9
// see https://github.com/ebitengine/purego/pull/7
//     https://github.com/ebitengine/purego/blob/4db9e9e813d0f24f3ccc85a843d2316d2d2a70c6/func.go#L104
struct sd_txt2img_options {
	const char* prompt;
	const char* negative_prompt;
	float cfg_scale;
	int width;
	int height;
	const char* sample_method;
	int sample_steps;
	int64_t seed;
};

struct sd_img2img_options {
	const uint8_t* init_img;
	int64_t init_img_size;
	const char* prompt;
	const char* negative_prompt;
	float cfg_scale;
	int width;
	int height;
	const char* sample_method;
	int sample_steps;
	float strength;
	int64_t seed;
};

sd_txt2img_options* new_sd_txt2img_options() {
	sd_txt2img_options* opt;
	opt = new sd_txt2img_options;
	return opt;
};

sd_img2img_options* new_sd_img2img_options() {
	sd_img2img_options* opt;
	opt = new sd_img2img_options;
	return opt;
};

// Implementation for txt2img options setters
void set_txt2img_prompt(sd_txt2img_options* opt, const char* prompt) {
	opt->prompt = prompt;
}

void set_txt2img_negative_prompt(sd_txt2img_options* opt, const char* negative_prompt) {
	opt->negative_prompt = negative_prompt;
}

void set_txt2img_cfg_scale(sd_txt2img_options* opt, float cfg_scale) {
	opt->cfg_scale = cfg_scale;
}

void set_txt2img_size(sd_txt2img_options* opt, int width, int height) {
	opt->width = width;
	opt->height = height;
}

void set_txt2img_sample_method(sd_txt2img_options* opt, const char* sample_method) {
	opt->sample_method = sample_method;
}

void set_txt2img_sample_steps(sd_txt2img_options* opt, int sample_steps) {
	opt->sample_steps = sample_steps;
}

void set_txt2img_seed(sd_txt2img_options* opt, int64_t seed) {
	opt->seed = seed;
}

// Implementation for img2img options setters
void set_img2img_init_img(sd_img2img_options* opt, const uint8_t* init_img, int64_t init_img_size) {
	// Assuming init_img is a pointer to image data
	// Depending on the actual image data representation, you may need to handle it accordingly
	// For simplicity, let's assume init_img is a pointer to a memory block containing image data
	opt->init_img = init_img;
	opt->init_img_size = init_img_size;
}

void set_img2img_prompt(sd_img2img_options* opt, const char* prompt) {
	opt->prompt = prompt;
}

void set_img2img_negative_prompt(sd_img2img_options* opt, const char* negative_prompt) {
	opt->negative_prompt = negative_prompt;
}

void set_img2img_cfg_scale(sd_img2img_options* opt, float cfg_scale) {
	// Assuming cfg_scale is a floating point number in string format
	opt->cfg_scale = cfg_scale;
}

void set_img2img_size(sd_img2img_options* opt, int width, int height) {
	opt->width = width;
	opt->height = height;
}

void set_img2img_sample_method(sd_img2img_options* opt, const char* sample_method) {
	opt->sample_method = sample_method;
}

void set_img2img_sample_steps(sd_img2img_options* opt, int sample_steps) {
	opt->sample_steps = sample_steps;
}

void set_img2img_strength(sd_img2img_options* opt, float strength) {
	// Assuming strength is a floating point number
	opt->strength = strength;
}

void set_img2img_seed(sd_img2img_options* opt, int64_t seed) {
	opt->seed = seed;
}

void* create_stable_diffusion(int n_threads, bool vae_decode_only, bool free_params_immediately, const char* rng_type) {
	auto s = std::string(rng_type);
	auto it = RNGTypeMap.find(s);
	if (it != RNGTypeMap.end()) {
		return new StableDiffusion(n_threads, vae_decode_only, free_params_immediately, it->second);
	}
	return NULL;
};

void destroy_stable_diffusion(void* sd) {
	auto* s = (StableDiffusion*)sd;
	delete s;
};

bool load_from_file(void* sd, const char* file_path, const char* schedule) {
	auto* s = (StableDiffusion*)sd;
	auto sc = std::string(schedule);
	auto it = ScheduleMap.find(sc);
	if (it != ScheduleMap.end()) {
		return s->load_from_file(std::string(file_path), it->second);
	}
	return false;
};

uint8_t* txt2img(void* sd, sd_txt2img_options* opt, int64_t* output_size) {
	auto sm = std::string(opt->sample_method);
	auto it = SampleMethodMap.find(sm);
	if (it != SampleMethodMap.end()) {
		auto* s = (StableDiffusion*)sd;
		std::vector<uint8_t> result = s->txt2img(
			/* const std::string &prompt */             std::string(opt->prompt),
			/* const std::string &negative_prompt */    std::string(opt->negative_prompt),
			/* float cfg_scale */                       opt->cfg_scale,
			/* int width */                             opt->width,
			/* int height */                            opt->height,
			/* SampleMethod sample_method */            it->second,
			/* int sample_steps */                      opt->sample_steps,
			/* int64_t seed */                          opt->seed
		);
		*output_size = result.size();
		delete opt;
		return result.data();
	}
	return NULL;
};

uint8_t* img2img(void* sd, sd_img2img_options* opt, int64_t* output_size) {
	auto sm = std::string(opt->sample_method);
	auto it = SampleMethodMap.find(sm);
	if (it != SampleMethodMap.end()) {
		auto* s = (StableDiffusion*)sd;
		std::vector<uint8_t> vec;
		std::memcpy(vec.data(), opt->init_img, opt->init_img_size);
		std::vector<uint8_t>  result = s->img2img(
			/* const std::vector<uint8_t>& init_img */  vec,
			/* const std::string &prompt */             std::string(opt->prompt),
			/* const std::string &negative_prompt */    std::string(opt->negative_prompt),
			/* float cfg_scale */                       opt->cfg_scale,
			/* int width */                             opt->width,
			/* int height */                            opt->height,
			/* SampleMethod sample_method */            it->second,
			/* int sample_steps */                      opt->sample_steps,
			/* float strength */                        opt->strength,
			/* int64_t seed */                          opt->seed
		);
		*output_size = result.size();
		delete opt;
		return result.data();
	}
	return NULL;
};

void set_stable_diffusion_log_level(const char* level) {
	auto ll = std::string(level);
	auto it = SDLogLevelMap.find(ll);
	if (it != SDLogLevelMap.end()) {
		set_sd_log_level(it->second);
	}
};

const char* get_stable_diffusion_system_info() {
	static std::string info = sd_get_system_info();
	return info.c_str();
};
