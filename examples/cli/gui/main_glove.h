
#define GLOVE_ENABLE_JSON
#include "glove.h"

glvm_SlvEnum_named(ProcessingMode, text_to_image, "txt2img", image_to_image, "img2img", convert_image, "convert");
glvm_SlvEnum_named(WeightType, weight_file_type, "", f32, "f32", f16, "f16", q4_0, "q4_0", q4_1, "q4_1", q5_0, "q5_0", q5_1, "q5_1", q8_0, "q8_0", q2_k, "q2_k", q3_k, "q3_k", q4_k, "q4_k");
glvm_SlvEnum_named(SamplingMethod, euler, "euler", euler_a, "euler_a", heun, "heun", dpm2, "dpm2", dpmpp2s_a, "dpm++2s_a", dpmpp2m, "dpm++2m", dpmpp2mv2, "dpm++2mv2", ipndm, "ipndm", ipndm_v, "ipndm_v", lcm, "lcm");
glvm_SlvEnum(Rng, std_default, cuda);
glvm_SlvEnum(Schedule, discrete, karras, exponential, ays, gits);

glvm_parametrization(GlvSDParamsPhotomaker, "Photomaker params",
                        stacked_id_embd_dir, SlvDirectory, "--stacked-id-embd-dir", "path to PHOTOMAKER stacked id embeddings.", SlvDirectory(),
                        input_id_images_dir, SlvDirectory, "--input-id-images-dir", "path to PHOTOMAKER input id images dir.", SlvDirectory(),
                        normalize_input, bool, "--normalize-input", "normalize PHOTOMAKER input id images", false);

glvm_parametrization(GlvSDParamsAdvanced, "Advanced params",
                        threads, int, "--threads", "number of threads to use during computation (default: -1). \nIf threads <= 0, then threads will be set to the number of CPU physical cores", -1,
                        taesd, SlvFile, "--taesd", "path to taesd. Using Tiny AutoEncoder for fast decoding (low quality)", SlvFile(SlvFile::IO::Read),
                        control_net, SlvFile, "--control-net", "path to control net model", SlvFile(SlvFile::IO::Read),
                        embd_dir, SlvDirectory, "--embd-dir", "path to embeddings.", SlvDirectory(),
                        upscale_model, SlvFile, "--upscale-model", "path to esrgan model. Upscale images after generate, just RealESRGAN_x4plus_anime_6B supported by now.", SlvFile(SlvFile::IO::Read),
                        upscale_repeats, unsigned int, "--upscale-repeats", "Run the ESRGAN upscaler this many times (default 1)", 1,
                        type, WeightType, "--type", "weight type (f32, f16, q4_0, q4_1, q5_0, q5_1, q8_0, q2_k, q3_k, q4_k) \nIf not specified, the default is the type of the weight file.", WeightType::weight_file_type,
                        schedule, Schedule, "--schedule", "Denoiser sigma schedule (default: discrete)", Schedule::discrete,
                        clip_skip, int, "--clip-skip", "ignore last layers of CLIP network; 1 ignores none, 2 ignores one layer (default: -1) \n<= 0 represents unspecified, will be 1 for SD1.x, 2 for SD2.x", -1,
                        vae_tiling, bool, "--vae-tiling", "process vae in tiles to reduce memory usage", false,
                        vae_on_cpu, bool, "--vae-on-cpu", "keep vae in cpu (for low vram)", false,
                        clip_on_cpu, bool, "--clip-on-cpu", "keep clip in cpu (for low vram)", false,
                        control_net_cpu, bool, "--control-net-cpu", "keep controlnet in cpu (for low vram)", false,
                        canny, bool, "--canny", "apply canny preprocessor (edge detection)", false,
                        color, bool, "--color", "colors the logging tags according to level", false,
                        verbose, bool, "--verbose", "print extra info", false);

glvm_parametrization(GlvSDParams, "SD params",
                        mode, ProcessingMode, "--mode", "run mode (txt2img or img2img or convert, default: txt2img)", ProcessingMode::text_to_image,
                        model, SlvFile, "--model", "path to full model", SlvFile("./", SlvFileExtensions({".safetensors", ".ckpt"}), SlvFile::IO::Read),
                        diffusion_model, SlvFile, "--diffusion-model", "path to the standalone diffusion model", SlvFile(SlvFileExtensions({".gguf"}), SlvFile::IO::Read),
                        clip_l, SlvFile, "--clip_l", "path to the clip-l text encoder", SlvFile(SlvFile::IO::Read),
                        t5xxl, SlvFile, "--t5xxl", "path to the the t5xxl text encoder", SlvFile(SlvFile::IO::Read),
                        vae, SlvFile, "--vae", "path to vae", SlvFile("", SlvFileExtensions({".safetensors", ".sft"}), SlvFile::IO::Read),
                        photomaker_params, GlvSDParamsPhotomaker, "Photomaker", "", GlvSDParamsPhotomaker(), 
                        lora_model_dir, SlvDirectory, "--lora-model-dir", "lora model directory", SlvDirectory(),
                        init_img, SlvFile, "--init-img", "path to the input image, required by img2img", SlvFile(SlvFile::IO::Read),
                        control_img, SlvFile, "--control-image", "path to image condition, control net", SlvFile(SlvFile::IO::Read),
                        output, SlvFile, "--output", "path to write result image to (default: ./output.png)", SlvFile("./output.png", SlvFileExtensions({".png"}), SlvFile::IO::Write),
                        prompt, std::string, "--prompt", "the prompt to render", "",
                        negative_prompt, std::string, "--negative-prompt", "the negative prompt (default: '')", "",
                        cfg_scale, float, "--cfg-scale", "unconditional guidance scale: (default: 7.0)", 7.0f,
                        strength, float, "--strength", "strength for noising/unnoising (default: 0.75)", 0.75f,
                        style_ratio, SlvProportion, "--style-ratio", "strength for keeping input identity (default: 20%)", 0.2f,
                        control_strength, SlvProportion, "--control-strength", "strength to apply Control Net (default: 0.9) \n1.0 corresponds to full destruction of information in init", 0.9f,
                        height, unsigned int, "--height", "image height, in pixel space (default: 512)", 512,
                        width, unsigned int, "--width", "image width, in pixel space (default: 512)", 512,
                        sampling_method, SamplingMethod, "--sampling-method", "{euler, euler_a, heun, dpm2, dpm++2s_a, dpm++2m, dpm++2mv2, ipndm, ipndm_v, lcm} \nsampling method (default: 'euler_a')", SamplingMethod::euler_a,
                        steps, unsigned int, "--steps", "number of sample steps (default: 20)", 20,
                        rng, Rng, "--rng", "RNG (default: cuda)", Rng::cuda,
                        seed, int, "--seed", "RNG seed (default: 42, use random seed for < 0)", 42,
                        batch_count, unsigned int, "--batch-count", "number of images to generate.", 1,
                        advanced_params, GlvSDParamsAdvanced, "Advanced", "", GlvSDParamsAdvanced())

GLOVE_CLI_PARAMETRIZATION_OUTPUT_DIRECTORY(GlvSDParams, "--output")
