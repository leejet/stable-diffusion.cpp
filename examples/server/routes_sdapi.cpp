#include "routes.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <regex>
#include <string_view>
#include <unordered_map>

#include "common/common.h"
#include "common/media_io.h"
#include "common/resource_owners.hpp"

namespace fs = std::filesystem;

static std::string extract_and_remove_sd_cpp_extra_args(std::string& text) {
    std::regex re("<sd_cpp_extra_args>(.*?)</sd_cpp_extra_args>");
    std::smatch match;

    std::string extracted;
    if (std::regex_search(text, match, re)) {
        extracted = match[1].str();
        text      = std::regex_replace(text, re, "");
    }
    return extracted;
}

static fs::path resolve_display_model_path(const ServerRuntime& runtime) {
    const auto& ctx = *runtime.ctx_params;
    if (!ctx.model_path.empty()) {
        return fs::path(ctx.model_path);
    }
    if (!ctx.diffusion_model_path.empty()) {
        return fs::path(ctx.diffusion_model_path);
    }
    return {};
}

static std::string lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

static enum sample_method_t get_sdapi_sample_method(std::string name) {
    enum sample_method_t result = str_to_sample_method(name.c_str());
    if (result != SAMPLE_METHOD_COUNT) {
        return result;
    }

    name = lower_ascii(name);
    static const std::unordered_map<std::string_view, sample_method_t> hardcoded{
        {"euler a", EULER_A_SAMPLE_METHOD},
        {"k_euler_a", EULER_A_SAMPLE_METHOD},
        {"euler", EULER_SAMPLE_METHOD},
        {"k_euler", EULER_SAMPLE_METHOD},
        {"heun", HEUN_SAMPLE_METHOD},
        {"k_heun", HEUN_SAMPLE_METHOD},
        {"dpm2", DPM2_SAMPLE_METHOD},
        {"k_dpm_2", DPM2_SAMPLE_METHOD},
        {"lcm", LCM_SAMPLE_METHOD},
        {"ddim", DDIM_TRAILING_SAMPLE_METHOD},
        {"dpm++ 2m", DPMPP2M_SAMPLE_METHOD},
        {"k_dpmpp_2m", DPMPP2M_SAMPLE_METHOD},
        {"res multistep", RES_MULTISTEP_SAMPLE_METHOD},
        {"k_res_multistep", RES_MULTISTEP_SAMPLE_METHOD},
        {"res 2s", RES_2S_SAMPLE_METHOD},
        {"k_res_2s", RES_2S_SAMPLE_METHOD},
        {"euler_cfg_pp", EULER_CFG_PP_SAMPLE_METHOD},
        {"k_euler_cfg_pp", EULER_CFG_PP_SAMPLE_METHOD},
        {"euler_a_cfg_pp", EULER_CFG_PP_SAMPLE_METHOD},
        {"k_euler_a_cfg_pp", EULER_CFG_PP_SAMPLE_METHOD},
    };
    auto it = hardcoded.find(name);
    return it != hardcoded.end() ? it->second : SAMPLE_METHOD_COUNT;
}

static void assign_solid_mask(SDImageOwner& mask_owner, int width, int height) {
    const size_t pixel_count = static_cast<size_t>(width) * static_cast<size_t>(height);
    uint8_t* raw_mask        = static_cast<uint8_t*>(malloc(pixel_count));
    if (raw_mask == nullptr) {
        mask_owner.reset({0, 0, 1, nullptr});
        return;
    }
    std::memset(raw_mask, 255, pixel_count);
    mask_owner.reset({(uint32_t)width, (uint32_t)height, 1, raw_mask});
}

static bool build_sdapi_img_gen_request(const json& j,
                                        ServerRuntime& runtime,
                                        bool img2img,
                                        ImgGenJobRequest& request,
                                        std::string& error_message) {
    std::string prompt          = j.value("prompt", "");
    std::string negative_prompt = j.value("negative_prompt", "");
    int width                   = j.value("width", 512);
    int height                  = j.value("height", 512);
    int steps                   = j.value("steps", runtime.default_gen_params->sample_params.sample_steps);
    float cfg_scale             = j.value("cfg_scale", runtime.default_gen_params->sample_params.guidance.txt_cfg);
    int64_t seed                = j.value("seed", -1);
    int batch_size              = j.value("batch_size", 1);
    int clip_skip               = j.value("clip_skip", -1);
    std::string sampler_name    = j.value("sampler_name", "");
    std::string scheduler_name  = j.value("scheduler", "");

    if (width <= 0 || height <= 0) {
        error_message = "width and height must be positive";
        return false;
    }

    if (prompt.empty()) {
        error_message = "prompt required";
        return false;
    }

    request.gen_params = *runtime.default_gen_params;

    request.gen_params.prompt                         = prompt;
    request.gen_params.negative_prompt                = negative_prompt;
    request.gen_params.seed                           = seed;
    request.gen_params.sample_params.sample_steps     = steps;
    request.gen_params.batch_count                    = batch_size;
    request.gen_params.sample_params.guidance.txt_cfg = cfg_scale;
    request.gen_params.width                          = j.value("width", -1);
    request.gen_params.height                         = j.value("height", -1);

    if (!img2img && j.value("enable_hr", false)) {
        request.gen_params.hires_enabled = true;
        request.gen_params.hires_scale   = j.value("hr_scale", request.gen_params.hires_scale);
        request.gen_params.hires_width   = j.value("hr_resize_x", request.gen_params.hires_width);
        request.gen_params.hires_height  = j.value("hr_resize_y", request.gen_params.hires_height);
        request.gen_params.hires_steps   = j.value("hr_steps", request.gen_params.hires_steps);
        request.gen_params.hires_denoising_strength =
            j.value("denoising_strength", request.gen_params.hires_denoising_strength);

        request.gen_params.hires_upscaler = j.value("hr_upscaler", request.gen_params.hires_upscaler);
    }

    std::string sd_cpp_extra_args_str = extract_and_remove_sd_cpp_extra_args(request.gen_params.prompt);
    if (!sd_cpp_extra_args_str.empty() && !request.gen_params.from_json_str(sd_cpp_extra_args_str)) {
        error_message = "invalid sd_cpp_extra_args";
        return false;
    }

    if (clip_skip > 0) {
        request.gen_params.clip_skip = clip_skip;
    }

    enum sample_method_t sample_method = get_sdapi_sample_method(sampler_name);
    if (sample_method != SAMPLE_METHOD_COUNT) {
        request.gen_params.sample_params.sample_method = sample_method;
    }

    enum scheduler_t scheduler = str_to_scheduler(scheduler_name.c_str());
    if (scheduler != SCHEDULER_COUNT) {
        request.gen_params.sample_params.scheduler = scheduler;
    }

    if (j.contains("lora") && j["lora"].is_array()) {
        request.gen_params.lora_map.clear();
        request.gen_params.high_noise_lora_map.clear();

        for (const auto& item : j["lora"]) {
            if (!item.is_object()) {
                continue;
            }

            std::string path   = item.value("path", "");
            float multiplier   = item.value("multiplier", 1.0f);
            bool is_high_noise = item.value("is_high_noise", false);

            if (path.empty()) {
                error_message = "lora.path required";
                return false;
            }

            std::string fullpath = get_lora_full_path(runtime, path);
            if (fullpath.empty()) {
                error_message = "invalid lora path: " + path;
                return false;
            }

            if (is_high_noise) {
                request.gen_params.high_noise_lora_map[fullpath] += multiplier;
            } else {
                request.gen_params.lora_map[fullpath] += multiplier;
            }
        }
    }

    if (img2img) {
        const int expected_width  = request.gen_params.width_and_height_are_set() ? request.gen_params.width : 0;
        const int expected_height = request.gen_params.width_and_height_are_set() ? request.gen_params.height : 0;

        if (j.contains("init_images") && j["init_images"].is_array() && !j["init_images"].empty()) {
            if (decode_base64_image(j["init_images"][0].get<std::string>(),
                                    3,
                                    expected_width,
                                    expected_height,
                                    request.gen_params.init_image)) {
                const sd_image_t& image = request.gen_params.init_image.get();
                request.gen_params.set_width_and_height_if_unset(image.width, image.height);
            }
        }

        if (j.contains("mask") && j["mask"].is_string()) {
            if (decode_base64_image(j["mask"].get<std::string>(),
                                    1,
                                    expected_width,
                                    expected_height,
                                    request.gen_params.mask_image)) {
                const sd_image_t& image = request.gen_params.mask_image.get();
                request.gen_params.set_width_and_height_if_unset(image.width, image.height);
            }
            sd_image_t& mask_image      = request.gen_params.mask_image.get();
            bool inpainting_mask_invert = j.value("inpainting_mask_invert", 0) != 0;
            if (inpainting_mask_invert && mask_image.data != nullptr) {
                for (uint32_t i = 0; i < mask_image.width * mask_image.height; ++i) {
                    mask_image.data[i] = 255 - mask_image.data[i];
                }
            }
        } else {
            const int resolved_width  = request.gen_params.get_resolved_width();
            const int resolved_height = request.gen_params.get_resolved_height();
            assign_solid_mask(request.gen_params.mask_image, resolved_width, resolved_height);
        }

        float denoising_strength = j.value("denoising_strength", -1.f);
        if (denoising_strength >= 0.f) {
            request.gen_params.strength = std::min(denoising_strength, 1.0f);
        }
    }

    if (j.contains("extra_images") && j["extra_images"].is_array()) {
        for (const auto& extra_image : j["extra_images"]) {
            if (!extra_image.is_string()) {
                continue;
            }
            SDImageOwner image_owner;
            if (decode_base64_image(extra_image.get<std::string>(),
                                    3,
                                    request.gen_params.width_and_height_are_set() ? request.gen_params.width : 0,
                                    request.gen_params.width_and_height_are_set() ? request.gen_params.height : 0,
                                    image_owner)) {
                const sd_image_t& image = image_owner.get();
                request.gen_params.set_width_and_height_if_unset(image.width, image.height);
                request.gen_params.ref_images.push_back(std::move(image_owner));
            }
        }
    }

    // Intentionally disable prompt-embedded LoRA tag parsing for server APIs.
    if (!request.gen_params.resolve_and_validate(IMG_GEN, "", runtime.ctx_params->hires_upscalers_dir, true)) {
        error_message = "invalid params";
        return false;
    }

    return true;
}

void register_sdapi_endpoints(httplib::Server& svr, ServerRuntime& rt) {
    ServerRuntime* runtime = &rt;

    auto sdapi_any2img = [runtime](const httplib::Request& req, httplib::Response& res, bool img2img) {
        try {
            if (req.body.empty()) {
                res.status = 400;
                res.set_content(R"({"error":"empty body"})", "application/json");
                return;
            }
            if (!runtime_supports_generation_mode(*runtime, IMG_GEN)) {
                res.status = 400;
                res.set_content(json({{"error", unsupported_generation_mode_error(IMG_GEN)}}).dump(), "application/json");
                return;
            }

            json j = json::parse(req.body);
            ImgGenJobRequest request;
            std::string error_message;
            if (!build_sdapi_img_gen_request(j, *runtime, img2img, request, error_message)) {
                res.status = 400;
                res.set_content(json({{"error", error_message}}).dump(), "application/json");
                return;
            }

            LOG_DEBUG("%s\n", request.gen_params.to_string().c_str());

            sd_img_gen_params_t img_gen_params = request.to_sd_img_gen_params_t();
            SDImageVec results;
            int num_results = 0;

            {
                std::lock_guard<std::mutex> lock(*runtime->sd_ctx_mutex);
                sd_image_t* raw_results = generate_image(runtime->sd_ctx, &img_gen_params);
                num_results             = request.gen_params.batch_count;
                results.adopt(raw_results, num_results);
            }

            if (results.empty()) {
                res.status = 500;
                res.set_content(R"({"error":"generate_image returned no results"})", "application/json");
                return;
            }

            json out;
            out["images"]     = json::array();
            out["parameters"] = j;
            out["info"]       = "";

            for (int i = 0; i < num_results; ++i) {
                if (results[i].data == nullptr) {
                    continue;
                }

                std::string params = request.gen_params.embed_image_metadata
                                         ? get_image_params(*runtime->ctx_params,
                                                            request.gen_params,
                                                            request.gen_params.seed + i)
                                         : "";
                auto image_bytes   = encode_image_to_vector(EncodedImageFormat::PNG,
                                                            results[i].data,
                                                            results[i].width,
                                                            results[i].height,
                                                            results[i].channel,
                                                            params);

                if (image_bytes.empty()) {
                    LOG_ERROR("write image to mem failed");
                    continue;
                }

                out["images"].push_back(base64_encode(image_bytes));
            }

            res.set_content(out.dump(), "application/json");
            res.status = 200;

        } catch (const std::exception& e) {
            res.status = 500;
            json err;
            err["error"]   = "server_error";
            err["message"] = e.what();
            res.set_content(err.dump(), "application/json");
        }
    };

    svr.Post("/sdapi/v1/txt2img", [sdapi_any2img](const httplib::Request& req, httplib::Response& res) {
        sdapi_any2img(req, res, false);
    });

    svr.Post("/sdapi/v1/img2img", [sdapi_any2img](const httplib::Request& req, httplib::Response& res) {
        sdapi_any2img(req, res, true);
    });

    svr.Get("/sdapi/v1/loras", [runtime](const httplib::Request&, httplib::Response& res) {
        refresh_lora_cache(*runtime);

        json result = json::array();
        {
            std::lock_guard<std::mutex> lock(*runtime->lora_mutex);
            for (const auto& e : *runtime->lora_cache) {
                json item;
                item["name"] = e.name;
                item["path"] = e.path;
                result.push_back(item);
            }
        }

        res.set_content(result.dump(), "application/json");
    });

    svr.Get("/sdapi/v1/upscalers", [runtime](const httplib::Request&, httplib::Response& res) {
        refresh_upscaler_cache(*runtime);

        auto make_builtin = [](const char* name) {
            json item;
            item["name"]       = name;
            item["model_name"] = nullptr;
            item["model_path"] = nullptr;
            item["model_url"]  = nullptr;
            item["scale"]      = 4;
            return item;
        };

        json result = json::array();
        result.push_back(make_builtin("None"));
        result.push_back(make_builtin("Lanczos"));
        result.push_back(make_builtin("Nearest"));

        {
            std::lock_guard<std::mutex> lock(*runtime->upscaler_mutex);
            for (const auto& e : *runtime->upscaler_cache) {
                json item;
                item["name"]       = e.name;
                item["model_name"] = e.model_name;
                item["model_path"] = e.fullpath;
                item["model_url"]  = nullptr;
                item["scale"]      = e.scale;
                result.push_back(item);
            }
        }

        res.set_content(result.dump(), "application/json");
    });

    svr.Get("/sdapi/v1/latent-upscale-modes", [](const httplib::Request&, httplib::Response& res) {
        json result = json::array({
            {{"name", "Latent"}},
            {{"name", "Latent (nearest)"}},
            {{"name", "Latent (nearest-exact)"}},
            {{"name", "Latent (antialiased)"}},
            {{"name", "Latent (bicubic)"}},
            {{"name", "Latent (bicubic antialiased)"}},
        });
        res.set_content(result.dump(), "application/json");
    });

    svr.Get("/sdapi/v1/samplers", [runtime](const httplib::Request&, httplib::Response& res) {
        std::vector<std::string> sampler_names;
        sampler_names.push_back("default");
        for (int i = 0; i < SAMPLE_METHOD_COUNT; i++) {
            sampler_names.push_back(sd_sample_method_name((sample_method_t)i));
        }
        json r = json::array();
        for (auto name : sampler_names) {
            json entry;
            entry["name"]    = name;
            entry["aliases"] = json::array({name});
            entry["options"] = json::object();
            r.push_back(entry);
        }
        res.set_content(r.dump(), "application/json");
    });

    svr.Get("/sdapi/v1/schedulers", [runtime](const httplib::Request&, httplib::Response& res) {
        std::vector<std::string> scheduler_names;
        scheduler_names.push_back("default");
        for (int i = 0; i < SCHEDULER_COUNT; i++) {
            scheduler_names.push_back(sd_scheduler_name((scheduler_t)i));
        }
        json r = json::array();
        for (auto name : scheduler_names) {
            json entry;
            entry["name"]  = name;
            entry["label"] = name;
            r.push_back(entry);
        }
        res.set_content(r.dump(), "application/json");
    });

    svr.Get("/sdapi/v1/sd-models", [runtime](const httplib::Request&, httplib::Response& res) {
        fs::path model_path = resolve_display_model_path(*runtime);
        json entry;
        entry["title"]      = model_path.stem();
        entry["model_name"] = model_path.stem();
        entry["filename"]   = model_path.filename();
        entry["hash"]       = "8888888888";
        entry["sha256"]     = "8888888888888888888888888888888888888888888888888888888888888888";
        entry["config"]     = nullptr;
        json r              = json::array();
        r.push_back(entry);
        res.set_content(r.dump(), "application/json");
    });

    svr.Get("/sdapi/v1/options", [runtime](const httplib::Request&, httplib::Response& res) {
        fs::path model_path = resolve_display_model_path(*runtime);
        json r;
        r["samples_format"]      = "png";
        r["sd_model_checkpoint"] = model_path.stem();
        res.set_content(r.dump(), "application/json");
    });
}
