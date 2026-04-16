#include "routes.h"

#include <algorithm>
#include <ctime>
#include <regex>

#include "common/common.h"
#include "common/media_io.h"
#include "common/resource_owners.hpp"

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

static bool build_openai_generation_request(const httplib::Request& req,
                                            ServerRuntime& runtime,
                                            ImgGenJobRequest& request,
                                            std::string& error_message) {
    if (req.body.empty()) {
        error_message = "empty body";
        return false;
    }

    json j                    = json::parse(req.body);
    std::string prompt        = j.value("prompt", "");
    int n                     = std::max(1, j.value("n", 1));
    std::string size          = j.value("size", "");
    std::string output_format = j.value("output_format", "png");
    int output_compression    = j.value("output_compression", 100);
    int width                 = runtime.default_gen_params->width > 0 ? runtime.default_gen_params->width : 512;
    int height                = runtime.default_gen_params->width > 0 ? runtime.default_gen_params->height : 512;
    if (!size.empty()) {
        auto pos = size.find('x');
        if (pos != std::string::npos) {
            try {
                width  = std::stoi(size.substr(0, pos));
                height = std::stoi(size.substr(pos + 1));
            } catch (...) {
            }
        }
    }

    if (prompt.empty()) {
        error_message = "prompt required";
        return false;
    }

    request.gen_params = *runtime.default_gen_params;
    if (!assign_output_options(request, output_format, output_compression, true, error_message)) {
        return false;
    }

    request.gen_params.prompt      = prompt;
    request.gen_params.width       = width;
    request.gen_params.height      = height;
    request.gen_params.batch_count = n;

    std::string sd_cpp_extra_args_str = extract_and_remove_sd_cpp_extra_args(request.gen_params.prompt);
    if (!sd_cpp_extra_args_str.empty() && !request.gen_params.from_json_str(sd_cpp_extra_args_str)) {
        error_message = "invalid sd_cpp_extra_args";
        return false;
    }

    // Intentionally disable prompt-embedded LoRA tag parsing for server APIs.
    if (!request.gen_params.resolve_and_validate(IMG_GEN, "", true)) {
        error_message = "invalid params";
        return false;
    }
    return true;
}

static bool build_openai_edit_request(const httplib::Request& req,
                                      ServerRuntime& runtime,
                                      ImgGenJobRequest& request,
                                      std::string& error_message) {
    if (!req.is_multipart_form_data()) {
        error_message = "Content-Type must be multipart/form-data";
        return false;
    }

    std::string prompt = req.form.get_field("prompt");
    if (prompt.empty()) {
        error_message = "prompt required";
        return false;
    }

    size_t image_count    = req.form.get_file_count("image[]");
    bool has_legacy_image = req.form.has_file("image");
    if (image_count == 0 && !has_legacy_image) {
        error_message = "at least one image[] required";
        return false;
    }

    std::vector<std::vector<uint8_t>> images_bytes;
    for (size_t i = 0; i < image_count; ++i) {
        auto file = req.form.get_file("image[]", i);
        images_bytes.emplace_back(file.content.begin(), file.content.end());
    }
    if (image_count == 0 && has_legacy_image) {
        auto file = req.form.get_file("image");
        images_bytes.emplace_back(file.content.begin(), file.content.end());
    }

    std::vector<uint8_t> mask_bytes;
    if (req.form.has_file("mask")) {
        auto file = req.form.get_file("mask");
        mask_bytes.assign(file.content.begin(), file.content.end());
    }

    int n = 1;
    if (req.form.has_field("n")) {
        try {
            n = std::stoi(req.form.get_field("n"));
        } catch (...) {
        }
    }

    std::string size = req.form.get_field("size");
    int width        = -1;
    int height       = -1;
    if (!size.empty()) {
        auto pos = size.find('x');
        if (pos != std::string::npos) {
            try {
                width  = std::stoi(size.substr(0, pos));
                height = std::stoi(size.substr(pos + 1));
            } catch (...) {
            }
        }
    }

    std::string output_format = req.form.has_field("output_format")
                                    ? req.form.get_field("output_format")
                                    : "png";

    int output_compression = 100;
    try {
        output_compression = std::stoi(req.form.get_field("output_compression"));
    } catch (...) {
    }

    request.gen_params = *runtime.default_gen_params;
    if (!assign_output_options(request, output_format, output_compression, false, error_message)) {
        return false;
    }

    request.gen_params.prompt      = prompt;
    request.gen_params.width       = width;
    request.gen_params.height      = height;
    request.gen_params.batch_count = n;

    for (auto& bytes : images_bytes) {
        int img_w           = 0;
        int img_h           = 0;
        uint8_t* raw_pixels = load_image_from_memory(
            reinterpret_cast<const char*>(bytes.data()),
            static_cast<int>(bytes.size()),
            img_w, img_h,
            width, height, 3);
        if (raw_pixels == nullptr) {
            continue;
        }

        SDImageOwner image_owner({(uint32_t)img_w, (uint32_t)img_h, 3, raw_pixels});
        request.gen_params.set_width_and_height_if_unset(image_owner.get().width, image_owner.get().height);
        request.gen_params.ref_images.push_back(std::move(image_owner));
    }

    if (!request.gen_params.ref_images.empty()) {
        request.gen_params.init_image = request.gen_params.ref_images.front();
    }

    if (!mask_bytes.empty()) {
        int expected_width  = 0;
        int expected_height = 0;
        if (request.gen_params.width_and_height_are_set()) {
            expected_width  = request.gen_params.width;
            expected_height = request.gen_params.height;
        }
        int mask_w = 0;
        int mask_h = 0;

        uint8_t* mask_raw = load_image_from_memory(
            reinterpret_cast<const char*>(mask_bytes.data()),
            static_cast<int>(mask_bytes.size()),
            mask_w, mask_h,
            expected_width, expected_height, 1);
        request.gen_params.mask_image.reset({(uint32_t)mask_w, (uint32_t)mask_h, 1, mask_raw});
        const sd_image_t& mask_image = request.gen_params.mask_image.get();
        request.gen_params.set_width_and_height_if_unset(mask_image.width, mask_image.height);
    } else {
        request.gen_params.mask_image.reset({
            (uint32_t)request.gen_params.get_resolved_width(),
            (uint32_t)request.gen_params.get_resolved_height(),
            1,
            nullptr,
        });
    }

    std::string sd_cpp_extra_args_str = extract_and_remove_sd_cpp_extra_args(request.gen_params.prompt);
    if (!sd_cpp_extra_args_str.empty() && !request.gen_params.from_json_str(sd_cpp_extra_args_str)) {
        error_message = "invalid sd_cpp_extra_args";
        return false;
    }

    // Intentionally disable prompt-embedded LoRA tag parsing for server APIs.
    if (!request.gen_params.resolve_and_validate(IMG_GEN, "", true)) {
        error_message = "invalid params";
        return false;
    }

    return true;
}

static bool execute_sync_img_gen_request(ServerRuntime& runtime,
                                         ImgGenJobRequest& request,
                                         SDImageVec& results,
                                         std::string& error_message) {
    sd_img_gen_params_t img_gen_params = request.to_sd_img_gen_params_t();
    int num_results                    = 0;

    {
        std::lock_guard<std::mutex> lock(*runtime.sd_ctx_mutex);
        sd_image_t* raw_results = generate_image(runtime.sd_ctx, &img_gen_params);
        num_results             = request.gen_params.batch_count;
        results.adopt(raw_results, num_results);
    }

    if (results.empty()) {
        error_message = "generate_image returned no results";
        return false;
    }
    return true;
}

void register_openai_api_endpoints(httplib::Server& svr, ServerRuntime& rt) {
    ServerRuntime* runtime = &rt;

    svr.Get("/v1/models", [runtime](const httplib::Request&, httplib::Response& res) {
        json r;
        r["data"] = json::array();
        r["data"].push_back({{"id", "sd-cpp-local"}, {"object", "model"}, {"owned_by", "local"}});
        res.set_content(r.dump(), "application/json");
    });

    svr.Post("/v1/images/generations", [runtime](const httplib::Request& req, httplib::Response& res) {
        try {
            ImgGenJobRequest request;
            std::string error_message;
            if (!build_openai_generation_request(req, *runtime, request, error_message)) {
                res.status = 400;
                res.set_content(json({{"error", error_message}}).dump(), "application/json");
                return;
            }

            LOG_DEBUG("%s\n", request.gen_params.to_string().c_str());

            SDImageVec results;
            if (!execute_sync_img_gen_request(*runtime, request, results, error_message)) {
                res.status = 500;
                res.set_content(json({{"error", error_message}}).dump(), "application/json");
                return;
            }

            json out;
            out["created"]       = static_cast<long long>(std::time(nullptr));
            out["data"]          = json::array();
            out["output_format"] = request.output_format;

            for (int i = 0; i < request.gen_params.batch_count; ++i) {
                if (results[i].data == nullptr) {
                    continue;
                }
                std::string params = request.gen_params.embed_image_metadata
                                         ? get_image_params(*runtime->ctx_params,
                                                            request.gen_params,
                                                            request.gen_params.seed + i)
                                         : "";
                auto image_bytes   = encode_image_to_vector(request.output_format == "jpeg"
                                                                ? EncodedImageFormat::JPEG
                                                            : request.output_format == "webp"
                                                                ? EncodedImageFormat::WEBP
                                                                : EncodedImageFormat::PNG,
                                                          results[i].data,
                                                          results[i].width,
                                                          results[i].height,
                                                          results[i].channel,
                                                          params,
                                                          request.output_compression);
                if (image_bytes.empty()) {
                    LOG_ERROR("write image to mem failed");
                    continue;
                }

                json item;
                item["b64_json"] = base64_encode(image_bytes);
                out["data"].push_back(item);
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
    });

    svr.Post("/v1/images/edits", [runtime](const httplib::Request& req, httplib::Response& res) {
        try {
            ImgGenJobRequest request;
            std::string error_message;
            if (!build_openai_edit_request(req, *runtime, request, error_message)) {
                res.status = 400;
                res.set_content(json({{"error", error_message}}).dump(), "application/json");
                return;
            }

            LOG_DEBUG("%s\n", request.gen_params.to_string().c_str());

            SDImageVec results;
            if (!execute_sync_img_gen_request(*runtime, request, results, error_message)) {
                res.status = 500;
                res.set_content(json({{"error", error_message}}).dump(), "application/json");
                return;
            }

            json out;
            out["created"]       = static_cast<long long>(std::time(nullptr));
            out["data"]          = json::array();
            out["output_format"] = request.output_format;

            for (int i = 0; i < request.gen_params.batch_count; ++i) {
                if (results[i].data == nullptr) {
                    continue;
                }
                std::string params = request.gen_params.embed_image_metadata
                                         ? get_image_params(*runtime->ctx_params,
                                                            request.gen_params,
                                                            request.gen_params.seed + i)
                                         : "";
                auto image_bytes   = encode_image_to_vector(request.output_format == "jpeg" ? EncodedImageFormat::JPEG : EncodedImageFormat::PNG,
                                                          results[i].data,
                                                          results[i].width,
                                                          results[i].height,
                                                          results[i].channel,
                                                          params,
                                                          request.output_compression);
                json item;
                item["b64_json"] = base64_encode(image_bytes);
                out["data"].push_back(item);
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
    });
}
