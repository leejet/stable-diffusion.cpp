#include "api.h"
#include "generation.h"
#include "options.h"
#include "result.h"
#include "tensor.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

#include "core/ggml_extend.h"
#include "core/option_types.h"
#include "core/util.h"

static thread_local char last_error[512] = {};

sd_ctx_t::sd_ctx_t()  = default;
sd_ctx_t::~sd_ctx_t() = default;

void sd::api::clear_error() {
    last_error[0] = '\0';
}

sd_status_t sd::api::error(sd_status_t status, const char* key, const char* reason) {
    snprintf(last_error, sizeof(last_error), "%s%s%s", key ? key : "", key ? ": " : "", reason);
    return status;
}

const char* sd_get_last_error(void) {
    return last_error;
}

const char* sd_type_name(enum sd_type_t type) {
    if (type == SD_TYPE_F8_E4M3) {
        return "f8_e4m3";
    }
    if (type == SD_TYPE_F8_E5M2) {
        return "f8_e5m2";
    }
    const auto ggml_type = sd_type_to_ggml_type(type);
    if (ggml_type != GGML_TYPE_COUNT) {
        return ggml_type_name(ggml_type);
    }
    return "NONE";
}

enum sd_type_t str_to_sd_type(const char* str) {
    if (!strcmp(str, "f8_e4m3")) {
        return SD_TYPE_F8_E4M3;
    }
    if (!strcmp(str, "f8_e5m2")) {
        return SD_TYPE_F8_E5M2;
    }
    for (int i = 0; i < std::min<int>(SD_TYPE_COUNT, GGML_TYPE_COUNT); i++) {
        auto trait = ggml_get_type_traits((ggml_type)i);
        if (!strcmp(str, trait->type_name)) {
            return (enum sd_type_t)i;
        }
    }
    return SD_TYPE_COUNT;
}

sd_status_t sd_set_preview_callback(sd_preview_cb_t cb, const char* mode, int interval, bool denoised, bool noisy, void* data) {
    sd::api::clear_error();
    if (!mode)
        return sd::api::error(SD_STATUS_INVALID_ARGUMENT, "preview.mode", "mode is required");
    const auto parsed = str_to_preview(mode);
    if (parsed == PREVIEW_COUNT)
        return sd::api::error(SD_STATUS_INVALID_ARGUMENT, "preview.mode", "unknown preview mode");
    sd_set_preview_callback_impl(cb, parsed, interval, denoised, noisy, data);
    return SD_STATUS_OK;
}

void sd_set_prompt_image_callback(sd_ctx_t* ctx, sd_prompt_image_cb_t callback, void* data) {
    if (!ctx || !ctx->engine)
        return;
    auto& engine = *ctx->engine;
    std::lock_guard<std::recursive_mutex> lock(engine.mutex);
    if (!engine.executing) {
        engine.prompt_image      = callback;
        engine.prompt_image_data = data;
    }
}

void sd_set_generation_progress_callback(sd_ctx_t* ctx, sd_generation_progress_cb_t callback, void* data) {
    if (!ctx || !ctx->engine)
        return;
    auto& engine = *ctx->engine;
    std::lock_guard<std::recursive_mutex> lock(engine.mutex);
    if (!engine.executing) {
        engine.progress      = callback;
        engine.progress_data = data;
        engine.sample_pass   = 0;
    }
}

sd_status_t sd_ctx_create(const sd_ctx_opt_t* options, sd_ctx_t** out) {
    if (out)
        *out = nullptr;
    return sd::api::call([&] {
        if (!out)
            return sd::api::error(SD_STATUS_INVALID_ARGUMENT, nullptr, "ctx_out is required");
        auto ctx    = std::make_unique<sd_ctx_t>();
        ctx->engine = std::make_unique<sd::api::GenerationContext>(options, sd::api::Context);
        *out        = ctx.release();
        return SD_STATUS_OK;
    },
                         SD_STATUS_LOAD_FAILED);
}

void sd_ctx_free(sd_ctx_t* context) {
    delete context;
}

sd_dgl_ctx_t* sd_ctx_get_dgl_ctx(sd_ctx_t* context) {
    return context && context->engine ? &context->engine->execution : nullptr;
}

static sd_status_t set_model_options(sd_ctx_t* context, const sd_ctx_opt_t* options, sd_result_kind_t kind) {
    return sd::api::call([&] {
        if (!context || !context->engine)
            return sd::api::error(SD_STATUS_INVALID_ARGUMENT, nullptr, "context is required");
        auto& engine = *context->engine;
        std::unique_lock<std::recursive_mutex> lock(engine.mutex, std::try_to_lock);
        if (!lock.owns_lock())
            return sd::api::error(SD_STATUS_BUSY, nullptr, "context is busy");
        sd::api::GenerationRun replacement(engine);
        sd::api::GenerationModel candidate(sd::api::copy_options(options), sd::api::Context);
        if (kind != SD_RESULT_INVALID) {
            sd::FileAccessScope access(engine.file_access);
            const bool video = kind == SD_RESULT_VIDEO;
            if (!sd::api::supports_generation(candidate.definition(engine), video))
                return sd::api::error(SD_STATUS_UNSUPPORTED, nullptr, video ? "selected model does not support vid_gen" : "selected model does not support img_gen");
        }
        engine.model = std::move(candidate);
        sd::api::clear_error();
        return SD_STATUS_OK;
    },
                         kind == SD_RESULT_INVALID ? SD_STATUS_INVALID_ARGUMENT : SD_STATUS_LOAD_FAILED);
}

sd_status_t sd_ctx_set_model_options(sd_ctx_t* context, const sd_ctx_opt_t* options) {
    return set_model_options(context, options, SD_RESULT_INVALID);
}

sd_status_t sd_ctx_set_model_options_for_generation(sd_ctx_t* context, const sd_ctx_opt_t* options, sd_result_kind_t kind) {
    if (kind != SD_RESULT_IMAGE && kind != SD_RESULT_VIDEO)
        return sd::api::error(SD_STATUS_INVALID_ARGUMENT, nullptr, "image or video result kind is required");
    return set_model_options(context, options, kind);
}

void sd_cancel_generation(sd_ctx_t* context, enum sd_cancel_mode_t mode) {
    if (!context || !context->engine)
        return;
    if (mode < SD_CANCEL_ALL || mode > SD_CANCEL_RESET)
        mode = SD_CANCEL_ALL;
    auto& engine = *context->engine;
    std::lock_guard<std::recursive_mutex> lock(engine.mutex);
    if (!engine.executing)
        return;
    engine.cancellation.store(mode);
    if (engine.active_run && mode != SD_CANCEL_NEW_LATENTS)
        engine.active_run->set_cancelled(mode == SD_CANCEL_ALL);
}

static const ModelDefinition* context_definition(const sd_ctx_t* context) {
    const ModelDefinition* definition = nullptr;
    sd::api::call([&] {
        if (!context || !context->engine)
            return sd::api::error(SD_STATUS_INVALID_ARGUMENT, nullptr, "context is required");
        auto& engine = *context->engine;
        std::lock_guard<std::recursive_mutex> lock(engine.mutex);
        const auto family = engine.model.family;
        if (family != VERSION_COUNT) {
            definition = find_model_definition(family);
        } else {
            sd::FileAccessScope access(engine.file_access);
            definition = &engine.model.definition(engine);
        }
        sd::api::clear_error();
        return SD_STATUS_OK;
    },
                  SD_STATUS_LOAD_FAILED);
    return definition;
}

bool sd_ctx_supports_image_generation(const sd_ctx_t* context) {
    const auto* definition = context_definition(context);
    return definition && sd::api::supports_generation(*definition, false);
}

bool sd_ctx_supports_video_generation(const sd_ctx_t* context) {
    const auto* definition = context_definition(context);
    return definition && sd::api::supports_generation(*definition, true);
}

const char* sd_get_model_version_name(const sd_ctx_t* context) {
    const auto* definition = context_definition(context);
    return definition ? definition->name : "Unknown";
}

sd_status_t sd_ctx_set_file_access_callback(sd_ctx_t* context, sd_dgl_file_access_cb_t callback, void* data) {
    return sd::api::call([&] {
        if (!context || !context->engine)
            return sd::api::error(SD_STATUS_INVALID_ARGUMENT, nullptr, "context is required");
        auto& engine = *context->engine;
        std::unique_lock<std::recursive_mutex> lock(engine.mutex, std::try_to_lock);
        if (!lock.owns_lock() || engine.executing)
            return sd::api::error(SD_STATUS_BUSY, nullptr, "context is busy");
        engine.file_access = {callback, data};
        return SD_STATUS_OK;
    });
}

sd_status_t sd_generate_image(sd_ctx_t* context, const sd_img_gen_opt_t* params, sd_result_t** out) {
    if (out)
        *out = nullptr;
    return sd::api::call([&] {
        if (!out || !context || !context->engine)
            return sd::api::error(SD_STATUS_INVALID_ARGUMENT, nullptr, "context and result_out are required");
        return sd::api::generate_images(*context->engine, params, out);
    },
                         SD_STATUS_GENERATION_FAILED);
}

sd_status_t sd_generate_video(sd_ctx_t* context, const sd_vid_gen_opt_t* params, sd_result_t** out) {
    if (out)
        *out = nullptr;
    return sd::api::call([&] {
        if (!out || !context || !context->engine)
            return sd::api::error(SD_STATUS_INVALID_ARGUMENT, nullptr, "context and result_out are required");
        return sd::api::generate_video(*context->engine, params, out);
    },
                         SD_STATUS_GENERATION_FAILED);
}
