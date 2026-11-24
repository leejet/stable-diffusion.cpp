#include "api/generation.h"

#include "api/result.h"
#include "dgl/executor.h"
#include "dgl/models/upscaler.h"
#include "dgl/run_plan.h"
#include "model/upscaler/scale.hpp"

struct sd_upscaler {
    sd::api::GenerationContext context;
    std::shared_ptr<sd::dgl::models::ImageUpscaler> model;
    sd::dgl::Graph program;
    std::string source;

    explicit sd_upscaler(const sd_opt_t* options)
        : context(options, sd::api::Upscaler) {}
};

namespace sd::api {
    static std::string upscale_source(const OptConfig& config, int repeats) {
        const auto& settings = config.context;
        std::string options;
        for (const auto& [name, spec] : std::map<std::string, const char*>{{"backend", settings.backend}, {"params_backend", settings.params_backend}}) {
            if (!spec || !*spec)
                continue;
            SDBackendAssignment assignment;
            std::string error;
            if (!sd_parse_backend_assignment(spec, &assignment, &error))
                throw dgl::Error(SD_STATUS_INVALID_ARGUMENT, error);
            options += ", " + name + " = " + nlohmann::json(assignment.get(SDBackendModule::UPSCALER)).dump();
        }
        if (config.post.conv_direct)
            options += ", direct = true";
        return render_template("upscale", {{"model_path", nlohmann::json(settings.model_path).dump()},
                                           {"tile_size", std::to_string(config.post.tile_size)},
                                           {"repeats", std::to_string(repeats)},
                                           {"loader_options", options}});
    }
}

sd_status_t sd_upscaler_create(const sd_upscaler_opt_t* options, sd_upscaler_t** out) {
    if (out)
        *out = nullptr;
    return sd::api::call([&] {
        if (!out)
            return sd::api::error(SD_STATUS_INVALID_ARGUMENT, nullptr, "out is required");
        auto result          = std::make_unique<sd_upscaler_t>(options);
        result->source       = sd::api::upscale_source(result->context.model.config, 1);
        result->program      = sd::dgl::compile(result->source, sd::dgl::operations(), true);
        const auto& resource = result->program.resources.at(0);
        auto arguments       = resource.arguments;
        sd::dgl::RunEvents events({}, false);
        sd::dgl::RunEventsScope scope(events);
        sd::dgl::RunState state{result->context.execution, events};
        sd::dgl::RunPlan plan(result->program, state);
        result->model = plan.load_resource(*resource.operation, arguments).as<sd::dgl::models::ImageUpscaler>();
        if (result->model->get_scale() <= 0)
            throw sd::dgl::Error(SD_STATUS_LOAD_FAILED, "invalid image upscaler scale");
        *out = result.release();
        return SD_STATUS_OK;
    },
                         SD_STATUS_LOAD_FAILED);
}

void sd_upscaler_free(sd_upscaler_t* upscaler) {
    delete upscaler;
}

int sd_upscaler_get_scale(const sd_upscaler_t* upscaler) {
    sd::api::clear_error();
    if (!upscaler || !upscaler->model) {
        sd::api::error(SD_STATUS_INVALID_ARGUMENT, nullptr, "upscaler is required");
        return 0;
    }
    return upscaler->model->get_scale();
}

int sd_upscaler_probe_scale(const char* path) {
    int scale = 0;
    sd::api::call([&] {
        if (!path || !*path)
            return sd::api::error(SD_STATUS_INVALID_ARGUMENT, "model_path", "model path is required");
        scale = sd::upscaling::probe_scale(path);
        return scale ? SD_STATUS_OK : sd::api::error(SD_STATUS_LOAD_FAILED, "model_path", "not a recognized RGB upscaler");
    },
                  SD_STATUS_LOAD_FAILED);
    return scale;
}

sd_status_t sd_upscale(sd_upscaler_t* upscaler, const sd_upscale_opt_t* options, sd_result_t** out) {
    if (out)
        *out = nullptr;
    return sd::api::call([&] {
        if (!out || !upscaler)
            return sd::api::error(SD_STATUS_INVALID_ARGUMENT, nullptr, "upscaler and out are required");
        sd::api::GenerationRun request(upscaler->context);
        auto snapshot = sd::api::copy_options(options);
        sd::OptConfig config;
        const auto status = sd::api::resolve(snapshot.get(), sd::api::Upscale, config);
        if (status != SD_STATUS_OK)
            return status;
        int64_t width = config.post.image.width, height = config.post.image.height;
        const auto scale = upscaler->model->get_scale();
        for (int i = 0; i < config.post.repeats; ++i) {
            width *= scale;
            height *= scale;
            if (width > 16384 || height > 16384 || width * height > 64 * 1024 * 1024)
                throw sd::dgl::Error(SD_STATUS_OUT_OF_RANGE, "upscaled output exceeds 16384 per axis or 64 megapixels");
        }
        const auto* input = sd::api::bound_tensor(snapshot.get(), "image");
        sd::api::GenerationProgram program;
        program.source = config.post.repeats == 1 ? upscaler->source : sd::api::upscale_source(upscaler->context.model.config, config.post.repeats);
        program.inputs.emplace("input_image", sd::dgl::Value::object(std::shared_ptr<sd_tensor_t>(sd_tensor_retain(input), sd_tensor_free)));
        auto value  = sd::api::execute_program(upscaler->context, program, config.post.repeats == 1 ? &upscaler->program : nullptr);
        auto result = std::make_unique<sd_result_t>();
        sd::api::append_program_image(*result, value, program.source);
        *out = result.release();
        return SD_STATUS_OK;
    },
                         SD_STATUS_GENERATION_FAILED);
}
