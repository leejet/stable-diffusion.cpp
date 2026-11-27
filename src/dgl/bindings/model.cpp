#include "dgl/bindings/model.h"

#include "dgl/bindings/conversion.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "dgl/context.h"
#include "dgl/models/detail_detector.h"
#include "dgl/models/upscaler.h"

namespace sd::dgl {
    PreparedResource execute_load_upscaler(Values& args) {
        PreparedResource prepared;
        prepared.revision = nlohmann::json::object();
        models::ImageUpscalerParams params{
            file_key(get_string(args, "path"), prepared.revision),
            trim(get_string(args, "backend")),
            trim(get_string(args, "params_backend")),
            get_bool(args, "direct"),
            int(get_int(args, "tile_size")),
            get_string(args, "linear_type") == "i8",
        };
        prepared.key  = {{"path", params.path}, {"backend", params.backend}, {"params_backend", params.params_backend}, {"direct", params.direct}, {"tile_size", params.tile_size}, {"linear_i8", params.linear_i8}};
        prepared.load = [params = std::move(params)](ResourceLoadRequest& request) mutable {
            auto model = request.context.make_model<models::ImageUpscaler>(request.weights_key);
            if (!model->init(std::move(params)))
                throw std::runtime_error("upscaler loading failed");
            return Value::object(std::move(model));
        };
        return prepared;
    }

    Value execute_upscale(Values& args, RunState&) {
        auto image           = image_view(args.at("image"));
        auto model           = object<models::ImageUpscaler>(args, "upscaler");
        const int64_t factor = model->get_scale();
        if (factor <= 0 || image.channel != 3 || int64_t(image.width) * factor > 16384 || int64_t(image.height) * factor > 16384 ||
            int64_t(image.width) * image.height * factor * factor > 64 * 1024 * 1024)
            throw Error(SD_STATUS_OUT_OF_RANGE, "upscale requires RGB and output at most 64 megapixels at the model's actual scale");
        if (args.count("factor") && get_int(args, "factor") != factor)
            LOG_WARN("upscale factor=%lld is ignored; the model's actual scale is %lld", (long long)get_int(args, "factor"), (long long)factor);
        auto output = model->upscale(image);
        std::unique_ptr<uint8_t, decltype(&free)> owner(output.data, free);
        if (!output.data)
            throw std::runtime_error("upscale failed");
        return Value::object(copy_image(output));
    }

    PreparedResource execute_load_detector(Values& args) {
        PreparedResource prepared;
        prepared.revision = nlohmann::json::object();
        models::DetailDetectorParams params{
            file_key(get_string(args, "path"), prepared.revision),
            trim(get_string(args, "backend")),
            trim(get_string(args, "params_backend")),
            get_string(args, "linear_type") == "i8",
        };
        prepared.key  = {{"path", params.path}, {"backend", params.backend}, {"params_backend", params.params_backend}, {"linear_i8", params.linear_i8}};
        prepared.load = [params = std::move(params)](ResourceLoadRequest& request) mutable {
            auto model = request.context.make_model<models::DetailDetector>(request.weights_key);
            if (!model->init(std::move(params)))
                throw std::runtime_error("detector loading failed");
            return Value::object(std::move(model));
        };
        return prepared;
    }

}  // namespace sd::dgl
