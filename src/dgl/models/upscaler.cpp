#include "dgl/models/upscaler.h"

#include <cstdlib>
#include <utility>

#include "core/ggml_extend_backend.h"
#include "core/util.h"
#include "model/upscaler/esrgan.hpp"
#include "model/upscaler/mosr.hpp"
#include "model_loader.h"
#include "stable-diffusion.h"

namespace sd::dgl::models {

    ImageUpscaler::ImageUpscaler(std::shared_ptr<ModelManager> manager, std::shared_ptr<SDBackendPool> backends)
        : Model(std::move(manager)), backend_manager(std::move(backends)) {
        params_placement_ = [this](const std::string& name) {
            std::string error;
            if (!backend_manager.init(params_.backend.c_str(), name.c_str(), nullptr, &error))
                throw std::runtime_error(error);
            params_.params_backend = name;
            return true;
        };
    }

    ImageUpscaler::~ImageUpscaler() {
        model.reset();
        release_weights();
    }

    int ImageUpscaler::get_scale() {
        return model != nullptr ? model->scale() : 1;
    }

    bool ImageUpscaler::init(ImageUpscalerParams params) {
        params_ = std::move(params);
        ggml_log_set(sd_ggml_log_callback, nullptr);
        ModelLoader loader;
        if (!loader.init_from_file_and_convert_name(params_.path, "", VERSION_ESRGAN)) {
            LOG_ERROR("init upscaler model loader failed: '%s'", params_.path.c_str());
            return false;
        }
        loader.set_wtype_override(model_data_type);
        return build(std::move(loader));
    }

    ModelLoader ImageUpscaler::base_loader() const {
        return model_weights->model_source(model->source_file());
    }

    bool ImageUpscaler::build_variant(const Model& source, ModelLoader loader) {
        const auto& base = static_cast<const ImageUpscaler&>(source);
        params_          = base.params_;
        model_data_type  = base.model_data_type;
        return build(std::move(loader));
    }

    bool ImageUpscaler::build(ModelLoader loader) {
        sd::ParallelScope tensor_scope(tensor_executor.get());
        std::string error;
        if (!backend_manager.init(params_.backend.c_str(), params_.params_backend.c_str(), nullptr, &error)) {
            LOG_ERROR("upscaler backend config failed: %s", error.c_str());
            return false;
        }
        model = build_image_upscale_model(model_weights, backend_manager, std::move(loader), params_.path,
                                          params_.direct, model_data_type);
        if (!model)
            return false;
        model->runner()->set_linear_i8_enabled(params_.linear_i8);
        return load_weights_eagerly();
    }

    sd::Tensor<float> ImageUpscaler::upscale_tensor(const sd::Tensor<float>& input_tensor) {
        Model::ExecutionScope scope(*this);
        scope.require_ready();
        if (model == nullptr) {
            return {};
        }
        return upscale_image_tensor(*model, input_tensor, n_threads, params_.tile_size);
    }

    bool ImageUpscaler::attach_loras() {
        const auto& loader = model_weights->model_source(model->source_file());
        const auto mode    = resolve_lora_mode(lora_mode(), loader, backend_manager, SDBackendModule::UPSCALER);
        return prepare_loras(VERSION_COUNT, backend_manager, SDBackendModule::UPSCALER,
                             n_threads, mode == LORA_APPLY_IMMEDIATELY);
    }

    void ImageUpscaler::set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) {
        model->runner()->set_weight_adapter(adapter);
    }

    sd::Image ImageUpscaler::upscale(sd::Image input_image) {
        sd::ParallelScope tensor_scope(tensor_executor.get());
        sd::Image upscaled_image = {0, 0, 0, nullptr};
        const int scale          = get_scale();
        int output_width         = (int)input_image.width * scale;
        int output_height        = (int)input_image.height * scale;
        LOG_INFO("upscaling from (%i x %i) to (%i x %i)",
                 input_image.width, input_image.height, output_width, output_height);

        sd::Tensor<float> input_tensor = sd_image_to_tensor(input_image);
        sd::Tensor<float> upscaled;
        int64_t t0 = ggml_time_ms();
        upscaled   = upscale_tensor(input_tensor);
        if (upscaled.empty()) {
            return upscaled_image;
        }
        sd::Image upscaled_data = tensor_to_sd_image(upscaled);
        int64_t t3              = ggml_time_ms();
        LOG_INFO("input_image_tensor upscaled, taking %.2fs", (t3 - t0) / 1000.0f);
        upscaled_image = upscaled_data;
        return upscaled_image;
    }

}  // namespace sd::dgl::models
