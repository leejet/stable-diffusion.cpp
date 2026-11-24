#ifndef __SD_DGL_MODELS_UPSCALER_H__
#define __SD_DGL_MODELS_UPSCALER_H__

#include "core/ggml_extend_backend.h"
#include "core/media.h"
#include "core/tensor.hpp"
#include "dgl/models/image_upscale.h"
#include "dgl/models/model_base.h"
#include "stable-diffusion.h"

#include <memory>
#include <string>

namespace sd::dgl::models {

    struct ImageUpscalerParams {
        std::string path;
        std::string backend;
        std::string params_backend;
        bool direct;
        int tile_size;
    };

    struct ImageUpscaler final : public Model {
        SDBackendManager backend_manager;
        ggml_type model_data_type = GGML_TYPE_F16;
        std::unique_ptr<ImageUpscaleModel> model;

        ImageUpscaler(std::shared_ptr<ModelManager> manager, std::shared_ptr<SDBackendPool> backends);
        ~ImageUpscaler() override;

        bool init(ImageUpscalerParams params);
        int get_scale();
        sd::Tensor<float> upscale_tensor(const sd::Tensor<float>& input_tensor);
        sd::Image upscale(sd::Image input_image);

    private:
        ImageUpscalerParams params_;
        bool build(ModelLoader loader);
        ModelLoader base_loader() const override;
        bool build_variant(const Model& base, ModelLoader loader) override;
        bool attach_loras() override;
        void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override;
    };

}  // namespace sd::dgl::models

#endif  // __SD_DGL_MODELS_UPSCALER_H__
