#ifndef __SD_DGL_MODELS_IMAGE_UPSCALE_H__
#define __SD_DGL_MODELS_IMAGE_UPSCALE_H__

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/tensor.hpp"
#include "ggml.h"
#include "model_manager.h"

class SDBackendManager;
struct GGMLRunner;
struct ESRGAN;
struct MoSR;

namespace sd::dgl::models {
    class ImageUpscaleModel {
        friend std::unique_ptr<ImageUpscaleModel> build_image_upscale_model(
            const std::shared_ptr<ModelManager::ResourceHandle>&,
            SDBackendManager&,
            ModelLoader,
            const std::string&,
            bool,
            ggml_type);
        friend Tensor<float> upscale_image_tensor(ImageUpscaleModel&, const Tensor<float>&, int, int);

        std::weak_ptr<ModelManager::ResourceHandle> manager_;
        std::unique_ptr<ESRGAN> esrgan_;
        std::unique_ptr<MoSR> mosr_;
        std::vector<ggml_tensor*> weights_;
        uint64_t source_file_ = 0;

        explicit ImageUpscaleModel(const std::shared_ptr<ModelManager::ResourceHandle>& manager);

    public:
        ~ImageUpscaleModel();
        GGMLRunner* runner() const;
        int scale() const;
        ModelLoader::FileId source_file() const { return source_file_; }
    };

    std::unique_ptr<ImageUpscaleModel> build_image_upscale_model(
        const std::shared_ptr<ModelManager::ResourceHandle>& manager,
        SDBackendManager& backends,
        ModelLoader loader,
        const std::string& path,
        bool conv_direct      = false,
        ggml_type weight_type = GGML_TYPE_F16);

    Tensor<float> upscale_image_tensor(ImageUpscaleModel& model,
                                       const Tensor<float>& input,
                                       int n_threads,
                                       int tile_size);
}

#endif  // __SD_DGL_MODELS_IMAGE_UPSCALE_H__
