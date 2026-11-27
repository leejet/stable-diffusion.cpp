#ifndef __SD_DGL_MODELS_DETAIL_DETECTOR_H__
#define __SD_DGL_MODELS_DETAIL_DETECTOR_H__

#include <memory>
#include <string>
#include <vector>

#include "config.h"
#include "core/ggml_extend_backend.h"
#include "dgl/models/model_base.h"
#include "stable-diffusion.h"

struct YOLOv8Runner;

namespace sd::dgl::models {

    struct DetailDetection {
        float x1         = 0.f;
        float y1         = 0.f;
        float x2         = 0.f;
        float y2         = 0.f;
        float confidence = 0.f;
        int class_id     = 0;
    };

    struct DetectionOptions {
        int input_size      = 640;
        float confidence    = 0.3f;
        float nms_threshold = 0.45f;
        int max_detections  = 100;
    };

    struct DetailDetectorParams {
        std::string path;
        std::string backend;
        std::string params_backend;
        bool linear_i8 = false;
    };

    struct DetailDetector final : public Model {
        SDBackendManager backend_manager;
        std::shared_ptr<YOLOv8Runner> detector;
        std::vector<std::string> class_names;

        DetailDetector(std::shared_ptr<ModelManager> manager, std::shared_ptr<SDBackendPool> backends);
        ~DetailDetector() override;

        bool init(DetailDetectorParams params);
        std::vector<DetailDetection> predict(sd::Image image,
                                             const DetectionOptions& params);

    private:
        DetailDetectorParams params_;
        bool build(ModelLoader loader);
        bool build_variant(const Model& base, ModelLoader loader) override;
        bool attach_loras() override;
        void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override;
    };

}  // namespace sd::dgl::models

#endif  // __SD_DGL_MODELS_DETAIL_DETECTOR_H__
