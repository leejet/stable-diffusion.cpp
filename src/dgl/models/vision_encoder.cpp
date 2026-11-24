#include "dgl/models/vision_encoder.h"

#include <algorithm>

#include "conditioning/conditioner.hpp"
#include "core/util.h"
#include "dgl/models/component_sources.h"

namespace sd::dgl::models {
    VisionEncoder::VisionEncoder(std::shared_ptr<ModelManager> manager, std::shared_ptr<SDBackendPool> backends)
        : Model(std::move(manager)),
          runtime_(*this, std::move(backends), ModelComponent::CLIPVision, SDBackendModule::CLIP_VISION, [this] { return build_runners(); }) {}

    bool VisionEncoder::init(ModelLoadParams params) {
        return runtime_.init(std::move(params), load_component_catalog<ModelComponent::CLIPVision>);
    }

    bool VisionEncoder::build_runners() {
        clip_vision = std::make_shared<FrozenCLIPVisionEmbedder>(runtime_.backend_for(), model_weights->loader().get_tensor_storage_map(), model_weights);
        runtime_.configure_runner(*clip_vision);
        return runtime_.register_runner_params(runtime_.component, clip_vision);
    }

    sd::Tensor<float> VisionEncoder::get_clip_vision_output(const sd::Tensor<float>& image,
                                                            bool return_pooled,
                                                            int clip_skip,
                                                            bool zero_out_masked,
                                                            bool zero_input) {
        sd::Tensor<float> output;
        if (zero_out_masked) {
            if (return_pooled) {
                output = sd::zeros<float>({clip_vision->vision_model.projection_dim});
            } else {
                output = sd::zeros<float>({clip_vision->vision_model.hidden_size, 257});
            }
        } else {
            auto pixel_values = clip_preprocess(image, clip_vision->vision_model.image_size, clip_vision->vision_model.image_size);
            if (zero_input)
                pixel_values.fill_(0.f);
            auto output_opt = clip_vision->compute(n_threads, pixel_values, return_pooled, clip_skip);
            if (output_opt.empty()) {
                LOG_ERROR("clip_vision compute failed");
                return {};
            }
            output = std::move(output_opt);
        }
        return output;
    }
}
