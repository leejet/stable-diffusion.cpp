#ifndef __SD_DGL_MODELS_IP_ADAPTER_H__
#define __SD_DGL_MODELS_IP_ADAPTER_H__

#include "core/tensor.hpp"
#include "dgl/models/fwd.h"
#include "dgl/models/model_runtime.h"

namespace sd::dgl::models {
    class IPAdapter final : public Model {
        ModelRuntime runtime_;

        bool refresh_weights() override { return runtime_.refresh_weights(); }
        bool attach_loras() override { return runtime_.attach_loras(); }
        bool build_variant(const Model& source, ModelLoader loader) override {
            return runtime_.build_variant(static_cast<const IPAdapter&>(source).runtime_, std::move(loader));
        }

        bool build_runners();

    public:
        SDVersion version() const { return runtime_.version; }
        bool init(ModelLoadParams params);
        std::shared_ptr<::IPAdapter::IPAdapterRunner> ip_adapter;

        IPAdapter(std::shared_ptr<ModelManager> manager, std::shared_ptr<SDBackendPool> backends);
        Tensor<float> project_image(const Tensor<float>& embedding);
        void validate_tokens(const Tensor<float>& tokens) const;
        void validate_target(const DiT& target) const;
    };
}

#endif  // __SD_DGL_MODELS_IP_ADAPTER_H__
