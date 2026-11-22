#ifndef __SD_DGL_OPS_IP_ADAPTER_H__
#define __SD_DGL_OPS_IP_ADAPTER_H__

#include "dgl/models/fwd.h"

#include "model/adapter/ip_adapter.hpp"

namespace sd::dgl::ops {

    struct IPAdapterBinding {
        std::shared_ptr<models::IPAdapter> model;
        std::shared_ptr<const Tensor<float>> tokens;
        float strength;
        float start_sigma;
        float end_sigma;
    };

    struct IPAdapterBindings {
        std::vector<IPAdapterBinding> entries;
    };

    using IPAdapterCache = std::map<const models::IPAdapter*, std::map<const Tensor<float>*, ::IPAdapter::AttentionCache>>;
}

#endif  // __SD_DGL_OPS_IP_ADAPTER_H__
