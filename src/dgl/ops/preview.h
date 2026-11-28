#ifndef __SD_DGL_OPS_PREVIEW_H__
#define __SD_DGL_OPS_PREVIEW_H__

#include "dgl/models/fwd.h"

#include "dgl/ops/sampling_observer.h"

namespace sd::dgl::ops {

    class PreviewObserver final : public SamplingObserver {
        std::shared_ptr<models::VAE> decoder_;
        bool circular_x_;
        bool circular_y_;

    public:
        explicit PreviewObserver(std::shared_ptr<models::VAE> decoder = {}, bool circular_x = false, bool circular_y = false);
        std::unique_ptr<SamplingObserverSession> start(const SamplingRun& run) const override;
    };
}

#endif  // __SD_DGL_OPS_PREVIEW_H__
