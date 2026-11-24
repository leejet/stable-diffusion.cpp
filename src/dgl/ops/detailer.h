#ifndef __SD_DGL_OPS_DETAILER_H__
#define __SD_DGL_OPS_DETAILER_H__

#include "dgl/models/fwd.h"

#include "dgl/models/detail_detector.h"

namespace sd::dgl::ops {
    enum class DetectionSort {
        None,
        LeftToRight,
        CenterToEdge,
        Area,
    };

    struct DetectionFilter {
        int top_k                = 0;
        bool top_k_by_confidence = false;
        float min_area_ratio     = 0.f;
        float max_area_ratio     = 1.f;
        DetectionSort sort_by    = DetectionSort::None;
    };

    struct DetailMask {
        int width  = 0;
        int height = 0;
        std::vector<uint8_t> data;
    };

    struct DetailCrop {
        int x1, y1, x2, y2;
    };

    std::vector<int64_t> filter_detection_indices(const std::vector<models::DetailDetection>& detections, int width, int height, const DetectionFilter& filter);
    DetailMask box_mask(const models::DetailDetection& detection, int width, int height);

    bool detail_crop_box(const sd::Image& mask, float aspect, int padding, bool only_masked, DetailCrop& crop);
}

#endif
