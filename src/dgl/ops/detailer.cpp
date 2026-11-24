#include "dgl/ops/detailer.h"

#include <algorithm>
#include <cmath>

#include "dgl/models/fwd.h"

namespace sd::dgl::ops {
    static float detection_area(const models::DetailDetection& detection) {
        return std::max(0.f, detection.x2 - detection.x1) * std::max(0.f, detection.y2 - detection.y1);
    }

    std::vector<int64_t> filter_detection_indices(const std::vector<models::DetailDetection>& detections,
                                                  int width,
                                                  int height,
                                                  const DetectionFilter& filter) {
        const float image_area = static_cast<float>(width) * height;
        std::vector<int64_t> indices;
        for (size_t i = 0; i < detections.size(); ++i) {
            const float ratio = detection_area(detections[i]) / image_area;
            if (ratio >= filter.min_area_ratio && ratio <= filter.max_area_ratio)
                indices.push_back(int64_t(i));
        }
        if (filter.top_k > 0 && static_cast<int>(indices.size()) > filter.top_k) {
            std::partial_sort(indices.begin(), indices.begin() + filter.top_k, indices.end(), [&](int64_t a, int64_t b) {
                return filter.top_k_by_confidence ? detections[a].confidence > detections[b].confidence
                                                  : detection_area(detections[a]) > detection_area(detections[b]);
            });
            indices.resize(filter.top_k);
        }
        if (filter.sort_by == DetectionSort::LeftToRight) {
            std::sort(indices.begin(), indices.end(), [&](int64_t a, int64_t b) { return detections[a].x1 < detections[b].x1; });
        } else if (filter.sort_by == DetectionSort::CenterToEdge) {
            const float cx = width * 0.5f, cy = height * 0.5f;
            std::sort(indices.begin(), indices.end(), [&](int64_t a, int64_t b) {
                const auto& first  = detections[a];
                const auto& second = detections[b];
                const float ax = (first.x1 + first.x2) * 0.5f - cx, ay = (first.y1 + first.y2) * 0.5f - cy;
                const float bx = (second.x1 + second.x2) * 0.5f - cx, by = (second.y1 + second.y2) * 0.5f - cy;
                return ax * ax + ay * ay < bx * bx + by * by;
            });
        } else if (filter.sort_by == DetectionSort::Area) {
            std::sort(indices.begin(), indices.end(), [&](int64_t a, int64_t b) { return detection_area(detections[a]) > detection_area(detections[b]); });
        }
        return indices;
    }

    DetailMask box_mask(const models::DetailDetection& detection, int width, int height) {
        DetailMask mask;
        mask.width  = width;
        mask.height = height;
        mask.data.assign(static_cast<size_t>(width) * height, 0);
        int x1 = std::clamp(static_cast<int>(std::floor(detection.x1)), 0, width);
        int y1 = std::clamp(static_cast<int>(std::floor(detection.y1)), 0, height);
        int x2 = std::clamp(static_cast<int>(std::ceil(detection.x2)), 0, width);
        int y2 = std::clamp(static_cast<int>(std::ceil(detection.y2)), 0, height);
        for (int y = y1; y < y2; ++y) {
            std::fill(mask.data.begin() + static_cast<size_t>(y) * width + x1,
                      mask.data.begin() + static_cast<size_t>(y) * width + x2,
                      255);
        }
        return mask;
    }

    bool detail_crop_box(const sd::Image& mask, float aspect, int padding, bool only_masked, DetailCrop& crop) {
        const int canvas_width = int(mask.width), canvas_height = int(mask.height);
        int x1 = canvas_width, y1 = canvas_height, x2 = 0, y2 = 0;
        for (int y = 0; y < canvas_height; ++y) {
            for (int x = 0; x < canvas_width; ++x) {
                if (mask.data[size_t(y) * canvas_width + x]) {
                    x1 = std::min(x1, x);
                    y1 = std::min(y1, y);
                    x2 = std::max(x2, x + 1);
                    y2 = std::max(y2, y + 1);
                }
            }
        }
        if (x2 <= x1 || y2 <= y1)
            return false;
        if (!only_masked) {
            crop = {0, 0, canvas_width, canvas_height};
            return true;
        }
        x1        = std::max(0, x1 - padding);
        y1        = std::max(0, y1 - padding);
        x2        = std::min(canvas_width, x2 + padding);
        y2        = std::min(canvas_height, y2 + padding);
        int width = x2 - x1, height = y2 - y1;
        if (float(width) / height < aspect)
            width = std::min(canvas_width, int(std::ceil(height * aspect)));
        else
            height = std::min(canvas_height, int(std::ceil(width / aspect)));
        const int x = std::clamp((x1 + x2) / 2 - width / 2, 0, canvas_width - width);
        const int y = std::clamp((y1 + y2) / 2 - height / 2, 0, canvas_height - height);
        crop        = {x, y, x + width, y + height};
        return true;
    }

}
