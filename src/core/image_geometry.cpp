#include "core/image_geometry.h"

#include "core/util.h"

namespace sd {
    int64_t align_image_dimension(int64_t size, int64_t multiple) {
        if (size <= 0 || multiple <= 0)
            throw std::invalid_argument("image dimensions and alignment must be positive");
        const auto remainder = size % multiple;
        const auto rounded   = size - remainder;
        if (remainder >= multiple - remainder) {
            if (rounded > std::numeric_limits<int64_t>::max() - multiple)
                throw std::invalid_argument("aligned image dimension is too large");
            return rounded + multiple;
        }
        return std::max(multiple, rounded);
    }

    int64_t align_video_frame_count(int64_t frames, int64_t multiple, int64_t minimum) {
        if (frames <= 0 || multiple <= 0 || minimum <= 0)
            throw std::invalid_argument("video frame count and alignment must be positive");
        const auto count     = std::max(frames, minimum);
        const auto remainder = (count - minimum) % multiple;
        const auto pad       = remainder ? multiple - remainder : 0;
        if (count > std::numeric_limits<int64_t>::max() - pad)
            throw std::invalid_argument("aligned video frame count is too large");
        return count + pad;
    }

    Tensor<float> fit_image_tensor(const Tensor<float>& image, int64_t multiple, const char* operation) {
        if (image.empty() || (image.dim() != 4 && image.dim() != 5))
            throw std::invalid_argument("expected an image or video pixel tensor");
        auto shape = image.shape();
        shape[0]   = align_image_dimension(shape[0], multiple);
        shape[1]   = align_image_dimension(shape[1], multiple);
        if (shape == image.shape())
            return {};
        LOG_VERBOSE("%s: resizing pixels from %lldx%lld to %lldx%lld (alignment %lld)", operation,
                    (long long)image.shape()[0], (long long)image.shape()[1],
                    (long long)shape[0], (long long)shape[1], (long long)multiple);
        return ops::interpolate(image, std::move(shape), ops::InterpolateMode::Lanczos, false, true);
    }

    Tensor<float> pad_video_tensor(const Tensor<float>& video, int64_t frames) {
        if (video.empty() || video.dim() != 5 || frames < video.shape()[2])
            throw std::invalid_argument("video padding requires a nonempty video and cannot remove frames");
        if (frames == video.shape()[2])
            return {};
        auto shape = video.shape();
        shape[2]   = frames;
        Tensor<float> output(shape);
        ops::slice_assign(&output, 2, 0, video.shape()[2], video);
        auto last = ops::slice(video, 2, video.shape()[2] - 1, video.shape()[2]);
        for (auto i = video.shape()[2]; i < frames; ++i)
            ops::slice_assign(&output, 2, i, i + 1, last);
        return output;
    }

    std::pair<int, int> select_image_size_bucket(int source_width, int source_height, int area, int factor, int fallback_width, int fallback_height) {
        int width               = fallback_width;
        int height              = fallback_height;
        double ratio            = static_cast<double>(source_width) / source_height;
        double best_crop        = INFINITY;
        int64_t best_area_error = INT64_MAX;
        int min_size            = 4 * factor;
        for (int w = min_size; w < 2LL * area / min_size; w += factor) {
            for (int h = min_size; h < 2LL * area / w; h += factor) {
                int64_t pixels = int64_t(w) * h;
                if (pixels < 0.9 * area || pixels > 1.1 * area) {
                    continue;
                }
                double candidate_ratio = static_cast<double>(w) / h;
                double crop            = std::round((1.0 - std::min(candidate_ratio / ratio, ratio / candidate_ratio)) * 1e9);
                int64_t area_error     = std::abs(pixels - area);
                if (crop < best_crop || (crop == best_crop && (area_error < best_area_error || (area_error == best_area_error && w > width)))) {
                    width           = w;
                    height          = h;
                    best_crop       = crop;
                    best_area_error = area_error;
                }
            }
        }
        return {width, height};
    }

    Tensor<float> resize_and_center_crop(const Tensor<float>& image, int width, int height, ops::InterpolateMode filter) {
        double scale = std::max(static_cast<double>(width) / image.shape()[0], static_cast<double>(height) / image.shape()[1]);
        int w        = static_cast<int>(std::ceil(image.shape()[0] * scale));
        int h        = static_cast<int>(std::ceil(image.shape()[1] * scale));
        auto resized = ops::interpolate(image, {w, h, image.shape()[2], image.shape()[3]}, filter);
        resized      = ops::slice(resized, 0, (w - width) / 2, (w - width) / 2 + width);
        return ops::slice(resized, 1, (h - height) / 2, (h - height) / 2 + height);
    }

}
