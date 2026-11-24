#include "dgl/ops/video_windows.h"

#include <algorithm>
#include <cstring>
#include "dgl/ops/tensor.h"
#include "dgl/ops/video.h"

namespace sd::dgl::ops {
    std::vector<std::shared_ptr<sd_tensor_t>> split_video_frames(const std::shared_ptr<sd_tensor_t>& frames, size_t size, size_t overlap) {
        if (!frames)
            throw std::runtime_error("video frame tensor is null");
        validate_video_frames(*frames);
        if (!size || overlap >= size)
            throw std::runtime_error("video overlap must be smaller than a positive window size");
        const auto count = size_t(frames->shape()[3]);
        if (size >= count)
            return {frames};
        std::vector<std::shared_ptr<sd_tensor_t>> windows;
        for (size_t start = 0; start < count;) {
            const auto end = std::min(start + size, count);
            windows.push_back(crop_tensor(*frames, {3}, {int64_t(start)}, {int64_t(end - start)}));
            if (end == count)
                break;
            start = end - overlap;
        }
        return windows;
    }

    std::shared_ptr<sd_tensor_t> join_video_windows(const std::vector<std::shared_ptr<sd_tensor_t>>& windows, size_t overlap) {
        if (windows.empty())
            throw std::runtime_error("video::join_windows requires at least one window");
        constexpr uint64_t byte_limit = uint64_t(4) * 1024 * 1024 * 1024;
        size_t count = 0, stride = 0;
        std::vector<int64_t> shape;
        for (size_t i = 0; i < windows.size(); ++i) {
            if (!windows[i])
                throw std::runtime_error("video frame tensor is null");
            validate_video_frames(*windows[i]);
            const auto& current = windows[i]->shape();
            if (!i) {
                shape  = current;
                stride = size_t(current[0] * current[1] * current[2]);
            } else if (!std::equal(shape.begin(), shape.begin() + 3, current.begin()) || size_t(current[3]) <= overlap)
                throw std::runtime_error("video windows have incompatible sizes or overlap");
            const auto retained = size_t(current[3]) - (i ? overlap : 0);
            const auto limit    = byte_limit / stride;
            if (retained > limit || count > limit - retained)
                throw std::runtime_error("joined video exceeds 4 GiB");
            count += retained;
        }
        if (windows.size() == 1)
            return windows.front();
        shape[3]          = int64_t(count);
        auto output       = make_tensor(SD_TENSOR_U8, std::move(shape));
        auto* destination = static_cast<uint8_t*>(output->mutable_data());
        for (size_t i = 0; i < windows.size(); ++i) {
            const auto skip  = (i ? overlap : 0) * stride;
            const auto bytes = windows[i]->bytes - skip;
            std::memcpy(destination, static_cast<const uint8_t*>(windows[i]->data()) + skip, bytes);
            destination += bytes;
        }
        return output;
    }
}
