#include "dgl/ops/video_timing.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace sd::dgl::ops {
    VideoFrameRate video_frame_rate(double rate) {
        if (!std::isfinite(rate) || rate <= 0 || rate > 1000000)
            return {};
        const auto nominal = int64_t(std::llround(rate * 1001 / 1000));
        const auto divisor = std::gcd(nominal * 1000, int64_t(1001));
        // Preserve common fractional clock rates despite the WebM float field.
        if (nominal > 0 && nominal * 1000 / divisor <= 1000000 &&
            static_cast<float>(double(nominal) * 1000 / 1001) == static_cast<float>(rate))
            return {nominal * 1000 / divisor, 1001 / divisor};
        int64_t n0 = 0, n1 = 1, d0 = 1, d1 = 0;
        long double value = rate;
        for (int i = 0; i < 64; ++i) {
            const auto whole = std::floor(value);
            const auto n = whole * n1 + n0, d = whole * d1 + d0;
            if (n > 1000000 || d > 1000000)
                break;
            n0 = n1;
            d0 = d1;
            n1 = int64_t(n);
            d1 = int64_t(d);
            if (n1 > 0 && static_cast<float>(double(n1) / d1) == static_cast<float>(rate))
                return {n1, d1};
            const auto fraction = value - whole;
            if (fraction <= std::numeric_limits<long double>::epsilon())
                break;
            value = 1 / fraction;
        }
        return {};
    }

    int64_t VideoSampling::time_ns(size_t frame) const {
        const uint64_t period = uint64_t(rate.den) * 1000000000;
        const auto whole = period / rate.num, remainder = period % rate.num;
        const auto limit = uint64_t(std::numeric_limits<int64_t>::max());
        if (whole && frame > limit / whole)
            return int64_t(limit);
        const auto integral   = uint64_t(frame) * whole;
        const auto fractional = (uint64_t(frame) / rate.num) * remainder + (uint64_t(frame) % rate.num) * remainder / rate.num;
        return fractional > limit - integral ? int64_t(limit) : int64_t(integral + fractional);
    }

    size_t VideoSampling::source_index(const std::vector<int64_t>& timestamps, size_t frame) const {
        return size_t(std::upper_bound(timestamps.begin(), timestamps.end(), time_ns(frame)) - timestamps.begin() - 1);
    }

    bool video_has_fixed_rate(const std::vector<int64_t>& timestamps, int64_t end_ns, VideoFrameRate rate, int64_t tolerance_ns) {
        if (rate.num < 1 || rate.num > 1000000 || rate.den < 1 || rate.den > 1000000 || timestamps.empty() ||
            tolerance_ns < 0 || end_ns < 0 || timestamps.front() < 0 || !std::is_sorted(timestamps.begin(), timestamps.end()))
            return false;
        VideoSampling sampling{rate};
        for (size_t i = 0; i < timestamps.size(); ++i)
            if (std::abs(timestamps[i] - sampling.time_ns(i)) > tolerance_ns)
                return false;
        return std::abs(end_ns - sampling.time_ns(timestamps.size())) <= tolerance_ns;
    }

    VideoSampling plan_video_sampling(const std::vector<int64_t>& timestamps, int64_t end_ns, VideoFrameRate declared_rate, int64_t tolerance_ns, int64_t fps_num, int64_t fps_den, size_t max_frames, size_t frame_bytes) {
        if (timestamps.empty() || timestamps.front() != 0 || !std::is_sorted(timestamps.begin(), timestamps.end()) ||
            end_ns < timestamps.back() || fps_num < 0 || fps_num > 1000000 || fps_den < 1 || fps_den > 1000000)
            throw std::runtime_error("invalid video timestamps or frame rate");
        const size_t capacity = frame_bytes ? (256 * 1024 * 1024) / frame_bytes : 0;
        if (!capacity)
            throw std::runtime_error("decoded video exceeds 256 MiB");
        if (!fps_num) {
            if (!video_has_fixed_rate(timestamps, end_ns, declared_rate, tolerance_ns)) {
                const auto period = timestamps.size() > 1 ? timestamps[1] : end_ns;
                declared_rate     = period > 0 ? video_frame_rate(1000000000.0 / period) : VideoFrameRate{};
            }
            if (!video_has_fixed_rate(timestamps, end_ns, declared_rate, tolerance_ns))
                throw std::runtime_error("video frame rate is variable or unknown; specify fps_num and fps_den");
            fps_num = declared_rate.num;
            fps_den = declared_rate.den;
        }
        const auto divisor = std::gcd(fps_num, fps_den);
        VideoSampling sampling{{fps_num / divisor, fps_den / divisor}};
        if (!end_ns && timestamps.size() == 1) {
            sampling.count = 1;
            return sampling;
        }
        if (end_ns <= 0)
            throw std::runtime_error("video duration must be positive");
        size_t first = 0, last = std::min(max_frames ? max_frames : capacity + 1, capacity + 1);
        while (first < last) {
            const auto middle = first + (last - first) / 2;
            if (sampling.time_ns(middle) < end_ns)
                first = middle + 1;
            else
                last = middle;
        }
        if (first > capacity)
            throw std::runtime_error("decoded video exceeds 256 MiB; set max_frames");
        sampling.count = first;
        return sampling;
    }
}
