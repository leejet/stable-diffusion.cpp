#include "dgl/ops/video_webp.h"

#include <cstring>
#include <fstream>
#include <limits>

#include "dgl/ops/tensor.h"
#include "dgl/ops/video_timing.h"

#ifdef SD_USE_WEBP
#include <webp/demux.h>

namespace sd::dgl::ops {
    VideoReadResult load_webp(const std::filesystem::path& path, size_t max_frames, int64_t fps_num, int64_t fps_den) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file || file.tellg() <= 0)
            throw std::runtime_error("cannot read WebP input");
        const auto length = uint64_t(file.tellg());
        if (length > std::numeric_limits<size_t>::max() || length > uint64_t(std::numeric_limits<std::streamsize>::max()))
            throw std::runtime_error("WebP input is too large");
        std::vector<uint8_t> bytes(size_t(length), 0);
        file.seekg(0);
        if (!file.read(reinterpret_cast<char*>(bytes.data()), std::streamsize(bytes.size())))
            throw std::runtime_error("truncated WebP input");
        WebPBitstreamFeatures features{};
        if (WebPGetFeatures(bytes.data(), bytes.size(), &features) != VP8_STATUS_OK ||
            features.width < 1 || features.width > 16384 || features.height < 1 || features.height > 16384)
            throw std::runtime_error("invalid WebP dimensions or bitstream");
        if (size_t(features.width) * features.height * 3 > 256 * 1024 * 1024)
            throw std::runtime_error("decoded video exceeds 256 MiB");
        WebPData data{bytes.data(), bytes.size()};
        WebPAnimDecoderOptions options;
        if (!WebPAnimDecoderOptionsInit(&options))
            throw std::runtime_error("cannot initialize WebP animation decoder");
        options.color_mode = MODE_RGBA;
        std::unique_ptr<WebPAnimDecoder, decltype(&WebPAnimDecoderDelete)> decoder(WebPAnimDecoderNew(&data, &options), WebPAnimDecoderDelete);
        WebPAnimInfo info{};
        if (!decoder || !WebPAnimDecoderGetInfo(decoder.get(), &info) || !info.frame_count ||
            info.frame_count > uint32_t(std::numeric_limits<int>::max()) ||
            info.canvas_width < 1 || info.canvas_width > 16384 || info.canvas_height < 1 || info.canvas_height > 16384)
            throw std::runtime_error("invalid WebP animation or canvas");
        const auto* demux = WebPAnimDecoderGetDemuxer(decoder.get());
        std::vector<int64_t> timestamps;
        timestamps.reserve(info.frame_count);
        int64_t end_ms = 0;
        for (uint32_t i = 0; i < info.frame_count; ++i) {
            WebPIterator frame{};
            if (!WebPDemuxGetFrame(demux, int(i + 1), &frame))
                throw std::runtime_error("cannot read WebP frame timing");
            const int duration  = frame.duration;
            const bool complete = frame.complete;
            WebPDemuxReleaseIterator(&frame);
            if (!complete || duration < 0 || end_ms > std::numeric_limits<int>::max() - duration)
                throw std::runtime_error("incomplete WebP frame or animation duration exceeds decoder limits");
            timestamps.push_back(end_ms * 1000000);
            end_ms += duration;
        }
        const size_t frame_bytes = size_t(info.canvas_width) * info.canvas_height * 3;
        const auto sampling      = plan_video_sampling(timestamps, end_ms * 1000000, {}, 0, fps_num, fps_den, max_frames, frame_bytes);
        VideoReadResult result;
        result.frames  = make_tensor(SD_TENSOR_U8, std::vector<int64_t>{3, info.canvas_width, info.canvas_height, int64_t(sampling.count)});
        result.fps_num = sampling.rate.num;
        result.fps_den = sampling.rate.den;
        auto* output   = static_cast<uint8_t*>(result.frames->mutable_data());
        uint8_t* rgba  = nullptr;
        size_t decoded = 0, previous = info.frame_count;
        for (size_t i = 0; i < sampling.count; ++i) {
            const auto source = sampling.source_index(timestamps, i);
            auto* destination = output + i * frame_bytes;
            if (source == previous) {
                std::memcpy(destination, destination - frame_bytes, frame_bytes);
                continue;
            }
            while (decoded <= source) {
                int timestamp = 0;
                if (!WebPAnimDecoderGetNext(decoder.get(), &rgba, &timestamp))
                    throw std::runtime_error("WebP frame decoding failed");
                const auto expected = decoded + 1 < timestamps.size() ? timestamps[decoded + 1] / 1000000 : end_ms;
                if (timestamp != expected)
                    throw std::runtime_error("WebP decoder timestamps do not match frame metadata");
                ++decoded;
            }
            for (size_t pixel = 0; pixel < frame_bytes / 3; ++pixel)
                for (size_t channel = 0; channel < 3; ++channel)
                    destination[pixel * 3 + channel] = uint8_t((uint32_t(rgba[pixel * 4 + channel]) * rgba[pixel * 4 + 3] + 127) / 255);
            previous = source;
        }
        return result;
    }
}
#else
namespace sd::dgl::ops {
    VideoReadResult load_webp(const std::filesystem::path&, size_t, int64_t, int64_t) {
        throw std::runtime_error("WebP video input requires a build with SD_WEBP=ON");
    }
}
#endif
