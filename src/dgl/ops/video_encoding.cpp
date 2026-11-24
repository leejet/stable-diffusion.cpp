#include "dgl/ops/media_encoding.h"

#include <cstring>
#include <memory>
#include <stdexcept>

#include "dgl/ops/video_timing.h"

#ifdef SD_USE_WEBM
#include <mkvmuxer/mkvmuxer.h>
#include <webp/encode.h>
#endif

namespace sd::dgl::ops {
#ifdef SD_USE_WEBM
    namespace media_encoding_detail {
        class MemoryWebMWriter : public mkvmuxer::IMkvWriter {
            std::vector<uint8_t> bytes_;
            size_t position_ = 0;

        public:
            mkvmuxer::int32 Write(const void* data, mkvmuxer::uint32 size) override {
                if ((!data && size) || size > size_t(INT64_MAX) - position_)
                    return -1;
                const auto end = position_ + size;
                if (end > bytes_.size())
                    bytes_.resize(end);
                if (size)
                    std::memcpy(bytes_.data() + position_, data, size);
                position_ = end;
                return 0;
            }
            mkvmuxer::int64 Position() const override { return int64_t(position_); }
            mkvmuxer::int32 Position(mkvmuxer::int64 value) override {
                if (value < 0 || uint64_t(value) > bytes_.size())
                    return -1;
                position_ = size_t(value);
                return 0;
            }
            bool Seekable() const override { return true; }
            void ElementStartNotify(mkvmuxer::uint64, mkvmuxer::int64) override {}
            std::vector<uint8_t> take_bytes() { return std::move(bytes_); }
        };

        static std::vector<uint8_t> vp8_frame(const sd::Image& image, int quality) {
            uint8_t* data   = nullptr;
            const auto size = WebPEncodeRGB(image.data, image.width, image.height, image.width * 3, float(quality), &data);
            std::unique_ptr<uint8_t, decltype(&WebPFree)> encoded(data, WebPFree);
            if (!size)
                throw std::runtime_error("cannot encode WebM VP8 frame");
            for (size_t offset = 12; offset <= size && size - offset >= 8;) {
                const auto* chunk     = data + offset;
                const uint32_t length = uint32_t(chunk[4]) | (uint32_t(chunk[5]) << 8) | (uint32_t(chunk[6]) << 16) | (uint32_t(chunk[7]) << 24);
                if (length > size - offset - 8)
                    break;
                if (!std::memcmp(chunk, "VP8 ", 4))
                    return {chunk + 8, chunk + 8 + length};
                offset += 8 + size_t(length) + (length & 1);
            }
            throw std::runtime_error("WebP encoder did not produce a VP8 frame");
        }

    }
#endif

    std::vector<uint8_t> encode_webm(const sd_tensor_t& frames, const VideoSaveOptions& options) {
#ifndef SD_USE_WEBM
        throw std::runtime_error("video::encode WebM requires a build with SD_WEBM=ON");
#else
        using namespace media_encoding_detail;
        validate_video_frames(frames);
        if (options.fps_num < 1 || options.fps_num > 1000000 || options.fps_den < 1 || options.fps_den > 1000000)
            throw std::runtime_error("WebM frame rate numerator and denominator must be in [1,1000000]");
        if (options.audio.samples && options.audio.samples->bytes)
            throw std::runtime_error("video::encode WebM does not support audio; use format=\"avi\" for PCM16 audio");
        MemoryWebMWriter writer;
        mkvmuxer::Segment segment;
        if (!segment.Init(&writer))
            throw std::runtime_error("cannot initialize WebM output");
        segment.set_mode(mkvmuxer::Segment::kFile);
        segment.OutputCues(true);
        segment.GetSegmentInfo()->set_timecode_scale(1000);
        segment.GetSegmentInfo()->set_writing_app("stable-diffusion.cpp");
        segment.GetSegmentInfo()->set_muxing_app("stable-diffusion.cpp");
        const auto first = video_frame_view(frames, 0);
        const auto track = segment.AddVideoTrack(first.width, first.height, 0);
        auto* video      = static_cast<mkvmuxer::VideoTrack*>(segment.GetTrackByNumber(track));
        if (!track || !video || !segment.CuesTrack(track))
            throw std::runtime_error("cannot create WebM VP8 track");
        video->set_codec_id("V_VP8");
        video->set_frame_rate(double(options.fps_num) / options.fps_den);
        const VideoSampling clock{{options.fps_num, options.fps_den}, size_t(frames.shape()[3])};
        video->set_default_duration(uint64_t(clock.time_ns(1)));
        if (!options.metadata.empty()) {
            auto* tag = segment.AddTag();
            if (!tag || !tag->add_simple_tag("COMMENT", options.metadata.c_str()))
                throw std::runtime_error("cannot write WebM generation metadata");
        }
        for (size_t frame = 0; frame < clock.count; ++frame) {
            auto packet = vp8_frame(video_frame_view(frames, frame), options.quality);
            if (!segment.AddFrame(packet.data(), packet.size(), track, uint64_t(clock.time_ns(frame)), true))
                throw std::runtime_error("cannot mux WebM video frame");
        }
        const auto duration = uint64_t(clock.time_ns(clock.count));
        segment.set_duration(double(duration) / segment.GetSegmentInfo()->timecode_scale());
        if (!segment.Finalize())
            throw std::runtime_error("cannot finalize WebM output");
        return writer.take_bytes();
#endif
    }
}
