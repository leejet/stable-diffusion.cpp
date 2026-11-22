#include "dgl/ops/video_webm.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>

#include "dgl/ops/tensor.h"
#include "dgl/ops/video_timing.h"

#ifdef SD_USE_WEBM
#include <mkvparser/mkvparser.h>
#include <webp/decode.h>

namespace sd::dgl::ops {
    class WebMReader : public mkvparser::IMkvReader {
        std::ifstream file_;
        long long size_;

    public:
        explicit WebMReader(const std::filesystem::path& path)
            : file_(path, std::ios::binary | std::ios::ate) {
            if (!file_ || file_.tellg() <= 0)
                throw std::runtime_error("cannot read WebM input");
            size_ = static_cast<long long>(file_.tellg());
        }
        int Read(long long position, long length, unsigned char* buffer) override {
            if (position < 0 || length < 0 || position > size_ || length > size_ - position)
                return -1;
            file_.clear();
            file_.seekg(position);
            return !length || file_.read(reinterpret_cast<char*>(buffer), length) ? 0 : -1;
        }
        int Length(long long* total, long long* available) override {
            *total = *available = size_;
            return 0;
        }
        std::vector<uint8_t> packet(long long position, size_t length) {
            if (length > 256 * 1024 * 1024 || length > size_t(std::numeric_limits<long>::max()))
                throw std::runtime_error("WebM packet exceeds 256 MiB");
            std::vector<uint8_t> bytes(length);
            if (Read(position, long(length), bytes.data()))
                throw std::runtime_error("truncated WebM packet");
            return bytes;
        }
    };

    struct WebMPacket {
        long long position;
        long size;
    };

    VideoReadResult load_webm(const std::filesystem::path& path, size_t max_frames, int64_t fps_num, int64_t fps_den, bool include_audio) {
        WebMReader reader(path);
        mkvparser::EBMLHeader header;
        long long position = 0;
        if (header.Parse(&reader, position) || !header.m_docType ||
            (std::strcmp(header.m_docType, "webm") && std::strcmp(header.m_docType, "matroska")))
            throw std::runtime_error("invalid WebM header");
        mkvparser::Segment* raw = nullptr;
        const auto status       = mkvparser::Segment::CreateInstance(&reader, position, raw);
        std::unique_ptr<mkvparser::Segment> segment(raw);
        if (status || !segment || segment->Load() < 0 || !segment->GetTracks() || !segment->GetInfo())
            throw std::runtime_error("cannot parse WebM segment");
        const auto scale = segment->GetInfo()->GetTimeCodeScale();
        if (scale < 1)
            throw std::runtime_error("invalid WebM time scale");
        const mkvparser::VideoTrack* video = nullptr;
        const mkvparser::AudioTrack* audio = nullptr;
        const auto* tracks                 = segment->GetTracks();
        for (unsigned long i = 0; i < tracks->GetTracksCount(); ++i) {
            const auto* track = tracks->GetTrackByIndex(i);
            if (track && track->GetType() == mkvparser::Track::kVideo && !video)
                video = static_cast<const mkvparser::VideoTrack*>(track);
            if (track && track->GetType() == mkvparser::Track::kAudio && !audio && include_audio)
                audio = static_cast<const mkvparser::AudioTrack*>(track);
        }
        if (!video || !video->GetCodecId() || std::strcmp(video->GetCodecId(), "V_VP8"))
            throw std::runtime_error("WebM input supports independent VP8 keyframes only");
        const auto width = video->GetWidth(), height = video->GetHeight();
        if (width < 1 || width > 16384 || height < 1 || height > 16384)
            throw std::runtime_error("invalid WebM video dimensions");
        const size_t frame_bytes = size_t(width) * size_t(height) * 3;
        int channels = 0, rate = 0;
        if (audio) {
            const auto sample_rate = audio->GetSamplingRate();
            if (!audio->GetCodecId() || std::strcmp(audio->GetCodecId(), "A_PCM/INT/LIT") || audio->GetBitDepth() != 16 ||
                audio->GetChannels() < 1 || audio->GetChannels() > 32 || !std::isfinite(sample_rate) ||
                sample_rate < 1000 || sample_rate > 384000 || sample_rate != std::floor(sample_rate) || audio->GetCodecDelay())
                throw std::runtime_error("WebM audio requires PCM16 without codec delay; use video::load(audio=false) to skip unsupported audio");
            channels = int(audio->GetChannels());
            rate     = int(sample_rate);
        }
        std::vector<WebMPacket> frames, packets;
        std::vector<int64_t> timestamps;
        int64_t last_duration = 0;
        for (auto* cluster = segment->GetFirst(); cluster && !cluster->EOS(); cluster = segment->GetNext(cluster)) {
            const mkvparser::BlockEntry* entry = nullptr;
            if (cluster->GetFirst(entry) < 0)
                throw std::runtime_error("invalid WebM cluster");
            while (entry && !entry->EOS()) {
                const auto* block = entry->GetBlock();
                const auto track  = block->GetTrackNumber();
                if (track == video->GetNumber() || (audio && track == audio->GetNumber())) {
                    const bool visual = track == video->GetNumber();
                    if (block->GetFrameCount() != 1 || (visual && (!block->IsKey() || block->IsInvisible())))
                        throw std::runtime_error(visual ? "WebM input requires one independent VP8 keyframe per block" : "laced WebM audio is unsupported; use video::load(audio=false)");
                    const auto& frame    = block->GetFrame(0);
                    const auto timestamp = block->GetTime(cluster);
                    if (frame.pos < 0 || frame.len <= 0 || timestamp < 0)
                        throw std::runtime_error("invalid WebM packet");
                    if (visual) {
                        frames.push_back({frame.pos, frame.len});
                        timestamps.push_back(timestamp);
                        last_duration = 0;
                        if (entry->GetKind() == mkvparser::BlockEntry::kBlockGroup) {
                            const auto duration = static_cast<const mkvparser::BlockGroup*>(entry)->GetDurationTimeCode();
                            if (duration > std::numeric_limits<int64_t>::max() / scale)
                                throw std::runtime_error("WebM frame duration exceeds int64 range");
                            if (duration > 0)
                                last_duration = duration * scale;
                        }
                    } else {
                        packets.push_back({frame.pos, frame.len});
                    }
                }
                if (cluster->GetNext(entry, entry) < 0)
                    throw std::runtime_error("invalid WebM block sequence");
            }
        }
        if (frames.empty())
            throw std::runtime_error("WebM contains no video frames");
        auto declared = video_frame_rate(video->GetFrameRate());
        if (!declared.num && video->GetDefaultDuration())
            declared = video_frame_rate(1000000000.0 / video->GetDefaultDuration());
        int64_t end = segment->GetDuration();
        if (last_duration > 0) {
            if (timestamps.back() > std::numeric_limits<int64_t>::max() - last_duration)
                throw std::runtime_error("WebM duration exceeds int64 range");
            end = timestamps.back() + last_duration;
        } else if (declared.num) {
            const VideoSampling timing{declared};
            end = timing.time_ns(frames.size());
            if (!video_has_fixed_rate(timestamps, end, declared, scale)) {
                if (timestamps.back() > std::numeric_limits<int64_t>::max() - timing.time_ns(1))
                    throw std::runtime_error("WebM duration exceeds int64 range");
                end = timestamps.back() + timing.time_ns(1);
            }
        }
        const auto sampling = plan_video_sampling(timestamps, end, declared, scale, fps_num, fps_den, max_frames, frame_bytes);
        VideoReadResult result;
        result.frames   = make_tensor(SD_TENSOR_U8, std::vector<int64_t>{3, width, height, int64_t(sampling.count)});
        result.fps_num  = sampling.rate.num;
        result.fps_den  = sampling.rate.den;
        auto* output    = static_cast<uint8_t*>(result.frames->mutable_data());
        size_t previous = frames.size();
        for (size_t i = 0; i < sampling.count; ++i) {
            const auto source = sampling.source_index(timestamps, i);
            auto* destination = output + i * frame_bytes;
            if (source == previous) {
                std::memcpy(destination, destination - frame_bytes, frame_bytes);
                continue;
            }
            auto bytes       = reader.packet(frames[source].position, size_t(frames[source].size));
            int actual_width = 0, actual_height = 0;
            if (bytes.size() < 10 || (bytes[0] & 1) || bytes[3] != 0x9d || bytes[4] != 0x01 || bytes[5] != 0x2a ||
                !WebPGetInfo(bytes.data(), bytes.size(), &actual_width, &actual_height) || actual_width != width || actual_height != height ||
                !WebPDecodeRGBInto(bytes.data(), bytes.size(), destination, frame_bytes, int(width * 3)))
                throw std::runtime_error("invalid or dependent VP8 frame in WebM input");
            previous = source;
        }
        if (audio && !packets.empty()) {
            const size_t block_bytes = size_t(channels) * 2;
            uint64_t total           = 0;
            for (const auto& packet : packets) {
                if (size_t(packet.size) % block_bytes)
                    throw std::runtime_error("invalid WebM PCM16 packet size; use video::load(audio=false)");
                total += uint64_t(packet.size) / block_bytes;
            }
            const auto requested = std::ceil(static_cast<long double>(sampling.count) * sampling.rate.den * rate / sampling.rate.num);
            const auto count     = std::min(static_cast<long double>(total), requested);
            if (count > (256 * 1024 * 1024) / (size_t(channels) * sizeof(float)))
                throw std::runtime_error("decoded video audio exceeds 256 MiB; set max_frames");
            const size_t wanted = size_t(count) * block_bytes;
            std::vector<uint8_t> data;
            data.reserve(wanted);
            for (const auto& packet : packets) {
                if (data.size() == wanted)
                    break;
                const auto bytes = reader.packet(packet.position, std::min(size_t(packet.size), wanted - data.size()));
                data.insert(data.end(), bytes.begin(), bytes.end());
            }
            if (!data.empty())
                result.audio = decode_pcm_audio(data.data(), data.size(), 1, 16, channels, rate);
        }
        return result;
    }
}
#else
namespace sd::dgl::ops {
    VideoReadResult load_webm(const std::filesystem::path&, size_t, int64_t, int64_t, bool) {
        throw std::runtime_error("WebM video input requires a build with SD_WEBM=ON");
    }
}
#endif
