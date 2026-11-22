#include "dgl/ops/video.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>

#include "api/metadata.h"
#include "core/file_access.h"
#include "dgl/ops/audio.h"
#include "dgl/ops/image.h"
#include "dgl/ops/video_webm.h"
#include "dgl/ops/video_webp.h"

namespace sd::dgl::ops {

    void validate_video_frames(const sd_tensor_t& frames) {
        const auto& shape = frames.shape();
        if (frames.type != SD_TENSOR_U8 || shape.size() != 4 || shape[0] != 3 ||
            shape[1] < 1 || shape[1] > 16384 || shape[2] < 1 || shape[2] > 16384 || shape[3] < 1)
            throw std::runtime_error("video frames must be U8 [3, width, height, frames]");
    }

    sd::Image video_frame_view(const sd_tensor_t& frames, size_t index) {
        validate_video_frames(frames);
        const auto& shape = frames.shape();
        if (index >= size_t(shape[3]))
            throw std::runtime_error("video frame index is out of range");
        const auto* data = static_cast<const uint8_t*>(frames.data()) + index * size_t(shape[1] * shape[2] * 3);
        return {uint32_t(shape[1]), uint32_t(shape[2]), 3, const_cast<uint8_t*>(data)};
    }

    VideoReadResult load_video(const std::filesystem::path& path, size_t max_frames, int64_t fps_num, int64_t fps_den, bool include_audio) {
        std::ifstream file(path, std::ios::binary);
        uint8_t header[12]{};
        if (!file || !file.read(reinterpret_cast<char*>(header), sizeof(header)))
            throw std::runtime_error("cannot read video input header");
        if (!std::memcmp(header, "RIFF", 4)) {
            if (!std::memcmp(header + 8, "AVI ", 4))
                return load_avi(path, max_frames, fps_num, fps_den, include_audio);
            if (!std::memcmp(header + 8, "WEBP", 4))
                return load_webp(path, max_frames, fps_num, fps_den);
        }
        if (header[0] == 0x1a && header[1] == 0x45 && header[2] == 0xdf && header[3] == 0xa3)
            return load_webm(path, max_frames, fps_num, fps_den, include_audio);
        throw std::runtime_error("unsupported video input; expected AVI, WebP or independent-frame VP8 WebM");
    }

    static uint32_t fourcc(const char* text) {
        return uint32_t(uint8_t(text[0])) | (uint32_t(uint8_t(text[1])) << 8) |
               (uint32_t(uint8_t(text[2])) << 16) | (uint32_t(uint8_t(text[3])) << 24);
    }

    static uint32_t read_u32(const uint8_t* p) {
        return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
    }

    struct AVIChunk {
        uint32_t id;
        uint64_t data, size, next;
    };

    class AVIReader {
        std::ifstream file_;
        uint64_t size_;
        size_t chunks_ = 0;

    public:
        explicit AVIReader(const std::filesystem::path& path)
            : file_(path, std::ios::binary | std::ios::ate) {
            if (!file_ || file_.tellg() < 12)
                throw std::runtime_error("cannot read AVI file");
            size_ = uint64_t(file_.tellg());
        }

        uint64_t size() const { return size_; }

        std::vector<uint8_t> read(uint64_t offset, uint64_t size) {
            if (offset > size_ || size > size_ - offset || size > 256 * 1024 * 1024)
                throw std::runtime_error("invalid AVI data size");
            std::vector<uint8_t> bytes(size_t(size), 0);
            file_.seekg(std::streamoff(offset));
            if (size && !file_.read(reinterpret_cast<char*>(bytes.data()), std::streamsize(size)))
                throw std::runtime_error("truncated AVI file");
            return bytes;
        }

        AVIChunk chunk(uint64_t offset, uint64_t end) {
            if (offset > end || end - offset < 8 || ++chunks_ > 1000000)
                throw std::runtime_error("invalid AVI chunk structure or too many chunks");
            auto bytes      = read(offset, 8);
            uint64_t size   = read_u32(bytes.data() + 4);
            uint64_t padded = size + (size & 1);
            if (padded > end - offset - 8)
                throw std::runtime_error("AVI chunk exceeds its container");
            return {read_u32(bytes.data()), offset + 8, size, offset + 8 + padded};
        }
    };

    struct AVIStream {
        int index = -1;
        int width = 0, height = 0;
        int64_t fps_num = 0, fps_den = 0;
        bool jpeg = false, top_down = false;
    };

    struct AVIAudioStream {
        int index    = -1;
        int encoding = 0, bits = 0, channels = 0;
        uint32_t rate = 0, block = 0;
    };

    static uint16_t read_u16(const uint8_t* data) {
        return uint16_t(data[0]) | (uint16_t(data[1]) << 8);
    }

    static void read_avi_stream(AVIReader& reader, uint64_t start, uint64_t end, int index, AVIStream& stream, AVIAudioStream* audio) {
        std::vector<uint8_t> header, format;
        for (uint64_t pos = start; pos < end;) {
            auto chunk = reader.chunk(pos, end);
            if (chunk.id == fourcc("strh")) {
                if (chunk.size < 56 || !header.empty())
                    throw std::runtime_error("invalid AVI stream header");
                header = reader.read(chunk.data, 56);
            } else if (chunk.id == fourcc("strf")) {
                if (chunk.size >= 16)
                    format = reader.read(chunk.data, std::min<uint64_t>(chunk.size, 40));
            }
            pos = chunk.next;
        }
        if (!header.empty() && read_u32(header.data()) == fourcc("auds") && audio && audio->index < 0) {
            if (format.size() < 16 || read_u32(header.data() + 28) != 0)
                throw std::runtime_error("invalid AVI audio header or nonzero stream start");
            auto p          = format.data();
            audio->encoding = read_u16(p);
            audio->channels = read_u16(p + 2);
            audio->rate     = read_u32(p + 4);
            audio->block    = read_u16(p + 12);
            audio->bits     = read_u16(p + 14);
            if (audio->channels < 1 || audio->channels > 32 || audio->rate < 1000 || audio->rate > 384000 ||
                !((audio->encoding == 1 && (audio->bits == 8 || audio->bits == 16 || audio->bits == 24 || audio->bits == 32)) || (audio->encoding == 3 && audio->bits == 32)) || audio->block != uint32_t(audio->channels * (audio->bits / 8)))
                throw std::runtime_error("AVI audio requires PCM or float32; use video::load(audio=false) to skip unsupported audio");
            audio->index = index;
        }
        if (header.empty() || read_u32(header.data()) != fourcc("vids") || stream.index >= 0)
            return;
        if (format.size() < 40 || read_u32(format.data()) < 40)
            throw std::runtime_error("missing AVI bitmap header");
        auto p                 = format.data();
        int64_t width          = read_u32(p + 4);
        uint32_t signed_height = read_u32(p + 8);
        bool top_down          = (signed_height & 0x80000000u) != 0;
        int64_t height         = top_down ? int64_t(0x100000000ull - signed_height) : signed_height;
        uint32_t compression   = read_u32(p + 16);
        bool jpeg              = compression == fourcc("MJPG") || compression == fourcc("mjpg");
        if (width < 1 || width > 16384 || height < 1 || height > 16384 || width * height > 64 * 1024 * 1024 ||
            p[12] != 1 || p[13] != 0 || p[14] != 24 || p[15] != 0 || (!jpeg && compression != 0))
            throw std::runtime_error("AVI requires 24-bit RGB or MJPEG frames with supported dimensions");
        int64_t num = read_u32(header.data() + 24), den = read_u32(header.data() + 20);
        if (!num || !den)
            throw std::runtime_error("AVI has no usable frame rate");
        int64_t divisor = std::gcd(num, den);
        num /= divisor;
        den /= divisor;
        if (num > 1000000 || den > 1000000 || read_u32(header.data() + 28) != 0)
            throw std::runtime_error("unsupported AVI frame rate or nonzero stream start");
        stream = {index, int(width), int(height), num, den, jpeg, top_down};
    }

    static void read_avi_header(AVIReader& reader, uint64_t start, uint64_t end, AVIStream& stream, AVIAudioStream* audio) {
        int index = 0;
        for (uint64_t pos = start; pos < end;) {
            auto chunk = reader.chunk(pos, end);
            if (chunk.id == fourcc("LIST")) {
                if (chunk.size < 4)
                    throw std::runtime_error("invalid AVI list");
                auto type = reader.read(chunk.data, 4);
                if (read_u32(type.data()) == fourcc("strl")) {
                    if (index >= 100)
                        throw std::runtime_error("too many AVI streams");
                    read_avi_stream(reader, chunk.data + 4, chunk.data + chunk.size, index++, stream, audio);
                }
            }
            pos = chunk.next;
        }
    }

    static void read_avi_frames(AVIReader& reader, uint64_t start, uint64_t end, int stream, std::vector<AVIChunk>& frames, int depth = 0, bool audio = false) {
        if (depth > 8)
            throw std::runtime_error("AVI lists are nested too deeply");
        char prefix[]       = {char('0' + stream / 10), char('0' + stream % 10), audio ? 'w' : 'd', 'b'};
        uint32_t raw        = fourcc(prefix);
        prefix[3]           = audio ? 'b' : 'c';
        uint32_t compressed = fourcc(prefix);
        for (uint64_t pos = start; pos < end;) {
            auto chunk = reader.chunk(pos, end);
            if (chunk.id == raw || chunk.id == compressed) {
                if (chunk.size == 0 && !audio)
                    throw std::runtime_error("AVI dropped frames are unsupported");
                if (chunk.size)
                    frames.push_back(chunk);
            } else if (chunk.id == fourcc("LIST")) {
                if (chunk.size < 4)
                    throw std::runtime_error("invalid AVI frame list");
                auto type = reader.read(chunk.data, 4);
                if (read_u32(type.data()) == fourcc("rec "))
                    read_avi_frames(reader, chunk.data + 4, chunk.data + chunk.size, stream, frames, depth + 1, audio);
            }
            pos = chunk.next;
        }
    }

    VideoReadResult load_avi(const std::filesystem::path& path, size_t max_frames, int64_t fps_num, int64_t fps_den, bool include_audio) {
        if (!std::filesystem::is_regular_file(path))
            throw std::runtime_error("video path is not a regular file");
        AVIReader reader(path);
        auto root = reader.read(0, 12);
        if (read_u32(root.data()) != fourcc("RIFF") || read_u32(root.data() + 8) != fourcc("AVI "))
            throw std::runtime_error("video input supports AVI (RGB or MJPEG) only");
        uint64_t end = uint64_t(read_u32(root.data() + 4)) + 8;
        if (end != reader.size() || end < 12)
            throw std::runtime_error("invalid AVI size; segmented OpenDML AVI is unsupported");
        AVIStream stream;
        AVIAudioStream audio;
        std::vector<AVIChunk> movies;
        bool header_found = false;
        for (uint64_t pos = 12; pos < end;) {
            auto chunk = reader.chunk(pos, end);
            if (chunk.id == fourcc("LIST")) {
                if (chunk.size < 4)
                    throw std::runtime_error("invalid AVI list");
                auto type = reader.read(chunk.data, 4);
                if (read_u32(type.data()) == fourcc("hdrl")) {
                    if (header_found)
                        throw std::runtime_error("duplicate AVI header");
                    read_avi_header(reader, chunk.data + 4, chunk.data + chunk.size, stream, include_audio ? &audio : nullptr);
                    header_found = true;
                } else if (read_u32(type.data()) == fourcc("movi")) {
                    movies.push_back(chunk);
                }
            }
            pos = chunk.next;
        }
        if (stream.index < 0)
            throw std::runtime_error("AVI has no video stream");
        std::vector<AVIChunk> frames;
        for (const auto& movie : movies)
            read_avi_frames(reader, movie.data + 4, movie.data + movie.size, stream.index, frames);
        if (frames.empty())
            throw std::runtime_error("AVI contains no video frames");
        if (!fps_num) {
            fps_num = stream.fps_num;
            fps_den = stream.fps_den;
        }
        int64_t divisor = std::gcd(fps_num, fps_den);
        fps_num /= divisor;
        fps_den /= divisor;
        uint64_t step_num = uint64_t(stream.fps_num) * fps_den;
        uint64_t step_den = uint64_t(stream.fps_den) * fps_num;
        uint64_t count    = (frames.size() * step_den + step_num - 1) / step_num;
        if (max_frames)
            count = std::min(count, uint64_t(max_frames));
        size_t frame_size = size_t(stream.width) * stream.height * 3;
        if (count > (256 * 1024 * 1024) / frame_size)
            throw std::runtime_error("decoded video exceeds 256 MiB; set max_frames");
        VideoReadResult video;
        video.frames  = make_tensor(SD_TENSOR_U8, std::vector<int64_t>{3, stream.width, stream.height, int64_t(count)});
        video.fps_num = fps_num;
        video.fps_den = fps_den;
        size_t last   = frames.size();
        std::shared_ptr<sd_tensor_t> image;
        for (uint64_t i = 0; i < count; ++i) {
            size_t source = size_t(i * step_num / step_den);
            if (source != last) {
                const auto& chunk = frames.at(source);
                size_t stride     = (size_t(stream.width) * 3 + 3) & ~size_t(3);
                if (!stream.jpeg && chunk.size != stride * stream.height)
                    throw std::runtime_error("invalid RGB AVI frame size");
                auto bytes = reader.read(chunk.data, chunk.size);
                if (stream.jpeg) {
                    image = decode_video_jpeg(bytes, stream.width, stream.height);
                } else {
                    image = allocate_image(stream.width, stream.height, 3);
                    for (int y = 0; y < stream.height; ++y) {
                        auto src = bytes.data() + stride * (stream.top_down ? y : stream.height - 1 - y);
                        auto dst = static_cast<uint8_t*>(image->mutable_data()) + size_t(y) * stream.width * 3;
                        for (int x = 0; x < stream.width; ++x) {
                            dst[x * 3]     = src[x * 3 + 2];
                            dst[x * 3 + 1] = src[x * 3 + 1];
                            dst[x * 3 + 2] = src[x * 3];
                        }
                    }
                }
                last = source;
            }
            std::memcpy(static_cast<uint8_t*>(video.frames->mutable_data()) + size_t(i) * frame_size, image->data(), frame_size);
        }
        if (audio.index >= 0) {
            std::vector<AVIChunk> packets;
            for (const auto& movie : movies)
                read_avi_frames(reader, movie.data + 4, movie.data + movie.size, audio.index, packets, 0, true);
            uint64_t wanted = (count * uint64_t(video.fps_den) * audio.rate + video.fps_num - 1) / video.fps_num;

            std::vector<uint8_t> data;
            for (const auto& packet : packets) {
                if (packet.size % audio.block)
                    throw std::runtime_error("incomplete AVI audio packet");
                uint64_t limit = wanted * audio.block;
                if (data.size() >= limit)
                    break;
                uint64_t size      = std::min<uint64_t>(packet.size, limit - data.size());
                uint64_t max_bytes = (256 * 1024 * 1024 / (audio.channels * sizeof(float))) * audio.block;
                if (data.size() + size > max_bytes)
                    throw std::runtime_error("decoded video audio exceeds 256 MiB; set max_frames");
                auto bytes = reader.read(packet.data, size);
                data.insert(data.end(), bytes.begin(), bytes.end());
            }
            if (!data.empty())
                video.audio = decode_pcm_audio(data.data(), data.size(), audio.encoding, audio.bits, audio.channels, audio.rate);
        }
        return video;
    }

    class AVIWriter {
        std::ofstream file_;
        std::vector<uint8_t> bytes_;
        uint64_t position_ = 0;
        bool memory_;

    public:
        explicit AVIWriter(const std::filesystem::path* path) : memory_(!path) {
            if (path)
                file_.open(*path, std::ios::binary | std::ios::trunc);
            if (!memory_ && !file_)
                throw std::runtime_error("cannot open AVI output");
        }
        uint64_t position() {
            if (memory_)
                return position_;
            auto pos = file_.tellp();
            if (pos < 0)
                throw std::runtime_error("cannot get AVI output position");
            return uint64_t(pos);
        }
        void write(const void* bytes, size_t size) {
            if (size > UINT32_MAX - position())
                throw std::runtime_error("AVI output exceeds 4 GiB");
            if (memory_) {
                const auto end = size_t(position_) + size;
                if (end > bytes_.size())
                    bytes_.resize(end);
                if (size)
                    std::memcpy(bytes_.data() + position_, bytes, size);
                position_ = end;
            } else if (!file_.write(static_cast<const char*>(bytes), std::streamsize(size)))
                throw std::runtime_error("cannot write AVI output");
        }
        void seek(uint64_t position) {
            if (memory_)
                position_ = position;
            else
                file_.seekp(std::streamoff(position));
        }
        void u32(uint32_t value) {
            uint8_t bytes[] = {uint8_t(value), uint8_t(value >> 8), uint8_t(value >> 16), uint8_t(value >> 24)};
            write(bytes, 4);
        }
        void id(const char* value) { write(value, 4); }
        uint64_t begin(const char* type) {
            id(type);
            uint64_t start = position();
            u32(0);
            return start;
        }
        void end(uint64_t start) {
            uint64_t next = position();
            uint32_t size = uint32_t(next - start - 4);
            seek(start);
            u32(size);
            seek(next);
            if (size & 1) {
                const char padding = 0;
                write(&padding, 1);
            }
        }
        void close() {
            if (memory_)
                return;
            file_.close();
            if (!file_)
                throw std::runtime_error("cannot close AVI output");
        }
        std::vector<uint8_t> take_bytes() { return std::move(bytes_); }
    };

    static std::vector<uint8_t> write_avi(const sd_tensor_t& frames, const std::filesystem::path* path, const VideoSaveOptions& options) {
        validate_video_frames(frames);
        if (options.fps_num < 1 || options.fps_num > UINT32_MAX || options.fps_den < 1 || options.fps_den > UINT32_MAX)
            throw std::runtime_error("video frame rate numerator and denominator must fit positive AVI 32-bit fields");
        const auto divisor     = std::gcd(options.fps_num, options.fps_den);
        const auto fps_num     = options.fps_num / divisor;
        const auto fps_den     = options.fps_den / divisor;
        const auto frame_count = size_t(frames.shape()[3]);
        const bool jpeg = options.jpeg, pcm16 = options.pcm16;
        const int quality = options.quality;
        const auto first  = video_frame_view(frames, 0);
        uint32_t width = first.width, height = first.height, stride = (width * 3 + 3) & ~uint32_t(3), frame_size = stride * height;
        uint64_t microseconds = (1000000ull * fps_den + fps_num / 2) / fps_num;
        if (microseconds > UINT32_MAX)
            throw std::runtime_error("frame duration exceeds AVI limits");
        sd::Audio audio{};
        if (options.audio.samples && options.audio.samples->bytes)
            audio = audio_view(*options.audio.samples, options.audio.sample_rate);
        if (options.trim_audio && audio.data) {
            const auto duration  = uint64_t(frame_count) * uint64_t(fps_den);
            const auto track_end = audio.sample_count * uint64_t(fps_num);
            // Compare before multiplying the video duration by the sample rate.
            if (duration <= (track_end - 1) / audio.sample_rate)
                audio.sample_count = duration * audio.sample_rate / uint64_t(fps_num);
            if (!audio.sample_count)
                audio.data = nullptr;
        }
        uint32_t block = audio.channels * (pcm16 ? 2 : 4);
        AVIWriter writer(path);
        auto riff = writer.begin("RIFF");
        writer.id("AVI ");
        auto header = writer.begin("LIST");
        writer.id("hdrl");
        auto main_header = writer.begin("avih");
        for (uint32_t value : {uint32_t(microseconds), 0u, 0u, audio.data ? 0x110u : 0x10u, uint32_t(frame_count), 0u, audio.data ? 2u : 1u, frame_size, width, height, 0u, 0u, 0u, 0u})
            writer.u32(value);
        writer.end(main_header);
        auto stream = writer.begin("LIST");
        writer.id("strl");
        auto stream_header = writer.begin("strh");
        writer.id("vids");
        writer.id(jpeg ? "MJPG" : "DIB ");
        for (uint32_t value : {0u, 0u, 0u, uint32_t(fps_den), uint32_t(fps_num), 0u, uint32_t(frame_count), frame_size, UINT32_MAX, 0u, 0u, width | (height << 16)})
            writer.u32(value);
        writer.end(stream_header);
        auto format = writer.begin("strf");
        for (uint32_t value : {40u, width, height, 1u | (24u << 16), jpeg ? fourcc("MJPG") : 0u, frame_size, 0u, 0u, 0u, 0u})
            writer.u32(value);
        writer.end(format);
        writer.end(stream);
        if (audio.data) {
            auto track = writer.begin("LIST");
            writer.id("strl");
            auto stream = writer.begin("strh");
            writer.id("auds");
            writer.u32(0);
            for (uint32_t value : {0u, 0u, 0u, block, audio.sample_rate * block, 0u, uint32_t(audio.sample_count), uint32_t(std::min<uint64_t>(audio.sample_count * block, 65536)), UINT32_MAX, block, 0u, 0u})
                writer.u32(value);
            writer.end(stream);
            auto format = writer.begin("strf");
            writer.u32((pcm16 ? 1u : 3u) | (audio.channels << 16));
            writer.u32(audio.sample_rate);
            writer.u32(audio.sample_rate * block);
            writer.u32(block | ((pcm16 ? 16u : 32u) << 16));
            writer.end(format);
            writer.end(track);
        }
        writer.end(header);
        const auto metadata = sd::metadata::avi_info(options.metadata);
        writer.write(metadata.data(), metadata.size());
        auto movie = writer.begin("LIST");
        writer.id("movi");
        struct Index {
            uint32_t id, offset, size;
        };
        std::vector<Index> index;
        auto packet = [&](const char* id, const std::vector<uint8_t>& bytes) {
            index.push_back({fourcc(id), uint32_t(writer.position() - movie - 4), uint32_t(bytes.size())});
            auto start = writer.begin(id);
            writer.write(bytes.data(), bytes.size());
            writer.end(start);
        };
        std::vector<uint8_t> bytes;
        if (!jpeg)
            bytes.resize(frame_size, 0);
        uint64_t sample = 0;
        for (size_t i = 0; i < frame_count; ++i) {
            const auto image = video_frame_view(frames, i);
            if (jpeg)
                bytes = encode_video_jpeg(image, quality);
            else {
                for (uint32_t y = 0; y < height; ++y) {
                    auto src = image.data + size_t(y) * width * 3;
                    auto dst = bytes.data() + size_t(height - 1 - y) * stride;
                    for (uint32_t x = 0; x < width; ++x) {
                        dst[x * 3]     = src[x * 3 + 2];
                        dst[x * 3 + 1] = src[x * 3 + 1];
                        dst[x * 3 + 2] = src[x * 3];
                    }
                }
            }
            packet(jpeg ? "00dc" : "00db", bytes);
            if (audio.data) {
                uint64_t end = std::min<uint64_t>(audio.sample_count, ((i + 1) * uint64_t(fps_den) * audio.sample_rate + fps_num - 1) / fps_num);
                if (end > sample)
                    packet("01wb", encode_pcm_audio(audio, pcm16, sample, end - sample));
                sample = end;
            }
        }
        if (audio.data && sample < audio.sample_count)
            packet("01wb", encode_pcm_audio(audio, pcm16, sample, audio.sample_count - sample));
        writer.end(movie);
        auto idx = writer.begin("idx1");
        for (const auto& item : index) {
            writer.u32(item.id);
            writer.u32(0x10);
            writer.u32(item.offset);
            writer.u32(item.size);
        }
        writer.end(idx);
        writer.end(riff);
        writer.close();
        return writer.take_bytes();
    }

    std::vector<uint8_t> encode_avi(const sd_tensor_t& frames, const VideoSaveOptions& options) {
        return write_avi(frames, nullptr, options);
    }

    void save_video(const sd_tensor_t& frames, const std::filesystem::path& path, const VideoSaveOptions& options) {
        auto extension = path.extension().u8string();
        std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) { return char(std::tolower(c)); });
        if (extension != ".avi")
            throw std::runtime_error("video output supports .avi (RGB or MJPEG) only");
        require_file_access(path.u8string(), SD_DGL_FILE_WRITE);
        write_avi(frames, &path, options);
    }

    VideoReadResult load_frame_directory(const std::filesystem::path& path, size_t max_frames, int64_t fps_num, int64_t fps_den) {
        auto images       = read_image_directory(path, max_frames, 3);
        const auto& shape = images.front()->shape();
        VideoReadResult result;
        result.fps_num     = fps_num ? fps_num : 24;
        result.fps_den     = fps_den;
        const auto divisor = std::gcd(result.fps_num, result.fps_den);
        result.fps_num /= divisor;
        result.fps_den /= divisor;
        const auto stride = images.front()->bytes;
        for (const auto& image : images)
            if (image->shape() != shape)
                throw std::runtime_error("video frame dimensions must match");
        if (images.size() > (256 * 1024 * 1024) / stride)
            throw std::runtime_error("video frames exceed 256 MiB");
        result.frames = make_tensor(SD_TENSOR_U8, std::vector<int64_t>{3, shape[1], shape[2], int64_t(images.size())});
        for (size_t i = 0; i < images.size(); ++i) {
            std::memcpy(static_cast<uint8_t*>(result.frames->mutable_data()) + i * stride, images[i]->data(), stride);
        }
        validate_video_frames(*result.frames);
        return result;
    }

}  // namespace sd::dgl
