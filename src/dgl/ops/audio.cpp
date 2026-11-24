#include "dgl/ops/audio.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>

#include "core/file_access.h"

namespace sd::dgl::ops {
    sd::Audio audio_view(const sd_tensor_t& tensor, int64_t sample_rate) {
        if (tensor.type != SD_TENSOR_F32 || tensor.shape().size() != 2 ||
            tensor.shape()[0] < 1 || tensor.shape()[0] > 32 || tensor.shape()[1] < 1 || sample_rate < 1000 || sample_rate > 384000)
            throw std::runtime_error("audio requires F32 [channels, samples] and a sample rate in [1000,384000]");
        const auto* samples = static_cast<const float*>(tensor.data());
        if (!std::all_of(samples, samples + tensor.bytes / sizeof(float), [](float x) { return std::isfinite(x); }))
            throw std::runtime_error("audio samples must be finite");
        sd::Audio view{};
        view.data         = const_cast<float*>(samples);
        view.channels     = uint32_t(tensor.shape()[0]);
        view.sample_count = uint64_t(tensor.shape()[1]);
        view.sample_rate  = uint32_t(sample_rate);
        return view;
    }

    AudioReadResult copy_audio(const sd::Audio& view) {
        if (!view.data || !view.channels || view.channels > 32 || !view.sample_count || view.sample_count > 256 * 1024 * 1024 / (view.channels * sizeof(float)))
            throw std::runtime_error("invalid audio or audio exceeds 256 MiB");
        AudioReadResult audio;
        audio.sample_rate = view.sample_rate;
        audio.samples     = make_tensor(SD_TENSOR_F32, std::vector<int64_t>{view.channels, int64_t(view.sample_count)}, view.data, size_t(view.sample_count) * view.channels * sizeof(float));
        audio_view(*audio.samples, audio.sample_rate);
        return audio;
    }

    static uint32_t audio_u32(const uint8_t* data) {
        return uint32_t(data[0]) | (uint32_t(data[1]) << 8) | (uint32_t(data[2]) << 16) | (uint32_t(data[3]) << 24);
    }
    static uint16_t audio_u16(const uint8_t* data) {
        return uint16_t(data[0]) | (uint16_t(data[1]) << 8);
    }

    AudioReadResult read_wav(const std::filesystem::path& path) {
        require_file_access(path.u8string(), SD_DGL_FILE_READ);
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file || file.tellg() < 12 || uint64_t(file.tellg()) > 256 * 1024 * 1024)
            throw std::runtime_error("cannot read WAV file or file exceeds 256 MiB");
        std::vector<uint8_t> bytes(size_t(file.tellg()));
        file.seekg(0);
        if (!file.read(reinterpret_cast<char*>(bytes.data()), std::streamsize(bytes.size())))
            throw std::runtime_error("cannot read WAV bytes");
        if (std::memcmp(bytes.data(), "RIFF", 4) || std::memcmp(bytes.data() + 8, "WAVE", 4) || uint64_t(audio_u32(bytes.data() + 4)) + 8 != bytes.size())
            throw std::runtime_error("audio input requires a RIFF WAV file");
        const uint8_t* format = nullptr;
        const uint8_t* data   = nullptr;
        size_t count          = 0;
        for (size_t offset = 12; offset < bytes.size();) {
            if (bytes.size() - offset < 8)
                throw std::runtime_error("truncated WAV chunk");
            uint64_t size = audio_u32(bytes.data() + offset + 4), padded = size + (size & 1);
            if (padded > bytes.size() - offset - 8)
                throw std::runtime_error("WAV chunk exceeds its container");
            if (!std::memcmp(bytes.data() + offset, "fmt ", 4)) {
                if (format || size < 16)
                    throw std::runtime_error("invalid WAV format chunk");
                format = bytes.data() + offset + 8;
            } else if (!std::memcmp(bytes.data() + offset, "data", 4)) {
                if (data)
                    throw std::runtime_error("multiple WAV data chunks are unsupported");
                data  = bytes.data() + offset + 8;
                count = size_t(size);
            }
            offset += 8 + size_t(padded);
        }
        if (!format || !data || !count)
            throw std::runtime_error("WAV has no format or samples");
        int encoding = audio_u16(format), channels = audio_u16(format + 2), bits = audio_u16(format + 14), block = audio_u16(format + 12);
        uint32_t rate = audio_u32(format + 4);
        if (channels < 1 || channels > 32 || rate < 1000 || rate > 384000 ||
            !((encoding == 1 && (bits == 8 || bits == 16 || bits == 24 || bits == 32)) || (encoding == 3 && bits == 32)) || block != channels * (bits / 8) || count % block)
            throw std::runtime_error("WAV requires interleaved PCM8/16/24/32 or float32 audio");
        return decode_pcm_audio(data, count, encoding, bits, channels, rate);
    }

    AudioReadResult decode_pcm_audio(const uint8_t* data, size_t bytes, int encoding, int bits, int channels, int64_t rate) {
        if (!data || !bytes || channels < 1 || channels > 32 || rate < 1000 || rate > 384000 ||
            !((encoding == 1 && (bits == 8 || bits == 16 || bits == 24 || bits == 32)) || (encoding == 3 && bits == 32)) || bytes % (channels * (bits / 8)))
            throw std::runtime_error("invalid PCM/float audio format or samples");
        size_t samples = bytes / size_t(bits / 8);
        if (samples > 256 * 1024 * 1024 / sizeof(float))
            throw std::runtime_error("decoded WAV exceeds 256 MiB");
        std::vector<float> decoded(samples);
        for (size_t i = 0; i < samples; ++i) {
            const uint8_t* src = data + i * size_t(bits / 8);
            if (encoding == 3) {
                uint32_t word = audio_u32(src);
                std::memcpy(&decoded[i], &word, 4);
            } else if (bits == 8)
                decoded[i] = (int(src[0]) - 128) / 128.f;
            else {
                uint32_t word = 0;
                for (int j = 0; j < bits / 8; ++j)
                    word |= uint32_t(src[j]) << (j * 8);
                int64_t signed_value = word;
                if (word & (uint32_t(1) << (bits - 1)))
                    signed_value -= int64_t(1) << bits;
                decoded[i] = float(double(signed_value) / double(int64_t(1) << (bits - 1)));
            }
        }
        sd::Audio view{};
        view.data         = decoded.data();
        view.channels     = channels;
        view.sample_count = samples / channels;
        view.sample_rate  = rate;
        return copy_audio(view);
    }

    std::vector<uint8_t> encode_pcm_audio(const sd::Audio& view, bool pcm16, uint64_t first, uint64_t count) {
        if (first > view.sample_count || count > view.sample_count - first)
            throw std::runtime_error("audio packet exceeds samples");
        std::vector<uint8_t> bytes(size_t(count) * view.channels * (pcm16 ? 2 : 4));
        for (uint64_t i = 0; i < count * view.channels; ++i) {
            float value   = view.data[first * view.channels + i];
            uint32_t word = 0;
            if (pcm16)
                word = uint16_t(std::clamp(int(std::lround(std::clamp(value, -1.f, 1.f) * 32768)), -32768, 32767));
            else
                std::memcpy(&word, &value, 4);
            int width = pcm16 ? 2 : 4;
            for (int j = 0; j < width; ++j)
                bytes[size_t(i) * width + j] = uint8_t(word >> (8 * j));
        }
        return bytes;
    }

    void write_wav(const sd_tensor_t& samples, int64_t sample_rate, const std::filesystem::path& path, bool pcm16) {
        require_file_access(path.u8string(), SD_DGL_FILE_WRITE);
        auto view          = audio_view(samples, sample_rate);
        uint64_t data_size = view.sample_count * view.channels * (pcm16 ? 2 : 4);
        if (data_size > UINT32_MAX - 48)
            throw std::runtime_error("WAV exceeds RIFF size limit");
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        if (!file)
            throw std::runtime_error("cannot open WAV output");
        auto u32 = [&](uint32_t value) {uint8_t data[]={uint8_t(value),uint8_t(value>>8),uint8_t(value>>16),uint8_t(value>>24)};file.write(reinterpret_cast<char*>(data),4); };
        auto u16 = [&](uint16_t value) {uint8_t data[]={uint8_t(value),uint8_t(value>>8)};file.write(reinterpret_cast<char*>(data),2); };
        file.write("RIFF", 4);
        u32(uint32_t(data_size) + (pcm16 ? 36 : 48));
        file.write("WAVEfmt ", 8);
        u32(16);
        u16(pcm16 ? 1 : 3);
        u16(uint16_t(view.channels));
        u32(view.sample_rate);
        u32(view.sample_rate * view.channels * (pcm16 ? 2 : 4));
        u16(uint16_t(view.channels * (pcm16 ? 2 : 4)));
        u16(pcm16 ? 16 : 32);
        if (!pcm16) {
            file.write("fact", 4);
            u32(4);
            u32(uint32_t(view.sample_count));
        }
        file.write("data", 4);
        u32(uint32_t(data_size));
        for (uint64_t i = 0; i < view.sample_count * view.channels; ++i) {
            if (pcm16) {
                int value = int(std::lround(std::clamp(view.data[i], -1.f, 1.f) * 32768));
                u16(uint16_t(std::clamp(value, -32768, 32767)));
            } else {
                uint32_t word;
                std::memcpy(&word, &view.data[i], 4);
                u32(word);
            }
        }
        file.close();
        if (!file)
            throw std::runtime_error("cannot write WAV output");
    }

}
