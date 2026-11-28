#ifndef __SD_DGL_OPS_AUDIO_H__
#define __SD_DGL_OPS_AUDIO_H__

#include <filesystem>
#include "core/media.h"
#include "dgl/tensor.h"

namespace sd::dgl::ops {
    struct AudioReadResult {
        std::shared_ptr<sd_tensor_t> samples;
        int64_t sample_rate = 0;
    };

    sd::Audio audio_view(const sd_tensor_t& samples, int64_t sample_rate);
    AudioReadResult copy_audio(const sd::Audio& audio);
    AudioReadResult read_wav(const std::filesystem::path& path);
    void write_wav(const sd_tensor_t& samples, int64_t sample_rate, const std::filesystem::path& path, bool pcm16);
    AudioReadResult decode_pcm_audio(const uint8_t* data, size_t bytes, int encoding, int bits, int channels, int64_t rate);
    std::vector<uint8_t> encode_pcm_audio(const sd::Audio& audio, bool pcm16, uint64_t first, uint64_t count);
}

#endif
