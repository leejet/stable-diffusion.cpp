#include "api/result.h"

#include <cstring>

#include "dgl/diagnostic.h"
#include "dgl/value.h"

namespace sd::api {
    static const dgl::Value& field(const dgl::Value& value, const char* name) {
        if (value.type().kind != dgl::Kind::Dict || !value.fields.count(name))
            throw dgl::Error(SD_STATUS_TYPE_MISMATCH, "generation result is missing '" + std::string(name) + "'");
        return value.fields.at(name);
    }

    static void append_frames(sd_result_t& result, const sd_tensor_t& frames, bool video) {
        const auto& shape = frames.shape();
        if (frames.type != SD_TENSOR_U8 || (shape.size() != 3 && shape.size() != 4) ||
            shape[0] < 1 || shape[0] > 4 || shape[1] <= 0 || shape[2] <= 0 || (shape.size() == 4 && shape[3] <= 0))
            throw dgl::Error(SD_STATUS_TYPE_MISMATCH, "generation pixels must be a packed U8 [C,W,H] image or [C,W,H,T] frames");
        const auto count = shape.size() == 4 ? shape[3] : 1;
        if (!video && count != 1)
            throw dgl::Error(SD_STATUS_TYPE_MISMATCH, "image program returned multiple frames");
        const auto bytes   = frames.bytes / count;
        const auto* pixels = static_cast<const uint8_t*>(frames.data());
        for (int64_t i = 0; i < count; ++i) {
            auto frame = allocate_tensor(SD_TENSOR_U8, {shape[0], shape[1], shape[2]});
            std::memcpy(frame.get()->mutable_data(), pixels + i * bytes, bytes);
            result.images.push_back(std::move(frame));
        }
    }

    void append_program_image(sd_result_t& result, const dgl::Value& value, const std::string& source) {
        const auto& image = field(value, "image");
        const auto first  = result.images.size();
        if (image.type() == dgl::Type{dgl::Kind::List, dgl::Kind::Tensor}) {
            for (const auto& item : image.items)
                append_frames(result, *item.as<sd_tensor_t>(), false);
        } else {
            append_frames(result, *image.as<sd_tensor_t>(), false);
        }
        result.kind = SD_RESULT_IMAGE;
        for (size_t i = first; i < result.images.size(); ++i)
            result.generation_infos.push_back(source);
    }

    void set_program_video(sd_result_t& result, const dgl::Value& value, const std::string& source) {
        const auto& video = field(value, "video");
        append_frames(result, *field(video, "frames").as<sd_tensor_t>(), true);
        const auto numerator   = field(video, "fps_num").scalar.get<int64_t>();
        const auto denominator = field(video, "fps_den").scalar.get<int64_t>();
        if (numerator <= 0 || denominator <= 0 || numerator % denominator || numerator / denominator > INT32_MAX)
            throw dgl::Error(SD_STATUS_OUT_OF_RANGE, "C video result requires a positive integral frame rate");
        result.kind             = SD_RESULT_VIDEO;
        result.fps              = int(numerator / denominator);
        result.generation_infos = {source};
        const auto audio        = value.fields.find("audio");
        if (audio == value.fields.end() || !audio->second.fields.count("samples"))
            return;
        const auto samples = field(audio->second, "samples").as<sd_tensor_t>();
        if (!samples->bytes)
            return;
        const auto rate = field(audio->second, "sample_rate").scalar.get<int64_t>();
        if (samples->type != SD_TENSOR_F32 || samples->shape().size() != 2 || samples->shape()[0] <= 0 || samples->shape()[0] > 32 || rate < 1000 || rate > 384000)
            throw dgl::Error(SD_STATUS_TYPE_MISMATCH, "generation audio must contain F32 [channels,samples] and a valid sample rate");
        result.audio_tensor = TensorRef(samples.get());
        result.audio        = {uint32_t(rate), result.audio_tensor.get()};
    }
}
