#include "dgl/bindings/video.h"

#include "dgl/bindings/audio.h"
#include "dgl/bindings/conversion.h"

namespace sd::dgl {
    OutputSignature video_signature() {
        return {{Kind::Dict}, {{"frames", {Kind::Tensor}}, {"fps_num", {Kind::Integer}}, {"fps_den", {Kind::Integer}}}};
    }

    Value video_result(std::shared_ptr<sd_tensor_t> frames, int64_t fps_num, int64_t fps_den) {
        return Value::dict({{"frames", Value::object(std::move(frames))},
                            {"fps_num", Value::literal(fps_num)},
                            {"fps_den", Value::literal(fps_den)}});
    }

    ops::VideoReadResult read_video_dict(const Value& value) {
        if (value.type().kind != Kind::Dict || !value.fields.count("frames") || !value.fields.count("fps_num") || !value.fields.count("fps_den") ||
            value.fields.at("frames").type().kind != Kind::Tensor || value.fields.at("fps_num").type().kind != Kind::Integer ||
            value.fields.at("fps_den").type().kind != Kind::Integer)
            throw std::runtime_error("video requires a dict with frames (tensor), fps_num and fps_den (integers)");
        ops::VideoReadResult result;
        result.frames  = value.fields.at("frames").as<sd_tensor_t>();
        result.fps_num = get_int(value.fields, "fps_num");
        result.fps_den = get_int(value.fields, "fps_den");
        if (result.fps_num < 1 || result.fps_num > 1000000 || result.fps_den < 1 || result.fps_den > 1000000)
            throw std::runtime_error("video frame rate numerator and denominator must be in [1,1000000]");
        return result;
    }

    Value execute_load_video(Values& args, RunState&) {
        const auto source = get_string(args, "path");
        require_file_access(source, SD_DGL_FILE_READ);
        const auto path       = std::filesystem::u8path(source);
        const auto max_frames = size_t(get_int(args, "max_frames"));
        const auto fps_num = get_int(args, "fps_num"), fps_den = get_int(args, "fps_den");
        auto result = std::filesystem::is_directory(path) ? ops::load_frame_directory(path, max_frames, fps_num, fps_den)
                                                          : ops::load_video(path, max_frames, fps_num, fps_den, get_bool(args, "audio"));
        return Value::dict({{"video", video_result(std::move(result.frames), result.fps_num, result.fps_den)},
                            {"audio", audio_result(std::move(result.audio))}});
    }

    Value execute_save_video(Values& args, RunState& state) {
        ops::VideoSaveOptions options;
        options.metadata = state.metadata;
        const auto video = read_video_dict(args.at("video"));
        options.fps_num  = video.fps_num;
        options.fps_den  = video.fps_den;
        if (args.count("audio"))
            options.audio = read_audio_dict(args.at("audio"), true);
        options.trim_audio = get_bool(args, "trim_audio");
        options.jpeg       = get_string(args, "codec") != "rgb";
        options.quality    = int(get_int(args, "quality"));
        options.pcm16      = get_string(args, "audio_format") == "pcm16";
        ops::save_video(*video.frames, std::filesystem::u8path(get_string(args, "path")), options);
        return Value::void_value();
    }
}
