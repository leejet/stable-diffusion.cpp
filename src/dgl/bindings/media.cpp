#include "dgl/bindings/media.h"

#include "core/file_access.h"
#include "dgl/bindings/audio.h"
#include "dgl/bindings/conversion.h"
#include "dgl/bindings/video.h"
#include "dgl/ops/image.h"
#include "dgl/ops/media_encoding.h"

namespace sd::dgl {
    OutputSignature media_signature() {
        return {{Kind::Dict}, {{"data", {Kind::Bytes}}, {"mime_type", {Kind::String}}}};
    }

    static Value media_result(std::vector<uint8_t> bytes, const char* mime_type) {
        return Value::dict({{"data", Value::object(std::make_shared<Bytes>(std::move(bytes)))},
                            {"mime_type", Value::literal(mime_type)}});
    }

    Value execute_encode_image(Values& args, RunState& state) {
        return media_result(ops::encode_image(*object<sd_tensor_t>(args, "image"), state.metadata), "image/png");
    }

    Value execute_encode_video(Values& args, RunState& state) {
        const auto video = read_video_dict(args.at("video"));
        ops::VideoSaveOptions options;
        options.fps_num  = video.fps_num;
        options.fps_den  = video.fps_den;
        options.metadata = state.metadata;
        options.pcm16    = true;
        if (args.count("audio"))
            options.audio = read_audio_dict(args.at("audio"), true);
        if (get_string(args, "format") == "avi")
            return media_result(ops::encode_avi(*video.frames, options), "video/x-msvideo");
        return media_result(ops::encode_webm(*video.frames, options), "video/webm");
    }

    Value execute_save_media(Values& args, RunState&) {
        const auto& media = args.at("media");
        const auto data = media.fields.find("data"), mime = media.fields.find("mime_type");
        if (data == media.fields.end() || data->second.type().kind != Kind::Bytes ||
            mime == media.fields.end() || mime->second.type().kind != Kind::String)
            throw std::runtime_error("media::save requires a dict with data (bytes) and mime_type (string)");
        const auto path = get_string(args, "path");
        require_file_access(path, SD_DGL_FILE_WRITE);
        ops::save_media(path, data->second.as<Bytes>()->data, mime->second.scalar.get<std::string>());
        return Value::void_value();
    }
}
