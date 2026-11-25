#include "dgl/bindings/audio.h"

#include <algorithm>
#include <cctype>
#include "dgl/bindings/conversion.h"

namespace sd::dgl {
    OutputSignature audio_signature() {
        return {{Kind::Dict}, {{"samples", {Kind::Tensor}}, {"sample_rate", {Kind::Integer}}}};
    }

    Value audio_result(ops::AudioReadResult audio) {
        if (!audio.samples)
            audio.samples = make_tensor(SD_TENSOR_F32, std::vector<int64_t>{1, 0});
        return Value::dict({{"samples", Value::object(std::move(audio.samples))},
                            {"sample_rate", Value::literal(audio.sample_rate)}});
    }

    ops::AudioReadResult read_audio_dict(const Value& value, bool allow_empty) {
        if (value.type().kind != Kind::Dict || !value.fields.count("samples") || !value.fields.count("sample_rate") ||
            value.fields.at("samples").type().kind != Kind::Tensor || value.fields.at("sample_rate").type().kind != Kind::Integer)
            throw std::runtime_error("audio requires a dict with samples (tensor) and sample_rate (integer)");
        ops::AudioReadResult result{value.fields.at("samples").as<sd_tensor_t>(), get_int(value.fields, "sample_rate")};
        if (allow_empty && result.sample_rate == 0 && result.samples->type == SD_TENSOR_F32 &&
            result.samples->shape() == std::vector<int64_t>{1, 0})
            return result;
        audio_view(*result.samples, result.sample_rate);
        return result;
    }

    Value execute_load_audio(Values& args, RunState&) {
        return audio_result(read_wav(std::filesystem::u8path(get_string(args, "path"))));
    }
    Value execute_save_audio(Values& args, RunState&) {
        auto path      = std::filesystem::u8path(get_string(args, "path"));
        auto extension = path.extension().u8string();
        std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) { return char(std::tolower(c)); });
        if (extension != ".wav")
            throw std::runtime_error("audio output supports .wav only");
        const auto audio = read_audio_dict(args.at("audio"));
        write_wav(*audio.samples, audio.sample_rate, path, get_string(args, "format") == "pcm16");
        return Value::void_value();
    }
}
