#include "value.h"

#include <cmath>
#include <stdexcept>

namespace sd::dgl {

    bool accepts(Type expected, Type actual) {
        if (expected.kind == Kind::Void || actual.kind == Kind::Void)
            return expected.kind == actual.kind;
        if (expected.kind == Kind::Any || actual.kind == Kind::Any)
            return true;
        if (expected.kind == Kind::Number && actual.kind == Kind::Integer)
            return true;
        if (expected.kind == Kind::Model)
            return is_model_kind(actual.kind);
        if (expected.kind == Kind::IntVector)
            return actual.kind == Kind::Tensor || (actual.kind == Kind::List && accepts({Kind::Integer}, {actual.element}));
        return expected.kind == actual.kind &&
               (expected.kind != Kind::List || accepts({expected.element}, {actual.element}));
    }

    bool accepts_list_element(Kind element, Kind actual) {
        return actual != Kind::Any && actual != Kind::List && actual != Kind::Void && accepts({element}, {actual});
    }

    bool is_model_kind(Kind kind) {
        switch (kind) {
            case Kind::Denoiser:
            case Kind::TextEncoder:
            case Kind::VAE:
            case Kind::ControlNet:
            case Kind::VisionEncoder:
            case Kind::AudioEncoder:
            case Kind::AudioVAE:
            case Kind::IPAdapter:
            case Kind::PhotoMaker:
            case Kind::PuLID:
            case Kind::AnimateDiff:
            case Kind::LatentUpscaler:
            case Kind::Upscaler:
            case Kind::Detector:
                return true;
            default:
                return false;
        }
    }

    Value Value::void_value() {
        Value value;
        value.type_.kind = Kind::Void;
        return value;
    }

    Value Value::literal(const nlohmann::json& scalar) {
        Value value;
        value.scalar = scalar;
        if (scalar.is_boolean())
            value.type_.kind = Kind::Boolean;
        else if (scalar.is_number_integer()) {
            if (scalar.is_number_unsigned() && scalar.get<uint64_t>() > uint64_t(INT64_MAX))
                throw std::runtime_error("integer exceeds int64 range");
            value.scalar     = scalar.get<int64_t>();
            value.type_.kind = Kind::Integer;
        } else if (scalar.is_number_float()) {
            if (!std::isfinite(scalar.get<double>()))
                throw std::runtime_error("number must be finite");
            value.type_.kind = Kind::Number;
        } else if (scalar.is_string())
            value.type_.kind = Kind::String;
        else
            throw std::runtime_error("expected a scalar value");
        return value;
    }

    Value Value::dict(Values fields) {
        for (const auto& [name, field] : fields)
            if (field.type().kind == Kind::Void)
                throw std::runtime_error("dict field '" + name + "' has no value");
        Value value;
        value.type_.kind = Kind::Dict;
        value.fields     = std::move(fields);
        return value;
    }

    Value Value::list(std::vector<Value> values, Kind declared_element) {
        Value value;
        value.type_   = {Kind::List, declared_element};
        auto& element = value.type_.element;
        if (element == Kind::Any && !values.empty()) {
            element = values.front().type_.kind;
            for (const auto& item : values)
                if ((element == Kind::Integer && item.type_.kind == Kind::Number) || (element == Kind::Number && item.type_.kind == Kind::Integer))
                    element = Kind::Number;
        }
        for (const auto& item : values) {
            if (!accepts_list_element(element, item.type_.kind))
                throw std::runtime_error("lists must contain values of one non-list type");
        }
        value.items = std::move(values);
        return value;
    }

    Value::Cached::Cached(Value value, bool retain)
        : value_(std::move(value)), object_(value_.object_) {
        if (!retain)
            value_.object_.reset();
    }

    std::optional<Value> Value::Cached::lock() const {
        if (auto object = object_.lock()) {
            auto value    = value_;
            value.object_ = std::move(object);
            return value;
        }
        return {};
    }

    std::string type_name(Type type) {
        if (type.kind == Kind::Bytes)
            return "bytes";
        static const char* names[] = {"any", "bool", "int", "number", "string", "upscaler", "detector", "list", "tensor", "dict", "denoiser", "text_encoder", "vae", "predictor", "control_net", "control_bindings", "ip_adapter_bindings", "vision_encoder", "audio_encoder", "audio_vae", "ip_adapter", "photomaker", "pulid", "pulid_bindings", "animatediff", "latent_upscaler", "rng"};
        if (type.kind == Kind::List)
            return "list<" + type_name({type.element}) + ">";
        if (type.kind == Kind::IntVector)
            return "i64 tensor or list<int>";
        if (type.kind == Kind::Model)
            return "model";
        if (type.kind == Kind::Lora)
            return "lora";
        if (type.kind == Kind::Function)
            return "fn";
        if (type.kind == Kind::SamplingObserver)
            return "sampling_observer";
        if (type.kind == Kind::Void)
            return "void";
        return names[static_cast<size_t>(type.kind)];
    }

    int64_t get_int(const Values& args, const std::string& name) {
        return args.at(name).scalar.get<int64_t>();
    }
    double get_number(const Values& args, const std::string& name) {
        return args.at(name).scalar.get<double>();
    }
    std::string get_string(const Values& args, const std::string& name) {
        return args.at(name).scalar.get<std::string>();
    }
    bool get_bool(const Values& args, const std::string& name) {
        return args.at(name).scalar.get<bool>();
    }

}  // namespace sd::dgl
