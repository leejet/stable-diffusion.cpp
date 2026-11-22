#ifndef __SD_DGL_VALUE_H__
#define __SD_DGL_VALUE_H__

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include "bytes.h"
#include "diagnostic.h"
#include "json.hpp"
#include "models/fwd.h"
#include "ops/fwd.h"

class RNG;
struct sd_tensor_t;

namespace sd::dgl {

    enum class Kind { Any,
                      Boolean,
                      Integer,
                      Number,
                      String,
                      Upscaler,
                      Detector,
                      List,
                      Tensor,
                      Dict,
                      Denoiser,
                      TextEncoder,
                      VAE,
                      Predictor,
                      ControlNet,
                      ControlBindings,
                      IPAdapterBindings,
                      VisionEncoder,
                      AudioEncoder,
                      AudioVAE,
                      IPAdapter,
                      PhotoMaker,
                      PuLID,
                      PuLIDBindings,
                      AnimateDiff,
                      LatentUpscaler,
                      Rng,
                      IntVector,
                      Model,
                      Lora,
                      Function,
                      SamplingObserver,
                      Void,
                      Bytes };

    struct Type {
        Kind kind    = Kind::Any;
        Kind element = Kind::Any;
        bool operator==(const Type& other) const { return kind == other.kind && element == other.element; }
    };

    struct ModelSource {
        std::string key;
        std::string revision;
    };

    std::string type_name(Type type);
    bool accepts(Type expected, Type actual);
    bool accepts_list_element(Kind element, Kind actual);

    template <class T>
    struct ObjectKind;

    template <>
    struct ObjectKind<RNG> : std::integral_constant<Kind, Kind::Rng> {};
    template <>
    struct ObjectKind<sd_tensor_t> : std::integral_constant<Kind, Kind::Tensor> {};
    template <>
    struct ObjectKind<models::AudioEncoder> : std::integral_constant<Kind, Kind::AudioEncoder> {};
    template <>
    struct ObjectKind<models::AudioVAE> : std::integral_constant<Kind, Kind::AudioVAE> {};
    template <>
    struct ObjectKind<models::ControlNet> : std::integral_constant<Kind, Kind::ControlNet> {};
    template <>
    struct ObjectKind<models::ControlBindings> : std::integral_constant<Kind, Kind::ControlBindings> {};
    template <>
    struct ObjectKind<ops::IPAdapterBindings> : std::integral_constant<Kind, Kind::IPAdapterBindings> {};
    template <>
    struct ObjectKind<models::DetailDetector> : std::integral_constant<Kind, Kind::Detector> {};
    template <>
    struct ObjectKind<models::DiffusionModel> : std::integral_constant<Kind, Kind::Denoiser> {};
    template <>
    struct ObjectKind<models::AnimateDiff> : std::integral_constant<Kind, Kind::AnimateDiff> {};
    template <>
    struct ObjectKind<models::LatentUpscaler> : std::integral_constant<Kind, Kind::LatentUpscaler> {};
    template <>
    struct ObjectKind<models::ImageUpscaler> : std::integral_constant<Kind, Kind::Upscaler> {};
    template <>
    struct ObjectKind<models::IPAdapter> : std::integral_constant<Kind, Kind::IPAdapter> {};
    template <>
    struct ObjectKind<models::LoraFile> : std::integral_constant<Kind, Kind::Lora> {};
    template <>
    struct ObjectKind<models::PhotoMaker> : std::integral_constant<Kind, Kind::PhotoMaker> {};
    template <>
    struct ObjectKind<models::PuLID> : std::integral_constant<Kind, Kind::PuLID> {};
    template <>
    struct ObjectKind<ops::PuLIDBindings> : std::integral_constant<Kind, Kind::PuLIDBindings> {};
    template <>
    struct ObjectKind<ops::Predictor> : std::integral_constant<Kind, Kind::Predictor> {};
    template <>
    struct ObjectKind<ops::SamplingObserver> : std::integral_constant<Kind, Kind::SamplingObserver> {};
    template <>
    struct ObjectKind<models::TextEncoder> : std::integral_constant<Kind, Kind::TextEncoder> {};
    template <>
    struct ObjectKind<models::VAE> : std::integral_constant<Kind, Kind::VAE> {};
    template <>
    struct ObjectKind<models::VisionEncoder> : std::integral_constant<Kind, Kind::VisionEncoder> {};

    template <>
    struct ObjectKind<Bytes> : std::integral_constant<Kind, Kind::Bytes> {};

    class Value {
        Type type_;
        std::shared_ptr<void> object_;

    public:
        class Cached;

        nlohmann::json scalar;
        std::vector<Value> items;
        std::map<std::string, Value> fields;
        std::shared_ptr<const models::LoraConfig> loras;
        std::shared_ptr<const ModelSource> model_source;

        Type type() const { return type_; }

        static Value literal(const nlohmann::json& scalar);
        static Value void_value();
        static Value list(std::vector<Value> values, Kind element = Kind::Any);
        static Value dict(std::map<std::string, Value> fields);

        template <class T>
        static Value object(std::shared_ptr<T> object) {
            if (!object)
                throw std::invalid_argument("DGL object payload is null");
            Value value;
            value.type_.kind = ObjectKind<T>::value;
            value.object_    = std::move(object);
            return value;
        }

        template <class T>
        const T* get_if() const {
            return type_.kind == ObjectKind<T>::value ? static_cast<const T*>(object_.get()) : nullptr;
        }

        template <class T>
        std::shared_ptr<T> as() const {
            constexpr Kind expected = ObjectKind<T>::value;
            if (type_.kind != expected)
                throw Error(SD_STATUS_TYPE_MISMATCH, "expected " + type_name({expected}) + ", got " + type_name(type_));
            if (!object_)
                throw std::logic_error("DGL object payload is missing");
            return std::static_pointer_cast<T>(object_);
        }
    };

    class Value::Cached {
        Value value_;
        std::weak_ptr<void> object_;

    public:
        Cached(Value value, bool retain);
        std::optional<Value> lock() const;
        bool expired() const { return object_.expired(); }
        bool idle() const { return value_.object_.use_count() == 1; }
        Type type() const { return value_.type(); }
    };

    using Values         = std::map<std::string, Value>;
    using ScriptFunction = std::function<Value(Value, const Value&)>;

    template <>
    struct ObjectKind<ScriptFunction> : std::integral_constant<Kind, Kind::Function> {};

    bool is_model_kind(Kind kind);
    int64_t get_int(const Values& args, const std::string& name);
    double get_number(const Values& args, const std::string& name);
    std::string get_string(const Values& args, const std::string& name);
    bool get_bool(const Values& args, const std::string& name);

}  // namespace sd::dgl

#endif  // __SD_DGL_VALUE_H__
