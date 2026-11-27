#ifndef __SD_DGL_REGISTRY_H__
#define __SD_DGL_REGISTRY_H__

#include <functional>
#include <map>
#include <optional>
#include <utility>

#include "value.h"

enum class SDBackendModule;

namespace sd::dgl {

    struct CallbackSignature {
        std::string args_parameter;
        std::map<std::string, Type> input_fields;
        std::map<std::string, Type> output_fields;
    };

    struct Argument {
        std::string name;
        Type type;
        bool required = true;
        std::optional<Value> default_value;
        std::function<void(const Value&)> validate;
        std::optional<CallbackSignature> callback;
    };

    struct RunState;
    struct ResourceLoadRequest;

    struct PreparedResource {
        nlohmann::json key;
        nlohmann::json revision;
        std::function<Value(ResourceLoadRequest&)> load;
    };

    struct OutputSignature {
        Type type;
        std::map<std::string, OutputSignature> fields;
        std::string same_type_as;

        OutputSignature(Kind kind = Kind::Any, Kind element = Kind::Any)
            : type{kind, element} {}
        OutputSignature(Type type, std::map<std::string, OutputSignature> fields = {})
            : type(type), fields(std::move(fields)) {}
    };

    struct Operation {
        enum class ModelUse { Automatic,
                              Metadata,
                              Capture,
                              Preview };
        std::vector<Argument> arguments;
        OutputSignature output;
        std::function<Value(Values&, RunState&)> execute;
        bool sampling = false;
        std::function<PreparedResource(Values&)> prepare;
        ModelUse model_use = ModelUse::Automatic;
        std::optional<SDBackendModule> backend_module;

        bool resource() const { return bool(prepare); }
    };

    class Registry {
    public:
        using Operations = std::map<std::string, Operation>;
        using Entry      = Operations::value_type;

    private:
        Operations operations_;
        std::map<std::string, std::string> aliases_;

        const std::string& canonical_name(const std::string& name) const {
            const auto found = aliases_.find(name);
            return found == aliases_.end() ? name : found->second;
        }

    public:
        void alias(const std::string& name, const std::string& target);
        Operation& operator[](const std::string& name);
        Operation& at(const std::string& name) { return operations_.at(canonical_name(name)); }
        const Operation& at(const std::string& name) const { return operations_.at(canonical_name(name)); }
        auto find(const std::string& name) { return operations_.find(canonical_name(name)); }
        auto find(const std::string& name) const { return operations_.find(canonical_name(name)); }
        auto count(const std::string& name) const { return operations_.count(canonical_name(name)); }
        auto begin() { return operations_.begin(); }
        auto begin() const { return operations_.begin(); }
        auto end() { return operations_.end(); }
        auto end() const { return operations_.end(); }
        const auto& aliases() const { return aliases_; }
    };

    Operation make_operation(std::vector<Argument> arguments, OutputSignature output, std::function<Value(Values&, RunState&)> execute);
    Operation make_resource_operation(std::vector<Argument> arguments, OutputSignature output, std::function<PreparedResource(Values&)> prepare);

    Argument required(const std::string& name, Kind kind);
    Argument optional(const std::string& name, Kind kind, const nlohmann::json& default_literal = nullptr);
    Argument integer(const std::string& name, int64_t low, int64_t high, const nlohmann::json& default_literal = nullptr);
    Argument number(const std::string& name, double low, double high, const nlohmann::json& default_literal = nullptr);
    Argument choice(const std::string& name, std::vector<std::string> choices, const nlohmann::json& default_literal = nullptr);
    Argument list_argument(const std::string& name, Kind element);
    Argument model_path(const std::string& name, bool required = true);
    const Registry& operations();
    std::string builtin_catalog();

}  // namespace sd::dgl

#endif  // __SD_DGL_REGISTRY_H__
