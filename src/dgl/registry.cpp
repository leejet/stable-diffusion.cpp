#include "registry.h"

#include <algorithm>
#include <cmath>
#include <sstream>

#include "dgl/bindings/registration.h"
#include "model_definition.h"
#include "stable-diffusion.h"

namespace sd::dgl {

    void Registry::alias(const std::string& name, const std::string& target) {
        if (count(name))
            throw std::runtime_error("function '" + name + "' is already registered");
        const auto found = find(target);
        if (found == end())
            throw std::runtime_error("unknown alias target '" + target + "'");
        aliases_.emplace(name, found->first);
    }

    Operation& Registry::operator[](const std::string& name) {
        if (aliases_.count(name))
            throw std::runtime_error("function '" + name + "' is registered as an alias");
        return operations_[name];
    }

    Operation make_operation(std::vector<Argument> arguments, OutputSignature output, std::function<Value(Values&, RunState&)> execute) {
        return {std::move(arguments), std::move(output), std::move(execute)};
    }

    Operation make_resource_operation(std::vector<Argument> arguments, OutputSignature output, std::function<PreparedResource(Values&)> prepare) {
        Operation operation{std::move(arguments), std::move(output)};
        operation.prepare = std::move(prepare);
        return operation;
    }

    Argument required(const std::string& name, Kind kind) {
        return {name, {kind}};
    }

    static std::optional<Value> literal_default(const nlohmann::json& literal) {
        if (literal.is_null())
            return std::nullopt;
        return Value::literal(literal);
    }

    Argument optional(const std::string& name, Kind kind, const nlohmann::json& default_literal) {
        return {name, {kind}, false, literal_default(default_literal)};
    }
    Argument integer(const std::string& name, int64_t low, int64_t high, const nlohmann::json& default_literal) {
        return {name, {Kind::Integer}, default_literal.is_null(), literal_default(default_literal), [=](const Value& v) {
                    auto n = v.scalar.get<int64_t>();
                    if (n < low || n > high)
                        throw std::runtime_error(name + " must be in [" + std::to_string(low) + ", " + std::to_string(high) + "]");
                }};
    }
    Argument number(const std::string& name, double low, double high, const nlohmann::json& default_literal) {
        return {name, {Kind::Number}, default_literal.is_null(), literal_default(default_literal), [=](const Value& v) {
                    double n = v.scalar.get<double>();
                    if (!std::isfinite(n) || n < low || n > high)
                        throw std::runtime_error(name + " is out of range");
                }};
    }
    Argument choice(const std::string& name, std::vector<std::string> choices, const nlohmann::json& default_literal) {
        return {name, {Kind::String}, default_literal.is_null(), literal_default(default_literal), [=](const Value& v) {
                    auto s = v.scalar.get<std::string>();
                    if (std::find(choices.begin(), choices.end(), s) == choices.end())
                        throw std::runtime_error("unknown " + name + " '" + s + "'");
                }};
    }
    Argument list_argument(const std::string& name, Kind element) {
        return {name, {Kind::List, element}};
    }
    Argument model_path(const std::string& name, bool required) {
        return required ? sd::dgl::required(name, Kind::String) : optional(name, Kind::String, "");
    }
    const Registry& operations() {
        static const Registry registry = [] {
            Registry result;
            register_builtin_operations(result);
            return result;
        }();
        return registry;
    }

    static void write_default(std::ostream& out, const Value& value) {
        if (value.type().kind == Kind::List) {
            out << '[';
            const char* separator = "";
            for (const auto& item : value.items) {
                out << separator;
                write_default(out, item);
                separator = ", ";
            }
            out << ']';
        } else if (value.type().kind == Kind::Dict) {
            out << '{';
            const char* separator = "";
            for (const auto& [name, field] : value.fields) {
                out << separator << name << " = ";
                write_default(out, field);
                separator = ", ";
            }
            out << '}';
        } else {
            out << (value.scalar.is_null() ? type_name(value.type()) : value.scalar.dump());
        }
    }

    static void write_output_fields(std::ostream& out, const OutputSignature& signature, const std::string& prefix = "") {
        for (const auto& [name, field] : signature.fields) {
            const auto path = prefix + name;
            out << "<br>`" << path << ": " << type_name(field.type) << "`";
            write_output_fields(out, field, path + ".");
        }
    }

    std::string builtin_catalog() {
        std::ostringstream out;
        out << "<!-- Generated by sd-cli -M dgl. Update with the dgl-update-builtins CMake target. -->\n"
               "# Built-in DGL functions\n\n"
               "Optional parameters are marked with `?`. Default values follow `=`. "
               "See [DGL syntax](dgl.md#dgl-syntax) for `fn`, `map` and `fold`.\n\n"
               "| Function | Parameters | Returns |\n"
               "|---|---|---|\n";
        const auto& registry = operations();
        std::map<std::string, const Operation*> functions;
        for (const auto& [name, operation] : registry)
            functions.emplace(name, &operation);
        for (const auto& [name, target] : registry.aliases())
            functions.emplace(name, &registry.at(target));
        for (const auto& [name, definition] : functions) {
            const auto& operation = *definition;
            out << "| `" << name << "`";
            if (operation.resource())
                out << " (resource)";
            if (const auto alias = registry.aliases().find(name); alias != registry.aliases().end())
                out << " (alias of `" << alias->second << "`)";
            out << " | ";
            bool first = true;
            for (const auto& argument : operation.arguments) {
                if (!first)
                    out << ", ";
                first = false;
                out << '`' << argument.name << (argument.required ? "" : "?") << ": " << type_name(argument.type);
                if (argument.default_value) {
                    out << " = ";
                    write_default(out, *argument.default_value);
                }
                out << '`';
            }
            out << " | ";
            if (operation.output.same_type_as.empty())
                out << '`' << type_name(operation.output.type) << '`';
            else
                out << "same type as `" << operation.output.same_type_as << '`';
            write_output_fields(out, operation.output);
            out << " |\n";
        }
        out << "\n## Model family IDs\n\n"
               "Pass the ID as the component loader's `family` argument. "
               "The default is `\"auto\"`.\n\n"
               "| ID | Model family |\n"
               "|---|---|\n";
        for (int version = 0; version < VERSION_COUNT; ++version) {
            const auto* definition = find_model_definition(static_cast<SDVersion>(version));
            out << "| `" << definition->id << "` | " << definition->name << " |\n";
        }
        return out.str();
    }

}  // namespace sd::dgl
