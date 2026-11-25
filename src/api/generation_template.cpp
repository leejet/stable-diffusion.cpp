#include "api/generation_template.h"

#include <algorithm>
#include <map>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "api_templates.h"

namespace sd::api {
    std::string literal(const nlohmann::json& value) {
        if (value.is_object()) {
            std::string result = "{";
            for (auto it = value.begin(); it != value.end(); ++it) {
                if (it != value.begin())
                    result += ", ";
                result += it.key() + " = " + literal(it.value());
            }
            return result + "}";
        }
        if (value.is_array()) {
            std::string result = "[";
            for (size_t i = 0; i < value.size(); ++i)
                result += (i ? ", " : "") + literal(value[i]);
            return result + "]";
        }
        return value.dump();
    }

    std::string arguments(const nlohmann::json& values) {
        if (!values.is_object())
            throw std::logic_error("template argument options must be an object");
        std::string result;
        for (auto it = values.begin(); it != values.end(); ++it)
            result += ", " + it.key() + " = " + literal(it.value());
        return result;
    }

    struct TemplatePart {
        enum class Kind { Text,
                          Variable,
                          Conditional,
                          Include } kind;
        std::string value;
        std::vector<std::pair<std::string, std::vector<TemplatePart>>> branches;
        std::vector<TemplatePart> otherwise;
    };

    static std::string trim_template(std::string_view text) {
        const auto start = text.find_first_not_of(" \t\r\n");
        if (start == std::string_view::npos)
            return "";
        return std::string(text.substr(start, text.find_last_not_of(" \t\r\n") - start + 1));
    }

    class TemplateParser {
        std::string name_;
        std::string_view source_;
        size_t offset_ = 0;

        [[noreturn]] void fail(const std::string& reason) const {
            size_t line = 1;
            for (size_t i = 0; i < offset_; ++i)
                line += source_[i] == '\n';
            throw std::logic_error("template '" + name_ + "', line " + std::to_string(line) + ": " + reason);
        }

        std::vector<TemplatePart> sequence(std::string& boundary) {
            std::vector<TemplatePart> parts;
            while (offset_ < source_.size()) {
                const auto variable = source_.find("{{", offset_);
                const auto control  = source_.find("{%", offset_);
                const auto begin    = std::min(variable, control);
                if (begin == std::string_view::npos) {
                    parts.push_back({TemplatePart::Kind::Text, std::string(source_.substr(offset_))});
                    offset_ = source_.size();
                    break;
                }
                if (begin > offset_)
                    parts.push_back({TemplatePart::Kind::Text, std::string(source_.substr(offset_, begin - offset_))});
                const bool is_variable = begin == variable;
                const auto end         = source_.find(is_variable ? "}}" : "%}", begin + 2);
                if (end == std::string_view::npos)
                    fail("unterminated tag");
                const auto command = trim_template(source_.substr(begin + 2, end - begin - 2));
                offset_            = end + 2;
                if (!is_variable) {
                    if (source_.substr(offset_, 2) == "\r\n")
                        offset_ += 2;
                    else if (offset_ < source_.size() && source_[offset_] == '\n')
                        ++offset_;
                }
                if (is_variable) {
                    if (command.empty())
                        fail("empty variable");
                    parts.push_back({TemplatePart::Kind::Variable, command});
                } else if (command == "else" || command == "endif" || command.rfind("elif ", 0) == 0) {
                    boundary = command;
                    return parts;
                } else if (command.rfind("if ", 0) == 0) {
                    TemplatePart conditional{TemplatePart::Kind::Conditional};
                    auto condition = trim_template(std::string_view(command).substr(3));
                    std::string closing;
                    do {
                        if (condition.empty())
                            fail("empty condition");
                        closing.clear();
                        conditional.branches.emplace_back(condition, sequence(closing));
                        condition = closing.rfind("elif ", 0) == 0 ? trim_template(std::string_view(closing).substr(5)) : "";
                    } while (!condition.empty());
                    if (closing == "else") {
                        closing.clear();
                        conditional.otherwise = sequence(closing);
                    }
                    if (closing != "endif")
                        fail("expected endif");
                    parts.push_back(std::move(conditional));
                } else if (command.rfind("include ", 0) == 0) {
                    const auto path = trim_template(std::string_view(command).substr(8));
                    if (path.size() < 3 || path.front() != '"' || path.back() != '"')
                        fail("include requires a quoted template name");
                    parts.push_back({TemplatePart::Kind::Include, path.substr(1, path.size() - 2)});
                } else {
                    fail("unknown directive '" + command + "'");
                }
            }
            return parts;
        }

    public:
        TemplateParser(std::string name, std::string_view source)
            : name_(std::move(name)), source_(source) {}

        std::vector<TemplatePart> parse() {
            std::string boundary;
            auto result = sequence(boundary);
            if (!boundary.empty())
                fail("unexpected '" + boundary + "'");
            return result;
        }
    };

    static const nlohmann::json& template_value(const nlohmann::json& values, const std::string& name) {
        const auto* value = &values;
        size_t start      = 0;
        do {
            const auto dot   = name.find('.', start);
            const auto field = name.substr(start, dot - start);
            if (!value->is_object() || !value->contains(field))
                throw std::logic_error("missing template variable '" + name + "'");
            value = &value->at(field);
            if (dot == std::string::npos)
                return *value;
            start = dot + 1;
        } while (start < name.size());
        throw std::logic_error("invalid template variable '" + name + "'");
    }

    static bool template_condition(const nlohmann::json& values, std::string name) {
        const bool invert = name.rfind("not ", 0) == 0;
        if (invert)
            name = trim_template(std::string_view(name).substr(4));
        const auto& value = template_value(values, name);
        if (!value.is_boolean())
            throw std::logic_error("template condition '" + name + "' must be boolean");
        return value.get<bool>() != invert;
    }

    using ParsedTemplates = std::map<std::string, std::vector<TemplatePart>>;

    static void render_parts(std::string& output, const std::vector<TemplatePart>& parts, const nlohmann::json& values, const ParsedTemplates& sources, int depth) {
        if (depth > 16)
            throw std::logic_error("template include depth exceeds 16");
        for (const auto& part : parts) {
            if (part.kind == TemplatePart::Kind::Text) {
                output += part.value;
            } else if (part.kind == TemplatePart::Kind::Variable) {
                const auto separator = part.value.find('|');
                const auto name      = trim_template(std::string_view(part.value).substr(0, separator));
                const auto& value    = template_value(values, name);
                const auto filter    = separator == std::string::npos ? "" : trim_template(std::string_view(part.value).substr(separator + 1));
                if (filter == "literal")
                    output += literal(value);
                else if (filter == "args")
                    output += arguments(value);
                else if (filter.empty())
                    output += value.is_string() ? value.get<std::string>() : literal(value);
                else
                    throw std::logic_error("unknown template filter '" + filter + "'");
            } else if (part.kind == TemplatePart::Kind::Include) {
                const auto include = sources.find(part.value);
                if (include == sources.end())
                    throw std::logic_error("unknown included template '" + part.value + "'");
                render_parts(output, include->second, values, sources, depth + 1);
            } else {
                const auto* selected = &part.otherwise;
                for (const auto& [condition, branch] : part.branches)
                    if (template_condition(values, condition)) {
                        selected = &branch;
                        break;
                    }
                render_parts(output, *selected, values, sources, depth);
            }
        }
    }

    std::string render_template(const std::string& name, const nlohmann::json& values) {
        static const auto sources = [] {
            ParsedTemplates result;
            for (const auto& [name, source] : templates::sources)
                result.emplace(name, TemplateParser(name, source).parse());
            return result;
        }();
        const auto source = sources.find(name);
        if (source == sources.end())
            throw std::logic_error("unknown generation template '" + name + "'");
        std::string result;
        render_parts(result, source->second, values, sources, 0);
        return result;
    }
}
