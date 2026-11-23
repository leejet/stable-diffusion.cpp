#include "api/generation_program.h"

#include <algorithm>
#include <cctype>

#include "api/tensor.h"

namespace sd::api {
    static std::string embedding_lookup(std::string value) {
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return std::tolower(c); });
        return value;
    }

    static void prepare_textual_inversion(GenerationTemplate& request) {
        const auto& config = request.context.model.config.context;
        if (!config.embedding_count)
            return;
        std::vector<std::pair<std::string, std::string>> embeddings;
        for (uint32_t i = 0; i < config.embedding_count; ++i) {
            auto path = text(config.embeddings[i].path);
            std::replace(path.begin(), path.end(), '\\', '/');
            if (path.find('>') != std::string::npos)
                throw dgl::Error(SD_STATUS_INVALID_ARGUMENT, "embedding path cannot contain '>'");
            embeddings.emplace_back(embedding_lookup(config.embeddings[i].name), std::move(path));
        }
        std::stable_sort(embeddings.begin(), embeddings.end(), [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
        const auto word = [](unsigned char c) { return std::isalnum(c) || c == '_' || c >= 128; };
        for (const auto* key : {"prompt", "negative_prompt"}) {
            const auto original = request.values.at(key).get<std::string>();
            const auto lookup   = embedding_lookup(original);
            std::string rewritten;
            for (size_t cursor = 0; cursor < original.size();) {
                if (original[cursor] == '<') {
                    const auto end = original.find('>', cursor);
                    if (end != std::string::npos) {
                        rewritten += original.substr(cursor, end + 1 - cursor);
                        cursor = end + 1;
                        continue;
                    }
                }
                bool matched = false;
                for (const auto& [name, path] : embeddings) {
                    const auto end = cursor + name.size();
                    if ((cursor && word(original[cursor - 1])) ||
                        lookup.compare(cursor, name.size(), name) || (end < original.size() && word(original[end])))
                        continue;
                    rewritten += "<embedding:" + path + ">";
                    cursor  = end;
                    matched = true;
                    break;
                }
                if (!matched)
                    rewritten += original[cursor++];
            }
            if (rewritten != original) {
                request.values[std::string("source_") + key] = original;
                request.values[key]                          = std::move(rewritten);
            }
        }
    }

    void prepare_prompt_images(GenerationTemplate& request) {
        prepare_textual_inversion(request);
        auto& context = request.context;
        if (!context.prompt_image)
            return;
        std::map<std::string, std::string> aliases;
        for (const auto* key : {"prompt", "negative_prompt"}) {
            const auto original = request.values.at(key).get<std::string>();
            std::string rewritten;
            size_t cursor = 0;
            while (cursor < original.size()) {
                const auto start = original.find("<image:", cursor);
                if (start == std::string::npos) {
                    rewritten += original.substr(cursor);
                    break;
                }
                size_t slashes = 0;
                while (start > slashes && original[start - slashes - 1] == '\\')
                    ++slashes;
                rewritten += original.substr(cursor, start - cursor);
                if (slashes % 2) {
                    rewritten += "<image:";
                    cursor = start + 7;
                    continue;
                }
                const auto end = original.find('>', start + 7);
                if (end == std::string::npos || end == start + 7)
                    throw dgl::Error(SD_STATUS_INVALID_ARGUMENT, "invalid inline prompt image");
                const auto source = original.substr(start + 7, end - start - 7);
                auto found        = aliases.find(source);
                if (found == aliases.end()) {
                    if (aliases.size() >= 32)
                        throw dgl::Error(SD_STATUS_OUT_OF_RANGE, "a request supports at most 32 prompt images");
                    context.check_cancelled();
                    const sd_tensor_t* tensor = nullptr;
                    if (!context.prompt_image(source.c_str(), context.prompt_image_data, &tensor) || !tensor)
                        throw dgl::Error(SD_STATUS_INVALID_ARGUMENT, "cannot resolve prompt image '" + source + "'");
                    image_view(*tensor);
                    auto copy        = std::make_shared<sd_tensor_t>(tensor->type, tensor->shape(), tensor->data(), tensor->bytes);
                    const auto alias = "prompt_image_" + std::to_string(aliases.size());
                    request.program.inputs.emplace(alias, dgl::Value::object(std::move(copy)));
                    found = aliases.emplace(source, alias).first;
                }
                rewritten += "<image:" + found->second + ">";
                cursor = end + 1;
            }
            if (rewritten != original) {
                request.values[std::string("source_") + key] = original;
                request.values[key]                          = rewritten;
            }
        }
        if (aliases.empty())
            return;
        std::string bindings;
        for (const auto& [source, alias] : aliases)
            bindings += (bindings.empty() ? "" : ", ") + alias + " = " + alias;
        const auto argument = ", image_bindings = {" + bindings + "}";
        request.fields["prompt_options"] += argument;
        request.fields["negative_prompt_options"] += argument;
        request.fields["unconditional_prompt_options"] += argument;
    }
}
