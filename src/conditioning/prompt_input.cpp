#include "conditioning/prompt_input.h"

#include <algorithm>
#include <map>
#include <stdexcept>

#include "core/util.h"

namespace sd {

    PromptInputTemplate::PromptInputTemplate(const PromptInput* input)
        : input_(input) {
        if (input == nullptr) {
            return;
        }
        uint32_t codepoint = 0xf0000;
        for (size_t i = 0; i < input->size(); ++i) {
            const auto& part = (*input)[i];
            if (part.type == PromptPart::Type::TEXT) {
                text += part.content;
                continue;
            }
            // One private-use code point survives JSON formatting and quote spelling.
            std::string marker;
            do {
                if (codepoint > 0xffffd) {
                    throw std::invalid_argument("Too many prompt markers");
                }
                marker = {static_cast<char>(0xf0 | (codepoint >> 18)),
                          static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)),
                          static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)),
                          static_cast<char>(0x80 | (codepoint & 0x3f))};
                ++codepoint;
            } while (std::any_of(input->begin(), input->end(), [&](const PromptPart& source) {
                return source.content.find(marker) != std::string::npos;
            }));
            markers_.emplace(marker, i);
            text += marker;
        }
    }

    PromptInput PromptInputTemplate::expand(const std::string& formatted_text, float weight) const {
        PromptInput result;
        size_t cursor = 0;
        size_t search = 0;
        while (input_ != nullptr && search < formatted_text.size()) {
            const size_t begin = formatted_text.find(static_cast<char>(0xf3), search);
            if (begin == std::string::npos) {
                break;
            }
            const auto marker = markers_.find(formatted_text.substr(begin, 4));
            search            = begin + 1;
            if (marker == markers_.end()) {
                continue;
            }
            if (begin > cursor) {
                result.push_back({PromptPart::Type::TEXT, formatted_text.substr(cursor, begin - cursor), {}, weight});
            }
            result.push_back((*input_)[marker->second]);
            cursor = begin + marker->first.size();
            search = cursor;
        }
        if (cursor < formatted_text.size()) {
            result.push_back({PromptPart::Type::TEXT, formatted_text.substr(cursor), {}, weight});
        }
        return result;
    }

    PromptInput parse_prompt_input(const std::string& text, bool allow_embeddings) {
        PromptInput parts;
        std::string plain_text;
        size_t cursor = 0;
        while (cursor < text.size()) {
            const size_t image       = text.find("<image:", cursor);
            const size_t embedding   = allow_embeddings ? text.find("<embedding:", cursor) : std::string::npos;
            const size_t begin       = std::min(image, embedding);
            const bool is_embedding  = embedding != std::string::npos && begin == embedding;
            const std::string marker = is_embedding ? "<embedding:" : "<image:";
            const size_t path_begin  = begin + marker.size();
            if (begin == std::string::npos) {
                plain_text += text.substr(cursor);
                break;
            }
            size_t slashes = 0;
            while (begin > slashes && text[begin - slashes - 1] == '\\') {
                ++slashes;
            }
            if (slashes % 2 != 0) {
                plain_text += text.substr(cursor, begin - cursor - 1);
                plain_text += marker;
                cursor = path_begin;
                continue;
            }
            const size_t end = text.find('>', path_begin);
            if (end == std::string::npos || end == path_begin) {
                throw std::invalid_argument("Invalid " + marker + "...> prompt marker at byte " + std::to_string(begin));
            }
            plain_text += text.substr(cursor, begin - cursor);
            if (!plain_text.empty()) {
                parts.push_back({PromptPart::Type::TEXT, std::move(plain_text), {}});
                plain_text.clear();
            }
            parts.push_back({is_embedding ? PromptPart::Type::EMBEDDING : PromptPart::Type::IMAGE, text.substr(path_begin, end - path_begin), {}});
            cursor = end + 1;
        }
        if (!plain_text.empty() || parts.empty()) {
            parts.push_back({PromptPart::Type::TEXT, std::move(plain_text), {}});
        }
        return parts;
    }

    bool prompt_has_images(const PromptInput& input) {
        return std::any_of(input.begin(), input.end(), [](const PromptPart& part) {
            return part.type == PromptPart::Type::IMAGE;
        });
    }

    std::string prompt_input_text(const PromptInput& input) {
        std::string text;
        for (const auto& part : input) {
            if (part.type == PromptPart::Type::TEXT) {
                text += part.content;
            }
        }
        return text;
    }

    bool resolve_prompt_images(PromptInput& input, const PromptImageLoader& loader) {
        constexpr size_t max_image_sources = 32;
        std::map<std::string, std::shared_ptr<const Tensor<float>>> images;
        for (const auto& part : input) {
            if (part.type == PromptPart::Type::IMAGE) {
                images.try_emplace(part.content);
                if (images.size() > max_image_sources) {
                    LOG_ERROR("A prompt supports at most %zu unique image sources", max_image_sources);
                    return false;
                }
            }
        }
        for (auto& part : input) {
            if (part.type != PromptPart::Type::IMAGE) {
                continue;
            }
            if (!loader) {
                return false;
            }
            auto& resolved = images.at(part.content);
            if (!resolved) {
                auto image = loader(part.content);
                if (image.empty()) {
                    return false;
                }
                resolved = std::make_shared<const Tensor<float>>(std::move(image));
            }
            part.image = resolved;
        }
        return true;
    }

}  // namespace sd
