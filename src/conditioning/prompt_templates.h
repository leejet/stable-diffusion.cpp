#ifndef __SD_CONDITIONING_PROMPT_TEMPLATES_H__
#define __SD_CONDITIONING_PROMPT_TEMPLATES_H__

#include <optional>
#include <string>

namespace sd::prompt_templates {
    enum class RoleFormat { ChatML,
                            Mistral,
                            GPTOSS,
                            Role,
                            Gemma };

    inline std::string with_system(const std::string& prefix, const std::optional<std::string>& system, RoleFormat format = RoleFormat::ChatML) {
        if (!system)
            return prefix;
        const char* open  = "<|im_start|>system\n";
        const char* close = "<|im_end|>\n";
        switch (format) {
            case RoleFormat::Mistral:
                open  = "[SYSTEM_PROMPT]";
                close = "[/SYSTEM_PROMPT]";
                break;
            case RoleFormat::GPTOSS:
                open  = "<|start|>system<|message|>";
                close = "<|end|>";
                break;
            case RoleFormat::Role:
                open  = "<role>SYSTEM</role>";
                close = "<|role_end|>";
                break;
            case RoleFormat::Gemma:
                open  = "<start_of_turn>system\n";
                close = "<end_of_turn>\n";
                break;
            case RoleFormat::ChatML:
                break;
        }
        const std::string begin(open), end(close);
        const auto start   = prefix.find(begin);
        const auto finish  = start == std::string::npos ? std::string::npos : prefix.find(end, start + begin.size());
        const auto segment = system->empty() ? std::string{} : begin + *system + end;
        if (finish == std::string::npos)
            return segment + prefix;
        return prefix.substr(0, start) + segment + prefix.substr(finish + end.size());
    }

    inline std::string ming_image(const std::string& text, const std::optional<std::string>& system, bool raw) {
        if (raw)
            return text;
        return with_system("<role>SYSTEM</role>你是一个友好的AI助手。\n\ndetailed thinking off<|role_end|><role>HUMAN</role>", system, RoleFormat::Role) +
               text + "<|role_end|><role>ASSISTANT</role>";
    }

    inline std::string llada_image(std::string text, const std::optional<std::string>& system, bool raw) {
        if (raw)
            return text;
        const auto first = text.find_first_not_of(" \t\n\r\f\v");
        const auto last  = text.find_last_not_of(" \t\n\r\f\v");
        text             = first == std::string::npos ? std::string{} : text.substr(first, last - first + 1);
        auto prompt      = text.empty() ? "<role>HUMAN</role> Generate an image.\n<role>ASSISTANT</role>\n<IMAGE1>"
                                        : "<role>HUMAN</role> Generate an image: " + text + "\n<role>ASSISTANT</role>\n<IMAGE1>";
        return with_system("", system, RoleFormat::Role) + prompt;
    }

    inline std::string system_message(const std::optional<std::string>& system, const std::string& fallback) {
        const auto& text = system ? *system : fallback;
        return text.empty() ? std::string{} : "<|im_start|>system\n" + text + "<|im_end|>\n";
    }

    inline std::string sensenova_u1(const std::string& text, const std::optional<std::string>& system) {
        static const std::string kSystemMessage =
            "You are an image generation and editing assistant that accurately understands and executes user intent.\n\n"
            "You support two modes:\n\n1. Think Mode:\nIf the task requires reasoning, you MUST start with a "
            "<think></think> block. Put all reasoning inside the block using plain text. DO NOT include any image tags. "
            "Keep it reasonable and directly useful for producing the final image.\n\n2. Non-Think Mode:\nIf no reasoning "
            "is needed, directly produce the final image.\n\nTask Types:\n\nA. Text-to-Image Generation:\n- Generate a "
            "high-quality image based on the user's description.\n- Ensure visual clarity, semantic consistency, and "
            "completeness.\n- DO NOT introduce elements that contradict or override the user's intent.\n\nB. Image Editing:\n"
            "- Use the provided image(s) as input or reference for modification or transformation.\n- The result can be an "
            "edited image or a new image based on the reference(s).\n- Preserve all unspecified attributes unless explicitly "
            "changed.\n\nGeneral Rules:\n- For any visible text in the image, follow the language specified for the rendered "
            "text in the user's description, not the language of the prompt. If no language is specified, use the user's input "
            "language.";

        return system_message(system, kSystemMessage) + "<|im_start|>user\n" + text +
               "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<img>";
    }
}

#endif  // __SD_CONDITIONING_PROMPT_TEMPLATES_H__
