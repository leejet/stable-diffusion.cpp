#ifndef __SD_API_GENERATION_PROGRAM_H__
#define __SD_API_GENERATION_PROGRAM_H__

#include "api/generation.h"

namespace sd::api {
    struct GenerationTemplate {
        GenerationContext& context;
        const ImageConfig& image;
        const sd_opt_t* options;
        const ModelDefinition& definition;
        GenerationProgram& program;
        nlohmann::json& values;
        std::map<std::string, std::string>& fields;
        const VideoConfig* video;
        std::string vae_options;
        int64_t seed;
        nlohmann::json flags = nlohmann::json::object();

        std::string render(const std::string& name, const nlohmann::json& overrides = nlohmann::json::object()) const;
    };

    std::string text(const char* value);
    nlohmann::json loader_options(const ContextConfig& settings, SDBackendModule module, const ModelDefinition& definition);
    void bind_tensor(GenerationProgram& program, const std::string& name, const sd_tensor_t* tensor);
    void prepare_image_program(GenerationTemplate& request);
    void prepare_video_program(GenerationTemplate& request);
    void prepare_hires_program(GenerationTemplate& request);
    void prepare_prompt_images(GenerationTemplate& request);
    void prepare_h3_references(GenerationTemplate& request, const std::string& first, const std::string& last);
}

#endif  // __SD_API_GENERATION_PROGRAM_H__
