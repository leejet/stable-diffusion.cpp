#ifndef __SD_API_GENERATION_TEMPLATE_H__
#define __SD_API_GENERATION_TEMPLATE_H__

#include <string>

#include "json.hpp"

namespace sd::api {
    std::string literal(const nlohmann::json& value);
    std::string arguments(const nlohmann::json& values);
    std::string render_template(const std::string& name, const nlohmann::json& values);
}

#endif
