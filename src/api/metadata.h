#ifndef __SD_API_METADATA_H__
#define __SD_API_METADATA_H__

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "json.hpp"

namespace sd::api {
    nlohmann::json generation_metadata(const std::string& source) noexcept;
}

namespace sd::metadata {
    constexpr std::size_t jpeg_comment_max_bytes = 65533;
    std::string generation_xmp(const std::string& source, const std::string& parameters = {});
    bool png_metadata(std::vector<uint8_t>& chunks, const std::string& source, const std::string& parameters);
    bool jpeg_metadata(std::vector<uint8_t>& comments, const std::string& source, const std::string& parameters);
    bool embed_png_metadata(std::vector<uint8_t>& image, const std::string& source, const std::string& parameters = {});
    bool embed_jpeg_metadata(std::vector<uint8_t>& image, const std::string& source, const std::string& parameters = {});
    std::vector<uint8_t> avi_info(const std::string& source);
}

#endif  // __SD_API_METADATA_H__
