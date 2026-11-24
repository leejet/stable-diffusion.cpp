#include "media_metadata.h"

#include <cstring>

#include "stable-diffusion.h"

std::string generation_xmp(const std::string& source) {
    const void* data = nullptr;
    size_t size      = 0;
    if (sd_get_metadata_chunk(source.c_str(), SD_METADATA_XMP, &data, &size) != SD_STATUS_OK || !size)
        return "";
    return std::string(static_cast<const char*>(data), size);
}

bool embed_jpeg_metadata(std::vector<uint8_t>& image, const std::string& source) {
    if (source.empty())
        return true;
    if (image.size() < 2)
        return false;
    const void* data = nullptr;
    size_t size      = 0;
    if (sd_get_metadata_chunk(source.c_str(), SD_METADATA_JPEG, &data, &size) != SD_STATUS_OK)
        return false;
    const auto* bytes = static_cast<const uint8_t*>(data);
    if (size)
        image.insert(image.begin() + 2, bytes, bytes + size);
    return true;
}

bool embed_png_metadata(std::vector<uint8_t>& image, const std::string& source) {
    if (source.empty())
        return true;
    if (image.size() < 33 || std::memcmp(image.data() + 12, "IHDR", 4))
        return false;
    const void* data = nullptr;
    size_t size      = 0;
    if (sd_get_metadata_chunk(source.c_str(), SD_METADATA_PNG, &data, &size) != SD_STATUS_OK)
        return false;
    const auto* bytes = static_cast<const uint8_t*>(data);
    if (size)
        image.insert(image.begin() + 33, bytes, bytes + size);
    return true;
}
