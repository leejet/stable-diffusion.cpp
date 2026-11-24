#include "dgl/ops/media_encoding.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace sd::dgl::ops {
    void save_media(const std::string& path, const std::vector<uint8_t>& data, const std::string& mime_type) {
        const char* expected = mime_type == "image/png" ? ".png" : mime_type == "video/webm"    ? ".webm"
                                                               : mime_type == "video/x-msvideo" ? ".avi"
                                                                                                : nullptr;
        if (!expected)
            throw std::runtime_error("media::save does not support MIME type '" + mime_type + "'");
        const auto output = std::filesystem::u8path(path);
        auto extension    = output.extension().u8string();
        std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) { return char(std::tolower(c)); });
        if (extension != expected)
            throw std::runtime_error("media::save path extension must be '" + std::string(expected) + "' for " + mime_type);
        if (data.empty() || data.size() > size_t(std::numeric_limits<std::streamsize>::max()))
            throw std::runtime_error("media::save requires nonempty bytes that fit a file stream");
        std::ofstream file(output, std::ios::binary | std::ios::trunc);
        file.write(reinterpret_cast<const char*>(data.data()), std::streamsize(data.size()));
        file.close();
        if (!file)
            throw std::runtime_error("cannot save media '" + path + "'");
    }
}
