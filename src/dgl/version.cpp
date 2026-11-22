#include "dgl/version.h"

#include <charconv>
#include <stdexcept>
#include <tuple>

namespace sd::dgl {
    Version Version::parse(std::string_view text) {
        constexpr const char* format_error = "DGL version must use MAJOR.MINOR.PATCH (for example, 0.0.0); integer and shortened versions are not supported";
        if (text.empty())
            throw std::invalid_argument(format_error);
        Version version;
        uint32_t* components[] = {&version.major, &version.minor, &version.patch};
        const char* next       = text.data();
        const char* end        = next + text.size();
        for (size_t i = 0; i < 3; ++i) {
            const auto parsed = std::from_chars(next, end, *components[i]);
            if (parsed.ec == std::errc::result_out_of_range)
                throw std::invalid_argument("DGL version component exceeds the unsigned 32-bit range");
            if (parsed.ec != std::errc{} ||
                (i < 2 ? parsed.ptr == end || *parsed.ptr != '.' : parsed.ptr != end))
                throw std::invalid_argument(format_error);
            next = parsed.ptr + (i < 2 ? 1 : 0);
        }
        return version;
    }

    std::string Version::str() const {
        return std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch);
    }

    bool Version::supports(const Version& required) const {
        return major == required.major && std::tie(minor, patch) >= std::tie(required.minor, required.patch);
    }
}
