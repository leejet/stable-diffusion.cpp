#ifndef __SD_DGL_VERSION_H__
#define __SD_DGL_VERSION_H__

#include <cstdint>
#include <string>
#include <string_view>

namespace sd::dgl {
    struct Version {
        uint32_t major = 0;
        uint32_t minor = 0;
        uint32_t patch = 0;

        static Version parse(std::string_view text);
        std::string str() const;
        bool supports(const Version& required) const;
    };

    inline constexpr Version runtime_version{0, 0, 0};
}

#endif  // __SD_DGL_VERSION_H__
