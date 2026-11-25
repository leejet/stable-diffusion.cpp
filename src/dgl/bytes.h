#ifndef __SD_DGL_BYTES_H__
#define __SD_DGL_BYTES_H__

#include <cstdint>
#include <utility>
#include <vector>

namespace sd::dgl {
    struct Bytes {
        const std::vector<uint8_t> data;

        explicit Bytes(std::vector<uint8_t> value)
            : data(std::move(value)) {}
    };
}

#endif
