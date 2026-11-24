#ifndef __SD_DGL_DIAGNOSTIC_H__
#define __SD_DGL_DIAGNOSTIC_H__

#include <stdexcept>

#include "stable-diffusion.h"

namespace sd::dgl {
    struct Error : std::runtime_error {
        sd_status_t status;

        Error(sd_status_t status, const std::string& message)
            : std::runtime_error(message), status(status) {}
    };
}  // namespace sd::dgl

#endif  // __SD_DGL_DIAGNOSTIC_H__
