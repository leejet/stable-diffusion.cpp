#include "core/cpu/registration.h"

#ifndef SD_USE_UPSTREAM_GGML
namespace sd_cpu {

    void register_extensions(ggml_backend_dev_t device) {
        register_i8(device);
        register_rope(device);
        register_modulate(device);
    }

}
#endif
