#include <cstdio>

#include "ggml-backend.h"
#include "stable-diffusion.h"

#ifndef SDCPP_BUILD_COMMIT
#define SDCPP_BUILD_COMMIT unknown
#endif

#ifndef SDCPP_BUILD_VERSION
#define SDCPP_BUILD_VERSION unknown
#endif

#define STRINGIZE2(x) #x
#define STRINGIZE(x) STRINGIZE2(x)

const char* sd_commit(void) {
    return STRINGIZE(SDCPP_BUILD_COMMIT);
}

const char* sd_version(void) {
    return STRINGIZE(SDCPP_BUILD_VERSION);
}

void sd_list_devices(void) {
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const char* name       = ggml_backend_dev_name(dev);
        const char* desc       = ggml_backend_dev_description(dev);
        std::printf("%s\t%s\n", name ? name : "", desc ? desc : "");
    }
}
