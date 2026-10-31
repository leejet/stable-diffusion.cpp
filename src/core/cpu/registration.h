#ifndef __SD_CORE_CPU_REGISTRATION_H__
#define __SD_CORE_CPU_REGISTRATION_H__

#include "ggml-backend.h"

#ifndef SD_USE_UPSTREAM_GGML
namespace sd_cpu {
    ggml_status parallel(ggml_backend_t backend, ggml_tensor* output, ggml_custom_op_t callback, void* user_data);
    void register_extensions(ggml_backend_dev_t device);
    void register_rope(ggml_backend_dev_t device);
    void register_i8(ggml_backend_dev_t device);
}
#endif

#endif
