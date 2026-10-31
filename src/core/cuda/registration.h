#ifndef __SD_CORE_CUDA_REGISTRATION_H__
#define __SD_CORE_CUDA_REGISTRATION_H__
#include "ggml-backend.h"
namespace sd_cuda {
    void register_extensions(ggml_backend_dev_t device);
}
namespace sd_hip {
    void register_extensions(ggml_backend_dev_t device);
}
namespace sd_musa {
    void register_extensions(ggml_backend_dev_t device);
}
#endif
