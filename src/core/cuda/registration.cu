#include "backend.h"

#ifdef SD_CUDA_CUTE_ATTENTION
void sd_ggml_register_cuda_i8_gated_residual(ggml_backend_dev_t device);
void sd_ggml_register_cuda_i8_projection(ggml_backend_dev_t device);
void sd_ggml_register_cuda_attention(ggml_backend_dev_t device);
#endif

namespace SD_CUDA_NAMESPACE {
    void register_rope(ggml_backend_dev_t device);
    void register_i8(ggml_backend_dev_t device);
    void register_convrot(ggml_backend_dev_t device);
    void register_affine(ggml_backend_dev_t device);
#if !defined(SD_CUDA_VENDOR_HIP) && !defined(SD_CUDA_VENDOR_MUSA)
    void register_sage_attention(ggml_backend_dev_t device);
    void register_sol_attention(ggml_backend_dev_t device);
#endif

    void register_extensions(ggml_backend_dev_t device) {
        if (cuda_device_info(device).device < 0)
            return;
        register_affine(device);
        register_i8(device);
        register_convrot(device);
        register_rope(device);
#if !defined(SD_CUDA_VENDOR_HIP) && !defined(SD_CUDA_VENDOR_MUSA)
        register_sage_attention(device);
        register_sol_attention(device);
#endif
#ifdef SD_CUDA_CUTE_ATTENTION
        sd_ggml_register_cuda_i8_projection(device);
        sd_ggml_register_cuda_i8_gated_residual(device);
        sd_ggml_register_cuda_attention(device);
#endif
    }
}
