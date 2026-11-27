#ifndef __SD_CORE_CUDA_INT8_COMMON_H__
#define __SD_CORE_CUDA_INT8_COMMON_H__

#include "runtime.h"

namespace SD_CUDA_NAMESPACE::i8 {
    template <int Width = 32>
    static __device__ __forceinline__ float warp_reduce_max(float value) {
#pragma unroll
        for (int offset = Width / 2; offset > 0; offset >>= 1)
            value = fmaxf(value, cuda_shuffle_xor(value, offset, Width));
        return value;
    }

    static bool aligned(const ggml_tensor* tensor, size_t alignment = 4) {
        if (!tensor)
            return true;
        size_t offset = 0;
        while (tensor->view_src) {
            offset += tensor->view_offs;
            tensor = tensor->view_src;
        }
        return (reinterpret_cast<uintptr_t>(tensor->data) + offset) % alignment == 0;
    }

    static bool blas_supported(ggml_backend_dev_t device, const ggml_ext_gpu_device_info& info, int64_t n) {
        return info.int8_blas && n % 4 == 0 &&
               ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(device), "ggml_ext_gpu_blas_v1");
    }
}

#endif  // __SD_CORE_CUDA_INT8_COMMON_H__
