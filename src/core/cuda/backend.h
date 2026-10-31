#ifndef __SD_CORE_CUDA_BACKEND_H__
#define __SD_CORE_CUDA_BACKEND_H__

#include "ggml-ext-gpu.h"
#include "ggml-impl.h"
#if defined(SD_CUDA_VENDOR_HIP)
#include "vendors/hip.h"
#define SD_CUDA_NAMESPACE sd_hip
#define SD_CUDA_NAME "hip"
#elif defined(SD_CUDA_VENDOR_MUSA)
#include "vendors/musa.h"
#define SD_CUDA_NAMESPACE sd_musa
#define SD_CUDA_NAME "musa"
#else
#include "vendors/cuda.h"
#define SD_CUDA_NAMESPACE sd_cuda
#define SD_CUDA_NAME "cuda"
#endif

namespace SD_CUDA_NAMESPACE {
#if defined(SD_CUDA_VENDOR_HIP)
    static constexpr auto cuda_kind = GGML_EXT_GPU_HIP;
#elif defined(SD_CUDA_VENDOR_MUSA)
    static constexpr auto cuda_kind = GGML_EXT_GPU_MUSA;
#else
    static constexpr auto cuda_kind = GGML_EXT_GPU_CUDA;
#endif
    static bool cuda_context(ggml_backend_t backend, ggml_ext_gpu_context* context) {
        auto reg   = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
        auto query = reinterpret_cast<ggml_ext_gpu_context_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_ext_gpu_context_v1"));
        return query && query(backend, context) && context->kind == cuda_kind;
    }
    static ggml_ext_gpu_device_info cuda_device_info(ggml_backend_dev_t device) {
        ggml_ext_gpu_device_info info = {};
        auto query                    = reinterpret_cast<ggml_ext_gpu_device_fn>(ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(device), "ggml_ext_gpu_device_v1"));
        if (!query || !query(device, &info) || info.kind != cuda_kind)
            return {cuda_kind, -1, 0, 0, false};
        return info;
    }
}
#endif
