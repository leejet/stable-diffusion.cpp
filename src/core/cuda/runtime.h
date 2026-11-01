#ifndef __SD_CORE_CUDA_RUNTIME_H__
#define __SD_CORE_CUDA_RUNTIME_H__

#include "ggml-ext.h"
#if defined(SD_CUDA_VENDOR_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#define SD_CUDA_NAMESPACE sd_hip
#define SD_CUDA_NAME "hip"
#elif defined(SD_CUDA_VENDOR_MUSA)
#include <musa_fp16.h>
#include <musa_runtime.h>
#define SD_CUDA_NAMESPACE sd_musa
#define SD_CUDA_NAME "musa"
#else
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#define SD_CUDA_NAMESPACE sd_cuda
#define SD_CUDA_NAME "cuda"
#endif

namespace SD_CUDA_NAMESPACE {
#if defined(SD_CUDA_VENDOR_HIP)
    using CUDAError                          = hipError_t;
    using CUDAStream                         = hipStream_t;
    static constexpr auto cuda_success       = hipSuccess;
    static constexpr auto cuda_invalid_value = hipErrorInvalidValue;
    static constexpr auto cuda_kind          = GGML_EXT_GPU_HIP;
    static CUDAError cuda_last_error() {
        return hipGetLastError();
    }
    static CUDAError cuda_memset_async(void* dst, int value, size_t bytes, CUDAStream stream) {
        return hipMemsetAsync(dst, value, bytes, stream);
    }
    static CUDAError cuda_copy_async(void* dst, const void* src, size_t bytes, CUDAStream stream) {
        return hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToDevice, stream);
    }
    template <typename Kernel>
    static CUDAError cuda_set_shared_memory(Kernel, int) {
        // HIP uses the device's shared-memory limit without CUDA's opt-in attribute.
        return cuda_success;
    }
#elif defined(SD_CUDA_VENDOR_MUSA)
    using CUDAError                          = musaError_t;
    using CUDAStream                         = musaStream_t;
    static constexpr auto cuda_success       = musaSuccess;
    static constexpr auto cuda_invalid_value = musaErrorInvalidValue;
    static constexpr auto cuda_kind          = GGML_EXT_GPU_MUSA;
    static CUDAError cuda_last_error() {
        return musaGetLastError();
    }
    static CUDAError cuda_memset_async(void* dst, int value, size_t bytes, CUDAStream stream) {
        return musaMemsetAsync(dst, value, bytes, stream);
    }
    static CUDAError cuda_copy_async(void* dst, const void* src, size_t bytes, CUDAStream stream) {
        return musaMemcpyAsync(dst, src, bytes, musaMemcpyDeviceToDevice, stream);
    }
    template <typename Kernel>
    static CUDAError cuda_set_shared_memory(Kernel kernel, int bytes) {
        return musaFuncSetAttribute(kernel, musaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    }
#else
    using CUDAError                          = cudaError_t;
    using CUDAStream                         = cudaStream_t;
    static constexpr auto cuda_success       = cudaSuccess;
    static constexpr auto cuda_invalid_value = cudaErrorInvalidValue;
    static constexpr auto cuda_kind          = GGML_EXT_GPU_CUDA;
    static CUDAError cuda_last_error() {
        return cudaGetLastError();
    }
    static CUDAError cuda_memset_async(void* dst, int value, size_t bytes, CUDAStream stream) {
        return cudaMemsetAsync(dst, value, bytes, stream);
    }
    static CUDAError cuda_copy_async(void* dst, const void* src, size_t bytes, CUDAStream stream) {
        return cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream);
    }
    template <typename Kernel>
    static CUDAError cuda_set_shared_memory(Kernel kernel, int bytes) {
        return cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    }
#endif
    template <typename T>
    static __device__ __forceinline__ T cuda_shuffle(T value, int lane, int width = 32) {
#if defined(SD_CUDA_VENDOR_HIP)
        return __shfl(value, lane, width);
#else
        return __shfl_sync(0xffffffff, value, lane, width);
#endif
    }
    template <typename T>
    static __device__ __forceinline__ T cuda_shuffle_xor(T value, int lane, int width = 32) {
#if defined(SD_CUDA_VENDOR_HIP)
        return __shfl_xor(value, lane, width);
#else
        return __shfl_xor_sync(0xffffffff, value, lane, width);
#endif
    }
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
