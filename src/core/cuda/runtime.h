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
    static __device__ __forceinline__ int32_t cuda_dot_i8x4(int32_t a, int32_t b, int32_t acc) {
#if defined(SD_CUDA_VENDOR_HIP) && (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1200__) || defined(__gfx1201__))
        return __builtin_amdgcn_sudot4(true, a, true, b, acc, false);
#elif defined(SD_CUDA_VENDOR_HIP) && (defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || defined(__gfx1033__) || defined(__gfx1034__) || defined(__gfx1035__) || defined(__gfx1036__) || defined(__gfx906__) || defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx942__))
        return __builtin_amdgcn_sdot4(a, b, acc, false);
#elif defined(SD_CUDA_VENDOR_MUSA) || (!defined(SD_CUDA_VENDOR_HIP) && __CUDA_ARCH__ >= 610)
        return __dp4a(a, b, acc);
#else
        uint32_t sum   = uint32_t(acc);
        const auto* av = reinterpret_cast<const int8_t*>(&a);
        const auto* bv = reinterpret_cast<const int8_t*>(&b);
        for (int i = 0; i < 4; ++i)
            sum += uint32_t(int(av[i]) * int(bv[i]));
        return int32_t(sum);
#endif
    }

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
