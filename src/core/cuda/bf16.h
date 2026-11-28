#ifndef __SD_CORE_CUDA_BF16_H__
#define __SD_CORE_CUDA_BF16_H__

#include "runtime.h"
#if defined(SD_CUDA_VENDOR_HIP)
#include <hip/hip_bf16.h>
#elif defined(SD_CUDA_VENDOR_MUSA)
#include <musa_bf16.h>
#else
#include <cuda_bf16.h>
#endif

namespace SD_CUDA_NAMESPACE {
#if defined(SD_CUDA_VENDOR_HIP)
    using CUDABFloat16  = __hip_bfloat16;
    using CUDABFloat162 = __hip_bfloat162;
#elif defined(SD_CUDA_VENDOR_MUSA)
    using CUDABFloat16  = __mt_bfloat16;
    using CUDABFloat162 = __mt_bfloat162;
#else
    using CUDABFloat16  = nv_bfloat16;
    using CUDABFloat162 = nv_bfloat162;
#endif
}

#endif
