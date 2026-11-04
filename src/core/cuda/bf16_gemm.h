#ifndef __SD_CORE_CUDA_BF16_GEMM_H__
#define __SD_CORE_CUDA_BF16_GEMM_H__

#include <cuda_runtime.h>
#include <cstdint>

enum class SDBF16GemmResult { Unsupported,
                              Success,
                              Failed };

SDBF16GemmResult sd_cuda_bf16_gemm(const void* weight, const void* input, float* output, int64_t rows, int64_t n, int64_t k, cudaStream_t stream);

#endif
