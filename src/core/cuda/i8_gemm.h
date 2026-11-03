#ifndef __SD_CORE_CUDA_I8_GEMM_H__
#define __SD_CORE_CUDA_I8_GEMM_H__
#include <cuda_runtime.h>
#include "ggml-ext.h"
struct SDI8CudaGemmPlan;
SDI8CudaGemmPlan* sd_cuda_i8_gemm_prepare(ggml_backend_dev_t device, int64_t rows, int64_t n, int64_t k, bool swiglu, size_t* workspace);
ggml_status sd_cuda_i8_gemm_execute(SDI8CudaGemmPlan* plan, const int8_t* x, const int8_t* w, const float* xs, const float* ws, float* out, void* scratch, cudaStream_t stream);
void sd_cuda_i8_gemm_free(SDI8CudaGemmPlan* plan);
#endif
