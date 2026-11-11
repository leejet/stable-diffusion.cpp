#ifndef __SD_CORE_CUDA_I8_GEMM_H__
#define __SD_CORE_CUDA_I8_GEMM_H__
#include <cuda_runtime.h>
#include "ggml-ext.h"
struct SDI8CudaGemmPlan;
SDI8CudaGemmPlan* sd_cuda_i8_gemm_prepare(ggml_backend_dev_t device, int64_t rows, int64_t n, int64_t k, bool swiglu, size_t* workspace, bool packed_w4 = false);
ggml_status sd_cuda_i8_gemm_execute(SDI8CudaGemmPlan* plan, const int8_t* x, const int8_t* w, const float* xs, const float* ws, float* out, void* scratch, cudaStream_t stream, const float* bias = nullptr);
// n is the width of each branch; gate/up weights and scales remain separate.
SDI8CudaGemmPlan* sd_cuda_i8_gemm_prepare_split(ggml_backend_dev_t device, int64_t rows, int64_t n, int64_t k, size_t* workspace);
ggml_status sd_cuda_i8_gemm_execute_split(SDI8CudaGemmPlan* plan, const int8_t* x, const int8_t* gate, const int8_t* up, const float* xs, const float* gate_scales, const float* up_scales, float* out, void* scratch, cudaStream_t stream);
void sd_cuda_i8_gemm_free(SDI8CudaGemmPlan* plan);
#endif
