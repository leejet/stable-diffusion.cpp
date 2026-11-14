#ifndef __SD_CORE_CUDA_HIP_I8_GEMM_H__
#define __SD_CORE_CUDA_HIP_I8_GEMM_H__

#include <hip/hip_runtime_api.h>
#include <hipblas/hipblas.h>
#include <cstdint>

namespace sd_hip {
    struct HipI8Gemm {
        int n, rows, k;
        const void* weight;
        const int8_t* input;
        int32_t* output;
        hipblasHandle_t blas;
        hipStream_t stream;

        hipblasStatus_t run_hipblas() const {
#if HIP_VERSION >= 60500000
            constexpr auto compute_type = HIPBLAS_COMPUTE_32I;
#else
            constexpr auto compute_type = HIPBLAS_R_32I;
#endif
            const int32_t alpha = 1, beta = 0;
            return hipblasGemmEx(blas, HIPBLAS_OP_T, HIPBLAS_OP_N, n, rows, k, &alpha,
                                 weight, HIPBLAS_R_8I, k, input, HIPBLAS_R_8I, k,
                                 &beta, output, HIPBLAS_R_32I, n, compute_type, HIPBLAS_GEMM_DEFAULT);
        }
    };

#ifdef SD_HIP_I8_LT
    bool hip_i8_gemm_lt(int device, const HipI8Gemm& gemm);
#endif
}

#endif
