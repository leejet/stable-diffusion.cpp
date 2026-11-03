#ifndef __SD_CORE_CUDA_I8_PROJECTION_H__
#define __SD_CORE_CUDA_I8_PROJECTION_H__

#include <cuda_runtime.h>
#include "ggml-ext.h"

bool sd_cuda_i8_rope_supports(ggml_backend_dev_t device, const ggml_tensor* mat, const ggml_tensor* view, const ggml_tensor* norm, const ggml_tensor* mul, const ggml_tensor* rope);
cudaError_t sd_cuda_i8_rope_launch(cudaStream_t stream, ggml_tensor* const* nodes, void* output, float scale);
void sd_ggml_register_cuda_i8_projection(ggml_backend_dev_t device);

#endif
