#ifndef __SD_CORE_CUDA_ROPE_H__
#define __SD_CORE_CUDA_ROPE_H__

#include <cuda_runtime.h>
#include "ggml-ext.h"

bool sd_cuda_rope_is_node(const ggml_tensor* node);
bool sd_cuda_rope_fusion_supports(const ggml_tensor* norm, const ggml_tensor* mul, const ggml_tensor* rope);
cudaError_t sd_cuda_rms_rope_launch(cudaStream_t stream, ggml_tensor* const* nodes, void* output, bool fp16, float scale);
void sd_ggml_register_cuda_rope(ggml_backend_dev_t device);

#endif
