#ifndef __SD_CORE_CUDA_ROPE_H__
#define __SD_CORE_CUDA_ROPE_H__

#include "runtime.h"

namespace SD_CUDA_NAMESPACE {
    bool rope_fusion_supports(const ggml_tensor* norm, const ggml_tensor* mul, const ggml_tensor* rope);
    CUDAError rms_rope_launch(CUDAStream stream, ggml_tensor* const* nodes, void* output, bool fp16, float scale);

}
#endif
