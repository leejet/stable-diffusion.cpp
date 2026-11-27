#ifndef __SD_CORE_CUDA_ROPE_H__
#define __SD_CORE_CUDA_ROPE_H__

#include "runtime.h"

namespace SD_CUDA_NAMESPACE {
    bool rope_fusion_supports(const ggml_tensor* norm, const ggml_tensor* mul, const ggml_tensor* rope);
    CUDAError rms_rope_launch(CUDAStream stream, ggml_tensor* const* nodes, void* output, bool fp16, float scale);
    bool hidden_rope_fusion_supports(const ggml_tensor* norm, const ggml_tensor* mul, const ggml_tensor* reshape, const ggml_tensor* rope);
    bool hidden_rope_view_of(const ggml_tensor* view, const ggml_tensor* source);
    bool hidden_rope_head_major(const ggml_tensor* view, const ggml_tensor* norm, const ggml_tensor* rope);
    CUDAError hidden_rope_launch(CUDAStream stream, ggml_tensor* const* nodes, void* output, bool fp16, float scale);

}
#endif
