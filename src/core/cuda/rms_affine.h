#ifndef __SD_CORE_CUDA_RMS_AFFINE_H__
#define __SD_CORE_CUDA_RMS_AFFINE_H__

#include "runtime.h"

namespace SD_CUDA_NAMESPACE {
    int rms_affine_match(const ggml_cgraph* graph, int first);
    ggml_status rms_affine_pack(CUDAStream stream, ggml_tensor* const* nodes, void* output);
}

#endif
