#include <algorithm>
#include <cmath>

#include "convrot_kernel.h"
#include "core/ggml_extensions.h"

namespace SD_CUDA_NAMESPACE {
    static constexpr int lora_threads = 256;

    static __global__ void combine_lora_weights(const int8_t* weight, const float* scale, const float* diff, float* merged, int64_t k, int64_t count, int group, bool scalar) {
        __shared__ float values[lora_threads];
        const int tid             = threadIdx.x;
        const float normalization = group > 1 ? rsqrtf(float(group)) : 1.f;
        for (int64_t base = int64_t(blockIdx.x) * blockDim.x; base < count; base += int64_t(gridDim.x) * blockDim.x) {
            const int64_t i = base + tid;
            float value     = i < count ? diff[i] : 0.f;
            if (group > 1) {
                values[tid] = value * normalization;
                __syncthreads();
                for (int stride = 1; stride < group; stride *= 4) {
                    const int j = tid / (4 * stride) * (4 * stride) + tid % stride;
                    value       = sd_convrot::convrot_h4_cuda(values[j], values[j + stride], values[j + 2 * stride], values[j + 3 * stride], (tid / stride) % 4);
                    __syncthreads();
                    values[tid] = value;
                    __syncthreads();
                }
            }
            if (i < count)
                merged[i] = __fadd_rn(value, __fmul_rn(float(weight[i]), scale[scalar ? 0 : i / k]));
        }
    }

    static __device__ SDI8LoRARowStats reduce_lora_stats(float maximum, uint32_t flags) {
        __shared__ SDI8LoRARowStats shared[lora_threads];
        const int tid = threadIdx.x;
        shared[tid]   = {maximum, flags};
        __syncthreads();
        for (int stride = lora_threads / 2; stride > 0; stride /= 2) {
            if (tid < stride) {
                shared[tid].maximum = fmaxf(shared[tid].maximum, shared[tid + stride].maximum);
                shared[tid].changed |= shared[tid + stride].changed;
            }
            __syncthreads();
        }
        const auto result = shared[0];
        __syncthreads();
        return result;
    }

    static __global__ void lora_row_stats(const float* merged, const float* diff, SDI8LoRARowStats* stats, int64_t k, int64_t rows) {
        for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
            float maximum    = 0.f;
            uint32_t changed = 0;
            for (int64_t col = threadIdx.x; col < k; col += blockDim.x) {
                const int64_t i = row * k + col;
                maximum         = fmaxf(maximum, isfinite(merged[i]) ? fabsf(merged[i]) : INFINITY);
                changed |= diff[i] != 0.f;
            }
            const auto result = reduce_lora_stats(maximum, changed);
            if (threadIdx.x == 0)
                stats[row] = result;
        }
    }

    static __global__ void lora_merge_summary(const SDI8LoRARowStats* stats, uint32_t* flags, float* scales, int64_t rows, bool scalar) {
        float maximum    = 0.f;
        uint32_t changed = 0;
        for (int64_t row = threadIdx.x; row < rows; row += blockDim.x) {
            const auto info = stats[row];
            if (scalar || info.changed)
                maximum = fmaxf(maximum, info.maximum);
            changed |= info.changed;
        }
        const auto result = reduce_lora_stats(maximum, changed);
        if (threadIdx.x == 0) {
            *flags = !result.changed ? 0 : SD_I8_LORA_CHANGED | (isfinite(result.maximum) ? 0 : SD_I8_LORA_NONFINITE);
            if (scalar)
                scales[0] = result.maximum / 127.f;
        }
    }

    static __global__ void quantize_lora_weights(const int8_t* weight, const float* old_scales, const float* merged, const SDI8LoRARowStats* stats, const uint32_t* flags, float* scales, int8_t* output, int64_t k, int64_t rows, bool scalar) {
        const bool apply = *flags == SD_I8_LORA_CHANGED;
        for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
            const bool changed = apply && (scalar || stats[row].changed);
            const float scale  = changed ? (scalar ? scales[0] : stats[row].maximum / 127.f) : old_scales[scalar ? 0 : row];
            if (threadIdx.x == 0 && (!scalar || (!apply && row == 0)))
                scales[scalar ? 0 : row] = scale;
            for (int64_t col = threadIdx.x; col < k; col += blockDim.x) {
                const int64_t i = row * k + col;
                if (changed) {
                    const int value = scale == 0.f ? 0 : __float2int_rn(merged[i] / scale);
                    output[i]       = static_cast<int8_t>(max(-127, min(127, value)));
                } else {
                    output[i] = weight[i];
                }
            }
        }
    }

    void register_i8_lora_merge(ggml_backend_dev_t device) {
        ggml_ext_kernel_desc desc = {};
        desc.abi_version          = GGML_EXT_ABI_VERSION;
        desc.name                 = "sd.i8_lora_merge." SD_CUDA_NAME ".v1";
        desc.op                   = GGML_OP_CUSTOM;
        desc.custom_op            = sd_ggml_i8_lora_merge_op();
        desc.priority             = 100;
        desc.supports             = [](ggml_backend_dev_t, const ggml_tensor* node, void*) {
            SDI8LoRAMerge op;
            return sd_ggml_i8_lora_merge_describe(node, &op);
        };
        desc.prepare = [](ggml_backend_t, ggml_tensor* const* nodes, int, void**, size_t* bytes, void*) {
            SDI8LoRAMerge op;
            if (!sd_ggml_i8_lora_merge_describe(nodes[0], &op))
                return GGML_STATUS_FAILED;
            *bytes = op.workspace_size();
            return GGML_STATUS_SUCCESS;
        };
        desc.execute = [](const ggml_ext_execution_context* e, ggml_tensor* const* nodes, int, void*, void*) {
            SDI8LoRAMerge op;
            ggml_ext_gpu_context gpu;
            if (!cuda_context(e->backend, &gpu) || !sd_ggml_i8_lora_merge_describe(nodes[0], &op))
                return GGML_STATUS_FAILED;
            auto stream               = static_cast<CUDAStream>(gpu.stream);
            auto* scratch             = static_cast<char*>(ggml_backend_buffer_get_base(e->workspace));
            auto* packed              = static_cast<char*>(nodes[0]->data);
            auto* merged              = reinterpret_cast<float*>(scratch);
            auto* stats               = reinterpret_cast<SDI8LoRARowStats*>(scratch + ggml_nbytes(op.diff));
            auto* flags               = reinterpret_cast<uint32_t*>(packed);
            auto* scales              = reinterpret_cast<float*>(packed + op.scale_offset());
            auto* output              = reinterpret_cast<int8_t*>(packed + op.weight_offset());
            const auto* base          = static_cast<const int8_t*>(op.weight->data);
            const auto* old           = static_cast<const float*>(op.scale->data);
            const auto* diff          = static_cast<const float*>(op.diff->data);
            const auto k              = op.weight->ne[0];
            const auto rows           = op.weight->ne[1];
            const auto count          = ggml_nelements(op.weight);
            const auto scalar         = op.scalar_scale();
            const unsigned row_blocks = unsigned(std::min<int64_t>(rows, 65535));
            const unsigned blocks     = unsigned(std::min<int64_t>((count + lora_threads - 1) / lora_threads, 65535));
            combine_lora_weights<<<blocks, lora_threads, 0, stream>>>(base, old, diff, merged, k, count, op.group, scalar);
            lora_row_stats<<<row_blocks, lora_threads, 0, stream>>>(merged, diff, stats, k, rows);
            lora_merge_summary<<<1, lora_threads, 0, stream>>>(stats, flags, scales, rows, scalar);
            quantize_lora_weights<<<row_blocks, lora_threads, 0, stream>>>(base, old, merged, stats, flags, scales, output, k, rows, scalar);
            return cuda_last_error() == cuda_success ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
        };
        desc.destroy = [](void*, void*) {};
        GGML_ASSERT(ggml_ext_register_kernel_ex(device, &desc, GGML_EXT_KERNEL_CAPTURE_SAFE | GGML_EXT_KERNEL_REPLAY_REUSABLE));
    }
}
