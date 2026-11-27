#include <climits>

#include "convrot_kernel.h"
#include "core/ggml_extensions.h"
#include "int8_common.h"

namespace SD_CUDA_NAMESPACE {
    namespace {
        using i8::aligned;
        using i8::blas_supported;
        using i8::warp_reduce_max;

        // Match the GGML Q8_0 block layout checked by requantize_q8_supports().
        struct Q8Block {
            static constexpr int size  = 32;
            static constexpr int pairs = size / 2;
            half scale;
            uint16_t values[pairs];
        };
        static_assert(sizeof(Q8Block) == 34);

        static __device__ __forceinline__ int8_t requantize_q8_value(float value) {
            return int8_t(__float2int_rn(fminf(127.f, fmaxf(-127.f, value))));
        }

        static __device__ __forceinline__ void load_rotated_group(const Q8Block* blocks, int lane, float (&values)[8]) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const auto& block  = blocks[j];
                const int8_t value = int8_t(block.values[lane / 2] >> (8 * (lane % 2)));
                values[j]          = (float(value) * __half2float(block.scale)) * (1.0f / 16.0f);
            }
            sd_convrot::rotate_h256(values, lane);
        }

        constexpr int requantize_warps  = 8;
        constexpr int blocks_per_group  = 256 / Q8Block::size;
        constexpr int blocks_per_round  = requantize_warps * blocks_per_group;
        constexpr int weights_per_round = blocks_per_round * Q8Block::size;

        template <int cached_groups>
        static __global__ void requantize_q8_weights(const Q8Block* weights, int8_t* decoded, float* scales, int64_t blocks_per_channel) {
            const int lane     = threadIdx.x;
            const int warp     = threadIdx.y;
            const auto* blocks = weights + int64_t(blockIdx.x) * blocks_per_channel;
            auto* output       = decoded + int64_t(blockIdx.x) * blocks_per_channel * Q8Block::size;

            __shared__ float maxima[requantize_warps];
            // Keep a bounded number of rotated groups in registers across the scale reduction.
            float retained[cached_groups > 0 ? cached_groups : 1][8];
            float maximum = 0.f;
#pragma unroll
            for (int64_t block = warp * blocks_per_group, slot = 0;
                 cached_groups > 0 ? slot < cached_groups : block < blocks_per_channel;
                 block += blocks_per_round, ++slot) {
                if constexpr (cached_groups > 0) {
                    if (block >= blocks_per_channel)
                        break;
                }
                float values[8];
                load_rotated_group(blocks + block, lane, values);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    if constexpr (cached_groups > 0)
                        retained[slot][j] = values[j];
                    maximum = fmaxf(maximum, fabsf(values[j]));
                }
            }
            maximum = warp_reduce_max(maximum);
            if (lane == 0)
                maxima[warp] = maximum;
            __syncthreads();
            if (warp == 0) {
                const float channel_max = warp_reduce_max(lane < requantize_warps ? maxima[lane] : 0.f);
                if (lane == 0) {
#if defined(SD_CUDA_VENDOR_MUSA)
                    scales[blockIdx.x] = channel_max / 127.f;
                    maxima[0]          = channel_max > 0.f ? 127.f / channel_max : 0.f;
#else
                    scales[blockIdx.x] = __fdiv_rn(channel_max, 127.f);
                    maxima[0]          = channel_max > 0.f ? __fdiv_rn(127.f, channel_max) : 0.f;
#endif
                }
            }
            __syncthreads();

            const float inverse = maxima[0];
#pragma unroll
            for (int64_t block = warp * blocks_per_group, slot = 0;
                 cached_groups > 0 ? slot < cached_groups : block < blocks_per_channel;
                 block += blocks_per_round, ++slot) {
                if constexpr (cached_groups > 0) {
                    if (block >= blocks_per_channel)
                        break;
                }
                float values[8];
                if constexpr (cached_groups == 0)
                    load_rotated_group(blocks + block, lane, values);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float value                          = cached_groups > 0 ? retained[slot][j] : values[j];
                    output[(block + j) * Q8Block::size + lane] = requantize_q8_value(value * inverse);
                }
            }
        }

        template <int cached_groups>
        static void launch_requantize(const Q8Block* weights, int8_t* values, float* scales, int64_t k, int64_t n, CUDAStream stream) {
            requantize_q8_weights<cached_groups><<<unsigned(n), dim3(Q8Block::size, requantize_warps), 0, stream>>>(
                weights, values, scales, k / Q8Block::size);
        }

        static bool requantize_q8_supports(ggml_backend_dev_t device, const ggml_tensor* node, void*) {
            const auto* weight = node->src[0];
            const int64_t k = weight->ne[0], n = weight->ne[1];
            const auto info = cuda_device_info(device);
            return info.device >= 0 && k <= INT_MAX / 16384 && n <= INT_MAX && aligned(weight, alignof(Q8Block)) &&
                   ggml_blck_size(GGML_TYPE_Q8_0) == Q8Block::size && ggml_type_size(GGML_TYPE_Q8_0) == sizeof(Q8Block) &&
                   (cuda_kind != GGML_EXT_GPU_CUDA || blas_supported(device, info, n));
        }

        static ggml_status requantize_q8_execute(const ggml_ext_execution_context* execution, ggml_tensor* const* nodes, int, void*, void*) {
            ggml_ext_gpu_context context;
            if (!cuda_context(execution->backend, &context))
                return GGML_STATUS_FAILED;
            auto* dst          = nodes[0];
            const auto* weight = dst->src[0];
            const int64_t k = weight->ne[0], n = weight->ne[1];
            auto* values        = static_cast<int8_t*>(dst->data);
            auto* scales        = reinterpret_cast<float*>(values + k * n);
            const auto* weights = static_cast<const Q8Block*>(weight->data);
            const auto stream   = static_cast<CUDAStream>(context.stream);
            if (cuda_kind == GGML_EXT_GPU_CUDA && k <= 2 * weights_per_round) {
                launch_requantize<2>(weights, values, scales, k, n, stream);
            } else if (cuda_kind == GGML_EXT_GPU_CUDA && k <= 4 * weights_per_round) {
                launch_requantize<4>(weights, values, scales, k, n, stream);
            } else if (cuda_kind == GGML_EXT_GPU_CUDA && k <= 8 * weights_per_round) {
                launch_requantize<8>(weights, values, scales, k, n, stream);
            } else {
                launch_requantize<0>(weights, values, scales, k, n, stream);
            }
            return cuda_last_error() == cuda_success ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
        }
    }

    void register_requantize(ggml_backend_dev_t device) {
        ggml_ext_kernel_desc desc = {};
        desc.abi_version          = GGML_EXT_ABI_VERSION;
        desc.op                   = GGML_OP_CUSTOM;
        desc.name                 = "sd.requantize_q8." SD_CUDA_NAME ".v1";
        desc.custom_op            = sd_ggml_requantize_q8_op();
        desc.supports             = requantize_q8_supports;
        desc.execute              = requantize_q8_execute;
        GGML_ASSERT(ggml_ext_register_kernel_ex(device, &desc, GGML_EXT_KERNEL_CAPTURE_SAFE | GGML_EXT_KERNEL_REPLAY_REUSABLE));
    }
}
