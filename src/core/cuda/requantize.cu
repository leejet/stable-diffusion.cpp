#include <climits>

#include "convrot_kernel.h"
#include "core/ggml_extensions.h"
#include "int8_common.h"

namespace SD_CUDA_NAMESPACE {
    namespace {
        using i8::aligned;
        using i8::blas_supported;
        using i8::warp_reduce_max;

        constexpr int block_size = SD_REQUANTIZE_I8_BLOCK_SIZE;

        // GGML stores FP16 parameters, then the Q5 high bits, then the quants.
        template <ggml_type type>
        struct QuantBlock {
            static constexpr int bits       = type == GGML_TYPE_Q8_0 ? 8 : (type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1 ? 5 : 4);
            static constexpr bool affine    = type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_1;
            static constexpr int high_words = bits == 5 ? 2 : 0;
            half params[affine ? 2 : 1];
            uint16_t payload[(bits == 8 ? block_size / 2 : block_size / 4) + high_words];

            __device__ __forceinline__ float decode(int lane) const {
                const float scale = __half2float(params[0]);
                if constexpr (bits == 8) {
                    const int8_t value = int8_t(payload[lane / 2] >> (8 * (lane % 2)));
                    return float(value) * scale;
                } else {
                    const int low_lane    = lane % 16;
                    const uint16_t packed = payload[high_words + low_lane / 2];
                    int value             = (packed >> (8 * (low_lane % 2) + 4 * (lane / 16))) & 15;
                    if constexpr (bits == 5)
                        value |= ((payload[lane / 16] >> low_lane) & 1) << 4;
                    if constexpr (affine)
                        return float(value) * scale + __half2float(params[1]);
                    return float(value - (1 << (bits - 1))) * scale;
                }
            }
        };

        static __device__ __forceinline__ int8_t requantize_i8_value(float value) {
            return int8_t(__float2int_rn(fminf(127.f, fmaxf(-127.f, value))));
        }

        template <ggml_type type>
        static __device__ __forceinline__ void load_rotated_group(const QuantBlock<type>* blocks, int lane, float (&values)[8]) {
#pragma unroll
            for (int j = 0; j < 8; ++j)
                values[j] = blocks[j].decode(lane) * (1.0f / 16.0f);
            sd_convrot::rotate_h256(values, lane);
        }

        constexpr int requantize_warps  = 8;
        constexpr int blocks_per_group  = 256 / block_size;
        constexpr int blocks_per_round  = requantize_warps * blocks_per_group;
        constexpr int weights_per_round = blocks_per_round * block_size;

        template <ggml_type type, int cached_groups>
        static __global__ void requantize_i8_weights(const QuantBlock<type>* weights, int8_t* decoded, float* scales, int64_t blocks_per_channel) {
            const int lane     = threadIdx.x;
            const int warp     = threadIdx.y;
            const auto* blocks = weights + int64_t(blockIdx.x) * blocks_per_channel;
            auto* output       = decoded + int64_t(blockIdx.x) * blocks_per_channel * block_size;

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
                    const float value                       = cached_groups > 0 ? retained[slot][j] : values[j];
                    output[(block + j) * block_size + lane] = requantize_i8_value(value * inverse);
                }
            }
        }

        static bool requantize_i8_supports(ggml_backend_dev_t device, const ggml_tensor* node, void*) {
            const auto* weight = node->src[0];
            const int64_t k = weight->ne[0], n = weight->ne[1];
            const auto info = cuda_device_info(device);
            return info.device >= 0 && k <= INT_MAX / 16384 && n <= INT_MAX && aligned(weight, alignof(uint16_t)) &&
                   (cuda_kind != GGML_EXT_GPU_CUDA || blas_supported(device, info, n));
        }

        template <ggml_type type>
        static void launch_requantize(const ggml_tensor* weight, int8_t* values, CUDAStream stream) {
            using Block = QuantBlock<type>;
            static_assert(sizeof(Block) == sd_ggml_requantize_i8_block_bytes(type) && alignof(Block) == alignof(uint16_t));
            const int64_t k = weight->ne[0], n = weight->ne[1];
            auto* scales        = reinterpret_cast<float*>(values + k * n);
            const auto* weights = static_cast<const Block*>(weight->data);
            const dim3 threads(block_size, requantize_warps);
            const int64_t blocks_per_channel = k / block_size;
            if (cuda_kind == GGML_EXT_GPU_CUDA && k <= 2 * weights_per_round) {
                requantize_i8_weights<type, 2><<<unsigned(n), threads, 0, stream>>>(weights, values, scales, blocks_per_channel);
            } else if (cuda_kind == GGML_EXT_GPU_CUDA && k <= 4 * weights_per_round) {
                requantize_i8_weights<type, 4><<<unsigned(n), threads, 0, stream>>>(weights, values, scales, blocks_per_channel);
            } else if (cuda_kind == GGML_EXT_GPU_CUDA && k <= 8 * weights_per_round) {
                requantize_i8_weights<type, 8><<<unsigned(n), threads, 0, stream>>>(weights, values, scales, blocks_per_channel);
            } else {
                requantize_i8_weights<type, 0><<<unsigned(n), threads, 0, stream>>>(weights, values, scales, blocks_per_channel);
            }
        }

        static ggml_status requantize_i8_execute(const ggml_ext_execution_context* execution, ggml_tensor* const* nodes, int, void*, void*) {
            ggml_ext_gpu_context context;
            if (!cuda_context(execution->backend, &context))
                return GGML_STATUS_FAILED;
            auto* dst          = nodes[0];
            const auto* weight = dst->src[0];
            auto* values       = static_cast<int8_t*>(dst->data);
            const auto stream  = static_cast<CUDAStream>(context.stream);
            switch (weight->type) {
                case GGML_TYPE_Q4_0:
                    launch_requantize<GGML_TYPE_Q4_0>(weight, values, stream);
                    break;
                case GGML_TYPE_Q4_1:
                    launch_requantize<GGML_TYPE_Q4_1>(weight, values, stream);
                    break;
                case GGML_TYPE_Q5_0:
                    launch_requantize<GGML_TYPE_Q5_0>(weight, values, stream);
                    break;
                case GGML_TYPE_Q5_1:
                    launch_requantize<GGML_TYPE_Q5_1>(weight, values, stream);
                    break;
                case GGML_TYPE_Q8_0:
                    launch_requantize<GGML_TYPE_Q8_0>(weight, values, stream);
                    break;
                default:
                    return GGML_STATUS_FAILED;
            }
            return cuda_last_error() == cuda_success ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
        }
    }

    void register_requantize(ggml_backend_dev_t device) {
        ggml_ext_kernel_desc desc = {};
        desc.abi_version          = GGML_EXT_ABI_VERSION;
        desc.op                   = GGML_OP_CUSTOM;
        desc.name                 = "sd.requantize_i8." SD_CUDA_NAME ".v1";
        desc.custom_op            = sd_ggml_requantize_i8_op();
        desc.supports             = requantize_i8_supports;
        desc.execute              = requantize_i8_execute;
        GGML_ASSERT(ggml_ext_register_kernel_ex(device, &desc, GGML_EXT_KERNEL_CAPTURE_SAFE | GGML_EXT_KERNEL_REPLAY_REUSABLE));
    }
}
