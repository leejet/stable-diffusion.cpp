#include <climits>
#include <type_traits>

#include "convrot_kernel.h"
#include "core/ggml_extensions.h"
#include "int8_common.h"

namespace SD_CUDA_NAMESPACE {
    namespace {
        using i8::aligned;
        using i8::blas_supported;
        using i8::warp_reduce_max;

        // GGML stores FP16 parameters, then the Q5 high bits, then the quants.
        template <ggml_type type>
        struct QuantBlock {
            static constexpr int block_size = 32;
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

        template <ggml_type type>
        struct QuantKBlock {
            static_assert(type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K);
            static constexpr int high_bytes = type == GGML_TYPE_Q5_K ? 32 : 0;
            half d, dmin;
            uint8_t scales[12];
            uint8_t quants[high_bytes + 128];

            __device__ __forceinline__ float decode(int i) const {
                const int group = i / 32;
                const int scale = group < 4 ? scales[group] & 63 : (scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4);
                const int min   = group < 4 ? scales[group + 4] & 63 : (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4);
                int value       = (quants[high_bytes + i / 64 * 32 + i % 32] >> (4 * (group % 2))) & 15;
                if constexpr (high_bytes != 0)
                    value |= ((quants[i % 32] >> group) & 1) << 4;
                return (__half2float(d) * scale) * value - __half2float(dmin) * min;
            }
        };

        template <>
        struct QuantKBlock<GGML_TYPE_Q2_K> {
            uint8_t scales[16];
            uint8_t quants[64];
            half d, dmin;

            __device__ __forceinline__ float decode(int i) const {
                const int scale = scales[i / 16];
                const int value = (quants[i / 128 * 32 + i % 32] >> (2 * (i / 32 % 4))) & 3;
                return (__half2float(d) * (scale & 15)) * value - __half2float(dmin) * (scale >> 4);
            }
        };

        template <>
        struct QuantKBlock<GGML_TYPE_Q3_K> {
            uint8_t high[32];
            uint8_t quants[64];
            uint8_t scales[12];
            half d;

            __device__ __forceinline__ float decode(int i) const {
                const int group      = i / 16;
                const int scale_low  = (scales[group % 8] >> (4 * (group / 8))) & 15;
                const int scale_high = (scales[8 + group % 4] >> (2 * (group / 4))) & 3;
                const int scale      = (scale_low | (scale_high << 4)) - 32;
                const int low        = (quants[i / 128 * 32 + i % 32] >> (2 * (i / 32 % 4))) & 3;
                const int value      = low - ((high[i % 32] >> (i / 32)) & 1 ? 0 : 4);
                return (__half2float(d) * scale) * value;
            }
        };

        template <>
        struct QuantKBlock<GGML_TYPE_Q6_K> {
            uint8_t low[128];
            uint8_t high[64];
            int8_t scales[16];
            half d;

            __device__ __forceinline__ float decode(int i) const {
                const int low_bits  = (low[i / 128 * 64 + i % 64] >> (4 * (i / 64 % 2))) & 15;
                const int high_bits = (high[i / 128 * 32 + i % 32] >> (2 * (i / 32 % 4))) & 3;
                const int value     = (low_bits | (high_bits << 4)) - 32;
                return (__half2float(d) * scales[i / 16]) * value;
            }
        };

        template <>
        struct QuantKBlock<GGML_TYPE_Q8_K> {
            float d;
            int8_t quants[256];
            int16_t sums[16];

            __device__ __forceinline__ float decode(int i) const {
                return d * quants[i];
            }
        };

        constexpr int lanes             = 32;
        constexpr int requantize_warps  = 8;
        constexpr int group_size        = 256;
        constexpr int weights_per_round = requantize_warps * group_size;

        template <ggml_type type>
        using WeightBlock = std::conditional_t<sd_ggml_requantize_i8_block(type).size == group_size, QuantKBlock<type>, QuantBlock<type>>;

        static __device__ __forceinline__ int8_t requantize_i8_value(float value) {
            return int8_t(__float2int_rn(fminf(127.f, fmaxf(-127.f, value))));
        }

        static __device__ __forceinline__ float h256_signed(float value, int i) {
            return (__popc(i & (i >> 1) & 0x55) & 1) ? -value : value;
        }

        static __device__ __forceinline__ void rotate_k_group(float (&values)[8], int lane) {
            // H4 is a Walsh-Hadamard transform with sign changes on index 3 and
            // a swap of the two index bits. Each lane owns eight adjacent weights
            // so the first three stages are local and the result can be stored as a pair of words.
#pragma unroll
            for (int stride = 1; stride < 8; stride *= 2) {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    if ((j & stride) == 0) {
                        const float a      = values[j];
                        const float b      = values[j + stride];
                        values[j]          = a + b;
                        values[j + stride] = a - b;
                    }
                }
            }
#pragma unroll
            for (int stride = 1; stride < lanes; stride *= 2) {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float value = values[j];
                    const float other = cuda_shuffle_xor(value, stride, lanes);
                    values[j]         = (lane & stride ? -value : value) + other;
                }
            }
        }

        template <ggml_type type, int block_size = sd_ggml_requantize_i8_block(type).size>
        static __device__ __forceinline__ void load_rotated_group(const WeightBlock<type>* blocks, int lane, float (&values)[8]) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                if constexpr (cuda_kind == GGML_EXT_GPU_CUDA && block_size == group_size) {
                    const int i      = lane * 8 + j;
                    const int source = ((i & 0x55) << 1) | ((i & 0xAA) >> 1);
                    values[j]        = h256_signed(blocks[0].decode(source) * (1.0f / 16.0f), i);
                } else {
                    const int i = j * lanes + lane;
                    values[j] = blocks[i / block_size].decode(i % block_size) * (1.0f / 16.0f);
                }
            }
            if constexpr (cuda_kind == GGML_EXT_GPU_CUDA && block_size == group_size) {
                rotate_k_group(values, lane);
            } else {
                sd_convrot::rotate_h256(values, lane);
            }
        }

        template <ggml_type type, int cached_groups, int block_size = sd_ggml_requantize_i8_block(type).size>
        static __global__ void requantize_i8_weights(const WeightBlock<type>* weights, int8_t* decoded, float* scales, int64_t blocks_per_channel) {
            constexpr int blocks_per_group = group_size / block_size;
            constexpr int blocks_per_round = requantize_warps * blocks_per_group;
            const int lane                 = threadIdx.x;
            const int warp                 = threadIdx.y;
            const auto* blocks             = weights + int64_t(blockIdx.x) * blocks_per_channel;
            auto* output                   = decoded + int64_t(blockIdx.x) * blocks_per_channel * block_size;

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
                load_rotated_group<type>(blocks + block, lane, values);
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
                    load_rotated_group<type>(blocks + block, lane, values);
                const float* rotated = cached_groups > 0 ? retained[slot] : values;
                if constexpr (cuda_kind == GGML_EXT_GPU_CUDA && block_size == group_size) {
                    uint2 packed = {};
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const auto lower = requantize_i8_value(h256_signed(rotated[j], lane * 8 + j) * inverse);
                        const auto upper = requantize_i8_value(h256_signed(rotated[j + 4], lane * 8 + j + 4) * inverse);
                        packed.x |= uint32_t(uint8_t(lower)) << (8 * j);
                        packed.y |= uint32_t(uint8_t(upper)) << (8 * j);
                    }
                    reinterpret_cast<uint2*>(output + block * block_size)[lane] = packed;
                } else {
#pragma unroll
                    for (int j = 0; j < 8; ++j)
                        output[block * block_size + j * lanes + lane] = requantize_i8_value(rotated[j] * inverse);
                }
            }
        }

        static bool requantize_i8_supports(ggml_backend_dev_t device, const ggml_tensor* node, void*) {
            const auto* weight = node->src[0];
            const int64_t k = weight->ne[0], n = weight->ne[1];
            const auto info        = cuda_device_info(device);
            const size_t alignment = weight->type == GGML_TYPE_Q8_K ? alignof(float) : alignof(uint16_t);
            const bool packed      = cuda_kind == GGML_EXT_GPU_CUDA && sd_ggml_requantize_i8_block(weight->type).size == group_size;
            return info.device >= 0 && k <= INT_MAX / 16384 && n <= INT_MAX && aligned(weight, alignment) &&
                   (!packed || aligned(node, alignof(uint2))) &&
                   (cuda_kind != GGML_EXT_GPU_CUDA || blas_supported(device, info, n));
        }

        template <ggml_type type>
        static void launch_requantize(const ggml_tensor* weight, int8_t* values, CUDAStream stream) {
            using Block          = WeightBlock<type>;
            constexpr auto block = sd_ggml_requantize_i8_block(type);
            static_assert(sizeof(Block) == block.bytes);
            static_assert(alignof(Block) == (type == GGML_TYPE_Q8_K ? alignof(float) : alignof(uint16_t)));
            const int64_t k = weight->ne[0], n = weight->ne[1];
            auto* scales        = reinterpret_cast<float*>(values + k * n);
            const auto* weights = static_cast<const Block*>(weight->data);
            const dim3 threads(lanes, requantize_warps);
            const int64_t blocks_per_channel = k / block.size;
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
                case GGML_TYPE_Q2_K:
                    launch_requantize<GGML_TYPE_Q2_K>(weight, values, stream);
                    break;
                case GGML_TYPE_Q3_K:
                    launch_requantize<GGML_TYPE_Q3_K>(weight, values, stream);
                    break;
                case GGML_TYPE_Q4_K:
                    launch_requantize<GGML_TYPE_Q4_K>(weight, values, stream);
                    break;
                case GGML_TYPE_Q5_K:
                    launch_requantize<GGML_TYPE_Q5_K>(weight, values, stream);
                    break;
                case GGML_TYPE_Q6_K:
                    launch_requantize<GGML_TYPE_Q6_K>(weight, values, stream);
                    break;
                case GGML_TYPE_Q8_K:
                    launch_requantize<GGML_TYPE_Q8_K>(weight, values, stream);
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
