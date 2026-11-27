#include <climits>
#include <cstdlib>

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

        static __device__ __forceinline__ uint8_t requantize_q8_value(float value) {
            return uint8_t(__float2int_rn(fminf(127.f, fmaxf(-127.f, value))));
        }

        static __global__ void requantize_q8_weights(const Q8Block* weights, int8_t* decoded, float* scales, int64_t blocks_per_channel) {
            constexpr int warps            = 8;
            constexpr int blocks_per_round = warps * 2;
            const int lane                 = threadIdx.x;
            const int warp                 = threadIdx.y;
            const int pair                 = lane % Q8Block::pairs;
            const int first_block           = warp * 2 + lane / Q8Block::pairs;
            const auto* blocks              = weights + int64_t(blockIdx.x) * blocks_per_channel;
            auto* output                    = reinterpret_cast<uint16_t*>(decoded) + int64_t(blockIdx.x) * blocks_per_channel * Q8Block::pairs;

            __shared__ float maxima[warps];
            float maximum = 0.f;
            for (int64_t i = first_block; i < blocks_per_channel; i += blocks_per_round) {
                const uint16_t values = blocks[i].values[pair];
                const int magnitude   = max(abs(int8_t(values)), abs(int8_t(values >> 8)));
                maximum               = fmaxf(maximum, float(magnitude) * fabsf(__half2float(blocks[i].scale)));
            }
            maximum = warp_reduce_max(maximum);
            if (lane == 0)
                maxima[warp] = maximum;
            __syncthreads();
            if (warp == 0) {
                const float channel_max = warp_reduce_max(lane < warps ? maxima[lane] : 0.f);
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
            for (int64_t i = first_block; i < blocks_per_channel; i += blocks_per_round) {
                const uint16_t values = blocks[i].values[pair];
                const float scale     = __half2float(blocks[i].scale);
                const uint8_t low     = requantize_q8_value((float(int8_t(values)) * scale) * inverse);
                const uint8_t high    = requantize_q8_value((float(int8_t(values >> 8)) * scale) * inverse);
                output[i * Q8Block::pairs + pair] = uint16_t(low) | (uint16_t(high) << 8);
            }
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
            auto* values = static_cast<int8_t*>(dst->data);
            auto* scales = reinterpret_cast<float*>(values + k * n);
            requantize_q8_weights<<<unsigned(n), dim3(Q8Block::size, 8), 0, static_cast<CUDAStream>(context.stream)>>>(
                static_cast<const Q8Block*>(weight->data), values, scales, k / Q8Block::size);
            return cuda_last_error() == cuda_success ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
        }
    }

    void register_requantize(ggml_backend_dev_t device) {
        ggml_ext_kernel_desc desc = {};
        desc.abi_version         = GGML_EXT_ABI_VERSION;
        desc.op                  = GGML_OP_CUSTOM;
        desc.name                = "sd.requantize_q8." SD_CUDA_NAME ".v1";
        desc.custom_op           = sd_ggml_requantize_q8_op();
        desc.supports            = requantize_q8_supports;
        desc.execute             = requantize_q8_execute;
        GGML_ASSERT(ggml_ext_register_kernel_ex(device, &desc, GGML_EXT_KERNEL_CAPTURE_SAFE | GGML_EXT_KERNEL_REPLAY_REUSABLE));
    }
}
