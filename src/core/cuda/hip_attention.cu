#define SD_CUDA_VENDOR_HIP 1
#include "runtime.h"

#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace sd_hip {
    static constexpr int WAVE_SIZE = 32;
    static constexpr int WMMA_TILE = 16;

    template <int HeadDim>
    struct AttentionTile {
        static constexpr int channels    = (HeadDim + WMMA_TILE - 1) / WMMA_TILE * WMMA_TILE;
        static constexpr int keys        = HeadDim > 128 ? 32 : 64;
        static constexpr int query_waves = HeadDim > 128 ? 4 : 8;
        static constexpr int queries     = query_waves * WMMA_TILE;
    };

#if __has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12)
#define SD_HIP_ATTN_WMMA_128B
#elif __has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_f16_w32)
#define SD_HIP_ATTN_WMMA_256B
#endif

#if defined(SD_HIP_ATTN_WMMA_128B) || defined(SD_HIP_ATTN_WMMA_256B)
    using Half8       = _Float16 __attribute__((ext_vector_type(8)));
    using Accumulator = float __attribute__((ext_vector_type(8)));
#ifdef SD_HIP_ATTN_WMMA_256B
    using HalfFragment                     = _Float16 __attribute__((ext_vector_type(16)));
    static constexpr bool replicate_inputs = true;
#else
    using HalfFragment                     = Half8;
    static constexpr bool replicate_inputs = false;
#endif

    static __device__ __forceinline__ HalfFragment load_fragment(const _Float16* ptr) {
        const auto low = *reinterpret_cast<const Half8*>(ptr);
#ifdef SD_HIP_ATTN_WMMA_256B
        // LDS rows are only 16-byte aligned, including when the fragment spans 32 bytes.
        const auto high = *reinterpret_cast<const Half8*>(ptr + 8);
        return __builtin_shufflevector(low, high, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
#else
        return low;
#endif
    }

    static __device__ __forceinline__ Accumulator attention_wmma(HalfFragment a, HalfFragment b, Accumulator acc) {
#ifdef SD_HIP_ATTN_WMMA_256B
        return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
#else
        return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc);
#endif
    }
#endif

    struct AttentionInput {
        const void* data;
        size_t token_stride;
        size_t head_stride;
        size_t batch_stride;

        template <typename T>
        __device__ const T* head(int index, int batch) const {
            return reinterpret_cast<const T*>(static_cast<const char*>(data) + index * head_stride + batch * batch_stride);
        }
    };

    template <int HeadDim>
    static __global__ void pack_attention_values(AttentionInput value, _Float16* packed, int keys) {
        constexpr int channels      = AttentionTile<HeadDim>::channels;
        constexpr int key_tile      = AttentionTile<HeadDim>::keys;
        constexpr int channel_tiles = (HeadDim + 31) / 32;
        // Each block transposes a 32x32 tile. The extra column avoids LDS bank conflicts.
        __shared__ _Float16 tile[32][33];
        const int head         = blockIdx.y;
        const int batch        = blockIdx.z;
        const int token        = (blockIdx.x / channel_tiles) * 32;
        const int channel      = (blockIdx.x % channel_tiles) * 32;
        const int x            = threadIdx.x;
        const int y            = threadIdx.y;
        const auto* input      = value.head<_Float16>(head, batch);
        const size_t stride    = value.token_stride / sizeof(_Float16);
        const size_t head_size = ((size_t(keys) + key_tile - 1) / key_tile) * key_tile * channels;
        auto* output           = packed + (size_t(batch) * gridDim.y + head) * head_size +
                                 size_t(token / key_tile) * key_tile * channels + channel * key_tile + token % key_tile;
#pragma unroll
        for (int row = y; row < 32; row += 8) {
            tile[row][x] = token + row < keys && channel + x < HeadDim ? input[(token + row) * stride + channel + x] : _Float16(0);
        }
        __syncthreads();
#pragma unroll
        for (int row = y; row < 32; row += 8) {
            if (channel + row < channels)
                output[row * key_tile + x] = tile[x][row];
        }
    }

    template <int HeadDim>
    static __global__ __launch_bounds__(AttentionTile<HeadDim>::query_waves* WAVE_SIZE) void wmma_attention(
        AttentionInput query,
        AttentionInput key,
        const _Float16* packed_value,
        float* dst,
        int queries,
        int keys,
        int kv_heads,
        int queries_per_kv,
        float scale) {
#if defined(SD_HIP_ATTN_WMMA_128B) || defined(SD_HIP_ATTN_WMMA_256B)
        constexpr int channels   = AttentionTile<HeadDim>::channels;
        constexpr int key_tile   = AttentionTile<HeadDim>::keys;
        constexpr int query_tile = AttentionTile<HeadDim>::queries;
        const int lane           = threadIdx.x % WMMA_TILE;
        const int half           = (threadIdx.x % WAVE_SIZE) / WMMA_TILE;
        // The 256-bit operand layout duplicates both fragments across the two half-waves
        // and distributes accumulator rows as even and odd indices.
        const int input_offset       = replicate_inputs ? 0 : half * 8;
        const int acc_offset         = replicate_inputs ? half : half * 8;
        constexpr int acc_stride     = replicate_inputs ? 2 : 1;
        const int wave               = threadIdx.x / WAVE_SIZE;
        const int head               = blockIdx.y;
        const int kv_head            = queries_per_kv == 1 ? head : head / queries_per_kv;
        const int batch              = blockIdx.z;
        const int q_index            = blockIdx.x * query_tile + wave * WMMA_TILE + lane;
        const float* q               = query.head<float>(head, batch);
        const _Float16* k            = key.head<_Float16>(kv_head, batch);
        const size_t q_stride        = query.token_stride / sizeof(float);
        const size_t k_stride        = key.token_stride / sizeof(_Float16);
        const size_t value_head_size = ((size_t(keys) + key_tile - 1) / key_tile) * key_tile * channels;
        const _Float16* v            = packed_value + (size_t(batch) * kv_heads + kv_head) * value_head_size;

        __shared__ __align__(16) _Float16 k_tile[key_tile][channels + 8];
        HalfFragment q_fragments[channels / WMMA_TILE] = {};
        Accumulator output[channels / WMMA_TILE]       = {};
        if (q_index < queries) {
#pragma unroll
            for (int d = 0; d < channels / WMMA_TILE; ++d) {
#pragma unroll
                for (int i = 0; i < (replicate_inputs ? 16 : 8); ++i) {
                    const int channel = d * WMMA_TILE + input_offset + i;
                    if (channel < HeadDim)
                        q_fragments[d][i] = q[q_index * q_stride + channel] * scale;
                }
            }
        }

        float maximum = -INFINITY;
        float sum     = 0.f;
        for (int base = 0; base < keys; base += key_tile) {
            const int tile_keys = min(keys - base, key_tile);
            for (int index = threadIdx.x; index < key_tile * channels / 8; index += blockDim.x) {
                const int token   = index / (channels / 8);
                const int channel = (index % (channels / 8)) * 8;
                Half8 k_fragment  = {};
                if (token < tile_keys && channel < HeadDim) {
                    const auto* row = k + (base + token) * k_stride;
                    if constexpr (HeadDim % 8 == 0) {
                        k_fragment = *reinterpret_cast<const Half8*>(row + channel);
                    } else {
#pragma unroll
                        for (int i = 0; i < 8; ++i)
                            if (channel + i < HeadDim)
                                k_fragment[i] = row[channel + i];
                    }
                }
                *reinterpret_cast<Half8*>(&k_tile[token][channel]) = k_fragment;
            }
            __syncthreads();

            // K * Q^T leaves each query in a pair of lanes.
            Accumulator scores[key_tile / WMMA_TILE] = {};
#pragma unroll
            for (int d = 0; d < channels / WMMA_TILE; ++d) {
#pragma unroll
                for (int t = 0; t < key_tile / WMMA_TILE; ++t) {
                    const auto k_fragment = load_fragment(&k_tile[t * WMMA_TILE + lane][d * WMMA_TILE + input_offset]);
                    scores[t]             = attention_wmma(k_fragment, q_fragments[d], scores[t]);
                }
            }

            float tile_max = -INFINITY;
#pragma unroll
            for (int t = 0; t < key_tile / WMMA_TILE; ++t) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    scores[t][i] = t * WMMA_TILE + acc_offset + acc_stride * i < tile_keys ? scores[t][i] : -INFINITY;
                    tile_max     = fmaxf(tile_max, scores[t][i]);
                }
            }
            const float next_max = fmaxf(maximum, fmaxf(tile_max, cuda_shuffle_xor(tile_max, 16)));
            const float rescale  = __expf(maximum - next_max);
            maximum              = next_max;
            float tile_sum       = 0.f;
            HalfFragment probabilities[key_tile / WMMA_TILE];
#pragma unroll
            for (int t = 0; t < key_tile / WMMA_TILE; ++t) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float p = __expf(scores[t][i] - maximum);
                    if constexpr (replicate_inputs) {
                        // Reassemble the 16 probabilities in key order for both
                        // half-waves before using them as the B operand of V^T * P.
                        const float peer            = cuda_shuffle_xor(p, WMMA_TILE);
                        probabilities[t][2 * i]     = half == 0 ? p : peer;
                        probabilities[t][2 * i + 1] = half == 0 ? peer : p;
                    } else {
                        probabilities[t][i] = p;
                    }
                    tile_sum += p;
                }
            }
            sum = fmaf(sum, rescale, tile_sum + cuda_shuffle_xor(tile_sum, 16));
#pragma unroll
            for (int d = 0; d < channels / WMMA_TILE; ++d) {
                output[d] *= rescale;
#pragma unroll
                for (int t = 0; t < key_tile / WMMA_TILE; ++t) {
                    const size_t offset   = size_t(base) * channels + (d * WMMA_TILE + lane) * key_tile + t * WMMA_TILE + input_offset;
                    const auto v_fragment = load_fragment(v + offset);
                    output[d]             = attention_wmma(v_fragment, probabilities[t], output[d]);
                }
            }
            __syncthreads();
        }

        if (q_index < queries) {
            const float inverse = 1.f / sum;
            float* out          = dst + ((size_t(batch) * queries + q_index) * gridDim.y + head) * HeadDim;
#pragma unroll
            for (int d = 0; d < channels / WMMA_TILE; ++d) {
                output[d] *= inverse;
                if constexpr (replicate_inputs || HeadDim % 8 != 0) {
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const int channel = d * WMMA_TILE + acc_offset + acc_stride * i;
                        if (channel < HeadDim)
                            out[channel] = output[d][i];
                    }
                } else {
                    const int channel = d * WMMA_TILE + acc_offset;
                    if (channel < HeadDim)
                        *reinterpret_cast<Accumulator*>(out + channel) = output[d];
                }
            }
        }
#elif defined(__HIP_DEVICE_COMPILE__)
        __builtin_trap();
#endif
    }

    static bool attention_aligned(const ggml_tensor* tensor, size_t alignment) {
        size_t offset = 0;
        while (tensor->view_src) {
            offset += tensor->view_offs;
            tensor = tensor->view_src;
        }
        return (reinterpret_cast<uintptr_t>(tensor->data) + offset) % alignment == 0;
    }

    static bool attention_layout(const ggml_tensor* tensor, ggml_type type, size_t alignment) {
        return tensor->type == type && tensor->nb[0] == ggml_type_size(type) &&
               tensor->nb[1] % alignment == 0 && tensor->nb[2] % alignment == 0 && tensor->nb[3] % alignment == 0 &&
               attention_aligned(tensor, alignment);
    }

    template <int HeadDim>
    static ggml_status launch_attention(const ggml_ext_execution_context* execution, ggml_tensor* const* nodes, CUDAStream stream) {
        using Tile = AttentionTile<HeadDim>;
        auto* dst  = nodes[0];
        auto* q    = dst->src[0];
        auto* k    = dst->src[1];
        auto* v    = dst->src[2];
        auto input = [](const ggml_tensor* t) { return AttentionInput{t->data, t->nb[1], t->nb[2], t->nb[3]}; };
        float scale;
        std::memcpy(&scale, dst->op_params, sizeof(scale));
        auto* packed             = static_cast<_Float16*>(ggml_backend_buffer_get_base(execution->workspace));
        const size_t value_tiles = ((v->ne[1] + Tile::keys - 1) / Tile::keys) * (Tile::keys / 32) * ((HeadDim + 31) / 32);
        const dim3 pack_grid(value_tiles, v->ne[2], v->ne[3]);
        pack_attention_values<HeadDim><<<pack_grid, dim3(32, 8), 0, stream>>>(input(v), packed, v->ne[1]);
        if (cuda_last_error() != cuda_success)
            return GGML_STATUS_FAILED;

        const dim3 grid((q->ne[1] + Tile::queries - 1) / Tile::queries, q->ne[2], q->ne[3]);
        wmma_attention<HeadDim><<<grid, Tile::query_waves * WAVE_SIZE, 0, stream>>>(
            input(q), input(k), packed, static_cast<float*>(dst->data), q->ne[1], k->ne[1], k->ne[2], q->ne[2] / k->ne[2], scale);
        return cuda_last_error() == cuda_success ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
    }

    struct AttentionKernel {
        int head_dim;
        int channels;
        int key_tile;
        int query_tile;
        ggml_status (*launch)(const ggml_ext_execution_context*, ggml_tensor* const*, CUDAStream);
    };

    template <int HeadDim>
    static constexpr AttentionKernel attention_kernel() {
        using Tile = AttentionTile<HeadDim>;
        return {HeadDim, Tile::channels, Tile::keys, Tile::queries, launch_attention<HeadDim>};
    }

    static const AttentionKernel* select_attention_kernel(int64_t head_dim) {
        static constexpr AttentionKernel kernels[] = {
            attention_kernel<40>(),
            attention_kernel<52>(),
            attention_kernel<64>(),
            attention_kernel<72>(),
            attention_kernel<80>(),
            attention_kernel<120>(),
            attention_kernel<128>(),
            attention_kernel<160>(),
            attention_kernel<256>(),
            attention_kernel<320>(),
        };
        for (const auto& kernel : kernels)
            if (head_dim == kernel.head_dim)
                return &kernel;
        return nullptr;
    }

    static bool wmma_attention_supports(ggml_backend_dev_t, const ggml_tensor* dst, void*) {
        const auto* q      = dst->src[0];
        const auto* k      = dst->src[1];
        const auto* v      = dst->src[2];
        const auto* kernel = select_attention_kernel(q->ne[0]);
        float params[3];
        std::memcpy(params, dst->op_params, sizeof(params));
        return kernel && !dst->src[3] && !dst->src[4] && std::isfinite(params[0]) && params[1] == 0.f && params[2] == 0.f &&
               q->ne[0] == k->ne[0] && ggml_are_same_shape(k, v) &&
               k->ne[2] > 0 && q->ne[2] % k->ne[2] == 0 && q->ne[3] == k->ne[3] && q->ne[2] <= 65535 && q->ne[3] <= 65535 &&
               q->ne[1] >= 512 && k->ne[1] >= 512 && q->ne[1] <= INT_MAX - kernel->query_tile &&
               k->ne[1] <= INT_MAX - kernel->key_tile &&
               attention_layout(q, GGML_TYPE_F32, sizeof(float)) &&
               attention_layout(k, GGML_TYPE_F16, q->ne[0] % 8 == 0 ? 16 : sizeof(_Float16)) &&
               attention_layout(v, GGML_TYPE_F16, sizeof(_Float16)) &&
               dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst) && !dst->view_src && attention_aligned(dst, 32);
    }

    static ggml_status wmma_attention_prepare(ggml_backend_t, ggml_tensor* const* nodes, int, void**, size_t* workspace, void*) {
        const auto* v      = nodes[0]->src[2];
        const auto* kernel = select_attention_kernel(v->ne[0]);
        if (!kernel)
            return GGML_STATUS_FAILED;
        const size_t tile = kernel->key_tile;
        *workspace        = ((size_t(v->ne[1]) + tile - 1) / tile) * tile * kernel->channels *
                            v->ne[2] * v->ne[3] * sizeof(_Float16);
        return GGML_STATUS_SUCCESS;
    }

    static void wmma_attention_destroy(void*, void*) {}

    static ggml_status wmma_attention_execute(const ggml_ext_execution_context* execution, ggml_tensor* const* nodes, int, void*, void*) {
        ggml_ext_gpu_context context;
        if (!execution->workspace || !cuda_context(execution->backend, &context)) {
            return GGML_STATUS_FAILED;
        }
        const auto* kernel = select_attention_kernel(nodes[0]->src[0]->ne[0]);
        return kernel ? kernel->launch(execution, nodes, static_cast<CUDAStream>(context.stream)) : GGML_STATUS_FAILED;
    }

    void register_wmma_attention(ggml_backend_dev_t device) {
        static const bool disabled = std::getenv("SD_HIP_DISABLE_WMMA_ATTN") != nullptr;
        const auto info            = cuda_device_info(device);
        int previous;
        if (disabled || info.device < 0 || hipGetDevice(&previous) != hipSuccess || hipSetDevice(info.device) != hipSuccess) {
            return;
        }
        hipFuncAttributes attributes{};
        const auto status   = hipFuncGetAttributes(&attributes, reinterpret_cast<const void*>(wmma_attention<128>));
        const auto restored = hipSetDevice(previous);
        // The device image uses a trap stub with no LDS when it lacks compatible wave32 WMMA.
        if (status != hipSuccess || restored != hipSuccess || attributes.sharedSizeBytes == 0) {
            return;
        }
        ggml_ext_kernel_desc desc = {};
        desc.abi_version          = GGML_EXT_ABI_VERSION;
        desc.name                 = "sd.wmma_attention.hip.v1";
        desc.op                   = GGML_OP_FLASH_ATTN_EXT;
        desc.priority             = 200;
        desc.supports             = wmma_attention_supports;
        desc.preferred            = wmma_attention_supports;
        desc.prepare              = wmma_attention_prepare;
        desc.execute              = wmma_attention_execute;
        desc.destroy              = wmma_attention_destroy;
        GGML_ASSERT(ggml_ext_register_kernel_ex(device, &desc, GGML_EXT_KERNEL_CAPTURE_SAFE | GGML_EXT_KERNEL_REPLAY_REUSABLE));
    }
}
