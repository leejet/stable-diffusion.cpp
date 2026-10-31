#include "core/int8_ops.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <climits>
#include <mutex>
#include <unordered_map>

namespace sd_i8_cuda {
    template <int Width = 32>
    static __device__ __forceinline__ float warp_reduce_max(float value) {
#pragma unroll
        for (int offset = Width / 2; offset > 0; offset >>= 1)
            value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, offset, Width));
        return value;
    }
    static __global__ void quantize_rowwise_i8_cuda(const float* src, int8_t* dst, float* scales, int64_t k, int64_t rows) {
        const int64_t row = blockIdx.x;
        const int tid     = threadIdx.x;
        if (row >= rows) {
            return;
        }

        __shared__ float maxima[256];
        const float* src_row = src + row * k;
        int8_t* dst_row      = dst + row * k;
        float local_max      = 0.0f;
        for (int64_t i = tid; i < k; i += blockDim.x) {
            local_max = fmaxf(local_max, fabsf(src_row[i]));
        }
        maxima[tid] = local_max;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                maxima[tid] = fmaxf(maxima[tid], maxima[tid + stride]);
            }
            __syncthreads();
        }

        const float scale = maxima[0] / 127.0f;
        if (tid == 0) {
            scales[row] = scale;
        }
        if (scale == 0.0f) {
            for (int64_t i = tid; i < k; i += blockDim.x) {
                dst_row[i] = 0;
            }
            return;
        }

        const float inv_scale = 1.0f / scale;
        for (int64_t i = tid; i < k; i += blockDim.x) {
            int value  = __float2int_rn(src_row[i] * inv_scale);
            value      = max(-127, min(127, value));
            dst_row[i] = (int8_t)value;
        }
    }
    static __device__ __forceinline__ float convrot_h4_cuda(float a, float b, float c, float d, int index) {
        a = index == 3 ? -a : a;
        b = index == 2 ? -b : b;
        c = index == 1 ? -c : c;
        d = index == 0 ? -d : d;
        return __fadd_rn(__fadd_rn(__fadd_rn(a, b), c), d);
    }
    static __global__ void quantize_convrot(const float* src, int8_t* dst, float* scales, int64_t k, int64_t rows, int64_t dst_stride) {
        constexpr int group_size = 256;
        const int64_t row        = blockIdx.x;
        const int tid            = threadIdx.x;
        if (row >= rows)
            return;
        extern __shared__ float values[];
        __shared__ float maxima[32];
        const float* src_row = src + row * k;
        int8_t* dst_row      = dst + row * dst_stride;
        {
            const int lane = tid % 32;
            for (int64_t group = tid / 32; group < k / group_size; group += blockDim.x / 32) {
                float v[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int64_t i   = group * group_size + j * 32 + lane;
                    const float value = src_row[i];
                    v[j]              = value * (1.0f / 16.0f);
                }
                // One logical warp owns H256, keeping every butterfly in registers.
#pragma unroll
                for (int stride = 1; stride <= 4; stride *= 4) {
                    const int base = lane & ~(3 * stride);
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const float a = __shfl_sync(0xffffffff, v[j], base, 32);
                        const float b = __shfl_sync(0xffffffff, v[j], base + stride, 32);
                        const float c = __shfl_sync(0xffffffff, v[j], base + 2 * stride, 32);
                        const float d = __shfl_sync(0xffffffff, v[j], base + 3 * stride, 32);
                        v[j]          = convrot_h4_cuda(a, b, c, d, (lane / stride) % 4);
                    }
                }
#pragma unroll
                for (int j = 0; j < 8; j += 2) {
                    const float a = __shfl_sync(0xffffffff, v[j], lane % 16, 32);
                    const float b = __shfl_sync(0xffffffff, v[j], lane % 16 + 16, 32);
                    const float c = __shfl_sync(0xffffffff, v[j + 1], lane % 16, 32);
                    const float d = __shfl_sync(0xffffffff, v[j + 1], lane % 16 + 16, 32);
                    v[j]          = convrot_h4_cuda(a, b, c, d, lane / 16);
                    v[j + 1]      = convrot_h4_cuda(a, b, c, d, 2 + lane / 16);
                }
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const float a = v[j], b = v[j + 2], c = v[j + 4], d = v[j + 6];
#pragma unroll
                    for (int t = 0; t < 4; ++t) {
                        values[group * group_size + (j + 2 * t) * 32 + lane] = convrot_h4_cuda(a, b, c, d, t);
                    }
                }
            }
            __syncthreads();
        }
        float local_max = 0.0f;
        for (int64_t i = tid; i < k; i += blockDim.x) {
            local_max = fmaxf(local_max, fabsf(values[i]));
        }
        local_max      = warp_reduce_max<32>(local_max);
        const int lane = tid & 31;
        const int warp = tid >> 5;
        if (lane == 0) {
            maxima[warp] = local_max;
        }
        __syncthreads();

        if (warp == 0) {
            local_max = lane < blockDim.x / 32 ? maxima[lane] : 0.0f;
            local_max = warp_reduce_max<32>(local_max);
            if (lane == 0) {
                maxima[0] = local_max / 127.0f;
                if (scales != nullptr) {
                    scales[row] = maxima[0];
                } else {
                    *((float*)(dst_row + k)) = maxima[0];
                }
            }
        }
        __syncthreads();

        const float row_scale = maxima[0];
        const float inv_scale = row_scale == 0.0f ? 0.0f : 1.0f / row_scale;
        for (int64_t i = tid; i < k; i += blockDim.x) {
            int value  = row_scale == 0.0f ? 0 : __float2int_rn(values[i] * inv_scale);
            value      = max(-127, min(127, value));
            dst_row[i] = (int8_t)value;
        }
    }

    static __global__ void convrot_group_amax_i8_cuda(
        const float* src,
        float* partial_max,
        int64_t k,
        int64_t groups,
        int64_t total_groups) {
        const int64_t index = blockIdx.x;
        const int tid       = threadIdx.x;
        if (index >= total_groups) {
            return;
        }

        __shared__ float values[256];
        __shared__ float maxima[2];
        const int64_t row   = index / groups;
        const int64_t group = index - row * groups;
        const float* input  = src + row * k + group * 256;
        values[tid]         = input[tid] * (1.0f / 16.0f);
        values[tid + 64]    = input[tid + 64] * (1.0f / 16.0f);
        values[tid + 128]   = input[tid + 128] * (1.0f / 16.0f);
        values[tid + 192]   = input[tid + 192] * (1.0f / 16.0f);
        __syncthreads();

#pragma unroll
        for (int stride = 1; stride < 256; stride *= 4) {
            const int base = (tid / stride) * 4 * stride + tid % stride;
            const int i0   = base;
            const int i1   = i0 + stride;
            const int i2   = i1 + stride;
            const int i3   = i2 + stride;
            const float a  = values[i0];
            const float b  = values[i1];
            const float c  = values[i2];
            const float d  = values[i3];
            values[i0]     = a + b + c - d;
            values[i1]     = a + b - c + d;
            values[i2]     = a - b + c + d;
            values[i3]     = -a + b + c + d;
            __syncthreads();
        }

        float value = fmaxf(fabsf(values[tid]), fabsf(values[tid + 64]));
        value       = fmaxf(value, fabsf(values[tid + 128]));
        value       = fmaxf(value, fabsf(values[tid + 192]));
        value       = warp_reduce_max<32>(value);
        if ((tid & 31) == 0) {
            maxima[tid >> 5] = value;
        }
        __syncthreads();
        if (tid == 0) {
            partial_max[index] = fmaxf(maxima[0], maxima[1]);
        }
    }

    static __global__ void reduce_convrot_row_amax_i8_cuda(
        const float* partial_max,
        float* scales,
        int64_t groups,
        int64_t rows) {
        const int64_t row = blockIdx.x;
        const int tid     = threadIdx.x;
        if (row >= rows) {
            return;
        }

        __shared__ float maxima[256];
        float value = 0.0f;
        for (int64_t group = tid; group < groups; group += blockDim.x) {
            value = fmaxf(value, partial_max[row * groups + group]);
        }
        maxima[tid] = value;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                maxima[tid] = fmaxf(maxima[tid], maxima[tid + stride]);
            }
            __syncthreads();
        }
        if (tid == 0) {
            scales[row] = maxima[0] / 127.0f;
        }
    }

    static __global__ void convrot_group_quantize_i8_cuda(
        const float* src,
        int8_t* dst,
        const float* scales,
        int64_t k,
        int64_t groups,
        int64_t total_groups,
        int64_t dst_stride) {
        const int64_t index = blockIdx.x;
        const int tid       = threadIdx.x;
        if (index >= total_groups) {
            return;
        }

        __shared__ float values[256];
        const int64_t row   = index / groups;
        const int64_t group = index - row * groups;
        const float* input  = src + row * k + group * 256;
        int8_t* output      = dst + row * dst_stride + group * 256;
        values[tid]         = input[tid] * (1.0f / 16.0f);
        values[tid + 64]    = input[tid + 64] * (1.0f / 16.0f);
        values[tid + 128]   = input[tid + 128] * (1.0f / 16.0f);
        values[tid + 192]   = input[tid + 192] * (1.0f / 16.0f);
        __syncthreads();

#pragma unroll
        for (int stride = 1; stride < 256; stride *= 4) {
            const int base = (tid / stride) * 4 * stride + tid % stride;
            const int i0   = base;
            const int i1   = i0 + stride;
            const int i2   = i1 + stride;
            const int i3   = i2 + stride;
            const float a  = values[i0];
            const float b  = values[i1];
            const float c  = values[i2];
            const float d  = values[i3];
            values[i0]     = a + b + c - d;
            values[i1]     = a + b - c + d;
            values[i2]     = a - b + c + d;
            values[i3]     = -a + b + c + d;
            __syncthreads();
        }

        const float row_scale = scales[row];
        const float inv_scale = row_scale == 0.0f ? 0.0f : 1.0f / row_scale;
        const int indices[4]  = {tid, tid + 64, tid + 128, tid + 192};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            int value          = row_scale == 0.0f ? 0 : __float2int_rn(values[indices[i]] * inv_scale);
            value              = max(-127, min(127, value));
            output[indices[i]] = (int8_t)value;
        }
        if (group == 0 && tid == 0 && dst_stride != k) {
            *((float*)(dst + row * dst_stride + k)) = row_scale;
        }
    }

    static __global__ void dequantize_i32_rows_cuda(
        const int32_t* src,
        const float* scales,
        const float* weight_scales,
        const float* bias,
        float* dst,
        int64_t n,
        int64_t rows) {
        const int64_t output = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
        const int64_t row    = blockIdx.y;
        if (output < n && row < rows) {
            const int64_t index          = row * n + output;
            const float activation_scale = scales[row];
            float value                  = (float)src[index] * activation_scale;
            if (weight_scales != nullptr) {
                value *= weight_scales[output];
            }
            if (bias != nullptr) {
                value += bias[output];
            }
            dst[index] = value;
        }
    }

    static __global__ void dequantize_i32_rows_vec4_cuda(
        const int4* src,
        const float* scales,
        const float4* weight_scales,
        const float4* bias,
        float4* dst,
        int64_t n4) {
        const int64_t output = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (output >= n4) {
            return;
        }
        const int64_t index = (int64_t)blockIdx.y * n4 + output;
        const int4 v        = src[index];
        const float scale   = scales[blockIdx.y];
        float4 value        = make_float4(v.x * scale, v.y * scale, v.z * scale, v.w * scale);
        if (weight_scales != nullptr) {
            const float4 w = weight_scales[output];
            value.x *= w.x;
            value.y *= w.y;
            value.z *= w.z;
            value.w *= w.w;
        }
        if (bias != nullptr) {
            const float4 b = bias[output];
            value.x += b.x;
            value.y += b.y;
            value.z += b.z;
            value.w += b.w;
        }
        dst[index] = value;
    }

    struct DeviceInfo {
        int ordinal   = -1;
        int major     = 0;
        int minor     = 0;
        size_t shared = 0;
    };
    static DeviceInfo device_info(ggml_backend_dev_t device) {
        static std::mutex mutex;
        static std::unordered_map<ggml_backend_dev_t, DeviceInfo> cache;
        std::lock_guard<std::mutex> lock(mutex);
        auto found = cache.find(device);
        if (found != cache.end())
            return found->second;
        DeviceInfo info;
        auto reg             = ggml_backend_dev_backend_reg(device);
        auto query           = reinterpret_cast<ggml_ext_cuda_device_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_ext_cuda_device_v1"));
        int ordinal          = query ? query(device) : -1;
        cudaDeviceProp props = {};
        if (ordinal >= 0 && cudaGetDeviceProperties(&props, ordinal) == cudaSuccess)
            info = {ordinal, props.major, props.minor, size_t(props.sharedMemPerBlockOptin)};
        cache.emplace(device, info);
        return info;
    }
    static bool aligned(const ggml_tensor* tensor) {
        if (!tensor)
            return true;
        size_t offset = 0;
        while (tensor->view_src) {
            offset += tensor->view_offs;
            tensor = tensor->view_src;
        }
        return (reinterpret_cast<uintptr_t>(tensor->data) + offset) % 4 == 0;
    }
    static bool supports(ggml_backend_dev_t device, const ggml_tensor* node, void*) {
        SDI8Operation op;
        if (!sd_ggml_i8_describe(node, &op) || !aligned(node))
            return false;
        const int64_t k = op.weight->ne[0], n = op.weight->ne[1], rows = ggml_nrows(node);
        if (k > INT_MAX || n > INT_MAX || rows > 65535 || k % 4 || n % 4 || (op.group_size && (op.group_size != 256 || rows > INT_MAX / (k / 256))))
            return false;
        for (auto* input : node->src)
            if (!aligned(input))
                return false;
        const auto info = device_info(device);
        return (info.major > 7 || (info.major == 7 && info.minor >= 5)) &&
               ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(device), "ggml_ext_cuda_blas_v1");
    }
    static bool convrot_supports(ggml_backend_dev_t device, const ggml_tensor* node, void*) {
        int group;
        if (!sd_ggml_convrot_describe(node, &group) || group != 256 || !aligned(node) || !aligned(node->src[0]))
            return false;
        const int64_t k = node->ne[0], rows = ggml_nrows(node->src[0]);
        return k <= INT_MAX && rows <= INT_MAX && rows <= INT_MAX / (k / 256) && device_info(device).ordinal >= 0;
    }
    struct Plan {
        DeviceInfo device;
        int64_t k, n, rows, padded;
        size_t quant, scales, accum, partial, output;
    };
    static size_t reserve(size_t& total, size_t bytes) {
        const size_t offset = total;
        total += (bytes + 255) / 256 * 256;
        return offset;
    }
    static ggml_status prepare(ggml_backend_t backend, ggml_tensor* const* nodes, int, void** state, size_t* workspace, void*) {
        const auto* dst = nodes[0];
        SDI8Operation op;
        const bool mat = sd_ggml_i8_describe(dst, &op);
        auto* p        = new Plan{};
        *state         = p;
        p->device      = device_info(ggml_backend_get_device(backend));
        p->k           = mat ? op.weight->ne[0] : dst->ne[0];
        p->n           = mat ? dst->ne[0] : 0;
        p->rows        = mat ? ggml_nrows(dst) : ggml_nrows(dst->src[0]);
        p->padded      = (p->rows + 3) / 4 * 4;
        size_t total   = 0;
        if (mat) {
            p->quant  = reserve(total, op.input->type == GGML_TYPE_F32 || (!op.packed && p->rows != p->padded) ? size_t(p->k) * p->padded : 0);
            p->scales = reserve(total, op.input->type == GGML_TYPE_F32 ? size_t(p->rows) * sizeof(float) : 0);
            p->accum  = reserve(total, size_t(p->n) * p->padded * sizeof(int32_t));
        }
        const bool large_rotation = (!mat || (op.input->type == GGML_TYPE_F32 && op.group_size == 256)) && size_t(p->k) * sizeof(float) + 128 > p->device.shared;
        p->partial                = reserve(total, large_rotation ? size_t(p->rows) * (p->k / 256) * sizeof(float) : 0);
        p->output                 = reserve(total, ggml_nbytes(dst));
        *workspace                = total;
        return GGML_STATUS_SUCCESS;
    }
    static void destroy(void* state, void*) {
        delete static_cast<Plan*>(state);
    }
    static bool overlaps(const ggml_tensor* a, const ggml_tensor* b) {
        if (!b)
            return false;
        const auto x = reinterpret_cast<uintptr_t>(a->data), y = reinterpret_cast<uintptr_t>(b->data);
        return x <= y ? y - x < ggml_nbytes(a) : x - y < ggml_nbytes(b);
    }
    static cudaError_t convrot_launch(const Plan& p, cudaStream_t stream, const float* src, int8_t* dst, float* scales, float* partial) {
        if (p.padded > p.rows) {
            auto error = cudaMemsetAsync(dst + p.k * p.rows, 0, size_t(p.k) * (p.padded - p.rows), stream);
            if (error != cudaSuccess)
                return error;
        }
        const size_t shared = size_t(p.k) * sizeof(float);
        if (shared + 128 <= p.device.shared) {
            static std::mutex mutex;
            static std::unordered_map<int, bool> configured;
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (!configured[p.device.ordinal]) {
                    auto error = cudaFuncSetAttribute(quantize_convrot, cudaFuncAttributeMaxDynamicSharedMemorySize, int(p.device.shared - 128));
                    if (error != cudaSuccess)
                        return error;
                    configured[p.device.ordinal] = true;
                }
            }
            int threads = p.k <= 1024 ? 128 : p.k <= 8192 ? 256
                                                          : 1024;
            quantize_convrot<<<unsigned(p.rows), threads, shared, stream>>>(src, dst, scales, p.k, p.rows, p.k);
        } else {
            const int64_t groups = p.k / 256, total_groups = p.rows * groups;
            convrot_group_amax_i8_cuda<<<unsigned(total_groups), 64, 0, stream>>>(src, partial, p.k, groups, total_groups);
            reduce_convrot_row_amax_i8_cuda<<<unsigned(p.rows), 256, 0, stream>>>(partial, scales, groups, p.rows);
            convrot_group_quantize_i8_cuda<<<unsigned(total_groups), 64, 0, stream>>>(src, dst, scales, p.k, groups, total_groups, p.k);
        }
        return cudaGetLastError();
    }
    static ggml_status execute(const ggml_ext_execution_context* execution, ggml_tensor* const* nodes, int, void* state, void*) {
        const auto& p = *static_cast<const Plan*>(state);
        auto reg      = ggml_backend_dev_backend_reg(ggml_backend_get_device(execution->backend));
        auto query    = reinterpret_cast<ggml_ext_cuda_context_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_ext_cuda_context_v1"));
        ggml_ext_cuda_context context;
        if (!query || !query(execution->backend, &context) || context.device != p.device.ordinal)
            return GGML_STATUS_FAILED;
        auto stream   = static_cast<cudaStream_t>(context.stream);
        auto* scratch = static_cast<char*>(ggml_backend_buffer_get_base(execution->workspace));
        auto* dst     = nodes[0];
        SDI8Operation op;
        const bool mat = sd_ggml_i8_describe(dst, &op);
        bool alias     = false;
        for (auto* input : dst->src)
            alias |= overlaps(dst, input);
        void* output      = alias ? scratch + p.output : dst->data;
        cudaError_t error = cudaSuccess;
        if (!mat) {
            auto* quant  = static_cast<int8_t*>(output);
            auto* scales = reinterpret_cast<float*>(quant + p.k * p.padded);
            error        = convrot_launch(p, stream, static_cast<const float*>(dst->src[0]->data), quant, scales, reinterpret_cast<float*>(scratch + p.partial));
        } else {
            auto blas_query = reinterpret_cast<ggml_ext_cuda_blas_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_ext_cuda_blas_v1"));
            auto handle     = blas_query ? static_cast<cublasHandle_t>(blas_query(execution->backend)) : nullptr;
            if (!handle)
                return GGML_STATUS_FAILED;
            auto* quant = reinterpret_cast<int8_t*>(scratch + p.quant);
            const float* scales;
            if (op.input->type == GGML_TYPE_F32) {
                auto* temporary_scales = reinterpret_cast<float*>(scratch + p.scales);
                scales                 = temporary_scales;
                if (op.group_size)
                    error = convrot_launch(p, stream, static_cast<const float*>(op.input->data), quant, temporary_scales, reinterpret_cast<float*>(scratch + p.partial));
                else {
                    if (p.padded > p.rows)
                        error = cudaMemsetAsync(quant + p.k * p.rows, 0, size_t(p.k) * (p.padded - p.rows), stream);
                    if (error == cudaSuccess)
                        quantize_rowwise_i8_cuda<<<unsigned(p.rows), 256, 0, stream>>>(static_cast<const float*>(op.input->data), quant, temporary_scales, p.k, p.rows);
                }
            } else {
                auto* data = static_cast<int8_t*>(op.input->data);
                scales     = op.packed ? reinterpret_cast<const float*>(data + p.k * p.padded) : static_cast<const float*>(op.input_scale->data);
                if (op.packed || p.rows == p.padded)
                    quant = data;
                else {
                    error = cudaMemcpyAsync(quant, data, size_t(p.k) * p.rows, cudaMemcpyDeviceToDevice, stream);
                    if (error == cudaSuccess)
                        error = cudaMemsetAsync(quant + p.k * p.rows, 0, size_t(p.k) * (p.padded - p.rows), stream);
                }
            }
            if (error != cudaSuccess || cudaGetLastError() != cudaSuccess)
                return GGML_STATUS_FAILED;
            auto* accum         = reinterpret_cast<int32_t*>(scratch + p.accum);
            const int32_t alpha = 1, beta = 0;
            if (cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, int(p.n), int(p.padded), int(p.k), &alpha,
                             op.weight->data, CUDA_R_8I, int(p.k), quant, CUDA_R_8I, int(p.k), &beta, accum, CUDA_R_32I, int(p.n), CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP) != CUBLAS_STATUS_SUCCESS)
                return GGML_STATUS_FAILED;
            const auto* ws   = op.weight_scale ? static_cast<const float*>(op.weight_scale->data) : nullptr;
            const auto* bias = op.bias ? static_cast<const float*>(op.bias->data) : nullptr;
            if ((reinterpret_cast<uintptr_t>(ws) | reinterpret_cast<uintptr_t>(bias) | reinterpret_cast<uintptr_t>(output)) % 16 == 0) {
                const dim3 grid(unsigned((p.n + 1023) / 1024), unsigned(p.rows));
                dequantize_i32_rows_vec4_cuda<<<grid, 256, 0, stream>>>(reinterpret_cast<const int4*>(accum), scales, reinterpret_cast<const float4*>(ws), reinterpret_cast<const float4*>(bias), static_cast<float4*>(output), p.n / 4);
            } else {
                const dim3 grid(unsigned((p.n + 255) / 256), unsigned(p.rows));
                dequantize_i32_rows_cuda<<<grid, 256, 0, stream>>>(accum, scales, ws, bias, static_cast<float*>(output), p.n, p.rows);
            }
            error = cudaGetLastError();
        }
        if (error == cudaSuccess && alias)
            error = cudaMemcpyAsync(dst->data, output, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, stream);
        return error == cudaSuccess ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
    }
}

void sd_ggml_register_cuda_i8(ggml_backend_dev_t device) {
    ggml_ext_kernel_desc desc = {};
    desc.abi_version          = GGML_EXT_ABI_VERSION;
    desc.name                 = "sd.linear_i8.cuda.v1";
    desc.op                   = GGML_OP_CUSTOM;
    desc.custom_op            = sd_ggml_i8_op();
    desc.priority             = 100;
    desc.supports             = sd_i8_cuda::supports;
    desc.prepare              = sd_i8_cuda::prepare;
    desc.execute              = sd_i8_cuda::execute;
    desc.destroy              = sd_i8_cuda::destroy;
    GGML_ASSERT(ggml_ext_register_kernel_ex(device, &desc, GGML_EXT_KERNEL_CAPTURE_SAFE | GGML_EXT_KERNEL_REPLAY_REUSABLE));
    desc.name      = "sd.quantize_i8_convrot.cuda.v1";
    desc.custom_op = sd_ggml_convrot_op();
    desc.supports  = sd_i8_cuda::convrot_supports;
    GGML_ASSERT(ggml_ext_register_kernel_ex(device, &desc, GGML_EXT_KERNEL_CAPTURE_SAFE | GGML_EXT_KERNEL_REPLAY_REUSABLE));
}
