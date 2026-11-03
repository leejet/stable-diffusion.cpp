#include "bf16.h"
#if defined(SD_CUDA_VENDOR_HIP)
#include <hipblas/hipblas.h>
#elif defined(SD_CUDA_VENDOR_MUSA)
#include <mublas.h>
#else
#include <cublas_v2.h>
#endif
#include <climits>
#include <cstdlib>
#include <cstring>
#include "rms_affine.h"
#include "runtime.h"
#ifdef SD_CUDA_BF16_CUTLASS
#include "bf16_gemm.h"
#endif

namespace SD_CUDA_NAMESPACE {
    namespace sd_bf16 {
        static float scale_value(const ggml_tensor* node) {
            float value;
            std::memcpy(&value, node->op_params, sizeof(value));
            return value;
        }

        static bool matrix(const ggml_tensor* node, ggml_type type) {
            return node && node->type == type && ggml_is_contiguous(node) &&
                   node->ne[2] == 1 && node->ne[3] == 1 && node->ne[0] <= INT_MAX && node->ne[1] <= INT_MAX;
        }

        static bool projection(const ggml_tensor* mat, const ggml_tensor* input) {
            return mat->op == GGML_OP_MUL_MAT && mat->src[1] == input && mat->op_params[0] == GGML_PREC_DEFAULT &&
                   matrix(input, GGML_TYPE_F32) && input->ne[1] >= 32 && ggml_nelements(input) <= INT_MAX &&
                   matrix(mat, GGML_TYPE_F32) && matrix(mat->src[0], GGML_TYPE_BF16) && !mat->view_src;
        }

        static int match(ggml_backend_dev_t, const ggml_cgraph* graph, int first, void*) {
            // An explicit cuBLAS compute override must retain the native conversion path.
            static const bool disabled = [] {
                const char* value = std::getenv("GGML_CUDA_DISABLE_FUSION");
                return (value && std::atoi(value)) || std::getenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE");
            }();
            if (disabled)
                return 0;
            auto* g = const_cast<ggml_cgraph*>(graph);
            if (first + 1 >= ggml_graph_n_nodes(g))
                return 0;
            const auto* glu = ggml_graph_node(g, first);
            int offset      = 0;
            if (glu->op == GGML_OP_RMS_NORM) {
                if (!rms_affine_match(graph, first) || first + 4 >= ggml_graph_n_nodes(g))
                    return 0;
                const auto* input = ggml_graph_node(g, first + 3);
                const auto* gate  = ggml_graph_node(g, first + 4);
                if (!projection(gate, input))
                    return 0;
                if (first + 6 >= ggml_graph_n_nodes(g))
                    return 5;
                const auto* up = ggml_graph_node(g, first + 5);
                glu            = ggml_graph_node(g, first + 6);
                if (!projection(up, input) || glu->op != GGML_OP_GLU || glu->src[0] != gate || glu->src[1] != up)
                    return 5;
                offset = 6;
            }
            if (first + offset + 1 >= ggml_graph_n_nodes(g))
                return 0;
            if (glu->op != GGML_OP_GLU || ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU ||
                !matrix(glu, GGML_TYPE_F32) || !matrix(glu->src[0], GGML_TYPE_F32) ||
                !matrix(glu->src[1], GGML_TYPE_F32) || !ggml_are_same_shape(glu, glu->src[0]) ||
                !ggml_are_same_shape(glu, glu->src[1]) || glu->ne[1] < 32 || ggml_nelements(glu) > INT_MAX)
                return 0;
            int count         = 2;
            const auto* input = glu;
            const auto* mat   = ggml_graph_node(g, first + offset + 1);
            if (mat->op == GGML_OP_SCALE) {
                float bias;
                std::memcpy(&bias, mat->op_params + 1, sizeof(bias));
                if (mat->src[0] != glu || bias != 0.f || first + offset + 2 >= ggml_graph_n_nodes(g))
                    return 0;
                input = mat;
                mat   = ggml_graph_node(g, first + offset + 2);
                count = 3;
            }
            if (!projection(mat, input))
                return 0;
            return offset + count;
        }

        struct Workspace {
            size_t norm = 0, gate = 0, up = 0, packed = 0, output = 0, size = 0;
            Workspace(ggml_tensor* const* nodes, int count) {
                if (nodes[0]->op == GGML_OP_RMS_NORM) {
                    norm = reserve(ggml_nelements(nodes[3]) * sizeof(CUDABFloat16));
                    if (count > 5) {
                        gate   = reserve(ggml_nbytes(nodes[4]));
                        up     = reserve(ggml_nbytes(nodes[5]));
                        packed = reserve(ggml_nelements(nodes[6]) * sizeof(CUDABFloat16));
                    }
                } else {
                    packed = reserve(ggml_nelements(nodes[0]) * sizeof(CUDABFloat16));
                }
                output = reserve(ggml_nbytes(nodes[count - 1]));
            }
            size_t reserve(size_t bytes) {
                const size_t offset = size;
                size += GGML_PAD(bytes, 256);
                return offset;
            }
        };

        static ggml_status prepare(ggml_backend_t, ggml_tensor* const* nodes, int count, void**, size_t* workspace, void*) {
            *workspace = Workspace(nodes, count).size;
            return GGML_STATUS_SUCCESS;
        }

        static void destroy(void*, void*) {}

        static __device__ __forceinline__ float swiglu(float x, float up, float scale) {
            return __fmul_rn(__fmul_rn(x / (1.f + expf(-x)), up), scale);
        }

        static __global__ void pack_swiglu(const float* gate, const float* up, CUDABFloat16* dst, int64_t size, float scale) {
            const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i < size) {
                dst[i] = __float2bfloat16(swiglu(gate[i], up[i], scale));
            }
        }

        static __global__ void pack_swiglu4(const float4* gate, const float4* up, CUDABFloat162* dst, int64_t size, float scale) {
            const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i < size) {
                const float4 x = gate[i], y = up[i];
                dst[2 * i]     = __float22bfloat162_rn(make_float2(swiglu(x.x, y.x, scale), swiglu(x.y, y.y, scale)));
                dst[2 * i + 1] = __float22bfloat162_rn(make_float2(swiglu(x.z, y.z, scale), swiglu(x.w, y.w, scale)));
            }
        }

        static bool overlaps(const ggml_tensor* a, const ggml_tensor* b) {
            const auto x = reinterpret_cast<uintptr_t>(a->data), y = reinterpret_cast<uintptr_t>(b->data);
            return x <= y ? y - x < ggml_nbytes(a) : x - y < ggml_nbytes(b);
        }

        static ggml_status gemm(void* handle, const ggml_tensor* w, const void* input, int rows, void* output, CUDAStream stream, bool tiled) {
#ifdef SD_CUDA_BF16_CUTLASS
            if (tiled) {
                const auto result = sd_cuda_bf16_gemm(w->data, input, static_cast<float*>(output), rows, w->ne[1], w->ne[0], stream);
                if (result == SDBF16GemmResult::Success)
                    return GGML_STATUS_SUCCESS;
                if (result == SDBF16GemmResult::Failed)
                    return GGML_STATUS_FAILED;
            }
#else
            (void)tiled;
#endif
            const float alpha = 1.f, beta = 0.f;
#if defined(SD_CUDA_VENDOR_HIP)
#if HIP_VERSION >= 60500000 || defined(HIPBLAS_V2)
            constexpr auto compute_type = HIPBLAS_COMPUTE_32F;
#else
            constexpr auto compute_type = HIPBLAS_R_32F;
#endif
#if HIP_VERSION >= 70000000 || defined(HIPBLAS_V2)
            constexpr auto bf16_type = HIP_R_16BF, f32_type = HIP_R_32F;
#else
            constexpr auto bf16_type = HIPBLAS_R_16B, f32_type = HIPBLAS_R_32F;
#endif
            return hipblasGemmEx(static_cast<hipblasHandle_t>(handle), HIPBLAS_OP_T, HIPBLAS_OP_N,
                                 int(w->ne[1]), rows, int(w->ne[0]), &alpha,
                                 w->data, bf16_type, int(w->ne[0]), input, bf16_type, int(w->ne[0]),
                                 &beta, output, f32_type, int(w->ne[1]), compute_type, HIPBLAS_GEMM_DEFAULT) == HIPBLAS_STATUS_SUCCESS
                       ? GGML_STATUS_SUCCESS
                       : GGML_STATUS_FAILED;
#elif defined(SD_CUDA_VENDOR_MUSA)
            return mublasGemmEx(static_cast<mublasHandle_t>(handle), MUBLAS_OP_T, MUBLAS_OP_N,
                                int(w->ne[1]), rows, int(w->ne[0]), &alpha,
                                w->data, MUSA_R_16BF, int(w->ne[0]), input, MUSA_R_16BF, int(w->ne[0]),
                                &beta, output, MUSA_R_32F, int(w->ne[1]), MUSA_R_32F, MUBLAS_GEMM_DEFAULT) == MUBLAS_STATUS_SUCCESS
                       ? GGML_STATUS_SUCCESS
                       : GGML_STATUS_FAILED;
#else
            return cublasGemmEx(static_cast<cublasHandle_t>(handle), CUBLAS_OP_T, CUBLAS_OP_N, int(w->ne[1]), rows, int(w->ne[0]),
                                &alpha, w->data, CUDA_R_16BF, int(w->ne[0]), input, CUDA_R_16BF, int(w->ne[0]),
                                &beta, output, CUDA_R_32F, int(w->ne[1]), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) == CUBLAS_STATUS_SUCCESS
                       ? GGML_STATUS_SUCCESS
                       : GGML_STATUS_FAILED;
#endif
        }

        static ggml_status execute(const ggml_ext_execution_context* execution, ggml_tensor* const* nodes, int count, void*, void* user_data) {
            const bool tiled = *static_cast<const bool*>(user_data);
            ggml_ext_gpu_context context;
            if (!cuda_context(execution->backend, &context))
                return GGML_STATUS_FAILED;
            const auto reg    = ggml_backend_dev_backend_reg(ggml_backend_get_device(execution->backend));
            const auto query  = reinterpret_cast<ggml_ext_gpu_blas_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_ext_gpu_blas_v1"));
            const auto handle = query ? query(execution->backend) : nullptr;
            if (!handle)
                return GGML_STATUS_FAILED;
            const auto stream = static_cast<CUDAStream>(context.stream);
            const bool norm   = nodes[0]->op == GGML_OP_RMS_NORM;
            const int offset  = norm ? 6 : 0;
            const auto* glu   = norm && count == 5 ? nullptr : nodes[offset];
            auto* dst         = nodes[count - 1];
            const auto* w     = dst->src[0];
            auto* base        = static_cast<char*>(ggml_backend_buffer_get_base(execution->workspace));
            const Workspace work(nodes, count);
            auto* packed     = reinterpret_cast<CUDABFloat16*>(base + work.packed);
            const bool alias = overlaps(dst, w);
            void* output     = alias ? base + work.output : dst->data;
            if (norm) {
                if (rms_affine_pack(stream, nodes, base + work.norm) != GGML_STATUS_SUCCESS)
                    return GGML_STATUS_FAILED;
                if (gemm(handle, nodes[4]->src[0], base + work.norm, int(nodes[3]->ne[1]), count == 5 ? output : base + work.gate, stream, tiled) != GGML_STATUS_SUCCESS)
                    return GGML_STATUS_FAILED;
                if (count == 5)
                    return !alias || cuda_copy_async(dst->data, output, ggml_nbytes(dst), stream) == cuda_success ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
                if (gemm(handle, nodes[5]->src[0], base + work.norm, int(nodes[3]->ne[1]), base + work.up, stream, tiled) != GGML_STATUS_SUCCESS)
                    return GGML_STATUS_FAILED;
            }
            const auto size   = ggml_nelements(glu);
            const void* gate  = norm ? base + work.gate : glu->src[0]->data;
            const void* up    = norm ? base + work.up : glu->src[1]->data;
            const float scale = count - offset == 3 ? scale_value(nodes[offset + 1]) : 1.f;
            if (size % 4 == 0 && (reinterpret_cast<uintptr_t>(gate) | reinterpret_cast<uintptr_t>(up)) % 16 == 0) {
                pack_swiglu4<<<unsigned((size / 4 + 255) / 256), 256, 0, stream>>>(static_cast<const float4*>(gate),
                                                                                   static_cast<const float4*>(up),
                                                                                   reinterpret_cast<CUDABFloat162*>(packed), size / 4, scale);
            } else {
                pack_swiglu<<<unsigned((size + 255) / 256), 256, 0, stream>>>(static_cast<const float*>(gate),
                                                                              static_cast<const float*>(up), packed, size, scale);
            }
            if (cuda_last_error() != cuda_success)
                return GGML_STATUS_FAILED;
            const auto status = gemm(handle, w, packed, int(glu->ne[1]), output, stream, tiled);
            if (status != GGML_STATUS_SUCCESS)
                return GGML_STATUS_FAILED;
            return !alias || cuda_copy_async(dst->data, output, ggml_nbytes(dst), stream) == cuda_success ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
        }
    }

    void register_bf16(ggml_backend_dev_t device) {
        const auto info = cuda_device_info(device);
        if (info.device < 0 || !ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(device), "ggml_ext_gpu_blas_v1"))
            return;
        bool tiled = false;
#if defined(SD_CUDA_VENDOR_HIP)
        hipDeviceProp_t props;
        if (hipGetDeviceProperties(&props, info.device) != hipSuccess || std::strncmp(props.gcnArchName, "gfx", 3))
            return;
        const auto arch = std::strtoul(props.gcnArchName + 3, nullptr, 16);
        // Match the native backend's BF16 operand policy; older GPUs use FP32 GEMM.
        if (arch < 0x1100 && !(arch >= 0x908 && arch < 0x1010 && arch != 0x909 && arch != 0x90c))
            return;
#elif !defined(SD_CUDA_VENDOR_MUSA)
        cudaDeviceProp props;
        if (cudaGetDeviceProperties(&props, info.device) != cuda_success || props.major < 8)
            return;
        tiled = props.major >= 8;
#endif
        static bool tiled_gemm[]  = {false, true};
        ggml_ext_kernel_desc desc = {};
        desc.user_data            = &tiled_gemm[tiled];
        desc.abi_version          = GGML_EXT_ABI_VERSION;
        desc.name                 = "sd.swiglu_bf16_matmul." SD_CUDA_NAME ".v1";
        desc.op                   = GGML_OP_GLU;
        desc.priority             = 200;
        desc.match                = sd_bf16::match;
        desc.prepare              = sd_bf16::prepare;
        desc.destroy              = sd_bf16::destroy;
        desc.execute              = sd_bf16::execute;
        GGML_ASSERT(ggml_ext_register_kernel_ex(device, &desc, GGML_EXT_KERNEL_CAPTURE_SAFE | GGML_EXT_KERNEL_REPLAY_REUSABLE));
        desc.name     = "sd.rms_affine_bf16_matmul." SD_CUDA_NAME ".v1";
        desc.op       = GGML_OP_RMS_NORM;
        desc.priority = 400;
        GGML_ASSERT(ggml_ext_register_kernel_ex(device, &desc, GGML_EXT_KERNEL_CAPTURE_SAFE | GGML_EXT_KERNEL_REPLAY_REUSABLE));
    }
}
