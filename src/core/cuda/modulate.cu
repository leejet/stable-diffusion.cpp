#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include "core/ggml_extensions.h"
#include "runtime.h"

namespace SD_CUDA_NAMESPACE {
    namespace sd_modulate {
        struct Parameters {
            int64_t ne[4], scale_ne[4], shift_ne[4];
            size_t x_stride[4], scale_stride[4], shift_stride[4];
            int64_t count;
        };

        static __global__ void modulate(const float* x, const float* scale, const float* shift, float* dst, Parameters p) {
            for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < p.count; i += int64_t(blockDim.x) * gridDim.x) {
                int64_t index = i;
                size_t xi = 0, si = 0, hi = 0;
#pragma unroll
                for (int d = 0; d < 4; ++d) {
                    const int64_t coord = index % p.ne[d];
                    index /= p.ne[d];
                    xi += coord * p.x_stride[d];
                    si += (coord % p.scale_ne[d]) * p.scale_stride[d];
                    if (shift) {
                        hi += (coord % p.shift_ne[d]) * p.shift_stride[d];
                    }
                }
                float value = __fadd_rn(x[xi], __fmul_rn(x[xi], scale[si]));
                dst[i]      = shift ? __fadd_rn(value, shift[hi]) : value;
            }
        }

        template <bool Shift>
        static __global__ void modulate_rows(const float4* x, const float4* scale, const float4* shift, float4* dst, int64_t row_mask, int64_t count) {
            for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += int64_t(blockDim.x) * gridDim.x) {
                const int64_t col = i & row_mask;
                const float4 a    = x[i];
                const float4 s    = scale[col];
                float4 value      = make_float4(__fadd_rn(a.x, __fmul_rn(a.x, s.x)), __fadd_rn(a.y, __fmul_rn(a.y, s.y)),
                                                __fadd_rn(a.z, __fmul_rn(a.z, s.z)), __fadd_rn(a.w, __fmul_rn(a.w, s.w)));
                if constexpr (Shift) {
                    const float4 b = shift[col];
                    value          = make_float4(__fadd_rn(value.x, b.x), __fadd_rn(value.y, b.y),
                                                 __fadd_rn(value.z, b.z), __fadd_rn(value.w, b.w));
                }
                dst[i] = value;
            }
        }

        static __device__ float warp_sum(float value) {
            for (int offset = 16; offset > 0; offset >>= 1)
                value += cuda_shuffle_xor(value, offset, 32);
            return value;
        }

        template <int Width>
        static __global__ __launch_bounds__(Width / 4) void rms_modulate_rows(
            const float4* x,
            const float4* scale,
            const float4* shift,
            float4* dst,
            float eps) {
            constexpr int warps = Width / (4 * 32);
            __shared__ float partials[warps];
            const int col       = threadIdx.x;
            const int lane      = col % 32;
            const int64_t index = int64_t(blockIdx.x) * (Width / 4) + col;
            const float4 a      = x[index];
            // Match the vectorized RMSNorm kernel's reduction order.
            const float partial = warp_sum(a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w);
            if (lane == 0)
                partials[col / 32] = partial;
            __syncthreads();
            const float r      = rsqrtf(warp_sum(lane < warps ? partials[lane] : 0.f) / Width + eps);
            const float4 y     = make_float4(__fmul_rn(a.x, r), __fmul_rn(a.y, r), __fmul_rn(a.z, r), __fmul_rn(a.w, r));
            const float4 s     = scale[col];
            const float4 value = make_float4(__fadd_rn(y.x, __fmul_rn(y.x, s.x)), __fadd_rn(y.y, __fmul_rn(y.y, s.y)),
                                             __fadd_rn(y.z, __fmul_rn(y.z, s.z)), __fadd_rn(y.w, __fmul_rn(y.w, s.w)));
            if (shift) {
                const float4 b = shift[col];
                dst[index]     = make_float4(__fadd_rn(value.x, b.x), __fadd_rn(value.y, b.y),
                                             __fadd_rn(value.z, b.z), __fadd_rn(value.w, b.w));
            } else {
                dst[index] = value;
            }
        }

        static bool vector_aligned(const ggml_tensor* tensor) {
            if (tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor))
                return false;
            size_t offset = 0;
            while (!tensor->data && tensor->view_src) {
                offset += tensor->view_offs;
                tensor = tensor->view_src;
            }
            return (reinterpret_cast<uintptr_t>(tensor->data) + offset) % 16 == 0;
        }

        static bool rms_modulate_supports(const ggml_tensor* norm, const ggml_tensor* mod) {
            if (norm->op != GGML_OP_RMS_NORM || !sd_ggml_modulate_describe(mod) || mod->src[0] != norm ||
                mod->src[1] == norm || mod->src[2] == norm)
                return false;
            const auto* input     = norm->src[0];
            const auto width      = input->ne[0];
            const auto row_vector = [&](const ggml_tensor* tensor) {
                return tensor->ne[0] == width && ggml_nelements(tensor) == width && vector_aligned(tensor);
            };
            return (width == 1024 || width == 2048 || width == 4096) && ggml_nrows(input) <= INT_MAX &&
                   vector_aligned(input) && row_vector(mod->src[1]) && (!mod->src[2] || row_vector(mod->src[2]));
        }

        static bool fusion_enabled() {
            static const bool disabled = [] {
                const char* setting = std::getenv("GGML_CUDA_DISABLE_FUSION");
                return setting && std::atoi(setting) != 0;
            }();
            return !disabled;
        }

        static int rms_match(ggml_backend_dev_t, const ggml_cgraph* graph, int first, void*) {
            auto* g = const_cast<ggml_cgraph*>(graph);
            return fusion_enabled() && first + 1 < ggml_graph_n_nodes(g) &&
                           rms_modulate_supports(ggml_graph_node(g, first), ggml_graph_node(g, first + 1))
                       ? 2
                       : 0;
        }

        static void rms_optimize(ggml_backend_dev_t, ggml_cgraph* graph, void*) {
            if (!fusion_enabled())
                return;
            for (int i = 1; i < ggml_graph_n_nodes(graph); ++i) {
                const auto* mod  = ggml_graph_node(graph, i);
                const auto* norm = mod->src[0];
                if (!norm || !rms_modulate_supports(norm, mod))
                    continue;
                for (int j = i - 1; j >= 0 && j >= i - 48; --j) {
                    if (ggml_graph_node(graph, j) == norm) {
                        if (j + 1 < i)
                            ggml_ext_graph_delay(graph, j, 1, i);
                        break;
                    }
                }
            }
        }

        static ggml_status rms_prepare(ggml_backend_t, ggml_tensor* const* nodes, int, void**, size_t* bytes, void*) {
            const auto* dst = nodes[1];
            *bytes          = dst->data || dst->buffer || dst->view_src ? ggml_nbytes(dst) : 0;
            return GGML_STATUS_SUCCESS;
        }

        static ggml_status rms_execute(const ggml_ext_execution_context* e, ggml_tensor* const* nodes, int, void*, void*) {
            ggml_ext_gpu_context context;
            if (!cuda_context(e->backend, &context))
                return GGML_STATUS_FAILED;
            const auto* norm  = nodes[0];
            auto* mod         = nodes[1];
            const auto* input = norm->src[0];
            const auto* scale = mod->src[1];
            const auto* shift = mod->src[2];
            const auto dst    = reinterpret_cast<uintptr_t>(mod->data);
            bool alias        = false;
            for (const auto* tensor : {input, scale, shift}) {
                if (tensor) {
                    const auto src = reinterpret_cast<uintptr_t>(tensor->data);
                    alias |= dst < src + ggml_nbytes(tensor) && src < dst + ggml_nbytes(mod);
                }
            }
            float eps;
            std::memcpy(&eps, norm->op_params, sizeof(eps));
            const auto width = input->ne[0];
            auto kernel      = width == 4096 ? rms_modulate_rows<4096> : width == 2048 ? rms_modulate_rows<2048>
                                                                                       : rms_modulate_rows<1024>;
            auto stream      = static_cast<CUDAStream>(context.stream);
            auto* output     = static_cast<float4*>(alias ? ggml_backend_buffer_get_base(e->workspace) : mod->data);
            kernel<<<static_cast<unsigned>(ggml_nrows(input)), static_cast<unsigned>(width / 4), 0, stream>>>(
                static_cast<const float4*>(input->data), static_cast<const float4*>(scale->data),
                shift ? static_cast<const float4*>(shift->data) : nullptr, output, eps);
            if (cuda_last_error() != cuda_success || (alias && cuda_copy_async(mod->data, output, ggml_nbytes(mod), stream) != cuda_success))
                return GGML_STATUS_FAILED;
            return GGML_STATUS_SUCCESS;
        }

        static bool supports(ggml_backend_dev_t, const ggml_tensor* node, void*) {
            return sd_ggml_modulate_describe(node);
        }

        static ggml_status prepare(ggml_backend_t, ggml_tensor* const* nodes, int, void**, size_t* bytes, void*) {
            *bytes = sd_ggml_modulate_workspace_size(nodes[0]);
            return GGML_STATUS_SUCCESS;
        }

        static void destroy(void*, void*) {}

        static ggml_status execute(const ggml_ext_execution_context* e, ggml_tensor* const* nodes, int, void*, void*) {
            ggml_ext_gpu_context context;
            if (!cuda_context(e->backend, &context)) {
                return GGML_STATUS_FAILED;
            }
            auto* dst          = nodes[0];
            const auto* x      = dst->src[0];
            const auto* scale  = dst->src[1];
            const auto* shift  = dst->src[2];
            Parameters p       = {};
            p.count            = ggml_nelements(dst);
            bool alias         = false;
            const auto address = reinterpret_cast<uintptr_t>(dst->data);
            for (const auto* input : {x, scale, shift}) {
                if (input) {
                    const auto src = reinterpret_cast<uintptr_t>(input->data);
                    alias |= address < src + ggml_nbytes(input) && src < address + ggml_nbytes(dst);
                }
            }
            for (int d = 0; d < 4; ++d) {
                p.ne[d]           = x->ne[d];
                p.scale_ne[d]     = scale->ne[d];
                p.shift_ne[d]     = shift ? shift->ne[d] : 1;
                p.x_stride[d]     = x->nb[d] / sizeof(float);
                p.scale_stride[d] = scale->nb[d] / sizeof(float);
                p.shift_stride[d] = shift ? shift->nb[d] / sizeof(float) : 0;
            }
            auto stream                   = static_cast<CUDAStream>(context.stream);
            auto* output                  = static_cast<float*>(alias ? ggml_backend_buffer_get_base(e->workspace) : dst->data);
            const auto width              = x->ne[0];
            const auto aligned_contiguous = [](const ggml_tensor* tensor) {
                return ggml_is_contiguous(tensor) && reinterpret_cast<uintptr_t>(tensor->data) % 16 == 0;
            };
            const auto is_broadcast_row = [&](const ggml_tensor* tensor) {
                return tensor->ne[0] == width && ggml_nelements(tensor) == width && aligned_contiguous(tensor);
            };
            if (width >= 4 && !(width & (width - 1)) && aligned_contiguous(x) &&
                is_broadcast_row(scale) && (!shift || is_broadcast_row(shift)) &&
                reinterpret_cast<uintptr_t>(output) % 16 == 0) {
                const auto vectors    = p.count / 4;
                const unsigned blocks = unsigned(std::min<int64_t>((vectors + 255) / 256, 65535));
                auto kernel           = shift ? modulate_rows<true> : modulate_rows<false>;
                kernel<<<blocks, 256, 0, stream>>>(static_cast<const float4*>(x->data), static_cast<const float4*>(scale->data),
                                                   shift ? static_cast<const float4*>(shift->data) : nullptr,
                                                   reinterpret_cast<float4*>(output), width / 4 - 1, vectors);
            } else {
                const unsigned blocks = unsigned(std::min<int64_t>((p.count + 255) / 256, 65535));
                modulate<<<blocks, 256, 0, stream>>>(static_cast<const float*>(x->data), static_cast<const float*>(scale->data),
                                                     shift ? static_cast<const float*>(shift->data) : nullptr, output, p);
            }
            if (cuda_last_error() != cuda_success || (alias && cuda_copy_async(dst->data, output, ggml_nbytes(dst), stream) != cuda_success)) {
                return GGML_STATUS_FAILED;
            }
            return GGML_STATUS_SUCCESS;
        }
    }

    void register_modulate(ggml_backend_dev_t device) {
        ggml_ext_kernel_desc desc = {};
        desc.abi_version          = GGML_EXT_ABI_VERSION;
        desc.name                 = "sd.modulate." SD_CUDA_NAME ".v1";
        desc.op                   = GGML_OP_CUSTOM;
        desc.custom_op            = sd_ggml_modulate_op();
        desc.priority             = 100;
        desc.supports             = sd_modulate::supports;
        desc.prepare              = sd_modulate::prepare;
        desc.destroy              = sd_modulate::destroy;
        desc.execute              = sd_modulate::execute;
        GGML_ASSERT(ggml_ext_register_kernel_ex(device, &desc, GGML_EXT_KERNEL_CAPTURE_SAFE | GGML_EXT_KERNEL_REPLAY_REUSABLE));

        desc.name      = "sd.rms_modulate." SD_CUDA_NAME ".v1";
        desc.op        = GGML_OP_RMS_NORM;
        desc.custom_op = nullptr;
        desc.priority  = 200;
        desc.supports  = nullptr;
        desc.match     = sd_modulate::rms_match;
        desc.optimize  = sd_modulate::rms_optimize;
        desc.prepare   = sd_modulate::rms_prepare;
        desc.execute   = sd_modulate::rms_execute;
        GGML_ASSERT(ggml_ext_register_kernel_ex(device, &desc, GGML_EXT_KERNEL_CAPTURE_SAFE | GGML_EXT_KERNEL_REPLAY_REUSABLE));
    }
}
