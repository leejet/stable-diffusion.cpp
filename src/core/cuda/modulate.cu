#include <algorithm>
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
            auto stream           = static_cast<CUDAStream>(context.stream);
            auto* output          = static_cast<float*>(alias ? ggml_backend_buffer_get_base(e->workspace) : dst->data);
            const unsigned blocks = unsigned(std::min<int64_t>((p.count + 255) / 256, 65535));
            modulate<<<blocks, 256, 0, stream>>>(static_cast<const float*>(x->data), static_cast<const float*>(scale->data),
                                                 shift ? static_cast<const float*>(shift->data) : nullptr, output, p);
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
    }
}
