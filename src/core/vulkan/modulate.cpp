#include "modulate.h"
#include <algorithm>
#include <climits>
#include "core/ggml_extensions.h"
#include "ggml-ext-vulkan.h"

namespace sd_vulkan_modulate {
    struct Parameters {
        uint32_t ne[4], scale_ne[4], shift_ne[4];
        uint32_t x_stride[4], scale_stride[4], shift_stride[4];
        uint32_t offsets[4];
        uint32_t count, has_shift;
    };
    static_assert(sizeof(Parameters) == 120, "Modulate push constant layout mismatch");

    static const ggml_ext_vulkan_api* api(ggml_backend_dev_t device) {
        auto query = reinterpret_cast<ggml_ext_vulkan_api_fn>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(device), "ggml_ext_vulkan_v1"));
        return query ? query() : nullptr;
    }

    static bool supports(ggml_backend_dev_t device, const ggml_tensor* node, void*) {
        const auto* service = api(device);
        ggml_ext_vulkan_device_info info;
        if (!service || !service->device_info(device, &info) || !sd_ggml_modulate_describe(node) ||
            info.max_group_size[0] < 256 || info.max_group_invocations < 256 || info.max_push_constants < sizeof(Parameters) || ggml_nelements(node) > UINT32_MAX - 255) {
            return false;
        }
        const ggml_tensor* tensors[] = {node, node->src[0], node->src[1], node->src[2]};
        for (const auto* t : tensors) {
            if (!t) {
                continue;
            }
            size_t offset     = 0;
            const auto* owner = t;
            while (!owner->data && owner->view_src) {
                offset += owner->view_offs;
                owner = owner->view_src;
            }
            if ((reinterpret_cast<uintptr_t>(owner->data) + offset) % 4 ||
                ggml_nbytes(t) > info.max_buffer_range - (info.buffer_alignment - 1)) {
                return false;
            }
            for (int d = 0; d < 4; ++d) {
                if (t->nb[d] / 4 > UINT32_MAX || t->ne[d] > UINT32_MAX) {
                    return false;
                }
            }
        }
        return (uint64_t(ggml_nelements(node)) + 255) / 256 <= uint64_t(info.max_group_count[0]) * info.max_group_count[1];
    }

    static ggml_status prepare(ggml_backend_t, ggml_tensor* const* nodes, int, void**, size_t* bytes, void*) {
        *bytes = sd_ggml_modulate_workspace_size(nodes[0]);
        return GGML_STATUS_SUCCESS;
    }

    static void destroy(void*, void*) {}

    static ggml_status execute(const ggml_ext_execution_context* e, ggml_tensor* const* nodes, int, void*, void*) {
        auto device         = ggml_backend_get_device(e->backend);
        const auto* service = api(device);
        ggml_ext_vulkan_device_info info;
        if (!service || !service->device_info(device, &info)) {
            return GGML_STATUS_FAILED;
        }
        const auto* dst                    = nodes[0];
        const auto* x                      = dst->src[0];
        const auto* scale                  = dst->src[1];
        const auto* shift                  = dst->src[2];
        ggml_ext_vulkan_binding bindings[] = {{x}, {scale}, {shift ? shift : scale}, {dst}};
        ggml_ext_vulkan_buffer_info buffers[4];
        for (int i = 0; i < 4; ++i) {
            if (!service->buffer_info(e->backend, &bindings[i], &buffers[i])) {
                return GGML_STATUS_FAILED;
            }
        }
        bool alias = false;
        for (int i = 0; i < 3; ++i) {
            alias |= buffers[i].storage == buffers[3].storage && buffers[i].offset < buffers[3].offset + buffers[3].size &&
                     buffers[3].offset < buffers[i].offset + buffers[i].size;
        }
        const auto destination = bindings[3];
        if (alias) {
            bindings[3] = {nullptr, e->workspace, 0, ggml_nbytes(dst)};
            if (!service->buffer_info(e->backend, &bindings[3], &buffers[3])) {
                return GGML_STATUS_FAILED;
            }
        }
        Parameters p = {};
        p.count      = uint32_t(ggml_nelements(dst));
        p.has_shift  = shift != nullptr;
        for (int d = 0; d < 4; ++d) {
            p.ne[d]           = uint32_t(x->ne[d]);
            p.scale_ne[d]     = uint32_t(scale->ne[d]);
            p.shift_ne[d]     = shift ? uint32_t(shift->ne[d]) : 1;
            p.x_stride[d]     = uint32_t(x->nb[d] / 4);
            p.scale_stride[d] = uint32_t(scale->nb[d] / 4);
            p.shift_stride[d] = shift ? uint32_t(shift->nb[d] / 4) : 0;
            p.offsets[d]      = buffers[d].shader_offset / 4;
        }
        const uint32_t count              = (p.count + 255) / 256;
        const uint32_t width              = std::min(count, info.max_group_count[0]);
        const uint32_t groups[]           = {width, (count + width - 1) / width, 1};
        ggml_ext_vulkan_pipeline pipeline = {};
        pipeline.name                     = "sd.modulate.vulkan";
        pipeline.spirv                    = sd_vulkan_shaders::modulate;
        pipeline.spirv_size               = sizeof(sd_vulkan_shaders::modulate);
        pipeline.binding_count            = 4;
        pipeline.push_constant_size       = sizeof(Parameters);
        auto status                       = service->dispatch(e->backend, &pipeline, bindings, &p, groups);
        if (status == GGML_STATUS_SUCCESS && alias) {
            status = service->copy(e->backend, &bindings[3], &destination);
        }
        return status;
    }
}

void sd_ggml_register_vulkan_modulate(ggml_backend_dev_t device) {
    if (!sd_vulkan_modulate::api(device)) {
        return;
    }
    ggml_ext_kernel_desc desc = {};
    desc.abi_version          = GGML_EXT_ABI_VERSION;
    desc.name                 = "sd.modulate.vulkan.v1";
    desc.op                   = GGML_OP_CUSTOM;
    desc.custom_op            = sd_ggml_modulate_op();
    desc.priority             = 100;
    desc.supports             = sd_vulkan_modulate::supports;
    desc.prepare              = sd_vulkan_modulate::prepare;
    desc.destroy              = sd_vulkan_modulate::destroy;
    desc.execute              = sd_vulkan_modulate::execute;
    GGML_ASSERT(ggml_ext_register_kernel(device, &desc));
}
