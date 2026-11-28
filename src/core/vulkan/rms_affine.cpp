#include <algorithm>
#include <climits>
#include <cstring>
#include "core/ggml_extensions.h"
#include "ggml-ext-vulkan.h"
#include "rms_modulate.h"

namespace sd_vulkan_rms_affine {
    struct Parameters {
        uint32_t ne[4], stride[4], offsets[5];
        float eps;
        uint32_t ne10;
    };
    static_assert(sizeof(Parameters) == 60, "RMS affine push constant layout");

    static const ggml_ext_vulkan_api* api(ggml_backend_dev_t device) {
        auto query = reinterpret_cast<ggml_ext_vulkan_api_fn>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(device), "ggml_ext_vulkan_v1"));
        return query ? query() : nullptr;
    }
    static bool bounded(const ggml_tensor* t, const ggml_ext_vulkan_device_info& info) {
        if (!t || t->type != GGML_TYPE_F32 || t->nb[0] != 4 || ggml_nbytes(t) > info.max_buffer_range - (info.buffer_alignment - 1))
            return false;
        size_t offset     = 0;
        const auto* owner = t;
        while (!owner->data && owner->view_src) {
            offset += owner->view_offs;
            owner = owner->view_src;
        }
        if ((reinterpret_cast<uintptr_t>(owner->data) + offset) % 4)
            return false;
        for (int d = 0; d < 4; ++d)
            if (t->ne[d] > UINT32_MAX || t->nb[d] % 4 || t->nb[d] / 4 > UINT32_MAX)
                return false;
        return true;
    }
    static const ggml_tensor* other(const ggml_tensor* node, const ggml_tensor* input) {
        return node->src[0] == input ? node->src[1] : node->src[1] == input ? node->src[0]
                                                                            : nullptr;
    }
    struct Operation {
        const ggml_tensor* x;
        const ggml_tensor* weight;
        const ggml_tensor* scale;
        const ggml_tensor* shift;
        const ggml_tensor* output;
    };
    static Operation describe(ggml_tensor* const* nodes, int count) {
        Operation op = {nodes[0]->src[0], other(nodes[1], nodes[0]), nullptr, nullptr, nodes[count - 1]};
        if (count == 3) {
            op.scale = nodes[2]->src[1];
            op.shift = nodes[2]->src[2];
        } else if (count == 5) {
            op.scale = other(nodes[2], nodes[1]);
            op.shift = other(nodes[4], nodes[3]);
        }
        return op;
    }
    static int match(ggml_backend_dev_t device, const ggml_cgraph* graph, int first, void*) {
        const auto* service = api(device);
        ggml_ext_vulkan_device_info info;
        auto* g = const_cast<ggml_cgraph*>(graph);
        if (!service || !service->device_info(device, &info) || info.fusion_disabled || first + 1 >= ggml_graph_n_nodes(g))
            return 0;
        ggml_tensor* nodes[5] = {};
        const int available   = std::min(5, ggml_graph_n_nodes(g) - first);
        for (int i = 0; i < available; ++i)
            nodes[i] = ggml_graph_node(g, first + i);
        if (nodes[0]->op != GGML_OP_RMS_NORM || nodes[1]->op != GGML_OP_MUL)
            return 0;
        const auto* x = nodes[0]->src[0];
        if (!bounded(x, info) || x->ne[0] != 3072 || available < 3 || !ggml_is_contiguous(x) ||
            x->ne[1] <= 1 || x->ne[1] > info.max_group_count[0] || x->ne[2] != 1 || x->ne[3] != 1 ||
            info.max_group_size[0] < 512 || info.max_group_invocations < 512)
            return 0;
        int count;
        if (sd_ggml_modulate_describe(nodes[2]) && nodes[2]->src[0] == nodes[1]) {
            count = 3;
        } else if (available == 5 && nodes[2]->op == GGML_OP_MUL && nodes[3]->op == GGML_OP_ADD &&
                   other(nodes[3], nodes[1]) == nodes[2] && nodes[4]->op == GGML_OP_ADD) {
            count = 5;
        } else {
            return 0;
        }
        const auto op = describe(nodes, count);
        if (!op.weight || !op.scale || (count == 5 && !op.shift))
            return 0;
        for (const auto* parameter : {op.weight, op.scale, op.shift}) {
            if (!parameter)
                continue;
            if (!bounded(parameter, info) || !ggml_is_contiguous(parameter) || parameter->ne[0] != x->ne[0] || ggml_nelements(parameter) != x->ne[0])
                return 0;
            for (int i = 0; i < count; ++i)
                if (parameter == nodes[i])
                    return 0;
        }
        for (int i = 0; i < count; ++i)
            if (!bounded(nodes[i], info) || !ggml_are_same_shape(nodes[i], x) || !ggml_is_contiguous(nodes[i]))
                return 0;
        return count;
    }
    static void optimize(ggml_backend_dev_t device, ggml_cgraph* graph, void*) {
        const auto* service = api(device);
        ggml_ext_vulkan_device_info info;
        if (!service || !service->device_info(device, &info) || info.fusion_disabled)
            return;
        for (int i = 2; i < ggml_graph_n_nodes(graph); ++i) {
            const auto* node = ggml_graph_node(graph, i);
            if (!sd_ggml_modulate_describe(node) && node->op != GGML_OP_MUL)
                continue;
            const auto* weighted = node->src[0];
            if (!weighted || weighted->op != GGML_OP_MUL || !weighted->src[0] ||
                weighted->src[0]->op != GGML_OP_RMS_NORM || weighted->ne[0] != 3072)
                continue;
            for (int j = i - 2; j >= std::max(0, i - 32); --j) {
                if (ggml_graph_node(graph, j) == weighted->src[0] && ggml_graph_node(graph, j + 1) == weighted) {
                    if (j + 2 < i)
                        ggml_ext_graph_delay(graph, j, 2, i);
                    break;
                }
            }
        }
    }
    static ggml_status prepare(ggml_backend_t, ggml_tensor* const* nodes, int count, void**, size_t* bytes, void*) {
        *bytes = ggml_nbytes(nodes[count - 1]);
        return GGML_STATUS_SUCCESS;
    }
    static void destroy(void*, void*) {}
    static ggml_status execute(const ggml_ext_execution_context* e, ggml_tensor* const* nodes, int count, void*, void*) {
        auto device         = ggml_backend_get_device(e->backend);
        const auto* service = api(device);
        ggml_ext_vulkan_device_info info;
        if (!service || !service->device_info(device, &info))
            return GGML_STATUS_FAILED;
        const auto op                      = describe(nodes, count);
        ggml_ext_vulkan_binding bindings[] = {{op.x}, {op.weight}, {op.output}, {op.scale}, {op.shift ? op.shift : op.scale}};
        ggml_ext_vulkan_buffer_info buffers[5];
        const int n = 5;
        for (int i = 0; i < n; ++i)
            if (!service->buffer_info(e->backend, bindings + i, buffers + i))
                return GGML_STATUS_FAILED;
        bool alias = false;
        for (int i = 0; i < n; ++i)
            if (i != 2)
                alias |= buffers[i].storage == buffers[2].storage && buffers[i].offset < buffers[2].offset + buffers[2].size &&
                         buffers[2].offset < buffers[i].offset + buffers[i].size;
        const auto destination = bindings[2];
        if (alias) {
            bindings[2] = {nullptr, e->workspace, 0, ggml_nbytes(op.output)};
            if (!service->buffer_info(e->backend, bindings + 2, buffers + 2))
                return GGML_STATUS_FAILED;
        }
        Parameters p = {};
        p.ne10       = uint32_t(op.weight->ne[0]);
        for (int i = 0; i < 4; ++i) {
            p.ne[i]     = uint32_t(op.x->ne[i]);
            p.stride[i] = uint32_t(op.x->nb[i] / 4);
        }
        for (int i = 0; i < n; ++i)
            p.offsets[i] = buffers[i].shader_offset / 4;
        std::memcpy(&p.eps, nodes[0]->op_params, sizeof(p.eps));
        const uint32_t has_shift          = op.shift != nullptr;
        ggml_ext_vulkan_pipeline pipeline = {};
        pipeline.name                     = "sd.rms_modulate.vulkan";
        pipeline.spirv                    = sd_vulkan_shaders::rms_modulate;
        pipeline.spirv_size               = sizeof(sd_vulkan_shaders::rms_modulate);
        pipeline.binding_count            = n;
        pipeline.push_constant_size       = sizeof(p);
        pipeline.constants                = &has_shift;
        pipeline.constant_count           = 1;
        uint32_t groups[]                 = {p.ne[1], std::min(p.ne[2], info.max_group_count[1]), std::min(p.ne[3], info.max_group_count[2])};
        auto status                       = service->dispatch(e->backend, &pipeline, bindings, &p, groups);
        if (status == GGML_STATUS_SUCCESS && alias)
            status = service->copy(e->backend, bindings + 2, &destination);
        return status;
    }
}
void sd_ggml_register_vulkan_rms_affine(ggml_backend_dev_t device) {
    if (!sd_vulkan_rms_affine::api(device))
        return;
    ggml_ext_kernel_desc desc = {};
    desc.abi_version          = GGML_EXT_ABI_VERSION;
    desc.name                 = "sd.rms_affine.vulkan.v1";
    desc.op                   = GGML_OP_RMS_NORM;
    desc.priority             = 150;
    desc.match                = sd_vulkan_rms_affine::match;
    desc.optimize             = sd_vulkan_rms_affine::optimize;
    desc.prepare              = sd_vulkan_rms_affine::prepare;
    desc.execute              = sd_vulkan_rms_affine::execute;
    desc.destroy              = sd_vulkan_rms_affine::destroy;
    GGML_ASSERT(ggml_ext_register_kernel(device, &desc));
}
