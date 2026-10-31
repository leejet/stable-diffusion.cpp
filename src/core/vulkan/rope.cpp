#include "core/ggml_extensions.h"
#include "ggml-ext-vulkan.h"
#include "rms_rope_apply.h"
#include "rms_rope_cast.h"
#include "rope_apply.h"

#include <algorithm>
#include <climits>
#include <cstring>

namespace sd_vulkan_rope {
    struct Parameters {
        uint32_t half_dim, rot_half, heads, tokens, pairs;
        uint32_t x_stride[4], pe_stride[4];
        uint32_t x_offset, pe_offset, dst_offset, interleaved;
    };
    struct FusedParameters {
        Parameters rope;
        float eps;
        uint32_t w_offset;
        float output_scale;
    };
    static_assert(sizeof(Parameters) == 68 && sizeof(FusedParameters) == 80, "RoPE push constant layout mismatch");
    static const ggml_ext_vulkan_api* api(ggml_backend_dev_t device) {
        auto query = reinterpret_cast<ggml_ext_vulkan_api_fn>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(device), "ggml_ext_vulkan_v1"));
        return query ? query() : nullptr;
    }
    static float param(const ggml_tensor* node, int i) {
        float value;
        std::memcpy(&value, node->op_params + i, sizeof(value));
        return value;
    }
    static bool is_rope(const ggml_tensor* node) {
        return node && (node->op == GGML_OP_ROPE_APPLY || ggml_ext_op_get(node) == sd_ggml_rope_op());
    }
    static bool aligned(const ggml_tensor* tensor, size_t alignment) {
        size_t offset = 0;
        while (!tensor->data && tensor->view_src) {
            offset += tensor->view_offs;
            tensor = tensor->view_src;
        }
        return (reinterpret_cast<uintptr_t>(tensor->data) + offset) % alignment == 0;
    }
    static bool bounded(const ggml_tensor* node, const ggml_ext_vulkan_device_info& info) {
        if (!node || !aligned(node, 4) || ggml_nbytes(node) > info.max_buffer_range - (info.buffer_alignment - 1))
            return false;
        for (int i = 0; i < 4; ++i)
            if (node->nb[i] % 4 || node->nb[i] / 4 > UINT32_MAX || node->ne[i] > UINT32_MAX)
                return false;
        return true;
    }
    static bool supports(ggml_backend_dev_t device, const ggml_tensor* node, void*) {
        const auto* service = api(device);
        ggml_ext_vulkan_device_info info;
        if (!service || !service->device_info(device, &info) || info.max_group_size[0] < 256 || info.max_group_invocations < 256)
            return false;
        const auto* x  = node->src[0];
        const auto* pe = node->src[1];
        return x && pe && x->type == GGML_TYPE_F32 && pe->type == GGML_TYPE_F32 && node->type == GGML_TYPE_F32 &&
               x->ne[0] % 2 == 0 && pe->ne[0] == 2 && pe->ne[1] == 2 && pe->ne[2] > 0 && pe->ne[2] <= x->ne[0] / 2 &&
               pe->ne[3] == x->ne[2] && node->ne[0] == x->ne[0] && node->ne[1] == x->ne[2] &&
               node->ne[2] == x->ne[1] * x->ne[3] && node->ne[3] == 1 && ggml_is_contiguous(node) &&
               bounded(x, info) && bounded(pe, info) && bounded(node, info) && ggml_nelements(node) / 2 <= UINT32_MAX &&
               (uint64_t(ggml_nelements(node) / 2) + 255) / 256 <= uint64_t(info.max_group_count[0]) * info.max_group_count[1];
    }
    static bool fused_supports(ggml_backend_dev_t device, const ggml_tensor* norm, const ggml_tensor* mul, const ggml_tensor* rope) {
        ggml_ext_vulkan_device_info info;
        if (!api(device)->device_info(device, &info) || info.fusion_disabled || !info.subgroup_shuffle || !info.full_subgroups ||
            info.subgroup_size != 32 || !is_rope(rope) || !supports(device, rope, nullptr) ||
            norm->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL || mul->src[0] != norm || rope->src[0] != mul)
            return false;
        const auto* x      = norm->src[0];
        const auto* weight = mul->src[1];
        return x && weight && x->type == GGML_TYPE_F32 && x->ne[0] == 128 && x->nb[0] == 4 &&
               norm->type == GGML_TYPE_F32 && mul->type == GGML_TYPE_F32 && ggml_are_same_shape(x, norm) && ggml_are_same_shape(norm, mul) &&
               weight->type == GGML_TYPE_F32 && weight->ne[0] == 128 && ggml_nelements(weight) == 128 &&
               ggml_is_contiguous(weight) && bounded(weight, info) && bounded(x, info) && (x->ne[1] + 3) / 4 <= info.max_group_count[0];
    }
    static bool cast_supports(ggml_backend_dev_t device, const ggml_tensor* rope, const ggml_tensor* scale, const ggml_tensor* copy) {
        ggml_ext_vulkan_device_info info;
        api(device)->device_info(device, &info);
        // Workspace writeback uses Vulkan's four-byte copy granularity.
        return info.float16 && scale->op == GGML_OP_SCALE && scale->src[0] == rope && scale->type == GGML_TYPE_F32 &&
               param(scale, 1) == 0.f && copy->op == GGML_OP_CPY && copy->src[0] == scale && copy->type == GGML_TYPE_F16 &&
               ggml_is_contiguous(copy) && aligned(copy, 4) && ggml_are_same_shape(scale, copy);
    }
    static int match(ggml_backend_dev_t device, const ggml_cgraph* graph, int first, void*) {
        auto* g = const_cast<ggml_cgraph*>(graph);
        if (first + 2 >= ggml_graph_n_nodes(g) || !fused_supports(device, ggml_graph_node(g, first), ggml_graph_node(g, first + 1), ggml_graph_node(g, first + 2)))
            return 0;
        return first + 4 < ggml_graph_n_nodes(g) && cast_supports(device, ggml_graph_node(g, first + 2), ggml_graph_node(g, first + 3), ggml_graph_node(g, first + 4)) ? 5 : 3;
    }
    static void optimize(ggml_backend_dev_t device, ggml_cgraph* graph, void*) {
        for (int last = 2; last < ggml_graph_n_nodes(graph); ++last) {
            const auto* output = ggml_graph_node(graph, last);
            const auto* scale  = output->op == GGML_OP_CPY ? output->src[0] : nullptr;
            const auto* rope   = scale && scale->op == GGML_OP_SCALE ? scale->src[0] : output;
            if (!is_rope(rope))
                continue;
            const auto* mul  = rope->src[0];
            const auto* norm = mul->src[0];
            if (!norm || !fused_supports(device, norm, mul, rope))
                continue;
            const int count = output == rope ? 3 : cast_supports(device, rope, scale, output) ? 5
                                                                                              : 0;
            if (!count)
                continue;
            const ggml_tensor* nodes[] = {norm, mul, rope, scale, output};
            int indices[5], next = count - 1;
            for (int i = last; i >= std::max(0, last - 64) && next >= 0; --i)
                if (ggml_graph_node(graph, i) == nodes[next])
                    indices[next--] = i;
            if (next >= 0)
                continue;
            int first = indices[0];
            for (int i = 1; i < count; ++i) {
                if (!ggml_ext_graph_delay(graph, first, i, indices[i]))
                    break;
                first = indices[i] - i;
            }
        }
    }
    static ggml_status prepare(ggml_backend_t, ggml_tensor* const* nodes, int count, void**, size_t* bytes, void*) {
        *bytes = ggml_nbytes(nodes[count - 1]);
        return GGML_STATUS_SUCCESS;
    }
    static void destroy(void*, void*) {}
    static bool overlaps(const ggml_ext_vulkan_buffer_info& a, const ggml_ext_vulkan_buffer_info& b) {
        return a.storage == b.storage && a.offset < b.offset + b.size && b.offset < a.offset + a.size;
    }
    static ggml_status execute(const ggml_ext_execution_context* execution, ggml_tensor* const* nodes, int count, void*, void*) {
        auto device         = ggml_backend_get_device(execution->backend);
        const auto* service = api(device);
        ggml_ext_vulkan_device_info info;
        if (!service || !service->device_info(device, &info))
            return GGML_STATUS_FAILED;
        const auto* rope                   = nodes[count == 1 ? 0 : 2];
        const auto* x                      = count == 1 ? rope->src[0] : nodes[0]->src[0];
        const auto* pe                     = rope->src[1];
        const auto* output                 = nodes[count - 1];
        ggml_ext_vulkan_binding bindings[] = {{x}, {pe}, {output}, {count == 1 ? nullptr : nodes[1]->src[1]}};
        ggml_ext_vulkan_buffer_info buffers[4];
        const int n_buffers = count == 1 ? 3 : 4;
        for (int i = 0; i < n_buffers; ++i)
            if (!service->buffer_info(execution->backend, bindings + i, buffers + i))
                return GGML_STATUS_FAILED;
        bool alias             = overlaps(buffers[2], buffers[0]) || overlaps(buffers[2], buffers[1]) || (count != 1 && overlaps(buffers[2], buffers[3]));
        const auto destination = bindings[2];
        if (alias) {
            bindings[2] = {nullptr, execution->workspace, 0, ggml_nbytes(output)};
            if (!service->buffer_info(execution->backend, bindings + 2, buffers + 2))
                return GGML_STATUS_FAILED;
        }
        FusedParameters params = {};
        auto& pc               = params.rope;
        pc.half_dim            = uint32_t(x->ne[0] / 2);
        pc.rot_half            = uint32_t(pe->ne[2]);
        pc.heads               = uint32_t(x->ne[1]);
        pc.tokens              = uint32_t(x->ne[2]);
        pc.pairs               = uint32_t(ggml_nelements(x) / 2);
        for (int i = 0; i < 4; ++i) {
            pc.x_stride[i]  = uint32_t(x->nb[i] / 4);
            pc.pe_stride[i] = uint32_t(pe->nb[i] / 4);
        }
        pc.x_offset   = buffers[0].shader_offset / 4;
        pc.pe_offset  = buffers[1].shader_offset / 4;
        pc.dst_offset = buffers[2].shader_offset / uint32_t(ggml_type_size(output->type));
        std::memcpy(&pc.interleaved, ggml_ext_op_get(rope) ? ggml_ext_op_attributes(rope) : rope->op_params, 4);
        uint32_t groups[3]                = {1, 1, 1};
        ggml_ext_vulkan_pipeline pipeline = {};
        pipeline.name                     = "sd.rope_apply.vulkan";
        pipeline.spirv                    = sd_vulkan_shaders::rope_apply;
        pipeline.spirv_size               = sizeof(sd_vulkan_shaders::rope_apply);
        pipeline.binding_count            = n_buffers;
        pipeline.push_constant_size       = sizeof(Parameters);
        if (count == 1) {
            const uint32_t n_groups = uint32_t((uint64_t(pc.pairs) + 255) / 256);
            groups[0]               = std::min(n_groups, info.max_group_count[0]);
            groups[1]               = (n_groups + groups[0] - 1) / groups[0];
        } else {
            pipeline.name                   = count == 5 ? "sd.rms_rope_cast.vulkan" : "sd.rms_rope_apply.vulkan";
            pipeline.spirv                  = count == 5 ? sd_vulkan_shaders::rms_rope_cast : sd_vulkan_shaders::rms_rope_apply;
            pipeline.spirv_size             = count == 5 ? sizeof(sd_vulkan_shaders::rms_rope_cast) : sizeof(sd_vulkan_shaders::rms_rope_apply);
            pipeline.push_constant_size     = sizeof(FusedParameters);
            pipeline.full_subgroups         = true;
            pipeline.required_subgroup_size = info.subgroup_size_control ? 32 : 0;
            params.eps                      = param(nodes[0], 0);
            params.w_offset                 = buffers[3].shader_offset / 4;
            params.output_scale             = count == 5 ? param(nodes[3], 0) : 1.f;
            groups[0]                       = (pc.heads + 3) / 4;
            groups[1]                       = std::min(pc.tokens, info.max_group_count[1]);
            groups[2]                       = std::min(uint32_t(x->ne[3]), info.max_group_count[2]);
        }
        auto status = service->dispatch(execution->backend, &pipeline, bindings, &params, groups);
        if (status == GGML_STATUS_SUCCESS && alias)
            status = service->copy(execution->backend, bindings + 2, &destination);
        return status;
    }
}

void sd_ggml_register_vulkan_rope(ggml_backend_dev_t device) {
    if (!sd_vulkan_rope::api(device))
        return;
    ggml_ext_kernel_desc desc = {};
    desc.abi_version          = GGML_EXT_ABI_VERSION;
    desc.name                 = "sd.rope_apply.vulkan.v1";
    desc.op                   = GGML_OP_CUSTOM;
    desc.custom_op            = sd_ggml_rope_op();
    desc.priority             = 100;
    desc.supports             = sd_vulkan_rope::supports;
    desc.prepare              = sd_vulkan_rope::prepare;
    desc.execute              = sd_vulkan_rope::execute;
    desc.destroy              = sd_vulkan_rope::destroy;
    GGML_ASSERT(ggml_ext_register_kernel(device, &desc));
    desc.name      = "sd.rope_apply.vulkan.override.v1";
    desc.op        = GGML_OP_ROPE_APPLY;
    desc.custom_op = nullptr;
    GGML_ASSERT(ggml_ext_register_kernel(device, &desc));
    desc.name     = "sd.rms_rope_apply.vulkan.v1";
    desc.op       = GGML_OP_RMS_NORM;
    desc.priority = 200;
    desc.supports = nullptr;
    desc.match    = sd_vulkan_rope::match;
    desc.optimize = sd_vulkan_rope::optimize;
    GGML_ASSERT(ggml_ext_register_kernel(device, &desc));
}
