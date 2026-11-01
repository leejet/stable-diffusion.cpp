#include <algorithm>
#include <array>
#include <climits>
#include <cstring>
#include <vector>
#include "core/ggml_extensions.h"
#include "ggml-ext-vulkan.h"
#include "modulate_convrot.h"
#include "mul_mat_i8_residual_cm2.h"
#include "mul_mat_i8_swiglu_split_cm2.h"
#include "mul_mat_i8_tensorwise.h"
#include "mul_mat_i8_tensorwise_cm1.h"
#include "mul_mat_i8_tensorwise_cm2.h"
#include "quantize_i8_convrot.h"
#include "quantize_i8_convrot_register.h"
#include "quantize_i8_convrot_subgroup.h"
#include "rms_modulate_convrot.h"
#include "sigmoid_quantize_i8_convrot_6144.h"
#include "swiglu_quantize_i8_convrot.h"
#include "swiglu_quantize_i8_convrot_16384.h"

namespace sd_vulkan_i8 {
    struct Device {
        const ggml_ext_vulkan_api* api;
        ggml_ext_vulkan_device_info info;
        ggml_ext_vulkan_compute_info compute;
        bool subgroup, cm2, cm2_small;
        uint32_t cm_m, cm_n, cm_k;
    };
    static bool device_info(ggml_backend_dev_t dev, Device& d) {
        auto reg     = ggml_backend_dev_backend_reg(dev);
        auto query   = reinterpret_cast<ggml_ext_vulkan_api_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_ext_vulkan_v1"));
        auto compute = reinterpret_cast<ggml_ext_vulkan_compute_api_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_ext_vulkan_compute_v1"));
        d            = {};
        if (!query || !compute)
            return false;
        d.api        = query();
        auto service = compute();
        if (!d.api->device_info(dev, &d.info) || !service->properties(dev, &d.compute))
            return false;
        d.subgroup = d.info.subgroup_size == 32 && d.info.full_subgroups && d.info.subgroup_shuffle && d.compute.subgroup_arithmetic && d.info.max_group_invocations >= 256;
        ggml_ext_vulkan_matrix_properties prop;
        for (uint32_t i = 0; service->matrix_properties(dev, i, &prop); ++i) {
            if (prop.a != GGML_TYPE_I8 || prop.b != GGML_TYPE_I8 || prop.accumulator != GGML_TYPE_I32 || prop.result != GGML_TYPE_I32 || prop.saturating || !prop.m || !prop.n || !prop.k)
                continue;
            if (!prop.flexible && !prop.workgroup_scope && !d.cm_m && sizeof(sd_vulkan_shaders::mul_mat_i8_tensorwise_cm1) > 4) {
                d.cm_m = prop.m;
                d.cm_n = prop.n;
                d.cm_k = prop.k;
            }
            if (prop.flexible && prop.workgroup_scope && prop.invocations == 128 && 128 % prop.m == 0 && 64 % prop.n == 0 && 64 % prop.k == 0 && sizeof(sd_vulkan_shaders::mul_mat_i8_tensorwise_cm2) > 4) {
                d.cm2 = true;
                d.cm2_small |= 64 % prop.m == 0 && 32 % prop.n == 0;
            }
        }
        return d.info.integer_dot_product && sizeof(sd_vulkan_shaders::mul_mat_i8_tensorwise) > 4;
    }
    static const ggml_tensor* node(const ggml_cgraph* graph, int i) {
        return ggml_graph_node(const_cast<ggml_cgraph*>(graph), i);
    }
    static int count_nodes(const ggml_cgraph* graph) {
        return ggml_graph_n_nodes(const_cast<ggml_cgraph*>(graph));
    }
    static int param_i32(const ggml_tensor* node, int i) {
        return node->op_params[i];
    }
    static float param_f32(const ggml_tensor* node, int i) {
        float value;
        std::memcpy(&value, node->op_params + i, 4);
        return value;
    }
    static bool is_convrot(const ggml_tensor* node) {
        int group;
        return sd_ggml_convrot_describe(node, &group);
    }
    static bool is_mat(const ggml_tensor* node) {
        SDI8Operation op;
        return sd_ggml_i8_describe(node, &op);
    }
    static bool aligned(const ggml_tensor* t, size_t alignment) {
        size_t offset = 0;
        while (!t->data && t->view_src) {
            offset += t->view_offs;
            t = t->view_src;
        }
        return (reinterpret_cast<uintptr_t>(t->data) + offset) % alignment == 0;
    }
    static bool bounded(const Device& d, const ggml_tensor* t, bool descriptor_aligned = true) {
        return t && ggml_nbytes(t) <= d.info.max_buffer_range - (d.info.buffer_alignment - 1) &&
               aligned(t, descriptor_aligned ? d.info.buffer_alignment : 4);
    }
    static bool can_modulate(const Device& d, int64_t k, bool affine) {
        return (k == 4096 || k == 6144 || (affine && k == 5376)) && d.info.max_group_invocations >= 512 && d.info.max_group_size[0] >= 512 && d.compute.shared_memory >= (k + 1024) * 4;
    }
    static bool can_quantize(const Device& d, const ggml_tensor* dst) {
        int group;
        if (!sd_ggml_convrot_describe(dst, &group) || group != 256 || !bounded(d, dst) || !bounded(d, dst->src[0]))
            return false;
        const auto* x = dst->src[0];
        return x->ne[0] <= UINT32_MAX && ggml_nrows(x) <= UINT32_MAX && d.info.max_group_invocations >= 256 && d.info.max_group_size[0] >= 256;
    }
    static bool can_matmul(const Device& d, const ggml_tensor* dst) {
        SDI8Operation op;
        if (!sd_ggml_i8_describe(dst, &op) || !op.packed || op.group_size != 256 || !op.weight_scale || !bounded(d, dst))
            return false;
        for (auto* t : {op.weight, op.input, op.weight_scale, op.bias})
            if (t && !bounded(d, t))
                return false;
        const uint64_t cols = op.weight->ne[1], rows = ggml_nrows(dst);
        return op.weight->ne[0] % 256 == 0 && op.weight->ne[0] <= UINT32_MAX && cols <= UINT32_MAX && rows <= UINT32_MAX &&
               (cols + 15) / 16 <= d.info.max_group_count[0] && (rows + 15) / 16 <= d.info.max_group_count[1] &&
               d.info.max_group_invocations >= 256 && d.info.max_group_size[0] >= 16 && d.info.max_group_size[1] >= 16;
    }
    static bool can_fuse_sigmoid_convrot(const Device& device, const ggml_cgraph* graph, int i) {
        if (i + 2 >= count_nodes(graph) || !device.subgroup) {
            return false;
        }
        const ggml_tensor* sigmoid = node(graph, i);
        const ggml_tensor* mul     = node(graph, i + 1);
        const ggml_tensor* quant   = node(graph, i + 2);
        if (sigmoid->op != GGML_OP_UNARY || ggml_get_unary_op(sigmoid) != GGML_UNARY_OP_SIGMOID ||
            mul->op != GGML_OP_MUL || mul->src[1] != sigmoid ||
            !is_convrot(quant) || quant->src[0] != mul ||
            sigmoid->ne[0] != 6144 || !ggml_are_same_shape(sigmoid, mul->src[0]) ||
            !can_quantize(device, quant)) {
            return false;
        }
        for (const ggml_tensor* tensor : {sigmoid->src[0], mul->src[0]}) {
            if (tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor) ||
                tensor->view_offs % device.info.buffer_alignment != 0) {
                return false;
            }
        }
        return true;
    }

    static bool can_fuse_swiglu_convrot(const Device& device, const ggml_cgraph* graph, int i) {
        if (i + 1 >= count_nodes(graph) || !device.subgroup) {
            return false;
        }
        const ggml_tensor* glu   = node(graph, i);
        const ggml_tensor* quant = node(graph, i + 1);
        if (glu->op != GGML_OP_GLU || ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU ||
            glu->src[1] == nullptr || param_i32(glu, 1) != 0 ||
            !is_convrot(quant) || quant->src[0] != glu ||
            !can_quantize(device, quant) || !ggml_is_contiguous(glu->src[0]) || !ggml_is_contiguous(glu->src[1])) {
            return false;
        }
        for (const ggml_tensor* tensor : {glu->src[0], glu->src[1]}) {
            if (tensor->type != GGML_TYPE_F32 || tensor->view_offs % device.info.buffer_alignment != 0) {
                return false;
            }
        }
        return true;
    }

    static int match_modulate_convrot(const Device& device, const ggml_cgraph* graph, int quant_idx, int* indices) {
        if (quant_idx < 1 || quant_idx >= count_nodes(graph)) {
            return 0;
        }
        const ggml_tensor* quant = node(graph, quant_idx);
        if (!is_convrot(quant) || !can_quantize(device, quant)) {
            return 0;
        }
        const ggml_tensor* nodes[4]{};

        const ggml_tensor* mod = nullptr;
        const bool affine      = sd_ggml_modulate_describe(quant->src[0]);
        const int count        = affine ? 2 : 3;
        if (affine) {
            nodes[0]                  = quant->src[0];
            nodes[1]                  = quant;
            mod                       = nodes[0]->src[1];
            const ggml_tensor* vector = nodes[0]->src[2];
            if (vector && (vector->type != GGML_TYPE_F32 || !ggml_is_contiguous(vector) ||
                           vector->ne[0] != nodes[0]->ne[0] || ggml_nelements(vector) != nodes[0]->ne[0])) {
                return 0;
            }
        } else {
            const ggml_tensor* mul = quant->src[0];
            if (mul->op != GGML_OP_MUL || mul->src[0]->op != GGML_OP_NORM) {
                return 0;
            }
            nodes[0] = mul->src[0];
            nodes[1] = mul;
            nodes[2] = quant;
            mod      = mul->src[1];
        }
        const ggml_tensor* x = nodes[0]->src[0];
        if ((x->ne[0] != 4096 && x->ne[0] != 6144 && x->ne[0] != 5376) ||
            !can_modulate(device, x->ne[0], affine) ||
            x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || mod->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(mod) || mod->ne[0] != x->ne[0] || ggml_nelements(mod) != x->ne[0]) {
            return 0;
        }
        for (int j = 0; j < count; ++j) {
            if (j < count - 1 && (nodes[j]->type != GGML_TYPE_F32 || !ggml_is_contiguous(nodes[j]) || !ggml_are_same_shape(x, nodes[j]))) {
                return 0;
            }
        }
        int next = count - 1;
        for (int j = quant_idx; j >= std::max(0, quant_idx - 64) && next >= 0; --j) {
            if (node(graph, j) == nodes[next]) {
                indices[next--] = j;
            }
        }
        return next < 0 ? count : 0;
    }

    static int match_rms_modulate_convrot(const Device& device, const ggml_cgraph* graph, int q, int* indices) {
        if (!can_modulate(device, 5376, true) || match_modulate_convrot(device, graph, q, indices + 2) != 2) {
            return 0;
        }
        const auto weighted = node(graph, indices[2])->src[0];
        if (weighted->op != GGML_OP_MUL || weighted->src[0]->op != GGML_OP_RMS_NORM || weighted->ne[0] != 5376) {
            return 0;
        }
        const auto norm   = weighted->src[0];
        const auto x      = norm->src[0];
        const auto weight = weighted->src[1];
        if (x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || weight->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(weight) || ggml_nelements(weight) != 5376 || weight->ne[0] != 5376) {
            return 0;
        }
        int next = 1;
        for (int j = indices[2] - 1; j >= std::max(0, indices[2] - 64) && next >= 0; --j) {
            if (node(graph, j) == (next == 1 ? weighted : norm)) {
                indices[next--] = j;
            }
        }

        return next < 0 ? 4 : 0;
    }

    static int can_fuse_modulate_convrot(const Device& device, const ggml_cgraph* graph, int i) {
        const ggml_op op = node(graph, i)->op;
        if (op == GGML_OP_RMS_NORM) {
            int indices[4];
            if (match_rms_modulate_convrot(device, graph, i + 3, indices) != 4) {
                return 0;
            }
            for (int j = 0; j < 4; ++j) {
                if (indices[j] != i + j) {
                    return 0;
                }
            }
            return 4;
        }
        if (op != GGML_OP_NORM && !sd_ggml_modulate_describe(node(graph, i))) {
            return 0;
        }
        const int count = op == GGML_OP_NORM ? 3 : 2;
        int indices[4];
        if (match_modulate_convrot(device, graph, i + count - 1, indices) != count) {
            return 0;
        }
        for (int j = 0; j < count; ++j) {
            if (indices[j] != i + j) {
                return 0;
            }
        }
        return count;
    }

    static bool match_i8_swiglu_split(const Device& device, const ggml_cgraph* graph, int i, int* indices) {
        if (i < 2 || i >= count_nodes(graph) || !device.cm2_small) {
            return false;
        }
        const ggml_tensor* glu = node(graph, i);
        if (glu->op != GGML_OP_GLU || ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU ||
            param_i32(glu, 1) != 0 || glu->src[1] == nullptr || glu->ne[0] % 128 != 0 ||
            glu->type != GGML_TYPE_F32 || !ggml_is_contiguous(glu) ||
            glu->view_offs % device.info.buffer_alignment != 0) {
            return false;
        }
        const ggml_tensor* gate = glu->src[0];
        const ggml_tensor* up   = glu->src[1];
        if (gate == up || gate->src[1] != up->src[1] || !ggml_are_same_shape(gate, up)) {
            return false;
        }
        for (auto mat : {gate, up}) {
            if (!is_mat(mat) || mat->view_src != nullptr || mat->src[3] != nullptr ||
                !can_matmul(device, mat)) {
                return false;
            }
        }
        int found = 0;
        for (int j = std::max(0, i - 32); j < i; ++j) {
            if (node(graph, j) == gate || node(graph, j) == up) {
                indices[found++] = j;
            }
        }
        if (found != 2) {
            return false;
        }
        indices[2] = i;

        return true;
    }

    static bool can_fuse_i8_swiglu_split(const Device& device, const ggml_cgraph* graph, int i) {
        int indices[3];
        return match_i8_swiglu_split(device, graph, i + 2, indices) && indices[0] == i && indices[1] == i + 1;
    }

    static bool can_fuse_i8_swiglu(const Device& device, const ggml_cgraph* graph, int i) {
        if (i + 1 >= count_nodes(graph) || !device.cm2_small) {
            return false;
        }
        const ggml_tensor* mat = node(graph, i);
        const ggml_tensor* glu = node(graph, i + 1);
        return is_mat(mat) && can_matmul(device, mat) &&
               mat->view_src == nullptr && mat->src[3] == nullptr && mat->ne[0] % 256 == 0 &&
               glu->op == GGML_OP_GLU && ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU &&
               param_i32(glu, 1) == 0 && glu->src[0] == mat && glu->src[1] == nullptr &&
               glu->type == GGML_TYPE_F32 && ggml_is_contiguous(glu) &&
               glu->view_offs % device.info.buffer_alignment == 0;
    }

    static int match_i8_residual(const Device& device, const ggml_cgraph* graph, int add_idx, int* indices) {
        if (add_idx < 2 || add_idx >= count_nodes(graph) || !device.cm2) {
            return 0;
        }
        const ggml_tensor* dst = node(graph, add_idx);
        if (dst->op != GGML_OP_ADD) {
            return 0;
        }
        const ggml_tensor* mul = dst->src[0]->op == GGML_OP_MUL ? dst->src[0] : dst->src[1];
        if (mul->op != GGML_OP_MUL) {
            return 0;
        }
        const ggml_tensor* scaled_mat = mul->src[0];
        const bool scaled             = scaled_mat->op == GGML_OP_SCALE;
        const ggml_tensor* mat        = scaled ? scaled_mat->src[0] : scaled_mat;
        if (!is_mat(mat) || !can_matmul(device, mat) || mat->src[3] != nullptr ||
            (scaled && param_f32(scaled_mat, 1) != 0.0f)) {
            return 0;
        }
        const int count             = scaled ? 4 : 3;
        const ggml_tensor* nodes[4] = {mat, scaled ? scaled_mat : mul, scaled ? mul : dst, dst};

        const ggml_tensor* gate     = mul->src[1];
        const ggml_tensor* residual = dst->src[0] == mul ? dst->src[1] : dst->src[0];
        if (!ggml_are_same_shape(mat, dst) || !ggml_are_same_shape(residual, dst) ||
            gate->ne[0] != mat->ne[0] || ggml_nelements(gate) != mat->ne[0]) {
            return 0;
        }
        for (const ggml_tensor* tensor : {mat, scaled_mat, mul, gate, residual, dst}) {
            if (tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor) ||
                tensor->view_offs % device.info.buffer_alignment != 0) {
                return 0;
            }
        }
        int next = count - 1;
        for (int j = add_idx; j >= std::max(0, add_idx - 32) && next >= 0; --j) {
            if (node(graph, j) == nodes[next]) {
                indices[next--] = j;
            }
        }
        return next < 0 ? count : 0;
    }

    static int can_fuse_i8_residual(const Device& device, const ggml_cgraph* graph, int i) {
        if (i + 2 >= count_nodes(graph) || !is_mat(node(graph, i))) {
            return 0;
        }
        const int count = node(graph, i + 1)->op == GGML_OP_SCALE ? 4 : 3;
        int indices[4];
        if (match_i8_residual(device, graph, i + count - 1, indices) != count) {
            return 0;
        }
        for (int j = 0; j < count; ++j) {
            if (indices[j] != i + j) {
                return 0;
            }
        }
        return count;
    }

    static int match_i8_row_concat(const Device& device, const ggml_cgraph* graph, int end, int* indices) {
        const ggml_tensor* concat = node(graph, end);
        if (concat->op != GGML_OP_CONCAT || param_i32(concat, 0) != 1 ||
            concat->type != GGML_TYPE_F32 || concat->ne[2] != 1 || concat->ne[3] != 1 ||
            !ggml_is_contiguous(concat) || !ggml_is_contiguous(concat->src[0]) ||
            concat->view_offs % device.info.buffer_alignment != 0 ||
            ggml_nbytes(concat->src[0]) % device.info.buffer_alignment != 0) {
            return 0;
        }
        const ggml_tensor* tail = concat->src[1];
        int count               = 0;
        for (int j = end - 1; j >= std::max(0, end - 32); --j) {
            if (node(graph, j) != tail) {
                continue;
            }
            if (is_mat(tail) && can_matmul(device, tail)) {
                indices[0] = j;
                count      = 1;
            } else {
                count = match_i8_residual(device, graph, j, indices);
            }
            break;
        }
        if (!count) {
            return 0;
        }
        indices[count++] = end;

        return count;
    }

    static int can_fuse_i8_row_concat(const Device& device, const ggml_cgraph* graph, int i) {
        if (!is_mat(node(graph, i))) {
            return 0;
        }
        for (int end = i + 1; end < std::min(i + 5, count_nodes(graph)); ++end) {
            int indices[5];
            const int count = match_i8_row_concat(device, graph, end, indices);
            if (count != end - i + 1) {
                continue;
            }
            bool consecutive = true;
            for (int j = 0; j < count; ++j) {
                consecutive &= indices[j] == i + j;
            }
            if (consecutive) {
                return count;
            }
        }
        return 0;
    }

    struct QuantParameters {
        uint32_t k, rows;
    };
    struct ModulateParameters {
        uint32_t rows, x_offset, m_offset, h_offset;
        float eps;
        uint32_t k, w_offset;
    };
    struct MatParameters {
        uint32_t n, rows, k, has_bias, row_offset;
        float output_scale;
        uint32_t group_cols;
    };
    static_assert(sizeof(QuantParameters) == 8 && sizeof(ModulateParameters) == 28 && sizeof(MatParameters) == 28, "INT8 push constant layout mismatch");
    template <size_t N>
    static ggml_ext_vulkan_pipeline shader(const char* name, const uint32_t (&spirv)[N], uint32_t bindings, uint32_t constants) {
        return {name, spirv, sizeof(spirv), bindings, constants, nullptr, 0, 0, false};
    }
#define SD_VK_PIPELINE(NAME, BINDINGS, PARAMS) shader("sd." #NAME ".vulkan", sd_vulkan_shaders::NAME, BINDINGS, sizeof(PARAMS))
    static ggml_ext_vulkan_binding range(ggml_ext_vulkan_binding binding, size_t offset, size_t size) {
        binding.offset += offset;
        binding.size = size;
        return binding;
    }
    static bool overlap(const ggml_ext_vulkan_buffer_info& a, const ggml_ext_vulkan_buffer_info& b) {
        return a.storage == b.storage && a.offset < b.offset + b.size && b.offset < a.offset + a.size;
    }
    static bool output_binding(const ggml_ext_execution_context* e, const Device& d, const ggml_tensor* output, const std::vector<ggml_ext_vulkan_binding>& inputs, ggml_ext_vulkan_binding& binding, bool& alias, const ggml_tensor* residual = nullptr) {
        binding = {output};
        alias   = false;
        ggml_ext_vulkan_buffer_info dst, src;
        if (!d.api->buffer_info(e->backend, &binding, &dst))
            return false;
        for (const auto& input : inputs) {
            if (!d.api->buffer_info(e->backend, &input, &src))
                return false;
            // The residual epilogue reads each element before writing that same element.
            if (residual && input.tensor == residual && dst.storage == src.storage && dst.offset == src.offset && dst.size == src.size)
                continue;
            alias |= overlap(dst, src);
        }
        if (alias)
            binding = {nullptr, e->workspace, 0, ggml_nbytes(output)};
        return true;
    }
    static ggml_status finish(const ggml_ext_execution_context* e, const Device& d, const ggml_tensor* output, const ggml_ext_vulkan_binding& binding, bool alias, ggml_status status) {
        if (status == GGML_STATUS_SUCCESS && alias) {
            ggml_ext_vulkan_binding dst = {output};
            return d.api->copy(e->backend, &binding, &dst);
        }
        return status;
    }
    static ggml_status quantize(const ggml_ext_execution_context* e, const Device& d, ggml_tensor* const* nodes, int count) {
        const auto* output                          = nodes[count - 1];
        const auto* first                           = nodes[0];
        const bool modulate                         = count > 1 && (first->op == GGML_OP_NORM || sd_ggml_modulate_describe(first) || first->op == GGML_OP_RMS_NORM);
        const bool rms                              = modulate && first->op == GGML_OP_RMS_NORM;
        const bool affine                           = modulate && (rms || sd_ggml_modulate_describe(first));
        const bool sigmoid                          = count > 1 && first->op == GGML_OP_UNARY;
        const auto* x                               = count == 1 ? output->src[0] : first->src[0];
        const auto* up                              = count == 1 || modulate ? nullptr : sigmoid ? nodes[1]->src[0]
                                                                                                 : first->src[1];
        const int extra                             = rms ? 2 : 0;
        const auto* mod                             = modulate ? nodes[extra + (affine ? 0 : 1)]->src[1] : nullptr;
        const auto* shift                           = modulate ? (affine ? nodes[extra]->src[2] : mod) : nullptr;
        const auto* weight                          = rms ? nodes[1]->src[1] : x;
        std::vector<ggml_ext_vulkan_binding> inputs = {{x}};
        if (up)
            inputs.push_back({up});
        if (modulate) {
            inputs.push_back({mod});
            if (shift)
                inputs.push_back({shift});
            if (rms)
                inputs.push_back({weight});
        }
        ggml_ext_vulkan_binding dst;
        bool alias;
        if (!output_binding(e, d, output, inputs, dst, alias))
            return GGML_STATUS_FAILED;
        const uint32_t k = uint32_t(x->ne[0]), rows = uint32_t(ggml_nrows(x));
        const size_t scale_offset = size_t(k) * ((uint64_t(rows) + 3) / 4 * 4);
        auto packed = range(dst, 0, scale_offset), scales = range(dst, scale_offset, size_t(rows) * 4);
        const uint32_t groups[] = {std::min(rows, d.info.max_group_count[0]), 1, 1};
        ggml_status status;
        if (modulate) {
            std::vector<ggml_ext_vulkan_binding> bindings = {{x}, {mod}, {shift ? shift : mod}, packed, scales};
            if (rms)
                bindings.push_back({weight});
            ModulateParameters pc = {rows, 0, 0, 0, affine && !rms ? 0.f : param_f32(first, 0), k, 0};
            uint32_t* offsets[]   = {&pc.x_offset, &pc.m_offset, &pc.h_offset, nullptr, nullptr, &pc.w_offset};
            for (size_t i = 0; i < bindings.size(); ++i)
                if (offsets[i]) {
                    ggml_ext_vulkan_buffer_info info;
                    if (!d.api->buffer_info(e->backend, &bindings[i], &info))
                        return GGML_STATUS_FAILED;
                    *offsets[i] = info.shader_offset / 4;
                }
            auto pipeline              = rms ? SD_VK_PIPELINE(rms_modulate_convrot, 6, ModulateParameters) : SD_VK_PIPELINE(modulate_convrot, 5, ModulateParameters);
            const uint32_t constants[] = {k, uint32_t(affine), uint32_t(shift != nullptr)};
            pipeline.constants         = constants;
            pipeline.constant_count    = 3;
            status                     = d.api->dispatch(e->backend, &pipeline, bindings.data(), &pc, groups);
        } else {
            ggml_ext_vulkan_binding bindings[] = {{x}, packed, scales, {up}};
            auto pipeline                      = SD_VK_PIPELINE(quantize_i8_convrot, 3, QuantParameters);
            bool subgroup                      = d.subgroup && k >= 1024;
            uint32_t width                     = k;
            if (subgroup) {
                pipeline = SD_VK_PIPELINE(quantize_i8_convrot_subgroup, 3, QuantParameters);
                if (!up && ((uint64_t(k) + 2047) / 2048) * 8 <= 64) {
                    pipeline                = SD_VK_PIPELINE(quantize_i8_convrot_register, 3, QuantParameters);
                    pipeline.constants      = &width;
                    pipeline.constant_count = 1;
                }
            }
            if (up) {
                subgroup = true;
                pipeline = k == 16384 ? SD_VK_PIPELINE(swiglu_quantize_i8_convrot_16384, 4, QuantParameters) : SD_VK_PIPELINE(swiglu_quantize_i8_convrot, 4, QuantParameters);
                if (sigmoid)
                    pipeline = SD_VK_PIPELINE(sigmoid_quantize_i8_convrot_6144, 4, QuantParameters);
            }
            if (subgroup) {
                pipeline.full_subgroups         = d.info.full_subgroups;
                pipeline.required_subgroup_size = d.info.subgroup_size_control ? 32 : 0;
            }
            QuantParameters pc = {k, rows};
            status             = d.api->dispatch(e->backend, &pipeline, bindings, &pc, groups);
        }
        return finish(e, d, output, dst, alias, status);
    }
    static uint32_t group_columns(uint32_t cols, uint32_t k, uint32_t tile_cols, bool swiglu) {
        if (!cols || !k || !tile_cols)
            return 1;
        const uint64_t col_tiles         = (uint64_t(cols) + tile_cols - 1) / tile_cols;
        const uint64_t weight_tile_bytes = uint64_t(tile_cols) * k * (swiglu ? 2 : 1);
        const uint64_t growth            = std::clamp(uint64_t(k) / cols, uint64_t(1), uint64_t(2));
        return uint32_t(std::clamp(uint64_t(8) * 1024 * 1024 * growth / weight_tile_bytes, uint64_t(1), std::min(col_tiles, uint64_t(32))));
    }
    static ggml_status multiply(const ggml_ext_execution_context* e, const Device& d, ggml_tensor* const* nodes, int count) {
        const auto* output      = nodes[count - 1];
        const auto* concat      = output->op == GGML_OP_CONCAT ? output : nullptr;
        const auto* tail        = concat ? concat->src[1] : output;
        const auto* glu         = tail->op == GGML_OP_GLU ? tail : nullptr;
        const auto* mat         = glu ? glu->src[0] : tail;
        const ggml_tensor *gate = nullptr, *residual = nullptr;
        float output_scale = 1.f;
        if (tail->op == GGML_OP_ADD) {
            const auto* mul = tail->src[0]->op == GGML_OP_MUL ? tail->src[0] : tail->src[1];
            gate            = mul->src[1];
            residual        = tail->src[0] == mul ? tail->src[1] : tail->src[0];
            mat             = mul->src[0];
            if (mat->op == GGML_OP_SCALE) {
                output_scale = param_f32(mat, 0);
                mat          = mat->src[0];
            }
        }
        const auto* up_mat = glu ? glu->src[1] : nullptr;
        SDI8Operation op, up;
        if (!sd_ggml_i8_describe(mat, &op) || (up_mat && !sd_ggml_i8_describe(up_mat, &up)))
            return GGML_STATUS_FAILED;
        const uint32_t k = uint32_t(op.weight->ne[0]), n = uint32_t(op.weight->ne[1]), rows = uint32_t(ggml_nrows(mat));
        const size_t scale_offset                   = size_t(k) * ((uint64_t(rows) + 3) / 4 * 4);
        std::vector<ggml_ext_vulkan_binding> inputs = {{op.weight}, {op.input}, {op.weight_scale}, {op.bias ? op.bias : op.weight_scale}};
        if (gate) {
            inputs.push_back({gate});
            inputs.push_back({residual});
        }
        if (up_mat) {
            inputs.push_back({up.weight});
            inputs.push_back({up.weight_scale});
        }
        if (concat && ggml_nbytes(concat->src[0]))
            inputs.push_back({concat->src[0]});
        ggml_ext_vulkan_binding dst;
        bool alias;
        if (!output_binding(e, d, output, inputs, dst, alias, concat ? nullptr : residual))
            return GGML_STATUS_FAILED;
        auto destination                   = concat ? range(dst, ggml_nbytes(concat->src[0]), ggml_nbytes(tail)) : dst;
        ggml_ext_vulkan_binding bindings[] = {{op.weight}, {op.input, nullptr, 0, scale_offset}, {op.weight_scale}, {op.input, nullptr, scale_offset, size_t(rows) * 4}, {op.bias ? op.bias : op.weight_scale}, destination, {up_mat ? up.weight : gate}, {up_mat ? up.weight_scale : residual}};
        auto dispatch                      = [&](ggml_ext_vulkan_pipeline pipeline, uint32_t tile_cols, uint32_t tile_rows,
                                                 std::initializer_list<uint32_t> constants, uint32_t row_offset, uint32_t dispatch_rows) {
            if (!dispatch_rows)
                return GGML_STATUS_SUCCESS;
            const uint32_t cols     = glu ? uint32_t(glu->ne[0]) : n;
            MatParameters pc        = {n * (up_mat ? 2u : 1u), rows, k, op.bias ? 1u : 0u, row_offset, output_scale, group_columns(cols, k, tile_cols, glu != nullptr)};
            pipeline.constants      = constants.begin();
            pipeline.constant_count = uint32_t(constants.size());
            const uint32_t groups[] = {(cols + tile_cols - 1) / tile_cols, (dispatch_rows + tile_rows - 1) / tile_rows, 1};
            return d.api->dispatch(e->backend, &pipeline, bindings, &pc, groups);
        };
        ggml_status status = GGML_STATUS_SUCCESS;
        if (d.cm2) {
            const bool large_k       = k % 4096 == 0 || (rows >= n && k >= n);
            const uint32_t full_rows = (glu || n % 128 == 0) ? rows / 128 * 128 : 0;
            auto pipeline            = residual ? SD_VK_PIPELINE(mul_mat_i8_residual_cm2, 8, MatParameters) : up_mat ? SD_VK_PIPELINE(mul_mat_i8_swiglu_split_cm2, 8, MatParameters)
                                                                                                                     : SD_VK_PIPELINE(mul_mat_i8_tensorwise_cm2, 6, MatParameters);
            const uint32_t tile_cols = glu ? 64 : 128;
            status                   = dispatch(pipeline, tile_cols, 128, {128, 1, glu ? 1u : 0u, tile_cols, glu || large_k ? 128u : 64u}, 0, full_rows);
            if (status == GGML_STATUS_SUCCESS) {
                const bool small         = d.cm2_small && (glu || n % 128 == 0) && rows - full_rows <= 32;
                const uint32_t tail_cols = small || glu ? 64 : 128, tail_rows = small ? 32 : glu ? 128
                                                                                                 : 64;
                status = dispatch(pipeline, tail_cols, tail_rows, {tail_rows, small ? 2u : 0u, glu ? 1u : 0u, tail_cols, small ? 256u : 64u}, full_rows, rows - full_rows);
            }
        } else {
            uint32_t coop_rows = 0;
            if (d.cm_m && k % d.cm_k == 0 && n % (4 * d.cm_n) == 0) {
                const uint32_t tile_rows        = 4 * d.cm_m;
                coop_rows                       = rows / tile_rows * tile_rows;
                auto pipeline                   = SD_VK_PIPELINE(mul_mat_i8_tensorwise_cm1, 6, MatParameters);
                pipeline.full_subgroups         = d.info.full_subgroups;
                pipeline.required_subgroup_size = d.info.subgroup_size_control ? d.info.subgroup_size : 0;
                status                          = dispatch(pipeline, 4 * d.cm_n, tile_rows, {d.info.subgroup_size, d.cm_m, d.cm_n, d.cm_k}, 0, coop_rows);
            }
            if (status == GGML_STATUS_SUCCESS)
                status = dispatch(SD_VK_PIPELINE(mul_mat_i8_tensorwise, 6, MatParameters), 16, 16, {}, coop_rows, rows - coop_rows);
        }
        if (status == GGML_STATUS_SUCCESS && concat && ggml_nbytes(concat->src[0])) {
            ggml_ext_vulkan_binding src = {concat->src[0]}, prefix = range(dst, 0, ggml_nbytes(concat->src[0]));
            status = d.api->copy(e->backend, &src, &prefix);
        }
        return finish(e, d, output, dst, alias, status);
    }
    static bool quant_supports(ggml_backend_dev_t dev, const ggml_tensor* n, void*) {
        Device d;
        return device_info(dev, d) && can_quantize(d, n);
    }
    static bool mat_supports(ggml_backend_dev_t dev, const ggml_tensor* n, void*) {
        Device d;
        return device_info(dev, d) && can_matmul(d, n);
    }
    static int match(ggml_backend_dev_t dev, const ggml_cgraph* graph, int first, void*) {
        Device d;
        if (!device_info(dev, d) || d.info.fusion_disabled)
            return 0;
        if (int count = can_fuse_modulate_convrot(d, graph, first))
            return count;
        if (int count = can_fuse_i8_row_concat(d, graph, first))
            return count;
        if (int count = can_fuse_i8_residual(d, graph, first))
            return count;
        if (can_fuse_i8_swiglu_split(d, graph, first))
            return 3;
        if (can_fuse_i8_swiglu(d, graph, first))
            return 2;
        if (can_fuse_sigmoid_convrot(d, graph, first))
            return 3;
        if (can_fuse_swiglu_convrot(d, graph, first))
            return 2;
        return 0;
    }
    static void optimize(ggml_backend_dev_t dev, ggml_cgraph* graph, void*) {
        Device d;
        if (!device_info(dev, d) || d.info.fusion_disabled || d.compute.graph_optimization_disabled)
            return;
        for (int last = 1; last < count_nodes(graph); ++last) {
            int indices[6];
            int count = match_rms_modulate_convrot(d, graph, last, indices);
            if (!count)
                count = match_modulate_convrot(d, graph, last, indices);
            if (!count)
                count = match_i8_row_concat(d, graph, last, indices);
            if (!count)
                count = match_i8_residual(d, graph, last, indices);
            if (!count && match_i8_swiglu_split(d, graph, last, indices))
                count = 3;
            if (!count)
                continue;
            int first = indices[0];
            for (int j = 1; j < count; ++j) {
                if (!ggml_ext_graph_delay(graph, first, j, indices[j]))
                    break;
                first = indices[j] - j;
            }
        }
    }
    static bool may_alias(const ggml_tensor* output, const ggml_tensor* input) {
        size_t offset     = 0;
        const size_t size = ggml_nbytes(input);
        while (!input->data && input->view_src) {
            offset += input->view_offs;
            input = input->view_src;
        }
        // Unallocated inputs get allocator storage, separate from a preallocated output.
        if (!input->data)
            return false;
        if (!input->buffer)
            return true;
        if (output->buffer != input->buffer)
            return false;
        const auto dst = reinterpret_cast<uintptr_t>(output->data);
        const auto src = reinterpret_cast<uintptr_t>(input->data) + offset;
        return dst < src + size && src < dst + ggml_nbytes(output);
    }
    static ggml_status prepare(ggml_backend_t, ggml_tensor* const* nodes, int count, void**, size_t* bytes, void*) {
        const auto* output = nodes[count - 1];
        // Fresh outputs cannot alias external inputs except the residual's elementwise reuse.
        const bool fresh = !output->data && !output->view_src &&
                           (output->op == GGML_OP_CUSTOM || output->op == GGML_OP_GLU || output->op == GGML_OP_CONCAT ||
                            (output->op == GGML_OP_ADD && count > 1));
        *bytes           = fresh ? 0 : ggml_nbytes(output);
        if (output->data && output->buffer && !output->view_src) {
            bool alias = false;
            for (int i = 0; i < count; ++i)
                for (const auto* src : nodes[i]->src)
                    if (src && std::find(nodes, nodes + count, src) == nodes + count)
                        alias |= may_alias(output, src);
            if (!alias)
                *bytes = 0;
        }
        return GGML_STATUS_SUCCESS;
    }
    static void destroy(void*, void*) {}
    static ggml_status execute(const ggml_ext_execution_context* e, ggml_tensor* const* nodes, int count, void*, void*) {
        Device d;
        if (!device_info(ggml_backend_get_device(e->backend), d))
            return GGML_STATUS_FAILED;
        return is_convrot(nodes[count - 1]) ? quantize(e, d, nodes, count) : multiply(e, d, nodes, count);
    }
#undef SD_VK_PIPELINE
}

void sd_ggml_register_vulkan_i8(ggml_backend_dev_t device) {
    using namespace sd_vulkan_i8;
    Device info;
    if (!device_info(device, info))
        return;
    ggml_ext_kernel_desc d = {};
    d.abi_version          = GGML_EXT_ABI_VERSION;
    d.priority             = 100;
    d.prepare              = prepare;
    d.destroy              = destroy;
    d.execute              = execute;
    d.name                 = "sd.quantize_i8_convrot.vulkan.v1";
    d.op                   = GGML_OP_CUSTOM;
    d.custom_op            = sd_ggml_convrot_op();
    d.supports             = quant_supports;
    GGML_ASSERT(ggml_ext_register_kernel(device, &d));
    d.name      = "sd.linear_i8.vulkan.v1";
    d.op        = GGML_OP_CUSTOM;
    d.custom_op = sd_ggml_i8_op();
    d.supports  = mat_supports;
    GGML_ASSERT(ggml_ext_register_kernel(device, &d));
    d.priority  = 200;
    d.supports  = nullptr;
    d.match     = match;
    d.optimize  = optimize;
    d.name      = "sd.linear_i8.fused.vulkan.v1";
    d.op        = GGML_OP_CUSTOM;
    d.custom_op = sd_ggml_i8_op();
    GGML_ASSERT(ggml_ext_register_kernel(device, &d));
    d.optimize  = nullptr;
    d.name      = "sd.modulate_convrot.vulkan.v1";
    d.custom_op = sd_ggml_modulate_op();
    GGML_ASSERT(ggml_ext_register_kernel(device, &d));
    d.custom_op = nullptr;
    const struct {
        ggml_op op;
        const char* name;
    } fusions[] = {
        {GGML_OP_NORM, "sd.norm_convrot.vulkan.v1"},
        {GGML_OP_RMS_NORM, "sd.rms_modulate_convrot.vulkan.v1"},
        {GGML_OP_GLU, "sd.swiglu_convrot.vulkan.v1"},
        {GGML_OP_UNARY, "sd.sigmoid_convrot.vulkan.v1"}};
    for (const auto& fusion : fusions) {
        d.op   = fusion.op;
        d.name = fusion.name;
        GGML_ASSERT(ggml_ext_register_kernel(device, &d));
    }
}
