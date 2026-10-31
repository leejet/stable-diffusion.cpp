#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include "convrot_kernel.h"
#include "core/int8_ops.h"

namespace sd_convrot {
    struct DeviceInfo {
        int ordinal   = -1;
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
        auto query           = reinterpret_cast<ggml_ext_cuda_device_fn>(ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(device), "ggml_ext_cuda_device_v1"));
        int ordinal          = query ? query(device) : -1;
        cudaDeviceProp props = {};
        if (ordinal >= 0 && cudaGetDeviceProperties(&props, ordinal) == cudaSuccess)
            info = {ordinal, size_t(props.sharedMemPerBlockOptin)};
        cache.emplace(device, info);
        return info;
    }
    static bool fusion_disabled() {
        static const bool disabled = [] { const char* v = std::getenv("GGML_CUDA_DISABLE_FUSION"); return v && std::atoi(v) != 0; }();
        return disabled;
    }
    static float param_f32(const ggml_tensor* tensor, int index) {
        float value;
        std::memcpy(&value, tensor->op_params + index, sizeof(value));
        return value;
    }
    static bool f32(const ggml_tensor* tensor) {
        return tensor && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor);
    }
    static bool vector(const ggml_tensor* tensor, int64_t k) {
        return f32(tensor) && tensor->ne[0] == k && ggml_nelements(tensor) == k;
    }
    enum Kind { SCALE,
                NORM,
                RMS_MOD,
                SWIGLU,
                SILU,
                SIGMOID };
    struct Operation {
        Kind kind;
        const ggml_tensor* input      = nullptr;
        const ggml_tensor* up         = nullptr;
        const ggml_tensor* modulation = nullptr;
        const ggml_tensor* weight     = nullptr;
        const ggml_tensor* shift      = nullptr;
        int64_t k, rows, stride, up_stride, gate_offset = 0, up_offset = 0;
        float eps = 0.f, scale = 1.f, bias = 0.f;
    };
    static bool describe(ggml_tensor* const* nodes, int count, Operation& op) {
        int group;
        const auto* first = nodes[0];
        const auto* quant = nodes[count - 1];
        if (!sd_ggml_convrot_describe(quant, &group) || group != 256)
            return false;
        const auto* value = quant->src[0];
        op.k              = value->ne[0];
        op.rows           = ggml_nrows(value);
        op.input          = first->src[0];
        if (!f32(op.input) || !f32(first) || op.k > INT_MAX || op.rows > INT_MAX)
            return false;
        op.stride    = op.input->ne[0];
        op.up_stride = op.k;
        if (count == 2 && first->op == GGML_OP_SCALE && value == first) {
            op.kind  = SCALE;
            op.scale = param_f32(first, 0);
            op.bias  = param_f32(first, 1);
        } else if (count == 3 && first->op == GGML_OP_NORM && nodes[1]->op == GGML_OP_MUL && nodes[1]->src[0] == first && value == nodes[1]) {
            op.kind       = NORM;
            op.modulation = nodes[1]->src[1];
            op.eps        = param_f32(first, 0);
            if (op.k < 1024 || !vector(op.modulation, op.k))
                return false;
        } else if (count == 6 && first->op == GGML_OP_RMS_NORM && nodes[1]->op == GGML_OP_MUL && nodes[1]->src[0] == first &&
                   nodes[2]->op == GGML_OP_MUL && nodes[2]->src[0] == nodes[1] &&
                   nodes[3]->op == GGML_OP_ADD && nodes[3]->src[0] == nodes[1] && nodes[3]->src[1] == nodes[2] &&
                   nodes[4]->op == GGML_OP_ADD && nodes[4]->src[0] == nodes[3] && value == nodes[4]) {
            op.kind       = RMS_MOD;
            op.weight     = nodes[1]->src[1];
            op.modulation = nodes[2]->src[1];
            op.shift      = nodes[4]->src[1];
            op.eps        = param_f32(first, 0);
            if (op.k < 1024 || !vector(op.weight, op.k) || !vector(op.modulation, op.k) || !vector(op.shift, op.k))
                return false;
        } else if (count == 2 && first->op == GGML_OP_GLU && ggml_get_glu_op(first) == GGML_GLU_OP_SWIGLU && value == first) {
            op.kind = SWIGLU;
            op.up   = first->src[1] ? first->src[1] : op.input;
            if (!f32(op.up))
                return false;
            op.up_stride = op.up->ne[0];
            if (!first->src[1]) {
                if (first->op_params[1])
                    op.gate_offset = op.k;
                else
                    op.up_offset = op.k;
            }
        } else if (count == 3 && first->op == GGML_OP_UNARY && nodes[1]->op == GGML_OP_MUL && value == nodes[1] &&
                   (nodes[1]->src[0] == first || nodes[1]->src[1] == first)) {
            auto unary = ggml_get_unary_op(first);
            if (unary != GGML_UNARY_OP_SILU && unary != GGML_UNARY_OP_SIGMOID)
                return false;
            op.kind = unary == GGML_UNARY_OP_SILU ? SILU : SIGMOID;
            op.up   = nodes[1]->src[0] == first ? nodes[1]->src[1] : nodes[1]->src[0];
            if (!f32(op.up) || !ggml_are_same_shape(op.input, op.up))
                return false;
        } else
            return false;
        for (int i = 0; i < count - 1; ++i) {
            if (!f32(nodes[i]) || !ggml_are_same_shape(nodes[i], value))
                return false;
            for (const auto* input : {op.input, op.up, op.modulation, op.weight, op.shift})
                if (input == nodes[i])
                    return false;
        }
        return true;
    }
    static bool supported(const Operation& op, const DeviceInfo& device) {
        const size_t extra = op.kind == NORM ? 384 : op.kind == RMS_MOD ? 256
                                                                        : 128;
        return device.ordinal >= 0 && size_t(op.k) * sizeof(float) + extra <= device.shared;
    }
    static int match(ggml_backend_dev_t device, const ggml_cgraph* graph, int first, void*) {
        if (fusion_disabled())
            return 0;
        auto* g             = const_cast<ggml_cgraph*>(graph);
        const int available = ggml_graph_n_nodes(g) - first;
        const auto* node    = ggml_graph_node(g, first);
        const int count     = node->op == GGML_OP_RMS_NORM ? 6 : node->op == GGML_OP_NORM || node->op == GGML_OP_UNARY ? 3
                                                                                                                       : 2;
        if (available < count)
            return 0;
        ggml_tensor* nodes[6];
        for (int j = 0; j < count; ++j)
            nodes[j] = ggml_graph_node(g, first + j);
        // Native INT8 epilogues consume GEMM and GLU together; splitting them loses that fusion.
        if (node->op == GGML_OP_GLU && nodes[count - 1]->op == GGML_OP_QUANTIZE_I8_CONVROT &&
            node->src[0]->op == GGML_OP_MUL_MAT && node->src[0]->src[0]->type == GGML_TYPE_I8 &&
            node->src[0]->src[2] && !node->src[0]->src[3])
            return 0;
        Operation op;
        return describe(nodes, count, op) && supported(op, device_info(device)) ? count : 0;
    }
    static void optimize(ggml_backend_dev_t device, ggml_cgraph* graph, void*) {
        if (fusion_disabled())
            return;
        for (int i = 2; i < ggml_graph_n_nodes(graph); ++i) {
            auto* quant = ggml_graph_node(graph, i);
            int group;
            if (!sd_ggml_convrot_describe(quant, &group) || group != 256)
                continue;
            ggml_tensor* chain[6] = {};
            int count             = 0;
            auto* value           = quant->src[0];
            if (value->op == GGML_OP_ADD && value->src[0]->op == GGML_OP_ADD) {
                auto* add    = value->src[0];
                auto* weight = add->src[0];
                if (weight->op == GGML_OP_MUL) {
                    chain[0] = weight->src[0];
                    chain[1] = weight;
                    chain[2] = add->src[1];
                    chain[3] = add;
                    chain[4] = value;
                    chain[5] = quant;
                    count    = 6;
                }
            } else if (value->op == GGML_OP_MUL) {
                for (int src = 0; src < 2; ++src)
                    if (value->src[src]->op == GGML_OP_UNARY) {
                        chain[0] = value->src[src];
                        chain[1] = value;
                        chain[2] = quant;
                        count    = 3;
                        break;
                    }
            }
            Operation op;
            if (!count || !describe(chain, count, op) || !supported(op, device_info(device)))
                continue;
            int begin = -1;
            for (int j = std::max(0, i - 64); j < i; ++j)
                if (ggml_graph_node(graph, j) == chain[0]) {
                    begin = j;
                    break;
                }
            if (begin < 0)
                continue;
            // Move the owning producer with its in-place consumers so alias checks retain their storage relationship.
            for (int part = 1; part < count; ++part) {
                int next = -1;
                for (int j = begin + part; j <= i; ++j)
                    if (ggml_graph_node(graph, j) == chain[part]) {
                        next = j;
                        break;
                    }
                if (next < 0 || (next > begin + part && !ggml_ext_graph_delay(graph, begin, part, next)))
                    break;
                begin = next - part;
            }
        }
    }
    struct Plan {
        DeviceInfo device;
        Operation op;
        bool configured = false;
    };
    static ggml_status prepare(ggml_backend_t backend, ggml_tensor* const* nodes, int count, void** state, size_t* workspace, void*) {
        auto* plan   = new Plan{};
        *state       = plan;
        plan->device = device_info(ggml_backend_get_device(backend));
        if (!describe(nodes, count, plan->op) || !supported(plan->op, plan->device))
            return GGML_STATUS_FAILED;
        *workspace = ggml_nbytes(nodes[count - 1]);
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
    template <bool Swiglu = false, bool Normalize = false, bool Scaled = false, bool Sigmoid = false, bool RMS = false>
    static cudaError_t launch(Plan& p, const Operation& op, int8_t* output, cudaStream_t stream) {
        constexpr bool warp = Scaled || RMS;
        constexpr int extra = Normalize ? 384 : RMS ? 256
                                                    : 128;
        if (!p.configured) {
            auto status = cudaFuncSetAttribute(quantize<Swiglu, false, Normalize, Scaled, warp, float, Sigmoid, RMS>,
                                               cudaFuncAttributeMaxDynamicSharedMemorySize, int(p.device.shared - extra));
            if (status != cudaSuccess)
                return status;
            p.configured = true;
        }
        const int threads    = Normalize ? 1024 : RMS                               ? (op.k <= 1024 ? 128 : op.k <= 6144 ? 256
                                                                                                        : op.k <= 8192   ? 512
                                                                                                                         : 1024)
                                              : Swiglu && !Sigmoid && op.k >= 16384 ? 512
                                                                                    : 256;
        const int64_t padded = (op.rows + 3) / 4 * 4;
        auto data            = [](const ggml_tensor* t) { return t ? static_cast<const float*>(t->data) : nullptr; };
        quantize<Swiglu, false, Normalize, Scaled, warp, float, Sigmoid, RMS><<<unsigned(op.rows), threads, size_t(op.k) * sizeof(float), stream>>>(
            data(op.input) + op.gate_offset, output, reinterpret_cast<float*>(output + op.k * padded), op.k, op.rows, op.k,
            op.up ? data(op.up) + op.up_offset : nullptr, op.stride, op.up_stride, nullptr, nullptr, data(op.modulation), op.eps, op.scale, op.bias, data(op.weight), data(op.shift));
        return cudaGetLastError();
    }
    static ggml_status execute(const ggml_ext_execution_context* execution, ggml_tensor* const* nodes, int count, void* state, void*) {
        auto& plan     = *static_cast<Plan*>(state);
        const auto& op = plan.op;
        auto* dst      = nodes[count - 1];
        auto reg       = ggml_backend_dev_backend_reg(ggml_backend_get_device(execution->backend));
        auto query     = reinterpret_cast<ggml_ext_cuda_context_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_ext_cuda_context_v1"));
        ggml_ext_cuda_context context;
        if (!query || !query(execution->backend, &context) || context.device != plan.device.ordinal)
            return GGML_STATUS_FAILED;
        auto stream = static_cast<cudaStream_t>(context.stream);
        bool alias  = false;
        for (const auto* input : {op.input, op.up, op.modulation, op.weight, op.shift})
            alias |= overlaps(dst, input);
        auto* output         = static_cast<int8_t*>(alias ? ggml_backend_buffer_get_base(execution->workspace) : dst->data);
        const int64_t padded = (op.rows + 3) / 4 * 4;
        if (padded > op.rows && cudaMemsetAsync(output + op.k * op.rows, 0, size_t(op.k) * (padded - op.rows), stream) != cudaSuccess)
            return GGML_STATUS_FAILED;
        cudaError_t error;
        switch (op.kind) {
            case SCALE:
                error = launch<false, false, true>(plan, op, output, stream);
                break;
            case NORM:
                error = launch<false, true>(plan, op, output, stream);
                break;
            case RMS_MOD:
                error = launch<false, false, false, false, true>(plan, op, output, stream);
                break;
            case SWIGLU:
            case SILU:
                error = launch<true>(plan, op, output, stream);
                break;
            case SIGMOID:
                error = launch<true, false, false, true>(plan, op, output, stream);
                break;
            default:
                return GGML_STATUS_FAILED;
        }
        if (error != cudaSuccess || (alias && cudaMemcpyAsync(dst->data, output, size_t(op.k) * padded + op.rows * sizeof(float), cudaMemcpyDeviceToDevice, stream) != cudaSuccess))
            return GGML_STATUS_FAILED;
        return GGML_STATUS_SUCCESS;
    }
}

void sd_ggml_register_cuda_convrot(ggml_backend_dev_t device) {
    for (auto op : {GGML_OP_SCALE, GGML_OP_NORM, GGML_OP_RMS_NORM, GGML_OP_GLU, GGML_OP_UNARY}) {
        ggml_ext_kernel_desc desc = {};
        desc.abi_version          = GGML_EXT_ABI_VERSION;
        const std::string name    = std::string("sd.convrot_fusion.") + ggml_op_name(op) + ".cuda.v1";
        desc.name                 = name.c_str();
        desc.op                   = op;
        desc.priority             = 225;
        desc.optimize             = op == GGML_OP_SCALE ? sd_convrot::optimize : nullptr;
        desc.match                = sd_convrot::match;
        desc.prepare              = sd_convrot::prepare;
        desc.execute              = sd_convrot::execute;
        desc.destroy              = sd_convrot::destroy;
        GGML_ASSERT(ggml_ext_register_kernel(device, &desc));
    }
}
