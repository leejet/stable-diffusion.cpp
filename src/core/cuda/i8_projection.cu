#include "core/cuda/i8_projection.h"
#include "core/cuda/rope.h"

#include <cuda_fp16.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>
#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>
#include <mutex>
#include <type_traits>
#include <unordered_map>

static __device__ __forceinline__ float warp_sum8(float value) {
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1)
        value = __fadd_rn(value, __shfl_xor_sync(0xffffffff, value, offset, 8));
    return value;
}

template <class Map, class Out, bool rms = true>
struct I8RopeStore : cutlass::epilogue::threadblock::VisitorAuxStore<Map, float, cutlass::FloatRoundStyle::round_to_nearest, cute::Stride<int64_t, cute::_1, int64_t>> {
    using Base = cutlass::epilogue::threadblock::VisitorAuxStore<Map, float, cutlass::FloatRoundStyle::round_to_nearest, cute::Stride<int64_t, cute::_1, int64_t>>;
    struct Arguments : Base::Arguments {
        const float* weight;
        const float* pe;
        float eps;
        float scale;
        int64_t tokens;
        half* half_output;
        int64_t prefix_tokens;
    };
    using Params        = Arguments;
    using SharedStorage = typename Base::SharedStorage;
    const Params* extra;
    template <class Shape>
    static constexpr Params to_underlying_arguments(Shape const&, Arguments const& args, void*) { return args; }
    CUTLASS_HOST_DEVICE I8RopeStore(Params const& p, SharedStorage const& s)
        : Base(p, s), extra(&p) {}
    template <class CB>
    struct Callbacks : CB {
        const Params* extra;
        CUTLASS_DEVICE Callbacks(CB const& base, const Params* p)
            : CB(base), extra(p) {}
        CUTLASS_DEVICE float sum32(float4 v) {
            float a = warp_sum8(__fmul_rn(v.x, v.x));
            float b = warp_sum8(__fmul_rn(v.y, v.y));
            float c = warp_sum8(__fmul_rn(v.z, v.z));
            float d = warp_sum8(__fmul_rn(v.w, v.w));
            return __fadd_rn(__fadd_rn(a, c), __fadd_rn(b, d));
        }
        CUTLASS_DEVICE void end_step(int step) {
            static_assert(Map::Detail::kAccessWidth == 16);
            auto src    = cute::filter(this->tC_rAux);
            auto coords = cute::filter(this->tC_cAux(cute::_, cute::_, cute::_, step));
            static_assert(cute::size(src) == 4);
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < cute::size(src); i += 2) {
                float4 va = *reinterpret_cast<const float4*>(&src(i)), vb = *reinterpret_cast<const float4*>(&src(i + 1));
                float norm = 1.0f;
                if constexpr (rms) {
                    // Preserve the standalone RMSNorm reduction tree across four 32-element groups.
                    float sum = __fadd_rn(sum32(va), sum32(vb));
                    sum       = __fadd_rn(sum, __shfl_xor_sync(0xffffffff, sum, 8));
                    norm      = rsqrtf(sum / 128.f + extra->eps);
                }
                CUTLASS_PRAGMA_UNROLL
                for (int j = 0; j < 2; ++j) {
                    auto coord = coords(i + j);
                    int row = cute::get<0>(coord), col = cute::get<1>(coord);
                    int heads = cute::get<1>(this->problem_shape) / 128;
                    if (cute::elem_less(coord, this->problem_shape)) {
                        float4 value  = j ? vb : va;
                        int local     = col % 128;
                        int64_t token = row % extra->tokens, batch = row / extra->tokens, head = col / 128;
                        float4 r = value;
                        if constexpr (rms) {
                            float4 weight = ((const float4*)extra->weight)[local / 4];
                            value.x       = __fmul_rn(__fmul_rn(norm, value.x), weight.x);
                            value.y       = __fmul_rn(__fmul_rn(norm, value.y), weight.y);
                            value.z       = __fmul_rn(__fmul_rn(norm, value.z), weight.z);
                            value.w       = __fmul_rn(__fmul_rn(norm, value.w), weight.w);
                            float4 p0 = ((const float4*)extra->pe)[token * 64 + local / 2], p1 = ((const float4*)extra->pe)[token * 64 + local / 2 + 1];
                            r.x = __fadd_rn(__fmul_rn(value.x, p0.x), __fmul_rn(value.y, p0.y));
                            r.y = __fadd_rn(__fmul_rn(value.x, p0.z), __fmul_rn(value.y, p0.w));
                            r.z = __fadd_rn(__fmul_rn(value.z, p1.x), __fmul_rn(value.w, p1.y));
                            r.w = __fadd_rn(__fmul_rn(value.z, p1.z), __fmul_rn(value.w, p1.w));
                        }
                        int64_t offset = ((batch * heads + head) * (extra->tokens + extra->prefix_tokens) + token + extra->prefix_tokens) * 128 + local;
                        if constexpr (std::is_same<Out, half>::value) {
                            half2 a    = __floats2half2_rn(__fmul_rn(r.x, extra->scale), __fmul_rn(r.y, extra->scale));
                            half2 b    = __floats2half2_rn(__fmul_rn(r.z, extra->scale), __fmul_rn(r.w, extra->scale));
                            half2* dst = (half2*)(extra->half_output + offset);
                            dst[0]     = a;
                            dst[1]     = b;
                        } else {
                            *((float4*)(this->params_ptr->ptr_aux + offset)) = r;
                        }
                    }
                }
            }
        }
    };
    template <class Shape>
    CUTLASS_DEVICE auto get_callbacks(cutlass::gemm::GemmCoord tile, int tid, Shape shape) {
        auto cb = Base::get_callbacks(tile, tid, shape);
        return Callbacks<decltype(cb)>(cb, extra);
    }
};

template <class Out, bool RMS = true>
struct ProjectionKernel {
    using Tile    = cutlass::gemm::GemmShape<128, 256, 64>;
    using Warp    = cutlass::gemm::GemmShape<64, 64, 64>;
    using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
    using Map     = cutlass::epilogue::threadblock::OutputTileThreadLayout<Tile, Warp, float, 4, 1>;
    using Col     = cutlass::epilogue::threadblock::VisitorColBroadcast<Map, float, cute::Stride<cute::_1, cute::_0, int>>;
    using Row     = cutlass::epilogue::threadblock::VisitorRowBroadcast<Map, float, cute::Stride<cute::_0, cute::_1, int>>;
    using Mul     = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
    using ScaleX  = cutlass::epilogue::threadblock::Sm80EVT<Mul, cutlass::epilogue::threadblock::VisitorAccFetch, Col>;
    using ScaleW  = cutlass::epilogue::threadblock::Sm80EVT<Mul, ScaleX, Row>;
    using Store   = I8RopeStore<Map, Out, RMS>;
    using EVT     = cutlass::epilogue::threadblock::Sm80EVT<Store, ScaleW>;
    using Kernel  = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
        int8_t,
        cutlass::layout::RowMajor,
        cutlass::ComplexTransform::kNone,
        16,
        int8_t,
        cutlass::layout::ColumnMajor,
        cutlass::ComplexTransform::kNone,
        16,
        float,
        cutlass::layout::RowMajor,
        4,
        int,
        float,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        Tile,
        Warp,
        cutlass::gemm::GemmShape<16, 8, 32>,
        EVT,
        Swizzle,
        4,
        cutlass::arch::OpMultiplyAddSaturate>::GemmKernel;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

static float param_f32(const ggml_tensor* tensor, int index) {
    float result;
    std::memcpy(&result, tensor->op_params + index, sizeof(result));
    return result;
}

static bool device_supports(ggml_backend_dev_t device) {
#ifdef SD_CUDA_FORCE_CUBLAS
    (void)device;
    return false;
#else
    static std::mutex mutex;
    static std::unordered_map<ggml_backend_dev_t, bool> cache;
    std::lock_guard<std::mutex> lock(mutex);
    auto found = cache.find(device);
    if (found != cache.end())
        return found->second;
    auto reg             = ggml_backend_dev_backend_reg(device);
    auto query           = reinterpret_cast<ggml_ext_cuda_device_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_ext_cuda_device_v1"));
    int ordinal          = query ? query(device) : -1;
    cudaDeviceProp props = {};
    size_t shared        = std::max({sizeof(typename ProjectionKernel<float>::Kernel::SharedStorage),
                                     sizeof(typename ProjectionKernel<half>::Kernel::SharedStorage), sizeof(typename ProjectionKernel<half, false>::Kernel::SharedStorage)});
    bool supported       = ordinal >= 0 && cudaGetDeviceProperties(&props, ordinal) == cudaSuccess && props.major >= 8 && props.sharedMemPerBlockOptin >= shared;
    cache.emplace(device, supported);
    return supported;
#endif
}

static bool aligned(const ggml_tensor* tensor) {
    size_t offset = 0;
    while (tensor->view_src) {
        offset += tensor->view_offs;
        tensor = tensor->view_src;
    }
    return (reinterpret_cast<uintptr_t>(tensor->data) + offset) % 16 == 0;
}

static bool projection_supports(ggml_backend_dev_t device, const ggml_tensor* mat, const ggml_tensor* view) {
    const auto* w  = mat->src[0];
    const auto* x  = mat->src[1];
    const auto* ws = mat->src[2];
    if (mat->op != GGML_OP_MUL_MAT || mat->type != GGML_TYPE_F32 || mat->op_params[4] <= 0 || !w || !x || !ws || mat->src[3] ||
        w->type != GGML_TYPE_I8 || x->type != GGML_TYPE_I8 || ws->type != GGML_TYPE_F32 ||
        w->ne[2] != 1 || w->ne[3] != 1 || mat->ne[0] != w->ne[1] || ggml_nelements(ws) != mat->ne[0] ||
        !ggml_is_contiguous(mat) || !ggml_is_contiguous(w) || !ggml_is_contiguous(x) || !ggml_is_contiguous(ws) ||
        !aligned(w) || !aligned(x) || !aligned(ws) || !aligned(mat) ||
        view->op != GGML_OP_RESHAPE || view->src[0] != mat || view->view_offs != 0 || !ggml_is_contiguous(view) ||
        view->ne[0] != 128 || view->ne[0] * view->ne[1] != mat->ne[0] || ggml_nelements(view) != ggml_nelements(mat))
        return false;
    const int64_t rows = ggml_nrows(mat), n = mat->ne[0], k = w->ne[0];
    if (rows < 4096 || rows > INT_MAX || n < 4096 || n > INT_MAX || k < 4096 || k > INT_MAX || n % 256 || k % 64 || n / 256 > 65535 ||
        x->ne[0] != k || x->ne[1] != (rows + 3) / 4 * 4 + (rows * 4 + k - 1) / k || x->ne[2] != 1 || x->ne[3] != 1)
        return false;
    return device_supports(device);
}

bool sd_cuda_i8_rope_supports(ggml_backend_dev_t device, const ggml_tensor* mat, const ggml_tensor* view, const ggml_tensor* norm, const ggml_tensor* mul, const ggml_tensor* rope) {
    if (!projection_supports(device, mat, view) || norm->src[0] != view ||
        !sd_cuda::rope_fusion_supports(norm, mul, rope) || rope->src[1]->ne[2] != 64 || !aligned(mul->src[1]) || !aligned(rope->src[1]))
        return false;
    int32_t interleaved;
    std::memcpy(&interleaved, ggml_ext_op_attributes(rope), sizeof(interleaved));
    return interleaved != 0;
}

template <class Out, bool RMS = true>
static cudaError_t projection_launch(cudaStream_t stream, const ggml_tensor* mat, const ggml_tensor* view, const ggml_tensor* norm, const ggml_tensor* mul, const ggml_tensor* rope, void* output, float scale, int64_t prefix_tokens) {
    using P        = ProjectionKernel<Out, RMS>;
    using Gemm     = typename P::Gemm;
    const int rows = int(ggml_nrows(mat)), n = int(mat->ne[0]), k = int(mat->src[0]->ne[0]);
    const auto* x  = static_cast<const int8_t*>(mat->src[1]->data);
    const auto* w  = static_cast<const int8_t*>(mat->src[0]->data);
    const auto* xs = reinterpret_cast<const float*>(x + int64_t(k) * ((int64_t(rows) + 3) / 4 * 4));
    const auto* ws = static_cast<const float*>(mat->src[2]->data);
    const auto* nw = RMS ? static_cast<const float*>(mul->src[1]->data) : nullptr;
    const auto* pe = RMS ? static_cast<const float*>(rope->src[1]->data) : nullptr;
    // The visitor forms FP32 addresses even when its only writes target the FP16 output.
    float* address_space = std::is_same<Out, half>::value ? static_cast<float*>(mat->data) : static_cast<float*>(output);
    if ((reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(w) | reinterpret_cast<uintptr_t>(xs) |
         reinterpret_cast<uintptr_t>(ws) | reinterpret_cast<uintptr_t>(nw) | reinterpret_cast<uintptr_t>(pe) |
         reinterpret_cast<uintptr_t>(address_space)) %
            16 ||
        reinterpret_cast<uintptr_t>(output) % 8)
        return cudaErrorInvalidValue;
    typename P::EVT::Arguments evt = {{{{}, {xs, 0.f, {}}, {}}, {ws, 0.f, {}}, {}},
                                      {{address_space, {n, cute::_1{}, int64_t(rows) * n}}, nw, pe, RMS ? param_f32(norm, 0) : 0.f, scale, view->ne[2], static_cast<half*>(output), prefix_tokens}};
    typename Gemm::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm, {rows, n, k}, 1, evt, x, w, nullptr, nullptr,
                                  int64_t(rows) * k, int64_t(n) * k, 0, 0, k, k, 0, 0);
    Gemm gemm;
    if (gemm.can_implement(args) != cutlass::Status::kSuccess)
        return cudaErrorInvalidValue;
    if (gemm.initialize(args, nullptr, stream) != cutlass::Status::kSuccess || gemm(stream) != cutlass::Status::kSuccess)
        return cudaErrorUnknown;
    return cudaGetLastError();
}

cudaError_t sd_cuda_i8_rope_launch(cudaStream_t stream, ggml_tensor* const* nodes, void* output, float scale) {
    return projection_launch<half>(stream, nodes[0], nodes[1], nodes[2], nodes[3], nodes[4], output, scale, 0);
}

static const ggml_tensor* node_at(const ggml_cgraph* graph, int index) {
    return ggml_graph_node(const_cast<ggml_cgraph*>(graph), index);
}

static int concat_match(const ggml_cgraph* graph, int first, bool key) {
    const int offset = key ? 5 : 2, count = key ? 7 : 5;
    if (first + count > ggml_graph_n_nodes(const_cast<ggml_cgraph*>(graph)))
        return 0;
    const auto* concat = node_at(graph, first + offset);
    const auto* copy   = node_at(graph, first + count - 1);
    if (concat->op != GGML_OP_CONCAT || concat->op_params[0] != (key ? 1 : 2) || concat->src[1] != node_at(graph, first + offset - 1) ||
        copy->op != GGML_OP_CPY || copy->src[0] != node_at(graph, first + count - 2) || copy->type != GGML_TYPE_F16 ||
        !ggml_is_contiguous(copy) || !ggml_are_same_shape(copy, copy->src[0]))
        return 0;
    for (int i = 0; i < 2; ++i)
        if (!concat->src[i] || concat->src[i]->type != GGML_TYPE_F32 || !ggml_is_contiguous(concat->src[i]))
            return 0;
    const auto* view   = node_at(graph, first + 1);
    const auto* prefix = concat->src[0];
    if (prefix->ne[key ? 1 : 2] > INT_MAX - view->ne[2] || (ggml_nelements(prefix) + 255) / 256 > INT_MAX)
        return 0;
    if (!key) {
        const auto* permute = node_at(graph, first + 3);
        if (permute->op != GGML_OP_PERMUTE || permute->src[0] != concat || permute->op_params[0] != 0 || permute->op_params[1] != 2 ||
            permute->op_params[2] != 1 || permute->op_params[3] != 3)
            return 0;
    }
    return count;
}

static int projection_match(ggml_backend_dev_t device, const ggml_cgraph* graph, int first, void*) {
    static const bool disabled = [] {const char* value=std::getenv("GGML_CUDA_DISABLE_FUSION");return value&&std::atoi(value)!=0; }();
    if (disabled || first + 4 >= ggml_graph_n_nodes(const_cast<ggml_cgraph*>(graph)))
        return 0;
    const auto* mat  = node_at(graph, first);
    const auto* view = node_at(graph, first + 1);
    const auto* norm = node_at(graph, first + 2);
    const bool key   = norm->op == GGML_OP_RMS_NORM;
    if (key) {
        if (!sd_cuda_i8_rope_supports(device, mat, view, norm, node_at(graph, first + 3), node_at(graph, first + 4)))
            return 0;
        if (int count = concat_match(graph, first, true))
            return count;
        if (first + 6 < ggml_graph_n_nodes(const_cast<ggml_cgraph*>(graph))) {
            const auto* rope  = node_at(graph, first + 4);
            const auto* scale = node_at(graph, first + 5);
            const auto* copy  = node_at(graph, first + 6);
            if (scale->op == GGML_OP_SCALE && scale->src[0] == rope && param_f32(scale, 1) == 0.f && copy->op == GGML_OP_CPY && copy->src[0] == scale &&
                copy->type == GGML_TYPE_F16 && ggml_are_same_shape(rope, scale) && ggml_are_same_shape(rope, copy) && ggml_is_contiguous(copy))
                return 7;
        }
        return 5;
    }
    return projection_supports(device, mat, view) ? concat_match(graph, first, false) : 0;
}

template <bool Aligned>
static __global__ void prefix_copy(const float* input, half2* output, int64_t count4, int64_t prefix, int64_t tokens, int64_t heads, bool permuted) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count4)
        return;
    const int64_t col = i % 32, row = i / 32;
    const int64_t token = row % prefix, head = (row / prefix) % heads, batch = row / (prefix * heads);
    const int64_t source_row = permuted ? (batch * prefix + token) * heads + head : (batch * heads + head) * prefix + token;
    const float* source      = input + source_row * 128 + col * 4;
    float4 value;
    if constexpr (Aligned)
        value = *reinterpret_cast<const float4*>(source);
    else
        value = make_float4(source[0], source[1], source[2], source[3]);
    const int64_t target   = ((batch * heads + head) * (prefix + tokens) + token) * 32 + col;
    output[2 * target]     = __floats2half2_rn(value.x, value.y);
    output[2 * target + 1] = __floats2half2_rn(value.z, value.w);
}

static ggml_status projection_prepare(ggml_backend_t, ggml_tensor* const* nodes, int count, void**, size_t* workspace, void*) {
    *workspace = ggml_nbytes(nodes[count - 1]);
    return GGML_STATUS_SUCCESS;
}
static void projection_destroy(void*, void*) {}

static bool overlaps(const ggml_tensor* a, const ggml_tensor* b) {
    const auto x = reinterpret_cast<uintptr_t>(a->data), y = reinterpret_cast<uintptr_t>(b->data);
    return x <= y ? y - x < ggml_nbytes(a) : x - y < ggml_nbytes(b);
}

static ggml_status projection_execute(const ggml_ext_execution_context* execution, ggml_tensor* const* nodes, int count, void*, void*) {
    const auto* mat    = nodes[0];
    const auto* view   = nodes[1];
    const bool key     = nodes[2]->op == GGML_OP_RMS_NORM;
    const auto* concat = nodes[key ? count == 7 ? 5 : 4 : 2];
    const bool joined  = concat->op == GGML_OP_CONCAT;
    auto* dst          = nodes[count - 1];
    auto reg           = ggml_backend_dev_backend_reg(ggml_backend_get_device(execution->backend));
    auto query         = reinterpret_cast<ggml_ext_cuda_context_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_ext_cuda_context_v1"));
    ggml_ext_cuda_context context;
    if (!query || !query(execution->backend, &context))
        return GGML_STATUS_FAILED;
    auto stream = static_cast<cudaStream_t>(context.stream);
    bool alias  = reinterpret_cast<uintptr_t>(dst->data) % (dst->type == GGML_TYPE_F32 ? 16 : 8) != 0;
    for (const auto* input : {mat->src[0], mat->src[1], mat->src[2], key ? nodes[3]->src[1] : nullptr, key ? nodes[4]->src[1] : nullptr, joined ? concat->src[0] : nullptr})
        if (input)
            alias |= overlaps(dst, input);
    void* output   = alias ? ggml_backend_buffer_get_base(execution->workspace) : dst->data;
    int64_t prefix = joined ? concat->src[0]->ne[key ? 1 : 2] : 0;
    float scale    = key && count == 7 && !joined ? param_f32(nodes[5], 0) : 1.f;
    cudaError_t error;
    if (dst->type == GGML_TYPE_F32)
        error = projection_launch<float>(stream, mat, view, nodes[2], nodes[3], nodes[4], output, scale, 0);
    else if (key)
        error = projection_launch<half>(stream, mat, view, nodes[2], nodes[3], nodes[4], output, scale, prefix);
    else
        error = projection_launch<half, false>(stream, mat, view, nullptr, nullptr, nullptr, output, scale, prefix);
    if (error == cudaSuccess && joined && prefix > 0) {
        const auto* input    = concat->src[0];
        const int64_t count4 = ggml_nelements(input) / 4;
        auto kernel          = reinterpret_cast<uintptr_t>(input->data) % 16 == 0 ? prefix_copy<true> : prefix_copy<false>;
        kernel<<<static_cast<unsigned int>((count4 + 255) / 256), 256, 0, stream>>>(static_cast<const float*>(input->data), static_cast<half2*>(output), count4, prefix, view->ne[2], view->ne[1], !key);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess && alias)
        error = cudaMemcpyAsync(dst->data, output, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, stream);
    return error == cudaSuccess ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
}

void sd_ggml_register_cuda_i8_projection(ggml_backend_dev_t device) {
    ggml_ext_kernel_desc desc = {};
    desc.abi_version          = GGML_EXT_ABI_VERSION;
    desc.name                 = "sd.i8_projection.cuda.v1";
    desc.op                   = GGML_OP_MUL_MAT;
    desc.priority             = 350;
    desc.match                = projection_match;
    desc.prepare              = projection_prepare;
    desc.execute              = projection_execute;
    desc.destroy              = projection_destroy;
    GGML_ASSERT(ggml_ext_register_kernel_ex(device, &desc, GGML_EXT_KERNEL_CAPTURE_SAFE | GGML_EXT_KERNEL_REPLAY_REUSABLE));
}
