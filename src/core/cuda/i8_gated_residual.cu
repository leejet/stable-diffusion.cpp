#include <cuda_runtime.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle_streamk.h>
#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>
#include <mutex>
#include <unordered_map>
#include "core/ggml_extensions.h"

namespace sd_i8_gated {
    struct BandedStreamK : cutlass::gemm::threadblock::ThreadblockSwizzleStreamK {
        using Base = cutlass::gemm::threadblock::ThreadblockSwizzleStreamK;
        using Base::Base;

        CUTLASS_DEVICE cutlass::gemm::GemmCoord get_tile_offset(int index) const {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 890
            const auto shape = tiled_shape();
            if (problem_size.m() >= 4096 && problem_size.n() >= 16384 && shape.m() >= shape.n()) {
                // Column bands retain weight tiles in Ada's L2 across successive row tiles.
                constexpr int band_width = 16;
                if (index >= shape.m() * shape.n()) {
                    return {shape.m(), shape.n(), get_batch_idx()};
                }
                const int band   = index / (shape.m() * band_width);
                const int width  = min(band_width, shape.n() - band * band_width);
                const int within = index - band * shape.m() * band_width;
                return {within / width, band * band_width + within % width, get_batch_idx()};
            }
#endif
            return Base::get_tile_offset(index);
        }
    };

    template <int M, int N, int WM>
    struct TileConfig {
        using Tile = cutlass::gemm::GemmShape<M, N, 64>;
        using Warp = cutlass::gemm::GemmShape<WM, 64, 64>;
    };

    enum Tile {
        TILE_128_256,
        TILE_128_128,
        TILE_64_128,
        TILE_16_128,
    };

    static bool small_rows_enabled(int64_t rows, int64_t n, int64_t k) {
        static const char* setting = std::getenv("GGML_CUDA_I8_SMALL_ROWS");
        return setting ? std::atoi(setting) != 0 : rows < 32 && n >= 4096 && k >= 4096;
    }

    static Tile select_tile(int64_t rows, int64_t n, int nsm) {
        if (rows < 32) {
            return TILE_16_128;
        }
        const int64_t columns = (n + 127) / 128;
        if (((rows + 15) / 16) * columns <= nsm) {
            return TILE_16_128;
        }
        if (((rows + 63) / 64) * columns <= nsm) {
            return TILE_64_128;
        }
        const int64_t wide_tiles   = ((rows + 127) / 128) * ((n + 255) / 256);
        const int64_t square_tiles = ((rows + 127) / 128) * columns;
        // A narrower tile costs more per output element, but can waste less of the last wave.
        const double wide_cost   = double((wide_tiles + nsm - 1) / nsm) / wide_tiles;
        const double square_cost = 1.08 * double((square_tiles + nsm - 1) / nsm) / square_tiles;
        return wide_cost <= square_cost ? TILE_128_256 : TILE_128_128;
    }

    template <class T>
    struct MulRound {
        CUTLASS_DEVICE T operator()(const T& a, const T& b) const {
            T result;
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < T::kElements; ++i) {
                result[i] = __fmul_rn(a[i], b[i]);
            }
            return result;
        }
    };

    template <class T>
    struct AddRound {
        CUTLASS_DEVICE T operator()(const T& a, const T& b) const {
            T result;
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < T::kElements; ++i) {
                result[i] = __fadd_rn(a[i], b[i]);
            }
            return result;
        }
    };

    template <class Config, bool scaled, bool stream_k>
    struct KernelTypes {
        using Tile         = typename Config::Tile;
        using Warp         = typename Config::Warp;
        using Swizzle      = std::conditional_t<stream_k, BandedStreamK, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>>;
        using Map          = cutlass::epilogue::threadblock::OutputTileThreadLayout<Tile, Warp, float, 4, 1>;
        using Col          = cutlass::epilogue::threadblock::VisitorColBroadcast<Map, float, cute::Stride<cute::_1, cute::_0, int>>;
        using Row          = cutlass::epilogue::threadblock::VisitorRowBroadcast<Map, float, cute::Stride<cute::_0, cute::_1, int>>;
        using Mul          = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
        using ScaleX       = cutlass::epilogue::threadblock::Sm80EVT<Mul, cutlass::epilogue::threadblock::VisitorAccFetch, Col>;
        using ScaleW       = cutlass::epilogue::threadblock::Sm80EVT<Mul, ScaleX, Row>;
        using Residual     = cutlass::epilogue::threadblock::VisitorAuxLoad<Map, float, cute::Stride<int64_t, cute::_1, int64_t>>;
        using GateMul      = cutlass::epilogue::threadblock::VisitorCompute<MulRound, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
        using Add          = cutlass::epilogue::threadblock::VisitorCompute<AddRound, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
        using Scalar       = cutlass::epilogue::threadblock::VisitorScalarBroadcast<float>;
        using ScaleOut     = cutlass::epilogue::threadblock::Sm80EVT<GateMul, ScaleW, Scalar>;
        using Scaled       = std::conditional_t<scaled, ScaleOut, ScaleW>;
        using Gated        = cutlass::epilogue::threadblock::Sm80EVT<GateMul, Scaled, Row>;
        using WithResidual = cutlass::epilogue::threadblock::Sm80EVT<Add, Gated, Residual>;
        using Output       = WithResidual;
        using Store        = cutlass::epilogue::threadblock::VisitorAuxStore<Map, float, cutlass::FloatRoundStyle::round_to_nearest, cute::Stride<int64_t, cute::_1, int64_t>>;
        using Epilogue     = cutlass::epilogue::threadblock::Sm80EVT<Store, Output>;
        using Kernel       = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
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
            Epilogue,
            Swizzle,
            4,
            cutlass::arch::OpMultiplyAddSaturate>::GemmKernel;
        using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
    };

    struct DeviceInfo {
        int ordinal        = -1;
        int major          = 0;
        int sms            = 0;
        int threads_per_sm = 0;
        size_t shared      = 0;
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
            info = {ordinal, props.major, props.multiProcessorCount, props.maxThreadsPerMultiProcessor, size_t(props.sharedMemPerBlockOptin)};
        cache.emplace(device, info);
        return info;
    }
    static float param_f32(const ggml_tensor* tensor, int index) {
        float value;
        std::memcpy(&value, tensor->op_params + index, sizeof(value));
        return value;
    }
    static bool aligned(const ggml_tensor* tensor) {
        if (!tensor)
            return false;
        size_t offset = 0;
        while (tensor->view_src) {
            offset += tensor->view_offs;
            tensor = tensor->view_src;
        }
        return (reinterpret_cast<uintptr_t>(tensor->data) + offset) % 16 == 0;
    }
    struct Operation {
        SDI8Operation mat;
        const ggml_tensor* gate;
        const ggml_tensor* residual;
        const ggml_tensor* prefix;
        float scale;
        int64_t rows, n, k;
    };
    static Operation describe(ggml_tensor* const* nodes, int count) {
        Operation op = {};
        sd_ggml_i8_describe(nodes[0], &op.mat);
        const bool concat = nodes[count - 1]->op == GGML_OP_CONCAT;
        const auto* add   = nodes[count - (concat ? 2 : 1)];
        const auto* mul   = nodes[count - (concat ? 3 : 2)];
        op.gate           = mul->src[1];
        op.residual       = add->src[0] == mul ? add->src[1] : add->src[0];
        op.prefix         = concat ? nodes[count - 1]->src[0] : nullptr;
        op.scale          = nodes[1]->op == GGML_OP_SCALE ? param_f32(nodes[1], 0) : 1.f;
        op.rows           = ggml_nrows(nodes[0]);
        op.n              = nodes[0]->ne[0];
        op.k              = op.mat.weight->ne[0];
        return op;
    }
    template <class Types>
    static typename Types::Gemm::Arguments arguments(const Operation& op, float* output, bool metadata) {
        const auto* x                            = metadata ? nullptr : static_cast<const int8_t*>(op.mat.input->data);
        const auto* w                            = metadata ? nullptr : static_cast<const int8_t*>(op.mat.weight->data);
        const auto* xs                           = metadata ? nullptr : op.mat.packed ? reinterpret_cast<const float*>(x + op.k * ((op.rows + 3) / 4 * 4))
                                                                                      : static_cast<const float*>(op.mat.input_scale->data);
        const auto* ws                           = metadata ? nullptr : static_cast<const float*>(op.mat.weight_scale->data);
        const auto* gate                         = metadata ? nullptr : static_cast<const float*>(op.gate->data);
        auto* residual                           = metadata ? nullptr : static_cast<float*>(op.residual->data);
        typename Types::ScaleW::Arguments scales = {{{}, {xs, 0.f, {}}, {}}, {ws, 0.f, {}}, {}};
        typename Types::Scaled::Arguments scaled;
        if constexpr (std::is_same<typename Types::Scaled, typename Types::ScaleW>::value)
            scaled = scales;
        else
            scaled = {scales, {{op.scale}}, {}};
        typename Types::Epilogue::Arguments epilogue = {{{scaled, {gate, 0.f, {}}, {}}, {residual, 0.f, {op.n, cute::_1{}, op.rows * op.n}}, {}}, {output, {op.n, cute::_1{}, op.rows * op.n}}};
        return {cutlass::gemm::GemmUniversalMode::kGemm, {int(op.rows), int(op.n), int(op.k)}, 1, epilogue, x, w, nullptr, nullptr, op.rows * op.k, op.n * op.k, 0, 0, op.k, op.k, 0, 0};
    }
    struct Plan {
        DeviceInfo device;
        size_t kernel_workspace                                                     = 0;
        size_t output_offset                                                        = 0;
        int occupancy                                                               = 0;
        ggml_status (*launch)(Plan&, const Operation&, float*, void*, cudaStream_t) = nullptr;
    };
    template <class Types>
    static ggml_status launch(Plan& plan, const Operation& op, float* output, void* scratch, cudaStream_t stream) {
        using Kernel = typename Types::Kernel;
        auto args    = arguments<Types>(op, output, false);
        if (Kernel::can_implement(args) != cutlass::Status::kSuccess)
            return GGML_STATUS_FAILED;
        if (plan.occupancy == 0) {
            if (sizeof(typename Kernel::SharedStorage) >= (48 << 10) &&
                cudaFuncSetAttribute(cutlass::Kernel2<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(typename Kernel::SharedStorage)) != cudaSuccess)
                return GGML_STATUS_FAILED;
            int occupancy = 0;
            if (cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&occupancy, cutlass::Kernel2<Kernel>,
                                                                       Kernel::kThreadCount, sizeof(typename Kernel::SharedStorage), cudaOccupancyDisableCachingOverride) != cudaSuccess ||
                occupancy <= 0)
                return GGML_STATUS_FAILED;
            plan.occupancy = occupancy;
        }
        typename Kernel::Params params(args, plan.device.sms, plan.occupancy);
        if (params.get_workspace_size() > plan.kernel_workspace || params.init_workspace(scratch, stream) != cutlass::Status::kSuccess)
            return GGML_STATUS_FAILED;
        cutlass::arch::synclog_setup();
        cutlass::Kernel2<Kernel><<<params.get_grid_dims(), Kernel::kThreadCount, sizeof(typename Kernel::SharedStorage), stream>>>(params);
        return cudaGetLastError() == cudaSuccess ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
    }
    template <class Config, bool Scaled, bool StreamK>
    static bool configure(const Operation& op, Plan& plan) {
        using Types  = KernelTypes<Config, Scaled, StreamK>;
        using Kernel = typename Types::Kernel;
        if (sizeof(typename Kernel::SharedStorage) > plan.device.shared)
            return false;
        auto args = arguments<Types>(op, nullptr, true);
        if (Kernel::can_implement(args) != cutlass::Status::kSuccess)
            return false;
        size_t workspace = 0;
        // Bound every possible occupancy without setting a CUDA device or initializing a kernel during graph measurement.
        for (int occupancy = 1; occupancy <= plan.device.threads_per_sm / Kernel::kThreadCount; ++occupancy) {
            typename Kernel::Params params(args, plan.device.sms, occupancy);
            const auto grid = params.get_grid_dims();
            if (grid.y > 65535 || grid.z > 65535)
                return false;
            workspace = std::max(workspace, params.get_workspace_size());
        }
        plan.kernel_workspace = workspace;
        plan.output_offset    = (workspace + 255) / 256 * 256;
        plan.launch           = launch<Types>;
        return true;
    }
    template <class Config>
    static bool configure(const Operation& op, Plan& plan) {
        if (op.scale != 1.f)
            return op.rows >= 8192 ? configure<Config, true, true>(op, plan) : configure<Config, true, false>(op, plan);
        return op.rows >= 8192 ? configure<Config, false, true>(op, plan) : configure<Config, false, false>(op, plan);
    }
    static bool configure(const Operation& op, Plan& plan) {
        auto try_tile = [&](Tile tile) {
            switch (tile) {
                case TILE_128_256:
                    return configure<TileConfig<128, 256, 64>>(op, plan);
                case TILE_128_128:
                    return configure<TileConfig<128, 128, 64>>(op, plan);
                case TILE_64_128:
                    return configure<TileConfig<64, 128, 32>>(op, plan);
                case TILE_16_128:
                    return configure<TileConfig<16, 128, 16>>(op, plan);
            }
            return false;
        };
        const auto tile = select_tile(op.rows, op.n, plan.device.sms);
        if (try_tile(tile))
            return true;
        for (auto fallback : {TILE_128_256, TILE_128_128, TILE_64_128, TILE_16_128})
            if (fallback != tile && try_tile(fallback))
                return true;
        return false;
    }
    static bool fusion_disabled() {
        static const bool disabled = [] { const char* v = std::getenv("GGML_CUDA_DISABLE_FUSION"); return v && std::atoi(v) != 0; }();
        return disabled;
    }
    static void optimize(ggml_backend_dev_t, ggml_cgraph* graph, void*) {
#ifndef SD_CUDA_FORCE_CUBLAS
        if (fusion_disabled())
            return;
        const int total = ggml_graph_n_nodes(graph);
        for (int i = 2; i + 1 < total; ++i) {
            const auto* mul = ggml_graph_node(graph, i);
            const auto* add = ggml_graph_node(graph, i + 1);
            if (mul->op != GGML_OP_MUL || add->op != GGML_OP_ADD || (add->src[0] != mul && add->src[1] != mul))
                continue;
            const auto* producer = mul->src[0];
            const bool scaled    = producer->op == GGML_OP_SCALE;
            const auto* mat      = scaled ? producer->src[0] : producer;
            SDI8Operation op;
            if (!sd_ggml_i8_describe(mat, &op) || op.input->type != GGML_TYPE_I8 || !op.weight_scale || op.bias)
                continue;
            int scale_index = -1;
            for (int j = i - 1; j >= 0 && j >= i - 64; --j) {
                if (scaled && ggml_graph_node(graph, j) == producer)
                    scale_index = j;
                if (ggml_graph_node(graph, j) != mat)
                    continue;
                if (scaled) {
                    if (scale_index < 0 || (j + 1 < scale_index && !ggml_ext_graph_delay(graph, j, 1, scale_index)))
                        break;
                    j = scale_index - 1;
                }
                ggml_ext_graph_delay(graph, j, scaled ? 2 : 1, i);
                break;
            }
        }
#endif
    }
    static int match(ggml_backend_dev_t device, const ggml_cgraph* graph, int first, void*) {
#ifdef SD_CUDA_FORCE_CUBLAS
        return 0;
#else
        const int available = ggml_graph_n_nodes(const_cast<ggml_cgraph*>(graph)) - first;
        if (fusion_disabled() || available < 3)
            return 0;
        ggml_tensor* nodes[5] = {};
        for (int i = 0; i < std::min(available, 5); ++i)
            nodes[i] = ggml_graph_node(const_cast<ggml_cgraph*>(graph), first + i);
        SDI8Operation mat;
        if (!sd_ggml_i8_describe(nodes[0], &mat) || mat.input->type != GGML_TYPE_I8 || !mat.weight_scale || mat.bias)
            return 0;
        const int64_t rows = ggml_nrows(nodes[0]), n = nodes[0]->ne[0], k = mat.weight->ne[0];
        if (rows < (small_rows_enabled(rows, n, k) ? 1 : 128) || rows > INT_MAX || n > INT_MAX || k > INT_MAX || n % 128 || k % 64 ||
            ((rows + 15) / 16) > INT_MAX / ((n + 127) / 128))
            return 0;
        for (const auto* t : {mat.weight, mat.input, mat.weight_scale, mat.packed ? mat.weight_scale : mat.input_scale})
            if (!aligned(t))
                return 0;
        const bool scaled = nodes[1]->op == GGML_OP_SCALE;
        int count         = scaled ? 4 : 3;
        if (available < count || (scaled && (nodes[1]->src[0] != nodes[0] || param_f32(nodes[1], 1) != 0.f)))
            return 0;
        const auto* mul = nodes[count - 2];
        const auto* add = nodes[count - 1];
        if (mul->op != GGML_OP_MUL || add->op != GGML_OP_ADD || mul->src[0] != nodes[count - 3] ||
            (add->src[0] != mul && add->src[1] != mul))
            return 0;
        const auto* gate     = mul->src[1];
        const auto* residual = add->src[0] == mul ? add->src[1] : add->src[0];
        if (!gate || !residual || gate->ne[0] != n || ggml_nelements(gate) != n || !aligned(gate) || !aligned(residual))
            return 0;
        for (const auto* t : {gate, residual, mul, add})
            if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) || (t != gate && !ggml_are_same_shape(nodes[0], t)))
                return 0;
        if (available > count) {
            const auto* concat = nodes[count];
            if (concat->op == GGML_OP_CONCAT && concat->src[1] == add && concat->op_params[0] == 1 &&
                concat->type == GGML_TYPE_F32 && concat->ne[2] == 1 && concat->ne[3] == 1 && ggml_is_contiguous(concat) &&
                concat->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(concat->src[0]))
                ++count;
        }
        const auto* prefix = nodes[count - 1]->op == GGML_OP_CONCAT ? nodes[count - 1]->src[0] : nullptr;
        for (int i = 0; i < count; ++i)
            if (nodes[i] == gate || nodes[i] == residual || nodes[i] == prefix)
                return 0;
        Plan plan;
        plan.device = device_info(device);
        if (plan.device.major < 8 || plan.device.sms <= 0)
            return 0;
        return configure(describe(nodes, count), plan) ? count : 0;
#endif
    }
    static ggml_status prepare(ggml_backend_t backend, ggml_tensor* const* nodes, int count, void** state, size_t* workspace, void*) {
        auto* plan   = new Plan;
        *state       = plan;
        plan->device = device_info(ggml_backend_get_device(backend));
        if (!configure(describe(nodes, count), *plan))
            return GGML_STATUS_FAILED;
        *workspace = plan->output_offset + ggml_nbytes(nodes[count - 1]);
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
    static ggml_status execute(const ggml_ext_execution_context* execution, ggml_tensor* const* nodes, int count, void* state, void*) {
        auto& plan    = *static_cast<Plan*>(state);
        const auto op = describe(nodes, count);
        auto* dst     = nodes[count - 1];
        auto reg      = ggml_backend_dev_backend_reg(ggml_backend_get_device(execution->backend));
        auto query    = reinterpret_cast<ggml_ext_cuda_context_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_ext_cuda_context_v1"));
        ggml_ext_cuda_context context;
        if (!query || !query(execution->backend, &context) || context.device != plan.device.ordinal)
            return GGML_STATUS_FAILED;
        auto stream   = static_cast<cudaStream_t>(context.stream);
        auto* scratch = static_cast<char*>(ggml_backend_buffer_get_base(execution->workspace));
        bool alias    = reinterpret_cast<uintptr_t>(dst->data) % 16 != 0;
        for (const auto* input : {op.mat.weight, op.mat.input, op.mat.input_scale, op.mat.weight_scale, op.gate, op.residual, op.prefix}) {
            if (input == op.residual && !op.prefix && input->data == dst->data)
                continue;
            alias |= overlaps(dst, input);
        }
        auto* output              = alias ? scratch + plan.output_offset : static_cast<char*>(dst->data);
        const size_t prefix_bytes = op.prefix ? ggml_nbytes(op.prefix) : 0;
        auto status               = plan.launch(plan, op, reinterpret_cast<float*>(output + prefix_bytes), scratch, stream);
        if (status != GGML_STATUS_SUCCESS)
            return status;
        if (prefix_bytes && cudaMemcpyAsync(output, op.prefix->data, prefix_bytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess)
            return GGML_STATUS_FAILED;
        if (alias && cudaMemcpyAsync(dst->data, output, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, stream) != cudaSuccess)
            return GGML_STATUS_FAILED;
        return GGML_STATUS_SUCCESS;
    }
}

void sd_ggml_register_cuda_i8_gated_residual(ggml_backend_dev_t device) {
    ggml_ext_kernel_desc desc = {};
    desc.abi_version          = GGML_EXT_ABI_VERSION;
    desc.name                 = "sd.i8_gated_residual.cuda.v1";
    desc.op                   = GGML_OP_MUL_MAT;
    desc.priority             = 325;
    desc.optimize             = sd_i8_gated::optimize;
    desc.match                = sd_i8_gated::match;
    desc.prepare              = sd_i8_gated::prepare;
    desc.execute              = sd_i8_gated::execute;
    desc.destroy              = sd_i8_gated::destroy;
    GGML_ASSERT(ggml_ext_register_kernel_ex(device, &desc, GGML_EXT_KERNEL_CAPTURE_SAFE | GGML_EXT_KERNEL_REPLAY_REUSABLE));
    desc.name      = "sd.custom_i8_gated_residual.cuda.v1";
    desc.op        = GGML_OP_CUSTOM;
    desc.custom_op = sd_ggml_i8_op();
    desc.optimize  = nullptr;
    GGML_ASSERT(ggml_ext_register_kernel_ex(device, &desc, GGML_EXT_KERNEL_CAPTURE_SAFE | GGML_EXT_KERNEL_REPLAY_REUSABLE));
}
