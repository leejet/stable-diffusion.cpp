#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle_streamk.h>
#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>
#include "backend.h"
#include "i8_gemm.cuh"
#include "i8_gemm.h"

namespace sd_i8_cutlass {
    // The iterator pairs gate/up columns while reading the original checkpoint layout.
    struct WeightLayout : cutlass::layout::PermuteBase {
        int stride, half_n;
        CUTLASS_HOST_DEVICE WeightLayout(cutlass::PitchLinearCoord extent, int64_t ld)
            : stride(static_cast<int>(ld)), half_n(extent.strided() / 2) {}
        CUTLASS_HOST_DEVICE WeightLayout(cutlass::MatrixCoord extent, int64_t ld)
            : stride(static_cast<int>(ld)), half_n(extent.column() / 2) {}
        CUTLASS_HOST_DEVICE int64_t operator()(cutlass::PitchLinearCoord c) const { return c.contiguous() + int64_t(c.strided() / 2 + (c.strided() % 2) * half_n) * stride; }
        CUTLASS_HOST_DEVICE int64_t operator()(cutlass::MatrixCoord c) const { return c.row() + int64_t(c.column() / 2 + (c.column() % 2) * half_n) * stride; }
    };

    template <class Map>
    struct SwigluStore : cutlass::epilogue::threadblock::VisitorAuxStore<Map, float, cutlass::FloatRoundStyle::round_to_nearest, cute::Stride<int64_t, cute::_1, int64_t>> {
        using Base = cutlass::epilogue::threadblock::VisitorAuxStore<Map, float, cutlass::FloatRoundStyle::round_to_nearest, cute::Stride<int64_t, cute::_1, int64_t>>;
        using Base::Base;
        template <class CB>
        struct Callbacks : CB {
            CUTLASS_DEVICE Callbacks(CB const& base)
                : CB(base) {}
            CUTLASS_DEVICE void end_step(int step) {
                auto src    = cute::filter(this->tC_rAux);
                auto coords = cute::filter(this->tC_cAux(cute::_, cute::_, cute::_, step));
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < cute::size(src); ++i) {
                    auto coord = coords(i);
                    if (cute::elem_less(coord, this->problem_shape)) {
                        // Each vector contains two gate/up pairs; only their SwiGLU results are stored.
                        const float4 a = *reinterpret_cast<const float4*>(&src(i));
                        float2 value   = make_float2(__fmul_rn(a.x / (1.f + expf(-a.x)), a.y), __fmul_rn(a.z / (1.f + expf(-a.z)), a.w));
                        int64_t n = cute::get<1>(this->problem_shape), m = cute::get<0>(this->problem_shape);
                        int64_t offset                                                 = cute::get<0>(coord) * (n / 2) + cute::get<1>(coord) / 2 + cute::get<2>(coord) * m * (n / 2);
                        *reinterpret_cast<float2*>(this->params_ptr->ptr_aux + offset) = value;
                    }
                }
            }
        };
        template <class Shape>
        CUTLASS_DEVICE auto get_callbacks(cutlass::gemm::GemmCoord tile, int tid, Shape shape) {
            auto cb = Base::get_callbacks(tile, tid, shape);
            return Callbacks<decltype(cb)>(cb);
        }
    };

    static __global__ void pair_scales(const float* src, float* dst, int n) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            dst[i] = src[i / 2 + (i % 2) * (n / 2)];
    }

    template <class Config, bool StreamK>
    struct DenseTypes {
        using Tile     = typename Config::Tile;
        using Warp     = typename Config::Warp;
        using Swizzle  = std::conditional_t<StreamK, BandedStreamK, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>>;
        using Map      = cutlass::epilogue::threadblock::OutputTileThreadLayout<Tile, Warp, float, 4, 1>;
        using Col      = cutlass::epilogue::threadblock::VisitorColBroadcast<Map, float, cute::Stride<cute::_1, cute::_0, int>>;
        using Row      = cutlass::epilogue::threadblock::VisitorRowBroadcast<Map, float, cute::Stride<cute::_0, cute::_1, int>>;
        using Mul      = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
        using ScaleX   = cutlass::epilogue::threadblock::Sm80EVT<Mul, cutlass::epilogue::threadblock::VisitorAccFetch, Col>;
        using ScaleW   = cutlass::epilogue::threadblock::Sm80EVT<Mul, ScaleX, Row>;
        using Store    = cutlass::epilogue::threadblock::VisitorAuxStore<Map, float, cutlass::FloatRoundStyle::round_to_nearest, cute::Stride<int64_t, cute::_1, int64_t>>;
        using Epilogue = cutlass::epilogue::threadblock::Sm80EVT<Store, ScaleW>;
        using Kernel   = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
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
    template <class Config, bool StreamK>
    struct SwigluTypes {
        using Tile            = typename Config::Tile;
        using Warp            = typename Config::Warp;
        using Swizzle         = std::conditional_t<StreamK, BandedStreamK, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>>;
        using Map             = cutlass::epilogue::threadblock::OutputTileThreadLayout<Tile, Warp, float, 4, 1>;
        using Col             = cutlass::epilogue::threadblock::VisitorColBroadcast<Map, float, cute::Stride<cute::_1, cute::_0, int>>;
        using Row             = cutlass::epilogue::threadblock::VisitorRowBroadcast<Map, float, cute::Stride<cute::_0, cute::_1, int>>;
        using Mul             = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
        using ScaleX          = cutlass::epilogue::threadblock::Sm80EVT<Mul, cutlass::epilogue::threadblock::VisitorAccFetch, Col>;
        using ScaleW          = cutlass::epilogue::threadblock::Sm80EVT<Mul, ScaleX, Row>;
        using Store           = SwigluStore<Map>;
        using Epilogue        = cutlass::epilogue::threadblock::Sm80EVT<Store, ScaleW>;
        using GemmBase        = typename cutlass::gemm::kernel::DefaultGemmUniversal<int8_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 16, int8_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 16, float, cutlass::layout::RowMajor, int, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, Tile, Warp, cutlass::gemm::GemmShape<16, 8, 32>, cutlass::epilogue::thread::LinearCombination<float, 4, int, float>, Swizzle, 4, cutlass::arch::OpMultiplyAddSaturate, cutlass::gemm::SharedMemoryClearOption::kNone, false, false, false, cutlass::layout::NoPermute, cutlass::layout::NoPermute, WeightLayout>::GemmKernel;
        using VisitorEpilogue = cutlass::epilogue::threadblock::EpilogueWithVisitorCallbacks<typename GemmBase::Epilogue, Epilogue, 1>;
        using Kernel          = std::conditional_t<StreamK,
                                          cutlass::gemm::kernel::GemmWithEpilogueVisitorStreamk<typename GemmBase::Mma, VisitorEpilogue, Swizzle>,
                                          cutlass::gemm::kernel::GemmWithEpilogueVisitor<typename GemmBase::Mma, VisitorEpilogue, Swizzle>>;
        using Gemm            = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
    };
    struct Plan {
        ggml_ext_gpu_device_info device;
        int64_t rows, n, k;
        bool swiglu;
        size_t kernel_bytes = 0, scratch_offset = 0;
        int occupancy                                                                                                       = 0;
        ggml_status (*launch)(Plan&, const int8_t*, const int8_t*, const float*, const float*, float*, void*, cudaStream_t) = nullptr;
    };
    template <class Types>
    static typename Types::Gemm::Arguments arguments(const Plan& p, const int8_t* x, const int8_t* w, const float* xs, const float* ws, float* out) {
        typename Types::ScaleW::Arguments scales     = {{{}, {xs, 0.f, {}}, {}}, {ws, 0.f, {}}, {}};
        typename Types::Epilogue::Arguments epilogue = {scales, {out, {p.n, cute::_1{}, p.rows * p.n}}};
        return {cutlass::gemm::GemmUniversalMode::kGemm, {int(p.rows), int(p.n), int(p.k)}, 1, epilogue, x, w, nullptr, nullptr, p.rows * p.k, p.n * p.k, 0, 0, p.k, p.k, 0, 0};
    }
    template <class Types>
    static ggml_status launch(Plan& p, const int8_t* x, const int8_t* w, const float* xs, const float* ws, float* out, void* scratch, cudaStream_t stream) {
        using Kernel = typename Types::Kernel;
        if ((reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(w) | reinterpret_cast<uintptr_t>(xs) |
             reinterpret_cast<uintptr_t>(ws) | reinterpret_cast<uintptr_t>(out)) %
            16)
            return GGML_STATUS_FAILED;
        auto* paired = static_cast<float*>(scratch);
        auto args    = arguments<Types>(p, x, w, xs, p.swiglu ? paired : ws, out);
        if (Kernel::can_implement(args) != cutlass::Status::kSuccess)
            return GGML_STATUS_FAILED;
        if (!p.occupancy) {
            if (sizeof(typename Kernel::SharedStorage) >= (48 << 10) &&
                cudaFuncSetAttribute(cutlass::Kernel2<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(typename Kernel::SharedStorage)) != cudaSuccess)
                return GGML_STATUS_FAILED;
            if (cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&p.occupancy, cutlass::Kernel2<Kernel>, Kernel::kThreadCount,
                                                                       sizeof(typename Kernel::SharedStorage), cudaOccupancyDisableCachingOverride) != cudaSuccess ||
                p.occupancy <= 0)
                return GGML_STATUS_FAILED;
        }
        typename Kernel::Params params(args, p.device.nsm, p.occupancy);
        if (params.get_workspace_size() > p.kernel_bytes)
            return GGML_STATUS_FAILED;
        if (p.swiglu) {
            pair_scales<<<unsigned((p.n + 255) / 256), 256, 0, stream>>>(ws, paired, int(p.n));
            if (cudaGetLastError() != cudaSuccess)
                return GGML_STATUS_FAILED;
        }
        if (params.init_workspace(static_cast<char*>(scratch) + p.scratch_offset, stream) != cutlass::Status::kSuccess)
            return GGML_STATUS_FAILED;
        cutlass::arch::synclog_setup();
        cutlass::Kernel2<Kernel><<<params.get_grid_dims(), Kernel::kThreadCount, sizeof(typename Kernel::SharedStorage), stream>>>(params);
        return cudaGetLastError() == cudaSuccess ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
    }
    template <class Types>
    static bool configure_kernel(Plan& p) {
        using Kernel = typename Types::Kernel;
        if (sizeof(typename Kernel::SharedStorage) > p.device.shared_memory)
            return false;
        auto args = arguments<Types>(p, nullptr, nullptr, nullptr, nullptr, nullptr);
        if (Kernel::can_implement(args) != cutlass::Status::kSuccess)
            return false;
        size_t bytes = 0;
        for (int occupancy = 1; occupancy <= p.device.max_threads_per_sm / Kernel::kThreadCount; ++occupancy) {
            typename Kernel::Params params(args, p.device.nsm, occupancy);
            const auto grid = params.get_grid_dims();
            if (grid.y > 65535 || grid.z > 65535)
                return false;
            bytes = std::max(bytes, params.get_workspace_size());
        }
        p.kernel_bytes   = bytes;
        p.scratch_offset = p.swiglu ? (size_t(p.n) * sizeof(float) + 255) / 256 * 256 : 0;
        p.launch         = launch<Types>;
        return true;
    }
    template <class Config>
    static bool configure_tile(Plan& p) {
        if (p.swiglu)
            return p.rows >= 8192 ? configure_kernel<SwigluTypes<Config, true>>(p) : configure_kernel<SwigluTypes<Config, false>>(p);
        return p.rows >= 8192 ? configure_kernel<DenseTypes<Config, true>>(p) : configure_kernel<DenseTypes<Config, false>>(p);
    }
    static bool configure(Plan& p) {
        auto try_tile = [&](Tile tile) {
            switch (tile) {
                case TILE_128_256:
                    return configure_tile<TileConfig<128, 256, 64>>(p);
                case TILE_128_128:
                    return configure_tile<TileConfig<128, 128, 64>>(p);
                case TILE_64_128:
                    return configure_tile<TileConfig<64, 128, 32>>(p);
                case TILE_16_128:
                    return configure_tile<TileConfig<16, 128, 16>>(p);
            }
            return false;
        };
        const auto tile = select_tile(p.rows, p.n, p.device.nsm);
        if (try_tile(tile))
            return true;
        for (auto fallback : {TILE_128_256, TILE_128_128, TILE_64_128, TILE_16_128})
            if (fallback != tile && try_tile(fallback))
                return true;
        return false;
    }
}
struct SDI8CudaGemmPlan : sd_i8_cutlass::Plan {};
SDI8CudaGemmPlan* sd_cuda_i8_gemm_prepare(ggml_backend_dev_t device, int64_t rows, int64_t n, int64_t k, bool swiglu, size_t* workspace) {
#ifdef SD_CUDA_FORCE_CUBLAS
    return nullptr;
#else
    const auto info = sd_cuda::cuda_device_info(device);
    if (info.device < 0 || info.cc < 800 || rows < (sd_i8_cutlass::small_rows_enabled(rows, n, k) ? 1 : 128) ||
        rows > INT_MAX || n < 1 || n > INT_MAX || k < 1 || k > INT_MAX || k % 64 || n % (swiglu ? 256 : 128))
        return nullptr;
    auto* p   = new SDI8CudaGemmPlan;
    p->device = info;
    p->rows   = rows;
    p->n      = n;
    p->k      = k;
    p->swiglu = swiglu;
    if (!sd_i8_cutlass::configure(*p)) {
        delete p;
        return nullptr;
    }
    *workspace = p->scratch_offset + p->kernel_bytes;
    return p;
#endif
}
ggml_status sd_cuda_i8_gemm_execute(SDI8CudaGemmPlan* plan, const int8_t* x, const int8_t* w, const float* xs, const float* ws, float* out, void* scratch, cudaStream_t stream) {
    return plan->launch(*plan, x, w, xs, ws, out, scratch, stream);
}
void sd_cuda_i8_gemm_free(SDI8CudaGemmPlan* plan) {
    delete plan;
}
