#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle_streamk.h>
#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>
#include <mutex>
#include <unordered_map>
#include "convrot_kernel.h"
#include "i8_gemm.h"

namespace sd_i8_cutlass {
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

    template <int M, int N, int WM, int Stages = 4, int Band = 1>
    struct TileConfig {
        using Tile                   = cutlass::gemm::GemmShape<M, N, 64>;
        using Warp                   = cutlass::gemm::GemmShape<WM, 64, 64>;
        static constexpr int kStages = Stages;
        static constexpr int kBand   = Band;
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

    struct DeviceInfo {
        int ordinal        = -1;
        int major          = 0;
        int minor          = 0;
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
            info = {ordinal, props.major, props.minor, props.multiProcessorCount, props.maxThreadsPerMultiProcessor, size_t(props.sharedMemPerBlockOptin)};
        cache.emplace(device, info);
        return info;
    }
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

    template <class Base>
    struct SplitWeightIterator : Base {
        using Layout      = typename Base::Layout;
        using Pointer     = typename Base::Pointer;
        using TensorCoord = typename Base::TensorCoord;
        using AccessType  = typename Base::AccessType;
        struct Params : Base::Params {
            const int8_t* up = nullptr;
            int64_t k        = 0;
            Params()         = default;
            CUTLASS_HOST_DEVICE Params(const Layout& layout)
                : Base::Params(layout), k(layout.stride(0)) {}
        };
        uintptr_t gate, up;
        size_t gate_bytes;
        CUTLASS_DEVICE SplitWeightIterator(const Params& params, Pointer pointer, TensorCoord extent, int thread, TensorCoord offset, const int* indices = nullptr)
            : Base(params, pointer, extent, thread, offset, indices),
              gate(reinterpret_cast<uintptr_t>(pointer)),
              up(reinterpret_cast<uintptr_t>(params.up)),
              gate_bytes(size_t(params.k) * (extent.column() / 2)) {}

        CUTLASS_DEVICE AccessType* get() const {
            auto* address = Base::get();
            if (!address)
                return nullptr;
            // Keep the packed SwiGLU iterator's column pairing, but redirect its
            // virtual up half to a separate allocation. Params.k stays full-sized
            // even when Stream-K restricts this iterator's K extent.
            const uintptr_t offset = reinterpret_cast<uintptr_t>(address) - gate;
            return offset < gate_bytes ? address : reinterpret_cast<AccessType*>(up + offset - gate_bytes);
        }
    };

    template <class Map>
    struct GatedStore : cutlass::epilogue::threadblock::VisitorAuxStore<Map, float, cutlass::FloatRoundStyle::round_to_nearest, cute::Stride<int64_t, cute::_1, int64_t>> {
        using Base = cutlass::epilogue::threadblock::VisitorAuxStore<Map, float, cutlass::FloatRoundStyle::round_to_nearest, cute::Stride<int64_t, cute::_1, int64_t>>;
        struct Arguments : Base::Arguments {
            bool situ = false;
        };
        using Params = Arguments;

        template <class Shape>
        static constexpr Params to_underlying_arguments(Shape const&, Arguments const& args, void*) {
            return args;
        }

        GatedStore() = default;
        CUTLASS_HOST_DEVICE GatedStore(Params const& params, typename Base::SharedStorage const& shared)
            : Base(params, shared), situ(params.situ) {}

        bool situ = false;

        template <class CB>
        struct Callbacks : CB {
            bool situ;

            CUTLASS_DEVICE Callbacks(CB const& base, bool situ)
                : CB(base), situ(situ) {}
            CUTLASS_DEVICE void end_step(int step) {
                auto src    = cute::filter(this->tC_rAux);
                auto coords = cute::filter(this->tC_cAux(cute::_, cute::_, cute::_, step));
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < cute::size(src); ++i) {
                    auto coord = coords(i);
                    if (cute::elem_less(coord, this->problem_shape)) {
                        // SwiGLU packs [gate | up] and SiTU packs [up | gate].
                        const float4 a     = *reinterpret_cast<const float4*>(&src(i));
                        const float2 value = situ ? make_float2(sd_cuda::sd_convrot::situ(a.y, a.x), sd_cuda::sd_convrot::situ(a.w, a.z))
                                                  : make_float2(__fmul_rn(a.x / (1.f + expf(-a.x)), a.y), __fmul_rn(a.z / (1.f + expf(-a.z)), a.w));
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
            return Callbacks<decltype(cb)>(cb, situ);
        }
    };

    static __global__ void pair_scales(const float* src, const float* up, float* dst, int n) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            dst[i] = up ? (i % 2 ? up[i / 2] : src[i / 2]) : src[i / 2 + (i % 2) * (n / 2)];
    }

    template <class Fragment>
    struct AddBias {
        CUTLASS_DEVICE Fragment operator()(const Fragment& value, const Fragment& bias) const {
            Fragment result;
            // Prevent contraction with the preceding scale multiplication.
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < Fragment::kElements; ++i)
                result[i] = __fadd_rn(value[i], bias[i]);
            return result;
        }
    };

    template <class Config, bool StreamK, bool PackedW4 = false>
    struct DenseTypes {
        static constexpr bool kSplit = false;
        static constexpr bool kGated = false;
        static constexpr bool kBias  = !PackedW4;
        using Weight                 = std::conditional_t<PackedW4, cutlass::int4b_t, int8_t>;
        using Tile                   = typename Config::Tile;
        using Warp                   = typename Config::Warp;
        using Swizzle                = std::conditional_t<StreamK, BandedStreamK, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>>;
        using Map                    = cutlass::epilogue::threadblock::OutputTileThreadLayout<Tile, Warp, float, 4, 1>;
        using Col                    = cutlass::epilogue::threadblock::VisitorColBroadcast<Map, float, cute::Stride<cute::_1, cute::_0, int>>;
        using Row                    = cutlass::epilogue::threadblock::VisitorRowBroadcast<Map, float, cute::Stride<cute::_0, cute::_1, int>>;
        using Mul                    = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
        using ScaleX                 = cutlass::epilogue::threadblock::Sm80EVT<Mul, cutlass::epilogue::threadblock::VisitorAccFetch, Col>;
        using ScaleW                 = cutlass::epilogue::threadblock::Sm80EVT<Mul, ScaleX, Row>;
        using Store                  = cutlass::epilogue::threadblock::VisitorAuxStore<Map, float, cutlass::FloatRoundStyle::round_to_nearest, cute::Stride<int64_t, cute::_1, int64_t>>;
        using Add                    = cutlass::epilogue::threadblock::VisitorCompute<AddBias, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
        using Biased                 = cutlass::epilogue::threadblock::Sm80EVT<Add, ScaleW, Row>;
        using Epilogue               = cutlass::epilogue::threadblock::Sm80EVT<Store, std::conditional_t<kBias, Biased, ScaleW>>;
        using Kernel                 = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
            int8_t,
            cutlass::layout::RowMajor,
            cutlass::ComplexTransform::kNone,
            16,
            Weight,
            cutlass::layout::ColumnMajor,
            cutlass::ComplexTransform::kNone,
            PackedW4 ? 32 : 16,
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
            std::conditional_t<PackedW4, cutlass::arch::OpMultiplyAddMixedInputUpcast, cutlass::arch::OpMultiplyAddSaturate>>::GemmKernel;
        using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
    };
    template <class Config, bool StreamK, bool Split = false>
    struct GatedTypes {
        static constexpr bool kSplit = Split;
        static constexpr bool kBias  = false;
        static constexpr bool kGated = true;
        using Weight                 = int8_t;
        using Tile                   = typename Config::Tile;
        using Warp                   = typename Config::Warp;
        using Swizzle                = std::conditional_t<StreamK, BandedStreamK, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<Config::kBand>>;
        using Map                    = cutlass::epilogue::threadblock::OutputTileThreadLayout<Tile, Warp, float, 4, 1>;
        using Col                    = cutlass::epilogue::threadblock::VisitorColBroadcast<Map, float, cute::Stride<cute::_1, cute::_0, int>>;
        using Row                    = cutlass::epilogue::threadblock::VisitorRowBroadcast<Map, float, cute::Stride<cute::_0, cute::_1, int>>;
        using Mul                    = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
        using ScaleX                 = cutlass::epilogue::threadblock::Sm80EVT<Mul, cutlass::epilogue::threadblock::VisitorAccFetch, Col>;
        using ScaleW                 = cutlass::epilogue::threadblock::Sm80EVT<Mul, ScaleX, Row>;
        using Store                  = GatedStore<Map>;
        using Epilogue               = cutlass::epilogue::threadblock::Sm80EVT<Store, ScaleW>;
        using GemmBase               = typename cutlass::gemm::kernel::DefaultGemmUniversal<int8_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 16, int8_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 16, float, cutlass::layout::RowMajor, int, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, Tile, Warp, cutlass::gemm::GemmShape<16, 8, 32>, cutlass::epilogue::thread::LinearCombination<float, 4, int, float>, Swizzle, Config::kStages, cutlass::arch::OpMultiplyAddSaturate, cutlass::gemm::SharedMemoryClearOption::kNone, false, false, false, cutlass::layout::NoPermute, cutlass::layout::NoPermute, WeightLayout>::GemmKernel;
        using VisitorEpilogue        = cutlass::epilogue::threadblock::EpilogueWithVisitorCallbacks<typename GemmBase::Epilogue, Epilogue, 1>;
        using BaseMma                = typename GemmBase::Mma;
        using SplitMma               = cutlass::gemm::threadblock::MmaMultistage<
            typename BaseMma::Shape,
            typename BaseMma::IteratorA,
            typename BaseMma::SmemIteratorA,
            BaseMma::kCacheOpA,
            SplitWeightIterator<typename BaseMma::IteratorB>,
            typename BaseMma::SmemIteratorB,
            BaseMma::kCacheOpB,
            typename BaseMma::ElementC,
            typename BaseMma::LayoutC,
            typename BaseMma::Policy,
            BaseMma::Detail::kStages>;
        using Mma    = std::conditional_t<Split, SplitMma, BaseMma>;
        using Kernel = std::conditional_t<StreamK,
                                          cutlass::gemm::kernel::GemmWithEpilogueVisitorStreamk<Mma, VisitorEpilogue, Swizzle>,
                                          cutlass::gemm::kernel::GemmWithEpilogueVisitor<Mma, VisitorEpilogue, Swizzle>>;
        using Gemm   = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
    };
    struct Plan {
        DeviceInfo device;
        int64_t rows, n, k;
        SDI8GemmActivation activation;
        bool packed_w4;
        bool split;
        size_t kernel_bytes = 0, scratch_offset = 0;
        int occupancy                                                                                                                                                  = 0;
        ggml_status (*launch)(Plan&, const int8_t*, const int8_t*, const float*, const float*, float*, void*, cudaStream_t, const int8_t*, const float*, const float*) = nullptr;
    };
    template <class Types>
    static typename Types::Gemm::Arguments arguments(const Plan& p, const int8_t* x, const int8_t* w, const float* xs, const float* ws, float* out, const float* bias) {
        typename Types::ScaleW::Arguments scales = {{{}, {xs, 0.f, {}}, {}}, {ws, 0.f, {}}, {}};
        typename Types::Epilogue::Arguments epilogue;
        if constexpr (Types::kBias) {
            // A negative zero default preserves signed-zero outputs.
            epilogue = {{scales, {bias, -0.f, {}}, {}}, {out, {p.n, cute::_1{}, p.rows * p.n}}};
        } else if constexpr (Types::kGated) {
            epilogue = {scales, {{out, {p.n, cute::_1{}, p.rows * p.n}}, p.activation == SDI8GemmActivation::Situ}};
        } else {
            epilogue = {scales, {out, {p.n, cute::_1{}, p.rows * p.n}}};
        }
        return {cutlass::gemm::GemmUniversalMode::kGemm, {int(p.rows), int(p.n), int(p.k)}, 1, epilogue, x, reinterpret_cast<const typename Types::Weight*>(w), nullptr, nullptr, p.rows * p.k, p.n * p.k, 0, 0, p.k, p.k, 0, 0};
    }
    template <class Types>
    static ggml_status launch(Plan& p, const int8_t* x, const int8_t* w, const float* xs, const float* ws, float* out, void* scratch, cudaStream_t stream, const int8_t* up, const float* up_scales, const float* bias) {
        using Kernel = typename Types::Kernel;
        if ((reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(w) | reinterpret_cast<uintptr_t>(xs) |
             reinterpret_cast<uintptr_t>(ws) | reinterpret_cast<uintptr_t>(out) | reinterpret_cast<uintptr_t>(up) | reinterpret_cast<uintptr_t>(up_scales) | reinterpret_cast<uintptr_t>(bias)) %
            16)
            return GGML_STATUS_FAILED;
        auto* paired = static_cast<float*>(scratch);
        auto args    = arguments<Types>(p, x, w, xs, p.activation != SDI8GemmActivation::None ? paired : ws, out, bias);
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
        typename Kernel::Params params(args, p.device.sms, p.occupancy);
        if constexpr (Types::kSplit) {
            if (!up || !up_scales)
                return GGML_STATUS_FAILED;
            params.params_B.up = up;
        }
        if (params.get_workspace_size() > p.kernel_bytes)
            return GGML_STATUS_FAILED;
        if (p.activation != SDI8GemmActivation::None) {
            pair_scales<<<unsigned((p.n + 255) / 256), 256, 0, stream>>>(ws, up_scales, paired, int(p.n));
            if (cudaGetLastError() != cudaSuccess)
                return GGML_STATUS_FAILED;
        }
        if (params.init_workspace(scratch ? static_cast<char*>(scratch) + p.scratch_offset : nullptr, stream) != cutlass::Status::kSuccess)
            return GGML_STATUS_FAILED;
        cutlass::arch::synclog_setup();
        cutlass::Kernel2<Kernel><<<params.get_grid_dims(), Kernel::kThreadCount, sizeof(typename Kernel::SharedStorage), stream>>>(params);
        return cudaGetLastError() == cudaSuccess ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
    }
    template <class Types>
    static bool configure_kernel(Plan& p) {
        using Kernel = typename Types::Kernel;
        if (sizeof(typename Kernel::SharedStorage) > p.device.shared)
            return false;
        auto args = arguments<Types>(p, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
        if (Kernel::can_implement(args) != cutlass::Status::kSuccess)
            return false;
        size_t bytes = 0;
        for (int occupancy = 1; occupancy <= p.device.threads_per_sm / Kernel::kThreadCount; ++occupancy) {
            typename Kernel::Params params(args, p.device.sms, occupancy);
            const auto grid = params.get_grid_dims();
            if (grid.y > 65535 || grid.z > 65535)
                return false;
            bytes = std::max(bytes, params.get_workspace_size());
        }
        p.kernel_bytes   = bytes;
        p.scratch_offset = p.activation != SDI8GemmActivation::None ? (size_t(p.n) * sizeof(float) + 255) / 256 * 256 : 0;
        p.launch         = launch<Types>;
        return true;
    }
    template <class Config>
    static bool configure_tile(Plan& p) {
        if (p.packed_w4)
            return configure_kernel<DenseTypes<Config, false, true>>(p);
        if (p.split)
            return p.rows >= 8192 ? configure_kernel<GatedTypes<Config, true, true>>(p) : configure_kernel<GatedTypes<Config, false, true>>(p);
        if (p.activation != SDI8GemmActivation::None)
            return p.rows >= 8192 ? configure_kernel<GatedTypes<Config, true>>(p) : configure_kernel<GatedTypes<Config, false>>(p);
        return p.rows >= 8192 ? configure_kernel<DenseTypes<Config, true>>(p) : configure_kernel<DenseTypes<Config, false>>(p);
    }
    static bool configure(Plan& p) {
        // This SiTU shape fits two resident CTAs per SM with the narrower, three-stage tile on Ada.
        if (p.device.major == 8 && p.device.minor == 9 && p.activation == SDI8GemmActivation::Situ && !p.split &&
            p.rows >= 4096 && p.rows < 8192 && p.n == 16384 && p.k == 3072 &&
            configure_kernel<GatedTypes<TileConfig<128, 128, 64, 3, 4>, false>>(p))
            return true;
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
        const auto tile = select_tile(p.rows, p.n, p.device.sms);
        if (try_tile(tile))
            return true;
        for (auto fallback : {TILE_128_256, TILE_128_128, TILE_64_128, TILE_16_128})
            if (fallback != tile && try_tile(fallback))
                return true;
        return false;
    }
}
struct SDI8CudaGemmPlan : sd_i8_cutlass::Plan {};
static SDI8CudaGemmPlan* prepare_gemm(ggml_backend_dev_t device, int64_t rows, int64_t n, int64_t k, SDI8GemmActivation activation, size_t* workspace, bool packed_w4, bool split) {
#ifdef SD_CUDA_FORCE_CUBLAS
    return nullptr;
#else
    const auto info  = sd_i8_cutlass::device_info(device);
    const bool gated = activation != SDI8GemmActivation::None;
    if ((packed_w4 && (gated || k > INT_MAX / 1024)) || info.major < 8 || rows < (sd_i8_cutlass::small_rows_enabled(rows, n, k) ? 1 : 128) ||
        rows > INT_MAX || n < 1 || n > INT_MAX || k < 1 || k > INT_MAX || k % 64 || n % (gated ? 256 : 128))
        return nullptr;
    auto* p       = new SDI8CudaGemmPlan;
    p->device     = info;
    p->rows       = rows;
    p->n          = n;
    p->k          = k;
    p->activation = activation;
    p->packed_w4  = packed_w4;
    p->split      = split;
    if (!sd_i8_cutlass::configure(*p)) {
        delete p;
        return nullptr;
    }
    *workspace = p->scratch_offset + p->kernel_bytes;
    return p;
#endif
}
SDI8CudaGemmPlan* sd_cuda_i8_gemm_prepare(ggml_backend_dev_t device, int64_t rows, int64_t n, int64_t k, SDI8GemmActivation activation, size_t* workspace, bool packed_w4) {
    return prepare_gemm(device, rows, n, k, activation, workspace, packed_w4, false);
}
SDI8CudaGemmPlan* sd_cuda_i8_gemm_prepare_split(ggml_backend_dev_t device, int64_t rows, int64_t n, int64_t k, size_t* workspace) {
    if (n > INT_MAX / 2 || k > INT_MAX / 16384)
        return nullptr;
    return prepare_gemm(device, rows, 2 * n, k, SDI8GemmActivation::Swiglu, workspace, false, true);
}
ggml_status sd_cuda_i8_gemm_execute(SDI8CudaGemmPlan* plan, const int8_t* x, const int8_t* w, const float* xs, const float* ws, float* out, void* scratch, cudaStream_t stream, const float* bias) {
    if (bias && (plan->activation != SDI8GemmActivation::None || plan->packed_w4))
        return GGML_STATUS_FAILED;
    return plan->launch(*plan, x, w, xs, ws, out, scratch, stream, nullptr, nullptr, bias);
}
ggml_status sd_cuda_i8_gemm_execute_split(SDI8CudaGemmPlan* plan, const int8_t* x, const int8_t* gate, const int8_t* up, const float* xs, const float* gate_scales, const float* up_scales, float* out, void* scratch, cudaStream_t stream) {
    return plan->launch(*plan, x, gate, xs, gate_scales, out, scratch, stream, up, up_scales, nullptr);
}
void sd_cuda_i8_gemm_free(SDI8CudaGemmPlan* plan) {
    delete plan;
}
