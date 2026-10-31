#ifndef __SD_CORE_CUDA_I8_GEMM_CUH__
#define __SD_CORE_CUDA_I8_GEMM_CUH__

#include <cutlass/gemm/threadblock/threadblock_swizzle_streamk.h>
#include <cstdlib>

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

}

#endif
