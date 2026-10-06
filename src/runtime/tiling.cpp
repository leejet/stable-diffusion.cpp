#include "runtime/tiling.h"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "core/util.h"
#include "ggml.h"

static int sd_tiling_calc_num_tiles(int dimension, int tile_size, float target_overlap_factor) {
    if (dimension <= tile_size) {
        return 1;
    } else if (dimension < 2 * tile_size) {
        return 2;
    } else if (dimension == 2 * tile_size) {
        return 3;
    } else {
        float target_num_tiles = 1.0f + (dimension - tile_size) / ((1.0f - target_overlap_factor) * tile_size);
        int num_tiles_lower    = static_cast<int>(std::floor(target_num_tiles));
        int num_tiles_upper    = static_cast<int>(std::ceil(target_num_tiles));
        int num_tiles_min      = 1 + (dimension - 2) / (tile_size - 1);  // positive adjacent overlap
        int num_tiles_max      = 2 * dimension / tile_size - 1;          // no triple overlap under Bresenham placement
        num_tiles_lower        = std::clamp(num_tiles_lower, num_tiles_min, num_tiles_max);
        num_tiles_upper        = std::clamp(num_tiles_upper, num_tiles_min, num_tiles_max);
        auto overlap_error     = [target_num_tiles](int num_tiles) -> float {
            return std::abs(1.0f / (num_tiles - 1.0f) - 1.0f / (target_num_tiles - 1.0f));
        };  // (dimension - tile_size) / tile_size factors out
        return (overlap_error(num_tiles_upper) < overlap_error(num_tiles_lower)) ? num_tiles_upper : num_tiles_lower;  // use lower if tie
    }
}

static float sd_tiling_calc_average_stride_factor(int dimension, int tile_size, int num_tiles) {
    return static_cast<float>(dimension - tile_size) / static_cast<float>(tile_size * (num_tiles - 1));
}

static void sd_tiling_calc_tiles(int& num_tiles_dim,
                                 float& tile_overlap_factor_dim,
                                 int small_dim,
                                 int tile_size,
                                 const float tile_overlap_factor,
                                 bool circular) {
    int tile_overlap     = static_cast<int>(tile_size * tile_overlap_factor);
    int non_tile_overlap = tile_size - tile_overlap;

    if (circular) {
        // circular means the last and first tile are overlapping (wraping around)
        num_tiles_dim = small_dim / non_tile_overlap;

        if (num_tiles_dim < 1) {
            num_tiles_dim = 1;
        }

        tile_overlap_factor_dim = (tile_size - small_dim / num_tiles_dim) / (float)tile_size;

        // if single tile and tile_overlap_factor is not 0, add one to ensure we have at least two overlapping tiles
        if (num_tiles_dim == 1 && tile_overlap_factor_dim > 0) {
            num_tiles_dim++;
            tile_overlap_factor_dim = 0.5;
        }
    } else {
        num_tiles_dim           = sd_tiling_calc_num_tiles(small_dim, tile_size, tile_overlap_factor);
        tile_overlap_factor_dim = (num_tiles_dim == 1) ? 0 : (1.0f - sd_tiling_calc_average_stride_factor(small_dim, tile_size, num_tiles_dim));
    }
}

static int64_t sd_tensor_plane_size(const sd::Tensor<float>& tensor) {
    GGML_ASSERT(tensor.dim() >= 2);
    return tensor.shape()[0] * tensor.shape()[1];
}

static sd::Tensor<float> sd_tensor_split_2d(const sd::Tensor<float>& input, int width, int height, int x, int y) {
    GGML_ASSERT(input.dim() >= 4);
    std::vector<int64_t> output_shape = input.shape();
    output_shape[0]                   = width;
    output_shape[1]                   = height;
    sd::Tensor<float> output(std::move(output_shape));
    int64_t input_width  = input.shape()[0];
    int64_t input_height = input.shape()[1];
    int64_t input_plane  = sd_tensor_plane_size(input);
    int64_t output_plane = sd_tensor_plane_size(output);
    int64_t plane_count  = input.numel() / input_plane;
    for (int iy = 0; iy < height; iy++) {
        for (int ix = 0; ix < width; ix++) {
            int64_t src_xy = (ix + x) % input_width + input_width * ((iy + y) % input_height);
            int64_t dst_xy = ix + width * iy;
            for (int64_t plane = 0; plane < plane_count; ++plane) {
                output[plane * output_plane + dst_xy] = input[plane * input_plane + src_xy];
            }
        }
    }
    return output;
}

static void sd_tensor_merge_2d(const sd::Tensor<float>& input,
                               sd::Tensor<float>* output,
                               int x,
                               int y,
                               int overlap_x,
                               int overlap_y,
                               bool circular_x,
                               bool circular_y,
                               int x_skip,
                               int y_skip) {
    GGML_ASSERT(output != nullptr);
    int64_t width        = input.shape()[0];
    int64_t height       = input.shape()[1];
    int64_t img_width    = output->shape()[0];
    int64_t img_height   = output->shape()[1];
    int64_t input_plane  = sd_tensor_plane_size(input);
    int64_t output_plane = sd_tensor_plane_size(*output);
    int64_t plane_count  = input.numel() / input_plane;
    GGML_ASSERT(output->numel() / output_plane == plane_count);

    // unclamped -> expects x in the range [0-1]
    auto smootherstep_f32 = [](const float x) -> float {
        GGML_ASSERT(x >= 0.f && x <= 1.f);
        return x * x * x * (x * (6.0f * x - 15.0f) + 10.0f);
    };

    for (int iy = y_skip; iy < height; iy++) {
        for (int ix = x_skip; ix < width; ix++) {
            int64_t src_xy = ix + width * iy;
            int64_t ox     = (x + ix) % img_width;
            int64_t oy     = (y + iy) % img_height;
            int64_t dst_xy = ox + img_width * oy;
            for (int64_t plane = 0; plane < plane_count; ++plane) {
                float new_value = input[plane * input_plane + src_xy];
                if (overlap_x > 0 || overlap_y > 0) {
                    float old_value   = (*output)[plane * output_plane + dst_xy];
                    const float x_f_0 = (circular_x || (overlap_x > 0 && x > 0)) ? (ix - x_skip) / float(overlap_x) : 1.f;
                    const float x_f_1 = (circular_x || (overlap_x > 0 && x < (img_width - width))) ? (width - ix) / float(overlap_x) : 1.f;
                    const float y_f_0 = (circular_y || (overlap_y > 0 && y > 0)) ? (iy - y_skip) / float(overlap_y) : 1.f;
                    const float y_f_1 = (circular_y || (overlap_y > 0 && y < (img_height - height))) ? (height - iy) / float(overlap_y) : 1.f;
                    const float x_f   = std::min(std::min(x_f_0, x_f_1), 1.f);
                    const float y_f   = std::min(std::min(y_f_0, y_f_1), 1.f);
                    (*output)[plane * output_plane + dst_xy] =
                        old_value + new_value * smootherstep_f32(y_f) * smootherstep_f32(x_f);
                } else {
                    (*output)[plane * output_plane + dst_xy] = new_value;
                }
            }
        }
    }
}

static void sd_tensor_merge_2d_non_circular(const sd::Tensor<float>& input,
                                            sd::Tensor<float>* output,
                                            int x,
                                            int y,
                                            int overlap_left,
                                            int overlap_right,
                                            int overlap_top,
                                            int overlap_bottom) {
    GGML_ASSERT(output != nullptr);

    int64_t in_width    = input.shape()[0];
    int64_t in_height   = input.shape()[1];
    int64_t out_width   = output->shape()[0];
    int64_t out_height  = output->shape()[1];
    int64_t in_size     = sd_tensor_plane_size(input);
    int64_t out_size    = sd_tensor_plane_size(*output);
    int64_t plane_count = input.numel() / in_size;

    GGML_ASSERT(output->numel() == plane_count * out_size);
    GGML_ASSERT(x >= 0 && y >= 0);
    GGML_ASSERT(x + in_width <= out_width);
    GGML_ASSERT(y + in_height <= out_height);
    GGML_ASSERT(overlap_left >= 0 && overlap_right >= 0);
    GGML_ASSERT(overlap_top >= 0 && overlap_bottom >= 0);

    auto smootherstep_f32 = [](const float x) -> float {
        return x * x * x * (x * (6.0f * x - 15.0f) + 10.0f);
    };
    for (int64_t plane = 0; plane < plane_count; ++plane) {
        for (int iy = 0; iy < in_height; ++iy) {
            float y_f = 1.0f;
            if (iy < overlap_top) {
                y_f = static_cast<float>(iy) / overlap_top;
            }
            if (iy >= in_height - overlap_bottom) {
                y_f = static_cast<float>(in_height - iy) / overlap_bottom;
            }
            const float y_weight = smootherstep_f32(std::clamp(y_f, 0.0f, 1.0f));
            for (int ix = 0; ix < in_width; ++ix) {
                float x_f = 1.0f;
                if (ix < overlap_left) {
                    x_f = static_cast<float>(ix) / overlap_left;
                }
                if (ix >= in_width - overlap_right) {
                    x_f = static_cast<float>(in_width - ix) / overlap_right;
                }
                float x_weight = smootherstep_f32(std::clamp(x_f, 0.0f, 1.0f));
                (*output)[plane * out_size + out_width * (y + iy) + (x + ix)] += x_weight * y_weight * input[plane * in_size + in_width * iy + ix];
            }
        }
    }
}

sd::Tensor<float> process_tiles_2d(const sd::Tensor<float>& input,
                                   int output_width,
                                   int output_height,
                                   int scale,
                                   int p_tile_size_w,
                                   int p_tile_size_h,
                                   float tile_overlap_factor,
                                   bool circular_x,
                                   bool circular_y,
                                   const TileProcessCallback& on_processing,
                                   bool silent) {
    sd::Tensor<float> output;
    int input_width  = static_cast<int>(input.shape()[0]);
    int input_height = static_cast<int>(input.shape()[1]);

    GGML_ASSERT(((input_width / output_width) == (input_height / output_height)) &&
                ((output_width / input_width) == (output_height / input_height)));
    GGML_ASSERT(((input_width / output_width) == scale) ||
                ((output_width / input_width) == scale));

    bool decode      = output_width > input_width;  // scale up
    int small_width  = decode ? input_width : output_width;
    int small_height = decode ? input_height : output_height;
    int scale_in     = decode ? 1 : scale;
    int scale_out    = decode ? scale : 1;

    int num_tiles_x;
    float tile_overlap_factor_x;
    sd_tiling_calc_tiles(num_tiles_x, tile_overlap_factor_x, small_width, p_tile_size_w, tile_overlap_factor, circular_x);

    int num_tiles_y;
    float tile_overlap_factor_y;
    sd_tiling_calc_tiles(num_tiles_y, tile_overlap_factor_y, small_height, p_tile_size_h, tile_overlap_factor, circular_y);

    int tile_width         = std::min(p_tile_size_w, small_width);
    int tile_height        = std::min(p_tile_size_h, small_height);
    int input_tile_width   = tile_width * scale_in;
    int input_tile_height  = tile_height * scale_in;
    int output_tile_width  = tile_width * scale_out;
    int output_tile_height = tile_height * scale_out;

    int num_tiles   = num_tiles_x * num_tiles_y;
    int tile_count  = 1;
    float last_time = 0.0f;
    if (!silent) {
        LOG_VERBOSE("num tiles : %d, %d ", num_tiles_x, num_tiles_y);
        LOG_VERBOSE("optimal overlap : %f, %f (targeting %f)", tile_overlap_factor_x, tile_overlap_factor_y, tile_overlap_factor);
        LOG_VERBOSE("processing %i tiles", num_tiles);
        pretty_progress(0, num_tiles, 0.0f);
    }
    if (circular_x || circular_y) {
        int tile_overlap_x     = static_cast<int32_t>(p_tile_size_w * tile_overlap_factor_x);
        int non_tile_overlap_x = p_tile_size_w - tile_overlap_x;
        int tile_overlap_y     = static_cast<int32_t>(p_tile_size_h * tile_overlap_factor_y);
        int non_tile_overlap_y = p_tile_size_h - tile_overlap_y;

        bool last_y = false;
        bool last_x = false;

        for (int y = 0; y < small_height && !last_y; y += non_tile_overlap_y) {
            int dy = 0;
            if (!circular_y && y + tile_height >= small_height) {
                int original_y = y;
                y              = small_height - tile_height;
                dy             = original_y - y;
                if (decode) {
                    dy *= scale;
                }
                last_y = true;
            }
            for (int x = 0; x < small_width && !last_x; x += non_tile_overlap_x) {
                int dx = 0;
                if (!circular_x && x + tile_width >= small_width) {
                    int original_x = x;
                    x              = small_width - tile_width;
                    dx             = original_x - x;
                    if (decode) {
                        dx *= scale;
                    }
                    last_x = true;
                }

                int x_in  = decode ? x : scale * x;
                int y_in  = decode ? y : scale * y;
                int x_out = decode ? x * scale : x;
                int y_out = decode ? y * scale : y;

                int overlap_x_out = decode ? tile_overlap_x * scale : tile_overlap_x;
                int overlap_y_out = decode ? tile_overlap_y * scale : tile_overlap_y;

                int64_t t1       = ggml_time_ms();
                auto input_tile  = sd_tensor_split_2d(input, input_tile_width, input_tile_height, x_in, y_in);
                auto output_tile = on_processing(input_tile);
                if (output_tile.empty()) {
                    return {};
                }
                GGML_ASSERT(output_tile.shape()[0] == output_tile_width && output_tile.shape()[1] == output_tile_height);
                if (output.empty()) {
                    std::vector<int64_t> output_shape = output_tile.shape();
                    output_shape[0]                   = output_width;
                    output_shape[1]                   = output_height;
                    output                            = sd::Tensor<float>::zeros(std::move(output_shape));
                }
                sd_tensor_merge_2d(output_tile, &output, x_out, y_out, overlap_x_out, overlap_y_out, circular_x, circular_y, dx, dy);

                if (!silent) {
                    int64_t t2 = ggml_time_ms();
                    last_time  = (t2 - t1) / 1000.0f;
                    pretty_progress(tile_count, num_tiles, last_time);
                }
                tile_count++;
            }
            last_x = false;
        }
    } else {
        for (int j = 0; j < num_tiles_y; ++j) {
            int y              = 0;
            int overlap_top    = 0;
            int overlap_bottom = 0;
            if (num_tiles_y > 1) {
                y = j * (small_height - tile_height) / (num_tiles_y - 1);
                if (j > 0) {
                    int y_prev  = (j - 1) * (small_height - tile_height) / (num_tiles_y - 1);
                    overlap_top = y_prev + tile_height - y;
                }
                if (j < num_tiles_y - 1) {
                    int y_next     = (j + 1) * (small_height - tile_height) / (num_tiles_y - 1);
                    overlap_bottom = y + tile_height - y_next;
                }
            }
            for (int i = 0; i < num_tiles_x; ++i) {
                int x             = 0;
                int overlap_left  = 0;
                int overlap_right = 0;
                if (num_tiles_x > 1) {
                    x = i * (small_width - tile_width) / (num_tiles_x - 1);
                    if (i > 0) {
                        int x_prev   = (i - 1) * (small_width - tile_width) / (num_tiles_x - 1);
                        overlap_left = x_prev + tile_width - x;
                    }
                    if (i < num_tiles_x - 1) {
                        int x_next    = (i + 1) * (small_width - tile_width) / (num_tiles_x - 1);
                        overlap_right = x + tile_width - x_next;
                    }
                }

                int64_t t1       = ggml_time_ms();
                auto input_tile  = sd_tensor_split_2d(input, input_tile_width, input_tile_height, x * scale_in, y * scale_in);
                auto output_tile = on_processing(input_tile);
                if (output_tile.empty()) {
                    return {};
                }
                GGML_ASSERT(output_tile.shape()[0] == output_tile_width && output_tile.shape()[1] == output_tile_height);
                if (output.empty()) {
                    std::vector<int64_t> output_shape = output_tile.shape();
                    output_shape[0]                   = output_width;
                    output_shape[1]                   = output_height;
                    output                            = sd::Tensor<float>::zeros(std::move(output_shape));
                }
                sd_tensor_merge_2d_non_circular(
                    output_tile, &output,
                    x * scale_out, y * scale_out,
                    overlap_left * scale_out,
                    overlap_right * scale_out,
                    overlap_top * scale_out,
                    overlap_bottom * scale_out);
                if (!silent) {
                    last_time = (ggml_time_ms() - t1) / 1000.0f;
                    pretty_progress(tile_count, num_tiles, last_time);
                }
                tile_count++;
            }
        }
    }
    if (!silent && tile_count < num_tiles) {
        pretty_progress(num_tiles, num_tiles, last_time);
    }
    if (output.empty()) {
        return {};
    }
    return output;
}
