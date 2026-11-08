#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>
#include <metal_tensor>
using namespace metal;
using namespace mpp::tensor_ops;

struct MatParams {
    uint k;
    uint n;
    uint rows;
    uint flags;
};

// TensorOps requires explicit signed INT8 and non-const tensor element types.
kernel void sd_i8_multiply_mpp(device int8_t* weight [[buffer(0)]],
                               device int8_t* input [[buffer(1)]],
                               device const float* xs [[buffer(2)]],
                               device const float* ws [[buffer(3)]],
                               device const float* bias [[buffer(4)]],
                               device float* output [[buffer(5)]],
                               constant MatParams& p [[buffer(6)]],
                               uint2 tile [[threadgroup_position_in_grid]]) {
    auto a                    = tensor<device int8_t, dextents<int, 2>, tensor_inline>(input, dextents<int, 2>(p.k, p.rows));
    auto b                    = tensor<device int8_t, dextents<int, 2>, tensor_inline>(weight, dextents<int, 2>(p.k, p.n));
    constexpr int chunk_size  = 1024;
    constexpr auto descriptor = matmul2d_descriptor(32, 32, chunk_size, false, true, false);
    matmul2d<descriptor, execution_simdgroups<4>> multiply;
    auto left  = a.slice(0, tile.y * 32);
    auto right = b.slice(0, tile.x * 32);
    auto acc   = multiply.get_destination_cooperative_tensor<decltype(left), decltype(right), int>();
    auto sums  = multiply.get_destination_cooperative_tensor<decltype(left), decltype(right), int>();
#pragma unroll
    for (uint i = 0; i < sums.get_capacity(); ++i) {
        if (sums.is_valid_element(i))
            sums[i] = 0;
    }
    // TensorOps can round long integer dot products on older GPUs. Bound each
    // partial sum to FP32's exact integer range, then combine with wrapping
    // integer addition outside TensorOps.
    for (uint k = 0; k < p.k; k += chunk_size) {
        left  = a.slice(k, tile.y * 32);
        right = b.slice(k, tile.x * 32);
        multiply.run(left, right, acc);
#pragma unroll
        for (uint i = 0; i < acc.get_capacity(); ++i) {
            if (acc.is_valid_element(i))
                sums[i] = as_type<int>(as_type<uint>(sums[i]) + as_type<uint>(acc[i]));
        }
    }
#pragma unroll
    for (uint i = 0; i < acc.get_capacity(); ++i) {
        if (acc.is_valid_element(i)) {
            auto position = acc.get_multidimensional_index(i);
            uint col = tile.x * 32 + position[0], row = tile.y * 32 + position[1];
            if (col < p.n && row < p.rows) {
                float value = float(sums[i]) * xs[row];
                if (p.flags & 1)
                    value *= ws[col];
                if (p.flags & 2)
                    value += bias[col];
                output[ulong(row) * p.n + col] = value;
            }
        }
    }
}
