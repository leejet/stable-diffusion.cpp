#include <metal_simdgroup_matrix>
#include <metal_stdlib>
using namespace metal;

struct QuantParams {
    uint k;
    uint rows;
    uint group;
    uint unused;
    ulong scale_offset;
};

kernel void sd_i8_rotate(device const float* input [[buffer(0)]],
                         device float* output [[buffer(1)]],
                         constant QuantParams& p [[buffer(2)]],
                         uint block [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float values[256];
    const ulong base = ulong(block) * p.group;
    values[tid]      = input[base + tid] * rsqrt(float(p.group));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1; stride < p.group; stride *= 4) {
        uint index = tid / (4 * stride) * (4 * stride) + tid % stride;
        float a = values[index], b = values[index + stride];
        float c = values[index + 2 * stride], d = values[index + 3 * stride];
        float value;
        switch ((tid / stride) % 4) {
            case 0:
                value = a + b + c - d;
                break;
            case 1:
                value = a + b - c + d;
                break;
            case 2:
                value = a - b + c + d;
                break;
            default:
                value = -a + b + c + d;
                break;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        values[tid] = value;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    output[base + tid] = values[tid];
}

kernel void sd_i8_quantize(device const float* input [[buffer(0)]],
                           device char* output [[buffer(1)]],
                           constant QuantParams& p [[buffer(2)]],
                           uint row [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float maxima[8];
    const ulong base = ulong(row) * p.k;
    float amax       = 0;
    for (uint i = tid; i < p.k; i += 256) {
        amax = max(amax, abs(input[base + i]));
    }
    amax = simd_max(amax);
    if (lane == 0)
        maxima[simd] = amax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    amax          = simd_max(lane < 8 ? maxima[lane] : 0.f);
    float scale   = amax / 127.f;
    float inverse = scale == 0 ? 0 : 1.f / scale;
    for (uint i = tid; i < p.k; i += 256) {
        output[base + i] = char(clamp(rint(input[base + i] * inverse), -127.f, 127.f));
    }
    if (tid == 0)
        ((device float*)(output + p.scale_offset))[row] = scale;
}

struct MatParams {
    uint k;
    uint n;
    uint rows;
    uint flags;
};

kernel void sd_i8_multiply(device const char* weight [[buffer(0)]],
                           device const char* input [[buffer(1)]],
                           device const float* xs [[buffer(2)]],
                           device const float* ws [[buffer(3)]],
                           device const float* bias [[buffer(4)]],
                           device float* output [[buffer(5)]],
                           constant MatParams& p [[buffer(6)]],
                           uint2 tile [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup half a[32 * 32];
    threadgroup half b[32 * 32];
    threadgroup float partial[32 * 32];
    uint sums[8]    = {};
    const uint row0 = tile.y * 32, col0 = tile.x * 32;
    const uint sr = (simd / 2) * 16, sc = (simd % 2) * 16;

    // Half represents every INT8 value exactly. Each 256-term dot product fits
    // in FP32's exact integer range; combine chunks as integers to avoid drift.
    for (uint chunk = 0; chunk < p.k; chunk += 256) {
        simdgroup_float8x8 acc[4];
        for (uint i = 0; i < 4; ++i)
            acc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
        for (uint kk = chunk; kk < min(chunk + 256, p.k); kk += 32) {
            for (uint index = tid; index < 1024; index += 128) {
                uint r = index / 32, k = index % 32;
                a[index] = row0 + r < p.rows && kk + k < p.k ? half(input[ulong(row0 + r) * p.k + kk + k]) : half(0);
                b[index] = col0 + r < p.n && kk + k < p.k ? half(weight[ulong(col0 + r) * p.k + kk + k]) : half(0);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint k = 0; k < 32; k += 8) {
                simdgroup_half8x8 ma[2], mb[2];
                for (uint i = 0; i < 2; ++i) {
                    simdgroup_load(ma[i], a + (sr + i * 8) * 32 + k, 32);
                    simdgroup_load(mb[i], b + (sc + i * 8) * 32 + k, 32, ulong2(0), true);
                }
                for (uint i = 0; i < 4; ++i) {
                    simdgroup_multiply_accumulate(acc[i], ma[i / 2], mb[i % 2], acc[i]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint i = 0; i < 4; ++i) {
            simdgroup_store(acc[i], partial + (sr + (i / 2) * 8) * 32 + sc + (i % 2) * 8, 32);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < 8; ++i)
            sums[i] += uint(int(partial[tid + 128 * i]));
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < 8; ++i) {
        uint index = tid + 128 * i, row = row0 + index / 32, col = col0 + index % 32;
        if (row < p.rows && col < p.n) {
            float value = float(as_type<int>(sums[i])) * xs[row];
            if (p.flags & 1)
                value *= ws[col];
            if (p.flags & 2)
                value += bias[col];
            output[ulong(row) * p.n + col] = value;
        }
    }
}
