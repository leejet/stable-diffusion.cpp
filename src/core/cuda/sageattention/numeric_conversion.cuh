/*
 * Copyright (c) 2024 by SageAttention team.
 *
 * Inspired by CUTLASS, https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/numeric_conversion.h
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SD_CORE_CUDA_SAGEATTENTION_NUMERIC_CONVERSION_CUH
#define SD_CORE_CUDA_SAGEATTENTION_NUMERIC_CONVERSION_CUH
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace sageattention {

#if (__CUDACC_VER_MAJOR__ * 10000 + __CUDACC_VER_MINOR__ * 100 >= 120400)
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 890))
#define FP8_CAST_ENABLED
#endif
#endif

#if defined(__CUDA_ARCH__)
#define RUNTIME_ASSERT(x) __brkpt()
#else

#define RUNTIME_ASSERT(x) assert(0 && x)
#endif

    __device__ __forceinline__ void unpack_half2_from_uint32_to_float(float* dest, uint32_t source) {
        uint16_t h0 = source & 0xFFFF;
        uint16_t h1 = (source >> 16) & 0xFFFF;
        asm("cvt.f32.f16 %0, %1;" : "=f"(dest[0]) : "h"(h0));
        asm("cvt.f32.f16 %0, %1;" : "=f"(dest[1]) : "h"(h1));
    }

    __device__ __forceinline__ void floatx4_to_e4m3x4(uint32_t* dest, float* source0, float* source1) {
#ifdef FP8_CAST_ENABLED
        asm volatile(
            "{\n"
            ".reg .b16 lo;\n"
            ".reg .b16 hi;\n"
            "cvt.rn.satfinite.e4m3x2.f32   lo, %2, %1;\n"
            "cvt.rn.satfinite.e4m3x2.f32   hi, %4, %3;\n"
            "mov.b32 %0, {lo, hi};\n"
            "}"
            : "=r"(dest[0]) : "f"(source0[0]), "f"(source0[1]), "f"(source1[0]), "f"(source1[1]));
#else
        RUNTIME_ASSERT("Unsupported CUDA architecture for FP8 CAST instruction");
#endif
    }

    __device__ __forceinline__ void floatx4_to_e5m2x4(uint32_t* dest, float* source0, float* source1) {
#ifdef FP8_CAST_ENABLED
        asm volatile(
            "{\n"
            ".reg .b16 lo;\n"
            ".reg .b16 hi;\n"
            "cvt.rn.satfinite.e5m2x2.f32   lo, %2, %1;\n"
            "cvt.rn.satfinite.e5m2x2.f32   hi, %4, %3;\n"
            "mov.b32 %0, {lo, hi};\n"
            "}"
            : "=r"(dest[0]) : "f"(source0[0]), "f"(source1[1]), "f"(source1[0]), "f"(source1[1]));
#else
        RUNTIME_ASSERT("Unsupported CUDA architecture for FP8 CAST instruction");
#endif
    }

    __device__ __forceinline__ void halfx4_to_e4m3x4(uint32_t* dest, uint32_t* source0, uint32_t* source1) {
#ifdef FP8_CAST_ENABLED
        asm volatile(
            "{\n"
            ".reg .b16 lo;\n"
            ".reg .b16 hi;\n"
            "cvt.rn.satfinite.e4m3x2.f16x2   lo, %1;\n"
            "cvt.rn.satfinite.e4m3x2.f16x2   hi, %2;\n"
            "mov.b32 %0, {lo, hi};\n"
            "}"
            : "=r"(dest[0]) : "r"(source0[0]), "r"(source1[0]));
#else
        RUNTIME_ASSERT("Unsupported CUDA architecture for FP8 CAST instruction");
#endif
    }

    __device__ __forceinline__ void halfx4_to_e5m2x4(uint32_t* dest, uint32_t* source0, uint32_t* source1) {
#ifdef FP8_CAST_ENABLED
        asm volatile(
            "{\n"
            ".reg .b16 lo;\n"
            ".reg .b16 hi;\n"
            "cvt.rn.satfinite.e5m2x2.f16x2   lo, %1;\n"
            "cvt.rn.satfinite.e5m2x2.f16x2   hi, %2;\n"
            "mov.b32 %0, {lo, hi};\n"
            "}"
            : "=r"(dest[0]) : "r"(source0[0]), "r"(source1[0]));
#else
        RUNTIME_ASSERT("Unsupported CUDA architecture for FP8 CAST instruction");
#endif
    }

    __device__ __forceinline__ void e4m3x4_to_halfx4(uint32_t* dest0, uint32_t* dest1, uint32_t* source) {
#ifdef FP8_CAST_ENABLED
        asm volatile(
            "{\n"
            ".reg .b16 lo, hi;\n"
            "mov.b32 {lo, hi}, %2;\n"
            "cvt.rn.f16x2.e4m3x2 %0, lo;\n"
            "cvt.rn.f16x2.e4m3x2 %1, hi;\n"
            "}\n" : "=r"(dest0[0]), "=r"(dest1[0]) : "r"(source[0]));
#else
        RUNTIME_ASSERT("Unsupported CUDA architecture for FP8 CAST instruction");
#endif
    }

    __device__ __forceinline__ void e5m2x4_to_halfx4(uint32_t* dest0, uint32_t* dest1, uint32_t* source) {
#ifdef FP8_CAST_ENABLED
        asm volatile(
            "{\n"
            ".reg .b16 lo, hi;\n"
            "mov.b32 {lo, hi}, %2;\n"
            "cvt.rn.f16x2.e5m2x2 %0, lo;\n"
            "cvt.rn.f16x2.e5m2x2 %1, hi;\n"
            "}\n" : "=r"(dest0[0]), "=r"(dest1[0]) : "r"(source[0]));
#else
        RUNTIME_ASSERT("Unsupported CUDA architecture for FP8 CAST instruction");
#endif
    }

    __device__ __forceinline__ int8_t float_to_int8_rn(float x) {
        uint32_t dst;
        asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(dst) : "f"(x));
        return reinterpret_cast<const int8_t&>(dst);
    }

}  // namespace sageattention

#endif
