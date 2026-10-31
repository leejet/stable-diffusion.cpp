/*
 * Copyright (c) 2024 by SageAttention team.
 *
 * This file is based on code from Flashinfer, https://github.com/flashinfer-ai/flashinfer/blob/v0.1.5/include/flashinfer/cp_async.cuh
 * Copyright (c) 2023 by FlashInfer team.
 * Small modifications made by SageAttention team, 2024 (e.g., renamed namespace).
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

#ifndef SD_CORE_CUDA_SAGEATTENTION_CP_ASYNC_CUH
#define SD_CORE_CUDA_SAGEATTENTION_CP_ASYNC_CUH
#include <cuda_runtime.h>

namespace sageattention {

namespace cp_async {

enum class SharedMemFillMode {
  kFillZero,  // Fill zero to shared memory when predicate is false
  kNoFill     // Do not fill zero to shared memory when predicate is false
};

enum class PrefetchMode {
  kNoPrefetch,  // Do not fetch additional data from global memory to L2
  kPrefetch     // Fetch additional data from global memory to L2
};

#if (__CUDACC_VER_MAJOR__ >= 11)
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800))
#define CP_ASYNC_ENABLED
#endif
#endif

/*!
 * \brief Wrapper of PTX cp.async.commit_group instruction, commit all prior uncommitted
 *   cp.async instructions to a group
 */
__device__ __forceinline__ void commit_group() {
#ifdef CP_ASYNC_ENABLED
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}

/*!
 * \brief Wrapper of PTX cp.async.wait_group instruction
 * \tparam n Wait till most recent n groups are committed
 */
template <size_t n>
__device__ __forceinline__ void wait_group() {
#ifdef CP_ASYNC_ENABLED
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
#endif
}

/*!
 * \brief Wrapper of PTX cp.async.cg.shared.global instruction, asynchronously copy data from
 *   global memory to shared memory
 * \tparam prefetch_mode Whether to fetch additional data from global memory to L2
 * \tparam T Data type
 * \param smem_ptr Pointer to shared memory
 * \param gmem_ptr Pointer to global memory
 */
template <PrefetchMode prefetch_mode, typename T>
__device__ __forceinline__ void load_128b(T* smem_ptr, const T* gmem_ptr) {
#ifdef CP_ASYNC_ENABLED
  uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  if constexpr (prefetch_mode == PrefetchMode::kPrefetch) {
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
                 "l"(gmem_ptr), "n"(16), "r"(16));
  } else {
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
                 "l"(gmem_ptr), "n"(16), "r"(16));
  }
#else
  *((uint4*)smem_ptr) = *((uint4*)gmem_ptr);
#endif
}

/*!
 * \brief Wrapper of PTX cp.async.cg.shared.global instruction, asynchronously copy data from
 *   global memory to shared memory with predicate.
 * \tparam prefetch_mode Whether to fetch additional data from global memory to L2
 * \tparam fill_mode Whether to fill zero to shared memory when predicate is false
 * \tparam T Data type
 * \param smem_ptr Pointer to shared memory
 * \param gmem_ptr Pointer to global memory
 * \param predicate Predicate value
 * \note fill zero is slower than not fill zero
 */
template <PrefetchMode prefetch_mode, SharedMemFillMode fill_mode, typename T>
__device__ __forceinline__ void pred_load_128b(T* smem_ptr, const T* gmem_ptr, bool predicate) {
#ifdef CP_ASYNC_ENABLED
  uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  if constexpr (fill_mode == SharedMemFillMode::kFillZero) {
    int src_in_bytes = predicate ? 16 : 0;
    if constexpr (prefetch_mode == PrefetchMode::kPrefetch) {
      asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
                   "l"(gmem_ptr), "n"(16), "r"(src_in_bytes));
    } else {
      asm volatile("cp.async.cg.shared.global [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
                   "l"(gmem_ptr), "n"(16), "r"(src_in_bytes));
    }
  } else {
    if constexpr (prefetch_mode == PrefetchMode::kPrefetch) {
      asm volatile(
          "{\n"
          " .reg .pred p;\n"
          " setp.ne.b32 p, %0, 0;\n"
          " @p cp.async.cg.shared.global.L2::128B [%1], [%2], %3;\n"
          "}\n" ::"r"((int)predicate),
          "r"(smem_int_ptr), "l"(gmem_ptr), "n"(16));
    } else {
      asm volatile(
          "{\n"
          " .reg .pred p;\n"
          " setp.ne.b32 p, %0, 0;\n"
          " @p cp.async.cg.shared.global [%1], [%2], %3;\n"
          "}\n" ::"r"((int)predicate),
          "r"(smem_int_ptr), "l"(gmem_ptr), "n"(16));
    }
  }
#else
  if (predicate) {
    *((uint4*)smem_ptr) = *((uint4*)gmem_ptr);
  } else {
    if constexpr (fill_mode == SharedMemFillMode::kFillZero) {
      *((uint4*)smem_ptr) = make_uint4(0, 0, 0, 0);
    }
  }
#endif
}

} // namespace cp_async

} // namespace sageattention

#endif
