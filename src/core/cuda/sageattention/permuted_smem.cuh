/*
 * Adapted from Flashinfer, https://github.com/flashinfer-ai/flashinfer/blob/v0.1.5/include/flashinfer/permuted_smem.cuh
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Modifications copyright (c) 2024 by SageAttention team.
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

#ifndef SD_CORE_CUDA_SAGEATTENTION_PERMUTED_SMEM_CUH
#define SD_CORE_CUDA_SAGEATTENTION_PERMUTED_SMEM_CUH
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>


#include "cp_async.cuh"
#include "mma.cuh"

namespace sageattention {

enum class SwizzleMode {
  k32B, // for k32B mode, a line of shared memory must have 32B (16 half value)
  k64B, // for k64B mode, a line of shared memory must have 64B (32 half value)
  k128B, // 128B already spans all banks in shared memory. a line of shared memory can have multiple 128B.
};

// Use 128bit as the granularity to fetch/store data per thread to maximize memory bandwidth
using b128_t = uint4;

/*!
 * \brief A stateless shared memory wrapper that uses templates to avoid runtime conditionals. It makes sure
 * that access to consecutive rows idx in the same column idx will make full use of the shared memory bank through
 * permutation in the granularity of 128bit.
 *
 * This struct treats all offsets to be the number of `b128_t` elements. It is designed to be stateless,
 * meaning it does not maintain any information about the current pointer position. The offset returnd by
 * the struct can be used to access the shared memory through the provided interface.
 *
 * The struct guarantees that the read to permuted offset (i, j) will be the value stored in permuted offset (i, j).
 * We assume that shared memory operation operates on at least two consecutive 128-bit values in a row within a warp.
 * Under this assumption, we do not permute for k32B mode.
 */
template <SwizzleMode swizzle_mode, uint32_t stride>
struct smem_t {
  // The base pointer.
  b128_t* base;
  // How many b128_t value a row contains
  // uint32_t stride;

  __device__ __forceinline__ smem_t() : base(nullptr) {}
  template <typename T>
  __device__ __forceinline__ smem_t(T* base) : base((b128_t*)base) {
    if constexpr (swizzle_mode == SwizzleMode::k128B) {
      static_assert(stride % 8 == 0, "Stride must be multiple of 8 for 128B swizzle mode");
    } else if constexpr (swizzle_mode == SwizzleMode::k64B) {
      static_assert(stride == 4, "Stride must be 4 for 64B swizzle mode");
    } else if constexpr (swizzle_mode == SwizzleMode::k32B) {
      static_assert(stride == 2, "Stride must be 2 for 32B swizzle mode");
    } else {
      static_assert(swizzle_mode != swizzle_mode, "Unsupported swizzle mode");
    }
  }

  /*!
   * \brief Set the base pointer.
   */
  template <typename T>
  __device__ __forceinline__ void set_base(T* new_base) {
    base = (b128_t*)new_base;
  }

  /*!
   * \brief Compute the element offset given coordinates in a permuted shared memory.
   * \param i The row index.
   * \param j The column index.
   */
  static __device__ __forceinline__ uint32_t get_permuted_offset(const uint32_t &i, const uint32_t &j) {
    if constexpr (swizzle_mode == SwizzleMode::k128B) {
      return i * stride + (j ^ (i % 8));
    } else if constexpr (swizzle_mode == SwizzleMode::k64B) {
      return i * stride + (j ^ ((i / 2) % 4));
    } else if constexpr (swizzle_mode == SwizzleMode::k32B) {
      return i * stride + j;
    }
  }

  /*!
  * \tparam step_size The step size to advance the offset in the permuted shared memory.
  * \param offset The current offset.
  */
  template <uint32_t step_size>
  static __device__ __forceinline__ uint32_t advance_offset_by_column(const uint32_t &offset) {
    if constexpr (swizzle_mode == SwizzleMode::k128B) {
      static_assert(step_size % 8 == 0,
                    "Unsupported step size");
      return offset + step_size;
    } else if constexpr (swizzle_mode == SwizzleMode::k64B) {
      static_assert(step_size == 4, "Unsupported step size");
      return offset + step_size;
    } else if constexpr (swizzle_mode == SwizzleMode::k32B) {
      static_assert(step_size == 2, "Unsupported step size");
      return offset + step_size;
    }
  }

  // ! use with care
  template <uint32_t step_size>
  static __device__ __forceinline__ uint32_t advance_offset_by_column(const uint32_t &offset, const uint32_t &step_idx) {
    if constexpr (swizzle_mode == SwizzleMode::k128B) {
      static_assert(step_size == 2 || step_size == 4 || step_size % 8 == 0,
                    "Unsupported step size");
      if constexpr (step_size == 2) {
        return (offset ^ (0x2 + (0x4 * (step_idx % 2 == 1)))) + (step_idx % 4 == 3) * 8;
      } else if constexpr (step_size == 4) {
        return (offset ^ 0x4) + (step_idx % 2 == 1) * 8;
      } else {
        // step_size % 8 == 0
        return offset + step_size;
      }
    } else if constexpr (swizzle_mode == SwizzleMode::k64B) {
      static_assert(step_size == 2 || step_size == 4, "Unsupported step size");
      if constexpr (step_size == 2) {
        return (offset ^ 0x2) + (step_idx % 2 == 1) * 4;
      } else {
        return offset + step_size;
      }
    } else if constexpr (swizzle_mode == SwizzleMode::k32B) {
      return offset + step_size;
    }
  }

  template <uint32_t step_size>
  static __device__ __forceinline__ uint32_t advance_offset_by_row(const uint32_t &offset) {
    if constexpr (swizzle_mode == SwizzleMode::k128B) {
      static_assert(step_size == 4 || step_size % 8 == 0, "Unsupported step size");
      if constexpr (step_size == 4) {
        return (offset ^ 0x4) + step_size * stride;
      } else {
        // step_size % 8 == 0
        return offset + step_size * stride;
      }
    } else if constexpr (swizzle_mode == SwizzleMode::k64B) {
      static_assert(step_size == 4 || step_size % 8 == 0, "Unsupported step size");
      if constexpr (step_size == 4) {
        return (offset ^ 0x2) + step_size * stride;
      } else {
        // step_size % 8 == 0
        return offset + step_size * stride;
      }
    } else if constexpr (swizzle_mode == SwizzleMode::k32B) {
      return offset + step_size * stride;
    }
  }

  __device__ __forceinline__ void ldmatrix_m8n8x2(const uint32_t &offset, uint32_t* R) const {
    b128_t* smem_ptr = base + offset;
    mma::ldmatrix_m8n8x2(R, smem_ptr);
  }

  __device__ __forceinline__ void ldmatrix_m8n8x4(const uint32_t &offset, uint32_t* R) const {
    b128_t* smem_ptr = base + offset;
    mma::ldmatrix_m8n8x4(R, smem_ptr);
  }

  __device__ __forceinline__ void ldmatrix_m8n8x4_trans(const uint32_t &offset, uint32_t* R) const {
    b128_t* smem_ptr = base + offset;
    mma::ldmatrix_m8n8x4_trans(R, smem_ptr);
  }

  template <cp_async::SharedMemFillMode fill_mode, typename T>
  __device__ __forceinline__ void load_128b_async(const uint32_t &offset, const T* gptr, bool predicate) const {
    b128_t* smem_ptr = base + offset;
    cp_async::pred_load_128b<cp_async::PrefetchMode::kPrefetch, fill_mode>(
        smem_ptr, reinterpret_cast<const b128_t*>(gptr), predicate);
  }

  template <typename T>
  __device__ __forceinline__ void load_128b_async(const uint32_t &offset, const T* gptr) const {
    b128_t* smem_ptr = base + offset;
    cp_async::load_128b<cp_async::PrefetchMode::kPrefetch>(smem_ptr, reinterpret_cast<const b128_t*>(gptr));
  }

  template <typename T>
  __device__ __forceinline__ void store_128b(const uint32_t &offset, T* gptr) const {
    *reinterpret_cast<b128_t*>(gptr) = *(base + offset);
  }
};

} // namespace sageattention

#endif
