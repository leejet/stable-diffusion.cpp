#include <cutlass/bfloat16.h>
#include <cutlass/gemm/device/gemm.h>
#include "bf16_gemm.h"

SDBF16GemmResult sd_cuda_bf16_gemm(const void* weight, const void* input, float* output, int64_t rows, int64_t n, int64_t k, cudaStream_t stream) {
#ifdef SD_CUDA_FORCE_CUBLAS
    return SDBF16GemmResult::Unsupported;
#else
    if (rows < 512 || rows > 2048 || (n < 4096 && (n < 2048 || k < 2 * n)) || k % 8 || n % 4 ||
        (reinterpret_cast<uintptr_t>(weight) | reinterpret_cast<uintptr_t>(input) | reinterpret_cast<uintptr_t>(output)) % 16)
        return SDBF16GemmResult::Unsupported;
    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::bfloat16_t, cutlass::layout::RowMajor, cutlass::bfloat16_t, cutlass::layout::ColumnMajor,
        float, cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<64, 128, 32>, cutlass::gemm::GemmShape<32, 64, 32>, cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<float, 4, float, float>, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;
    Gemm gemm;
    const Gemm::Arguments args({int(rows), int(n), int(k)},
                               {static_cast<const cutlass::bfloat16_t*>(input), k},
                               {static_cast<const cutlass::bfloat16_t*>(weight), k}, {output, n}, {output, n}, {1.f, 0.f});
    if (gemm.can_implement(args) != cutlass::Status::kSuccess)
        return SDBF16GemmResult::Unsupported;
    return gemm(args, nullptr, stream) == cutlass::Status::kSuccess ? SDBF16GemmResult::Success : SDBF16GemmResult::Failed;
#endif
}
