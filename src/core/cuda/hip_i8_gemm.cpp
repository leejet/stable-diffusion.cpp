#include "hip_i8_gemm.h"

#include <hipblaslt/hipblaslt.h>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>
#include <list>
#include <map>
#include <mutex>
#include <tuple>

#include "core/util.h"

namespace sd_hip {
    using GemmShape = std::tuple<int, int, int>;

    struct HipLtPlan {
        GemmShape shape;
        hipblasLtMatmulDesc_t op   = nullptr;
        hipblasLtMatrixLayout_t a = nullptr;
        hipblasLtMatrixLayout_t b = nullptr;
        hipblasLtMatrixLayout_t c = nullptr;
        hipblasLtMatmulAlgo_t algo{};
        bool use_lt = false;

        explicit HipLtPlan(GemmShape shape)
            : shape(shape) {}

        ~HipLtPlan() {
            if (c)
                hipblasLtMatrixLayoutDestroy(c);
            if (b)
                hipblasLtMatrixLayoutDestroy(b);
            if (a)
                hipblasLtMatrixLayoutDestroy(a);
            if (op)
                hipblasLtMatmulDescDestroy(op);
        }

        bool init(const HipI8Gemm& gemm) {
            const hipblasOperation_t transpose = HIPBLAS_OP_T;
            return hipblasLtMatmulDescCreate(&op, HIPBLAS_COMPUTE_32I, HIP_R_32I) == HIPBLAS_STATUS_SUCCESS &&
                   hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose)) == HIPBLAS_STATUS_SUCCESS &&
                   hipblasLtMatrixLayoutCreate(&a, HIP_R_8I, gemm.k, gemm.n, gemm.k) == HIPBLAS_STATUS_SUCCESS &&
                   hipblasLtMatrixLayoutCreate(&b, HIP_R_8I, gemm.k, gemm.rows, gemm.k) == HIPBLAS_STATUS_SUCCESS &&
                   hipblasLtMatrixLayoutCreate(&c, HIP_R_32I, gemm.n, gemm.rows, gemm.n) == HIPBLAS_STATUS_SUCCESS;
        }

        hipblasStatus_t run(hipblasLtHandle_t handle, const HipI8Gemm& gemm, const hipblasLtMatmulAlgo_t& choice) const {
            const int32_t alpha = 1, beta = 0;
            return hipblasLtMatmul(handle, op, &alpha, gemm.weight, a, gemm.input, b,
                                   &beta, gemm.output, c, gemm.output, c, &choice, nullptr, 0, gemm.stream);
        }
    };

    struct HipLtDevice {
        int ordinal;
        std::mutex mutex;
        std::map<hipStream_t, hipblasLtHandle_t> handles;
        std::list<HipLtPlan> plans;

        explicit HipLtDevice(int ordinal)
            : ordinal(ordinal) {}

        ~HipLtDevice() {
            if (handles.empty())
                return;
            int previous;
            if (hipGetDevice(&previous) == hipSuccess && hipSetDevice(ordinal) == hipSuccess) {
                for (const auto& entry : handles)
                    (void)hipblasLtDestroy(entry.second);
                (void)hipSetDevice(previous);
            }
        }
    };

    struct HipGemmTimer {
        hipEvent_t start = nullptr;
        hipEvent_t end   = nullptr;

        ~HipGemmTimer() {
            if (end)
                (void)hipEventDestroy(end);
            if (start)
                (void)hipEventDestroy(start);
        }

        template <typename Launch>
        float measure(hipStream_t stream, Launch launch) const {
            constexpr int trials = 3;
            if (hipEventRecord(start, stream) != hipSuccess)
                return 0.f;
            for (int i = 0; i < trials; ++i) {
                if (launch() != HIPBLAS_STATUS_SUCCESS)
                    return 0.f;
            }
            float ms;
            return hipEventRecord(end, stream) == hipSuccess && hipEventSynchronize(end) == hipSuccess &&
                           hipEventElapsedTime(&ms, start, end) == hipSuccess
                       ? ms / trials
                       : 0.f;
        }
    };

    static void tune(int device, hipblasLtHandle_t handle, const HipI8Gemm& gemm, HipLtPlan& plan) {
        if (!plan.init(gemm))
            return;

        hipblasLtMatmulPreference_t preference = nullptr;
        if (hipblasLtMatmulPreferenceCreate(&preference) != HIPBLAS_STATUS_SUCCESS)
            return;
        // A zero-workspace kernel can use the existing ggml accumulator buffer.
        constexpr uint64_t workspace = 0;
        std::array<hipblasLtMatmulHeuristicResult_t, 8> choices{};
        int count = 0;
        if (hipblasLtMatmulPreferenceSetAttribute(preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace, sizeof(workspace)) == HIPBLAS_STATUS_SUCCESS) {
            const auto status = hipblasLtMatmulAlgoGetHeuristic(handle, plan.op, plan.a, plan.b, plan.c, plan.c,
                                                                preference, int(choices.size()), choices.data(), &count);
            if (status != HIPBLAS_STATUS_SUCCESS)
                count = 0;
        }
        hipblasLtMatmulPreferenceDestroy(preference);
        if (count == 0)
            return;

        HipGemmTimer timer;
        if (hipEventCreate(&timer.start) != hipSuccess || hipEventCreate(&timer.end) != hipSuccess)
            return;
        const auto blas_status = gemm.run_hipblas();
        const bool blas_unsupported = blas_status == HIPBLAS_STATUS_NOT_SUPPORTED || blas_status == HIPBLAS_STATUS_ARCH_MISMATCH;
        float baseline = 0.f;
        if (blas_status == HIPBLAS_STATUS_SUCCESS) {
            baseline = timer.measure(gemm.stream, [&] { return gemm.run_hipblas(); });
            if (baseline <= 0.f)
                return;
        } else if (!blas_unsupported) {
            return;
        }

        float best = blas_unsupported ? std::numeric_limits<float>::infinity() : baseline * 0.95f;
        for (int i = 0; i < count; ++i) {
            const auto& candidate = choices[i];
            if (candidate.state != HIPBLAS_STATUS_SUCCESS || candidate.workspaceSize != 0 ||
                plan.run(handle, gemm, candidate.algo) != HIPBLAS_STATUS_SUCCESS)
                continue;
            const float ms = timer.measure(gemm.stream, [&] { return plan.run(handle, gemm, candidate.algo); });
            if (ms > 0.f && ms < best) {
                best        = ms;
                plan.algo   = candidate.algo;
                plan.use_lt = true;
            }
        }
        if (plan.use_lt) {
            if (blas_unsupported) {
                LOG_VERBOSE("hipBLASLt INT8 GEMM %d x %d x %d on HIP device %d: %.3f ms (hipBLAS unsupported)",
                            gemm.n, gemm.rows, gemm.k, device, best);
            } else {
                LOG_VERBOSE("hipBLASLt INT8 GEMM %d x %d x %d on HIP device %d: %.3f -> %.3f ms",
                            gemm.n, gemm.rows, gemm.k, device, baseline, best);
            }
        }
    }

    static HipLtDevice& lt_device(int ordinal) {
        static std::mutex mutex;
        static std::map<int, HipLtDevice> devices;
        std::lock_guard<std::mutex> lock(mutex);
        return devices.try_emplace(ordinal, ordinal).first->second;
    }

    bool hip_i8_gemm_lt(int device, const HipI8Gemm& gemm) {
        static const bool disabled = std::getenv("SD_HIP_DISABLE_I8_LT") != nullptr;
        if (disabled || gemm.n < 1024 || gemm.rows < 1024 || gemm.k < 1024 || gemm.k > INT32_MAX / 16384 ||
            (gemm.n | gemm.k) % 16 != 0 ||
            (reinterpret_cast<uintptr_t>(gemm.weight) | reinterpret_cast<uintptr_t>(gemm.input) |
             reinterpret_cast<uintptr_t>(gemm.output)) % 16 != 0) {
            return false;
        }
        auto& state = lt_device(device);
        std::lock_guard<std::mutex> lock(state.mutex);
        const GemmShape shape(gemm.n, gemm.rows, gemm.k);
        auto plan = std::find_if(state.plans.begin(), state.plans.end(), [&](const auto& entry) { return entry.shape == shape; });
        const bool needs_tuning = plan == state.plans.end();
        if (!needs_tuning) {
            state.plans.splice(state.plans.begin(), state.plans, plan);
            if (!plan->use_lt)
                return false;
        }

        const auto stream_handle = state.handles.find(gemm.stream);
        hipblasLtHandle_t handle = stream_handle == state.handles.end() ? nullptr : stream_handle->second;
        if (needs_tuning || !handle) {
            // Handle creation and tuning must run outside graph capture.
            hipStreamCaptureStatus capture;
            if (hipStreamIsCapturing(gemm.stream, &capture) != hipSuccess || capture != hipStreamCaptureStatusNone)
                return false;
        }
        if (!handle) {
            // A host lock does not wait for work submitted on another stream.
            if (hipblasLtCreate(&handle) != HIPBLAS_STATUS_SUCCESS)
                return false;
            state.handles.emplace(gemm.stream, handle);
        }

        if (needs_tuning) {
            state.plans.emplace_front(shape);
            plan = state.plans.begin();
            tune(device, handle, gemm, *plan);
            if (state.plans.size() > 64)
                state.plans.pop_back();
            if (!plan->use_lt)
                return false;
        }
        if (plan->run(handle, gemm, plan->algo) == HIPBLAS_STATUS_SUCCESS)
            return true;
        plan->use_lt = false;
        return false;
    }
}
