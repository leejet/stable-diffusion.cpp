#include "core/probe/conv_direct.h"

#include <exception>

#include "core/ggml_extend_backend.h"
#include "core/probe/backend_probe.h"
#include "core/util.h"

static void build_conv_direct_probe(BackendProbeGraph& graph, bool conv3d) {
    constexpr int channels = 64;
    if (conv3d) {
        auto input   = ggml_new_tensor_4d(graph.ctx, GGML_TYPE_F32, 16, 16, 4, channels);
        auto weight  = ggml_new_tensor_4d(graph.ctx, GGML_TYPE_F16, 3, 3, 3, channels * channels);
        graph.inputs = {input, weight};
        graph.output = ggml_conv_3d_direct(graph.ctx, weight, input, 1, 1, 1, 1, 1, 1, 1, 1, 1, channels, 1, channels);
    } else {
        auto input   = ggml_new_tensor_4d(graph.ctx, GGML_TYPE_F32, 32, 32, channels, 1);
        auto weight  = ggml_new_tensor_4d(graph.ctx, GGML_TYPE_F16, 3, 3, channels, channels);
        graph.inputs = {input, weight};
        graph.output = ggml_conv_2d_direct(graph.ctx, weight, input, 1, 1, 1, 1, 1, 1);
    }
}

bool sd_probe_conv_direct(ggml_backend_t backend, bool conv3d) {
    if (backend == nullptr) {
        return false;
    }
    const char* name = ggml_backend_name(backend);
    const int dims   = conv3d ? 3 : 2;
    try {
        BackendProbeGraph capability(backend);
        if (capability.ctx == nullptr) {
            LOG_INFO("conv%dd direct auto: %s -> off (probe graph unavailable)", dims, name);
            return false;
        }
        build_conv_direct_probe(capability, conv3d);
        if (!capability.supported()) {
            LOG_INFO("conv%dd direct auto: %s -> off (unsupported probe graph)", dims, name);
            return false;
        }
        if (sd_backend_supports_conv_implicit_gemm(backend, conv3d)) {
            LOG_INFO("conv%dd direct auto: %s -> on (implicit GEMM)", dims, name);
            return true;
        }
        LOG_INFO("conv%dd direct auto: %s -> off (no implicit GEMM path)", dims, name);
        return false;
    } catch (const std::exception& error) {
        LOG_WARN("conv%dd direct auto: %s -> off (probe failed: %s)", dims, name, error.what());
        return false;
    }
}
