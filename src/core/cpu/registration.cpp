#include "core/cpu/registration.h"

#ifndef SD_USE_UPSTREAM_GGML
#include <cstring>
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
namespace sd_cpu {

    ggml_status parallel(ggml_backend_t backend, ggml_tensor* output, ggml_custom_op_t callback, void* user_data) {
        if (!ggml_backend_is_cpu(backend) || !output || !callback) {
            return GGML_STATUS_FAILED;
        }
        ggml_tensor node             = *output;
        ggml_custom_op_params params = {callback, GGML_N_TASKS_MAX, user_data};
        node.op                      = GGML_OP_CUSTOM;
        node.flags |= GGML_TENSOR_FLAG_COMPUTE;
        memcpy(node.op_params, &params, sizeof(params));
        ggml_tensor* nodes[] = {&node};
        ggml_cgraph graph    = {};
        graph.n_nodes        = 1;
        graph.nodes          = nodes;
        // Reuse the configured thread pool without dispatching this synthetic node through extensions.
        return backend->iface.graph_compute(backend, &graph);
    }

    void register_extensions(ggml_backend_dev_t device) {
        register_i8(device);
        register_rope(device);
    }

}
#endif
