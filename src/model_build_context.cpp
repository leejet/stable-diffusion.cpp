#include "model_build_context.h"

#include "core/util.h"

namespace sd::model_builders {
    bool ensure_model_backend_pair(SDBackendManager& backends, SDBackendModule module) {
        if (backends.runtime_backend(module) == nullptr) {
            LOG_ERROR("failed to initialize %s backend", sd_backend_module_name(module));
            return false;
        }
        if (backends.params_backend(module) == nullptr) {
            LOG_ERROR("failed to initialize %s params backend", sd_backend_module_name(module));
            return false;
        }
        return true;
    }
}  // namespace sd::model_builders
