#ifndef __SD_DGL_MODELS_MODEL_RUNTIME_H__
#define __SD_DGL_MODELS_MODEL_RUNTIME_H__

#include <functional>
#include <optional>
#include <set>

#include "core/ggml_extend_backend.h"
#include "dgl/models/model_base.h"
#include "dgl/models/model_load_params.h"
#include "model_build_context.h"

struct GGMLRunner;
enum class BackendProbeOperation;

namespace sd::dgl::models {

    class ModelRuntime {
        using LoadCatalog = std::function<bool(ModelLoader&, const ModelLoadParams&, SDVersion&, FamilyProbeCache&)>;

        Model& model_;
        const std::function<bool()> build_runners_;
        std::set<ModelComponent> registered_components_;
        std::string backend_spec_;
        std::string params_backend_spec_;
        std::string split_mode_;
        bool ready_                  = false;
        bool apply_lora_immediately_ = false;
        uint64_t catalog_revision_   = 0;

        bool rebuild();
        void configure_weight_loading();
        bool configure_backends();
        bool split_active(SDSplitMode mode);
        bool supports_operation(BackendProbeOperation operation);

        template <typename T>
        bool register_row_split_runner_params(ModelComponent component, const std::shared_ptr<T>& runner, const std::vector<ggml_backend_t>& backends, std::map<std::string, ggml_tensor*> tensors, const std::map<ggml_tensor*, enum ggml_op>& ops, ModelManager::ResidencyMode mode, size_t* size);

        template <typename T>
        bool register_layer_split_runner_params(ModelComponent component, const std::shared_ptr<T>& runner, const std::vector<ggml_backend_t>& backends, std::map<std::string, ggml_tensor*> tensors, const std::map<ggml_tensor*, enum ggml_op>& ops, ModelManager::ResidencyMode mode, size_t* size);

    public:
        const ModelComponent component;
        const SDBackendModule module;
        SDVersion version = VERSION_COUNT;
        ModelLoadParams params;
        SDBackendManager backend_manager;

        ModelRuntime(Model& model, std::shared_ptr<SDBackendPool> backends, ModelComponent component, SDBackendModule module, std::function<bool()> build_runners);
        ~ModelRuntime();

        bool init(ModelLoadParams params, const LoadCatalog& load_catalog);
        bool build_variant(const ModelRuntime& base, ModelLoader loader);
        bool refresh_weights();
        bool attach_loras();
        bool set_params_placement(const std::string& name);

        sd::model_builders::Context model_build_context();
        bool flash_attention_for();
        bool conv_direct_for(bool conv3d);
        void configure_runner(GGMLRunner& runner, bool conv_direct = false);
        bool ensure_backend_pair();
        ggml_backend_t backend_for();
        ggml_backend_t params_backend_for();

        template <typename T>
        bool register_runner_params(ModelComponent component, const std::shared_ptr<T>& runner, size_t* size = nullptr);
    };

}  // namespace sd::dgl::models

#endif  // __SD_DGL_MODELS_MODEL_RUNTIME_H__
