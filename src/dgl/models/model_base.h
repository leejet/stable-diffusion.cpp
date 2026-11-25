#ifndef __SD_DGL_MODELS_MODEL_BASE_H__
#define __SD_DGL_MODELS_MODEL_BASE_H__

#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

#include "core/parallel.h"
#include "dgl/models/lora.h"

class BackendProbeCache;

namespace sd::dgl::models {

    class FamilyProbeCache;

    struct ModelExecutionOptions {
        std::shared_ptr<sd::ParallelExecutor> tensor_executor;
        bool enable_mmap               = false;
        bool eager_load                = false;
        bool disable_prefetch          = false;
        bool disable_segmented_compute = false;
        std::shared_ptr<BackendProbeCache> backend_probes;
        std::shared_ptr<FamilyProbeCache> family_probes;
        bool defer_initialization = false;
    };

    class Model {
        friend class ModelRuntime;

        struct RunnerRegistration {
            std::function<void()> end;
            std::function<void(const std::shared_ptr<WeightAdapter>&)> set_adapter;
        };

        std::vector<RunnerRegistration> runners_;
        std::recursive_mutex execution_mutex_;
        std::shared_ptr<const LoraConfig> loras_;
        LoraExecution lora_execution_;
        bool executing_        = false;
        bool weights_released_ = false;
        bool eager_load_       = false;
        ModelLoader::Dependencies family_dependencies_;
        std::shared_ptr<BackendProbeCache> backend_probes_;
        std::shared_ptr<FamilyProbeCache> family_probes_;
        bool defer_initialization_   = false;
        bool initialization_pending_ = false;
        bool eager_load_supported_   = true;

        void end_runners();

    protected:
        std::function<bool(const std::string&)> params_placement_;
        explicit Model(std::shared_ptr<ModelManager> manager);
        virtual ModelLoader base_loader() const;
        virtual bool build_variant(const Model& base, ModelLoader loader) = 0;
        virtual bool attach_loras()                                       = 0;
        virtual void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter);
        virtual bool refresh_weights() { return true; }
        virtual bool finish_loading() { return true; }
        virtual void finish_execution() {}
        virtual void invalidate_cached_state() {}

        bool has_loras() const;
        lora_apply_mode_t lora_mode() const;
        bool apply_loras();
        bool prepare_loras(SDVersion version, SDBackendManager& backends, SDBackendModule module, int n_threads, bool immediately);
        void reset_loras();
        bool load_weights_eagerly() const;
        // Release the resource while registered tensors and their backends are alive.
        void release_weights();

    public:
        const std::shared_ptr<ModelManager> model_manager;
        const std::shared_ptr<ModelManager::ResourceHandle> model_weights;
        int n_threads = -1;
        std::shared_ptr<sd::ParallelExecutor> tensor_executor;

        class ExecutionScope {
            Model& model_;
            std::unique_lock<std::recursive_mutex> lock_;
            bool ready_    = false;
            bool acquired_ = false;
            std::optional<sd::ParallelScope> tensor_scope_;

        public:
            explicit ExecutionScope(Model& model, bool use_tensor_executor = true);
            ~ExecutionScope();
            void require_ready() const;
        };

        Model(const Model&)            = delete;
        Model& operator=(const Model&) = delete;
        virtual ~Model();
        void configure_execution(const ModelExecutionOptions& options);
        bool finish_initialization();
        bool place_params(const std::string& name, ggml_backend_t backend, bool disk);
        void set_shared_weights(std::string key) { model_weights->set_shared_weights(std::move(key)); }
        bool init_variant(const Model& base, std::shared_ptr<const LoraConfig> loras);
        ModelLoader::Dependencies dependencies() const;
    };

}  // namespace sd::dgl::models

#endif  // __SD_DGL_MODELS_MODEL_BASE_H__
