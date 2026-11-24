#ifndef __SD_DGL_CONTEXT_H__
#define __SD_DGL_CONTEXT_H__

#include <mutex>

#include "core/file_access.h"
#include "core/ggml_extend_backend.h"
#include "dgl/models/model_base.h"
#include "model_manager.h"
#include "registry.h"
#include "run_events.h"

namespace sd::dgl {
    struct Context;

    struct ResourceLoadRequest {
        Context& context;
        std::string weights_key;
    };

    std::string file_key(const std::string& path, nlohmann::json& revision);

    struct Context {
        using ResourceKey = std::pair<std::string, std::string>;  // Component and LoRA configuration.

        struct Resource {
            std::string revision;
            std::string lora_revision;
            Value::Cached value;
            ModelLoader::Dependencies dependencies;
            uint64_t last_use = 0;
        };

        std::mutex mutex;
        RunEvents* active_run = nullptr;
        bool generating       = false;
        RunCallbacks callbacks;
        FileAccessPolicy file_access;
        std::shared_ptr<SDBackendPool> backends = std::make_shared<SDBackendPool>();
        std::shared_ptr<ModelManager> models    = std::make_shared<ModelManager>();
        std::map<ResourceKey, Resource> resources;
        uint64_t resource_epoch = 0;
        const bool keep_resources;
        bool batched_cfg    = true;
        bool embed_metadata = true;

        explicit Context(bool keep_resources = true);
        void set(const std::string& key, const std::string& value);
        Value load(const Registry::Entry& registration, Values& args, bool prepare_only = false);
        void finish_resource_loading(const Value& value) const;
        Value derive_model(const Value& base, std::shared_ptr<const models::LoraConfig> loras);
        bool evict_idle_resource();
        void prune_expired_resources();
        void unload_models();

        sd::ParallelExecutor* tensor_executor() const { return model_execution_.tensor_executor.get(); }
        bool auto_fit() const { return auto_fit_; }

        template <class T>
        std::shared_ptr<T> make_model(const std::string& weights_key) const {
            auto model = std::make_shared<T>(models, backends);
            model->set_shared_weights(weights_key);
            model->configure_execution(model_execution_);
            return model;
        }

    private:
        std::optional<Value> find_resource(const ResourceKey& identity, const std::string& revision, const std::string& lora_revision, uint64_t last_use);
        Value cache_resource(ResourceKey identity, std::string revision, std::string lora_revision, ModelLoader::Dependencies dependencies, Value value, uint64_t last_use);
        bool auto_fit_ = true;
        models::ModelExecutionOptions model_execution_;
        size_t conditioning_cache_size_;
        void set_model_execution_option(const std::string& key, const std::string& value);
    };

}  // namespace sd::dgl

struct sd_dgl_ctx_t : sd::dgl::Context {
    using Context::Context;
};

#endif  // __SD_DGL_CONTEXT_H__
