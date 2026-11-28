#ifndef __MODEL_MANAGER_H__
#define __MODEL_MANAGER_H__

#include <cstdint>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "core/ggml_graph_cut.h"
#include "device_residency_manager.h"
#include "model_component.h"
#include "model_loader.h"

struct LoraModel;

class ModelManager : public DeviceResidencyManager {
public:
    enum class ResidencyMode {
        Disk,
        ParamBackend,
    };

    using ResourceId = uint64_t;

    struct ResourceMemory {
        size_t params_bytes   = 0;
        size_t largest_tensor = 0;
        std::vector<ggml_backend_t> compute_backends;
        ggml_backend_t params_backend = nullptr;
        ResidencyMode mode            = ResidencyMode::ParamBackend;
        bool split                    = false;
    };

    struct RAMBudget {
        size_t weights = 0;
        size_t total   = 0;
    };

    struct LoraSpec {
        std::string path;
        float multiplier   = 1.0f;
        bool is_high_noise = false;
        std::string tensor_name_prefix_filter;
        bool required               = false;
        ModelLoader::FileId file_id = 0;
        uint64_t file_revision      = 0;
        std::weak_ptr<LoraModel> prepared;
    };

private:
    static constexpr size_t MAX_RESIDENCY_BLOCK_BYTES = 1024ULL * 1024ULL * 1024ULL;

    struct Resource {
        ResourceId id = 0;
        ModelLoader model_loader_;
        std::map<ModelLoader::FileId, ModelLoader> model_sources_;
        std::map<ggml_backend_t, ggml_backend_buffer_type_t> split_buffer_types_;
        std::map<ggml_backend_buffer_type_t, std::vector<std::pair<ggml_backend_t, size_t>>> split_buffer_limits_;
        bool warned_split_lora_skip_ = false;
        std::vector<LoraSpec> loras_;
        std::set<std::string> lora_targets_;
        std::string shared_weights_key_;
        std::set<ModelLoader::FileId> lora_sources_;
        bool applying_loras_             = false;
        SDVersion lora_version_          = VERSION_COUNT;
        uint64_t current_lora_epoch_     = 0;
        int n_threads_                   = 0;
        bool enable_mmap_                = false;
        bool writable_mmap_              = false;
        bool segmented_compute_disabled_ = false;
        bool prefetch_disabled_          = false;
        bool layer_split_                = false;
    };

    Resource default_resource_;
    std::set<std::string> common_ignore_tensors_;
    std::map<ResourceId, Resource> resources_;
    ResourceId next_resource_id_ = 1;
    std::map<std::string, size_t> planned_next_use_;
    bool retain_weights_ = false;
    mutable sd::ggml_graph_cut::MaxVramAssignment max_vram_;
    size_t max_ram_bytes_ = 0;
    std::string host_memory_error_;
    std::function<bool()> evict_idle_resource_;

    struct TensorState {
        Resource* resource = nullptr;
        std::string name;
        ggml_tensor* tensor      = nullptr;
        ModelComponent component = ModelComponent::Count;
        std::vector<TensorStorage> sources;
        ModelLoader::FileId source_file = 0;
        SDVersion source_version        = VERSION_COUNT;

        ResidencyMode residency_mode                 = ResidencyMode::ParamBackend;
        ggml_backend_t compute_backend               = nullptr;
        ggml_backend_t params_backend                = nullptr;
        ggml_backend_buffer_type_t split_buffer_type = nullptr;
        bool params_follow_compute_backend           = false;
        bool metadata_validated                      = false;
        enum ggml_op usage_op                        = GGML_OP_NONE;
        TensorState* lora_int8_partner               = nullptr;
        bool is_int8_weight                          = false;

        int pin_count = 0;

        bool loaded_to_params_backend  = false;
        bool staged_to_compute_backend = false;
        uint64_t applied_lora_epoch    = UINT64_MAX;
        uint64_t last_use_epoch        = 0;
    };

    struct ParamsStorageBlock {
        ggml_backend_buffer_t buffer = nullptr;
        std::vector<MmapTensorStore> mmap_tensor_stores;
        std::vector<TensorState*> states;
        bool shared = false;
    };

    struct ComputeStagingBlock {
        ggml_backend_t compute_backend = nullptr;
        ggml_backend_buffer_t buffer   = nullptr;
        ggml_context* staging_ctx      = nullptr;
        std::vector<std::pair<TensorState*, ggml_tensor*>> staged_tensors;
    };

    struct PrefetchBlock {
        std::vector<TensorState*> states;
        ggml_backend_t compute_backend  = nullptr;
        ggml_backend_t transfer_backend = nullptr;
        ggml_backend_event_t event      = nullptr;
        bool submitted                  = false;
        std::vector<std::unique_ptr<ComputeStagingBlock>> staging_blocks;
    };

    struct RuntimeResidency {
        ggml_backend_t compute_backend = nullptr;
        size_t resident_bytes          = 0;
    };

    struct ResolvedTensorStates {
        std::vector<ggml_tensor*> tensors;
        std::vector<TensorState*> states;
    };

    // Auxiliary models retain their own catalogs and conversion settings.
    std::vector<std::unique_ptr<TensorState>> tensor_states_;
    std::map<const ggml_tensor*, TensorState*> tensor_states_by_tensor_;
    mutable std::list<ResolvedTensorStates> resolved_tensor_states_;
    std::vector<std::unique_ptr<ParamsStorageBlock>> params_storage_blocks_;
    std::vector<std::unique_ptr<ComputeStagingBlock>> compute_staging_blocks_;
    std::map<ggml_backend_buffer_type_t, std::vector<std::pair<ggml_backend_t, size_t>>> split_buffer_devices_;
    std::map<uintptr_t, std::unique_ptr<PrefetchBlock>> prefetch_blocks_;
    std::map<ggml_backend_t, ggml_backend_t> prefetch_backends_;
    std::map<std::pair<uintptr_t, ggml_backend_t>, RuntimeResidency> runtime_residencies_;
    std::unordered_set<uintptr_t> runtime_owners_;
    uint64_t residency_epoch_ = 0;

    void finish_compute_backend_usage(const std::vector<TensorState*>& states);
    void release_all();
    void invalidate_sources(const std::unordered_set<TensorState*>& states);

    ggml_backend_t prefetch_backend_for(ggml_backend_t compute_backend);
    bool populate_prefetch_block(PrefetchBlock& block, PrefetchBuffer* prefetch_buffer);
    void synchronize_prefetch_block(PrefetchBlock& block);
    void free_prefetch_block(PrefetchBlock& block);
    void protect_prefetched_states(std::unordered_set<TensorState*>& states) const;
    void clear_all_prefetched_params();
    void release_prefetch();

    bool resolve_required_tensor_states(const std::vector<ggml_tensor*>& tensors,
                                        std::vector<TensorState*>& required_states,
                                        ggml_backend_t compute_backend = nullptr) const;
    bool should_ignore(const TensorState& state) const;
    static const ModelLoader& source_loader(const Resource& resource, ModelLoader::FileId file);
    bool is_optional_missing_tensor(const std::string& name) const;
    bool validate_tensor(const TensorState& state) const;

    bool load_tensors_to_params_backend(const std::vector<TensorState*>& states);
    void reuse_shared_weights(std::vector<TensorState*>& need_load);
    void detach_shared_weight(TensorState& state);
    static bool has_lora_int8_pair(const TensorState& state);
    static bool needs_lora_staging(const TensorState& state);
    bool apply_loras_to_params(const std::vector<TensorState*>& states);
    bool mmap_params(const std::vector<TensorState*>& states,
                     std::vector<ParamsStorageBlock*>& created_storage_blocks);
    static std::vector<TensorStorage> find_tensor_sources(const TensorState& state, const String2TensorStorage& sources);
    bool load_tensor_parts(TensorState& state);
    bool can_mmap_storage(const TensorState& state) const;
    size_t host_weight_bytes() const;
    bool evict_host_weight_block(const std::unordered_set<TensorState*>& protected_states);
    bool ensure_host_capacity(size_t pending, const std::vector<TensorState*>& required_states, bool weight_allocation = true, size_t pending_device_commit = 0);
    bool alloc_params_buffers(const std::vector<TensorState*>& states,
                              const std::vector<TensorState*>& required_states,
                              std::vector<ParamsStorageBlock*>& created_storage_blocks);
    bool load_tensors(const std::vector<TensorState*>& states);
    bool stage_tensors_to_compute_backend(const std::vector<TensorState*>& states);
    size_t compute_backend_alloc_size(const std::vector<TensorState*>& states,
                                      bool missing_only) const;
    size_t compute_backend_resident_bytes(ggml_backend_t compute_backend) const;
    std::map<ggml_backend_t, size_t> split_devices_for(const std::vector<TensorState*>& states) const;
    using CapacityCheck = DeviceMemoryCapacity;
    CapacityCheck check_capacity(const DeviceMemoryRequest& request,
                                 const std::vector<TensorState*>& states,
                                 DeviceMemoryCheck check,
                                 bool log_details = false) const;

    ggml_backend_buffer_type_t params_buffer_type_for(const TensorState& state) const;
    ggml_backend_buffer_type_t split_buffer_type_for(const TensorState& state) const;
    void release_compute_staging_blocks(bool force                                            = false,
                                        const std::unordered_set<TensorState*>* target_states = nullptr);
    void release_params_storage_blocks(bool force                                            = false,
                                       const std::unordered_set<TensorState*>* target_states = nullptr);
    void free_compute_staging_block(ComputeStagingBlock& block);
    void free_params_storage_block(ParamsStorageBlock& block);
    void erase_params_storage_block(ParamsStorageBlock* block);
    bool apply_resource_loras(Resource& resource, const std::vector<TensorState*>& states);
    void reset_lora_applied_params(Resource& resource);
    bool unregister_tensor_states(const std::unordered_set<TensorState*>& states, size_t* size, bool keep_registration = false);
    size_t other_runtime_resident_bytes(uintptr_t owner_id,
                                        ggml_backend_t compute_backend) const;

    Resource& resource_at(ResourceId id) { return id == 0 ? default_resource_ : resources_.at(id); }
    bool set_loader(Resource& resource, ModelLoader loader);
    bool place_resource_params(Resource& resource, ggml_backend_t backend, ResidencyMode mode);
    ResourceMemory resource_memory(const Resource& resource) const;
    size_t next_weight_use(const TensorState& state) const;
    bool del_file(Resource& resource, ModelLoader::FileId file);
    ModelLoader::FileId register_model_source(Resource& resource, ModelLoader loader);
    bool unregister_model_source(Resource& resource, ModelLoader::FileId file);
    size_t registered_params_size(const Resource& resource, const std::set<ModelComponent>& components) const;
    void set_split_buffer_type(Resource& resource, ggml_backend_t backend, ggml_backend_buffer_type_t type, const std::vector<std::pair<ggml_backend_t, size_t>>& device_limits);
    bool prepare_lora_sources(Resource& resource, std::vector<LoraSpec>& loras);
    bool set_loras(Resource& resource, std::vector<LoraSpec> loras, SDVersion version, std::set<std::string> targets);
    std::set<std::string> tensor_names(const Resource& resource) const;
    bool register_param_tensors(Resource& resource, ModelComponent component, std::map<std::string, ggml_tensor*> tensors, ResidencyMode residency_mode, ggml_backend_t compute_backend, ggml_backend_t params_backend, size_t* size, bool allow_split_buffer, bool follow_compute, const std::map<ggml_tensor*, enum ggml_op>* tensor_ops, ModelLoader::FileId source_file, SDVersion source_version);
    bool unregister_param_tensors(Resource& resource, ModelComponent component, size_t* size);
    bool unregister_param_tensors(const std::vector<ggml_tensor*>& tensors, const Resource* resource);
    bool validate_param_tensors(const std::vector<ggml_tensor*>& tensors, const Resource* resource);
    bool validate_registered_tensors(const Resource& resource);
    bool load_all_params_eagerly(const Resource& resource);

public:
    ~ModelManager() override;

    class ResourceHandle;

    ResourceId create_resource();
    bool remove_resource(ResourceId id);
    void set_retain_weights(bool retain) { retain_weights_ = retain; }
    bool retains_idle_weights() const override { return retain_weights_; }
    void release_idle_gpu_staging();
    void set_max_vram(sd::ggml_graph_cut::MaxVramAssignment limit) { max_vram_ = std::move(limit); }
    bool set_max_ram(size_t bytes);
    const sd::ggml_graph_cut::MaxVramAssignment& max_vram() const { return max_vram_; }
    size_t max_vram_bytes(ggml_backend_t backend) const { return max_vram_.bytes_for_backend(backend); }
    size_t max_ram() const { return max_ram_bytes_; }
    void set_idle_resource_eviction(std::function<bool()> evict) { evict_idle_resource_ = std::move(evict); }
    const std::string& host_memory_error() const { return host_memory_error_; }
    void clear_host_memory_error() { host_memory_error_.clear(); }
    size_t available_vram_for_fit(ggml_backend_t backend, size_t free_bytes, size_t total_bytes) const;
    RAMBudget available_ram_for_fit() const;
    void set_weight_use_priorities(std::map<std::string, size_t> priorities) { planned_next_use_ = std::move(priorities); }

    const ModelLoader& loader() const { return default_resource_.model_loader_; }

    bool set_loader(ModelLoader loader);
    bool add_file(const std::string& path, const std::string& prefix = "", ModelLoader::FileId* id = nullptr, bool force = false);
    bool del_file(ModelLoader::FileId id);
    bool refresh_files();
    ModelLoader::FileId register_model_source(ModelLoader loader);
    bool unregister_model_source(ModelLoader::FileId file);
    ModelLoader::FileVersions source_versions(const std::set<ModelComponent>& components, const ModelLoader& loader) const;
    size_t registered_params_size(const std::set<ModelComponent>& components) const;

    void prepare_file_io() { default_resource_.model_loader_.process_model_files(default_resource_.enable_mmap_, default_resource_.writable_mmap_); }
    bool load_float_tensor(const std::string& name, std::vector<float>& data) {
        return default_resource_.model_loader_.load_float_tensor(name, data, default_resource_.n_threads_, default_resource_.enable_mmap_);
    }

    void set_n_threads(int n_threads) {
        default_resource_.n_threads_ = n_threads;
        default_resource_.model_loader_.set_n_threads(n_threads);
    }
    void set_segmented_compute_disabled(bool disabled) {
        default_resource_.segmented_compute_disabled_ = disabled;
    }
    void set_prefetch_disabled(bool disabled) { default_resource_.prefetch_disabled_ = disabled; }
    void set_enable_mmap(bool enable_mmap) { default_resource_.enable_mmap_ = enable_mmap; }
    void set_writable_mmap(bool writable_mmap) { default_resource_.writable_mmap_ = writable_mmap; }
    void set_common_ignore_tensors(std::set<std::string> ignore_tensors);
    bool prepare_lora_sources(std::vector<LoraSpec>& loras);
    bool set_loras(std::vector<LoraSpec> loras, SDVersion version, std::set<std::string> targets = {});
    void set_split_buffer_type(ggml_backend_t compute_backend, ggml_backend_buffer_type_t split_buft, const std::vector<std::pair<ggml_backend_t, size_t>>& device_limits);

    static bool tensor_shape_supports_split_buffer(const ggml_tensor* tensor);

    std::set<std::string> tensor_names() const;

    bool register_param_tensors(ModelComponent component,
                                std::map<std::string, ggml_tensor*> tensors,
                                ResidencyMode residency_mode,
                                ggml_backend_t compute_backend,
                                ggml_backend_t params_backend,
                                size_t* registered_tensor_size                         = nullptr,
                                bool allow_split_buffer                                = false,
                                bool params_follow_compute_backend                     = false,
                                const std::map<ggml_tensor*, enum ggml_op>* tensor_ops = nullptr,
                                ModelLoader::FileId source_file                        = 0,
                                SDVersion source_version                               = VERSION_COUNT);

    ggml_tensor* resolve_param_tensor(ggml_tensor* tensor) const override;
    bool unregister_param_tensors(const std::vector<ggml_tensor*>& tensors);

    bool unregister_param_tensors(ModelComponent component,
                                  size_t* registered_tensor_size = nullptr);

    template <typename Runner>
    bool register_runner_params(ModelComponent component,
                                Runner& runner,
                                ResidencyMode residency_mode,
                                ggml_backend_t compute_backend,
                                ggml_backend_t params_backend,
                                size_t* registered_tensor_size = nullptr) {
        std::map<std::string, ggml_tensor*> tensors;
        runner.get_param_tensors(tensors);
        return register_param_tensors(component,
                                      std::move(tensors),
                                      residency_mode,
                                      compute_backend,
                                      params_backend,
                                      registered_tensor_size);
    }

    template <typename Runner>
    bool register_runner_params(ModelComponent component,
                                Runner& runner,
                                const std::string& prefix,
                                ResidencyMode residency_mode,
                                ggml_backend_t compute_backend,
                                ggml_backend_t params_backend,
                                size_t* registered_tensor_size = nullptr) {
        std::map<std::string, ggml_tensor*> tensors;
        runner.get_param_tensors(tensors, prefix);
        return register_param_tensors(component,
                                      std::move(tensors),
                                      residency_mode,
                                      compute_backend,
                                      params_backend,
                                      registered_tensor_size);
    }

    bool validate_registered_tensors();
    bool validate_param_tensors(const std::vector<ggml_tensor*>& tensors);
    bool load_all_params_eagerly();

    bool assign_compute_backend(const std::vector<ggml_tensor*>& tensors,
                                ggml_backend_t compute_backend) override;
    bool prepare_params(const std::vector<ggml_tensor*>& tensors) override;
    bool ensure_host_memory(size_t pending, const std::vector<ggml_tensor*>& protected_params, size_t pending_device_commit = 0) override;
    void register_runtime_owner(uintptr_t owner_id) override;
    void remove_runtime_owner(uintptr_t owner_id) override;
    bool fits_compute_backend_capacity(const DeviceMemoryRequest& request,
                                       const std::vector<ggml_tensor*>& required_params,
                                       DeviceMemoryCheck check) const override;
    bool segmented_compute_enabled() const override { return !default_resource_.segmented_compute_disabled_; }
    bool prefetch_enabled() const override { return !default_resource_.prefetch_disabled_; }
    void release_compute_backend_params(const std::vector<ggml_tensor*>& tensors) override;
    void evict_compute_backend_params(const std::vector<ggml_tensor*>& tensors) override;
    WeightResidencyInfo inspect_compute_backend_params(
        const std::vector<ggml_tensor*>& tensors) const override;
    void update_runtime_residency(uintptr_t owner_id,
                                  ggml_backend_t compute_backend,
                                  size_t resident_bytes) override;
    bool ensure_compute_backend_capacity(
        const DeviceMemoryRequest& request,
        const std::vector<ggml_tensor*>& required_params,
        const std::vector<std::vector<ggml_tensor*>>& preferred_eviction_order,
        const std::vector<ggml_tensor*>& protected_params,
        std::string* device_memory_error) override;
    DeviceMemoryCapacity inspect_compute_backend_capacity(
        const DeviceMemoryRequest& request,
        const std::vector<ggml_tensor*>& params,
        DeviceMemoryCheck check = DeviceMemoryCheck::Inspect,
        bool log_details        = false) const override;
    std::vector<WeightResidencyBlock> evictable_compute_backend_blocks(
        ggml_backend_t backend,
        const std::vector<ggml_tensor*>& protected_params) const override;
    bool evict_compute_backend_block(const WeightResidencyBlock& block, PrefetchBuffer* prefetch_buffer = nullptr) override;
    WeightPrefetchResult prefetch_params(
        uintptr_t owner_id,
        const std::vector<ggml_tensor*>& tensors) override;
    WeightPrefetchResult prepare_prefetch_params(
        uintptr_t owner_id,
        const std::vector<ggml_tensor*>& tensors,
        PrefetchBuffer* prefetch_buffer = nullptr) override;
    void submit_prefetched_params(uintptr_t owner_id) override;
    bool activate_prefetched_params(uintptr_t owner_id,
                                    const std::vector<ggml_tensor*>& tensors) override;
    void clear_prefetched_params(uintptr_t owner_id) override;
};

// Runner policy is resource-local. All handles share the manager's engine,
// weight residency and memory budgets.
class ModelManager::ResourceHandle final : public DeviceResidencyManager {
    const std::shared_ptr<ModelManager> manager_;
    const ResourceId id_;

    Resource& resource() const { return manager_->resource_at(id_); }

public:
    explicit ResourceHandle(std::shared_ptr<ModelManager> manager)
        : manager_(std::move(manager)), id_(manager_->create_resource()) {}
    ResourceHandle(std::shared_ptr<ModelManager> manager, ResourceId id)
        : manager_(std::move(manager)), id_(id) {}

    bool remove() { return manager_->remove_resource(id_); }
    ResourceMemory memory() const { return manager_->resource_memory(resource()); }
    void set_layer_split(bool enabled) { resource().layer_split_ = enabled; }
    bool place_params(ggml_backend_t backend, ResidencyMode mode) { return manager_->place_resource_params(resource(), backend, mode); }
    const ModelLoader& loader() const { return resource().model_loader_; }
    ModelLoader base_loader() const;
    ModelLoader::Dependencies dependencies() const;
    bool set_loader(ModelLoader loader) { return manager_->set_loader(resource(), std::move(loader)); }
    bool del_file(ModelLoader::FileId file) { return manager_->del_file(resource(), file); }
    ModelLoader::FileId register_model_source(ModelLoader loader) {
        return manager_->register_model_source(resource(), std::move(loader));
    }
    const ModelLoader& model_source(ModelLoader::FileId file) const { return resource().model_sources_.at(file); }
    bool unregister_model_source(ModelLoader::FileId file) { return manager_->unregister_model_source(resource(), file); }
    void prepare_file_io() {
        auto& r = resource();
        r.model_loader_.process_model_files(r.enable_mmap_, r.writable_mmap_);
    }
    bool load_float_tensor(const std::string& name, std::vector<float>& data) {
        auto& r = resource();
        return r.model_loader_.load_float_tensor(name, data, r.n_threads_, r.enable_mmap_);
    }
    size_t registered_params_size(const std::set<ModelComponent>& components) const {
        return manager_->registered_params_size(resource(), components);
    }
    void set_n_threads(int n_threads) {
        auto& r      = resource();
        r.n_threads_ = n_threads;
        r.model_loader_.set_n_threads(n_threads);
    }
    void set_enable_mmap(bool enabled) { resource().enable_mmap_ = enabled; }
    void set_segmented_compute_disabled(bool disabled) { resource().segmented_compute_disabled_ = disabled; }
    void set_prefetch_disabled(bool disabled) { resource().prefetch_disabled_ = disabled; }
    void set_shared_weights(std::string key) { resource().shared_weights_key_ = std::move(key); }
    void set_split_buffer_type(ggml_backend_t backend, ggml_backend_buffer_type_t type, const std::vector<std::pair<ggml_backend_t, size_t>>& limits) {
        manager_->set_split_buffer_type(resource(), backend, type, limits);
    }
    bool prepare_lora_sources(std::vector<LoraSpec>& loras) { return manager_->prepare_lora_sources(resource(), loras); }
    bool set_loras(std::vector<LoraSpec> loras, SDVersion version, std::set<std::string> targets) {
        return manager_->set_loras(resource(), std::move(loras), version, std::move(targets));
    }
    std::set<std::string> tensor_names() const { return manager_->tensor_names(resource()); }
    bool register_param_tensors(ModelComponent component, std::map<std::string, ggml_tensor*> tensors, ResidencyMode mode, ggml_backend_t compute_backend, ggml_backend_t params_backend, size_t* size = nullptr, bool allow_split_buffer = false, bool follow_compute = false, const std::map<ggml_tensor*, enum ggml_op>* tensor_ops = nullptr, ModelLoader::FileId source_file = 0, SDVersion source_version = VERSION_COUNT) {
        return manager_->register_param_tensors(resource(), component, std::move(tensors), mode, compute_backend, params_backend,
                                                size, allow_split_buffer, follow_compute, tensor_ops, source_file, source_version);
    }
    bool unregister_param_tensors(ModelComponent component) {
        return manager_->unregister_param_tensors(resource(), component, nullptr);
    }
    bool unregister_param_tensors(const std::vector<ggml_tensor*>& tensors) {
        return manager_->unregister_param_tensors(tensors, &resource());
    }
    bool validate_registered_tensors() { return manager_->validate_registered_tensors(resource()); }
    bool validate_param_tensors(const std::vector<ggml_tensor*>& tensors) {
        return manager_->validate_param_tensors(tensors, &resource());
    }
    bool load_all_params_eagerly() { return manager_->load_all_params_eagerly(resource()); }

    std::shared_ptr<sd::ExecutionEngine> execution_engine() override { return manager_->execution_engine(); }
    bool segmented_compute_enabled() const override { return !resource().segmented_compute_disabled_; }
    bool prefetch_enabled() const override { return !resource().prefetch_disabled_; }
    bool retains_idle_weights() const override { return manager_->retains_idle_weights(); }
    size_t context_vram_limit(ggml_backend_t backend) const override { return manager_->max_vram_bytes(backend); }
    void register_runtime_owner(uintptr_t owner) override { manager_->register_runtime_owner(owner); }
    void remove_runtime_owner(uintptr_t owner) override { manager_->remove_runtime_owner(owner); }
    bool fits_compute_backend_capacity(const DeviceMemoryRequest& request, const std::vector<ggml_tensor*>& params, DeviceMemoryCheck check) const override {
        return manager_->fits_compute_backend_capacity(request, params, check);
    }
    bool assign_compute_backend(const std::vector<ggml_tensor*>& tensors, ggml_backend_t backend) override {
        return manager_->assign_compute_backend(tensors, backend);
    }
    ggml_tensor* resolve_param_tensor(ggml_tensor* tensor) const override { return manager_->resolve_param_tensor(tensor); }
    bool prepare_params(const std::vector<ggml_tensor*>& tensors) override { return manager_->prepare_params(tensors); }
    void release_compute_backend_params(const std::vector<ggml_tensor*>& tensors) override {
        manager_->release_compute_backend_params(tensors);
    }
    void evict_compute_backend_params(const std::vector<ggml_tensor*>& tensors) override {
        manager_->evict_compute_backend_params(tensors);
    }
    WeightResidencyInfo inspect_compute_backend_params(const std::vector<ggml_tensor*>& tensors) const override {
        return manager_->inspect_compute_backend_params(tensors);
    }
    void update_runtime_residency(uintptr_t owner, ggml_backend_t backend, size_t bytes) override {
        manager_->update_runtime_residency(owner, backend, bytes);
    }
    bool ensure_host_memory(size_t pending, const std::vector<ggml_tensor*>& protected_params, size_t pending_device_commit = 0) override {
        return manager_->ensure_host_memory(pending, protected_params, pending_device_commit);
    }
    bool ensure_compute_backend_capacity(const DeviceMemoryRequest& request,
                                         const std::vector<ggml_tensor*>& required,
                                         const std::vector<std::vector<ggml_tensor*>>& preferred,
                                         const std::vector<ggml_tensor*>& protected_params,
                                         std::string* device_memory_error) override {
        return manager_->ensure_compute_backend_capacity(request, required, preferred, protected_params, device_memory_error);
    }
    DeviceMemoryCapacity inspect_compute_backend_capacity(const DeviceMemoryRequest& request,
                                                          const std::vector<ggml_tensor*>& params,
                                                          DeviceMemoryCheck check,
                                                          bool log_details) const override {
        return manager_->inspect_compute_backend_capacity(request, params, check, log_details);
    }
    std::vector<WeightResidencyBlock> evictable_compute_backend_blocks(
        ggml_backend_t backend,
        const std::vector<ggml_tensor*>& protected_params) const override {
        return manager_->evictable_compute_backend_blocks(backend, protected_params);
    }
    bool evict_compute_backend_block(const WeightResidencyBlock& block, PrefetchBuffer* prefetch_buffer) override {
        return manager_->evict_compute_backend_block(block, prefetch_buffer);
    }
    WeightPrefetchResult prefetch_params(uintptr_t owner, const std::vector<ggml_tensor*>& tensors) override {
        return manager_->prefetch_params(owner, tensors);
    }
    WeightPrefetchResult prepare_prefetch_params(uintptr_t owner, const std::vector<ggml_tensor*>& tensors, PrefetchBuffer* prefetch_buffer) override {
        return manager_->prepare_prefetch_params(owner, tensors, prefetch_buffer);
    }
    void submit_prefetched_params(uintptr_t owner) override { manager_->submit_prefetched_params(owner); }
    bool activate_prefetched_params(uintptr_t owner, const std::vector<ggml_tensor*>& tensors) override {
        return manager_->activate_prefetched_params(owner, tensors);
    }
    void clear_prefetched_params(uintptr_t owner) override { manager_->clear_prefetched_params(owner); }
};

#endif  // __MODEL_MANAGER_H__
