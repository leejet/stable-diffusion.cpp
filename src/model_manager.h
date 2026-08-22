#ifndef __MODEL_MANAGER_H__
#define __MODEL_MANAGER_H__

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "model_loader.h"
#include "weight_manager.h"

class ModelManager : public RunnerWeightManager {
public:
    enum class ResidencyMode {
        Disk,
        ParamBackend,
    };

    struct LoraSpec {
        std::string path;
        float multiplier   = 1.0f;
        bool is_high_noise = false;
        std::string tensor_name_prefix_filter;
        bool required = false;
    };

private:
    struct TensorState {
        std::string name;
        ggml_tensor* tensor = nullptr;
        std::string desc;

        ResidencyMode residency_mode       = ResidencyMode::ParamBackend;
        ggml_backend_t compute_backend     = nullptr;
        ggml_backend_t params_backend      = nullptr;
        bool allow_split_buffer            = false;
        bool params_follow_compute_backend = false;
        bool metadata_validated            = false;
        enum ggml_op usage_op              = GGML_OP_NONE;

        int active_prepare_count = 0;

        bool loaded_to_params_backend  = false;
        bool staged_to_compute_backend = false;
        uint64_t applied_lora_epoch    = UINT64_MAX;
    };

    struct ParamsStorageBlock {
        ggml_backend_buffer_t buffer = nullptr;
        std::vector<MmapTensorStore> mmap_tensor_stores;
        std::vector<TensorState*> states;
    };

    struct ComputeStagingBlock {
        ggml_backend_t compute_backend = nullptr;
        ggml_backend_buffer_t buffer   = nullptr;
        ggml_context* staging_ctx      = nullptr;
        std::vector<std::pair<TensorState*, ggml_tensor*>> staged_tensors;
    };

    struct PrefetchKey {
        uintptr_t owner_id  = 0;
        uint64_t segment_id = 0;

        bool operator<(const PrefetchKey& other) const {
            return owner_id < other.owner_id ||
                   (owner_id == other.owner_id && segment_id < other.segment_id);
        }
    };

    struct PrefetchBlock {
        PrefetchKey key;
        std::vector<TensorState*> states;
        ggml_backend_t compute_backend  = nullptr;
        ggml_backend_t transfer_backend = nullptr;
        ggml_backend_event_t event      = nullptr;
        ggml_context* staging_ctx       = nullptr;
        ggml_backend_buffer_t buffer    = nullptr;
        std::vector<std::pair<TensorState*, ggml_tensor*>> staged_tensors;
    };

    ModelLoader model_loader_;
    std::vector<std::unique_ptr<TensorState>> tensor_states_;
    std::map<std::string, TensorState*> tensor_states_by_name_;
    std::vector<std::unique_ptr<ParamsStorageBlock>> params_storage_blocks_;
    std::vector<std::unique_ptr<ComputeStagingBlock>> compute_staging_blocks_;
    std::map<ggml_backend_t, ggml_backend_buffer_type_t> split_buffer_types_;
    std::map<PrefetchKey, std::unique_ptr<PrefetchBlock>> prefetch_blocks_;
    std::map<ggml_backend_t, ggml_backend_t> prefetch_backends_;
    bool warned_split_lora_skip_ = false;
    std::set<std::string> common_ignore_tensors_;
    std::vector<LoraSpec> loras_;
    SDVersion lora_version_      = VERSION_COUNT;
    uint64_t current_lora_epoch_ = 0;
    int n_threads_               = 0;
    bool enable_mmap_            = false;
    bool writable_mmap_          = false;

    size_t finish_compute_backend_usage(const std::vector<TensorState*>& states);
    void release_all();

    ParamPrefetchResult populate_prefetch_block(PrefetchBlock& block);
    ggml_backend_t prefetch_backend_for(ggml_backend_t compute_backend);
    void synchronize_prefetch_block(PrefetchBlock& block);
    void free_prefetch_block(PrefetchBlock& block);
    void clear_all_param_prefetches();

    bool resolve_required_tensor_states(const std::vector<ggml_tensor*>& tensors,
                                        std::vector<TensorState*>& required_states) const;
    bool should_ignore(const TensorState& state) const;
    bool is_optional_missing_tensor(const std::string& name) const;
    bool validate_tensor(const TensorState& state) const;

    bool load_tensors_to_params_backend(const std::vector<TensorState*>& states);
    bool apply_loras_to_params(const std::vector<TensorState*>& states);
    bool mmap_params(const std::vector<TensorState*>& states,
                     std::vector<ParamsStorageBlock*>& created_storage_blocks);
    bool can_mmap_storage(const TensorState& state) const;
    bool alloc_params_buffers(const std::vector<TensorState*>& states,
                              std::vector<ParamsStorageBlock*>& created_storage_blocks);
    bool load_tensors(const std::vector<TensorState*>& states);
    bool stage_tensors_to_compute_backend(const std::vector<TensorState*>& states);

    ggml_backend_buffer_type_t params_buffer_type_for(const TensorState& state) const;
    ggml_backend_buffer_type_t split_buffer_type_for(const TensorState& state) const;
    size_t release_compute_staging_blocks(bool force = false);
    void release_params_storage_blocks(bool force                                            = false,
                                       const std::unordered_set<TensorState*>* target_states = nullptr);
    void free_compute_staging_block(ComputeStagingBlock& block);
    void free_params_storage_block(ParamsStorageBlock& block);
    void erase_params_storage_block(ParamsStorageBlock* block);
    void reset_lora_applied_params();

public:
    ~ModelManager() override;

    ModelLoader& loader() { return model_loader_; }
    const ModelLoader& loader() const { return model_loader_; }

    void set_n_threads(int n_threads) {
        n_threads_ = n_threads;
        model_loader_.set_n_threads(n_threads);
    }
    void set_enable_mmap(bool enable_mmap) { enable_mmap_ = enable_mmap; }
    void set_writable_mmap(bool writable_mmap) { writable_mmap_ = writable_mmap; }
    void set_common_ignore_tensors(std::set<std::string> ignore_tensors);
    void set_loras(std::vector<LoraSpec> loras, SDVersion version);
    void set_split_buffer_type(ggml_backend_t compute_backend, ggml_backend_buffer_type_t split_buft);

    static bool tensor_shape_supports_split_buffer(const ggml_tensor* tensor);

    std::set<std::string> tensor_names() const;

    bool register_param_tensors(const std::string& desc,
                                std::map<std::string, ggml_tensor*> tensors,
                                ResidencyMode residency_mode,
                                ggml_backend_t compute_backend,
                                ggml_backend_t params_backend,
                                size_t* registered_tensor_size                         = nullptr,
                                bool allow_split_buffer                                = false,
                                bool params_follow_compute_backend                     = false,
                                const std::map<ggml_tensor*, enum ggml_op>* tensor_ops = nullptr);

    bool unregister_param_tensors(const std::string& desc,
                                  size_t* registered_tensor_size = nullptr);

    template <typename Runner>
    bool register_runner_params(const std::string& desc,
                                Runner& runner,
                                ResidencyMode residency_mode,
                                ggml_backend_t compute_backend,
                                ggml_backend_t params_backend,
                                size_t* registered_tensor_size = nullptr) {
        std::map<std::string, ggml_tensor*> tensors;
        runner.get_param_tensors(tensors);
        return register_param_tensors(desc,
                                      std::move(tensors),
                                      residency_mode,
                                      compute_backend,
                                      params_backend,
                                      registered_tensor_size);
    }

    template <typename Runner>
    bool register_runner_params(const std::string& desc,
                                Runner& runner,
                                const std::string& prefix,
                                ResidencyMode residency_mode,
                                ggml_backend_t compute_backend,
                                ggml_backend_t params_backend,
                                size_t* registered_tensor_size = nullptr) {
        std::map<std::string, ggml_tensor*> tensors;
        runner.get_param_tensors(tensors, prefix);
        return register_param_tensors(desc,
                                      std::move(tensors),
                                      residency_mode,
                                      compute_backend,
                                      params_backend,
                                      registered_tensor_size);
    }

    bool validate_registered_tensors();
    bool load_all_params_eagerly();

    bool assign_compute_backend(const std::vector<ggml_tensor*>& tensors,
                                ggml_backend_t compute_backend) override;
    bool prepare_params(const std::vector<ggml_tensor*>& tensors) override;
    size_t release_compute_backend_params(const std::vector<ggml_tensor*>& tensors) override;
    void release_params_backend_params(const std::vector<ggml_tensor*>& tensors) override;
    ParamPrefetchResult enqueue_param_prefetch(
        uintptr_t owner_id,
        uint64_t segment_id,
        const std::vector<ggml_tensor*>& tensors) override;
    bool activate_param_prefetch(uintptr_t owner_id,
                                 uint64_t segment_id,
                                 const std::vector<ggml_tensor*>& tensors) override;
    void clear_param_prefetches(uintptr_t owner_id) override;
    size_t streaming_allocation_bytes(
        uintptr_t owner_id,
        ggml_backend_t compute_backend,
        const std::unordered_set<const ggml_tensor*>& resident_tensors) const override;
};

#endif  // __MODEL_MANAGER_H__
