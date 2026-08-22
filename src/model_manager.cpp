#include "model_manager.h"

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <iterator>
#include <mutex>
#include <unordered_set>

#include "core/ggml_extend_backend.h"
#include "core/util.h"
#include "model/adapter/lora.hpp"

static size_t aligned_offset(const void* buffer, size_t offset, size_t alignment) {
    GGML_ASSERT(alignment != 0 && (alignment & (alignment - 1)) == 0);
    size_t align = (alignment - ((reinterpret_cast<uintptr_t>(buffer) + offset) % alignment)) % alignment;
    return offset + align;
}

static size_t saturating_add(size_t lhs, size_t rhs) {
    return rhs > SIZE_MAX - lhs ? SIZE_MAX : lhs + rhs;
}

static bool lora_specs_equal(const std::vector<ModelManager::LoraSpec>& lhs,
                             const std::vector<ModelManager::LoraSpec>& rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (lhs[i].path != rhs[i].path ||
            lhs[i].multiplier != rhs[i].multiplier ||
            lhs[i].is_high_noise != rhs[i].is_high_noise ||
            lhs[i].tensor_name_prefix_filter != rhs[i].tensor_name_prefix_filter ||
            lhs[i].required != rhs[i].required) {
            return false;
        }
    }
    return true;
}

static std::string lora_id(const ModelManager::LoraSpec& lora) {
    return lora.is_high_noise ? "|high_noise|" + lora.path : lora.path;
}

static bool backend_supports_host_buffer(ggml_backend_t backend) {
    if (backend == nullptr) {
        return false;
    }
    if (sd_backend_is_cpu(backend)) {
        return true;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev == nullptr) {
        return false;
    }
    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(dev, &props);
    return props.caps.buffer_from_host_ptr;
}

static bool device_supports_param_op(ggml_backend_dev_t device,
                                     ggml_tensor* weight,
                                     enum ggml_op op,
                                     ggml_backend_buffer_type_t buft) {
    if (op == GGML_OP_NONE) {
        return true;
    }
    if (device == nullptr || weight == nullptr || buft == nullptr || weight->buffer != nullptr) {
        return false;
    }

    ggml_init_params params;
    params.mem_size   = ggml_tensor_overhead() * 2;
    params.mem_buffer = nullptr;
    params.no_alloc   = true;
    ggml_context* ctx = ggml_init(params);
    if (ctx == nullptr) {
        return false;
    }

    ggml_tensor* op_tensor = nullptr;
    if (op == GGML_OP_GET_ROWS) {
        ggml_tensor* indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        op_tensor            = ggml_get_rows(ctx, weight, indices);
    }
    if (op_tensor == nullptr) {
        ggml_free(ctx);
        return false;
    }

    weight->buffer = ggml_backend_buft_alloc_buffer(buft, 0);
    if (weight->buffer == nullptr) {
        ggml_free(ctx);
        return false;
    }
    bool supported = ggml_backend_dev_supports_op(device, op_tensor);
    ggml_backend_buffer_free(weight->buffer);
    weight->buffer = nullptr;
    ggml_free(ctx);
    return supported;
}

ggml_backend_t ModelManager::prefetch_backend_for(ggml_backend_t compute_backend) {
    auto existing = prefetch_backends_.find(compute_backend);
    if (existing != prefetch_backends_.end()) {
        return existing->second;
    }
    if (compute_backend == nullptr) {
        return nullptr;
    }
    ggml_backend_dev_t device = ggml_backend_get_device(compute_backend);
    if (device == nullptr || ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        return nullptr;
    }
    ggml_backend_t transfer_backend = ggml_backend_dev_init(device, nullptr);
    if (transfer_backend == nullptr) {
        LOG_WARN("model manager failed to create a prefetch backend for %s",
                 ggml_backend_name(compute_backend));
        prefetch_backends_[compute_backend] = nullptr;
        return nullptr;
    }
    prefetch_backends_[compute_backend] = transfer_backend;
    return transfer_backend;
}

void ModelManager::synchronize_prefetch_block(PrefetchBlock& block) {
    if (block.event != nullptr) {
        ggml_backend_event_synchronize(block.event);
        ggml_backend_event_free(block.event);
        block.event = nullptr;
    } else if (block.transfer_backend != nullptr) {
        ggml_backend_synchronize(block.transfer_backend);
    }
    block.transfer_backend = nullptr;
}

void ModelManager::free_prefetch_block(PrefetchBlock& block) {
    synchronize_prefetch_block(block);
    block.staged_tensors.clear();
    if (block.buffer != nullptr) {
        ggml_backend_buffer_free(block.buffer);
        block.buffer = nullptr;
    }
    if (block.staging_ctx != nullptr) {
        ggml_free(block.staging_ctx);
        block.staging_ctx = nullptr;
    }
}

ParamPrefetchResult ModelManager::populate_prefetch_block(PrefetchBlock& block) {
    if (block.states.empty() || block.compute_backend == nullptr) {
        return ParamPrefetchResult::FAILURE;
    }

    block.transfer_backend = prefetch_backend_for(block.compute_backend);
    if (block.transfer_backend == nullptr) {
        return ParamPrefetchResult::FAILURE;
    }

    ggml_init_params init_params;
    init_params.mem_size   = std::max<size_t>(1, block.states.size()) * ggml_tensor_overhead();
    init_params.mem_buffer = nullptr;
    init_params.no_alloc   = true;
    block.staging_ctx      = ggml_init(init_params);
    if (block.staging_ctx == nullptr) {
        LOG_WARN("model manager failed to create the segment prefetch tensor context");
        return ParamPrefetchResult::FAILURE;
    }

    block.staged_tensors.reserve(block.states.size());
    for (TensorState* state : block.states) {
        if (state == nullptr || state->tensor == nullptr || state->tensor->buffer == nullptr ||
            state->tensor->data == nullptr ||
            state->params_backend == nullptr || state->staged_to_compute_backend ||
            state->active_prepare_count > 0) {
            LOG_WARN("model manager segment prefetch source state changed before transfer");
            return ParamPrefetchResult::FAILURE;
        }
        ggml_tensor* staging_tensor = ggml_dup_tensor(block.staging_ctx, state->tensor);
        ggml_set_name(staging_tensor, state->tensor->name);
        block.staged_tensors.push_back({state, staging_tensor});
    }

    ggml_backend_buffer_type_t buffer_type = ggml_backend_get_default_buffer_type(block.compute_backend);
    if (buffer_type == nullptr) {
        LOG_WARN("model manager failed to resolve the segment prefetch buffer type");
        return ParamPrefetchResult::FAILURE;
    }
    block.buffer = ggml_backend_alloc_ctx_tensors_from_buft(block.staging_ctx, buffer_type);
    if (block.buffer == nullptr) {
        LOG_DEBUG("model manager failed to allocate the segment prefetch weight buffer");
        return ParamPrefetchResult::ALLOCATION_FAILURE;
    }
    ggml_backend_buffer_set_usage(block.buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    for (const auto& pair : block.staged_tensors) {
        TensorState* state          = pair.first;
        ggml_tensor* staging_tensor = pair.second;
        const bool host_source      = state->tensor->buffer != nullptr &&
                                      ggml_backend_buffer_is_host(state->tensor->buffer);
        if (host_source &&
            (!ggml_is_contiguous(state->tensor) || !ggml_is_contiguous(staging_tensor) ||
             ggml_nbytes(state->tensor) != ggml_nbytes(staging_tensor))) {
            LOG_WARN("model manager segment prefetch requires contiguous host parameter tensors");
            return ParamPrefetchResult::FAILURE;
        }
    }

    for (const auto& pair : block.staged_tensors) {
        TensorState* state          = pair.first;
        ggml_tensor* staging_tensor = pair.second;
        const bool host_source      = state->tensor->buffer != nullptr &&
                                      ggml_backend_buffer_is_host(state->tensor->buffer);
        if (host_source) {
            ggml_backend_tensor_set_async(block.transfer_backend,
                                          staging_tensor,
                                          state->tensor->data,
                                          0,
                                          ggml_nbytes(state->tensor));
        } else {
            ggml_backend_tensor_copy_async(state->params_backend,
                                           block.transfer_backend,
                                           state->tensor,
                                           staging_tensor);
        }
    }

    ggml_backend_dev_t device = ggml_backend_get_device(block.transfer_backend);
    block.event               = ggml_backend_event_new(device);
    if (block.event != nullptr) {
        ggml_backend_event_record(block.event, block.transfer_backend);
    }

    LOG_DEBUG("model manager queued segment %" PRIu64
              " prefetch (%6.2f MB, %zu tensors) to %s",
              block.key.segment_id,
              ggml_backend_buffer_get_size(block.buffer) / (1024.f * 1024.f),
              block.states.size(),
              ggml_backend_name(block.compute_backend));
    return ParamPrefetchResult::SUCCESS;
}

ParamPrefetchResult ModelManager::enqueue_param_prefetch(
    uintptr_t owner_id,
    uint64_t segment_id,
    const std::vector<ggml_tensor*>& tensors) {
    if (tensors.empty()) {
        return ParamPrefetchResult::SUCCESS;
    }

    std::vector<TensorState*> required_states;
    if (!resolve_required_tensor_states(tensors, required_states) ||
        !load_tensors_to_params_backend(required_states)) {
        return ParamPrefetchResult::FAILURE;
    }

    std::vector<TensorState*> states;
    states.reserve(required_states.size());
    ggml_backend_t compute_backend = nullptr;
    for (TensorState* state : required_states) {
        if (state == nullptr || should_ignore(*state) || is_optional_missing_tensor(state->name) ||
            state->compute_backend == state->params_backend || state->staged_to_compute_backend) {
            continue;
        }
        if (state->active_prepare_count > 0) {
            LOG_WARN("cannot prefetch active tensor '%s'", state->name.c_str());
            return ParamPrefetchResult::FAILURE;
        }
        if (compute_backend == nullptr) {
            compute_backend = state->compute_backend;
        } else if (compute_backend != state->compute_backend) {
            LOG_WARN("segment prefetch cannot span multiple compute backends");
            return ParamPrefetchResult::FAILURE;
        }
        states.push_back(state);
    }
    PrefetchKey key{owner_id, segment_id};
    if (states.empty()) {
        auto existing = prefetch_blocks_.find(key);
        if (existing != prefetch_blocks_.end()) {
            std::unique_ptr<PrefetchBlock> stale = std::move(existing->second);
            prefetch_blocks_.erase(existing);
            free_prefetch_block(*stale);
        }
        return ParamPrefetchResult::SUCCESS;
    }
    if (compute_backend == nullptr || sd_backend_is_cpu(compute_backend)) {
        LOG_WARN("segment prefetch requires a non-CPU compute backend");
        return ParamPrefetchResult::FAILURE;
    }
    auto block             = std::make_unique<PrefetchBlock>();
    block->key             = key;
    block->states          = std::move(states);
    block->compute_backend = compute_backend;

    auto existing = prefetch_blocks_.find(key);
    if (existing != prefetch_blocks_.end()) {
        const auto& existing_states = existing->second->states;
        const bool same_states      = existing->second->compute_backend == compute_backend &&
                                      existing_states.size() == block->states.size() &&
                                      std::is_permutation(existing_states.begin(),
                                                          existing_states.end(),
                                                          block->states.begin());
        if (same_states) {
            return ParamPrefetchResult::SUCCESS;
        }
        clear_param_prefetches(owner_id);
    }

    ParamPrefetchResult result = populate_prefetch_block(*block);
    if (result != ParamPrefetchResult::SUCCESS) {
        free_prefetch_block(*block);
        return result;
    }
    prefetch_blocks_.emplace(key, std::move(block));
    return ParamPrefetchResult::SUCCESS;
}

bool ModelManager::activate_param_prefetch(uintptr_t owner_id,
                                           uint64_t segment_id,
                                           const std::vector<ggml_tensor*>& tensors) {
    std::vector<TensorState*> required_states;
    if (!resolve_required_tensor_states(tensors, required_states)) {
        return false;
    }
    PrefetchKey key{owner_id, segment_id};
    const bool already_staged = std::all_of(required_states.begin(), required_states.end(),
                                            [&](TensorState* state) {
                                                return state == nullptr || should_ignore(*state) ||
                                                       is_optional_missing_tensor(state->name) ||
                                                       state->compute_backend == state->params_backend ||
                                                       state->staged_to_compute_backend;
                                            });
    if (already_staged) {
        auto existing = prefetch_blocks_.find(key);
        if (existing != prefetch_blocks_.end()) {
            std::unique_ptr<PrefetchBlock> stale = std::move(existing->second);
            prefetch_blocks_.erase(existing);
            free_prefetch_block(*stale);
        }
        return true;
    }

    auto existing = prefetch_blocks_.find(key);
    if (existing == prefetch_blocks_.end()) {
        LOG_WARN("segment %" PRIu64 " was not queued for prefetch", segment_id);
        return false;
    }
    std::unique_ptr<PrefetchBlock> block = std::move(existing->second);
    prefetch_blocks_.erase(existing);
    synchronize_prefetch_block(*block);

    LOG_DEBUG("model manager activated prefetched segment %" PRIu64
              " (%6.2f MB, %zu tensors) on %s",
              segment_id,
              ggml_backend_buffer_get_size(block->buffer) / (1024.f * 1024.f),
              block->states.size(),
              ggml_backend_name(block->compute_backend));

    for (const auto& pair : block->staged_tensors) {
        TensorState* state          = pair.first;
        ggml_tensor* staging_tensor = pair.second;
        if (state == nullptr || state->tensor == nullptr || state->staged_to_compute_backend ||
            state->active_prepare_count > 0 || staging_tensor == nullptr) {
            LOG_WARN("segment %" PRIu64 " cannot be activated because tensor state changed", segment_id);
            free_prefetch_block(*block);
            return false;
        }
    }
    for (auto& pair : block->staged_tensors) {
        TensorState* state          = pair.first;
        ggml_tensor* staging_tensor = pair.second;
        std::swap(state->tensor->buffer, staging_tensor->buffer);
        std::swap(state->tensor->data, staging_tensor->data);
        std::swap(state->tensor->extra, staging_tensor->extra);
        state->staged_to_compute_backend = true;
    }

    auto staging_block             = std::make_unique<ComputeStagingBlock>();
    staging_block->compute_backend = block->compute_backend;
    staging_block->buffer          = block->buffer;
    staging_block->staging_ctx     = block->staging_ctx;
    staging_block->staged_tensors  = std::move(block->staged_tensors);
    block->buffer                  = nullptr;
    block->staging_ctx             = nullptr;
    compute_staging_blocks_.push_back(std::move(staging_block));
    return true;
}

void ModelManager::clear_param_prefetches(uintptr_t owner_id) {
    for (auto it = prefetch_blocks_.begin(); it != prefetch_blocks_.end();) {
        if (it->first.owner_id == owner_id) {
            free_prefetch_block(*it->second);
            it = prefetch_blocks_.erase(it);
        } else {
            ++it;
        }
    }
}

size_t ModelManager::streaming_allocation_bytes(
    uintptr_t owner_id,
    ggml_backend_t compute_backend,
    const std::unordered_set<const ggml_tensor*>& resident_tensors) const {
    size_t bytes = 0;
    for (const auto& block : compute_staging_blocks_) {
        if (block == nullptr || block->buffer == nullptr ||
            block->compute_backend != compute_backend) {
            continue;
        }
        const bool contains_resident = std::any_of(
            block->staged_tensors.begin(),
            block->staged_tensors.end(),
            [&](const std::pair<TensorState*, ggml_tensor*>& pair) {
                return pair.first != nullptr &&
                       resident_tensors.count(pair.first->tensor) != 0;
            });
        if (contains_resident) {
            bytes = saturating_add(bytes, ggml_backend_buffer_get_size(block->buffer));
        }
    }
    for (const auto& entry : prefetch_blocks_) {
        const PrefetchBlock* block = entry.second.get();
        if (entry.first.owner_id != owner_id || block == nullptr || block->buffer == nullptr ||
            block->compute_backend != compute_backend) {
            continue;
        }
        bytes = saturating_add(bytes, ggml_backend_buffer_get_size(block->buffer));
    }
    return bytes;
}

void ModelManager::clear_all_param_prefetches() {
    for (auto& entry : prefetch_blocks_) {
        free_prefetch_block(*entry.second);
    }
    prefetch_blocks_.clear();
}

ModelManager::~ModelManager() {
    clear_all_param_prefetches();
    release_all();
    for (auto& entry : prefetch_backends_) {
        if (entry.second != nullptr) {
            ggml_backend_free(entry.second);
        }
    }
    prefetch_backends_.clear();
}

void ModelManager::set_common_ignore_tensors(std::set<std::string> ignore_tensors) {
    common_ignore_tensors_ = std::move(ignore_tensors);
}

void ModelManager::set_loras(std::vector<LoraSpec> loras, SDVersion version) {
    if (loras.empty() && loras_.empty()) {
        lora_version_ = version;
        return;
    }
    if (lora_version_ == version && lora_specs_equal(loras_, loras)) {
        return;
    }

    loras_        = std::move(loras);
    lora_version_ = version;
    current_lora_epoch_++;
    reset_lora_applied_params();
}

std::set<std::string> ModelManager::tensor_names() const {
    std::set<std::string> names;
    for (const auto& state : tensor_states_) {
        if (state != nullptr) {
            names.insert(state->name);
        }
    }
    return names;
}

size_t estimate_tensors_size(const std::map<std::string, ggml_tensor*>& tensors) {
    size_t size = 0;
    std::unordered_set<ggml_tensor*> seen;
    for (const auto& pair : tensors) {
        ggml_tensor* tensor = pair.second;
        if (tensor == nullptr || seen.find(tensor) != seen.end()) {
            continue;
        }
        seen.insert(tensor);
        size += ggml_nbytes(tensor);
    }
    return size;
}

void ModelManager::set_split_buffer_type(ggml_backend_t compute_backend, ggml_backend_buffer_type_t split_buft) {
    if (compute_backend == nullptr) {
        return;
    }
    if (split_buft == nullptr) {
        split_buffer_types_.erase(compute_backend);
        return;
    }
    split_buffer_types_[compute_backend] = split_buft;
}

bool ModelManager::tensor_shape_supports_split_buffer(const ggml_tensor* tensor) {
    return tensor != nullptr &&
           tensor->view_src == nullptr &&
           ggml_is_contiguous(tensor) &&
           ggml_n_dims(tensor) == 2 &&
           tensor->ne[0] >= 256 &&
           tensor->ne[1] >= 256;
}

ggml_backend_buffer_type_t ModelManager::split_buffer_type_for(const TensorState& state) const {
    if (!state.allow_split_buffer || !tensor_shape_supports_split_buffer(state.tensor)) {
        return nullptr;
    }
    auto it = split_buffer_types_.find(state.compute_backend);
    return it != split_buffer_types_.end() ? it->second : nullptr;
}

bool ModelManager::register_param_tensors(const std::string& desc,
                                          std::map<std::string, ggml_tensor*> tensors,
                                          ResidencyMode residency_mode,
                                          ggml_backend_t compute_backend,
                                          ggml_backend_t params_backend,
                                          size_t* registered_tensor_size,
                                          bool allow_split_buffer,
                                          bool params_follow_compute_backend,
                                          const std::map<ggml_tensor*, enum ggml_op>* tensor_ops) {
    if (desc.empty()) {
        LOG_ERROR("model manager tensor desc is empty");
        return false;
    }
    if (registered_tensor_size != nullptr) {
        *registered_tensor_size += estimate_tensors_size(tensors);
    }

    std::vector<std::unique_ptr<TensorState>> new_states;
    new_states.reserve(tensors.size());

    for (const auto& pair : tensors) {
        const std::string& name = pair.first;
        ggml_tensor* tensor     = pair.second;
        if (tensor == nullptr) {
            continue;
        }
        if (tensor_states_by_name_.find(name) != tensor_states_by_name_.end()) {
            LOG_ERROR("model manager tensor name '%s' is already registered", name.c_str());
            return false;
        }
        ggml_set_name(tensor, name.c_str());

        auto state                           = std::make_unique<TensorState>();
        state->name                          = name;
        state->tensor                        = tensor;
        state->desc                          = desc;
        state->residency_mode                = residency_mode;
        state->compute_backend               = compute_backend;
        state->params_backend                = params_backend;
        state->allow_split_buffer            = allow_split_buffer;
        state->params_follow_compute_backend = params_follow_compute_backend;
        if (tensor_ops != nullptr) {
            auto op_it = tensor_ops->find(tensor);
            if (op_it != tensor_ops->end()) {
                state->usage_op = op_it->second;
            }
        }
        new_states.push_back(std::move(state));
    }

    for (auto& state : new_states) {
        TensorState* registered_state                  = state.get();
        tensor_states_by_name_[registered_state->name] = registered_state;
        tensor_states_.push_back(std::move(state));
    }
    return true;
}

bool ModelManager::unregister_param_tensors(const std::string& desc, size_t* registered_tensor_size) {
    if (desc.empty()) {
        return true;
    }

    std::unordered_set<TensorState*> target_states;
    size_t released_size = 0;
    for (auto& state : tensor_states_) {
        if (state == nullptr || state->desc != desc) {
            continue;
        }
        if (state->active_prepare_count > 0) {
            LOG_ERROR("model manager cannot unregister active %s tensor '%s'",
                      desc.c_str(),
                      state->name.c_str());
            return false;
        }
        target_states.insert(state.get());
        if (state->tensor != nullptr) {
            released_size += ggml_nbytes(state->tensor);
        }
    }

    if (target_states.empty()) {
        return true;
    }

    clear_all_param_prefetches();
    release_compute_staging_blocks(false);

    std::vector<ParamsStorageBlock*> storage_blocks_to_release;
    std::unordered_set<TensorState*> affected_storage_states;
    for (const auto& block : params_storage_blocks_) {
        if (block == nullptr) {
            continue;
        }
        bool has_target_state = false;
        for (TensorState* state : block->states) {
            if (state != nullptr && target_states.count(state) > 0) {
                has_target_state = true;
                break;
            }
        }
        if (!has_target_state) {
            continue;
        }
        storage_blocks_to_release.push_back(block.get());
        for (TensorState* state : block->states) {
            if (state != nullptr) {
                affected_storage_states.insert(state);
            }
        }
    }

    for (TensorState* state : affected_storage_states) {
        if (state == nullptr) {
            continue;
        }
        if (state->active_prepare_count > 0 || state->staged_to_compute_backend) {
            LOG_ERROR("model manager cannot unregister %s while tensor '%s' is active",
                      desc.c_str(),
                      state->name.c_str());
            return false;
        }
    }

    for (ParamsStorageBlock* block : storage_blocks_to_release) {
        if (block != nullptr) {
            free_params_storage_block(*block);
            erase_params_storage_block(block);
        }
    }

    for (auto it = tensor_states_by_name_.begin(); it != tensor_states_by_name_.end();) {
        if (target_states.count(it->second) > 0) {
            it = tensor_states_by_name_.erase(it);
        } else {
            ++it;
        }
    }
    tensor_states_.erase(std::remove_if(tensor_states_.begin(),
                                        tensor_states_.end(),
                                        [&](const std::unique_ptr<TensorState>& s) {
                                            return s == nullptr || target_states.count(s.get()) > 0;
                                        }),
                         tensor_states_.end());

    if (registered_tensor_size != nullptr) {
        if (released_size > *registered_tensor_size) {
            *registered_tensor_size = 0;
        } else {
            *registered_tensor_size -= released_size;
        }
    }
    return true;
}

bool ModelManager::load_all_params_eagerly() {
    std::vector<TensorState*> all_states;
    all_states.reserve(tensor_states_.size());
    for (const auto& s : tensor_states_) {
        if (s != nullptr) {
            all_states.push_back(s.get());
        }
    }
    return load_tensors_to_params_backend(all_states);
}

bool ModelManager::validate_registered_tensors() {
    bool ok = true;
    for (const auto& state : tensor_states_) {
        if (state == nullptr) {
            ok = false;
            continue;
        }
        bool state_ok = validate_tensor(*state);
        if (state_ok) {
            state->metadata_validated = true;
        }
        ok = state_ok && ok;
    }
    return ok;
}

bool ModelManager::load_tensors_to_params_backend(const std::vector<TensorState*>& states) {
    std::vector<TensorState*> need_load;
    need_load.reserve(states.size());
    for (TensorState* state : states) {
        if (state == nullptr || should_ignore(*state) || is_optional_missing_tensor(state->name)) {
            continue;
        }
        if (!state->metadata_validated) {
            if (!validate_tensor(*state)) {
                return false;
            }
            state->metadata_validated = true;
        }
        if (!state->loaded_to_params_backend) {
            need_load.push_back(state);
        }
    }
    if (need_load.empty()) {
        return true;
    }

    std::vector<ParamsStorageBlock*> created_storage_blocks;
    if (!mmap_params(need_load, created_storage_blocks)) {
        for (ParamsStorageBlock* block : created_storage_blocks) {
            if (block != nullptr) {
                free_params_storage_block(*block);
                erase_params_storage_block(block);
            }
        }
        return false;
    }

    std::vector<TensorState*> need_alloc;
    need_alloc.reserve(need_load.size());
    for (TensorState* state : need_load) {
        if (state->tensor != nullptr && state->tensor->data == nullptr && state->tensor->view_src == nullptr) {
            need_alloc.push_back(state);
        }
    }

    if (!alloc_params_buffers(need_alloc, created_storage_blocks) ||
        !load_tensors(need_load)) {
        for (ParamsStorageBlock* block : created_storage_blocks) {
            if (block != nullptr) {
                free_params_storage_block(*block);
                erase_params_storage_block(block);
            }
        }
        return false;
    }
    for (ParamsStorageBlock* block : created_storage_blocks) {
        if (block != nullptr && block->buffer != nullptr) {
            LOG_DEBUG("model manager prepared params backend buffer (%6.2f MB, %zu tensors, %s)",
                      ggml_backend_buffer_get_size(block->buffer) / (1024.f * 1024.f),
                      block->states.size(),
                      ggml_backend_buffer_is_host(block->buffer) ? "RAM" : "VRAM");
        }
    }

    return true;
}

bool ModelManager::stage_tensors_to_compute_backend(const std::vector<TensorState*>& states) {
    std::map<std::pair<ggml_backend_t, ggml_backend_buffer_type_t>, std::vector<TensorState*>> states_by_staging_target;
    for (TensorState* state : states) {
        if (state == nullptr || should_ignore(*state) || is_optional_missing_tensor(state->name)) {
            continue;
        }
        if (state->compute_backend == nullptr) {
            LOG_ERROR("model manager compute backend is null for tensor '%s'", state->name.c_str());
            return false;
        }
        if (state->params_backend == nullptr) {
            LOG_ERROR("model manager params backend is null for tensor '%s'", state->name.c_str());
            return false;
        }
        if (state->compute_backend == state->params_backend || state->staged_to_compute_backend) {
            continue;
        }
        if (!state->loaded_to_params_backend || state->tensor == nullptr || state->tensor->data == nullptr) {
            LOG_ERROR("model manager tensor '%s' is not loaded to params backend", state->name.c_str());
            return false;
        }
        ggml_backend_buffer_type_t staging_buft = split_buffer_type_for(*state);
        if (staging_buft == nullptr) {
            staging_buft = ggml_backend_get_default_buffer_type(state->compute_backend);
        }
        states_by_staging_target[{state->compute_backend, staging_buft}].push_back(state);
    }

    for (const auto& pair : states_by_staging_target) {
        ggml_backend_t compute_backend          = pair.first.first;
        ggml_backend_buffer_type_t staging_buft = pair.first.second;
        const std::vector<TensorState*>& states = pair.second;
        if (states.empty()) {
            continue;
        }

        int64_t t0 = ggml_time_ms();

        ggml_init_params init_params;
        init_params.mem_size   = std::max<size_t>(1, states.size()) * ggml_tensor_overhead();
        init_params.mem_buffer = nullptr;
        init_params.no_alloc   = true;

        ggml_context* staging_ctx = ggml_init(init_params);
        GGML_ASSERT(staging_ctx != nullptr);

        std::vector<std::pair<TensorState*, ggml_tensor*>> staged_tensors;
        staged_tensors.reserve(states.size());
        for (TensorState* state : states) {
            ggml_tensor* staging_tensor = ggml_dup_tensor(staging_ctx, state->tensor);
            ggml_set_name(staging_tensor, state->tensor->name);
            staged_tensors.push_back({state, staging_tensor});
        }

        ggml_backend_buffer_t compute_buffer = ggml_backend_alloc_ctx_tensors_from_buft(staging_ctx, staging_buft);
        if (compute_buffer == nullptr) {
            LOG_ERROR("model manager alloc compute params backend buffer failed, num_tensors = %zu",
                      staged_tensors.size());
            ggml_free(staging_ctx);
            return false;
        }
        ggml_backend_buffer_set_usage(compute_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        for (auto& staged_tensor : staged_tensors) {
            TensorState* state          = staged_tensor.first;
            ggml_tensor* managed_tensor = state->tensor;
            ggml_tensor* staging_tensor = staged_tensor.second;
            ggml_backend_tensor_copy(managed_tensor, staging_tensor);
            std::swap(managed_tensor->buffer, staging_tensor->buffer);
            std::swap(managed_tensor->data, staging_tensor->data);
            std::swap(managed_tensor->extra, staging_tensor->extra);
        }
        ggml_backend_synchronize(compute_backend);

        auto block             = std::make_unique<ComputeStagingBlock>();
        block->compute_backend = compute_backend;
        block->buffer          = compute_buffer;
        block->staging_ctx     = staging_ctx;
        block->staged_tensors  = std::move(staged_tensors);
        for (auto& staged_tensor : block->staged_tensors) {
            TensorState* state               = staged_tensor.first;
            state->staged_to_compute_backend = true;
        }
        compute_staging_blocks_.push_back(std::move(block));

        int64_t t1 = ggml_time_ms();
        LOG_DEBUG("model manager staged compute params (%6.2f MB, %zu tensors) to %s, taking %.2fs",
                  ggml_backend_buffer_get_size(compute_buffer) / (1024.f * 1024.f),
                  states.size(),
                  ggml_backend_name(compute_backend),
                  (t1 - t0) * 1.0f / 1000);
    }

    return true;
}

bool ModelManager::apply_loras_to_params(const std::vector<TensorState*>& states) {
    if (loras_.empty()) {
        return true;
    }

    struct LoraApplyGroup {
        std::map<std::string, ggml_tensor*> model_tensors;
        std::vector<TensorState*> states;
    };

    std::map<ggml_backend_t, LoraApplyGroup> groups;
    for (TensorState* state : states) {
        if (state == nullptr || state->tensor == nullptr ||
            should_ignore(*state) || is_optional_missing_tensor(state->name)) {
            continue;
        }
        if (state->applied_lora_epoch == current_lora_epoch_) {
            continue;
        }
        if (state->compute_backend == nullptr) {
            LOG_ERROR("model manager compute backend is null for lora target tensor '%s'", state->name.c_str());
            return false;
        }
        if (state->tensor->buffer != nullptr &&
            ggml_backend_buffer_get_type(state->tensor->buffer) == split_buffer_type_for(*state)) {
            if (!warned_split_lora_skip_) {
                LOG_WARN(
                    "model manager skipping direct lora application to row-split tensors "
                    "(use --lora-apply-mode at_runtime with row split)");
                warned_split_lora_skip_ = true;
            }
            state->applied_lora_epoch = current_lora_epoch_;
            continue;
        }
        if (state->tensor->data == nullptr) {
            LOG_ERROR("model manager lora target tensor '%s' is not prepared", state->name.c_str());
            return false;
        }
        LoraApplyGroup& group            = groups[state->compute_backend];
        group.model_tensors[state->name] = state->tensor;
        group.states.push_back(state);
    }

    if (groups.empty()) {
        return true;
    }

    std::set<std::string> all_tensor_names = tensor_names();
    for (auto& group_pair : groups) {
        ggml_backend_t compute_backend = group_pair.first;
        LoraApplyGroup& group          = group_pair.second;
        for (const LoraSpec& lora_spec : loras_) {
            if (group.model_tensors.empty()) {
                continue;
            }

            std::string id = lora_id(lora_spec);
            auto lora      = std::make_shared<LoraModel>(id,
                                                    compute_backend,
                                                    compute_backend,
                                                    lora_spec.path,
                                                    lora_spec.is_high_noise ? "model.high_noise_" : "",
                                                    lora_version_);

            LoraModel::filter_t lora_tensor_filter = nullptr;
            if (!lora_spec.tensor_name_prefix_filter.empty()) {
                lora_tensor_filter = [&](const std::string& tensor_name) {
                    return starts_with(tensor_name, lora_spec.tensor_name_prefix_filter);
                };
            }
            if (!lora->load_from_file(n_threads_, lora_tensor_filter)) {
                LOG_WARN("load lora tensors from %s failed", lora_spec.path.c_str());
                if (lora_spec.required) {
                    return false;
                }
                continue;
            }
            if (lora->lora_tensors.empty()) {
                if (lora_spec.required) {
                    LOG_ERROR("required lora has no tensors: %s", lora_spec.path.c_str());
                    return false;
                }
                continue;
            }
            lora->multiplier = lora_spec.multiplier;
            lora->apply(group.model_tensors, all_tensor_names, lora_version_, n_threads_, false);
            lora->release_loaded_tensors();
        }

        for (TensorState* state : group.states) {
            if (state != nullptr) {
                state->applied_lora_epoch = current_lora_epoch_;
            }
        }
    }
    return true;
}

void ModelManager::reset_lora_applied_params() {
    clear_all_param_prefetches();
    release_compute_staging_blocks(true);
    release_params_storage_blocks(true);
    for (auto& state : tensor_states_) {
        state->applied_lora_epoch = UINT64_MAX;
    }
}

bool ModelManager::should_ignore(const TensorState& state) const {
    for (const auto& ignore_prefix : common_ignore_tensors_) {
        if (starts_with(state.name, ignore_prefix)) {
            return true;
        }
    }
    return false;
}

bool ModelManager::is_optional_missing_tensor(const std::string& name) const {
    return name.find("cond_stage_model.transformer.text_model.encoder.layers.23") != std::string::npos ||
           name.find("alphas_cumprod") != std::string::npos;
}

bool ModelManager::validate_tensor(const TensorState& state) const {
    if (state.tensor == nullptr || should_ignore(state) || is_optional_missing_tensor(state.name)) {
        return true;
    }

    const auto& tensor_storage_map = model_loader_.get_tensor_storage_map();
    auto ts_it                     = tensor_storage_map.find(state.name);
    if (ts_it == tensor_storage_map.end()) {
        LOG_ERROR("%s tensor '%s' not in model metadata", state.desc.c_str(), state.name.c_str());
        return false;
    }

    const TensorStorage& tensor_storage = ts_it->second;
    if (state.tensor->ne[0] != tensor_storage.ne[0] ||
        state.tensor->ne[1] != tensor_storage.ne[1] ||
        state.tensor->ne[2] != tensor_storage.ne[2] ||
        state.tensor->ne[3] != tensor_storage.ne[3]) {
        LOG_ERROR(
            "%s tensor '%s' has wrong shape in model metadata: got [%d, %d, %d, %d], expected [%d, %d, %d, %d]",
            state.desc.c_str(),
            state.name.c_str(),
            (int)tensor_storage.ne[0], (int)tensor_storage.ne[1], (int)tensor_storage.ne[2], (int)tensor_storage.ne[3],
            (int)state.tensor->ne[0], (int)state.tensor->ne[1], (int)state.tensor->ne[2], (int)state.tensor->ne[3]);
        return false;
    }
    return true;
}

bool ModelManager::mmap_params(const std::vector<TensorState*>& states,
                               std::vector<ParamsStorageBlock*>& created_storage_blocks) {
    std::map<std::string, ggml_tensor*> mmap_candidates;
    std::map<std::string, TensorState*> mmap_states;
    for (TensorState* state : states) {
        if (state == nullptr || !can_mmap_storage(*state) || state->tensor == nullptr ||
            state->tensor->data != nullptr || state->tensor->view_src != nullptr) {
            continue;
        }
        mmap_candidates[state->name] = state->tensor;
        mmap_states[state->name]     = state;
    }
    if (mmap_candidates.empty()) {
        return true;
    }

    auto mmap_store = model_loader_.mmap_tensors(mmap_candidates, {}, writable_mmap_);
    if (mmap_store.empty()) {
        return true;
    }

    auto block                = std::make_unique<ParamsStorageBlock>();
    block->mmap_tensor_stores = std::move(mmap_store);
    ParamsStorageBlock* raw   = block.get();
    for (const auto& pair : mmap_states) {
        TensorState* state = pair.second;
        if (state != nullptr && state->tensor != nullptr && state->tensor->data != nullptr) {
            block->states.push_back(state);
        }
    }

    if (!block->states.empty()) {
        params_storage_blocks_.push_back(std::move(block));
        created_storage_blocks.push_back(raw);
    }
    return true;
}

bool ModelManager::can_mmap_storage(const TensorState& state) const {
    if (!enable_mmap_ || state.residency_mode != ResidencyMode::ParamBackend) {
        return false;
    }
    if (state.compute_backend == nullptr || state.params_backend == nullptr) {
        return false;
    }
    return sd_backend_is_cpu(state.compute_backend) ||
           sd_backend_is_cpu(state.params_backend) ||
           backend_supports_host_buffer(state.compute_backend);
}

bool ModelManager::alloc_params_buffers(const std::vector<TensorState*>& states,
                                        std::vector<ParamsStorageBlock*>& created_storage_blocks) {
    std::map<std::pair<ggml_backend_buffer_type_t, int>, std::vector<TensorState*>> states_by_buffer_type;
    for (TensorState* state : states) {
        if (state == nullptr || state->tensor == nullptr) {
            continue;
        }
        ggml_backend_buffer_type_t params_buft = params_buffer_type_for(*state);
        if (params_buft == nullptr) {
            return false;
        }
        states_by_buffer_type[{params_buft, static_cast<int>(state->residency_mode)}].push_back(state);
    }

    for (const auto& pair : states_by_buffer_type) {
        ggml_backend_buffer_type_t params_buft  = pair.first.first;
        const std::vector<TensorState*>& states = pair.second;
        size_t alignment                        = ggml_backend_buft_get_alignment(params_buft);
        size_t max_size                         = ggml_backend_buft_get_max_size(params_buft);

        auto alloc_chunk = [&](const std::vector<TensorState*>& chunk, size_t chunk_size) -> bool {
            if (chunk.empty() || chunk_size == 0) {
                return true;
            }

            ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(params_buft, chunk_size);
            if (buffer == nullptr) {
                LOG_ERROR("model manager alloc params backend buffer failed, size = %.2fMB",
                          chunk_size / (1024.0 * 1024.0));
                return false;
            }
            ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

            std::vector<ggml_tensor*> initialized_tensors;
            void* base    = ggml_backend_buffer_get_base(buffer);
            size_t offset = aligned_offset(base, 0, ggml_backend_buffer_get_alignment(buffer));
            for (TensorState* state : chunk) {
                ggml_tensor* tensor     = state->tensor;
                size_t tensor_size      = GGML_PAD(ggml_backend_buffer_get_alloc_size(buffer, tensor),
                                                   ggml_backend_buffer_get_alignment(buffer));
                enum ggml_status status = ggml_backend_tensor_alloc(buffer, tensor, static_cast<char*>(base) + offset);
                if (status != GGML_STATUS_SUCCESS) {
                    LOG_ERROR("model manager failed to initialize params tensor '%s'", ggml_get_name(tensor));
                    for (ggml_tensor* initialized : initialized_tensors) {
                        initialized->buffer = nullptr;
                        initialized->data   = nullptr;
                        initialized->extra  = nullptr;
                    }
                    LOG_DEBUG("model manager releasing params backend buffer (%6.2f MB, %zu tensors, %s)",
                              ggml_backend_buffer_get_size(buffer) / (1024.f * 1024.f),
                              initialized_tensors.size(),
                              ggml_backend_buffer_is_host(buffer) ? "RAM" : "VRAM");
                    ggml_backend_buffer_free(buffer);
                    return false;
                }
                initialized_tensors.push_back(tensor);
                offset += tensor_size;
            }

            auto block              = std::make_unique<ParamsStorageBlock>();
            block->buffer           = buffer;
            block->states           = chunk;
            ParamsStorageBlock* raw = block.get();
            params_storage_blocks_.push_back(std::move(block));
            created_storage_blocks.push_back(raw);

            return true;
        };

        std::vector<TensorState*> chunk;
        size_t chunk_size = 0;
        for (TensorState* state : states) {
            ggml_tensor* tensor = state->tensor;
            size_t tensor_size  = GGML_PAD(ggml_backend_buft_get_alloc_size(params_buft, tensor), alignment);
            // Some backends, e.g. Vulkan, report a preferred chunk size here rather than a
            // hard per-tensor allocation limit. Oversized tensors are allocated alone.
            if (!chunk.empty() && max_size > 0 && chunk_size + tensor_size > max_size) {
                if (!alloc_chunk(chunk, chunk_size)) {
                    return false;
                }
                chunk.clear();
                chunk_size = 0;
            }
            chunk.push_back(state);
            chunk_size += tensor_size;
        }

        if (!alloc_chunk(chunk, chunk_size)) {
            return false;
        }
    }

    return true;
}

bool ModelManager::load_tensors(const std::vector<TensorState*>& states) {
    std::map<std::string, TensorState*> states_by_name;
    std::set<std::string> target_tensor_names;
    for (TensorState* state : states) {
        if (state == nullptr) {
            continue;
        }
        states_by_name[state->name] = state;
        target_tensor_names.insert(state->name);
    }
    if (states_by_name.empty()) {
        return true;
    }

    std::set<std::string> loaded_names;
    std::mutex loaded_names_mutex;
    auto on_new_tensor_cb = [&](const TensorStorage& tensor_storage, ggml_tensor** dst_tensor) -> bool {
        const std::string& name = tensor_storage.name;
        *dst_tensor             = nullptr;

        auto state_it = states_by_name.find(name);
        if (state_it == states_by_name.end()) {
            return true;
        }

        TensorState* state = state_it->second;
        if (state == nullptr || state->tensor == nullptr) {
            LOG_ERROR("model manager tensor '%s' is null", name.c_str());
            return false;
        }

        if (state->tensor->ne[0] != tensor_storage.ne[0] ||
            state->tensor->ne[1] != tensor_storage.ne[1] ||
            state->tensor->ne[2] != tensor_storage.ne[2] ||
            state->tensor->ne[3] != tensor_storage.ne[3]) {
            LOG_ERROR(
                "model manager tensor '%s' has wrong shape in model file: got [%d, %d, %d, %d], expected [%d, %d, %d, %d]",
                name.c_str(),
                (int)tensor_storage.ne[0], (int)tensor_storage.ne[1], (int)tensor_storage.ne[2], (int)tensor_storage.ne[3],
                (int)state->tensor->ne[0], (int)state->tensor->ne[1], (int)state->tensor->ne[2], (int)state->tensor->ne[3]);
            return false;
        }

        {
            std::lock_guard<std::mutex> lock(loaded_names_mutex);
            loaded_names.insert(name);
        }
        *dst_tensor = state->tensor;
        return true;
    };

    if (!model_loader_.load_tensors(on_new_tensor_cb, enable_mmap_, &target_tensor_names)) {
        LOG_ERROR("model manager load tensors failed");
        return false;
    }

    bool missing = false;
    for (const auto& pair : states_by_name) {
        const std::string& name = pair.first;
        if (loaded_names.find(name) == loaded_names.end()) {
            LOG_ERROR("model manager tensor '%s' was not loaded", name.c_str());
            missing = true;
        }
    }
    if (missing) {
        return false;
    }

    for (const auto& pair : states_by_name) {
        pair.second->loaded_to_params_backend = true;
    }
    return true;
}

ggml_backend_buffer_type_t ModelManager::params_buffer_type_for(const TensorState& state) const {
    if (state.params_backend == nullptr) {
        LOG_ERROR("model manager params backend is null for tensor '%s'", state.name.c_str());
        return nullptr;
    }
    ggml_backend_buffer_type_t params_buft = nullptr;
    if (state.compute_backend != nullptr && state.params_backend != state.compute_backend) {
        ggml_backend_dev_t compute_dev = ggml_backend_get_device(state.compute_backend);
        if (compute_dev != nullptr) {
            params_buft = ggml_backend_dev_host_buffer_type(compute_dev);
        }
    } else if (state.params_backend == state.compute_backend) {
        params_buft = split_buffer_type_for(state);
    }
    if (params_buft == nullptr) {
        params_buft = ggml_backend_get_default_buffer_type(state.params_backend);
    }
    if (state.usage_op != GGML_OP_NONE &&
        state.compute_backend != nullptr) {
        ggml_backend_dev_t compute_dev = ggml_backend_get_device(state.compute_backend);
        if (device_supports_param_op(compute_dev, state.tensor, state.usage_op, params_buft)) {
            return params_buft;
        }

        ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        params_buft                = cpu_dev != nullptr ? ggml_backend_dev_buffer_type(cpu_dev) : nullptr;
        if (!device_supports_param_op(cpu_dev, state.tensor, state.usage_op, params_buft)) {
            LOG_ERROR("model manager has no compatible buffer for tensor '%s' used by %s",
                      state.name.c_str(),
                      ggml_op_name(state.usage_op));
            return nullptr;
        }
    }
    return params_buft;
}

void ModelManager::free_compute_staging_block(ComputeStagingBlock& block) {
    for (auto& staged_tensor : block.staged_tensors) {
        TensorState* state          = staged_tensor.first;
        ggml_tensor* staging_tensor = staged_tensor.second;
        if (state == nullptr || state->tensor == nullptr || staging_tensor == nullptr) {
            continue;
        }
        ggml_tensor* managed_tensor = state->tensor;
        managed_tensor->buffer      = staging_tensor->buffer;
        managed_tensor->data        = staging_tensor->data;
        managed_tensor->extra       = staging_tensor->extra;
        staging_tensor->buffer      = nullptr;
        staging_tensor->data        = nullptr;
        staging_tensor->extra       = nullptr;

        state->staged_to_compute_backend = false;
        state->applied_lora_epoch        = UINT64_MAX;
    }

    if (block.buffer != nullptr) {
        LOG_DEBUG("model manager releasing compute params (%6.2f MB, %zu tensors) from %s",
                  ggml_backend_buffer_get_size(block.buffer) / (1024.f * 1024.f),
                  block.staged_tensors.size(),
                  block.compute_backend != nullptr ? ggml_backend_name(block.compute_backend) : "unknown");
        ggml_backend_buffer_free(block.buffer);
        block.buffer = nullptr;
    }
    if (block.staging_ctx != nullptr) {
        ggml_free(block.staging_ctx);
        block.staging_ctx = nullptr;
    }
    block.staged_tensors.clear();
}

size_t ModelManager::release_compute_staging_blocks(bool force) {
    size_t released_bytes = 0;
    for (auto it = compute_staging_blocks_.begin(); it != compute_staging_blocks_.end();) {
        ComputeStagingBlock* block = it->get();
        bool can_release           = force;
        if (!can_release) {
            can_release = std::all_of(block->staged_tensors.begin(),
                                      block->staged_tensors.end(),
                                      [](const std::pair<TensorState*, ggml_tensor*>& pair) {
                                          TensorState* state = pair.first;
                                          if (state == nullptr) {
                                              return true;
                                          }
                                          return state->active_prepare_count == 0;
                                      });
        }

        if (can_release) {
            if (block->buffer != nullptr) {
                released_bytes = saturating_add(
                    released_bytes,
                    ggml_backend_buffer_get_size(block->buffer));
            }
            free_compute_staging_block(*block);
            it = compute_staging_blocks_.erase(it);
        } else {
            ++it;
        }
    }
    return released_bytes;
}

void ModelManager::free_params_storage_block(ParamsStorageBlock& block) {
    if (block.buffer != nullptr) {
        LOG_DEBUG("model manager releasing params backend buffer (%6.2f MB, %zu tensors, %s)",
                  ggml_backend_buffer_get_size(block.buffer) / (1024.f * 1024.f),
                  block.states.size(),
                  ggml_backend_buffer_is_host(block.buffer) ? "RAM" : "VRAM");
        ggml_backend_buffer_free(block.buffer);
        block.buffer = nullptr;
    }
    block.mmap_tensor_stores.clear();

    for (TensorState* state : block.states) {
        if (state == nullptr || state->tensor == nullptr) {
            continue;
        }
        state->tensor->buffer = nullptr;
        state->tensor->data   = nullptr;
        state->tensor->extra  = nullptr;

        state->loaded_to_params_backend = false;
        state->applied_lora_epoch       = UINT64_MAX;
    }
    block.states.clear();
}

void ModelManager::release_params_storage_blocks(bool force,
                                                 const std::unordered_set<TensorState*>* target_states) {
    for (auto it = params_storage_blocks_.begin(); it != params_storage_blocks_.end();) {
        ParamsStorageBlock* block = it->get();
        bool can_release          = force;
        if (!can_release) {
            can_release = std::all_of(block->states.begin(),
                                      block->states.end(),
                                      [target_states](TensorState* state) {
                                          if (state == nullptr) {
                                              return true;
                                          }
                                          if (target_states != nullptr &&
                                              target_states->find(state) == target_states->end()) {
                                              return false;
                                          }
                                          return state->active_prepare_count == 0 &&
                                                 !state->staged_to_compute_backend &&
                                                 state->residency_mode == ResidencyMode::Disk;
                                      });
        }

        if (can_release) {
            free_params_storage_block(*block);
            it = params_storage_blocks_.erase(it);
        } else {
            ++it;
        }
    }
}

void ModelManager::erase_params_storage_block(ParamsStorageBlock* block) {
    auto it = std::find_if(params_storage_blocks_.begin(),
                           params_storage_blocks_.end(),
                           [block](const std::unique_ptr<ParamsStorageBlock>& item) {
                               return item.get() == block;
                           });
    if (it != params_storage_blocks_.end()) {
        params_storage_blocks_.erase(it);
    }
}

void ModelManager::release_all() {
    for (auto& state : tensor_states_) {
        state->active_prepare_count = 0;
        state->applied_lora_epoch   = UINT64_MAX;
    }
    release_compute_staging_blocks(true);
    release_params_storage_blocks(true);
}

bool ModelManager::resolve_required_tensor_states(const std::vector<ggml_tensor*>& tensors,
                                                  std::vector<TensorState*>& required_states) const {
    required_states.clear();
    std::unordered_set<TensorState*> seen;
    for (ggml_tensor* tensor : tensors) {
        if (tensor == nullptr) {
            continue;
        }
        const char* raw_name = ggml_get_name(tensor);
        if (raw_name == nullptr || raw_name[0] == '\0') {
            LOG_ERROR("model manager unnamed tensor is not registered");
            return false;
        }
        auto state_it = tensor_states_by_name_.find(raw_name);
        if (state_it == tensor_states_by_name_.end()) {
            LOG_ERROR("model manager tensor '%s' is not registered", raw_name);
            return false;
        }
        TensorState* state = state_it->second;
        if (state == nullptr) {
            LOG_ERROR("model manager tensor '%s' has no tensor state", raw_name);
            return false;
        }
        if (seen.insert(state).second) {
            required_states.push_back(state);
        }
    }
    return true;
}

bool ModelManager::assign_compute_backend(const std::vector<ggml_tensor*>& tensors,
                                          ggml_backend_t compute_backend) {
    if (tensors.empty()) {
        return true;
    }
    if (compute_backend == nullptr) {
        LOG_ERROR("model manager cannot assign tensors to a null compute backend");
        return false;
    }

    std::vector<TensorState*> required_states;
    if (!resolve_required_tensor_states(tensors, required_states)) {
        return false;
    }

    bool any_change = false;
    for (TensorState* state : required_states) {
        if (state == nullptr || state->tensor == nullptr) {
            continue;
        }

        const bool params_follow_compute = state->params_follow_compute_backend ||
                                           state->residency_mode == ResidencyMode::Disk;
        const bool compute_changes = state->compute_backend != compute_backend;
        const bool params_changes  = params_follow_compute && state->params_backend != compute_backend;
        if (!compute_changes && !params_changes) {
            continue;
        }

        if (state->active_prepare_count > 0 || state->staged_to_compute_backend) {
            LOG_ERROR("model manager cannot move active tensor '%s' to another compute backend",
                      state->name.c_str());
            return false;
        }
        if (params_changes && state->loaded_to_params_backend) {
            LOG_ERROR("model manager cannot move loaded tensor '%s' to another params backend",
                      state->name.c_str());
            return false;
        }

        any_change = true;
    }

    if (any_change) {
        clear_all_param_prefetches();
    }
    for (TensorState* state : required_states) {
        if (state == nullptr || state->tensor == nullptr) {
            continue;
        }

        const bool params_follow_compute = state->params_follow_compute_backend ||
                                           state->residency_mode == ResidencyMode::Disk;
        state->compute_backend           = compute_backend;
        if (params_follow_compute) {
            state->params_backend = compute_backend;
        }
    }

    return true;
}

bool ModelManager::prepare_params(const std::vector<ggml_tensor*>& tensors) {
    if (tensors.empty()) {
        return true;
    }

    std::vector<TensorState*> required_states;
    if (!resolve_required_tensor_states(tensors, required_states)) {
        return false;
    }

    if (!load_tensors_to_params_backend(required_states)) {
        return false;
    }

    if (!stage_tensors_to_compute_backend(required_states)) {
        release_compute_staging_blocks(false);
        release_params_storage_blocks(false);
        return false;
    }

    if (!apply_loras_to_params(required_states)) {
        release_compute_staging_blocks(false);
        release_params_storage_blocks(false);
        return false;
    }

    for (TensorState* state : required_states) {
        if (state == nullptr) {
            continue;
        }
        state->active_prepare_count++;
    }
    return true;
}

size_t ModelManager::finish_compute_backend_usage(const std::vector<TensorState*>& states) {
    if (states.empty()) {
        return 0;
    }

    std::unordered_set<TensorState*> unique_states;
    for (TensorState* state : states) {
        if (state == nullptr || !unique_states.insert(state).second) {
            continue;
        }
        if (state->active_prepare_count > 0) {
            state->active_prepare_count--;
        }
    }
    return release_compute_staging_blocks(false);
}

size_t ModelManager::release_compute_backend_params(const std::vector<ggml_tensor*>& tensors) {
    if (tensors.empty()) {
        return 0;
    }
    std::vector<TensorState*> required_states;
    if (!resolve_required_tensor_states(tensors, required_states)) {
        return 0;
    }
    return finish_compute_backend_usage(required_states);
}

void ModelManager::release_params_backend_params(const std::vector<ggml_tensor*>& tensors) {
    if (tensors.empty()) {
        return;
    }
    std::vector<TensorState*> required_states;
    if (!resolve_required_tensor_states(tensors, required_states)) {
        return;
    }
    if (required_states.empty()) {
        return;
    }
    std::unordered_set<TensorState*> target_states(required_states.begin(), required_states.end());
    release_params_storage_blocks(false, &target_states);
}
