#include "model_manager.h"

#include "core/backend_fit.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <mutex>
#include <tuple>
#include <unordered_set>

#include "core/ggml_extend_backend.h"
#include "core/util.h"
#include "model/adapter/lora.hpp"

static size_t aligned_offset(const void* buffer, size_t offset, size_t alignment) {
    GGML_ASSERT(alignment != 0 && (alignment & (alignment - 1)) == 0);
    size_t align = (alignment - ((reinterpret_cast<uintptr_t>(buffer) + offset) % alignment)) % alignment;
    return offset + align;
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
            lhs[i].required != rhs[i].required ||
            lhs[i].file_id != rhs[i].file_id || lhs[i].file_revision != rhs[i].file_revision) {
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
    } else if (op == GGML_OP_CUSTOM) {
        op_tensor         = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        op_tensor->op     = op;
        op_tensor->src[0] = weight;
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

ModelManager::~ModelManager() {
    release_all();
    release_prefetch();
}

ModelManager::ResourceId ModelManager::create_resource() {
    const auto id                               = next_resource_id_++;
    resources_.try_emplace(id).first->second.id = id;
    return id;
}

bool ModelManager::remove_resource(ResourceId id) {
    auto found = resources_.find(id);
    if (found == resources_.end() || found->second.applying_loras_)
        return false;
    std::unordered_set<TensorState*> states;
    for (const auto& state : tensor_states_) {
        if (state->resource == &found->second)
            states.insert(state.get());
    }
    if (!unregister_tensor_states(states, nullptr))
        return false;
    resources_.erase(found);
    return true;
}

void ModelManager::set_common_ignore_tensors(std::set<std::string> ignore_tensors) {
    common_ignore_tensors_ = std::move(ignore_tensors);
}

bool ModelManager::prepare_lora_sources(std::vector<LoraSpec>& loras) {
    return prepare_lora_sources(default_resource_, loras);
}

bool ModelManager::set_loras(std::vector<LoraSpec> loras, SDVersion version, std::set<std::string> targets) {
    return set_loras(default_resource_, std::move(loras), version, std::move(targets));
}

std::set<std::string> ModelManager::tensor_names() const {
    return tensor_names(default_resource_);
}

void ModelManager::set_split_buffer_type(ggml_backend_t backend, ggml_backend_buffer_type_t type, const std::vector<std::pair<ggml_backend_t, size_t>>& device_limits) {
    set_split_buffer_type(default_resource_, backend, type, device_limits);
}

bool ModelManager::register_param_tensors(ModelComponent component, std::map<std::string, ggml_tensor*> tensors, ResidencyMode mode, ggml_backend_t compute_backend, ggml_backend_t params_backend, size_t* size, bool allow_split_buffer, bool follow_compute, const std::map<ggml_tensor*, enum ggml_op>* tensor_ops, ModelLoader::FileId source_file, SDVersion source_version) {
    return register_param_tensors(default_resource_, component, std::move(tensors), mode, compute_backend, params_backend,
                                  size, allow_split_buffer, follow_compute, tensor_ops, source_file, source_version);
}

bool ModelManager::unregister_param_tensors(ModelComponent component, size_t* size) {
    return unregister_param_tensors(default_resource_, component, size);
}

bool ModelManager::load_all_params_eagerly() {
    return load_all_params_eagerly(default_resource_);
}

bool ModelManager::validate_registered_tensors() {
    return validate_registered_tensors(default_resource_);
}

bool ModelManager::prepare_lora_sources(Resource& resource, std::vector<LoraSpec>& loras) {
    ModelLoader candidate = resource.model_loader_;
    std::vector<LoraSpec> resolved;
    std::set<ModelLoader::FileId> sources;
    for (auto spec : loras) {
        const std::string prefix = spec.is_high_noise ? "lora.model.high_noise_" : "lora.";
        if (!candidate.add_file(spec.path, prefix, &spec.file_id, false, ModelLoader::FileScope::Isolated)) {
            if (spec.required)
                return false;
            LOG_WARN("cannot register LoRA source '%s'", spec.path.c_str());
            continue;
        }
        spec.file_revision = candidate.file_revision(spec.file_id);
        sources.insert(spec.file_id);
        resolved.push_back(std::move(spec));
    }
    for (auto id : resource.lora_sources_) {
        if (sources.count(id) == 0)
            candidate.del_file(id);
    }
    if (!set_loader(resource, std::move(candidate)))
        return false;
    resource.lora_sources_ = std::move(sources);
    loras                  = std::move(resolved);
    return true;
}

bool ModelManager::set_loras(Resource& resource, std::vector<LoraSpec> loras, SDVersion version, std::set<std::string> targets) {
    if (std::any_of(loras.begin(), loras.end(), [](const LoraSpec& spec) { return spec.file_id == 0; }) &&
        !prepare_lora_sources(resource, loras))
        return false;
    for (auto& spec : loras) {
        spec.file_revision = resource.model_loader_.file_revision(spec.file_id);
        if (spec.file_revision == 0)
            return false;
    }
    if (resource.lora_version_ == version && lora_specs_equal(resource.loras_, loras) &&
        resource.lora_targets_ == targets) {
        resource.loras_ = std::move(loras);
        return true;
    }
    if (!runtime_owners_.empty() || std::any_of(tensor_states_.begin(), tensor_states_.end(), [&](const auto& state) {
            return state->resource == &resource && state->pin_count != 0;
        })) {
        LOG_ERROR("cannot change LoRA configuration during execution");
        return false;
    }
    resource.loras_        = std::move(loras);
    resource.lora_targets_ = std::move(targets);
    resource.lora_version_ = version;
    resource.current_lora_epoch_++;
    resolved_tensor_states_.clear();
    reset_lora_applied_params(resource);
    return true;
}

std::set<std::string> ModelManager::tensor_names(const Resource& resource) const {
    std::set<std::string> names;
    for (const auto& state : tensor_states_) {
        if (state->resource == &resource && state->component != ModelComponent::LoRA &&
            (!resource.shared_weights_key_.empty() || resource.model_sources_.count(state->source_file) == 0)) {
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

void ModelManager::set_split_buffer_type(Resource& resource, ggml_backend_t compute_backend, ggml_backend_buffer_type_t split_buft, const std::vector<std::pair<ggml_backend_t, size_t>>& device_limits) {
    if (compute_backend == nullptr) {
        return;
    }
    if (split_buft == nullptr) {
        resource.split_buffer_types_.erase(compute_backend);
        return;
    }
    resource.split_buffer_types_[compute_backend] = split_buft;
    resource.split_buffer_limits_[split_buft]     = device_limits;
    split_buffer_devices_[split_buft]             = device_limits;
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
    if (!tensor_shape_supports_split_buffer(state.tensor)) {
        return nullptr;
    }
    return state.split_buffer_type;
}

bool ModelManager::register_param_tensors(Resource& resource, ModelComponent component, std::map<std::string, ggml_tensor*> tensors, ResidencyMode residency_mode, ggml_backend_t compute_backend, ggml_backend_t params_backend, size_t* registered_tensor_size, bool allow_split_buffer, bool params_follow_compute_backend, const std::map<ggml_tensor*, enum ggml_op>* tensor_ops, ModelLoader::FileId source_file, SDVersion source_version) {
    if (component == ModelComponent::Count) {
        LOG_ERROR("model manager tensor component is invalid");
        return false;
    }
    if (registered_tensor_size != nullptr) {
        *registered_tensor_size += estimate_tensors_size(tensors);
    }

    const auto scoped_sources = source_file != 0 ? source_loader(resource, source_file).file_tensors(source_file, source_version) : String2TensorStorage{};
    const auto& sources       = source_file != 0 ? scoped_sources : resource.model_loader_.get_tensor_storage_map();
    std::unordered_set<ggml_tensor*> new_tensors;
    std::vector<std::unique_ptr<TensorState>> new_states;
    new_states.reserve(tensors.size());

    for (const auto& pair : tensors) {
        const std::string& name = pair.first;
        ggml_tensor* tensor     = pair.second;
        if (tensor == nullptr) {
            continue;
        }
        if (tensor_states_by_tensor_.count(tensor) != 0 || !new_tensors.insert(tensor).second) {
            LOG_ERROR("model manager tensor name '%s' is already registered", name.c_str());
            return false;
        }
        ggml_set_name(tensor, name.c_str());

        auto state             = std::make_unique<TensorState>();
        state->resource        = &resource;
        state->name            = name;
        state->tensor          = tensor;
        state->component       = component;
        state->source_file     = source_file;
        state->source_version  = source_version;
        state->sources         = find_tensor_sources(*state, sources);
        state->is_int8_weight  = !state->sources.empty() && state->sources.front().is_int8_tensorwise;
        state->residency_mode  = residency_mode;
        state->compute_backend = compute_backend;
        state->params_backend  = params_backend;
        auto split_buffer      = resource.split_buffer_types_.find(compute_backend);
        if (allow_split_buffer && split_buffer != resource.split_buffer_types_.end()) {
            state->split_buffer_type = split_buffer->second;
        }
        state->params_follow_compute_backend = params_follow_compute_backend;
        if (tensor_ops != nullptr) {
            auto op_it = tensor_ops->find(tensor);
            if (op_it != tensor_ops->end()) {
                state->usage_op = op_it->second;
            }
        }
        new_states.push_back(std::move(state));
    }

    std::map<std::string_view, TensorState*> weight_scales;
    for (const auto& state : new_states) {
        if (ends_with(state->name, ".weight_scale"))
            weight_scales.emplace(state->name, state.get());
    }
    for (const auto& state : new_states) {
        if (!state->is_int8_weight)
            continue;
        const auto scale = weight_scales.find(state->name + "_scale");
        if (scale != weight_scales.end()) {
            state->lora_int8_partner         = scale->second;
            scale->second->lora_int8_partner = state.get();
        }
    }

    resolved_tensor_states_.clear();
    for (auto& state : new_states) {
        TensorState* registered_state                      = state.get();
        tensor_states_by_tensor_[registered_state->tensor] = registered_state;
        tensor_states_.push_back(std::move(state));
    }
    return true;
}

bool ModelManager::unregister_param_tensors(Resource& resource, ModelComponent component, size_t* registered_tensor_size) {
    std::unordered_set<TensorState*> states;
    for (auto& state : tensor_states_) {
        if (state->resource == &resource && state->component == component)
            states.insert(state.get());
    }
    return unregister_tensor_states(states, registered_tensor_size);
}

bool ModelManager::unregister_param_tensors(const std::vector<ggml_tensor*>& tensors) {
    return unregister_param_tensors(tensors, nullptr);
}

bool ModelManager::unregister_param_tensors(const std::vector<ggml_tensor*>& tensors, const Resource* resource) {
    std::unordered_set<TensorState*> states;
    for (auto tensor : tensors) {
        auto found = tensor_states_by_tensor_.find(tensor);
        if (found != tensor_states_by_tensor_.end()) {
            if (resource && found->second->resource != resource) {
                LOG_ERROR("cannot unregister a tensor from a different model resource");
                return false;
            }
            states.insert(found->second);
        }
    }
    return unregister_tensor_states(states, nullptr);
}

bool ModelManager::unregister_tensor_states(const std::unordered_set<TensorState*>& target_states,
                                            size_t* registered_tensor_size,
                                            bool keep_registration) {
    size_t released_size = 0;
    for (auto& state : tensor_states_) {
        if (state == nullptr || target_states.count(state.get()) == 0) {
            continue;
        }
        if (state->pin_count > 0) {
            LOG_ERROR("model manager cannot unregister active %s tensor '%s'",
                      model_component_name(state->component),
                      state->name.c_str());
            return false;
        }
        if (state->tensor != nullptr) {
            released_size += ggml_nbytes(state->tensor);
        }
    }

    if (target_states.empty()) {
        return true;
    }

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
        if (!block->shared ||
            std::all_of(block->states.begin(), block->states.end(), [&](auto* state) { return target_states.count(state); }))
            storage_blocks_to_release.push_back(block.get());
        for (TensorState* state : block->states) {
            if (state != nullptr && (!block->shared || target_states.count(state))) {
                affected_storage_states.insert(state);
            }
        }
    }

    for (auto it = prefetch_blocks_.begin(); it != prefetch_blocks_.end();) {
        if (std::any_of(it->second->states.begin(), it->second->states.end(), [&](TensorState* state) {
                return target_states.count(state) || affected_storage_states.count(state);
            })) {
            free_prefetch_block(*it->second);
            it = prefetch_blocks_.erase(it);
        } else {
            ++it;
        }
    }
    // An idle staging block must be released as a whole before unregistering any member.
    auto staging_states = target_states;
    for (const auto& block : compute_staging_blocks_) {
        const bool intersects = std::any_of(block->staged_tensors.begin(), block->staged_tensors.end(), [&](const auto& entry) {
            return entry.first && target_states.count(entry.first);
        });
        const bool idle       = std::all_of(block->staged_tensors.begin(), block->staged_tensors.end(), [](const auto& entry) {
            return !entry.first || entry.first->pin_count == 0;
        });
        if (intersects && idle)
            for (const auto& entry : block->staged_tensors)
                if (entry.first)
                    staging_states.insert(entry.first);
    }
    release_compute_staging_blocks(false, &staging_states);

    for (TensorState* state : affected_storage_states) {
        if (state == nullptr) {
            continue;
        }
        if (state->pin_count > 0 || state->staged_to_compute_backend) {
            LOG_ERROR("model manager cannot unregister %s while tensor '%s' is active",
                      model_component_name(state->component),
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
    for (auto& block : params_storage_blocks_) {
        if (block->shared) {
            auto& states = block->states;
            states.erase(std::remove_if(states.begin(), states.end(), [&](auto* state) {
                             return target_states.count(state) != 0;
                         }),
                         states.end());
        }
    }

    resolved_tensor_states_.clear();
    if (keep_registration) {
        for (auto* state : target_states) {
            state->tensor->buffer           = nullptr;
            state->tensor->data             = nullptr;
            state->tensor->extra            = nullptr;
            state->loaded_to_params_backend = false;
            state->applied_lora_epoch       = UINT64_MAX;
        }
        return true;
    }
    for (auto* state : target_states) {
        if (state->lora_int8_partner) {
            state->lora_int8_partner->lora_int8_partner = nullptr;
            state->lora_int8_partner                    = nullptr;
        }
    }
    for (auto it = tensor_states_by_tensor_.begin(); it != tensor_states_by_tensor_.end();) {
        if (target_states.count(it->second) > 0) {
            it = tensor_states_by_tensor_.erase(it);
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

ModelManager::ResourceMemory ModelManager::resource_memory(const Resource& resource) const {
    ResourceMemory result;
    result.split = resource.layer_split_;
    for (const auto& state : tensor_states_) {
        if (state->resource != &resource || !state->tensor)
            continue;
        const auto bytes = ggml_nbytes(state->tensor);
        result.params_bytes += bytes;
        result.largest_tensor = std::max(result.largest_tensor, bytes);
        if (state->compute_backend && std::find(result.compute_backends.begin(), result.compute_backends.end(), state->compute_backend) == result.compute_backends.end())
            result.compute_backends.push_back(state->compute_backend);
        result.params_backend = state->params_backend;
        result.mode           = state->residency_mode;
        result.split          = result.split || state->split_buffer_type || state->tensor->view_src;
    }
    return result;
}

bool ModelManager::place_resource_params(Resource& resource, ggml_backend_t backend, ResidencyMode mode) {
    if (resource.layer_split_)
        return false;
    std::unordered_set<TensorState*> changed;
    for (const auto& state : tensor_states_)
        if (state->resource == &resource && (state->params_backend != backend || state->residency_mode != mode)) {
            if (state->split_buffer_type || state->tensor->view_src)
                return false;
            changed.insert(state.get());
        }
    if (!changed.empty() && !unregister_tensor_states(changed, nullptr, true))
        return false;
    for (auto* state : changed) {
        state->params_backend                = backend;
        state->residency_mode                = mode;
        state->params_follow_compute_backend = mode == ResidencyMode::Disk;
    }
    return true;
}

size_t ModelManager::next_weight_use(const TensorState& state) const {
    if (planned_next_use_.empty())
        return 0;
    const auto found = planned_next_use_.find(state.resource->shared_weights_key_);
    return found == planned_next_use_.end() ? SIZE_MAX : found->second;
}

bool ModelManager::load_all_params_eagerly(const Resource& resource) {
    std::vector<TensorState*> all_states;
    all_states.reserve(tensor_states_.size());
    for (const auto& s : tensor_states_) {
        if (s->resource == &resource) {
            all_states.push_back(s.get());
        }
    }
    return load_tensors_to_params_backend(all_states);
}

bool ModelManager::validate_registered_tensors(const Resource& resource) {
    bool ok = true;
    std::vector<TensorState*> share;
    for (const auto& state : tensor_states_) {
        if (state->resource != &resource)
            continue;
        bool state_ok = validate_tensor(*state);
        if (state_ok) {
            state->metadata_validated = true;
            if (!resource.shared_weights_key_.empty() && !state->loaded_to_params_backend)
                share.push_back(state.get());
        }
        ok = state_ok && ok;
    }
    if (ok)
        reuse_shared_weights(share);
    return ok;
}

bool ModelManager::validate_param_tensors(const std::vector<ggml_tensor*>& tensors) {
    return validate_param_tensors(tensors, nullptr);
}

bool ModelManager::validate_param_tensors(const std::vector<ggml_tensor*>& tensors, const Resource* resource) {
    std::vector<TensorState*> states;
    if (!resolve_required_tensor_states(tensors, states)) {
        return false;
    }
    for (auto* state : states) {
        if (resource && state->resource != resource) {
            LOG_ERROR("cannot validate a tensor from a different model resource");
            return false;
        }
        if (!validate_tensor(*state)) {
            return false;
        }
        state->metadata_validated = true;
    }
    return true;
}

bool ModelManager::load_tensors_to_params_backend(const std::vector<TensorState*>& states) {
    std::vector<TensorState*> need_load;
    need_load.reserve(states.size());
    for (TensorState* state : states) {
        if (state == nullptr || should_ignore(*state) || (state->sources.empty() && is_optional_missing_tensor(state->name))) {
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
    reuse_shared_weights(need_load);
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

    if (!alloc_params_buffers(need_alloc, states, created_storage_blocks) ||
        !load_tensors(need_load)) {
        for (ParamsStorageBlock* block : created_storage_blocks) {
            if (block != nullptr) {
                free_params_storage_block(*block);
                erase_params_storage_block(block);
            }
        }
        return false;
    }
    struct PrepareStats {
        size_t bytes   = 0;
        size_t tensors = 0;
        size_t blocks  = 0;
    };
    std::map<ggml_backend_buffer_type_t, PrepareStats> prepared;
    for (ParamsStorageBlock* block : created_storage_blocks) {
        if (block != nullptr && block->buffer != nullptr) {
            auto& stats = prepared[ggml_backend_buffer_get_type(block->buffer)];
            stats.bytes += ggml_backend_buffer_get_size(block->buffer);
            stats.tensors += block->states.size();
            ++stats.blocks;
        }
    }
    for (const auto& entry : prepared) {
        LOG_VERBOSE("model manager prepared params backend buffers (%6.2f MB, %zu tensors, %zu blocks, %s) on %s",
                    entry.second.bytes / (1024.f * 1024.f),
                    entry.second.tensors, entry.second.blocks,
                    ggml_backend_buft_is_host(entry.first) ? "RAM" : "VRAM",
                    ggml_backend_buft_name(entry.first));
    }

    std::set<std::string_view> shared_groups;
    bool shared_loras = false;
    for (auto* state : need_load) {
        if (!state->resource->shared_weights_key_.empty()) {
            shared_groups.insert(state->resource->shared_weights_key_);
            shared_loras |= state->component == ModelComponent::LoRA;
        }
    }
    if (!shared_groups.empty()) {
        std::vector<TensorState*> share;
        for (const auto& state : tensor_states_) {
            if (!state->loaded_to_params_backend && state->metadata_validated &&
                (shared_groups.count(state->resource->shared_weights_key_) != 0 ||
                 (shared_loras && state->component == ModelComponent::LoRA)))
                share.push_back(state.get());
        }
        // A sibling can outlive the runner that first read the shared weights.
        reuse_shared_weights(share);
    }

    return true;
}

bool ModelManager::has_lora_int8_pair(const TensorState& state) {
    return state.lora_int8_partner &&
           state.resource->lora_targets_.count(state.is_int8_weight ? state.name : state.lora_int8_partner->name) != 0;
}

bool ModelManager::needs_lora_staging(const TensorState& state) {
    return state.component != ModelComponent::LoRA &&
           (has_lora_int8_pair(state) || (!state.resource->shared_weights_key_.empty() &&
                                          state.resource->lora_targets_.count(state.name) != 0));
}

bool ModelManager::stage_tensors_to_compute_backend(const std::vector<TensorState*>& states) {
    std::map<std::tuple<ggml_backend_t, ggml_backend_buffer_type_t, Resource*, ModelLoader::FileId>, std::vector<TensorState*>> states_by_staging_target;
    for (TensorState* state : states) {
        if (state == nullptr || should_ignore(*state) || (state->sources.empty() && is_optional_missing_tensor(state->name))) {
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
        // Custom CPU operators must retain host weights even when the runner uses a GPU.
        if (state->staged_to_compute_backend ||
            (!needs_lora_staging(*state) &&
             (state->usage_op == GGML_OP_CUSTOM || state->compute_backend == state->params_backend))) {
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
        states_by_staging_target[{state->compute_backend, staging_buft, state->resource, state->source_file}].push_back(state);
    }

    for (const auto& pair : states_by_staging_target) {
        ggml_backend_t compute_backend                 = std::get<0>(pair.first);
        ggml_backend_buffer_type_t staging_buft        = std::get<1>(pair.first);
        const std::vector<TensorState*>& target_states = pair.second;
        if (target_states.empty()) {
            continue;
        }

        const size_t alignment = ggml_backend_buft_get_alignment(staging_buft);
        size_t backend_limit   = ggml_backend_buft_get_max_size(staging_buft);
        if (!ggml_backend_buft_is_host(staging_buft) &&
            (backend_limit == 0 || backend_limit > MAX_RESIDENCY_BLOCK_BYTES)) {
            backend_limit = MAX_RESIDENCY_BLOCK_BYTES;
        }

        const int64_t t0     = ggml_time_ms();
        size_t staged_bytes  = 0;
        size_t staged_blocks = 0;
        auto stage_chunk     = [&](const std::vector<TensorState*>& chunk, size_t bytes) -> bool {
            if (chunk.empty()) {
                return true;
            }
            if (retain_weights_ && ggml_backend_buft_is_host(staging_buft) && !ensure_host_capacity(bytes, states)) {
                return false;
            }
#ifdef _WIN32
            // WDDM device allocations also consume the system commit budget.
            if (retain_weights_ && !ggml_backend_buft_is_host(staging_buft) &&
                sd_backend_is(compute_backend, "CUDA") && !ensure_host_capacity(0, states, false, bytes)) {
                return false;
            }
#endif
            if (retain_weights_ && !ggml_backend_buft_is_host(staging_buft) &&
                (max_vram_.bytes_for_backend(compute_backend) > 0 || std::any_of(chunk.begin(), chunk.end(), [](const auto* state) {
                     return needs_lora_staging(*state) && state->params_backend == state->compute_backend;
                 }))) {
                std::vector<ggml_tensor*> protected_params;
                for (auto* state : states)
                    protected_params.push_back(state->tensor);
                if (!execution_engine()->ensure_capacity(*this, {compute_backend, 0, bytes}, {}, {}, protected_params))
                    return false;
            }
            ggml_init_params init_params;
            init_params.mem_size   = std::max<size_t>(1, chunk.size()) * ggml_tensor_overhead();
            init_params.mem_buffer = nullptr;
            init_params.no_alloc   = true;

            ggml_context* staging_ctx = ggml_init(init_params);
            GGML_ASSERT(staging_ctx != nullptr);
            std::vector<std::pair<TensorState*, ggml_tensor*>> staged_tensors;
            staged_tensors.reserve(chunk.size());
            for (TensorState* state : chunk) {
                ggml_tensor* staging_tensor = ggml_dup_tensor(staging_ctx, state->tensor);
                ggml_set_name(staging_tensor, state->tensor->name);
                staged_tensors.push_back({state, staging_tensor});
            }

            ggml_backend_buffer_t compute_buffer =
                ggml_backend_alloc_ctx_tensors_from_buft(staging_ctx, staging_buft);
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
                state->staged_to_compute_backend = true;
            }
            ggml_backend_synchronize(compute_backend);

            auto block             = std::make_unique<ComputeStagingBlock>();
            block->compute_backend = compute_backend;
            block->buffer          = compute_buffer;
            block->staging_ctx     = staging_ctx;
            block->staged_tensors  = std::move(staged_tensors);
            staged_bytes += ggml_backend_buffer_get_size(compute_buffer);
            ++staged_blocks;
            compute_staging_blocks_.push_back(std::move(block));
            return true;
        };

        std::vector<TensorState*> chunk;
        size_t chunk_size = 0;
        for (TensorState* state : target_states) {
            auto* partner = has_lora_int8_pair(*state) ? state->lora_int8_partner : nullptr;
            if (partner && !state->is_int8_weight)
                continue;
            size_t tensor_size = GGML_PAD(
                ggml_backend_buft_get_alloc_size(staging_buft, state->tensor), alignment);
            if (partner) {
                // Evict and reload the merged weight and its scale together.
                const size_t scale_size = GGML_PAD(ggml_backend_buft_get_alloc_size(staging_buft, partner->tensor), alignment);
                tensor_size += std::min(scale_size, SIZE_MAX - tensor_size);
            }
            if (!chunk.empty() && backend_limit > 0 &&
                tensor_size > backend_limit - std::min(chunk_size, backend_limit)) {
                if (!stage_chunk(chunk, chunk_size)) {
                    return false;
                }
                chunk.clear();
                chunk_size = 0;
            }
            chunk.push_back(state);
            if (partner)
                chunk.push_back(partner);
            chunk_size = tensor_size > SIZE_MAX - chunk_size ? SIZE_MAX : chunk_size + tensor_size;
        }
        if (!stage_chunk(chunk, chunk_size)) {
            return false;
        }
        LOG_VERBOSE("model manager staged compute params (%6.2f MB, %zu tensors, %zu blocks) to %s, taking %.2fs",
                    staged_bytes / (1024.f * 1024.f),
                    target_states.size(),
                    staged_blocks,
                    ggml_backend_name(compute_backend),
                    (ggml_time_ms() - t0) / 1000.f);
    }

    return true;
}

bool ModelManager::apply_loras_to_params(const std::vector<TensorState*>& states) {
    std::map<Resource*, std::vector<TensorState*>> groups;
    for (auto* state : states) {
        if (state && !state->resource->loras_.empty() && !state->resource->applying_loras_)
            groups[state->resource].push_back(state);
    }
    for (const auto& [resource, tensors] : groups) {
        if (!apply_resource_loras(*resource, tensors))
            return false;
    }
    return true;
}

bool ModelManager::apply_resource_loras(Resource& resource, const std::vector<TensorState*>& states) {
    struct ApplyGuard {
        Resource& resource;
        ~ApplyGuard() { resource.applying_loras_ = false; }
    } guard{resource};
    resource.applying_loras_ = true;

    struct LoraApplyGroup {
        std::map<std::string, ggml_tensor*> model_tensors;
        std::map<ggml_tensor*, LoraInt8Weight> int8_weights;
        std::vector<TensorState*> states;
    };

    using ApplyTarget = std::tuple<ggml_backend_t, ggml_backend_t, ResidencyMode>;
    std::map<ApplyTarget, LoraApplyGroup> groups;
    for (TensorState* state : states) {
        if (state == nullptr || state->tensor == nullptr || state->component == ModelComponent::LoRA ||
            should_ignore(*state) || (state->sources.empty() && is_optional_missing_tensor(state->name))) {
            continue;
        }
        if (needs_lora_staging(*state)) {
            if (!state->staged_to_compute_backend)
                return false;
        } else if (!resource.shared_weights_key_.empty()) {
            continue;
        } else if (state->component == ModelComponent::LatentUpsampler || resource.model_sources_.count(state->source_file) != 0) {
            continue;
        }
        if (state->applied_lora_epoch == resource.current_lora_epoch_) {
            continue;
        }
        if (state->compute_backend == nullptr) {
            LOG_ERROR("model manager compute backend is null for lora target tensor '%s'", state->name.c_str());
            return false;
        }
        if (state->tensor->buffer != nullptr &&
            ggml_backend_buffer_get_type(state->tensor->buffer) == split_buffer_type_for(*state)) {
            if (!resource.warned_split_lora_skip_) {
                LOG_WARN(
                    "model manager skipping direct lora application to row-split tensors "
                    "(use --lora-apply-mode at_runtime with row split)");
                resource.warned_split_lora_skip_ = true;
            }
            state->applied_lora_epoch = resource.current_lora_epoch_;
            continue;
        }
        if (state->tensor->data == nullptr) {
            LOG_ERROR("model manager lora target tensor '%s' is not prepared", state->name.c_str());
            return false;
        }
        LoraApplyGroup& group = groups[{state->compute_backend, state->params_backend, state->residency_mode}];
        if (state->is_int8_weight) {
            if (resource.lora_targets_.count(state->name) == 0)
                continue;
            if (!state->lora_int8_partner || !state->lora_int8_partner->staged_to_compute_backend) {
                LOG_ERROR("model manager INT8 LoRA target '%s' has no staged weight scale", state->name.c_str());
                return false;
            }
            const auto& storage               = state->sources.front();
            group.int8_weights[state->tensor] = {state->lora_int8_partner->tensor, storage.int8_convrot ? storage.int8_convrot_group_size : 0};
        } else if (has_lora_int8_pair(*state)) {
            group.states.push_back(state);
            continue;
        }
        group.model_tensors[state->name] = state->tensor;
        group.states.push_back(state);
    }

    if (groups.empty()) {
        return true;
    }

    std::set<std::string> all_tensor_names = tensor_names(resource);
    for (auto& group_pair : groups) {
        ggml_backend_t compute_backend = std::get<0>(group_pair.first);
        LoraApplyGroup& group          = group_pair.second;
        for (const LoraSpec& lora_spec : resource.loras_) {
            if (group.model_tensors.empty()) {
                continue;
            }

            std::string id     = lora_id(lora_spec);
            const auto* target = group.states.front();
            auto lora          = lora_spec.prepared.lock();
            if (lora && (lora->backend() != compute_backend || lora->params_backend != target->params_backend))
                lora.reset();
            const bool temporary = !lora;
            if (temporary) {
                // The temporary runner is destroyed before this manager call returns.
                auto borrowed_manager = std::shared_ptr<ModelManager>(this, [](ModelManager*) {});
                auto weights          = std::make_shared<ResourceHandle>(std::move(borrowed_manager), resource.id);
                lora                  = std::make_shared<LoraModel>(id, compute_backend, target->params_backend,
                                                                    std::move(weights), lora_spec.file_id, resource.lora_version_,
                                                                    target->residency_mode);
            }

            LoraModel::filter_t lora_tensor_filter = nullptr;
            if (!lora_spec.tensor_name_prefix_filter.empty()) {
                lora_tensor_filter = [&](const std::string& tensor_name) {
                    return starts_with(tensor_name, lora_spec.tensor_name_prefix_filter);
                };
            }
            if (temporary && !lora->init_params(resource.n_threads_, lora_tensor_filter)) {
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
            if (!lora->apply(group.model_tensors, all_tensor_names, resource.lora_version_, resource.n_threads_, false, group.int8_weights))
                return false;
            if (temporary)
                lora->release_loaded_tensors();
            else
                lora->runner_end();
        }

        for (TensorState* state : group.states) {
            if (state != nullptr) {
                state->applied_lora_epoch = resource.current_lora_epoch_;
            }
        }
    }
    return true;
}

void ModelManager::reset_lora_applied_params(Resource& resource) {
    std::unordered_set<TensorState*> affected;
    for (auto& state : tensor_states_) {
        if (state->resource == &resource && state->component != ModelComponent::LoRA && state->applied_lora_epoch != UINT64_MAX) {
            affected.insert(state.get());
        }
    }
    if (!resource.shared_weights_key_.empty()) {
        for (auto it = compute_staging_blocks_.begin(); it != compute_staging_blocks_.end();) {
            if (std::any_of((*it)->staged_tensors.begin(), (*it)->staged_tensors.end(), [&](const auto& item) {
                    return affected.count(item.first) != 0;
                })) {
                free_compute_staging_block(**it);
                it = compute_staging_blocks_.erase(it);
            } else {
                ++it;
            }
        }
    } else {
        invalidate_sources(affected);
    }
}

bool ModelManager::should_ignore(const TensorState& state) const {
    if (state.resource != &default_resource_ || state.resource->model_sources_.count(state.source_file) != 0) {
        return false;
    }
    for (const auto& ignore_prefix : common_ignore_tensors_) {
        if (starts_with(state.name, ignore_prefix)) {
            return true;
        }
    }
    return false;
}

bool ModelManager::is_optional_missing_tensor(const std::string& name) const {
    return name.find("cond_stage_model.transformer.text_model.encoder.layers.23.") != std::string::npos ||
           name.find("alphas_cumprod") != std::string::npos;
}

bool ModelManager::validate_tensor(const TensorState& state) const {
    if (state.tensor == nullptr || should_ignore(state) || (state.sources.empty() && is_optional_missing_tensor(state.name))) {
        return true;
    }

    if (state.sources.empty()) {
        LOG_ERROR("%s tensor '%s' not in model metadata", model_component_name(state.component), state.name.c_str());
        return false;
    }

    TensorStorage tensor_storage = state.sources.front();
    if (state.sources.size() > 1) {
        const int dim = tensor_storage.n_dims - 1;
        if (dim < 0 || dim >= GGML_MAX_DIMS) {
            return false;
        }
        tensor_storage.ne[dim] = 0;
        for (const auto& part : state.sources) {
            if (part.n_dims != tensor_storage.n_dims || part.ne[dim] < 0 ||
                part.ne[dim] > state.tensor->ne[dim] - tensor_storage.ne[dim]) {
                LOG_ERROR("invalid tensor part '%s' for '%s'", part.name.c_str(), state.name.c_str());
                return false;
            }
            for (int i = 0; i < GGML_MAX_DIMS; ++i) {
                if (i != dim && part.ne[i] != tensor_storage.ne[i]) {
                    LOG_ERROR("incompatible tensor part '%s' for '%s'", part.name.c_str(), state.name.c_str());
                    return false;
                }
            }
            tensor_storage.ne[dim] += part.ne[dim];
        }
    }
    if (state.tensor->ne[0] != tensor_storage.ne[0] ||
        state.tensor->ne[1] != tensor_storage.ne[1] ||
        state.tensor->ne[2] != tensor_storage.ne[2] ||
        state.tensor->ne[3] != tensor_storage.ne[3]) {
        LOG_ERROR(
            "%s tensor '%s' has wrong shape in model metadata: got [%d, %d, %d, %d], expected [%d, %d, %d, %d]",
            model_component_name(state.component),
            state.name.c_str(),
            (int)tensor_storage.ne[0], (int)tensor_storage.ne[1], (int)tensor_storage.ne[2], (int)tensor_storage.ne[3],
            (int)state.tensor->ne[0], (int)state.tensor->ne[1], (int)state.tensor->ne[2], (int)state.tensor->ne[3]);
        return false;
    }
    return true;
}

bool ModelManager::mmap_params(const std::vector<TensorState*>& states,
                               std::vector<ParamsStorageBlock*>& created_storage_blocks) {
    // A GPU that computes on mmapped params in place cannot address a CPU buffer, and nothing
    // stages them for it, so they are mapped through a buffer of that GPU's device.
    struct MmapGroup {
        std::map<std::string, ggml_tensor*> candidates;
        std::map<std::string, TensorState*> states;
    };
    std::map<std::pair<Resource*, ggml_backend_dev_t>, MmapGroup> groups;
    for (TensorState* state : states) {
        if (state == nullptr || !can_mmap_storage(*state) || state->tensor == nullptr ||
            state->tensor->data != nullptr || state->tensor->view_src != nullptr) {
            continue;
        }
        ggml_backend_dev_t device = nullptr;
        if (!sd_backend_is_cpu(state->compute_backend) && !sd_backend_is_cpu(state->params_backend)) {
            device = ggml_backend_get_device(state->compute_backend);
        }
        MmapGroup& group              = groups[{state->resource, device}];
        group.candidates[state->name] = state->tensor;
        group.states[state->name]     = state;
    }

    for (auto& [target, group] : groups) {
        auto [resource, device] = target;
        // Device buffers wrap read-only mappings only; params that LoRAs are merged into in place
        // are loaded instead.
        if (device != nullptr && resource->writable_mmap_) {
            continue;
        }
        auto mmap_store = resource->model_loader_.mmap_tensors(group.candidates, {}, resource->writable_mmap_, device);
        if (mmap_store.empty()) {
            continue;
        }

        auto block                = std::make_unique<ParamsStorageBlock>();
        block->mmap_tensor_stores = std::move(mmap_store);
        ParamsStorageBlock* raw   = block.get();
        for (const auto& pair : group.states) {
            TensorState* state = pair.second;
            if (state != nullptr && state->tensor != nullptr && state->tensor->data != nullptr) {
                block->states.push_back(state);
            }
        }

        if (!block->states.empty()) {
            params_storage_blocks_.push_back(std::move(block));
            created_storage_blocks.push_back(raw);
        }
    }
    return true;
}

bool ModelManager::can_mmap_storage(const TensorState& state) const {
    if (state.sources.size() > 1 || state.source_file != 0 || !state.resource->enable_mmap_ || state.residency_mode != ResidencyMode::ParamBackend) {
        return false;
    }
    if (state.compute_backend == nullptr || state.params_backend == nullptr) {
        return false;
    }
    if (max_ram_bytes_ != 0 && state.resource->writable_mmap_) {
        return false;
    }
    return sd_backend_is_cpu(state.compute_backend) ||
           sd_backend_is_cpu(state.params_backend) ||
           backend_supports_host_buffer(state.compute_backend);
}

size_t ModelManager::host_weight_bytes() const {
    size_t bytes    = 0;
    auto add        = [&](size_t size) { bytes += std::min(size, SIZE_MAX - bytes); };
    auto add_buffer = [&](ggml_backend_buffer_t buffer) {
        if (buffer && ggml_backend_buffer_is_host(buffer))
            add(ggml_backend_buffer_get_size(buffer));
    };
    for (const auto& block : params_storage_blocks_) {
        add_buffer(block->buffer);
        // Private writable mappings can hold anonymous pages after merging a LoRA.
        if (!block->mmap_tensor_stores.empty() && block->states.front()->resource->writable_mmap_) {
            for (auto* state : block->states)
                add(ggml_nbytes(state->tensor));
        }
    }
    for (const auto& block : compute_staging_blocks_)
        add_buffer(block->buffer);
    for (const auto& entry : prefetch_blocks_) {
        for (const auto& block : entry.second->staging_blocks)
            add_buffer(block->buffer);
    }
    return bytes;
}

bool ModelManager::evict_host_weight_block(const std::unordered_set<TensorState*>& protected_states) {
    auto idle = [&](TensorState* state) {
        return state->pin_count == 0 && !protected_states.count(state);
    };
    std::map<TensorState*, ComputeStagingBlock*> staging_for_tensor;
    auto staging_idle = [&](const ComputeStagingBlock& block) {
        return std::all_of(block.staged_tensors.begin(), block.staged_tensors.end(),
                           [&](const auto& entry) { return idle(entry.first); });
    };
    size_t farthest              = 0;
    uint64_t oldest              = UINT64_MAX;
    ParamsStorageBlock* storage  = nullptr;
    ComputeStagingBlock* staging = nullptr;
    bool prefer_staged           = false;
    auto better                  = [&](size_t next, uint64_t last, bool shared) {
        return next > farthest || (next == farthest &&
                                   ((shared && !prefer_staged) || (shared == prefer_staged && last < oldest)));
    };
    for (const auto& block : compute_staging_blocks_) {
        uint64_t last_use = 0;
        size_t next_use   = planned_next_use_.empty() ? 0 : SIZE_MAX;
        for (const auto& entry : block->staged_tensors) {
            auto* state = entry.first;
            staging_for_tensor.emplace(state, block.get());
            last_use = std::max(last_use, state->last_use_epoch);
            next_use = std::min(next_use, next_weight_use(*state));
        }
        const bool shared = std::any_of(block->staged_tensors.begin(), block->staged_tensors.end(), [](const auto& entry) {
            return !entry.first->resource->shared_weights_key_.empty();
        });
        if (better(next_use, last_use, shared) &&
            ggml_backend_buffer_is_host(block->buffer) && staging_idle(*block)) {
            farthest      = next_use;
            oldest        = last_use;
            staging       = block.get();
            prefer_staged = shared;
        }
    }

    std::unordered_set<ComputeStagingBlock*> dependent_staging;
    for (const auto& block : params_storage_blocks_) {
        if (!block->buffer || !ggml_backend_buffer_is_host(block->buffer))
            continue;
        uint64_t last_use = 0;
        size_t next_use   = planned_next_use_.empty() ? 0 : SIZE_MAX;
        std::unordered_set<ComputeStagingBlock*> dependencies;
        const bool reclaimable = std::all_of(block->states.begin(), block->states.end(), [&](TensorState* state) {
            if (!idle(state) || state->sources.empty() || state->applied_lora_epoch != UINT64_MAX ||
                state->tensor->view_src != nullptr)
                return false;
            last_use = std::max(last_use, state->last_use_epoch);
            next_use = std::min(next_use, next_weight_use(*state));
            if (state->staged_to_compute_backend) {
                auto it = staging_for_tensor.find(state);
                if (it == staging_for_tensor.end())
                    return false;
                dependencies.insert(it->second);
            }
            return true;
        });
        if (reclaimable && better(next_use, last_use, false) &&
            std::all_of(dependencies.begin(), dependencies.end(), [&](const auto* block) { return staging_idle(*block); })) {
            farthest          = next_use;
            oldest            = last_use;
            storage           = block.get();
            prefer_staged     = false;
            dependent_staging = std::move(dependencies);
        }
    }
    if (!storage && !staging)
        return false;
    if (!storage)
        dependent_staging.insert(staging);
    for (auto it = compute_staging_blocks_.begin(); it != compute_staging_blocks_.end();) {
        if (dependent_staging.count(it->get())) {
            LOG_VERBOSE("model manager evicted %.2f MiB of staged weights on %s for RAM",
                        ggml_backend_buffer_get_size((*it)->buffer) / (1024.0 * 1024.0),
                        ggml_backend_name((*it)->compute_backend));
            free_compute_staging_block(**it);
            it = compute_staging_blocks_.erase(it);
        } else {
            ++it;
        }
    }
    if (storage) {
        LOG_VERBOSE("model manager evicted %.2f MiB of weight storage from RAM",
                    ggml_backend_buffer_get_size(storage->buffer) / (1024.0 * 1024.0));
        free_params_storage_block(*storage);
        erase_params_storage_block(storage);
    }
    return true;
}

bool ModelManager::ensure_host_capacity(size_t pending, const std::vector<TensorState*>& required_states, bool weight_allocation, size_t pending_device_commit) {
    constexpr size_t headroom = 1024ULL * 1024 * 1024;
    size_t available          = 0;
    auto fits                 = [&] {
        const size_t resident  = max_ram_bytes_ ? host_weight_bytes() : 0;
        available              = weight_allocation && max_ram_bytes_ ? max_ram_bytes_ - std::min(max_ram_bytes_, resident) : SIZE_MAX;
        const int64_t free_ram = sd::backend_fit::available_ram_bytes();
        if (free_ram >= 0) {
            const auto free_bytes = static_cast<uint64_t>(free_ram);
            available             = static_cast<size_t>(std::min<uint64_t>(
                available, free_bytes > headroom ? free_bytes - headroom : 0));
            if (free_bytes < headroom)
                return false;
        }
        if (pending_device_commit > 0) {
            const int64_t free_commit = sd::backend_fit::available_commit_bytes();
            if (free_commit >= 0) {
                const auto free_bytes = static_cast<uint64_t>(free_commit);
                const auto reserve    = pending_device_commit + std::min(headroom, SIZE_MAX - pending_device_commit);
                if (free_bytes < reserve) {
                    available = 0;
                    return false;
                }
                available = std::min(available, static_cast<size_t>(free_bytes - reserve));
            }
        }
        return pending <= available && (!max_ram_bytes_ || resident <= max_ram_bytes_);
    };
    host_memory_error_.clear();
    if (fits())
        return true;

    if (!weight_allocation || !max_ram_bytes_ || pending <= max_ram_bytes_) {
        std::unordered_set<TensorState*> protected_states(required_states.begin(), required_states.end());
        protect_prefetched_states(protected_states);
        while (evict_host_weight_block(protected_states) || (evict_idle_resource_ && evict_idle_resource_())) {
            if (fits())
                return true;
        }
    }
    const size_t resident = host_weight_bytes();
    host_memory_error_    = std::string(weight_allocation ? "RAM weight capacity exceeded. Requested " : "RAM graph capacity exceeded. Requested ") + std::to_string(pending) +
                            " bytes, " + std::to_string(available) + " bytes available, " +
                            std::to_string(resident) + " bytes resident";
    if (max_ram_bytes_)
        host_memory_error_ += ", max_ram is " + std::to_string(max_ram_bytes_) + " bytes";
    if (pending_device_commit)
        host_memory_error_ += ", pending device commit is " + std::to_string(pending_device_commit) + " bytes";
    LOG_ERROR("%s", host_memory_error_.c_str());
    return false;
}

bool ModelManager::set_max_ram(size_t bytes) {
    const auto previous = max_ram_bytes_;
    max_ram_bytes_      = bytes;
    if (ensure_host_capacity(0, {}))
        return true;
    max_ram_bytes_ = previous;
    return false;
}

bool ModelManager::ensure_host_memory(size_t pending, const std::vector<ggml_tensor*>& protected_params, size_t pending_device_commit) {
    std::vector<TensorState*> states;
    return resolve_required_tensor_states(protected_params, states) && ensure_host_capacity(pending, states, false, pending_device_commit);
}

bool ModelManager::alloc_params_buffers(const std::vector<TensorState*>& states,
                                        const std::vector<TensorState*>& required_states,
                                        std::vector<ParamsStorageBlock*>& created_storage_blocks) {
    std::map<std::tuple<ggml_backend_buffer_type_t, int, Resource*, ModelLoader::FileId>, std::vector<TensorState*>> states_by_buffer_type;
    std::vector<ggml_tensor*> protected_params;
    size_t host_bytes = 0;
    for (TensorState* state : states) {
        if (state == nullptr || state->tensor == nullptr) {
            continue;
        }
        ggml_backend_buffer_type_t params_buft = params_buffer_type_for(*state);
        if (params_buft == nullptr) {
            return false;
        }
        if (retain_weights_ && ggml_backend_buft_is_host(params_buft)) {
            const size_t bytes = GGML_PAD(ggml_backend_buft_get_alloc_size(params_buft, state->tensor),
                                          ggml_backend_buft_get_alignment(params_buft));
            host_bytes += std::min(bytes, SIZE_MAX - host_bytes);
        }
        states_by_buffer_type[{params_buft, static_cast<int>(state->residency_mode), state->resource, state->source_file}].push_back(state);
    }
    if (host_bytes > 0 && !ensure_host_capacity(host_bytes, required_states))
        return false;
    if (retain_weights_) {
        for (auto* state : required_states) {
            if (state)
                protected_params.push_back(state->tensor);
        }
    }

    for (const auto& pair : states_by_buffer_type) {
        ggml_backend_buffer_type_t params_buft  = std::get<0>(pair.first);
        const std::vector<TensorState*>& states = pair.second;
        size_t alignment                        = ggml_backend_buft_get_alignment(params_buft);
        size_t max_size                         = ggml_backend_buft_get_max_size(params_buft);
        if (!ggml_backend_buft_is_host(params_buft) &&
            (max_size == 0 || max_size > MAX_RESIDENCY_BLOCK_BYTES)) {
            max_size = MAX_RESIDENCY_BLOCK_BYTES;
        }

        auto alloc_chunk = [&](const std::vector<TensorState*>& chunk, size_t chunk_size) -> bool {
            if (chunk.empty() || chunk_size == 0) {
                return true;
            }

            std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> cached_buffer(nullptr, ggml_backend_buffer_free);
            if (retain_weights_ && !ggml_backend_buft_is_host(params_buft)) {
                const auto* state      = chunk.front();
                auto backend           = state->residency_mode == ResidencyMode::Disk ? state->compute_backend : state->params_backend;
                const auto& placements = state->resource->split_buffer_limits_;
                auto placement         = placements.find(params_buft);
                const auto devices     = placement == placements.end()
                                             ? std::vector<std::pair<ggml_backend_t, size_t>>{{backend, 0}}
                                             : placement->second;
                if (placement == placements.end() && backend != nullptr &&
                    ggml_backend_get_default_buffer_type(backend) == params_buft) {
                    cached_buffer.reset(sd_backend_try_alloc_buffer_from_cache(params_buft, chunk_size));
                }
                const size_t reused_bytes = cached_buffer ? chunk_size : 0;
                for (const auto& [device, limit] : devices) {
                    if (device && !execution_engine()->ensure_capacity(
                                      *this, {device, 0, chunk_size, 0, limit, reused_bytes}, {}, {}, protected_params)) {
                        LOG_ERROR("model manager cannot fit params buffer on %s", ggml_backend_name(device));
                        return false;
                    }
                }
#ifdef _WIN32
                if (sd_backend_is(backend, "CUDA") && !ensure_host_capacity(0, required_states, false, chunk_size - reused_bytes))
                    return false;
#endif
            }
            ggml_backend_buffer_t buffer = cached_buffer ? cached_buffer.release() : ggml_backend_buft_alloc_buffer(params_buft, chunk_size);
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
                    LOG_VERBOSE("model manager releasing params backend buffer (%6.2f MB, %zu tensors, %s)",
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

bool ModelManager::load_tensor_parts(TensorState& state) {
    auto ctx = std::unique_ptr<ggml_context, decltype(&ggml_free)>(
        ggml_init({state.sources.size() * ggml_tensor_overhead(), nullptr, true}), ggml_free);
    if (!ctx) {
        return false;
    }
    const size_t size = ggml_nbytes(state.tensor);
    std::vector<uint8_t> buffer;
    void* data = state.tensor->data;
    if (!ggml_backend_buffer_is_host(state.tensor->buffer)) {
        buffer.resize(size);
        data = buffer.data();
    }
    std::map<std::string, ggml_tensor*> parts;
    std::set<std::string> names;
    size_t offset = 0;
    for (const auto& source : state.sources) {
        auto part              = ggml_new_tensor(ctx.get(), state.tensor->type, source.n_dims, source.ne);
        const size_t part_size = ggml_nbytes(part);
        if (part_size > size - offset) {
            return false;
        }
        part->data         = static_cast<uint8_t*>(data) + offset;
        parts[source.name] = part;
        names.insert(source.name);
        offset += part_size;
    }
    if (offset != size) {
        return false;
    }
    std::set<std::string> loaded;
    std::mutex mutex;
    auto callback = [&](const TensorStorage& source, ggml_tensor** dst) {
        *dst      = nullptr;
        auto part = parts.find(source.name);
        if (part != parts.end()) {
            *dst = part->second;
            std::lock_guard<std::mutex> lock(mutex);
            loaded.insert(source.name);
        }
        return true;
    };
    auto& resource     = *state.resource;
    const bool success = state.source_file == 0
                             ? resource.model_loader_.load_tensors(callback, resource.enable_mmap_, &names, false)
                             : source_loader(resource, state.source_file).load_file_tensors(state.source_file, state.source_version, callback, names, resource.enable_mmap_);
    if (!success || loaded != names) {
        return false;
    }
    if (!buffer.empty()) {
        // Upload the assembled tensor once, including for row-split backend buffers.
        ggml_backend_tensor_set(state.tensor, buffer.data(), 0, size);
    }
    state.loaded_to_params_backend = true;
    return true;
}

bool ModelManager::load_tensors(const std::vector<TensorState*>& states) {
    using ReadGroup = std::tuple<Resource*, ModelLoader::FileId, SDVersion>;
    using ReadBatch = std::map<std::string, std::vector<ggml_tensor*>>;
    std::map<ReadGroup, std::vector<ReadBatch>> groups;
    size_t part_count = 0;
    for (const auto* state : states) {
        if (state != nullptr && state->sources.size() > 1 && ggml_backend_buffer_is_host(state->tensor->buffer)) {
            part_count += state->sources.size();
        }
    }
    auto parts_ctx = std::unique_ptr<ggml_context, decltype(&ggml_free)>(
        part_count > 0 ? ggml_init({part_count * ggml_tensor_overhead(), nullptr, true}) : nullptr, ggml_free);
    if (part_count > 0 && !parts_ctx) {
        return false;
    }
    auto add_destination = [&](const TensorState& state, const std::string& name, ggml_tensor* tensor) {
        auto& batches = groups[{state.resource, state.source_file, state.source_version}];
        // The loader supplies one destination per name; incompatible destinations need another batch.
        auto batch = std::find_if(batches.begin(), batches.end(), [&](const ReadBatch& candidate) {
            auto found = candidate.find(name);
            return found == candidate.end() ||
                   (found->second.front()->type == tensor->type && ggml_are_same_shape(found->second.front(), tensor));
        });
        if (batch == batches.end()) {
            batches.emplace_back();
            batch = std::prev(batches.end());
        }
        (*batch)[name].push_back(tensor);
    };
    for (auto* state : states) {
        if (state == nullptr)
            continue;
        if (state->sources.size() > 1) {
            if (ggml_backend_buffer_is_host(state->tensor->buffer)) {
                const size_t size = ggml_nbytes(state->tensor);
                size_t offset     = 0;
                for (const auto& source : state->sources) {
                    auto* part             = ggml_new_tensor(parts_ctx.get(), state->tensor->type, source.n_dims, source.ne);
                    const size_t part_size = ggml_nbytes(part);
                    if (part_size > size - offset) {
                        return false;
                    }
                    part->data   = static_cast<uint8_t*>(state->tensor->data) + offset;
                    part->buffer = state->tensor->buffer;
                    add_destination(*state, source.name, part);
                    offset += part_size;
                }
                if (offset != size) {
                    return false;
                }
            } else if (!load_tensor_parts(*state)) {
                return false;
            }
            continue;
        }
        add_destination(*state, state->name, state->tensor);
    }
    for (auto& group : groups) {
        for (auto& batch : group.second) {
            std::set<std::string> names;
            std::set<std::string> loaded;
            std::mutex mutex;
            for (const auto& entry : batch)
                names.insert(entry.first);
            auto callback = [&](const TensorStorage& source, ggml_tensor** dst) {
                *dst       = nullptr;
                auto found = batch.find(source.name);
                if (found == batch.end())
                    return true;
                *dst = found->second.front();
                std::lock_guard<std::mutex> lock(mutex);
                loaded.insert(source.name);
                return true;
            };
            const auto [resource, file, version] = group.first;
            bool success                         = file == 0 ? resource->model_loader_.load_tensors(callback, resource->enable_mmap_, &names)
                                                             : source_loader(*resource, file).load_file_tensors(file, version, callback, names, resource->enable_mmap_);
            if (!success || loaded != names)
                return false;
            for (auto& entry : batch) {
                auto* first = entry.second.front();
                for (auto* tensor : entry.second) {
                    if (tensor != first)
                        ggml_backend_tensor_copy(first, tensor);
                }
            }
        }
    }
    for (auto* state : states) {
        if (state != nullptr)
            state->loaded_to_params_backend = true;
    }
    return true;
}

ggml_backend_buffer_type_t ModelManager::params_buffer_type_for(const TensorState& state) const {
    if (state.params_backend == nullptr) {
        LOG_ERROR("model manager params backend is null for tensor '%s'", state.name.c_str());
        return nullptr;
    }
    ggml_backend_buffer_type_t params_buft = nullptr;
    if (state.compute_backend != nullptr && state.params_backend != state.compute_backend &&
        sd_backend_is_cpu(state.params_backend)) {
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
        ggml_backend_buffer_free(block.buffer);
        block.buffer = nullptr;
    }
    if (block.staging_ctx != nullptr) {
        ggml_free(block.staging_ctx);
        block.staging_ctx = nullptr;
    }
    block.staged_tensors.clear();
}

void ModelManager::protect_prefetched_states(std::unordered_set<TensorState*>& states) const {
    for (const auto& entry : prefetch_blocks_)
        states.insert(entry.second->states.begin(), entry.second->states.end());
}

void ModelManager::release_idle_gpu_staging() {
    std::unordered_set<TensorState*> protected_states;
    protect_prefetched_states(protected_states);
    std::unordered_set<TensorState*> target_states;
    for (const auto& block : compute_staging_blocks_) {
        if (!block->buffer || ggml_backend_buffer_is_host(block->buffer))
            continue;
        // Staging swaps the original storage into the block's tensor metadata.
        for (const auto& [state, original] : block->staged_tensors) {
            if (state && state->loaded_to_params_backend && original && original->data && original->buffer &&
                ggml_backend_buffer_is_host(original->buffer) && !protected_states.count(state))
                target_states.insert(state);
        }
    }
    if (!target_states.empty())
        release_compute_staging_blocks(false, &target_states);
}

void ModelManager::release_compute_staging_blocks(bool force,
                                                  const std::unordered_set<TensorState*>* target_states) {
    struct ReleaseStats {
        size_t bytes   = 0;
        size_t tensors = 0;
        size_t blocks  = 0;
    };
    std::map<ggml_backend_t, ReleaseStats> released;
    for (auto it = compute_staging_blocks_.begin(); it != compute_staging_blocks_.end();) {
        ComputeStagingBlock* block = it->get();
        bool can_release           = force;
        if (!can_release) {
            can_release = std::all_of(block->staged_tensors.begin(),
                                      block->staged_tensors.end(),
                                      [target_states](const std::pair<TensorState*, ggml_tensor*>& pair) {
                                          TensorState* state = pair.first;
                                          if (state == nullptr) {
                                              return true;
                                          }
                                          if (target_states != nullptr &&
                                              target_states->find(state) == target_states->end()) {
                                              return false;
                                          }
                                          return state->pin_count == 0;
                                      });
        }

        if (can_release) {
            if (block->buffer != nullptr) {
                auto& stats = released[block->compute_backend];
                stats.bytes += ggml_backend_buffer_get_size(block->buffer);
                stats.tensors += block->staged_tensors.size();
                ++stats.blocks;
            }
            free_compute_staging_block(*block);
            it = compute_staging_blocks_.erase(it);
        } else {
            ++it;
        }
    }
    for (const auto& entry : released) {
        LOG_DEBUG("model manager released compute params (%6.2f MB, %zu tensors, %zu blocks) from %s",
                  entry.second.bytes / (1024.f * 1024.f),
                  entry.second.tensors, entry.second.blocks,
                  entry.first != nullptr ? ggml_backend_name(entry.first) : "unknown");
    }
}

void ModelManager::free_params_storage_block(ParamsStorageBlock& block) {
    if (block.buffer != nullptr) {
        LOG_DEBUG("free params buffer: %.2f MB(%s), %zu tensors on %s",
                  ggml_backend_buffer_get_size(block.buffer) / (1024.0 * 1024.0),
                  ggml_backend_buffer_is_host(block.buffer) ? "RAM" : "VRAM",
                  block.states.size(), ggml_backend_buffer_name(block.buffer));
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
                                          return state->pin_count == 0 &&
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
    clear_all_prefetched_params();
    runtime_residencies_.clear();
    runtime_owners_.clear();
    for (auto& state : tensor_states_) {
        state->pin_count          = 0;
        state->applied_lora_epoch = UINT64_MAX;
    }
    release_compute_staging_blocks(true);
    release_params_storage_blocks(true);
}

ggml_tensor* ModelManager::resolve_param_tensor(ggml_tensor* tensor) const {
    for (auto* current = tensor; current != nullptr; current = current->view_src) {
        if (tensor_states_by_tensor_.count(current) != 0)
            return current;
    }
    return nullptr;
}

bool ModelManager::resolve_required_tensor_states(const std::vector<ggml_tensor*>& tensors,
                                                  std::vector<TensorState*>& required_states,
                                                  ggml_backend_t compute_backend) const {
    required_states.clear();
    required_states.reserve(tensors.size());
    auto append_states = [&](const std::vector<TensorState*>& states) {
        for (TensorState* state : states) {
            if (compute_backend == nullptr || state->compute_backend == nullptr ||
                state->compute_backend == compute_backend) {
                required_states.push_back(state);
            }
        }
    };
    for (auto it = resolved_tensor_states_.begin(); it != resolved_tensor_states_.end(); ++it) {
        if (it->tensors == tensors) {
            append_states(it->states);
            resolved_tensor_states_.splice(resolved_tensor_states_.begin(), resolved_tensor_states_, it);
            return true;
        }
    }
    std::vector<TensorState*> states;
    states.reserve(tensors.size());
    std::unordered_set<TensorState*> seen;
    seen.reserve(tensors.size());
    bool cacheable = true;
    for (ggml_tensor* tensor : tensors) {
        if (tensor == nullptr) {
            continue;
        }
        auto found = tensor_states_by_tensor_.find(tensor);
        // Unregistered views can be rebound without changing the parameter list.
        cacheable &= found != tensor_states_by_tensor_.end();
        for (auto view = tensor->view_src; found == tensor_states_by_tensor_.end() && view != nullptr; view = view->view_src) {
            found = tensor_states_by_tensor_.find(view);
        }
        if (found == tensor_states_by_tensor_.end()) {
            LOG_ERROR("model manager tensor '%s' is not registered", ggml_get_name(tensor));
            return false;
        }
        TensorState* state = found->second;
        if (seen.insert(state).second) {
            states.push_back(state);
        }
        if (has_lora_int8_pair(*state) && seen.insert(state->lora_int8_partner).second) {
            states.push_back(state->lora_int8_partner);
        }
    }
    append_states(states);
    if (cacheable && !tensors.empty()) {
        static constexpr size_t MAX_RESOLVED_LISTS = 4;
        resolved_tensor_states_.push_front({tensors, std::move(states)});
        if (resolved_tensor_states_.size() > MAX_RESOLVED_LISTS) {
            resolved_tensor_states_.pop_back();
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

    clear_all_prefetched_params();
    for (TensorState* state : required_states) {
        if (state == nullptr || state->tensor == nullptr) {
            continue;
        }

        const bool params_follow_compute = state->params_follow_compute_backend ||
                                           state->residency_mode == ResidencyMode::Disk;
        const bool compute_changes       = state->compute_backend != compute_backend;
        const bool params_changes        = params_follow_compute && state->params_backend != compute_backend;
        if (!compute_changes && !params_changes) {
            continue;
        }

        if (params_changes)
            detach_shared_weight(*state);
        if (retain_weights_ && state->pin_count == 0 &&
            (state->staged_to_compute_backend || (params_changes && state->loaded_to_params_backend))) {
            const auto candidates = evictable_compute_backend_blocks(state->compute_backend, {});
            const auto found      = std::find_if(candidates.begin(), candidates.end(), [&](const auto& block) {
                return std::find(block.tensors.begin(), block.tensors.end(), state->tensor) != block.tensors.end();
            });
            if (found != candidates.end())
                evict_compute_backend_block(*found);
        }
        if (params_changes)
            detach_shared_weight(*state);
        if (state->pin_count > 0 || state->staged_to_compute_backend) {
            LOG_ERROR("model manager cannot move active tensor '%s' to another compute backend",
                      state->name.c_str());
            return false;
        }
        if (params_changes && state->loaded_to_params_backend) {
            LOG_ERROR("model manager cannot move loaded tensor '%s' to another params backend",
                      state->name.c_str());
            return false;
        }

        state->compute_backend = compute_backend;
        if (params_follow_compute) {
            state->params_backend = compute_backend;
        }
    }

    return true;
}

size_t ModelManager::compute_backend_alloc_size(const std::vector<TensorState*>& states,
                                                bool missing_only) const {
    size_t total_size = 0;
    std::unordered_set<TensorState*> seen;
    for (TensorState* state : states) {
        if (state == nullptr || state->tensor == nullptr) {
            continue;
        }
        if (state->usage_op == GGML_OP_CUSTOM && !sd_backend_is_cpu(state->compute_backend) && !needs_lora_staging(*state)) {
            continue;
        }
        const bool compute_resident =
            state->compute_backend == state->params_backend && !needs_lora_staging(*state)
                ? state->loaded_to_params_backend
                : state->staged_to_compute_backend;
        if (missing_only && compute_resident) {
            continue;
        }
        if (!seen.insert(state).second || should_ignore(*state) || (state->sources.empty() && is_optional_missing_tensor(state->name))) {
            continue;
        }

        ggml_backend_buffer_type_t buffer_type = nullptr;
        if (state->compute_backend == state->params_backend && !needs_lora_staging(*state)) {
            buffer_type = params_buffer_type_for(*state);
        } else {
            buffer_type = split_buffer_type_for(*state);
            if (buffer_type == nullptr && state->compute_backend != nullptr) {
                buffer_type = ggml_backend_get_default_buffer_type(state->compute_backend);
            }
        }
        if (buffer_type == nullptr) {
            continue;
        }
        const size_t alignment   = ggml_backend_buft_get_alignment(buffer_type);
        const size_t tensor_size = ggml_backend_buft_get_alloc_size(buffer_type, state->tensor);
        const size_t alloc_size  = GGML_PAD(tensor_size, alignment);
        if (alloc_size > SIZE_MAX - total_size) {
            return SIZE_MAX;
        }
        total_size += alloc_size;
    }
    return total_size;
}

size_t ModelManager::compute_backend_resident_bytes(ggml_backend_t compute_backend) const {
    if (compute_backend == nullptr) {
        return 0;
    }
    ggml_backend_dev_t compute_device = ggml_backend_get_device(compute_backend);
    if (compute_device == nullptr) {
        return 0;
    }

    size_t total_size = 0;
    std::unordered_set<ggml_backend_buffer_t> seen;
    auto add_buffer = [&](ggml_backend_buffer_t buffer) {
        if (buffer == nullptr || ggml_backend_buffer_is_host(buffer) || !seen.insert(buffer).second) {
            return;
        }
        ggml_backend_buffer_type_t buffer_type = ggml_backend_buffer_get_type(buffer);
        auto split_devices                     = split_buffer_devices_.find(buffer_type);
        const bool on_device                   = split_devices == split_buffer_devices_.end()
                                                     ? buffer_type != nullptr && ggml_backend_buft_get_device(buffer_type) == compute_device
                                                     : std::any_of(split_devices->second.begin(), split_devices->second.end(), [&](const auto& entry) {
                                         return ggml_backend_get_device(entry.first) == compute_device;
                                                       });
        if (!on_device) {
            return;
        }
        const size_t buffer_size = ggml_backend_buffer_get_size(buffer);
        total_size               = buffer_size > SIZE_MAX - total_size ? SIZE_MAX : total_size + buffer_size;
    };

    // The loader may retain device mappings after their parameter blocks are released.
    for (auto buffer : default_resource_.model_loader_.get_device_mmap_buffers())
        add_buffer(buffer);
    for (const auto& entry : resources_) {
        for (auto buffer : entry.second.model_loader_.get_device_mmap_buffers())
            add_buffer(buffer);
    }
    for (const auto& block : params_storage_blocks_) {
        if (block != nullptr) {
            add_buffer(block->buffer);
            for (const auto& store : block->mmap_tensor_stores) {
                add_buffer(store.mmbuffer.get());
            }
        }
    }
    for (const auto& block : compute_staging_blocks_) {
        if (block != nullptr) {
            add_buffer(block->buffer);
        }
    }
    for (const auto& entry : prefetch_blocks_) {
        if (entry.second != nullptr) {
            for (const auto& block : entry.second->staging_blocks) {
                if (block != nullptr) {
                    add_buffer(block->buffer);
                }
            }
        }
    }
    return total_size;
}

ModelManager::RAMBudget ModelManager::available_ram_for_fit() const {
    size_t reclaimable = 0;
    auto add           = [&](ggml_backend_buffer_t buffer) {
        if (buffer && ggml_backend_buffer_is_host(buffer))
            reclaimable += std::min(ggml_backend_buffer_get_size(buffer), SIZE_MAX - reclaimable);
    };
    std::map<const TensorState*, bool> staged_idle;
    for (const auto& block : compute_staging_blocks_) {
        const bool idle = std::all_of(block->staged_tensors.begin(), block->staged_tensors.end(), [](const auto& entry) {
            return entry.first->pin_count == 0;
        });
        for (const auto& [state, _] : block->staged_tensors)
            staged_idle[state] = idle;
        if (idle)
            add(block->buffer);
    }
    for (const auto& block : params_storage_blocks_)
        if (block->mmap_tensor_stores.empty() && std::all_of(block->states.begin(), block->states.end(), [&](const auto* state) {
                return state->pin_count == 0 && !state->sources.empty() && state->applied_lora_epoch == UINT64_MAX &&
                       !state->tensor->view_src && (!state->staged_to_compute_backend || (staged_idle.count(state) && staged_idle.at(state)));
            }))
            add(block->buffer);
    const auto resident = host_weight_bytes();
    const auto fixed    = resident - std::min(resident, reclaimable);
    int64_t free        = sd::backend_fit::available_ram_bytes();
    const auto commit   = sd::backend_fit::available_commit_bytes();
    if (commit >= 0)
        free = free < 0 ? commit : std::min(free, commit);
    constexpr size_t reserve = 1024ULL * 1024 * 1024;
    size_t available         = free < 0 ? 0 : size_t(std::max<int64_t>(free - reserve, 0));
    available += std::min(reclaimable, SIZE_MAX - available);
    const auto weights = max_ram_bytes_ ? std::min(available, max_ram_bytes_ > fixed ? max_ram_bytes_ - fixed : 0) : available;
    return {weights, available};
}

size_t ModelManager::available_vram_for_fit(ggml_backend_t backend, size_t free_bytes, size_t total_bytes) const {
    if (backend == nullptr || total_bytes == 0 || free_bytes > total_bytes)
        return free_bytes;
    size_t reclaimable = 0;
    for (const auto& block : evictable_compute_backend_blocks(backend, {}))
        reclaimable += std::min(block.bytes, total_bytes - reclaimable);
    const auto resident  = compute_backend_resident_bytes(backend);
    const auto limit     = max_vram_.bytes_for_backend(backend);
    const auto remaining = resident - std::min(resident, reclaimable) + other_runtime_resident_bytes(0, backend);
    size_t available     = free_bytes + std::min(reclaimable, total_bytes - free_bytes);
    if (limit > 0)
        available = std::min(available, limit > remaining ? limit - remaining : 0);
    return available;
}

void ModelManager::update_runtime_residency(uintptr_t owner_id,
                                            ggml_backend_t compute_backend,
                                            size_t resident_bytes) {
    if (owner_id == 0) {
        return;
    }
    if (compute_backend == nullptr || resident_bytes == 0) {
        runtime_residencies_.erase({owner_id, compute_backend});
        return;
    }
    runtime_residencies_[{owner_id, compute_backend}] = {compute_backend, resident_bytes};
}

size_t ModelManager::other_runtime_resident_bytes(uintptr_t owner_id,
                                                  ggml_backend_t compute_backend) const {
    if (compute_backend == nullptr) {
        return 0;
    }
    ggml_backend_dev_t compute_device = ggml_backend_get_device(compute_backend);
    if (compute_device == nullptr) {
        return 0;
    }
    size_t total_size = 0;
    for (const auto& entry : runtime_residencies_) {
        if (entry.first.first == owner_id || entry.second.compute_backend == nullptr ||
            ggml_backend_get_device(entry.second.compute_backend) != compute_device) {
            continue;
        }
        total_size = entry.second.resident_bytes > SIZE_MAX - total_size
                         ? SIZE_MAX
                         : total_size + entry.second.resident_bytes;
    }
    return total_size;
}

bool ModelManager::prepare_params(const std::vector<ggml_tensor*>& tensors) {
    if (tensors.empty()) {
        return true;
    }

    std::vector<TensorState*> required_states;
    if (!resolve_required_tensor_states(tensors, required_states)) {
        return false;
    }

    for (const auto* state : required_states) {
        if (state != nullptr && state->sources.empty()) {
            LOG_ERROR("required tensor '%s' is missing from model metadata", state->name.c_str());
            return false;
        }
    }

    if (!load_tensors_to_params_backend(required_states)) {
        return false;
    }

    if (!stage_tensors_to_compute_backend(required_states) ||
        (retain_weights_ && !ensure_host_capacity(0, required_states))) {
        const std::unordered_set<TensorState*> failed(required_states.begin(), required_states.end());
        release_compute_staging_blocks(false, &failed);
        release_params_storage_blocks(false, &failed);
        return false;
    }

    // LoRA execution may reclaim other residency blocks while these weights are in use.
    const uint64_t use_epoch = ++residency_epoch_;
    for (TensorState* state : required_states) {
        if (state != nullptr) {
            state->pin_count++;
            state->last_use_epoch = use_epoch;
        }
    }
    if (!apply_loras_to_params(required_states)) {
        finish_compute_backend_usage(required_states);
        std::unordered_set<TensorState*> failed(required_states.begin(), required_states.end());
        invalidate_sources(failed);
        return false;
    }
    return true;
}

void ModelManager::finish_compute_backend_usage(const std::vector<TensorState*>& states) {
    if (states.empty()) {
        return;
    }

    std::unordered_set<TensorState*> target_states;
    for (TensorState* state : states) {
        if (state == nullptr || !target_states.insert(state).second) {
            continue;
        }
        if (state->pin_count > 0) {
            state->pin_count--;
        }
    }
}

void ModelManager::release_compute_backend_params(const std::vector<ggml_tensor*>& tensors) {
    if (tensors.empty()) {
        return;
    }
    std::vector<TensorState*> required_states;
    if (!resolve_required_tensor_states(tensors, required_states)) {
        return;
    }
    finish_compute_backend_usage(required_states);
}

void ModelManager::evict_compute_backend_params(const std::vector<ggml_tensor*>& tensors) {
    if (tensors.empty()) {
        return;
    }
    std::unordered_set<TensorState*> target_states;
    // Session cleanup may follow unregistration, after tensor metadata has been freed.
    for (auto* tensor : tensors) {
        const auto found = tensor_states_by_tensor_.find(tensor);
        if (found != tensor_states_by_tensor_.end() &&
            (!retain_weights_ || found->second->residency_mode == ResidencyMode::Disk)) {
            target_states.insert(found->second);
        }
    }
    if (target_states.empty()) {
        return;
    }

    for (const auto& block : compute_staging_blocks_) {
        const bool intersects = std::any_of(
            block->staged_tensors.begin(),
            block->staged_tensors.end(),
            [&](const std::pair<TensorState*, ggml_tensor*>& pair) {
                return pair.first != nullptr && target_states.count(pair.first) > 0;
            });
        const bool fully_evictable = std::all_of(
            block->staged_tensors.begin(),
            block->staged_tensors.end(),
            [](const std::pair<TensorState*, ggml_tensor*>& pair) {
                return pair.first == nullptr || pair.first->pin_count == 0;
            });
        if (intersects && fully_evictable) {
            for (const auto& pair : block->staged_tensors) {
                if (pair.first != nullptr) {
                    target_states.insert(pair.first);
                }
            }
        }
    }
    release_compute_staging_blocks(false, &target_states);

    for (const auto& block : params_storage_blocks_) {
        const bool intersects = std::any_of(
            block->states.begin(),
            block->states.end(),
            [&](TensorState* state) {
                return state != nullptr && target_states.count(state) > 0;
            });
        const bool fully_evictable = std::all_of(
            block->states.begin(),
            block->states.end(),
            [](TensorState* state) {
                return state == nullptr ||
                       (state->pin_count == 0 && !state->staged_to_compute_backend &&
                        state->residency_mode == ResidencyMode::Disk);
            });
        if (intersects && fully_evictable) {
            target_states.insert(block->states.begin(), block->states.end());
        }
    }
    release_params_storage_blocks(false, &target_states);
}
WeightResidencyInfo ModelManager::inspect_compute_backend_params(
    const std::vector<ggml_tensor*>& tensors) const {
    WeightResidencyInfo info;
    std::vector<TensorState*> states;
    if (!resolve_required_tensor_states(tensors, states)) {
        return info;
    }

    ggml_backend_t prefetch_compute_backend = nullptr;
    bool has_missing_params                 = false;
    bool prefetch_candidate                 = true;
    bool single_prefetch_buffer             = true;
    for (TensorState* state : states) {
        if (state == nullptr || should_ignore(*state) ||
            (state->sources.empty() && is_optional_missing_tensor(state->name))) {
            continue;
        }
        const bool compute_resident =
            state->compute_backend == state->params_backend && !needs_lora_staging(*state)
                ? state->loaded_to_params_backend
                : state->staged_to_compute_backend;
        if (compute_resident) {
            continue;
        }
        has_missing_params = true;
        if (state->usage_op == GGML_OP_CUSTOM || has_lora_int8_pair(*state)) {
            single_prefetch_buffer = false;
        }
        if (split_buffer_type_for(*state) != nullptr) {
            prefetch_candidate = false;
        }
        if (state->compute_backend == state->params_backend ||
            state->compute_backend == nullptr || sd_backend_is_cpu(state->compute_backend)) {
            prefetch_candidate = false;
            continue;
        }
        if (prefetch_compute_backend == nullptr) {
            prefetch_compute_backend = state->compute_backend;
        } else if (prefetch_compute_backend != state->compute_backend) {
            prefetch_candidate = false;
        }
    }
    info.missing_bytes = compute_backend_alloc_size(states, true);
    if (has_missing_params && prefetch_candidate && prefetch_compute_backend != nullptr) {
        ggml_backend_dev_t device = ggml_backend_get_device(prefetch_compute_backend);
        if (device != nullptr) {
            ggml_backend_dev_props props{};
            ggml_backend_dev_get_props(device, &props);
            info.async_prefetch_supported = props.caps.async;
        }
        if (info.async_prefetch_supported && single_prefetch_buffer && info.missing_bytes > 0 &&
            (sd_backend_is(prefetch_compute_backend, "CUDA") || sd_backend_is(prefetch_compute_backend, "ROCm"))) {
            auto buft = ggml_backend_get_default_buffer_type(prefetch_compute_backend);
            if (info.missing_bytes <= std::min(MAX_RESIDENCY_BLOCK_BYTES, ggml_backend_buft_get_max_size(buft))) {
                info.single_prefetch_buffer_type = buft;
            }
        }
    }
    return info;
}

void ModelManager::register_runtime_owner(uintptr_t owner_id) {
    runtime_owners_.insert(owner_id);
}

void ModelManager::remove_runtime_owner(uintptr_t owner_id) {
    runtime_owners_.erase(owner_id);
    for (auto it = runtime_residencies_.begin(); it != runtime_residencies_.end();) {
        if (it->first.first == owner_id) {
            it = runtime_residencies_.erase(it);
        } else {
            ++it;
        }
    }
}

std::map<ggml_backend_t, size_t> ModelManager::split_devices_for(const std::vector<TensorState*>& states) const {
    std::map<ggml_backend_t, size_t> devices;
    for (const auto* state : states) {
        const auto& placements = state->resource->split_buffer_limits_;
        auto placement         = placements.find(split_buffer_type_for(*state));
        if (placement == placements.end())
            continue;
        for (const auto& [backend, limit] : placement->second) {
            auto [it, inserted] = devices.emplace(backend, limit);
            if (!inserted && limit > 0)
                it->second = it->second == 0 ? limit : std::min(it->second, limit);
        }
    }
    return devices;
}

ModelManager::CapacityCheck ModelManager::check_capacity(
    const DeviceMemoryRequest& request,
    const std::vector<TensorState*>& states,
    DeviceMemoryCheck check,
    bool log_details) const {
    CapacityCheck result;
    if (request.compute_backend == nullptr || sd_backend_is_cpu(request.compute_backend)) {
        return result;
    }
    auto add             = [](size_t a, size_t b) { return b > SIZE_MAX - a ? SIZE_MAX : a + b; };
    const size_t missing = compute_backend_alloc_size(states, true);
    // Backend scratch buffers and pipelines are not included in graph measurements.
    size_t safety_margin = 512ULL * 1024ULL * 1024ULL;
#ifdef _WIN32
    if (sd_backend_is(request.compute_backend, "CUDA")) {
        // Avoid WDDM allocation churn when transient buffers compete with reusable staging buffers.
        safety_margin = 1024ULL * 1024ULL * 1024ULL;
    }
#endif
    const size_t prefetch_reused_bytes = request.prefetch_buffer && request.prefetch_buffer->buffer ? request.prefetch_buffer->bytes : 0;
    const size_t reused_bytes          = add(request.reused_buffer_bytes, prefetch_reused_bytes);
    const size_t pending_bytes         = add(request.pending_allocation_bytes, missing);
    result.required_device_bytes       = add(pending_bytes - std::min(pending_bytes, reused_bytes), safety_margin);
    result.required_budget_bytes       = add(request.runtime_peak_bytes(), missing);

    const bool allow_reclamation = check == DeviceMemoryCheck::Reclaimable;
    std::vector<ggml_tensor*> protected_params;
    if (allow_reclamation && retain_weights_) {
        protected_params.reserve(states.size());
        for (const auto* state : states)
            protected_params.push_back(state->tensor);
    }
    auto account_device = [&](ggml_backend_t backend, size_t limit) {
        auto device = ggml_backend_get_device(backend);
        if (device == nullptr)
            return;
        const size_t weights_resident = compute_backend_resident_bytes(backend);
        const size_t other_runtime    = other_runtime_resident_bytes(request.owner_id, backend);
        size_t reclaimable            = 0;
        if (allow_reclamation && retain_weights_) {
            for (const auto& block : evictable_compute_backend_blocks(backend, protected_params))
                reclaimable += std::min(block.bytes, weights_resident - reclaimable);
        }
        const size_t resident = add(weights_resident - reclaimable, other_runtime);

        if (retain_weights_) {
            const auto shared_limit = max_vram_.bytes_for_backend(backend);
            if (shared_limit > 0)
                limit = limit == 0 ? shared_limit : std::min(limit, shared_limit);
        }

        size_t free_bytes = 0, total_bytes = 0, cached_bytes = 0;
        const bool has_cache_accounting = sd_backend_get_buffer_cache_memory(device, free_bytes, total_bytes, cached_bytes);
        if (check == DeviceMemoryCheck::TrimBufferCache) {
            if (has_cache_accounting) {
                const size_t in_use = add(resident, request.runtime_resident_bytes);
                const size_t peak   = add(resident, result.required_budget_bytes);
                // If trimming alone cannot admit a prefetch, an evicted weight buffer may satisfy it without growing the pool.
                const bool wait_for_eviction = request.prefetch_buffer && !request.prefetch_buffer->buffer &&
                                               limit > 0 && peak > limit && add(in_use, cached_bytes) <= limit;
                size_t cache_limit           = limit == 0 ? SIZE_MAX : limit - std::min(limit, wait_for_eviction ? in_use : peak);
                if (!wait_for_eviction && total_bytes > 0 && free_bytes < result.required_device_bytes) {
                    const size_t shortfall = result.required_device_bytes - free_bytes;
                    cache_limit            = std::min(cache_limit, cached_bytes - std::min(cached_bytes, shortfall));
                }
                sd_backend_set_buffer_cache_limit(device, cache_limit);
                if (cached_bytes > cache_limit) {
                    const size_t previous_cached_bytes = cached_bytes;
                    sd_backend_get_buffer_cache_memory(device, free_bytes, total_bytes, cached_bytes);
                    // Virtual CUDA devices may regain only a fraction of the bytes released.
                    if (cached_bytes > 0 && free_bytes < result.required_device_bytes) {
                        sd_backend_set_buffer_cache_limit(device, 0);
                        sd_backend_get_buffer_cache_memory(device, free_bytes, total_bytes, cached_bytes);
                    }
                    LOG_DEBUG("model manager trimmed the idle buffer pool on %s from %.2f MiB to %.2f MiB",
                              ggml_backend_name(backend), previous_cached_bytes / (1024.0 * 1024.0), cached_bytes / (1024.0 * 1024.0));
                }
            } else if (limit > 0) {
                // An older backend may expose reuse without exposing the retained bytes.
                sd_backend_set_buffer_cache_enabled(backend, false);
                ggml_backend_dev_memory(device, &free_bytes, &total_bytes);
            }
        } else if (allow_reclamation && cached_bytes > 0) {
            ggml_backend_dev_memory(device, &free_bytes, &total_bytes);
            cached_bytes = 0;
        }

        const size_t committed = add(resident, cached_bytes);
        if (limit > 0) {
            result.requires_reclamation |= committed > limit;
            result.available_budget_bytes = std::min(result.available_budget_bytes, limit - std::min(limit, committed));
        }
        if (log_details) {
            LOG_WARN("model manager memory on %s: reported free %.2f MiB / total %.2f MiB, tracked weights %.2f MiB / other runtime %.2f MiB / current runtime %.2f MiB / idle buffer pool %.2f MiB",
                     ggml_backend_name(backend),
                     free_bytes / (1024.0 * 1024.0), total_bytes / (1024.0 * 1024.0),
                     weights_resident / (1024.0 * 1024.0), other_runtime / (1024.0 * 1024.0),
                     request.runtime_resident_bytes / (1024.0 * 1024.0), cached_bytes / (1024.0 * 1024.0));
        }
        if (free_bytes == 0 && total_bytes == 0)
            return;
        // Vulkan's heap budget subtraction can underflow when usage exceeds the budget.
        if (total_bytes > 0 && free_bytes > total_bytes && sd_backend_is(backend, "Vulkan")) {
            if (ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_IGPU) {
                result.available_device_bytes = 0;
                return;
            }
            LOG_VERBOSE("ignoring impossible free memory report on %s (%.2f MB free / %.2f MB total), using tracked residency",
                        ggml_backend_name(backend), free_bytes / (1024.0 * 1024.0), total_bytes / (1024.0 * 1024.0));
            free_bytes                   = total_bytes;
            result.wrapped_device_report = true;
        }
        free_bytes = add(free_bytes, reclaimable);
        if (total_bytes > 0) {
            const size_t used = add(committed, add(request.runtime_resident_bytes, reused_bytes));
            free_bytes        = std::min(free_bytes, used < total_bytes ? total_bytes - used : 0);
        }
        result.available_device_bytes = std::min(result.available_device_bytes, free_bytes);
    };
    // GGML exposes only the split buffer's total size, so charge it on each participant.
    auto devices         = split_devices_for(states);
    auto primary         = devices.find(request.compute_backend);
    size_t primary_limit = request.max_backend_bytes;
    if (primary != devices.end()) {
        if (primary->second > 0)
            primary_limit = primary_limit == 0 ? primary->second : std::min(primary_limit, primary->second);
        devices.erase(primary);
    }
    account_device(request.compute_backend, primary_limit);
    for (const auto& [backend, limit] : devices)
        account_device(backend, limit);
    return result;
}

DeviceMemoryCapacity ModelManager::inspect_compute_backend_capacity(
    const DeviceMemoryRequest& request,
    const std::vector<ggml_tensor*>& params,
    DeviceMemoryCheck check,
    bool log_details) const {
    std::vector<TensorState*> states;
    if (request.compute_backend == nullptr || !resolve_required_tensor_states(params, states, request.compute_backend)) {
        DeviceMemoryCapacity result;
        result.valid = false;
        return result;
    }
    return check_capacity(request, states, check, log_details);
}

std::vector<WeightResidencyBlock> ModelManager::evictable_compute_backend_blocks(
    ggml_backend_t backend,
    const std::vector<ggml_tensor*>& protected_params) const {
    std::vector<WeightResidencyBlock> result;
    if (backend == nullptr) {
        return result;
    }
    std::vector<TensorState*> protected_list;
    if (!resolve_required_tensor_states(protected_params, protected_list)) {
        return result;
    }
    std::unordered_set<TensorState*> protected_states(protected_list.begin(), protected_list.end());
    protect_prefetched_states(protected_states);
    const auto device = ggml_backend_get_device(backend);
    const bool cpu    = sd_backend_is_cpu(backend);
    auto on_device    = [&](ggml_backend_buffer_t buffer) {
        if (buffer == nullptr) {
            return false;
        }
        if (cpu) {
            return ggml_backend_buffer_is_host(buffer);
        }
        if (ggml_backend_buffer_is_host(buffer)) {
            return false;
        }
        const auto type  = ggml_backend_buffer_get_type(buffer);
        const auto split = split_buffer_devices_.find(type);
        if (split != split_buffer_devices_.end()) {
            return std::any_of(split->second.begin(), split->second.end(), [&](const auto& entry) {
                return ggml_backend_get_device(entry.first) == device;
            });
        }
        return ggml_backend_buft_get_device(type) == device;
    };
    auto append = [&](ggml_backend_buffer_t buffer, const std::vector<TensorState*>& states, bool storage) {
        if (!on_device(buffer) || states.empty()) {
            return;
        }
        WeightResidencyBlock candidate;
        candidate.buffer   = buffer;
        candidate.backend  = backend;
        candidate.bytes    = ggml_backend_buffer_get_size(buffer);
        candidate.next_use = planned_next_use_.empty() ? 0 : SIZE_MAX;
        for (auto* state : states) {
            if (state == nullptr) {
                continue;
            }
            if (state->pin_count != 0 || protected_states.count(state) != 0) {
                return;
            }
            if (storage) {
                if (state->staged_to_compute_backend) {
                    return;
                }
                if (state->residency_mode != ResidencyMode::Disk) {
                    const bool direct_gpu_storage = !cpu && state->params_backend &&
                                                    !sd_backend_is_cpu(state->params_backend) &&
                                                    state->compute_backend == state->params_backend;
                    if ((!retain_weights_ && !direct_gpu_storage) ||
                        state->sources.empty() || state->applied_lora_epoch != UINT64_MAX ||
                        state->split_buffer_type != nullptr || state->tensor->view_src != nullptr) {
                        return;
                    }
                    candidate.requires_demotion = true;
                }
            }
            candidate.tensors.push_back(state->tensor);
            candidate.next_use       = std::min(candidate.next_use, next_weight_use(*state));
            candidate.last_use_epoch = std::max(candidate.last_use_epoch, state->last_use_epoch);
        }
        result.push_back(std::move(candidate));
    };
    for (const auto& block : compute_staging_blocks_) {
        std::vector<TensorState*> states;
        states.reserve(block->staged_tensors.size());
        for (const auto& entry : block->staged_tensors) {
            states.push_back(entry.first);
        }
        append(block->buffer, states, false);
    }
    for (const auto& block : params_storage_blocks_) {
        // Loader-retained mappings cannot be reclaimed by freeing a storage block.
        if (block->mmap_tensor_stores.empty()) {
            append(block->buffer, block->states, true);
        }
    }
    return result;
}

bool ModelManager::evict_compute_backend_block(const WeightResidencyBlock& block, PrefetchBuffer* prefetch_buffer) {
    const auto candidates = evictable_compute_backend_blocks(block.backend, {});
    const auto found      = std::find_if(candidates.begin(), candidates.end(), [&](const auto& candidate) {
        return candidate.buffer == block.buffer && candidate.tensors == block.tensors &&
               candidate.requires_demotion == block.requires_demotion;
    });
    if (found == candidates.end()) {
        return false;
    }
    for (auto it = compute_staging_blocks_.begin(); it != compute_staging_blocks_.end(); ++it) {
        if ((*it)->buffer == block.buffer) {
            LOG_DEBUG("model manager evicted %.2f MiB of staged weights on %s",
                      block.bytes / (1024.0 * 1024.0), ggml_backend_name(block.backend));
            if (prefetch_buffer != nullptr && !prefetch_buffer->buffer &&
                prefetch_buffer->bytes == block.bytes && prefetch_buffer->type == ggml_backend_buffer_get_type(block.buffer)) {
                // The old weights are unpinned. Hand the allocation to the next upload before it can enter the idle pool.
                prefetch_buffer->buffer.reset((*it)->buffer);
                (*it)->buffer = nullptr;
            }
            free_compute_staging_block(**it);
            compute_staging_blocks_.erase(it);
            return true;
        }
    }
    for (auto it = params_storage_blocks_.begin(); it != params_storage_blocks_.end(); ++it) {
        if ((*it)->buffer == block.buffer) {
            LOG_DEBUG("model manager evicted %.2f MiB of weight storage on %s",
                      block.bytes / (1024.0 * 1024.0), ggml_backend_name(block.backend));
            if (block.requires_demotion && !retain_weights_) {
                for (auto* state : (*it)->states) {
                    if (state != nullptr) {
                        state->residency_mode = ResidencyMode::Disk;
                    }
                }
            }
            free_params_storage_block(**it);
            params_storage_blocks_.erase(it);
            return true;
        }
    }
    return false;
}

bool ModelManager::fits_compute_backend_capacity(
    const DeviceMemoryRequest& request,
    const std::vector<ggml_tensor*>& required_params,
    DeviceMemoryCheck check) const {
    std::vector<TensorState*> states;
    if (!resolve_required_tensor_states(required_params, states, request.compute_backend)) {
        return false;
    }
    return check == DeviceMemoryCheck::Reclaimable
               ? check_capacity(request, states, DeviceMemoryCheck::Inspect).fits() || check_capacity(request, states, check).fits()
               : check_capacity(request, states, check).fits();
}

bool ModelManager::ensure_compute_backend_capacity(
    const DeviceMemoryRequest& request,
    const std::vector<ggml_tensor*>& required_params,
    const std::vector<std::vector<ggml_tensor*>>& preferred_eviction_order,
    const std::vector<ggml_tensor*>& protected_params,
    std::string* device_memory_error) {
    if (retain_weights_) {
        std::vector<TensorState*> states;
        if (!resolve_required_tensor_states(required_params, states, request.compute_backend))
            return false;
        const auto devices = split_devices_for(states);
        if (devices.size() > 1) {
            auto protected_weights = protected_params;
            protected_weights.insert(protected_weights.end(), required_params.begin(), required_params.end());
            const auto missing = compute_backend_alloc_size(states, true);
            for (const auto& [backend, limit] : devices) {
                if (backend == request.compute_backend)
                    continue;
                const auto pending = request.pending_allocation_bytes;
                const DeviceMemoryRequest peer{backend, request.owner_id,
                                               pending > SIZE_MAX - missing ? SIZE_MAX : pending + missing,
                                               request.runtime_resident_bytes, limit};
                if (!execution_engine()->ensure_capacity(*this, peer, {}, preferred_eviction_order, protected_weights, device_memory_error))
                    return false;
            }
        }
    }
    return execution_engine()->ensure_capacity(*this, request, required_params, preferred_eviction_order, protected_params, device_memory_error);
}
