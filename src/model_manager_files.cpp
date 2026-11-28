#include "model_manager.h"

#include <algorithm>
#include <string_view>

#include "core/util.h"

static bool same_tensor_encoding(const TensorStorage& a, const TensorStorage& b) {
    return a.offset == b.offset && a.index_in_zip == b.index_in_zip &&
           a.storage_key == b.storage_key && a.type == b.type && a.expected_type == b.expected_type &&
           a.n_dims == b.n_dims && std::equal(a.ne, a.ne + SD_MAX_DIMS, b.ne) &&
           a.is_f8_e4m3 == b.is_f8_e4m3 && a.is_f8_e5m2 == b.is_f8_e5m2 &&
           a.is_f64 == b.is_f64 && a.is_i64 == b.is_i64 &&
           a.is_int8_tensorwise == b.is_int8_tensorwise && a.w4_convrot_kind == b.w4_convrot_kind && a.int8_convrot == b.int8_convrot &&
           a.int8_convrot_group_size == b.int8_convrot_group_size;
}

static bool same_tensor_source(const TensorStorage& a, const TensorStorage& b) {
    return a.file_id == b.file_id && a.file_revision == b.file_revision &&
           a.file_index == b.file_index && same_tensor_encoding(a, b);
}

ModelLoader ModelManager::ResourceHandle::base_loader() const {
    const auto& source = resource();
    auto loader        = source.model_loader_;
    for (auto file : source.lora_sources_)
        loader.del_file(file);
    return loader;
}

ModelLoader::Dependencies ModelManager::ResourceHandle::dependencies() const {
    const auto& source = resource();
    auto result        = source.model_loader_.dependencies();
    for (const auto& [_, loader] : source.model_sources_) {
        auto dependencies = loader.dependencies();
        result.insert(result.end(), dependencies.begin(), dependencies.end());
    }
    return result;
}

void ModelManager::reuse_shared_weights(std::vector<TensorState*>& need_load) {
    if (std::none_of(need_load.begin(), need_load.end(), [](const auto* state) {
            return !state->resource->shared_weights_key_.empty();
        }))
        return;
    std::map<std::pair<std::string_view, std::string_view>, std::vector<std::pair<TensorState*, ParamsStorageBlock*>>> resident;
    auto group = [](const TensorState& state) -> std::string_view {
        return state.component == ModelComponent::LoRA ? std::string_view{} : state.resource->shared_weights_key_;
    };
    std::map<TensorState*, ggml_tensor*> original_params;
    for (const auto& block : compute_staging_blocks_)
        for (const auto& [state, original] : block->staged_tensors)
            original_params.emplace(state, original);
    for (const auto& block : params_storage_blocks_) {
        for (auto* state : block->states) {
            if (!state->resource->shared_weights_key_.empty() && !state->sources.empty())
                resident[{group(*state), state->name}].emplace_back(state, block.get());
        }
    }
    size_t reused = 0;
    need_load.erase(std::remove_if(need_load.begin(), need_load.end(), [&](auto* state) {
                        if (state->resource->shared_weights_key_.empty() || state->sources.empty() ||
                            state->tensor->view_src || split_buffer_type_for(*state))
                            return false;
                        const auto found = resident.find({group(*state), state->name});
                        if (found == resident.end())
                            return false;
                        for (auto [candidate, block] : found->second) {
                            if (candidate->component != state->component || candidate->residency_mode != state->residency_mode ||
                                candidate->tensor->type != state->tensor->type ||
                                !std::equal(state->tensor->ne, state->tensor->ne + GGML_MAX_DIMS, candidate->tensor->ne) ||
                                !std::equal(state->tensor->nb, state->tensor->nb + GGML_MAX_DIMS, candidate->tensor->nb))
                                continue;
                            auto* tensor = candidate->tensor;
                            if (auto original = original_params.find(candidate); original != original_params.end())
                                tensor = original->second;
                            if (!tensor->buffer || !tensor->data ||
                                ggml_backend_buffer_get_type(tensor->buffer) != params_buffer_type_for(*state))
                                continue;
                            const auto& loader = source_loader(*state->resource, state->source_file);
                            const auto& other  = source_loader(*candidate->resource, candidate->source_file);
                            if (!std::equal(state->sources.begin(), state->sources.end(),
                                            candidate->sources.begin(), candidate->sources.end(),
                                            [&](const TensorStorage& a, const TensorStorage& b) {
                                                return same_tensor_encoding(a, b) &&
                                                       loader.same_weight_file(a.file_id, a.file_index, other, b.file_id, b.file_index);
                                            }))
                                continue;
                            block->states.push_back(state);
                            block->shared                   = true;
                            state->tensor->buffer           = tensor->buffer;
                            state->tensor->data             = tensor->data;
                            state->tensor->extra            = tensor->extra;
                            state->loaded_to_params_backend = true;
                            ++reused;
                            return true;
                        }
                        return false;
                    }),
                    need_load.end());
    if (reused)
        LOG_VERBOSE("model manager reused %zu resident weight tensors", reused);
}

void ModelManager::detach_shared_weight(TensorState& state) {
    if (state.resource->shared_weights_key_.empty() || !state.loaded_to_params_backend ||
        state.staged_to_compute_backend || state.pin_count != 0)
        return;
    for (auto& block : params_storage_blocks_) {
        if (!block->shared || block->states.size() < 2)
            continue;
        auto found = std::find(block->states.begin(), block->states.end(), &state);
        if (found == block->states.end())
            continue;
        block->states.erase(found);
        state.tensor->buffer           = nullptr;
        state.tensor->data             = nullptr;
        state.tensor->extra            = nullptr;
        state.loaded_to_params_backend = false;
        state.applied_lora_epoch       = UINT64_MAX;
        return;
    }
}

std::vector<TensorStorage> ModelManager::find_tensor_sources(const TensorState& state, const String2TensorStorage& sources) {
    auto first = sources.find(state.name);
    if (first == sources.end()) {
        return {};
    }
    std::vector<TensorStorage> result{first->second};
    if (state.component == ModelComponent::LoRA ||
        std::equal(first->second.ne, first->second.ne + GGML_MAX_DIMS, state.tensor->ne)) {
        return result;
    }
    for (size_t i = 1;; ++i) {
        auto part = sources.find(state.name + "." + std::to_string(i));
        if (part == sources.end()) {
            break;
        }
        result.push_back(part->second);
    }
    return result;
}

void ModelManager::invalidate_sources(const std::unordered_set<TensorState*>& states) {
    auto affected = states;
    for (const auto& block : params_storage_blocks_) {
        if (std::any_of(block->states.begin(), block->states.end(), [&](TensorState* state) { return states.count(state) != 0; })) {
            affected.insert(block->states.begin(), block->states.end());
        }
    }
    for (auto it = prefetch_blocks_.begin(); it != prefetch_blocks_.end();) {
        if (std::any_of(it->second->states.begin(), it->second->states.end(), [&](TensorState* state) { return affected.count(state) != 0; })) {
            free_prefetch_block(*it->second);
            it = prefetch_blocks_.erase(it);
        } else {
            ++it;
        }
    }
    for (auto it = compute_staging_blocks_.begin(); it != compute_staging_blocks_.end();) {
        if (std::any_of((*it)->staged_tensors.begin(), (*it)->staged_tensors.end(), [&](const auto& entry) { return affected.count(entry.first) != 0; })) {
            ggml_backend_synchronize((*it)->compute_backend);
            free_compute_staging_block(**it);
            it = compute_staging_blocks_.erase(it);
        } else {
            ++it;
        }
    }
    for (auto it = params_storage_blocks_.begin(); it != params_storage_blocks_.end();) {
        if (std::any_of((*it)->states.begin(), (*it)->states.end(), [&](TensorState* state) { return affected.count(state) != 0; })) {
            free_params_storage_block(**it);
            it = params_storage_blocks_.erase(it);
        } else {
            ++it;
        }
    }
    for (auto* state : affected) {
        state->metadata_validated = false;
        state->applied_lora_epoch = UINT64_MAX;
    }
}

bool ModelManager::set_loader(ModelLoader loader) {
    return set_loader(default_resource_, std::move(loader));
}

bool ModelManager::del_file(ModelLoader::FileId file) {
    return del_file(default_resource_, file);
}

ModelLoader::FileId ModelManager::register_model_source(ModelLoader loader) {
    return register_model_source(default_resource_, std::move(loader));
}

bool ModelManager::unregister_model_source(ModelLoader::FileId file) {
    return unregister_model_source(default_resource_, file);
}

size_t ModelManager::registered_params_size(const std::set<ModelComponent>& components) const {
    return registered_params_size(default_resource_, components);
}

bool ModelManager::set_loader(Resource& resource, ModelLoader loader) {
    if (!runtime_owners_.empty() || std::any_of(tensor_states_.begin(), tensor_states_.end(), [&](const auto& state) {
            return state->resource == &resource && state->pin_count != 0;
        })) {
        LOG_ERROR("cannot update model sources during execution");
        return false;
    }
    std::map<std::pair<ModelLoader::FileId, SDVersion>, String2TensorStorage> scoped;
    auto sources_for = [&](const TensorState& state) -> const String2TensorStorage& {
        if (state.source_file == 0)
            return loader.get_tensor_storage_map();
        auto key   = std::make_pair(state.source_file, state.source_version);
        auto found = scoped.find(key);
        if (found == scoped.end())
            found = scoped.emplace(key, resource.model_sources_.count(key.first) != 0
                                            ? source_loader(resource, key.first).file_tensors(key.first, key.second)
                                            : loader.file_tensors(key.first, key.second))
                        .first;
        return found->second;
    };
    bool lora_changed = false;
    for (const auto& spec : resource.loras_) {
        lora_changed |= loader.file_revision(spec.file_id) != spec.file_revision;
    }
    std::unordered_set<TensorState*> changed;
    for (const auto& state : tensor_states_) {
        if (state->resource != &resource)
            continue;
        const auto sources = find_tensor_sources(*state, sources_for(*state));
        if (sources.size() != state->sources.size() ||
            !std::equal(sources.begin(), sources.end(), state->sources.begin(), same_tensor_source) ||
            (lora_changed && state->component != ModelComponent::LoRA && state->applied_lora_epoch != UINT64_MAX)) {
            changed.insert(state.get());
        }
    }
    invalidate_sources(changed);
    for (auto* state : changed) {
        state->sources = find_tensor_sources(*state, sources_for(*state));
    }
    if (lora_changed) {
        ++resource.current_lora_epoch_;
        for (auto& spec : resource.loras_)
            spec.file_revision = loader.file_revision(spec.file_id);
    }
    resource.model_loader_ = std::move(loader);
    resource.model_loader_.set_n_threads(resource.n_threads_);
    return true;
}

bool ModelManager::add_file(const std::string& path, const std::string& prefix, ModelLoader::FileId* id, bool force) {
    const auto& resource  = default_resource_;
    ModelLoader candidate = resource.model_loader_;
    ModelLoader::FileId added_id;
    if (!candidate.add_file(path, prefix, &added_id, force) || !set_loader(default_resource_, std::move(candidate))) {
        return false;
    }
    if (id != nullptr) {
        *id = added_id;
    }
    return true;
}

bool ModelManager::del_file(Resource& resource, ModelLoader::FileId id) {
    ModelLoader candidate = resource.model_loader_;
    return candidate.del_file(id) && set_loader(resource, std::move(candidate));
}

bool ModelManager::refresh_files() {
    const auto& resource  = default_resource_;
    ModelLoader candidate = resource.model_loader_;
    return candidate.refresh_files() && set_loader(default_resource_, std::move(candidate));
}

ModelLoader::FileVersions ModelManager::source_versions(const std::set<ModelComponent>& components, const ModelLoader& loader) const {
    const auto& resource = default_resource_;
    ModelLoader::FileVersions versions;
    const auto& sources = loader.get_tensor_storage_map();
    for (const auto& state : tensor_states_) {
        if (state->resource != &resource || components.count(state->component) == 0) {
            continue;
        }
        if (state->source_file != 0) {
            versions[state->source_file] = resource.model_sources_.count(state->source_file) != 0
                                               ? source_loader(resource, state->source_file).file_revision(state->source_file)
                                               : loader.file_revision(state->source_file);
            continue;
        }
        for (const auto& source : find_tensor_sources(*state, sources)) {
            versions[source.file_id] = source.file_revision;
        }
    }
    return versions;
}

size_t ModelManager::registered_params_size(const Resource& resource, const std::set<ModelComponent>& components) const {
    size_t bytes = 0;
    std::unordered_set<const ggml_tensor*> seen;
    for (const auto& state : tensor_states_) {
        if (state->resource == &resource && components.count(state->component) != 0 && state->tensor != nullptr && seen.insert(state->tensor).second) {
            bytes += ggml_nbytes(state->tensor);
        }
    }
    return bytes;
}

const ModelLoader& ModelManager::source_loader(const Resource& resource, ModelLoader::FileId file) {
    auto found = resource.model_sources_.find(file);
    return found == resource.model_sources_.end() ? resource.model_loader_ : found->second;
}

ModelLoader::FileId ModelManager::register_model_source(Resource& resource, ModelLoader loader) {
    const auto versions = loader.file_versions();
    if (versions.size() != 1) {
        LOG_ERROR("a model source must contain exactly one root model file");
        return 0;
    }
    const auto file = versions.begin()->first;
    if (resource.model_loader_.file_revision(file) != 0 || resource.model_sources_.count(file) != 0) {
        LOG_ERROR("model source is already registered");
        return 0;
    }
    loader.set_n_threads(resource.n_threads_);
    resource.model_sources_.emplace(file, std::move(loader));
    return file;
}

bool ModelManager::unregister_model_source(Resource& resource, ModelLoader::FileId file) {
    if (std::any_of(tensor_states_.begin(), tensor_states_.end(), [&](const auto& state) {
            return state->resource == &resource && state->source_file == file;
        })) {
        LOG_ERROR("cannot unregister a model source while its weights are registered");
        return false;
    }
    return resource.model_sources_.erase(file) != 0;
}
