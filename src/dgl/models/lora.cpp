#include "dgl/models/lora.h"

#include <string_view>

#include "core/ggml_extend_backend.h"
#include "model/adapter/lora.hpp"

namespace sd::dgl::models {

    static bool is_patch_tensor(std::string_view suffix) {
        const auto index = suffix.find('.');
        if (index != std::string_view::npos && index != 0 &&
            suffix.substr(0, index).find_first_not_of("0123456789") == std::string_view::npos)
            suffix.remove_prefix(index + 1);
        return suffix == "diff" || suffix == "lora_up" || suffix == "lora_down" ||
               suffix.substr(0, 6) == "hada_w" || suffix.substr(0, 6) == "lokr_w";
    }

    void LoraExecution::end() {
        for (const auto& model : models_)
            model->runner_end();
    }

    void LoraExecution::reset() {
        end();
        adapter_.reset();
        models_.clear();
        ready_ = false;
    }

    bool LoraExecution::prepare(const LoraConfig& config, const std::shared_ptr<ModelManager::ResourceHandle>& weights, SDVersion version, SDBackendManager& backends, SDBackendModule module, int n_threads, bool immediately) {
        if (ready_)
            return true;
        std::vector<ModelManager::LoraSpec> specs;
        for (const auto& binding : config.bindings) {
            ModelManager::LoraSpec spec;
            spec.path       = binding.file->path;
            spec.multiplier = binding.weight;
            spec.required   = true;
            specs.push_back(std::move(spec));
        }
        if (!weights->prepare_lora_sources(specs))
            return false;

        const auto backend        = backends.runtime_backend(module);
        const auto params_backend = backends.params_backend(module);
        if (!backend || !params_backend)
            return false;
        const auto residency = backends.params_backend_is_disk(module) ? ModelManager::ResidencyMode::Disk
                                                                       : ModelManager::ResidencyMode::ParamBackend;
        const auto names     = weights->tensor_names();
        std::vector<std::shared_ptr<LoraModel>> models;
        std::set<std::string> targets;
        for (auto& spec : specs) {
            auto model = std::make_shared<LoraModel>(spec.path, backend, params_backend, weights, spec.file_id,
                                                     version, residency, backends.params_backend_follows_runtime(module));
            if (!model->init_params(n_threads))
                return false;
            model->preprocess_lora_tensors(names);
            std::set<std::string_view> keys;
            for (const auto& [name, _] : model->lora_tensors)
                keys.insert(name);
            bool matches = false;
            for (const auto& name : names) {
                const auto prefix = "lora." + name + ".";
                for (auto key = keys.lower_bound(prefix);
                     key != keys.end() && key->substr(0, prefix.size()) == prefix; ++key) {
                    if (is_patch_tensor(key->substr(prefix.size()))) {
                        targets.insert(name);
                        matches = true;
                        break;
                    }
                }
            }
            if (!matches)
                throw std::runtime_error("LoRA '" + spec.path + "' has no matching model weights");
            model->multiplier = spec.multiplier;
            spec.prepared     = model;
            models.push_back(std::move(model));
        }
        if (!weights->set_loras(immediately ? std::move(specs) : std::vector<ModelManager::LoraSpec>{},
                                version, immediately ? std::move(targets) : std::set<std::string>{}))
            return false;
        if (!immediately && !models.empty())
            adapter_ = std::make_shared<MultiLoraAdapter>(models);
        if (!models.empty())
            LOG_INFO("apply lora %s", immediately ? "immediately" : "at runtime");
        models_ = std::move(models);
        ready_  = true;
        return true;
    }

}  // namespace sd::dgl::models
