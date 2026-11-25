#include "dgl/models/model_base.h"

#include <stdexcept>
#include <typeinfo>

#include "core/probe/backend_probe.h"
#include "dgl/models/component_sources.h"
#include "dgl/run_events.h"

namespace sd::dgl::models {

    Model::Model(std::shared_ptr<ModelManager> manager)
        : backend_probes_(std::make_shared<BackendProbeCache>()),
          family_probes_(std::make_shared<FamilyProbeCache>()),
          model_manager(std::move(manager)),
          model_weights(std::make_shared<ModelManager::ResourceHandle>(model_manager)) {}

    Model::~Model() {
        release_weights();
    }

    void Model::configure_execution(const ModelExecutionOptions& options) {
        if (options.backend_probes)
            backend_probes_ = options.backend_probes;
        if (options.family_probes)
            family_probes_ = options.family_probes;
        tensor_executor         = options.tensor_executor;
        n_threads               = tensor_executor->num_threads();
        eager_load_             = options.eager_load;
        defer_initialization_   = options.defer_initialization;
        initialization_pending_ = options.defer_initialization;
        model_weights->set_n_threads(n_threads);
        model_weights->set_enable_mmap(options.enable_mmap);
        model_weights->set_prefetch_disabled(options.disable_prefetch);
        model_weights->set_segmented_compute_disabled(options.disable_segmented_compute);
    }

    bool Model::load_weights_eagerly() const {
        if (defer_initialization_ || !eager_load_ || model_weights->load_all_params_eagerly())
            return true;
        LOG_ERROR("model params eager load failed");
        return false;
    }

    Model::ExecutionScope::ExecutionScope(Model& model, bool use_tensor_executor)
        : model_(model), lock_(model.execution_mutex_, std::try_to_lock) {
        if (!lock_.owns_lock() || model_.executing_)
            return;
        model_.executing_ = acquired_ = true;
        try {
            if (model_.tensor_executor)
                tensor_scope_.emplace(model_.tensor_executor.get());
            report_run_progress("preparing");
            ready_ = model_.refresh_weights() && model_.finish_initialization() && model_.apply_loras();
            if (!use_tensor_executor)
                tensor_scope_.reset();
        } catch (...) {
            model_.executing_ = false;
            throw;
        }
    }

    Model::ExecutionScope::~ExecutionScope() {
        if (ready_) {
            model_.end_runners();
            model_.finish_execution();
        }
        if (acquired_)
            model_.executing_ = false;
    }

    void Model::ExecutionScope::require_ready() const {
        if (!ready_)
            throw std::runtime_error("cannot access model component");
    }

    ModelLoader Model::base_loader() const {
        return model_weights->base_loader();
    }

    bool Model::finish_initialization() {
        if (!initialization_pending_)
            return true;
        defer_initialization_ = false;
        if (eager_load_ && eager_load_supported_ && !load_weights_eagerly())
            return false;
        if (!finish_loading())
            return false;
        initialization_pending_ = false;
        return true;
    }

    bool Model::place_params(const std::string& name, ggml_backend_t backend, bool disk) {
        const auto current = model_weights->memory();
        const auto mode    = disk ? ModelManager::ResidencyMode::Disk : ModelManager::ResidencyMode::ParamBackend;
        if (current.params_backend == backend && current.mode == mode)
            return true;
        if (!model_weights->place_params(backend, disk ? ModelManager::ResidencyMode::Disk : ModelManager::ResidencyMode::ParamBackend))
            return false;
        set_weight_adapter(nullptr);
        reset_loras();
        return !params_placement_ || params_placement_(name);
    }

    bool Model::init_variant(const Model& base, std::shared_ptr<const LoraConfig> loras) {
        if (typeid(*this) != typeid(base))
            throw std::invalid_argument("LoRA variant must have the same model type as its base");
        loras_               = std::move(loras);
        family_dependencies_ = base.family_dependencies_;
        return build_variant(base, base.base_loader()) && apply_loras();
    }

    ModelLoader::Dependencies Model::dependencies() const {
        auto result = model_weights->dependencies();
        result.insert(result.end(), family_dependencies_.begin(), family_dependencies_.end());
        return result;
    }

    bool Model::has_loras() const {
        return loras_ && !loras_->bindings.empty();
    }

    lora_apply_mode_t Model::lora_mode() const {
        return loras_ ? loras_->mode : LORA_APPLY_AUTO;
    }

    bool Model::apply_loras() {
        return !has_loras() || attach_loras();
    }

    bool Model::prepare_loras(SDVersion version, SDBackendManager& backends, SDBackendModule module, int n_threads, bool immediately) {
        if (!lora_execution_.prepare(*loras_, model_weights, version, backends, module, n_threads, immediately))
            return false;
        set_weight_adapter(lora_execution_.adapter());
        return true;
    }

    void Model::end_runners() {
        for (const auto& runner : runners_)
            runner.end();
        lora_execution_.end();
    }

    void Model::set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) {
        for (const auto& runner : runners_) {
            if (runner.set_adapter)
                runner.set_adapter(adapter);
        }
    }

    void Model::reset_loras() {
        lora_execution_.reset();
    }

    void Model::release_weights() {
        if (weights_released_)
            return;
        end_runners();
        Model::set_weight_adapter(nullptr);
        reset_loras();
        GGML_ASSERT(model_weights->remove());
        runners_.clear();
        weights_released_ = true;
    }

}  // namespace sd::dgl::models
