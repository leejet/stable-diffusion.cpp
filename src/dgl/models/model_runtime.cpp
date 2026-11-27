#include "dgl/models/model_runtime.h"

#include <algorithm>
#include <cmath>
#include <type_traits>

#include "conditioning/conditioner.hpp"
#include "core/ggml_runner.h"
#include "core/ggml_tensor_utils.h"
#include "core/layer_split_partition.h"
#include "core/probe/backend_probe.h"
#include "core/util.h"
#include "dgl/models/component_sources.h"
#include "dgl/models/control.h"
#include "dgl/models/ip_adapter.h"
#include "dgl/models/vae.h"
#include "model/adapter/ip_adapter.hpp"
#include "model/adapter/pmid.hpp"
#include "model/adapter/pulid.hpp"
#include "model/audio/wav2vec2.hpp"
#include "model/diffusion/animatediff.hpp"
#include "model/diffusion/control.hpp"
#include "model/diffusion/z_image.hpp"
#include "model/upscaler/h3_latent_upscaler.hpp"
#include "model/upscaler/ltx_latent_upscaler.hpp"
#include "model/vae/audio_vae.hpp"
#include "model/vae/vae.hpp"
#include "model_definition.h"
#include "name_conversion.h"

namespace sd::dgl::models {

    template <typename T, typename = void>
    struct has_set_runtime_backends : std::false_type {};
    template <typename T>
    struct has_set_runtime_backends<T,
                                    std::void_t<decltype(std::declval<T&>().set_runtime_backends(
                                        std::declval<const std::vector<ggml_backend_t>&>()))>> : std::true_type {};

    ModelRuntime::ModelRuntime(Model& model, std::shared_ptr<SDBackendPool> backends, ModelComponent component, SDBackendModule module, std::function<bool()> build_runners)
        : model_(model),
          build_runners_(std::move(build_runners)),
          component(component),
          module(module),
          backend_manager(std::move(backends)) {
        model_.params_placement_ = [this](const std::string& name) { return set_params_placement(name); };
    }

    ModelRuntime::~ModelRuntime() {
        model_.release_weights();
    }

    bool ModelRuntime::refresh_weights() {
        bool changed;
        if (!model_.model_weights->loader().files_changed(changed, false))
            return false;
        if (!changed && ready_ && catalog_revision_ == model_.model_weights->loader().revision())
            return true;

        ModelLoader candidate = model_.model_weights->loader();
        if (!candidate.refresh_files(false))
            return false;
        ready_ = false;
        model_.end_runners();
        model_.set_weight_adapter(nullptr);
        model_.reset_loras();
        model_.invalidate_cached_state();
        for (auto registered : registered_components_) {
            if (!model_.model_weights->unregister_param_tensors(registered))
                return false;
        }
        registered_components_.clear();
        model_.runners_.clear();
        return model_.model_weights->set_loader(std::move(candidate)) && rebuild();
    }

    ggml_backend_t ModelRuntime::backend_for() {
        ggml_backend_t module_backend = backend_manager.runtime_backend(module);
        if (module_backend == nullptr) {
            LOG_ERROR("failed to initialize %s backend", sd_backend_module_name(module));
        }
        return module_backend;
    }

    ggml_backend_t ModelRuntime::params_backend_for() {
        ggml_backend_t module_backend = backend_manager.params_backend(module);
        if (module_backend == nullptr) {
            LOG_ERROR("failed to initialize %s params backend", sd_backend_module_name(module));
        }
        return module_backend;
    }

    bool ModelRuntime::ensure_backend_pair() {
        if (backend_for() == nullptr) {
            return false;
        }
        return params_backend_for() != nullptr;
    }

    template <typename T>
    bool ModelRuntime::register_runner_params(ModelComponent component,
                                              const std::shared_ptr<T>& model,
                                              size_t* params_mem_size) {
        if (model == nullptr) {
            return true;
        }
        model->set_linear_i8_enabled(params.linear_i8);
        // Keep tensor metadata alive until ModelManager unregisters the resource.
        Model::RunnerRegistration registration{[model] { model->runner_end(); }, {}};
        if constexpr (std::is_base_of_v<GGMLRunner, T> || std::is_same_v<T, Conditioner> ||
                      std::is_same_v<T, DiffusionModelRunner>) {
            registration.set_adapter = [model](std::shared_ptr<WeightAdapter> adapter) {
                model->set_weight_adapter(std::move(adapter));
            };
        }
        model_.runners_.push_back(std::move(registration));
        registered_components_.insert(component);
        std::map<std::string, ggml_tensor*> group_tensors;
        std::map<ggml_tensor*, enum ggml_op> tensor_ops;
        model->get_param_tensors(group_tensors);
        if constexpr (std::is_base_of_v<Conditioner, T>) {
            model->get_param_tensor_ops(tensor_ops);
        }
        ModelManager::ResidencyMode residency_mode =
            backend_manager.params_backend_is_disk(module) ? ModelManager::ResidencyMode::Disk : ModelManager::ResidencyMode::ParamBackend;

        std::vector<ggml_backend_t> module_backends = backend_manager.runtime_backends(module);
        if (module_backends.size() > 1) {
            if constexpr (has_set_runtime_backends<T>::value) {
                if (module == SDBackendModule::DIFFUSION || module == SDBackendModule::TE) {
                    if (backend_manager.split_mode(module) == SDSplitMode::ROW) {
                        return register_row_split_runner_params(component,
                                                                model,
                                                                module_backends,
                                                                std::move(group_tensors),
                                                                tensor_ops,
                                                                residency_mode,
                                                                params_mem_size);
                    }
                    return register_layer_split_runner_params(component,
                                                              model,
                                                              module_backends,
                                                              std::move(group_tensors),
                                                              tensor_ops,
                                                              residency_mode,
                                                              params_mem_size);
                }
            }
            LOG_WARN("%s module does not support multiple runtime backends; using %s",
                     sd_backend_module_name(module),
                     sd::layer_split_backend_device_display_name(module_backends[0]).c_str());
        }
        return model_.model_weights->register_param_tensors(component,
                                                            std::move(group_tensors),
                                                            residency_mode,
                                                            backend_for(),
                                                            params_backend_for(),
                                                            params_mem_size,
                                                            false,
                                                            false,
                                                            &tensor_ops);
    }

    template <typename T>
    bool ModelRuntime::register_row_split_runner_params(ModelComponent component,
                                                        const std::shared_ptr<T>& model,
                                                        const std::vector<ggml_backend_t>& module_backends,
                                                        std::map<std::string, ggml_tensor*> group_tensors,
                                                        const std::map<ggml_tensor*, enum ggml_op>& tensor_ops,
                                                        ModelManager::ResidencyMode residency_mode,
                                                        size_t* params_mem_size) {
        ggml_backend_t main_backend = module_backends[0];

        auto fall_back_to_layer_split = [&](const char* reason) {
            LOG_WARN("%s: row split unavailable (%s); falling back to layer split", model_component_name(component), reason);
            return register_layer_split_runner_params(component,
                                                      model,
                                                      module_backends,
                                                      std::move(group_tensors),
                                                      tensor_ops,
                                                      residency_mode,
                                                      params_mem_size);
        };

        ggml_backend_dev_t main_dev = ggml_backend_get_device(main_backend);
        ggml_backend_reg_t reg      = main_dev != nullptr ? ggml_backend_dev_backend_reg(main_dev) : nullptr;
        if (reg == nullptr) {
            return fall_back_to_layer_split("no backend registry");
        }
        const size_t reg_dev_count = ggml_backend_reg_dev_count(reg);
        std::vector<float> tensor_split(reg_dev_count, 0.0f);
        constexpr int64_t compute_headroom_bytes = 2ll * 1024 * 1024 * 1024;
        for (ggml_backend_t backend : module_backends) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend);
            int reg_index          = -1;
            for (size_t i = 0; i < reg_dev_count; i++) {
                if (ggml_backend_reg_dev_get(reg, i) == dev) {
                    reg_index = (int)i;
                    break;
                }
            }
            if (reg_index < 0) {
                return fall_back_to_layer_split("devices span different backend registries");
            }
            size_t free_bytes = 0, total_bytes = 0;
            ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
            int64_t usable_bytes    = std::max<int64_t>((int64_t)free_bytes - compute_headroom_bytes,
                                                        (int64_t)free_bytes / 8);
            tensor_split[reg_index] = usable_bytes > 0 ? (float)((double)usable_bytes / (1024.0 * 1024.0)) : 1.0f;
        }

        ggml_backend_buffer_type_t split_buft = backend_manager.split_buffer_type(main_backend, tensor_split);
        if (split_buft == nullptr) {
            return fall_back_to_layer_split("backend has no split buffer type");
        }
        std::vector<std::pair<ggml_backend_t, size_t>> split_device_limits;
        for (auto backend : module_backends) {
            split_device_limits.emplace_back(backend, 0);
        }
        model_.model_weights->set_split_buffer_type(main_backend, split_buft, split_device_limits);
        model->set_linear_i8_enabled(false);

        std::map<std::string, ggml_tensor*> split_tensors;
        if constexpr (std::is_base_of_v<Conditioner, T>) {
            model->get_layer_split_param_tensors(split_tensors);
        } else {
            split_tensors = group_tensors;
        }

        std::map<std::string, ggml_tensor*> row_split_map;
        std::map<std::string, ggml_tensor*> regular_map;
        size_t row_split_bytes = 0;
        for (const auto& kv : group_tensors) {
            if (split_tensors.count(kv.first) != 0 &&
                sd::layer_split_tensor_block_index(kv.first) >= 0 &&
                ModelManager::tensor_shape_supports_split_buffer(kv.second)) {
                row_split_map[kv.first] = kv.second;
                row_split_bytes += ggml_nbytes(kv.second);
            } else {
                regular_map[kv.first] = kv.second;
            }
        }
        if (row_split_map.empty()) {
            return fall_back_to_layer_split("no row-splittable transformer block weights found");
        }

        LOG_INFO("%s row split: %zu tensors (%.1f MB) split across %zu devices (main %s)",
                 model_component_name(component),
                 row_split_map.size(),
                 row_split_bytes / (1024.f * 1024.f),
                 module_backends.size(),
                 sd::layer_split_backend_device_display_name(main_backend).c_str());

        if (!model_.model_weights->register_param_tensors(component,
                                                          std::move(row_split_map),
                                                          residency_mode,
                                                          main_backend,
                                                          params_backend_for(),
                                                          params_mem_size,
                                                          /*allow_split_buffer=*/true,
                                                          false,
                                                          &tensor_ops)) {
            return false;
        }
        return model_.model_weights->register_param_tensors(component,
                                                            std::move(regular_map),
                                                            residency_mode,
                                                            main_backend,
                                                            params_backend_for(),
                                                            params_mem_size,
                                                            false,
                                                            false,
                                                            &tensor_ops);
    }

    template <typename T>
    bool ModelRuntime::register_layer_split_runner_params(ModelComponent component,
                                                          const std::shared_ptr<T>& model,
                                                          const std::vector<ggml_backend_t>& module_backends,
                                                          std::map<std::string, ggml_tensor*> group_tensors,
                                                          const std::map<ggml_tensor*, enum ggml_op>& tensor_ops,
                                                          ModelManager::ResidencyMode residency_mode,
                                                          size_t* params_mem_size) {
        bool has_cpu_device = false;
        for (ggml_backend_t backend : module_backends) {
            has_cpu_device = has_cpu_device || sd_backend_is_cpu(backend);
        }
        if (has_cpu_device) {
            // The scheduler reserves the CPU slot for its fallback backend, and
            // CPU weight participation is what --params-backend <module>=cpu is
            // for; a CPU device in a split list is almost certainly a mistake.
            LOG_WARN(
                "%s: layer split across a CPU device is not supported; using %s "
                "(use --params-backend %s=cpu to keep weights in RAM)",
                model_component_name(component),
                sd::layer_split_backend_device_display_name(module_backends[0]).c_str(),
                sd_backend_module_name(module));
            return model_.model_weights->register_param_tensors(component,
                                                                std::move(group_tensors),
                                                                residency_mode,
                                                                module_backends[0],
                                                                params_backend_for(),
                                                                params_mem_size,
                                                                false,
                                                                false,
                                                                &tensor_ops);
        }

        model->set_runtime_backends(module_backends);
        model->set_graph_cut_layer_split_enabled(true);
        model_.model_weights->set_layer_split(true);
        const bool params_follow_runtime      = backend_manager.params_backend_follows_runtime(module) ||
                                                backend_manager.params_backend_is_disk(module);
        ggml_backend_t initial_params_backend = params_follow_runtime ? module_backends[0] : params_backend_for();
        if (initial_params_backend == nullptr) {
            return false;
        }

        LOG_INFO("%s graph-cut layer split: deferring %zu tensors across %zu runtime backends until first graph",
                 model_component_name(component),
                 group_tensors.size(),
                 module_backends.size());

        return model_.model_weights->register_param_tensors(component,
                                                            std::move(group_tensors),
                                                            residency_mode,
                                                            module_backends[0],
                                                            initial_params_backend,
                                                            params_mem_size,
                                                            false,
                                                            params_follow_runtime,
                                                            &tensor_ops);
    }

    bool ModelRuntime::split_active(SDSplitMode mode) {
        return (module == SDBackendModule::DIFFUSION || module == SDBackendModule::TE) &&
               backend_manager.split_mode(module) == mode && backend_manager.runtime_backends(module).size() > 1;
    }

    bool ModelRuntime::configure_backends() {
        std::string error;
        if (!backend_manager.init(backend_spec_.c_str(), params_backend_spec_.c_str(), split_mode_.c_str(), &error)) {
            LOG_ERROR("backend config failed: %s", error.c_str());
            return false;
        }
        return ensure_backend_pair();
    }

    bool ModelRuntime::init(ModelLoadParams load_params, const LoadCatalog& load_catalog) {
        auto family = model_.family_probes_->resolve(load_params.family, load_params.family_from, model_.family_dependencies_);
        params      = std::move(load_params);
#ifdef SD_USE_UPSTREAM_GGML
        LOG_WARN(
            "Using upstream GGML: INT8 tensorwise/convrot is disabled and FP8 weights are "
            "converted to F16 at load time. Some operators may be unsupported and performance "
            "may be lower than with patched GGML.");
#endif
        if (!validate_tensor_types(params.wtype, params.tensor_type_rules.c_str()))
            return false;
        for (float scale : {params.conv_scale, params.linear_scale, params.attn_scale}) {
            if (!std::isfinite(scale) || scale < 0.f || (scale > 0.f && !std::isfinite(1.f / scale))) {
                LOG_ERROR("scale overrides must be finite positive values, or 0 to keep model defaults");
                return false;
            }
        }
        split_mode_          = trim(params.split_mode);
        backend_spec_        = trim(params.backend);
        params_backend_spec_ = trim(params.params_backend);
        ggml_log_set(sd_ggml_log_callback, nullptr);

        sd::ParallelScope tensor_scope(model_.tensor_executor.get());
        ModelLoader loader;
        {
            if (!load_catalog(loader, params, family, *model_.family_probes_))
                return false;
            version = family;
            if (version == VERSION_COUNT) {
                LOG_ERROR("get sd version from file failed: '%s'", params.path.c_str());
                return false;
            }
            LOG_DEBUG("%s family: %s", model_component_name(component), sd_version_name(version));
        }

        if (!configure_backends())
            return false;
        return model_.model_weights->set_loader(std::move(loader)) && rebuild();
    }

    bool ModelRuntime::build_variant(const ModelRuntime& base, ModelLoader loader) {
        params               = base.params;
        version              = base.version;
        backend_spec_        = base.backend_spec_;
        params_backend_spec_ = base.params_backend_spec_;
        split_mode_          = base.split_mode_;
        sd::ParallelScope tensor_scope(model_.tensor_executor.get());
        return configure_backends() && model_.model_weights->set_loader(std::move(loader)) && rebuild();
    }

    bool ModelRuntime::set_params_placement(const std::string& name) {
        std::string error;
        if (!backend_manager.init(backend_spec_.c_str(), name.c_str(), split_mode_.c_str(), &error))
            throw std::runtime_error(error);
        params_backend_spec_ = name;
        if (!ensure_backend_pair())
            return false;
        configure_weight_loading();
        return true;
    }

    void ModelRuntime::configure_weight_loading() {
        model_.model_weights->prepare_file_io();
        if (!model_.has_loras())
            return;
        const auto mode      = model_.lora_mode();
        const auto& loader   = model_.model_weights->loader();
        const bool row_split = mode != LORA_APPLY_AT_RUNTIME && split_active(SDSplitMode::ROW);
        if (mode == LORA_APPLY_IMMEDIATELY) {
            const auto& tensors     = loader.get_tensor_storage_map();
            apply_lora_immediately_ = !row_split &&
                                      std::none_of(tensors.begin(), tensors.end(), [](const auto& entry) {
                                          return entry.second.w4_convrot_kind != 0;
                                      });
        } else {
            apply_lora_immediately_ = resolve_lora_mode(mode, loader, backend_manager, module, row_split) == LORA_APPLY_IMMEDIATELY;
        }
    }

    sd::model_builders::Context ModelRuntime::model_build_context() {
        sd::model_builders::ModelBuildParams build_params;
        build_params.model_args   = params.model_args;
        build_params.linear_scale = params.linear_scale;
        build_params.attn_scale   = params.attn_scale;
        return {std::move(build_params), version, model_.model_weights->loader().get_tensor_storage_map(), backend_manager, model_.model_weights};
    }

    bool ModelRuntime::flash_attention_for() {
        return params.flash_attn_mode == OperatorMode::Auto ? supports_operation(BackendProbeOperation::FlashAttention) : params.flash_attn_mode == OperatorMode::On;
    }

    bool ModelRuntime::conv_direct_for(bool conv3d) {
        return params.conv_direct_mode == OperatorMode::Auto ? supports_operation(conv3d ? BackendProbeOperation::Conv3D : BackendProbeOperation::Conv2D) : params.conv_direct_mode == OperatorMode::On;
    }

    bool ModelRuntime::supports_operation(BackendProbeOperation operation) {
        const auto devices = backend_manager.runtime_backends(module);
        bool enabled       = !devices.empty();
        bool any_enabled   = false;
        for (auto backend : devices) {
            const bool supported = model_.backend_probes_->supports(operation, backend, model_.n_threads);
            enabled              = enabled && supported;
            any_enabled          = any_enabled || supported;
        }
        if (any_enabled && !enabled) {
            static constexpr const char* names[] = {"flash attention", "conv2d direct", "conv3d direct"};
            LOG_INFO("%s auto: %s -> off (compute devices disagree)", names[static_cast<size_t>(operation)], sd_backend_module_name(module));
        }
        return enabled;
    }

    void ModelRuntime::configure_runner(GGMLRunner& runner, bool conv_direct) {
        runner.set_scale_overrides(params.linear_scale, params.attn_scale, params.conv_scale);
        runner.set_flash_attention_enabled(flash_attention_for());
        if (conv_direct) {
            runner.set_conv2d_direct_enabled(conv_direct_for(false));
            runner.set_conv3d_direct_enabled(conv_direct_for(true));
        }
    }

    bool ModelRuntime::rebuild() {
        model_.model_weights->set_layer_split(false);
        configure_weight_loading();
        if (!build_runners_())
            return false;

        if (!model_.model_weights->validate_registered_tensors()) {
            LOG_ERROR("model metadata validation failed");
            return false;
        }
        model_.eager_load_supported_   = !split_active(SDSplitMode::LAYER);
        model_.initialization_pending_ = true;
        if (!model_.defer_initialization_ && !model_.finish_initialization()) {
            return false;
        }
        const auto bytes = model_.model_weights->registered_params_size(registered_components_);
        LOG_INFO("%s weights: %.2f MiB", model_component_name(component), bytes / (1024.0 * 1024.0));
        catalog_revision_ = model_.model_weights->loader().revision();
        ready_            = true;
        return true;
    }

    bool ModelRuntime::attach_loras() {
        if (model_.lora_mode() == LORA_APPLY_IMMEDIATELY && !apply_lora_immediately_)
            throw std::runtime_error("immediately is unsupported by this model's weight layout");
        if (!apply_lora_immediately_ &&
            std::none_of(model_.runners_.begin(), model_.runners_.end(), [](const auto& runner) { return bool(runner.set_adapter); }))
            throw std::runtime_error("at_runtime is unsupported by this model runner");

        sd::ParallelScope tensor_scope(model_.tensor_executor.get());
        if (!model_.prepare_loras(version, backend_manager, module, model_.n_threads, apply_lora_immediately_))
            return false;
        catalog_revision_ = model_.model_weights->loader().revision();
        return true;
    }

    template bool ModelRuntime::register_runner_params(ModelComponent, const std::shared_ptr<Conditioner>&, size_t*);
    template bool ModelRuntime::register_runner_params(ModelComponent, const std::shared_ptr<DiffusionModelRunner>&, size_t*);
    template bool ModelRuntime::register_runner_params(ModelComponent, const std::shared_ptr<::VAE>&, size_t*);
    template bool ModelRuntime::register_runner_params(ModelComponent, const std::shared_ptr<::ControlNet>&, size_t*);
    template bool ModelRuntime::register_runner_params(ModelComponent, const std::shared_ptr<ZImage::ControlNetRunner>&, size_t*);
    template bool ModelRuntime::register_runner_params(ModelComponent, const std::shared_ptr<PhotoMakerIDEncoder>&, size_t*);
    template bool ModelRuntime::register_runner_params(ModelComponent, const std::shared_ptr<PuLIDRunner>&, size_t*);
    template bool ModelRuntime::register_runner_params(ModelComponent, const std::shared_ptr<::AnimateDiff::MotionRunner>&, size_t*);
    template bool ModelRuntime::register_runner_params(ModelComponent, const std::shared_ptr<H3VUpsampler::H3LatentUpscalerRunner>&, size_t*);
    template bool ModelRuntime::register_runner_params(ModelComponent, const std::shared_ptr<LTXVUpsampler::LatentUpsamplerRunner>&, size_t*);
    template bool ModelRuntime::register_runner_params(ModelComponent, const std::shared_ptr<::IPAdapter::IPAdapterRunner>&, size_t*);
    template bool ModelRuntime::register_runner_params(ModelComponent, const std::shared_ptr<FrozenCLIPVisionEmbedder>&, size_t*);
    template bool ModelRuntime::register_runner_params(ModelComponent, const std::shared_ptr<Wav2Vec2::Wav2Vec2ModelRunner>&, size_t*);
    template bool ModelRuntime::register_runner_params(ModelComponent, const std::shared_ptr<AudioVAERunner>&, size_t*);

}  // namespace sd::dgl::models
