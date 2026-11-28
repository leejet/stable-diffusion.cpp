#include "dgl/ops/predictor.h"

#include <algorithm>

#include "dgl/models/animatediff.h"
#include "dgl/models/circular_axes.h"
#include "dgl/models/control.h"
#include "dgl/models/dit.h"
#include "dgl/models/fwd.h"
#include "dgl/models/ip_adapter.h"
#include "dgl/models/pulid.h"
#include "dgl/ops/spectrum_cache.h"
#include "runtime/sample-cache.h"

namespace sd::dgl::ops {
    float GuidanceBranches::scale_at(int step) const {
        const auto index = static_cast<size_t>(step - 1);
        return index < scale_schedule.size() ? scale_schedule[index] : scale;
    }

    bool StepWindow::active(const models::DiffusionPredictionStep& step) const {
        const auto size  = static_cast<float>(step.schedule_size);
        const auto start = static_cast<int>(start_fraction * size);
        const auto stop  = static_cast<int>(stop_fraction * size);
        return step.index > start && step.index < stop;
    }

    bool SkipLayerPredictor::active(float sigma, const models::DiffusionPredictionStep& step) const {
        return scale != 0.f && sigmas.active(sigma) && (!steps || steps->active(step));
    }

    static const GuidanceBranches* guidance_branches(const Predictor& predictor) {
        if (const auto* cfg = std::get_if<CfgPredictor>(&predictor.node))
            return &cfg->branches;
        if (const auto* apg = std::get_if<ApgPredictor>(&predictor.node))
            return &apg->branches;
        return nullptr;
    }

    static bool provides_unconditional(const Predictor& predictor, int first_step, int last_step) {
        if (const auto* branches = guidance_branches(predictor))
            return branches->negative && !branches->image_only;
        if (const auto* slg = std::get_if<SkipLayerPredictor>(&predictor.node))
            return provides_unconditional(*slg->base, first_step, last_step);
        if (const auto* transition = std::get_if<SwitchPredictor>(&predictor.node)) {
            if (transition->after_steps < first_step)
                return provides_unconditional(*transition->after, first_step, last_step);
            if (transition->after_steps >= last_step)
                return provides_unconditional(*transition->before, first_step, last_step);
            return provides_unconditional(*transition->before, first_step, transition->after_steps) &&
                   provides_unconditional(*transition->after, transition->after_steps + 1, last_step);
        }
        return false;
    }

    static const Predictor& selected_predictor(const Predictor& predictor, float sigma, const models::DiffusionPredictionStep& step) {
        if (const auto* transition = std::get_if<SwitchPredictor>(&predictor.node))
            return selected_predictor(step.index > transition->after_steps ? *transition->after : *transition->before, sigma, step);
        if (const auto* slg = std::get_if<SkipLayerPredictor>(&predictor.node); slg && !slg->active(sigma, step))
            return selected_predictor(*slg->base, sigma, step);
        return predictor;
    }

    static const Predictor* unconditional_predictor(const Predictor& predictor, float sigma, const models::DiffusionPredictionStep& step) {
        const auto& selected = selected_predictor(predictor, sigma, step);
        if (const auto* branches = guidance_branches(selected))
            return branches->image_only ? nullptr : branches->negative.get();
        if (const auto* slg = std::get_if<SkipLayerPredictor>(&selected.node))
            return unconditional_predictor(*slg->base, sigma, step);
        return nullptr;
    }

    static PredictorCacheKey active_prediction(const Predictor& root, float sigma, const models::DiffusionPredictionStep& step, bool need_unconditional) {
        PredictorCacheKey active;
        std::set<const Predictor*> visited;
        const auto collect = [&](const auto& self, const Predictor& predictor) -> void {
            if (!visited.insert(&predictor).second)
                return;
            const auto& selected = selected_predictor(predictor, sigma, step);
            if (&selected != &predictor) {
                if (std::holds_alternative<SwitchPredictor>(predictor.node))
                    active.switches.push_back(&selected);
                self(self, selected);
            } else if (const auto* model = std::get_if<ModelPredictor>(&predictor.node)) {
                active.models.push_back(model);
            } else if (const auto* guidance = guidance_branches(predictor)) {
                const auto scale = guidance->scale_at(step.index);
                active.guidance.emplace_back(&predictor, scale);
                self(self, *guidance->positive);
                if (scale != 1.f || guidance->image_only)
                    self(self, *guidance->negative);
                if (guidance->image_only)
                    self(self, *guidance->image_only);
            } else if (const auto* slg = std::get_if<SkipLayerPredictor>(&predictor.node)) {
                self(self, *slg->base);
                if (slg->active(sigma, step)) {
                    active.slg.emplace_back(&predictor, slg->scale);
                    self(self, *slg->positive);
                    self(self, *slg->skipped);
                }
            }
        };
        collect(collect, root);
        if (need_unconditional)
            if (const auto* unconditional = unconditional_predictor(root, sigma, step))
                collect(collect, *unconditional);
        return active;
    }

    template <class Binding>
    struct ConditioningWindow {
        const Binding* binding;
        SigmaWindow range;
    };

    struct ConditioningWindows {
        std::vector<ConditioningWindow<models::ControlBinding>> controls;
        std::vector<ConditioningWindow<IPAdapterBinding>> ip_adapters;
        std::vector<ConditioningWindow<PuLIDBinding>> pulid;
    };

    struct ActiveConditioning {
        std::vector<const models::ControlBinding*> controls;
        std::vector<const IPAdapterBinding*> ip_adapters;
        std::vector<const PuLIDBinding*> pulid;
    };

    struct EvaluationState {
        std::map<const models::DenoisingParameterization*, sd_sample::SampleStepCacheDispatcher> caches;
    };

    static bool compatible_condition_tensors(const Tensor<float>& positive, const Tensor<float>& negative, size_t rank) {
        if (positive.empty() || negative.empty())
            return positive.empty() && negative.empty();
        if ((positive.dim() != rank && positive.dim() != rank - 1) ||
            (negative.dim() != rank && negative.dim() != rank - 1))
            return false;
        auto positive_shape = positive.shape(), negative_shape = negative.shape();
        positive_shape.resize(rank, 1);
        negative_shape.resize(rank, 1);
        return positive_shape == negative_shape && positive_shape.back() == 1;
    }

    static Tensor<float> batch_condition_tensor(const Tensor<float>& positive, const Tensor<float>& negative, size_t rank) {
        if (positive.empty())
            return {};
        auto shape = positive.shape();
        shape.resize(rank, 1);
        shape.back() = 2;
        Tensor<float> result(shape);
        auto next = std::copy(positive.values().begin(), positive.values().end(), result.values().begin());
        std::copy(negative.values().begin(), negative.values().end(), next);
        return result;
    }

    struct PredictionExecution::Resources {
        struct PreparedModel {
            models::Model::ExecutionScope scope;

            explicit PreparedModel(models::Model& model)
                : scope(model, false) { scope.require_ready(); }
        };

        const ModelPredictor& primary;
        const models::DenoisingParameterization& parameterization;
        const std::vector<float>& sigmas;
        const PredictionExecutionOptions options;
        std::map<models::Model*, PreparedModel> prepared;
        std::map<const ModelPredictor*, ConditioningWindows> windows;
        std::map<const ModelPredictor*, uint64_t> qwen_prefix_ids;
        std::map<const models::DenoisingParameterization*, sd_sample::ConditionCacheRuntime> caches;
        std::map<const ModelPredictor*, ActiveConditioning> previous_conditioning;
        IPAdapterCache ip_adapter_cache;
        PuLIDCache pulid_cache;
        SpectrumCache spectrum;
        std::map<std::pair<const ModelPredictor*, const ModelPredictor*>, SDCondition> batched_conditions;

        Resources(const ModelPredictor& primary, const std::vector<float>& sigmas, const PredictionExecutionOptions& options)
            : primary(primary), parameterization(primary.resolved_parameterization()), sigmas(sigmas), options(options), spectrum(options.cache, sigmas.size() - 1) {}

        void check_noise_convention(const ModelPredictor& leaf) const {
            if (!parameterization.compatible_noise(leaf.resolved_parameterization()))
                throw std::runtime_error("predictors in one sample must use compatible noise conventions");
        }

        void prepare_predictor(const ModelPredictor& leaf) {
            auto& model = leaf.dit();
            prepared.try_emplace(&model, model);
            const auto* parameterization = &leaf.resolved_parameterization();
            if (options.cache && options.cache->mode != SD_CACHE_DISABLED && options.cache->mode != SD_CACHE_SPECTRUM && !caches.count(parameterization)) {
                auto runtime = sd_sample::init_condition_cache_runtime(model.version(), options.cache, parameterization->native(), sigmas);
                caches.emplace(parameterization, std::move(runtime));
            }
        }

        template <class Bindings, class Windows>
        void add_windows(const Bindings& bindings, Windows& target) {
            if (!bindings)
                return;
            for (const auto& binding : bindings->entries) {
                if (binding.strength != 0.f)
                    target.push_back({&binding, {binding.start_sigma, binding.end_sigma}});
            }
        }

        models::DiffusionPrediction predict_model(const ModelPredictor& leaf, const Tensor<float>& x, float sigma, models::DiffusionPredictionStep step, EvaluationState& state);
        models::DiffusionPrediction predict_batch(const ModelPredictor& leaf, const SDCondition& condition, const Tensor<float>& x, float sigma, const models::DiffusionPredictionStep& step);
    };

    PredictionExecution::PredictionExecution(const Predictor& root, const std::vector<float>& sigmas, const PredictionExecutionOptions& options)
        : root_(root) {
        if (options.need_unconditional && !provides_unconditional(root, 1, static_cast<int>(sigmas.size() - 1)))
            throw std::runtime_error("CFG++ requires a predictor with an unconditional branch");
        resources_      = std::make_unique<Resources>(primary_predictor(root), sigmas, options);
        auto& resources = *resources_;
        std::set<const Predictor*> visited;
        std::set<models::DiT*> models;
        std::set<const models::DenoisingParameterization*> parameterizations;
        const auto collect = [&](const auto& self, const Predictor& predictor) -> void {
            if (!visited.insert(&predictor).second)
                return;
            if (const auto* model = std::get_if<ModelPredictor>(&predictor.node)) {
                models.insert(&model->dit());
                resources.check_noise_convention(*model);
                parameterizations.insert(&model->resolved_parameterization());
                if (model->controls || model->ip_adapters || model->pulid) {
                    auto& windows = resources.windows[model];
                    resources.add_windows(model->controls, windows.controls);
                    resources.add_windows(model->ip_adapters, windows.ip_adapters);
                    resources.add_windows(model->pulid, windows.pulid);
                } else if (model->dit().version() == VERSION_QWEN_IMAGE_2_1) {
                    // This leaf's text, image slots and references are fixed for the sample.
                    resources.qwen_prefix_ids.emplace(model, resources.qwen_prefix_ids.size() + 1);
                }
            } else if (const auto* branches = guidance_branches(predictor)) {
                self(self, *branches->positive);
                self(self, *branches->negative);
                if (branches->image_only)
                    self(self, *branches->image_only);
            } else if (const auto* slg = std::get_if<SkipLayerPredictor>(&predictor.node)) {
                self(self, *slg->base);
                if (slg->scale != 0.f) {
                    self(self, *slg->positive);
                    self(self, *slg->skipped);
                }
            } else {
                const auto& transition = std::get<SwitchPredictor>(predictor.node);
                self(self, *transition.before);
                self(self, *transition.after);
            }
        };
        collect(collect, root_);
        if (options.cache && options.cache->mode == SD_CACHE_SPECTRUM &&
            (options.need_unconditional || models.size() != 1 || parameterizations.size() != 1 || options.dual_schedule))
            throw std::runtime_error("Spectrum requires one denoiser, a single sigma schedule and a non-CFG++ sampler");
    }

    PredictionExecution::~PredictionExecution() = default;

    models::DiT& PredictionExecution::dit() const {
        return resources_->primary.dit();
    }

    const models::DenoisingParameterization& PredictionExecution::parameterization() const {
        return resources_->parameterization;
    }

    class IPAdapterAttention final : public AttentionProcessor {
        std::vector<std::pair<const ::IPAdapter::AttentionCache*, float>> adapters_;

    public:
        IPAdapterAttention(const std::vector<const IPAdapterBinding*>& bindings, IPAdapterCache& cache) {
            adapters_.reserve(bindings.size());
            for (const auto* binding : bindings) {
                auto& projections = cache[binding->model.get()];
                auto entry        = projections.find(binding->tokens.get());
                if (entry == projections.end()) {
                    auto& adapter = *binding->model;
                    models::Model::ExecutionScope scope(adapter, false);
                    scope.require_ready();
                    auto kv = adapter.ip_adapter->compute_attention(adapter.n_threads, *binding->tokens);
                    if (kv.empty())
                        throw std::runtime_error("IP-Adapter key/value projection failed");
                    entry = projections.emplace(binding->tokens.get(), std::move(kv)).first;
                }
                const auto projection = std::find_if(adapters_.begin(), adapters_.end(), [&](const auto& adapter) {
                    return adapter.first == &entry->second;
                });
                if (projection == adapters_.end()) {
                    adapters_.emplace_back(&entry->second, binding->strength);
                } else {
                    projection->second += binding->strength;
                }
            }
        }

        ggml_tensor* apply(GGMLRunnerContext& ctx, const std::string& layer, ggml_tensor* query, ggml_tensor* output, int64_t heads) const override {
            for (const auto& [cache, strength] : adapters_) {
                const auto entry = cache->find(layer);
                if (entry == cache->end())
                    continue;
                const auto& kv   = entry->second;
                const auto input = [&](const Tensor<float>& tensor) {
                    auto* value = sd::make_ggml_tensor(ctx.ggml_ctx, tensor, false);
                    ctx.set_backend_tensor_data(value, tensor.data());
                    if (value->ne[2] == query->ne[2])
                        return value;
                    if (value->ne[2] != 1)
                        throw std::runtime_error("IP-Adapter token batch must be one or match the UNet batch");
                    return ggml_repeat_4d(ctx.ggml_ctx, value, value->ne[0], value->ne[1], query->ne[2], 1);
                };
                auto* key   = input(kv.key);
                auto* value = input(kv.value);
                auto* image = ggml_ext_attention_ext(&ctx, query, key, value, heads, nullptr, false, ctx.flash_attn_enabled);
                output      = ggml_add(ctx.ggml_ctx, output, ggml_scale(ctx.ggml_ctx, image, strength));
            }
            return output;
        }
    };

    const models::DenoisingParameterization& ModelPredictor::resolved_parameterization() const {
        if (!parameterization)
            parameterization = dit().make_parameterization();
        return *parameterization;
    }

    const ModelPredictor& primary_predictor(const Predictor& predictor) {
        if (const auto* model = std::get_if<ModelPredictor>(&predictor.node))
            return *model;
        if (const auto* branches = guidance_branches(predictor))
            return primary_predictor(*branches->positive);
        if (const auto* slg = std::get_if<SkipLayerPredictor>(&predictor.node))
            return primary_predictor(*slg->base);
        return primary_predictor(*std::get<SwitchPredictor>(predictor.node).before);
    }

    models::DiT& primary_dit(const Predictor& predictor) {
        return primary_predictor(predictor).dit();
    }

    template <class Compatible>
    static const models::DenoisingParameterization& common_parameterization(const Predictor& predictor, Compatible compatible, const char* error) {
        const auto& parameterization = primary_predictor(predictor).resolved_parameterization();
        std::set<const Predictor*> visited;
        const auto check = [&](const auto& self, const Predictor& current) -> void {
            if (!visited.insert(&current).second)
                return;
            if (const auto* model = std::get_if<ModelPredictor>(&current.node)) {
                if (!compatible(parameterization, model->resolved_parameterization()))
                    throw std::runtime_error(error);
            } else if (const auto* branches = guidance_branches(current)) {
                self(self, *branches->positive);
                self(self, *branches->negative);
                if (branches->image_only)
                    self(self, *branches->image_only);
            } else if (const auto* slg = std::get_if<SkipLayerPredictor>(&current.node)) {
                self(self, *slg->base);
                if (slg->scale != 0.f) {
                    self(self, *slg->positive);
                    self(self, *slg->skipped);
                }
            } else {
                const auto& transition = std::get<SwitchPredictor>(current.node);
                self(self, *transition.before);
                self(self, *transition.after);
            }
        };
        check(check, predictor);
        return parameterization;
    }

    const models::DenoisingParameterization& sampling_parameterization(const Predictor& predictor) {
        return common_parameterization(
            predictor, [](const auto& a, const auto& b) { return a.compatible_noise(b); },
            "predictors must use compatible noise conventions");
    }

    const models::DenoisingParameterization& scaling_parameterization(const Predictor& predictor) {
        return common_parameterization(
            predictor, [](const auto& a, const auto& b) { return a.compatible_scalings(b); },
            "denoiser_scalings requires matching scaling rules across all branches; pass a model predictor");
    }

    const ModelPredictor& schedule_predictor(const Predictor& predictor) {
        common_parameterization(
            predictor, [](const auto& a, const auto& b) { return &a == &b; },
            "ambiguous schedule source: make_sigmas and t_to_sigma require one shared denoising parameterization; pass the model predictor that defines the schedule");
        return primary_predictor(predictor);
    }

    bool provides_model_output(const Predictor& predictor) {
        std::set<const Predictor*> visited;
        const auto check = [&](const auto& self, const Predictor& branch) -> bool {
            if (!visited.insert(&branch).second)
                return true;
            if (const auto* transition = std::get_if<SwitchPredictor>(&branch.node))
                return self(self, *transition->before) && self(self, *transition->after);
            return std::holds_alternative<ModelPredictor>(branch.node);
        };
        return check(check, predictor);
    }

    models::DiffusionPrediction PredictionExecution::Resources::predict_model(const ModelPredictor& leaf, const Tensor<float>& x, float sigma, models::DiffusionPredictionStep step, EvaluationState& state) {
        auto& engine = leaf.dit();
        prepare_predictor(leaf);
        models::CircularAxesScope circular_axes(engine, leaf.circular_x, leaf.circular_y);
        std::map<models::ControlNet*, models::CircularAxesScope> control_axes;
        if (auto* motion = leaf.model->motion.get()) {
            if (prepared.try_emplace(motion, *motion).second)
                motion->validate_latent(x);
            step.unet_block_processor = motion;
        }
        step.compute_controls = [&step](const Tensor<float>& input, const Tensor<float>& timesteps, const SDCondition& condition) {
            return compute_unet_controls(step.controls, input, timesteps, condition);
        };
        std::vector<const IPAdapterBinding*> active_adapters;
        std::vector<const PuLIDBinding*> active_pulid;
        const auto bindings = windows.find(&leaf);
        if (bindings != windows.end()) {
            for (const auto& window : bindings->second.controls) {
                if (window.range.active(sigma)) {
                    auto& model = *window.binding->model;
                    prepared.try_emplace(&model, model);
                    control_axes.try_emplace(&model, model, leaf.circular_x, leaf.circular_y);
                    step.controls.push_back(window.binding);
                }
            }
            for (const auto& window : bindings->second.ip_adapters)
                if (window.range.active(sigma))
                    active_adapters.push_back(window.binding);
            for (const auto& window : bindings->second.pulid) {
                if (window.range.active(sigma)) {
                    auto& model = *window.binding->model;
                    prepared.try_emplace(&model, model);
                    active_pulid.push_back(window.binding);
                }
            }
        }
        const auto cache = caches.find(&leaf.resolved_parameterization());
        if (cache != caches.end()) {
            step.cache = &state.caches.try_emplace(&leaf.resolved_parameterization(), cache->second, step.solver_index, sigma).first->second;
            if (bindings != windows.end()) {
                auto& previous = previous_conditioning[&leaf];
                if (previous.controls != step.controls || previous.ip_adapters != active_adapters || previous.pulid != active_pulid) {
                    step.cache->invalidate_condition(&leaf.condition);
                    previous = {step.controls, active_adapters, active_pulid};
                }
            }
        }
        IPAdapterAttention attention(active_adapters, ip_adapter_cache);
        if (!active_adapters.empty())
            step.attn2_processor = &attention;
        PuLIDAttention pulid(active_pulid, pulid_cache);
        if (!active_pulid.empty())
            step.flux_image_processor = &pulid;
        const auto prefix   = qwen_prefix_ids.find(&leaf);
        step.qwen_prefix_id = prefix == qwen_prefix_ids.end() ? 0 : prefix->second;
        sd::ParallelScope tensor_scope(engine.tensor_executor.get());
        auto prediction = engine.predict_diffusion(x, leaf.resolved_parameterization(), sigma, leaf.condition, leaf.references,
                                                   leaf.options, step);
        if (prediction.output.empty())
            throw std::runtime_error("diffusion prediction failed");
        return prediction;
    }

    models::DiffusionPrediction PredictionExecution::Resources::predict_batch(const ModelPredictor& leaf, const SDCondition& condition, const Tensor<float>& x, float sigma, const models::DiffusionPredictionStep& step) {
        auto& model = leaf.dit();
        prepare_predictor(leaf);
        models::CircularAxesScope circular_axes(model, leaf.circular_x, leaf.circular_y);
        sd::ParallelScope tensor_scope(model.tensor_executor.get());
        auto prediction = model.predict_diffusion(sd::ops::concat(x, x, 3), leaf.resolved_parameterization(), sigma, condition, {},
                                                  leaf.options, step);
        if (prediction.output.empty())
            throw std::runtime_error("batched diffusion prediction failed");
        return prediction;
    }

    class PredictionExecution::Evaluation {
        struct Result {
            models::DiffusionPrediction prediction;
            Tensor<float> denoised;
        };

        PredictionExecution& execution_;
        const Tensor<float>& x_;
        float sigma_;
        const models::DiffusionPredictionStep& step_;
        EvaluationState& state_;
        const Predictor* unconditional_;
        std::map<const Predictor*, Result> results_;

        const Predictor& active_branch(const Predictor& predictor) const {
            return selected_predictor(predictor, sigma_, step_);
        }

        void batch_cfg(const GuidanceBranches& branches) {
            auto& resources = *execution_.resources_;
            if (!resources.options.batched_cfg || branches.image_only || x_.dim() != 4 || x_.shape().back() != 1)
                return;
            const auto& positive = active_branch(*branches.positive);
            const auto& negative = active_branch(*branches.negative);
            if (&positive == &negative || results_.count(&positive) || results_.count(&negative))
                return;
            const auto* pos = std::get_if<ModelPredictor>(&positive.node);
            const auto* neg = std::get_if<ModelPredictor>(&negative.node);
            if (!pos || !neg || &pos->dit() != &neg->dit() || !sd_version_is_unet(pos->dit().version()) ||
                !pos->resolved_parameterization().compatible_forward(neg->resolved_parameterization()) ||
                pos->circular_x != neg->circular_x || pos->circular_y != neg->circular_y ||
                pos->model->motion || neg->model->motion || pos->controls || neg->controls ||
                pos->ip_adapters || neg->ip_adapters || pos->pulid || neg->pulid ||
                pos->options.transform || neg->options.transform ||
                !pos->references.empty() || !neg->references.empty() ||
                !pos->condition.c_ref_images.empty() || !neg->condition.c_ref_images.empty())
                return;

            if (pos && neg) {
                resources.prepare_predictor(*pos);
                resources.prepare_predictor(*neg);
                for (const auto* model : {pos, neg}) {
                    const auto cache = resources.caches.find(&model->resolved_parameterization());
                    if (cache != resources.caches.end() && cache->second.enabled())
                        return;
                }
            }
            const auto key = std::make_pair(pos, neg);
            auto packed    = resources.batched_conditions.find(key);
            if (packed == resources.batched_conditions.end()) {
                if (!compatible_condition_tensors(pos->condition.c_crossattn, neg->condition.c_crossattn, 3) ||
                    !compatible_condition_tensors(pos->condition.c_vector, neg->condition.c_vector, 2) ||
                    !compatible_condition_tensors(pos->condition.c_concat, neg->condition.c_concat, 4))
                    return;
                for (const auto* condition : {&pos->condition, &neg->condition}) {
                    if (condition->c_concat.empty())
                        continue;
                    const auto& shape = condition->c_concat.shape();
                    if (shape.size() != 4 || shape[0] != x_.shape()[0] || shape[1] != x_.shape()[1])
                        return;
                }
                SDCondition condition{batch_condition_tensor(pos->condition.c_crossattn, neg->condition.c_crossattn, 3),
                                      batch_condition_tensor(pos->condition.c_vector, neg->condition.c_vector, 2),
                                      batch_condition_tensor(pos->condition.c_concat, neg->condition.c_concat, 4)};
                packed = resources.batched_conditions.emplace(key, std::move(condition)).first;
                LOG_VERBOSE("batching compatible positive and negative UNet branches");
            }
            auto prediction = resources.predict_batch(*pos, packed->second, x_, sigma_, step_);
            auto parts      = sd::ops::chunk(prediction.output, 2, 3);
            results_.emplace(&positive, Result{{std::move(parts[0]), prediction.skip_scale, prediction.output_scale}});
            results_.emplace(&negative, Result{{std::move(parts[1]), prediction.skip_scale, prediction.output_scale}});
        }

        Tensor<float> to_denoised(const models::DiffusionPrediction& prediction, const Tensor<float>& raw) const {
            return x_ * prediction.skip_scale + raw * prediction.output_scale;
        }

        const Tensor<float>& denoised(Result& result) const {
            if (result.denoised.empty())
                result.denoised = to_denoised(result.prediction, result.prediction.output);
            return result.denoised;
        }

        const Tensor<float>& denoised(const Predictor& predictor) {
            return denoised(evaluate(predictor));
        }

        const Tensor<float>& matching_raw(const Predictor& predictor, const models::DiffusionPrediction& positive) {
            const auto& prediction = evaluate(predictor).prediction;
            if (prediction.output.empty())
                throw std::runtime_error("APG requires model outputs from each branch");
            if (prediction.skip_scale != positive.skip_scale || prediction.output_scale != positive.output_scale)
                throw std::runtime_error("APG predictors must use matching prediction scalings");
            return prediction.output;
        }

        Result cfg(const GuidanceBranches& branches) {
            const auto scale = branches.scale_at(step_.index);
            if (scale != 1.f || branches.negative.get() == unconditional_)
                batch_cfg(branches);
            const auto& pos = denoised(*branches.positive);
            if (branches.image_only) {
                const auto& neg = denoised(*branches.negative);
                const auto& img = denoised(*branches.image_only);
                return {{}, neg + (img - neg) * branches.image_scale + (pos - img) * scale};
            }
            if (scale == 1.f)
                return {{}, pos};
            const auto& neg = denoised(*branches.negative);
            return {{}, neg + (pos - neg) * scale};
        }

        Result apg(const ApgPredictor& node) {
            const auto& branches = node.branches;
            const auto scale     = branches.scale_at(step_.index);
            auto& positive       = evaluate(*branches.positive);
            if (scale == 1.f && !branches.image_only)
                return {{}, denoised(positive)};

            const auto& prediction = positive.prediction;
            if (prediction.output.empty())
                throw std::runtime_error("APG requires model outputs from each branch");
            const auto& neg = matching_raw(*branches.negative, prediction);
            const auto* img = branches.image_only ? &matching_raw(*branches.image_only, prediction) : nullptr;

            auto& guidance = execution_.apg_.try_emplace(&node, branches.scale, branches.image_scale, node.params).first->second;
            sd::guidance::GuidanceInput input;
            input.step            = step_.index;
            input.schedule_size   = step_.schedule_size;
            input.pred_cond       = &prediction.output;
            input.pred_uncond     = img ? img : &neg;
            input.pred_img_uncond = img ? &neg : nullptr;
            guidance.set_guidance_scale(scale);
            auto result = guidance.forward(input, {});
            return {{}, to_denoised(prediction, result.pred)};
        }

        Result slg(const SkipLayerPredictor& node) {
            auto& base          = evaluate(*node.base);
            const auto& pred    = denoised(base);
            const auto& pos     = denoised(*node.positive);
            const auto& skipped = denoised(*node.skipped);
            return {{}, pred + (pos - skipped) * node.scale};
        }

        Result& evaluate(const Predictor& predictor) {
            const auto& selected = selected_predictor(predictor, sigma_, step_);
            if (&selected != &predictor)
                return evaluate(selected);

            auto found = results_.find(&predictor);
            if (found != results_.end())
                return found->second;

            Result result;
            if (const auto* model = std::get_if<ModelPredictor>(&predictor.node)) {
                result.prediction = execution_.resources_->predict_model(*model, x_, sigma_, step_, state_);
            } else if (const auto* node = std::get_if<CfgPredictor>(&predictor.node)) {
                result = cfg(node->branches);
            } else if (const auto* node = std::get_if<ApgPredictor>(&predictor.node)) {
                result = apg(*node);
            } else {
                result = slg(std::get<SkipLayerPredictor>(predictor.node));
            }
            return results_.emplace(&predictor, std::move(result)).first->second;
        }

    public:
        Evaluation(PredictionExecution& execution, const Tensor<float>& x, float sigma, const models::DiffusionPredictionStep& step, EvaluationState& state)
            : execution_(execution), x_(x), sigma_(sigma), step_(step), state_(state), unconditional_(execution.resources_->options.need_unconditional ? unconditional_predictor(execution.root_, sigma, step) : nullptr) {}

        sd::guidance::GuiderOutput run(bool need_unconditional) {
            auto& result = evaluate(execution_.root_);
            sd::guidance::GuiderOutput output;
            if (need_unconditional) {
                if (!unconditional_)
                    throw std::runtime_error("CFG++ requires a predictor with an unconditional branch");
                output.pred_uncond = denoised(*unconditional_);
            }
            denoised(result);
            output.pred = std::move(result.denoised);
            return output;
        }
    };

    sd::guidance::GuiderOutput PredictionExecution::evaluate(const Tensor<float>& x, float sigma, const models::DiffusionPredictionStep& step) {
        auto& resources = *resources_;
        EvaluationState state;
        if (!resources.options.cache || resources.options.cache->mode == SD_CACHE_DISABLED)
            return Evaluation(*this, x, sigma, step, state).run(resources.options.need_unconditional);
        auto active = active_prediction(root_, sigma, step, resources.options.need_unconditional);
        for (const auto* leaf : active.models) {
            const auto windows = resources.windows.find(leaf);
            if (windows != resources.windows.end()) {
                const auto append = [&](const auto& entries) {
                    for (const auto& entry : entries)
                        active.conditioning.push_back(entry.range.active(sigma));
                };
                append(windows->second.controls);
                append(windows->second.ip_adapters);
                append(windows->second.pulid);
            }
        }
        auto predicted = resources.spectrum.predict(active, x.shape(), step);
        if (!predicted.empty()) {
            sd::guidance::GuiderOutput result;
            result.pred = std::move(predicted);
            return result;
        }
        for (auto& [parameterization, cache] : resources.caches) {
            std::vector<const void*> conditions;
            for (const auto* leaf : active.models)
                if (&leaf->resolved_parameterization() == parameterization)
                    conditions.push_back(&leaf->condition);
            cache.retain_conditions(conditions);
            state.caches.try_emplace(parameterization, cache, step.solver_index, sigma);
        }
        auto result = Evaluation(*this, x, sigma, step, state).run(resources.options.need_unconditional);
        resources.spectrum.record(result.pred, step);
        return result;
    }

    void PredictionExecution::log_cache_summary() const {
        for (const auto& [parameterization, cache] : resources_->caches)
            sd_sample::log_condition_cache_summary(cache, resources_->sigmas.size() - 1);
        resources_->spectrum.log_summary();
    }
}  // namespace sd::dgl::ops
