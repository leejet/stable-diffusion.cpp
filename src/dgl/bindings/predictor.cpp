#include "dgl/bindings/predictor.h"

#include <cmath>

#include "dgl/bindings/arguments.h"
#include "dgl/bindings/condition.h"
#include "dgl/context.h"
#include "dgl/model_value.h"
#include "dgl/models/control.h"
#include "dgl/models/dit.h"
#include "dgl/models/ip_adapter.h"
#include "dgl/models/pulid.h"
#include "dgl/ops/predictor.h"

namespace sd::dgl {
    using ops::ApgPredictor;
    using ops::CfgPredictor;
    using ops::GuidanceBranches;
    using ops::ModelPredictor;
    using ops::Predictor;
    using ops::SkipLayerPredictor;
    using ops::SwitchPredictor;

    template <class Node>
    static Value predictor_value(Node node) {
        return Value::object(std::make_shared<Predictor>(Predictor{std::move(node)}));
    }

    static void bind_predictor_options(ModelPredictor& predictor, const Values& args) {
        predictor.circular_x = get_bool(args, "circular_x");
        predictor.circular_y = get_bool(args, "circular_y");
        auto& options        = predictor.options;
        for (const auto& [name, field] : {std::pair{"vace_context", &options.vace_context}, {"audio_embedding", &options.audio_embedding}, {"video_positions", &options.video_positions}}) {
            if (args.count(name)) {
                *field = float_tensor(args, name);
            }
        }
        if (args.count("skip_layers")) {
            for (const auto& layer : args.at("skip_layers").items) {
                options.skip_layers.push_back(int(layer.scalar.get<int64_t>()));
            }
        }
        options.fps               = float(get_number(args, "fps"));
        options.audio_length      = int(get_int(args, "audio_length"));
        options.video_sigma_shift = float(get_number(args, "video_sigma_shift"));
        options.audio_sigma_shift = float(get_number(args, "audio_sigma_shift"));
        options.vace_strength     = float(get_number(args, "vace_strength"));
        if (args.count("transform")) {
            auto fn           = object<ScriptFunction>(args, "transform");
            auto captures     = args.count("transform_args") ? args.at("transform_args") : Value::dict({});
            options.transform = [fn = std::move(fn), captures = std::move(captures)](models::PredictionParameters step) {
                const auto result = (*fn)(Value::dict({{"sigma", Value::literal(step.sigma)},
                                                       {"timestep", Value::literal(step.timestep)},
                                                       {"input_scale", Value::literal(step.input_scale)},
                                                       {"skip_scale", Value::literal(step.skip_scale)},
                                                       {"output_scale", Value::literal(step.output_scale)}}),
                                          captures);
                const auto update = [&](const char* name, float& target) {
                    if (!result.fields.count(name))
                        return;
                    target = float(get_number(result.fields, name));
                    if (!std::isfinite(target))
                        throw std::runtime_error(std::string("prediction transform returned a non-finite ") + name);
                };
                update("timestep", step.timestep);
                update("input_scale", step.input_scale);
                update("skip_scale", step.skip_scale);
                update("output_scale", step.output_scale);
                return step;
            };
        }
    }

    static ops::SigmaWindow sigma_window(const Values& args) {
        const auto start = args.count("start_sigma") ? float(get_number(args, "start_sigma")) : INFINITY;
        const auto end   = args.count("end_sigma") ? float(get_number(args, "end_sigma")) : 0.f;
        if (start < end)
            throw std::runtime_error("start_sigma must be greater than or equal to end_sigma");
        return {start, end};
    }

    Value apply_controlnet(Values& args, RunState&) {
        auto fields             = args.at("condition").fields;
        const auto previous     = fields.find("controls");
        auto controls           = previous == fields.end() ? std::make_shared<models::ControlBindings>()
                                                           : std::make_shared<models::ControlBindings>(*previous->second.as<models::ControlBindings>());
        auto model              = object<ControlNet>(args, "model");
        auto hint               = std::shared_ptr<const sd::Tensor<float>>(object<sd_tensor_t>(args, "hint"), &float_tensor(args, "hint"));
        const auto type         = str_to_sd_control_type(get_string(args, "type").c_str());
        const auto [start, end] = sigma_window(args);
        model->validate_hint(*hint, type);
        controls->entries.push_back({std::move(model), std::move(hint), type, float(get_number(args, "strength")), start, end});
        fields.insert_or_assign("controls", Value::object(std::move(controls)));
        return Value::dict(std::move(fields));
    }

    Value apply_ip_adapter(Values& args, RunState&) {
        auto fields             = args.at("condition").fields;
        const auto previous     = fields.find("ip_adapters");
        auto bindings           = previous == fields.end() ? std::make_shared<ops::IPAdapterBindings>()
                                                           : std::make_shared<ops::IPAdapterBindings>(*previous->second.as<ops::IPAdapterBindings>());
        auto adapter            = object<IPAdapter>(args, "adapter");
        auto tokens             = std::shared_ptr<const sd::Tensor<float>>(object<sd_tensor_t>(args, "tokens"), &float_tensor(args, "tokens"));
        const auto [start, end] = sigma_window(args);
        adapter->validate_tokens(*tokens);
        bindings->entries.push_back({std::move(adapter), std::move(tokens), float(get_number(args, "strength")), start, end});
        fields.insert_or_assign("ip_adapters", Value::object(std::move(bindings)));
        return Value::dict(std::move(fields));
    }

    Value apply_pulid(Values& args, RunState&) {
        auto fields             = args.at("condition").fields;
        const auto previous     = fields.find("pulid");
        auto bindings           = previous == fields.end() ? std::make_shared<ops::PuLIDBindings>()
                                                           : std::make_shared<ops::PuLIDBindings>(*previous->second.as<ops::PuLIDBindings>());
        auto pulid              = object<models::PuLID>(args, "pulid");
        auto identity           = std::shared_ptr<const sd::Tensor<float>>(object<sd_tensor_t>(args, "embedding"), &float_tensor(args, "embedding"));
        const auto [start, end] = sigma_window(args);
        pulid->validate_embedding(*identity);
        bindings->entries.push_back({std::move(pulid), std::move(identity), float(get_number(args, "strength")), start, end});
        fields.insert_or_assign("pulid", Value::object(std::move(bindings)));
        return Value::dict(std::move(fields));
    }

    Value apply_animatediff(Values& args, RunState&) {
        return with_animatediff(args.at("model"), object<models::AnimateDiff>(args, "motion"));
    }

    Value make_predictor(Values& args, RunState&) {
        ModelPredictor output;
        output.model                           = object<models::DiffusionModel>(args, "model");
        auto& engine                           = output.dit();
        output.parameterization                = engine.make_parameterization(str_to_prediction(get_string(args, "prediction").c_str()));
        const auto ref_index_mode              = get_string(args, "ref_index_mode");
        output.options.ref_index_mode          = ref_index_mode == "increase"   ? Rope::RefIndexMode::INCREASE
                                                 : ref_index_mode == "decrease" ? Rope::RefIndexMode::DECREASE
                                                                                : Rope::RefIndexMode::FIXED;
        output.options.force_ref_timestep_zero = get_bool(args, "force_ref_timestep_zero");
        const auto ref_rope_anchor             = get_string(args, "ref_rope_anchor");
        if (ref_rope_anchor != "default" && engine.version() != VERSION_MOSS_IMAGE)
            throw std::runtime_error("reference RoPE anchoring is not configurable for this model");
        output.options.ref_rope_anchor = ref_rope_anchor == "image_start"  ? Rope::RefRoPEAnchor::IMAGE_START
                                         : ref_rope_anchor == "text_start" ? Rope::RefRoPEAnchor::TEXT_START
                                                                           : Rope::RefRoPEAnchor::MODEL_DEFAULT;
        output.options.guidance        = float(get_number(args, "guidance"));
        bind_predictor_options(output, args);
        const auto& condition = args.at("condition").fields;
        if (const auto controls = condition.find("controls"); controls != condition.end()) {
            output.controls = controls->second.as<models::ControlBindings>();
            for (const auto& binding : output.controls->entries)
                binding.model->validate_target(engine);
        }
        if (const auto adapters = condition.find("ip_adapters"); adapters != condition.end()) {
            output.ip_adapters = adapters->second.as<ops::IPAdapterBindings>();
            for (const auto& binding : output.ip_adapters->entries)
                if (binding.strength != 0.f)
                    binding.model->validate_target(engine);
        }
        if (const auto bindings = condition.find("pulid"); bindings != condition.end()) {
            output.pulid = bindings->second.as<ops::PuLIDBindings>();
            for (const auto& binding : output.pulid->entries)
                if (binding.strength != 0.f)
                    binding.model->validate_target(engine);
        }
        output.condition = read_condition(condition, args);
        auto& c          = output.condition;
        if (args.count("ref_latents")) {
            output.references = float_tensor_list(args.at("ref_latents"));
        }
        if (const auto lq = condition.find("lq_frames"); lq != condition.end()) {
            if (engine.version() != VERSION_RELAY_VSR || !output.references.empty() || !c.c_ref_images.empty())
                throw std::runtime_error("lq_frames requires a RelayVSR predictor with one low-quality frame tensor");
            c.c_ref_images.push_back(pixels(lq->second));
        }
        validate_reference_blocks(c, output.references.empty() ? c.c_ref_images.size() : output.references.size());
        return predictor_value(std::move(output));
    }

    static GuidanceBranches guidance_branches(const Values& args) {
        GuidanceBranches output;
        output.positive    = object<Predictor>(args, "positive");
        output.negative    = object<Predictor>(args, "negative");
        output.scale       = float(get_number(args, "scale"));
        output.image_scale = float(get_number(args, "image_scale"));
        if (args.count("image_only"))
            output.image_only = object<Predictor>(args, "image_only");
        if (args.count("schedule")) {
            for (const auto& item : args.at("schedule").items) {
                const auto scale = item.scalar.get<float>();
                if (!(scale >= 0.f && scale <= 1000.f))
                    throw std::runtime_error("guidance schedule values must be between zero and 1000");
                output.scale_schedule.push_back(scale);
            }
        }
        return output;
    }

    Value make_cfg(Values& args, RunState&) {
        return predictor_value(CfgPredictor{guidance_branches(args)});
    }

    Value make_apg(Values& args, RunState&) {
        auto branches = guidance_branches(args);
        sd::guidance::AdaptiveProjectedGuidanceParams params{
            float(get_number(args, "eta")),
            float(get_number(args, "momentum")),
            float(get_number(args, "norm_threshold")),
            float(get_number(args, "norm_threshold_smoothing")),
        };
        if (!sd::guidance::is_adaptive_projected_guidance_enabled(params))
            return predictor_value(CfgPredictor{std::move(branches)});

        for (const auto* branch : {branches.positive.get(), branches.negative.get(), branches.image_only.get()})
            if (branch && !ops::provides_model_output(*branch))
                throw std::runtime_error("APG branches must provide raw model outputs");
        return predictor_value(ApgPredictor{std::move(branches), params});
    }

    Value make_slg(Values& args, RunState&) {
        std::optional<ops::StepWindow> steps;
        if (args.count("start_step_fraction") || args.count("stop_step_fraction")) {
            const auto start = args.count("start_step_fraction") ? float(get_number(args, "start_step_fraction")) : 0.f;
            const auto stop  = args.count("stop_step_fraction") ? float(get_number(args, "stop_step_fraction")) : 1.f;
            if (stop < start)
                throw std::runtime_error("SLG start_step_fraction must be less than or equal to stop_step_fraction");
            steps = ops::StepWindow{start, stop};
        }
        return predictor_value(SkipLayerPredictor{
            object<Predictor>(args, "base"),
            object<Predictor>(args, "positive"),
            object<Predictor>(args, "skipped"),
            float(get_number(args, "scale")),
            sigma_window(args),
            steps,
        });
    }

    Value switch_predictor(Values& args, RunState&) {
        return predictor_value(SwitchPredictor{
            object<Predictor>(args, "before"),
            object<Predictor>(args, "after"),
            int(get_int(args, "after_steps")),
        });
    }

}  // namespace sd::dgl
