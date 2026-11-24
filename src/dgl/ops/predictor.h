#ifndef __SD_DGL_OPS_PREDICTOR_H__
#define __SD_DGL_OPS_PREDICTOR_H__

#include "dgl/models/fwd.h"

#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <variant>

#include "conditioning/conditioner.hpp"
#include "dgl/models/denoising_parameterization.h"
#include "dgl/models/diffusion_model.h"
#include "dgl/models/diffusion_prediction.h"
#include "dgl/ops/control.h"
#include "dgl/ops/ip_adapter.h"
#include "dgl/ops/pulid.h"
#include "runtime/guidance.h"

namespace sd {
    struct CacheConfig;
}

namespace sd::dgl::ops {
    struct Predictor;

    struct ModelPredictor {
        std::shared_ptr<const models::DiffusionModel> model;
        mutable std::shared_ptr<const models::DenoisingParameterization> parameterization;
        SDCondition condition;
        std::shared_ptr<models::ControlBindings> controls;
        std::shared_ptr<IPAdapterBindings> ip_adapters;
        std::shared_ptr<PuLIDBindings> pulid;
        models::DiffusionPredictionOptions options;
        std::vector<sd::Tensor<float>> references;
        bool circular_x = false;
        bool circular_y = false;

        models::DiT& dit() const { return *model->dit; }
        const models::DenoisingParameterization& resolved_parameterization() const;
    };

    struct GuidanceBranches {
        std::shared_ptr<Predictor> positive, negative, image_only;
        float scale       = 1.f;
        float image_scale = 1.f;
        std::vector<float> scale_schedule;

        float scale_at(int step) const;
    };

    struct CfgPredictor {
        GuidanceBranches branches;
    };

    struct ApgPredictor {
        GuidanceBranches branches;
        sd::guidance::AdaptiveProjectedGuidanceParams params;
    };

    struct SigmaWindow {
        float start_sigma = std::numeric_limits<float>::infinity();
        float end_sigma   = 0.f;

        bool active(float sigma) const { return sigma <= start_sigma && sigma >= end_sigma; }
        bool bounded() const { return std::isfinite(start_sigma) || end_sigma > 0.f; }
    };

    struct StepWindow {
        float start_fraction, stop_fraction;

        bool active(const models::DiffusionPredictionStep& step) const;
    };

    struct SkipLayerPredictor {
        std::shared_ptr<Predictor> base, positive, skipped;
        float scale;
        SigmaWindow sigmas;
        std::optional<StepWindow> steps;

        bool active(float sigma, const models::DiffusionPredictionStep& step) const;
    };

    struct SwitchPredictor {
        std::shared_ptr<Predictor> before, after;
        int after_steps;
    };

    struct Predictor {
        std::variant<ModelPredictor, CfgPredictor, ApgPredictor, SkipLayerPredictor, SwitchPredictor> node;
    };

    bool provides_model_output(const Predictor& predictor);
    models::DiT& primary_dit(const Predictor& predictor);
    const ModelPredictor& primary_predictor(const Predictor& predictor);
    const models::DenoisingParameterization& sampling_parameterization(const Predictor& predictor);
    const models::DenoisingParameterization& scaling_parameterization(const Predictor& predictor);
    const ModelPredictor& schedule_predictor(const Predictor& predictor);

    struct PredictionExecutionOptions {
        const CacheConfig* cache = nullptr;
        bool need_unconditional  = false;
        bool dual_schedule       = false;
        bool batched_cfg         = true;
    };

    class PredictionExecution {
        class Evaluation;
        struct Resources;

        const Predictor& root_;
        std::unique_ptr<Resources> resources_;
        std::map<const ApgPredictor*, sd::guidance::AdaptiveProjectedGuidance> apg_;

    public:
        PredictionExecution(const Predictor& root, const std::vector<float>& sigmas, const PredictionExecutionOptions& options);
        ~PredictionExecution();

        models::DiT& dit() const;
        const models::DenoisingParameterization& parameterization() const;

        sd::guidance::GuiderOutput evaluate(const Tensor<float>& x, float sigma, const models::DiffusionPredictionStep& step);
        void log_cache_summary() const;
    };
}

#endif  // __SD_DGL_OPS_PREDICTOR_H__
