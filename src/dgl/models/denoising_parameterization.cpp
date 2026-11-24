#include "dgl/models/denoising_parameterization.h"

#include <cmath>
#include <stdexcept>

#include "core/util.h"
#include "model/diffusion/minimax_h3.hpp"
#include "model_definition.h"
#include "runtime/denoiser.hpp"

namespace sd::dgl::models {
    static void calculate_alphas_cumprod(float* alphas_cumprod,
                                         float linear_start = 0.00085f,
                                         float linear_end   = 0.0120f,
                                         int timesteps      = TIMESTEPS) {
        float ls_sqrt = sqrtf(linear_start);
        float le_sqrt = sqrtf(linear_end);
        float amount  = le_sqrt - ls_sqrt;
        float product = 1.0f;
        for (int i = 0; i < timesteps; i++) {
            float beta = ls_sqrt + amount * ((float)i / (timesteps - 1));
            product *= 1.0f - powf(beta, 2.0f);
            alphas_cumprod[i] = product;
        }
    }

    static void calculate_alphas_cumprod_linear_beta(float* alphas_cumprod,
                                                     float beta_start,
                                                     float beta_end,
                                                     int timesteps = TIMESTEPS) {
        float product = 1.0f;
        for (int i = 0; i < timesteps; i++) {
            float beta = beta_start + (beta_end - beta_start) * ((float)i / (timesteps - 1));
            product *= 1.0f - beta;
            alphas_cumprod[i] = product;
        }
    }

    DenoisingParameterization::DenoisingParameterization(SDVersion version, prediction_t prediction, const String2TensorStorage& tensors, const std::vector<float>& file_alphas)
        : version_(version), prediction_(prediction) {
        const auto* definition = find_model_definition(version);
        if (!definition)
            throw std::runtime_error("unknown diffusion model family");
        const auto& policy      = definition->sampling;
        default_flow_shift_     = policy.flow_shift_from_weights ? policy.flow_shift_from_weights(tensors) : policy.default_flow_shift;
        const auto set_denoiser = [&](auto instance) {
            using T   = typename decltype(instance)::element_type;
            denoiser_ = std::move(instance);
            copy_     = [](const Denoiser& source) -> std::shared_ptr<Denoiser> {
                return std::make_shared<T>(static_cast<const T&>(source));
            };
        };
        switch (prediction_) {
            case EPS_PRED:
                LOG_INFO("running in eps-prediction mode");
                set_denoiser(std::make_shared<CompVisDenoiser>());
                break;
            case V_PRED:
                LOG_INFO("running in v-prediction mode");
                set_denoiser(std::make_shared<CompVisVDenoiser>());
                break;
            case EDM_V_PRED:
                LOG_INFO("running in v-prediction EDM mode");
                set_denoiser(std::make_shared<EDMVDenoiser>());
                break;
            case FLOW_PRED: {
                LOG_INFO("running in FLOW mode");
                set_denoiser(std::make_shared<DiscreteFlowDenoiser>());
                break;
            }
            case FLUX_FLOW_PRED: {
                LOG_INFO("running in Flux FLOW mode");
                set_denoiser(std::make_shared<FluxFlowDenoiser>());
                break;
            }
            case FLUX3_FLOW_PRED: {
                LOG_INFO("running in Flux.3 FLOW mode");
                set_denoiser(std::make_shared<Flux3FlowDenoiser>());
                break;
            }
            case H3_AV_FLOW_PRED: {
                if (version_ == VERSION_MINIMAX_H3_IMAGE) {
                    LOG_INFO("running in MiniMax H3 Image FLOW mode");
                    set_denoiser(std::make_shared<MiniMaxH3ImageDenoiser>(MiniMaxH3::pdd_sigmas));
                    break;
                }
                LOG_INFO("running in MiniMax H3 AV FLOW mode");
                set_denoiser(std::make_shared<DiscreteFlowDenoiser>());
                break;
            }
            case MING_IMAGE_FLOW_PRED: {
                LOG_INFO("running in Ming-Image FLOW mode");
                set_denoiser(std::make_shared<MingImageFlowDenoiser>());
                break;
            }
            case SEFI_FLOW_PRED: {
                LOG_INFO("running in SeFi-Image dual-time FLOW mode");
                set_denoiser(std::make_shared<SefiFlowDenoiser>());
                break;
            }
            case MINIT2I_FLOW_PRED: {
                LOG_INFO("running in MiniT2I FLOW mode");
                set_denoiser(std::make_shared<MiniT2IFlowDenoiser>());
                break;
            }
            case SENSENOVA_U1_FLOW_PRED: {
                LOG_INFO("running in SenseNova U1.5 FLOW mode");
                set_denoiser(std::make_shared<SenseNovaU1FlowDenoiser>(default_flow_shift_));
                break;
            }
            default: {
                LOG_ERROR("Unknown predition type %i", prediction_);
                throw std::runtime_error("unknown prediction type");
            }
        }

        if (auto* flow = dynamic_cast<DiscreteFlowDenoiser*>(denoiser_.get()))
            flow->set_shift(default_flow_shift_);

        auto comp_vis_denoiser = std::dynamic_pointer_cast<CompVisDenoiser>(denoiser_);
        if (!comp_vis_denoiser) {
            return;
        }
        std::vector<float> alphas_cumprod(TIMESTEPS);
        if (file_alphas.size() == TIMESTEPS) {
            alphas_cumprod = file_alphas;
        } else if (sd_version_is_pixart(version_)) {
            // PixArt checkpoints train with a linear beta schedule (0.0001 -> 0.02)
            // instead of the scaled_linear schedule used by SD1.x/SDXL.
            calculate_alphas_cumprod_linear_beta(alphas_cumprod.data(), 0.0001f, 0.02f);
        } else {
            calculate_alphas_cumprod(alphas_cumprod.data());
        }
        for (int i = 0; i < TIMESTEPS; i++) {
            comp_vis_denoiser->sigmas[i]     = std::sqrt((1 - alphas_cumprod[i]) / alphas_cumprod[i]);
            comp_vis_denoiser->log_sigmas[i] = std::log(comp_vis_denoiser->sigmas[i]);
        }
    }

    std::shared_ptr<Denoiser> DenoisingParameterization::schedule(float flow_shift, int sequence_length) const {
        auto schedule = copy_(*denoiser_);
        if (auto* flow = dynamic_cast<DiscreteFlowDenoiser*>(schedule.get())) {
            flow->set_shift(flow_shift == 0.f ? default_flow_shift_ : flow_shift);
        }
        if (auto* ming = dynamic_cast<MingImageFlowDenoiser*>(schedule.get()); ming && sequence_length > 0) {
            ming->set_sequence_length(sequence_length);
        }
        return schedule;
    }

    float DenoisingParameterization::schedule_sigma(double t, float flow_shift, int sequence_length) const {
        auto schedule = this->schedule(flow_shift, sequence_length);
        if (dynamic_cast<MiniT2IFlowDenoiser*>(schedule.get()) || dynamic_cast<SenseNovaU1FlowDenoiser*>(schedule.get()))
            return schedule->t_to_sigma(static_cast<float>(1.0 - t));
        if (dynamic_cast<DiscreteFlowDenoiser*>(schedule.get()) && !dynamic_cast<MiniMaxH3ImageDenoiser*>(schedule.get()))
            return t == 0.0 ? 0.f : schedule->t_to_sigma(static_cast<float>(t * TIMESTEPS - 1.0));
        return schedule->t_to_sigma(static_cast<float>(t * (TIMESTEPS - 1)));
    }

    float DenoisingParameterization::timestep(float sigma) const {
        float t = denoiser_->sigma_to_t(sigma);
        if (sd_version_is_anima(version_)) {
            return t / static_cast<float>(TIMESTEPS);
        }
        if (sd_version_is_boogu_image(version_)) {
            return t / static_cast<float>(TIMESTEPS);
        }
        if (version_ == VERSION_HIDREAM_O1) {
            return 1.0f - t / static_cast<float>(TIMESTEPS);
        }
        if (sd_version_is_z_image(version_) || sd_version_is_z_image_l2p(version_) || sd_version_is_ideogram4(version_) || version_ == VERSION_MING_IMAGE) {
            return 1000.f - t;
        }
        return t;
    }

    bool DenoisingParameterization::is_flow() const {
        return dynamic_cast<DiscreteFlowDenoiser*>(denoiser_.get()) != nullptr;
    }

    bool DenoisingParameterization::compatible_noise(const DenoisingParameterization& other) const {
        for (const auto special : {MINIT2I_FLOW_PRED, SENSENOVA_U1_FLOW_PRED, SEFI_FLOW_PRED}) {
            if (prediction_ == special || other.prediction_ == special)
                return prediction_ == other.prediction_;
        }
        return is_flow() == other.is_flow();
    }

    bool DenoisingParameterization::compatible_forward(const DenoisingParameterization& other) const {
        return version_ == other.version_ && prediction_ == other.prediction_;
    }

    bool DenoisingParameterization::compatible_scalings(const DenoisingParameterization& other) const {
        const auto* left  = dynamic_cast<const CompVisDenoiser*>(denoiser_.get());
        const auto* right = dynamic_cast<const CompVisDenoiser*>(other.denoiser_.get());
        if (left || right)
            return left && right && left->sigma_data == right->sigma_data &&
                   bool(dynamic_cast<const CompVisVDenoiser*>(left)) == bool(dynamic_cast<const CompVisVDenoiser*>(right));
        if (is_flow() && other.is_flow())
            return bool(dynamic_cast<const SenseNovaU1FlowDenoiser*>(denoiser_.get())) ==
                   bool(dynamic_cast<const SenseNovaU1FlowDenoiser*>(other.denoiser_.get()));
        return prediction_ == other.prediction_;
    }

    std::vector<float> DenoisingParameterization::scalings(float sigma) const {
        return denoiser_->get_scalings(sigma);
    }

    Tensor<float> DenoisingParameterization::add_noise(float sigma, const Tensor<float>& noise, const Tensor<float>& base) const {
        return denoiser_->noise_scaling(sigma, noise, base);
    }

    Tensor<float> DenoisingParameterization::finish(float sigma, const Tensor<float>& latent) const {
        return denoiser_->inverse_noise_scaling(sigma, latent);
    }
}
