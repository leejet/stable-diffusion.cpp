#ifndef __SD_MODEL_DIFFUSION_MODEL_HPP__
#define __SD_MODEL_DIFFUSION_MODEL_HPP__

#include <string>
#include <utility>
#include <variant>

#include "core/ggml_runner.h"
#include "core/tensor_ggml.hpp"
#include "model/common/rope_circular.hpp"
#include "model_manager.h"

enum class UNetBlockStage { Down,
                            Middle,
                            Up };

struct UNetBlockProcessor {
    virtual ~UNetBlockProcessor()                                                                                                             = default;
    virtual ggml_tensor* apply(GGMLRunnerContext& ctx, UNetBlockStage stage, int level, int block, ggml_tensor* activation, int frames) const = 0;
    virtual size_t extra_graph_size() const                                                                                                   = 0;
};

struct UNetDiffusionExtra {
    int num_video_frames                           = -1;
    const std::vector<sd::Tensor<float>>* controls = nullptr;
    float control_strength                         = 0.f;
    const sd::Tensor<float>* ip_context            = nullptr;
    float ip_scale                                 = 1.f;
    const AttentionProcessor* attn2_processor      = nullptr;
    const UNetBlockProcessor* block_processor      = nullptr;
};

struct SkipLayerDiffusionExtra {
    const std::vector<int>* skip_layers = nullptr;
};

struct FluxImageBlockProcessor {
    virtual ~FluxImageBlockProcessor()                                                                          = default;
    virtual bool applies(bool single_stream, int block) const                                                   = 0;
    virtual ggml_tensor* apply(GGMLRunnerContext& ctx, bool single_stream, int block, ggml_tensor* image) const = 0;
    virtual size_t extra_graph_size() const                                                                     = 0;
};

struct FluxDiffusionExtra {
    const sd::Tensor<float>* guidance              = nullptr;
    const std::vector<int>* skip_layers            = nullptr;
    const sd::Tensor<float>* pulid_id              = nullptr;
    float pulid_id_weight                          = 1.0f;
    const FluxImageBlockProcessor* image_processor = nullptr;
};

struct Flux3DiffusionExtra {
    const sd::Tensor<float>* target_timestep = nullptr;
};

struct AnimaDiffusionExtra {
    const sd::Tensor<int32_t>* t5_ids   = nullptr;
    const sd::Tensor<float>* t5_weights = nullptr;
};

struct QwenImage21DiffusionExtra {
    const sd::Tensor<int32_t>* image_slots = nullptr;
    // Nonzero IDs identify immutable prefix inputs within one sampling run.
    uint64_t prefix_id = 0;
};

struct WanDiffusionExtra {
    const sd::Tensor<float>* vace_context = nullptr;
    float vace_strength                   = 1.f;
    // S2V audio, sd::Tensor layout: [dim, T_latent*4, layers].
    const sd::Tensor<float>* audio_embed = nullptr;
};

struct HiDreamO1DiffusionExtra {
    const sd::Tensor<int32_t>* input_ids                               = nullptr;
    const sd::Tensor<int32_t>* input_pos                               = nullptr;
    const sd::Tensor<int32_t>* token_types                             = nullptr;
    const sd::Tensor<int32_t>* vinput_mask                             = nullptr;
    const std::vector<std::pair<int, sd::Tensor<float>>>* image_embeds = nullptr;
};

struct LTXAVDiffusionExtra {
    const sd::Tensor<float>* audio_x         = nullptr;
    const sd::Tensor<float>* audio_timesteps = nullptr;
    int audio_length                         = 0;
    float frame_rate                         = 24.f;
    const sd::Tensor<float>* video_positions = nullptr;
    const sd::Tensor<float>* audio_context   = nullptr;
};

enum class MiniMaxH3ReferenceKind : int32_t {
    IMAGE,
    VIDEO,
    AUDIO,
    VIDEO_AUDIO,
};

struct MiniMaxH3ReferenceBlock {
    MiniMaxH3ReferenceKind kind = MiniMaxH3ReferenceKind::IMAGE;
    int32_t video_index         = -1;
    int32_t audio_index         = -1;
};

struct MiniMaxH3DiffusionExtra {
    const sd::Tensor<int32_t>* text_token_tags                    = nullptr;
    const sd::Tensor<int32_t>* keyframe_indices                   = nullptr;
    const std::vector<sd::Tensor<float>>* reference_audio_latents = nullptr;
    const std::vector<MiniMaxH3ReferenceBlock>* reference_blocks  = nullptr;
    int audio_length                                              = 0;
    float video_sigma_shift                                       = 12.f;
    float audio_sigma_shift                                       = 3.f;
    float video_sigma_next                                        = NAN;
};

struct MiniT2IDiffusionExtra {
    const sd::Tensor<float>* mask = nullptr;
};

struct SenseNovaU1DiffusionExtra {
    const sd::Tensor<int32_t>* input_ids = nullptr;
};

struct HunyuanVideoDiffusionExtra {
    const sd::Tensor<float>* guidance   = nullptr;
    const sd::Tensor<float>* byt5       = nullptr;
    const sd::Tensor<float>* vision     = nullptr;
    const sd::Tensor<float>* timestep_r = nullptr;
};

struct LLaDAImageDiffusionExtra {
    // SigVQ semantic features of the reference image; present only in editing mode.
    const sd::Tensor<float>* semantic = nullptr;
};

struct MingImageDiffusionExtra {
    const sd::Tensor<float>* direct_context = nullptr;
};

namespace ZImage {
    struct ControlNetRunner;
}

struct ZImageControlInput {
    const ZImage::ControlNetRunner* runner;
    const sd::Tensor<float>* latent;
    float strength;
};

struct ZImageDiffusionExtra {
    const sd::Tensor<float>* control_latent         = nullptr;
    float strength                                  = 1.f;
    const std::vector<ZImageControlInput>* controls = nullptr;
};

using DiffusionExtraParams = std::variant<std::monostate,
                                          UNetDiffusionExtra,
                                          SkipLayerDiffusionExtra,
                                          FluxDiffusionExtra,
                                          Flux3DiffusionExtra,
                                          AnimaDiffusionExtra,
                                          QwenImage21DiffusionExtra,
                                          WanDiffusionExtra,
                                          HiDreamO1DiffusionExtra,
                                          LTXAVDiffusionExtra,
                                          MiniMaxH3DiffusionExtra,
                                          MiniT2IDiffusionExtra,
                                          SenseNovaU1DiffusionExtra,
                                          HunyuanVideoDiffusionExtra,
                                          LLaDAImageDiffusionExtra,
                                          MingImageDiffusionExtra,
                                          ZImageDiffusionExtra>;

struct DiffusionParams {
    const sd::Tensor<float>* x                        = nullptr;
    const sd::Tensor<float>* timesteps                = nullptr;
    const sd::Tensor<float>* context                  = nullptr;
    const sd::Tensor<float>* c_concat                 = nullptr;
    const sd::Tensor<float>* y                        = nullptr;
    const std::vector<sd::Tensor<float>>* ref_latents = nullptr;
    Rope::RefIndexMode ref_index_mode                 = Rope::RefIndexMode::FIXED;
    Rope::RefRoPEAnchor ref_rope_anchor               = Rope::RefRoPEAnchor::MODEL_DEFAULT;
    bool force_ref_timestep_zero                      = false;
    DiffusionExtraParams extra                        = std::monostate{};
};

template <typename T>
static inline const T* diffusion_extra_as(const DiffusionParams& params) {
    const auto* extra = std::get_if<T>(&params.extra);
    GGML_ASSERT(extra != nullptr);
    return extra;
}

template <typename T>
static inline const sd::Tensor<T>& tensor_or_empty(const sd::Tensor<T>* tensor) {
    static const sd::Tensor<T> kEmpty;
    return tensor != nullptr ? *tensor : kEmpty;
}

struct DiffusionModelRunner : public GGMLRunner {
protected:
    std::string prefix;

    std::vector<float> finish_rope_pe(Rope::Embedding embedding) {
        Rope::apply_circular(embedding, circular_x_enabled, circular_y_enabled);
        return std::move(embedding.values);
    }

public:
    DiffusionModelRunner(ggml_backend_t backend,
                         const std::string& prefix,
                         std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
        : GGMLRunner(backend, weight_manager),
          prefix(prefix) {}

    virtual sd::Tensor<float> compute(int n_threads,
                                      const DiffusionParams& diffusion_params) = 0;

    virtual int get_patch_size() const {
        return 1;
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) {
        get_param_tensors(tensors, prefix);
    }

    virtual void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors,
                                   const std::string& prefix) = 0;
};

#endif  // __SD_MODEL_DIFFUSION_MODEL_HPP__
