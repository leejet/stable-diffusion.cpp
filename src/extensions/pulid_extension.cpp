#include "extensions/generation_extension.h"

#include <cstring>
#include <variant>

#include "core/tensor_ggml.hpp"
#include "core/util.h"
#include "gguf.h"

// Load the precomputed PuLID identity embedding produced by
// scripts/pulid_extract_id.py into a sd::Tensor<float> (always materialized as
// fp32 for the diffusion path). Returns an empty tensor on any failure (the
// caller treats empty as "PuLID off").
//
// The file is a standard gguf container holding a single tensor named
// "pulid_id" with shape [token_dim, num_tokens] (ggml order; typically
// [2048, 32]) in f16 / bf16 / f32. Using gguf rather than a bespoke header
// means the shape + dtype are self-describing and we reuse ggml's reader.
static sd::Tensor<float> load_pulid_id_embedding(const char* path) {
    sd::Tensor<float> empty;
    if (path == nullptr || strlen(path) == 0) {
        return empty;
    }

    struct ggml_context* ctx_data   = nullptr;
    struct gguf_init_params gp       = {/*.no_alloc =*/false, /*.ctx =*/&ctx_data};
    struct gguf_context* gguf_ctx    = gguf_init_from_file(path, gp);
    if (gguf_ctx == nullptr || ctx_data == nullptr) {
        LOG_WARN("PuLID id-embedding: cannot read gguf '%s'", path);
        if (gguf_ctx != nullptr)
            gguf_free(gguf_ctx);
        if (ctx_data != nullptr)
            ggml_free(ctx_data);
        return empty;
    }

    struct ggml_tensor* t = ggml_get_tensor(ctx_data, "pulid_id");
    if (t == nullptr) {
        LOG_WARN("PuLID id-embedding: no 'pulid_id' tensor in '%s'", path);
        gguf_free(gguf_ctx);
        ggml_free(ctx_data);
        return empty;
    }

    const int64_t token_dim  = t->ne[0];
    const int64_t num_tokens = t->ne[1];
    if (token_dim <= 0 || num_tokens <= 0 || token_dim > 65536 || num_tokens > 1024 ||
        t->ne[2] != 1 || t->ne[3] != 1) {
        LOG_WARN("PuLID id-embedding: implausible shape [%lld, %lld] in '%s'",
                 (long long)token_dim, (long long)num_tokens, path);
        gguf_free(gguf_ctx);
        ggml_free(ctx_data);
        return empty;
    }

    const size_t n_elem = (size_t)token_dim * (size_t)num_tokens;
    sd::Tensor<float> out({token_dim, num_tokens, 1});
    float* dst = out.data();
    if (t->type == GGML_TYPE_F32) {
        memcpy(dst, t->data, n_elem * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t* src = reinterpret_cast<const ggml_fp16_t*>(t->data);
        for (size_t i = 0; i < n_elem; i++) {
            dst[i] = ggml_fp16_to_fp32(src[i]);
        }
    } else if (t->type == GGML_TYPE_BF16) {
        const ggml_bf16_t* src = reinterpret_cast<const ggml_bf16_t*>(t->data);
        for (size_t i = 0; i < n_elem; i++) {
            dst[i] = ggml_bf16_to_fp32(src[i]);
        }
    } else {
        LOG_WARN("PuLID id-embedding: unsupported tensor type %s in '%s'",
                 ggml_type_name(t->type), path);
        gguf_free(gguf_ctx);
        ggml_free(ctx_data);
        return empty;
    }

    LOG_INFO("PuLID id-embedding: loaded [%lld, %lld] type=%s from '%s'",
             (long long)token_dim, (long long)num_tokens, ggml_type_name(t->type), path);
    gguf_free(gguf_ctx);
    ggml_free(ctx_data);
    return out;
}

// PuLID-Flux identity injection as a generation extension.
//
// Unlike PhotoMaker, PuLID does NOT modify the conditioning -- it injects an
// identity embedding via cross-attention *inside* the Flux denoise forward (the
// pulid_ca.* blocks). Those cross-attention weights are part of the Flux
// diffusion model and are loaded into the model tensor map before the model is
// constructed (see SDImpl ctor, gated on sd_ctx_params.pulid_weights_path), so
// this extension does not own a separate model. Its job is purely runtime:
//   - prepare_condition: load the per-generation id-embedding file.
//   - before_diffusion:  hand that embedding (+ weight) to FluxDiffusionExtra,
//                        which flux.hpp reads to drive the pulid_ca injection.
struct PuLIDExtension : public GenerationExtension {
    bool enabled = false;
    sd::Tensor<float> id_embedding;  // per-generation; empty when PuLID is off for this request
    float id_weight = 1.0f;

    const char* name() const override {
        return "pulid";
    }

    bool is_enabled() const override {
        return enabled;
    }

    bool init(const GenerationExtensionInitContext& ctx) override {
        enabled = strlen(SAFE_STR(ctx.params->pulid_weights_path)) > 0;
        return true;
    }

    void reset_runtime_condition() override {
        id_embedding = {};
        id_weight    = 1.0f;
    }

    bool prepare_condition(GenerationExtensionConditionContext& ctx) override {
        reset_runtime_condition();
        if (!enabled) {
            return false;
        }
        id_embedding = load_pulid_id_embedding(ctx.pulid_params.id_embedding_path);
        id_weight    = ctx.pulid_params.id_weight;
        return false;  // PuLID does not modify the conditioning
    }

    void before_diffusion(DiffusionParams& params, int /*step*/) const override {
        if (!enabled || id_embedding.empty()) {
            return;
        }
        if (auto* flux_extra = std::get_if<FluxDiffusionExtra>(&params.extra)) {
            flux_extra->pulid_id        = &id_embedding;
            flux_extra->pulid_id_weight = id_weight;
        }
    }
};

std::shared_ptr<GenerationExtension> create_pulid_extension() {
    return std::make_shared<PuLIDExtension>();
}
