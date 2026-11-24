#include "dgl/models/photomaker.h"

#include <algorithm>

#include "core/image_tensor.h"
#include "core/util.h"
#include "dgl/models/component_sources.h"
#include "extensions/photomaker_embedding.h"
#include "model/adapter/pmid.hpp"

namespace sd::dgl::models {
    PhotoMaker::PhotoMaker(std::shared_ptr<ModelManager> manager, std::shared_ptr<SDBackendPool> backends)
        : Model(std::move(manager)),
          runtime_(*this, std::move(backends), ModelComponent::PhotoMaker, SDBackendModule::PHOTOMAKER, [this] { return build_runners(); }) {}

    bool PhotoMaker::init(ModelLoadParams params) {
        return runtime_.init(std::move(params), load_component_catalog<ModelComponent::PhotoMaker>);
    }

    bool PhotoMaker::build_runners() {
        if (!sd_version_is_sdxl(version()))
            throw std::runtime_error("PhotoMaker requires the SDXL family");
        const auto& weights = model_weights->loader().get_tensor_storage_map();
        const bool v2       = std::any_of(weights.begin(), weights.end(), [](const auto& entry) {
            return starts_with(entry.first, "pmid.qformer_perceiver.");
        });
        runner_             = std::make_shared<PhotoMakerIDEncoder>(runtime_.backend_for(), weights, "pmid", version(),
                                                                    v2 ? PM_VERSION_2 : PM_VERSION_1, 0.f, model_weights);
        runner_->set_scale_overrides(runtime_.params.linear_scale,
                                     runtime_.params.attn_scale);
        return runtime_.register_runner_params(runtime_.component, runner_);
    }

    Tensor<float> PhotoMaker::project_images(const std::vector<Tensor<float>>& images, const Tensor<float>* embedding) {
        const bool v2       = runner_->get_version() == PM_VERSION_2;
        const int64_t count = images.size();
        if (count == 0 || count > 75 / (v2 ? 2 : 1))
            throw std::runtime_error("PhotoMaker image count exceeds the available prompt slots");
        if (v2 && (!embedding || embedding->shape() != std::vector<int64_t>{sd::photomaker::id_embedding_width, count}))
            throw std::runtime_error("PhotoMaker v2 requires a float32 embedding with 512 values per image");
        if (!v2 && embedding)
            throw std::runtime_error("PhotoMaker identity embeddings require v2");

        constexpr int64_t image_size = 224;
        Tensor<float> pixels({image_size, image_size, 3, count});
        auto target = pixels.values().begin();
        for (const auto& image : images) {
            if (image.dim() != 4 || image.shape()[3] != 1 || image.shape()[2] == 2)
                throw std::runtime_error("PhotoMaker expects individual grayscale, RGB or RGBA images");
            const auto preprocessed = clip_preprocess(image.shape()[2] == 3 ? image : sd::ensure_image_tensor_channels(image, 3),
                                                      image_size, image_size);
            target                  = std::copy(preprocessed.values().begin(), preprocessed.values().end(), target);
        }
        static const Tensor<float> empty;
        return runner_->compute_tokens(n_threads, pixels, embedding ? *embedding : empty);
    }

    Tensor<float> PhotoMaker::fuse_prompt(const Tensor<float>& prompt, const Tensor<float>& tokens, const Tensor<uint8_t>& mask) {
        const bool single_prompt = prompt.dim() == 2 || (prompt.dim() == 3 && prompt.shape()[2] == 1);
        if (tokens.dim() != 2 || tokens.shape()[0] != 2048 || tokens.empty() ||
            !single_prompt || prompt.shape()[0] != 2048 || mask.dim() != 1 || prompt.shape()[1] != mask.numel())
            throw std::runtime_error("PhotoMaker requires SDXL text and identity tokens and one mask entry per text token");
        auto begin       = std::find(mask.values().begin(), mask.values().end(), uint8_t{1});
        const auto slots = tokens.shape()[1];
        if (mask.values().end() - begin < slots ||
            !std::all_of(begin, begin + slots, [](uint8_t value) { return value == 1; }) || !std::all_of(mask.values().begin(), begin, [](uint8_t value) { return value == 0; }) || !std::all_of(begin + slots, mask.values().end(), [](uint8_t value) { return value == 0; }))
            throw std::runtime_error("PhotoMaker mask requires one contiguous slot for each identity token");
        std::vector<bool> selected(mask.values().begin(), mask.values().end());
        return runner_->compute(n_threads, {}, prompt, tokens, selected);
    }
}
