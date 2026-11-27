#include "dgl/models/component_sources.h"

#include <algorithm>

#include "core/file_access.h"
#include "core/util.h"
#include "dgl/models/audio_encoder.h"
#include "dgl/models/audio_vae.h"
#include "dgl/models/control.h"
#include "dgl/models/ip_adapter.h"
#include "dgl/models/photomaker.h"
#include "dgl/models/pulid.h"
#include "dgl/models/vae.h"
#include "model_loader.h"

namespace sd::dgl::models {

    SDVersion resolve_family(SDVersion family, const std::string& family_from, ModelLoader::Dependencies* dependencies, ModelLoader* metadata) {
        if (family_from.empty())
            return family;
        if (family != VERSION_COUNT)
            throw std::runtime_error("family_from requires family = \"auto\"");

        ModelLoader probe;
        ModelLoader::FileId file;
        if (!probe.add_file(family_from, "", &file))
            throw std::runtime_error("cannot read model family from '" + family_from + "'");
        family = probe.get_sd_version();
        if (family == VERSION_COUNT) {
            probe.set_file_prefix(file, "model.diffusion_model.");
            family = probe.get_sd_version();
        }
        if (family == VERSION_COUNT)
            throw std::runtime_error("cannot identify model family from '" + family_from + "'");
        if (dependencies)
            *dependencies = probe.dependencies();
        if (metadata)
            *metadata = std::move(probe);
        return family;
    }

    static std::string family_probe_key(const std::string& path) {
        require_file_access(path, SD_DGL_FILE_READ_MODEL);
        std::error_code error;
        auto canonical = std::filesystem::weakly_canonical(std::filesystem::u8path(path), error);
        return error ? path : canonical.u8string();
    }

    const FamilyProbeCache::Entry* FamilyProbeCache::find(const std::string& path) {
        auto entry = entries_.find(path);
        if (entry == entries_.end())
            return nullptr;
        bool changed = false;
        if (ModelLoader::dependencies_changed(entry->second.dependencies, changed) && !changed)
            return &entry->second;
        entries_.erase(entry);
        return nullptr;
    }

    void FamilyProbeCache::store(const std::string& path, Entry entry) {
        bool changed = false;
        if (!ModelLoader::dependencies_changed(entry.dependencies, changed) || changed)
            throw std::runtime_error("model sources changed while detecting family from '" + path + "'");
        const auto family = entry.family;
        entries_.insert_or_assign(path, std::move(entry));
        LOG_INFO("Detected model family: %s (%s)", sd_version_name(family), std::filesystem::u8path(path).filename().u8string().c_str());
    }

    SDVersion FamilyProbeCache::resolve(SDVersion family, const std::string& family_from, ModelLoader::Dependencies& dependencies) {
        dependencies.clear();
        if (family_from.empty())
            return family;
        if (family != VERSION_COUNT)
            throw std::runtime_error("family_from requires family = \"auto\"");
        const auto path = family_probe_key(family_from);
        std::lock_guard<std::mutex> lock(mutex_);
        if (const auto* entry = find(path)) {
            dependencies = entry->dependencies;
            return entry->family;
        }
        family = resolve_family(family, path, &dependencies);
        store(path, {family, dependencies});
        return family;
    }

    void FamilyProbeCache::remember(const std::string& source, SDVersion family, const ModelLoader& loader) {
        if (family == VERSION_COUNT)
            return;
        const auto path = family_probe_key(source);
        std::lock_guard<std::mutex> lock(mutex_);
        if (const auto* entry = find(path); entry && entry->family == family)
            return;
        store(path, {family, loader.dependencies()});
    }

    bool init_component_loader(ModelLoader& loader, const ModelLoadParams& params, ModelComponent component, SDVersion& family, const WeightSources& sources, FamilyProbeCache* family_probes) {
        const std::string path = params.path;
        ModelLoader::FileId file;
        if (!loader.add_file(path, "", &file)) {
            return false;
        }
        const auto& weights = loader.get_tensor_storage_map();
        auto has_prefix     = [&](const char* prefix) {
            return std::any_of(weights.begin(), weights.end(), [prefix](const auto& entry) {
                return starts_with(entry.first, prefix);
            });
        };
        const auto detected = loader.get_sd_version();
        if (family == VERSION_COUNT)
            family = detected;
        std::string prefix;
        switch (component) {
            case ModelComponent::Diffusion:
                if (!has_prefix("model.diffusion_model.") &&
                    !has_prefix("diffusion_model.") && !has_prefix("transformer.") && !has_prefix("unet.")) {
                    prefix = "model.diffusion_model.";
                }
                break;
            case ModelComponent::VAE:
                if (detected == VERSION_COUNT && !has_prefix("first_stage_model.") && !has_prefix("vae.")) {
                    prefix = "vae.";
                }
                if (family == VERSION_COUNT && detected == VERSION_COUNT) {
                    for (const auto* key : {"decoder.conv1.weight", "first_stage_model.decoder.conv1.weight", "vae.decoder.conv1.weight"}) {
                        auto conv = weights.find(key);
                        if (conv == weights.end() || conv->second.n_dims < 4) {
                            continue;
                        }
                        const auto channels = conv->second.ne[conv->second.n_dims - 2];
                        if (channels == 16 || channels == 48) {
                            family = channels == 48 ? VERSION_WAN2_2_TI2V : VERSION_WAN2;
                            break;
                        }
                    }
                }
                break;
            case ModelComponent::Conditioner:
                if (detected == VERSION_COUNT && !has_prefix("cond_stage_model.") && !has_prefix("conditioner.embedders.") &&
                    !has_prefix("text_encoders.") && !has_prefix("te.") && !has_prefix("clip_l.")) {
                    if (has_prefix("encoder.block.") || has_prefix("enc.blk.") || has_prefix("encoder.blk.")) {
                        prefix = "text_encoders.t5xxl.transformer.";
                        if (family == VERSION_COUNT && detected == VERSION_COUNT) {
                            family = VERSION_WAN2;
                        }
                    } else if (has_prefix("text_model.")) {
                        prefix = "clip_l.";
                    } else {
                        prefix = "text_encoders.llm.";
                    }
                }
                break;
            case ModelComponent::ControlNet:
                if (family == VERSION_COUNT && weights.count("control_all_x_embedder.2-1.weight"))
                    family = VERSION_Z_IMAGE;
                break;
            case ModelComponent::AudioVAE:
            case ModelComponent::IPAdapter:
                break;
            case ModelComponent::LatentUpsampler: {
                const auto looks_like_ltx = [&](const std::string& p) {
                    return weights.count(p + "initial_conv.weight") && weights.count(p + "initial_norm.weight");
                };
                const auto looks_like_h3 = [&](const std::string& p) {
                    return weights.count(p + "conv_in.weight") && weights.count(p + "embed.2.weight");
                };
                const bool ltx_prefixed = looks_like_ltx("ltx_latent_upsampler.");
                const bool h3_prefixed  = looks_like_h3("h3_latent_upscaler.");
                const bool ltx          = ltx_prefixed || looks_like_ltx("");
                const bool h3           = h3_prefixed || looks_like_h3("");
                if (ltx == h3)
                    throw std::runtime_error("cannot identify the latent upscaler checkpoint as LTX or MiniMax H3");
                const auto detected_family = ltx ? VERSION_LTXAV : VERSION_MINIMAX_H3;
                if (family != VERSION_COUNT && family != detected_family)
                    throw std::runtime_error("latent upscaler weights do not match the requested family");
                family = detected_family;
                if (!ltx_prefixed && !h3_prefixed)
                    prefix = ltx ? "ltx_latent_upsampler." : "h3_latent_upscaler.";
                break;
            }
            case ModelComponent::PuLID:
                if (!has_prefix("model.diffusion_model."))
                    prefix = "model.diffusion_model.";
                if (family == VERSION_COUNT)
                    family = VERSION_FLUX;
                break;
            case ModelComponent::AnimateDiff:
                if (has_prefix("motion_module."))
                    prefix = "model.diffusion_model.";
                else if (!has_prefix("model.diffusion_model.motion_module."))
                    prefix = "model.diffusion_model.motion_module.";
                if (family == VERSION_COUNT)
                    family = VERSION_SD1;
                break;
            case ModelComponent::PhotoMaker:
                if (!has_prefix("pmid.")) {
                    prefix = "pmid.";
                }
                break;
            case ModelComponent::CLIPVision:
                if (!has_prefix("clip_vision.")) {
                    prefix = "clip_vision.";
                }
                if (family == VERSION_COUNT) {
                    family = VERSION_WAN2;
                }
                break;
            case ModelComponent::AudioEncoder:
                if (!has_prefix("wav2vec2.")) {
                    prefix = "wav2vec2.";
                }
                if (family == VERSION_COUNT) {
                    family = VERSION_WAN2_2_S2V;
                }
                break;
            default:
                return false;
        }
        if (!loader.set_file_prefix(file, prefix)) {
            return false;
        }
        if (family_probes && params.family == VERSION_COUNT) {
            const auto source_family = detected == VERSION_COUNT ? loader.get_sd_version() : detected;
            family_probes->remember(path, source_family, loader);
        }
        for (const auto& [source, source_prefix] : sources) {
            if (!source.empty() && !loader.init_from_file(source, source_prefix))
                return false;
        }
        loader.convert_tensors_name(family);
        if (family == VERSION_COUNT) {
            family = loader.get_sd_version();
        }
        if (family == VERSION_COUNT) {
            LOG_ERROR("cannot identify component family for '%s'; specify family", path.c_str());
            return false;
        }
        loader.set_wtype_override(sd_type_to_ggml_type(params.wtype), params.tensor_type_rules.c_str());
        return true;
    }

}  // namespace sd::dgl::models
