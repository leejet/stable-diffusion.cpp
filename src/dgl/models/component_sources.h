#ifndef __SD_DGL_MODELS_COMPONENT_SOURCES_H__
#define __SD_DGL_MODELS_COMPONENT_SOURCES_H__

#include <mutex>
#include <utility>
#include <vector>

#include "dgl/models/model_load_params.h"
#include "model.h"
#include "model_component.h"
#include "model_loader.h"
#include "stable-diffusion.h"

class ModelLoader;

namespace sd::dgl::models {
    SDVersion resolve_family(SDVersion family, const std::string& family_from, ModelLoader::Dependencies* dependencies = nullptr, ModelLoader* metadata = nullptr);

    class FamilyProbeCache {
        struct Entry {
            SDVersion family;
            ModelLoader::Dependencies dependencies;
        };

        std::mutex mutex_;
        std::map<std::string, Entry> entries_;

        const Entry* find(const std::string& path);
        void store(const std::string& path, Entry entry);

    public:
        SDVersion resolve(SDVersion family, const std::string& family_from, ModelLoader::Dependencies& dependencies);
        void remember(const std::string& path, SDVersion family, const ModelLoader& loader);
    };

    using WeightSources = std::vector<std::pair<std::string, std::string>>;

    bool init_component_loader(ModelLoader& loader, const ModelLoadParams& params, ModelComponent component, SDVersion& family, const WeightSources& sources = {}, FamilyProbeCache* family_probes = nullptr);

    template <ModelComponent Component>
    bool load_component_catalog(ModelLoader& loader, const ModelLoadParams& params, SDVersion& family, FamilyProbeCache& family_probes) {
        return init_component_loader(loader, params, Component, family, {}, &family_probes);
    }
}

#endif  // __SD_DGL_MODELS_COMPONENT_SOURCES_H__
