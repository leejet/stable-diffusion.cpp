#ifndef __SD_DGL_MODELS_LORA_H__
#define __SD_DGL_MODELS_LORA_H__

#include <memory>
#include <string>
#include <vector>

#include "core/option_types.h"
#include "model_manager.h"

struct LoraModel;
struct WeightAdapter;
class SDBackendManager;
enum class SDBackendModule;

namespace sd::dgl::models {

    struct LoraFile {
        std::string path;
    };

    struct LoraBinding {
        std::shared_ptr<const LoraFile> file;
        float weight = 1.0f;
    };

    struct LoraConfig {
        std::vector<LoraBinding> bindings;
        lora_apply_mode_t mode = LORA_APPLY_AUTO;
    };

    lora_apply_mode_t resolve_lora_mode(lora_apply_mode_t mode, const ModelLoader& loader, SDBackendManager& backends, SDBackendModule module, bool row_split = false);

    class LoraExecution {
        std::vector<std::shared_ptr<LoraModel>> models_;
        std::shared_ptr<WeightAdapter> adapter_;
        SDBackendModule module_{};
        bool ready_ = false;

    public:
        void end();
        void reset();
        bool prepare(const LoraConfig& config, const std::shared_ptr<ModelManager::ResourceHandle>& weights, SDVersion version, SDBackendManager& backends, SDBackendModule module, int n_threads, bool immediately);
        const std::shared_ptr<WeightAdapter>& adapter() const { return adapter_; }
    };

}  // namespace sd::dgl::models

#endif  // __SD_DGL_MODELS_LORA_H__
