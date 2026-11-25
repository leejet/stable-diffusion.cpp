#include "dgl/bindings/lora.h"

#include <cmath>

#include "core/file_access.h"
#include "dgl/context.h"
#include "dgl/executor.h"
#include "dgl/models/lora.h"

namespace sd::dgl {

    static std::shared_ptr<models::LoraFile> lora_file(std::string path) {
        require_file_access(path, SD_DGL_FILE_READ_MODEL);
        if (path.empty())
            throw std::runtime_error("LoRA path is empty");
        return std::make_shared<models::LoraFile>(models::LoraFile{std::move(path)});
    }

    static float lora_weight(const Values& args) {
        const auto weight = float(get_number(args, "weight"));
        if (!std::isfinite(weight))
            throw std::runtime_error("LoRA weight must be finite");
        return weight;
    }

    PreparedResource load_lora(Values& args) {
        PreparedResource prepared;
        prepared.revision = nlohmann::json::object();
        auto path         = file_key(get_string(args, "path"), prepared.revision);
        prepared.key      = {{"path", path}};
        prepared.load     = [path = std::move(path)](ResourceLoadRequest&) {
            return Value::object(lora_file(path));
        };
        return prepared;
    }

    Value apply_lora(Values& args, RunState& run) {
        const auto& model = args.at("model");
        auto config       = model.loras ? std::make_shared<models::LoraConfig>(*model.loras)
                                        : std::make_shared<models::LoraConfig>();
        if (args.count("mode"))
            config->mode = str_to_lora_apply_mode(get_string(args, "mode").c_str());
        config->bindings.push_back({args.at("lora").as<models::LoraFile>(), lora_weight(args)});
        return run.context.derive_model(model, std::move(config));
    }
}  // namespace sd::dgl
