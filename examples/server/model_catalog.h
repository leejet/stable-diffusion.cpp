#ifndef __SD_EXAMPLES_SERVER_MODEL_CATALOG_H__
#define __SD_EXAMPLES_SERVER_MODEL_CATALOG_H__

#include <chrono>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <string>

#include "runtime.h"

struct ModelRoot {
    std::filesystem::path path;
    std::string id;
    std::string label;
    bool primary = false;
};

struct ModelFile {
    std::string category;
    std::string name;
    std::string relative_path;
    ModelRoot root;
};

std::string model_path_key(const std::filesystem::path& path);
bool model_path_inside(const std::filesystem::path& root, const std::filesystem::path& path);
ModelRoot make_model_root(const std::filesystem::path& path, bool primary);
std::vector<ModelRoot> get_model_roots(const SDSvrParams& params);
std::vector<std::filesystem::path> model_category_directories(const ModelRoot& root, const std::string& category);

struct ServerModels {
    std::shared_mutex runtime_mutex;
    std::mutex catalog_mutex;
    std::mutex status_mutex;
    SDCtxPtr context;
    SDContextParams base_context;
    SDGenerationParams base_defaults;
    std::map<std::string, json> recipes;
    std::map<std::string, ModelFile> files;
    std::chrono::steady_clock::time_point load_started_at;
    json active_recipe  = nullptr;
    json catalog_errors = json::array();
    json status         = {{"state", "empty"}, {"recipe", nullptr}, {"models", json::object()}, {"error", nullptr}, {"load_elapsed_seconds", nullptr}};
};

void refresh_model_catalog(ServerRuntime& runtime);
json model_status(ServerRuntime& runtime);
void initialize_server_models(ServerRuntime& runtime);

json resolve_recipe_features(const json& recipe);
json model_generation_features(const ServerRuntime& runtime);

// Resolves the load request into context/generation params. `body` is updated
// so `models` reflects the fully resolved selection (recipe defaults merged in),
// letting the caller report exactly which components got loaded.
void prepare_model_load(ServerRuntime& runtime, json& body, SDContextParams& context, SDGenerationParams& defaults, json* loaded_recipe = nullptr);

// True when both parameter sets resolve to the same component files. File IDs
// stay stable if a model directory symlink is retargeted inside the configured
// root, so the idempotent-load check compares the resolved paths too.
bool component_paths_match(const SDContextParams& a, const SDContextParams& b);

#endif
