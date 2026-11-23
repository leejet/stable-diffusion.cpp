#ifndef __SD_EXAMPLES_SERVER_MODEL_CATALOG_H__
#define __SD_EXAMPLES_SERVER_MODEL_CATALOG_H__

#include <map>
#include <mutex>
#include <set>
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

struct ServerFileAccess {
    std::vector<ModelRoot> models;
    std::filesystem::path output;
    std::vector<std::filesystem::path> model_bundles;
    std::set<std::string> configured_files;
};

struct ServerModels {
    std::shared_mutex model_mutex;
    std::mutex catalog_mutex;
    ServerFileAccess file_access;
    ServerFileAccess workspace_file_access;
    SDCtxPtr context;
    SDContextParams base_context;
    SDGenerationParams base_defaults;
    std::map<std::string, json> recipes;
    std::map<std::string, ModelFile> files;
    json current_recipe = nullptr;
    json current_files = json::object();
    uint64_t revision = 0;
    json catalog_errors = json::array();
};

struct ServerModelSelection {
    SDContextParams context;
    SDGenerationParams defaults;
    json recipe;
    json files;
    SDOptPtr options;
};

void refresh_model_catalog(ServerRuntime& runtime);
json model_status(const ServerRuntime& runtime);
void initialize_server_models(ServerRuntime& runtime);

json resolve_recipe_features(const json& recipe);
json model_generation_features(const ServerRuntime& runtime);

ServerModelSelection resolve_model_selection(ServerRuntime& runtime, const json& body, SDMode mode);
// The caller holds the generation mutex across the switch and generation.
bool set_current_model(ServerRuntime& runtime, ServerModelSelection&& selection, SDMode mode, std::string& error_message);
void add_startup_model_sources(ServerFileAccess& policy, const SDContextParams& config);

#endif
