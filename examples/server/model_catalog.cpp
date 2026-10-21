#include "model_catalog.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "common/log.h"
#include "recipes_builtin.h"

namespace fs = std::filesystem;

struct ModelSlot {
    const char* id;
    const char* label;
    std::vector<std::string> categories;
    std::string SDContextParams::*field;
};

static const ModelSlot k_model_slots[] = {
    {"model", "Checkpoint", {"checkpoints"}, &SDContextParams::model_path},
    {"diffusion_model", "Diffusion model", {"diffusion_models"}, &SDContextParams::diffusion_model_path},
    {"high_noise_diffusion_model", "High-noise diffusion model", {"diffusion_models"}, &SDContextParams::high_noise_diffusion_model_path},
    {"uncond_diffusion_model", "Unconditional diffusion model", {"diffusion_models"}, &SDContextParams::uncond_diffusion_model_path},
    {"clip_l", "CLIP-L", {"text_encoders"}, &SDContextParams::clip_l_path},
    {"clip_g", "CLIP-G", {"text_encoders"}, &SDContextParams::clip_g_path},
    {"clip_vision", "CLIP Vision", {"clip_vision"}, &SDContextParams::clip_vision_path},
    {"t5xxl", "T5", {"text_encoders"}, &SDContextParams::t5xxl_path},
    {"llm", "LLM Text Encoder", {"text_encoders"}, &SDContextParams::llm_path},
    {"llm_vision", "LLM Vision Encoder", {"text_encoders", "clip_vision"}, &SDContextParams::llm_vision_path},
    {"tokenizer", "Tokenizer", {"tokenizers"}, &SDContextParams::tokenizer},
    {"vae", "VAE", {"vae"}, &SDContextParams::vae_path},
    {"audio_vae", "Audio VAE", {"vae"}, &SDContextParams::audio_vae_path},
    {"audio_encoder", "Audio Encoder", {"text_encoders"}, &SDContextParams::audio_encoder_path},
    {"embeddings_connectors", "Embeddings Connectors", {"text_encoders"}, &SDContextParams::embeddings_connectors_path},
    {"tae", "TAE", {"vae", "vae_approx"}, &SDContextParams::taesd_path},
    {"taesd", "TAE", {"vae_approx", "vae"}, &SDContextParams::taesd_path},
    {"control_net", "ControlNet", {"controlnet"}, &SDContextParams::control_net_path},
    {"ip_adapter", "IP-Adapter", {"ipadapter"}, &SDContextParams::ip_adapter_path},
    {"motion_module", "Motion Module", {"motion_modules"}, &SDContextParams::motion_module_path},
    {"photo_maker", "PhotoMaker", {"photomaker"}, &SDContextParams::photo_maker_path},
    {"pulid_weights", "PuLID", {"pulid"}, &SDContextParams::pulid_weights_path},
};

static const ModelSlot* find_model_slot(const std::string& id) {
    for (const auto& slot : k_model_slots) {
        if (id == slot.id) {
            return &slot;
        }
    }
    return nullptr;
}

static bool matches_model_category(const ModelSlot& slot, const std::string& category) {
    return std::find(slot.categories.begin(), slot.categories.end(), category) != slot.categories.end();
}

static void clear_model_paths(SDContextParams& params) {
    for (const auto& slot : k_model_slots) {
        (params.*slot.field).clear();
    }
}

std::string model_path_key(const fs::path& path) {
#ifdef _WIN32
    const auto value = path.generic_wstring();
    const int length = LCMapStringEx(LOCALE_NAME_INVARIANT, LCMAP_LOWERCASE, value.data(),
                                     static_cast<int>(value.size()), nullptr, 0, nullptr, nullptr, 0);
    if (length > 0) {
        std::wstring folded(length, L'\0');
        LCMapStringEx(LOCALE_NAME_INVARIANT, LCMAP_LOWERCASE, value.data(),
                      static_cast<int>(value.size()), &folded[0], length, nullptr, nullptr, 0);
        return fs::path(folded).generic_u8string();
    }
#endif
    return path.generic_u8string();
}

bool model_path_inside(const fs::path& root, const fs::path& path) {
    auto parent = root.begin();
    auto child  = path.begin();
    for (; parent != root.end(); ++parent, ++child) {
        if (child == path.end() || model_path_key(*parent) != model_path_key(*child)) {
            return false;
        }
    }
    return child != path.end();
}

ModelRoot make_model_root(const fs::path& path, bool primary) {
    ModelRoot root;
    const auto absolute = fs::absolute(path);
    std::error_code ec;
    root.path = fs::weakly_canonical(absolute, ec);
    if (ec)
        root.path = absolute.lexically_normal();
    root.primary  = primary;
    uint64_t hash = 14695981039346656037ULL;
    for (unsigned char c : model_path_key(root.path)) {
        hash ^= c;
        hash *= 1099511628211ULL;
    }
    std::ostringstream id;
    id << '@' << std::hex << std::setfill('0') << std::setw(16) << hash;
    root.id    = id.str();
    root.label = root.path.filename().u8string() + " [" + root.id.substr(1, 8) + "]";
    return root;
}

std::vector<ModelRoot> get_model_roots(const SDSvrParams& params) {
    const auto directories = params.model_dirs.empty() ? std::vector<std::string>{"models"} : params.model_dirs;
    std::vector<ModelRoot> roots;
    std::set<std::string> seen;
    for (size_t i = 0; i < directories.size(); ++i) {
        auto root = make_model_root(fs::u8path(directories[i]), i == 0);
        if (seen.insert(model_path_key(root.path)).second)
            roots.push_back(std::move(root));
    }
    return roots;
}

std::vector<fs::path> model_category_directories(const ModelRoot& root, const std::string& category) {
    std::vector<fs::path> directories;
    if (!fs::exists(root.path))
        return directories;
    for (const auto& entry : fs::directory_iterator(root.path)) {
        auto name = entry.path().filename().u8string();
        std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return std::tolower(c); });
        if (name == category || (category == "checkpoints" && name == "stable-diffusion") ||
            (category == "loras" && name == "lora")) {
            directories.push_back(entry.path());
        }
    }
    std::sort(directories.begin(), directories.end(), [&](const fs::path& a, const fs::path& b) {
        const bool a_standard = a.filename().u8string() == category;
        const bool b_standard = b.filename().u8string() == category;
        return a_standard != b_standard ? a_standard : a < b;
    });
    return directories;
}

static bool valid_identifier(const std::string& id) {
    return !id.empty() && id.size() <= 128 &&
           std::all_of(id.begin(), id.end(), [](unsigned char c) {
               return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                      (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.';
           });
}

static void check_keys(const json& object, const std::set<std::string>& keys) {
    if (!object.is_object()) {
        throw std::runtime_error("expected an object");
    }
    for (auto it = object.begin(); it != object.end(); ++it) {
        if (!keys.count(it.key())) {
            throw std::runtime_error("unknown field: " + it.key());
        }
    }
}

static void apply_recipe_defaults(const json& defaults, SDGenerationParams& params) {
    check_keys(defaults, {"width", "height", "steps", "cfg_scale", "strength", "distilled_guidance", "sampling_method", "scheduler", "custom_sigmas",
                          "video_frames", "fps", "flow_shift", "moe_boundary", "high_noise_steps", "high_noise_cfg_scale",
                          "high_noise_flow_shift", "high_noise_sampling_method", "high_noise_scheduler", "img_cfg_scale", "shifted_timestep"});
    for (const char* key : {"width", "height", "steps", "video_frames", "fps", "high_noise_steps", "shifted_timestep"}) {
        if (!defaults.contains(key)) {
            continue;
        }
        if (!defaults[key].is_number_integer()) {
            throw std::runtime_error(std::string(key) + " must be an integer");
        }
        const int64_t value     = defaults[key].get<int64_t>();
        const std::string field = key;
        const bool dimension    = field == "width" || field == "height";
        const int64_t maximum   = dimension || field == "video_frames" ? 4096 : field == "fps" ? 240
                                                                                               : 1000;
        const int64_t minimum   = dimension ? 64 : field == "shifted_timestep" ? 0
                                                                               : 1;
        if ((value < minimum || value > maximum) && !(field == "high_noise_steps" && value == -1)) {
            throw std::runtime_error(std::string(key) + " is out of range");
        }
        if (std::string(key) == "width")
            params.width = static_cast<int>(value);
        if (std::string(key) == "height")
            params.height = static_cast<int>(value);
        if (std::string(key) == "steps")
            params.sample_params.sample_steps = static_cast<int>(value);
        if (field == "shifted_timestep")
            params.sample_params.shifted_timestep = static_cast<int>(value);
        if (field == "video_frames")
            params.video_frames = static_cast<int>(value);
        if (field == "fps")
            params.fps = static_cast<int>(value);
        if (field == "high_noise_steps")
            params.high_noise_sample_params.sample_steps = static_cast<int>(value);
    }
    for (const char* key : {"cfg_scale", "strength", "moe_boundary", "img_cfg_scale", "distilled_guidance", "flow_shift", "high_noise_cfg_scale", "high_noise_flow_shift"}) {
        if (!defaults.contains(key)) {
            continue;
        }
        if (!defaults[key].is_number()) {
            throw std::runtime_error(std::string(key) + " must be a number");
        }
        const float value = defaults[key].get<float>();
        if (!std::isfinite(value) || value < 0 || value > ((std::string(key) == "strength" || std::string(key) == "moe_boundary") ? 1.f : 1000.f)) {
            throw std::runtime_error(std::string(key) + " is out of range");
        }
        if (std::string(key) == "strength")
            params.strength = value;
        if (std::string(key) == "moe_boundary")
            params.moe_boundary = value;
        if (std::string(key) == "cfg_scale")
            params.sample_params.guidance.txt_cfg = value;
        if (std::string(key) == "img_cfg_scale")
            params.sample_params.guidance.img_cfg = value;
        if (std::string(key) == "distilled_guidance")
            params.sample_params.guidance.distilled_guidance = value;
        if (std::string(key) == "flow_shift")
            params.sample_params.flow_shift = value;
        if (std::string(key) == "high_noise_cfg_scale")
            params.high_noise_sample_params.guidance.txt_cfg = value;
        if (std::string(key) == "high_noise_flow_shift")
            params.high_noise_sample_params.flow_shift = value;
    }
    if (defaults.contains("custom_sigmas")) {
        const auto& sigmas = defaults.at("custom_sigmas");
        if (!sigmas.is_array() || sigmas.size() == 1 || sigmas.size() > 1001)
            throw std::runtime_error("custom_sigmas must be empty or contain 2-1001 values");
        for (const auto& sigma : sigmas) {
            if (!sigma.is_number() || !std::isfinite(sigma.get<float>()) || sigma.get<float>() < 0)
                throw std::runtime_error("custom_sigmas must contain finite nonnegative numbers");
        }
        params.custom_sigmas = sigmas.get<std::vector<float>>();
    }
    for (const char* key : {"sampling_method", "high_noise_sampling_method"}) {
        if (!defaults.contains(key))
            continue;
        auto& sample    = std::string(key) == "sampling_method" ? params.sample_params : params.high_noise_sample_params;
        const auto name = defaults.at(key).get<std::string>();
        bool found      = false;
        for (int i = 0; i < SAMPLE_METHOD_COUNT; ++i) {
            if (name == sd_sample_method_name(static_cast<sample_method_t>(i))) {
                sample.sample_method = static_cast<sample_method_t>(i);
                found                = true;
                break;
            }
        }
        if (!found)
            throw std::runtime_error(std::string("unknown ") + key);
    }
    for (const char* key : {"scheduler", "high_noise_scheduler"}) {
        if (!defaults.contains(key))
            continue;
        auto& sample    = std::string(key) == "scheduler" ? params.sample_params : params.high_noise_sample_params;
        const auto name = defaults.at(key).get<std::string>();
        bool found      = false;
        for (int i = 0; i < SCHEDULER_COUNT; ++i) {
            if (name == sd_scheduler_name(static_cast<scheduler_t>(i)) ||
                (name == "normal" && i == DISCRETE_SCHEDULER)) {
                sample.scheduler = static_cast<scheduler_t>(i);
                found            = true;
                break;
            }
        }
        if (!found)
            throw std::runtime_error(std::string("unknown ") + key);
    }
}

static void apply_recipe_context(const json& settings, SDContextParams& params) {
    check_keys(settings, {"vae_format", "model_args"});
    if (settings.contains("vae_format")) {
        const auto format = settings.at("vae_format").get<std::string>();
        if (format != "auto" && format != "flux" && format != "sd3" && format != "flux2" && format != "wan")
            throw std::runtime_error("unknown vae_format");
        params.vae_format = format;
    }
    if (settings.contains("model_args")) {
        params.model_args = settings.at("model_args").get<std::string>();
    }
}

static const std::map<std::string, bool> k_recipe_features = {
    {"img2img", true},
    {"mask_image", true},
    {"lora", true},
    {"vae_tiling", true},
    {"cache", true},
    {"cache_dit", true},
    {"preview", true},
    {"flow_shift", false},
    {"distilled_guidance", false},
    {"shifted_timestep", false},
    {"img_cfg", false},
    {"slg", false},
    {"clip_skip", false},
    {"ref_images", false},
    {"controlnet", false},
    {"control_image", false},
    {"ip_adapter", false},
    {"init_image", false},
    {"end_image", false},
    {"control_frames", false},
    {"vace", false},
    {"high_noise", false},
    {"cache_unet", false},
    {"preview_tae", false},
};

static void validate_recipe_features(const json& recipe) {
    for (const char* key : {"features", "disabled_features"}) {
        if (!recipe.contains(key))
            continue;
        const auto& values = recipe.at(key);
        if (!values.is_array() || values.size() > k_recipe_features.size())
            throw std::runtime_error(std::string(key) + " must be an array of feature names");
        std::set<std::string> seen;
        for (const auto& value : values) {
            const auto name = value.get<std::string>();
            if (!k_recipe_features.count(name) || !seen.insert(name).second)
                throw std::runtime_error("unknown or duplicate feature: " + name);
        }
    }
}

json resolve_recipe_features(const json& recipe) {
    json features = k_recipe_features;
    for (const auto& slot : recipe.at("slots")) {
        const auto id = slot.at("id").get<std::string>();
        if (id == "ip_adapter")
            features["ip_adapter"] = true;
        if (id == "control_net")
            features["controlnet"] = true;
        if (id == "high_noise_diffusion_model")
            features["high_noise"] = true;
        if (id == "tae" || id == "taesd")
            features["preview_tae"] = true;
    }
    for (const auto& name : recipe.value("features", json::array()))
        features[name.get<std::string>()] = true;
    for (const auto& name : recipe.value("disabled_features", json::array()))
        features[name.get<std::string>()] = false;
    return features;
}

json model_generation_features(const ServerRuntime& runtime) {
    if (!runtime.sd_ctx || runtime.models->active_recipe.is_null()) {
        json features = k_recipe_features;
        for (auto& feature : features)
            feature = false;
        return features;
    }
    json features = resolve_recipe_features(runtime.models->active_recipe);
    if (runtime.models->active_recipe.value("source", "") == "command_line")
        return features;
    // Optional components only become usable once selected and loaded.
    features["ip_adapter"]  = features.value("ip_adapter", false) && !runtime.ctx_params->ip_adapter_path.empty();
    features["controlnet"]  = features.value("controlnet", false) && !runtime.ctx_params->control_net_path.empty();
    features["high_noise"]  = features.value("high_noise", false) && !runtime.ctx_params->high_noise_diffusion_model_path.empty();
    features["preview_tae"] = features.value("preview_tae", false) && !runtime.ctx_params->taesd_path.empty();
    if (!runtime.ctx_params->pulid_weights_path.empty())
        features["slg"] = false;
    return features;
}

static void validate_recipe(const json& recipe) {
    check_keys(recipe, {"schema_version", "id", "name", "description", "model_url", "slots", "required_any", "defaults", "context", "features", "disabled_features"});
    if (!recipe.at("schema_version").is_number_integer() || recipe.at("schema_version") != 1 ||
        !valid_identifier(recipe.at("id").get<std::string>()) || recipe.at("name").get<std::string>().empty()) {
        throw std::runtime_error("invalid schema_version, id or name");
    }
    if (recipe.contains("description") && !recipe["description"].is_string()) {
        throw std::runtime_error("description must be a string");
    }
    if (recipe.contains("model_url")) {
        const auto url           = recipe.at("model_url").get<std::string>();
        const std::string prefix = "https://huggingface.co/";
        if (url.size() <= prefix.size() || url.size() > 2048 || url.rfind(prefix, 0) != 0 ||
            std::any_of(url.begin(), url.end(), [](unsigned char c) { return std::isspace(c) || std::iscntrl(c); })) {
            throw std::runtime_error("model_url must be a Hugging Face HTTPS URL");
        }
    }
    const auto& slots = recipe.at("slots");
    if (!slots.is_array() || slots.empty() || slots.size() > 32) {
        throw std::runtime_error("slots must be a nonempty array of at most 32 entries");
    }
    std::set<std::string> ids;
    bool has_required_model = false;
    for (const auto& slot : slots) {
        check_keys(slot, {"id", "label", "category", "required"});
        const auto id    = slot.at("id").get<std::string>();
        const auto* spec = find_model_slot(id);
        if (!spec || !ids.insert(id == "taesd" ? "tae" : id).second ||
            (slot.contains("category") && !matches_model_category(*spec, slot.at("category").get<std::string>())) ||
            (slot.contains("label") && slot.at("label").get<std::string>().empty()) || !slot.at("required").is_boolean()) {
            throw std::runtime_error("invalid or duplicate model slot: " + id);
        }
        has_required_model |= (id == "model" || id == "diffusion_model") && slot["required"].get<bool>();
    }
    if (!has_required_model) {
        throw std::runtime_error("a required model or diffusion_model slot is needed");
    }
    if (recipe.contains("required_any")) {
        const auto& groups = recipe.at("required_any");
        if (!groups.is_array() || groups.size() > 32)
            throw std::runtime_error("required_any must be an array of at most 32 slot groups");
        for (const auto& group : groups) {
            if (!group.is_array() || group.size() < 2 || group.size() > slots.size())
                throw std::runtime_error("required_any groups need at least two declared slots");
            std::set<std::string> seen;
            for (const auto& value : group) {
                const auto id = value.get<std::string>();
                if (!seen.insert(id).second || std::none_of(slots.begin(), slots.end(), [&](const json& slot) { return slot.at("id") == id; }))
                    throw std::runtime_error("unknown or duplicate required_any slot: " + id);
            }
        }
    }
    validate_recipe_features(recipe);
    SDContextParams context;
    apply_recipe_context(recipe.value("context", json::object()), context);
    SDGenerationParams defaults;
    apply_recipe_defaults(recipe.value("defaults", json::object()), defaults);
}

static json normalize_recipe(json recipe) {
    validate_recipe(recipe);
    recipe["effective_features"] = resolve_recipe_features(recipe);
    for (auto& slot : recipe.at("slots")) {
        const auto& spec   = *find_model_slot(slot.at("id").get<std::string>());
        slot["label"]      = slot.value("label", std::string(spec.label));
        slot["category"]   = slot.value("category", spec.categories.front());
        slot["categories"] = spec.categories;
    }
    return recipe;
}

static bool supported_model_file(const fs::path& path, const std::string& category) {
    std::string ext = path.extension().u8string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
    if (category == "tokenizers") {
        return ext == ".json";
    }
    return ext == ".gguf" || ext == ".safetensors" || ext == ".sft" ||
           ext == ".ckpt" || ext == ".pt" || ext == ".pth" || ext == ".bin";
}

void refresh_model_catalog(ServerRuntime& runtime) {
    auto& models = *runtime.models;
    std::lock_guard<std::mutex> lock(models.catalog_mutex);
    std::map<std::string, json> recipes;
    std::map<std::string, ModelFile> files;
    json errors = json::array();
    auto report = [&](const std::string& file, const std::string& message) {
        errors.push_back({{"file", file}, {"message", message}});
    };
    for (const char* builtin : k_builtin_recipes) {
        auto recipe      = normalize_recipe(json::parse(builtin));
        const auto id    = recipe.at("id").get<std::string>();
        recipe["source"] = "builtin";
        recipes[id]      = std::move(recipe);
    }

    std::set<std::string> seen_files;
    for (const auto& source : get_model_roots(*runtime.svr_params)) {
        const auto& root  = source.path;
        auto display_path = [&](const fs::path& path) {
            return source.id + "/" + path.lexically_relative(root).generic_u8string();
        };
        const fs::path recipe_dir = root / "recipes";
        std::map<std::string, std::vector<json>> user_recipes;
        try {
            if (fs::exists(recipe_dir)) {
                if (!model_path_inside(root, fs::canonical(recipe_dir)))
                    throw std::runtime_error("recipes directory escapes model directory");
                for (const auto& entry : fs::directory_iterator(recipe_dir)) {
                    const std::string display = display_path(entry.path());
                    if (entry.path().extension() != ".json")
                        continue;
                    try {
                        if (!entry.is_regular_file() || !model_path_inside(root, fs::canonical(entry.path())))
                            throw std::runtime_error("recipe must be a file inside the model directory");
                        if (entry.file_size() > 1024 * 1024)
                            throw std::runtime_error("recipe exceeds 1 MiB");
                        std::ifstream input(entry.path(), std::ios::binary);
                        json recipe;
                        input >> recipe;
                        recipe           = normalize_recipe(std::move(recipe));
                        recipe["source"] = "user";
                        recipe["file"]   = entry.path().filename().u8string();
                        recipe["root"]   = source.id;
                        const auto id    = recipe.at("id").get<std::string>();
                        user_recipes[id].push_back(std::move(recipe));
                    } catch (const std::exception& e) {
                        report(display, e.what());
                    }
                }
            }
        } catch (const std::exception& e) {
            report(display_path(recipe_dir), e.what());
        }
        for (auto& item : user_recipes) {
            if (item.second.size() != 1) {
                for (const auto& recipe : item.second)
                    report(source.id + "/recipes/" + recipe["file"].get<std::string>(), "duplicate user recipe id: " + item.first);
                continue;
            }
            const auto existing = recipes.find(item.first);
            if (existing != recipes.end()) {
                if (existing->second["source"] == "user") {
                    LOG_WARN("recipe '%s' from %s is shadowed by an earlier model directory", item.first.c_str(), source.label.c_str());
                    continue;
                }
                LOG_INFO("user recipe '%s' from %s overrides the builtin", item.first.c_str(), source.label.c_str());
            }
            recipes[item.first] = std::move(item.second.front());
        }

        std::set<std::string> categories;
        for (const auto& slot : k_model_slots)
            categories.insert(slot.categories.begin(), slot.categories.end());
        for (const auto& category : categories) {
            try {
                for (const auto& directory : model_category_directories(source, category)) {
                    if (!model_path_inside(root, fs::canonical(directory))) {
                        report(display_path(directory), "category directory escapes model directory");
                        continue;
                    }
                    for (fs::recursive_directory_iterator it(directory), end; it != end; ++it) {
                        std::error_code path_error;
                        const fs::path canonical = fs::canonical(it->path(), path_error);
                        if (path_error || !model_path_inside(root, canonical)) {
                            it.disable_recursion_pending();
                            report(display_path(it->path()), path_error ? path_error.message() : "path escapes model directory");
                            continue;
                        }
                        if (!it->is_regular_file() || !supported_model_file(it->path(), category) ||
                            !seen_files.insert(model_path_key(canonical)).second)
                            continue;
                        const auto relative = it->path().lexically_relative(root).generic_u8string();
                        const auto id       = source.primary ? relative : source.id + "/" + relative;
                        files[id]           = {category, it->path().lexically_relative(directory).generic_u8string(), relative, source};
                    }
                }
            } catch (const std::exception& e) {
                report(source.id + "/" + category, e.what());
            }
        }
    }
    std::map<std::pair<std::string, std::string>, size_t> names;
    for (const auto& entry : files)
        ++names[{entry.second.category, entry.second.name}];
    for (auto& entry : files) {
        auto& file = entry.second;
        if (names[{file.category, file.name}] > 1)
            file.name = (file.root.path / fs::u8path(file.relative_path)).generic_u8string();
    }
    models.recipes        = std::move(recipes);
    models.files          = std::move(files);
    models.catalog_errors = std::move(errors);
}

json model_status(ServerRuntime& runtime) {
    std::lock_guard<std::mutex> lock(runtime.models->status_mutex);
    auto& models = *runtime.models;
    json status  = models.status;
    if (status["state"] == "loading" && !status["load_elapsed_seconds"].is_null()) {
        status["load_elapsed_seconds"] = std::chrono::duration<double>(std::chrono::steady_clock::now() - models.load_started_at).count();
    }
    return status;
}

void initialize_server_models(ServerRuntime& runtime) {
    auto& models        = *runtime.models;
    models.base_context = *runtime.ctx_params;
    clear_model_paths(models.base_context);
    models.base_defaults = *runtime.default_gen_params;
    if (runtime.sd_ctx) {
        models.status["state"]  = "ready";
        models.status["origin"] = "command_line";
        json features           = json::array();
        for (const auto& feature : k_recipe_features)
            features.push_back(feature.first);
        models.active_recipe = {
            {"id", "@command-line"},
            {"name", "Command-line model (all features)"},
            {"source", "command_line"},
            {"description", "Uses the loaded command-line model and its generation defaults."},
            {"slots", json::array()},
            {"defaults", json::object()},
            {"features", features},
        };
        models.active_recipe["effective_features"] = resolve_recipe_features(models.active_recipe);
        models.status["runtime_recipe"]            = models.active_recipe;
    }
    refresh_model_catalog(runtime);
}

void prepare_model_load(ServerRuntime& runtime, const json& body, SDContextParams& context, SDGenerationParams& defaults, json* loaded_recipe) {
    check_keys(body, {"recipe", "models"});
    const auto recipe_id = body.at("recipe").get<std::string>();
    auto selection       = body.at("models");
    if (!selection.is_object())
        throw std::runtime_error("models must be an object");
    auto& models = *runtime.models;
    std::lock_guard<std::mutex> lock(models.catalog_mutex);
    const auto found = models.recipes.find(recipe_id);
    if (found == models.recipes.end())
        throw std::runtime_error("unknown recipe");
    const auto& recipe = found->second;
    if (loaded_recipe)
        *loaded_recipe = recipe;
    context  = models.base_context;
    defaults = models.base_defaults;
    apply_recipe_context(recipe.value("context", json::object()), context);
    apply_recipe_defaults(recipe.value("defaults", json::object()), defaults);
    std::set<std::string> allowed;
    for (const auto& slot : recipe.at("slots")) {
        const auto id = slot.at("id").get<std::string>();
        allowed.insert(id);
        if (id == "tae" || id == "taesd") {
            const std::string alias = id == "tae" ? "taesd" : "tae";
            allowed.insert(alias);
            if (selection.contains(alias)) {
                if (selection.contains(id))
                    throw std::runtime_error("specify only one of tae and taesd");
                selection[id] = selection.at(alias);
            }
        }
        const std::string file_id = selection.value(id, std::string());
        if (file_id.empty()) {
            if (slot.at("required").get<bool>())
                throw std::runtime_error("missing required model: " + id);
            continue;
        }
        const auto file = models.files.find(file_id);
        if (file == models.files.end() || !matches_model_category(*find_model_slot(id), file->second.category)) {
            throw std::runtime_error("unknown model or incorrect category: " + id);
        }
        const auto& root    = file->second.root.path;
        const fs::path path = fs::canonical(root / fs::u8path(file->second.relative_path));
        if (!model_path_inside(root, path) || !fs::is_regular_file(path)) {
            throw std::runtime_error("model file must remain inside the model directory: " + id);
        }
        context.*find_model_slot(id)->field = path.u8string();
    }
    for (auto it = selection.begin(); it != selection.end(); ++it) {
        if (!allowed.count(it.key()))
            throw std::runtime_error("unknown model slot: " + it.key());
    }
    for (const auto& group : recipe.value("required_any", json::array())) {
        if (std::none_of(group.begin(), group.end(), [&](const json& value) {
                return !selection.value(value.get<std::string>(), std::string()).empty();
            })) {
            throw std::runtime_error("select at least one model from: " + group.dump());
        }
    }
    if (!context.resolve_and_validate(IMG_GEN))
        throw std::runtime_error("invalid model context parameters");
}
