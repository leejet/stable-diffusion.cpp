#include "context.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <iterator>
#include <limits>

#include "core/ggml_graph_cut.h"
#include "core/probe/backend_probe.h"
#include "core/util.h"
#include "dgl/model_value.h"
#include "dgl/models/component_sources.h"
#include "dgl/models/lora.h"
#include "dgl/run_events.h"

namespace sd::dgl {

    Context::Context(bool keep_resources)
        : keep_resources(keep_resources),
          model_execution_{std::make_shared<sd::ParallelExecutor>(sd_get_num_physical_cores())},
          conditioning_cache_size_(4) {
        model_execution_.backend_probes       = std::make_shared<BackendProbeCache>();
        model_execution_.family_probes        = std::make_shared<models::FamilyProbeCache>();
        model_execution_.defer_initialization = true;
        models->set_retain_weights(true);
        models->set_idle_resource_eviction([this] { return evict_idle_resource(); });
    }

    void Context::prune_expired_resources() {
        for (auto it = resources.begin(); it != resources.end();) {
            if (it->second.value.expired())
                it = resources.erase(it);
            else
                ++it;
        }
    }

    bool Context::evict_idle_resource() {
        auto oldest = resources.end();
        for (auto it = resources.begin(); it != resources.end(); ++it) {
            if (it->second.value.idle() &&
                (oldest == resources.end() || it->second.last_use < oldest->second.last_use))
                oldest = it;
        }
        if (oldest == resources.end())
            return false;
        LOG_VERBOSE("DGL unloaded idle %s", type_name(oldest->second.value.type()).c_str());
        resources.erase(oldest);
        return true;
    }

    void Context::unload_models() {
        const auto cached = resources.size();
        for (auto it = resources.begin(); it != resources.end();) {
            if (is_model_kind(it->second.value.type().kind))
                it = resources.erase(it);
            else
                ++it;
        }
        if (const auto unloaded = cached - resources.size(); unloaded != 0)
            LOG_VERBOSE("DGL unloaded %zu cached model components", unloaded);
    }

    static size_t parse_max_ram(const std::string& value) {
        constexpr long double gib_bytes = 1024.0L * 1024 * 1024;
        size_t end                      = 0;
        long double gib                 = 0;
        try {
            gib = std::stold(value, &end);
        } catch (const std::exception&) {
            throw std::runtime_error("max_ram expects a nonnegative number of GiB");
        }
        const long double bytes = gib * gib_bytes;
        if (value.find_first_not_of(" \t\r\n", end) != std::string::npos ||
            !std::isfinite(gib) || gib < 0 || (gib > 0 && bytes < 1) ||
            bytes >= std::ldexp(1.0L, std::numeric_limits<size_t>::digits))
            throw std::runtime_error("max_ram expects a nonnegative number of GiB");
        return static_cast<size_t>(bytes);
    }

    static int parse_context_int(const std::string& key, const std::string& value, int minimum, int maximum) {
        const auto text   = trim(value);
        const auto* end   = text.data() + text.size();
        int number        = 0;
        const auto parsed = std::from_chars(text.data(), end, number);
        if (parsed.ec != std::errc{} || parsed.ptr != end || number < minimum || number > maximum)
            throw std::runtime_error(key + " expects an integer from " + std::to_string(minimum) + " to " + std::to_string(maximum));
        return number;
    }

    void Context::set_model_execution_option(const std::string& key, const std::string& value) {
        if (key == "n_threads") {
            const auto count = parse_context_int(key, value, 1, 4096);
            if (count == model_execution_.tensor_executor->num_threads())
                return;
            model_execution_.tensor_executor = std::make_shared<sd::ParallelExecutor>(count);
        } else {
            static constexpr std::pair<const char*, bool models::ModelExecutionOptions::*> flags[] = {
                {"enable_mmap", &models::ModelExecutionOptions::enable_mmap},
                {"eager_load", &models::ModelExecutionOptions::eager_load},
                {"disable_prefetch", &models::ModelExecutionOptions::disable_prefetch},
                {"disable_segmented_compute", &models::ModelExecutionOptions::disable_segmented_compute},
            };
            auto flag = std::find_if(std::begin(flags), std::end(flags), [&](const auto& item) { return key == item.first; });
            if (flag == std::end(flags))
                throw std::runtime_error("unknown DGL context option '" + key + "'");
            bool enabled;
            if (!parse_strict_bool(value, enabled))
                throw std::runtime_error(key + " expects a boolean");
            auto& current = model_execution_.*(flag->second);
            if (current == enabled)
                return;
            current = enabled;
        }
        // Runner sessions and weight storage take their settings from the model at initialization.
        unload_models();
    }

    void Context::set(const std::string& key, const std::string& value) {
        if (key == "embed_metadata") {
            bool enabled;
            if (!parse_strict_bool(value, enabled))
                throw std::runtime_error("embed_metadata expects a boolean");
            embed_metadata = enabled;
        } else if (key == "auto_fit") {
            if (!parse_strict_bool(value, auto_fit_))
                throw std::runtime_error("auto_fit expects a boolean");
        } else if (key == "batched_cfg") {
            bool enabled;
            if (!parse_strict_bool(value, enabled))
                throw std::runtime_error("batched_cfg expects a boolean");
            batched_cfg = enabled;
        } else if (key == "max_vram") {
            ggml_graph_cut::MaxVramAssignment limit;
            std::string error;
            if (!limit.parse(value, &error) || !limit.canonicalize_backend_keys(&error))
                throw std::runtime_error(error);
            models->set_max_vram(std::move(limit));
        } else if (key == "max_ram") {
            if (!models->set_max_ram(parse_max_ram(value)))
                throw std::runtime_error(models->host_memory_error());
        } else if (key == "conditioning_cache_size") {
            conditioning_cache_size_ = static_cast<size_t>(parse_context_int(key, value, 0, 4096));
            for (const auto& [_, resource] : resources) {
                if (auto cached = resource.value.lock())
                    set_conditioning_cache_size(*cached, conditioning_cache_size_);
            }
        } else {
            set_model_execution_option(key, value);
        }
    }

    std::string file_key(const std::string& path, nlohmann::json& revision) {
        if (path.empty())
            return path;
        require_file_access(path, SD_DGL_FILE_READ_MODEL);
        std::error_code error;
        auto file = std::filesystem::weakly_canonical(std::filesystem::u8path(path), error);
        if (error)
            return path;
        const auto canonical = file.u8string();
        const auto timestamp = std::filesystem::last_write_time(file, error);
        if (!error) {
            const auto bytes    = std::filesystem::file_size(file, error);
            revision[canonical] = {timestamp.time_since_epoch().count(), error ? 0 : bytes};
        }
        return canonical;
    }

    std::optional<Value> Context::find_resource(const ResourceKey& identity, const std::string& revision, const std::string& lora_revision, uint64_t last_use) {
        auto existing = resources.find(identity);
        if (existing == resources.end())
            return std::nullopt;
        bool changed = false;
        if (existing->second.revision == revision && existing->second.lora_revision == lora_revision &&
            ModelLoader::dependencies_changed(existing->second.dependencies, changed) && !changed) {
            if (auto value = existing->second.value.lock()) {
                existing->second.last_use = last_use;
                return value;
            }
        }
        resources.erase(existing);
        return std::nullopt;
    }

    Value Context::cache_resource(ResourceKey identity, std::string revision, std::string lora_revision, ModelLoader::Dependencies dependencies, Value value, uint64_t last_use) {
        bool changed = false;
        if (!ModelLoader::dependencies_changed(dependencies, changed) || changed)
            throw std::runtime_error("model sources changed while loading");
        set_conditioning_cache_size(value, conditioning_cache_size_);
        resources.insert_or_assign(std::move(identity), Resource{std::move(revision), std::move(lora_revision),
                                                                 Value::Cached(value, keep_resources), std::move(dependencies), last_use});
        return value;
    }

    Value Context::load(const Registry::Entry& registration, Values& args, bool prepare_only) {
        const auto& [name, operation] = registration;
        const auto kind               = operation.output.type.kind;
        nlohmann::json key            = {{"operation", name}};
        auto prepared                 = operation.prepare(args);
        key["params"]                 = std::move(prepared.key);
        auto revision                 = std::move(prepared.revision);
        ResourceKey identity{key.dump(), {}};
        auto argument_revision = revision.dump();
        const auto last_use    = ++resource_epoch;
        if (auto value = find_resource(identity, argument_revision, {}, last_use)) {
            if (!prepare_only)
                finish_resource_loading(*value);
            return std::move(*value);
        }
        const bool model = is_model_kind(kind);
        if (model)
            report_run_progress("loading");
        ResourceLoadRequest request{*this, model ? identity.first : ""};
        auto value        = prepared.load(request);
        auto dependencies = model ? model_from_value(value).dependencies() : ModelLoader::Dependencies{};
        for (const auto& dependency : dependencies)
            revision[dependency.path] = {dependency.exists, dependency.modified.time_since_epoch().count(), dependency.size};
        if (model)
            value.model_source = std::make_shared<ModelSource>(ModelSource{identity.first, revision.dump()});
        if (!prepare_only)
            finish_resource_loading(value);
        return cache_resource(std::move(identity), std::move(argument_revision), {}, std::move(dependencies), std::move(value), last_use);
    }

    void Context::finish_resource_loading(const Value& value) const {
        if (model_execution_.eager_load && is_model_kind(value.type().kind) && !model_from_value(value).finish_initialization())
            throw std::runtime_error("cannot initialize model weights");
    }

    Value Context::derive_model(const Value& base, std::shared_ptr<const models::LoraConfig> loras) {
        if (!base.model_source)
            throw std::runtime_error("model has no weight source");
        const auto& source      = *base.model_source;
        nlohmann::json files    = nlohmann::json::object();
        nlohmann::json bindings = nlohmann::json::array();
        for (const auto& binding : loras->bindings)
            bindings.push_back({file_key(binding.file->path, files), binding.weight});
        auto lora_revision = files.dump();
        ResourceKey identity{source.key, nlohmann::json::array({int(loras->mode), std::move(bindings)}).dump()};

        auto inherit_placement = [&](Value value) {
            const auto memory = model_from_value(base).model_weights->memory();
            const bool disk   = memory.mode == ModelManager::ResidencyMode::Disk;
            if (!memory.split && memory.params_backend &&
                !model_from_value(value).place_params(disk ? "disk" : ggml_backend_name(memory.params_backend), memory.params_backend, disk))
                throw std::runtime_error("cannot place LoRA model weights");
            if (model_execution_.eager_load && !model_from_value(value).finish_initialization())
                throw std::runtime_error("cannot initialize LoRA model weights");
            return bind_model_variant(base, std::move(value));
        };
        const auto last_use = ++resource_epoch;
        if (auto value = find_resource(identity, source.revision, lora_revision, last_use))
            return inherit_placement(std::move(*value));
        auto value         = build_model_variant(*this, base, loras);
        value.model_source = base.model_source;
        value.loras        = std::move(loras);
        auto dependencies  = model_from_value(value).dependencies();
        return inherit_placement(cache_resource(std::move(identity), source.revision, std::move(lora_revision), std::move(dependencies), std::move(value), last_use));
    }

}  // namespace sd::dgl
