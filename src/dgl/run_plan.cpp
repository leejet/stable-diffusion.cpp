#include "run_plan.h"

#include <algorithm>
#include <set>

#include "context.h"
#include "core/backend_fit.h"
#include "core/util.h"
#include "executor.h"
#include "model_value.h"

namespace sd::dgl {
    static constexpr size_t MiB = 1024ULL * 1024;

    static size_t add_size(size_t a, size_t b) {
        return b > SIZE_MAX - a ? SIZE_MAX : a + b;
    }

    RunPlan::RunPlan(const Graph& graph, RunState& state)
        : graph_(graph), state_(state), enabled_(state.context.auto_fit()), vae_preview_(state.events.preview_callback() && (state.events.callbacks().mode == PREVIEW_VAE || state.events.callbacks().mode == PREVIEW_TAE) && (state.events.callbacks().noisy || state.events.callbacks().denoised)) {
        if (graph.resources.empty())
            return;
        if (std::any_of(graph.resources.begin(), graph.resources.end(), [](const auto& call) {
                const auto backend = call.arguments.find("backend");
                return is_model_kind(call.operation->second.output.type.kind) && backend != call.arguments.end() &&
                       trim(backend->second.scalar.template get<std::string>()).empty();
            })) {
            SDBackendManager backends(state.context.backends);
            default_backend_ = enabled_ ? sd::backend_fit::select_default_backend(state.context.models->max_vram(), [&](auto device, auto free, auto total) {
                return state.context.models->available_vram_for_fit(backends.existing_backend(device), free, total);
            })
                                        : sd_backend_resolve_name("");
        }
        if (!enabled_)
            return;
        for (const auto& call : graph.resources) {
            state.events.check_cancelled();
            if (call.deferred)
                continue;
            try {
                auto args  = call.arguments;
                auto value = load_resource(*call.operation, args);
                add_model(call, value);
                prepared_.emplace(call.expression, std::move(value));
            } catch (const std::bad_alloc&) {
                throw;
            } catch (const Error& error) {
                if (error.status == SD_STATUS_CANCELLED)
                    throw;
                throw Error(error.status, "line " + std::to_string(call.expression->line) + ", column " +
                                              std::to_string(call.expression->column) + ": " + error.what());
            } catch (const std::exception& error) {
                throw Error(SD_STATUS_LOAD_FAILED, "line " + std::to_string(call.expression->line) + ", column " +
                                                       std::to_string(call.expression->column) + ": " + error.what());
            }
        }
        for (const auto& entry : models_)
            for (auto backend : entry.memory.compute_backends) {
                if (budgets_.count(backend) || sd_backend_is_cpu(backend))
                    continue;
                size_t free = 0, total = 0;
                ggml_backend_dev_memory(ggml_backend_get_device(backend), &free, &total);
                auto available = state.context.models->available_vram_for_fit(backend, free, total);
                if (!state.context.models->max_vram_bytes(backend))
                    available -= std::min(available, 512 * MiB);
                budgets_[backend] = available;
            }
        const auto ram = state.context.models->available_ram_for_fit();
        ram_budget_    = ram.weights;
        host_budget_   = ram.total;
        place_models();
        if (!models_.empty()) {
            LOG_VERBOSE("auto-fit plan: %zu models, %zu phases", models_.size(), graph.phases.size());
            for (const auto& [backend, bytes] : budgets_)
                LOG_VERBOSE("  %s: available weight/compute budget %.2f MiB", ggml_backend_name(backend), bytes / double(MiB));
            for (const auto& entry : models_)
                LOG_VERBOSE("  %s: %.2f MiB params -> %s", entry.name.c_str(), entry.memory.params_bytes / double(MiB),
                            entry.memory.mode == ModelManager::ResidencyMode::Disk ? "disk" : entry.memory.params_backend ? ggml_backend_name(entry.memory.params_backend)
                                                                                                                          : "(none)");
            for (const auto& phase : graph.phases) {
                std::string names;
                std::set<size_t> seen;
                for (const auto& use : phase.resources) {
                    const auto found = origins_.find(use.resource);
                    if (active(use) && found != origins_.end() && seen.insert(found->second).second) {
                        if (!names.empty())
                            names += ", ";
                        names += models_[found->second].name;
                    }
                }
                LOG_VERBOSE("  phase %s: %s", phase.name.c_str(), names.empty() ? "(deferred)" : names.c_str());
            }
        }
        update_priorities();
    }

    RunPlan::~RunPlan() {
        state_.context.models->set_weight_use_priorities({});
    }

    Value RunPlan::load_resource(const Registry::Entry& operation, Values& args) {
        auto backend = args.find("backend");
        if (!default_backend_.empty() && is_model_kind(operation.second.output.type.kind) && backend != args.end() &&
            trim(backend->second.scalar.get<std::string>()).empty())
            backend->second = Value::literal(default_backend_);
        return state_.context.load(operation, args, true);
    }

    size_t RunPlan::add_model(const Graph::ResourceCall& call, const Value& value) {
        if (!is_model_kind(value.type().kind) || !value.model_source)
            return SIZE_MAX;
        const auto& key = value.model_source->key;
        auto found      = identities_.find(key);
        size_t index;
        if (found == identities_.end()) {
            auto& model          = model_from_value(value);
            auto memory          = model.model_weights->memory();
            const auto forced    = call.arguments.find("params_backend");
            const bool automatic = !memory.split && (forced == call.arguments.end() || trim(forced->second.scalar.get<std::string>()).empty());
            const auto kind      = value.type().kind;
            const auto reserve   = kind == Kind::Denoiser || kind == Kind::TextEncoder ? 2048 * MiB : 1024 * MiB;
            index                = models_.size();
            models_.push_back({key, call.expression->name, kind, std::move(memory), reserve, automatic});
            identities_[key] = index;
        } else
            index = found->second;
        origins_[call.expression] = index;
        return index;
    }

    bool RunPlan::active(const Graph::ResourceUse& use) const {
        return !use.preview || vae_preview_;
    }

    void RunPlan::place_models() {
        std::vector<std::pair<size_t, Value>> order;
        for (size_t i = 0; i < models_.size(); ++i) {
            auto& entry  = models_[i];
            entry.placed = true;
            auto cached  = state_.context.resources.find({entry.key, {}});
            if (cached == state_.context.resources.end())
                continue;
            if (auto value = cached->second.value.lock()) {
                entry.placed = !entry.automatic;
                order.emplace_back(i, std::move(*value));
            }
        }
        std::stable_sort(order.begin(), order.end(), [&](const auto& a, const auto& b) {
            const auto& left  = models_[a.first];
            const auto& right = models_[b.first];
            if ((left.kind == Kind::Denoiser) != (right.kind == Kind::Denoiser))
                return left.kind == Kind::Denoiser;
            return left.memory.params_bytes > right.memory.params_bytes;
        });
        for (const auto& [index, value] : order) {
            place(index, model_from_value(value));
            models_[index].placed = true;
        }
    }

    size_t RunPlan::peak_bytes(ggml_backend_t backend, size_t candidate, bool workspace) const {
        size_t peak = 0;
        for (size_t phase_index = current_; phase_index < graph_.phases.size(); ++phase_index) {
            const auto& phase = graph_.phases[phase_index];
            std::set<size_t> active_models;
            for (const auto& use : phase.resources) {
                auto found = origins_.find(use.resource);
                if (active(use) && found != origins_.end())
                    active_models.insert(found->second);
            }
            if (!active_models.count(candidate))
                continue;
            size_t bytes = 0;
            for (auto index : active_models) {
                const auto& entry  = models_[index];
                const auto& memory = entry.memory;
                const bool compute = std::find(memory.compute_backends.begin(), memory.compute_backends.end(), backend) != memory.compute_backends.end();
                // Unplaced automatic models reserve staging space before competing for parameter storage.
                const bool placed = index == candidate || entry.placed || !entry.automatic;
                if (placed && memory.mode != ModelManager::ResidencyMode::Disk && memory.params_backend == backend)
                    bytes = add_size(bytes, memory.params_bytes);
                else if (compute)
                    bytes = add_size(bytes, memory.largest_tensor);
                if (compute && workspace)
                    bytes = add_size(bytes, entry.reserve);
            }
            peak = std::max(peak, bytes);
        }
        return peak;
    }

    void RunPlan::place(size_t index, models::Model& model) {
        auto& entry  = models_[index];
        auto& memory = entry.memory;
        if (!entry.automatic || memory.compute_backends.empty())
            return;
        auto target           = memory.compute_backends.front();
        memory.params_backend = target;
        memory.mode           = ModelManager::ResidencyMode::ParamBackend;
        bool fits             = true;
        for (auto backend : memory.compute_backends)
            fits = fits && (sd_backend_is_cpu(backend) ? peak_bytes(backend, index, false) <= ram_budget_ && peak_bytes(backend, index) <= host_budget_ : peak_bytes(backend, index) <= budgets_[backend]);
        bool disk = false;
        if (!fits) {
            target = state_.context.backends->count("CPU") ? state_.context.backends->at("CPU").get() : nullptr;
            if (!target) {
                SDBackendManager backends(state_.context.backends);
                std::string error;
                if (!backends.init("cpu", "cpu", nullptr, &error))
                    throw std::runtime_error(error);
                target = backends.runtime_backend(SDBackendModule::DIFFUSION);
            }
            memory.params_backend = target;
            disk                  = peak_bytes(target, index, false) > ram_budget_ || peak_bytes(target, index) > host_budget_;
            if (disk) {
                target                = memory.compute_backends.front();
                memory.params_backend = target;
                memory.mode           = ModelManager::ResidencyMode::Disk;
            }
        }
        const auto name = disk ? std::string("disk") : std::string(ggml_backend_name(target));
        if (!model.place_params(name, target, disk)) {
            memory = model.model_weights->memory();
            LOG_DEBUG("preserved active or split placement for %s", entry.name.c_str());
        }
    }

    void RunPlan::update_priorities() {
        std::map<std::string, size_t> priorities;
        for (const auto& model : models_)
            priorities[model.key] = SIZE_MAX;
        for (size_t phase = current_; phase < graph_.phases.size(); ++phase)
            for (const auto& use : graph_.phases[phase].resources) {
                auto found = origins_.find(use.resource);
                if (active(use) && found != origins_.end()) {
                    auto& next = priorities[models_[found->second].key];
                    next       = std::min(next, phase);
                }
            }
        state_.context.models->set_weight_use_priorities(std::move(priorities));
    }

    void RunPlan::enter(const Expr* statement, const std::vector<const Expr*>& call_path, bool after) {
        if (!enabled_)
            return;
        for (size_t i = current_; i < graph_.phases.size(); ++i)
            if (graph_.phases[i].statement == statement && graph_.phases[i].call_path == call_path && graph_.phases[i].after == after) {
                current_ = i;
                update_priorities();
                break;
            }
    }

    void RunPlan::resource_loaded(const Expr* expression, const Value& value) {
        if (!enabled_)
            return;
        prepared_.erase(expression);
        for (const auto& call : graph_.resources) {
            if (call.expression != expression)
                continue;
            const bool known = origins_.count(expression) != 0;
            const auto index = add_model(call, value);
            if (index == SIZE_MAX)
                break;
            auto& model  = model_from_value(value);
            auto current = model.model_weights->memory();
            auto& entry  = models_[index];
            if (known && current.params_bytes == entry.memory.params_bytes && current.compute_backends == entry.memory.compute_backends) {
                const bool disk = entry.memory.mode == ModelManager::ResidencyMode::Disk;
                if (entry.automatic && entry.memory.params_backend && !model.place_params(disk ? "disk" : ggml_backend_name(entry.memory.params_backend), entry.memory.params_backend, disk))
                    entry.memory = std::move(current);
                break;
            }
            entry.memory = std::move(current);
            for (auto backend : entry.memory.compute_backends)
                if (!sd_backend_is_cpu(backend) && !budgets_.count(backend)) {
                    size_t free = 0, total = 0;
                    ggml_backend_dev_memory(ggml_backend_get_device(backend), &free, &total);
                    auto available = state_.context.models->available_vram_for_fit(backend, free, total);
                    if (!state_.context.models->max_vram_bytes(backend))
                        available -= std::min(available, 512 * MiB);
                    budgets_[backend] = available;
                }
            const auto ram = state_.context.models->available_ram_for_fit();
            ram_budget_    = ram.weights;
            host_budget_   = ram.total;
            place_models();
            update_priorities();
            LOG_DEBUG("updated the remaining plan for %s", entry.name.c_str());
            break;
        }
    }
}
