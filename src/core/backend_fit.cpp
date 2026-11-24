#include "backend_fit.h"
#include "config.h"
#include "core/layer_split_partition.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#endif

#include "core/ggml_extend_backend.h"
#include "core/util.h"
#include "ggml-backend.h"
#include "name_conversion.h"

namespace sd::backend_fit {

    static constexpr int64_t MiB = 1024ll * 1024;

    enum class ComponentKind {
        DIT,
        CONDITIONER,
        VAE,
    };

    struct Component {
        ComponentKind kind;
        const char* name;
        int64_t params_bytes  = 0;
        int64_t reserve_bytes = 0;
        int64_t staging_bytes = 0;
    };

    struct Device {
        std::string name;
        std::string description;
        std::string registry;
        std::string device_id;
        std::string shadowed_by;
        bool discrete        = false;
        int64_t free_bytes   = 0;
        int64_t budget_bytes = 0;
    };

    enum class ParamsLocation {
        MAIN_GPU,
        CPU,
        OTHER_GPU,
        DISK,
    };

    struct Decision {
        ParamsLocation params_location = ParamsLocation::DISK;
        size_t params_device           = SIZE_MAX;
        std::vector<int64_t> shares;
    };

    struct Runtime {
        std::string name;
        std::vector<size_t> devices;
        bool estimate_shares = false;
    };

    struct Plan {
        bool valid         = false;
        size_t main_device = SIZE_MAX;
        std::vector<Runtime> runtimes;
        std::vector<Decision> decisions;
    };

    static bool classify_tensor(const std::string& name, ComponentKind& out) {
        auto contains = [&](const char* s) { return name.find(s) != std::string::npos; };

        if (is_cond_stage_model_name(name)) {
            out = ComponentKind::CONDITIONER;
            return true;
        }
        if (contains("model.diffusion_model.") || contains("unet.") || name.rfind("diffusion.", 0) == 0) {
            out = ComponentKind::DIT;
            return true;
        }
        if (contains("first_stage_model.") ||
            name.rfind("vae.", 0) == 0 ||
            name.rfind("tae.", 0) == 0) {
            out = ComponentKind::VAE;
            return true;
        }
        if (contains("text_encoders") ||
            contains("cond_stage_model") ||
            contains("te.text_model.") ||
            contains("conditioner") ||
            name.rfind("text_encoder.", 0) == 0 ||
            contains(".aggregate_embed.")) {
            out = ComponentKind::CONDITIONER;
            return true;
        }
        return false;
    }

    static std::vector<Component> estimate_components(ModelLoader& loader, ggml_type override_wtype, const char* standalone_name = nullptr) {
        int64_t bytes[3]          = {0, 0, 0};
        int64_t largest_tensor[3] = {0, 0, 0};
        for (const auto& [name, stored_tensor] : loader.get_tensor_storage_map()) {
            TensorStorage ts   = stored_tensor;
            ComponentKind kind = ComponentKind::DIT;
            if (!standalone_name && !classify_tensor(ts.name, kind)) {
                continue;
            }
            if (ts.expected_type != GGML_TYPE_COUNT) {
                ts.type = ts.expected_type;
            } else if (override_wtype != GGML_TYPE_COUNT && loader.tensor_should_be_converted(ts, override_wtype)) {
                ts.type = override_wtype;
            }
            const int64_t tensor_bytes = (int64_t)ts.nbytes() + 64;
            bytes[int(kind)] += tensor_bytes;
            largest_tensor[int(kind)] = std::max(largest_tensor[int(kind)], tensor_bytes);
        }

        return {
            {ComponentKind::DIT, standalone_name ? standalone_name : "DiT", bytes[int(ComponentKind::DIT)], 2048 * MiB, largest_tensor[int(ComponentKind::DIT)]},
            {ComponentKind::CONDITIONER, "Conditioner", bytes[int(ComponentKind::CONDITIONER)], 2048 * MiB, largest_tensor[int(ComponentKind::CONDITIONER)]},
            {ComponentKind::VAE, "VAE", bytes[int(ComponentKind::VAE)], 1024 * MiB, largest_tensor[int(ComponentKind::VAE)]},
        };
    }

    static std::string budget_key(std::string name) {
        std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return (char)std::tolower(c); });
        return name;
    }

    // A GPU with memory of its own, listed once: not an integrated GPU, whose memory is the system RAM, and not the
    // second listing of a card that a preferred backend already lists.
    static bool has_own_memory(const Device& device) {
        return device.discrete && device.shadowed_by.empty();
    }

    // Backends built on a generic graphics API rank after the backend the GPU vendor provides.
    static int registry_rank(const std::string& registry) {
        return registry == "Vulkan" || registry == "OpenCL" ? 1 : 0;
    }

    // A GPU is listed once per backend that supports it: an NVIDIA card is both CUDA0 and Vulkan0. Counting it twice
    // makes a plan promise memory that exists once, so the listings that share a PCI bus id are reduced to the one
    // of the best-ranked backend, and the others are marked as shadowed by it. They stay in the list, because the
    // user can still name them. Devices whose backend reports no PCI bus id cannot be matched and are left alone.
    static void mark_shadowed_devices(std::vector<Device>& devices) {
        auto preferred_over = [&](size_t a, size_t b) {
            const int rank_a = registry_rank(devices[a].registry);
            const int rank_b = registry_rank(devices[b].registry);
            return rank_a < rank_b || (rank_a == rank_b && a < b);
        };
        for (size_t di = 0; di < devices.size(); ++di) {
            if (devices[di].device_id.empty()) {
                continue;
            }
            size_t owner = di;
            for (size_t other = 0; other < devices.size(); ++other) {
                if (devices[other].device_id == devices[di].device_id && preferred_over(other, owner)) {
                    owner = other;
                }
            }
            if (owner != di) {
                devices[di].shadowed_by = devices[owner].name;
            }
        }
    }

    static std::vector<Device> enumerate_gpu_devices(const sd::ggml_graph_cut::MaxVramAssignment& budgets,
                                                     bool include_other_devices,
                                                     const AvailableVRAM& available_vram) {
        std::vector<Device> out;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            const auto type        = ggml_backend_dev_type(dev);
            if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU &&
                (!include_other_devices || type == GGML_BACKEND_DEVICE_TYPE_CPU)) {
                continue;
            }
            Device device;
            device.name        = ggml_backend_dev_name(dev);
            device.description = ggml_backend_dev_description(dev);
            device.discrete    = type == GGML_BACKEND_DEVICE_TYPE_GPU;
            if (ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev)) {
                device.registry = ggml_backend_reg_name(reg);
            }
            ggml_backend_dev_props props{};
            ggml_backend_dev_get_props(dev, &props);
            if (props.device_id != nullptr) {
                device.device_id = props.device_id;
            }
            size_t free_bytes = 0, total_bytes = 0;
            ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
            if (available_vram)
                free_bytes = available_vram(dev, free_bytes, total_bytes);
            device.free_bytes = (int64_t)free_bytes;

            float gib = budgets.default_gib;
            auto it   = budgets.backend_gib.find(budget_key(device.name));
            if (it != budgets.backend_gib.end()) {
                gib = it->second;
            }
            if (gib > 0.f) {
                device.budget_bytes = (int64_t)std::min(gib * 1024.0 * MiB, (double)device.free_bytes);
            } else if (gib < 0.f) {
                device.budget_bytes = (int64_t)std::max<double>(device.free_bytes + gib * 1024.0 * MiB, 0);
            } else {
                device.budget_bytes = std::max<int64_t>(device.free_bytes - 512 * MiB, 0);
            }
            out.push_back(std::move(device));
        }
        mark_shadowed_devices(out);
        return out;
    }

    int64_t available_ram_bytes() {
#if defined(_WIN32)
        MEMORYSTATUSEX status{};
        status.dwLength = sizeof(status);
        if (GlobalMemoryStatusEx(&status)) {
            // Commit exhaustion can reject allocations while physical RAM is still available.
            return (int64_t)std::min(status.ullAvailPhys, status.ullAvailPageFile);
        }
#elif defined(__linux__)
        std::ifstream meminfo("/proc/meminfo");
        std::string key, unit;
        int64_t kib = 0;
        while (meminfo >> key >> kib >> unit) {
            if (key == "MemAvailable:" && unit == "kB" && kib >= 0) {
                return kib * 1024;
            }
        }
#elif defined(__APPLE__)
        const mach_port_t host = mach_host_self();
        vm_size_t page_size    = 0;
        vm_statistics64_data_t stats{};
        mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
        const bool ok                = host_page_size(host, &page_size) == KERN_SUCCESS &&
                                       host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&stats, &count) == KERN_SUCCESS;
        mach_port_deallocate(mach_task_self(), host);
        if (ok) {
            return ((int64_t)stats.free_count + stats.inactive_count) * page_size;
        }
#endif
        return -1;
    }

    int64_t available_commit_bytes() {
#ifdef _WIN32
        MEMORYSTATUSEX status{};
        status.dwLength = sizeof(status);
        if (GlobalMemoryStatusEx(&status))
            return static_cast<int64_t>(status.ullAvailPageFile);
#endif
        return -1;
    }

    // The device with the largest budget, preferring a graphics card. An integrated GPU reports the shared system
    // RAM as its memory, which is more than a card has but computes several times slower, so it is chosen only when
    // no card has a budget. The second listing of a card that another backend lists is never chosen.
    static size_t select_main_device(const std::vector<Device>& devices) {
        size_t main_device = SIZE_MAX;
        for (size_t di = 0; di < devices.size(); ++di) {
            if (devices[di].budget_bytes <= 0 || !devices[di].shadowed_by.empty()) {
                continue;
            }
            if (main_device == SIZE_MAX || devices[di].discrete > devices[main_device].discrete ||
                (devices[di].discrete == devices[main_device].discrete &&
                 devices[di].budget_bytes > devices[main_device].budget_bytes)) {
                main_device = di;
            }
        }
        return main_device;
    }

    static constexpr int64_t SPLIT_BOUNDARY_MARGIN = 256 * MiB;
    static constexpr int64_t SPLIT_SLACK_PERCENT   = 10;

    // The shares the layer split will give the devices, filled the way --layer-split-fill makes it fill them. The
    // slack and the margin per boundary cover blocks that cannot be cut where a limit ends; an even cut into many
    // pieces stands in for the blocks of the graph, which are not known yet.
    static bool estimate_split_shares(int64_t params_bytes,
                                      const std::vector<int64_t>& capacities,
                                      sd::LayerSplitFill fill,
                                      std::vector<int64_t>& shares) {
        int64_t total = 0;
        for (int64_t capacity : capacities) {
            total += capacity;
        }
        const int64_t boundaries = std::max<int64_t>((int64_t)capacities.size() - 1, 0);
        const int64_t required   = params_bytes + params_bytes * SPLIT_SLACK_PERCENT / 100 + boundaries * SPLIT_BOUNDARY_MARGIN;
        shares.assign(capacities.size(), 0);
        if (total < required) {
            return false;
        }
        constexpr int64_t pieces = 256;
        std::vector<int64_t> segments(pieces, params_bytes / pieces);
        segments.back() += params_bytes % pieces;
        return sd::layer_split_place(segments, sd::layer_split_fill_limits(fill, segments, capacities), shares);
    }

    static Plan compute_plan(const std::vector<Component>& components,
                             const std::vector<Device>& devices,
                             int64_t ram_budget_bytes,
                             const std::vector<Runtime>& runtimes = {}) {
        Plan plan;
        plan.main_device = select_main_device(devices);
        plan.runtimes    = runtimes;
        if (plan.runtimes.empty()) {
            if (plan.main_device == SIZE_MAX) {
                return plan;
            }
            plan.runtimes.resize(components.size(), {devices[plan.main_device].name, {plan.main_device}});
        }

        std::vector<size_t> order(components.size());
        for (size_t ci = 0; ci < components.size(); ++ci) {
            order[ci] = ci;
        }
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return components[a].kind < components[b].kind;
        });

        std::vector<int64_t> remaining;
        for (const Device& device : devices) {
            remaining.push_back(std::max<int64_t>(device.budget_bytes, 0));
        }
        ram_budget_bytes = std::max<int64_t>(ram_budget_bytes, 0);
        plan.decisions.resize(components.size());

        auto uses_device = [&](size_t ci, size_t di) {
            const auto& runtime_devices = plan.runtimes[ci].devices;
            return std::find(runtime_devices.begin(), runtime_devices.end(), di) != runtime_devices.end();
        };
        auto uses_gpu = [&](size_t ci, size_t di) {
            const auto& runtime_devices = plan.runtimes[ci].devices;
            return std::any_of(runtime_devices.begin(), runtime_devices.end(), [&](size_t used) {
                return used == di || (!devices[di].device_id.empty() && devices[used].device_id == devices[di].device_id);
            });
        };
        auto headroom_for = [&](size_t ci, size_t di) {
            // Higher-priority offloaded weights need cache space on their compute devices.
            int64_t headroom = 0;
            for (size_t other = 0; other < components.size(); ++other) {
                if (components[other].params_bytes == 0 || !uses_device(other, di)) {
                    continue;
                }
                const bool resident          = other == ci || plan.decisions[other].params_location == ParamsLocation::MAIN_GPU;
                const int64_t cached_weights = components[other].kind < components[ci].kind
                                                   ? components[other].params_bytes
                                                   : components[other].staging_bytes;
                headroom                     = std::max(headroom, components[other].reserve_bytes +
                                                                      (resident ? 0 : cached_weights));
            }
            return headroom;
        };

        for (size_t ci : order) {
            const Component& comp = components[ci];
            Decision& decision    = plan.decisions[ci];
            if (comp.params_bytes == 0) {
                continue;
            }

            const auto& runtime_devices = plan.runtimes[ci].devices;
            std::vector<int64_t> shares;
            bool fits_runtime = false;
            if (plan.runtimes[ci].estimate_shares) {
                std::vector<int64_t> capacities;
                for (size_t di : runtime_devices) {
                    capacities.push_back(std::max<int64_t>(remaining[di] - headroom_for(ci, di), 0));
                }
                std::vector<enum ggml_backend_dev_type> device_types;
                for (size_t di : runtime_devices) {
                    device_types.push_back(devices[di].discrete ? GGML_BACKEND_DEVICE_TYPE_GPU : GGML_BACKEND_DEVICE_TYPE_IGPU);
                }
                const sd::LayerSplitFill fill = sd::resolve_layer_split_fill(sd::layer_split_fill(), device_types);
                fits_runtime                  = !runtime_devices.empty() && estimate_split_shares(comp.params_bytes, capacities, fill, shares);
            } else {
                fits_runtime = !runtime_devices.empty() &&
                               std::all_of(runtime_devices.begin(), runtime_devices.end(), [&](size_t di) {
                                   const int64_t headroom = headroom_for(ci, di);
                                   return headroom <= remaining[di] && comp.params_bytes <= remaining[di] - headroom;
                               });
            }
            if (fits_runtime) {
                decision.params_location = ParamsLocation::MAIN_GPU;
                decision.params_device   = runtime_devices.front();
                decision.shares          = shares;
                // Exact split allocations are unavailable until the runners build their plans.
                for (size_t i = 0; i < runtime_devices.size(); ++i) {
                    remaining[runtime_devices[i]] -= shares.empty() ? comp.params_bytes : shares[i];
                }
                continue;
            }
            if (comp.params_bytes <= ram_budget_bytes) {
                decision.params_location = ParamsLocation::CPU;
                ram_budget_bytes -= comp.params_bytes;
                continue;
            }
            if (runtime_devices.empty()) {
                continue;
            }

            // Weights wait on another GPU only if the memory there is that GPU's own: not the system RAM an
            // integrated GPU shares, which the RAM budget above has just found too small, and not the card the
            // module computes on, seen through another backend.
            size_t best = SIZE_MAX;
            for (size_t di = 0; di < devices.size(); ++di) {
                if (!has_own_memory(devices[di]) || uses_gpu(ci, di)) {
                    continue;
                }
                const int64_t headroom = headroom_for(ci, di);
                if (headroom <= remaining[di] && comp.params_bytes <= remaining[di] - headroom &&
                    (best == SIZE_MAX || remaining[di] > remaining[best])) {
                    best = di;
                }
            }
            if (best != SIZE_MAX) {
                decision.params_location = ParamsLocation::OTHER_GPU;
                decision.params_device   = best;
                remaining[best] -= comp.params_bytes;
            }
        }
        plan.valid = true;
        return plan;
    }

    static std::string params_backend_name(const Decision& decision, const std::vector<Device>& devices) {
        switch (decision.params_location) {
            case ParamsLocation::MAIN_GPU:
            case ParamsLocation::OTHER_GPU:
                return devices[decision.params_device].name;
            case ParamsLocation::CPU:
                return "cpu";
            case ParamsLocation::DISK:
                return "disk";
        }
        return "disk";
    }

    static std::string format_gib(int64_t bytes) {
        char text[32];
        snprintf(text, sizeof(text), "%.1f GiB", (double)bytes / (1024.0 * MiB));
        return text;
    }

    static std::string device_note(const Device& device) {
        if (!device.shadowed_by.empty()) {
            return " (same GPU as " + device.shadowed_by + ", not used automatically)";
        }
        return device.discrete ? "" : " (shares the system memory)";
    }

    static void print_plan(const Plan& plan,
                           const std::vector<Component>& components,
                           const std::vector<Device>& devices,
                           int64_t free_ram,
                           int64_t ram_budget) {
        LOG_INFO("auto-fit plan:");
        LOG_INFO("  devices:");
        for (const Device& device : devices) {
            LOG_INFO("    %-12s %-32s free %6lld MiB, budget %6lld MiB%s",
                     device.name.c_str(), device.description.c_str(),
                     (long long)(device.free_bytes / MiB), (long long)(device.budget_bytes / MiB),
                     device_note(device).c_str());
        }
        if (free_ram < 0) {
            LOG_WARN("auto-fit: available RAM is unknown; skipping CPU parameter residency");
        } else {
            LOG_INFO("    RAM          free %6lld MiB, params budget %6lld MiB",
                     (long long)(free_ram / MiB), (long long)(ram_budget / MiB));
        }
        LOG_INFO("  compute-device weight cache priority: diffusion > te > vae");
        LOG_INFO("  components (params: compute device -> RAM -> other GPU -> disk):");
        for (size_t ci = 0; ci < components.size(); ++ci) {
            const Component& comp = components[ci];
            if (comp.params_bytes == 0) {
                continue;
            }
            const std::string params = plan.decisions[ci].params_location == ParamsLocation::MAIN_GPU
                                           ? plan.runtimes[ci].name
                                           : params_backend_name(plan.decisions[ci], devices);
            LOG_INFO("    %-12s params %6lld MiB, compute reserve %5lld MiB -> compute %s, params %s",
                     comp.name, (long long)(comp.params_bytes / MiB), (long long)(comp.reserve_bytes / MiB),
                     plan.runtimes[ci].name.c_str(), params.c_str());
            const auto& shares = plan.decisions[ci].shares;
            if (shares.size() > 1) {
                std::string split;
                for (size_t i = 0; i < shares.size(); ++i) {
                    split += (i == 0 ? "" : " + ") + format_gib(shares[i]) + " on " + devices[plan.runtimes[ci].devices[i]].name;
                }
                LOG_INFO("    %-12s split: %s", "", split.c_str());
            }
        }
    }

    // A listing that a preferred backend already provides is left out of the automatic choice, and print_plan marks it
    // on its device line. When the user names it anyway (--backend vulkan0 for a card that CUDA lists as CUDA0), the
    // run goes through the slower of the two backends and nothing else in the log says so.
    static std::vector<std::string> shadowed_runtime_warnings(const Plan& plan, const std::vector<Device>& devices) {
        std::vector<std::string> warnings;
        std::vector<size_t> seen;
        for (const Runtime& runtime : plan.runtimes) {
            for (size_t di : runtime.devices) {
                if (di >= devices.size() || devices[di].shadowed_by.empty() || std::find(seen.begin(), seen.end(), di) != seen.end()) {
                    continue;
                }
                seen.push_back(di);
                const Device& device = devices[di];
                std::string text     = "auto-fit: " + device.name + " is the same GPU as " + device.shadowed_by + "; the automatic choice uses " + device.shadowed_by + " for this card.";
                for (const Device& owner : devices) {
                    if (owner.name == device.shadowed_by && registry_rank(owner.registry) < registry_rank(device.registry)) {
                        text += " Backends on a generic graphics API (Vulkan, OpenCL) are usually slower than the one the GPU vendor provides: name " + owner.name +
                                " in --backend, or leave --backend out, unless " + device.name + " is wanted.";
                        break;
                    }
                }
                warnings.push_back(std::move(text));
            }
        }
        return warnings;
    }

    // Weights that fit neither the compute device nor the RAM budget stay in the model file and are read again every
    // time their module runs, the diffusion model at every sampling step. print_plan lists "params disk" per module;
    // this says what it costs and what helps. Empty when nothing streams, or when the RAM is unknown (print_plan says so).
    static std::string disk_streaming_warning(const Plan& plan, const std::vector<Component>& components, int64_t free_ram, int64_t ram_budget) {
        std::string names;
        int64_t bytes  = 0;
        bool diffusion = false;
        for (size_t ci = 0; ci < components.size() && ci < plan.decisions.size(); ++ci) {
            if (components[ci].params_bytes <= 0 || plan.decisions[ci].params_location != ParamsLocation::DISK) {
                continue;
            }
            names += (names.empty() ? "" : ", ") + std::string(components[ci].name);
            bytes += components[ci].params_bytes;
            diffusion = diffusion || components[ci].kind == ComponentKind::DIT;
        }
        if (bytes == 0 || free_ram < 0) {
            return "";
        }
        char text[640];
        snprintf(text, sizeof(text),
                 "auto-fit: the weights of %s (%lld MiB) stay on disk and are read again every time they run%s: free RAM is %lld MiB and the budget for weights %lld MiB. "
                 "Close other programs to free RAM, or shrink the weights with --type (for example q8_0) or a quantized model file, to keep them in memory.",
                 names.c_str(), (long long)(bytes / MiB), diffusion ? ", the diffusion model at every sampling step" : "",
                 (long long)(free_ram / MiB), (long long)(ram_budget / MiB));
        return text;
    }

    static void append_assignment(std::string& spec, const char* key, const std::string& value) {
        if (!spec.empty()) {
            spec += ",";
        }
        spec += key;
        spec += "=";
        spec += value;
    }

    static const char* module_key(ComponentKind kind) {
        switch (kind) {
            case ComponentKind::DIT:
                return "diffusion";
            case ComponentKind::CONDITIONER:
                return "te";
            case ComponentKind::VAE:
                return "vae";
        }
        return "";
    }

    static std::string lowercase(std::string text) {
        std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return (char)std::tolower(c); });
        return text;
    }

    bool parse_multi_gpu_mode(const std::string& text, MultiGpuMode& mode) {
        const std::string value = lowercase(trim(text));
        if (value.empty() || value == "auto") {
            mode = MultiGpuMode::AUTO;
        } else if (value == "on") {
            mode = MultiGpuMode::ON;
        } else if (value == "off") {
            mode = MultiGpuMode::OFF;
        } else {
            return false;
        }
        return true;
    }

    static void set_assignment(std::string& spec, const char* key, const std::string& value) {
        const std::string prefix = std::string(key) + "=";
        std::string out;
        bool replaced = false;
        for (const std::string& part : split_string(spec, ',')) {
            std::string token = trim(part);
            if (token.empty()) {
                continue;
            }
            if (!replaced && token.compare(0, prefix.size(), prefix) == 0) {
                token    = prefix + value;
                replaced = true;
            }
            out += out.empty() ? "" : ",";
            out += token;
        }
        if (!replaced) {
            out += out.empty() ? "" : ",";
            out += prefix + value;
        }
        spec = out;
    }

    static bool splittable(ComponentKind kind) {
        return kind == ComponentKind::DIT || kind == ComponentKind::CONDITIONER;
    }

    // A module whose --split-mode is row is not split automatically: row split keeps its own weight distribution.
    static std::vector<bool> split_allowed(const std::vector<Component>& components, const std::string& split_mode_spec) {
        std::vector<bool> allowed(components.size(), true);
        SDBackendAssignment assignment;
        std::string error;
        if (split_mode_spec.empty() || !sd_parse_backend_assignment(split_mode_spec, &assignment, &error)) {
            return allowed;
        }
        const SDBackendModule modules[] = {SDBackendModule::DIFFUSION, SDBackendModule::TE, SDBackendModule::VAE};
        for (size_t ci = 0; ci < components.size(); ++ci) {
            if (lowercase(trim(assignment.get(modules[int(components[ci].kind)]))) == "row") {
                allowed[ci] = false;
            }
        }
        return allowed;
    }

    // The discrete GPUs of the backend that has the GPU with the most memory, largest budget first. Integrated
    // GPUs share the system RAM and GPUs of another backend would need their own runtime, so neither is included,
    // and a card is included once even when two backends list it.
    static std::vector<size_t> multi_gpu_group(const std::vector<Device>& devices) {
        auto usable = [&](size_t di) { return has_own_memory(devices[di]) && devices[di].budget_bytes > 0; };
        size_t best = SIZE_MAX;
        for (size_t di = 0; di < devices.size(); ++di) {
            if (usable(di) && (best == SIZE_MAX || devices[di].budget_bytes > devices[best].budget_bytes)) {
                best = di;
            }
        }
        std::vector<size_t> group;
        for (size_t di = 0; best != SIZE_MAX && di < devices.size(); ++di) {
            if (usable(di) && devices[di].registry == devices[best].registry) {
                group.push_back(di);
            }
        }
        std::stable_sort(group.begin(), group.end(), [&](size_t a, size_t b) {
            return devices[a].budget_bytes > devices[b].budget_bytes;
        });
        return group;
    }

    static std::string join_device_names(const std::vector<Device>& devices, const std::vector<size_t>& indices, const char* separator) {
        std::string out;
        for (size_t di : indices) {
            out += out.empty() ? "" : separator;
            out += devices[di].name;
        }
        return out;
    }

    // Places the modules the user did not assign, in priority order, where their weights stay resident: a module
    // keeps the device it has unless it does not fit there next to the modules placed before it, and then it moves
    // to the first GPU of the group that can hold it. The diffusion model and the text encoder are split across GPUs
    // when no single one can hold them. With --multi-gpu on, the text encoder and the VAE also move away from the
    // diffusion GPU when they would fit. Returns true when a runtime changed.
    static bool plan_multi_gpu_runtimes(const std::vector<Component>& components,
                                        const std::vector<Device>& devices,
                                        int64_t ram_budget_bytes,
                                        MultiGpuMode mode,
                                        const std::vector<bool>& explicit_runtime,
                                        const std::vector<bool>& allow_split,
                                        std::vector<Runtime>& runtimes,
                                        std::vector<std::string>& notes) {
        bool has_free_module = false;
        for (size_t ci = 0; ci < components.size(); ++ci) {
            has_free_module = has_free_module || (!explicit_runtime[ci] && components[ci].params_bytes > 0);
        }
        if (mode == MultiGpuMode::OFF || !has_free_module) {
            return false;
        }

        const std::vector<size_t> group = multi_gpu_group(devices);
        if (group.size() < 2) {
            size_t discrete = 0;
            for (const Device& device : devices) {
                discrete += has_own_memory(device) && device.budget_bytes > 0 ? 1 : 0;
            }
            if (discrete >= 2) {
                notes.push_back(
                    "several GPUs were found but they belong to different backends; only the best one is used "
                    "unless modules are assigned with --backend");
            }
            return false;
        }

        std::vector<size_t> order(components.size());
        for (size_t ci = 0; ci < components.size(); ++ci) {
            order[ci] = ci;
        }
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return components[a].kind < components[b].kind;
        });

        auto single   = [&](size_t di) { return Runtime{devices[di].name, {di}}; };
        auto plan_now = [&]() { return compute_plan(components, devices, ram_budget_bytes, runtimes); };
        auto resident = [](const Plan& plan, size_t ci) { return plan.decisions[ci].params_location == ParamsLocation::MAIN_GPU; };

        const std::vector<Runtime> original = runtimes;
        std::vector<std::string> planned_notes;
        for (size_t ci : order) {
            const Component& comp = components[ci];
            if (explicit_runtime[ci] || comp.params_bytes == 0) {
                continue;
            }

            const Runtime previous = runtimes[ci];
            const bool fits_before = resident(plan_now(), ci);

            std::vector<Runtime> candidates = {previous};
            for (size_t di : group) {
                if (devices[di].name != previous.name) {
                    candidates.push_back(single(di));
                }
            }
            if (mode == MultiGpuMode::ON && comp.kind != ComponentKind::DIT) {
                std::vector<size_t> diffusion_devices;
                for (size_t other = 0; other < components.size(); ++other) {
                    if (components[other].kind == ComponentKind::DIT && components[other].params_bytes > 0) {
                        diffusion_devices = runtimes[other].devices;
                    }
                }
                std::stable_partition(candidates.begin(), candidates.end(), [&](const Runtime& candidate) {
                    return std::none_of(candidate.devices.begin(), candidate.devices.end(), [&](size_t di) {
                        return std::find(diffusion_devices.begin(), diffusion_devices.end(), di) != diffusion_devices.end();
                    });
                });
            }
            if (splittable(comp.kind) && allow_split[ci]) {
                for (size_t count = 2; count <= group.size(); ++count) {
                    Runtime split;
                    split.devices         = std::vector<size_t>(group.begin(), group.begin() + count);
                    split.name            = join_device_names(devices, split.devices, "&");
                    split.estimate_shares = true;
                    candidates.push_back(std::move(split));
                }
            }

            Runtime chosen = previous;
            for (const Runtime& candidate : candidates) {
                runtimes[ci] = candidate;
                if (resident(plan_now(), ci)) {
                    chosen = candidate;
                    break;
                }
            }
            runtimes[ci] = chosen;
            if (chosen.name == previous.name) {
                continue;
            }

            const std::string module = module_key(comp.kind);
            if (chosen.devices.size() > 1) {
                planned_notes.push_back(module + " is split across " + chosen.name + ": its weights (" + format_gib(comp.params_bytes) +
                                        ") do not fit on " + previous.name + " next to the modules placed before it");
            } else if (fits_before) {
                planned_notes.push_back(module + " runs on " + chosen.name + " (--multi-gpu on keeps " + previous.name + " for the diffusion model)");
            } else {
                planned_notes.push_back(module + " runs on " + chosen.name + ": it does not fit on " + previous.name + " next to the modules placed before it");
            }
        }

        bool changed = false;
        for (size_t ci = 0; ci < components.size(); ++ci) {
            changed = changed || runtimes[ci].name != original[ci].name;
        }
        const Plan final_plan = plan_now();

        if (!changed && mode == MultiGpuMode::AUTO) {
            bool all_resident        = true;
            std::vector<size_t> idle = group;
            std::vector<std::string> hosts;
            for (size_t ci = 0; ci < components.size(); ++ci) {
                if (components[ci].params_bytes == 0) {
                    continue;
                }
                all_resident = all_resident && resident(final_plan, ci);
                if (std::find(hosts.begin(), hosts.end(), runtimes[ci].name) == hosts.end()) {
                    hosts.push_back(runtimes[ci].name);
                }
                for (size_t di : runtimes[ci].devices) {
                    idle.erase(std::remove(idle.begin(), idle.end(), di), idle.end());
                }
            }
            if (all_resident && !idle.empty()) {
                std::string host_names;
                for (const std::string& host : hosts) {
                    host_names += (host_names.empty() ? "" : ", ") + host;
                }
                planned_notes.push_back(std::to_string(group.size()) + " GPUs found (" + join_device_names(devices, group, ", ") +
                                        "); everything fits on " + host_names + ", so " + join_device_names(devices, idle, ", ") +
                                        " will not be used (--multi-gpu on would put the text encoder and the VAE on the other GPUs)");
            }
        }
        notes.insert(notes.end(), planned_notes.begin(), planned_notes.end());
        return changed;
    }

    // Plans the multi-GPU runtimes. With --layer-split-fill auto it plans once for each way to fill a split and keeps
    // the plan with more modules resident on the GPUs, proportional on a tie; the layer split is then told to fill
    // the devices the way the plan counted on.
    static bool plan_multi_gpu_runtimes_with_fill(const std::vector<Component>& components,
                                                  const std::vector<Device>& devices,
                                                  int64_t ram_budget_bytes,
                                                  MultiGpuMode mode,
                                                  const std::vector<bool>& explicit_runtime,
                                                  const std::vector<bool>& allow_split,
                                                  std::vector<Runtime>& runtimes,
                                                  std::vector<std::string>& notes) {
        const sd::LayerSplitFill requested_fill = sd::layer_split_fill();
        std::vector<sd::LayerSplitFill> fills   = {requested_fill};
        if (requested_fill == sd::LayerSplitFill::AUTO) {
            fills = {sd::LayerSplitFill::PROPORTIONAL, sd::LayerSplitFill::FILL_FIRST};
        }
        std::vector<Runtime> best_runtimes;
        std::vector<std::string> best_notes;
        bool changed                   = false;
        bool splits                    = false;
        int best_resident              = -1;
        sd::LayerSplitFill chosen_fill = requested_fill;
        for (sd::LayerSplitFill fill : fills) {
            sd::set_layer_split_fill(fill);
            std::vector<Runtime> candidate = runtimes;
            std::vector<std::string> candidate_notes;
            const bool candidate_changed = plan_multi_gpu_runtimes(components, devices, ram_budget_bytes, mode, explicit_runtime,
                                                                   allow_split, candidate, candidate_notes);
            const Plan trial             = compute_plan(components, devices, ram_budget_bytes, candidate);
            int resident                 = -1;
            if (trial.valid) {
                resident = 0;
                for (size_t ci = 0; ci < components.size(); ++ci) {
                    resident += components[ci].params_bytes > 0 && trial.decisions[ci].params_location == ParamsLocation::MAIN_GPU ? 1 : 0;
                }
            }
            if (resident > best_resident) {
                best_resident = resident;
                chosen_fill   = fill;
                changed       = candidate_changed;
                best_runtimes = std::move(candidate);
                best_notes    = std::move(candidate_notes);
                splits        = changed && std::any_of(best_runtimes.begin(), best_runtimes.end(), [](const Runtime& runtime) {
                             return runtime.estimate_shares && runtime.devices.size() > 1;
                         });
            }
        }
        sd::set_layer_split_fill(splits ? chosen_fill : requested_fill);
        if (splits && requested_fill == sd::LayerSplitFill::AUTO) {
            best_notes.push_back(std::string("the split fills the GPUs ") +
                                 (chosen_fill == sd::LayerSplitFill::FILL_FIRST ? "first one first" : "in proportion to their memory") +
                                 " (--layer-split-fill auto), which keeps the most modules on the GPUs");
        }
        notes.insert(notes.end(), best_notes.begin(), best_notes.end());
        if (changed) {
            runtimes = std::move(best_runtimes);
        }
        return changed;
    }

    static bool resolve_runtimes(const std::vector<Component>& components,
                                 const std::vector<Device>& devices,
                                 std::string& runtime_spec,
                                 std::vector<Runtime>& runtimes,
                                 std::vector<bool>& explicit_runtime,
                                 std::string& error) {
        SDBackendAssignment assignment;
        if (!sd_parse_backend_assignment(runtime_spec, &assignment, &error)) {
            return false;
        }
        const size_t main_device        = select_main_device(devices);
        const SDBackendModule modules[] = {SDBackendModule::DIFFUSION, SDBackendModule::TE, SDBackendModule::VAE};
        for (size_t ci = 0; ci < components.size(); ++ci) {
            const Component& comp = components[ci];
            std::string name      = assignment.get(modules[int(comp.kind)]);
            explicit_runtime[ci]  = !name.empty();
            if (name.empty()) {
                name = main_device == SIZE_MAX ? "cpu" : devices[main_device].name;
                if (comp.params_bytes > 0) {
                    append_assignment(runtime_spec, module_key(comp.kind), name);
                }
            }
            Runtime runtime;
            for (const std::string& part : split_string(name, '&')) {
                if (trim(part).empty()) {
                    continue;
                }
                const std::string resolved = sd_backend_resolve_name(part);
                if (resolved.empty()) {
                    error = "backend '" + part + "' was not found";
                    return false;
                }
                if (!runtime.name.empty()) {
                    runtime.name += "&";
                }
                runtime.name += resolved;
                for (size_t di = 0; di < devices.size(); ++di) {
                    if (devices[di].name == resolved &&
                        std::find(runtime.devices.begin(), runtime.devices.end(), di) == runtime.devices.end()) {
                        runtime.devices.push_back(di);
                    }
                }
            }
            runtimes.push_back(std::move(runtime));
        }
        return true;
    }

    static bool derive_component_specs(const std::vector<Component>& components,
                                       sd::ggml_graph_cut::MaxVramAssignment& budgets,
                                       std::string& runtime_spec,
                                       std::string& params_spec,
                                       int& memory_guard_percent,
                                       const Options& options) {
        std::string error;
        if (!budgets.canonicalize_backend_keys(&error)) {
            LOG_ERROR("%s", error.c_str());
            return false;
        }

        // Resolve once to ensure dynamic backends are loaded before enumerating devices.
        sd_backend_resolve_name("");
        const auto devices = enumerate_gpu_devices(budgets, !runtime_spec.empty(), options.available_vram);
        std::vector<Runtime> runtimes;
        std::vector<bool> explicit_runtime(components.size(), false);
        if (!runtime_spec.empty() && !resolve_runtimes(components, devices, runtime_spec, runtimes, explicit_runtime, error)) {
            LOG_ERROR("%s", error.c_str());
            return false;
        }
        const int64_t free_ram = available_ram_bytes();
        // Without a GPU everything runs on the CPU and the compute buffers come out of RAM too.
        const bool cpu_only = devices.empty() && std::all_of(runtimes.begin(), runtimes.end(), [](const Runtime& runtime) {
                                  return runtime.devices.empty();
                              });
        if (cpu_only && runtimes.empty()) {
            if (free_ram < 0) {
                LOG_WARN("auto-fit: no GPU devices and available RAM is unknown; using the default backend");
                return true;
            }
            runtimes.assign(components.size(), Runtime{"CPU", {}});
        }
        int64_t compute_reserve = 0;
        if (cpu_only) {
            for (const Component& comp : components) {
                compute_reserve = std::max(compute_reserve, comp.reserve_bytes);
            }
        }
        int64_t ram_budget = std::max<int64_t>(free_ram - std::max<int64_t>(2048 * MiB, free_ram / 10) - compute_reserve, 0);
        if (options.max_ram_bytes)
            ram_budget = static_cast<int64_t>(std::min<uint64_t>(ram_budget, options.max_ram_bytes));

        const bool append_runtime_names = runtimes.empty();
        std::vector<std::string> multi_gpu_notes;
        if (!cpu_only) {
            std::vector<Runtime> base = runtimes;
            if (base.empty() && select_main_device(devices) != SIZE_MAX) {
                const size_t main_device = select_main_device(devices);
                base.assign(components.size(), Runtime{devices[main_device].name, {main_device}});
            }
            std::vector<Runtime> planned = base;
            const bool changed           = !planned.empty() &&
                                 plan_multi_gpu_runtimes_with_fill(components, devices, ram_budget, options.multi_gpu, explicit_runtime,
                                                                   split_allowed(components, options.split_mode_spec), planned, multi_gpu_notes);
            if (changed) {
                if (!append_runtime_names) {
                    for (size_t ci = 0; ci < components.size(); ++ci) {
                        if (!explicit_runtime[ci] && components[ci].params_bytes > 0 && planned[ci].name != runtimes[ci].name) {
                            set_assignment(runtime_spec, module_key(components[ci].kind), planned[ci].name);
                        }
                    }
                }
                runtimes = std::move(planned);
            }
        }

        const auto plan = compute_plan(components, devices, ram_budget, runtimes);
        params_spec.clear();
        if (!plan.valid) {
            if (devices.empty()) {
                LOG_WARN("auto-fit: no GPU devices; using the default backend");
            } else {
                LOG_WARN("auto-fit: no GPU memory budget available; using CPU");
                runtime_spec = "cpu";
            }
            return true;
        }

        print_plan(plan, components, devices, free_ram, ram_budget);
        for (const std::string& note : multi_gpu_notes) {
            LOG_INFO("auto-fit: %s", note.c_str());
        }
        for (const std::string& warning : shadowed_runtime_warnings(plan, devices)) {
            LOG_WARN("%s", warning.c_str());
        }
        if (!cpu_only) {
            const std::string streaming = disk_streaming_warning(plan, components, free_ram, ram_budget);
            if (!streaming.empty()) {
                LOG_WARN("%s", streaming.c_str());
            }
        }
        if (cpu_only) {
            const bool streams_from_disk = std::any_of(plan.decisions.begin(), plan.decisions.end(), [](const Decision& decision) {
                return decision.params_location == ParamsLocation::DISK;
            });
            if (!streams_from_disk) {
                LOG_INFO("auto-fit: CPU only; the weights fit in RAM");
                return true;
            }
            if (memory_guard_percent <= 0) {
                memory_guard_percent = 90;
                LOG_INFO("auto-fit: CPU only and some weights do not fit in RAM; enabling the memory guard at %d%% so the model runs in segments", memory_guard_percent);
            }
        }
        for (size_t ci = 0; ci < components.size(); ++ci) {
            if (components[ci].params_bytes == 0) {
                continue;
            }
            const char* key = module_key(components[ci].kind);
            if (append_runtime_names) {
                append_assignment(runtime_spec, key, plan.runtimes[ci].name);
            }
            if (plan.decisions[ci].params_location != ParamsLocation::MAIN_GPU) {
                append_assignment(params_spec, key, params_backend_name(plan.decisions[ci], devices));
            }
        }

        // Keep the planner's safety margin when the runner resolves its device limits.
        for (const Device& device : devices) {
            if (device.budget_bytes > 0) {
                budgets.backend_gib[budget_key(device.name)] = (float)(device.budget_bytes / (1024.0 * MiB));
            }
        }
        budgets.resolved_backend_bytes.clear();

        LOG_INFO("auto-fit: --backend \"%s\"%s%s%s",
                 runtime_spec.empty() ? "(default)" : runtime_spec.c_str(),
                 params_spec.empty() ? "" : " --params-backend \"",
                 params_spec.c_str(), params_spec.empty() ? "" : "\"");
        return true;
    }

    std::string select_default_backend(const sd::ggml_graph_cut::MaxVramAssignment& budgets, const AvailableVRAM& available_vram) {
        const auto fallback = sd_backend_resolve_name("");
        const auto devices  = enumerate_gpu_devices(budgets, false, available_vram);
        const auto selected = select_main_device(devices);
        return selected == SIZE_MAX ? fallback : devices[selected].name;
    }

    bool derive_backend_specs(ModelLoader& loader,
                              ggml_type override_wtype,
                              sd::ggml_graph_cut::MaxVramAssignment& budgets,
                              std::string& runtime_spec,
                              std::string& params_spec,
                              int& memory_guard_percent,
                              const Options& options) {
        return derive_component_specs(estimate_components(loader, override_wtype), budgets, runtime_spec, params_spec,
                                      memory_guard_percent, options);
    }

    bool derive_backend_specs(ModelLoader& loader,
                              ggml_type override_wtype,
                              sd::ggml_graph_cut::MaxVramAssignment& budgets,
                              std::string& runtime_spec,
                              std::string& params_spec,
                              const AvailableVRAM& available_vram,
                              size_t max_ram_bytes) {
        Options options;
        options.available_vram   = available_vram;
        options.max_ram_bytes    = max_ram_bytes;
        int memory_guard_percent = 0;
        return derive_backend_specs(loader, override_wtype, budgets, runtime_spec, params_spec, memory_guard_percent, options);
    }

    bool derive_backend_specs(ModelLoader& loader,
                              ggml_type override_wtype,
                              sd::ggml_graph_cut::MaxVramAssignment& budgets,
                              std::string& runtime_spec,
                              std::string& params_spec) {
        return derive_backend_specs(loader, override_wtype, budgets, runtime_spec, params_spec, {});
    }

    bool prepare_vae_decode_retry_tiling(sd::TilingConfig& tiling_params,
                                         bool prefer_temporal_tiling,
                                         ggml_status status,
                                         int latent_tile_size_w,
                                         int latent_tile_size_h,
                                         int scale_factor) {
        // Execution failures can leave the device unusable; tiling only helps with allocation failures.
        if (status != GGML_STATUS_ALLOC_FAILED) {
            return false;
        }
        const char* retry_mode = nullptr;
        if (prefer_temporal_tiling && !tiling_params.temporal_tiling) {
            tiling_params.temporal_tiling = true;
            retry_mode                    = tiling_params.enabled ? "spatial+temporal" : "temporal";
        } else {
            if (latent_tile_size_w <= 0 || latent_tile_size_h <= 0 || scale_factor <= 0) {
                return false;
            }
            auto smaller_tile = [&](int size) {
                return std::min(size, std::max(4, size / 2));
            };
            const int tile_size_w = smaller_tile(latent_tile_size_w);
            const int tile_size_h = smaller_tile(latent_tile_size_h);
            if (tile_size_w == latent_tile_size_w && tile_size_h == latent_tile_size_h) {
                return false;
            }
            tiling_params.enabled     = true;
            tiling_params.rel_size_w  = 0.0f;
            tiling_params.rel_size_h  = 0.0f;
            tiling_params.tile_size_w = tile_size_w * scale_factor;
            tiling_params.tile_size_h = tile_size_h * scale_factor;
            retry_mode                = tiling_params.temporal_tiling ? "spatial+temporal" : "spatial";
            LOG_WARN("Reducing VAE decode tiles from %dx%d to %dx%d image pixels",
                     latent_tile_size_w * scale_factor, latent_tile_size_h * scale_factor,
                     tiling_params.tile_size_w, tiling_params.tile_size_h);
        }

        LOG_WARN("VAE decode ran out of memory; retrying with %s tiling",
                 retry_mode);
        return true;
    }

}  // namespace sd::backend_fit
