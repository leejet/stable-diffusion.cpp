#ifndef __SD_BACKEND_FIT_HPP__
#define __SD_BACKEND_FIT_HPP__

// Auto-fit algorithm for distributing DiT, VAE, and conditioner across the
// available GPU devices and system RAM.
//
// Each component is treated as a single atomic unit that lives entirely on
// one device (plus its compute buffer on the same device). There is no
// intra-tensor row split: cross-device parallelism comes from placing
// different components on different GPUs, not from splitting individual
// matmul weights — the equivalent of llama.cpp's LLAMA_SPLIT_MODE_LAYER
// at the component granularity.
//
// Placement priority: DiT + compute buffer -> VAE -> Conditioner.
// Overflow falls back to CPU (or GPU_OFFLOAD_PARAMS for components that
// support streaming params from RAM at compute time).

#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"

#include "model.h"
#include "util.h"

namespace backend_fit {

constexpr int64_t MiB           = 1024 * 1024;
constexpr int     DEVICE_ID_CPU = -1;

enum class ComponentKind {
    DIT,
    VAE,
    CONDITIONER,
};

enum class Placement {
    CPU,
    GPU,
    GPU_OFFLOAD_PARAMS,  // params in RAM, compute on GPU
};

struct Component {
    ComponentKind kind;
    std::string   name;
    int64_t       params_bytes     = 0;
    int64_t       compute_bytes    = 0;
    bool          supports_offload = false;
};

struct Device {
    int                id = DEVICE_ID_CPU;
    std::string        name;
    std::string        description;
    int64_t            free_bytes  = 0;
    int64_t            total_bytes = 0;
    ggml_backend_dev_t dev         = nullptr;  // backing ggml device handle (GPU only)
};

struct Decision {
    ComponentKind kind;
    std::string   name;
    Placement     placement       = Placement::CPU;
    int           device_id       = DEVICE_ID_CPU;
    int64_t       on_device_bytes = 0;
    int64_t       on_host_bytes   = 0;
};

struct Plan {
    std::vector<Decision>  decisions;
    std::map<int, int64_t> device_bytes;
    int64_t                host_bytes  = 0;
    bool                   any_changes = false;
};

struct ComputeReserves {
    int64_t dit_bytes         = int64_t(2048) * MiB;
    int64_t vae_bytes         = int64_t(1024) * MiB;
    int64_t conditioner_bytes = int64_t(512) * MiB;
};

// --- Classification -------------------------------------------------------

inline bool classify_tensor(const std::string& name, ComponentKind& out) {
    auto contains = [&](const char* s) { return name.find(s) != std::string::npos; };

    if (contains("model.diffusion_model.") || contains("unet.")) {
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
        name.rfind("text_encoder.", 0) == 0) {
        out = ComponentKind::CONDITIONER;
        return true;
    }

    return false;
}

// --- Memory estimation ----------------------------------------------------

inline std::vector<Component> estimate_components(ModelLoader&           loader,
                                                  ggml_type              override_wtype,
                                                  int64_t                alignment,
                                                  const ComputeReserves& reserves) {
    auto& storage = loader.get_tensor_storage_map();

    int64_t bytes[3] = {0, 0, 0};

    for (auto& [name, ts_const] : storage) {
        TensorStorage ts = ts_const;
        if (is_unused_tensor(ts.name)) {
            continue;
        }

        ComponentKind k;
        if (!classify_tensor(ts.name, k)) {
            continue;
        }

        if (override_wtype != GGML_TYPE_COUNT &&
            loader.tensor_should_be_converted(ts, override_wtype)) {
            ts.type = override_wtype;
        } else if (ts.expected_type != GGML_TYPE_COUNT && ts.expected_type != ts.type) {
            ts.type = ts.expected_type;
        }

        bytes[int(k)] += ts.nbytes() + alignment;
    }

    std::vector<Component> out;
    out.reserve(3);
    out.push_back({ComponentKind::DIT, "DiT",
                   bytes[int(ComponentKind::DIT)], reserves.dit_bytes, true});
    out.push_back({ComponentKind::VAE, "VAE",
                   bytes[int(ComponentKind::VAE)], reserves.vae_bytes, false});
    out.push_back({ComponentKind::CONDITIONER, "Conditioner",
                   bytes[int(ComponentKind::CONDITIONER)], reserves.conditioner_bytes, true});
    return out;
}

// --- Device enumeration ---------------------------------------------------

inline std::vector<Device> enumerate_gpu_devices() {
    std::vector<Device> out;
    int gpu_idx = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        Device d;
        d.id          = gpu_idx++;
        d.dev         = dev;
        d.name        = ggml_backend_dev_name(dev);
        d.description = ggml_backend_dev_description(dev);
        size_t free_b = 0, total_b = 0;
        ggml_backend_dev_memory(dev, &free_b, &total_b);
        d.free_bytes  = int64_t(free_b);
        d.total_bytes = int64_t(total_b);
        out.push_back(d);
    }
    return out;
}

// --- Core algorithm -------------------------------------------------------

// Peak per device = MAX of any single component's footprint on that device,
// because free_params_immediately frees params between phases so components
// time-share VRAM.
inline int64_t gpu_peak(int                           gpu_idx,
                        const std::vector<Placement>& pl,
                        const std::vector<int>&       dev,
                        const std::vector<Component>& components) {
    int64_t peak = 0;
    for (size_t i = 0; i < components.size(); i++) {
        if (dev[i] != gpu_idx) continue;
        int64_t footprint = 0;
        if (pl[i] == Placement::GPU || pl[i] == Placement::GPU_OFFLOAD_PARAMS) {
            footprint = components[i].params_bytes + components[i].compute_bytes;
        }
        peak = std::max(peak, footprint);
    }
    return peak;
}

inline Plan compute_plan(const std::vector<Component>& components,
                         const std::vector<Device>&    devices,
                         int64_t                       margin_bytes,
                         bool                          allow_multi_gpu = true) {
    const size_t nC = components.size();
    const size_t nG = devices.size();

    std::vector<int64_t> cap(nG, 0);
    for (size_t g = 0; g < nG; g++) {
        cap[g] = std::max<int64_t>(0, devices[g].free_bytes - margin_bytes);
    }

    struct OptionSlot {
        Placement placement;
        int       device_idx;
    };

    auto build_options = [&](const Component& c) {
        std::vector<OptionSlot> opts;
        for (size_t g = 0; g < nG; g++) {
            opts.push_back({Placement::GPU, int(g)});
            if (c.supports_offload) {
                opts.push_back({Placement::GPU_OFFLOAD_PARAMS, int(g)});
            }
        }
        opts.push_back({Placement::CPU, -1});
        return opts;
    };

    std::vector<std::vector<OptionSlot>> options;
    options.reserve(nC);
    for (const Component& c : components) {
        options.push_back(build_options(c));
    }

    auto priority_weight = [](ComponentKind k) -> int {
        switch (k) {
            case ComponentKind::DIT:         return 300;
            case ComponentKind::CONDITIONER: return 120;
            case ComponentKind::VAE:         return 60;
        }
        return 1;
    };

    auto score = [&](const std::vector<Placement>& pl, const std::vector<int>& dev) {
        int64_t       s = 0;
        std::set<int> gpus_used;
        for (size_t i = 0; i < nC; i++) {
            const int pw = priority_weight(components[i].kind);
            if (pl[i] == Placement::GPU) {
                s += 10 * pw;
                gpus_used.insert(dev[i]);
            } else if (pl[i] == Placement::GPU_OFFLOAD_PARAMS) {
                s += 5 * pw;
                gpus_used.insert(dev[i]);
            } else {
                s -= 10 * pw;
            }
        }
        if (allow_multi_gpu) {
            s += 2 * int64_t(gpus_used.size());
        }
        return s;
    };

    std::vector<size_t>    idx(nC, 0);
    std::vector<Placement> best_pl;
    std::vector<int>       best_dev;
    int64_t                best_score = std::numeric_limits<int64_t>::min();
    bool                   found_any  = false;

    while (true) {
        std::vector<Placement> pl(nC);
        std::vector<int>       dev(nC);
        for (size_t i = 0; i < nC; i++) {
            pl[i]  = options[i][idx[i]].placement;
            dev[i] = options[i][idx[i]].device_idx;
        }
        // Constraint: when multi-GPU is disabled, all GPU placements must
        // share the same device index.
        if (!allow_multi_gpu) {
            int common = -1;
            bool ok = true;
            for (size_t i = 0; i < nC; i++) {
                if (pl[i] == Placement::GPU || pl[i] == Placement::GPU_OFFLOAD_PARAMS) {
                    if (common < 0) common = dev[i];
                    else if (dev[i] != common) { ok = false; break; }
                }
            }
            if (ok) {
                bool feasible = true;
                for (size_t g = 0; g < nG; g++) {
                    if (gpu_peak(int(g), pl, dev, components) > cap[g]) { feasible = false; break; }
                }
                if (feasible) {
                    int64_t sc = score(pl, dev);
                    if (sc > best_score) {
                        best_score = sc; best_pl = pl; best_dev = dev; found_any = true;
                    }
                }
            }
        } else {
            bool feasible = true;
            for (size_t g = 0; g < nG; g++) {
                if (gpu_peak(int(g), pl, dev, components) > cap[g]) { feasible = false; break; }
            }
            if (feasible) {
                int64_t sc = score(pl, dev);
                if (sc > best_score) {
                    best_score = sc; best_pl = pl; best_dev = dev; found_any = true;
                }
            }
        }

        size_t pos = 0;
        while (pos < nC) {
            idx[pos]++;
            if (idx[pos] < options[pos].size()) break;
            idx[pos] = 0;
            pos++;
        }
        if (pos >= nC) break;
    }

    Plan plan;
    if (!found_any) {
        best_pl.assign(nC, Placement::CPU);
        best_dev.assign(nC, -1);
    }

    for (size_t i = 0; i < nC; i++) {
        const Component& c = components[i];
        Decision         d;
        d.kind      = c.kind;
        d.name      = c.name;
        d.placement = best_pl[i];
        if (best_pl[i] == Placement::CPU) {
            d.device_id      = DEVICE_ID_CPU;
            d.on_host_bytes  = c.params_bytes + c.compute_bytes;
            plan.any_changes = true;
        } else {
            d.device_id = devices[best_dev[i]].id;
            if (best_pl[i] == Placement::GPU) {
                d.on_device_bytes = c.params_bytes + c.compute_bytes;
            } else {
                d.on_device_bytes = c.params_bytes + c.compute_bytes;
                d.on_host_bytes   = c.params_bytes;
                plan.any_changes  = true;
            }
        }
        plan.decisions.push_back(d);
        plan.host_bytes += d.on_host_bytes;
    }

    for (size_t g = 0; g < nG; g++) {
        plan.device_bytes[devices[g].id] = gpu_peak(int(g), best_pl, best_dev, components);
    }
    return plan;
}

inline const char* placement_str(Placement p) {
    switch (p) {
        case Placement::CPU: return "CPU";
        case Placement::GPU: return "GPU";
        case Placement::GPU_OFFLOAD_PARAMS: return "GPU(params->RAM)";
    }
    return "?";
}

inline void print_plan(const Plan&                   plan,
                       const std::vector<Component>& components,
                       const std::vector<Device>&    devices,
                       int64_t                       margin_bytes) {
    LOG_INFO("auto-fit plan (margin=%lld MiB per GPU):", (long long)(margin_bytes / MiB));
    LOG_INFO("  available devices:");
    if (devices.empty()) {
        LOG_INFO("    (no GPU devices detected — all components will run on CPU)");
    }
    for (const Device& d : devices) {
        LOG_INFO("    %-12s %-32s free %6lld / %6lld MiB",
                 d.name.c_str(), d.description.c_str(),
                 (long long)(d.free_bytes / MiB),
                 (long long)(d.total_bytes / MiB));
    }
    LOG_INFO("  components:");
    for (const Component& c : components) {
        LOG_INFO("    %-12s params %6lld MiB, compute reserve %6lld MiB",
                 c.name.c_str(),
                 (long long)(c.params_bytes / MiB),
                 (long long)(c.compute_bytes / MiB));
    }
    LOG_INFO("  decisions:");
    for (const Decision& d : plan.decisions) {
        if (d.placement == Placement::CPU) {
            LOG_INFO("    %-12s -> CPU                (RAM %lld MiB)",
                     d.name.c_str(), (long long)(d.on_host_bytes / MiB));
        } else if (d.placement == Placement::GPU) {
            LOG_INFO("    %-12s -> GPU %d              (VRAM %lld MiB)",
                     d.name.c_str(), d.device_id,
                     (long long)(d.on_device_bytes / MiB));
        } else {
            LOG_INFO("    %-12s -> GPU %d (params RAM) (VRAM %lld MiB, RAM %lld MiB)",
                     d.name.c_str(), d.device_id,
                     (long long)(d.on_device_bytes / MiB),
                     (long long)(d.on_host_bytes / MiB));
        }
    }
    LOG_INFO("  projected per-device peak:");
    for (const Device& d : devices) {
        int64_t peak = 0;
        auto    it   = plan.device_bytes.find(d.id);
        if (it != plan.device_bytes.end()) peak = it->second;
        LOG_INFO("    %-12s peak %6lld / %6lld MiB free  (remaining %lld MiB)",
                 d.name.c_str(),
                 (long long)(peak / MiB),
                 (long long)(d.free_bytes / MiB),
                 (long long)((d.free_bytes - peak) / MiB));
    }
    LOG_INFO("    %-12s host RAM additional %lld MiB", "CPU",
             (long long)(plan.host_bytes / MiB));
}

inline const Decision* find_decision(const Plan& plan, ComponentKind kind) {
    for (const Decision& d : plan.decisions) {
        if (d.kind == kind) return &d;
    }
    return nullptr;
}

}  // namespace backend_fit

#endif  // __SD_BACKEND_FIT_HPP__
