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

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
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
    GPU_OFFLOAD_PARAMS,    // params in RAM, compute on GPU
    GPU_LAYER_SPLIT,       // params split across multiple GPUs at block boundaries (sched-based)
    GPU_TENSOR_SPLIT,      // matmul weights row-split across GPUs (CUDA split-buft, single backend)
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

    // Populated when placement == GPU_LAYER_SPLIT. Contains the device IDs
    // that share this component (in order) and each device's estimated share
    // of the params. The order also defines block-range partitioning: the
    // i-th device gets a contiguous range of blocks proportional to share[i].
    std::vector<int>     split_device_ids;
    std::vector<int64_t> split_share_bytes;
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

enum class MultiGpuMode {
    OFF,    // never split a single component across GPUs
    ROW,    // CUDA-only: row-split matmul weights via cuda_split_buffer_type
    LAYER,  // generic: assign block-indexed tensors to per-block backends + sched
};

inline const char* multi_gpu_mode_str(MultiGpuMode m) {
    switch (m) {
        case MultiGpuMode::OFF:   return "off";
        case MultiGpuMode::ROW:   return "row";
        case MultiGpuMode::LAYER: return "layer";
    }
    return "?";
}

inline MultiGpuMode str_to_multi_gpu_mode(const std::string& s) {
    if (s == "off")   return MultiGpuMode::OFF;
    if (s == "row")   return MultiGpuMode::ROW;
    if (s == "layer") return MultiGpuMode::LAYER;
    return MultiGpuMode::ROW;  // default
}

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
        name.rfind("text_encoder.", 0) == 0 ||
        // Connector / text projection layers that run on the conditioner
        // backend (e.g. LTX-2's text_embedding_projection: video/audio
        // aggregate embeds + projection that map LLM hidden states into
        // DiT-input space).
        name.rfind("text_embedding_projection.", 0) == 0 ||
        contains(".aggregate_embed.")) {
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

// Per-GPU share for a layer-split component: free-VRAM-weighted partition
// of params, plus the full compute reserve on each participating device.
// (Compute reserve is per-device since each shard activates its own kernels.)
inline std::vector<int64_t> layer_split_shares(int64_t                    params_bytes,
                                               int64_t                    compute_bytes,
                                               const std::vector<Device>& devices,
                                               const std::vector<size_t>& gpu_idxs) {
    std::vector<int64_t> out(gpu_idxs.size(), 0);
    int64_t total_free = 0;
    for (size_t k = 0; k < gpu_idxs.size(); k++) {
        total_free += std::max<int64_t>(0, devices[gpu_idxs[k]].free_bytes);
    }
    if (total_free <= 0) return out;
    for (size_t k = 0; k < gpu_idxs.size(); k++) {
        double r = double(std::max<int64_t>(0, devices[gpu_idxs[k]].free_bytes)) / double(total_free);
        out[k]   = int64_t(double(params_bytes) * r) + compute_bytes;
    }
    return out;
}

// Peak per device = MAX of any single component's footprint on that device,
// because free_params_immediately frees params between phases so components
// time-share VRAM.
inline int64_t gpu_peak(int                           gpu_idx,
                        const std::vector<Placement>& pl,
                        const std::vector<int>&       dev,
                        const std::vector<Component>& components,
                        const std::vector<Device>&    devices = {}) {
    int64_t peak = 0;
    for (size_t i = 0; i < components.size(); i++) {
        int64_t footprint = 0;
        if (pl[i] == Placement::GPU || pl[i] == Placement::GPU_OFFLOAD_PARAMS) {
            if (dev[i] != gpu_idx) continue;
            footprint = components[i].params_bytes + components[i].compute_bytes;
        } else if (pl[i] == Placement::GPU_TENSOR_SPLIT) {
            // Row-split: every GPU in the mask gets a free-VRAM-weighted
            // share of params; the compute reserve lands on the BIGGEST
            // GPU (which becomes the runner's main backend).
            const int mask = dev[i];
            if (!(mask & (1 << gpu_idx))) continue;
            std::vector<size_t> gpu_idxs;
            for (size_t k = 0; k < devices.size(); k++) {
                if (mask & (1 << k)) gpu_idxs.push_back(k);
            }
            int slot = -1;
            int biggest_slot = 0;
            int64_t biggest_mem = -1;
            for (size_t k = 0; k < gpu_idxs.size(); k++) {
                if (int(gpu_idxs[k]) == gpu_idx) slot = int(k);
                if (devices[gpu_idxs[k]].total_bytes > biggest_mem) {
                    biggest_mem  = devices[gpu_idxs[k]].total_bytes;
                    biggest_slot = int(k);
                }
            }
            if (slot < 0) continue;
            auto shares = layer_split_shares(components[i].params_bytes,
                                             /*compute_bytes=*/0,
                                             devices, gpu_idxs);
            footprint = shares[slot];
            if (slot == biggest_slot) {
                footprint += components[i].compute_bytes;
            }
        } else if (pl[i] == Placement::GPU_LAYER_SPLIT) {
            // dev[i] holds the bitmask of participating GPU indices into the
            // devices[] vector (encoded by the planner). Look up our slot.
            const int mask = dev[i];
            std::vector<size_t> gpu_idxs;
            for (size_t k = 0; k < devices.size(); k++) {
                if (mask & (1 << k)) gpu_idxs.push_back(k);
            }
            // Find this gpu's slot in gpu_idxs.
            int slot = -1;
            for (size_t k = 0; k < gpu_idxs.size(); k++) {
                if (int(gpu_idxs[k]) == gpu_idx) { slot = int(k); break; }
            }
            if (slot < 0) continue;
            auto shares = layer_split_shares(components[i].params_bytes,
                                             components[i].compute_bytes,
                                             devices, gpu_idxs);
            footprint = shares[slot];
        }
        peak = std::max(peak, footprint);
    }
    return peak;
}

inline Plan compute_plan(const std::vector<Component>& components,
                         const std::vector<Device>&    devices,
                         int64_t                       margin_bytes,
                         bool                          allow_multi_gpu = true,
                         MultiGpuMode                  mode = MultiGpuMode::ROW) {
    const size_t nC = components.size();
    const size_t nG = devices.size();
    if (!allow_multi_gpu) {
        mode = MultiGpuMode::OFF;
    }

    std::vector<int64_t> cap(nG, 0);
    for (size_t g = 0; g < nG; g++) {
        cap[g] = std::max<int64_t>(0, devices[g].free_bytes - margin_bytes);
    }

    struct OptionSlot {
        Placement placement;
        int       device_idx;
    };

    // Layer-split is only meaningful for components made up of many similarly
    // shaped blocks. DiT and Conditioner (LLM transformer) qualify; the VAE
    // is too structurally heterogeneous for naive block partitioning.
    auto supports_layer_split = [](ComponentKind k) {
        return k == ComponentKind::DIT || k == ComponentKind::CONDITIONER;
    };

    auto build_options = [&](const Component& c) {
        std::vector<OptionSlot> opts;
        for (size_t g = 0; g < nG; g++) {
            opts.push_back({Placement::GPU, int(g)});
            if (c.supports_offload) {
                opts.push_back({Placement::GPU_OFFLOAD_PARAMS, int(g)});
            }
        }
        // Multi-GPU split: one option type per mode. Encoded as a bitmask
        // of participating GPUs in device_idx.
        if (mode == MultiGpuMode::ROW && nG >= 2 && supports_layer_split(c.kind)) {
            // Row-split spans all GPUs; single option with all bits set.
            int all_mask = (1 << nG) - 1;
            opts.push_back({Placement::GPU_TENSOR_SPLIT, all_mask});
        }
        if (mode == MultiGpuMode::LAYER && nG >= 2 && supports_layer_split(c.kind)) {
            // Layer-split: enumerate non-trivial subsets (size >= 2).
            const int max_mask = 1 << nG;
            for (int mask = 1; mask < max_mask; mask++) {
                if (__builtin_popcount(mask) < 2) continue;
                opts.push_back({Placement::GPU_LAYER_SPLIT, mask});
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
            } else if (pl[i] == Placement::GPU_TENSOR_SPLIT) {
                // Row-split: cheaper than layer-split (no sched cross-
                // backend doubling) but pays per-matmul cross-device
                // reductions. Score it slightly above LAYER_SPLIT so the
                // planner prefers it when both fit.
                s += 8 * pw;
                for (size_t g = 0; g < nG; g++) {
                    if (dev[i] & (1 << g)) gpus_used.insert(int(g));
                }
            } else if (pl[i] == Placement::GPU_LAYER_SPLIT) {
                // Better than CPU but worse than fitting on a single GPU
                // (cross-GPU traffic between blocks).
                s += 7 * pw;
                for (size_t g = 0; g < nG; g++) {
                    if (dev[i] & (1 << g)) gpus_used.insert(int(g));
                }
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
                    if (gpu_peak(int(g), pl, dev, components, devices) > cap[g]) { feasible = false; break; }
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
                if (gpu_peak(int(g), pl, dev, components, devices) > cap[g]) { feasible = false; break; }
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
        } else if (best_pl[i] == Placement::GPU_TENSOR_SPLIT) {
            std::vector<size_t> gpu_idxs;
            for (size_t k = 0; k < nG; k++) {
                if (best_dev[i] & (1 << k)) gpu_idxs.push_back(k);
            }
            auto shares = layer_split_shares(c.params_bytes, /*compute_bytes=*/0,
                                             devices, gpu_idxs);
            // Sort participating GPUs by descending TOTAL memory so the
            // largest device is the "main" (gets the row-split's compute
            // buffer + sub-runners that don't get their own spec). This
            // matches the user's preference: always use the bigger GPU
            // as main for splits.
            std::vector<size_t> order(gpu_idxs.size());
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
                return devices[gpu_idxs[a]].total_bytes > devices[gpu_idxs[b]].total_bytes;
            });

            int64_t max_share = 0;
            for (size_t pos = 0; pos < order.size(); pos++) {
                size_t k = order[pos];
                d.split_device_ids.push_back(devices[gpu_idxs[k]].id);
                int64_t share = shares[k];
                if (pos == 0) share += c.compute_bytes;  // main (= biggest) gets compute
                d.split_share_bytes.push_back(share);
                max_share = std::max(max_share, share);
            }
            d.device_id        = d.split_device_ids.empty() ? DEVICE_ID_CPU : d.split_device_ids[0];
            d.on_device_bytes  = max_share;
            plan.any_changes   = true;
        } else if (best_pl[i] == Placement::GPU_LAYER_SPLIT) {
            std::vector<size_t> gpu_idxs;
            for (size_t k = 0; k < nG; k++) {
                if (best_dev[i] & (1 << k)) gpu_idxs.push_back(k);
            }
            auto shares = layer_split_shares(c.params_bytes, c.compute_bytes,
                                             devices, gpu_idxs);
            // Sort participating GPUs by descending TOTAL memory so the
            // physically bigger GPU is listed first (and becomes the runner's
            // main backend). Sub-runners that don't get the layer-split spec
            // (e.g. the LTX-2 text projection) follow the main backend.
            std::vector<size_t> order(gpu_idxs.size());
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
                return devices[gpu_idxs[a]].total_bytes > devices[gpu_idxs[b]].total_bytes;
            });

            int64_t max_share = 0;
            for (size_t pos = 0; pos < order.size(); pos++) {
                size_t k = order[pos];
                d.split_device_ids.push_back(devices[gpu_idxs[k]].id);
                d.split_share_bytes.push_back(shares[k]);
                max_share = std::max(max_share, shares[k]);
            }
            d.device_id        = d.split_device_ids.empty() ? DEVICE_ID_CPU : d.split_device_ids[0];
            d.on_device_bytes  = max_share;
            plan.any_changes   = true;
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
        plan.device_bytes[devices[g].id] = gpu_peak(int(g), best_pl, best_dev, components, devices);
    }
    return plan;
}

inline const char* placement_str(Placement p) {
    switch (p) {
        case Placement::CPU: return "CPU";
        case Placement::GPU: return "GPU";
        case Placement::GPU_OFFLOAD_PARAMS: return "GPU(params->RAM)";
        case Placement::GPU_LAYER_SPLIT: return "GPU(layer-split)";
        case Placement::GPU_TENSOR_SPLIT: return "GPU(row-split)";
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
        } else if (d.placement == Placement::GPU_LAYER_SPLIT ||
                   d.placement == Placement::GPU_TENSOR_SPLIT) {
            std::string ids;
            const char* tag = d.placement == Placement::GPU_TENSOR_SPLIT ? "row" : "layer";
            for (size_t k = 0; k < d.split_device_ids.size(); k++) {
                if (k > 0) ids += "+";
                ids += "GPU" + std::to_string(d.split_device_ids[k]);
                ids += "(" + std::to_string(d.split_share_bytes[k] / MiB) + "MiB)";
            }
            LOG_INFO("    %-12s -> %s-split %s",
                     d.name.c_str(), tag, ids.c_str());
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
