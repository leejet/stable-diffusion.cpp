#ifndef __SD_BACKEND_FIT_HPP__
#define __SD_BACKEND_FIT_HPP__

// Auto-fit algorithm for distributing DiT, VAE, and conditioner (LLM +
// connector) across available GPU devices and system RAM.
//
// Inspired by llama.cpp's common_fit_params (tools/fit-params), but much
// coarser: sd.cpp treats each of {DiT, VAE, Conditioner} as a single atomic
// unit that lives entirely on one device (plus the DiT's compute buffer on
// the same GPU). There is no per-layer tensor_buft_overrides mechanism in
// sd.cpp today — the existing `offload_params_to_cpu` knob is the only way to
// "split" a model (it keeps params in RAM and streams them to the runtime
// backend per forward pass).
//
// Placement priority: DiT + compute buffer → VAE → Conditioner (+connector).
// Overflow falls back to CPU (or GPU_OFFLOAD_PARAMS for DiT).

#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "ggml.h"

#ifdef SD_USE_CUDA
#include "ggml-cuda.h"
#endif
#if defined(SD_USE_VULKAN)
#include "ggml-backend.h"
#endif

#include "model.h"
#include "util.h"

namespace backend_fit {

constexpr int64_t MiB          = 1024 * 1024;
constexpr int     DEVICE_ID_CPU = -1;

enum class ComponentKind {
    DIT,
    VAE,
    CONDITIONER,  // LLM + connector (share a backend)
};

enum class Placement {
    CPU,
    GPU,
    GPU_OFFLOAD_PARAMS,    // params in RAM, compute on GPU (DiT-only)
    GPU_TENSOR_SPLIT,      // params row-split across all GPUs (multi-GPU only)
};

struct Component {
    ComponentKind kind;
    std::string   name;
    int64_t       params_bytes     = 0;  // weight memory for this component
    int64_t       compute_bytes    = 0;  // reserved compute buffer on the chosen device
    bool          supports_offload = false;  // true only for DiT
};

struct Device {
    int         id          = DEVICE_ID_CPU;
    std::string name;
    std::string description;
    int64_t     free_bytes  = 0;
    int64_t     total_bytes = 0;
};

struct Decision {
    ComponentKind kind;
    std::string   name;
    Placement     placement       = Placement::CPU;
    int           device_id       = DEVICE_ID_CPU;
    int64_t       on_device_bytes = 0;  // contribution to device_id's device memory
    int64_t       on_host_bytes   = 0;  // contribution to host RAM
};

struct Plan {
    std::vector<Decision>  decisions;
    std::map<int, int64_t> device_bytes;   // device_id -> bytes used
    int64_t                host_bytes = 0;
    bool                   any_changes = false;  // true if a non-default placement was chosen
};

// Defaults chosen to leave enough headroom for typical diffusion/video models.
// Configurable via the CLI (--fit-compute-reserve-* in MiB).
struct ComputeReserves {
    int64_t dit_bytes         = int64_t(2048) * MiB;  // video DiT compute buffer
    int64_t vae_bytes         = int64_t(1024) * MiB;  // video VAE compute buffer
    int64_t conditioner_bytes = int64_t(512)  * MiB;  // LLM + connector combined
};

// --- Classification -------------------------------------------------------

// Classify a tensor name into a ComponentKind. Returns false if the tensor is
// unused / not a primary weight we should count.
inline bool classify_tensor(const std::string& name, ComponentKind& out) {
    // Connector lives inside `model.diffusion_model.*` by prefix but runs on
    // the conditioner's backend, so it gets charged to CONDITIONER.
    auto contains = [&](const char* s) { return name.find(s) != std::string::npos; };

    // LTX-2 specific: the checkpoint carries audio-to-video branch weights
    // (`.audio_*`, `.audio_to_video_*`, `.video_to_audio_*`, `audio_patchify_*`,
    // `audio_scale_shift_*`, `audio_prompt_*`) that the video-only LTX2
    // diffusion module does NOT wire in. They're logged as "unknown tensor"
    // warnings at load time and skipped. Excluding them here keeps the DiT
    // params estimate honest (~9 GB) instead of including ~4 GB of audio
    // tensors that never touch the GPU.
    if (contains(".audio_") ||
        contains("audio_patchify") ||
        contains("audio_aggregate") ||
        contains("audio_scale_shift") ||
        contains("audio_prompt") ||
        contains("a2v_ca_audio") ||
        contains("a2v_ca_video")) {
        return false;
    }

    if (contains("embeddings_connector") ||
        contains("aggregate_embed") ||
        contains("text_embedding_projection")) {
        out = ComponentKind::CONDITIONER;
        return true;
    }

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

// Sum params bytes per component using the same alignment padding and
// dtype-conversion rules as ModelLoader::get_params_mem_size.
inline std::vector<Component> estimate_components(ModelLoader&             loader,
                                                  ggml_type                override_wtype,
                                                  int64_t                  alignment,
                                                  const ComputeReserves&   reserves) {
    auto& storage = loader.get_tensor_storage_map();

    int64_t bytes[3] = {0, 0, 0};  // DIT, VAE, CONDITIONER
    int     counts[3] = {0, 0, 0};

    for (auto& [name, ts_const] : storage) {
        // Work on a copy so we can apply the dtype override without mutating.
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
            // Honor per-tensor retypes (e.g. LTX-2 Gemma → q8_0 fix in
            // stable-diffusion.cpp) when computing component size.
            ts.type = ts.expected_type;
        }

        int idx = int(k);
        bytes[idx]  += ts.nbytes() + alignment;
        counts[idx] += 1;
    }

    std::vector<Component> out;
    out.reserve(3);

    out.push_back(Component{
        ComponentKind::DIT, "DiT",
        bytes[int(ComponentKind::DIT)], reserves.dit_bytes,
        /*supports_offload=*/true,
    });
    out.push_back(Component{
        ComponentKind::VAE, "VAE",
        bytes[int(ComponentKind::VAE)], reserves.vae_bytes,
        /*supports_offload=*/false,
    });
    out.push_back(Component{
        ComponentKind::CONDITIONER, "Conditioner",
        bytes[int(ComponentKind::CONDITIONER)], reserves.conditioner_bytes,
        /*supports_offload=*/true,  // Gemma/etc. can stream params to GPU per encode
    });

    (void)counts;
    return out;
}

// --- Device enumeration ---------------------------------------------------

inline std::vector<Device> enumerate_gpu_devices() {
    std::vector<Device> out;

#if defined(SD_USE_CUDA)
    int count = ggml_backend_cuda_get_device_count();
    for (int i = 0; i < count; i++) {
        Device d;
        d.id = i;
        char desc_buf[256] = {0};
        ggml_backend_cuda_get_device_description(i, desc_buf, sizeof(desc_buf));
        d.description = desc_buf;
        d.name        = "CUDA" + std::to_string(i);
        size_t free_b = 0, total_b = 0;
        ggml_backend_cuda_get_device_memory(i, &free_b, &total_b);
        d.free_bytes  = int64_t(free_b);
        d.total_bytes = int64_t(total_b);
        out.push_back(d);
    }
#elif defined(SD_USE_VULKAN)
    int count = ggml_backend_vk_get_device_count();
    for (int i = 0; i < count; i++) {
        Device d;
        d.id   = i;
        d.name = "Vulkan" + std::to_string(i);
        // Vulkan backend does not expose a direct free-memory API; enumerate
        // via ggml_backend_dev so we can reuse ggml_backend_dev_memory.
        ggml_backend_dev_t dev = nullptr;
        for (size_t j = 0; j < ggml_backend_dev_count(); j++) {
            ggml_backend_dev_t candidate = ggml_backend_dev_get(j);
            if (ggml_backend_dev_type(candidate) == GGML_BACKEND_DEVICE_TYPE_GPU &&
                std::string(ggml_backend_dev_name(candidate)).find("Vulkan") != std::string::npos) {
                if (int(j) == i) { dev = candidate; break; }
            }
        }
        if (dev) {
            d.description = ggml_backend_dev_description(dev);
            size_t free_b = 0, total_b = 0;
            ggml_backend_dev_memory(dev, &free_b, &total_b);
            d.free_bytes  = int64_t(free_b);
            d.total_bytes = int64_t(total_b);
        }
        out.push_back(d);
    }
#endif

    return out;
}

// --- Core algorithm -------------------------------------------------------

// Peak VRAM per GPU is computed from two contributions:
//   1. `nonoffload_sum` — sum of params of every non-offload component on
//      that GPU. These live on VRAM from LOAD through their free-after-use
//      point, overlapping during the load window.
//   2. `max_active_footprint` — the largest per-phase compute footprint,
//      where a non-offload component's phase contributes just its compute
//      buffer, and an offload component's phase contributes params+compute
//      (its runtime buffer is full-size while active, freed by
//      `free_compute_buffer_immediately=true` between phases).
// peak = nonoffload_sum + max_active_footprint. This is conservative: it
// assumes the load-time accumulation overlaps with an active compute phase
// of the worst-case component. In practice load finishes before any compute
// starts so this over-counts by max_active_footprint during load — safe.
// Compute the per-GPU split share of a tensor-split component, weighted by
// each device's free VRAM. Returns a vector of size devices.size() with each
// device's portion of params_bytes (the same ratio applies to compute_bytes).
inline std::vector<double> tensor_split_ratios(const std::vector<Device>& devices) {
    double total = 0.0;
    std::vector<double> ratios(devices.size(), 0.0);
    for (size_t g = 0; g < devices.size(); g++) {
        ratios[g] = std::max<int64_t>(0, devices[g].free_bytes);
        total += ratios[g];
    }
    if (total <= 0.0) {
        // Fallback: equal split.
        std::fill(ratios.begin(), ratios.end(), 1.0 / std::max<size_t>(1, devices.size()));
        return ratios;
    }
    for (auto& r : ratios) r /= total;
    return ratios;
}

// Peak GPU memory per device. Components time-share VRAM at runtime
// (free_params_immediately frees params between phases), so peak per device
// is the MAX of any single component's resident footprint on that device,
// not the SUM. Footprint = params + compute (for whichever placement mode
// applies).
inline int64_t gpu_peak(int                           gpu_idx,
                        const std::vector<Placement>& pl,
                        const std::vector<int>&       dev,
                        const std::vector<Component>& components,
                        const std::vector<Device>&    devices = {}) {
    int64_t peak = 0;

    std::vector<double> split_ratios;
    bool                any_split = false;
    for (size_t i = 0; i < components.size(); i++) {
        if (pl[i] == Placement::GPU_TENSOR_SPLIT) { any_split = true; break; }
    }
    if (any_split && !devices.empty()) {
        split_ratios = tensor_split_ratios(devices);
    }

    for (size_t i = 0; i < components.size(); i++) {
        const Component& c = components[i];
        int64_t          footprint = 0;
        if (pl[i] == Placement::GPU) {
            if (dev[i] != gpu_idx) continue;
            footprint = c.params_bytes + c.compute_bytes;
        } else if (pl[i] == Placement::GPU_OFFLOAD_PARAMS) {
            if (dev[i] != gpu_idx) continue;
            footprint = c.params_bytes + c.compute_bytes;
        } else if (pl[i] == Placement::GPU_TENSOR_SPLIT) {
            if (gpu_idx < 0 || size_t(gpu_idx) >= split_ratios.size()) continue;
            double r       = split_ratios[gpu_idx];
            int64_t share  = int64_t(double(c.params_bytes + c.compute_bytes) * r);
            footprint      = share;
        }
        peak = std::max(peak, footprint);
    }
    return peak;
}

inline Plan compute_plan(const std::vector<Component>& components,
                          const std::vector<Device>&    devices,
                          int64_t                       margin_bytes,
                          bool                          allow_tensor_split = false) {
    // Enumeration approach: for each component we have up to (1 + 2 * nGPU)
    // placement options — CPU, or non-offload / offload on each GPU (offload
    // only when the component supports it). We try all combinations, filter
    // infeasible ones (any GPU's computed peak exceeds its free-margin cap),
    // and pick the combination with the best score.
    //
    // Score rewards GPU placement (heavily), non-offload over offload
    // (avoids per-step stream cost), and GPU diversity (use multiple GPUs
    // when possible instead of packing onto one). Priority runtime hot
    // components are weighted higher: DiT >> Conditioner > VAE.
    const size_t nC = components.size();
    const size_t nG = devices.size();

    std::vector<int64_t> cap(nG, 0);
    for (size_t g = 0; g < nG; g++) {
        cap[g] = devices[g].free_bytes - margin_bytes;
        if (cap[g] < 0) cap[g] = 0;
    }

    struct OptionSlot {
        Placement placement;
        int       device_idx;  // index into devices, or -1 for CPU
    };

    auto build_options = [&](const Component& c) {
        std::vector<OptionSlot> opts;
        for (size_t g = 0; g < nG; g++) {
            opts.push_back({Placement::GPU, int(g)});
            if (c.supports_offload) {
                opts.push_back({Placement::GPU_OFFLOAD_PARAMS, int(g)});
            }
        }
        // Tensor split: only for the heavy components (DiT, Conditioner) and
        // only when there is more than one GPU. The VAE is too light to be
        // worth splitting and isn't currently wired for split anyway.
        if (allow_tensor_split && nG >= 2 &&
            (c.kind == ComponentKind::DIT || c.kind == ComponentKind::CONDITIONER)) {
            opts.push_back({Placement::GPU_TENSOR_SPLIT, -1});
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
            case ComponentKind::DIT:         return 300;  // runs N times per generation
            case ComponentKind::CONDITIONER: return 120;  // one large forward per prompt
            case ComponentKind::VAE:         return 60;   // one decode per generation
        }
        return 1;
    };

    auto score = [&](const std::vector<Placement>& pl,
                     const std::vector<int>&       dev) {
        int64_t s = 0;
        std::set<int> gpus_used;
        for (size_t i = 0; i < nC; i++) {
            const int pw = priority_weight(components[i].kind);
            if (pl[i] == Placement::GPU) {
                s += 10 * pw;
                gpus_used.insert(dev[i]);
            } else if (pl[i] == Placement::GPU_OFFLOAD_PARAMS) {
                s += 5 * pw;  // still on GPU but with per-step stream overhead
                gpus_used.insert(dev[i]);
            } else if (pl[i] == Placement::GPU_TENSOR_SPLIT) {
                // Better than CPU but worse than fitting on a single GPU
                // (cross-GPU traffic per layer). Use 7 * pw so it's preferred
                // over OFFLOAD_PARAMS only when the latter would not fit.
                s += 7 * pw;
                for (size_t g = 0; g < devices.size(); g++) gpus_used.insert(int(g));
            } else {
                s -= 10 * pw;
            }
        }
        s += 2 * int64_t(gpus_used.size());  // mild spread bonus
        return s;
    };

    std::vector<size_t>   idx(nC, 0);
    std::vector<Placement> best_pl;
    std::vector<int>       best_dev;
    int64_t                best_score = std::numeric_limits<int64_t>::min();
    bool                   found_any  = false;

    // Iterate the cartesian product of options.
    while (true) {
        std::vector<Placement> pl(nC);
        std::vector<int>       dev(nC);
        for (size_t i = 0; i < nC; i++) {
            pl[i]  = options[i][idx[i]].placement;
            dev[i] = options[i][idx[i]].device_idx;
        }
        // Feasibility check: peak on each GPU vs cap.
        bool feasible = true;
        for (size_t g = 0; g < nG; g++) {
            if (gpu_peak(int(g), pl, dev, components, devices) > cap[g]) {
                feasible = false;
                break;
            }
        }
        if (feasible) {
            int64_t sc = score(pl, dev);
            if (sc > best_score) {
                best_score = sc;
                best_pl    = pl;
                best_dev   = dev;
                found_any  = true;
            }
        }

        // Advance mixed-radix counter.
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
        // Degenerate: no feasible solution (even all-CPU must be feasible by
        // construction; but guard anyway). Fall back to CPU for everything.
        best_pl.assign(nC, Placement::CPU);
        best_dev.assign(nC, -1);
    }

    std::vector<double> split_ratios;
    for (size_t i = 0; i < nC; i++) {
        if (best_pl[i] == Placement::GPU_TENSOR_SPLIT) {
            split_ratios = tensor_split_ratios(devices);
            break;
        }
    }

    for (size_t i = 0; i < nC; i++) {
        const Component& c = components[i];
        Decision         d;
        d.kind      = c.kind;
        d.name      = c.name;
        d.placement = best_pl[i];
        if (best_pl[i] == Placement::CPU) {
            d.device_id     = DEVICE_ID_CPU;
            d.on_host_bytes = c.params_bytes + c.compute_bytes;
            plan.any_changes = true;
        } else if (best_pl[i] == Placement::GPU_TENSOR_SPLIT) {
            // device_id == DEVICE_ID_CPU is a sentinel meaning "all GPUs"
            // for split decisions. on_device_bytes records the largest
            // per-GPU share (peak). on_host_bytes stays 0.
            d.device_id           = DEVICE_ID_CPU;
            int64_t max_share     = 0;
            for (size_t g = 0; g < nG; g++) {
                int64_t share = int64_t(double(c.params_bytes + c.compute_bytes) *
                                         split_ratios[g]);
                max_share = std::max(max_share, share);
            }
            d.on_device_bytes = max_share;
            plan.any_changes  = true;
        } else {
            d.device_id = devices[best_dev[i]].id;
            if (best_pl[i] == Placement::GPU) {
                d.on_device_bytes = c.params_bytes + c.compute_bytes;
            } else {  // GPU_OFFLOAD_PARAMS
                d.on_device_bytes = c.params_bytes + c.compute_bytes;  // peak during its compute
                d.on_host_bytes   = c.params_bytes;
                plan.any_changes  = true;
            }
        }
        plan.decisions.push_back(d);
        plan.host_bytes += d.on_host_bytes;
    }

    // Report per-device peak using the same model as feasibility check.
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
        case Placement::GPU_TENSOR_SPLIT:   return "GPU(tensor-split)";
    }
    return "?";
}

inline void print_plan(const Plan&                   plan,
                       const std::vector<Component>& components,
                       const std::vector<Device>&    devices,
                       int64_t                       margin_bytes) {
    LOG_INFO("auto-fit plan (margin=%lld MiB per GPU):",
             (long long)(margin_bytes / MiB));
    LOG_INFO("  available devices:");
    if (devices.empty()) {
        LOG_INFO("    (no GPU devices detected — all components will run on CPU)");
    }
    for (const Device& d : devices) {
        LOG_INFO("    %-8s %-32s free %6lld / %6lld MiB",
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
        } else if (d.placement == Placement::GPU_TENSOR_SPLIT) {
            LOG_INFO("    %-12s -> tensor-split       (VRAM peak %lld MiB on largest-share GPU)",
                     d.name.c_str(),
                     (long long)(d.on_device_bytes / MiB));
        } else {
            LOG_INFO("    %-12s -> GPU %d (params RAM) (VRAM %lld MiB, RAM %lld MiB)",
                     d.name.c_str(), d.device_id,
                     (long long)(d.on_device_bytes / MiB),
                     (long long)(d.on_host_bytes / MiB));
        }
    }
    LOG_INFO("  projected per-device peak (MAX of assigned components, "
             "since free_params_immediately lets components time-share VRAM):");
    for (const Device& d : devices) {
        int64_t peak = 0;
        auto it = plan.device_bytes.find(d.id);
        if (it != plan.device_bytes.end()) peak = it->second;
        const int64_t remaining = d.free_bytes - peak;
        LOG_INFO("    %-8s peak %6lld / %6lld MiB free  (remaining %lld MiB)",
                 d.name.c_str(),
                 (long long)(peak / MiB),
                 (long long)(d.free_bytes / MiB),
                 (long long)(remaining / MiB));
    }
    LOG_INFO("    %-8s host RAM additional %lld MiB", "CPU",
             (long long)(plan.host_bytes / MiB));
}

// Convenience: look up the decision for a specific component.
inline const Decision* find_decision(const Plan& plan, ComponentKind kind) {
    for (const Decision& d : plan.decisions) {
        if (d.kind == kind) return &d;
    }
    return nullptr;
}

}  // namespace backend_fit

#endif  // __SD_BACKEND_FIT_HPP__
