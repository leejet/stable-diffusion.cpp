#include "core/layer_split_partition.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <unordered_set>
#include <utility>

#include "core/device_health.h"
#include "core/util.h"

namespace sd {

    static bool layer_split_path_segment_starts_at(const std::string& name, size_t pos) {
        return pos == 0 || name[pos - 1] == '.';
    }

    static bool layer_split_has_path_segment(const std::string& name, const char* segment) {
        size_t pos = name.find(segment);
        while (pos != std::string::npos) {
            if (layer_split_path_segment_starts_at(name, pos)) {
                return true;
            }
            pos = name.find(segment, pos + 1);
        }
        return false;
    }

    int layer_split_tensor_block_index(const std::string& name) {
        static const char* unet_block_segments[] = {"input_blocks.", "output_blocks.", "middle_block.",
                                                    "down_blocks.", "up_blocks.", "mid_block."};
        for (const char* segment : unet_block_segments) {
            if (layer_split_has_path_segment(name, segment)) {
                return -1;
            }
        }

        static const char* block_keywords[] = {"transformer_blocks.", "joint_blocks.", "double_blocks.",
                                               "single_blocks.", "blocks.", "block.", "layers."};
        for (const char* keyword : block_keywords) {
            size_t pos = name.find(keyword);
            while (pos != std::string::npos) {
                if (!layer_split_path_segment_starts_at(name, pos)) {
                    pos = name.find(keyword, pos + 1);
                    continue;
                }
                pos += std::strlen(keyword);
                size_t end = pos;
                while (end < name.size() && name[end] >= '0' && name[end] <= '9') {
                    end++;
                }
                if (end > pos && (end == name.size() || name[end] == '.')) {
                    return std::atoi(name.substr(pos, end - pos).c_str());
                }
                break;
            }
        }
        return -1;
    }

    std::string layer_split_backend_device_display_name(ggml_backend_t backend) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        const char* name       = dev != nullptr ? ggml_backend_dev_name(dev) : ggml_backend_name(backend);
        return name != nullptr ? name : "unknown";
    }

    static size_t graph_cut_layer_split_backend_vram_limit(const std::vector<size_t>& backend_vram_limits,
                                                           size_t backend_index,
                                                           size_t primary_backend_vram_limit) {
        if (backend_index < backend_vram_limits.size()) {
            return backend_vram_limits[backend_index];
        }
        return backend_index == 0 ? primary_backend_vram_limit : 0;
    }

    static std::vector<int64_t> graph_cut_layer_split_backend_capacities(const std::vector<ggml_backend_t>& backends,
                                                                         const std::vector<size_t>& backend_vram_limits,
                                                                         size_t primary_backend_vram_limit) {
        std::vector<int64_t> capacities(backends.size(), layer_split_unknown_capacity);
        constexpr int64_t compute_headroom_bytes = 2ll * 1024 * 1024 * 1024;
        for (size_t i = 0; i < backends.size(); i++) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backends[i]);
            size_t free_bytes = 0, total_bytes = 0;
            if (dev != nullptr) {
                ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
            }
            if (free_bytes > 0) {
                capacities[i] = std::max<int64_t>((int64_t)free_bytes - compute_headroom_bytes, 0);
            }
            size_t limit_bytes = graph_cut_layer_split_backend_vram_limit(backend_vram_limits,
                                                                          i,
                                                                          primary_backend_vram_limit);
            if (limit_bytes > 0) {
                capacities[i] = std::min<int64_t>(capacities[i], (int64_t)limit_bytes);
            }
        }
        return capacities;
    }

    // Whether filling the devices in order, each up to `fraction` of its capacity, places every
    // segment. Mirrors the placement loop in partition_graph_cut_layer_split().
    static bool graph_cut_layer_split_fits(const std::vector<int64_t>& segment_bytes,
                                           const std::vector<int64_t>& capacities,
                                           double fraction) {
        size_t backend = 0;
        int64_t used   = 0;
        for (int64_t bytes : segment_bytes) {
            while (backend + 1 < capacities.size() && bytes > 0 &&
                   used + bytes > (int64_t)((double)capacities[backend] * fraction)) {
                backend++;
                used = 0;
            }
            if (bytes > 0 && used + bytes > (int64_t)((double)capacities[backend] * fraction)) {
                return false;
            }
            used += bytes;
        }
        return true;
    }

    // The proportional limits: the same fraction of every capacity, as small as possible while all segments still
    // fit. Filling the first device up to its capacity would leave only the fixed headroom for compute, which is too
    // small for large video graphs. A fraction of 1 is plain capacity-based placement, so every layout that fits the
    // capacities is still found.
    static std::vector<int64_t> graph_cut_layer_split_proportional_limits(const std::vector<int64_t>& segment_bytes,
                                                                          const std::vector<int64_t>& capacities) {
        const bool capacities_known = std::all_of(capacities.begin(), capacities.end(), [](int64_t capacity) {
            return capacity < layer_split_unknown_capacity;
        });
        if (!capacities_known || !graph_cut_layer_split_fits(segment_bytes, capacities, 1.0)) {
            return capacities;
        }
        double low  = 0.0;
        double high = 1.0;
        for (int i = 0; i < 32; i++) {
            const double mid = (low + high) / 2;
            if (graph_cut_layer_split_fits(segment_bytes, capacities, mid)) {
                high = mid;
            } else {
                low = mid;
            }
        }
        std::vector<int64_t> limits(capacities.size());
        for (size_t i = 0; i < capacities.size(); i++) {
            limits[i] = (int64_t)((double)capacities[i] * high);
        }
        return limits;
    }

    bool parse_layer_split_fill(const std::string& text, LayerSplitFill& fill) {
        std::string value = text;
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (value.empty() || value == "auto") {
            fill = LayerSplitFill::AUTO;
        } else if (value == "proportional") {
            fill = LayerSplitFill::PROPORTIONAL;
        } else if (value == "fill-first") {
            fill = LayerSplitFill::FILL_FIRST;
        } else {
            return false;
        }
        return true;
    }

    const char* layer_split_fill_name(LayerSplitFill fill) {
        switch (fill) {
            case LayerSplitFill::PROPORTIONAL:
                return "proportional";
            case LayerSplitFill::FILL_FIRST:
                return "fill-first";
            default:
                return "auto";
        }
    }

    static std::atomic<int> process_layer_split_fill{static_cast<int>(LayerSplitFill::AUTO)};

    void set_layer_split_fill(LayerSplitFill fill) {
        process_layer_split_fill.store(static_cast<int>(fill));
    }

    LayerSplitFill layer_split_fill() {
        return static_cast<LayerSplitFill>(process_layer_split_fill.load());
    }

    LayerSplitFill resolve_layer_split_fill(LayerSplitFill fill, const std::vector<enum ggml_backend_dev_type>& device_types) {
        if (fill != LayerSplitFill::AUTO) {
            return fill;
        }
        const bool discrete_gpus_only = std::all_of(device_types.begin(), device_types.end(), [](enum ggml_backend_dev_type type) {
            return type == GGML_BACKEND_DEVICE_TYPE_GPU;
        });
        return discrete_gpus_only ? LayerSplitFill::PROPORTIONAL : LayerSplitFill::FILL_FIRST;
    }

    std::vector<int64_t> layer_split_fill_limits(LayerSplitFill fill,
                                                 const std::vector<int64_t>& segment_bytes,
                                                 const std::vector<int64_t>& capacities) {
        if (fill == LayerSplitFill::FILL_FIRST) {
            return capacities;
        }
        return graph_cut_layer_split_proportional_limits(segment_bytes, capacities);
    }

    std::vector<int64_t> layer_split_apply_shares(const std::vector<int64_t>& capacities, const std::vector<double>& shares) {
        std::vector<int64_t> result = capacities;
        for (size_t i = 0; i < result.size() && i < shares.size(); i++) {
            const double share = std::min(std::max(shares[i], 0.0), 1.0);
            if (share <= 0.0) {
                result[i] = 0;
            } else if (result[i] < layer_split_unknown_capacity) {
                result[i] = (int64_t)((double)result[i] * share);
            }
        }
        return result;
    }

    bool layer_split_place(const std::vector<int64_t>& segment_bytes,
                           const std::vector<int64_t>& limits,
                           std::vector<int64_t>& bytes_by_device) {
        bytes_by_device.assign(limits.size(), 0);
        if (limits.empty()) {
            return segment_bytes.empty();
        }
        size_t device = 0;
        int64_t used  = 0;
        for (int64_t bytes : segment_bytes) {
            while (device + 1 < limits.size() && bytes > 0 && used + bytes > limits[device]) {
                device++;
                used = 0;
            }
            if (bytes > 0 && used + bytes > limits[device]) {
                return false;
            }
            used += bytes;
            bytes_by_device[device] += bytes;
        }
        return true;
    }

    bool partition_graph_cut_layer_split(const char* desc,
                                         ggml_cgraph* gf,
                                         const sd::ggml_graph_cut::Plan& plan,
                                         const std::vector<ggml_backend_t>& split_backends,
                                         const std::vector<size_t>& backend_vram_limits,
                                         size_t primary_backend_vram_limit,
                                         std::unordered_map<const ggml_tensor*, ggml_backend_t>& param_assignments,
                                         const std::function<ggml_tensor*(ggml_tensor*)>& canonical_param_tensor,
                                         GraphCutLayerSplitAssignment* assignment_out) {
        GGML_ASSERT(gf != nullptr);
        GGML_ASSERT(assignment_out != nullptr);
        GGML_ASSERT(canonical_param_tensor != nullptr);
        GGML_ASSERT(!split_backends.empty());

        GraphCutLayerSplitAssignment assignment;
        assignment.segment_count = plan.segments.size();
        assignment.tensors_by_backend.resize(split_backends.size());
        assignment.bytes_by_backend.resize(split_backends.size(), 0);
        assignment.first_segment_by_backend.resize(split_backends.size(), plan.segments.size());
        assignment.last_segment_by_backend.resize(split_backends.size(), 0);

        std::vector<std::vector<ggml_tensor*>> segment_params(plan.segments.size());
        std::vector<int64_t> segment_param_bytes(plan.segments.size(), 0);
        std::unordered_set<ggml_tensor*> seen_params;
        for (size_t seg_idx = 0; seg_idx < plan.segments.size(); seg_idx++) {
            std::vector<ggml_tensor*> params = sd::ggml_graph_cut::param_tensors(gf, plan.segments[seg_idx]);
            for (ggml_tensor* raw_param : params) {
                ggml_tensor* param = canonical_param_tensor(raw_param);
                if (param == nullptr || !seen_params.insert(param).second) {
                    continue;
                }
                segment_params[seg_idx].push_back(param);
                segment_param_bytes[seg_idx] += (int64_t)ggml_nbytes(param);
            }
        }

        int64_t total_param_bytes = 0;
        for (int64_t bytes : segment_param_bytes) {
            total_param_bytes += bytes;
        }
        if (total_param_bytes <= 0) {
            LOG_ERROR("%s graph-cut layer split found no graph params to assign", desc);
            return false;
        }

        std::vector<int64_t> backend_capacities = graph_cut_layer_split_backend_capacities(split_backends,
                                                                                           backend_vram_limits,
                                                                                           primary_backend_vram_limit);
        // Existing placements may already occupy the reported free VRAM. Reuse
        // them; execution checks missing weights and reclaims memory as needed.
        const bool reuse_assignments = std::all_of(seen_params.begin(), seen_params.end(), [&](ggml_tensor* param) {
            return param_assignments.count(param) != 0;
        });

        std::vector<enum ggml_backend_dev_type> device_types;
        for (ggml_backend_t backend : split_backends) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend);
            device_types.push_back(dev != nullptr ? ggml_backend_dev_type(dev) : GGML_BACKEND_DEVICE_TYPE_CPU);
        }
        const LayerSplitFill fill = resolve_layer_split_fill(layer_split_fill(), device_types);
        if (!reuse_assignments) {
            LOG_INFO("%s layer split fill: %s", desc, layer_split_fill_name(fill));
            std::vector<double> shares;
            for (ggml_backend_t backend : split_backends) {
                shares.push_back(device_health().share(device_health_name(backend)));
                if (shares.back() < 1.0) {
                    LOG_INFO("%s layer split: %s offers %.0f%% of its capacity (device health)",
                             desc, layer_split_backend_device_display_name(backend).c_str(), shares.back() * 100.0);
                }
            }
            backend_capacities = layer_split_apply_shares(backend_capacities, shares);
        }
        // Move to the next device at the fill limit; the capacity stays the hard limit.
        const std::vector<int64_t> backend_limits = reuse_assignments
                                                        ? backend_capacities
                                                        : layer_split_fill_limits(fill, segment_param_bytes, backend_capacities);

        std::vector<ggml_backend_t> backend_by_segment(plan.segments.size(), split_backends[0]);
        size_t current_backend = 0;
        int64_t current_used   = 0;
        for (size_t seg_idx = 0; seg_idx < plan.segments.size(); seg_idx++) {
            int64_t bytes = segment_param_bytes[seg_idx];
            while (!reuse_assignments && current_backend + 1 < split_backends.size() &&
                   bytes > 0 &&
                   current_used + bytes > backend_limits[current_backend]) {
                current_backend++;
                current_used = 0;
            }
            if (!reuse_assignments && bytes > 0 && current_used + bytes > backend_capacities[current_backend]) {
                LOG_ERROR("%s graph-cut layer split: segment %zu needs %.1f MB on %s, but only %.1f MB is available under current VRAM limits",
                          desc,
                          seg_idx,
                          (current_used + bytes) / (1024.0 * 1024.0),
                          layer_split_backend_device_display_name(split_backends[current_backend]).c_str(),
                          backend_capacities[current_backend] / (1024.0 * 1024.0));
                return false;
            }
            current_used += bytes;

            for (ggml_tensor* param : segment_params[seg_idx]) {
                ggml_backend_t target_backend = split_backends[current_backend];
                auto assigned_it              = param_assignments.find(param);
                if (assigned_it == param_assignments.end()) {
                    param_assignments[param]            = target_backend;
                    assignment.has_new_param_assignment = true;
                } else {
                    target_backend = assigned_it->second;
                }

                auto backend_it = std::find(split_backends.begin(), split_backends.end(), target_backend);
                if (backend_it == split_backends.end()) {
                    LOG_ERROR("%s graph-cut layer split tensor '%s' is assigned to an unavailable backend",
                              desc,
                              ggml_get_name(param));
                    return false;
                }
                size_t backend_idx = (size_t)std::distance(split_backends.begin(), backend_it);
                if (reuse_assignments) {
                    current_backend = backend_idx;
                }
                assignment.first_segment_by_backend[backend_idx] = std::min(assignment.first_segment_by_backend[backend_idx], seg_idx);
                assignment.last_segment_by_backend[backend_idx]  = std::max(assignment.last_segment_by_backend[backend_idx], seg_idx + 1);
                assignment.tensors_by_backend[backend_idx].push_back(param);
                assignment.bytes_by_backend[backend_idx] += (int64_t)ggml_nbytes(param);
            }
            backend_by_segment[seg_idx] = split_backends[current_backend];
        }

        const int n_nodes = ggml_graph_n_nodes(gf);
        for (size_t seg_idx = 0; seg_idx < plan.segments.size(); seg_idx++) {
            ggml_backend_t backend = backend_by_segment[seg_idx];
            const auto& segment    = plan.segments[seg_idx];
            auto assign_node       = [&](int node_index) {
                if (node_index < 0 || node_index >= n_nodes) {
                    return;
                }
                ggml_tensor* node = ggml_graph_node(gf, node_index);
                if (node == nullptr) {
                    return;
                }
                if (assignment.node_assignments.find(node) == assignment.node_assignments.end()) {
                    assignment.node_assignments[node] = backend;
                }
            };
            for (int node_index : segment.internal_node_indices) {
                assign_node(node_index);
            }
            for (int node_index : segment.output_node_indices) {
                assign_node(node_index);
            }
        }

        *assignment_out = std::move(assignment);
        return true;
    }

    void log_graph_cut_layer_split_assignment(const char* desc,
                                              const std::vector<ggml_backend_t>& split_backends,
                                              const GraphCutLayerSplitAssignment& assignment) {
        for (size_t i = 0; i < split_backends.size(); i++) {
            if (i >= assignment.tensors_by_backend.size() ||
                assignment.tensors_by_backend[i].empty()) {
                continue;
            }
            size_t first_segment = assignment.first_segment_by_backend[i] == assignment.segment_count
                                       ? 0
                                       : assignment.first_segment_by_backend[i];
            size_t last_segment  = assignment.last_segment_by_backend[i];
            if (assignment.has_new_param_assignment) {
                LOG_INFO("%s graph-cut layer split: %s <- segments [%zu, %zu), %zu tensors, %.1f MB",
                         desc,
                         layer_split_backend_device_display_name(split_backends[i]).c_str(),
                         first_segment,
                         last_segment,
                         assignment.tensors_by_backend[i].size(),
                         assignment.bytes_by_backend[i] / (1024.0 * 1024.0));
            } else {
                LOG_VERBOSE("%s graph-cut layer split: %s <- segments [%zu, %zu), %zu tensors, %.1f MB",
                            desc,
                            layer_split_backend_device_display_name(split_backends[i]).c_str(),
                            first_segment,
                            last_segment,
                            assignment.tensors_by_backend[i].size(),
                            assignment.bytes_by_backend[i] / (1024.0 * 1024.0));
            }
        }
    }

}  // namespace sd
