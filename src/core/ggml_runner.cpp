#include <algorithm>
#include <map>
#include <utility>

#include "core/ggml_extend.hpp"
#include "core/segment_graph_bindings.h"
#include "core/segment_weight_pipeline.h"

using namespace sd;

static size_t add_bytes(size_t a, size_t b) {
    return b > SIZE_MAX - a ? SIZE_MAX : a + b;
}

ComputeWorkspace::Measurement GGMLRunner::measure(ggml_cgraph* graph, size_t direct_bytes) {
    auto external_backend = [&](const ggml_tensor* tensor) -> ggml_backend_t {
        if (!params_tensor_set_.count(tensor)) {
            return nullptr;
        }
        auto placement = graph_cut_layer_split_assignments_.find(tensor);
        return placement == graph_cut_layer_split_assignments_.end() ? runtime_backend : placement->second;
    };
    auto assign_nodes = [&](ggml_backend_sched_t scheduler, ggml_cgraph* copy) {
        pin_multi_device_nodes(scheduler, copy, graph);
    };
    return workspace_.measure(graph, direct_bytes, external_backend, assign_nodes);
}

std::vector<DeviceMemoryRequest> GGMLRunner::memory_requests(
    const std::vector<BackendBufferSize>& sizes,
    size_t pending_cache_bytes) const {
    std::vector<DeviceMemoryRequest> requests;
    for (const auto& size : sizes) {
        const size_t retained    = retained_runtime_buffer_bytes(size.backend);
        const size_t reusable    = workspace_.bytes(size.backend);
        const size_t cache_bytes = size.backend == runtime_backend ? pending_cache_bytes : 0;
        const size_t pending     = add_bytes(size.bytes > reusable ? size.bytes - reusable : 0, cache_bytes);
        size_t limit             = max_graph_vram_bytes;
        if (is_multi_device()) {
            size_t index = 0;
            if (size.backend != runtime_backend) {
                auto position = std::find(extra_runtime_backends.begin(), extra_runtime_backends.end(), size.backend);
                index         = static_cast<size_t>(position - extra_runtime_backends.begin()) + 1;
            }
            if (index < graph_cut_layer_split_backend_vram_limits_.size()) {
                limit = graph_cut_layer_split_backend_vram_limits_[index];
            }
        }
        requests.push_back({size.backend, reinterpret_cast<uintptr_t>(this), pending,
                            retained, limit});
    }
    return requests;
}

bool GGMLRunner::fits(const std::vector<DeviceMemoryRequest>& requests,
                      const std::vector<ggml_tensor*>& params) const {
    auto manager = residency_manager.lock();
    if (manager == nullptr) {
        return params.empty();
    }
    for (const auto& request : requests) {
        if (!manager->fits_compute_backend_capacity(request, params)) {
            return false;
        }
    }
    return true;
}

bool GGMLRunner::execute_segment(ggml_cgraph* graph, int n_threads) {
    if (sd_backend_is_cpu(runtime_backend)) {
        sd_backend_cpu_set_n_threads(runtime_backend, n_threads);
    }
    if (workspace_.cpu_backend() != nullptr) {
        sd_backend_cpu_set_n_threads(workspace_.cpu_backend(), n_threads);
    }
    auto scheduler = workspace_.scheduler();
    ggml_status status;
    if (scheduler != nullptr) {
        if (sd_get_backend_eval_callback() != nullptr && !multi_device_eval_callback_warned) {
            LOG_WARN("%s: eval callback is not supported with the backend scheduler; ignoring", get_desc().c_str());
            multi_device_eval_callback_warned = true;
        }
        status = ggml_backend_sched_graph_compute(scheduler, graph);
    } else {
        status = sd_backend_graph_compute_with_eval_callback(runtime_backend, graph,
                                                             sd_get_backend_eval_callback(),
                                                             sd_get_backend_eval_callback_data());
    }
    workspace_.synchronize();
    if (status != GGML_STATUS_SUCCESS) {
        LOG_ERROR("%s compute failed: %s", get_desc().c_str(), ggml_status_to_string(status));
        return false;
    }
    const std::string description = get_desc();
    if (!debug_tensors.empty()) {
        std::unordered_set<const ggml_tensor*> graph_tensors;
        const int leaf_count = ggml_graph_cut::leaf_count(graph);
        const int node_count = ggml_graph_n_nodes(graph);
        graph_tensors.reserve(static_cast<size_t>(leaf_count + node_count));
        for (int index = 0; index < leaf_count; ++index) {
            graph_tensors.insert(ggml_graph_cut::leaf_tensor(graph, index));
        }
        for (int index = 0; index < node_count; ++index) {
            graph_tensors.insert(ggml_graph_node(graph, index));
        }

        for (const auto& entry : debug_tensors) {
            ggml_tensor* tensor = entry.first;
            if (tensor == nullptr || graph_tensors.find(tensor) == graph_tensors.end()) {
                continue;
            }
            ggml_backend_buffer_t buffer =
                tensor->view_src != nullptr ? tensor->view_src->buffer : tensor->buffer;
            if (buffer == nullptr) {
                LOG_WARN("%s skip debug tensor '%s': tensor buffer not set",
                         description.c_str(),
                         entry.second.c_str());
                continue;
            }
            if (tensor->type != GGML_TYPE_F32) {
                LOG_WARN("%s skip debug tensor '%s': only GGML_TYPE_F32 is supported, got %s",
                         description.c_str(),
                         entry.second.c_str(),
                         ggml_type_name(tensor->type));
                continue;
            }
            auto debug_tensor = make_sd_tensor_from_ggml<float>(tensor);
            print_sd_tensor(debug_tensor, false, entry.second.c_str());
        }
    }

    return true;
}

std::optional<Tensor<float>> GGMLRunner::execute_graph(ggml_cgraph* graph, int n_threads, bool no_return, const std::function<bool()>& read_outputs) {
    if (!assign_graph_cut_layer_split_backends(graph)) {
        return std::nullopt;
    }
    const auto params = collect_used_param_tensors(graph);
    ggml_graph_cut::Plan plan;
    if (!resolve_graph_cut_plan(graph, &plan)) {
        return std::nullopt;
    }
    const auto full_measurement = measure(graph, plan.compute_buffer_size);
    if (full_measurement.buffers.empty()) {
        return std::nullopt;
    }
    auto manager         = residency_manager.lock();
    const bool segmented = !is_multi_device() && !sd_backend_is_cpu(runtime_backend) &&
                           manager != nullptr && manager->segmented_compute_enabled() &&
                           plan.valid && plan.has_cuts && plan.segments.size() > 1 &&
                           !fits(memory_requests(full_measurement.buffers, cache_.pending_bytes(graph)), params);
    if (!segmented) {
        ggml_graph_cut::Segment segment;
        segment.group_name          = "graph";
        segment.compute_buffer_size = plan.compute_buffer_size;
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            segment.internal_node_indices.push_back(i);
        }
        for (int i = 0; i < ggml_graph_cut::leaf_count(graph); ++i) {
            auto tensor = ggml_graph_cut::leaf_tensor(graph, i);
            ggml_graph_cut::Segment::InputRef input;
            input.leaf_index = i;
            input.type       = canonical_param_tensor(tensor) != nullptr
                                   ? ggml_graph_cut::Segment::INPUT_PARAM
                                   : ggml_graph_cut::Segment::INPUT_EXTERNAL;
            segment.input_refs.push_back(input);
        }
        plan.segments = {std::move(segment)};
    }
    const bool segments_changed = plan.segments.size() != logged_segment_count_;
    if (segments_changed && (segmented || logged_segment_count_ > 1)) {
        LOG_DEBUG("%s using %zu segment%s", get_desc().c_str(),
                  plan.segments.size(), plan.segments.size() == 1 ? "" : "s");
    }
    SegmentGraphBindings bindings(cut_cache_, plan, graph);
    SegmentWeightPipeline weights(manager, runtime_backend, reinterpret_cast<uintptr_t>(this),
                                  graph, plan, params_tensor_set_,
                                  segmented && manager != nullptr && manager->prefetch_enabled());

    std::map<ggml_backend_t, size_t> peak_compute_bytes;
    auto track_compute_buffer = [&](ggml_backend_t backend) {
        if (backend != nullptr) {
            auto& peak = peak_compute_bytes[backend];
            peak       = std::max(peak, workspace_.bytes(backend));
        }
    };
    std::optional<Tensor<float>> output = Tensor<float>();
    for (size_t index = 0; index < plan.segments.size(); ++index) {
        const auto& segment = plan.segments[index];
        const bool last     = index + 1 == plan.segments.size();
        auto fail_segment   = [&](const char* phase) {
            LOG_ERROR("%s segment %zu/%zu (%s) failed during %s", get_desc().c_str(),
                        index + 1, plan.segments.size(), segment.group_name.c_str(), phase);
            return std::nullopt;
        };
        cut_cache_.prune(segment.live_cut_names);
        bindings.reset(segment);
        if (!bindings.bind_cached_inputs(segment, get_desc().c_str())) {
            return fail_segment("input binding");
        }
        ggml_context* segment_context = nullptr;
        auto segment_graph            = segmented
                                            ? ggml_graph_cut::build_segment_graph(graph, segment, &segment_context)
                                            : graph;
        struct SegmentCleanup {
            GGMLRunner& runner;
            SegmentWeightPipeline& weights;
            SegmentGraphBindings& bindings;
            ggml_context* context;
            ~SegmentCleanup() {
                runner.workspace_.segment_end();
                bindings.restore();
                weights.segment_end();
                ggml_free(context);
                runner.sync_runtime_residency();
            }
        } segment_cleanup{*this, weights, bindings, segment_context};

        auto measurement = segmented ? measure(segment_graph, segment.compute_buffer_size) : full_measurement;
        if (!workspace_.prepare(measurement)) {
            return fail_segment("workspace preparation");
        }
        const size_t cut_bytes       = last ? 0 : cut_cache_.estimate_output_bytes(graph, segment);
        const size_t new_cache_bytes = add_bytes(cut_bytes, cache_.pending_bytes(segment_graph));
        auto ensure_capacity         = [&]() {
            sync_runtime_residency();
            auto requests = memory_requests(measurement.buffers, new_cache_bytes);
            if (!fits(requests, weights.params(index)) && workspace_.release_excess(measurement)) {
                sync_runtime_residency();
                requests = memory_requests(measurement.buffers, new_cache_bytes);
            }
            return weights.ensure_segment_capacity(index, requests);
        };
        if (!weights.segment_start(index, ensure_capacity)) {
            return fail_segment("weight preparation");
        }
        // Preparing weights can execute LoRA graphs and reclaim an idle workspace.
        if (!workspace_.measurement_matches(segment_graph, measurement)) {
            measurement = measure(segment_graph, segment.compute_buffer_size);
        }
        if (!workspace_.prepare(measurement) || !ensure_capacity()) {
            return fail_segment("workspace capacity check");
        }
        if (!workspace_.allocate(segment_graph, [&](ggml_backend_sched_t scheduler, ggml_cgraph* current) {
                pin_multi_device_nodes(scheduler, current);
            })) {
            return fail_segment("workspace allocation");
        }
        for (const auto& size : measurement.buffers) {
            track_compute_buffer(size.backend);
        }
        if (workspace_.scheduler() != nullptr) {
            track_compute_buffer(workspace_.cpu_backend());
        }
        if (!ensure_capacity()) {
            return fail_segment("allocated capacity check");
        }
        copy_data_to_backend_tensor(segment_graph, false);
        auto prefetch_requests = memory_requests(measurement.buffers, new_cache_bytes);
        if (!prefetch_requests.empty()) {
            weights.enqueue_next(index, prefetch_requests.front());
        }
        if (!execute_segment(segment_graph, n_threads) ||
            !cache_.capture(segment_graph) ||
            !cut_cache_.capture(graph, segment, get_desc().c_str())) {
            return fail_segment("execution or output caching");
        }
        sync_runtime_residency();
        if (last) {
            if (read_outputs && !read_outputs()) {
                return fail_segment("output finalization");
            }
            if (!no_return) {
                auto result = ggml_get_tensor(compute_ctx, final_result_name.c_str());
                output      = read_graph_tensor<float>(result, "output");
                if (!output.has_value()) {
                    return fail_segment("output readback");
                }
            }
        }
        // Final outputs and their callbacks may still be views of consumed cuts.
        cut_cache_.prune(segment.future_cut_names);
    }
    if (segments_changed || peak_compute_bytes != logged_compute_bytes_) {
        for (const auto& entry : peak_compute_bytes) {
            LOG_DEBUG("%s compute buffer size: %.2f MB(%s) on %s (peak across %zu segment%s)",
                      get_desc().c_str(), entry.second / (1024.0 * 1024.0),
                      sd_backend_is_cpu(entry.first) ? "RAM" : "VRAM", ggml_backend_name(entry.first),
                      plan.segments.size(), plan.segments.size() == 1 ? "" : "s");
        }
        logged_compute_bytes_ = std::move(peak_compute_bytes);
        logged_segment_count_ = plan.segments.size();
    }
    return output;
}
