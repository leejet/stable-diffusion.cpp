#ifndef __CHUNK_GRAPH_HPP__
#define __CHUNK_GRAPH_HPP__

#include <array>
#include <functional>
#include <string>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include "util.h"

namespace LayerStreaming {

// Shared helper that compiles K consecutive transformer layers into a single
// ggml graph and dispatches them as one ggml_backend_graph_compute call,
// instead of one tiny graph per layer. Reusable across runners (z_image,
// flux, mmdit, anima, qwen_image, ...).
//
// Cached state (ggml_context, gallocr, cgraph) survives across compute() calls
// on the runner's main compute_ctx. Inputs are shape-bound, so the graph is
// rebuilt whenever shape / layer count changes (e.g. between two queue jobs
// with different prompt lengths).
class ChunkGraph {
public:
    using BuildFn = std::function<ggml_tensor*(
        ggml_context*                       ctx,
        const std::vector<ggml_tensor*>&    inputs,
        int                                 K)>;

    ChunkGraph() = default;
    ~ChunkGraph() { clear(); }
    ChunkGraph(const ChunkGraph&)            = delete;
    ChunkGraph& operator=(const ChunkGraph&) = delete;

    // Build (or keep cached) a graph for K layers with the given input shapes.
    // The cached graph is reused only if K and every input shape match the
    // last build; otherwise the old graph is freed and a fresh one is built.
    //
    // build_fn receives the freshly created input tensors (one per entry of
    // input_shapes, in the same order) and must wire them through K layers,
    // returning the output tensor. The output is automatically marked as a
    // graph output.
    //
    // Returns false on allocator / context failure; on success the graph is
    // ready to dispatch.
    bool ensure_built(ggml_backend_t                              backend,
                      int                                          K,
                      const std::vector<std::array<int64_t, 4>>&  input_shapes,
                      ggml_type                                    input_type,
                      BuildFn                                      build_fn,
                      size_t                                       graph_node_capacity,
                      const std::string&                           desc_tag) {
        if (gf_ != nullptr && layer_count_ == K && shapes_match(input_shapes)) {
            return true;
        }
        clear();

        // 16 MB headroom for op metadata is plenty for typical K (~30 layers).
        size_t ctx_size = 16 * 1024 * 1024;
        ctx_ = ggml_init({ctx_size, nullptr, true});
        if (ctx_ == nullptr) {
            LOG_ERROR("%s chunk_ctx alloc failed", desc_tag.c_str());
            return false;
        }

        gf_ = ggml_new_graph_custom(ctx_, graph_node_capacity, false);

        inputs_.clear();
        inputs_.reserve(input_shapes.size());
        for (const auto& shape : input_shapes) {
            ggml_tensor* t = ggml_new_tensor_4d(ctx_, input_type,
                                                 shape[0], shape[1], shape[2], shape[3]);
            ggml_set_input(t);
            inputs_.push_back(t);
        }

        out_ = build_fn(ctx_, inputs_, K);
        if (out_ == nullptr) {
            LOG_ERROR("%s chunk build_fn returned null", desc_tag.c_str());
            clear();
            return false;
        }
        ggml_set_output(out_);
        ggml_build_forward_expand(gf_, out_);

        allocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (allocr_ == nullptr) {
            LOG_ERROR("%s chunk gallocr_new failed", desc_tag.c_str());
            clear();
            return false;
        }
        if (!ggml_gallocr_reserve(allocr_, gf_)) {
            LOG_ERROR("%s chunk gallocr_reserve failed", desc_tag.c_str());
            clear();
            return false;
        }
        size_t buf_size = ggml_gallocr_get_buffer_size(allocr_, 0);
        LOG_INFO("%s chunk graph: %d layers, compute buffer = %.2f MB",
                 desc_tag.c_str(), K, buf_size / (1024.0 * 1024.0));

        layer_count_   = K;
        cached_shapes_ = input_shapes;
        return true;
    }

    // Allocate/upload-inputs/compute/read-output for one step. host_data and
    // host_nbytes must have one entry per input (matching the order passed to
    // ensure_built). out_buf must be sized for at least ggml_nbytes(out_).
    bool dispatch(ggml_backend_t                       backend,
                  const std::vector<const void*>&      host_data,
                  const std::vector<size_t>&           host_nbytes,
                  void*                                out_buf,
                  size_t                               out_nbytes) {
        if (gf_ == nullptr) {
            return false;
        }
        if (host_data.size() != inputs_.size() || host_nbytes.size() != inputs_.size()) {
            LOG_ERROR("chunk dispatch: host_data/host_nbytes size mismatch");
            return false;
        }
        if (!ggml_gallocr_alloc_graph(allocr_, gf_)) {
            LOG_ERROR("chunk alloc_graph failed");
            return false;
        }
        for (size_t i = 0; i < inputs_.size(); i++) {
            ggml_backend_tensor_set(inputs_[i], host_data[i], 0, host_nbytes[i]);
        }
        ggml_status status = ggml_backend_graph_compute(backend, gf_);
        if (status != GGML_STATUS_SUCCESS) {
            LOG_ERROR("chunk compute failed: %s", ggml_status_to_string(status));
            return false;
        }
        ggml_backend_tensor_get(out_, out_buf, 0, out_nbytes);
        return true;
    }

    ggml_tensor* output() const { return out_; }
    int          layer_count() const { return layer_count_; }
    bool         is_built() const { return gf_ != nullptr; }

    void clear() {
        if (allocr_ != nullptr) {
            ggml_gallocr_free(allocr_);
            allocr_ = nullptr;
        }
        if (ctx_ != nullptr) {
            ggml_free(ctx_);
            ctx_ = nullptr;
        }
        gf_          = nullptr;
        out_         = nullptr;
        inputs_.clear();
        cached_shapes_.clear();
        layer_count_ = 0;
    }

private:
    bool shapes_match(const std::vector<std::array<int64_t, 4>>& shapes) const {
        if (shapes.size() != cached_shapes_.size()) {
            return false;
        }
        for (size_t i = 0; i < shapes.size(); i++) {
            for (int j = 0; j < 4; j++) {
                if (shapes[i][j] != cached_shapes_[i][j]) {
                    return false;
                }
            }
        }
        return true;
    }

    ggml_context*                          ctx_           = nullptr;
    ggml_gallocr_t                         allocr_        = nullptr;
    ggml_cgraph*                           gf_            = nullptr;
    std::vector<ggml_tensor*>              inputs_;
    ggml_tensor*                           out_           = nullptr;
    int                                    layer_count_   = 0;
    std::vector<std::array<int64_t, 4>>    cached_shapes_;
};

}  // namespace LayerStreaming

#endif
