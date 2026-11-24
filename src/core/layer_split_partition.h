#ifndef __SD_CORE_LAYER_SPLIT_PARTITION_H__
#define __SD_CORE_LAYER_SPLIT_PARTITION_H__

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"

#include "core/ggml_graph_cut.h"

namespace sd {

    struct GraphCutLayerSplitAssignment {
        std::vector<std::vector<ggml_tensor*>> tensors_by_backend;
        std::vector<int64_t> bytes_by_backend;
        std::vector<size_t> first_segment_by_backend;
        std::vector<size_t> last_segment_by_backend;
        std::unordered_map<const ggml_tensor*, ggml_backend_t> node_assignments;
        size_t segment_count          = 0;
        bool has_new_param_assignment = false;
    };

    // How the layer split fills the devices of a multi-device module (--layer-split-fill). PROPORTIONAL sizes the
    // ranges in proportion to the device capacities, so every device keeps the same share of room for computing;
    // FILL_FIRST fills each device up to its capacity before it moves to the next one, so the first device takes as
    // much as it can. AUTO lets auto-fit pick, for a split it plans, the fill that keeps more modules on the GPUs
    // (proportional on a tie); a split assigned by hand is PROPORTIONAL across discrete GPUs, and FILL_FIRST as soon
    // as an integrated GPU, the CPU or another kind of device takes part, so that the fastest device carries the most.
    enum class LayerSplitFill {
        AUTO,
        PROPORTIONAL,
        FILL_FIRST,
    };

    // Accepts auto, proportional and fill-first in any case; an empty string means auto.
    bool parse_layer_split_fill(const std::string& text, LayerSplitFill& fill);
    const char* layer_split_fill_name(LayerSplitFill fill);
    // The choice of the process: a context sets it from its options, AUTO until then.
    void set_layer_split_fill(LayerSplitFill fill);
    LayerSplitFill layer_split_fill();
    // AUTO resolved for the devices of one split, given in split order; any other choice is returned as it is.
    LayerSplitFill resolve_layer_split_fill(LayerSplitFill fill, const std::vector<enum ggml_backend_dev_type>& device_types);
    // How far each device is filled before the split moves on to the next one; the capacity stays the hard limit.
    std::vector<int64_t> layer_split_fill_limits(LayerSplitFill fill,
                                                 const std::vector<int64_t>& segment_bytes,
                                                 const std::vector<int64_t>& capacities);
    // Fills the devices in order, each up to its limit, as the layer split does, and tells what each device takes.
    // False when a segment fits on no device.
    bool layer_split_place(const std::vector<int64_t>& segment_bytes,
                           const std::vector<int64_t>& limits,
                           std::vector<int64_t>& bytes_by_device);

    std::string layer_split_backend_device_display_name(ggml_backend_t backend);
    int layer_split_tensor_block_index(const std::string& name);
    bool partition_graph_cut_layer_split(const char* desc,
                                         ggml_cgraph* gf,
                                         const sd::ggml_graph_cut::Plan& plan,
                                         const std::vector<ggml_backend_t>& split_backends,
                                         const std::vector<size_t>& backend_vram_limits,
                                         size_t primary_backend_vram_limit,
                                         std::unordered_map<const ggml_tensor*, ggml_backend_t>& param_assignments,
                                         const std::function<ggml_tensor*(ggml_tensor*)>& canonical_param_tensor,
                                         GraphCutLayerSplitAssignment* assignment_out);
    void log_graph_cut_layer_split_assignment(const char* desc,
                                              const std::vector<ggml_backend_t>& split_backends,
                                              const GraphCutLayerSplitAssignment& assignment);

}  // namespace sd

#endif  // __SD_CORE_LAYER_SPLIT_PARTITION_H__
