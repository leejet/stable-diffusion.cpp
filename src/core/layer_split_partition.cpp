#include "core/layer_split_partition.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "core/util.h"

namespace sd {

    // Parse a transformer block index out of a weight name, or -1 if the tensor
    // does not belong to a block ("model.diffusion_model.transformer_blocks.12.*"
    // -> 12, "text_encoders.llm.model.layers.30.*" -> 30).
    static int tensor_block_index(const std::string& name) {
        static const char* block_keywords[] = {"transformer_blocks.", "joint_blocks.", "double_blocks.",
                                               "single_blocks.", "blocks.", "block.", "layers."};
        for (const char* keyword : block_keywords) {
            size_t pos = name.find(keyword);
            if (pos == std::string::npos) {
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
        }
        return -1;
    }

    std::string layer_split_backend_device_display_name(ggml_backend_t backend) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        const char* name       = dev != nullptr ? ggml_backend_dev_name(dev) : ggml_backend_name(backend);
        return name != nullptr ? name : "unknown";
    }

    std::vector<std::map<std::string, ggml_tensor*>> partition_layer_split_tensors(
        const std::string& desc,
        const std::map<std::string, ggml_tensor*>& tensors,
        const std::map<std::string, ggml_tensor*>& split_tensors,
        const std::vector<ggml_backend_t>& backends) {
        std::vector<std::map<std::string, ggml_tensor*>> partitions(backends.size());

        std::map<int, int64_t> block_bytes;
        int64_t total_block_bytes = 0;
        int64_t other_bytes       = 0;
        int n_blocks              = 0;
        for (const auto& kv : tensors) {
            int64_t bytes = (int64_t)ggml_nbytes(kv.second);
            int idx       = split_tensors.count(kv.first) != 0 ? tensor_block_index(kv.first) : -1;
            if (idx >= 0) {
                block_bytes[idx] += bytes;
                total_block_bytes += bytes;
                n_blocks = std::max(n_blocks, idx + 1);
            } else {
                other_bytes += bytes;
            }
        }
        if (n_blocks == 0) {
            LOG_WARN("%s: no transformer blocks found for a layer split; keeping the module on %s",
                     desc.c_str(),
                     layer_split_backend_device_display_name(backends[0]).c_str());
            partitions[0] = tensors;
            return partitions;
        }

        // Weight each device by its free memory minus a fixed compute headroom:
        // every device participating in a layer split also hosts a share of the
        // scheduler's compute buffers (the activations of its block range), which
        // for large models runs into gigabytes; without the headroom the weight
        // share fills the device exactly and the compute allocation OOMs. The main
        // backend additionally holds the non-block tensors, so those are
        // subtracted from its block budget below.
        constexpr int64_t compute_headroom_bytes = 2ll * 1024 * 1024 * 1024;
        std::vector<double> device_weights(backends.size(), 1.0);
        double weight_sum = 0.0;
        for (size_t i = 0; i < backends.size(); i++) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backends[i]);
            size_t free_bytes = 0, total_bytes = 0;
            if (dev != nullptr) {
                ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
            }
            // Keep a small share even for tight devices instead of dropping them.
            int64_t usable_bytes = std::max<int64_t>((int64_t)free_bytes - compute_headroom_bytes,
                                                     (int64_t)free_bytes / 8);
            device_weights[i]    = usable_bytes > 0 ? (double)usable_bytes : 1.0;
            weight_sum += device_weights[i];
        }

        std::vector<int64_t> block_budgets(backends.size(), 0);
        for (size_t i = 0; i < backends.size(); i++) {
            int64_t budget = (int64_t)((double)(total_block_bytes + other_bytes) * device_weights[i] / weight_sum);
            if (i == 0) {
                budget = std::max<int64_t>(budget - other_bytes, 0);
            }
            block_budgets[i] = budget;
        }

        // Assign contiguous block ranges: boundaries[i] is the first block index
        // NOT owned by backend i. Every backend keeps at least one block while
        // blocks remain, and the last backend absorbs the remainder.
        std::vector<int> boundaries(backends.size(), n_blocks);
        size_t current = 0;
        int64_t used   = 0;
        for (int b = 0; b < n_blocks; b++) {
            int64_t bytes = block_bytes.count(b) != 0 ? block_bytes[b] : 0;
            if (current + 1 < backends.size() && used > 0 && used + bytes > block_budgets[current]) {
                boundaries[current] = b;
                current++;
                used = 0;
            }
            used += bytes;
        }

        for (const auto& kv : tensors) {
            size_t target = 0;
            int idx       = split_tensors.count(kv.first) != 0 ? tensor_block_index(kv.first) : -1;
            if (idx >= 0) {
                while (target < boundaries.size() && idx >= boundaries[target]) {
                    target++;
                }
                target = std::min(target, backends.size() - 1);
            }
            partitions[target][kv.first] = kv.second;
        }

        int range_start = 0;
        for (size_t i = 0; i < backends.size(); i++) {
            int range_end = boundaries[i];
            LOG_INFO("%s layer split: %s <- blocks [%d, %d)%s",
                     desc.c_str(),
                     layer_split_backend_device_display_name(backends[i]).c_str(),
                     range_start,
                     range_end,
                     i == 0 ? " + non-block tensors" : "");
            range_start = range_end;
        }
        return partitions;
    }

}  // namespace sd
