#ifndef __SD_MODEL_IO_QUANT_CONFIG_H__
#define __SD_MODEL_IO_QUANT_CONFIG_H__

#include <map>
#include <string>
#include <vector>

#include "tensor_storage.h"

// Claims weights that were quantized by an external toolchain and repacks their
// description so the rest of the loader sees an ordinary ggml-typed tensor.
//
// Two families are recognized:
//   * ComfyUI `_quantization_metadata` (formats `convrot_w4a4`, `int8_tensorwise`),
//     where a `<layer>.weight_scale` sidecar holds per-output-channel scales.
//   * bitsandbytes NF4, where `<layer>.weight.absmax` / `.quant_map` /
//     `.quant_state.bitsandbytes__nf4` describe a packed 4-bit codebook tensor.
//
// Claimed weights get their `type`, `ne[]` and `quant` fields rewritten to the
// decoded form; consumed sidecars and unclaimed byte tensors are erased from
// `tensor_storages`. Returns false only on a malformed configuration.
bool sd_apply_quant_metadata(const std::string& file_path,
                             const std::map<std::string, std::string>& metadata,
                             std::vector<TensorStorage>& tensor_storages);

#endif  // __SD_MODEL_IO_QUANT_CONFIG_H__
