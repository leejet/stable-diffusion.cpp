#ifndef __SD_MODEL_IO_QUANT_IO_H__
#define __SD_MODEL_IO_QUANT_IO_H__

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "ggml.h"

// Support for weights that arrive already quantized by an external toolchain,
// carrying their dequantization parameters in sidecar tensors rather than in the
// block layout ggml expects. Each supported pack is repacked at load time into a
// native ggml type, so no new ggml type or backend kernel is required.
enum SDQuantPack {
    SD_QUANT_PACK_NONE = 0,
    // Two symmetric int4 per byte, earlier element in the LOW nibble, values in
    // [-7,7] (nibble 8 is unused). One F32 scale per output channel.
    // Repacked losslessly into Q4_0.
    SD_QUANT_PACK_INT4,
    // One int8 per byte, one F32 scale per output channel. Repacked into Q8_0.
    SD_QUANT_PACK_INT8,
    // bitsandbytes NF4: two 4-bit codebook indices per byte, earlier element in
    // the HIGH nibble (opposite of INT4 above), with a per-block absmax. The
    // codebook is non-uniform so there is no exact ggml equivalent; decoded to F32
    // and left to the normal conversion path to retarget.
    SD_QUANT_PACK_NF4,
};

struct SDQuantParams {
    SDQuantPack pack = SD_QUANT_PACK_NONE;
    // ConvRot rotation group size; 0 when the layer is not rotated. Non-zero
    // obliges the consumer to rotate activations before the matmul, because the
    // stored weight is W*H rather than W.
    int convrot_groupsize = 0;
    // NF4 absmax block size (bitsandbytes default 64).
    int block_size = 0;
    // Per-output-channel scales (INT4/INT8) or per-block absmax (NF4).
    std::vector<float> scales;
    // NF4 codebook, 16 entries.
    std::vector<float> codebook;
};

// ggml type a pack is repacked into.
ggml_type sd_quant_pack_target_type(SDQuantPack pack);

// On-disk byte count for `nelements` logical elements.
int64_t sd_quant_pack_nbytes(SDQuantPack pack, int64_t nelements);

const char* sd_quant_pack_name(SDQuantPack pack);

// The canonical bitsandbytes NF4 codebook, used when a checkpoint omits quant_map.
const float* sd_nf4_default_codebook();

// Repack `nrows` rows of `ne0` packed int4 values (plus one scale per row) into
// contiguous Q4_0 blocks. Requires ne0 % 32 == 0. src and dst must not overlap.
void sd_quant_int4_to_q4_0(const void* src, const float* scales, int64_t ne0, int64_t nrows, void* dst);

// As above for int8 -> Q8_0. Requires ne0 % 32 == 0.
void sd_quant_int8_to_q8_0(const void* src, const float* scales, int64_t ne0, int64_t nrows, void* dst);

// Decode NF4 to F32. `absmax` has one entry per `block_size` elements.
void sd_quant_nf4_to_f32(const void* src,
                         const float* absmax,
                         const float* codebook,
                         int64_t block_size,
                         int64_t nelements,
                         float* dst);

#endif  // __SD_MODEL_IO_QUANT_IO_H__
