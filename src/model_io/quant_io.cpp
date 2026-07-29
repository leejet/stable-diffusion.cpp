#include "model_io/quant_io.h"

#include <cstring>

ggml_type sd_quant_pack_target_type(SDQuantPack pack) {
    switch (pack) {
        case SD_QUANT_PACK_INT4:
            return GGML_TYPE_Q4_0;
        case SD_QUANT_PACK_INT8:
            return GGML_TYPE_Q8_0;
        case SD_QUANT_PACK_NF4:
            return GGML_TYPE_F32;
        default:
            return GGML_TYPE_COUNT;
    }
}

int64_t sd_quant_pack_nbytes(SDQuantPack pack, int64_t nelements) {
    switch (pack) {
        case SD_QUANT_PACK_INT4:
        case SD_QUANT_PACK_NF4:
            return (nelements + 1) / 2;
        case SD_QUANT_PACK_INT8:
            return nelements;
        default:
            return 0;
    }
}

const char* sd_quant_pack_name(SDQuantPack pack) {
    switch (pack) {
        case SD_QUANT_PACK_INT4:
            return "int4";
        case SD_QUANT_PACK_INT8:
            return "int8";
        case SD_QUANT_PACK_NF4:
            return "nf4";
        default:
            return "none";
    }
}

const float* sd_nf4_default_codebook() {
    static const float codebook[16] = {
        -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
        -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
        0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
        0.44070982933044434f, 0.5626170039176941f, 0.7229568362236023f, 1.0f};
    return codebook;
}

// Sign-extend a 4-bit two's-complement nibble to [-8,7].
static inline int sd_nibble_to_int4(uint8_t nibble) {
    return (int)(nibble & 0x0F) - (int)((nibble & 0x08) << 1);
}

void sd_quant_int4_to_q4_0(const void* src, const float* scales, int64_t ne0, int64_t nrows, void* dst) {
    const int qk           = 32;  // QK4_0
    const int64_t nb_row   = ne0 / qk;
    const size_t block_sz  = sizeof(ggml_fp16_t) + qk / 2;
    const uint8_t* src_u8  = (const uint8_t*)src;
    uint8_t* dst_u8        = (uint8_t*)dst;
    const int64_t row_pack = ne0 / 2;

    for (int64_t r = 0; r < nrows; r++) {
        const uint8_t* srow = src_u8 + r * row_pack;
        // One scale per output channel; every block in the row shares it.
        const ggml_fp16_t d = ggml_fp32_to_fp16(scales[r]);

        for (int64_t b = 0; b < nb_row; b++) {
            uint8_t* blk = dst_u8 + (r * nb_row + b) * block_sz;
            memcpy(blk, &d, sizeof(ggml_fp16_t));
            uint8_t* qs = blk + sizeof(ggml_fp16_t);

            // Source packs adjacent pairs (elements 2i, 2i+1 share a byte, earlier
            // in the low nibble). Q4_0 packs element j with element j+16. Both the
            // pairing and the bias differ, so this is a permutation, not a copy.
            for (int j = 0; j < qk / 2; j++) {
                const int64_t e0 = b * qk + j;
                const int64_t e1 = b * qk + j + qk / 2;

                const uint8_t p0 = srow[e0 >> 1];
                const uint8_t p1 = srow[e1 >> 1];
                const int v0     = sd_nibble_to_int4((e0 & 1) ? (p0 >> 4) : p0);
                const int v1     = sd_nibble_to_int4((e1 & 1) ? (p1 >> 4) : p1);

                // Values live in [-7,7], so +8 stays inside the [0,15] nibble range.
                qs[j] = (uint8_t)((v0 + 8) & 0x0F) | (uint8_t)(((v1 + 8) & 0x0F) << 4);
            }
        }
    }
}

void sd_quant_int8_to_q8_0(const void* src, const float* scales, int64_t ne0, int64_t nrows, void* dst) {
    const int qk          = 32;  // QK8_0
    const int64_t nb_row  = ne0 / qk;
    const size_t block_sz = sizeof(ggml_fp16_t) + qk;
    const int8_t* src_i8  = (const int8_t*)src;
    uint8_t* dst_u8       = (uint8_t*)dst;

    for (int64_t r = 0; r < nrows; r++) {
        const int8_t* srow  = src_i8 + r * ne0;
        const ggml_fp16_t d = ggml_fp32_to_fp16(scales[r]);

        for (int64_t b = 0; b < nb_row; b++) {
            uint8_t* blk = dst_u8 + (r * nb_row + b) * block_sz;
            memcpy(blk, &d, sizeof(ggml_fp16_t));
            // Q8_0 stores quants in natural order, so this half is a straight copy.
            memcpy(blk + sizeof(ggml_fp16_t), srow + b * qk, qk);
        }
    }
}

void sd_quant_nf4_to_f32(const void* src,
                         const float* absmax,
                         const float* codebook,
                         int64_t block_size,
                         int64_t nelements,
                         float* dst) {
    const uint8_t* src_u8 = (const uint8_t*)src;
    if (codebook == nullptr) {
        codebook = sd_nf4_default_codebook();
    }
    if (block_size <= 0) {
        block_size = 64;
    }

    for (int64_t i = 0; i < nelements; i++) {
        const uint8_t packed = src_u8[i >> 1];
        // bitsandbytes writes the earlier element into the HIGH nibble.
        const uint8_t idx = (i & 1) ? (packed & 0x0F) : (packed >> 4);
        const float scale = absmax != nullptr ? absmax[i / block_size] : 1.0f;
        dst[i]            = codebook[idx] * scale;
    }
}
