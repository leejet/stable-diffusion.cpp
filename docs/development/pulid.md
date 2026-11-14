# PuLID implementation and validation

For model files, identity preparation, and generation commands, see the
[PuLID usage guide](../models/pulid.md).

## Architecture summary

The PuLID-Flux contribution to the Flux denoise loop is a stack of 20
small cross-attention modules (`PerceiverAttentionCA`) inserted between
the Flux transformer blocks:

- After every 2nd of the 19 double-stream blocks (10 hook points)
- After every 4th of the 38 single-stream blocks (10 hook points)

Each cross-attention layer takes the current image tokens as query, the
32-token / 2048-dim identity embedding as key+value, and adds its output
(scaled by `id_weight`, typically 1.0) back to the image tokens.

## Format (gguf)

The embedding is a standard **gguf** container holding a single tensor:

```
tensor name : "pulid_id"
shape       : [token_dim, num_tokens]   (ggml order; typically [2048, 32])
type        : F16 (also accepts F32 / BF16)
metadata    : general.architecture = "pulid", pulid.version = 1
```

stable-diffusion.cpp loads it with the normal gguf reader
(`gguf_init_from_file`) and converts to fp32 at load time -- no bespoke
parser. Total file size for the typical (32, 2048, fp16) case is ~131 KB.

## Verification

A three-way SHA-256 check is the recommended sanity test when bringing up
a new combination of model + backend + hardware:

| Run                                          | Expected hash relation             |
|----------------------------------------------|------------------------------------|
| A: no `--pulid-*` flags                      | baseline                           |
| B: PuLID flags, `--pulid-id-weight 0.0`      | **byte-identical to A**            |
| C: PuLID flags, `--pulid-id-weight 1.0`      | **different from A,B**, preserves source identity |

If A and C differ but A and B differ too, the injection is allocating
or computing something even at zero weight -- likely a bug.
