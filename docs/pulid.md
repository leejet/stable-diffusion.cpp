# PuLID-Flux face-identity preservation

stable-diffusion.cpp supports the [PuLID-Flux](https://github.com/ToTheBeginning/PuLID)
identity-injection technique on top of Flux.1 (schnell or dev) models.
Given a single source portrait, PuLID-Flux produces new generations that
preserve the source person's face across arbitrary scenes, poses, and
prompts.

Unlike PhotoMaker (which extracts the identity inside the inference
process from a directory of images), PuLID-Flux's identity extractor is
a heavy stack (insightface ArcFace + EVA-CLIP-L + IDFormer encoder) that
is impractical to port to C++/ggml. To keep this implementation small and
cross-vendor, **stable-diffusion.cpp consumes a precomputed identity
embedding** produced by an external Python tool that runs once per source
portrait. Everything downstream of that one-shot extraction is C++ and
runs on any backend (Vulkan, CUDA, Metal, ROCm, CPU).

## Architecture summary

The PuLID-Flux contribution to the Flux denoise loop is a stack of 20
small cross-attention modules (`PerceiverAttentionCA`) inserted between
the Flux transformer blocks:

- After every 2nd of the 19 double-stream blocks (10 hook points)
- After every 4th of the 38 single-stream blocks (10 hook points)

Each cross-attention layer takes the current image tokens as query, the
32-token / 2048-dim identity embedding as key+value, and adds its output
(scaled by `id_weight`, typically 1.0) back to the image tokens.

## Required weights

Three files in addition to the standard Flux weight set:

1. **Flux base** (transformer + VAE + clip_l + t5xxl) -- exactly as
   [docs/flux.md](flux.md) describes.
2. **PuLID weights** -- download from
   [guozinan/PuLID](https://huggingface.co/guozinan/PuLID):
   - `pulid_flux_v0.9.0.safetensors` or `pulid_flux_v0.9.1.safetensors`
     (recommended; this implementation is verified against v0.9.1)
   - **v1.1 (`pulid_v1.1.safetensors`) is NOT yet supported** -- it uses
     renamed keys (`id_adapter_attn_layers.*` instead of `pulid_ca.*`)
     and possibly different module structure. Future PR.
3. **Identity embedding (.pulidembd)** -- produced by the precompute
   tool below.

## Precompute the identity embedding

The precompute tool runs the PyTorch identity-extraction stack on a
single portrait image and writes the resulting `(32, 2048)` embedding
to a `.pulidembd` binary file (about 131 KB). Run it once per source
person; the same file is reused for any number of generations.

A reference Python script is provided alongside this docs file at
[`scripts/pulid_extract_id.py`](../scripts/pulid_extract_id.py). It
requires:
- A working CUDA / CPU PyTorch + diffusers stack
- `insightface`, `facexlib`, `eva-clip`, `torchvision`
- The PuLID weights file (same one stable-diffusion.cpp will load below)
- The ToTheBeginning/PuLID repo's `pulid/pipeline_flux.py` (and its
  dependencies under `pulid/` and `flux/`) -- recommended to vendor
  rather than pip-install due to upstream packaging quirks

Run it as:

```
python pulid_extract_id.py \
  --portrait /path/to/source-photo.jpg \
  --pulid-weights /path/to/pulid_flux_v0.9.1.safetensors \
  --out /path/to/source.pulidembd
```

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

## Command-line usage

```
.\bin\Release\sd-cli.exe \
  --diffusion-model     models\flux1-schnell-Q4_K_S.gguf \
  --vae                 models\ae.safetensors \
  --clip_l              models\clip_l.safetensors \
  --t5xxl               models\t5xxl_fp16.safetensors \
  --pulid-weights       models\pulid_flux_v0.9.1.safetensors \
  --pulid-id-embedding  source.pulidembd \
  --pulid-id-weight     1.0 \
  -p "candid photograph of a young woman on a beach at sunset" \
  --cfg-scale 1.0 --sampling-method euler --steps 4 -W 512 -H 512 \
  --seed 42 --clip-on-cpu \
  -o out.png
```

For Flux Dev (instead of Schnell), add `--guidance 3.5` and `--steps 20`.

## Flags

| Flag                       | Purpose                                                           |
|----------------------------|-------------------------------------------------------------------|
| `--pulid-weights <path>`   | Path to `pulid_flux_v0.9.x.safetensors`. Loaded with the model.   |
| `--pulid-id-embedding <p>` | Path to a `.pulidembd` binary produced by the precompute tool.    |
| `--pulid-id-weight <f>`    | Identity-injection strength. Typical 0.7-1.2; default 1.0.        |

All three flags must be set together to activate PuLID. Setting only
`--pulid-weights` (no embedding) loads the weights but disables injection
at runtime. Setting `--pulid-id-weight 0` zeros out the contribution
(useful for falsification testing: outputs should be byte-identical to
a no-PuLID run with the same seed).

## Memory budget

At 512x512, 4 steps (Schnell), the 20 cross-attention layers add roughly
10% to denoise time and almost nothing to peak VRAM. Tested on a 12 GB
consumer card alongside Flux Schnell Q4 GGUF + CPU-offloaded clip_l and
t5xxl + GPU-resident VAE.

At 1024x1024 with Flux Dev Q4 + 20 steps + PuLID, the VAE decode compute
buffer doesn't fit on a 12 GB card even with `--vae-on-cpu`. Workaround:
explicitly route VAE to the CPU backend instead of the offload flag:

```
--backend "diffusion=vulkan0,vae=cpu"
```

The `--vae-on-cpu` flag offloads VAE weights but leaves the compute graph
on the default backend; this is existing stable-diffusion.cpp behavior,
not a PuLID-specific issue. Documented here because anyone running PuLID
at 1024 will hit it.

## Backend selection

The standard `--backend` flag works as documented. Common patterns:

```
# AMD Vulkan
--backend "diffusion=vulkan0,vae=cpu"

# NVIDIA Vulkan
--backend "diffusion=vulkan1,vae=cpu"

# CUDA
--backend "diffusion=cuda0,vae=cpu"
```

The PuLID cross-attention layers run on the same backend as the main
diffusion model. They have not yet been independently profiled on every
backend; only Vulkan and CPU have been tested by the original contributor.

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

## Limitations / not yet supported

- **`--skip-layers` (skip-layer-guidance / SLG) combined with PuLID** is not
  supported. The `pulid_ca` index advances per non-skipped block, so a
  skipped block silently misaligns the cross-attention weight assignment
  vs. the trained intervals. The reference PyTorch implementation does
  not have SLG either, so there is no well-defined behavior to emulate.
  Use either feature alone.
- **PuLID v1.1 weights** (`pulid_v1.1.safetensors`, renamed key layout).
- **Multiple ID images.** The reference PyTorch implementation can fuse
  several portraits into one embedding for stronger identity. This
  implementation accepts a single embedding produced from one or more
  images by the external precompute tool.
- **Negative-prompt branch of CFG.** PuLID only injects on the positive
  conditioning path in the published reference, and the implementation
  here follows that. Flux's distilled guidance doesn't run a separate
  uncond branch in normal use, so this matters only for `--true-cfg`
  workflows that aren't standard for Flux.
- **Backends other than Vulkan and CPU** are untested by the original
  contributor. The implementation is pure-ggml and should work on CUDA,
  ROCm, and Metal, but verification by users on those backends is
  welcomed.
