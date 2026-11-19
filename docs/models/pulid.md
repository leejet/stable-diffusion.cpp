# PuLID-Flux

[PuLID-Flux](https://github.com/ToTheBeginning/PuLID) preserves a source portrait
identity while generating new images with FLUX.1 dev or schnell. Compute the
identity embedding once with the external Python tool, then reuse it.

## Model files

| Component | Weights and downloads | CLI option | Requirement |
| --- | --- | --- | --- |
| FLUX.1 base components | [FLUX.1 guide](flux.md) | `--diffusion-model`, `--vae`, `--clip_l`, `--t5xxl` | Required; dev or schnell |
| PuLID v0.9.x weights | [guozinan/PuLID](https://huggingface.co/guozinan/PuLID) | `--pulid-weights` | v0.9.1 is the tested version; v1.1 is unsupported |
| Identity embedding | `source.pulidembd` from preparation below | `--pulid-id-embedding` | Required for identity preservation |

### Prepare the identity embedding

The precompute tool extracts an identity embedding from a single portrait
and writes it to a `.pulidembd` file. Run it once per source
person; the same file is reused for any number of generations.

Use
[`scripts/pulid_extract_id.py`](../../scripts/pulid_extract_id.py). It
requires:
- A working CUDA / CPU PyTorch stack
- `insightface`, `facexlib`, `eva-clip`, `torchvision`, `opencv-python`,
  `huggingface_hub`, `gguf`
- The PuLID weights file (same one stable-diffusion.cpp will load below)
- The ToTheBeginning/PuLID repo's `pulid/` package (including
  `pulid/pipeline_flux.py`) and `eva_clip/` package on `PYTHONPATH`; `flux/`
  is not needed for embedding extraction

From the source repository root, run:

**Windows (PowerShell)**

```powershell
python scripts\pulid_extract_id.py --portrait .\inputs\source-photo.jpg --pulid-weights .\models\pulid\pulid_flux_v0.9.1.safetensors --out .\inputs\source.pulidembd
```

**Linux/macOS**

```sh
python scripts/pulid_extract_id.py --portrait ./inputs/source-photo.jpg --pulid-weights ./models/pulid/pulid_flux_v0.9.1.safetensors --out ./inputs/source.pulidembd
```

The script writes the embedding to the repository's `inputs/` directory.
The examples below read the same file from the build directory.

## Usage

CLI examples use the [shared path and shell conventions](README.md#example-conventions).

### Web UI

Follow the [Web UI guide](../../examples/server/README.md#web-ui) to start the
server, then select the recipe matching your weights. Fill the component slots
below and click **Load models**.
Use the files and compatibility requirements in [Model files](#model-files).

| Recipe | Component slots |
| --- | --- |
| FLUX.1 dev PuLID (`flux1-pulid`) | `diffusion_model`, `clip_l`, `t5xxl`, `pulid_weights`, `vae`, `tae` |

Slot IDs correspond to CLI component options: `diffusion_model` uses
`--diffusion-model`, `llm` uses `--llm`, and `model` uses `-m` / `--model`.
Select optional components when required by your checkpoint or task.

### CLI

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  --diffusion-model ..\models\diffusion_models\flux1-schnell-Q4_K_S.gguf `
  --vae ..\models\vae\ae.safetensors `
  --clip_l ..\models\text_encoders\clip_l.safetensors `
  --t5xxl ..\models\text_encoders\t5xxl_fp16.safetensors `
  --pulid-weights ..\models\pulid\pulid_flux_v0.9.1.safetensors `
  --pulid-id-embedding ..\inputs\source.pulidembd `
  -p "candid photograph of a young woman on a beach at sunset" `
  --cfg-scale 1.0 `
  --sampling-method euler `
  --steps 4 `
  -W 512 `
  -H 512 `
  --pulid-id-weight 1.0 `
  --clip-on-cpu `
  --seed 42 `
  -o out.png
```

**Linux/macOS**

```sh
./bin/sd-cli \
  --diffusion-model ../models/diffusion_models/flux1-schnell-Q4_K_S.gguf \
  --vae ../models/vae/ae.safetensors \
  --clip_l ../models/text_encoders/clip_l.safetensors \
  --t5xxl ../models/text_encoders/t5xxl_fp16.safetensors \
  --pulid-weights ../models/pulid/pulid_flux_v0.9.1.safetensors \
  --pulid-id-embedding ../inputs/source.pulidembd \
  -p "candid photograph of a young woman on a beach at sunset" \
  --cfg-scale 1.0 \
  --sampling-method euler \
  --steps 4 \
  -W 512 \
  -H 512 \
  --pulid-id-weight 1.0 \
  --clip-on-cpu \
  --seed 42 \
  -o out.png
```

For Flux Dev (instead of Schnell), add `--guidance 3.5` and `--steps 20`.

## Options

| Flag                       | Purpose                                                           |
|----------------------------|-------------------------------------------------------------------|
| `--pulid-weights <path>`   | Path to `pulid_flux_v0.9.x.safetensors`. Loaded with the model.   |
| `--pulid-id-embedding <p>` | Path to a `.pulidembd` binary produced by the precompute tool.    |
| `--pulid-id-weight <f>`    | Identity-injection strength. Typical 0.7-1.2; default 1.0.        |

Supply both the PuLID weights and identity embedding to activate identity
conditioning. `--pulid-id-weight` defaults to 1.0; set it to 0 to disable identity
injection. Loading `--pulid-weights` without an embedding leaves identity
conditioning inactive.

### Backend selection

The standard `--backend` flag works as documented. Common patterns:

**Windows (PowerShell) arguments**

```text
# AMD Vulkan
--backend "diffusion=vulkan0,vae=cpu"

# NVIDIA Vulkan
--backend "diffusion=vulkan1,vae=cpu"

# CUDA
--backend "diffusion=cuda0,vae=cpu"
```

**Linux/macOS arguments**

```text
# AMD Vulkan
--backend diffusion=vulkan0,vae=cpu

# NVIDIA Vulkan
--backend diffusion=vulkan1,vae=cpu

# CUDA
--backend diffusion=cuda0,vae=cpu
```

The PuLID cross-attention layers run on the same backend as the main
diffusion model. They have not yet been independently profiled on every
backend; only Vulkan and CPU have been tested by the original contributor.

## Notes and limitations

PuLID v1.1 weights are unsupported. Use PuLID and skip-layer guidance
(`--skip-layers`) separately. One precomputed identity embedding is accepted per
request; prepare it using the external tool. Identity conditioning is applied
to the positive CFG branch.

At 512x512, 4 steps (Schnell), the 20 cross-attention layers add roughly
10% to denoise time and almost nothing to peak VRAM. Tested on a 12 GB
consumer card alongside Flux Schnell Q4 GGUF + CPU-offloaded clip_l and
t5xxl + GPU-resident VAE.

At 1024x1024 with Flux Dev Q4 + 20 steps + PuLID, the VAE decode compute
buffer doesn't fit on a 12 GB card even with `--vae-on-cpu`. Workaround:
explicitly route VAE to the CPU backend instead of the offload flag:

**Windows (PowerShell) arguments**

```text
--backend "diffusion=vulkan0,vae=cpu"
```

**Linux/macOS arguments**

```text
--backend diffusion=vulkan0,vae=cpu
```

The `--vae-on-cpu` flag offloads VAE weights but leaves the compute graph
on the default backend; this is existing stable-diffusion.cpp behavior,
not a PuLID-specific issue. Documented here because anyone running PuLID
at 1024 will hit it.

The original contributor tested Vulkan and CPU. Other backends were not
validated in those tests. See [backend selection](../guides/backend.md) for
device placement. Internal format details and validation procedures are in the
[PuLID developer notes](../development/pulid.md).
