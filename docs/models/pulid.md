# PuLID-Flux

[PuLID-Flux](https://github.com/ToTheBeginning/PuLID) preserves a source portrait
identity while generating new images with FLUX.1 dev or schnell. Compute the
identity embedding once with the external Python tool, then reuse it.

**Currently supported only through dgl.**

## Model files

| Component | Weights and downloads | DGL usage | Requirement |
| --- | --- | --- | --- |
| FLUX.1 base components | [FLUX.1 guide](flux.md) | `load_dit`, `load_vae`, `load_text_encoder` | Required; dev or schnell |
| PuLID v0.9.x weights | [guozinan/PuLID](https://huggingface.co/guozinan/PuLID) | `pulid::load` | v0.9.1 is the tested version; v1.1 is unsupported |
| Identity embedding | `source.pulidembd` from preparation below | `pulid::load_embedding` | Required for identity preservation |

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

### Web UI

Open the server's **DGL editor**, edit the same script, and submit it.
Model and input paths must satisfy the server's file-access policy; save outputs
inside its `output/` directory. See the [Web UI guide](../../examples/server/README.md#web-ui).

### DGL

| Example | Description |
| --- | --- |
| [Identity-guided generation](../dgl/pulid/pulid-flux1-schnell.dgl) | Use a prepared identity embedding with FLUX.1 schnell. |

#### Run a script

Edit paths, prompts and settings in the script's `var` block, then run from
the build directory. Paths inside the script resolve from the working directory.

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe -M dgl --script ..\docs\dgl\pulid\pulid-flux1-schnell.dgl
```

**Linux/macOS**

```sh
./bin/sd-cli -M dgl --script ../docs/dgl/pulid/pulid-flux1-schnell.dgl
```

Choose another script from the table to run a different task. Add `--check`
to validate syntax and types without loading weights. See the
[shared run instructions](../dgl/README.md#run-an-example) for build paths
and the server's file-access rules.

Load PuLID weights separately with `pulid::load` on the same compute backend
as the FLUX.1 denoiser. Read the identity embedding with
`pulid::load_embedding` and bind it to a text condition with `pulid::apply`
before calling `predictor`. Each binding sets its own `strength` and can
set `start_sigma` and `end_sigma` to bound the contribution in sigma space.
Omitting both bounds enables the binding throughout sampling.

For FLUX.1 dev, choose its checkpoint and set `steps = 20`. The predictor's
default guidance is 3.5.

## Options

Load weights with `pulid::load`, read the prepared `.pulidembd` file with
`pulid::load_embedding`, and attach it to the positive condition with
`pulid::apply(..., strength=1.0)`. Typical identity strengths are 0.7–1.2.
Optional `start_sigma` and `end_sigma` bound the active noise interval.

### Backend selection

DGL selects devices on individual loaders. Use the same compute backend for
`load_dit` and `pulid::load`; for example `backend="cuda0"`, `"vulkan0"` or
`"vulkan1"`, according to the available devices. To decode on CPU set
`load_vae(..., backend="cpu")`. `params_backend="cpu"` offloads weights while
leaving computation on the selected compute device.
See [backend selection](../guides/backend.md).

## Notes and limitations

PuLID v1.1 weights are unsupported. One precomputed identity embedding is
used by this script and attached to the positive condition. Prepare it with
the external tool before running the program.

Large VAE decode workspaces can exceed GPU memory at 1024 × 1024; explicitly
select `backend="cpu"` on `load_vae` when needed. The original contributor
validated PuLID on Vulkan and CPU; those results do not establish support or
performance on every backend.
