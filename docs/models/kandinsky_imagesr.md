# Kandinsky ImageSR 2.0

Kandinsky ImageSR performs 2× and 4× super-resolution on RGB images. Both scales use the same weights. The default is 4×.

## Model files

| Component | Weights and downloads | CLI option | Requirement |
| --- | --- | --- | --- |
| ImageSR transformer | [Transformer files](https://huggingface.co/kandinskylab/Kandinsky-ImageSR-2.0/tree/main/transformer) | `--upscale-model` | Keep the configuration, shard index and all shards under `transformer/`. |
| FLUX VAE | [VAE files](https://huggingface.co/kandinskylab/Kandinsky-ImageSR-2.0/tree/main/vae) | Same bundle directory | Keep the supplied VAE under `vae/`. |

From the repository root, download both components with the Hugging Face CLI.

```sh
hf download kandinskylab/Kandinsky-ImageSR-2.0 --include 'transformer/*' 'vae/*' --local-dir models/Kandinsky-ImageSR-2.0
```

## Usage

Run the following examples from the build directory with an RGB image named `input.png`.

### DGL

| Task | Script |
| --- | --- |
| Upscale an image at 2× | [Kandinsky ImageSR upscaling](../dgl/kandinsky_imagesr/upscale.dgl) |

Set `factor = 4` in the script for 4× output.

**Windows PowerShell**

```powershell
.\bin\Release\sd-cli.exe -M dgl --script ..\docs\dgl\kandinsky_imagesr\upscale.dgl
```

**Linux / macOS**

```sh
./bin/sd-cli -M dgl --script ../docs/dgl/kandinsky_imagesr/upscale.dgl
```

### Legacy CLI

**Windows PowerShell**

```powershell
.\bin\Release\sd-cli.exe -M upscale --upscale-model ..\models\Kandinsky-ImageSR-2.0 `
    -i input.png --upscale-factor 2 -o output.png
```

**Linux / macOS**

```sh
./bin/sd-cli -M upscale --upscale-model ../models/Kandinsky-ImageSR-2.0 \
    -i input.png --upscale-factor 2 -o output.png
```

### API

For the [C API](../guides/c_api.md), pass the bundle directory as `model_path` and set the integer `factor` on the per-run options before calling `sd_upscale`.

For [server hires upscaling](../../examples/server/api.md), place the bundle directly under `--hires-upscalers-dir` and use its directory name in `hires.upscaler`. The hires pass selects 2× when it covers both target dimensions and 4× for larger targets.

## Notes and limitations

ImageSR processes overlapping tiles with a fixed output tile size of 1024×1024. Standalone output dimensions are exactly the input dimensions multiplied by the selected scale, including for small and non-square inputs.
