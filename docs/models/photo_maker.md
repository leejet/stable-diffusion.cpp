# PhotoMaker

[PhotoMaker](https://github.com/TencentARC/PhotoMaker) personalizes image
generation from identity images. It requires an SDXL base model.

## Model files

| Component | Weights and downloads | CLI option | Requirement |
| --- | --- | --- | --- |
| SDXL checkpoint | [SDXL guide](sd.md) | `-m` / `--model` | Required; PhotoMaker is SDXL-only |
| SDXL VAE | VAE compatible with the selected SDXL checkpoint | `--vae` | Use an override when required |
| PhotoMaker v1 | [Safetensors weights](https://huggingface.co/bssrdf/PhotoMaker) | `--photo-maker` | For v1; the original .bin release is unsupported |
| PhotoMaker v2 | [Safetensors weights](https://huggingface.co/bssrdf/PhotoMakerV2) | `--photo-maker` | Alternative to v1; requires precomputed face embeddings |
| Identity images | A directory of portraits of the same person | `--pm-id-images-dir` | Required |
| PhotoMaker v2 embeddings | `id_embeds.bin` from the preparation below | `--pm-id-embed-path` | Required for v2 |

### Prepare PhotoMaker v2 embeddings

[PhotoMaker v2](https://github.com/TencentARC/PhotoMaker/blob/main/README_pmv2.md)
adds a face-embedding preparation step. From the source repository root, run
[`scripts/face_detect.py`](../../scripts/face_detect.py) on the identity image directory:

**Windows (PowerShell)**

```powershell
python scripts\face_detect.py .\inputs\photomaker_examples\scarletthead_woman
```

**Linux/macOS**

```sh
python scripts/face_detect.py ./inputs/photomaker_examples/scarletthead_woman
```

The script requires its Python dependencies, including PyTorch, NumPy, Diffusers,
and InsightFace. It writes `id_embeds.bin` inside the input directory. Compute it
once and reuse it for later generations. The image directory and `id_embeds.bin` stay under the repository's `inputs/`
directory.

## Usage

CLI examples use the [shared path and shell conventions](README.md#example-conventions).

### Web UI

Follow the [Web UI guide](../../examples/server/README.md#web-ui) to start the
server, then select the recipe matching your weights. Fill the component slots
below and click **Load models**.
Use the files and compatibility requirements in [Model files](#model-files).

| Recipe | Component slots |
| --- | --- |
| SDXL PhotoMaker (`sdxl-photomaker`) | `model`, `photo_maker`, `vae`, `tae` |

Slot IDs correspond to CLI component options: `diffusion_model` uses
`--diffusion-model`, `llm` uses `--llm`, and `model` uses `-m` / `--model`.
Select optional components when required by your checkpoint or task.

Select the appropriate PhotoMaker weights and supply the identity images.
PhotoMaker v2 also needs the prepared identity embedding.

### CLI

#### PhotoMaker Version 1

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  -m ..\models\checkpoints\sdxlUnstableDiffusers_v11.safetensors `
  --vae ..\models\vae\sdxl_vae.safetensors `
  --photo-maker ..\models\photomaker\photomaker-v1.safetensors `
  --pm-id-images-dir ..\inputs\photomaker_examples\scarletthead_woman `
  -p "a girl img, retro futurism, retro game art style but extremely beautiful, intricate details, masterpiece, best quality, space-themed, cosmic, celestial, stars, galaxies, nebulas, planets, science fiction, highly detailed" `
  -n "realistic, photo-realistic, worst quality, greyscale, bad anatomy, bad hands, error, text" `
  --cfg-scale 5.0 `
  --sampling-method euler `
  --steps 50 `
  -H 1024 `
  -W 1024 `
  --pm-style-strength 10 `
  --vae-on-cpu
```

**Linux/macOS**

```sh
./bin/sd-cli \
  -m ../models/checkpoints/sdxlUnstableDiffusers_v11.safetensors \
  --vae ../models/vae/sdxl_vae.safetensors \
  --photo-maker ../models/photomaker/photomaker-v1.safetensors \
  --pm-id-images-dir ../inputs/photomaker_examples/scarletthead_woman \
  -p "a girl img, retro futurism, retro game art style but extremely beautiful, intricate details, masterpiece, best quality, space-themed, cosmic, celestial, stars, galaxies, nebulas, planets, science fiction, highly detailed" \
  -n "realistic, photo-realistic, worst quality, greyscale, bad anatomy, bad hands, error, text" \
  --cfg-scale 5.0 \
  --sampling-method euler \
  --steps 50 \
  -H 1024 \
  -W 1024 \
  --pm-style-strength 10 \
  --vae-on-cpu
```

#### PhotoMaker Version 2

From the build directory, use the prepared identity embedding:

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  -m ..\models\checkpoints\sdxlUnstableDiffusers_v11.safetensors `
  --vae ..\models\vae\sdxl_vae.safetensors `
  --photo-maker ..\models\photomaker\photomaker-v2.safetensors `
  --pm-id-images-dir ..\inputs\photomaker_examples\scarletthead_woman `
  --pm-id-embed-path ..\inputs\photomaker_examples\scarletthead_woman\id_embeds.bin `
  -p "a girl img, retro futurism, retro game art style but extremely beautiful, intricate details, masterpiece, best quality, space-themed, cosmic, celestial, stars, galaxies, nebulas, planets, science fiction, highly detailed" `
  -n "realistic, photo-realistic, worst quality, greyscale, bad anatomy, bad hands, error, text" `
  --cfg-scale 5.0 `
  --sampling-method euler `
  --steps 50 `
  -H 1024 `
  -W 1024 `
  --pm-style-strength 10 `
  --vae-on-cpu
```

**Linux/macOS**

```sh
./bin/sd-cli \
  -m ../models/checkpoints/sdxlUnstableDiffusers_v11.safetensors \
  --vae ../models/vae/sdxl_vae.safetensors \
  --photo-maker ../models/photomaker/photomaker-v2.safetensors \
  --pm-id-images-dir ../inputs/photomaker_examples/scarletthead_woman \
  --pm-id-embed-path ../inputs/photomaker_examples/scarletthead_woman/id_embeds.bin \
  -p "a girl img, retro futurism, retro game art style but extremely beautiful, intricate details, masterpiece, best quality, space-themed, cosmic, celestial, stars, galaxies, nebulas, planets, science fiction, highly detailed" \
  -n "realistic, photo-realistic, worst quality, greyscale, bad anatomy, bad hands, error, text" \
  --cfg-scale 5.0 \
  --sampling-method euler \
  --steps 50 \
  -H 1024 \
  -W 1024 \
  --pm-style-strength 10 \
  --vae-on-cpu
```

## Options

The prompt must include a class word followed by the trigger word `img`, for
example `a girl img`. Class words include `man`, `woman`, `girl`, and `boy`.
For Asian identity images, the original workflow places `Asian` before the class word.

`--pm-style-strength` accepts 0-100 and defaults to 20. The original examples use
10-20; lower values follow the source identity more closely. Use CFG 5 and
1024 x 1024 as in the example. On GPUs with 8 GB or less, `--vae-on-cpu` can help
with VAE memory usage.
