# TAESD and related tiny autoencoders

Tiny autoencoders provide an alternative to full VAE encoding, decoding, or
previews. Choose the file matching the base model.

## Model files

| Component | Weights and downloads | CLI option | Requirement |
| --- | --- | --- | --- |
| TAESD | [Safetensors weights](https://huggingface.co/madebyollin/taesd/blob/main/diffusion_pytorch_model.safetensors) | `--taesd` / `--tae` | For compatible SD 1.x models |
| TAEHV for Wan 2.1 | [taew2_1.safetensors](https://github.com/madebyollin/taehv/blob/main/safetensors/taew2_1.safetensors) | `--taesd` / `--tae` | Qwen Image, Wan 2.1, and Wan 2.2 A14B |
| TAEHV for Wan 2.2 5B | [taew2_2.safetensors](https://github.com/madebyollin/taehv/blob/main/safetensors/taew2_2.safetensors) | `--taesd` / `--tae` | Wan 2.2 TI2V 5B only |
| TAEQI2.1 | [taeqi2_1.safetensors](https://huggingface.co/madebyollin/taeqi2_1/blob/main/taeqi2_1.safetensors) | `--taesd` / `--tae` | Qwen Image 2.1, including RGBA |

### Qwen-Image and wan (TAEHV)

[TAEHV](https://github.com/madebyollin/taehv) provides tiny autoencoders for the
model families in the table. Replace `--vae PATH` with `--taesd PATH` using the
matching file. A tiny autoencoder can reduce VAE memory requirements, with a
possible quality tradeoff.

### Qwen Image 2.1 (TAEQI2.1)

[TAEQI2.1](https://github.com/madebyollin/taesd) supports Qwen Image 2.1 RGBA
images. Download the official file directly; conversion is unnecessary. Replace
`--vae PATH` with `--taesd taeqi2_1.safetensors` in the
[Qwen Image 2.1 examples](qwen_image_2.1.md).

### Download commands

Save the files under the repository's `models/vae/` directory. From the build
directory, create it if needed:

**Windows (PowerShell)**

```powershell
New-Item -ItemType Directory -Force ..\models\vae | Out-Null
```

**Linux/macOS**

```sh
mkdir -p ../models/vae
```

**Windows (PowerShell)**

```powershell
curl.exe -L -o ..\models\vae\diffusion_pytorch_model.safetensors https://huggingface.co/madebyollin/taesd/resolve/main/diffusion_pytorch_model.safetensors
```

**Linux/macOS**

```sh
curl -L -o ../models/vae/diffusion_pytorch_model.safetensors https://huggingface.co/madebyollin/taesd/resolve/main/diffusion_pytorch_model.safetensors
```

**Windows (PowerShell)**

```powershell
curl.exe -L -o ..\models\vae\taew2_1.safetensors https://github.com/madebyollin/taehv/raw/refs/heads/main/safetensors/taew2_1.safetensors
```

**Linux/macOS**

```sh
curl -L -o ../models/vae/taew2_1.safetensors https://github.com/madebyollin/taehv/raw/refs/heads/main/safetensors/taew2_1.safetensors
```

**Windows (PowerShell)**

```powershell
curl.exe -L -o ..\models\vae\taew2_2.safetensors https://github.com/madebyollin/taehv/raw/refs/heads/main/safetensors/taew2_2.safetensors
```

**Linux/macOS**

```sh
curl -L -o ../models/vae/taew2_2.safetensors https://github.com/madebyollin/taehv/raw/refs/heads/main/safetensors/taew2_2.safetensors
```

**Windows (PowerShell)**

```powershell
curl.exe -L -o ..\models\vae\taeqi2_1.safetensors https://huggingface.co/madebyollin/taeqi2_1/resolve/main/taeqi2_1.safetensors
```

**Linux/macOS**

```sh
curl -L -o ../models/vae/taeqi2_1.safetensors https://huggingface.co/madebyollin/taeqi2_1/resolve/main/taeqi2_1.safetensors
```

## Usage

CLI examples use the [shared path and shell conventions](README.md#example-conventions).

### Web UI

Use a compatible base model recipe that offers the `tae` slot, and select the
tiny autoencoder matching that model. See [Web UI guide](../../examples/server/README.md#web-ui).

### CLI

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  -m ..\models\checkpoints\v1-5-pruned-emaonly.safetensors `
  --taesd ..\models\vae\diffusion_pytorch_model.safetensors `
  -p "a lovely cat"
```

**Linux/macOS**

```sh
./bin/sd-cli \
  -m ../models/checkpoints/v1-5-pruned-emaonly.safetensors \
  --taesd ../models/vae/diffusion_pytorch_model.safetensors \
  -p "a lovely cat"
```

## Options

For previews only, keep the full `--vae PATH` and add
`--taesd PATH --taesd-preview-only --preview tae`.

`sd-server` accepts `--taesd` and `--taesd-preview-only` at startup; choose the
preview method in each request. For device placement and convolution settings,
see [backend selection](../guides/backend.md).
