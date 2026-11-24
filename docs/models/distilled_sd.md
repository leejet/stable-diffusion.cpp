# Distilled Stable Diffusion models

This guide covers SSD-1B, Vega, Tiny SD/SD2, and SDXS checkpoints. Choose the
base-model family and preparation steps matching the checkpoint.

## Model files

| Component | Weights and downloads | CLI option | Requirement |
| --- | --- | --- | --- |
| SSD-1B | [SSD-1B-A1111.safetensors](https://huggingface.co/segmind/SSD-1B/resolve/main/SSD-1B-A1111.safetensors) | `-m` / `--model` | Single-file SDXL-family checkpoint |
| Vega | [Segmind Vega](https://huggingface.co/segmind/Segmind-Vega) | `-m` / `--model` | Single-file SDXL-family checkpoint |
| Tiny SD / SD2 | [Repositories and conversion steps](#tiny-sd-and-tiny-sd2) | `-m` / `--model` | Convert Diffusers directories first |
| SDXS | [Ready-to-run SDXS weights](#sdxs-512-dreamshaper-weights) | `-m` / `--model` | Use the compatible autoencoder and one-step settings |

### SSD1B

Note that not all of these models follow the standard parameter naming conventions. However, several useful SSD-1B models are available online, such as:

 * https://huggingface.co/segmind/SSD-1B/resolve/main/SSD-1B-A1111.safetensors
 * https://huggingface.co/hassenhamdi/SSD-1B-fp8_e4m3fn/resolve/main/SSD-1B_fp8_e4m3fn.safetensors

Useful LoRAs are also available:

 * https://huggingface.co/seungminh/lora-swarovski-SSD-1B/resolve/main/pytorch_lora_weights.safetensors
 * https://huggingface.co/kylielee505/mylcmlorassd/resolve/main/pytorch_lora_weights.safetensors

### Vega

Segmind's Vega model is available online here:

 * https://huggingface.co/segmind/Segmind-Vega/resolve/main/segmind-vega.safetensors

VegaRT is an example for an LCM-LoRA:

 * https://huggingface.co/segmind/Segmind-VegaRT/resolve/main/pytorch_lora_weights.safetensors

Both files can be used out-of-the-box, unlike the models described in next sections.

### Tiny SD and Tiny SD2

These models require conversion before use. You will need a Python script provided by the diffusers team, available on GitHub:

 * https://raw.githubusercontent.com/huggingface/diffusers/refs/heads/main/scripts/convert_diffusers_to_original_stable_diffusion.py

#### SD2.x

NotaAI provides the following model online:

* https://huggingface.co/nota-ai/bk-sdm-v2-tiny

Creating a .safetensors file involves two steps. First, run this short Python script to download the model from Hugging Face:

```python
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("nota-ai/bk-sdm-v2-tiny",cache_dir="./")
```

Second, create the .safetensors file by running:

**Windows (PowerShell)**

```powershell
python convert_diffusers_to_original_stable_diffusion.py --model_path models--nota-ai--bk-sdm-v2-tiny/snapshots/68277af553777858cd47e133f92e4db47321bc74 --checkpoint_path bk-sdm-v2-tiny.safetensors --half --use_safetensors
```

**Linux/macOS**

```sh
python convert_diffusers_to_original_stable_diffusion.py --model_path models--nota-ai--bk-sdm-v2-tiny/snapshots/68277af553777858cd47e133f92e4db47321bc74 --checkpoint_path bk-sdm-v2-tiny.safetensors --half --use_safetensors
```

This will generate the **file bk-sdm-v2-tiny.safetensors**, which is now ready for use with sd.cpp.

#### SD1.x

Several Tiny SD 1.x models are available online, such as:

 * https://huggingface.co/segmind/tiny-sd
 * https://huggingface.co/segmind/portrait-finetuned
 * https://huggingface.co/nota-ai/bk-sdm-tiny

These models also require conversion, partly because some tensors are stored in a non-contiguous manner. To create a usable checkpoint file, follow these simple steps:
Download and prepare the model using Python:

Download the model using Python on your computer, for example this way:

```python
import torch
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("segmind/tiny-sd")
unet=pipe.unet
for param in unet.parameters():
    param.data = param.data.contiguous()     # <- important here
pipe.save_pretrained("segmindtiny-sd", safe_serialization=True)
```

Run the conversion script:

**Windows (PowerShell)**

```powershell
python convert_diffusers_to_original_stable_diffusion.py --model_path .\segmindtiny-sd --checkpoint_path .\segmind_tiny-sd.safetensors --half --use_safetensors
```

**Linux/macOS**

```sh
python convert_diffusers_to_original_stable_diffusion.py --model_path ./segmindtiny-sd --checkpoint_path ./segmind_tiny-sd.safetensors --half --use_safetensors
```

The file segmind_tiny-sd.safetensors will be generated and is now ready for use with sd.cpp. You can follow a similar process for the other models mentioned above.

### SDXS-512-DreamShaper weights

Another very tiny and **incredibly fast**  model is SDXS by IDKiro et al.  The authors refer to it as *"Real-Time One-Step Latent Diffusion Models with Image Conditions"*. For details read the paper: https://arxiv.org/pdf/2403.16627 . Once again the authors removed some more blocks of U-Net part and unlike other SD1 models they use an adjusted _AutoEncoderTiny_ instead of default _AutoEncoderKL_ for the VAE part.

Some ready-to-run SDXS-512 model files are available online, such as:

* https://huggingface.co/akleine/sdxs-512
* https://huggingface.co/concedo/sdxs-512-tinySDdistilled-GGUF

### SDXS-512-0.9 weights

Even though the name "SDXS-512-0.9" is similar to "SDXS-512-DreamShaper", it is *completely different* but also **incredibly fast**. Sometimes it is preferred, so try it yourself.

Download a ready-to-run file from here:

* https://huggingface.co/akleine/sdxs-09

For the use of this model, both options `--cfg-scale 1` and `--steps 1` are again absolutely necessary.

## Usage

CLI examples use the [shared path and shell conventions](README.md#example-conventions).

### Web UI

Follow the [Web UI guide](../../examples/server/README.md#web-ui) to start the
server, then select the recipe matching your weights. Fill the component slots
below and click **Load models**.
Use the files and compatibility requirements in [Model files](#model-files).

| Recipe | Component slots |
| --- | --- |
| SDXL (`sdxl-checkpoint`) | `model`, `vae`, `tae` |
| SD 1.x (`sd1-checkpoint`) | `model`, `vae`, `tae` |
| SD 2.1 512 (`sd2-512-checkpoint`) | `model`, `vae`, `tae` |
| SDXS (`sdxs`) | `model`, `vae` |

Slot IDs correspond to CLI component options: `diffusion_model` uses
`--diffusion-model`, `llm` uses `--llm`, and `model` uses `-m` / `--model`.
Select optional components when required by your checkpoint or task.

### CLI

#### SSD1B

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe -m ..\models\checkpoints\SSD-1B-A1111.safetensors -p "a lovely cat"
```

**Linux/macOS**

```sh
./bin/sd-cli -m ../models/checkpoints/SSD-1B-A1111.safetensors -p "a lovely cat"
```

#### SDXS-512-DreamShaper

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  -m ..\models\checkpoints\sdxs.safetensors `
  -p "portrait of a lovely cat" `
  --cfg-scale 1 `
  --steps 1
```

**Linux/macOS**

```sh
./bin/sd-cli \
  -m ../models/checkpoints/sdxs.safetensors \
  -p "portrait of a lovely cat" \
  --cfg-scale 1 \
  --steps 1
```

Both options: `--cfg-scale 1` and  `--steps 1` are mandatory here.

## Notes and limitations

SSD-1B, Vega, and Tiny SD variants reduce the base model size. Details are in the
[SSD-1B paper](https://arxiv.org/abs/2401.02677v1) and
[Tiny SD paper](https://arxiv.org/pdf/2305.15798.pdf). Follow the [SDXS generation settings](#sdxs-512-dreamshaper); CFG 1 and one step are required.

Run the Python conversion commands in [Model files](#model-files) from the source repository root,
with the listed conversion script and Python dependencies installed. Return to
the build directory for the platform-specific CLI commands.
