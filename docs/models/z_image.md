# Z-Image

Z-Image and Z-Image-Turbo generate images from text using Qwen3 4B and the FLUX VAE.

You can run Z-Image with stable-diffusion.cpp on GPUs with 4GB of VRAM — or even less.

## Model files

| Component | Weights and downloads | CLI option | Requirement |
| --- | --- | --- | --- |
| Z-Image-Turbo | [Safetensors](https://huggingface.co/Comfy-Org/z_image_turbo/tree/main/split_files/diffusion_models)<br>[GGUF](https://huggingface.co/leejet/Z-Image-Turbo-GGUF/tree/main) | `--diffusion-model` | Choose matching variant |
| Z-Image | [Safetensors](https://huggingface.co/Comfy-Org/z_image/tree/main/split_files/diffusion_models)<br>[GGUF](https://huggingface.co/unsloth/Z-Image-GGUF/tree/main) | `--diffusion-model` | Choose matching variant |
| vae | [Safetensors](https://huggingface.co/black-forest-labs/FLUX.1-schnell/tree/main) | `--vae` | Use matching VAE |
| Qwen3 4b | [Safetensors](https://huggingface.co/Comfy-Org/z_image_turbo/tree/main/split_files/text_encoders)<br>[GGUF](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/tree/main) | `--llm` | Use matching text encoder |
| ControlNet | Z-Image control checkpoint with `control_all_x_embedder.2-1.weight` | `--control-net` or DGL `controlnet::load` | Optional. Match the DiT architecture |

## Usage

### Web UI

Follow the [Web UI guide](../../examples/server/README.md#web-ui) to start the
server, then select the recipe matching your weights. Fill the component slots
below and click **Load models**.
Use the files and compatibility requirements in [Model files](#model-files).

| Recipe | Component slots |
| --- | --- |
| Z-Image (`z-image`) | `diffusion_model`, `llm`, `vae`, `tae` |
| Z-Image Turbo (`z-image-turbo`) | `diffusion_model`, `llm`, `vae`, `tae` |

Slot IDs correspond to CLI component options: `diffusion_model` uses
`--diffusion-model`, `llm` uses `--llm`, and `model` uses `-m` / `--model`.
Select optional components when required by your checkpoint or task.

### DGL

| Example | Description |
| --- | --- |
| [Turbo: text to image](../dgl/z_image/z-image-turbo-txt2img.dgl) | Distilled generation. |
| [Base: text to image](../dgl/z_image/z-image-base-txt2img.dgl) | Base checkpoint. |
| [Turbo with ControlNet](../dgl/z_image/z-image-turbo-controlnet.dgl) | Load and apply control weights separately. |

#### Run a script

Edit paths, prompts and settings in the script's `var` block, then run from
the build directory. Paths inside the script resolve from the working directory.

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe -M dgl --script ..\docs\dgl\z_image\z-image-turbo-txt2img.dgl
```

**Linux/macOS**

```sh
./bin/sd-cli -M dgl --script ../docs/dgl/z_image/z-image-turbo-txt2img.dgl
```

Choose another script from the table to run a different task. Add `--check`
to validate syntax and types without loading weights. See the
[shared run instructions](../dgl/README.md#run-an-example) for build paths
and the server's file-access rules.

#### ControlNet

The [Z-Image ControlNet script](../dgl/z_image/z-image-turbo-controlnet.dgl)
loads the diffusion and control weights as separate models. Set the four
checkpoint paths and the prepared control image path in the script. The script
encodes the image with the FLUX VAE in posterior mode, converts it to diffusion
latents and pads the channel axis to 33 before calling `controlnet::apply`.
Each control model shares the DiT's compute backend. Repeated
`controlnet::apply` calls combine the controls before sampling. In the current
Z-Image model, control conditions and reference latents require separate
predictions.

### Legacy CLI

CLI examples use the [shared path and shell conventions](README.md#example-conventions).

#### Z-Image-Turbo

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  --diffusion-model ..\models\diffusion_models\z_image_turbo-Q3_K.gguf `
  --vae ..\models\vae\ae.sft `
  --llm ..\models\text_encoders\Qwen3-4B-Instruct-2507-Q4_K_M.gguf `
  -p "A cinematic, melancholic photograph of a solitary hooded figure walking through a sprawling, rain-slicked metropolis at night. The city lights are a chaotic blur of neon orange and cool blue, reflecting on the wet asphalt. The scene evokes a sense of being a single component in a vast machine. Superimposed over the image in a sleek, modern, slightly glitched font is the philosophical quote: 'THE CITY IS A CIRCUIT BOARD, AND I AM A BROKEN TRANSISTOR.' -- moody, atmospheric, profound, dark academic" `
  --cfg-scale 1.0 `
  --steps 8 `
  -H 1024 `
  -W 512 `
  --offload-to-cpu `
  -v
```

**Linux/macOS**

```sh
./bin/sd-cli \
  --diffusion-model ../models/diffusion_models/z_image_turbo-Q3_K.gguf \
  --vae ../models/vae/ae.sft \
  --llm ../models/text_encoders/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  -p "A cinematic, melancholic photograph of a solitary hooded figure walking through a sprawling, rain-slicked metropolis at night. The city lights are a chaotic blur of neon orange and cool blue, reflecting on the wet asphalt. The scene evokes a sense of being a single component in a vast machine. Superimposed over the image in a sleek, modern, slightly glitched font is the philosophical quote: 'THE CITY IS A CIRCUIT BOARD, AND I AM A BROKEN TRANSISTOR.' -- moody, atmospheric, profound, dark academic" \
  --cfg-scale 1.0 \
  --steps 8 \
  -H 1024 \
  -W 512 \
  --offload-to-cpu \
  -v
```

<img width="256" alt="z-image example" src="../../assets/z_image/q3_K.png" />

#### Z-Image-Base

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  --diffusion-model ..\models\diffusion_models\z_image_bf16.safetensors `
  --vae ..\models\vae\ae.sft `
  --llm ..\models\text_encoders\qwen_3_4b.safetensors `
  -p "A cinematic, melancholic photograph of a solitary hooded figure walking through a sprawling, rain-slicked metropolis at night. The city lights are a chaotic blur of neon orange and cool blue, reflecting on the wet asphalt. The scene evokes a sense of being a single component in a vast machine. Superimposed over the image in a sleek, modern, slightly glitched font is the philosophical quote: 'THE CITY IS A CIRCUIT BOARD, AND I AM A BROKEN TRANSISTOR.' -- moody, atmospheric, profound, dark academic" `
  --cfg-scale 5.0 `
  -H 1024 `
  -W 512 `
  --offload-to-cpu `
  -v
```

**Linux/macOS**

```sh
./bin/sd-cli \
  --diffusion-model ../models/diffusion_models/z_image_bf16.safetensors \
  --vae ../models/vae/ae.sft \
  --llm ../models/text_encoders/qwen_3_4b.safetensors \
  -p "A cinematic, melancholic photograph of a solitary hooded figure walking through a sprawling, rain-slicked metropolis at night. The city lights are a chaotic blur of neon orange and cool blue, reflecting on the wet asphalt. The scene evokes a sense of being a single component in a vast machine. Superimposed over the image in a sleek, modern, slightly glitched font is the philosophical quote: 'THE CITY IS A CIRCUIT BOARD, AND I AM A BROKEN TRANSISTOR.' -- moody, atmospheric, profound, dark academic" \
  --cfg-scale 5.0 \
  -H 1024 \
  -W 512 \
  --offload-to-cpu \
  -v
```

<img width="256" alt="z-image example" src="../../assets/z_image/base_bf16.png" />

#### Comparison of Different Quantization Types

| bf16 | q8_0 | q6_K | q5_0 | q4_K | q4_0 | q3_K | q2_K|
|---|---|---|---|---|---|---|---|
| <img width="256" alt="bf16" src="../../assets/z_image/bf16.png" /> | <img width="256" alt="q8_0" src="../../assets/z_image/q8_0.png" /> | <img width="256" alt="q6_K" src="../../assets/z_image/q6_K.png" /> | <img width="256" alt="q5_0" src="../../assets/z_image/q5_0.png" />  | <img width="256" alt="q4_K" src="../../assets/z_image/q4_K.png" /> | <img width="256" alt="q4_0" src="../../assets/z_image/q4_0.png" /> | <img width="256" alt="q3_K" src="../../assets/z_image/q3_K.png" /> | <img width="256" alt="q2_K" src="../../assets/z_image/q2_K.png" /> |
