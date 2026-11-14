# Qwen Image Edit

Qwen Image Edit supports reference-image editing with the original, 2509, and 2511 checkpoints.

## Model files

| Component | Weights and downloads | CLI option | Requirement |
| --- | --- | --- | --- |
| Qwen Image | Qwen Image Edit<br>[Safetensors](https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/tree/main/split_files/diffusion_models)<br>[GGUF](https://huggingface.co/QuantStack/Qwen-Image-Edit-GGUF/tree/main)<br>Qwen Image Edit 2509<br>[Safetensors](https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/tree/main/split_files/diffusion_models)<br>[GGUF](https://huggingface.co/QuantStack/Qwen-Image-Edit-2509-GGUF/tree/main)<br>Qwen Image Edit 2511<br>[Safetensors](https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/tree/main/split_files/diffusion_models)<br>[GGUF](https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF/tree/main) | `--diffusion-model` | Required |
| vae | [Safetensors](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/tree/main/split_files/vae) | `--vae` | Use matching VAE |
| qwen_2.5_vl 7b | [Safetensors](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/tree/main/split_files/text_encoders)<br>[GGUF](https://huggingface.co/mradermacher/Qwen2.5-VL-7B-Instruct-GGUF/tree/main) | `--llm` | Use matching text encoder |

## Usage

CLI examples use the [shared path and shell conventions](README.md#example-conventions).

### Web UI

Follow the [Web UI guide](../../examples/server/README.md#web-ui) to start the
server, then select the recipe matching your weights. Fill the component slots
below and click **Load models**.
Use the files and compatibility requirements in [Model files](#model-files).

| Recipe | Component slots |
| --- | --- |
| Qwen Image Edit (`qwen-image-edit`) | `diffusion_model`, `llm`, `llm_vision`, `vae`, `tae` |
| Qwen Image Edit 2509 (`qwen-image-edit-2509`) | `diffusion_model`, `llm`, `llm_vision`, `vae`, `tae` |
| Qwen Image Edit 2511 (`qwen-image-edit-2511`) | `diffusion_model`, `llm`, `llm_vision`, `vae`, `tae` |

Slot IDs correspond to CLI component options: `diffusion_model` uses
`--diffusion-model`, `llm` uses `--llm`, and `model` uses `-m` / `--model`.
Select optional components when required by your checkpoint or task.

### CLI

#### Qwen Image Edit

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  --diffusion-model ..\models\diffusion_models\Qwen_Image_Edit-Q8_0.gguf `
  --vae ..\models\vae\qwen_image_vae.safetensors `
  --llm ..\models\text_encoders\qwen_2.5_vl_7b.safetensors `
  -r ..\inputs\flux\flux1-dev-q8_0.png `
  -p "change 'flux.cpp' to 'edit.cpp'" `
  --cfg-scale 2.5 `
  --sampling-method euler `
  --flow-shift 3 `
  --offload-to-cpu `
  -v `
  --seed 1118877715456453
```

**Linux/macOS**

```sh
./bin/sd-cli \
  --diffusion-model ../models/diffusion_models/Qwen_Image_Edit-Q8_0.gguf \
  --vae ../models/vae/qwen_image_vae.safetensors \
  --llm ../models/text_encoders/qwen_2.5_vl_7b.safetensors \
  -r ../inputs/flux/flux1-dev-q8_0.png \
  -p "change 'flux.cpp' to 'edit.cpp'" \
  --cfg-scale 2.5 \
  --sampling-method euler \
  --flow-shift 3 \
  --offload-to-cpu \
  -v \
  --seed 1118877715456453
```

<img alt="qwen_image_edit" src="../../assets/qwen/qwen_image_edit.png" />

#### Qwen Image Edit 2509

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  --diffusion-model ..\models\diffusion_models\Qwen-Image-Edit-2509-Q4_K_S.gguf `
  --vae ..\models\vae\qwen_image_vae.safetensors `
  --llm ..\models\text_encoders\Qwen2.5-VL-7B-Instruct-Q8_0.gguf `
  --llm_vision ..\models\text_encoders\Qwen2.5-VL-7B-Instruct.mmproj-Q8_0.gguf `
  -r ..\inputs\flux\flux1-dev-q8_0.png `
  -p "change 'flux.cpp' to 'Qwen Image Edit 2509'" `
  --cfg-scale 2.5 `
  --sampling-method euler `
  --flow-shift 3 `
  --offload-to-cpu `
  -v
```

**Linux/macOS**

```sh
./bin/sd-cli \
  --diffusion-model ../models/diffusion_models/Qwen-Image-Edit-2509-Q4_K_S.gguf \
  --vae ../models/vae/qwen_image_vae.safetensors \
  --llm ../models/text_encoders/Qwen2.5-VL-7B-Instruct-Q8_0.gguf \
  --llm_vision ../models/text_encoders/Qwen2.5-VL-7B-Instruct.mmproj-Q8_0.gguf \
  -r ../inputs/flux/flux1-dev-q8_0.png \
  -p "change 'flux.cpp' to 'Qwen Image Edit 2509'" \
  --cfg-scale 2.5 \
  --sampling-method euler \
  --flow-shift 3 \
  --offload-to-cpu \
  -v
```

<img alt="qwen_image_edit_2509" src="../../assets/qwen/qwen_image_edit_2509.png" />

#### Qwen Image Edit 2511

To use the new Qwen Image Edit 2511 mode, `--model-args qwen_image_zero_cond_t=true` must be enabled; otherwise, image editing quality will degrade significantly.

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  --diffusion-model ..\models\diffusion_models\qwen-image-edit-2511-Q4_K_M.gguf `
  --vae ..\models\vae\qwen_image_vae.safetensors `
  --llm ..\models\text_encoders\qwen_2.5_vl_7b.safetensors `
  --model-args qwen_image_zero_cond_t=true `
  -r ..\inputs\flux\flux1-dev-q8_0.png `
  -p "change 'flux.cpp' to 'edit.cpp'" `
  --cfg-scale 2.5 `
  --sampling-method euler `
  --flow-shift 3 `
  --offload-to-cpu `
  -v
```

**Linux/macOS**

```sh
./bin/sd-cli \
  --diffusion-model ../models/diffusion_models/qwen-image-edit-2511-Q4_K_M.gguf \
  --vae ../models/vae/qwen_image_vae.safetensors \
  --llm ../models/text_encoders/qwen_2.5_vl_7b.safetensors \
  --model-args qwen_image_zero_cond_t=true \
  -r ../inputs/flux/flux1-dev-q8_0.png \
  -p "change 'flux.cpp' to 'edit.cpp'" \
  --cfg-scale 2.5 \
  --sampling-method euler \
  --flow-shift 3 \
  --offload-to-cpu \
  -v
```

<img alt="qwen_image_edit_2509" src="../../assets/qwen/qwen_image_edit_2511.png" />
