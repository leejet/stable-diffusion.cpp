# LongCat Image

LongCat-Image uses a LongCat diffusion transformer, the FLUX VAE, and Qwen2.5-VL as the LLM text encoder.

## Model files

| Component | Weights and downloads | CLI option | Requirement |
| --- | --- | --- | --- |
| LongCat Image | [Safetensors](https://huggingface.co/Comfy-Org/LongCat-Image/tree/main/split_files/diffusion_models)<br>[GGUF](https://huggingface.co/vantagewithai/LongCat-Image-GGUF/tree/main/comfy) | `--diffusion-model` | Choose matching variant |
| LongCat Image Edit | LongCat Image Edit Turbo: [Download](https://huggingface.co/meituan-longcat/LongCat-Image-Edit-Turbo)<br>[GGUF](https://huggingface.co/vantagewithai/LongCat-Image-Edit-GGUF/tree/main) | `--diffusion-model` | Choose matching variant |
| vae | [Safetensors](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/ae.safetensors) | `--vae` | Use matching VAE |
| qwen_2.5_vl 7b | [Safetensors](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/tree/main/split_files/text_encoders)<br>[GGUF](https://huggingface.co/mradermacher/Qwen2.5-VL-7B-Instruct-GGUF/tree/main)<br>For image editing with GGUF text encoders, also download the matching mmproj file and pass it with `--llm_vision`. | `--llm` | Use matching text encoder |

## Usage

CLI examples use the [shared path and shell conventions](README.md#example-conventions).

### Web UI

Follow the [Web UI guide](../../examples/server/README.md#web-ui) to start the
server, then select the recipe matching your weights. Fill the component slots
below and click **Load models**.
Use the files and compatibility requirements in [Model files](#model-files).

| Recipe | Component slots |
| --- | --- |
| LongCat Image (`longcat-image`) | `diffusion_model`, `llm`, `vae`, `tae` |
| LongCat Image Edit (`longcat-image-edit`) | `diffusion_model`, `llm`, `vae`, `tae` |

Slot IDs correspond to CLI component options: `diffusion_model` uses
`--diffusion-model`, `llm` uses `--llm`, and `model` uses `-m` / `--model`.
Select optional components when required by your checkpoint or task.

### CLI

LongCat uses quoted text for character-level text rendering. Put target text inside single quotes, double quotes, or Chinese quotes.

#### LongCat Image

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  --diffusion-model ..\models\diffusion_models\LongCat-Image-Q4_K_M.gguf `
  --vae ..\models\vae\ae.sft `
  --llm ..\models\text_encoders\Qwen2.5-VL-7B-Instruct-Q8_0.gguf `
  -p "a lovely cat holding a sign says 'longcat.cpp'" `
  --cfg-scale 5.0 `
  --sampling-method euler `
  --flow-shift 3 `
  --offload-to-cpu `
  -v
```

**Linux/macOS**

```sh
./bin/sd-cli \
  --diffusion-model ../models/diffusion_models/LongCat-Image-Q4_K_M.gguf \
  --vae ../models/vae/ae.sft \
  --llm ../models/text_encoders/Qwen2.5-VL-7B-Instruct-Q8_0.gguf \
  -p "a lovely cat holding a sign says 'longcat.cpp'" \
  --cfg-scale 5.0 \
  --sampling-method euler \
  --flow-shift 3 \
  --offload-to-cpu \
  -v
```

<img alt="longcat example" src="../../assets/longcat/example.png" />
