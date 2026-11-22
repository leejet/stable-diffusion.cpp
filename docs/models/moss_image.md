# MOSS-Image

MOSS-Image uses a 5B diffusion transformer, the FLUX.2 VAE, and Qwen3.5 9B as the LLM text encoder. Base and SFT support text-to-image generation. The default reference-guided workflow uses MOSS-Image-Edit-HQ with SFT. The chinoll LoRA has a dedicated DGL example for Base or SFT.

## Model files

| Component | Weights and downloads | CLI option | Requirement |
| --- | --- | --- | --- |
| MOSS-Image Base or SFT | [Safetensors](https://huggingface.co/Comfy-Org/MOSS-Image/tree/main/diffusion_models)<br>gguf:<br>Base: [Download](https://huggingface.co/leejet/MOSS-Image-Base-GGUF/tree/main)<br>SFT: [Download](https://huggingface.co/leejet/MOSS-Image-SFT-GGUF/tree/main) | `--diffusion-model` | Required |
| vae | [Safetensors](https://huggingface.co/Comfy-Org/MOSS-Image/tree/main/vae) | `--vae` | Use matching VAE |
| Qwen3.5 9B | [GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/tree/main) | `--llm` | Use matching text encoder |
| `tokenizer.json` | [Download](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base/tree/main/tokenizer) | `--tokenizer` | Required; match text encoder |
| Qwen3.5 9B vision projector | Matching Qwen3.5 9B `mmproj` or an encoder with embedded vision weights | `--llm_vision` | Required for HQ visual prompt conditioning and inline prompt images |
| MOSS-Image-Edit-HQ LoRA | [`armV5-moss-edit-lora.safetensors`](https://huggingface.co/akhaliq/MOSS-Image-Edit-HQ/tree/main) | `--lora-model-dir` and `<lora:moss/armV5-moss-edit-lora:1.0>` in the prompt | Required for the default editing example |
| chinoll Moss-Image-Edit LoRA | [`adapter_model.safetensors`](https://huggingface.co/chinoll/Moss-Image-Edit/tree/main) | `lora_path` in the chinoll DGL script | Required for the chinoll workflow |

Pass the external `tokenizer.json` with `--tokenizer`; Qwen3.5 uses a different vocabulary from the embedded Qwen tokenizer.

For [inline prompt images](../guides/prompt_images.md), use a matching Qwen3.5
vision projector. Pass it with `--llm_vision` in the CLI or `vision_path` on
`load_text_encoder` in DGL. See the
[DGL example](../dgl/moss_image/moss-image-sft-inline-images.dgl).

## Usage

### Web UI

Follow the [Web UI guide](../../examples/server/README.md#web-ui) to start the
server, then select the recipe matching your weights. Fill the component slots
below and click **Load models**.
Use the files and compatibility requirements in [Model files](#model-files).

| Recipe | Component slots |
| --- | --- |
| MOSS Image Base (`moss-image-base`) | `diffusion_model`, `llm`, `tokenizer`, `vae`, `tae` |
| MOSS Image SFT (`moss-image-sft`) | `diffusion_model`, `llm`, `tokenizer`, `vae`, `tae` |

Slot IDs correspond to CLI component options: `diffusion_model` uses
`--diffusion-model`, `llm` uses `--llm`, and `model` uses `-m` / `--model`.
Select optional components when required by your checkpoint or task.

### DGL

| Example | Description |
| --- | --- |
| [Base: text to image](../dgl/moss_image/moss-image-base-txt2img.dgl) | MOSS-Image Base. |
| [SFT: text to image](../dgl/moss_image/moss-image-sft-txt2img.dgl) | MOSS-Image SFT. |
| [Inline images](../dgl/moss_image/moss-image-sft-inline-images.dgl) | Insert named images into the prompt. |
| [MOSS-Image-Edit-HQ](../dgl/moss_image/moss-image-edit.dgl) | Edit with SFT and the HQ LoRA. |
| [chinoll Moss-Image-Edit](../dgl/moss_image/moss-image-edit-chinoll.dgl) | Edit with Base or SFT and the chinoll LoRA. |

The HQ script uses the same input and settings as the [CLI edit example](#moss-image-edit).
It passes the references to both the FLUX.2 VAE and Qwen3.5 vision encoder. The
chinoll script passes them to the VAE and uses `ref_rope_anchor = "text_start"`.
Both scripts cap the reference long edge at 1024 pixels and align both dimensions
to 16 for VAE encoding. Change `lora_path` and `reference_paths` to use your
files. Use `(phrase:1.2)` to weight a phrase in the prompt.

#### Run a script

Edit paths, prompts and settings in the script's `var` block, then run from
the build directory. Paths inside the script resolve from the working directory.

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe -M dgl --script ..\docs\dgl\moss_image\moss-image-base-txt2img.dgl
```

**Linux/macOS**

```sh
./bin/sd-cli -M dgl --script ../docs/dgl/moss_image/moss-image-base-txt2img.dgl
```

Choose another script from the table to run a different task. Add `--check`
to validate syntax and types without loading weights. See the
[shared run instructions](../dgl/README.md#run-an-example) for build paths
and the server's file-access rules.

### Legacy CLI

CLI examples use the [shared path and shell conventions](README.md#example-conventions).

#### MOSS-Image Base

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  --diffusion-model ..\models\diffusion_models\moss_image_base_int8_convrot_attn.safetensors `
  --llm ..\models\text_encoders\Qwen3.5-9B-Q4_K_M.gguf `
  --vae ..\models\vae\flux2_ae.safetensors `
  --tokenizer ..\models\tokenizers\tokenizer_qwen3.5.json `
  -p "a lovely cat holding a sign says 'moss-image.cpp'" `
  --cfg-scale 3.0 `
  -H 1024 `
  -W 1024 `
  --offload-to-cpu `
  -v
```

**Linux/macOS**

```sh
./bin/sd-cli \
  --diffusion-model ../models/diffusion_models/moss_image_base_int8_convrot_attn.safetensors \
  --llm ../models/text_encoders/Qwen3.5-9B-Q4_K_M.gguf \
  --vae ../models/vae/flux2_ae.safetensors \
  --tokenizer ../models/tokenizers/tokenizer_qwen3.5.json \
  -p "a lovely cat holding a sign says 'moss-image.cpp'" \
  --cfg-scale 3.0 \
  -H 1024 \
  -W 1024 \
  --offload-to-cpu \
  -v
```

#### MOSS-Image SFT

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  --diffusion-model ..\models\diffusion_models\moss_image_sft_int8_convrot_attn.safetensors `
  --llm ..\models\text_encoders\Qwen3.5-9B-Q4_K_M.gguf `
  --vae ..\models\vae\flux2_ae.safetensors `
  --tokenizer ..\models\tokenizers\tokenizer_qwen3.5.json `
  -p "a lovely cat holding a sign says 'moss-image.cpp'" `
  --cfg-scale 3.0 `
  -H 1024 `
  -W 1024 `
  --offload-to-cpu `
  -v
```

**Linux/macOS**

```sh
./bin/sd-cli \
  --diffusion-model ../models/diffusion_models/moss_image_sft_int8_convrot_attn.safetensors \
  --llm ../models/text_encoders/Qwen3.5-9B-Q4_K_M.gguf \
  --vae ../models/vae/flux2_ae.safetensors \
  --tokenizer ../models/tokenizers/tokenizer_qwen3.5.json \
  -p "a lovely cat holding a sign says 'moss-image.cpp'" \
  --cfg-scale 3.0 \
  -H 1024 \
  -W 1024 \
  --offload-to-cpu \
  -v
```

<img src="../../assets/moss/sft.png" />

#### MOSS-Image Edit

Save the HQ LoRA as `models/loras/moss/armV5-moss-edit-lora.safetensors`, or
adjust the LoRA tag below. The CLI and C API use the HQ layout by default.
Pass `-r` once per reference in the CLI, or set `ref_images` in the C API.
Set `llm_vision_path` in the C API for a separate vision projector. A text-only
encoder still supplies the text condition and the VAE supplies reference latents.

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe `
  --diffusion-model ..\models\diffusion_models\moss_image_sft_int8_convrot_attn.safetensors `
  --llm ..\models\text_encoders\Qwen3.5-9B-Q4_K_M.gguf `
  --llm_vision ..\models\text_encoders\Qwen3.5-9B-mmproj-BF16.gguf `
  --vae ..\models\vae\flux2_ae.safetensors `
  --tokenizer ..\models\tokenizers\tokenizer_qwen3.5.json `
  --lora-model-dir ..\models\loras `
  -r ..\assets\flux\flux1-dev-q8_0.png `
  -p "change 'flux.cpp' to 'moss.cpp'<lora:moss/armV5-moss-edit-lora:1.0>" `
  --cfg-scale 3.0 `
  -v
```

**Linux/macOS**

```sh
./bin/sd-cli \
  --diffusion-model ../models/diffusion_models/moss_image_sft_int8_convrot_attn.safetensors \
  --llm ../models/text_encoders/Qwen3.5-9B-Q4_K_M.gguf \
  --llm_vision ../models/text_encoders/Qwen3.5-9B-mmproj-BF16.gguf \
  --vae ../models/vae/flux2_ae.safetensors \
  --tokenizer ../models/tokenizers/tokenizer_qwen3.5.json \
  --lora-model-dir ../models/loras \
  -r ../assets/flux/flux1-dev-q8_0.png \
  -p "change 'flux.cpp' to 'moss.cpp'<lora:moss/armV5-moss-edit-lora:1.0>" \
  --cfg-scale 3.0 \
  -v
```

## Notes and limitations

Use output dimensions divisible by 16. The default discrete scheduler uses a
flow shift of 5. Use `--flow-shift` to override it.

Text-to-image and HQ editing skip layer 22 in the negative predictor. The
chinoll workflow uses all layers in both CFG branches. [HQ demo](https://huggingface.co/spaces/akhaliq/moss-image-edit-hq-demo/blob/main/app.py) ·
[chinoll sampler](https://github.com/Chenkin-x/Moss-Image-Edit/blob/main/moss_edit/sampling.py)
