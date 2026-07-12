# Image Editing

Image editing in `stable-diffusion.cpp` allows you to use reference images to guide the generation process, enabling tasks like identity preservation, style transfer, or layout modification.

## Supported Models

Depending on the architecture, different models handle reference images differently.

| Model | Default Preset |
| :--- | :--- | 
| [**FLUX.1-Kontext-dev**](./kontext.md) | `flux_kontext` |
| [**LongCat Image Edit**](./longcat_image.md) | `flux_kontext` |
| [**Qwen Image Edit**](./qwen_image_edit.md) | `qwen` |
| **Qwen Image LAYERED** | `qwen_layered` |
| [**Flux.2 [Dev] / Flux.2 [Klein]**](./flux2.md) | `flux2` |
| [**Boogu Image Edit**](./boogu_image.md) | `z_image_omni` |
| **Krea2 (Community Edit LoRAs)** | `krea2_ostris_edit` |

---

## Configuring Reference Modes (`--ref-image-mode`)

Different DiT-based editing models require different configurations to process reference images correctly (e.g., whether to use a Vision Language Model (VLM) encoder or pass VAE-encoded images directly to the DiT).

To simplify this, we provide **Presets**. By default, the system automatically selects the best preset based on the model architecture. However, you can override this using the `--ref-image-mode` argument.

### Usage
The `--ref-image-mode` argument accepts a comma-separated list of key-value pairs:

**Using a preset:**
`--ref-image-mode "preset=qwen_layered"`

**Using a preset with a specific override:**
`--ref-image-mode "preset=krea2_edit,force_timestep_0=true"`

### Available Presets

| Preset | Primary Use Case |
| :--- | :--- | 
| `flux_kontext` | FLUX.1 Kontext, LongCat Image Edit |
| `flux2` | FLUX.2 models |
| `qwen` | Qwen Image Edit |
| `qwen_layered` | Qwen Image Layered |
| `z_image_omni` | Boogu, Z-Image Omni |
| `krea2_ostris_edit` | Most Krea2 Community edit LoRAs (trained with Ostris script) |
| `krea2_edit` | Specifically for [lbouaraba/krea2edit](https://huggingface.co/conradlocke/krea2-identity-edit). (or similar) |
| `default` | Uses the automatic detection based on model architecture. |

---

## Advanced Parameter Reference

If presets are insufficient, you can manually configure the following parameters via `--ref-image-mode`:

| Key | Type | Description |
| :--- | :--- | :--- |
| `preset` | string | Sets a predefined group of parameters. |
| `use_vlm` | bool | Whether references are passed to the VLM encoder (if the model supports it). |
| `pass_to_dit` | bool | Whether VAE-encoded references are passed directly to the DiT. |
| `ref_index_mode` | enum | Behavior of the RoPE index: `fixed`, `increase`, or `decrease`. |
| `force_timestep_0` | bool | Forces timestep=0 for reference tokens (Krea2 architecture only). |
| `resize_vae_refs` | bool | Whether to resize VAE references. |
| `vae_refs_max_size` | int | Maximum pixel size for VAE references. |
| `cond_refs_resize_mode` | enum | How to resize condition references: `longest_side`, `area`, or `none`. |
| `cond_refs_max_size` | int | Maximum pixel size for condition references. |
| `cond_refs_min_size` | int | Minimum pixel size for condition references. |
| `cond_refs_size` | int | Shortcut to set both min and max size to the same value. |
