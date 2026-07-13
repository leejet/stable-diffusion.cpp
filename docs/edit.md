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
| **Anima (Community Edit LoRAs)** | `cosmos_reference` |

Stable-diffusion.spp also supports basic Unet-based editing models like instruct-pix2pix or CosXL-Edit. This document is not about those.

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
| `cosmos_reference` | For Anima |
| `default` | Uses the automatic detection based on model architecture. |

---

## Advanced Parameter Reference

If presets are insufficient, you can manually configure the following parameters via `--ref-image-mode`:

| Key | Type | Description | Allowed Values |
| :--- | :--- | :--- | :--- |
| `preset` | string | Overrides the automatic preset. | (See the Presets table above) |
| `use_vlm` | bool | Whether references are passed to the VLM encoder. | `true`, `false` |
| `pass_to_dit` | bool | Whether VAE-encoded references are passed directly to the DiT. | `true`, `false` |
| `ref_index_mode` | string | Behavior of the RoPE index. | `fixed`, `increase`, `decrease` |
| `force_timestep_0` | bool | Forces timestep=0 for reference tokens. | `true`, `false` (Krea2 only) |
| `resize_vae_refs` | bool | Whether to resize VAE references. | `true`, `false` |
| `vae_refs_max_size` | int | Maximum pixel size for VAE references. | Integer |
| `cond_refs_resize_mode` | string | How to resize condition references. | `longest_side`, `area`, `none` |
| `cond_refs_max_size` | int | Maximum pixel size for condition references. | Integer |
| `cond_refs_min_size` | int | Minimum pixel size for condition references. | Integer |
| `cond_refs_size` | int | Shortcut to set both min and max size to the same value. | Integer |

### Preset Default Values

For a technical overview of how each preset is configured, see the table below.

| Preset | VLM | RoPE Index | Cond Resize | Special Notes |
| :--- | :---: | :---: | :---: | :--- |
| `flux_kontext` | No | `fixed` | `none` | |
| `flux2` | No | `increase` | `none` | |
| `qwen` | Yes | `increase` | `area` | |
| `qwen_layered` | Yes | `decrease` | `area` | |
| `z_image_omni` | Yes | `fixed` | `area` | |
| `krea2_ostris_edit`| Yes | `increase` | `area` | `force_timestep_0 = true` |
| `krea2_edit` | Yes | `increase` | `longest` | `cond_refs_size = 768` |
| `cosmos_reference` | No | `fixed` | `none` | `resize_vae_refs = false` |

**Additional Default Notes:**
- **Condition Sizes:** For most presets, `cond_refs_max_size` and `cond_refs_min_size` are set to `-1`, meaning the values are model-dependent and handled automatically.
- **VAE Reference Size:** `vae_refs_max_size` defaults to $1024 \times 1024$ pixels (`1048576`).
