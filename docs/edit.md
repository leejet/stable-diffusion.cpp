# Image Editing

Image editing in `stable-diffusion.cpp` allows you to use reference images to guide the generation process, enabling tasks like identity preservation, style transfer, or layout modification.


## Supported Models

Depending on the architecture, different models handle reference images differently.

| Model | Default Preset |
| :--- | :--- |
| [**FLUX.1-Kontext-dev**](./kontext.md) | `flux_kontext` |
| [**LongCat Image Edit**](./longcat_image.md) | `longcat` |
| [**Qwen Image Edit**](./qwen_image_edit.md) | `qwen` |
| **Qwen Image LAYERED** | `qwen_layered` |
| [**Flux.2 [Dev] / Flux.2 [Klein]**](./flux2.md) | `flux2` |
| [**Boogu Image Edit**](./boogu_image.md) | `z_image_omni` |
| **Krea2 (Community Edit LoRAs)** | `krea2_ostris_edit` |
| [**Mage-Flow-Edit**](./mage_flow.md#image-editing) | `mage_flow` |
| **Anima (Community Edit LoRAs)** | `cosmos_reference` |

Stable-diffusion.spp also supports basic Unet-based editing models like instruct-pix2pix or CosXL-Edit. This document is not about those.

---

## Configuring Reference Modes (`--ref-image-args`)

Different DiT-based editing models require different configurations to process reference images correctly (e.g., whether to use a Vision Language Model (VLM) encoder or pass VAE-encoded images directly to the DiT).

To simplify this, we provide **Presets**. By default, the system automatically selects the best preset based on the model architecture. However, you can override this using the `--ref-image-args` argument.

### Usage
The `--ref-image-args` argument accepts a comma-separated list of key-value pairs:

**Using a preset:**
`--ref-image-args "preset=qwen_layered"`

**Using a preset with a specific override:**
`--ref-image-args "preset=krea2_edit,force_ref_timestep_zero=true"`

### Available Presets

| Preset | Primary Use Case |
| :--- | :--- |
| `flux_kontext` | FLUX.1 Kontext |
| `longcat` | LongCat Image Edit |
| `flux2` | FLUX.2 models |
| `qwen` | Qwen Image Edit |
| `qwen_layered` | Qwen Image Layered |
| `z_image_omni` | Boogu, Z-Image Omni |
| `krea2_ostris_edit` | Most Krea2 Community edit LoRAs (trained with Ostris script) |
| `mage_flow` | Mage-Flow-Edit |
| `krea2_edit` | Specifically for [lbouaraba/krea2edit](https://huggingface.co/conradlocke/krea2-identity-edit). (or similar) |
| `cosmos_reference` | For Anima |
| `default` | Uses the automatic detection based on model architecture. |

---

## Advanced Parameter Reference

If presets are insufficient, you can manually configure the following parameters via `--ref-image-args`:

| Key | Type | Description | Allowed Values |
| :--- | :--- | :--- | :--- |
| `preset` | string | Overrides the automatic preset. | (See the Presets table above) |
| `pass_to_vlm` | bool | Whether reference images are passed to the VLM encoder. | `true`, `false` |
| `pass_to_dit` | bool | Whether VAE-encoded references are passed directly to the DiT. | `true`, `false` |
| `ref_index_mode` | string | Behavior of the RoPE index. | `fixed`, `increase`, `decrease` |
| `force_ref_timestep_zero` | bool | Forces timestep=0 for reference tokens. | `true`, `false` (Krea2 only) |
| `resize_before_vae` | bool | Whether reference images are resized before VAE encoding. | `true`, `false` |
| `vae_input_max_pixels` | int | Maximum pixel area for VAE reference inputs. | Integer |
| `vlm_resize_mode` | string | How to resize VLM reference inputs. | `longest_side`, `area`, `none` |
| `vlm_max_size` | int | Maximum VLM input size; interpreted according to `vlm_resize_mode`. | Integer |
| `vlm_min_size` | int | Minimum VLM input size; interpreted according to `vlm_resize_mode`. | Integer |
| `vlm_size` | int | Shortcut to set both VLM min and max size to the same value. | Integer |

### Preset Default Values

For a technical overview of how each preset is configured, see the table below.

| Preset | VLM | RoPE Index | Cond Resize | Special Notes |
| :--- | :---: | :---: | :---: | :--- |
| `flux_kontext` | No | `fixed` | `none` | |
| `longcat` | Yes | `fixed` | `area` | |
| `flux2` | No | `increase` | `none` | |
| `qwen` | Yes | `increase` | `area` | |
| `qwen_layered` | Yes | `decrease` | `area` | |
| `mage_flow` | Yes | `increase` | `longest` | `vlm_max_size = 384`, VAE input resized to target |
| `z_image_omni` | Yes | `fixed` | `area` | |
| `krea2_ostris_edit`| Yes | `increase` | `area` | `force_ref_timestep_zero = true` |
| `krea2_edit` | Yes | `increase` | `longest` | `vlm_size = 768` |
| `cosmos_reference` | No | `fixed` | `none` | `resize_before_vae = false` |

**Additional Default Notes:**
- **VLM Input Sizes:** For most presets, `vlm_max_size` and `vlm_min_size` are set to `-1`, meaning the values are model-dependent and handled automatically. In `area` mode they represent pixel area; in `longest_side` mode they represent a side length in pixels.
- **VAE Input Size:** `vae_input_max_pixels` defaults to $1024 \times 1024$ pixels (`1048576`).
