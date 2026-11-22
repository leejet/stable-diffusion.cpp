# Image editing

Reference images can guide identity, appearance, layout and instruction editing.
Pass them with `-r IMAGE` in the CLI or `ref_images` in the C API. The generated
DGL follows the selected model's default reference policy.

`--img-cfg-scale` (`sample.guidance.img_cfg` in the C API) controls image guidance
for inpainting, UNet editing and supported reference-latent models such as
FLUX, Qwen Image Edit, LongCat and Mage Flow. A value other than 1 uses separate
positive, negative-with-image and negative-without-image predictions. The
generated DGL preserves this split during hires refinement. Models or inputs
that do not support image guidance report an error.

## Model defaults

| Model | Reference handling |
| --- | --- |
| FLUX.1 Kontext | VAE reference latents with fixed position indexing |
| FLUX.2 / FLUX.3 Klein | VAE reference latents with increasing position indexing |
| Qwen Image Edit | Vision inputs and VAE reference latents |
| Qwen Image 2.1 | Vision inputs derived from the VAE reference pixels |
| Qwen Image Layered | References use decreasing position indexing |
| Fibo 2.0 | Shared center crops for VAE and vision encoding; multiple references use aspect-ratio buckets |
| [Anima Edit](../models/anima.md) | Matching community editing LoRA; VAE reference latents |
| [Krea2 Edit](../models/krea2.md) | Matching community editing LoRA; vision inputs and VAE reference latents |
| [MOSS-Image Edit](../models/moss_image.md#moss-image-edit) | HQ LoRA with VAE reference latents and optional vision conditioning |
| LLaDA-Image | Reference pixels go to the text encoder's native image components and reference latents to the diffusion model |
| MiniMax-H3 | Aspect-preserving references aligned to the model grid |

See the [model guides](../models/README.md) for required components and supported
editing checkpoints. Select community editing LoRAs separately; reference-image
inputs and editing recipes do not load them automatically.
Instruct-Pix2Pix and CosXL-Edit also use `-r` for their source
image; they construct concatenated image conditioning rather than DiT reference
token sequences.

## Custom reference processing

Reference-policy overrides are not native generation parameters. Write a DGL
program to choose resizing, cropping, model inputs and reference position rules.
`--ref-image-args` and the C API `ref_image_args` key have been removed.

For example, a script can prepare its VAE input explicitly:

```dgl
source = image::load("reference.png")
pixels = image::to_f32(tensor::interpolate(source, size = [1024, 768], axes = [1, 2]))
reference = vae_to_dit_latents(vae, vae_encode(vae, pixels, rng = rng))
condition = encode_prompt(text, prompt, media = [pixels])
pos = predictor(dit, condition, ref_latents = [reference], ref_index_mode = "increase")
```

These lines belong inside a complete DGL program with explicit model and RNG
loading. Start from the [Qwen Image 2.1 edit example](../dgl/qwen_image_2.1/qwen-image-2-1-edit.dgl).
Vision and VAE inputs can be prepared separately when a workflow needs different
reference images on those paths. Required model alignment and encoder-native
preprocessing still apply.
