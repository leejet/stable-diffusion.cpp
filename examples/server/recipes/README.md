# Built-in recipe policy

Read these rules before adding or changing JSON files in this directory. A recipe
represents a supported model-loading and capability configuration, not a saved
prompt, a style preset, or a collection of sampling settings.

## Required rules

1. **Community editing adapters get a separate recipe.** When the base model
   does not natively support instruction/reference-image editing and needs a
   community LoRA to gain it, provide a dedicated editing recipe. Keep editing
   inputs disabled in the base recipe. Examples: `anima` / `anima-edit` and
   `krea2-raw` / `krea2-raw-edit`.
2. **Split different task weights; share native multi-task weights.** If editing
   and text-to-image use different generation checkpoints, use separate recipes.
   If the same checkpoint natively supports both, use one recipe. Examples:
   `boogu-image` / `boogu-image-edit` are separate; `flux2-dev` and
   `qwen-image-21` each cover both generation and editing. Apply the same rule
   to video: separate HunyuanVideo 1.5 T2V/I2V and Wan 2.2 A14B T2V/I2V weights,
   but share one `wan22-ti2v-5b` recipe for its native T2V/I2V checkpoint.
3. **Capability-extending weights get a separate recipe.** ControlNet,
   IP-Adapter, PuLID, PhotoMaker, motion modules, and similar adapters that add
   a distinct conditioning or generation capability belong in dedicated
   recipes. Make the extension's model slots required when the schema provides
   them. Do not add these as optional slots on the ordinary base recipe.
4. **A LoRA alone is not a reason to split.** If adding a LoRA does not change
   supported functionality, reuse the base recipe. Style, character, quality,
   acceleration and distillation LoRAs do not get dedicated built-in recipes.
   Users select the LoRA and change steps, guidance or other generation settings
   separately. Standalone base/turbo/distilled checkpoints still follow the task,
   component and layer-count rules; a different filename alone is not a reason
   for another recipe.
5. **Keep native components in the model's recipe.** Native vision encoders,
   projectors, connectors and other components required by the model belong in
   its existing recipe. Splitting a native component into another file does not
   make it a capability-extending adapter under rule 3. For example, Qwen vision
   projectors and LLaDA/LTX connectors do not get separate recipes. A slot may be
   optional when its weights can be bundled in another selected file.
6. **Layer-count variants share the family recipe.** Adding, repeating, removing
   or pruning model layers does not justify a separate recipe when the runtime
   supports the variant with the same components and capabilities. This also
   applies to distilled checkpoints whose relevant difference is reduced depth.
   Use `flux1-dev` for FLUX.1 Lite, `sdxl-checkpoint` for SSD-1B and Segmind Vega,
   and the corresponding SD 1/2 checkpoint recipes for Tiny SD/Tiny SD2. Only an
   independent difference in task, required component compatibility or latent
   format can justify splitting such a variant; document that difference.

## Interpretation and review

- Instruction/reference-image editing is distinct from generic image-to-image
  denoising. The shared `img2img` capability does not imply native editing.
- Native components and external capability adapters are distinguished by their
  role, not by whether their weights are stored in a separate file. LLaDA's
  SigVQ-containing connectors remain in its shared generation/editing recipe.
- Do not merge models solely because they descend from the same family. Flex.2
  has different native inpainting/control conditioning, beyond its layer count
  ([model card](https://huggingface.co/ostris/Flex.2-preview)). SDXS uses an
  adjusted AutoEncoderTiny and a one-step model, rather than only a shallower
  ordinary SD checkpoint ([runtime guide](../../../docs/distilled_sd.md#sdxs-512-dreamshaper)).
  Their native components stay in their own model recipe.
- Quantization, file format, prompt wording, seeds, step counts and CFG values
  are not grounds for another recipe. Differences in required components,
  latent formats or supported capabilities must be documented.
- Evaluate what an adapter actually changes. Do not infer new functionality just
  from a LoRA's name or from the presence of another weight file. In particular,
  do not create an Omni-Kontext recipe solely because a LoRA is selected.
- A recipe does not load generation LoRAs automatically. Community-edit recipes
  must explain that the matching LoRA is required and must be selected separately.
- Only advertise features implemented by the current runtime. Check the relevant
  model documentation and code, and match each task's slots and feature flags to
  its weights. Record any unverified inference paths in the change description.
- Before submitting, validate all recipe JSON, inspect the embedded recipe API,
  and check that task-specific inputs appear only on the appropriate recipes.
  A catalog check is not a substitute for model-generation verification.

These are built-in catalog rules. Users may keep personal presets in
`<model-dir>/recipes/`; they need not add those presets to the built-in list.
See [the recipe schema and feature reference](../api.md#user-recipe-format)
for supported fields.

## Output dimension alignment

Set `size_multiple` when the runtime's output dimension alignment differs from
the default of `8`. Match `GenerationRequest::align_image_size` in
[`src/pipeline/request.cpp`](../../../src/pipeline/request.cpp): the VAE scale
factor multiplied by the model definition's `diffusion_down_factor`. The scale
comes from [`VAE::get_scale_factor`](../../../src/model/vae/vae.hpp), with the
PiD override in [`diffusion_engine.cpp`](../../../src/pipeline/diffusion_engine.cpp);
model geometry is defined under [`src/model/integrations`](../../../src/model/integrations/).

Use the runtime's output alignment, not the nominal patch size: some models pad
patches internally and crop their output back to the requested size. Pixel-space
models with an alignment of `1` explicitly use `size_multiple: 1`. MiniT2I also
has this alignment, but its separate fixed `512 × 512` output restriction still
applies. Reference-image preprocessing constraints are independent of this field.

## Video length constraints

Video recipes declare `video_constraints` to match `align_video_frames` in
[`diffusion_engine.cpp`](../../../src/pipeline/diffusion_engine.cpp) and the FPS
and AnimateDiff overrides in
[`request.cpp`](../../../src/pipeline/request.cpp). Wan, HunyuanVideo and LingBot
round down to `4n + 1`; LTX rounds down to `8n + 1`. MiniMax-H3 rounds up to
`17n + 5`, with at least 5 frames and fixed 24 FPS. AnimateDiff accepts whole
frames up to 32. Wan 2.2 S2V fixes FPS at 16.

Declare fixed FPS only where the runtime enforces it; recommended FPS belongs
in `defaults.fps`. These constraints guide UI conversion and estimates, not
runtime behavior. Audio-driven output and temporal upscaling may still change
the final frame count. See the [recipe schema](../api.md#user-recipe-format)
for field defaults and validation rules.
