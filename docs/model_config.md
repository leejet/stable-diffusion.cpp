# Model Configuration Conventions

This document describes model configuration structs, weight-based configuration
detection, and the definitions used to integrate models into the runtime.

## Config Types

Model configuration should live in a model-specific `*Config` struct.

Examples:

- `ZImageConfig`
- `UNetConfig`
- `MMDiTConfig`
- `LLMConfig`

Preserve established acronym casing in type names, such as `UNet`, `MMDiT`,
`LLM`, `VAE`, and `T5`.

Place the config struct near the top of the model header, before the main model
blocks and runner types that consume it.

## Config Variables

Variables and members that hold a config should be named `config`.

Examples:

```cpp
UNetConfig config;
UnetModelBlock unet;

MMDiTRunner(...)
    : DiffusionModelRunner(backend, params_backend, prefix),
      config(MMDiTConfig::detect_from_weights(tensor_storage_map, prefix)),
      mmdit(config) {
}
```

Avoid alternate names such as `params`, `params_cfg`, `model_params`, or
model-specific aliases unless an existing public API requires them.

## Weight Detection

If a model can derive configuration from loaded weight metadata, expose that
logic as a static method on the config type:

```cpp
static XxxConfig detect_from_weights(const String2TensorStorage& tensor_storage_map,
                                     const std::string& prefix);
```

Additional selector arguments are allowed when required by an existing model
family, for example `SDVersion version` or an architecture enum:

```cpp
static UNetConfig detect_from_weights(const String2TensorStorage& tensor_storage_map,
                                      const std::string& prefix,
                                      SDVersion version = VERSION_SD1);
```

Use `TensorStorage` metadata, especially `n_dims` and `ne`, to infer shapes.
Do not load or parse tensor data for config detection.

Detection should respect `prefix`. For nested weights, construct full names from
`prefix + "." + suffix` or filter entries with `starts_with(name, prefix)`.

Do not add persistent config fields such as `inferred_from_weights` only to
record whether detection happened. If the function needs to decide whether to
print a verbose line, keep that as local control flow inside `detect_from_weights`.

## Logging

When config values are inferred from weights, print one `LOG_VERBOSE` line at the
end of `detect_from_weights`.

Example:

```cpp
LOG_VERBOSE("llm: num_layers = %" PRId64 ", vocab_size = %" PRId64 ", hidden_size = %" PRId64 ", intermediate_size = %" PRId64,
            config.num_layers,
            config.vocab_size,
            config.hidden_size,
            config.intermediate_size);
```

Only print the config detection log when the function actually inferred values
from weights. Do not duplicate the same config summary in runner constructors or
model loading code.

Use the correct format specifiers for field types, such as `%" PRId64 "` for
`int64_t` and `%d` for `int`.

## Runner And Model Responsibilities

Runners should detect the config once and pass it into the model block:

```cpp
struct XxxRunner : public DiffusionModelRunner {
    XxxConfig config;
    XxxModel model;

    XxxRunner(..., const String2TensorStorage& tensor_storage_map, const std::string prefix)
        : DiffusionModelRunner(backend, params_backend, prefix),
          config(XxxConfig::detect_from_weights(tensor_storage_map, prefix)),
          model(config) {
        model.init(params_ctx, tensor_storage_map, prefix);
    }
};
```

Model blocks should consume `config` directly instead of re-scanning weights in
their constructors. Keep config-derived behavior centralized in the config
struct.

If a model has no weight-derived config today, it may still provide
`detect_from_weights` for API consistency, but it should not print a config
detection log unless it actually derives values from weights.

## Model Integration Definitions

Every `SDVersion` has an explicit `ModelDefinition` in
`src/model/integrations/`. Related variants share an integration file and runner
factory. Network blocks and their weight-derived configuration remain under
`src/model/diffusion/`; reusable text encoder implementations remain under
`src/model/te/`. Weight-name conversion remains in `src/name_conversion.cpp`.

To add a model:

1. Add its identity to `SDVersion` without renumbering existing values.
2. Implement the network and `*Config::detect_from_weights` following the
   conventions above, or reuse an existing runner.
3. Add a definition in its integration file, including its component
   factory, traits, geometry, and sampling policy.
4. Add the family's explicit registration in `src/model_registry.cpp`. For a new
   variant of a registered family, add its definition to that family's list.
5. Add version detection signatures and any required weight-name mappings in
   `src/model_loader.cpp` and `src/name_conversion.cpp` respectively. Update shared
   conditioning or runtime code only when existing behavior cannot serve the model.

Use the existing family integration as the starting point for a variant. For a
new family, `src/model/integrations/moss_image.cpp` is a compact example of a
conditioner/diffusion factory and its definition. Definitions collect model
construction and defaults; they do not replace version detection, name conversion,
or model-specific inference algorithms.

The registry checks for missing, duplicate, and invalid version registrations.
Explicit references keep integrations reachable in static-library builds. The
existing CMake glob includes integration sources automatically. No legacy
builder, display-name table, or prediction-default fallback needs updating.

### Detection And Weight Conversion

Version detection remains in `ModelLoader::get_sd_version()` in
`src/model_loader.cpp`. Add model signatures there and preserve the order of
overlapping checks. Detection uses tensor names and metadata, and can run before
and after name conversion, so signatures must work at both stages. Do not load
tensor data to identify an architecture. The registry looks up definitions by
the detected version; it does not perform detection.

Weight-name mappings and their version-based dispatch remain centralized in
`src/name_conversion.cpp`. Add new mappings there when needed. Inference, GGUF
export, quantization, and LoRA conversion share the `convert_tensor_name` entry
point; preserve its prefix normalization and component conversion order.

### Components And Conditioning

The core factory receives `model_builders::Context`, the tokenizer configuration,
and the output runners. Backend checks, optional IP-Adapter setup, scale
overrides, and tokenizer usage validation remain in the common builder. A
factory returns failure if any required additional backend cannot initialize.

LLM architecture selection, tokenizer requirements, prompt preparation, and
encoding remain centralized in `src/conditioning/conditioner.hpp`. Integration
factories instantiate the conditioner with the model version; models requiring
new LLM conditioning behavior still need changes in that shared implementation.

`vae_kind` is the single source of truth for the VAE family; compatibility
helpers derive their results from it. `KL`, `FLUX`, and `FLUX2` share the
`AutoEncoderKL` implementation while retaining their distinct latent formats.
The related traits have separate responsibilities:

- `supports_video_generation` describes the model's video generation capability.
- `video_tae` selects `TinyVideoAutoEncoder` instead of `TinyImageAutoEncoder`.
  Image models can use a video TAE, so this is independent of video generation.
- `fake_vae` selects FakeVAE for pixel-space models. It is separate from
  `vae_kind`, which also controls latent-format compatibility helpers.
- `vae_format_override` allows PiD's VAE format override.

`configure_vae` handles model-specific KL adjustments such as
PixArt's scale and SDXL's convolution scale. VAE compatibility traits also drive
latent transformations and previews through the existing helper functions.

### Sampling And Geometry

`ModelSamplingPolicy` groups the defaults first: `default_prediction`,
`default_flow_shift`, `default_sample_method`, and `default_scheduler`. Set only
the values that differ from the struct defaults. `prediction_from_weights` and
`flow_shift_from_weights` preserve checkpoint-dependent defaults during automatic
prediction selection; `probe_v_prediction` enables SD2's runtime v-prediction
probe. Explicit prediction settings take precedence over automatic selection.
MOSS's default unconditional skip layer stays
in the diffusion engine, with active unconditional SLG taking precedence.

The resolved `prediction_t` selects the denoiser directly. Flux3, MiniMax H3 AV,
and Ming-Image use `FLUX3_FLOW_PRED`, `H3_AV_FLOW_PRED`, and
`MING_IMAGE_FLOW_PRED`; their CLI overrides are `flux3_flow`, `h3_av_flow`, and
`ming_image_flow`. Explicit `sd3_flow` always selects `DiscreteFlowDenoiser`,
and `flux_flow` always selects `FluxFlowDenoiser`, regardless of model version.
Automatic selection retains each model's default implementation. New prediction
values are appended before `PREDICTION_COUNT`, preserving existing valid values;
clients using the automatic-selection sentinel must rebuild with the new header.

`ModelGeometry` declares `diffusion_down_factor`, `latent_channels`, and
`image_channels`. `ref_image_preset` defaults to `"default"`; set it only for models
that need another reference-image preset.

The common pipeline executes these policies; specialized video, multimodal,
cache, and extension algorithms remain in their existing runtime components.
Model-specific diffusion extra parameters are assembled in the diffusion engine.
Video frame conversion and alignment also remain in the diffusion engine.
`get_vae_scale_factor()` retains PiD's pixel-space special case and otherwise
uses the VAE's scale factor.

Validate migrations against the previous implementation using synthetic metadata,
name conversions, prompt preparation, default/override selection, and focused
builds. Real generation checks are still needed to validate numerical behavior
when changing network or encoding algorithms.
