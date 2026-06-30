# Model Configuration Conventions

This document describes the conventions for model configuration structs and
weight-based configuration detection.

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
print a debug line, keep that as local control flow inside `detect_from_weights`.

## Logging

When config values are inferred from weights, print one `LOG_DEBUG` line at the
end of `detect_from_weights`.

Example:

```cpp
LOG_DEBUG("llm: num_layers = %" PRId64 ", vocab_size = %" PRId64 ", hidden_size = %" PRId64 ", intermediate_size = %" PRId64,
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
