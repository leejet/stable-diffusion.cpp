# C API Internals

The public declarations live in [stable-diffusion.h](../../include/stable-diffusion.h).
See the [C API guide](../guides/c_api.md) for keys, value types and ownership rules.

## Internal modules

| Module | Responsibility |
| --- | --- |
| `src/api/options.h/.cpp` | Opaque options, key registration, setter validation and conversion to typed configuration. |
| `src/config.h/.cpp` | Typed configuration and the library defaults for each configuration group. |
| `src/api/tensor.h/.cpp` | Opaque tensor handles backed by `sd::Tensor<T>`, reference counting and conversion to private media views. |
| `src/core/media.h` | Private raw media views used by inference and codecs. |
| `src/api/result.h/.cpp` | Internal result storage, opaque result accessors, metadata attachment and media ownership. |
| `src/api/api.h/.cpp` | Context, generation, callback and query entry points, plus thread-local diagnostics and exception-to-status handling. |
| `src/generation_info.h/.cpp` | Construction of generation metadata JSON. |
| `src/pipeline/generation.h` | Internal generation entry declarations. |
| `src/stable-diffusion.cpp` | Internal typed image/video generation and model-specific validation. |
| `src/core/option_types.h/.cpp` | Internal mode enums, name lookup and parsing. |

Options are validated and converted once at the API boundary. The runtime reads
typed configuration fields. Internal callers such as ADetailer pass typed
configuration directly to the internal generation entry; they do not rebuild
options or call public setters.

The option map alone records whether a key was supplied. Zero, false, empty
strings and explicit automatic choices remain entries until `unset` removes
them. Each conversion starts with a fresh `OptConfig`, then overlays supplied
values and validates combinations. Reusing the output configuration cannot
retain a previous request's overrides. ADetailer applies its operation-specific
strength default before the overlay.

Fixed defaults live in the configuration constructors. Internal automatic
policies (such as model-selected sampler/scheduler and automatic eta/flow shift)
are resolved by the pipeline after refreshing the model sources. Their internal
sentinel values describe an automatic policy, not the presence of an input key.
Do not infer presence by comparing a typed value against its default.

CLI/server inference inputs use `std::optional`; their option builders emit
only engaged fields. Recipe defaults fill missing startup inputs; request
fields then override the configured recipe/startup values. Explicit empty
numeric arrays replace inherited arrays, and JSON `null` clears an optional
input. Frontend-only settings such as output formats and listen ports keep
local defaults. Frontend checks cover request syntax, file inputs and server
resource policy. Parameter ranges are checked by the public setters used by the
option builders; combinations and model applicability are checked when executing.
Examples include only public library headers and do not query library defaults.
Capability defaults describe configured overrides; null means that the library
selects the value, and result metadata reports effective values. Post-generation
ADetailer clears inherited dimensions and strength before applying pass overrides,
so its operation defaults also come from the library.

Result storage owns tensors and metadata. Tensor handles contain a dtype-specific
`sd::Tensor<T>` and an atomic reference count; options retain handles rather than
copying tensor payloads. Raw pipeline output buffers are converted to tensors at
the boundary and released there. Internal producers with an `sd::Tensor<T>` can
move it directly into a handle. Video results keep per-frame tensors and assemble
a contiguous video tensor only when requested. Cached assembly is synchronized.
Preview tensors are independent snapshots and may be retained by callbacks.
Metadata construction is separate from its attachment to a result.

These headers are private implementation details and are not installed as part
of the public C API.
