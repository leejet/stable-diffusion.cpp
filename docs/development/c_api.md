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
| `src/api/generation.*` | Persistent DGL context, metadata probing, request execution, callbacks and cancellation. |
| `src/api/generation_program.*`, `generation_image.cpp`, `generation_video.cpp`, `generation_h3.cpp`, `generation_prompt.cpp` | Template selection, literal rendering and host input bindings. |
| `src/api/generation_template.*` | Embedded template parsing, variable formatting and conditional rendering. |
| `src/api/templates/` | Complete DGL execution flows and a few shared helpers. |
| `src/api/program_result.cpp` | Adaptation of DGL tensors and dictionaries to C results. |
| `src/api/generation_metadata.cpp` | Image parameter and WebUI infotext extraction through the metadata C API. |
| `src/api/image_metadata.cpp` | Container metadata reading and text/JSON reports through `sd_read_metadata()`. |
| `src/api/media_metadata.cpp`, `metadata.h` | Shared metadata encoding and C access to PNG/JPEG/XMP/AVI chunks. |
| `src/core/option_types.h/.cpp` | Internal mode enums, name lookup and parsing. |

Options are validated and converted at the API boundary. The translator renders
a concrete DGL program and binds media through `main` tensor inputs. All retained
inference entries use the same DGL compiler, execution context and model cache.
There is no separate native pipeline.

The option map alone records whether a key was supplied. Zero, false, empty
strings and explicit automatic choices remain entries until `unset` removes
them. Each conversion starts with a fresh `OptConfig`, then overlays supplied
values and validates combinations. Reusing the output configuration cannot
retain a previous request's overrides.

Fixed defaults live in the configuration constructors. Internal automatic
policies (such as model-selected sampler/scheduler and automatic eta/flow shift)
are resolved by the translator after refreshing the model metadata. Their internal
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
selects the value, and the executed DGL records the effective values.

Result storage owns tensors and metadata. Tensor handles contain a dtype-specific
`sd::Tensor<T>` and an atomic reference count; options retain handles rather than
copying tensor payloads. DGL outputs are adapted at the result boundary.
Internal producers with an `sd::Tensor<T>` can
move it directly into a handle. Video results keep per-frame tensors and assemble
a contiguous video tensor only when requested. Cached assembly is synchronized.
Preview tensors are independent snapshots and may be retained by callbacks.
Results retain the exact compiled DGL source. WebUI conversion and media metadata
embedding are separate from inference and never rebuild the execution program.
`src/api/generation_metadata.cpp` parses the source once to produce image
parameters and WebUI infotext together. Examples obtain this JSON through
`sd_get_generation_metadata()` without compiling or calling the DGL parser.
Container reading and metadata chunk generation also use the public C API.
Examples have no private `src` include path and retain generated DGL source
without legacy JSON conversion or LoRA path rewriting.
The translator uses an explicit `llm_path` as the text encoder's main `path`, falling back to
the checkpoint and other configured encoder paths when it is absent.

Context creation snapshots options without loading weights. Family probing is
lazy and tracks actual file dependencies. The complete request stays busy during
probing, compilation, batch gaps and execution. File permissions are checked
before probing and on every DGL resource access, including cache hits.
Generation templates load their required components as part of execution.
Capability queries read the detected family under the context lock and probe it
on demand. They leave the generation cancellation state untouched.

C++ validates inputs, resolves model defaults, prepares geometry and binds host
tensors. Templates describe complete execution flows and optional stages;
operations and models retain computation algorithms. Split templates when the
flow changes substantially, and use conditions for smaller variations. Do not
search and patch rendered source to change execution. See the
[template guide](../../src/api/templates/README.md).

These headers are private implementation details and are not installed as part
of the public C API.
