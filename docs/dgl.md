# DGL scripting guide

Use DGL (Diffusion Graph Language) to compose model and tensor operations in a
script. Values can feed several calls, and functions can process lists. A script
can load its own models and media or receive named input tensors from a host.
Applications can generate a complete script from request settings.

- **Getting started** · [Build and run](#build-and-run) · [Model examples](dgl/README.md)
- **Writing scripts** · [Language syntax](#dgl-syntax) · [Tensor operations and indexing](#tensor-functions) · [Built-in catalog](dgl-builtins.md)
- **Generation** · [Models and sampling](#independent-generation-components) · [Media I/O](#image-and-video-files) · [Regions and masks](#regions-and-masks) · [Worked examples](#advanced-component-programs)
- **Embedding DGL** · [C API](#c-api) · [Architecture](development/dgl.md)

## Build and run

```sh
cmake -S . -B build-dgl -DSD_BUILD_EXAMPLES=ON -DSD_WEBP=OFF -DSD_WEBM=OFF
cmake --build build-dgl --config Release --target sd-cli
```

Use the normal backend build options, such as `-DSD_CUDA=ON`, when needed.

Save the following program as `tensor-example.dgl`. It runs without model weights.

```dgl
# version: 0.0.0
fn main() {
    return tensor::zeros([4, 4])
}
```

```sh
./build-dgl/bin/sd-cli -M dgl --script tensor-example.dgl --tensor result tensor.npy
```

For generation, edit the model path, prompt and seed in the [SD 1.5 example](dgl/sd/sd1-5-txt2img.dgl).

```sh
./build-dgl/bin/sd-cli -M dgl --script docs/dgl/sd/sd1-5-txt2img.dgl --check
./build-dgl/bin/sd-cli -M dgl --script docs/dgl/sd/sd1-5-txt2img.dgl --tensor latent latent.npy
```

To repeat a run, execute the same `.dgl` file again with the same input files.

`--check` checks syntax, function signatures, references and static types without
loading weights. It does not check model-specific tensor compatibility.

`--tensor NAME FILE.npy` exports a named output, and can be repeated. The result
summary and tensor names appear in the verbose log (`-v`). `image::save(..., pixels)`
writes PNG files during execution. Rerunning the script writes to the same paths.
Paths are relative to the process working directory.

DGL mode uses the CLI's log levels and prints logs to stderr. The default level
is `info` (`[I]`). Use `-v` or `--verbose` for verbose (`[V]`) logs, and
`--log-level debug` to include debug (`[D]`) logs.
Sampling reports its method and step count at the start, and elapsed time on
completion, at the info level.

## DGL syntax

Execution starts at the single `fn main()` declaration. Standalone scripts use
a zero-argument entry point.

Use an optional `var { ... }` block before the functions to group values
such as model paths, prompts, and seeds. Each entry is a scalar, list, or
dict literal. The names are available directly in `main()` as if their
bindings appeared at the start of its body.

```dgl
var {
    seeds = [41, 42]
}

fn next_seed(seed) {
    return seed + 1
}

fn main() {
    return map(seeds, process = next_seed)
}
```

### Functions and bindings

- Declare user-defined functions at file scope using the `fn` keyword and
  pass values to them as arguments. Put local bindings inside a function.
  When constructing a script, encode string values as JSON string literals.
- Bind each name once, and reference values declared earlier in the same scope.
  Names in `var` share the entry point's binding scope.
  Positional arguments precede named arguments. Unknown or duplicate arguments
  are errors. Argument expressions are evaluated left to right as written at the
  call site. A `return` statement, when present, must be last. A function may
  finish without a return value, or use bare `return;` to finish explicitly.
- Call functions such as `image::save(...)` directly as statements. Saving
  functions return `void`; using them in assignments, arguments, containers
  or returned expressions is a compile-time error.
- Strings use JSON escaping. Comments start with `#` or `//`, on their own line
  or after code. Only `# version: ...` is reserved for the file's version header.
  Semicolons are optional. Lists use `[a, b]`. A dictionary (`dict`) groups
  named values using `{name = value, ...}`. Access a field with `dict_value.name`.
  A prompt condition contains the fields supplied by its encoder. Reading an
  absent field reports its name and source location. See how to read the
  [built-in return types](#built-in-functions).
- Functions run sequentially. Recursion, cycles and runtime-dependent control
  flow are not supported. Intermediate values are released after their last use
  unless a function retains them or the program returns them.

### Lists, map and fold

- `map(items, process = function, other_parameter = value)` passes each item as
  the first function parameter. Additional arguments are shared across calls.
- `fold(items, initial, process = function, other_parameter = value)` passes
  the previous result and the current item as the first two parameters. The
  function must return the same type and fields as `initial`. An empty typed
  list returns `initial`.
- `items[index]` selects a list element; `items[start:end]` selects a list slice.
  `tensor::to_list(tensor::arange(seed, seed + count))` supplies consecutive
  integer seeds to `map`. The range end is exclusive.
- Lists contain one element type; integer and floating-point literals may be
  mixed in numeric lists. With a list of dictionaries, `map`, `fold` and
  dynamic indexing expose shared fields and consistent tensor dtype/rank facts.
  A known index into a known list keeps the selected item's own information.
  `list::replace` requires a replacement compatible with the list's element
  type; incompatible types are rejected during compilation.

### Conditional expressions

Use `if condition { expression } else { expression }` to choose a value:

```dgl
result = if enable_refiner { refine(latent) } else { latent }
```

The condition must be a boolean known before resource planning, such as a
`var` value or a literal passed to a function. Conditions derived from tensors,
model outputs or runtime function calls are rejected. Both branches are checked
for valid calls and matching result types and dictionary fields, even when one
is disabled. Each branch can contain local bindings and standalone calls,
followed by a final expression that supplies its result. An `if` used as a value
requires `else`. Conditional expressions can be nested, used as arguments,
indexed or returned.
A conditional used as a standalone statement discards its result, including
`void` results.

```dgl
result = if enable_refiner {
    refiner = load_unet(refiner_weights)
    enlarged = tensor::interpolate(latent, [128, 128])
    refine(enlarged, refiner)
} else {
    latent
}
```

Each branch has its own scope and can read outer bindings. Local names are
unavailable outside the branch, and the two branches may use the same local
names. Bind each name once; shadowing an outer binding is rejected. Use the
final expression instead of `return`, which remains a function statement.
An empty block or one ending in a binding has type `void`. Semicolons remain
optional separators and do not change the final expression's result.
Insert `;` before a following list expression when it could otherwise be read
as indexing of the preceding expression.

A standalone `if` may omit `else` and has type `void`:

```dgl
if save_latent {
    tensor::save("latent.npy", latent)
}
```

The selected block executes for its side effects. Assignments, arguments,
containers and returned values still require the two-branch form.

Only the selected branch contributes model resources and weight-use phases to
the run plan, and only that branch executes. Each statement in the selected
block retains its own planning boundary, including blocks nested in function
calls or argument expressions. Load optional models inside the branch or inside
the function it calls. Loading a model before the conditional
still loads it unconditionally. For example:

```dgl
# version: 0.0.0

var {
    enable_refiner = false
    refiner_weights = "optional-refiner.safetensors"
}

fn main() {
    model = if enable_refiner { load_unet(refiner_weights) } else { load_unet("base.safetensors") }
    return model
}
```

Here the disabled refiner file need not exist. Keep optional model loads
inside their selected branch. Use `map` and `fold` for collections of work. Loops whose next step depends on a computed result are
not supported.

### DGL versions

The optional version header declares the minimum DGL runtime version required
within one major version.

```dgl
# version: 0.0.0
```

Write three unsigned 32-bit decimal integers (`MAJOR.MINOR.PATCH`). The header
may appear once, before all code. Blank lines and ordinary comments may precede
it. Omitting it uses the current runtime. Forms such as `1`, `1.2` and `v1.2.0`
are format errors.

**Development.** The runtime, examples and C API entry templates currently use
`0.0.0`. Development snapshots have no compatibility guarantee. Identify a
snapshot by its repository commit. The first stable release, `1.0.0`, establishes
compatibility for syntax, built-in signatures and documented behavior.
Development and stable versions reject each other. Verify a development script
before updating its declaration.

**Stable releases.** The runtime must have the same major version and meet or
exceed the declared minimum. Components are compared numerically. A `1.2.0`
script can run on `1.2.3`, `1.3.0` or `1.10.0`. On `1.1.9` or `2.0.0` the
compiler reports the required and supported versions. Compatible scripts use
the current implementation and its available built-ins.

| Increment | When to use it |
| --- | --- |
| Major | Existing legal scripts need edits, or their documented meaning changes. |
| Minor | A new capability preserves the behavior of existing legal scripts. |
| Patch | A fix restores documented behavior, or an optimization preserves it. |

Use the highest applicable increment for a release. DGL is versioned independently
of sd.cpp, the C API and its templates. Raise a script's declaration when it
depends on a new capability or fix. Each complete C API entry template declares
its own minimum, which also covers the helpers it includes. The rendered header
is part of the source retained in `generation_info` and embedded in generation
metadata when enabled.

### Built-in functions

The [built-in function catalog](dgl-builtins.md) lists registered functions,
parameters, defaults and return types. For `encode_prompt` and the nested
conditions from `photomaker::encode_prompt`, the **Returns** column lists
possible fields across models. For example, a condition contains `ref_audios`
only when the encoder supplies audio references. Run
`sd-cli -M dgl --list-builtins` to inspect the catalog for your executable.

To regenerate the checked-in catalog after changing a registration, run
`cmake --build build-dgl --config Release --target dgl-update-builtins`.
The `dgl-check-builtins` target verifies that the file matches the registry.

## Tensor functions

Tensor types are U8, I32, I64, F32 and F64. Dimensions are stored contiguous-first:

| Value | Type | Shape |
|---|---|---|
| RGB image | U8 | `[3, width, height]` |
| RGB video | U8 | `[3, width, height, frame_count]` |
| Audio samples | F32 | `[channels, samples]` |
| Diffusion latent | F32 | Model-specific, including four- and five-dimensional layouts |
| Region boxes | I64 | `[4, region_count]` |
| Detection coordinates | F32 | `[4, detection_count]` |
| Detection scores / class IDs | F32 / I64 | `[detection_count]` |
| Conditioning | F32/I32 | Model-specific; inspect each tensor |

Rank-zero tensors are scalars. Zero-sized dimensions represent empty tensors.

### Operations and broadcasting

Qualified function names use `::`. Call a tensor function with
`tensor::reshape(x, [4, 8])` and access a dictionary field with `condition.crossattn`.

| Function | Arguments and result |
|---|---|
| `tensor::zeros`, `tensor::full` | Take a shape and optional `dtype` (default `f32`). `full` also takes a scalar fill value. |
| `tensor::zeros_like`, `tensor::randn_like` | Use the input shape and dtype. `randn_like` supports floating-point tensors and requires an RNG object. |
| `tensor::shape`, `tensor::numel`, `tensor::reshape` | Read the shape as a rank-one I64 tensor or the element count as a scalar. Reshape preserves element count and dtype. |
| `tensor::unsqueeze` | Insert a length-one axis, preserving dtype and element order. `axis` accepts `[-rank-1, rank]`; output rank is at most 8. |
| `tensor::pad` | Constant padding with `[before0, after0, before1, after1, ...]` in DGL axis order. `mode="constant"` and `value=0` are defaults. |
| `tensor::vector` | Construct a rank-one tensor from numeric values. The default dtype is `f32`. Read elements with bracket indexing. |
| `tensor::to_list` | Convert a rank-one tensor of at most 10,000 elements into `list<int>` or `list<number>`. `element="auto"` infers the element type from a known dtype; `"int"` and `"number"` declare the expected category. |
| `tensor::arange`, `tensor::index_select` | Generate a rank-one I64 range with exclusive end, or gather slices in index order along one axis. |
| `tensor::crop`, `tensor::replace_slice`, `tensor::concat` | Crop takes lists of axes, starts and sizes. Replacement and concatenation require matching dtype and dimensions outside the selected axis. Use bracket syntax for basic slicing. |
| `tensor::cast` | Explicitly convert to a `dtype`. Integer conversion truncates toward zero and rejects out-of-range values. |
| `tensor::broadcast_to` | Broadcast a tensor to an explicit shape and preserve its dtype. |
| `tensor::add`, `tensor::mul` | Accept floating-point tensors of the same dtype and numeric scalars. |
| `tensor::lerp` | Compute `a * (1 - weight) + b * weight`. Two U8 endpoints accept a scalar or floating-point weight tensor, with output rounded and clamped to U8. |
| `tensor::affine`, `tensor::clamp` | Apply `scale` and `offset`, or constrain values to `min` and `max`. Both take a floating-point tensor. |
| `tensor::interpolate` | Resize two axes of a U8 or floating-point tensor using `size=[width, height]`, `axes=[0, 1]` by default, and a `mode` such as `bilinear` or `bicubic`. |
| `tensor::translate`, `tensor::index_fill` | Shift along selected axes with constant fill, or fill entire slices at selected indices along one axis. |
| `tensor::grayscale`, `tensor::threshold` | Convert the channel axis to grayscale or select `low` below the threshold and `high` at or above it. |
| `tensor::maximum` | Take the elementwise maximum of two tensors with the same dtype and broadcastable shapes. |
| `tensor::max_pool2d`, `tensor::min_pool2d` | Pool over two axes with a configurable kernel and stride. `padding_after` defaults to `padding_before`. Set `pad_value` to include a constant outside the tensor. |
| `tensor::gaussian_blur` | Blur two axes with edge replication. `radius` sets the kernel half-width and `sigma` sets the Gaussian spread. |
| `tensor::literal` | Construct a tensor from its `dtype`, `shape` and little-endian Base64 `data`. |

Broadcasting aligns dimensions from the lowest axis and treats missing higher
dimensions as 1. A numeric scalar broadcasts to all elements.
Shape, axis and size arguments accept rank-one I64 tensors and lists of
integers. List elements can be scalar expressions, including indexed dimensions;
there is no need to create or modify a tensor just to describe a shape:

```dgl
shape = tensor::shape(latent)
mask_shape = [shape[0], shape[1], shape[2], 4, shape[4]]
mask = tensor::index_fill(tensor::zeros(mask_shape), axis = 2, indices = [0], value = 1)
```

`latent_shape(vae, width, height, ...)` and
`audio_latent_shape(audio_vae, frames, fps)` also return I64 shape tensors.
The audio shape uses the loaded audio VAE's model family and the video duration
(`frames / fps`), keeping each family's latent layout and rounding rules.
Both are metadata queries; `audio_latent_shape` does not load device weights
or run the audio VAE.

```dgl
sound_shape = audio_latent_shape(audio_vae, frames = frames, fps = 24)
sound_base = tensor::zeros(sound_shape)
```

`math::product` multiplies dimensions, and `math::ceil_div`
divides nonnegative integers rounding upward. Schedule functions use scalar arithmetic for indices and `math::floor(steps * strength)`
to convert a fraction of steps to a count.

### Read-only list and tensor indexing

Use brackets after any list or tensor expression, including a function call:

```dgl
first_sigma = sigmas[0]
refine_sigmas = full_sigmas[-13:]
width = tensor::shape(pixels)[0]
patch = x[1:3, :, 0]
```

Lists accept exactly one integer index or slice. Integer indices may be negative;
out-of-range indices fail. Slices are half-open, accept omitted or negative
endpoints and clip to the list bounds. A slice preserves the input's element type,
even when empty or when a `list<number>` slice happens to contain only integers.
Integer values do not need to be converted to floating point.

```dgl
first = items[0]
last = items[-1]
batch = items[2:5]
rest = items[1:]
final_steps = items[-3:]
pixel = patches[0][1, 2, 3]
```

`list::at` and the global `tail` function have been removed. To select a suffix,
use `items[-count:]`. This clips to the whole list when `count` exceeds its length;
`count=0` gives `items[-0:]`, equivalent to `items[:]`, rather than an empty list.
Use `items[:0]` for an empty slice. Empty `map` results retain the callback's
inferred return type for that call, such as `list<int>` or `list<tensor>`.
`list::replace` remains the functional update operation; `map` and `fold`
continue to iterate lists only.

Axes follow DGL shape order: axis 0 is the lowest, contiguous dimension.
Integer indices remove their axis; slices keep it. Omitted trailing axes are
retained. For `x` shaped `[8,6,4,2]`:

| Expression | Result |
|---|---|
| `x[1:3, :, 0]` | Tensor `[2,6,2]` |
| `x[1:3, :, 0:1]` | Tensor `[2,6,1,2]` |
| `x[-1]` | Tensor `[6,4,2]` |
| `x[1,2,0,0]` | Numeric scalar |
| `x[3:1]` | Empty tensor `[0,6,4,2]` |

Integer indices and slice endpoints are integer expressions. Negative indices
count backward from an axis's end; out-of-range integer indices fail. Slices
are half-open, support omitted endpoints, and clip endpoints to the axis bounds.
`x[start:]`, `x[:end]` and `x[:]` are supported. Endpoints can contain scalar
arithmetic, for example `sigmas[count - 13:]`.

An explicit slice always returns a tensor, including empty and length-one
slices. With only integer indices, consuming all axes returns a scalar; otherwise the
result is a tensor. The compiler retains known rank and element types from
tensor constructors, shape queries and schedules, so fully consumed indices
keep their numeric types in lists and user functions. `--check` rejects known
non-integer indices/endpoints and too many indices for a known rank.
File-loaded tensor ranks and other unknown properties remain runtime checks. Integral tensor elements become DGL integers;
floating-point elements become DGL numbers.

Brackets form an indexing expression. The checker selects list or tensor rules
from the input type, and execution dispatches to the corresponding operation.
Known list elements and indices also select precise model origins during
resource planning; dynamic indices conservatively combine candidate origins.
Step sizes, ellipsis, boolean masks, index tensors and indexed assignment remain
unsupported.

### Axis insertion and constant padding

`tensor::unsqueeze(input, axis)` inserts one length-one dimension and returns
a contiguous tensor with the same values and dtype. Negative axes count from
the end of the insertion positions; `axis=-1` appends a dimension.

`tensor::pad(input, pad, mode="constant", value=0)` takes an even-length list
or rank-one I64 tensor of padding pairs. The first pair is before/after axis 0,
the second is axis 1, and so on. Unspecified higher axes receive no padding.
Only `constant` mode is supported. Lengths must be nonnegative, the number of
pairs must fit the input rank, and padded shapes retain the 256 MiB limit.
The fill value follows the input dtype's scalar-conversion rules.
An empty or all-zero padding list borrows the input without changing it.

Frame assembly and constant padding use tensor operations directly.
For example, turn one image into a video tensor and append constant frames:

```dgl
image = image::load("input.png")
first_frame = tensor::unsqueeze(image::to_f32(image), axis = 2)
frames = tensor::pad(first_frame, [0, 0, 0, 0, 0, 32], value = 0.5)
```

Image normalization runs before padding, so `value=0.5` means half-intensity F32
pixels. The frame axis remains explicit even for a single frame. Use
`tensor::concat` to assemble multiple frames and `tensor::pad` for constant
padding. Negative padding lengths fail instead of discarding input frames.

### Tensor values as lists

`tensor::to_list(input, element="auto")` explicitly converts a rank-one tensor
into a script list for `map`, `fold` or list functions. U8, I32 and I64 produce
`list<int>`; F32 and F64 produce `list<number>`. Integers are read directly as
int64 values, preserving I64 values above 2^53. NaN and infinity are rejected.
An empty tensor produces an empty list with the same element type.

```dgl
fn add_seed(total, seed) {
    return total + seed
}

fn main() {
    batch_seeds = tensor::to_list(tensor::arange(42, 46))
    empty = tensor::to_list(tensor::arange(42, 42))
    return {
        seeds = batch_seeds,
        sum = fold(batch_seeds, 0, process = add_seed),
        empty_sum = fold(empty, 0, process = add_seed)
    }
}
```

The seeds are `[42, 43, 44, 45]`, their sum is `174`, and the empty fold
returns its initial value `0`. A map callback can also construct an RNG from
each seed and return an independent noise tensor.

The compiler must know the list element type before checking the map or fold
function. With `auto`, it uses a known tensor dtype, such as I64 from `arange`
or the declared dtype of `tensor::vector`. If dtype is unknown, including an
unconverted `tensor::load()` result or a host tensor parameter, declare it:

```dgl
items = tensor::to_list(tensor::load("seeds.npy"), element = "int")
```

`element` must be known at compile time. It declares a type category and does
not cast values: F32 with `element="int"` and I64 with `element="number"` are
errors. Use `tensor::cast` before conversion to change dtype explicitly. Known
rank or dtype mismatches fail compilation; runtime validation checks the actual
input as well. Multidimensional tensors require an explicit reshape first.
Each element creates a script value, so the limit is 10,000 items, including
when the input came from a file or the host.

The former `seeds(seed, count)` function is removed. Use
`tensor::to_list(tensor::arange(seed, seed + count))`, or a plain integer for a
single fixed seed. Integer range bounds do not wrap at the int64 limit.

### Scalar arithmetic

Scalars support `+`, `-`, `*` and `/`, parentheses, and unary `+` and `-`.
Multiplication and division bind before addition and subtraction; binary
operators of equal precedence associate left to right. Indexing binds before
unary signs, so `-sigmas[0]` negates the selected scalar.

Integer `+`, `-` and `*` preserve the integer type and reject int64 overflow.
A floating-point operand produces a floating-point result. `/` always returns
a floating-point value and rejects division by zero. Use `math::floor`,
`math::ceil` or `math::ceil_div` when an integer count is required.

```dgl
missing = count - 1
area = width * height
retained = math::floor((steps - 1) * strength)
sigma = sigmas[count - 13]
```

Operators accept numeric scalars. Use `tensor::add`, `tensor::mul` and other
explicit tensor functions for tensor arithmetic. Operators lower to the same
`math::add`, `math::sub`, `math::mul` and `math::div` implementations as calls.
The `var` block continues to accept literals, including signed numbers;
write computed expressions in function bodies.

### Tensor files

Use `tensor::save` to write a tensor from any step to a NumPy `.npy` file.
The former top-level names `load_tensor` and `save_tensor` have been removed;
use `tensor::load` and `tensor::save`, with the same parameters and behavior.
Saving functions take the output path first, followed by the data to write.
Saving functions have no return value. Call them as standalone statements;
return tensors or media separately when they should appear in the result.
`tensor::load` reads the same format, including files exported by `--tensor`.

```dgl
# version: 0.0.0

fn main() {
    latent = tensor::load("latent.npy")
    tensor::save("latent-copy.npy", latent)
    return latent
}
```

`tensor::load` supports `u8`, `i32`, `i64`, `f32` and `f64` tensors up to rank 8
and 256 MiB. The NumPy shape is the reverse of the DGL shape, so a DGL
image shaped `[3, width, height]` appears as `(height, width, 3)` in NumPy.

## Independent generation components

A generation script loads the model components, encodes the prompt, builds a
predictor and a noise schedule, then samples and decodes the latent. See
[model loading](#model-loading-and-reuse), [prompt formatting](#prompt-encoding),
[predictors](#predictors-and-guidance) and [sampling](#noise-schedules-and-sampling).

### Model loading and reuse

Use `load_unet` for UNet architectures such as SD 1.x, SD 2.x and SDXL,
and `load_dit` for diffusion transformers. The names resolve to the same loader.
`load_text_encoder` loads the text encoder and `load_vae` loads the VAE.
A single checkpoint can supply all three components, as in
[SD 1.5 text-to-image](dgl/sd/sd1-5-txt2img.dgl).
For standalone weights whose tensor names are shared across model families,
set `family` to a lowercase [model family ID](dgl-builtins.md#model-family-ids)
such as `"sd1"`, `"flux"` or `"qwen_image_2_1"`. With `family = "auto"`,
you can pass a diffusion checkpoint as `family_from` to detect the family from
its metadata. Use `vision_path` when the text encoder's vision weights are in
a separate file. `path` selects the main text encoder weights; use `clip_l_path`,
`clip_g_path` and `t5_path` when those components are in separate files.

Component loaders accept `backend`, `params_backend`,
`wtype`, `tensor_type_rules`, `split_mode`, `flash_attn_mode`, `conv_direct_mode`
plus `sage_attn`. Pass one setting per loader parameter.
Use `"auto"`, `"on"` or `"off"` for `flash_attn_mode` and `conv_direct_mode`.
These options apply only to the model returned by that loader. For example:

```dgl
unet = load_unet("sdxl.safetensors", flash_attn_mode = "on", conv_direct_mode = "off")
vae = load_vae("sdxl.safetensors", flash_attn_mode = "off", conv_direct_mode = "on", conv_scale = 0.03125)
```

To place two components separately, set `backend = "cpu"` on the text encoder
and `backend = "cuda0"` on the DiT. The DiT and text encoder
also accept device lists such as `"cuda0&cuda1"` for split execution.

Loader option `conv_scale` overrides Conv2D input scaling, including grouped
Conv2D. Output scaling is compensated after convolution. The default `0`
keeps model defaults. For an SDXL VAE, set `conv_scale = 0.03125` explicitly,
including when loading the VAE from a separate file. Use `conv_scale = 1` for unscaled convolution.
This does not change VAE latent normalization or model weights.

Model-loading functions accept literals and scalar literal bindings from
`var` or `main()`.
An execution context caches components by their weight sources and loader
options. Programs run in the same context can share a component even if they
give it different variable names. The cache checks each loader's declared
files, including its configured tokenizer slots. Encoded conditions also track any textual inversion embeddings
used by the prompt. Each run calls generation and image-processing
functions with its own inputs and sampling state. A context permits one active run.

The CLI defaults to `--run-once on`, which releases models after their last use.
Use `--run-once off` to cache models for reuse during the invocation, subject to
the context's memory limits. Freeing the context releases its cache.

The executor tracks the last use of each binding and dictionary field. A component
stays loaded while another live value, such as a predictor, still uses it.
Repeated loads share a live component. With `--run-once on`, loading the same
component again after its last use reads its weights again.

### LoRA combinations

`load_lora` creates a reusable adapter handle. `apply_lora` registers the
adapter's weights and derives a model of the same type with its own ordered
adapter list. Both models can be used independently.
For example, a low-resolution pass can use `draft` and a later pass can add a
second adapter for refinement.

```dgl
base = load_unet("sd15.safetensors")
style = load_lora("style.safetensors")
detail = load_lora("detail.safetensors")
draft = apply_lora(base, style, weight = 0.8, mode = "immediately")
refine = apply_lora(draft, detail, weight = 0.4)
```

The optional `mode` applies to the entire returned combination. Omitting it
inherits the input model's mode. The same adapter handle can be used on any
weighted model when the adapter contains matching parameters for that model.

`at_runtime` applies adapters in the forward graph. `immediately` builds and
caches merged buffers for the affected parameters on first use. Compatible
combinations share the original parameter buffers and the adapter tensors
across both modes. Changing mode or strength can reuse the original weights
while they remain resident. Persistent contexts retain idle buffers subject
to RAM and VRAM pressure. Run-once contexts can release shared buffers after
their last use.
In both persistent and run-once contexts, `auto` selects `immediately` for
unquantized and [INT8 tensorwise weights](guides/int8_convrot.md) on the compute
backend without row splitting. It selects `at_runtime` for other quantized
formats, offloaded weights, or row-split weights.

### Prompt encoding

`encode_prompt` returns named conditioning tensors. `predictor` binds them to a
model.
Wan and LingBot encoders zero UMT5 embeddings at padding positions before
returning the condition, as required by their fixed-length diffusion text input.
Embeddings for actual prompt tokens are unchanged.

`encode_prompt` uses the text encoder's model default for an omitted or
nonpositive `clip_skip`, including the penultimate CLIP layer for SD2 and SDXL.
Explicit positive values override that default. Requesting a layer whose weights
are absent reports an encoding error.

LLM-based encoders accept optional `system` and `raw` arguments. This includes
the Qwen/Mistral/GPT-OSS/Gemma paths, Anima, Fibo, Ming-Image, LLaDA-Image,
LTX, HiDream-O1 and SenseNova U1.5.

- Omit `system` to keep the model's default prompt formatting.
- Set `system="custom instruction"` to replace only the system text, retaining
  the encoder's role markers and the rest of its template. Encoders without a
  default system segment insert one using their role format.
- Set `system=""` to omit the whole system segment, including its role markers.
  An encoder without a default system segment keeps its existing template.
- `raw=false` is the default. With `raw=true`, `text` is the complete tokenizer
  input: no chat template, caption rewriting, prompt-weight parsing or inline
  image/embedding resolution. Model-required padding, hidden-layer selection,
  learned queries and projections still apply.

Passing `system` together with `raw=true` is an error, even for `system=""`.
Raw text mode also rejects `media` and `image_bindings` arguments;
use normal template mode for multimodal input. System changes update template
crop offsets and prefix-dependent padding, and image token positions are
computed from the resulting prefix. Default formatting remains unchanged.
CLIP and T5-only encoders reject these formatting overrides.

For an already loaded LLM text encoder:

```dgl
custom = encode_prompt(text, prompt, system = "Describe the requested image.")
without_system = encode_prompt(text, prompt, system = "")
raw = encode_prompt(text, tokenizer_input, raw = true)
```

The former `unconditional` parameter has been removed. SenseNova's legacy
negative format also omits `<think>`, so `system=""` does not reproduce it.
Use the full negative tokenizer input instead:

```dgl
negative = encode_prompt(text,
    "<|im_start|>user\nblurry<|im_end|>\n<|im_start|>assistant\n<img>",
    raw = true)
```

Native CLI/C API generation constructs that complete negative input automatically,
for empty and nonempty negative prompts. Ordinary CLIP negative prompts use the
same encoding as positive prompts and need neither `system` nor `raw`.

#### Media inputs

`media` supplies ordered inputs to the encoder's existing model-specific prompt
template. An image tensor is shorthand for `{image = tensor}`. Image-only
lists can use either form; mixed media lists use dictionaries so every list
element has the same DGL type:

```dgl
clip = video::load("inputs/clip.webm")
sound = audio::load("inputs/voice.wav")
condition = encode_prompt(text, prompt, media = [
    {image = source_image},
    {video = clip.video, stride = 12},
    {audio = sound}
])
```

Each dictionary has exactly one `image`, `video` or `audio` field. Video uses
`{frames, fps_num, fps_den}`; audio uses `{samples, sample_rate}`. Only video
accepts `stride`, an integer frame interval in `1..65536`, defaulting to `1`.
Sampling retains source-frame timestamps. A video does not implicitly include
its soundtrack: add `{audio = clip.audio}` separately when needed. Unsupported
media types or image counts are errors. MiniMax H3's audio prompt input adds
its audio marker; audio encoding and predictor references remain separate.

`image_bindings = {subject = tensor}` supplies images referenced by
`<image:subject>` in the text. Explicit bindings take precedence over file
paths. An invalid referenced binding is an error; an absent name uses the
existing path reader and file-access callback. Unreferenced bindings do not
enter conditioning or its cache key. `raw=true` rejects `media` and
`image_bindings`, including explicitly empty containers.

Multimodal prompt encoding can consume reference-video frames independently:

```dgl
condition = encode_prompt(text, prompt, media = [{ video = {frames = frames, fps_num = 24, fps_den = 1}, stride = 12 }])
```

A `video` media entry takes the same video dict. Its `frames` can be a
U8 `[3,w,h,t]` or F32 `[w,h,t,3,1]` tensor. `stride` defaults to 1 frame. The encoder receives selected frames with timestamps
`frame_index * fps_den / fps_num`. At 24 FPS, stride 12 selects timestamps
0, 0.5, 1.0 seconds and so on. Only selected frames are converted to the
encoder's float pixel layout. Audio media uses `{audio = sound}`, where
`sound` is the same samples/sample-rate dict used by audio I/O.

#### Textual inversion

For textual inversion, use an inline file path with an SD 1.x, SD 2.x or SDXL
CLIP encoder:

```dgl
text = load_text_encoder(model_path)
negative = encode_prompt(text, "<embedding:embeddings/badhandV4.pt>, blurry")
```

`<embedding:...>` follows the same path and escaping rules as `<image:...>`.
Paths resolve from the process working directory. An escaped marker is ordinary
text, and a bare filename or former trigger word is also ordinary text. Other
encoders reject embedding markers. The vectors replace token embeddings before
CLIP adds positional embeddings; prompt weights, token chunks and SDXL dual-CLIP
zero-padding still apply. No tokenizer aliases or loader `embeddings` argument
are used.

Every encoding checks `SD_DGL_FILE_READ_MODEL` permission before inspecting the
files or looking up a cached condition. Cached conditions track canonical paths
and file versions, including referenced shards. Replacing an embedding invalidates
the condition without rebuilding the text encoder. The server's model directory
allowlist also applies to these paths. Native CLI/C API textual inversion is
unchanged.

#### Conditioning fields

`zero_out(condition, fields = ["crossattn"])` returns a new
condition dict with only the selected tensor fields cleared. Omit `fields` to
clear every tensor field and every element of tensor lists. Shapes, dtypes and
non-tensor metadata are preserved; the input dict is unchanged. A selected field
must exist and contain a tensor or tensor list. Clearing a condition is a tensor
operation, and does not make it valid unconditional input for every model.
For SDXL, clear text embeddings while preserving size and crop conditioning:

```dgl
fn sdxl_zero_text(cond) {
    zeroed = zero_out(cond, fields = ["crossattn"])
    pooled_zeros = tensor::zeros_like(cond.vector[:1280])
    adm = tensor::replace_slice(cond.vector, pooled_zeros, axis = 0, start = 0)
    return {crossattn = zeroed.crossattn, vector = adm}
}
```

The first 1280 values of SDXL's `vector` contain pooled text embeddings; its
remaining 1536 values encode size and crop information. This helper clears
`crossattn` and pooled text only. The CLI/C API calls it for an empty negative
prompt with non-EDM-v prediction. The SDXL examples expose `zero_negative`
as an explicit choice: set it to `false` to use the encoded negative prompt,
including nonempty negative text. Both branches return the two SDXL condition
fields, `crossattn` and `vector`, to keep their static dictionary types identical.

A conditioning dictionary can use tensors in the fields
consumed by `predictor`, including model-specific token and reference metadata.

### Predictors and guidance

`predictor` binds a condition to a diffusion network, with optional `concat` and `ref_latents` inputs. The
[Qwen Image 2.1 editing program](dgl/qwen_image_2.1/qwen-image-2-1-edit.dgl) passes the
reference pixels to the text encoder and their VAE latent to the predictor.
[Text-to-image](dgl/qwen_image_2.1/qwen-image-2-1-txt2img.dgl) uses the same
components with text conditions.
Reference images are prepared by the script. DGL does not select reference
presets or apply a reference resize policy based on the model family.
Use `tensor::interpolate` and `tensor::crop` before encoding. Pass prepared
pixels to `encode_prompt` and prepared latents to `predictor` explicitly.
`predictor(ref_index_mode="increase")` selects reference position indexing;
the default is `"fixed"`. Use `"decrease"` for layered references and set
`force_ref_timestep_zero=true` when required by the checkpoint.
For MOSS-Image, `ref_rope_anchor` aligns reference tokens with the start of the
image (`"image_start"`) or text (`"text_start"`) along the RoPE sequence axis.
`"default"` uses the model's alignment. See the
[MOSS-Image editing examples](models/moss_image.md#dgl).

`prediction` belongs to `predictor`, for example
`pos = predictor(dit, condition, prediction="eps")`. `load_dit` and its
`load_unet` alias load the model without selecting a prediction parameterization.
Omitting `prediction` selects `"auto"`, using the model's declared defaults and,
when needed, its checkpoint detection. The explicit values are `eps`, `v`,
`edm_v`, `flow`, `flux_flow`, `sefi_flow`, `minit2i_flow`, `sensenova_u1_flow`,
`flux3_flow`, `h3_av_flow`, and `ming_image_flow`.

`predictor(transform=step_fn, transform_args={...})` calls a script function
for each model evaluation. Its two arguments are the step dictionary and
the supplied `transform_args`. The step contains `sigma`, the model's
`timestep`, and `input_scale`, `skip_scale`, and `output_scale`. Return a
dictionary with the timestep or scale fields to override. `denoiser_scalings`
returns the standard coefficients for a supplied sigma. The
[NitroFusion example](dgl/sd/nitrosd-realism-nitrofusion.dgl) defines its timestep
rounding and output scaling in this function.

`cfg(pos, neg, scale=4.5, schedule=[7, 6, 5])` uses a scale per step and
falls back to `scale` after the list ends. `cfg` and `apg` both accept
`image_only` and `image_scale` for image editing. APG uses the underlying
model outputs and requires matching prediction scalings across its branches.
Each `sample` call starts its own APG momentum.

SLG takes a base prediction, a complete positive branch, and a skipped branch.
It adds `scale * (positive - skipped)` to the base prediction. Reusing the same
positive predictor in APG or CFG and SLG shares the forward pass within each
model evaluation.

```dgl
pos = predictor(dit, pos_cond)
neg = predictor(dit, neg_cond)
skipped = predictor(dit, pos_cond, skip_layers = [7, 8, 9])
guided = apg(pos, neg, scale = 4.5, eta = 0, momentum = -0.5)
pred = slg(guided, pos, skipped, scale = 2.8, end_sigma = t_to_sigma(pos, t = 0.8))
latent = sample(start, sigmas, predict = pred, rng = rng)
```

`start_sigma` and `end_sigma` define an inclusive sigma window for SLG.
To use the step-based window, set `start_step_fraction` and `stop_step_fraction`.
The step window uses the sigma tensor length (including its endpoint) times
each fraction, rounded down. Only step indices strictly inside the window
activate SLG. When both windows are provided, SLG applies within their
intersection.

Guidance schedules, SLG step windows, and `switch_predictor(after_steps=...)`
use logical steps starting at one. Multiple model evaluations within one solver
step, such as Heun's predictor and corrector, use the same step index.

CFG++ samplers request the unconditional prediction from the supplied
predictor. SLG and `switch_predictor` can forward this output from a two-branch
CFG or APG predictor. Every reachable switch branch must supply it; single-branch
predictors and CFG/APG with `image_only` are rejected before model execution.

### Random streams

`make_rng("cuda", seed = 42)` creates a seeded RNG object. The algorithm
can be `"cuda"` (Philox on every backend), `"cpu"` (MT19937), or
`"std_default"` (also `"std"`). `vae_encode`, `tensor::randn_like` and `sample`
each require an RNG object. Pass the same object to advance one stream in
execution order. Each call to `make_rng` creates an independent stream,
including when a cached execution context runs a script again.
For example, `tensor::randn_like(base, rng = make_rng("cpu", seed = 42))`
uses a separate MT19937 stream.

For the same RNG sequence as the generation C API, generate the initial noise with
`tensor::randn_like(base, rng = rng)`, pass that tensor to `add_noise`, then pass
the same `rng` to `sample`. For images, start a new stream from the image seed
before initial noise, and start another before the hires pass, including any
hires VAE re-encoding. For video, share the request stream from VAE encoding
through high- and low-noise sampling and hires refinement.

### Noise schedules and sampling

The sampler receives a start tensor, a rank-one F32 sigma tensor and a
predictor. `make_sigmas` returns a schedule tensor, and
`tensor::vector([1, 0.5, 0])` builds one from explicit values.
`sigmas[0]` reads the starting sigma. Return a schedule by name
to export it as one `.npy` array. A single zero sigma requests zero sampling
steps for ordinary samplers. `flow_shift` on `make_sigmas` configures that
table. Sampling uses the table's values and the predictor's inputs.

`make_sigmas`, `add_noise`, `t_to_sigma`, and `denoiser_scalings` take a predictor
as their first `predict` argument. These queries do not evaluate the predictor.
Build the required prompt/condition and predictor before constructing its
schedule or adding noise. Changing prediction does not reload the model or
its LoRA variant.

`make_sigmas` and `t_to_sigma` accept a composition when its branches share
one `DenoisingParameterization` object. Predictors from the same model with the
same resolved `prediction` share that object, so ordinary CFG can use `guided`
for its schedule, noise addition and sampling:

```dgl
pos = predictor(unet, pos_cond, prediction = "eps")
neg = predictor(unet, neg_cond, prediction = "eps")
guided = cfg(pos, neg, scale = cfg_scale)
sigmas = make_sigmas(guided, latent = base, scheduler = scheduler, steps = steps)
start = add_noise(guided, base, tensor::randn_like(base, rng = rng), sigma = sigmas[0])
latent = sample(start, sigmas, predict = guided, method = "euler_a", rng = rng)
```

Different models or different parameterization objects make the schedule source
ambiguous, even when their noise conventions or scaling rules are compatible.
Pass the branch that defines the desired schedule in that case. For example,
Wan 2.2 can use `make_sigmas(lo, ...)` for the low-noise model's schedule and
`add_noise(hi, ...)` before sampling with the high-noise model.
`denoiser_scalings` accepts a composition only when every branch has the same
scaling rules. For example,
a mixture of `eps` and `v` has no common coefficients, even if a queried sigma
happens to give equal values; query the desired leaf instead. `v` and `edm_v`
share scaling rules, but have different time mappings.

`add_noise` and sampling finish conversion require only a common noise
convention, so sampling still supports CFG or switched predictors with different
parameterizations such as `eps` and `v`. Compatible CFG branches with different
parameterizations execute separately. APG additionally requires matching
raw-output scalings.

`make_sigmas` defaults to the `"discrete"` scheduler and rejects `"auto"`.
Choose a different scheduler explicitly when the model requires it. Omit
`flow_shift` when no override is needed; its default is zero.

`sample(method=...)` defaults to `"euler"`. Choose a sampling method explicitly
when another algorithm is needed; `"auto"` is not accepted. `sefi::sample` also
defaults to `"euler"` and only supports that method.

`t_to_sigma(predict, t, flow_shift=...)` converts a normalized noise time
before the shift to a sigma. `t=1` is the noisy end, and `t=0` is the
low-noise end of the model's time mapping. A model with a discrete training
grid can have a positive minimum sigma. Pass `latent` or `sequence_length`
for time mappings that depend on the latent size.

In the [Wan 2.2 image-to-video program](dgl/wan/wan2-2-i2v-a14b.dgl), bracket slices
include the same boundary sigma in both sampling segments. The second Euler segment
takes the first segment's noisy output. Its final latent, the handoff and the
decoded pixel tensor can all be returned. The
[Wan 2.1 text-to-video program](dgl/wan/wan2-1-t2v-1-3b.dgl) uses one predictor
and the full schedule. `video::save` accepts a video dict containing decoded RGB
frame tensors and `fps_num`/`fps_den`, and writes AVI.

### Sampling caches

Pass `cache(mode="ucache", reuse_threshold=0.8)` to `sample(cache=...)` to
enable a sampling cache. Available modes are `disabled`, `easycache`, `ucache`,
`dbcache`, `taylorseer`, `cache_dit` and `spectrum`. Support depends on the
denoiser and sampler. Spectrum requires one denoiser, one sigma schedule and a
sampler without CFG++. Each sample call owns its cache state. The runtime
prepares each denoiser and active conditioning component on first use in that
call and reuses its execution scope for the remaining steps. With
`switch_predictor`, a later branch enters execution when the solver reaches it.
Spectrum discards its extrapolation history when an active switch changes
branches or an active CFG/APG guidance scale changes. The new configuration
starts with real model evaluations before extrapolation resumes. Guidance
changes inside unused branches do not invalidate the active history. Active SLG
nodes are part of the key even when they share every model leaf with the base
predictor. For CFG++, only the branch that supplies the root unconditional
output is additionally retained; nested guidance does not inherit that demand.
With caching disabled, execution skips the activity-key traversal entirely.

Condition-level caches (`easycache`, `ucache`, `dbcache`, `taylorseer` and
`cache_dit`) store separate histories for model conditions. Only enabled caches
prevent CFG batching; an unsupported cache request leaves batching available.
When an active condition leaves, its cache is invalidated. If it was the anchor,
the cache selects a new active condition and rebuilds its calibration.

DGL's `dbcache` reuses whole-model input/output residuals after checking input
change against `residual_diff_threshold`. `taylorseer` extrapolates each
condition's raw model output using `taylorseer_n_derivatives`; after enough real
samples, `taylorseer_skip_interval` bounds the number of extrapolated steps
between real evaluations. `cache_dit` combines that extrapolation with the
residual-difference gate. These are condition-level algorithms, not block-level
DBCache/TaylorSeer. `Fn_compute_blocks`, `Bn_compute_blocks` and
`scm_policy_dynamic` are not DGL options. Model-block caching requires a runner
integration and is not exposed by this path.

`max_warmup_steps`, `max_cached_steps`, `max_continuous_cached_steps`, and a `1`
in `scm_mask` force real computation as applicable; forced steps still update
history. Negative values for `max_cached_steps` and `max_continuous_cached_steps`
leave the corresponding limit unset. Condition-level reuse only runs on
completed-step evaluations; intermediate solver evaluations execute the model
normally.

Spectrum owns the composed predictor's output history independently. Its
`spectrum_warmup_steps` counts logical solver steps, and
`spectrum_stop_percent` sets the stopping position on that same timeline.
Heun's predictor and corrector use separate histories at their logical step
indices; neither evaluation advances the other's warmup. Active branch or
guidance changes reset both histories. Real Spectrum evaluations can still use
batched CFG. Logs report reused predictor evaluations, rather than dividing
Heun's evaluation count by the number of sigma intervals.

### Sampling previews

Load a tiny autoencoder with `load_tae("tae.safetensors", family = "sd1")`, or use
`family_from` to identify its model family from a diffusion checkpoint.
Pass it to `vae_encode` or `vae_decode` like any other autoencoder.
For preview-only use, set `decode_only = true` on `load_tae` and pass it to
`preview(decoder=...)`. Attach the returned observer to `sample` or
`sefi::sample`. Load the final decoder separately with `load_vae`.

```dgl
tae = load_tae("tae.safetensors", family = "sd1", decode_only = true)
observer = preview(decoder = tae)
latent = sample(start, sigmas, predict = guided, rng = rng, observer = observer)
```

`preview()` needs no decoder for latent projection. The context's preview
callback selects the mode, interval and noisy or denoised previews. A sampling
call without an observer uses a shared projection preview observer. Without a
decoder, preview falls back to projection. An explicit observer is only needed
to supply a decoder or custom preview settings. An observer can be reused
across sampling calls; each call has its own observation state. The observer
retains its decoder, and decoding resources are acquired only for a selected
preview and released afterwards. Without an enabled preview callback, it
performs no preview decoding.

### Media dimensions

Model functions adapt pixel inputs to their own spatial requirements. Dimensions
are rounded to the nearest legal multiple, with half-way values rounded up and
at least one multiple retained. Already compatible inputs pass through without
resizing. Automatic resizing uses Lanczos filtering; it does not impose a
resolution or pixel-area budget. Use `tensor::interpolate` or `tensor::crop`
before encoding when you want to choose the geometry yourself.

`vae_encode` normally aligns to the VAE's spatial compression factor.
`encode_prompt` independently applies its vision encoder's requirements; the
two paths can see different sizes of the same reference. Qwen Image 2.1 has an
additional model constraint: each vision image slot expands into four reference
latent tokens. Its VAE input therefore aligns to 32 pixels, while the spatial
compression factor remains 16. Neither path changes the source tensor. Existing
fixed-size vision preprocessing still performs its required crop and normalization.

`latent_shape` aligns requested pixel dimensions to the model family's diffusion
and VAE grid before returning a latent shape. Use that shape for the output
canvas; encoded reference latents can have a different shape. Both
`latent_shape` and video `vae_encode` round temporal lengths up to the supported
grid: `1 + 4k` for Wan, LingBot, Hunyuan Video and SeedVR2, `1 + 8k` for LTX,
and `5 + 17k` for H3 video. Encoding repeats the last frame when padding is
needed, preserving every input frame. Single-image encoding remains a single
image. Decoding returns the actual aligned size; there is no automatic crop
back to the original dimensions. For joint audio/video scripts, derive audio
lengths from the aligned video duration.

`detections::detect` rounds positive `input_size` values to a multiple of 32
(minimum 32, maximum 4096); detections remain in original-image coordinates.
The H3 latent upscaler similarly aligns positive target dimensions to even
latent units. LTX upscaling still gets its fixed scale from the checkpoint.
Only actual adaptations emit verbose logs. Invalid ranks, channels, nonpositive
dimensions and incompatible conditioning relationships remain errors.

### VAE latent spaces

`vae_encode(vae, pixels, rng)` returns a latent in the VAE's own space.
`vae_to_dit_latents(vae, latent)` applies the model's scaling, shift
or normalization before sampling. After sampling, use
`dit_to_vae_latents(vae, latent)` before `vae_decode(vae, latent)`.
Both spaces use float tensors. A saved tensor needs the matching conversion
for its space and VAE weights.
`vae_to_unet_latents` is an alias of `vae_to_dit_latents`, and
`unet_to_vae_latents` is an alias of `dit_to_vae_latents`; each alias resolves
to the same function and uses the same VAE-specific conversion.
Use the UNet names with UNet models and the DiT names with diffusion transformers.

`vae_encode` defaults to `posterior="sample"`. Set `posterior="mode"` to
take the posterior mode, as InstructPix2Pix does for its reference image.
The mode leaves the RNG state untouched. The two options give the same result
for deterministic encoders.

```dgl
rng = make_rng("cuda", seed = seed)
reference = vae_encode(vae, pixels, rng = rng, posterior = "mode")
predict = predictor(unet, condition, concat = reference)
latent = sample(start, sigmas, predict = predict, rng = rng)
image = vae_decode(vae, unet_to_vae_latents(vae, latent))
```

`vae_encode` and `vae_decode` accept `tiling=tiling(tile_size_w=32,
tile_size_h=32)`. Use `temporal_tiling=true` for supported video VAEs.
`predictor(circular_x=true, circular_y=true)` enables circular convolution
padding for that prediction branch and its active ControlNets. Branches can
share a model while using different settings; each forward call restores the
previous settings afterwards. Model weights are reused.

Set the same arguments independently on `vae_encode`, `vae_decode`, or
`preview(decoder=..., circular_x=true, circular_y=true)` for VAE operations
and preview decoding. Latent projection does not use convolution padding.
Configure predictor, VAE and preview padding consistently for seamless
generation. `sample` and `sefi::sample` do not accept circular padding options.

Load an F32 latent with `tensor::load` and use the matching latent space for the
receiving function.

Use `vae_decode(..., dtype = "f32")` to postprocess normalized float pixels in
`[width, height, channels, batch]` or `[width, height, time, channels, batch]`
layout. The default `"u8"` returns packed pixels.

## Image and video files

Media I/O uses `image::load` / `image::save`, `video::load` / `video::save`
and `audio::load` / `audio::save`. The former global names and the aliases
`read_image`, `read_video`, `read_audio` and `save` have been removed.
Model loading, inference and latent operations keep their current names.
`image::to_f32` and `image::to_u8` convert pixel representations for both images
and video frames; ordinary tensor dtype changes use `tensor::cast`.

### Image files

`image::load(path)` decodes PNG, JPEG, BMP and other formats supported by
stb_image into a U8 tensor shaped `[channels, width, height]`. Grayscale with
alpha is decoded as RGBA. `image::save(path, pixels)` writes a U8 pixel tensor
to PNG and returns no value. A write failure interrupts execution.

```dgl
# version: 0.0.0
fn main() {
    pixels = image::load("input.png")
    patch = tensor::crop(pixels, axes = [1, 2], start = [16, 32], size = [256, 256])
    output = tensor::interpolate(patch, axes = [1, 2], size = [512, 512])
    image::save("output.png", output)
    return output
}
```

Use `mode="nearest_exact"` for half-pixel nearest-neighbor sampling.
`mode="nearest_max"` uses nearest-neighbor sampling when enlarging and the
maximum of each source region when shrinking, preserving active mask pixels.

### Pixel formats

`image::to_f32(image)` converts image/video pixels to the F32
representation used by VAE and conditioning functions:

| Input | Output | Values |
|---|---|---|
| U8 image `[c, w, h]` | F32 `[w, h, c, 1]` | Divided by 255 |
| U8 video `[c, w, h, t]` | F32 `[w, h, t, c, 1]` | Divided by 255 |
| F32 pixels in either output layout | Same F32 layout | Preserved without rescaling |

The default preserves the input channel count, including masks and alpha.
Set `channels=3` to request RGB or `channels=4` for RGBA. `channels=0` explicitly
selects the default preservation behavior. Supported channel conversions
are grayscale to RGB/RGBA, RGB to RGBA (opaque alpha), and RGBA to RGB (drop
alpha). One- or two-channel data can be preserved, but RGB-to-grayscale and
grayscale-with-alpha conversion are not provided by this function. F32 input
with preserved or matching channels is borrowed without copying. For example,
`image::to_f32(mask)` keeps a single-channel mask, while a ControlNet RGB hint
can use `image::to_f32(hint, channels=3)`.

`image::to_u8(pixels)` reverses the layout conversion. It clamps normalized F32
pixels to `[0, 1]`, scales by 255 and rounds to U8, producing `[c, w, h]` for an
image or `[c, w, h, t]` for a video. It also accepts U8 pixels. For example,
`image::save("output.png", image::to_u8(pixels))` saves a processed F32 image.
Both functions return tensors; they introduce no separate image value type.

### Encoded media and bytes

`image::encode(image, format="png")` encodes a U8 pixel tensor in memory.
`video::encode(video, audio?, format="webm")` encodes a complete video using
the video dict's `fps_num` / `fps_den`. An optional audio dict is supported
with `format="avi"`. Both return an ordinary `{data: bytes, mime_type: string}` dict.

`bytes` is an immutable binary value, including embedded zero bytes. Assignments,
arguments and return values share its buffer. It is distinct from strings and
U8 tensors and is not converted to a flat tensor result.

```dgl
cover = image::encode(pixels)
clip = video::encode(video)
media::save("output/cover.png", cover)
media::save("output/clip.webm", clip)
return {cover = cover, clip = clip, latent = latent}
```

`media::save(path, media)` writes these exact bytes, with no re-encoding or
metadata changes. It checks file-write permission and the extension against the
MIME type. Omit the save calls for in-memory results only. The server's DGL
editor displays returned media from the existing job result; ordinary tensors
remain available for inspection.

| Encoding | MIME type | Requirements |
| --- | --- | --- |
| PNG | `image/png` | Built in |
| VP8 WebM | `video/webm` | `SD_WEBM=ON`; uses independent VP8 frames |
| MJPEG + PCM16 AVI | `video/x-msvideo` | `format="avi"`; offered as a download in the editor |

Unavailable encoders fail explicitly. WebM encoding rejects nonempty audio;
use `video::encode(video, audio=sound, format="avi")` for a PCM16 soundtrack.
Audio is never silently dropped. See [build options](build.md#webp-and-webm-support).
`video::load` still supports only PCM16 WebM audio: pass `audio=false` to read
frames from files with unsupported audio codecs.

Encoding embeds the current program when the context's `embed_metadata` option
is enabled. `media::save` preserves that decision. The existing `image::save`
and `video::save` functions remain available for direct file output; AVI file
saving continues to stream its packets instead of buffering the entire file.

### Reading video

Video frames use U8 tensors shaped `[3,width,height,frame_count]`, with
interleaved RGB channels. Media I/O groups frames and timing in a normal video
dict, and samples and sample rate in a normal audio dict. Computation operates
on the tensors inside these dicts.
`video::load` detects local files by their contents:

| Input | Supported data | Build option |
| --- | --- | --- |
| AVI | 24-bit RGB or MJPEG, PCM or IEEE float audio | Always available |
| WebP | Static images or one cycle of animation; no audio | `SD_WEBP=ON` |
| WebM | Independent VP8 keyframes and optional PCM16 audio, as written by this project's CLI | `SD_WEBM=ON` |

WebP animation decoding handles partial frames, blending and disposal before
returning full RGB canvases. Transparent pixels are composited onto black.
WebM interframe VP8, VP9, laced packets and delayed audio are unsupported;
unsupported audio reports an error suggesting `audio=false`. MP4 and segmented
OpenDML AVI are unsupported. Paths support spaces and Unicode.

```dgl
# version: 0.0.0

fn main() {
    input = video::load("input.avi", max_frames = 24)
    frames = tensor::interpolate(input.video.frames, size = [512, 512], axes = [1, 2])
    video = {frames = frames, fps_num = input.video.fps_num, fps_den = input.video.fps_den}
    video::save("output.avi", video, audio = input.audio)
    return { frames = frames, audio = input.audio }
}
```

Save the preceding program as `video.dgl` and run it with:

```sh
./build-dgl/bin/sd-cli -M dgl --script video.dgl --tensor frames frames.npy
```

`video::load` returns `{video, audio}`. `video` is a normal dict with `frames`
(tensor), `fps_num` and `fps_den`; `audio` is a normal dict with `samples`
(tensor) and `sample_rate`. With no audio stream, or
with `audio=false`, samples are an empty F32 `[1,0]` tensor and the sample rate
is 0. Passing this empty audio to `video::save` writes no audio track.

Loading defaults to all frames within the 256 MiB tensor limit. Set `max_frames`
to a positive value to read only that many initial frames. Exceeding the limit
fails instead of silently truncating. Frames are decoded as RGB at the first
video stream's rational frame rate. Setting `fps_num` and `fps_den` resamples
by repeating or skipping frames: each output timestamp uses the latest source
frame at or before that timestamp. `max_frames` applies after resampling.
WebP and WebM use frame timestamps to infer a fixed source frame rate.
Variable or unknown rates require explicit `fps_num` / `fps_den`, including
static WebP images without a frame duration. Static WebP returns a single frame
at the requested rate. For example, `video::load("still.webp", fps_num=24)`.
Animated WebP is read once regardless of its loop count. Streams must start at
time zero. AVI streams with dropped frames are unsupported. Audio is limited
to the selected video duration; AVI accepts PCM or IEEE float samples, and WebM
accepts PCM16 without codec delay, concatenating packet samples in stream order. Use `audio=false` to read frames
from files with unsupported audio codecs.

`video::load` also reads an image directory in filename order.
`image::load_dir(path, max_images=0)` returns a list of individual images in
filename order, which can have different sizes. A positive `max_images` limits
the list; zero loads every supported image. Both directory readers accept PNG,
JPEG and BMP. Use `video::load` when the directory represents a time sequence,
or `image::load_dir` to process independent images.

`video::split_windows(frames, size, overlap=0)` returns a list of frame tensors.
`video::join_windows(windows, overlap=0)` joins frame tensors, dropping the specified
overlap from the start of each window after the first, then concatenates the
remaining frames without blending. These operations only handle pixels;
keep frame rate and audio outside the window processing and pass them when saving.
A single full-length window or a single-window join borrows the input tensor.
Joined video has a 4 GiB limit. The 256 MiB loading limit
applies when reading source video.

### Writing video

`video::save(path, video, codec="auto", quality=90)` writes `.avi` only.
The video dict requires `frames`, `fps_num` and `fps_den`. Both frame-rate
integers must be in [1,1000000]; timing is explicit.
Saving accepts already allocated frame tensors larger than the loading limit
and encodes them one frame at a time. It sets the playback rate without
resampling frames. Supply `audio` for a soundtrack. `trim_audio=true` truncates
audio longer than the video duration;
shorter audio is retained unchanged. The default is `false`.

```dgl
video = {frames = pixels, fps_num = 24, fps_den = 1}
video::save("output.avi", video, audio = sound, trim_audio = true)
processed = {frames = processed_frames, fps_num = input.video.fps_num, fps_den = input.video.fps_den}
video::save("upscaled.avi", processed, audio = input.audio)
```

`codec="auto"` selects `mjpeg`; `quality` ranges from 1 to 90, with higher
values preserving more detail. Use `codec="rgb"` for uncompressed, pixel-exact
RGB output; it ignores `quality` and produces larger files. Both codecs support
odd frame dimensions. Audio is saved as float32 by default;
`audio_format="pcm16"` writes PCM16. AVI output is limited to less than 4 GiB.
Saving has no return value and does not alter the supplied frames or audio.
Return frame tensors, frame rate and audio explicitly when needed as graph
results. Audio trimming applies only to the file. MJPEG decoding need not match
the input pixels exactly.

### Audio files

`audio::load(path)` reads WAV with PCM8/16/24/32 or float32 samples.
`audio::save(path, audio, format="f32")` writes WAV, with `pcm16` also available.
`audio::load` and `audio_vae_decode` return a normal dict with `samples` (F32
`[channels,samples]` tensor) and `sample_rate` (Hz). Construct the same dict to
save generated or processed samples. Rates range from 1,000 to 384,000 Hz,
with 1..32 channels and a 256 MiB sample limit. Samples must be finite.

```dgl
sound = audio::load("input.wav")
audio::save("output.wav", sound)
track = {samples = sound.samples, sample_rate = sound.sample_rate}
video::save("output.avi", video, audio = track)
```

`encode_audio(encoder, samples, sample_rate=...)` and
`audio_vae_encode(vae, samples, sample_rate=...)` accept tensors and an explicit
sample rate. Their existing model-specific resampling remains in effect.
`audio_vae_decode` reports the decoder's actual output sample rate.

## Regions and masks

Use detections to select objects, regions to crop rectangles, and masks to
control which pixels a refinement replaces.

### Rectangular regions

A region collection is an ordinary dict `{width, height, boxes}`. Canvas dimensions
are integers in `1..16384`; `boxes` is an I64 tensor with shape `[4,N]`, limited to
10,000 boxes. Each column is `[x,y,width,height]` in source-image pixels and covers
`[x,x+width)` by `[y,y+height)`. Box sizes must be positive and boxes must lie
entirely inside the canvas. An empty collection has shape `[4,0]`.

`regions::from_boxes(width, height, boxes)` validates a flat `list<int>` or an I64
`[4,N]` tensor and constructs the dict. A valid input tensor is shared without a
copy. `regions::crop(image, regions)` accepts the same dict, checks its canvas
against the image and returns U8 patches in box order. `regions::paste(background,
foregrounds, regions)` resizes each patch with bilinear interpolation and copies
its pixels into the corresponding rectangle. Later boxes overwrite earlier ones,
including the alpha channel; pasting does not alpha-blend.

```dgl
r = regions::from_boxes(512, 512, [10, 20, 80, 60, 100, 40, 30, 50])
selected = {width = r.width, height = r.height, boxes = r.boxes[:, 1:2]}
patches = regions::crop(image, selected)
result = regions::paste(image, patches, selected)
```

The slice `[:,1:2]` preserves the `[4,1]` collection shape. `[:,1]` produces a
single `[4]` box. Handwritten dicts receive the same validation as constructed
collections. Boxes can be saved and loaded as ordinary tensors:

```dgl
tensor::save("boxes.npy", r.boxes)
restored = regions::from_boxes(r.width, r.height, tensor::load("boxes.npy"))
selected = {width = restored.width, height = restored.height, boxes = restored.boxes[:, 1:2]}
```

### Detections

Detection results are ordinary dicts with `xyxy` (F32 `[4,N]`), `scores`
(F32 `[N]`) and `class_ids` (I64 `[N]`). The fields share the same ordering and
count, limited to 4,096 detections. Empty results keep shapes `[4,0]`, `[0]` and
`[0]`. Scores must be finite and in `[0,1]`, class IDs must be nonnegative, and
coordinates must be finite.

`detections::detect(detector, image, confidence=0.3)` runs the detector once.
`detections::filter(image, detections, ...)` filters or sorts all three fields by
the same indices. `detections::box_masks(image, detections)` rasterizes full-canvas
rectangle masks. Filtering and mask creation require boxes inside the image.
`detections::to_regions(image, detections)` rounds starts down and ends up,
clips coordinates to the canvas and skips empty boxes, preserving order.

```dgl
found = detections::detect(detector, image, confidence = 0.3)
selected = detections::filter(image, found, top_k = 4, top_k_by = "confidence")
r = detections::to_regions(image, selected)
patches = regions::crop(image, r)
result = regions::paste(image, patches, r)
```

`regions::plan_mask_crops` returns per-mask `{box, mask, index}` tasks for `fold`,
not a region collection. Masked compositing can be expressed with a crop,
`tensor::lerp` and slice replacement:

```dgl
background = tensor::crop(image, axes = [1, 2], start = box[:2], size = box[2:])
patch = tensor::lerp(background, foreground, mask)
rows = tensor::crop(image, axes = [2], start = [box[1]], size = [box[3]])
band = tensor::replace_slice(rows, patch, axis = 1, start = box[0])
result = tensor::replace_slice(image, band, axis = 2, start = box[1])
```

### Detail refinement

`detections::detect(detector, image, confidence=0.3)` returns a dict with F32
`xyxy` `[4,N]`, F32 `scores` `[N]` and I64 `class_ids` `[N]`. Use `input_size`
to change the detector input size, `nms` for the overlap threshold, and
`max_detections` to cap the result at up to 4,096 objects.
`detections::filter` applies `min_area_ratio` and `max_area_ratio` before
selecting up to `top_k` boxes. Set `top_k=0` to keep every box between the area
limits. Use `top_k_by="area"` or `"confidence"` to rank them for selection,
and `sort_by` to set the output order (`"none"`, `"left_to_right"`,
`"center_to_edge"`, or `"area"`). Coordinates, scores and class IDs stay aligned.
Use `detections::box_masks` to rasterize the selected boxes, or
`detections::to_regions` to crop them without running detection again.

Apply mask transforms before passing full-canvas masks to
`regions::plan_mask_crops(masks, aspect=[1, 1], padding=32)`. Each nonempty mask produces
a dictionary containing `box` (I64 `[x, y, width, height]`), the original `mask`,
and its zero-based `index` among nonempty masks. Set `only_masked=false` to
use the whole canvas. Choose prompts and seeds in the script. Use `index` to
look up per-region values or to preserve seed offsets when selecting a subset.

The [ADetailer program](dgl/adetailer/adetailer-existing-image.dgl) uses `fold` to crop and
resize pixels from the latest canvas before inpainting each region. It
defines sampler settings in DGL and uses `tensor::gaussian_blur` on the
full-canvas mask for feathering. It resizes the decoded patch to the box,
blends with `tensor::lerp` and writes it back with `tensor::replace_slice`.
Crop resizing uses the selected box as the interpolation boundary.

### Mask composition

Use `tensor::translate` to shift a mask, `tensor::maximum` to merge masks and
`tensor::gaussian_blur` to soften their boundaries before blending a patch.
Convert U8 mask weights to floating point and divide by 255 before using them
with `tensor::lerp`. To create a binary U8 mask from RGB, call
`tensor::grayscale` and then `tensor::threshold` with `low=0, high=255`.
Swap those output values to invert the binary mask.

Use maximum pooling for dilation and minimum pooling for erosion. Specify
`pad_value=0` for zero-filled borders. For a four-pixel window anchored like
ADetailer, set `padding_before=[2, 2]` and `padding_after=[1, 1]` to preserve
the mask size. The Wan examples fill selected time indices with
`tensor::index_fill` to build a frame mask.

## Advanced component programs

Browse the [example index](dgl/README.md) for complete scripts organized by model.

### Image refinement

This example encodes an input image, samples the tail of a noise schedule and returns both the refined image and its latent.

```dgl
# version: 0.0.0

var {
    weights = "sd15.safetensors"

    input_path = "input.png"
    output_path = "output.png"

    prompt = "A small house in a forest clearing"

    seed = 42
}

fn refine(image, vae, unet, text, prompt, seed) {
    encode_rng = make_rng("cuda", seed = seed)
    resized = tensor::interpolate(image, axes = [1, 2], size = [512, 512])
    base = vae_to_unet_latents(vae, vae_encode(vae, resized, rng = encode_rng))
    rng = make_rng("cuda", seed = seed)
    pos = predictor(unet, encode_prompt(text, prompt), prediction = "eps")
    neg = predictor(unet, encode_prompt(text, ""), prediction = "eps")
    guided = cfg(pos, neg, scale = 7)
    full_sigmas = make_sigmas(guided, latent = base, scheduler = "karras", steps = 20)
    sigmas = full_sigmas[-13:]
    start = add_noise(guided, base, tensor::randn_like(base, rng = rng), sigma = sigmas[0])
    latent = sample(start, sigmas, predict = guided, rng = rng)
    return { image = vae_decode(vae, unet_to_vae_latents(vae, latent)), latent = latent }
}

fn main() {
    vae = load_vae(weights)
    unet = load_unet(weights)
    text = load_text_encoder(weights)
    refined = refine(image::load(input_path), vae, unet, text, prompt, seed)
    image::save(output_path, refined.image)
    return { image = refined.image, latent = refined.latent }
}
```

### Conditioning, inpainting and adapters

- [SD 1.5 ControlNet](dgl/sd/sd1-5-controlnet.dgl) and
  [Z-Image ControlNet](dgl/z_image/z-image-turbo-controlnet.dgl) load control
  weights with `controlnet::load`. Use `controlnet::apply(condition, model, hint)`
  to attach a control to a condition before passing it to `predictor`. Apply
  it again on the returned condition to add another control. Set each binding's
  `strength` and use `start_sigma` and `end_sigma` for its inclusive range
  `end_sigma <= sigma <= start_sigma`. Both bounds are optional. Use
  `t_to_sigma` to derive bounds from noise time, for example
  `end_sigma=t_to_sigma(pos, t=0.2)`. The same bounds apply after slicing
  the sigma tensor. Specify `type` on each binding of a classic Union
  checkpoint. Supply RGB float pixels for SD 1.5 or a packed 33-channel
  diffusion latent for Z-Image. Z-Image runs control blocks on the DiT compute
  backend as part of its graph.
- [Inpainting](dgl/sd/sd1-5-inpainting.dgl) separately encodes the original and
  masked pixels. Its `latent_mask` function resizes and broadcasts the mask.
  `tensor::max_pool2d` expands it for per-step blending. Pass the clean
  `base` and the `denoise_mask` to `sample`.
  Set `mask_input=true` for a program that anchors known pixels or keyframes at the
  input of each prediction as well.
- [IP-Adapter](dgl/ip_adapter/ip-adapter-sd1-5.dgl) encodes the image with
  `encode_vision` and projects both the conditional and unconditional vision
  embeddings with `ip_adapter::project`. Attach each set of tokens with
  `ip_adapter::apply(condition, adapter, tokens)` before constructing its
  predictor. Each call adds an adapter with its own `strength` and optional
  `start_sigma` and `end_sigma` bounds.
  [Classic and Plus use different vision inputs](models/ip_adapter.md#dgl).
- [PuLID](dgl/pulid/pulid-flux1-schnell.dgl) loads the attention weights with
  `pulid::load` and the precomputed identity with `pulid::load_embedding`.
  `pulid::apply(condition, pulid, embedding)` binds an identity to one
  predictor branch. Set its `strength` and optional `start_sigma` and
  `end_sigma` bounds.
- [PhotoMaker](dgl/photo_maker/photomaker-v1-sdxl.dgl) precomputes reusable identity
  tokens with `photomaker::encode_images`. `photomaker::encode_prompt` returns the
  plain condition, the prompt with identity slots, and a slot mask.
  `photomaker::apply` fuses the identity into that condition.
  `switch_predictor` selects when to use it.
  [PhotoMaker v2](dgl/photo_maker/photomaker-v2-sdxl.dgl) also passes the
  float32 face embeddings from `tensor::load` to `photomaker::encode_images`.
- [Inline prompt images](guides/prompt_images.md#dgl) use
  `<image:path>` in `encode_prompt`. DGL reads the file during execution.
  `image_bindings` can also supply tensors produced earlier in the graph.

- [ADetailer](dgl/adetailer/adetailer-existing-image.dgl) filters detections, rasterizes
  boxes and uses tensor pooling and blur to transform the masks. Use
  `tensor::maximum` in a `fold` to merge selected masks before planning regions.
  A separate `fold` composites each result before cropping the next one.

### Model-specific sampling

- [SeFi](dgl/sefi_image/sefi-image-5b-rl.dgl) uses `sefi::make_sigmas` for the texture and semantic
  schedules as F32 tensors and passes both explicitly to `sefi::sample`.
  The SeFi solver uses Euler and accepts repeated sigma values in either
  schedule. It shares mask, finish, cache and callback handling with `sample`.
  Slice out texture channels `[16, 144)` before converting them for the VAE.
- [AnimateDiff](dgl/animatediff/animatediff-realisticvision-v6-txt2video-cpu-offload.dgl) loads a motion module with
  `animatediff::load` and attaches it to a compatible SD 1.5 UNet with
  `animatediff::apply`. The frame count comes from the batch axis of the UNet
  latent. [Wan I2V](dgl/wan/wan2-1-i2v-14b.dgl),
  [VACE](dgl/wan/wan2-1-vace-1-3b-video2video.dgl) and [S2V](dgl/wan/wan2-2-s2v-14b.dgl)
  show the vision, control and driving-audio ports respectively.
- [LTX](dgl/ltx2/ltx-2-3-dev-txt2video.dgl) and [MiniMax-H3](dgl/minimax_h3/minimax-h3-text-to-av.dgl)
  build video and audio latents separately. Their `pack_av` and `unpack_audio`
  functions pack unbatched latents for joint prediction and recover the audio.
  Zero-padding the audio tail fills whole video channels. For a packed AV mask, broadcast
  the video mask to the video latent shape and fill the packed audio channels
  with ones. The video VAE and `audio_vae_decode` run separately before saving. Pass `audio=audio` and
  `trim_audio=true` to `video::save` to match the video duration, as in
  the MiniMax-H3 and Wan S2V examples. A video model can
  receive the audio latent length and frame rate explicitly on its predictor.
  The MiniMax-H3 script scales the clean audio by `video_shift / audio_shift`
  when packing it for sampling. It divides by that ratio when unpacking the
  final latent. For masked sampling, pack the clean `base` with the same
  scale. The predictor receives both shifts as explicit inputs.
  LTX text encoding also runs the video and audio embedding connectors and
  static caption projections. Pass a separate connector file to
  `load_text_encoder(connectors_path=...)`; use `projection_path` for static
  conditioning weights stored in the diffusion checkpoint. `encode_prompt`
  returns the prepared video condition in `crossattn` and the audio condition
  in `extra_crossattn[0]`. Both are cached together and passed directly to the
  DiT. `load_dit` does not accept `connectors_path`.

#### Reference inputs and timing

MiniMax-H3 reference blocks use a native `list<dict>` directly, both in
`predictor(..., reference_blocks=...)` and in the condition's `reference_blocks`
field. The former `reference_blocks(...)` packing helper has been removed.
Each block has a `kind` of `image`, `video`, `audio` or `video_audio`.
`image` and `video` require a `video` index into `ref_latents`; `audio` requires
an `audio` index into `ref_audios`; `video_audio` requires both. Indices are
zero-based nonnegative integers, and list order determines reference order.
If `ref_latents` is empty, video indices refer to the condition's `ref_images`.
The explicit argument overrides the condition field, including an empty list.
Malformed blocks or out-of-range indices fail when constructing the predictor,
with the block number and field in the diagnostic.

```dgl
p = predictor(dit, condition,
    ref_latents = [portrait, clip], ref_audios = [soundtrack],
    reference_blocks = [
        {kind = "image", video = 0},
        {kind = "video_audio", video = 1, audio = 0}
    ])
```

`ltx::video_positions` builds F32 `[2,3,N,1]` token bounds in seconds and pixels.
Pass the target latent before concatenating keyframes. `keyframe_frame_index`
is the keyframe's zero-based position in the output video;
`keyframe_pixel_frames` is its span in original frames;
`keyframe_latent_frames` is the number of appended latent frames. The former
`keyframe_frame` and `keyframe_frames` parameter names are removed.

`wan::s2v_audio_window(features, latent_frames=...)` aligns audio features to
Wan S2V's timeline. It returns `[feature_dim, 4 * latent_frames, layers]`,
resampling 50 Hz features to 30 Hz before selecting positions at 16 fps.
Samples beyond the audio are zero-filled. The old `frames` argument is removed;
use `latent_frames = tensor::shape(base)[2]` to follow the generated video.

### Upscaling and hires passes

- Image hires resizes the first-pass latent and chooses a second-pass sigma
  tail with tensor indexing. Keep the last `floor(steps * strength)` steps
  and the terminal sigma.
  [Video Hires](dgl/ltx2/ltx-2-3-spatial-upscale.dgl) loads a separate
  `load_latent_upscaler` resource, runs `upscale_latent`, then repacks
  the audio and samples a second time. LTX upscalers take and return VAE
  latents, with the scale inferred from the weights. MiniMax H3 upscalers
  take and return diffusion latents and accept target `width` and `height`
  in latent units. Both adapters accept `[W,H,T,C]` or `[W,H,T,C,1]`,
  with channels matching the checkpoint. A single frame uses `T=1`; ordinary
  image latents `[W,H,C,B]` are a different layout and are not interchangeable.
  The latent must also come from a compatible model/VAE. SeedVR2 uses the
  separate pixel-resize, VAE encode, DiT sampling and VAE decode workflow;
  it is not a `load_latent_upscaler` backend. The former `upscale_video_latent`
  name is removed without an alias.
- [RelayVSR](dgl/relay_vsr/relayvsr-x2-video-upscale.dgl) selects keyframes with tensor
  functions and assembles a predictor condition with the checkpoint's
  `crossattn` embedding and the low-resolution `lq_frames` tensor. A single
  Euler step produces exportable keyframe latents. `relayvsr::decode` reads
  those latents, the I64 keyframe indices, and the original U8 frames to
  reconstruct the video. Use a matching 2x or 4x DiT and Flash VAE.
- [SeedVR2](dgl/seedvr2/seedvr2-3b-video-upscale.dgl) loads its DiT and VAE with
  `load_dit` and `load_vae`. The example resizes and pads each pixel window
  with tensor functions, encodes it with `vae_encode`, and explicitly
  assembles the predictor's `crossattn` embedding and `concat` condition.
  A shared RNG feeds the noise and the one-step Euler sampler. The script
  decodes and crops each result before joining the windows. `video::join_windows` discards the
  specified overlap from the returned frame tensors. Frame rate and audio are
  passed separately to `video::save` after joining.

### Function names and namespaces

Generic model components and inference stay at the top level. Family- or
extension-specific operations use their own namespace, such as
`controlnet::load` / `controlnet::apply`, `ip_adapter::project` and
`photomaker::encode_images`. Extension `apply` functions retain their own
execution and resource-lifetime rules; their names do not imply deferred execution.

Generic `load_lora`, `apply_lora`, `encode_prompt`, `sample`
and component loaders remain top-level.

The following global names have been replaced without aliases. Parameter names,
argument order, defaults and result fields are unchanged:

| Previous name | Current name |
|---|---|
| `load_controlnet` | `controlnet::load` |
| `apply_controlnet` | `controlnet::apply` |
| `load_ip_adapter` | `ip_adapter::load` |
| `ip_adapter_tokens` | `ip_adapter::project` |
| `apply_ip_adapter` | `ip_adapter::apply` |
| `load_photomaker` | `photomaker::load` |
| `photomaker_tokens` | `photomaker::encode_images` |
| `encode_photomaker_prompt` | `photomaker::encode_prompt` |
| `apply_photomaker` | `photomaker::apply` |
| `load_pulid` | `pulid::load` |
| `load_pulid_embedding` | `pulid::load_embedding` |
| `apply_pulid` | `pulid::apply` |
| `load_animatediff` | `animatediff::load` |
| `apply_animatediff` | `animatediff::apply` |
| `ltx_video_positions` | `ltx::video_positions` |
| `s2v_audio_window` | `wan::s2v_audio_window` |
| `sefi_sigmas` | `sefi::make_sigmas` |

## C API

Include `stable-diffusion.h` for the DGL and tensor APIs.

Structured results expose `bytes` as `SD_DGL_VALUE_BYTES`.
`sd_dgl_value_get_bytes(value, &data, &size)` returns a borrowed `const void*`
and length, with no terminator or text conversion. The pointer remains valid
until `sd_dgl_result_free`. Existing enum values and getters are unchanged.

`sd_dgl_check` checks source without creating a context or loading models.
Pass `false` for a self-contained script, as the CLI's `--check` does, or
`true` to allow named tensor parameters in `fn main(...)`.

```c
const char* source = "fn main() { return load_unet(\"model.safetensors\") }";
sd_status_t status = sd_dgl_check(source, false);
if (status != SD_STATUS_OK) {
    fprintf(stderr, "%s\n", sd_get_last_error());
}
```

Execution requires an explicit context. Create one with `sd_dgl_ctx_create`
to share loaded components across scripts, or `sd_dgl_ctx_create_once` to
release components after their last use in each run. Configure it while idle,
then pass source directly to `sd_dgl_run`. Every run compiles the source once;
there is no public compiled-program handle or graph cache. A normal execution
does not need a preceding `sd_dgl_check` call.

This example configures a context, runs a script without host inputs, and reads
its structured result:

```c
const char* source = "fn main() { return { pixels = tensor::zeros([2, 3]) } }";
sd_dgl_ctx_t* ctx = sd_dgl_ctx_create();
sd_dgl_result_t* result = NULL;
if (ctx && sd_dgl_ctx_set(ctx, "max_ram", "24")) {
    sd_status_t status = sd_dgl_run(ctx, source, NULL, 0, &result);
    if (status == SD_STATUS_OK) {
        const sd_dgl_value_t* root = sd_dgl_result_root(result);
        const sd_tensor_t* pixels = sd_dgl_value_tensor(sd_dgl_value_field(root, "pixels"));
        sd_tensor_save(pixels, "pixels.npy");
    } else {
        fprintf(stderr, "%s\n", sd_get_last_error());
    }
}
sd_dgl_result_free(result);
sd_dgl_ctx_free(ctx);
```

Keep the context alive until all calls using it return. Free it with
`sd_dgl_ctx_free` when its cached components are no longer needed.

### Host tensor inputs

Standalone scripts can read tensors with `tensor::load(path)` or embed them
with `tensor::literal`. A C host can pass source with named tensor parameters
directly to `sd_dgl_run`. Supply exactly one binding per parameter. Compilation
always permits these parameters; a missing binding reports a missing input.
Unknown or duplicate names and null tensors are also rejected.
Bindings share the existing immutable tensor handles. The same source can be
run with different bindings on subsequent calls; each call compiles it anew.
Use `sd_dgl_check(source, true)` when only validation is needed.

```c
/* ctx is an existing DGL context and pixels is an immutable tensor. */
const char* source = "fn main(pixels) { return { image = pixels, label = \"input\" } }";
sd_dgl_result_t* result = NULL;
const sd_dgl_tensor_input_t input = {"pixels", pixels};
sd_status_t status = sd_dgl_run(ctx, source, &input, 1, &result);
if (status == SD_STATUS_OK) {
    const sd_dgl_value_t* root = sd_dgl_result_root(result);
    const sd_tensor_t* image = sd_dgl_value_tensor(sd_dgl_value_field(root, "image"));
    sd_tensor_save(image, "pixels-copy.npy");
}
sd_dgl_result_free(result);
```

### Context execution settings

Set these options with `sd_dgl_ctx_set` while the context is idle or use the
corresponding `sd-cli` and `sd-server` flags. All models in the context use the
same settings.

| Context key | CLI and server flag | Default |
| --- | --- | --- |
| `n_threads` | `--threads N` or `-t N` | Physical core count |
| `auto_fit` | `--auto-fit on\|off` | On |
| `batched_cfg` | `--batched-cfg on\|off` | On |
| `embed_metadata` | `--disable-image-metadata` to turn off | On |
| `enable_mmap` | `--mmap` | Off |
| `eager_load` | `--eager-load` | Off |
| `disable_prefetch` | `--disable-prefetch` | Off |
| `disable_segmented_compute` | `--disable-segmented-compute` | Off |

The context accepts a thread count from 1 to 4096 and boolean strings such as
`"on"` and `"off"`. Changing threading, placement or weight-loading options
discards cached models so they can be loaded with the new settings on next use.
Models using graph-cut layer splitting prepare weights lazily even when eager
loading is enabled.

`image::save` (PNG) and `video::save` (AVI) embed the exact executed DGL source
by default. Set `sd_dgl_ctx_set(ctx, "embed_metadata", "off")` to disable this
for all saves in that context. The CLI forwards `--disable-image-metadata` to
this setting in DGL mode. Changing it leaves loaded models cached. Metadata
records the program text; host tensor contents and input files are not embedded.

`batched_cfg` combines compatible positive and negative `cfg` branches into one
UNet forward with batch size two. The branches must use the same model, matching
conditioning shapes and matching circular settings. Predictors with transforms,
references, ControlNet, IP-Adapter, PuLID, AnimateDiff or active sampling caches
retain separate forwards, as do branches using different models or incompatible
shapes. Branch selection through `switch_predictor` is resolved before checking
compatibility. CFG with scale one only computes the positive branch unless the
sampler also needs the unconditional prediction. Batching uses more compute
memory and can produce small floating-point differences. Set `batched_cfg` to
`"off"` to force separate forwards. Changing it affects subsequent runs without
invalidating loaded models.

Register RPC devices in the host with `sd_add_rpc_servers` or use
`--rpc-servers HOST:PORT` at CLI or server startup. A loader can then select a
registered RPC device with its `backend` option.

Set `conditioning_cache_size` with `sd_dgl_ctx_set` or use
`--conditioning-cache-size` with `sd-cli` or `sd-server`. The default is 4 entries
per text encoder instance, including each LoRA variant. Lowering the capacity
evicts the least recently used entries immediately. A capacity of 0 clears the
caches and disables caching. Loaded encoders and their weights are reused when
the capacity changes.

### File access and memory budgets

Install a file access callback with `sd_dgl_ctx_set_file_access_callback` before
running scripts in a context. The callback receives a UTF-8 path and an access
type for input reads, model reads or writes. Returning false ends the run with a
file access error. Keep its user data valid throughout the run. Model reads
include checkpoints, referenced shards and external tokenizers. Without a
callback, file access follows the process's permissions.

#### Memory budgets

```c
sd_dgl_ctx_t* ctx = sd_dgl_ctx_create();
sd_dgl_result_t* result_a = NULL;
sd_dgl_result_t* result_b = NULL;
if (ctx && sd_dgl_ctx_set(ctx, "max_ram", "24") &&
    sd_dgl_ctx_set(ctx, "max_vram", "cuda0=12")) {
    /* source_a and source_b can reuse the same encoder and VAE. */
    sd_status_t status = sd_dgl_run(ctx, source_a, NULL, 0, &result_a);
    if (status == SD_STATUS_OK) {
        status = sd_dgl_run(ctx, source_b, NULL, 0, &result_b);
    }
    if (status != SD_STATUS_OK) {
        fprintf(stderr, "%s\n", sd_get_last_error());
    }
}
sd_dgl_result_free(result_a);
sd_dgl_result_free(result_b);
sd_dgl_ctx_free(ctx);
```

The context's `max_vram` applies to all components on each device. Set the
context limits from the CLI with `--max-vram` and `--max-ram`. Idle GPU weights may
stay resident between runs. Under pressure the context reclaims idle buffers
and reloads weights from their configured parameter backend when needed.
With `params_backend="disk"`, weights are read on demand and their buffers
are released after use.

`max_ram` caps allocated host weight buffers across the context, including
pinned host buffers used by CUDA. Set it in GiB with `sd_dgl_ctx_set`.
A value of `0` disables the explicit cap and keeps the system memory pressure
check, which subtracts 1 GiB from the available RAM reported by the OS.
On Windows, this also accounts for the remaining commit capacity, including
CUDA weight staging, prefetch, and compute buffer allocations under WDDM. Device
allocations reserve commit capacity without charging their size to physical RAM. Graph
metadata allocations request space from the same memory manager and can reclaim
idle weights before creating a scheduler. Temporary graph memory does not count
against the host weight limit.
For CPU execution, automatic placement checks weight storage against this cap
and separately checks weights plus workspace against available system memory.
Lowering the cap trims idle weights immediately. Before a new allocation,
the context can evict old CPU weights together with their idle GPU copies, or
unload an idle component. Reclaimed weights reload from their source files on
the next use. A request whose active weights cannot fit reports the required
and available capacity. Automatic placement also uses the cap when choosing
where to keep new weights.

The cap covers weight buffers. Allow separate RAM for graph workspaces,
intermediate tensors and other process allocations. The OS manages pages in
read-only weight mappings. With a cap enabled, weights that need writable
mappings for in-place LoRA merges use allocated buffers.

#### Automatic weight placement

Set `backend` and `params_backend` on loaders in the script. The context's
`auto_fit` setting chooses placement when `params_backend` is empty; it is
enabled by default and cannot be overridden by individual loaders.
Each run prepares one weight-placement plan for the graph, including models
already cached in the context. Compilation tracks actual model use separately
from metadata queries and model captures. For example, `latent_shape` does not
need VAE weights, while a VAE preview observer keeps the decoder in the sampling
working set when VAE/TAE preview callbacks are enabled. Unrelated scalar
operations do not change the model-use phases.

If a loader omits `backend`, the run compares available GPU budgets before
constructing runners and uses the selected device consistently for default
loaders. Explicit backend choices and split layouts remain unchanged.

Before computation, the runtime constructs missing runner metadata without
loading all weights, then combines registered parameter sizes, phase working
sets, reclaimable cached buffers, device capacity and the context budgets.
Automatic placement prioritizes diffusion models (`load_dit` / `load_unet`)
over other components. It reserves their device storage first when the stage
working set fits with other components offloaded. Larger models are considered
first within each priority group.
Explicit parameter backends and split layouts remain constraints. Automatic
parameter storage can move between the compute device, CPU and disk on later
runs without reconstructing a compatible model. Under pressure, weights with
no remaining use or the farthest next use are reclaimed first. Idle weights
remain cached when there is no pressure.

GGML sizes compute workspaces from the actual graphs built during execution.
The first plan reserves conservative workspace headroom; actual workspace
allocations and changes in external memory use are still checked at runtime.
Models first encountered inside an iteration join the remaining plan when
their loader executes. LoRA buffers and merged variants also retain runtime
allocation checks.

### Progress, preview and cancellation

Register callbacks on an idle context with `sd_dgl_ctx_set_progress_callback`
and `sd_dgl_ctx_set_preview_callback`. They run synchronously on the executing
thread and receive events for that context's current run. A null callback
clears the registration. The sampling pass counter starts at one on each
run. Progress can report model loading in tensors, sampling in steps, video
upscaling in frames, or a stage with no known total.
Without a progress callback, the DGL execution layer displays sampling and
weight-loading progress bars. With a callback, the host controls their display.

The default observer handles previews. Preview mode `proj` uses latent projection. Modes `tae` and `vae`
use the decoder supplied to `preview(decoder=...)`, falling back to projection. A positive interval
previews each matching completed logical step. Intermediate model evaluations,
such as Heun's first evaluation, do not produce duplicate previews or progress
events. Zero previews the final step of the first sampling pass.
A negative interval selects the step at its absolute value. The callback
receives the pass number, total steps and borrowed frame tensors. Use
`sd_tensor_retain` if a frame must outlive the callback.

`sd_dgl_ctx_cancel` can be called from a callback or another thread. An
idle context accepts the call without queuing a cancellation. The current
run stops at an execution boundary, such as between sampling steps or
after an in-flight model load completes. Cancellation requested during compilation
is checked after compilation, before binding inputs and executing. The API returns
`SD_STATUS_CANCELLED` with a null result. Files already written by the
script remain on disk. Keep the context alive
until calls using it have returned. Concurrent or reentrant runs on the
same context return `SD_STATUS_BUSY`.

### Results and diagnostics

`sd_dgl_result_root` exposes the returned value with typed scalar access,
tensor handles and nested dict or list children. Audio and video reads return
ordinary dicts and tensors, using the same result traversal as other values.
Region collections are ordinary dicts with integer `width` and `height` and an
I64 `boxes` tensor shaped `[4,N]`, with `[x, y, width, height]` per column.
A root region dict flattens to `boxes`, `width` and `height`; a field named
`regions` flattens to `regions.boxes`, `regions.width` and `regions.height`.
Export root coordinates with `--tensor boxes boxes.npy`, replacing the old
`--tensor result boxes.npy`. Cast old I32 files to I64 before reuse. Runtime objects such as models and RNGs appear as opaque values.

A function without a return value has result type `void`, no output tensors,
and an `SD_DGL_VALUE_VOID` root through the structured C API.

A single returned tensor is named `result`. Named returns retain their names;
lists add numeric path components. Conditioning expands into fields such as
`positive.crossattn` and `positive.vector`. Use `sd_dgl_result_count/name/tensor`
to enumerate the actual outputs. `sd_tensor_save` writes `.npy` using the reversed
shape so NumPy sees normal row-major dimensions; RGB becomes `(height,width,3)`.

In the flat tensor view, strings are UTF-8 U8 arrays without a terminating zero
byte.

Values and tensors are borrowed from the result. Use `sd_tensor_retain`
to keep a tensor past the result's lifetime. Results remain valid after
freeing the context. All destructors accept `NULL`.

Failed runs leave `*result_out` as `NULL`. Both check and run return
`sd_status_t`; compare it explicitly with `SD_STATUS_OK`. Use
`sd_get_last_error()` for the thread-local diagnostic for DGL and other C
API calls. `sd_dgl_error()` reads the same diagnostic. Syntax and
execution diagnostics include source locations when available.
