# ggml operator extensions

The patched ggml exposes `ggml-ext.h` for application-owned operators, kernel
overrides, and graph fusion rules. Registrations are in-process and scoped to a
backend device. Register them before capability queries or graph allocation;
callback code and user data must remain alive until all dependent graphs and
backends have been destroyed. Registration names are unique, and duplicate
registrations fail rather than replacing live implementations.

## Operator definitions

`ggml_ext_register_op` registers a versioned name, attribute size, and an output
metadata inference callback. `ggml_ext_op_create` creates an ordinary ggml tensor
with explicit input dependencies. The initial interface supports one contiguous,
out-of-place output with an existing `ggml_type`.

Attributes are copied into the unused portion of `op_params`. They are small,
plain values; use `memcpy` when reading attributes that need stricter alignment.
Auxiliary tensors must appear in `src`, not in an attribute pointer or user data.
Registered operators use `GGML_OP_CUSTOM` internally, with an identifiable
callback marker. Unsupported devices do not mistake them for CPU callbacks.
These process-local descriptors are not a graph serialization format.

## Kernel selection

`ggml_ext_register_kernel(device, &desc)` associates a kernel with a built-in
`ggml_op`, or with a registered custom operator. `supports` describes correctness
constraints; `preferred` optionally decides whether to override an available
native implementation. Higher priorities are considered first. When native
execution is unsupported, a supported extension can run regardless of its
preference callback.

Capability queries include registered single-node implementations, so the
scheduler can select a device even when its native implementation rejects the
operator. Callbacks inspect metadata only and must not recursively call the
public capability query for the same device and operator. A fusion-only
registration does not claim single-node support.

`ggml_ext_dev_supports_native_op` queries the original device implementation
without extension dispatch. Fusion policies can use it to select a supported
extension even when their performance preference normally favors native kernels.

`prepare` creates optional host-side plan state and reports workspace bytes.
It can run during memory measurement, before tensor addresses are assigned, and
must not launch work or allocate device buffers. Workspace is allocated by ggml
before execution and included in scheduler memory estimates and buffer-size
reports. `execute` receives the backend, workspace buffer, and current tensor
addresses. `destroy` releases prepared state. A preparation or execution error
is propagated; execution never retries a native implementation after launch.

Workspace is scratch storage shared by sequential extension steps in a plan;
its capacity is the maximum request, not the sum across nodes. Contents are not
preserved between calls. Queue work in backend execution order, and finish any
private-stream use before subsequent backend work can access the workspace.
Persistent state belongs in `prepare`/`destroy` state or explicit tensor inputs.

Overriding a built-in operator preserves its numerical and aliasing contract,
including allocator reuse of input storage. Explicit views/in-place nodes are
not intercepted by this first interface. Rebuild the scheduler allocation after
changing graph metadata, selection policy, or registrations.

## Fusion

A `match` callback returns the number of consecutive nodes it can replace.
The scheduler freezes this selection before allocating tensors. ggml checks
compute flags, external uses, output flags, and views; the initial implementation
accepts a conservative set of pure operators with only the final node exposed
as the fused output. The rule remains responsible for mathematical equivalence,
dtype, shape, layout, and hardware constraints. Internal views are accepted when
their storage belongs to the fused subgraph. In-place intermediates are accepted
only when their storage owner is an earlier node in the same match. External
consumers and output flags still prevent eliminating those results. The final
output must own storage, except for a terminal `CPY`, which retains its original destination and write
semantics. The extension must handle input/output storage overlap, including
allocator reuse, with scratch space when necessary. Other in-place writes are
not eligible for extension fusion when they target storage outside the match.

An optional `optimize` callback runs before matching and allocation. Use
`ggml_ext_graph_delay` to move a producer range after independent nodes. The
helper rejects dependencies, unknown side effects, and writes that alias moved
inputs or outputs. It can cross independent copies and known in-place operators
when their storage does not alias the moved inputs or outputs. Owning dtype
conversions can also be moved or crossed.
Reordering must preserve edges and tensor metadata. Fusion legality is checked
again after reordering.

External inputs stay allocated through the final node. Native graph optimization
and execution operate on the remaining contiguous ranges, so native fusion cannot
consume nodes selected by an extension. Intermediate allocations are retained
in this initial implementation; only their computation is removed.

Automatic extension fusion is enabled in scheduler execution with one pipeline
copy and without evaluation callbacks. Multiple pipeline copies and evaluation
callbacks retain individual operations. Direct backend graph computation supports
single-node extensions; it does not fuse an already allocated graph. Direct calls
use temporary plans and synchronize before releasing them; scheduler plans and
workspaces are reused across executions.

Use the backend graph APIs or the scheduler. The legacy CPU `ggml_graph_compute`
and backend `graph_plan_*` interfaces do not dispatch these extensions.

## Backend services

Services are discovered through `ggml_backend_reg_get_proc_address`:

| Service | Purpose |
| --- | --- |
| `ggml_ext_cpu_parallel_v1` | Run an application callback on the CPU backend's configured thread pool, with its abort callback. |
| `ggml_ext_cuda_context_v1` | Borrow the current CUDA device and stream, ordered with native work. |
| `ggml_ext_cuda_device_v1` | Get the physical CUDA ordinal for capability checks before backend creation, including virtual ggml devices. |
| `ggml_ext_cuda_blas_v1` | Borrow the current stream's cuBLAS handle during execution, with host scalar pointers. |

The borrowed cuBLAS handle remains owned by the backend. Extensions must not retain
or destroy it, or change its stream, pointer mode, math mode, or workspace. This
service is exposed by the CUDA backend, not HIP/MUSA.

Workspace is a backend buffer. Its base is a CUDA device pointer on CUDA; callers
must use the representation appropriate to their backend. A Vulkan command
recording service is not implemented yet.

## sd.cpp integration

`src/core/ggml_extensions.cpp` registers the CPU implementation of
`sd.rope_apply.v1`, an override for existing `GGML_OP_ROPE_APPLY` nodes, and a
`RMSNorm -> Mul -> RoPE` fusion. The fusion accepts a one-dimensional weight
vector. Both interleaved and split-half RoPE, partial rotation, and strided input
rows are supported. Model construction uses the custom operator on CPU and CUDA
and retains existing backend implementations elsewhere.

`src/core/cuda/rope.cu` supplies standalone CUDA RoPE, the same builtin override,
and `RMSNorm -> Mul -> RoPE [-> Scale -> FP16 Copy]` fusion. CuTe attention shares
its RMSNorm/RoPE kernel. Partial rotation and arbitrary float-aligned input and
position-encoding strides are supported by the standalone kernel; RMSNorm fusion
requires contiguous position encoding and weights. Copy fusion uses workspace
when the destination overlaps an input. These registrations do not require
CUTLASS. The old CUDA RoPE dispatch is removed; HIP and MUSA retain their existing
implementations.

`SD_USE_UPSTREAM_GGML` excludes these registrations. The stable-diffusion public C
API is unchanged. INT8 ConvRot retains its existing packed activation layout and
separate weight scale tensor.

`src/core/cuda/cute_attention.cu` supplies CuTe Flash Attention on CUDA when
CUTLASS headers are configured. It registers an override of `FLASH_ATTN_EXT`,
with the existing sequence-length preference and native fallback. It also
registers Q preparation (`RMSNorm -> Mul -> RoPE`), output scaling, and packed
ConvRot output fusion. These kernels use only the public extension API, borrowed
CUDA stream, and scheduler-owned workspace; they do not use ggml's CUDA pool or
private conversion helpers. The existing CUDA fusion and CuTe disable environment
variables are honored.

`src/core/cuda/i8_projection.cu` supplies the INT8 GEMM projection epilogues:
`GEMM -> Reshape -> RMSNorm -> Mul -> RoPE`, optional scaling and FP16 copy,
and K/V prefix concatenation with FP16 output. CuTe can consume a directly
produced FP16 Q without materializing the FP32 GEMM or RoPE intermediates.
Independent affine operands, including broadcasted constants, are computed before
their normalization/projection
chain when graph dependencies allow it. The attention optimizer can then move
the complete five-node projection when safe.
These rules accept both builtin and custom RoPE nodes. They preserve the original
large-matrix selection thresholds, full interleaved head dimension 128, packed
activation layout, and separate weight scales. External consumers prevent fusion;
aliasing terminal copies use scheduler workspace. The CUDA projection kernels
and matchers now live in sd.cpp. `GGML_CUDA_FORCE_CUBLAS` disables these epilogues.

`src/core/int8_ops.cpp` defines `sd.linear_i8.v1` and
`sd.quantize_i8_convrot.v1`, with CPU implementations and overrides of the old
builtins. The CPU kernels use the configured thread pool and scheduler workspace;
the old CPU INT8 GEMM and ConvRot implementations have been removed from ggml.
Custom nodes execute on CPU or CUDA when supported. Model code retains builtin
nodes while the existing CUDA and Vulkan fusion rules are being migrated.

For already quantized activations, `sd_ggml_mul_mat_i8` accepts ordinary I8 weight
and activation tensors, a separate F32 activation scale tensor with one value per
row, optional F32 weight scales with one value per output channel, and optional
bias. It computes `float(int32_dot(W, X)) * X_scale * W_scale + bias`, in that
order. All scales are explicit graph dependencies; there is no quantized-tensor
wrapper. Shapes use ggml's low-to-high dimension order. Call
`sd_ggml_register_extensions` before scheduling these custom nodes.

`sd_ggml_linear_i8` also accepts F32 activations (quantized into temporary
workspace) or the output of `sd_ggml_quantize_i8_convrot`. ConvRot keeps the existing
packed I8 activation rows, padded to a multiple of four rows, followed by F32 row
scales. Weight scales remain separate. CPU ConvRot groups are 1, 4, 16, 64, or 256.
Unsupported layouts and malformed scale shapes are rejected. Separate CPU quantize
and GEMM phases propagate cancellation, and overlapping outputs use workspace.

`src/core/cuda/int8.cu` implements the same custom schemas on CUDA. It uses the
borrowed cuBLAS handle for I8-by-I8-to-I32 GEMM and application-owned quantization,
ConvRot, and rescaling kernels. F32 inputs are quantized into workspace; ordinary
I8 inputs retain separate scale tensors and use a padded temporary only when
needed by cuBLAS. Packed ConvRot inputs are consumed directly. Large ConvRot rows
use multiple kernels when they exceed shared memory. Aliasing outputs use scratch.
The registrations also intercept builtin INT8 GEMM and standalone ConvRot nodes.
CUDA GEMM requires Turing or newer, contiguous tensors, K/N divisible by four,
float-aligned inputs, and row counts that fit a padded signed 32-bit dimension;
CUDA ConvRot uses groups of 256.
Unsupported custom nodes can fall back to CPU through the scheduler.

When CUTLASS is configured, `src/core/cuda/i8_gemm.cu` supplies dense and SwiGLU
INT8 GEMM kernels on Ampere or newer. It preserves the native tile selection,
small-row policy, Stream-K scheduling, Ada column bands, and separate scaling
order. The same extension matches `GEMM -> SwiGLU [-> ConvRot]` and row
concatenation for builtin and custom GEMM nodes. Weight scales remain explicit
inputs, including the gate/up scale permutation recorded on every SwiGLU launch.
Plans reserve bounded Stream-K workspace, cache occupancy on first execution,
and initialize barriers on every launch, so CUDA Graph replay observes updated
inputs and scales. External consumers retain ordinary execution; overlapping
outputs use workspace. The CUTLASS SwiGLU/ConvRot chain can write its packed
output directly after GEMM, because quantization reads only its workspace
intermediate. Shapes or alignments unsuitable for CUTLASS use cuBLAS.
Custom I8 nodes also retain cuBLAS when the reduction could overflow I32, to
preserve their previous accumulation behavior. Fusion disable and force-cuBLAS
settings are honored. No CUTLASS headers are needed for the cuBLAS path.

The native CUDA CUTLASS GEMM implementation and its SwiGLU/row-concat matchers
have been removed from ggml. Native cuBLAS GEMM and standalone ConvRot remain
compatibility implementations for callers without these registrations. HIP
retains its native INT8/activation fusion paths.

`src/core/cuda/i8_gated_residual.cu` implements
`INT8 GEMM -> [Scale] -> Mul(gate) -> Add(residual) -> [Concat]` as a registered
CUDA fusion, for both builtin GEMM and custom I8 nodes with separate activation
scales. The corresponding native CUDA fusion rules and CUTLASS residual epilogue
have been removed from ggml. Its optimizer moves independent gate producers ahead
of GEMM and optional scaling through `ggml_ext_graph_delay`, before fusion matching.
The former native INT8 affine reorder rule has also moved into this extension.
It preserves the original tile selection, small-row
threshold, Stream-K scheduling, Ada column bands, and separate FP32 rounding of
gate multiplication and residual addition. The plan bounds Stream-K workspace
over possible occupancy values using host metadata; execution verifies the
actual requirement against that bound. Device occupancy is cached on first
execution; each launch rebuilds parameters for the current tensor addresses and
initializes its Stream-K barriers. Overlapping outputs use scratch, except
for an exact in-place residual without concatenation. External consumers and
unsupported layouts retain ordinary operator execution. Fusion and force-cuBLAS
disable settings are honored.

`src/core/cuda/convrot.cu` registers CUDA fusion for `Scale -> ConvRot`,
`Norm -> Mul -> ConvRot`, `RMSNorm -> Mul(weight) -> Mul(modulation) -> Add ->
Add(shift) -> ConvRot`, `SwiGLU -> ConvRot`, and `SiLU/Sigmoid -> Mul -> ConvRot`.
Both builtin and custom ConvRot outputs are accepted. SwiGLU accepts packed,
swapped, or separate gate/up inputs. Independent modulation and up-branch
producers are moved ahead of their consumers through the safe graph-delay API.
Reordering keeps an owning producer together with in-place consumers, such as
Krea2's RMSNorm weight multiplication, so the alias checks can preserve their
storage relationship.
The application kernel preserves the original reduction order, thread selection,
ConvRot row padding, and separate scale tail. Prepared plans cache operation
metadata, use the borrowed CUDA stream, and reserve scheduler workspace for
overlapping outputs. Long rows or external intermediate consumers retain
ordinary execution. These fusions do not require CUTLASS and honor
`GGML_CUDA_DISABLE_FUSION`.

The corresponding scale, normalization, and unary-activation fusion rules in
ggml now only serve HIP. CUDA GEMM/SwiGLU/ConvRot chains are selected by the
application INT8 extension before the standalone activation fusion.

CUDA kernels can opt into device replay with `ggml_ext_register_kernel_ex` and
`GGML_EXT_KERNEL_CAPTURE_SAFE`. The original registration function retains its
per-execution callback behavior, and the descriptor ABI is unchanged. Opting in
means the callback enqueues replayable device work: it has no required
per-execution host side effects or decisions based on tensor contents, and any
private streams rejoin the borrowed stream. Lazy setup must complete during the
first ordinary execution. Tensor metadata, attributes, and host plan parameters
remain frozen until scheduler reallocation; tensor contents may change normally.

When every selected extension opts in, the CUDA backend can capture the whole
scheduled plan, including its native fragments, into one CUDA Graph. The first
execution warms up kernels and library state while retaining native fragment
caches. The next stable execution captures
and launches the graph; later calls replay it. Changing a tensor address or the
execution stream invalidates the capture and repeats warmup, even when the graph
UID is unchanged. Capture failures propagate without retrying partially executed
work. An enclosing stream capture records the plan directly instead of starting
a nested capture.

`GGML_EXT_KERNEL_REPLAY_REUSABLE` additionally permits reuse across scheduler
plan rebuilds and requires `CAPTURE_SAFE`. Recorded work must depend only on
complete tensor metadata, addresses, the device and immutable registration data.
Captured resources must belong to the backend or the supplied workspace, never
per-plan host state. Workspace contents must be initialized by recorded device
work on each execution. Prepare/destroy still run for every rebuilt host plan.
The backend cache retains only the executor and workspace; it compares graph
structure, tensor metadata/addresses, selected kernels and workspace requirements
exactly, without dereferencing old graph objects. A new metadata arena can reuse
an equivalent capture. Each backend retains at most 16 retired plans and 256 MiB
of their workspaces, evicting the oldest entries first. Larger workspaces retain
plan-local capture only. Backend destruction synchronizes and releases the cache.
This allows sd.cpp's per-step scheduler rebuilds to retain device replay.

Mixed plans containing an extension without the flag retain ordinary dispatch.
Evaluation callbacks, multiple pipeline copies, and direct backend graph calls
also retain their previous behavior. `GGML_CUDA_DISABLE_GRAPHS` disables whole-plan
capture as well as native CUDA Graph caching. Native operations that cannot be
captured retain the existing fragment execution path. Whole-plan replay requires
CUDA VMM so native scratch addresses remain valid for the backend lifetime;
legacy scratch pools retain the existing fragment path. HIP and MUSA do not yet
provide this plan executor. sd.cpp's CUDA registrations opt in to reusable replay.

Remaining CUDA fusion rules and Vulkan INT8/ConvRot implementations still need
to be externalized. Vulkan execution services and kernels also remain pending.
This implementation does not yet externalize all model-specific kernels or add
dynamic dtypes.
