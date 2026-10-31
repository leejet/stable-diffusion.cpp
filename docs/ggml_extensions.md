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
their storage belongs to the fused subgraph. The final output must own storage,
except for a terminal `CPY`, which retains its original destination and write
semantics. The extension must handle input/output storage overlap, including
allocator reuse, with scratch space when necessary. Other in-place writes are
not eligible for extension fusion.

An optional `optimize` callback runs before matching and allocation. Use
`ggml_ext_graph_delay` to move a producer range after independent nodes. The
helper rejects dependencies, unknown side effects, and writes that alias moved
inputs or outputs. It can cross independent copies, including dtype conversions.
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

The combined INT8 GEMM epilogue feeding FP16 Q directly into attention still needs
to be externalized. The current extension can fuse RMSNorm/RoPE with attention,
while other INT8 GEMM and standalone projection fusions retain their native
implementations. Native INT8 projection matchers do not recognize custom RoPE
yet, so those graphs currently run GEMM followed by the external RoPE/attention
path. Restoring their combined epilogues is part of the remaining migration.
Vulkan execution services and kernels also remain pending.
This implementation does not yet externalize all model-specific kernels or add
dynamic dtypes.
