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

`prepare` creates optional host-side plan state and reports workspace bytes.
It can run during memory measurement, before tensor addresses are assigned, and
must not launch work or allocate device buffers. Workspace is allocated by ggml
before execution and included in scheduler memory estimates and buffer-size
reports. `execute` receives the backend, workspace buffer, and current tensor
addresses. `destroy` releases prepared state. A preparation or execution error
is propagated; execution never retries a native implementation after launch.

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
dtype, shape, layout, and hardware constraints.

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
rows are supported. Model construction uses the custom operator on CPU and
retains existing backend implementations elsewhere.

`SD_USE_UPSTREAM_GGML` excludes these registrations. The stable-diffusion public C
API is unchanged. INT8 ConvRot retains its existing packed activation layout and
separate weight scale tensor.

CuTe Flash Attention still resides in ggml, including its existing fusion paths.
Migrating it requires moving its pool/conversion dependencies onto the extension
workspace and CUDA service, then registering its selection and fusion rules in
sd.cpp. This initial implementation establishes and validates the mechanism;
it does not yet externalize all model-specific kernels or add dynamic dtypes.
