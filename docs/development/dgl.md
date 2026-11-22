# DGL architecture

A DGL program describes a workflow in terms of model and tensor operations.
The CLI can run a handwritten script. The generation C API renders a script
from request parameters and binds input tensors to its entry point. Both paths
use the same compiler and executor. The C API keeps the rendered source in
`generation_info`. See the [language guide](../dgl.md) and
[C API internals](c_api.md).

## From source to execution

1. **Compile the script.** The compiler parses the source, resolves calls and
   arguments against the built-in registry or user functions, and checks their
   types. It records which values each statement needs and when operations
   use model weights. For a configuration-dependent `if`, it checks both
   branches and records resource uses from the selected branch. This is enough
   for `--check` to validate a program without loading model weights.
2. **Plan the run.** With auto-fit enabled, the runtime combines the graph's
   weight-use phases with model parameter sizes, weights already resident in
   the context, and current RAM and device budgets. It chooses where to keep
   weights and respects explicit backend and split settings. Models first
   encountered inside an iteration join the plan when their loader executes.
3. **Execute the graph.** The executor evaluates the bound calls in source
   order and releases intermediate values after their last use. As execution
   advances, the model manager uses the plan's next-use information to reclaim
   idle weight buffers under memory pressure. The value returned by `main`
   becomes the program result.

## Built-ins, operations and models

The registry describes a built-in's script name, parameters, defaults and
return type. The compiler uses that contract to check calls. At execution,
the binding converts script values to typed C++ arguments, calls an operation
or model loader, and wraps its result for the script. The same registry supplies
the [built-in function catalog](../dgl-builtins.md).

Operations implement work such as sampling, tensor slicing and media decoding.
They take concrete tensors, model objects, options and event interfaces. Models
own their runners, weight registrations and model-specific inference. Operations
can call models. Both use shared tensor, backend and runtime primitives. Script
values, syntax, registry entries and executor state stay on the language side
of the binding. Model code is independent of operations and bindings.

For example, a sampling run can use a classifier-free guidance (CFG) predictor.

```dgl
pos = predictor(model, positive)
guided = cfg(pos, predictor(model, negative), scale = 7)
latent = sample(start, sigmas, predict = guided, rng = rng)
```

The first two statements describe which model and conditions to evaluate and how
to combine the predictions. They keep the model reachable. The sampling
operation drives the solver and evaluates that predictor at each required
step. The resource plan therefore accounts for diffusion weights during
`sample`. Each sampling call owns its own prediction history and progress state.

## What the context retains

A context owns backend settings, memory budgets and the model cache. A
persistent context can reuse a loaded component across scripts when its
normalized loader settings and source dependencies still match. A run-once
context can release it after its last use. Releasing a script binding and
evicting cached weights are therefore separate lifetime decisions.

One context runs one program at a time. Each run gets a fresh placement plan
and its own progress, preview and cancellation state. Two runs of the same
source can choose different weight placement as cached resources or available
memory change. See [context execution settings](../dgl.md#context-execution-settings).
