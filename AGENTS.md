# Instructions for stable-diffusion.cpp

This document is for AI coding agents working in this repository. It should
describe agent-specific workflow, repository routing, editing boundaries, and
project-specific pitfalls.

For general contribution rules, including PR scope, commit conventions, code
style, dependency updates, security hygiene, and AI-assisted contribution policy,
see `CONTRIBUTING.md`.

---

## Agent Operating Rules

Before analyzing or modifying the repository:

1. Read this file.
2. Use `rg` / `rg --files` or directory listing commands to confirm the current
   tree before relying on a path.
3. Start from `src/` and relevant `docs/` for runtime behavior.
4. Read the relevant code before editing.
5. Prefer the smallest change that fits the existing architecture.
6. Report focused verification and mention any tests not run.

Agents must not:

* Run `git push`, create PRs, or submit issue/PR comments on the user's behalf.
* Create commits unless the user explicitly requests that specific commit.
* Modify `ggml/`, `thirdparty/`, or `examples/server/frontend/` unless
  explicitly requested and necessary.
* Read large local model files or tokenizer vocabulary files.
* Rewrite unrelated code for style-only reasons.
* Add secrets, model weights, generated binaries, local absolute paths, or
  machine-specific output.

When a change is large, architectural, or likely to affect public behavior,
pause and present a short plan before editing.

---

## Repository Map and Editing Boundaries

This is a routing map for agents, not a full architecture document. The layout
can change, so verify paths before using them. Do not inspect excluded
large-data directories while checking the tree.

### Primary Project Code

Core implementation lives under `src/`.

Current source layout includes:

* `src/core/` - shared tensor, ggml integration, backend, graph, RNG, and utility
  code.
* `src/model/` - model families and model components.
* `src/model_io/` - model file loading, GGUF, safetensors, pickle, and related
  serialization helpers.
* `src/runtime/` - sampling, denoising, guidance, caching, preprocessing, and
  runtime execution helpers.
* `src/tokenizers/` - tokenizer implementations.
* `src/conditioning/` - conditioning and prompt-related implementation.
* `src/extensions/` - optional feature extensions.
* top-level `src/*.cpp` and `src/*.h` files - public implementation entry
  points, model loading, conversion, versioning, and shared managers.

`src/tokenizers/vocab/` contains large tokenizer vocabulary data. Do not read or
parse files in this directory; reference the path only when necessary.

### Public API

`include/` contains the C API exposed by the project. Currently the primary
public header is `include/stable-diffusion.h`.

Treat public headers as stable API. Avoid breaking compatibility unless the user
explicitly requests it. If public behavior changes, update relevant examples or
documentation.

### Examples

`examples/` contains programs demonstrating library usage.

* `examples/cli/` - command line program for running models, testing features,
  and debugging.
* `examples/common/` - shared example support code.
* `examples/server/` - server application built on top of the library.
* `examples/server/frontend/` - git submodule containing independent frontend
  code. Avoid modifying it unless explicitly requested.

### Documentation and Tooling

* `docs/` - documentation for supported models, build options, behavior, and
  workflows.
* `scripts/` - development, model processing, build automation, formatting, and
  tooling scripts.
* `cmake/` - CMake support modules.
* `docker/` - Docker-related project files.
* `assets/` - documentation assets; not runtime code.

### External, Local, and Generated State

* `ggml/` - git submodule for the ggml dependency.
* `thirdparty/` - vendored third-party dependencies.
* `models/` - local model storage. Ignore this directory and do not read model
  files.
* `test/` - local testing scripts. Use only when relevant to the task.
* `build/`, `build_*`, and similar directories - generated build outputs.
  Inspect them only when debugging a build result.

---

## Agent Workflow for Code Changes

1. Identify the relevant modules under `src/`.
2. Check whether the change touches the public API in `include/`.
3. Consult relevant `docs/` and examples before changing user-facing behavior.
4. Follow existing local patterns before adding new abstractions.
5. Keep edits scoped to the requested behavior.
6. Run the narrowest useful build, test, or inspection command available.

Follow `CONTRIBUTING.md` for formatting, naming, PR expectations, dependency
update policy, and security rules.

---

## Code Comments

Keep comments rare and useful.

Do not add comments that only describe what the code does. Add comments only
when the code cannot fully express the logic, the logic is unusually complex, or
there are historical reasons, invariants, constraints, compatibility concerns,
or known pitfalls that future maintainers need to understand.

Do not add task-specific comments that will be meaningless after review.

Examples from the current codebase:

```cpp
// GOOD: explains a safety constraint that is not obvious from the assignment.
// From src/model_io/pickle_io.cpp.
// Non-tensor checkpoint metadata can use REDUCE for arbitrary
// Python objects. Do not execute it; keep stack shape only.
stack.push_back(make_none_value());

// BAD: describes only what the next line does.
// Set the token count to zero.
token_count = 0;
```

---

## Text File Encoding

When reading or editing repository text files:

* Prefer UTF-8 with LF for Markdown, frontend source, JSON, and other text-first
  project files unless the file already clearly uses a different encoding.
* Do not assume terminal output encoding matches file encoding on Windows.
* A file that looks garbled in PowerShell output may still be valid UTF-8.
* When inspecting UTF-8 files in PowerShell, prefer explicit UTF-8 reads such as:
  * `[Console]::OutputEncoding = [System.Text.Encoding]::UTF8`
  * `Get-Content -Encoding utf8 <path>`
* Avoid rewriting a file purely because console output looked garbled; verify
  the actual file encoding first.

---

## Tensor and Layout Notes

Additional tensor/layout rules for this codebase:

* `sd::Tensor` shape order is not PyTorch/NumPy-style. `shape()[0]` is the
  lowest and most contiguous dimension, and higher indices are higher
  dimensions.
* Broadcasting for `sd::Tensor` must align dimensions from low to high dimension
  indices. If one tensor has fewer dimensions, append implicit `1`s at the
  higher-dimension end.
* `ggml_n_dims` / `ggml_n_dims(tensor)` can drop trailing singleton high
  dimensions. Do not assume a logical trailing dimension of `1` will still be
  counted in ggml metadata.
* Internal tensor-returning interfaces use an empty `sd::Tensor` to represent
  null, absent, or failure states. Do not add `std::optional<sd::Tensor<...>>`
  for internal APIs unless a distinct semantic state is truly required.
