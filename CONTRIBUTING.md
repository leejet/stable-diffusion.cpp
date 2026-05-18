# Contributing

This document collects general contribution conventions for this repository.

## Before You Start

Before opening a PR, please search existing PRs to avoid duplicating ongoing work.

For large-scale refactors or changes with broad impact, please open an issue first to discuss the approach before submitting a PR.

If you want to update a third-party dependency, please open an issue first instead of submitting a direct PR. See [Dependency Updates](#dependency-updates) for details.

## Pull Requests

Keep each PR focused on one clear change. Large or overly complex PRs are harder to review and may not be merged.

Follow Conventional Commit-style subjects seen in history: `feat:`, `fix:`, `refactor:`, `ci:`, `docs:`, `chore:`. Keep subjects imperative and scoped.

PRs should include:

- What changed and why (short problem/solution summary).
- Verification evidence when applicable (commands and key outputs).
- Linked issue/PR context when applicable.
- Screenshots or sample outputs for UI/visual behavior changes.

## Code Style

Format code according to the repository style before submitting changes.

Formatting follows `.clang-format` (Chromium base, 4-space indent, no tabs). Run `format-code.sh` before opening a PR. Keep C++ standard at C++17-compatible patterns used in this repo.

Naming conventions:

- Use `PascalCase` for class/struct/type names.
- In `PascalCase` names, preserve common abbreviations in uppercase, for example `SD`, `API`, `HTTP`, `JSON`, `RGB`, `VAE`, `TAE`, `LoRA`, and `WebP`.
- Use `snake_case` for functions, methods, variables, and file names unless an existing API requires a different style.
- Use a trailing underscore for private data member names, for example `hidden_size_` or `tokenizer_`.
- Use `.h` for C and C++ header files. Do not introduce new `.hpp` headers.
- Use macro-based header include guards instead of `#pragma once`.
- Format header include guards as `__SD_{PATH}__`, where `{PATH}` is the header path in uppercase snake case without the file extension. For example, `src/sample.h` should use `__SD_SAMPLE_H__`.
- Do not introduce anonymous namespaces in new or modified code; prefer `static` file-local functions/variables or an explicit named namespace when scoping is needed.
- In `class`/`struct` definitions, place data members before member functions unless an existing type already clearly follows a different pattern.
- Keep `test_*.cpp` / `test_*.py` naming for tests.

Some older code in the project may not fully follow the current conventions. Please do not submit PRs that only rewrite existing code to match style rules.

## AI-Assisted Contributions

AI tools may be used to assist development, but contributors are responsible for the quality and correctness of the submitted code.

If any part of a contribution was generated with AI assistance, the contributor must perform a thorough human review before submitting the PR and understand every changed line.

Do not list AI tools as co-authors. The human contributor is the sole responsible author of the submitted code.

Please do not submit AI-generated code that you do not understand, and do not include meaningless experiments, temporary test code, or unrelated generated output in a PR.

## Dependency Updates

Do not submit PRs that update `ggml`. `ggml` updates are performed only after local validation by the maintainer.

Other third-party dependencies are not updated unless necessary. If you want to update a dependency, please open an issue first instead of submitting a direct PR.

## Security & Configuration

Do not commit model weights, secrets, or local absolute paths. Keep large binaries out of git unless intentionally tracked release assets.
