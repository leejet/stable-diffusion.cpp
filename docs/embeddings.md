# Textual Inversion Embeddings

Textual inversion embeddings add a new trigger word (or a few vectors) to the
CLIP text encoder. They are supported for **SD 1.x, SD 2.x and SDXL**
checkpoints, which use `FrozenCLIPEmbedderWithCustomWords`. Other families
(SD3, FLUX, Qwen Image, ...) do not use this code path and ignore `--embd-dir`.

Supported weight files are `.safetensors`, `.pt`, `.ckpt` and `.gguf`.

## Directory layout

Point `--embd-dir` at the folder holding the embedding files. Subdirectories
are scanned recursively:

```sh
./bin/sd-cli -m ../models/v1-5-pruned-emaonly.safetensors \
  --embd-dir ../models/embeddings \
  -p "ehartnoveau, girl playing the guitar" \
  --steps 20 -o output.png
```

The trigger word defaults to the file name without its extension. For
`embeddings/ehartnoveau.safetensors` the trigger word is `ehartnoveau`.
File names are matched case-insensitively, and a relative path without its
extension (for example `style/ehartnoveau` for
`embeddings/style/ehartnoveau.safetensors`) also works as a trigger word.

When two files share the same stem, the first file found keeps the bare stem
and every file keeps its own relative path. Prefer unique stems or use the
relative path form to disambiguate.

Run with `-v` to list the discovered embeddings at startup:

```text
embedding_dir: "/path/to/embeddings",
embeddings: {
  "ehartnoveau": "/path/to/embeddings/ehartnoveau.safetensors",
  ...
}
```

## Using embeddings in prompts

### 1. Bare trigger word

Type the trigger word like any other word:

```text
ehartnoveau, girl playing the guitar
```

Normal prompt weighting applies, so `(ehartnoveau:1.2)` boosts it and
`[ehartnoveau]` weakens it.

### 2. Explicit `<embedding:...>` tag

When the file name is hard to type, contains spaces, lives in a subdirectory,
or collides with an ordinary word, reference it explicitly. This mirrors the
`<lora:...>` syntax:

```text
a lovely cat, <embedding:ehartnoveau:0.8>
a lovely cat, <embedding:style/ehartnoveau:0.8>
a lovely cat, <emb:my embedding>
```

Rules:

- The tag prefix is `embedding` or the short alias `emb`, case-insensitive:
  `<embedding:...>`, `<Embedding:...>`, `<emb:...>` all work.
- The name may be a file stem (`my_emb`), a relative path with forward or
  backward slashes (`sub/dir/my_emb`), and may include the file extension
  (`my_emb.safetensors`), which is ignored for the lookup.
- The weight is optional. `<embedding:my_emb>` means weight `1.0`.
  With a weight, the tag expands to `(trigger:weight)` before the usual
  `(text:weight)` parser runs, for example
  `<embedding:ehartnoveau:0.8>` becomes `(ehartnoveau:0.8)`.
- An unknown name logs a warning and falls back to the stripped name as plain
  text, so a typo does not leak `<embedding:...>` tokens into the tokenizer.

## Troubleshooting

- `embedding '...' not found`: the file is not under `--embd-dir`, has an
  unsupported extension, or the name was mistyped. Check the verbose startup
  listing for the exact key.
- No visible effect: make sure the checkpoint is SD 1.x, SD 2.x or SDXL.
  Other models ignore textual inversion files.
- Weight has no effect and the log mentions `contains attention syntax`:
  the trigger word itself contains `()[]:\`. Rename the file to a plain
  stem and use the tag form.
