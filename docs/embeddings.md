# Textual Inversion Embeddings

Textual inversion embeddings add a new trigger word (or a few vectors) to the
CLIP text encoder. They are supported for **SD 1.x, SD 2.x and SDXL**
checkpoints, which use `FrozenCLIPEmbedderWithCustomWords`. Other families
(SD3, FLUX, Qwen Image, ...) do not use this code path and ignore `--embd-dir`.

Supported weight files are `.safetensors`, `.pt`, `.ckpt` and `.gguf`.

## Directory layout

Point `--embd-dir` at the folder holding the embedding files. Only the top
level of the directory is scanned; subdirectory files are not discovered:

```sh
./bin/sd-cli -m ../models/v1-5-pruned-emaonly.safetensors \
  --embd-dir ../models/embeddings \
  -p "easynegative, girl playing the guitar" \
  --steps 20 -o output.png
```

The trigger word defaults to the file name without its extension. For
`embeddings/easynegative.safetensors` the trigger word is `easynegative`.

Run with `-v` to list the discovered embeddings at startup:

```text
embedding_dir: "/path/to/embeddings",
embeddings: {
  "easynegative": "/path/to/embeddings/easynegative.safetensors",
  ...
}
```

The same `--embd-dir` startup option applies to `sd-server`; prompts sent to
the server resolve the same way.

## Using embeddings in prompts

The CLI and the server both accept the two forms below in `prompt` and
`negative_prompt`. Prompt weights, token chunks and SDXL dual-CLIP
zero-padding still apply.

### 1. Bare trigger word via `--embd-dir`

Type the trigger word like any other word:

```text
easynegative, girl playing the guitar
```

Before generation the prompt is scanned for `--embd-dir` file stems.
Matching is case-insensitive, longest-stem-first and word-boundary aware, and
existing `<...>` tags are left untouched. A match is rewritten to an explicit
`<embedding:...>` tag with the stored file path, so the vectors replace token
embeddings before CLIP adds positional embeddings.

Normal prompt weighting applies, so `(easynegative:1.2)` boosts it and
`[easynegative]` weakens it.

### 2. Explicit `<embedding:...>` tag

Reference a file directly by path:

```text
a lovely cat, <embedding:embeddings/badhandV4.pt>
```

Rules:

- The content is a file path, not a trigger word. Paths resolve from the
  process working directory and are canonicalized before loading. The server
  also requires the path to pass its model-directory allowlist.
- `<embedding:...>` follows the same path and escaping rules as
  `<image:...>`. Use `\<embedding:...>` for literal text. A bare filename
  without `--embd-dir` is ordinary text in DGL `encode_prompt`; use the tag.
- The tag requires an SD 1.x, SD 2.x or SDXL CLIP encoder. Other encoders
  reject it, and a missing file fails generation.
- A path may not contain `>`.
- Replacing the file invalidates the cached condition without rebuilding the
  text encoder.

Weight follows the normal attention syntax around the tag:

```text
a lovely cat, (<embedding:embeddings/badhandV4.pt>:0.8)
```

### DGL

```dgl
text = load_text_encoder(model_path)
negative = encode_prompt(text, "<embedding:embeddings/badhandV4.pt>, blurry")
```

No tokenizer aliases or loader `embeddings` argument are used.

## Troubleshooting

- `cannot inspect textual inversion '...'`: the path is wrong, the file is
  unreadable, or file access (`SD_DGL_FILE_READ_MODEL` and the server
  allowlist) rejected it.
- No visible effect: make sure the checkpoint is SD 1.x, SD 2.x or SDXL.
  Other models reject embedding markers.
- Bare word does nothing: the file is not directly under `--embd-dir`, has an
  unsupported extension, collided with another stem, or was split by prompt
  weighting punctuation. Check the verbose startup listing for the exact key,
  and prefer unique stems.
