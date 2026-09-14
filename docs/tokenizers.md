# JSON tokenizers

Use a Hugging Face `tokenizer.json` to supply the tokenizer vocabulary, merges,
added tokens, and processing stages. **PiD (including PiD 1.5) and Lens (including
Lens Turbo) require an external JSON**; their Gemma 2 and GPT-OSS tokenizers are
not embedded. Initialization fails if the main tokenizer is missing. Other
models keep their embedded tokenizer when this option is omitted.

```shell
sd-cli --diffusion-model model.gguf --llm text_encoder.gguf \
  --tokenizer tokenizer_gemma2.json --vae vae.safetensors -p "a cat"
```

Choose the JSON belonging to the text encoder checkpoint. Checking that IDs fit
the embedding table does not establish that two vocabularies have the same
meaning. The JSON file is loaded when the text encoder is created; its embedded
vocabulary is not loaded in this case.

| Model | Required text encoder tokenizer | Example |
| --- | --- | --- |
| PiD / PiD 1.5 | Gemma 2 matching the text encoder checkpoint | `--tokenizer tokenizer_gemma2.json` |
| Lens / Lens Turbo | GPT-OSS matching the text encoder checkpoint | `--tokenizer tokenizer_gpt_oss.json` |

The Gemma 3/4 tokenizer used by LTX-2 remains embedded.

| Option | Encoder |
| --- | --- |
| `--tokenizer FILE` | Main LLM/BPE encoder: Gemma 2, Gemma 3, Qwen 2/3, Mistral, GPT-OSS; also Anima and HiDream-O1 |
| `--tokenizer FILE` | Shared CLIP tokenizer in SD1/SD2/SDXL, or CLIP-L in Flux |
| `--tokenizer clip-l=FILE` | Separate CLIP-L in SD3 or Flux |
| `--tokenizer clip-g=FILE` | Separate CLIP-G in SD3 |

Use comma-separated assignments to configure multiple slots, for example
`--tokenizer main=main.json,clip-l=clip_l.json,clip-g=clip_g.json`.
A plain file path is equivalent to `main=FILE`. You may also repeat `--tokenizer`
with explicit assignments, such as `--tokenizer main=main.json --tokenizer clip-l=clip.json`.
Empty assignment paths, unknown keys, malformed assignments and
duplicate slots are rejected. Commas separate entries in the assignment form;
quote the complete argument when paths contain spaces.

SD3 overrides must name the `clip-l` or `clip-g` slot. SDXL uses one shared
tokenizer for both CLIP encoders. Do not supply both `main` and `clip-l` for Flux.
A slot targeting an absent or unsupported encoder fails initialization.
T5/SentencePiece Unigram tokenizers are outside this implementation's scope.

For example, SD3 can load the same CLIP JSON into both slots:

```shell
sd-cli --diffusion-model sd3.gguf --clip_l clip_l.safetensors \
  --clip_g clip_g.safetensors --t5xxl t5xxl.gguf --vae vae.safetensors \
  --tokenizer clip-l=tokenizer_clip.json,clip-g=tokenizer_clip.json \
  -p "a cat"
```

The C API accepts the same string in `sd_ctx_params_t::tokenizer`. A null or
empty value keeps an embedded tokenizer where available; PiD and Lens require
a nonempty main tokenizer path. The CLI passes the string through;
`TokenizerConfig` parses and validates it when text encoders are initialized.

```c
sd_ctx_params_t params;
sd_ctx_params_init(&params);
params.tokenizer = "clip-l=tokenizer_clip.json,clip-g=tokenizer_clip.json";
```

Rebuild applications against the updated public header when using the updated
library.

## Supported components

| Stage | Supported configurations |
| --- | --- |
| Normalizer | `Sequence`, `NFC`, `Lowercase`, `Replace` with String/Regex patterns |
| PreTokenizer | `Sequence`, `Split` with String/Regex patterns, all five delimiter behaviors and `invert`; `ByteLevel` with `add_prefix_space` and `use_regex` |
| Model | Deterministic `BPE`, string or array-pair merges, `unk_token`, `fuse_unk`, `byte_fallback`, `ignore_merges`, `end_of_word_suffix` |
| PostProcessor | Single-sequence `TemplateProcessing` with at most one prefix and one suffix token, `RobertaProcessing`, `ByteLevel` |
| Decoder | `Sequence`, `Replace`, `ByteLevel`, `ByteFallback`, `Fuse` |
| AddedToken | Special and ordinary added tokens, original IDs, raw or normalized matching, leftmost-longest matching |

`ByteLevel.use_regex` defaults to true when omitted. ByteLevel postprocessing
changes offsets only and adds no tokens. Added tokens with `single_word`,
`lstrip`, or `rstrip` enabled, nonzero BPE dropout, nonempty
`continuing_subword_prefix`, and unsupported component types fail loading.
New added-token IDs must follow the model vocabulary consecutively; configurations
whose IDs Hugging Face would reassign are rejected.
JSON `padding` and `truncation` must be null. This API returns IDs, not offsets,
type IDs, or paired-input encodings; the pair template is not used.

The pipeline covers the CLIP, Gemma 2, Gemma 3, GPT-OSS, Mistral 3, Qwen 2 and
Qwen 3 JSON configurations used by the differential test. It does not imply
support for every tokenizer published under those model names.

## Prompt integration

Prompt attention parsing and model-specific chat/image templates remain in the
conditioner. Raw `encode()` does not add BOS/EOS. The conditioner concatenates
weighted prompt fragments, then the existing padding/chunking step applies the
JSON single-sequence template once per sequence or CLIP chunk. Padding ID,
direction, length limits and attention masks remain text encoder policies.
CLIP requires both BOS and EOS because its chunking reserves those positions.

The internal `encode()`, `tokenize()`, and `decode()` interfaces return a success
flag and write to an output parameter. A successful result may be empty; a failed
call clears its output. JSON tokenizer input, normalization, and regex failures
return `false` with diagnostic information instead of throwing. Invalid
JSON, unsupported stages, conflicting IDs and IDs outside the encoder embedding
table fail initialization instead of falling back to the embedded tokenizer.
