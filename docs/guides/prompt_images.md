# Images in prompts

Use `<image:source>` to place an image between text fragments. Multiple images
are read in prompt order, including adjacent or repeated references:

```text
Use the subject from <image:subject.png> with the colors from <image:palette.png>.
```

Inline images are available to conditioners using Qwen2.5-VL, Qwen3-VL, or
Qwen3.5 with compatible vision weights. Supply the vision encoder with
`--llm_vision` or use a combined text/vision encoder. Each model keeps its own
prompt template, text weighting, padding, and output-layer selection. Missing
vision weights or a text-only LLM produce an error. See [MOSS-Image setup](../models/moss_image.md)
for a Qwen3.5 example.

## CLI

Pass image markers in the usual `-p` or `-n` argument. Relative image paths are
resolved against the current working directory. Spaces, colons, and parentheses
are allowed inside paths, including Windows paths:

```text
-p "Draw <image:inputs/subject.png> in the style of <image:C:\references\ink study.png>"
```

A path must be nonempty and cannot contain `>`. Empty or unclosed markers are
errors. Write `\<image:example.png>` to include a marker as literal text. A marker
is escaped when preceded by an odd number of backslashes; one backslash is
removed. Backslashes inside image paths are preserved.

Images contribute visual information at their specified positions. They do not
automatically become edit source images or determine the output dimensions.
Continue using `--ref-image` and `ref_images` for existing editing workflows.
Positive and negative prompts resolve their own image markers independently.

The context limit includes visual tokens and the model's prompt template. An
image-containing prompt that exceeds the limit fails; images are never partially
truncated.

Each prompt supports at most 32 unique image sources, checked before loading any
of them. Repeated sources share their resolved pixels. Qwen2.5-VL also limits
each resized image to 8192 vision patches (2048 visual tokens), keeping its
window mask within 256 MiB. Larger grids fail before vision encoding; use a
smaller source image if this limit is exceeded.

## C API

Register a context-local image resolver with `sd_set_prompt_image_callback()`,
then call the existing `generate_image()` or `generate_video()` function. The
library passes each marker's `source` string unchanged to the callback, allowing
paths, application resource IDs, or in-memory images. The library does not open
files or fetch URLs itself.

```c
bool load_prompt_image(const char* source, void* user_data, sd_image_t* image_out);

params.prompt = "Draw the subject from <image:subject> in a watercolor landscape.";
sd_image_generation_result_t* result = NULL;
sd_set_prompt_image_callback(ctx, load_prompt_image, application_images);
bool ok = generate_image(ctx, &params, &result);
sd_set_prompt_image_callback(ctx, NULL, NULL);
```

The loader returns `true` with an image containing interleaved 8-bit pixels and
one to four channels. The caller owns the pixels. The library copies them before
calling the loader again, so a loader may reuse its buffer. Repeated sources
within one prompt are loaded once and appear at every specified position.
Resolved image pixels participate in conditioning-cache comparisons.

Register or clear the callback while the context is idle. Different contexts
have independent callbacks. Generation parameter structures and generation
entry points are unchanged.

## Server

Native `/sdcpp/v1` image and video requests accept a `prompt_images` object
mapping IDs to base64 images or image data URIs:

```json
{
  "prompt": "Use <image:subject> with the colors from <image:palette>.",
  "prompt_images": {
    "subject": "data:image/png;base64,...",
    "palette": "data:image/png;base64,..."
  }
}
```

For OpenAI-compatible `/v1/images/generations`, `/v1/images/edits`, and
`/sdapi/v1` image requests, pass `prompt_images` only through the existing
`sd_cpp_extra_args` block embedded in `prompt`:

```json
{
  "prompt": "Use <image:subject>. <sd_cpp_extra_args>{\"prompt_images\":{\"subject\":\"data:image/png;base64,...\"}}</sd_cpp_extra_args>"
}
```

For multipart `/v1/images/edits` requests, put the same block in the `prompt`
form field. Standard `image` and `image[]` uploads continue to be edit references.
Compatibility APIs do not read separate `prompt_images` fields or file uploads.

The server resolves IDs only against the images owned by that request. An ID
never opens a server file or fetches a URL, and queued requests retain their own
image data until execution finishes.
