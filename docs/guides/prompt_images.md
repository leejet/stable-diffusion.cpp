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
then call `sd_generate_image()` or `sd_generate_video()`. The
library passes each marker's `source` string unchanged to the callback, allowing
paths, application resource IDs, or in-memory images. The library does not open
files or fetch URLs itself.

```c
bool load_prompt_image(const char* source, void* user_data, const sd_tensor_t** image_out);

sd_img_gen_opt_t* opt = sd_img_gen_opt_create();
sd_status_t status = sd_opt_set_string(opt, "prompt",
    "Draw the subject from <image:subject> in a watercolor landscape.");
sd_result_t* result = NULL;
sd_set_prompt_image_callback(ctx, load_prompt_image, application_images);
if (status == SD_STATUS_OK)
    status = sd_generate_image(ctx, opt, &result);
sd_set_prompt_image_callback(ctx, NULL, NULL);
sd_result_free(result);
sd_opt_free(opt);
```

The loader returns `true` with a borrowed U8 tensor of shape
`[channels,width,height]`, containing interleaved pixels and one to four channels.
The caller owns the tensor and keeps it valid until the next callback. The
library copies its pixels before calling the loader again. Repeated sources
within one prompt are loaded once and appear at every specified position.
Resolved image pixels participate in conditioning-cache comparisons.

Register or clear the callback while the context is idle. Different contexts
have independent callbacks.

## DGL
The complete [MOSS-Image prompt-image example](../dgl/moss_image/moss-image-sft-inline-images.dgl)
loads all components, binds the prompt image and saves the generated result.
See the [DGL example collection](../dgl/README.md) for run instructions.

Use the same markers with `encode_prompt`. DGL reads image paths relative to the
working directory.
LLM encoders accept `system` to customize the template while retaining image
inputs. `raw=true` is the complete text-tokenizer-input path and cannot be
combined with image/video bindings or `system`. See
[prompt formatting](../dgl.md#independent-generation-components).

```text
condition = encode_prompt(encoder, "Draw <image:inputs/subject.png> in watercolor.")
```

To compose images inside the graph, pass a dictionary of tensors with keys
matching the markers. Explicit bindings take precedence over opening a path.
Only referenced bindings enter conditioning; additional dictionary entries do
not add images. A referenced binding must contain a valid image tensor. If a
name is absent, the normal file reader and permission callback still apply,
including when the encoder has cached a previous result.

```text
subject = image::load("inputs/subject.png")
condition = encode_prompt(encoder, "Draw <image:subject> in watercolor.",
                          image_bindings = {subject = subject})
```

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
