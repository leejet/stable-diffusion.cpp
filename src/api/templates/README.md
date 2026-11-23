# C API generation templates

The generation C API selects an entry template for the request, renders a
complete DGL program and runs it with the input tensors supplied by the host.
`generation_info` contains the executed source. CMake embeds the templates
during configuration.

| Template | Flow |
| --- | --- |
| [image.dgl](image.dgl) | Text-to-image and ordinary image-to-image |
| [image_inpaint.dgl](image_inpaint.dgl) | Masked sampling and native inpaint conditioning |
| [image_edit.dgl](image_edit.dgl) | Native reference-image editing |
| [image_pixel.dgl](image_pixel.dgl) | Pixel-space diffusion, with optional VAE-encoded references |
| `image_h3.dgl` | MiniMax-H3 Image and P-H3 latent packing |
| [image_layered.dgl](image_layered.dgl) | Qwen layered generation and per-layer decoding |
| [image_sefi.dgl](image_sefi.dgl) | SeFi dual schedules |
| [video_wan.dgl](video_wan.dgl) | Wan/Hunyuan video, VACE, S2V and dual experts |
| [video_ltx.dgl](video_ltx.dgl) | LTX video/audio generation and keyframes |
| `video_h3.dgl` | MiniMax-H3 video/audio generation and references |
| [upscale.dgl](upscale.dgl) | Standalone image upscaling, including repeats |

## Editing a flow

C++ resolves model families, request defaults, dimensions and host tensor
bindings. The entry template expresses the resulting workflow as DGL calls.
Use a template condition for an optional stage and an include for a helper
shared by several flows. Put the minimum DGL version at the top of each entry.
When a helper needs a newer version, update the entries that use it.
See the [DGL version rules](../../../docs/dgl.md#dgl-versions).

For example, the image entry loads the components with these lines.

```text
    base_model = {{model_loader}}(model_path{{model_options}})
    base_text = load_text_encoder(text_path{{text_options}})
    vae = {{vae_loader}}(vae_path{{vae_options}})
```

Here `model_path` and `text_path` are DGL variables in the rendered program.
C++ supplies `model_loader` as a function name and `model_options` as an
already formatted argument list, including its leading comma when nonempty.

The value C++ supplies determines how to insert it into the script.

| Template expression | Input | Rendered DGL |
| --- | --- | --- |
| `{{ value \| literal }}` | A raw value, such as a prompt or file path | A DGL literal with strings quoted and escaped |
| `{{ options \| args }}` | An object such as `{"family": "sd1"}` | `, family = "sd1"` (empty text for an empty object) |
| `{{model_options}}` | A trusted, preformatted argument list | The supplied text verbatim |

Conditions use boolean fields, for example `{% if flags.has_init %}`, with
optional `elif`, `else` and `not`. Close them with `{% endif %}`. An include
such as `{% include "image_helpers" %}` uses the embedded template name.
Substitutions are evaluated once, so template-like text in a prompt stays text.
