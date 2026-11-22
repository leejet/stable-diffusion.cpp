# IP-Adapter

[IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) uses a reference image
to guide SD 1.5 or SDXL generation alongside
the text prompt. Classic and Plus adapters are supported, including combination
with Control Net.

**Currently supported only through dgl.**

## Model files

| Component | Weights and downloads | DGL usage | Requirement |
| --- | --- | --- | --- |
| Base checkpoint | [SD 1.5 or SDXL](sd.md) | `load_unet`, `load_text_encoder`, `load_vae` | Required; match adapter family |
| CLIP Vision H | `clip_vision_h.safetensors` from [IP-Adapter](https://huggingface.co/h94/IP-Adapter) | `load_vision_encoder` | Required; ViT-H/14 |
| SD 1.5 adapter | [`models/ip-adapter_sd15.safetensors`](https://huggingface.co/h94/IP-Adapter) | `ip_adapter::load` | For SD 1.5 |
| SDXL adapter | [`sdxl_models/ip-adapter_sdxl_vit-h.safetensors`](https://huggingface.co/h94/IP-Adapter) | `ip_adapter::load` | For SDXL |
| SD 1.5 Plus | [`models/ip-adapter-plus_sd15.safetensors`](https://huggingface.co/h94/IP-Adapter) | `ip_adapter::load` | Alternative for SD 1.5 |
| SDXL Plus | [`sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors`](https://huggingface.co/h94/IP-Adapter) | `ip_adapter::load` | Alternative for SDXL |

## Usage

### Web UI

Open the server's **DGL editor**, edit the same script, and submit it.
Model and input paths must satisfy the server's file-access policy; save outputs
inside its `output/` directory. See the [Web UI guide](../../examples/server/README.md#web-ui).

### DGL

| Example | Description |
| --- | --- |
| [SD 1.5 Classic](../dgl/ip_adapter/ip-adapter-sd1-5.dgl) | Use a reference image with SD 1.5. |
| [SDXL Classic](../dgl/ip_adapter/ip-adapter-sdxl.dgl) | Use a reference image with SDXL. |
| [SD 1.5 Plus](../dgl/ip_adapter/ip-adapter-plus-sd1-5.dgl) | Use unpooled CLIP Vision features. |
| [SDXL with ControlNet](../dgl/ip_adapter/ip-adapter-sdxl-controlnet.dgl) | Combine appearance and pose conditions. |

#### Run a script

Edit paths, prompts and settings in the script's `var` block, then run from
the build directory. Paths inside the script resolve from the working directory.

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe -M dgl --script ..\docs\dgl\ip_adapter\ip-adapter-sd1-5.dgl
```

**Linux/macOS**

```sh
./bin/sd-cli -M dgl --script ../docs/dgl/ip_adapter/ip-adapter-sd1-5.dgl
```

Choose another script from the table to run a different task. Add `--check`
to validate syntax and types without loading weights. See the
[shared run instructions](../dgl/README.md#run-an-example) for build paths
and the server's file-access rules.

The [SD 1.5 Classic example](../dgl/ip_adapter/ip-adapter-sd1-5.dgl) loads the
image and attention projections from one checkpoint with `ip_adapter::load`.
`ip_adapter::project` returns the projected image tokens as a tensor.
`ip_adapter::apply` binds those tokens to one text condition, so the positive
and negative CFG branches can use different tensors. Apply it again to the
returned condition to add another adapter. `strength` scales each adapter's
attention contribution. `start_sigma` and `end_sigma` bound the contribution
from high to low noise. Use `t_to_sigma` to convert a noise time to a bound.

Classic projects a pooled CLIP Vision embedding for the positive branch and a
zero embedding for the negative branch. Plus uses penultimate hidden states
from the reference image and from a zeroed, preprocessed CLIP Vision input.
Use the Plus checkpoint and these calls in the example for Plus.

```dgl
reference = image::load(input_path)
embedding = encode_vision(vision, reference, pooled = false, clip_skip = 2)
empty_embedding = encode_vision(vision, reference, pooled = false, clip_skip = 2, zero_input = true)
positive_image = ip_adapter::project(adapter, embedding)
negative_image = ip_adapter::project(adapter, empty_embedding)
```

For SDXL use an SDXL checkpoint and `family = "sdxl"` on `ip_adapter::load`.

#### Example (SDXL)

The SDXL example passes `tiling()` to `vae_decode` to reduce decode memory.
Set `params_backend="cpu"` on a loader to offload its weights.

#### Plus variants

Use Plus weights matching the base model. The Plus script requests unpooled
CLIP Vision features and encodes a zero input for its negative image condition.
Use the SDXL with ControlNet script to combine both conditions.

The startup log line `IP-Adapter: 16 image tokens` (versus `4` for the
classic adapters) confirms a Plus file was loaded.

#### Combining with Control Net

Apply `controlnet::apply` and `ip_adapter::apply` to the same condition to
preserve the reference appearance while controlling pose.

## Options

Set the reference path and component paths in `var`. `ip_adapter::apply`'s
`strength` scales the image attention contribution; 0.6–0.8 is a useful starting
range. Optional `start_sigma` / `end_sigma` restrict its sampling interval.
Classic and Plus use different CLIP Vision representations as shown above.
Use `controlnet::apply` independently for pose or edge conditioning.
