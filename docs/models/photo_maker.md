# PhotoMaker

[PhotoMaker](https://github.com/TencentARC/PhotoMaker) personalizes image
generation from identity images. It requires an SDXL base model.

**Currently supported only through dgl.**

## Model files

| Component | Weights and downloads | DGL usage | Requirement |
| --- | --- | --- | --- |
| SDXL checkpoint | [SDXL guide](sd.md) | `load_unet`, `load_text_encoder` | Required; PhotoMaker is SDXL-only |
| SDXL VAE | VAE compatible with the selected SDXL checkpoint | `load_vae` | Use an override when required |
| PhotoMaker v1 | [Safetensors weights](https://huggingface.co/bssrdf/PhotoMaker) | `photomaker::load` | For v1; the original .bin release is unsupported |
| PhotoMaker v2 | [Safetensors weights](https://huggingface.co/bssrdf/PhotoMakerV2) | `photomaker::load` | Alternative to v1; requires precomputed face embeddings |
| Identity images | A directory of portraits of the same person | `image::load_dir` | Required |
| PhotoMaker v2 embeddings | `id_embeds.npy` from the preparation below | `tensor::load` | Required for v2 |

### Prepare PhotoMaker v2 embeddings

[PhotoMaker v2](https://github.com/TencentARC/PhotoMaker/blob/main/README_pmv2.md)
adds a face-embedding preparation step. From the source repository root, run
[`scripts/face_detect.py`](../../scripts/face_detect.py) on the identity image directory:

**Windows (PowerShell)**

```powershell
python scripts\face_detect.py .\inputs\photomaker_examples\scarletthead_woman
```

**Linux/macOS**

```sh
python scripts/face_detect.py ./inputs/photomaker_examples/scarletthead_woman
```

The script uses NumPy, Diffusers, and InsightFace. It writes
`id_embeds.npy` with one float32 embedding per image, in filename order.
Reuse the file with the same reference images, in the same order.
These DGL examples use `.npy` embeddings.

For DGL, read the file with `tensor::load` and pass it as
`photomaker::encode_images(id_embedding=...)` alongside the images in the same order.
See the [PhotoMaker v2 example](../dgl/photo_maker/photomaker-v2-sdxl.dgl).

## Usage

### Web UI

Open the server's **DGL editor**, edit the same script, and submit it.
Model and input paths must satisfy the server's file-access policy; save outputs
inside its `output/` directory. See the [Web UI guide](../../examples/server/README.md#web-ui).

### DGL

| Example | Description |
| --- | --- |
| [PhotoMaker v1](../dgl/photo_maker/photomaker-v1-sdxl.dgl) | Generate from reference identity images with SDXL. |
| [PhotoMaker v2](../dgl/photo_maker/photomaker-v2-sdxl.dgl) | Use reference images and prepared `.npy` identity embeddings. |

#### Run a script

Edit paths, prompts and settings in the script's `var` block, then run from
the build directory. Paths inside the script resolve from the working directory.

**Windows (PowerShell)**

```powershell
.\bin\Release\sd-cli.exe -M dgl --script ..\docs\dgl\photo_maker\photomaker-v1-sdxl.dgl
```

**Linux/macOS**

```sh
./bin/sd-cli -M dgl --script ../docs/dgl/photo_maker/photomaker-v1-sdxl.dgl
```

Choose another script from the table to run a different task. Add `--check`
to validate syntax and types without loading weights. See the
[shared run instructions](../dgl/README.md#run-an-example) for build paths
and the server's file-access rules.

Load the identity encoder with `photomaker::load` and pass the reference
images to `photomaker::encode_images`. Save the resulting tensor with `tensor::save`
to reuse the same identity across prompts or programs. Place the `img`
trigger after the class word in each prompt. `photomaker::encode_prompt`
uses the token count to allocate identity slots and returns their mask.
`photomaker::apply` merges the tokens into those slots. Use
`switch_predictor` to select the activation step and `apply_lora` to load
the checkpoint's LoRA onto the denoiser.

## Options

The prompt must contain a class word followed by the trigger word `img`, for
example `a woman img`. The scripts expose `style_strength` in the range 0–100
and pass `floor(steps * style_strength / 100)` to `switch_predictor`. Before that
step sampling uses the plain text condition; afterward it uses the merged
identity tokens. The examples use CFG 5 and 1024 × 1024.

The VAE is explicitly placed on CPU with `load_vae(..., backend="cpu")`.
Change that loader to select another compute device; identity encoding,
LoRA application and prompt encoding remain separate operations.
