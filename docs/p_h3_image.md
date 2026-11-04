# P-H3-Image

[P-H3-Image](https://huggingface.co/PrunaAI/P-H3-Image) is Pruna AI's image model built on
[MiniMax-H3](./minimax_h3.md): MiniMax-H3's 33B transformer generating a single frame, a
Qwen3-VL-4B text encoder whose hidden states a text connector maps to the transformer's text
input, and a VAE with a single-frame image decoder. It does text-to-image and edits with one to
eight images. It is guidance-distilled: there is no CFG and no negative prompt.

## Model files

Pass the components separately, in MiniMax-H3's original layout:

- `--diffusion-model`: the MiniMax-H3 transformer
- `--llm`: the Qwen3-VL-4B text encoder with the text connector's tensors under `text_connector.`
- `--vae`: the VAE with the single-frame image decoder

The text connector is what identifies P-H3-Image: the same transformer without it loads as
MiniMax-H3.

The diffusers checkpoint in PrunaAI/P-H3-Image does not load directly.

### Download weights

- gguf: https://huggingface.co/SearchingMan/P-H3-Image-GGUF/tree/main
    - `p_h3_image-Q4_K_M.gguf`, `p_h3_image_text_encoder-Q8_0.gguf`, `p_h3_image_vae-F16.gguf`

## Text-to-image

```sh
.\bin\Release\sd-cli.exe --diffusion-model ..\models\p_h3_image-Q4_K_M.gguf --llm ..\models\p_h3_image_text_encoder-Q8_0.gguf --vae ..\models\p_h3_image_vae-F16.gguf -p "A German Expressionist oil painting of a Berlin street at night in the manner of Ernst Ludwig Kirchner, with gaunt figures hurrying past a glowing gas lamp." -W 2048 -H 2048 --diffusion-fa -v
```

The model works best at 8 megapixels (2912x2912, 3360x2496, 3872x2176); 2 and 4 megapixels
show more artifacts in faces, hands and small details. Width and height are rounded to
multiples of 32.

Sampling uses the checkpoint's 8-step sigma grid, `1.0, 0.984753, 0.967905, 0.951688, 0.925372,
0.881923, 0.810083, 0.528718, 0.0`, whatever `--steps` says; pass `--sigmas` to sample another
schedule.

## Image editing

Pass the image to edit, then up to seven more references, with `-r`. The prompt is the edit
instruction; it is encoded after `Edit <Picture 1>. ` unless it already starts with that or with
`subject_definitions:`. Refer to the other images as `<Picture 2>` and so on.

```sh
.\bin\Release\sd-cli.exe --diffusion-model ..\models\p_h3_image-Q4_K_M.gguf --llm ..\models\p_h3_image_text_encoder-Q8_0.gguf --vae ..\models\p_h3_image_vae-F16.gguf -r street.png -p "Make it a snowy winter night: heavy snow falls across the whole scene, the pavement and rooftops are covered in fresh white snow, and the figures wear snow on their hats and shoulders. Keep the painting style, composition, figures and lamp unchanged." -W 2048 -H 2048 --diffusion-fa -v
```

Each image keeps its own size, scaled into 1 to 8 megapixels; the text encoder sees it at about
one megapixel. `--ref-image-args` size settings do not apply. Edits work best at 4 or 8 megapixels and with an instruction that describes the
result, including what to keep: a bare "let it snow" on a 1 megapixel canvas comes out
oversaturated with little snow, while a full description on a 2048x2048 canvas edits cleanly.

## Notes

- The reference pipeline rewrites every prompt with a Qwen3.5-9B prompt upsampler first. sd.cpp
  renders the prompt as given, so detailed prompts work best.
- `generate_image()` runs P-H3-Image through the MiniMax-H3 video pipeline with one frame;
  `generate_video()` with `--video-frames 1` does the same.
- P-H3-Image is licensed under the MiniMax H3 Community License, which excludes the European
  Union, the United Kingdom, the Republic of Korea and the United States unless MiniMax grants a
  separate license. Its Qwen-derived parts are under Apache 2.0.
