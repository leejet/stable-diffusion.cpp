<p align="center">
  <img src="./assets/a%20lovely%20cat.png" width="256x">
</p>

# stable-diffusion.cpp

Inference of [Stable Diffusion](https://github.com/CompVis/stable-diffusion) in pure C/C++

## Features

- Plain C/C++ implementation based on [ggml](https://github.com/ggerganov/ggml), working in the same way as [llama.cpp](https://github.com/ggerganov/llama.cpp)
- 16-bit, 32-bit float support
- 4-bit, 5-bit and 8-bit integer quantization support
- Accelerated memory-efficient CPU inference
    - Only requires ~2.3GB when using txt2img with fp16 precision to generate a 512x512 image
- AVX, AVX2 and AVX512 support for x86 architectures
- SD1.x and SD2.x support
- Original `txt2img` and `img2img` mode
- Negative prompt
- [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) style tokenizer (not all the features, only token weighting for now)
- LoRA support, same as [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#lora)
- Latent Consistency Models support(LCM/LCM-LoRA)
- Sampling method
    - `Euler A`
    - `Euler`
    - `Heun`
    - `DPM2`
    - `DPM++ 2M`
    - [`DPM++ 2M v2`](https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/8457)
    - `DPM++ 2S a`
    - [`LCM`](https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/13952)
- Cross-platform reproducibility (`--rng cuda`, consistent with the `stable-diffusion-webui GPU RNG`)
- Embedds generation parameters into png output as webui-compatible text string
- Supported platforms
    - Linux
    - Mac OS
    - Windows
    - Android (via Termux)

### TODO

- [ ] More sampling methods
- [ ] GPU support
- [ ] Make inference faster
    - The current implementation of ggml_conv_2d is slow and has high memory usage
- [ ] Continuing to reduce memory usage (quantizing the weights of ggml_conv_2d)
- [ ] k-quants support

## Usage

### Get the Code

```
git clone --recursive https://github.com/leejet/stable-diffusion.cpp
cd stable-diffusion.cpp
```

- If you have already cloned the repository, you can use the following command to update the repository to the latest code.

```
cd stable-diffusion.cpp
git pull origin master
git submodule init
git submodule update
```

### Convert weights

- download original weights(.ckpt or .safetensors). For example
    - Stable Diffusion v1.4 from https://huggingface.co/CompVis/stable-diffusion-v-1-4-original
    - Stable Diffusion v1.5 from https://huggingface.co/runwayml/stable-diffusion-v1-5
    - Stable Diffuison v2.1 from https://huggingface.co/stabilityai/stable-diffusion-2-1

    ```shell
    curl -L -O https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt
    # curl -L -O https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors
    # curl -L -O https://huggingface.co/stabilityai/stable-diffusion-2-1/blob/main/v2-1_768-nonema-pruned.safetensors
    ```

- convert weights to ggml model format

    ```shell
    cd models
    pip install -r requirements.txt
    # (optional) python convert_diffusers_to_original_stable_diffusion.py --model_path [path to diffusers weights] --checkpoint_path [path to weights]
    python convert.py [path to weights] --out_type [output precision]
    # For example, python convert.py sd-v1-4.ckpt --out_type f16
    ```

### Quantization

You can specify the output model format using the --out_type parameter

- `f16` for 16-bit floating-point
- `f32` for 32-bit floating-point
- `q8_0` for 8-bit integer quantization 
- `q5_0` or `q5_1` for 5-bit integer quantization 
- `q4_0` or `q4_1` for 4-bit integer quantization

### Build

#### Build from scratch

```shell
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

##### Using OpenBLAS

```
cmake .. -DGGML_OPENBLAS=ON
cmake --build . --config Release
```

### Run

```
usage: ./bin/sd [arguments]

arguments:
  -h, --help                         show this help message and exit
  -M, --mode [txt2img or img2img]    generation mode (default: txt2img)
  -t, --threads N                    number of threads to use during computation (default: -1).
                                     If threads <= 0, then threads will be set to the number of CPU physical cores
  -m, --model [MODEL]                path to model
  --lora-model-dir [DIR]             lora model directory
  -i, --init-img [IMAGE]             path to the input image, required by img2img
  -o, --output OUTPUT                path to write result image to (default: .\output.png)
  -p, --prompt [PROMPT]              the prompt to render
  -n, --negative-prompt PROMPT       the negative prompt (default: "")
  --cfg-scale SCALE                  unconditional guidance scale: (default: 7.0)
  --strength STRENGTH                strength for noising/unnoising (default: 0.75)
                                     1.0 corresponds to full destruction of information in init image
  -H, --height H                     image height, in pixel space (default: 512)
  -W, --width W                      image width, in pixel space (default: 512)
  --sampling-method {euler, euler_a, heun, dpm2, dpm++2s_a, dpm++2m, dpm++2mv2, lcm}
                                     sampling method (default: "euler_a")
  --steps  STEPS                     number of sample steps (default: 20)
  --rng {std_default, cuda}          RNG (default: cuda)
  -s SEED, --seed SEED               RNG seed (default: 42, use random seed for < 0)
  --schedule {discrete, karras}      Denoiser sigma schedule (default: discrete)
  -v, --verbose                      print extra info
```

#### txt2img example

```
./bin/sd -m ../models/sd-v1-4-ggml-model-f16.bin -p "a lovely cat"
```

Using formats of different precisions will yield results of varying quality.

| f32  | f16  |q8_0  |q5_0  |q5_1  |q4_0  |q4_1  |
| ----  |----  |----  |----  |----  |----  |----  |
| ![](./assets/f32.png) |![](./assets/f16.png) |![](./assets/q8_0.png) |![](./assets/q5_0.png) |![](./assets/q5_1.png) |![](./assets/q4_0.png) |![](./assets/q4_1.png) |

#### img2img example

- `./output.png` is the image generated from the above txt2img pipeline


```
./bin/sd --mode img2img -m ../models/sd-v1-4-ggml-model-f16.bin -p "cat with blue eyes" -i ./output.png -o ./img2img_output.png --strength 0.4
```

<p align="center">
  <img src="./assets/img2img_output.png" width="256x">
</p>

#### with LoRA

- convert lora weights to ggml model format

    ```shell
    cd models
    python convert.py [path to weights] --lora
    # For example, python convert.py marblesh.safetensors
    ```

- You can specify the directory where the lora weights are stored via `--lora-model-dir`. If not specified, the default is the current working directory.

- LoRA is specified via prompt, just like [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#lora).

Here's a simple example:

```
./bin/sd -m ../models/v1-5-pruned-emaonly-ggml-model-f16.bin -p "a lovely cat<lora:marblesh:1>" --lora-model-dir ../models
```

`../models/marblesh-ggml-lora.bin` will be applied to the model

#### LCM/LCM-LoRA

- Download LCM-LoRA form https://huggingface.co/latent-consistency/lcm-lora-sdv1-5
- Specify LCM-LoRA by adding `<lora:lcm-lora-sdv1-5:1>` to prompt
- It's advisable to set `--cfg-scale` to `1.0` instead of the default `7.0`. For `--steps`, a range of `2-8` steps is recommended. For `--sampling-method`, `lcm`/`euler_a` is recommended.

Here's a simple example:

```
./bin/sd -m ../models/v1-5-pruned-emaonly-ggml-model-f16.bin -p "a lovely cat<lora:lcm-lora-sdv1-5:1>" --steps 4 --lora-model-dir ../models -v --cfg-scale 1
```

| without LCM-LoRA (--cfg-scale 7)  | with LCM-LoRA (--cfg-scale 1)  |
| ----  |----    |
| ![](./assets/without_lcm.png) |![](./assets/with_lcm.png)  |


### Docker

#### Building using Docker

```shell
docker build -t sd .
```

#### Run

```shell
docker run -v /path/to/models:/models -v /path/to/output/:/output sd [args...]
# For example
# docker run -v ./models:/models -v ./build:/output sd -m /models/sd-v1-4-ggml-model-f16.bin -p "a lovely cat" -v -o /output/output.png
```

## Memory/Disk Requirements

| precision | f32  | f16  |q8_0  |q5_0  |q5_1  |q4_0  |q4_1  |
| ----         | ----  |----  |----  |----  |----  |----  |----  |
|  **Disk**        | 2.7G | 2.0G | 1.7G | 1.6G | 1.6G | 1.5G | 1.5G |
|  **Memory**(txt2img - 512 x 512) | ~2.8G | ~2.3G | ~2.1G | ~2.0G | ~2.0G | ~2.0G | ~2.0G |

## Contributors

Thank you to all the people who have already contributed to stable-diffusion.cpp!

[![Contributors](https://contrib.rocks/image?repo=leejet/stable-diffusion.cpp)](https://github.com/leejet/stable-diffusion.cpp/graphs/contributors)

## References

- [ggml](https://github.com/ggerganov/ggml)
- [stable-diffusion](https://github.com/CompVis/stable-diffusion)
- [stable-diffusion-stability-ai](https://github.com/Stability-AI/stablediffusion)
- [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
- [k-diffusion](https://github.com/crowsonkb/k-diffusion)
- [latent-consistency-model](https://github.com/luosiallen/latent-consistency-model)
