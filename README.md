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
- Original `txt2img` and `img2img` mode
- Negative prompt
- [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) style tokenizer (not all the features, only token weighting for now)
- Sampling method
    - `Euler A`
- Supported platforms
    - Linux
    - Mac OS
    - Windows

### TODO

- [ ] More sampling methods
- [ ] GPU support
- [ ] Make inference faster
    - The current implementation of ggml_conv_2d is slow and has high memory usage
- [ ] Continuing to reduce memory usage (quantizing the weights of ggml_conv_2d)
- [ ] LoRA support
- [ ] k-quants support
- [ ] Cross-platform reproducibility (perhaps ensuring consistency with the original SD)
- [ ] Adapting to more weight formats

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

    ```shell
    curl -L -O https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt
    # curl -L -O https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors
    ```

- convert weights to ggml model format

    ```shell
    cd models
    pip install -r requirements.txt
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

```shell
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

#### Using OpenBLAS

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
  -i, --init-img [IMAGE]             path to the input image, required by img2img
  -o, --output OUTPUT                path to write result image to (default: .\output.png)
  -p, --prompt [PROMPT]              the prompt to render
  -n, --negative-prompt PROMPT       the negative prompt (default: "")
  --cfg-scale SCALE                  unconditional guidance scale: (default: 7.0)
  --strength STRENGTH                strength for noising/unnoising (default: 0.75)
                                     1.0 corresponds to full destruction of information in init image
  -H, --height H                     image height, in pixel space (default: 512)
  -W, --width W                      image width, in pixel space (default: 512)
  --sample-method SAMPLE_METHOD      sample method (default: "eular a")
  --steps  STEPS                     number of sample steps (default: 20)
  -s SEED, --seed SEED               RNG seed (default: 42, use random seed for < 0)
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

## Memory/Disk Requirements

| precision | f32  | f16  |q8_0  |q5_0  |q5_1  |q4_0  |q4_1  |
| ----         | ----  |----  |----  |----  |----  |----  |----  |
|  **Disk**        | 2.7G | 2.0G | 1.7G | 1.6G | 1.6G | 1.5G | 1.5G |
|  **Memory**(txt2img - 512 x 512) | ~2.8G | ~2.3G | ~2.1G | ~2.0G | ~2.0G | ~2.0G | ~2.0G |


## References

- [ggml](https://github.com/ggerganov/ggml)
- [stable-diffusion](https://github.com/CompVis/stable-diffusion)
- [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
- [k-diffusion](https://github.com/crowsonkb/k-diffusion)
