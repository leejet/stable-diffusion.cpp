<p align="center">
  <img src="./assets/cat_with_sd_cpp_42.png" width="360x">
</p>

# stable-diffusion.cpp

Diffusion model(SD,Flux,Wan,...) inference in pure C/C++

***Note that this project is under active development. \
API and command-line option may change frequently.***

## Features

- Plain C/C++ implementation based on [ggml](https://github.com/ggerganov/ggml), working in the same way as [llama.cpp](https://github.com/ggerganov/llama.cpp)
- Super lightweight and without external dependencies
- Supported models
  - Image Models
    - SD1.x, SD2.x, [SD-Turbo](https://huggingface.co/stabilityai/sd-turbo)
    - SDXL, [SDXL-Turbo](https://huggingface.co/stabilityai/sdxl-turbo)
      - !!!The VAE in SDXL encounters NaN issues under FP16, but unfortunately, the ggml_conv_2d only operates under FP16. Hence, a parameter is needed to specify the VAE that has fixed the FP16 NaN issue. You can find it here: [SDXL VAE FP16 Fix](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/blob/main/sdxl_vae.safetensors).
    - [SD3/SD3.5](./docs/sd3.md)
    - [Flux-dev/Flux-schnell](./docs/flux.md)
    - [Chroma](./docs/chroma.md)
  - Image Edit Models
    - [FLUX.1-Kontext-dev](./docs/kontext.md)
  - Video Models
    - [Wan2.1/Wan2.2](./docs/wan.md)
  - [PhotoMaker](https://github.com/TencentARC/PhotoMaker) support.
  - Control Net support with SD 1.5
  - LoRA support, same as [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#lora)
  - Latent Consistency Models support (LCM/LCM-LoRA)
  - Faster and memory efficient latent decoding with [TAESD](https://github.com/madebyollin/taesd)
  - Upscale images generated with [ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- 16-bit, 32-bit float support
- 2-bit, 3-bit, 4-bit, 5-bit and 8-bit integer quantization support
- Accelerated memory-efficient CPU inference
    - Only requires ~2.3GB when using txt2img with fp16 precision to generate a 512x512 image, enabling Flash Attention just requires ~1.8GB.
- AVX, AVX2 and AVX512 support for x86 architectures
- Full CUDA, Metal, Vulkan, OpenCL and SYCL backend for GPU acceleration.
- Can load ckpt, safetensors and diffusers models/checkpoints. Standalone VAEs models
    - No need to convert to `.ggml` or `.gguf` anymore!
- Flash Attention for memory usage optimization
- Negative prompt
- [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) style tokenizer (not all the features, only token weighting for now)
- VAE tiling processing for reduce memory usage
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
    - Android (via Termux, [Local Diffusion](https://github.com/rmatif/Local-Diffusion))

## Usage

For most users, you can download the built executable program from the latest [release](https://github.com/leejet/stable-diffusion.cpp/releases/latest).
If the built product does not meet your requirements, you can choose to build it manually.

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

### Download weights

- download original weights(.ckpt or .safetensors). For example
    - Stable Diffusion v1.4 from https://huggingface.co/CompVis/stable-diffusion-v-1-4-original
    - Stable Diffusion v1.5 from https://huggingface.co/runwayml/stable-diffusion-v1-5
    - Stable Diffuison v2.1 from https://huggingface.co/stabilityai/stable-diffusion-2-1
    - Stable Diffusion 3 2B from https://huggingface.co/stabilityai/stable-diffusion-3-medium

    ```shell
    curl -L -O https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt
    # curl -L -O https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors
    # curl -L -O https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-nonema-pruned.safetensors
    # curl -L -O https://huggingface.co/stabilityai/stable-diffusion-3-medium/resolve/main/sd3_medium_incl_clips_t5xxlfp16.safetensors
    ```

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

##### Using CUDA

This provides BLAS acceleration using the CUDA cores of your Nvidia GPU. Make sure to have the CUDA toolkit installed. You can download it from your Linux distro's package manager (e.g. `apt install nvidia-cuda-toolkit`) or from here: [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads). Recommended to have at least 4 GB of VRAM.

```
cmake .. -DSD_CUDA=ON
cmake --build . --config Release
```

##### Using HipBLAS
This provides BLAS acceleration using the ROCm cores of your AMD GPU. Make sure to have the ROCm toolkit installed.
To build for another GPU architecture than installed in your system, set `$GFX_NAME` manually to the desired architecture (replace first command). This is also necessary if your GPU is not officially supported by ROCm, for example you have to set `$GFX_NAME` manually to `gfx1030` for consumer RDNA2 cards.

Windows User Refer to [docs/hipBLAS_on_Windows.md](docs%2FhipBLAS_on_Windows.md) for a comprehensive guide.

```
if command -v rocminfo; then export GFX_NAME=$(rocminfo | awk '/ *Name: +gfx[1-9]/ {print $2; exit}'); else echo "rocminfo missing!"; fi
if [ -z "${GFX_NAME}" ]; then echo "Error: Couldn't detect GPU!"; else echo "Building for GPU: ${GFX_NAME}"; fi
cmake .. -G "Ninja" -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DSD_HIPBLAS=ON -DCMAKE_BUILD_TYPE=Release -DGPU_TARGETS=$GFX_NAME -DAMDGPU_TARGETS=$GFX_NAME -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON
cmake --build . --config Release
```

##### Using MUSA

This provides BLAS acceleration using the MUSA cores of your Moore Threads GPU. Make sure to have the MUSA toolkit installed.

```bash
cmake .. -DCMAKE_C_COMPILER=/usr/local/musa/bin/clang -DCMAKE_CXX_COMPILER=/usr/local/musa/bin/clang++ -DSD_MUSA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

##### Using Metal

Using Metal makes the computation run on the GPU. Currently, there are some issues with Metal when performing operations on very large matrices, making it highly inefficient at the moment. Performance improvements are expected in the near future.

```
cmake .. -DSD_METAL=ON
cmake --build . --config Release
```

##### Using Vulkan

Install Vulkan SDK from https://www.lunarg.com/vulkan-sdk/.

```
cmake .. -DSD_VULKAN=ON
cmake --build . --config Release
```

##### Using OpenCL (for Adreno GPU)

Currently, it supports only Adreno GPUs and is primarily optimized for Q4_0 type

To build for Windows ARM please refers to [Windows 11 Arm64
](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/OPENCL.md#windows-11-arm64)

Building for Android:

  Android NDK:
       Download and install the Android NDK from the [official Android developer site](https://developer.android.com/ndk/downloads).

Setup OpenCL Dependencies for NDK:

You need to provide OpenCL headers and the ICD loader library to your NDK sysroot.

*   OpenCL Headers:
    ```bash
    # In a temporary working directory
    git clone https://github.com/KhronosGroup/OpenCL-Headers
    cd OpenCL-Headers
    # Replace <YOUR_NDK_PATH> with your actual NDK installation path
    # e.g., cp -r CL /path/to/android-ndk-r26c/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include
    sudo cp -r CL <YOUR_NDK_PATH>/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include
    cd ..
    ```

*   OpenCL ICD Loader:
    ```bash
    # In the same temporary working directory
    git clone https://github.com/KhronosGroup/OpenCL-ICD-Loader
    cd OpenCL-ICD-Loader
    mkdir build_ndk && cd build_ndk

    # Replace <YOUR_NDK_PATH> in the CMAKE_TOOLCHAIN_FILE and OPENCL_ICD_LOADER_HEADERS_DIR
    cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_TOOLCHAIN_FILE=<YOUR_NDK_PATH>/build/cmake/android.toolchain.cmake \
      -DOPENCL_ICD_LOADER_HEADERS_DIR=<YOUR_NDK_PATH>/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include \
      -DANDROID_ABI=arm64-v8a \
      -DANDROID_PLATFORM=24 \
      -DANDROID_STL=c++_shared

    ninja
    # Replace <YOUR_NDK_PATH>
    # e.g., cp libOpenCL.so /path/to/android-ndk-r26c/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android
    sudo cp libOpenCL.so <YOUR_NDK_PATH>/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android
    cd ../..
    ```

Build `stable-diffusion.cpp` for Android with OpenCL:

```bash
mkdir build-android && cd build-android

# Replace <YOUR_NDK_PATH> with your actual NDK installation path
# e.g., -DCMAKE_TOOLCHAIN_FILE=/path/to/android-ndk-r26c/build/cmake/android.toolchain.cmake
cmake .. -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=<YOUR_NDK_PATH>/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DGGML_OPENMP=OFF \
  -DSD_OPENCL=ON

ninja
```
*(Note: Don't forget to include `LD_LIBRARY_PATH=/vendor/lib64` in your command line before running the binary)*

##### Using SYCL

Using SYCL makes the computation run on the Intel GPU. Please make sure you have installed the related driver and [Intel® oneAPI Base toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html) before start. More details and steps can refer to [llama.cpp SYCL backend](https://github.com/ggerganov/llama.cpp/blob/master/docs/backend/SYCL.md#linux).

```
# Export relevant ENV variables
source /opt/intel/oneapi/setvars.sh

# Option 1: Use FP32 (recommended for better performance in most cases)
cmake .. -DSD_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx

# Option 2: Use FP16
cmake .. -DSD_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_SYCL_F16=ON

cmake --build . --config Release
```

Example of text2img by using SYCL backend:

- download `stable-diffusion` model weight, refer to [download-weight](#download-weights).

- run `./bin/sd -m ../models/sd3_medium_incl_clips_t5xxlfp16.safetensors --cfg-scale 5 --steps 30 --sampling-method euler  -H 1024 -W 1024 --seed 42 -p "fantasy medieval village world inside a glass sphere , high detail, fantasy, realistic, light effect, hyper detail, volumetric lighting, cinematic, macro, depth of field, blur, red light and clouds from the back, highly detailed epic cinematic concept art cg render made in maya, blender and photoshop, octane render, excellent composition, dynamic dramatic cinematic lighting, aesthetic, very inspirational, world inside a glass sphere by james gurney by artgerm with james jean, joe fenton and tristan eaton by ross tran, fine details, 4k resolution"`

<p align="center">
  <img src="./assets/sycl_sd3_output.png" width="360x">
</p>



##### Using Flash Attention

Enabling flash attention for the diffusion model reduces memory usage by varying amounts of MB.
eg.:
 - flux 768x768 ~600mb
 - SD2 768x768 ~1400mb

For most backends, it slows things down, but for cuda it generally speeds it up too.
At the moment, it is only supported for some models and some backends (like cpu, cuda/rocm, metal).

Run by adding `--diffusion-fa` to the arguments and watch for:
```
[INFO ] stable-diffusion.cpp:312  - Using flash attention in the diffusion model
```
and the compute buffer shrink in the debug log:
```
[DEBUG] ggml_extend.hpp:1004 - flux compute buffer size: 650.00 MB(VRAM)
```

### Run

```
usage: ./bin/sd [arguments]

arguments:
  -h, --help                         show this help message and exit
  -M, --mode [MODE]                  run mode, one of: [img_gen, vid_gen, convert], default: img_gen
  -t, --threads N                    number of threads to use during computation (default: -1)
                                     If threads <= 0, then threads will be set to the number of CPU physical cores
  --offload-to-cpu                   place the weights in RAM to save VRAM, and automatically load them into VRAM when needed
  -m, --model [MODEL]                path to full model
  --diffusion-model                  path to the standalone diffusion model
  --high-noise-diffusion-model       path to the standalone high noise diffusion model
  --clip_l                           path to the clip-l text encoder
  --clip_g                           path to the clip-g text encoder
  --clip_vision                      path to the clip-vision encoder
  --t5xxl                            path to the t5xxl text encoder
  --vae [VAE]                        path to vae
  --taesd [TAESD_PATH]               path to taesd. Using Tiny AutoEncoder for fast decoding (low quality)
  --control-net [CONTROL_PATH]       path to control net model
  --embd-dir [EMBEDDING_PATH]        path to embeddings
  --upscale-model [ESRGAN_PATH]      path to esrgan model. Upscale images after generate, just RealESRGAN_x4plus_anime_6B supported by now
  --upscale-repeats                  Run the ESRGAN upscaler this many times (default 1)
  --type [TYPE]                      weight type (examples: f32, f16, q4_0, q4_1, q5_0, q5_1, q8_0, q2_K, q3_K, q4_K)
                                     If not specified, the default is the type of the weight file
  --tensor-type-rules [EXPRESSION]   weight type per tensor pattern (example: "^vae\.=f16,model\.=q8_0")
  --lora-model-dir [DIR]             lora model directory
  -i, --init-img [IMAGE]             path to the init image, required by img2img
  --mask [MASK]                      path to the mask image, required by img2img with mask
  -i, --end-img [IMAGE]              path to the end image, required by flf2v
  --control-image [IMAGE]            path to image condition, control net
  -r, --ref-image [PATH]             reference image for Flux Kontext models (can be used multiple times)
  --control-video [PATH]             path to control video frames, It must be a directory path.
                                     The video frames inside should be stored as images in lexicographical (character) order
                                     For example, if the control video path is `frames`, the directory contain images such as 00.png, 01.png, 鈥?etc.
  --increase-ref-index               automatically increase the indices of references images based on the order they are listed (starting with 1).
  -o, --output OUTPUT                path to write result image to (default: ./output.png)
  -p, --prompt [PROMPT]              the prompt to render
  -n, --negative-prompt PROMPT       the negative prompt (default: "")
  --cfg-scale SCALE                  unconditional guidance scale: (default: 7.0)
  --img-cfg-scale SCALE              image guidance scale for inpaint or instruct-pix2pix models: (default: same as --cfg-scale)
  --guidance SCALE                   distilled guidance scale for models with guidance input (default: 3.5)
  --slg-scale SCALE                  skip layer guidance (SLG) scale, only for DiT models: (default: 0)
                                     0 means disabled, a value of 2.5 is nice for sd3.5 medium
  --eta SCALE                        eta in DDIM, only for DDIM and TCD: (default: 0)
  --skip-layers LAYERS               Layers to skip for SLG steps: (default: [7,8,9])
  --skip-layer-start START           SLG enabling point: (default: 0.01)
  --skip-layer-end END               SLG disabling point: (default: 0.2)
  --scheduler {discrete, karras, exponential, ays, gits, smoothstep, sgm_uniform, simple} Denoiser sigma scheduler (default: discrete)
  --sampling-method {euler, euler_a, heun, dpm2, dpm++2s_a, dpm++2m, dpm++2mv2, ipndm, ipndm_v, lcm, ddim_trailing, tcd}
                                     sampling method (default: "euler" for Flux/SD3/Wan, "euler_a" otherwise)
  --timestep-shift N                 shift timestep for NitroFusion models, default: 0, recommended N for NitroSD-Realism around 250 and 500 for NitroSD-Vibrant
  --steps  STEPS                     number of sample steps (default: 20)
  --high-noise-cfg-scale SCALE       (high noise) unconditional guidance scale: (default: 7.0)
  --high-noise-img-cfg-scale SCALE   (high noise) image guidance scale for inpaint or instruct-pix2pix models: (default: same as --cfg-scale)
  --high-noise-guidance SCALE        (high noise) distilled guidance scale for models with guidance input (default: 3.5)
  --high-noise-slg-scale SCALE       (high noise) skip layer guidance (SLG) scale, only for DiT models: (default: 0)
                                     0 means disabled, a value of 2.5 is nice for sd3.5 medium
  --high-noise-eta SCALE             (high noise) eta in DDIM, only for DDIM and TCD: (default: 0)
  --high-noise-skip-layers LAYERS    (high noise) Layers to skip for SLG steps: (default: [7,8,9])
  --high-noise-skip-layer-start      (high noise) SLG enabling point: (default: 0.01)
  --high-noise-skip-layer-end END    (high noise) SLG disabling point: (default: 0.2)
  --high-noise-scheduler {discrete, karras, exponential, ays, gits, smoothstep, sgm_uniform, simple} Denoiser sigma scheduler (default: discrete)
  --high-noise-sampling-method {euler, euler_a, heun, dpm2, dpm++2s_a, dpm++2m, dpm++2mv2, ipndm, ipndm_v, lcm, ddim_trailing, tcd}
                                     (high noise) sampling method (default: "euler_a")
  --high-noise-steps  STEPS          (high noise) number of sample steps (default: -1 = auto)
                                     SLG will be enabled at step int([STEPS]*[START]) and disabled at int([STEPS]*[END])
  --strength STRENGTH                strength for noising/unnoising (default: 0.75)
  --control-strength STRENGTH        strength to apply Control Net (default: 0.9)
                                     1.0 corresponds to full destruction of information in init image
  -H, --height H                     image height, in pixel space (default: 512)
  -W, --width W                      image width, in pixel space (default: 512)
  --rng {std_default, cuda}          RNG (default: cuda)
  -s SEED, --seed SEED               RNG seed (default: 42, use random seed for < 0)
  -b, --batch-count COUNT            number of images to generate
  --clip-skip N                      ignore last layers of CLIP network; 1 ignores none, 2 ignores one layer (default: -1)
                                     <= 0 represents unspecified, will be 1 for SD1.x, 2 for SD2.x
  --vae-tiling                       process vae in tiles to reduce memory usage
  --vae-tile-size [X]x[Y]            tile size for vae tiling (default: 32x32)
  --vae-relative-tile-size [X]x[Y]   relative tile size for vae tiling, in fraction of image size if < 1, in number of tiles per dim if >=1 (overrides --vae-tile-size)
  --vae-tile-overlap OVERLAP         tile overlap for vae tiling, in fraction of tile size (default: 0.5)
  --vae-on-cpu                       keep vae in cpu (for low vram)
  --clip-on-cpu                      keep clip in cpu (for low vram)
  --diffusion-fa                     use flash attention in the diffusion model (for low vram)
                                     Might lower quality, since it implies converting k and v to f16.
                                     This might crash if it is not supported by the backend.
  --diffusion-conv-direct            use Conv2d direct in the diffusion model
                                     This might crash if it is not supported by the backend.
  --vae-conv-direct                  use Conv2d direct in the vae model (should improve the performance)
                                     This might crash if it is not supported by the backend.
  --control-net-cpu                  keep controlnet in cpu (for low vram)
  --canny                            apply canny preprocessor (edge detection)
  --color                            colors the logging tags according to level
  --chroma-disable-dit-mask          disable dit mask for chroma
  --chroma-enable-t5-mask            enable t5 mask for chroma
  --chroma-t5-mask-pad  PAD_SIZE     t5 mask pad size of chroma
  --video-frames                     video frames (default: 1)
  --fps                              fps (default: 24)
  --moe-boundary BOUNDARY            timestep boundary for Wan2.2 MoE model. (default: 0.875)
                                     only enabled if `--high-noise-steps` is set to -1
  --flow-shift SHIFT                 shift value for Flow models like SD3.x or WAN (default: auto)
  --vace-strength                    wan vace strength
  --photo-maker                      path to PHOTOMAKER model
  --pm-id-images-dir [DIR]           path to PHOTOMAKER input id images dir
  --pm-id-embed-path [PATH]          path to PHOTOMAKER v2 id embed
  --pm-style-strength                strength for keeping PHOTOMAKER input identity (default: 20)
  -v, --verbose                      print extra info
```

#### txt2img example

```sh
./bin/sd -m ../models/sd-v1-4.ckpt -p "a lovely cat"
# ./bin/sd -m ../models/v1-5-pruned-emaonly.safetensors -p "a lovely cat"
# ./bin/sd -m ../models/sd_xl_base_1.0.safetensors --vae ../models/sdxl_vae-fp16-fix.safetensors -H 1024 -W 1024 -p "a lovely cat" -v
# ./bin/sd -m ../models/sd3_medium_incl_clips_t5xxlfp16.safetensors -H 1024 -W 1024 -p 'a lovely cat holding a sign says \"Stable Diffusion CPP\"' --cfg-scale 4.5 --sampling-method euler -v --clip-on-cpu
# ./bin/sd --diffusion-model  ../models/flux1-dev-q3_k.gguf --vae ../models/ae.sft --clip_l ../models/clip_l.safetensors --t5xxl ../models/t5xxl_fp16.safetensors  -p "a lovely cat holding a sign says 'flux.cpp'" --cfg-scale 1.0 --sampling-method euler -v --clip-on-cpu
# ./bin/sd -m  ..\models\sd3.5_large.safetensors --clip_l ..\models\clip_l.safetensors --clip_g ..\models\clip_g.safetensors --t5xxl ..\models\t5xxl_fp16.safetensors  -H 1024 -W 1024 -p 'a lovely cat holding a sign says \"Stable diffusion 3.5 Large\"' --cfg-scale 4.5 --sampling-method euler -v --clip-on-cpu
```

Using formats of different precisions will yield results of varying quality.

| f32  | f16  |q8_0  |q5_0  |q5_1  |q4_0  |q4_1  |
| ----  |----  |----  |----  |----  |----  |----  |
| ![](./assets/f32.png) |![](./assets/f16.png) |![](./assets/q8_0.png) |![](./assets/q5_0.png) |![](./assets/q5_1.png) |![](./assets/q4_0.png) |![](./assets/q4_1.png) |

#### img2img example

- `./output.png` is the image generated from the above txt2img pipeline


```
./bin/sd -m ../models/sd-v1-4.ckpt -p "cat with blue eyes" -i ./output.png -o ./img2img_output.png --strength 0.4
```

<p align="center">
  <img src="./assets/img2img_output.png" width="256x">
</p>

## More Guides

- [LoRA](./docs/lora.md)
- [LCM/LCM-LoRA](./docs/lcm.md)
- [Using PhotoMaker to personalize image generation](./docs/photo_maker.md)
- [Using ESRGAN to upscale results](./docs/esrgan.md)
- [Using TAESD to faster decoding](./docs/taesd.md)
- [Docker](./docs/docker.md)
- [Quantization and GGUF](./docs/quantization_and_gguf.md)

## Bindings

These projects wrap `stable-diffusion.cpp` for easier use in other languages/frameworks.

* Golang (non-cgo): [seasonjs/stable-diffusion](https://github.com/seasonjs/stable-diffusion)
* Golang (cgo): [Binozo/GoStableDiffusion](https://github.com/Binozo/GoStableDiffusion)
* C#: [DarthAffe/StableDiffusion.NET](https://github.com/DarthAffe/StableDiffusion.NET)
* Python: [william-murray1204/stable-diffusion-cpp-python](https://github.com/william-murray1204/stable-diffusion-cpp-python)
* Rust: [newfla/diffusion-rs](https://github.com/newfla/diffusion-rs)
* Flutter/Dart: [rmatif/Local-Diffusion](https://github.com/rmatif/Local-Diffusion)

## UIs

These projects use `stable-diffusion.cpp` as a backend for their image generation.

- [Jellybox](https://jellybox.com)
- [Stable Diffusion GUI](https://github.com/fszontagh/sd.cpp.gui.wx)
- [Stable Diffusion CLI-GUI](https://github.com/piallai/stable-diffusion.cpp)
- [Local Diffusion](https://github.com/rmatif/Local-Diffusion)
- [sd.cpp-webui](https://github.com/daniandtheweb/sd.cpp-webui)
- [LocalAI](https://github.com/mudler/LocalAI)

## Contributors

Thank you to all the people who have already contributed to stable-diffusion.cpp!

[![Contributors](https://contrib.rocks/image?repo=leejet/stable-diffusion.cpp)](https://github.com/leejet/stable-diffusion.cpp/graphs/contributors)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=leejet/stable-diffusion.cpp&type=Date)](https://star-history.com/#leejet/stable-diffusion.cpp&Date)

## References

- [ggml](https://github.com/ggerganov/ggml)
- [stable-diffusion](https://github.com/CompVis/stable-diffusion)
- [sd3-ref](https://github.com/Stability-AI/sd3-ref)
- [stable-diffusion-stability-ai](https://github.com/Stability-AI/stablediffusion)
- [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [k-diffusion](https://github.com/crowsonkb/k-diffusion)
- [latent-consistency-model](https://github.com/luosiallen/latent-consistency-model)
- [generative-models](https://github.com/Stability-AI/generative-models/)
- [PhotoMaker](https://github.com/TencentARC/PhotoMaker)
- [Wan2.1](https://github.com/Wan-Video/Wan2.1)
- [Wan2.2](https://github.com/Wan-Video/Wan2.2)