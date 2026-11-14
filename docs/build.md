# Build from scratch

## Get the Code

```shell
git clone --recursive https://github.com/leejet/stable-diffusion.cpp
cd stable-diffusion.cpp
```

If you already cloned the repository without submodules, initialize them from the repository root:

```shell
git submodule update --init --recursive
```

To update an existing checkout and its submodules:

```shell
cd stable-diffusion.cpp
git pull origin master
git submodule update --init --recursive
```

## Selecting a GGML source tree

By default, sd.cpp builds the patched GGML submodule in `ggml/`. To build with
an upstream GGML checkout instead, enable `SD_USE_UPSTREAM_GGML` and set
`SD_GGML_SOURCE_DIR`:

```shell
cmake -S . -B build-upstream -DSD_USE_UPSTREAM_GGML=ON -DSD_GGML_SOURCE_DIR=../ggml-upstream
cmake --build build-upstream --config Release
```

Upstream compatibility mode uses the selected library and matching private
headers. The patched GGML build uses public headers, including `ggml-ext.h` and
`ggml-graph.h`. Backend options such as `-DSD_CUDA=ON` apply to the selected tree
as usual.

`SD_USE_UPSTREAM_GGML` defaults to `OFF`, which enables the patched GGML
extensions. Set it to `ON` when using upstream GGML; it selects the compatibility
mode and does not download or replace the GGML source tree. Upstream mode keeps
the original FP8 safetensors handling: FP8 tensors are converted to F16 at load
time (one byte per element in the file, two in RAM and VRAM). INT8
tensorwise/convrot is disabled and its model files are rejected with an explicit
error. FP8 GGUF files, FP8 weight type requests and tensor type rules are also
rejected; no automatic conversion is performed.

Upstream GGML may lack some operators and performance optimizations provided by
the patched version. A warning is emitted during CMake configuration and when
creating an inference context. Ordinary floating-point and shared GGML
quantization types remain available, subject to backend operator support.

`SD_USE_SYSTEM_GGML=ON` instead links an installed GGML CMake package, located
with `ggml_DIR` or `CMAKE_PREFIX_PATH`. An installed patched package must include
the extension and graph APIs; its source tree is not required for private
headers. The installed library must use the same ABI settings as sd.cpp,
including `GGML_MAX_NAME`. For an upstream package, also set
`SD_USE_UPSTREAM_GGML=ON` and point `SD_GGML_SOURCE_DIR` to its matching source
tree for the compatibility layer's private headers.

## WebP and WebM Support in Examples

The example applications (`examples/cli` and `examples/server`) use `libwebp` to support WebP image I/O, and `examples/cli` can also use `libwebm` for `.webm` video output. Both are enabled by default. WebM output currently reuses `libwebp` to encode each frame as VP8 before muxing with `libwebm`.

If you do not want WebP/WebM support, you can disable them at configure time:

```shell
mkdir build && cd build
cmake .. -DSD_WEBP=OFF -DSD_WEBM=OFF
cmake --build . --config Release
```

If the submodules are not available, you can also link against system packages instead:

```shell
mkdir build && cd build
cmake .. -DSD_USE_SYSTEM_WEBP=ON -DSD_USE_SYSTEM_WEBM=ON
cmake --build . --config Release
```

## Build (CPU only)

If you don't have a GPU or CUDA installed, you can build a CPU-only version.

```shell
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

## Build with OpenBLAS

```shell
mkdir build && cd build
cmake .. -DGGML_OPENBLAS=ON
cmake --build . --config Release
```

## Build with CUDA

The sd.cpp SageAttention extension is included when using CUDA with patched
GGML (`SD_USE_UPSTREAM_GGML=OFF`).
See [SageAttention](guides/sage_attention.md) for GPU requirements and `--sage-attn` usage.

This provides GPU acceleration using NVIDIA GPU. Make sure to have the CUDA toolkit installed. You can download it from your Linux distro's package manager (e.g. `apt install nvidia-cuda-toolkit`) or from here: [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads). Recommended to have at least 4 GB of VRAM.

```shell
mkdir build && cd build
cmake .. -DSD_CUDA=ON
cmake --build . --config Release
```

When building GGML from source, sd.cpp's CUDA extensions use the same target
architectures as GGML, including native GPU detection with `GGML_NATIVE=ON`.
To select target GPUs explicitly, pass `-DCMAKE_CUDA_ARCHITECTURES=89` (Ada),
for example, or a quoted list such as `-DCMAKE_CUDA_ARCHITECTURES="80;89"`.

CUDA extensions use sd.cpp's `thirdparty/cutlass` submodule by default
for eligible INT8 GEMM and CuTe attention operations. If its headers are missing,
CMake stops with an initialization hint; it does not download them automatically.
Run `git submodule update --init --recursive` from the repository root before
configuring.

To build with the existing CUDA implementations without CUTLASS, configure from
the repository root with:

```shell
cmake -S . -B build -DSD_CUDA=ON -DSD_CUDA_CUTLASS_PATH:PATH=
cmake --build build --config Release
```

Alternatively, set `SD_CUDA_CUTLASS_PATH` to another local CUTLASS source
directory (tested with 3.9.2). Existing CMake caches retain their configured path;
pass `-USD_CUDA_CUTLASS_PATH` when reconfiguring to restore the bundled default.
Existing `GGML_CUDA_CUTLASS_PATH` caches are migrated on first configuration:
the old bundled path becomes `thirdparty/cutlass`, while custom paths and an
empty disabled value are preserved. An explicitly set `SD_CUDA_CUTLASS_PATH`
takes precedence. Non-CUDA builds do not require CUTLASS.

## Build with HIPBLAS

Use `-DSD_HIPBLAS=ON` for AMD GPU acceleration through HIP and hipBLAS.
Install a compatible AMD GPU driver and the ROCm SDK, including the HIP compiler,
hipBLAS, rocBLAS, and their development files. See
[AMD's ROCm installation guide](https://rocm.docs.amd.com/en/latest/install/rocm.html)
for supported GPUs, operating systems, and installation methods.

When the hipBLASLt development package (version 1.0 or later) is available,
CMake also enables INT8 GEMM tuning. The first run of an eligible matrix shape
benchmarks the no-workspace candidates returned by hipBLASLt for the current
device, including CDNA devices, and caches the result in the process. It
selects a hipBLASLt candidate that runs successfully and beats hipBLAS, or a
working candidate if hipBLAS does not support the operation.
Set `SD_HIP_DISABLE_I8_LT=1` to use the hipBLAS GEMM path for comparison.

With a compiled wave32 WMMA kernel, the HIP flash attention path supports
head dimensions 40, 52, 64, 72, 80, 120, 128, 160, 256, and 320, including
grouped-query attention. It requires unmasked attention with at least 512
query tokens and 512 key tokens. Set `SD_HIP_DISABLE_WMMA_ATTN=1` to use
ggml's native attention path for comparison.

**Linux**

With ROCm installed, use the default CMake generator and host compilers:

```shell
mkdir build && cd build
cmake .. -DSD_HIPBLAS=ON
cmake --build . --config Release
```

For an SDK outside the default installation path, set `ROCM_PATH` to its root
and add `$ROCM_PATH/bin` to `PATH` before configuring. If its shared libraries
are not on the system library search path, add `$ROCM_PATH/lib` and
`$ROCM_PATH/lib64` to `LD_LIBRARY_PATH` when running the executable.

**Windows**

Follow the Windows environment setup in [Ninja](#ninja). The Windows HIP build
requires ROCm Clang with Ninja; the Visual Studio generator is not supported.

Set `HIP_PATH` to the SDK root. The HIP SDK installer normally sets this variable;
for [ROCm wheels](https://github.com/ROCm/TheRock/blob/main/RELEASES.md), run
`rocm-sdk init` and use `rocm-sdk path --root`. The build commands below cover
both the installer and wheel SDK compiler layouts.

For the wheel SDK, also set the device library path before building:

```powershell
$env:HIP_DEVICE_LIB_PATH = "$env:HIP_PATH\lib\llvm\amdgcn\bitcode"
```

Then configure and build:

```powershell
$env:ROCM_PATH = $env:HIP_PATH
$env:PATH = "$env:HIP_PATH\bin;$env:HIP_PATH\lib\llvm\bin;$env:PATH"
cmake -S . -B build-hip -G Ninja -DSD_HIPBLAS=ON -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release
cmake --build build-hip --config Release
```

Keep the SDK's `bin` directory on `PATH` when running `build-hip\bin\sd-cli.exe`
so its runtime DLLs can be found.

**GPU targets and verification**

CMake normally detects the installed GPU. To select a target explicitly or build
on a machine without an AMD GPU, add `-DGPU_TARGETS=gfx1100` to the configure
command (for example, for an RX 7900 XTX, XT, or GRE). For multiple targets, use a
quoted list such as `-DGPU_TARGETS="gfx1100;gfx1200"`. Check AMD's
[GPU specifications](https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html)
for the correct target; the installed SDK libraries must also support it.

Use a fresh build directory when switching SDKs or compilers. Run the resulting
`sd-cli --version` to check that the executable starts without loading a model.

## Build with MUSA

This provides GPU acceleration using Moore Threads GPUs. The configuration below follows upstream validation with MUSA SDK 5.2.0 and MTT S5000 (PH1, `mp_31`). Install that toolkit before building. Older SDKs such as 4.2 may lack the CUB headers required by the current backend.

```shell
mkdir build && cd build
cmake .. -DCMAKE_C_COMPILER=/usr/local/musa/bin/clang -DCMAKE_CXX_COMPILER=/usr/local/musa/bin/clang++ -DSD_MUSA=ON -DMUSA_ARCHITECTURES=31 -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

The MUSA Docker image uses the same SDK 5.2.0 development image as upstream for both build and runtime stages, and targets `mp_31` only. It includes the development tools and PyTorch from that base image, so it is larger than a minimal runtime image. Other GPU architectures require separate validation.

## Build with Metal

Using Metal makes the computation run on the GPU. Currently, there are some issues with Metal when performing operations on very large matrices, making it highly inefficient at the moment. Performance improvements are expected in the near future.

```shell
mkdir build && cd build
cmake .. -DSD_METAL=ON
cmake --build . --config Release
```

## Build with Vulkan

Install Vulkan SDK from https://www.lunarg.com/vulkan-sdk/.
On Ubuntu, install the Vulkan development packages and SPIR-V headers:

```shell
sudo apt-get install build-essential libvulkan-dev glslc spirv-headers
```

```shell
mkdir build && cd build
cmake .. -DSD_VULKAN=ON
cmake --build . --config Release
```

## Build with OpenCL (for Adreno GPU)

Currently, it supports only Adreno GPUs and is primarily optimized for Q4_0 type

To build for Windows ARM please refers to [Windows 11 Arm64](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/OPENCL.md#windows-11-arm64)

For Android, prepare [Ninja](#ninja) and install the Android NDK from the
[official Android developer site](https://developer.android.com/ndk/downloads).

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
    ```shell
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

    cmake --build .
    # Replace <YOUR_NDK_PATH>
    # e.g., cp libOpenCL.so /path/to/android-ndk-r26c/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android
    sudo cp libOpenCL.so <YOUR_NDK_PATH>/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android
    cd ../..
    ```

Build `stable-diffusion.cpp` for Android with OpenCL:

```shell
mkdir build-android && cd build-android

# Replace <YOUR_NDK_PATH> with your actual NDK installation path
# e.g., -DCMAKE_TOOLCHAIN_FILE=/path/to/android-ndk-r26c/build/cmake/android.toolchain.cmake
cmake .. -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=<YOUR_NDK_PATH>/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DGGML_OPENMP=OFF \
  -DSD_OPENCL=ON

cmake --build .
```
*(Note: Don't forget to include `LD_LIBRARY_PATH=/vendor/lib64` in your command line before running the binary)*

## Build with SYCL

Using SYCL makes the computation run on the Intel GPU. Please make sure you have installed the related driver and [Intel® oneAPI Base toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html) before start. More details and steps can refer to [llama.cpp SYCL backend](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md#linux).

```shell
# Export relevant ENV variables
source /opt/intel/oneapi/setvars.sh

# Make and enter build dir
mkdir build && cd build

# Option 1: Use FP32 (recommended for better performance in most cases)
cmake .. -DSD_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx

# Option 2: Use FP16
cmake .. -DSD_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_SYCL_F16=ON

cmake --build . --config Release
```

## Build tools and compiler cache

The settings in this section can be combined with the backend options in
this guide.

### Ninja

Install Ninja and make sure `ninja --version` works in your build terminal.
For native Windows builds with MSVC or ROCm Clang, install Visual Studio
2022 C++ build tools with the Windows SDK, CMake, and Ninja, then open an
x64 **Developer PowerShell for VS 2022**.

From the repository root, configure and build a Release version with:

```shell
cmake -S . -B build-ninja -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-ninja --parallel 8
```

This builds the CPU backend by default. Add backend options such as
`-DSD_CUDA=ON` or `-DSD_VULKAN=ON` to the configure command as needed.
With MSVC, you can explicitly select the compiler by adding
`-DCMAKE_C_COMPILER=cl.exe -DCMAKE_CXX_COMPILER=cl.exe`.
Adjust `--parallel 8` for your available CPU and memory resources.

Use a separate build directory when switching from another CMake generator,
such as Visual Studio. With this Ninja generator, select the build type at
configure time using `CMAKE_BUILD_TYPE`.

### ccache (optional)

Install [ccache](https://ccache.dev/download.html) and add it to `PATH`.
Check that it is available and optionally set a cache size limit:

```shell
ccache --version
ccache --max-size=20G
```

To cache C and C++ compilation, configure with compiler launchers:

```shell
cmake -S . -B build-ninja -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
cmake --build build-ninja --parallel 8
ccache --show-stats
```

For CUDA builds, also add `-DSD_CUDA=ON` and
`-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache` to the configure command so CUDA
compilation is cached as well. Keep any other backend or compiler options
used for your build.

These [compiler launcher settings](https://cmake.org/cmake/help/latest/prop_tgt/LANG_COMPILER_LAUNCHER.html)
work with Ninja and Makefile generators. The Visual Studio generator does
not use them.

The first compilation populates the cache. After a successful build, you can
check reuse by rebuilding the same configuration with the cache intact:

```shell
ccache --zero-stats
cmake --build build-ninja --clean-first --parallel 8
ccache --show-stats
```

Look for cache hits in the statistics. A normal incremental build with no
changed files skips compilation and does not produce new cache hits.

### MSVC compilation parallelism

MSVC builds limit parallel C/C++ compilation within each project by default.
For `n` logical processors, the limit is `n` when `n <= 2`, `n - 1` when
`3 <= n <= 8`, and `n - 2` when `n > 8`. The default is calculated when the
build directory is first configured with this setting. Override it with
`-DSD_MSVC_MP_COUNT=4`, or pass `-USD_MSVC_MP_COUNT` to recalculate it.

For Visual Studio builds, keep project-level parallelism at one to avoid
multiplying the per-project limit:

```shell
cmake --build build --config Release --parallel 1
```

This setting controls MSVC's `/MP` option, not CUDA compiler parallelism or a
hard CPU usage limit. For Ninja job limits, see [Ninja](#ninja).
