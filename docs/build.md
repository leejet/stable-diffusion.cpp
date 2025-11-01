# Build from scratch

## Get the Code

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

This provides GPU acceleration using NVIDIA GPU. Make sure to have the CUDA toolkit installed. You can download it from your Linux distro's package manager (e.g. `apt install nvidia-cuda-toolkit`) or from here: [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads). Recommended to have at least 4 GB of VRAM.

```shell
mkdir build && cd build
cmake .. -DSD_CUDA=ON
cmake --build . --config Release
```

## Build with HipBLAS

This provides GPU acceleration using AMD GPU. Make sure to have the ROCm toolkit installed.
To build for another GPU architecture than installed in your system, set `$GFX_NAME` manually to the desired architecture (replace first command). This is also necessary if your GPU is not officially supported by ROCm, for example you have to set `$GFX_NAME` manually to `gfx1030` for consumer RDNA2 cards.

Windows User Refer to [docs/hipBLAS_on_Windows.md](docs%2FhipBLAS_on_Windows.md) for a comprehensive guide.

```shell
mkdir build && cd build
if command -v rocminfo; then export GFX_NAME=$(rocminfo | awk '/ *Name: +gfx[1-9]/ {print $2; exit}'); else echo "rocminfo missing!"; fi
if [ -z "${GFX_NAME}" ]; then echo "Error: Couldn't detect GPU!"; else echo "Building for GPU: ${GFX_NAME}"; fi
cmake .. -G "Ninja" -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DSD_HIPBLAS=ON -DCMAKE_BUILD_TYPE=Release -DGPU_TARGETS=$GFX_NAME -DAMDGPU_TARGETS=$GFX_NAME -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON
cmake --build . --config Release
```

## Build with MUSA

This provides GPU acceleration using Moore Threads GPU. Make sure to have the MUSA toolkit installed.

```shell
mkdir build && cd build
cmake .. -DCMAKE_C_COMPILER=/usr/local/musa/bin/clang -DCMAKE_CXX_COMPILER=/usr/local/musa/bin/clang++ -DSD_MUSA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

## Build with Metal

Using Metal makes the computation run on the GPU. Currently, there are some issues with Metal when performing operations on very large matrices, making it highly inefficient at the moment. Performance improvements are expected in the near future.

```shell
mkdir build && cd build
cmake .. -DSD_METAL=ON
cmake --build . --config Release
```

## Build with Vulkan

Install Vulkan SDK from https://www.lunarg.com/vulkan-sdk/.

```shell
mkdir build && cd build
cmake .. -DSD_VULKAN=ON
cmake --build . --config Release
```

## Build with OpenCL (for Adreno GPU)

Currently, it supports only Adreno GPUs and is primarily optimized for Q4_0 type

To build for Windows ARM please refers to [Windows 11 Arm64](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/OPENCL.md#windows-11-arm64)

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

    ninja
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

ninja
```
*(Note: Don't forget to include `LD_LIBRARY_PATH=/vendor/lib64` in your command line before running the binary)*

## Build with SYCL

Using SYCL makes the computation run on the Intel GPU. Please make sure you have installed the related driver and [Intel® oneAPI Base toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html) before start. More details and steps can refer to [llama.cpp SYCL backend](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md#linux).

```shell
# Export relevant ENV variables
source /opt/intel/oneapi/setvars.sh

# Option 1: Use FP32 (recommended for better performance in most cases)
cmake .. -DSD_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx

# Option 2: Use FP16
cmake .. -DSD_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_SYCL_F16=ON

cmake --build . --config Release
```
