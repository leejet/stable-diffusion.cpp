# Building and Using the RPC Server with `stable-diffusion.cpp`

This guide covers how to build a version of [the RPC server from `llama.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md) that is compatible with your version of `stable-diffusion.cpp` to manage multi-backends setups. RPC allows you to offload specific model components to a remote server.

> **Note on Model Location:** The model files (e.g., `.safetensors` or `.gguf`) remain on the **Client** machine. The client parses the file and transmits the necessary tensor data and computational graphs to the server. The server does not need to store the model files locally.

## 1. Building `stable-diffusion.cpp` with RPC client

First, you should build the client application from source. It requires `GGML_RPC=ON` to include the RPC backend to your client.

```bash
mkdir build
cd build
cmake .. \
    -DGGML_RPC=ON \
    # Add other build flags here (e.g., -DSD_VULKAN=ON)
cmake --build . --config Release -j $(nproc)
```

> **Note:** Ensure you add the other flags you would normally use (e.g., `-DSD_VULKAN=ON`, `-DSD_CUDA=ON`, `-DSD_HIPBLAS=ON`, or `-DGGML_METAL=ON`), for more information about building `stable-diffusion.cpp` from source, please refer to the [build.md](build.md) documentation.

## 2. Ensure `llama.cpp` is at the correct commit

`stable-diffusion.cpp`'s RPC client is designed to work with a specific version of `llama.cpp` (compatible with the `ggml` submodule) to ensure API compatibility. The commit hash for `llama.cpp` is stored in `ggml/scripts/sync-llama.last`.

> **Start from Root:** Perform these steps from the root of your `stable-diffusion.cpp` directory.

1.  Read the target commit hash from the submodule tracker:

    ```bash
    # Linux / WSL / MacOS
    HASH=$(cat ggml/scripts/sync-llama.last)

    # Windows (PowerShell)
    $HASH = Get-Content -Path "ggml\scripts\sync-llama.last"
    ```

2.  Clone `llama.cpp` at the target commit .
    ```bash
    git clone https://github.com/ggml-org/llama.cpp.git
    cd llama.cpp
    git checkout $HASH
    ```
    To save on download time and storage, you can use a shallow clone to download only the target commit:
    ```bash
    mkdir -p llama.cpp
    cd llama.cpp
    git init
    git remote add origin https://github.com/ggml-org/llama.cpp.git
    git fetch --depth 1 origin $HASH
    git checkout FETCH_HEAD
    ```

## 3. Build `llama.cpp` (RPC Server)

The RPC server acts as the worker. You must explicitly enable the **backend** (the hardware interface, such as CUDA for Nvidia, Metal for Apple Silicon, or Vulkan) when building, otherwise the server will default to using only the CPU.

To find the correct flags for your system, refer to the official documentation for the [`llama.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md) repository.

> **Crucial:** You must include the compiler flags required to satisfy the API compatibility with `stable-diffusion.cpp` (`-DGGML_MAX_NAME=128`). Without this flag, `GGML_MAX_NAME` will default to `64` for the server, and data transfers between the client and server will fail. Of course, `-DGGML_RPC` must also be enabled.
>
> I recommend disabling the `LLAMA_CURL` flag to avoid unnecessary dependencies, and disabling shared library builds to avoid potential conflicts.

> **Build Target:** We are specifically building the `rpc-server` target. This prevents the build system from compiling the entire `llama.cpp` suite (like `llama-server`), making the build significantly faster.

### Linux / WSL (Vulkan)

```bash
mkdir build
cd build
cmake .. -DGGML_RPC=ON \
    -DGGML_VULKAN=ON \        # Ensure backend is enabled
    -DGGML_BUILD_SHARED_LIBS=OFF \
    -DLLAMA_CURL=OFF \
    -DCMAKE_C_FLAGS=-DGGML_MAX_NAME=128 \
    -DCMAKE_CXX_FLAGS=-DGGML_MAX_NAME=128
cmake --build . --config Release --target rpc-server -j $(nproc)
```

### macOS (Metal)

```bash
mkdir build
cd build
cmake .. -DGGML_RPC=ON \
    -DGGML_METAL=ON \
    -DGGML_BUILD_SHARED_LIBS=OFF \
    -DLLAMA_CURL=OFF \
    -DCMAKE_C_FLAGS=-DGGML_MAX_NAME=128 \
    -DCMAKE_CXX_FLAGS=-DGGML_MAX_NAME=128
cmake --build . --config Release --target rpc-server
```

### Windows (Visual Studio 2022, Vulkan)

```powershell
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64 `
    -DGGML_RPC=ON `
    -DGGML_VULKAN=ON `
    -DGGML_BUILD_SHARED_LIBS=OFF `
    -DLLAMA_CURL=OFF `
    -DCMAKE_C_FLAGS=-DGGML_MAX_NAME=128 `
    -DCMAKE_CXX_FLAGS=-DGGML_MAX_NAME=128
cmake --build . --config Release --target rpc-server
```

## 4. Usage

Once both applications are built, you can run the server and the client to manage your GPU allocation.

### Step A: Run the RPC Server

Start the server. It listens for connections on the default address (usually `localhost:50052`). If your server is on a different machine, ensure the server binds to the correct interface and your firewall allows the connection.

**On the Server :**
If running on the same machine, you can use the default address:

```bash
./rpc-server
```

If you want to allow connections from other machines on the network:

```bash
./rpc-server --host 0.0.0.0
```

> **Security Warning:** The RPC server does not currently support authentication or encryption. **Only run the server on trusted local networks**. Never expose the RPC server directly to the open internet.

> **Drivers & Hardware:** Ensure the Server machine has the necessary drivers installed and functional (e.g., Nvidia Drivers for CUDA, Vulkan SDK, or Metal). If no devices are found, the server will simply fallback to CPU usage.

### Step B: Check if the client is able to connect to the server and see the available devices

We're assuming the server is running on your local machine, and listening on the default port `50052`. If it's running on a different machine, you can replace `localhost` with the IP address of the server.

**On the Client:**

```bash
./sd-cli --rpc localhost:50052 --list-devices
```

If the server is running and the client is able to connect, you should see `RPC0    localhost:50052` in the list of devices.

Example output:
(Client built without GPU acceleration, two GPUs available on the server)

```
List of available GGML devices:
Name    Description
-------------------
CPU     AMD Ryzen 9 5900X 12-Core Processor
RPC0    localhost:50052
RPC1    localhost:50052
```

### Step C: Run with RPC device

If everything is working correctly, you can now run the client while offloading some or all of the work to the RPC server.

Example: Setting the main backend to the RPC0 device for doing all the work on the server.

```bash
./sd-cli -m models/sd1.5.safetensors -p "A cat" --rpc localhost:50052 --main-backend-device RPC0
```

---

## 5. Scaling: Multiple RPC Servers

You can connect the client to multiple RPC servers simultaneously to scale out your hardware usage.

Example: A main machine (192.168.1.10) with 3 GPUs, with one GPU running CUDA and the other two running Vulkan, and a second machine (192.168.1.11) only one GPU.

**On the first machine (Running two server instances):**

**Terminal 1 (CUDA):**

```bash
# Linux / WSL
export CUDA_VISIBLE_DEVICES=0
cd ./build_cuda/bin/Release
./rpc-server --host 0.0.0.0

# Windows PowerShell
$env:CUDA_VISIBLE_DEVICES="0"
cd .\build_cuda\bin\Release
./rpc-server --host 0.0.0.0
```

**Terminal 2 (Vulkan):**

```bash
cd ./build_vulkan/bin/Release
# ignore the first GPU (used by CUDA server)
./rpc-server --host 0.0.0.0 --port 50053 -d Vulkan1,Vulkan2
```

**On the second machine:**

```bash
cd ./build/bin/Release
./rpc-server --host 0.0.0.0
```

**On the Client:**
Pass multiple server addresses separated by commas.

```bash
./sd-cli --rpc 192.168.1.10:50052,192.168.1.10:50053,192.168.1.11:50052 --list-devices
```

The client will map these servers to sequential device IDs (e.g., RPC0 from the first server, RPC2, RPC3 from the second, and RPC4 from the third). With this setup, you could for example use RPC0 for the main backend, RPC1 and RPC2 for the text encoders, and RPC3 for the VAE.

---

## 6. Performance Considerations

RPC performance is heavily dependent on network bandwidth, as large weights and activations must be transferred back and forth over the network, especially for large models, or when using high resolutions. For best results, ensure your network connection is stable and has sufficient bandwidth (>1Gbps recommended).
