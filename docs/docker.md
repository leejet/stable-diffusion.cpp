# Docker

## Run CLI

```shell
docker run --rm -v /path/to/models:/models -v /path/to/output/:/output ghcr.io/leejet/stable-diffusion.cpp:master [args...]
# For example
# docker run --rm -v ./models:/models -v ./build:/output ghcr.io/leejet/stable-diffusion.cpp:master -m /models/sd-v1-4.ckpt -p "a lovely cat" -v -o /output/output.png
```

## Run server

```shell
docker run --rm --init -v /path/to/models:/models -v /path/to/output/:/output -p "1234:1234" --entrypoint "/sd-server" ghcr.io/leejet/stable-diffusion.cpp:master [args...]
# For example
# docker run --rm --init -v ./models:/models -v ./build:/output -p "1234:1234" --entrypoint "/sd-server" ghcr.io/leejet/stable-diffusion.cpp:master -m /models/sd-v1-4.ckpt -p "a lovely cat" -v -o /output/output.png
```

## Building using Docker

```shell
docker build -f docker/Dockerfile -t sd .
```

## Building variants using Docker

Vulkan:

```shell
docker build -f docker/Dockerfile.vulkan -t sd .
```

CUDA:

```shell
docker build -f docker/Dockerfile.cuda -t sd-cuda .
```

Useful `--build-arg`s for `docker/Dockerfile.cuda`:

* `CUDA_VERSION` (default `12.6.3`) - CUDA base image tag to build against.
* `CUDA_ARCHITECTURES` - target `CMAKE_CUDA_ARCHITECTURES` (e.g. `110` for
  Jetson Thor / compute capability 11.0). Leave unset to use CMake's default
  architecture list.
* `GGML_CUDA_ENABLE_DYNAMIC_CPU_BACKENDS` (default `ON`) - builds a
  multi-variant, `dlopen`-dispatched CPU backend (`GGML_CPU_ALL_VARIANTS`) so
  one image runs across different host CPUs. Set to `OFF` for a
  `GGML_NATIVE=ON` build tuned for the exact build machine.
* `GGML_CUDA_FA_ALL_QUANTS` - set to `ON` to build flash-attention kernels for
  all K/V quantization combinations.

Example, for a native Jetson Thor build:

```shell
docker build -f docker/Dockerfile.cuda \
  --build-arg CUDA_VERSION=13.0.0 \
  --build-arg CUDA_ARCHITECTURES=110 \
  --build-arg GGML_CUDA_ENABLE_DYNAMIC_CPU_BACKENDS=OFF \
  -t sd-cuda .
```

`docker/Dockerfile.cuda` produces a minimal `FROM scratch` runtime image
(built in a `rootfs` stage that copies the binaries plus their `ldd`-resolved
shared library dependencies, skipping `libcuda.so*` since the NVIDIA
Container Toolkit injects the driver at run time). The frontend needs no
separate copy step: `sd-server` compiles the built single-file frontend
straight into the binary as a generated header, so it's already inside the
`sd-cli`/`sd-server` executables copied into the image.

Because of this, the image layout differs from the other `docker/Dockerfile*`
variants:

* Binaries live at `/opt/sd-cpp/bin/sd-cli` and `/opt/sd-cpp/bin/sd-server`
  (not `/sd-cli` / `/sd-server`), and there is no shell in the image.
* The default `ENTRYPOINT` is `/opt/sd-cpp/bin/sd-cli`; run the server with
  `--entrypoint /opt/sd-cpp/bin/sd-server`.
* The container runs as non-root `uid:gid 1000:1000` with `WORKDIR
  /home/sdcpp`, so files written to bind-mounted output directories are
  owned by that uid on the host.

```shell
# CLI
docker run --rm --gpus all \
  -v /path/to/models:/models -v /path/to/output/:/output \
  sd-cuda -m /models/sd-v1-4.ckpt -p "a lovely cat" -v -o /output/output.png

# Server
docker run --rm --gpus all \
  -v /path/to/models:/models \
  -p 1234:1234 \
  --entrypoint /opt/sd-cpp/bin/sd-server \
  sd-cuda -m /models/sd-v1-4.ckpt -l 0.0.0.0 --listen-port 1234
```

## Run locally built image's CLI

```shell
docker run --rm -v /path/to/models:/models -v /path/to/output/:/output sd [args...]
# For example
# docker run --rm -v ./models:/models -v ./build:/output sd -m /models/sd-v1-4.ckpt -p "a lovely cat" -v -o /output/output.png
```
