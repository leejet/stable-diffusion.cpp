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

Initialize the submodules from the repository root before building. Docker copies
the local checkout, so its build context must include the submodule contents:

```shell
git submodule update --init --recursive
docker build -f docker/Dockerfile -t sd .
```

## Building variants using Docker

### CUDA (NVIDIA GPU):

```shell
docker build -f docker/Dockerfile.cuda -t sd-cuda .
```

To select GPU architectures, pass `--build-arg "CUDA_ARCHITECTURES=89"` (for
example, RTX 4090). Run the image on a host configured with NVIDIA Container
Toolkit, using `docker run --rm --gpus all ... sd-cuda [args...]`.

### Vulkan:

```shell
docker build -f docker/Dockerfile.vulkan -t sd .
```

### ROCm (AMD GPU):

Defining one or multiple targets with `--build-arg AMDGPU_TARGETS` is strongly recommended.
Building the image will otherwise likely fail or build for a target that does not fit your
hardware.

```shell
# Build for one target
docker build -f docker/Dockerfile.rocm --build-arg "AMDGPU_TARGETS=gfx1100" -t sd .
# Build for multiple targets
docker build -f docker/Dockerfile.rocm --build-arg "AMDGPU_TARGETS=gfx1030;gfx1100" -t sd .
```

## Run locally built image's CLI

```shell
docker run --rm -v /path/to/models:/models -v /path/to/output/:/output sd [args...]
# For example
# docker run --rm -v ./models:/models -v ./build:/output sd -m /models/sd-v1-4.ckpt -p "a lovely cat" -v -o /output/output.png
```

### ROCm (AMD GPU)

The user that executes this command needs to be in the `video` group and have read and write
access to `/dev/dri` and `/dev/kfd`.

`--device` and `--group-add` allow the container access to the GPU.

```shell
docker run --rm --device=/dev/dri --device=/dev/kfd --group-add video \
  -v /path/to/models:/models -v /path/to/output/:/output \
  sd [args...]
# For example
# docker run --rm --device=/dev/dri --device=/dev/kfd --group-add video \
#   -v ./models:/models -v ./build:/output \
#   sd -m /models/sd-v1-4.ckpt -v -o /output/output.png \
#   -p "a lovely cat"
```
