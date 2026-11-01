#!/usr/bin/env bash

set -e

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

for f in src/*.cpp src/*.h src/*.hpp \
         src/conditioning/*.cpp src/conditioning/*.h src/conditioning/*.hpp \
         src/core/*.cpp src/core/*.h src/core/*.hpp \
         src/core/cpu/*.cpp src/core/cpu/*.h \
         src/core/cuda/*.cu src/core/cuda/*.cuh src/core/cuda/*.h \
         src/core/cuda/sageattention/*.cuh \
         src/core/probe/*.cpp src/core/probe/*.h \
         src/core/vulkan/*.cpp src/core/vulkan/*.h src/core/vulkan/shaders/*.comp \
         src/extensions/*.cpp src/extensions/*.h src/extensions/*.hpp \
         src/pipeline/*.cpp src/pipeline/*.h \
         src/runtime/*.cpp src/runtime/*.h src/runtime/*.hpp \
         src/model/*/*.cpp src/model/*/*.h src/model/*/*.hpp \
         src/tokenizers/*.h src/tokenizers/*.cpp \
         src/model_io/*.h src/model_io/*.cpp examples/cli/*.cpp examples/cli/*.h examples/server/*.cpp \
         examples/common/*.hpp examples/common/*.h examples/common/*.cpp; do
  [[ -e "$f" ]] || continue
  echo "formatting '$f'"
  # if [ "$f" != "stable-diffusion.h" ]; then
  #   clang-tidy -fix -p build_linux/ "$f"
  # fi
  clang-format -style=file -i "$f"
done
