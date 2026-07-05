for f in src/*.cpp src/*.h src/*.hpp \
         src/conditioning/*.cpp src/conditioning/*.h src/conditioning/*.hpp \
         src/core/*.cpp src/core/*.h src/core/*.hpp \
         src/extensions/*.cpp src/extensions/*.h src/extensions/*.hpp \
         src/runtime/*.cpp src/runtime/*.h src/runtime/*.hpp \
         src/model/*/*.cpp src/model/*/*.h src/model/*/*.hpp \
         src/tokenizers/*.h src/tokenizers/*.cpp src/tokenizers/vocab/*.h src/tokenizers/vocab/*.cpp \
         src/model_io/*.h src/model_io/*.cpp examples/cli/*.cpp examples/cli/*.h examples/server/*.cpp \
         examples/common/*.hpp examples/common/*.h examples/common/*.cpp; do
  [[ -e "$f" ]] || continue
  [[ "$f" == vocab* ]] && continue
  echo "formatting '$f'"
  # if [ "$f" != "stable-diffusion.h" ]; then
  #   clang-tidy -fix -p build_linux/ "$f"
  # fi
  clang-format -style=file -i "$f"
done
