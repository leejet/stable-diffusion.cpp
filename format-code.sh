for f in src/*.cpp src/*.h src/*.hpp src/tokenizers/*.h src/tokenizers/*.cpp src/tokenizers/vocab/*.h src/tokenizers/vocab/*.cpp \
         examples/cli/*.cpp examples/cli/*.h examples/server/*.cpp \
         examples/common/*.hpp examples/common/*.h examples/common/*.cpp; do
  [[ "$f" == vocab* ]] && continue
  echo "formatting '$f'"
  # if [ "$f" != "stable-diffusion.h" ]; then
  #   clang-tidy -fix -p build_linux/ "$f"
  # fi
  clang-format -style=file -i "$f"
done