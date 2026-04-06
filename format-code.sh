for f in src/*.cpp src/*.h src/*.hpp src/vocab/*.h src/vocab/*.cpp \
         examples/cli/*.cpp examples/cli/*.h examples/server/*.cpp \
         examples/common/*.hpp examples/common/*.h examples/common/*.cpp; do
  [[ "$f" == vocab* ]] && continue
  echo "formatting '$f'"
  # if [ "$f" != "stable-diffusion.h" ]; then
  #   clang-tidy -fix -p build_linux/ "$f"
  # fi
  clang-format -style=file -i "$f"
done