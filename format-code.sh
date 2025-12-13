for f in *.cpp *.h *.hpp examples/cli/*.cpp examples/common/*.hpp examples/cli/*.h examples/server/*.cpp; do
  [[ "$f" == vocab* ]] && continue
  echo "formatting '$f'"
  # if [ "$f" != "stable-diffusion.h" ]; then
  #   clang-tidy -fix -p build_linux/ "$f"
  # fi
  clang-format -style=file -i "$f"
done