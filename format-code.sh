for f in *.cpp *.h *.hpp examples/cli/*.cpp examples/cli/*.h; do
  [[ "$f" == vocab* ]] && continue
  echo "formatting '$f'"
  clang-format -style=file -i "$f"
done