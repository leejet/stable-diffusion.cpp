file(READ "${INPUT}" SPIRV HEX)
string(REGEX REPLACE "(..)(..)(..)(..)" "0x\\4\\3\\2\\1," SPIRV_WORDS "${SPIRV}")
file(WRITE "${OUTPUT}" "#include <cstdint>\nnamespace sd_vulkan_shaders {\nstatic const uint32_t ${NAME}[] = {${SPIRV_WORDS}};\n}\n")
