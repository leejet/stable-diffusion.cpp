find_package(Vulkan COMPONENTS glslc REQUIRED)
set(SD_VULKAN_GENERATED "${CMAKE_CURRENT_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY "${SD_VULKAN_GENERATED}")
foreach(NAME rope_apply rms_rope_apply rms_rope_cast)
    set(SHADER "${NAME}")
    set(FLAGS)
    if(NAME STREQUAL "rms_rope_cast")
        set(SHADER rms_rope_apply)
        set(FLAGS -DCAST_F16)
    endif()
    set(SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SHADER}.comp")
    set(SPV "${SD_VULKAN_GENERATED}/${NAME}.spv")
    set(HEADER "${SD_VULKAN_GENERATED}/${NAME}.h")
    # Keep native RoPE rounding: spirv-opt changes the subgroup reduction results.
    add_custom_command(OUTPUT "${HEADER}"
        COMMAND "${Vulkan_GLSLC_EXECUTABLE}" --target-env=vulkan1.2 ${FLAGS} "${SOURCE}" -o "${SPV}"
        COMMAND "${CMAKE_COMMAND}" "-DINPUT=${SPV}" "-DOUTPUT=${HEADER}" "-DNAME=${NAME}"
                -P "${PROJECT_SOURCE_DIR}/cmake/embed_spirv.cmake"
        DEPENDS "${SOURCE}" "${PROJECT_SOURCE_DIR}/cmake/embed_spirv.cmake"
        BYPRODUCTS "${SPV}"
        VERBATIM)
    list(APPEND SD_VULKAN_HEADERS "${HEADER}")
endforeach()
add_custom_target(sd-vulkan-shaders DEPENDS ${SD_VULKAN_HEADERS})
add_dependencies(${SD_LIB} sd-vulkan-shaders)
target_sources(${SD_LIB} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/rope.cpp")
target_include_directories(${SD_LIB} PRIVATE "${SD_VULKAN_GENERATED}")
