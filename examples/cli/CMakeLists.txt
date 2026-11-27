set(TARGET sd-cli)

add_executable(${TARGET}
    ../common/common.cpp
    ../common/log.cpp
    ../common/media_io.cpp
    ../common/media_metadata.cpp
    main.cpp
    dgl.cpp
    media_input.cpp
)
if(APPLE)
    sd_set_macos_rpaths(${TARGET})
endif()
target_include_directories(${TARGET} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE stable-diffusion zip ${CMAKE_THREAD_LIBS_INIT})
if(SD_WEBP)
    target_compile_definitions(${TARGET} PRIVATE SD_USE_WEBP)
    target_link_libraries(${TARGET} PRIVATE webp libwebpmux)
endif()
if(SD_WEBM)
    target_compile_definitions(${TARGET} PRIVATE SD_USE_WEBM)
    target_link_libraries(${TARGET} PRIVATE webm)
endif()
target_compile_features(${TARGET} PUBLIC c_std_11 cxx_std_17)

set(SD_DGL_BUILTIN_CATALOG "${PROJECT_SOURCE_DIR}/docs/dgl-builtins.md")
add_custom_target(dgl-update-builtins
    COMMAND $<TARGET_FILE:${TARGET}> -M dgl --write-builtins "${SD_DGL_BUILTIN_CATALOG}"
    DEPENDS ${TARGET}
    VERBATIM)
add_custom_target(dgl-check-builtins
    COMMAND $<TARGET_FILE:${TARGET}> -M dgl --check-builtins "${SD_DGL_BUILTIN_CATALOG}"
    DEPENDS ${TARGET}
    VERBATIM)
