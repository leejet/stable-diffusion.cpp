add_executable(sd-test-image-decode
    image_decode.cpp
    ../examples/common/common.cpp
    ../examples/common/log.cpp
    ../examples/common/media_io.cpp
)

target_include_directories(sd-test-image-decode PRIVATE
    "${PROJECT_SOURCE_DIR}/examples"
    "${PROJECT_SOURCE_DIR}/src"
)

target_link_libraries(sd-test-image-decode PRIVATE stable-diffusion zip ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(sd-test-image-decode PUBLIC c_std_11 cxx_std_17)

add_test(NAME sd-test-image-decode COMMAND sd-test-image-decode)
