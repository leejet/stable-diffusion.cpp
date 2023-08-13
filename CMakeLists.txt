cmake_minimum_required(VERSION 3.12)
project(stable-diffusion)

set(SD_LIB stable-diffusion)
set(SD_TARGET sd)

add_subdirectory(ggml)

add_library(${SD_LIB} stable-diffusion.h stable-diffusion.cpp)
add_executable(${SD_TARGET} main.cpp stb_image_write.h)

target_link_libraries(${SD_LIB} PUBLIC ggml)
target_link_libraries(${SD_TARGET} ${SD_LIB})

target_compile_features(${SD_TARGET} PUBLIC cxx_std_11)

set(CMAKE_CXX_STANDARD 11)