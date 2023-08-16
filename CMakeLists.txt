cmake_minimum_required(VERSION 3.12)
project(stable-diffusion)

# deps
add_subdirectory(ggml)

set(SD_LIB stable-diffusion)

add_library(${SD_LIB} stable-diffusion.h stable-diffusion.cpp)
target_link_libraries(${SD_LIB} PUBLIC ggml)
target_include_directories(${SD_LIB} PUBLIC .)
target_compile_features(${SD_LIB} PUBLIC cxx_std_11)

# examples
add_subdirectory(examples)
