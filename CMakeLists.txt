cmake_minimum_required(VERSION 3.12)
project("stable-diffusion")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(SD_STANDALONE ON)
else()
    set(SD_STANDALONE OFF)
endif()

#
# Option list
#

# general
#option(SD_BUILD_TESTS                "sd: build tests"    ${SD_STANDALONE})
option(SD_BUILD_EXAMPLES             "sd: build examples" ${SD_STANDALONE})
option(BUILD_SHARED_LIBS             "sd: build shared libs" OFF)
#option(SD_BUILD_SERVER               "sd: build server example"                           ON)


# deps
add_subdirectory(ggml)

set(SD_LIB stable-diffusion)

add_library(${SD_LIB} stable-diffusion.h stable-diffusion.cpp)
target_link_libraries(${SD_LIB} PUBLIC ggml)
target_include_directories(${SD_LIB} PUBLIC .)
target_compile_features(${SD_LIB} PUBLIC cxx_std_11)


if (SD_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

