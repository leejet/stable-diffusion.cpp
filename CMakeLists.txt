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
option(SD_BUILD_EXAMPLES              "sd: build examples" ${SD_STANDALONE})
option(SD_BUILD_SHARED_LIBS           "sd: build shared libs" OFF)
#option(SD_BUILD_SERVER               "sd: build server example"                           ON)

# deps
add_subdirectory(ggml)
include_directories(ggml/include)

if (WIN32 OR MSVC)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".lib")
elseif (UNIX)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
endif()

set(SD_LIB stable-diffusion)
if (SD_BUILD_SHARED_LIBS)
    set_target_properties(ggml PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_library(${SD_LIB} SHARED stable-diffusion.cpp stable-diffusion-c.cpp stable-diffusion.h stable-diffusion-c.h)
    set_target_properties(${SD_LIB} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(${SD_LIB} PRIVATE STABLE_DIFFUSION_SHARED STABLE_DIFFUSION_BUILD)
    target_link_libraries(${SD_LIB} PRIVATE ggml ${SD_EXTRA_LIBS} ${CMAKE_THREAD_LIBS_INIT})
else ()
    add_library(${SD_LIB} stable-diffusion.h stable-diffusion.cpp)
    target_link_libraries(${SD_LIB} PUBLIC ggml)
endif ()

target_include_directories(${SD_LIB} PUBLIC .)
target_compile_features(${SD_LIB} PUBLIC cxx_std_11)


if (SD_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
