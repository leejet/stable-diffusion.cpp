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
option(SD_CUDA                       "sd: cuda backend" OFF)
option(SD_HIPBLAS                    "sd: rocm backend" OFF)
option(SD_METAL                      "sd: metal backend" OFF)
option(SD_VULKAN                     "sd: vulkan backend" OFF)
option(SD_OPENCL                     "sd: opencl backend" OFF)
option(SD_SYCL                       "sd: sycl backend" OFF)
option(SD_MUSA                       "sd: musa backend" OFF)
option(SD_FAST_SOFTMAX               "sd: x1.5 faster softmax, indeterministic (sometimes, same seed don't generate same image), cuda only" OFF)
option(SD_BUILD_SHARED_LIBS          "sd: build shared libs" OFF)
option(SD_USE_SYSTEM_GGML            "sd: use system-installed GGML library" OFF)
#option(SD_BUILD_SERVER               "sd: build server example"                           ON)

if(SD_CUDA)
    message("-- Use CUDA as backend stable-diffusion")
    set(GGML_CUDA ON)
    add_definitions(-DSD_USE_CUDA)
endif()

if(SD_METAL)
    message("-- Use Metal as backend stable-diffusion")
    set(GGML_METAL ON)
    add_definitions(-DSD_USE_METAL)
endif()

if (SD_VULKAN)
    message("-- Use Vulkan as backend stable-diffusion")
    set(GGML_VULKAN ON)
    add_definitions(-DSD_USE_VULKAN)
endif ()

if (SD_OPENCL)
    message("-- Use OpenCL as backend stable-diffusion")
    set(GGML_OPENCL ON)
    add_definitions(-DSD_USE_OPENCL)
endif ()

if (SD_HIPBLAS)
    message("-- Use HIPBLAS as backend stable-diffusion")
    set(GGML_HIP ON)
    add_definitions(-DSD_USE_CUDA)
    if(SD_FAST_SOFTMAX)
        set(GGML_CUDA_FAST_SOFTMAX ON)
    endif()
endif ()

if(SD_MUSA)
    message("-- Use MUSA as backend stable-diffusion")
    set(GGML_MUSA ON)
    add_definitions(-DSD_USE_CUDA)
    if(SD_FAST_SOFTMAX)
        set(GGML_CUDA_FAST_SOFTMAX ON)
    endif()
endif()

set(SD_LIB stable-diffusion)

file(GLOB SD_LIB_SOURCES
    "*.h"
    "*.cpp"
    "*.hpp"
)

# we can get only one share lib
if(SD_BUILD_SHARED_LIBS)
    message("-- Build shared library")
    message(${SD_LIB_SOURCES})
    set(BUILD_SHARED_LIBS OFF)
    add_library(${SD_LIB} SHARED ${SD_LIB_SOURCES})
    add_definitions(-DSD_BUILD_SHARED_LIB)
    target_compile_definitions(${SD_LIB} PRIVATE -DSD_BUILD_DLL)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
else()
    message("-- Build static library")
    set(BUILD_SHARED_LIBS OFF)
    add_library(${SD_LIB} STATIC ${SD_LIB_SOURCES})
endif()

if(SD_SYCL)
    message("-- Use SYCL as backend stable-diffusion")
    set(GGML_SYCL ON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-narrowing -fsycl")
    add_definitions(-DSD_USE_SYCL)
    # disable fast-math on host, see:
    # https://www.intel.com/content/www/us/en/docs/cpp-compiler/developer-guide-reference/2021-10/fp-model-fp.html
    if (WIN32)
        set(SYCL_COMPILE_OPTIONS /fp:precise)
    else()
        set(SYCL_COMPILE_OPTIONS -fp-model=precise)
    endif()
    message("-- Turn off fast-math for host in SYCL backend")
    target_compile_options(${SD_LIB} PRIVATE ${SYCL_COMPILE_OPTIONS})
endif()

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

if (NOT SD_USE_SYSTEM_GGML)
    # see https://github.com/ggerganov/ggml/pull/682
    add_definitions(-DGGML_MAX_NAME=128)
endif()

# deps
# Only add ggml if it hasn't been added yet
if (NOT TARGET ggml)
    if (SD_USE_SYSTEM_GGML)
        find_package(ggml REQUIRED)
        if (NOT ggml_FOUND)
            message(FATAL_ERROR "System-installed GGML library not found.")
        endif()
        add_library(ggml ALIAS ggml::ggml)
    else()
        add_subdirectory(ggml)
    endif()
endif()

add_subdirectory(thirdparty)

target_link_libraries(${SD_LIB} PUBLIC ggml zip)
target_include_directories(${SD_LIB} PUBLIC . thirdparty)
target_compile_features(${SD_LIB} PUBLIC cxx_std_11)


if (SD_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

