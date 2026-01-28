cmake_minimum_required(VERSION 3.12)
project("stable-diffusion")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if (MSVC)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
    add_compile_definitions(_SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING)
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
option(SD_BUILD_SHARED_GGML_LIB      "sd: build ggml as a separate shared lib" OFF)
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

find_program(GIT_EXE NAMES git git.exe NO_CMAKE_FIND_ROOT_PATH)
if(GIT_EXE)
    execute_process(COMMAND ${GIT_EXE} describe --tags --abbrev=7 --dirty=+
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE SDCPP_BUILD_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    execute_process(COMMAND ${GIT_EXE} rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE SDCPP_BUILD_COMMIT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
endif()

if(NOT SDCPP_BUILD_VERSION)
    set(SDCPP_BUILD_VERSION unknown)
endif()
message(STATUS "stable-diffusion.cpp version ${SDCPP_BUILD_VERSION}")

if(NOT SDCPP_BUILD_COMMIT)
    set(SDCPP_BUILD_COMMIT unknown)
endif()
message(STATUS "stable-diffusion.cpp commit ${SDCPP_BUILD_COMMIT}")

set_property(
  SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/version.cpp
  APPEND PROPERTY COMPILE_DEFINITIONS
  SDCPP_BUILD_COMMIT=${SDCPP_BUILD_COMMIT} SDCPP_BUILD_VERSION=${SDCPP_BUILD_VERSION}
)

if(SD_BUILD_SHARED_LIBS)
    message("-- Build shared library")
    message(${SD_LIB_SOURCES})
    if(NOT SD_BUILD_SHARED_GGML_LIB)
        set(BUILD_SHARED_LIBS OFF)
    endif()
    add_library(${SD_LIB} SHARED ${SD_LIB_SOURCES})
    add_definitions(-DSD_BUILD_SHARED_LIB)
    target_compile_definitions(${SD_LIB} PRIVATE -DSD_BUILD_DLL)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
else()
    message("-- Build static library")
    if(NOT SD_BUILD_SHARED_GGML_LIB)
        set(BUILD_SHARED_LIBS OFF)
    endif()
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
target_compile_features(${SD_LIB} PUBLIC c_std_11 cxx_std_17)


if (SD_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

set(SD_PUBLIC_HEADERS stable-diffusion.h)
set_target_properties(${SD_LIB} PROPERTIES PUBLIC_HEADER "${SD_PUBLIC_HEADERS}")

install(TARGETS ${SD_LIB} LIBRARY PUBLIC_HEADER)
