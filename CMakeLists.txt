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
option(SD_CUBLAS                     "sd: cuda backend" OFF)
option(SD_HIPBLAS                    "sd: rocm backend" OFF)
option(SD_FLASH_ATTN            "sd: use flash attention for x4 less memory usage" OFF)
option(BUILD_SHARED_LIBS             "sd: build shared libs" OFF)
#option(SD_BUILD_SERVER               "sd: build server example"                           ON)

if(SD_CUBLAS)
	message("Use CUBLAS as backend stable-diffusion")
    set(GGML_CUBLAS ON)
    add_definitions(-DSD_USE_CUBLAS)
endif()

if(SD_FLASH_ATTN)
    message("Use Flash Attention for memory optimization")
    add_definitions(-DSD_USE_FLASH_ATTENTION)
endif()


set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

# deps
if (SD_HIPBLAS)
    add_library(ggml OBJECT
            ${CMAKE_SOURCE_DIR}/ggml/src/ggml.c
            ${CMAKE_SOURCE_DIR}/ggml/src/ggml-alloc.c
            ${CMAKE_SOURCE_DIR}/ggml/src/ggml-backend.c
            ${CMAKE_SOURCE_DIR}/ggml/include/ggml/ggml.h
            ${CMAKE_SOURCE_DIR}/ggml/include/ggml/ggml-alloc.h
            ${CMAKE_SOURCE_DIR}/ggml/src/ggml-backend.h
    )
else ()
    add_subdirectory(ggml)
endif ()

if (SD_HIPBLAS)
    message("Use HIPBLAS as backend stable-diffusion")
    add_definitions(-DSD_USE_HIPBLAS)
    add_definitions(-DSD_USE_CUBLAS)

    list(APPEND CMAKE_PREFIX_PATH /opt/rocm)

    if (NOT ${CMAKE_C_COMPILER_ID} MATCHES "Clang")
        message(WARNING "Only LLVM is supported for HIP, hint: CC=/opt/rocm/llvm/bin/clang")
    endif()
    if (NOT ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
        message(WARNING "Only LLVM is supported for HIP, hint: CXX=/opt/rocm/llvm/bin/clang++")
    endif()

    find_package(hip)
    find_package(hipblas)
    find_package(rocblas)

    if (${hipblas_FOUND} AND ${hip_FOUND})
        message(STATUS "HIP and hipBLAS found")
        add_compile_definitions(GGML_USE_HIPBLAS GGML_USE_CUBLAS)
        add_library(ggml-rocm OBJECT ggml-cuda.cu ggml-cuda.h)
        if (BUILD_SHARED_LIBS)
            set_target_properties(ggml-rocm PROPERTIES POSITION_INDEPENDENT_CODE ON)
        endif()
        set_source_files_properties(ggml-cuda.cu PROPERTIES LANGUAGE CXX)
        target_link_libraries(ggml-rocm PRIVATE hip::device PUBLIC hip::host roc::rocblas roc::hipblas)

        if (LLAMA_STATIC)
            message(FATAL_ERROR "Static linking not supported for HIP/ROCm")
        endif()
        set(LLAMA_EXTRA_LIBS ${SD_EXTRA_LIBS} ggml-rocm)
    else()
        message(WARNING "hipBLAS or HIP not found. Try setting CMAKE_PREFIX_PATH=/opt/rocm")
    endif()
endif()


set(SD_LIB stable-diffusion)

add_library(${SD_LIB} stable-diffusion.h stable-diffusion.cpp)
target_link_libraries(${SD_LIB} PUBLIC ggml ${SD_EXTRA_LIBS})
target_include_directories(${SD_LIB} PUBLIC .)
target_compile_features(${SD_LIB} PUBLIC cxx_std_11)

add_subdirectory(common)

if (SD_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

