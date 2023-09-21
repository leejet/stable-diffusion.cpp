cmake_minimum_required(VERSION 3.12)
project("stable-diffusion")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif ()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(SD_STANDALONE ON)
else ()
    set(SD_STANDALONE OFF)
endif ()

#
# Option list
#

# general
#option(SD_BUILD_TESTS                "sd: build tests"    ${SD_STANDALONE})
option(SD_BUILD_EXAMPLES "sd: build examples" ${SD_STANDALONE})
#option(SD_BUILD_SERVER               "sd: build server example"                           ON)

# Instruction set specific
option(SD_AVX                    "sd: enable AVX"                                     ON)
option(SD_AVX2                   "sd: enable AVX2"                                    ON)
option(SD_AVX512                 "sd: enable AVX512"                                  OFF)
option(SD_FMA                    "sd: enable FMA"                                     ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

if (MSVC)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)

    if (BUILD_SHARED_LIBS)
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif ()
endif ()

if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
    message(STATUS "ARM detected")
    if (MSVC)
        # TODO [llama.cpp]: arm msvc?
    else()
        if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
            add_compile_options(-mcpu=native)
        endif()
        # TODO [llama.cpp]: armv6,7,8 version specific flags
    endif()
elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "^(x86_64|i686|AMD64)$")
    message(STATUS "x86 detected")
    if (MSVC)
        if (SD_AVX512)
            add_compile_options($<$<COMPILE_LANGUAGE:C>:/arch:AVX512>)
            add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/arch:AVX512>)
            # MSVC has no compile-time flags enabling specific
            # AVX512 extensions, neither it defines the
            # macros corresponding to the extensions.
            # Do it manually.
        elseif (SD_AVX2)
            add_compile_options($<$<COMPILE_LANGUAGE:C>:/arch:AVX2>)
            add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
        elseif (SD_AVX)
            add_compile_options($<$<COMPILE_LANGUAGE:C>:/arch:AVX>)
            add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/arch:AVX>)
        endif()
    else()
        add_compile_options(-mf16c)
        if (SD_FMA)
            add_compile_options(-mfma)
        endif()
        if (SD_AVX)
            add_compile_options(-mavx)
        endif()
        if (SD_AVX2)
            add_compile_options(-mavx2)
        endif()
        if (SD_AVX512)
            add_compile_options(-mavx512f)
            add_compile_options(-mavx512bw)
        endif()
    endif()
else()
    # TODO [llama.cpp]: support PowerPC
    message(STATUS "Unknown architecture")
endif()
# deps
#add_subdirectory(ggml)

add_library(ggml OBJECT
        ${CMAKE_SOURCE_DIR}/ggml/src/ggml.c
        ${CMAKE_SOURCE_DIR}/ggml/src/ggml-alloc.c
        ${CMAKE_SOURCE_DIR}/ggml/include/ggml/ggml.h
        ${CMAKE_SOURCE_DIR}/ggml/include/ggml/ggml-alloc.h)

target_include_directories(ggml PUBLIC ${CMAKE_SOURCE_DIR}/ggml/include/ggml)
target_compile_features(ggml PUBLIC c_std_11)
# Don't bump
if (MSVC)
    target_link_libraries(ggml PUBLIC ${SD_EXTRA_LIBS} Threads::Threads)
else()
    target_link_libraries(ggml PUBLIC m ${SD_EXTRA_LIBS} Threads::Threads)
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties(ggml PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

set(SD_LIB stable-diffusion)


if (BUILD_SHARED_LIBS)
    add_library(${SD_LIB} SHARED stable-diffusion.cpp stable-diffusion-c.cpp stable-diffusion.h stable-diffusion-c.h)
    set_target_properties(${SD_LIB} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(${SD_LIB} PRIVATE STABLE_DIFFUSION_SHARED STABLE_DIFFUSION_BUILD)
    target_link_libraries(${SD_LIB} PRIVATE ggml ${SD_EXTRA_LIBS} ${CMAKE_THREAD_LIBS_INIT})
    target_include_directories(${SD_LIB} PUBLIC .)
else ()
    add_library(${SD_LIB} stable-diffusion.h stable-diffusion.cpp)
    target_link_libraries(${SD_LIB} PUBLIC ggml ${SD_EXTRA_LIBS})
    target_include_directories(${SD_LIB} PUBLIC .)
endif ()

target_compile_features(${SD_LIB} PUBLIC cxx_std_11)

if (SD_BUILD_EXAMPLES AND NOT BUILD_SHARED_LIBS)
    add_subdirectory(examples)
endif ()