cmake_minimum_required(VERSION 3.12)
project("stable-diffusion")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

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

if (MINGW)
        set(BUILD_SHARED_LIBS_DEFAULT OFF)
    else()
        set(BUILD_SHARED_LIBS_DEFAULT ON)
endif()

option(BUILD_SHARED_LIBS "build shared libraries" ${BUILD_SHARED_LIBS_DEFAULT})

if (WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

if (MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:/utf-8>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/utf-8>")
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:/bigobj>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/bigobj>")
endif()

if (SD_STANDALONE)
    # enable parallel builds for msbuild
    list(APPEND CMAKE_VS_GLOBALS UseMultiToolTask=true)
    list(APPEND CMAKE_VS_GLOBALS EnforceProcessCountAcrossBuilds=true)
endif()


#
# Option list
#
# general
#option(SD_BUILD_TESTS                "sd: build tests"    ${SD_STANDALONE})
option(SD_BUILD_EXAMPLES             "sd: build examples" ${SD_STANDALONE})
option(SD_WEBP                       "sd: enable WebP image I/O support" ON)
option(SD_USE_SYSTEM_GGML            "sd: use system-installed GGML library" OFF)
#option(SD_BUILD_SERVER               "sd: build server example"                           ON)

# Required for relocatable CMake package
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/build-info.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/common.cmake)

if (NOT DEFINED SD_BUILD_NUMBER)
    set(SD_BUILD_NUMBER        ${BUILD_NUMBER})
endif()
if (NOT DEFINED SD_BUILD_COMMIT)
    set(SD_BUILD_COMMIT        ${BUILD_COMMIT})
endif()
set(SD_INSTALL_VERSION 0.0.${SD_BUILD_NUMBER})

# override ggml options
set(GGML_ALL_WARNINGS   ${SD_ALL_WARNINGS})
set(GGML_FATAL_WARNINGS ${SD_FATAL_WARNINGS})

if (NOT DEFINED GGML_CUDA_GRAPHS)
    set(GGML_CUDA_GRAPHS_DEFAULT ON)
endif()

# Ref: https://github.com/ggml-org/llama.cpp/blob/master/CMakeLists.txt#L145
# transition helpers
function (sd_option_depr TYPE OLD)
    if (${OLD})
        set(NEW "${ARGV2}")
        if(NEW)
            message(${TYPE} "${OLD} is deprecated, use ${NEW} instead")
            set(${NEW} ON PARENT_SCOPE)
        else()
            message(${TYPE} "${OLD} is deprecated and will be ignored")
        endif()
    endif()
endfunction()

sd_option_depr(FATAL_ERROR SD_HIPBLAS               GGML_HIP)
sd_option_depr(FATAL_ERROR SD_BUILD_SHARED_LIBS     BUILD_SHARED_LIBS)
sd_option_depr(FATAL_ERROR SD_BUILD_SHARED_GGML_LIB BUILD_SHARED_LIBS)
sd_option_depr(WARNING     SD_CUDA                  GGML_CUDA)
sd_option_depr(WARNING     SD_METAL                 GGML_METAL)
sd_option_depr(WARNING     SD_VULKAN                GGML_VULKAN)
sd_option_depr(WARNING     SD_OPENCL                GGML_OPENCL)
sd_option_depr(WARNING     SD_SYCL                  GGML_SYCL)
sd_option_depr(WARNING     SD_MUSA                  GGML_MUSA)
if(SD_WEBP)
    add_compile_definitions(SD_USE_WEBP)
endif()

set(SD_LIB stable-diffusion)

file(GLOB SD_LIB_SOURCES
    "src/*.h"
    "src/*.cpp"
    "src/*.hpp"
    "src/vocab/*.h"
    "src/vocab/*.cpp"
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
  SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/version.cpp
  APPEND PROPERTY COMPILE_DEFINITIONS
  SDCPP_BUILD_COMMIT=${SDCPP_BUILD_COMMIT} SDCPP_BUILD_VERSION=${SDCPP_BUILD_VERSION}
)

# Is this needed?
if(GGML_SYCL)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-narrowing -fsycl")
    # disable fast-math on host, see:
    # https://www.intel.com/content/www/us/en/docs/cpp-compiler/developer-guide-reference/2021-10/fp-model-fp.html
    if (WIN32)
        set(SYCL_COMPILE_OPTIONS /fp:precise)
    else()
        set(SYCL_COMPILE_OPTIONS -fp-model=precise)
    endif()
    message("-- Turn off fast-math for host in SYCL backend")
    list(APPEND SD_TARGET_PRIVATE_COMPILE_OPTIONS ${SYCL_COMPILE_OPTIONS})
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
add_subdirectory(src)

if (SD_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(SD_INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR} CACHE PATH "Location of header  files")
set(SD_LIB_INSTALL_DIR     ${CMAKE_INSTALL_LIBDIR}     CACHE PATH "Location of library files")
set(SD_BIN_INSTALL_DIR     ${CMAKE_INSTALL_BINDIR}     CACHE PATH "Location of binary  files")
set(SD_PUBLIC_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/stable-diffusion.h
)

set_target_properties(${SD_LIB}
    PROPERTIES
        PUBLIC_HEADER "${SD_PUBLIC_HEADERS}")

install(TARGETS ${SD_LIB}
    RUNTIME       DESTINATION ${SD_BIN_INSTALL_DIR}
    LIBRARY       DESTINATION ${SD_LIB_INSTALL_DIR}
    ARCHIVE       DESTINATION ${SD_LIB_INSTALL_DIR}
    PUBLIC_HEADER DESTINATION ${SD_INCLUDE_INSTALL_DIR})

configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/sd-config.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/sd-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sd
    PATH_VARS SD_INCLUDE_INSTALL_DIR
              SD_LIB_INSTALL_DIR
              SD_BIN_INSTALL_DIR )

write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/sd-version.cmake
    VERSION ${SD_INSTALL_VERSION}
    COMPATIBILITY SameMajorVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/sd-config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/sd-version.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sd)

configure_file(cmake/sd.pc.in
        "${CMAKE_CURRENT_BINARY_DIR}/sd.pc"
        @ONLY)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/sd.pc"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
