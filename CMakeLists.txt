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
    add_compile_options(
        $<$<COMPILE_LANGUAGE:C>:/MP>
        $<$<COMPILE_LANGUAGE:CXX>:/MP>
    )
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

set(SD_SUBMODULE_WEBP FALSE)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/libwebp/CMakeLists.txt")
    set(SD_SUBMODULE_WEBP TRUE)
endif()
if(SD_SUBMODULE_WEBP)
    set(SD_WEBP_DEFAULT ON)
else()
    set(SD_WEBP_DEFAULT ${SD_USE_SYSTEM_WEBP})
endif()

set(SD_SUBMODULE_WEBM FALSE)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/libwebm/CMakeLists.txt")
    set(SD_SUBMODULE_WEBM TRUE)
endif()
if(SD_SUBMODULE_WEBM)
    set(SD_WEBM_DEFAULT ON)
else()
    set(SD_WEBM_DEFAULT ${SD_USE_SYSTEM_WEBM})
endif()

#
# Option list
#
# general
#option(SD_BUILD_TESTS                "sd: build tests"    ${SD_STANDALONE})
option(SD_BUILD_EXAMPLES             "sd: build examples" ${SD_STANDALONE})
option(SD_WEBP                       "sd: enable WebP image I/O support" ${SD_WEBP_DEFAULT})
option(SD_USE_SYSTEM_WEBP            "sd: link against system libwebp" OFF)
option(SD_WEBM                       "sd: enable WebM video output support" ${SD_WEBM_DEFAULT})
option(SD_USE_SYSTEM_WEBM            "sd: link against system libwebm" OFF)
option(SD_CUDA                       "sd: cuda backend" OFF)
option(SD_HIPBLAS                    "sd: rocm backend" OFF)
option(SD_METAL                      "sd: metal backend" OFF)
option(SD_VULKAN                     "sd: vulkan backend" OFF)
option(SD_OPENCL                     "sd: opencl backend" OFF)
option(SD_SYCL                       "sd: sycl backend" OFF)
option(SD_MUSA                       "sd: musa backend" OFF)
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

if(SD_CUDA)
    set(GGML_CUDA ON)
    add_definitions(-DSD_USE_CUDA)
endif()

if(SD_HIPBLAS)
    set(GGML_HIP ON)
    add_definitions(-DSD_USE_CUDA)
endif()

if(SD_METAL)
    set(GGML_METAL ON)
    add_definitions(-DSD_USE_METAL)
endif()

if(SD_VULKAN)
    set(GGML_VULKAN ON)
    add_definitions(-DSD_USE_VULKAN)
endif()

if(SD_OPENCL)
    set(GGML_OPENCL ON)
    add_definitions(-DSD_USE_OPENCL)
endif()

if(SD_SYCL)
    set(GGML_SYCL ON)
    add_definitions(-DSD_USE_SYCL)
endif()

if(SD_MUSA)
    set(GGML_MUSA ON)
    add_definitions(-DSD_USE_CUDA)
endif()

if(SD_WEBP)
    if(NOT SD_SUBMODULE_WEBP AND NOT SD_USE_SYSTEM_WEBP)
        message(FATAL_ERROR "WebP support enabled but no source found.
          Either initialize the submodule:\n  git submodule update --init thirdparty/libwebp\n\n"
          "Or link against system library:\n  cmake (...) -DSD_USE_SYSTEM_WEBP=ON")
    endif()
    if(SD_USE_SYSTEM_WEBP)
        find_package(WebP REQUIRED)
        add_library(webp ALIAS WebP::webp)
        # libwebp CMake target naming is not consistent across versions/distros.
        # Some export WebP::libwebpmux, others export WebP::webpmux.
        if(TARGET WebP::libwebpmux)
            add_library(libwebpmux ALIAS WebP::libwebpmux)
        elseif(TARGET WebP::webpmux)
            add_library(libwebpmux ALIAS WebP::webpmux)
        else()
            message(FATAL_ERROR
                "Could not find a compatible webpmux target in system WebP package. "
                "Expected WebP::libwebpmux or WebP::webpmux."
            )
        endif()
    endif()
endif()

if(SD_WEBM)
    if(NOT SD_WEBP)
        message(FATAL_ERROR "SD_WEBM requires SD_WEBP because WebM output reuses libwebp VP8 encoding.")
    endif()
    if(NOT SD_SUBMODULE_WEBM AND NOT SD_USE_SYSTEM_WEBM)
        message(FATAL_ERROR "WebM support enabled but no source found.
          Either initialize the submodule:\n  git submodule update --init thirdparty/libwebm\n\n"
          "Or link against system library:\n  cmake (...) -DSD_USE_SYSTEM_WEBM=ON")
    endif()
    if(SD_USE_SYSTEM_WEBM)
        find_path(WEBM_INCLUDE_DIR
            NAMES mkvmuxer/mkvmuxer.h mkvparser/mkvparser.h common/webmids.h
            PATH_SUFFIXES webm
            REQUIRED)
        find_library(WEBM_LIBRARY
            NAMES webm libwebm
            REQUIRED)

        add_library(webm UNKNOWN IMPORTED)
        set_target_properties(webm PROPERTIES
            IMPORTED_LOCATION "${WEBM_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${WEBM_INCLUDE_DIR}")
    endif()
endif()

set(SD_LIB stable-diffusion)

file(GLOB SD_LIB_SOURCES CONFIGURE_DEPENDS
    "src/*.h"
    "src/*.cpp"
    "src/*.hpp"
    "src/model_io/*.h"
    "src/model_io/*.cpp"
    "src/tokenizers/*.h"
    "src/tokenizers/*.cpp"
    "src/tokenizers/vocab/*.h"
    "src/tokenizers/vocab/*.cpp"
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
