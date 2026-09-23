set(Z_TARGET zip)
add_library(${Z_TARGET} OBJECT zip.c zip.h miniz.h)
target_include_directories(${Z_TARGET} PUBLIC .)

function(sd_add_oniguruma)
    include(CheckIncludeFiles)
    include(CheckSymbolExists)
    include(CheckTypeSize)
    include(GNUInstallDirs)

    foreach(header IN ITEMS alloca.h stdint.h sys/times.h sys/time.h sys/types.h unistd.h inttypes.h)
        string(TOUPPER "${header}" name)
        string(REGEX REPLACE "[./]" "_" name "${name}")
        check_include_files("${header}" SD_ONIG_HAVE_${name})
        set(HAVE_${name} "${SD_ONIG_HAVE_${name}}")
    endforeach()
    check_type_size(int SD_ONIG_SIZEOF_INT)
    check_type_size(long SD_ONIG_SIZEOF_LONG)
    check_type_size("long long" SD_ONIG_SIZEOF_LONG_LONG)
    check_type_size("void*" SD_ONIG_SIZEOF_VOIDP)
    foreach(name IN ITEMS SIZEOF_INT SIZEOF_LONG SIZEOF_LONG_LONG SIZEOF_VOIDP)
        set(${name} "${SD_ONIG_${name}}")
    endforeach()
    if(HAVE_ALLOCA_H)
        check_symbol_exists(alloca "alloca.h" SD_ONIG_HAVE_ALLOCA)
    else()
        check_symbol_exists(alloca "stdlib.h;malloc.h" SD_ONIG_HAVE_ALLOCA)
    endif()
    set(HAVE_ALLOCA "${SD_ONIG_HAVE_ALLOCA}")
    set(PACKAGE onig)
    set(PACKAGE_VERSION 6.9.10)
    set(VERSION "${PACKAGE_VERSION}")
    set(USE_CRNL_AS_LINE_TERMINATOR 0)
    configure_file(oniguruma/config.h.cmake.in oniguruma/config.h)

    # unicode.c includes its four data tables; they are not separate translation units.
    add_library(onig OBJECT
        oniguruma/ascii.c
        oniguruma/regcomp.c
        oniguruma/regenc.c
        oniguruma/regerror.c
        oniguruma/regexec.c
        oniguruma/regparse.c
        oniguruma/st.c
        oniguruma/unicode.c
        oniguruma/unicode_fold1_key.c
        oniguruma/unicode_fold2_key.c
        oniguruma/unicode_fold3_key.c
        oniguruma/unicode_unfold_key.c
        oniguruma/utf8.c)
    target_include_directories(onig PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/oniguruma"
                                   PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/oniguruma")
    target_compile_definitions(onig PUBLIC ONIG_STATIC)
    set_target_properties(onig PROPERTIES POSITION_INDEPENDENT_CODE ON)

    install(FILES oniguruma/COPYING
            DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/stable-diffusion/oniguruma")
endfunction()

sd_add_oniguruma()

add_library(sd-utf8proc OBJECT utf8proc/utf8proc.c)
target_include_directories(sd-utf8proc PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/utf8proc")
target_compile_definitions(sd-utf8proc PUBLIC UTF8PROC_STATIC)
set_target_properties(sd-utf8proc PROPERTIES POSITION_INDEPENDENT_CODE ON)
include(GNUInstallDirs)
install(FILES utf8proc/LICENSE.md DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/stable-diffusion/utf8proc")

if(SD_WEBP AND NOT SD_USE_SYSTEM_WEBP)
    set(WEBP_BUILD_ANIM_UTILS OFF)
    set(WEBP_BUILD_CWEBP OFF)
    set(WEBP_BUILD_DWEBP OFF)
    set(WEBP_BUILD_GIF2WEBP OFF)
    set(WEBP_BUILD_IMG2WEBP OFF)
    set(WEBP_BUILD_VWEBP OFF)
    set(WEBP_BUILD_WEBPINFO OFF)
    set(WEBP_BUILD_WEBPMUX OFF)
    set(WEBP_BUILD_EXTRAS OFF)
    set(WEBP_BUILD_WEBP_JS OFF)
    set(WEBP_BUILD_FUZZTEST OFF)
    set(WEBP_BUILD_LIBWEBPMUX ON)

    add_subdirectory(libwebp EXCLUDE_FROM_ALL)
endif()

if(SD_WEBM AND NOT SD_USE_SYSTEM_WEBM)
    if(MSVC)
        set(MSVC_RUNTIME dll)
    endif()
    set(ENABLE_WEBMTS OFF)
    set(ENABLE_WEBMINFO OFF)
    set(ENABLE_TESTS OFF)
    set(ENABLE_WEBM_PARSER OFF)
    set(ENABLE_SAMPLE_PROGRAMS OFF)

    set(SD_LIBWEBM_PARENT_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

    add_subdirectory(libwebm EXCLUDE_FROM_ALL)

    # libwebm mutates the global CMAKE_CXX_FLAGS for non-MSVC compilers to force
    # C++11. Restore the parent flags so the main project keeps its own C++17
    # requirements, then pin the libwebm targets to C++17 explicitly.
    set(CMAKE_CXX_FLAGS "${SD_LIBWEBM_PARENT_CXX_FLAGS}" CACHE STRING "" FORCE)
    target_compile_features(mkvmuxer PRIVATE cxx_std_17)
    target_compile_features(mkvparser PRIVATE cxx_std_17)
    target_compile_features(webm PRIVATE cxx_std_17)

    target_include_directories(webm INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/libwebm)
endif()
