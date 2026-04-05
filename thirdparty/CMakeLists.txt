set(Z_TARGET zip)
add_library(${Z_TARGET} OBJECT zip.c zip.h miniz.h)
target_include_directories(${Z_TARGET} PUBLIC .)

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
