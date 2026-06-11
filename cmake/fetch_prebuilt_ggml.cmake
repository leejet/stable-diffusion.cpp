include_guard(GLOBAL)

include(CMakeParseArguments)

function(_ggml_prebuilt_platform out_platform out_archive_ext)
    string(TOLOWER "${CMAKE_SYSTEM_NAME}" _ggml_system_name)
    string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _ggml_system_processor)

    if (_ggml_system_name STREQUAL "linux" AND _ggml_system_processor MATCHES "^(x86_64|amd64)$")
        set(_ggml_platform "linux-x86_64")
        set(_ggml_archive_ext ".tar.gz")
    elseif (_ggml_system_name STREQUAL "darwin" AND _ggml_system_processor MATCHES "^(arm64|aarch64)$")
        set(_ggml_platform "macos-arm64")
        set(_ggml_archive_ext ".tar.gz")
    elseif (WIN32 AND _ggml_system_processor MATCHES "^(x86_64|amd64)$")
        set(_ggml_platform "windows-x86_64")
        set(_ggml_archive_ext ".zip")
    else()
        message(FATAL_ERROR
            "Unsupported prebuilt ggml package for ${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}. "
            "Provide ROOT_DIR explicitly or extend cmake/fetch_prebuilt_ggml.cmake.")
    endif()

    set(${out_platform} "${_ggml_platform}" PARENT_SCOPE)
    set(${out_archive_ext} "${_ggml_archive_ext}" PARENT_SCOPE)
endfunction()

function(_ggml_locate_package_root out_root_dir search_root)
    if (NOT EXISTS "${search_root}")
        set(${out_root_dir} "" PARENT_SCOPE)
        return()
    endif()

    set(_ggml_package_root "")
    set(_ggml_search_roots "${search_root}")

    # Some archives unpack into an extra top-level directory such as
    # ggml-windows-x86_64/..., even when the archive file itself includes a tag.
    file(GLOB _ggml_child_entries
        LIST_DIRECTORIES true
        RELATIVE "${search_root}"
        "${search_root}/*")
    set(_ggml_child_dirs "")
    foreach(_ggml_child_entry IN LISTS _ggml_child_entries)
        if (IS_DIRECTORY "${search_root}/${_ggml_child_entry}")
            list(APPEND _ggml_child_dirs "${search_root}/${_ggml_child_entry}")
        endif()
    endforeach()
    list(LENGTH _ggml_child_dirs _ggml_child_dir_count)
    if (_ggml_child_dir_count EQUAL 1)
        list(APPEND _ggml_search_roots ${_ggml_child_dirs})
    endif()

    foreach(_ggml_search_dir IN LISTS _ggml_search_roots)
        file(GLOB_RECURSE _ggml_config_candidates
            LIST_DIRECTORIES false
            "${_ggml_search_dir}/ggml-config.cmake")

        foreach(_ggml_config IN LISTS _ggml_config_candidates)
            if (_ggml_config MATCHES [[[/\\]lib[/\\]cmake[/\\]ggml[/\\]ggml-config\.cmake$]])
                get_filename_component(_ggml_config_dir "${_ggml_config}" DIRECTORY)
                get_filename_component(_ggml_package_root "${_ggml_config_dir}/../../.." ABSOLUTE)
                break()
            endif()
        endforeach()

        if (_ggml_package_root)
            break()
        endif()
    endforeach()

    set(${out_root_dir} "${_ggml_package_root}" PARENT_SCOPE)
endfunction()

function(_ggml_archive_stem out_stem archive_name)
    get_filename_component(_ggml_archive_name "${archive_name}" NAME)
    string(REGEX REPLACE "(\\.tar\\.gz|\\.zip)$" "" _ggml_archive_stem "${_ggml_archive_name}")
    set(${out_stem} "${_ggml_archive_stem}" PARENT_SCOPE)
endfunction()

function(_ggml_download_archive download_url archive_path)
    set(_ggml_temp_archive_path "${archive_path}.tmp")
    file(REMOVE "${_ggml_temp_archive_path}")

    message(STATUS "Downloading prebuilt ggml package from ${download_url}")
    file(DOWNLOAD
        "${download_url}"
        "${_ggml_temp_archive_path}"
        SHOW_PROGRESS
        STATUS _ggml_download_status
        LOG _ggml_download_log
        TLS_VERIFY ON)

    list(GET _ggml_download_status 0 _ggml_download_code)
    list(GET _ggml_download_status 1 _ggml_download_message)
    if (NOT _ggml_download_code EQUAL 0)
        file(REMOVE "${_ggml_temp_archive_path}")
        message(FATAL_ERROR
            "Failed to download ${download_url}: ${_ggml_download_message}\n${_ggml_download_log}")
    endif()

    if (NOT EXISTS "${_ggml_temp_archive_path}")
        message(FATAL_ERROR
            "Download completed but did not create ${_ggml_temp_archive_path}.")
    endif()

    file(SIZE "${_ggml_temp_archive_path}" _ggml_downloaded_size)
    if (_ggml_downloaded_size EQUAL 0)
        file(REMOVE "${_ggml_temp_archive_path}")
        message(FATAL_ERROR
            "Downloaded ${download_url} but received an empty archive. "
            "Check GGML_RELEASE_TAG/GGML_ASSET_NAME/GGML_DOWNLOAD_URL and retry.")
    endif()

    file(REMOVE "${archive_path}")
    file(RENAME "${_ggml_temp_archive_path}" "${archive_path}")
endfunction()

function(ggml_import_prebuilt)
    set(options)
    set(one_value_args
        ROOT_DIR
        OUT_ROOT_DIR
        RELEASE_BASE_URL
        RELEASE_TAG
        VERSION
        ASSET_NAME
        DOWNLOAD_URL
        CACHE_DIR)
    cmake_parse_arguments(GGML "${options}" "${one_value_args}" "" ${ARGN})

    if (GGML_RELEASE_TAG AND GGML_VERSION)
        message(FATAL_ERROR "ggml_import_prebuilt accepts RELEASE_TAG or VERSION, but not both.")
    endif()

    if (EXISTS "${GGML_ROOT_DIR}")
        get_filename_component(_ggml_input_root "${GGML_ROOT_DIR}" ABSOLUTE)
        _ggml_locate_package_root(_ggml_package_root "${_ggml_input_root}")

        if (NOT _ggml_package_root)
            message(FATAL_ERROR
                "Could not locate lib/cmake/ggml/ggml-config.cmake under ROOT_DIR=${_ggml_input_root}.")
        endif()
    else()
        _ggml_prebuilt_platform(_ggml_platform _ggml_archive_ext)

        if (GGML_DOWNLOAD_URL)
            set(_ggml_download_url "${GGML_DOWNLOAD_URL}")
        else()
            if (NOT GGML_RELEASE_BASE_URL)
                message(FATAL_ERROR
                    "ggml_import_prebuilt requires ROOT_DIR, DOWNLOAD_URL, or RELEASE_BASE_URL + RELEASE_TAG.")
            endif()

            if (GGML_RELEASE_TAG)
                set(_ggml_release_tag "${GGML_RELEASE_TAG}")
            elseif (GGML_VERSION)
                set(_ggml_release_tag "${GGML_VERSION}")
            else()
                message(FATAL_ERROR
                    "ggml_import_prebuilt requires RELEASE_TAG or VERSION when RELEASE_BASE_URL is used.")
            endif()

            string(REGEX REPLACE "/$" "" _ggml_release_base_url "${GGML_RELEASE_BASE_URL}")
        endif()

        if (GGML_ASSET_NAME)
            set(_ggml_asset_name "${GGML_ASSET_NAME}")
        elseif (GGML_DOWNLOAD_URL)
            string(REGEX REPLACE "[?#].*$" "" _ggml_download_url_path "${_ggml_download_url}")
            get_filename_component(_ggml_asset_name "${_ggml_download_url_path}" NAME)
        elseif (DEFINED _ggml_release_tag)
            set(_ggml_asset_name "ggml-${_ggml_release_tag}-${_ggml_platform}${_ggml_archive_ext}")
        else()
            set(_ggml_asset_name "ggml-${_ggml_platform}${_ggml_archive_ext}")
        endif()

        if (NOT GGML_DOWNLOAD_URL)
            set(_ggml_download_url "${_ggml_release_base_url}/${_ggml_release_tag}/${_ggml_asset_name}")
        endif()

        if (GGML_CACHE_DIR)
            get_filename_component(_ggml_cache_dir "${GGML_CACHE_DIR}" ABSOLUTE)
        else()
            set(_ggml_cache_dir "${CMAKE_BINARY_DIR}/_deps/ggml-prebuilt")
        endif()

        set(_ggml_archive_dir "${_ggml_cache_dir}/archives")
        set(_ggml_archive_path "${_ggml_archive_dir}/${_ggml_asset_name}")
        _ggml_archive_stem(_ggml_extract_dir_name "${_ggml_asset_name}")
        set(_ggml_extract_dir "${_ggml_cache_dir}/packages/${_ggml_extract_dir_name}")

        file(MAKE_DIRECTORY "${_ggml_archive_dir}")

        set(_ggml_use_cached_archive FALSE)
        if (EXISTS "${_ggml_archive_path}")
            file(SIZE "${_ggml_archive_path}" _ggml_archive_size)
            if (_ggml_archive_size GREATER 0)
                set(_ggml_use_cached_archive TRUE)
                message(STATUS "Using cached prebuilt ggml archive: ${_ggml_archive_path}")
            else()
                message(STATUS "Removing empty cached prebuilt ggml archive: ${_ggml_archive_path}")
                file(REMOVE "${_ggml_archive_path}")
            endif()
        endif()

        if (NOT _ggml_use_cached_archive)
            _ggml_download_archive("${_ggml_download_url}" "${_ggml_archive_path}")
        endif()

        _ggml_locate_package_root(_ggml_package_root "${_ggml_extract_dir}")
        if (NOT _ggml_package_root)
            foreach(_ggml_extract_attempt RANGE 1 2)
                file(REMOVE_RECURSE "${_ggml_extract_dir}")
                file(MAKE_DIRECTORY "${_ggml_extract_dir}")

                if (_ggml_extract_attempt EQUAL 1)
                    message(STATUS "Extracting prebuilt ggml package to ${_ggml_extract_dir}")
                else()
                    message(STATUS "Retrying ggml extraction with a refreshed archive")
                endif()

                file(ARCHIVE_EXTRACT
                    INPUT "${_ggml_archive_path}"
                    DESTINATION "${_ggml_extract_dir}")

                _ggml_locate_package_root(_ggml_package_root "${_ggml_extract_dir}")
                if (_ggml_package_root)
                    break()
                endif()

                if (_ggml_extract_attempt EQUAL 1)
                    message(STATUS
                        "Archive ${_ggml_archive_path} did not produce a usable ggml package layout; refreshing cache.")
                    file(REMOVE "${_ggml_archive_path}")
                    _ggml_download_archive("${_ggml_download_url}" "${_ggml_archive_path}")
                endif()
            endforeach()
        endif()

        if (NOT _ggml_package_root)
            message(FATAL_ERROR
                "Archive ${_ggml_archive_path} did not produce lib/cmake/ggml/ggml-config.cmake under "
                "${_ggml_extract_dir}. Check GGML_RELEASE_TAG/GGML_ASSET_NAME/GGML_DOWNLOAD_URL.")
        endif()
    endif()

    if (NOT GGML_OUT_ROOT_DIR)
        set(GGML_OUT_ROOT_DIR GGML_ROOT_DIR)
    endif()

    find_package(ggml CONFIG REQUIRED
        PATHS "${_ggml_package_root}"
        NO_DEFAULT_PATH)

    # Some prebuilt ggml packages do not export public include directories or
    # the GGML_BACKEND_DL usage requirement on ggml::ggml-base. Patch the
    # imported targets locally so downstream targets (for example llama) can
    # resolve #include "ggml.h" correctly and inherit the backend loader define.
    if (TARGET ggml::ggml)
        get_target_property(_ggml_interface_includes ggml::ggml INTERFACE_INCLUDE_DIRECTORIES)
        if (NOT _ggml_interface_includes)
            set_target_properties(ggml::ggml PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${_ggml_package_root}/include")
        endif()
    endif()

    if (TARGET ggml::ggml-base)
        get_target_property(_ggml_base_interface_includes ggml::ggml-base INTERFACE_INCLUDE_DIRECTORIES)
        if (NOT _ggml_base_interface_includes)
            set_target_properties(ggml::ggml-base PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${_ggml_package_root}/include")
        endif()

        if (GGML_BACKEND_DL)
            get_target_property(_ggml_base_interface_compile_definitions ggml::ggml-base INTERFACE_COMPILE_DEFINITIONS)
            if (_ggml_base_interface_compile_definitions MATCHES "-NOTFOUND$")
                set(_ggml_base_interface_compile_definitions "")
            endif()

            if (NOT _ggml_base_interface_compile_definitions MATCHES "GGML_BACKEND_DL")
                list(APPEND _ggml_base_interface_compile_definitions GGML_BACKEND_DL)
                set_target_properties(ggml::ggml-base PROPERTIES
                    INTERFACE_COMPILE_DEFINITIONS "${_ggml_base_interface_compile_definitions}")
            endif()
        endif()
    endif()

    set(GGML_INCLUDE_DIR "${_ggml_package_root}/include" PARENT_SCOPE)
    set(GGML_LIB_DIR "${_ggml_package_root}/lib" PARENT_SCOPE)
    set(GGML_BIN_DIR "${_ggml_package_root}/bin" PARENT_SCOPE)
    set(${GGML_OUT_ROOT_DIR} "${_ggml_package_root}" PARENT_SCOPE)
    set(GGML_PREBUILT_ROOT_DIR "${_ggml_package_root}" PARENT_SCOPE)
endfunction()

function(ggml_copy_runtime_binaries target_name)
    set(options)
    set(one_value_args ROOT_DIR)
    cmake_parse_arguments(GGML "${options}" "${one_value_args}" "" ${ARGN})

    if (NOT TARGET "${target_name}")
        message(FATAL_ERROR "ggml_copy_runtime_binaries expected an existing target, got '${target_name}'.")
    endif()

    if (NOT GGML_ROOT_DIR)
        message(FATAL_ERROR "ggml_copy_runtime_binaries requires ROOT_DIR.")
    endif()

    get_filename_component(_ggml_runtime_root "${GGML_ROOT_DIR}" ABSOLUTE)
    set(_ggml_runtime_bin_dir "${_ggml_runtime_root}/bin")
    set(_ggml_runtime_lib_dir "${_ggml_runtime_root}/lib")
    set(_ggml_runtime_files "")

    if (EXISTS "${_ggml_runtime_lib_dir}")
        file(GLOB _ggml_runtime_lib_files
            LIST_DIRECTORIES false
            "${_ggml_runtime_lib_dir}/*.dll"
            "${_ggml_runtime_lib_dir}/*.so"
            "${_ggml_runtime_lib_dir}/*.so.*"
            "${_ggml_runtime_lib_dir}/*.dylib"
            "${_ggml_runtime_lib_dir}/*.dylib.*")
        list(APPEND _ggml_runtime_files ${_ggml_runtime_lib_files})
    endif()

    if (EXISTS "${_ggml_runtime_bin_dir}")
        add_custom_command(TARGET "${target_name}" POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${_ggml_runtime_bin_dir}"
                "$<TARGET_FILE_DIR:${target_name}>"
            COMMENT "Copying ggml runtime binaries to build directory...")
    elseif(NOT _ggml_runtime_files)
        message(STATUS "No ggml runtime files found under ${_ggml_runtime_root}; skipping copy step.")
    endif()

    if (_ggml_runtime_files)
        add_custom_command(TARGET "${target_name}" POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${_ggml_runtime_files}
                "$<TARGET_FILE_DIR:${target_name}>"
            COMMENT "Copying ggml runtime libraries to build directory...")
    endif()
endfunction()
