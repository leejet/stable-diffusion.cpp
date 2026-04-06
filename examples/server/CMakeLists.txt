set(TARGET sd-server)

option(SD_SERVER_BUILD_FRONTEND "Build server frontend with pnpm" ON)

set(FRONTEND_DIR "${CMAKE_CURRENT_SOURCE_DIR}/frontend")
set(GENERATED_HTML_HEADER "${FRONTEND_DIR}/dist/gen_index_html.h")

set(HAVE_FRONTEND_BUILD OFF)

if(SD_SERVER_BUILD_FRONTEND AND EXISTS "${FRONTEND_DIR}")
    if(WIN32)
        find_program(PNPM_EXECUTABLE NAMES pnpm.cmd pnpm)
    else()
        find_program(PNPM_EXECUTABLE NAMES pnpm)
    endif()

    if(PNPM_EXECUTABLE)
        message(STATUS "Frontend dir found: ${FRONTEND_DIR}")
        message(STATUS "pnpm found: ${PNPM_EXECUTABLE}")

        set(HAVE_FRONTEND_BUILD ON)

        add_custom_target(${TARGET}_frontend_install
            COMMAND "${PNPM_EXECUTABLE}" -C "${FRONTEND_DIR}" install
            WORKING_DIRECTORY "${FRONTEND_DIR}"
            COMMENT "Installing frontend dependencies"
            VERBATIM
        )

        add_custom_target(${TARGET}_frontend_build
            COMMAND "${PNPM_EXECUTABLE}" -C "${FRONTEND_DIR}" run build
            WORKING_DIRECTORY "${FRONTEND_DIR}"
            COMMENT "Building frontend"
            VERBATIM
        )

        add_custom_target(${TARGET}_frontend_header
            COMMAND "${PNPM_EXECUTABLE}" -C "${FRONTEND_DIR}" run build:header
            WORKING_DIRECTORY "${FRONTEND_DIR}"
            COMMENT "Generating gen_index_html.h"
            VERBATIM
        )

        add_dependencies(${TARGET}_frontend_build ${TARGET}_frontend_install)
        add_dependencies(${TARGET}_frontend_header ${TARGET}_frontend_build)

        add_custom_target(${TARGET}_frontend
            DEPENDS ${TARGET}_frontend_header
        )

        set_source_files_properties("${GENERATED_HTML_HEADER}" PROPERTIES GENERATED TRUE)
    else()
        message(WARNING "pnpm not found, frontend build disabled")
    endif()
else()
    message(STATUS "Frontend disabled or directory not found: ${FRONTEND_DIR}")
endif()

add_executable(${TARGET}
    ../common/common.cpp
    ../common/log.cpp
    ../common/media_io.cpp
    main.cpp
)

if(HAVE_FRONTEND_BUILD)
    add_dependencies(${TARGET} ${TARGET}_frontend)
    target_sources(${TARGET} PRIVATE "${GENERATED_HTML_HEADER}")
    target_include_directories(${TARGET} PRIVATE "${FRONTEND_DIR}/dist")
    target_compile_definitions(${TARGET} PRIVATE HAVE_INDEX_HTML)
    message(STATUS "HAVE_INDEX_HTML enabled")
else()
    message(STATUS "HAVE_INDEX_HTML disabled")
endif()

install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE stable-diffusion ${CMAKE_THREAD_LIBS_INIT})
if(SD_WEBP)
    target_compile_definitions(${TARGET} PRIVATE SD_USE_WEBP)
    target_link_libraries(${TARGET} PRIVATE webp libwebpmux)
endif()
if(SD_WEBM)
    target_compile_definitions(${TARGET} PRIVATE SD_USE_WEBM)
    target_link_libraries(${TARGET} PRIVATE webm)
endif()

# due to httplib; it contains a pragma for MSVC, but other things need explicit flags
if(WIN32 AND NOT MSVC)
    target_link_libraries(${TARGET} PRIVATE ws2_32)
endif()

target_compile_features(${TARGET} PUBLIC c_std_11 cxx_std_17)
