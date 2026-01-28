set(TARGET sd-server)

set(GENERATED_HTML_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/frontend/dist/gen_index_html.h")

if(EXISTS ${GENERATED_HTML_HEADER})
    message(STATUS "Found generated header: ${GENERATED_HTML_HEADER}")
    add_executable(${TARGET} main.cpp ${GENERATED_HTML_HEADER})
    target_compile_definitions(${TARGET} PRIVATE HAVE_INDEX_HTML)
else()
    message(WARNING "Header ${GENERATED_HTML_HEADER} not found. Skipping index_html inclusion.")
    add_executable(${TARGET} main.cpp)
endif()

install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE stable-diffusion ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PUBLIC c_std_11 cxx_std_17)