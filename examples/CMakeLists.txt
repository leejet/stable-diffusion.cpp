sd_add_compile_flags()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/thirdparty)

add_subdirectory(cli)
add_subdirectory(server)
