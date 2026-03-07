include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(cli)
add_subdirectory(server)

# --- Streaming API Extensions ---
add_subdirectory(stream_img2img)
# --------------------------------