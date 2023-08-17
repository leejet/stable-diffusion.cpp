# TODO: move into its own subdirectoy
# TODO: make stb libs a target (maybe common)
set(SD_TARGET sd)

add_executable(${SD_TARGET} main.cpp stb_image.h stb_image_write.h)
install(TARGETS ${SD_TARGET} RUNTIME)
target_link_libraries(${SD_TARGET} PRIVATE stable-diffusion ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${SD_TARGET} PUBLIC cxx_std_11)
