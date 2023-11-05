set(TARGET common)

add_library(${TARGET} OBJECT common.cpp common.h stb_image.h stb_image_write.h)

target_include_directories(${TARGET} PUBLIC .)
target_link_libraries(${TARGET} PRIVATE stable-diffusion ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PUBLIC cxx_std_11)

set(Z_TARGET zip)

add_library(${Z_TARGET} OBJECT zip.c zip.h miniz.h)

target_include_directories(${Z_TARGET} PUBLIC .)
target_link_libraries(${Z_TARGET} PRIVATE stable-diffusion ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${Z_TARGET} PUBLIC cxx_std_11)