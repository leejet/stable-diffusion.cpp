set(TARGET convert)

add_executable(${TARGET} convert.cpp vocab.hpp)
target_link_libraries(${TARGET} PRIVATE ggml zip ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PUBLIC cxx_std_11)
