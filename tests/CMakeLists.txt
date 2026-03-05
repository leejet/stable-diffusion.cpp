set(TARGET test-binary-reader)

add_executable(${TARGET} test_binary_reader.cpp)
target_link_libraries(${TARGET} PRIVATE stable-diffusion)
target_include_directories(${TARGET} PRIVATE ..)
target_compile_features(${TARGET} PUBLIC c_std_11 cxx_std_17)

add_test(NAME ${TARGET} COMMAND ${TARGET})
