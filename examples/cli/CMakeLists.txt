set(TARGET sd-cli)

add_executable(${TARGET}
    ../common/log.cpp
    ../common/media_io.cpp
    image_metadata.cpp
    main.cpp
)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE stable-diffusion zip ${CMAKE_THREAD_LIBS_INIT})
if(SD_WEBP)
    target_compile_definitions(${TARGET} PRIVATE SD_USE_WEBP)
    target_link_libraries(${TARGET} PRIVATE webp libwebpmux)
endif()
if(SD_WEBM)
    target_compile_definitions(${TARGET} PRIVATE SD_USE_WEBM)
    target_link_libraries(${TARGET} PRIVATE webm)
endif()
target_compile_features(${TARGET} PUBLIC c_std_11 cxx_std_17)
