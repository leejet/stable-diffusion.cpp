sd_add_compile_flags()

#
# libraries
#

# stable-diffusion
file(GLOB SD_SOURCES "*.cpp" "*.hpp" "*.h" "*.hpp")
file(GLOB SD_VOCAB_SOURCES "vocab/*.h" "vocab/*.cpp")

add_library(${SD_LIB}
            ../include/stable-diffusion.h
            ${SD_SOURCES}
            ${SD_VOCAB_SOURCES}
)

set_target_properties(${SD_LIB} PROPERTIES
    VERSION ${SD_INSTALL_VERSION}
    SOVERSION 0
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
)



target_include_directories(${SD_LIB} PRIVATE .)
target_include_directories(${SD_LIB} PUBLIC ../include)
target_compile_features(${SD_LIB} PRIVATE c_std_11 cxx_std_17)

target_link_libraries(${SD_LIB} PUBLIC ggml PRIVATE zip)

if (BUILD_SHARED_LIBS)
    set_target_properties(${SD_LIB} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(${SD_LIB} PRIVATE SD_BUILD_DLL)
    target_compile_definitions(${SD_LIB} PUBLIC  SD_BUILD_SHARED_LIB )
endif()
