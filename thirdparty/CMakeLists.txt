set(Z_TARGET zip)
add_library(${Z_TARGET} OBJECT zip.c zip.h miniz.h)
target_include_directories(${Z_TARGET} PUBLIC .)

if (BUILD_SHARED_LIBS)
    set_target_properties(${Z_TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
