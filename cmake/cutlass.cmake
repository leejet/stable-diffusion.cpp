set(sd_cutlass_default "${PROJECT_SOURCE_DIR}/thirdparty/cutlass")
if(NOT DEFINED SD_CUDA_CUTLASS_PATH AND DEFINED GGML_CUDA_CUTLASS_PATH)
    get_filename_component(sd_old_cutlass_default "${SD_GGML_SOURCE_DIR}/src/ggml-cuda/vendors/cutlass" ABSOLUTE)
    if(NOT GGML_CUDA_CUTLASS_PATH STREQUAL sd_old_cutlass_default)
        set(sd_cutlass_default "${GGML_CUDA_CUTLASS_PATH}")
    endif()
    message(STATUS "Migrating GGML_CUDA_CUTLASS_PATH to SD_CUDA_CUTLASS_PATH")
endif()
set(SD_CUDA_CUTLASS_PATH "${sd_cutlass_default}" CACHE PATH "sd: CUTLASS source directory for INT8 GEMM and CuTe attention, empty to disable")
unset(GGML_CUDA_CUTLASS_PATH CACHE)
if(SD_CUDA_CUTLASS_PATH)
    if(NOT EXISTS "${SD_CUDA_CUTLASS_PATH}/include/cutlass/gemm/kernel/default_gemm_universal_with_visitor.h" OR
       NOT EXISTS "${SD_CUDA_CUTLASS_PATH}/include/cute/tensor.hpp")
        message(FATAL_ERROR
            "CUTLASS headers not found at ${SD_CUDA_CUTLASS_PATH}. "
            "Run git submodule update --init --recursive from the sd.cpp repository root, "
            "set SD_CUDA_CUTLASS_PATH to a CUTLASS source tree (tested with 3.9.2), "
            "or set SD_CUDA_CUTLASS_PATH to an empty string to disable CUTLASS.")
    endif()
endif()
