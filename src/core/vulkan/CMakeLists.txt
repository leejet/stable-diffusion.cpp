find_package(Vulkan COMPONENTS glslc REQUIRED)
set(SD_VULKAN_GENERATED "${CMAKE_CURRENT_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY "${SD_VULKAN_GENERATED}")

function(sd_vulkan_feature NAME EXTENSION)
    set(SOURCE "${SD_VULKAN_GENERATED}/feature_${NAME}.comp")
    set(EXTENSIONS "#extension ${EXTENSION} : require")
    if(NAME STREQUAL "SD_VK_CM2")
        string(PREPEND EXTENSIONS "#extension GL_KHR_cooperative_matrix : require\n")
    endif()
    file(WRITE "${SOURCE}" "#version 460\n${EXTENSIONS}\nlayout(local_size_x = 1) in;\nvoid main() {}\n")
    execute_process(COMMAND "${Vulkan_GLSLC_EXECUTABLE}" --target-env=vulkan1.3 "${SOURCE}"
                    -o "${SD_VULKAN_GENERATED}/feature_${NAME}.spv"
                    RESULT_VARIABLE RESULT OUTPUT_QUIET ERROR_QUIET)
    if(RESULT EQUAL 0)
        set(${NAME} ON PARENT_SCOPE)
    else()
        set(${NAME} OFF PARENT_SCOPE)
        message(STATUS "sd.cpp Vulkan: ${EXTENSION} unavailable in glslc")
    endif()
endfunction()

sd_vulkan_feature(SD_VK_DOT GL_EXT_integer_dot_product)
sd_vulkan_feature(SD_VK_CM1 GL_KHR_cooperative_matrix)
sd_vulkan_feature(SD_VK_CM2 GL_NV_cooperative_matrix2)

function(sd_vulkan_shader NAME SHADER ENABLED)
    set(HEADER "${SD_VULKAN_GENERATED}/${NAME}.h")
    if(ENABLED)
        set(SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SHADER}.comp")
        set(SPV "${SD_VULKAN_GENERATED}/${NAME}.spv")
        set(ENV vulkan1.2)
        if(NAME MATCHES "_cm2$")
            set(ENV vulkan1.3)
        endif()
        add_custom_command(OUTPUT "${HEADER}"
            COMMAND "${Vulkan_GLSLC_EXECUTABLE}" --target-env=${ENV} ${ARGN} "${SOURCE}" -o "${SPV}"
            COMMAND "${CMAKE_COMMAND}" "-DINPUT=${SPV}" "-DOUTPUT=${HEADER}" "-DNAME=${NAME}"
                    -P "${PROJECT_SOURCE_DIR}/cmake/embed_spirv.cmake"
            DEPENDS "${SOURCE}" "${PROJECT_SOURCE_DIR}/cmake/embed_spirv.cmake"
            BYPRODUCTS "${SPV}" VERBATIM)
    else()
        file(WRITE "${HEADER}" "#include <cstdint>\nnamespace sd_vulkan_shaders { static constexpr uint32_t ${NAME}[] = {0}; }\n")
    endif()
    set(SD_VULKAN_HEADERS ${SD_VULKAN_HEADERS} "${HEADER}" PARENT_SCOPE)
endfunction()

# Preserve the native compilation policy: optimization changes RoPE reductions
# and is disabled for KHR cooperative matrix shaders.
sd_vulkan_shader(rope_apply rope_apply ON)
sd_vulkan_shader(rms_rope_apply rms_rope_apply ON)
sd_vulkan_shader(rms_rope_cast rms_rope_apply ON -DCAST_F16)
sd_vulkan_shader(quantize_i8_convrot quantize_i8_convrot ${SD_VK_DOT} -O)
sd_vulkan_shader(quantize_i8_convrot_subgroup quantize_i8_convrot_subgroup ${SD_VK_DOT} -O)
sd_vulkan_shader(quantize_i8_convrot_register quantize_i8_convrot_subgroup ${SD_VK_DOT} -O -DSINGLE_PASS=1)
sd_vulkan_shader(swiglu_quantize_i8_convrot quantize_i8_convrot_subgroup ${SD_VK_DOT} -O -DSWIGLU=1)
sd_vulkan_shader(swiglu_quantize_i8_convrot_16384 quantize_i8_convrot_subgroup ${SD_VK_DOT} -O -DSWIGLU=1 -DSINGLE_PASS=1 -DSTATIC_K=16384)
sd_vulkan_shader(sigmoid_quantize_i8_convrot_6144 quantize_i8_convrot_subgroup ${SD_VK_DOT} -O -DSIGMOID_MUL=1 -DSINGLE_PASS=1 -DSTATIC_K=6144)
sd_vulkan_shader(modulate_convrot modulate_convrot ${SD_VK_DOT} -O)
sd_vulkan_shader(rms_modulate_convrot modulate_convrot ${SD_VK_DOT} -O -DRMS_AFFINE=1)
sd_vulkan_shader(mul_mat_i8_tensorwise mul_mat_i8_tensorwise ${SD_VK_DOT} -O)
sd_vulkan_shader(mul_mat_i8_tensorwise_cm1 mul_mat_i8_tensorwise_cm1 ${SD_VK_CM1})
sd_vulkan_shader(mul_mat_i8_tensorwise_cm2 mul_mat_i8_tensorwise_cm2 ${SD_VK_CM2} -O)
sd_vulkan_shader(mul_mat_i8_swiglu_split_cm2 mul_mat_i8_tensorwise_cm2 ${SD_VK_CM2} -O -DSPLIT_SWIGLU=1)
sd_vulkan_shader(mul_mat_i8_residual_cm2 mul_mat_i8_tensorwise_cm2 ${SD_VK_CM2} -O -DRESIDUAL=1)
add_custom_target(sd-vulkan-shaders DEPENDS ${SD_VULKAN_HEADERS})
add_dependencies(${SD_LIB} sd-vulkan-shaders)
target_sources(${SD_LIB} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/rope.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/int8.cpp")
target_include_directories(${SD_LIB} PRIVATE "${SD_VULKAN_GENERATED}")
