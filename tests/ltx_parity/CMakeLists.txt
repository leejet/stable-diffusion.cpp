set(TARGET sd-ltx-parity)

add_executable(${TARGET}
    test_ltx_parity.cpp
)

target_link_libraries(${TARGET} PRIVATE stable-diffusion ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PUBLIC c_std_11 cxx_std_17)

set(GEMMA_TARGET sd-gemma-parity)

add_executable(${GEMMA_TARGET}
    test_gemma_parity.cpp
)

target_link_libraries(${GEMMA_TARGET} PRIVATE stable-diffusion ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${GEMMA_TARGET} PUBLIC c_std_11 cxx_std_17)

set(GEMMA_TOK_TARGET sd-gemma-tokenizer-test)

add_executable(${GEMMA_TOK_TARGET}
    test_gemma_tokenizer.cpp
)

target_link_libraries(${GEMMA_TOK_TARGET} PRIVATE stable-diffusion ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${GEMMA_TOK_TARGET} PUBLIC c_std_11 cxx_std_17)

set(S2D_TARGET sd-s2d-primitives-test)

add_executable(${S2D_TARGET}
    test_s2d_primitives.cpp
)

target_link_libraries(${S2D_TARGET} PRIVATE stable-diffusion ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${S2D_TARGET} PUBLIC c_std_11 cxx_std_17)

set(VAE_TARGET sd-vae-parity)

add_executable(${VAE_TARGET}
    test_vae_parity.cpp
)

target_link_libraries(${VAE_TARGET} PRIVATE stable-diffusion ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${VAE_TARGET} PUBLIC c_std_11 cxx_std_17)

set(CONN_TARGET sd-connector-parity)

add_executable(${CONN_TARGET}
    test_connector_parity.cpp
)

target_link_libraries(${CONN_TARGET} PRIVATE stable-diffusion ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${CONN_TARGET} PUBLIC c_std_11 cxx_std_17)

set(VAE_RT_TARGET sd-ltx2-vae-roundtrip)

add_executable(${VAE_RT_TARGET}
    test_ltx2_vae_roundtrip.cpp
)

target_link_libraries(${VAE_RT_TARGET} PRIVATE stable-diffusion ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${VAE_RT_TARGET} PUBLIC c_std_11 cxx_std_17)

set(GEMMA_CC_TARGET sd-gemma-cpu-vs-cuda)

add_executable(${GEMMA_CC_TARGET}
    test_gemma_cpu_vs_cuda.cpp
)

target_link_libraries(${GEMMA_CC_TARGET} PRIVATE stable-diffusion ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${GEMMA_CC_TARGET} PUBLIC c_std_11 cxx_std_17)

set(MM_PARITY_TARGET sd-mm-f32-parity)

add_executable(${MM_PARITY_TARGET}
    test_mm_f32_parity.cpp
)

target_link_libraries(${MM_PARITY_TARGET} PRIVATE stable-diffusion ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${MM_PARITY_TARGET} PUBLIC c_std_11 cxx_std_17)

set(CONT_PARITY_TARGET sd-cont-parity)

add_executable(${CONT_PARITY_TARGET}
    test_cont_parity.cpp
)

target_link_libraries(${CONT_PARITY_TARGET} PRIVATE stable-diffusion ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${CONT_PARITY_TARGET} PUBLIC c_std_11 cxx_std_17)

set(SOFTMAX_PARITY_TARGET sd-softmax-parity)

add_executable(${SOFTMAX_PARITY_TARGET}
    test_softmax_parity.cpp
)

target_link_libraries(${SOFTMAX_PARITY_TARGET} PRIVATE stable-diffusion ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${SOFTMAX_PARITY_TARGET} PUBLIC c_std_11 cxx_std_17)

set(ATTN_CHAIN_TARGET sd-attn-chain-parity)

add_executable(${ATTN_CHAIN_TARGET}
    test_attn_chain_parity.cpp
)

target_link_libraries(${ATTN_CHAIN_TARGET} PRIVATE stable-diffusion ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${ATTN_CHAIN_TARGET} PUBLIC c_std_11 cxx_std_17)

set(AV_BLOCK_SMOKE_TARGET sd-av-block-smoke)

add_executable(${AV_BLOCK_SMOKE_TARGET}
    test_av_block_smoke.cpp
)

target_link_libraries(${AV_BLOCK_SMOKE_TARGET} PRIVATE stable-diffusion ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${AV_BLOCK_SMOKE_TARGET} PUBLIC c_std_11 cxx_std_17)

set(AV_BLOCK_PARITY_TARGET sd-av-block-parity)

add_executable(${AV_BLOCK_PARITY_TARGET}
    test_av_block_parity.cpp
)

target_link_libraries(${AV_BLOCK_PARITY_TARGET} PRIVATE stable-diffusion ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${AV_BLOCK_PARITY_TARGET} PUBLIC c_std_11 cxx_std_17)

set(AV_MODEL_PARITY_TARGET sd-av-model-parity)

add_executable(${AV_MODEL_PARITY_TARGET}
    test_av_model_parity.cpp
)

target_link_libraries(${AV_MODEL_PARITY_TARGET} PRIVATE stable-diffusion ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${AV_MODEL_PARITY_TARGET} PUBLIC c_std_11 cxx_std_17)
