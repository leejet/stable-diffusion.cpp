// Unit test: FLUX.2-klein diffusers→BFL tensor name conversion
// Tests convert_tensor_name() with VERSION_FLUX2_KLEIN through the full
// public API path: prefix mapping → convert_diffusion_model_name() →
// convert_diffusers_dit_to_original_flux2()
//
// Build:
//   c++ -std=c++17 -I../src -I../ggml/include -I../thirdparty \
//       tests/test_flux2_name_conversion.cpp \
//       -L build/bin -lstable-diffusion \
//       -L build/ggml/src -lggml -lggml-base -lggml-cpu \
//       -framework Foundation -framework Metal -framework Accelerate \
//       -o build/bin/test_flux2_name_conversion

#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

#include "model.h"
#include "name_conversion.h"

static int g_pass = 0;
static int g_fail = 0;

// When loading via --diffusion-model, tensors get prefix "model.diffusion_model."
// and convert_tensor_name strips "transformer." → "model.diffusion_model."
// So input names like "transformer_blocks.0.attn.to_q.weight" become
// "model.diffusion_model.transformer_blocks.0.attn.to_q.weight" after prefix map,
// then convert_diffusion_model_name strips "model.diffusion_model." prefix,
// calls convert_diffusers_dit_to_original_flux2(), and re-adds the prefix.

static void check(const char* test_name,
                  const std::string& diffusers_name,
                  const std::string& expected_bfl_name,
                  SDVersion version = VERSION_FLUX2_KLEIN) {
    // Simulate the input as it comes from diffusers file loaded via --diffusion-model:
    // The prefix "model.diffusion_model." is added by init_from_file_and_convert_name
    std::string input = "model.diffusion_model." + diffusers_name;
    std::string expected = "model.diffusion_model." + expected_bfl_name;

    std::string result = convert_tensor_name(input, version);

    if (result == expected) {
        g_pass++;
    } else {
        g_fail++;
        fprintf(stderr, "FAIL [%s]\n  input:    %s\n  expected: %s\n  got:      %s\n\n",
                test_name, input.c_str(), expected.c_str(), result.c_str());
    }
}

int main() {
    printf("=== FLUX.2-klein diffusers→BFL name conversion tests ===\n\n");

    // ---------------------------------------------------------------
    // 1. Time/guidance embedders
    // ---------------------------------------------------------------
    check("time_in.in_layer",
          "time_guidance_embed.timestep_embedder.linear_1.weight",
          "time_in.in_layer.weight");
    check("time_in.out_layer",
          "time_guidance_embed.timestep_embedder.linear_2.weight",
          "time_in.out_layer.weight");

    // ---------------------------------------------------------------
    // 2. Input embedders
    // ---------------------------------------------------------------
    check("txt_in (context_embedder)",
          "context_embedder.weight",
          "txt_in.weight");
    check("img_in (x_embedder)",
          "x_embedder.weight",
          "img_in.weight");

    // ---------------------------------------------------------------
    // 3. Shared modulations (.linear. → .lin.)
    // ---------------------------------------------------------------
    check("double_mod_img.weight",
          "double_stream_modulation_img.linear.weight",
          "double_stream_modulation_img.lin.weight");
    check("double_mod_img.bias",
          "double_stream_modulation_img.linear.bias",
          "double_stream_modulation_img.lin.bias");
    check("double_mod_txt.weight",
          "double_stream_modulation_txt.linear.weight",
          "double_stream_modulation_txt.lin.weight");
    check("double_mod_txt.bias",
          "double_stream_modulation_txt.linear.bias",
          "double_stream_modulation_txt.lin.bias");
    check("single_mod.weight",
          "single_stream_modulation.linear.weight",
          "single_stream_modulation.lin.weight");
    check("single_mod.bias",
          "single_stream_modulation.linear.bias",
          "single_stream_modulation.lin.bias");

    // ---------------------------------------------------------------
    // 4. Double blocks — block 0 (first)
    // ---------------------------------------------------------------
    // img attention split q/k/v → fused qkv
    check("dbl0.img_attn.q",
          "transformer_blocks.0.attn.to_q.weight",
          "double_blocks.0.img_attn.qkv.weight");
    check("dbl0.img_attn.k",
          "transformer_blocks.0.attn.to_k.weight",
          "double_blocks.0.img_attn.qkv.weight.1");
    check("dbl0.img_attn.v",
          "transformer_blocks.0.attn.to_v.weight",
          "double_blocks.0.img_attn.qkv.weight.2");
    check("dbl0.img_attn.proj",
          "transformer_blocks.0.attn.to_out.0.weight",
          "double_blocks.0.img_attn.proj.weight");
    check("dbl0.img_attn.norm_q",
          "transformer_blocks.0.attn.norm_q.weight",
          "double_blocks.0.img_attn.norm.query_norm.scale");
    check("dbl0.img_attn.norm_k",
          "transformer_blocks.0.attn.norm_k.weight",
          "double_blocks.0.img_attn.norm.key_norm.scale");

    // txt attention split q/k/v
    check("dbl0.txt_attn.q",
          "transformer_blocks.0.attn.add_q_proj.weight",
          "double_blocks.0.txt_attn.qkv.weight");
    check("dbl0.txt_attn.k",
          "transformer_blocks.0.attn.add_k_proj.weight",
          "double_blocks.0.txt_attn.qkv.weight.1");
    check("dbl0.txt_attn.v",
          "transformer_blocks.0.attn.add_v_proj.weight",
          "double_blocks.0.txt_attn.qkv.weight.2");
    check("dbl0.txt_attn.proj",
          "transformer_blocks.0.attn.to_add_out.weight",
          "double_blocks.0.txt_attn.proj.weight");
    check("dbl0.txt_attn.norm_q",
          "transformer_blocks.0.attn.norm_added_q.weight",
          "double_blocks.0.txt_attn.norm.query_norm.scale");
    check("dbl0.txt_attn.norm_k",
          "transformer_blocks.0.attn.norm_added_k.weight",
          "double_blocks.0.txt_attn.norm.key_norm.scale");

    // img MLP (SwiGLU)
    check("dbl0.img_mlp.in",
          "transformer_blocks.0.ff.linear_in.weight",
          "double_blocks.0.img_mlp.0.weight");
    check("dbl0.img_mlp.out",
          "transformer_blocks.0.ff.linear_out.weight",
          "double_blocks.0.img_mlp.2.weight");

    // txt MLP (SwiGLU)
    check("dbl0.txt_mlp.in",
          "transformer_blocks.0.ff_context.linear_in.weight",
          "double_blocks.0.txt_mlp.0.weight");
    check("dbl0.txt_mlp.out",
          "transformer_blocks.0.ff_context.linear_out.weight",
          "double_blocks.0.txt_mlp.2.weight");

    // ---------------------------------------------------------------
    // 5. Double blocks — block 4 (last for klein-4B with depth=5)
    // ---------------------------------------------------------------
    check("dbl4.img_attn.q",
          "transformer_blocks.4.attn.to_q.weight",
          "double_blocks.4.img_attn.qkv.weight");
    check("dbl4.txt_attn.v",
          "transformer_blocks.4.attn.add_v_proj.weight",
          "double_blocks.4.txt_attn.qkv.weight.2");
    check("dbl4.img_mlp.out",
          "transformer_blocks.4.ff.linear_out.weight",
          "double_blocks.4.img_mlp.2.weight");

    // ---------------------------------------------------------------
    // 6. Single blocks — block 0 (first)
    // ---------------------------------------------------------------
    check("sgl0.linear1 (fused qkv+mlp)",
          "single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight",
          "single_blocks.0.linear1.weight");
    check("sgl0.linear2 (out proj)",
          "single_transformer_blocks.0.attn.to_out.weight",
          "single_blocks.0.linear2.weight");
    check("sgl0.norm_q",
          "single_transformer_blocks.0.attn.norm_q.weight",
          "single_blocks.0.norm.query_norm.scale");
    check("sgl0.norm_k",
          "single_transformer_blocks.0.attn.norm_k.weight",
          "single_blocks.0.norm.key_norm.scale");

    // ---------------------------------------------------------------
    // 7. Single blocks — block 19 (last for klein-4B with depth_single=20)
    // ---------------------------------------------------------------
    check("sgl19.linear1",
          "single_transformer_blocks.19.attn.to_qkv_mlp_proj.weight",
          "single_blocks.19.linear1.weight");
    check("sgl19.linear2",
          "single_transformer_blocks.19.attn.to_out.weight",
          "single_blocks.19.linear2.weight");

    // ---------------------------------------------------------------
    // 8. Final layers
    // ---------------------------------------------------------------
    check("final_layer.linear",
          "proj_out.weight",
          "final_layer.linear.weight");
    check("final_layer.adaLN",
          "norm_out.linear.weight",
          "final_layer.adaLN_modulation.1.weight");

    // ---------------------------------------------------------------
    // 9. Higher block indices (for 9B: depth=10 double, depth_single=42)
    // ---------------------------------------------------------------
    check("dbl9.img_attn.q (9B)",
          "transformer_blocks.9.attn.to_q.weight",
          "double_blocks.9.img_attn.qkv.weight");
    check("sgl41.linear1 (9B)",
          "single_transformer_blocks.41.attn.to_qkv_mlp_proj.weight",
          "single_blocks.41.linear1.weight");

    // ---------------------------------------------------------------
    // 10. VERSION_FLUX2 (non-klein) should also route to flux2 converter
    // ---------------------------------------------------------------
    check("flux2_version_routing",
          "double_stream_modulation_img.linear.weight",
          "double_stream_modulation_img.lin.weight",
          VERSION_FLUX2);

    // ---------------------------------------------------------------
    // 11. Identity: names that don't match any mapping should pass through
    // ---------------------------------------------------------------
    check("unknown_passthrough",
          "some_unknown_tensor.weight",
          "some_unknown_tensor.weight");

    // ---------------------------------------------------------------
    // Summary
    // ---------------------------------------------------------------
    printf("\n=== Results: %d passed, %d failed, %d total ===\n",
           g_pass, g_fail, g_pass + g_fail);

    if (g_fail > 0) {
        printf("FAILED\n");
        return 1;
    }
    printf("ALL PASSED\n");
    return 0;
}
