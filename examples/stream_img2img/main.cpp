#include "stable-diffusion.h"
#include <cstdio>
#include <cstdlib>
#include <ctime>

// Stb image for saving output (requires #define STB_IMAGE_WRITE_IMPLEMENTATION in one file)
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// Helper to generate a dummy 512x512 gradient image that shifts over time
sd_image_t get_next_frame(int frame_idx) {
    int width = 512;
    int height = 512;
    uint8_t* data = (uint8_t*)malloc(width * height * 3);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            data[(y * width + x) * 3 + 0] = (x + frame_idx * 50) % 256; 
            data[(y * width + x) * 3 + 1] = (y + frame_idx * 50) % 256; 
            data[(y * width + x) * 3 + 2] = 128;                        
        }
    }
    return { (uint32_t)width, (uint32_t)height, 3, data };
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <model_path> [diffusion_model_path] [vae_path] [t5xxl_path]\n", argv[0]);
        return 1;
    }
    
    sd_ctx_params_t params;
    sd_ctx_params_init(&params);
    params.model_path = argv[1];
    if (argc > 2) params.diffusion_model_path = argv[2];
    if (argc > 3) params.vae_path = argv[3];
    if (argc > 4) params.t5xxl_path = argv[4];

    params.n_threads = -1; // auto
    
    printf("Initializing sd context...\n");
    sd_ctx_t* ctx = new_sd_ctx(&params);
    if (!ctx) {
        printf("Failed to load model\n");
        return 1;
    }

    // Pre-calculate condition (Text Encoder)
    printf("Pre-encoding prompt...\n");
    clock_t start = clock();
    sd_condition_t* cond = sd_encode_condition(ctx,
        "cinematic style, oil painting, beautiful landscape",  // positive prompt
        ""                                                     // negative prompt
    );
    clock_t end = clock();
    printf("Prompt encoded in %.3f seconds.\n", (double)(end - start) / CLOCKS_PER_SEC);

    if (!cond) {
        printf("Failed to encode condition\n");
        return 1;
    }

    // Pre-calculate reference image (VAE Encode reference image)
    printf("Pre-encoding reference image...\n");
    sd_image_t ref_img = get_next_frame(100);
    start = clock();
    sd_image_latent_t* ref_latents[] = {
        sd_encode_ref_image(ctx, &ref_img)
    };
    end = clock();
    printf("Reference image encoded in %.3f seconds.\n", (double)(end - start) / CLOCKS_PER_SEC);
    free(ref_img.data);
    
    if (!ref_latents[0]) {
        printf("Failed to encode ref image\n");
        return 1;
    }

    // Frame processing loop
    int num_frames = 3;
    for (int i = 0; i < num_frames; i++) {
        printf("\nProcessing frame %d/%d...\n", i + 1, num_frames);
        sd_image_t frame = get_next_frame(i);

        start = clock();

        sd_image_t result = sd_img2img_with_cond(
            ctx,
            frame,
            cond,
            ref_latents, 1,   // 1 reference image
            0.6f,             // strength
            4,                // steps (distilled models)
            1.0f,             // cfg_scale
            42                // seed
        );

        end = clock();
        double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
        printf("Frame %d processed in %.3f seconds.\n", i + 1, elapsed);

        if (result.data) {
            char filename[256];
            snprintf(filename, sizeof(filename), "output_frame_%d.png", i);
            stbi_write_png(filename, result.width, result.height, result.channel, result.data, 0);
            printf("Saved result to %s\n", filename);
            free(result.data);
        } else {
            printf("Failed to generate result for frame %d\n", i + 1);
        }
        free(frame.data);
    }

    // Free resources
    if (ref_latents[0]) sd_free_image_latent(ref_latents[0]);
    if (cond) sd_free_condition(cond);
    free_sd_ctx(ctx);

    printf("Done.\n");
    return 0;
}
