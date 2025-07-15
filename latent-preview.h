
// https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/latent_formats.py#L152-L169
const float flux_latent_rgb_proj[16][3] = {
    {-0.0346f, 0.0244f, 0.0681f},
    {0.0034f, 0.0210f, 0.0687f},
    {0.0275f, -0.0668f, -0.0433f},
    {-0.0174f, 0.0160f, 0.0617f},
    {0.0859f, 0.0721f, 0.0329f},
    {0.0004f, 0.0383f, 0.0115f},
    {0.0405f, 0.0861f, 0.0915f},
    {-0.0236f, -0.0185f, -0.0259f},
    {-0.0245f, 0.0250f, 0.1180f},
    {0.1008f, 0.0755f, -0.0421f},
    {-0.0515f, 0.0201f, 0.0011f},
    {0.0428f, -0.0012f, -0.0036f},
    {0.0817f, 0.0765f, 0.0749f},
    {-0.1264f, -0.0522f, -0.1103f},
    {-0.0280f, -0.0881f, -0.0499f},
    {-0.1262f, -0.0982f, -0.0778f}};

// https://github.com/Stability-AI/sd3.5/blob/main/sd3_impls.py#L228-L246
const float sd3_latent_rgb_proj[16][3] = {
    {-0.0645f, 0.0177f, 0.1052f},
    {0.0028f, 0.0312f, 0.0650f},
    {0.1848f, 0.0762f, 0.0360f},
    {0.0944f, 0.0360f, 0.0889f},
    {0.0897f, 0.0506f, -0.0364f},
    {-0.0020f, 0.1203f, 0.0284f},
    {0.0855f, 0.0118f, 0.0283f},
    {-0.0539f, 0.0658f, 0.1047f},
    {-0.0057f, 0.0116f, 0.0700f},
    {-0.0412f, 0.0281f, -0.0039f},
    {0.1106f, 0.1171f, 0.1220f},
    {-0.0248f, 0.0682f, -0.0481f},
    {0.0815f, 0.0846f, 0.1207f},
    {-0.0120f, -0.0055f, -0.0867f},
    {-0.0749f, -0.0634f, -0.0456f},
    {-0.1418f, -0.1457f, -0.1259f},
};

// https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/latent_formats.py#L32-L38
const float sdxl_latent_rgb_proj[4][3] = {
    {0.3651f, 0.4232f, 0.4341f},
    {-0.2533f, -0.0042f, 0.1068f},
    {0.1076f, 0.1111f, -0.0362f},
    {-0.3165f, -0.2492f, -0.2188f}};

// https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/latent_formats.py#L32-L38
const float sd_latent_rgb_proj[4][3]{
    {0.3512f, 0.2297f, 0.3227f},
    {0.3250f, 0.4974f, 0.2350f},
    {-0.2829f, 0.1762f, 0.2721f},
    {-0.2120f, -0.2616f, -0.7177f}};

void preview_latent_image(uint8_t* buffer, struct ggml_tensor* latents, const float (*latent_rgb_proj)[3], int width, int height, int dim) {
    size_t buffer_head = 0;
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            size_t latent_id = (i * latents->nb[0] + j * latents->nb[1]);
            float r = 0, g = 0, b = 0;
            for (int d = 0; d < dim; d++) {
                float value = *(float*)((char*)latents->data + latent_id + d * latents->nb[2]);
                r += value * latent_rgb_proj[d][0];
                g += value * latent_rgb_proj[d][1];
                b += value * latent_rgb_proj[d][2];
            }

            // change range
            r = r * .5f + .5f;
            g = g * .5f + .5f;
            b = b * .5f + .5f;

            // clamp rgb values to [0,1] range
            r = r >= 0 ? r <= 1 ? r : 1 : 0;
            g = g >= 0 ? g <= 1 ? g : 1 : 0;
            b = b >= 0 ? b <= 1 ? b : 1 : 0;

            buffer[buffer_head++] = (uint8_t)(r * 255);
            buffer[buffer_head++] = (uint8_t)(g * 255);
            buffer[buffer_head++] = (uint8_t)(b * 255);
        }
    }
}