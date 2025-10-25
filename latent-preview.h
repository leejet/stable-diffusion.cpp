#include <cstddef>
#include <cstdint>
#include "ggml.h"
const float wan_21_latent_rgb_proj[16][3] = {
    {-0.1299, -0.1692, 0.2932},
    {0.0671, 0.0406, 0.0442},
    {0.3568, 0.2548, 0.1747},
    {0.0372, 0.2344, 0.1420},
    {0.0313, 0.0189, -0.0328},
    {0.0296, -0.0956, -0.0665},
    {-0.3477, -0.4059, -0.2925},
    {0.0166, 0.1902, 0.1975},
    {-0.0412, 0.0267, -0.1364},
    {-0.1293, 0.0740, 0.1636},
    {0.0680, 0.3019, 0.1128},
    {0.0032, 0.0581, 0.0639},
    {-0.1251, 0.0927, 0.1699},
    {0.0060, -0.0633, 0.0005},
    {0.3477, 0.2275, 0.2950},
    {0.1984, 0.0913, 0.1861}};
float wan_21_latent_rgb_bias[3] = {-0.1223, -0.1889, -0.1976};

const float wan_22_latent_rgb_proj[48][3] = {
    {0.0119, 0.0103, 0.0046},
    {-0.1062, -0.0504, 0.0165},
    {0.0140, 0.0409, 0.0491},
    {-0.0813, -0.0677, 0.0607},
    {0.0656, 0.0851, 0.0808},
    {0.0264, 0.0463, 0.0912},
    {0.0295, 0.0326, 0.0590},
    {-0.0244, -0.0270, 0.0025},
    {0.0443, -0.0102, 0.0288},
    {-0.0465, -0.0090, -0.0205},
    {0.0359, 0.0236, 0.0082},
    {-0.0776, 0.0854, 0.1048},
    {0.0564, 0.0264, 0.0561},
    {0.0006, 0.0594, 0.0418},
    {-0.0319, -0.0542, -0.0637},
    {-0.0268, 0.0024, 0.0260},
    {0.0539, 0.0265, 0.0358},
    {-0.0359, -0.0312, -0.0287},
    {-0.0285, -0.1032, -0.1237},
    {0.1041, 0.0537, 0.0622},
    {-0.0086, -0.0374, -0.0051},
    {0.0390, 0.0670, 0.2863},
    {0.0069, 0.0144, 0.0082},
    {0.0006, -0.0167, 0.0079},
    {0.0313, -0.0574, -0.0232},
    {-0.1454, -0.0902, -0.0481},
    {0.0714, 0.0827, 0.0447},
    {-0.0304, -0.0574, -0.0196},
    {0.0401, 0.0384, 0.0204},
    {-0.0758, -0.0297, -0.0014},
    {0.0568, 0.1307, 0.1372},
    {-0.0055, -0.0310, -0.0380},
    {0.0239, -0.0305, 0.0325},
    {-0.0663, -0.0673, -0.0140},
    {-0.0416, -0.0047, -0.0023},
    {0.0166, 0.0112, -0.0093},
    {-0.0211, 0.0011, 0.0331},
    {0.1833, 0.1466, 0.2250},
    {-0.0368, 0.0370, 0.0295},
    {-0.3441, -0.3543, -0.2008},
    {-0.0479, -0.0489, -0.0420},
    {-0.0660, -0.0153, 0.0800},
    {-0.0101, 0.0068, 0.0156},
    {-0.0690, -0.0452, -0.0927},
    {-0.0145, 0.0041, 0.0015},
    {0.0421, 0.0451, 0.0373},
    {0.0504, -0.0483, -0.0356},
    {-0.0837, 0.0168, 0.0055}};
float wan_22_latent_rgb_bias[3] = {0.0317, -0.0878, -0.1388};

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
float flux_latent_rgb_bias[3] = {-0.0329, -0.0718, -0.0851};

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
float sd3_latent_rgb_bias[3] = {0, 0, 0};

// https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/latent_formats.py#L32-L38
const float sdxl_latent_rgb_proj[4][3] = {
    {0.3651f, 0.4232f, 0.4341f},
    {-0.2533f, -0.0042f, 0.1068f},
    {0.1076f, 0.1111f, -0.0362f},
    {-0.3165f, -0.2492f, -0.2188f}};
float sdxl_latent_rgb_bias[3] = {0.1084, -0.0175, -0.0011};

// https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/latent_formats.py#L32-L38
const float sd_latent_rgb_proj[4][3]{
    {0.3512f, 0.2297f, 0.3227f},
    {0.3250f, 0.4974f, 0.2350f},
    {-0.2829f, 0.1762f, 0.2721f},
    {-0.2120f, -0.2616f, -0.7177f}};
float sd_latent_rgb_bias[3] = {0,0,0};

void preview_latent_video(uint8_t* buffer, struct ggml_tensor* latents, const float (*latent_rgb_proj)[3], const float latent_rgb_bias[3], int width, int height, int frames, int dim) {
    size_t buffer_head = 0;
    for (int k = 0; k < frames; k++) {
        for (int j = 0; j < height; j++) {
            for (int i = 0; i < width; i++) {
                size_t latent_id = (i * latents->nb[0] + j * latents->nb[1] + k * latents->nb[2]);
                float r = 0, g = 0, b = 0;
                if(latent_rgb_proj!=NULL){
                    for (int d = 0; d < dim; d++) {
                        float value = *(float*)((char*)latents->data + latent_id + d * latents->nb[ggml_n_dims(latents) - 1]);
                        r += value * latent_rgb_proj[d][0];
                        g += value * latent_rgb_proj[d][1];
                        b += value * latent_rgb_proj[d][2];
                    }
                } else {
                    // interpret first 3 channels as RGB
                    r = *(float*)((char*)latents->data + latent_id + 0 * latents->nb[ggml_n_dims(latents) - 1]);
                    g = *(float*)((char*)latents->data + latent_id + 1 * latents->nb[ggml_n_dims(latents) - 1]);
                    b = *(float*)((char*)latents->data + latent_id + 2 * latents->nb[ggml_n_dims(latents) - 1]);
                }
                if(latent_rgb_bias!=NULL){
                    // bias
                    r += latent_rgb_bias[0];
                    g += latent_rgb_bias[1];
                    b += latent_rgb_bias[2];
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
}
