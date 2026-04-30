// LTX-2 VAE encode→decode round-trip sanity check on a real 22B VAE checkpoint.
//
// Purpose: rule out whether the blocky output from the end-to-end LTX-2 pipeline
// is caused by a broken VAE decoder. Constructs a simple synthetic video
// (color gradient ramps), runs the real LTX-2 VAE through encode→decode, and
// reports the reconstruction MSE + dumps the first output frame's values.
//
// If MSE is small (<0.05 for bounded [-1,1] input), the VAE is sound and the
// structural issue must live upstream (DiT / conditioning).  If MSE is high,
// the VAE itself is miscomputing and that explains the pipeline output.
//
// Usage: sd-ltx2-vae-roundtrip <path/to/video_vae.safetensors> [WIDTH [HEIGHT [FRAMES]]]

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "ltxvae.hpp"
#include "model.h"
#include "tensor.hpp"

namespace {

void apply_pcs_duplication(String2TensorStorage& tsm) {
    // Mirror stable-diffusion.cpp's LTX-2-specific duplication so the nested
    // encoder.per_channel_statistics.* / decoder.per_channel_statistics.* paths
    // that our VideoEncoder/Decoder blocks look up exist.
    const std::string top_pre = "first_stage_model.per_channel_statistics.";
    std::vector<std::pair<std::string, std::string>> to_copy;
    for (const auto& kv : tsm) {
        const std::string& k = kv.first;
        if (k.rfind(top_pre, 0) == 0) {
            to_copy.push_back({k, k.substr(top_pre.size())});
        }
    }
    size_t copied = 0;
    for (auto& pair : to_copy) {
        auto src_it = tsm.find(pair.first);
        if (src_it == tsm.end()) continue;
        for (const char* sub : {"encoder", "decoder"}) {
            std::string dst = "first_stage_model." + std::string(sub) +
                              ".per_channel_statistics." + pair.second;
            if (tsm.find(dst) != tsm.end()) continue;
            TensorStorage dup = src_it->second;
            dup.name          = dst;
            tsm[dst]          = dup;
            copied++;
        }
    }
    std::printf("[pcs] duplicated %zu entries\n", copied);
}

// Builds a 9-frame [W, H, T, 3] synthetic video in [-1, 1].  Produces a
// spatial gradient ramp in R/G/B channels so reconstruction is easy to eyeball.
sd::Tensor<float> make_synthetic_video(int W, int H, int T) {
    sd::Tensor<float> v({W, H, T, 3});
    for (int t = 0; t < T; ++t) {
        float tphase = static_cast<float>(t) / std::max(T - 1, 1);
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                float r = static_cast<float>(w) / std::max(W - 1, 1) * 2.0f - 1.0f;
                float g = static_cast<float>(h) / std::max(H - 1, 1) * 2.0f - 1.0f;
                float b = tphase * 2.0f - 1.0f;
                v.index(w, h, t, 0) = r;
                v.index(w, h, t, 1) = g;
                v.index(w, h, t, 2) = b;
            }
        }
    }
    return v;
}

struct DiffStats {
    float max_abs = 0.f, mean_abs = 0.f, mse = 0.f;
};
DiffStats diff_stats(const float* a, const float* b, int64_t n) {
    DiffStats s;
    double sum_abs = 0.0, sum_sq = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        float d   = a[i] - b[i];
        float ad  = std::fabs(d);
        s.max_abs = std::max(s.max_abs, ad);
        sum_abs += ad;
        sum_sq += static_cast<double>(d) * d;
    }
    s.mean_abs = static_cast<float>(sum_abs / std::max<int64_t>(n, 1));
    s.mse      = static_cast<float>(sum_sq / std::max<int64_t>(n, 1));
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    sd_set_log_callback(
        [](enum sd_log_level_t /*level*/, const char* text, void* /*data*/) {
            std::fputs(text, stderr);
        },
        nullptr);

    const char* vae_path = (argc >= 2)
                               ? argv[1]
                               : "/media/ilintar/D_SSD/models/ltx-2/ltx-2.3-22b-dev_video_vae.safetensors";
    int W = (argc >= 3) ? std::atoi(argv[2]) : 128;
    int H = (argc >= 4) ? std::atoi(argv[3]) : 128;
    int T = (argc >= 5) ? std::atoi(argv[4]) : 9;

    std::printf("[cfg] vae_path = %s\n", vae_path);
    std::printf("[cfg] input video = %dx%d, %d frames\n", W, H, T);

    ModelLoader loader;
    // The raw 22B video_vae.safetensors ships tensors as `encoder.*`, `decoder.*`,
    // `per_channel_statistics.*` with no top-level prefix.  Passing prefix="vae." on
    // init adds that so the subsequent convert_tensors_name() remaps `vae.` →
    // `first_stage_model.` via name_conversion.cpp.
    if (!loader.init_from_file(vae_path, "vae.")) {
        std::fprintf(stderr, "fatal: init_from_file failed for %s\n", vae_path);
        return 1;
    }
    loader.convert_tensors_name();
    auto& tsm = loader.get_tensor_storage_map();
    std::printf("[state_dict] loaded %zu tensors\n", tsm.size());
    apply_pcs_duplication(tsm);

    // Allow toggling timestep conditioning from env so we can A/B the prod config
    // (the real 22B checkpoint is timestep_conditioned).
    bool tcond = std::getenv("VAE_TIMESTEP_COND") != nullptr;
    std::printf("[cfg] timestep_cond = %s\n", tcond ? "true" : "false");

    ggml_backend_t backend = ggml_backend_cpu_init();
    LTXVAE::LTX2VAERunner vae(backend, /*offload=*/false, tsm,
                              /*prefix=*/"first_stage_model",
                              VERSION_LTX2,
                              /*in_ch=*/3, /*latent_ch=*/128, /*patch=*/4,
                              /*decoder_base_ch=*/128, /*timestep_cond=*/tcond,
                              LTXVAE::LTX2VAERunner::ltx2_22b_enc_specs(),
                              LTXVAE::LTX2VAERunner::ltx2_22b_dec_specs());
    vae.alloc_params_buffer();
    std::map<std::string, ggml_tensor*> vae_params;
    vae.get_param_tensors(vae_params, "first_stage_model");
    std::printf("[vae] requesting %zu param tensors\n", vae_params.size());
    if (!loader.load_tensors(vae_params)) {
        std::fprintf(stderr, "fatal: vae load_tensors failed (weights unmatched?)\n");
        return 1;
    }

    // Build synthetic [W, H, T, 3] video in [-1, 1].
    auto video = make_synthetic_video(W, H, T);
    std::printf("[input] shape = [W=%d, H=%d, T=%d, C=3] min=%.3f max=%.3f mean=%.3f\n",
                W, H, T, *std::min_element(video.data(), video.data() + video.numel()),
                *std::max_element(video.data(), video.data() + video.numel()),
                [&]() {
                    double s = 0;
                    for (int64_t i = 0; i < video.numel(); ++i) s += video.data()[i];
                    return static_cast<float>(s / video.numel());
                }());

    // --- Encode ---
    std::printf("[encode] running…\n");
    auto latent = vae._compute(/*n_threads=*/1, video, /*decode_graph=*/false);
    std::printf("[encode] latent shape = [");
    for (size_t i = 0; i < latent.shape().size(); ++i)
        std::printf("%s%lld", (i ? ", " : ""), (long long)latent.shape()[i]);
    std::printf("] numel=%lld\n", (long long)latent.numel());
    if (latent.empty()) {
        std::fprintf(stderr, "fatal: encode produced empty output\n");
        return 2;
    }

    // Latent first 8 values for eyeballing.
    std::printf("[encode] first 8 latent values: ");
    for (int i = 0; i < 8 && i < latent.numel(); ++i)
        std::printf("%+.3f ", latent.data()[i]);
    std::printf("\n");

    // Encoder's output layout is [W_lat, H_lat, T_lat, C_lat].  The decoder's
    // expected input is the same layout.
    // --- Decode ---
    std::printf("[decode] running…\n");
    auto recon = vae._compute(/*n_threads=*/1, latent, /*decode_graph=*/true);
    std::printf("[decode] recon shape = [");
    for (size_t i = 0; i < recon.shape().size(); ++i)
        std::printf("%s%lld", (i ? ", " : ""), (long long)recon.shape()[i]);
    std::printf("] numel=%lld\n", (long long)recon.numel());
    if (recon.empty()) {
        std::fprintf(stderr, "fatal: decode produced empty output\n");
        return 3;
    }

    if (recon.numel() != video.numel()) {
        std::fprintf(stderr, "fatal: recon numel %lld != input numel %lld "
                             "(enc/dec changed element count)\n",
                     (long long)recon.numel(), (long long)video.numel());
        return 4;
    }

    std::printf("[decode] first 8 recon values: ");
    for (int i = 0; i < 8 && i < recon.numel(); ++i)
        std::printf("%+.3f ", recon.data()[i]);
    std::printf("\n[input ] first 8  input values: ");
    for (int i = 0; i < 8 && i < video.numel(); ++i)
        std::printf("%+.3f ", video.data()[i]);
    std::printf("\n");

    // Per-channel mean(input) - mean(recon) bias diagnostic. The encoder→decoder
    // round-trip should be near-zero biased. If a per-channel bias is visible, the
    // VAE is shifting the output range and that explains "everything looks dark".
    // sd::Tensor layout is [W, H, T, C] (ggml memory order), C is outermost.
    auto sh = video.shape();          // {W, H, T, 3} from make_synthetic_video
    int64_t W_in = sh[0], H_in = sh[1], T_in = sh[2], C_in = sh[3];
    auto rsh = recon.shape();
    int64_t W_o = rsh[0], H_o = rsh[1], T_o = rsh[2], C_o = (rsh.size() >= 4 ? rsh[3] : 3);
    std::printf("\n=== per-channel bias (recon - input) ===\n");
    std::printf("  ch  in_mean   recon_mean  bias        in_std  recon_std\n");
    auto chmean = [](const float* d, int64_t W, int64_t H, int64_t T, int64_t C, int64_t c) {
        double s = 0; int64_t n = W * H * T;
        for (int64_t t = 0; t < T; ++t)
            for (int64_t h = 0; h < H; ++h)
                for (int64_t w = 0; w < W; ++w)
                    s += d[((c * T + t) * H + h) * W + w];
        return s / std::max<int64_t>(n, 1);
    };
    auto chstd = [](const float* d, int64_t W, int64_t H, int64_t T, int64_t C, int64_t c, double mean) {
        double s = 0; int64_t n = W * H * T;
        for (int64_t t = 0; t < T; ++t)
            for (int64_t h = 0; h < H; ++h)
                for (int64_t w = 0; w < W; ++w) {
                    double v = d[((c * T + t) * H + h) * W + w] - mean;
                    s += v * v;
                }
        return std::sqrt(s / std::max<int64_t>(n, 1));
    };
    for (int64_t c = 0; c < std::min<int64_t>(C_in, C_o); ++c) {
        double im = chmean(video.data(), W_in, H_in, T_in, C_in, c);
        double rm = chmean(recon.data(), W_o,  H_o,  T_o,  C_o,  c);
        double is = chstd (video.data(), W_in, H_in, T_in, C_in, c, im);
        double rs = chstd (recon.data(), W_o,  H_o,  T_o,  C_o,  c, rm);
        std::printf("  %lld  %+.4f    %+.4f    %+.4f    %.4f   %.4f\n",
                    (long long)c, im, rm, rm - im, is, rs);
    }

    // Diff.
    auto s = diff_stats(recon.data(), video.data(), recon.numel());
    std::printf("\n=== round-trip diff ===\n");
    std::printf("  max_abs  = %.3e\n", s.max_abs);
    std::printf("  mean_abs = %.3e\n", s.mean_abs);
    std::printf("  mse      = %.3e\n", s.mse);

    // Loose pass thresholds.  LTX-2 VAE is lossy but mean_abs <0.1 for a smooth
    // gradient is a reasonable ceiling.  Anything much worse means structural
    // divergence, not just compression.
    const float tol_mse = 0.05f;
    bool pass           = s.mse < tol_mse;
    std::printf("%s (tol: mse < %.1e)\n",
                pass ? "VAE round-trip: PASS" : "VAE round-trip: FAIL",
                tol_mse);
    return pass ? 0 : 5;
}
