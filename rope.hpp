#ifndef __ROPE_HPP__
#define __ROPE_HPP__

#include <vector>
#include "ggml_extend.hpp"

namespace Rope {
    template <class T>
    __STATIC_INLINE__ std::vector<T> linspace(T start, T end, int num) {
        std::vector<T> result(num);
        if (num == 1) {
            result[0] = start;
            return result;
        }
        T step = (end - start) / (num - 1);
        for (int i = 0; i < num; ++i) {
            result[i] = start + i * step;
        }
        return result;
    }

    __STATIC_INLINE__ std::vector<std::vector<float>> transpose(const std::vector<std::vector<float>>& mat) {
        int rows = mat.size();
        int cols = mat[0].size();
        std::vector<std::vector<float>> transposed(cols, std::vector<float>(rows));
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                transposed[j][i] = mat[i][j];
            }
        }
        return transposed;
    }

    __STATIC_INLINE__ std::vector<float> flatten(const std::vector<std::vector<float>>& vec) {
        std::vector<float> flat_vec;
        for (const auto& sub_vec : vec) {
            flat_vec.insert(flat_vec.end(), sub_vec.begin(), sub_vec.end());
        }
        return flat_vec;
    }

    __STATIC_INLINE__ std::vector<std::vector<float>> rope(const std::vector<float>& pos, int dim, int theta) {
        assert(dim % 2 == 0);
        int half_dim = dim / 2;

        std::vector<float> scale = linspace(0.f, (dim * 1.f - 2) / dim, half_dim);

        std::vector<float> omega(half_dim);
        for (int i = 0; i < half_dim; ++i) {
            omega[i] = 1.0 / std::pow(theta, scale[i]);
        }

        int pos_size = pos.size();
        std::vector<std::vector<float>> out(pos_size, std::vector<float>(half_dim));
        for (int i = 0; i < pos_size; ++i) {
            for (int j = 0; j < half_dim; ++j) {
                out[i][j] = pos[i] * omega[j];
            }
        }

        std::vector<std::vector<float>> result(pos_size, std::vector<float>(half_dim * 4));
        for (int i = 0; i < pos_size; ++i) {
            for (int j = 0; j < half_dim; ++j) {
                result[i][4 * j]     = std::cos(out[i][j]);
                result[i][4 * j + 1] = -std::sin(out[i][j]);
                result[i][4 * j + 2] = std::sin(out[i][j]);
                result[i][4 * j + 3] = std::cos(out[i][j]);
            }
        }

        return result;
    }

    float find_correction_factor(float num_rotations, int dim, float base, float max_position_embeddings) {
        return (dim * std::log(max_position_embeddings / (num_rotations * 2 * 3.14159265358979323846))) / (2 * std::log(base));
    }

    std::pair<int, int> find_correction_range(float low_ratio, float high_ratio, int dim, float base, float ori_max_pe_len) {
        float low  = std::floor(find_correction_factor(low_ratio, dim, base, ori_max_pe_len));
        float high = std::ceil(find_correction_factor(high_ratio, dim, base, ori_max_pe_len));
        return {std::max(0, static_cast<int>(low)), std::min(dim / 2, static_cast<int>(high))};
    }

    std::vector<float> linear_ramp_mask(int min, int max, int dim) {
        if (min == max) {
            max += 0.001f;  // Prevent singularity
        }
        std::vector<float> ramp(dim);
        for (int i = 0; i < dim; ++i) {
            ramp[i] = std::max(0.0f, std::min(1.0f, static_cast<float>(i - min) / (max - min)));
        }
        return ramp;
    }

    __STATIC_INLINE__ std::vector<std::vector<float>> rope_ext(
        const std::vector<float>& pos,
        int dim,
        float theta                 = 10000.0f,
        bool use_real               = false,
        float linear_factor         = 1.0f,
        float ntk_factor            = 1.0f,
        bool repeat_interleave_real = true,
        bool yarn                   = false,
        int max_pe_len              = -1,
        int ori_max_pe_len          = 64,
        bool dype                   = false,
        float current_timestep      = 1.0f) {
        assert(dim % 2 == 0);
        int half_dim = dim / 2;

        // Compute frequencies
        std::vector<float> freqs_base(half_dim);
        std::vector<float> freqs_linear(half_dim);
        std::vector<float> freqs_ntk(half_dim);
        std::vector<float> freqs(half_dim);

        if (yarn && max_pe_len > ori_max_pe_len) {
            float beta_0  = 1.25f;
            float beta_1  = 0.75f;
            float gamma_0 = 16.0f;
            float gamma_1 = 2.0f;

            float scale = std::max(1.0f, static_cast<float>(max_pe_len) / ori_max_pe_len);
            // d,t,s
            float new_base = theta * std::pow(scale, half_dim / (half_dim - 1));
            for (int i = 0; i < half_dim; ++i) {
                float exponent  = static_cast<float>(i) / half_dim;
                freqs_base[i]   = 1.0f / std::pow(theta, exponent);
                freqs_linear[i] = 1.0f / (scale * std::pow(theta, exponent));
                freqs_ntk[i]    = 1.0f / std::pow(new_base, exponent);
            }

            if (dype) {
                beta_0  = std::pow(beta_0, 2.0f * current_timestep * current_timestep);
                beta_1  = std::pow(beta_1, 2.0f * current_timestep * current_timestep);
                gamma_0 = std::pow(gamma_0, 2.0f * current_timestep * current_timestep);
                gamma_1 = std::pow(gamma_1, 2.0f * current_timestep * current_timestep);
            }

            // Apply correction range and linear ramp mask
            auto [low, high] = find_correction_range(beta_0, beta_1, dim, theta, ori_max_pe_len);
            auto mask        = linear_ramp_mask(low, high, half_dim);
            for (int i = 0; i < half_dim; ++i) {
                freqs[i] = freqs_linear[i] * mask[i] + freqs_ntk[i] * (1.0f - mask[i]);
            }

            // Apply gamma correction
            auto [low_gamma, high_gamma] = find_correction_range(gamma_0, gamma_1, dim, theta, ori_max_pe_len);
            auto mask_gamma              = linear_ramp_mask(low_gamma, high_gamma, half_dim);
            for (int i = 0; i < half_dim; ++i) {
                freqs[i] = freqs[i] * mask_gamma[i] + freqs_base[i] * (1.0f - mask_gamma[i]);
            }
        } else {
            float theta_ntk = theta * ntk_factor;
            for (int i = 0; i < half_dim; ++i) {
                float exponent = static_cast<float>(i) / half_dim;
                freqs[i]       = 1.0f / std::pow(theta_ntk, exponent) / linear_factor;
            }
        }

        // Outer product of pos and freqs
        std::vector<std::vector<float>> freqs_outer(pos.size(), std::vector<float>(half_dim));
        for (size_t i = 0; i < pos.size(); ++i) {
            for (int j = 0; j < half_dim; ++j) {
                freqs_outer[i][j] = pos[i] * freqs[j];
            }
        }

        std::vector<std::vector<float>> result;
        result.resize(pos.size(), std::vector<float>(half_dim * 4));
        for (size_t i = 0; i < pos.size(); ++i) {
            for (int j = 0; j < half_dim; ++j) {
                result[i][4 * j]     = std::cos(freqs_outer[i][j]);   // cos
                result[i][4 * j + 1] = -std::sin(freqs_outer[i][j]);  // -sin
                result[i][4 * j + 2] = std::sin(freqs_outer[i][j]);   // sin
                result[i][4 * j + 3] = std::cos(freqs_outer[i][j]);   // cos
            }
        }

        return result;
    }

    // Generate IDs for image patches and text
    __STATIC_INLINE__ std::vector<std::vector<float>> gen_txt_ids(int bs, int context_len) {
        return std::vector<std::vector<float>>(bs * context_len, std::vector<float>(3, 0.0));
    }

    __STATIC_INLINE__ std::vector<std::vector<float>> gen_img_ids(int h, int w, int patch_size, int bs, int index = 0, int h_offset = 0, int w_offset = 0) {
        int h_len = (h + (patch_size / 2)) / patch_size;
        int w_len = (w + (patch_size / 2)) / patch_size;

        std::vector<std::vector<float>> img_ids(h_len * w_len, std::vector<float>(3, 0.0));

        std::vector<float> row_ids = linspace<float>(h_offset, h_len - 1 + h_offset, h_len);
        std::vector<float> col_ids = linspace<float>(w_offset, w_len - 1 + w_offset, w_len);

        for (int i = 0; i < h_len; ++i) {
            for (int j = 0; j < w_len; ++j) {
                img_ids[i * w_len + j][0] = index;
                img_ids[i * w_len + j][1] = row_ids[i];
                img_ids[i * w_len + j][2] = col_ids[j];
            }
        }

        std::vector<std::vector<float>> img_ids_repeated(bs * img_ids.size(), std::vector<float>(3));
        for (int i = 0; i < bs; ++i) {
            for (int j = 0; j < img_ids.size(); ++j) {
                img_ids_repeated[i * img_ids.size() + j] = img_ids[j];
            }
        }
        return img_ids_repeated;
    }

    __STATIC_INLINE__ std::vector<std::vector<float>> concat_ids(const std::vector<std::vector<float>>& a,
                                                                 const std::vector<std::vector<float>>& b,
                                                                 int bs) {
        size_t a_len = a.size() / bs;
        size_t b_len = b.size() / bs;
        std::vector<std::vector<float>> ids(a.size() + b.size(), std::vector<float>(3));
        for (int i = 0; i < bs; ++i) {
            for (int j = 0; j < a_len; ++j) {
                ids[i * (a_len + b_len) + j] = a[i * a_len + j];
            }
            for (int j = 0; j < b_len; ++j) {
                ids[i * (a_len + b_len) + a_len + j] = b[i * b_len + j];
            }
        }
        return ids;
    }

    __STATIC_INLINE__ std::vector<float> embed_nd(const std::vector<std::vector<float>>& ids,
                                                  int bs,
                                                  int theta,
                                                  const std::vector<int>& axes_dim) {
        std::vector<std::vector<float>> trans_ids = transpose(ids);
        size_t pos_len                            = ids.size() / bs;
        int num_axes                              = axes_dim.size();
        // for (int i = 0; i < pos_len; i++) {
        //     std::cout << trans_ids[0][i] << " " << trans_ids[1][i] << " " << trans_ids[2][i] << std::endl;
        // }

        int emb_dim = 0;
        for (int d : axes_dim)
            emb_dim += d / 2;

        std::vector<std::vector<float>> emb(bs * pos_len, std::vector<float>(emb_dim * 2 * 2, 0.0));
        int offset = 0;
        for (int i = 0; i < num_axes; ++i) {
            std::vector<std::vector<float>> rope_emb = rope(trans_ids[i], axes_dim[i], theta);  // [bs*pos_len, axes_dim[i]/2 * 2 * 2]
            for (int b = 0; b < bs; ++b) {
                for (int j = 0; j < pos_len; ++j) {
                    for (int k = 0; k < rope_emb[0].size(); ++k) {
                        emb[b * pos_len + j][offset + k] = rope_emb[j][k];
                    }
                }
            }
            offset += rope_emb[0].size();
        }

        return flatten(emb);
    }

    std::vector<float> embed_nd_ext(
        const std::vector<std::vector<float>>& ids,
        int bs,
        float theta,
        const std::vector<int>& axes_dim,
        bool yarn                      = false,
        std::vector<int> max_pe_len    = {},
        int ori_max_pe_len             = 64,
        bool dype                      = false,
        float current_timestep         = 1.0f,
        std::vector<float> ntk_factors = {}) {
        std::vector<std::vector<float>> trans_ids = transpose(ids);
        size_t pos_len                            = ids.size() / bs;
        int num_axes                              = axes_dim.size();

        if (ntk_factors.size() == 0) {
            ntk_factors = std::vector<float>(num_axes, 1.0f);
        }
        if (max_pe_len.size() == 0) {
            max_pe_len = std::vector<int>(num_axes, -1);
        }

        int emb_dim = 0;
        for (int d : axes_dim) {
            emb_dim += d;
        }

        std::vector<std::vector<float>> emb(bs * pos_len, std::vector<float>(emb_dim * 2, 0.0f));
        int offset = 0;

        for (int i = 0; i < num_axes; ++i) {
            std::vector<std::vector<float>> rope_emb = rope_ext(
                trans_ids[i], axes_dim[i], theta, false, 1.0f, ntk_factors[i], true, yarn, max_pe_len[i], ori_max_pe_len, dype, current_timestep);

            for (int b = 0; b < bs; ++b) {
                for (size_t j = 0; j < pos_len; ++j) {
                    for (size_t k = 0; k < rope_emb[j].size(); ++k) {
                        emb[b * pos_len + j][offset + k] = rope_emb[j][k];
                    }
                }
            }
            offset += static_cast<int>(axes_dim[i] * 2);
        }

        return flatten(emb);
    }

    __STATIC_INLINE__ std::vector<std::vector<float>> gen_refs_ids(int patch_size,
                                                                   int bs,
                                                                   const std::vector<ggml_tensor*>& ref_latents,
                                                                   bool increase_ref_index) {
        std::vector<std::vector<float>> ids;
        uint64_t curr_h_offset = 0;
        uint64_t curr_w_offset = 0;
        int index              = 1;
        for (ggml_tensor* ref : ref_latents) {
            uint64_t h_offset = 0;
            uint64_t w_offset = 0;
            if (!increase_ref_index) {
                if (ref->ne[1] + curr_h_offset > ref->ne[0] + curr_w_offset) {
                    w_offset = curr_w_offset;
                } else {
                    h_offset = curr_h_offset;
                }
            }

            auto ref_ids = gen_img_ids(ref->ne[1], ref->ne[0], patch_size, bs, index, h_offset, w_offset);
            ids          = concat_ids(ids, ref_ids, bs);

            if (increase_ref_index) {
                index++;
            }

            curr_h_offset = std::max(curr_h_offset, ref->ne[1] + h_offset);
            curr_w_offset = std::max(curr_w_offset, ref->ne[0] + w_offset);
        }
        return ids;
    }

    __STATIC_INLINE__ std::vector<std::vector<float>> gen_flux_ids(int h,
                                                                   int w,
                                                                   int patch_size,
                                                                   int bs,
                                                                   int context_len,
                                                                   const std::vector<ggml_tensor*>& ref_latents,
                                                                   bool increase_ref_index) {
        auto txt_ids = gen_txt_ids(bs, context_len);
        auto img_ids = gen_img_ids(h, w, patch_size, bs);

        auto ids = concat_ids(txt_ids, img_ids, bs);
        if (ref_latents.size() > 0) {
            auto refs_ids = gen_refs_ids(patch_size, bs, ref_latents, increase_ref_index);
            ids           = concat_ids(ids, refs_ids, bs);
        }
        return ids;
    }

    // Generate flux positional embeddings
    __STATIC_INLINE__ std::vector<float> gen_flux_pe(int h,
                                                     int w,
                                                     int patch_size,
                                                     int bs,
                                                     int context_len,
                                                     const std::vector<ggml_tensor*>& ref_latents,
                                                     bool increase_ref_index,
                                                     int theta,
                                                     const std::vector<int>& axes_dim,
                                                     bool use_yarn          = false,
                                                     bool use_dype          = false,
                                                     bool use_ntk           = false,
                                                     float current_timestep = 1.0f) {
        int base_resolution = 1024;
        // set it via environment variable for now (TODO: arg)
        const char* env_base_resolution = getenv("FLUX_DYPE_BASE_RESOLUTION");
        if (env_base_resolution != nullptr) {
            base_resolution = atoi(env_base_resolution);
        }
        int base_patches                    = base_resolution / 16;
        std::vector<std::vector<float>> ids = gen_flux_ids(h, w, patch_size, bs, context_len, ref_latents, increase_ref_index);
        std::vector<int> max_pos_vec        = {};
        std::vector<float> ntk_factor_vec   = {};
        for (int i = 0; i < axes_dim.size(); i++) {
            float max_pos_f = 0.0f;
            for (const auto& row : ids) {
                float val = row[i];
                if (val > max_pos_f) {
                    max_pos_f = val;
                }
            }
            int max_pos = static_cast<int>(max_pos_f) + 1;
            max_pos_vec.push_back(max_pos);
            float ntk_factor = 1.0f;
            if (use_ntk) {
                float base_ntk = pow((float)max_pos / base_patches, (float)axes_dim[i] / (axes_dim[i] - 2));
                ntk_factor     = use_dype ? pow(base_ntk, 2.0f * current_timestep * current_timestep) : base_ntk;
                ntk_factor     = std::max(1.0f, ntk_factor);
            }
            ntk_factor_vec.push_back(ntk_factor);
        }
        if (use_yarn || use_ntk) {
            return embed_nd_ext(ids, bs, theta, axes_dim, use_yarn, max_pos_vec, base_patches, use_dype, current_timestep, ntk_factor_vec);
        } else {
            return embed_nd(ids, bs, theta, axes_dim);
        }
    }

    __STATIC_INLINE__ std::vector<std::vector<float>> gen_qwen_image_ids(int h,
                                                                         int w,
                                                                         int patch_size,
                                                                         int bs,
                                                                         int context_len,
                                                                         const std::vector<ggml_tensor*>& ref_latents,
                                                                         bool increase_ref_index) {
        int h_len        = (h + (patch_size / 2)) / patch_size;
        int w_len        = (w + (patch_size / 2)) / patch_size;
        int txt_id_start = std::max(h_len, w_len);
        auto txt_ids     = linspace<float>(txt_id_start, context_len + txt_id_start, context_len);
        std::vector<std::vector<float>> txt_ids_repeated(bs * context_len, std::vector<float>(3));
        for (int i = 0; i < bs; ++i) {
            for (int j = 0; j < txt_ids.size(); ++j) {
                txt_ids_repeated[i * txt_ids.size() + j] = {txt_ids[j], txt_ids[j], txt_ids[j]};
            }
        }
        auto img_ids = gen_img_ids(h, w, patch_size, bs);
        auto ids     = concat_ids(txt_ids_repeated, img_ids, bs);
        if (ref_latents.size() > 0) {
            auto refs_ids = gen_refs_ids(patch_size, bs, ref_latents, increase_ref_index);
            ids           = concat_ids(ids, refs_ids, bs);
        }
        return ids;
    }

    // Generate qwen_image positional embeddings
    __STATIC_INLINE__ std::vector<float> gen_qwen_image_pe(int h,
                                                           int w,
                                                           int patch_size,
                                                           int bs,
                                                           int context_len,
                                                           const std::vector<ggml_tensor*>& ref_latents,
                                                           bool increase_ref_index,
                                                           int theta,
                                                           const std::vector<int>& axes_dim) {
        std::vector<std::vector<float>> ids = gen_qwen_image_ids(h, w, patch_size, bs, context_len, ref_latents, increase_ref_index);
        return embed_nd(ids, bs, theta, axes_dim);
    }

    __STATIC_INLINE__ std::vector<std::vector<float>> gen_vid_ids(int t,
                                                                  int h,
                                                                  int w,
                                                                  int pt,
                                                                  int ph,
                                                                  int pw,
                                                                  int bs,
                                                                  int t_offset = 0,
                                                                  int h_offset = 0,
                                                                  int w_offset = 0) {
        int t_len = (t + (pt / 2)) / pt;
        int h_len = (h + (ph / 2)) / ph;
        int w_len = (w + (pw / 2)) / pw;

        std::vector<std::vector<float>> vid_ids(t_len * h_len * w_len, std::vector<float>(3, 0.0));

        std::vector<float> t_ids = linspace<float>(t_offset, t_len - 1 + t_offset, t_len);
        std::vector<float> h_ids = linspace<float>(h_offset, h_len - 1 + h_offset, h_len);
        std::vector<float> w_ids = linspace<float>(w_offset, w_len - 1 + w_offset, w_len);

        for (int i = 0; i < t_len; ++i) {
            for (int j = 0; j < h_len; ++j) {
                for (int k = 0; k < w_len; ++k) {
                    int idx         = i * h_len * w_len + j * w_len + k;
                    vid_ids[idx][0] = t_ids[i];
                    vid_ids[idx][1] = h_ids[j];
                    vid_ids[idx][2] = w_ids[k];
                }
            }
        }

        std::vector<std::vector<float>> vid_ids_repeated(bs * vid_ids.size(), std::vector<float>(3));
        for (int i = 0; i < bs; ++i) {
            for (int j = 0; j < vid_ids.size(); ++j) {
                vid_ids_repeated[i * vid_ids.size() + j] = vid_ids[j];
            }
        }
        return vid_ids_repeated;
    }

    // Generate wan positional embeddings
    __STATIC_INLINE__ std::vector<float> gen_wan_pe(int t,
                                                    int h,
                                                    int w,
                                                    int pt,
                                                    int ph,
                                                    int pw,
                                                    int bs,
                                                    int theta,
                                                    const std::vector<int>& axes_dim) {
        std::vector<std::vector<float>> ids = gen_vid_ids(t, h, w, pt, ph, pw, bs);
        return embed_nd(ids, bs, theta, axes_dim);
    }

    __STATIC_INLINE__ std::vector<std::vector<float>> gen_qwen2vl_ids(int grid_h,
                                                                      int grid_w,
                                                                      int merge_size,
                                                                      const std::vector<int>& window_index) {
        std::vector<std::vector<float>> ids(grid_h * grid_w, std::vector<float>(2, 0.0));
        int index = 0;
        for (int ih = 0; ih < grid_h; ih += merge_size) {
            for (int iw = 0; iw < grid_w; iw += merge_size) {
                for (int iy = 0; iy < merge_size; iy++) {
                    for (int ix = 0; ix < merge_size; ix++) {
                        int inverse_index = window_index[index / (merge_size * merge_size)];
                        int i             = inverse_index * (merge_size * merge_size) + index % (merge_size * merge_size);

                        GGML_ASSERT(i < grid_h * grid_w);

                        ids[i][0] = ih + iy;
                        ids[i][1] = iw + ix;
                        index++;
                    }
                }
            }
        }
        return ids;
    }

    // Generate qwen2vl positional embeddings
    __STATIC_INLINE__ std::vector<float> gen_qwen2vl_pe(int grid_h,
                                                        int grid_w,
                                                        int merge_size,
                                                        const std::vector<int>& window_index,
                                                        int theta,
                                                        const std::vector<int>& axes_dim) {
        std::vector<std::vector<float>> ids = gen_qwen2vl_ids(grid_h, grid_w, merge_size, window_index);
        return embed_nd(ids, 1, theta, axes_dim);
    }

    __STATIC_INLINE__ struct ggml_tensor* apply_rope(struct ggml_context* ctx,
                                                     struct ggml_tensor* x,
                                                     struct ggml_tensor* pe,
                                                     bool rope_interleaved = true) {
        // x: [N, L, n_head, d_head]
        // pe: [L, d_head/2, 2, 2], [[cos, -sin], [sin, cos]]
        int64_t d_head = x->ne[0];
        int64_t n_head = x->ne[1];
        int64_t L      = x->ne[2];
        int64_t N      = x->ne[3];
        x              = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));  // [N, n_head, L, d_head]
        if (rope_interleaved) {
            x = ggml_reshape_4d(ctx, x, 2, d_head / 2, L, n_head * N);  // [N * n_head, L, d_head/2, 2]
            x = ggml_cont(ctx, ggml_permute(ctx, x, 3, 0, 1, 2));       // [2, N * n_head, L, d_head/2]
        } else {
            x = ggml_reshape_4d(ctx, x, d_head / 2, 2, L, n_head * N);       // [N * n_head, L, 2, d_head/2]
            x = ggml_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 3, 1));  // [2, N * n_head, L, d_head/2]
        }

        int64_t offset = x->nb[2] * x->ne[2];
        auto x_0       = ggml_view_3d(ctx, x, x->ne[0], x->ne[1], x->ne[2], x->nb[1], x->nb[2], offset * 0);  // [N * n_head, L, d_head/2]
        auto x_1       = ggml_view_3d(ctx, x, x->ne[0], x->ne[1], x->ne[2], x->nb[1], x->nb[2], offset * 1);  // [N * n_head, L, d_head/2]
        x_0            = ggml_reshape_4d(ctx, x_0, 1, x_0->ne[0], x_0->ne[1], x_0->ne[2]);                    // [N * n_head, L, d_head/2, 1]
        x_1            = ggml_reshape_4d(ctx, x_1, 1, x_1->ne[0], x_1->ne[1], x_1->ne[2]);                    // [N * n_head, L, d_head/2, 1]
        auto temp_x    = ggml_new_tensor_4d(ctx, x_0->type, 2, x_0->ne[1], x_0->ne[2], x_0->ne[3]);
        x_0            = ggml_repeat(ctx, x_0, temp_x);  // [N * n_head, L, d_head/2, 2]
        x_1            = ggml_repeat(ctx, x_1, temp_x);  // [N * n_head, L, d_head/2, 2]

        pe        = ggml_cont(ctx, ggml_permute(ctx, pe, 3, 0, 1, 2));  // [2, L, d_head/2, 2]
        offset    = pe->nb[2] * pe->ne[2];
        auto pe_0 = ggml_view_3d(ctx, pe, pe->ne[0], pe->ne[1], pe->ne[2], pe->nb[1], pe->nb[2], offset * 0);  // [L, d_head/2, 2]
        auto pe_1 = ggml_view_3d(ctx, pe, pe->ne[0], pe->ne[1], pe->ne[2], pe->nb[1], pe->nb[2], offset * 1);  // [L, d_head/2, 2]

        auto x_out = ggml_add_inplace(ctx, ggml_mul(ctx, x_0, pe_0), ggml_mul(ctx, x_1, pe_1));  // [N * n_head, L, d_head/2, 2]
        if (!rope_interleaved) {
            x_out = ggml_cont(ctx, ggml_permute(ctx, x_out, 1, 0, 2, 3));  // [N * n_head, L, x, d_head/2]
        }
        x_out = ggml_reshape_3d(ctx, x_out, d_head, L, n_head * N);  // [N*n_head, L, d_head]
        return x_out;
    }

    __STATIC_INLINE__ struct ggml_tensor* attention(GGMLRunnerContext* ctx,
                                                    struct ggml_tensor* q,
                                                    struct ggml_tensor* k,
                                                    struct ggml_tensor* v,
                                                    struct ggml_tensor* pe,
                                                    struct ggml_tensor* mask,
                                                    float kv_scale        = 1.0f,
                                                    bool rope_interleaved = true) {
        // q,k,v: [N, L, n_head, d_head]
        // pe: [L, d_head/2, 2, 2]
        // return: [N, L, n_head*d_head]
        q = apply_rope(ctx->ggml_ctx, q, pe, rope_interleaved);  // [N*n_head, L, d_head]
        k = apply_rope(ctx->ggml_ctx, k, pe, rope_interleaved);  // [N*n_head, L, d_head]

        auto x = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v, v->ne[1], mask, false, true, ctx->flash_attn_enabled, kv_scale);  // [N, L, n_head*d_head]
        return x;
    }
};  // namespace Rope

#endif  // __ROPE_HPP__
