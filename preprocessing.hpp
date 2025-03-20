#ifndef __PREPROCESSING_HPP__
#define __PREPROCESSING_HPP__

#include "ggml_extend.hpp"

void convolve(struct ggml_tensor* input, struct ggml_tensor* output, struct ggml_tensor* kernel, int padding);

void gaussian_kernel(struct ggml_tensor* kernel);

void grayscale(struct ggml_tensor* rgb_img, struct ggml_tensor* grayscale);

void prop_hypot(struct ggml_tensor* x, struct ggml_tensor* y, struct ggml_tensor* h);

void prop_arctan2(struct ggml_tensor* x, struct ggml_tensor* y, struct ggml_tensor* h);

void normalize_tensor(struct ggml_tensor* g);

void non_max_supression(struct ggml_tensor* result, struct ggml_tensor* G, struct ggml_tensor* D);

void threshold_hystersis(struct ggml_tensor* img, float high_threshold, float low_threshold, float weak, float strong);

uint8_t* preprocess_canny(uint8_t* img, int width, int height, float high_threshold, float low_threshold, float weak, float strong, bool inverse);

#endif  // __PREPROCESSING_HPP__
