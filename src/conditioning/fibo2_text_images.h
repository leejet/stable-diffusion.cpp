#ifndef __SD_CONDITIONING_FIBO2_TEXT_IMAGES_H__
#define __SD_CONDITIONING_FIBO2_TEXT_IMAGES_H__

#include <string>
#include <vector>

#include "core/tensor.hpp"

namespace sd {
    bool render_fibo2_text_images(const std::vector<std::string>& texts,
                                  const std::vector<std::string>& font_paths,
                                  std::vector<Tensor<float>>& images);
}

#endif  // __SD_CONDITIONING_FIBO2_TEXT_IMAGES_H__
