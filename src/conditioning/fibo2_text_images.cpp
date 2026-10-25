#include "conditioning/fibo2_text_images.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>

#include "core/util.h"
#include "utf8proc/utf8proc.h"

#ifdef SD_USE_FREETYPE
#include <ft2build.h>
#include FT_FREETYPE_H

namespace sd {
    static constexpr size_t max_text_image_pixels = 16777216;

    struct FiboTextFonts {
        FT_Library library = nullptr;
        std::vector<std::vector<uint8_t>> data;
        std::vector<FT_Face> faces;

        ~FiboTextFonts() {
            for (auto face : faces)
                FT_Done_Face(face);
            if (library)
                FT_Done_FreeType(library);
        }
    };

    struct FiboTextGlyph {
        int x, y, width, height;
        std::vector<uint8_t> coverage;
        bool run_start = false;
    };

    struct FiboTextLine {
        int left = 0, top = 0, width = 1, height = 0;
        std::vector<FiboTextGlyph> glyphs;
    };

    static bool text_space(utf8proc_int32_t ch) {
        return (ch >= 9 && ch <= 13) || (ch >= 28 && ch <= 32) || ch == 0x85 ||
               ch == 0xa0 || ch == 0x1680 || (ch >= 0x2000 && ch <= 0x200a) ||
               ch == 0x2028 || ch == 0x2029 || ch == 0x202f || ch == 0x205f || ch == 0x3000;
    }

    static bool text_characters(const std::string& text, std::vector<utf8proc_int32_t>& characters) {
        bool space = false;
        for (size_t offset = 0; offset < text.size();) {
            utf8proc_int32_t ch;
            auto length = utf8proc_iterate(reinterpret_cast<const utf8proc_uint8_t*>(text.data() + offset), text.size() - offset, &ch);
            if (length <= 0) {
                LOG_ERROR("Fibo text_render contains invalid UTF-8");
                return false;
            }
            offset += length;
            if (text_space(ch)) {
                space = !characters.empty();
            } else {
                if (space)
                    characters.push_back(' ');
                characters.push_back(ch);
                space = false;
            }
        }
        return true;
    }

    static FT_Face text_face(const FiboTextFonts& fonts, utf8proc_int32_t ch) {
        if (ch < 128)
            return fonts.faces.front();
        for (auto face : fonts.faces) {
            if (FT_Get_Char_Index(face, ch) != 0)
                return face;
        }
        return fonts.faces.front();
    }

    static bool text_line(const std::vector<utf8proc_int32_t>& characters,
                          size_t first,
                          size_t last,
                          const FiboTextFonts& fonts,
                          bool mixed,
                          size_t& remaining_coverage,
                          FiboTextLine& line) {
        FT_Pos pen            = 0;
        FT_Face previous_face = nullptr;
        FT_UInt previous      = 0;
        int right = 0, top = std::numeric_limits<int>::max(), bottom = std::numeric_limits<int>::min();
        for (size_t i = first; i < last; ++i) {
            auto face  = text_face(fonts, characters[i]);
            auto index = FT_Get_Char_Index(face, characters[i]);
            if (previous_face == face && previous && index && FT_HAS_KERNING(face)) {
                FT_Vector kerning{};
                FT_Get_Kerning(face, previous, index, FT_KERNING_DEFAULT, &kerning);
                // Pillow BASIC adds the rounded pixel delta to a 26.6 advance.
                pen += (kerning.x + 32) >> 6;
            }
            if (FT_Load_Glyph(face, index, FT_LOAD_RENDER)) {
                LOG_ERROR("could not render Fibo text glyph U+%04X", characters[i]);
                return false;
            }
            const auto& bitmap = face->glyph->bitmap;
            if (bitmap.pixel_mode != FT_PIXEL_MODE_GRAY && bitmap.width && bitmap.rows) {
                LOG_ERROR("Fibo text rendering requires grayscale outline fonts");
                return false;
            }
            if (bitmap.width > max_text_image_pixels || bitmap.rows > max_text_image_pixels ||
                (bitmap.rows && bitmap.width > remaining_coverage / bitmap.rows)) {
                LOG_ERROR("Fibo text glyph coverage exceeds the maximum area of %zu pixels", max_text_image_pixels);
                return false;
            }
            size_t glyph_pixels = size_t(bitmap.width) * bitmap.rows;
            remaining_coverage -= glyph_pixels;
            FiboTextGlyph glyph{static_cast<int>((pen + 32) >> 6) + face->glyph->bitmap_left,
                                -face->glyph->bitmap_top,
                                static_cast<int>(bitmap.width),
                                static_cast<int>(bitmap.rows),
                                {}};
            glyph.run_start = previous_face != face;
            glyph.coverage.resize(glyph_pixels);
            for (int y = 0; y < glyph.height; ++y) {
                int row = bitmap.pitch < 0 ? glyph.height - 1 - y : y;
                std::copy_n(bitmap.buffer + row * std::abs(bitmap.pitch), glyph.width, glyph.coverage.data() + size_t(y) * glyph.width);
            }
            line.left = std::min(line.left, glyph.x);
            right     = std::max(right, glyph.x + glyph.width);
            if (glyph.height) {
                top    = std::min(top, glyph.y);
                bottom = std::max(bottom, glyph.y + glyph.height);
            }
            pen += face->glyph->advance.x;
            right = std::max(right, static_cast<int>((pen + 32) >> 6));
            line.glyphs.push_back(std::move(glyph));
            previous_face = face;
            previous      = index;
        }
        line.top    = top == std::numeric_limits<int>::max() ? 0 : top;
        line.width  = std::max(1, mixed ? static_cast<int>(std::nearbyint(pen / 64.0)) : right - line.left);
        line.height = top == std::numeric_limits<int>::max() ? 0 : bottom - top;
        return true;
    }

    static int text_multiply_alpha(int value, int alpha) {
        int product = value * alpha + 128;
        return (product + (product >> 8)) >> 8;
    }
}
#endif

bool sd::render_fibo2_text_images(const std::vector<std::string>& texts,
                                  const std::vector<std::string>& font_paths,
                                  std::vector<Tensor<float>>& images) {
    images.clear();
    if (texts.empty())
        return true;
#ifndef SD_USE_FREETYPE
    LOG_ERROR("Fibo text rendering requires a build with SD_FREETYPE=ON");
    return false;
#else
    if (font_paths.empty()) {
        LOG_ERROR("Fibo text rendering requires text_render_fonts (Aileron, then optional fallback fonts)");
        return false;
    }
    FiboTextFonts fonts;
    if (FT_Init_FreeType(&fonts.library)) {
        LOG_ERROR("could not initialize FreeType");
        return false;
    }
    for (const auto& path : font_paths) {
        if (path.empty()) {
            LOG_ERROR("Fibo text_render_fonts contains an empty font path");
            return false;
        }
        std::ifstream file(std::filesystem::u8path(path), std::ios::binary);
        if (!file) {
            LOG_ERROR("could not read text-rendering font '%s'", path.c_str());
            return false;
        }
        fonts.data.emplace_back(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        const auto& data = fonts.data.back();
        FT_Face face     = nullptr;
        if (data.size() > static_cast<size_t>(std::numeric_limits<FT_Long>::max()) ||
            FT_New_Memory_Face(fonts.library, data.data(), static_cast<FT_Long>(data.size()), 0, &face)) {
            LOG_ERROR("could not open text-rendering font '%s'", path.c_str());
            return false;
        }
        fonts.faces.push_back(face);
        if (FT_Set_Pixel_Sizes(face, 0, 20)) {
            LOG_ERROR("could not set text-rendering font size");
            return false;
        }
    }
    FiboTextLine minimum;
    size_t minimum_coverage = max_text_image_pixels;
    if (!text_line({'A', 'g'}, 0, 2, fonts, false, minimum_coverage, minimum))
        return false;
    size_t remaining_pixels = max_text_image_pixels;
    for (const auto& text : texts) {
        std::vector<utf8proc_int32_t> characters;
        if (!text_characters(text, characters))
            return false;
        bool mixed = std::any_of(characters.begin(), characters.end(), [&](auto ch) {
            return ch >= 128 && FT_Get_Char_Index(fonts.faces.front(), ch) == 0;
        });
        std::vector<FiboTextLine> lines;
        size_t remaining_coverage = max_text_image_pixels;
        int width = 1, height = 0;
        for (size_t first = 0; first < characters.size() || lines.empty(); first += 256) {
            FiboTextLine line;
            if (!text_line(characters, first, std::min(first + 256, characters.size()), fonts, mixed, remaining_coverage, line))
                return false;
            if (characters.empty()) {
                FiboTextLine space;
                if (!text_line({' '}, 0, 1, fonts, false, remaining_coverage, space))
                    return false;
                line.width = space.width;
            }
            line.height = mixed ? static_cast<int>((fonts.faces.front()->size->metrics.ascender - fonts.faces.front()->size->metrics.descender) >> 6)
                                : std::max(minimum.height, line.height);
            width       = std::max(width, line.width);
            height += line.height + (lines.empty() ? 0 : 4);
            if ((int64_t(width) + 4) * (int64_t(height) + 4) > int64_t(remaining_pixels)) {
                LOG_ERROR("Fibo text images exceed the maximum total area of %zu pixels", max_text_image_pixels);
                return false;
            }
            lines.push_back(std::move(line));
        }
        remaining_pixels -= size_t(width + 4) * (height + 4);
        auto image = sd::full<float>({width + 4, height + 4, 3, 1}, 1.f);
        int row    = 2;
        for (const auto& line : lines) {
            int x_offset = mixed ? 2 : 2 - line.left;
            int baseline = mixed ? row + static_cast<int>(fonts.faces.front()->size->metrics.ascender >> 6) : row - line.top;
            for (size_t first = 0; first < line.glyphs.size();) {
                size_t last = first + 1;
                while (last < line.glyphs.size() && !line.glyphs[last].run_start)
                    ++last;
                int left = width + 4, top = height + 4, right = 0, bottom = 0;
                for (size_t i = first; i < last; ++i) {
                    const auto& glyph = line.glyphs[i];
                    if (glyph.width && glyph.height) {
                        left   = std::min(left, glyph.x);
                        top    = std::min(top, glyph.y);
                        right  = std::max(right, glyph.x + glyph.width);
                        bottom = std::max(bottom, glyph.y + glyph.height);
                    }
                }
                int run_width = std::max(0, right - left), run_height = std::max(0, bottom - top);
                std::vector<uint8_t> coverage(size_t(run_width) * run_height, 0);
                // Pillow composites glyphs into a mask per font run, rounding each blend to 8 bits.
                for (size_t i = first; i < last; ++i) {
                    const auto& glyph = line.glyphs[i];
                    for (int y = 0; y < glyph.height; ++y) {
                        for (int x = 0; x < glyph.width; ++x) {
                            auto& pixel = coverage[size_t(glyph.y - top + y) * run_width + glyph.x - left + x];
                            int alpha   = glyph.coverage[size_t(y) * glyph.width + x];
                            pixel       = static_cast<uint8_t>(alpha + text_multiply_alpha(pixel, 255 - alpha));
                        }
                    }
                }
                for (int y = 0; y < run_height; ++y) {
                    for (int x = 0; x < run_width; ++x) {
                        int target_x = x_offset + left + x, target_y = baseline + top + y;
                        if (target_x < 0 || target_x >= width + 4 || target_y < 0 || target_y >= height + 4)
                            continue;
                        int pixel   = static_cast<int>(std::lround(image.index(target_x, target_y, 0, 0) * 255.f));
                        float value = text_multiply_alpha(pixel, 255 - coverage[size_t(y) * run_width + x]) / 255.f;
                        for (int channel = 0; channel < 3; ++channel) {
                            image.index(target_x, target_y, channel, 0) = value;
                        }
                    }
                }
                first = last;
            }
            row += line.height + 4;
        }
        images.push_back(std::move(image));
    }
    return true;
#endif
}
