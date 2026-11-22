#include "api/metadata.h"

#include <cstring>
#include <limits>
#include <stdexcept>

#include "api/api.h"

namespace sd::metadata {
    static std::string xml_text(const std::string& info) {
        std::string escaped;
        for (char c : info) {
            switch (c) {
                case '&':
                    escaped += "&amp;";
                    break;
                case '<':
                    escaped += "&lt;";
                    break;
                case '>':
                    escaped += "&gt;";
                    break;
                case '"':
                    escaped += "&quot;";
                    break;
                case '\'':
                    escaped += "&apos;";
                    break;
                default:
                    escaped += c;
                    break;
            }
        }
        return escaped;
    }

    std::string generation_xmp(const std::string& source, const std::string& parameters) {
        if (source.empty())
            return "";
        return "<x:xmpmeta xmlns:x=\"adobe:ns:meta/\"><rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">"
               "<rdf:Description rdf:about=\"\" xmlns:sdcpp=\"https://github.com/leejet/stable-diffusion.cpp/ns/1.0/\">"
               "<sdcpp:dgl>" +
               xml_text(source) + "</sdcpp:dgl><sdcpp:parameters>" + xml_text(parameters) +
               "</sdcpp:parameters></rdf:Description></rdf:RDF></x:xmpmeta>";
    }

    static bool jpeg_comment(std::vector<uint8_t>& comments, const std::string& label, const std::string& value) {
        if (value.empty())
            return true;
        const size_t size = label.size() + 1 + value.size() + 1;
        if (size > jpeg_comment_max_bytes)
            return false;
        const size_t length = size + 2;
        std::vector<uint8_t> segment{0xff, 0xfe, uint8_t(length >> 8), uint8_t(length)};
        segment.insert(segment.end(), label.begin(), label.end());
        segment.push_back(0);
        segment.insert(segment.end(), value.begin(), value.end());
        segment.push_back(0);
        comments.insert(comments.end(), segment.begin(), segment.end());
        return true;
    }

    bool jpeg_metadata(std::vector<uint8_t>& comments, const std::string& source, const std::string& parameters) {
        return source.empty() || (jpeg_comment(comments, "parameters", parameters) && jpeg_comment(comments, "dgl", source));
    }

    bool embed_jpeg_metadata(std::vector<uint8_t>& image, const std::string& source, const std::string& parameters) {
        if (source.empty())
            return true;
        std::vector<uint8_t> comments;
        if (image.size() < 2 || !jpeg_metadata(comments, source, parameters))
            return false;
        image.insert(image.begin() + 2, comments.begin(), comments.end());
        return true;
    }

    static void append_u32(std::vector<uint8_t>& bytes, uint32_t value) {
        for (int i = 3; i >= 0; --i)
            bytes.push_back(uint8_t(value >> (8 * i)));
    }

    static bool png_text(std::vector<uint8_t>& chunks, const std::string& label, const std::string& value) {
        if (value.empty())
            return true;
        const size_t size = label.size() + 5 + value.size();
        if (size > std::numeric_limits<uint32_t>::max())
            return false;
        append_u32(chunks, uint32_t(size));
        const size_t start = chunks.size();
        chunks.insert(chunks.end(), {'i', 'T', 'X', 't'});
        chunks.insert(chunks.end(), label.begin(), label.end());
        chunks.insert(chunks.end(), 5, 0);
        chunks.insert(chunks.end(), value.begin(), value.end());
        uint32_t crc = 0xffffffff;
        for (size_t i = start; i < chunks.size(); ++i) {
            crc ^= chunks[i];
            for (int bit = 0; bit < 8; ++bit)
                crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1)));
        }
        append_u32(chunks, ~crc);
        return true;
    }

    bool png_metadata(std::vector<uint8_t>& chunks, const std::string& source, const std::string& parameters) {
        return source.empty() || (png_text(chunks, "dgl", source) && png_text(chunks, "parameters", parameters));
    }

    bool embed_png_metadata(std::vector<uint8_t>& image, const std::string& source, const std::string& parameters) {
        if (source.empty())
            return true;
        if (image.size() < 33 || std::memcmp(image.data() + 12, "IHDR", 4))
            return false;
        std::vector<uint8_t> chunks;
        if (!png_metadata(chunks, source, parameters))
            return false;
        image.insert(image.begin() + 33, chunks.begin(), chunks.end());
        return true;
    }

    std::vector<uint8_t> avi_info(const std::string& source) {
        if (source.empty())
            return {};
        if (source.size() > std::numeric_limits<uint32_t>::max() - 22)
            throw std::runtime_error("generation metadata exceeds AVI container capacity");
        const auto size = uint32_t(source.size() + 1);
        std::vector<uint8_t> bytes{'L', 'I', 'S', 'T'};
        auto append_le = [&](uint32_t value) {
            for (int i = 0; i < 4; ++i)
                bytes.push_back(uint8_t(value >> (8 * i)));
        };
        append_le(12 + size + (size & 1));
        bytes.insert(bytes.end(), {'I', 'N', 'F', 'O', 'I', 'C', 'M', 'T'});
        append_le(size);
        bytes.insert(bytes.end(), source.begin(), source.end());
        bytes.push_back(0);
        if (size & 1)
            bytes.push_back(0);
        return bytes;
    }
}

static thread_local std::vector<uint8_t> metadata_chunk_bytes;

sd_status_t sd_get_metadata_chunk(const char* source, sd_metadata_format_t format, const void** data_out, size_t* size_out) {
    if (data_out)
        *data_out = nullptr;
    if (size_out)
        *size_out = 0;
    return sd::api::call([&] {
        if (!source || !data_out || !size_out)
            return sd::api::error(SD_STATUS_INVALID_ARGUMENT, nullptr, "source, data_out and size_out are required");
        auto& bytes = metadata_chunk_bytes;
        bytes.clear();
        if (format == SD_METADATA_AVI) {
            bytes = sd::metadata::avi_info(source);
        } else {
            if (format != SD_METADATA_PNG && format != SD_METADATA_JPEG && format != SD_METADATA_XMP)
                return sd::api::error(SD_STATUS_INVALID_ARGUMENT, nullptr, "unknown metadata format");
            const auto info       = sd::api::generation_metadata(source);
            const auto parameters = info.value("infotext", "");
            if (format == SD_METADATA_XMP) {
                const auto text = sd::metadata::generation_xmp(source, parameters);
                bytes.assign(text.begin(), text.end());
            } else if (!(format == SD_METADATA_PNG ? sd::metadata::png_metadata(bytes, source, parameters)
                                                   : sd::metadata::jpeg_metadata(bytes, source, parameters))) {
                return sd::api::error(SD_STATUS_OUT_OF_RANGE, nullptr, "generation metadata exceeds container capacity");
            }
        }
        *data_out = bytes.empty() ? nullptr : bytes.data();
        *size_out = bytes.size();
        return SD_STATUS_OK;
    });
}
