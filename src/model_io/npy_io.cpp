#include "npy_io.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstring>
#include <limits>
#include <optional>
#include <string_view>

namespace sd::npy {

    static constexpr uint8_t npy_magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    static constexpr struct {
        std::string_view code;
        sd_tensor_type_t type;
        size_t width;
    } npy_types[] = {
        {"u1", SD_TENSOR_U8, 1},
        {"i4", SD_TENSOR_I32, 4},
        {"i8", SD_TENSOR_I64, 8},
        {"f4", SD_TENSOR_F32, 4},
        {"f8", SD_TENSOR_F64, 8},
    };

    static char native_byte_order() {
        const uint16_t one = 1;
        return *reinterpret_cast<const uint8_t*>(&one) ? '<' : '>';
    }

    struct NpyHeader {
        std::string_view descriptor;
        std::vector<int64_t> shape;
        bool fortran_order;
    };

    class NpyHeaderParser {
        std::string_view input_;

        void skip_whitespace() {
            const auto first = input_.find_first_not_of(" \t\r\n");
            input_.remove_prefix(first == std::string_view::npos ? input_.size() : first);
        }

        bool consume(char token) {
            skip_whitespace();
            if (input_.empty() || input_.front() != token)
                return false;
            input_.remove_prefix(1);
            return true;
        }

        void require(char token) {
            if (!consume(token))
                throw std::runtime_error("invalid NumPy tensor header");
        }

        std::string_view read_string() {
            skip_whitespace();
            if (input_.empty() || (input_.front() != '\'' && input_.front() != '"'))
                throw std::runtime_error("expected a string in NumPy tensor header");
            const auto closing_quote = input_.find(input_.front(), 1);
            if (closing_quote == std::string_view::npos)
                throw std::runtime_error("unterminated string in NumPy tensor header");
            const auto value = input_.substr(1, closing_quote - 1);
            input_.remove_prefix(closing_quote + 1);
            return value;
        }

        bool read_bool() {
            skip_whitespace();
            if (input_.substr(0, 4) == "True") {
                input_.remove_prefix(4);
                return true;
            }
            if (input_.substr(0, 5) == "False") {
                input_.remove_prefix(5);
                return false;
            }
            throw std::runtime_error("invalid fortran_order in NumPy tensor header");
        }

        std::vector<int64_t> read_shape() {
            require('(');
            std::vector<int64_t> shape;
            if (consume(')'))
                return shape;
            do {
                skip_whitespace();
                int64_t dimension = 0;
                const auto result = std::from_chars(input_.data(), input_.data() + input_.size(), dimension);
                if (result.ec != std::errc{} || dimension < 0 || shape.size() == 8)
                    throw std::runtime_error("invalid NumPy tensor shape (maximum rank 8)");
                input_.remove_prefix(result.ptr - input_.data());
                shape.push_back(dimension);
                if (!consume(',')) {
                    if (shape.size() == 1)
                        throw std::runtime_error("expected a shape tuple in NumPy tensor header");
                    require(')');
                    break;
                }
            } while (!consume(')'));
            return shape;
        }

    public:
        explicit NpyHeaderParser(std::string_view input)
            : input_(input) {}

        NpyHeader parse() {
            std::optional<std::string_view> descriptor;
            std::optional<std::vector<int64_t>> shape;
            std::optional<bool> fortran_order;
            require('{');
            for (int i = 0; i < 3; ++i) {
                if (i != 0)
                    require(',');
                const auto key = read_string();
                require(':');
                if (key == "descr" && !descriptor)
                    descriptor = read_string();
                else if (key == "shape" && !shape)
                    shape = read_shape();
                else if (key == "fortran_order" && !fortran_order)
                    fortran_order = read_bool();
                else
                    throw std::runtime_error("unexpected or duplicate key in NumPy tensor header");
            }
            consume(',');
            require('}');
            skip_whitespace();
            if (!input_.empty())
                throw std::runtime_error("unexpected data in NumPy tensor header");
            std::reverse(shape->begin(), shape->end());
            return {*descriptor, std::move(*shape), *fortran_order};
        }
    };

    static void copy_fortran_order(uint8_t* target, const uint8_t* source, const std::vector<int64_t>& shape, size_t bytes, size_t width) {
        std::array<size_t, 8> strides{};
        std::array<size_t, 8> indices{};
        size_t stride = width;
        for (size_t i = shape.size(); i > 0; --i) {
            strides[i - 1] = stride;
            stride *= size_t(shape[i - 1]);
        }

        // DGL stores axis zero contiguously. Fortran data makes the opposite axis contiguous.
        size_t offset = 0;
        for (size_t pos = 0; pos < bytes; pos += width) {
            std::memcpy(target + pos, source + offset, width);
            for (size_t axis = 0; axis < shape.size(); ++axis) {
                offset += strides[axis];
                if (++indices[axis] < size_t(shape[axis]))
                    break;
                offset -= indices[axis] * strides[axis];
                indices[axis] = 0;
            }
        }
    }

    static size_t checked_payload_size(const std::vector<int64_t>& shape, size_t width) {
        size_t count = 1;
        for (int64_t dim : shape) {
            if (count && uint64_t(dim) > std::numeric_limits<size_t>::max() / count)
                throw std::runtime_error("invalid or overflowing tensor shape");
            count *= size_t(dim);
        }
        if (count > uint64_t(INT64_MAX) || count > (std::numeric_limits<size_t>::max() - 7) / width)
            throw std::runtime_error("tensor byte size overflow");
        return count * width;
    }

    Reader::Reader(const std::string& path, size_t max_bytes)
        : file_(std::filesystem::u8path(path), std::ios::binary | std::ios::ate) {
        if (!file_)
            throw std::runtime_error("cannot open tensor input '" + path + "'");
        const std::streamoff file_size = file_.tellg();
        file_.seekg(0);
        uint8_t prefix[8];
        if (!file_.read(reinterpret_cast<char*>(prefix), sizeof(prefix)) ||
            !std::equal(std::begin(npy_magic), std::end(npy_magic), prefix))
            throw std::runtime_error("tensor input is not a NumPy .npy file");
        if (prefix[6] < 1 || prefix[6] > 3 || prefix[7] != 0)
            throw std::runtime_error("unsupported NumPy tensor format version");
        const size_t length_size = prefix[6] == 1 ? 2 : 4;
        uint8_t length[4]{};
        if (!file_.read(reinterpret_cast<char*>(length), length_size))
            throw std::runtime_error("cannot read NumPy tensor header length");
        const uint32_t header_size = uint32_t(length[0]) | (uint32_t(length[1]) << 8) |
                                     (uint32_t(length[2]) << 16) | (uint32_t(length[3]) << 24);
        if (header_size == 0 || header_size > 64 * 1024)
            throw std::runtime_error("invalid NumPy tensor header length");
        std::string header_text(header_size, '\0');
        if (!file_.read(header_text.data(), header_text.size()) || header_text.back() != '\n')
            throw std::runtime_error("cannot read NumPy tensor header");
        auto header = NpyHeaderParser(header_text).parse();

        const char order = header.descriptor.empty() ? '\0' : header.descriptor.front();
        const auto code  = header.descriptor.empty() ? header.descriptor : header.descriptor.substr(1);
        const auto dtype = std::find_if(std::begin(npy_types), std::end(npy_types), [&](const auto& entry) {
            return entry.code == code;
        });
        if (dtype == std::end(npy_types) || (order != '<' && order != '>' && order != '=' && !(order == '|' && dtype->width == 1)))
            throw std::runtime_error("unsupported NumPy tensor dtype '" + std::string(header.descriptor) + "'");

        const auto bytes = checked_payload_size(header.shape, dtype->width);
        if (bytes > max_bytes)
            throw std::runtime_error("NumPy tensor exceeds the byte limit (" + std::to_string(max_bytes) + " bytes)");
        const auto payload = std::streamoff(sizeof(prefix) + length_size + header_size);
        if (file_size < payload || uint64_t(file_size - payload) != bytes)
            throw std::runtime_error("NumPy tensor payload size does not match its shape");

        shape_         = std::move(header.shape);
        type_          = dtype->type;
        bytes_         = bytes;
        width_         = dtype->width;
        fortran_order_ = header.fortran_order;
        swap_bytes_    = width_ > 1 && order != '=' && order != native_byte_order();
    }

    void Reader::read(void* buffer, size_t bytes) {
        if (bytes != bytes_ || (bytes && !buffer))
            throw std::runtime_error("NumPy tensor destination size does not match its shape");
        if (!bytes_)
            return;
        auto* target = static_cast<uint8_t*>(buffer);
        if (fortran_order_ && shape_.size() > 1) {
            std::vector<uint8_t> source(bytes_);
            if (!file_.read(reinterpret_cast<char*>(source.data()), bytes_))
                throw std::runtime_error("cannot read NumPy tensor payload");
            copy_fortran_order(target, source.data(), shape_, bytes_, width_);
        } else if (!file_.read(reinterpret_cast<char*>(target), bytes_)) {
            throw std::runtime_error("cannot read NumPy tensor payload");
        }
        if (swap_bytes_) {
            for (size_t offset = 0; offset < bytes_; offset += width_)
                std::reverse(target + offset, target + offset + width_);
        }
    }

    void write(const std::filesystem::path& path, sd_tensor_type_t type, const std::vector<int64_t>& shape, const void* data, size_t bytes) {
        const auto dtype = std::find_if(std::begin(npy_types), std::end(npy_types), [&](const auto& entry) {
            return entry.type == type;
        });
        if (dtype == std::end(npy_types))
            throw std::runtime_error("invalid tensor type");
        std::string descriptor(1, dtype->width == 1 ? '|' : native_byte_order());
        descriptor += dtype->code;
        std::string header = "{'descr': '" + descriptor + "', 'fortran_order': False, 'shape': (";
        for (auto it = shape.rbegin(); it != shape.rend(); ++it)
            header += std::to_string(*it) + ", ";
        header += "), }";
        header.append((64 - (10 + header.size() + 1) % 64) % 64, ' ');
        header += '\n';

        std::ofstream file(path, std::ios::binary);
        if (!file)
            throw std::runtime_error("cannot open tensor output '" + path.u8string() + "'");
        const uint8_t format[] = {1, 0, uint8_t(header.size()), uint8_t(header.size() >> 8)};
        file.write(reinterpret_cast<const char*>(npy_magic), sizeof(npy_magic));
        file.write(reinterpret_cast<const char*>(format), sizeof(format));
        file.write(header.data(), header.size());
        if (bytes)
            file.write(static_cast<const char*>(data), bytes);
        file.close();
        if (!file)
            throw std::runtime_error("cannot write tensor output '" + path.u8string() + "'");
    }

}  // namespace sd::npy
