#include "lexer.h"

#include <cctype>
#include <stdexcept>

#include "dgl/version.h"

namespace sd::dgl {

    [[noreturn]] static void fail(int line, int column, const std::string& message) {
        throw std::runtime_error("line " + std::to_string(line) + ", column " + std::to_string(column) + ": " + message);
    }

    Lexer::Lexer(const std::string& source)
        : source_(source) {
        if (source_.compare(0, 3, "\xef\xbb\xbf") == 0)
            offset_ = 3;
    }

    void Lexer::comment(size_t begin, int line, int column) {
        if (source_[begin] != '#')
            return;
        std::string text = source_.substr(begin + 1, offset_ - begin - 1);
        size_t key       = text.find_first_not_of(" \t\r");
        if (key == std::string::npos || text.compare(key, 7, "version") != 0)
            return;
        size_t colon = text.find_first_not_of(" \t", key + 7);
        if (colon == std::string::npos || text[colon] != ':')
            return;
        if (seen_token_)
            fail(line, column, "version header must precede all code");
        if (seen_version_)
            fail(line, column, "duplicate version header");
        size_t value = text.find_first_not_of(" \t\r", colon + 1);
        Version required;
        try {
            required = Version::parse(value == std::string::npos ? "" : text.substr(value, text.find_last_not_of(" \t\r") - value + 1));
        } catch (const std::invalid_argument& error) {
            fail(line, column, error.what());
        }
        if (!runtime_version.supports(required)) {
            std::string message = "script requires DGL " + required.str() + "; runtime supports DGL " + runtime_version.str() +
                                  " (the major version must match and the runtime must meet the script's minimum version)";
            if (required.major == 0 && required.minor == 0 && required.patch == 0)
                message += "; 0.0.0 is a development script: verify it before updating its version declaration";
            fail(line, column, message);
        }
        seen_version_ = true;
    }

    void Lexer::advance_char() {
        if (source_[offset_++] == '\n') {
            ++line_;
            column_ = 1;
        } else
            ++column_;
    }

    Token Lexer::next() {
        while (offset_ < source_.size()) {
            if (std::isspace(static_cast<unsigned char>(source_[offset_])))
                advance_char();
            else if (source_[offset_] == '#' || (source_[offset_] == '/' && offset_ + 1 < source_.size() && source_[offset_ + 1] == '/')) {
                size_t begin = offset_;
                int line = line_, column = column_;
                while (offset_ < source_.size() && source_[offset_] != '\n')
                    advance_char();
                comment(begin, line, column);
            } else
                break;
        }
        Token token{"", line_, column_};
        if (offset_ == source_.size())
            return token;
        seen_token_  = true;
        size_t begin = offset_;
        char c       = source_[offset_];
        advance_char();
        if (c == '"') {
            bool closed = false;
            while (offset_ < source_.size()) {
                char current = source_[offset_];
                if (current == '\n' || current == '\r')
                    fail(token.line, token.column, "unescaped newline in string");
                advance_char();
                if (current == '"') {
                    closed = true;
                    break;
                }
                if (current == '\\' && offset_ < source_.size())
                    advance_char();
            }
            if (!closed)
                fail(token.line, token.column, "unterminated string");
        } else if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
            while (offset_ < source_.size() && (std::isalnum(static_cast<unsigned char>(source_[offset_])) || source_[offset_] == '_'))
                advance_char();
        } else if (c == ':' && offset_ < source_.size() && source_[offset_] == ':') {
            advance_char();
        } else if (std::isdigit(static_cast<unsigned char>(c))) {
            while (offset_ < source_.size() && std::isdigit(static_cast<unsigned char>(source_[offset_])))
                advance_char();
            if (offset_ < source_.size() && source_[offset_] == '.') {
                advance_char();
                while (offset_ < source_.size() && std::isdigit(static_cast<unsigned char>(source_[offset_])))
                    advance_char();
            }
            if (offset_ < source_.size() && (source_[offset_] == 'e' || source_[offset_] == 'E')) {
                advance_char();
                if (offset_ < source_.size() && (source_[offset_] == '+' || source_[offset_] == '-'))
                    advance_char();
                while (offset_ < source_.size() && std::isdigit(static_cast<unsigned char>(source_[offset_])))
                    advance_char();
            }
        }
        token.text = source_.substr(begin, offset_ - begin);
        return token;
    }

}  // namespace sd::dgl
