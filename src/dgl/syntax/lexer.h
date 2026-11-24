#ifndef __SD_DGL_LEXER_H__
#define __SD_DGL_LEXER_H__

#include <cstddef>
#include <string>

namespace sd::dgl {

    struct Token {
        std::string text;
        int line   = 1;
        int column = 1;
    };

    class Lexer {
        const std::string& source_;
        size_t offset_     = 0;
        int line_          = 1;
        int column_        = 1;
        bool seen_token_   = false;
        bool seen_version_ = false;

        void comment(size_t begin, int line, int column);
        void advance_char();

    public:
        explicit Lexer(const std::string& source);
        Token next();
    };

}  // namespace sd::dgl

#endif  // __SD_DGL_LEXER_H__
