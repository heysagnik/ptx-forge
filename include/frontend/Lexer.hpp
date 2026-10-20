#pragma once
#include <string_view>
#include <vector>

#include "frontend/Token.hpp"

namespace ptxforge::frontend {

class Lexer {
public:
    explicit Lexer(std::string_view src) : src_(src) {}
    Token next();
    std::vector<Token> tokenize();

private:
    char peek(size_t o = 0) const { return pos_ + o < src_.size() ? src_[pos_ + o] : '\0'; }
    char advance();
    void skip_trivia();
    Token make(TokenKind k, size_t start, uint32_t line, uint32_t col) const;
    Token lex_number(size_t start, uint32_t line, uint32_t col);
    Token lex_string(size_t start, uint32_t line, uint32_t col);

    std::string_view src_;
    size_t pos_ = 0;
    uint32_t line_ = 1;
    uint32_t col_ = 1;
};

}  // namespace ptxforge::frontend
