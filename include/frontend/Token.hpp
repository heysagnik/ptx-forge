#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ptxforge::frontend {

enum class TokenKind {
    Eof,
    Identifier,  // mnemonics (may contain dots), labels, symbols: mul.wide.s32, $L__BB0_1
    Directive,   // .reg  .entry  .b32
    Register,    // %r1  %tid.x
    Integer,
    Float,       // 1.5, 0f3F800000, 0d3FF0000000000000
    String,
    LBrace, RBrace, LBracket, RBracket, LParen, RParen,
    Comma, Semi, Colon, At, Bang, Plus, Minus, Less, Greater, Pipe,
    Unknown
};

struct Token {
    TokenKind kind = TokenKind::Eof;
    std::string text;
    uint32_t line = 1;
    uint32_t col = 1;
    size_t offset = 0;
    size_t length = 0;
    int64_t int_value = 0;
    double float_value = 0.0;
};

struct ParseError : std::runtime_error {
    uint32_t line;
    uint32_t col;
    ParseError(const std::string& msg, uint32_t l, uint32_t c)
        : std::runtime_error(msg), line(l), col(c) {}
};

}  // namespace ptxforge::frontend
