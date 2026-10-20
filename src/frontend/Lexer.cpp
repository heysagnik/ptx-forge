#include "frontend/Lexer.hpp"

#include <cctype>
#include <cstring>
#include <string>

namespace ptxforge::frontend {

namespace {
bool is_ident_start(char c) { return std::isalpha(static_cast<unsigned char>(c)) || c == '_' || c == '$'; }
bool is_ident_char(char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '$'; }
bool is_hex(char c) { return std::isxdigit(static_cast<unsigned char>(c)) != 0; }
bool is_dec(char c) { return std::isdigit(static_cast<unsigned char>(c)) != 0; }
}  // namespace

char Lexer::advance() {
    char c = src_[pos_++];
    if (c == '\n') { ++line_; col_ = 1; } else { ++col_; }
    return c;
}

void Lexer::skip_trivia() {
    for (;;) {
        char c = peek();
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') { advance(); continue; }
        if (c == '/' && peek(1) == '/') { while (pos_ < src_.size() && peek() != '\n') advance(); continue; }
        if (c == '/' && peek(1) == '*') {
            uint32_t l = line_, cl = col_;
            advance(); advance();
            while (pos_ < src_.size() && !(peek() == '*' && peek(1) == '/')) advance();
            if (pos_ >= src_.size()) throw ParseError("unterminated block comment", l, cl);
            advance(); advance();
            continue;
        }
        break;
    }
}

Token Lexer::make(TokenKind k, size_t start, uint32_t line, uint32_t col) const {
    Token t;
    t.kind = k;
    t.offset = start;
    t.length = pos_ - start;
    t.text = std::string(src_.substr(start, pos_ - start));
    t.line = line;
    t.col = col;
    return t;
}

Token Lexer::lex_string(size_t start, uint32_t line, uint32_t col) {
    advance();  // opening quote
    while (pos_ < src_.size() && peek() != '"') {
        if (peek() == '\\' && pos_ + 1 < src_.size()) advance();
        if (peek() == '\n') throw ParseError("unterminated string literal", line, col);
        advance();
    }
    if (pos_ >= src_.size()) throw ParseError("unterminated string literal", line, col);
    advance();  // closing quote
    return make(TokenKind::String, start, line, col);
}

Token Lexer::lex_number(size_t start, uint32_t line, uint32_t col) {
    // Hex-encoded float literals: 0fXXXXXXXX (binary32) / 0dXXXXXXXXXXXXXXXX (binary64).
    if (peek() == '0' && (peek(1) == 'f' || peek(1) == 'F' || peek(1) == 'd' || peek(1) == 'D')) {
        bool dbl = (peek(1) == 'd' || peek(1) == 'D');
        size_t digits = dbl ? 16 : 8;
        bool ok = true;
        for (size_t i = 0; i < digits; ++i) ok = ok && is_hex(peek(2 + i));
        if (ok && !is_ident_char(peek(2 + digits))) {
            std::string hex(src_.substr(pos_ + 2, digits));
            for (size_t i = 0; i < digits + 2; ++i) advance();
            Token t = make(TokenKind::Float, start, line, col);
            uint64_t bits = std::stoull(hex, nullptr, 16);
            if (dbl) {
                std::memcpy(&t.float_value, &bits, sizeof(double));
            } else {
                uint32_t b32 = static_cast<uint32_t>(bits);
                float f;
                std::memcpy(&f, &b32, sizeof(float));
                t.float_value = f;
            }
            return t;
        }
    }
    if (peek() == '0' && (peek(1) == 'x' || peek(1) == 'X') && is_hex(peek(2))) {
        advance(); advance();
        while (is_hex(peek())) advance();
        Token t = make(TokenKind::Integer, start, line, col);
        t.int_value = static_cast<int64_t>(std::stoull(t.text.substr(2), nullptr, 16));
        if (peek() == 'U' || peek() == 'u') advance();
        return t;
    }
    while (is_dec(peek())) advance();
    bool is_float = false;
    if (peek() == '.' && is_dec(peek(1))) {
        is_float = true;
        advance();
        while (is_dec(peek())) advance();
    }
    if ((peek() == 'e' || peek() == 'E') &&
        (is_dec(peek(1)) || ((peek(1) == '+' || peek(1) == '-') && is_dec(peek(2))))) {
        is_float = true;
        advance();
        if (peek() == '+' || peek() == '-') advance();
        while (is_dec(peek())) advance();
    }
    Token t = make(is_float ? TokenKind::Float : TokenKind::Integer, start, line, col);
    if (is_float) {
        t.float_value = std::stod(t.text);
    } else {
        t.int_value = static_cast<int64_t>(std::stoull(t.text));
        if (peek() == 'U' || peek() == 'u') advance();
    }
    return t;
}

Token Lexer::next() {
    skip_trivia();
    size_t start = pos_;
    uint32_t line = line_, col = col_;
    if (pos_ >= src_.size()) return make(TokenKind::Eof, start, line, col);

    char c = peek();
    if (is_dec(c)) return lex_number(start, line, col);
    if (c == '"') return lex_string(start, line, col);

    if (c == '.' && (std::isalpha(static_cast<unsigned char>(peek(1))) || peek(1) == '_')) {
        advance();
        while (is_ident_char(peek())) advance();
        return make(TokenKind::Directive, start, line, col);
    }
    if (c == '%') {
        advance();
        while (is_ident_char(peek()) || peek() == '.') advance();
        return make(TokenKind::Register, start, line, col);
    }
    if (is_ident_start(c)) {
        advance();
        while (is_ident_char(peek()) || (peek() == '.' && is_ident_char(peek(1)))) advance();
        return make(TokenKind::Identifier, start, line, col);
    }

    advance();
    TokenKind k;
    switch (c) {
        case '{': k = TokenKind::LBrace; break;
        case '}': k = TokenKind::RBrace; break;
        case '[': k = TokenKind::LBracket; break;
        case ']': k = TokenKind::RBracket; break;
        case '(': k = TokenKind::LParen; break;
        case ')': k = TokenKind::RParen; break;
        case ',': k = TokenKind::Comma; break;
        case ';': k = TokenKind::Semi; break;
        case ':': k = TokenKind::Colon; break;
        case '@': k = TokenKind::At; break;
        case '!': k = TokenKind::Bang; break;
        case '+': k = TokenKind::Plus; break;
        case '-': k = TokenKind::Minus; break;
        case '<': k = TokenKind::Less; break;
        case '>': k = TokenKind::Greater; break;
        case '|': k = TokenKind::Pipe; break;
        default: k = TokenKind::Unknown; break;
    }
    return make(k, start, line, col);
}

std::vector<Token> Lexer::tokenize() {
    std::vector<Token> out;
    for (;;) {
        out.push_back(next());
        if (out.back().kind == TokenKind::Eof) break;
    }
    return out;
}

}  // namespace ptxforge::frontend
