#pragma once
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "frontend/Token.hpp"
#include "ir/Module.hpp"

namespace ptxforge::frontend {

// Recursive-descent parser for the PTX ISA subset emitted by nvcc / Triton / LLVM NVPTX.
// Throws ParseError (with line/column) on malformed input.
class Parser {
public:
    explicit Parser(std::string_view source);
    ir::Module parse();

private:
    const Token& cur() const { return tokens_[pos_]; }
    const Token& peek(size_t n = 1) const;
    const Token& advance();
    bool accept(TokenKind k);
    const Token& expect(TokenKind k, const char* what);
    [[noreturn]] void error(const std::string& msg, const Token& at) const;
    std::string slice(size_t first_tok, size_t last_tok) const;

    void parse_function(const std::string& linkage);
    void parse_param_list(std::vector<ir::Param>& out);
    ir::VarDecl parse_var_decl(size_t first_tok);
    void skip_directive();
    void skip_balanced_braces();

    void parse_body(ir::Function& fn);
    void parse_reg_decl(ir::Function& fn);
    void parse_instruction(ir::Function& fn);
    void parse_operand(ir::Function& fn, std::vector<ir::Operand>& out);
    ir::MemoryRef parse_memref(ir::Function& fn);
    ir::Operand resolve_register(ir::Function& fn, const Token& t);

    std::string src_;
    std::vector<Token> tokens_;
    size_t pos_ = 0;
    ir::Module module_;
    std::unordered_map<std::string, uint32_t> reg_map_;
};

ir::Module parse_ptx(std::string_view source);

}  // namespace ptxforge::frontend
