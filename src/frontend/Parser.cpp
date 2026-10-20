#include "frontend/Parser.hpp"

#include <algorithm>

#include "frontend/Lexer.hpp"

namespace ptxforge::frontend {

using namespace ir;

namespace {
bool is_special_register(const std::string& name) {
    static const char* kPrefixes[] = {
        "%tid", "%ntid", "%ctaid", "%nctaid", "%laneid", "%warpid", "%nwarpid", "%smid",
        "%gridid", "%lanemask_eq", "%lanemask_le", "%lanemask_lt", "%lanemask_ge",
        "%lanemask_gt", "%clock", "%clock64", "%globaltimer", "%pm", "%envreg",
        "%total_smem_size", "%dynamic_smem_size", "%aggr_smem_size", "%clusterid",
        "%nclusterid", "%cluster_ctaid", "%cluster_nctaid", "%cluster_ctarank",
        "%cluster_nctarank", "%is_explicit_cluster", "%reserved_smem_offset"};
    std::string root = name.substr(0, name.find('.'));
    // %pm0..%pm7, %envreg0..31 carry a numeric suffix
    for (auto* p : kPrefixes) {
        std::string pre = p;
        if (root.compare(0, pre.size(), pre) == 0 &&
            std::all_of(root.begin() + static_cast<long>(pre.size()), root.end(),
                        [](char c) { return std::isdigit(static_cast<unsigned char>(c)) != 0; }))
            return true;
    }
    return false;
}
}  // namespace

Parser::Parser(std::string_view source) : src_(source) {
    Lexer lx(src_);
    tokens_ = lx.tokenize();
}

const Token& Parser::peek(size_t n) const {
    size_t i = std::min(pos_ + n, tokens_.size() - 1);
    return tokens_[i];
}

const Token& Parser::advance() {
    const Token& t = tokens_[pos_];
    if (pos_ + 1 < tokens_.size()) ++pos_;
    return t;
}

bool Parser::accept(TokenKind k) {
    if (cur().kind != k) return false;
    advance();
    return true;
}

void Parser::error(const std::string& msg, const Token& at) const {
    throw ParseError(msg, at.line, at.col);
}

const Token& Parser::expect(TokenKind k, const char* what) {
    if (cur().kind != k) {
        error(std::string("expected ") + what + ", found '" + cur().text + "'", cur());
    }
    return advance();
}

std::string Parser::slice(size_t first_tok, size_t last_tok) const {
    const Token& a = tokens_[first_tok];
    const Token& b = tokens_[last_tok];
    return src_.substr(a.offset, b.offset + b.length - a.offset);
}

ir::Module parse_ptx(std::string_view source) { return Parser(source).parse(); }

void Parser::skip_balanced_braces() {
    int depth = 0;
    do {
        if (cur().kind == TokenKind::Eof) error("unterminated '{'", cur());
        if (cur().kind == TokenKind::LBrace) ++depth;
        if (cur().kind == TokenKind::RBrace) --depth;
        advance();
    } while (depth > 0);
}

// Directives like .file / .loc are newline-terminated (no ';'); .section carries a brace body.
void Parser::skip_directive() {
    const Token start = advance();
    while (cur().kind != TokenKind::Eof) {
        if (cur().kind == TokenKind::LBrace) { skip_balanced_braces(); return; }
        if (cur().line != start.line) break;
        if (cur().kind == TokenKind::Semi) { advance(); return; }
        advance();
    }
    if (start.text == ".section" && cur().kind == TokenKind::LBrace) skip_balanced_braces();
}

Module Parser::parse() {
    std::string linkage;
    size_t decl_start = pos_;
    bool have_decl_start = false;
    while (cur().kind != TokenKind::Eof) {
        const Token& t = cur();
        if (t.kind != TokenKind::Directive) error("unexpected token '" + t.text + "' at module scope", t);
        const std::string& d = t.text;
        if (d == ".version") {
            advance();
            if (cur().kind != TokenKind::Float && cur().kind != TokenKind::Integer)
                error("expected version number", cur());
            module_.version = advance().text;
        } else if (d == ".target") {
            advance();
            bool first = true;
            do {
                const Token& id = expect(TokenKind::Identifier, "target name");
                if (first) module_.target = id.text; else module_.target_options.push_back(id.text);
                first = false;
            } while (accept(TokenKind::Comma));
        } else if (d == ".address_size") {
            advance();
            module_.address_size = static_cast<uint32_t>(expect(TokenKind::Integer, "address size").int_value);
        } else if (d == ".visible" || d == ".extern" || d == ".weak" || d == ".common" || d == ".no_return") {
            if (!have_decl_start) { decl_start = pos_; have_decl_start = true; }
            linkage = d;
            advance();
            continue;
        } else if (d == ".entry" || d == ".func") {
            parse_function(linkage);
        } else if (d == ".global" || d == ".const" || d == ".shared" || d == ".local" ||
                   d == ".tex" || d == ".surf" || d == ".sampler") {
            module_.globals.push_back(parse_var_decl(have_decl_start ? decl_start : pos_));
        } else {
            skip_directive();
        }
        linkage.clear();
        have_decl_start = false;
    }
    if (module_.version.empty()) throw ParseError("missing .version directive", 1, 1);
    if (module_.target.empty()) throw ParseError("missing .target directive", 1, 1);
    return std::move(module_);
}

VarDecl Parser::parse_var_decl(size_t first_tok) {
    VarDecl v;
    DataType elem = DataType::B8;
    uint64_t count = 1;
    bool have_space = false;
    while (cur().kind == TokenKind::Directive) {
        const std::string name = cur().text.substr(1);
        if (name == "align") {
            advance();
            v.align = static_cast<uint32_t>(expect(TokenKind::Integer, "alignment").int_value);
            continue;
        }
        if (!have_space) {
            if (auto s = parse_space(name)) { v.space = *s; have_space = true; advance(); continue; }
        }
        if (auto t = parse_type(name)) elem = *t;
        advance();
    }
    if (cur().kind != TokenKind::Identifier) error("expected variable name", cur());
    v.name = advance().text;
    if (accept(TokenKind::LBracket)) {
        count = cur().kind == TokenKind::Integer ? static_cast<uint64_t>(advance().int_value) : 0;
        expect(TokenKind::RBracket, "']'");
    }
    while (cur().kind != TokenKind::Semi && cur().kind != TokenKind::Eof) advance();  // initializer
    size_t last = pos_;
    expect(TokenKind::Semi, "';'");
    v.bytes = (type_bits(elem) / 8) * count;
    v.raw = slice(first_tok, last);
    return v;
}

void Parser::parse_param_list(std::vector<Param>& out) {
    expect(TokenKind::LParen, "'('");
    if (accept(TokenKind::RParen)) return;
    do {
        size_t first = pos_;
        if (cur().kind != TokenKind::Directive || cur().text != ".param")
            error("expected .param in parameter list", cur());
        Param p;
        DataType elem = DataType::B8;
        uint64_t count = 1;
        size_t last = pos_;
        int depth = 0;
        while (cur().kind != TokenKind::Eof) {
            if (depth == 0 && (cur().kind == TokenKind::Comma || cur().kind == TokenKind::RParen)) break;
            if (cur().kind == TokenKind::LBracket) ++depth;
            if (cur().kind == TokenKind::RBracket) --depth;
            if (cur().kind == TokenKind::Directive) {
                if (auto t = parse_type(cur().text.substr(1))) elem = *t;
            } else if (cur().kind == TokenKind::Identifier) {
                p.name = cur().text;
            } else if (cur().kind == TokenKind::Integer && depth > 0) {
                count = static_cast<uint64_t>(cur().int_value);
            }
            last = pos_;
            advance();
        }
        p.bytes = (type_bits(elem) / 8) * count;
        p.raw = slice(first, last);
        out.push_back(std::move(p));
    } while (accept(TokenKind::Comma));
    expect(TokenKind::RParen, "')'");
}

void Parser::parse_function(const std::string& linkage) {
    Function fn;
    fn.is_entry = advance().text == ".entry";
    fn.linkage = linkage;
    reg_map_.clear();
    if (!fn.is_entry && cur().kind == TokenKind::LParen) parse_param_list(fn.ret_params);
    fn.name = expect(TokenKind::Identifier, "function name").text;
    if (cur().kind == TokenKind::LParen) parse_param_list(fn.params);

    size_t first = pos_;
    while (cur().kind != TokenKind::LBrace && cur().kind != TokenKind::Semi && cur().kind != TokenKind::Eof)
        advance();
    if (pos_ > first) fn.header_directives = slice(first, pos_ - 1);
    if (accept(TokenKind::Semi)) return;  // forward declaration: nothing to analyze
    expect(TokenKind::LBrace, "'{'");
    parse_body(fn);
    module_.functions.push_back(std::move(fn));
}

void Parser::parse_body(Function& fn) {
    for (;;) {
        const Token& t = cur();
        switch (t.kind) {
            case TokenKind::Eof: error("unterminated function body", t);
            case TokenKind::RBrace: advance(); return;
            case TokenKind::LBrace: advance(); parse_body(fn); continue;
            case TokenKind::Directive: {
                const std::string& d = t.text;
                if (d == ".reg") {
                    parse_reg_decl(fn);
                } else if (d == ".shared" || d == ".local" || d == ".const" || d == ".global" || d == ".param") {
                    fn.vars.push_back(parse_var_decl(pos_));
                } else if (d == ".pragma") {
                    size_t first = pos_;
                    while (cur().kind != TokenKind::Semi && cur().kind != TokenKind::Eof) advance();
                    size_t last = pos_;
                    expect(TokenKind::Semi, "';'");
                    VarDecl v;
                    v.raw = slice(first, last);
                    v.space = StateSpace::Reg;
                    fn.vars.push_back(std::move(v));
                } else {
                    skip_directive();  // .loc, .file, ...
                }
                continue;
            }
            case TokenKind::Identifier:
                if (peek().kind == TokenKind::Colon) {
                    fn.labels.emplace_back(t.text, fn.instructions.size());
                    advance();
                    advance();
                    continue;
                }
                [[fallthrough]];
            case TokenKind::At:
                parse_instruction(fn);
                continue;
            default:
                error("unexpected token '" + t.text + "' in function body", t);
        }
    }
}

void Parser::parse_reg_decl(Function& fn) {
    advance();  // .reg
    DataType type = DataType::B32;
    bool have_type = false;
    while (cur().kind == TokenKind::Directive) {
        if (auto t = parse_type(cur().text.substr(1))) { type = *t; have_type = true; }
        advance();
    }
    if (!have_type) error("missing register type in .reg declaration", cur());

    auto declare = [&](const std::string& name) {
        auto it = reg_map_.find(name);
        if (it != reg_map_.end() && fn.regs[it->second].type == type) return;  // re-declared in nested scope
        VirtualReg r = fn.add_reg(name, type);
        reg_map_[name] = r.id;
    };
    do {
        if (cur().kind != TokenKind::Register && cur().kind != TokenKind::Identifier)
            error("expected register name", cur());
        std::string base = advance().text;
        if (accept(TokenKind::Less)) {
            int64_t n = expect(TokenKind::Integer, "register count").int_value;
            expect(TokenKind::Greater, "'>'");
            for (int64_t i = 0; i < n; ++i) declare(base + std::to_string(i));
        } else {
            declare(base);
        }
    } while (accept(TokenKind::Comma));
    expect(TokenKind::Semi, "';'");
}

Operand Parser::resolve_register(Function& fn, const Token& t) {
    auto it = reg_map_.find(t.text);
    if (it != reg_map_.end()) return fn.regs[it->second];
    if (is_special_register(t.text)) return t.text;
    error("use of undeclared register '" + t.text + "'", t);
}

MemoryRef Parser::parse_memref(Function& fn) {
    expect(TokenKind::LBracket, "'['");
    MemoryRef m;
    const Token& b = cur();
    if (b.kind == TokenKind::Register) {
        Operand op = resolve_register(fn, b);
        if (auto* r = std::get_if<VirtualReg>(&op)) m.base_reg = *r; else m.symbol = b.text;
        advance();
    } else if (b.kind == TokenKind::Identifier) {
        auto it = reg_map_.find(b.text);
        if (it != reg_map_.end()) m.base_reg = fn.regs[it->second]; else m.symbol = b.text;
        advance();
    } else if (b.kind == TokenKind::Integer) {
        m.offset = advance().int_value;
    } else {
        error("expected address expression", b);
    }
    while (cur().kind == TokenKind::Plus || cur().kind == TokenKind::Minus) {
        bool neg = advance().kind == TokenKind::Minus;
        int64_t v = expect(TokenKind::Integer, "offset").int_value;
        m.offset += neg ? -v : v;
    }
    expect(TokenKind::RBracket, "']'");
    return m;
}

void Parser::parse_operand(Function& fn, std::vector<Operand>& out) {
    const Token& t = cur();
    switch (t.kind) {
        case TokenKind::Register: {
            Operand first = resolve_register(fn, t);
            advance();
            if (cur().kind == TokenKind::Pipe) {  // %p|%q dual destination
                advance();
                const Token& t2 = expect(TokenKind::Register, "register after '|'");
                Operand second = resolve_register(fn, t2);
                auto* a = std::get_if<VirtualReg>(&first);
                auto* b = std::get_if<VirtualReg>(&second);
                if (!a || !b) error("'|' operands must be virtual registers", t2);
                out.push_back(RegVec{{*a, *b}, '|'});
            } else {
                out.push_back(std::move(first));
            }
            return;
        }
        case TokenKind::Integer: out.emplace_back(advance().int_value); return;
        case TokenKind::Float: {
            const Token& f = advance();
            out.emplace_back(ImmFloat{f.float_value, f.text});
            return;
        }
        case TokenKind::Minus: {
            advance();
            if (cur().kind == TokenKind::Integer) { out.emplace_back(-advance().int_value); return; }
            if (cur().kind == TokenKind::Float) {
                const Token& f = advance();
                out.emplace_back(ImmFloat{-f.float_value, "-" + f.text});
                return;
            }
            error("expected number after '-'", cur());
        }
        case TokenKind::Plus: advance(); parse_operand(fn, out); return;
        case TokenKind::LBracket: out.emplace_back(parse_memref(fn)); return;
        case TokenKind::LBrace: {
            advance();
            RegVec v;
            do {
                const Token& e = cur();
                if (e.kind == TokenKind::Register) {
                    Operand op = resolve_register(fn, e);
                    auto* r = std::get_if<VirtualReg>(&op);
                    if (!r) error("special register not allowed inside vector operand", e);
                    v.regs.push_back(*r);
                    advance();
                } else if (e.kind == TokenKind::Identifier && reg_map_.count(e.text)) {
                    v.regs.push_back(fn.regs[reg_map_[e.text]]);
                    advance();
                } else {
                    error("only registers are supported inside vector operands", e);
                }
            } while (accept(TokenKind::Comma));
            expect(TokenKind::RBrace, "'}'");
            out.emplace_back(std::move(v));
            return;
        }
        case TokenKind::LParen: {  // call argument lists: flattened
            advance();
            if (!accept(TokenKind::RParen)) {
                do { parse_operand(fn, out); } while (accept(TokenKind::Comma));
                expect(TokenKind::RParen, "')'");
            }
            return;
        }
        case TokenKind::Identifier: {
            auto it = reg_map_.find(t.text);
            if (it != reg_map_.end()) out.emplace_back(fn.regs[it->second]);
            else out.emplace_back(t.text);
            advance();
            return;
        }
        default:
            error("unexpected token '" + t.text + "' in operand list", t);
    }
}

void Parser::parse_instruction(Function& fn) {
    PTXInstruction ins;
    ins.line_number = cur().line;
    if (accept(TokenKind::At)) {
        bool neg = accept(TokenKind::Bang);
        const Token& pt = expect(TokenKind::Register, "predicate register");
        Operand p = resolve_register(fn, pt);
        auto* r = std::get_if<VirtualReg>(&p);
        if (!r) error("predicate must be a virtual register", pt);
        ins.predicate = PredicateGuard{*r, neg};
    }
    const Token& mn = expect(TokenKind::Identifier, "instruction mnemonic");
    classify_mnemonic(ins, mn.text);

    if (cur().kind != TokenKind::Semi) {
        do { parse_operand(fn, ins.operands); } while (accept(TokenKind::Comma));
    }
    expect(TokenKind::Semi, "';'");

    // Destination detection: first operand is a register (or register vector) and the opcode writes one.
    switch (ins.opcode) {
        case Opcode::ST: case Opcode::BRA: case Opcode::RET: case Opcode::EXIT:
        case Opcode::BAR_SYNC: case Opcode::CP_ASYNC: case Opcode::RED: case Opcode::CALL:
            ins.has_dest = false;
            break;
        default:
            ins.has_dest = !ins.operands.empty() &&
                           (std::holds_alternative<VirtualReg>(ins.operands[0]) ||
                            std::holds_alternative<RegVec>(ins.operands[0]));
    }
    // Memory operands inherit the instruction's state space.
    size_t mem_idx = 0;
    for (auto& op : ins.operands) {
        if (auto* m = std::get_if<MemoryRef>(&op)) {
            if (ins.opcode == Opcode::CP_ASYNC) m->space = mem_idx == 0 ? StateSpace::Shared : StateSpace::Global;
            else m->space = ins.space.value_or(StateSpace::Global);
            ++mem_idx;
        }
    }
    fn.instructions.push_back(std::move(ins));
}

}  // namespace ptxforge::frontend
