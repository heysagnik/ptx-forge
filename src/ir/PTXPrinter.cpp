#include "ir/PTXPrinter.hpp"

#include <sstream>

namespace ptxforge::ir {

namespace {
std::string format_memref(const MemoryRef& m) {
    if (!m.base_reg && m.symbol.empty()) return "[" + std::to_string(m.offset) + "]";  // absolute address
    std::string s = "[" + (m.base_reg ? m.base_reg->name : m.symbol);
    if (m.offset > 0) s += "+" + std::to_string(m.offset);
    else if (m.offset < 0) s += std::to_string(m.offset);
    return s + "]";
}
}  // namespace

std::string format_operand(const Operand& op) {
    struct V {
        std::string operator()(const VirtualReg& r) const { return r.name; }
        std::string operator()(const MemoryRef& m) const { return format_memref(m); }
        std::string operator()(int64_t v) const { return std::to_string(v); }
        std::string operator()(const ImmFloat& f) const { return f.text; }
        std::string operator()(const std::string& s) const { return s; }
        std::string operator()(const RegVec& v) const {
            std::string s = v.kind == '{' ? "{" : "";
            for (size_t i = 0; i < v.regs.size(); ++i) {
                if (i) s += v.kind == '{' ? ", " : "|";
                s += v.regs[i].name;
            }
            return s + (v.kind == '{' ? "}" : "");
        }
    };
    return std::visit(V{}, op);
}

std::string format_instruction(const PTXInstruction& ins) {
    std::string s;
    if (ins.predicate) s += std::string("@") + (ins.predicate->is_negated ? "!" : "") + ins.predicate->pred_reg.name + " ";
    s += ins.mnemonic;
    for (size_t i = 0; i < ins.operands.size(); ++i) {
        s += i ? ", " : " ";
        s += format_operand(ins.operands[i]);
    }
    return s + ";";
}

std::string print_function(const Function& fn) {
    std::ostringstream os;
    if (!fn.linkage.empty()) os << fn.linkage << " ";
    os << (fn.is_entry ? ".entry " : ".func ");
    if (!fn.is_entry && !fn.ret_params.empty()) {
        os << "(";
        for (size_t i = 0; i < fn.ret_params.size(); ++i) os << (i ? ", " : "") << fn.ret_params[i].raw;
        os << ") ";
    }
    os << fn.name << "(";
    for (size_t i = 0; i < fn.params.size(); ++i) os << (i ? ",\n\t" : "\n\t") << fn.params[i].raw;
    os << "\n)\n";
    if (!fn.header_directives.empty()) os << fn.header_directives << "\n";
    os << "{\n";

    // Register declarations, grouped by type.
    for (int t = 0; t <= static_cast<int>(DataType::Pred); ++t) {
        auto type = static_cast<DataType>(t);
        size_t n = 0;
        for (auto& r : fn.regs) {
            if (r.type != type) continue;
            if (n % 8 == 0) os << (n ? ";\n" : "") << "\t.reg ." << to_string(type) << " ";
            else os << ", ";
            os << r.name;
            ++n;
        }
        if (n) os << ";\n";
    }
    if (fn.spill_frame_bytes)
        os << "\t.local .align " << fn.spill_frame_align << " .b8 __spill_depot[" << fn.spill_frame_bytes << "];\n";
    for (auto& v : fn.vars) os << "\t" << v.raw << "\n";

    size_t next_label = 0;
    auto emit_labels_at = [&](size_t idx) {
        while (next_label < fn.labels.size() && fn.labels[next_label].second == idx) {
            os << fn.labels[next_label].first << ":\n";
            ++next_label;
        }
    };
    for (size_t i = 0; i < fn.instructions.size(); ++i) {
        emit_labels_at(i);
        os << "\t" << format_instruction(fn.instructions[i]) << "\n";
    }
    emit_labels_at(fn.instructions.size());
    os << "}\n";
    return os.str();
}

std::string print_module(const Module& m) {
    std::ostringstream os;
    os << ".version " << m.version << "\n.target " << m.target;
    for (auto& o : m.target_options) os << ", " << o;
    os << "\n.address_size " << m.address_size << "\n\n";
    for (auto& g : m.globals) os << g.raw << "\n";
    for (auto& f : m.functions) os << "\n" << print_function(f);
    return os.str();
}

}  // namespace ptxforge::ir
