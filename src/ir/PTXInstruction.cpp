#include "ir/PTXInstruction.hpp"

#include <algorithm>
#include <unordered_map>

namespace ptxforge::ir {

uint32_t type_bits(DataType t) {
    switch (t) {
        case DataType::B8: case DataType::U8: case DataType::S8: return 8;
        case DataType::B16: case DataType::U16: case DataType::S16:
        case DataType::F16: case DataType::BF16: return 16;
        case DataType::B64: case DataType::U64: case DataType::S64: case DataType::F64: return 64;
        case DataType::B128: return 128;
        case DataType::Pred: return 1;
        default: return 32;
    }
}

uint32_t reg_slots(DataType t) {
    if (t == DataType::Pred) return 0;
    uint32_t bits = type_bits(t);
    return bits <= 32 ? 1 : bits / 32;
}

namespace {
struct TypeName { const char* name; DataType type; };
constexpr TypeName kTypes[] = {
    {"b8", DataType::B8},   {"b16", DataType::B16}, {"b32", DataType::B32},   {"b64", DataType::B64},
    {"b128", DataType::B128},
    {"u8", DataType::U8},   {"u16", DataType::U16}, {"u32", DataType::U32},   {"u64", DataType::U64},
    {"s8", DataType::S8},   {"s16", DataType::S16}, {"s32", DataType::S32},   {"s64", DataType::S64},
    {"f16", DataType::F16}, {"f16x2", DataType::F16X2}, {"bf16", DataType::BF16},
    {"bf16x2", DataType::BF16X2}, {"tf32", DataType::TF32},
    {"f32", DataType::F32}, {"f64", DataType::F64}, {"pred", DataType::Pred},
};
struct SpaceName { const char* name; StateSpace space; };
constexpr SpaceName kSpaces[] = {
    {"reg", StateSpace::Reg},       {"sreg", StateSpace::SReg},   {"param", StateSpace::Param},
    {"global", StateSpace::Global}, {"shared", StateSpace::Shared}, {"local", StateSpace::Local},
    {"const", StateSpace::Const},
};
}  // namespace

std::optional<DataType> parse_type(const std::string& s) {
    for (auto& t : kTypes) if (s == t.name) return t.type;
    return std::nullopt;
}

std::optional<StateSpace> parse_space(const std::string& s) {
    for (auto& t : kSpaces) if (s == t.name) return t.space;
    return std::nullopt;
}

const char* to_string(DataType t) {
    for (auto& e : kTypes) if (e.type == t) return e.name;
    return "b32";
}

const char* to_string(StateSpace s) {
    for (auto& e : kSpaces) if (e.space == s) return e.name;
    return "global";
}

const char* to_string(Opcode o) {
    switch (o) {
        case Opcode::LD: return "ld";       case Opcode::ST: return "st";
        case Opcode::MOV: return "mov";     case Opcode::CVT: return "cvt";
        case Opcode::CVTA: return "cvta";   case Opcode::ADD: return "add";
        case Opcode::SUB: return "sub";     case Opcode::MUL: return "mul";
        case Opcode::MAD: return "mad";     case Opcode::FMA: return "fma";
        case Opcode::DIV: return "div";     case Opcode::REM: return "rem";
        case Opcode::NEG: return "neg";     case Opcode::ABS: return "abs";
        case Opcode::MIN: return "min";     case Opcode::MAX: return "max";
        case Opcode::NOT: return "not";     case Opcode::SHL: return "shl";
        case Opcode::SHR: return "shr";     case Opcode::AND: return "and";
        case Opcode::OR: return "or";       case Opcode::XOR: return "xor";
        case Opcode::SETP: return "setp";   case Opcode::SEL: return "selp";
        case Opcode::BRA: return "bra";     case Opcode::RET: return "ret";
        case Opcode::EXIT: return "exit";   case Opcode::BAR_SYNC: return "bar.sync";
        case Opcode::CP_ASYNC: return "cp.async"; case Opcode::MMA_SYNC: return "mma.sync";
        case Opcode::ATOM: return "atom";   case Opcode::RED: return "red";
        case Opcode::CALL: return "call";   case Opcode::OTHER: return "other";
    }
    return "other";
}

namespace {
const std::unordered_map<std::string, Opcode>& root_table() {
    static const std::unordered_map<std::string, Opcode> t = {
        {"ld", Opcode::LD},     {"st", Opcode::ST},     {"mov", Opcode::MOV},   {"cvt", Opcode::CVT},
        {"cvta", Opcode::CVTA}, {"add", Opcode::ADD},   {"sub", Opcode::SUB},   {"mul", Opcode::MUL},
        {"mad", Opcode::MAD},   {"fma", Opcode::FMA},   {"div", Opcode::DIV},   {"rem", Opcode::REM},
        {"neg", Opcode::NEG},   {"abs", Opcode::ABS},   {"min", Opcode::MIN},   {"max", Opcode::MAX},
        {"not", Opcode::NOT},   {"shl", Opcode::SHL},   {"shr", Opcode::SHR},   {"and", Opcode::AND},
        {"or", Opcode::OR},     {"xor", Opcode::XOR},   {"setp", Opcode::SETP}, {"selp", Opcode::SEL},
        {"bra", Opcode::BRA},   {"ret", Opcode::RET},   {"exit", Opcode::EXIT}, {"atom", Opcode::ATOM},
        {"red", Opcode::RED},   {"call", Opcode::CALL},
    };
    return t;
}
}  // namespace

void classify_mnemonic(PTXInstruction& ins, const std::string& mnemonic) {
    ins.mnemonic = mnemonic;
    ins.modifiers.clear();
    std::vector<std::string> parts;
    size_t start = 0;
    for (;;) {
        size_t dot = mnemonic.find('.', start);
        parts.push_back(mnemonic.substr(start, dot == std::string::npos ? dot : dot - start));
        if (dot == std::string::npos) break;
        start = dot + 1;
    }
    ins.modifiers.assign(parts.begin() + 1, parts.end());
    auto has_mod = [&](const char* m) {
        return std::find(ins.modifiers.begin(), ins.modifiers.end(), m) != ins.modifiers.end();
    };

    const std::string& root = parts[0];
    auto it = root_table().find(root);
    if (it != root_table().end()) {
        ins.opcode = it->second;
    } else if ((root == "bar" || root == "barrier") && has_mod("sync")) {
        ins.opcode = Opcode::BAR_SYNC;
    } else if (root == "cp" && !parts.empty() && parts.size() > 1 && parts[1] == "async") {
        ins.opcode = Opcode::CP_ASYNC;
    } else if (root == "mma" && parts.size() > 1 && parts[1] == "sync") {
        ins.opcode = Opcode::MMA_SYNC;
    } else {
        ins.opcode = Opcode::OTHER;
    }

    ins.space.reset();
    ins.src_type.reset();
    bool have_type = false;
    ins.type = DataType::B32;
    for (auto& m : ins.modifiers) {
        if (!ins.space) {
            if (auto s = parse_space(m)) { ins.space = s; continue; }
        }
        if (auto t = parse_type(m)) {
            if (!have_type) { ins.type = *t; have_type = true; }
            else if (!ins.src_type) { ins.src_type = *t; }
        }
    }
}

namespace {
template <class F>
void for_each_reg(Operand& op, F&& f) {
    if (auto* r = std::get_if<VirtualReg>(&op)) f(*r);
    else if (auto* v = std::get_if<RegVec>(&op)) for (auto& r2 : v->regs) f(r2);
}
}  // namespace

std::vector<VirtualReg> PTXInstruction::defs() const {
    std::vector<VirtualReg> out;
    if (!has_dest || operands.empty()) return out;
    if (auto* r = std::get_if<VirtualReg>(&operands[0])) out.push_back(*r);
    else if (auto* v = std::get_if<RegVec>(&operands[0])) out = v->regs;
    return out;
}

std::vector<VirtualReg> PTXInstruction::get_used_registers() const {
    std::vector<VirtualReg> out;
    if (predicate) out.push_back(predicate->pred_reg);
    for (size_t i = 0; i < operands.size(); ++i) {
        const bool dest = has_dest && i == 0;
        if (auto* r = std::get_if<VirtualReg>(&operands[i])) {
            if (!dest) out.push_back(*r);
        } else if (auto* m = std::get_if<MemoryRef>(&operands[i])) {
            if (m->base_reg) out.push_back(*m->base_reg);
        } else if (auto* v = std::get_if<RegVec>(&operands[i])) {
            if (!dest) out.insert(out.end(), v->regs.begin(), v->regs.end());
        }
    }
    return out;
}

bool PTXInstruction::has_side_effects() const {
    switch (opcode) {
        case Opcode::ST: case Opcode::BRA: case Opcode::RET: case Opcode::EXIT:
        case Opcode::BAR_SYNC: case Opcode::CP_ASYNC: case Opcode::ATOM:
        case Opcode::RED: case Opcode::CALL: case Opcode::OTHER:
            return true;
        default:
            break;
    }
    return std::find(modifiers.begin(), modifiers.end(), "volatile") != modifiers.end();
}

std::string PTXInstruction::branch_target() const {
    if (opcode != Opcode::BRA || operands.empty()) return {};
    if (auto* s = std::get_if<std::string>(&operands[0])) return *s;
    return {};
}

void PTXInstruction::replace_use(uint32_t old_id, const VirtualReg& nw) {
    if (predicate && predicate->pred_reg.id == old_id) predicate->pred_reg = nw;
    for (size_t i = 0; i < operands.size(); ++i) {
        if (has_dest && i == 0) continue;
        if (auto* m = std::get_if<MemoryRef>(&operands[i])) {
            if (m->base_reg && m->base_reg->id == old_id) m->base_reg = nw;
        } else {
            for_each_reg(operands[i], [&](VirtualReg& r) { if (r.id == old_id) r = nw; });
        }
    }
}

void PTXInstruction::replace_def(uint32_t old_id, const VirtualReg& nw) {
    if (!has_dest || operands.empty()) return;
    for_each_reg(operands[0], [&](VirtualReg& r) { if (r.id == old_id) r = nw; });
}

}  // namespace ptxforge::ir
