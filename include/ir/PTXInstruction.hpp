#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace ptxforge::ir {

enum class StateSpace { Reg, SReg, Param, Global, Shared, Local, Const };

enum class DataType {
    B8, B16, B32, B64, B128,
    U8, U16, U32, U64,
    S8, S16, S32, S64,
    F16, F16X2, BF16, BF16X2, TF32, F32, F64,
    Pred
};

enum class Opcode {
    LD, ST, MOV, CVT, CVTA,
    ADD, SUB, MUL, MAD, FMA, DIV, REM, NEG, ABS, MIN, MAX, NOT,
    SHL, SHR, AND, OR, XOR,
    SETP, SEL,
    BRA, RET, EXIT, BAR_SYNC,
    CP_ASYNC, MMA_SYNC,
    ATOM, RED, CALL,
    OTHER  // anything else; `mnemonic` holds the full text
};

// Number of 32-bit physical registers a virtual register of this type occupies.
// Predicates live in a separate predicate register file and cost 0 here.
uint32_t reg_slots(DataType t);
uint32_t type_bits(DataType t);
const char* to_string(DataType t);
const char* to_string(StateSpace s);
const char* to_string(Opcode o);
std::optional<DataType> parse_type(const std::string& s);   // "f32" -> F32
std::optional<StateSpace> parse_space(const std::string& s);  // "global" -> Global

struct VirtualReg {
    std::string name;  // e.g. "%r12", "%rd4", "%f8", "%p0"
    DataType type = DataType::B32;
    uint32_t id = 0;   // dense 0..N-1 index, used by bitset liveness

    bool operator==(const VirtualReg& o) const { return id == o.id; }
};

struct MemoryRef {
    StateSpace space = StateSpace::Global;
    std::optional<VirtualReg> base_reg;  // [%rd1 + 16]
    std::string symbol;                  // [kernel_param_0] or [__spill_depot + 8]
    int64_t offset = 0;
};

// Floating-point immediate; keeps the original spelling (0f3F800000) for lossless printing.
struct ImmFloat {
    double value = 0.0;
    std::string text;
};

// {%f1, %f2, ...} vector operand, or the `%p|%q` dual-destination form (separator '|').
struct RegVec {
    std::vector<VirtualReg> regs;
    char kind = '{';
};

// std::string operands carry branch labels, symbols and special registers (%tid.x).
using Operand = std::variant<VirtualReg, MemoryRef, int64_t, ImmFloat, std::string, RegVec>;

struct PredicateGuard {
    VirtualReg pred_reg;
    bool is_negated = false;
};

struct PTXInstruction {
    uint32_t line_number = 0;
    std::optional<PredicateGuard> predicate;  // @%p0 / @!%p0
    Opcode opcode = Opcode::OTHER;
    std::string mnemonic;                     // "mul.wide.s32"
    std::vector<std::string> modifiers;       // everything after the first dot
    std::optional<StateSpace> space;          // ld.global / st.shared
    DataType type = DataType::B32;            // first type suffix
    std::optional<DataType> src_type;         // second type suffix (cvt, mma)
    std::vector<Operand> operands;            // textual order
    bool has_dest = false;                    // operands[0] is written

    // Registers written by this instruction (destination operand only).
    std::vector<VirtualReg> defs() const;
    // Registers read: predicate, source operands, and address bases (even for stores).
    std::vector<VirtualReg> get_used_registers() const;

    // A predicated write may not execute, so the old value must stay live.
    bool is_conditional_def() const { return predicate.has_value() && has_dest; }
    bool is_branch() const { return opcode == Opcode::BRA; }
    bool is_terminator() const {
        return opcode == Opcode::BRA || opcode == Opcode::RET || opcode == Opcode::EXIT;
    }
    // Terminator that always leaves the block (no fall-through edge).
    bool is_unconditional_terminator() const { return is_terminator() && !predicate; }
    bool has_side_effects() const;
    std::string branch_target() const;  // "" if not a direct branch

    void replace_use(uint32_t old_id, const VirtualReg& nw);
    void replace_def(uint32_t old_id, const VirtualReg& nw);
};

// Decide opcode/space/type fields from the dotted mnemonic. Does not touch operands.
void classify_mnemonic(PTXInstruction& ins, const std::string& mnemonic);

}  // namespace ptxforge::ir
