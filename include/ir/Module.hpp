#pragma once
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "ir/PTXInstruction.hpp"

namespace ptxforge::ir {

struct Param {
    std::string name;
    uint64_t bytes = 0;
    std::string raw;  // original declaration text, e.g. ".param .u64 vecadd_param_0"
};

// A variable declaration (.shared / .local / .global / .const / .param / pragma) kept verbatim.
struct VarDecl {
    std::string name;
    StateSpace space = StateSpace::Global;
    uint64_t bytes = 0;
    uint32_t align = 1;
    std::string raw;
};

struct Function {
    std::string name;
    bool is_entry = true;
    std::string linkage;                 // ".visible", ".weak", ...
    std::vector<Param> params;
    std::vector<Param> ret_params;       // .func only
    std::string header_directives;       // .maxntid / .reqntid / .minnctapersm ...
    std::vector<VarDecl> vars;           // body-level declarations, in source order
    std::vector<VirtualReg> regs;        // indexed by VirtualReg::id
    std::vector<PTXInstruction> instructions;
    std::vector<std::pair<std::string, size_t>> labels;  // label -> index of following instruction

    // Spill frame created by the register allocator (0 = none).
    uint32_t spill_frame_bytes = 0;
    uint32_t spill_frame_align = 4;

    const VirtualReg& reg(uint32_t id) const { return regs[id]; }
    VirtualReg add_reg(const std::string& name, DataType type) {
        VirtualReg r{name, type, static_cast<uint32_t>(regs.size())};
        regs.push_back(r);
        return r;
    }
    uint64_t static_shared_bytes() const {
        uint64_t n = 0;
        for (auto& v : vars) if (v.space == StateSpace::Shared) n += v.bytes;
        return n;
    }
};

struct Module {
    std::string version;                     // "8.0"
    std::string target;                      // "sm_80"
    std::vector<std::string> target_options; // "debug", "texmode_independent", ...
    uint32_t address_size = 64;
    std::vector<VarDecl> globals;
    std::vector<Function> functions;

    Function* find(const std::string& name) {
        for (auto& f : functions) if (f.name == name) return &f;
        return nullptr;
    }
};

}  // namespace ptxforge::ir
