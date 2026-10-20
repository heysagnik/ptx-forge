#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "ir/PTXInstruction.hpp"
#include "support/BitSet.hpp"

namespace ptxforge::ir {

struct BasicBlock {
    uint32_t id = 0;
    std::vector<std::string> labels;  // every label that names this block (may be empty)
    std::vector<PTXInstruction> insts;
    std::vector<uint32_t> succs;
    std::vector<uint32_t> preds;
    uint32_t loop_depth = 0;  // number of natural loops containing this block

    // Filled by LivenessAnalysis (indexed by VirtualReg::id).
    BitSet ue_var;    // upward-exposed uses
    BitSet var_kill;  // unconditional definitions
    BitSet live_in;
    BitSet live_out;

    std::string name() const { return labels.empty() ? "BB" + std::to_string(id) : labels.front(); }
};

}  // namespace ptxforge::ir
