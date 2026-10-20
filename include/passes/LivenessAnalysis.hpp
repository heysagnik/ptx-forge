#pragma once
#include <cstdint>
#include <vector>

#include "passes/Pass.hpp"
#include "support/BitSet.hpp"

namespace ptxforge::passes {

struct PressurePoint {
    uint32_t block = 0;
    uint32_t index = 0;  // instruction index within the block
    uint32_t line = 0;   // source line of the instruction
    uint32_t slots = 0;  // 32-bit register equivalents live at this point
};

// Iterative backward dataflow over the CFG:
//   LIVE_OUT(B) = U LIVE_IN(S)        LIVE_IN(B) = UEVar(B) U (LIVE_OUT(B) \ VarKill(B))
// A predicated write is a *conditional* definition: it reads the old value and kills nothing.
class LivenessAnalysis : public Pass {
public:
    const char* name() const override { return "liveness"; }
    bool run(ir::Function& fn, ir::ControlFlowGraph& cfg) override;

    // Live set *after* instruction [block][index].
    const std::vector<std::vector<BitSet>>& live_after() const { return live_after_; }
    // 32-bit-equivalent register pressure at each instruction: max(live before, live after + defs).
    const std::vector<std::vector<uint32_t>>& pressure() const { return pressure_; }
    uint32_t max_pressure() const { return peak_.slots; }
    const PressurePoint& peak() const { return peak_; }
    uint32_t max_live_predicates() const { return max_preds_; }
    // Width of each virtual register in 32-bit slots (0 for predicates).
    const std::vector<uint32_t>& slots() const { return slots_; }

private:
    std::vector<std::vector<BitSet>> live_after_;
    std::vector<std::vector<uint32_t>> pressure_;
    std::vector<uint32_t> slots_;
    PressurePoint peak_;
    uint32_t max_preds_ = 0;
};

}  // namespace ptxforge::passes
