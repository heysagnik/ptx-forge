#pragma once
#include <cstddef>

#include "passes/Pass.hpp"

namespace ptxforge::passes {

// Removes side-effect-free instructions whose destination registers are never read
// (liveness-driven, iterated to a fixed point so dead chains and dead loop-carried values go away).
class DeadCodeElim : public Pass {
public:
    const char* name() const override { return "dce"; }
    bool run(ir::Function& fn, ir::ControlFlowGraph& cfg) override;
    size_t removed() const { return removed_; }

private:
    size_t removed_ = 0;
};

}  // namespace ptxforge::passes
