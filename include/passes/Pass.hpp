#pragma once
#include "ir/ControlFlowGraph.hpp"
#include "ir/Module.hpp"

namespace ptxforge::passes {

// A pass works on one function and its CFG. Returns true if it changed the IR.
class Pass {
public:
    virtual ~Pass() = default;
    virtual const char* name() const = 0;
    virtual bool run(ir::Function& fn, ir::ControlFlowGraph& cfg) = 0;
};

}  // namespace ptxforge::passes
