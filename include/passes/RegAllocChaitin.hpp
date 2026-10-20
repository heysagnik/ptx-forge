#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "passes/LivenessAnalysis.hpp"
#include "passes/Pass.hpp"

namespace ptxforge::passes {

// Undirected register interference graph. Two virtual registers interfere when one is defined
// while the other is live; only registers of the same class (general vs predicate) can interfere.
class InterferenceGraph {
public:
    static InterferenceGraph build(const ir::Function& fn, const ir::ControlFlowGraph& cfg,
                                   const LivenessAnalysis& live);

    size_t num_nodes() const { return nodes_.size(); }
    size_t num_edges() const { return edges_; }
    bool has_edge(uint32_t a, uint32_t b) const { return rows_[a].test(b); }
    const std::vector<uint32_t>& nodes() const { return nodes_; }
    const std::vector<uint32_t>& neighbors(uint32_t r) const { return adj_[r]; }

private:
    std::vector<char> present_;
    std::vector<uint32_t> nodes_;
    std::vector<BitSet> rows_;
    std::vector<std::vector<uint32_t>> adj_;
    size_t edges_ = 0;
};

struct AllocOptions {
    uint32_t max_regs = 255;
    uint32_t max_preds = 7;    // predicate register file size (P0..P6)
    uint32_t max_rounds = 24;
    bool rewrite_spills = true;
};

struct SpillCandidate {
    uint32_t reg = 0;
    std::string name;
    double cost = 0.0;  // sum(10^loop_depth) over defs+uses, divided by degree
};

struct AllocResult {
    uint32_t k = 0;
    uint32_t peak_pressure = 0;
    bool fits_without_spill = false;
    bool converged = false;
    uint32_t rounds = 0;
    uint32_t regs_used = 0;
    uint32_t preds_used = 0;
    bool pred_overflow = false;
    size_t rig_nodes = 0;
    size_t rig_edges = 0;
    std::vector<SpillCandidate> potential_spills;
    std::vector<SpillCandidate> actual_spills;
    uint32_t spill_stores = 0;
    uint32_t spill_loads = 0;
    uint32_t frame_bytes = 0;
    std::unordered_map<uint32_t, uint32_t> assignment;
};

// Chaitin-Briggs optimistic graph coloring with width-aware (1/2/4 slot, aligned) nodes.
// Spilled registers are rewritten to .local loads/stores and allocation is re-run until it converges.
class RegAllocChaitin : public Pass {
public:
    explicit RegAllocChaitin(AllocOptions opts = {}) : opts_(opts) {}
    const char* name() const override { return "regalloc-chaitin-briggs"; }
    bool run(ir::Function& fn, ir::ControlFlowGraph& cfg) override;
    const AllocResult& result() const { return result_; }

private:
    AllocOptions opts_;
    AllocResult result_;
};

struct SpillRewriteResult {
    uint32_t loads = 0;
    uint32_t stores = 0;
    std::vector<uint32_t> temps;  // fresh short-lived registers (must never be spilled again)
};

// Replaces every def/use of the given registers with a fresh temporary plus st.local / ld.local
// against `__spill_depot`, growing fn.spill_frame_bytes.
SpillRewriteResult rewrite_spills(ir::Function& fn, ir::ControlFlowGraph& cfg,
                                  const std::vector<uint32_t>& spilled);

}  // namespace ptxforge::passes
