#include "passes/DeadCodeElim.hpp"

#include "passes/LivenessAnalysis.hpp"

namespace ptxforge::passes {

using namespace ir;

namespace {

// Stage 1: flow-insensitive mark & sweep. A register is "needed" if a side-effecting instruction (store,
// branch, atomic, ...) transitively depends on it. This removes values that only feed themselves around
// a loop, which pure liveness cannot (their uses keep them live).
size_t mark_and_sweep(Function& fn, ControlFlowGraph& cfg) {
    BitSet needed(fn.regs.size());
    for (bool changed = true; changed;) {
        changed = false;
        for (auto& b : cfg.blocks())
            for (auto& ins : b.insts) {
                auto defs = ins.defs();
                bool essential = ins.has_side_effects() || !ins.has_dest || defs.empty();
                for (auto& d : defs) essential = essential || needed.test(d.id);
                if (!essential) continue;
                for (auto& u : ins.get_used_registers())
                    if (!needed.test(u.id)) { needed.set(u.id); changed = true; }
            }
    }
    size_t removed = 0;
    for (auto& b : cfg.blocks()) {
        std::vector<PTXInstruction> kept;
        kept.reserve(b.insts.size());
        for (auto& ins : b.insts) {
            auto defs = ins.defs();
            bool dead = ins.has_dest && !defs.empty() && !ins.has_side_effects();
            for (auto& d : defs) dead = dead && !needed.test(d.id);
            if (dead) ++removed; else kept.push_back(std::move(ins));
        }
        b.insts = std::move(kept);
    }
    return removed;
}

// Stage 2: flow-sensitive backward sweep using liveness (removes dead stores to registers that are
// read elsewhere, e.g. a value overwritten before its next use).
size_t liveness_sweep(Function& fn, ControlFlowGraph& cfg) {
    LivenessAnalysis live;
    live.run(fn, cfg);
    size_t removed = 0;
    for (auto& b : cfg.blocks()) {
        BitSet cur = b.live_out;
        std::vector<PTXInstruction> kept;  // collected in reverse
        kept.reserve(b.insts.size());
        for (size_t i = b.insts.size(); i-- > 0;) {
            const PTXInstruction& ins = b.insts[i];
            auto defs = ins.defs();
            bool dead = ins.has_dest && !defs.empty() && !ins.has_side_effects();
            for (auto& d : defs) dead = dead && !cur.test(d.id);
            if (dead) { ++removed; continue; }  // its uses stay un-marked, so chains collapse

            if (!ins.is_conditional_def())
                for (auto& d : defs) cur.reset(d.id);
            for (auto& u : ins.get_used_registers()) cur.set(u.id);
            if (ins.is_conditional_def())
                for (auto& d : defs) cur.set(d.id);
            kept.push_back(ins);
        }
        if (kept.size() != b.insts.size()) b.insts.assign(kept.rbegin(), kept.rend());
    }
    return removed;
}

}  // namespace

bool DeadCodeElim::run(Function& fn, ControlFlowGraph& cfg) {
    removed_ = 0;
    for (;;) {
        size_t n = mark_and_sweep(fn, cfg) + liveness_sweep(fn, cfg);
        removed_ += n;
        if (n == 0) break;
    }
    return removed_ > 0;
}

}  // namespace ptxforge::passes
