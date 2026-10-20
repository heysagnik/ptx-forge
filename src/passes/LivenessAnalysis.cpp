#include "passes/LivenessAnalysis.hpp"

#include <algorithm>

namespace ptxforge::passes {

using namespace ir;

bool LivenessAnalysis::run(Function& fn, ControlFlowGraph& cfg) {
    const size_t nregs = fn.regs.size();
    auto& blocks = cfg.blocks();

    slots_.assign(nregs, 0);
    for (size_t i = 0; i < nregs; ++i) slots_[i] = reg_slots(fn.regs[i].type);

    for (auto& b : blocks) {
        b.ue_var = BitSet(nregs);
        b.var_kill = BitSet(nregs);
        b.live_in = BitSet(nregs);
        b.live_out = BitSet(nregs);
        for (auto& ins : b.insts) {
            for (auto& u : ins.get_used_registers())
                if (!b.var_kill.test(u.id)) b.ue_var.set(u.id);
            for (auto& d : ins.defs()) {
                if (ins.is_conditional_def()) {
                    if (!b.var_kill.test(d.id)) b.ue_var.set(d.id);  // old value flows through
                } else {
                    b.var_kill.set(d.id);
                }
            }
        }
    }

    for (bool changed = true; changed;) {
        changed = false;
        for (size_t k = blocks.size(); k-- > 0;) {
            auto& b = blocks[k];
            BitSet out(nregs);
            for (auto s : b.succs) out.union_with(blocks[s].live_in);
            BitSet in = out;
            in.subtract(b.var_kill);
            in.union_with(b.ue_var);
            if (!(out == b.live_out)) { b.live_out = std::move(out); changed = true; }
            if (!(in == b.live_in)) { b.live_in = std::move(in); changed = true; }
        }
    }

    live_after_.assign(blocks.size(), {});
    pressure_.assign(blocks.size(), {});
    peak_ = {};
    max_preds_ = 0;
    auto weight = [&](const BitSet& s) {
        uint32_t w = 0;
        s.for_each([&](size_t r) { w += slots_[r]; });
        return w;
    };
    auto pred_count = [&](const BitSet& s) {
        uint32_t c = 0;
        s.for_each([&](size_t r) { c += fn.regs[r].type == DataType::Pred; });
        return c;
    };
    for (auto& b : blocks) {
        const size_t n = b.insts.size();
        live_after_[b.id].assign(n, BitSet());
        pressure_[b.id].assign(n, 0);
        BitSet live = b.live_out;
        for (size_t i = n; i-- > 0;) {
            const PTXInstruction& ins = b.insts[i];
            live_after_[b.id][i] = live;
            auto defs = ins.defs();

            uint32_t after = weight(live);
            for (auto& d : defs) if (!live.test(d.id)) after += slots_[d.id];  // dead defs still occupy a register
            uint32_t preds = pred_count(live);

            if (!ins.is_conditional_def())
                for (auto& d : defs) live.reset(d.id);
            for (auto& u : ins.get_used_registers()) live.set(u.id);
            if (ins.is_conditional_def())
                for (auto& d : defs) live.set(d.id);

            uint32_t p = std::max(weight(live), after);
            pressure_[b.id][i] = p;
            max_preds_ = std::max({max_preds_, preds, pred_count(live)});
            if (p > peak_.slots) peak_ = {b.id, static_cast<uint32_t>(i), ins.line_number, p};
        }
    }
    return false;
}

}  // namespace ptxforge::passes
