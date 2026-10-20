#include "passes/RegAllocChaitin.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>

namespace ptxforge::passes {

using namespace ir;

InterferenceGraph InterferenceGraph::build(const Function& fn, const ControlFlowGraph& cfg,
                                           const LivenessAnalysis& live) {
    InterferenceGraph g;
    const size_t n = fn.regs.size();
    g.present_.assign(n, 0);
    g.rows_.assign(n, BitSet(n));
    g.adj_.assign(n, {});

    auto is_pred = [&](size_t r) { return fn.regs[r].type == DataType::Pred; };
    auto add_edge = [&](size_t a, size_t b) {
        if (a == b || is_pred(a) != is_pred(b)) return;
        if (!g.rows_[a].test(b)) { g.rows_[a].set(b); g.rows_[b].set(a); ++g.edges_; }
    };

    for (auto& b : cfg.blocks()) {
        for (size_t i = 0; i < b.insts.size(); ++i) {
            const PTXInstruction& ins = b.insts[i];
            for (auto& u : ins.get_used_registers()) g.present_[u.id] = 1;
            auto defs = ins.defs();
            for (auto& d : defs) g.present_[d.id] = 1;
            // A definition interferes with everything live right after it (and with co-defined regs).
            const BitSet& after = live.live_after()[b.id][i];
            for (auto& d : defs) {
                after.for_each([&](size_t l) { add_edge(d.id, l); });
                for (auto& d2 : defs) add_edge(d.id, d2.id);
            }
        }
    }
    // Values live into the entry block (uninitialised / parameter-like) coexist at function start.
    if (!cfg.blocks().empty()) {
        std::vector<size_t> entry;
        cfg.blocks()[0].live_in.for_each([&](size_t r) { entry.push_back(r); });
        for (size_t x = 0; x < entry.size(); ++x)
            for (size_t y = x + 1; y < entry.size(); ++y) add_edge(entry[x], entry[y]);
    }

    for (size_t r = 0; r < n; ++r) {
        if (!g.present_[r]) continue;
        g.nodes_.push_back(static_cast<uint32_t>(r));
        g.rows_[r].for_each([&](size_t o) { g.adj_[r].push_back(static_cast<uint32_t>(o)); });
    }
    return g;
}

namespace {

struct ColorOutcome {
    std::unordered_map<uint32_t, uint32_t> color;  // reg -> first slot
    std::vector<uint32_t> spilled;                 // Select failed
    std::vector<SpillCandidate> potentials;        // pushed as potential spills during Simplify
};

// `width[r]` is 1, 2 or 4 slots. A node is trivially colorable ("degree < K") when its neighbours
// cannot block every aligned run of its width: blocked_runs < K / width.
ColorOutcome color_class(const InterferenceGraph& g, const std::vector<uint32_t>& nodes,
                         const std::vector<uint32_t>& width, uint32_t K, const std::vector<double>& cost,
                         const std::unordered_set<uint32_t>& nospill, const Function& fn) {
    const size_t n = width.size();
    std::vector<char> in_class(n, 0), removed(n, 1);
    for (auto v : nodes) { in_class[v] = 1; removed[v] = 0; }

    // Remaining-neighbour counts by neighbour width.
    std::vector<uint32_t> c1(n, 0), c2(n, 0), c4(n, 0);
    auto bump = [&](uint32_t v, uint32_t w, int d) {
        uint32_t& c = w == 1 ? c1[v] : (w == 2 ? c2[v] : c4[v]);
        c = static_cast<uint32_t>(static_cast<int>(c) + d);
    };
    for (auto v : nodes)
        for (auto u : g.neighbors(v))
            if (in_class[u]) bump(v, width[u], +1);

    auto blocked = [&](uint32_t v) -> uint32_t {
        switch (width[v]) {
            case 1: return c1[v] + 2 * c2[v] + 4 * c4[v];
            case 2: return c1[v] + c2[v] + 2 * c4[v];
            default: return c1[v] + c2[v] + c4[v];
        }
    };
    auto trivially_colorable = [&](uint32_t v) { return blocked(v) < K / width[v]; };
    auto degree = [&](uint32_t v) { return c1[v] + c2[v] + c4[v]; };

    ColorOutcome out;
    std::vector<std::pair<uint32_t, bool>> stack;  // (reg, potential spill)
    std::vector<uint32_t> work;
    std::vector<char> queued(n, 0);
    for (auto v : nodes) if (trivially_colorable(v)) { work.push_back(v); queued[v] = 1; }

    auto remove_node = [&](uint32_t v, bool potential) {
        removed[v] = 1;
        stack.emplace_back(v, potential);
        for (auto u : g.neighbors(v)) {
            if (!in_class[u] || removed[u]) continue;
            bump(u, width[v], -1);
            if (!queued[u] && trivially_colorable(u)) { work.push_back(u); queued[u] = 1; }
        }
    };

    size_t remaining = nodes.size();
    while (remaining > 0) {
        if (!work.empty()) {  // 1. Simplify
            uint32_t v = work.back();
            work.pop_back();
            if (removed[v]) continue;
            remove_node(v, false);
            --remaining;
            continue;
        }
        // 2. Potential spill: cheapest cost/degree among the remaining nodes.
        uint32_t best = 0;
        double best_metric = std::numeric_limits<double>::infinity();
        bool found = false;
        for (auto v : nodes) {
            if (removed[v]) continue;
            double metric = nospill.count(v) ? std::numeric_limits<double>::max()
                                             : cost[v] / std::max<uint32_t>(1, degree(v));
            if (!found || metric < best_metric) { best = v; best_metric = metric; found = true; }
        }
        out.potentials.push_back({best, fn.regs[best].name, best_metric});
        remove_node(best, true);
        --remaining;
    }

    // 3. Select: pop and assign the lowest aligned run not used by a colored neighbour.
    std::vector<char> occupied(K, 0);
    for (size_t s = stack.size(); s-- > 0;) {
        uint32_t v = stack[s].first;
        std::fill(occupied.begin(), occupied.end(), 0);
        for (auto u : g.neighbors(v)) {
            auto it = out.color.find(u);
            if (it == out.color.end()) continue;
            for (uint32_t k = it->second; k < it->second + width[u] && k < K; ++k) occupied[k] = 1;
        }
        bool placed = false;
        for (uint32_t start = 0; start + width[v] <= K && !placed; start += width[v]) {
            bool free_run = true;
            for (uint32_t k = start; k < start + width[v]; ++k) free_run = free_run && !occupied[k];
            if (free_run) { out.color[v] = start; placed = true; }
        }
        if (!placed) out.spilled.push_back(v);
    }
    return out;
}

}  // namespace

bool RegAllocChaitin::run(Function& fn, ControlFlowGraph& cfg) {
    result_ = {};
    result_.k = opts_.max_regs;
    std::unordered_set<uint32_t> nospill;
    bool rewrote = false;

    for (uint32_t round = 0; round < opts_.max_rounds; ++round) {
        result_.rounds = round + 1;
        LivenessAnalysis live;
        live.run(fn, cfg);
        InterferenceGraph rig = InterferenceGraph::build(fn, cfg, live);
        if (round == 0) {
            result_.peak_pressure = live.max_pressure();
            result_.rig_nodes = rig.num_nodes();
            result_.rig_edges = rig.num_edges();
        }

        // Spill cost: sum over defs+uses of 10^loop_depth.
        std::vector<double> cost(fn.regs.size(), 0.0);
        for (auto& b : cfg.blocks()) {
            double w = std::pow(10.0, std::min<uint32_t>(b.loop_depth, 8));
            for (auto& ins : b.insts) {
                for (auto& u : ins.get_used_registers()) cost[u.id] += w;
                for (auto& d : ins.defs()) cost[d.id] += w;
            }
        }

        std::vector<uint32_t> gen_nodes, pred_nodes, width(fn.regs.size(), 1);
        for (auto v : rig.nodes()) {
            if (fn.regs[v].type == DataType::Pred) { pred_nodes.push_back(v); continue; }
            width[v] = std::max<uint32_t>(1, reg_slots(fn.regs[v].type));
            gen_nodes.push_back(v);
        }
        // Registers that cannot be spilled (spill temps, 128-bit values) are never chosen as candidates.
        for (auto v : gen_nodes) if (width[v] > 2) nospill.insert(v);

        ColorOutcome gen = color_class(rig, gen_nodes, width, opts_.max_regs, cost, nospill, fn);
        ColorOutcome prd = color_class(rig, pred_nodes, std::vector<uint32_t>(fn.regs.size(), 1),
                                       opts_.max_preds, cost, nospill, fn);

        result_.pred_overflow = !prd.spilled.empty();
        result_.preds_used = 0;
        for (auto& [r, c] : prd.color) result_.preds_used = std::max(result_.preds_used, c + 1);

        if (round == 0) {
            result_.fits_without_spill = gen.spilled.empty();
            std::sort(gen.potentials.begin(), gen.potentials.end(),
                      [](auto& a, auto& b) { return a.cost < b.cost; });
            result_.potential_spills = gen.potentials;
        }

        if (gen.spilled.empty()) {
            result_.converged = true;
            result_.assignment = gen.color;
            result_.regs_used = 0;
            for (auto& [r, c] : gen.color) result_.regs_used = std::max(result_.regs_used, c + width[r]);
            break;
        }

        for (auto v : gen.spilled)
            result_.actual_spills.push_back({v, fn.regs[v].name, cost[v]});
        bool stuck = !opts_.rewrite_spills ||
                     std::any_of(gen.spilled.begin(), gen.spilled.end(), [&](uint32_t v) { return nospill.count(v) > 0; });
        if (stuck) break;

        SpillRewriteResult rw = rewrite_spills(fn, cfg, gen.spilled);
        rewrote = true;
        result_.spill_loads += rw.loads;
        result_.spill_stores += rw.stores;
        for (auto t : rw.temps) nospill.insert(t);
    }
    result_.frame_bytes = fn.spill_frame_bytes;
    return rewrote;
}

namespace {
PTXInstruction make_spill_access(bool is_load, const VirtualReg& r, int64_t offset, uint32_t line) {
    PTXInstruction ins;
    ins.line_number = line;
    classify_mnemonic(ins, std::string(is_load ? "ld" : "st") + ".local.b" + std::to_string(type_bits(r.type)));
    MemoryRef m;
    m.space = StateSpace::Local;
    m.symbol = "__spill_depot";
    m.offset = offset;
    if (is_load) { ins.operands = {r, m}; ins.has_dest = true; }
    else         { ins.operands = {m, r}; ins.has_dest = false; }
    return ins;
}
}  // namespace

SpillRewriteResult rewrite_spills(Function& fn, ControlFlowGraph& cfg, const std::vector<uint32_t>& spilled) {
    SpillRewriteResult res;
    std::unordered_map<uint32_t, int64_t> slot;
    for (auto id : spilled) {
        uint32_t bytes = std::max<uint32_t>(4, type_bits(fn.regs[id].type) / 8);
        uint32_t off = (fn.spill_frame_bytes + bytes - 1) / bytes * bytes;
        slot[id] = off;
        fn.spill_frame_bytes = off + bytes;
        fn.spill_frame_align = std::max(fn.spill_frame_align, bytes);
    }

    for (auto& b : cfg.blocks()) {
        std::vector<PTXInstruction> out;
        out.reserve(b.insts.size());
        for (auto ins : b.insts) {
            std::vector<PTXInstruction> before, after;
            std::unordered_map<uint32_t, VirtualReg> temp_of;
            auto temp_for = [&](uint32_t id) -> VirtualReg {
                auto it = temp_of.find(id);
                if (it != temp_of.end()) return it->second;
                const VirtualReg& orig = fn.regs[id];
                VirtualReg t = fn.add_reg("%__sp" + std::to_string(fn.regs.size()), orig.type);
                res.temps.push_back(t.id);
                temp_of[id] = t;
                return t;
            };

            // Reloads for every spilled source (and for the old value of a conditionally-written dest).
            std::vector<uint32_t> loaded;
            auto need_load = [&](uint32_t id) {
                if (!slot.count(id) || std::find(loaded.begin(), loaded.end(), id) != loaded.end()) return;
                loaded.push_back(id);
                VirtualReg t = temp_for(id);
                before.push_back(make_spill_access(true, t, slot[id], ins.line_number));
                ++res.loads;
            };
            auto used = ins.get_used_registers();
            for (auto& u : used) need_load(u.id);
            auto defs = ins.defs();
            if (ins.is_conditional_def()) for (auto& d : defs) need_load(d.id);

            for (auto id : loaded) ins.replace_use(id, temp_of[id]);
            for (auto& d : defs) {
                if (!slot.count(d.id)) continue;
                VirtualReg t = temp_for(d.id);
                ins.replace_def(d.id, t);
                after.push_back(make_spill_access(false, t, slot[d.id], ins.line_number));
                ++res.stores;
            }
            for (auto& x : before) out.push_back(std::move(x));
            out.push_back(std::move(ins));
            for (auto& x : after) out.push_back(std::move(x));
        }
        b.insts = std::move(out);
    }
    return res;
}

}  // namespace ptxforge::passes
