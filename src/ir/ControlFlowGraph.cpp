#include "ir/ControlFlowGraph.hpp"

#include <algorithm>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "ir/PTXPrinter.hpp"

namespace ptxforge::ir {

ControlFlowGraph ControlFlowGraph::build(const Function& fn) {
    ControlFlowGraph cfg;
    const size_t n = fn.instructions.size();
    std::vector<std::vector<std::string>> labels_at(n + 1);
    for (auto& [name, idx] : fn.labels) labels_at[std::min(idx, n)].push_back(name);

    BasicBlock* cur = nullptr;
    auto start_block = [&]() {
        cfg.blocks_.emplace_back();
        cfg.blocks_.back().id = static_cast<uint32_t>(cfg.blocks_.size() - 1);
        cur = &cfg.blocks_.back();
    };
    // `cur` points into the vector, so reserve up-front to keep it stable.
    cfg.blocks_.reserve(n + labels_at.size() + 1);

    for (size_t i = 0; i < n; ++i) {
        if (!labels_at[i].empty()) {
            if (!cur || !cur->insts.empty()) start_block();
            for (auto& l : labels_at[i]) cur->labels.push_back(l);
        } else if (!cur) {
            start_block();
        }
        cur->insts.push_back(fn.instructions[i]);
        if (fn.instructions[i].is_terminator()) cur = nullptr;
    }
    if (!labels_at[n].empty()) {
        if (!cur || !cur->insts.empty()) start_block();
        for (auto& l : labels_at[n]) cur->labels.push_back(l);
    }
    if (cfg.blocks_.empty()) start_block();

    cfg.compute_edges();
    cfg.compute_loops();
    return cfg;
}

void ControlFlowGraph::compute_edges() {
    std::unordered_map<std::string, uint32_t> by_label;
    for (auto& b : blocks_)
        for (auto& l : b.labels) by_label[l] = b.id;

    for (auto& b : blocks_) { b.succs.clear(); b.preds.clear(); }
    auto add_edge = [&](uint32_t from, uint32_t to) {
        auto& s = blocks_[from].succs;
        if (std::find(s.begin(), s.end(), to) != s.end()) return;
        s.push_back(to);
        blocks_[to].preds.push_back(from);
    };
    for (auto& b : blocks_) {
        bool falls_through = true;
        if (!b.insts.empty()) {
            const PTXInstruction& last = b.insts.back();
            if (last.is_branch()) {
                std::string target = last.branch_target();
                auto it = by_label.find(target);
                if (it == by_label.end())
                    throw std::runtime_error("line " + std::to_string(last.line_number) +
                                             ": branch to unknown label '" + target + "'");
                add_edge(b.id, it->second);
            }
            if (last.is_unconditional_terminator()) falls_through = false;
        }
        if (falls_through && b.id + 1 < blocks_.size()) add_edge(b.id, b.id + 1);
    }
}

size_t ControlFlowGraph::edge_count() const {
    size_t e = 0;
    for (auto& b : blocks_) e += b.succs.size();
    return e;
}

std::vector<uint32_t> ControlFlowGraph::reverse_post_order() const {
    std::vector<uint32_t> post;
    std::vector<char> seen(blocks_.size(), 0);
    std::vector<std::pair<uint32_t, size_t>> stack;
    if (blocks_.empty()) return post;
    stack.emplace_back(0, 0);
    seen[0] = 1;
    while (!stack.empty()) {
        auto& [b, i] = stack.back();
        if (i < blocks_[b].succs.size()) {
            uint32_t s = blocks_[b].succs[i++];
            if (!seen[s]) { seen[s] = 1; stack.emplace_back(s, 0); }
        } else {
            post.push_back(b);
            stack.pop_back();
        }
    }
    std::reverse(post.begin(), post.end());
    return post;
}

void ControlFlowGraph::compute_loops() {
    loops_.clear();
    back_edges_.clear();
    for (auto& b : blocks_) b.loop_depth = 0;
    const size_t n = blocks_.size();
    auto rpo = reverse_post_order();
    if (rpo.empty()) return;

    std::vector<char> reachable(n, 0);
    for (auto b : rpo) reachable[b] = 1;
    std::vector<BitSet> dom(n, BitSet(n));
    for (size_t b = 0; b < n; ++b)
        if (reachable[b]) for (size_t i = 0; i < n; ++i) if (reachable[i]) dom[b].set(i);
    dom[0].clear();
    dom[0].set(0);
    for (bool changed = true; changed;) {
        changed = false;
        for (auto b : rpo) {
            if (b == 0) continue;
            BitSet nd;
            bool first = true;
            for (auto p : blocks_[b].preds) {
                if (!reachable[p]) continue;
                if (first) { nd = dom[p]; first = false; } else nd.intersect_with(dom[p]);
            }
            if (first) continue;
            nd.set(b);
            if (!(nd == dom[b])) { dom[b] = nd; changed = true; }
        }
    }

    // Back edge t -> h where h dominates t. Merge loops sharing a header.
    std::unordered_map<uint32_t, std::vector<uint32_t>> tails_by_header;
    for (auto t : rpo)
        for (auto s : blocks_[t].succs)
            if (reachable[s] && dom[t].test(s)) {
                back_edges_.emplace_back(t, s);
                tails_by_header[s].push_back(t);
            }
    std::vector<uint32_t> headers;
    for (auto& [h, _] : tails_by_header) headers.push_back(h);
    std::sort(headers.begin(), headers.end());
    for (auto h : headers) {
        BitSet body(n);
        body.set(h);
        std::vector<uint32_t> work;
        for (auto t : tails_by_header[h])
            if (!body.test(t)) { body.set(t); work.push_back(t); }
        while (!work.empty()) {
            uint32_t b = work.back();
            work.pop_back();
            for (auto p : blocks_[b].preds)
                if (reachable[p] && !body.test(p)) { body.set(p); work.push_back(p); }
        }
        NaturalLoop loop;
        loop.header = h;
        body.for_each([&](size_t b) { loop.body.push_back(static_cast<uint32_t>(b)); ++blocks_[b].loop_depth; });
        loops_.push_back(std::move(loop));
    }
}

void ControlFlowGraph::linearize(Function& fn) const {
    fn.instructions.clear();
    fn.labels.clear();
    for (auto& b : blocks_) {
        for (auto& l : b.labels) fn.labels.emplace_back(l, fn.instructions.size());
        for (auto& i : b.insts) fn.instructions.push_back(i);
    }
}

namespace {
std::string dot_escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\' || c == '{' || c == '}' || c == '<' || c == '>' || c == '|') o += '\\';
        o += c;
    }
    return o;
}
}  // namespace

std::string ControlFlowGraph::to_dot(const std::string& graph_name) const {
    std::ostringstream os;
    os << "digraph \"" << dot_escape(graph_name) << "\" {\n"
       << "  node [shape=box, fontname=\"Courier\", fontsize=10];\n";
    for (auto& b : blocks_) {
        os << "  bb" << b.id << " [label=\"" << dot_escape(b.name()) << "  (loop depth " << b.loop_depth << ")\\l";
        for (auto& i : b.insts) os << dot_escape(format_instruction(i)) << "\\l";
        os << "\"];\n";
    }
    for (auto& b : blocks_)
        for (auto s : b.succs) {
            bool back = std::find(back_edges_.begin(), back_edges_.end(), std::make_pair(b.id, s)) != back_edges_.end();
            os << "  bb" << b.id << " -> bb" << s;
            if (back) os << " [style=dashed, color=red]";
            os << ";\n";
        }
    os << "}\n";
    return os.str();
}

}  // namespace ptxforge::ir
