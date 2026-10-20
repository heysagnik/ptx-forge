#pragma once
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "ir/BasicBlock.hpp"
#include "ir/Module.hpp"

namespace ptxforge::ir {

struct NaturalLoop {
    uint32_t header = 0;
    std::vector<uint32_t> body;  // includes the header
};

class ControlFlowGraph {
public:
    // Splits the linear instruction stream at labels and after every terminator.
    // Throws std::runtime_error on a branch to an unknown label.
    static ControlFlowGraph build(const Function& fn);

    std::vector<BasicBlock>& blocks() { return blocks_; }
    const std::vector<BasicBlock>& blocks() const { return blocks_; }
    size_t edge_count() const;
    const std::vector<NaturalLoop>& loops() const { return loops_; }
    const std::vector<std::pair<uint32_t, uint32_t>>& back_edges() const { return back_edges_; }

    // Reverse post-order over blocks reachable from the entry (block 0).
    std::vector<uint32_t> reverse_post_order() const;

    // Rebuilds fn.instructions / fn.labels from the (possibly rewritten) blocks.
    void linearize(Function& fn) const;

    // Graphviz export; each node lists its instructions, back edges are dashed.
    std::string to_dot(const std::string& graph_name) const;

private:
    void compute_edges();
    void compute_loops();

    std::vector<BasicBlock> blocks_;
    std::vector<NaturalLoop> loops_;
    std::vector<std::pair<uint32_t, uint32_t>> back_edges_;
};

}  // namespace ptxforge::ir
