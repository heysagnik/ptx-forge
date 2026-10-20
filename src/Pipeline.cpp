#include "Pipeline.hpp"

#include <algorithm>
#include <sstream>

#include "ir/ControlFlowGraph.hpp"
#include "ir/PTXPrinter.hpp"
#include "passes/DeadCodeElim.hpp"
#include "passes/LivenessAnalysis.hpp"

namespace ptxforge {

using namespace ir;
using namespace passes;

namespace {
std::string build_heatmap(const ControlFlowGraph& cfg, const LivenessAnalysis& live, uint32_t limit) {
    std::ostringstream os;
    const uint32_t scale = std::max(live.max_pressure(), limit);
    const uint32_t width = 40;
    for (auto& b : cfg.blocks()) {
        os << "  " << b.name() << "  [BB" << b.id << ", loop depth " << b.loop_depth << "]\n";
        for (size_t i = 0; i < b.insts.size(); ++i) {
            uint32_t p = live.pressure()[b.id][i];
            uint32_t bar = scale ? (p * width + scale - 1) / scale : 0;
            char line[64];
            std::snprintf(line, sizeof(line), "  %5u %4u ", b.insts[i].line_number, p);
            os << line << '|' << std::string(bar, '#') << std::string(width - bar, ' ') << '|'
               << (p > limit ? '!' : ' ') << ' ' << format_instruction(b.insts[i]) << '\n';
        }
    }
    return os.str();
}
}  // namespace

KernelAnalysis analyze_function(const Function& original, const Module& module, const AnalysisOptions& opts) {
    Function fn = original;
    KernelAnalysis out;
    out.name = fn.name;
    out.target = module.target;
    out.ptx_version = module.version;
    out.declared_regs = fn.regs.size();
    for (auto& r : fn.regs) ++out.declared_by_type[to_string(r.type)];

    ControlFlowGraph cfg = ControlFlowGraph::build(fn);
    DeadCodeElim dce;
    dce.run(fn, cfg);
    out.dce_removed = dce.removed();
    out.blocks = cfg.blocks().size();
    out.edges = cfg.edge_count();
    out.loops = cfg.loops().size();
    for (auto& b : cfg.blocks()) out.instructions += b.insts.size();
    out.cfg_dot = cfg.to_dot(fn.name);

    LivenessAnalysis live;
    live.run(fn, cfg);
    out.peak_slots = live.max_pressure();
    out.peak_block = live.peak().block;
    out.peak_line = live.peak().line;
    out.peak_loop_depth = cfg.blocks()[live.peak().block].loop_depth;
    out.max_live_preds = live.max_live_predicates();
    out.heatmap = build_heatmap(cfg, live, opts.max_regs);

    AllocOptions ao;
    ao.max_regs = opts.max_regs;
    RegAllocChaitin alloc(ao);
    bool rewrote = alloc.run(fn, cfg);
    out.alloc = alloc.result();
    if (rewrote) {
        cfg.linearize(fn);
        out.spilled_ptx = print_function(fn);
    }
    return out;
}

std::string emit_spilled_module(const Module& module, const KernelAnalysis& analysis) {
    if (analysis.spilled_ptx.empty()) return {};
    Module head;
    head.version = module.version;
    head.target = module.target;
    head.target_options = module.target_options;
    head.address_size = module.address_size;
    head.globals = module.globals;
    return print_module(head) + "\n" + analysis.spilled_ptx;
}

GroundTruth fetch_ground_truth(backend::CubinProvider& provider, const std::string& ptx_source,
                               const std::string& kernel_name, uint32_t max_regs, const std::string& target_sm) {
    GroundTruth gt;
    gt.source = provider.name();
    backend::CompileOptions co;
    co.max_regs = max_regs < 255 ? max_regs : 0;
    co.target_sm = target_sm;
    backend::CompileResult cr = provider.compile(ptx_source, co);
    if (!cr.ok) {
        gt.error = cr.error_log.empty() ? "compilation failed" : cr.error_log;
        return gt;
    }
    try {
        gt.cubin = backend::parse_cubin(cr.cubin);
    } catch (const std::exception& e) {
        gt.error = std::string("cubin parse error: ") + e.what();
        return gt;
    }
    if (const auto* k = gt.cubin.find_kernel(kernel_name)) gt.kernel = *k;
    else if (gt.cubin.kernels.size() == 1) gt.kernel = gt.cubin.kernels.front();  // pre-built cubin with a different name
    auto it = cr.ptxas_stats.find(kernel_name);
    if (it != cr.ptxas_stats.end()) gt.ptxas = it->second;
    gt.ok = gt.kernel.has_value();
    if (!gt.ok) gt.error = "kernel '" + kernel_name + "' not found in cubin";
    return gt;
}

}  // namespace ptxforge
