#pragma once
#include <cstdint>
#include <map>
#include <optional>
#include <string>

#include "backend/CubinProvider.hpp"
#include "backend/ElfCubinParser.hpp"
#include "ir/Module.hpp"
#include "passes/RegAllocChaitin.hpp"

namespace ptxforge {

struct AnalysisOptions {
    uint32_t max_regs = 255;  // K for the allocator (--maxrregcount)
    bool dce = true;
    bool rewrite_spills = true;
};

// Everything the static pipeline learns about one kernel.
struct KernelAnalysis {
    std::string name;
    std::string target;
    std::string ptx_version;
    size_t blocks = 0;
    size_t edges = 0;
    size_t loops = 0;
    size_t dce_removed = 0;
    size_t instructions = 0;  // after DCE, before spill code
    size_t declared_regs = 0;
    std::map<std::string, size_t> declared_by_type;  // "b32" -> 34
    uint32_t peak_slots = 0;
    uint32_t peak_block = 0;
    uint32_t peak_line = 0;
    uint32_t peak_loop_depth = 0;
    uint32_t max_live_preds = 0;
    passes::AllocResult alloc;
    std::string cfg_dot;       // CFG after DCE
    std::string heatmap;       // per-instruction register pressure
    std::string spilled_ptx;   // function text after spill rewriting (empty if nothing spilled)
    uint64_t static_shared_bytes = 0;
};

// Runs CFG -> DCE -> liveness -> interference graph -> Chaitin-Briggs on a copy of `fn`.
KernelAnalysis analyze_function(const ir::Function& fn, const ir::Module& module, const AnalysisOptions& opts);

// What ptxas / the cubin actually contain for the same kernel.
struct GroundTruth {
    bool ok = false;
    std::string source;
    std::string error;
    backend::CubinInfo cubin;
    std::optional<backend::KernelInfo> kernel;
    std::optional<backend::PtxasFunctionStats> ptxas;  // from the JIT / ptxas info log
};

GroundTruth fetch_ground_truth(backend::CubinProvider& provider, const std::string& ptx_source,
                               const std::string& kernel_name, uint32_t max_regs, const std::string& target_sm);

}  // namespace ptxforge
