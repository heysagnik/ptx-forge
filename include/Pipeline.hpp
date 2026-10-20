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
    uint32_t max_regs = 255;
};

struct KernelAnalysis {
    std::string name;
    std::string target;
    std::string ptx_version;
    size_t blocks = 0;
    size_t edges = 0;
    size_t loops = 0;
    size_t dce_removed = 0;
    size_t instructions = 0;
    size_t declared_regs = 0;
    std::map<std::string, size_t> declared_by_type;
    uint32_t peak_slots = 0;
    uint32_t peak_block = 0;
    uint32_t peak_line = 0;
    uint32_t peak_loop_depth = 0;
    uint32_t max_live_preds = 0;
    passes::AllocResult alloc;
    std::string cfg_dot;
    std::string heatmap;
    std::string spilled_ptx;   // function text after spill rewriting (empty if nothing spilled)
};

KernelAnalysis analyze_function(const ir::Function& fn, const ir::Module& module, const AnalysisOptions& opts);

// A complete PTX module (header, globals, rewritten kernel) from a KernelAnalysis that spilled.
// Empty if the allocator inserted no spill code.
std::string emit_spilled_module(const ir::Module& module, const KernelAnalysis& analysis);

struct GroundTruth {
    bool ok = false;
    std::string source;
    std::string error;
    backend::CubinInfo cubin;
    std::optional<backend::KernelInfo> kernel;
    std::optional<backend::PtxasFunctionStats> ptxas;
};

GroundTruth fetch_ground_truth(backend::CubinProvider& provider, const std::string& ptx_source,
                               const std::string& kernel_name, uint32_t max_regs, const std::string& target_sm);

}  // namespace ptxforge
