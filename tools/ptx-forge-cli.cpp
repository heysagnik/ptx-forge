// ptx-forge: PTX static analyzer, Chaitin-Briggs register allocator and ELF cubin inspector.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>

#include "Pipeline.hpp"
#include "backend/DriverJITLinker.hpp"
#include "frontend/Parser.hpp"
#include "ir/PTXPrinter.hpp"

using namespace ptxforge;

namespace {

constexpr const char* kRule =
    "================================================================================";
constexpr const char* kThin =
    "--------------------------------------------------------------------------------";

void usage() {
    std::cout <<
        "ptx-forge - PTX static analyzer, register allocator and cubin inspector\n"
        "\n"
        "usage:\n"
        "  ptx-forge analyze <kernel.ptx> [options]\n"
        "  ptx-forge inspect <file.cubin>\n"
        "\n"
        "analyze options:\n"
        "  --maxrregcount=K        physical register budget for the allocator (default 255)\n"
        "  --kernel=NAME           analyze only this entry\n"
        "  --export-cfg=FILE.dot   write the control-flow graph as Graphviz\n"
        "  --heatmap               print per-instruction register pressure\n"
        "  --emit-spilled=FILE     write PTX with spill code (.local) inserted\n"
        "  --no-dce                skip dead code elimination\n"
        "  --no-spill-rewrite      only predict spills, do not rewrite\n"
        "  --ground-truth=MODE     auto (default) | jit | ptxas | none\n"
        "  --cubin=FILE            compare against a pre-built cubin instead of compiling\n"
        "  --ptxas=PATH            ptxas executable to use for --ground-truth=ptxas\n";
}

bool read_file(const std::string& path, std::string& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    return true;
}

bool starts_with(const std::string& s, const char* p) { return s.rfind(p, 0) == 0; }

std::string num(uint64_t v) {
    std::string s = std::to_string(v), r;
    for (size_t i = 0; i < s.size(); ++i) {
        if (i && (s.size() - i) % 3 == 0) r += ',';
        r += s[i];
    }
    return r;
}

std::string hex(uint64_t v) {
    char b[32];
    std::snprintf(b, sizeof(b), "0x%llX", static_cast<unsigned long long>(v));
    return b;
}

std::string dot_path_for(const std::string& base, const std::string& kernel, bool multi) {
    if (!multi) return base;
    size_t dot = base.rfind('.');
    std::string stem = dot == std::string::npos ? base : base.substr(0, dot);
    std::string ext = dot == std::string::npos ? "" : base.substr(dot);
    return stem + "." + kernel + ext;
}

struct Args {
    std::string command, input;
    uint32_t max_regs = 255;
    std::string kernel, export_cfg, emit_spilled, mode = "auto", cubin, ptxas;
    bool heatmap = false, dce = true, rewrite = true;
};

bool parse_args(int argc, char** argv, Args& a) {
    if (argc < 2) return false;
    a.command = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string s = argv[i];
        auto val = [&](const char* prefix) { return s.substr(std::strlen(prefix)); };
        if (starts_with(s, "--maxrregcount=")) {
            long v = std::strtol(val("--maxrregcount=").c_str(), nullptr, 10);
            if (v < 1 || v > 255) { std::cerr << "error: --maxrregcount must be in [1, 255]\n"; return false; }
            a.max_regs = static_cast<uint32_t>(v);
        }
        else if (starts_with(s, "--kernel=")) a.kernel = val("--kernel=");
        else if (starts_with(s, "--export-cfg=")) a.export_cfg = val("--export-cfg=");
        else if (starts_with(s, "--emit-spilled=")) a.emit_spilled = val("--emit-spilled=");
        else if (starts_with(s, "--ground-truth=")) a.mode = val("--ground-truth=");
        else if (starts_with(s, "--cubin=")) a.cubin = val("--cubin=");
        else if (starts_with(s, "--ptxas=")) a.ptxas = val("--ptxas=");
        else if (s == "--heatmap") a.heatmap = true;
        else if (s == "--no-dce") a.dce = false;
        else if (s == "--no-spill-rewrite") a.rewrite = false;
        else if (s == "-h" || s == "--help") return false;
        else if (starts_with(s, "--")) { std::cerr << "error: unknown option " << s << "\n"; return false; }
        else if (a.input.empty()) a.input = s;
        else { std::cerr << "error: unexpected argument " << s << "\n"; return false; }
    }
    return true;
}

void print_cubin_header(const backend::CubinInfo& c) {
    std::printf("  ELF Header                 : ELF%d, ABI=%s (%s), Machine=%s (%s), SM=%u\n",
                c.ei_class == 2 ? 64 : 32, hex(c.os_abi).c_str(),
                (c.os_abi == backend::kElfOsAbiCuda || c.os_abi == backend::kElfOsAbiCudaV2) ? "CUDA" : "?", hex(c.machine).c_str(),
                c.machine == backend::kEmCuda ? "EM_CUDA" : "?", c.sm_arch);
}

void print_kernel_info(const backend::KernelInfo& k) {
    std::printf("  Section .text.<kernel>     : %s bytes (%s SASS instructions)\n", num(k.text_bytes).c_str(),
                num(k.sass_instructions).c_str());
    std::printf("  Section .nv.shared         : %s bytes static shared memory\n", num(k.shared_bytes).c_str());
    std::printf("  Section .nv.constant0      : %s bytes\n", num(k.constant_bytes).c_str());
}

int cmd_inspect(const Args& a) {
    if (a.input.empty()) { usage(); return 2; }
    backend::CubinInfo c;
    try { c = backend::parse_cubin_file(a.input); }
    catch (const std::exception& e) { std::cerr << "error: " << e.what() << "\n"; return 1; }

    std::cout << kRule << "\n PTX-FORGE: ELF Cubin Inspector  (" << a.input << ")\n" << kRule << "\n";
    print_cubin_header(c);
    std::printf("  Sections                   : %zu, Symbols: %zu, Kernels: %zu\n\n", c.sections.size(),
                c.symbols.size(), c.kernels.size());
    std::printf("  %-4s %-40s %10s %6s\n", "idx", "section", "size", "info");
    for (size_t i = 0; i < c.sections.size(); ++i)
        std::printf("  %-4zu %-40s %10s %6s\n", i, c.sections[i].name.c_str(), num(c.sections[i].size).c_str(),
                    hex(c.sections[i].info).c_str());
    for (auto& k : c.kernels) {
        std::cout << "\n" << kThin << "\n Kernel " << k.name << "\n" << kThin << "\n";
        print_kernel_info(k);
        std::printf("  Physical registers         : %u (%s)\n", k.registers(),
                    k.has_reg_count ? "EIATTR_REGCOUNT" : ".text sh_info >> 24");
        std::printf("  EIATTR_FRAME_SIZE          : %u bytes\n", k.frame_size);
        std::printf("  EIATTR_MIN_STACK_SIZE      : %u bytes\n", k.min_stack_size);
        std::cout << "  .nv.info attributes        :";
        for (auto& at : k.attrs) std::cout << " " << at.name().substr(7);
        std::cout << "\n";
    }
    return 0;
}

void print_report(const KernelAnalysis& r, const AnalysisOptions& opts, const GroundTruth* gt, bool gt_attempted,
                  const std::string& gt_note) {
    const auto& al = r.alloc;
    std::cout << kRule << "\n PTX-FORGE: Static Analysis, Register Allocator & ELF Cubin Inspector\n" << kRule << "\n";
    std::cout << "Target Kernel : " << r.name << " (PTX ISA " << r.ptx_version << ", Target: " << r.target << ")\n";
    std::printf("Basic Blocks  : %zu  (Edges: %zu, Natural Loops Detected: %zu)\n", r.blocks, r.edges, r.loops);
    std::printf("Dead Code Elim: Removed %zu dead virtual register definition%s\n\n", r.dce_removed,
                r.dce_removed == 1 ? "" : "s");

    std::cout << "[1] Liveness & Interference Graph Analysis\n" << kThin << "\n";
    std::string types;
    for (auto& [t, n] : r.declared_by_type) types += (types.empty() ? "." : ", .") + t + ": " + std::to_string(n);
    std::printf("  Virtual Registers Declared : %zu (%s)\n", r.declared_regs, types.c_str());
    std::printf("  RIG Nodes / Interference   : %zu nodes, %zu interference edges\n", al.rig_nodes, al.rig_edges);
    std::printf("  Max Live 32-bit Equivalent : %u registers (at BB%u, Line %u%s)\n", r.peak_slots, r.peak_block,
                r.peak_line, r.peak_loop_depth ? ", inside loop" : "");
    std::printf("  Max Live Predicates        : %u (hardware: 7)\n\n", r.max_live_preds);

    std::printf("[2] Chaitin-Briggs Register Allocation Simulation (Target Limit: K = %u)\n%s\n", opts.max_regs, kThin);
    if (!al.converged)
        std::printf("  Allocation Status          : FAILED (could not color within %u registers after %u rounds)\n",
                    opts.max_regs, al.rounds);
    else if (al.fits_without_spill)
        std::printf("  Allocation Status          : OK (peak demand %u <= %u limit)\n", r.peak_slots, opts.max_regs);
    else
        std::printf("  Allocation Status          : SPILLED (%u peak demand > %u limit)\n", r.peak_slots, opts.max_regs);
    std::printf("  Predicted Registers Used   : %u physical registers (+%u predicate)\n", al.regs_used, al.preds_used);
    if (!al.potential_spills.empty()) {
        std::string s;
        size_t shown = 0;
        for (auto& c : al.potential_spills) {
            if (shown++ == 4) { s += ", ..."; break; }
            char b[64];
            std::snprintf(b, sizeof(b), "%s (cost: %.2f)", c.name.c_str(), c.cost);
            s += (s.empty() ? "" : ", ") + std::string(b);
        }
        std::printf("  Potential Spill Candidates : %s\n", s.c_str());
    }
    if (!al.actual_spills.empty()) {
        std::string s;
        size_t shown = 0;
        for (auto& c : al.actual_spills) {
            if (shown++ == 4) { s += ", ..."; break; }
            s += (s.empty() ? "" : ", ") + c.name;
        }
        std::printf("  Spilled by Select          : %zu register%s: %s\n", al.actual_spills.size(),
                    al.actual_spills.size() == 1 ? "" : "s", s.c_str());
        std::printf("  Estimated .local Frame     : %u bytes (%u spill / %u fill instructions inserted)\n",
                    al.frame_bytes, al.spill_stores, al.spill_loads);
    } else {
        std::printf("  Estimated .local Frame     : 0 bytes (no spill code)\n");
    }
    if (al.pred_overflow) std::printf("  Warning                    : predicate pressure exceeds 7 hardware predicates\n");
    std::printf("\n");

    std::cout << "[3] Ground-Truth Verification via CUDA Driver JIT & ELF Cubin Inspection\n" << kThin << "\n";
    if (!gt_attempted) {
        std::cout << "  (skipped: " << gt_note << ")\n";
    } else if (!gt || !gt->ok) {
        std::cout << "  Ground truth unavailable   : " << (gt ? gt->error : gt_note) << "\n";
    } else {
        std::cout << "  Source                     : " << gt->source << "\n";
        print_cubin_header(gt->cubin);
        const auto& k = *gt->kernel;
        print_kernel_info(k);
        long actual_regs = k.registers();
        std::printf("  ptxas Actual Reg Count     : %ld physical registers (via %s)\n", actual_regs,
                    k.has_reg_count ? "EIATTR_REGCOUNT" : ".text sh_info");
        long diff = actual_regs - static_cast<long>(al.regs_used);
        std::printf("  Prediction vs ptxas        : %u predicted, %ld actual (%+ld)\n", al.regs_used, actual_regs, diff);
        std::printf("  ptxas Actual Spill Frame   : %u bytes stack frame (predicted %u)", k.frame_size, al.frame_bytes);
        if (gt->ptxas)
            std::printf(", %u B spill stores, %u B spill loads", gt->ptxas->spill_store_bytes, gt->ptxas->spill_load_bytes);
        std::printf("\n");
    }
    std::cout << kRule << "\n";
}

int cmd_analyze(const Args& a) {
    if (a.input.empty()) { usage(); return 2; }
    std::string source;
    if (!read_file(a.input, source)) { std::cerr << "error: cannot read " << a.input << "\n"; return 1; }

    ir::Module module;
    try {
        module = frontend::parse_ptx(source);
    } catch (const frontend::ParseError& e) {
        std::cerr << a.input << ":" << e.line << ":" << e.col << ": error: " << e.what() << "\n";
        return 1;
    }

    AnalysisOptions opts;
    opts.max_regs = a.max_regs;
    opts.dce = a.dce;
    opts.rewrite_spills = a.rewrite;

    // Ground-truth provider selection.
    std::unique_ptr<backend::CubinProvider> provider;
    std::string gt_note = "disabled (--ground-truth=none)";
    if (!a.cubin.empty()) {
        provider = std::make_unique<backend::FileCubinProvider>(a.cubin);
    } else if (a.mode == "jit" || a.mode == "auto") {
        auto jit = std::make_unique<backend::DriverJITLinker>();
        if (jit->available()) provider = std::move(jit);
        else gt_note = jit->unavailable_reason();
    }
    if (!provider && (a.mode == "ptxas" || a.mode == "auto")) {
        auto p = std::make_unique<backend::PtxasProvider>(a.ptxas);
        if (p->available()) provider = std::move(p);
        else if (a.mode == "ptxas") gt_note = "ptxas not found";
        else gt_note += "; ptxas not found";
    }
    if (a.mode != "auto" && a.mode != "jit" && a.mode != "ptxas" && a.mode != "none") {
        std::cerr << "error: unknown --ground-truth mode '" << a.mode << "'\n";
        return 2;
    }

    int analyzed = 0;
    int total = 0;
    for (auto& f : module.functions) if (f.is_entry && (a.kernel.empty() || f.name == a.kernel)) ++total;
    if (total == 0) { std::cerr << "error: no matching .entry kernel found\n"; return 1; }

    for (auto& fn : module.functions) {
        if (!fn.is_entry || (!a.kernel.empty() && fn.name != a.kernel)) continue;
        KernelAnalysis r;
        try {
            r = analyze_function(fn, module, opts);
        } catch (const std::exception& e) {
            std::cerr << "error: " << fn.name << ": " << e.what() << "\n";
            return 1;
        }
        GroundTruth gt;
        bool attempted = provider != nullptr;
        if (attempted)
            gt = fetch_ground_truth(*provider, source, fn.name, a.max_regs, a.cubin.empty() ? module.target : "");

        if (analyzed++) std::cout << "\n";
        print_report(r, opts, attempted ? &gt : nullptr, attempted, gt_note);

        if (a.heatmap) std::cout << "\nRegister pressure heatmap (32-bit registers live; '!' = exceeds K=" << opts.max_regs
                                 << ")\n  line  live\n" << r.heatmap;
        if (!a.export_cfg.empty()) {
            std::string path = dot_path_for(a.export_cfg, r.name, total > 1);
            std::ofstream(path) << r.cfg_dot;
            std::cout << "\nCFG written to " << path << "\n";
        }
        if (!a.emit_spilled.empty()) {
            if (r.spilled_ptx.empty()) {
                std::cout << "\nNo spill code generated for " << r.name << "; --emit-spilled skipped\n";
            } else {
                std::string path = dot_path_for(a.emit_spilled, r.name, total > 1);
                ir::Module out;
                out.version = module.version;
                out.target = module.target;
                out.target_options = module.target_options;
                out.address_size = module.address_size;
                out.globals = module.globals;
                std::ofstream(path) << ir::print_module(out) << "\n" << r.spilled_ptx;
                std::cout << "\nPTX with spill code written to " << path << "\n";
            }
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    Args a;
    if (!parse_args(argc, argv, a)) { usage(); return 2; }
    if (a.command == "analyze") return cmd_analyze(a);
    if (a.command == "inspect") return cmd_inspect(a);
    usage();
    return 2;
}
