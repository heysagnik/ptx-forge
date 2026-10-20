// ptx-forge: PTX static analyzer, register allocator and cubin inspector.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "Pipeline.hpp"
#include "backend/CudaDriver.hpp"
#include "backend/DriverJITLinker.hpp"
#include "backend/ElfCubinParser.hpp"
#include "backend/GpuRunner.hpp"
#include "backend/PtxasProvider.hpp"
#include "frontend/Parser.hpp"
#include "ir/PTXPrinter.hpp"

using namespace ptxforge;

namespace {

struct Args {
    std::string command, input;
    uint32_t max_regs = 255;
    std::string kernel, export_cfg, emit_spilled, mode = "auto", cubin, ptxas;
    bool heatmap = false;
    std::string grid = "1", block = "1";
    std::vector<std::string> kernel_args;
};

constexpr const char* kRule = "================================================================================";
constexpr const char* kThin = "--------------------------------------------------------------------------------";

void usage() {
    std::cout <<
        "ptx-forge - PTX static analyzer, register allocator and cubin inspector\n"
        "\n"
        "usage:\n"
        "  ptx-forge analyze <kernel.ptx> [options]\n"
        "  ptx-forge verify <kernel.ptx> --grid=X[,Y,Z] --block=X[,Y,Z] --arg=... [options]\n"
        "  ptx-forge inspect <file.cubin>\n"
        "\n"
        "analyze options:\n"
        "  --maxrregcount=K        physical register budget for the allocator (default 255)\n"
        "  --kernel=NAME           analyze only this entry\n"
        "  --export-cfg=FILE.dot   write the control-flow graph as Graphviz\n"
        "  --heatmap               print per-instruction register pressure\n"
        "  --emit-spilled=FILE     write PTX with spill code (.local) inserted\n"
        "  --ground-truth=MODE     auto (default) | jit | ptxas | none\n"
        "  --cubin=FILE            compare against a pre-built cubin instead of compiling\n"
        "  --ptxas=PATH            ptxas executable to use for --ground-truth=ptxas\n"
        "\n"
        "verify runs the kernel on the GPU: original PTX, the same PTX capped by the driver at K registers,\n"
        "and ptx-forge's spill-rewritten PTX, then compares every output buffer bit for bit.\n"
        "  --grid / --block        launch shape (default 1)\n"
        "  --arg=buf:BYTES|u32:N|u64:N|f32:X   one per kernel parameter, in order (repeatable)\n";
}

bool starts_with(const std::string& s, const char* p) { return s.rfind(p, 0) == 0; }

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
        else if (starts_with(s, "--grid=")) a.grid = val("--grid=");
        else if (starts_with(s, "--block=")) a.block = val("--block=");
        else if (starts_with(s, "--arg=")) a.kernel_args.push_back(val("--arg="));
        else if (s == "--heatmap") a.heatmap = true;
        else if (s == "-h" || s == "--help") return false;
        else if (starts_with(s, "--")) { std::cerr << "error: unknown option " << s << "\n"; return false; }
        else if (a.input.empty()) a.input = s;
        else { std::cerr << "error: unexpected argument " << s << "\n"; return false; }
    }
    return true;
}

std::optional<ir::Module> load_module(const std::string& path, std::string& source) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "error: cannot read " << path << "\n"; return std::nullopt; }
    source.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    try {
        return frontend::parse_ptx(source);
    } catch (const frontend::ParseError& e) {
        std::cerr << path << ":" << e.line << ":" << e.col << ": error: " << e.what() << "\n";
        return std::nullopt;
    }
}

const ir::Function* find_kernel(const ir::Module& m, const std::string& name) {
    for (auto& f : m.functions)
        if (f.is_entry && (name.empty() || f.name == name)) return &f;
    return nullptr;
}

std::string with_commas(uint64_t v) {
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

void print_cubin_header(const backend::CubinInfo& c) {
    const bool cuda_abi = c.os_abi == backend::kElfOsAbiCuda || c.os_abi == backend::kElfOsAbiCudaV2;
    std::printf("  ELF Header                 : ELF%d, ABI=%s (%s), Machine=%s (%s), SM=%u\n",
                c.ei_class == 2 ? 64 : 32, hex(c.os_abi).c_str(), cuda_abi ? "CUDA" : "?", hex(c.machine).c_str(),
                c.machine == backend::kEmCuda ? "EM_CUDA" : "?", c.sm_arch);
}

void print_kernel_sections(const backend::KernelInfo& k) {
    std::printf("  Section .text.<kernel>     : %s bytes (%s SASS instructions)\n", with_commas(k.text_bytes).c_str(),
                with_commas(k.sass_instructions).c_str());
    std::printf("  Section .nv.shared         : %s bytes static shared memory\n", with_commas(k.shared_bytes).c_str());
    std::printf("  Section .nv.constant0      : %s bytes\n", with_commas(k.constant_bytes).c_str());
}

int cmd_inspect(const Args& a) {
    if (a.input.empty()) { usage(); return 2; }
    backend::CubinInfo c;
    try {
        c = backend::parse_cubin_file(a.input);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    std::cout << kRule << "\n PTX-FORGE: ELF Cubin Inspector  (" << a.input << ")\n" << kRule << "\n";
    print_cubin_header(c);
    std::printf("  Sections                   : %zu, Symbols: %zu, Kernels: %zu\n\n", c.sections.size(),
                c.symbols.size(), c.kernels.size());
    std::printf("  %-4s %-40s %10s %6s\n", "idx", "section", "size", "info");
    for (size_t i = 0; i < c.sections.size(); ++i)
        std::printf("  %-4zu %-40s %10s %6s\n", i, c.sections[i].name.c_str(),
                    with_commas(c.sections[i].size).c_str(), hex(c.sections[i].info).c_str());

    for (auto& k : c.kernels) {
        std::cout << "\n" << kThin << "\n Kernel " << k.name << "\n" << kThin << "\n";
        print_kernel_sections(k);
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

std::string output_path(const std::string& base, const std::string& kernel, bool multi) {
    if (!multi) return base;
    size_t dot = base.rfind('.');
    std::string stem = dot == std::string::npos ? base : base.substr(0, dot);
    std::string ext = dot == std::string::npos ? "" : base.substr(dot);
    return stem + "." + kernel + ext;
}

std::string list_first(const std::vector<passes::SpillCandidate>& v, bool with_cost) {
    std::string s;
    size_t shown = 0;
    for (auto& c : v) {
        if (shown++ == 4) { s += ", ..."; break; }
        char b[64];
        if (with_cost) std::snprintf(b, sizeof(b), "%s (cost: %.2f)", c.name.c_str(), c.cost);
        else std::snprintf(b, sizeof(b), "%s", c.name.c_str());
        s += (s.empty() ? "" : ", ") + std::string(b);
    }
    return s;
}

void print_report(const KernelAnalysis& r, const AnalysisOptions& opts, const GroundTruth* gt,
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
    if (!al.potential_spills.empty())
        std::printf("  Potential Spill Candidates : %s\n", list_first(al.potential_spills, true).c_str());
    if (!al.actual_spills.empty()) {
        std::printf("  Spilled by Select          : %zu register%s: %s\n", al.actual_spills.size(),
                    al.actual_spills.size() == 1 ? "" : "s", list_first(al.actual_spills, false).c_str());
        std::printf("  Estimated .local Frame     : %u bytes (%u spill / %u fill instructions inserted)\n",
                    al.frame_bytes, al.spill_stores, al.spill_loads);
    } else {
        std::printf("  Estimated .local Frame     : 0 bytes (no spill code)\n");
    }
    if (al.pred_overflow) std::printf("  Warning                    : predicate pressure exceeds 7 hardware predicates\n");
    std::printf("\n");

    std::cout << "[3] Ground-Truth Verification via CUDA Driver JIT & ELF Cubin Inspection\n" << kThin << "\n";
    if (!gt) {
        std::cout << "  (skipped: " << gt_note << ")\n";
    } else if (!gt->ok) {
        std::cout << "  Ground truth unavailable   : " << gt->error << "\n";
    } else {
        std::cout << "  Source                     : " << gt->source << "\n";
        print_cubin_header(gt->cubin);
        const auto& k = *gt->kernel;
        print_kernel_sections(k);
        long actual = k.registers();
        std::printf("  ptxas Actual Reg Count     : %ld physical registers (via %s)\n", actual,
                    k.has_reg_count ? "EIATTR_REGCOUNT" : ".text sh_info");
        std::printf("  Prediction vs ptxas        : %u predicted, %ld actual (%+ld)\n", al.regs_used, actual,
                    actual - static_cast<long>(al.regs_used));
        std::printf("  ptxas Actual Spill Frame   : %u bytes stack frame (predicted %u)", k.frame_size, al.frame_bytes);
        if (gt->ptxas)
            std::printf(", %u B spill stores, %u B spill loads", gt->ptxas->spill_store_bytes,
                        gt->ptxas->spill_load_bytes);
        std::printf("\n");
    }
    std::cout << kRule << "\n";
}

int cmd_analyze(const Args& a) {
    if (a.input.empty()) { usage(); return 2; }
    if (a.mode != "auto" && a.mode != "jit" && a.mode != "ptxas" && a.mode != "none") {
        std::cerr << "error: unknown --ground-truth mode '" << a.mode << "'\n";
        return 2;
    }
    std::string source;
    auto module = load_module(a.input, source);
    if (!module) return 1;

    AnalysisOptions opts;
    opts.max_regs = a.max_regs;

    backend::CudaDriver driver;
    std::unique_ptr<backend::CubinProvider> provider;
    std::string gt_note = "disabled (--ground-truth=none)";
    if (!a.cubin.empty()) {
        provider = std::make_unique<backend::FileCubinProvider>(a.cubin);
    } else if (a.mode == "jit" || a.mode == "auto") {
        if (driver.available()) provider = std::make_unique<backend::DriverJITLinker>(driver);
        else gt_note = driver.unavailable_reason();
    }
    if (!provider && (a.mode == "ptxas" || a.mode == "auto")) {
        auto p = std::make_unique<backend::PtxasProvider>(a.ptxas);
        if (p->available()) provider = std::move(p);
        else gt_note += a.mode == "ptxas" ? "ptxas not found" : "; ptxas not found";
    }

    size_t total = 0;
    for (auto& f : module->functions) total += f.is_entry && (a.kernel.empty() || f.name == a.kernel);
    if (total == 0) { std::cerr << "error: no matching .entry kernel found\n"; return 1; }

    bool first = true;
    for (auto& fn : module->functions) {
        if (!fn.is_entry || (!a.kernel.empty() && fn.name != a.kernel)) continue;
        KernelAnalysis r;
        try {
            r = analyze_function(fn, *module, opts);
        } catch (const std::exception& e) {
            std::cerr << "error: " << fn.name << ": " << e.what() << "\n";
            return 1;
        }
        GroundTruth gt;
        if (provider)
            gt = fetch_ground_truth(*provider, source, fn.name, a.max_regs, a.cubin.empty() ? module->target : "");

        if (!first) std::cout << "\n";
        first = false;
        print_report(r, opts, provider ? &gt : nullptr, gt_note);

        if (a.heatmap)
            std::cout << "\nRegister pressure heatmap (32-bit registers live; '!' = exceeds K=" << opts.max_regs
                      << ")\n  line  live\n" << r.heatmap;
        if (!a.export_cfg.empty()) {
            std::string path = output_path(a.export_cfg, r.name, total > 1);
            std::ofstream(path) << r.cfg_dot;
            std::cout << "\nCFG written to " << path << "\n";
        }
        if (!a.emit_spilled.empty()) {
            if (r.spilled_ptx.empty()) {
                std::cout << "\nNo spill code generated for " << r.name << "; --emit-spilled skipped\n";
            } else {
                std::string path = output_path(a.emit_spilled, r.name, total > 1);
                std::ofstream(path) << emit_spilled_module(*module, r);
                std::cout << "\nPTX with spill code written to " << path << "\n";
            }
        }
    }
    return 0;
}

bool parse_dims(const std::string& s, uint32_t out[3]) {
    out[0] = out[1] = out[2] = 1;
    std::stringstream ss(s);
    std::string part;
    for (int i = 0; i < 3 && std::getline(ss, part, ','); ++i) {
        long v = std::strtol(part.c_str(), nullptr, 10);
        if (v < 1) return false;
        out[i] = static_cast<uint32_t>(v);
    }
    return true;
}

bool parse_kernel_arg(const std::string& s, uint32_t index, backend::KernelArg& out) {
    size_t colon = s.find(':');
    if (colon == std::string::npos) return false;
    std::string kind = s.substr(0, colon), val = s.substr(colon + 1);
    out = {};
    out.seed = index;
    if (kind == "buf") {
        out.kind = backend::KernelArg::Kind::Buffer;
        out.bytes = static_cast<size_t>(std::strtoull(val.c_str(), nullptr, 10));
        return out.bytes > 0;
    }
    if (kind == "u32" || kind == "u64") {
        out.kind = kind == "u32" ? backend::KernelArg::Kind::U32 : backend::KernelArg::Kind::U64;
        out.value = std::strtoull(val.c_str(), nullptr, 10);
        return true;
    }
    if (kind == "f32") {
        out.kind = backend::KernelArg::Kind::F32;
        float f = std::strtof(val.c_str(), nullptr);
        uint32_t bits;
        std::memcpy(&bits, &f, 4);
        out.value = bits;
        return true;
    }
    return false;
}

struct Variant {
    std::string name;
    std::string ptx;
    uint32_t cap;
    backend::RunResult run;
};

std::string compare_to(const backend::RunResult& base, const backend::RunResult& other, bool& ok) {
    for (size_t b = 0; b < base.buffers.size(); ++b)
        for (size_t k = 0; k < base.buffers[b].size(); ++k)
            if (base.buffers[b][k] != other.buffers[b][k]) {
                ok = false;
                return "MISMATCH (buffer " + std::to_string(b) + ", byte " + std::to_string(k) + ")";
            }
    return "identical";
}

// The driver's own message (usually the ptxas line) is on the second line of the error.
std::string failure_detail(const std::string& error) {
    size_t nl = error.find('\n');
    std::string detail = nl == std::string::npos ? error : error.substr(nl + 1);
    detail = detail.substr(0, detail.find('\n'));
    return detail.empty() ? error.substr(0, nl) : detail;
}

int cmd_verify(const Args& a) {
    if (a.input.empty()) { usage(); return 2; }
    std::string source;
    auto module = load_module(a.input, source);
    if (!module) return 1;
    const ir::Function* fn = find_kernel(*module, a.kernel);
    if (!fn) { std::cerr << "error: no matching .entry kernel found\n"; return 1; }
    if (a.kernel_args.size() != fn->params.size()) {
        std::cerr << "error: " << fn->name << " takes " << fn->params.size() << " parameters, got "
                  << a.kernel_args.size() << " --arg values\n";
        return 2;
    }
    std::vector<backend::KernelArg> args(a.kernel_args.size());
    for (size_t i = 0; i < args.size(); ++i)
        if (!parse_kernel_arg(a.kernel_args[i], static_cast<uint32_t>(i), args[i])) {
            std::cerr << "error: bad --arg '" << a.kernel_args[i] << "' (use buf:BYTES, u32:N, u64:N or f32:X)\n";
            return 2;
        }
    backend::LaunchConfig cfg;
    if (!parse_dims(a.grid, cfg.grid) || !parse_dims(a.block, cfg.block)) {
        std::cerr << "error: --grid / --block expect X[,Y[,Z]] with positive values\n";
        return 2;
    }

    AnalysisOptions opts;
    opts.max_regs = a.max_regs;
    KernelAnalysis r = analyze_function(*fn, *module, opts);

    backend::CudaDriver driver;
    if (!driver.available()) { std::cerr << "error: " << driver.unavailable_reason() << "\n"; return 1; }
    backend::GpuRunner gpu(driver);

    std::vector<Variant> variants;
    variants.push_back({"original, driver default", source, 0, {}});
    if (a.max_regs < 255)
        variants.push_back({"original, driver cap " + std::to_string(a.max_regs), source, a.max_regs, {}});
    if (!r.spilled_ptx.empty()) {
        std::string spilled = emit_spilled_module(*module, r);
        variants.push_back({"ptx-forge spill code, no cap", spilled, 0, {}});
        variants.push_back({"ptx-forge spill code, cap " + std::to_string(a.max_regs), spilled, a.max_regs, {}});
    }
    for (auto& v : variants) v.run = gpu.run(v.ptx, fn->name, args, cfg, v.cap);

    std::cout << kRule << "\n PTX-FORGE: GPU verification on " << driver.device_name() << " ("
              << driver.device_arch() << ")\n" << kRule << "\n";
    std::printf("Kernel        : %s   grid=%s block=%s\n", fn->name.c_str(), a.grid.c_str(), a.block.c_str());
    std::printf("Prediction    : peak %u live registers, K=%u -> %s (%u B .local predicted)\n\n", r.peak_slots,
                a.max_regs, r.alloc.fits_without_spill ? "fits" : "spills", r.alloc.frame_bytes);
    if (r.spilled_ptx.empty() && a.max_regs < 255)
        std::cout << "(allocator predicts no spill at K=" << a.max_regs << ", so there is no rewritten PTX to run)\n\n";

    const backend::RunResult& base = variants[0].run;
    bool all_ok = base.ok;
    std::printf("  %-34s %5s %9s  %s\n", "variant", "regs", "local B", "output vs baseline");
    for (size_t i = 0; i < variants.size(); ++i) {
        const Variant& v = variants[i];
        std::string verdict;
        if (!v.run.ok) {
            verdict = "FAILED: " + failure_detail(v.run.error);
            all_ok = false;
        } else if (i == 0) {
            verdict = "reference";
        } else if (!base.ok) {
            verdict = "no baseline";
        } else {
            verdict = compare_to(base, v.run, all_ok);
        }
        std::printf("  %-34s %5d %9d  %s\n", v.name.c_str(), v.run.registers, v.run.local_bytes, verdict.c_str());
    }
    std::cout << kRule << "\n";
    return all_ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    Args a;
    if (!parse_args(argc, argv, a)) { usage(); return 2; }
    if (a.command == "analyze") return cmd_analyze(a);
    if (a.command == "verify") return cmd_verify(a);
    if (a.command == "inspect") return cmd_inspect(a);
    usage();
    return 2;
}
