#include <cstring>
#include <fstream>
#include <iterator>

#include "Pipeline.hpp"
#include "Test.hpp"
#include "TestUtil.hpp"
#include "backend/CubinProvider.hpp"
#include "backend/DriverJITLinker.hpp"
#include "backend/ElfCubinParser.hpp"

using namespace ptxforge::backend;
using ptxforge::frontend::parse_ptx;

namespace {
std::vector<uint8_t> load_bytes(const std::string& rel) {
    std::ifstream f(std::string(PTXFORGE_TEST_DIR) + "/" + rel, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

template <class T>
void put(std::vector<uint8_t>& b, size_t off, T v) { std::memcpy(b.data() + off, &v, sizeof(T)); }

// Builds a tiny but structurally valid cubin: .shstrtab, .symtab/.strtab, one .text.k section with the
// register count in sh_info >> 24, .nv.shared.k, and a global .nv.info with REGCOUNT + FRAME_SIZE for symbol 1.
std::vector<uint8_t> synthetic_cubin(uint32_t regs, uint32_t frame, uint32_t shared) {
    std::string shstr = std::string("\0", 1) + ".shstrtab" + std::string("\0", 1) + ".strtab" + std::string("\0", 1) +
                        ".symtab" + std::string("\0", 1) + ".nv.info" + std::string("\0", 1) + ".text.k" +
                        std::string("\0", 1) + ".nv.shared.k" + std::string("\0", 1);
    auto off_of = [&](const char* n) { return static_cast<uint32_t>(shstr.find(n)); };
    std::string strtab = std::string("\0", 1) + "k" + std::string("\0", 1);
    std::vector<uint8_t> sym(48, 0);                // null symbol + symbol "k"
    put<uint32_t>(sym, 24, 1);                      // st_name
    sym[28] = 0x12;                                 // GLOBAL FUNC
    put<uint16_t>(sym, 30, 5);                      // st_shndx = .text.k
    std::vector<uint8_t> info;                      // two SVAL records: (sym 1, value)
    for (auto [attr, val] : {std::pair<uint8_t, uint32_t>{0x2f, regs}, {0x11, frame}}) {
        info.push_back(4); info.push_back(attr); info.push_back(8); info.push_back(0);
        for (uint32_t w : {1u, val}) for (int i = 0; i < 4; ++i) info.push_back(static_cast<uint8_t>(w >> (8 * i)));
    }
    std::vector<uint8_t> text(64, 0);

    std::vector<uint8_t> file(64, 0);
    auto append = [&](const void* p, size_t n) {
        size_t o = file.size();
        file.insert(file.end(), static_cast<const uint8_t*>(p), static_cast<const uint8_t*>(p) + n);
        return o;
    };
    size_t o_shstr = append(shstr.data(), shstr.size());
    size_t o_str = append(strtab.data(), strtab.size());
    while (file.size() % 8) file.push_back(0);
    size_t o_sym = append(sym.data(), sym.size());
    size_t o_info = append(info.data(), info.size());
    size_t o_text = append(text.data(), text.size());
    while (file.size() % 8) file.push_back(0);
    size_t shoff = file.size();

    struct Sh { uint32_t name, type; uint64_t off, size; uint32_t link, info; uint64_t entsize; };
    Sh sh[] = {
        {0, 0, 0, 0, 0, 0, 0},
        {off_of(".shstrtab"), 3, o_shstr, shstr.size(), 0, 0, 0},
        {off_of(".strtab"), 3, o_str, strtab.size(), 0, 0, 0},
        {off_of(".symtab"), 2, o_sym, sym.size(), 2, 0, 24},
        {off_of(".nv.info"), 0x70000000, o_info, info.size(), 3, 0, 0},
        {off_of(".text.k"), 1, o_text, text.size(), 3, regs << 24, 0},
        {off_of(".nv.shared.k"), 8, 0, shared, 0, 0, 0},
    };
    file.resize(shoff + sizeof(sh) / sizeof(sh[0]) * 64, 0);
    for (size_t i = 0; i < sizeof(sh) / sizeof(sh[0]); ++i) {
        size_t o = shoff + i * 64;
        put<uint32_t>(file, o, sh[i].name);
        put<uint32_t>(file, o + 4, sh[i].type);
        put<uint64_t>(file, o + 24, sh[i].off);
        put<uint64_t>(file, o + 32, sh[i].size);
        put<uint32_t>(file, o + 40, sh[i].link);
        put<uint32_t>(file, o + 44, sh[i].info);
        put<uint64_t>(file, o + 56, sh[i].entsize);
    }
    std::memcpy(file.data(), "\x7f" "ELF", 4);
    file[4] = 2; file[5] = 1; file[6] = 1; file[7] = kElfOsAbiCudaV2; file[8] = 8;
    put<uint16_t>(file, 16, 2);
    put<uint16_t>(file, 18, kEmCuda);
    put<uint32_t>(file, 20, 1);
    put<uint64_t>(file, 40, shoff);
    put<uint32_t>(file, 48, 0x5005604);   // sm_86 in bits 8..15
    put<uint16_t>(file, 52, 64);
    put<uint16_t>(file, 58, 64);
    put<uint16_t>(file, 60, static_cast<uint16_t>(sizeof(sh) / sizeof(sh[0])));
    put<uint16_t>(file, 62, 1);
    return file;
}
}  // namespace

// Ground truth for the checked-in cubins was cross-checked with `cuobjdump --dump-resource-usage`.
TEST(ElfCubin, ParsesRealPtxasOutput) {
    CubinInfo c = parse_cubin(load_bytes("data/vectorAdd_sm80.cubin"));
    EXPECT_EQ(c.machine, kEmCuda);
    EXPECT_EQ(c.ei_class, 2);
    EXPECT_EQ(c.os_abi, kElfOsAbiCudaV2);
    EXPECT_EQ(c.sm_arch, 80u);
    ASSERT_EQ(c.kernels.size(), 1u);
    const KernelInfo& k = c.kernels[0];
    EXPECT_EQ(k.name, "_Z9vectorAddPKfS0_Pfi");
    EXPECT_EQ(k.registers(), 12u);
    EXPECT_EQ(k.sass_instructions, 32u);
    EXPECT_EQ(k.text_bytes, 512u);
    EXPECT_EQ(k.frame_size, 0u);
    EXPECT_EQ(k.shared_bytes, 0u);
    EXPECT_EQ(k.constant_bytes, 380u);
}

TEST(ElfCubin, ReadsSpillFrameFromGlobalNvInfo) {
    CubinInfo c = parse_cubin(load_bytes("data/pressure_loop_k32_sm80.cubin"));
    ASSERT_EQ(c.kernels.size(), 1u);
    EXPECT_EQ(c.kernels[0].registers(), 32u);
    EXPECT_TRUE(c.kernels[0].has_reg_count);
    EXPECT_EQ(c.kernels[0].frame_size, 160u);
    EXPECT_EQ(c.kernels[0].min_stack_size, 160u);
    EXPECT_EQ(c.kernels[0].sass_instructions, 328u);
    EXPECT_EQ(c.kernels[0].reg_count_sh_info, 32u);  // legacy location agrees with the attribute
}

TEST(ElfCubin, SectionAndSymbolTables) {
    CubinInfo c = parse_cubin(load_bytes("data/vectorAdd_sm80.cubin"));
    EXPECT_TRUE(c.find_section(".shstrtab") != nullptr);
    EXPECT_TRUE(c.find_section(".nv.info") != nullptr);
    EXPECT_TRUE(c.find_section(".text._Z9vectorAddPKfS0_Pfi") != nullptr);
    bool found = false;
    for (auto& s : c.symbols) found |= (s.name == "_Z9vectorAddPKfS0_Pfi" && s.size == 512);
    EXPECT_TRUE(found);
}

TEST(ElfCubin, SyntheticImage) {
    CubinInfo c = parse_cubin(synthetic_cubin(40, 24, 8192));
    ASSERT_EQ(c.kernels.size(), 1u);
    const KernelInfo& k = c.kernels[0];
    EXPECT_EQ(k.name, "k");
    EXPECT_EQ(k.registers(), 40u);
    EXPECT_EQ(k.reg_count_sh_info, 40u);
    EXPECT_EQ(k.frame_size, 24u);
    EXPECT_EQ(k.shared_bytes, 8192u);
    EXPECT_EQ(k.sass_instructions, 4u);
}

TEST(ElfCubin, RejectsMalformedInput) {
    EXPECT_THROW(parse_cubin({}), ElfError);
    EXPECT_THROW(parse_cubin(std::vector<uint8_t>(128, 0)), ElfError);

    auto img = synthetic_cubin(8, 0, 0);
    auto wrong_machine = img;
    wrong_machine[18] = 0x3e;  // EM_X86_64
    EXPECT_THROW(parse_cubin(wrong_machine), ElfError);
    auto truncated = img;
    truncated.resize(img.size() - 100);
    EXPECT_THROW(parse_cubin(truncated), ElfError);
    auto elf32 = img;
    elf32[4] = 1;
    EXPECT_THROW(parse_cubin(elf32), ElfError);
}

TEST(NvInfo, AttributeNames) {
    NvInfoAttr a;
    a.attr = 0x2f;
    EXPECT_EQ(a.name(), "EIATTR_REGCOUNT");
    a.attr = 0x77;
    EXPECT_EQ(a.name(), "EIATTR_0x77");
}

TEST(PtxasLog, ParsesRegistersAndSpills) {
    const char* log =
        "ptxas info    : Compiling entry function 'foo' for 'sm_80'\n"
        "ptxas info    : Function properties for foo\n"
        "    160 bytes stack frame, 300 bytes spill stores, 304 bytes spill loads\n"
        "ptxas info    : Used 32 registers, used 0 barriers, 372 bytes cmem[0]\n";
    auto stats = parse_ptxas_log(log);
    ASSERT_EQ(stats.count("foo"), 1u);
    EXPECT_EQ(stats["foo"].registers, 32u);
    EXPECT_EQ(stats["foo"].stack_bytes, 160u);
    EXPECT_EQ(stats["foo"].spill_store_bytes, 300u);
    EXPECT_EQ(stats["foo"].spill_load_bytes, 304u);
}

TEST(Providers, FileProviderServesCubinAndReportsMissingFile) {
    FileCubinProvider ok(std::string(PTXFORGE_TEST_DIR) + "/data/vectorAdd_sm80.cubin");
    auto r = ok.compile("", {});
    EXPECT_TRUE(r.ok);
    EXPECT_GT(r.cubin.size(), 1000u);
    FileCubinProvider missing("does/not/exist.cubin");
    EXPECT_FALSE(missing.compile("", {}).ok);
}

// These need real hardware / toolkit; they self-skip (pass) when it is not present.
TEST(Providers, DriverJitCompilesVectorAdd) {
    DriverJITLinker jit;
    if (!jit.available()) return;  // no driver / GPU: nothing to verify
    std::string ptx = testutil::read_file("kernels/vectorAdd.ptx");
    CompileOptions opts;
    auto r = jit.compile(ptx, opts);
    ASSERT_TRUE(r.ok);
    CubinInfo c = parse_cubin(r.cubin);
    ASSERT_EQ(c.kernels.size(), 1u);
    EXPECT_GT(c.kernels[0].registers(), 0u);
    EXPECT_EQ(c.kernels[0].name, "_Z9vectorAddPKfS0_Pfi");
}

TEST(Providers, DriverJitReportsPtxErrors) {
    DriverJITLinker jit;
    if (!jit.available()) return;
    auto r = jit.compile(".version 8.0\n.target sm_80\n.address_size 64\n.entry bad() { bogus; }\n", {});
    EXPECT_FALSE(r.ok);
    EXPECT_FALSE(r.error_log.empty());
}

TEST(Providers, PtxasHonoursMaxRegisters) {
    PtxasProvider p;
    if (!p.available()) return;
    CompileOptions opts;
    opts.target_sm = "sm_80";
    opts.max_regs = 32;
    auto r = p.compile(testutil::read_file("kernels/pressure_loop.ptx"), opts);
    ASSERT_TRUE(r.ok);
    CubinInfo c = parse_cubin(r.cubin);
    EXPECT_EQ(c.kernels[0].registers(), 32u);
    EXPECT_GT(c.kernels[0].frame_size, 0u);
}

// ---- real execution; skipped when there is no GPU -------------------------------------------

TEST(GpuRun, VectorAddComputesTheSum) {
    DriverJITLinker gpu;
    if (!gpu.available()) return;
    std::vector<KernelArg> args(4);
    for (uint32_t i = 0; i < 3; ++i) { args[i].kind = KernelArg::Kind::Buffer; args[i].bytes = 4096; args[i].seed = i; }
    args[3].kind = KernelArg::Kind::U32;
    args[3].value = 1000;
    LaunchConfig cfg;
    cfg.grid[0] = 4;
    cfg.block[0] = 256;
    auto r = gpu.run(testutil::read_file("kernels/vectorAdd.ptx"), "_Z9vectorAddPKfS0_Pfi", args, cfg);
    ASSERT_TRUE(r.ok);
    ASSERT_EQ(r.buffers.size(), 3u);
    auto a = fill_pattern(0, 4096), b = fill_pattern(1, 4096), untouched = fill_pattern(2, 4096);
    for (size_t i = 0; i < 1024; ++i) {
        float fa, fb, fc;
        std::memcpy(&fa, a.data() + i * 4, 4);
        std::memcpy(&fb, b.data() + i * 4, 4);
        std::memcpy(&fc, r.buffers[2].data() + i * 4, 4);
        if (i < 1000) {
            EXPECT_EQ(fc, fa + fb);
        } else {
            EXPECT_TRUE(std::memcmp(r.buffers[2].data() + i * 4, untouched.data() + i * 4, 4) == 0);  // bounds check held
        }
    }
    EXPECT_GT(r.registers, 0);
}

TEST(GpuRun, SpilledKernelProducesIdenticalOutput) {
    DriverJITLinker gpu;
    if (!gpu.available()) return;
    auto module = parse_ptx(testutil::read_file("kernels/pressure_loop.ptx"));
    ptxforge::AnalysisOptions opts;
    opts.max_regs = 32;
    auto analysis = ptxforge::analyze_function(module.functions[0], module, opts);
    std::string spilled = ptxforge::emit_spilled_module(module, analysis);
    ASSERT_TRUE(!spilled.empty());

    std::vector<KernelArg> args(3);
    for (uint32_t i = 0; i < 2; ++i) { args[i].kind = KernelArg::Kind::Buffer; args[i].bytes = 4096; args[i].seed = i; }
    args[2].kind = KernelArg::Kind::U32;
    args[2].value = 3;
    LaunchConfig cfg;
    cfg.block[0] = 64;
    auto original = gpu.run(testutil::read_file("kernels/pressure_loop.ptx"), "pressure_loop", args, cfg);
    auto rewritten = gpu.run(spilled, "pressure_loop", args, cfg, 32);
    ASSERT_TRUE(original.ok);
    ASSERT_TRUE(rewritten.ok);
    EXPECT_EQ(rewritten.registers, 32);
    EXPECT_GT(rewritten.local_bytes, 0);
    EXPECT_TRUE(original.buffers == rewritten.buffers);
}

TEST(GpuRun, ReportsMissingKernelAndBadPtx) {
    DriverJITLinker gpu;
    if (!gpu.available()) return;
    LaunchConfig cfg;
    EXPECT_FALSE(gpu.run(testutil::read_file("kernels/vectorAdd.ptx"), "nope", {}, cfg).ok);
    EXPECT_FALSE(gpu.run(".version 8.0\n.target sm_80\n.address_size 64\n.entry bad() { bogus; }\n", "bad", {}, cfg).ok);
}
