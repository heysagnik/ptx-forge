#include <unordered_set>

#include "Pipeline.hpp"
#include "Test.hpp"
#include "TestUtil.hpp"
#include "passes/DeadCodeElim.hpp"
#include "passes/LivenessAnalysis.hpp"
#include "passes/RegAllocChaitin.hpp"

using namespace ptxforge;
using namespace ptxforge::ir;
using namespace ptxforge::passes;
using ptxforge::frontend::parse_ptx;

namespace {
struct Ctx {
    Module m;
    ControlFlowGraph cfg;
    Function& fn() { return m.functions[0]; }
};
Ctx make(const std::string& body) {
    Ctx c;
    c.m = testutil::parse_kernel(body);
    c.cfg = ControlFlowGraph::build(c.m.functions[0]);
    return c;
}
size_t reg_id(const Function& f, const std::string& name) {
    for (auto& r : f.regs) if (r.name == name) return r.id;
    return ~size_t(0);
}
}  // namespace

TEST(CFG, VectorAddShape) {
    Module m = parse_ptx(testutil::read_file("kernels/vectorAdd.ptx"));
    auto cfg = ControlFlowGraph::build(m.functions[0]);
    EXPECT_EQ(cfg.blocks().size(), 3u);
    EXPECT_EQ(cfg.edge_count(), 3u);  // 0->1 (fall-through), 0->2 (taken), 1->2
    EXPECT_EQ(cfg.loops().size(), 0u);
    EXPECT_EQ(cfg.blocks()[2].labels.front(), "$L__BB0_2");
}

TEST(CFG, GemmHasTwoNestedLoops) {
    Module m = parse_ptx(testutil::read_file("kernels/gemm_tiled.ptx"));
    auto cfg = ControlFlowGraph::build(m.functions[0]);
    EXPECT_EQ(cfg.blocks().size(), 7u);
    EXPECT_EQ(cfg.loops().size(), 2u);
    uint32_t max_depth = 0;
    for (auto& b : cfg.blocks()) max_depth = std::max(max_depth, b.loop_depth);
    EXPECT_EQ(max_depth, 2u);
    EXPECT_EQ(cfg.back_edges().size(), 2u);
}

TEST(CFG, UnconditionalBranchHasNoFallThrough) {
    auto c = make(".reg .b32 %r<2>;\nbra L2;\nadd.s32 %r1, %r1, 1;\nL2:\nret;");
    ASSERT_EQ(c.cfg.blocks().size(), 3u);
    EXPECT_EQ(c.cfg.blocks()[0].succs.size(), 1u);
    EXPECT_EQ(c.cfg.blocks()[0].succs[0], 2u);
    EXPECT_EQ(c.cfg.blocks()[1].preds.size(), 0u);  // unreachable
}

TEST(CFG, UnknownBranchTargetThrows) {
    Module m = testutil::parse_kernel("bra nowhere;\nret;");
    EXPECT_THROW(ControlFlowGraph::build(m.functions[0]), std::runtime_error);
}

TEST(CFG, DotExportMarksBackEdges) {
    Module m = parse_ptx(testutil::read_file("kernels/gemm_tiled.ptx"));
    auto dot = ControlFlowGraph::build(m.functions[0]).to_dot("gemm");
    EXPECT_TRUE(dot.find("digraph") != std::string::npos);
    EXPECT_TRUE(dot.find("style=dashed") != std::string::npos);
}

TEST(CFG, LinearizeRoundTrips) {
    Module m = parse_ptx(testutil::read_file("kernels/gemm_tiled.ptx"));
    Function copy = m.functions[0];
    auto cfg = ControlFlowGraph::build(copy);
    cfg.linearize(copy);
    EXPECT_EQ(copy.instructions.size(), m.functions[0].instructions.size());
    EXPECT_EQ(copy.labels.size(), m.functions[0].labels.size());
}

TEST(Liveness, StraightLinePressure) {
    auto c = make(".reg .b32 %r<6>;\n"
                  "mov.u32 %r1, 1;\nmov.u32 %r2, 2;\nmov.u32 %r3, 3;\n"
                  "add.s32 %r4, %r1, %r2;\nadd.s32 %r5, %r4, %r3;\n"
                  "st.global.u32 [%r5], %r5;\nret;");
    LivenessAnalysis live;
    live.run(c.fn(), c.cfg);
    // r1,r2,r3 are live together; r4 reuses a dying source's register, so the peak is 3, not 4.
    EXPECT_EQ(live.max_pressure(), 3u);
}

TEST(Liveness, SixtyFourBitRegistersCountDouble) {
    auto c = make(".reg .b64 %rd<4>;\n"
                  "mov.u64 %rd1, 1;\nmov.u64 %rd2, 2;\nadd.s64 %rd3, %rd1, %rd2;\n"
                  "st.global.u64 [%rd3], %rd3;\nret;");
    LivenessAnalysis live;
    live.run(c.fn(), c.cfg);
    EXPECT_EQ(live.max_pressure(), 4u);  // rd1 + rd2 live = 2 regs * 2 slots; rd3 reuses a dying source
}

TEST(Liveness, PredicatesDoNotCostGeneralRegisters) {
    auto c = make(".reg .pred %p<3>; .reg .b32 %r<3>;\n"
                  "setp.lt.s32 %p1, %r1, %r2;\n@%p1 add.s32 %r1, %r1, 1;\nst.global.u32 [%r1], %r1;\nret;");
    LivenessAnalysis live;
    live.run(c.fn(), c.cfg);
    EXPECT_EQ(live.max_pressure(), 2u);
    EXPECT_EQ(live.max_live_predicates(), 1u);
}

TEST(Liveness, PredicatedWriteKeepsOldValueLive) {
    auto c = make(".reg .pred %p<2>; .reg .b32 %r<4>; .reg .b64 %rd<2>;\n"
                  "mov.u32 %r1, 5;\n"
                  "setp.lt.u32 %p1, %r2, %r3;\n"
                  "@%p1 mov.u32 %r1, 7;\n"
                  "st.global.u32 [%rd1], %r1;\nret;");
    LivenessAnalysis live;
    live.run(c.fn(), c.cfg);
    // After `mov.u32 %r1, 5` the value must still be live across the predicated redefinition.
    const auto& after_first_mov = live.live_after()[0][0];
    EXPECT_TRUE(after_first_mov.test(reg_id(c.fn(), "%r1")));
    EXPECT_TRUE(c.cfg.blocks()[0].var_kill.test(reg_id(c.fn(), "%r1")));  // killed by the unconditional mov only
}

TEST(Liveness, LoopCarriedValueStaysLive) {
    auto c = make(".reg .pred %p<2>; .reg .b32 %r<4>;\n"
                  "mov.u32 %r1, 0;\nmov.u32 %r2, 10;\n"
                  "L1:\nadd.s32 %r1, %r1, 1;\nsetp.lt.s32 %p1, %r1, %r2;\n@%p1 bra L1;\n"
                  "st.global.u32 [%r1], %r1;\nret;");
    LivenessAnalysis live;
    live.run(c.fn(), c.cfg);
    size_t r1 = reg_id(c.fn(), "%r1"), r2 = reg_id(c.fn(), "%r2");
    const auto& loop = c.cfg.blocks()[1];
    EXPECT_TRUE(loop.live_in.test(r1));
    EXPECT_TRUE(loop.live_in.test(r2));
    EXPECT_TRUE(loop.live_out.test(r2));  // r2 flows around the back edge
    EXPECT_FALSE(c.cfg.blocks()[2].live_out.test(r1));
}

TEST(Liveness, DiamondJoinMergesBothArms) {
    auto c = make(".reg .pred %p<2>; .reg .b32 %r<5>;\n"
                  "setp.lt.s32 %p1, %r1, %r2;\n@%p1 bra ELSE;\n"
                  "mov.u32 %r3, 1;\nbra JOIN;\nELSE:\nmov.u32 %r3, 2;\n"
                  "JOIN:\nadd.s32 %r4, %r3, %r1;\nst.global.u32 [%r4], %r4;\nret;");
    LivenessAnalysis live;
    live.run(c.fn(), c.cfg);
    size_t r1 = reg_id(c.fn(), "%r1");
    // %r1 is read at the join, so it must be live through both arms.
    EXPECT_TRUE(c.cfg.blocks()[1].live_out.test(r1));
    EXPECT_TRUE(c.cfg.blocks()[2].live_out.test(r1));
}

TEST(DCE, RemovesDeadChains) {
    auto c = make(".reg .b32 %r<6>; .reg .b64 %rd<2>;\n"
                  "mov.u32 %r1, 1;\nadd.s32 %r2, %r1, 1;\nadd.s32 %r3, %r2, 1;\n"  // %r3 never used
                  "mov.u32 %r4, 9;\nst.global.u32 [%rd1], %r4;\nret;");
    DeadCodeElim dce;
    EXPECT_TRUE(dce.run(c.fn(), c.cfg));
    EXPECT_EQ(dce.removed(), 3u);
    EXPECT_EQ(c.cfg.blocks()[0].insts.size(), 3u);
}

TEST(DCE, KeepsSideEffectsAndBranchInputs) {
    auto c = make(".reg .pred %p<2>; .reg .b32 %r<4>; .reg .b64 %rd<2>;\n"
                  "ld.global.u32 %r1, [%rd1];\n"
                  "setp.eq.s32 %p1, %r1, 0;\n@%p1 bra L;\n"
                  "atom.global.add.u32 %r2, [%rd1], 1;\n"  // result unused but atomic has side effects
                  "L:\nret;");
    DeadCodeElim dce;
    dce.run(c.fn(), c.cfg);
    EXPECT_EQ(dce.removed(), 0u);
}

TEST(DCE, PredicatedWriteIsNotTreatedAsKill) {
    auto c = make(".reg .pred %p<2>; .reg .b32 %r<4>; .reg .b64 %rd<2>;\n"
                  "mov.u32 %r1, 5;\nsetp.lt.u32 %p1, %r2, %r3;\n@%p1 mov.u32 %r1, 7;\n"
                  "st.global.u32 [%rd1], %r1;\nret;");
    DeadCodeElim dce;
    dce.run(c.fn(), c.cfg);
    EXPECT_EQ(dce.removed(), 0u);  // the first mov is still the fall-back value
}

TEST(DCE, RemovesDeadLoopCarriedValue) {
    auto c = make(".reg .pred %p<2>; .reg .b32 %r<4>;\n"
                  "mov.u32 %r1, 0;\nmov.u32 %r2, 0;\n"
                  "L1:\nadd.s32 %r1, %r1, 1;\nadd.s32 %r2, %r2, 2;\nsetp.lt.s32 %p1, %r1, 10;\n@%p1 bra L1;\nret;");
    DeadCodeElim dce;
    dce.run(c.fn(), c.cfg);
    EXPECT_EQ(dce.removed(), 2u);  // %r2 only feeds itself
}

namespace {
// Independent oracle: no two interfering registers may overlap in physical slots.
bool assignment_valid(Function& fn, ControlFlowGraph& cfg, const AllocResult& r, uint32_t K) {
    LivenessAnalysis live;
    live.run(fn, cfg);
    auto rig = InterferenceGraph::build(fn, cfg, live);
    for (auto v : rig.nodes()) {
        if (fn.regs[v].type == DataType::Pred) continue;
        auto it = r.assignment.find(v);
        if (it == r.assignment.end()) return false;
        uint32_t w = reg_slots(fn.regs[v].type);
        if (it->second % w != 0 || it->second + w > K) return false;  // alignment + budget
        for (auto u : rig.neighbors(v)) {
            if (fn.regs[u].type == DataType::Pred) continue;
            uint32_t uw = reg_slots(fn.regs[u].type), us = r.assignment.at(u);
            if (it->second < us + uw && us < it->second + w) return false;
        }
    }
    return true;
}
}  // namespace

TEST(Interference, SimultaneouslyLiveRegistersInterfere) {
    auto c = make(".reg .b32 %r<5>;\n"
                  "mov.u32 %r1, 1;\nmov.u32 %r2, 2;\nadd.s32 %r3, %r1, %r2;\nadd.s32 %r4, %r3, 1;\n"
                  "st.global.u32 [%r4], %r4;\nret;");
    LivenessAnalysis live;
    live.run(c.fn(), c.cfg);
    auto rig = InterferenceGraph::build(c.fn(), c.cfg, live);
    size_t r1 = reg_id(c.fn(), "%r1"), r2 = reg_id(c.fn(), "%r2"), r3 = reg_id(c.fn(), "%r3"), r4 = reg_id(c.fn(), "%r4");
    EXPECT_TRUE(rig.has_edge(r1, r2));
    EXPECT_FALSE(rig.has_edge(r1, r3));  // r1 dies where r3 is born
    EXPECT_FALSE(rig.has_edge(r3, r4));
    EXPECT_TRUE(rig.has_edge(r2, r1));   // symmetric
}

TEST(RegAlloc, NoSpillWhenBudgetIsLarge) {
    Module m = parse_ptx(testutil::read_file("kernels/gemm_tiled.ptx"));
    auto cfg = ControlFlowGraph::build(m.functions[0]);
    RegAllocChaitin ra({});
    EXPECT_FALSE(ra.run(m.functions[0], cfg));
    const auto& r = ra.result();
    EXPECT_TRUE(r.converged);
    EXPECT_TRUE(r.fits_without_spill);
    EXPECT_EQ(r.frame_bytes, 0u);
    EXPECT_GE(r.regs_used, r.peak_pressure);  // can't color below the peak live set
    EXPECT_TRUE(assignment_valid(m.functions[0], cfg, r, 255));
}

TEST(RegAlloc, TightBudgetForcesSpill) {
    auto c = make(".reg .b64 %rd<2>; .reg .f32 %f<8>;\n"
                  "ld.global.f32 %f1, [%rd1];\nld.global.f32 %f2, [%rd1+4];\n"
                  "ld.global.f32 %f3, [%rd1+8];\nld.global.f32 %f4, [%rd1+12];\n"
                  "add.f32 %f5, %f1, %f2;\nadd.f32 %f6, %f3, %f4;\nadd.f32 %f7, %f5, %f6;\n"
                  "st.global.f32 [%rd1], %f7;\nret;");
    AllocOptions o;
    o.max_regs = 5;  // %rd1 takes 2 slots; 4 floats are live at once
    RegAllocChaitin ra(o);
    EXPECT_TRUE(ra.run(c.fn(), c.cfg));
    const auto& r = ra.result();
    EXPECT_FALSE(r.fits_without_spill);
    EXPECT_TRUE(r.converged);
    EXPECT_LE(r.regs_used, 5u);
    EXPECT_GT(r.frame_bytes, 0u);
    EXPECT_GT(r.spill_loads + r.spill_stores, 0u);
    EXPECT_TRUE(assignment_valid(c.fn(), c.cfg, r, 5));
}

TEST(RegAlloc, SixtyFourBitValuesGetAlignedPairs) {
    auto c = make(".reg .b64 %rd<6>; .reg .b32 %r<3>;\n"
                  "mov.u32 %r1, 1;\nmov.u64 %rd1, 1;\nmov.u64 %rd2, 2;\nmov.u64 %rd3, 3;\n"
                  "add.s64 %rd4, %rd1, %rd2;\nadd.s64 %rd5, %rd4, %rd3;\n"
                  "st.global.u32 [%rd5], %r1;\nret;");
    RegAllocChaitin ra({});
    ra.run(c.fn(), c.cfg);
    for (auto& [reg, slot] : ra.result().assignment)
        if (reg_slots(c.fn().regs[reg].type) == 2) EXPECT_EQ(slot % 2, 0u);
    EXPECT_TRUE(assignment_valid(c.fn(), c.cfg, ra.result(), 255));
}

TEST(RegAlloc, SpillCodeUsesLocalDepotAndReparses) {
    Module m = parse_ptx(testutil::read_file("kernels/pressure_loop.ptx"));
    Function& fn = m.functions[0];
    auto cfg = ControlFlowGraph::build(fn);
    AllocOptions o;
    o.max_regs = 32;
    RegAllocChaitin ra(o);
    EXPECT_TRUE(ra.run(fn, cfg));
    const auto& r = ra.result();
    EXPECT_TRUE(r.converged);
    EXPECT_FALSE(r.fits_without_spill);
    EXPECT_LE(r.regs_used, 32u);
    EXPECT_GT(r.frame_bytes, 0u);
    EXPECT_TRUE(assignment_valid(fn, cfg, r, 32));

    cfg.linearize(fn);
    std::string text = print_function(fn);
    EXPECT_TRUE(text.find(".local .align") != std::string::npos);
    EXPECT_TRUE(text.find("st.local.b32 [__spill_depot") != std::string::npos);
    EXPECT_TRUE(text.find("ld.local.b32") != std::string::npos);

    // The rewritten kernel is itself valid input for the front end.
    Module m2 = parse_ptx(".version 8.0\n.target sm_80\n.address_size 64\n" + text);
    EXPECT_EQ(m2.functions[0].instructions.size(), fn.instructions.size());
}

TEST(RegAlloc, PredicatedSpilledDefReloadsOldValue) {
    // %f1 is spilled; the predicated redefinition must reload the old value before and store after.
    auto c = make(".reg .pred %p<2>; .reg .b64 %rd<2>; .reg .f32 %f<6>; .reg .b32 %r<2>;\n"
                  "ld.global.f32 %f1, [%rd1];\nld.global.f32 %f2, [%rd1+4];\nld.global.f32 %f3, [%rd1+8];\n"
                  "setp.lt.s32 %p1, %r1, 3;\n@%p1 mov.f32 %f1, %f2;\n"
                  "add.f32 %f4, %f1, %f2;\nadd.f32 %f5, %f4, %f3;\nst.global.f32 [%rd1], %f5;\nret;");
    AllocOptions o;
    o.max_regs = 4;
    RegAllocChaitin ra(o);
    ra.run(c.fn(), c.cfg);
    EXPECT_TRUE(ra.result().converged);
    EXPECT_TRUE(assignment_valid(c.fn(), c.cfg, ra.result(), 4));

    // Every predicated write to a spill temp must be preceded by a reload of the old value.
    int checked = 0;
    for (auto& b : c.cfg.blocks())
        for (size_t i = 0; i < b.insts.size(); ++i) {
            const auto& ins = b.insts[i];
            if (!ins.predicate || !ins.has_dest || ins.defs().empty()) continue;
            if (c.fn().regs[ins.defs()[0].id].name.rfind("%__sp", 0) != 0) continue;
            ++checked;
            ASSERT_TRUE(i > 0);
            EXPECT_TRUE(b.insts[i - 1].opcode == Opcode::LD);
            EXPECT_EQ(b.insts[i - 1].defs()[0].id, ins.defs()[0].id);
        }
    EXPECT_GE(checked, 1);
}

TEST(RegAlloc, ImpossibleBudgetFailsWithinRoundLimit) {
    auto c = make(".reg .b64 %rd<4>;\nmov.u64 %rd1, 1;\nmov.u64 %rd2, 2;\nadd.s64 %rd3, %rd1, %rd2;\n"
                  "st.global.u64 [%rd3], %rd3;\nret;");
    AllocOptions o;
    o.max_regs = 2;  // a 64-bit add needs three pairs (or at least two live + one result)
    RegAllocChaitin ra(o);
    ra.run(c.fn(), c.cfg);
    EXPECT_FALSE(ra.result().converged);
    EXPECT_LE(ra.result().rounds, o.max_rounds);
}

TEST(RegAlloc, SpillCostPrefersCheapOutOfLoopValues) {
    Module m = parse_ptx(testutil::read_file("kernels/pressure_loop.ptx"));
    auto cfg = ControlFlowGraph::build(m.functions[0]);
    AllocOptions o;
    o.max_regs = 32;
    o.rewrite_spills = false;
    RegAllocChaitin ra(o);
    ra.run(m.functions[0], cfg);
    const auto& cand = ra.result().potential_spills;
    ASSERT_GE(cand.size(), 2u);
    EXPECT_TRUE(cand.front().cost <= cand.back().cost);  // sorted cheapest first
    EXPECT_FALSE(ra.result().converged);
}

TEST(Pipeline, AnalyzesGemm) {
    Module m = parse_ptx(testutil::read_file("kernels/gemm_tiled.ptx"));
    AnalysisOptions o;
    o.max_regs = 32;
    KernelAnalysis a = analyze_function(m.functions[0], m, o);
    EXPECT_EQ(a.name, "gemm_tiled");
    EXPECT_EQ(a.blocks, 7u);
    EXPECT_EQ(a.loops, 2u);
    EXPECT_GT(a.peak_slots, 0u);
    EXPECT_TRUE(a.alloc.converged);
    EXPECT_TRUE(a.heatmap.find("fma.rn.f32") != std::string::npos);
    EXPECT_TRUE(a.cfg_dot.find("digraph") != std::string::npos);
}

TEST(Pipeline, DoesNotMutateInput) {
    Module m = parse_ptx(testutil::read_file("kernels/pressure_loop.ptx"));
    size_t before = m.functions[0].instructions.size();
    AnalysisOptions o;
    o.max_regs = 32;
    KernelAnalysis a = analyze_function(m.functions[0], m, o);
    EXPECT_EQ(m.functions[0].instructions.size(), before);
    EXPECT_FALSE(a.spilled_ptx.empty());
}
