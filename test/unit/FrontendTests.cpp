#include "Test.hpp"
#include "TestUtil.hpp"
#include "frontend/Lexer.hpp"

using namespace ptxforge;
using namespace ptxforge::frontend;
using namespace ptxforge::ir;

TEST(Lexer, PredicatedBranch) {
    auto toks = Lexer("@%p1 bra $L__BB0_2;").tokenize();
    ASSERT_EQ(toks.size(), 6u);
    EXPECT_TRUE(toks[0].kind == TokenKind::At);
    EXPECT_TRUE(toks[1].kind == TokenKind::Register);
    EXPECT_EQ(toks[1].text, "%p1");
    EXPECT_TRUE(toks[2].kind == TokenKind::Identifier);
    EXPECT_EQ(toks[3].text, "$L__BB0_2");
    EXPECT_TRUE(toks[4].kind == TokenKind::Semi);
}

TEST(Lexer, DottedMnemonicsAndDirectives) {
    auto toks = Lexer(".reg .b32 %r<5>; mul.wide.s32 %rd1, %r1, 4;").tokenize();
    EXPECT_TRUE(toks[0].kind == TokenKind::Directive);
    EXPECT_EQ(toks[0].text, ".reg");
    EXPECT_EQ(toks[1].text, ".b32");
    EXPECT_TRUE(toks[3].kind == TokenKind::Less);
    EXPECT_EQ(toks[4].int_value, 5);
    EXPECT_EQ(toks[7].text, "mul.wide.s32");
}

TEST(Lexer, NumericLiterals) {
    auto toks = Lexer("0f3F800000 0d4000000000000000 0x1F 42U 8.0 1.5e2").tokenize();
    EXPECT_TRUE(toks[0].kind == TokenKind::Float);
    EXPECT_EQ(toks[0].float_value, 1.0);
    EXPECT_EQ(toks[1].float_value, 2.0);
    EXPECT_EQ(toks[2].int_value, 31);
    EXPECT_EQ(toks[3].int_value, 42);
    EXPECT_TRUE(toks[4].kind == TokenKind::Float);
    EXPECT_EQ(toks[5].float_value, 150.0);
}

TEST(Lexer, CommentsAndLineTracking) {
    auto toks = Lexer("// c1\n/* multi\nline */ ret;").tokenize();
    EXPECT_EQ(toks[0].text, "ret");
    EXPECT_EQ(toks[0].line, 3u);
}

TEST(Lexer, UnterminatedCommentThrows) { EXPECT_THROW(Lexer("/* oops").tokenize(), ParseError); }

TEST(Parser, ModuleDirectives) {
    Module m = parse_ptx(testutil::read_file("kernels/vectorAdd.ptx"));
    EXPECT_EQ(m.version, "8.0");
    EXPECT_EQ(m.target, "sm_80");
    EXPECT_EQ(m.address_size, 64u);
    ASSERT_EQ(m.functions.size(), 1u);
    const Function& f = m.functions[0];
    EXPECT_EQ(f.name, "_Z9vectorAddPKfS0_Pfi");
    EXPECT_TRUE(f.is_entry);
    EXPECT_EQ(f.params.size(), 4u);
    EXPECT_EQ(f.params[0].bytes, 8u);
    EXPECT_EQ(f.params[3].bytes, 4u);
}

TEST(Parser, RegisterDeclarationsExpandRanges) {
    Module m = parse_ptx(testutil::read_file("kernels/vectorAdd.ptx"));
    const Function& f = m.functions[0];
    EXPECT_EQ(f.regs.size(), 23u);  // p<2> + r<6> + f<4> + rd<11>
    EXPECT_EQ(f.regs[0].name, "%p0");
    EXPECT_TRUE(f.regs[0].type == DataType::Pred);
    for (uint32_t i = 0; i < f.regs.size(); ++i) EXPECT_EQ(f.regs[i].id, i);
}

TEST(Parser, InstructionOperandsAndPredicates) {
    Module m = parse_ptx(testutil::read_file("kernels/vectorAdd.ptx"));
    const Function& f = m.functions[0];
    const PTXInstruction& ld = f.instructions[0];
    EXPECT_TRUE(ld.opcode == Opcode::LD);
    EXPECT_TRUE(ld.space == StateSpace::Param);
    EXPECT_TRUE(ld.type == DataType::U64);
    ASSERT_TRUE(ld.has_dest);
    auto* mem = std::get_if<MemoryRef>(&ld.operands[1]);
    ASSERT_TRUE(mem != nullptr);
    EXPECT_EQ(mem->symbol, "_Z9vectorAddPKfS0_Pfi_param_0");

    // setp + predicated bra
    const PTXInstruction* bra = nullptr;
    for (auto& i : f.instructions) if (i.opcode == Opcode::BRA) bra = &i;
    ASSERT_TRUE(bra != nullptr);
    ASSERT_TRUE(bra->predicate.has_value());
    EXPECT_EQ(bra->predicate->pred_reg.name, "%p1");
    EXPECT_FALSE(bra->predicate->is_negated);
    EXPECT_EQ(bra->branch_target(), "$L__BB0_2");
    EXPECT_FALSE(bra->has_dest);
}

TEST(Parser, NegatedPredicateAndMemoryOffsets) {
    Module m = testutil::parse_kernel(
        ".reg .pred %p<2>; .reg .b64 %rd<3>; .reg .f32 %f<2>;\n"
        "@!%p1 ld.global.f32 %f1, [%rd1+16];\n"
        "st.global.f32 [%rd2-8], %f1;\n"
        "ret;");
    const auto& is = m.functions[0].instructions;
    ASSERT_EQ(is.size(), 3u);
    EXPECT_TRUE(is[0].predicate->is_negated);
    EXPECT_EQ(std::get<MemoryRef>(is[0].operands[1]).offset, 16);
    EXPECT_EQ(std::get<MemoryRef>(is[1].operands[0]).offset, -8);
    EXPECT_FALSE(is[1].has_dest);  // stores never define a register
}

TEST(Parser, VectorAndDualDestinations) {
    Module m = testutil::parse_kernel(
        ".reg .pred %p<3>; .reg .b64 %rd<2>; .reg .f32 %f<5>; .reg .b32 %r<3>;\n"
        "ld.global.v4.f32 {%f1, %f2, %f3, %f4}, [%rd1];\n"
        "setp.lt.s32 %p1|%p2, %r1, %r2;\n"
        "ret;");
    const auto& is = m.functions[0].instructions;
    EXPECT_EQ(is[0].defs().size(), 4u);
    EXPECT_EQ(is[1].defs().size(), 2u);
    EXPECT_EQ(is[1].get_used_registers().size(), 2u);
}

TEST(Parser, SpecialRegistersAreNotVirtual) {
    Module m = testutil::parse_kernel(".reg .b32 %r<2>;\nmov.u32 %r1, %tid.x;\nret;");
    const PTXInstruction& mov = m.functions[0].instructions[0];
    EXPECT_TRUE(std::holds_alternative<std::string>(mov.operands[1]));
    EXPECT_EQ(mov.get_used_registers().size(), 0u);
}

TEST(Parser, LocLineDirectivesAndPragmasAreSkipped) {
    Module m = testutil::parse_kernel(
        ".reg .b32 %r<3>;\n.loc 1 5 3\nadd.s32 %r1, %r2, 1;\n.loc 1 6 3, function_name $L__info_string0\nret;");
    EXPECT_EQ(m.functions[0].instructions.size(), 2u);
}

TEST(Parser, UndeclaredRegisterReportsLine) {
    try {
        testutil::parse_kernel(".reg .b32 %r<2>;\nadd.s32 %r1, %r1, %r9;\nret;");
        EXPECT_TRUE(false);
    } catch (const ParseError& e) {
        EXPECT_EQ(e.line, 7u);  // 5 lines of module preamble + .reg line + instruction line
    }
}

TEST(Parser, MissingSemicolonThrows) {
    EXPECT_THROW(testutil::parse_kernel(".reg .b32 %r<2>;\nadd.s32 %r1, %r1, 1\nret;"), ParseError);
}

TEST(Parser, MissingVersionThrows) { EXPECT_THROW(parse_ptx(".target sm_80\n"), ParseError); }

TEST(Parser, SharedMemoryFootprint) {
    Module m = parse_ptx(testutil::read_file("kernels/gemm_tiled.ptx"));
    EXPECT_EQ(m.functions[0].static_shared_bytes(), 2048u);
}

TEST(Printer, RoundTripIsStable) {
    for (const char* file : {"kernels/vectorAdd.ptx", "kernels/gemm_tiled.ptx", "kernels/pressure_loop.ptx"}) {
        Module a = parse_ptx(testutil::read_file(file));
        std::string once = print_module(a);
        Module b = parse_ptx(once);
        EXPECT_EQ(print_module(b), once);
        EXPECT_EQ(b.functions[0].instructions.size(), a.functions[0].instructions.size());
    }
}
