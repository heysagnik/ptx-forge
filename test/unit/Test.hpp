#pragma once
// Uses GoogleTest when it is installed (PTXFORGE_USE_GTEST), otherwise a tiny compatible subset
// (TEST, EXPECT_*/ASSERT_*), so the suite builds and runs on machines without gtest.
#ifdef PTXFORGE_USE_GTEST
#include <gtest/gtest.h>
#else
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace ptxtest {
struct Case { const char* suite; const char* name; std::function<void()> fn; };
inline std::vector<Case>& registry() { static std::vector<Case> r; return r; }
inline int& failures() { static int f = 0; return f; }
struct Register { Register(const char* s, const char* n, std::function<void()> f) { registry().push_back({s, n, std::move(f)}); } };
struct AssertAbort {};
inline void fail(const char* file, int line, const std::string& msg, bool fatal) {
    ++failures();
    std::cerr << file << ":" << line << ": Failure\n  " << msg << "\n";
    if (fatal) throw AssertAbort{};
}
template <class A, class B>
std::string cmp_msg(const char* a, const char* b, const A& va, const B& vb, const char* op) {
    std::ostringstream os;
    os << "Expected: (" << a << ") " << op << " (" << b << "), actual: " << va << " vs " << vb;
    return os.str();
}
}  // namespace ptxtest

#define TEST(suite, name)                                                                    \
    static void ptxtest_##suite##_##name();                                                  \
    static ptxtest::Register ptxtest_reg_##suite##_##name(#suite, #name, ptxtest_##suite##_##name); \
    static void ptxtest_##suite##_##name()

#define PTXTEST_CMP(a, b, op, fatal)                                                         \
    do {                                                                                     \
        auto&& va_ = (a); auto&& vb_ = (b);                                                  \
        if (!(va_ op vb_)) ptxtest::fail(__FILE__, __LINE__, ptxtest::cmp_msg(#a, #b, va_, vb_, #op), fatal); \
    } while (0)
#define EXPECT_EQ(a, b) PTXTEST_CMP(a, b, ==, false)
#define EXPECT_NE(a, b) PTXTEST_CMP(a, b, !=, false)
#define EXPECT_LT(a, b) PTXTEST_CMP(a, b, <, false)
#define EXPECT_LE(a, b) PTXTEST_CMP(a, b, <=, false)
#define EXPECT_GT(a, b) PTXTEST_CMP(a, b, >, false)
#define EXPECT_GE(a, b) PTXTEST_CMP(a, b, >=, false)
#define ASSERT_EQ(a, b) PTXTEST_CMP(a, b, ==, true)
#define ASSERT_GE(a, b) PTXTEST_CMP(a, b, >=, true)
#define EXPECT_TRUE(c) do { if (!(c)) ptxtest::fail(__FILE__, __LINE__, "Expected true: " #c, false); } while (0)
#define EXPECT_FALSE(c) do { if (c) ptxtest::fail(__FILE__, __LINE__, "Expected false: " #c, false); } while (0)
#define ASSERT_TRUE(c) do { if (!(c)) ptxtest::fail(__FILE__, __LINE__, "Expected true: " #c, true); } while (0)
#define EXPECT_THROW(stmt, ex)                                                               \
    do {                                                                                     \
        bool thrown_ = false;                                                                \
        try { stmt; } catch (const ex&) { thrown_ = true; } catch (...) {}                   \
        if (!thrown_) ptxtest::fail(__FILE__, __LINE__, "Expected " #ex " from: " #stmt, false); \
    } while (0)
#endif
