// Runner for the built-in test shim. Not compiled when linking against gtest_main.
#include "Test.hpp"

#ifndef PTXFORGE_USE_GTEST
int main(int argc, char** argv) {
    std::string filter = argc > 1 ? argv[1] : "";
    int ran = 0, failed_cases = 0;
    for (auto& c : ptxtest::registry()) {
        std::string full = std::string(c.suite) + "." + c.name;
        if (!filter.empty() && full.find(filter) == std::string::npos) continue;
        int before = ptxtest::failures();
        try { c.fn(); }
        catch (const ptxtest::AssertAbort&) {}
        catch (const std::exception& e) { ptxtest::fail(__FILE__, __LINE__, std::string("unexpected exception: ") + e.what(), false); }
        ++ran;
        bool ok = ptxtest::failures() == before;
        failed_cases += !ok;
        std::cout << (ok ? "[  OK  ] " : "[ FAIL ] ") << full << "\n";
    }
    std::cout << ran - failed_cases << "/" << ran << " tests passed\n";
    return failed_cases ? 1 : 0;
}
#endif
