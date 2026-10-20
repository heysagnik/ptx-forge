#include "backend/CubinProvider.hpp"

#include <fstream>
#include <iterator>
#include <regex>
#include <sstream>

namespace ptxforge::backend {

std::map<std::string, PtxasFunctionStats> parse_ptxas_log(const std::string& log) {
    static const std::regex compiling(R"(Compiling entry function '([^']+)')");
    static const std::regex props(R"(Function properties for (\S+))");
    static const std::regex used(R"(Used (\d+) registers)");
    static const std::regex stack(R"((\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads)");

    std::map<std::string, PtxasFunctionStats> out;
    std::string current;
    std::istringstream in(log);
    std::string line;
    std::smatch m;
    while (std::getline(in, line)) {
        if (std::regex_search(line, m, compiling)) { current = m[1]; continue; }
        if (std::regex_search(line, m, props)) { current = m[1]; continue; }
        if (current.empty()) continue;
        if (std::regex_search(line, m, used)) {
            out[current].registers = static_cast<uint32_t>(std::stoul(m[1]));
            out[current].has_registers = true;
        }
        if (std::regex_search(line, m, stack)) {
            out[current].stack_bytes = static_cast<uint32_t>(std::stoul(m[1]));
            out[current].spill_store_bytes = static_cast<uint32_t>(std::stoul(m[2]));
            out[current].spill_load_bytes = static_cast<uint32_t>(std::stoul(m[3]));
        }
    }
    return out;
}

CompileResult FileCubinProvider::compile(std::string_view, const CompileOptions&) {
    CompileResult r;
    std::ifstream f(path_, std::ios::binary);
    if (!f) { r.error_log = "cannot open " + path_; return r; }
    r.cubin.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    r.ok = true;
    return r;
}

}  // namespace ptxforge::backend
