#pragma once
#include <fstream>
#include <iterator>
#include <string>

#include "frontend/Parser.hpp"
#include "ir/ControlFlowGraph.hpp"
#include "ir/PTXPrinter.hpp"

#ifndef PTXFORGE_TEST_DIR
#define PTXFORGE_TEST_DIR "test"
#endif

namespace testutil {

inline std::string read_file(const std::string& rel) {
    std::ifstream f(std::string(PTXFORGE_TEST_DIR) + "/" + rel, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// Wraps a kernel body (register declarations + instructions) into a complete PTX module.
inline std::string wrap_kernel(const std::string& body) {
    return ".version 8.0\n.target sm_80\n.address_size 64\n"
           ".visible .entry k(.param .u64 k_param_0)\n{\n" + body + "\n}\n";
}

inline ptxforge::ir::Module parse_kernel(const std::string& body) {
    return ptxforge::frontend::parse_ptx(wrap_kernel(body));
}

}  // namespace testutil
