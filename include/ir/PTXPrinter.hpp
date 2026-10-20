#pragma once
#include <string>

#include "ir/Module.hpp"

namespace ptxforge::ir {

std::string format_operand(const Operand& op);
std::string format_instruction(const PTXInstruction& ins);
std::string print_function(const Function& fn);
std::string print_module(const Module& m);

}  // namespace ptxforge::ir
