#pragma once
#include "backend/CubinProvider.hpp"
#include "backend/CudaDriver.hpp"

namespace ptxforge::backend {

// Compiles PTX to a cubin entirely in memory through the driver JIT linker
// (cuLinkCreate / cuLinkAddData / cuLinkComplete).
class DriverJITLinker : public CubinProvider {
public:
    explicit DriverJITLinker(CudaDriver& driver) : driver_(driver) {}
    std::string name() const override { return "CUDA driver JIT (cuLink*)"; }
    CompileResult compile(std::string_view ptx, const CompileOptions& opts) override;

private:
    CudaDriver& driver_;
};

}  // namespace ptxforge::backend
