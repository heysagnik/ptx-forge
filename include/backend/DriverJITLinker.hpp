#pragma once
#include <memory>
#include <string>

#include "backend/CubinProvider.hpp"

namespace ptxforge::backend {

// Compiles PTX to a cubin entirely in memory through the CUDA Driver JIT linker
// (cuLinkCreate / cuLinkAddData / cuLinkComplete). The driver library (nvcuda.dll / libcuda.so.1)
// is loaded at run time, so ptx-forge builds and runs without the CUDA SDK or a GPU installed.
class DriverJITLinker : public CubinProvider {
public:
    DriverJITLinker();
    ~DriverJITLinker() override;

    // True if the driver library loaded and a CUDA device is present.
    bool available() const;
    const std::string& unavailable_reason() const;
    std::string device_name() const;
    std::string device_arch() const;  // "sm_86"

    std::string name() const override { return "CUDA driver JIT (cuLink*)"; }
    CompileResult compile(std::string_view ptx, const CompileOptions& opts) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Runs the ptxas executable (CUDA_PATH/bin or PATH) on a temporary file.
class PtxasProvider : public CubinProvider {
public:
    explicit PtxasProvider(std::string ptxas_path = {});
    bool available() const { return !path_.empty(); }
    const std::string& path() const { return path_; }
    std::string name() const override { return "ptxas executable"; }
    CompileResult compile(std::string_view ptx, const CompileOptions& opts) override;

private:
    std::string path_;
};

}  // namespace ptxforge::backend
