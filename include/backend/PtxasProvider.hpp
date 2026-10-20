#pragma once
#include <string>

#include "backend/CubinProvider.hpp"

namespace ptxforge::backend {

// Runs the ptxas executable (CUDA_PATH/bin or PATH) on a temporary file. Needs no GPU.
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
