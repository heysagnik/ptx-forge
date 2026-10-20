#pragma once
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace ptxforge::backend {

struct CompileOptions {
    uint32_t max_regs = 0;    // 0 = driver default
    std::string target_sm;    // "sm_80"; empty = provider default (device arch for the driver)
};

// Per-function numbers reported by ptxas ("ptxas info : Used N registers ...").
struct PtxasFunctionStats {
    uint32_t registers = 0;
    uint32_t stack_bytes = 0;
    uint32_t spill_store_bytes = 0;
    uint32_t spill_load_bytes = 0;
    bool has_registers = false;
};

struct CompileResult {
    bool ok = false;
    std::vector<uint8_t> cubin;
    std::string info_log;
    std::string error_log;
    std::map<std::string, PtxasFunctionStats> ptxas_stats;  // by mangled function name
};

// Source of ground truth. Implementations: the CUDA driver JIT linker, the ptxas executable,
// or a pre-built .cubin on disk (so everything works on a machine without a GPU).
class CubinProvider {
public:
    virtual ~CubinProvider() = default;
    virtual std::string name() const = 0;
    virtual CompileResult compile(std::string_view ptx, const CompileOptions& opts) = 0;
};

// Parses `ptxas -v` / CU_JIT_INFO_LOG output.
std::map<std::string, PtxasFunctionStats> parse_ptxas_log(const std::string& log);

// Returns the stored cubin regardless of the PTX passed in.
class FileCubinProvider : public CubinProvider {
public:
    explicit FileCubinProvider(std::string path) : path_(std::move(path)) {}
    std::string name() const override { return "cubin file " + path_; }
    CompileResult compile(std::string_view ptx, const CompileOptions& opts) override;

private:
    std::string path_;
};

}  // namespace ptxforge::backend
