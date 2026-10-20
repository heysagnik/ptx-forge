#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "backend/CubinProvider.hpp"

namespace ptxforge::backend {

// One kernel parameter. Buffers are device allocations of `bytes`, filled with fill_pattern(seed).
struct KernelArg {
    enum class Kind { Buffer, U32, U64, F32 } kind = Kind::Buffer;
    size_t bytes = 0;     // Buffer
    uint64_t value = 0;   // U32 / U64, or the bit pattern of an F32
    uint32_t seed = 0;    // Buffer fill seed
};

struct LaunchConfig {
    uint32_t grid[3] = {1, 1, 1};
    uint32_t block[3] = {1, 1, 1};
    uint32_t dynamic_shared = 0;
};

struct RunResult {
    bool ok = false;
    std::string error;
    std::vector<std::vector<uint8_t>> buffers;  // contents of each Buffer arg after the launch
    int registers = 0;       // CU_FUNC_ATTRIBUTE_NUM_REGS
    int local_bytes = 0;     // CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES (spill + stack)
    int shared_bytes = 0;    // static shared memory
    int max_threads = 0;
};

// Deterministic float-valued test data in [-1, 1), `bytes` long.
std::vector<uint8_t> fill_pattern(uint32_t seed, size_t bytes);

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

    // JIT-loads `ptx` (optionally capped at max_regs registers), launches `kernel` on the GPU and
    // copies the buffers back. Also reports the register / local-memory use the driver chose.
    RunResult run(std::string_view ptx, const std::string& kernel, const std::vector<KernelArg>& args,
                  const LaunchConfig& cfg, uint32_t max_regs = 0);

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
