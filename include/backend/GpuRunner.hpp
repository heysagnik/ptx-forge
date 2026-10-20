#pragma once
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "backend/CudaDriver.hpp"

namespace ptxforge::backend {

// One kernel parameter. Buffers are device allocations of `bytes`, filled with fill_pattern(seed).
struct KernelArg {
    enum class Kind { Buffer, U32, U64, F32 } kind = Kind::Buffer;
    size_t bytes = 0;
    uint64_t value = 0;  // U32 / U64, or the bit pattern of an F32
    uint32_t seed = 0;
};

struct LaunchConfig {
    uint32_t grid[3] = {1, 1, 1};
    uint32_t block[3] = {1, 1, 1};
    uint32_t dynamic_shared = 0;
};

struct RunResult {
    bool ok = false;
    std::string error;
    std::vector<std::vector<uint8_t>> buffers;
    int registers = 0;
    int local_bytes = 0;                        // local memory per thread (spills + stack)
    int shared_bytes = 0;
};

// Deterministic float-valued test data in [-1, 1), `bytes` long.
std::vector<uint8_t> fill_pattern(uint32_t seed, size_t bytes);

// Loads PTX through the driver (optionally capped at max_regs registers), launches one kernel
// and copies its buffers back.
class GpuRunner {
public:
    explicit GpuRunner(CudaDriver& driver) : driver_(driver) {}
    RunResult run(std::string_view ptx, const std::string& kernel, const std::vector<KernelArg>& args,
                  const LaunchConfig& cfg, uint32_t max_regs = 0);

private:
    CudaDriver& driver_;
};

}  // namespace ptxforge::backend
