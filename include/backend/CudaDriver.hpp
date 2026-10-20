#pragma once
#include <cstddef>
#include <memory>
#include <string>

namespace ptxforge::backend {

// The few CUDA driver types and constants ptx-forge needs, so building does not require cuda.h.
// Values were checked against cuda.h.
namespace cuda {
using CUresult = int;
using CUdevice = int;
using CUcontext = void*;
using CUlinkState = void*;
using CUmodule = void*;
using CUfunction = void*;
using CUdeviceptr = unsigned long long;
using CUjit_option = int;

constexpr CUjit_option kJitMaxRegisters = 0;
constexpr CUjit_option kJitInfoLogBuffer = 3;
constexpr CUjit_option kJitInfoLogBufferSize = 4;
constexpr CUjit_option kJitErrorLogBuffer = 5;
constexpr CUjit_option kJitErrorLogBufferSize = 6;
constexpr CUjit_option kJitTarget = 9;
constexpr CUjit_option kJitLogVerbose = 12;
constexpr int kJitInputPtx = 1;
constexpr int kFuncSharedBytes = 1;
constexpr int kFuncLocalBytes = 3;
constexpr int kFuncNumRegs = 4;

// Entry points resolved at run time from nvcuda.dll / libcuda.so.1.
struct Api {
    CUresult (*cuInit)(unsigned) = nullptr;
    CUresult (*cuGetErrorString)(CUresult, const char**) = nullptr;
    CUresult (*cuLinkCreate)(unsigned, CUjit_option*, void**, CUlinkState*) = nullptr;
    CUresult (*cuLinkAddData)(CUlinkState, int, void*, size_t, const char*, unsigned, CUjit_option*, void**) = nullptr;
    CUresult (*cuLinkComplete)(CUlinkState, void**, size_t*) = nullptr;
    CUresult (*cuLinkDestroy)(CUlinkState) = nullptr;
    CUresult (*cuModuleLoadDataEx)(CUmodule*, const void*, unsigned, CUjit_option*, void**) = nullptr;
    CUresult (*cuModuleUnload)(CUmodule) = nullptr;
    CUresult (*cuModuleGetFunction)(CUfunction*, CUmodule, const char*) = nullptr;
    CUresult (*cuFuncGetAttribute)(int*, int, CUfunction) = nullptr;
    CUresult (*cuMemAlloc)(CUdeviceptr*, size_t) = nullptr;
    CUresult (*cuMemFree)(CUdeviceptr) = nullptr;
    CUresult (*cuMemcpyHtoD)(CUdeviceptr, const void*, size_t) = nullptr;
    CUresult (*cuMemcpyDtoH)(void*, CUdeviceptr, size_t) = nullptr;
    CUresult (*cuLaunchKernel)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                               void*, void**, void**) = nullptr;
    CUresult (*cuCtxSynchronize)() = nullptr;
};
}  // namespace cuda

// Loads the CUDA driver, picks device 0 and keeps its primary context. Shared by the JIT linker and
// the GPU runner. If anything is missing, available() is false and unavailable_reason() says why.
class CudaDriver {
public:
    CudaDriver();
    ~CudaDriver();
    CudaDriver(const CudaDriver&) = delete;
    CudaDriver& operator=(const CudaDriver&) = delete;

    bool available() const;
    const std::string& unavailable_reason() const;
    std::string device_name() const;
    std::string device_arch() const;

    const cuda::Api& api() const;
    void make_current() const;
    std::string error_string(cuda::CUresult r) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ptxforge::backend
