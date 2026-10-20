#include "backend/CudaDriver.hpp"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace ptxforge::backend {

namespace {
constexpr int kAttrCcMajor = 75;
constexpr int kAttrCcMinor = 76;

#ifdef _WIN32
using LibHandle = HMODULE;
LibHandle open_lib() { return LoadLibraryA("nvcuda.dll"); }
void* sym(LibHandle h, const char* n) { return reinterpret_cast<void*>(GetProcAddress(h, n)); }
void close_lib(LibHandle h) { FreeLibrary(h); }
#else
using LibHandle = void*;
LibHandle open_lib() { return dlopen("libcuda.so.1", RTLD_NOW); }
void* sym(LibHandle h, const char* n) { return dlsym(h, n); }
void close_lib(LibHandle h) { dlclose(h); }
#endif
}  // namespace

struct CudaDriver::Impl {
    LibHandle lib = nullptr;
    std::string reason;
    cuda::Api api;
    cuda::CUdevice dev = 0;
    cuda::CUcontext ctx = nullptr;
    int cc_major = 0, cc_minor = 0;
    char dev_name[128] = {0};

    cuda::CUresult (*cuDeviceGet)(cuda::CUdevice*, int) = nullptr;
    cuda::CUresult (*cuDeviceGetName)(char*, int, cuda::CUdevice) = nullptr;
    cuda::CUresult (*cuDeviceGetAttribute)(int*, int, cuda::CUdevice) = nullptr;
    cuda::CUresult (*cuDevicePrimaryCtxRetain)(cuda::CUcontext*, cuda::CUdevice) = nullptr;
    cuda::CUresult (*cuDevicePrimaryCtxRelease)(cuda::CUdevice) = nullptr;
    cuda::CUresult (*cuCtxSetCurrent)(cuda::CUcontext) = nullptr;
};

CudaDriver::CudaDriver() : impl_(std::make_unique<Impl>()) {
    Impl& d = *impl_;
    cuda::Api& a = d.api;
    d.lib = open_lib();
    if (!d.lib) { d.reason = "CUDA driver library (nvcuda.dll / libcuda.so.1) not found"; return; }

    auto load = [&](auto& fn, const char* name) {
        fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(sym(d.lib, name));
        return fn != nullptr;
    };
    bool ok = load(a.cuInit, "cuInit") && load(a.cuGetErrorString, "cuGetErrorString") &&
              load(a.cuLinkCreate, "cuLinkCreate_v2") && load(a.cuLinkAddData, "cuLinkAddData_v2") &&
              load(a.cuLinkComplete, "cuLinkComplete") && load(a.cuLinkDestroy, "cuLinkDestroy") &&
              load(a.cuModuleLoadDataEx, "cuModuleLoadDataEx") && load(a.cuModuleUnload, "cuModuleUnload") &&
              load(a.cuModuleGetFunction, "cuModuleGetFunction") && load(a.cuFuncGetAttribute, "cuFuncGetAttribute") &&
              load(a.cuMemAlloc, "cuMemAlloc_v2") && load(a.cuMemFree, "cuMemFree_v2") &&
              load(a.cuMemcpyHtoD, "cuMemcpyHtoD_v2") && load(a.cuMemcpyDtoH, "cuMemcpyDtoH_v2") &&
              load(a.cuLaunchKernel, "cuLaunchKernel") && load(a.cuCtxSynchronize, "cuCtxSynchronize") &&
              load(d.cuDeviceGet, "cuDeviceGet") && load(d.cuDeviceGetName, "cuDeviceGetName") &&
              load(d.cuDeviceGetAttribute, "cuDeviceGetAttribute") &&
              load(d.cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain") &&
              load(d.cuDevicePrimaryCtxRelease, "cuDevicePrimaryCtxRelease_v2") &&
              load(d.cuCtxSetCurrent, "cuCtxSetCurrent");
    if (!ok) { d.reason = "CUDA driver library is missing required entry points"; return; }

    if (auto r = a.cuInit(0)) { d.reason = "cuInit failed: " + error_string(r); return; }
    if (auto r = d.cuDeviceGet(&d.dev, 0)) { d.reason = "no CUDA device: " + error_string(r); return; }
    d.cuDeviceGetName(d.dev_name, sizeof(d.dev_name), d.dev);
    d.cuDeviceGetAttribute(&d.cc_major, kAttrCcMajor, d.dev);
    d.cuDeviceGetAttribute(&d.cc_minor, kAttrCcMinor, d.dev);
    if (auto r = d.cuDevicePrimaryCtxRetain(&d.ctx, d.dev)) {
        d.reason = "context creation failed: " + error_string(r);
        return;
    }
    d.cuCtxSetCurrent(d.ctx);
}

CudaDriver::~CudaDriver() {
    if (impl_->ctx && impl_->cuDevicePrimaryCtxRelease) impl_->cuDevicePrimaryCtxRelease(impl_->dev);
    if (impl_->lib) close_lib(impl_->lib);
}

bool CudaDriver::available() const { return impl_->ctx != nullptr; }
const std::string& CudaDriver::unavailable_reason() const { return impl_->reason; }
std::string CudaDriver::device_name() const { return impl_->dev_name; }
std::string CudaDriver::device_arch() const {
    return "sm_" + std::to_string(impl_->cc_major) + std::to_string(impl_->cc_minor);
}
const cuda::Api& CudaDriver::api() const { return impl_->api; }
void CudaDriver::make_current() const {
    if (impl_->ctx) impl_->cuCtxSetCurrent(impl_->ctx);
}

std::string CudaDriver::error_string(cuda::CUresult r) const {
    const char* s = nullptr;
    if (impl_->api.cuGetErrorString && impl_->api.cuGetErrorString(r, &s) == 0 && s) return s;
    return "CUDA error " + std::to_string(r);
}

}  // namespace ptxforge::backend
