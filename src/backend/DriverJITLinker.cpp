#include "backend/DriverJITLinker.hpp"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace ptxforge::backend {

namespace {

// --- Minimal subset of cuda.h, so no CUDA SDK is needed to build. Values verified against cuda.h. ---
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
constexpr int kFuncMaxThreads = 0;
constexpr int kFuncSharedBytes = 1;
constexpr int kFuncLocalBytes = 3;
constexpr int kFuncNumRegs = 4;
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

struct DriverJITLinker::Impl {
    LibHandle lib = nullptr;
    std::string reason;
    CUdevice dev = 0;
    CUcontext ctx = nullptr;
    int cc_major = 0, cc_minor = 0;
    char dev_name[128] = {0};

    CUresult (*cuInit)(unsigned) = nullptr;
    CUresult (*cuDeviceGet)(CUdevice*, int) = nullptr;
    CUresult (*cuDeviceGetName)(char*, int, CUdevice) = nullptr;
    CUresult (*cuDeviceGetAttribute)(int*, int, CUdevice) = nullptr;
    CUresult (*cuDevicePrimaryCtxRetain)(CUcontext*, CUdevice) = nullptr;
    CUresult (*cuDevicePrimaryCtxRelease)(CUdevice) = nullptr;
    CUresult (*cuCtxSetCurrent)(CUcontext) = nullptr;
    CUresult (*cuLinkCreate)(unsigned, CUjit_option*, void**, CUlinkState*) = nullptr;
    CUresult (*cuLinkAddData)(CUlinkState, int, void*, size_t, const char*, unsigned, CUjit_option*, void**) = nullptr;
    CUresult (*cuLinkComplete)(CUlinkState, void**, size_t*) = nullptr;
    CUresult (*cuLinkDestroy)(CUlinkState) = nullptr;
    CUresult (*cuGetErrorString)(CUresult, const char**) = nullptr;
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

    std::string err(CUresult r) const {
        const char* s = nullptr;
        if (cuGetErrorString && cuGetErrorString(r, &s) == 0 && s) return s;
        return "CUDA error " + std::to_string(r);
    }
};

DriverJITLinker::DriverJITLinker() : impl_(std::make_unique<Impl>()) {
    Impl& d = *impl_;
    d.lib = open_lib();
    if (!d.lib) { d.reason = "CUDA driver library (nvcuda.dll / libcuda.so.1) not found"; return; }
    auto load = [&](auto& fn, const char* name) {
        fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(sym(d.lib, name));
        return fn != nullptr;
    };
    bool ok = load(d.cuInit, "cuInit") && load(d.cuDeviceGet, "cuDeviceGet") &&
              load(d.cuDeviceGetName, "cuDeviceGetName") && load(d.cuDeviceGetAttribute, "cuDeviceGetAttribute") &&
              load(d.cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain") &&
              load(d.cuDevicePrimaryCtxRelease, "cuDevicePrimaryCtxRelease_v2") &&
              load(d.cuCtxSetCurrent, "cuCtxSetCurrent") && load(d.cuLinkCreate, "cuLinkCreate_v2") &&
              load(d.cuLinkAddData, "cuLinkAddData_v2") && load(d.cuLinkComplete, "cuLinkComplete") &&
              load(d.cuLinkDestroy, "cuLinkDestroy") && load(d.cuGetErrorString, "cuGetErrorString") &&
              load(d.cuModuleLoadDataEx, "cuModuleLoadDataEx") && load(d.cuModuleUnload, "cuModuleUnload") &&
              load(d.cuModuleGetFunction, "cuModuleGetFunction") && load(d.cuFuncGetAttribute, "cuFuncGetAttribute") &&
              load(d.cuMemAlloc, "cuMemAlloc_v2") && load(d.cuMemFree, "cuMemFree_v2") &&
              load(d.cuMemcpyHtoD, "cuMemcpyHtoD_v2") && load(d.cuMemcpyDtoH, "cuMemcpyDtoH_v2") &&
              load(d.cuLaunchKernel, "cuLaunchKernel") && load(d.cuCtxSynchronize, "cuCtxSynchronize");
    if (!ok) { d.reason = "CUDA driver library is missing required cuLink* entry points"; return; }
    if (CUresult r = d.cuInit(0)) { d.reason = "cuInit failed: " + d.err(r); return; }
    if (CUresult r = d.cuDeviceGet(&d.dev, 0)) { d.reason = "no CUDA device: " + d.err(r); return; }
    d.cuDeviceGetName(d.dev_name, sizeof(d.dev_name), d.dev);
    d.cuDeviceGetAttribute(&d.cc_major, kAttrCcMajor, d.dev);
    d.cuDeviceGetAttribute(&d.cc_minor, kAttrCcMinor, d.dev);
    if (CUresult r = d.cuDevicePrimaryCtxRetain(&d.ctx, d.dev)) { d.reason = "context creation failed: " + d.err(r); return; }
    d.cuCtxSetCurrent(d.ctx);
}

DriverJITLinker::~DriverJITLinker() {
    if (impl_->ctx && impl_->cuDevicePrimaryCtxRelease) impl_->cuDevicePrimaryCtxRelease(impl_->dev);
    if (impl_->lib) close_lib(impl_->lib);
}

bool DriverJITLinker::available() const { return impl_->ctx != nullptr; }
const std::string& DriverJITLinker::unavailable_reason() const { return impl_->reason; }
std::string DriverJITLinker::device_name() const { return impl_->dev_name; }
std::string DriverJITLinker::device_arch() const {
    return "sm_" + std::to_string(impl_->cc_major) + std::to_string(impl_->cc_minor);
}

CompileResult DriverJITLinker::compile(std::string_view ptx, const CompileOptions& opts) {
    CompileResult res;
    Impl& d = *impl_;
    if (!available()) { res.error_log = d.reason; return res; }
    d.cuCtxSetCurrent(d.ctx);

    std::vector<char> info(16384, 0), error(16384, 0);
    std::vector<CUjit_option> o = {kJitInfoLogBuffer, kJitInfoLogBufferSize, kJitErrorLogBuffer,
                                   kJitErrorLogBufferSize, kJitLogVerbose};
    std::vector<void*> v = {info.data(), reinterpret_cast<void*>(static_cast<uintptr_t>(info.size())),
                            error.data(), reinterpret_cast<void*>(static_cast<uintptr_t>(error.size())),
                            reinterpret_cast<void*>(static_cast<uintptr_t>(1))};
    if (opts.max_regs) {
        o.push_back(kJitMaxRegisters);
        v.push_back(reinterpret_cast<void*>(static_cast<uintptr_t>(opts.max_regs)));
    }
    if (opts.target_sm.rfind("sm_", 0) == 0) {  // cross-compile for a specific arch (CU_TARGET_COMPUTE_xx)
        o.push_back(kJitTarget);
        v.push_back(reinterpret_cast<void*>(static_cast<uintptr_t>(std::stoul(opts.target_sm.substr(3)))));
    }

    CUlinkState link = nullptr;
    CUresult r = d.cuLinkCreate(static_cast<unsigned>(o.size()), o.data(), v.data(), &link);
    if (r) { res.error_log = "cuLinkCreate: " + d.err(r); return res; }

    std::string text(ptx);  // cuLinkAddData wants a NUL-terminated PTX string; keep the NUL in the size
    r = d.cuLinkAddData(link, kJitInputPtx, text.data(), text.size() + 1, "kernel.ptx", 0, nullptr, nullptr);
    if (r) {
        res.error_log = std::string("cuLinkAddData: ") + d.err(r) + "\n" + error.data();
        d.cuLinkDestroy(link);
        return res;
    }
    void* cubin = nullptr;
    size_t size = 0;
    r = d.cuLinkComplete(link, &cubin, &size);
    if (r) {
        res.error_log = std::string("cuLinkComplete: ") + d.err(r) + "\n" + error.data();
        d.cuLinkDestroy(link);
        return res;
    }
    res.cubin.assign(static_cast<uint8_t*>(cubin), static_cast<uint8_t*>(cubin) + size);  // copy before destroy
    d.cuLinkDestroy(link);
    res.info_log = info.data();
    res.error_log = error.data();
    res.ptxas_stats = parse_ptxas_log(res.info_log);
    res.ok = true;
    return res;
}

std::vector<uint8_t> fill_pattern(uint32_t seed, size_t bytes) {
    std::vector<uint8_t> out(bytes, 0);
    uint32_t x = 0x9E3779B9u ^ (seed * 2654435761u);
    for (size_t i = 0; i + 4 <= bytes; i += 4) {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;  // xorshift32
        float f = static_cast<float>(x >> 8) / 8388608.0f - 1.0f;  // [-1, 1)
        std::memcpy(out.data() + i, &f, 4);
    }
    return out;
}

RunResult DriverJITLinker::run(std::string_view ptx, const std::string& kernel, const std::vector<KernelArg>& args,
                               const LaunchConfig& cfg, uint32_t max_regs) {
    RunResult res;
    Impl& d = *impl_;
    if (!available()) { res.error = d.reason; return res; }
    d.cuCtxSetCurrent(d.ctx);

    std::vector<char> error(16384, 0);
    std::vector<CUjit_option> o = {kJitErrorLogBuffer, kJitErrorLogBufferSize};
    std::vector<void*> v = {error.data(), reinterpret_cast<void*>(static_cast<uintptr_t>(error.size()))};
    if (max_regs) {
        o.push_back(kJitMaxRegisters);
        v.push_back(reinterpret_cast<void*>(static_cast<uintptr_t>(max_regs)));
    }

    std::string text(ptx);
    CUmodule mod = nullptr;
    std::vector<CUdeviceptr> dev(args.size(), 0);
    auto cleanup = [&] {
        for (auto p : dev) if (p) d.cuMemFree(p);
        if (mod) d.cuModuleUnload(mod);
    };
    auto fail = [&](const std::string& what, CUresult r) {
        res.error = what + ": " + d.err(r) + (error[0] ? std::string("\n") + error.data() : "");
        cleanup();
        return res;
    };

    CUresult r = d.cuModuleLoadDataEx(&mod, text.c_str(), static_cast<unsigned>(o.size()), o.data(), v.data());
    if (r) { mod = nullptr; return fail("cuModuleLoadDataEx", r); }
    CUfunction fn = nullptr;
    if ((r = d.cuModuleGetFunction(&fn, mod, kernel.c_str()))) return fail("cuModuleGetFunction(" + kernel + ")", r);
    d.cuFuncGetAttribute(&res.registers, kFuncNumRegs, fn);
    d.cuFuncGetAttribute(&res.local_bytes, kFuncLocalBytes, fn);
    d.cuFuncGetAttribute(&res.shared_bytes, kFuncSharedBytes, fn);
    d.cuFuncGetAttribute(&res.max_threads, kFuncMaxThreads, fn);

    // Marshal arguments. Scalars live in `scalars`, buffers' device pointers in `dev`.
    std::vector<uint64_t> scalars(args.size(), 0);
    std::vector<void*> params(args.size());
    for (size_t i = 0; i < args.size(); ++i) {
        const KernelArg& a = args[i];
        if (a.kind == KernelArg::Kind::Buffer) {
            if ((r = d.cuMemAlloc(&dev[i], a.bytes))) { dev[i] = 0; return fail("cuMemAlloc", r); }
            auto init = fill_pattern(a.seed, a.bytes);
            if ((r = d.cuMemcpyHtoD(dev[i], init.data(), a.bytes))) return fail("cuMemcpyHtoD", r);
            scalars[i] = dev[i];
        } else {
            scalars[i] = a.value;  // little-endian: the low bytes are what a u32/f32 param reads
        }
        params[i] = &scalars[i];
    }

    r = d.cuLaunchKernel(fn, cfg.grid[0], cfg.grid[1], cfg.grid[2], cfg.block[0], cfg.block[1], cfg.block[2],
                         cfg.dynamic_shared, nullptr, params.data(), nullptr);
    if (r) return fail("cuLaunchKernel", r);
    if ((r = d.cuCtxSynchronize())) return fail("kernel execution", r);

    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i].kind != KernelArg::Kind::Buffer) continue;
        res.buffers.emplace_back(args[i].bytes);
        if ((r = d.cuMemcpyDtoH(res.buffers.back().data(), dev[i], args[i].bytes))) return fail("cuMemcpyDtoH", r);
    }
    cleanup();
    res.ok = true;
    return res;
}

PtxasProvider::PtxasProvider(std::string ptxas_path) : path_(std::move(ptxas_path)) {
    namespace fs = std::filesystem;
    if (!path_.empty()) return;
#ifdef _WIN32
    const char* exe = "ptxas.exe";
#else
    const char* exe = "ptxas";
#endif
    std::vector<fs::path> candidates;
    for (const char* var : {"CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"})
        if (const char* p = std::getenv(var)) candidates.push_back(fs::path(p) / "bin" / exe);
    if (const char* path = std::getenv("PATH")) {
        std::string s = path;
#ifdef _WIN32
        const char sep = ';';
#else
        const char sep = ':';
#endif
        size_t start = 0;
        while (start <= s.size()) {
            size_t e = s.find(sep, start);
            if (e == std::string::npos) e = s.size();
            if (e > start) candidates.push_back(fs::path(s.substr(start, e - start)) / exe);
            start = e + 1;
        }
    }
    for (auto& c : candidates) {
        std::error_code ec;
        if (fs::is_regular_file(c, ec)) { path_ = c.string(); return; }
    }
}

CompileResult PtxasProvider::compile(std::string_view ptx, const CompileOptions& opts) {
    namespace fs = std::filesystem;
    CompileResult res;
    if (path_.empty()) { res.error_log = "ptxas not found (set CUDA_PATH or pass --ptxas=<path>)"; return res; }
    fs::path dir = fs::temp_directory_path() / ("ptxforge_" + std::to_string(std::rand()));
    std::error_code ec;
    fs::create_directories(dir, ec);
    fs::path in = dir / "kernel.ptx", out = dir / "kernel.cubin", log = dir / "ptxas.log";
    { std::ofstream f(in, std::ios::binary); f.write(ptx.data(), static_cast<std::streamsize>(ptx.size())); }

    std::string arch = opts.target_sm.empty() ? "sm_75" : opts.target_sm;
    std::string cmd = "\"" + path_ + "\" -v -arch=" + arch;
    if (opts.max_regs) cmd += " --maxrregcount=" + std::to_string(opts.max_regs);
    cmd += " -o \"" + out.string() + "\" \"" + in.string() + "\" > \"" + log.string() + "\" 2>&1";
#ifdef _WIN32
    cmd = "\"" + cmd + "\"";  // cmd.exe /c strips the outer quotes
#endif
    int rc = std::system(cmd.c_str());
    std::ifstream lf(log);
    std::string text((std::istreambuf_iterator<char>(lf)), std::istreambuf_iterator<char>());
    lf.close();
    if (rc == 0) {
        std::ifstream cf(out, std::ios::binary);
        res.cubin.assign(std::istreambuf_iterator<char>(cf), std::istreambuf_iterator<char>());
        res.info_log = text;
        res.ptxas_stats = parse_ptxas_log(text);
        res.ok = !res.cubin.empty();
    } else {
        res.error_log = text;
    }
    fs::remove_all(dir, ec);
    return res;
}

}  // namespace ptxforge::backend
