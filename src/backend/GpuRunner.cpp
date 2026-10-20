#include "backend/GpuRunner.hpp"

#include <cstring>

namespace ptxforge::backend {

std::vector<uint8_t> fill_pattern(uint32_t seed, size_t bytes) {
    std::vector<uint8_t> out(bytes, 0);
    uint32_t x = 0x9E3779B9u ^ (seed * 2654435761u);
    for (size_t i = 0; i + 4 <= bytes; i += 4) {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        float f = static_cast<float>(x >> 8) / 8388608.0f - 1.0f;
        std::memcpy(out.data() + i, &f, 4);
    }
    return out;
}

RunResult GpuRunner::run(std::string_view ptx, const std::string& kernel, const std::vector<KernelArg>& args,
                         const LaunchConfig& cfg, uint32_t max_regs) {
    using namespace cuda;
    RunResult res;
    if (!driver_.available()) { res.error = driver_.unavailable_reason(); return res; }
    const Api& api = driver_.api();
    driver_.make_current();

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
        for (auto p : dev) if (p) api.cuMemFree(p);
        if (mod) api.cuModuleUnload(mod);
    };
    auto fail = [&](const std::string& what, CUresult r) {
        res.error = what + ": " + driver_.error_string(r) + (error[0] ? std::string("\n") + error.data() : "");
        cleanup();
        return res;
    };

    CUresult r = api.cuModuleLoadDataEx(&mod, text.c_str(), static_cast<unsigned>(o.size()), o.data(), v.data());
    if (r) { mod = nullptr; return fail("cuModuleLoadDataEx", r); }
    CUfunction fn = nullptr;
    if ((r = api.cuModuleGetFunction(&fn, mod, kernel.c_str()))) return fail("cuModuleGetFunction(" + kernel + ")", r);
    api.cuFuncGetAttribute(&res.registers, kFuncNumRegs, fn);
    api.cuFuncGetAttribute(&res.local_bytes, kFuncLocalBytes, fn);
    api.cuFuncGetAttribute(&res.shared_bytes, kFuncSharedBytes, fn);

    std::vector<uint64_t> slots(args.size(), 0);
    std::vector<void*> params(args.size());
    for (size_t i = 0; i < args.size(); ++i) {
        const KernelArg& a = args[i];
        if (a.kind == KernelArg::Kind::Buffer) {
            if ((r = api.cuMemAlloc(&dev[i], a.bytes))) { dev[i] = 0; return fail("cuMemAlloc", r); }
            auto init = fill_pattern(a.seed, a.bytes);
            if ((r = api.cuMemcpyHtoD(dev[i], init.data(), a.bytes))) return fail("cuMemcpyHtoD", r);
            slots[i] = dev[i];
        } else {
            slots[i] = a.value;  // little-endian: a u32/f32 parameter reads the low bytes
        }
        params[i] = &slots[i];
    }

    r = api.cuLaunchKernel(fn, cfg.grid[0], cfg.grid[1], cfg.grid[2], cfg.block[0], cfg.block[1], cfg.block[2],
                           cfg.dynamic_shared, nullptr, params.data(), nullptr);
    if (r) return fail("cuLaunchKernel", r);
    if ((r = api.cuCtxSynchronize())) return fail("kernel execution", r);

    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i].kind != KernelArg::Kind::Buffer) continue;
        res.buffers.emplace_back(args[i].bytes);
        if ((r = api.cuMemcpyDtoH(res.buffers.back().data(), dev[i], args[i].bytes))) return fail("cuMemcpyDtoH", r);
    }
    cleanup();
    res.ok = true;
    return res;
}

}  // namespace ptxforge::backend
