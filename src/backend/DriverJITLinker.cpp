#include "backend/DriverJITLinker.hpp"

#include <vector>

namespace ptxforge::backend {

CompileResult DriverJITLinker::compile(std::string_view ptx, const CompileOptions& opts) {
    using namespace cuda;
    CompileResult res;
    if (!driver_.available()) { res.error_log = driver_.unavailable_reason(); return res; }
    const Api& api = driver_.api();
    driver_.make_current();

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
    if (opts.target_sm.rfind("sm_", 0) == 0) {
        o.push_back(kJitTarget);
        v.push_back(reinterpret_cast<void*>(static_cast<uintptr_t>(std::stoul(opts.target_sm.substr(3)))));
    }

    CUlinkState link = nullptr;
    CUresult r = api.cuLinkCreate(static_cast<unsigned>(o.size()), o.data(), v.data(), &link);
    if (r) { res.error_log = "cuLinkCreate: " + driver_.error_string(r); return res; }

    std::string text(ptx);  // cuLinkAddData wants a NUL-terminated PTX string, with the NUL counted in the size
    r = api.cuLinkAddData(link, kJitInputPtx, text.data(), text.size() + 1, "kernel.ptx", 0, nullptr, nullptr);
    if (r) {
        res.error_log = "cuLinkAddData: " + driver_.error_string(r) + "\n" + error.data();
        api.cuLinkDestroy(link);
        return res;
    }
    void* cubin = nullptr;
    size_t size = 0;
    r = api.cuLinkComplete(link, &cubin, &size);
    if (r) {
        res.error_log = "cuLinkComplete: " + driver_.error_string(r) + "\n" + error.data();
        api.cuLinkDestroy(link);
        return res;
    }
    res.cubin.assign(static_cast<uint8_t*>(cubin), static_cast<uint8_t*>(cubin) + size);  // copy before destroy
    api.cuLinkDestroy(link);
    res.info_log = info.data();
    res.error_log = error.data();
    res.ptxas_stats = parse_ptxas_log(res.info_log);
    res.ok = true;
    return res;
}

}  // namespace ptxforge::backend
