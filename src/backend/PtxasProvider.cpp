#include "backend/PtxasProvider.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>

namespace ptxforge::backend {

namespace fs = std::filesystem;

PtxasProvider::PtxasProvider(std::string ptxas_path) : path_(std::move(ptxas_path)) {
    if (!path_.empty()) return;
#ifdef _WIN32
    const char* exe = "ptxas.exe";
    const char sep = ';';
#else
    const char* exe = "ptxas";
    const char sep = ':';
#endif
    std::vector<fs::path> candidates;
    for (const char* var : {"CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"})
        if (const char* p = std::getenv(var)) candidates.push_back(fs::path(p) / "bin" / exe);
    if (const char* path = std::getenv("PATH")) {
        std::string s = path;
        for (size_t start = 0; start <= s.size();) {
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
