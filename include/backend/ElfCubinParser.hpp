#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ptxforge::backend {

struct ElfError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Minimal ELF64 structures (no dependency on <elf.h>, so this builds on Windows too).
struct ElfSection {
    std::string name;
    uint32_t type = 0;
    uint64_t flags = 0;
    uint64_t addr = 0;
    uint64_t offset = 0;
    uint64_t size = 0;
    uint32_t link = 0;
    uint32_t info = 0;
    uint64_t addralign = 0;
    uint64_t entsize = 0;
};

struct ElfSymbol {
    std::string name;
    uint64_t value = 0;
    uint64_t size = 0;
    uint8_t type = 0;     // STT_*
    uint8_t binding = 0;  // STB_*
    uint16_t shndx = 0;
};

// One record of a .nv.info / .nv.info.<kernel> section.
struct NvInfoAttr {
    uint8_t format = 0;  // 1=NVAL 2=BVAL 3=HVAL 4=SVAL
    uint8_t attr = 0;    // EIATTR_*
    std::vector<uint8_t> payload;
    std::string name() const;  // EIATTR_REGCOUNT, or EIATTR_0xNN when unknown
    uint32_t u32(size_t word) const;  // payload word, 0 if out of range
};

struct KernelInfo {
    std::string name;
    uint64_t text_bytes = 0;
    uint64_t sass_instructions = 0;
    bool has_reg_count = false;      // EIATTR_REGCOUNT present (older toolchains)
    uint32_t reg_count = 0;          // EIATTR_REGCOUNT
    // Best available physical register count: the attribute if present, else the sh_info byte.
    uint32_t registers() const { return has_reg_count ? reg_count : reg_count_sh_info; }
    uint32_t reg_count_sh_info = 0;  // sh_info >> 24 of .text.<kernel>: where current ptxas stores the count
    uint32_t frame_size = 0;         // EIATTR_FRAME_SIZE (local-memory stack frame)
    uint32_t min_stack_size = 0;     // EIATTR_MIN_STACK_SIZE
    uint64_t shared_bytes = 0;       // .nv.shared.<kernel>
    uint64_t constant_bytes = 0;     // .nv.constant0.<kernel> (params + driver constants)
    uint64_t local_bytes = 0;        // .nv.local.<kernel>
    std::vector<NvInfoAttr> attrs;
};

struct CubinInfo {
    uint8_t ei_class = 0;
    uint8_t ei_data = 0;
    uint8_t os_abi = 0;
    uint8_t abi_version = 0;
    uint16_t type = 0;
    uint16_t machine = 0;
    uint32_t flags = 0;
    uint32_t sm_arch = 0;  // decoded from e_flags (e.g. 86 for sm_86)
    std::vector<ElfSection> sections;
    std::vector<ElfSymbol> symbols;
    std::vector<KernelInfo> kernels;
    std::vector<NvInfoAttr> global_attrs;  // records of the global .nv.info section

    const KernelInfo* find_kernel(const std::string& name) const;
    const ElfSection* find_section(const std::string& name) const;
};

constexpr uint16_t kEmCuda = 0xBE;
constexpr uint8_t kElfOsAbiCuda = 0x33;    // ELFOSABI_CUDA (older toolchains)
constexpr uint8_t kElfOsAbiCudaV2 = 0x41;  // ELFOSABI_CUDA_V2 (current toolchains)

// Parses a cubin (ELF64 little-endian, e_machine = EM_CUDA) from memory. Throws ElfError.
CubinInfo parse_cubin(const std::vector<uint8_t>& image);
CubinInfo parse_cubin_file(const std::string& path);

}  // namespace ptxforge::backend
