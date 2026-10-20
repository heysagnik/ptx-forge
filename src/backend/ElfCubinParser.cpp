#include "backend/ElfCubinParser.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>

namespace ptxforge::backend {

namespace {

constexpr uint8_t kEiClass64 = 2;
constexpr uint8_t kEiDataLsb = 1;
constexpr uint8_t kEiattrRegcount = 0x2f;
constexpr uint8_t kEiattrFrameSize = 0x11;
constexpr uint8_t kEiattrMinStackSize = 0x12;

template <class T>
T read(const std::vector<uint8_t>& buf, uint64_t off) {
    if (off > buf.size() || sizeof(T) > buf.size() - off) throw ElfError("ELF image truncated");
    T v;
    std::memcpy(&v, buf.data() + off, sizeof(T));  // host is little-endian; cubins are LSB
    return v;
}

std::string cstr_at(const std::vector<uint8_t>& buf, uint64_t base, uint64_t size, uint32_t off) {
    if (off >= size || base + size > buf.size()) return {};
    const char* p = reinterpret_cast<const char*>(buf.data() + base + off);
    size_t max = static_cast<size_t>(size - off);
    return std::string(p, strnlen(p, max));
}

bool starts_with(const std::string& s, const char* p) { return s.rfind(p, 0) == 0; }

// SVAL payload layout for these attributes: u32 symbol index, u32 value.
void apply_resource_attr(KernelInfo& k, const NvInfoAttr& a) {
    if (a.format != 4) return;
    if (a.attr == kEiattrRegcount) { k.has_reg_count = true; k.reg_count = a.u32(1); }
    else if (a.attr == kEiattrFrameSize) k.frame_size = a.u32(1);
    else if (a.attr == kEiattrMinStackSize) k.min_stack_size = a.u32(1);
}

}  // namespace

std::string NvInfoAttr::name() const {
    switch (attr) {
        case 0x01: return "EIATTR_PAD";
        case 0x04: return "EIATTR_CTAIDZ_USED";
        case 0x05: return "EIATTR_MAX_THREADS";
        case 0x0a: return "EIATTR_PARAM_CBANK";
        case 0x0f: return "EIATTR_EXTERNS";
        case 0x10: return "EIATTR_REQNTID";
        case 0x11: return "EIATTR_FRAME_SIZE";
        case 0x12: return "EIATTR_MIN_STACK_SIZE";
        case 0x17: return "EIATTR_KPARAM_INFO";
        case 0x19: return "EIATTR_CBANK_PARAM_SIZE";
        case 0x1b: return "EIATTR_MAXREG_COUNT";
        case 0x1c: return "EIATTR_EXIT_INSTR_OFFSETS";
        case 0x1e: return "EIATTR_CRS_STACK_SIZE";
        case 0x23: return "EIATTR_MAX_STACK_SIZE";
        case 0x2f: return "EIATTR_REGCOUNT";
        case 0x33: return "EIATTR_STATISTICS";
        case 0x37: return "EIATTR_CUDA_API_VERSION";
        default: {
            char b[24];
            std::snprintf(b, sizeof(b), "EIATTR_0x%02x", attr);
            return b;
        }
    }
}

uint32_t NvInfoAttr::u32(size_t word) const {
    if ((word + 1) * 4 > payload.size()) return 0;
    uint32_t v;
    std::memcpy(&v, payload.data() + word * 4, 4);
    return v;
}

const KernelInfo* CubinInfo::find_kernel(const std::string& n) const {
    for (auto& k : kernels) if (k.name == n) return &k;
    return nullptr;
}

const ElfSection* CubinInfo::find_section(const std::string& n) const {
    for (auto& s : sections) if (s.name == n) return &s;
    return nullptr;
}

static std::vector<NvInfoAttr> parse_nv_info(const std::vector<uint8_t>& img, const ElfSection& sec) {
    std::vector<NvInfoAttr> out;
    uint64_t pos = sec.offset, end = sec.offset + sec.size;
    if (end > img.size()) throw ElfError("section " + sec.name + " extends past end of file");
    while (pos + 2 <= end) {
        NvInfoAttr a;
        a.format = img[pos];
        a.attr = img[pos + 1];
        uint64_t len = 0;
        switch (a.format) {
            case 1: len = 2; break;
            case 2: len = 3; a.payload.assign(1, 0); break;
            case 3: len = 4; a.payload.assign(2, 0); break;
            case 4: {
                if (pos + 4 > end) throw ElfError("truncated SVAL record in " + sec.name);
                uint16_t sz = read<uint16_t>(img, pos + 2);
                a.payload.assign(sz, 0);
                len = 4 + sz;
                break;
            }
            default: return out;  // unknown encoding: stop rather than mis-parse
        }
        if (pos + len > end) throw ElfError("truncated record in " + sec.name);
        if (!a.payload.empty()) {
            uint64_t payload_off = a.format == 4 ? pos + 4 : pos + 2;
            std::memcpy(a.payload.data(), img.data() + payload_off, a.payload.size());
        }
        out.push_back(std::move(a));
        pos += len;
    }
    return out;
}

CubinInfo parse_cubin(const std::vector<uint8_t>& img) {
    if (img.size() < 64 || std::memcmp(img.data(), "\x7f" "ELF", 4) != 0) throw ElfError("not an ELF file");
    CubinInfo info;
    info.ei_class = img[4];
    info.ei_data = img[5];
    info.os_abi = img[7];
    info.abi_version = img[8];
    if (info.ei_class != kEiClass64) throw ElfError("only ELF64 cubins are supported");
    if (info.ei_data != kEiDataLsb) throw ElfError("only little-endian ELF is supported");

    info.type = read<uint16_t>(img, 16);
    info.machine = read<uint16_t>(img, 18);
    uint64_t shoff = read<uint64_t>(img, 40);
    info.flags = read<uint32_t>(img, 48);
    uint16_t shentsize = read<uint16_t>(img, 58);
    uint16_t shnum = read<uint16_t>(img, 60);
    uint16_t shstrndx = read<uint16_t>(img, 62);
    if (info.machine != kEmCuda) throw ElfError("not a CUDA ELF (e_machine != EM_CUDA)");
    // e_flags bits 8..15 hold the SM number (0x56 = sm_86); older cubins used the low byte.
    info.sm_arch = (info.flags >> 8) & 0xff ? (info.flags >> 8) & 0xff : info.flags & 0xff;
    if (shentsize < 64 || shoff == 0) throw ElfError("missing section header table");

    struct Raw { uint32_t name_off; ElfSection s; };
    std::vector<Raw> raw(shnum);
    for (uint16_t i = 0; i < shnum; ++i) {
        uint64_t o = shoff + static_cast<uint64_t>(i) * shentsize;
        raw[i].name_off = read<uint32_t>(img, o);
        ElfSection& s = raw[i].s;
        s.type = read<uint32_t>(img, o + 4);
        s.flags = read<uint64_t>(img, o + 8);
        s.addr = read<uint64_t>(img, o + 16);
        s.offset = read<uint64_t>(img, o + 24);
        s.size = read<uint64_t>(img, o + 32);
        s.link = read<uint32_t>(img, o + 40);
        s.info = read<uint32_t>(img, o + 44);
        s.addralign = read<uint64_t>(img, o + 48);
        s.entsize = read<uint64_t>(img, o + 56);
    }
    if (shstrndx >= shnum) throw ElfError("bad e_shstrndx");
    const ElfSection& shstr = raw[shstrndx].s;
    for (auto& r : raw) {
        r.s.name = cstr_at(img, shstr.offset, shstr.size, r.name_off);
        info.sections.push_back(r.s);
    }

    for (auto& s : info.sections) {
        if (s.name != ".symtab" || s.entsize < 24 || s.link >= info.sections.size()) continue;
        const ElfSection& str = info.sections[s.link];
        for (uint64_t o = 0; o + 24 <= s.size; o += s.entsize) {
            ElfSymbol sym;
            uint32_t name_off = read<uint32_t>(img, s.offset + o);
            uint8_t st_info = img[s.offset + o + 4];
            sym.type = st_info & 0xf;
            sym.binding = st_info >> 4;
            sym.shndx = read<uint16_t>(img, s.offset + o + 6);
            sym.value = read<uint64_t>(img, s.offset + o + 8);
            sym.size = read<uint64_t>(img, s.offset + o + 16);
            sym.name = cstr_at(img, str.offset, str.size, name_off);
            info.symbols.push_back(std::move(sym));
        }
    }

    for (auto& s : info.sections) {
        if (!starts_with(s.name, ".text.")) continue;
        KernelInfo k;
        k.name = s.name.substr(6);
        k.text_bytes = s.size;
        k.sass_instructions = s.size / 16;  // 128-bit SASS encodings (sm_70+)
        k.reg_count_sh_info = s.info >> 24;
        if (const ElfSection* ni = info.find_section(".nv.info." + k.name)) {
            k.attrs = parse_nv_info(img, *ni);
            for (auto& a : k.attrs) apply_resource_attr(k, a);
        }
        if (auto* sh = info.find_section(".nv.shared." + k.name)) k.shared_bytes = sh->size;
        if (auto* c = info.find_section(".nv.constant0." + k.name)) k.constant_bytes = c->size;
        info.kernels.push_back(std::move(k));
    }

    // Current toolchains store REGCOUNT / FRAME_SIZE / MIN_STACK_SIZE for every function in the global
    // .nv.info section, each record keyed by the function's symbol index.
    if (const ElfSection* gi = info.find_section(".nv.info")) {
        info.global_attrs = parse_nv_info(img, *gi);
        for (auto& a : info.global_attrs) {
            if (a.format != 4 || a.payload.size() < 8) continue;
            uint32_t sym = a.u32(0);
            if (sym >= info.symbols.size()) continue;
            for (auto& k : info.kernels)
                if (k.name == info.symbols[sym].name) apply_resource_attr(k, a);
        }
    }
    return info;
}

CubinInfo parse_cubin_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw ElfError("cannot open " + path);
    std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return parse_cubin(buf);
}

}  // namespace ptxforge::backend
