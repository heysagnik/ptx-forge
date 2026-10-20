#pragma once
#include <bit>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ptxforge {

// Dense, fixed-capacity bit set used for liveness sets and interference rows.
class BitSet {
public:
    BitSet() = default;
    explicit BitSet(size_t n) : words_((n + 63) / 64, 0), size_(n) {}

    size_t size() const { return size_; }
    bool test(size_t i) const { return (words_[i >> 6] >> (i & 63)) & 1ull; }
    void set(size_t i) { words_[i >> 6] |= 1ull << (i & 63); }
    void reset(size_t i) { words_[i >> 6] &= ~(1ull << (i & 63)); }
    void clear() { for (auto& w : words_) w = 0; }

    // this |= o; returns true if any bit changed.
    bool union_with(const BitSet& o) {
        bool changed = false;
        for (size_t i = 0; i < words_.size(); ++i) {
            uint64_t n = words_[i] | o.words_[i];
            changed |= (n != words_[i]);
            words_[i] = n;
        }
        return changed;
    }
    // this &= ~o
    void subtract(const BitSet& o) {
        for (size_t i = 0; i < words_.size(); ++i) words_[i] &= ~o.words_[i];
    }
    void intersect_with(const BitSet& o) {
        for (size_t i = 0; i < words_.size(); ++i) words_[i] &= o.words_[i];
    }
    size_t count() const {
        size_t c = 0;
        for (auto w : words_) c += static_cast<size_t>(std::popcount(w));
        return c;
    }
    bool any() const {
        for (auto w : words_) if (w) return true;
        return false;
    }
    template <class F>
    void for_each(F&& f) const {
        for (size_t wi = 0; wi < words_.size(); ++wi) {
            uint64_t w = words_[wi];
            while (w) {
                f(wi * 64 + static_cast<size_t>(std::countr_zero(w)));
                w &= w - 1;
            }
        }
    }
    bool operator==(const BitSet& o) const { return size_ == o.size_ && words_ == o.words_; }

private:
    std::vector<uint64_t> words_;
    size_t size_ = 0;
};

}  // namespace ptxforge
