#ifndef KERNEL_LAUNCHER_PROBLEM_KEY_H
#define KERNEL_LAUNCHER_PROBLEM_KEY_H

#include <array>
#include <vector>

#include "kernel_launcher/primitive.hpp"
#include "kernel_launcher/utils.hpp"

namespace kernel_launcher {

/**
 * @addtogroup wisdom
 * @{
 */

// Maximum number of dimensions of a problem size (x, y, z).
static constexpr size_t MAX_PROBLEM_DIMS = 3;

class FeatureVector {
  public:
    using const_iterator = std::vector<std::pair<Symbol, Primitive>>::const_iterator;

    const_iterator begin() const {
        return features_.begin();
    }

    const_iterator end() const {
        return features_.end();
    }

    void add(std::string_view key, Primitive values);
    Primitive get(std::string_view key) const;
    size_t hash() const;

    bool operator==(const FeatureVector& that) const;

    bool operator!=(const FeatureVector& that) const {
        return !(*this == that);
    }

  private:
    std::vector<std::pair<Symbol, Primitive>> features_;
};

struct ProblemKey {
    Symbol tuning_key;
    std::array<int64_t, MAX_PROBLEM_DIMS> problem_size;
    FeatureVector features;

    ProblemKey(Symbol tuning_key, int64_t x = 1, int64_t y = 1, int64_t z = 1) :
        tuning_key(tuning_key),
        problem_size {x, y, z} {}

    ProblemKey(std::string_view tuning_key, int64_t x = 1, int64_t y = 1, int64_t z = 1) :
        ProblemKey(Symbol(tuning_key), x, y, z) {}

    size_t hash() const;
    bool operator==(const ProblemKey& that) const;

    bool operator!=(const ProblemKey& that) const {
        return !(*this == that);
    }
};

/** @} */

}  // namespace kernel_launcher

namespace std {
template<>
struct hash<kernel_launcher::FeatureVector> {
    size_t operator()(const kernel_launcher::FeatureVector& v) const noexcept {
        return v.hash();
    }
};

template<>
struct hash<kernel_launcher::ProblemKey> {
    size_t operator()(const kernel_launcher::ProblemKey& v) const noexcept {
        return v.hash();
    }
};
}  // namespace std

#endif
