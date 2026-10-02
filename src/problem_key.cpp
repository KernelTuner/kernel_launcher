#include "kernel_launcher/problem_key.hpp"

namespace kernel_launcher {

void FeatureVector::add(std::string_view key, Primitive values) {
    for (auto& [k, v] : features_) {
        if (k.get() == key) {
            v = std::move(values);
            return;
        }
    }

    features_.emplace_back(Symbol(key), std::move(values));
}

Primitive FeatureVector::get(std::string_view key) const {
    for (const auto& [k, v] : features_) {
        if (k.get() == key) {
            return v;
        }
    }

    throw std::runtime_error(concat("FeatureVector: unknown feature '", key, "'"));
}

bool FeatureVector::operator==(const FeatureVector& that) const {
    if (features_.size() != that.features_.size()) {
        return false;
    }

    for (const auto& [key, value] : features_) {
        bool found = false;

        for (const auto& [k, v] : that.features_) {
            if (k == key) {
                found = v == value;
                break;
            }
        }

        if (!found) {
            return false;
        }
    }

    return true;
}

size_t FeatureVector::hash() const {
    size_t seed = 0;

    for (const auto& [key, value] : features_) {
        size_t h = 0;
        hash_combine(h, key);
        hash_combine(h, value);
        seed ^= h;
    }

    return seed;
}

bool ProblemKey::operator==(const ProblemKey& that) const {
    return tuning_key == that.tuning_key && problem_size == that.problem_size
        && features == that.features;
}

size_t ProblemKey::hash() const {
    size_t seed = 0;
    hash_combine(seed, tuning_key);

    for (int64_t dim : problem_size) {
        hash_combine(seed, dim);
    }

    hash_combine(seed, features);
    return seed;
}

}  // namespace kernel_launcher
