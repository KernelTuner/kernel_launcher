#ifndef KERNEL_LAUNCHER_UTILS_H
#define KERNEL_LAUNCHER_UTILS_H

#include <cstddef>
#include <functional>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <typeinfo>
#include <vector>

namespace kernel_launcher {

/**
 * @addtogroup types
 * @{
 */

/**
 * Returns a reference to an interned copy of `data` that remains valid for
 * the lifetime of the program. Equal strings return the same object. This
 * function is thread-safe.
 *
 * This is intended for short strings that are passed around a lot (e.g.,
 * variable names or function names).
 */
const std::string& intern_string(std::string_view data);

/**
 * An interned string (see `intern_string`). A `Symbol` is a single pointer,
 * so copying is cheap, and equality and hashing compare only that pointer.
 */
class Symbol {
  public:
    constexpr Symbol() = default;
    // Explicit, since interning a string keeps it alive for the lifetime of the program.
    explicit Symbol(std::string_view data) : ptr_(data.empty() ? nullptr : &intern_string(data)) {}

    bool empty() const {
        return ptr_ == nullptr;
    }

    const char* c_str() const {
        return get().c_str();
    }

    size_t size() const {
        return get().size();
    }

    const std::string& get() const {
        static const std::string empty_string;
        return ptr_ != nullptr ? *ptr_ : empty_string;
    }

    operator std::string_view() const {
        return get();
    }

    bool operator==(const Symbol& that) const {
        return ptr_ == that.ptr_;
    }

    bool operator!=(const Symbol& that) const {
        return !(*this == that);
    }

    bool operator<(const Symbol& that) const {
        return get() < that.get();
    }

    size_t hash() const noexcept {
        // Strings are interned, so equal strings share the same object.
        return std::hash<const std::string*> {}(ptr_);
    }

  private:
    // `nullptr` represents the empty string to allow a `constexpr` default constructor.
    const std::string* ptr_ = nullptr;
};

inline std::ostream& operator<<(std::ostream& stream, const Symbol& symbol) {
    return stream << symbol.get();
}

/** @} */

/**
 * @addtogroup utils
 * @{
 */

/**
 * Takes a `std::type_info` and returns the human-readable name of the type
 * as it appears in source code. Throws `std::runtime_error` if demangling
 * fails or is not supported on this platform.
 */
std::string demangle_type_info(const std::type_info& type);

/**
 * Combines the hash of the given value with the given seed value.
 *
 * This is based on the `hash_combine` function from Boost, as implemented up
 * to Boost 1.55, but uses the 64-bit golden ratio constant so that the upper
 * bits of the seed are also mixed. See:
 * https://github.com/boostorg/functional/blob/boost-1.55.0/include/boost/functional/hash/hash.hpp
 */
template<typename T>
void hash_combine(size_t& seed, const T& value) {
    size_t v = std::hash<T> {}(value);
    seed ^= v + size_t(0x9e3779b97f4a7c15ULL) + (seed << 6) + (seed >> 2);
}

/**
 * Combines the hashes of the given values with the given seed value. The
 * size of the vector is also included, so that e.g. `{1}` and `{1, 0}`
 * hash differently.
 */
template<typename T>
void hash_combine(size_t& seed, const std::vector<T>& values) {
    hash_combine(seed, values.size());

    for (const auto& value : values) {
        hash_combine(seed, value);
    }
}

/**
 * Combines the hashes of multiple fields into a single hash value. Useful
 * for structs: `return hash_fields(name_, value_, date_, ...);`.
 */
template<typename... Ts>
size_t hash_fields(const Ts&... values) {
    size_t seed = 0;
    (hash_combine(seed, values), ...);
    return seed;
}

/**
 * Concatenates the given values into a single string. Useful for cases such
 * as `concat("error: value ", x, " is not convertible to ", type)`.
 *
 * Each value is written to a `std::ostringstream` using `operator<<`.
 */
template<typename... Ts>
std::string concat(const Ts&... values) {
    std::ostringstream stream;
    (stream << ... << values);
    return stream.str();
}

/**
 * Returns `true` if messages written to `log_debug()` are printed.
 *
 * The minimum log level is set by the environment variable
 * `KERNEL_LAUNCHER_LOG`: DEBUG, INFO, or WARN (lowercase variants and WARNING
 * are also accepted; default is INFO). An invalid value falls back to DEBUG
 * and logs a warning.
 *
 * The arguments of `log_debug() << ...` are evaluated even if the message is
 * discarded, so check this function first if they are expensive to compute.
 */
bool log_debug_enabled();

/**
 * Returns `true` if messages written to `log_info()` are printed. See
 * `log_debug_enabled`.
 */
bool log_info_enabled();

/**
 * Returns `true` if messages written to `log_warning()` are printed. See
 * `log_debug_enabled`.
 */
bool log_warning_enabled();

/**
 * Returns the stream for debug messages. This writes to stderr, or discards
 * the message if `log_debug_enabled()` is `false`.
 */
std::ostream& log_debug();

/**
 * Returns the stream for info messages. This writes to stderr, or discards
 * the message if `log_info_enabled()` is `false`.
 */
std::ostream& log_info();

/**
 * Returns the stream for warning messages. This writes to stderr, or discards
 * the message if `log_warning_enabled()` is `false`.
 */
std::ostream& log_warning();

/** @} */

}  // namespace kernel_launcher

namespace std {
template<>
struct hash<kernel_launcher::Symbol> {
    size_t operator()(const kernel_launcher::Symbol& v) const noexcept {
        return v.hash();
    }
};
}  // namespace std

#endif
