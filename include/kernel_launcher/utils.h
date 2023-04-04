#ifndef KERNEL_LAUNCHER_UTILS_H
#define KERNEL_LAUNCHER_UTILS_H

#include <cuda_runtime_api.h>

#include <functional>
#include <iosfwd>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <typeindex>
#include <vector>

namespace kernel_launcher {

bool log_debug_enabled();
bool log_info_enabled();
bool log_warning_enabled();

std::ostream& log_debug();
std::ostream& log_info();
std::ostream& log_warning();

std::string demangle_type_info(const std::type_info& type);

namespace detail {
template<typename T>
inline const std::string& demangle_type_info_for() {
    static std::string result = demangle_type_info(typeid(T));
    return result;
}

struct TypeInfoInternalImpl {
    size_t alignment;
    size_t size;
    const std::type_info& type_info;
    const std::string& (*name_fun)();
    const TypeInfoInternalImpl* remove_pointer_type;
    const TypeInfoInternalImpl* remove_const;
    const TypeInfoInternalImpl* add_const;
    bool is_const;
    bool is_empty;
    bool is_trivial_copy;
};

template<typename T>
static constexpr TypeInfoInternalImpl type_impl_for = {
    alignof(T),
    sizeof(T),
    typeid(T),
    demangle_type_info_for<T>,
    &type_impl_for<typename std::remove_pointer<T>::type>,
    &type_impl_for<typename std::remove_const<T>::type>,
    &type_impl_for<typename std::add_const<T>::type>,
    std::is_const<T>::value,
    std::is_empty<T>::value,
    std::is_trivially_copyable<T>::value,
};
}  // namespace detail

/**
 * Runtime specification of a compile time type. Use the function `type_of<T>()`
 * to obtain an instance of `TypeInfo` for some type `T`.
 *
 * For example, `type_of<int>().name()` should return `"int"` and
 * `type_of<float*>().is_pointer()` should return `true`.
 */
struct TypeInfo {
  private:
    using Impl = detail::TypeInfoInternalImpl;

    constexpr TypeInfo(const Impl* impl) : impl_(impl) {}

  public:
    TypeInfo() : TypeInfo(nullptr) {}

    /**
     * Returns the `TypeInfo` for type `T`.
     */
    template<typename T>
    static constexpr TypeInfo of() {
        return &detail::type_impl_for<T>;
    }

    /**
     * Canonical name of this type in C++.
     */
    const std::string& name() const {
        return (impl_->name_fun)();
    }

    /**
     * Storage size of this type in bytes.
     */
    constexpr size_t size() const {
        return impl_->size;
    }

    /**
     * Alignment of this type in bytes.
     */
    constexpr size_t alignment() const {
        return impl_->alignment;
    }

    /**
     * Is this type a pointer.
     */
    constexpr bool is_pointer() const {
        return impl_->remove_pointer_type != impl_;
    }

    /**
     * Is this type `const`. For example, `const int` and `float* const` are
     * constant, but `int` and `const float*` are not constant.
     */
    constexpr bool is_const() const {
        return impl_->is_const;
    }

    /**
     * Check if this type is empty, i.e., a class with no non-static fields.
     * Similar to `std::is_empty` but callable at runtime.
     */
    constexpr bool is_empty() const {
        return impl_->is_empty;
    }

    /**
     * Check if this type is trivial copyable, i.e., no ctor or dtor.
     * Similar to `std::is_trivially_copyable` but callable at runtime.
     */
    constexpr bool is_trivially_copyable() const {
        return impl_->is_trivial_copy;
    }

    /**
     * If this type is a pointer, returns the type pointed to by this type.
     * Similar to `std::remove_pointer` but callable at runtime.
     */
    constexpr TypeInfo remove_pointer() const {
        return impl_->remove_pointer_type;
    }

    /**
     * If this type is `const`, returns the type without `const`.
     * Similar to `std::remove_const` but callable at runtime.
     */
    constexpr TypeInfo remove_const() const {
        return impl_->remove_const;
    }

    /**
     * If this type is not `const`, returns the type with `const`.
     * Similar to `std::add_const` but callable at runtime.
     */
    constexpr TypeInfo add_const() const {
        return impl_->add_const;
    }

    bool operator==(const TypeInfo& that) const {
        return this->impl_->type_info == that.impl_->type_info

            // Type info does not store if type is const. Check this separately.
            && this->impl_->is_const == that.impl_->is_const;
    }

    bool operator!=(const TypeInfo& that) const {
        return !(*this == that);
    }

    uint64_t hash() const {
        return std::type_index(impl_->type_info).hash_code();
    }

  private:
    const Impl* impl_ = nullptr;
};

/**
 * Returns the `TypeInfo` for type `T`.
 */
template<typename T>
inline constexpr TypeInfo type_of() {
    return TypeInfo::template of<T>();
}

/**
 * Returns the `TypeInfo` for type `T`.
 */
template<typename T>
inline constexpr TypeInfo type_of(const T&) {
    return TypeInfo::template of<T>();
}

/**
 * Returns name of type `T` as a string.
 */
template<typename T>
inline const std::string& type_name() {
    return type_of<T>().name();
}

/**
 * Returns name of type `T` as a string.
 */
template<typename T>
inline const std::string& type_name(const T&) {
    return type_of<T>().name();
}

inline std::ostream& operator<<(std::ostream& os, const TypeInfo& t) {
    return os << t.name();
}

struct TemplateArg {
#define KERNEL_LAUNCHER_TEMPLATE_ARG_CTOR(type)                  \
    TemplateArg(type i) {                                        \
        inner_ = std::string("(" #type ")") + std::to_string(i); \
    }

    KERNEL_LAUNCHER_TEMPLATE_ARG_CTOR(char);
    KERNEL_LAUNCHER_TEMPLATE_ARG_CTOR(signed char);
    KERNEL_LAUNCHER_TEMPLATE_ARG_CTOR(unsigned char);

    KERNEL_LAUNCHER_TEMPLATE_ARG_CTOR(short);
    KERNEL_LAUNCHER_TEMPLATE_ARG_CTOR(int);
    KERNEL_LAUNCHER_TEMPLATE_ARG_CTOR(long);
    KERNEL_LAUNCHER_TEMPLATE_ARG_CTOR(long long);
    KERNEL_LAUNCHER_TEMPLATE_ARG_CTOR(unsigned short);
    KERNEL_LAUNCHER_TEMPLATE_ARG_CTOR(unsigned int);
    KERNEL_LAUNCHER_TEMPLATE_ARG_CTOR(unsigned long);
    KERNEL_LAUNCHER_TEMPLATE_ARG_CTOR(unsigned long long);
#undef KERNEL_LAUNCHER_TEMPLATE_ARG_CTOR

    TemplateArg(bool b) {
        inner_ = b ? "(bool)true" : "(bool)false";
    }

    TemplateArg(TypeInfo type) {
        inner_ = type.name();
    }

    template<typename T>
    static TemplateArg from_type() {
        return TemplateArg(TypeInfo::of<T>());
    }

    static TemplateArg from_string(std::string s) {
        TemplateArg t(0);
        t.inner_ = std::move(s);
        return t;
    }

    const std::string& get() const {
        return inner_;
    }

    const std::string& to_string() const {
        return get();
    }

  private:
    std::string inner_;
};

/**
 * 3-dimensional vector that describes the `dimensions` of a compute-problem.
 */
struct ProblemSize {
    /**
     * Constructor.
     */
    constexpr ProblemSize(uint32_t x = 1, uint32_t y = 1, uint32_t z = 1) :
        x(x),
        y(y),
        z(z) {}

    constexpr ProblemSize(dim3 v) : ProblemSize(v.x, v.y, v.z) {}

    /**
     * Convert this `ProblemSize` to a `dim3`.
     */
    constexpr operator dim3() const {
        return {x, y, z};
    }

    /**
     * Returns the `i`-th component.
     */
    constexpr uint32_t& operator[](size_t i) {
        return data()[i];
    }

    /**
     * Returns the `i`-th component.
     */
    constexpr const uint32_t& operator[](size_t i) const {
        return data()[i];
    }

    constexpr size_t size() const {
        return 3;
    }

    constexpr uint32_t* data() {
        return &x;
    }

    constexpr const uint32_t* data() const {
        return &x;
    }

    constexpr bool operator==(const ProblemSize& that) const {
        return x == that.x && y == that.y && z == that.z;
    }

    constexpr bool operator!=(const ProblemSize& that) const {
        return !(*this == that);
    }

    friend std::ostream& operator<<(std::ostream& s, const ProblemSize& p) {
        if (p.z != 1) {
            return s << "(" << p.x << ", " << p.y << ", " << p.z << ")";
        } else if (p.y != 1) {
            return s << "(" << p.x << ", " << p.y << ")";
        } else {
            return s << p.x;
        }
    }

    // These should be public
    /**
     * X axis.
     */
    uint32_t x;

    /**
     * Y axis.
     */
    uint32_t y;

    /**
     * Z axis.
     */
    uint32_t z;
};

template<
    typename L,
    typename R,
    typename std::enable_if<
        std::is_signed<L>::value == std::is_signed<R>::value,
        std::nullptr_t>::type = nullptr>
inline bool cmp_less(L left, R right) {
    return left < right;
}

template<
    typename L,
    typename R,
    typename std::enable_if<
        std::is_signed<L>::value && !std::is_signed<R>::value,
        std::nullptr_t>::type = nullptr>
inline bool cmp_less(L left, R right) {
    using UL = std::make_unsigned_t<L>;
    return left < 0 || UL(left) < right;
}

template<
    typename L,
    typename R,
    typename std::enable_if<
        !std::is_signed<L>::value && std::is_signed<R>::value,
        std::nullptr_t>::type = nullptr>
inline bool cmp_less(L left, R right) {
    using UR = std::make_unsigned_t<R>;
    return right >= 0 && left < UR(right);
}

template<typename T, typename L, typename R>
inline bool in_range(T val, L min, R max) {
    return !cmp_less(val, min) && !cmp_less(max, val);
}

template<typename R, typename T>
inline bool in_range(T val) {
    return in_range(
        val,
        std::numeric_limits<R>::min(),
        std::numeric_limits<R>::max());
}

template<typename L, typename R>
auto div_ceil(L left, R right) {
    return (left / right) + (left % right != 0);
}

bool safe_double_to_int64(double input, int64_t& output);
bool safe_int64_add(int64_t lhs, int64_t rhs, int64_t& output);
bool safe_int64_sub(int64_t lhs, int64_t rhs, int64_t& output);
bool safe_int64_mul(int64_t lhs, int64_t rhs, int64_t& output);
bool safe_int64_div(int64_t lhs, int64_t rhs, int64_t& output);

bool string_match(const char* pattern, const char* input);
std::vector<std::string>
string_split(const char* input, const std::vector<char>& delims);
std::vector<std::string> string_split(const char* input, char delim);

using hash_t = uint64_t;

hash_t hash_string(const char* buffer, size_t num_bytes);

inline hash_t hash_string(const std::vector<int8_t>& v) {
    return hash_string((char*)v.data(), v.size());
}

inline hash_t hash_string(const std::vector<uint8_t>& v) {
    return hash_string((char*)v.data(), v.size());
}

inline hash_t hash_string(const std::string& v) {
    return hash_string(v.data(), v.size());
}

inline hash_t hash_combine(hash_t a, hash_t b) {
    return a + 0x9e3779b9 + (b << 6) + (b >> 2);
}

inline hash_t hash_fields() {
    return 0;
}

template<typename T, typename... Rest>
inline hash_t hash_fields(const T& first, const Rest&... rest) {
    return hash_combine(std::hash<T> {}(first), hash_fields(rest...));
}

}  // namespace kernel_launcher

namespace std {
template<>
struct hash<kernel_launcher::TypeInfo> {
    size_t operator()(const kernel_launcher::TypeInfo& info) const {
        return info.hash();
    }
};

template<>
struct hash<kernel_launcher::ProblemSize> {
    size_t operator()(const kernel_launcher::ProblemSize& p) const {
        return kernel_launcher::hash_fields(p.x, p.y, p.z);
    }
};
}  // namespace std

#endif  //KERNEL_LAUNCHER_UTILS_H
