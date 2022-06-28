#ifndef KERNEL_LAUNCHER_UTILS_H
#define KERNEL_LAUNCHER_UTILS_H

#include <iosfwd>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

namespace kernel_launcher {

std::string demangle_type_info(const std::type_info& type);

struct TypeInfo {
  private:
    struct Impl {
        size_t alignment;
        size_t size;
        bool is_const;
        const std::type_info& type_info;
        std::string name;
        const Impl* pointee_type;
    };

    TypeInfo(const Impl* impl) : impl_(impl) {}

  public:
    TypeInfo() : TypeInfo(nullptr) {}

    template<typename T>
    static TypeInfo of() {
        static const Impl result = {
            alignof(T),
            sizeof(T),
            std::is_const<T>::value,
            typeid(T),
            demangle_type_info(typeid(T)),
            std::is_pointer<T>::value
                ? TypeInfo::template of<typename std::remove_pointer<T>::type>()
                      .impl_
                : nullptr,
        };

        return TypeInfo {&result};
    }

    const std::string& name() const {
        return impl_->name;
    }

    size_t size() const {
        return impl_->size;
    }

    size_t alignment() const {
        return impl_->alignment;
    }

    bool is_pointer() const {
        return impl_->pointee_type != nullptr;
    }

    bool is_const() const {
        return impl_->is_const;
    }

    TypeInfo remove_pointer() const {
        return is_pointer() ? TypeInfo {impl_->pointee_type} : *this;
    }

    bool operator==(const TypeInfo& that) const {
        return this->impl_->type_info == that.impl_->type_info;
    }

    bool operator!=(const TypeInfo& that) const {
        return !(*this == that);
    }

  private:
    const Impl* impl_ = nullptr;
};

template<typename T>
inline TypeInfo type_of() {
    return TypeInfo::template of<T>();
}

template<typename T>
inline TypeInfo type_of(const T&) {
    return TypeInfo::template of<T>();
}

template<typename T>
inline const std::string& type_name() {
    return type_of<T>().name();
}

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

inline bool safe_double_to_int64(double input, int64_t& output) {
    static constexpr double min_val =
        static_cast<double>(std::numeric_limits<int64_t>::min());
    static constexpr double max_val = static_cast<double>(
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) + 1);
    int64_t v = static_cast<int64_t>(input);

    if (input >= min_val && input < max_val && input == double(v)) {
        output = v;
        return true;
    }

    output = 0;
    return false;
}

using hash_t = uint64_t;

inline hash_t hash_string(const char* buffer, size_t num_bytes) {
    hash_t hash = 0xcbf29ce484222325;
    hash_t prime = 0x100000001b3;

    for (size_t i = 0; i < num_bytes; i++) {
        hash = (hash ^ (hash_t)(unsigned char)buffer[i]) * prime;
    }

    return hash;
}

inline hash_t hash_string(const std::vector<char>& v) {
    return hash_string(v.data(), v.size());
}

inline hash_t hash_string(const std::string& v) {
    return hash_string(v.data(), v.size());
}

inline hash_t hash_combine(hash_t a, hash_t b) {
    return a + 0x9e3779b9 + (b << 6) + (b >> 2);
}

}  // namespace kernel_launcher

#endif  //KERNEL_LAUNCHER_UTILS_H
