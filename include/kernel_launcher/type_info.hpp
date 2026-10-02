#ifndef KERNEL_LAUNCHER_TYPE_INFO_H
#define KERNEL_LAUNCHER_TYPE_INFO_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <typeinfo>

#include "kernel_launcher/primitive.hpp"
#include "kernel_launcher/utils.hpp"

namespace kernel_launcher {

class KernelArg;

namespace detail {
using ToPrimitiveFn = Primitive (*)(const void*);

// Returns the function that converts a `T` to a `Primitive`, or `nullptr` if
// `T` has no `PrimitiveTraits`. Pointers are excluded: kernel arguments of
// type `char*` point to GPU memory and must not be read as strings.
template<typename T>
constexpr ToPrimitiveFn to_primitive_fn_for() {
    if constexpr (HasPrimitiveTraits<T>::value && !std::is_pointer<T>::value) {
        return [](const void* ptr) { return Primitive(*static_cast<const T*>(ptr)); };
    } else {
        return nullptr;
    }
}

struct TypeInfoImpl {
    size_t size;
    size_t alignment;
    bool is_pointer;
    bool is_const;
    const TypeInfoImpl* remove_pointer;
    const TypeInfoImpl* remove_const;
    const std::type_info* type_info;
    std::string_view name;
    ToPrimitiveFn to_primitive;
};

template<typename T>
static const TypeInfoImpl type_info_for = {
    sizeof(T),
    alignof(T),
    std::is_pointer<T>::value,
    std::is_const<T>::value,
    &type_info_for<typename std::remove_pointer<T>::type>,
    &type_info_for<typename std::remove_const<T>::type>,
    &typeid(T),
    // Interned so the demangled name outlives this static.
    intern_string(demangle_type_info(typeid(T))),
    to_primitive_fn_for<T>()};

template<>
constexpr TypeInfoImpl type_info_for<void> = {
    0,
    0,
    std::is_pointer<void>::value,
    std::is_const<void>::value,
    &type_info_for<typename std::remove_pointer<void>::type>,
    &type_info_for<typename std::remove_const<void>::type>,
    &typeid(void),
    "void",
    nullptr};
}  // namespace detail

/**
 * @addtogroup types
 * @{
 */

/**
 * Describes a C++ type: its size, alignment, qualifiers, and demangled name.
 */
class TypeInfo {
  public:
    /**
     * Returns the type information of `T`.
     */
    template<typename T>
    static TypeInfo of() noexcept {
        return {&detail::type_info_for<T>};
    }

    /**
     * Returns the type information of `T`, deduced from the given value.
     */
    template<typename T>
    static TypeInfo of(const T&) noexcept {
        return {&detail::type_info_for<T>};
    }

    /**
     * Returns a type that only exists by name, for example a struct that is defined in the
     * kernel source but not on the host. The `name` must resolve to that type in the kernel
     * source and `size` must match its size on the device. Opaque types compare equal if their
     * name, size, and alignment are equal, and they never compare equal to a C++ type.
     */
    static TypeInfo opaque(std::string_view name, size_t size, size_t alignment);

    /**
     * Returns the size of this type in bytes, like `sizeof(T)`.
     */
    size_t size_in_bytes() const noexcept {
        return impl_->size;
    }

    /**
     * Returns the alignment of this type in bytes, like `alignof(T)`.
     */
    size_t alignment_in_bytes() const noexcept {
        return impl_->alignment;
    }

    /**
     * Returns `true` if this type is `const`..
     */
    bool is_const() const noexcept {
        return impl_->is_const;
    }

    /**
     * Returns `true` if this type is a pointer type.
     */
    bool is_pointer() const noexcept {
        return impl_->is_pointer;
    }

    /**
     * Returns `true` if this type is `void` or `const void`.
     */
    bool is_void() const noexcept {
        // Top-level `const` is ignored by `operator==`, so this also matches `const void`.
        return *this == TypeInfo::of<void>();
    }

    /**
     * Returns the demangled name of this type, for example `const float*`.
     */
    std::string_view name() const noexcept {
        return impl_->name;
    }

    /**
     * Returns `true` if values of this type can be converted to a `Primitive`. This basically
     * asks if `Primitive(T(value))` is valid where `T` this type.
     */
    bool is_primitive() const noexcept {
        return impl_->to_primitive != nullptr;
    }

    /**
     * Converts the value at `ptr` (which must be of this type) to a `Primitive`.
     * Throws an exception if `is_primitive()` is `false`.
     */
    Primitive to_primitive(const void* ptr) const;

    /**
     * Returns the pointee type if this is a pointer type, or this type otherwise.
     */
    TypeInfo remove_pointer() const {
        return {impl_->remove_pointer};
    }

    /**
     * Returns this type without its top-level `const` qualifier.
     */
    TypeInfo remove_const() const {
        return {impl_->remove_const};
    }

    /**
     * Returns `true` if both objects describe the same type.
     */
    bool operator==(const TypeInfo& that) const noexcept {
        // Opaque types have no `type_info`, but they are interned by name, size, and alignment.
        if (impl_->type_info == nullptr || that.impl_->type_info == nullptr) {
            return impl_ == that.impl_;
        }

        return *impl_->type_info == *that.impl_->type_info;
    }

    /**
     * Returns `true` if both objects describe different types.
     */
    bool operator!=(const TypeInfo& that) const noexcept {
        return !(*this == that);
    }

    /**
     * Returns a hash value that is consistent with `operator==`.
     */
    size_t hash() const noexcept {
        if (impl_->type_info == nullptr) {
            return std::hash<const void*> {}(impl_);
        }

        return impl_->type_info->hash_code();
    }

  private:
    TypeInfo(const detail::TypeInfoImpl* impl) : impl_(impl) {}

    const detail::TypeInfoImpl* impl_ = &detail::type_info_for<void>;
};

/**
 * A parameter passed to a template instantiation. Can be either a `TypeInfo`
 * object, a `Primitive`, or any value of type `T` that is an integral or enum
 * type.
 */
class TemplateArg {
  public:
    TemplateArg(TypeInfo type) : repr_(type.name()) {}

    TemplateArg(Primitive value) {
        switch (value.kind()) {
            case Primitive::Kind::Int64:
                *this = TemplateArg(value.to<std::intmax_t>());
                break;
            case Primitive::Kind::Bool:
                *this = TemplateArg(value.to<bool>());
                break;
            default:
                throw std::runtime_error(
                    "TemplateArg: only integral and boolean primitives can be used as template arguments");
        }
    }

    template<
        typename T,
        typename =
            typename std::enable_if<std::is_integral<T>::value || std::is_enum<T>::value>::type>
    TemplateArg(T value) {
        if constexpr (std::is_same<T, bool>::value) {
            repr_ = value ? "true" : "false";
        } else if constexpr (std::is_enum<T>::value) {
            using U = typename std::underlying_type<T>::type;
            repr_ =
                concat("(", TypeInfo::of<T>().name(), ")", integer_literal(static_cast<U>(value)));
        } else {
            // Cast to `T` so that the argument keeps its type, which matters
            // for deduced parameters such as `template<auto N>`.
            repr_ = concat("(", TypeInfo::of<T>().name(), ")", integer_literal(value));
        }
    }

    /**
     * Constructs a template argument directly from its string representation,
     * which is inserted verbatim into the template argument list. No
     * validation is performed. This is intended for internal use only (e.g.,
     * when deserializing); prefer the other constructors.
     */
    static TemplateArg unsafe_from_string(std::string repr) {
        TemplateArg arg;
        arg.repr_ = std::move(repr);
        return arg;
    }

    const std::string& get() const {
        return repr_;
    }

    bool operator==(const TemplateArg& that) const {
        return repr_ == that.repr_;
    }

    bool operator!=(const TemplateArg& that) const {
        return !(*this == that);
    }

    size_t hash() const noexcept {
        return hash_fields(repr_);
    }

  private:
    TemplateArg() = default;

    // Formats `value` as an integer literal that cannot overflow when parsed.
    // Values are widened first, since `char` types would otherwise be
    // streamed as characters.
    template<typename T>
    static std::string integer_literal(T value) {
        if constexpr (!std::is_signed<T>::value) {
            return concat(std::uintmax_t(value), "ull");
        } else if (value == std::numeric_limits<T>::min()) {
            // `-9223372036854775808` parses as `-(9223372036854775808)`,
            // which overflows, so write the minimum as `(min + 1) - 1`.
            return concat("(", std::intmax_t(value) + 1, "ll - 1)");
        } else {
            return concat(std::intmax_t(value), "ll");
        }
    }

    std::string repr_;
};

/** @} */

}  // namespace kernel_launcher

namespace std {
template<>
struct hash<kernel_launcher::TypeInfo> {
    size_t operator()(const kernel_launcher::TypeInfo& v) const noexcept {
        return v.hash();
    }
};

template<>
struct hash<kernel_launcher::TemplateArg> {
    size_t operator()(const kernel_launcher::TemplateArg& v) const noexcept {
        return v.hash();
    }
};
}  // namespace std

#endif
