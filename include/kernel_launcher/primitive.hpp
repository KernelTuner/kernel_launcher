#ifndef KERNEL_LAUNCHER_PRIMITIVE_H
#define KERNEL_LAUNCHER_PRIMITIVE_H

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

#include "kernel_launcher/utils.hpp"

namespace kernel_launcher {

class Primitive;

/**
 * @addtogroup types
 * @{
 */

/**
 * Customization point that describes how a value of type `T` is converted to
 * and from a `Primitive`. A specialization provides two static functions:
 *
 * - `static Primitive wrap(const T& value)`: converts `value` to a
 *   `Primitive`. Required; enables `Primitive(value)`.
 * - `static T unwrap(const Primitive& value)`: converts a `Primitive` back
 *   to `T`, throwing an exception if that is not possible. Optional; enables
 *   `prim.to<T>()`.
 *
 * The second template parameter can be used to specialize for a family of
 * types using `std::enable_if_t`.
 */
template<typename T, typename Enable = void>
struct PrimitiveTraits {};

/** @} */

namespace detail {
// Indicates whether `PrimitiveTraits<T>` is defined for type `T`.
template<typename T, typename = void>
struct HasPrimitiveTraits: std::false_type {};

template<typename T>
struct HasPrimitiveTraits<
    T,
    std::void_t<decltype(PrimitiveTraits<T>::wrap(std::declval<const T&>()))>>: std::true_type {};
}  // namespace detail

/**
 * @addtogroup types
 * @{
 */

/**
 * A dynamically typed scalar value. A `Primitive` holds exactly one of:
 *
 * - `Empty`: no value (comparable to `None`, `nil`, or `nullptr`).
 * - `Int64`: a signed integer, stored as `std::intmax_t`.
 * - `Float64`: a floating-point number, stored as `double`.
 * - `Bool`: a boolean, `true` or `false`.
 * - `String`: a string, stored as an interned `Symbol`.
 *
 * Any type `T` with a `PrimitiveTraits<T>` specialization can be converted
 * to a `Primitive` using `Primitive(value)` and back using `prim.to<T>()`.
 */
class Primitive {
  public:
    /**
     * The kind of value held by a `Primitive`.
     */
    enum struct Kind { Empty, Int64, Float64, Bool, String };

    /**
     * Tag type that selects the kind of primitive to construct, for example
     * `Primitive(Primitive::KindTag<Primitive::Kind::Int64> {}, 1)`.
     */
    template<Kind k>
    using KindTag = std::integral_constant<Kind, k>;

    Primitive(const Primitive&) = default;
    Primitive(Primitive&&) noexcept = default;
    Primitive& operator=(const Primitive&) = default;
    Primitive& operator=(Primitive&&) noexcept = default;

    /**
     * Constructs an empty primitive.
     */
    Primitive(KindTag<Kind::Empty> = {}) : kind_(Kind::Empty) {}

    /**
     * Constructs an integer primitive.
     */
    Primitive(KindTag<Kind::Int64>, std::intmax_t value) : kind_(Kind::Int64) {
        value_.int_val = value;
    }

    /**
     * Constructs a floating-point primitive.
     */
    Primitive(KindTag<Kind::Float64>, double value) : kind_(Kind::Float64) {
        value_.fp_val = value;
    }

    /**
     * Constructs a boolean primitive.
     */
    Primitive(KindTag<Kind::Bool>, bool value) : kind_(Kind::Bool) {
        value_.bool_val = value;
    }

    /**
     * Constructs a string primitive.
     */
    Primitive(KindTag<Kind::String>, Symbol value) : kind_(Kind::String) {
        value_.str_val = value;
    }

    /**
     * Constructs a primitive from a value of type `T`. Only participates in
     * overload resolution if `PrimitiveTraits<T>` is defined.
     */
    template<
        typename T,
        typename = std::enable_if_t<std::conjunction_v<
            std::negation<std::is_same<T, Primitive>>,
            detail::HasPrimitiveTraits<T>>>>
    Primitive(const T& value) : Primitive(PrimitiveTraits<T>::wrap(value)) {}

    /**
     * Constructs a primitive from a value of type `T`. Returns an empty
     * primitive if `PrimitiveTraits<T>` is not defined.
     */
    template<typename T>
    static Primitive from_or_empty(const T& value) {
        if constexpr (detail::HasPrimitiveTraits<T>::value) {
            return Primitive(value);
        } else {
            return {};
        }
    }

    /**
     * Converts this primitive to a value of type `T` using
     * `PrimitiveTraits<T>::%unwrap`. Throws an exception if the conversion
     * fails.
     */
    template<typename T>
    T to() const {
        return PrimitiveTraits<T>::unwrap(*this);
    }

    /**
     * Returns the kind of value held by this primitive.
     */
    Kind kind() const noexcept {
        return kind_;
    }

    /**
     * Returns the name of the kind of this primitive, as given by
     * `primitive_kind_name`.
     */
    const char* kind_name() const noexcept;

    /**
     * Returns `true` if this primitive is empty.
     */
    bool is_empty() const noexcept {
        return kind_ == Kind::Empty;
    }

    /**
     * Returns `true` if this primitive is an integer, float, or bool.
     */
    bool is_numeric() const noexcept {
        return kind_ == Kind::Int64 || kind_ == Kind::Float64 || kind_ == Kind::Bool;
    }

    /**
     * Converts this primitive to an integer:
     *
     * - `Int64`: returns the value.
     * - `Float64`: returns the value if it is an integer that fits in
     *   `std::intmax_t`; throws an exception otherwise.
     * - `Bool`: returns `0` or `1`.
     * - `Empty`, `String`: throws an exception.
     */
    std::intmax_t to_integer() const;

    /**
     * Converts this primitive to a floating-point number:
     *
     * - `Int64`: returns the value converted to `double`.
     * - `Float64`: returns the value.
     * - `Bool`: returns `0.0` or `1.0`.
     * - `Empty`, `String`: throws an exception.
     */
    double to_float() const;

    /**
     * Converts this primitive to a boolean:
     *
     * - `Empty`: returns `false`.
     * - `Int64`: returns `true` if the value is nonzero.
     * - `Float64`: returns `true` if the value is nonzero.
     * - `Bool`: returns the value.
     * - `String`: returns `true` if the string is non-empty.
     */
    bool to_bool() const;

    /**
     * Converts this primitive to a string. Supported for all kinds:
     *
     * - `Empty`: returns an empty string.
     * - `Int64`: returns the decimal representation.
     * - `Float64`: returns the shortest representation that parses back to
     *   the same value. It always reads as a floating-point number (e.g.,
     *   `2.0`, not `2`).
     * - `Bool`: returns `"true"` or `"false"`.
     * - `String`: returns the string.
     */
    std::string to_string() const;

    /**
     * Converts this primitive to an interned string. For `String` this
     * returns the stored symbol; other kinds are converted using `to_string`.
     */
    Symbol to_symbol() const;

    /**
     * Returns a hash of this primitive. Primitives that compare equal have
     * equal hashes, including numeric values of different kinds (e.g., `1`,
     * `1.0`, and `true`).
     */
    size_t hash() const noexcept;

    /**
     * Rounds this primitive to the nearest integer, with halfway cases
     * rounded away from zero. Non-floating-point kinds are converted using
     * `to_integer`. Throws an exception if the result does not fit in
     * `std::intmax_t`.
     */
    std::intmax_t round() const;

    /**
     * Rounds this primitive down to the nearest integer. Non-floating-point
     * kinds are converted using `to_integer`. Throws an exception if the
     * result does not fit in `std::intmax_t`.
     */
    std::intmax_t floor() const;

    /**
     * Rounds this primitive up to the nearest integer. Non-floating-point
     * kinds are converted using `to_integer`. Throws an exception if the
     * result does not fit in `std::intmax_t`.
     */
    std::intmax_t ceil() const;

  private:
    friend bool operator==(const Primitive& a, const Primitive& b);
    friend bool operator<(const Primitive& a, const Primitive& b);

    Kind kind_ = Kind::Empty;
    union {
        std::intmax_t int_val;
        double fp_val;
        bool bool_val;
        Symbol str_val;
    } value_ {};
};

/**
 * Returns `true` if `a` and `b` hold the same value. Numeric kinds (`Int64`,
 * `Float64`, `Bool`) are compared by value regardless of kind, so `1`, `1.0`,
 * and `true` are all equal. Integers and floats are compared exactly, without
 * loss of precision. An empty primitive only equals another empty primitive,
 * and a string only equals an identical string.
 *
 * A NaN is not equal to anything, including itself.
 */
bool operator==(const Primitive& a, const Primitive& b);

/**
 * Returns `true` if `a` is ordered before `b`. Empty primitives are ordered
 * before numeric values, which are ordered before strings. Numeric values
 * are ordered by value regardless of kind, and strings lexicographically.
 *
 * A NaN is unordered with respect to every numeric value, as for `double`.
 * Primitives holding NaN should thus not be sorted or used as keys in
 * containers such as `std::map` or `std::unordered_map`.
 */
bool operator<(const Primitive& a, const Primitive& b);

/**
 * @name Arithmetic operators
 *
 * Both operands must be numeric (`Int64`, `Float64`, or `Bool`); an exception
 * is thrown otherwise. The only exception is `+`, which concatenates two
 * strings.
 *
 * If either operand is `Float64`, the operation is performed on `double`
 * and the result is `Float64`. Otherwise, the operation is performed on
 * integers and the result is `Int64` (also for two `Bool` operands).
 *
 * Integer operations throw an exception on overflow and on division or
 * modulo by zero. Integer division truncates towards zero. Floating-point
 * operations follow IEEE rules instead (yielding infinity or NaN), and
 * floating-point modulo is `std::fmod`.
 *
 * @{
 */
Primitive operator+(const Primitive& a, const Primitive& b);
Primitive operator-(const Primitive& a, const Primitive& b);
Primitive operator*(const Primitive& a, const Primitive& b);
Primitive operator/(const Primitive& a, const Primitive& b);
Primitive operator%(const Primitive& a, const Primitive& b);
Primitive operator-(const Primitive& a);
/** @} */

/**
 * Returns `true` if `a` is falsy, that is, if `a.to_bool()` is `false`.
 */
bool operator!(const Primitive& a);

/**
 * Returns `a` if it is falsy (see `Primitive::to_bool`), otherwise `b`, like
 * `and` in Python. The result is thus one of the operands and not a `bool`;
 * use `to_bool` to test it. Unlike the built-in `&&`, both operands are
 * always evaluated.
 */
Primitive operator&&(const Primitive& a, const Primitive& b);

/**
 * Returns `a` if it is truthy (see `Primitive::to_bool`), otherwise `b`, like
 * `or` in Python. The result is thus one of the operands and not a `bool`;
 * use `to_bool` to test it. Unlike the built-in `||`, both operands are
 * always evaluated.
 */
Primitive operator||(const Primitive& a, const Primitive& b);

/**
 * Returns a human-readable name for `kind`: `"empty"`, `"int"`, `"float"`,
 * `"bool"`, or `"string"`.
 */
const char* primitive_kind_name(Primitive::Kind kind);

std::ostream& operator<<(std::ostream& stream, const Primitive& value);
std::ostream& operator<<(std::ostream& stream, Primitive::Kind kind);

inline bool operator!=(const Primitive& a, const Primitive& b) {
    return !(a == b);
}

inline bool operator<=(const Primitive& a, const Primitive& b) {
    return a < b || a == b;
}

inline bool operator>(const Primitive& a, const Primitive& b) {
    return b < a;
}

inline bool operator>=(const Primitive& a, const Primitive& b) {
    return b <= a;
}

template<>
struct PrimitiveTraits<Primitive> {
    static Primitive wrap(Primitive value) {
        return value;
    }

    static Primitive unwrap(const Primitive& value) {
        return value;
    }
};

template<>
struct PrimitiveTraits<bool> {
    static Primitive wrap(bool value) {
        return {Primitive::KindTag<Primitive::Kind::Bool> {}, value};
    }

    static bool unwrap(const Primitive& value) {
        return value.to_bool();
    }
};

template<typename T>
struct PrimitiveTraits<T, std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>>> {
    static Primitive wrap(T value) {
        if constexpr (!std::is_signed_v<T>) {
            if (std::uintmax_t(value) > std::uintmax_t(std::numeric_limits<std::intmax_t>::max())) {
                throw std::overflow_error("Primitive: integer value does not fit in intmax_t");
            }
        }

        return {Primitive::KindTag<Primitive::Kind::Int64> {}, std::intmax_t(value)};
    }

    static T unwrap(const Primitive& value) {
        std::intmax_t v = value.to_integer();

        if constexpr (std::is_signed_v<T>) {
            if (v < std::intmax_t(std::numeric_limits<T>::min())
                || v > std::intmax_t(std::numeric_limits<T>::max())) {
                throw std::overflow_error("Primitive: integer value does not fit in target type");
            }
        } else {
            if (v < 0 || std::uintmax_t(v) > std::uintmax_t(std::numeric_limits<T>::max())) {
                throw std::overflow_error("Primitive: integer value does not fit in target type");
            }
        }

        return T(v);
    }
};

template<typename T>
struct PrimitiveTraits<T, std::enable_if_t<std::is_floating_point_v<T>>> {
    static Primitive wrap(T value) {
        return {Primitive::KindTag<Primitive::Kind::Float64> {}, double(value)};
    }

    static T unwrap(const Primitive& value) {
        return T(value.to_float());
    }
};

template<>
struct PrimitiveTraits<Symbol> {
    static Primitive wrap(Symbol value) {
        return {Primitive::KindTag<Primitive::Kind::String> {}, value};
    }

    static Symbol unwrap(const Primitive& value) {
        return value.to_symbol();
    }
};

template<>
struct PrimitiveTraits<std::string_view> {
    static Primitive wrap(std::string_view value) {
        return {Primitive::KindTag<Primitive::Kind::String> {}, Symbol(value)};
    }

    // The returned view points to an interned string, so it remains valid.
    static std::string_view unwrap(const Primitive& value) {
        return value.to_symbol().get();
    }
};

template<>
struct PrimitiveTraits<std::string> {
    static Primitive wrap(std::string_view value) {
        return {Primitive::KindTag<Primitive::Kind::String> {}, Symbol(value)};
    }

    static std::string unwrap(const Primitive& value) {
        return value.to_string();
    }
};

template<typename T>
struct PrimitiveTraits<T, std::enable_if_t<std::is_convertible_v<T, const char*>>> {
    // A null pointer (including `nullptr` itself) yields an empty primitive.
    static Primitive wrap(const char* value) {
        if (value == nullptr) {
            return {};
        }

        return {Primitive::KindTag<Primitive::Kind::String> {}, Symbol(value)};
    }
};

/** @} */

}  // namespace kernel_launcher

namespace std {
template<>
struct hash<kernel_launcher::Primitive> {
    size_t operator()(const kernel_launcher::Primitive& v) const noexcept {
        return v.hash();
    }
};
}  // namespace std

#endif
