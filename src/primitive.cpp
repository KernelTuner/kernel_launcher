#include "kernel_launcher/primitive.hpp"

#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

#include "nlohmann/json.hpp"

namespace kernel_launcher {

namespace {
// 2^63: exactly representable as a double, and the smallest double that no
// longer fits in an intmax_t. Computed as max()+1 in unsigned arithmetic to
// avoid signed-overflow UB (intmax_t::max()+1 would overflow intmax_t).
constexpr double INTMAX_AS_DOUBLE_LIMIT =
    double(std::uintmax_t(std::numeric_limits<std::intmax_t>::max()) + 1);

// Converts a double to an intmax_t. Throws if the double is not an integer
// exactly representable as an intmax_t (i.e. the conversion would lose
// precision).
std::intmax_t safe_double_to_intmax(double d) {
    if (std::isnan(d) || std::isinf(d) || d != std::floor(d) || d < -INTMAX_AS_DOUBLE_LIMIT
        || d >= INTMAX_AS_DOUBLE_LIMIT) {
        throw std::runtime_error(concat("Primitive: could not convert ", d, " to an integer"));
    }

    return static_cast<std::intmax_t>(d);
}

enum class Ordering { Less, Equal, Greater };

// Compares an intmax_t and a double exactly: never converts the intmax_t to
// a double (which would lose precision for large magnitudes) nor truncates
// the double to an intmax_t (which would ignore its fractional part).
// Returns std::nullopt if the double is NaN (the two are unordered).
std::optional<Ordering> safe_compare(std::intmax_t i, double d) {
    if (std::isnan(d)) {
        return std::nullopt;
    }
    if (d >= INTMAX_AS_DOUBLE_LIMIT) {
        return Ordering::Less;
    }
    if (d < -INTMAX_AS_DOUBLE_LIMIT) {
        return Ordering::Greater;
    }

    double floor_d = std::floor(d);
    std::intmax_t fi = static_cast<std::intmax_t>(floor_d);

    if (i == fi) {
        return d > floor_d ? Ordering::Less : Ordering::Equal;
    }

    return i < fi ? Ordering::Less : Ordering::Greater;
}

bool primitive_is_float(const Primitive& v) {
    return v.kind() == Primitive::Kind::Float64;
}

// Formats a double using the fewest digits that still parse back to the same
// value. Integral values keep a trailing `.0`.
//
// This is what `std::to_chars` does, but libstdc++ only supports that for
// floating-point types since GCC 11. We instead use the implementation that
// nlohmann/json uses to serialize numbers. Note that this function lives in
// the `detail` namespace and is not part of the public API of that library,
// so it might need to be revisited when upgrading `thirdparty/nlohmann`.
std::string double_to_string(double d) {
    // `nlohmann::detail::to_chars` only accepts finite values.
    if (std::isnan(d)) {
        return "nan";
    }
    if (std::isinf(d)) {
        return d < 0 ? "-inf" : "inf";
    }

    // Same buffer size as used by the serializer of nlohmann/json. The
    // result is not null-terminated.
    char buffer[64];
    char* end = nlohmann::detail::to_chars(buffer, buffer + sizeof(buffer), d);
    return std::string(buffer, end);
}
}  // namespace

std::intmax_t Primitive::to_integer() const {
    switch (kind_) {
        case Kind::Int64:
            return value_.int_val;
        case Kind::Bool:
            return value_.bool_val ? 1 : 0;
        case Kind::Float64:
            return safe_double_to_intmax(value_.fp_val);
        default:
            throw std::runtime_error("Primitive: cannot convert value to integer");
    }
}

std::intmax_t Primitive::round() const {
    if (kind_ == Kind::Float64) {
        return safe_double_to_intmax(std::round(value_.fp_val));
    } else {
        return to_integer();
    }
}

std::intmax_t Primitive::floor() const {
    if (kind_ == Kind::Float64) {
        return safe_double_to_intmax(std::floor(value_.fp_val));
    } else {
        return to_integer();
    };
}

std::intmax_t Primitive::ceil() const {
    if (kind_ == Kind::Float64) {
        return safe_double_to_intmax(std::ceil(value_.fp_val));
    } else {
        return to_integer();
    }
}

double Primitive::to_float() const {
    switch (kind_) {
        case Kind::Int64:
            return double(value_.int_val);
        case Kind::Float64:
            return double(value_.fp_val);
        case Kind::Bool:
            return value_.bool_val ? 1.0 : 0.0;
        default:
            throw std::runtime_error("Primitive: cannot convert value to float");
    }
}

bool Primitive::to_bool() const {
    switch (kind_) {
        case Kind::Empty:
            return false;
        case Kind::Int64:
            return value_.int_val != 0;
        case Kind::Float64:
            return value_.fp_val != 0.0;
        case Kind::Bool:
            return value_.bool_val;
        case Kind::String:
            return !value_.str_val.empty();
        default:
            throw std::runtime_error("Primitive: cannot convert value to bool");
    }
}

size_t Primitive::hash() const noexcept {
    switch (kind_) {
        case Kind::Empty:
            return std::hash<int> {}(0);
        case Kind::Int64:
            return std::hash<double> {}(double(value_.int_val));
        case Kind::Bool:
            return std::hash<double> {}(value_.bool_val ? 1.0 : 0.0);
        case Kind::Float64:
            return std::hash<double> {}(value_.fp_val);
        case Kind::String:
            return value_.str_val.hash();
        default:
            return 0;
    }
}

std::string Primitive::to_string() const {
    switch (kind_) {
        case Kind::Empty:
            return "";
        case Kind::Int64:
            return std::to_string(value_.int_val);
        case Kind::Float64:
            return double_to_string(value_.fp_val);
        case Kind::Bool:
            return value_.bool_val ? "true" : "false";
        case Kind::String:
            return std::string(value_.str_val.get());
        default:
            throw std::runtime_error("Primitive: cannot convert value to string");
    }
}

Symbol Primitive::to_symbol() const {
    return kind_ == Kind::String ? value_.str_val : Symbol(to_string());
}

bool operator==(const Primitive& a, const Primitive& b) {
    bool a_num = a.is_numeric();
    bool b_num = b.is_numeric();

    if (a_num && b_num) {
        bool a_float = a.kind_ == Primitive::Kind::Float64;
        bool b_float = b.kind_ == Primitive::Kind::Float64;

        if (a_float && b_float) {
            return a.value_.fp_val == b.value_.fp_val;
        } else if (a_float) {
            auto cmp = safe_compare(b.to_integer(), a.value_.fp_val);
            return cmp == Ordering::Equal;
        } else if (b_float) {
            auto cmp = safe_compare(a.to_integer(), b.value_.fp_val);
            return cmp == Ordering::Equal;
        } else {
            return a.to_integer() == b.to_integer();
        }
    }

    if (a.kind_ != b.kind_) {
        return false;
    }

    switch (a.kind_) {
        case Primitive::Kind::Empty:
            return true;
        case Primitive::Kind::String:
            return a.value_.str_val == b.value_.str_val;
        default:
            return false;
    }
}

bool operator<(const Primitive& a, const Primitive& b) {
    if (a.is_numeric() && b.is_numeric()) {
        bool a_float = a.kind_ == Primitive::Kind::Float64;
        bool b_float = b.kind_ == Primitive::Kind::Float64;

        if (a_float && b_float) {
            return a.value_.fp_val < b.value_.fp_val;
        } else if (a_float) {
            // Note: safe_compare requires (int, double), so flip the
            // ordering and check for Greater instead of Less.
            auto cmp = safe_compare(b.to_integer(), a.value_.fp_val);
            return cmp == Ordering::Greater;
        } else if (b_float) {
            auto cmp = safe_compare(a.to_integer(), b.value_.fp_val);
            return cmp == Ordering::Less;
        } else {
            return a.to_integer() < b.to_integer();
        }
    }

    if (a.kind_ != b.kind_) {
        return a.kind_ < b.kind_;
    }

    switch (a.kind_) {
        case Primitive::Kind::Empty:
            return false;
        case Primitive::Kind::String:
            return a.value_.str_val < b.value_.str_val;
        default:
            return false;
    }
}

const char* primitive_kind_name(Primitive::Kind kind) {
    switch (kind) {
        case Primitive::Kind::Empty:
            return "empty";
        case Primitive::Kind::Int64:
            return "int";
        case Primitive::Kind::Float64:
            return "float";
        case Primitive::Kind::Bool:
            return "bool";
        case Primitive::Kind::String:
            return "string";
    }
    return "unknown";
}

const char* Primitive::kind_name() const noexcept {
    return primitive_kind_name(kind_);
}

std::ostream& operator<<(std::ostream& stream, const Primitive& value) {
    return stream << value.to_string();
}

std::ostream& operator<<(std::ostream& stream, Primitive::Kind kind) {
    return stream << primitive_kind_name(kind);
}

namespace {
// Throws unless both operands are numeric (Bool, Int64 or Float64).
void require_numeric(const Primitive& a, const Primitive& b, const char* verb) {
    if (!a.is_numeric() || !b.is_numeric()) {
        throw std::runtime_error(
            concat("Primitive: cannot ", verb, " ", a.kind(), " and ", b.kind()));
    }
}
}  // namespace

// Numeric kinds (Bool < Int64 < Float64) promote to the widest one, and
// Bool + Bool is an integer. String + String concatenates. Every other
// combination, including anything involving Empty, is an error.
Primitive operator+(const Primitive& a, const Primitive& b) {
    bool a_num = a.is_numeric();
    bool b_num = b.is_numeric();

    if (a_num && b_num) {
        if (primitive_is_float(a) || primitive_is_float(b)) {
            return Primitive(a.to_float() + b.to_float());
        }

        std::intmax_t result;
        if (__builtin_add_overflow(a.to_integer(), b.to_integer(), &result)) {
            throw std::overflow_error(
                concat("Primitive: integer overflow in addition of ", a, " and ", b));
        }
        return Primitive(result);
    }

    if (a.kind() == Primitive::Kind::String && b.kind() == Primitive::Kind::String) {
        return {
            Primitive::KindTag<Primitive::Kind::String> {},
            Symbol(concat(a.to_string(), b.to_string()))};
    }

    throw std::runtime_error(concat("Primitive: cannot add ", a.kind(), " and ", b.kind()));
}

// Same rules as operator*: only numeric kinds are allowed, with promotion
// Bool < Int64 < Float64 and an error on integer overflow.
Primitive operator-(const Primitive& a, const Primitive& b) {
    require_numeric(a, b, "subtract");

    if (primitive_is_float(a) || primitive_is_float(b)) {
        return Primitive(a.to_float() - b.to_float());
    }

    std::intmax_t result;
    if (__builtin_sub_overflow(a.to_integer(), b.to_integer(), &result)) {
        throw std::overflow_error(
            concat("Primitive: integer overflow in subtraction of ", a, " and ", b));
    }
    return Primitive(result);
}

// Same promotion rules as operator+, but only numeric kinds are allowed:
// strings and Empty are errors.
Primitive operator*(const Primitive& a, const Primitive& b) {
    require_numeric(a, b, "multiply");

    if (primitive_is_float(a) || primitive_is_float(b)) {
        return Primitive(a.to_float() * b.to_float());
    }

    std::intmax_t result;
    if (__builtin_mul_overflow(a.to_integer(), b.to_integer(), &result)) {
        throw std::overflow_error(
            concat("Primitive: integer overflow in multiplication of ", a, " and ", b));
    }

    return Primitive(result);
}

// Same promotion rules as operator*. Int64 / Int64 truncates towards zero
// like C++ and throws on division by zero; a Float64 operand gives IEEE
// float division (division by zero yields inf/NaN).
Primitive operator/(const Primitive& a, const Primitive& b) {
    require_numeric(a, b, "divide");

    if (primitive_is_float(a) || primitive_is_float(b)) {
        return Primitive(a.to_float() / b.to_float());
    }

    std::intmax_t lhs = a.to_integer();
    std::intmax_t divisor = b.to_integer();

    if (divisor == 0) {
        throw std::runtime_error("Primitive: integer division by zero");
    }

    if (lhs == std::numeric_limits<std::intmax_t>::min() && divisor == -1) {
        throw std::overflow_error(
            concat("Primitive: integer overflow in division of ", a, " and ", b));
    }

    return Primitive(lhs / divisor);
}

// Int64 % Int64 is the C++ remainder. If either operand is Float64 the
// result is Float64 via std::fmod (modulo by zero yields NaN). In both cases
// the result has the sign of the dividend.
Primitive operator%(const Primitive& a, const Primitive& b) {
    require_numeric(a, b, "take modulo of");

    if (primitive_is_float(a) || primitive_is_float(b)) {
        return Primitive(std::fmod(a.to_float(), b.to_float()));
    }

    std::intmax_t lhs = a.to_integer();
    std::intmax_t divisor = b.to_integer();

    if (divisor == 0) {
        throw std::runtime_error("Primitive: integer modulo by zero");
    }

    // INTMAX_MIN % -1 is 0, but traps on some hardware.
    if (divisor == -1) {
        return Primitive(std::intmax_t(0));
    }

    return Primitive(lhs % divisor);
}

// Bool is promoted to Int64 (-true == -1). String and Empty are errors.
Primitive operator-(const Primitive& a) {
    switch (a.kind()) {
        case Primitive::Kind::Float64:
            return Primitive(-a.to_float());
        case Primitive::Kind::Int64:
        case Primitive::Kind::Bool: {
            std::intmax_t result;
            if (__builtin_sub_overflow(std::intmax_t(0), a.to_integer(), &result)) {
                throw std::overflow_error("Primitive: integer overflow in negation");
            }
            return Primitive(result);
        }
        default:
            throw std::runtime_error(concat("Primitive: cannot negate ", a.kind()));
    }
}

bool operator!(const Primitive& a) {
    return !a.to_bool();
}

// Returns `a` if it is falsy, otherwise `b`.
Primitive operator&&(const Primitive& a, const Primitive& b) {
    return a.to_bool() ? b : a;
}

// Returns `a` if it is truthy, otherwise `b`.
Primitive operator||(const Primitive& a, const Primitive& b) {
    return a.to_bool() ? a : b;
}

}  // namespace kernel_launcher
