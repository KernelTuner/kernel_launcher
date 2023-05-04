#ifndef KERNEL_LAUNCHER_VALUE_H
#define KERNEL_LAUNCHER_VALUE_H

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernel_launcher/utils.h"

namespace kernel_launcher {

struct Value;
struct CastException: std::runtime_error {
    CastException(std::string msg) : std::runtime_error(std::move(msg)) {}
};

using int64_t = std::int64_t;
const std::string& intern_string(const char* input);

/**
 * Represents a value that can be one of the following five types:
 *
 * * integer (`int64_t`).
 * * floating-point value (`double`).
 * * boolean (`bool`).
 * * string (`std::string`).
 * * empty
 *
 * It is possible to convert a C++ value to a `Value` instance and back to
 * a C++ value.
 *
 * `Value` also overloads several operators such as addition and multiplication.
 */
struct Value {
  private:
    template<typename T>
    struct TypeIndicator {};

  public:
    using integer_type = int64_t;
    using float_type = double;
    using bool_type = bool;
    using string_type = std::string;

    enum class DataType { empty_, int_, double_, bool_, string_ };

    static constexpr DataType type_empty = DataType::empty_;
    static constexpr DataType type_int = DataType::int_;
    static constexpr DataType type_double = DataType::double_;
    static constexpr DataType type_string = DataType::string_;
    static constexpr DataType type_bool = DataType::bool_;

    Value() = default;
    Value(std::nullptr_t) : Value() {}

    Value(const Value& val) {
        *this = val;
    }

    Value(Value&& val) noexcept {
        *this = std::move(val);
    }

    Value(const char* value) :
        dtype_(type_string),
        string_val_(&intern_string(value)) {}

    Value(const std::string& v) : Value(v.c_str()) {}
    Value(const TemplateArg& arg) : Value(arg.to_string()) {}
    Value(double v) : dtype_(type_double), double_val_(v) {}
    Value(bool v) : dtype_(type_bool), bool_val_(v) {}

    Value& operator=(const Value& that);
    bool operator==(const Value& that) const;
    bool operator<(const Value& that) const;
    bool operator!=(const Value& that) const {
        return !(*this == that);
    }

    bool operator>(const Value& that) const {
        return that < *this;
    }

    bool operator<=(const Value& that) const {
        return *this == that || *this < that;
    }

    bool operator>=(const Value& that) const {
        return that <= *this;
    }

    friend std::ostream& operator<<(std::ostream& s, const Value& point);

    size_t hash() const;

    DataType data_type() const {
        return dtype_;
    }

    /**
     * Returns `true` if this `Value` is empty.
     */
    bool is_empty() const {
        return dtype_ == type_empty;
    }

    /**
     * Returns `true` if this `Value` is an integer.
     */
    bool is_integer() const {
        return dtype_ == type_int;
    }

    /**
     * Returns `true` if this `Value` is a string.
     */
    bool is_string() const {
        return dtype_ == type_string;
    }

    /**
     * Returns `true` if this `Value` is a boolean.
     */
    bool is_bool() const {
        return dtype_ == type_bool;
    }

    /**
     * Returns `true` if this `Value` is a floating-point number.
     */
    bool is_double() const {
        return dtype_ == type_double;
    }

    /**
     * Convert this value to an integer. The data type of this value should
     * be `integer`, `double`, or `boolean`.
     */
    integer_type to_integer() const;

    /**
     * Convert this value to a string.
     */
    std::string to_string() const;

    /**
     * Convert this value to a `bool`.
     */
    bool to_bool() const;

    /**
     * Convert this value to a `double`.
     */
    double to_double() const;

    /**
     * Convert this value to a `float`.
     */
    float to_float() const;

    /**
     * Convert this value to a `TemplateArg`.
     */
    TemplateArg to_template_arg() const;

    explicit operator bool() const {
        return to_bool();
    }

    explicit operator double() const {
        return to_double();
    }

    explicit operator float() const {
        return to_float();
    }

    Value round() const;
    Value floor() const;
    Value ceil() const;

    /**
     * Returns `true` if this value is convertible to an instance of `T`.
     */
    template<typename T>
    bool is() const {
        return is(TypeIndicator<T> {});
    }

    /**
     * Convert this `Value` to an instance of type `T` if this is possible.
     */
    template<typename T>
    T to() const {
        return to(TypeIndicator<T> {});
    }

  private:
    bool is(TypeIndicator<Value>) const {
        return true;
    }

    Value to(TypeIndicator<Value>) const {
        return *this;
    }

    bool is(TypeIndicator<TemplateArg>) const {
        return is_string() || is_integer() || is_bool();
    }

    TemplateArg to(TypeIndicator<TemplateArg>) const {
        return to_template_arg();
    }

    bool is(TypeIndicator<bool>) const {
        return is_bool();
    }

    bool to(TypeIndicator<bool>) const {
        return to_bool();
    }

    bool is(TypeIndicator<double>) const {
        return is_double();
    }

    double to(TypeIndicator<double>) const {
        return to_double();
    }

    bool is(TypeIndicator<float>) const {
        return is_double();
    }

    float to(TypeIndicator<float>) const {
        return static_cast<float>(to_double());
    }

    bool is(TypeIndicator<std::string>) const {
        return is_string();
    }

    std::string to(TypeIndicator<std::string>) const {
        return to_string();
    }

#define KERNEL_LAUNCHER_INTEGER_FUNS(type_name, human_name)               \
  public:                                                                 \
    int is_##human_name() const {                                         \
        return dtype_ == type_bool                                        \
            || (dtype_ == type_int && in_range<type_name>(int_val_));     \
    }                                                                     \
                                                                          \
    type_name to_##human_name() const {                                   \
        if (dtype_ == type_bool) {                                        \
            return bool_val_ ? 1 : 0;                                     \
        } else if (dtype_ == type_int && in_range<type_name>(int_val_)) { \
            return (type_name)int_val_;                                   \
        } else {                                                          \
            throw std::runtime_error("failed to convert type");           \
        }                                                                 \
    }

#define KERNEL_LAUNCHER_INTEGER_IMPL(type_name, human_name)     \
    KERNEL_LAUNCHER_INTEGER_FUNS(type_name, human_name)         \
                                                                \
    Value(type_name i) :                                        \
        dtype_(type_int),                                       \
        int_val_(static_cast<integer_type>(i)) {                \
        if (!in_range<integer_type>(i)) {                       \
            throw std::runtime_error("failed to convert type"); \
        }                                                       \
    }                                                           \
                                                                \
  private:                                                      \
    bool is(TypeIndicator<type_name>) const {                   \
        return this->is_##human_name();                         \
    }                                                           \
    type_name to(TypeIndicator<type_name>) const {              \
        return this->to_##human_name();                         \
    }

    // char is the only type were `char`, `signed char`, and `unsigned char` are different types
    KERNEL_LAUNCHER_INTEGER_IMPL(char, char)
    KERNEL_LAUNCHER_INTEGER_IMPL(signed char, signed_char)
    KERNEL_LAUNCHER_INTEGER_IMPL(unsigned char, unsigned_char)

    // Other integer types
    KERNEL_LAUNCHER_INTEGER_IMPL(short, short)
    KERNEL_LAUNCHER_INTEGER_IMPL(int, int)
    KERNEL_LAUNCHER_INTEGER_IMPL(long, long)
    KERNEL_LAUNCHER_INTEGER_IMPL(long long, longlong)
    KERNEL_LAUNCHER_INTEGER_IMPL(unsigned short, unsigned_short)
    KERNEL_LAUNCHER_INTEGER_IMPL(unsigned int, unsigned_int)
    KERNEL_LAUNCHER_INTEGER_IMPL(unsigned long, unsigned_long)
    KERNEL_LAUNCHER_INTEGER_IMPL(unsigned long long, unsigned_longlong)

    // types from stdint.h
    KERNEL_LAUNCHER_INTEGER_FUNS(std::int8_t, int8)
    KERNEL_LAUNCHER_INTEGER_FUNS(std::int16_t, int16)
    KERNEL_LAUNCHER_INTEGER_FUNS(std::int32_t, int32)
    KERNEL_LAUNCHER_INTEGER_FUNS(std::int64_t, int64)
    KERNEL_LAUNCHER_INTEGER_FUNS(std::uint8_t, uint8)
    KERNEL_LAUNCHER_INTEGER_FUNS(std::uint16_t, uint16)
    KERNEL_LAUNCHER_INTEGER_FUNS(std::uint32_t, uint32)
    KERNEL_LAUNCHER_INTEGER_FUNS(std::uint64_t, uint64)
    KERNEL_LAUNCHER_INTEGER_FUNS(std::size_t, size_t)

#undef KERNEL_LAUNCHER_INTEGER_IMPL
#undef KERNEL_LAUNCHER_INTEGER_FUNS

  private:
    DataType dtype_ = type_empty;
    union {
        integer_type int_val_;
        double double_val_;
        bool bool_val_;
        const std::string* string_val_;
    };
};

Value operator+(const Value& lhs, const Value& rhs);
Value operator+(const Value& v);
Value operator-(const Value& lhs, const Value& rhs);
Value operator-(const Value& v);
Value operator*(const Value& lhs, const Value& rhs);
Value operator/(const Value& lhs, const Value& rhs);
Value operator%(const Value& lhs, const Value& rhs);
Value operator&&(const Value& lhs, const Value& rhs);
Value operator||(const Value& lhs, const Value& rhs);
Value operator!(const Value& v);

struct Variable {
    virtual ~Variable() = default;
    virtual bool equals(const Variable& that) const = 0;
    virtual size_t hash() const = 0;

    bool operator==(const Variable& that) const {
        return equals(that);
    }

    bool operator!=(const Variable& that) const {
        return !equals(that);
    }
};

/**
 * A tunable parameter that is returned by `ConfigSpace`.
 */
struct TunableParam: Variable {
  private:
    struct Impl {
        friend ::kernel_launcher::TunableParam;

        Impl(
            std::string name,
            std::vector<Value> values,
            std::vector<double> priors,
            Value default_value) :
            name_(std::move(name)),
            values_(std::move(values)),
            priors_(std::move(priors)),
            default_value_(std::move(default_value)) {}

      private:
        std::string name_;
        std::vector<Value> values_;
        std::vector<double> priors_;
        Value default_value_;
    };

  public:
    TunableParam() = delete;
    TunableParam(const TunableParam&) = default;
    TunableParam& operator=(const TunableParam&) = default;

    // We want TunableParam to be copyable but not movable. These two
    // statements cause compile errors. Instead, simply forward the move
    // operations to the copy operations.
    // TunableParam(TunableParam&&) = delete;
    // TunableParam& operator=(TunableParam&&) = delete;
    TunableParam(TunableParam&& v) : TunableParam(v) {}
    TunableParam& operator=(TunableParam&& v) noexcept {
        return *this = v;
    }

    /**
     * Construct a new parameter.
     */
    TunableParam(
        std::string name,
        std::vector<Value> values,
        std::vector<double> priors,
        Value default_value);

    /**
     * Construct a new parameter.
     */
    TunableParam(
        std::string name,
        std::vector<Value> values,
        Value default_value) :
        TunableParam(
            std::move(name),
            std::move(values),
            std::vector<double>(values.size(), 1.0),
            std::move(default_value)) {}

    /**
     * The name of this parameter.
     */
    const std::string& name() const {
        return inner_->name_;
    }

    /**
     * The default value of this parameter.
     */
    const Value& default_value() const {
        return inner_->default_value_;
    }

    /**
     * The allowed values for this parameter.
     */
    const std::vector<Value>& values() const {
        return inner_->values_;
    }

    /**
     * Checks if `needle` is in the values returned by `values()`.
     */
    bool has_value(const Value& needle) const {
        bool found = false;

        // Maybe binary search?
        for (const auto& v : inner_->values_) {
            if (v == needle) {
                found = true;
            }
        }

        return found;
    }

    const std::vector<double>& priors() const {
        return inner_->priors_;
    }

    /**
     * Short-hand for `values().at(i)`.
     */
    const Value& at(size_t i) const {
        return values().at(i);
    }

    /**
     * Short-hand for `values().at(i)`.
     */
    const Value& operator[](size_t i) const {
        return at(i);
    }

    /**
     * Short-hand for `values().size()`.
     */
    size_t size() const {
        return values().size();
    }

    bool operator==(const TunableParam& that) const {
        return inner_.get() == that.inner_.get();
    }

    bool operator!=(const TunableParam& that) const {
        return !(*this == that);
    }

    size_t hash() const override {
        return (size_t)inner_.get();
    }

    bool equals(const Variable& v) const override {
        if (auto that = dynamic_cast<const TunableParam*>(&v)) {
            return *that == *this;
        } else {
            return false;
        }
    }

  private:
    std::shared_ptr<Impl> inner_;
};

}  // namespace kernel_launcher

namespace std {
template<>
struct hash<kernel_launcher::Value> {
    std::size_t operator()(const kernel_launcher::Value& v) const {
        return v.hash();
    }
};

template<>
struct hash<kernel_launcher::TunableParam> {
    std::size_t operator()(const kernel_launcher::TunableParam& v) const {
        return v.hash();
    }
};

template<>
struct hash<kernel_launcher::Variable> {
    std::size_t operator()(const kernel_launcher::Variable& v) const {
        return size_t {v.hash()};
    }
};
}  // namespace std

#endif  //KERNEL_LAUNCHER_VALUE_H
