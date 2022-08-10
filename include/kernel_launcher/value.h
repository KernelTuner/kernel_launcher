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

struct TunableValue;
struct CastException: std::runtime_error {
    CastException(std::string msg) : std::runtime_error(std::move(msg)) {}
};

using int64_t = std::int64_t;
const std::string& intern_string(const char* input);

struct TunableValue {
    template<typename T>
    struct TypeIndicator {};
    using integer_type = int64_t;

    enum class DataType {
        empty_,
        int_,
        double_,
        string_,
        bool_,
    };

    static constexpr DataType type_empty = DataType::empty_;
    static constexpr DataType type_int = DataType::int_;
    static constexpr DataType type_double = DataType::double_;
    static constexpr DataType type_string = DataType::string_;
    static constexpr DataType type_bool = DataType::bool_;

    TunableValue() = default;

    TunableValue(const TunableValue& val) {
        *this = val;
    }

    TunableValue(TunableValue&& val) noexcept {
        *this = std::move(val);
    }

    TunableValue(const char* value) :
        dtype_(type_string),
        string_val_(&intern_string(value)) {}

    TunableValue(const std::string& v) : TunableValue(v.c_str()) {}
    TunableValue(const TemplateArg& arg) : TunableValue(arg.to_string()) {}
    TunableValue(double v) : dtype_(type_double), double_val_(v) {}
    TunableValue(bool v) : dtype_(type_bool), bool_val_(v) {}

    TunableValue& operator=(const TunableValue& that);
    bool operator==(const TunableValue& that) const;
    bool operator<(const TunableValue& that) const;
    bool operator!=(const TunableValue& that) const {
        return !(*this == that);
    }

    bool operator>(const TunableValue& that) const {
        return that < *this;
    }

    bool operator<=(const TunableValue& that) const {
        return *this == that || *this < that;
    }

    bool operator>=(const TunableValue& that) const {
        return that <= *this;
    }

    friend std::ostream& operator<<(std::ostream& s, const TunableValue& point);

    size_t hash() const;

    DataType data_type() const {
        return dtype_;
    }

    bool is_empty() const {
        return dtype_ == type_empty;
    }

    bool is_integer() const {
        return dtype_ == type_int;
    }

    bool is_string() const {
        return dtype_ == type_string;
    }

    bool is_bool() const {
        return dtype_ == type_bool;
    }

    bool is_double() const {
        return dtype_ == type_double;
    }

    bool is_float() const {
        return is_double();
    }

    integer_type to_integer() const;
    std::string to_string() const;
    bool to_bool() const;
    double to_double() const;
    float to_float() const;
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

    template<typename T>
    bool is() const {
        return is(TypeIndicator<T> {});
    }

    template<typename T>
    T to() const {
        return to(TypeIndicator<T> {});
    }

  private:
    bool is(TypeIndicator<TunableValue>) const {
        return true;
    }

    TunableValue to(TypeIndicator<TunableValue>) const {
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
            throw std::runtime_error("");                                 \
        }                                                                 \
    }

#define KERNEL_LAUNCHER_INTEGER_IMPL(type_name, human_name)      \
    KERNEL_LAUNCHER_INTEGER_FUNS(type_name, human_name)          \
                                                                 \
    TunableValue(type_name i) : dtype_(type_int), int_val_(i) {} \
                                                                 \
  private:                                                       \
    bool is(TypeIndicator<type_name>) const {                    \
        return this->is_##human_name();                          \
    }                                                            \
    type_name to(TypeIndicator<type_name>) const {               \
        return this->to_##human_name();                          \
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

TunableValue operator+(const TunableValue& lhs, const TunableValue& rhs);
TunableValue operator+(const TunableValue& v);
TunableValue operator-(const TunableValue& lhs, const TunableValue& rhs);
TunableValue operator-(const TunableValue& v);
TunableValue operator*(const TunableValue& lhs, const TunableValue& rhs);
TunableValue operator/(const TunableValue& lhs, const TunableValue& rhs);
TunableValue operator%(const TunableValue& lhs, const TunableValue& rhs);
TunableValue operator&&(const TunableValue& lhs, const TunableValue& rhs);
TunableValue operator||(const TunableValue& lhs, const TunableValue& rhs);
TunableValue operator!(const TunableValue& v);

struct TunableParam {
  private:
    struct Impl {
        friend ::kernel_launcher::TunableParam;

        Impl(
            std::string name,
            std::vector<TunableValue> values,
            std::vector<double> priors,
            TunableValue default_value) :
            name_(std::move(name)),
            values_(std::move(values)),
            priors_(std::move(priors)),
            default_value_(std::move(default_value)) {}

      private:
        std::string name_;
        std::vector<TunableValue> values_;
        std::vector<double> priors_;
        TunableValue default_value_;
    };

  public:
    TunableParam(
        std::string name,
        std::vector<TunableValue> values,
        std::vector<double> priors,
        TunableValue default_value);

    TunableParam(
        std::string name,
        std::vector<TunableValue> values,
        TunableValue default_value) :
        TunableParam(
            std::move(name),
            std::move(values),
            std::vector<double>(values.size(), 1.0),
            std::move(default_value)) {}

    const std::string& name() const {
        return inner_->name_;
    }

    const TunableValue& default_value() const {
        return inner_->default_value_;
    }

    const std::vector<TunableValue>& values() const {
        return inner_->values_;
    }

    bool has_value(const TunableValue& needle) const {
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

    const TunableValue& at(size_t i) const {
        return values().at(i);
    }

    const TunableValue& operator[](size_t i) const {
        return at(i);
    }

    size_t size() const {
        return values().size();
    }

    size_t hash() const {
        return (size_t)inner_.get();
    }

    bool operator==(const TunableParam& that) const {
        return inner_.get() == that.inner_.get();
    }

    bool operator!=(const TunableParam& that) const {
        return !(*this == that);
    }

  private:
    std::shared_ptr<Impl> inner_;
};

}  // namespace kernel_launcher

namespace std {
template<>
struct hash<kernel_launcher::TunableValue> {
    std::size_t operator()(const kernel_launcher::TunableValue& v) const {
        return v.hash();
    }
};

template<>
struct hash<kernel_launcher::TunableParam> {
    std::size_t operator()(const kernel_launcher::TunableParam& v) const {
        return v.hash();
    }
};
}  // namespace std

#endif  //KERNEL_LAUNCHER_VALUE_H
