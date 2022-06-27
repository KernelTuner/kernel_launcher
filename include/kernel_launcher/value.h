#ifndef KERNEL_LAUNCHER_VALUE_H
#define KERNEL_LAUNCHER_VALUE_H

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "utils.h"

namespace kernel_launcher {

struct TunableValue;
struct CastException: std::runtime_error {
    CastException(std::string msg) : std::runtime_error(std::move(msg)) {}
};

[[noreturn]] void throw_cast_exception(const TunableValue& v, TypeInfo type);

[[noreturn]] void throw_invalid_operation_exception(
    const std::string& op,
    const TunableValue& lhs,
    const TunableValue& rhs);

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

    DataType data_type() const {
        return dtype_;
    }

    TunableValue& operator=(const TunableValue& that) {
        dtype_ = that.dtype_;
        switch (dtype_) {
            case type_int:
                int_val_ = that.int_val_;
                break;
            case type_double:
                double_val_ = that.double_val_;
                break;
            case type_bool:
                bool_val_ = that.bool_val_;
                break;
            case type_string:
                string_val_ = that.string_val_;
                break;
            case type_empty:
                break;
        }
        return *this;
    }

    bool operator==(const TunableValue& that) const {
        if (this->dtype_ != that.dtype_) {
            return false;
        }

        switch (dtype_) {
            case type_empty:
                return true;
            case type_int:
                return this->int_val_ == that.int_val_;
            case type_double:
                return this->double_val_ == that.double_val_;
            case type_bool:
                return this->bool_val_ == that.bool_val_;
            case type_string:
                return *this->string_val_ == *that.string_val_;
            default:
                return false;
        }
    }

    bool operator!=(const TunableValue& that) const {
        return !(*this == that);
    }

    bool operator<(const TunableValue& that) const {
        if (this->dtype_ != that.dtype_) {
            return this->dtype_ < that.dtype_;
        }

        switch (dtype_) {
            case type_empty:
                return true;
            case type_int:
                return this->int_val_ < that.int_val_;
            case type_double:
                return this->double_val_ < that.double_val_;
            case type_bool:
                return this->bool_val_ < that.bool_val_;
            case type_string:
                return *this->string_val_ < *that.string_val_;
            default:
                return false;
        }
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

    size_t hash() const {
        switch (dtype_) {
            case type_int:
                return std::hash<integer_type> {}(int_val_);
            case type_double:
                return std::hash<double> {}(double_val_);
            case type_string:
                return std::hash<std::string> {}(*string_val_);
            case type_bool:
                return std::hash<bool> {}(bool_val_);
            default:
                return 0;
        }
    }

    bool is_empty() const {
        return dtype_ == type_empty;
    }

    bool is_integer() const {
        return dtype_ == type_int;
    }

    integer_type to_integer() const {
        if (dtype_ == type_bool) {
            return bool_val_;
        } else if (dtype_ == type_int) {
            return int_val_;
        } else {
            throw_cast_exception(*this, type_of<integer_type>());
        }
    }

    bool is_string() const {
        return dtype_ == type_string;
    }

    std::string to_string() const {
        switch (dtype_) {
            case type_int:
                return std::to_string(int_val_);
            case type_double:
                return std::to_string(double_val_);
            case type_bool:
                return bool_val_ ? "true" : "false";
            case type_string:
                return *string_val_;
            default:
                return "";
        }
    }

    bool is_bool() const {
        return dtype_ == type_bool;
    }

    bool to_bool() const {
        switch (dtype_) {
            case type_int:
                return int_val_ != 0;
            case type_double:
                return double_val_ != 0.0;
            case type_bool:
                return bool_val_;
            case type_string:
                return !string_val_->empty();
            default:
                return false;
        }
    }

    explicit operator bool() const {
        return to_bool();
    }

    bool is_double() const {
        return dtype_ == type_double;
    }

    double to_double() const {
        switch (dtype_) {
            case type_int:
                return (double)int_val_;
            case type_double:
                return double_val_;
            case type_bool:
                return bool_val_ ? 1.0 : 0.0;
            default:
                throw_cast_exception(*this, type_of<double>());
        }
    }

    explicit operator double() const {
        return to_double();
    }

    bool is_float() const {
        return is_double();
    }

    float to_float() const {
        return (float)to_double();
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
        return is_string();
    }

    TemplateArg to(TypeIndicator<TemplateArg>) const {
        switch (dtype_) {
            case DataType::string_:
                return TemplateArg::from_string(*string_val_);
            case DataType::int_:
                return TemplateArg {int_val_};
            case DataType::bool_:
                return TemplateArg {bool_val_};
            default:
                throw_cast_exception(*this, type_of<TemplateArg>());
        }
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

inline TunableValue
operator+(const TunableValue& lhs, const TunableValue& rhs) {
    if (lhs.is_bool() && rhs.is_bool()) {
        return lhs.to_bool() || rhs.to_bool();
    } else if (lhs.is_double() || rhs.is_double()) {
        return lhs.to_double() + rhs.to_double();
    } else if (lhs.is_integer() || rhs.is_integer()) {
        return lhs.to_integer() + rhs.to_integer();
    } else if (lhs.is_string() && rhs.is_string()) {
        return lhs.to_string() + rhs.to_string();
    }

    throw_invalid_operation_exception("+", lhs, rhs);
}

inline TunableValue
operator-(const TunableValue& lhs, const TunableValue& rhs) {
    if (lhs.is_double() || rhs.is_double()) {
        return lhs.to_double() - rhs.to_double();
    } else if (lhs.is_integer() || rhs.is_integer()) {
        return lhs.to_integer() - rhs.to_integer();
    } else {
        throw_invalid_operation_exception("-", lhs, rhs);
    }
}

inline TunableValue operator-(const TunableValue& v) {
    return TunableValue(0) - v;
}

inline TunableValue
operator*(const TunableValue& lhs, const TunableValue& rhs) {
    if (lhs.is_bool() && rhs.is_bool()) {
        return lhs.to_bool() && rhs.to_bool();
    } else if (lhs.is_double() || rhs.is_double()) {
        return lhs.to_double() * rhs.to_double();
    } else if (lhs.is_integer() || rhs.is_integer()) {
        return lhs.to_integer() * rhs.to_integer();
    } else {
        throw_invalid_operation_exception("*", lhs, rhs);
    }
}

inline TunableValue
operator/(const TunableValue& lhs, const TunableValue& rhs) {
    if (lhs.is_double() || rhs.is_double()) {
        return lhs.to_double() / rhs.to_double();
    } else if (
        (lhs.is_integer() || rhs.is_integer()) && rhs.to_integer() != 0) {
        return lhs.to_integer() / rhs.to_integer();
    }

    throw_invalid_operation_exception("/", lhs, rhs);
}

inline TunableValue
operator%(const TunableValue& lhs, const TunableValue& rhs) {
    if (lhs.is_double() || rhs.is_double()) {
        return std::fmod(lhs.to_double(), rhs.to_double());
    } else if (
        (lhs.is_integer() || rhs.is_integer()) && rhs.to_integer() != 0) {
        return lhs.to_integer() % rhs.to_integer();
    }

    throw_invalid_operation_exception("%", lhs, rhs);
}

inline TunableValue
operator&&(const TunableValue& lhs, const TunableValue& rhs) {
    return lhs.to_bool() && rhs.to_bool();
}

inline TunableValue
operator||(const TunableValue& lhs, const TunableValue& rhs) {
    return lhs.to_bool() || rhs.to_bool();
}

inline TunableValue operator!(const TunableValue& v) {
    return !v.to_bool();
}

struct TunableParam {
  private:
    struct Impl {
        friend TunableParam;

        Impl(
            std::string name,
            std::vector<TunableValue> values,
            TunableValue default_value) :
            name_(std::move(name)),
            values_(std::move(values)),
            default_value_(std::move(default_value)) {}

      private:
        std::string name_;
        std::vector<TunableValue> values_;
        TunableValue default_value_;
    };

  public:
    TunableParam(
        std::string name,
        std::vector<TunableValue> values,
        TunableValue default_value) {
        inner_ = std::make_shared<Impl>(
            std::move(name),
            std::move(values),
            std::move(default_value));
    }

    const std::string& name() const {
        return inner_->name_;
    }

    size_t hash() const {
        return (size_t)inner_.get();
    }

    const TunableValue& default_value() const {
        return inner_->default_value_;
    }

    const std::vector<TunableValue>& values() const {
        return inner_->values_;
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
