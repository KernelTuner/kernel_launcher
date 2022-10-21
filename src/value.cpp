#include "kernel_launcher/value.h"

#include <atomic>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>

namespace kernel_launcher {

static const char* data_type_name(TunableValue::DataType dtype) {
    switch (dtype) {
        case TunableValue::DataType::int_:
            return "int";
        case TunableValue::DataType::double_:
            return "double";
        case TunableValue::DataType::string_:
            return "string";
        case TunableValue::DataType::bool_:
            return "bool";
        default:
            return "empty";
    }
}

[[noreturn]] void throw_cast_exception(const TunableValue& v, TypeInfo type) {
    std::stringstream ss;
    ss << "cannot cast value \"" << v.to_string()
       << "\" (data type: " << data_type_name(v.data_type()) << ") to type "
       << type.name();
    throw CastException(ss.str());
}

[[noreturn]] void throw_invalid_operation_exception(
    const std::string& op,
    const TunableValue& lhs,
    const TunableValue& rhs) {
    std::stringstream ss;
    ss << "cannot apply operator \"" << op << "\" to values \""
       << lhs.to_string()
       << "\" (data type: " << data_type_name(lhs.data_type()) << ") and "
       << rhs.to_string()
       << "\" (data type: " << data_type_name(rhs.data_type()) << ")";
    throw CastException(ss.str());
}

const std::string& intern_string(const char* input) {
    auto equal = [](const char* a, const char* b) { return strcmp(a, b) == 0; };
    auto hash = [](const char* v) { return hash_string(v, ::strlen(v)); };

    static std::mutex lock;
    static std::unordered_map<
        const char*,
        std::unique_ptr<std::string>,
        decltype(hash),
        decltype(equal)>
        table(32, hash, equal);

    std::lock_guard<std::mutex> guard(lock);

    auto it = table.find(input);
    if (it == table.end()) {
        auto value = std::make_unique<std::string>(input);
        auto key = value->c_str();

        auto result = table.insert(std::make_pair(key, std::move(value)));
        it = result.first;
    }

    return *(it->second);
}

TunableValue& TunableValue::operator=(const TunableValue& that) {
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

std::ostream& operator<<(std::ostream& s, const TunableValue& point) {
    return s << point.to_string();
}

size_t TunableValue::hash() const {
    int64_t v = 0;

    switch (dtype_) {
        case type_int:
            return std::hash<integer_type> {}(int_val_);
        case type_bool:
            return std::hash<integer_type> {}(bool_val_);
        case type_double:
            if (safe_double_to_int64(double_val_, v)) {
                return std::hash<integer_type> {}(v);
            } else {
                return 0;
            }
        case type_string:
            return std::hash<std::string> {}(*string_val_);
        default:
            return 0;
    }
}

bool TunableValue::operator==(const TunableValue& that) const {
    DataType l = this->dtype_;
    DataType r = that.dtype_;

    if (l == type_empty && r == type_empty) {
        return true;
    }

    if (l == type_string && r == type_string) {
        return *this->string_val_ == *that.string_val_;
    }

    if ((l == type_bool || l == type_int)
        && (r == type_bool || r == type_int)) {
        return this->to_integer() == that.to_integer();
    }

    if ((l == type_bool || l == type_double)
        && (r == type_bool || r == type_double)) {
        return this->to_double() == that.to_double();
    }

    if (l == type_double && r == type_int) {
        integer_type result;

        if (safe_double_to_int64(this->double_val_, result)) {
            return result == that.int_val_;
        }
    }

    if (l == type_int && r == type_double) {
        integer_type result;

        if (safe_double_to_int64(that.double_val_, result)) {
            return this->int_val_ == result;
        }
    }

    // FIXME: add more cases?

    return false;
}

bool TunableValue::operator<(const TunableValue& that) const {
    DataType l = this->dtype_;
    DataType r = that.dtype_;

    if (l == type_empty && r == type_empty) {
        return false;
    }

    if (l == type_string && r == type_string) {
        return *this->string_val_ < *that.string_val_;
    }

    if ((l == type_bool || l == type_int)
        && (r == type_bool || r == type_int)) {
        return this->to_integer() < that.to_integer();
    }

    if ((l == type_bool || l == type_double || l == type_int)
        && (r == type_bool || r == type_double || r == type_int)) {
        return this->to_double() < that.to_double();
    }

    return l < r;
}

bool TunableValue::to_bool() const {
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

TunableValue::integer_type TunableValue::to_integer() const {
    if (dtype_ == type_bool) {
        return bool_val_;
    } else if (dtype_ == type_int) {
        return int_val_;
    } else if (dtype_ == type_double) {
        int64_t result;

        if (safe_double_to_int64(double_val_, result)) {
            return result;
        }
    }

    throw_cast_exception(*this, type_of<integer_type>());
}

std::string TunableValue::to_string() const {
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

double TunableValue::to_double() const {
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

float TunableValue::to_float() const {
    return (float)to_double();
}

TemplateArg TunableValue::to_template_arg() const {
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

TunableValue operator+(const TunableValue& lhs, const TunableValue& rhs) {
    if (lhs.is_bool() && rhs.is_bool()) {
        return lhs.to_bool() || rhs.to_bool();
    } else if (lhs.is_double() || rhs.is_double()) {
        return lhs.to_double() + rhs.to_double();
    } else if (lhs.is_string() && rhs.is_string()) {
        return lhs.to_string() + rhs.to_string();
    } else if (lhs.is_integer() || rhs.is_integer()) {
        TunableValue::integer_type out;
        if (safe_int64_add(lhs.to_integer(), rhs.to_integer(), out)) {
            return out;
        }
    }

    throw_invalid_operation_exception("+", lhs, rhs);
}

TunableValue operator+(const TunableValue& v) {
    if (v.is_bool() || v.is_integer() || v.is_double()) {
        return v;
    }

    throw_invalid_operation_exception("+", 0, v);
}

TunableValue operator-(const TunableValue& lhs, const TunableValue& rhs) {
    if (lhs.is_double() || rhs.is_double()) {
        return lhs.to_double() - rhs.to_double();
    } else if (lhs.is_integer() || rhs.is_integer()) {
        TunableValue::integer_type out;
        if (safe_int64_sub(lhs.to_integer(), rhs.to_integer(), out)) {
            return out;
        }
    }

    throw_invalid_operation_exception("-", lhs, rhs);
}

TunableValue operator-(const TunableValue& v) {
    if (v.is_double()) {
        return -v.to_double();
    } else if (v.is_integer()) {
        TunableValue::integer_type out;
        if (safe_int64_sub(0, v.to_integer(), out)) {
            return out;
        }
    }

    throw_invalid_operation_exception("-", 0, v);
}

TunableValue operator*(const TunableValue& lhs, const TunableValue& rhs) {
    if (lhs.is_bool() && rhs.is_bool()) {
        return lhs.to_bool() && rhs.to_bool();
    } else if (lhs.is_double() || rhs.is_double()) {
        return lhs.to_double() * rhs.to_double();
    } else if (lhs.is_integer() || rhs.is_integer()) {
        TunableValue::integer_type out;
        if (safe_int64_mul(lhs.to_integer(), rhs.to_integer(), out)) {
            return out;
        }
    }

    throw_invalid_operation_exception("*", lhs, rhs);
}

TunableValue operator/(const TunableValue& lhs, const TunableValue& rhs) {
    if (lhs.is_double() || rhs.is_double()) {
        return lhs.to_double() / rhs.to_double();
    } else if (lhs.is_integer() || rhs.is_integer()) {
        TunableValue::integer_type out;
        if (safe_int64_div(lhs.to_integer(), rhs.to_integer(), out)) {
            return out;
        }
    }

    throw_invalid_operation_exception("/", lhs, rhs);
}

TunableValue operator%(const TunableValue& lhs, const TunableValue& rhs) {
    if (lhs.is_double() || rhs.is_double()) {
        return std::fmod(lhs.to_double(), rhs.to_double());
    } else if (
        (lhs.is_integer() || rhs.is_integer()) && rhs.to_integer() != 0) {
        return lhs.to_integer() % rhs.to_integer();
    }

    throw_invalid_operation_exception("%", lhs, rhs);
}

TunableValue operator&&(const TunableValue& lhs, const TunableValue& rhs) {
    return lhs.to_bool() && rhs.to_bool();
}

TunableValue operator||(const TunableValue& lhs, const TunableValue& rhs) {
    return lhs.to_bool() || rhs.to_bool();
}

TunableValue operator!(const TunableValue& v) {
    return !v.to_bool();
}

static std::atomic<uint64_t> global_var_counter;
Variable::Variable() : id_(global_var_counter += 1) {}

TunableParam::TunableParam(
    std::string name,
    std::vector<TunableValue> values,
    std::vector<double> priors,
    TunableValue default_value) {
    if (name.empty()) {
        throw std::runtime_error("name cannot be empty");
    }

    bool found = false;
    for (const auto& p : values) {
        found |= p == default_value;
    }

    if (!found) {
        throw std::runtime_error(
            "default value for parameter " + name
            + " must be a valid value for this parameter");
    }

    if (priors.size() != values.size()) {
        throw std::runtime_error("invalid number of priors");
    }

    for (double v : priors) {
        if (!std::isfinite(v) || v < 0) {
            throw std::runtime_error(
                "priors must be non-negative finite numbers");
        }
    }

    inner_ = std::make_shared<Impl>(
        std::move(name),
        std::move(values),
        std::move(priors),
        std::move(default_value));
}
}  // namespace kernel_launcher