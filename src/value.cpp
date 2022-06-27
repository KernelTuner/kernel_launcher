#include "kernel_launcher/value.h"

#include <cstring>
#include <mutex>
#include <sstream>
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
    auto hash = [](const char* v) {
        size_t h = 0;
        for (; *v; v++) {
            h = h * 31 + *v;
        }
        return h;
    };

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

    return *(it->second.get());
}

std::ostream& operator<<(std::ostream& s, const TunableValue& point) {
    return s << point.to_string();
}
}  // namespace kernel_launcher