#include <iomanip>
#include <sstream>
#include <chrono>
#include <unistd.h>

#include "kernel_launcher/fs.h"
#include "kernel_launcher/expr.h"
#include "kernel_launcher/wisdom.h"
#include "nlohmann/json.hpp"

namespace kernel_launcher {
using json = nlohmann::json;

std::string sanitize_tuning_key(std::string key) {
    std::stringstream output;
    bool last_was_space = false;

    for (char c : key) {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
            || (c >= '0' && c <= '9') || c == ' ' || c == '_' || c == '-') {
            output << c;
        } else {
            output << '$' << std::hex << std::setw(2) << std::setfill('0')
                   << ((unsigned char)c);
        }
    }

    return output.str();
}

static json value_to_json(const TunableValue& expr) {
    switch (expr.data_type()) {
        case TunableValue::type_int:
            return expr.to_integer();
        case TunableValue::type_double:
            return expr.to_double();
        case TunableValue::type_string:
            return expr.to_string();
        case TunableValue::type_bool:
            return expr.to_bool();
        default:
            return nullptr;
    }
}

//struct SelectExpr: BaseExpr {

static json expr_to_json(const BaseExpr& expr) {
    if (const ScalarExpr* v = dynamic_cast<const ScalarExpr*>(&expr)) {
        return value_to_json(v->value());
    }

    if (const SharedExpr* v = dynamic_cast<const SharedExpr*>(&expr)) {
        return expr_to_json(v->inner());
    }

    std::string op;
    std::vector<json> operands;

    if (const ParamExpr* pe = dynamic_cast<const ParamExpr*>(&expr)) {
        op = "parameter";
        operands.push_back(pe->parameter().name());
    } else if (const UnaryExpr* ue = dynamic_cast<const UnaryExpr*>(&expr)) {
        op = ue->op_name();
        operands.push_back(expr_to_json(ue->operand()));
    } else if (const BinaryExpr* be = dynamic_cast<const BinaryExpr*>(&expr)) {
        op = be->op_name();
        operands.push_back(expr_to_json(be->left_operand()));
        operands.push_back(expr_to_json(be->right_operand()));
    } else if (const SelectExpr* se = dynamic_cast<const SelectExpr*>(&expr)) {
        op = "select";
        operands.push_back(expr_to_json(se->condition()));

        for (const auto& p : se->operands()) {
            operands.push_back(expr_to_json(p));
        }
    } else {
        throw std::runtime_error(
            "could not serialize expression: " + expr.to_string());
    }

    return {{"operator", op}, {"operands", operands}};
}

template <typename C>
static std::vector<json> expr_list_to_json(C collection) {
    std::vector<json> result;

    for (const auto& entry: collection) {
        result.push_back(expr_to_json(entry));
    }

    return result;
}

static json environment_json() {
    int driver_version = -1, runtime_version = -1;
    cudaDriverGetVersion(&driver_version);
    cudaRuntimeGetVersion(&runtime_version);

    int device;
    cuda_safe_call(cudaGetDevice(&device));

    cudaDeviceProp prop;
    cuda_safe_call(cudaGetDeviceProperties(&prop, device));

    std::stringstream uuid;
    uuid << std::hex << std::setfill('0') << std::setw(2);
    for (size_t i = 0; i < 16; i++)
    {
        if (i % 2 == 0 && i >= 4 && i <= 10) uuid << "-";
        uuid << (unsigned int) (unsigned char) prop.uuid.bytes[i];
    }

    char hostname[256] = {0};
    gethostname(hostname, sizeof(hostname));

    std::time_t t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::stringstream date;
    date << std::put_time( std::localtime( &t ), "%FT%T%z" );

    nlohmann::json env;
    env["host_name"] = hostname;
    env["date"] = date.str();
    env["runtime_version"] = runtime_version;
    env["driver_version"] = driver_version;
    env["device_name"] = prop.name;
    env["device_ordinal"] = device;
    env["device_uuid"] = uuid.str();
    return env;
}

struct KernelBuilderSerializerHack {
    static json builder_to_json(const KernelBuilder& builder) {

        std::vector<json> restrictions;
        for (auto e: expr_list_to_json(builder.restrictions_)) {
            restrictions.push_back(e);
        }
        for (auto e: expr_list_to_json(builder.assertions_)) {
            restrictions.push_back(e);
        }

        std::unordered_map<std::string, json> defines;
        for (const auto& p: builder.defines_) {
            defines[p.first] = expr_to_json(p.second);
        }

        json result;
        result["kernel_name"] = builder.kernel_name_;
        result["kernel_file"] = builder.kernel_source_.file_name();
        result["compile_flags"] = expr_list_to_json(builder.compile_flags_);
        result["block_size"] = expr_list_to_json(builder.block_size_);
        result["grid_divisors"] = expr_list_to_json(builder.grid_divisors_);
        result["shared_memory"] = expr_to_json(builder.shared_mem_);
        result["template_args"] = expr_list_to_json(builder.template_args_);
        result["defines"] = std::move(defines);
        result["restrictions"] = std::move(restrictions);
        result["config"] = nullptr;
        result["environment"] = environment_json();

        return result;
    }

    static Config json_to_config(const json& config, const KernelBuilder& builder) {
        Config result;

        for (const auto& key: builder.params_) {
            json value = config[key.name()];
            TunableValue found_value;
            bool is_valid = false;

            for (const TunableValue& v: key.values()) {
                if (value_to_json(v) == value) {
                    is_valid = true;
                    found_value = v;
                }
            }

            if (!is_valid) {
                throw std::runtime_error("key not found: " + key.name());
            }

            result.insert(key, found_value);
        }

        return result;
    }
};

static json builder_to_json(const KernelBuilder& builder) {
    return KernelBuilderSerializerHack::builder_to_json(builder);
}

static Config json_to_config(const json& config, const KernelBuilder& builder) {
    return KernelBuilderSerializerHack::json_to_config(config, builder);
}

WisdomResult read_wisdom_file(
    const std::string& path,
    const std::string& tuning_key,
    const KernelBuilder& builder,
    Config& config) {
    std::string file_name = path + "/" + sanitize_tuning_key(tuning_key) + ".json";
    std::vector<char> content;

    if (!read_file(file_name, content)) {
        return WisdomResult::NotFound;
    }

    try {
        json input = json::parse(content.begin(), content.end())["config"];
        if (!input.is_object()) {
            return WisdomResult::Untuned;
        }

        config = json_to_config(input, builder);
        return WisdomResult::Success;
    } catch(const std::exception& e) {
        // FIXME: log error
        return WisdomResult::IoError;
    }
}

void write_wisdom_file(
    const std::string& path,
    const std::string& tuning_key,
    const KernelBuilder& builder,
    const std::string& data_dir,
    const std::vector<const KernelArg*>& inputs,
    const std::vector<const KernelArg*>& outputs = {}) {


}

}  // namespace kernel_launcher