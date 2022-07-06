#include "kernel_launcher/wisdom.h"

#include <unistd.h>

#include <chrono>
#include <iomanip>
#include <random>
#include <sstream>

#include "kernel_launcher/expr.h"
#include "kernel_launcher/fs.h"
#include "nlohmann/json.hpp"

namespace kernel_launcher {
using json = nlohmann::json;

std::string sanitize_tuning_key(std::string key) {
    std::stringstream output;

    for (char c : key) {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
            || (c >= '0' && c <= '9') || c == ' ' || c == '_' || c == '-') {
            output << c;
        } else {
            output << '$' << std::hex << std::setw(2) << std::setfill('0')
                   << ((unsigned int)(unsigned char)c);
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

template<typename C>
static std::vector<json> expr_list_to_json(C collection) {
    std::vector<json> result;

    for (const auto& entry : collection) {
        result.push_back(expr_to_json(entry));
    }

    return result;
}

static json environment_json(const std::string& tuning_key, CudaDevice device) {
    int driver_version = -1, runtime_version = -1;
    cuDriverGetVersion(&driver_version);  // ignore errors
    cudaRuntimeGetVersion(&runtime_version);

    int nvrtc_version = NvrtcCompiler::version();

    char hostname[256] = {0};
    gethostname(hostname, sizeof(hostname));

    std::time_t t =
        std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::stringstream date;
    date << std::put_time(std::localtime(&t), "%FT%T%z");

    nlohmann::json env;
    env["key"] = tuning_key;
    env["host_name"] = hostname;
    env["date"] = date.str();
    env["runtime_version"] = runtime_version;
    env["driver_version"] = driver_version;
    env["nvrtc_version"] = nvrtc_version;
    env["device_name"] = device.name();
    env["device_ordinal"] = device.ordinal();
    env["device_uuid"] = device.uuid();
    return env;
}

static void check_environment_json(
    const json& env,
    const std::string& tuning_key,
    CudaDevice device) {
    // The key MUST match. All other checks are just warnings
    if (tuning_key != env["key"]) {
        throw std::runtime_error("expecting tuning key " + tuning_key + ", but"
                                 " found key " + env["key"].dump());
    }

    if (device.name() != env["device_name"]) {
        log_warning() << "kernel " << tuning_key << " was tuned for device "
                      << env["device_name"] << ", but the current device is"
                      << device.name() << ". Performance might be suboptimal!";
    }

    int driver_version = -1, runtime_version = -1;
    cuDriverGetVersion(&driver_version);  // ignore errors
    cudaRuntimeGetVersion(&runtime_version);

    if (driver_version != env["driver_version"]) {
        log_warning() << "kernel " << tuning_key
                      << " was tuned for CUDA driver version "
                      << env["driver_version"] << ", but the current version is"
                      << driver_version << ". Performance might be suboptimal!";
    }

    if (runtime_version != env["runtime_version"]) {
        log_warning() << "kernel " << tuning_key
                      << " was tuned for CUDA runtime version "
                      << env["runtime_version"]
                      << ", but the current version is" << runtime_version
                      << ". Performance might be suboptimal!";
    }
}

static json tunable_param_to_json(const TunableParam& param) {
    std::vector<json> result;
    result.push_back(value_to_json(param.default_value()));

    for (const auto& v : param.values()) {
        if (v != param.default_value()) {
            result.push_back(value_to_json(v));
        }
    }

    return result;
}

struct KernelBuilderSerializerHack {
    static json builder_to_json(const KernelBuilder& builder) {
        std::vector<json> restrictions;
        for (auto e : expr_list_to_json(builder.restrictions_)) {
            restrictions.push_back(e);
        }
        for (auto e : expr_list_to_json(builder.assertions_)) {
            restrictions.push_back(e);
        }

        std::unordered_map<std::string, json> parameters;
        for (const auto& p : builder.params_) {
            parameters[p.name()] = tunable_param_to_json(p);
        }

        std::unordered_map<std::string, json> defines;
        for (const auto& p : builder.defines_) {
            defines[p.first] = expr_to_json(p.second);
        }

        json result;
        result["kernel_name"] = builder.kernel_name_;
        result["kernel_file"] = builder.kernel_source_.file_name();
        result["parameters"] = parameters;
        result["compile_flags"] = expr_list_to_json(builder.compile_flags_);
        result["block_size"] = expr_list_to_json(builder.block_size_);
        result["grid_divisors"] = expr_list_to_json(builder.grid_divisors_);
        result["shared_memory"] = expr_to_json(builder.shared_mem_);
        result["template_args"] = expr_list_to_json(builder.template_args_);
        result["defines"] = std::move(defines);
        result["restrictions"] = std::move(restrictions);

        return result;
    }

    static Config
    json_to_config(const json& config, const KernelBuilder& builder) {
        Config result;

        for (const auto& key : builder.params_) {
            json value = config[key.name()];
            if (value.is_null()) {
                throw std::runtime_error("key not found: " + key.name());
            }

            TunableValue found_value;
            bool is_valid = false;

            for (const TunableValue& v : key.values()) {
                if (value_to_json(v) == value) {
                    is_valid = true;
                    found_value = v;
                }
            }

            if (!is_valid) {
                throw std::runtime_error(
                    "invalid value given for: " + key.name());
            }

            result.insert(key, found_value);
        }

        return result;
    }
};

static json builder_to_json(const KernelBuilder& builder) {
    return KernelBuilderSerializerHack::builder_to_json(builder);
}

struct DataFile {
    hash_t hash;
    size_t size;
    std::string file_name;
};

static const DataFile& write_kernel_arg(
    const std::string& tuning_key,
    const std::string& data_dir,
    const std::vector<char>& data,
    std::vector<DataFile>& previous_files) {
    static constexpr size_t random_suffix_length = 8;
    static constexpr char random_chars[] =
        "0123456789"
        "abcdefghijklmnopqrstuvwxyz";

    hash_t hash = hash_string(data);

    // Find if a previous file has the same hash and size
    for (const auto& p : previous_files) {
        if (p.hash == hash && p.size == data.size()) {
            return p;
        }
    }

    // -2 since:
    //  * uniform_int_distribution is inclusive
    //  * random_chars includes the nul byte
    std::uniform_int_distribution<size_t> dist {0, sizeof(random_chars) - 2};

    std::string path;
    std::mt19937 rng {std::random_device {}()};

    for (size_t retry = 0; retry < 10; retry++) {
        std::string file_name = sanitize_tuning_key(tuning_key);
        file_name += "_";

        for (size_t i = 0; i < random_suffix_length; i++) {
            file_name += random_chars[dist(rng)];
        }

        file_name += ".bin";

        path = path_join(data_dir, file_name);
        if (!write_file(path, data)) {
            continue;
        }

        log_debug() << "writing " << data.size() << " bytes to " << path
                    << " for kernel " << tuning_key << std::endl;

        previous_files.push_back({hash, data.size(), file_name});
        return previous_files.back();
    }

    throw std::runtime_error("failed to write data to: " + path);
}

static json kernel_args_to_json(
    const std::string& tuning_key,
    const std::string& data_dir,
    const std::vector<const KernelArg*>& inputs,
    const std::vector<const KernelArg*>& outputs) {
    std::vector<DataFile> previous_files;
    std::vector<json> result;

    for (size_t i = 0; i < inputs.size(); i++) {
        const KernelArg* input = inputs[i];
        std::vector<char> data = input->to_bytes();

        json entry;
        entry["type"] = input->type_info().name();

        if (input->type_info().is_pointer()) {
            const DataFile& input_file =
                write_kernel_arg(tuning_key, data_dir, data, previous_files);

            entry["kind"] = "array";
            entry["hash"] = input_file.hash;
            entry["file"] = input_file.file_name;

            if (i < outputs.size()) {
                data = outputs[i]->to_bytes();

                const DataFile& output_file = write_kernel_arg(
                    tuning_key,
                    data_dir,
                    data,
                    previous_files);

                // Only add the output reference file if it does not match the
                // input file.
                if (input_file.hash != output_file.hash) {
                    entry["reference_file"] = output_file.file_name;
                    entry["reference_hash"] = output_file.hash;
                }
            }
        } else {
            entry["kind"] = "scalar";
            entry["data"] = data;
        }

        result.push_back(entry);
    }

    return result;
}

static json wisdom_to_json(
    const std::string& tuning_key,
    const KernelBuilder& builder,
    const std::string& data_dir,
    dim3 problem_size,
    const std::vector<const KernelArg*>& inputs,
    const std::vector<const KernelArg*>& outputs,
    CudaDevice device) {
    json result;
    result["environment"] = environment_json(tuning_key, device);
    result["kernel"] = builder_to_json(builder);
    result["arguments"] =
        kernel_args_to_json(tuning_key, data_dir, inputs, outputs);
    result["config"] = nullptr;
    result["problem_size"] = std::vector<unsigned int> {
        problem_size.x,
        problem_size.y,
        problem_size.z,
    };

    return result;
}

static Config json_to_config(const json& config, const KernelBuilder& builder) {
    return KernelBuilderSerializerHack::json_to_config(config, builder);
}

WisdomResult read_wisdom_file(
    const std::string& tuning_key,
    const KernelBuilder& builder,
    const std::string& wisdom_dir,
    const CudaDevice device,
    Config& config) {
    std::string file_path =
        path_join(wisdom_dir, sanitize_tuning_key(tuning_key) + ".json");
    std::vector<char> content;

    log_debug() << "reading wisdom file from " << file_path << " for kernel "
                << tuning_key << std::endl;

    try {
        if (!read_file(file_path, content)) {
            return WisdomResult::NotFound;
        }

        json input = json::parse(content.begin(), content.end());

        // Not tuned, return invalid
        if (!input["config"].is_object()) {
            return WisdomResult::Invalid;
        }

        // Check if environment is correct
        check_environment_json(input["environment"], tuning_key, device);

        config = json_to_config(input["config"], builder);
        return WisdomResult::Success;
    } catch (const std::exception& e) {
        log_info() << "error occurred while reading wisdom file " << file_path
                   << ": " << e.what() << std::endl;
        return WisdomResult::IoError;
    }
}

void write_wisdom_file(
    const std::string& tuning_key,
    const KernelBuilder& builder,
    const std::string& wisdom_dir,
    const std::string& data_dir,
    dim3 problem_size,
    const std::vector<const KernelArg*>& inputs,
    const std::vector<const KernelArg*>& outputs,
    CudaDevice device) {
    std::string file_name =
        path_join(wisdom_dir, sanitize_tuning_key(tuning_key) + ".json");

    try {
        json content_json = wisdom_to_json(
            tuning_key,
            builder,
            data_dir,
            problem_size,
            inputs,
            outputs,
            device);

        std::string content_str = content_json.dump(4);
        std::vector<char> content(content_str.begin(), content_str.end());

        log_debug() << "writing wisdom file to " << file_name << " for kernel "
                    << tuning_key << std::endl;

        write_file(file_name, content);
    } catch (const std::exception& e) {
        log_warning() << "error occurred while write wisdom file " << file_name
                      << ": " << e.what() << std::endl;
    }
}

}  // namespace kernel_launcher