#include "kernel_launcher/export.h"

#include <unistd.h>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>

#include "nlohmann/json.hpp"

namespace kernel_launcher {

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

static nlohmann::json value_to_json(const TunableValue& expr) {
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

static nlohmann::json expr_to_json(const BaseExpr& expr) {
    if (const ScalarExpr* v = dynamic_cast<const ScalarExpr*>(&expr)) {
        return value_to_json(v->value());
    }

    if (const SharedExpr* v = dynamic_cast<const SharedExpr*>(&expr)) {
        return expr_to_json(v->inner());
    }

    std::string op;
    std::vector<nlohmann::json> operands;

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
static std::vector<nlohmann::json> expr_list_to_json(C collection) {
    std::vector<nlohmann::json> result;

    for (const auto& entry : collection) {
        result.push_back(expr_to_json(entry));
    }

    return result;
}

static nlohmann::json environment_json() {
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
    env["host_name"] = hostname;
    env["date"] = date.str();
    env["runtime_version"] = runtime_version;
    env["driver_version"] = driver_version;
    env["nvrtc_version"] = nvrtc_version;
    //    env["device_name"] = device.name();
    //    env["device_ordinal"] = device.ordinal();
    //    env["device_uuid"] = device.uuid();
    return env;
}

static nlohmann::json tunable_param_to_json(const TunableParam& param) {
    std::vector<nlohmann::json> values;
    std::vector<nlohmann::json> priors;
    size_t n = param.values().size();

    for (size_t i = 0; i < n; i++) {
        const TunableValue& v = param.values()[i];
        double prior = param.priors()[i];

        if (v != param.default_value()) {
            values.insert(values.begin(), value_to_json(v));
            values.insert(values.begin(), prior);
        } else {
            values.push_back(value_to_json(v));
            values.push_back(prior);
        }
    }

    return nlohmann::json::object_t {
        {"name", param.name()},
        {"values", std::move(values)},
        {"priors", std::move(priors)},
    };
}

struct KernelBuilderSerializerHack {
    static nlohmann::json builder_to_json(const KernelBuilder& builder) {
        std::vector<nlohmann::json> restrictions;
        for (auto e : expr_list_to_json(builder.restrictions_)) {
            restrictions.push_back(e);
        }
        for (auto e : expr_list_to_json(builder.assertions_)) {
            restrictions.push_back(e);
        }

        std::vector<nlohmann::json> parameters;
        for (const auto& p : builder.params_) {
            parameters.push_back(tunable_param_to_json(p));
        }

        std::unordered_map<std::string, nlohmann::json> defines;
        for (const auto& p : builder.defines_) {
            defines[p.first] = expr_to_json(p.second);
        }

        nlohmann::json result;
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
};

struct DataFile {
    hash_t hash;
    size_t size;
    std::string file_name;
};

static const DataFile& write_kernel_arg(
    const std::string& tuning_key,
    const std::string& data_dir,
    const std::vector<uint8_t>& data,
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
        if (!write_file(path, (char*)data.data(), data.size())) {
            continue;
        }

        log_debug() << "writing " << data.size() << " bytes to " << path
                    << " for kernel " << tuning_key << std::endl;

        previous_files.push_back({hash, data.size(), file_name});
        return previous_files.back();
    }

    throw std::runtime_error("failed to write data to: " + path);
}

static nlohmann::json kernel_args_to_json(
    const std::string& tuning_key,
    const std::string& data_dir,
    const std::vector<TypeInfo>& param_types,
    const std::vector<std::vector<uint8_t>>& inputs,
    const std::vector<std::vector<uint8_t>>& outputs) {
    std::vector<DataFile> previous_files;
    std::vector<nlohmann::json> result;

    size_t nargs = param_types.size();
    if (inputs.size() != nargs
        || (!outputs.empty() && outputs.size() != nargs)) {
        throw std::invalid_argument("invalid number of arguments");
    }

    for (size_t i = 0; i < nargs; i++) {
        TypeInfo dtype = param_types[i];

        nlohmann::json entry;
        entry["type"] = dtype.name();

        if (dtype.is_pointer()) {
            if (inputs[i].size() % dtype.size() != 0) {
                throw std::invalid_argument("invalid input argument");
            }

            const DataFile& input_file = write_kernel_arg(
                tuning_key,
                data_dir,
                inputs[i],
                previous_files);

            entry["kind"] = "array";
            entry["hash"] = input_file.hash;
            entry["file"] = input_file.file_name;

            if (!outputs.empty() && !outputs[i].empty()) {
                if (inputs[i].size() != outputs[i].size()) {
                    throw std::invalid_argument("invalid output argument");
                }

                const DataFile& output_file = write_kernel_arg(
                    tuning_key,
                    data_dir,
                    outputs[i],
                    previous_files);

                // Only add the output reference file if it does not match the
                // input file.
                if (input_file.hash != output_file.hash) {
                    entry["reference_file"] = output_file.file_name;
                    entry["reference_hash"] = output_file.hash;
                }
            }
        } else {
            if (inputs[i].size() != dtype.size()) {
                throw std::invalid_argument("invalid argument");
            }

            entry["kind"] = "scalar";
            entry["data"] = inputs[i];
        }

        result.emplace_back(std::move(entry));
    }

    return result;
}

static nlohmann::json wisdom_to_json(
    const std::string& tuning_key,
    const KernelBuilder& builder,
    const std::string& data_dir,
    ProblemSize problem_size,
    const std::vector<TypeInfo>& param_types,
    const std::vector<std::vector<uint8_t>>& inputs,
    const std::vector<std::vector<uint8_t>>& outputs) {
    nlohmann::json result;
    result["key"] = sanitize_tuning_key(tuning_key);
    result["environment"] = environment_json();
    result["kernel"] = KernelBuilderSerializerHack::builder_to_json(builder);
    result["arguments"] =
        kernel_args_to_json(tuning_key, data_dir, param_types, inputs, outputs);
    result["problem_size"] = std::vector<unsigned int> {
        problem_size.x,
        problem_size.y,
        problem_size.z,
    };

    return result;
}

static std::string tuning_key_to_file_name(
    const std::string& directory,
    const std::string& tuning_key,
    ProblemSize problem_size) {
    std::string file = sanitize_tuning_key(tuning_key);
    file += "_";
    file += std::to_string(problem_size.x);

    if (problem_size.y != 1 || problem_size.z != 1) {
        file += 'x';
        file += std::to_string(problem_size.y);
    }

    if (problem_size.z != 1) {
        file += 'x';
        file += std::to_string(problem_size.z);
    }

    file += ".json";
    return path_join(directory, file);
}

bool tuning_file_exists(
    const std::string& directory,
    const std::string& tuning_key,
    ProblemSize problem_size) {
    std::string file_name =
        tuning_key_to_file_name(directory, tuning_key, problem_size);
    return (bool)std::ifstream(file_name);
}

void export_tuning_file(
    const std::string& directory,
    const std::string& tuning_key,
    const KernelBuilder& builder,
    ProblemSize problem_size,
    const std::vector<TypeInfo>& param_types,
    const std::vector<std::vector<uint8_t>>& inputs,
    const std::vector<std::vector<uint8_t>>& outputs) {
    std::string file_name =
        tuning_key_to_file_name(directory, tuning_key, problem_size);

    try {
        nlohmann::json content_json = wisdom_to_json(
            tuning_key,
            builder,
            directory,
            problem_size,
            param_types,
            inputs,
            outputs);

        log_debug() << "writing wisdom file to " << file_name << " for kernel "
                    << tuning_key << std::endl;

        std::string content = content_json.dump(4);
        write_file(file_name, content);
    } catch (const std::exception& e) {
        log_warning() << "error occurred while write wisdom file " << file_name
                      << ": " << e.what() << std::endl;
    }
}

}  // namespace kernel_launcher