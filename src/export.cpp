#include "kernel_launcher/export.h"

#include <unistd.h>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>

#include "nlohmann/json.hpp"
#include "teeny-sha1/teeny-sha1.h"

#ifdef KERNEL_LAUNCHER_USE_ZLIB
    #include <zlib.h>
#endif

namespace kernel_launcher {

using json = nlohmann::ordered_json;

static std::string sanitize_tuning_key(const std::string& key) {
    std::string output = key;

    for (char& c : output) {
        bool valid = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z')
            || (c >= 'A' && c <= 'Z') || c == '_' || c == '-' || c == '.';

        if (!valid) {
            c = '_';
        }
    }

    return output;
}

static json value_to_json(const Value& expr) {
    switch (expr.data_type()) {
        case Value::type_int:
            return expr.to_integer();
        case Value::type_double:
            return expr.to_double();
        case Value::type_string:
            return expr.to_string();
        case Value::type_bool:
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

    if (const auto* pe = dynamic_cast<const ParamExpr*>(&expr)) {
        op = "parameter";
        operands.emplace_back(pe->parameter().name());
    } else if (const auto* ue = dynamic_cast<const UnaryExpr*>(&expr)) {
        op = ue->op_name();
        operands.emplace_back(expr_to_json(ue->operand()));
    } else if (const auto* be = dynamic_cast<const BinaryExpr*>(&expr)) {
        op = be->op_name();
        operands.emplace_back(expr_to_json(be->left_operand()));
        operands.emplace_back(expr_to_json(be->right_operand()));
    } else if (const auto* se = dynamic_cast<const SelectExpr*>(&expr)) {
        op = "select";
        operands.emplace_back(expr_to_json(se->condition()));

        for (const auto& p : se->operands()) {
            operands.push_back(expr_to_json(p));
        }
    } else if (
        const auto* de = dynamic_cast<const DeviceAttributeExpr*>(&expr)) {
        op = "device_attribute";
        operands.emplace_back(de->to_string());
    } else if (const auto* pse = dynamic_cast<const ProblemExpr*>(&expr)) {
        op = "problem_size";
        operands.emplace_back(pse->axis());
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

static json environment_json() {
    int driver_version = -1;
    cuDriverGetVersion(&driver_version);  // ignore errors

    int runtime_version = -1;
    cudaRuntimeGetVersion(&runtime_version);

    int nvrtc_version = NvrtcCompiler::version();

    char hostname[256] = {0};
    gethostname(hostname, sizeof(hostname));

    std::time_t t =
        std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::stringstream date;
    date << std::put_time(std::localtime(&t), "%FT%T%z");

    json env;
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

static json tunable_param_to_json(const TunableParam& param) {
    std::vector<json> values;
    std::vector<json> priors;
    size_t n = param.values().size();

    for (size_t i = 0; i < n; i++) {
        const Value& v = param.values()[i];
        double prior = param.priors()[i];

        values.emplace_back(value_to_json(v));
        priors.emplace_back(prior);
    }

    return {
        {"name", param.name()},
        {"values", std::move(values)},
        {"priors", std::move(priors)},
        {"default", value_to_json(param.default_value())}};
}

struct KernelBuilderSerializerHack {
    static json config_space_to_json(const KernelBuilder& builder) {
        std::vector<json> restrictions;
        for (auto e : expr_list_to_json(builder.restrictions_)) {
            restrictions.emplace_back(std::move(e));
        }
        for (auto e : expr_list_to_json(builder.assertions_)) {
            restrictions.emplace_back(std::move(e));
        }

        std::vector<json> parameters;
        for (const auto& p : builder.params_) {
            parameters.push_back(tunable_param_to_json(p));
        }

        return {
            {"parameters", std::move(parameters)},
            {"restrictions", std::move(restrictions)}};
    }

    static json builder_to_json(const KernelBuilder& builder) {
        std::vector<json> headers;
        for (const auto& source : builder.preheaders_) {
            json content = nullptr;
            if (source.content() != nullptr) {
                content = *source.content();
            }

            headers.push_back({
                {"file", source.file_name()},
                {"content", std::move(content)},
            });
        }

        json defines = json::object();
        for (const auto& p : builder.defines_) {
            defines[p.first] = expr_to_json(p.second);
        }

        json result;
        const std::string* content = builder.kernel_source_.content();
        if (content != nullptr) {
            result["source"] = *content;
        } else {
            result["file"] = builder.kernel_source_.file_name();
        }

        result["name"] = builder.kernel_name_;
        result["compile_flags"] = expr_list_to_json(builder.compile_flags_);
        result["block_size"] = expr_list_to_json(builder.block_size_);
        result["grid_size"] = expr_list_to_json(builder.grid_size_);
        result["shared_memory"] = expr_to_json(builder.shared_mem_);
        result["template_args"] = expr_list_to_json(builder.template_args_);
        result["defines"] = std::move(defines);
        result["headers"] = std::move(headers);

        return result;
    }
};

struct DataFile {
    std::string hash;
    size_t size;
    std::string file_name;
};

static const DataFile& write_kernel_arg(
    const std::string& tuning_key,
    const std::string& data_dir,
    const std::vector<uint8_t>& raw_data,
    std::vector<DataFile>& previous_files) {
    static constexpr size_t random_suffix_length = 8;
    static constexpr char random_chars[] =
        "0123456789"
        "abcdefghijklmnopqrstuvwxyz";

    char hash_data[41] = {0};  // 20 bytes digest = 40 char in hex + 1 nul byte
    sha1digest(nullptr, hash_data, raw_data.data(), raw_data.size());
    std::string hash = hash_data;

    // Find if a previous file has the same hash and size
    for (const auto& p : previous_files) {
        if (p.hash == hash && p.size == raw_data.size()) {
            return p;
        }
    }

    const uint8_t* data = raw_data.data();
    size_t data_size = raw_data.size();
    const char* file_suffix = ".bin";

    // Compress data using zlib if it is available.
#ifdef KERNEL_LAUNCHER_USE_ZLIB
    // Buffer to store compressed data.
    std::vector<uint8_t> compressed_data;
    compressed_data.resize(data_size);

    // zlib cannot handle buffers larger than what fits into uInt
    if (data_size <= std::numeric_limits<uInt>::max()) {
        // Initialize zstream
        z_stream zs;
        zs.zalloc = Z_NULL;
        zs.zfree = Z_NULL;
        zs.opaque = Z_NULL;
        zs.avail_in = uInt(data_size);
        zs.next_in =
            const_cast<uint8_t*>(raw_data.data());  // next_in should be const
        zs.avail_out = uInt(data_size);
        zs.next_out = compressed_data.data();

        // Using basic zlib API.
        deflateInit2(
            &zs,
            Z_DEFAULT_COMPRESSION,
            Z_DEFLATED,
            15 | 16,
            8,
            Z_DEFAULT_STRATEGY);
        deflate(&zs, Z_FINISH);
        deflateEnd(&zs);

        // Only use the compressed data if it is actually smaller than the original data.
        if (zs.total_out < raw_data.size()) {
            data = compressed_data.data();
            data_size = zs.total_out;
            file_suffix = ".bin.gz";
        }
    }
#endif

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

        file_name += file_suffix;

        path = path_join(data_dir, file_name);
        if (!write_file(path, (char*)data, data_size)) {
            continue;
        }

        log_debug() << "writing " << data_size << " bytes to " << path
                    << " for kernel " << tuning_key << std::endl;

        previous_files.push_back({hash, raw_data.size(), file_name});
        return previous_files.back();
    }

    throw std::runtime_error("failed to write data to: " + path);
}

static json kernel_args_to_json(
    const std::string& tuning_key,
    const std::string& data_dir,
    const std::vector<TypeInfo>& param_types,
    const std::vector<std::vector<uint8_t>>& inputs,
    const std::vector<std::vector<uint8_t>>& outputs) {
    std::vector<DataFile> previous_files;
    std::vector<json> result;

    size_t nargs = param_types.size();
    if (inputs.size() != nargs
        || (!outputs.empty() && outputs.size() != nargs)) {
        throw std::invalid_argument("invalid number of arguments");
    }

    for (size_t i = 0; i < nargs; i++) {
        TypeInfo dtype = param_types[i];

        json entry;
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

static json kernel_to_json(
    const std::string& tuning_key,
    const KernelBuilder& builder,
    const std::string& data_dir,
    ProblemSize problem_size,
    const std::vector<TypeInfo>& param_types,
    const std::vector<std::vector<uint8_t>>& inputs,
    const std::vector<std::vector<uint8_t>>& outputs) {
    json result;
    result["key"] = tuning_key;
    result["environment"] = environment_json();
    result["config_space"] =
        KernelBuilderSerializerHack::config_space_to_json(builder);
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
        json content_json = kernel_to_json(
            tuning_key,
            builder,
            directory,
            problem_size,
            param_types,
            inputs,
            outputs);

        log_info() << "writing capture to " << file_name << " for kernel "
                   << tuning_key << std::endl;

        std::string content = content_json.dump(4);
        write_file(file_name, content);
    } catch (const std::exception& e) {
        log_warning() << "error occurred while writing capture of kernel "
                      << tuning_key << " to file " << file_name << ": "
                      << e.what() << std::endl;
    }
}

}  // namespace kernel_launcher
