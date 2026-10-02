#include <unistd.h>

#include <array>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <random>
#include <sstream>

#ifdef KERNEL_LAUNCHER_USE_ZLIB
    #include <zlib.h>
#endif

#include "kernel_launcher/capture.hpp"
#include "kernel_launcher/compiler.hpp"
#include "kernel_launcher/file_loader.hpp"
#include "kernel_launcher/gpu_utils.hpp"
#include "nlohmann/json.hpp"
#include "teeny-sha1/teeny-sha1.h"

namespace kernel_launcher {

using json = nlohmann::ordered_json;

static std::string sanitize_tuning_key(std::string_view key) {
    std::string output(key);

    for (char& c : output) {
        bool valid = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
            || c == '_' || c == '-' || c == '.';

        if (!valid) {
            c = '_';
        }
    }

    return output;
}

static json value_to_json(const Primitive& expr) {
    switch (expr.kind()) {
        case Primitive::Kind::Int64:
            return expr.to_integer();
        case Primitive::Kind::Float64:
            return expr.to_float();
        case Primitive::Kind::String:
            return expr.to_string();
        case Primitive::Kind::Bool:
            return expr.to_bool();
        default:
            return nullptr;
    }
}

static json expr_to_json(const ExprBase<Primitive>& expr, const EvalContext& eval) {
    if (const ScalarExpr* v = dynamic_cast<const ScalarExpr*>(&expr)) {
        return value_to_json(v->value());
    }

    if (const ExprPtr* v = dynamic_cast<const ExprPtr*>(&expr)) {
        return expr_to_json(v->inner(), eval);
    }

    std::string op;
    std::vector<json> operands;

    if (const auto* pe = dynamic_cast<const TunableParam*>(&expr)) {
        op = "parameter";
        operands.emplace_back(std::string(pe->name()));
    } else if (const auto* ue = dynamic_cast<const UnaryExpr*>(&expr)) {
        op = ue->op_name();
        operands.emplace_back(expr_to_json(ue->operand(), eval));
    } else if (const auto* be = dynamic_cast<const BinaryExpr*>(&expr)) {
        op = be->op_name();
        operands.emplace_back(expr_to_json(be->left_operand(), eval));
        operands.emplace_back(expr_to_json(be->right_operand(), eval));
    } else if (const auto* se = dynamic_cast<const SelectExpr*>(&expr)) {
        op = "select";
        operands.emplace_back(expr_to_json(se->index(), eval));

        for (const auto& p : se->options()) {
            operands.push_back(expr_to_json(p, eval));
        }
    } else if (const auto* de = dynamic_cast<const DeviceAttributeExpr*>(&expr)) {
        op = "device_attribute";
        operands.emplace_back(de->to_string());
    } else if (const auto* var = dynamic_cast<const Variable*>(&expr)) {
        Primitive value;
        if (eval.try_lookup(*var, value)) {
            return value_to_json(value);
        }

        throw std::runtime_error("could not serialize unknown variable: " + expr.to_string());
    } else {
        throw std::runtime_error("could not serialize expression: " + expr.to_string());
    }

    return {{"operator", op}, {"operands", operands}};
}

template<typename C>
static std::vector<json> expr_list_to_json(C collection, const EvalContext& eval) {
    std::vector<json> result;

    for (const auto& entry : collection) {
        result.push_back(expr_to_json(entry.inner(), eval));
    }

    return result;
}

static json environment_json() {
    // Ignore errors: the environment is informational only.
    int driver_version = -1;
    try {
        driver_version = gpu_driver_version();
    } catch (const std::exception&) {
    }

    int runtime_version = -1;
    try {
        runtime_version = gpu_runtime_version();
    } catch (const std::exception&) {
    }

    int nvrtc_version = RuntimeCompiler::version();

    char hostname[256] = {0};
    gethostname(hostname, sizeof(hostname));

    std::time_t t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
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

    for (const Primitive& v : param.values()) {
        values.emplace_back(value_to_json(v));
    }

    return {
        {"name", std::string(param.name())},
        {"values", std::move(values)},
        {"default", value_to_json(param.default_value())}};
}

static json features_to_json(const FeatureVector& features) {
    json result = json::object();

    for (const auto& [key, value] : features) {
        result[std::string(key.get())] = value_to_json(value);
    }

    return result;
}

struct KernelBuilderSerializerHack {
    static json config_space_to_json(const KernelBuilder& builder, const EvalContext& eval) {
        std::vector<json> restrictions;
        for (auto e : expr_list_to_json(builder.restrictions_, eval)) {
            restrictions.emplace_back(std::move(e));
        }
        for (auto e : expr_list_to_json(builder.assertions_, eval)) {
            restrictions.emplace_back(std::move(e));
        }

        std::vector<json> parameters;
        for (const auto& p : builder.params_) {
            parameters.push_back(tunable_param_to_json(p));
        }

        return {{"parameters", std::move(parameters)}, {"restrictions", std::move(restrictions)}};
    }

    static json builder_to_json(const KernelBuilder& builder, const EvalContext& eval) {
        json result;
        if (auto content = builder.kernel_source_.content()) {
            result["source"] = std::string(*content);
        } else {
            result["file"] = builder.kernel_source_.file_name();
        }

        result["name"] = builder.kernel_name_;
        result["compile_flags"] = expr_list_to_json(builder.compiler_flags_, eval);
        result["shared_memory"] =  //
            expr_to_json(builder.shared_memory_.inner(), eval);

        std::vector<json> template_args;
        for (const auto& arg : builder.template_args_) {
            if (const auto* expr = std::get_if<Expr<Primitive>>(&arg)) {
                template_args.push_back(expr_to_json(expr->inner(), eval));
            } else {
                template_args.push_back(std::get<TemplateArg>(arg).get());
            }
        }
        result["template_args"] = std::move(template_args);

        std::vector<json> headers;
        for (const auto& source : builder.headers_) {
            json header_content = nullptr;
            if (auto content = source.content()) {
                header_content = std::string(*content);
            }

            headers.push_back({
                {"file", source.file_name()},
                {"content", std::move(header_content)},
            });
        }
        result["headers"] = std::move(headers);

        json defines = json::object();
        for (const auto& p : builder.defines_) {
            defines[p.first] = expr_to_json(p.second.inner(), eval);
        }
        result["defines"] = std::move(defines);

        result["block_size"] = expr_list_to_json(builder.resolve_block_size(), eval);
        result["grid_size"] = expr_list_to_json(builder.resolve_grid_size(), eval);

        return result;
    }
};

struct DataFile {
    std::string hash;
    size_t size;
    std::string file_name;
};

static const DataFile& write_kernel_arg(
    std::string_view tuning_key,
    std::string_view data_dir,
    const std::vector<std::byte>& raw_data,
    std::vector<DataFile>& previous_files) {
    static constexpr size_t random_suffix_length = 8;
    static constexpr char random_chars[] =
        "0123456789"
        "abcdefghijklmnopqrstuvwxyz";

    const uint8_t* data = reinterpret_cast<const uint8_t*>(raw_data.data());
    size_t data_size = raw_data.size();

    char hash_data[41] = {0};  // 20 bytes digest = 40 char in hex + 1 nul byte
    sha1digest(nullptr, hash_data, data, data_size);
    std::string hash = hash_data;

    // Find if a previous file has the same hash and size
    for (const auto& p : previous_files) {
        if (p.hash == hash && p.size == raw_data.size()) {
            return p;
        }
    }

    const char* file_suffix = ".bin";

    // Compress data using zlib if it is available.
#ifdef KERNEL_LAUNCHER_USE_ZLIB
    // Buffer to store compressed data.
    std::vector<uint8_t> compressed_data;
    compressed_data.resize(data_size);

    // zlib cannot handle buffers larger than what fits into uInt
    if (data_size <= std::numeric_limits<uInt>::max()) {
        // Initialize zstream
        // Zero-initialized, so zalloc, zfree, and opaque are Z_NULL.
        z_stream zs {};
        zs.zalloc = Z_NULL;
        zs.zfree = Z_NULL;
        zs.opaque = Z_NULL;
        zs.avail_in = uInt(data_size);
        zs.next_in = const_cast<uint8_t*>(data);  // next_in should be const
        zs.avail_out = uInt(data_size);
        zs.next_out = compressed_data.data();

        // Using basic zlib API.
        int status =
            deflateInit2(&zs, Z_DEFAULT_COMPRESSION, Z_DEFLATED, 15 | 16, 8, Z_DEFAULT_STRATEGY);

        if (status == Z_OK) {
            status = deflate(&zs, Z_FINISH);

            // Always frees the state. Its return value only reports whether
            // the stream was finished, which `status` already tells us.
            deflateEnd(&zs);

            // Anything other than Z_STREAM_END means the output buffer was
            // too small (i.e., the data is not compressible). Only use the
            // compressed data if it is actually smaller than the original.
            if (status == Z_STREAM_END && zs.total_out < data_size) {
                data = compressed_data.data();
                data_size = zs.total_out;
                file_suffix = ".bin.gz";
            }
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
        if (!write_file(path, std::string_view((char*)data, data_size))) {
            continue;
        }

        log_debug() << "writing " << data_size << " bytes to " << path << " for kernel "
                    << tuning_key << std::endl;

        previous_files.push_back({hash, raw_data.size(), file_name});
        return previous_files.back();
    }

    throw std::runtime_error("failed to write data to: " + path);
}

static json kernel_args_to_json(
    std::string_view tuning_key,
    std::string_view data_dir,
    const std::vector<KernelArg>& arguments,
    const std::vector<std::vector<std::byte>>& input_arrays,
    const std::vector<std::vector<std::byte>>& output_arrays) {
    std::vector<DataFile> previous_files;
    std::vector<json> result;

    if (input_arrays.size() != arguments.size()) {
        throw std::invalid_argument("invalid number of input arrays");
    }

    if (!output_arrays.empty() && output_arrays.size() != arguments.size()) {
        throw std::invalid_argument("invalid number of output arrays");
    }

    // Check all arrays before writing any of them, so no data files are left
    // behind if one of them is invalid.
    for (size_t i = 0; i < arguments.size(); i++) {
        if (!arguments[i].type().is_pointer()) {
            continue;
        }

        // Data type of primitive scalar type for pointer
        TypeInfo prim_type = arguments[i].type().remove_pointer();

        // The array of a `void*` can be any number of bytes, since `void` has no size.
        if (!prim_type.is_void() && input_arrays[i].size() % prim_type.size_in_bytes() != 0) {
            throw std::invalid_argument("invalid input array size");
        }

        if (!output_arrays.empty() && !output_arrays[i].empty()
            && input_arrays[i].size() != output_arrays[i].size()) {
            throw std::invalid_argument("invalid output array size");
        }
    }

    for (size_t i = 0; i < arguments.size(); i++) {
        const KernelArg& argument = arguments[i];
        json entry;
        entry["type"] = argument.type().name();

        if (argument.type().is_pointer()) {
            const DataFile& input_file =
                write_kernel_arg(tuning_key, data_dir, input_arrays[i], previous_files);

            entry["kind"] = "array";
            entry["hash"] = input_file.hash;
            entry["file"] = input_file.file_name;

            if (!output_arrays.empty() && !output_arrays[i].empty()) {
                const DataFile& output_file =
                    write_kernel_arg(tuning_key, data_dir, output_arrays[i], previous_files);

                // Only add the output reference file if it does not match the
                // input file.
                if (input_file.hash != output_file.hash) {
                    entry["reference_file"] = output_file.file_name;
                    entry["reference_hash"] = output_file.hash;
                }
            }
        } else {
            entry["kind"] = "scalar";
            const auto* bytes = static_cast<const uint8_t*>(argument.data());
            entry["data"] = std::vector<uint8_t>(bytes, bytes + argument.type().size_in_bytes());
        }

        result.emplace_back(std::move(entry));
    }

    return result;
}

struct ArgsEval: EvalContext {
    explicit ArgsEval(const ProblemKey& problem, const std::vector<KernelArg>& args) :
        args_(args),
        problem_(problem) {}

    bool try_lookup(const Variable& v, Primitive& out) const override {
        if (const auto* that = dynamic_cast<const ArgumentExpr*>(&v)) {
            size_t i = that->index();

            if (i < args_.size()) {
                out = that->get(args_[i]);

                if (!out.is_empty()) {
                    return true;
                }
            }
        }

        if (const auto* that = dynamic_cast<const ProblemSizeExpr*>(&v)) {
            if (that->axis() < problem_.problem_size.size()) {
                out = Primitive(problem_.problem_size[that->axis()]);
                return true;
            }
        }

        if (const auto* that = dynamic_cast<const FeatureExpr*>(&v)) {
            out = problem_.features.get(that->name());
            return true;
        }

        return false;
    }

  private:
    const std::vector<KernelArg>& args_;
    const ProblemKey& problem_;
};

static json kernel_to_json(
    const KernelBuilder& builder,
    std::string_view data_dir,
    const ProblemKey& problem,
    const std::vector<KernelArg>& arguments,
    const std::vector<std::vector<std::byte>>& input_arrays,
    const std::vector<std::vector<std::byte>>& output_arrays) {
    const std::string& tuning_key = problem.tuning_key.get();
    ArgsEval eval(problem, arguments);

    json result;
    result["key"] = tuning_key;
    result["environment"] = environment_json();
    result["config_space"] = KernelBuilderSerializerHack::config_space_to_json(builder, eval);
    result["kernel"] = KernelBuilderSerializerHack::builder_to_json(builder, eval);
    result["arguments"] =
        kernel_args_to_json(tuning_key, data_dir, arguments, input_arrays, output_arrays);
    result["problem_size"] = problem.problem_size;

    if (problem.features.begin() != problem.features.end()) {
        result["features"] = features_to_json(problem.features);
    }

    return result;
}

static std::string tuning_key_to_file_name(std::string_view directory, const ProblemKey& problem) {
    const std::array<int64_t, MAX_PROBLEM_DIMS>& size = problem.problem_size;

    std::string file = sanitize_tuning_key(problem.tuning_key);
    file += "_";
    file += std::to_string(size[0]);

    if (size[1] != 1 || size[2] != 1) {
        file += 'x';
        file += std::to_string(size[1]);
    }

    if (size[2] != 1) {
        file += 'x';
        file += std::to_string(size[2]);
    }

    file += ".json";
    return path_join(directory, file);
}

bool capture_file_exists(std::string_view directory, const ProblemKey& problem_key) {
    std::string file_name = tuning_key_to_file_name(directory, problem_key);
    // Unlike opening the file, this also finds files that are not readable.
    return access(file_name.c_str(), F_OK) == 0;
}

void export_capture_file(
    std::string_view directory,
    const KernelBuilder& builder,
    const ProblemKey& problem_key,
    const std::vector<KernelArg>& arguments,
    const std::vector<std::vector<std::byte>>& input_arrays,
    const std::vector<std::vector<std::byte>>& output_arrays) {
    const std::string& tuning_key = problem_key.tuning_key.get();
    std::string file_name = tuning_key_to_file_name(directory, problem_key);

    try {
        // Check before writing the data files, so they are not left behind.
        if (capture_file_exists(directory, problem_key)) {
            log_warning() << "capture of kernel " << tuning_key << " already exists at "
                          << file_name << ", not writing a new capture" << std::endl;
            return;
        }

        json content_json =
            kernel_to_json(builder, directory, problem_key, arguments, input_arrays, output_arrays);

        std::string content = content_json.dump(4);
        if (!write_file(file_name, content)) {
            log_warning() << "capture of kernel " << tuning_key << " already exists at "
                          << file_name << ", not writing a new capture" << std::endl;
            return;
        }

        log_info() << "wrote capture to " << file_name << " for kernel " << tuning_key << std::endl;
    } catch (const std::exception& e) {
        log_warning() << "error occurred while writing capture of kernel " << tuning_key
                      << " to file " << file_name << ": " << e.what() << std::endl;
    }
}

}  // namespace kernel_launcher
