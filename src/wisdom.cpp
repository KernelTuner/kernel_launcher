#include "kernel_launcher/wisdom.h"

#include "kernel_launcher/export.h"
#include "nlohmann/json.hpp"

namespace kernel_launcher {

bool WisdomKernelSettings::does_kernel_require_tuning(
    const std::string& tuning_key,
    ProblemSize problem_size) const {
    bool matches = false;

    for (const std::string& pattern : tuning_patterns_) {
        if (tuning_key.find(pattern) != std::string::npos) {
            matches = true;
            break;
        }
    }

    if (!matches) {
        return false;
    }

    if (tuning_file_exists(tuning_dir_, tuning_key, problem_size)) {
        return false;
    }

    return true;
}

std::shared_ptr<WisdomKernelSettings> global_wisdom_settings;

void set_global_wisdom_directory(std::string dir) {
    WisdomKernelSettings s = *default_wisdom_settings();
    global_wisdom_settings = std::make_shared<WisdomKernelSettings>(
        std::move(dir),
        s.tuning_directory(),
        s.tuning_patterns());
}

void set_global_tuning_directory(std::string dir) {
    WisdomKernelSettings s = *default_wisdom_settings();
    global_wisdom_settings = std::make_shared<WisdomKernelSettings>(
        s.wisdom_directory(),
        std::move(dir),
        s.tuning_patterns());
}

void add_global_tuning_pattern(std::string pattern) {
    WisdomKernelSettings s = *default_wisdom_settings();

    std::vector<std::string> patterns = s.tuning_patterns();
    patterns.emplace_back(std::move(pattern));

    global_wisdom_settings = std::make_shared<WisdomKernelSettings>(
        s.wisdom_directory(),
        s.tuning_directory(),
        patterns);
}

std::shared_ptr<WisdomKernelSettings> default_wisdom_settings() {
    if (global_wisdom_settings) {
        return global_wisdom_settings;
    }

    std::string wisdom_dir = ".";
    std::string tuning_dir = ".";
    std::vector<std::string> tuning_patterns = {};
    const char* value;

    if ((value = getenv("KERNEL_LAUNCHER_WISDOM"))) {
        wisdom_dir = value;
        tuning_dir = value;
    }

    if ((value = getenv("KERNEL_LAUNCHER_TUNE"))) {
        std::string all_patterns = value;
        size_t begin = 0;

        while (true) {
            size_t end = all_patterns.find(',', begin);
            std::string pattern = all_patterns.substr(begin, end);
            begin = end + 1;

            if (!pattern.empty()) {
                if (pattern == "1" || pattern == "*" || pattern == "all"
                    || pattern == "true") {
                    pattern = "";
                }

                tuning_patterns.emplace_back(std::move(pattern));
            }

            if (end == std::string::npos) {
                break;
            }
        }
    }

    return global_wisdom_settings = std::make_shared<WisdomKernelSettings>(
               WisdomKernelSettings {wisdom_dir, tuning_dir, tuning_patterns});
}

static std::string sanitize_tuning_key(const std::string& key) {
    std::string output;
    output.resize(key.size());

    for (size_t i = 0; i < key.size(); i++) {
        char c = key[i];
        bool valid = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z')
            || (c >= 'A' && c <= 'Z') || c == '_' || c == '-' || c == '.';

        if (valid) {
            output[i] = c;
        } else {
            output[i] = '_';
        }
    }

    return output;
}

static void parse_header(
    nlohmann::json& header,
    const ConfigSpace& space,
    std::vector<TunableParam>& params,
    std::string& objective) {
    if (header["version_number"] != "1.0") {
        throw std::runtime_error("invalid version");
    }

    if (!header["objective"].is_string()) {
        throw std::runtime_error("invalid objective");
    }

    objective = header["objective"];
    params.clear();

    const nlohmann::json& params_json = header["tunable_parameters"];
    if (!params_json.is_array()) {
        throw std::runtime_error("key \"tunable_parameters\" not found");
    }

    for (const nlohmann::json& record : params_json) {
        if (!record.is_string()) {
            throw std::runtime_error("invalid parameter");
        }

        bool found = false;
        for (const TunableParam& p : space.parameters()) {
            if (p.name() == record) {
                found = true;
                params.push_back(p);
                break;
            }
        }

        if (!found) {
            throw std::runtime_error(
                "parameter \"" + std::string(record)
                + "\" was not found in the configuration space");
        }
    }

    if (params.size() != space.parameters().size()) {
        throw std::runtime_error("invalid number of parameters");
    }
}

static TunableValue json_to_tunable_value(const nlohmann::json& value) {
    switch (value.type()) {
        case nlohmann::json::value_t::number_integer:
            return (int64_t)value;
        case nlohmann::detail::value_t::number_unsigned:
            return (uint64_t)value;
        case nlohmann::detail::value_t::boolean:
            return (bool)value;
        case nlohmann::detail::value_t::number_float:
            return (double)value;
        case nlohmann::json::value_t::string:
            return (const std::string&)value;
        default:
            //            throw std::runtime_error("invalid tunable value");
            return nullptr;
    }
}

template<typename F>
static void process_wisdom_file(
    const std::string& wisdom_dir,
    const std::string& tuning_key,
    const ConfigSpace& space,
    F callback) {
    std::string path = path_join(wisdom_dir, sanitize_tuning_key(tuning_key));

    if (tuning_key.find(".json") == std::string::npos) {
        path += ".json";
    }

    std::ifstream stream(path);
    if (!stream) {
        log_debug() << "wisdom file not found: " << path << "\n";
        return;
    }

    std::string line;
    if (!std::getline(stream, line)) {
        log_debug() << "error while reading wisdom file: " << path << "\n";
        return;
    }

    std::string objective_key;
    std::vector<TunableParam> params;
    std::string err;

    // Parse header
    try {
        nlohmann::json header = nlohmann::json(line);
        parse_header(header, space, params, objective_key);
    } catch (const std::exception& e) {
        log_warning() << path
                      << ":1: file is ignored, error while parsing: " << err
                      << "\n";
        return;
    }

    // Parse each line
    nlohmann::json record;
    Config config;

    for (size_t lineno = 2; std::getline(stream, line); lineno++) {
        if (line.empty()) {
            continue;
        }

        try {
            record = nlohmann::json(line);
        } catch (const std::exception& e) {
            log_warning() << path << ":" << lineno
                          << ": line is ignored, error while parsing json: "
                          << e.what() << "\n";
            continue;
        }

        nlohmann::json& config_json = record["config"];
        nlohmann::json& problem_json = record["problem_size"];
        nlohmann::json& env_json = record["environment"];

        if (config_json.size() != params.size()) {
            continue;
        }

        for (size_t i = 0; i < params.size(); i++) {
            config.insert(params[i], json_to_tunable_value(config_json[i]));
        }

        ProblemSize problem_size;
        if (problem_json.size() > 0)
            problem_size.x = problem_json[0];
        if (problem_json.size() > 1)
            problem_size.y = problem_json[1];
        if (problem_json.size() > 2)
            problem_size.z = problem_json[2];

        const auto& device_name =
            env_json["device_name"].get_ref<const std::string&>();
        double objective = record[objective_key];

        callback(config, problem_size, device_name, objective);
    }
}

Config load_best_config(
    const std::string& wisdom_dir,
    const std::string& tuning_key,
    const ConfigSpace& space,
    CudaDevice device,
    ProblemSize problem_size,
    WisdomResult* result_out) {
    std::string device_name = device.name();

    WisdomResult best_type = WisdomResult::NotFound;
    std::string best_device;
    double best_score = 0.0;
    Config best_config = space.default_config();
    ProblemSize best_problem_size;

    process_wisdom_file(
        wisdom_dir,
        tuning_key,
        space,
        [&](const Config& config,
            ProblemSize row_problem,
            const std::string& row_device,
            double score) {
            WisdomResult type;
            if (row_device != device_name) {
                type = WisdomResult::DeviceMismatch;
            } else if (row_problem != problem_size) {
                type = WisdomResult::ProblemSizeMismatch;
            } else {
                type = WisdomResult::Ok;
            }

            if (type > best_type || (type == best_type && score < best_score)) {
                best_type = type;
                best_score = score;
                best_config = Config(config);
                best_problem_size = problem_size;
                best_device = row_device;
            }
        });

    if (best_type == WisdomResult::NotFound) {
        log_warning() << "no wisdom found for key " << tuning_key
                      << " , using default kernel configuration.\n";
    } else if (best_type == WisdomResult::DeviceMismatch) {
        log_warning() << "no wisdom found for key " << tuning_key
                      << " and device " << device_name
                      << ", using configuration for different device: "
                      << best_device << ".\n";
    } else if (best_type == WisdomResult::ProblemSizeMismatch) {
        log_info() << "no wisdom found for key " << tuning_key << ", device "
                   << device_name << ", and problem size " << problem_size
                   << ", using configuration for different problem size: "
                   << best_problem_size << ".\n";
    }

    if (result_out != nullptr) {
        *result_out = best_type;
    }

    return best_config;
}

struct WisdomKernelImpl {
    bool compiled_;
    std::string tuning_key_;
    KernelBuilder builder_;
    KernelInstance instance_;
    Compiler compiler_;
    std::vector<TypeInfo> param_types_;
    std::shared_ptr<WisdomKernelSettings> settings_;
};

WisdomKernel::WisdomKernel() = default;
WisdomKernel::~WisdomKernel() = default;

void WisdomKernel::initialize(
    std::string tuning_key,
    KernelBuilder builder,
    Compiler compiler,
    std::shared_ptr<WisdomKernelSettings> settings) {
    impl_ = std::make_unique<WisdomKernelImpl>(WisdomKernelImpl {
        false,
        std::move(tuning_key),
        std::move(builder),
        KernelInstance {},
        std::move(compiler),
        std::vector<TypeInfo> {},
        std::move(settings)});
}

void WisdomKernel::clear() {
    if (impl_) {
        impl_->compiled_ = false;
    }
}

WisdomResult WisdomKernel::compile(
    ProblemSize problem_size,
    CudaDevice device,
    std::vector<TypeInfo> param_types) {
    if (!impl_) {
        throw std::runtime_error("WisdomKernel has not been initialized");
    }

    WisdomResult result;
    Config config = load_best_config(
        impl_->settings_->wisdom_directory(),
        impl_->tuning_key_,
        impl_->builder_,
        device,
        problem_size,
        &result);

    impl_->param_types_ = std::move(param_types);
    impl_->instance_ =
        impl_->builder_.compile(config, impl_->param_types_, impl_->compiler_);
    return result;
}

static void assert_types_equal(
    const std::vector<KernelArg>& args,
    const std::vector<TypeInfo>& params) {
    bool is_equal = true;
    if (args.size() == params.size()) {
        for (size_t i = 0; i < args.size(); i++) {
            if (args[i].type().remove_const() != params[i].remove_const()) {
                is_equal = false;
            }
        }
    } else {
        is_equal = false;
    }

    if (is_equal) {
        return;
    }

    std::string msg =
        "invalid argument types: kernel compiled for parameter types (";

    for (size_t i = 0; i < args.size(); i++) {
        if (i != 0)
            msg += ", ";
        msg += args[i].type().name();
    }

    msg += "), but was called with argument types (";

    for (size_t i = 0; i < params.size(); i++) {
        if (i != 0)
            msg += ", ";
        msg += params[i].name();
    }

    msg += ")";
    throw std::runtime_error(msg);
}

void WisdomKernel::launch(
    cudaStream_t stream,
    ProblemSize problem_size,
    const std::vector<KernelArg>& args) {
    if (!impl_) {
        throw std::runtime_error("WisdomKernel has not been initialized");
    }

    std::vector<void*> ptrs;
    for (const KernelArg& arg : args) {
        ptrs.push_back(arg.as_void_ptr());
    }

    if (!impl_->compiled_) {
        std::vector<TypeInfo> param_types;
        for (const KernelArg& arg : args) {
            param_types.push_back(arg.type());
        }

        WisdomResult result =
            compile(problem_size, CudaDevice::current(), param_types);
        bool write_tuning = result != WisdomResult::Ok
            && impl_->settings_->does_kernel_require_tuning(
                impl_->tuning_key_,
                problem_size);

        if (write_tuning) {
            std::vector<std::vector<char>> inputs;
            std::vector<std::vector<char>> outputs;

            KERNEL_LAUNCHER_CUDA_CHECK(cuStreamSynchronize(stream));
            KERNEL_LAUNCHER_CUDA_CHECK(cuCtxSynchronize());

            for (const KernelArg& arg : args) {
                inputs.push_back(arg.to_bytes());
            }

            impl_->instance_.launch(stream, problem_size, ptrs.data());

            KERNEL_LAUNCHER_CUDA_CHECK(cuStreamSynchronize(stream));
            KERNEL_LAUNCHER_CUDA_CHECK(cuCtxSynchronize());

            for (const KernelArg& arg : args) {
                outputs.push_back(arg.to_bytes());
            }

            try {
                export_tuning_file(
                    impl_->settings_->tuning_directory(),
                    impl_->tuning_key_,
                    impl_->builder_,
                    problem_size,
                    param_types,
                    inputs,
                    outputs);
            } catch (const std::exception& err) {
                log_warning()
                    << "error ignored while writing tuning file for \""
                    << impl_->tuning_key_ << "\": " << err.what();
            }

        } else {
            impl_->instance_.launch(stream, problem_size, ptrs.data());
        }
    } else {
        assert_types_equal(args, impl_->param_types_);
        impl_->instance_.launch(stream, problem_size, ptrs.data());
    }
}

static bool is_inline_scalar(TypeInfo type) {
    return type.size() <= sizeof(size_t) * 2;
}

KernelArg::KernelArg(TypeInfo type, void* data) {
    type_ = type;
    scalar_ = false;

    if (is_inline_scalar(type_)) {
        ::memcpy(data_.small_scalar.data(), data, type.size());
    } else {
        data_.large_scalar = new char[type.size()];
        ::memcpy(data_.large_scalar, data, type.size());
    }
}

KernelArg::KernelArg(TypeInfo type, void* ptr, size_t nelements) {
    type_ = type;
    scalar_ = false;
    data_.array.ptr = ptr;
    data_.array.nelements = nelements;
}

KernelArg::KernelArg(const KernelArg& that) : KernelArg() {
    type_ = that.type_;
    scalar_ = that.scalar_;

    if (that.is_array()) {
        data_.array = that.data_.array;
    } else if (is_inline_scalar(type_)) {
        data_.small_scalar = that.data_.small_scalar;
    } else {
        data_.large_scalar = new char[type_.size()];
        ::memcpy(data_.large_scalar, that.data_.large_scalar, type_.size());
    }
}

KernelArg::KernelArg(KernelArg&& that) : KernelArg() {
    std::swap(this->data_, that.data_);
    std::swap(this->type_, that.type_);
    std::swap(this->scalar_, that.scalar_);
}

bool KernelArg::is_array() const {
    return scalar_;
}

bool KernelArg::is_scalar() const {
    return !scalar_;
}

TypeInfo KernelArg::type() const {
    return type_;
}

std::vector<char> KernelArg::to_bytes() const {
    std::vector<char> result;

    if (is_array()) {
        result.resize(type_.size() * data_.array.nelements);
        KERNEL_LAUNCHER_CUDA_CHECK(cuMemcpy(
            reinterpret_cast<CUdeviceptr>(result.data()),
            reinterpret_cast<CUdeviceptr>(data_.array.ptr),
            result.size()));
    } else {
        result.resize(type_.size());

        if (is_inline_scalar(type_)) {
            ::memcpy(result.data(), data_.small_scalar.data(), type_.size());
        } else {
            ::memcpy(result.data(), data_.large_scalar, type_.size());
        }
    }

    return result;
}

KernelArg::KernelArg() : type_(type_of<int>()), scalar_(true) {}

void* KernelArg::as_void_ptr() const {
    if (is_array()) {
        return const_cast<void*>(static_cast<const void*>(&data_.array.ptr));
    } else if (is_inline_scalar(type_)) {
        return const_cast<void*>(
            static_cast<const void*>(data_.small_scalar.data()));
    } else {
        return data_.large_scalar;
    }
}

KernelArg::~KernelArg() {
    if (is_scalar() && is_inline_scalar(type_)) {
        delete[] data_.large_scalar;
    }
}

}  // namespace kernel_launcher