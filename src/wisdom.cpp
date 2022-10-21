#include "kernel_launcher/wisdom.h"

#include <mutex>

#include "kernel_launcher/export.h"
#include "nlohmann/json.hpp"

namespace kernel_launcher {

std::shared_ptr<DefaultOracle> global_default_wisdom = nullptr;

static std::shared_ptr<DefaultOracle> set_global_wisdom(DefaultOracle oracle) {
    auto ptr = std::make_shared<DefaultOracle>(std::move(oracle));
    atomic_store(&global_default_wisdom, ptr);
    return ptr;
}

static std::shared_ptr<DefaultOracle> get_global_wisdom() {
    auto ptr = atomic_load(&global_default_wisdom);

    if (!ptr) {
        ptr = set_global_wisdom(DefaultOracle::from_env());
    }

    return ptr;
}

DefaultOracle::DefaultOracle() : DefaultOracle(*get_global_wisdom()) {}

DefaultOracle::DefaultOracle(
    std::string wisdom_dir,
    std::string capture_dir,
    std::vector<std::string> capture_patterns,
    bool force_capture) :
    wisdom_dir_(std::move(wisdom_dir)),
    capture_dir_(std::move(capture_dir)),
    capture_patterns_(std::move(capture_patterns)),
    force_capture_(force_capture) {}

DefaultOracle DefaultOracle::from_env() {
    std::string wisdom_dir = ".";
    std::string capture_dir = ".";
    std::vector<std::string> capture_patterns = {};
    const char* value;

    if ((value = getenv("KERNEL_LAUNCHER_WISDOM"))) {
        wisdom_dir = value;
        capture_dir = value;
    }

    if ((value = getenv("KERNEL_LAUNCHER_DIR"))) {
        capture_dir = value;
    }

    std::string patterns = "";
    bool force = false;

    // Try the following environment keys in order
    const char* env_keys[4] = {
        "KERNEL_LAUNCHER_CAPTURE_FORCE",
        "KERNEL_LAUNCHER_CAPTURE",
        "KERNEL_LAUNCHER_TUNE_FORCE",
        "KERNEL_LAUNCHER_TUNE",
    };

    for (const char* key : env_keys) {
        if ((value = getenv(key)) == nullptr) {
            continue;
        }

        if (!patterns.empty()) {
            log_warning() << "environment key " << key << " was ignored\n";
            continue;
        }

        patterns = value;
        force = strstr(value, "FORCE") != nullptr;
    }

    if (patterns == "1" || patterns == "true" || patterns == "TRUE") {
        patterns = "*";
    }

    if (patterns == "0" || patterns == "false" || patterns == "FALSE") {
        patterns = "";
    }

    for (std::string pattern : string_split(patterns.c_str(), ',')) {
        if (pattern.empty()) {
            continue;
        }

        capture_patterns.push_back(std::move(pattern));
    }

    // Print info message on which kernels will be tuned.
    if (!capture_patterns.empty()) {
        std::stringstream ss;

        bool needs_comma = false;
        for (const auto& pattern : capture_patterns) {
            if (needs_comma) {
                ss << ", ";
            } else {
                needs_comma = true;
            }

            ss << pattern;
        }

        log_info() << "capture enabled for the following kernels: " << ss.str()
                   << "\n";
    }

    return DefaultOracle(
        std::move(wisdom_dir),
        std::move(capture_dir),
        std::move(capture_patterns),
        force);
}

Config DefaultOracle::load_config(
    const std::string& tuning_key,
    const ConfigSpace& space,
    ProblemSize problem_size,
    CudaDevice device,
    bool* should_capture_out) const {
    WisdomResult result = WisdomResult::Ok;
    Config config = load_best_config(
        wisdom_dir_,
        tuning_key,
        space,
        device.name(),
        device.arch(),
        problem_size,
        &result);

    if (should_capture_out) {
        *should_capture_out =
            this->should_capture_kernel(tuning_key, problem_size, result);
    }

    return config;
}

void DefaultOracle::capture_kernel(
    const std::string& tuning_key,
    const KernelBuilder& builder,
    ProblemSize problem_size,
    const std::vector<TypeInfo>& param_types,
    const std::vector<std::vector<uint8_t>>& inputs,
    const std::vector<std::vector<uint8_t>>& outputs) const {
    export_tuning_file(
        capture_dir_,
        tuning_key,
        builder,
        problem_size,
        param_types,
        inputs,
        outputs);
}

bool DefaultOracle::should_capture_kernel(
    const std::string& tuning_key,
    ProblemSize problem_size,
    WisdomResult result) const {
    bool matches = false;

    // If wisdom was found for this kernel and we do not force tuning,
    // then there is no need to tune this kernel.
    if (result == WisdomResult::Ok && !force_capture_) {
        return false;
    }

    for (const std::string& pattern : capture_patterns_) {
        if (string_match(pattern.c_str(), tuning_key.c_str())) {
            matches = true;
            break;
        }
    }

    if (!matches) {
        return false;
    }

    if (tuning_file_exists(capture_dir_, tuning_key, problem_size)) {
        return false;
    }

    return true;
}

void set_global_wisdom_directory(std::string dir) {
    auto wisdom = get_global_wisdom();

    set_global_wisdom(DefaultOracle(
        std::move(dir),
        wisdom->capture_directory(),
        wisdom->capture_patterns(),
        wisdom->is_capture_forced()));
}

void set_global_tuning_directory(std::string dir) {
    auto wisdom = get_global_wisdom();

    set_global_wisdom(DefaultOracle(
        wisdom->wisdom_directory(),
        std::move(dir),
        wisdom->capture_patterns(),
        wisdom->is_capture_forced()));
}

void add_global_capture_pattern(std::string pattern) {
    auto wisdom = get_global_wisdom();
    std::vector<std::string> patterns = wisdom->capture_patterns();
    patterns.push_back(std::move(pattern));

    set_global_wisdom(DefaultOracle(
        wisdom->wisdom_directory(),
        wisdom->capture_directory(),
        patterns,
        wisdom->is_capture_forced()));
}

WisdomSettings default_wisdom_settings() {
    return get_global_wisdom();
}

WisdomSettings::WisdomSettings() : WisdomSettings(get_global_wisdom()) {}

WisdomSettings::WisdomSettings(std::shared_ptr<Oracle> oracle) :
    impl_(std::move(oracle)) {
    if (!oracle) {
        throw std::runtime_error("Oracle cannot be null");
    }
}

WisdomSettings::WisdomSettings(
    std::string wisdom_dir,
    std::string capture_dir,
    std::vector<std::string> capture_patterns,
    bool force_capture) :
    WisdomSettings(std::make_shared<DefaultOracle>(
        std::move(wisdom_dir),
        std::move(capture_dir),
        std::move(capture_patterns),
        force_capture)) {}

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
    const std::string& line,
    const ConfigSpace& space,
    std::vector<TunableParam>& params,
    std::string& objective) {
    nlohmann::json header = nlohmann::json::parse(line);

    if (header["version"] != "1.0") {
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
static void parse_line(
    const std::string& line,
    const std::vector<TunableParam>& params,
    const std::string& objective_key,
    F callback,
    Config& best_config) {
    std::string empty_string;

    nlohmann::json record = nlohmann::json::parse(line);
    nlohmann::json& config_json = record["config"];
    nlohmann::json& problem_json = record["problem_size"];
    nlohmann::json& env_json = record["environment"];

    if (config_json.size() != params.size()) {
        throw std::runtime_error("invalid configuration");
    }

    ProblemSize problem_size;
    if (problem_json.size() > 0)
        problem_size.x = problem_json[0];
    if (problem_json.size() > 1)
        problem_size.y = problem_json[1];
    if (problem_json.size() > 2)
        problem_size.z = problem_json[2];

    const auto device_name =
        env_json["device_name"].get_ptr<const std::string*>();
    double objective = record[objective_key];

    bool is_better = callback(
        problem_size,
        device_name != nullptr ? *device_name : empty_string,
        objective);

    if (is_better) {
        Config config;
        for (size_t i = 0; i < params.size(); i++) {
            config.insert(params[i], json_to_tunable_value(config_json[i]));
        }

        best_config = std::move(config);
    }
}

template<typename F>
static Config process_wisdom_file(
    const std::string& wisdom_dir,
    const std::string& tuning_key,
    const ConfigSpace& space,
    F callback) {
    Config best_config = space.default_config();

    std::string path =
        path_join(wisdom_dir, sanitize_tuning_key(tuning_key)) + ".wisdom";

    std::ifstream stream(path);
    if (!stream) {
        log_debug() << "wisdom file not found: " << path << "\n";
        return best_config;
    }

    std::string line;
    if (!std::getline(stream, line)) {
        log_debug() << "error while reading wisdom file: " << path << "\n";
        return best_config;
    }

    std::string objective_key;
    std::vector<TunableParam> params;
    std::string err;

    // Parse header
    try {
        parse_header(line, space, params, objective_key);
    } catch (const std::exception& e) {
        log_warning() << path << ":1: file is ignored, error while parsing: "
                      << e.what() << "\n";
        return best_config;
    }

    // Parse each line
    for (size_t lineno = 2; std::getline(stream, line); lineno++) {
        if (line.empty()) {
            continue;
        }

        try {
            parse_line(line, params, objective_key, callback, best_config);
        } catch (const std::exception& e) {
            log_warning() << path << ":" << lineno
                          << ": line is ignored, error while parsing: "
                          << e.what() << "\n";
        }
    }

    return best_config;
}

Config load_best_config(
    const std::string& wisdom_dir,
    const std::string& tuning_key,
    const ConfigSpace& space,
    const std::string& device_name,
    CudaArch device_arch,
    ProblemSize problem_size,
    WisdomResult* result_out) {
    if (space.parameters().empty()) {
        if (result_out != nullptr) {
            *result_out = WisdomResult::Ok;
        }

        return Config {};
    }

    WisdomResult best_type = WisdomResult::NotFound;
    std::string best_device;
    double best_score = 0.0;
    ProblemSize best_problem_size;
    uint64_t best_distance = std::numeric_limits<uint64_t>::max();

    Config best_config = process_wisdom_file(
        wisdom_dir,
        tuning_key,
        space,
        [&](ProblemSize row_problem,
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

            uint64_t l = problem_size.x * problem_size.y * problem_size.z;
            uint64_t r = row_problem.x * row_problem.y * row_problem.z;
            uint64_t distance = l > r ? l - r : r - l;

            if (type > best_type
                || (type == best_type
                    && (distance < best_distance
                        || (distance == best_distance
                            && score < best_score)))) {
                best_type = type;
                best_score = score;
                best_problem_size = row_problem;
                best_device = row_device;
                best_distance = distance;
                return true;
            }

            return false;
        });

    if (best_type == WisdomResult::NotFound) {
        log_warning() << "no wisdom found for kernel \"" << tuning_key
                      << "\" in directory \"" << wisdom_dir
                      << "\", using default kernel configuration\n";
    } else if (best_type == WisdomResult::DeviceMismatch) {
        log_warning() << "no wisdom found for kernel \"" << tuning_key
                      << "\" and device \"" << device_name
                      << "\", using configuration for different device \""
                      << best_device << "\".\n";
    } else if (best_type == WisdomResult::ProblemSizeMismatch) {
        log_info() << "no wisdom found for kernel \"" << tuning_key
                   << "\", device \"" << device_name << "\", and problem size "
                   << problem_size
                   << ", using configuration for different problem size: "
                   << best_problem_size << ".\n";
    }

    log_debug() << "for kernel \"" << tuning_key << "\", device \""
                << device_name << "\", and problem size " << problem_size
                << ", using configuration: " << best_config << ".\n";

    if (result_out != nullptr) {
        *result_out = best_type;
    }

    return best_config;
}

struct WisdomKernelImpl {
    std::mutex mutex_;
    bool compiled_;
    std::string tuning_key_;
    KernelBuilder builder_;
    KernelInstance instance_;
    Compiler compiler_;
    std::vector<TypeInfo> param_types_;
    WisdomSettings settings_;
};

WisdomKernel::WisdomKernel() = default;
WisdomKernel::WisdomKernel(WisdomKernel&&) noexcept = default;
WisdomKernel::~WisdomKernel() = default;

void WisdomKernel::initialize(
    std::string tuning_key,
    KernelBuilder builder,
    Compiler compiler,
    WisdomSettings settings) {
    impl_ = std::unique_ptr<WisdomKernelImpl>(new WisdomKernelImpl {
        {},
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
        std::lock_guard<std::mutex> guard(impl_->mutex_);
        impl_->compiled_ = false;
    }
}

void compile_impl(
    WisdomKernelImpl* impl,
    ProblemSize problem_size,
    CudaDevice device,
    std::vector<TypeInfo> param_types,
    bool* should_capture = nullptr) {
    Config config = impl->settings_.load_config(
        impl->tuning_key_,
        impl->builder_,
        problem_size,
        device,
        should_capture);

    // Assign result to temporary variable since compile may throw
    auto instance =
        impl->builder_.compile(config, param_types, impl->compiler_);

    // Compile was successful. Overwrite fields of impl
    impl->instance_ = std::move(instance);
    impl->param_types_ = std::move(param_types);
    impl->compiled_ = true;
}

void WisdomKernel::compile(
    ProblemSize problem_size,
    CudaDevice device,
    std::vector<TypeInfo> param_types) {
    if (!impl_) {
        throw std::runtime_error("WisdomKernel has not been initialized");
    }

    std::lock_guard<std::mutex> guard(impl_->mutex_);
    compile_impl(
        impl_.get(),
        problem_size,
        device,
        std::move(param_types),
        nullptr);
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

    for (size_t i = 0; i < params.size(); i++) {
        if (i != 0)
            msg += ", ";
        msg += params[i].name();
    }

    msg += "), but was called with argument types (";

    for (size_t i = 0; i < args.size(); i++) {
        if (i != 0)
            msg += ", ";
        msg += args[i].type().name();
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

    std::lock_guard<std::mutex> guard(impl_->mutex_);
    if (!impl_->compiled_) {
        std::vector<TypeInfo> param_types;
        for (const KernelArg& arg : args) {
            param_types.push_back(arg.type());
        }

        bool should_capture = false;
        compile_impl(
            impl_.get(),
            problem_size,
            CudaDevice::current(),
            param_types,
            &should_capture);

        if (should_capture) {
            std::vector<std::vector<uint8_t>> inputs;
            std::vector<std::vector<uint8_t>> outputs;

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
                impl_->settings_.capture_kernel(
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

            return;
        }
    }

    assert_types_equal(args, impl_->param_types_);
    impl_->instance_.launch(stream, problem_size, ptrs.data());
}

static bool is_inline_scalar(TypeInfo type) {
    return type.size() <= sizeof(size_t) * 2;
}

KernelArg::KernelArg(TypeInfo type, void* data) {
    type_ = type.remove_const();
    scalar_ = true;

    if (is_inline_scalar(type_)) {
        ::memcpy(data_.small_scalar.data(), data, type.size());
    } else {
        data_.large_scalar = new char[type.size()];
        ::memcpy(data_.large_scalar, data, type.size());
    }
}

KernelArg::KernelArg(TypeInfo type, void* ptr, size_t nelements) {
    type_ = type.remove_const();
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

KernelArg::~KernelArg() {
    if (is_scalar() && !is_inline_scalar(type_)) {
        delete[](char*) data_.large_scalar;
    }
}

KernelArg::KernelArg(KernelArg&& that) noexcept : KernelArg() {
    std::swap(this->data_, that.data_);
    std::swap(this->type_, that.type_);
    std::swap(this->scalar_, that.scalar_);
}

void KernelArg::assert_type_matches(TypeInfo t) const {
    bool valid;

    if (t.is_pointer()) {
        // Matches if both are pointers and pointee type matches (possible
        // when adding a const modifier).
        valid = type_.is_pointer()
            && (t.remove_pointer() == type_.remove_pointer()
                || t.remove_pointer() == type_.remove_pointer().add_const());
    } else {
        valid = t.remove_const() == type_.remove_const();
    }

    if (!valid) {
        throw std::runtime_error(
            "cannot cast kernel argument of type `" + type_.name()
            + "` to type `" + t.name() + "`");
    }
}

bool KernelArg::is_array() const {
    return !scalar_;
}

bool KernelArg::is_scalar() const {
    return scalar_;
}

TypeInfo KernelArg::type() const {
    return type_;
}

std::vector<uint8_t> KernelArg::to_bytes() const {
    std::vector<uint8_t> result;

    if (is_array()) {
        result.resize(type_.remove_pointer().size() * data_.array.nelements);
        KERNEL_LAUNCHER_CUDA_CHECK(cuMemcpy(
            reinterpret_cast<CUdeviceptr>(result.data()),
            reinterpret_cast<CUdeviceptr>(data_.array.ptr),
            result.size()));
    } else {
        // If the type is a pointer, exporting it to bytes will return
        // the memory address of the pointer and not the data of the buffer it
        // points to This is likely a bug on the user side. Todo: find a better
        //  way f handling this error (maybe already in KernelArg ctor?).
        if (type_.is_pointer()) {
            throw std::runtime_error("a raw pointer type was provided as "
                "kernel argument (" + type_.name() + ") which cannot be "
                "exported since the corresponding buffer size is unknown");
        }

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

}  // namespace kernel_launcher
