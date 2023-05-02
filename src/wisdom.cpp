#include "kernel_launcher/wisdom.h"

#include <mutex>

#include "kernel_launcher/export.h"
#include "nlohmann/json.hpp"

namespace kernel_launcher {

static std::string string_comma_join(const std::vector<std::string>& items) {
    std::stringstream ss;

    bool needs_comma = false;
    for (const auto& pattern : items) {
        if (needs_comma) {
            ss << ", ";
        } else {
            needs_comma = true;
        }

        ss << pattern;
    }

    return ss.str();
}

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

static void parse_header(
    const std::string& path,
    const std::string& line,
    const std::string& tuning_key,
    const ConfigSpace& space,
    std::vector<TunableParam>& params,
    std::string& objective) {
    constexpr static const char* WISDOM_VERSION = "1.0";

    nlohmann::json header = nlohmann::json::parse(line);

    if (header["version"] != WISDOM_VERSION) {
        throw std::runtime_error(
            "invalid version: " + header["version"].dump());
    }

    // Should we throw an exception if the key is incorrect?
    if (header["key"] != tuning_key) {
        throw std::runtime_error("invalid key: " + header["key"].dump());
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

    std::vector<TunableParam> remaining = space.parameters();

    for (const nlohmann::json& record : params_json) {
        if (!record.is_string()) {
            throw std::runtime_error("invalid parameter");
        }

        bool found = false;
        for (auto it = remaining.begin(); it != remaining.end(); it++) {
            if (it->name() == record) {
                found = true;
                params.push_back(*it);
                remaining.erase(it);
                break;
            }
        }

        if (!found) {
            throw std::runtime_error(
                "parameter " + record.dump()
                + +" was not found in the tunable parameters");
        }
    }

    // For each remaining parameter, print a warning. We use the default value
    // for parameters that are not found in the wisdom file. This allows
    // adding additional tunable parameters and still load older wisdom files.
    for (const auto& param : remaining) {
        log_warning() << "parameter \"" << param.name()
                      << "\" is in configuration space of kernel \""
                      << tuning_key
                      << "\" but it was not found in wisdom file: " << path
                      << "\n";
    }
}

static Value json_to_tunable_value(const nlohmann::json& value) {
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
            std::string repr = value.dump();
            throw std::runtime_error("cannot interpret json value: " + repr);
    }
}

static const std::string EMPTY_STRING;

struct WisdomRecordImpl {
    nlohmann::json record;
    std::vector<TunableParam> params;
    std::string objective_key;
    const ConfigSpace& config_space;
};

ProblemSize WisdomRecord::problem_size() const {
    const nlohmann::json& problem_json = impl_.record["problem_size"];

    ProblemSize problem_size;
    for (size_t i = 0; i < 3; i++) {
        if (problem_json.size() > i) {
            problem_size[i] = problem_json[i];
        }
    }

    return problem_size;
}

double WisdomRecord::objective() const {
    return impl_.record[impl_.objective_key];
}

const std::string& WisdomRecord::environment(const char* key) const {
    if (const auto* value = impl_.record.at("environment")
                                .at(key)
                                .get_ptr<const std::string*>()) {
        return *value;
    }

    log_warning() << "not found " << key << "\n";
    return EMPTY_STRING;
}

const std::string& WisdomRecord::device_name() const {
    return environment("device_name");
}

Config WisdomRecord::config() const {
    const nlohmann::json& config_json = impl_.record["config"];
    if (config_json.size() != impl_.params.size()) {
        throw std::runtime_error("invalid configuration");
    }

    Config result = impl_.config_space.default_config();

    for (size_t i = 0; i < impl_.params.size(); i++) {
        Value value = json_to_tunable_value(config_json[i]);
        if (!impl_.params[i].has_value(value)) {
            throw std::runtime_error(
                "invalid configuration: parameter \"" + impl_.params[i].name()
                + "\" cannot have \"" + value.to_string() + "\"");
        }

        result.insert(impl_.params[i], value);
    }

    return result;
}

template<typename F>
static bool process_wisdom_file_impl(
    const std::string& wisdom_dir,
    const std::string& tuning_key,
    const ConfigSpace& space,
    F callback) {
    std::string path =
        path_join(wisdom_dir, sanitize_tuning_key(tuning_key)) + ".wisdom";

    std::ifstream stream(path);
    if (!stream) {
        log_debug() << "wisdom file not found: " << path << "\n";
        return false;
    }

    std::string line;
    if (!std::getline(stream, line)) {
        log_debug() << "error while reading wisdom file: " << path << "\n";
        return false;
    }

    std::string objective_key;
    std::vector<TunableParam> params;

    // Parse header
    try {
        parse_header(path, line, tuning_key, space, params, objective_key);
    } catch (const std::exception& e) {
        log_warning() << path << ":1: file is ignored, error while parsing: "
                      << e.what() << "\n";
        return false;
    }

    WisdomRecordImpl impl {
        nullptr,
        std::move(params),
        std::move(objective_key),
        space};

    // Parse each line
    for (size_t lineno = 2; std::getline(stream, line); lineno++) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        try {
            impl.record = nlohmann::json::parse(line);
            callback(WisdomRecord {impl});
        } catch (const std::exception& e) {
            log_warning() << path << ":" << lineno
                          << ": line is ignored, error while parsing: "
                          << e.what() << "\n";
        }
    }

    return true;
}

bool process_wisdom_file(
    const std::string& wisdom_dir,
    const std::string& tuning_key,
    const ConfigSpace& space,
    std::function<void(const WisdomRecord&)> callback) {
    if (!callback) {
        return false;
    }

    return process_wisdom_file_impl(
        wisdom_dir,
        tuning_key,
        space,
        std::move(callback));
}

Config load_best_config(
    const std::vector<std::string>& wisdom_dirs,
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

    Config best_config = space.default_config();
    WisdomResult best_type = WisdomResult::NotFound;
    std::string best_device;
    double best_score = 0.0;
    ProblemSize best_problem_size;
    uint64_t best_distance = std::numeric_limits<uint64_t>::max();

    log_debug() << "parsing " << tuning_key << "\n";
    auto callback = [&](const WisdomRecord& record) {
        double score = record.objective();
        const std::string& row_device = record.device_name();
        ProblemSize row_problem = record.problem_size();

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

        if (type < best_type
            || (type == best_type
                && (distance < best_distance
                    || (distance == best_distance && score < best_score)))) {
            best_type = type;
            best_score = score;
            best_problem_size = row_problem;
            best_device = row_device;
            best_distance = distance;
            best_config = record.config();
            return true;
        }

        return false;
    };

    for (const auto& dir : wisdom_dirs) {
        if (process_wisdom_file_impl(dir, tuning_key, space, callback)) {
            break;
        }
    }

    if (best_type == WisdomResult::NotFound) {
        log_warning() << "using default configuration for kernel \""
                      << tuning_key
                      << "\", no wisdom found in the following directories: "
                      << string_comma_join(wisdom_dirs) << "\n";
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

Config load_best_config(
    const std::string& wisdom_dir,
    const std::string& tuning_key,
    const ConfigSpace& space,
    const std::string& device_name,
    CudaArch device_arch,
    ProblemSize problem_size,
    WisdomResult* result_out) {
    return load_best_config(
        std::vector<std::string> {wisdom_dir},
        tuning_key,
        space,
        device_name,
        device_arch,
        problem_size,
        result_out);
}

std::shared_ptr<DefaultWisdomSettings> global_default_wisdom = nullptr;

static std::shared_ptr<DefaultWisdomSettings>
set_global_wisdom(DefaultWisdomSettings oracle) {
    auto ptr = std::make_shared<DefaultWisdomSettings>(std::move(oracle));
    atomic_store(&global_default_wisdom, ptr);
    return ptr;
}

static std::shared_ptr<DefaultWisdomSettings> get_global_wisdom() {
    auto ptr = atomic_load(&global_default_wisdom);

    if (!ptr) {
        ptr = set_global_wisdom(DefaultWisdomSettings::from_env());
    }

    return ptr;
}

DefaultWisdomSettings::DefaultWisdomSettings() :
    DefaultWisdomSettings(*get_global_wisdom()) {}

DefaultWisdomSettings::DefaultWisdomSettings(
    std::vector<std::string> wisdom_dirs,
    std::string capture_dir,
    std::vector<CaptureRule> capture_rules) :
    wisdom_dirs_(std::move(wisdom_dirs)),
    capture_dir_(std::move(capture_dir)),
    capture_rules_(std::move(capture_rules)) {}

static std::vector<CaptureRule> determine_capture_rules() {
    const char* value;
    int skip = 0;

    if ((value = getenv("KERNEL_LAUNCHER_CAPTURE_SKIP")) != nullptr
        || (value = getenv("KERNEL_LAUNCHER_SKIP")) != nullptr) {
        bool valid;

        try {
            skip = std::stoi(value);
            valid = skip >= 0;
        } catch (const std::exception& e) {
            valid = false;
        }

        if (!valid) {
            log_warning() << "failed to parse KERNEL_LAUNCHER_CAPTURE_SKIP, "
                             "kernel capturing is now be disabled";
            return {};
        }
    }

    std::vector<CaptureRule> result = {};

    // Try the following environment keys
    const char* env_keys[6] = {
        "KERNEL_LAUNCHER_CAPTURE_FORCE",
        "KERNEL_LAUNCHER_CAPTURE",
        "KERNEL_LAUNCHER_FORCE_CAPTURE",
        "KERNEL_LAUNCHER_TUNE_FORCE",
        "KERNEL_LAUNCHER_FORCE_TUNE",
        "KERNEL_LAUNCHER_TUNE",
    };

    for (const char* key : env_keys) {
        // Check if variable exists
        if ((value = getenv(key)) == nullptr) {
            continue;
        }

        bool force = strstr(key, "FORCE") != nullptr;
        std::string patterns = value;

        // Some patterns are special cased
        if (patterns == "1" || patterns == "true" || patterns == "TRUE") {
            patterns = "*";
        }

        if (patterns == "0" || patterns == "false" || patterns == "FALSE") {
            patterns = "";
        }

        for (auto pattern : string_split(patterns.c_str(), {',', '|', ';'})) {
            if (!pattern.empty()) {
                result.emplace_back(std::move(pattern), force, skip);
            }
        }
    }

    return result;
}

DefaultWisdomSettings DefaultWisdomSettings::from_env() {
    std::vector<std::string> wisdom_dirs = {"."};
    std::string capture_dir = ".";
    const char* value;

    if ((value = getenv("KERNEL_LAUNCHER_WISDOM")) != nullptr) {
        for (std::string dir : string_split(value, {':', ';', ','})) {
            if (!dir.empty()) {
                wisdom_dirs.emplace_back(std::move(dir));
            }
        }

        if (!wisdom_dirs.empty()) {
            capture_dir = wisdom_dirs[0];
        }
    }

    // capture directory
    if ((value = getenv("KERNEL_LAUNCHER_CAPTURE_DIR")) != nullptr
        || (value = getenv("KERNEL_LAUNCHER_DIR")) != nullptr) {
        capture_dir = value;
    }

    auto capture_rules = determine_capture_rules();

    // Print info message on which kernels will be tuned.
    if (!capture_rules.empty() && log_info_enabled()) {
        std::vector<std::string> names;
        for (const auto& p : capture_rules) {
            names.push_back(p.pattern);
        }

        log_info() << "the following kernels will be captured: "
                   << string_comma_join(names) << "\n";
    }

    return DefaultWisdomSettings(
        std::move(wisdom_dirs),
        std::move(capture_dir),
        std::move(capture_rules));
}

Config DefaultWisdomSettings::load_config(
    const std::string& tuning_key,
    const ConfigSpace& space,
    ProblemSize problem_size,
    CudaDevice device,
    int* capture_skip_launches_out) const {
    WisdomResult result = WisdomResult::Ok;
    Config config = load_best_config(
        wisdom_dirs_,
        tuning_key,
        space,
        device.name(),
        device.arch(),
        problem_size,
        &result);

    if (capture_skip_launches_out != nullptr) {
        int skip;

        if (!this->should_capture_kernel(
                tuning_key,
                problem_size,
                result,
                skip)) {
            skip = -1;
        }

        *capture_skip_launches_out = skip;
    }

    return config;
}

void DefaultWisdomSettings::capture_kernel(
    const std::string& tuning_key,
    const KernelBuilder& builder,
    ProblemSize problem_size,
    const std::vector<KernelArg>& arguments,
    const std::vector<std::vector<uint8_t>>& input_arrays,
    const std::vector<std::vector<uint8_t>>& output_arrays) const {
    export_capture_file(
        capture_dir_,
        tuning_key,
        builder,
        problem_size,
        arguments,
        input_arrays,
        output_arrays);
}

bool DefaultWisdomSettings::should_capture_kernel(
    const std::string& tuning_key,
    ProblemSize problem_size,
    WisdomResult result,
    int& skip_launches_out) const {
    bool matches = false;
    bool forced = false;
    int skip = std::numeric_limits<int>::max();

    for (const auto& rule : capture_rules_) {
        if (string_match(rule.pattern.c_str(), tuning_key.c_str())) {
            matches = true;
            forced |= rule.force;
            skip = std::min(skip, rule.skip_launches);
        }
    }

    // No rule matches. We are done.
    if (!matches) {
        return false;
    }

    // If wisdom was found for this kernel, and we do not force tuning,
    // then there is no need to capture this kernel.
    if (result == WisdomResult::Ok && !forced) {
        return false;
    }

    if (capture_file_exists(capture_dir_, tuning_key, problem_size)) {
        return false;
    }

    skip_launches_out = skip;
    return true;
}

void append_global_wisdom_directory(std::string dir) {
    auto wisdom = get_global_wisdom();

    auto dirs = wisdom->wisdom_directories();
    dirs.push_back(std::move(dir));

    set_global_wisdom(DefaultWisdomSettings(
        std::move(dirs),
        wisdom->capture_directory(),
        wisdom->capture_rules()));
}

void set_global_wisdom_directory(std::string dir) {
    auto wisdom = get_global_wisdom();

    set_global_wisdom(DefaultWisdomSettings(
        std::vector<std::string> {std::move(dir)},
        wisdom->capture_directory(),
        wisdom->capture_rules()));
}

void set_global_capture_directory(std::string dir) {
    auto wisdom = get_global_wisdom();

    set_global_wisdom(DefaultWisdomSettings(
        wisdom->wisdom_directories(),
        std::move(dir),
        wisdom->capture_rules()));
}

void add_global_capture_pattern(CaptureRule rule) {
    auto wisdom = get_global_wisdom();
    std::vector<CaptureRule> rules = wisdom->capture_rules();
    rules.push_back(std::move(rule));

    set_global_wisdom(DefaultWisdomSettings(
        wisdom->wisdom_directories(),
        wisdom->capture_directory(),
        rules));
}

WisdomSettings default_wisdom_settings() {
    return get_global_wisdom();
}

WisdomSettings::WisdomSettings() : WisdomSettings(get_global_wisdom()) {}

WisdomSettings::WisdomSettings(std::shared_ptr<IWisdomSettings> oracle) :
    impl_(std::move(oracle)) {
    if (!impl_) {
        throw std::runtime_error("Oracle cannot be null");
    }
}

WisdomSettings::WisdomSettings(
    std::string wisdom_dir,
    std::string capture_dir,
    std::vector<CaptureRule> capture_rules) :
    WisdomSettings(std::make_shared<DefaultWisdomSettings>(
        std::vector<std::string> {std::move(wisdom_dir)},
        std::move(capture_dir),
        std::move(capture_rules))) {}

}  // namespace kernel_launcher
