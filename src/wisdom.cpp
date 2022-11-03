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
                      << "\" but it was not found in wisdom file:" << path
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
    for (size_t i = 0; i < 3; i++) {
        if (problem_json.size() > i) {
            problem_size[i] = problem_json[i];
        }
    }

    const auto* const device_name =
        env_json["device_name"].get_ptr<const std::string*>();
    double objective = record[objective_key];

    bool is_better = callback(
        problem_size,
        device_name != nullptr ? *device_name : empty_string,
        objective);

    if (is_better) {
        for (size_t i = 0; i < params.size(); i++) {
            best_config.insert(
                params[i],
                json_to_tunable_value(config_json[i]));
        }
    }
}

template<typename F>
static bool process_wisdom_file(
    const std::string& wisdom_dir,
    const std::string& tuning_key,
    const ConfigSpace& space,
    Config& best_config,
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
    std::string err;

    // Parse header
    try {
        parse_header(path, line, tuning_key, space, params, objective_key);
    } catch (const std::exception& e) {
        log_warning() << path << ":1: file is ignored, error while parsing: "
                      << e.what() << "\n";
        return true;
    }

    // Parse each line
    for (size_t lineno = 2; std::getline(stream, line); lineno++) {
        if (line.empty() || line[0] == '#') {
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

    return true;
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

    for (const auto& dir : wisdom_dirs) {
        bool success = process_wisdom_file(
            dir,
            tuning_key,
            space,
            best_config,
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

        if (success) {
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
    std::vector<std::string> wisdom_dirs,
    std::string capture_dir,
    std::vector<std::string> capture_patterns,
    bool force_capture) :
    wisdom_dirs_(std::move(wisdom_dirs)),
    capture_dir_(std::move(capture_dir)),
    capture_patterns_(std::move(capture_patterns)),
    force_capture_(force_capture) {}

DefaultOracle DefaultOracle::from_env() {
    std::vector<std::string> wisdom_dirs = {"."};
    std::string capture_dir = ".";
    std::vector<std::string> capture_patterns = {};
    const char* value;

    if ((value = getenv("KERNEL_LAUNCHER_WISDOM")) != nullptr) {
        for (std::string dir : string_split(value, ':')) {
            if (!dir.empty()) {
                wisdom_dirs.emplace_back(std::move(dir));
            }
        }

        if (!wisdom_dirs.empty()) {
            capture_dir = wisdom_dirs[0];
        }
    }

    if ((value = getenv("KERNEL_LAUNCHER_DIR")) != nullptr) {
        capture_dir = value;
    }

    std::string patterns;
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

        log_info() << "capture enabled for the following kernels: "
                   << string_comma_join(capture_patterns) << "\n";
    }

    return DefaultOracle(
        std::move(wisdom_dirs),
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
        wisdom_dirs_,
        tuning_key,
        space,
        device.name(),
        device.arch(),
        problem_size,
        &result);

    if (should_capture_out != nullptr) {
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

void append_global_wisdom_directory(std::string dir) {
    auto wisdom = get_global_wisdom();

    auto dirs = wisdom->wisdom_directories();
    dirs.push_back(std::move(dir));

    set_global_wisdom(DefaultOracle(
        std::move(dirs),
        wisdom->capture_directory(),
        wisdom->capture_patterns(),
        wisdom->is_capture_forced()));
}

void set_global_wisdom_directory(std::string dir) {
    auto wisdom = get_global_wisdom();

    set_global_wisdom(DefaultOracle(
        std::vector<std::string> {std::move(dir)},
        wisdom->capture_directory(),
        wisdom->capture_patterns(),
        wisdom->is_capture_forced()));
}

void set_global_capture_directory(std::string dir) {
    auto wisdom = get_global_wisdom();

    set_global_wisdom(DefaultOracle(
        wisdom->wisdom_directories(),
        std::move(dir),
        wisdom->capture_patterns(),
        wisdom->is_capture_forced()));
}

void add_global_capture_pattern(std::string pattern) {
    auto wisdom = get_global_wisdom();
    std::vector<std::string> patterns = wisdom->capture_patterns();
    patterns.push_back(std::move(pattern));

    set_global_wisdom(DefaultOracle(
        wisdom->wisdom_directories(),
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
    if (!impl_) {
        throw std::runtime_error("Oracle cannot be null");
    }
}

WisdomSettings::WisdomSettings(
    std::string wisdom_dir,
    std::string capture_dir,
    std::vector<std::string> capture_patterns,
    bool force_capture) :
    WisdomSettings(std::make_shared<DefaultOracle>(
        std::vector<std::string> {std::move(wisdom_dir)},
        std::move(capture_dir),
        std::move(capture_patterns),
        force_capture)) {}

}  // namespace kernel_launcher
