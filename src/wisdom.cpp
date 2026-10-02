#include "kernel_launcher/wisdom.hpp"

#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <tuple>

#include "kernel_launcher/file_loader.hpp"
#include "nlohmann/json.hpp"

namespace kernel_launcher {

static Primitive json_to_primitive(const nlohmann::json& value) {
    if (value.is_boolean()) {
        return Primitive(value.get<bool>());
    } else if (value.is_number_integer()) {
        return Primitive(value.get<std::intmax_t>());
    } else if (value.is_number_float()) {
        return Primitive(value.get<double>());
    } else if (value.is_string()) {
        return Primitive(value.get<std::string>());
    } else {
        throw std::runtime_error("invalid value: " + value.dump());
    }
}

static std::string sanitize_tuning_key(std::string_view key) {
    std::string output {key};

    for (char& c : output) {
        bool valid = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
            || c == '_' || c == '-' || c == '.';

        if (!valid) {
            c = '_';
        }
    }

    return output;
}

struct WisdomRecord::Impl {
    Config config;
    double score = 0.0;
    std::array<int64_t, MAX_PROBLEM_DIMS> problem_size {};
    nlohmann::json environment = {};
    nlohmann::json features = {};
};

const Config& WisdomRecord::config() const {
    return impl_->config;
}

double WisdomRecord::objective() const {
    return impl_->score;
}

std::array<int64_t, MAX_PROBLEM_DIMS> WisdomRecord::problem_size() const {
    return impl_->problem_size;
}

std::string_view WisdomRecord::env(std::string_view key) const {
    const nlohmann::json& environment = impl_->environment;
    auto it = environment.find(key);

    // not found
    if (it == environment.end()) {
        return "";
    }

    // found
    if (const auto* p = it->get_ptr<const std::string*>()) {
        return *p;
    }

    // in rare cases, the environment may contain a field which is not a string. To work around
    // this, we dump the content to a string and then intern it to keep it alive after this call.
    return intern_string(it->dump());
}

Primitive WisdomRecord::feature(std::string_view key) const {
    const nlohmann::json& features = impl_->features;
    auto it = features.find(key);

    if (it == features.end()) {
        return {};
    }

    try {
        return json_to_primitive(*it);
    } catch (const std::exception& e) {
        log_warning() << "invalid value for feature \"" << key << "\": " << e.what() << "\n";
        return {};
    }
}

// Must match `WISDOM_VERSION` and `WISDOM_OBJECTIVE` in `python/kernel_launcher/wisdom.py`,
// which also writes these files.
static constexpr const char* WISDOM_VERSION = "1.0";
static constexpr const char* WISDOM_OBJECTIVE = "time";

static const nlohmann::json&
get_field(const nlohmann::json& object, const char* name, nlohmann::json::value_t type) {
    auto it = object.find(name);

    if (it == object.end()) {
        throw std::runtime_error(concat("missing field \"", name, "\""));
    }

    if (it->type() != type) {
        throw std::runtime_error(concat(
            "field \"",
            name,
            "\" must be of type ",
            nlohmann::json(type).type_name(),
            ", got: ",
            it->dump()));
    }

    return *it;
}

static void parse_header(
    const std::string& line,
    std::string_view tuning_key,
    const ConfigSpace& space,
    std::vector<TunableParam>& params_out,
    std::string& objective_out) {
    auto header = nlohmann::json::parse(line);

    if (!header.is_object()) {
        throw std::runtime_error("header must be a JSON object, got: " + header.dump());
    }

    // Check the version first, since the other fields may differ between versions.
    const auto& version = get_field(header, "version", nlohmann::json::value_t::string);
    if (version != WISDOM_VERSION) {
        throw std::runtime_error("version is invalid: " + version.dump() + " != " + WISDOM_VERSION);
    }

    objective_out = get_field(header, "objective", nlohmann::json::value_t::string);
    if (objective_out.empty()) {
        throw std::runtime_error("objective is not defined");
    }

    const auto& key = get_field(header, "key", nlohmann::json::value_t::string);
    if (key != tuning_key) {
        throw std::runtime_error(
            "tuning key is invalid: " + key.dump() + " != " + std::string(tuning_key));
    }

    std::vector<TunableParam> remaining = space.parameters();
    const auto& params_json =
        get_field(header, "tunable_parameters", nlohmann::json::value_t::array);

    for (const auto& record : params_json) {
        if (!record.is_string()) {
            throw std::runtime_error("invalid tunable parameter: " + record.dump());
        }

        bool found = false;
        for (auto it = remaining.begin(); it != remaining.end(); it++) {
            if (it->name() == record) {
                found = true;
                params_out.push_back(*it);
                remaining.erase(it);
                break;
            }
        }

        if (!found) {
            throw std::runtime_error(
                "parameter was not found in the tunable parameters: " + record.dump());
        }
    }

    // For each remaining parameter, print a warning. We will use the default value
    // for parameters that are not found in the wisdom file. This allows
    // adding additional tunable parameters and still load older wisdom files.
    for (const auto& param : remaining) {
        log_warning() << "parameter \"" << param.name()
                      << "\" is in configuration space of kernel \"" << tuning_key
                      << "\" but it was not found in wisdom file, "
                      << "default value " << param.default_value() << " will be used instead\n";
    }
}

static nlohmann::json parse_optional_object(nlohmann::json& record, const char* key) {
    auto it = record.find(key);

    if (it == record.end()) {
        return nlohmann::json::object();
    }

    if (!it->is_object()) {
        throw std::runtime_error(concat("invalid ", key, ": ", it->dump()));
    }

    return std::move(*it);
}

static void parse_record(
    const std::string& line,
    const ConfigSpace& space,
    const std::vector<TunableParam>& params,
    const std::string& objective,
    WisdomRecord::Impl& impl_out) {
    auto record = nlohmann::json::parse(line);

    if (!record.is_object()) {
        throw std::runtime_error("record must be a JSON object, got: " + record.dump());
    }

    const nlohmann::json& values = get_field(record, "config", nlohmann::json::value_t::array);

    if (values.size() != params.size()) {
        throw std::runtime_error(
            concat("field \"config\" must have ", params.size(), " values, got: ", values.dump()));
    }

    // Insert the matching value from the parameter itself rather than the parsed value, since
    // the Python writer may store floats such as `2.0` as integers `2`.
    for (size_t i = 0; i < params.size(); i++) {
        Primitive parsed = json_to_primitive(values[i]);
        const auto& allowed = params[i].values();
        auto it = std::find(allowed.begin(), allowed.end(), parsed);

        if (it == allowed.end()) {
            throw std::runtime_error(
                concat("invalid value for parameter ", params[i].name(), ": ", parsed));
        }

        impl_out.config.insert(params[i].name(), *it);
    }

    if (!space.is_valid(impl_out.config)) {
        throw std::runtime_error(concat("config not valid for this problem: ", impl_out.config));
    }

    const auto& problem_size = get_field(record, "problem_size", nlohmann::json::value_t::array);
    if (problem_size.size() > MAX_PROBLEM_DIMS) {
        throw std::runtime_error("invalid problem size: " + problem_size.dump());
    }

    for (size_t i = 0; i < MAX_PROBLEM_DIMS; i++) {
        impl_out.problem_size[i] = i < problem_size.size() ? problem_size[i].get<int64_t>() : 1;
    }

    impl_out.score = record.at(objective).get<double>();
    impl_out.environment = parse_optional_object(record, "environment");
    impl_out.features = parse_optional_object(record, "features");
}

template<typename F>
bool process_wisdom_file_impl(
    std::string_view wisdom_file,
    const ConfigSpace& space,
    std::string_view tuning_key,
    const F& callback) {
    std::ifstream stream {std::string(wisdom_file)};

    if (!stream) {
        // A missing wisdom file is normal (the kernel has not been tuned yet), so stay silent.
        if (errno != ENOENT && errno != ENOTDIR) {
            log_warning() << wisdom_file << ": failed to open file: " << std::strerror(errno)
                          << "\n";
        }

        return false;
    }

    std::string line;
    if (!std::getline(stream, line)) {
        if (stream.bad()) {
            log_warning() << wisdom_file << ": error while reading file\n";
        } else {
            log_warning() << wisdom_file << ": file is empty\n";
        }

        return false;
    }

    std::string objective;
    std::vector<TunableParam> params;
    try {
        parse_header(line, tuning_key, space, params, objective);
    } catch (const std::exception& e) {
        log_warning() << wisdom_file << ":1: invalid header: " << e.what() << "\n";
        return false;
    }

    size_t lineno = 1;
    WisdomRecord::Impl impl = {space.default_config()};

    while (std::getline(stream, line)) {
        lineno++;

        // skip empty lines and commented lines
        if (line.empty() || line[0] == '#' || line.rfind("//", 0) == 0) {
            continue;
        }

        try {
            parse_record(line, space, params, objective, impl);

            callback(WisdomRecord(&impl));
        } catch (const std::exception& e) {
            log_warning() << wisdom_file << ":" << lineno << ": skipping invalid line: " << e.what()
                          << "\n";
        }
    }

    if (stream.bad()) {
        log_warning() << wisdom_file << ":" << lineno << ": error while reading file\n";
    }

    return true;
}

template<typename F>
bool process_wisdom_files_impl(
    const std::vector<std::string_view>& wisdom_dirs,
    const ConfigSpace& space,
    std::string_view tuning_key,
    const F& callback) {
    bool found = false;

    for (std::string_view wisdom_dir : wisdom_dirs) {
        std::string wisdom_file =
            path_join(wisdom_dir, sanitize_tuning_key(tuning_key)) + ".wisdom";

        found |= process_wisdom_file_impl(wisdom_file, space, tuning_key, callback);
    }

    if (!found) {
        std::string dirs;
        for (std::string_view wisdom_dir : wisdom_dirs) {
            if (!dirs.empty()) {
                dirs += ", ";
            }

            dirs += wisdom_dir;
        }

        if (dirs.empty()) {
            dirs = "<no directories specified>";
        }

        log_warning() << "no wisdom file found for kernel \"" << tuning_key
                      << "\" in any of the directories: " << dirs << "\n";
    }

    return found;
}

void process_wisdom_files(
    const std::vector<std::string_view>& wisdom_dirs,
    const ConfigSpace& space,
    std::string_view tuning_key,
    std::function<void(WisdomRecord)> callback) {
    process_wisdom_files_impl(wisdom_dirs, space, tuning_key, callback);
}

static double problem_distance(const WisdomRecord& record, const ProblemKey& problem) {
    auto a = record.problem_size();
    auto b = problem.problem_size;
    double result = 0.0;

    if (a != b) {
        for (size_t i = 0; i < MAX_PROBLEM_DIMS; i++) {
            double x = double(std::max(a[i], int64_t(1)));
            double y = double(std::max(b[i], int64_t(1)));
            double diff = std::log(x / y);
            result += diff * diff;
        }
    }

    return result;
}

static int feature_distance(const WisdomRecord& record, const ProblemKey& problem) {
    int features_incorrect = 0;

    for (const auto& p : problem.features) {
        if (record.feature(p.first.get()) != p.second) {
            features_incorrect++;
        }
    }

    return features_incorrect;
}

static int device_distance(const WisdomRecord& record, const GPUArch& arch) {
    if (record.env("device_name") == arch.name()) {
        return 0;
    }

    std::string_view target = arch.get();
    std::string other(record.env("compute_capability"));

    // Older wisdom files store the bare compute capability (e.g. "80" instead of "sm_80").
    if (!other.empty() && other[0] >= '0' && other[0] <= '9') {
        other = "sm_" + other;
    }

    if (!target.empty() && other == target) {
        return 1;
    }

    // The prefix is the vendor: `sm_` for NVIDIA and `gfx` for AMD.
    bool both_nvidia = target.find("sm_") == 0 && other.find("sm_") == 0;
    bool both_amd = target.find("gfx") == 0 && other.find("gfx") == 0;

    if (both_nvidia || both_amd) {
        return 2;
    }

    return 3;
}

Config load_wisdom_config(
    const std::vector<std::string_view>& wisdom_dirs,
    const GPUArch& arch,
    const ConfigSpace& space,
    const ProblemKey& problem,
    WisdomResult* result_out) {
    Config best_config = space.default_config();
    std::tuple<int, double, int, double> best_score = {
        std::numeric_limits<int>::max(),
        std::numeric_limits<double>::max(),
        std::numeric_limits<int>::max(),
        std::numeric_limits<double>::max()};

    // Only set once a record has been accepted, since a file can exist without
    // containing any configuration that is valid for this space.
    bool found = false;

    process_wisdom_files_impl(
        wisdom_dirs,
        space,
        problem.tuning_key,
        [&](const WisdomRecord& record) {
            found = true;
            auto [best_d1, best_d2, best_d3, best_d4] = best_score;

            double d1 = device_distance(record, arch);
            if (d1 > best_d1) {
                return;
            }

            double d2 = problem_distance(record, problem);
            if (d1 == best_d1 && d2 > best_d2) {
                return;
            }

            double d3 = feature_distance(record, problem);
            if (d1 == best_d1 && d2 == best_d2 && d3 > best_d3) {
                return;
            }

            double d4 = record.objective();
            if (d1 == best_d1 && d2 == best_d2 && d3 == best_d3 && d4 > best_d4) {
                return;
            }

            best_score = std::make_tuple(d1, d2, d3, d4);
            best_config = record.config();
        });

    auto [best_d1, best_d2, best_d3, best_d4] = best_score;

    if (!found) {
        *result_out = WisdomResult::NotFound;
    } else if (best_d1 != 0) {
        *result_out = WisdomResult::DeviceMismatch;
    } else if (best_d2 != 0 || best_d3 != 0) {
        *result_out = WisdomResult::ProblemMismatch;
    } else {
        *result_out = WisdomResult::Ok;
    }

    return best_config;
}

static nlohmann::json primitive_to_json(const Primitive& value) {
    switch (value.kind()) {
        case Primitive::Kind::Int64:
            return value.to_integer();
        case Primitive::Kind::Float64:
            return value.to_float();
        case Primitive::Kind::String:
            return value.to_string();
        case Primitive::Kind::Bool:
            return value.to_bool();
        default:
            throw std::runtime_error(concat("invalid value: ", value));
    }
}

static nlohmann::json create_header(
    std::string_view tuning_key,
    const std::vector<TunableParam>& params,
    std::string_view objective_key) {
    auto names = nlohmann::json::array();
    for (const auto& param : params) {
        names.push_back(std::string(param.name()));
    }

    return {
        {"version", WISDOM_VERSION},
        {"objective", objective_key},
        {"tunable_parameters", std::move(names)},
        {"key", std::string(tuning_key)}};
}

// The environment is informational only, so a version that cannot be determined is written as -1
// instead of failing the write.
static int version_or_unknown(int (*get_version)()) {
    try {
        return get_version();
    } catch (const std::exception&) {
        return -1;
    }
}

// Key names match `_build_environment` in `python/kernel_launcher/wisdom.py`.
static nlohmann::json create_environment(const GPUArch& arch) {
    std::time_t t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm local_time {};
    localtime_r(&t, &local_time);
    std::stringstream date;
    date << std::put_time(&local_time, "%FT%T%z");

    char host_name[256] = {0};
    gethostname(host_name, sizeof(host_name) - 1);

    return {
        {"device_name", arch.name()},
        {"compute_capability", arch.get()},
        {"date", date.str()},
        {"host_name", host_name},
        {"cuda_driver_version", version_or_unknown(gpu_driver_version)},
        {"cuda_version", version_or_unknown(gpu_runtime_version)},
        {"nvrtc_version", version_or_unknown(RuntimeCompiler::version)}};
}

static nlohmann::json create_record(
    const GPUArch& arch,
    const ProblemKey& problem,
    const std::vector<TunableParam>& params,
    std::string_view objective_key,
    const Config& config,
    double objective) {
    auto values = nlohmann::json::array();
    for (const auto& param : params) {
        values.push_back(primitive_to_json(config[param.name()]));
    }

    auto features = nlohmann::json::object();
    for (const auto& [key, value] : problem.features) {
        features[std::string(key.get())] = primitive_to_json(value);
    }

    return {
        {"config", std::move(values)},
        {"problem_size", problem.problem_size},
        {objective_key, objective},
        {"environment", create_environment(arch)},
        {"features", std::move(features)}};
}

void write_wisdom_record(
    std::string_view wisdom_dir,
    const GPUArch& arch,
    const ConfigSpace& space,
    const ProblemKey& problem,
    const Config& config,
    double objective) {
    // JSON cannot represent NaN or infinity, and the reader would skip the record.
    if (!std::isfinite(objective)) {
        throw std::invalid_argument(concat("objective must be finite, got: ", objective));
    }

    if (!space.is_valid(config)) {
        throw std::runtime_error(concat("config not valid for this problem: ", config));
    }

    std::string wisdom_file =
        path_join(wisdom_dir, sanitize_tuning_key(problem.tuning_key)) + ".wisdom";

    // Serializes writes from multiple threads. This does not protect against other processes.
    static std::mutex mutex;
    std::lock_guard<std::mutex> guard(mutex);

    // Opening with `in | app` creates the file if it does not exist, reads from the start, and
    // always writes at the end. If two processes create the same file concurrently, both may
    // write a header. This is harmless: the second header is skipped as an invalid record.
    std::fstream stream {wisdom_file, std::ios::in | std::ios::app};

    if (!stream) {
        throw std::runtime_error(
            concat("failed to open ", wisdom_file, ": ", std::strerror(errno)));
    }

    // For a new file, use the parameters from `space`. Otherwise, use the order and objective
    // from the existing header.
    std::vector<TunableParam> params = space.parameters();
    std::string objective_key = WISDOM_OBJECTIVE;
    std::string line;
    bool has_header = bool(std::getline(stream, line));

    if (stream.bad()) {
        throw std::runtime_error(concat("failed to read ", wisdom_file));
    }

    // Clear the EOF flag from reading.
    stream.clear();

    if (has_header) {
        params.clear();

        try {
            parse_header(line, problem.tuning_key, space, params, objective_key);
        } catch (const std::exception& e) {
            throw std::runtime_error(concat(wisdom_file, ":1: invalid header: ", e.what()));
        }

        // Unlike when reading, parameters that are missing from the file are an error, since
        // their values cannot be stored.
        if (params.size() != space.parameters().size()) {
            throw std::runtime_error(concat(
                wisdom_file,
                ":1: invalid header: not all tunable parameters are in the wisdom file"));
        }

        // The file may not end with a newline (for example, if it was edited by hand), in which
        // case the record would be appended to the last line.
        char last;
        stream.seekg(-1, std::ios::end);

        if (!stream.get(last)) {
            throw std::runtime_error(concat("failed to read ", wisdom_file));
        }

        stream.seekp(0, std::ios::end);

        if (last != '\n') {
            stream << "\n";
        }
    } else {
        stream.seekp(0, std::ios::end);
        stream << create_header(problem.tuning_key, params, objective_key).dump() << "\n";
    }

    auto record = create_record(arch, problem, params, objective_key, config, objective);
    stream << record.dump() << "\n";

    if (!stream.flush()) {
        throw std::runtime_error(concat("failed to write ", wisdom_file));
    }
}
}  // namespace kernel_launcher
