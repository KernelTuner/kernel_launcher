#include <cstring>
#include <iostream>
#include <stdexcept>

#if __clang__ || __GNUC__
    #include <cxxabi.h>
#endif

#include "kernel_launcher/utils.h"

namespace kernel_launcher {

std::string demangle_type_info(const std::type_info& type) {
#if __clang__ || __GNUC__
    const char* mangled_name = type.name();
    int status = ~0;
    // TOOD: look into how portable this solution is on different platforms :-/
    char* undecorated_name =
        abi::__cxa_demangle(mangled_name, nullptr, nullptr, &status);

    // status == 0: OK
    // status == -1: memory allocation failure
    // status == -2: name is invalid
    // status == -3: one of the other arguments is invalid
    if (status != 0) {
        throw std::runtime_error(
            std::string("__cxa_demangle failed for ") + mangled_name);
    }

    std::string result = undecorated_name;
    free(undecorated_name);
    return result;
#else
    throw std::runtime_error("this platform does not support `__cxa_demangle`");
#endif
}

enum struct LogLevel {
    Debug,
    Info,
    Warning,
    Unknown,
};

struct DummyStream: std::ostream {
    DummyStream() = default;
};

bool log_enabled(LogLevel level) {
    static constexpr const char* ENV_KEY = "KERNEL_LAUNCHER_LOG";
    static LogLevel min_level = LogLevel::Unknown;

    if (min_level == LogLevel::Unknown) {
        const char* env_raw = getenv(ENV_KEY);
        std::string env = env_raw != nullptr ? env_raw : "";

        if (env.empty() || env == "INFO" || env == "info") {
            min_level = LogLevel::Info;
        } else if (
            env == "WARN" || env == "warn" || env == "WARNING"
            || env == "warning") {
            min_level = LogLevel::Warning;
        } else if (env == "DEBUG" || env == "debug") {
            min_level = LogLevel::Debug;
        } else {
            min_level = LogLevel::Debug;
            log_warning() << "invalid " << ENV_KEY
                          << ": must be DEBUG, WARN, or INFO" << std::endl;
        }
    }

    return level >= min_level;
}

bool log_debug_enabled() {
    return log_enabled(LogLevel::Debug);
}

bool log_info_enabled() {
    return log_enabled(LogLevel::Info);
}

bool log_warning_enabled() {
    return log_enabled(LogLevel::Warning);
}

static std::ostream& log_level(LogLevel level) {
    static DummyStream dummy_stream;

    if (!log_enabled(level)) {
        return dummy_stream;
    }

    const char* name = [=]() {
        switch (level) {
            case LogLevel::Debug:
                return "DEBUG";
            case LogLevel::Info:
                return "INFO";
            case LogLevel::Warning:
                return "WARN";
            default:
                return "???";
        }
    }();

    return std::cerr << "KERNEL_LAUNCHER [" << name << "] ";
}

std::ostream& log_debug() {
    return log_level(LogLevel::Debug);
}

std::ostream& log_info() {
    return log_level(LogLevel::Info);
}

std::ostream& log_warning() {
    return log_level(LogLevel::Warning);
}

bool safe_double_to_int64(double input, int64_t& output) {
    static constexpr double min_val =
        static_cast<double>(std::numeric_limits<int64_t>::min());
    static constexpr double max_val = static_cast<double>(
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) + 1);
    int64_t v = static_cast<int64_t>(input);

    if (input >= min_val && input < max_val && input == double(v)) {
        output = v;
        return true;
    }

    output = 0;
    return false;
}

static constexpr int64_t I64_MIN = std::numeric_limits<int64_t>::min();
//static constexpr int64_t I64_MAX = std::numeric_limits<int64_t>::max();

bool safe_int64_add(int64_t lhs, int64_t rhs, int64_t& output) {
    // TODO: Check how portable these `__builtin_*_overflow` functions are
    //    output = lhs + rhs;
    //    return (rhs >= 0) ? (lhs <= I64_MAX - rhs) : (lhs >= I64_MIN - rhs);
    return !__builtin_saddl_overflow(lhs, rhs, &output);
}

bool safe_int64_sub(int64_t lhs, int64_t rhs, int64_t& output) {
    //    output = lhs - rhs;
    //    return (rhs >= 0) ? (lhs >= I64_MIN + rhs) : (lhs <= I64_MAX + rhs);
    return !__builtin_ssubl_overflow(lhs, rhs, &output);
}

bool safe_int64_mul(int64_t lhs, int64_t rhs, int64_t& output) {
    //    bool in_bounds;
    //
    //    if (rhs == 0) {
    //        in_bounds = true;
    //    } else if ((lhs > 0) == (rhs > 0)) {
    //        in_bounds = lhs <= I64_MAX / rhs;
    //    } else {
    //        in_bounds = lhs >= I64_MIN / rhs;
    //    }
    //
    //    output = lhs * rhs;
    //    return in_bounds;
    return !__builtin_smull_overflow(lhs, rhs, &output);
}

bool safe_int64_div(int64_t lhs, int64_t rhs, int64_t& output) {
    if (rhs != 0 && (rhs != -1 || lhs != I64_MIN)) {
        output = lhs / rhs;
        return true;
    }

    return false;
}

bool string_match(const char* pattern, const char* input) {
    // advance pattern and input until we find the first '*'
    while (*pattern != '*') {
        // Reach end of string. strings match!
        if (*input == '\0' && *pattern == '\0') {
            return true;
        }

        // character mismatch, no match.
        if (*input != *pattern) {
            return false;
        }

        input++;
        pattern++;
    }

    // Get the next non-* character in pattern.
    while (*pattern == '*') {
        pattern++;
    }

    char next = *pattern;

    // trailing *. This always matches the input.
    if (next == '\0') {
        return true;
    }

    while (*input != '\0') {
        if (next == *input) {
            if (string_match(pattern + 1, input + 1)) {
                return true;
            }
        }

        input++;
    }

    return false;
}

std::vector<std::string>
string_split(const char* input, const std::vector<char>& delims) {
    size_t start = 0;
    std::vector<std::string> result;

    while (input[start] != '\0') {
        size_t end = start;

        while (input[end] != '\0') {
            bool is_delim = false;

            for (char delim : delims) {
                if (input[end] == delim) {
                    is_delim = true;
                }
            }

            if (is_delim) {
                break;
            }

            end++;
        }

        if (input[end] == '\0') {
            break;
        }

        result.emplace_back(input + start, end - start);
        start = end + 1;
    }

    result.emplace_back(input + start);
    return result;
}

std::vector<std::string> string_split(const char* input, char delim) {
    return string_split(input, std::vector<char> {delim});
}

hash_t hash_string(const char* buffer, size_t num_bytes) {
    // Simple FNV1a hash
    static constexpr hash_t prime = 0x100000001b3;
    static constexpr hash_t hash_init = 0xcbf29ce484222325;

    hash_t hash = hash_init;

    for (size_t i = 0; i < num_bytes; i++) {
        hash = (hash ^ (hash_t)(unsigned char)buffer[i]) * prime;
    }

    return hash;
}

}  // namespace kernel_launcher
