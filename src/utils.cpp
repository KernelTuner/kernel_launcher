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
    DummyStream() noexcept = default;
};

static std::ostream& log_level(LogLevel level) {
    static constexpr const char* ENV_KEY = "KERNEL_LAUNCHER_LOG";
    static DummyStream dummy_stream;
    static LogLevel min_level = LogLevel::Unknown;

    if (min_level == LogLevel::Unknown) {
        const char* env = getenv(ENV_KEY);

        if (env == nullptr || strcmp(env, "INFO") == 0) {
            min_level = LogLevel::Info;
        } else if (strcmp(env, "WARN") == 0) {
            min_level = LogLevel::Warning;
        } else {
            min_level = LogLevel::Debug;

            if (strcmp(env, "DEBUG") == 0) {
                log_warning() << "invalid " << ENV_KEY
                              << ": must be DEBUG, WARN, or INFO" << std::endl;
            }
        }
    }

    if (level < min_level) {
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

}  // namespace kernel_launcher