#include <atomic>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

#if defined(__clang__) || defined(__GNUC__)
    #include <cxxabi.h>
#endif

#include "kernel_launcher/utils.hpp"

namespace kernel_launcher {

const std::string& intern_string(std::string_view data) {
    static std::mutex mutex;
    // Intentionally leaked, so interned strings stay valid during static destruction.
    static auto& pool = *new std::unordered_map<std::string_view, std::unique_ptr<std::string>>();

    std::lock_guard<std::mutex> guard(mutex);
    auto it = pool.find(data);
    if (it == pool.end()) {
        auto value = std::make_unique<std::string>(data);
        std::string_view key = *value;
        it = pool.emplace(key, std::move(value)).first;
    }

    return *it->second;
}

std::string demangle_type_info(const std::type_info& type) {
#if defined(__clang__) || defined(__GNUC__)
    const char* mangled_name = type.name();
    int status = ~0;
    // TODO: look into how portable this solution is on different platforms :-/
    std::unique_ptr<char, void (*)(void*)> undecorated_name(
        abi::__cxa_demangle(mangled_name, nullptr, nullptr, &status),
        std::free);

    // status == 0: OK
    // status == -1: memory allocation failure
    // status == -2: name is invalid
    // status == -3: one of the other arguments is invalid
    if (status != 0) {
        throw std::runtime_error(std::string("__cxa_demangle failed for ") + mangled_name);
    }

    return std::string(undecorated_name.get());
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

// Stream without a buffer, so everything written to it is discarded.
struct DummyStream: std::ostream {
    DummyStream() : std::ostream(nullptr) {}
};

static bool log_enabled(LogLevel level) {
    static constexpr const char* ENV_KEY = "KERNEL_LAUNCHER_LOG";

    // Threads racing on the first call may both parse the environment, but
    // they all store the same value.
    static std::atomic<LogLevel> min_level {LogLevel::Unknown};
    LogLevel current = min_level.load(std::memory_order_relaxed);

    if (current == LogLevel::Unknown) {
        const char* env_raw = getenv(ENV_KEY);
        std::string env = env_raw != nullptr ? env_raw : "";
        bool invalid = false;

        if (env.empty() || env == "INFO" || env == "info") {
            current = LogLevel::Info;
        } else if (env == "WARN" || env == "warn" || env == "WARNING" || env == "warning") {
            current = LogLevel::Warning;
        } else if (env == "DEBUG" || env == "debug") {
            current = LogLevel::Debug;
        } else {
            current = LogLevel::Debug;
            invalid = true;
        }

        // Store before logging below, since log_warning() calls this function.
        min_level.store(current, std::memory_order_relaxed);

        if (invalid) {
            log_warning() << "invalid " << ENV_KEY << ": must be DEBUG, WARN, or INFO" << std::endl;
        }
    }

    return level >= current;
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
    // Writing to a stream without a buffer still updates its error state, so
    // each thread needs its own instance to avoid a data race.
    thread_local DummyStream dummy_stream;

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

}  // namespace kernel_launcher
