#include <limits.h>
#include <unistd.h>

#include <fstream>
#include <sstream>

#define KERNEL_LAUNCHER_EMBEDDED_DATA (0)
#if KERNEL_LAUNCHER_EMBEDDED_DATA
    #include <dlfcn.h>
#endif

#include "kernel_launcher/fs.h"

namespace kernel_launcher {
bool read_file(const std::string& path, std::vector<char>& result) {
    std::ifstream stream(path);

    if (stream) {
        stream.seekg(0, std::ios::end);
        std::streampos len = stream.tellg();
        if (len == -1) {
            return false;
        }
        stream.seekg(0, std::ios::beg);

        result.resize(static_cast<size_t>(len));
        stream.read(&result[0], len);

        // Check if the stream is still valid after reading
        if (stream) {
            return true;
        }
    }

    return false;
}

bool write_file(
    const std::string& path,
    const std::vector<char>& content,
    bool overwrite) {
    std::ofstream stream(path, std::ios::ate);

    if (stream) {
        if (stream.tellp() < 0 || (stream.tellp() > 0 && !overwrite)) {
            return false;
        }

        stream.write(content.data(), std::streamsize(content.size()));

        // Check if the stream is still valid after writing
        if (stream) {
            return true;
        }
    }

    return false;
}

// This code based on jitify.hpp from NVIDIA/jitify
#if KERNEL_LAUNCHER_EMBEDDED_DATA
struct EmbeddedData {
    EmbeddedData(EmbeddedData const&) = delete;

    EmbeddedData() {
        app_ = dlopen(NULL, RTLD_LAZY);
        dlerror();  // Clear any existing error
    }

    ~EmbeddedData() {
        if (app_) {
            dlclose(app_);
        }
    }

    const char* resolve_symbol(std::string key) const {
        static constexpr const char* illegal_symbols = "/\\.-: ?%*|\"<>";

        size_t i = key.find_first_of(illegal_symbols);
        while (i != std::string::npos) {
            key[i] = '_';
            i = key.find_first_of(illegal_symbols, i + 1);
        }

        if (app_) {
            const char* data = (const char*)dlsym(app_, key.c_str());

            if (data) {
                return data;
            }
        }

        return nullptr;
    }

    bool find(const std::string& key, std::vector<char>& result) {
        const char* begin = resolve_symbol("_binary_" + key + "_start");
        const char* end = resolve_symbol("_binary_" + key + "_end");

        if (!begin || !end || begin > end) {
            return false;
        }

        size_t length = end - begin;
        result.resize(length);
        std::copy(begin, end, result.data());
        return true;
    }

  private:
    void* app_;
};
#else
struct EmbeddedData {
    bool find(const std::string& key, std::vector<char>& result) {
        return false;
    }
};
#endif

static void add_env_directories(std::vector<std::string>& result) {
    static constexpr const char* ENV_KEY = "KERNEL_LAUNCHER_INCLUDE_PATH";
    const char* paths = getenv(ENV_KEY);

    if (paths) {
        size_t offset = 0;
        while (paths[offset] != '\0') {
            size_t count = 0;

            while (true) {
                char c = paths[offset + count];
                if (c == '\0' || c == ';') {
                    break;
                }
                count++;
            }

            if (count > 0) {
                result.push_back(std::string(paths + offset, count));
            }

            offset += count + 1;
        }
    }
}

FileResolver::FileResolver(std::vector<std::string> dirs) {
    // working directory
    char cwd[PATH_MAX + 1];
    if (getcwd(cwd, sizeof cwd)) {
        search_dirs_.push_back(cwd);
    }

    // directories given by user
    for (const std::string& d : dirs) {
        if (!d.empty()) {
            dirs.push_back(d);
        }
    }

    // directories on file path
    add_env_directories(search_dirs_);
}

static std::string
join_paths(const std::string& left, const std::string& right) {
    if (right.empty() || right[0] == '/') {
        return left;
    }

    if (left.empty() || left == ".") {
        return right;
    }

    std::string result = left;
    if (result.back() != '/') {
        result += '/';
    }

    result += right;
    return result;
}

std::vector<char> FileResolver::read(const std::string& path) const {
    if (!path.empty()) {
        std::vector<char> result;
        if (read_file(path, result)) {
            return result;
        }

        for (const std::string& dir : search_dirs_) {
            std::string full_path = join_paths(dir, path);

            if (read_file(full_path, result)) {
                return result;
            }
        }

        EmbeddedData data;
        if (data.find(path, result)) {
            return result;
        }
    }

    std::stringstream msg;
    msg << "could not found file \"" << path
        << "\" in the following directories: \".\"";

    for (const std::string& dir : search_dirs_) {
        msg << ", \"" << dir << "\"";
    }

    throw std::runtime_error(msg.str());
}

}  // namespace kernel_launcher