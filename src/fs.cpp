#include <limits.h>
#include <unistd.h>

#include <fstream>
#include <sstream>

#ifndef KERNEL_LAUNCHER_EMBEDDED_DATA
    #define KERNEL_LAUNCHER_EMBEDDED_DATA (0)
#endif
#if KERNEL_LAUNCHER_EMBEDDED_DATA
    #include <dlfcn.h>
#endif

#include "kernel_launcher/fs.h"
#include "kernel_launcher/utils.h"

namespace kernel_launcher {

std::string path_join(const std::string& left, const std::string& right) {
    if (left.empty() || left == "." || right.front() == '/') {
        return right;
    }

    std::string result = left;
    if (result.back() != '/') {
        result += '/';
    }

    result += right;
    return result;
}

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

static void add_env_directories(std::vector<std::string>& result) {
    static constexpr const char* ENV_KEY = "KERNEL_LAUNCHER_INCLUDE";
    const char* paths = getenv(ENV_KEY);

    if (paths) {
        while (true) {
            if (paths[0] == '\0') {
                break;
            }

            if (paths[0] == ';') {
                paths++;
                continue;
            }

            size_t count = 0;

            while (true) {
                char c = paths[count];
                if (c == '\0' || c == ';') {
                    break;
                }
                count++;
            }

            if (count > 0) {
                result.emplace_back(paths, count);
            }

            paths += count;
        }
    }
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
#endif

DefaultLoader::DefaultLoader(
    const std::vector<std::string>& dirs,
    bool include_cwd) {
    // Add environment directories first
    add_env_directories(search_dirs_);

    // working directory
    if (include_cwd) {
        char cwd[PATH_MAX + 1];
        if (getcwd(cwd, sizeof cwd)) {
            search_dirs_.push_back(cwd);
        }
    }

    // directories given by user
    for (const std::string& d : dirs) {
        search_dirs_.push_back(d);
    }
}

std::vector<char> DefaultLoader::load(
    const std::string& file_name,
    const std::vector<std::string>& more_dirs) const {
    if (!file_name.empty()) {
        std::vector<char> result;

        for (const std::vector<std::string>& dirs : {search_dirs_, more_dirs}) {
            for (const std::string& dir : dirs) {
                if (dir.empty()) {
                    continue;
                }

                std::string full_path = path_join(dir, file_name);

                if (read_file(full_path, result)) {
                    return result;
                }
            }
        }

#if KERNEL_LAUNCHER_EMBEDDED_DATA
        if (EmbeddedData().find(file_name, result)) {
            return result;
        }
#endif
    }

    log_info() << "could not find file " << file_name
               << ", searched following directories:" << std::endl;
    for (const std::vector<std::string>& dirs : {search_dirs_, more_dirs}) {
        for (const std::string& dir : dirs) {
            log_info() << " - " << dir << std::endl;
        }
    }

    throw std::runtime_error("could not find file: " + file_name);
}

}  // namespace kernel_launcher