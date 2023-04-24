#include <limits.h>
#include <unistd.h>

#include <cstring>
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

template<typename C>
bool read_file_generic(const std::string& path, C& result) {
    std::ifstream stream(path);

    if (!stream) {
        return false;
    }

    stream.seekg(0, std::ios::end);
    std::streampos len = stream.tellg();
    if (len == -1) {
        return false;
    }

    stream.seekg(0, std::ios::beg);

    result.resize(static_cast<size_t>(len));
    stream.read(&result[0], len);

    // Check if the stream is still valid after reading
    if (!stream) {
        log_warning() << "IO error while reading: " << path << "\n";
        return false;
    }

    return true;
}

bool read_file(const std::string& path, std::vector<char>& result) {
    return read_file_generic(path, result);
}

bool read_file(const std::string& path, std::string& result) {
    if (!read_file_generic(path, result)) {
        return false;
    }

    // Check if string contains interior nul bytes (bad!)
    if (result.find(char(0)) != std::string::npos) {
        log_warning() << "IO error, file contains NUL byte:" << path << "\n";
        return false;
    }

    return true;
}

bool write_file(
    const std::string& path,
    const char* content,
    size_t nbytes,
    bool overwrite) {
    std::fstream stream;

    // Check if the file already exists. Note that there exists a race condition here where it is possible that the file
    // is created between the check and opening for writing. However, there is no portable way to perform this check
    // atomically until `std::ios::noreplace` is stable.
    if (!overwrite) {
        stream.open(path, std::ios::in);
        bool exists = (bool)stream;
        stream.close();

        if (exists) {
            return false;
        }
    }

    // Open file for writing.
    stream.open(path, std::ios::out | std::ios::binary);
    if (!stream) {
        return false;
    }

    // Write data
    stream.write(content, static_cast<std::streamsize>(nbytes));
    if (!stream) {
        return false;
    }

    stream.close();
    return true;
}

static void add_env_directories(std::vector<std::string>& result) {
    static constexpr const char* ENV_KEY = "KERNEL_LAUNCHER_INCLUDE";
    const char* paths = getenv(ENV_KEY);

    // Environment value is not set. Exit now.
    if (paths == nullptr) {
        return;
    }

    for (std::string path : string_split(paths, {':', ',', ';'})) {
        if (!path.empty()) {
            result.push_back(path);
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
        if (getcwd(cwd, sizeof cwd) != nullptr) {
            search_dirs_.emplace_back(cwd);
        }
    }

    // directories given by user
    for (const std::string& d : dirs) {
        search_dirs_.push_back(d);
    }
}

std::string DefaultLoader::load(const std::string& file_name) const {
    if (!file_name.empty()) {
        std::string result;

        for (const std::string& dir : search_dirs_) {
            if (dir.empty()) {
                continue;
            }

            std::string full_path = path_join(dir, file_name);

            if (read_file(full_path, result)) {
                return result;
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
    for (const std::string& dir : search_dirs_) {
        log_info() << " - " << dir << std::endl;
    }

    throw std::runtime_error("could not find file: " + file_name);
}

ForwardLoader::ForwardLoader(
    std::vector<std::string> dirs,
    std::shared_ptr<FileLoader> parent) :
    search_dirs_(std::move(dirs)),
    parent_(std::move(parent)) {}

std::string ForwardLoader::load(const std::string& file_name) const {
    if (!file_name.empty()) {
        std::string result;

        for (const std::string& dir : search_dirs_) {
            std::string full_path = path_join(dir, file_name);

            if (read_file(full_path, result)) {
                return result;
            }
        }
    }

    return parent_->load(file_name);
}

}  // namespace kernel_launcher