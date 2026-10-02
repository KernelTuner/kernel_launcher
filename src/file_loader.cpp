#include "kernel_launcher/file_loader.hpp"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>

#include "kernel_launcher/utils.hpp"

namespace kernel_launcher {

std::string path_join(std::string_view left, std::string_view right) {
    if (left.empty() || (!right.empty() && right.front() == '/')) {
        return std::string(right);
    }

    std::string result(left);

    if (result.back() != '/') {
        result += '/';
    }

    result += right;
    return result;
}

bool write_file(const std::string& path, std::string_view content, bool overwrite) {
    // "x" makes creation fail if the file already exists.
    FILE* file = std::fopen(path.c_str(), overwrite ? "wb" : "wbx");

    if (file == nullptr) {
        if (!overwrite && errno == EEXIST) {
            return false;
        }

        throw std::runtime_error("failed to open " + path + ": " + std::strerror(errno));
    }

    size_t written = std::fwrite(content.data(), 1, content.size(), file);
    int write_errno = errno;
    bool closed = std::fclose(file) == 0;

    if (written != content.size() || !closed) {
        if (written == content.size()) {
            write_errno = errno;
        }

        // Do not leave a partially written file behind.
        std::remove(path.c_str());
        throw std::runtime_error("failed to write " + path + ": " + std::strerror(write_errno));
    }

    return true;
}

// Returns `std::nullopt` if the file does not exist or is a directory. Throws
// on other errors.
static std::optional<std::string> read_file(const std::string& file_name) {
    // Closes the file on all exit paths, including exceptions.
    std::unique_ptr<FILE, int (*)(FILE*)> file(std::fopen(file_name.c_str(), "rb"), std::fclose);

    if (file == nullptr) {
        if (errno == ENOENT || errno == ENOTDIR) {
            return std::nullopt;
        }

        throw std::runtime_error("failed to open " + file_name + ": " + std::strerror(errno));
    }

    std::string content;
    char buffer[4096];
    size_t n;

    while ((n = std::fread(buffer, 1, sizeof(buffer), file.get())) > 0) {
        content.append(buffer, n);
    }

    if (std::ferror(file.get()) != 0) {
        // Opening a directory succeeds, but reading from it fails.
        if (errno == EISDIR) {
            return std::nullopt;
        }

        throw std::runtime_error("failed to read " + file_name + ": " + std::strerror(errno));
    }

    return content;
}

// `detail` is appended to the message, it describes where was searched.
[[noreturn]] static void throw_not_found(std::string_view path, const std::string& detail) {
    std::string message = concat("could not find file '", path, "'", detail);
    log_warning() << message << std::endl;
    throw FileNotFoundException(message);
}

static std::vector<std::string> parse_include_dirs_from_env() {
    std::vector<std::string> result;
    const char* env = std::getenv("KERNEL_LAUNCHER_INCLUDE");

    if (env == nullptr) {
        return result;
    }

    std::string_view remaining = env;

    while (true) {
        size_t end = remaining.find(':');
        std::string_view dir = remaining.substr(0, end);

        if (!dir.empty()) {
            result.emplace_back(dir);
        }

        if (end == std::string_view::npos) {
            break;
        }

        remaining.remove_prefix(end + 1);
    }

    return result;
}

const std::vector<std::string>& include_dirs_from_env() {
    // Initialization of a function-local static is thread-safe.
    static const std::vector<std::string> dirs = parse_include_dirs_from_env();
    return dirs;
}

std::string FileLoader::load(std::string_view path) const {
    if (auto content = try_load(path)) {
        return std::move(*content);
    }

    throw_not_found(path, "");
}

DefaultLoader::DefaultLoader(
    std::vector<std::string> dirs,
    std::shared_ptr<const FileLoader> loader) :
    dirs_(std::move(dirs)),
    loader_(std::move(loader)) {}

std::optional<std::string> DefaultLoader::try_load(std::string_view name) const {
    if (loader_ != nullptr) {
        if (auto content = loader_->try_load(name)) {
            return content;
        }
    }

    for (const auto& dir : dirs_) {
        if (auto content = read_file(path_join(dir, name))) {
            return content;
        }
    }

    return std::nullopt;
}

std::string DefaultLoader::load(std::string_view name) const {
    if (auto content = try_load(name)) {
        return std::move(*content);
    }

    std::string detail;

    if (dirs_.empty()) {
        detail = " (no search directories)";
    } else {
        detail = " in directories: " + dirs_[0];

        for (size_t i = 1; i < dirs_.size(); i++) {
            detail += ", " + dirs_[i];
        }
    }

    throw_not_found(name, detail);
}

}  // namespace kernel_launcher
