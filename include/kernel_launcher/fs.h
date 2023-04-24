#ifndef KERNEL_LAUNCHER_FS_H
#define KERNEL_LAUNCHER_FS_H

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace kernel_launcher {

std::string path_join(const std::string& left, const std::string& right);

bool read_file(const std::string& path, std::vector<char>& result);
bool read_file(const std::string& path, std::string& result);

bool write_file(
    const std::string& path,
    const char* content,
    size_t nbytes,
    bool overwrite = false);

inline bool write_file(
    const std::string& path,
    const std::vector<char>& content,
    bool overwrite = false) {
    return write_file(path, content.data(), content.size(), overwrite);
}

inline bool write_file(
    const std::string& path,
    const std::vector<int8_t>& content,
    bool overwrite = false) {
    return write_file(
        path,
        (const char*)content.data(),
        content.size(),
        overwrite);
}

inline bool write_file(
    const std::string& path,
    const std::vector<uint8_t>& content,
    bool overwrite = false) {
    return write_file(
        path,
        (const char*)content.data(),
        content.size(),
        overwrite);
}

inline bool write_file(
    const std::string& path,
    const std::string& content,
    bool overwrite = false) {
    return write_file(path, content.data(), content.size(), overwrite);
}

/**
 * Interface for object's that can load a file's contents given the file's name.
 * The `DefaultLoader` simply loads the file from the local file system.
 */
struct FileLoader {
    virtual ~FileLoader() = default;

    /**
     * Return the content of a file given its name. Implementations of this
     * interface are free to resolve the file path in any way possible.
     */
    virtual std::string load(const std::string& file_name) const = 0;
};

/**
 * A `FileLoader` that simply loads a file from the local file system.
 */
struct DefaultLoader: FileLoader {
    explicit DefaultLoader(
        const std::vector<std::string>& dirs,
        bool include_cwd = true);
    DefaultLoader() : DefaultLoader(std::vector<std::string> {}) {}
    std::string load(const std::string& file_name) const override;

  private:
    std::vector<std::string> search_dirs_;
};

struct ForwardLoader: FileLoader {
    explicit ForwardLoader(
        std::vector<std::string> dirs,
        std::shared_ptr<FileLoader> parent);
    std::string load(const std::string& file_name) const override;

  private:
    std::vector<std::string> search_dirs_;
    std::shared_ptr<FileLoader> parent_;
};

}  // namespace kernel_launcher

#endif  //KERNEL_LAUNCHER_FS_H
