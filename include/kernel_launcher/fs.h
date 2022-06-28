#ifndef KERNEL_LAUNCHER_FS_H
#define KERNEL_LAUNCHER_FS_H

#include <string>
#include <unordered_map>
#include <vector>

namespace kernel_launcher {

bool read_file(const std::string& path, std::vector<char>& result);
bool write_file(
    const std::string& path,
    const std::vector<char>& content,
    bool overwrite = false);

struct FileResolver {
    explicit FileResolver(std::vector<std::string> dirs);
    FileResolver() : FileResolver(std::vector<std::string> {}) {}
    std::vector<char> read(const std::string& path) const;

    const std::vector<std::string>& directories() const {
        return search_dirs_;
    }

  private:
    std::vector<std::string> search_dirs_;
};

}  // namespace kernel_launcher

#endif  //KERNEL_LAUNCHER_FS_H
