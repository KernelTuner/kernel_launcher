#ifndef KERNEL_LAUNCHER_FS_H
#define KERNEL_LAUNCHER_FS_H

#include <string>
#include <unordered_map>
#include <vector>

namespace kernel_launcher {

std::string path_join(const std::string& left, const std::string& right);

bool read_file(const std::string& path, std::vector<char>& result);
bool write_file(
    const std::string& path,
    const std::vector<char>& content,
    bool overwrite = false);

struct FileLoader {
    virtual ~FileLoader() = default;
    virtual std::vector<char> load(
        const std::string& file_name,
        const std::vector<std::string>& dirs = {}) const = 0;
};

struct DefaultLoader: FileLoader {
    explicit DefaultLoader(
        const std::vector<std::string>& dirs,
        bool include_cwd = true);
    DefaultLoader() : DefaultLoader(std::vector<std::string> {}) {}
    std::vector<char> load(
        const std::string& file_name,
        const std::vector<std::string>& include_dirs) const override;

  private:
    std::vector<std::string> search_dirs_;
};

}  // namespace kernel_launcher

#endif  //KERNEL_LAUNCHER_FS_H
