#ifndef KERNEL_LAUNCHER_FILE_LOADER_H
#define KERNEL_LAUNCHER_FILE_LOADER_H

#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace kernel_launcher {

/**
 * @addtogroup utils
 * @{
 */

/**
 * Join two paths. If `right` is absolute or `left` is empty, `right` is
 * returned unchanged.
 */
std::string path_join(std::string_view left, std::string_view right);

/**
 * Write `content` to the file at `path`. If the file already exists and
 * `overwrite` is false, nothing is written and `false` is returned. Throws if
 * the file cannot be written for any other reason, in which case no partially
 * written file is left behind.
 */
bool write_file(const std::string& path, std::string_view content, bool overwrite = false);

/** @} */

/**
 * @addtogroup compiler
 * @{
 */

/**
 * Directories listed in the environment variable `KERNEL_LAUNCHER_INCLUDE`,
 * separated by `:` (like `PATH`). Empty entries are ignored. The variable is
 * read once, on the first call.
 */
const std::vector<std::string>& include_dirs_from_env();

class FileNotFoundException: public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

class FileLoader {
  public:
    virtual ~FileLoader() = default;

    /**
     * Return the content of the file at `path`, or `std::nullopt` if it does
     * not exist. Implementations are free to resolve the path in any way
     * possible. Throws only on actual errors (e.g., permissions, I/O).
     */
    virtual std::optional<std::string> try_load(std::string_view path) const = 0;

    /**
     * Like `try_load`, but logs a warning and throws `FileNotFoundException`
     * if the file does not exist.
     */
    virtual std::string load(std::string_view path) const;
};

/**
 * A `FileLoader` that first asks `loader` (if not null), and then reads
 * `dir/name` from disk for each directory in `dirs`, in order. Directories
 * named `dir/name` are skipped. If the file is not found, both the warning
 * logged by `load` and the message of the exception it throws list the
 * directories that were searched.
 */
class DefaultLoader: public FileLoader {
  public:
    explicit DefaultLoader(
        std::vector<std::string> dirs = {"."},
        std::shared_ptr<const FileLoader> loader = nullptr);

    std::optional<std::string> try_load(std::string_view name) const override;
    std::string load(std::string_view name) const override;

  private:
    std::vector<std::string> dirs_;
    std::shared_ptr<const FileLoader> loader_;
};

/** @} */

}  // namespace kernel_launcher

#endif
