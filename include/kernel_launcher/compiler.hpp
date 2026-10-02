#ifndef KERNEL_LAUNCHER_COMPILER_H
#define KERNEL_LAUNCHER_COMPILER_H

#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "kernel_launcher/file_loader.hpp"
#include "kernel_launcher/gpu_utils.hpp"
#include "kernel_launcher/type_info.hpp"

namespace kernel_launcher {

/**
 * @addtogroup compiler
 * @{
 */

/**
 * Thrown when runtime compilation of a kernel fails, either because the source does not
 * compile or because the runtime compiler (NVRTC or HIPRTC) returns an error.
 */
class CompilerException: public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

/**
 * A source file, given either as just a file name or as a file name together with its content.
 * In the first case, the content is loaded by a `FileLoader` when the file is compiled.
 */
class FileSource {
  public:
    /**
     * Construct a source that is loaded from the file `path` when it is compiled.
     */
    FileSource(std::string path);

    /**
     * Construct a source with the given `content`. No file is loaded, `path` is only used as
     * the file name in compiler messages and to resolve relative includes.
     */
    FileSource(std::string path, std::string content);

    /**
     * Construct a source that is loaded from the file `path` when it is compiled.
     */
    FileSource(const char* path) : FileSource(std::string(path)) {}

    size_t hash() const noexcept;
    bool operator==(const FileSource& that) const;

    bool operator!=(const FileSource& that) const {
        return !(*this == that);
    }

    /**
     * Returns the file name of this source.
     */
    const std::string& file_name() const noexcept {
        return path_;
    }

    /**
     * Returns the content if it was given at construction. The returned view refers to this
     * object, it is only valid as long as this `FileSource` is alive.
     */
    std::optional<std::string_view> content() const {
        return content_;
    }

    /**
     * Returns the content if it was given, otherwise loads the file using
     * `loader.load(file_name())`, which throws if it cannot be found.
     */
    std::string read(const FileLoader& loader) const;

  private:
    std::string path_;
    std::optional<std::string> content_;
};

/**
 * The definition of a single kernel: everything a compiler needs to compile one kernel
 * function into a `GPUKernel`.
 */
struct KernelDef {
    /// Name of the kernel function, without template arguments.
    std::string kernel_name;

    /// The source file that contains the kernel function.
    FileSource source;

    /// Headers that are included before `source`, in order.
    std::vector<FileSource> headers;

    /// The parameter types of the kernel function. Used to select the correct overload.
    std::vector<TypeInfo> param_types;

    /// The template arguments of the kernel function. Empty if it is not a template.
    std::vector<TemplateArg> template_args;

    /// Additional flags that are passed to the compiler. These are passed as is, so they must
    /// be NVRTC flags for CUDA and clang flags for HIP. Use `-I<dir>` for include directories,
    /// since `--include-path=<dir>` is only supported by NVRTC.
    std::vector<std::string> compiler_flags;

    KernelDef(std::string kernel_name, FileSource source) :
        kernel_name(std::move(kernel_name)),
        source(std::move(source)) {}

    size_t calculate_hash() const noexcept;
    bool operator==(const KernelDef& that) const;

    bool operator!=(const KernelDef& that) const {
        return !(*this == that);
    }
};

/**
 * Interface for classes that compile a `KernelDef` into a `GPUKernel`.
 */
class CompilerBase {
  public:
    virtual ~CompilerBase() = default;

    /**
     * Compile the kernel `def` for the device of `ctx` and load it into `ctx`. Throws
     * `CompilerException` if compilation fails.
     */
    virtual std::unique_ptr<GPUKernel> compile(GPUContextRef ctx, KernelDef def) = 0;
};

/**
 * Compiles kernels at runtime, using NVRTC for CUDA or HIPRTC for HIP.
 */
class RuntimeCompiler: public CompilerBase {
  public:
    /**
     * Construct a compiler. Source files, headers given by `KernelDef::headers`, and files
     * that are included by the kernel source are looked up in the following order:
     *
     * 1. The given `loader`, if it is not null.
     * 2. The directories in the environment variable `KERNEL_LAUNCHER_INCLUDE`.
     * 3. The include directories in `default_options` and `KernelDef::compiler_flags` (`-I`
     *    and `--include-path=`).
     * 4. The given `search_dirs`.
     *
     * The `default_options` are passed to the compiler for every kernel, before the flags in
     * `KernelDef::compiler_flags`.
     */
    explicit RuntimeCompiler(
        std::shared_ptr<const FileLoader> loader = nullptr,
        std::vector<std::string> search_dirs = {"."},
        std::vector<std::string> default_options = {}) :
        loader_(std::move(loader)),
        search_dirs_(std::move(search_dirs)),
        default_options_(std::move(default_options)) {}

    /**
     * Returns the version of the runtime compiler as `major * 1000 + minor * 10`. For NVRTC
     * this is the same encoding as `CUDA_VERSION` (e.g. 12020 for 12.2). For HIPRTC this is
     * the HIPRTC version, which cannot be compared to a CUDA version.
     */
    static int version();

    std::unique_ptr<GPUKernel> compile(GPUContextRef ctx, KernelDef def) override;

    /**
     * Compiles `def` for the architecture `arch` into a binary image (a cubin for CUDA, a code
     * object for HIP) that can be loaded with `GPUKernel::load_image`. The image is written to
     * `image_out` and the mangled name of the kernel function to `symbol_out`. 
     */
    void compile_image(GPUArch arch, KernelDef def, std::string& image_out, std::string& symbol_out)
        const;

  private:
    std::shared_ptr<const FileLoader> loader_;
    std::vector<std::string> search_dirs_;
    std::vector<std::string> default_options_;
};

// Deprecated: This is the old name, here for backwards compatibility
using NvrtcCompiler = RuntimeCompiler;

/** @} */

}  // namespace kernel_launcher

namespace std {
template<>
struct hash<kernel_launcher::FileSource> {
    size_t operator()(const kernel_launcher::FileSource& v) const noexcept {
        return v.hash();
    }
};

template<>
struct hash<kernel_launcher::KernelDef> {
    size_t operator()(const kernel_launcher::KernelDef& v) const noexcept {
        return v.calculate_hash();
    }
};
}  // namespace std

#endif
