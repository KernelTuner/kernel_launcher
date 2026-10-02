#ifndef KERNEL_LAUNCHER_PRAGMA_H
#define KERNEL_LAUNCHER_PRAGMA_H

#include "kernel_launcher/compiler.hpp"
#include "kernel_launcher/file_loader.hpp"
#include "kernel_launcher/primitive.hpp"
#include "kernel_launcher/registry.hpp"

namespace kernel_launcher {

/**
 * @addtogroup builder
 * @{
 */

/**
 * Parses the given `FileSource`, searches the source code for the kernel
 * with the given `kernel_name`, extract the KernelLauncher-specific pragmas
 * for that kernel, and returns `KernelBuilder`.
 *
 * @param kernel_name The name of the kernel in the source code. It may contain
 * namespaces such as `mypackage::kernels::vector_add`.
 * @param source The source code. Can be either a filename (like `"kernel.cu"`)
 * or a filename+content pair (like `{"kernel.cu", "#include <stdin.h>..."}`).
 * @param template_args Optional; template arguments passed to the kernel.
 * Only integral and boolean values are supported.
 */
KernelBuilder build_pragma_kernel(
    std::string_view kernel_name,
    const FileSource& source,
    const std::vector<Primitive>& template_args = {},
    const FileLoader& fs = DefaultLoader {});

/**
 * This is a `KernelDescriptor` that uses `build_pragma_kernel` to construct
 * a `KernelBuilder`.
 */
struct PragmaKernel: KernelDescriptor {
    /**
     * Construct `PragmaKernel`.
     *
     * @param path Filename of the source file.
     * @param kernel_name The name of the kernel in the source code. It may
     * contain namespaces such as `mypackage::kernels::vector_add`.
     * @param template_args Optional; template arguments passed to the kernel.
     */
    PragmaKernel(
        std::string kernel_name,
        std::string path,
        std::vector<Primitive> template_args = {});

    KernelBuilder build() const override;
    bool equals(const KernelDescriptor& that) const override;
    size_t hash() const override;

  private:
    std::string kernel_name_;
    std::string file_path_;
    std::vector<Primitive> template_args_;
};

/**
 * Describes the kernel `kernel_name` in the file `path`, built using
 * `build_pragma_kernel`. Pass the result to `launch` or `Registry::lookup`.
 * Usage: `launch(pragma_kernel("vector_add", "kernel.cu", {256}), ...)`
 */
inline PragmaKernel pragma_kernel(
    std::string kernel_name,
    std::string path,
    std::vector<Primitive> template_args = {}) {
    return {std::move(kernel_name), std::move(path), std::move(template_args)};
}

/** @} */

}  // namespace kernel_launcher

#endif  //KERNEL_LAUNCHER_PRAGMA_H
