#ifndef KERNEL_LAUNCHER_PRAGMA_H
#define KERNEL_LAUNCHER_PRAGMA_H

#include "kernel_launcher/registry.h"

namespace kernel_launcher {

/**
 * Parses the given `KernelSource`, searches the source code for the kernel
 * with the given `kernel_name`, extract the KernelLauncher-specific pragmas
 * for that kernel, and returns `KernelBuilder`.
 *
 * @param kernel_name The name of the kernel in the source code. It may contain
 * namespaces such as `mypackage::kernels::vector_add`.
 * @param source The source code. Can be either a filename (like `"kernel.cu"`)
 * or a filename+content pair (like `{"kernel.cu", "#include <stdin.h>..."}`).
 * @param template_args Optional; template arguments passed to the kernel.
 */
KernelBuilder build_pragma_kernel(
    const std::string& kernel_name,
    const KernelSource& source,
    const std::vector<Value>& template_args = {},
    const FileLoader& fs = DefaultLoader {});

/**
 * This is a `IKernelDescriptor` that uses `build_pragma_kernel` to construct
 * a `KernelBuilder`.
 */
struct PragmaKernel: IKernelDescriptor {
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
        std::vector<Value> template_args = {});

    KernelBuilder build() const override;
    bool equals(const IKernelDescriptor& that) const override;
    hash_t hash() const override;

  private:
    std::string kernel_name_;
    std::string file_path_;
    std::vector<Value> template_args_;
};

}  // namespace kernel_launcher

#endif  //KERNEL_LAUNCHER_PRAGMA_H
