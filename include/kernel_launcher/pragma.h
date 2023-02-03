#ifndef KERNEL_LAUNCHER_PRAGMA_H
#define KERNEL_LAUNCHER_PRAGMA_H

#include "kernel_launcher/registry.h"

namespace kernel_launcher {

KernelBuilder build_pragma_kernel(
    const KernelSource& source,
    const std::string& name,
    const std::vector<Value>& template_args = {},
    const FileLoader& fs = DefaultLoader {});

struct PragmaKernel: IKernelDescriptor {
    PragmaKernel(
        std::string path,
        std::string name,
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
