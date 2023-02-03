#include "kernel_launcher/pragma.h"

#include "kernel_launcher/internal/directives.h"

namespace kernel_launcher {

KernelBuilder build_pragma_kernel(
    const KernelSource& source,
    const std::string& name,
    const std::vector<Value>& template_args,
    const FileLoader& fs) {
    std::string filename = source.file_name();
    std::string content = source.read(fs);

    internal::TokenStream stream(filename, content);
    std::vector<internal::KernelDef> kernels = internal::parse_kernels(stream);

    for (const auto& kernel : kernels) {
        if (stream.matches(kernel.name, name)) {
            return internal::process_kernel(stream, kernel, template_args);
        }
    }

    throw std::runtime_error(
        "kernel '" + name + "' was not found in file \'" + filename + "\'");
}

PragmaKernel::PragmaKernel(
    std::string path,
    std::string name,
    std::vector<Value> template_args) :
    kernel_name_(std::move(name)),
    template_args_(std::move(template_args)) {
    // Resolve absolute file path
    const char* abs_path = realpath(path.c_str(), nullptr);
    if (abs_path == nullptr) {
        throw std::runtime_error("failed to resolve path: '" + path + "'");
    }

    file_path_ = abs_path;
}

KernelBuilder PragmaKernel::build() const {
    return build_pragma_kernel(
        KernelSource(file_path_),
        kernel_name_,
        template_args_);
}

bool PragmaKernel::equals(const IKernelDescriptor& that) const {
    if (const auto* m = dynamic_cast<const PragmaKernel*>(&that)) {
        return m->kernel_name_ == this->kernel_name_
            && m->file_path_ == this->file_path_
            && m->template_args_ == this->template_args_;
    }

    return false;
}

hash_t PragmaKernel::hash() const {
    hash_t h = hash_fields(kernel_name_, file_path_);

    for (const auto& v : template_args_) {
        h = hash_combine(h, hash_fields(v));
    }

    return h;
}

}  // namespace kernel_launcher
