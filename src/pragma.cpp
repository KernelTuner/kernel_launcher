#include "kernel_launcher/pragma.h"

#include "kernel_launcher/internal/directives.h"

namespace kernel_launcher {

KernelBuilder build_pragma_kernel(
    const std::string& kernel_name,
    const KernelSource& source,
    const std::vector<Value>& template_args,
    const FileLoader& fs) {
    // Read file
    std::string filename = source.file_name();
    std::string content = source.read(fs);

    // Tokenize content of file
    internal::TokenStream stream(filename, content);

    // Extract annotated kernels from token stream
    auto result = internal::extract_annotated_kernels(stream);
    auto processed_source = KernelSource(filename, result.processed_source);

    for (const auto& kernel : result.kernels) {
        if (stream.matches(kernel.name, kernel_name)) {
            return internal::builder_from_annotated_kernel(
                stream,
                processed_source,
                kernel,
                template_args);
        }
    }

    throw std::runtime_error(
        "kernel '" + kernel_name + "' was not found in file \'" + filename
        + "\'");
}

PragmaKernel::PragmaKernel(
    std::string kernel_name,
    std::string path,
    std::vector<Value> template_args) :
    kernel_name_(std::move(kernel_name)),
    template_args_(std::move(template_args)) {
    /* We cannot resolve the file path at this moment since we do not what
     * type of `FileLoader` will be used during compilation.  */
    //const char* abs_path = realpath(path.c_str(), nullptr);
    //if (abs_path == nullptr) {
    //    throw std::runtime_error("failed to resolve path: '" + path + "'");
    //}

    file_path_ = path;
}

KernelBuilder PragmaKernel::build() const {
    return build_pragma_kernel(
        kernel_name_,
        KernelSource(file_path_),
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
