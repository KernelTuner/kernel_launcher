#ifndef KERNEL_LAUNCHER_COMPILER_H
#define KERNEL_LAUNCHER_COMPILER_H

#include <cuda.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "kernel_launcher/cuda.h"
#include "kernel_launcher/fs.h"
#include "kernel_launcher/utils.h"

namespace kernel_launcher {

struct KernelSource {
    KernelSource(std::string filename) :
        filename_(std::move(filename)),
        has_content_(false) {}

    KernelSource(const char* filename) : KernelSource(std::string(filename)) {}

    KernelSource(std::string filename, std::string content) :
        filename_(std::move(filename)),
        content_(std::move(content)),
        has_content_(true) {}

    std::string file_name() const {
        return filename_;
    }

    const std::string* content() const {
        return has_content_ ? &content_ : nullptr;
    }

    std::string read(const FileLoader& fs) const;

  private:
    std::string filename_;
    std::string content_;
    bool has_content_;
};

struct KernelDef {
    KernelDef(std::string name, KernelSource source);
    void add_template_arg(TemplateArg arg);
    void add_parameter(TypeInfo dtype);
    void add_compiler_option(std::string option);
    void add_preincluded_header(KernelSource source);

    std::string name;
    KernelSource source;
    std::vector<KernelSource> preheaders;
    std::vector<TemplateArg> template_args;
    std::vector<TypeInfo> param_types;
    std::vector<std::string> options;
};

/**
 * Base clase for CUDA kernel compilers.
 */
struct ICompiler {
    virtual ~ICompiler() {}
    virtual void compile_ptx(
        KernelDef def,
        CudaArch arch_version,
        std::string& ptx_out,
        std::string& symbol_out) const = 0;

    /**
     * Compile the given kernel definition for the given CUDA context.
     *
     * @param ctx The CUDA context.
     * @param def The kernel definition.
     */
    virtual CudaModule compile(CudaContextHandle ctx, KernelDef def) const;
};

/**
 * Wrapper around `std::shared_ptr<ICompiler>`.
 */
struct Compiler: ICompiler {
    Compiler() = default;
    Compiler(Compiler&) = default;
    Compiler(const Compiler&) = default;
    Compiler(Compiler&&) = default;
    Compiler& operator=(const Compiler&) = default;
    Compiler& operator=(Compiler&&) = default;

    bool is_initialized() {
        return bool(inner_);
    }

    /**
     * Construct an instance of `Compiler` around an instance of type `C`
     * that should derive from `ICompiler`.
     *
     * @tparam C Type should derive from `ICompiler`.
     */
    template<typename C>
    Compiler(C&& compiler) :
        inner_(std::make_shared<typename std::decay<C>::type>(
            std::forward<C>(compiler))) {}

    void compile_ptx(
        KernelDef def,
        CudaArch arch,
        std::string& ptx_out,
        std::string& symbol_out) const override;

    CudaModule compile(CudaContextHandle ctx, KernelDef def) const override;

  private:
    std::shared_ptr<ICompiler> inner_;
};

struct NvrtcException: std::runtime_error {
    NvrtcException(const std::string& msg) : std::runtime_error(msg) {}
};

/**
 * `ICompiler` that uses [NVRTC](https://docs.nvidia.com/cuda/nvrtc/index.html)
 * to compile CUDA kernels.
 */
struct NvrtcCompiler: ICompiler {
    NvrtcCompiler(
        std::vector<std::string> options = {},
        std::shared_ptr<FileLoader> fs = {});

    static int version();

    void compile_ptx(
        KernelDef def,
        CudaArch arch,
        std::string& ptx_out,
        std::string& symbol_out) const override;

  private:
    std::shared_ptr<FileLoader> fs_;
    std::vector<std::string> default_options_;
};

/**
 * Retrieve the default compiler that is used to compile kernels. Use
 * `set_global_default_compiler` to change the default compiler.
 */
Compiler default_compiler();

/**
 * Set the global default compiler that is returned by `default_compiler`.
 */
void set_global_default_compiler(Compiler c);

}  // namespace kernel_launcher

#endif  //KERNEL_LAUNCHER_COMPILER_H
