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
        has_content_(false) {
        //
    }

    KernelSource(std::string filename, std::string content) :
        filename_(std::move(filename)),
        content_(std::move(content)),
        has_content_(true) {
        //
    }

    std::string file_name() const {
        return filename_;
    }

    std::string
    read(const FileLoader& fs, const std::vector<std::string>& dirs) const;

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

struct CompilerBase {
    virtual ~CompilerBase() {}
    virtual void compile_ptx(
        const KernelDef& def,
        int arch_version,
        std::string& ptx_out,
        std::string& symbol_out) const = 0;

    virtual CudaModule compile(CudaContextHandle ctx, KernelDef def) const;
};

struct Compiler: CompilerBase {
    Compiler() = default;
    Compiler(Compiler&) = default;
    Compiler(const Compiler&) = default;
    Compiler(Compiler&&) = default;
    Compiler& operator=(const Compiler&) = default;
    Compiler& operator=(Compiler&&) = default;

    template<typename C>
    Compiler(C&& compiler) :
        inner_(std::make_shared<typename std::decay<C>::type>(
            std::forward<C>(compiler))) {}

    void compile_ptx(
        const KernelDef& def,
        int arch_version,
        std::string& ptx_out,
        std::string& symbol_out) const override;
    CudaModule compile(CudaContextHandle ctx, KernelDef def) const override;

  private:
    std::shared_ptr<CompilerBase> inner_;
};

struct NvrtcException: std::runtime_error {
    NvrtcException(std::string msg) : std::runtime_error(std::move(msg)) {}
};

struct NvrtcCompiler: CompilerBase {
    NvrtcCompiler(
        std::vector<std::string> options = {},
        std::shared_ptr<FileLoader> fs = {});

    static int version();

    void compile_ptx(
        const KernelDef& def,
        int arch_version,
        std::string& ptx_out,
        std::string& name_out) const override;

  private:
    std::shared_ptr<FileLoader> fs_;
    std::vector<std::string> default_options_;
};

}  // namespace kernel_launcher

#endif  //KERNEL_LAUNCHER_COMPILER_H
