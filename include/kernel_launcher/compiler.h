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

/***
 */
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

    std::string read(const FileResolver& fs) const {
        if (has_content_) {
            return content_;
        } else {
            const std::vector<char>& content = fs.read(filename_);
            return std::string(content.begin(), content.end());
        }
    }

  private:
    std::string filename_;
    std::string content_;
    bool has_content_;
};

struct KernelDef {
    std::string kernel_name;
    KernelSource source;
    std::vector<TemplateArg> template_args;
    std::vector<TypeInfo> param_types;
    std::vector<std::string> options;
};

struct CompilerBase {
    virtual ~CompilerBase() {}
    virtual CudaModule compile(const KernelDef&) const = 0;
};

struct Compiler: CompilerBase {
    Compiler() = default;
    Compiler(Compiler&) = default;
    Compiler(const Compiler&) = default;
    Compiler(Compiler&&) = default;

    template<typename C>
    Compiler(C&& compiler) :
        inner_(std::make_shared<typename std::decay<C>::type>(
            std::forward<C>(compiler))) {}

    CudaModule compile(const KernelDef& def) const override {
        return inner_->compile(def);
    }

  private:
    std::shared_ptr<CompilerBase> inner_;
};

struct NvrtcException: std::runtime_error {
    NvrtcException(std::string msg) : std::runtime_error(std::move(msg)) {}
};

struct NvrtcCompiler: CompilerBase {
    NvrtcCompiler(std::vector<std::string> options = {}, FileResolver fs = {}) :
        default_options_(std::move(options)),
        fs_(std::move(fs)) {}

    static int version();

    void compile_ptx(
        const KernelDef& def,
        std::string& ptx,
        std::string& symbol_name,
        CudaDevice = CudaDevice::current()) const;

    CudaModule compile(const KernelDef& spec) const override;

  private:
    mutable std::vector<std::pair<std::string, std::string>> file_cache_;
    std::vector<std::string> default_options_;
    FileResolver fs_;
};

}  // namespace kernel_launcher

#endif  //KERNEL_LAUNCHER_COMPILER_H
