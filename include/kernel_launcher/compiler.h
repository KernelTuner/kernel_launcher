#ifndef KERNEL_LAUNCHER_COMPILER_H
#define KERNEL_LAUNCHER_COMPILER_H

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

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

struct KernelModule {
    KernelModule(const char* image, const char* fun_name);
    ~KernelModule();
    KernelModule() = default;
    KernelModule(const KernelModule&) = delete;
    KernelModule& operator=(const KernelModule&) = delete;

    KernelModule(KernelModule&& that) noexcept {
        *this = std::move(that);
    }

    KernelModule& operator=(KernelModule&& that) noexcept {
        std::swap(that.module_, module_);
        std::swap(that.fun_ptr_, fun_ptr_);
        return *this;
    }

    void launch(
        CUstream stream,
        dim3 grid_size,
        dim3 block_size,
        uint32_t shared_mem,
        void** args) const;

    CUfunction function() const {
        return fun_ptr_;
    }

    bool valid() const {
        return module_ != nullptr;
    }

  private:
    CUfunction fun_ptr_ = nullptr;
    CUmodule module_ = nullptr;
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
    virtual KernelModule compile(const KernelDef&) const = 0;
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

    KernelModule compile(const KernelDef& def) const override {
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

    void compile_ptx(
        const KernelDef& def,
        std::string& ptx,
        std::string& symbol_name) const;

    KernelModule compile(const KernelDef& spec) const override;

  private:
    mutable std::vector<std::pair<std::string, std::string>> file_cache_;
    std::vector<std::string> default_options_;
    FileResolver fs_;
};

}  // namespace kernel_launcher

#endif  //KERNEL_LAUNCHER_COMPILER_H
