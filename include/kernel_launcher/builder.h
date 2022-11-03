#ifndef KERNEL_LAUNCHER_BUILDER_H
#define KERNEL_LAUNCHER_BUILDER_H

#include <unordered_map>
#include <vector>

#include "kernel_launcher/arg.h"
#include "kernel_launcher/compiler.h"
#include "kernel_launcher/config.h"

namespace kernel_launcher {

struct KernelArg;
using ProblemExtractor =
    std::function<ProblemSize(const std::vector<KernelArg>&)>;

struct KernelInstance {
    KernelInstance() = default;

    KernelInstance(
        CudaModule module,
        ProblemExtractor problem,
        std::array<TypedExpr<uint32_t>, 3> block_size,
        std::array<TypedExpr<uint32_t>, 3> grid_size,
        TypedExpr<uint32_t> shared_mem) :
        module_(std::move(module)),
        problem_extractor_(std::move(problem)),
        block_size_(std::move(block_size)),
        grid_size_(std::move(grid_size)),
        shared_mem_(std::move(shared_mem)) {}

    void launch(
        cudaStream_t stream,
        const std::vector<KernelArg>& args,
        const Eval& fallback) const;

    void launch(cudaStream_t stream, const std::vector<KernelArg>& args) const;

  private:
    CudaModule module_;
    ProblemExtractor problem_extractor_;
    std::array<TypedExpr<uint32_t>, 3> block_size_ = {1, 1, 1};
    std::array<TypedExpr<uint32_t>, 3> grid_size_ = {0, 0, 0};
    TypedExpr<uint32_t> shared_mem_ = 0;
};

struct KernelBuilderSerializerHack;
struct WisdomKernel;
template<typename... Args>
struct Kernel;

struct KernelBuilder: ConfigSpace {
    friend ::kernel_launcher::KernelBuilderSerializerHack;
    friend ::kernel_launcher::WisdomKernel;

    template<typename... Args>
    friend struct ::kernel_launcher::Kernel;

    KernelBuilder(
        std::string kernel_name,
        KernelSource kernel_source,
        ConfigSpace space = {});

    const std::string& kernel_name() const {
        return kernel_name_;
    }

    KernelBuilder& block_size(
        TypedExpr<uint32_t> x,
        TypedExpr<uint32_t> y = 1,
        TypedExpr<uint32_t> z = 1);

    KernelBuilder& grid_size(
        TypedExpr<uint32_t> x,
        TypedExpr<uint32_t> y = 1,
        TypedExpr<uint32_t> z = 1);

    KernelBuilder& grid_divisors(
        TypedExpr<uint32_t> x,
        TypedExpr<uint32_t> y = 1,
        TypedExpr<uint32_t> z = 1);

    KernelBuilder& shared_memory(TypedExpr<uint32_t> smem);
    KernelBuilder& template_arg(TypedExpr<TemplateArg> arg);
    KernelBuilder& assertion(TypedExpr<bool> e);
    KernelBuilder& define(std::string name, TypedExpr<std::string> value);
    KernelBuilder& compiler_flag(TypedExpr<std::string> opt);
    KernelBuilder& include_header(KernelSource source);
    KernelBuilder& tuning_key(std::string);

    KernelBuilder& problem_size(
        TypedExpr<uint32_t> x,
        TypedExpr<uint32_t> y = 1,
        TypedExpr<uint32_t> z = 1);
    KernelBuilder& problem_size(ProblemSize p);
    KernelBuilder& problem_size(ProblemExtractor);

    template<typename T, typename... Ts>
    KernelBuilder& template_args(T&& first, Ts&&... rest) {
        template_arg(std::forward<T>(first));
        return template_args(std::forward<Ts>(rest)...);
    }

    KernelBuilder& template_args() {
        return *this;
    }

    template<typename T, typename... Ts>
    KernelBuilder& compiler_flags(T&& first, Ts&&... rest) {
        compiler_flag(std::forward<T>(first));
        return compiler_flags(std::forward<Ts>(rest)...);
    }

    KernelBuilder& compiler_flags() {
        return *this;
    }

    KernelBuilder& define(ParamExpr p) {
        return define(p.parameter().name(), p);
    }

    std::array<TypedExpr<uint32_t>, 3> tune_block_size(
        std::vector<uint32_t> xs,
        std::vector<uint32_t> ys = {1u},
        std::vector<uint32_t> zs = {1u}) {
        block_size(
            tune("BLOCK_SIZE_X", xs),
            tune("BLOCK_SIZE_Y", ys),
            tune("BLOCK_SIZE_Z", zs));
        return block_size_;
    }

    template<typename T>
    TypedExpr<T> tune_define(std::string name, std::vector<T> values) {
        TypedExpr<T> param = this->tune(name, values);
        define(std::move(name), param);
        return param;
    }

    template<typename T>
    TypedExpr<T>
    tune_define(std::string name, std::initializer_list<T> values) {
        return tune_define(name, std::vector<T>(values));
    }

    KernelInstance compile(
        const Config& config,
        const std::vector<TypeInfo>& param_types,
        const ICompiler& compiler = default_compiler(),
        CudaContextHandle ctx = CudaContextHandle::current()) const;

  private:
    KernelDef
    build(const Eval& eval, const std::vector<TypeInfo>& param_types) const;

    std::string kernel_name_;
    KernelSource kernel_source_;
    std::string tuning_key_;
    ProblemExtractor problem_extractor_;
    std::vector<KernelSource> preheaders_;
    std::array<TypedExpr<uint32_t>, 3> block_size_ = {1u, 1u, 1u};
    std::array<TypedExpr<uint32_t>, 3> grid_size_ = {1u, 1u, 1u};
    bool grid_set_ = false;
    TypedExpr<uint32_t> shared_mem_ = {0u};
    std::vector<TypedExpr<TemplateArg>> template_args_ {};
    std::vector<TypedExpr<std::string>> compile_flags_ {};
    std::vector<TypedExpr<bool>> assertions_ {};
    std::unordered_map<std::string, TypedExpr<std::string>> defines_ {};
};

}  // namespace kernel_launcher

#endif  //KERNEL_LAUNCHER_BUILDER_H