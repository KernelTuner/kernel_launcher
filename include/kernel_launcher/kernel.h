#ifndef KERNEL_LAUNCHER_KERNEL_H
#define KERNEL_LAUNCHER_KERNEL_H

#include <cuda.h>

#include <iostream>
#include <utility>

#include "kernel_launcher/compiler.h"
#include "kernel_launcher/config.h"

namespace kernel_launcher {

struct KernelInstance {
    KernelInstance() = default;

    KernelInstance(
        CudaModule module,
        std::array<TypedExpr<uint32_t>, 3> block_size,
        std::array<TypedExpr<uint32_t>, 3> grid_size,
        uint32_t shared_mem) :
        module_(std::move(module)),
        block_size_(std::move(block_size)),
        grid_size_(std::move(grid_size)),
        shared_mem_(shared_mem) {}

    void
    launch(cudaStream_t stream, ProblemSize problem_size, void** args) const;

  private:
    CudaModule module_;
    std::array<TypedExpr<uint32_t>, 3> block_size_ = {1, 1, 1};
    std::array<TypedExpr<uint32_t>, 3> grid_size_ = {0, 0, 0};
    uint32_t shared_mem_ = {};
};

struct KernelBuilderSerializerHack;

struct KernelBuilder: ConfigSpace {
    friend ::kernel_launcher::KernelBuilderSerializerHack;

    KernelBuilder(
        std::string kernel_name,
        KernelSource kernel_source,
        ConfigSpace space = {}) :
        ConfigSpace(std::move(space)),
        kernel_name_(std::move(kernel_name)),
        kernel_source_(std::move(kernel_source)) {
        block_size(1, 1, 1);
    }

    KernelBuilder(std::string kernel_name, std::string kernel_file) :
        KernelBuilder(
            std::move(kernel_name),
            KernelSource(std::move(kernel_file))) {}

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

    KernelDef
    build(const Config& config, const std::vector<TypeInfo>& param_types) const;

    KernelInstance compile(
        const Config& config,
        const std::vector<TypeInfo>& param_types,
        const ICompiler& compiler = default_compiler(),
        CudaContextHandle ctx = CudaContextHandle::current()) const;

  private:
    std::string kernel_name_;
    KernelSource kernel_source_;
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

template<typename... Args>
struct KernelLaunch {
    KernelLaunch(
        cudaStream_t stream,
        dim3 problem_size,
        const KernelInstance& kernel) :
        stream_(stream),
        problem_size_(problem_size),
        kernel_ref_(kernel) {
        //
    }

    void launch(Args... args) const {
        std::array<void*, sizeof...(Args)> raw_args = {&args...};
        kernel_ref_.launch(stream_, problem_size_, raw_args.data());
    }

    void operator()(Args... args) {
        return launch(args...);
    }

  private:
    cudaStream_t stream_;
    dim3 problem_size_;
    const KernelInstance& kernel_ref_;
};

template<typename... Args>
struct Kernel {
    using launch_type = KernelLaunch<Args...>;

    Kernel() = default;
    void compile(
        const KernelBuilder& builder,
        const Config& config,
        const ICompiler& compiler = default_compiler(),
        CudaContextHandle ctx = CudaContextHandle::current()) {
        instance_ = builder.compile(
            config,
            std::vector<TypeInfo> {type_of<Args>()...},
            compiler,
            ctx);
    }

    launch_type instantiate(cudaStream_t stream, dim3 problem_size) {
        return launch_type(stream, problem_size, instance_);
    }

    launch_type operator()(cudaStream_t stream, dim3 problem_size) {
        return instantiate(stream, problem_size);
    }

    launch_type operator()(dim3 problem_size) {
        return instantiate(nullptr, problem_size);
    }

    launch_type operator()(
        cudaStream_t stream,
        uint32_t problem_x,
        uint32_t problem_y,
        uint32_t problem_z = 1) {
        return instantiate(stream, dim3(problem_x, problem_y, problem_z));
    }

    launch_type
    operator()(uint32_t problem_x, uint32_t problem_y, uint32_t problem_z = 1) {
        return instantiate(dim3(problem_x, problem_y, problem_z));
    }

  private:
    KernelInstance instance_;
};

}  // namespace kernel_launcher

#endif  //KERNEL_LAUNCHER_KERNEL_H
