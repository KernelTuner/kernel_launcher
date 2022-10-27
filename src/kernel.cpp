#include "kernel_launcher/kernel.h"

namespace kernel_launcher {

struct ProblemSizeEval: Eval {
    ProblemSizeEval(ProblemSize p) : problem_(p) {}

    bool lookup(const Variable& v, TunableValue& value) const override {
        if (v == problem_size_x().variable()) {
            value = problem_[0];
        } else if (v == problem_size_y().variable()) {
            value = problem_[1];
        } else if (v == problem_size_z().variable()) {
            value = problem_[2];
        } else {
            return false;
        }

        return true;
    }

  private:
    ProblemSize problem_;
};

void KernelInstance::launch(
    cudaStream_t stream,
    ProblemSize problem_size,
    void** args) const {
    ProblemSizeEval ctx(problem_size);
    dim3 grid_size = {
        ctx(grid_size_[0]),
        ctx(grid_size_[1]),
        ctx(grid_size_[2])};

    dim3 block_size = {
        ctx(block_size_[0]),
        ctx(block_size_[1]),
        ctx(block_size_[2])};

    return module_.launch(stream, grid_size, block_size, shared_mem_, args);
}

KernelBuilder& KernelBuilder::block_size(
    TypedExpr<uint32_t> x,
    TypedExpr<uint32_t> y,
    TypedExpr<uint32_t> z) {
    block_size_[0] = std::move(x);
    block_size_[1] = std::move(y);
    block_size_[2] = std::move(z);

    // if `grid_size` or `grid_divisors` has not been called explicitly yet,
    // then we set the grid size implicitly here.
    if (!grid_set_) {
        grid_size_[0] = div_ceil(problem_size(0), block_size_[0]);
        grid_size_[1] = div_ceil(problem_size(1), block_size_[1]);
        grid_size_[2] = div_ceil(problem_size(2), block_size_[2]);
    }

    return *this;
}

KernelBuilder& KernelBuilder::grid_size(
    TypedExpr<uint32_t> x,
    TypedExpr<uint32_t> y,
    TypedExpr<uint32_t> z) {
    grid_set_ = true;
    grid_size_[0] = std::move(x);
    grid_size_[1] = std::move(y);
    grid_size_[2] = std::move(z);
    return *this;
}

KernelBuilder& KernelBuilder::grid_divisors(
    TypedExpr<uint32_t> x,
    TypedExpr<uint32_t> y,
    TypedExpr<uint32_t> z) {
    return grid_size(
        div_ceil(problem_size(0), x),
        div_ceil(problem_size(1), y),
        div_ceil(problem_size(2), z));
}

KernelBuilder& KernelBuilder::shared_memory(TypedExpr<uint32_t> smem) {
    shared_mem_ = std::move(smem);
    return *this;
}

KernelBuilder& KernelBuilder::template_arg(TypedExpr<TemplateArg> arg) {
    template_args_.push_back(std::move(arg));
    return *this;
}

KernelBuilder& KernelBuilder::assertion(TypedExpr<bool> e) {
    assertions_.push_back(std::move(e));
    return *this;
}

KernelBuilder&
KernelBuilder::define(std::string name, TypedExpr<std::string> value) {
    defines_.emplace(std::move(name), std::move(value));
    return *this;
}

KernelBuilder& KernelBuilder::compiler_flag(TypedExpr<std::string> opt) {
    compile_flags_.emplace_back(std::move(opt));
    return *this;
}

KernelBuilder& KernelBuilder::include_header(KernelSource source) {
    preheaders_.push_back(std::move(source));
    return *this;
}

KernelDef KernelBuilder::build(
    const Config& config,
    const std::vector<TypeInfo>& param_types) const {
    if (!is_valid(config)) {
        std::stringstream ss;
        ss << "invalid configuration: " << config;
        throw std::runtime_error(ss.str());
    }

    const Eval& eval = config;
    KernelDef def(kernel_name_, kernel_source_);

    for (TypeInfo param : param_types) {
        def.add_parameter(param);
    }

    for (const auto& p : template_args_) {
        def.add_template_arg(eval(p));
    }

    for (const auto& p : compile_flags_) {
        std::string option = eval(p);

        if (!option.empty()) {
            def.add_compiler_option(std::move(option));
        }
    }

    for (const auto& source : preheaders_) {
        def.add_preincluded_header(source);
    }

    for (const auto& p : defines_) {
        def.add_compiler_option("--define-macro");
        def.add_compiler_option(p.first + "=" + eval(p.second));
    }

    return def;
}

KernelInstance KernelBuilder::compile(
    const Config& config,
    const std::vector<TypeInfo>& param_types,
    const ICompiler& compiler,
    CudaContextHandle ctx) const {
    CudaModule module = compiler.compile(ctx, build(config, param_types));

    const Eval& eval = config;
    std::array<TypedExpr<uint32_t>, 3> block_size = {
        block_size_[0].resolve(eval),
        block_size_[1].resolve(eval),
        block_size_[2].resolve(eval)};

    std::array<TypedExpr<uint32_t>, 3> grid_size = {
        grid_size_[0].resolve(eval),
        grid_size_[1].resolve(eval),
        grid_size_[2].resolve(eval)};

    uint32_t shared_mem = eval(shared_mem_);

    return KernelInstance(
        std::move(module),
        std::move(block_size),
        std::move(grid_size),
        shared_mem);
}

}  // namespace kernel_launcher