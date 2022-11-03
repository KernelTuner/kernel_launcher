#include "kernel_launcher/builder.h"

#include "kernel_launcher/utils.h"

namespace kernel_launcher {

struct ArgsEval: Eval {
    ArgsEval(const std::vector<KernelArg>& args) : args_(args) {}

    bool lookup(const Variable& v, TunableValue& out) const override {
        if (const auto* that = dynamic_cast<const ArgExpr*>(&v)) {
            size_t i = that->get();

            if (i < args_.size()) {
                out = args_[i].to_value_or_empty();

                if (!out.is_empty()) {
                    return true;
                }
            }
        }

        return false;
    }

  private:
    const std::vector<KernelArg>& args_;
};

struct ProblemSizeEval: Eval {
    ProblemSizeEval(ProblemSize p, ArgsEval args, const Eval& fallback) :
        problem_(p),
        args_(std::move(args)),
        fallback_(fallback) {}

    bool lookup(const Variable& v, TunableValue& value) const override {
        if (const auto* that = dynamic_cast<const ProblemExpr*>(&v)) {
            if (that->axis() < problem_.size()) {
                value = problem_[that->axis()];
                return true;
            }
        }

        if (const auto* that = dynamic_cast<const DeviceAttributeExpr*>(&v)) {
            value = CudaDevice::current().attribute(that->get());
            return true;
        }

        if (args_.lookup(v, value)) {
            return true;
        }

        if (fallback_.lookup(v, value)) {
            return true;
        }

        return false;
    }

  private:
    ProblemSize problem_;
    ArgsEval args_;
    const Eval& fallback_;
};

struct DummyEval: Eval {
    bool lookup(const Variable& v, TunableValue& value) const override {
        return false;
    }
};

void KernelInstance::launch(
    cudaStream_t stream,
    const std::vector<KernelArg>& args,
    const Eval& fallback) const {
    ProblemSize problem_size = problem_extractor_(args);
    ProblemSizeEval eval {problem_size, args, fallback};

    dim3 grid_size = {
        eval(grid_size_[0]),
        eval(grid_size_[1]),
        eval(grid_size_[2])};

    dim3 block_size = {
        eval(block_size_[0]),
        eval(block_size_[1]),
        eval(block_size_[2])};

    uint32_t smem = eval(shared_mem_);

    std::vector<void*> ptrs {args.size()};
    for (size_t i = 0; i < args.size(); i++) {
        ptrs[i] = args[i].as_void_ptr();
    }

    module_.launch(stream, grid_size, block_size, smem, ptrs.data());
}

void KernelInstance::launch(
    cudaStream_t stream,
    const std::vector<KernelArg>& args) const {
    launch(stream, args, DummyEval {});
}

KernelBuilder::KernelBuilder(
    std::string kernel_name,
    KernelSource kernel_source,
    ConfigSpace space) :
    ConfigSpace(std::move(space)),
    kernel_name_(std::move(kernel_name)),
    kernel_source_(std::move(kernel_source)),
    tuning_key_(kernel_name_) {
    problem_size(0, 0, 0);
    block_size(1, 1, 1);
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
        grid_size_[0] = div_ceil(problem_size_x, block_size_[0]);
        grid_size_[1] = div_ceil(problem_size_y, block_size_[1]);
        grid_size_[2] = div_ceil(problem_size_z, block_size_[2]);
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
        div_ceil(problem_size_x, x),
        div_ceil(problem_size_y, y),
        div_ceil(problem_size_z, z));
}

KernelBuilder& KernelBuilder::tuning_key(std::string key) {
    tuning_key_ = std::move(key);
    return *this;
}

KernelBuilder& KernelBuilder::problem_size(ProblemExtractor f) {
    if (!f) {
        throw std::runtime_error(
            "provided function cannot be uninitialized in "
            "KernelBuilder::problem_size");
    }

    problem_extractor_ = std::move(f);
    return *this;
}

KernelBuilder& KernelBuilder::problem_size(ProblemSize p) {
    problem_extractor_ = [=](const std::vector<KernelArg>& /*unused*/) {
        return p;
    };
    return *this;
}

KernelBuilder& KernelBuilder::problem_size(
    TypedExpr<uint32_t> x,
    TypedExpr<uint32_t> y,
    TypedExpr<uint32_t> z) {
    problem_extractor_ = [=](const std::vector<KernelArg>& args) {
        ArgsEval eval(args);
        return ProblemSize {eval(x), eval(y), eval(z)};
    };

    return *this;
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
        problem_extractor_,
        std::move(block_size),
        std::move(grid_size),
        shared_mem);
}

}  // namespace kernel_launcher