#include "kernel_launcher/builder.h"

#include "kernel_launcher/utils.h"

namespace kernel_launcher {

struct ArgsEval: Eval {
    explicit ArgsEval(
        const std::vector<KernelArg>& args,
        const Eval& fallback) :
        args_(args),
        fallback_(fallback) {}

    bool lookup(const Variable& v, Value& out) const override {
        if (const auto* that = dynamic_cast<const ArgExpr*>(&v)) {
            size_t i = that->get();

            if (i < args_.size()) {
                out = args_[i].to_value_or_empty();

                if (!out.is_empty()) {
                    return true;
                }
            }
        }

        return fallback_.lookup(v, out);
    }

  private:
    const std::vector<KernelArg>& args_;
    const Eval& fallback_;
};

struct DeviceAttrEval: Eval {
    DeviceAttrEval(CudaDevice device, const Eval& fallback) :
        device_(device),
        fallback_(fallback) {}

    bool lookup(const Variable& v, Value& out) const override {
        if (const auto* that = dynamic_cast<const DeviceAttributeExpr*>(&v)) {
            out = device_.attribute(that->get());
            return true;
        }

        return fallback_.lookup(v, out);
    }

    CudaDevice device_;
    const Eval& fallback_;
};

struct ProblemSizeEval: Eval {
    ProblemSizeEval(ProblemSize p, const Eval& fallback) :
        problem_(p),
        fallback_(fallback) {}

    bool lookup(const Variable& v, Value& value) const override {
        if (const auto* that = dynamic_cast<const ProblemExpr*>(&v)) {
            if (that->axis() < problem_.size()) {
                value = problem_[that->axis()];
                return true;
            }
        }

        return fallback_.lookup(v, value);
    }

  private:
    ProblemSize problem_;
    const Eval& fallback_;
};

struct DummyEval: Eval {
    bool lookup(const Variable& v, Value& value) const override {
        return false;
    }
};

void KernelInstance::launch(
    cudaStream_t stream,
    ProblemSize problem_size,
    const std::vector<KernelArg>& args,
    const Eval& fallback) const {
    ArgsEval args_eval {args, fallback};
    ProblemSizeEval eval {problem_size, args_eval};

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

    if (log_debug_enabled()) {
        auto p = problem_size;
        auto b = block_size;
        auto g = grid_size;

        log_debug() << "launching kernel " << module_.function_name() << "\n";
        log_debug() << " - device: " << CudaDevice::current().name() << "\n";
        log_debug() << " - problem size: ["  //
                    << p.x << ", " << p.y << ", " << p.z << "]\n";
        log_debug() << " - grid size: ["  //
                    << g.x << ", " << g.y << ", " << g.z << "]\n";
        log_debug() << " - block size: ["  //
                    << b.x << ", " << b.y << ", " << b.z << "]\n";

        if (smem > 0) {
            log_debug() << " - shared memory: " << smem << " bytes\n";
        }

        if (stream != nullptr) {
            log_debug() << " - stream: " << stream << "\n";
        }

        log_debug() << " - using " << args.size() << " arguments:\n";

        for (const auto& arg : args) {
            log_debug() << " - - " << arg << "\n";
        }
    }

    module_.launch(stream, grid_size, block_size, smem, ptrs.data());
}

void KernelInstance::launch(
    cudaStream_t stream,
    ProblemSize problem_size,
    const std::vector<KernelArg>& args) const {
    launch(stream, problem_size, args, DummyEval {});
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

KernelBuilder& KernelBuilder::argument_processor(ArgumentsProcessor f) {
    if (!f) {
        throw std::runtime_error(
            "null pointer given in `KernelBuilder::argument_processor(...)`");
    }

    args_processors_.push_back(std::move(f));
    return *this;
}

KernelBuilder& KernelBuilder::buffer_size(ArgExpr arg, TypedExpr<size_t> len) {
    return argument_processor([=](auto& args, auto& fallback) {
        ArgsEval eval {args, fallback};
        size_t i = arg.get();
        size_t n = eval(len);
        args[i] = args[i].to_array(n);
    });
}

KernelBuilder& KernelBuilder::block_size(
    TypedExpr<uint32_t> x,
    TypedExpr<uint32_t> y,
    TypedExpr<uint32_t> z) {
    block_size_set_ = true;
    block_size_[0] = std::move(x);
    block_size_[1] = std::move(y);
    block_size_[2] = std::move(z);
    return *this;
}

KernelBuilder& KernelBuilder::grid_size(
    TypedExpr<uint32_t> x,
    TypedExpr<uint32_t> y,
    TypedExpr<uint32_t> z) {
    grid_size_set_ = true;
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

KernelBuilder& KernelBuilder::problem_size(ProblemProcessor f) {
    if (!f) {
        throw std::runtime_error(
            "provided function cannot be null in "
            "`KernelBuilder::problem_size(...)`");
    }

    problem_processor_ = std::move(f);
    return *this;
}

KernelBuilder& KernelBuilder::problem_size(ProblemSize p) {
    problem_processor_ = [=](const std::vector<KernelArg>& /*unused*/) {
        return p;
    };
    return *this;
}

KernelBuilder& KernelBuilder::problem_size(
    TypedExpr<uint32_t> x,
    TypedExpr<uint32_t> y,
    TypedExpr<uint32_t> z) {
    problem_processor_ = [=](const std::vector<KernelArg>& args) {
        ArgsEval eval(args, DummyEval {});
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
    const Eval& eval,
    const std::vector<TypeInfo>& param_types) const {
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

ProblemProcessor KernelBuilder::problem_processor() const {
    if (!problem_processor_) {
        throw std::runtime_error(
            "no problem size configured. Please call "
            "`KernelBuilder::problem_size()` first.");
    }

    ProblemProcessor f = problem_processor_;
    std::vector<ArgumentsProcessor> procs = args_processors_;

    if (procs.empty()) {
        return f;
    }

    return [=](auto& args) {
        for (const auto& proc : procs) {
            proc(args, DummyEval {});
        }

        return f(args);
    };
}

static const TunableParam*
find_param(const ConfigSpace& space, const std::vector<std::string>& keys) {
    for (const auto& key : keys) {
        for (const auto& param : space.parameters()) {
            if (strcasecmp(param.name().c_str(), key.c_str()) == 0) {
                return &param;
            }
        }
    }

    return nullptr;
}

/**
 * Determine the block size for a specific axis. This should be the block size
 * set by the user (by calling `block_size(...)`) or, as a fallback, a
 * tunable variable named something like `block_size_x`.
 *
 * @param axis 0,1,2 for X,Y,Z
 */
TypedExpr<uint32_t> KernelBuilder::determine_block_size(size_t axis) const {
    if (block_size_set_) {
        return block_size_[axis];
    }

    if (axis == 0) {
        if (const auto* p = find_param(
                *this,
                {"block_size",
                 "block_dim",
                 "block_size_x",
                 "block_dim_x",
                 "block_x"})) {
            return *p;
        }

        // The block size for the X axis MUST be found. Otherwise, the kernel
        // will always run for block size `(1, 1, 1)` which is likely an error
        // by the user.
        throw std::runtime_error(
            "no block size configured. Please call "
            "`KernelBuilder::block_size(...)` first.");
    } else if (axis == 1) {
        if (const auto* p =
                find_param(*this, {"block_size_y", "block_dim_y", "block_y"})) {
            return *p;
        }
    } else if (axis == 2) {
        if (const auto* p =
                find_param(*this, {"block_size_z", "block_dim_z", "block_z"})) {
            return *p;
        }
    }

    return 1;
}

/**
 * Determine the grid size for a specific axis. This should be the grid size
 * set by the user (by calling `grid_size(...)`) or, as a fallback, a
 * tunable variable named something like `grid_size_x`.
 *
 * @param axis 0,1,2 for X,Y,Z
 */
TypedExpr<uint32_t> KernelBuilder::determine_grid_size(size_t axis) const {
    // First, return grid size set by the user
    if (grid_size_set_) {
        return grid_size_[axis];
    }

    // Second, find variable named "grid_size_x"
    if (axis == 0) {
        if (const auto* p = find_param(
                *this,
                {"grid_size", "grid_dim", "grid_size_x", "grid_dim_x"})) {
            return *p;
        }
    } else if (axis == 1) {
        if (const auto* p =
                find_param(*this, {"grid_size_y", "grid_dim_y", "grid_y"})) {
            return *p;
        }
    } else if (axis == 2) {
        if (const auto* p =
                find_param(*this, {"grid_size_z", "grid_dim_z", "grid_z"})) {
            return *p;
        }
    }

    // Third, find variable like "grid_div_x" and divide problem by divisor
    TypedExpr<uint32_t> divisor = determine_block_size(axis);

    if (axis == 0) {
        if (const auto* p = find_param(
                *this,
                {"grid_div", "grid_divisor", "grid_div_x", "grid_divisor_x"})) {
            divisor = *p;
        }
    } else if (axis == 1) {
        if (const auto* p =
                find_param(*this, {"grid_div_y", "grid_divisor_y"})) {
            divisor = *p;
        }
    } else if (axis == 2) {
        if (const auto* p =
                find_param(*this, {"grid_div_z", "grid_divisor_z"})) {
            divisor = *p;
        }
    }

    return div_ceil(kernel_launcher::problem_size(axis), divisor);
}

KernelInstance KernelBuilder::compile(
    const Config& config,
    const std::vector<TypeInfo>& param_types,
    const ICompiler& compiler,
    CudaContextHandle ctx) const {
    DeviceAttrEval eval = {ctx.device(), config};

    if (!is_valid(eval)) {
        std::stringstream ss;
        ss << "invalid configuration: " << config;
        throw std::runtime_error(ss.str());
    }

    std::array<TypedExpr<uint32_t>, 3> block_size = {
        determine_block_size(0).resolve(eval),
        determine_block_size(1).resolve(eval),
        determine_block_size(2).resolve(eval)};

    std::array<TypedExpr<uint32_t>, 3> grid_size = {
        determine_grid_size(0).resolve(eval),
        determine_grid_size(1).resolve(eval),
        determine_grid_size(2).resolve(eval)};

    TypedExpr<uint32_t> shared_mem = shared_mem_.resolve(eval);

    CudaModule module = compiler.compile(ctx, build(eval, param_types));
    return {
        std::move(module),
        std::move(block_size),
        std::move(grid_size),
        shared_mem};
}

}  // namespace kernel_launcher