#include "kernel_launcher/kernel.h"

namespace kernel_launcher {

KernelDef KernelBuilder::build(
    const Config& config,
    const std::vector<TypeInfo>& param_types) const {
    if (!is_valid(config)) {
        throw std::runtime_error("invalid config");
    }

    Eval eval = {config.get()};
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

    for (const auto& p : defines_) {
        def.add_compiler_option("--define-macro");
        def.add_compiler_option(p.first + "=" + eval(p.second));
    }

    return def;
}

KernelInstance KernelBuilder::compile(
    const Config& config,
    const std::vector<TypeInfo>& param_types,
    const CompilerBase& compiler,
    CudaContextHandle ctx) const {
    CudaModule module = compiler.compile(ctx, build(config, param_types));

    Eval eval = {config.get()};
    dim3 block_size = {
        eval(block_size_[0]),
        eval(block_size_[1]),
        eval(block_size_[2])};

    dim3 grid_divisor = {
        eval(grid_divisors_[0]),
        eval(grid_divisors_[1]),
        eval(grid_divisors_[2])};

    uint32_t shared_mem = eval(shared_mem_);

    return KernelInstance(
        std::move(module),
        block_size,
        grid_divisor,
        shared_mem);
}

}  // namespace kernel_launcher