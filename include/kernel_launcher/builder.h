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
    std::function<auto(const std::vector<KernelArg>&)->ProblemSize>;

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

    /**
     * Launch this kernel on the given stream using the given argument.
     *
     * @param stream
     * @param args
     */
    void launch(cudaStream_t stream, const std::vector<KernelArg>& args) const;

  private:
    CudaModule module_;
    ProblemExtractor problem_extractor_;
    std::array<TypedExpr<uint32_t>, 3> block_size_ = {1, 1, 1};
    std::array<TypedExpr<uint32_t>, 3> grid_size_ = {0, 0, 0};
    TypedExpr<uint32_t> shared_mem_ = 0;
};

struct KernelBuilderSerializerHack;

/**
 * A `KernelBuilder` is essentially a _blueprint_ that describes the information
 * required to compile and run a CUDA kernel for a given configuration. Most
 * methods take expressions that will be evaluated for a particular `Config`.
 *
 * The most important methods are:
 * * `tuning_key`: set the tuning key.
 * * `problem_size`: set the problem size.
 * * `block_size`: set the thread block size.
 * * `grid_divisors`: calculate grid size by dividing the problem size.
 * * `shared_mem`: set amount of shared memory in bytes.
 * * `template_arg`: Add a template argument.
 * * `compiler_flag`: Add a compiler_flag.
 * * `define`: Define a preprocessor variable.
 */
struct KernelBuilder: ConfigSpace {
    friend ::kernel_launcher::KernelBuilderSerializerHack;

    /**
     * Set the tuning key that will be used to find wisdom files for this kernel.
     *
     * @return `this`
     */
    KernelBuilder& tuning_key(std::string);

    /**
     * Set problem size for this kernel by providing a `ProblemSize` instance.
     *
     * @return `this`
     */
    KernelBuilder& problem_size(ProblemSize p);

    /**
     * Set the problem size for this kernel by providing an expression for
     * each dimension. These expressions can contain `ArgExpr` expressions
     * such as `arg0`, `arg1`, etc.
     *
     * @return `this`
     */
    KernelBuilder& problem_size(
        TypedExpr<uint32_t> x,
        TypedExpr<uint32_t> y = 1,
        TypedExpr<uint32_t> z = 1);

    /**
     * Set the problem size for this kernel by providing a function to
     * extract the problem size from the kernel arguments.
     *
     * @return `this`
     */
    KernelBuilder&
        problem_size(std::function<ProblemSize(const std::vector<KernelArg>&)>);

    /**
     * Construct a new `KernelBuilder`.
     * @param kernel_name Function name of the kernel. This should be the
     * fully qualified name of the function, i.e. including the namespace.
     * This name should not contain template parameters (this can be added
     * by calling `template_args`).
     * @param kernel_source The kernel source code. Can be either the file name
     * as a string or a `KernelSource` instance.
     */
    KernelBuilder(
        std::string kernel_name,
        KernelSource kernel_source,
        ConfigSpace space = {});

    /**
     * Set the block size for this kernel (i.e., number of threads per
     * thread block).
     *
     * @param x Block size along X.
     * @param y Block size along Y.
     * @param z Block size along Z.
     * @return `this`
     */
    KernelBuilder& block_size(
        TypedExpr<uint32_t> x,
        TypedExpr<uint32_t> y = 1,
        TypedExpr<uint32_t> z = 1);

    /**
     * Set the grid size for this kernel (i.e., number of thread blocks
     * along each direction).
     *
     * @param x Grid size along X.
     * @param y Grid size along Y.
     * @param z Grid size along Z.
     * @return `this`
     */
    KernelBuilder& grid_size(
        TypedExpr<uint32_t> x,
        TypedExpr<uint32_t> y = 1,
        TypedExpr<uint32_t> z = 1);

    /**
     * Set the grid size for this kernel (i.e., number of thread blocks
     * along each direction) by dividing the `problem_size` by the given
     * `divisors`. For example, if the problem size is `(100, 100)` and
     * the divisors are `(5, 15)` then the grid size will be `(20, 7)`.
     *
     * @param x Grid divisor along X.
     * @param y Grid divisor along Y.
     * @param z Grid divisor along Z.
     * @return `this`
     */
    KernelBuilder& grid_divisors(
        TypedExpr<uint32_t> x,
        TypedExpr<uint32_t> y = 1,
        TypedExpr<uint32_t> z = 1);

    /**
     * Set the amount of shared memory in bytes.
     * @return `this`
     */
    KernelBuilder& shared_memory(TypedExpr<uint32_t> smem);

    KernelBuilder& assertion(TypedExpr<bool> e);

    /**
     * Add a preprocessor variable definition with the provided `name` and
     * `value`.
     *
     * @return `this`
     */
    KernelBuilder& define(std::string name, TypedExpr<std::string> value);
    KernelBuilder& define(ParamExpr p) {
        return define(p.parameter().name(), p);
    }

    /**
     * Add a header files that must be preincluded during compilation.
     *
     * @return `this`.
     */
    KernelBuilder& include_header(KernelSource source);

    /**
     * Add one or more template arguments. Each argument must be convertible
     * to an instance of `TemplateArg`.
     *
     * @return `this`
     */
    template<typename... Ts>
    KernelBuilder& template_args(TypedExpr<TemplateArg> first, Ts&&... rest) {
        template_arg(std::move(first));
        return template_args(std::forward<Ts>(rest)...);
    }

    KernelBuilder& template_arg(TypedExpr<TemplateArg> arg);
    KernelBuilder& template_args() {
        return *this;
    }

    /**
     * Add one or more types `Ts...` as template arguments to this kernel.
     *
     * Short-hand for:
     *
     * ```
     * builder.template_args(type_of<Ts>()...);
     * ```
     *
     * @return `this`
     */
    template<typename... Ts>
    KernelBuilder& template_types() {
        return template_args(type_of<Ts>()...);
    }

    /**
     * Add type `T` as a template argument to this kernel.
     *
     * Short-hand for:
     *
     * ```
     * builder.template_arg(type_of<T>());
     * ```
     *
     * @return `this`
     */
    template<typename T>
    KernelBuilder& template_type() {
        return template_arg(type_of<T>());
    }

    /**
     * Add one ore more compilation flags that will be passed to the compiler.
     * Each argument must be convertible to a string.
     *
     * @return `this`
     */
    template<typename... Ts>
    KernelBuilder& compiler_flags(TypedExpr<std::string> first, Ts&&... rest) {
        compiler_flag(std::move(first));
        return compiler_flags(std::forward<Ts>(rest)...);
    }

    KernelBuilder& compiler_flag(TypedExpr<std::string> opt);
    KernelBuilder& compiler_flags() {
        return *this;
    }

    /**
     * Short-hand for:
     *
     * ```
     * builder.block_size(
     *      builder.tune("BLOCK_SIZE_X", xs),
     *      builder.tune("BLOCK_SIZE_Y", ys),
     *      builder.tune("BLOCK_SIZE_Z", zs));
     * ```
     */
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

    /**
     * Short-hand for
     *
     * ```
     * builder.define(name, builder.tune(name, values));
     * ```
     */
    template<typename T = Value>
    TypedExpr<T> tune_define(std::string name, std::vector<T> values) {
        TypedExpr<T> param = this->tune(name, values);
        define(std::move(name), param);
        return param;
    }

    /**
     * Compile an instance of this kernel using the given configuration.
     *
     * @param config The configuration.
     * @param param_types The types of the parameters of this kernel.
     * @param compiler The CUDA compiler to use.
     * @param ctx The CUDA context for this CUDA kernel.
     * @return
     */
    KernelInstance compile(
        const Config& config,
        const std::vector<TypeInfo>& param_types,
        const ICompiler& compiler = default_compiler(),
        CudaContextHandle ctx = CudaContextHandle::current()) const;

    /**
     * Returns the function name of this kernel.
     */
    const std::string& kernel_name() const {
        return kernel_name_;
    }

    /**
     * Returns the tuning key for this kernel.
     */
    const std::string& tuning_key() const {
        return tuning_key_;
    }

    ProblemSize extract_problem_size(const std::vector<KernelArg>&) const;

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
