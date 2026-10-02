#ifndef KERNEL_LAUNCHER_KERNEL_BUILDER_H
#define KERNEL_LAUNCHER_KERNEL_BUILDER_H

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "kernel_launcher/compiler.hpp"
#include "kernel_launcher/config.hpp"
#include "kernel_launcher/expr.hpp"
#include "kernel_launcher/gpu_utils.hpp"
#include "kernel_launcher/kernel_arg.hpp"
#include "kernel_launcher/problem_key.hpp"

namespace kernel_launcher {

/**
 * @addtogroup builder
 * @{
 */

/**
 * A `KernelBuilder` is essentially a _blueprint_ that describes the information
 * required to compile and run a GPU kernel for a given configuration. Most
 * methods take expressions that will be evaluated at a later point in time. For example,
 * the block size could be calculated based on a wisdom configuration.
 *
 * The builder will be used for these:
 * - `build_problem_key()`: extracts the `ProblemKey` from the user arguments.
 * - `build_config_space()`: gives the `ConfigSpace` for a specific `ProblemKey`.
 * - `build_kernel_def()`: gives the `KernelDef` for a specific `Config` and `ProblemKey`.
 * - `build_launch_config()`: gives the `LaunchConfig` required to launch the kernel.
 *
 * The most important setter methods are:
 * - `tuning_key()`: set the tuning key.
 * - `problem_size()`: set the problem size.
 * - `feature()`: add a named feature to the problem key.
 * - `block_size()`: set the thread block size.
 * - `grid_size()`: set the grid size explicitly.
 * - `grid_divisor()`: calculate grid size by dividing the problem size.
 * - `shared_memory()`: set amount of shared memory in bytes.
 * - `assertion()`: add a condition that is checked before each launch.
 * - `buffer_size()`: set the length of the buffer of a pointer argument.
 * - `template_arg()`: add a template argument.
 * - `compiler_flag()`: add a compiler flag.
 * - `define()`: define a preprocessor variable.
 *
 * If `block_size()` is not called, the block size is derived from the tunable parameters with
 * names such as `block_size`, `block_size_x`, `block_size_y`, and `block_size_z` (see
 * `resolve_block_size()`).
 */
class KernelBuilder: public ConfigSpace {
  public:
    /**
     * Create a builder for the kernel `kernel_name` in `kernel_source`, with the tunable
     * parameters of `space`. The tuning key defaults to the kernel name.
     */
    KernelBuilder(std::string kernel_name, FileSource kernel_source, ConfigSpace space = {}) :
        ConfigSpace(std::move(space)),
        kernel_name_(std::move(kernel_name)),
        kernel_source_(std::move(kernel_source)),
        tuning_key_(kernel_name_) {}

    /**
     * Evaluate the problem size and features for the given kernel arguments. Also attaches the
     * buffer sizes (see `buffer_size()`) to the pointer arguments in `args`.
     */
    ProblemKey build_problem_key(std::vector<KernelArg>& args) const;

    /**
     * Return the configuration space with the problem and device variables substituted.
     */
    ConfigSpace build_config_space(const GPUDevice& device, const ProblemKey& problem_key) const;

    /**
     * Return the `KernelDef` (template arguments, compiler flags, defines, headers) needed to
     * compile the kernel for the given `config`.
     */
    KernelDef build_kernel_def(
        const GPUDevice& device,
        const Config& config,
        const ProblemKey& problem_key,
        std::vector<TypeInfo> param_types) const;

    /**
     * Check the assertions and evaluate the block size, grid size, and shared memory needed to
     * launch the compiled `kernel`.
     */
    LaunchConfig build_launch_config(
        const GPUDevice& device,
        const GPUKernel& kernel,
        const Config& config,
        const ProblemKey& problem_key) const;

    /**
     * Set the tuning key. The tuning key is part of the `ProblemKey` and is used for finding the
     * wisdom file. If the tuning key is not set, the kernel name will be used as tuning key.
     */
    KernelBuilder& tuning_key(std::string_view name) {
        tuning_key_ = Symbol(name);
        return *this;
    }

    /**
     * Set the problem size, which is part of the problem key and is used to compute the grid size.
     */
    KernelBuilder& problem_size(Expr<int64_t> x, Expr<int64_t> y = 1, Expr<int64_t> z = 1) {
        problem_size_ = {std::move(x), std::move(y), std::move(z)};
        return *this;
    }

    /**
     * Add a named feature to the problem key, which is used to select the wisdom configuration.
     */
    KernelBuilder& feature(std::string_view name, Expr<Primitive> value) {
        features_.emplace_back(Symbol(name), std::move(value));
        return *this;
    }

    /**
     * Set the block size for this kernel (i.e., number of threads per
     * thread block).
     */
    KernelBuilder& block_size(Expr<uint32_t> x, Expr<uint32_t> y = 1, Expr<uint32_t> z = 1) {
        block_size_ = std::array {std::move(x), std::move(y), std::move(z)};
        return *this;
    }
    /**
     * Set the grid size for this kernel (i.e., number of thread blocks
     * along each axis).
     */
    KernelBuilder& grid_size(Expr<uint32_t> x, Expr<uint32_t> y = 1, Expr<uint32_t> z = 1) {
        grid_size_ = std::array {std::move(x), std::move(y), std::move(z)};
        return *this;
    }
    /**
     * Set the grid size for this kernel (i.e., number of thread blocks
     * along each direction) by dividing the `problem_size` by the given
     * `divisors`. For example, if the problem size is `(100, 100)` and
     * the divisors are `(5, 15)` then the grid size will be `(20, 7)`.
     *
     * Note that it is possible to _either_ set the grid size explicitly (using `grid_size()`) or
     * to compute it using the divisors, but not both. If neither is set, the block size is used
     * as the divisors.
     */
    KernelBuilder& grid_divisor(Expr<uint32_t> x, Expr<uint32_t> y = 1, Expr<uint32_t> z = 1) {
        grid_divisors_ = std::array {std::move(x), std::move(y), std::move(z)};
        return *this;
    }

    /**
     * Set the amount of dynamic shared memory in bytes.
     */
    KernelBuilder& shared_memory(Expr<uint32_t> smem) {
        shared_memory_ = std::move(smem);
        return *this;
    }

    /**
     * Add a condition that is checked before each launch. Launching throws if it is false.
     */
    KernelBuilder& assertion(Expr<bool> e) {
        assertions_.push_back(std::move(e));
        return *this;
    }

    /**
     * Add a preprocessor variable definition with the provided `name` and
     * `value`.
     */
    KernelBuilder& define(std::string name, Expr<Symbol> value) {
        defines_.emplace_back(std::move(name), std::move(value));
        return *this;
    }

    /**
     * Define a preprocessor variable with the name and value of tunable parameter `param`.
     */
    KernelBuilder& define(const TunableParam& param) {
        return define(param.name().get(), param);
    }

    /**
     * Set the length (in elements) of the buffer that a pointer argument points to. The length
     * can depend on the kernel arguments, the problem size (e.g., `problem_size_x()`), and the
     * features. An exception is thrown at launch if the argument is a `void` pointer, since
     * `void` has no element size.
     */
    KernelBuilder& buffer_size(ArgumentBufferSize buffer) {
        buffer_sizes_.push_back(std::move(buffer));
        return *this;
    }

    /**
     * Set the length (in elements) of the buffer that pointer argument `arg` points to.
     */
    KernelBuilder& buffer_size(ArgumentExpr arg, Expr<size_t> length) {
        return buffer_size(ArgumentBufferSize {arg.index(), length.inner()});
    }

    /**
     * Set the buffer sizes of multiple pointer arguments at once (see `buffer_size()`).
     */
    template<typename... Ts>
    KernelBuilder& buffers(Ts... buffers) {
        (buffer_size(ArgumentBufferSize {buffers}), ...);
        return *this;
    }

    /**
     * Add a header file that is made available when compiling the kernel.
     */
    KernelBuilder& include_header(FileSource source) {
        headers_.push_back(std::move(source));
        return *this;
    }

    /**
     * Add a fixed template argument to the kernel.
     */
    KernelBuilder& template_arg(const TemplateArg& arg) {
        template_args_.emplace_back(arg);
        return *this;
    }

    /**
     * Add a template argument that is evaluated for each configuration.
     */
    KernelBuilder& template_arg(Expr<Primitive> arg) {
        template_args_.emplace_back(std::move(arg));
        return *this;
    }

    /**
     * Add a template argument, either as a fixed `TemplateArg` or as an expression.
     */
    template<typename T>
    KernelBuilder& template_arg(T arg) {
        if constexpr (std::is_convertible<T, TemplateArg>::value) {
            return template_arg(TemplateArg(std::move(arg)));
        } else {
            return template_arg(Expr<Primitive>(std::move(arg)));
        }
    }

    /**
     * Add multiple template arguments at once (see `template_arg()`).
     */
    template<typename... Ts>
    KernelBuilder& template_args(const Ts&... args) {
        (template_arg(args), ...);
        return *this;
    }

    /**
     * Add a flag that is passed to the compiler.
     */
    KernelBuilder& compiler_flag(Expr<Symbol> option) {
        compiler_flags_.push_back(std::move(option));
        return *this;
    }

    /**
     * Add multiple compiler flags at once (see `compiler_flag()`).
     */
    template<typename... Ts>
    KernelBuilder& compiler_flags(Ts&&... options) {
        (compiler_flag(std::forward<Ts>(options)), ...);
        return *this;
    }

    /**
     * Return the block size set by `block_size()`, or else derive it from tunable parameters
     * with names such as `block_size_x`. Throws if no block size can be found.
     */
    std::array<Expr<uint32_t>, 3> resolve_block_size() const;

    /**
     * Return the grid size set by `grid_size()`, or else compute it by dividing the problem size
     * by the grid divisors (or the block size if no divisors are set).
     */
    std::array<Expr<uint32_t>, 3> resolve_grid_size() const;

  private:
    friend struct KernelBuilderSerializerHack;

    std::string kernel_name_;
    FileSource kernel_source_;
    Symbol tuning_key_;
    std::array<Expr<int64_t>, MAX_PROBLEM_DIMS> problem_size_ = {1, 1, 1};
    std::vector<std::pair<Symbol, Expr<Primitive>>> features_;

    std::vector<Expr<bool>> assertions_;
    std::vector<std::pair<std::string, Expr<Symbol>>> defines_;
    std::vector<std::variant<Expr<Primitive>, TemplateArg>> template_args_;
    std::vector<FileSource> headers_;
    std::vector<ArgumentBufferSize> buffer_sizes_;
    std::vector<Expr<Symbol>> compiler_flags_;

    std::optional<std::array<Expr<uint32_t>, 3>> block_size_;
    std::optional<std::array<Expr<uint32_t>, 3>> grid_size_;
    std::optional<std::array<Expr<uint32_t>, 3>> grid_divisors_;
    Expr<uint32_t> shared_memory_;
};

/** @} */

}  // namespace kernel_launcher

#endif
