#ifndef KERNEL_LAUNCHER_KERNEL_H
#define KERNEL_LAUNCHER_KERNEL_H

#include <atomic>
#include <cstddef>
#include <memory>
#include <vector>

#include "kernel_launcher/compiler.hpp"
#include "kernel_launcher/config.hpp"
#include "kernel_launcher/gpu_utils.hpp"
#include "kernel_launcher/kernel_arg.hpp"
#include "kernel_launcher/kernel_builder.hpp"
#include "kernel_launcher/problem_key.hpp"
#include "kernel_launcher/wisdom.hpp"

namespace kernel_launcher {

/**
 * @addtogroup kernel
 * @{
 */

/**
 * A kernel that has been compiled for one specific problem and configuration. Copies of an
 * instance are cheap and share the same compiled kernel.
 *
 * The arguments of a launch are passed as untyped `KernelArg` objects. See `KernelInstance` for
 * a wrapper that checks the argument types at compile time.
 */
class RawKernelInstance {
  public:
    /**
     * Constructs an instance from an already compiled kernel. Use `compile_kernel_raw` or
     * `Kernel::compile_for` to compile a kernel instead.
     */
    RawKernelInstance(
        std::shared_ptr<GPUKernel> kernel,
        ProblemKey problem,
        Config config,
        LaunchConfig launch_config);

    /**
     * Captures a launch of this kernel to `wisdom`, after first skipping `skip_launches`
     * launches. Replaces any capture that is still pending. The capture is shared by all copies
     * of this instance.
     *
     * The `builder` should be the builder that this instance was compiled from. Launches with
     * arguments that do not match the problem of this instance are neither captured nor counted
     * as skipped.
     */
    void capture_next_launch(
        std::shared_ptr<WisdomProvider> wisdom,
        KernelBuilder builder,
        size_t skip_launches = 0) const;

    /**
     * Launches this kernel on the given stream. The `args` must match the parameter types that
     * this kernel was compiled with. This is not checked.
     */
    void launch_args(GPUStream stream, const std::vector<KernelArg>& args) const;

    /**
     * Returns the compiled kernel.
     */
    const GPUKernel& kernel() const {
        return *state_->kernel;
    }

    /**
     * Returns the problem that this kernel was compiled for.
     */
    const ProblemKey& problem() const {
        return state_->problem;
    }

    /**
     * Returns the configuration that this kernel was compiled with.
     */
    const Config& config() const {
        return state_->config;
    }

    /**
     * Returns the grid size, block size, and other properties used to launch this kernel.
     */
    const LaunchConfig& launch_config() const {
        return state_->launch_config;
    }

  private:
    struct PendingCapture;

    struct State {
        State(
            std::shared_ptr<GPUKernel> kernel,
            ProblemKey problem,
            Config config,
            LaunchConfig launch_config);
        ~State();

        std::shared_ptr<GPUKernel> kernel;
        ProblemKey problem;
        Config config;
        LaunchConfig launch_config;

        // Owned. Null when no capture is pending.
        mutable std::atomic<PendingCapture*> pending_capture {nullptr};
    };

    std::shared_ptr<const State> state_;
};

/**
 * A `RawKernelInstance` where the types of the kernel arguments are given by `Ts`. Each argument
 * is converted using `into_kernel_arg` when the kernel is launched.
 */
template<typename... Ts>
class KernelInstance {
  public:
    explicit KernelInstance(RawKernelInstance inner) : inner_(std::move(inner)) {}

    /**
     * See `RawKernelInstance::capture_next_launch`.
     */
    void capture_next_launch(
        std::shared_ptr<WisdomProvider> wisdom,
        KernelBuilder builder,
        size_t skip_launches = 0) const {
        inner_.capture_next_launch(std::move(wisdom), std::move(builder), skip_launches);
    }

    /**
     * Launches this kernel on the given stream with the given arguments.
     */
    void launch(GPUStream stream, const Ts&... args) const {
        inner_.launch_args(stream, std::vector<KernelArg> {into_kernel_arg(args)...});
    }

    /**
     * Launches this kernel on the default stream with the given arguments.
     */
    void launch(const Ts&... args) const {
        launch(GPUStream {}, args...);
    }

    /**
     * Returns the compiled kernel.
     */
    const GPUKernel& kernel() const {
        return inner_.kernel();
    }

    /**
     * Returns the problem that this kernel was compiled for.
     */
    const ProblemKey& problem() const {
        return inner_.problem();
    }

    /**
     * Returns the configuration that this kernel was compiled with.
     */
    const Config& config() const {
        return inner_.config();
    }

    /**
     * Returns the grid size, block size, and other properties used to launch this kernel.
     */
    const LaunchConfig& launch_config() const {
        return inner_.launch_config();
    }

  private:
    RawKernelInstance inner_;
};

/**
 * Compiles the kernel described by `builder` for the given problem and configuration, where
 * `param_types` are the types of the kernel parameters. Throws an exception if `config` is not
 * valid for this problem and the device of `context`.
 */
RawKernelInstance compile_kernel_raw(
    const KernelBuilder& builder,
    const ProblemKey& problem,
    const Config& config,
    std::vector<TypeInfo> param_types,
    CompilerBase& compiler,
    GPUContextRef context = GPUContextRef::current());

/**
 * Compiles the kernel described by `builder` for the given problem and configuration, where
 * `Ts` are the types of the kernel parameters. See `compile_kernel_raw`.
 */
template<typename... Ts>
KernelInstance<Ts...> compile_kernel(
    const KernelBuilder& builder,
    const ProblemKey& problem,
    const Config& config,
    CompilerBase& compiler,
    GPUContextRef context = GPUContextRef::current()) {
    return KernelInstance<Ts...>(
        compile_kernel_raw(builder, problem, config, {TypeInfo::of<Ts>()...}, compiler, context));
}

/**
 * Compiles the kernel described by `builder` using a `RuntimeCompiler`.
 */
template<typename... Ts>
KernelInstance<Ts...> compile_kernel(
    const KernelBuilder& builder,
    const ProblemKey& problem,
    const Config& config,
    GPUContextRef context = GPUContextRef::current()) {
    RuntimeCompiler compiler;
    return compile_kernel<Ts...>(builder, problem, config, compiler, context);
}

/**
 * A kernel that is compiled on demand. On each launch, the problem is determined from the
 * arguments, and the kernel is compiled for that problem if this has not been done before. The
 * compiled instances are cached for the lifetime of this object.
 *
 * The configuration for a problem is loaded from the wisdom provider in `KernelSettings`. If no
 * wisdom provider is given, the default configuration is used.
 *
 * Copies of a `Kernel` share the same cache. A default-constructed `Kernel` is not initialized
 * and throws an exception when it is used (see `is_initialized`).
 */
class Kernel {
  public:
    /**
     * Constructs a kernel that is not initialized.
     */
    Kernel() = default;

    Kernel(KernelBuilder builder, KernelSettings settings = {});

    /**
     * Returns `true` if this kernel was constructed from a `KernelBuilder`, and `false` if it
     * was default-constructed.
     */
    bool is_initialized() const {
        return impl_ != nullptr;
    }

    /**
     * Returns the instance for the given problem and parameter types, compiling it if it is not
     * in the cache yet. The problem should come from `KernelBuilder::build_problem_key`.
     */
    RawKernelInstance compile_for(
        ProblemKey problem,
        std::vector<TypeInfo> param_types,
        GPUContextRef context = GPUContextRef::current());

    /**
     * Returns the instance for the problem given by `args`, compiling it if it is not in the
     * cache yet. Also attaches the buffer sizes to the pointer arguments in `args` (see
     * `KernelBuilder::build_problem_key`).
     */
    RawKernelInstance
    compile_args(std::vector<KernelArg>& args, GPUContextRef context = GPUContextRef::current());

    /**
     * Launches this kernel on the given stream, compiling it first if needed.
     */
    void launch_args(GPUStream stream, std::vector<KernelArg> args) {
        compile_args(args).launch_args(stream, args);
    }

    /**
     * Launches this kernel on the given stream, compiling it first if needed. Each argument is
     * converted using `into_kernel_arg`.
     */
    template<typename... Ts>
    void launch(GPUStream stream, const Ts&... args) {
        launch_args(stream, std::vector<KernelArg> {into_kernel_arg(args)...});
    }

    /**
     * Launches this kernel on the default stream, compiling it first if needed.
     */
    template<typename... Ts>
    void launch(const Ts&... args) {
        launch(GPUStream {}, args...);
    }

    /**
     * See `launch`.
     */
    template<typename... Ts>
    void operator()(GPUStream stream, const Ts&... args) {
        launch(stream, args...);
    }

    /**
     * See `launch`.
     */
    template<typename... Ts>
    void operator()(const Ts&... args) {
        launch(args...);
    }

    /**
     * Returns the instance for the problem given by `args` without launching it, compiling it
     * if it is not in the cache yet.
     */
    template<typename... Ts>
    KernelInstance<Ts...> compile(const Ts&... args) {
        std::vector<KernelArg> kargs {into_kernel_arg(args)...};
        return KernelInstance<Ts...>(compile_args(kargs));
    }

    /**
     * Returns the instance for the given problem, where `Ts` are the types of the kernel
     * parameters, compiling it if it is not in the cache yet.
     */
    template<typename... Ts>
    KernelInstance<Ts...>
    compile_for(ProblemKey problem, GPUContextRef context = GPUContextRef::current()) {
        return KernelInstance<Ts...>(
            compile_for(std::move(problem), {TypeInfo::of<Ts>()...}, context));
    }

  private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

/** @} */

}  // namespace kernel_launcher

#endif
