#ifndef KERNEL_LAUNCHER_KERNEL_H
#define KERNEL_LAUNCHER_KERNEL_H

#include <cstring>

#include "kernel_launcher/arg.h"
#include "kernel_launcher/wisdom.h"

namespace kernel_launcher {

/**
 * An instance of a CUDA kernel with fixed argument types.
 *
 * @tparam Args Types of the kernel arguments.
 */
template<typename... Args>
struct Kernel {
    Kernel() = default;

    /**
     * Compile a CUDA kernel using the given `KernelBuilder` with the given
     * `Config` and store the result in the current `Kernel` object.
     */
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

    /**
     * Delete this kernel and reset its contents.
     */
    void clear() {
        instance_ = {};
    }

    /**
     * Launch this kernel onto the given stream with the given arguments.
     */
    void launch(cudaStream_t stream, Args... args) {
        std::vector<KernelArg> kargs = {KernelArg::for_scalar<Args>(args)...};
        instance_.launch(stream, kargs);
    }

    /**
     * Launch this kernel with the given arguments on the default CUDA stream.
     */
    void launch(Args... args) {
        return launch(cudaStream_t(nullptr), std::move(args)...);
    }

    void operator()(cudaStream_t stream, Args... args) {
        return launch(stream, std::move(args)...);
    }

    void operator()(Args... args) {
        return launch(cudaStream_t(nullptr), std::move(args)...);
    }

    KernelInstance instance_;
};

struct WisdomKernelImpl;

/**
 * An instance of a CUDA kernel where the configuration is loaded from a wisdom
 * file using the provided `WisdomSettings`.
 */
struct WisdomKernel {
    WisdomKernel();
    WisdomKernel(WisdomKernel&&) noexcept;
    ~WisdomKernel();

    /**
     * Initialize this kernel. This will not directly compile the kernel.
     * Compilation of the kernel happens on the first call to `launch` or
     * to `compile`.
     *
     * @param builder The kernel's specifications.
     * @param compiler The compiler to use for compilation.
     * @param settings The wisdom settings to use load configurations.
     */
    WisdomKernel(
        KernelBuilder builder,
        Compiler compiler = default_compiler(),
        WisdomSettings settings = default_wisdom_settings()) :
        WisdomKernel() {
        initialize(
            std::move(builder),
            std::move(compiler),
            std::move(settings));
    }

    void initialize(
        KernelBuilder builder,
        Compiler compiler = default_compiler(),
        WisdomSettings settings = default_wisdom_settings());

    /**
     * Explicitly compile this kernel for the given problem size and parameter
     * types.
     *
     * @param problem_size Use to find the configuration from the wisdom file.
     * @param param_types Types of kernel parameters.
     * @param context CUDA context to use for compilation.
     */
    void compile(
        ProblemSize problem_size,
        std::vector<TypeInfo> param_types,
        CudaContextHandle context = CudaContextHandle::current());

    /**
     * Delete this kernel and reset its contents.
     */
    void clear();

    void launch(cudaStream_t stream, const std::vector<KernelArg>& args);

    void launch(cudaStream_t stream, std::vector<KernelArg>& args) {
        launch(stream, (const std::vector<KernelArg>&)args);
    }

    /**
     * Launch this kernel onto the given stream with the given arguments.
     * This will compile the kernel if it has not yet been compiled.
     */
    template<typename... Args>
    void launch(cudaStream_t stream, Args&&... args) {
        return launch(stream, {into_kernel_arg(std::forward<Args>(args))...});
    }

    /**
     * Launch this kernel with the given arguments on the default CUDA stream.
     * This will compile the kernel if it has not yet been compiled.
     */
    template<typename... Args>
    void launch(Args&&... args) {
        return launch(
            cudaStream_t(nullptr),
            {into_kernel_arg(std::forward<Args>(args))...});
    }

    template<typename... Args>
    void operator()(cudaStream_t stream, Args&&... args) {
        return launch(stream, std::forward<Args>(args)...);
    }

    template<typename... Args>
    void operator()(Args&&... args) {
        return launch(std::forward<Args>(args)...);
    }

  private:
    std::unique_ptr<WisdomKernelImpl> impl_;
};

}  // namespace kernel_launcher

#endif  //KERNEL_LAUNCHER_KERNEL_H
