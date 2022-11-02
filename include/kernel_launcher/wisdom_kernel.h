#ifndef KERNEL_LAUNCHER_WISDOM_KERNEL_H
#define KERNEL_LAUNCHER_WISDOM_KERNEL_H

#include <cstring>

#include "kernel_launcher/kernel.h"
#include "kernel_launcher/wisdom.h"

namespace kernel_launcher {

struct WisdomKernel;
struct WisdomKernelImpl;

struct WisdomKernel {
    WisdomKernel();
    WisdomKernel(WisdomKernel&&) noexcept;
    ~WisdomKernel();

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

    void compile(
        ProblemSize problem_size,
        CudaDevice device,
        std::vector<TypeInfo> param_types);

    void launch(cudaStream_t stream, const std::vector<KernelArg>& args);

    void launch(cudaStream_t stream, std::vector<KernelArg>& args) {
        launch(stream, (const std::vector<KernelArg>&)args);
    }

    void clear();

    template<typename... Args>
    void launch(cudaStream_t stream, Args&&... args) {
        return launch(stream, {into_kernel_arg(std::forward<Args>(args))...});
    }

    template<typename... Args>
    void launch(nullptr_t, Args&&... args) {
        return launch(
            cudaStream_t(nullptr),
            {into_kernel_arg(std::forward<Args>(args))...});
    }

    template<typename... Args>
    void launch(Args&&... args) {
        return launch(
            cudaStream_t(nullptr),
            {into_kernel_arg(std::forward<Args>(args))...});
    }

    template<typename... Args>
    void operator()(Args&&... args) {
        return launch(std::forward<Args>(args)...);
    }

  private:
    std::unique_ptr<WisdomKernelImpl> impl_;
};

}  // namespace kernel_launcher

#endif  //KERNEL_LAUNCHER_WISDOM_KERNEL_H
