#include "kernel_launcher/kernel.h"

#include <algorithm>
#include <mutex>

#include "kernel_launcher/wisdom.h"

namespace kernel_launcher {

struct WisdomKernelImpl {
    std::mutex mutex_;
    bool compiled_;
    KernelBuilder builder_;
    KernelInstance instance_;
    ProblemProcessor problem_processor_;
    Compiler compiler_;
    std::vector<TypeInfo> param_types_;
    WisdomSettings settings_;
    bool capture_required_;
    int capture_skip_;
};

WisdomKernel::WisdomKernel() = default;
WisdomKernel::WisdomKernel(WisdomKernel&&) noexcept = default;
WisdomKernel::~WisdomKernel() = default;

void WisdomKernel::initialize(
    KernelBuilder builder,
    Compiler compiler,
    WisdomSettings settings) {
    auto problem_processor = builder.problem_processor();
    impl_ = std::unique_ptr<WisdomKernelImpl>(new WisdomKernelImpl {
        {},
        false,
        std::move(builder),
        KernelInstance {},
        std::move(problem_processor),
        std::move(compiler),
        std::vector<TypeInfo> {},
        std::move(settings),
        false,
        0});
}

void WisdomKernel::clear() {
    if (impl_) {
        std::lock_guard<std::mutex> guard(impl_->mutex_);
        impl_->compiled_ = false;
    }
}

void compile_impl(
    WisdomKernelImpl* impl,
    ProblemSize problem_size,
    CudaContextHandle context,
    std::vector<TypeInfo> param_types) {
    const std::string& tuning_key = impl->builder_.tuning_key();
    int capture_skip = 0;

    Config config = impl->settings_.load_config(
        tuning_key,
        impl->builder_,
        problem_size,
        context.device(),
        &capture_skip);

    // Assign result to temporary variable since compile may throw
    auto instance =
        impl->builder_.compile(config, param_types, impl->compiler_, context);

    // Compile was successful. Overwrite fields of impl
    impl->instance_ = std::move(instance);
    impl->param_types_ = std::move(param_types);
    impl->compiled_ = true;
    impl->capture_required_ = capture_skip >= 0;
    impl->capture_skip_ = capture_skip;
}

void WisdomKernel::compile(
    ProblemSize problem_size,
    std::vector<TypeInfo> param_types,
    CudaContextHandle context) {
    if (!impl_) {
        throw std::runtime_error("WisdomKernel has not been initialized");
    }

    std::lock_guard<std::mutex> guard(impl_->mutex_);
    compile_impl(impl_.get(), problem_size, context, std::move(param_types));
}

void WisdomKernel::compile(
    std::vector<KernelArg> args,
    CudaContextHandle context) {
    if (!impl_) {
        throw std::runtime_error("WisdomKernel has not been initialized");
    }

    std::lock_guard<std::mutex> guard(impl_->mutex_);
    ProblemSize problem_size = impl_->problem_processor_(args);

    std::vector<TypeInfo> param_types;
    for (const KernelArg& arg : args) {
        param_types.push_back(arg.type());
    }

    compile_impl(impl_.get(), problem_size, context, std::move(param_types));
}

static void assert_types_equal(
    const std::vector<KernelArg>& args,
    const std::vector<TypeInfo>& params) {
    bool is_equal = true;
    if (args.size() == params.size()) {
        for (size_t i = 0; i < args.size(); i++) {
            if (args[i].type().remove_const() != params[i].remove_const()) {
                is_equal = false;
            }
        }
    } else {
        is_equal = false;
    }

    if (is_equal) {
        return;
    }

    std::string msg =
        "invalid argument types: kernel compiled for parameter types (";

    for (size_t i = 0; i < params.size(); i++) {
        if (i != 0) {
            msg += ", ";
        }
        msg += params[i].name();
    }

    msg += "), but was called with argument types (";

    for (size_t i = 0; i < args.size(); i++) {
        if (i != 0) {
            msg += ", ";
        }
        msg += args[i].type().name();
    }

    msg += ")";
    throw std::runtime_error(msg);
}

static void launch_captured_impl(
    WisdomKernelImpl* impl_,
    cudaStream_t stream,
    ProblemSize problem_size,
    const std::vector<KernelArg>& args) {
    const std::string& tuning_key = impl_->builder_.tuning_key();
    std::vector<std::vector<uint8_t>> inputs;
    std::vector<std::vector<uint8_t>> outputs;

    KERNEL_LAUNCHER_CUDA_CHECK(cuStreamSynchronize(stream));
    KERNEL_LAUNCHER_CUDA_CHECK(cuCtxSynchronize());

    for (const KernelArg& arg : args) {
        if (arg.is_array()) {
            inputs.emplace_back(arg.copy_array());
        }
    }

    impl_->instance_.launch(stream, problem_size, args);

    KERNEL_LAUNCHER_CUDA_CHECK(cuStreamSynchronize(stream));
    KERNEL_LAUNCHER_CUDA_CHECK(cuCtxSynchronize());

    for (const KernelArg& arg : args) {
        if (arg.is_array()) {
            outputs.emplace_back(arg.copy_array());
        }
    }

    try {
        impl_->settings_.capture_kernel(
            tuning_key,
            impl_->builder_,
            problem_size,
            args,
            inputs,
            outputs);
    } catch (const std::exception& err) {
        log_warning() << "error ignored while writing tuning file for \""
                      << tuning_key << "\": " << err.what();
    }
}

void WisdomKernel::capture_next_launch(int skip_launches) {
    if (!impl_) {
        throw std::runtime_error("WisdomKernel has not been initialized");
    }

    std::lock_guard<std::mutex> guard(impl_->mutex_);
    impl_->capture_required_ = true;
    impl_->capture_skip_ = skip_launches;
}

void WisdomKernel::launch_args(
    cudaStream_t stream,
    std::vector<KernelArg> args) {
    if (!impl_) {
        throw std::runtime_error("WisdomKernel has not been initialized");
    }

    std::lock_guard<std::mutex> guard(impl_->mutex_);
    ProblemSize problem_size = impl_->problem_processor_(args);

    if (!impl_->compiled_) {
        std::vector<TypeInfo> param_types;
        for (const KernelArg& arg : args) {
            param_types.push_back(arg.type());
        }

        compile_impl(
            impl_.get(),
            problem_size,
            CudaContextHandle::current(),
            param_types);
    }

    assert_types_equal(args, impl_->param_types_);

    bool should_capture = false;
    if (impl_->capture_required_) {
        if (impl_->capture_skip_ <= 0) {
            impl_->capture_required_ = false;
            should_capture = true;
        } else {
            impl_->capture_skip_ -= 1;
        }
    }

    if (should_capture) {
        launch_captured_impl(impl_.get(), stream, problem_size, args);
    } else {
        impl_->instance_.launch(stream, problem_size, args);
    }
}

}  // namespace kernel_launcher