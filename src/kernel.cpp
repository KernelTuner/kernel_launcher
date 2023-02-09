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
        std::move(settings)});
}

void WisdomKernel::clear() {
    if (impl_) {
        std::lock_guard<std::mutex> guard(impl_->mutex_);
        impl_->compiled_ = false;
    }
}

void compile_impl(
    WisdomKernelImpl* impl,
    const std::string& tuning_key,
    ProblemSize problem_size,
    CudaContextHandle context,
    std::vector<TypeInfo> param_types,
    bool* should_capture = nullptr) {
    Timer config_timer;
    Config config = impl->settings_.load_config(
        tuning_key,
        impl->builder_,
        problem_size,
        context.device(),
        should_capture);
    config_timer.print_elapsed("load_config");

    // Assign result to temporary variable since compile may throw
    auto instance =
        impl->builder_.compile(config, param_types, impl->compiler_, context);

    // Compile was successful. Overwrite fields of impl
    impl->instance_ = std::move(instance);
    impl->param_types_ = std::move(param_types);
    impl->compiled_ = true;
}

void WisdomKernel::compile(
    ProblemSize problem_size,
    std::vector<TypeInfo> param_types,
    CudaContextHandle context) {
    if (!impl_) {
        throw std::runtime_error("WisdomKernel has not been initialized");
    }

    const std::string& tuning_key = impl_->builder_.tuning_key();

    std::lock_guard<std::mutex> guard(impl_->mutex_);
    compile_impl(
        impl_.get(),
        tuning_key,
        problem_size,
        context,
        std::move(param_types),
        nullptr);
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

void WisdomKernel::launch(cudaStream_t stream, std::vector<KernelArg> args) {
    Timer start_time;

    if (!impl_) {
        throw std::runtime_error("WisdomKernel has not been initialized");
    }

    ProblemSize problem_size = impl_->problem_processor_(args);

    std::lock_guard<std::mutex> guard(impl_->mutex_);
    if (!impl_->compiled_) {
        const std::string& tuning_key = impl_->builder_.tuning_key();

        std::vector<TypeInfo> param_types;
        for (const KernelArg& arg : args) {
            param_types.push_back(arg.type());
        }

        bool should_capture = false;
        Timer compile_time;
        compile_impl(
            impl_.get(),
            tuning_key,
            problem_size,
            CudaContextHandle::current(),
            param_types,
            &should_capture);
        compile_time.print_elapsed("compile_impl");

        if (should_capture) {
            std::vector<std::vector<uint8_t>> inputs;
            std::vector<std::vector<uint8_t>> outputs;

            KERNEL_LAUNCHER_CUDA_CHECK(cuStreamSynchronize(stream));
            KERNEL_LAUNCHER_CUDA_CHECK(cuCtxSynchronize());

            for (const KernelArg& arg : args) {
                inputs.push_back(arg.to_bytes());
            }

            impl_->instance_.launch(stream, problem_size, args);

            KERNEL_LAUNCHER_CUDA_CHECK(cuStreamSynchronize(stream));
            KERNEL_LAUNCHER_CUDA_CHECK(cuCtxSynchronize());

            for (const KernelArg& arg : args) {
                outputs.push_back(arg.to_bytes());
            }

            try {
                Timer capture_time;
                impl_->settings_.capture_kernel(
                    tuning_key,
                    impl_->builder_,
                    problem_size,
                    param_types,
                    inputs,
                    outputs);
                capture_time.print_elapsed("capture");
            } catch (const std::exception& err) {
                log_warning()
                    << "error ignored while writing tuning file for \""
                    << tuning_key << "\": " << err.what();
            }
            start_time.print_elapsed("launch");
        }
    }

    assert_types_equal(args, impl_->param_types_);
    impl_->instance_.launch(stream, problem_size, args);
    start_time.print_elapsed("wisdom_launch");
}

}  // namespace kernel_launcher