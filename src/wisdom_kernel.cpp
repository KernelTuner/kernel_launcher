#include "kernel_launcher/wisdom_kernel.h"
#include "kernel_launcher/wisdom.h"
#include <mutex>

namespace kernel_launcher {

struct WisdomKernelImpl {
    std::mutex mutex_;
    bool compiled_;
    std::string tuning_key_;
    KernelBuilder builder_;
    KernelInstance instance_;
    Compiler compiler_;
    std::vector<TypeInfo> param_types_;
    WisdomSettings settings_;
};

WisdomKernel::WisdomKernel() = default;
WisdomKernel::WisdomKernel(WisdomKernel&&) noexcept = default;
WisdomKernel::~WisdomKernel() = default;

void WisdomKernel::initialize(
    std::string tuning_key,
    KernelBuilder builder,
    Compiler compiler,
    WisdomSettings settings) {
    impl_ = std::unique_ptr<WisdomKernelImpl>(new WisdomKernelImpl {
        {},
        false,
        std::move(tuning_key),
        std::move(builder),
        KernelInstance {},
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
    ProblemSize problem_size,
    CudaDevice device,
    std::vector<TypeInfo> param_types,
    bool* should_capture = nullptr) {
    Config config = impl->settings_.load_config(
        impl->tuning_key_,
        impl->builder_,
        problem_size,
        device,
        should_capture);

    // Assign result to temporary variable since compile may throw
    auto instance =
        impl->builder_.compile(config, param_types, impl->compiler_);

    // Compile was successful. Overwrite fields of impl
    impl->instance_ = std::move(instance);
    impl->param_types_ = std::move(param_types);
    impl->compiled_ = true;
}

void WisdomKernel::compile(
    ProblemSize problem_size,
    CudaDevice device,
    std::vector<TypeInfo> param_types) {
    if (!impl_) {
        throw std::runtime_error("WisdomKernel has not been initialized");
    }

    std::lock_guard<std::mutex> guard(impl_->mutex_);
    compile_impl(
        impl_.get(),
        problem_size,
        device,
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
        if (i != 0)
            msg += ", ";
        msg += params[i].name();
    }

    msg += "), but was called with argument types (";

    for (size_t i = 0; i < args.size(); i++) {
        if (i != 0)
            msg += ", ";
        msg += args[i].type().name();
    }

    msg += ")";
    throw std::runtime_error(msg);
}

void WisdomKernel::launch(
    cudaStream_t stream,
    ProblemSize problem_size,
    const std::vector<KernelArg>& args) {
    if (!impl_) {
        throw std::runtime_error("WisdomKernel has not been initialized");
    }

    std::vector<void*> ptrs;
    for (const KernelArg& arg : args) {
        ptrs.push_back(arg.as_void_ptr());
    }

    std::lock_guard<std::mutex> guard(impl_->mutex_);
    if (!impl_->compiled_) {
        std::vector<TypeInfo> param_types;
        for (const KernelArg& arg : args) {
            param_types.push_back(arg.type());
        }

        bool should_capture = false;
        compile_impl(
            impl_.get(),
            problem_size,
            CudaDevice::current(),
            param_types,
            &should_capture);

        if (should_capture) {
            std::vector<std::vector<uint8_t>> inputs;
            std::vector<std::vector<uint8_t>> outputs;

            KERNEL_LAUNCHER_CUDA_CHECK(cuStreamSynchronize(stream));
            KERNEL_LAUNCHER_CUDA_CHECK(cuCtxSynchronize());

            for (const KernelArg& arg : args) {
                inputs.push_back(arg.to_bytes());
            }

            impl_->instance_.launch(stream, problem_size, ptrs.data());

            KERNEL_LAUNCHER_CUDA_CHECK(cuStreamSynchronize(stream));
            KERNEL_LAUNCHER_CUDA_CHECK(cuCtxSynchronize());

            for (const KernelArg& arg : args) {
                outputs.push_back(arg.to_bytes());
            }

            try {
                impl_->settings_.capture_kernel(
                    impl_->tuning_key_,
                    impl_->builder_,
                    problem_size,
                    param_types,
                    inputs,
                    outputs);
            } catch (const std::exception& err) {
                log_warning()
                    << "error ignored while writing tuning file for \""
                    << impl_->tuning_key_ << "\": " << err.what();
            }

            return;
        }
    }

    assert_types_equal(args, impl_->param_types_);
    impl_->instance_.launch(stream, problem_size, ptrs.data());
}

static bool is_inline_scalar(TypeInfo type) {
    return type.size() <= sizeof(size_t) * 2;
}

KernelArg::KernelArg(TypeInfo type, void* data) {
    type_ = type.remove_const();
    scalar_ = true;

    if (is_inline_scalar(type_)) {
        ::memcpy(data_.small_scalar.data(), data, type.size());
    } else {
        data_.large_scalar = new char[type.size()];
        ::memcpy(data_.large_scalar, data, type.size());
    }
}

KernelArg::KernelArg(TypeInfo type, void* ptr, size_t nelements) {
    type_ = type.remove_const();
    scalar_ = false;
    data_.array.ptr = ptr;
    data_.array.nelements = nelements;
}

KernelArg::KernelArg(const KernelArg& that) : KernelArg() {
    type_ = that.type_;
    scalar_ = that.scalar_;

    if (that.is_array()) {
        data_.array = that.data_.array;
    } else if (is_inline_scalar(type_)) {
        data_.small_scalar = that.data_.small_scalar;
    } else {
        data_.large_scalar = new char[type_.size()];
        ::memcpy(data_.large_scalar, that.data_.large_scalar, type_.size());
    }
}

KernelArg::~KernelArg() {
    if (is_scalar() && !is_inline_scalar(type_)) {
        delete[](char*) data_.large_scalar;
    }
}

KernelArg::KernelArg(KernelArg&& that) noexcept : KernelArg() {
    std::swap(this->data_, that.data_);
    std::swap(this->type_, that.type_);
    std::swap(this->scalar_, that.scalar_);
}

void KernelArg::assert_type_matches(TypeInfo t) const {
    bool valid;

    if (t.is_pointer()) {
        // Matches if both are pointers and pointee type matches (possible
        // when adding a const modifier).
        valid = type_.is_pointer()
            && (t.remove_pointer() == type_.remove_pointer()
                || t.remove_pointer() == type_.remove_pointer().add_const());
    } else {
        valid = t.remove_const() == type_.remove_const();
    }

    if (!valid) {
        throw std::runtime_error(
            "cannot cast kernel argument of type `" + type_.name()
            + "` to type `" + t.name() + "`");
    }
}

bool KernelArg::is_array() const {
    return !scalar_;
}

bool KernelArg::is_scalar() const {
    return scalar_;
}

TypeInfo KernelArg::type() const {
    return type_;
}

std::vector<uint8_t> KernelArg::to_bytes() const {
    std::vector<uint8_t> result;

    if (is_array()) {
        result.resize(type_.remove_pointer().size() * data_.array.nelements);
        KERNEL_LAUNCHER_CUDA_CHECK(cuMemcpy(
            reinterpret_cast<CUdeviceptr>(result.data()),
            reinterpret_cast<CUdeviceptr>(data_.array.ptr),
            result.size()));
    } else {
        // If the type is a pointer, exporting it to bytes will return
        // the memory address of the pointer and not the data of the buffer it
        // points to This is likely a bug on the user side. Todo: find a better
        //  way f handling this error (maybe already in KernelArg ctor?).
        if (type_.is_pointer()) {
            throw std::runtime_error("a raw pointer type was provided as "
                "kernel argument (" + type_.name() + ") which cannot be "
                "exported since the corresponding buffer size is unknown");
        }

        result.resize(type_.size());

        if (is_inline_scalar(type_)) {
            ::memcpy(result.data(), data_.small_scalar.data(), type_.size());
        } else {
            ::memcpy(result.data(), data_.large_scalar, type_.size());
        }
    }

    return result;
}

KernelArg::KernelArg() : type_(type_of<int>()), scalar_(true) {}

void* KernelArg::as_void_ptr() const {
    if (is_array()) {
        return const_cast<void*>(static_cast<const void*>(&data_.array.ptr));
    } else if (is_inline_scalar(type_)) {
        return const_cast<void*>(
            static_cast<const void*>(data_.small_scalar.data()));
    } else {
        return data_.large_scalar;
    }
}
}