#ifndef KERNEL_LAUNCHER_WISDOM_H
#define KERNEL_LAUNCHER_WISDOM_H

#include <cuda_runtime_api.h>

#include <fstream>
#include <memory>
#include <string>

#include "kernel_launcher/export.h"
#include "kernel_launcher/kernel.h"

namespace kernel_launcher {

/// Returned by `load_best_config` to indicate the result:
/// - NotFound: Wisdom file was not found. Default config is returned.
/// - DeviceMismatch: File was found, but did not contain results for the
///                   the current device. Results for another device was chosen.
/// - ProblemSizeMismatch: File was found, but did not contain results for
///                        the current problem size. Results for another size
///                        was selected instead.
/// - Ok: Device and problem size was found exactly.
enum struct WisdomResult {
    NotFound = 0,
    DeviceMismatch = 1,
    ProblemSizeMismatch = 2,
    Ok = 3
};

Config load_best_config(
    const std::string& wisdom_dir,
    const std::string& tuning_key,
    const ConfigSpace& space,
    const std::string& device_name,
    CudaArch device_arch,
    ProblemSize problem_size,
    WisdomResult* result_out);

inline Config load_best_config(
    const std::string& wisdom_dir,
    const std::string& tuning_key,
    const ConfigSpace& space,
    ProblemSize problem_size,
    WisdomResult* result = nullptr) {
    CudaDevice device = CudaDevice::current();

    return load_best_config(
        wisdom_dir,
        tuning_key,
        space,
        device.name(),
        device.arch(),
        problem_size,
        result);
}

struct Oracle {
    virtual ~Oracle() = default;

    virtual Config load_config(
        const std::string& tuning_key,
        const ConfigSpace& space,
        ProblemSize problem_size,
        CudaDevice device,
        bool* should_capture_out) const = 0;

    virtual void capture_kernel(
        const std::string& tuning_key,
        const KernelBuilder& builder,
        ProblemSize problem_size,
        const std::vector<TypeInfo>& param_types,
        const std::vector<std::vector<uint8_t>>& inputs,
        const std::vector<std::vector<uint8_t>>& outputs) const = 0;
};

struct DefaultOracle: Oracle {
    static DefaultOracle from_env();

    DefaultOracle();
    DefaultOracle(
        std::string wisdom_dir,
        std::string capture_dir,
        std::vector<std::string> capture_patterns = {},
        bool force_capture = false);

    virtual ~DefaultOracle() = default;

    virtual Config load_config(
        const std::string& tuning_key,
        const ConfigSpace& space,
        ProblemSize problem_size,
        CudaDevice device,
        bool* should_capture_out) const override;

    virtual void capture_kernel(
        const std::string& tuning_key,
        const KernelBuilder& builder,
        ProblemSize problem_size,
        const std::vector<TypeInfo>& param_types,
        const std::vector<std::vector<uint8_t>>& inputs,
        const std::vector<std::vector<uint8_t>>& outputs) const override;

    virtual bool should_capture_kernel(
        const std::string& tuning_key,
        ProblemSize problem_size,
        WisdomResult result) const;

    bool should_capture_kernel(
        const std::string& tuning_key,
        ProblemSize problem_size) const {
        return should_capture_kernel(
            tuning_key,
            problem_size,
            WisdomResult::NotFound);
    }

    const std::string& wisdom_directory() const {
        return wisdom_dir_;
    }

    const std::string& capture_directory() const {
        return capture_dir_;
    }

    const std::vector<std::string>& capture_patterns() const {
        return capture_patterns_;
    }

    bool is_capture_forced() const {
        return force_capture_;
    }

  private:
    std::string wisdom_dir_;
    std::string capture_dir_;
    std::vector<std::string> capture_patterns_;
    bool force_capture_;
};

struct WisdomSettings {
    WisdomSettings();
    WisdomSettings(
        std::string wisdom_dir,
        std::string capture_dir,
        std::vector<std::string> capture_patterns = {},
        bool force_capture = false);
    WisdomSettings(std::shared_ptr<Oracle> oracle);

    template<typename T>
    WisdomSettings(std::shared_ptr<T> ptr) :
        WisdomSettings(std::shared_ptr<Oracle> {std::move(ptr)}) {}

    WisdomSettings(const WisdomSettings&) = default;

    Config load_config(
        const std::string& tuning_key,
        const ConfigSpace& space,
        ProblemSize problem_size,
        CudaDevice device,
        bool* should_capture_out = nullptr) const {
        return impl_->load_config(
            tuning_key,
            space,
            problem_size,
            device,
            should_capture_out);
    }

    void capture_kernel(
        const std::string& tuning_key,
        const KernelBuilder& builder,
        ProblemSize problem_size,
        const std::vector<TypeInfo>& param_types,
        const std::vector<std::vector<uint8_t>>& inputs,
        const std::vector<std::vector<uint8_t>>& outputs) const {
        return impl_->capture_kernel(
            tuning_key,
            builder,
            problem_size,
            param_types,
            inputs,
            outputs);
    }

  private:
    std::shared_ptr<Oracle> impl_;
};

void set_global_wisdom_directory(std::string);
void set_global_tuning_directory(std::string);
void add_global_capture_pattern(std::string);
WisdomSettings default_wisdom_settings();

struct KernelArg {
  private:
    KernelArg(TypeInfo type, void* data);
    KernelArg(TypeInfo type, void* ptr, size_t nelements);

  public:
    KernelArg();
    KernelArg(KernelArg&&) noexcept;
    KernelArg(const KernelArg&);
    ~KernelArg();

    template<typename T>
    static KernelArg for_scalar(T value) {
        static_assert(sizeof(T) == type_of<T>().size(), "internal error");
        return KernelArg(type_of<T>(), (void*)&value);
    }

    template<typename T>
    static KernelArg for_array(T* value, size_t nelements) {
        static_assert(sizeof(T) == type_of<T>().size(), "internal error");
        static_assert(sizeof(T*) == type_of<T*>().size(), "internal error");
        return KernelArg(type_of<T*>(), (void*)value, nelements);
    }

    template<typename T>
    T to() const {
        static_assert(
            std::is_trivially_copyable<T>::value,
            "type must be trivial");
        assert_type_matches(type_of<T>());

        T result = {};
        ::memcpy(&result, as_void_ptr(), sizeof(T));
        return result;
    }

    void assert_type_matches(TypeInfo t) const;
    bool is_scalar() const;
    bool is_array() const;
    TypeInfo type() const;
    std::vector<uint8_t> to_bytes() const;
    void* as_void_ptr() const;

  private:
    TypeInfo type_;
    bool scalar_;
    union {
        struct {
            void* ptr;
            size_t nelements;
        } array;
        std::array<uint8_t, 2 * sizeof(size_t)> small_scalar;
        void* large_scalar;
    } data_;
};

template<typename T, typename Enabled = void>
struct IntoKernelArg;

template<>
struct IntoKernelArg<KernelArg> {
    static KernelArg into(KernelArg arg) {
        return arg;
    }
};

template<typename T>
struct IntoKernelArg<
    T,
    typename std::enable_if<std::is_scalar<T>::value>::type> {
    static KernelArg convert(T value) {
        return KernelArg::for_scalar<T>(value);
    }
};

template<typename T>
struct IntoKernelArg<
    CudaSpan<T>,
    typename std::enable_if<std::is_trivially_copyable<T>::value>::type> {
    static KernelArg convert(CudaSpan<T> s) {
        return KernelArg::for_array<T>(s.data(), s.size());
    }
};

template<typename T>
KernelArg into_kernel_arg(T&& value) {
    return IntoKernelArg<typename std::decay<T>::type>::convert(value);
}

struct WisdomKernel;

struct WisdomKernelLaunch {
    WisdomKernelLaunch(
        cudaStream_t stream,
        ProblemSize problem_size,
        WisdomKernel& kernel) :
        stream_(stream),
        problem_size_(problem_size),
        kernel_ref_(kernel) {
        //
    }

    template<typename... Args>
    void launch(Args&&... args) const;

    template<typename... Args>
    void operator()(Args&&... args) const {
        return launch(std::forward<Args>(args)...);
    }

  private:
    cudaStream_t stream_;
    ProblemSize problem_size_;
    WisdomKernel& kernel_ref_;
};

struct WisdomKernelImpl;

struct WisdomKernel {
    using launch_type = WisdomKernelLaunch;
    WisdomKernel();
    WisdomKernel(WisdomKernel&&) noexcept;
    ~WisdomKernel();

    WisdomKernel(
        std::string tuning_key,
        KernelBuilder builder,
        Compiler compiler = default_compiler(),
        WisdomSettings settings = default_wisdom_settings()) :
        WisdomKernel() {
        initialize(
            std::move(tuning_key),
            std::move(builder),
            std::move(compiler),
            std::move(settings));
    }

    void initialize(
        std::string tuning_key,
        KernelBuilder builder,
        Compiler compiler = default_compiler(),
        WisdomSettings settings = default_wisdom_settings());

    void compile(
        ProblemSize problem_size,
        CudaDevice device,
        std::vector<TypeInfo> param_types);

    void launch(
        cudaStream_t stream,
        ProblemSize problem_size,
        const std::vector<KernelArg>& args);
    void clear();

    launch_type instantiate(cudaStream_t stream, ProblemSize problem_size) {
        return launch_type(stream, problem_size, *this);
    }

    launch_type operator()(cudaStream_t stream, ProblemSize problem_size) {
        return instantiate(stream, problem_size);
    }

    launch_type operator()(ProblemSize problem_size) {
        return instantiate(nullptr, problem_size);
    }

    launch_type operator()(
        cudaStream_t stream,
        uint32_t problem_x,
        uint32_t problem_y,
        uint32_t problem_z = 1) {
        return instantiate(
            stream,
            ProblemSize(problem_x, problem_y, problem_z));
    }

    launch_type
    operator()(uint32_t problem_x, uint32_t problem_y, uint32_t problem_z = 1) {
        return instantiate(
            nullptr,
            ProblemSize(problem_x, problem_y, problem_z));
    }

  private:
    std::unique_ptr<WisdomKernelImpl> impl_;
};

template<typename... Args>
void WisdomKernelLaunch::launch(Args&&... args) const {
    std::vector<KernelArg> kargs {into_kernel_arg(std::forward<Args>(args))...};
    kernel_ref_.launch(stream_, problem_size_, kargs);
}

}  // namespace kernel_launcher

#endif  //KERNEL_LAUNCHER_WISDOM_H
