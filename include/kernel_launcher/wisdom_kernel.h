#ifndef KERNEL_LAUNCHER_WISDOM_KERNEL_H
#define KERNEL_LAUNCHER_WISDOM_KERNEL_H

#include <cstring>

#include "kernel_launcher/kernel.h"
#include "kernel_launcher/wisdom.h"

namespace kernel_launcher {

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

    TunableValue to_value() const;
    TunableValue to_value_or_empty() const;
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
    static KernelArg convert(KernelArg arg) {
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

#endif  //KERNEL_LAUNCHER_WISDOM_KERNEL_H
