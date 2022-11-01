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
struct WisdomKernelImpl;

struct ArgExpr: BaseExpr {
    constexpr ArgExpr(uint8_t i) : index_(i) {};
    std::string to_string() const override;
    TunableValue eval(const Eval& eval) const override;
    Expr resolve(const Eval& eval) const override;

  private:
    uint8_t index_;
};

extern ArgExpr arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8;

inline ArgExpr arg(uint8_t i) {
    return i;
}

using ProblemExtractor =
    std::function<ProblemSize(const std::vector<KernelArg>&)>;

struct WisdomKernelBuilder: KernelBuilder {
    friend WisdomKernel;

    WisdomKernelBuilder(std::string kernel_name, KernelSource source);

    WisdomKernelBuilder& tuning_key(std::string);

    WisdomKernelBuilder& problem_size(
        TypedExpr<uint32_t> x,
        TypedExpr<uint32_t> y = 1,
        TypedExpr<uint32_t> z = 1);
    WisdomKernelBuilder& problem_size(ProblemSize p);
    WisdomKernelBuilder& problem_size(ProblemExtractor);

  private:
    std::string tuning_key_;
    ProblemExtractor problem_extractor_;
};

struct WisdomKernel {
    WisdomKernel();
    WisdomKernel(WisdomKernel&&) noexcept;
    ~WisdomKernel();

    WisdomKernel(
        WisdomKernelBuilder builder,
        Compiler compiler = default_compiler(),
        WisdomSettings settings = default_wisdom_settings()) :
        WisdomKernel() {
        initialize(
            std::move(builder),
            std::move(compiler),
            std::move(settings));
    }

    void initialize(
        WisdomKernelBuilder builder,
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
