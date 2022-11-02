#ifndef KERNEL_LAUNCHER_KERNEL_H
#define KERNEL_LAUNCHER_KERNEL_H

#include <cuda.h>

#include <cstring>
#include <iostream>
#include <utility>

#include "kernel_launcher/compiler.h"
#include "kernel_launcher/config.h"

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

using ProblemExtractor =
    std::function<ProblemSize(const std::vector<KernelArg>&)>;

struct KernelInstance {
    KernelInstance() = default;

    KernelInstance(
        CudaModule module,
        ProblemExtractor problem,
        std::array<TypedExpr<uint32_t>, 3> block_size,
        std::array<TypedExpr<uint32_t>, 3> grid_size,
        TypedExpr<uint32_t> shared_mem) :
        module_(std::move(module)),
        problem_extractor_(std::move(problem)),
        block_size_(std::move(block_size)),
        grid_size_(std::move(grid_size)),
        shared_mem_(std::move(shared_mem)) {}

    void launch(
        cudaStream_t stream,
        const std::vector<KernelArg>& args,
        const Eval& eval) const;

    void launch(cudaStream_t stream, const std::vector<KernelArg>& args) const;

  private:
    CudaModule module_;
    ProblemExtractor problem_extractor_;
    std::array<TypedExpr<uint32_t>, 3> block_size_ = {1, 1, 1};
    std::array<TypedExpr<uint32_t>, 3> grid_size_ = {0, 0, 0};
    TypedExpr<uint32_t> shared_mem_ = 0;
};

struct KernelBuilderSerializerHack;
struct WisdomKernel;
template<typename... Args>
struct Kernel;

struct KernelBuilder: ConfigSpace {
    friend ::kernel_launcher::KernelBuilderSerializerHack;
    friend ::kernel_launcher::WisdomKernel;

    template<typename... Args>
    friend struct ::kernel_launcher::Kernel;

    KernelBuilder(
        std::string kernel_name,
        KernelSource kernel_source,
        ConfigSpace space = {});

    const std::string& kernel_name() const {
        return kernel_name_;
    }

    KernelBuilder& block_size(
        TypedExpr<uint32_t> x,
        TypedExpr<uint32_t> y = 1,
        TypedExpr<uint32_t> z = 1);

    KernelBuilder& grid_size(
        TypedExpr<uint32_t> x,
        TypedExpr<uint32_t> y = 1,
        TypedExpr<uint32_t> z = 1);

    KernelBuilder& grid_divisors(
        TypedExpr<uint32_t> x,
        TypedExpr<uint32_t> y = 1,
        TypedExpr<uint32_t> z = 1);

    KernelBuilder& shared_memory(TypedExpr<uint32_t> smem);
    KernelBuilder& template_arg(TypedExpr<TemplateArg> arg);
    KernelBuilder& assertion(TypedExpr<bool> e);
    KernelBuilder& define(std::string name, TypedExpr<std::string> value);
    KernelBuilder& compiler_flag(TypedExpr<std::string> opt);
    KernelBuilder& include_header(KernelSource source);
    KernelBuilder& tuning_key(std::string);

    KernelBuilder& problem_size(
        TypedExpr<uint32_t> x,
        TypedExpr<uint32_t> y = 1,
        TypedExpr<uint32_t> z = 1);
    KernelBuilder& problem_size(ProblemSize p);
    KernelBuilder& problem_size(ProblemExtractor);

    template<typename T, typename... Ts>
    KernelBuilder& template_args(T&& first, Ts&&... rest) {
        template_arg(std::forward<T>(first));
        return template_args(std::forward<Ts>(rest)...);
    }

    KernelBuilder& template_args() {
        return *this;
    }

    template<typename T, typename... Ts>
    KernelBuilder& compiler_flags(T&& first, Ts&&... rest) {
        compiler_flag(std::forward<T>(first));
        return compiler_flags(std::forward<Ts>(rest)...);
    }

    KernelBuilder& compiler_flags() {
        return *this;
    }

    KernelBuilder& define(ParamExpr p) {
        return define(p.parameter().name(), p);
    }

    std::array<TypedExpr<uint32_t>, 3> tune_block_size(
        std::vector<uint32_t> xs,
        std::vector<uint32_t> ys = {1u},
        std::vector<uint32_t> zs = {1u}) {
        block_size(
            tune("BLOCK_SIZE_X", xs),
            tune("BLOCK_SIZE_Y", ys),
            tune("BLOCK_SIZE_Z", zs));
        return block_size_;
    }

    template<typename T>
    TypedExpr<T> tune_define(std::string name, std::vector<T> values) {
        TypedExpr<T> param = this->tune(name, values);
        define(std::move(name), param);
        return param;
    }

    template<typename T>
    TypedExpr<T>
    tune_define(std::string name, std::initializer_list<T> values) {
        return tune_define(name, std::vector<T>(values));
    }

    KernelDef
    build(const Config& config, const std::vector<TypeInfo>& param_types) const;

    KernelInstance compile(
        const Config& config,
        const std::vector<TypeInfo>& param_types,
        const ICompiler& compiler = default_compiler(),
        CudaContextHandle ctx = CudaContextHandle::current()) const;

  private:
    std::string kernel_name_;
    KernelSource kernel_source_;
    std::string tuning_key_;
    ProblemExtractor problem_extractor_;
    std::vector<KernelSource> preheaders_;
    std::array<TypedExpr<uint32_t>, 3> block_size_ = {1u, 1u, 1u};
    std::array<TypedExpr<uint32_t>, 3> grid_size_ = {1u, 1u, 1u};
    bool grid_set_ = false;
    TypedExpr<uint32_t> shared_mem_ = {0u};
    std::vector<TypedExpr<TemplateArg>> template_args_ {};
    std::vector<TypedExpr<std::string>> compile_flags_ {};
    std::vector<TypedExpr<bool>> assertions_ {};
    std::unordered_map<std::string, TypedExpr<std::string>> defines_ {};
};

template<typename... Args>
struct Kernel {
    Kernel() = default;
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

    void launch(cudaStream_t stream, Args... args) {
        std::vector<KernelArg> kargs = {KernelArg::for_scalar<Args>(args)...};
        instance_.launch(stream, kargs);
    }

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

}  // namespace kernel_launcher

#endif  //KERNEL_LAUNCHER_KERNEL_H
