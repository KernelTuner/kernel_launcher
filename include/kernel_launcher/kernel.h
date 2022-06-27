#ifndef KERNEL_LAUNCHER_KERNEL_H
#define KERNEL_LAUNCHER_KERNEL_H

#include "kernel_launcher/config.h"

namespace kernel_launcher {

struct KernelSource {
    KernelSource(std::string filename) :
        filename_(std::move(filename)),
        has_content_(false) {
        //
    }

    KernelSource(std::string filename, std::string content) :
        filename_(std::move(filename)),
        content_(std::move(content)),
        has_content_(true) {
        //
    }

    std::string file_name() const {
        return filename_;
    }

    std::string read() const {
        if (has_content_) {
            return content_;
        }

        std::ifstream t(filename_);
        return {
            (std::istreambuf_iterator<char>(t)),
            std::istreambuf_iterator<char>()};
    }

  private:
    std::string filename_;
    std::string content_;
    bool has_content_;
};

struct KernelBuilder: ConfigSpace {
    KernelBuilder(std::string kernel_name, KernelSource kernel_source): kernel_name_(kernel_name), kernel_source_(kernel_source) {}

    const std::string& kernel_name() const {
        return kernel_name_;
    }

    KernelBuilder& block_size(
        TypedExpr<uint32_t> x,
        TypedExpr<uint32_t> y = 1,
        TypedExpr<uint32_t> z = 1) {
        block_size_[0] = std::move(x);
        block_size_[1] = std::move(y);
        block_size_[2] = std::move(z);
        return *this;
    }

    KernelBuilder& grid_divisors(
        TypedExpr<uint32_t> x,
        TypedExpr<uint32_t> y = 1,
        TypedExpr<uint32_t> z = 1) {
        grid_divisors_[0] = std::move(x);
        grid_divisors_[1] = std::move(y);
        grid_divisors_[2] = std::move(z);
        return *this;
    }

    KernelBuilder& shared_memory(TypedExpr<uint32_t> smem) {
        shared_mem_ = smem;
        return *this;
    }

    KernelBuilder& template_arg(TypedExpr<TemplateArg> arg) {
        template_args_.push_back(std::move(arg));
        return *this;
    }

    KernelBuilder& template_arg(TemplateArg arg) {
        return template_arg(TypedExpr<TemplateArg>(arg));
    }

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

    KernelBuilder& compiler_flag(TypedExpr<std::string> opt) {
        compile_flags_.emplace_back(std::move(opt));
        return *this;
    }

    KernelBuilder& define(std::string name, TypedExpr<std::string> value) {
        defines_[name] = std::move(value);
        return *this;
    }

    KernelBuilder& define(const ParamExpr& p) {
        return define(p.parameter().name(), p);
    }

    void assertion(TypedExpr<bool> e) {
        assertions_.push_back(e);
    }

    std::array<TypedExpr<uint32_t>, 3> tune_block_size(
        std::vector<uint32_t> xs,
        std::vector<uint32_t> ys = {1u},
        std::vector<uint32_t> zs = {1u}) {
        block_size(
            this->tune("BLOCK_SIZE_X", xs),
            this->tune("BLOCK_SIZE_Y", ys),
            this->tune("BLOCK_SIZE_Z", zs));
        return block_size_;
    }

    template<typename T>
    TypedExpr<T> tune_define(std::string name, std::vector<T> values) {
        TypedExpr<T> param = this->tune(name, values);
        define(std::move(name), param);
        return param;
    }

  private:
    std::string kernel_name_;
    KernelSource kernel_source_;
    std::array<TypedExpr<uint32_t>, 3> block_size_ = {1u, 1u, 1u};
    std::array<TypedExpr<uint32_t>, 3> grid_divisors_ = {1u, 1u, 1u};
    TypedExpr<uint32_t> shared_mem_ = {0u};
    std::vector<TypedExpr<TemplateArg>> template_args_ {};
    std::vector<TypedExpr<std::string>> compile_flags_ {};
    std::vector<TypedExpr<bool>> assertions_ {};
    std::unordered_map<std::string, TypedExpr<std::string>> defines_ {};
};

struct RawKernelInstance {
};

}  // namespace kernel_launcher

#endif  //KERNEL_LAUNCHER_KERNEL_H
