#ifndef KERNEL_LAUNCHER_CACHE_H
#define KERNEL_LAUNCHER_CACHE_H

#include <memory>
#include <mutex>
#include <unordered_map>

#include "kernel_launcher/wisdom_kernel.h"

namespace kernel_launcher {

struct IKernelDescriptor {
    virtual ~IKernelDescriptor() = default;
    virtual std::string tuning_key() const = 0;
    virtual KernelBuilder build() const = 0;
    virtual bool equals(const IKernelDescriptor& that) const = 0;
    virtual hash_t hash() const {
        return 0;
    }
};

struct KernelDescriptor {
    KernelDescriptor(KernelDescriptor&&) noexcept = default;
    KernelDescriptor(KernelDescriptor&) noexcept = default;
    KernelDescriptor(const KernelDescriptor&) = default;

    template<typename D>
    KernelDescriptor(D&& descriptor) {
        using T = typename std::decay<D>::type;
        descriptor_type_ = type_of<T>();
        descriptor_ = std::make_shared<T>(std::forward<D>(descriptor));
        hash_ = hash_fields(descriptor_type_, descriptor_->hash());
    }

    const IKernelDescriptor& get() const {
        return *descriptor_;
    }

    hash_t hash() const {
        return hash_;
    }

    bool operator==(const KernelDescriptor& that) const {
        return that.hash_ == hash_ && that.descriptor_type_ == descriptor_type_
            && that.descriptor_->equals(*descriptor_);
    }

    bool operator!=(const KernelDescriptor& that) const {
        return !(*this == that);
    }

  private:
    hash_t hash_;
    TypeInfo descriptor_type_;
    std::shared_ptr<IKernelDescriptor> descriptor_;
};
}  // namespace kernel_launcher

namespace std {
template<>
struct hash<kernel_launcher::KernelDescriptor> {
    size_t operator()(const kernel_launcher::KernelDescriptor& d) const {
        return d.hash();
    }
};
}  // namespace std

namespace kernel_launcher {
struct KernelRegistry {
    explicit KernelRegistry(
        Compiler compiler = default_compiler(),
        WisdomSettings settings = default_wisdom_settings()) :
        compiler_(std::move(compiler)),
        settings_(std::move(settings)) {}

    WisdomKernel& lookup(KernelDescriptor descriptor) const {
        return lookup_internal(std::move(descriptor));
    }

    WisdomKernelLaunch instantiate(
        KernelDescriptor descriptor,
        cudaStream_t stream,
        ProblemSize problem_size) const {
        return lookup(std::move(descriptor)).instantiate(stream, problem_size);
    }

    WisdomKernelLaunch operator()(
        KernelDescriptor descriptor,
        cudaStream_t stream,
        ProblemSize problem_size) const {
        return instantiate(std::move(descriptor), stream, problem_size);
    }

    WisdomKernelLaunch
    operator()(KernelDescriptor descriptor, ProblemSize problem_size) const {
        return instantiate(std::move(descriptor), nullptr, problem_size);
    }

  private:
    WisdomKernel& lookup_internal(KernelDescriptor key) const;

    Compiler compiler_;
    WisdomSettings settings_;
    mutable std::mutex mutex_;
    mutable std::
        unordered_map<KernelDescriptor, std::unique_ptr<WisdomKernel>>
            cache_ = {};
};

const KernelRegistry& default_registry();

inline WisdomKernelLaunch
launch(KernelDescriptor descriptor, cudaStream_t stream, ProblemSize size) {
    return default_registry().instantiate(descriptor, stream, size);
}

inline WisdomKernelLaunch
launch(KernelDescriptor descriptor, ProblemSize size) {
    return launch(descriptor, (cudaStream_t) nullptr, size);
}

}  // namespace kernel_launcher

#endif  //KERNEL_LAUNCHER_CACHE_H
