#ifndef KERNEL_LAUNCHER_CACHE_H
#define KERNEL_LAUNCHER_CACHE_H

#include <memory>
#include <mutex>
#include <unordered_map>

#include "kernel_launcher/wisdom.h"

namespace kernel_launcher {

struct KernelDescriptor {
    virtual ~KernelDescriptor() = default;
    virtual std::string tuning_key() const = 0;
    virtual KernelBuilder build() const = 0;
    virtual bool equals(const KernelDescriptor& that) const = 0;
    virtual hash_t hash() const {
        return 0;
    }
};

struct AnyKernelDescriptor {
    AnyKernelDescriptor(AnyKernelDescriptor&&) noexcept = default;
    AnyKernelDescriptor(const AnyKernelDescriptor&) = default;

    template<typename D>
    AnyKernelDescriptor(D&& descriptor) {
        using T = typename std::decay<D>::type;
        descriptor_type_ = type_of<T>();
        descriptor_ = std::make_unique<T>(std::forward<D>(descriptor));
        hash_ = hash_fields(descriptor_type_, descriptor_->hash());
    }

    const KernelDescriptor& descriptor() const {
        return *descriptor_;
    }

    hash_t hash() const {
        return hash_;
    }

    bool operator==(const AnyKernelDescriptor& that) const {
        return that.hash_ == hash_ && that.descriptor_type_ == descriptor_type_
            && that.descriptor_->equals(*descriptor_);
    }

    bool operator!=(const AnyKernelDescriptor& that) const {
        return !(*this == that);
    }

  private:
    hash_t hash_;
    TypeInfo descriptor_type_;
    std::shared_ptr<KernelDescriptor> descriptor_;
};
}

namespace std {
template <>
struct hash<kernel_launcher::AnyKernelDescriptor> {
    size_t operator()(const kernel_launcher::AnyKernelDescriptor& d) const {
        return d.hash();
    }
};
}

namespace kernel_launcher {
struct KernelRegistry {
    explicit KernelRegistry(
        Compiler compiler = default_compiler(),
        WisdomSettings settings = default_wisdom_settings()) :
        compiler_(std::move(compiler)),
        settings_(std::move(settings)) {}

    WisdomKernel& lookup(AnyKernelDescriptor descriptor) const {
        return lookup_internal(std::move(descriptor));
    }

    WisdomKernelLaunch instantiate(
        AnyKernelDescriptor descriptor,
        cudaStream_t stream,
        ProblemSize problem_size) const {
        return lookup(std::move(descriptor)).instantiate(stream, problem_size);
    }

    WisdomKernelLaunch operator()(
        AnyKernelDescriptor descriptor,
        cudaStream_t stream,
        ProblemSize problem_size) const {
        return instantiate(std::move(descriptor), stream, problem_size);
    }

    WisdomKernelLaunch
    operator()(AnyKernelDescriptor descriptor, ProblemSize problem_size) const {
        return instantiate(std::move(descriptor), nullptr, problem_size);
    }

  private:
    WisdomKernel& lookup_internal(AnyKernelDescriptor key) const;

    Compiler compiler_;
    WisdomSettings settings_;
    mutable std::mutex mutex_;
    mutable std::
        unordered_map<AnyKernelDescriptor, std::unique_ptr<WisdomKernel>>
            cache_ = {};
};

const KernelRegistry& default_registry();

template<typename D>
WisdomKernelLaunch
launch(D&& descriptor, cudaStream_t stream, ProblemSize size) {
    return default_registry().instantiate(
        std::forward<D>(descriptor, stream, size));
}

template<typename D>
WisdomKernelLaunch launch(D&& descriptor, ProblemSize size) {
    return launch(std::forward<D>(descriptor), (cudaStream_t) nullptr, size);
}

}  // namespace kernel_launcher


#endif  //KERNEL_LAUNCHER_CACHE_H
