#ifndef KERNEL_LAUNCHER_CACHE_H
#define KERNEL_LAUNCHER_CACHE_H

#include <memory>
#include <mutex>
#include <unordered_map>

#include "kernel_launcher/kernel.h"

namespace kernel_launcher {

/**
 * This interface is used to implement an abstract representation for a kernel.
 * For example, one may implement a `MatrixMultiplyDescriptor` that represents
 * a kernel for matrix multiplication. The `build` method should return the
 * `KernelBuilder` that be used to build the specific kernel.
 *
 * The interface is used in combination with `KernelRegistry` to look up if this
 * kernel has already been compiled. The registry is essentially a hash table
 * that maps `IKernelDescriptor` objects to `KernelInstance` objects. This is
 * why this descriptor class must implement `equals` and `hash`.
 */
struct IKernelDescriptor {
    virtual ~IKernelDescriptor() = default;

    /**
     * Should return the `KernelBuilder` that can be used to build the kernel
     * associated with this descriptor.
     */
    virtual KernelBuilder build() const = 0;

    /**
     * Check if this descriptor is equal to another descriptor.
     */
    virtual bool equals(const IKernelDescriptor& that) const = 0;

    /**
     * Return a hash of this descriptor. This is used to test for equality of
     * two descriptors:
     *
     * * Two descriptors that return the same hash MAY be identical.
     * * Two descriptors that return different hashes MUST be different.
     *
     * This method is optional; its default implementation just returns `0`.
     */
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

/**
 * A registry can be used to cache kernel compilations. Each registry is
 * essentially a table that maps `IKernelDescriptor` objects to
 * `WisdomKernel` objects.
 *
 * There is a single global registry that is available with `default_registry`.
 * However, it is also possible to construct a local `KernelRegistry`.
 *
 * This class is thread-safe.
 */
struct KernelRegistry {
    /**
     * Construct new registry.
     */
    explicit KernelRegistry(
        Compiler compiler = default_compiler(),
        WisdomSettings settings = default_wisdom_settings()) :
        compiler_(std::move(compiler)),
        settings_(std::move(settings)) {}

    /**
     * Look up the `WisdomKernel` associated with the given descriptor. This
     * will instantiate a new `WisdomKernel` if not yet available.
     */
    WisdomKernel& lookup(KernelDescriptor descriptor) const {
        return lookup_internal(std::move(descriptor));
    }

    template<typename... Args>
    void launch(KernelDescriptor descriptor, Args&&... args) const {
        return lookup(std::move(descriptor))
            .launch(std::forward<Args>(args)...);
    }

  private:
    WisdomKernel& lookup_internal(KernelDescriptor key) const;

    Compiler compiler_;
    WisdomSettings settings_;
    mutable std::mutex mutex_;
    mutable std::unordered_map<KernelDescriptor, std::unique_ptr<WisdomKernel>>
        cache_ = {};
};

/**
 * Returns the global `KernelRegistry` for the program. See `KernelRegistry`
 * for more information.
 */
const KernelRegistry& default_registry();

/**
 * Launch the kernel given a `KernelDescriptor` using the global registry.
 * This is a short-hand for
 * `default_registry().lookup(descriptor).launch(args...)`
 */
template<typename... Args>
void launch(KernelDescriptor descriptor, Args&&... args) {
    return default_registry().launch(
        descriptor,
        (cudaStream_t) nullptr,
        std::forward<Args>(args)...);
}

}  // namespace kernel_launcher

#endif  //KERNEL_LAUNCHER_CACHE_H
