#ifndef KERNEL_LAUNCHER_REGISTRY_H
#define KERNEL_LAUNCHER_REGISTRY_H

#include <cstddef>
#include <memory>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <utility>

#include "kernel_launcher/kernel.hpp"
#include "kernel_launcher/utils.hpp"

namespace kernel_launcher {

/**
 * @addtogroup registry
 * @{
 */

/**
 * A `KernelDescriptor` describes a particular kernel, for example, "`matmul` kernel with precision
 * of float and input 1024x1024". It is used by `Registry` to create a map from `KernelDescriptor`
 * to `Kernel` objects.
 *
 * The most important method is `build`, which should return the corresponding `KernelBuilder`.
 * The methods `equals` and `hash` are needed for the hash table.
 */
class KernelDescriptor {
  public:
    virtual ~KernelDescriptor() = default;

    /**
     * Returns the `KernelBuilder` corresponding to this kernel.
     */
    virtual KernelBuilder build() const = 0;

    /**
     * Returns if this `KernelDescriptor` equals another kernel descriptor. Use
     * `dynamic_cast<const MyKernelDescriptor*>` to cast the descriptor to your own type.
     */
    virtual bool equals(const KernelDescriptor& that) const = 0;

    /**
     * Returns a hash. It should be the case that if `a.equals(b)`, then `a.hash() == b.hash()`.
     */
    virtual size_t hash() const = 0;
};

/**
 * Creates a `KernelDescriptor` from a function pointer that returns a `KernelBuilder`. If the
 * function takes arguments, then these can be passed as well.
 */
template<typename... Args>
class FnKernelDescriptor: public KernelDescriptor {
  public:
    using function_type = KernelBuilder (*)(Args...);

    FnKernelDescriptor(function_type fn, std::decay_t<Args>... args) :
        fn_(fn),
        args_(std::move(args)...) {}

    KernelBuilder build() const override {
        return std::apply(fn_, args_);
    }

    bool equals(const KernelDescriptor& that) const override {
        const auto* other = dynamic_cast<const FnKernelDescriptor*>(&that);
        return other != nullptr && fn_ == other->fn_ && args_ == other->args_;
    }

    size_t hash() const override {
        // TODO: arguments may not always be hashable. Maybe we need something that skips all
        // fields that are not hashable.
        // return std::apply([&](const auto&... args) { return hash_fields(fn_, args...); }, args_);

        return hash_fields(fn_);
    }

  private:
    function_type fn_;
    std::tuple<std::decay_t<Args>...> args_;
};

/**
 * Creates a `KernelDescriptor` from a function that returns a `KernelBuilder`. This is useful for
 * passing it to `Registry::lookup` or `launch`.
 *
 * For example:
 *
 * ```
 * static kernel_launcher::KernelBuilder vector_add_kernel(int n) {
 *   kernel_launcher::KernelBuilder builder(...);
 *   ...
 *   return builder;
 * }
 *
 * void launch_vector_add(float* A, float* B, float* C) {
 *   kernel_launcher::launch(kernel_launcher::fn_kernel(vector_add_kernel, 100), A, B, C);
 * }
 * ```
 */
template<typename... Args>
FnKernelDescriptor<Args...> fn_kernel(KernelBuilder (*fn)(Args...), std::decay_t<Args>... args) {
    return {fn, std::move(args)...};
}

/**
 * Wrapper around a `std::unique_ptr<const KernelDescriptor>`.
 */
class DescriptorKey {
  public:
    DescriptorKey(std::unique_ptr<const KernelDescriptor> desc) : desc_(std::move(desc)) {
        const KernelDescriptor& inner = *desc_;
        hash_combine(hash_, typeid(inner).hash_code());
        hash_combine(hash_, desc_->hash());
    }

    template<
        typename D,
        typename = std::enable_if_t<std::is_base_of_v<KernelDescriptor, std::decay_t<D>>>>
    DescriptorKey(D&& desc) :
        DescriptorKey(std::make_unique<const std::decay_t<D>>(std::forward<D>(desc))) {}

    /**
     * Shorthand for `DescriptorKey(FnKernelDescriptor(fn))`.
     */
    DescriptorKey(KernelBuilder (*fn)()) : DescriptorKey(FnKernelDescriptor(fn)) {}

    /**
     * Returns the underlying `KernelDescriptor`.
     */
    const KernelDescriptor& get() const {
        return *desc_;
    }

    /**
     * Returns the hash of the `KernelDescriptor`.
     */
    size_t hash() const {
        return hash_;
    }

    bool operator==(const DescriptorKey& that) const {
        const KernelDescriptor& a = *desc_;
        const KernelDescriptor& b = *that.desc_;
        return hash_ == that.hash_ && typeid(a) == typeid(b) && a.equals(b);
    }

    bool operator!=(const DescriptorKey& that) const {
        return !(*this == that);
    }

  private:
    std::unique_ptr<const KernelDescriptor> desc_;
    size_t hash_ = 0;
};

/**
 * A `Registry` is basically a table that maps `KernelDescriptor` to a `Kernel` object.
 *
 * The registry can be used to cache kernel compilations for cases where it is difficult to store
 * the Kernel objects locally somewhere. There is a single global registry that is available with
 * `default_registry`. However, it is also possible to construct a local `Registry`.
 *
 * This class is thread-safe.
 */
class Registry {
  public:
    explicit Registry(KernelSettings settings = {});

    /**
     * Look up the `Kernel` associated with the given key, or instantiate a new `Kernel`.
     */
    Kernel lookup(DescriptorKey key);

    /**
     * Shorthand for `registry.lookup(key).launch(args...)`.
     */
    template<typename... Ts>
    void launch(DescriptorKey key, const Ts&... args) {
        lookup(std::move(key)).launch(args...);
    }

  private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

/**
 * Returns the global `Registry`.
 */
Registry& default_registry();

/**
 * Shorthand for `default_registry().lookup(key).launch(args...)`.
 */
template<typename... Ts>
void launch(DescriptorKey key, const Ts&... args) {
    default_registry().lookup(std::move(key)).launch(args...);
}

/** @} */

}  // namespace kernel_launcher

#endif
