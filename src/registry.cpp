#include "kernel_launcher/registry.hpp"

#include <mutex>
#include <unordered_map>

namespace kernel_launcher {

struct DescriptorKeyHash {
    size_t operator()(const DescriptorKey& k) const noexcept {
        return k.hash();
    }
};

struct Registry::Impl {
    Impl(KernelSettings&& settings) : settings(std::move(settings)) {}

    std::mutex mutex;
    KernelSettings settings;
    std::unordered_map<DescriptorKey, Kernel, DescriptorKeyHash> kernels;
};

Registry::Registry(KernelSettings settings) : impl_(std::make_shared<Impl>(std::move(settings))) {}

Kernel Registry::lookup(DescriptorKey key) {
    std::lock_guard guard {impl_->mutex};

    if (auto it = impl_->kernels.find(key); it != impl_->kernels.end()) {
        return it->second;
    }

    Kernel kernel(key.get().build(), impl_->settings);
    impl_->kernels.emplace(std::move(key), kernel);
    return kernel;
}

Registry& default_registry() {
    static Registry registry;
    return registry;
}

}  // namespace kernel_launcher
