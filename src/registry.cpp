#include "kernel_launcher/registry.h"

namespace kernel_launcher {
KernelRegistry* global_default_registry = nullptr;

const KernelRegistry& default_registry() {
    if (global_default_registry == nullptr) {
        global_default_registry = new KernelRegistry;
    }

    return *global_default_registry;
}

WisdomKernel& KernelRegistry::lookup_internal(KernelDescriptor key) const {
    std::lock_guard<std::mutex> guard(mutex_);

    auto it = cache_.find(key);
    if (it != cache_.end()) {
        return *it->second;
    }

    WisdomKernel kernel(key.get().build(), compiler_, settings_);

    auto entry = cache_.emplace(
        std::move(key),
        std::make_unique<WisdomKernel>(std::move(kernel)));
    return *entry.first->second;
}

}  // namespace kernel_launcher