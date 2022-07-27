#include "kernel_launcher/registry.h"

namespace kernel_launcher {
KernelRegistry* global_default_registry = nullptr;

const KernelRegistry& default_registry() {
    if (global_default_registry == nullptr) {
        global_default_registry = new KernelRegistry;
    }

    return *global_default_registry;
}

size_t KernelRegistry::CacheKey::hasher::operator()(const CacheKey& key) const {
    return key.hash_;
}

bool KernelRegistry::CacheKey::equals::operator()(
    const CacheKey& lhs,
    const CacheKey& rhs) const {
    return lhs.hash_ == rhs.hash_
        && lhs.descriptor_type_ == rhs.descriptor_type_
        && lhs.descriptor_->equals(*rhs.descriptor_);
}

WisdomKernel& KernelRegistry::lookup_internal(CacheKey key) const {
    std::lock_guard<std::mutex> guard(mutex_);

    auto it = cache_.find(key);
    if (it != cache_.end()) {
        return *it->second;
    }

    WisdomKernel kernel(
        key.descriptor().tuning_key(),
        key.descriptor().build(),
        compiler_,
        settings_);

    auto entry = cache_.emplace(
        std::move(key),
        std::make_unique<WisdomKernel>(std::move(kernel)));
    return *entry.first->second;
}

}  // namespace kernel_launcher