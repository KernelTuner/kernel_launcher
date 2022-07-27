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

struct KernelRegistry {
  private:
    struct CacheKey {
        struct hasher {
            size_t operator()(const CacheKey& key) const;
        };

        struct equals {
            bool operator()(const CacheKey& lhs, const CacheKey& rhs) const;
        };

        CacheKey(CacheKey&&) noexcept = default;

        template<typename D>
        explicit CacheKey(D&& descriptor) {
            using T = typename std::decay<D>::type;
            descriptor_type_ = type_of<T>();
            descriptor_ = std::make_unique<T>(std::forward<D>(descriptor));
            hash_ = hash_fields(descriptor_type_, descriptor_->hash());
        }

        const KernelDescriptor& descriptor() const {
            return *descriptor_;
        }

      private:
        hash_t hash_;
        TypeInfo descriptor_type_;
        std::unique_ptr<KernelDescriptor> descriptor_;
    };

  public:
    explicit KernelRegistry(
        Compiler compiler = default_compiler(),
        WisdomSettings settings = default_wisdom_settings()) :
        compiler_(std::move(compiler)),
        settings_(std::move(settings)) {}

    template<typename D>
    WisdomKernel& lookup(D&& descriptor) const {
        return lookup_internal(CacheKey(std::forward<D>(descriptor)));
    }

    template<typename D>
    WisdomKernelLaunch
    instantiate(D&& descriptor, cudaStream_t stream, ProblemSize problem_size)
        const {
        return lookup(std::forward<D>(descriptor))
            .instantiate(stream, problem_size);
    }

    template<typename D>
    WisdomKernelLaunch
    operator()(D&& descriptor, cudaStream_t stream, ProblemSize problem_size)
        const {
        return instantiate(std::forward<D>(descriptor), stream, problem_size);
    }

    template<typename D>
    WisdomKernelLaunch
    operator()(D&& descriptor, ProblemSize problem_size) const {
        return instantiate(std::forward<D>(descriptor), nullptr, problem_size);
    }

  private:
    WisdomKernel& lookup_internal(CacheKey key) const;

    Compiler compiler_;
    WisdomSettings settings_;
    mutable std::mutex mutex_;
    mutable std::unordered_map<
        CacheKey,
        std::unique_ptr<WisdomKernel>,
        CacheKey::hasher,
        CacheKey::equals>
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
