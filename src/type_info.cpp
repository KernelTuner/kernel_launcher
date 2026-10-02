#include "kernel_launcher/type_info.hpp"

#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <tuple>

namespace kernel_launcher {

TypeInfo TypeInfo::opaque(std::string_view name, size_t size, size_t alignment) {
    if (name.empty()) {
        throw std::invalid_argument("TypeInfo::opaque: name must not be empty");
    }

    if (alignment == 0 || (alignment & (alignment - 1)) != 0 || size % alignment != 0) {
        throw std::invalid_argument(concat(
            "TypeInfo::opaque: invalid size ",
            size,
            " and alignment ",
            alignment,
            " for type '",
            name,
            "'"));
    }

    // Interned by name, size, and alignment, so equal opaque types share the same impl.
    using Key = std::tuple<std::string_view, size_t, size_t>;
    static std::mutex mutex;
    static std::map<Key, std::unique_ptr<detail::TypeInfoImpl>> pool;

    std::lock_guard<std::mutex> guard(mutex);
    auto it = pool.find(Key {name, size, alignment});

    if (it == pool.end()) {
        auto impl = std::make_unique<detail::TypeInfoImpl>();
        impl->size = size;
        impl->alignment = alignment;
        impl->is_pointer = false;
        impl->is_const = false;
        impl->remove_pointer = impl.get();
        impl->remove_const = impl.get();
        impl->type_info = nullptr;
        impl->name = intern_string(name);
        impl->to_primitive = nullptr;

        Key key {impl->name, size, alignment};
        it = pool.emplace(key, std::move(impl)).first;
    }

    return {it->second.get()};
}

Primitive TypeInfo::to_primitive(const void* ptr) const {
    if (!is_primitive()) {
        throw std::runtime_error(concat(
            "TypeInfo::to_primitive: type '",
            name(),
            "' cannot be converted to a primitive value"));
    }

    return impl_->to_primitive(ptr);
}

}  // namespace kernel_launcher
