#include "kernel_launcher/kernel_arg.hpp"

#include <utility>

namespace kernel_launcher {

KernelArg KernelArg::unsafe_construct(TypeInfo type, const void* data, size_t nbytes) {
    if (nbytes != type.size_in_bytes()) {
        throw std::invalid_argument(concat(
            "KernelArg::unsafe_construct: type '",
            type.name(),
            "' has size ",
            type.size_in_bytes(),
            ", but ",
            nbytes,
            " bytes were given"));
    }

    KernelArg arg(type, 0);

    if (arg.is_heap()) {
        arg.storage_.heap_data = new std::byte[nbytes];
        std::memcpy(arg.storage_.heap_data, data, nbytes);
    } else if (nbytes > 0) {
        // `data` may be null for a type of size zero, which `memcpy` does not allow.
        std::memcpy(arg.storage_.inline_data, data, nbytes);
    }

    return arg;
}

KernelArg::KernelArg(const KernelArg& that) :
    type_(that.type_),
    nbytes_(that.nbytes_),
    storage_(that.storage_) {
    if (is_heap()) {
        size_t size = type_.size_in_bytes();
        storage_.heap_data = new std::byte[size];
        std::memcpy(storage_.heap_data, that.storage_.heap_data, size);
    }
}

KernelArg::KernelArg(KernelArg&& that) noexcept :
    type_(that.type_),
    nbytes_(that.nbytes_),
    storage_(that.storage_) {
    // `void` has size zero, so `that` no longer owns `heap_data`.
    that.type_ = TypeInfo::of<void>();
}

KernelArg& KernelArg::operator=(KernelArg&& that) noexcept {
    std::swap(type_, that.type_);
    std::swap(nbytes_, that.nbytes_);
    std::swap(storage_, that.storage_);
    return *this;
}

KernelArg::~KernelArg() {
    if (is_heap()) {
        delete[] storage_.heap_data;
    }
}

KernelArg& KernelArg::operator=(const KernelArg& that) {
    if (this != &that) {
        *this = KernelArg(that);
    }

    return *this;
}

KernelArg KernelArg::with_buffer_size(size_t nbytes) const {
    if (!type_.is_pointer()) {
        throw std::runtime_error("KernelArg::with_buffer_size: argument is not a pointer");
    }

    KernelArg arg = *this;
    arg.nbytes_ = nbytes;
    return arg;
}

Primitive KernelArg::to_primitive_or_empty() const {
    if (type_.is_primitive()) {
        return type_.to_primitive(data());
    } else {
        return {};
    }
}

Primitive KernelArg::to_primitive() const {
    Primitive result = to_primitive_or_empty();

    if (result.is_empty()) {
        throw std::runtime_error(concat(
            "KernelArg::to_primitive: argument type '",
            type_.name(),
            "' cannot be converted to a primitive value"));
    }

    return result;
}

}  // namespace kernel_launcher
