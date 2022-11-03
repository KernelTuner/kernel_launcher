#include "kernel_launcher/arg.h"

#include <cstring>

namespace kernel_launcher {

static bool is_inline_scalar(TypeInfo type) {
    return type.size() <= sizeof(size_t) * 2;
}

KernelArg::KernelArg(TypeInfo type, void* data) {
    type_ = type.remove_const();
    scalar_ = true;

    if (is_inline_scalar(type_)) {
        ::memcpy(data_.small_scalar.data(), data, type.size());
    } else {
        data_.large_scalar = new char[type.size()];
        ::memcpy(data_.large_scalar, data, type.size());
    }
}

KernelArg::KernelArg(TypeInfo type, void* ptr, size_t nelements) {
    type_ = type.remove_const();
    scalar_ = false;
    data_.array.ptr = ptr;
    data_.array.nelements = nelements;
}

KernelArg::KernelArg(const KernelArg& that) : KernelArg() {
    type_ = that.type_;
    scalar_ = that.scalar_;

    if (that.is_array()) {
        data_.array = that.data_.array;
    } else if (is_inline_scalar(type_)) {
        data_.small_scalar = that.data_.small_scalar;
    } else {
        data_.large_scalar = new char[type_.size()];
        ::memcpy(data_.large_scalar, that.data_.large_scalar, type_.size());
    }
}

KernelArg::~KernelArg() {
    if (is_scalar() && !is_inline_scalar(type_)) {
        delete[](char*) data_.large_scalar;
    }
}

KernelArg::KernelArg(KernelArg&& that) noexcept : KernelArg() {
    std::swap(this->data_, that.data_);
    std::swap(this->type_, that.type_);
    std::swap(this->scalar_, that.scalar_);
}

Value KernelArg::to_value_or_empty() const {
    TypeInfo ty = type_.remove_const();
    void* ptr = as_void_ptr();

#define IMPL_FOR_VALUE(T)             \
    if (ty == type_of<T>()) {         \
        T x;                          \
        ::memcpy(&x, ptr, sizeof(T)); \
        return x;                     \
    }

    IMPL_FOR_VALUE(char);
    IMPL_FOR_VALUE(signed char);
    IMPL_FOR_VALUE(unsigned char);

    IMPL_FOR_VALUE(signed short);
    IMPL_FOR_VALUE(signed int);
    IMPL_FOR_VALUE(signed long);
    IMPL_FOR_VALUE(signed long long);

    IMPL_FOR_VALUE(unsigned short);
    IMPL_FOR_VALUE(unsigned int);
    IMPL_FOR_VALUE(unsigned long);
    IMPL_FOR_VALUE(unsigned long long);

    IMPL_FOR_VALUE(bool);
    IMPL_FOR_VALUE(float);
    IMPL_FOR_VALUE(double);

    return {};
}

Value KernelArg::to_value() const {
    Value v = to_value_or_empty();

    if (v.is_empty()) {
        throw std::runtime_error(
            "cannot convert value of type \"" + type_.name()
            + "\" instance of kernel_launcher::Value");
    }

    return v;
}

void KernelArg::assert_type_matches(TypeInfo t) const {
    bool valid;

    if (t.is_pointer()) {
        // Matches if both are pointers and pointee type matches (possible
        // when adding a const modifier).
        valid = type_.is_pointer()
            && (t.remove_pointer() == type_.remove_pointer()
                || t.remove_pointer() == type_.remove_pointer().add_const());
    } else {
        valid = t.remove_const() == type_.remove_const();
    }

    if (!valid) {
        throw std::runtime_error(
            "cannot cast kernel argument of type `" + type_.name()
            + "` to type `" + t.name() + "`");
    }
}

bool KernelArg::is_array() const {
    return !scalar_;
}

bool KernelArg::is_scalar() const {
    return scalar_;
}

TypeInfo KernelArg::type() const {
    return type_;
}

std::vector<uint8_t> KernelArg::to_bytes() const {
    std::vector<uint8_t> result;

    if (is_array()) {
        result.resize(type_.remove_pointer().size() * data_.array.nelements);
        KERNEL_LAUNCHER_CUDA_CHECK(cuMemcpy(
            reinterpret_cast<CUdeviceptr>(result.data()),
            reinterpret_cast<CUdeviceptr>(data_.array.ptr),
            result.size()));
    } else {
        // If the type is a pointer, exporting it to bytes will return
        // the memory address of the pointer and not the data of the buffer it
        // points to This is likely a bug on the user side. Todo: find a better
        //  way f handling this error (maybe already in KernelArg ctor?).
        if (type_.is_pointer()) {
            throw std::runtime_error("a raw pointer type was provided as "
                "kernel argument (" + type_.name() + ") which cannot be "
                "exported since the corresponding buffer size is unknown");
        }

        result.resize(type_.size());

        if (is_inline_scalar(type_)) {
            ::memcpy(result.data(), data_.small_scalar.data(), type_.size());
        } else {
            ::memcpy(result.data(), data_.large_scalar, type_.size());
        }
    }

    return result;
}

KernelArg::KernelArg() : type_(type_of<int>()), scalar_(true) {}

void* KernelArg::as_void_ptr() const {
    if (is_array()) {
        return const_cast<void*>(static_cast<const void*>(&data_.array.ptr));
    } else if (is_inline_scalar(type_)) {
        return const_cast<void*>(
            static_cast<const void*>(data_.small_scalar.data()));
    } else {
        return data_.large_scalar;
    }
}

}  // namespace kernel_launcher