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
    *this = that;
}

KernelArg::KernelArg(KernelArg&& that) noexcept : KernelArg() {
    *this = std::move(that);
}

KernelArg::~KernelArg() {
    if (is_scalar() && !is_inline_scalar(type_)) {
        delete[](char*) data_.large_scalar;
    }
}

KernelArg& KernelArg::operator=(const KernelArg& that) {
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

    return *this;
}

KernelArg& KernelArg::operator=(KernelArg&& that) noexcept {
    std::swap(this->data_, that.data_);
    std::swap(this->type_, that.type_);
    std::swap(this->scalar_, that.scalar_);
    return *this;
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

KernelArg KernelArg::to_array(size_t nelements) const {
    if (is_array()) {
        if (nelements > data_.array.nelements) {
            throw std::runtime_error(
                "array of type `" + type_.remove_pointer().name()
                + "` cannot be be resized to " + std::to_string(nelements)
                + " elements, it only has "
                + std::to_string(data_.array.nelements) + " elements");
        }

        return {type_, data_.array.ptr, nelements};
    } else {
        if (!type_.is_pointer()) {
            throw std::runtime_error(
                "argument of type `" + type_.name() + "` is not a pointer type "
                "and thus cannot be converted into an array");
        }

        return {type_, *(void**)as_void_ptr(), nelements};
    }
}

Value KernelArg::to_value() const {
    Value v = to_value_or_empty();

    if (v.is_empty()) {
        throw std::runtime_error(
            "cannot convert value of type `" + type_.name()
            + "` instance of kernel_launcher::Value");
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
    // If this type is pointer, we check if it is a NULL pointer. It makes no
    // sense to export non-NULL pointer since the address will be invalid
    // when reading the exported pointer.
    if (type_.is_pointer()) {
        void* ptr;
        ::memcpy(&ptr, as_void_ptr(), sizeof(void*));

        if (ptr != nullptr) {
            std::string msg = "a raw pointer of type `" + type_.name() + "` "
                "was provided as kernel argument which cannot be exported "
                "since the corresponding buffer size is unknown";

            throw std::runtime_error(msg);
        }
    }

    size_t nbytes = type_.size();
    std::vector<uint8_t> result(nbytes);
    ::memcpy(result.data(), as_void_ptr(), nbytes);
    return result;
}

std::vector<uint8_t> KernelArg::copy_array() const {
    if (is_array()) {
        size_t nbytes = type_.remove_pointer().size() * data_.array.nelements;
        std::vector<uint8_t> result(nbytes);

        if (nbytes > 0) {
            KERNEL_LAUNCHER_CUDA_CHECK(cuMemcpy(
                reinterpret_cast<CUdeviceptr>(result.data()),
                reinterpret_cast<CUdeviceptr>(data_.array.ptr),
                nbytes));
        }

        return result;
    }

    std::string msg;

    if (type_.is_pointer()) {
        msg = "a raw pointer of type `" + type_.name() + "` was provided as "
            "kernel argument which cannot be exported since the "
            "corresponding buffer size is unknown";
    } else {
        msg = "a scalar of type `" + type_.name() + "` was provided as "
            "kernel argument which cannot be exported since it is not "
            "an array";
    }

    throw std::runtime_error(msg);
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

std::ostream& operator<<(std::ostream& os, const KernelArg& arg) {
    // There are four possible representations:
    // - pointer which is an array (length is known)
    // - pointer which is not an array (length is unknown)
    // - scalars convertible to `Value`
    // - scalars without a representation
    if (arg.type().is_pointer()) {
        void* ptr;
        ::memcpy(&ptr, arg.as_void_ptr(), sizeof(ptr));
        os << "array " << ptr;

        if (arg.is_array()) {
            os << " of length " << arg.data_.array.nelements;
        }
    } else {
        Value v = arg.to_value_or_empty();

        if (!v.is_empty()) {
            os << "scalar " << v;
        } else {
            os << "scalar <...>";
        }
    }

    return os << " (type: " << arg.type_.name() << ")";
}

}  // namespace kernel_launcher