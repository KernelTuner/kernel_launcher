#ifndef KERNEL_LAUNCHER_ARG_H
#define KERNEL_LAUNCHER_ARG_H

#include <cuda.h>

#include <cstring>
#include <iostream>
#include <utility>

#include "kernel_launcher/builder.h"
#include "kernel_launcher/compiler.h"
#include "kernel_launcher/config.h"

namespace kernel_launcher {

struct KernelArg {
  private:
    KernelArg(TypeInfo type, void* data);
    KernelArg(TypeInfo type, void* ptr, size_t nelements);

  public:
    KernelArg();
    KernelArg(KernelArg&&) noexcept;
    KernelArg(const KernelArg&);
    ~KernelArg();

    template<typename T>
    static KernelArg for_scalar(T value) {
        static_assert(sizeof(T) == type_of<T>().size(), "internal error");
        return KernelArg(type_of<T>(), (void*)&value);
    }

    template<typename T>
    static KernelArg for_array(T* value, size_t nelements) {
        static_assert(sizeof(T) == type_of<T>().size(), "internal error");
        static_assert(sizeof(T*) == type_of<T*>().size(), "internal error");
        return KernelArg(type_of<T*>(), (void*)value, nelements);
    }

    template<typename T>
    T to() const {
        static_assert(
            std::is_trivially_copyable<T>::value,
            "type must be trivial");
        assert_type_matches(type_of<T>());

        T result = {};
        ::memcpy(&result, as_void_ptr(), sizeof(T));
        return result;
    }

    Value to_value() const;
    Value to_value_or_empty() const;
    void assert_type_matches(TypeInfo t) const;
    bool is_scalar() const;
    bool is_array() const;
    TypeInfo type() const;
    std::vector<uint8_t> to_bytes() const;
    void* as_void_ptr() const;

  private:
    TypeInfo type_;
    bool scalar_;
    union {
        struct {
            void* ptr;
            size_t nelements;
        } array;
        std::array<uint8_t, 2 * sizeof(size_t)> small_scalar;
        void* large_scalar;
    } data_;
};

template<typename T, typename Enabled = void>
struct IntoKernelArg;

template<>
struct IntoKernelArg<KernelArg> {
    static KernelArg convert(KernelArg arg) {
        return arg;
    }
};

template<typename T>
struct IntoKernelArg<
    T,
    typename std::enable_if<std::is_scalar<T>::value>::type> {
    static KernelArg convert(T value) {
        return KernelArg::for_scalar<T>(value);
    }
};

template<typename T>
struct IntoKernelArg<
    CudaSpan<T>,
    typename std::enable_if<std::is_trivially_copyable<T>::value>::type> {
    static KernelArg convert(CudaSpan<T> s) {
        return KernelArg::for_array<T>(s.data(), s.size());
    }
};

template<typename T>
KernelArg into_kernel_arg(T&& value) {
    return IntoKernelArg<typename std::decay<T>::type>::convert(value);
}

}  // namespace kernel_launcher

#endif  //KERNEL_LAUNCHER_ARG_H
