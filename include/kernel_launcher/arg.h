#ifndef KERNEL_LAUNCHER_ARG_H
#define KERNEL_LAUNCHER_ARG_H

#include <cuda.h>

#include <cstring>
#include <iostream>
#include <utility>

#include "kernel_launcher/compiler.h"
#include "kernel_launcher/config.h"

namespace kernel_launcher {

/**
 * An argument that will be passed to a kernel. A `KernelArg` can either be:
 *
 *  * A scalar such as `int`, `float`, `double`, etc.
 *  * An array consisting of a pointer value (e.g., `int*`, `float*`) and a
 *    length that indicates the number of elements of the array's type.
 *
 * This class contains method to construct `KernelArg` and to convert it
 * back into C++ value. Use `into_kernel_arg` to convert a value into
 * `KernelArg`.
 */
struct KernelArg {
  private:
    KernelArg(TypeInfo type, void* data);
    KernelArg(TypeInfo type, void* ptr, size_t nelements);

  public:
    KernelArg();
    KernelArg(KernelArg&&) noexcept;
    KernelArg(const KernelArg&);
    ~KernelArg();

    KernelArg& operator=(const KernelArg&);
    KernelArg& operator=(KernelArg&&) noexcept;

    /**
     * Construct a `KernelArg` from a scalar.
     */
    template<typename T>
    static KernelArg from_scalar(T value) {
        static_assert(sizeof(T) == type_of<T>().size(), "internal error");
        return KernelArg(type_of<T>(), (void*)&value);
    }

    /**
     * Construct a `KernelArg` from an array.
     */
    template<typename T>
    static KernelArg from_array(T* value, size_t nelements) {
        static_assert(sizeof(T) == type_of<T>().size(), "internal error");
        static_assert(sizeof(T*) == type_of<T*>().size(), "internal error");
        return KernelArg(type_of<T*>(), (void*)value, nelements);
    }

    /**
     * Convert this `KernelArg` to a value of type `T`. Throws an exception if
     * this value is not of type `T`.
     */
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

    /**
     * Convert this `KernelArg` to a `Value`. This is only valid for integer
     * types (`int`, `long`, `short`, etc) and floating-point types (`float`,
     * `double`, etc). Throws an exception if the inner type of this `KernelArg`
     * cannot is not an integer or floating-point type.
     */
    Value to_value() const;

    /**
     * Identical to `to_value`, except returns an empty `Value` on error
     * instead of throwing an exception.
     */
    Value to_value_or_empty() const;

    /**
     * Internal type of this object.
     */
    TypeInfo type() const;

    KernelArg to_array(size_t nelements) const;
    void assert_type_matches(TypeInfo t) const;
    bool is_scalar() const;
    bool is_array() const;
    std::vector<uint8_t> copy_array() const;
    void* as_void_ptr() const;
    std::vector<uint8_t> to_bytes() const;

    friend std::ostream& operator<<(std::ostream&, const KernelArg&);

    friend std::ostream& operator<<(std::ostream&, const KernelArg&);

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

/**
 * See `into_kernel_arg(T&&)`.
 */
template<typename T, typename Enabled = void>
struct IntoKernelArg {};

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
        return KernelArg::from_scalar<T>(value);
    }
};

template<typename T>
struct IntoKernelArg<
    CudaSpan<T>,
    typename std::enable_if<std::is_trivially_copyable<T>::value>::type> {
    static KernelArg convert(CudaSpan<T> s) {
        return KernelArg::from_array<T>(s.data(), s.size());
    }
};

/**
 * Convert the given `value` into a `KernelArg`. This is done by calling
 * `IntoKernelArg<T>::convert(value)` which, by default, just calls
 * `KernelArg::from_scalar<T>(value)`.
 *
 * It is possible to overload this function by specializing `IntoKernelArg`.
 * For example:
 *
 * ```
 * namespace kernel_launcher {
 * template <>
 * struct IntoKernelArg<mypackage::MyIntegerType> {
 *   static KernelArg convert(mypackage::MyIntegerType m) {
 *     return KernelArg::from_scalar(m.to_int());
 *   }
 * };
 *
 * template <>
 * struct IntoKernelArg<mypackage::MyArrayType> {
 *   static KernelArg convert(mypackage::MyArrayType arr) {
 *     return KernelArg::from_array(arr.ptr(), arr.length());
 *   }
 * };
 * }
 * ```
 */
template<typename T>
KernelArg into_kernel_arg(T&& value) {
    return IntoKernelArg<typename std::decay<T>::type>::convert(
        std::forward<T>(value));
}

}  // namespace kernel_launcher

#endif  //KERNEL_LAUNCHER_ARG_H
