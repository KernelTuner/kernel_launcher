#ifndef KERNEL_LAUNCHER_KERNEL_ARG_H
#define KERNEL_LAUNCHER_KERNEL_ARG_H

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <type_traits>

#include "kernel_launcher/primitive.hpp"
#include "kernel_launcher/type_info.hpp"
#include "kernel_launcher/utils.hpp"

namespace kernel_launcher {

/**
 * @addtogroup kernel
 * @{
 */

/**
 * A single argument that is passed to a kernel. It stores a copy of the value together with its
 * type. Pointer arguments additionally store the size of the buffer that they point to, which is
 * needed to capture the contents of the buffer.
 */
class KernelArg {
  public:
    /**
     * Constructs an argument that holds a copy of `value`. The type `T` must be trivially
     * copyable. If `T` is a pointer type, the size of its buffer is zero (see `buffer_size`).
     */
    template<typename T>
    static KernelArg of(const T& value) {
        static_assert(
            std::is_trivially_copyable<T>::value,
            "KernelArg only supports trivially copyable types");
        static_assert(
            !std::is_array<T>::value,
            "KernelArg::of: arrays are passed by value, pass a pointer instead");

        return unsafe_construct(TypeInfo::of<T>(), &value, sizeof(T));
    }

    /**
     * Constructs a pointer argument that points to a buffer of `nbytes` bytes. Note that the
     * size is in bytes, not in number of elements.
     */
    template<typename T>
    static KernelArg of(T* ptr, size_t nbytes) {
        static_assert(sizeof(T*) <= INLINE_CAPACITY, "argument too large for KernelArg");

        KernelArg arg(TypeInfo::of<T*>(), nbytes);
        std::memcpy(arg.storage_.inline_data, &ptr, sizeof(T*));
        return arg;
    }

    /**
     * Constructs an argument of the given type from `nbytes` bytes at `data`. The `nbytes` must
     * equal the size of `type`. Combined with `TypeInfo::opaque`, this can pass types that only
     * exist in the kernel source.
     */
    static KernelArg unsafe_construct(TypeInfo type, const void* data, size_t nbytes);

    KernelArg(const KernelArg& that);
    KernelArg(KernelArg&& that) noexcept;
    KernelArg& operator=(const KernelArg& that);
    KernelArg& operator=(KernelArg&& that) noexcept;
    ~KernelArg();

    /**
     * Returns the type of this argument.
     */
    TypeInfo type() const {
        return type_;
    }

    /**
     * Returns a pointer to the stored value, which is `type().size_in_bytes()` bytes long.
     */
    const void* data() const {
        return is_heap() ? storage_.heap_data : storage_.inline_data;
    }

    /**
     * Returns `true` if this argument has type `T`, ignoring references and top-level `const`.
     */
    template<typename T>
    bool has_type() const {
        using U = typename std::decay<T>::type;
        return type_ == TypeInfo::of<U>();
    }

    /**
     * Returns the stored value as a `T`. Throws an exception if this argument does not have
     * type `T` (see `has_type`).
     */
    template<typename T>
    T get() const {
        using U = typename std::decay<T>::type;

        static_assert(
            !std::is_reference<T>::value,
            "KernelArg::get: T must not be a reference type");
        static_assert(
            std::is_trivially_copyable<U>::value,
            "KernelArg only supports trivially copyable types");

        if (type_ != TypeInfo::of<U>()) {
            throw std::runtime_error(concat(
                "KernelArg::get: argument has type '",
                type_.name(),
                "', but type '",
                TypeInfo::of<U>().name(),
                "' was requested"));
        }

        U value;
        std::memcpy(&value, data(), sizeof(U));
        return value;
    }

    /**
     * Returns the size in bytes of the buffer that this argument points to. This is zero if the
     * size is unknown or if this argument is not a pointer.
     */
    size_t buffer_size() const {
        return nbytes_;
    }

    /**
     * Returns a copy of this argument where the buffer size is set to `nbytes` bytes. Throws an
     * exception if this argument is not a pointer.
     */
    KernelArg with_buffer_size(size_t nbytes) const;

    /**
     * Converts the stored value to a `Primitive`. Returns an empty `Primitive` if the type of
     * this argument is not primitive (see `TypeInfo::is_primitive`), which includes pointers.
     */
    Primitive to_primitive_or_empty() const;

    /**
     * Converts the stored value to a `Primitive`. Throws an exception if the type of this
     * argument is not primitive (see `TypeInfo::is_primitive`), which includes pointers.
     */
    Primitive to_primitive() const;

  private:
    static constexpr size_t INLINE_CAPACITY = 16;

    KernelArg(TypeInfo type, size_t nbytes) : type_(type), nbytes_(nbytes) {
        // The destructor may free `heap_data`, so it must never be left uninitialized.
        storage_.heap_data = nullptr;
    }

    // The size of `type_` determines which member of `storage_` is active.
    bool is_heap() const {
        return type_.size_in_bytes() > INLINE_CAPACITY;
    }

    TypeInfo type_;
    size_t nbytes_;
    // Values that do not fit in `inline_data` are stored in `heap_data` instead, which is owned.
    union {
        alignas(std::max_align_t) std::byte inline_data[INLINE_CAPACITY];
        std::byte* heap_data;
    } storage_;
};

/**
 * Determines how a value of type `T` is converted into a `KernelArg` when it is passed to a
 * kernel. By default, the value is copied using `KernelArg::of`. Specialize this struct to pass
 * your own types to a kernel, for example, a buffer type that should be passed as a pointer:
 *
 * ```
 * template<typename T>
 * struct IntoKernelArg<my_buffer<T>> {
 *     static KernelArg call(const my_buffer<T>& buffer) {
 *         return KernelArg::of(buffer.device_ptr(), buffer.size() * sizeof(T));
 *     }
 * };
 * ```
 *
 * The second template parameter is always `void` and can be used to enable a specialization
 * conditionally using `std::enable_if`.
 */
template<typename T, typename = void>
struct IntoKernelArg {
    static KernelArg call(const T& value) {
        return KernelArg::of(value);
    }
};

/** @} */

/**
 * A pointer to `size` elements of type `T` in GPU memory. Passing this as a kernel argument
 * passes the pointer and records the size of the buffer it points to.
 *
 * @ingroup gpu
 */
template<typename T>
class gpu_span {
  public:
    gpu_span(T* ptr, size_t size) : ptr_(ptr), size_(size) {}

    /**
     * Converts a `gpu_span<T>` into a `gpu_span<const T>`.
     */
    gpu_span(const gpu_span<typename std::remove_const<T>::type>& that) :
        ptr_(that.data()),
        size_(that.size()) {}

    /**
     * Returns the pointer to the first element.
     */
    T* data() const {
        return ptr_;
    }

    /**
     * Returns the number of elements, not the number of bytes.
     */
    size_t size() const {
        return size_;
    }

  private:
    T* ptr_;
    size_t size_;
};

/**
 * @addtogroup kernel
 * @{
 */

template<typename T>
struct IntoKernelArg<gpu_span<T>> {
    static KernelArg call(const gpu_span<T>& span) {
        return KernelArg::of(span.data(), span.size() * sizeof(T));
    }
};

/**
 * Converts `value` into a `KernelArg` using `IntoKernelArg`. This is how the arguments of a
 * kernel launch are converted.
 */
template<typename T>
KernelArg into_kernel_arg(T&& value) {
    return IntoKernelArg<typename std::decay<T>::type>::call(value);
}

/** @} */

}  // namespace kernel_launcher

#endif
