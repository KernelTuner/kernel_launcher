#ifndef KERNEL_LAUNCHER_CUDA_H
#define KERNEL_LAUNCHER_CUDA_H

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <functional>
#include <stdexcept>
#include <string>

#define KERNEL_LAUNCHER_CUDA_CHECK(expr) \
    ::kernel_launcher::cuda_check(expr, #expr)

namespace kernel_launcher {

/**
 * Exception thrown when a CUDA function returns a non-zero error code.
 */
struct CudaException: std::runtime_error {
    CudaException(CUresult err, std::string msg) :
        std::runtime_error(msg),
        err_(err) {}

    /**
     * The `CUresult` of this error.
     */
    CUresult error() const {
        return err_;
    }

  private:
    CUresult err_;
};

void cuda_check(CUresult result, const char* msg);

/**
 * Wrapper around `CUfunction` and the accompanying `CUmodule`.
 */
struct CudaModule {
    CudaModule(
        const char* image,
        const char* lowered_name,
        const char* human_name = nullptr);
    ~CudaModule();
    CudaModule() = default;
    CudaModule(const CudaModule&) = delete;
    CudaModule& operator=(const CudaModule&) = delete;

    CudaModule(CudaModule&& that) noexcept {
        *this = std::move(that);
    }

    CudaModule& operator=(CudaModule&& that) noexcept {
        std::swap(that.module_, module_);
        std::swap(that.fun_ptr_, fun_ptr_);
        std::swap(that.fun_name_, fun_name_);
        return *this;
    }

    void launch(
        CUstream stream,
        dim3 grid_size,
        dim3 block_size,
        uint32_t shared_mem,
        void** args) const;

    const std::string& function_name() const {
        return fun_name_;
    }

    CUfunction function() const {
        return fun_ptr_;
    }

    bool valid() const {
        return module_ != nullptr;
    }

  private:
    std::string fun_name_;
    CUfunction fun_ptr_ = nullptr;
    CUmodule module_ = nullptr;
};

/**
 * Represent the architecture of a CUDA-capable device. The architecture
 * is represented using two numbers: the _major_ and the _minor_ version.
 * For example, `sm_82` has major version "8" and minor version "2".
 */
struct CudaArch {
    /**
     * Create new `CudaArch` object from `major` and `minor` version
     */
    CudaArch(int major, int minor) : major_(major), minor_(minor) {}

    /**
     * Create new `CudaArch` object from compound version number (such as 82)
     */
    CudaArch(int version) : major_(version / 10), minor_(version % 10) {}
    CudaArch() : CudaArch(0, 0) {}

    /**
     * Returns the compund version (for example, `82` for version `8.2`)
     */
    int get() const {
        return major_ * 10 + minor_;
    }

    /**
     * Return the major version.
     */
    int major() const {
        return major_;
    }

    /**
     * Return the minor version.
     */
    int minor() const {
        return minor_;
    }

    bool operator==(const CudaArch& that) const {
        return major_ == that.major_ && minor_ == that.minor_;
    }

    bool operator!=(const CudaArch& that) const {
        return !(*this == that);
    }

  private:
    int major_;
    int minor_;
};

/**
 * Wrapper around `CUdevice`.
 */
struct CudaDevice {
    CudaDevice() = default;
    explicit CudaDevice(CUdevice d);

    /**
     * Returns the number of CUDA-capable devices in the current context.
     */
    static int count();

    /**
     * Returns the device associated with the current `CudaContextHandle`.
     */
    static CudaDevice current();

    /**
     * Returns the name of the current device
     */
    std::string name() const;

    /**
     * Returns the value of this device for the given attribute.
     */
    int attribute(CUdevice_attribute key) const;

    /**
     * Returns the ordinal number associated with this device in CUDA.
     */
    int ordinal() const;

    /**
     * Returns the device's 128-bit UUID as a string in hexadecimal notation.
     */
    std::string uuid() const;
    CudaArch arch() const;

    /**
     * Returns the underlying `CUcontext`.
     */
    CUdevice get() const {
        return device_;
    }

    operator CUdevice() const {
        return get();
    }

  private:
    CUdevice device_ = -1;
};

/**
 * Wrapper around `CUcontext`.
 */
struct CudaContextHandle {
    CudaContextHandle() = default;
    CudaContextHandle(CUcontext c) : context_(c) {};

    /**
     * Returns the current CUDA context or throws an error if CUDA has not
     * been initialized yet.
     */
    static CudaContextHandle current();

    /**
     * Returns the `CudaDevice` associated with this context.
     */
    CudaDevice device() const;

    void with(std::function<void()> f) const;

    /**
     * Returns the underlying `CUcontext`.
     */
    CUcontext get() const {
        return context_;
    }

  private:
    CUcontext context_ = nullptr;
};

/**
 * Represents a contiguous sequence of objects of type `T` stored in GPU
 * memory.
 */
template<typename T>
struct CudaSpan {
    /**
     * Create new span for `nelements` objects starting at address `ptr`.
     */
    CudaSpan(T* ptr, size_t nelements) : ptr_(ptr), nelem_(nelements) {}

    /**
     * Create empty span.
     */
    CudaSpan() : ptr_(nullptr), nelem_(0) {}

    /**
     * Returns a pointer to the beginning of the sequence.
     */
    T* data() const {
        return ptr_;
    }

    /**
     * Returns the number of elements in the span.
     */
    size_t size() const {
        return nelem_;
    }

    operator T*() const {
        return ptr_;
    }

    operator const T*() const {
        return ptr_;
    }

  private:
    T* ptr_;
    size_t nelem_;
};

template<typename T>
struct CudaSpan<const T> {
    /**
     * Create new span for `nelements` objects starting at address `ptr`.
     */
    CudaSpan(const T* ptr, size_t nelements) : ptr_(ptr), nelem_(nelements) {}

    /**
     * Create new span from the given span.
     */
    CudaSpan(CudaSpan<T> that) : ptr_(that.data()), nelem_(that.size()) {}

    /**
     * Create empty span.
     */
    CudaSpan() : ptr_(nullptr), nelem_(0) {}

    /**
     * Returns a pointer to the beginning of the sequence.
     */
    const T* data() const {
        return ptr_;
    }

    /**
     * Returns the number of elements in the span.
     */
    size_t size() const {
        return nelem_;
    }

    operator const T*() const {
        return ptr_;
    }

  private:
    const T* ptr_;
    size_t nelem_;
};

/**
 * Shorthand for `CudaSpan<T>(ptr, nelements)`
 */
template<typename T>
CudaSpan<T> cuda_span(T* ptr, size_t nelements) {
    return {ptr, nelements};
}

void cuda_raw_copy(const void* src, void* dst, size_t num_bytes);

template<typename T>
void cuda_copy(const T* src, T* dst, size_t num_elements) {
    static_assert(std::is_trivially_copyable<T>::value, "must be trivial type");
    cuda_raw_copy(
        static_cast<const void*>(src),
        static_cast<void*>(dst),
        num_elements * sizeof(T));
}

template<typename T>
void cuda_copy(CudaSpan<T> src, CudaSpan<const T> dst) {
    if (src.size() != dst.size()) {
        throw std::runtime_error("spans have different sizes");
    }

    cuda_copy(src.data(), dst.data(), src.size());
}

}  // namespace kernel_launcher

#endif  //KERNEL_LAUNCHER_CUDA_H
