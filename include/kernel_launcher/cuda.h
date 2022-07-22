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
struct CudaException: std::runtime_error {
    CudaException(CUresult err, std::string msg) :
        std::runtime_error(msg),
        err_(err) {}

    CUresult error() const {
        return err_;
    }

  private:
    CUresult err_;
};

void cuda_check(CUresult expr, const char* msg);

struct CudaModule {
    CudaModule(const char* image, const char* fun_name);
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
        return *this;
    }

    void launch(
        CUstream stream,
        dim3 grid_size,
        dim3 block_size,
        uint32_t shared_mem,
        void** args) const;

    CUfunction function() const {
        return fun_ptr_;
    }

    bool valid() const {
        return module_ != nullptr;
    }

  private:
    CUfunction fun_ptr_ = nullptr;
    CUmodule module_ = nullptr;
};

struct CudaDevice {
    CudaDevice() = default;
    explicit CudaDevice(CUdevice d);

    static int count();
    static CudaDevice current();
    std::string name() const;
    int attribute(CUdevice_attribute key) const;
    int ordinal() const;
    std::string uuid() const;

    CUdevice get() const {
        return device_;
    }

    operator CUdevice() const {
        return get();
    }

  private:
    CUdevice device_ = -1;
};

struct CudaContextHandle {
    CudaContextHandle() = default;
    CudaContextHandle(CUcontext c) : context_(c) {};
    static CudaContextHandle current();
    CudaDevice device() const;
    void with(std::function<void()> f) const;

  private:
    CUcontext context_ = nullptr;
};

void cuda_raw_copy(const void* src, void* dst, size_t num_bytes);

template<typename T>
void cuda_copy(const T* src, T* dst, size_t num_elements) {
    static_assert(std::is_trivially_copyable<T>::value, "must be trivial type");
    cuda_raw_copy(
        static_cast<const void*>(src),
        static_cast<void*>(dst),
        num_elements * sizeof(T));
}

}  // namespace kernel_launcher

#endif  //KERNEL_LAUNCHER_CUDA_H
