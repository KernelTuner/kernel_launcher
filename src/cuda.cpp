#include "kernel_launcher/cuda.h"

#include <iomanip>
#include <iostream>
#include <sstream>

#include "kernel_launcher/utils.h"

namespace kernel_launcher {

void cuda_check(CUresult result, const char* msg) {
    if (result != CUDA_SUCCESS) {
        const char* name = "???";
        const char* description = "???";

        // Ignore error since we are already handling another error
        cuGetErrorName(result, &name);
        cuGetErrorString(result, &description);

        std::stringstream display;
        display << "CUDA error: " << name << " (" << description
                << "): " << msg;
        throw CudaException(result, display.str());
    }
}

CudaModule::CudaModule(
    const char* image,
    const char* lowered_name,
    const char* human_name) {
    if (human_name != nullptr) {
        fun_name_ = human_name;
    }

    KERNEL_LAUNCHER_CUDA_CHECK(
        cuModuleLoadDataEx(&module_, image, 0, nullptr, nullptr));
    KERNEL_LAUNCHER_CUDA_CHECK(
        cuModuleGetFunction(&fun_ptr_, module_, lowered_name));
}

CudaModule::~CudaModule() {
    if (valid()) {
        CUresult result = cuModuleUnload(module_);
        if (result != CUDA_SUCCESS) {
            log_warning() << "error during cuModuleUnload has been ignored: "
                          << result << std::endl;
        }

        module_ = nullptr;
        fun_ptr_ = nullptr;
    }
}

void CudaModule::launch(
    CUstream stream,
    dim3 grid_size,
    dim3 block_size,
    uint32_t shared_mem,
    void** args) const {
    if (!valid()) {
        throw std::runtime_error("CudaModule has not been initialized");
    }

    KERNEL_LAUNCHER_CUDA_CHECK(cuLaunchKernel(
        fun_ptr_,
        grid_size.x,
        grid_size.y,
        grid_size.z,
        block_size.x,
        block_size.y,
        block_size.z,
        shared_mem,
        stream,
        args,
        nullptr));
}

CudaDevice::CudaDevice(CUdevice d) : device_(d) {
    // Just try some operation on this device to check if it is valid.
    size_t tmp;
    KERNEL_LAUNCHER_CUDA_CHECK(cuDeviceTotalMem(&tmp, d));
}

int CudaDevice::count() {
    int count = 0;
    KERNEL_LAUNCHER_CUDA_CHECK(cuDeviceGetCount(&count));
    return count;
}

CudaDevice CudaDevice::current() {
    CUdevice dev = 0;
    KERNEL_LAUNCHER_CUDA_CHECK(cuCtxGetDevice(&dev));
    return CudaDevice(dev);
}

std::string CudaDevice::name() const {
    char name[64] = {0};
    KERNEL_LAUNCHER_CUDA_CHECK(cuDeviceGetName(name, sizeof name, device_));
    return name;
}

int CudaDevice::attribute(CUdevice_attribute key) const {
    int value;
    KERNEL_LAUNCHER_CUDA_CHECK(cuDeviceGetAttribute(&value, key, device_));
    return value;
}

int CudaDevice::ordinal() const {
    return device_;
}

std::string CudaDevice::uuid() const {
    CUuuid uuid;
    KERNEL_LAUNCHER_CUDA_CHECK(cuDeviceGetUuid(&uuid, device_));

    std::stringstream result;
    result << std::hex << std::setfill('0') << std::setw(2);
    for (size_t i = 0; i < 16; i++) {
        if (i % 2 == 0 && i >= 4 && i <= 10) {
            result << "-";
        }

        result << (unsigned int)(unsigned char)uuid.bytes[i];
    }

    return result.str();
}

CudaArch CudaDevice::arch() const {
    int minor = attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
    int major = attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
    return CudaArch(major, minor);
}

CudaContextHandle CudaContextHandle::current() {
    CUcontext ctx = nullptr;
    KERNEL_LAUNCHER_CUDA_CHECK(cuCtxGetCurrent(&ctx));

    // cuCtxGetCurrent will just return null if there is no current context, we consider it an error.
    if (ctx == nullptr) {
        throw std::runtime_error(
            "CUDA context not initialized for current thread");
    }

    return {ctx};
}

CudaDevice CudaContextHandle::device() const {
    CUdevice d = -1;
    with([&]() { KERNEL_LAUNCHER_CUDA_CHECK(cuCtxGetDevice(&d)); });
    return CudaDevice(d);
}

void CudaContextHandle::with(std::function<void()> f) const {
    KERNEL_LAUNCHER_CUDA_CHECK(cuCtxPushCurrent(context_));
    try {
        f();
        KERNEL_LAUNCHER_CUDA_CHECK(cuCtxPopCurrent(nullptr));
    } catch (...) {
        // Ignore errors. There is not much we can do at this point.
        cuCtxPopCurrent(nullptr);
        throw;
    }
}

void cuda_raw_copy(const void* src, void* dst, size_t num_bytes) {
    KERNEL_LAUNCHER_CUDA_CHECK(cuMemcpy(
        reinterpret_cast<CUdeviceptr>(dst),
        reinterpret_cast<CUdeviceptr>(src),
        num_bytes));
}

}  // namespace kernel_launcher