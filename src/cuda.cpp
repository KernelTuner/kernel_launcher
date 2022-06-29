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

CudaModule::CudaModule(const char* image, const char* fun_name) {
    KERNEL_LAUNCHER_CUDA_CHECK(
        cuModuleLoadDataEx(&module_, image, 0, nullptr, nullptr));
    KERNEL_LAUNCHER_CUDA_CHECK(
        cuModuleGetFunction(&fun_ptr_, module_, fun_name));
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
        if (i % 2 == 0 && i >= 4 && i <= 10)
            result << "-";
        result << (unsigned int)(unsigned char)uuid.bytes[i];
    }

    return result.str();
}
}  // namespace kernel_launcher