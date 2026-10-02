#ifndef KERNEL_LAUNCHER_TEST_HELPERS_H
#define KERNEL_LAUNCHER_TEST_HELPERS_H

#include <cuda.h>
#include <stdlib.h>

#include <stdexcept>
#include <string>
#include <unordered_map>

#include "catch.hpp"
#include "kernel_launcher/file_loader.hpp"
#include "kernel_launcher/gpu_utils.hpp"

namespace kernel_launcher {
namespace test {

// A `FileLoader` that serves files from memory.
struct MapLoader: FileLoader {
    std::optional<std::string> try_load(std::string_view path) const override {
        auto it = files.find(std::string(path));

        if (it == files.end()) {
            return std::nullopt;
        }

        return it->second;
    }

    std::unordered_map<std::string, std::string> files;
};

// Makes the primary context of the first GPU current on this thread. Returns
// false if no GPU is available.
inline bool init_gpu_context() {
    CUcontext context = nullptr;
    CUdevice device;

    try {
        if (GPUDevice::count() == 0) {
            return false;
        }
    } catch (const GPUException&) {
        return false;
    }

    if (cuDeviceGet(&device, 0) != CUDA_SUCCESS
        || cuDevicePrimaryCtxRetain(&context, device) != CUDA_SUCCESS) {
        return false;
    }

    return context != nullptr && cuCtxSetCurrent(context) == CUDA_SUCCESS;
}

// A fixed architecture, for compiling kernels without a GPU.
inline GPUArch fixed_arch() {
#if defined(__HIP_PLATFORM_AMD__)
    return GPUArch("gfx90a");
#else
    return GPUArch("sm_80");
#endif
}

// Creates a new, empty directory and returns its path.
inline std::string make_temp_dir() {
    std::string path = "/tmp/kernel_launcher_test_XXXXXX";

    if (mkdtemp(&path[0]) == nullptr) {
        throw std::runtime_error("failed to create temporary directory");
    }

    return path;
}

}  // namespace test
}  // namespace kernel_launcher

// Ends the current test case if no GPU is available.
#define REQUIRE_GPU_OR_SKIP()                                     \
    do {                                                          \
        if (!::kernel_launcher::test::init_gpu_context()) {       \
            WARN("no GPU available, skipping remainder of test"); \
            return;                                               \
        }                                                         \
    } while (0)

#endif
