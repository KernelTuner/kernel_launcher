#ifndef KERNEL_LAUNCHER_TEST_UTILS_H
#define KERNEL_LAUNCHER_TEST_UTILS_H

#include "kernel_launcher/kernel.h"

#include <string>

inline std::string assets_directory() {
    std::string assets_dir = __FILE__;
    assets_dir = assets_dir.substr(0, assets_dir.rfind('/'));
    assets_dir += "/assets";
    return assets_dir;
}

inline kernel_launcher::KernelBuilder build_vector_add_kernel() {
    using namespace kernel_launcher;

    static constexpr const char* kernel_source = R"(
    template <typename T>
    __global__
    void vector_add(int n, int *c, const int* a, const int* b) {
        for (int k = 0; k < ELEMENTS_PER_THREAD; k++) {
            int index = (blockIdx.x * ELEMENTS_PER_THREAD + k) * blockDim.x + threadIdx.x;

            if (index < n) {
                c[index] = a[index] + b[index];
            }
        }
    }
    )";

    KernelBuilder builder(
        "vector_add",
        KernelSource("vector_add.cu", kernel_source));
    auto tb = builder.tune("threads_per_block", {1, 32, 128, 256});
    auto et = builder.tune("elements_per_thread", {1, 2, 4});
    auto eb = et * tb;

    builder.define("ELEMENTS_PER_THREAD", et)
        .template_args(type_of<int>())
        .block_size(tb)
        .grid_divisors(eb);
    return builder;
}

template <typename T>
struct CudaVector {
    CudaVector(size_t n = 0) {
        KERNEL_LAUNCHER_CUDA_CHECK(cuMemAlloc(&ptr_, n * sizeof(T)));
    }

    ~CudaVector() {
        if (ptr_) {
            KERNEL_LAUNCHER_CUDA_CHECK(cuMemFree(ptr_));
            ptr_ = 0;
        }
    }

    T* get() {
        return (T*) ptr_;
    }

    operator T*() {
        return get();
    }

  private:
    CUdeviceptr ptr_ = 0;
};

#endif  //KERNEL_LAUNCHER_TEST_UTILS_H
