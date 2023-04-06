#ifndef KERNEL_LAUNCHER_TEST_UTILS_H
#define KERNEL_LAUNCHER_TEST_UTILS_H

#include <string>

#include "kernel_launcher/kernel.h"

struct point3 {
    unsigned long long x;
    unsigned long long y;
    unsigned long long z;

    bool operator==(point3 v) const {
        return x == v.x && y == v.y && z == v.z;
    }
};

namespace kernel_launcher {
template<>
struct IntoKernelArg<point3> {
    static KernelArg convert(point3 value) {
        return KernelArg::from_scalar<point3>(value);
    }
};
}  // namespace kernel_launcher

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
    auto tb = builder.tune("threads_per_block", {1, 32, 128, 256}, 256);
    auto et = builder.tune("elements_per_thread", {1, 2, 4});
    auto eb = et * tb;


    builder.restriction(eb >= 32 && eb <= 1024);

    builder
        .problem_size(arg0)
        .define("ELEMENTS_PER_THREAD", et)
        .template_type<int>()
        .block_size(tb)
        .grid_size(div_ceil(arg0, eb));
    return builder;
}


inline kernel_launcher::KernelBuilder build_matmul_kernel() {
    using namespace kernel_launcher;

    std::string path = kernel_launcher::path_join(assets_directory(), "matmul_kernel.cu");
    KernelBuilder builder("matmul_kernel", path);
    auto bx = builder.tune("block_size_x", {16, 32, 64});
    auto by = builder.tune("block_size_y", {1, 2, 4, 8, 16, 32}, 16);
    auto tx = builder.tune("tile_size_x", {1, 2, 4, 8});
    auto ty = builder.tune("tile_size_y", {1, 2, 4, 8});

    auto smem_floats = by * ty * bx * (1 + tx);
    builder.restriction(smem_floats * sizeof(float) <=  48 * 1024);
    builder.restriction(bx == by * ty);

    builder
        .template_args(bx, by, tx, ty)
        .block_size(bx, by)
        .grid_divisors(bx * tx, by * ty);

    return builder;
}

template <typename T>
struct CudaVector {
    CudaVector(size_t n = 0) {
        KERNEL_LAUNCHER_CUDA_CHECK(cuMemAlloc(&ptr_, n * sizeof(T)));
    }

    CudaVector(const std::vector<T> &values): CudaVector(values.size()) {
        kernel_launcher::cuda_copy(values.data(), get(), values.size());
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
